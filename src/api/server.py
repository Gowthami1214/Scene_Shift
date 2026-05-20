"""
FastAPI Backend Server for SceneShift.

Endpoints:
  POST /api/upload           — Secure image upload
  POST /api/process          — Trigger async pipeline
  GET  /api/result/{job_id}  — Retrieve final result image
  GET  /api/status/{job_id}  — Job status polling
  GET  /api/device           — Device info
  GET  /api/health           — Health check
  WS   /ws/{job_id}          — Live WebSocket progress stream

Architecture:
  - Job state stored in an in-memory dict (extensible to Redis)
  - Pipeline runs in asyncio thread pool (non-blocking)
  - WebSocket broadcasts ProgressEvent updates in real-time
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from src.pipeline.orchestrator import (
    AsyncPipelineOrchestrator,
    PipelineRequest,
    ProgressEvent,
)
from src.utils.device import get_device_info
from src.utils.image_io import pil_to_numpy, save_image, temp_output_path
from src.utils.validators import (
    ValidationError,
    validate_blend_mode,
    validate_image_bytes,
    validate_prompt,
    validate_strength,
    validate_style_preset,
)


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="SceneShift API",
        description=(
            "Real-Time Semantic Object Editing & Intelligent Scene Transformation. "
            "AI-powered pipeline: segmentation → inpainting → background → composite → shadow → harmonize."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()

# ── Shared state ───────────────────────────────────────────────────────────────
# Job registry: {job_id: {"status": ..., "progress": ..., "result_path": ...}}
_jobs: Dict[str, Dict[str, Any]] = {}

# WebSocket connection registry: {job_id: List[WebSocket]}
_ws_connections: Dict[str, List[WebSocket]] = {}

# Pipeline orchestrator (lazily initialized on first request)
_orchestrator: Optional[AsyncPipelineOrchestrator] = None


def get_orchestrator() -> AsyncPipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initializing pipeline orchestrator…")
        _orchestrator = AsyncPipelineOrchestrator()
    return _orchestrator


# ── Pydantic models ────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    job_id: str
    object_prompt: str = Field(default="object", max_length=500)
    background_prompt: str = Field(default="natural landscape background", max_length=500)
    style_preset: str = Field(default="Realistic")
    blend_mode: str = Field(default="alpha")
    segmentation_mode: str = Field(default="auto")
    click_x: Optional[float] = None
    click_y: Optional[float] = None
    strength: float = Field(default=0.85, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_steps: int = Field(default=30, ge=5, le=150)
    shadow_opacity: float = Field(default=0.5, ge=0.0, le=1.0)
    shadow_blur: int = Field(default=25, ge=0, le=100)
    shadow_dir_x: float = Field(default=1.0)
    shadow_dir_y: float = Field(default=0.5)
    enable_shadow: bool = True
    enable_harmonization: bool = True
    use_sd_background: bool = True
    seed: Optional[int] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    stage: str = "idle"
    message: str = ""
    elapsed_s: float = 0.0
    result_path: Optional[str] = None
    error: Optional[str] = None


# ── Health & info endpoints ────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check():
    """Return server health status."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/device", tags=["System"])
async def device_info():
    """Return compute device information."""
    return get_device_info()


# ── Upload endpoint ────────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["Pipeline"])
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload an image and receive a job_id for subsequent processing.

    Validates:
      - File size ≤ 50 MB
      - Valid image extension (.jpg, .png, .webp, .bmp, .tiff)
      - Decodable image content
      - Dimensions [64, 4096]
    """
    try:
        data = await file.read()
        w, h = validate_image_bytes(data, file.filename or "upload.png")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job_id = str(uuid.uuid4())[:8]

    # Save the uploaded image
    upload_path = temp_output_path(job_id, ".png", "upload")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(str(upload_path), format="PNG")

    _jobs[job_id] = {
        "status": "uploaded",
        "progress": 0.0,
        "stage": "idle",
        "message": "Image uploaded successfully.",
        "upload_path": str(upload_path),
        "result_path": None,
        "error": None,
        "created_at": time.time(),
        "elapsed_s": 0.0,
        "image_width": w,
        "image_height": h,
    }

    logger.info(f"[Job {job_id}] Image uploaded: {file.filename} ({w}x{h})")
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": "uploaded",
            "image_width": w,
            "image_height": h,
        }
    )


# ── Process endpoint ───────────────────────────────────────────────────────────

async def _run_pipeline(job_id: str, req: ProcessRequest) -> None:
    """Background task: runs the full pipeline and updates job state."""
    job = _jobs.get(job_id)
    if not job:
        return

    t_start = time.perf_counter()

    async def on_progress(event: ProgressEvent) -> None:
        """Update job state and broadcast to all WebSocket subscribers."""
        _jobs[job_id].update({
            "status": "processing",
            "progress": event.progress,
            "stage": event.stage,
            "message": event.message,
            "elapsed_s": event.elapsed_s,
        })
        await _broadcast_progress(job_id, event)

    try:
        _jobs[job_id]["status"] = "processing"

        # Load uploaded image
        upload_path = job["upload_path"]
        img_pil = Image.open(upload_path).convert("RGB")
        image_np = pil_to_numpy(img_pil)

        # Build click points if interactive
        click_points = []
        if req.segmentation_mode == "interactive" and \
                req.click_x is not None and req.click_y is not None:
            img_w, img_h = img_pil.size
            # Normalize or pixel coords
            cx = int(req.click_x * img_w) if req.click_x <= 1.0 else int(req.click_x)
            cy = int(req.click_y * img_h) if req.click_y <= 1.0 else int(req.click_y)
            click_points = [(cx, cy)]

        # Validate parameters
        try:
            style = validate_style_preset(req.style_preset)
            blend = validate_blend_mode(req.blend_mode)
            obj_prompt = validate_prompt(req.object_prompt, "object_prompt")
            bg_prompt = validate_prompt(req.background_prompt, "background_prompt")
        except ValidationError as exc:
            raise ValueError(str(exc))

        pipeline_req = PipelineRequest(
            image=image_np,
            object_prompt=obj_prompt,
            background_prompt=bg_prompt,
            style_preset=style,
            blend_mode=blend,
            segmentation_mode=req.segmentation_mode,
            click_points=click_points,
            strength=req.strength,
            guidance_scale=req.guidance_scale,
            num_steps=req.num_steps,
            shadow_direction=(req.shadow_dir_x, req.shadow_dir_y),
            shadow_opacity=req.shadow_opacity,
            shadow_blur=req.shadow_blur,
            enable_shadow=req.enable_shadow,
            enable_harmonization=req.enable_harmonization,
            use_sd_background=req.use_sd_background,
            seed=req.seed,
            output_size=(512, 512),
        )

        orchestrator = get_orchestrator()
        result = await orchestrator.run_async(
            pipeline_req,
            job_id=job_id,
            progress_callback=on_progress,
        )

        # Update final state
        _jobs[job_id].update({
            "status": "done",
            "progress": 1.0,
            "stage": "done",
            "message": f"Completed in {result.total_time_s:.1f}s",
            "result_path": result.output_path,
            "elapsed_s": time.perf_counter() - t_start,
            "timings": result.timings,
        })

        # Final WebSocket notification
        await _broadcast_progress(
            job_id,
            ProgressEvent("done", 1.0, "Pipeline complete!", time.perf_counter() - t_start),
        )

    except Exception as exc:
        logger.error(f"[Job {job_id}] Pipeline error: {exc}", exc_info=True)
        _jobs[job_id].update({
            "status": "error",
            "progress": 0.0,
            "stage": "error",
            "message": str(exc),
            "error": str(exc),
            "elapsed_s": time.perf_counter() - t_start,
        })
        await _broadcast_progress(
            job_id,
            ProgressEvent("error", 0.0, f"Error: {exc}", time.perf_counter() - t_start),
        )


@app.post("/api/process", tags=["Pipeline"])
async def process_image(
    req: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Trigger the full AI pipeline for an uploaded image.
    Processing runs asynchronously; monitor progress via WebSocket /ws/{job_id}.
    """
    job_id = req.job_id
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if _jobs[job_id]["status"] == "processing":
        raise HTTPException(status_code=409, detail="Job is already processing.")

    # Add pipeline run as a background task
    background_tasks.add_task(_run_pipeline, job_id, req)

    return JSONResponse(content={"job_id": job_id, "status": "queued"})


# ── Status endpoint ────────────────────────────────────────────────────────────

@app.get("/api/status/{job_id}", response_model=JobStatus, tags=["Pipeline"])
async def get_job_status(job_id: str) -> JobStatus:
    """Poll job processing status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0.0),
        stage=job.get("stage", "idle"),
        message=job.get("message", ""),
        elapsed_s=job.get("elapsed_s", 0.0),
        result_path=job.get("result_path"),
        error=job.get("error"),
    )


# ── Result endpoint ────────────────────────────────────────────────────────────

@app.get("/api/result/{job_id}", tags=["Pipeline"])
async def get_result(job_id: str, format: str = "png") -> FileResponse:
    """
    Return the final composited image as a downloadable file.

    Args:
        job_id: Job identifier from /api/upload.
        format: Output format: "png" or "jpeg".
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.get("status") != "done":
        raise HTTPException(
            status_code=202,
            detail=f"Job not yet complete. Status: {job.get('status')}",
        )

    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found.")

    media_type = "image/jpeg" if format.lower() == "jpeg" else "image/png"
    return FileResponse(
        result_path,
        media_type=media_type,
        filename=f"sceneshift_{job_id}.{format}",
    )


# ── WebSocket progress stream ──────────────────────────────────────────────────

async def _broadcast_progress(job_id: str, event: ProgressEvent) -> None:
    """Send a progress event to all WebSocket subscribers for a job."""
    connections = _ws_connections.get(job_id, [])
    if not connections:
        return

    msg = json.dumps({
        "stage": event.stage,
        "progress": round(event.progress, 3),
        "message": event.message,
        "elapsed_s": round(event.elapsed_s, 2),
    })

    dead = []
    for ws in connections:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)

    # Clean up disconnected clients
    for ws in dead:
        connections.remove(ws)


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint for real-time pipeline progress updates.

    Clients connect here and receive JSON messages:
      {"stage": "...", "progress": 0.0-1.0, "message": "...", "elapsed_s": 0.0}
    """
    await websocket.accept()

    if job_id not in _ws_connections:
        _ws_connections[job_id] = []
    _ws_connections[job_id].append(websocket)

    logger.info(f"WebSocket connected: job={job_id}")

    try:
        # Send current status immediately on connect
        job = _jobs.get(job_id, {})
        await websocket.send_text(json.dumps({
            "stage": job.get("stage", "idle"),
            "progress": job.get("progress", 0.0),
            "message": job.get("message", "Connected."),
            "elapsed_s": job.get("elapsed_s", 0.0),
        }))

        # Keep connection alive until client disconnects
        while True:
            try:
                # Ping keepalive
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"heartbeat"}')
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: job={job_id}")
    finally:
        conns = _ws_connections.get(job_id, [])
        if websocket in conns:
            conns.remove(websocket)


# ── Server entrypoint ──────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI server with Uvicorn."""
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
