"""
Pipeline Orchestrator for SceneShift.

Coordinates all AI pipeline stages in sequence, manages timing,
broadcasts WebSocket progress events, and handles GPU memory lifecycle.

Stage execution order:
  1. Segmentation     (YOLO auto / SAM2 interactive)
  2. Mask Refinement  (morpho + edge + feathering)
  3. Object Editing   (Stable Diffusion inpainting)
  4. Background Gen   (Stable Diffusion / procedural)
  5. Compositing      (alpha / Poisson / Laplacian)
  6. Shadow Synthesis (directional Gaussian shadow)
  7. Color Harmony    (LAB luminance + chroma)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.models.background import BackgroundGenerator
from src.models.compositing import CompositingEngine
from src.models.editing import ObjectEditor
from src.models.segmentation import SegmentationEngine, SegmentationResult
from src.pipeline.harmonization import ColorHarmonizer
from src.pipeline.mask_utils import refine_mask
from src.pipeline.shadow import ShadowSynthesizer
from src.utils.device import get_device, clear_gpu_cache
from src.utils.image_io import resize_for_pipeline, save_image, temp_output_path


# ── Stage Enum ────────────────────────────────────────────────────────────────

class PipelineStage(str, Enum):
    IDLE = "idle"
    SEGMENTING = "segmenting"
    REFINING_MASK = "refining_mask"
    EDITING_OBJECT = "editing_object"
    GENERATING_BACKGROUND = "generating_background"
    COMPOSITING = "compositing"
    ADDING_SHADOW = "adding_shadow"
    HARMONIZING = "harmonizing"
    DONE = "done"
    ERROR = "error"


# ── Progress event ────────────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    stage: str
    progress: float          # 0.0 → 1.0
    message: str
    elapsed_s: float = 0.0


# ── Pipeline request/result ───────────────────────────────────────────────────

@dataclass
class PipelineRequest:
    image: np.ndarray                             # Input image (RGB uint8)
    object_prompt: str = "object"
    background_prompt: str = "natural background"
    style_preset: str = "Realistic"
    blend_mode: str = "alpha"
    segmentation_mode: str = "auto"              # "auto" | "interactive"
    click_points: List[Tuple[int, int]] = field(default_factory=list)
    strength: float = 0.85
    guidance_scale: float = 7.5
    num_steps: int = 30
    shadow_direction: Tuple[float, float] = (1.0, 0.5)
    shadow_opacity: float = 0.5
    shadow_blur: int = 25
    enable_shadow: bool = True
    enable_harmonization: bool = True
    seed: Optional[int] = None
    use_sd_background: bool = True
    output_size: Tuple[int, int] = (512, 512)


@dataclass
class PipelineResult:
    job_id: str
    final_image: np.ndarray                      # Final composited image
    intermediate: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    total_time_s: float = 0.0
    output_path: Optional[str] = None


# ── Orchestrator ──────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    Manages the full SceneShift processing pipeline.

    All heavy models are lazily loaded and cached in GPU memory.
    Progress is broadcast via an optional async callback.
    """

    def __init__(self, device=None):
        self.device = device or get_device()

        # Model instances (lazy-loaded)
        self._segmentation = SegmentationEngine(device=self.device)
        self._editor = ObjectEditor(device=self.device)
        self._bg_gen = BackgroundGenerator(device=self.device)
        self._compositor = CompositingEngine()
        self._shadow = ShadowSynthesizer()
        self._harmonizer = ColorHarmonizer()

    def preload_all(self) -> None:
        """Eagerly load all models into GPU memory (optional startup warmup)."""
        logger.info("Preloading all SceneShift models…")
        self._segmentation.preload_models()
        logger.info("All models preloaded.")

    def _progress(
        self,
        callback: Optional[Callable],
        stage: str,
        pct: float,
        msg: str,
        t0: float,
    ) -> None:
        """Fire progress callback if provided."""
        if callback is None:
            return
        event = ProgressEvent(
            stage=stage,
            progress=pct,
            message=msg,
            elapsed_s=time.perf_counter() - t0,
        )
        # Support both sync and async callbacks
        if asyncio.iscoroutinefunction(callback):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(callback(event))
                else:
                    loop.run_until_complete(callback(event))
            except Exception as exc:
                logger.warning(f"Progress callback error: {exc}")
        else:
            try:
                callback(event)
            except Exception as exc:
                logger.warning(f"Progress callback error: {exc}")

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def run(
        self,
        request: PipelineRequest,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Execute the full SceneShift pipeline synchronously.

        Args:
            request: PipelineRequest with all parameters.
            job_id: Unique job identifier (auto-generated if None).
            progress_callback: Callable(ProgressEvent) for live updates.

        Returns:
            PipelineResult with the final composited image.
        """
        job_id = job_id or str(uuid.uuid4())[:8]
        t_total = time.perf_counter()
        timings: Dict[str, float] = {}
        intermediate: Dict[str, Any] = {}

        logger.info(f"[Job {job_id}] Pipeline starting | "
                    f"mode={request.segmentation_mode} | "
                    f"style={request.style_preset}")

        image = request.image
        orig_h, orig_w = image.shape[:2]

        # Resize for pipeline if too large (preserves aspect)
        image, orig_size = resize_for_pipeline(image, max_dim=max(*request.output_size))
        h, w = image.shape[:2]

        try:
            # ── Stage 1: Segmentation ──────────────────────────────────────────
            self._progress(progress_callback, "segmenting", 0.05,
                           "Segmenting object…", t_total)
            t0 = time.perf_counter()

            seg_result: Optional[SegmentationResult] = None
            if request.segmentation_mode == "interactive" and request.click_points:
                seg_result = self._segmentation.interactive_segment(
                    image, request.click_points
                )
            else:
                seg_result = self._segmentation.auto_segment(image)

            if seg_result is None:
                logger.warning(f"[Job {job_id}] No object detected; using full image.")
                raw_mask = np.ones((h, w), dtype=np.uint8) * 255
                label = "unknown"
            else:
                raw_mask = seg_result.mask
                label = seg_result.label

            timings["segmentation"] = time.perf_counter() - t0
            intermediate["raw_mask"] = raw_mask
            intermediate["label"] = label
            self._progress(progress_callback, "segmenting", 0.15,
                           f"Object '{label}' detected.", t_total)

            # ── Stage 2: Mask Refinement ───────────────────────────────────────
            self._progress(progress_callback, "refining_mask", 0.18,
                           "Refining mask boundaries…", t_total)
            t0 = time.perf_counter()

            refined = refine_mask(raw_mask, image_rgb=image, feather_radius=4)
            timings["mask_refinement"] = time.perf_counter() - t0
            intermediate["alpha_mask"] = refined.alpha_mask
            intermediate["binary_mask"] = refined.binary_mask

            self._progress(progress_callback, "refining_mask", 0.25,
                           "Mask refined.", t_total)

            # ── Stage 3: Object Editing (SD Inpainting) ────────────────────────
            self._progress(progress_callback, "editing_object", 0.28,
                           f"Editing object with '{request.style_preset}' style…",
                           t_total)
            t0 = time.perf_counter()

            edit_result = self._editor.edit(
                image_rgb=image,
                mask=refined.binary_mask,
                object_prompt=request.object_prompt,
                style_preset=request.style_preset,
                strength=request.strength,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_steps,
                seed=request.seed,
                target_size=request.output_size,
            )
            timings["editing"] = time.perf_counter() - t0
            intermediate["edited_image"] = edit_result.edited_image
            self._progress(progress_callback, "editing_object", 0.50,
                           "Object editing complete.", t_total)

            # ── Stage 4: Background Generation ────────────────────────────────
            self._progress(progress_callback, "generating_background", 0.52,
                           "Generating background scene…", t_total)
            t0 = time.perf_counter()

            bg_result = self._bg_gen.generate(
                prompt=request.background_prompt,
                output_size=request.output_size,
                use_sd=request.use_sd_background,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_steps,
                seed=request.seed,
            )
            timings["background"] = time.perf_counter() - t0
            intermediate["background"] = bg_result.image
            self._progress(progress_callback, "generating_background", 0.72,
                           "Background ready.", t_total)

            # ── Stage 5: Compositing ───────────────────────────────────────────
            self._progress(progress_callback, "compositing", 0.74,
                           f"Compositing ({request.blend_mode})…", t_total)
            t0 = time.perf_counter()

            comp_result = self._compositor.composite(
                foreground=edit_result.edited_image,
                background=bg_result.image,
                alpha_mask=refined.alpha_mask,
                binary_mask=refined.binary_mask,
                mode=request.blend_mode,
            )
            timings["compositing"] = time.perf_counter() - t0
            intermediate["composite"] = comp_result.composite
            self._progress(progress_callback, "compositing", 0.80,
                           "Compositing complete.", t_total)

            composite = comp_result.composite

            # ── Stage 6: Shadow Synthesis ──────────────────────────────────────
            if request.enable_shadow:
                self._progress(progress_callback, "adding_shadow", 0.82,
                               "Synthesizing shadow…", t_total)
                t0 = time.perf_counter()

                shadow_result = self._shadow.synthesize(
                    composite=composite,
                    binary_mask=refined.binary_mask,
                    light_direction=request.shadow_direction,
                    blur_radius=request.shadow_blur,
                    opacity=request.shadow_opacity,
                )
                composite = shadow_result.image_with_shadow
                timings["shadow"] = time.perf_counter() - t0
                intermediate["shadow_image"] = composite
                self._progress(progress_callback, "adding_shadow", 0.88,
                               "Shadow added.", t_total)

            # ── Stage 7: Color Harmonization ───────────────────────────────────
            if request.enable_harmonization:
                self._progress(progress_callback, "harmonizing", 0.90,
                               "Harmonizing colors…", t_total)
                t0 = time.perf_counter()

                harm_result = self._harmonizer.harmonize(
                    composite=composite,
                    background=bg_result.image,
                    alpha_mask=refined.alpha_mask,
                )
                composite = harm_result.harmonized
                timings["harmonization"] = time.perf_counter() - t0
                self._progress(progress_callback, "harmonizing", 0.96,
                               "Color harmonization complete.", t_total)

            # ── Save output ────────────────────────────────────────────────────
            self._progress(progress_callback, "finalizing", 0.98,
                           "Finalizing output…", t_total)
            out_path = save_image(composite, temp_output_path(job_id, ".png", "final"))
            timings["total"] = time.perf_counter() - t_total

            self._progress(progress_callback, "done", 1.0,
                           f"Done in {timings['total']:.1f}s", t_total)

            logger.info(
                f"[Job {job_id}] Pipeline complete in {timings['total']:.2f}s | "
                f"seg={timings.get('segmentation', 0):.2f}s | "
                f"edit={timings.get('editing', 0):.2f}s | "
                f"bg={timings.get('background', 0):.2f}s"
            )

            clear_gpu_cache()

            return PipelineResult(
                job_id=job_id,
                final_image=composite,
                intermediate=intermediate,
                timings=timings,
                total_time_s=timings["total"],
                output_path=str(out_path),
            )

        except Exception as exc:
            logger.error(f"[Job {job_id}] Pipeline error: {exc}", exc_info=True)
            self._progress(progress_callback, "error", 0.0,
                           f"Error: {exc}", t_total)
            clear_gpu_cache()
            raise


# ── Async wrapper ──────────────────────────────────────────────────────────────

class AsyncPipelineOrchestrator:
    """
    Async-safe wrapper around PipelineOrchestrator.
    Runs the synchronous pipeline in a thread pool to avoid blocking the
    FastAPI event loop.
    """

    def __init__(self, device=None):
        self._sync = PipelineOrchestrator(device=device)

    async def run_async(
        self,
        request: PipelineRequest,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """Run the pipeline in a thread pool executor."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Default thread pool
            lambda: self._sync.run(request, job_id, progress_callback),
        )
        return result

    def preload_all(self) -> None:
        self._sync.preload_all()
