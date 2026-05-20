"""
SceneShift — Gradio Frontend Application
=========================================
Real-Time Semantic Object Editing & Intelligent Scene Transformation

Run with:
    python app.py

Access at:  http://localhost:7860
FastAPI at: http://localhost:8000
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import uvicorn
from loguru import logger
from PIL import Image

# ── Configure Loguru ──────────────────────────────────────────────────────────
# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add("logs/sceneshift.log", rotation="10 MB", retention="7 days", level="DEBUG")

Path("logs").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# ── Import SceneShift modules ──────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio not installed. Run: pip install gradio>=4.7.1")

from src.api.server import app as fastapi_app
from src.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineRequest,
    ProgressEvent,
)
from src.utils.device import get_device, get_device_info
from src.utils.image_io import pil_to_numpy, numpy_to_pil, save_image
from src.utils.validators import (
    STYLE_PRESETS,
    BLEND_MODES,
    validate_image_bytes,
    ValidationError,
)

# ── Style presets ──────────────────────────────────────────────────────────────
# STYLE_PRESETS is a list of strings e.g. ["Realistic", "Cinematic", ...]
STYLE_NAMES = list(STYLE_PRESETS)

# Display label → internal key mapping for blend modes
BLEND_MODE_LABELS = {
    "Alpha Blending": "alpha",
    "Poisson Blending": "poisson",
    "Laplacian Pyramid": "laplacian",
}

# Display label → internal key mapping for segmentation modes
SEG_MODE_LABELS = {
    "Automatic (YOLO)": "auto",
    "Interactive (SAM2 — Click)": "interactive",
}

# ── Global orchestrator (singleton) ───────────────────────────────────────────
_orchestrator: Optional[PipelineOrchestrator] = None


def get_pipeline() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initializing pipeline…")
        _orchestrator = PipelineOrchestrator(device=get_device())
    return _orchestrator


# ── FastAPI background thread ──────────────────────────────────────────────────

def _start_fastapi_server(port: int = 8000) -> None:
    """Start FastAPI server in a background daemon thread."""
    logger.info(f"Starting FastAPI server on port {port}…")
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )


def launch_fastapi_background(port: int = 8000) -> None:
    t = threading.Thread(target=_start_fastapi_server, args=(port,), daemon=True)
    t.start()
    time.sleep(1.5)
    logger.info(f"FastAPI server running at http://localhost:{port}")


# ── Core pipeline function ─────────────────────────────────────────────────────

def run_sceneshift(
    input_image: np.ndarray,
    object_prompt: str,
    background_prompt: str,
    style_preset: str,
    blend_mode_label: str,
    seg_mode_label: str,
    strength: float,
    guidance_scale: float,
    num_steps: int,
    enable_shadow: bool,
    shadow_opacity: float,
    shadow_blur: int,
    shadow_dir_x: float,
    shadow_dir_y: float,
    enable_harmonization: bool,
    use_sd_background: bool,
    seed_val: int,
    click_x_pct: float,
    click_y_pct: float,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Main pipeline callback for Gradio.

    Returns:
        (result_image, original_image, status_text)
    """
    if input_image is None:
        return None, None, "⚠️ Please upload an image first."

    if not object_prompt.strip():
        object_prompt = "object"
    if not background_prompt.strip():
        background_prompt = "natural background"

    blend_mode = BLEND_MODE_LABELS.get(blend_mode_label, "alpha")
    seg_mode = SEG_MODE_LABELS.get(seg_mode_label, "auto")
    seed = int(seed_val) if seed_val > 0 else None

    # Build click points
    click_points = []
    if seg_mode == "interactive" and click_x_pct > 0 and click_y_pct > 0:
        h, w = input_image.shape[:2]
        cx = int(click_x_pct * w)
        cy = int(click_y_pct * h)
        click_points = [(cx, cy)]
        logger.info(f"Interactive click: ({cx}, {cy})")

    progress(0.02, desc="🔄 Initializing pipeline…")

    # Progress stage → Gradio progress bar mapping
    stage_to_progress = {
        "segmenting": 0.15,
        "refining_mask": 0.25,
        "editing_object": 0.50,
        "generating_background": 0.70,
        "compositing": 0.80,
        "adding_shadow": 0.88,
        "harmonizing": 0.95,
        "done": 1.00,
        "error": 0.00,
    }

    def on_progress(event: ProgressEvent) -> None:
        pct = stage_to_progress.get(event.stage, event.progress)
        progress(pct, desc=f"⚙️ {event.message}")

    try:
        req = PipelineRequest(
            image=input_image,
            object_prompt=object_prompt,
            background_prompt=background_prompt,
            style_preset=style_preset,
            blend_mode=blend_mode,
            segmentation_mode=seg_mode,
            click_points=click_points,
            strength=strength,
            guidance_scale=guidance_scale,
            num_steps=int(num_steps),
            shadow_direction=(shadow_dir_x, shadow_dir_y),
            shadow_opacity=shadow_opacity,
            shadow_blur=int(shadow_blur),
            enable_shadow=enable_shadow,
            enable_harmonization=enable_harmonization,
            use_sd_background=use_sd_background,
            seed=seed,
            output_size=(512, 512),
        )

        pipeline = get_pipeline()
        result = pipeline.run(req, progress_callback=on_progress)

        progress(1.0, desc="✅ Processing complete!")

        timings = result.timings
        status = (
            f"✅ **Complete** in `{result.total_time_s:.1f}s`\n\n"
            f"| Stage | Time |\n|---|---|\n"
            f"| Segmentation | `{timings.get('segmentation', 0):.2f}s` |\n"
            f"| Mask Refinement | `{timings.get('mask_refinement', 0):.2f}s` |\n"
            f"| Object Editing | `{timings.get('editing', 0):.2f}s` |\n"
            f"| Background Gen | `{timings.get('background', 0):.2f}s` |\n"
            f"| Compositing | `{timings.get('compositing', 0):.2f}s` |\n"
            f"| Shadow | `{timings.get('shadow', 0):.2f}s` |\n"
            f"| Harmonization | `{timings.get('harmonization', 0):.2f}s` |\n\n"
            f"Style: **{style_preset}** | Blend: **{blend_mode_label}**"
        )

        return result.final_image, input_image, status

    except Exception as exc:
        logger.error(f"Pipeline error: {exc}", exc_info=True)
        return None, input_image, f"❌ Error: {exc}"


# ── Gradio UI ──────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #0f1623 100%) !important;
    min-height: 100vh;
}

/* Header */
.sceneshift-header {
    text-align: center;
    padding: 2.5rem 1rem 1.8rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.1) 50%, rgba(244,114,182,0.05));
    border-bottom: 1px solid rgba(139,92,246,0.3);
    margin-bottom: 1.5rem;
    border-radius: 20px;
    box-shadow: 0 4px 30px rgba(139, 92, 246, 0.15);
}

.sceneshift-header h1 {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #a5b4fc, #c084fc, #f472b6, #a5b4fc) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -1px;
}

.sceneshift-header p {
    color: #94a3b8 !important;
    font-size: 1.05rem !important;
    font-weight: 400 !important;
}

/* Panels */
.panel-card {
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.78), rgba(8, 13, 28, 0.72)) !important;
    border: 1px solid rgba(139, 92, 246, 0.28) !important;
    border-radius: 18px !important;
    padding: 1.25rem !important;
    backdrop-filter: blur(16px) saturate(180%);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* Scrollable Panel for Controls */
.left-control-panel {
    max-height: 100vh;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 1.25rem 1rem 1.25rem 1.25rem !important;
    scrollbar-width: thin;
    scrollbar-color: rgba(139, 92, 246, 0.75) rgba(13, 17, 28, 0.35);
}
.left-control-panel::-webkit-scrollbar {
    width: 8px;
}
.left-control-panel::-webkit-scrollbar-track {
    background: rgba(13, 17, 28, 0.3);
    border-radius: 10px;
}
.left-control-panel::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.85), rgba(236, 72, 153, 0.7));
    border-radius: 10px;
    border: 2px solid rgba(13, 17, 28, 0.75);
}
.left-control-panel::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.8), rgba(139, 92, 246, 0.8));
}

.results-panel {
    position: sticky !important;
    top: 1rem;
    align-self: flex-start;
    width: 100%;
}

/* Buttons */
.run-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}

.run-btn:hover {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
}

.clear-btn {
    background: rgba(239,68,68,0.12) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.clear-btn:hover {
    background: rgba(239,68,68,0.25) !important;
    transform: translateY(-1px) !important;
}

/* Labels */
label span, .label-wrap span {
    color: #c4b5fd !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #8b5cf6 !important;
}

/* Dropdown */
select, .gr-dropdown select {
    background: rgba(15,20,35,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Textboxes */
textarea, input[type="text"] {
    background: rgba(15,20,35,0.9) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: rgba(139,92,246,0.6) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}

/* Section headers */
.section-title {
    color: #818cf8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid rgba(99,102,241,0.2) !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Status box */
.status-box {
    background: rgba(15,20,35,0.6) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}

/* Image upload area */
.image-container {
    border: 2px dashed rgba(139, 92, 246, 0.3) !important;
    border-radius: 16px !important;
    background: rgba(10,10,18,0.6) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.image-container:hover {
    border-color: rgba(168, 85, 247, 0.8) !important;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.25) !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: #8b5cf6 !important;
}

/* Badge pill */
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* Custom interactive progress bar CSS */
.progress-container {
    padding: 1.5rem;
    background: rgba(10, 12, 22, 0.85);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    margin: 1rem 0;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);
}
.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
    color: #e2e8f0;
}
.progress-percent {
    font-weight: 700;
    color: #a78bfa;
}
.progress-bar-bg {
    width: 100%;
    height: 8px;
    background: rgba(30, 41, 59, 0.8);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.2rem;
    position: relative;
}
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
    background-size: 200% 100%;
    border-radius: 10px;
    transition: width 0.4s ease;
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.6);
    animation: progressShimmer 1.2s linear infinite;
}
.progress-stages {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 0.5rem;
}
.stage-item {
    padding: 0.4rem 0.6rem;
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.75rem;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.3s ease;
}
.stage-item.active {
    background: rgba(139, 92, 246, 0.15);
    border-color: rgba(139, 92, 246, 0.5);
    color: #e2e8f0;
    font-weight: 500;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.1);
}
.stage-item.completed {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.4);
    color: #a7f3d0;
}
.stage-icon {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #64748b;
}
.stage-item.active .stage-icon {
    background: #a78bfa;
    box-shadow: 0 0 6px #a78bfa;
    animation: pulse 1.5s infinite;
}
.stage-item.completed .stage-icon {
    background: #10b981;
}

@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.6; }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes progressShimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

.comparison-container {
    gap: 1.5rem !important;
    margin-top: 1rem !important;
}

@media (max-width: 1100px) {
    .left-control-panel {
        max-height: 70vh;
    }
    .results-panel {
        position: static !important;
    }
    .comparison-container {
        flex-direction: column !important;
    }
}
"""

def make_progress_html(current_stage: str, pct: float, message: str) -> str:
    """Construct HTML markup for the custom progress bar and checklist."""
    stages = [
        ("segmenting", "Segmenting"),
        ("refining_mask", "Refining Mask"),
        ("editing_object", "Editing Object"),
        ("generating_background", "Generating Background"),
        ("compositing", "Compositing"),
        ("harmonizing", "Harmonizing"),
        ("finalizing", "Finalizing"),
    ]

    stage_idx = -1
    for i, (key, _) in enumerate(stages):
        if key == current_stage:
            stage_idx = i
            break

    pct_val = int(pct * 100)

    if current_stage == "done":
        pct_val = 100
        html = f"""
        <div class="progress-container" style="border-color: rgba(16, 185, 129, 0.4); box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);">
            <div class="progress-header">
                <span style="color: #a7f3d0; font-weight: 600;">✅ {message}</span>
                <span class="progress-percent" style="color: #10b981;">100%</span>
            </div>
            <div class="progress-bar-bg" style="margin-bottom: 0px;">
                <div class="progress-bar-fill" style="width: 100%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
            </div>
        </div>
        """
        return html
    elif current_stage == "error":
        html = f"""
        <div class="progress-container" style="border-color: rgba(239, 68, 68, 0.4); box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);">
            <div class="progress-header">
                <span style="color: #f87171; font-weight: 600;">❌ {message}</span>
            </div>
            <div class="progress-bar-bg" style="margin-bottom: 0px;">
                <div class="progress-bar-fill" style="width: 100%; background: #ef4444;"></div>
            </div>
        </div>
        """
        return html

    html = f"""
    <div class="progress-container">
        <div class="progress-header">
            <span>⚙️ {message}</span>
            <span class="progress-percent">{pct_val}%</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {pct_val}%"></div>
        </div>
        <div class="progress-stages">
    """
    for i, (key, name) in enumerate(stages):
        if i < stage_idx:
            status_class = "completed"
        elif i == stage_idx:
            status_class = "active"
        else:
            status_class = ""
        html += f"""
            <div class="stage-item {status_class}">
                <span class="stage-icon"></span>
                <span>{name}</span>
            </div>
        """
    html += """
        </div>
    </div>
    """
    return html


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks UI."""

    demo = gr.Blocks(title="SceneShift - AI Scene Transformation")
    demo.css = CSS
    demo.theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    with demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="sceneshift-header">
            <h1>🎨 SceneShift</h1>
            <p>Real-Time Semantic Object Editing &amp; Intelligent Scene Transformation</p>
            <div style="margin-top:0.8rem;">
                <span class="badge">YOLOv8</span>&nbsp;
                <span class="badge">SAM2</span>&nbsp;
                <span class="badge">Stable Diffusion</span>&nbsp;
                <span class="badge">Poisson Blending</span>&nbsp;
                <span class="badge">LAB Harmonization</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left panel — Controls ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=320, elem_classes=["left-control-panel", "panel-card"]):

                gr.HTML('<div class="section-title">📥 Input Image</div>')
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    elem_classes=["image-container"],
                    height=280,
                )

                gr.HTML('<div class="section-title" style="margin-top:1rem;">🎯 Segmentation</div>')
                seg_mode = gr.Dropdown(
                    choices=list(SEG_MODE_LABELS.keys()),
                    value="Automatic (YOLO)",
                    label="Segmentation Mode",
                )

                with gr.Row():
                    click_x = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.01,
                        label="Click X (fraction)",
                        visible=False,
                    )
                    click_y = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.01,
                        label="Click Y (fraction)",
                        visible=False,
                    )

                def toggle_click(mode):
                    is_interactive = mode == "Interactive (SAM2 — Click)"
                    return gr.update(visible=is_interactive), gr.update(visible=is_interactive)

                seg_mode.change(toggle_click, seg_mode, [click_x, click_y])

                gr.HTML('<div class="section-title" style="margin-top:1rem;">✨ Style & Prompts</div>')
                style_preset = gr.Dropdown(
                    choices=STYLE_NAMES,
                    value="Realistic",
                    label="Style Preset",
                )
                object_prompt = gr.Textbox(
                    label="Object Prompt",
                    placeholder="e.g., golden retriever dog, sports car, ancient statue…",
                    lines=2,
                )
                bg_prompt = gr.Textbox(
                    label="Background Prompt",
                    placeholder="e.g., sunset beach, cyberpunk city, enchanted forest…",
                    lines=2,
                )

                gr.HTML('<div class="section-title" style="margin-top:1rem;">🔀 Compositing</div>')
                blend_mode = gr.Dropdown(
                    choices=list(BLEND_MODE_LABELS.keys()),
                    value="Alpha Blending",
                    label="Blend Mode",
                )

                gr.HTML('<div class="section-title" style="margin-top:1rem;">⚙️ Pipeline Settings</div>')
                with gr.Accordion("Advanced Settings", open=False):
                    strength = gr.Slider(0.3, 1.0, value=0.65, step=0.05, label="Inpainting Strength")
                    guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                    steps = gr.Slider(10, 100, value=30, step=5, label="Inference Steps")
                    seed = gr.Number(value=42, label="Seed (0 = random)", precision=0)

                    gr.HTML('<div class="section-title" style="margin-top:0.8rem;">🌑 Shadow Settings</div>')
                    enable_shadow = gr.Checkbox(value=True, label="Enable Shadow")
                    shadow_opacity = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Shadow Opacity")
                    shadow_blur = gr.Slider(0, 80, value=35, step=5, label="Blur Radius (Penumbra)")
                    with gr.Row():
                        shadow_x = gr.Slider(-1.0, 1.0, value=1.0, step=0.1, label="Light Dir X")
                        shadow_y = gr.Slider(-1.0, 1.0, value=0.5, step=0.1, label="Light Dir Y")

                    gr.HTML('<div class="section-title" style="margin-top:0.8rem;">🎨 Harmonization</div>')
                    enable_harm = gr.Checkbox(value=True, label="Enable Color Harmonization")
                    use_sd_bg = gr.Checkbox(value=True, label="Use Stable Diffusion Background")

            # ── Right panel — Results ──────────────────────────────────────────
            with gr.Column(scale=2, elem_classes=["panel-card", "results-panel"]):

                gr.HTML('<div class="section-title">🖼️ Results</div>')

                with gr.Tabs():
                    with gr.Tab("✨ Result"):
                        progress_display = gr.HTML(
                            value="",
                            elem_id="progress-display",
                            visible=False,
                        )
                        result_image = gr.Image(
                            label="SceneShift Output",
                            type="numpy",
                            interactive=False,
                            height=420,
                            elem_classes=["image-container"],
                        )

                    with gr.Tab("🔀 Before / After"):
                        with gr.Row(elem_classes=["comparison-container"]):
                            before_img = gr.Image(
                                label="📸 Original Image",
                                type="numpy",
                                interactive=False,
                                height=350,
                                elem_classes=["image-container"],
                            )
                            after_img = gr.Image(
                                label="🎨 Transformed Image",
                                type="numpy",
                                interactive=False,
                                height=350,
                                elem_classes=["image-container"],
                            )

                # ── Action buttons ─────────────────────────────────────────────
                with gr.Row():
                    run_btn = gr.Button(
                        "🚀 Transform Scene",
                        variant="primary",
                        elem_classes=["run-btn"],
                        size="lg",
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        elem_classes=["clear-btn"],
                        size="lg",
                    )

                # ── Download ───────────────────────────────────────────────────
                gr.HTML('<div class="section-title" style="margin-top:1rem;">📥 Download</div>')
                download_btn = gr.DownloadButton(
                    label="⬇️ Download Result (PNG)",
                    value=None,
                    visible=False,
                )
                download_file = gr.File(
                    label="Result PNG",
                    value=None,
                    visible=False,
                    interactive=False,
                )

                # ── Status ─────────────────────────────────────────────────────
                gr.HTML('<div class="section-title" style="margin-top:1rem;">📊 Pipeline Status</div>')
                status_md = gr.Markdown(
                    value="_Upload an image and click **Transform Scene** to begin._",
                    elem_classes=["status-box"],
                )

        # ── Pipeline execution ─────────────────────────────────────────────────
        _result_store = gr.State(None)  # Store result path for download

        # Instantly update before image when user uploads a file, preventing loading spinner on it
        input_image.change(
            fn=lambda img: img,
            inputs=input_image,
            outputs=before_img,
        )

        import queue

        def on_run(
            img, obj_p, bg_p, style, blend, seg, s, g, n,
            en_sh, sh_op, sh_bl, sh_x, sh_y, en_ha, use_sd,
            sd_val, cx, cy,
        ):
            if img is None:
                yield None, None, "⚠️ Please upload an image first.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                return

            if not obj_p.strip():
                obj_p = "object"
            if not bg_p.strip():
                bg_p = "natural background"

            blend_mode = BLEND_MODE_LABELS.get(blend, "alpha")
            seg_mode = SEG_MODE_LABELS.get(seg, "auto")
            seed = int(sd_val) if sd_val > 0 else None

            # Build click points
            click_points = []
            if seg_mode == "interactive" and cx > 0 and cy > 0:
                h, w = img.shape[:2]
                click_points = [(int(cx * w), int(cy * h))]

            q = queue.Queue()

            stage_to_progress = {
                "segmenting": 0.12,
                "refining_mask": 0.22,
                "editing_object": 0.48,
                "generating_background": 0.70,
                "compositing": 0.82,
                "adding_shadow": 0.88,
                "harmonizing": 0.94,
                "finalizing": 0.98,
                "done": 1.00,
                "error": 0.00,
            }

            def progress_callback(event: ProgressEvent):
                pct = stage_to_progress.get(event.stage, event.progress)
                q.put(("progress", (event.stage, pct, event.message)))

            def worker():
                try:
                    req = PipelineRequest(
                        image=img,
                        object_prompt=obj_p,
                        background_prompt=bg_p,
                        style_preset=style,
                        blend_mode=blend_mode,
                        segmentation_mode=seg_mode,
                        click_points=click_points,
                        strength=s,
                        guidance_scale=g,
                        num_steps=int(n),
                        shadow_direction=(sh_x, sh_y),
                        shadow_opacity=sh_op,
                        shadow_blur=int(sh_bl),
                        enable_shadow=en_sh,
                        enable_harmonization=en_ha,
                        use_sd_background=use_sd,
                        seed=seed,
                        output_size=(512, 512),
                    )
                    pipeline = get_pipeline()
                    res = pipeline.run(req, progress_callback=progress_callback)
                    q.put(("done", res))
                except Exception as exc:
                    q.put(("error", str(exc)))

            # Start worker thread
            t = threading.Thread(target=worker, daemon=True)
            t.start()

            # Initial progress display
            progress_html = make_progress_html("segmenting", 0.05, "Initializing pipeline...")
            yield gr.update(), gr.update(), gr.update(), gr.update(visible=False), gr.update(value=progress_html, visible=True), gr.update(visible=False)

            while t.is_alive() or not q.empty():
                try:
                    msg_type, data = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if msg_type == "progress":
                    stage, pct, msg = data
                    progress_html = make_progress_html(stage, pct, msg)
                    yield gr.update(), gr.update(), gr.update(), gr.update(visible=False), gr.update(value=progress_html, visible=True), gr.update(visible=False)
                elif msg_type == "done":
                    result = data
                    # Save result for download
                    dl_path = Path("outputs") / f"sceneshift_{uuid.uuid4().hex[:6]}.png"
                    dl_path.parent.mkdir(parents=True, exist_ok=True)
                    numpy_to_pil(result.final_image).save(dl_path)
                    dl_path = str(dl_path.resolve())
                    
                    timings = result.timings
                    status_text = (
                        f"✅ **Complete** in `{result.total_time_s:.1f}s`\n\n"
                        f"| Stage | Time |\n|---|---|\n"
                        f"| Segmentation | `{timings.get('segmentation', 0):.2f}s` |\n"
                        f"| Mask Refinement | `{timings.get('mask_refinement', 0):.2f}s` |\n"
                        f"| Object Editing | `{timings.get('editing', 0):.2f}s` |\n"
                        f"| Background Gen | `{timings.get('background', 0):.2f}s` |\n"
                        f"| Compositing | `{timings.get('compositing', 0):.2f}s` |\n"
                        f"| Shadow | `{timings.get('shadow', 0):.2f}s` |\n"
                        f"| Harmonization | `{timings.get('harmonization', 0):.2f}s` |\n\n"
                        f"Style: **{style}** | Blend: **{blend}**"
                    )
                    
                    done_html = make_progress_html("done", 1.0, f"Done in {result.total_time_s:.1f}s")
                    yield result.final_image, result.final_image, status_text, gr.update(value=dl_path, visible=True), gr.update(value=done_html, visible=True), gr.update(value=dl_path, visible=True)
                    return
                elif msg_type == "error":
                    err_msg = data
                    error_html = make_progress_html("error", 0.0, f"Error: {err_msg}")
                    yield None, None, f"❌ Error: {err_msg}", gr.update(visible=False), gr.update(value=error_html, visible=True), gr.update(visible=False)
                    return

        run_btn.click(
            fn=on_run,
            inputs=[
                input_image, object_prompt, bg_prompt,
                style_preset, blend_mode, seg_mode,
                strength, guidance, steps,
                enable_shadow, shadow_opacity, shadow_blur, shadow_x, shadow_y,
                enable_harm, use_sd_bg, seed,
                click_x, click_y,
            ],
            outputs=[result_image, after_img, status_md, download_btn, progress_display, download_file],
            show_progress=False,  # Use custom progress HTML overlay instead
        )

        def on_clear():
            return (
                None, None, None, None,
                "_Upload an image and click **Transform Scene** to begin._",
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
            )

        clear_btn.click(
            fn=on_clear,
            outputs=[input_image, result_image, before_img, after_img, status_md, download_btn, progress_display, download_file],
        )

        # ── Footer ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:1.5rem 0 0.5rem;color:#475569;font-size:0.82rem;">
            SceneShift v1.0.0 &nbsp;|&nbsp; GPU-Accelerated AI Image Editing
            &nbsp;|&nbsp;
            <a href="http://localhost:8000/docs" target="_blank"
               style="color:#818cf8;text-decoration:none;">
               API Docs ↗
            </a>
        </div>
        """)

    return demo


# ── Main entrypoint ────────────────────────────────────────────────────────────

def main():
    """Launch SceneShift with Gradio frontend and FastAPI backend."""

    # Print startup banner
    device_info = get_device_info()
    device_str = (
        f"{device_info['device_name']} "
        f"({device_info.get('vram_gb', 'N/A')} GB VRAM)"
        if device_info.get('vram_gb') else device_info['device_name']
    )

    print("\n" + "=" * 65)
    print("  SceneShift - Real-Time AI Scene Transformation")
    print("=" * 65)
    print(f"  Device    : {device_str}")
    print(f"  PyTorch   : {device_info.get('torch_version', 'unknown')}")
    print(f"  CUDA      : {device_info.get('cuda_version', 'N/A')}")
    print("-" * 65)
    print("  Gradio UI : http://localhost:7860")
    print("  FastAPI   : http://localhost:8000")
    print("  API Docs  : http://localhost:8000/docs")
    print("=" * 65 + "\n")

    # Launch FastAPI in background thread
    launch_fastapi_background(port=8000)

    # Build and launch Gradio
    demo = build_ui()
    demo.queue(max_size=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
