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
import re
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
from src.models.lama_inpainting import LocalInpainter
from src.models.segmentation import SegmentationEngine, SegmentationResult
from src.pipeline.harmonization import ColorHarmonizer
from src.pipeline.mask_utils import refine_mask
from src.pipeline.shadow import ShadowSynthesizer
from src.utils.device import get_device, clear_gpu_cache
from src.utils.image_io import resize_for_pipeline, save_image, temp_output_path, upscale_to_original
from src.pipeline.preprocessing import PhotoPreprocessor
from src.pipeline.face_preservation import FaceProtector, FaceRestorer, SkinTonePreserver
from src.pipeline.object_removal import ObjectRemovalPipeline
from src.pipeline.prompt_parser import parse_prompt, ParsedIntent, BackgroundType, parse_raw_background_prompt
from src.pipeline.execution_planner import (
    build_execution_plan,
    BackgroundStrategy,
    ForegroundStrategy,
    ExecutionPlan,
)
def parse_color_string(color_str: str) -> Tuple[int, int, int]:
    """Parses color string (e.g. 'color:#0000FF' or 'color:rgb(0,0,255)') to RGB tuple."""
    if not color_str.startswith("color:"):
        return (255, 255, 255)
        
    val = color_str[len("color:"):].strip()
    
    # 1. Hex format: e.g. #0000FF
    if val.startswith("#"):
        hex_val = val[1:]
        if len(hex_val) == 3:
            hex_val = "".join(c*2 for c in hex_val)
        try:
            r = int(hex_val[0:2], 16)
            g = int(hex_val[2:4], 16)
            b = int(hex_val[4:6], 16)
            return (r, g, b)
        except ValueError:
            return (255, 255, 255)
            
    # 2. RGB format: e.g. rgb(0,0,255)
    if val.startswith("rgb("):
        try:
            match = re.search(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', val)
            if match:
                r, g, b = map(int, match.groups())
                return (r, g, b)
        except Exception:
            pass
            
    return (255, 255, 255)


# ── Stage Enum ────────────────────────────────────────────────────────────────

class PipelineStage(str, Enum):
    IDLE = "idle"
    PREPROCESSING = "preprocessing"
    REMOVING_OBJECTS = "removing_objects"
    SEGMENTING = "segmenting"
    REFINING_MASK = "refining_mask"
    EDITING_OBJECT = "editing_object"
    GENERATING_BACKGROUND = "generating_background"
    COMPOSITING = "compositing"
    ADDING_SHADOW = "adding_shadow"
    HARMONIZING = "harmonizing"
    PRESERVING_IDENTITY = "preserving_identity"
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
    enable_preprocessing: bool = True
    enable_face_preservation: bool = True
    enable_matting: bool = True
    remove_objects: List[str] = field(default_factory=list)
    solid_background: bool = False
    command: Optional[str] = None
    custom_background: Optional[np.ndarray] = None


@dataclass
class PipelineResult:
    job_id: str
    final_image: np.ndarray                      # Final composited image
    intermediate: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    total_time_s: float = 0.0
    output_path: Optional[str] = None
    execution_plan: Optional[ExecutionPlan] = None


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
        self._preprocessor = PhotoPreprocessor()
        self._face_protector = FaceProtector()
        self._face_restorer = FaceRestorer()
        self._skin_tone_preserver = SkinTonePreserver()
        self._local_inpainter = LocalInpainter()
        self._object_remover = ObjectRemovalPipeline(device=self.device, face_protector=self._face_protector)


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

        # ── Layer 1 & 2: Intent parsing & Execution Planning ──────────────────────
        # Always run through the parser and planner to enforce safety guards.
        if request.command:
            logger.info(f"[Job {job_id}] Parsing intent from command: '{request.command}'")
            intent = parse_prompt(request.command)
            logger.info(
                f"[Job {job_id}] ParsedIntent → "
                f"remove={intent.remove_targets} | "
                f"bg_request='{intent.background_request}' | "
                f"bg_color='{intent.background_color}' | "
                f"styles={intent.style_descriptors} | "
                f"confidence={intent.parse_confidence:.2f}"
            )
        else:
            logger.info(
                f"[Job {job_id}] Synthesizing intent from request params: "
                f"bg_prompt='{request.background_prompt}', "
                f"remove={request.remove_objects}"
            )
            # Parse background_prompt directly
            bg_req, bg_col, bg_type = None, None, None
            transparency = False
            if request.background_prompt:
                bg_req, bg_col, bg_type = parse_raw_background_prompt(request.background_prompt)
                transparency = (bg_type == BackgroundType.TRANSPARENT)

            # Extract style descriptors from style_preset
            style_descriptors = []
            if request.style_preset and request.style_preset.lower() != "realistic":
                style_descriptors.append(request.style_preset.lower())

            intent = ParsedIntent(
                remove_targets=list(request.remove_objects),
                replace_targets={},
                background_request=bg_req,
                background_color=bg_col,
                background_type=bg_type,
                style_descriptors=style_descriptors,
                preserve_identity=request.enable_face_preservation,
                preserve_foreground=(not request.use_sd_background),
                transparency_requested=transparency,
                raw_prompt=request.background_prompt or "",
                parse_confidence=1.0,
            )

        execution_plan = build_execution_plan(
            intent,
            has_custom_background=(request.custom_background is not None),
            use_sd_background=request.use_sd_background,
            current_style_preset=request.style_preset,
        )
        logger.info(
            f"[Job {job_id}] ExecutionPlan reasoning: {execution_plan.reasoning}"
        )

        # Apply plan to request — orchestrator reads plan flags, not raw intent
        if execution_plan.removal_targets:
            request.remove_objects = execution_plan.removal_targets

        # Background prompt: color hex or scene description
        if execution_plan.background_color_hex:
            request.background_prompt = f"color:{execution_plan.background_color_hex}"
        elif execution_plan.background_scene_prompt:
            request.background_prompt = execution_plan.background_scene_prompt

        # Solid background flag for compositing stage
        request.solid_background = execution_plan.background_strategy in (
            BackgroundStrategy.COLOR_FILL,
            BackgroundStrategy.TRANSPARENT_FILL,
        )

        # Feature flags from plan (combine with request preferences to respect explicit disabling)
        request.enable_face_preservation = execution_plan.enable_face_preservation and request.enable_face_preservation
        request.enable_harmonization = execution_plan.enable_harmonization and request.enable_harmonization
        request.enable_matting = execution_plan.enable_matting and request.enable_matting

        # Style preset (from plan, falls back to original request value)
        if execution_plan.style_preset:
            request.style_preset = execution_plan.style_preset

        # SD safety guard: override use_sd_background with planner decision
        request.use_sd_background = execution_plan.sd_generation_approved

        logger.info(
            f"[Job {job_id}] Applied plan → "
            f"remove={request.remove_objects} | "
            f"bg='{request.background_prompt}' | "
            f"solid={request.solid_background} | "
            f"sd_bg={request.use_sd_background} | "
            f"sd_inpaint={execution_plan.sd_inpainting_approved} | "
            f"style='{request.style_preset}'"
        )

        logger.info(f"[Job {job_id}] Pipeline starting | "
                    f"mode={request.segmentation_mode} | "
                    f"style={request.style_preset}")

        logger.info(f"Parsed intent: {intent}")
        logger.info(f"Background type: {execution_plan.background_type.name if hasattr(execution_plan.background_type, 'name') else str(execution_plan.background_type)}")
        logger.info(f"Use diffusion: {execution_plan.use_diffusion}")
        logger.info(f"Use local inpainting: {execution_plan.use_local_inpainting}")
        logger.info(f"Use color compositing: {execution_plan.use_color_compositing}")
        logger.info(f"Preserve all humans: {execution_plan.preserve_all_people}")

        image = request.image
        orig_h, orig_w = image.shape[:2]
        original_size = (orig_w, orig_h)

        try:
            # ── Stage 0: Preprocessing ─────────────────────────────────────────
            if request.enable_preprocessing:
                self._progress(progress_callback, "preprocessing", 0.02,
                               "Preprocessing image (artifact/noise removal, exposure correction)...", t_total)
                t0 = time.perf_counter()
                preproc_res = self._preprocessor.preprocess(
                    image,
                    enable_denoise=True,
                    enable_exposure=True,
                    enable_white_balance=True,
                    enable_artifact_removal=True,
                    min_resolution=512,
                )
                image = preproc_res.image
                original_size = (preproc_res.original_size[1], preproc_res.original_size[0])  # Track original size as (width, height)
                timings["preprocessing"] = time.perf_counter() - t0
                intermediate["preprocessing_corrections"] = preproc_res.applied_corrections
                self._progress(progress_callback, "preprocessing", 0.04,
                               f"Preprocessing complete. Applied: {', '.join(preproc_res.applied_corrections) or 'none'}", t_total)

            # ── Stage 0.5: Targeted Object Removal ─────────────────────────────
            if request.remove_objects:
                self._progress(progress_callback, "removing_objects", 0.04,
                               f"Detecting and removing targets ({', '.join(request.remove_objects)})...", t_total)
                t0 = time.perf_counter()
                removal_res = self._object_remover.remove(
                    image,
                    local_inpainter=self._local_inpainter,
                    targets=request.remove_objects,
                )
                image = removal_res.image
                timings["object_removal"] = time.perf_counter() - t0
                intermediate["object_removal_mask"] = removal_res.object_mask
                self._progress(progress_callback, "removing_objects", 0.05,
                               "Targets removed from image.", t_total)
                logger.info("Localized object removal completed...")

            # Save original high-resolution image with preprocessing/removal applied
            high_res_foreground = image.copy()

            # Resize for pipeline if too large (preserves aspect)
            image, pipeline_size = resize_for_pipeline(image, max_dim=max(*request.output_size))
            h, w = image.shape[:2]

            # ── Stage 1: Segmentation ──────────────────────────────────────────
            self._progress(progress_callback, "segmenting", 0.05,
                           "Segmenting object…", t_total)
            t0 = time.perf_counter()

            seg_result: Optional[SegmentationResult] = None
            if request.enable_matting:
                if request.segmentation_mode == "interactive" and request.click_points:
                    seg_result = self._segmentation.interactive_segment(
                        image, request.click_points
                    )
                    if seg_result:
                        alpha = self._segmentation._matting.refine(image, coarse_mask=seg_result.mask)
                        seg_result.alpha_matte = alpha
                        seg_result.mask = (alpha > 0.5).astype(np.uint8) * 255
                else:
                    seg_result = self._segmentation.segment_with_matting(image)
            else:
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
                alpha_matte = None
            else:
                raw_mask = seg_result.mask
                label = seg_result.label
                alpha_matte = seg_result.alpha_matte

            timings["segmentation"] = time.perf_counter() - t0
            intermediate["raw_mask"] = raw_mask
            intermediate["label"] = label
            if alpha_matte is not None:
                intermediate["alpha_matte"] = alpha_matte
            self._progress(progress_callback, "segmenting", 0.15,
                           f"Object '{label}' detected.", t_total)

            # ── Stage 2: Mask Refinement ───────────────────────────────────────
            self._progress(progress_callback, "refining_mask", 0.18,
                           "Refining mask boundaries…", t_total)
            t0 = time.perf_counter()

            refined = refine_mask(raw_mask, image_rgb=image, alpha_matte=alpha_matte, feather_radius=4)
            timings["mask_refinement"] = time.perf_counter() - t0
            intermediate["alpha_mask"] = refined.alpha_mask
            intermediate["binary_mask"] = refined.binary_mask

            self._progress(progress_callback, "refining_mask", 0.25,
                           "Mask refined.", t_total)

            # ── Stage 2.5: Face Protection Detection ───────────────────────────
            faces = []
            face_mask = None
            skin_ref = None
            if request.enable_face_preservation:
                self._progress(progress_callback, "preserving_identity", 0.26,
                               "Detecting and protecting faces…", t_total)
                t0 = time.perf_counter()
                faces = self._face_protector.detect_faces(image)
                if faces:
                    face_mask = self._face_protector.create_protection_mask(image.shape[:2], faces)
                    skin_ref = self._skin_tone_preserver.extract_skin_reference(image, faces)
                    intermediate["faces_detected"] = len(faces)
                timings["face_detection"] = time.perf_counter() - t0

            # ── Stage 3: Object Editing (SD Inpainting) ────────────────────────
            # ── Stage 3: Object Editing (SD Inpainting) ──────────────────────────────
            # SD inpainting is only activated when the execution plan explicitly
            # approves it (plan.sd_inpainting_approved). For backwards-compat
            # requests without a command, fall back to the original heuristic.
            if execution_plan is not None:
                skip_editing = not execution_plan.sd_inpainting_approved
            else:
                skip_editing = (
                    request.solid_background
                    or request.object_prompt in ("", "object", "none")
                )
            
            if skip_editing:
                logger.info("Skipping subject editing/inpainting stage to preserve original subject pixels.")
                from src.models.editing import EditingResult
                edit_result = EditingResult(
                    edited_image=image,
                    edited_crop=image,
                    prompt_used="",
                    style_preset=request.style_preset,
                    inference_time_s=0.0,
                )
                timings["editing"] = 0.0
                intermediate["edited_image"] = image
            else:
                self._progress(progress_callback, "editing_object", 0.28,
                               f"Editing object with '{request.style_preset}' style…",
                               t_total)
                t0 = time.perf_counter()

                edit_mask = refined.binary_mask.copy()
                if request.enable_face_preservation and faces and face_mask is not None:
                    # Subtract face mask to prevent SD from modifying the face
                    edit_mask = cv2.subtract(edit_mask, face_mask)

                edit_result = self._editor.edit(
                    image_rgb=image,
                    mask=edit_mask,
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
            if request.custom_background is not None:
                self._progress(progress_callback, "generating_background", 0.52,
                               "Processing custom background image...", t_total)
                t0 = time.perf_counter()
                bg_img = cv2.resize(request.custom_background, (w, h), interpolation=cv2.INTER_LANCZOS4)
                timings["background"] = time.perf_counter() - t0
                intermediate["background"] = bg_img
                self._progress(progress_callback, "generating_background", 0.72,
                               "Custom background ready.", t_total)
            elif request.solid_background:
                logger.info("Generating solid RGB background...")
                self._progress(progress_callback, "generating_background", 0.52,
                               f"Creating solid background ({request.background_prompt or 'white'})...", t_total)
                t0 = time.perf_counter()
                if request.background_prompt == "transparent":
                    bg_img = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    bg_color = (255, 255, 255)  # default white
                    if request.background_prompt and request.background_prompt.startswith("color:"):
                        bg_color = parse_color_string(request.background_prompt)
                    elif request.background_prompt == "solid white":
                        bg_color = (255, 255, 255)
                    bg_img = np.zeros((h, w, 3), dtype=np.uint8)
                    bg_img[:] = bg_color
                timings["background"] = time.perf_counter() - t0
                intermediate["background"] = bg_img
                self._progress(progress_callback, "generating_background", 0.72,
                               "Solid background ready.", t_total)
            else:
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
                bg_img = bg_result.image
                timings["background"] = time.perf_counter() - t0
                intermediate["background"] = bg_img
                self._progress(progress_callback, "generating_background", 0.72,
                               "Background ready.", t_total)

            # ── Stage 5: Compositing ───────────────────────────────────────────
            is_transparent = request.solid_background and request.background_prompt == "transparent"

            if is_transparent:
                logger.info("Transparent background requested; bypassing compositing, shadows, and color harmonization.")
                composite = edit_result.edited_image
                timings["compositing"] = 0.0
                timings["shadow"] = 0.0
                timings["harmonization"] = 0.0
            else:
                self._progress(progress_callback, "compositing", 0.74,
                               f"Compositing ({request.blend_mode})…", t_total)
                t0 = time.perf_counter()

                comp_result = self._compositor.composite(
                    foreground=edit_result.edited_image,
                    background=bg_img,
                    alpha_mask=refined.alpha_mask,
                    binary_mask=refined.binary_mask,
                    mode=request.blend_mode,
                )
                timings["compositing"] = time.perf_counter() - t0
                intermediate["composite"] = comp_result.composite
                self._progress(progress_callback, "compositing", 0.80,
                               "Compositing complete.", t_total)
                logger.info("Foreground compositing completed...")

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
                if request.enable_harmonization and not request.solid_background:
                    self._progress(progress_callback, "harmonizing", 0.90,
                                   "Harmonizing colors…", t_total)
                    t0 = time.perf_counter()

                    harm_result = self._harmonizer.harmonize(
                        composite=composite,
                        background=bg_img,
                        alpha_mask=refined.alpha_mask,
                    )
                    composite = harm_result.harmonized
                    timings["harmonization"] = time.perf_counter() - t0
                    self._progress(progress_callback, "harmonizing", 0.94,
                                   "Color harmonization complete.", t_total)

            # ── Stage 7.5: Skin Tone Preservation & Face Restoration ───────────
            if request.enable_face_preservation and faces:
                self._progress(progress_callback, "preserving_identity", 0.95,
                               "Restoring identity and skin tones...", t_total)
                t0 = time.perf_counter()

                # Skin tone preservation
                if skin_ref is not None:
                    composite = self._skin_tone_preserver.apply_skin_preservation(
                        composite, skin_ref, faces, strength=0.5
                    )
                # Face restoration
                restore_res = self._face_restorer.restore(composite, image, faces, fidelity=0.8)
                composite = restore_res.restored_image
                timings["face_restoration"] = time.perf_counter() - t0
                self._progress(progress_callback, "preserving_identity", 0.97,
                               f"Face restoration complete. Restored {restore_res.num_faces_restored} face(s).", t_total)

            # ── Stage 7.8: Transparent Background ──────────────────────────────
            if is_transparent:
                alpha_channel = (refined.alpha_mask * 255).clip(0, 255).astype(np.uint8)
                # Ensure sizes match before stacking
                if alpha_channel.shape[:2] != composite.shape[:2]:
                    alpha_channel = cv2.resize(alpha_channel, (composite.shape[1], composite.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                composite = np.dstack((composite, alpha_channel))

            # ── Stage 8: Upscale back to Original Size ──────────────────────────
            self._progress(progress_callback, "finalizing", 0.98,
                           "Upscaling back to original dimensions...", t_total)
            composite = upscale_to_original(composite, original_size)

            # If subject editing was skipped, restore the exact high-resolution original foreground pixels
            if skip_editing:
                logger.info("Restoring original high-resolution foreground pixels exactly.")
                # Upscale alpha mask to original size
                alpha_mask_high = cv2.resize(refined.alpha_mask, original_size, interpolation=cv2.INTER_LANCZOS4)
                if is_transparent:
                    # Stack high-res foreground with upscaled alpha mask directly
                    alpha_channel = (alpha_mask_high * 255).clip(0, 255).astype(np.uint8)
                    composite = np.dstack((high_res_foreground, alpha_channel))
                else:
                    # Blend original high-res foreground onto composite
                    alpha_mask_3c = np.dstack([alpha_mask_high] * 3)
                    composite = (high_res_foreground * alpha_mask_3c + composite * (1.0 - alpha_mask_3c)).clip(0, 255).astype(np.uint8)

            # ── Save output ────────────────────────────────────────────────────
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
                execution_plan=execution_plan,
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
