"""
Object Editing Module for SceneShift.

Uses Stable Diffusion Inpainting (via Hugging Face Diffusers) to perform
text-guided semantic editing of a selected object within a masked region.

Supports 8 style presets with automatic prompt augmentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from src.utils.device import get_device, get_dtype, clear_gpu_cache
from src.utils.image_io import numpy_to_pil, pil_to_numpy, mask_to_pil


# ── Style Presets ──────────────────────────────────────────────────────────────

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "Realistic": {
        "positive": (
            "photorealistic, 8K UHD, RAW photo, sharp focus, "
            "natural lighting, high detail, professional photography"
        ),
        "negative": (
            "cartoon, painting, illustration, blurry, noisy, "
            "low quality, deformed, artifact"
        ),
    },
    "Cinematic": {
        "positive": (
            "cinematic photography, film grain, dramatic lighting, "
            "anamorphic lens, depth of field, color graded, "
            "movie still, award-winning cinematography"
        ),
        "negative": (
            "amateur, flat lighting, oversaturated, cartoonish, "
            "low detail, noisy"
        ),
    },
    "Cyberpunk": {
        "positive": (
            "cyberpunk style, neon lights, futuristic cityscape, "
            "holographic HUD, rain-slicked streets, dramatic neon glow, "
            "blade runner aesthetic, highly detailed"
        ),
        "negative": (
            "nature, daytime, sunny, pastoral, vintage, old-fashioned, "
            "low quality, blurry"
        ),
    },
    "Cartoon": {
        "positive": (
            "2D cartoon illustration, flat shading, bold outlines, "
            "vibrant colors, clean lines, Pixar-style, cel-shaded, "
            "children's book illustration"
        ),
        "negative": (
            "photorealistic, 3D render, dark, gritty, blurry, "
            "detailed texture, harsh shadows"
        ),
    },
    "Pencil Sketch": {
        "positive": (
            "pencil sketch, graphite drawing, fine line art, "
            "cross-hatching, charcoal, monochrome, detailed pencilwork, "
            "traditional artwork, high contrast"
        ),
        "negative": (
            "color, photograph, digital art, oil painting, "
            "low detail, blurry"
        ),
    },
    "Fantasy": {
        "positive": (
            "epic fantasy art, magical atmosphere, mystical lighting, "
            "ethereal glow, intricate details, concept art, "
            "artstation trending, highly detailed"
        ),
        "negative": (
            "modern, realistic, photographic, industrial, plain, "
            "low quality, simple"
        ),
    },
    "Vintage": {
        "positive": (
            "vintage photography, 1960s film aesthetic, grain, "
            "warm faded tones, lomography, retro color palette, "
            "analog film, aged photograph"
        ),
        "negative": (
            "modern, digital, sharp, vibrant, oversaturated, "
            "high-contrast, clean, clinical"
        ),
    },
    "Minimalist": {
        "positive": (
            "minimalist design, clean composition, negative space, "
            "simple shapes, flat colors, modern aesthetic, "
            "Scandinavian design, elegant"
        ),
        "negative": (
            "cluttered, complex, ornate, busy, detailed texture, "
            "dark, gritty, photorealistic"
        ),
    },
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EditingResult:
    """Output from the object editing stage."""
    edited_image: np.ndarray    # Full edited image (H x W x 3, uint8, RGB)
    edited_crop: np.ndarray     # Cropped edited region
    prompt_used: str            # Final positive prompt sent to SD
    style_preset: str           # Style preset name
    inference_time_s: float     # SD inference wall-clock time


# ── Object Editor ─────────────────────────────────────────────────────────────

class ObjectEditor:
    """
    Stable Diffusion Inpainting-based object editor.

    Loads runwayml/stable-diffusion-inpainting (or a SDXL-Inpainting
    model in future) and applies text-guided editing within the masked region.

    Model is cached in GPU memory across calls for fast repeated inference.
    """

    # Default SD inpainting model; can be swapped for SDXL-inpainting
    DEFAULT_MODEL_ID = "runwayml/stable-diffusion-inpainting"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        enable_xformers: bool = True,
    ):
        """
        Args:
            model_id: Hugging Face model ID for the inpainting pipeline.
            device: Compute device (auto-detected if None).
            enable_xformers: Use xformers memory-efficient attention if available.
        """
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device or get_device()
        self.dtype = get_dtype(self.device)
        self.enable_xformers = enable_xformers
        self._pipeline = None   # Lazy-loaded

    def _load_pipeline(self) -> None:
        """Load and cache the Stable Diffusion inpainting pipeline."""
        if self._pipeline is not None:
            return

        try:
            from diffusers import StableDiffusionInpaintPipeline

            logger.info(
                f"Loading SD Inpainting pipeline '{self.model_id}' "
                f"on {self.device} ({self.dtype})…"
            )
            t0 = time.perf_counter()

            self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self._pipeline = self._pipeline.to(self.device)

            # Load fast DPM Solver scheduler
            from diffusers import DPMSolverMultistepScheduler
            self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipeline.scheduler.config
            )
            logger.info("Loaded DPMSolverMultistepScheduler for object editor.")

            # Enable memory optimizations
            if self.enable_xformers and self.device.type == "cuda":
                try:
                    self._pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory-efficient attention enabled.")
                except Exception:
                    logger.warning("xformers not available; using default attention.")

            if self.device.type == "cuda":
                self._pipeline.enable_attention_slicing()
                self._pipeline.enable_vae_slicing()
                if hasattr(self._pipeline, "enable_vae_tiling"):
                    self._pipeline.enable_vae_tiling()
                
                # Only use CPU offload if low VRAM is detected (e.g. < 4.5 GB)
                gpu_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
                if gpu_mem < 4.5:
                    self._pipeline.enable_model_cpu_offload()
                    logger.info("Low VRAM detected; model CPU offload enabled.")
                else:
                    logger.info(f"Sufficient VRAM ({gpu_mem:.1f} GB); keeping model on GPU.")

            elapsed = time.perf_counter() - t0
            logger.info(f"SD Inpainting pipeline ready in {elapsed:.2f}s")

        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Run: pip install diffusers transformers"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SD Inpainting model '{self.model_id}': {exc}"
            ) from exc

    def _build_prompt(
        self,
        object_prompt: str,
        style_preset: str,
        extra_positive: str = "",
        extra_negative: str = "",
    ) -> tuple[str, str]:
        """
        Construct the final positive and negative prompts by injecting
        style-preset modifiers.

        Args:
            object_prompt: User-supplied object description.
            style_preset: Style preset name.
            extra_positive: Additional positive tokens.
            extra_negative: Additional negative tokens.

        Returns:
            (positive_prompt, negative_prompt) strings.
        """
        preset = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["Realistic"])
        preset_pos = preset["positive"]
        preset_neg = preset["negative"]

        # Append high-quality elements automatically to prompts
        quality_pos = (
            "ultra realistic, detailed, sharp focus, cinematic lighting, "
            "natural skin texture, realistic fabric texture, preserved identity, raw photo"
        )
        quality_neg = (
            "blurry, distorted, transparent, ghosting, artifacts, low quality, "
            "washed out, oversmoothed, extra limbs, deformed hands, plastic skin"
        )

        # Compose: object description first, then style, then quality, then extras
        parts_pos = [p for p in [object_prompt, preset_pos, quality_pos, extra_positive] if p.strip()]
        parts_neg = [p for p in [preset_neg, quality_neg, extra_negative] if p.strip()]

        positive = ", ".join(parts_pos)
        negative = ", ".join(parts_neg)
        return positive, negative

    def edit(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        object_prompt: str,
        style_preset: str = "Realistic",
        strength: float = 0.85,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        extra_negative_prompt: str = "",
        seed: Optional[int] = None,
        target_size: tuple[int, int] = (512, 512),
    ) -> EditingResult:
        """
        Apply SD inpainting to edit the masked region.

        Args:
            image_rgb: Source image (H x W x 3, uint8, RGB).
            mask: Binary mask, white=inpaint region (H x W, uint8, 0/255).
            object_prompt: Text description of the desired object.
            style_preset: One of the 8 STYLE_PRESETS keys.
            strength: Inpainting strength [0, 1]. Higher = more change.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Denoising steps (speed vs quality).
            extra_negative_prompt: Additional negative tokens.
            seed: Random seed for reproducible outputs.
            target_size: (width, height) to resize to before inference.

        Returns:
            EditingResult with the full edited image.
        """
        self._load_pipeline()

        positive, negative = self._build_prompt(
            object_prompt, style_preset, extra_negative=extra_negative_prompt
        )
        logger.info(
            f"SD Inpainting | style='{style_preset}' | "
            f"steps={num_inference_steps} | strength={strength}"
        )
        logger.debug(f"Positive prompt: {positive[:120]}…")

        # ── Prepare inputs ─────────────────────────────────────────────────────
        orig_h, orig_w = image_rgb.shape[:2]
        tw, th = target_size

        pil_image = numpy_to_pil(image_rgb).resize((tw, th), Image.LANCZOS)
        pil_mask = mask_to_pil(mask).resize((tw, th), Image.NEAREST)

        # ── Generator for reproducibility ──────────────────────────────────────
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # CPU diffusion is extremely slow; CUDA can afford higher-quality steps.
        if self.device.type == "cpu":
            num_inference_steps = min(num_inference_steps, 15)
        else:
            num_inference_steps = min(num_inference_steps, 35)

        # ── Run inference ──────────────────────────────────────────────────────
        t0 = time.perf_counter()

        with torch.inference_mode():
            autocast_ctx = (
                torch.autocast(self.device.type, dtype=self.dtype)
                if self.device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with autocast_ctx:
                output = self._pipeline(
                    prompt=positive,
                    negative_prompt=negative,
                    image=pil_image,
                    mask_image=pil_mask,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    height=th,
                    width=tw,
                )

        elapsed = time.perf_counter() - t0
        logger.info(f"SD Inpainting completed in {elapsed:.2f}s")

        # ── Post-process ───────────────────────────────────────────────────────
        edited_pil: Image.Image = output.images[0]
        # Resize back to original resolution
        edited_pil = edited_pil.resize((orig_w, orig_h), Image.LANCZOS)
        edited_np = pil_to_numpy(edited_pil)

        # Extract crop from the edited region using mask bounds
        ys, xs = np.where(mask > 127)
        if len(xs) > 0:
            x1, y1 = max(0, int(xs.min()) - 5), max(0, int(ys.min()) - 5)
            x2 = min(orig_w, int(xs.max()) + 5)
            y2 = min(orig_h, int(ys.max()) + 5)
            edited_crop = edited_np[y1:y2, x1:x2]
        else:
            edited_crop = edited_np

        clear_gpu_cache()

        return EditingResult(
            edited_image=edited_np,
            edited_crop=edited_crop,
            prompt_used=positive,
            style_preset=style_preset,
            inference_time_s=elapsed,
        )

    def get_style_presets(self) -> list[str]:
        """Return list of available style preset names."""
        return list(STYLE_PRESETS.keys())
