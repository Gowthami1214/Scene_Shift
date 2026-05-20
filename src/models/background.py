"""
Background Generation Module for SceneShift.

Generates contextual backgrounds using:
  1. Stable Diffusion text-to-image (primary)
  2. Procedural gradient generation (fallback — no GPU required)

The generated background is returned as an RGB NumPy array.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from src.utils.device import get_device, get_dtype, clear_gpu_cache
from src.utils.image_io import pil_to_numpy


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BackgroundResult:
    """Output from the background generation stage."""
    image: np.ndarray           # RGB background (H x W x 3, uint8)
    method: str                 # "stable_diffusion" | "procedural"
    prompt_used: str            # Prompt (or description for procedural)
    inference_time_s: float     # Generation time in seconds


# ── Background Generator ──────────────────────────────────────────────────────

class BackgroundGenerator:
    """
    Generates photorealistic backgrounds using Stable Diffusion text-to-image.

    Falls back to NumPy/OpenCV procedural gradients if SD is unavailable
    or the user chooses fast mode.

    Target: ~14 s on NVIDIA RTX 3080.
    """

    DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device or get_device()
        self.dtype = get_dtype(self.device)
        self._pipeline = None   # Lazy-loaded

    def _load_pipeline(self) -> None:
        """Load Stable Diffusion text-to-image pipeline (cached)."""
        if self._pipeline is not None:
            return

        try:
            from diffusers import StableDiffusionPipeline

            logger.info(
                f"Loading SD text-to-image pipeline '{self.model_id}' "
                f"on {self.device}…"
            )
            t0 = time.perf_counter()

            self._pipeline = StableDiffusionPipeline.from_pretrained(
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
            logger.info("Loaded DPMSolverMultistepScheduler for background generator.")

            if self.device.type == "cuda":
                try:
                    self._pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory-efficient attention enabled.")
                except Exception:
                    pass
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
            logger.info(f"SD text-to-image pipeline ready in {elapsed:.2f}s")

        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Run: pip install diffusers transformers"
            )
        except Exception as exc:
            logger.warning(
                f"Cannot load SD pipeline: {exc}. "
                "Falling back to procedural generation."
            )
            self._pipeline = "fallback"

    # ── Stable Diffusion generation ────────────────────────────────────────────

    def generate_sd(
        self,
        prompt: str,
        output_size: Tuple[int, int] = (512, 512),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        negative_prompt: str = (
            "people, person, face, text, watermark, low quality, blurry, artifact, "
            "flat lighting, distorted perspective, oversaturated, cartoon, painting"
        ),
    ) -> BackgroundResult:
        """
        Generate a background image from a text prompt using Stable Diffusion.

        Args:
            prompt: Text description of the desired background scene.
            output_size: (width, height) of the generated image.
            guidance_scale: CFG guidance scale.
            num_inference_steps: Denoising steps.
            seed: Random seed for reproducibility.
            negative_prompt: Things to suppress in the background.

        Returns:
            BackgroundResult with the generated RGB image.
        """
        self._load_pipeline()

        if self._pipeline == "fallback":
            logger.warning("SD unavailable — using procedural background.")
            return self.generate_procedural(prompt, output_size)

        tw, th = output_size
        # SD requires dimensions divisible by 8
        tw = (tw // 8) * 8
        th = (th // 8) * 8

        # Enrich the background prompt
        full_prompt = (
            f"{prompt}, "
            "cinematic establishing shot, photorealistic environment, rich natural colors, "
            "realistic perspective, soft golden-hour light, high detail, professional photography, "
            "shallow depth of field, no objects in foreground"
        )

        # CPU diffusion is extremely slow; CUDA can afford higher-quality steps.
        if self.device.type == "cpu":
            num_inference_steps = min(num_inference_steps, 15)
        else:
            num_inference_steps = min(num_inference_steps, 35)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        t0 = time.perf_counter()
        logger.info(f"Generating SD background: '{prompt[:80]}…'")

        with torch.inference_mode():
            autocast_ctx = (
                torch.autocast(self.device.type, dtype=self.dtype)
                if self.device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with autocast_ctx:
                output = self._pipeline(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    height=th,
                    width=tw,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )

        elapsed = time.perf_counter() - t0
        logger.info(f"Background generated in {elapsed:.2f}s")

        bg_pil: Image.Image = output.images[0]
        bg_np = pil_to_numpy(bg_pil)

        clear_gpu_cache()

        return BackgroundResult(
            image=bg_np,
            method="stable_diffusion",
            prompt_used=full_prompt,
            inference_time_s=elapsed,
        )

    # ── Procedural fallback ────────────────────────────────────────────────────

    def generate_procedural(
        self,
        prompt: str,
        output_size: Tuple[int, int] = (512, 512),
    ) -> BackgroundResult:
        """
        Generate a procedural gradient background using NumPy + OpenCV.

        Analyzes keywords in the prompt to select a color palette.

        Args:
            prompt: Text description (used for keyword-based palette selection).
            output_size: (width, height).

        Returns:
            BackgroundResult with the generated gradient RGB image.
        """
        t0 = time.perf_counter()
        tw, th = output_size

        # ── Keyword-based palette selection ────────────────────────────────────
        prompt_lower = prompt.lower()

        palette_map = {
            ("sunset", "dusk", "orange", "warm"): (
                (20, 40, 120), (220, 100, 50), (255, 180, 80)
            ),
            ("ocean", "sea", "beach", "underwater", "water"): (
                (0, 10, 80), (10, 80, 160), (100, 180, 220)
            ),
            ("forest", "jungle", "nature", "green"): (
                (10, 30, 10), (30, 80, 30), (80, 160, 60)
            ),
            ("night", "dark", "space", "galaxy", "stars"): (
                (5, 5, 20), (10, 10, 50), (30, 20, 80)
            ),
            ("city", "urban", "street"): (
                (20, 20, 30), (60, 60, 80), (100, 100, 130)
            ),
            ("sky", "cloud", "day", "blue"): (
                (100, 160, 230), (160, 200, 240), (220, 235, 255)
            ),
            ("cyberpunk", "neon", "futuristic"): (
                (10, 5, 30), (80, 0, 120), (0, 200, 180)
            ),
            ("desert", "sand"): (
                (180, 140, 60), (210, 180, 100), (240, 210, 140)
            ),
            ("snow", "winter", "ice", "arctic"): (
                (180, 200, 230), (210, 225, 245), (240, 248, 255)
            ),
        }

        top, mid, bot = (30, 60, 120), (120, 140, 180), (200, 220, 240)
        for keywords, colors in palette_map.items():
            if any(kw in prompt_lower for kw in keywords):
                top, mid, bot = colors
                break

        # ── Generate three-band gradient ───────────────────────────────────────
        bg = np.zeros((th, tw, 3), dtype=np.float32)

        for y in range(th):
            t = y / (th - 1) if th > 1 else 0.0
            if t < 0.5:
                # Top → Mid
                ratio = t * 2.0
                color = np.array(top, dtype=np.float32) * (1 - ratio) + \
                        np.array(mid, dtype=np.float32) * ratio
            else:
                # Mid → Bottom
                ratio = (t - 0.5) * 2.0
                color = np.array(mid, dtype=np.float32) * (1 - ratio) + \
                        np.array(bot, dtype=np.float32) * ratio
            bg[y, :] = color

        # ── Add subtle noise texture for realism ──────────────────────────────
        noise = np.random.normal(0, 4, bg.shape).astype(np.float32)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

        # ── Apply gentle Gaussian blur (atmospheric haze) ──────────────────────
        bg = cv2.GaussianBlur(bg, (15, 15), 5)

        elapsed = time.perf_counter() - t0
        logger.info(f"Procedural background generated in {elapsed:.3f}s")

        return BackgroundResult(
            image=bg,
            method="procedural",
            prompt_used=prompt,
            inference_time_s=elapsed,
        )

    def generate(
        self,
        prompt: str,
        output_size: Tuple[int, int] = (512, 512),
        use_sd: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
    ) -> BackgroundResult:
        """
        Generate background, preferring SD with procedural fallback.

        Args:
            prompt: Background scene description.
            output_size: (width, height).
            use_sd: Try Stable Diffusion first (falls back automatically).
            guidance_scale: Classifier-free guidance scale for SD generation.
            num_inference_steps: Denoising steps for SD generation.
            seed: Random seed.

        Returns:
            BackgroundResult.
        """
        if use_sd:
            try:
                result = self.generate_sd(
                    prompt,
                    output_size,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                )
                return self._apply_prompt_blur(result)
            except Exception as exc:
                logger.warning(f"SD generation failed ({exc}); using procedural.")
        return self._apply_prompt_blur(self.generate_procedural(prompt, output_size))

    def _apply_prompt_blur(self, result: BackgroundResult) -> BackgroundResult:
        """Apply shallow-depth background blur when requested by the prompt."""
        prompt_lower = result.prompt_used.lower()
        blur_terms = ("blurred background", "shallow depth", "depth of field", "bokeh")
        if not any(term in prompt_lower for term in blur_terms):
            return result

        blurred = cv2.GaussianBlur(result.image, (0, 0), sigmaX=3.0, sigmaY=3.0)
        return BackgroundResult(
            image=blurred,
            method=result.method,
            prompt_used=result.prompt_used,
            inference_time_s=result.inference_time_s,
        )
