"""
Real-Photo Preprocessing Module for SceneShift.

Detects and fixes common issues in mobile/DSLR photographs before they
enter the compositing pipeline:
  1. JPEG compression artifact removal  — bilateral filtering
  2. Exposure normalization             — CLAHE on L* channel
  3. White balance correction           — gray-world assumption
  4. Noise reduction                    — adaptive Non-Local Means
  5. Minimum resolution enforcement     — Lanczos upscale
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List

import cv2
import numpy as np
from loguru import logger


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PreprocessingResult:
    """Output of the photo preprocessing stage."""
    image: np.ndarray           # Preprocessed RGB (H x W x 3, uint8)
    original_size: tuple        # (H, W) before any resize
    applied_corrections: list   # Names of corrections that fired
    preprocessing_time_s: float


# ── Thresholds ────────────────────────────────────────────────────────────────

_NOISE_THRESHOLD = 100.0        # Laplacian variance below this → noisy
_EXPOSURE_UNDER_THRESHOLD = 60  # L* channel mean below this → underexposed
_EXPOSURE_OVER_THRESHOLD = 190  # L* channel mean above this → overexposed
_ARTIFACT_NOISE_THRESHOLD = 80.0  # Apply artifact removal below this noise level


# ── PhotoPreprocessor ─────────────────────────────────────────────────────────

class PhotoPreprocessor:
    """Automatic real-photo preprocessing pipeline.

    Detects and fixes common issues in mobile/DSLR photographs:
    - JPEG compression artifacts
    - Poor exposure (under/over-exposed)
    - Incorrect white balance
    - High ISO noise
    - Low resolution
    """

    def preprocess(
        self,
        image_rgb: np.ndarray,
        enable_denoise: bool = True,
        enable_exposure: bool = True,
        enable_white_balance: bool = True,
        enable_artifact_removal: bool = True,
        min_resolution: int = 512,
    ) -> PreprocessingResult:
        """Run full preprocessing pipeline with auto-detection.

        Each correction is auto-detected: it only fires if the image
        actually needs it.  The order of operations is chosen to avoid
        cascading artifacts (denoise before exposure, etc.).

        Args:
            image_rgb: Input RGB image (H x W x 3, uint8).
            enable_denoise: Allow adaptive noise reduction.
            enable_exposure: Allow CLAHE exposure normalization.
            enable_white_balance: Allow gray-world white balance.
            enable_artifact_removal: Allow bilateral JPEG artifact removal.
            min_resolution: Shortest side must be at least this many pixels.

        Returns:
            PreprocessingResult with the corrected image and metadata.
        """
        t0 = time.perf_counter()
        applied: List[str] = []

        h, w = image_rgb.shape[:2]
        original_size = (h, w)
        result = image_rgb.copy()

        # ── Detect noise level (used by several stages) ───────────────────
        noise_level = self._detect_noise_level(result)
        logger.debug(f"Detected noise level (Laplacian var): {noise_level:.1f}")

        # ── Stage 1: JPEG artifact removal ────────────────────────────────
        if enable_artifact_removal and noise_level < _ARTIFACT_NOISE_THRESHOLD:
            result = self._remove_jpeg_artifacts(result)
            applied.append("jpeg_artifact_removal")
            logger.info("Applied JPEG artifact removal (bilateral filter)")

        # ── Stage 2: Exposure normalization ───────────────────────────────
        if enable_exposure:
            exposure = self._detect_exposure(result)
            if exposure != "normal":
                result = self._normalize_exposure(result)
                applied.append(f"exposure_normalization ({exposure})")
                logger.info(f"Applied exposure normalization — detected {exposure}exposed")

        # ── Stage 3: White balance correction ─────────────────────────────
        if enable_white_balance:
            needs_wb = self._needs_white_balance(result)
            if needs_wb:
                result = self._correct_white_balance(result)
                applied.append("white_balance_correction")
                logger.info("Applied gray-world white balance correction")

        # ── Stage 4: Noise reduction ──────────────────────────────────────
        if enable_denoise and noise_level < _NOISE_THRESHOLD:
            result = self._reduce_noise(result, noise_level)
            applied.append("noise_reduction")
            logger.info(f"Applied adaptive noise reduction (noise={noise_level:.1f})")

        # ── Stage 5: Minimum resolution enforcement ──────────────────────
        result = self._ensure_min_resolution(result, min_resolution)
        if result.shape[:2] != (h, w):
            applied.append("resolution_upscale")
            logger.info(
                f"Upscaled from {w}x{h} → "
                f"{result.shape[1]}x{result.shape[0]} (min_resolution={min_resolution})"
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Preprocessing completed in {elapsed:.3f}s — "
            f"{len(applied)} correction(s) applied"
        )

        return PreprocessingResult(
            image=result,
            original_size=original_size,
            applied_corrections=applied,
            preprocessing_time_s=elapsed,
        )

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _detect_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance.

        A sharp, clean image has a high Laplacian variance.  A noisy or
        blurry image has a low variance (typically < 100).

        Args:
            image: RGB image (H x W x 3, uint8).

        Returns:
            Laplacian variance (higher = cleaner).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _detect_exposure(self, image: np.ndarray) -> str:
        """Classify exposure as 'under', 'over', or 'normal'.

        Uses the mean of the L* channel in CIE-LAB space.

        Args:
            image: RGB image (H x W x 3, uint8).

        Returns:
            One of ``'under'``, ``'over'``, or ``'normal'``.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l_mean = float(lab[:, :, 0].mean())
        if l_mean < _EXPOSURE_UNDER_THRESHOLD:
            return "under"
        if l_mean > _EXPOSURE_OVER_THRESHOLD:
            return "over"
        return "normal"

    def _needs_white_balance(self, image: np.ndarray) -> bool:
        """Check whether the image has a visible color cast.

        The gray-world assumption says that the average color of a
        well-balanced image is neutral gray.  If any channel deviates
        significantly from the overall mean, a correction is needed.

        Args:
            image: RGB image (H x W x 3, uint8).

        Returns:
            ``True`` if a correction is warranted.
        """
        means = image.mean(axis=(0, 1))  # Per-channel means (R, G, B)
        avg_gray = means.mean()
        deviation = np.abs(means - avg_gray).max()
        # Threshold: > 8 intensity units of deviation signals a color cast
        return float(deviation) > 8.0

    # ── Correction methods ────────────────────────────────────────────────────

    def _remove_jpeg_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Remove JPEG blocking artifacts with a bilateral filter.

        The bilateral filter smooths flat regions (where blocking is
        visible) while preserving strong edges.

        Args:
            image: RGB image (H x W x 3, uint8).

        Returns:
            Filtered image with same shape and dtype.
        """
        return cv2.bilateralFilter(image, 5, 50, 50)

    def _normalize_exposure(self, image: np.ndarray) -> np.ndarray:
        """Normalize exposure using CLAHE on the L* channel in LAB space.

        CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances
        local contrast without over-amplifying noise.

        Args:
            image: RGB image (H x W x 3, uint8).

        Returns:
            Exposure-corrected RGB image (uint8).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_corrected = clahe.apply(l_channel)
        lab_corrected = cv2.merge([l_corrected, a_channel, b_channel])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2RGB)

    def _correct_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Auto white balance using the gray-world assumption.

        Each channel is scaled so that its mean equals the overall
        luminance mean.  Values are clipped to [0, 255].

        Args:
            image: RGB image (H x W x 3, uint8).

        Returns:
            White-balanced RGB image (uint8).
        """
        result = image.astype(np.float32)
        means = result.mean(axis=(0, 1))  # [R_mean, G_mean, B_mean]
        avg_gray = means.mean()

        for c in range(3):
            if means[c] > 0:
                result[:, :, c] *= avg_gray / means[c]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _reduce_noise(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """Adaptive noise reduction using Non-Local Means Denoising.

        The filter strength (``h``) is proportional to the detected noise
        level so that clean images get minimal filtering.

        Args:
            image: RGB image (H x W x 3, uint8).
            noise_level: Laplacian variance estimate (lower = noisier).

        Returns:
            Denoised RGB image (uint8).
        """
        # Inverse relationship: lower noise_level → stronger denoising
        h = max(3.0, (_NOISE_THRESHOLD - noise_level) * 0.1)
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h,
            h,
            7,
            21,
        )

    def _ensure_min_resolution(self, image: np.ndarray, min_size: int) -> np.ndarray:
        """Upscale if the shortest side is below *min_size* using Lanczos.

        Args:
            image: RGB image (H x W x 3, uint8).
            min_size: Minimum number of pixels for the shortest side.

        Returns:
            Possibly upscaled image (uint8).
        """
        h, w = image.shape[:2]
        shortest = min(h, w)
        if shortest >= min_size:
            return image

        scale = min_size / shortest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
