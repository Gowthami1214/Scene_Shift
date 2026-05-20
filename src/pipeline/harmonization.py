"""
LAB Color Harmonization Module for SceneShift.

Ensures the composited foreground matches the ambient lighting, color tone,
and contrast of the background scene.

Pipeline:
  1. LAB Luminance Equalization  — match foreground L* to background L*
  2. Contrast Harmonization      — adjust standard deviation in luminance
  3. Chromatic Adaptation        — transfer a* and b* channels for color match
  4. Edge-Only Unsharp Masking   — sharpen structural edges without halos
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class HarmonizationResult:
    """Output of the LAB color harmonization stage."""
    harmonized: np.ndarray     # Full harmonized RGB image (H x W x 3, uint8)
    fg_harmonized: np.ndarray  # Foreground-only harmonized patch
    harmonization_time_s: float


# ── LAB Statistics Transfer ───────────────────────────────────────────────────

def _lab_statistics(lab_img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute per-channel statistics (mean, std) within a masked region.

    Args:
        lab_img: LAB image (H x W x 3, float32).
        mask: Binary mask (H x W, float32 or uint8); 1 = include pixel.

    Returns:
        dict with 'mean' and 'std' arrays (shape [3]).
    """
    binary = (mask > 0.5).astype(bool)
    means = []
    stds = []
    for c in range(3):
        channel = lab_img[:, :, c][binary]
        means.append(float(channel.mean()) if len(channel) > 0 else 128.0)
        stds.append(float(channel.std()) if len(channel) > 0 else 1.0)
    return {"mean": np.array(means), "std": np.array(stds)}


def is_skin_tone(image_rgb: np.ndarray) -> np.ndarray:
    """
    Detect human skin tones in YCrCb color space.
    Returns a soft mask (H x W, float32, [0, 1]).
    """
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    # Standard human skin color range in YCrCb space
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    # Soften edges
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask_blurred.astype(np.float32) / 255.0


def lab_color_transfer(
    source_lab: np.ndarray,
    target_stats: dict,
    source_stats: dict,
    mask: np.ndarray,
    transfer_channels: tuple = (0, 1, 2),   # L*, a*, b*
    blend_factor: float = 0.8,
    skin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Transfer LAB color statistics from target to source.
    Applies boundary weighting and skin tone protection.
    """
    result = source_lab.copy()
    binary = mask.astype(bool) if mask.ndim == 2 else mask[:, :, 0].astype(bool)

    # Compute boundary weighting: stronger transfer near the edge, softer inside
    mask_u8 = (mask > 0.5).astype(np.uint8)
    dist_inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_dist = 40.0
    boundary_weight = 1.0 - np.clip(dist_inside / max_dist, 0.0, 1.0)

    for c in transfer_channels:
        μ_src = source_stats["mean"][c]
        σ_src = max(source_stats["std"][c], 1e-6)
        μ_tgt = target_stats["mean"][c]
        σ_tgt = max(target_stats["std"][c], 1e-6)

        ch = source_lab[:, :, c].copy()
        normalized = (ch - μ_src) / σ_src
        transferred = normalized * σ_tgt + μ_tgt

        # Base blend factor
        eff_blend = blend_factor * (0.4 + 0.6 * boundary_weight)

        # Protect skin tones on chromatic channels (a*, b*)
        if skin_mask is not None and c in (1, 2):
            eff_blend = eff_blend * (1.0 - skin_mask * 0.8)

        blended = ch * (1.0 - eff_blend) + transferred * eff_blend
        result[:, :, c] = np.where(binary, blended, ch)

    return result


# ── Luminance Equalization ────────────────────────────────────────────────────

def luminance_equalize(
    foreground_lab: np.ndarray,
    background_lab: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    blend_factor: float = 0.75,
) -> np.ndarray:
    """
    Match foreground luminance (L* channel) to background ambient luminance.

    Args:
        foreground_lab: Full image in LAB (H x W x 3, float32).
        background_lab: Background in LAB (H x W x 3, float32).
        fg_mask: Foreground mask (H x W, float32, [0,1]).
        bg_mask: Background mask (complement of fg; H x W, float32, [0,1]).
        blend_factor: Transfer strength.

    Returns:
        LAB image with equalized luminance.
    """
    fg_stats = _lab_statistics(foreground_lab, fg_mask)
    bg_stats = _lab_statistics(background_lab, bg_mask)

    # Transfer only the L* channel (luminance equalization)
    result = lab_color_transfer(
        foreground_lab, bg_stats, fg_stats, fg_mask,
        transfer_channels=(0,),  # L* only
        blend_factor=blend_factor,
    )
    return result


# ── Contrast Harmonization ────────────────────────────────────────────────────

def contrast_harmonize(
    lab_image: np.ndarray,
    mask: np.ndarray,
    target_contrast: float = 0.85,
    clahe_clip: float = 2.0,
    clahe_grid: tuple = (8, 8),
) -> np.ndarray:
    """
    Harmonize local contrast using CLAHE on the L* channel.

    Args:
        lab_image: LAB image (H x W x 3, float32).
        mask: Region to process (H x W, float32, [0,1]).
        target_contrast: Target contrast scaling [0.5, 1.5].
        clahe_clip: CLAHE clip limit.
        clahe_grid: CLAHE tile grid size.

    Returns:
        Contrast-harmonized LAB image.
    """
    result = lab_image.copy()

    # Extract L channel (scale to [0, 255] for CLAHE)
    l_channel = np.clip(result[:, :, 0], 0, 100) / 100.0 * 255
    l_uint8 = l_channel.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    l_clahe = clahe.apply(l_uint8)

    # Blend original and CLAHE-enhanced L channel
    l_original = result[:, :, 0]
    l_enhanced = l_clahe.astype(np.float32) / 255.0 * 100.0

    binary = (mask > 0.5).astype(np.float32)
    result[:, :, 0] = (
        l_original * (1.0 - binary * (1 - target_contrast)) +
        l_enhanced * binary * (1 - target_contrast)
    )

    return result


# ── Edge-Only Unsharp Masking ─────────────────────────────────────────────────

def edge_unsharp_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    sigma: float = 1.5,
    strength: float = 0.4,
    edge_threshold: int = 30,
) -> np.ndarray:
    """
    Apply unsharp masking only along structural edges (not in flat regions).

    This sharpens object details without amplifying noise in smooth areas.

    Args:
        image_rgb: Input RGB image (H x W x 3, uint8).
        mask: Foreground region mask (H x W, float32, [0,1]).
        sigma: Gaussian blur sigma for unsharp base.
        strength: Sharpening strength [0, 1].
        edge_threshold: Canny threshold for edge detection.

    Returns:
        Sharpened RGB image (H x W x 3, uint8).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
    edge_mask = (edges / 255.0).astype(np.float32)

    # Apply only at edges within the foreground mask
    combined_mask = edge_mask * mask
    if combined_mask.ndim == 2:
        combined_mask = combined_mask[:, :, np.newaxis]

    # Gaussian blur for unsharp base
    blurred = cv2.GaussianBlur(
        image_rgb.astype(np.float32),
        (0, 0),
        sigma,
    )

    # Unsharp: sharpened = original + strength * (original - blurred)
    sharpened = image_rgb.astype(np.float32) + strength * (
        image_rgb.astype(np.float32) - blurred
    )

    # Blend: sharp at edges, original everywhere else
    result = (
        sharpened * combined_mask +
        image_rgb.astype(np.float32) * (1.0 - combined_mask)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Color Harmonizer ──────────────────────────────────────────────────────────

class ColorHarmonizer:
    """
    Full LAB color harmonization pipeline for SceneShift.

    Ensures the composited foreground integrates seamlessly with the background
    in terms of luminance, contrast, and chromatic color.
    """

    def harmonize(
        self,
        composite: np.ndarray,
        background: np.ndarray,
        alpha_mask: np.ndarray,
        luminance_blend: float = 0.45,
        chroma_blend: float = 0.16,
        sharpen_strength: float = 0.25,
        apply_clahe: bool = True,
    ) -> HarmonizationResult:
        """
        Full harmonization pipeline: luminance → contrast → chroma → sharpen.

        Args:
            composite: Full composited RGB image (H x W x 3, uint8).
            background: Background-only RGB image (H x W x 3, uint8).
            alpha_mask: Foreground alpha mask (H x W, float32, [0,1]).
            luminance_blend: L* transfer strength.
            chroma_blend: a*b* transfer strength.
            sharpen_strength: Edge unsharp strength.
            apply_clahe: Whether to apply CLAHE contrast harmonization.

        Returns:
            HarmonizationResult with harmonized image.
        """
        t0 = time.perf_counter()

        h, w = composite.shape[:2]

        # Resize background to match composite if needed
        bg = cv2.resize(background, (w, h), interpolation=cv2.INTER_LANCZOS4) \
            if background.shape[:2] != (h, w) else background

        # ── Convert to LAB ─────────────────────────────────────────────────────
        composite_lab = cv2.cvtColor(composite, cv2.COLOR_RGB2Lab).astype(np.float32)
        bg_lab = cv2.cvtColor(bg, cv2.COLOR_RGB2Lab).astype(np.float32)

        fg_mask = np.clip(alpha_mask, 0, 1)
        bg_mask = 1.0 - fg_mask

        # ── Stage 1: Luminance Equalization ────────────────────────────────────
        # Skin mask detection for chromatic channels protection
        skin_mask = is_skin_tone(composite)

        # Luminance statistics before adaptation
        fg_stats = _lab_statistics(composite_lab, fg_mask)
        bg_stats = _lab_statistics(bg_lab, bg_mask)

        result_lab = lab_color_transfer(
            composite_lab, bg_stats, fg_stats, fg_mask,
            transfer_channels=(0,),  # L* only
            blend_factor=luminance_blend,
        )

        # ── Stage 2: Contrast Harmonization ───────────────────────────────────
        if apply_clahe:
            result_lab = contrast_harmonize(result_lab, fg_mask)

        # ── Stage 3: Chromatic Adaptation (a*, b* channels) ──────────────────
        if chroma_blend > 0:
            fg_stats = _lab_statistics(result_lab, fg_mask)
            bg_stats = _lab_statistics(bg_lab, bg_mask)
            result_lab = lab_color_transfer(
                result_lab, bg_stats, fg_stats, fg_mask,
                transfer_channels=(1, 2),  # a* and b* only
                blend_factor=chroma_blend,
                skin_mask=skin_mask,
            )

        # ── Convert back to RGB ───────────────────────────────────────────────
        result_lab_uint8 = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab_uint8, cv2.COLOR_Lab2RGB)

        # ── Stage 4: Edge-Only Unsharp Masking ────────────────────────────────
        if sharpen_strength > 0:
            result_rgb = edge_unsharp_mask(
                result_rgb, fg_mask, strength=sharpen_strength
            )

        elapsed = time.perf_counter() - t0
        logger.info(f"Color harmonization completed in {elapsed:.3f}s")

        # Extract foreground patch
        ys, xs = np.where(fg_mask > 0.5)
        if len(xs) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            fg_patch = result_rgb[y1:y2, x1:x2]
        else:
            fg_patch = result_rgb

        return HarmonizationResult(
            harmonized=result_rgb,
            fg_harmonized=fg_patch,
            harmonization_time_s=elapsed,
        )
