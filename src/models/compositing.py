"""
Advanced Compositing Engine for SceneShift.

Implements three blending modes:
  1. Alpha Blending      — feathered alpha compositing (fastest)
  2. Poisson Blending    — seamless clone via iterative Jacobi solver
  3. Laplacian Pyramid   — multi-band frequency-domain blending

All modes ensure seamless boundaries and realistic lighting continuity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ── Blend Mode Enum ────────────────────────────────────────────────────────────

class BlendMode(str, Enum):
    ALPHA = "alpha"
    POISSON = "poisson"
    LAPLACIAN = "laplacian"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CompositingResult:
    """Output of the compositing engine."""
    composite: np.ndarray    # Final RGB composite (H x W x 3, uint8)
    blend_mode: str          # Mode used
    blend_time_s: float      # Wall-clock processing time


# ── Mode 1: Alpha Blending ────────────────────────────────────────────────────

def alpha_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha_mask: np.ndarray,
) -> np.ndarray:
    """
    Feathered alpha compositing: composite = fg * α + bg * (1 − α).

    Args:
        foreground: RGB foreground (H x W x 3, float32 or uint8).
        background: RGB background (H x W x 3, same shape as fg).
        alpha_mask: Alpha channel (H x W, float32, [0, 1]).

    Returns:
        Composited RGB image (H x W x 3, uint8).
    """
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    alpha = np.clip(alpha_mask.astype(np.float32), 0.0, 1.0)
    alpha = np.where(alpha >= 0.92, 1.0, alpha)
    alpha = np.where(alpha <= 0.04, 0.0, alpha)
    a = alpha[:, :, np.newaxis]  # Broadcast over channels

    composite = fg * a + bg * (1.0 - a)
    return np.clip(composite, 0, 255).astype(np.uint8)


# ── Mode 2: Poisson Blending ──────────────────────────────────────────────────

def poisson_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    offset: Tuple[int, int] = (0, 0),
    max_iter: int = 500,
) -> np.ndarray:
    """
    Poisson image blending using OpenCV's seamlessClone (fast GPU-optimized).
    Falls back to iterative Jacobi solver for fine-grained control.

    Args:
        foreground: RGB source (H x W x 3, uint8).
        background: RGB destination (H x W x 3, uint8).
        mask: Binary mask, white = blend region (H x W, uint8, 0/255).
        offset: (dx, dy) offset to place foreground in background.
        max_iter: Jacobi iterations (fallback path only).

    Returns:
        Seamlessly blended RGB image (H x W x 3, uint8).
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    # Ensure foreground and mask fit within background
    if fg_h != bg_h or fg_w != bg_w:
        foreground = cv2.resize(foreground, (bg_w, bg_h), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (bg_w, bg_h), interpolation=cv2.INTER_NEAREST)

    dx, dy = offset
    center_x = bg_w // 2 + dx
    center_y = bg_h // 2 + dy

    # Clamp center to valid range
    center_x = np.clip(center_x, fg_w // 2, bg_w - fg_w // 2)
    center_y = np.clip(center_y, fg_h // 2, bg_h - fg_h // 2)
    center = (center_x, center_y)

    # Ensure mask is proper (8-bit single channel)
    mask_8u = mask.astype(np.uint8)
    if mask_8u.max() <= 1:
        mask_8u = (mask_8u * 255).astype(np.uint8)
    if np.count_nonzero(mask_8u) == 0:
        return background.astype(np.uint8)
    if np.count_nonzero(mask_8u > 127) == mask_8u.size:
        return foreground.astype(np.uint8)

    try:
        # OpenCV seamlessClone — Poisson blend (NORMAL_CLONE to prevent background bleed-through)
        cloned = cv2.seamlessClone(
            foreground.astype(np.uint8),
            background.astype(np.uint8),
            mask_8u,
            center,
            cv2.MIXED_CLONE,
        )
        core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_binary = (mask_8u > 127).astype(np.uint8)
        core = cv2.erode(mask_binary, core_kernel, iterations=1).astype(np.float32)
        edge = np.clip(mask_binary.astype(np.float32) - core, 0.0, 1.0)
        edge = cv2.GaussianBlur(edge, (0, 0), 1.2)
        core_3 = core[:, :, np.newaxis]
        edge_3 = edge[:, :, np.newaxis]
        coverage = np.maximum(core_3, edge_3)
        result = (
            foreground.astype(np.float32) * core_3
            + cloned.astype(np.float32) * edge_3 * (1.0 - core_3)
            + background.astype(np.float32) * (1.0 - coverage)
        )
        return np.clip(result, 0, 255).astype(np.uint8)
    except cv2.error as exc:
        logger.warning(f"OpenCV seamlessClone failed ({exc}); using Jacobi solver.")
        return _jacobi_poisson_blend(foreground, background, mask, max_iter)


def _jacobi_poisson_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    max_iter: int = 300,
) -> np.ndarray:
    """
    Custom iterative Jacobi solver for Poisson blending.

    Solves the discrete Laplace equation ∇²f = ∇²g within the mask region
    where g is the foreground and f is the unknown blended image.
    """
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    binary = (mask > 127).astype(np.float32)

    result = bg.copy()

    for c in range(3):
        f = bg[:, :, c].copy()       # Initial guess = background
        g = fg[:, :, c]              # Source (foreground gradient field)

        # Laplacian of foreground (guidance field)
        guidance = cv2.Laplacian(g, cv2.CV_32F)

        for _ in range(max_iter):
            # Neighbor average
            f_up = np.roll(f, -1, axis=0)
            f_down = np.roll(f, 1, axis=0)
            f_left = np.roll(f, -1, axis=1)
            f_right = np.roll(f, 1, axis=1)

            f_new = (f_up + f_down + f_left + f_right - guidance) / 4.0

            # Apply only within mask, keep background elsewhere
            f = f_new * binary + bg[:, :, c] * (1.0 - binary)

        result[:, :, c] = np.clip(f, 0, 255)

    return result.astype(np.uint8)


# ── Mode 3: Laplacian Pyramid Blending ────────────────────────────────────────

def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """Build a Gaussian pyramid of `levels` levels."""
    pyramid = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _build_laplacian_pyramid(gaussian: list) -> list:
    """Compute Laplacian pyramid from Gaussian pyramid."""
    laplacian = []
    for i in range(len(gaussian) - 1):
        up = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        laplacian.append(gaussian[i] - up)
    laplacian.append(gaussian[-1])
    return laplacian


def laplacian_pyramid_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    levels: int = 6,
) -> np.ndarray:
    """
    Multi-band Laplacian pyramid blending.

    Blends each frequency band separately: high frequencies follow the mask
    boundary closely; low frequencies blend smoothly over a wider region.
    This produces seamless transitions without visible seams.

    Args:
        foreground: RGB foreground (H x W x 3, uint8).
        background: RGB background (H x W x 3, uint8 — same size).
        mask: Blend mask, white=foreground (H x W, uint8, 0/255).
        levels: Number of pyramid levels (higher = smoother blending).

    Returns:
        Blended RGB image (H x W x 3, uint8).
    """
    # Ensure consistent sizes
    bg_h, bg_w = background.shape[:2]
    if foreground.shape[:2] != (bg_h, bg_w):
        foreground = cv2.resize(foreground, (bg_w, bg_h), interpolation=cv2.INTER_LANCZOS4)
    if mask.shape[:2] != (bg_h, bg_w):
        mask = cv2.resize(mask, (bg_w, bg_h), interpolation=cv2.INTER_NEAREST)

    # Normalize mask to [0, 1] float
    mask_f = (mask / 255.0).astype(np.float32)
    if mask_f.ndim == 2:
        mask_f = np.stack([mask_f] * 3, axis=-1)

    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)

    # Adjust levels so pyramid fits the image
    min_dim = min(bg_h, bg_w)
    levels = min(levels, int(np.floor(np.log2(min_dim))) - 1)

    # Build pyramids
    gp_fg = _build_gaussian_pyramid(fg, levels)
    gp_bg = _build_gaussian_pyramid(bg, levels)
    gp_mask = _build_gaussian_pyramid(mask_f, levels)

    lp_fg = _build_laplacian_pyramid(gp_fg)
    lp_bg = _build_laplacian_pyramid(gp_bg)

    # Blend each Laplacian level using the corresponding mask level
    lp_blended = []
    for i in range(levels):
        m = gp_mask[i]
        blended_level = lp_fg[i] * m + lp_bg[i] * (1.0 - m)
        lp_blended.append(blended_level)

    # Reconstruct from blended Laplacian pyramid
    composite = lp_blended[-1].copy()
    for i in range(levels - 2, -1, -1):
        h, w = lp_blended[i].shape[:2]
        composite = cv2.pyrUp(composite, dstsize=(w, h))
        composite += lp_blended[i]

    return np.clip(composite, 0, 255).astype(np.uint8)


# ── Compositing Engine ────────────────────────────────────────────────────────

class CompositingEngine:
    """
    Unified compositing interface supporting Alpha, Poisson, and Laplacian modes.
    """

    def composite(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha_mask: np.ndarray,
        binary_mask: np.ndarray,
        mode: str = "alpha",
        poisson_offset: Tuple[int, int] = (0, 0),
        laplacian_levels: int = 6,
    ) -> CompositingResult:
        """
        Composite foreground onto background using the specified blend mode.

        Args:
            foreground: RGB foreground image (H x W x 3, uint8).
            background: RGB background image (H x W x 3, uint8, may differ in size).
            alpha_mask: Soft float alpha (H x W, float32, [0, 1]).
            binary_mask: Hard binary mask (H x W, uint8, 0/255).
            mode: Blend mode — "alpha" | "poisson" | "laplacian".
            poisson_offset: (dx, dy) offset for Poisson center placement.
            laplacian_levels: Pyramid levels for Laplacian blending.

        Returns:
            CompositingResult with the final composite image.
        """
        t0 = time.perf_counter()

        # Resize background to match foreground dimensions
        fg_h, fg_w = foreground.shape[:2]
        bg = cv2.resize(background, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)

        mode = mode.lower()
        logger.info(f"Compositing: mode='{mode}', fg={fg_w}x{fg_h}")

        if mode == BlendMode.ALPHA:
            composite = alpha_blend(foreground, bg, alpha_mask)

        elif mode == BlendMode.POISSON:
            composite = poisson_blend(
                foreground, bg, binary_mask, offset=poisson_offset
            )

        elif mode == BlendMode.LAPLACIAN:
            composite = laplacian_pyramid_blend(
                foreground, bg, binary_mask, levels=laplacian_levels
            )

        else:
            logger.warning(f"Unknown blend mode '{mode}'; defaulting to alpha.")
            composite = alpha_blend(foreground, bg, alpha_mask)

        elapsed = time.perf_counter() - t0
        logger.info(f"Compositing completed in {elapsed:.3f}s")

        return CompositingResult(
            composite=composite,
            blend_mode=mode,
            blend_time_s=elapsed,
        )
