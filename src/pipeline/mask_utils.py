"""
Mask Refinement Pipeline for SceneShift.

Implements a three-stage mask refinement process:
  1. Morphological Denoising  — removes small artifacts, fills holes
  2. Edge Alignment           — Sobel gradient + Canny-based boundary correction
  3. Distance-Transform Feathering — smooth sub-pixel alpha boundaries

Output: a refined alpha mask (H x W, float32, [0, 1]) with smooth edges.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from loguru import logger


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MaskRefinementResult:
    """Output of the mask refinement pipeline."""
    binary_mask: np.ndarray      # Hard binary mask (H x W, uint8, 0/255)
    alpha_mask: np.ndarray       # Soft alpha mask (H x W, float32, [0, 1])
    edge_map: np.ndarray         # Edge detection map (H x W, uint8)
    refinement_time_s: float     # Processing time in seconds


# ── Stage 1: Morphological Denoising ─────────────────────────────────────────

def morphological_denoise(
    mask: np.ndarray,
    min_area: int = 500,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Remove small disconnected objects and fill interior holes.

    Args:
        mask: Binary mask (H x W, uint8, 0 or 255).
        min_area: Minimum contour area to retain (pixels).
        kernel_size: Morphological kernel size.

    Returns:
        Denoised binary mask (H x W, uint8, 0/255).
    """
    # Ensure binary
    binary = (mask > 127).astype(np.uint8)

    # ── Remove small objects via connected components ──────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):          # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 1

    # ── Fill holes (binary complement → flood fill → complement) ──────────────
    # Scipy's binary_fill_holes is more robust than OpenCV flood fill
    filled = ndimage.binary_fill_holes(cleaned).astype(np.uint8)

    # ── Morphological closing to smooth jagged edges ──────────────────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, small_kernel, iterations=1)

    return (opened * 255).astype(np.uint8)


# ── Stage 2: Edge Alignment via Guided Filtering ──────────────────────────────

def guided_filter_gray(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    r: int = 4,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Fast edge-preserving Guided Filter using a grayscale guidance image.
    Snaps raw mask boundaries to high-contrast edges in the source image.

    Args:
        image_rgb: Source RGB image (H x W x 3, uint8) to guide alignment.
        mask: Input binary mask (H x W, uint8, 0/255).
        r: Filtering window radius.
        eps: Regularization parameter.

    Returns:
        Filtered soft mask (H x W, float32, [0, 1]).
    """
    I_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p_float = mask.astype(np.float32) / 255.0

    wsize = (2 * r + 1, 2 * r + 1)

    def box_filter(img):
        return cv2.boxFilter(img, -1, wsize, borderType=cv2.BORDER_REFLECT)

    mean_I = box_filter(I_gray)
    mean_p = box_filter(p_float)
    mean_Ip = box_filter(I_gray * p_float)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I_gray * I_gray)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a)
    mean_b = box_filter(b)

    q = mean_a * I_gray + mean_b
    return np.clip(q, 0.0, 1.0)


def edge_align_mask(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    dilation_px: int = 3,
    canny_low: int = 30,
    canny_high: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align mask boundaries using a combination of Guided Filter and Sobel/Canny edges.
    """
    # Produce the edge map for debugging/status reporting
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    canny_edges = cv2.Canny(blurred, canny_low, canny_high)
    combined = cv2.addWeighted(gradient_mag, 0.4, canny_edges, 0.6, 0)
    edge_map = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Snapshot Snap to Guided Filter boundaries
    guided = guided_filter_gray(image_rgb, mask, r=3, eps=5e-4)
    
    # Convert guided output back to binary
    aligned_mask = (guided > 0.42).astype(np.uint8) * 255
    aligned_mask = cv2.morphologyEx(
        aligned_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    return aligned_mask, edge_map


# ── Stage 3: Boundary-Only Hermite Feathering ─────────────────────────────────

def feather_mask(
    mask: np.ndarray,
    feather_radius: int = 3,
) -> np.ndarray:
    """
    Apply Hermite smoothstep feathering strictly to a narrow border band of the mask.
    Guarantees the core body is 100% solid to prevent transparency and bleed-through.

    Args:
        mask: Binary mask (H x W, uint8, 0/255).
        feather_radius: Transition band radius in pixels.

    Returns:
        Alpha mask (H x W, float32, [0, 1]).
    """
    binary = (mask > 127).astype(np.uint8)
    if feather_radius <= 0:
        return binary.astype(np.float32)

    # Distance transform inside and outside the mask
    dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)

    # Signed distance (positive inside, negative outside)
    signed_dist = dist_inside.astype(np.float32) - dist_outside.astype(np.float32)

    # Normalize signed distance to [-1, 1] band
    norm_dist = np.clip(signed_dist / float(feather_radius), -1.0, 1.0)

    # Shift/Scale to [0, 1]
    t = 0.5 * (norm_dist + 1.0)

    # Hermite smoothstep interpolation: 3t^2 - 2t^3
    alpha = 3 * (t ** 2) - 2 * (t ** 3)

    alpha[dist_inside >= feather_radius] = 1.0
    alpha[dist_outside >= feather_radius] = 0.0

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def refine_mask(
    raw_mask: np.ndarray,
    image_rgb: Optional[np.ndarray] = None,
    min_area: int = 500,
    morph_kernel: int = 5,
    feather_radius: int = 3,
    use_edge_alignment: bool = True,
) -> MaskRefinementResult:
    """
    Full three-stage mask refinement pipeline.
    """
    t0 = time.perf_counter()

    # Ensure uint8 binary input
    if raw_mask.dtype != np.uint8:
        raw_mask = (raw_mask > 0).astype(np.uint8) * 255
    if raw_mask.max() <= 1:
        raw_mask = (raw_mask * 255).astype(np.uint8)

    logger.debug(f"Mask refinement: input shape {raw_mask.shape}, "
                 f"nonzero={np.count_nonzero(raw_mask)}")

    # ── Stage 1: Morphological Denoising ──────────────────────────────────────
    denoised = morphological_denoise(raw_mask, min_area=min_area, kernel_size=morph_kernel)

    # ── Stage 2: Edge Alignment ────────────────────────────────────────────────
    if use_edge_alignment and image_rgb is not None:
        aligned, edge_map = edge_align_mask(denoised, image_rgb)
    else:
        aligned = denoised
        edge_map = np.zeros(raw_mask.shape[:2], dtype=np.uint8)

    # ── Stage 3: Hermite Feathering ────────────────────────────────────────────
    alpha = feather_mask(aligned, feather_radius=feather_radius)

    elapsed = time.perf_counter() - t0
    logger.info(f"Mask refinement completed in {elapsed:.3f}s")

    return MaskRefinementResult(
        binary_mask=aligned,
        alpha_mask=alpha,
        edge_map=edge_map,
        refinement_time_s=elapsed,
    )


def apply_alpha_composite(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Apply alpha compositing: out = fg * alpha + bg * (1 - alpha).

    Args:
        foreground: RGB foreground (H x W x 3, uint8 or float32).
        background: RGB background (H x W x 3, same dtype).
        alpha: Alpha channel (H x W, float32, [0, 1]).

    Returns:
        Composited RGB image (H x W x 3, uint8).
    """
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    alpha = np.where(alpha >= 0.92, 1.0, alpha)
    alpha = np.where(alpha <= 0.04, 0.0, alpha)
    a = alpha[:, :, np.newaxis]  # (H x W x 1) for broadcasting

    composite = fg * a + bg * (1.0 - a)
    return np.clip(composite, 0, 255).astype(np.uint8)
