"""
Shadow Synthesis Module for SceneShift.

Generates physically plausible soft shadows beneath composited objects.
Supports configurable light direction, blur radius, and opacity.

Shadow model: Perspective projection of the object mask with
Gaussian blur (penumbra), cast along the specified light direction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from loguru import logger


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ShadowResult:
    """Output of the shadow synthesis stage."""
    image_with_shadow: np.ndarray  # RGB with shadow (H x W x 3, uint8)
    shadow_layer: np.ndarray       # Shadow-only RGBA layer (H x W x 4, uint8)
    synthesis_time_s: float


# ── Shadow Synthesizer ────────────────────────────────────────────────────────

class ShadowSynthesizer:
    """
    Generates and composites physically plausible cast shadows.

    Shadow rendering model:
        1. Project the binary mask along the light direction vector
           (affine shear transform)
        2. Apply progressive Gaussian blur (penumbra simulation)
        3. Apply an opacity falloff with distance from the object base
        4. Composite the shadow beneath the foreground object

    This models directional ambient occlusion / contact shadow correctly.
    """

    def synthesize(
        self,
        composite: np.ndarray,
        binary_mask: np.ndarray,
        light_direction: Tuple[float, float] = (1.0, 0.5),
        shadow_length: float = 0.15,
        blur_radius: int = 25,
        opacity: float = 0.55,
        shadow_color: Tuple[int, int, int] = (10, 10, 15),
        contact_tightness: float = 0.7,
    ) -> ShadowResult:
        """
        Synthesize and composite a directional shadow.

        Args:
            composite: RGB composited image (H x W x 3, uint8).
            binary_mask: Binary mask of the foreground object (H x W, uint8).
            light_direction: (dx, dy) normalized light direction vector.
                             Positive x = light from left (shadow to right).
                             Positive y = light from top (shadow down).
            shadow_length: Shadow stretch factor as fraction of image height.
            blur_radius: Gaussian blur kernel size for penumbra softness.
                         Larger = softer/wider shadow.
            opacity: Shadow maximum opacity [0, 1].
            shadow_color: RGB color of the shadow (dark tint).
            contact_tightness: [0, 1] controls contact shadow vs. penumbra ratio.
                                1.0 = tight contact shadow.

        Returns:
            ShadowResult with the shadow-composited image.
        """
        t0 = time.perf_counter()

        h, w = composite.shape[:2]

        # ── Step 1: Directional Cast Shadow ────────────────────────────────────
        # Normalize and scale light direction
        dx, dy = light_direction
        mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        dx, dy = dx / mag, dy / mag

        shadow_shift_x = int(dx * w * shadow_length)
        shadow_shift_y = int(dy * h * shadow_length)

        # Affine transform matrix: shear + translate the mask
        # This creates a perspective-like shadow cast effect
        src_pts = np.float32([[0, 0], [w, 0], [0, h]])
        dst_pts = np.float32([
            [shadow_shift_x // 2, shadow_shift_y // 2],
            [w + shadow_shift_x // 2, shadow_shift_y // 2],
            [shadow_shift_x, h + shadow_shift_y],
        ])
        M = cv2.getAffineTransform(src_pts, dst_pts)

        shadow_mask_float = (binary_mask / 255.0).astype(np.float32)
        shadow_projected = cv2.warpAffine(
            shadow_mask_float, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Multi-scale Gaussian blur (penumbra)
        if blur_radius > 0:
            blur_k = max(3, (blur_radius * 2 + 1))
            if blur_k % 2 == 0:
                blur_k += 1

            shadow_blurred = cv2.GaussianBlur(shadow_projected, (blur_k, blur_k), blur_radius / 3)
            shadow_combined = (
                shadow_projected * contact_tightness
                + shadow_blurred * (1.0 - contact_tightness)
            )
        else:
            shadow_combined = shadow_projected

        shadow_combined = np.clip(shadow_combined, 0.0, 1.0)

        # Distance-based opacity falloff
        binary = (binary_mask > 127).astype(np.uint8)
        dist_from_obj = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        dist_norm = np.clip(dist_from_obj / (h * shadow_length + 1), 0, 1)
        falloff = 1.0 - dist_norm ** 0.5
        falloff = np.clip(falloff, 0, 1)

        shadow_alpha_cast = shadow_combined * opacity * falloff

        # ── Step 2: Tight Contact Shadow ───────────────────────────────────────
        # Shift mask down by a few pixels for contact points (e.g. feet, base)
        contact_shift_y = max(1, int(h * 0.004))
        contact_mask = np.roll(shadow_mask_float, contact_shift_y, axis=0)
        contact_mask[:contact_shift_y, :] = 0.0

        # Blur contact mask slightly for tight contact shadow
        contact_blur_k = 5
        contact_blurred = cv2.GaussianBlur(contact_mask, (contact_blur_k, contact_blur_k), 1.2)

        # Contact shadow has high opacity but falls off extremely rapidly
        contact_falloff = 1.0 - np.clip(dist_from_obj / 15.0, 0.0, 1.0)
        shadow_alpha_contact = contact_blurred * 0.85 * contact_falloff

        # ── Step 3: Combine and Mask out Foreground ────────────────────────────
        # Combine directional cast shadow and tight contact shadow
        shadow_alpha = np.maximum(shadow_alpha_cast, shadow_alpha_contact)

        # IMPORTANT: Mask out the foreground subject to prevent shadow bleeding on top of the subject
        fg_mask_float = (binary_mask > 127).astype(np.float32)
        shadow_alpha = shadow_alpha * (1.0 - fg_mask_float)

        # ── Step 4: Composite shadow beneath the foreground ───────────────────
        result = composite.astype(np.float32)

        r, g, b = shadow_color
        for c_idx, c_val in enumerate([r, g, b]):
            shadow_contribution = shadow_alpha * c_val
            # Darken the composite where shadow falls
            result[:, :, c_idx] = (
                result[:, :, c_idx] * (1.0 - shadow_alpha * 0.8) +
                shadow_contribution * 0.2
            )

        result = np.clip(result, 0, 255).astype(np.uint8)

        # ── Build RGBA shadow layer for export ─────────────────────────────────
        shadow_layer_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        shadow_layer_alpha = (shadow_alpha * 255).clip(0, 255).astype(np.uint8)
        shadow_layer = np.dstack([shadow_layer_rgb, shadow_layer_alpha])

        elapsed = time.perf_counter() - t0
        logger.info(f"Shadow synthesis completed in {elapsed:.3f}s")

        return ShadowResult(
            image_with_shadow=result,
            shadow_layer=shadow_layer,
            synthesis_time_s=elapsed,
        )

    def add_contact_shadow(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray,
        blur_radius: int = 20,
        opacity: float = 0.45,
    ) -> np.ndarray:
        """
        Add a simple ambient contact shadow directly below the object.
        Simulates ground plane contact for objects on flat surfaces.

        Args:
            image: RGB image (H x W x 3, uint8).
            binary_mask: Object binary mask (H x W, uint8).
            blur_radius: Blur kernel for soft shadow edge.
            opacity: Shadow opacity.

        Returns:
            RGB image with contact shadow (H x W x 3, uint8).
        """
        h, w = image.shape[:2]

        # Project mask straight down (y-only, no x shear)
        shadow = (binary_mask / 255.0).astype(np.float32)
        # Shift down by 5% of height for slight offset
        shift = max(1, int(h * 0.03))
        shadow = np.roll(shadow, shift, axis=0)
        shadow[:shift, :] = 0

        # Blur for softness
        if blur_radius > 1:
            k = max(3, blur_radius * 2 + 1)
            if k % 2 == 0:
                k += 1
            shadow = cv2.GaussianBlur(shadow, (k, k), blur_radius / 2.5)

        shadow = np.clip(shadow * opacity, 0, 1)

        result = image.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= (1.0 - shadow * 0.7)

        return np.clip(result, 0, 255).astype(np.uint8)
