"""
Functional and structural tests for SceneShift mask refinement pipeline.
Tests morphological denoising, edge alignment, and feathering stages.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np
import pytest

from src.pipeline.mask_utils import (
    MaskRefinementResult,
    feather_mask,
    morphological_denoise,
    edge_align_mask,
    refine_mask,
    apply_alpha_composite,
)


class TestMorphologicalDenoising(unittest.TestCase):
    """Tests for Stage 1: Morphological denoising."""

    def _solid_circle_mask(self, size=256, radius=80) -> np.ndarray:
        """Create a clean circular mask."""
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
        return mask

    def _noisy_mask(self, size=256) -> np.ndarray:
        """Create a mask with small noise blobs."""
        mask = self._solid_circle_mask(size)
        # Add noise blobs (small, should be removed)
        for (x, y) in [(10, 10), (240, 240), (15, 240), (230, 20)]:
            cv2.circle(mask, (x, y), 5, 255, -1)
        return mask

    def test_removes_small_objects(self):
        """Small noise blobs below min_area should be removed."""
        mask = self._noisy_mask()
        result = morphological_denoise(mask, min_area=200)
        # Count connected components after denoising
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        # Should only have the main circle (1 component + background)
        self.assertEqual(num_labels, 2, "Small blobs should be eliminated.")

    def test_fills_holes(self):
        """Holes inside the mask should be filled."""
        mask = self._solid_circle_mask()
        # Create a hole inside the circle
        cv2.circle(mask, (128, 128), 20, 0, -1)
        result = morphological_denoise(mask, min_area=100)
        # After hole filling, the center should be nonzero
        self.assertGreater(result[128, 128], 0, "Hole should be filled.")

    def test_output_is_binary(self):
        """Output should contain only 0 and 255."""
        mask = self._noisy_mask()
        result = morphological_denoise(mask)
        unique = np.unique(result)
        self.assertTrue(
            set(unique).issubset({0, 255}),
            f"Non-binary values found: {unique}",
        )

    def test_preserves_large_object(self):
        """Large foreground object must not be removed."""
        mask = self._solid_circle_mask()
        before_area = np.count_nonzero(mask)
        result = morphological_denoise(mask, min_area=100)
        after_area = np.count_nonzero(result)
        # Should retain most of the mask (within 20% of original)
        self.assertGreater(after_area, before_area * 0.7)


class TestEdgeAlignment(unittest.TestCase):
    """Tests for Stage 2: Edge alignment."""

    def _synthetic_image_and_mask(self, size=128):
        """Create an image with a clear edge and matching mask."""
        image = np.zeros((size, size, 3), dtype=np.uint8)
        # Left half white, right half black — sharp edge at center
        image[:, :size // 2] = 200
        image[:, size // 2:] = 30
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[:, :size // 2] = 255
        return image, mask

    def test_returns_correct_shapes(self):
        """Refined mask and edge map must match input dimensions."""
        image, mask = self._synthetic_image_and_mask(128)
        refined, edge_map = edge_align_mask(mask, image)
        self.assertEqual(refined.shape, mask.shape)
        self.assertEqual(edge_map.shape, mask.shape)

    def test_edge_map_has_edges(self):
        """Edge map should detect the boundary in a high-contrast image."""
        image, mask = self._synthetic_image_and_mask(128)
        _, edge_map = edge_align_mask(mask, image)
        # Edge map should have non-zero pixels (edge detected)
        self.assertGreater(edge_map.sum(), 0, "Edge map should not be empty.")

    def test_refined_mask_is_binary(self):
        """Refined mask should be 0 or 255."""
        image, mask = self._synthetic_image_and_mask(128)
        refined, _ = edge_align_mask(mask, image)
        unique = np.unique(refined)
        self.assertTrue(set(unique).issubset({0, 255}))


class TestFeathering(unittest.TestCase):
    """Tests for Stage 3: Distance-transform feathering."""

    def _circle_mask(self, size=128, radius=50) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
        return mask

    def test_output_range(self):
        """Alpha mask values should be in [0, 1]."""
        mask = self._circle_mask()
        alpha = feather_mask(mask, feather_radius=10)
        # tanh-sigmoid never reaches exact 0/1 at finite distances,
        # so check near-zero and near-one bounds
        self.assertLess(float(alpha.min()), 0.1, "Min should be near zero")
        self.assertGreater(float(alpha.max()), 0.9, "Max should be near one")
        # All values must be in valid [0, 1] range
        self.assertGreaterEqual(float(alpha.min()), 0.0)
        self.assertLessEqual(float(alpha.max()), 1.0)

    def test_center_is_opaque(self):
        """Center of circular mask should be near 1.0."""
        mask = self._circle_mask(128, 50)
        alpha = feather_mask(mask, feather_radius=8)
        center_val = float(alpha[64, 64])
        self.assertGreater(center_val, 0.85, "Center should be near-opaque.")

    def test_outside_is_transparent(self):
        """Corners far outside the mask should be near 0.0."""
        mask = self._circle_mask(128, 40)
        alpha = feather_mask(mask, feather_radius=8)
        corner_val = float(alpha[0, 0])
        self.assertLess(corner_val, 0.15, "Corner should be near-transparent.")

    def test_output_dtype(self):
        """Output should be float32."""
        mask = self._circle_mask()
        alpha = feather_mask(mask)
        self.assertEqual(alpha.dtype, np.float32)

    def test_smooth_transition(self):
        """The transition from 0→1 should be smooth (no hard jumps)."""
        mask = self._circle_mask(256, 100)
        alpha = feather_mask(mask, feather_radius=20)
        # Extract radial slice through center
        row = alpha[128, :]
        # Check max step between adjacent pixels
        max_step = float(np.abs(np.diff(row)).max())
        self.assertLess(max_step, 0.2, "Transition should be smooth.")


class TestRefinesMaskPipeline(unittest.TestCase):
    """Integration tests for the full refine_mask pipeline."""

    def test_full_pipeline_returns_result(self):
        """Full pipeline should return a MaskRefinementResult."""
        size = 128
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (64, 64), 40, 255, -1)
        image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

        result = refine_mask(mask, image_rgb=image)
        self.assertIsInstance(result, MaskRefinementResult)
        self.assertEqual(result.binary_mask.shape, (size, size))
        self.assertEqual(result.alpha_mask.shape, (size, size))
        self.assertEqual(result.alpha_mask.dtype, np.float32)

    def test_empty_mask_handled(self):
        """Empty (all-zero) mask should not raise errors."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = refine_mask(mask)
        self.assertIsNotNone(result)

    def test_full_mask_handled(self):
        """All-255 mask should not raise errors."""
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        result = refine_mask(mask)
        self.assertIsNotNone(result)


class TestAlphaComposite(unittest.TestCase):
    """Tests for alpha compositing helper."""

    def test_full_alpha_gives_foreground(self):
        """With alpha=1 everywhere, output should equal foreground."""
        fg = np.full((32, 32, 3), 200, dtype=np.uint8)
        bg = np.zeros((32, 32, 3), dtype=np.uint8)
        alpha = np.ones((32, 32), dtype=np.float32)
        result = apply_alpha_composite(fg, bg, alpha)
        np.testing.assert_array_almost_equal(result, fg, decimal=0)

    def test_zero_alpha_gives_background(self):
        """With alpha=0 everywhere, output should equal background."""
        fg = np.full((32, 32, 3), 200, dtype=np.uint8)
        bg = np.full((32, 32, 3), 50, dtype=np.uint8)
        alpha = np.zeros((32, 32), dtype=np.float32)
        result = apply_alpha_composite(fg, bg, alpha)
        np.testing.assert_array_almost_equal(result, bg, decimal=0)

    def test_half_alpha_blends(self):
        """With alpha=0.5 everywhere, output should be midpoint."""
        fg = np.full((16, 16, 3), 200, dtype=np.uint8)
        bg = np.full((16, 16, 3), 100, dtype=np.uint8)
        alpha = np.full((16, 16), 0.5, dtype=np.float32)
        result = apply_alpha_composite(fg, bg, alpha)
        expected = 150  # (200*0.5 + 100*0.5)
        np.testing.assert_array_almost_equal(result, expected, decimal=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
