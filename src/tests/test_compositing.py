"""
Structural and functional tests for the compositing engine.
Tests Alpha, Poisson, and Laplacian pyramid blending modes.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np
import pytest

from src.models.compositing import (
    CompositingEngine,
    BlendMode,
    alpha_blend,
    poisson_blend,
    laplacian_pyramid_blend,
    _build_gaussian_pyramid,
    _build_laplacian_pyramid,
)


def _checkerboard(size=64, square=16) -> np.ndarray:
    """Create a checkerboard test image (RGB)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((y // square) + (x // square)) % 2 == 0:
                img[y, x] = [200, 200, 200]
            else:
                img[y, x] = [50, 50, 80]
    return img


def _solid_color(h=64, w=64, color=(100, 150, 200)) -> np.ndarray:
    """Create a solid-color test image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _circle_mask(h=64, w=64, radius=20) -> np.ndarray:
    """Create a circular binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
    return mask


class TestAlphaBlend(unittest.TestCase):
    """Tests for feathered alpha blending."""

    def test_full_alpha(self):
        """Alpha=1 → output equals foreground."""
        fg = _solid_color(color=(200, 100, 50))
        bg = _solid_color(color=(30, 30, 80))
        alpha = np.ones((64, 64), dtype=np.float32)
        result = alpha_blend(fg, bg, alpha)
        np.testing.assert_array_almost_equal(result, fg, decimal=0)

    def test_zero_alpha(self):
        """Alpha=0 → output equals background."""
        fg = _solid_color(color=(200, 100, 50))
        bg = _solid_color(color=(30, 30, 80))
        alpha = np.zeros((64, 64), dtype=np.float32)
        result = alpha_blend(fg, bg, alpha)
        np.testing.assert_array_almost_equal(result, bg, decimal=0)

    def test_output_shape(self):
        """Output shape must match input."""
        fg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        alpha = np.random.rand(128, 128).astype(np.float32)
        result = alpha_blend(fg, bg, alpha)
        self.assertEqual(result.shape, (128, 128, 3))

    def test_output_dtype(self):
        """Output should be uint8."""
        fg = _solid_color()
        bg = _solid_color(color=(10, 10, 10))
        alpha = np.full((64, 64), 0.5, dtype=np.float32)
        result = alpha_blend(fg, bg, alpha)
        self.assertEqual(result.dtype, np.uint8)

    def test_clipped_values(self):
        """All pixel values should be in [0, 255]."""
        fg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        alpha = np.random.rand(64, 64).astype(np.float32)
        result = alpha_blend(fg, bg, alpha)
        self.assertTrue(result.min() >= 0)
        self.assertTrue(result.max() <= 255)


class TestPoissonBlend(unittest.TestCase):
    """Tests for Poisson seamless blending."""

    def test_output_shape_matches_background(self):
        """Output should match background dimensions."""
        fg = _checkerboard(64)
        bg = _solid_color(64, 64, (20, 40, 80))
        mask = _circle_mask(64, 64, 20)
        result = poisson_blend(fg, bg, mask)
        self.assertEqual(result.shape, bg.shape)

    def test_output_dtype(self):
        """Result must be uint8."""
        fg = _checkerboard(64)
        bg = _solid_color(64, 64)
        mask = _circle_mask(64, 64)
        result = poisson_blend(fg, bg, mask)
        self.assertEqual(result.dtype, np.uint8)

    def test_pixel_range(self):
        """All pixels in [0, 255]."""
        fg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = _circle_mask(64, 64, 20)
        result = poisson_blend(fg, bg, mask)
        self.assertTrue(result.min() >= 0)
        self.assertTrue(result.max() <= 255)

    def test_all_zeros_mask(self):
        """All-zero mask: result should be very close to background."""
        fg = _solid_color(64, 64, (200, 50, 50))
        bg = _solid_color(64, 64, (10, 10, 80))
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = poisson_blend(fg, bg, mask)
        # With zero mask, background dominates
        diff = np.abs(result.astype(int) - bg.astype(int)).mean()
        self.assertLess(diff, 30.0, "Zero-mask Poisson should stay close to BG.")


class TestLaplacianPyramidBlend(unittest.TestCase):
    """Tests for Laplacian pyramid multi-band blending."""

    def test_output_shape(self):
        """Output shape must match inputs."""
        fg = _checkerboard(64)
        bg = _solid_color(64, 64)
        mask = _circle_mask(64, 64, 25)
        result = laplacian_pyramid_blend(fg, bg, mask, levels=4)
        self.assertEqual(result.shape, (64, 64, 3))

    def test_output_dtype(self):
        """Result should be uint8."""
        fg = _checkerboard(64)
        bg = _solid_color(64, 64, (80, 40, 20))
        mask = _circle_mask(64, 64, 25)
        result = laplacian_pyramid_blend(fg, bg, mask, levels=3)
        self.assertEqual(result.dtype, np.uint8)

    def test_pixel_range(self):
        """All pixels in [0, 255]."""
        fg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = _circle_mask(64, 64, 20)
        result = laplacian_pyramid_blend(fg, bg, mask, levels=4)
        self.assertTrue(result.min() >= 0)
        self.assertTrue(result.max() <= 255)

    def test_all_white_mask(self):
        """All-white mask: result should be close to foreground."""
        fg = _solid_color(64, 64, (200, 100, 50))
        bg = _solid_color(64, 64, (20, 20, 80))
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        result = laplacian_pyramid_blend(fg, bg, mask, levels=3)
        diff = np.abs(result.astype(int) - fg.astype(int)).mean()
        self.assertLess(diff, 30.0, "Full mask should mostly show foreground.")


class TestGaussianPyramid(unittest.TestCase):
    """Tests for Gaussian pyramid builder."""

    def test_correct_number_of_levels(self):
        img = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = _build_gaussian_pyramid(img, levels=4)
        self.assertEqual(len(pyramid), 4)

    def test_decreasing_resolution(self):
        img = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = _build_gaussian_pyramid(img, levels=4)
        for i in range(len(pyramid) - 1):
            self.assertLess(pyramid[i + 1].shape[0], pyramid[i].shape[0])


class TestLaplacianPyramid(unittest.TestCase):
    """Tests for Laplacian pyramid builder."""

    def test_correct_length(self):
        img = np.random.rand(128, 128, 3).astype(np.float32)
        gp = _build_gaussian_pyramid(img, levels=4)
        lp = _build_laplacian_pyramid(gp)
        self.assertEqual(len(lp), len(gp))


class TestCompositingEngine(unittest.TestCase):
    """Integration tests for the CompositingEngine unified interface."""

    def setUp(self):
        self.engine = CompositingEngine()
        self.fg = _checkerboard(64)
        self.bg = _solid_color(64, 64, (20, 40, 80))
        self.alpha = np.zeros((64, 64), dtype=np.float32)
        self.mask = _circle_mask(64, 64, 22)
        # Set alpha values within circle
        ys, xs = np.where(self.mask > 0)
        self.alpha[ys, xs] = 1.0

    def test_alpha_mode(self):
        result = self.engine.composite(self.fg, self.bg, self.alpha, self.mask, mode="alpha")
        self.assertEqual(result.composite.shape, (64, 64, 3))
        self.assertEqual(result.composite.dtype, np.uint8)
        self.assertEqual(result.blend_mode, "alpha")

    def test_poisson_mode(self):
        result = self.engine.composite(self.fg, self.bg, self.alpha, self.mask, mode="poisson")
        self.assertEqual(result.composite.shape, (64, 64, 3))
        self.assertEqual(result.composite.dtype, np.uint8)

    def test_laplacian_mode(self):
        result = self.engine.composite(self.fg, self.bg, self.alpha, self.mask, mode="laplacian")
        self.assertEqual(result.composite.shape, (64, 64, 3))
        self.assertEqual(result.composite.dtype, np.uint8)

    def test_invalid_mode_falls_back_to_alpha(self):
        result = self.engine.composite(self.fg, self.bg, self.alpha, self.mask, mode="INVALID")
        self.assertIsNotNone(result.composite)

    def test_blend_time_recorded(self):
        result = self.engine.composite(self.fg, self.bg, self.alpha, self.mask, mode="alpha")
        self.assertGreater(result.blend_time_s, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
