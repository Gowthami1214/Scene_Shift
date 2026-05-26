"""
Unit and functional tests for the PhotoPreprocessor module.
"""

from __future__ import annotations

import unittest
import cv2
import numpy as np

from src.pipeline.preprocessing import PhotoPreprocessor, PreprocessingResult


def _flat_color_image(h: int = 128, w: int = 128, color: tuple = (100, 120, 150)) -> np.ndarray:
    """Create a flat-colored image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


class TestPhotoPreprocessor(unittest.TestCase):
    """Tests for individual preprocessing components and pipeline integration."""

    def setUp(self):
        self.preprocessor = PhotoPreprocessor()

    def test_detect_noise_level_on_clean_image(self):
        # A flat image has 0 noise (low variance)
        img = _flat_color_image(64, 64)
        noise = self.preprocessor._detect_noise_level(img)
        self.assertEqual(noise, 0.0)

    def test_detect_exposure_normal(self):
        # LAB L* channel mean around 128 is normal
        img = _flat_color_image(64, 64, color=(128, 128, 128))
        exposure = self.preprocessor._detect_exposure(img)
        self.assertEqual(exposure, "normal")

    def test_detect_exposure_under(self):
        # L* channel mean very low
        img = _flat_color_image(64, 64, color=(10, 10, 10))
        exposure = self.preprocessor._detect_exposure(img)
        self.assertEqual(exposure, "under")

    def test_detect_exposure_over(self):
        # L* channel mean very high
        img = _flat_color_image(64, 64, color=(240, 240, 240))
        exposure = self.preprocessor._detect_exposure(img)
        self.assertEqual(exposure, "over")

    def test_needs_white_balance_false(self):
        # Equal channel values
        img = _flat_color_image(64, 64, color=(100, 100, 100))
        self.assertFalse(self.preprocessor._needs_white_balance(img))

    def test_needs_white_balance_true(self):
        # Imbalanced channels (heavy red color cast)
        img = _flat_color_image(64, 64, color=(180, 50, 50))
        self.assertTrue(self.preprocessor._needs_white_balance(img))

    def test_ensure_min_resolution_noop(self):
        # 512x512 is already >= min_resolution
        img = _flat_color_image(512, 512)
        out = self.preprocessor._ensure_min_resolution(img, min_size=256)
        self.assertEqual(out.shape, (512, 512, 3))

    def test_ensure_min_resolution_upscale(self):
        # 128x128 is upscaled to shortest side = 256
        img = _flat_color_image(128, 128)
        out = self.preprocessor._ensure_min_resolution(img, min_size=256)
        self.assertEqual(out.shape, (256, 256, 3))

    def test_auto_white_balance_correction(self):
        # Imbalanced color cast should become more balanced
        img = _flat_color_image(32, 32, color=(200, 100, 100))
        corrected = self.preprocessor._correct_white_balance(img)
        means = corrected.mean(axis=(0, 1))
        # Deviation should be smaller after AWB
        avg = means.mean()
        self.assertTrue(all(abs(m - avg) < 1.0 for m in means))

    def test_exposure_normalization_clahe(self):
        # Low contrast / dark image should get normalized
        img = _flat_color_image(64, 64, color=(20, 20, 20))
        normalized = self.preprocessor._normalize_exposure(img)
        self.assertGreater(normalized.mean(), img.mean())

    def test_full_preprocessing_pipeline(self):
        # Small, dark, color-casted, noisy image
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[:] = (200, 50, 50)  # Red color cast, darkish
        # Add random noise
        np.random.seed(42)
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        res = self.preprocessor.preprocess(
            img,
            enable_denoise=True,
            enable_exposure=True,
            enable_white_balance=True,
            enable_artifact_removal=True,
            min_resolution=256,
        )

        self.assertIsInstance(res, PreprocessingResult)
        # Should be upscaled to 256
        self.assertEqual(res.image.shape, (256, 256, 3))
        self.assertEqual(res.original_size, (128, 128))
        self.assertIn("resolution_upscale", res.applied_corrections)
        # One or more of the other corrections should have fired
        self.assertGreater(len(res.applied_corrections), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
