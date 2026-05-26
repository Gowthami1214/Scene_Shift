"""
Unit tests for the non-generative LocalInpainter.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.models.lama_inpainting import LocalInpainter


class TestLocalInpainter(unittest.TestCase):
    """Test suite verifying fast local inpainting and texture fallbacks."""

    def setUp(self):
        # Create a solid color image (RGB)
        self.image = np.zeros((128, 128, 3), dtype=np.uint8)
        self.image[:] = [120, 150, 180]
        
        # Add some local texture/noise
        noise = np.random.normal(0, 5, self.image.shape)
        self.image = np.clip(self.image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Draw a white rectangle to represent the object to remove
        cv2.rectangle(self.image, (40, 40), (80, 80), (255, 255, 255), -1)

        # Create binary mask representing the object
        self.mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(self.mask, (40, 40), (80, 80), 255, -1)

        # Initialize local inpainter with LaMa disabled to force OpenCV fallback testing
        self.inpainter = LocalInpainter(use_lama=False)

    def test_inpaint_empty_mask(self):
        """Empty mask should return a copy of the original image without modifications."""
        empty_mask = np.zeros((128, 128), dtype=np.uint8)
        result = self.inpainter.inpaint(self.image, empty_mask)
        np.testing.assert_array_equal(result, self.image)

    def test_opencv_fallback_blends_correctly(self):
        """Verify that regions inside the mask are inpainted and regions outside are preserved."""
        result = self.inpainter.inpaint(self.image, self.mask)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)

        # Pixels inside the mask should no longer be solid white (255, 255, 255)
        # They should blend back to the background color (approx [120, 150, 180])
        self.assertLess(result[60, 60, 0], 255)
        self.assertLess(result[60, 60, 1], 255)

        # Pixels far outside the mask (e.g. at (10, 10)) must remain identical to the original image
        np.testing.assert_array_equal(result[10, 10], self.image[10, 10])

    def test_preserves_grain_fallbacks(self):
        """Verify that the texture fallback successfully injects some noise matching reference std."""
        result = self.inpainter.inpaint(self.image, self.mask)
        # Check standard deviation inside the inpainted zone (it should contain grain, not be perfectly flat)
        inpainted_patch = result[45:75, 45:75]
        patch_std = inpainted_patch.std()
        self.assertGreater(patch_std, 0.5, "Inpainted patch should not be flat; it should contain texture grain.")


if __name__ == "__main__":
    unittest.main()
