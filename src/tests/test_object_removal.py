"""
Unit tests for the targeted object removal pipeline stage.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from src.pipeline.object_removal import ObjectRemovalPipeline, ObjectRemovalResult


class TestObjectRemovalPipeline(unittest.TestCase):
    """Tests for ObjectRemovalPipeline detection, masking, and inpaint blending."""

    def setUp(self):
        # Create dummy image (RGB)
        self.image = np.zeros((256, 256, 3), dtype=np.uint8)
        # Draw a white rectangle in the middle to represent an object
        cv2.rectangle(self.image, (100, 120), (150, 170), (255, 255, 255), -1)

        # Mock FaceProtector
        self.mock_face_protector = MagicMock()
        # Mock LocalInpainter
        self.mock_inpainter = MagicMock()
        
        # Initialize pipeline
        self.pipeline = ObjectRemovalPipeline(face_protector=self.mock_face_protector)

    def test_torso_heuristic_fallback_no_face(self):
        """If no face is detected, it should fall back to a center-lower box."""
        self.mock_face_protector.detect_faces.return_value = []
        h, w = self.image.shape[:2]
        boxes = [self.pipeline._chest_region_box(h, w, [])]
        
        self.assertEqual(len(boxes), 1)
        bx1, by1, bx2, by2 = boxes[0]
        self.assertEqual(bx1, w // 3)
        self.assertEqual(bx2, w * 2 // 3)
        self.assertEqual(by1, h // 2)
        self.assertEqual(by2, h * 5 // 6)

    def test_torso_heuristic_fallback_with_face(self):
        """If a face is detected, chest bounding box should be placed below the face."""
        mock_face = MagicMock()
        mock_face.bbox = (80, 20, 120, 60)  # fx1, fy1, fx2, fy2
        self.mock_face_protector.detect_faces.return_value = [mock_face]
        
        h, w = self.image.shape[:2]
        boxes = [self.pipeline._chest_region_box(h, w, [mock_face])]
        self.assertEqual(len(boxes), 1)
        bx1, by1, bx2, by2 = boxes[0]
        # Should be below fy2 (60)
        self.assertGreaterEqual(by1, 60)
        self.assertLess(bx1, bx2)
        self.assertGreater(by2, by1)

    def test_generate_removal_mask_without_grabcut(self):
        """Mask generation without GrabCut should just fill the raw boxes."""
        boxes = [(100, 120, 150, 170)]
        mask = self.pipeline.generate_removal_mask(self.image, boxes, faces=[], targets=["backpack"], use_grabcut=False)
        self.assertEqual(mask.shape, (256, 256))
        # Raw box should be filled (with dilation, so even larger)
        self.assertEqual(mask[130, 120], 255)

    def test_generate_removal_mask_lanyard_path(self):
        """Lanyard path should be drawn from face bottom to the target box."""
        mock_face = MagicMock()
        mock_face.bbox = (100, 20, 140, 60)  # Face width=40, bottom=60
        faces = [mock_face]
        boxes = [(110, 140, 130, 160)]  # Target box below face
        
        mask = self.pipeline.generate_removal_mask(self.image, boxes, faces, targets=["lanyard"], use_grabcut=False)
        # Ensure some pixels along the neck-to-card line are white (255)
        # e.g., neck center is around (120, 60), card top is around (120, 140)
        # Midpoint of path is (120, 100)
        self.assertEqual(mask[100, 120], 255)

    def test_remove_empty_mask(self):
        """If no targets are detected or heuristic produces an empty mask, it should skip inpainting."""
        # Stub detect_targets to return empty
        self.pipeline.detect_targets = MagicMock(return_value=[])
        self.mock_face_protector.detect_faces.return_value = []
        
        res = self.pipeline.remove(self.image, local_inpainter=self.mock_inpainter)
        self.assertTrue(np.array_equal(res.image, self.image))
        self.assertEqual(np.count_nonzero(res.object_mask), 0)
        self.mock_inpainter.inpaint.assert_not_called()

    def test_remove_success(self):
        """If targets are found, local_inpainter.inpaint should be called and result returned."""
        # Mock detection boxes
        self.pipeline.detect_targets = MagicMock(return_value=[(100, 120, 150, 170)])
        self.mock_face_protector.detect_faces.return_value = []
        
        # Create an inpainted image (different color in the box)
        edited_img = self.image.copy()
        cv2.rectangle(edited_img, (100, 120), (150, 170), (0, 0, 0), -1)  # Painted black
        self.mock_inpainter.inpaint.return_value = edited_img
        
        res = self.pipeline.remove(self.image, local_inpainter=self.mock_inpainter)
        self.mock_inpainter.inpaint.assert_called_once()
        
        # Original image was white in (120, 130), in res it should be blended towards black
        # Check that it's changed inside the mask
        self.assertLess(res.image[130, 120, 0], 255)
        # Outside mask (e.g. at 10, 10) it should remain exactly identical
        np.testing.assert_array_equal(res.image[10, 10], self.image[10, 10])


if __name__ == "__main__":
    unittest.main()
