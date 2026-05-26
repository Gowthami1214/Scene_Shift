"""
Unit tests for the Face Preservation components.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from src.pipeline.face_preservation import (
    FaceRegion,
    FaceProtector,
    FaceRestorer,
    SkinTonePreserver,
)


class TestFaceProtector(unittest.TestCase):
    """Tests for the FaceProtector class."""

    def setUp(self):
        self.protector = FaceProtector()

    def test_create_protection_mask_no_faces(self):
        mask = self.protector.create_protection_mask((100, 100), [])
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_create_protection_mask_with_faces(self):
        faces = [
            FaceRegion(bbox=(10, 10, 30, 30), confidence=0.9, center=(20, 20), area=400)
        ]
        mask = self.protector.create_protection_mask((100, 100), faces, padding_fraction=0.0, feather_radius=0)
        self.assertEqual(mask.shape, (100, 100))
        # Mask should have pixels set around the ellipse center
        self.assertGreater(np.count_nonzero(mask), 0)
        # Bounding box region should contain active pixels
        self.assertEqual(mask[20, 20], 255)
        # Background should be 0
        self.assertEqual(mask[0, 0], 0)

    @patch("src.pipeline.face_preservation.FaceProtector._load_detectors")
    def test_detect_faces_returns_empty_when_no_net(self, mock_load):
        self.protector._dnn_net = None
        self.protector._haar_cascade = None
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.protector.detect_faces(img)
        self.assertEqual(len(faces), 0)


class TestFaceRestorer(unittest.TestCase):
    """Tests for the FaceRestorer class."""

    def setUp(self):
        self.restorer = FaceRestorer()

    def test_restore_no_faces_returns_same_image(self):
        proc = np.ones((64, 64, 3), dtype=np.uint8) * 100
        orig = np.ones((64, 64, 3), dtype=np.uint8) * 200
        res = self.restorer.restore(proc, orig, [])
        self.assertTrue(np.array_equal(res.restored_image, proc))
        self.assertEqual(res.num_faces_restored, 0)

    def test_restore_face_crop_enhancement(self):
        proc = np.ones((128, 128, 3), dtype=np.uint8) * 100
        orig = np.ones((128, 128, 3), dtype=np.uint8) * 128
        faces = [
            FaceRegion(bbox=(20, 20, 80, 80), confidence=0.9, center=(50, 50), area=3600)
        ]
        
        res = self.restorer.restore(proc, orig, faces)
        self.assertEqual(res.restored_image.shape, (128, 128, 3))
        self.assertEqual(res.num_faces_restored, 1)
        # The region inside the face should be modified/enhanced
        face_pixels = res.restored_image[30:70, 30:70]
        # Should not be identical to original proc image
        self.assertFalse(np.array_equal(face_pixels, proc[30:70, 30:70]))


class TestSkinTonePreserver(unittest.TestCase):
    """Tests for the SkinTonePreserver class."""

    def setUp(self):
        self.preserver = SkinTonePreserver()

    def test_extract_skin_reference_no_faces(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ref = self.preserver.extract_skin_reference(img, [])
        self.assertIsNone(ref)

    def test_extract_skin_reference_with_skin(self):
        # Create skin tone color image (e.g. RGB=(230, 180, 150) -> YCrCb in range)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (230, 180, 150)
        faces = [
            FaceRegion(bbox=(10, 10, 90, 90), confidence=0.9, center=(50, 50), area=6400)
        ]
        ref = self.preserver.extract_skin_reference(img, faces)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.shape, (3,))

    def test_apply_skin_preservation(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (230, 180, 150)  # Skin color
        faces = [
            FaceRegion(bbox=(10, 10, 90, 90), confidence=0.9, center=(50, 50), area=6400)
        ]
        # Reference is slightly different skin tone
        ref_lab = np.array([75.0, 10.0, 15.0])
        
        corrected = self.preserver.apply_skin_preservation(img, ref_lab, faces, strength=0.5)
        self.assertEqual(corrected.shape, (100, 100, 3))
        # Image should be shifted towards the reference
        self.assertFalse(np.array_equal(corrected, img))


if __name__ == "__main__":
    unittest.main(verbosity=2)
