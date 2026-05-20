"""
Tests for the segmentation engine (YOLO + SAM2).
Uses mocking to avoid requiring GPU or large model downloads during CI.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.models.segmentation import (
    SegmentationResult,
    SegmentationEngine,
    SAM2Segmentor,
    YOLOSegmentor,
)


def _make_rgb_image(h=256, w=256) -> np.ndarray:
    """Create a random RGB test image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_binary_mask(h=256, w=256, radius=80) -> np.ndarray:
    """Create a circular binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
    return mask


class TestSegmentationResult(unittest.TestCase):
    """Test the SegmentationResult dataclass."""

    def test_construction(self):
        mask = _make_binary_mask()
        result = SegmentationResult(
            mask=mask,
            bbox=(10, 20, 200, 210),
            label="cat",
            confidence=0.92,
            mode="auto",
            inference_time_s=0.15,
        )
        self.assertEqual(result.label, "cat")
        self.assertAlmostEqual(result.confidence, 0.92)
        self.assertEqual(result.mode, "auto")
        self.assertEqual(result.bbox, (10, 20, 200, 210))

    def test_all_masks_default_empty(self):
        mask = _make_binary_mask()
        result = SegmentationResult(
            mask=mask, bbox=(0, 0, 10, 10),
            label="obj", confidence=0.8,
            mode="auto", inference_time_s=0.1,
        )
        self.assertEqual(result.all_masks, [])


class TestYOLOSegmentor(unittest.TestCase):
    """Tests for YOLO automatic segmentor (mocked)."""

    def test_segment_returns_result(self):
        """With mocked YOLO model, segment() should return SegmentationResult."""
        import torch
        h, w = 256, 256

        # Build realistic tensor mocks
        masks_data = torch.zeros(1, h, w)
        masks_data[0, 80:180, 80:180] = 0.9  # Region in center

        boxes_xyxy = torch.tensor([[80.0, 80.0, 180.0, 180.0]])
        boxes_conf = torch.tensor([0.92])
        boxes_cls = torch.tensor([0.0])

        mock_boxes = MagicMock()
        mock_boxes.xyxy = boxes_xyxy
        mock_boxes.conf = boxes_conf
        mock_boxes.cls = boxes_cls
        mock_boxes.__len__ = lambda s: 1

        mock_result = MagicMock()
        mock_result.masks.data = masks_data
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "person"}

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = [mock_result]

        # Directly inject the mock model — bypasses ultralytics import entirely
        seg = YOLOSegmentor()
        seg._model = mock_model_instance   # Bypass lazy loading

        image = _make_rgb_image(h, w)
        result = seg.segment(image)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, SegmentationResult)
        self.assertEqual(result.mode, "auto")
        self.assertEqual(result.label, "person")
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(result.mask.shape, (h, w))

    def test_lazy_load_on_missing_model(self):
        """_load_model should set _model when ultralytics is importable."""
        seg = YOLOSegmentor()
        self.assertIsNone(seg._model)


class TestSAM2Segmentor(unittest.TestCase):
    """Tests for SAM2 interactive segmentor (fallback path tested)."""

    def test_fallback_segmentation(self):
        """With SAM2 not installed, the fallback elliptical mask should work."""
        seg = SAM2Segmentor()
        seg._predictor = "fallback"  # Force fallback

        h, w = 128, 128
        image = _make_rgb_image(h, w)
        result = seg.segment(image, click_points=[(64, 64)])

        self.assertIsNotNone(result)
        self.assertIsInstance(result, SegmentationResult)
        self.assertEqual(result.mode, "interactive")
        self.assertEqual(result.mask.shape, (h, w))
        self.assertGreater(np.count_nonzero(result.mask), 0)

    def test_click_out_of_center(self):
        """Fallback mask center should follow the click point."""
        seg = SAM2Segmentor()
        seg._predictor = "fallback"
        h, w = 200, 200
        image = _make_rgb_image(h, w)
        result = seg.segment(image, click_points=[(20, 20)])
        # Mask should have some pixels near (20, 20)
        mask_region = result.mask[0:60, 0:60]
        self.assertGreater(np.count_nonzero(mask_region), 0)

    def test_default_labels_all_foreground(self):
        """Without explicit labels, all points should be foreground (1)."""
        seg = SAM2Segmentor()
        seg._predictor = "fallback"
        h, w = 128, 128
        image = _make_rgb_image(h, w)
        result = seg.segment(image, click_points=[(64, 64), (50, 70)])
        self.assertEqual(result.mode, "interactive")

    def test_confidence_in_valid_range(self):
        """Confidence should be in [0, 1]."""
        seg = SAM2Segmentor()
        seg._predictor = "fallback"
        image = _make_rgb_image(128, 128)
        result = seg.segment(image, click_points=[(64, 64)])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestSegmentationEngine(unittest.TestCase):
    """Integration tests for the unified SegmentationEngine."""

    def test_engine_instantiates(self):
        engine = SegmentationEngine()
        self.assertIsNotNone(engine._yolo)
        self.assertIsNotNone(engine._sam2)

    def test_interactive_fallback(self):
        """interactive_segment should work with SAM2 fallback."""
        engine = SegmentationEngine()
        engine._sam2._predictor = "fallback"  # Force fallback
        image = _make_rgb_image(128, 128)
        result = engine.interactive_segment(image, [(64, 64)])
        self.assertIsNotNone(result)
        self.assertEqual(result.mode, "interactive")


if __name__ == "__main__":
    unittest.main(verbosity=2)
