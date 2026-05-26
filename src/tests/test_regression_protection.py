"""
Deterministic Regression Protection, Safety Guard, and Snapshot Tests for SceneShift.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from src.models.background import BackgroundGenerator
from src.models.editing import ObjectEditor
from src.pipeline.prompt_parser import parse_prompt, BackgroundType
from src.pipeline.execution_planner import build_execution_plan, BackgroundStrategy, ForegroundStrategy
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineRequest
from src.models.segmentation import YOLOSegmentor


def _make_image(h=64, w=64) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestDiffusionSafetyGuard(unittest.TestCase):
    """Test suite for the Diffusion Safety Guard layer (regression protection)."""

    def test_background_generator_safety_guard(self):
        bg_gen = BackgroundGenerator()
        # Non-generative prompts must raise RuntimeError
        blocked_prompts = [
            "white background",
            "transparent",
            "color:#0000FF",
            "passport background",
            "studio background",
            "solid white",
            "blue",
        ]
        for prompt in blocked_prompts:
            with self.assertRaises(RuntimeError) as ctx:
                bg_gen.generate_sd(prompt=prompt)
            self.assertIn("Diffusion Safety Guard", str(ctx.exception))

        # Generative scene prompts should NOT trigger the safety guard error
        # (they will try to load model, so we mock load_pipeline to avoid actual model load)
        bg_gen._load_pipeline = MagicMock()
        bg_gen._pipeline = "fallback"  # Trigger procedural fallback logic after load
        res = bg_gen.generate_sd(prompt="cozy coffee shop")
        self.assertIsNotNone(res)

    def test_object_editor_safety_guard(self):
        editor = ObjectEditor()
        # Simple object removal / empty prompts must raise RuntimeError
        blocked_prompts = [
            "",
            "none",
            "object",
            "remove lanyard",
            "delete watermark",
            "erase glasses",
        ]
        image = _make_image()
        mask = np.zeros((64, 64), dtype=np.uint8)
        for prompt in blocked_prompts:
            with self.assertRaises(RuntimeError) as ctx:
                editor.edit(image_rgb=image, mask=mask, object_prompt=prompt)
            self.assertIn("Diffusion Safety Guard", str(ctx.exception))


class TestForegroundPreservation(unittest.TestCase):
    """Test suite to verify that original subject pixels are preserved exactly (pixel-stable)."""

    @patch("src.pipeline.orchestrator.SegmentationEngine")
    @patch("src.pipeline.orchestrator.ObjectEditor")
    @patch("src.pipeline.orchestrator.BackgroundGenerator")
    def test_foreground_pixels_unchanged(self, MockBG, MockEditor, MockSeg):
        h, w = 64, 64
        original_image = _make_image(h, w)
        # Create a circle in the center of the mask (foreground)
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dummy_mask, (w // 2, h // 2), 20, 255, -1)

        # Mock segmentor
        from src.models.segmentation import SegmentationResult
        mock_seg = MockSeg.return_value
        mock_seg.segment_with_matting.return_value = SegmentationResult(
            mask=dummy_mask,
            bbox=(12, 12, 52, 52),
            label="person",
            confidence=0.95,
            mode="auto+matting",
            inference_time_s=0.1,
            alpha_matte=(dummy_mask / 255.0).astype(np.float32),
        )

        orch = PipelineOrchestrator()
        orch._segmentation = mock_seg

        # Request solid background: SD editing skipped, foreground must be pixel-perfect copies
        req = PipelineRequest(
            image=original_image,
            command="change background to white",
            enable_preprocessing=False,
            enable_face_preservation=False,
            enable_matting=True,
            enable_shadow=False,
            enable_harmonization=False,
        )

        result = orch.run(req, job_id="test_fg_preservation")
        self.assertIsNotNone(result.final_image)

        # Foreground mask region should be exactly identical to original image pixels
        fg_mask_3c = np.dstack([dummy_mask / 255.0] * 3)
        original_fg = (original_image * fg_mask_3c).astype(np.uint8)
        output_fg = (result.final_image * fg_mask_3c).astype(np.uint8)

        np.testing.assert_array_equal(output_fg, original_fg)


class TestBackgroundRoutingRegression(unittest.TestCase):
    """Test suite verifying correct background command routing decisions (no diffusion)."""

    def test_routing_regression_cases(self):
        test_cases = [
            ("change background to white", BackgroundType.SOLID_COLOR, False, BackgroundStrategy.COLOR_FILL),
            ("change background to blue", BackgroundType.SOLID_COLOR, False, BackgroundStrategy.COLOR_FILL),
            ("remove background", BackgroundType.TRANSPARENT, False, BackgroundStrategy.TRANSPARENT_FILL),
            ("passport background", BackgroundType.SOLID_COLOR, False, BackgroundStrategy.COLOR_FILL),
            ("studio background", BackgroundType.STUDIO, False, BackgroundStrategy.COLOR_FILL),
            ("replace background with a beautiful beach", BackgroundType.GENERATED_SCENE, True, BackgroundStrategy.SD_GENERATION),
            ("change background to educational campus", BackgroundType.GENERATED_SCENE, True, BackgroundStrategy.SD_GENERATION),
        ]

        for cmd, expected_type, expected_diffusion, expected_strategy in test_cases:
            intent = parse_prompt(cmd)
            plan = build_execution_plan(intent)
            
            self.assertEqual(plan.background_type, expected_type, f"Cmd '{cmd}' failed background_type check")
            self.assertEqual(plan.use_diffusion, expected_diffusion, f"Cmd '{cmd}' failed use_diffusion check")
            self.assertEqual(plan.background_strategy, expected_strategy, f"Cmd '{cmd}' failed strategy check")


class TestMultiPersonForegroundMerge(unittest.TestCase):
    """Test suite verifying YOLO segmentor correctly unions masks of all detected persons."""

    def test_segment_multiple_people_union(self):
        import torch
        h, w = 128, 128
        # We mock YOLOv8-seg detections: two boxes of class "person" (class 0)
        masks_data = torch.zeros(2, h, w)
        masks_data[0, 10:40, 10:40] = 0.9   # Person 1 mask
        masks_data[1, 60:90, 60:90] = 0.9   # Person 2 mask

        boxes_xyxy = torch.tensor([
            [10.0, 10.0, 40.0, 40.0],
            [60.0, 60.0, 90.0, 90.0]
        ])
        boxes_conf = torch.tensor([0.88, 0.92])
        boxes_cls = torch.tensor([0.0, 0.0]) # Class 0 is "person"

        mock_boxes = MagicMock()
        mock_boxes.xyxy = boxes_xyxy
        mock_boxes.conf = boxes_conf
        mock_boxes.cls = boxes_cls
        mock_boxes.__len__ = lambda s: 2

        mock_result = MagicMock()
        mock_result.masks.data = masks_data
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        seg = YOLOSegmentor()
        seg._model = mock_model

        image = np.zeros((h, w, 3), dtype=np.uint8)
        result = seg.segment(image)

        self.assertIsNotNone(result)
        self.assertEqual(result.label, "person")
        # Unified bounding box enclosing both people (x1=10, y1=10, x2=90, y2=90)
        self.assertEqual(result.bbox, (10, 10, 90, 90))
        # Mask should contain union of both regions
        self.assertEqual(result.mask[25, 25], 255)
        self.assertEqual(result.mask[75, 75], 255)
        self.assertEqual(result.mask[50, 50], 0)


class TestExecutionPlanSnapshot(unittest.TestCase):
    """Snapshot tests for ExecutionPlan outputs to catch future routing regressions."""

    def test_execution_plan_snapshots(self):
        # We define a snapshot dict matching commands to expected plan properties.
        # If any future refactoring changes these properties, the tests will fail.
        snapshots = {
            "white background": {
                "background_type": BackgroundType.SOLID_COLOR,
                "use_diffusion": False,
                "use_color_compositing": True,
                "background_strategy": BackgroundStrategy.COLOR_FILL,
                "preserve_all_people": True,
                "remove_people": False,
            },
            "remove backpack": {
                "background_type": BackgroundType.NONE,
                "use_diffusion": False,
                "foreground_strategy": ForegroundStrategy.PRESERVE_PIXELS,
                "use_local_inpainting": True,
            },
            "remove the person on the left and replace background with sunset": {
                "background_type": BackgroundType.GENERATED_SCENE,
                "use_diffusion": True,
                "preserve_all_people": False,
                "remove_people": True,
                "background_strategy": BackgroundStrategy.SD_GENERATION,
            },
            "passport background": {
                "background_type": BackgroundType.SOLID_COLOR,
                "use_diffusion": False,
                "use_color_compositing": True,
                "preserve_all_people": True,
            }
        }

        for cmd, expected_props in snapshots.items():
            intent = parse_prompt(cmd)
            plan = build_execution_plan(intent)

            for prop, expected_val in expected_props.items():
                actual_val = getattr(plan, prop)
                self.assertEqual(
                    actual_val, expected_val,
                    f"Snapshot regression detected for command '{cmd}'. Property '{prop}' expected {expected_val}, got {actual_val}"
                )
