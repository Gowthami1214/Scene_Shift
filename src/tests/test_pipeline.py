"""
End-to-end pipeline tests for SceneShift.
Tests the full orchestration pipeline with mocked AI models.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    ProgressEvent,
)
from src.pipeline.shadow import ShadowSynthesizer
from src.pipeline.harmonization import ColorHarmonizer
from src.pipeline.mask_utils import refine_mask


def _make_image(h=128, w=128) -> np.ndarray:
    """Random RGB image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_mask(h=128, w=128, radius=40) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
    return mask


class TestPipelineStageEnum(unittest.TestCase):
    """Tests for PipelineStage enumeration."""

    def test_all_stages_present(self):
        stages = [s.value for s in PipelineStage]
        for expected in ["idle", "segmenting", "refining_mask", "editing_object",
                         "generating_background", "compositing", "done", "error"]:
            self.assertIn(expected, stages)


class TestProgressEvent(unittest.TestCase):
    """Tests for ProgressEvent dataclass."""

    def test_valid_construction(self):
        ev = ProgressEvent(stage="segmenting", progress=0.15,
                           message="Running YOLO", elapsed_s=0.5)
        self.assertEqual(ev.stage, "segmenting")
        self.assertAlmostEqual(ev.progress, 0.15)

    def test_progress_range_values(self):
        """Progress can be 0.0 → 1.0."""
        for pct in [0.0, 0.5, 1.0]:
            ev = ProgressEvent("done", pct, "ok")
            self.assertEqual(ev.progress, pct)


class TestShadowSynthesis(unittest.TestCase):
    """Tests for the shadow synthesis module."""

    def setUp(self):
        self.synth = ShadowSynthesizer()
        self.image = _make_image(128, 128)
        self.mask = _make_mask(128, 128, 40)

    def test_shadow_output_shape(self):
        result = self.synth.synthesize(self.image, self.mask)
        self.assertEqual(result.image_with_shadow.shape, (128, 128, 3))

    def test_shadow_output_dtype(self):
        result = self.synth.synthesize(self.image, self.mask)
        self.assertEqual(result.image_with_shadow.dtype, np.uint8)

    def test_shadow_pixel_range(self):
        result = self.synth.synthesize(self.image, self.mask)
        self.assertGreaterEqual(result.image_with_shadow.min(), 0)
        self.assertLessEqual(result.image_with_shadow.max(), 255)

    def test_shadow_layer_4_channel(self):
        result = self.synth.synthesize(self.image, self.mask)
        self.assertEqual(result.shadow_layer.shape, (128, 128, 4))

    def test_zero_opacity_minimal_change(self):
        """With near-zero opacity, shadow should barely change the image."""
        result = self.synth.synthesize(self.image, self.mask, opacity=0.01)
        diff = np.abs(result.image_with_shadow.astype(int) - self.image.astype(int))
        self.assertLess(diff.mean(), 20.0)

    def test_contact_shadow(self):
        result = self.synth.add_contact_shadow(self.image, self.mask)
        self.assertEqual(result.shape, (128, 128, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_timing_recorded(self):
        result = self.synth.synthesize(self.image, self.mask)
        self.assertGreater(result.synthesis_time_s, 0.0)


class TestColorHarmonization(unittest.TestCase):
    """Tests for LAB color harmonization."""

    def setUp(self):
        self.harmonizer = ColorHarmonizer()
        self.image = _make_image(128, 128)
        self.bg = _make_image(128, 128)
        mask = _make_mask(128, 128, 40)
        self.alpha = (mask / 255.0).astype(np.float32)

    def test_output_shape(self):
        result = self.harmonizer.harmonize(self.image, self.bg, self.alpha)
        self.assertEqual(result.harmonized.shape, (128, 128, 3))

    def test_output_dtype(self):
        result = self.harmonizer.harmonize(self.image, self.bg, self.alpha)
        self.assertEqual(result.harmonized.dtype, np.uint8)

    def test_output_pixel_range(self):
        result = self.harmonizer.harmonize(self.image, self.bg, self.alpha)
        self.assertGreaterEqual(result.harmonized.min(), 0)
        self.assertLessEqual(result.harmonized.max(), 255)

    def test_harmonization_time_recorded(self):
        result = self.harmonizer.harmonize(self.image, self.bg, self.alpha)
        self.assertGreater(result.harmonization_time_s, 0.0)

    def test_different_bg_size_handled(self):
        """Background of different size should be auto-resized."""
        bg_large = _make_image(256, 256)  # Different size
        result = self.harmonizer.harmonize(self.image, bg_large, self.alpha)
        self.assertEqual(result.harmonized.shape, (128, 128, 3))

    def test_zero_alpha_mask(self):
        """All-zero alpha (no foreground) should not raise."""
        alpha_zero = np.zeros((128, 128), dtype=np.float32)
        result = self.harmonizer.harmonize(self.image, self.bg, alpha_zero)
        self.assertIsNotNone(result.harmonized)


class TestPipelineRequest(unittest.TestCase):
    """Tests for PipelineRequest dataclass."""

    def test_defaults(self):
        image = _make_image(64, 64)
        req = PipelineRequest(image=image)
        self.assertEqual(req.style_preset, "Realistic")
        self.assertEqual(req.blend_mode, "alpha")
        self.assertEqual(req.segmentation_mode, "auto")
        self.assertAlmostEqual(req.strength, 0.85)
        self.assertTrue(req.enable_shadow)
        self.assertTrue(req.enable_harmonization)

    def test_custom_params(self):
        image = _make_image(64, 64)
        req = PipelineRequest(
            image=image,
            object_prompt="golden dragon",
            background_prompt="fantasy forest",
            style_preset="Fantasy",
            blend_mode="laplacian",
            strength=0.7,
            seed=42,
        )
        self.assertEqual(req.style_preset, "Fantasy")
        self.assertEqual(req.blend_mode, "laplacian")
        self.assertEqual(req.seed, 42)


class TestOrchestratorMocked(unittest.TestCase):
    """
    Integration test for the pipeline orchestrator using mocked AI models.
    Verifies the orchestration flow without requiring GPU or model downloads.
    """

    @patch("src.pipeline.orchestrator.SegmentationEngine")
    @patch("src.pipeline.orchestrator.ObjectEditor")
    @patch("src.pipeline.orchestrator.BackgroundGenerator")
    def test_pipeline_completes(
        self,
        MockBG,
        MockEditor,
        MockSeg,
    ):
        """Full pipeline should complete and return a PipelineResult."""
        h, w = 64, 64
        dummy_image = _make_image(h, w)
        dummy_mask = _make_mask(h, w, 20)

        # Mock segmentation
        from src.models.segmentation import SegmentationResult
        mock_seg = MockSeg.return_value
        mock_seg.auto_segment.return_value = SegmentationResult(
            mask=dummy_mask,
            bbox=(10, 10, 50, 50),
            label="cat",
            confidence=0.95,
            mode="auto",
            inference_time_s=0.1,
        )

        # Mock object editor
        from src.models.editing import EditingResult
        mock_editor = MockEditor.return_value
        mock_editor.edit.return_value = EditingResult(
            edited_image=dummy_image,
            edited_crop=dummy_image[10:50, 10:50],
            prompt_used="test prompt",
            style_preset="Realistic",
            inference_time_s=2.5,
        )

        # Mock background generator
        from src.models.background import BackgroundResult
        mock_bg = MockBG.return_value
        mock_bg.generate.return_value = BackgroundResult(
            image=dummy_image,
            method="procedural",
            prompt_used="test bg",
            inference_time_s=0.1,
        )

        orch = PipelineOrchestrator()
        # Replace instances with mocks
        orch._segmentation = mock_seg
        orch._editor = mock_editor
        orch._bg_gen = mock_bg

        image = _make_image(h, w)
        req = PipelineRequest(
            image=image,
            object_prompt="a cat",
            background_prompt="green field",
            style_preset="Realistic",
            blend_mode="alpha",
            num_steps=5,
            enable_shadow=True,
            enable_harmonization=True,
            use_sd_background=False,  # Use procedural
        )

        progress_events = []
        def track_progress(event):
            progress_events.append(event)

        result = orch.run(req, job_id="test_job", progress_callback=track_progress)

        self.assertIsInstance(result, PipelineResult)
        self.assertIsNotNone(result.final_image)
        self.assertEqual(result.final_image.shape[2], 3)
        self.assertGreater(result.total_time_s, 0.0)
        self.assertGreater(len(progress_events), 0, "Progress callback should be called.")
        self.assertIn("segmentation", result.timings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
