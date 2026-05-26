"""
Unit tests for the open-vocabulary SemanticDetector abstraction layer.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from src.models.detection import SemanticDetector, DetectionResult


class TestSemanticDetector(unittest.TestCase):
    """Test suite for the unified SemanticDetector."""

    def test_detection_result_dataclass(self):
        res = DetectionResult(box=(10, 20, 30, 40), label="cup", confidence=0.88)
        self.assertEqual(res.box, (10, 20, 30, 40))
        self.assertEqual(res.label, "cup")
        self.assertEqual(res.confidence, 0.88)

    @patch("transformers.Owlv2Processor")
    @patch("transformers.Owlv2ForObjectDetection")
    def test_owlv2_detection(self, MockModel, MockProcessor):
        # Setup mocks
        mock_proc_instance = MockProcessor.from_pretrained.return_value
        mock_model_instance = MockModel.from_pretrained.return_value

        # Mock processor return value (inputs dictionary-like)
        mock_proc_instance.return_value = {"pixel_values": MagicMock()}

        # Mock post-process outputs
        import torch
        mock_post_process_res = {
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 110.0]])
        }
        mock_proc_instance.post_process_object_detection.return_value = [mock_post_process_res]

        detector = SemanticDetector()
        detector._owlv2_processor = mock_proc_instance
        detector._owlv2_model = mock_model_instance.to.return_value

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        res = detector.detect(prompt="lanyard", image=image, model_type="owlv2")

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].box, (10, 20, 100, 110))
        self.assertEqual(res[0].label, "lanyard")
        self.assertAlmostEqual(res[0].confidence, 0.95)

    @patch("transformers.GroundingDinoProcessor")
    @patch("transformers.GroundingDinoForObjectDetection")
    def test_groundingdino_detection(self, MockModel, MockProcessor):
        # Setup mocks
        mock_proc_instance = MockProcessor.from_pretrained.return_value
        mock_model_instance = MockModel.from_pretrained.return_value

        inputs_mock = MagicMock()
        inputs_mock.input_ids = MagicMock()
        inputs_mock.to.return_value = inputs_mock
        mock_proc_instance.return_value = inputs_mock

        # Mock post-process outputs
        import torch
        mock_post_process_res = {
            "scores": torch.tensor([0.87]),
            "labels": ["badge"],
            "boxes": torch.tensor([[5.0, 15.0, 80.0, 90.0]])
        }
        mock_proc_instance.post_process_grounded_object_detection.return_value = [mock_post_process_res]

        detector = SemanticDetector()
        detector._gd_processor = mock_proc_instance
        detector._gd_model = mock_model_instance.to.return_value

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        res = detector.detect(prompt="badge", image=image, model_type="groundingdino")

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].box, (5, 15, 80, 90))
        self.assertEqual(res[0].label, "badge")
        self.assertAlmostEqual(res[0].confidence, 0.87)

    @patch("transformers.AutoProcessor")
    @patch("transformers.AutoModelForCausalLM")
    def test_florence_detection(self, MockModel, MockProcessor):
        # Setup mocks
        mock_proc_instance = MockProcessor.from_pretrained.return_value
        mock_model_instance = MockModel.from_pretrained.return_value

        inputs_mock = MagicMock()
        inputs_mock.to.return_value = inputs_mock
        mock_proc_instance.return_value = inputs_mock

        mock_model_instance.dtype = torch.float32

        # Mock generate decodes
        mock_proc_instance.batch_decode.return_value = ["<CAPTION_TO_PHRASE_GROUNDING>dummy"]

        # Mock post-process generation
        mock_post_process_res = {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "boxes": [[15, 25, 95, 105]],
                "labels": ["tissue"]
            }
        }
        mock_proc_instance.post_process_generation.return_value = mock_post_process_res

        detector = SemanticDetector()
        detector._florence_processor = mock_proc_instance
        detector._florence_model = mock_model_instance.to.return_value

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        res = detector.detect(prompt="tissue", image=image, model_type="florence2")

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].box, (15, 25, 95, 105))
        self.assertEqual(res[0].label, "tissue")
        self.assertAlmostEqual(res[0].confidence, 1.0)

    @patch("transformers.Owlv2Processor")
    @patch("transformers.Owlv2ForObjectDetection")
    def test_fallback_behavior(self, MockModel, MockProcessor):
        # Setup OWLv2 model failure to trigger fallback to other models
        mock_proc_instance = MockProcessor.from_pretrained.return_value
        mock_model_instance = MockModel.from_pretrained.return_value

        # Make OWLv2 raise an exception
        mock_proc_instance.side_effect = RuntimeError("Failed loading OWLv2")

        # Mock GroundingDINO to succeed instead
        import torch
        mock_gd_proc = MagicMock()
        mock_gd_model = MagicMock()
        inputs_mock = MagicMock()
        inputs_mock.input_ids = MagicMock()
        inputs_mock.to.return_value = inputs_mock
        mock_gd_proc.return_value = inputs_mock

        mock_post_process_res = {
            "scores": torch.tensor([0.90]),
            "labels": ["badge"],
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        }
        mock_gd_proc.post_process_grounded_object_detection.return_value = [mock_post_process_res]

        detector = SemanticDetector()
        detector._owlv2_processor = mock_proc_instance
        detector._owlv2_model = mock_model_instance.to.return_value
        detector._gd_processor = mock_gd_proc
        detector._gd_model = mock_gd_model.to.return_value

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        # Try loading and running owlv2, should fallback to groundingdino
        res = detector.detect(prompt="badge", image=image, model_type="owlv2")

        # GroundingDINO succeeded
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].box, (10, 10, 50, 50))
        self.assertEqual(res[0].label, "badge")


if __name__ == "__main__":
    unittest.main()
