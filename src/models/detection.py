"""
Open Vocabulary Object Detection Module for SceneShift.

Integrates SOTA zero-shot detectors:
  1. OWLv2 (google/owlv2-base-patch16-ensemble)
  2. GroundingDINO (IDEA-Research/groundingdino-tiny)
  3. Florence-2 (microsoft/Florence-2-base)

Provides a unified interface with automatic fallbacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image

from src.utils.device import get_device
from src.utils.image_io import numpy_to_pil


@dataclass
class DetectionResult:
    """Standardized output of the semantic detection abstraction layer."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    label: str
    confidence: float


class SemanticDetector:
    """
    Unified Open-Vocabulary Semantic Object Detector.

    Supports querying arbitrary objects using text descriptions,
    handling Florence-2, GroundingDINO, and OWLv2 with seamless fallback.
    """

    OWLV2_MODEL_ID = "google/owlv2-base-patch16-ensemble"
    DINO_MODEL_ID = "IDEA-Research/groundingdino-tiny"
    FLORENCE_MODEL_ID = "microsoft/Florence-2-base"

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._owlv2_processor = None
        self._owlv2_model = None
        self._gd_processor = None
        self._gd_model = None
        self._florence_processor = None
        self._florence_model = None

    def _move_inputs(self, inputs):
        """Move HF processor outputs to the active device across output types."""
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def _model_dtype(self, model) -> torch.dtype:
        """Return a usable model dtype, falling back when mocks omit it."""
        dtype = getattr(model, "dtype", None)
        return dtype if isinstance(dtype, torch.dtype) else torch.float32

    def _load_owlv2(self) -> None:
        """Lazy-load OWLv2 model."""
        if self._owlv2_model is not None:
            return
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        logger.info(f"Loading OWLv2 model '{self.OWLV2_MODEL_ID}' on {self.device}...")
        t0 = time.perf_counter()
        self._owlv2_processor = Owlv2Processor.from_pretrained(self.OWLV2_MODEL_ID)
        self._owlv2_model = Owlv2ForObjectDetection.from_pretrained(self.OWLV2_MODEL_ID).to(self.device)
        logger.info(f"OWLv2 loaded in {time.perf_counter() - t0:.2f}s")

    def _load_groundingdino(self) -> None:
        """Lazy-load GroundingDINO model."""
        if self._gd_model is not None:
            return
        from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
        logger.info(f"Loading GroundingDINO model '{self.DINO_MODEL_ID}' on {self.device}...")
        t0 = time.perf_counter()
        self._gd_processor = GroundingDinoProcessor.from_pretrained(self.DINO_MODEL_ID)
        self._gd_model = GroundingDinoForObjectDetection.from_pretrained(self.DINO_MODEL_ID).to(self.device)
        logger.info(f"GroundingDINO loaded in {time.perf_counter() - t0:.2f}s")

    def _load_florence(self) -> None:
        """Lazy-load Florence-2 model."""
        if self._florence_model is not None:
            return
        from transformers import AutoProcessor, AutoModelForCausalLM
        logger.info(f"Loading Florence-2 model '{self.FLORENCE_MODEL_ID}' on {self.device}...")
        t0 = time.perf_counter()
        self._florence_processor = AutoProcessor.from_pretrained(self.FLORENCE_MODEL_ID, trust_remote_code=True)
        # Use dtype float16/bfloat16 if GPU is available to improve performance
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._florence_model = AutoModelForCausalLM.from_pretrained(
            self.FLORENCE_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        ).to(self.device)
        logger.info(f"Florence-2 loaded in {time.perf_counter() - t0:.2f}s")

    def detect(
        self,
        prompt: str,
        image: np.ndarray,
        model_type: str = "owlv2",
        threshold: float = 0.15,
    ) -> List[DetectionResult]:
        """
        Detect objects in an image using open-vocabulary text query.

        Args:
            prompt:     The query description (e.g. "lanyard", "face").
            image:      RGB image as a numpy array (H x W x 3).
            model_type: "owlv2" | "groundingdino" | "florence2".
            threshold:  Detection confidence threshold.

        Returns:
            List of DetectionResult.
        """
        pil_img = numpy_to_pil(image)
        h, w = image.shape[:2]

        order = ["owlv2", "groundingdino", "florence2"]
        # Prioritize the requested model type
        if model_type in order:
            order.remove(model_type)
            order.insert(0, model_type)

        errors = []
        for model in order:
            try:
                if model == "owlv2":
                    return self._detect_owlv2(pil_img, prompt, threshold)
                elif model == "groundingdino":
                    return self._detect_groundingdino(pil_img, prompt, threshold)
                elif model == "florence2":
                    return self._detect_florence(pil_img, prompt, threshold)
            except Exception as exc:
                logger.warning(f"Model {model} detection failed: {exc}. Trying fallback...")
                errors.append(f"{model}: {exc}")

        logger.error(f"All open-vocabulary detectors failed: {errors}")
        return []

    def _detect_owlv2(self, pil_img: Image.Image, prompt: str, threshold: float) -> List[DetectionResult]:
        self._load_owlv2()
        w, h = pil_img.size

        inputs = self._move_inputs(
            self._owlv2_processor(text=[[prompt]], images=pil_img, return_tensors="pt")
        )
        with torch.no_grad():
            outputs = self._owlv2_model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(self.device)
        results = self._owlv2_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(v.item()) for v in box]
            bx1, by1, bx2, by2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            detections.append(DetectionResult(
                box=(bx1, by1, bx2, by2),
                label=prompt,
                confidence=float(score.item())
            ))
        return detections

    def _detect_groundingdino(self, pil_img: Image.Image, prompt: str, threshold: float) -> List[DetectionResult]:
        self._load_groundingdino()
        w, h = pil_img.size

        # GroundingDINO query must end with a period
        text = prompt.lower().strip()
        if not text.endswith("."):
            text += "."

        inputs = self._move_inputs(
            self._gd_processor(images=pil_img, text=text, return_tensors="pt")
        )
        with torch.no_grad():
            outputs = self._gd_model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(self.device)
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs.get("input_ids")
        results = self._gd_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(v.item()) for v in box]
            bx1, by1, bx2, by2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            detections.append(DetectionResult(
                box=(bx1, by1, bx2, by2),
                label=label if label else prompt,
                confidence=float(score.item())
            ))
        return detections

    def _detect_florence(self, pil_img: Image.Image, prompt: str, threshold: float) -> List[DetectionResult]:
        self._load_florence()
        w, h = pil_img.size

        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        input_text = task_prompt + prompt

        # Support float16/float32 depending on model dtype
        model_dtype = self._model_dtype(self._florence_model)
        inputs = self._florence_processor(text=input_text, images=pil_img, return_tensors="pt")
        inputs = self._move_inputs(inputs)
        if isinstance(inputs, dict) and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

        with torch.no_grad():
            generated_ids = self._florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

        generated_text = self._florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed_answer = self._florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(w, h)
        )

        detections = []
        grounding_results = parsed_answer.get(task_prompt, {})
        boxes = grounding_results.get("boxes", [])
        labels = grounding_results.get("labels", [])

        for box, label in zip(boxes, labels):
            # Florence-2 returns normalized coords or pixel coords depending on processor postprocess
            bx1, by1, bx2, by2 = [int(v) for v in box]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            detections.append(DetectionResult(
                box=(bx1, by1, bx2, by2),
                label=label,
                confidence=1.0  # Florence-2 phrase grounding does not return continuous scores natively
            ))
        return detections
