"""
Hybrid Segmentation Engine for SceneShift.

Implements two segmentation modes:
  - Automatic: YOLOv8x-seg detects and segments the largest foreground object.
  - Interactive: SAM2 (Segment Anything Model 2) provides click-based segmentation.

Both modes return a binary mask, bounding box, and class label.
Models are cached in memory for fast repeated inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from src.utils.device import get_device, get_dtype, clear_gpu_cache


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    """Output of the segmentation engine."""
    mask: np.ndarray                       # Binary mask (H x W, uint8, 0/255)
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2) in pixels
    label: str                             # Object class label
    confidence: float                      # Detection confidence [0, 1]
    mode: str                              # "auto" | "interactive"
    inference_time_s: float                # Wall-clock inference seconds
    all_masks: List[np.ndarray] = field(default_factory=list)  # All candidate masks


# ── YOLO Automatic Segmentor ──────────────────────────────────────────────────

class YOLOSegmentor:
    """
    Automatic segmentation using YOLOv8x-seg.

    Detects the highest-confidence foreground object and returns
    its segmentation mask, bounding box, and class label.

    Target: < 0.2 s on NVIDIA RTX 3080.
    """

    MODEL_ID = "yolov8x-seg.pt"

    def __init__(self, conf_threshold: float = 0.35, device: Optional[torch.device] = None):
        """
        Args:
            conf_threshold: Minimum YOLO confidence score.
            device: Target device; auto-detected if None.
        """
        self.conf_threshold = conf_threshold
        self.device = device or get_device()
        self._model = None  # Lazy-loaded

    def _load_model(self) -> None:
        """Load YOLOv8x-seg model (cached after first call)."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8x-seg on {self.device}…")
            t0 = time.perf_counter()
            self._model = YOLO(self.MODEL_ID)
            elapsed = time.perf_counter() - t0
            logger.info(f"YOLOv8x-seg loaded in {elapsed:.2f}s")
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def segment(self, image_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """
        Run automatic segmentation on an RGB image.

        Args:
            image_rgb: Input image (H x W x 3, uint8, RGB).

        Returns:
            SegmentationResult for the largest detected object,
            or None if no object is found above the confidence threshold.
        """
        self._load_model()
        h, w = image_rgb.shape[:2]

        t0 = time.perf_counter()

        # Convert RGB → BGR for YOLO
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        device_str = str(self.device) if self.device.type != "cpu" else "cpu"

        results = self._model(
            image_bgr,
            conf=self.conf_threshold,
            device=device_str,
            verbose=False,
        )

        elapsed = time.perf_counter() - t0

        # ── Parse results ──────────────────────────────────────────────────────
        if not results or results[0].masks is None:
            logger.warning("YOLO: no segmentation masks returned.")
            return None

        result = results[0]
        boxes = result.boxes
        masks_data = result.masks.data  # Tensor (N, H', W')
        class_names = result.names

        if len(boxes) == 0:
            logger.warning("YOLO: no objects detected.")
            return None

        # Select largest object by mask area
        areas = masks_data.sum(dim=(1, 2))
        best_idx = int(areas.argmax())

        # Resize mask to original resolution
        raw_mask = masks_data[best_idx].cpu().numpy()
        mask_full = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_full > 0.5).astype(np.uint8) * 255

        # Bounding box
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        # Class label + confidence
        cls_id = int(boxes.cls[best_idx].item())
        label = class_names.get(cls_id, f"class_{cls_id}")
        conf = float(boxes.conf[best_idx].item())

        # Collect all masks
        all_masks = []
        for i in range(len(masks_data)):
            m = masks_data[i].cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            all_masks.append((m > 0.5).astype(np.uint8) * 255)

        logger.info(
            f"YOLO detected '{label}' (conf={conf:.2f}) in {elapsed:.3f}s. "
            f"Bbox: ({x1},{y1})→({x2},{y2})"
        )

        return SegmentationResult(
            mask=binary_mask,
            bbox=(x1, y1, x2, y2),
            label=label,
            confidence=conf,
            mode="auto",
            inference_time_s=elapsed,
            all_masks=all_masks,
        )


# ── SAM2 Interactive Segmentor ────────────────────────────────────────────────

class SAM2Segmentor:
    """
    Interactive segmentation using SAM2 (Segment Anything Model 2).

    The user provides one or more click points on the target object.
    SAM2 generates a high-confidence segmentation mask.

    Target: < 1 s on NVIDIA RTX 3080.
    Falls back to simulated circular mask on systems without SAM2.
    """

    SAM2_CHECKPOINT = "sam2_hiera_large.pt"
    SAM2_CONFIG = "sam2_hiera_l.yaml"

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._predictor = None  # Lazy-loaded

    def _load_model(self) -> None:
        """Load SAM2 predictor (cached after first call)."""
        if self._predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"Loading SAM2 on {self.device}…")
            t0 = time.perf_counter()
            sam2_model = build_sam2(
                self.SAM2_CONFIG,
                self.SAM2_CHECKPOINT,
                device=self.device,
            )
            self._predictor = SAM2ImagePredictor(sam2_model)
            elapsed = time.perf_counter() - t0
            logger.info(f"SAM2 loaded in {elapsed:.2f}s")

        except ImportError:
            logger.warning(
                "SAM2 not installed. Using fallback circular-mask segmentation. "
                "Install: pip install git+https://github.com/facebookresearch/sam2.git"
            )
            self._predictor = "fallback"

    def segment(
        self,
        image_rgb: np.ndarray,
        click_points: List[Tuple[int, int]],
        click_labels: Optional[List[int]] = None,
    ) -> SegmentationResult:
        """
        Segment object at specified click point(s).

        Args:
            image_rgb: Input image (H x W x 3, uint8, RGB).
            click_points: List of (x, y) pixel coordinates (foreground clicks).
            click_labels: 1=foreground, 0=background per point.
                          Defaults to all foreground.

        Returns:
            SegmentationResult with the best SAM2 mask.
        """
        self._load_model()
        h, w = image_rgb.shape[:2]

        if click_labels is None:
            click_labels = [1] * len(click_points)

        t0 = time.perf_counter()

        # ── SAM2 real inference ────────────────────────────────────────────────
        if self._predictor != "fallback":
            self._predictor.set_image(image_rgb)

            points_np = np.array(click_points, dtype=np.float32)
            labels_np = np.array(click_labels, dtype=np.int32)

            masks, scores, _ = self._predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True,
            )

            # Select mask with highest IoU score
            best_idx = int(np.argmax(scores))
            binary_mask = (masks[best_idx] > 0).astype(np.uint8) * 255
            confidence = float(scores[best_idx])
            all_masks = [(m > 0).astype(np.uint8) * 255 for m in masks]

        else:
            # ── Fallback: elliptical mask around click centroid ──────────────
            logger.info("Using fallback elliptical mask segmentation.")
            cx = int(np.mean([p[0] for p in click_points]))
            cy = int(np.mean([p[1] for p in click_points]))
            radius_x = max(50, w // 5)
            radius_y = max(50, h // 5)
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                binary_mask,
                (cx, cy),
                (radius_x, radius_y),
                0, 0, 360,
                255,
                -1,
            )
            confidence = 0.75
            all_masks = [binary_mask]

        elapsed = time.perf_counter() - t0

        # Compute bounding box from mask
        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

        logger.info(
            f"SAM2 segmented at {click_points} in {elapsed:.3f}s "
            f"(conf={confidence:.2f})"
        )

        return SegmentationResult(
            mask=binary_mask,
            bbox=(x1, y1, x2, y2),
            label="interactive_object",
            confidence=confidence,
            mode="interactive",
            inference_time_s=elapsed,
            all_masks=all_masks,
        )


# ── Unified Segmentation Engine ───────────────────────────────────────────────

class SegmentationEngine:
    """
    Unified interface combining YOLOv8 (auto) and SAM2 (interactive) modes.
    Models are lazily loaded and cached for the lifetime of the instance.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._yolo = YOLOSegmentor(device=self.device)
        self._sam2 = SAM2Segmentor(device=self.device)

    def auto_segment(self, image_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """Automatic mode: YOLO detects the largest foreground object."""
        return self._yolo.segment(image_rgb)

    def interactive_segment(
        self,
        image_rgb: np.ndarray,
        click_points: List[Tuple[int, int]],
        click_labels: Optional[List[int]] = None,
    ) -> SegmentationResult:
        """Interactive mode: SAM2 segments at specified click points."""
        return self._sam2.segment(image_rgb, click_points, click_labels)

    def preload_models(self) -> None:
        """Eagerly load all models into GPU memory at startup."""
        logger.info("Preloading segmentation models…")
        self._yolo._load_model()
        self._sam2._load_model()
        logger.info("Segmentation models ready.")
