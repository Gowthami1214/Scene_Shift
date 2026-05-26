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
    alpha_matte: Optional[np.ndarray] = None  # Soft alpha (H x W, float32, [0,1])


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
        except Exception as exc:
            logger.warning(
                f"YOLOv8x-seg not available ({exc}). Using fallback circular-mask segmentation."
            )
            self._model = "fallback"

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

        if self._model == "fallback":
            logger.info("Using fallback circular-mask YOLO segmentation.")
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(binary_mask, (w // 2, h // 2), min(w, h) // 4, 255, -1)
            x1, y1 = w // 4, h // 4
            x2, y2 = w * 3 // 4, h * 3 // 4
            return SegmentationResult(
                mask=binary_mask,
                bbox=(x1, y1, x2, y2),
                label="person",
                confidence=0.85,
                mode="auto",
                inference_time_s=time.perf_counter() - t0,
                all_masks=[binary_mask],
            )

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

        # Find all detections of class "person"
        person_indices = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            label = class_names.get(cls_id, f"class_{cls_id}")
            if label == "person":
                person_indices.append(i)

        # Collect all masks first
        all_masks = []
        for i in range(len(masks_data)):
            m = masks_data[i].cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            all_masks.append((m > 0.5).astype(np.uint8) * 255)

        if person_indices:
            # Union of all person masks
            union_mask = np.zeros((h, w), dtype=np.uint8)
            bboxes = []
            confidences = []
            
            for idx in person_indices:
                union_mask = cv2.bitwise_or(union_mask, all_masks[idx])
                xyxy = boxes.xyxy[idx].cpu().numpy()
                bboxes.append([int(v) for v in xyxy])
                confidences.append(float(boxes.conf[idx].item()))
                
            binary_mask = union_mask
            bboxes = np.array(bboxes)
            x1 = int(bboxes[:, 0].min())
            y1 = int(bboxes[:, 1].min())
            x2 = int(bboxes[:, 2].max())
            y2 = int(bboxes[:, 3].max())
            
            label = "person"
            conf = float(np.mean(confidences))
            
            logger.info(f"Detected people: {len(person_indices)}")
            logger.info(f"Foreground masks merged: True")
            logger.info(f"Preserve all humans: True")
        else:
            # Fall back to largest detected object by area
            areas = masks_data.sum(dim=(1, 2))
            best_idx = int(areas.argmax())

            binary_mask = all_masks[best_idx]
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            cls_id = int(boxes.cls[best_idx].item())
            label = class_names.get(cls_id, f"class_{cls_id}")
            conf = float(boxes.conf[best_idx].item())

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


# ── BiRefNet Alpha Matting ─────────────────────────────────────────────────────

class BiRefNetMatting:
    """High-quality alpha matting using BiRefNet for sub-pixel edge quality.

    Refines coarse YOLO masks to production-quality alpha mattes,
    especially for hair, fur, transparent objects, and fine edges.
    """
    MODEL_ID = "ZhengPeng7/BiRefNet"

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._model = None
        self._transform = None

    def _load_model(self) -> None:
        """Lazy-load BiRefNet model from HuggingFace."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms

            logger.info(f"Loading BiRefNet matting model '{self.MODEL_ID}'...")
            t0 = time.perf_counter()

            self._model = AutoModelForImageSegmentation.from_pretrained(
                self.MODEL_ID, trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()

            self._transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            elapsed = time.perf_counter() - t0
            logger.info(f"BiRefNet loaded in {elapsed:.2f}s")
        except Exception as exc:
            logger.warning(f"BiRefNet not available ({exc}). Falling back to mask refinement only.")
            self._model = "fallback"

    def refine(self, image_rgb: np.ndarray, coarse_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Produce high-quality alpha matte from image.

        Args:
            image_rgb: Input RGB image (H x W x 3, uint8).
            coarse_mask: Optional coarse mask to guide cropping.

        Returns:
            Alpha matte (H x W, float32, [0, 1]) with sub-pixel edges.
        """
        self._load_model()

        if self._model == "fallback":
            # Return coarse mask as float if BiRefNet unavailable
            if coarse_mask is not None:
                return (coarse_mask / 255.0).astype(np.float32) if coarse_mask.max() > 1 else coarse_mask.astype(np.float32)
            return np.ones(image_rgb.shape[:2], dtype=np.float32)

        from PIL import Image as PILImage

        h, w = image_rgb.shape[:2]
        pil_img = PILImage.fromarray(image_rgb)

        input_tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self._model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        # Resize prediction back to original size
        pred_np = pred.numpy()
        alpha = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        # If coarse mask provided, use it to constrain the alpha matte
        # This prevents BiRefNet from picking up other objects
        if coarse_mask is not None:
            coarse_float = (coarse_mask / 255.0).astype(np.float32) if coarse_mask.max() > 1 else coarse_mask.astype(np.float32)
            # Dilate coarse mask to give BiRefNet some room for edge refinement
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            dilated = cv2.dilate(coarse_float, kernel, iterations=1)
            alpha = alpha * dilated

        logger.info(f"BiRefNet alpha matte generated: shape={alpha.shape}, range=[{alpha.min():.3f}, {alpha.max():.3f}]")
        return alpha


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
        self._matting = BiRefNetMatting(device=self.device)

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

    def segment_with_matting(
        self,
        image_rgb: np.ndarray,
        target_class: Optional[str] = None,
    ) -> Optional[SegmentationResult]:
        """Full segmentation pipeline: YOLO detection + BiRefNet alpha matting.

        Args:
            image_rgb: Input RGB image (H x W x 3, uint8).
            target_class: Optional class name to filter (e.g., 'person').

        Returns:
            SegmentationResult with soft alpha mask from BiRefNet.
        """
        t0 = time.perf_counter()

        # Step 1: YOLO detection for coarse mask + bounding box
        yolo_result = self.auto_segment(image_rgb)
        if yolo_result is None:
            logger.warning("YOLO detected nothing. Running BiRefNet on full image.")
            alpha = self._matting.refine(image_rgb)
            binary = (alpha > 0.5).astype(np.uint8) * 255
            h, w = image_rgb.shape[:2]
            return SegmentationResult(
                mask=binary, bbox=(0, 0, w, h), label="unknown",
                confidence=0.5, mode="auto+matting",
                inference_time_s=time.perf_counter() - t0, all_masks=[],
                alpha_matte=alpha,
            )

        # Filter by target class if specified
        if target_class and yolo_result.label.lower() != target_class.lower():
            logger.info(f"YOLO detected '{yolo_result.label}' but target is '{target_class}'. Running BiRefNet on full image.")
            alpha = self._matting.refine(image_rgb)
        else:
            # Step 2: BiRefNet refinement using YOLO coarse mask
            alpha = self._matting.refine(image_rgb, coarse_mask=yolo_result.mask)

        binary = (alpha > 0.5).astype(np.uint8) * 255
        elapsed = time.perf_counter() - t0

        return SegmentationResult(
            mask=binary, bbox=yolo_result.bbox, label=yolo_result.label,
            confidence=yolo_result.confidence, mode="auto+matting",
            inference_time_s=elapsed, all_masks=[],
            alpha_matte=alpha,
        )

    def preload_models(self) -> None:
        """Eagerly load all models into GPU memory at startup."""
        logger.info("Preloading segmentation models…")
        self._yolo._load_model()
        self._sam2._load_model()
        self._matting._load_model()
        logger.info("Segmentation models ready.")
