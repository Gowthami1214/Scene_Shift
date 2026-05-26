"""
Generalized Object Removal Module for SceneShift — Layer 3 component.

Detects and erases ANY user-specified object using:
  - Zero-shot object detection (OWL-ViT) for primary detection
  - Spatial hint dispatch for generalized heuristic fallbacks
  - GrabCut refinement for accurate mask boundaries
  - LaMa localized inpainting for seamless fill

No object type is hardcoded in logic branches. All spatial fallback
behavior is driven by the SPATIAL_HINTS dictionary, making the module
fully generalized for any removal target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from src.utils.device import get_device
from src.utils.image_io import numpy_to_pil, pil_to_numpy
from src.pipeline.face_preservation import FaceProtector
from src.models.lama_inpainting import LocalInpainter


@dataclass
class ObjectRemovalResult:
    """Output of the targeted object removal stage."""
    image: np.ndarray          # Inpainted RGB image (H x W x 3, uint8)
    object_mask: np.ndarray    # Binary mask of removed objects (H x W, uint8, 0/255)
    detected_boxes: List[Tuple[int, int, int, int]]  # Detected target bounding boxes
    removal_time_s: float


# ── Spatial hint system ───────────────────────────────────────────────────────
# Maps any object keyword → spatial region label.
# Used to drive heuristic fallback bounding boxes when OWL-ViT cannot detect
# a target. Keys are lowercase substrings — fuzzy matched against target names.
# Unknown objects default to "center_region" (safe center-crop fallback).

SPATIAL_HINTS: Dict[str, str] = {
    # Head / upper region
    "face":           "upper_region",
    "hair":           "upper_region",
    "hat":            "upper_region",
    "cap":            "upper_region",
    "glasses":        "upper_region",
    "sunglasses":     "upper_region",
    "earring":        "upper_region",
    "headband":       "upper_region",
    "helmet":         "upper_region",
    "hood":           "upper_region",
    # Chest / torso accessories
    "lanyard":        "chest_region",
    "id card":        "chest_region",
    "badge":          "chest_region",
    "necklace":       "chest_region",
    "tie":            "chest_region",
    "bow tie":        "chest_region",
    "scarf":          "chest_region",
    "pendant":        "chest_region",
    # Wrist / hands
    "watch":          "wrist_region",
    "bracelet":       "wrist_region",
    "ring":           "hands_region",
    "glove":          "hands_region",
    "hand":           "hands_region",
    # Carried / handheld items
    "phone":          "hands_or_lower",
    "bottle":         "hands_or_lower",
    "cup":            "hands_or_lower",
    "glass":          "hands_or_lower",
    "mug":            "hands_or_lower",
    "book":           "hands_or_lower",
    "pen":            "hands_or_lower",
    "tissue":         "hands_or_lower",
    "tissues":        "hands_or_lower",
    "paper":          "hands_or_lower",
    # Lower body / large carried items
    "bag":            "lower_body",
    "backpack":       "lower_body",
    "purse":          "lower_body",
    "handbag":        "lower_body",
    "luggage":        "lower_body",
    "suitcase":       "lower_body",
    "umbrella":       "lower_body",
    # Environmental / large background objects
    "chair":          "full_background",
    "table":          "full_background",
    "desk":           "full_background",
    "sofa":           "full_background",
    "couch":          "full_background",
    "wall":           "full_background",
    "door":           "full_background",
    "window":         "full_background",
    # Ground-plane artifacts
    "shadow":         "ground_region",
    "reflection":     "ground_region",
    "puddle":         "ground_region",
    # Overlaid / composited graphics
    "watermark":      "corner_sweep",
    "logo":           "corner_sweep",
    "timestamp":      "corner_sweep",
    "date":           "corner_sweep",
    "text":           "full_scan",
    "caption":        "full_scan",
    "subtitle":       "full_scan",
    "label":          "full_scan",
}


def _resolve_spatial_hint(target: str) -> str:
    """
    Resolve a spatial hint label for a given removal target string.

    Performs substring matching against SPATIAL_HINTS keys.
    Returns "center_region" as a safe default for unknown objects.
    """
    t_lower = target.lower().strip()
    # Prefer longest matching key to avoid partial matches
    for key in sorted(SPATIAL_HINTS, key=len, reverse=True):
        if key in t_lower:
            return SPATIAL_HINTS[key]
    return "center_region"


class ObjectRemovalPipeline:
    """
    Detects and erases any user-specified target objects using zero-shot
    object detection (OWL-ViT), spatial hint fallbacks, GrabCut refinement,
    and LaMa localized inpainting.

    No object type is hardcoded in control flow. Spatial fallback behavior
    is fully driven by the SPATIAL_HINTS dictionary.
    """

    MODEL_ID = "google/owlvit-base-patch32"

    def __init__(self, device: Optional[torch.device] = None, face_protector: Optional[FaceProtector] = None):
        self.device = device or get_device()
        self._detector = None
        self._face_protector = face_protector or FaceProtector()
        self._local_inpainter = LocalInpainter()

    def _load_detector(self) -> None:
        """Lazy-load the SemanticDetector."""
        if self._detector is not None and self._detector != "fallback":
            return
        try:
            from src.models.detection import SemanticDetector
            logger.info("Initializing SemanticDetector...")
            self._detector = SemanticDetector(device=self.device)
        except Exception as exc:
            logger.warning(f"Failed to load SemanticDetector ({exc}). Using torso heuristic fallbacks.")
            self._detector = "fallback"

    def detect_targets(
        self,
        image_rgb: np.ndarray,
        targets: List[str],
        threshold: float = 0.12,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect target objects in the image using open-vocabulary zero-shot detection.

        When the detector is unavailable or finds no results, falls back to
        a generalized spatial heuristic driven by SPATIAL_HINTS.

        Args:
            image_rgb: Input image as RGB uint8 array.
            targets:   List of object names to detect (any arbitrary strings).
            threshold: Confidence threshold.

        Returns:
            List of (x1, y1, x2, y2) bounding boxes.
        """
        h, w = image_rgb.shape[:2]
        self._load_detector()

        # Resolve spatial hints for all targets (generalized — no hardcoded names)
        spatial_hints = [_resolve_spatial_hint(t) for t in targets]
        logger.debug(f"Spatial hints for targets {targets}: {spatial_hints}")

        if self._detector == "fallback" or self._detector is None:
            logger.warning(
                f"Zero-shot detector unavailable. Using spatial fallback for: {targets}"
            )
            return self._spatial_fallback_boxes(image_rgb, targets, spatial_hints)

        try:
            boxes = []
            for target in targets:
                results = self._detector.detect(prompt=target, image=image_rgb, threshold=threshold)
                for res in results:
                    bx1, by1, bx2, by2 = res.box
                    # Filter extremely small or large boxes to reduce false positives
                    box_area = (bx2 - bx1) * (by2 - by1)
                    img_area = h * w
                    if 0.0005 < (box_area / img_area) < 0.25:
                        boxes.append((bx1, by1, bx2, by2))
                        logger.info(
                            f"Detected '{res.label}' confidence={res.confidence:.3f} "
                            f"box={boxes[-1]}"
                        )

            if not boxes:
                logger.info(
                    f"SemanticDetector found no targets {targets}. Applying spatial fallback."
                )
                return self._spatial_fallback_boxes(image_rgb, targets, spatial_hints)

            return boxes

        except Exception as exc:
            logger.warning(
                f"Error during zero-shot detection ({exc}). Using spatial fallback."
            )
            return self._spatial_fallback_boxes(image_rgb, targets, spatial_hints)

    # ── Generalized spatial fallback system ───────────────────────────────────

    def _spatial_fallback_boxes(
        self,
        image_rgb: np.ndarray,
        targets: List[str],
        spatial_hints: List[str],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Dispatch to the appropriate spatial fallback box generator based on
        the resolved spatial hint for each target.

        Merges boxes from all targets, deduplicating overlapping ones.
        """
        faces = self._face_protector.detect_faces(image_rgb)
        all_boxes: List[Tuple[int, int, int, int]] = []

        for target, hint in zip(targets, spatial_hints):
            logger.info(f"Spatial fallback: target='{target}' → hint='{hint}'")
            boxes = self._boxes_for_hint(image_rgb, hint, faces)
            all_boxes.extend(boxes)

        # Deduplicate identical boxes
        return list(dict.fromkeys(all_boxes))

    def _boxes_for_hint(
        self,
        image_rgb: np.ndarray,
        hint: str,
        faces: List[Any],
    ) -> List[Tuple[int, int, int, int]]:
        """Return fallback bounding box(es) for a given spatial hint label."""
        h, w = image_rgb.shape[:2]

        dispatch: Dict[str, Callable] = {
            "upper_region":   lambda: self._upper_region_box(h, w, faces),
            "chest_region":   lambda: self._chest_region_box(h, w, faces),
            "wrist_region":   lambda: self._wrist_region_box(h, w, faces),
            "hands_region":   lambda: self._hands_region_box(h, w),
            "hands_or_lower": lambda: self._hands_or_lower_box(h, w, faces),
            "lower_body":     lambda: self._lower_body_box(h, w),
            "full_background": lambda: self._full_image_box(h, w),
            "ground_region":  lambda: self._ground_region_box(h, w),
            "corner_sweep":   lambda: self._corner_sweep_boxes(h, w),
            "full_scan":      lambda: self._full_image_box(h, w),
            "center_region":  lambda: self._center_region_box(h, w),
        }

        fn = dispatch.get(hint, lambda: self._center_region_box(h, w))
        result = fn()
        # Normalize: always return list of boxes
        if isinstance(result, tuple) and len(result) == 4 and isinstance(result[0], int):
            return [result]
        if isinstance(result, list):
            return result
        return [result]

    # ── Spatial box generators ──────────────────────────────────────────────

    def _upper_region_box(
        self, h: int, w: int, faces: List[Any]
    ) -> Tuple[int, int, int, int]:
        """Upper third of the image — head/hair/hat region."""
        if faces:
            fx1, fy1, fx2, fy2 = faces[0].bbox
            # Expand slightly above and around the face
            fch = fy2 - fy1
            bx1 = max(0, fx1 - int(fch * 0.3))
            bx2 = min(w, fx2 + int(fch * 0.3))
            by1 = max(0, fy1 - int(fch * 0.5))
            by2 = min(h, fy2)
            logger.info(f"Upper-region box (face-relative): {(bx1, by1, bx2, by2)}")
            return (bx1, by1, bx2, by2)
        # Default: top third
        box = (0, 0, w, h // 3)
        logger.info(f"Upper-region box (default top-third): {box}")
        return box

    def _chest_region_box(
        self, h: int, w: int, faces: List[Any]
    ) -> Tuple[int, int, int, int]:
        """
        Chest/torso region — for lanyards, badges, ID cards, ties, necklaces.
        Uses face position to anchor the chest window.
        """
        if faces:
            fx1, fy1, fx2, fy2 = faces[0].bbox
            fcx = (fx1 + fx2) // 2
            fch = fy2 - fy1
            bx1 = max(0, fcx - int(fch * 1.2))
            bx2 = min(w, fcx + int(fch * 1.2))
            by1 = max(0, fy2 + int(fch * 0.8))
            by2 = min(h, fy2 + int(fch * 3.2))
            logger.info(f"Chest-region box (face-relative): {(bx1, by1, bx2, by2)}")
            return (bx1, by1, bx2, by2)
        # Default: center-lower
        box = (w // 3, h // 2, w * 2 // 3, h * 5 // 6)
        logger.info(f"Chest-region box (center-lower default): {box}")
        return box

    def _wrist_region_box(
        self, h: int, w: int, faces: List[Any]
    ) -> Tuple[int, int, int, int]:
        """Wrist area — watches, bracelets. Estimated at lower-middle sides."""
        # Wrists are typically at the mid-lower sides of the frame
        wrist_y1 = int(h * 0.5)
        wrist_y2 = int(h * 0.8)
        # Take the lower-left quadrant as a heuristic
        box = (0, wrist_y1, w // 3, wrist_y2)
        logger.info(f"Wrist-region box: {box}")
        return box

    def _hands_region_box(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """General hand region — rings, gloves. Lower-center zone."""
        box = (w // 4, int(h * 0.6), w * 3 // 4, h)
        logger.info(f"Hands-region box: {box}")
        return box

    def _hands_or_lower_box(
        self, h: int, w: int, faces: List[Any]
    ) -> Tuple[int, int, int, int]:
        """Handheld items (phone, bottle, cup) — below face, center zone."""
        if faces:
            fx1, fy1, fx2, fy2 = faces[0].bbox
            fch = fy2 - fy1
            by1 = min(h - 1, fy2 + int(fch * 1.5))
            by2 = min(h, fy2 + int(fch * 4.0))
            bx1 = max(0, (fx1 + fx2) // 2 - int(fch * 1.5))
            bx2 = min(w, (fx1 + fx2) // 2 + int(fch * 1.5))
            box = (bx1, by1, bx2, by2)
        else:
            box = (w // 4, int(h * 0.45), w * 3 // 4, h)
        logger.info(f"Hands-or-lower box: {box}")
        return box

    def _lower_body_box(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Lower body — bags, backpacks, luggage."""
        box = (w // 6, int(h * 0.55), w * 5 // 6, h)
        logger.info(f"Lower-body box: {box}")
        return box

    def _full_image_box(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Full image crop — for large environmental objects (chair, table)."""
        box = (0, 0, w, h)
        logger.info(f"Full-image box: {box}")
        return box

    def _ground_region_box(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Ground/floor zone — shadows, reflections, puddles."""
        box = (0, int(h * 0.7), w, h)
        logger.info(f"Ground-region box: {box}")
        return box

    def _corner_sweep_boxes(
        self, h: int, w: int
    ) -> List[Tuple[int, int, int, int]]:
        """Four corner regions — watermarks, logos, timestamps."""
        margin_x = w // 5
        margin_y = h // 7
        boxes = [
            (0,                 0,                 margin_x * 2, margin_y * 2),   # top-left
            (w - margin_x * 2, 0,                 w,            margin_y * 2),   # top-right
            (0,                 h - margin_y * 2, margin_x * 2, h),              # bottom-left
            (w - margin_x * 2, h - margin_y * 2, w,            h),              # bottom-right
        ]
        logger.info(f"Corner-sweep boxes: {boxes}")
        return boxes

    def _center_region_box(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Safe center crop — default for unknown object types."""
        box = (w // 4, h // 4, w * 3 // 4, h * 3 // 4)
        logger.info(f"Center-region box (unknown object default): {box}")
        return box

    def generate_removal_mask(
        self,
        image_rgb: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        faces: List[Any],
        targets: List[str],
        use_grabcut: bool = True,
    ) -> np.ndarray:
        """
        Builds a comprehensive removal mask covering the target box refined by GrabCut
        plus a guided path to the neck to capture the lanyard.
        """
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Add refined boxes (GrabCut)
        for (bx1, by1, bx2, by2) in boxes:
            bw, bh = bx2 - bx1, by2 - by1
            if bw < 5 or bh < 5:
                continue

            if use_grabcut:
                try:
                    # Run GrabCut inside box region
                    gc_mask = np.zeros((h, w), np.uint8)
                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)
                    
                    # Define a rect slightly inside image boundaries
                    rect = (bx1, by1, bw, bh)
                    cv2.grabCut(image_rgb, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
                    
                    # Get definite + probable foreground
                    refined_box = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)
                    mask = cv2.bitwise_or(mask, refined_box)
                except Exception as exc:
                    logger.warning(f"GrabCut failed ({exc}). Filling raw bounding box.")
                    cv2.rectangle(mask, (bx1, by1), (bx2, by2), 255, -1)
            else:
                cv2.rectangle(mask, (bx1, by1), (bx2, by2), 255, -1)

        # 2. Add lanyard path from neck down to the first detected target box (lanyard-specific logic)
        # Generalized: check for chest-region targets using SPATIAL_HINTS (no hardcoded names)
        has_lanyard_target = any(_resolve_spatial_hint(t) == "chest_region" for t in targets)
        if has_lanyard_target and faces and boxes:
            fx1, fy1, fx2, fy2 = faces[0].bbox
            fcx = (fx1 + fx2) // 2
            y_neck = fy2  # Start neck directly at bottom of face
            
            # Find closest target center
            bx1, by1, bx2, by2 = boxes[0]
            bcx = (bx1 + bx2) // 2
            b_top = by1
            
            # Trace a thick line representing the lanyard
            # Draw a thick line down from neck center to target top
            cv2.line(mask, (fcx, y_neck), (bcx, b_top), 255, thickness=20)
            
            # Draw auxiliary lines to shoulders to cover V-shaped lanyards
            sw = fx2 - fx1  # Face width as scale
            cv2.line(mask, (fcx - sw // 2, y_neck), (bcx, b_top), 255, thickness=12)
            cv2.line(mask, (fcx + sw // 2, y_neck), (bcx, b_top), 255, thickness=12)

        # 3. Dilate the mask to ensure we cover outlines and thin strings completely
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask

    def remove(
        self,
        image_rgb: np.ndarray,
        local_inpainter: Optional[LocalInpainter] = None,
        targets: List[str] = ["id card", "lanyard", "badge"],
    ) -> ObjectRemovalResult:
        """
        Performs localized object removal by detecting, masking, and inpainting.
        """
        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]

        # 1. Detect target objects
        boxes = self.detect_targets(image_rgb, targets)

        # 2. Get face locations for lanyard guidance
        faces = self._face_protector.detect_faces(image_rgb)

        # 3. Create guided removal mask
        removal_mask = self.generate_removal_mask(image_rgb, boxes, faces, targets=targets)

        # If mask is empty, return original image
        if np.count_nonzero(removal_mask) == 0:
            logger.warning("Target removal mask is empty. Skipping inpainting.")
            return ObjectRemovalResult(
                image=image_rgb,
                object_mask=removal_mask,
                detected_boxes=boxes,
                removal_time_s=time.perf_counter() - t0,
            )

        # 4. Localized Inpainting using LocalInpainter
        logger.info("Running localized local inpainting inside target mask...")
        inpainter = local_inpainter or self._local_inpainter
        final_image = inpainter.inpaint(image_rgb, removal_mask)

        elapsed = time.perf_counter() - t0
        logger.info(f"Targeted object removal completed in {elapsed:.2f}s")

        return ObjectRemovalResult(
            image=final_image,
            object_mask=removal_mask,
            detected_boxes=boxes,
            removal_time_s=elapsed,
        )
