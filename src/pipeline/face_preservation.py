"""
Face Preservation Pipeline for SceneShift.

Detects, protects, and restores human faces to maintain identity
through the AI scene transformation pipeline.

Components:
  1. FaceProtector  — MediaPipe-based face detection + protection mask generation
  2. FaceRestorer   — CodeFormer/GFPGAN face quality restoration
  3. SkinTonePreserver — Histogram-based skin tone matching
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class FaceRegion:
    """Detected face bounding box and landmarks."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[int, int]          # (cx, cy)
    area: int                        # bbox area in pixels


@dataclass
class FacePreservationResult:
    """Output of the face preservation pipeline."""
    protection_mask: np.ndarray       # (H x W, uint8, 0/255) — 255 = protect
    faces: List[FaceRegion]           # Detected face regions
    num_faces: int
    detection_time_s: float


@dataclass
class FaceRestorationResult:
    """Output of face restoration post-processing."""
    restored_image: np.ndarray        # Full image with restored faces (H x W x 3, uint8)
    num_faces_restored: int
    restoration_time_s: float


# ── Face Protector ────────────────────────────────────────────────────────────

class FaceProtector:
    """
    Detects faces in images and generates protection masks
    that prevent Stable Diffusion from modifying face regions.

    Uses OpenCV's DNN face detector (Caffe model) as primary detector,
    with Haar cascade as fallback. Both are CPU-optimized and fast.
    """

    def __init__(self):
        self._dnn_net = None
        self._haar_cascade = None

    def _load_detectors(self) -> None:
        """Load face detection models."""
        # Try OpenCV DNN face detector first (more accurate)
        try:
            self._dnn_net = cv2.dnn.readNetFromCaffe(
                cv2.data.haarcascades + "../deploy.prototxt",
                cv2.data.haarcascades + "../res10_300x300_ssd_iter_140000.caffemodel",
            )
        except Exception:
            self._dnn_net = None

        # Always load Haar cascade as fallback
        try:
            self._haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            self._haar_cascade = None

    def detect_faces(
        self,
        image_rgb: np.ndarray,
        min_confidence: float = 0.5,
        min_face_fraction: float = 0.01,
    ) -> List[FaceRegion]:
        """
        Detect faces in an RGB image.

        Args:
            image_rgb: Input image (H x W x 3, uint8, RGB).
            min_confidence: Minimum detection confidence.
            min_face_fraction: Minimum face area as fraction of image area.

        Returns:
            List of FaceRegion objects.
        """
        if self._haar_cascade is None:
            self._load_detectors()

        h, w = image_rgb.shape[:2]
        img_area = h * w
        min_face_area = int(img_area * min_face_fraction)
        faces = []

        # Method 1: OpenCV DNN SSD face detector
        if self._dnn_net is not None:
            faces = self._detect_dnn(image_rgb, min_confidence, min_face_area)

        # Method 2: Haar cascade fallback
        if not faces and self._haar_cascade is not None:
            faces = self._detect_haar(image_rgb, min_face_area)

        logger.info(f"Face detection: found {len(faces)} face(s)")
        return faces

    def _detect_dnn(
        self, image_rgb: np.ndarray, min_conf: float, min_area: int
    ) -> List[FaceRegion]:
        """DNN-based face detection (SSD ResNet)."""
        h, w = image_rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < min_conf:
                continue

            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            faces.append(FaceRegion(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                area=area,
            ))

        return faces

    def _detect_haar(
        self, image_rgb: np.ndarray, min_area: int
    ) -> List[FaceRegion]:
        """Haar cascade face detection (fallback)."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        faces = []
        for (x, y, fw, fh) in rects:
            area = fw * fh
            if area < min_area:
                continue
            faces.append(FaceRegion(
                bbox=(x, y, x + fw, y + fh),
                confidence=0.7,  # Haar doesn't provide confidence
                center=(x + fw // 2, y + fh // 2),
                area=area,
            ))

        return faces

    def create_protection_mask(
        self,
        image_shape: Tuple[int, int],
        faces: List[FaceRegion],
        padding_fraction: float = 0.25,
        feather_radius: int = 15,
    ) -> np.ndarray:
        """
        Generate a mask that marks face regions for protection.
        White (255) = protect from SD modification.

        Args:
            image_shape: (H, W) of the image.
            faces: Detected face regions.
            padding_fraction: Extra padding around face bbox as fraction of face size.
            feather_radius: Gaussian blur radius for soft mask edges.

        Returns:
            Protection mask (H x W, uint8, 0/255).
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            fw, fh = x2 - x1, y2 - y1

            # Add padding around face
            pad_x = int(fw * padding_fraction)
            pad_y = int(fh * padding_fraction)

            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            # Draw filled ellipse for more natural face shape
            cx = (px1 + px2) // 2
            cy = (py1 + py2) // 2
            rx = (px2 - px1) // 2
            ry = (py2 - py1) // 2
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # Feather edges for smooth blending
        if feather_radius > 0 and np.any(mask > 0):
            ksize = feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_radius / 2)
            mask = (mask > 30).astype(np.uint8) * 255

        return mask

    def detect_and_protect(
        self,
        image_rgb: np.ndarray,
        min_confidence: float = 0.5,
    ) -> FacePreservationResult:
        """
        Full pipeline: detect faces and generate protection mask.

        Args:
            image_rgb: Input RGB image.
            min_confidence: Minimum face detection confidence.

        Returns:
            FacePreservationResult with protection mask and face regions.
        """
        t0 = time.perf_counter()

        faces = self.detect_faces(image_rgb, min_confidence)
        h, w = image_rgb.shape[:2]

        if faces:
            protection_mask = self.create_protection_mask((h, w), faces)
        else:
            protection_mask = np.zeros((h, w), dtype=np.uint8)

        elapsed = time.perf_counter() - t0
        logger.info(f"Face preservation: {len(faces)} faces detected in {elapsed:.3f}s")

        return FacePreservationResult(
            protection_mask=protection_mask,
            faces=faces,
            num_faces=len(faces),
            detection_time_s=elapsed,
        )


# ── Face Restorer ─────────────────────────────────────────────────────────────

class FaceRestorer:
    """
    Restores face quality after AI compositing using targeted upscaling
    and detail enhancement on detected face regions only.

    Uses OpenCV-based restoration (no external model dependencies) with:
    - CLAHE local contrast enhancement
    - Bilateral filtering for skin smoothing
    - Unsharp masking for detail recovery
    - Color consistency matching with original face
    """

    def restore(
        self,
        processed_image: np.ndarray,
        original_image: np.ndarray,
        faces: List[FaceRegion],
        fidelity: float = 0.7,
    ) -> FaceRestorationResult:
        """
        Restore face quality in the processed image.

        Args:
            processed_image: The composited/harmonized image (H x W x 3, uint8, RGB).
            original_image: The original input image for reference.
            faces: Detected face regions from FaceProtector.
            fidelity: Blend weight [0, 1] — 1.0 = keep more original face detail.

        Returns:
            FaceRestorationResult with the restored image.
        """
        t0 = time.perf_counter()

        if not faces:
            return FaceRestorationResult(
                restored_image=processed_image,
                num_faces_restored=0,
                restoration_time_s=0.0,
            )

        result = processed_image.copy()
        h, w = result.shape[:2]
        restored_count = 0

        for face in faces:
            x1, y1, x2, y2 = face.bbox

            # Add padding for context
            pad = int(max(x2 - x1, y2 - y1) * 0.15)
            fx1 = max(0, x1 - pad)
            fy1 = max(0, y1 - pad)
            fx2 = min(w, x2 + pad)
            fy2 = min(h, y2 + pad)

            if fx2 - fx1 < 10 or fy2 - fy1 < 10:
                continue

            # Extract face crops
            proc_face = result[fy1:fy2, fx1:fx2].copy()
            orig_face = original_image[fy1:fy2, fx1:fx2].copy()

            if proc_face.shape != orig_face.shape:
                orig_face = cv2.resize(orig_face, (proc_face.shape[1], proc_face.shape[0]))

            # Step 1: Enhance processed face with CLAHE
            enhanced = self._enhance_face(proc_face)

            # Step 2: Match color distribution to original face
            color_matched = self._match_face_color(enhanced, orig_face)

            # Step 3: Blend enhanced face with original at fidelity weight
            blended = cv2.addWeighted(
                color_matched.astype(np.float32), 1.0 - fidelity * 0.3,
                orig_face.astype(np.float32), fidelity * 0.3,
                0.0,
            )
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            # Step 4: Create soft elliptical mask for seamless paste-back
            face_mask = np.zeros((fy2 - fy1, fx2 - fx1), dtype=np.float32)
            cy = (fy2 - fy1) // 2
            cx = (fx2 - fx1) // 2
            ry = int(cy * 0.85)
            rx = int(cx * 0.85)
            cv2.ellipse(face_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 5)
            face_mask_3ch = face_mask[:, :, np.newaxis]

            # Step 5: Composite restored face back
            result_region = result[fy1:fy2, fx1:fx2].astype(np.float32)
            result[fy1:fy2, fx1:fx2] = np.clip(
                blended.astype(np.float32) * face_mask_3ch +
                result_region * (1.0 - face_mask_3ch),
                0, 255
            ).astype(np.uint8)

            restored_count += 1

        elapsed = time.perf_counter() - t0
        logger.info(f"Face restoration: {restored_count} face(s) restored in {elapsed:.3f}s")

        return FaceRestorationResult(
            restored_image=result,
            num_faces_restored=restored_count,
            restoration_time_s=elapsed,
        )

    def _enhance_face(self, face_crop: np.ndarray) -> np.ndarray:
        """Apply CLAHE + unsharp masking to enhance face details."""
        # Convert to LAB for luminance-only enhancement
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB).astype(np.float32)

        # CLAHE on L channel
        l_channel = np.clip(lab[:, :, 0], 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l_channel)
        lab[:, :, 0] = l_enhanced.astype(np.float32)

        enhanced = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

        # Gentle unsharp masking for detail
        blurred = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), 1.5)
        sharpened = enhanced.astype(np.float32) + 0.3 * (enhanced.astype(np.float32) - blurred)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _match_face_color(
        self, processed_face: np.ndarray, original_face: np.ndarray
    ) -> np.ndarray:
        """Match the color distribution of processed face to original face in LAB space."""
        proc_lab = cv2.cvtColor(processed_face, cv2.COLOR_RGB2LAB).astype(np.float32)
        orig_lab = cv2.cvtColor(original_face, cv2.COLOR_RGB2LAB).astype(np.float32)

        result = proc_lab.copy()
        for c in range(3):
            p_mean, p_std = proc_lab[:, :, c].mean(), max(proc_lab[:, :, c].std(), 1e-6)
            o_mean, o_std = orig_lab[:, :, c].mean(), max(orig_lab[:, :, c].std(), 1e-6)

            # Partial transfer (50% weight) to avoid over-correction
            transferred = (proc_lab[:, :, c] - p_mean) / p_std * o_std + o_mean
            result[:, :, c] = proc_lab[:, :, c] * 0.5 + transferred * 0.5

        return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


# ── Skin Tone Preserver ───────────────────────────────────────────────────────

class SkinTonePreserver:
    """
    Preserves original skin tone through the pipeline by extracting
    skin color histograms from the original image and matching them
    in the final output.
    """

    def extract_skin_reference(
        self, image_rgb: np.ndarray, face_regions: List[FaceRegion]
    ) -> Optional[np.ndarray]:
        """
        Extract reference skin tone histogram from original face regions.

        Returns:
            Mean skin color in LAB space (3,) or None if no skin detected.
        """
        if not face_regions:
            return None

        skin_pixels = []
        for face in face_regions:
            x1, y1, x2, y2 = face.bbox
            face_crop = image_rgb[y1:y2, x1:x2]

            # Detect skin pixels using YCrCb thresholds
            ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_RGB2YCrCb)
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            skin_mask = cv2.inRange(ycrcb, lower, upper)

            # Extract skin-colored pixels
            lab_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
            skin_px = lab_crop[skin_mask > 0]

            if len(skin_px) > 0:
                skin_pixels.append(skin_px)

        if not skin_pixels:
            return None

        all_skin = np.vstack(skin_pixels)
        return all_skin.mean(axis=0)  # Mean LAB color

    def apply_skin_preservation(
        self,
        image_rgb: np.ndarray,
        reference_skin_lab: np.ndarray,
        face_regions: List[FaceRegion],
        strength: float = 0.4,
    ) -> np.ndarray:
        """
        Match skin tones in the processed image to the reference.

        Args:
            image_rgb: Processed image to correct.
            reference_skin_lab: Mean LAB skin color from original.
            face_regions: Face regions to apply correction.
            strength: Correction strength [0, 1].

        Returns:
            Skin-tone corrected image.
        """
        result = image_rgb.copy()

        for face in face_regions:
            x1, y1, x2, y2 = face.bbox
            pad = int(max(x2 - x1, y2 - y1) * 0.1)
            fx1, fy1 = max(0, x1 - pad), max(0, y1 - pad)
            fx2 = min(result.shape[1], x2 + pad)
            fy2 = min(result.shape[0], y2 + pad)

            face_crop = result[fy1:fy2, fx1:fx2]

            # Detect skin in processed image
            ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_RGB2YCrCb)
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            skin_mask = cv2.inRange(ycrcb, lower, upper)
            skin_mask_f = cv2.GaussianBlur(skin_mask, (7, 7), 0).astype(np.float32) / 255.0

            if skin_mask_f.max() < 0.1:
                continue

            # Shift skin tone toward reference
            lab_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
            current_skin = lab_crop[skin_mask > 0]
            if len(current_skin) == 0:
                continue

            current_mean = current_skin.mean(axis=0)
            shift = (reference_skin_lab - current_mean) * strength

            # Apply shift only to skin pixels (weighted by skin mask)
            for c in range(3):
                lab_crop[:, :, c] += shift[c] * skin_mask_f

            corrected = cv2.cvtColor(
                np.clip(lab_crop, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
            )
            result[fy1:fy2, fx1:fx2] = corrected

        return result
