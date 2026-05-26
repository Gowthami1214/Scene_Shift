"""
LaMa ONNX Inpainting and Fast Texture-Preserving Fallback for SceneShift.
Designed for pixel-perfect, local object removal without Stable Diffusion.
"""

from __future__ import annotations

import os
import urllib.request
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


class LocalInpainter:
    """
    Non-generative local inpainting engine.
    Attempts to download and run the LaMa ONNX model.
    Falls back to a fast, texture-preserving OpenCV patch inpainter to maintain local photo grain.
    """

    MODEL_URL = "https://huggingface.co/anyisalin/simple-lama-inpainting/resolve/main/lama_fp32.onnx"
    MODEL_DIR = Path("models")
    MODEL_PATH = MODEL_DIR / "lama_fp32.onnx"

    def __init__(self, use_lama: bool = True):
        self.use_lama = use_lama
        self.session = None
        self._initialized = False

    def _download_model(self) -> bool:
        """Download LaMa ONNX model if not cached."""
        if self.MODEL_PATH.is_file():
            return True
        
        try:
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading LaMa ONNX model from {self.MODEL_URL}...")
            t0 = time.perf_counter()
            # Set a timeout for download to avoid blocking the pipeline indefinitely
            req = urllib.request.Request(self.MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8.0) as response, open(self.MODEL_PATH, "wb") as out_file:
                out_file.write(response.read())
            elapsed = time.perf_counter() - t0
            logger.info(f"LaMa ONNX downloaded successfully in {elapsed:.2f}s")
            return True
        except Exception as exc:
            logger.warning(f"LaMa ONNX model download failed/timed out ({exc}). Using OpenCV texture fallback.")
            if self.MODEL_PATH.is_file():
                try:
                    self.MODEL_PATH.unlink()
                except Exception:
                    pass
            return False

    def _init_session(self) -> None:
        """Lazy-initialize ONNX Runtime session."""
        if self._initialized:
            return
        self._initialized = True

        if not self.use_lama:
            return

        # Attempt download if missing
        if not self._download_model():
            return

        try:
            import onnxruntime as ort
            logger.info(f"Loading LaMa ONNX session from {self.MODEL_PATH}...")
            t0 = time.perf_counter()
            # Prefer CUDA provider if GPU is available
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.MODEL_PATH), providers=providers)
            elapsed = time.perf_counter() - t0
            logger.info(f"LaMa ONNX session loaded in {elapsed:.2f}s")
        except Exception as exc:
            logger.warning(f"Failed to load ONNX session for LaMa ({exc}). Using OpenCV texture fallback.")
            self.session = None

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the image within the target mask.

        Args:
            image_rgb: RGB image (H x W x 3, uint8).
            mask: Binary mask, white=inpaint region (H x W, uint8, 0/255).

        Returns:
            Inpainted RGB image (H x W x 3, uint8).
        """
        h, w = image_rgb.shape[:2]
        if np.count_nonzero(mask) == 0:
            return image_rgb.copy()

        self._init_session()

        if self.session is not None:
            try:
                # 1. Preprocess: Pad image/mask to multiple of 8 (LaMa VAE requirements)
                ph = ((h + 7) // 8) * 8
                pw = ((w + 7) // 8) * 8

                pad_h = ph - h
                pad_w = pw - w

                img_pad = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                mask_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

                # 2. Reshape & normalize inputs to (1, 3, H, W) and (1, 1, H, W) float32
                img_input = img_pad.transpose(2, 0, 1).astype(np.float32) / 255.0
                img_input = np.expand_dims(img_input, axis=0)

                mask_input = (mask_pad > 0).astype(np.float32)
                mask_input = np.expand_dims(np.expand_dims(mask_input, axis=0), axis=0)

                # Run inference
                inputs = {
                    "image": img_input,
                    "mask": mask_input
                }
                outputs = self.session.run(None, inputs)
                out_pad = outputs[0][0]  # (3, H, W)

                # 3. Postprocess: Convert back to (H, W, 3) and crop padding
                out_pad = np.clip(out_pad * 255.0, 0, 255).astype(np.uint8)
                out_pad = out_pad.transpose(1, 2, 0)
                out_rgb = out_pad[:h, :w]

                # Paste inpainted pixels strictly inside mask for pixel-perfect preservation
                mask_normalized = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
                feathered = cv2.GaussianBlur(mask_normalized, (5, 5), 0)
                if len(feathered.shape) == 2:
                    feathered = feathered[:, :, np.newaxis]

                final = out_rgb.astype(np.float32) * feathered + image_rgb.astype(np.float32) * (1.0 - feathered)
                return np.clip(final, 0, 255).astype(np.uint8)

            except Exception as exc:
                logger.warning(f"LaMa ONNX inference failed ({exc}). Falling back to OpenCV texture inpainter.")
                return self._inpaint_opencv_fallback(image_rgb, mask)
        else:
            return self._inpaint_opencv_fallback(image_rgb, mask)

    def _inpaint_opencv_fallback(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Fast texture-preserving OpenCV patch fallback.
        Inpaints structure, measures surrounding reference noise, and adds grain back.
        """
        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return image_rgb.copy()

        # 1. Structure inpainting
        structure = cv2.inpaint(image_rgb, mask, 5, cv2.INPAINT_TELEA)

        # 2. Estimate and add local texture grain/noise
        y1, y2 = max(0, ys.min() - 20), min(h, ys.max() + 20)
        x1, x2 = max(0, xs.min() - 20), min(w, xs.max() + 20)

        roi_img = image_rgb[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]
        ref_pixels = roi_img[roi_mask == 0]

        if len(ref_pixels) > 10:
            # Get color standard deviation as noise scale
            ref_std = ref_pixels.std(axis=0)
            # Inject matching Gaussian noise
            noise = np.random.normal(0, ref_std * 0.45, structure.shape).astype(np.float32)
            
            mask_f = (mask > 0)[:, :, np.newaxis].astype(np.float32)
            noisy_structure = structure.astype(np.float32) + noise * mask_f
            structure = np.clip(noisy_structure, 0, 255).astype(np.uint8)

        # 3. Blend cleanly using feathered edge
        mask_normalized = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        feathered = cv2.GaussianBlur(mask_normalized, (11, 11), 0)
        if len(feathered.shape) == 2:
            feathered = feathered[:, :, np.newaxis]

        final = structure.astype(np.float32) * feathered + image_rgb.astype(np.float32) * (1.0 - feathered)
        return np.clip(final, 0, 255).astype(np.uint8)
