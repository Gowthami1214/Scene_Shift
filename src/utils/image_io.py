"""
Image I/O utilities for SceneShift.
Provides unified image loading, saving, format conversion, and resizing.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger


# ── Type aliases ──────────────────────────────────────────────────────────────
ImageArray = np.ndarray          # HxWxC BGR or HxW grayscale
PILImage = Image.Image


# ── Loading ───────────────────────────────────────────────────────────────────

def load_image_rgb(path: Union[str, Path]) -> ImageArray:
    """
    Load an image file as an RGB NumPy array (HxWx3, uint8).

    Args:
        path: Path to image file (supports PNG, JPEG, WEBP, BMP, TIFF).

    Returns:
        RGB image array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot decode image: {path}")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def pil_to_numpy(img: PILImage, mode: str = "RGB") -> ImageArray:
    """Convert PIL Image to NumPy array (uint8)."""
    if img.mode != mode:
        img = img.convert(mode)
    return np.array(img, dtype=np.uint8)


def numpy_to_pil(arr: ImageArray, mode: str = "RGB") -> PILImage:
    """Convert NumPy array to PIL Image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_pil(path: Union[str, Path]) -> PILImage:
    """Load image as PIL Image (RGBA-safe)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(str(path)).convert("RGB")


# ── Saving ────────────────────────────────────────────────────────────────────

def save_image(
    image: Union[ImageArray, PILImage],
    path: Union[str, Path],
    quality: int = 95,
) -> Path:
    """
    Save an image to disk.

    Args:
        image: RGB NumPy array or PIL Image.
        path: Output file path (.png or .jpg).
        quality: JPEG quality (1-100); ignored for PNG.

    Returns:
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        pil_img = numpy_to_pil(image)
    else:
        pil_img = image

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        pil_img.save(str(path), format="JPEG", quality=quality, optimize=True)
    elif ext == ".png":
        pil_img.save(str(path), format="PNG", optimize=True)
    elif ext == ".webp":
        pil_img.save(str(path), format="WEBP", quality=quality)
    else:
        pil_img.save(str(path))

    logger.debug(f"Saved image: {path} ({pil_img.size[0]}x{pil_img.size[1]})")
    return path


# ── Resizing ──────────────────────────────────────────────────────────────────

def resize_to_target(
    image: ImageArray,
    target_size: Tuple[int, int],
    keep_aspect: bool = True,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> ImageArray:
    """
    Resize an image to the target (width, height).

    Args:
        image: Input RGB NumPy array.
        target_size: (width, height) tuple.
        keep_aspect: Pad/letterbox to preserve aspect ratio.
        interpolation: OpenCV interpolation flag.

    Returns:
        Resized image array.
    """
    tw, th = target_size
    h, w = image.shape[:2]

    if not keep_aspect:
        return cv2.resize(image, (tw, th), interpolation=interpolation)

    # Compute letterbox scale
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)

    # Pad to target with neutral gray
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    x_off = (tw - nw) // 2
    y_off = (th - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def resize_for_pipeline(
    image: ImageArray,
    max_dim: int = 768,
) -> Tuple[ImageArray, Tuple[int, int]]:
    """
    Resize image so the longest dimension is max_dim (keeps aspect ratio).

    Returns:
        (resized_image, (original_width, original_height))
    """
    h, w = image.shape[:2]
    orig_size = (w, h)
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return image, orig_size
    nw, nh = int(w * scale), int(h * scale)
    # Round to multiples of 8 (required by Stable Diffusion VAE)
    nw = (nw // 8) * 8
    nh = (nh // 8) * 8
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    return resized, orig_size


# ── Base64 helpers ────────────────────────────────────────────────────────────

def image_to_base64(image: Union[ImageArray, PILImage], fmt: str = "PNG") -> str:
    """Encode image as base64 string (for JSON API responses)."""
    if isinstance(image, np.ndarray):
        pil_img = numpy_to_pil(image)
    else:
        pil_img = image

    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def base64_to_image(b64_str: str) -> PILImage:
    """Decode a base64-encoded image string to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ── Mask utilities ────────────────────────────────────────────────────────────

def mask_to_pil(mask: np.ndarray) -> PILImage:
    """Convert binary/grayscale mask array to PIL grayscale image."""
    if mask.dtype != np.uint8:
        mask = (mask * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def crop_with_padding(
    image: ImageArray,
    bbox: Tuple[int, int, int, int],
    padding: int = 20,
) -> Tuple[ImageArray, Tuple[int, int, int, int]]:
    """
    Crop image to bounding box with optional padding.

    Args:
        image: Input RGB image.
        bbox: (x1, y1, x2, y2) bounding box.
        padding: Extra pixels to include around bbox.

    Returns:
        (cropped_image, padded_bbox)
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def temp_output_path(job_id: str, suffix: str = ".png", stage: str = "result") -> Path:
    """Generate a deterministic temporary output path for a pipeline job."""
    out_dir = Path("outputs") / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stage}{suffix}"
