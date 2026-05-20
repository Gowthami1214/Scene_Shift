"""
Input validation utilities for SceneShift API.
Provides secure file validation, prompt sanitization, and parameter bounds checking.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from loguru import logger


# ── Constants ──────────────────────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_IMAGE_DIM = 4096
MIN_IMAGE_DIM = 64
MAX_PROMPT_LENGTH = 500

STYLE_PRESETS = [
    "Realistic",
    "Cinematic",
    "Cyberpunk",
    "Cartoon",
    "Pencil Sketch",
    "Fantasy",
    "Vintage",
    "Minimalist",
]

BLEND_MODES = ["alpha", "poisson", "laplacian"]


# ── File validation ────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_image_bytes(
    data: bytes,
    filename: str,
    max_mb: float = MAX_FILE_SIZE_MB,
) -> Tuple[int, int]:
    """
    Validate raw image bytes for security and correctness.

    Checks:
        - File size within limit
        - Valid image extension
        - Decodable by Pillow (true file-type check)
        - Dimension bounds

    Args:
        data: Raw file bytes.
        filename: Original filename (for extension check).
        max_mb: Maximum allowed file size in MB.

    Returns:
        (width, height) of the image.

    Raises:
        ValidationError: On any validation failure.
    """
    # 1. Size check
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(
            f"File too large: {size_mb:.1f} MB (max {max_mb} MB)"
        )

    # 2. Extension check
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension: '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 3. True file-type check via Pillow (prevents polyglot attacks)
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()          # raises if corrupt
    except Exception as exc:
        raise ValidationError(f"Image decoding failed: {exc}") from exc

    # 4. Re-open for dimension check (verify() closes the stream)
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
    except Exception as exc:
        raise ValidationError(f"Cannot read image dimensions: {exc}") from exc

    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        raise ValidationError(
            f"Image too small: {w}x{h}. Minimum: {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}"
        )
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        raise ValidationError(
            f"Image too large: {w}x{h}. Maximum: {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}"
        )

    logger.debug(f"Validated image: {filename} ({w}x{h}, {size_mb:.1f} MB)")
    return w, h


def validate_image_path(path: Path) -> Tuple[int, int]:
    """Validate an image file from disk path."""
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    data = path.read_bytes()
    return validate_image_bytes(data, path.name)


# ── Parameter validation ───────────────────────────────────────────────────────

def validate_prompt(prompt: str, field_name: str = "prompt") -> str:
    """
    Sanitize and validate a text prompt.

    Args:
        prompt: Raw user-supplied prompt string.
        field_name: Field identifier for error messages.

    Returns:
        Stripped, validated prompt string.

    Raises:
        ValidationError: If prompt is too long or contains forbidden patterns.
    """
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"'{field_name}' too long: {len(prompt)} chars (max {MAX_PROMPT_LENGTH})"
        )
    # Basic XSS / injection guard (prompts go to SD, not HTML — minimal risk)
    forbidden = ["<script", "javascript:", "data:text"]
    for token in forbidden:
        if token.lower() in prompt.lower():
            raise ValidationError(f"Invalid content in '{field_name}'")
    return prompt


def validate_style_preset(preset: str) -> str:
    """Validate style preset against allowed values."""
    if preset not in STYLE_PRESETS:
        raise ValidationError(
            f"Invalid style preset '{preset}'. "
            f"Allowed: {', '.join(STYLE_PRESETS)}"
        )
    return preset


def validate_blend_mode(mode: str) -> str:
    """Validate compositing blend mode."""
    mode = mode.lower()
    if mode not in BLEND_MODES:
        raise ValidationError(
            f"Invalid blend mode '{mode}'. "
            f"Allowed: {', '.join(BLEND_MODES)}"
        )
    return mode


def validate_strength(value: float, name: str = "strength") -> float:
    """Validate a [0.0, 1.0] float parameter."""
    if not (0.0 <= value <= 1.0):
        raise ValidationError(
            f"'{name}' must be between 0.0 and 1.0, got {value}"
        )
    return value


def validate_click_point(
    x: float,
    y: float,
    img_width: int,
    img_height: int,
) -> Tuple[int, int]:
    """
    Validate a user click point is within image bounds.

    Args:
        x, y: Click coordinates (may be fractional 0-1 if normalized).
        img_width, img_height: Image dimensions.

    Returns:
        Integer (px, py) pixel coordinates.
    """
    # Accept both normalized [0,1] and absolute pixel coords
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        px, py = int(x * img_width), int(y * img_height)
    else:
        px, py = int(x), int(y)

    if not (0 <= px < img_width and 0 <= py < img_height):
        raise ValidationError(
            f"Click point ({px}, {py}) out of bounds for "
            f"image size ({img_width}x{img_height})"
        )
    return px, py
