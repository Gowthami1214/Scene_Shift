"""
Device management utilities for SceneShift.
Provides GPU/CPU device resolution with CUDA 12.1 support and graceful fallback.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from loguru import logger


@lru_cache(maxsize=1)
def get_device(force_cpu: bool = False) -> torch.device:
    """
    Resolve the best available compute device.

    Priority:
        1. CUDA (GPU) — if available and not forced CPU
        2. MPS  (Apple Silicon)
        3. CPU  (universal fallback)

    Args:
        force_cpu: Override and always return CPU device.

    Returns:
        torch.device instance ready for model/tensor allocation.
    """
    if force_cpu or os.getenv("SCENESHIFT_FORCE_CPU", "0") == "1":
        logger.info("Device override: CPU (forced)")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        logger.info(
            f"Device selected: CUDA — {props.name} "
            f"({vram_gb:.1f} GB VRAM, "
            f"Compute {props.major}.{props.minor})"
        )
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device selected: MPS (Apple Silicon)")
        return torch.device("mps")

    logger.warning(
        "No GPU detected — running on CPU. "
        "Performance will be significantly reduced. "
        "For optimal results, use an NVIDIA RTX GPU with CUDA 12.1+."
    )
    return torch.device("cpu")


def get_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    """
    Return the optimal floating-point dtype for the given device.

    Uses float16 on CUDA (fast, lower VRAM) and float32 on CPU/MPS.

    Args:
        device: Target device; resolves automatically if not provided.

    Returns:
        torch.dtype (float16 or float32).
    """
    dev = device or get_device()
    if dev.type == "cuda":
        return torch.float16
    return torch.float32


def clear_gpu_cache() -> None:
    """Free unused GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU cache cleared.")


def get_device_info() -> dict:
    """
    Return a structured dictionary of device information.

    Returns:
        dict with keys: device_type, device_name, vram_gb, cuda_version.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "device_type": "cuda",
            "device_name": props.name,
            "vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "device_type": "mps",
            "device_name": "Apple Silicon GPU",
            "vram_gb": None,
            "cuda_version": None,
            "torch_version": torch.__version__,
        }
    else:
        return {
            "device_type": "cpu",
            "device_name": "CPU",
            "vram_gb": None,
            "cuda_version": None,
            "torch_version": torch.__version__,
        }
