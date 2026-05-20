# ==============================================================================
# SceneShift — Dockerfile
# Multi-stage build for GPU-accelerated AI image editing pipeline
# Base: NVIDIA CUDA 12.1 + cuDNN 8 + Ubuntu 22.04
# ==============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3.10 as python/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM2 from GitHub
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/sam2.git"

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Create output and log directories ─────────────────────────────────────────
RUN mkdir -p outputs logs

# ── Model cache directory (mount as volume for persistence) ───────────────────
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

# ── CUDA environment ────────────────────────────────────────────────────────────
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 7860 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
