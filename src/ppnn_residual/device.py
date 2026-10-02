"""Device selection and determinism settings (Stage 3). CPU behaviour of earlier stages is unchanged."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def set_deterministic(seed: int, device: torch.device) -> None:
    """Seed everything and request deterministic kernels. Must be called before CUDA work starts
    (CUBLAS_WORKSPACE_CONFIG is read when cuBLAS initializes)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_info(device: torch.device) -> dict:
    d = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        d.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version())
    else:
        d.update(threads=torch.get_num_threads())
    return d
