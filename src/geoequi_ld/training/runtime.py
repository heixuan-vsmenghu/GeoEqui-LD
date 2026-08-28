"""Deterministic runtime helpers shared by training scripts."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import Generator


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch CPU/CUDA, and deterministic backends."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``/CPU/CUDA with an explicit availability check."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def make_generator(seed: int) -> Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_data_loader_worker(worker_id: int) -> None:
    """Derive NumPy/Python worker seeds from PyTorch's worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
