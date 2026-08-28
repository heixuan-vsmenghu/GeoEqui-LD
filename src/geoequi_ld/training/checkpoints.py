"""Portable supervised checkpoint save/load helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: Mapping[str, Any],
    seed: int,
    metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model, optimizer, epoch, configuration, seed, and metrics."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "seed": int(seed),
        "config": dict(config),
        "metrics": dict(metrics),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def read_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    required = {
        "epoch",
        "seed",
        "config",
        "metrics",
        "model_state_dict",
        "optimizer_state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    return payload


def restore_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = read_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
