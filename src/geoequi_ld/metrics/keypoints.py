"""Metrics with an explicit common pixel coordinate system."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def radial_errors(
    predicted_xy: Tensor, target_xy: Tensor, valid_mask: Tensor | None = None
) -> Tensor:
    """Return Euclidean point errors for tensors shaped ``[..., K, 2]``."""

    if predicted_xy.shape != target_xy.shape or predicted_xy.shape[-1] != 2:
        raise ValueError(
            "Prediction and target must share [...,K,2] shape, got "
            f"{tuple(predicted_xy.shape)} and {tuple(target_xy.shape)}"
        )
    if not torch.is_floating_point(predicted_xy) or not torch.is_floating_point(target_xy):
        raise TypeError("Metric inputs must use floating dtypes")
    errors = torch.linalg.vector_norm(predicted_xy - target_xy, dim=-1)
    if valid_mask is not None:
        mask = valid_mask.to(device=errors.device, dtype=torch.bool)
        if mask.shape != errors.shape:
            raise ValueError(
                f"valid_mask must have shape {tuple(errors.shape)}, got {tuple(mask.shape)}"
            )
        errors = torch.where(mask, errors, torch.full_like(errors, torch.nan))
    return errors


def absolute_angle_error(predicted_degrees: Tensor, target_degrees: Tensor) -> Tensor:
    """Return absolute unsigned AoP error in degrees."""

    if predicted_degrees.shape != target_degrees.shape:
        raise ValueError("Predicted and target angle tensors must have the same shape")
    return torch.abs(predicted_degrees - target_degrees)


def summarize_keypoint_metrics(
    predicted_xy: Tensor,
    target_xy: Tensor,
    *,
    keypoint_names: Sequence[str],
    valid_mask: Tensor | None = None,
) -> dict[str, float]:
    """Summarize per-keypoint MRE and their global finite mean."""

    errors = radial_errors(predicted_xy, target_xy, valid_mask)
    if errors.ndim != 2:
        raise ValueError("Summary expects batched inputs [N,K,2]")
    if len(keypoint_names) != errors.shape[1]:
        raise ValueError("keypoint_names length must match K")
    output: dict[str, float] = {}
    for index, name in enumerate(keypoint_names):
        values = errors[:, index]
        finite = values[torch.isfinite(values)]
        output[f"MRE_{name}"] = float(finite.mean().item()) if finite.numel() else float("nan")
    finite_all = errors[torch.isfinite(errors)]
    output["MRE_ALL"] = float(finite_all.mean().item()) if finite_all.numel() else float("nan")
    return output
