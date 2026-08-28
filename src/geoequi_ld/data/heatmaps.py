"""Gaussian heatmap targets with explicit coordinate and validity handling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor


def generate_gaussian_heatmaps(
    points_xy: Tensor,
    *,
    size_hw: Sequence[int] = (256, 256),
    sigma: float = 4.0,
    valid_mask: Tensor | None = None,
    out_of_bounds: Literal["error", "mask"] = "error",
) -> Tensor:
    """Create one unnormalized Gaussian heatmap per keypoint.

    ``points_xy`` may be ``[K,2]`` or ``[B,K,2]`` and must already be in the
    output heatmap pixel coordinate system.  Invalid channels are filled with
    zeros.  ``sigma`` is measured in heatmap pixels.
    """

    if points_xy.ndim not in (2, 3) or points_xy.shape[-1] != 2:
        raise ValueError(f"Expected [K,2] or [B,K,2], got {tuple(points_xy.shape)}")
    if not torch.is_floating_point(points_xy):
        raise TypeError("Point tensors must use a floating dtype")
    if len(size_hw) != 2:
        raise ValueError("size_hw must be [height, width]")
    height, width = int(size_hw[0]), int(size_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError("Heatmap dimensions must be positive")
    if sigma <= 0 or not torch.isfinite(torch.tensor(sigma)):
        raise ValueError("sigma must be finite and positive")
    if out_of_bounds not in {"error", "mask"}:
        raise ValueError("out_of_bounds must be 'error' or 'mask'")

    squeeze_batch = points_xy.ndim == 2
    points = points_xy.unsqueeze(0) if squeeze_batch else points_xy
    batch_size, keypoint_count, _ = points.shape
    if valid_mask is None:
        valid = torch.ones((batch_size, keypoint_count), dtype=torch.bool, device=points.device)
    else:
        valid = valid_mask.to(device=points.device, dtype=torch.bool)
        if squeeze_batch and valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != (batch_size, keypoint_count):
            raise ValueError(
                "valid_mask must have shape "
                f"{(batch_size, keypoint_count)}, got {tuple(valid.shape)}"
            )

    x, y = points.unbind(dim=-1)
    finite = torch.isfinite(points).all(dim=-1)
    in_bounds = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    problematic = valid & (~finite | ~in_bounds)
    if out_of_bounds == "error" and bool(problematic.any()):
        bad = torch.nonzero(problematic, as_tuple=False).tolist()
        raise ValueError(f"Valid keypoints are non-finite or outside the heatmap: {bad[:8]}")
    effective_valid = valid & finite & in_bounds

    y_grid = torch.arange(height, device=points.device, dtype=points.dtype).view(1, 1, height, 1)
    x_grid = torch.arange(width, device=points.device, dtype=points.dtype).view(1, 1, 1, width)
    x_center = x.view(batch_size, keypoint_count, 1, 1)
    y_center = y.view(batch_size, keypoint_count, 1, 1)
    squared_distance = (x_grid - x_center).square() + (y_grid - y_center).square()
    heatmaps = torch.exp(-squared_distance / (2.0 * float(sigma) ** 2))
    heatmaps = heatmaps * effective_valid[..., None, None].to(dtype=heatmaps.dtype)
    return heatmaps.squeeze(0) if squeeze_batch else heatmaps
