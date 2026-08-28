"""Coordinate decoders for supervised landmark heatmaps."""

from __future__ import annotations

import torch
from torch import Tensor


def spatial_argmax(
    heatmaps: Tensor,
    *,
    align_corners: bool,
) -> Tensor:
    """Decode each heatmap maximum into normalized ``[x, y]`` coordinates.

    No quarter-pixel, smoothing, or other post-processing is applied.  The
    conversion follows the same ``align_corners`` contract as DSNT and image
    resizing, so a 256-to-512 mapping is not approximated by multiplying by 2.
    """

    if heatmaps.ndim != 4:
        raise ValueError("heatmaps must have shape [B,K,H,W]")
    height, width = heatmaps.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError("heatmap dimensions must be positive")
    flat_indices = heatmaps.flatten(start_dim=-2).argmax(dim=-1)
    y = torch.div(flat_indices, width, rounding_mode="floor").to(heatmaps.dtype)
    x = torch.remainder(flat_indices, width).to(heatmaps.dtype)
    if align_corners:
        if height == 1 or width == 1:
            raise ValueError("align_corners=True requires heatmap dimensions greater than one")
        x = 2.0 * x / float(width - 1) - 1.0
        y = 2.0 * y / float(height - 1) - 1.0
    else:
        x = (2.0 * x + 1.0) / float(width) - 1.0
        y = (2.0 * y + 1.0) / float(height) - 1.0
    return torch.stack((x, y), dim=-1)


def decode_heatmaps(
    heatmaps: Tensor,
    *,
    method: str,
    dsnt: torch.nn.Module,
    align_corners: bool,
) -> Tensor:
    """Decode heatmaps with a predeclared method."""

    normalized = method.strip().lower()
    if normalized == "dsnt":
        return dsnt(heatmaps)
    if normalized == "argmax":
        return spatial_argmax(heatmaps, align_corners=align_corners)
    raise ValueError(f"Unsupported decoder: {method!r}")
