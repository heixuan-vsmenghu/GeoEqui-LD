"""Explicit conversions between pixel and normalized coordinate systems.

All public functions use ``[..., 2]`` tensors ordered as ``[x, y]`` while
image tensors and sizes use ``[height, width]``.  The default
``align_corners=True`` contract maps the centres of the two corner pixels to
``-1`` and ``1``.  This makes normalized zero equal to pixel 127.5 for a
256-pixel axis and is the convention used throughout Phase 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

SizeHW = tuple[int, int]


def _size_hw(size_hw: Sequence[int]) -> SizeHW:
    if len(size_hw) != 2:
        raise ValueError(f"Expected [height, width], got {tuple(size_hw)!r}")
    height, width = (int(size_hw[0]), int(size_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}")
    return height, width


def _require_points(points_xy: Tensor) -> None:
    if points_xy.ndim < 1 or points_xy.shape[-1] != 2:
        raise ValueError(
            f"Points must have shape [..., 2] in [x, y] order, got {tuple(points_xy.shape)}"
        )
    if not torch.is_floating_point(points_xy):
        raise TypeError("Point tensors must use a floating dtype")


def pixel_to_normalized(
    points_xy: Tensor,
    size_hw: Sequence[int],
    *,
    align_corners: bool = True,
) -> Tensor:
    """Convert pixel-centre coordinates to the normalized ``[-1, 1]`` grid."""

    _require_points(points_xy)
    height, width = _size_hw(size_hw)
    x, y = points_xy.unbind(dim=-1)

    if align_corners:
        if width == 1 or height == 1:
            raise ValueError("align_corners=True is undefined for unit spatial dimensions")
        x_norm = 2.0 * x / float(width - 1) - 1.0
        y_norm = 2.0 * y / float(height - 1) - 1.0
    else:
        x_norm = 2.0 * (x + 0.5) / float(width) - 1.0
        y_norm = 2.0 * (y + 0.5) / float(height) - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def normalized_to_pixel(
    points_xy: Tensor,
    size_hw: Sequence[int],
    *,
    align_corners: bool = True,
) -> Tensor:
    """Convert normalized grid coordinates to pixel-centre coordinates."""

    _require_points(points_xy)
    height, width = _size_hw(size_hw)
    x_norm, y_norm = points_xy.unbind(dim=-1)

    if align_corners:
        if width == 1 or height == 1:
            raise ValueError("align_corners=True is undefined for unit spatial dimensions")
        x = (x_norm + 1.0) * float(width - 1) / 2.0
        y = (y_norm + 1.0) * float(height - 1) / 2.0
    else:
        x = (x_norm + 1.0) * float(width) / 2.0 - 0.5
        y = (y_norm + 1.0) * float(height) / 2.0 - 0.5
    return torch.stack((x, y), dim=-1)


def resize_points(
    points_xy: Tensor,
    from_size_hw: Sequence[int],
    to_size_hw: Sequence[int],
    *,
    align_corners: bool = True,
) -> Tensor:
    """Map points between two resized views using the shared grid contract."""

    normalized = pixel_to_normalized(points_xy, from_size_hw, align_corners=align_corners)
    return normalized_to_pixel(normalized, to_size_hw, align_corners=align_corners)


def points_in_bounds(points_xy: Tensor, size_hw: Sequence[int]) -> Tensor:
    """Return a boolean mask indicating whether every point lies in the image."""

    _require_points(points_xy)
    height, width = _size_hw(size_hw)
    x, y = points_xy.unbind(dim=-1)
    return (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
