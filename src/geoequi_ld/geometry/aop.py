"""Stable, differentiable Angle of Progression geometry."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def compute_aop(
    points_xy: Tensor,
    *,
    vertex_index: int,
    pubic_axis_other_index: int,
    fetal_head_index: int,
    output_unit: Literal["degrees", "radians"] = "degrees",
    eps: float = 1e-8,
    invalid: Literal["raise", "mask"] = "raise",
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute the unsigned angle between pubic-axis and fetal-head rays.

    Point semantics are deliberately supplied by the caller.  This prevents a
    hidden PS1/PS2 assumption from leaking into reusable geometry code.

    Args:
        points_xy: Tensor shaped ``[..., K, 2]`` in any common coordinate system.
        vertex_index: Index of the AoP vertex.
        pubic_axis_other_index: Other endpoint of the pubic symphysis axis.
        fetal_head_index: Fetal-head tangent point index.
        invalid: ``"raise"`` rejects zero-length rays. ``"mask"`` returns
            ``(angle, valid_mask)`` and sets invalid angles to zero.
    """

    if points_xy.ndim < 2 or points_xy.shape[-1] != 2:
        raise ValueError(f"Expected points [..., K, 2], got {tuple(points_xy.shape)}")
    if not torch.is_floating_point(points_xy):
        raise TypeError("Point tensors must use a floating dtype")
    keypoint_count = points_xy.shape[-2]
    indices = (vertex_index, pubic_axis_other_index, fetal_head_index)
    if any(index < 0 or index >= keypoint_count for index in indices):
        raise IndexError(f"AoP indices {indices} are invalid for K={keypoint_count}")
    if len(set(indices)) != 3:
        raise ValueError("AoP requires three distinct keypoint indices")
    if eps <= 0:
        raise ValueError("eps must be positive")

    vertex = points_xy[..., vertex_index, :]
    pubic_vector = points_xy[..., pubic_axis_other_index, :] - vertex
    fetal_vector = points_xy[..., fetal_head_index, :] - vertex
    pubic_norm = torch.linalg.vector_norm(pubic_vector, dim=-1)
    fetal_norm = torch.linalg.vector_norm(fetal_vector, dim=-1)
    valid = torch.isfinite(points_xy).all(dim=(-1, -2)) & (pubic_norm > eps) & (fetal_norm > eps)
    if invalid == "raise" and not bool(valid.all()):
        raise ValueError("AoP is undefined for non-finite points or a zero-length ray")
    if invalid not in {"raise", "mask"}:
        raise ValueError("invalid must be either 'raise' or 'mask'")

    # atan2(|cross|, dot) yields the same unsigned angle as arccos while being
    # better conditioned near 0 and 180 degrees.
    dot = torch.sum(pubic_vector * fetal_vector, dim=-1)
    cross = (
        pubic_vector[..., 0] * fetal_vector[..., 1] - pubic_vector[..., 1] * fetal_vector[..., 0]
    )
    angles = torch.atan2(cross.abs(), dot)
    if output_unit == "degrees":
        angles = torch.rad2deg(angles)
    elif output_unit != "radians":
        raise ValueError("output_unit must be 'degrees' or 'radians'")

    if invalid == "mask":
        angles = torch.where(valid, angles, torch.zeros_like(angles))
        return angles, valid
    return angles
