"""Differentiable two-view geometry consistency for Phase 2A.

The functions in this module deliberately reuse the transform convention in
``geoequi_ld.geometry.transforms``: a forward matrix maps normalized original
coordinates to normalized view coordinates.  Predictions are therefore
inverse-mapped before they are compared.  No image or label lookup happens in
this module.

The Phase 2A engineering loss is

``angle_difference_degrees + 0.1 * normalized_coordinate_distance``.

The angle is measured after converting the common original coordinates to
original-image pixels.  This matters for non-square images: independently
normalizing x and y does not preserve Euclidean angles when the two pixel axes
have different extents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from geoequi_ld.geometry.aop import compute_aop
from geoequi_ld.geometry.coordinates import normalized_to_pixel
from geoequi_ld.geometry.transforms import (
    apply_similarity_transform,
    invert_similarity_transform,
)


@dataclass(frozen=True)
class GeometryConsistencyResult:
    """Scalar losses and detached bookkeeping for one paired-view batch.

    ``angle_loss`` is in degrees. ``coordinate_loss`` is a Euclidean distance
    in the common normalized ``[-1, 1]`` original-image coordinates.
    ``total_loss`` stays attached to both prediction graphs.
    """

    angle_loss: Tensor
    coordinate_loss: Tensor
    total_loss: Tensor
    valid_point_count: int
    valid_angle_count: int
    skip_reason: str | None

    @property
    def no_valid_geometry(self) -> bool:
        """Whether no complete, non-degenerate AoP structure was available."""

        return self.valid_angle_count == 0


def _require_paired_points(points_view1: Tensor, points_view2: Tensor) -> tuple[int, int]:
    if points_view1.ndim != 3 or points_view1.shape[-1] != 2:
        raise ValueError(
            "points_view1 must have shape [batch, keypoints, 2], "
            f"got {tuple(points_view1.shape)}"
        )
    if points_view2.shape != points_view1.shape:
        raise ValueError(
            "points_view2 must match points_view1, "
            f"got {tuple(points_view2.shape)} versus {tuple(points_view1.shape)}"
        )
    if not torch.is_floating_point(points_view1) or not torch.is_floating_point(points_view2):
        raise TypeError("Prediction coordinates must use a floating dtype")
    if points_view1.device != points_view2.device:
        raise ValueError("Both prediction views must be on the same device")
    if points_view1.dtype != points_view2.dtype:
        raise ValueError("Both prediction views must use the same dtype")
    batch_size, keypoint_count = points_view1.shape[:2]
    if batch_size < 1:
        raise ValueError("At least one paired view is required")
    if keypoint_count < 3:
        raise ValueError("AoP geometry requires at least three keypoints")
    return batch_size, keypoint_count


def _batched_similarity_matrix(
    forward_matrix: Tensor,
    *,
    batch_size: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if forward_matrix.ndim == 2:
        forward_matrix = forward_matrix.unsqueeze(0)
    if forward_matrix.ndim != 3 or forward_matrix.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have shape [3, 3] or [batch, 3, 3]")
    if forward_matrix.shape[0] == 1 and batch_size > 1:
        forward_matrix = forward_matrix.expand(batch_size, -1, -1)
    if forward_matrix.shape[0] != batch_size:
        raise ValueError(f"{name} batch dimension must be 1 or {batch_size}")
    if not torch.is_floating_point(forward_matrix):
        raise TypeError(f"{name} must use a floating dtype")

    matrix = forward_matrix.to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} contains non-finite values")

    # A 2-D similarity may include rotation, but its two linear columns must be
    # orthogonal and have equal length. Reflections and projective rows are not
    # part of the Phase 2A uniform-scale/translation augmentation contract.
    linear = matrix[:, :2, :2]
    gram = linear.transpose(-1, -2) @ linear
    scale_squared = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1)
    target_gram = scale_squared[:, None, None] * torch.eye(
        2, device=matrix.device, dtype=matrix.dtype
    )
    tolerance = 1e-5 if matrix.dtype in (torch.float32, torch.complex64) else 1e-8
    similarity_ok = torch.isclose(gram, target_gram, atol=tolerance, rtol=tolerance).all(
        dim=(-1, -2)
    )
    affine_row = torch.tensor([0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype)
    affine_ok = torch.isclose(
        matrix[:, 2, :], affine_row.expand(batch_size, -1), atol=tolerance, rtol=tolerance
    ).all(dim=-1)
    determinant = torch.linalg.det(linear)
    valid = similarity_ok & affine_ok & (scale_squared > tolerance) & (determinant > tolerance)
    if not bool(valid.all()):
        raise ValueError(
            f"{name} must be a finite, non-singular, orientation-preserving similarity "
            "transform; anisotropic scaling and shear are not valid augmentations"
        )
    return matrix


def _visibility_mask(
    mask: Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
    name: str,
) -> Tensor:
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean tensor")
    return mask.to(device=device)


def transformed_keypoint_visibility(
    points_original: Tensor,
    forward_matrix: Tensor,
) -> Tensor:
    """Compute synthetic visibility from known normalized points and a transform.

    This helper is intended for labelled/synthetic diagnostics.  It must not be
    described as true visibility for an unlabelled image merely because a model
    prediction lies inside the frame.
    """

    if points_original.ndim != 3 or points_original.shape[-1] != 2:
        raise ValueError("points_original must have shape [batch, keypoints, 2]")
    if not torch.is_floating_point(points_original):
        raise TypeError("points_original must use a floating dtype")
    matrix = _batched_similarity_matrix(
        forward_matrix,
        batch_size=points_original.shape[0],
        reference=points_original,
        name="forward_matrix",
    )
    finite = torch.isfinite(points_original).all(dim=-1)
    safe_points = torch.where(
        finite.unsqueeze(-1), points_original, torch.zeros_like(points_original)
    )
    transformed = apply_similarity_transform(safe_points, matrix)
    in_frame = (transformed >= -1.0).all(dim=-1) & (transformed <= 1.0).all(dim=-1)
    return finite & in_frame


def _masked_mean(values: Tensor, mask: Tensor, graph_zero: Tensor) -> Tensor:
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    count = mask.sum().to(dtype=values.dtype)
    mean = safe_values.sum() / count.clamp_min(1.0)
    return torch.where(mask.any(), mean, graph_zero)


def _safe_aop_degrees(
    points_pixels: Tensor,
    *,
    angle_indices: tuple[int, int, int],
    visible: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute AoP without exposing autograd to ``atan2(0, 0)``.

    ``compute_aop(..., invalid="mask")`` correctly masks the forward value for
    a zero-length ray, but the derivative of ``atan2`` at the origin is not
    defined.  Replacing invalid structures by a fixed right-angle triangle
    before that call keeps the skipped path finite while the returned validity
    mask still reflects the real prediction.
    """

    vertex_index, ps2_index, fh1_index = angle_indices
    vertex = points_pixels[:, vertex_index]
    ps2_ray = points_pixels[:, ps2_index] - vertex
    fh1_ray = points_pixels[:, fh1_index] - vertex
    finite = torch.isfinite(points_pixels[:, list(angle_indices)]).all(dim=(-1, -2))
    nondegenerate = (
        visible
        & finite
        & (torch.linalg.vector_norm(ps2_ray, dim=-1) > eps)
        & (torch.linalg.vector_norm(fh1_ray, dim=-1) > eps)
    )

    fallback = torch.zeros_like(points_pixels)
    fallback[:, ps2_index, 0] = 1.0
    fallback[:, fh1_index, 1] = 1.0
    safe_points = torch.where(nondegenerate[:, None, None], points_pixels, fallback)
    angles = compute_aop(
        safe_points,
        vertex_index=vertex_index,
        pubic_axis_other_index=ps2_index,
        fetal_head_index=fh1_index,
        output_unit="degrees",
        eps=eps,
        invalid="raise",
    )
    return angles, nondegenerate


def geometry_consistency_loss(
    points_view1: Tensor,
    points_view2: Tensor,
    transform1: Tensor,
    transform2: Tensor,
    *,
    visibility_view1: Tensor,
    visibility_view2: Tensor,
    image_size_hw: Sequence[int] = (512, 512),
    angle_indices: tuple[int, int, int] = (0, 1, 2),
    coordinate_weight: float = 0.1,
    eps: float = 1e-8,
) -> GeometryConsistencyResult:
    """Compare two predictions after inverse mapping to their common origin.

    Args:
        points_view1: Normalized ``[B, K, 2]`` predictions in view 1.
        points_view2: Normalized ``[B, K, 2]`` predictions in view 2.
        transform1: Forward original-to-view-1 similarity transform.
        transform2: Forward original-to-view-2 similarity transform.
        visibility_view1: Explicit per-point visibility in view 1.
        visibility_view2: Explicit per-point visibility in view 2.
        image_size_hw: Original pixel ``[height, width]`` used for AoP only.
        angle_indices: ``(PS1 vertex, PS2 endpoint, FH1 endpoint)`` indices.
        coordinate_weight: Phase 2A engineering coefficient, fixed by protocol
            to 0.1 for its acceptance tests.

    Coordinates are compared in normalized original-image units.  AoP is
    computed in original pixels and returned in degrees.  A batch without a
    valid angle is explicitly marked ``no_valid_geometry``; graph-connected
    zero terms are returned when no corresponding valid values exist.
    """

    batch_size, keypoint_count = _require_paired_points(points_view1, points_view2)
    if len(image_size_hw) != 2 or any(int(value) <= 1 for value in image_size_hw):
        raise ValueError("image_size_hw must contain height and width greater than one")
    if len(angle_indices) != 3 or len(set(angle_indices)) != 3:
        raise ValueError("angle_indices must contain three distinct keypoint indices")
    if any(index < 0 or index >= keypoint_count for index in angle_indices):
        raise IndexError(f"angle_indices {angle_indices} are invalid for K={keypoint_count}")
    if not torch.isfinite(torch.tensor(coordinate_weight)) or coordinate_weight < 0:
        raise ValueError("coordinate_weight must be finite and non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    matrix1 = _batched_similarity_matrix(
        transform1,
        batch_size=batch_size,
        reference=points_view1,
        name="transform1",
    )
    matrix2 = _batched_similarity_matrix(
        transform2,
        batch_size=batch_size,
        reference=points_view2,
        name="transform2",
    )
    visible1 = _visibility_mask(
        visibility_view1,
        shape=(batch_size, keypoint_count),
        device=points_view1.device,
        name="visibility_view1",
    )
    visible2 = _visibility_mask(
        visibility_view2,
        shape=(batch_size, keypoint_count),
        device=points_view2.device,
        name="visibility_view2",
    )

    finite1 = torch.isfinite(points_view1).all(dim=-1)
    finite2 = torch.isfinite(points_view2).all(dim=-1)
    safe_view1 = torch.where(finite1.unsqueeze(-1), points_view1, torch.zeros_like(points_view1))
    safe_view2 = torch.where(finite2.unsqueeze(-1), points_view2, torch.zeros_like(points_view2))

    original1 = apply_similarity_transform(safe_view1, invert_similarity_transform(matrix1))
    original2 = apply_similarity_transform(safe_view2, invert_similarity_transform(matrix2))
    valid_points = visible1 & visible2 & finite1 & finite2

    coordinate_distances = torch.linalg.vector_norm(original1 - original2, dim=-1)
    graph_zero = (original1.sum() + original2.sum()) * 0.0
    coordinate_loss = _masked_mean(coordinate_distances, valid_points, graph_zero)

    original1_pixels = normalized_to_pixel(original1, image_size_hw, align_corners=True)
    original2_pixels = normalized_to_pixel(original2, image_size_hw, align_corners=True)
    selected_visibility = valid_points[:, list(angle_indices)].all(dim=-1)
    angle1, nondegenerate1 = _safe_aop_degrees(
        original1_pixels,
        angle_indices=angle_indices,
        visible=selected_visibility,
        eps=eps,
    )
    angle2, nondegenerate2 = _safe_aop_degrees(
        original2_pixels,
        angle_indices=angle_indices,
        visible=selected_visibility,
        eps=eps,
    )
    valid_angles = nondegenerate1 & nondegenerate2
    angle_loss = _masked_mean((angle1 - angle2).abs(), valid_angles, graph_zero)
    total_loss = angle_loss + float(coordinate_weight) * coordinate_loss

    # These conversions are bookkeeping only.  The three loss tensors above
    # remain connected to both prediction branches.
    valid_point_count = int(valid_points.sum().detach().cpu())
    valid_angle_count = int(valid_angles.sum().detach().cpu())
    skip_reason = "no_valid_geometry" if valid_angle_count == 0 else None
    return GeometryConsistencyResult(
        angle_loss=angle_loss,
        coordinate_loss=coordinate_loss,
        total_loss=total_loss,
        valid_point_count=valid_point_count,
        valid_angle_count=valid_angle_count,
        skip_reason=skip_reason,
    )


def combine_supervised_and_geometry(
    supervised_loss: Tensor,
    geometry_loss: Tensor,
    *,
    lambda_geo: float,
) -> Tensor:
    """Add the auxiliary term without altering supervision when ``lambda_geo=0``."""

    if supervised_loss.ndim != 0 or geometry_loss.ndim != 0:
        raise ValueError("supervised_loss and geometry_loss must be scalar tensors")
    if not torch.isfinite(torch.tensor(lambda_geo)) or lambda_geo < 0:
        raise ValueError("lambda_geo must be finite and non-negative")
    return supervised_loss + float(lambda_geo) * geometry_loss


__all__ = [
    "GeometryConsistencyResult",
    "combine_supervised_and_geometry",
    "geometry_consistency_loss",
    "transformed_keypoint_visibility",
]
