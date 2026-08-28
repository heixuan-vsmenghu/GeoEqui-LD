"""Similarity transforms shared by images, points, and consistency losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def make_similarity_transform(
    scale: float | Tensor = 1.0,
    translation_xy: Sequence[float] | Tensor = (0.0, 0.0),
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create a forward 3x3 transform in normalized coordinates.

    The returned matrix maps original normalized coordinates to a transformed
    view.  Translation is therefore expressed in normalized ``[-1, 1]`` units.
    Only uniform scaling is accepted because AoP is not invariant to anisotropic
    scaling.
    """

    scale_tensor = torch.as_tensor(scale, dtype=dtype, device=device)
    translation = torch.as_tensor(translation_xy, dtype=dtype, device=device)
    if scale_tensor.ndim != 0:
        raise ValueError("Phase 0 make_similarity_transform expects a scalar uniform scale")
    if translation.shape != (2,):
        raise ValueError(f"translation_xy must contain [tx, ty], got {tuple(translation.shape)}")
    if not torch.isfinite(scale_tensor) or scale_tensor <= 0:
        raise ValueError("Scale must be finite and strictly positive")
    if not torch.isfinite(translation).all():
        raise ValueError("Translation must be finite")

    matrix = torch.eye(3, dtype=dtype, device=device)
    matrix[0, 0] = scale_tensor
    matrix[1, 1] = scale_tensor
    matrix[0, 2] = translation[0]
    matrix[1, 2] = translation[1]
    return matrix


def _validate_matrix(matrix: Tensor) -> None:
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected a [..., 3, 3] homogeneous matrix, got {tuple(matrix.shape)}")
    if not torch.is_floating_point(matrix):
        raise TypeError("Transform matrices must use a floating dtype")


def apply_similarity_transform(points_xy: Tensor, forward_matrix: Tensor) -> Tensor:
    """Apply a forward homogeneous transform to normalized ``[x, y]`` points."""

    if points_xy.ndim < 1 or points_xy.shape[-1] != 2:
        raise ValueError(f"Expected points [..., 2], got {tuple(points_xy.shape)}")
    _validate_matrix(forward_matrix)
    if not torch.is_floating_point(points_xy):
        raise TypeError("Point tensors must use a floating dtype")
    ones = torch.ones_like(points_xy[..., :1])
    homogeneous = torch.cat((points_xy, ones), dim=-1)
    transformed = torch.matmul(homogeneous, forward_matrix.transpose(-1, -2))
    denominator = transformed[..., 2:3]
    if torch.any(denominator.abs() <= torch.finfo(transformed.dtype).eps):
        raise ValueError("Transform produced an invalid homogeneous denominator")
    return transformed[..., :2] / denominator


def invert_similarity_transform(forward_matrix: Tensor) -> Tensor:
    """Invert a finite, non-singular homogeneous transform."""

    _validate_matrix(forward_matrix)
    determinant = torch.linalg.det(forward_matrix)
    eps = torch.finfo(forward_matrix.dtype).eps
    if torch.any(~torch.isfinite(determinant)) or torch.any(determinant.abs() <= eps):
        raise ValueError("Similarity transform is singular or non-finite")
    return torch.linalg.inv(forward_matrix)


def map_points_between_views(
    points_view1: Tensor, transform1: Tensor, transform2: Tensor
) -> Tensor:
    """Map points from view 1 to view 2 through their shared original view."""

    original = apply_similarity_transform(points_view1, invert_similarity_transform(transform1))
    return apply_similarity_transform(original, transform2)


def warp_image(
    image: Tensor,
    forward_matrix: Tensor,
    *,
    output_size_hw: Sequence[int] | None = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
    """Warp an image with the same forward transform used for keypoints.

    ``grid_sample`` expects an inverse mapping from output to input, so the
    forward point transform is inverted internally.  Input may be ``[C,H,W]``
    or ``[B,C,H,W]``; the returned rank matches the input rank.
    """

    if image.ndim not in (3, 4):
        raise ValueError(f"Expected image [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    squeeze_batch = image.ndim == 3
    batch = image.unsqueeze(0) if squeeze_batch else image
    if not torch.is_floating_point(batch):
        raise TypeError("Image tensors must use a floating dtype")

    _validate_matrix(forward_matrix)
    matrix = forward_matrix.to(device=batch.device, dtype=batch.dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.shape[0] == 1 and batch.shape[0] > 1:
        matrix = matrix.expand(batch.shape[0], -1, -1)
    if matrix.shape[0] != batch.shape[0]:
        raise ValueError("Transform batch dimension must be 1 or match the image batch")

    if output_size_hw is None:
        out_height, out_width = int(batch.shape[-2]), int(batch.shape[-1])
    else:
        if len(output_size_hw) != 2:
            raise ValueError("output_size_hw must be [height, width]")
        out_height, out_width = int(output_size_hw[0]), int(output_size_hw[1])
        if out_height <= 0 or out_width <= 0:
            raise ValueError("Output dimensions must be positive")

    inverse = invert_similarity_transform(matrix)
    theta = inverse[:, :2, :]
    output_shape = (batch.shape[0], batch.shape[1], out_height, out_width)
    grid = F.affine_grid(theta, output_shape, align_corners=align_corners)
    warped = F.grid_sample(
        batch,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return warped.squeeze(0) if squeeze_batch else warped


def horizontal_flip_points(
    points_xy: Tensor,
    *,
    width: int,
    swap_indices: tuple[int, int] | None = None,
) -> Tensor:
    """Flip pixel coordinates and optionally swap semantic left/right channels."""

    if width <= 0:
        raise ValueError("width must be positive")
    if points_xy.shape[-1] != 2:
        raise ValueError("Expected points with final dimension 2")
    flipped = points_xy.clone()
    flipped[..., 0] = float(width - 1) - flipped[..., 0]
    if swap_indices is not None:
        first, second = swap_indices
        swapped = flipped.clone()
        swapped[..., first, :] = flipped[..., second, :]
        swapped[..., second, :] = flipped[..., first, :]
        flipped = swapped
    return flipped
