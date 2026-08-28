from __future__ import annotations

import pytest
import torch

from geoequi_ld.geometry.aop import compute_aop


def _angle(points: torch.Tensor) -> torch.Tensor:
    return compute_aop(
        points,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
        output_unit="degrees",
    )


def test_right_angle_is_ninety_degrees() -> None:
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(_angle(points), torch.tensor(90.0), atol=1e-5, rtol=0.0)


def test_angle_is_invariant_to_translation_and_uniform_scale() -> None:
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
    translated = points + torch.tensor([13.0, -7.0])
    uniformly_scaled = points * 3.5
    torch.testing.assert_close(_angle(translated), _angle(points), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(_angle(uniformly_scaled), _angle(points), atol=1e-5, rtol=0.0)


def test_angle_is_invariant_to_rotation() -> None:
    points = torch.tensor([[0.0, 0.0], [2.0, 1.0], [-1.0, 3.0]])
    radians = torch.tensor(0.73)
    rotation = torch.tensor(
        [[torch.cos(radians), -torch.sin(radians)], [torch.sin(radians), torch.cos(radians)]]
    )
    rotated = points @ rotation.T
    torch.testing.assert_close(_angle(rotated), _angle(points), atol=1e-5, rtol=0.0)


def test_non_uniform_scale_is_allowed_to_change_angle() -> None:
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
    anisotropic = points * torch.tensor([2.0, 1.0])
    assert abs(float(_angle(anisotropic) - _angle(points))) > 1.0


def test_zero_length_vector_is_explicit() -> None:
    invalid_points = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="zero-length"):
        _angle(invalid_points)
    angle, valid = compute_aop(
        invalid_points,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
        invalid="mask",
    )
    assert not bool(valid)
    assert torch.isfinite(angle)
