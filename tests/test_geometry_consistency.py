from __future__ import annotations

import pytest
import torch

from geoequi_ld.geometry.aop import compute_aop
from geoequi_ld.geometry.coordinates import normalized_to_pixel, pixel_to_normalized
from geoequi_ld.geometry.transforms import (
    apply_similarity_transform,
    make_similarity_transform,
    warp_image,
)
from geoequi_ld.geometry_consistency import (
    combine_supervised_and_geometry,
    geometry_consistency_loss,
    transformed_keypoint_visibility,
)


def _points(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [[[-0.40, -0.25], [0.30, -0.10], [-0.15, 0.55]]],
        dtype=dtype,
    )


def _visible(batch_size: int = 1) -> torch.Tensor:
    return torch.ones((batch_size, 3), dtype=torch.bool)


def _paired_views(
    original: torch.Tensor,
    transform1: torch.Tensor,
    transform2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        apply_similarity_transform(original, transform1),
        apply_similarity_transform(original, transform2),
    )


def test_inverse_mapping_recovers_known_points_and_zero_consistency() -> None:
    original = _points()
    transform1 = make_similarity_transform(0.85, (0.10, -0.08), dtype=original.dtype)
    transform2 = make_similarity_transform(1.10, (-0.06, 0.04), dtype=original.dtype)
    view1, view2 = _paired_views(original, transform1, transform2)

    result = geometry_consistency_loss(
        view1,
        view2,
        transform1,
        transform2,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    )

    torch.testing.assert_close(
        result.angle_loss,
        torch.zeros_like(result.angle_loss),
        atol=1e-10,
        rtol=0,
    )
    torch.testing.assert_close(
        result.coordinate_loss,
        torch.zeros_like(result.coordinate_loss),
        atol=1e-10,
        rtol=0,
    )
    assert result.valid_point_count == 3
    assert result.valid_angle_count == 1
    assert result.skip_reason is None


def test_moving_one_prediction_has_explainable_loss_and_finite_gradients() -> None:
    original = _points()
    identity = make_similarity_transform(dtype=original.dtype)
    view1 = original.clone().requires_grad_()
    view2 = original.clone()
    view2[:, 2, 0] += 0.20
    view2.requires_grad_()

    result = geometry_consistency_loss(
        view1,
        view2,
        identity,
        identity,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    )
    # Only one of three corresponding points moved by 0.2 normalized units.
    torch.testing.assert_close(
        result.coordinate_loss,
        torch.tensor(0.2 / 3.0, dtype=original.dtype),
        atol=1e-10,
        rtol=0,
    )
    assert float(result.angle_loss) > 0
    result.total_loss.backward()
    assert view1.grad is not None and torch.isfinite(view1.grad).all()
    assert view2.grad is not None and torch.isfinite(view2.grad).all()
    assert torch.count_nonzero(view1.grad) > 0
    assert torch.count_nonzero(view2.grad) > 0


def test_translation_unit_uniform_scale_and_anisotropic_rejection() -> None:
    width = 101
    center = torch.tensor([[[0.0, 0.0]]])
    translated = apply_similarity_transform(
        center,
        make_similarity_transform(1.0, (0.1, 0.0)),
    )
    center_px = normalized_to_pixel(center, (51, width), align_corners=True)
    translated_px = normalized_to_pixel(translated, (51, width), align_corners=True)
    # tx=0.1 is in [-1,1] grid units: 0.1*(W-1)/2 = 0.05*(W-1) pixels.
    torch.testing.assert_close(
        translated_px[..., 0] - center_px[..., 0],
        torch.tensor([[0.05 * (width - 1)]]),
    )

    original = _points(torch.float32)
    transform1 = make_similarity_transform(0.7, (0.15, -0.10))
    transform2 = make_similarity_transform(1.2, (-0.12, 0.08))
    view1, view2 = _paired_views(original, transform1, transform2)
    result = geometry_consistency_loss(
        view1,
        view2,
        transform1,
        transform2,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    )
    assert float(result.angle_loss) < 1e-4

    anisotropic = torch.diag(torch.tensor([1.2, 0.8, 1.0]))
    with pytest.raises(ValueError, match="anisotropic"):
        geometry_consistency_loss(
            original,
            original,
            anisotropic,
            transform2,
            visibility_view1=_visible(),
            visibility_view2=_visible(),
        )


def test_coordinates_must_be_compared_after_mapping_to_common_origin() -> None:
    original = _points()
    transform1 = make_similarity_transform(0.8, (0.2, -0.1), dtype=original.dtype)
    transform2 = make_similarity_transform(1.15, (-0.15, 0.12), dtype=original.dtype)
    view1, view2 = _paired_views(original, transform1, transform2)
    assert float(torch.linalg.vector_norm(view1 - view2, dim=-1).mean()) > 0.1

    result = geometry_consistency_loss(
        view1,
        view2,
        transform1,
        transform2,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    )
    assert float(result.coordinate_loss) < 1e-10


def test_warped_bright_point_and_visibility_agree_at_boundary_and_crop() -> None:
    image = torch.zeros((1, 21, 21), dtype=torch.float32)
    image[0, 10, 10] = 1.0
    original_pixel = torch.tensor([[[10.0, 10.0]]])
    original = pixel_to_normalized(original_pixel, (21, 21), align_corners=True)
    shift_to_edge = make_similarity_transform(1.0, (1.0, 0.0))
    warped = warp_image(image, shift_to_edge, mode="nearest", align_corners=True)
    transformed = apply_similarity_transform(original, shift_to_edge)
    expected_pixel = normalized_to_pixel(transformed, (21, 21), align_corners=True)
    peak = int(torch.argmax(warped).detach())
    peak_xy = torch.tensor([[[float(peak % 21), float(peak // 21)]]])
    torch.testing.assert_close(peak_xy, expected_pixel)
    assert bool(transformed_keypoint_visibility(original, shift_to_edge)[0, 0])

    cropped = make_similarity_transform(1.0, (1.1, 0.0))
    cropped_image = warp_image(image, cropped, mode="nearest", align_corners=True)
    assert not bool(transformed_keypoint_visibility(original, cropped)[0, 0])
    assert torch.count_nonzero(cropped_image) == 0


def test_degenerate_geometry_is_marked_invalid_not_rewarded() -> None:
    points = torch.zeros((1, 3, 2), requires_grad=True)
    identity = make_similarity_transform()
    result = geometry_consistency_loss(
        points,
        points,
        identity,
        identity,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    )
    assert result.valid_angle_count == 0
    assert result.skip_reason == "no_valid_geometry"
    assert result.no_valid_geometry
    torch.testing.assert_close(result.total_loss, torch.zeros_like(result.total_loss))
    result.total_loss.backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_two_consistent_but_wrong_views_do_not_prove_accuracy() -> None:
    truth = _points()
    consistently_wrong = truth + torch.tensor([0.25, -0.20], dtype=truth.dtype)
    transform1 = make_similarity_transform(0.9, (0.1, 0.0), dtype=truth.dtype)
    transform2 = make_similarity_transform(1.1, (-0.1, 0.05), dtype=truth.dtype)
    view1, view2 = _paired_views(consistently_wrong, transform1, transform2)
    result = geometry_consistency_loss(
        view1,
        view2,
        transform1,
        transform2,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    )
    assert float(result.total_loss) < 1e-10
    assert float(torch.linalg.vector_norm(consistently_wrong - truth, dim=-1).mean()) > 0.2


def test_batch_transforms_do_not_mix_samples() -> None:
    original = torch.cat((_points(), _points() + torch.tensor([0.1, -0.15])), dim=0)
    transform1 = torch.stack(
        (
            make_similarity_transform(0.8, (0.1, -0.1), dtype=original.dtype),
            make_similarity_transform(1.2, (-0.2, 0.05), dtype=original.dtype),
        )
    )
    transform2 = torch.stack(
        (
            make_similarity_transform(1.1, (-0.1, 0.03), dtype=original.dtype),
            make_similarity_transform(0.7, (0.15, -0.12), dtype=original.dtype),
        )
    )
    view1, view2 = _paired_views(original, transform1, transform2)
    result = geometry_consistency_loss(
        view1,
        view2,
        transform1,
        transform2,
        visibility_view1=_visible(2),
        visibility_view2=_visible(2),
    )
    assert float(result.total_loss) < 1e-10
    assert result.valid_point_count == 6
    assert result.valid_angle_count == 2

    mixed = geometry_consistency_loss(
        view1,
        view2.flip(0),
        transform1,
        transform2,
        visibility_view1=_visible(2),
        visibility_view2=_visible(2),
    )
    assert float(mixed.coordinate_loss) > 0.05


def test_angle_uses_pixel_axes_for_non_square_images() -> None:
    image_size = (101, 201)
    original_pixels = torch.tensor(
        [[[50.0, 50.0], [150.0, 50.0], [50.0, 80.0]]], dtype=torch.float64
    )
    original = pixel_to_normalized(original_pixels, image_size, align_corners=True)
    changed_pixels = original_pixels.clone()
    changed_pixels[:, 2, 0] += 30.0
    changed = pixel_to_normalized(changed_pixels, image_size, align_corners=True)
    identity = make_similarity_transform(dtype=original.dtype)

    result = geometry_consistency_loss(
        original,
        changed,
        identity,
        identity,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
        image_size_hw=image_size,
    )
    expected1 = compute_aop(
        original_pixels,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
    )
    expected2 = compute_aop(
        changed_pixels,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
    )
    torch.testing.assert_close(result.angle_loss, (expected1 - expected2).abs().mean())

    # Directly measuring in separately normalized x/y axes gives another angle
    # and is precisely the non-square-image mistake guarded by this test.
    normalized1 = compute_aop(
        original,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
    )
    normalized2 = compute_aop(
        changed,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
    )
    assert abs(float(result.angle_loss - (normalized1 - normalized2).abs())) > 10.0


def test_both_prediction_paths_receive_gradients() -> None:
    original = _points(torch.float32)
    transform1 = make_similarity_transform(0.9, (0.08, -0.04))
    transform2 = make_similarity_transform(1.1, (-0.05, 0.07))
    view1, view2 = _paired_views(original, transform1, transform2)
    prediction1 = (view1 + torch.tensor([0.02, -0.01])).detach().requires_grad_()
    prediction2 = (view2 + torch.tensor([-0.03, 0.04])).detach().requires_grad_()
    result = geometry_consistency_loss(
        prediction1,
        prediction2,
        transform1,
        transform2,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    )
    gradient1, gradient2 = torch.autograd.grad(result.total_loss, (prediction1, prediction2))
    assert torch.isfinite(gradient1).all() and torch.count_nonzero(gradient1) > 0
    assert torch.isfinite(gradient2).all() and torch.count_nonzero(gradient2) > 0


def test_lambda_geo_zero_preserves_supervised_gradient_exactly() -> None:
    weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    supervised = (weight * 2.0 - 1.0).square()
    baseline_gradient = torch.autograd.grad(supervised, weight, retain_graph=True)[0]

    original = _points() * weight
    identity = make_similarity_transform(dtype=original.dtype)
    perturbed = original + torch.tensor([0.03, -0.02], dtype=original.dtype)
    auxiliary = geometry_consistency_loss(
        original,
        perturbed,
        identity,
        identity,
        visibility_view1=_visible(),
        visibility_view2=_visible(),
    ).total_loss
    combined = combine_supervised_and_geometry(supervised, auxiliary, lambda_geo=0.0)
    combined_gradient = torch.autograd.grad(combined, weight)[0]
    torch.testing.assert_close(combined_gradient, baseline_gradient, atol=0.0, rtol=0.0)
