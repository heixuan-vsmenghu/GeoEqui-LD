from __future__ import annotations

import torch

from geoequi_ld.geometry.coordinates import normalized_to_pixel, pixel_to_normalized
from geoequi_ld.geometry.transforms import (
    apply_similarity_transform,
    horizontal_flip_points,
    invert_similarity_transform,
    make_similarity_transform,
    map_points_between_views,
    warp_image,
)


def test_forward_then_inverse_recovers_points() -> None:
    points = torch.tensor([[-0.5, 0.2], [0.1, -0.3], [0.7, 0.6]])
    transform = make_similarity_transform(1.15, (0.08, -0.06))
    recovered = apply_similarity_transform(
        apply_similarity_transform(points, transform),
        invert_similarity_transform(transform),
    )
    torch.testing.assert_close(recovered, points, atol=1e-6, rtol=1e-6)


def test_mapping_between_two_views_matches_direct_composition() -> None:
    original = torch.tensor([[-0.2, 0.1], [0.4, -0.5]])
    first = make_similarity_transform(0.9, (0.1, 0.0))
    second = make_similarity_transform(1.1, (-0.05, 0.07))
    in_first = apply_similarity_transform(original, first)
    mapped = map_points_between_views(in_first, first, second)
    expected = apply_similarity_transform(original, second)
    torch.testing.assert_close(mapped, expected, atol=1e-6, rtol=1e-6)


def test_image_and_point_use_the_same_forward_transform() -> None:
    image = torch.zeros((1, 33, 33), dtype=torch.float32)
    original_pixel = torch.tensor([[16.0, 16.0]])
    image[0, 16, 16] = 1.0
    transform = make_similarity_transform(1.0, (0.2, -0.1))
    warped = warp_image(image, transform, align_corners=True)
    transformed_normalized = apply_similarity_transform(
        pixel_to_normalized(original_pixel, (33, 33), align_corners=True),
        transform,
    )
    expected_pixel = normalized_to_pixel(transformed_normalized, (33, 33), align_corners=True)[0]
    peak_index = int(torch.argmax(warped).item())
    peak_xy = torch.tensor([peak_index % 33, peak_index // 33], dtype=torch.float32)
    torch.testing.assert_close(peak_xy, expected_pixel, atol=0.6, rtol=0.0)


def test_horizontal_flip_can_swap_semantic_endpoint_channels() -> None:
    points = torch.tensor([[7.0, 2.0], [2.0, 3.0], [5.0, 4.0]])
    flipped = horizontal_flip_points(points, width=10, swap_indices=(0, 1))
    expected = torch.tensor([[7.0, 3.0], [2.0, 2.0], [4.0, 4.0]])
    torch.testing.assert_close(flipped, expected)
