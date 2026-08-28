from __future__ import annotations

import torch

from geoequi_ld.geometry.coordinates import normalized_to_pixel, pixel_to_normalized, resize_points


def test_normalized_zero_is_pixel_127_5_for_width_256() -> None:
    normalized = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    pixel = normalized_to_pixel(normalized, (256, 256), align_corners=True)
    torch.testing.assert_close(pixel, torch.tensor([[127.5, 127.5]]))


def test_align_corners_boundaries_map_to_corner_pixel_centres() -> None:
    normalized = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32)
    pixel = normalized_to_pixel(normalized, (256, 256), align_corners=True)
    torch.testing.assert_close(pixel, torch.tensor([[0.0, 0.0], [255.0, 255.0]]))


def test_pixel_normalized_round_trip() -> None:
    points = torch.tensor([[0.0, 0.0], [511.0, 511.0], [123.25, 400.75]], dtype=torch.float64)
    recovered = normalized_to_pixel(
        pixel_to_normalized(points, (512, 512), align_corners=True),
        (512, 512),
        align_corners=True,
    )
    torch.testing.assert_close(recovered, points, atol=1e-10, rtol=1e-10)


def test_resize_points_preserves_normalized_location() -> None:
    original = torch.tensor([[0.0, 0.0], [511.0, 511.0], [255.5, 255.5]])
    resized = resize_points(original, (512, 512), (256, 256), align_corners=True)
    expected = torch.tensor([[0.0, 0.0], [255.0, 255.0], [127.5, 127.5]])
    torch.testing.assert_close(resized, expected)
