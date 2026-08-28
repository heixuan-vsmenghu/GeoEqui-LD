from __future__ import annotations

import torch

from geoequi_ld.geometry.coordinates import normalized_to_pixel
from geoequi_ld.models.decoding import spatial_argmax


def test_spatial_argmax_maps_heatmap_corners_with_align_corners() -> None:
    heatmaps = torch.zeros((1, 2, 3, 5))
    heatmaps[0, 0, 0, 0] = 1.0
    heatmaps[0, 1, 2, 4] = 1.0
    normalized = spatial_argmax(heatmaps, align_corners=True)
    torch.testing.assert_close(
        normalized,
        torch.tensor([[[ -1.0, -1.0], [1.0, 1.0]]]),
    )


def test_spatial_argmax_uses_coordinate_contract_not_times_two() -> None:
    heatmaps = torch.zeros((1, 1, 256, 256))
    heatmaps[0, 0, 100, 200] = 7.0
    normalized = spatial_argmax(heatmaps, align_corners=True)
    original = normalized_to_pixel(normalized, (512, 512), align_corners=True)
    expected = torch.tensor([[[200.0 * 511.0 / 255.0, 100.0 * 511.0 / 255.0]]])
    torch.testing.assert_close(original, expected)
