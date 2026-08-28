from __future__ import annotations

import pytest
import torch

from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps


def _peak_xy(heatmap: torch.Tensor) -> tuple[int, int]:
    width = heatmap.shape[-1]
    flat_index = int(torch.argmax(heatmap).item())
    return flat_index % width, flat_index // width


def test_gaussian_peak_matches_integer_target_and_channel_order() -> None:
    points = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    heatmaps = generate_gaussian_heatmaps(points, size_hw=(64, 64), sigma=2.0)
    assert heatmaps.shape == (3, 64, 64)
    assert [_peak_xy(channel) for channel in heatmaps] == [(10, 20), (30, 40), (50, 60)]


def test_boundary_points_are_supported() -> None:
    points = torch.tensor([[0.0, 0.0], [63.0, 63.0], [0.0, 63.0]])
    heatmaps = generate_gaussian_heatmaps(points, size_hw=(64, 64), sigma=2.0)
    assert [_peak_xy(channel) for channel in heatmaps] == [(0, 0), (63, 63), (0, 63)]
    assert torch.isfinite(heatmaps).all()


def test_invalid_mask_produces_zero_channel() -> None:
    points = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    heatmaps = generate_gaussian_heatmaps(
        points,
        size_hw=(64, 64),
        sigma=2.0,
        valid_mask=torch.tensor([True, False, True]),
    )
    assert float(heatmaps[1].abs().sum()) == 0.0


def test_out_of_bounds_valid_point_is_not_silently_clipped() -> None:
    with pytest.raises(ValueError, match="outside"):
        generate_gaussian_heatmaps(
            torch.tensor([[64.0, 5.0]]),
            size_hw=(64, 64),
            sigma=2.0,
            out_of_bounds="error",
        )
