from __future__ import annotations

import pytest
import torch

from geoequi_ld.metrics.keypoints import (
    absolute_angle_error,
    radial_errors,
    summarize_keypoint_metrics,
)


def test_known_radial_distance_is_ten_pixels() -> None:
    predicted = torch.tensor([[[26.0, 38.0]]])
    target = torch.tensor([[[20.0, 30.0]]])
    torch.testing.assert_close(radial_errors(predicted, target), torch.tensor([[10.0]]))


def test_known_angle_difference_is_three_point_five_degrees() -> None:
    error = absolute_angle_error(torch.tensor([102.4]), torch.tensor([98.9]))
    torch.testing.assert_close(error, torch.tensor([3.5]), atol=1e-5, rtol=0.0)


def test_summary_reports_each_keypoint_and_global_mean() -> None:
    target = torch.zeros((2, 3, 2))
    predicted = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]],
            [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]],
        ]
    )
    summary = summarize_keypoint_metrics(predicted, target, keypoint_names=("PS1", "PS2", "FH1"))
    assert summary["MRE_PS1"] == pytest.approx(2.5)
    assert summary["MRE_PS2"] == pytest.approx(2.5)
    assert summary["MRE_FH1"] == pytest.approx(5.0)
    assert summary["MRE_ALL"] == pytest.approx(10.0 / 3.0)
