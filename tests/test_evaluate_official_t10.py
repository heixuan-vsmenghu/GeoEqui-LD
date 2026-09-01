from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.evaluate_official_t10 import (
    aop_degrees,
    decode_argmax_64_to_512,
    main,
    summarize_validation_metrics,
)


def test_decode_argmax_maps_each_64_heatmap_bin_to_eight_pixels() -> None:
    heatmaps = torch.zeros((1, 3, 64, 64))
    heatmaps[0, 0, 7, 11] = 1.0
    heatmaps[0, 1, 20, 30] = 2.0
    heatmaps[0, 2, 63, 63] = 3.0

    decoded = decode_argmax_64_to_512(heatmaps)

    torch.testing.assert_close(
        decoded,
        torch.tensor([[[88.0, 56.0], [240.0, 160.0], [504.0, 504.0]]]),
    )


def test_summary_uses_official_aop_csv_values_as_ground_truth() -> None:
    points = np.asarray(
        [
            [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
            [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0]],
        ]
    )
    official_aop = np.asarray([80.0, 40.0])

    metrics, radial_errors, predicted_aop = summarize_validation_metrics(
        points, points, official_aop
    )

    np.testing.assert_allclose(predicted_aop, [90.0, 45.0])
    np.testing.assert_allclose(radial_errors, 0.0)
    assert metrics["MRE_PS1"] == pytest.approx(0.0)
    assert metrics["MRE_PS2"] == pytest.approx(0.0)
    assert metrics["MRE_FH1"] == pytest.approx(0.0)
    assert metrics["MRE_ALL"] == pytest.approx(0.0)
    assert metrics["AoP_absolute_error_deg"] == pytest.approx(7.5)
    assert metrics["aop_mae_valid_deg"] == pytest.approx(7.5)
    assert metrics["n_evaluable_aop"] == 2
    assert metrics["n_valid_aop"] == 2
    assert metrics["aop_invalid_prediction_count"] == 0
    assert metrics["aop_valid_ratio"] == pytest.approx(1.0)


def test_summary_penalizes_and_counts_degenerate_predicted_aop() -> None:
    target = np.asarray(
        [
            [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
            [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
        ]
    )
    prediction = target.copy()
    prediction[1] = 0.0

    metrics, _, predicted_aop = summarize_validation_metrics(
        prediction,
        target,
        np.asarray([90.0, 90.0]),
    )

    assert predicted_aop[0] == pytest.approx(90.0)
    assert np.isnan(predicted_aop[1])
    assert metrics["n_valid_aop"] == 1
    assert metrics["aop_invalid_prediction_count"] == 1
    assert metrics["aop_valid_ratio"] == pytest.approx(0.5)
    assert metrics["aop_invalid_prediction_ratio"] == pytest.approx(0.5)
    assert metrics["aop_mae_valid_deg"] == pytest.approx(0.0)
    assert metrics["AoP_absolute_error_deg"] == pytest.approx(90.0)
    assert metrics["aop_invalid_penalty_deg"] == pytest.approx(180.0)


def _write_fake_official_model(code_dir: Path, heatmaps: torch.Tensor) -> Path:
    code_dir.mkdir()
    (code_dir / "heatmap_net.py").write_text(
        """
import torch
import torch.nn as nn

class TinyOfficialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmaps = nn.Parameter(torch.zeros(1, 3, 64, 64))

    def forward(self, images):
        return self.heatmaps.expand(images.shape[0], -1, -1, -1)

def get_heatmap_model(num_keypoints=3, heatmap_size=64):
    assert num_keypoints == 3
    assert heatmap_size == 64
    return TinyOfficialModel()
""".lstrip(),
        encoding="utf-8",
    )
    checkpoint = code_dir.parent / "checkpoint.pth"
    torch.save({"model_state_dict": {"heatmaps": heatmaps}}, checkpoint)
    return checkpoint


def _write_validation_fixture(root: Path, target_points: np.ndarray) -> tuple[Path, Path, Path]:
    images = root / "images"
    images.mkdir()
    filenames = [f"validation_{index}.jpg" for index in range(3)]
    for index, filename in enumerate(filenames):
        Image.new("RGB", (512, 512), color=(index * 30, 0, 0)).save(images / filename)

    landmarks = root / "landmarks.csv"
    with landmarks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filename", "PS1", "PS2", "AOP Tangency"])
        writer.writeheader()
        for filename in filenames:
            writer.writerow(
                {
                    "Filename": filename,
                    "PS1": repr(tuple(target_points[0])),
                    "PS2": repr(tuple(target_points[1])),
                    "AOP Tangency": repr(tuple(target_points[2])),
                }
            )

    angle = float(aop_degrees(target_points[None])[0][0])
    aop = root / "aop.csv"
    with aop.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filename", "AOP"])
        writer.writeheader()
        for filename in filenames:
            writer.writerow({"Filename": filename, "AOP": angle})
    return images, landmarks, aop


def test_cli_writes_validation_metrics_predictions_and_three_cases(tmp_path: Path) -> None:
    heatmaps = torch.zeros((1, 3, 64, 64))
    bins = ((10, 10), (20, 10), (10, 20))
    for channel, (x, y) in enumerate(bins):
        heatmaps[0, channel, y, x] = 1.0
    target_points = np.asarray([(80.0, 80.0), (160.0, 80.0), (80.0, 160.0)])
    code_dir = tmp_path / "official_code"
    checkpoint = _write_fake_official_model(code_dir, heatmaps)
    images, landmarks, aop = _write_validation_fixture(tmp_path, target_points)
    output = tmp_path / "runs" / "validation"

    exit_code = main(
        [
            "--checkpoint",
            str(checkpoint),
            "--images",
            str(images),
            "--landmarks",
            str(landmarks),
            "--aop",
            str(aop),
            "--official-code-dir",
            str(code_dir),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--save-cases",
        ]
    )

    assert exit_code == 0
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"] == "validation"
    assert metrics["n_images"] == 3
    assert metrics["MRE_ALL"] == pytest.approx(0.0)
    assert metrics["AoP_absolute_error_deg"] == pytest.approx(0.0)
    assert (output / "predictions.csv").is_file()
    case_names = sorted(
        path.name.split("_", 1)[0] for path in (output / "visualizations").glob("*.png")
    )
    assert case_names == [
        "good",
        "median",
        "poor",
    ]
