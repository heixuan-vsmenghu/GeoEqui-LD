from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from scripts.generate_baseline_one_page_pdf import main, parse_training_log


def test_parse_official_training_epoch_lines(tmp_path: Path) -> None:
    log = tmp_path / "training.log"
    log.write_text(
        """Using device: cuda
Epoch 1/150 - Training Loss: 0.0300, Coordinate Distance: 0.2500, Learning Rate: 0.000100
progress output that should be ignored
Epoch 2/150 - Training Loss: 2.0e-2, Coordinate Distance: 0.2, Learning Rate: 0.000100
Epoch 3/150 - Training Loss: 0.0100, Coordinate Distance: 1.5e-1, Learning Rate: 0.000100
""",
        encoding="utf-8",
    )

    history = parse_training_log(log)

    np.testing.assert_array_equal(history.epochs, [1, 2, 3])
    np.testing.assert_allclose(history.train_loss, [0.03, 0.02, 0.01])
    np.testing.assert_allclose(history.coordinate_distance, [0.25, 0.20, 0.15])
    assert history.declared_epochs == 150


def _write_metrics(path: Path, split: str, offset: float) -> None:
    path.write_text(
        json.dumps(
            {
                "split": split,
                "n_images": 100 if split == "validation" else 501,
                "MRE_PS1": 10.0 + offset,
                "MRE_PS2": 20.0 + offset,
                "MRE_FH1": 40.0 + offset,
                "MRE_ALL": 70.0 / 3.0 + offset,
                "AoP_absolute_error_deg": 9.0 + offset,
            }
        ),
        encoding="utf-8",
    )


def _write_case_image(path: Path, color: str, label: str) -> None:
    image = Image.new("RGB", (360, 280), color=color)
    draw = ImageDraw.Draw(image)
    draw.line((60, 220, 170, 80), fill="yellow", width=4)
    draw.line((60, 220, 290, 180), fill="yellow", width=4)
    draw.text((12, 12), label, fill="white")
    image.save(path)


def test_cli_generates_exactly_one_pdf_page_from_synthetic_inputs(tmp_path: Path) -> None:
    log = tmp_path / "training.log"
    log.write_text(
        "\n".join(
            f"Epoch {epoch}/150 - Training Loss: {0.1 / epoch:.6f}, "
            f"Coordinate Distance: {0.5 / epoch:.6f}, Learning Rate: 0.000100"
            for epoch in range(1, 6)
        ),
        encoding="utf-8",
    )
    validation = tmp_path / "validation_metrics.json"
    testing = tmp_path / "testing_metrics.json"
    _write_metrics(validation, "validation", 0.0)
    _write_metrics(testing, "testing", 1.0)
    images = tuple(tmp_path / f"{name}.png" for name in ("good", "median", "poor"))
    for path, color, label in zip(
        images,
        ("#16324f", "#4f3b16", "#4f1616"),
        ("good", "median", "poor"),
        strict=True,
    ):
        _write_case_image(path, color, label)
    output = tmp_path / "runs" / "baseline_reproduction" / "advisor.pdf"

    exit_code = main(
        [
            "--training-log",
            str(log),
            "--validation-metrics",
            str(validation),
            "--testing-metrics",
            str(testing),
            "--good-image",
            str(images[0]),
            "--median-image",
            str(images[1]),
            "--poor-image",
            str(images[2]),
            "--output",
            str(output),
            "--issue",
            "Synthetic issue used only by this test.",
        ]
    )

    assert exit_code == 0
    content = output.read_bytes()
    assert content.startswith(b"%PDF")
    assert len(content) > 10_000
    assert len(re.findall(rb"/Type\s*/Page\b", content)) == 1


def test_metrics_split_mismatch_is_rejected(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics, "testing", 0.0)

    from scripts.generate_baseline_one_page_pdf import load_metrics

    with pytest.raises(ValueError, match="Expected 'validation' metrics"):
        load_metrics(metrics, "validation")
