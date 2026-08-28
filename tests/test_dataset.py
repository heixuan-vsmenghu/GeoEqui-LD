from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoequi_ld.data.dataset import IUGCLabeledDataset, IUGCUnlabeledDataset


def _write_disguised_png(path: Path, *, size: tuple[int, int] = (32, 32)) -> None:
    pixels = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0])
    Image.fromarray(pixels, mode="L").save(path, format="PNG")


def test_labeled_dataset_decodes_by_content_and_maps_validation_schema(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_disguised_png(image_dir / "sample.jpg")
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Filename", "PS1", "PS2", "AOP Tangency", "AoP"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "Filename": "sample.jpg",
                "PS1": "(24, 4)",
                "PS2": "(4, 8)",
                "AOP Tangency": "(20, 24)",
                "AoP": "87.5",
            }
        )

    dataset = IUGCLabeledDataset(
        image_dir=image_dir,
        labels_csv=csv_path,
        source_columns={"PS1": "PS1", "PS2": "PS2", "FH1": "AOP Tangency"},
        input_size_hw=(32, 32),
        heatmap_size_hw=(16, 16),
        sigma=2.0,
    )
    sample = dataset[0]
    assert sample["image"].shape == (1, 32, 32)
    assert sample["heatmaps"].shape == (3, 16, 16)
    assert sample["points_original_px"].tolist() == [[24.0, 4.0], [4.0, 8.0], [20.0, 24.0]]
    assert float(sample["aop_degrees"]) == pytest.approx(87.5)


def test_labeled_dataset_rejects_missing_image(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        'Filename,PS1,PS2,FH1\nmissing.jpg,"(1, 1)","(2, 2)","(3, 3)"\n',
        encoding="utf-8",
    )
    dataset = IUGCLabeledDataset(
        image_dir=image_dir,
        labels_csv=csv_path,
        source_columns={"PS1": "PS1", "PS2": "PS2", "FH1": "FH1"},
        input_size_hw=(8, 8),
        heatmap_size_hw=(4, 4),
        sigma=1.0,
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        _ = dataset[0]


def test_unlabeled_dataset_enumerates_nested_images(tmp_path: Path) -> None:
    nested = tmp_path / "case" / "frames"
    nested.mkdir(parents=True)
    _write_disguised_png(nested / "frame.jpg", size=(16, 16))
    dataset = IUGCUnlabeledDataset(image_dir=tmp_path, input_size_hw=(32, 32))
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["filename"] == "case/frames/frame.jpg"
    assert sample["image"].shape == (1, 32, 32)
