"""IUGC labeled and unlabeled datasets with explicit schema mapping."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps
from geoequi_ld.geometry.coordinates import pixel_to_normalized, resize_points

IUGC2025_SOURCE_COLUMNS: dict[str, str] = {
    "PS1": "PS1",
    "PS2": "PS2",
    "FH1": "FH1",
}


def parse_point(value: object) -> tuple[float, float]:
    """Parse a CSV point encoded as ``(x, y)``, ``[x, y]``, or a pair."""

    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid point literal: {value!r}") from exc
    if not isinstance(parsed, tuple | list) or len(parsed) != 2:
        raise ValueError(f"Expected a two-value point, got {parsed!r}")
    x, y = float(parsed[0]), float(parsed[1])
    if not np.isfinite([x, y]).all():
        raise ValueError(f"Point contains NaN or Inf: {(x, y)}")
    return x, y


def _load_grayscale(path: Path) -> tuple[Tensor, tuple[int, int]]:
    try:
        with Image.open(path) as image:
            original_width, original_height = image.size
            grayscale = image.convert("L")
            array = np.asarray(grayscale, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not decode image: {path}") from exc
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor, (original_height, original_width)


def _resize_image(image: Tensor, size_hw: tuple[int, int]) -> Tensor:
    if tuple(image.shape[-2:]) == tuple(size_hw):
        return image
    return torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=size_hw,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)


class IUGCLabeledDataset(Dataset[dict[str, Any]]):
    """Load normalized IUGC CSV rows and generate Phase 0 heatmap targets.

    The source-to-semantic mapping is a required constructor argument in the
    public API. Phase 0 keeps the benchmark names ``PS1/PS2/FH1``. The data
    audit supports the descriptive aliases ``PS1=PS_R``, ``PS2=PS_L``, and
    ``FH1=FH_T``, but renaming model channels would add no useful information.
    """

    def __init__(
        self,
        *,
        image_dir: str | Path,
        labels_csv: str | Path,
        source_columns: Mapping[str, str],
        keypoint_order: Sequence[str] = ("PS1", "PS2", "FH1"),
        input_size_hw: tuple[int, int] = (512, 512),
        heatmap_size_hw: tuple[int, int] = (256, 256),
        sigma: float = 4.0,
        align_corners: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.labels_csv = Path(labels_csv)
        self.keypoint_order = tuple(keypoint_order)
        self.source_columns = dict(source_columns)
        self.input_size_hw = tuple(int(v) for v in input_size_hw)
        self.heatmap_size_hw = tuple(int(v) for v in heatmap_size_hw)
        self.sigma = float(sigma)
        self.align_corners = bool(align_corners)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.labels_csv.is_file():
            raise FileNotFoundError(f"Label CSV does not exist: {self.labels_csv}")
        if len(self.keypoint_order) != 3 or len(set(self.keypoint_order)) != 3:
            raise ValueError("Phase 0 expects exactly three unique keypoint names")
        missing_mapping = [name for name in self.keypoint_order if name not in self.source_columns]
        if missing_mapping:
            raise ValueError(f"Missing source-column mapping for: {missing_mapping}")

        self.rows = pd.read_csv(self.labels_csv)
        required_columns = {
            "Filename",
            *(self.source_columns[name] for name in self.keypoint_order),
        }
        missing_columns = sorted(required_columns - set(self.rows.columns))
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {missing_columns}")
        if self.rows.empty:
            raise ValueError("Label CSV contains no rows")
        duplicate_names = self.rows["Filename"].astype(str).duplicated(keep=False)
        if bool(duplicate_names.any()):
            names = self.rows.loc[duplicate_names, "Filename"].astype(str).tolist()
            raise ValueError(f"Duplicate filenames in label CSV: {names[:8]}")

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        filename = str(row["Filename"])
        image_path = self.image_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Labeled image is missing: {image_path}")

        original_image, original_size_hw = _load_grayscale(image_path)
        image = _resize_image(original_image, self.input_size_hw)
        points_original = torch.tensor(
            [parse_point(row[self.source_columns[name]]) for name in self.keypoint_order],
            dtype=torch.float32,
        )
        points_input = resize_points(
            points_original,
            original_size_hw,
            self.input_size_hw,
            align_corners=self.align_corners,
        )
        points_heatmap = resize_points(
            points_input,
            self.input_size_hw,
            self.heatmap_size_hw,
            align_corners=self.align_corners,
        )
        points_normalized = pixel_to_normalized(
            points_input,
            self.input_size_hw,
            align_corners=self.align_corners,
        )
        valid_mask = torch.ones(len(self.keypoint_order), dtype=torch.bool)
        heatmaps = generate_gaussian_heatmaps(
            points_heatmap,
            size_hw=self.heatmap_size_hw,
            sigma=self.sigma,
            valid_mask=valid_mask,
            out_of_bounds="error",
        )
        aop_value = (
            float(row["AoP"])
            if "AoP" in self.rows.columns and pd.notna(row["AoP"])
            else float("nan")
        )

        return {
            "filename": filename,
            "image": image,
            "heatmaps": heatmaps,
            "points_original_px": points_original,
            "points_input_px": points_input,
            "points_heatmap_px": points_heatmap,
            "points_normalized": points_normalized,
            "valid_mask": valid_mask,
            "aop_degrees": torch.tensor(aop_value, dtype=torch.float32),
            "original_size_hw": torch.tensor(original_size_hw, dtype=torch.int64),
        }


class IUGCUnlabeledDataset(Dataset[dict[str, Any]]):
    """Recursively enumerate unlabeled images without inventing labels."""

    SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self, *, image_dir: str | Path, input_size_hw: tuple[int, int] = (512, 512)
    ) -> None:
        self.image_dir = Path(image_dir)
        self.input_size_hw = tuple(int(v) for v in input_size_hw)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Unlabeled image directory does not exist: {self.image_dir}")
        self.files = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_SUFFIXES
        )
        if not self.files:
            raise ValueError(f"No supported images found below: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        original, original_size_hw = _load_grayscale(path)
        return {
            "filename": str(path.relative_to(self.image_dir).as_posix()),
            "image": _resize_image(original, self.input_size_hw),
            "original_size_hw": torch.tensor(original_size_hw, dtype=torch.int64),
        }
