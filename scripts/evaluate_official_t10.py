#!/usr/bin/env python
"""Evaluate one explicit official T10 checkpoint on one labeled official split."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_CODE_DIR = REPOSITORY_ROOT / "third_party" / "IUGC2025"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "runs" / "baseline_reproduction" / "validation"
LANDMARK_NAMES = ("PS1", "PS2", "FH1")
INPUT_SIZE = 512
HEATMAP_SIZE = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Official T10 checkpoint to evaluate; no checkpoint is selected implicitly",
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument(
        "--aop",
        type=Path,
        required=True,
        help="Official split CSV containing Filename and AOP",
    )
    parser.add_argument(
        "--split-name",
        choices=("validation", "testing"),
        default="validation",
    )
    parser.add_argument(
        "--fh1-column",
        default="AOP Tangency",
        help="Validation-landmark column mapped to the official model's FH1 channel",
    )
    parser.add_argument("--official-code-dir", type=Path, default=DEFAULT_OFFICIAL_CODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device")
    parser.add_argument(
        "--save-cases",
        action="store_true",
        help="Save good/median/poor overlays below output-dir/visualizations",
    )
    return parser


def _parse_point(value: object) -> tuple[float, float]:
    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid landmark coordinate: {value!r}") from exc
    if not isinstance(parsed, tuple | list) or len(parsed) != 2:
        raise ValueError(f"Expected a two-value landmark coordinate, got {parsed!r}")
    point = (float(parsed[0]), float(parsed[1]))
    if not np.isfinite(point).all():
        raise ValueError(f"Landmark coordinate must be finite, got {point!r}")
    return point


def _read_aop_by_filename(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"AoP CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Filename", "AOP"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Validation AoP CSV is missing columns: {sorted(missing)}")
        values: dict[str, float] = {}
        for row in reader:
            filename = str(row["Filename"])
            if filename in values:
                raise ValueError(f"Duplicate Filename in validation AoP CSV: {filename}")
            angle = float(row["AOP"])
            if not np.isfinite(angle):
                raise ValueError(f"Non-finite validation AoP for {filename}: {angle}")
            values[filename] = angle
    if not values:
        raise ValueError(f"AoP CSV contains no rows: {path}")
    return values


class OfficialValidationDataset(Dataset[dict[str, Any]]):
    """Load the official validation images and landmark/AoP CSV files."""

    def __init__(
        self,
        *,
        images: Path,
        landmarks_csv: Path,
        aop_csv: Path,
        fh1_column: str,
    ) -> None:
        if not images.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {images}")
        if not landmarks_csv.is_file():
            raise FileNotFoundError(f"Landmark CSV does not exist: {landmarks_csv}")
        self.images = images
        self.transform = transforms.Compose(
            [transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()]
        )
        aop_by_filename = _read_aop_by_filename(aop_csv)
        source_columns = ("PS1", "PS2", fh1_column)
        self.rows: list[tuple[str, np.ndarray, float]] = []

        with landmarks_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"Filename", *source_columns}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Landmark CSV is missing columns: {sorted(missing)}"
                )
            seen: set[str] = set()
            for row in reader:
                filename = str(row["Filename"])
                if filename in seen:
                    raise ValueError(f"Duplicate Filename in landmarks: {filename}")
                if filename not in aop_by_filename:
                    raise ValueError(f"AoP is missing for landmark row: {filename}")
                seen.add(filename)
                points = np.asarray(
                    [_parse_point(row[column]) for column in source_columns], dtype=np.float32
                )
                self.rows.append((filename, points, aop_by_filename[filename]))

        if not self.rows:
            raise ValueError(f"Landmark CSV contains no rows: {landmarks_csv}")
        unused_aop = sorted(set(aop_by_filename).difference(row[0] for row in self.rows))
        if unused_aop:
            raise ValueError(
                "Validation landmark/AoP filename sets differ; "
                f"first AoP-only filename: {unused_aop[0]}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, points, aop = self.rows[index]
        image_path = self.images / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Validation image is missing: {image_path}")
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))
        return {
            "filename": filename,
            "image_path": str(image_path),
            "image": image,
            "points": torch.from_numpy(points.copy()),
            "aop": torch.tensor(aop, dtype=torch.float64),
        }


def _load_python_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("iugc2025_official_heatmap_net", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_model(official_code_dir: Path, checkpoint_path: Path) -> nn.Module:
    """Instantiate the upstream U-Net and load an explicitly supplied checkpoint."""

    module_path = official_code_dir / "heatmap_net.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Official heatmap_net.py does not exist: {module_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    module = _load_python_module(module_path)
    factory = getattr(module, "get_heatmap_model", None)
    if not callable(factory):
        raise AttributeError(
            f"Official model factory get_heatmap_model is missing from {module_path}"
        )
    model = factory(num_keypoints=3, heatmap_size=HEATMAP_SIZE)
    if not isinstance(model, nn.Module):
        raise TypeError("Official get_heatmap_model did not return torch.nn.Module")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must be a state dict or contain model_state_dict")
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint model_state_dict must be a mapping")
    model.load_state_dict(state_dict, strict=True)
    return model


def decode_argmax_64_to_512(heatmaps: Tensor) -> Tensor:
    """Apply the upstream hard argmax decoder and map each heatmap bin by x/y * 8."""

    expected = (3, HEATMAP_SIZE, HEATMAP_SIZE)
    if heatmaps.ndim != 4 or tuple(heatmaps.shape[1:]) != expected:
        raise ValueError(
            f"Official T10 output must be [N, 3, 64, 64], got {tuple(heatmaps.shape)}"
        )
    flat_indices = torch.argmax(heatmaps.reshape(heatmaps.shape[0], 3, -1), dim=2)
    y_indices = torch.div(flat_indices, HEATMAP_SIZE, rounding_mode="floor").float()
    x_indices = (flat_indices % HEATMAP_SIZE).float()
    scale = INPUT_SIZE / HEATMAP_SIZE
    return torch.stack((x_indices * scale, y_indices * scale), dim=-1)


def aop_degrees(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unsigned PS1-vertex AoP and a mask for non-degenerate triples."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (3, 2):
        raise ValueError(f"Expected points shaped [N, 3, 2], got {array.shape}")
    pubic_ray = array[:, 1] - array[:, 0]
    fetal_ray = array[:, 2] - array[:, 0]
    denominator = np.linalg.norm(pubic_ray, axis=1) * np.linalg.norm(fetal_ray, axis=1)
    valid = np.isfinite(array).all(axis=(1, 2)) & (denominator > 0.0)
    dot = np.sum(pubic_ray * fetal_ray, axis=1)
    cosine = np.divide(
        dot,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=valid,
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angles[~valid] = np.nan
    return angles, valid


def summarize_validation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_aop: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    official_aop = np.asarray(target_aop, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[1:] != (3, 2):
        raise ValueError(
            f"Prediction and target must share [N, 3, 2] shape, got {pred.shape} and {truth.shape}"
        )
    if official_aop.shape != (pred.shape[0],) or not np.isfinite(official_aop).all():
        raise ValueError("Official validation AoP must contain one finite value per image")

    radial_errors = np.linalg.norm(pred - truth, axis=2)
    predicted_aop, valid_aop = aop_degrees(pred)
    if not bool(valid_aop.all()):
        invalid_count = int((~valid_aop).sum())
        raise ValueError(
            f"AoP is undefined for {invalid_count} prediction(s); refusing a partial mean"
        )
    aop_absolute_errors = np.abs(predicted_aop - official_aop)
    point_means = radial_errors.mean(axis=0)
    metrics: dict[str, float | int] = {
        "n_images": int(pred.shape[0]),
        "MRE_PS1": float(point_means[0]),
        "MRE_PS2": float(point_means[1]),
        "MRE_FH1": float(point_means[2]),
        "MRE_ALL": float(radial_errors.mean()),
        "AoP_absolute_error_deg": float(aop_absolute_errors.mean()),
    }
    return metrics, radial_errors, predicted_aop


def _select_representative_indices(case_mre: np.ndarray) -> dict[str, int]:
    if case_mre.ndim != 1 or case_mre.size < 3:
        raise ValueError("At least three validation images are required for case visualization")
    order = np.argsort(case_mre, kind="stable")
    return {
        "good": int(order[0]),
        "median": int(order[len(order) // 2]),
        "poor": int(order[-1]),
    }


def _save_overlay(
    *,
    image_path: Path,
    filename: str,
    prediction: np.ndarray,
    target: np.ndarray,
    predicted_aop: float,
    target_aop: float,
    case_name: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    figure, axis = plt.subplots(figsize=(7, 7), dpi=150)
    axis.imshow(image)
    colors = ("#ff3b30", "#34c759", "#007aff")
    for index, (name, color) in enumerate(zip(LANDMARK_NAMES, colors, strict=True)):
        axis.scatter(
            prediction[index, 0],
            prediction[index, 1],
            c=color,
            marker="x",
            s=70,
            linewidths=2,
            label=f"Pred {name}",
        )
        axis.scatter(
            target[index, 0],
            target[index, 1],
            facecolors="none",
            edgecolors=color,
            marker="o",
            s=70,
            linewidths=1.6,
            label=f"GT {name}",
        )
    axis.plot(prediction[[0, 1], 0], prediction[[0, 1], 1], color="#ffd60a")
    axis.plot(prediction[[0, 2], 0], prediction[[0, 2], 1], color="#ffd60a")
    axis.set_title(
        f"{case_name} | {filename}\n"
        f"AoP prediction {predicted_aop:.2f}° | ground truth {target_aop:.2f}°"
    )
    axis.axis("off")
    axis.legend(loc="lower center", ncol=3, fontsize=7, framealpha=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _write_predictions(
    path: Path,
    filenames: Sequence[str],
    predictions: np.ndarray,
    targets: np.ndarray,
    radial_errors: np.ndarray,
    predicted_aop: np.ndarray,
    target_aop: np.ndarray,
) -> None:
    fieldnames = ["Filename"]
    for name in LANDMARK_NAMES:
        fieldnames.extend(
            [f"{name}_x_pred", f"{name}_y_pred", f"{name}_x_gt", f"{name}_y_gt"]
        )
    fieldnames.extend(
        [
            "MRE_image",
            "AoP_prediction_deg",
            "AoP_ground_truth_deg",
            "AoP_absolute_error_deg",
        ]
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, filename in enumerate(filenames):
            row: dict[str, str | float] = {"Filename": filename}
            for point_index, name in enumerate(LANDMARK_NAMES):
                row[f"{name}_x_pred"] = float(predictions[index, point_index, 0])
                row[f"{name}_y_pred"] = float(predictions[index, point_index, 1])
                row[f"{name}_x_gt"] = float(targets[index, point_index, 0])
                row[f"{name}_y_gt"] = float(targets[index, point_index, 1])
            row["MRE_image"] = float(radial_errors[index].mean())
            row["AoP_prediction_deg"] = float(predicted_aop[index])
            row["AoP_ground_truth_deg"] = float(target_aop[index])
            row["AoP_absolute_error_deg"] = float(
                abs(predicted_aop[index] - target_aop[index])
            )
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")

    device = _resolve_device(args.device)
    dataset = OfficialValidationDataset(
        images=args.images,
        landmarks_csv=args.landmarks,
        aop_csv=args.aop,
        fh1_column=args.fh1_column,
    )
    loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_official_model(args.official_code_dir, args.checkpoint).to(device)
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_angles: list[np.ndarray] = []
    filenames: list[str] = []
    image_paths: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            heatmaps = model(batch["image"].to(device, non_blocking=True))
            decoded = decode_argmax_64_to_512(heatmaps).cpu().numpy()
            predictions.append(decoded)
            targets.append(batch["points"].numpy())
            target_angles.append(batch["aop"].numpy())
            filenames.extend(batch["filename"])
            image_paths.extend(batch["image_path"])

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    target_aop_array = np.concatenate(target_angles, axis=0).astype(np.float64)
    metrics, errors, predicted_aop = summarize_validation_metrics(
        prediction_array, target_array, target_aop_array
    )
    result: dict[str, object] = {
        "split": args.split_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "official_code_dir": str(args.official_code_dir.resolve()),
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "heatmap_size": [HEATMAP_SIZE, HEATMAP_SIZE],
        "decoder": "per-channel hard argmax; heatmap x/y multiplied by 8",
        **metrics,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    _write_predictions(
        output_dir / "predictions.csv",
        filenames,
        prediction_array,
        target_array,
        errors,
        predicted_aop,
        target_aop_array,
    )

    if args.save_cases:
        case_mre = errors.mean(axis=1)
        for case_name, index in _select_representative_indices(case_mre).items():
            stem = Path(filenames[index]).stem
            _save_overlay(
                image_path=Path(image_paths[index]),
                filename=filenames[index],
                prediction=prediction_array[index],
                target=target_array[index],
                predicted_aop=float(predicted_aop[index]),
                target_aop=float(target_aop_array[index]),
                case_name=case_name,
                output_path=output_dir / "visualizations" / f"{case_name}_{stem}.png",
            )

    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
