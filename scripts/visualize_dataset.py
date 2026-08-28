#!/usr/bin/env python3
"""Create local, restricted landmark overlays for Phase 0 visual inspection.

Every invocation requires an explicit acknowledgement because the generated
images are derived from restricted medical data and must not be committed or
uploaded without permission from the data owner and supervisor.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geoequi_ld.utils.annotations import (  # noqa: E402
    SOURCE_KEYPOINTS,
    load_annotations,
    parse_point,
)
from geoequi_ld.utils.config import ConfigError, load_config, resolve_splits  # noqa: E402
from geoequi_ld.utils.hashing import stable_identifier  # noqa: E402
from geoequi_ld.utils.io import ensure_output_directory, write_text  # noqa: E402

COLORS = {"PS1": "#00E5FF", "PS2": "#FFB000", "FH1": "#FF4DA6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "phase0_baseline.yaml"),
        help="Phase 0 YAML configuration",
    )
    parser.add_argument("--split", help="Labeled split to visualize")
    parser.add_argument("--samples", type=int, help="Number of random images")
    parser.add_argument("--seed", type=int, help="Sampling seed; defaults to project.seed")
    parser.add_argument(
        "--output-dir",
        help="Override the gitignored local visualization directory",
    )
    parser.add_argument(
        "--include-filenames",
        action="store_true",
        help="Include raw source filenames in titles and the local manifest",
    )
    parser.add_argument(
        "--acknowledge-restricted-output",
        action="store_true",
        help=(
            "Required acknowledgement that generated medical-data derivatives will not be committed"
        ),
    )
    return parser.parse_args()


def _resolve_output(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("visualization.output_dir must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _find_image(image_dir: Path, filename: str) -> Path:
    normalized = Path(filename.replace("\\", "/"))
    direct = image_dir / normalized
    if direct.is_file():
        return direct
    matches = [path for path in image_dir.rglob(normalized.name) if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No image matches annotation file ID {stable_identifier(filename)}"
        )
    raise ValueError(f"Ambiguous image basename for file ID {stable_identifier(filename)}")


def _aop_degrees(points: dict[str, tuple[float, float]], definition: dict[str, Any]) -> float:
    vertex = np.asarray(points[str(definition["vertex_key"])], dtype=np.float64)
    axis = np.asarray(points[str(definition["pubic_axis_other_key"])], dtype=np.float64) - vertex
    head = np.asarray(points[str(definition["fetal_head_key"])], dtype=np.float64) - vertex
    denominator = float(np.linalg.norm(axis) * np.linalg.norm(head))
    if denominator <= 1e-12:
        raise ValueError("AoP contains a zero-length defining vector")
    cosine = float(np.dot(axis, head) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _draw_overlay(
    image_path: Path,
    points: dict[str, tuple[float, float]],
    output_path: Path,
    display_name: str,
    aop_definition: dict[str, Any] | None,
) -> tuple[int, int, float | None]:
    try:
        with Image.open(image_path) as source:
            source.load()
            width, height = source.size
            image = source.convert("L")
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not decode selected image: {type(exc).__name__}") from exc

    figure, axis = plt.subplots(figsize=(7, 7), dpi=150)
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    for name in SOURCE_KEYPOINTS:
        x, y = points[name]
        axis.scatter(
            [x],
            [y],
            s=72,
            c=COLORS[name],
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        axis.text(
            x + 7,
            y - 7,
            name,
            color=COLORS[name],
            fontsize=10,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.5},
        )

    angle: float | None = None
    if aop_definition is not None:
        vertex_key = str(aop_definition["vertex_key"])
        axis_key = str(aop_definition["pubic_axis_other_key"])
        head_key = str(aop_definition["fetal_head_key"])
        vx, vy = points[vertex_key]
        ax, ay = points[axis_key]
        hx, hy = points[head_key]
        axis.plot([vx, ax], [vy, ay], color="#FFB000", linewidth=2.0)
        axis.plot([vx, hx], [vy, hy], color="#00E676", linewidth=2.0)
        angle = _aop_degrees(points, aop_definition)

    angle_text = "AoP definition unresolved" if angle is None else f"AoP={angle:.2f}°"
    axis.set_title(f"{display_name} | {width}×{height} | {angle_text}", fontsize=11)
    axis.text(
        0.01,
        0.01,
        "RESTRICTED LOCAL OUTPUT — DO NOT COMMIT",
        transform=axis.transAxes,
        color="white",
        fontsize=9,
        weight="bold",
        bbox={"facecolor": "#A00000", "alpha": 0.82, "edgecolor": "none", "pad": 3},
    )
    axis.set_xlim(0, width - 1)
    axis.set_ylim(height - 1, 0)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", facecolor="black")
    plt.close(figure)
    return width, height, angle


def run(args: argparse.Namespace) -> int:
    if not args.acknowledge_restricted_output:
        raise ConfigError(
            "Refusing to generate data-derived images without --acknowledge-restricted-output"
        )
    config = load_config(Path(args.config).resolve())
    splits = {split.name: split for split in resolve_splits(config, PROJECT_ROOT)}
    visual_config = config.get("visualization", {})
    if not isinstance(visual_config, dict):
        raise ConfigError("visualization must be a mapping")
    split_name = str(args.split or visual_config.get("split", "train"))
    if split_name not in splits:
        raise ConfigError(f"Unknown or disabled split: {split_name}")
    split = splits[split_name]
    if split.csv is None:
        raise ConfigError(f"Split {split_name!r} has no annotation CSV")
    if not split.image_dir.is_dir() or not split.csv.is_file():
        raise FileNotFoundError(f"Configured inputs for split {split_name!r} are unavailable")
    if split.aop_csv is not None and not split.aop_csv.is_file():
        raise FileNotFoundError(f"Configured paired AoP CSV for {split_name!r} is unavailable")

    table = load_annotations(split.csv, split.aop_csv)
    sample_count = int(args.samples or visual_config.get("samples", 20))
    if sample_count <= 0:
        raise ConfigError("--samples must be positive")
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    output_value = args.output_dir or visual_config.get(
        "output_dir", "artifacts/phase0/visualizations"
    )
    output_dir = ensure_output_directory(_resolve_output(output_value), [split.image_dir])
    include_filenames = bool(
        args.include_filenames or visual_config.get("include_filenames", False)
    )

    aop_definition: dict[str, Any] | None = None
    task = config.get("task", {})
    if isinstance(task, dict) and isinstance(task.get("aop"), dict):
        candidate = task["aop"]
        fields = {"vertex_key", "pubic_axis_other_key", "fetal_head_key"}
        if bool(candidate.get("source_definition_confirmed", False)) and fields <= candidate.keys():
            aop_definition = candidate

    indices = list(range(len(table.rows)))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(sample_count, len(indices))]
    manifest_rows: list[dict[str, object]] = []
    for output_index, row_index in enumerate(indices):
        row = table.rows.iloc[row_index]
        filename = str(row["Filename"])
        file_id = stable_identifier(f"{split_name}:{filename}")
        image_path = _find_image(split.image_dir, filename)
        points = {name: parse_point(row[name]) for name in SOURCE_KEYPOINTS}
        output_name = f"sample_{output_index:03d}_{file_id}.png"
        display_name = filename if include_filenames else f"file_id={file_id}"
        width, height, angle = _draw_overlay(
            image_path,
            points,
            output_dir / output_name,
            display_name,
            aop_definition,
        )
        manifest_rows.append(
            {
                "sample_index": output_index,
                "file_id": file_id,
                "filename": filename if include_filenames else "",
                "width": width,
                "height": height,
                "aop_degrees": "" if angle is None else f"{angle:.8f}",
                "output_file": output_name,
            }
        )

    manifest_path = output_dir / "visualization_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_text(
        output_dir / "DO_NOT_COMMIT.txt",
        """RESTRICTED LOCAL OUTPUT — DO NOT COMMIT OR UPLOAD

These overlays are derived from medical images and annotations. Keep this
directory local unless the dataset owner and supervisor have explicitly
confirmed that publication and redistribution are permitted.
""",
    )
    print("WARNING: generated overlays are restricted local data derivatives.")
    print(f"Created {len(manifest_rows)} overlays in {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"visualization error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
