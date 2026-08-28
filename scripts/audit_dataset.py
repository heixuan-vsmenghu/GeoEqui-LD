#!/usr/bin/env python3
"""Read-only IUGC dataset audit with privacy-preserving generated reports.

The script never changes source images or annotation files. By default, report
rows contain deterministic IDs rather than local absolute paths or filenames.
Generated reports belong in the gitignored ``reports/phase0/generated`` tree.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geoequi_ld.utils.annotations import (  # noqa: E402
    SOURCE_KEYPOINTS,
    AnnotationTable,
    load_annotations,
    parse_point,
)
from geoequi_ld.utils.config import (  # noqa: E402
    ConfigError,
    SplitPaths,
    load_config,
    resolve_splits,
)
from geoequi_ld.utils.hashing import (  # noqa: E402
    sequence_signature,
    sha256_file,
    sha256_image_pixels,
    stable_identifier,
)
from geoequi_ld.utils.io import (  # noqa: E402
    ensure_output_directory,
    write_csv_rows,
    write_json,
    write_text,
)

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    split: str
    source_relative_path: str
    relative_path: str
    file_id: str
    extension: str
    size_bytes: int
    width: int | None
    height: int | None
    mode: str | None
    decodable: bool
    file_sha256: str | None
    pixel_sha256: str | None
    annotation_present: bool
    error_type: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "phase0_baseline.yaml"),
        help="Phase 0 YAML configuration (default: configs/phase0_baseline.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the generated report directory; relative paths use the project root",
    )
    parser.add_argument(
        "--hash-mode",
        choices=("none", "file", "pixels", "both"),
        help="Override audit.hash_mode (default configuration uses both)",
    )
    parser.add_argument(
        "--include-relative-paths",
        action="store_true",
        help="Include source-relative paths in local reports; never includes absolute paths",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        help="Development-only bounded scan; resulting statistics are explicitly marked partial",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status for any integrity issue, not only missing required inputs",
    )
    return parser.parse_args()


def _resolve_generated_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("audit.output_dir must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _privacy_path(relative_path: str, include_paths: bool) -> str:
    if not include_paths:
        return ""
    # Prevent spreadsheet software from interpreting a path as a formula.
    return "'" + relative_path if relative_path.startswith(("=", "+", "-", "@")) else relative_path


def _annotation_index(table: AnnotationTable | None) -> tuple[set[str], dict[str, str]]:
    if table is None:
        return set(), {}
    relative_names = {str(value).replace("\\", "/") for value in table.rows["Filename"]}
    by_basename: dict[str, list[str]] = defaultdict(list)
    for name in relative_names:
        by_basename[Path(name).name].append(name)
    unique_basename = {
        basename: names[0] for basename, names in by_basename.items() if len(names) == 1
    }
    return relative_names, unique_basename


def _match_annotation(
    relative_path: str, exact_names: set[str], unique_basenames: dict[str, str]
) -> str | None:
    if relative_path in exact_names:
        return relative_path
    return unique_basenames.get(Path(relative_path).name)


def _aop_degrees(points: dict[str, tuple[float, float]], definition: dict[str, Any]) -> float:
    vertex_key = str(definition["vertex_key"])
    axis_key = str(definition["pubic_axis_other_key"])
    head_key = str(definition["fetal_head_key"])
    vertex = np.asarray(points[vertex_key], dtype=np.float64)
    axis = np.asarray(points[axis_key], dtype=np.float64) - vertex
    head = np.asarray(points[head_key], dtype=np.float64) - vertex
    denominator = float(np.linalg.norm(axis) * np.linalg.norm(head))
    if denominator <= 1e-12:
        raise ValueError("AoP contains a zero-length defining vector")
    cosine = float(np.dot(axis, head) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _load_split_annotations(
    split: SplitPaths,
) -> tuple[AnnotationTable | None, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if split.csv is None:
        return None, issues
    if not split.csv.is_file():
        issues.append(
            {
                "severity": "error" if split.required else "warning",
                "split": split.name,
                "code": "missing_annotation_csv",
                "detail": "Configured landmark CSV was not found",
            }
        )
        return None, issues
    if split.aop_csv is not None and not split.aop_csv.is_file():
        issues.append(
            {
                "severity": "error" if split.required else "warning",
                "split": split.name,
                "code": "missing_aop_csv",
                "detail": "Configured paired AoP CSV was not found",
            }
        )
        return None, issues
    try:
        return load_annotations(split.csv, split.aop_csv), issues
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        issues.append(
            {
                "severity": "error",
                "split": split.name,
                "code": "invalid_annotation_schema",
                "detail": type(exc).__name__,
            }
        )
        return None, issues


def _enumerate_images(image_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(image_dir).as_posix().lower(),
    )
    return images if limit is None else images[:limit]


def _scan_images(
    split: SplitPaths,
    table: AnnotationTable | None,
    hash_mode: str,
    include_paths: bool,
    limit: int | None,
) -> tuple[list[ImageRecord], dict[str, Any], dict[str, tuple[int, int]]]:
    exact_names, unique_basenames = _annotation_index(table)
    image_paths = _enumerate_images(split.image_dir, limit)
    records: list[ImageRecord] = []
    matched_sizes: dict[str, tuple[int, int]] = {}
    size_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    matched_annotations: set[str] = set()

    for path in image_paths:
        relative_path = path.relative_to(split.image_dir).as_posix()
        annotation_name = _match_annotation(relative_path, exact_names, unique_basenames)
        if annotation_name is not None:
            matched_annotations.add(annotation_name)
        width: int | None = None
        height: int | None = None
        mode: str | None = None
        file_digest: str | None = None
        pixel_digest: str | None = None
        error_type: str | None = None
        decodable = False
        try:
            if hash_mode in ("file", "both"):
                file_digest = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                decodable = True
                if hash_mode in ("pixels", "both"):
                    pixel_digest = sha256_image_pixels(image)
            size_counts[f"{width}x{height}"] += 1
            mode_counts[str(mode)] += 1
            if annotation_name is not None and width is not None and height is not None:
                matched_sizes[annotation_name] = (width, height)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            error_type = type(exc).__name__

        records.append(
            ImageRecord(
                split=split.name,
                source_relative_path=relative_path,
                relative_path=_privacy_path(relative_path, include_paths),
                file_id=stable_identifier(f"{split.name}:{relative_path}"),
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                width=width,
                height=height,
                mode=mode,
                decodable=decodable,
                file_sha256=file_digest,
                pixel_sha256=pixel_digest,
                annotation_present=annotation_name is not None,
                error_type=error_type,
            )
        )

    annotation_rows = 0 if table is None else int(len(table.rows))
    summary = {
        "images_scanned": len(records),
        "scan_is_partial": limit is not None,
        "annotation_rows": annotation_rows,
        "matched_annotation_rows": len(matched_annotations),
        "annotation_rows_without_image": max(annotation_rows - len(matched_annotations), 0),
        "images_without_annotation": sum(
            1 for record in records if table is not None and not record.annotation_present
        ),
        "decodable_images": sum(record.decodable for record in records),
        "corrupt_images": sum(not record.decodable for record in records),
        "image_size_distribution": dict(sorted(size_counts.items())),
        "image_mode_distribution": dict(sorted(mode_counts.items())),
        "bytes_scanned": sum(record.size_bytes for record in records),
    }
    return records, summary, matched_sizes


def _audit_labels(
    table: AnnotationTable | None,
    matched_sizes: dict[str, tuple[int, int]],
    aop_definition: dict[str, Any] | None,
) -> dict[str, Any]:
    if table is None:
        return {
            "format": None,
            "source_columns": {},
            "grouping_columns": [],
            "duplicate_filenames": 0,
            "missing_values": 0,
            "invalid_point_rows": 0,
            "out_of_bounds_points": 0,
            "coordinate_range_px": None,
            "aop_rows": 0,
            "aop_recomputed_rows": 0,
            "aop_max_abs_difference_deg": None,
        }

    rows = table.rows
    duplicate_filenames = int(rows["Filename"].astype(str).duplicated(keep=False).sum())
    missing_values = int(rows.isna().sum().sum())
    invalid_point_rows = 0
    out_of_bounds_points = 0
    valid_points: list[tuple[float, float]] = []
    aop_differences: list[float] = []

    for _, row in rows.iterrows():
        try:
            points = {name: parse_point(row[name]) for name in SOURCE_KEYPOINTS}
        except (TypeError, ValueError):
            invalid_point_rows += 1
            continue
        valid_points.extend(points.values())
        filename = str(row["Filename"]).replace("\\", "/")
        image_size = matched_sizes.get(filename)
        if image_size is None:
            basename_matches = [
                size
                for name, size in matched_sizes.items()
                if Path(name).name == Path(filename).name
            ]
            image_size = basename_matches[0] if len(basename_matches) == 1 else None
        if image_size is not None:
            width, height = image_size
            out_of_bounds_points += sum(
                not (0.0 <= x < float(width) and 0.0 <= y < float(height))
                for x, y in points.values()
            )
        if aop_definition is not None and "AoP" in rows.columns and pd.notna(row.get("AoP")):
            try:
                computed = _aop_degrees(points, aop_definition)
                aop_differences.append(abs(computed - float(row["AoP"])))
            except (KeyError, TypeError, ValueError):
                pass

    point_array = np.asarray(valid_points, dtype=np.float64)
    coordinate_range = None
    if point_array.size:
        coordinate_range = {
            "x_min": float(np.min(point_array[:, 0])),
            "x_max": float(np.max(point_array[:, 0])),
            "y_min": float(np.min(point_array[:, 1])),
            "y_max": float(np.max(point_array[:, 1])),
        }
    aop_rows = int(rows["AoP"].notna().sum()) if "AoP" in rows else 0
    return {
        "format": table.format_name,
        "source_columns": table.source_columns,
        "grouping_columns": list(table.grouping_columns),
        "duplicate_filenames": duplicate_filenames,
        "missing_values": missing_values,
        "invalid_point_rows": invalid_point_rows,
        "out_of_bounds_points": out_of_bounds_points,
        "coordinate_range_px": coordinate_range,
        "aop_rows": aop_rows,
        "aop_recomputed_rows": len(aop_differences),
        "aop_max_abs_difference_deg": (max(aop_differences) if aop_differences else None),
    }


def _duplicate_rows(records: list[ImageRecord], include_paths: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for kind, attribute in (("file_sha256", "file_sha256"), ("pixel_sha256", "pixel_sha256")):
        groups: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            digest = getattr(record, attribute)
            if digest:
                groups[digest].append(record)
        for digest, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            splits = sorted({member.split for member in members})
            output.append(
                {
                    "group_type": kind,
                    "digest": digest,
                    "count": len(members),
                    "splits": ";".join(splits),
                    "cross_split": len(splits) > 1,
                    "min_cross_split_frame_gap": "",
                    "member_ids": ";".join(member.file_id for member in members),
                    "relative_paths": (
                        ";".join(member.relative_path for member in members)
                        if include_paths
                        else ""
                    ),
                }
            )

    sequences: dict[str, list[tuple[ImageRecord, int]]] = defaultdict(list)
    for record in records:
        # Raw relative paths remain in memory only; reports contain a hashed prefix.
        signature = sequence_signature(record.source_relative_path)
        if signature is not None:
            prefix, frame = signature
            sequences[prefix].append((record, frame))
    for prefix, members in sorted(sequences.items()):
        splits = sorted({record.split for record, _ in members})
        if len(splits) < 2:
            continue
        cross_gaps = [
            abs(frame_a - frame_b)
            for record_a, frame_a in members
            for record_b, frame_b in members
            if record_a.split != record_b.split
        ]
        output.append(
            {
                "group_type": "filename_sequence",
                "digest": stable_identifier(prefix),
                "count": len(members),
                "splits": ";".join(splits),
                "cross_split": True,
                "min_cross_split_frame_gap": min(cross_gaps) if cross_gaps else "",
                "member_ids": ";".join(record.file_id for record, _ in members),
                "relative_paths": (
                    ";".join(record.relative_path for record, _ in members) if include_paths else ""
                ),
            }
        )
    return output


def _collect_integrity_issues(
    split: SplitPaths,
    summary: dict[str, Any],
    labels: dict[str, Any],
    expected_count: object,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    checks = (
        ("corrupt_images", "corrupt_images"),
        ("annotation_rows_without_image", "annotations_without_images"),
        ("images_without_annotation", "images_without_annotations"),
        ("invalid_point_rows", "invalid_point_rows"),
        ("out_of_bounds_points", "out_of_bounds_points"),
        ("duplicate_filenames", "duplicate_annotation_filenames"),
    )
    merged = {**summary, **labels}
    for field, code in checks:
        value = int(merged.get(field, 0) or 0)
        if value:
            issues.append(
                {
                    "severity": "error",
                    "split": split.name,
                    "code": code,
                    "count": value,
                }
            )
    if isinstance(expected_count, int) and not summary.get("scan_is_partial"):
        actual = int(summary["images_scanned"])
        if actual != expected_count:
            issues.append(
                {
                    "severity": "warning",
                    "split": split.name,
                    "code": "expected_count_mismatch",
                    "expected": expected_count,
                    "actual": actual,
                }
            )
    return issues


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 0 dataset audit",
        "",
        "> Generated locally. The report excludes absolute paths and raw coordinates.",
        "> Source images, annotations, and derived overlays must not be committed.",
        "",
        f"Generated (UTC): `{payload['generated_utc']}`",
        f"Hash mode: `{payload['hash_mode']}`",
        f"Partial scan: `{payload['partial_scan']}`",
        "",
        "## Split summary",
        "",
        "| Split | Images | Annotation rows | Corrupt | Missing image labels | Unlabeled images |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, split in payload["splits"].items():
        lines.append(
            f"| {name} | {split['images_scanned']} | {split['annotation_rows']} | "
            f"{split['corrupt_images']} | {split['annotation_rows_without_image']} | "
            f"{split['images_without_annotation']} |"
        )
    lines.extend(["", "## Findings", ""])
    if payload["issues"]:
        for issue in payload["issues"]:
            detail = ", ".join(
                f"{key}={value}"
                for key, value in issue.items()
                if key not in {"severity", "split", "code"}
            )
            suffix = f" ({detail})" if detail else ""
            lines.append(f"- **{issue['severity']}** `{issue['split']}/{issue['code']}`{suffix}")
    else:
        lines.append("- No configured integrity issue was detected.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Filename prefixes are reported only as uninterpreted sequence candidates.",
            "- A shared prefix is not proof of a shared patient or video.",
            "- PS1/PS2/FH1 are source fields; PS_R/PS_L/FH_T aliases remain unresolved.",
            "- Dataset licensing must be confirmed before publishing data-derived artifacts.",
        ]
    )
    return "\n".join(lines)


def _data_required_report(
    missing_inputs: list[dict[str, Any]], expected_mismatches: list[dict[str, Any]]
) -> str:
    lines = [
        "# Phase 0 data requirements",
        "",
        "This file records missing or unresolved inputs without exposing local paths.",
        "",
    ]
    if not missing_inputs and not expected_mismatches:
        lines.append(
            "All required configured inputs were found. Optional data may still be absent."
        )
    if missing_inputs:
        lines.extend(["## Missing configured inputs", ""])
        for item in missing_inputs:
            requirement = "required" if item["required"] else "optional"
            lines.append(f"- `{item['split']}`: missing {item['kind']} ({requirement}).")
    if expected_mismatches:
        lines.extend(["", "## Count differences", ""])
        for item in expected_mismatches:
            lines.append(
                f"- `{item['split']}`: found {item['actual']}, configured expectation "
                f"{item['expected']}. No explanation is assumed."
            )
    lines.extend(
        [
            "",
            "The unlabeled count remains deliberately unspecified because available public "
            "materials disagree.",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    splits = resolve_splits(config, PROJECT_ROOT)
    audit_config = config.get("audit", {})
    if not isinstance(audit_config, dict):
        raise ConfigError("audit must be a mapping")
    output_value = args.output_dir or audit_config.get("output_dir", "reports/phase0/generated")
    output_dir = ensure_output_directory(
        _resolve_generated_path(output_value), [split.image_dir for split in splits]
    )
    hash_mode = str(args.hash_mode or audit_config.get("hash_mode", "both"))
    if hash_mode not in {"none", "file", "pixels", "both"}:
        raise ConfigError(f"Unsupported hash mode: {hash_mode}")
    include_paths = bool(
        args.include_relative_paths or audit_config.get("include_relative_paths", False)
    )
    if args.max_images_per_split is not None and args.max_images_per_split <= 0:
        raise ConfigError("--max-images-per-split must be positive")

    task = config.get("task", {})
    aop_definition: dict[str, Any] | None = None
    if isinstance(task, dict) and isinstance(task.get("aop"), dict):
        candidate = task["aop"]
        required_aop = {"vertex_key", "pubic_axis_other_key", "fetal_head_key"}
        if (
            bool(candidate.get("source_definition_confirmed", False))
            and required_aop <= candidate.keys()
        ):
            aop_definition = candidate

    expected_counts = config.get("data", {}).get("expected_counts", {})
    if not isinstance(expected_counts, dict):
        expected_counts = {}

    all_records: list[ImageRecord] = []
    split_payloads: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    missing_inputs: list[dict[str, Any]] = []

    for split in splits:
        if not split.image_dir.is_dir():
            severity = "error" if split.required else "warning"
            issues.append(
                {
                    "severity": severity,
                    "split": split.name,
                    "code": "missing_image_directory",
                    "detail": "Configured image directory was not found",
                }
            )
            missing_inputs.append(
                {"split": split.name, "kind": "image directory", "required": split.required}
            )
            split_payloads[split.name] = {
                "required": split.required,
                "available": False,
                "images_scanned": 0,
                "scan_is_partial": False,
                "annotation_rows": 0,
                "matched_annotation_rows": 0,
                "annotation_rows_without_image": 0,
                "images_without_annotation": 0,
                "decodable_images": 0,
                "corrupt_images": 0,
                "image_size_distribution": {},
                "image_mode_distribution": {},
                "bytes_scanned": 0,
                "labels": _audit_labels(None, {}, None),
            }
            continue

        table, annotation_issues = _load_split_annotations(split)
        issues.extend(annotation_issues)
        if split.csv is not None and table is None:
            missing_inputs.append(
                {"split": split.name, "kind": "valid annotation CSV", "required": split.required}
            )
        records, summary, matched_sizes = _scan_images(
            split,
            table,
            hash_mode,
            include_paths,
            args.max_images_per_split,
        )
        label_summary = _audit_labels(table, matched_sizes, aop_definition)
        split_payloads[split.name] = {
            "required": split.required,
            "available": True,
            **summary,
            "labels": label_summary,
        }
        issues.extend(
            _collect_integrity_issues(
                split, summary, label_summary, expected_counts.get(split.name)
            )
        )
        all_records.extend(records)

    duplicate_rows = _duplicate_rows(all_records, include_paths)
    cross_split_file_groups = sum(
        row["group_type"] == "file_sha256" and bool(row["cross_split"]) for row in duplicate_rows
    )
    cross_split_pixel_groups = sum(
        row["group_type"] == "pixel_sha256" and bool(row["cross_split"]) for row in duplicate_rows
    )
    if cross_split_file_groups or cross_split_pixel_groups:
        issues.append(
            {
                "severity": "error",
                "split": "global",
                "code": "cross_split_exact_duplicates",
                "file_hash_groups": cross_split_file_groups,
                "pixel_hash_groups": cross_split_pixel_groups,
            }
        )

    expected_mismatches = [
        {
            "split": issue["split"],
            "expected": issue["expected"],
            "actual": issue["actual"],
        }
        for issue in issues
        if issue["code"] == "expected_count_mismatch"
    ]
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config_path.name,
        "privacy": {
            "absolute_paths_included": False,
            "relative_paths_included": include_paths,
            "raw_coordinates_included": False,
        },
        "hash_mode": hash_mode,
        "partial_scan": args.max_images_per_split is not None,
        "splits": split_payloads,
        "global": {
            "images_scanned": len(all_records),
            "bytes_scanned": sum(record.size_bytes for record in all_records),
            "duplicate_file_hash_groups": sum(
                row["group_type"] == "file_sha256" for row in duplicate_rows
            ),
            "duplicate_pixel_hash_groups": sum(
                row["group_type"] == "pixel_sha256" for row in duplicate_rows
            ),
            "cross_split_file_hash_groups": cross_split_file_groups,
            "cross_split_pixel_hash_groups": cross_split_pixel_groups,
            "cross_split_filename_sequence_candidates": sum(
                row["group_type"] == "filename_sequence" for row in duplicate_rows
            ),
        },
        "issues": issues,
    }

    inventory_fields = [
        "split",
        "file_id",
        "relative_path",
        "extension",
        "size_bytes",
        "width",
        "height",
        "mode",
        "decodable",
        "file_sha256",
        "pixel_sha256",
        "annotation_present",
        "error_type",
    ]
    inventory_rows = [
        {
            "split": record.split,
            "file_id": record.file_id,
            "relative_path": record.relative_path,
            "extension": record.extension,
            "size_bytes": record.size_bytes,
            "width": record.width if record.width is not None else "",
            "height": record.height if record.height is not None else "",
            "mode": record.mode or "",
            "decodable": record.decodable,
            "file_sha256": record.file_sha256 or "",
            "pixel_sha256": record.pixel_sha256 or "",
            "annotation_present": record.annotation_present,
            "error_type": record.error_type or "",
        }
        for record in all_records
    ]
    duplicate_fields = [
        "group_type",
        "digest",
        "count",
        "splits",
        "cross_split",
        "min_cross_split_frame_gap",
        "member_ids",
        "relative_paths",
    ]
    write_json(output_dir / "dataset_statistics.json", payload)
    write_csv_rows(output_dir / "file_inventory.csv", inventory_fields, inventory_rows)
    write_csv_rows(output_dir / "duplicate_report.csv", duplicate_fields, duplicate_rows)
    write_text(output_dir / "DATA_AUDIT.md", _markdown_report(payload))
    write_text(
        output_dir / "DATA_REQUIRED.md",
        _data_required_report(missing_inputs, expected_mismatches),
    )

    print(f"Audit complete: {len(all_records)} images; reports saved to {output_dir}")
    print("Privacy: absolute paths and raw coordinates were not written.")
    required_missing = any(item["required"] for item in missing_inputs)
    if required_missing:
        return 2
    if args.strict and issues:
        return 1
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
