"""Normalize supported public IUGC annotation CSV layouts."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_KEYPOINTS = ("PS1", "PS2", "FH1")
GROUPING_COLUMN_NAMES = ("patient_id", "case_id", "video_id", "sequence_id", "frame_id")


@dataclass(frozen=True)
class AnnotationTable:
    """Canonical annotations plus non-inferred source metadata."""

    rows: pd.DataFrame
    source_columns: dict[str, str]
    grouping_columns: tuple[str, ...]
    format_name: str


def _column_token(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _find_column(columns: Sequence[object], candidates: Sequence[str]) -> str | None:
    lookup = {_column_token(column): str(column) for column in columns}
    for candidate in candidates:
        match = lookup.get(_column_token(candidate))
        if match is not None:
            return match
    return None


def parse_point(value: object) -> tuple[float, float]:
    """Parse a finite point encoded as ``(x, y)``, ``[x, y]``, or a pair."""

    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid point literal: {value!r}") from exc
    if not isinstance(parsed, tuple | list | np.ndarray) or len(parsed) != 2:
        raise ValueError(f"Expected a two-value point, got {parsed!r}")
    point = (float(parsed[0]), float(parsed[1]))
    if not np.isfinite(point).all():
        raise ValueError(f"Point contains NaN or Inf: {point}")
    return point


def load_annotations(csv_path: Path, aop_csv_path: Path | None = None) -> AnnotationTable:
    """Load a combined CSV or a landmarks/AoP CSV pair into canonical columns.

    Supported landmark layouts are ``Filename,PS1,PS2,FH1`` and
    ``Filename,PS1,PS2,AOP Tangency``. Angle columns are optional in the
    landmark file and may be supplied by a second ``Filename,AOP`` file.
    """

    landmarks = pd.read_csv(csv_path)
    if landmarks.empty:
        raise ValueError(f"Annotation CSV contains no rows: {csv_path.name}")

    filename = _find_column(landmarks.columns, ("Filename", "Image filename"))
    ps1 = _find_column(landmarks.columns, ("PS1",))
    ps2 = _find_column(landmarks.columns, ("PS2",))
    fh1 = _find_column(landmarks.columns, ("FH1", "AOP Tangency"))
    missing = [
        label
        for label, source in (("Filename", filename), ("PS1", ps1), ("PS2", ps2), ("FH1", fh1))
        if source is None
    ]
    if missing:
        raise ValueError(
            f"Unsupported annotation schema in {csv_path.name}; missing canonical fields {missing}"
        )

    source_columns = {
        "Filename": str(filename),
        "PS1": str(ps1),
        "PS2": str(ps2),
        "FH1": str(fh1),
    }
    canonical = landmarks[[filename, ps1, ps2, fh1]].rename(
        columns={filename: "Filename", ps1: "PS1", ps2: "PS2", fh1: "FH1"}
    )
    angle_column = _find_column(landmarks.columns, ("AoP", "Angle of Progression"))
    if angle_column is not None and angle_column != fh1:
        canonical["AoP"] = pd.to_numeric(landmarks[angle_column], errors="coerce")
        source_columns["AoP"] = angle_column

    format_name = "combined"
    if aop_csv_path is not None:
        angles = pd.read_csv(aop_csv_path)
        angle_filename = _find_column(angles.columns, ("Filename", "Image filename"))
        angle_value = _find_column(angles.columns, ("AoP", "Angle of Progression"))
        if angle_filename is None or angle_value is None:
            raise ValueError(
                f"Unsupported AoP schema in {aop_csv_path.name}; expected Filename and AOP"
            )
        angle_frame = angles[[angle_filename, angle_value]].rename(
            columns={angle_filename: "Filename", angle_value: "AoP_from_pair"}
        )
        if bool(angle_frame["Filename"].astype(str).duplicated().any()):
            raise ValueError(f"Duplicate filenames in AoP CSV: {aop_csv_path.name}")
        canonical = canonical.merge(angle_frame, on="Filename", how="left", validate="many_to_one")
        pair_values = pd.to_numeric(canonical.pop("AoP_from_pair"), errors="coerce")
        if "AoP" in canonical:
            existing = pd.to_numeric(canonical["AoP"], errors="coerce")
            conflict = (
                existing.notna() & pair_values.notna() & ((existing - pair_values).abs() > 1e-6)
            )
            if bool(conflict.any()):
                raise ValueError("AoP values conflict between landmark and paired angle CSV files")
            canonical["AoP"] = existing.fillna(pair_values)
        else:
            canonical["AoP"] = pair_values
        source_columns["AoP"] = angle_value
        format_name = "paired_landmarks_and_aop"

    grouping_columns = tuple(
        str(column)
        for column in landmarks.columns
        if _column_token(column) in {_column_token(name) for name in GROUPING_COLUMN_NAMES}
    )
    canonical["Filename"] = canonical["Filename"].astype(str).str.replace("\\", "/", regex=False)
    return AnnotationTable(
        rows=canonical,
        source_columns=source_columns,
        grouping_columns=grouping_columns,
        format_name=format_name,
    )
