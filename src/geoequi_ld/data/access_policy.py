"""Fail-closed data access policy for the Phase 0.5 validation-only audit."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from geoequi_ld.utils.hashing import sha256_file

PHASE05_ALLOWED_SPLITS = frozenset({"train", "validation"})
PHASE05_FORBIDDEN_NAMES = frozenset({"test", "testing"})
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class LabeledSplitSpec:
    role: str
    image_dir: Path
    labels_csv: Path
    fh1_column: str
    expected_fingerprint: Mapping[str, str | int]


def _contains_forbidden_component(value: str) -> bool:
    components = [part.casefold() for part in re.split(r"[\\/]+", value) if part]
    return any(part in PHASE05_FORBIDDEN_NAMES for part in components)


def _reject_forbidden_values(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text.casefold() in PHASE05_FORBIDDEN_NAMES:
                raise PermissionError(f"Phase 0.5 forbids split key {key_text!r} at {location}")
            _reject_forbidden_values(nested, location=f"{location}.{key_text}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_values(nested, location=f"{location}[{index}]")
    elif isinstance(value, str) and _contains_forbidden_component(value):
        raise PermissionError(f"Phase 0.5 forbids a testing path at {location}")


def load_phase05_local_splits(path: str | Path) -> dict[str, LabeledSplitSpec]:
    """Load exactly train/validation paths, rejecting testing before path access."""

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("Phase 0.5 local configuration must be a mapping")
    _reject_forbidden_values(loaded)
    phase = str(loaded.get("phase", "")).strip().casefold()
    if phase not in {"phase0.5", "phase05"}:
        raise ValueError("Local configuration phase must be phase0.5")
    raw_splits = loaded.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("Local configuration requires a splits mapping")
    roles = {str(role).strip().casefold() for role in raw_splits}
    if roles != PHASE05_ALLOWED_SPLITS:
        raise PermissionError(
            "Phase 0.5 local configuration must contain exactly train and validation"
        )
    result: dict[str, LabeledSplitSpec] = {}
    for raw_role, raw_spec in raw_splits.items():
        role = str(raw_role).strip().casefold()
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Split {role!r} must be a mapping")
        allowed_keys = {"image_dir", "labels_csv", "fh1_column", "expected_fingerprint"}
        unknown = set(raw_spec) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown fields for split {role!r}: {sorted(unknown)}")
        image_dir_text = str(raw_spec.get("image_dir", ""))
        labels_csv_text = str(raw_spec.get("labels_csv", ""))
        if not image_dir_text or not labels_csv_text:
            raise ValueError(f"Split {role!r} requires image_dir and labels_csv")
        expected_fingerprint = raw_spec.get("expected_fingerprint")
        if not isinstance(expected_fingerprint, Mapping):
            raise ValueError(f"Split {role!r} requires a local expected_fingerprint mapping")
        if _contains_forbidden_component(image_dir_text) or _contains_forbidden_component(
            labels_csv_text
        ):
            raise PermissionError(f"Phase 0.5 forbids testing paths for split {role!r}")
        result[role] = LabeledSplitSpec(
            role=role,
            image_dir=Path(image_dir_text),
            labels_csv=Path(labels_csv_text),
            fh1_column=str(raw_spec.get("fh1_column", "FH1")),
            expected_fingerprint=dict(expected_fingerprint),
        )
    return result


def fingerprint_labeled_split(spec: LabeledSplitSpec) -> dict[str, str | int]:
    """Hash the exact label CSV and every referenced image into one aggregate digest."""

    if spec.role not in PHASE05_ALLOWED_SPLITS:
        raise PermissionError(f"Split role is not permitted in Phase 0.5: {spec.role!r}")
    if _contains_forbidden_component(str(spec.image_dir)) or _contains_forbidden_component(
        str(spec.labels_csv)
    ):
        raise PermissionError("Phase 0.5 refuses to fingerprint a testing path")
    if not spec.image_dir.is_dir() or not spec.labels_csv.is_file():
        raise FileNotFoundError(f"Missing local data for permitted split {spec.role!r}")
    image_root = spec.image_dir.resolve(strict=True)
    labels_path = spec.labels_csv.resolve(strict=True)
    if not labels_path.is_relative_to(image_root):
        raise PermissionError(f"Labels for split {spec.role!r} must stay inside its image root")
    with spec.labels_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    filenames = [str(row.get("Filename", "")) for row in rows]
    if not filenames or any(not name for name in filenames):
        raise ValueError(f"Split {spec.role!r} has an empty or invalid Filename column")
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Split {spec.role!r} contains duplicate filenames")
    digest = hashlib.sha256()
    labels_digest = sha256_file(spec.labels_csv)
    digest.update(f"labels:{labels_digest}\n".encode("ascii"))
    for filename in sorted(filenames):
        windows_name = PureWindowsPath(filename)
        posix_name = PurePosixPath(filename)
        components = [part for part in re.split(r"[\\/]+", filename) if part]
        if (
            windows_name.is_absolute()
            or posix_name.is_absolute()
            or ".." in components
            or _contains_forbidden_component(filename)
        ):
            raise PermissionError(f"Forbidden filename component in split {spec.role!r}")
        image_path = (image_root / Path(*components)).resolve(strict=False)
        if not image_path.is_relative_to(image_root):
            raise PermissionError(f"Image path escapes split root for {spec.role!r}")
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix in split {spec.role!r}")
        resolved_image = image_path.resolve(strict=True)
        if not image_path.is_file() or not resolved_image.is_relative_to(image_root):
            raise FileNotFoundError(f"A referenced image is missing in split {spec.role!r}")
        image_digest = sha256_file(image_path)
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(image_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "sample_count": len(filenames),
        "labels_sha256": labels_digest,
        "aggregate_sha256": digest.hexdigest(),
    }


def verify_fingerprint(
    actual: Mapping[str, str | int],
    expected: Mapping[str, str | int],
    *,
    role: str,
) -> None:
    required = {"sample_count", "labels_sha256", "aggregate_sha256"}
    if set(actual) != required or not required.issubset(expected):
        raise ValueError(f"Incomplete fingerprint contract for split {role!r}")
    mismatches = [key for key in sorted(required) if actual[key] != expected[key]]
    if mismatches:
        raise PermissionError(
            f"Phase 0.5 data fingerprint mismatch for {role!r}: {', '.join(mismatches)}"
        )
