from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from geoequi_ld.data.access_policy import (
    LabeledSplitSpec,
    fingerprint_labeled_split,
    load_phase05_local_splits,
    verify_fingerprint,
)


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("forbidden", ["test", "testing", "TESTING"])
def test_phase05_rejects_forbidden_split_key_before_data_access(
    tmp_path: Path,
    forbidden: str,
) -> None:
    config = tmp_path / "local.yaml"
    _write_yaml(
        config,
        {
            "phase": "phase0.5",
            "splits": {
                "train": {"image_dir": "train", "labels_csv": "train.csv"},
                "validation": {"image_dir": "validation", "labels_csv": "val.csv"},
                forbidden: {"image_dir": "must-not-open", "labels_csv": "must-not-open.csv"},
            },
        },
    )
    with pytest.raises(PermissionError):
        load_phase05_local_splits(config)


def test_phase05_rejects_testing_path_hidden_under_train_role(tmp_path: Path) -> None:
    config = tmp_path / "local.yaml"
    _write_yaml(
        config,
        {
            "phase": "phase0.5",
            "splits": {
                "train": {"image_dir": "dataset/Testing", "labels_csv": "train.csv"},
                "validation": {"image_dir": "validation", "labels_csv": "val.csv"},
            },
        },
    )
    with pytest.raises(PermissionError, match="testing path"):
        load_phase05_local_splits(config)


def test_fingerprint_verification_detects_content_change(tmp_path: Path) -> None:
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    (image_dir / "sample.png").write_bytes(b"synthetic-image")
    labels = image_dir / "labels.csv"
    labels.write_text(
        'Filename,PS1,PS2,FH1\nsample.png,"(1, 2)","(3, 4)","(5, 6)"\n',
        encoding="utf-8",
    )
    spec = LabeledSplitSpec("train", image_dir, labels, "FH1", {})
    expected = fingerprint_labeled_split(spec)
    verify_fingerprint(expected, expected, role="train")
    (image_dir / "sample.png").write_bytes(b"changed")
    changed = fingerprint_labeled_split(spec)
    with pytest.raises(PermissionError, match="aggregate_sha256"):
        verify_fingerprint(changed, expected, role="train")
