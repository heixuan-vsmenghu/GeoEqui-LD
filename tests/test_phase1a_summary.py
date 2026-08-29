from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.summarize_phase1a import (
    HISTORY_COLUMNS,
    _read_history,
    _reject_forbidden_split_keys,
    _selection_key,
    _write_curve,
)


def _history_row(epoch: int) -> dict[str, object]:
    ps1 = float(epoch)
    ps2 = float(epoch + 1)
    fh1 = float(epoch + 2)
    overall = (ps1 + ps2 + fh1) / 3.0
    aop = float(25 - epoch)
    return {
        "epoch": epoch,
        "train_time_sec": 1.0,
        "validation_time_sec": 0.5,
        "epoch_time_sec": 1.5,
        "train_total_loss": 1.0,
        "train_heatmap_mse": 0.1,
        "train_coordinate_smooth_l1": 0.01,
        "train_distribution_js": 0.2,
        "train_batches": 300,
        "val_total_loss": 1.0,
        "val_heatmap_mse": 0.1,
        "val_coordinate_smooth_l1": 0.01,
        "val_distribution_js": 0.2,
        "val_MRE_PS1": ps1,
        "val_MRE_PS2": ps2,
        "val_MRE_FH1": fh1,
        "val_MRE_ALL": overall,
        "val_n_samples": 100,
        "val_decoder": "dsnt",
        "val_n_valid_aop": 100,
        "val_n_evaluable_aop": 100,
        "val_aop_invalid_prediction_count": 0,
        "val_aop_mae_valid_deg": aop,
        "val_aop_mae_deg": aop,
    }


def _write_history(path: Path, *, epochs: int = 20) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(_history_row(epoch) for epoch in range(1, epochs + 1))


def test_phase1a_history_requires_all_20_epochs_and_recomputes_selection(tmp_path: Path) -> None:
    path = tmp_path / "train_log.csv"
    _write_history(path)

    history = _read_history(path)

    assert len(history) == 20
    assert min(history, key=_selection_key)["epoch"] == 20

    _write_history(path, epochs=19)
    with pytest.raises(ValueError, match="history row count"):
        _read_history(path)


def test_phase1a_summary_rejects_forbidden_split_keys() -> None:
    with pytest.raises(PermissionError, match="forbidden split-derived"):
        _reject_forbidden_split_keys(
            {"validation": {}, "testing_metrics": {"MRE_ALL": 1.0}},
            context="fixture",
        )


def test_phase1a_curve_has_no_png_text_or_path_metadata(tmp_path: Path) -> None:
    history = tuple(
        {
            "epoch": epoch,
            "MRE_PS1": float(epoch),
            "MRE_PS2": float(epoch + 1),
            "MRE_FH1": float(epoch + 2),
            "MRE_ALL": float(epoch + 1),
            "aop_mae_deg": float(25 - epoch),
        }
        for epoch in range(1, 21)
    )
    path = tmp_path / "validation_metrics.png"

    _write_curve(path, history)

    with Image.open(path) as image:
        assert image.size == (1600, 672)
        assert image.info == {}


def test_published_phase1a_aggregate_is_sanitized_and_keeps_a4_failure_explicit() -> None:
    report_root = Path("reports/phase1a")
    aggregate_path = report_root / "aggregate_results.json"
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))

    assert payload["formal_run"]["best"]["epoch"] == 3
    assert payload["formal_run"]["last"]["epoch"] == 20
    assert payload["gates"]["A4_unet_B0"]["execution_status"] == "completed"
    assert payload["gates"]["A4_unet_B0"]["learning_outcome"] == "not_learned"
    assert not payload["gates"]["A4_unet_B0"]["learned_all_three_keypoints"]
    assert payload["gates"]["B4_hrnet_B2"]["n_valid_aop"] == 4

    public_text = "\n".join(
        (report_root / name).read_text(encoding="utf-8")
        for name in (
            "aggregate_results.json",
            "sanitized_config.yaml",
            "HRNET_IMPLEMENTATION.md",
            "PHASE1A_SUMMARY.md",
        )
    )
    windows_drive_prefix = "H" + ":\\"
    for forbidden in (
        windows_drive_prefix,
        "fingerprint",
        "sha256",
        "best_checkpoint",
        "private_predictions",
        "started_at_utc",
    ):
        assert forbidden not in public_text
