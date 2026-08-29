from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.summarize_phase1b import (
    HISTORY_COLUMNS,
    _public_hygiene,
    _read_history,
    _selection_key,
    _write_curve,
)


def _row(epoch: int) -> dict[str, object]:
    ps1 = float(epoch + 1)
    ps2 = float(epoch + 2)
    fh1 = float(epoch + 3)
    return {
        "epoch": epoch,
        "train_time_sec": 1.0,
        "validation_time_sec": 0.5,
        "epoch_time_sec": 1.5,
        "train_total_loss": 0.5,
        "train_heatmap_mse": 0.01,
        "train_coordinate_smooth_l1": 0.02,
        "train_distribution_js": 0.2,
        "train_batches": 300,
        "val_total_loss": 0.5,
        "val_heatmap_mse": 0.01,
        "val_coordinate_smooth_l1": 0.02,
        "val_distribution_js": 0.2,
        "val_MRE_PS1": ps1,
        "val_MRE_PS2": ps2,
        "val_MRE_FH1": fh1,
        "val_MRE_ALL": (ps1 + ps2 + fh1) / 3.0,
        "val_n_samples": 100,
        "val_decoder": "dsnt",
        "val_n_valid_aop": 100,
        "val_n_evaluable_aop": 100,
        "val_aop_invalid_prediction_count": 0,
        "val_aop_mae_valid_deg": float(20 - epoch),
        "val_aop_mae_deg": float(20 - epoch),
    }


def _write_history(path: Path, epochs: int) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(_row(epoch) for epoch in range(1, epochs + 1))


def test_history_accepts_intentional_partial_16_epoch_run(tmp_path: Path) -> None:
    path = tmp_path / "train_log.csv"
    _write_history(path, 16)

    history = _read_history(path, expected_epochs=16)

    assert len(history) == 16
    assert min(history, key=_selection_key)["epoch"] == 16
    with pytest.raises(ValueError, match="history row count"):
        _read_history(path, expected_epochs=20)


def test_public_hygiene_rejects_paths_hashes_and_per_sample_material() -> None:
    _public_hygiene("validation aggregate only", context="safe fixture")
    with pytest.raises(ValueError, match="absolute path"):
        windows_path = "C" + ":\\Users\\person\\result.json"
        _public_hygiene(windows_path, context="path fixture")
    with pytest.raises(ValueError, match="forbidden public detail"):
        _public_hygiene({"sha256": "secret"}, context="hash fixture")
    with pytest.raises(ValueError, match="forbidden public detail"):
        _public_hygiene("sample_00.png", context="sample fixture")


def test_curve_has_fixed_size_and_no_png_text_metadata(tmp_path: Path) -> None:
    shared = tuple({"epoch": epoch, **_curve_metrics(epoch)} for epoch in range(1, 21))
    split = tuple({"epoch": epoch, **_curve_metrics(epoch + 1)} for epoch in range(1, 17))
    path = tmp_path / "validation_metrics.png"

    _write_curve(path, shared, split)

    with Image.open(path) as image:
        assert image.size == (1600, 672)
        assert image.info == {}


def _curve_metrics(value: int) -> dict[str, float]:
    return {
        "MRE_PS1": float(value),
        "MRE_PS2": float(value + 1),
        "MRE_FH1": float(value + 2),
        "MRE_ALL": float(value + 1),
        "aop_mae_deg": float(value / 2),
    }


def test_published_phase1b_results_keep_partial_run_and_claim_boundary_explicit() -> None:
    root = Path("reports/phase1b")
    aggregate = json.loads((root / "aggregate_results.json").read_text(encoding="utf-8"))
    h2 = aggregate["decoder_control"]["h2_split"]

    assert h2["status"] == "budget_exhausted"
    assert h2["partial"] is True
    assert h2["epochs_completed"] == 16
    assert h2["epochs_requested"] == 20
    assert h2["formal_elapsed_seconds"] < h2["formal_allocation_seconds"] == 7200.0
    assert h2["training_guard_seconds"] == 6600.0
    assert h2["post_evaluation_reserve_seconds"] == 600.0
    assert h2["ledger_closing_reserve_seconds"] == 120.0
    assert h2["training_subbudget_exhausted"] is True
    assert h2["formal_allocation_exceeded"] is False
    assert h2["aggregate_gpu_cap_exceeded"] is False
    assert aggregate["resources"]["allocation_or_total_limit_exceeded"] is False
    assert aggregate["data_scope"]["testing_frozen"] is True
    assert aggregate["decoder_control"]["selected_best_delta_h2_minus_h1"]["MRE_PS2"] < 0
    assert aggregate["decoder_control"]["selected_best_delta_h2_minus_h1"]["MRE_FH1"] > 0
    matched = aggregate["decoder_control"]["matched_epoch16"]["delta_h2_minus_h1"]
    assert matched["MRE_PS2"] == pytest.approx(3.129101, abs=1e-6)
    assert matched["MRE_FH1"] == pytest.approx(-9.264206, abs=1e-6)
    assert aggregate["decoder_control"]["parameter_counts"] == {
        "h1_shared": 29318355,
        "h2_split": 29332275,
        "h2_minus_h1": 13920,
    }
    assert h2["best_train"] == {
        "MRE_PS1": 13.443938,
        "MRE_PS2": 15.123833,
        "MRE_FH1": 36.349663,
        "MRE_ALL": 21.639143,
        "aop_mae_deg": 9.69658,
        "n_valid_aop": 300,
        "n_evaluable_aop": 300,
    }
    assert h2["best_validation_minus_train"] == {
        "MRE_PS1": 4.120486,
        "MRE_PS2": 9.069195,
        "MRE_FH1": 15.447784,
        "MRE_ALL": 9.545824,
        "aop_mae_deg": 3.866484,
    }
    assert h2["last_observed_train"] == {
        "MRE_PS1": 11.998107,
        "MRE_PS2": 13.200794,
        "MRE_FH1": 22.453241,
        "MRE_ALL": 15.884047,
        "aop_mae_deg": 8.795021,
        "n_valid_aop": 300,
        "n_evaluable_aop": 300,
    }
    assert h2["last_observed_validation_minus_train"] == {
        "MRE_PS1": 5.4089,
        "MRE_PS2": 17.909628,
        "MRE_FH1": 14.28618,
        "MRE_ALL": 12.534904,
        "aop_mae_deg": 5.138007,
    }
    gaps = aggregate["bn_diagnostic"]["endpoints"]
    assert gaps["best"]["original_bn_validation_minus_train"]["MRE_ALL"] == pytest.approx(
        8.376735, abs=1e-6
    )
    assert gaps["last"]["original_bn_validation_minus_train"]["MRE_ALL"] == pytest.approx(
        16.017719, abs=1e-6
    )
    assert aggregate["historical_context_only"]["unet_B2"] == {
        "best_epoch": 15,
        "MRE_PS1": 13.416825,
        "MRE_PS2": 20.357471,
        "MRE_FH1": 40.564011,
        "MRE_ALL": 24.779436,
        "aop_mae_deg": 8.51385,
        "boundary": "Different architecture and budget; descriptive context only.",
    }
    assert any("not the complete" in item for item in aggregate["limitations"])

    published = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "aggregate_results.json",
            "sanitized_config.yaml",
            "BN_DIAGNOSTICS.md",
            "DECODER_COMPARISON.md",
            "PHASE1B_SUMMARY.md",
        )
    )
    _public_hygiene(published, context="published Phase 1B files")
    assert "16/20" in published
    assert "budget_exhausted" in published
    assert "PS2" in published and "FH1" in published
