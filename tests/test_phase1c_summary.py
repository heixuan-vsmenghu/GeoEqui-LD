from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.summarize_phase1c import (
    HISTORY_COLUMNS,
    _public_hygiene,
    _read_history,
    _selection_key,
    _write_curves,
)


def _metrics(epoch: int, *, train: bool) -> dict[str, object]:
    offset = 0.0 if train else 1.0
    ps1 = float(epoch + offset)
    ps2 = float(epoch + 1 + offset)
    fh1 = float(epoch + 2 + offset)
    aop = float(20 - epoch + offset)
    prefix = "train" if train else "val"
    samples = 300 if train else 100
    return {
        f"{prefix}_total_loss": 0.5,
        f"{prefix}_heatmap_mse": 0.01,
        f"{prefix}_coordinate_smooth_l1": 0.02,
        f"{prefix}_distribution_js": 0.2,
        f"{prefix}_MRE_PS1": ps1,
        f"{prefix}_MRE_PS2": ps2,
        f"{prefix}_MRE_FH1": fh1,
        f"{prefix}_MRE_ALL": (ps1 + ps2 + fh1) / 3.0,
        f"{prefix}_n_samples": samples,
        f"{prefix}_decoder": "dsnt",
        f"{prefix}_n_valid_aop": samples,
        f"{prefix}_n_evaluable_aop": samples,
        f"{prefix}_aop_invalid_prediction_count": 0,
        f"{prefix}_aop_mae_valid_deg": aop,
        f"{prefix}_aop_mae_deg": aop,
        f"{prefix}_aop_valid_rate": 1.0,
        f"{prefix}_selection_aop_penalized_deg": aop,
    }


def _history_row(epoch: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "optimization_time_sec": 2.0,
        "train_evaluation_time_sec": 1.0,
        "validation_time_sec": 0.5,
        "epoch_time_sec": 3.5,
        "optimization_total_loss": 0.5,
        "optimization_heatmap_mse": 0.01,
        "optimization_coordinate_smooth_l1": 0.02,
        "optimization_distribution_js": 0.2,
        "optimization_batches": 300,
        **_metrics(epoch, train=True),
        **_metrics(epoch, train=False),
    }


def _write_history(path: Path, *, epochs: int = 16) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(_history_row(epoch) for epoch in range(1, epochs + 1))


def test_history_requires_complete_16_epochs_and_recomputes_selection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train_log.csv"
    _write_history(path)

    history = _read_history(path)

    assert len(history) == 16
    assert min(history, key=_selection_key)["epoch"] == 16
    _write_history(path, epochs=15)
    with pytest.raises(ValueError, match="history row count"):
        _read_history(path)


def test_public_hygiene_rejects_paths_hashes_and_per_sample_material() -> None:
    _public_hygiene("aggregate validation metrics", context="safe fixture")
    with pytest.raises(ValueError, match="absolute path"):
        _public_hygiene("C" + ":\\Users\\person\\result.json", context="path fixture")
    with pytest.raises(ValueError, match="provenance hash"):
        _public_hygiene("a" * 64, context="hash fixture")
    with pytest.raises(ValueError, match="forbidden public detail"):
        _public_hygiene("sample_00.png", context="sample fixture")


def _phase1b_fixture() -> dict[str, dict[str, object]]:
    def point(value: float) -> dict[str, float]:
        return {
            "MRE_PS1": value,
            "MRE_PS2": value + 1,
            "MRE_FH1": value + 2,
            "MRE_ALL": value + 1,
            "aop_mae_deg": value / 2,
        }

    return {
        "h1_best": point(2.0),
        "h2_best": point(3.0),
        "h1_e16": point(4.0),
        "h2_e16": point(5.0),
        "h2_gap_e3": point(1.0),
        "h2_gap_e16": point(2.0),
    }


def test_curves_have_fixed_size_and_no_png_text_metadata(tmp_path: Path) -> None:
    history = tuple(
        {
            "epoch": epoch,
            "train": {
                "MRE_PS1": float(epoch),
                "MRE_PS2": float(epoch + 1),
                "MRE_FH1": float(epoch + 2),
                "MRE_ALL": float(epoch + 1),
                "aop_mae_deg": float(epoch / 2),
            },
            "validation": {
                "MRE_PS1": float(epoch + 1),
                "MRE_PS2": float(epoch + 2),
                "MRE_FH1": float(epoch + 3),
                "MRE_ALL": float(epoch + 2),
                "aop_mae_deg": float(epoch / 2 + 1),
            },
        }
        for epoch in range(1, 17)
    )

    _write_curves(tmp_path, history, _phase1b_fixture())

    for name in ("validation_metrics.png", "h3_train_validation_gap.png"):
        with Image.open(tmp_path / name) as image:
            assert image.size == (1600, 672)
            assert image.info == {}


def test_published_phase1c_results_are_sanitized_and_keep_claim_boundary() -> None:
    root = Path("reports/phase1c")
    aggregate = json.loads((root / "aggregate_results.json").read_text(encoding="utf-8"))
    comparison = aggregate["supervised_comparison"]

    assert aggregate["data_scope"] == {
        "train_samples": 300,
        "validation_samples": 100,
        "testing_frozen": True,
        "unlabeled_training_used": False,
    }
    assert aggregate["determinism"]["warn_only"] is True
    assert aggregate["determinism"]["strict_bitwise_reproducibility_claimed"] is False
    assert aggregate["architecture"]["parameter_counts"] == {
        "h1_shared": 29318355,
        "h2_split": 29332275,
        "h3_specialized": 29372695,
        "h3_minus_h2": 40420,
        "ps_enhancer": 17148,
        "fh_enhancer": 23272,
    }
    assert aggregate["gates"]["deformable_operator"]["actual_operator"] == (
        "torchvision.ops.DeformConv2d"
    )
    tiny = aggregate["gates"]["four_sample_learning"]
    assert tiny["status"] == "PASS"
    assert tiny["MRE_ALL"] == pytest.approx(4.689429, abs=1e-6)
    assert tiny["aop_mae_deg"] == pytest.approx(1.388813, abs=1e-6)
    assert tiny["execution_history"]["failed_attempts_before_pass"] == 1
    assert tiny["execution_history"]["minimal_gate_fixes"] == 1
    assert tiny["execution_history"][
        "model_loss_threshold_or_step_limit_changed"
    ] is False
    assert aggregate["formal_run"]["selected_epoch"] == 14
    assert aggregate["formal_run"]["epochs_completed"] == 16
    assert comparison["selected_best"]["delta_h3_minus_h2"] == {
        "MRE_PS1": -5.138397,
        "MRE_PS2": -2.747203,
        "MRE_FH1": -10.966629,
        "MRE_ALL": -6.284077,
        "aop_mae_deg": -3.273884,
    }
    assert comparison["matched_epoch3"]["delta_h3_minus_h2"]["MRE_PS2"] == (
        pytest.approx(2.744942, abs=1e-6)
    )
    assert comparison["matched_epoch16"]["delta_h3_minus_h2"]["MRE_FH1"] == (
        pytest.approx(3.891468, abs=1e-6)
    )
    assert comparison["matched_epoch16"]["delta_h3_minus_h2"]["aop_mae_deg"] == (
        pytest.approx(0.346216, abs=1e-6)
    )
    assert comparison["consistent_all_keypoint_improvement"] is False
    assert aggregate["unlabeled_intake"]["trainable_files_available"] == 0
    assert aggregate["unlabeled_intake"]["conflicting_documented_counts"] == [
        31421,
        31121,
    ]

    public_text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "aggregate_results.json",
            "sanitized_config.yaml",
            "SPECIALIZED_ARCHITECTURE.md",
            "SPECIALIZED_COMPARISON.md",
            "PHASE1C_SUMMARY.md",
            "UNLABELED_INTAKE.md",
        )
    )
    _public_hygiene(public_text, context="published Phase 1C files")
    assert "16/16" in public_text
    assert "warn-only" in public_text
    assert "位级复现" in public_text
    assert "forward_ps" in public_text and "forward_fh" in public_text
    assert "testing" in public_text and "冻结" in public_text
