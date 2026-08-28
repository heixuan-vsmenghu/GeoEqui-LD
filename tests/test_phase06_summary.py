from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.summarize_phase06 import (
    HISTORY_COLUMNS,
    METRIC_NAMES,
    MILESTONES,
    VARIANTS,
    ValidatedRun,
    _build_conclusions,
    _catch_comparison,
    _effect_answer,
    _effect_diagnostic,
    _format_epoch,
    summarize_phase06,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(variant: str, epoch: int) -> dict[str, Any]:
    centers = {"B0": 160, "B1": 100, "B2": 80}
    mre_bases = {"B0": 70.0, "B1": 40.0, "B2": 30.0}
    aop_bases = {"B0": 25.0, "B1": 12.0, "B2": 10.0}
    distance = abs(epoch - centers[variant])
    overall = mre_bases[variant] + distance * 0.10
    aop = aop_bases[variant] + distance * 0.05
    ps1 = overall - 2.0
    ps2 = overall + 0.5
    fh1 = overall + 1.5
    # Mirror harmless CSV/JSON decimal rounding instead of constructing an exact identity.
    overall = (ps1 + ps2 + fh1) / 3.0 + 4.0e-6
    coordinate = 0.1 if variant in {"B1", "B2"} else 0.0
    distribution = 0.02 if variant == "B2" else 0.0
    heatmap = 1.0 / (epoch + 1)
    return {
        "total_loss": heatmap + 10.0 * coordinate + distribution,
        "heatmap_mse": heatmap,
        "coordinate_smooth_l1": coordinate,
        "distribution_js": distribution,
        "MRE_PS1": ps1,
        "MRE_PS2": ps2,
        "MRE_FH1": fh1,
        "MRE_ALL": overall,
        "n_samples": 100,
        "decoder": "dsnt",
        "n_valid_aop": 100,
        "n_evaluable_aop": 100,
        "aop_invalid_prediction_count": 0,
        "aop_mae_valid_deg": aop,
        "aop_mae_deg": aop,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_history(path: Path, variant: str) -> None:
    rows: list[dict[str, Any]] = []
    for epoch in range(1, 201):
        metrics = _metrics(variant, epoch)
        rows.append(
            {
                "epoch": epoch,
                "train_time_sec": 1.0,
                "validation_time_sec": 0.5,
                "epoch_time_sec": 1.5,
                "train_total_loss": metrics["total_loss"],
                "train_heatmap_mse": metrics["heatmap_mse"],
                "train_coordinate_smooth_l1": metrics["coordinate_smooth_l1"],
                "train_distribution_js": metrics["distribution_js"],
                "train_batches": 300,
                **{f"val_{key}": value for key, value in metrics.items()},
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_run(
    run_root: Path,
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    variant: str,
) -> None:
    run_dir = run_root / variant / "seed_42"
    run_dir.mkdir(parents=True)
    training = dict(protocol["training"])
    weights = protocol["variants"][variant]["weights"]
    training.update(
        {
            "heatmap_loss_weight": float(weights[0]),
            "coordinate_loss_weight": float(weights[1]),
            "distribution_loss_weight": float(weights[2]),
        }
    )
    initialization = hashlib.sha256(b"same-initialization").hexdigest()
    commit = "1" * 40
    provenance = {
        "protocol_sha256": protocol_sha256,
        "git_commit": commit,
        "git_dirty": False,
        "parent_phase05_commit": protocol["project"]["parent_phase05_commit"],
        "parent_phase05_tag": protocol["project"]["parent_phase05_tag"],
    }
    config = {
        "schema_version": 1,
        "phase": "phase0.6-long-budget-fidelity",
        "variant": variant,
        "variant_description": protocol["variants"][variant]["description"],
        "training": training,
        "optimizer": protocol["optimizer"],
        "selection": {
            "split": "validation",
            "common_decoder": "dsnt",
            "checkpoint_selection": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
            "milestones": list(MILESTONES),
        },
        "testing_frozen": True,
        "data": {
            "train": {
                "sample_count": 300,
                "labels_sha256": "2" * 64,
                "aggregate_sha256": "3" * 64,
                "source_columns": {"PS1": "PS1", "PS2": "PS2", "FH1": "FH1"},
            },
            "validation": {
                "sample_count": 100,
                "labels_sha256": "4" * 64,
                "aggregate_sha256": "5" * 64,
                "source_columns": {
                    "PS1": "PS1",
                    "PS2": "PS2",
                    "FH1": "AOP Tangency",
                },
            },
        },
        "model": {
            "class": "HeatmapUNet",
            "trainable_parameters": 484171,
            "initialization_sha256": initialization,
        },
        "order_audit": {
            "algorithm": "RandomSampler-compatible randperm",
            "generator_seed": 42,
            "per_epoch_filename_order_sha256": True,
        },
        "provenance": provenance,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "config.json", config)
    _write_history(run_dir / "train_log.csv", variant)
    order = {
        "schema_version": 1,
        "contains_filenames": False,
        "records": [
            {
                "epoch": epoch,
                "sample_count": 300,
                "filename_order_sha256": hashlib.sha256(
                    f"shared-order-{epoch}".encode()
                ).hexdigest(),
            }
            for epoch in range(1, 201)
        ],
    }
    _write_json(run_dir / "training_order.json", order)
    _write_json(
        run_dir / "environment.json",
        {
            "platform": "synthetic-platform",
            "python": "3.11.4",
            "torch": "2.5.1+cpu",
            "device": "cpu",
            "cuda_available": False,
            "torch_cuda": None,
            "cudnn": None,
        },
    )
    (run_dir / "best.pt").write_bytes(f"best-{variant}".encode())
    (run_dir / "last.pt").write_bytes(f"last-{variant}".encode())

    best_epoch = {"B0": 160, "B1": 100, "B2": 80}[variant]
    best_metrics = _metrics(variant, best_epoch)
    last_metrics = _metrics(variant, 200)
    metrics_file = {
        "status": "completed",
        "selection_split": "validation",
        "checkpoint_metric": "aop_mae_deg",
        "selection_tiebreak": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
        "best_epoch": best_epoch,
        "best_value": best_metrics["aop_mae_deg"],
        "best_validation_metrics": best_metrics,
        "last_validation_metrics": last_metrics,
        "best_checkpoint": str((run_dir / "best.pt").resolve()),
        "last_checkpoint": str((run_dir / "last.pt").resolve()),
    }
    _write_json(run_dir / "metrics.json", metrics_file)
    milestone_metrics = {
        str(epoch): {
            "epoch": epoch,
            **{name: _metrics(variant, epoch)[name] for name in METRIC_NAMES},
        }
        for epoch in MILESTONES
    }
    milestone_metrics["best"] = {
        "epoch": best_epoch,
        **{name: best_metrics[name] for name in METRIC_NAMES},
    }
    result = {
        "status": "completed",
        "phase": "phase0.6-long-budget-fidelity",
        "variant": variant,
        "seed": 42,
        "epochs_completed": 200,
        "selection_split": "validation",
        "selection_decoder": "dsnt",
        "testing_frozen": True,
        "best_epoch": best_epoch,
        "best_validation_metrics": {"dsnt": best_metrics},
        "milestone_validation_metrics": milestone_metrics,
        "last_validation_metrics": last_metrics,
        "order_audit": {
            "recorded_epochs": 200,
            "samples_per_epoch": 300,
            "contains_filenames": False,
        },
        "resources": {
            "training_runtime_sec": 300.0,
            "evaluation_runtime_sec": 1.0,
            "peak_gpu_allocated_mb": None,
            "peak_gpu_reserved_mb": None,
        },
        "model": config["model"],
        "provenance": {
            **provenance,
            "best_checkpoint_sha256": _sha256(run_dir / "best.pt"),
        },
    }
    _write_json(run_dir / "phase06_result.json", result)


@pytest.fixture
def phase06_matrix(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_protocol = Path("configs/phase06_long_budget.yaml")
    protocol = yaml.safe_load(source_protocol.read_text(encoding="utf-8"))
    protocol_path = tmp_path / "phase06.yaml"
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    protocol_sha256 = _sha256(protocol_path)
    run_root = tmp_path / "runs"
    for variant in VARIANTS:
        _write_run(
            run_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            variant=variant,
        )
    return protocol_path, run_root, tmp_path / "reports"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_summary_validates_three_runs_and_writes_sanitized_reports(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    aggregate = summarize_phase06(
        protocol_path=protocol_path,
        run_root=run_root,
        report_root=report_root,
    )

    assert aggregate["integrity"]["validated_run_count"] == 3
    assert [run["best_epoch"] for run in aggregate["runs"]] == [160, 100, 80]
    assert len(aggregate["conclusions"]) == 5
    assert aggregate["comparisons"]["B0_after_epoch20"]["clearly_improved_by_descriptive_rule"]
    for path in (
        report_root / "PHASE06_SUMMARY.md",
        report_root / "LONG_BUDGET_COMPARISON.md",
        report_root / "aggregate_results.json",
        report_root / "sanitized_config.yaml",
        report_root / "curves" / "validation_metrics.png",
    ):
        assert path.is_file()
    public_text = (report_root / "aggregate_results.json").read_text(encoding="utf-8")
    assert "total_loss" not in public_text
    assert '"weights"' not in public_text
    assert "labels_sha256" not in public_text
    assert "initialization_sha256" not in public_text
    assert "1" * 40 not in public_text
    assert str(run_root.resolve()) not in public_text


def test_summary_accepts_realistic_nonidentical_mre_rounding(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    sample = _metrics("B0", 20)
    exact_mean = sum(sample[name] for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1")) / 3.0
    assert 0 < abs(sample["MRE_ALL"] - exact_mean) < 1.0e-5

    aggregate = summarize_phase06(
        protocol_path=protocol_path,
        run_root=run_root,
        report_root=report_root,
    )
    assert aggregate["integrity"]["all_best_tuples_recomputed"]


def test_summary_accepts_zero_valid_aop_with_finite_penalty_and_reports_collapse(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    run_dir = run_root / "B0" / "seed_42"
    history_path = run_dir / "train_log.csv"
    rows = _read_csv(history_path)
    final_row = rows[-1]
    final_row["val_n_valid_aop"] = "0"
    final_row["val_aop_invalid_prediction_count"] = "100"
    final_row["val_aop_mae_valid_deg"] = "nan"
    final_row["val_aop_mae_deg"] = "180.0"
    _rewrite_csv(history_path, rows)

    def mark_payload(payload: dict[str, Any]) -> None:
        payload["n_valid_aop"] = 0
        payload["aop_invalid_prediction_count"] = 100
        payload["aop_mae_valid_deg"] = None
        payload["aop_mae_deg"] = 180.0

    result_path = run_dir / "phase06_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mark_payload(result["last_validation_metrics"])
    result["milestone_validation_metrics"]["200"]["aop_mae_deg"] = 180.0
    _write_json(result_path, result)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    mark_payload(metrics["last_validation_metrics"])
    _write_json(metrics_path, metrics)

    aggregate = summarize_phase06(
        protocol_path=protocol_path,
        run_root=run_root,
        report_root=report_root,
    )
    b0 = next(run for run in aggregate["runs"] if run["variant"] == "B0")
    diagnostics = b0["aop_validity_diagnostics"]
    assert diagnostics["any_invalid_prediction_epochs"] == [200]
    assert diagnostics["first_any_invalid_prediction_epoch"] == 200
    assert diagnostics["zero_valid_epochs"] == [200]
    assert diagnostics["full_collapse_epochs"] == [200]
    assert diagnostics["full_collapse_after_epoch20_count"] == 1
    summary = (report_root / "PHASE06_SUMMARY.md").read_text(encoding="utf-8")
    assert "连续 1 轮为AoP full-collapse" in summary
    assert "后期解码崩溃" in summary


def test_effect_classifies_early_best_without_sustained_endpoint_as_speed() -> None:
    def run(variant: str, history: tuple[dict[str, float | int], ...]) -> ValidatedRun:
        return ValidatedRun(
            variant=variant,
            config={},
            result={},
            history=history,
            environment={},
            order_records=(),
        )

    reference = run(
        "B1",
        (
            {"epoch": 1, "aop_mae_deg": 20.0, "MRE_ALL": 20.0},
            {"epoch": 194, "aop_mae_deg": 10.0, "MRE_ALL": 10.0},
            {"epoch": 200, "aop_mae_deg": 11.0, "MRE_ALL": 11.0},
        ),
    )
    candidate = run(
        "B2",
        (
            {"epoch": 1, "aop_mae_deg": 12.0, "MRE_ALL": 12.0},
            {"epoch": 15, "aop_mae_deg": 9.7, "MRE_ALL": 9.0},
            {"epoch": 200, "aop_mae_deg": 13.0, "MRE_ALL": 13.0},
        ),
    )

    diagnostic = _effect_diagnostic(candidate, reference)

    assert diagnostic["classification"] == "mainly_convergence_speed"
    assert diagnostic["selected_best_benefit"]
    assert not diagnostic["sustained_endpoint_benefit"]
    assert diagnostic["epoch200_relation"] == "worse_or_equal_on_both"
    answer = _effect_answer("JS项（B2相对B1）", diagnostic)
    assert "主要表现为加快收敛" in answer
    assert "validation-selected best也更优" in answer
    assert "epoch 200端点的两项主指标均更差" in answer


def test_summary_rejects_undefined_valid_only_aop_when_valid_predictions_exist(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    path = run_root / "B1" / "seed_42" / "phase06_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["best_validation_metrics"]["dsnt"]["n_valid_aop"] > 0
    result["best_validation_metrics"]["dsnt"]["aop_mae_valid_deg"] = None
    _write_json(path, result)

    with pytest.raises(ValueError, match="undefined only when n_valid_aop is zero"):
        summarize_phase06(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_first_conclusion_distinguishes_best_from_reversed_epoch200_endpoint() -> None:
    def row(epoch: int, values: tuple[float, float]) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "aop_mae_deg": values[0],
            "MRE_ALL": values[1],
            "_payload": {
                "n_valid_aop": 100,
                "n_evaluable_aop": 100,
                "aop_invalid_prediction_count": 0,
            },
        }

    def run(variant: str, best: tuple[float, float], endpoint: tuple[float, float]) -> ValidatedRun:
        return ValidatedRun(
            variant=variant,
            config={},
            result={},
            history=(
                row(50, best),
                row(200, endpoint),
            ),
            environment={},
            order_records=(),
        )

    runs = {
        "B0": run("B0", (10.0, 10.0), (20.0, 20.0)),
        "B1": run("B1", (12.0, 12.0), (15.0, 15.0)),
        "B2": run("B2", (8.0, 8.0), (8.0, 8.0)),
    }
    catch = _catch_comparison(runs)
    diagnostic = {
        "classification": "no_clear_advantage",
        "earliest_epoch_matching_reference_best_on_both_primary_metrics": None,
    }
    first = _build_conclusions(
        runs,
        catch=catch,
        coordinate=diagnostic,
        js=diagnostic,
    )[0]["answer"]

    assert catch["B1"]["best_within_200_relation"] == "better_or_equal_on_both"
    assert catch["B1"]["epoch200_relation"] == "worse_or_equal_on_both"
    assert "预算内best" in first
    assert "epoch 200端点" in first
    assert "两个口径结论不同" in first
    assert _format_epoch(None) == "未达到"


def test_summary_rejects_incomplete_history(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    path = run_root / "B0" / "seed_42" / "train_log.csv"
    rows = _read_csv(path)
    _rewrite_csv(path, rows[:-1])
    with pytest.raises(ValueError, match="row count"):
        summarize_phase06(protocol_path=protocol_path, run_root=run_root, report_root=report_root)


def test_summary_rejects_inconsistent_overall_mre(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    path = run_root / "B1" / "seed_42" / "train_log.csv"
    rows = _read_csv(path)
    rows[49]["val_MRE_ALL"] = str(float(rows[49]["val_MRE_ALL"]) + 1.0)
    _rewrite_csv(path, rows)
    with pytest.raises(ValueError, match="MRE_ALL"):
        summarize_phase06(protocol_path=protocol_path, run_root=run_root, report_root=report_root)


def test_summary_rejects_wrong_best_tuple(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    path = run_root / "B2" / "seed_42" / "phase06_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["best_epoch"] = 81
    _write_json(path, result)
    with pytest.raises(ValueError, match="best epoch"):
        summarize_phase06(protocol_path=protocol_path, run_root=run_root, report_root=report_root)


def test_summary_rejects_cross_variant_training_order_change(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    path = run_root / "B2" / "seed_42" / "training_order.json"
    order = json.loads(path.read_text(encoding="utf-8"))
    order["records"][0]["filename_order_sha256"] = "f" * 64
    _write_json(path, order)
    with pytest.raises(ValueError, match="training order"):
        summarize_phase06(protocol_path=protocol_path, run_root=run_root, report_root=report_root)


def test_summary_rejects_cross_variant_environment_change(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    path = run_root / "B1" / "seed_42" / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["torch"] = "different"
    _write_json(path, environment)
    with pytest.raises(ValueError, match="environment"):
        summarize_phase06(protocol_path=protocol_path, run_root=run_root, report_root=report_root)


def test_summary_requires_exact_heatmap_unet_parameter_count(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    run_dir = run_root / "B0" / "seed_42"
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["trainable_parameters"] = 484170
    _write_json(config_path, config)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result_path = run_dir / "phase06_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["model"]["trainable_parameters"] = 484170
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="parameter count"):
        summarize_phase06(protocol_path=protocol_path, run_root=run_root, report_root=report_root)


def test_summary_rejects_forbidden_split_derived_key(
    phase06_matrix: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase06_matrix
    path = run_root / "B0" / "seed_42" / "phase06_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["testing_metrics"] = {"MRE_ALL": 1.0}
    _write_json(path, result)
    with pytest.raises(PermissionError, match="forbidden split-derived"):
        summarize_phase06(protocol_path=protocol_path, run_root=run_root, report_root=report_root)
