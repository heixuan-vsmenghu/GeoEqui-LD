from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.summarize_phase05 import CONFIRMATION_SEEDS, RUN_MATRIX, summarize_phase05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(variant: str, seed: int, epoch: int, *, decoder: str = "dsnt") -> dict[str, Any]:
    seed_offset = (seed - 42) * 0.5
    if variant == "B0":
        overall = 160.0 - epoch * 0.2 + seed_offset
        aop = 65.0 - epoch * 0.8 + seed_offset
    elif variant == "B1":
        overall = 52.0 - epoch * 0.5 + seed_offset
        aop = 31.0 - epoch * 0.5 + seed_offset
    else:
        overall = 38.0 - epoch * 0.5 + seed_offset
        aop = 27.0 - epoch * 0.6 + seed_offset
    if decoder == "argmax":
        overall += 5.0
        aop += 4.0
    heatmap = 1.0
    coordinate = 0.1 if variant in {"B1", "B2"} else 0.0
    distribution = 0.2 if variant == "B2" else 0.0
    return {
        "total_loss": heatmap + 10.0 * coordinate + distribution,
        "heatmap_mse": heatmap,
        "coordinate_smooth_l1": coordinate,
        "distribution_js": distribution,
        "MRE_PS1": overall - 1.0,
        "MRE_PS2": overall,
        "MRE_FH1": overall + 1.0,
        "MRE_ALL": overall,
        "n_samples": 100,
        "decoder": decoder,
        "n_valid_aop": 100,
        "n_evaluable_aop": 100,
        "aop_invalid_prediction_count": 0,
        "aop_mae_valid_deg": aop,
        "aop_mae_deg": aop,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_run(
    run_root: Path,
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    commit: str,
    variant: str,
    seed: int,
) -> None:
    run_dir = run_root / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    training = dict(protocol["training"])
    training["seed"] = seed
    weights = protocol["variants"][variant]["weights"]
    training.update(
        {
            "heatmap_loss_weight": float(weights[0]),
            "coordinate_loss_weight": float(weights[1]),
            "distribution_loss_weight": float(weights[2]),
        }
    )
    initialization = hashlib.sha256(f"initialization-{seed}".encode()).hexdigest()
    data = {
        "train": {
            "sample_count": 300,
            "labels_sha256": "1" * 64,
            "aggregate_sha256": "2" * 64,
            "source_columns": {"PS1": "PS1", "PS2": "PS2", "FH1": "FH1"},
        },
        "validation": {
            "sample_count": 100,
            "labels_sha256": "3" * 64,
            "aggregate_sha256": "4" * 64,
            "source_columns": {
                "PS1": "PS1",
                "PS2": "PS2",
                "FH1": "AOP Tangency",
            },
        },
    }
    model = {
        "class": "HeatmapUNet",
        "trainable_parameters": 484171,
        "initialization_sha256": initialization,
    }
    provenance = {
        "protocol_sha256": protocol_sha256,
        "git_commit": commit,
        "git_dirty": False,
    }
    config = {
        "schema_version": 1,
        "phase": "phase0.5-supervised-ablation",
        "variant": variant,
        "variant_description": protocol["variants"][variant]["description"],
        "training": training,
        "selection": {
            "split": "validation",
            "common_decoder": "dsnt",
            "checkpoint_selection": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
        },
        "testing_frozen": True,
        "data": data,
        "model": model,
        "provenance": provenance,
    }
    _write_json(run_dir / "config.json", config)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    fieldnames = [
        "epoch",
        "train_time_sec",
        "validation_time_sec",
        "epoch_time_sec",
        "train_total_loss",
        "train_heatmap_mse",
        "train_coordinate_smooth_l1",
        "train_distribution_js",
        "train_batches",
        "val_total_loss",
        "val_heatmap_mse",
        "val_coordinate_smooth_l1",
        "val_distribution_js",
        "val_MRE_PS1",
        "val_MRE_PS2",
        "val_MRE_FH1",
        "val_MRE_ALL",
        "val_n_samples",
        "val_decoder",
        "val_n_valid_aop",
        "val_n_evaluable_aop",
        "val_aop_invalid_prediction_count",
        "val_aop_mae_valid_deg",
        "val_aop_mae_deg",
    ]
    with (run_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, 21):
            metric = _metrics(variant, seed, epoch)
            heatmap = 1.0
            coordinate = 0.1 if float(weights[1]) > 0 else 0.0
            distribution = 0.2 if float(weights[2]) > 0 else 0.0
            total = (
                float(weights[0]) * heatmap
                + float(weights[1]) * coordinate
                + float(weights[2]) * distribution
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_time_sec": 1.0,
                    "validation_time_sec": 0.2,
                    "epoch_time_sec": 1.2,
                    "train_batches": 300,
                    "train_total_loss": total,
                    "train_heatmap_mse": heatmap,
                    "train_coordinate_smooth_l1": coordinate,
                    "train_distribution_js": distribution,
                    "val_total_loss": total,
                    "val_heatmap_mse": heatmap,
                    "val_coordinate_smooth_l1": coordinate,
                    "val_distribution_js": distribution,
                    **{f"val_{name}": value for name, value in metric.items()},
                }
            )

    best = _metrics(variant, seed, 20)
    decoders = {"dsnt": best}
    if variant == "B0":
        decoders["argmax"] = _metrics(variant, seed, 20, decoder="argmax")
    (run_dir / "best.pt").write_bytes(f"best-{variant}-{seed}".encode())
    (run_dir / "last.pt").write_bytes(f"last-{variant}-{seed}".encode())
    resources = {
        "training_runtime_sec": 25.0,
        "evaluation_runtime_sec": 1.0,
        "peak_gpu_allocated_mb": 100.0,
        "peak_gpu_reserved_mb": 120.0,
    }
    result = {
        "status": "completed",
        "phase": "phase0.5-supervised-ablation",
        "variant": variant,
        "seed": seed,
        "selection_split": "validation",
        "selection_decoder": "dsnt",
        "testing_frozen": True,
        "best_epoch": 20,
        "best_validation_metrics": decoders,
        "last_validation_metrics": best,
        "resources": resources,
        "model": model,
        "provenance": {
            **provenance,
            "best_checkpoint_sha256": _sha256(run_dir / "best.pt"),
        },
    }
    _write_json(run_dir / "phase05_result.json", result)
    _write_json(
        run_dir / "metrics.json",
        {
            "status": "completed",
            "selection_split": "validation",
            "checkpoint_metric": "aop_mae_deg",
            "selection_tiebreak": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
            "best_epoch": 20,
            "best_value": best["aop_mae_deg"],
            "best_validation_metrics": best,
            "last_validation_metrics": best,
            "best_checkpoint": "best.pt",
            "last_checkpoint": "last.pt",
        },
    )


@pytest.fixture
def phase05_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository_root = Path(__file__).resolve().parents[1]
    source_protocol = repository_root / "configs" / "phase05_ablation.yaml"
    protocol_path = tmp_path / "phase05_ablation.yaml"
    protocol_path.write_text(source_protocol.read_text(encoding="utf-8"), encoding="utf-8")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = _sha256(protocol_path)
    commit = "a" * 40
    run_root = tmp_path / "runs"
    for variant, seed in RUN_MATRIX:
        _write_run(
            run_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            commit=commit,
            variant=variant,
            seed=seed,
        )
    selection = {
        "schema_version": 1,
        "phase": "phase0.5-supervised-ablation",
        "testing_frozen": True,
        "git_commit": commit,
        "protocol_sha256": protocol_sha256,
        "screening_seed": 42,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "rule": ["aop_mae_deg", "MRE_ALL", "simpler_objective"],
        "selected_variants": ["B2", "B1"],
        "input_result_sha256": {
            variant: _sha256(run_root / variant / "seed_42" / "phase05_result.json")
            for variant in ("B0", "B1", "B2")
        },
    }
    _write_json(run_root / "selection.json", selection)
    return protocol_path, run_root, tmp_path / "reports"


def test_summary_validates_nine_runs_and_writes_sanitized_reports(
    phase05_fixture: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    aggregate = summarize_phase05(
        protocol_path=protocol_path,
        run_root=run_root,
        report_root=report_root,
    )

    assert aggregate["integrity"]["validated_run_count"] == 9
    assert aggregate["confirmation"]["paired_delta_B2_minus_B1"]["MRE_ALL"][
        "mean"
    ] == pytest.approx(-14.0)
    assert (report_root / "PHASE05_SUMMARY.md").is_file()
    assert (report_root / "SUPERVISED_ABLATION.md").is_file()
    assert (report_root / "curves" / "seed42_validation_metrics.png").is_file()
    assert (report_root / "curves" / "confirmation_validation_metrics.png").is_file()

    public_json = (report_root / "aggregate_results.json").read_text(encoding="utf-8")
    assert "protocol_sha256" not in public_json
    assert "initialization_sha256" not in public_json
    assert "best_checkpoint_sha256" not in public_json
    assert "git_commit" not in public_json
    assert "total_loss" not in public_json
    assert str(tmp_path) not in public_json
    assert "1" * 64 not in public_json


def test_summary_rejects_different_initialization_for_same_seed(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    run_dir = run_root / "B2" / "seed_44"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "phase05_result.json").read_text(encoding="utf-8"))
    changed = "f" * 64
    config["model"]["initialization_sha256"] = changed
    result["model"]["initialization_sha256"] = changed
    _write_json(run_dir / "config.json", config)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(run_dir / "phase05_result.json", result)

    with pytest.raises(ValueError, match="initialization"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_summary_rejects_checkpoint_digest_mismatch(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    (run_root / "B1" / "seed_43" / "best.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="best checkpoint digest"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_summary_rejects_inconsistent_weighted_total_loss(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    log_path = run_root / "B2" / "seed_43" / "train_log.csv"
    with log_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["val_total_loss"] = "999"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="inconsistent val total loss"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_summary_rejects_reused_initialization_across_different_seeds(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    source_config = json.loads(
        (run_root / "B1" / "seed_43" / "config.json").read_text(encoding="utf-8")
    )
    reused = source_config["model"]["initialization_sha256"]
    for variant in ("B1", "B2"):
        run_dir = run_root / variant / "seed_44"
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "phase05_result.json").read_text(encoding="utf-8"))
        config["model"]["initialization_sha256"] = reused
        result["model"]["initialization_sha256"] = reused
        _write_json(run_dir / "config.json", config)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        _write_json(run_dir / "phase05_result.json", result)

    with pytest.raises(ValueError, match="Different registered seeds"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_summary_rejects_inconsistent_overall_mre(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    result_path = run_root / "B1" / "seed_43" / "phase05_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["best_validation_metrics"]["dsnt"]["MRE_ALL"] += 10.0
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="MRE_ALL"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_summary_rejects_forbidden_split_derived_key(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    metrics_path = run_root / "B1" / "seed_43" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["best_validation_metrics"]["testingMetrics"] = {"MRE_ALL": 1.0}
    _write_json(metrics_path, metrics)

    with pytest.raises(ValueError, match="metrics_best fields mismatch"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_summary_rejects_unregistered_history_column(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    log_path = run_root / "B1" / "seed_43" / "train_log.csv"
    with log_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["testing_loss"] = "0"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="exact registered CSV column set"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )


def test_summary_rejects_changed_aop_anatomical_roles(
    phase05_fixture: tuple[Path, Path, Path],
) -> None:
    protocol_path, run_root, report_root = phase05_fixture
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["training"]["aop_vertex_index"] = 1
    protocol["training"]["aop_pubic_axis_other_index"] = 0
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.aop_vertex_index"):
        summarize_phase05(
            protocol_path=protocol_path,
            run_root=run_root,
            report_root=report_root,
        )
