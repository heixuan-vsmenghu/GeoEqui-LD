#!/usr/bin/env python
# ruff: noqa: E501
"""Validate the fixed Phase 1A artifacts and publish a sanitized summary.

The summarizer is deliberately allowlist-only: it opens the exact diagnostic,
gate, formal-run, and reference files registered below. It never discovers
siblings under ``runs/`` and it rejects split-derived keys for testing. Public
outputs contain aggregate train/validation evidence only; paths, hashes,
timestamps, environment details, losses, checkpoints, and per-sample data are
not copied into the report directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RUN_ROOT = REPOSITORY_ROOT / "runs" / "phase1a"
CANONICAL_REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase1a"
CANONICAL_PHASE06_AGGREGATE = REPOSITORY_ROOT / "reports" / "phase06" / "aggregate_results.json"

PHASE = "phase1a-supervised-hrnet-reference"
EXPERIMENT = "H1_shared_B2_seed42_20e"
SELECTION_ORDER = ("aop_mae_deg", "MRE_ALL", "earlier_epoch")
METRIC_NAMES = ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
HISTORY_COLUMNS = (
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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate fixed Phase 1A artifacts and publish validation-only reports."
    )
    parser.add_argument("--run-root", type=Path, default=CANONICAL_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=CANONICAL_REPORT_ROOT)
    parser.add_argument("--phase06-aggregate", type=Path, default=CANONICAL_PHASE06_AGGREGATE)
    return parser


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Required regular file is missing: {path.name}")


def _reject_forbidden_split_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
            if normalized == "testing_accessed":
                if item is not False:
                    raise PermissionError(f"{context} records testing access")
            elif normalized != "testing_frozen" and tokens & {"test", "testing"}:
                raise PermissionError(f"{context} contains forbidden split-derived key: {key}")
            _reject_forbidden_split_keys(item, context=context)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _reject_forbidden_split_keys(item, context=context)


def _read_json(path: Path) -> dict[str, Any]:
    _require_regular_file(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path.name}")
    _reject_forbidden_split_keys(loaded, context=path.name)
    return loaded


def _require_equal(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch: expected {expected!r}, got {actual!r}")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _finite(value: Any, *, context: str, nonnegative: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be finite")
    if nonnegative and converted < 0:
        raise ValueError(f"{context} must be non-negative")
    return converted


def _integer(value: Any, *, context: str, nonnegative: bool = False) -> int:
    converted = _finite(value, context=context)
    integer = int(converted)
    if converted != integer or (nonnegative and integer < 0):
        raise ValueError(f"{context} must be an integer")
    return integer


def _close(actual: Any, expected: Any, *, context: str, tolerance: float = 1.0e-6) -> None:
    actual_value = _finite(actual, context=context)
    expected_value = _finite(expected, context=context)
    if not math.isclose(actual_value, expected_value, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{context} differs between artifacts")


def _validate_metrics(value: Any, *, expected_samples: int, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a metric mapping")
    required = {
        "MRE_PS1",
        "MRE_PS2",
        "MRE_FH1",
        "MRE_ALL",
        "n_samples",
        "decoder",
        "n_valid_aop",
        "n_evaluable_aop",
        "aop_invalid_prediction_count",
        "aop_mae_valid_deg",
        "aop_mae_deg",
    }
    if not required.issubset(value):
        raise ValueError(f"{context} is missing required validation metrics")
    for name in METRIC_NAMES:
        _finite(value[name], context=f"{context}.{name}", nonnegative=True)
    _require_equal(value["decoder"], "dsnt", context=f"{context}.decoder")
    for name in ("n_samples", "n_valid_aop", "n_evaluable_aop", "aop_invalid_prediction_count"):
        _integer(value[name], context=f"{context}.{name}", nonnegative=True)
    _require_equal(value["n_samples"], expected_samples, context=f"{context}.n_samples")
    _require_equal(value["n_evaluable_aop"], expected_samples, context=f"{context}.n_evaluable")
    _require_equal(
        value["n_valid_aop"] + value["aop_invalid_prediction_count"],
        expected_samples,
        context=f"{context}.AoP counts",
    )
    if value["n_valid_aop"] == 0:
        valid_only = value["aop_mae_valid_deg"]
        if valid_only is not None and not math.isnan(float(valid_only)):
            raise ValueError(f"{context}.aop_mae_valid_deg must be undefined with zero valid AoP")
    else:
        _finite(value["aop_mae_valid_deg"], context=f"{context}.valid AoP", nonnegative=True)
    expected_mre = sum(float(value[name]) for name in METRIC_NAMES[:3]) / 3.0
    _close(value["MRE_ALL"], expected_mre, context=f"{context}.MRE_ALL", tolerance=1.0e-5)
    return value


def _history_metrics(row: Mapping[str, str], *, context: str) -> dict[str, Any]:
    return _validate_metrics(
        {
            "MRE_PS1": _finite(row["val_MRE_PS1"], context=f"{context}.MRE_PS1"),
            "MRE_PS2": _finite(row["val_MRE_PS2"], context=f"{context}.MRE_PS2"),
            "MRE_FH1": _finite(row["val_MRE_FH1"], context=f"{context}.MRE_FH1"),
            "MRE_ALL": _finite(row["val_MRE_ALL"], context=f"{context}.MRE_ALL"),
            "n_samples": _integer(row["val_n_samples"], context=f"{context}.n_samples"),
            "decoder": row["val_decoder"],
            "n_valid_aop": _integer(row["val_n_valid_aop"], context=f"{context}.n_valid"),
            "n_evaluable_aop": _integer(
                row["val_n_evaluable_aop"], context=f"{context}.n_evaluable"
            ),
            "aop_invalid_prediction_count": _integer(
                row["val_aop_invalid_prediction_count"], context=f"{context}.n_invalid"
            ),
            "aop_mae_valid_deg": row["val_aop_mae_valid_deg"],
            "aop_mae_deg": _finite(row["val_aop_mae_deg"], context=f"{context}.AoP"),
        },
        expected_samples=100,
        context=context,
    )


def _read_history(path: Path) -> tuple[dict[str, Any], ...]:
    _require_regular_file(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_equal(tuple(reader.fieldnames or ()), HISTORY_COLUMNS, context="history columns")
        raw_rows = list(reader)
    _require_equal(len(raw_rows), 20, context="history row count")
    rows: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(raw_rows, start=1):
        epoch = _integer(row["epoch"], context="history epoch")
        _require_equal(epoch, expected_epoch, context="history epoch sequence")
        for name in HISTORY_COLUMNS[1:16]:
            if name == "val_decoder":
                continue
            _finite(row[name], context=f"history.epoch{epoch}.{name}", nonnegative=True)
        _require_equal(
            _integer(row["train_batches"], context=f"history.epoch{epoch}.train_batches"),
            300,
            context=f"history.epoch{epoch}.train_batches",
        )
        metrics = _history_metrics(row, context=f"history.epoch{epoch}")
        rows.append({"epoch": epoch, **{name: float(metrics[name]) for name in METRIC_NAMES}})
    return tuple(rows)


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    return float(row["aop_mae_deg"]), float(row["MRE_ALL"]), int(row["epoch"])


def _compare_metrics(actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str) -> None:
    for name in METRIC_NAMES:
        _close(actual[name], expected[name], context=f"{context}.{name}")
    for name in ("n_samples", "n_valid_aop", "n_evaluable_aop", "aop_invalid_prediction_count"):
        if name in actual and name in expected:
            _require_equal(actual[name], expected[name], context=f"{context}.{name}")
    if "decoder" in actual and "decoder" in expected:
        _require_equal(actual["decoder"], expected["decoder"], context=f"{context}.decoder")


def _json_equivalent(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_equivalent(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_equivalent(item) for item in value]
    return value


def _validate_checkpoint(
    path: Path,
    *,
    expected_epoch: int,
    expected_config: Mapping[str, Any],
    expected_metrics: Mapping[str, Any],
) -> int:
    _require_regular_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path.name} checkpoint must be a mapping")
    _require_exact_keys(
        checkpoint,
        {
            "format_version",
            "epoch",
            "seed",
            "model_state_dict",
            "optimizer_state_dict",
            "config",
            "metrics",
            "extra",
        },
        context=f"{path.name} checkpoint",
    )
    _require_equal(checkpoint["epoch"], expected_epoch, context=f"{path.name} epoch")
    _require_equal(checkpoint["seed"], 42, context=f"{path.name} seed")
    _reject_forbidden_split_keys(checkpoint["config"], context=f"{path.name}.config")
    _require_equal(
        _json_equivalent(checkpoint["config"]),
        _json_equivalent(expected_config),
        context=f"{path.name} config",
    )
    checkpoint_metrics = _validate_metrics(
        checkpoint["metrics"], expected_samples=100, context=f"{path.name}.metrics"
    )
    _compare_metrics(checkpoint_metrics, expected_metrics, context=f"{path.name}.metrics")
    state_dict = checkpoint["model_state_dict"]
    if not state_dict or not checkpoint["optimizer_state_dict"]:
        raise ValueError(f"{path.name} has an empty training state")
    buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    trainable_parameters = sum(
        int(tensor.numel())
        for name, tensor in state_dict.items()
        if not str(name).endswith(buffer_suffixes)
    )
    _require_equal(
        trainable_parameters, 29_318_355, context=f"{path.name} trainable parameter count"
    )
    return trainable_parameters


def _validate_formal_run(run_root: Path) -> dict[str, Any]:
    run_dir = run_root / EXPERIMENT
    config = _read_json(run_dir / "config.json")
    result = _read_json(run_dir / "formal_result.json")
    metrics = _read_json(run_dir / "metrics.json")
    history = _read_history(run_dir / "train_log.csv")

    _require_equal(config.get("phase"), "phase1a", context="formal config phase")
    _require_equal(config.get("experiment_name"), EXPERIMENT, context="experiment name")
    _require_equal(config.get("testing_frozen"), True, context="testing freeze")
    training = config.get("training")
    model = config.get("model")
    optimizer = config.get("optimizer")
    data = config.get("data")
    runtime = config.get("runtime")
    if not all(isinstance(item, dict) for item in (training, model, optimizer, data, runtime)):
        raise ValueError("formal config mappings are incomplete")
    assert isinstance(training, dict)
    assert isinstance(model, dict)
    assert isinstance(optimizer, dict)
    assert isinstance(data, dict)
    assert isinstance(runtime, dict)
    expected_training = {
        "seed": 42,
        "batch_size": 1,
        "epochs": 20,
        "learning_rate": 0.001,
        "heatmap_loss_weight": 1.0,
        "coordinate_loss_weight": 10.0,
        "distribution_loss_weight": 1.0,
        "dsnt_temperature": 0.05,
        "input_size_hw": [512, 512],
        "heatmap_size_hw": [256, 256],
        "keypoint_order": ["PS1", "PS2", "FH1"],
    }
    for key, expected in expected_training.items():
        _require_equal(training.get(key), expected, context=f"formal training.{key}")
    expected_model = {
        "backbone": "hrnet_w32",
        "timm_version": "1.0.28",
        "pretrained": False,
        "in_channels": 1,
        "out_channels": 3,
        "feature_location": "",
        "out_indices": [1],
        "feature_channels": 32,
        "feature_reduction": 4,
        "decoder_channels": [32, 16],
        "decoder_normalization": "BatchNorm2d",
        "decoder_activation": "GELU",
        "interpolation_mode": "bilinear",
        "class": "HRNetW32SharedHeatmap",
    }
    for key, expected in expected_model.items():
        _require_equal(model.get(key), expected, context=f"formal model.{key}")
    _require_equal(optimizer.get("class"), "Adam", context="optimizer class")
    _require_equal(optimizer.get("foreach"), False, context="optimizer foreach")
    _require_equal(data.get("train_count"), 300, context="formal train count")
    _require_equal(data.get("validation_count"), 100, context="formal validation count")
    _require_equal(data.get("paths_embedded"), False, context="formal path policy")
    _require_equal(runtime.get("allocated_seconds"), 7200.0, context="formal allocation")
    _require_equal(runtime.get("formal_cap_seconds"), 7200.0, context="formal cap")

    _require_equal(result.get("status"), "completed", context="formal status")
    _require_equal(result.get("partial"), False, context="formal partial flag")
    _require_equal(result.get("epochs_completed"), 20, context="formal completed epochs")
    _require_equal(result.get("epochs_requested"), 20, context="formal requested epochs")
    _require_equal(result.get("selection_split"), "validation", context="selection split")
    _require_equal(tuple(result.get("selection_order", ())), SELECTION_ORDER, context="selection")
    _require_equal(metrics.get("status"), "completed", context="metrics status")
    _require_equal(metrics.get("epochs_completed"), 20, context="metrics epochs")
    _require_equal(metrics.get("selection_split"), "validation", context="metrics split")
    _require_equal(
        tuple(metrics.get("selection_tiebreak", ())), SELECTION_ORDER, context="metrics selection"
    )
    result_runtime = _finite(
        result.get("runtime_elapsed_sec"), context="formal elapsed", nonnegative=True
    )
    result_allocation = _finite(
        result.get("runtime_allocated_sec"), context="formal allocated", nonnegative=True
    )
    metrics_runtime = _finite(
        metrics.get("runtime_elapsed_sec"), context="metrics elapsed", nonnegative=True
    )
    metrics_limit = _finite(
        metrics.get("runtime_limit_sec"), context="metrics runtime limit", nonnegative=True
    )
    _close(result_runtime, metrics_runtime, context="formal runtime records")
    _require_equal(result_allocation, 7200.0, context="result runtime allocation")
    _require_equal(metrics_limit, 7200.0, context="metrics runtime limit")
    if result_runtime > result_allocation:
        raise ValueError("formal run exceeded its two-hour cap")
    _require_equal(
        Path(str(metrics.get("best_checkpoint"))).resolve(),
        (run_dir / "best.pt").resolve(),
        context="registered best checkpoint",
    )
    _require_equal(
        Path(str(metrics.get("last_checkpoint"))).resolve(),
        (run_dir / "last.pt").resolve(),
        context="registered last checkpoint",
    )

    best_row = min(history, key=_selection_key)
    last_row = history[-1]
    _require_equal(result.get("best_epoch"), best_row["epoch"], context="result best epoch")
    _require_equal(metrics.get("best_epoch"), best_row["epoch"], context="metrics best epoch")
    best_metrics = _validate_metrics(
        metrics.get("best_validation_metrics"), expected_samples=100, context="best metrics"
    )
    last_metrics = _validate_metrics(
        metrics.get("last_validation_metrics"), expected_samples=100, context="last metrics"
    )
    result_best = _validate_metrics(
        result.get("best_validation_metrics"), expected_samples=100, context="formal best metrics"
    )
    _compare_metrics(best_metrics, best_row, context="best history tuple")
    _compare_metrics(last_metrics, last_row, context="last history tuple")
    _compare_metrics(result_best, best_metrics, context="formal/metrics best")
    _close(metrics.get("best_value"), best_row["aop_mae_deg"], context="best value")

    best_parameter_count = _validate_checkpoint(
        run_dir / "best.pt",
        expected_epoch=best_row["epoch"],
        expected_config=config,
        expected_metrics=best_metrics,
    )
    last_parameter_count = _validate_checkpoint(
        run_dir / "last.pt",
        expected_epoch=20,
        expected_config=config,
        expected_metrics=last_metrics,
    )
    _require_equal(
        best_parameter_count, last_parameter_count, context="best/last parameter count"
    )
    return {
        "config": config,
        "result": result,
        "metrics": metrics,
        "history": history,
        "best": {"epoch": best_row["epoch"], **{name: float(best_row[name]) for name in METRIC_NAMES}},
        "last": {"epoch": 20, **{name: float(last_row[name]) for name in METRIC_NAMES}},
        "trainable_parameters": best_parameter_count,
    }


def _validate_b3(run_root: Path) -> dict[str, Any]:
    value = _read_json(run_root / "B3_structure_probe" / "b3_result.json")
    _require_equal(value.get("gate_id"), "B3", context="B3 gate id")
    _require_equal(value.get("gate"), "PASS", context="B3 gate")
    _require_equal(value.get("status"), "completed", context="B3 status")
    checks = value.get("checks")
    if not isinstance(checks, dict) or not checks or any(item is not True for item in checks.values()):
        raise ValueError("B3 structural checks are incomplete")
    feature = value.get("feature_contract")
    stage4 = value.get("stage4")
    timing = value.get("timing_seconds")
    memory = value.get("cuda_memory_mb")
    if not all(isinstance(item, dict) for item in (feature, stage4, timing, memory)):
        raise ValueError("B3 structural evidence is incomplete")
    assert isinstance(feature, dict)
    assert isinstance(stage4, dict)
    assert isinstance(timing, dict)
    assert isinstance(memory, dict)
    _require_equal(feature.get("timm_version"), "1.0.28", context="B3 timm")
    _require_equal(feature.get("backbone_name"), "hrnet_w32", context="B3 backbone")
    _require_equal(feature.get("feature_location"), "", context="B3 feature location")
    _require_equal(feature.get("out_indices"), [1], context="B3 out indices")
    _require_equal(feature.get("channels"), [32], context="B3 channels")
    _require_equal(feature.get("reductions"), [4], context="B3 reduction")
    _require_equal(
        stage4.get("output_shapes"),
        [[1, 32, 128, 128], [1, 64, 64, 64], [1, 128, 32, 32], [1, 256, 16, 16]],
        context="B3 stage4 shapes",
    )
    gradients = stage4.get("input_gradient_l1")
    if not isinstance(gradients, list) or len(gradients) != 4:
        raise ValueError("B3 stage4 gradient evidence must contain four scales")
    if any(_finite(item, context="B3 stage4 gradient", nonnegative=True) <= 0 for item in gradients):
        raise ValueError("B3 did not propagate gradients through every stage4 scale")
    warmed_step = _finite(timing.get("warmed_full_step"), context="B3 warmed step", nonnegative=True)
    first_step = _finite(timing.get("first_adam_step"), context="B3 first step", nonnegative=True)
    total_time = _finite(timing.get("total"), context="B3 total time", nonnegative=True)
    allocated_time = _finite(timing.get("allocated"), context="B3 allocated time", nonnegative=True)
    if total_time > allocated_time:
        raise ValueError("B3 exceeded its allocated probe time")
    allocated_memory = _finite(
        memory.get("peak_allocated"), context="B3 allocated memory", nonnegative=True
    )
    reserved = _finite(memory.get("peak_reserved"), context="B3 reserved memory", nonnegative=True)
    return {
        "raw": value,
        "first_step_sec": first_step,
        "warmed_step_sec": warmed_step,
        "total_time_sec": total_time,
        "allocated_time_sec": allocated_time,
        "peak_allocated_mb": allocated_memory,
        "peak_reserved_mb": reserved,
    }


def _validate_tiny_gate(run_root: Path, name: str) -> dict[str, Any]:
    value = _read_json(run_root / name / "tiny_gate_result.json")
    _require_equal(value.get("gate_id"), name, context=f"{name} id")
    if name == "A4_unet_B0":
        if value.get("gate") not in {"PASS", "NOT_APPLICABLE"}:
            raise ValueError("A4 legacy/new diagnostic status is invalid")
        if value.get("gate") == "NOT_APPLICABLE":
            _require_equal(
                value.get("diagnostic_completion"), "completed", context="A4 completion"
            )
            _require_equal(value.get("learning_outcome"), "not_learned", context="A4 outcome")
    else:
        _require_equal(value.get("gate"), "PASS", context=f"{name} strict gate")
    _require_equal(value.get("status"), "completed", context=f"{name} status")
    expected_steps = 1000 if name == "A4_unet_B0" else 500
    _require_equal(value.get("steps_completed"), expected_steps, context=f"{name} steps")
    _require_equal(value.get("max_steps"), expected_steps, context=f"{name} max steps")
    _require_equal(value.get("within_total_allocation"), True, context=f"{name} allocation")
    _require_equal(value.get("augmentation"), "disabled", context=f"{name} augmentation")
    _require_equal(value.get("batch_size"), 1, context=f"{name} batch size")
    _require_equal(value.get("precision"), "float32", context=f"{name} precision")
    raw_evaluation = value.get("eval_mode")
    if not isinstance(raw_evaluation, dict):
        raise ValueError(f"{name} eval metrics are missing")
    evaluation = _validate_metrics(
        {
            **raw_evaluation,
            "decoder": "dsnt",
            "aop_mae_valid_deg": raw_evaluation.get("aop_mae_deg"),
        },
        expected_samples=4,
        context=f"{name}.eval",
    )
    _require_equal(evaluation.get("coordinate_error_count"), 0, context=f"{name} coordinate")
    _require_equal(evaluation.get("nonfinite_count"), 0, context=f"{name} finite")
    visualization = value.get("visualization")
    if not isinstance(visualization, dict):
        raise ValueError(f"{name} visualization audit is missing")
    _require_equal(
        visualization.get("programmatic_check_passed"), True, context=f"{name} visual checks"
    )
    if name == "B4_hrnet_B2":
        _require_equal(visualization.get("manual_review_status"), "passed", context="B4 review")
        if float(evaluation["MRE_ALL"]) > 5.0 or evaluation["n_valid_aop"] != 4:
            raise ValueError("B4 did not meet the predeclared tiny-overfit gate")
    return {"raw": value, "eval": evaluation}


def _validate_diagnostics(report_root: Path) -> dict[str, Any]:
    endpoints = _read_json(report_root / "B0_CHECKPOINT_DIAGNOSTICS.json")
    sanity = _read_json(report_root / "HEATMAP_DECODE_SANITY.json")
    mean = _read_json(report_root / "TRAIN_MEAN_BASELINE.json")
    _require_equal(endpoints.get("status"), "completed", context="endpoint diagnostics status")
    _require_equal(endpoints.get("checkpoint_count"), 6, context="diagnostic checkpoint count")
    checkpoint_rows = endpoints.get("checkpoints")
    if not isinstance(checkpoint_rows, list):
        raise ValueError("diagnostic checkpoints must be a list")
    by_id = {row.get("checkpoint_id"): row for row in checkpoint_rows if isinstance(row, dict)}
    expected_ids = {f"{variant}_{endpoint}" for variant in ("B0", "B1", "B2") for endpoint in ("best", "last")}
    _require_equal(set(by_id), expected_ids, context="diagnostic checkpoint ids")
    for checkpoint_id, row in by_id.items():
        decoders = row.get("decoder_metrics")
        if not isinstance(decoders, dict):
            raise ValueError(f"{checkpoint_id} decoder metrics are missing")
        _validate_metrics(
            {**decoders.get("dsnt", {}), "decoder": "dsnt"},
            expected_samples=100,
            context=f"{checkpoint_id}.dsnt",
        )
        _validate_metrics(
            {**decoders.get("argmax", {}), "decoder": "dsnt"},
            expected_samples=100,
            context=f"{checkpoint_id}.argmax",
        )
    b0_best = by_id["B0_best"]
    b0_last = by_id["B0_last"]
    _require_equal(b0_best.get("epoch"), 120, context="B0 best epoch")
    _require_equal(b0_last.get("epoch"), 200, context="B0 last epoch")
    last_dsnt = b0_last["decoder_metrics"]["dsnt"]
    _require_equal(last_dsnt["n_valid_aop"], 0, context="B0 last valid AoP")
    raw_std = b0_last["heatmap_diagnostics"]["overall"]["raw_std"]["mean"]
    _close(raw_std, 0.0, context="B0 last raw heatmap std")

    _require_equal(sanity.get("status"), "synthetic_only", context="sanity scope")
    _require_equal(sanity.get("synthetic_geometry_valid"), True, context="sanity geometry")
    cases = sanity.get("cases")
    if not isinstance(cases, list):
        raise ValueError("synthetic cases must be a list")
    by_case = {case.get("case_id"): case for case in cases if isinstance(case, dict)}
    expected_cases = {
        "gaussian_argmax",
        "gaussian_dsnt_t1",
        "gaussian_dsnt_t0.05",
        "gaussian_amplitude_0.1_dsnt_t0.05",
        "gaussian_amplitude_0.01_dsnt_t0.05",
        "zero_heatmaps_dsnt_t0.05",
        "flat_heatmaps_dsnt_t0.05",
    }
    _require_equal(set(by_case), expected_cases, context="synthetic case ids")
    if float(by_case["gaussian_argmax"]["MRE_ALL"]) != 0.0:
        raise ValueError("Gaussian argmax sanity failed")
    if float(by_case["gaussian_dsnt_t0.05"]["MRE_ALL"]) >= 0.01:
        raise ValueError("temperature-0.05 Gaussian DSNT sanity failed")
    if float(by_case["gaussian_dsnt_t1"]["MRE_ALL"]) <= 100.0:
        raise ValueError("temperature-1 sanity no longer demonstrates diffuse decoding")
    for case_id in ("zero_heatmaps_dsnt_t0.05", "flat_heatmaps_dsnt_t0.05"):
        _require_equal(by_case[case_id]["aop_official_valid"], False, context=case_id)

    _require_equal(mean.get("status"), "completed", context="mean baseline status")
    _require_equal(mean.get("fit_split"), "train", context="mean baseline fit split")
    _require_equal(mean.get("evaluation_split"), "validation", context="mean baseline split")
    mean_metrics = _validate_metrics(
        {**mean.get("validation_metrics", {}), "decoder": "dsnt"},
        expected_samples=100,
        context="mean baseline",
    )
    return {
        "endpoints": endpoints,
        "by_id": by_id,
        "sanity": sanity,
        "by_case": by_case,
        "mean": mean,
        "mean_metrics": mean_metrics,
    }


def _validate_phase06_reference(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    _require_equal(value.get("phase"), "phase0.6-long-budget-fidelity", context="reference phase")
    _require_equal(value.get("selection_split"), "validation", context="reference split")
    _require_equal(value.get("testing_frozen"), True, context="reference freeze")
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Phase 0.6 reference runs are missing")
    matches = [run for run in runs if isinstance(run, dict) and run.get("variant") == "B2"]
    _require_equal(len(matches), 1, context="reference B2 count")
    best = matches[0].get("checkpoints", {}).get("best")
    if not isinstance(best, dict):
        raise ValueError("Phase 0.6 B2 best checkpoint is missing")
    _require_equal(best.get("epoch"), 15, context="reference B2 best epoch")
    metrics = best.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Phase 0.6 B2 best metrics are missing")
    _close(metrics.get("MRE_ALL"), 24.779436111450195, context="reference B2 MRE")
    _close(metrics.get("aop_mae_deg"), 8.513850212097168, context="reference B2 AoP")
    return {"epoch": 15, "MRE_ALL": float(metrics["MRE_ALL"]), "aop_mae_deg": float(metrics["aop_mae_deg"])}


def _validate_gpu_budget(run_root: Path) -> dict[str, Any]:
    value = _read_json(run_root / "gpu_budget.json")
    _require_equal(value.get("total_limit_seconds"), 10800.0, context="GPU budget limit")
    _require_equal(value.get("active_run"), None, context="GPU active run")
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise ValueError("GPU budget ledger runs are missing")
    expected = ["B3_structure_probe", "A4_unet_B0", "B4_hrnet_B2", EXPERIMENT]
    _require_equal([run.get("name") for run in runs], expected, context="GPU budget run order")
    elapsed = 0.0
    for run in runs:
        _require_equal(run.get("status"), "completed", context=f"{run.get('name')} budget status")
        elapsed += _finite(run.get("elapsed_seconds"), context="GPU elapsed", nonnegative=True)
    if elapsed > float(value["total_limit_seconds"]):
        raise ValueError("Phase 1A exceeded the total GPU budget")
    return {"total_limit_seconds": 10800.0, "elapsed_seconds": elapsed}


def _validate_independent_audit(run_root: Path) -> dict[str, Any]:
    value = _read_json(run_root / "independent_validation_audit.json")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "audit_name",
            "status",
            "read_only",
            "evaluation_split",
            "testing_accessed",
            "sample_count",
            "checkpoints",
            "gpu_elapsed_seconds_approx",
            "notes",
        },
        context="independent validation audit",
    )
    _require_equal(value["schema_version"], 1, context="audit schema")
    _require_equal(
        value["audit_name"], "H1_best_last_validation_recheck", context="audit name"
    )
    _require_equal(value["status"], "completed", context="audit status")
    _require_equal(value["read_only"], True, context="audit read-only flag")
    _require_equal(value["evaluation_split"], "validation", context="audit split")
    _require_equal(value["testing_accessed"], False, context="audit testing access")
    _require_equal(value["sample_count"], 100, context="audit sample count")
    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, dict):
        raise ValueError("audit checkpoint evidence is missing")
    _require_equal(set(checkpoints), {"best", "last"}, context="audit checkpoints")
    elapsed_parts = 0.0
    for name, epoch in (("best", 3), ("last", 20)):
        checkpoint = checkpoints[name]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"audit {name} evidence is invalid")
        _require_exact_keys(
            checkpoint,
            {"epoch", "saved_metric_max_abs_delta", "evaluation_elapsed_seconds_approx"},
            context=f"audit {name}",
        )
        _require_equal(checkpoint["epoch"], epoch, context=f"audit {name} epoch")
        _close(
            checkpoint["saved_metric_max_abs_delta"], 0.0, context=f"audit {name} delta"
        )
        elapsed_parts += _finite(
            checkpoint["evaluation_elapsed_seconds_approx"],
            context=f"audit {name} elapsed",
            nonnegative=True,
        )
    total = _finite(
        value["gpu_elapsed_seconds_approx"], context="audit GPU elapsed", nonnegative=True
    )
    _close(total, elapsed_parts, context="audit elapsed sum", tolerance=1.0e-3)
    return {"elapsed_seconds": total, "metrics_reproduced": True}


def _metric_snapshot(row: Mapping[str, Any]) -> dict[str, float | int]:
    return {"epoch": int(row["epoch"]), **{name: float(row[name]) for name in METRIC_NAMES}}


def _build_public_aggregate(validated: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = validated["diagnostics"]
    b3 = validated["b3"]
    a4 = validated["a4"]
    b4 = validated["b4"]
    formal = validated["formal"]
    reference = validated["reference"]
    budget = validated["budget"]
    independent_audit = validated["independent_audit"]
    b0_best = diagnostics["by_id"]["B0_best"]["decoder_metrics"]
    b0_last = diagnostics["by_id"]["B0_last"]["decoder_metrics"]
    mean_metrics = diagnostics["mean_metrics"]
    t005 = diagnostics["by_case"]["gaussian_dsnt_t0.05"]
    t1 = diagnostics["by_case"]["gaussian_dsnt_t1"]
    low_amplitude = diagnostics["by_case"]["gaussian_amplitude_0.1_dsnt_t0.05"]
    return {
        "schema_version": 1,
        "phase": PHASE,
        "scope": "supervised train/validation diagnostics and HRNet-W32 reference",
        "data_scope": {
            "train_samples": 300,
            "validation_samples": 100,
            "testing_frozen": True,
        },
        "integrity": {
            "fixed_input_allowlist": True,
            "formal_history_epochs_verified": 20,
            "best_selection_tuple_recomputed": list(SELECTION_ORDER),
            "best_and_last_checkpoints_verified": True,
            "independent_validation_metrics_reproduced": bool(
                independent_audit["metrics_reproduced"]
            ),
            "public_outputs_sanitized": True,
        },
        "b0_diagnostics": {
            "best_epoch": 120,
            "best_dsnt": {
                "MRE_ALL": float(b0_best["dsnt"]["MRE_ALL"]),
                "aop_mae_valid_deg": float(b0_best["dsnt"]["aop_mae_valid_deg"]),
                "aop_penalized_selection_score_deg": float(
                    b0_best["dsnt"]["aop_mae_deg"]
                ),
                "n_valid_aop": int(b0_best["dsnt"]["n_valid_aop"]),
                "n_evaluable_aop": int(b0_best["dsnt"]["n_evaluable_aop"]),
                "n_invalid_aop": int(b0_best["dsnt"]["aop_invalid_prediction_count"]),
            },
            "best_argmax": {
                "MRE_ALL": float(b0_best["argmax"]["MRE_ALL"]),
                "aop_mae_valid_deg": float(b0_best["argmax"]["aop_mae_valid_deg"]),
                "aop_penalized_selection_score_deg": float(
                    b0_best["argmax"]["aop_mae_deg"]
                ),
            },
            "last_epoch": 200,
            "last_dsnt": {
                "MRE_ALL": float(b0_last["dsnt"]["MRE_ALL"]),
                "aop_mae_valid_deg": None,
                "aop_penalized_selection_score_deg": float(
                    b0_last["dsnt"]["aop_mae_deg"]
                ),
                "n_valid_aop": int(b0_last["dsnt"]["n_valid_aop"]),
                "n_evaluable_aop": int(b0_last["dsnt"]["n_evaluable_aop"]),
                "n_invalid_aop": int(b0_last["dsnt"]["aop_invalid_prediction_count"]),
                "raw_heatmap_spatial_std": float(
                    diagnostics["by_id"]["B0_last"]["heatmap_diagnostics"]["overall"]["raw_std"]["mean"]
                ),
            },
            "train_mean_coordinate_reference": {
                "MRE_ALL": float(mean_metrics["MRE_ALL"]),
                "aop_mae_deg": float(mean_metrics["aop_mae_deg"]),
                "n_valid_aop": int(mean_metrics["n_valid_aop"]),
            },
            "synthetic_decode_checks": {
                "gaussian_argmax_MRE_ALL": 0.0,
                "gaussian_dsnt_t0.05_MRE_ALL": float(t005["MRE_ALL"]),
                "gaussian_dsnt_t1_MRE_ALL": float(t1["MRE_ALL"]),
                "amplitude_0.1_dsnt_t0.05_MRE_ALL": float(low_amplitude["MRE_ALL"]),
                "zero_and_flat_maps_produce_invalid_aop": True,
            },
            "interpretation": "Endpoint evidence is consistent with a diffuse-to-flat response, but missing transition checkpoints prevent a causal localization of the collapse.",
        },
        "gates": {
            "A4_unet_B0": {
                "execution_status": "completed",
                "learning_outcome": "not_learned",
                "steps": int(a4["raw"]["steps_completed"]),
                "MRE_PS1": float(a4["eval"]["MRE_PS1"]),
                "MRE_PS2": float(a4["eval"]["MRE_PS2"]),
                "MRE_FH1": float(a4["eval"]["MRE_FH1"]),
                "MRE_ALL": float(a4["eval"]["MRE_ALL"]),
                "n_valid_aop": int(a4["eval"]["n_valid_aop"]),
                "learned_all_three_keypoints": False,
            },
            "B3_structure_probe": {
                "gate": "PASS",
                "trainable_parameters": int(formal["trainable_parameters"]),
                "peak_allocated_mb": float(b3["peak_allocated_mb"]),
                "peak_reserved_mb": float(b3["peak_reserved_mb"]),
                "first_adam_step_sec": float(b3["first_step_sec"]),
                "warmed_full_step_sec": float(b3["warmed_step_sec"]),
                "probe_elapsed_sec": float(b3["total_time_sec"]),
                "probe_allocated_sec": float(b3["allocated_time_sec"]),
                "all_four_stage4_scales_have_gradients": True,
                "checkpoint_roundtrip": True,
            },
            "B4_hrnet_B2": {
                "gate": "PASS",
                "steps": int(b4["raw"]["steps_completed"]),
                "MRE_ALL": float(b4["eval"]["MRE_ALL"]),
                "aop_mae_deg": float(b4["eval"]["aop_mae_deg"]),
                "n_valid_aop": int(b4["eval"]["n_valid_aop"]),
                "manual_visual_review": "passed",
            },
        },
        "formal_run": {
            "experiment": EXPERIMENT,
            "status": "completed",
            "epochs_completed": 20,
            "seed": 42,
            "selection_split": "validation",
            "selection_order": list(SELECTION_ORDER),
            "best": _metric_snapshot(formal["best"]),
            "last": _metric_snapshot(formal["last"]),
            "all_validation_aop_valid_at_best_and_last": True,
        },
        "reference_only": {
            "phase06_unet_B2": {
                "best_epoch": int(reference["epoch"]),
                "MRE_ALL": float(reference["MRE_ALL"]),
                "aop_mae_deg": float(reference["aop_mae_deg"]),
                "comparison_boundary": "Different architecture; descriptive context only, not a causal architecture comparison.",
            }
        },
        "resources": {
            "gpu_budget_limit_minutes": float(budget["total_limit_seconds"]) / 60.0,
            "experiment_ledger_gpu_minutes": float(budget["elapsed_seconds"]) / 60.0,
            "independent_validation_audit_gpu_minutes_approx": float(
                independent_audit["elapsed_seconds"]
            )
            / 60.0,
            "total_audited_gpu_minutes_approx": (
                float(budget["elapsed_seconds"]) + float(independent_audit["elapsed_seconds"])
            )
            / 60.0,
        },
        "limitations": [
            "Single seed for the 20-epoch HRNet reference.",
            "Validation volatility is observed; batch-size-1 BatchNorm is a plausible risk, not a causally established explanation.",
            "The shared decoder is a supervised reference, not the planned PS/FH-decoupled model.",
            "No EMA teacher, pseudo-labels, unlabeled consistency objective, or semi-supervised claim.",
        ],
    }


def _sanitized_config() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "data_scope": {
            "allowed_splits": ["train", "validation"],
            "testing_frozen": True,
            "train_samples": 300,
            "validation_samples": 100,
        },
        "model": {
            "class": "HRNetW32SharedHeatmap",
            "backbone": "hrnet_w32",
            "pretrained": False,
            "in_channels": 1,
            "stage4_high_resolution_feature": [32, 128, 128],
            "decoder_channels": [32, 16, 3],
            "output_heatmap_hw": [256, 256],
            "trainable_parameters": 29_318_355,
        },
        "training": {
            "seed": 42,
            "epochs": 20,
            "batch_size": 1,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "optimizer_foreach": False,
            "precision": "float32",
            "supervision": "heatmap MSE + coordinate SmoothL1 + distribution JS",
            "dsnt_temperature": 0.05,
        },
        "selection": {
            "split": "validation",
            "order": list(SELECTION_ORDER),
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_curve(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for name, label in (
        ("MRE_PS1", "PS1"),
        ("MRE_PS2", "PS2"),
        ("MRE_FH1", "FH1"),
        ("MRE_ALL", "Overall"),
    ):
        axes[0].plot(epochs, [float(row[name]) for row in history], label=label, linewidth=1.8)
    axes[0].set_title("Validation landmark error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MRE (px)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        epochs,
        [float(row["aop_mae_deg"]) for row in history],
        color="#d95f02",
        linewidth=2.0,
    )
    axes[1].set_title("Validation AoP error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (deg)")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.set_xticks([1, 5, 10, 15, 20])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        clean = rendered.convert("RGB")
        clean.save(path, format="PNG", optimize=True)


def _write_hrnet_report(path: Path, aggregate: Mapping[str, Any]) -> None:
    b3 = aggregate["gates"]["B3_structure_probe"]
    b4 = aggregate["gates"]["B4_hrnet_B2"]
    formal = aggregate["formal_run"]
    best = formal["best"]
    last = formal["last"]
    resources = aggregate["resources"]
    text = f"""# Phase 1A：HRNet 监督参考实现

这一步先把老师资料里的 HRNet-W32 接到现有三关键点流程里，范围保持得比较窄：灰度输入、共享三通道热图头、纯监督 B2 损失。这里还没有做 PS/FH 解耦，也没有加入 EMA、伪标签或无标签一致性。

## 接入方式

- `timm==1.0.28`，`hrnet_w32`，`pretrained=False`，单通道输入；
- `feature_info` 核验为 `channels=(32,)`、`reduction=(4,)`，固定 `feature_location=''`、`out_indices=(1,)`；hook 路径为 `backbone.stage4.2`，取其四尺度融合后的高分辨率输出 `[B, 32, 128, 128]`，不是 stem；
- 共享解码头为 `3×3 Conv 32→32 + BN + GELU`、`3×3 Conv 32→16 + BN + GELU`、`1×1 Conv 16→3`，最后双线性插值到 `256×256`；
- 模型共有 **{b3['trainable_parameters']:,}** 个可训练参数；
- 优化器仍是 Adam，batch size 为 1，FP32；为控制 4 GB 显存峰值，Adam 使用 `foreach=False`，没有偷偷改输入尺寸或启用 AMP。

## B3 结构探针

B3 通过。512×512、batch 1 的完整 Adam 更新中，stage4 四个尺度都有非零梯度；backbone 与 decoder 均发生参数更新，train/eval 切换和 checkpoint 往返也一致。峰值 allocated / reserved 显存分别为 **{b3['peak_allocated_mb'] / 1024:.2f} / {b3['peak_reserved_mb'] / 1024:.2f} GiB**。第一次 Adam 完整更新约 **{b3['first_adam_step_sec']:.3f} s**，预热后的完整训练步约 **{b3['warmed_full_step_sec']:.3f} s**；探针实际用时 **{b3['probe_elapsed_sec']:.2f} s**，低于分配的 **{b3['probe_allocated_sec']:.0f} s**。

## B4 四样本门槛

HRNet 在固定 4 张训练样本上跑满 500 步，eval 模式得到 MRE_ALL **{b4['MRE_ALL']:.4f} px**、AoP MAE **{b4['aop_mae_deg']:.4f}°**，4/4 AoP 有效，没有非有限值或坐标换算错误。四张叠加图已逐张检查，关键点通道、目标圆圈和预测叉号没有发现可见错位，因此 B4 通过。

作为诊断对照，同样跑满预算的轻量 U-Net 纯 MSE 只把 PS1 学到约 3 px；PS2 与 FH1 分别约 184 px 和 84 px，整体 MRE 为 90.372 px。它的执行过程完整、数值也有限，但没有通过三点学习判据。

## 20 轮监督参考

H1 完整跑完 20 轮，checkpoint 只按 validation 的 `(AoP MAE, MRE_ALL, 较早 epoch)` 选择：

| checkpoint | epoch | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE | 有效 AoP |
|---|---:|---:|---:|---:|---:|---:|---:|
| best | {best['epoch']} | {best['MRE_PS1']:.3f} | {best['MRE_PS2']:.3f} | {best['MRE_FH1']:.3f} | {best['MRE_ALL']:.3f} | {best['aop_mae_deg']:.3f}° | 100/100 |
| last | {last['epoch']} | {last['MRE_PS1']:.3f} | {last['MRE_PS2']:.3f} | {last['MRE_FH1']:.3f} | {last['MRE_ALL']:.3f} | {last['aop_mae_deg']:.3f}° | 100/100 |

best 出现在第 3 轮，之后 validation 有明显波动；第 20 轮仍保持 100/100 有效 AoP，但两项选择指标都不如 best。波动是实际观察，batch size 1 下的 BatchNorm 统计量只是一个可能风险，目前不能把它写成已证实的原因。这组结果更适合作为“HRNet 已正确接入并能训练”的监督参考，不应包装成稳定性已经解决。

旧的 U-Net B2 在另一轮实验中的 validation best 是 MRE_ALL 24.779 px、AoP MAE 8.514°。由于 backbone、参数量和训练阶段都不同，这个数字只放在旁边帮助定位量级，不用来得出“HRNet 更好/更差”的因果结论。

实验 ledger 记录的 GPU 用时为 **{resources['experiment_ledger_gpu_minutes']:.2f} 分钟**；另一次只读 best/last validation 复算约 **{resources['independent_validation_audit_gpu_minutes_approx']:.2f} 分钟**，所有保存指标复现差值为 0。两部分合计约 **{resources['total_audited_gpu_minutes_approx']:.2f} 分钟**，低于本阶段 180 分钟总预算。

曲线见 [validation_metrics.png](curves/validation_metrics.png)，公开汇总见 [aggregate_results.json](aggregate_results.json)。
"""
    path.write_text(text, encoding="utf-8")


def _write_phase_summary(path: Path, aggregate: Mapping[str, Any]) -> None:
    b0 = aggregate["b0_diagnostics"]
    a4 = aggregate["gates"]["A4_unet_B0"]
    b4 = aggregate["gates"]["B4_hrnet_B2"]
    formal = aggregate["formal_run"]
    reference = aggregate["reference_only"]["phase06_unet_B2"]
    mean = b0["train_mean_coordinate_reference"]
    sanity = b0["synthetic_decode_checks"]
    resources = aggregate["resources"]
    text = f"""# Phase 1A 小结

这轮主要做了两件事：先把 Phase 0.6 里纯 MSE 的异常拆开看，再把 HRNet-W32 作为共享解码头的监督参考接进来。所有判断都只用 train 和 validation，testing 没有读取或重新评估。

## 已实现测试

- B0/B1/B2 共 6 个保存端点的完整 validation 诊断，包括 raw heatmap、softmax 概率、DSNT/argmax、射线长度和 AoP 无效原因；
- 标准高斯、低振幅高斯、零热图和平坦热图的解码检查；
- B3 的 stage4 四尺度、梯度、参数更新、train/eval 与 checkpoint 往返检查；
- B4 的四样本数值门槛、坐标换算检查和 4 张叠加图人工检查；
- H1 的 20 行 validation 日志、best 选择 tuple、best/last checkpoint 配置和保存指标复算。

最终测试命令为 `python -m pytest -q -p no:cacheprovider`，具体结果以提交前的收口验证为准。

GitHub Actions 覆盖 ruff、pytest 与 7 个 Phase 1A 命令行入口，远端结果以当前分支的 Actions 状态为准。

## 真实运行

B0 的 best 在第 120 轮。用 DSNT 解码时 MRE_ALL 为 **{b0['best_dsnt']['MRE_ALL']:.3f} px**、有效样本 AoP MAE 为 **{b0['best_dsnt']['aop_mae_valid_deg']:.3f}°**；同一 checkpoint 改用 argmax，MRE_ALL 是 **{b0['best_argmax']['MRE_ALL']:.3f} px**。到第 200 轮，三张输出热图已经变成空间常数，raw heatmap 的空间标准差为 0，DSNT 三点落在同一中心位置，因此有效 AoP 变成 0/100；此时 180° 是无效预测的惩罚选择分数，不是实测 AoP MAE。

用 train 标签均值直接在 validation 上预测，MRE_ALL 为 **{mean['MRE_ALL']:.3f} px**、AoP MAE 为 **{mean['aop_mae_deg']:.3f}°**。这不是图像模型，只是说明 B0 best 的 DSNT 坐标甚至没有超过一个不看图的均值参考。现有证据支持“响应过平、DSNT 被背景质量主导”这一描述；由于没有保存崩溃转折区间的 checkpoint，不能再往前写成已经定位到某一个训练机制。

四样本纯 MSE 诊断也完整跑满 1000 步，但结果不是成功拟合：PS1 约 {a4['MRE_PS1']:.3f} px，PS2 约 {a4['MRE_PS2']:.3f} px，FH1 约 {a4['MRE_FH1']:.3f} px，整体 MRE **{a4['MRE_ALL']:.3f} px**。因此这里把“程序完整执行”和“三点学会”分开记录。

B3 结构探针通过；B4 固定四样本跑满 500 步，eval MRE_ALL **{b4['MRE_ALL']:.4f} px**，4/4 AoP 有效，叠加图人工检查通过。H1 监督参考完整跑完 20 轮，best 为第 {formal['best']['epoch']} 轮：MRE_ALL **{formal['best']['MRE_ALL']:.3f} px**、AoP MAE **{formal['best']['aop_mae_deg']:.3f}°**；第 20 轮分别为 **{formal['last']['MRE_ALL']:.3f} px** 和 **{formal['last']['aop_mae_deg']:.3f}°**，best/last 都是 100/100 有效 AoP。独立只读复算的 best/last 保存指标最大差值为 0。

## 仅合成

标准高斯用 argmax 的误差是 0；温度 0.05 的 DSNT 误差只有 **{sanity['gaussian_dsnt_t0.05_MRE_ALL']:.4f} px**，但温度 1 时会被大片背景质量拉向中心，误差变成 **{sanity['gaussian_dsnt_t1_MRE_ALL']:.3f} px**。即使仍是正确位置的高斯，把振幅缩到 0.1 后，当前温度 0.05 的 DSNT 误差也达到 **{sanity['amplitude_0.1_dsnt_t0.05_MRE_ALL']:.3f} px**。零热图和三张平坦热图都会得到无效 AoP。这一组只验证数学与接口，不替代真实模型结果，也不单独证明 B0 崩溃的训练原因。

## 结果未改善

旧 U-Net B2 的 validation best 是 {reference['MRE_ALL']:.3f} px / {reference['aop_mae_deg']:.3f}°。两者架构不同，这里不做因果比较。H1 的 best 很早、后续波动明显；batch size 1 的 BatchNorm 是需要留意的风险，但现有实验没有证明它就是波动原因。

A4 虽然跑满预算，但没有学会 PS2/FH1；H1 第 20 轮也没有超过自身第 3 轮的 validation best。因此本轮证明的是 HRNet 接入与训练闭环成立，不是正式性能已经改善，更不是半监督方法已经有效。

## 预算内未完成

无。B3、A4、B4 和 H1 都在各自上限内结束；实验 ledger 为 **{resources['experiment_ledger_gpu_minutes']:.2f} 分钟**，加上独立只读复算后总审计口径约 **{resources['total_audited_gpu_minutes_approx']:.2f} 分钟**，低于 180 分钟总预算。

## 外部阻塞

就 Phase 1A 的监督参考范围而言没有外部阻塞。后续半监督阶段仍缺少可核验的无标签池，但这不影响本轮监督诊断与 HRNet 接入验收。

## 结论边界

HRNet-W32 已按 stage4 最终融合的高分辨率分支接入，四样本门槛证明模型、坐标和反向传播链路能共同工作，20 轮完整运行也证明正式训练闭环可复现。它现在仍只是共享解码头的增强监督参考：没有 PS/FH 解耦、EMA 教师、伪标签或半监督损失，也没有 testing 结论。

实现细节见 [HRNET_IMPLEMENTATION.md](HRNET_IMPLEMENTATION.md)，脱敏数字见 [aggregate_results.json](aggregate_results.json)。
"""
    path.write_text(text, encoding="utf-8")


def summarize_phase1a(
    *,
    run_root: Path = CANONICAL_RUN_ROOT,
    report_root: Path = CANONICAL_REPORT_ROOT,
    phase06_aggregate: Path = CANONICAL_PHASE06_AGGREGATE,
) -> dict[str, Any]:
    diagnostics = _validate_diagnostics(report_root)
    validated = {
        "diagnostics": diagnostics,
        "b3": _validate_b3(run_root),
        "a4": _validate_tiny_gate(run_root, "A4_unet_B0"),
        "b4": _validate_tiny_gate(run_root, "B4_hrnet_B2"),
        "formal": _validate_formal_run(run_root),
        "reference": _validate_phase06_reference(phase06_aggregate),
        "budget": _validate_gpu_budget(run_root),
        "independent_audit": _validate_independent_audit(run_root),
    }
    aggregate = _build_public_aggregate(validated)
    report_root.mkdir(parents=True, exist_ok=True)
    curve_root = report_root / "curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_root / "aggregate_results.json", aggregate)
    (report_root / "sanitized_config.yaml").write_text(
        yaml.safe_dump(_sanitized_config(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_curve(curve_root / "validation_metrics.png", validated["formal"]["history"])
    _write_hrnet_report(report_root / "HRNET_IMPLEMENTATION.md", aggregate)
    _write_phase_summary(report_root / "PHASE1A_SUMMARY.md", aggregate)
    return aggregate


def main() -> None:
    args = build_parser().parse_args()
    aggregate = summarize_phase1a(
        run_root=args.run_root,
        report_root=args.report_root,
        phase06_aggregate=args.phase06_aggregate,
    )
    best = aggregate["formal_run"]["best"]
    print(
        "Phase 1A summary validated: "
        f"best epoch={best['epoch']}, MRE_ALL={best['MRE_ALL']:.3f}, "
        f"AoP MAE={best['aop_mae_deg']:.3f}"
    )


if __name__ == "__main__":
    main()
