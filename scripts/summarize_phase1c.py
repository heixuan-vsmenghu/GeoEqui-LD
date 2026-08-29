#!/usr/bin/env python
# ruff: noqa: E501
"""Validate fixed Phase 1C artifacts and publish sanitized aggregate reports.

Only the explicitly named operator, tiny-gate, formal-run, Phase 1B aggregate,
and protocol files are opened.  The script never discovers dataset contents,
loads model weights, or accesses the frozen testing split.  Public outputs are
rebuilt from an allowlist of aggregate fields; provenance hashes, timestamps,
local paths, per-sample material, and model state are deliberately omitted.
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
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase1c"
OPERATOR_PROBE = (
    REPOSITORY_ROOT
    / "runs"
    / "phase1c"
    / "P1C_deform_cuda_probe_retry4"
    / "operator_probe.json"
)
TINY_RESULT = (
    REPOSITORY_ROOT
    / "runs"
    / "phase1c"
    / "P1C_specialized_tiny_B2_retry1"
    / "tiny_result.json"
)
TINY_REVIEW = (
    REPOSITORY_ROOT
    / "runs"
    / "phase1c"
    / "P1C_specialized_tiny_review_retry1"
    / "tiny_review.json"
)
FORMAL_ROOT = REPOSITORY_ROOT / "runs" / "phase1c" / "H3_specialized_B2_seed42_16e"
PHASE1B_AGGREGATE = REPOSITORY_ROOT / "reports" / "phase1b" / "aggregate_results.json"
PROTOCOL_CONFIG = REPOSITORY_ROOT / "configs" / "phase1c_specialized_enhancers.yaml"

METRIC_NAMES = ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
SELECTION_ORDER = ("penalized_aop_mae_deg", "MRE_ALL", "earlier_epoch")
HISTORY_COLUMNS = (
    "epoch",
    "optimization_time_sec",
    "train_evaluation_time_sec",
    "validation_time_sec",
    "epoch_time_sec",
    "optimization_total_loss",
    "optimization_heatmap_mse",
    "optimization_coordinate_smooth_l1",
    "optimization_distribution_js",
    "optimization_batches",
    "train_total_loss",
    "train_heatmap_mse",
    "train_coordinate_smooth_l1",
    "train_distribution_js",
    "train_MRE_PS1",
    "train_MRE_PS2",
    "train_MRE_FH1",
    "train_MRE_ALL",
    "train_n_samples",
    "train_decoder",
    "train_n_valid_aop",
    "train_n_evaluable_aop",
    "train_aop_invalid_prediction_count",
    "train_aop_mae_valid_deg",
    "train_aop_mae_deg",
    "train_aop_valid_rate",
    "train_selection_aop_penalized_deg",
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
    "val_aop_valid_rate",
    "val_selection_aop_penalized_deg",
)

# This is a curated execution-history fact.  The failed private gate artifact is
# intentionally not opened or copied into the public reporting pipeline.
TINY_GATE_HISTORY = {
    "failed_attempts_before_pass": 1,
    "minimal_gate_fixes": 1,
    "fix_scope": (
        "The gradient audit was narrowed to parameters participating in the configured "
        "final-feature path, while every dedicated enhancer and decoder parameter still "
        "had to receive a finite gradient."
    ),
    "model_loss_threshold_or_step_limit_changed": False,
}

UNLABELED_AUDIT = {
    "complete_pool_present": False,
    "trainable_files_available": 0,
    "configured_directory_present": False,
    "default_data_root_available": False,
    "environment_override_configured": False,
    "partial_archive_usable": False,
    "partial_archive_completion_percent": 21.816,
    "conflicting_documented_counts": [31421, 31121],
    "example_subset_count": 2045,
    "example_subset_can_be_assumed_additional": False,
    "source_traceability": "Kaggle and Zenodo records are identifiable.",
    "license_status": "Conflicting or incomplete license/signature evidence remains unresolved.",
    "labeled_overlap_status": "Not audited; testing remained frozen, so complete overlap clearance is unknown.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate fixed Phase 1C artifacts and publish sanitized reports."
    )
    parser.add_argument("--operator-probe", type=Path, default=OPERATOR_PROBE)
    parser.add_argument("--tiny-result", type=Path, default=TINY_RESULT)
    parser.add_argument("--tiny-review", type=Path, default=TINY_REVIEW)
    parser.add_argument(
        "--formal-result", type=Path, default=FORMAL_ROOT / "formal_result.json"
    )
    parser.add_argument("--train-log", type=Path, default=FORMAL_ROOT / "train_log.csv")
    parser.add_argument(
        "--key-metrics", type=Path, default=FORMAL_ROOT / "key_checkpoint_metrics.json"
    )
    parser.add_argument("--phase1b-aggregate", type=Path, default=PHASE1B_AGGREGATE)
    parser.add_argument("--protocol-config", type=Path, default=PROTOCOL_CONFIG)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    return parser


def _require_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Required regular file is missing: {path.name}")


def _reject_forbidden_split_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
            if normalized == "testing_frozen":
                if item is not True:
                    raise PermissionError(f"{context} does not preserve the testing freeze")
            elif tokens & {"test", "testing"}:
                raise PermissionError(
                    f"{context} contains a forbidden split-derived field: {key}"
                )
            _reject_forbidden_split_keys(item, context=context)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _reject_forbidden_split_keys(item, context=context)


def _read_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path.name}")
    _reject_forbidden_split_keys(value, context=path.name)
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    _require_file(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path.name}")
    _reject_forbidden_split_keys(value, context=path.name)
    return value


def _finite(value: Any, *, context: str, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    if nonnegative and number < 0:
        raise ValueError(f"{context} must be non-negative")
    return number


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    number = _finite(value, context=context)
    integer = int(number)
    if number != integer or integer < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return integer


def _require(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


def _close(
    actual: Any, expected: Any, *, context: str, tolerance: float = 1.0e-5
) -> None:
    left = _finite(actual, context=context)
    right = _finite(expected, context=context)
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{context} differs between artifacts")


def _metric_view(
    value: Mapping[str, Any], *, context: str, expected_samples: int
) -> dict[str, Any]:
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
        "aop_mae_deg",
    }
    if not required.issubset(value):
        raise ValueError(f"{context} is missing required aggregate metrics")
    result = {
        name: _finite(value[name], context=f"{context}.{name}", nonnegative=True)
        for name in METRIC_NAMES
    }
    result.update(
        {
            "n_samples": _integer(value["n_samples"], context=f"{context}.n_samples"),
            "n_valid_aop": _integer(
                value["n_valid_aop"], context=f"{context}.n_valid_aop"
            ),
            "n_evaluable_aop": _integer(
                value["n_evaluable_aop"], context=f"{context}.n_evaluable_aop"
            ),
            "aop_invalid_prediction_count": _integer(
                value["aop_invalid_prediction_count"],
                context=f"{context}.aop_invalid_prediction_count",
            ),
        }
    )
    _require(value["decoder"], "dsnt", context=f"{context}.decoder")
    _require(result["n_samples"], expected_samples, context=f"{context}.n_samples")
    _require(
        result["n_evaluable_aop"], expected_samples, context=f"{context}.n_evaluable"
    )
    _require(
        result["n_valid_aop"] + result["aop_invalid_prediction_count"],
        expected_samples,
        context=f"{context}.AoP counts",
    )
    _close(
        result["MRE_ALL"],
        sum(result[name] for name in METRIC_NAMES[:3]) / 3.0,
        context=f"{context}.MRE_ALL",
    )
    if "aop_mae_valid_deg" in value:
        _close(
            result["aop_mae_deg"],
            value["aop_mae_valid_deg"],
            context=f"{context}.AoP representations",
        )
    return result


def _history_metrics(
    row: Mapping[str, str], *, prefix: str, expected_samples: int, context: str
) -> dict[str, Any]:
    raw = {
        "MRE_PS1": row[f"{prefix}_MRE_PS1"],
        "MRE_PS2": row[f"{prefix}_MRE_PS2"],
        "MRE_FH1": row[f"{prefix}_MRE_FH1"],
        "MRE_ALL": row[f"{prefix}_MRE_ALL"],
        "n_samples": row[f"{prefix}_n_samples"],
        "decoder": row[f"{prefix}_decoder"],
        "n_valid_aop": row[f"{prefix}_n_valid_aop"],
        "n_evaluable_aop": row[f"{prefix}_n_evaluable_aop"],
        "aop_invalid_prediction_count": row[
            f"{prefix}_aop_invalid_prediction_count"
        ],
        "aop_mae_valid_deg": row[f"{prefix}_aop_mae_valid_deg"],
        "aop_mae_deg": row[f"{prefix}_aop_mae_deg"],
    }
    metrics = _metric_view(raw, context=context, expected_samples=expected_samples)
    _close(
        row[f"{prefix}_aop_valid_rate"],
        metrics["n_valid_aop"] / metrics["n_evaluable_aop"],
        context=f"{context}.valid_rate",
    )
    _close(
        row[f"{prefix}_selection_aop_penalized_deg"],
        metrics["aop_mae_deg"],
        context=f"{context}.selection_score",
    )
    return metrics


def _read_history(path: Path, *, expected_epochs: int = 16) -> tuple[dict[str, Any], ...]:
    _require_file(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()), HISTORY_COLUMNS, context="history columns")
        raw_rows = list(reader)
    _require(len(raw_rows), expected_epochs, context="history row count")
    history: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(raw_rows, start=1):
        epoch = _integer(row["epoch"], context="history epoch", minimum=1)
        _require(epoch, expected_epoch, context="history epoch sequence")
        for name in (
            "optimization_time_sec",
            "train_evaluation_time_sec",
            "validation_time_sec",
            "epoch_time_sec",
            "optimization_total_loss",
            "optimization_heatmap_mse",
            "optimization_coordinate_smooth_l1",
            "optimization_distribution_js",
            "train_total_loss",
            "train_heatmap_mse",
            "train_coordinate_smooth_l1",
            "train_distribution_js",
            "val_total_loss",
            "val_heatmap_mse",
            "val_coordinate_smooth_l1",
            "val_distribution_js",
        ):
            _finite(row[name], context=f"history e{epoch}.{name}", nonnegative=True)
        _require(
            _integer(row["optimization_batches"], context="optimization batches"),
            300,
            context="optimization batches",
        )
        train = _history_metrics(
            row,
            prefix="train",
            expected_samples=300,
            context=f"history e{epoch} train",
        )
        validation = _history_metrics(
            row,
            prefix="val",
            expected_samples=100,
            context=f"history e{epoch} validation",
        )
        history.append({"epoch": epoch, "train": train, "validation": validation})
    return tuple(history)


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    metrics = row["validation"]
    return (
        float(metrics["aop_mae_deg"]),
        float(metrics["MRE_ALL"]),
        int(row["epoch"]),
    )


def _rounded_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: round(float(value[name]), 6)
        for name in METRIC_NAMES
    } | {
        "n_valid_aop": int(value["n_valid_aop"]),
        "n_evaluable_aop": int(value["n_evaluable_aop"]),
    }


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: round(float(left[name]) - float(right[name]), 6)
        for name in METRIC_NAMES
    }


def _gap(row: Mapping[str, Any]) -> dict[str, float]:
    return _delta(row["validation"], row["train"])


def _phase1b_metrics(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    result = {
        name: _finite(value[name], context=f"{context}.{name}", nonnegative=True)
        for name in METRIC_NAMES
    }
    result.update(
        {
            "n_valid_aop": _integer(
                value.get("n_valid_aop", 100), context=f"{context}.n_valid"
            ),
            "n_evaluable_aop": _integer(
                value.get("n_evaluable_aop", 100), context=f"{context}.n_evaluable"
            ),
        }
    )
    _require(result["n_valid_aop"], 100, context=f"{context}.n_valid")
    _require(result["n_evaluable_aop"], 100, context=f"{context}.n_evaluable")
    _close(
        result["MRE_ALL"],
        sum(result[name] for name in METRIC_NAMES[:3]) / 3.0,
        context=f"{context}.MRE_ALL",
    )
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("experiment_name"), "H3_specialized_B2_seed42_16e", context="config experiment")
    _require(config.get("testing_frozen"), True, context="config testing freeze")
    training = config.get("training", {})
    expected_training = {
        "seed": 42,
        "input_size_hw": [512, 512],
        "heatmap_size_hw": [256, 256],
        "sigma_heatmap_px": 4.0,
        "align_corners": True,
        "dsnt_temperature": 0.05,
        "keypoint_order": ["PS1", "PS2", "FH1"],
        "batch_size": 1,
        "epochs": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "heatmap_loss_weight": 1.0,
        "coordinate_loss_weight": 10.0,
        "distribution_loss_weight": 1.0,
        "max_grad_norm": 5.0,
    }
    for key, expected in expected_training.items():
        _require(training.get(key), expected, context=f"config training.{key}")
    model = config.get("model", {})
    expected_model = {
        "class": "HRNetW32SpecializedHeatmap",
        "backbone": "hrnet_w32",
        "timm_version": "1.0.28",
        "torchvision_version": "0.20.1",
        "pretrained": False,
        "in_channels": 1,
        "out_channels": 3,
        "out_indices": [1],
        "feature_channels": 32,
        "feature_reduction": 4,
        "concatenation_order": ["PS1", "PS2", "FH1"],
        "ps_deformable_operator": "torchvision.ops.DeformConv2d",
        "ps_offset_channels": 18,
        "ps_mask_channels": 9,
        "ps_spatial_attention_channels": 1,
        "ps_normalization": "LayerNorm2d",
        "fh_aspp_dilations": [1, 3, 6],
        "fh_se_hidden_channels": 8,
        "fh_normalization": "LayerNorm2d",
    }
    for key, expected in expected_model.items():
        _require(model.get(key), expected, context=f"config model.{key}")
    optimizer = config.get("optimizer", {})
    _require(optimizer.get("class"), "Adam", context="config optimizer")
    _require(optimizer.get("foreach"), False, context="config optimizer.foreach")
    resources = config.get("resources", {})
    for key, expected in {
        "precision": "float32",
        "amp_enabled": False,
        "operator_probe_max_seconds": 300,
        "tiny_max_steps": 500,
        "tiny_max_seconds": 2400,
        "formal_max_seconds": 9000,
        "total_gpu_max_seconds": 10800,
    }.items():
        _require(resources.get(key), expected, context=f"config resources.{key}")


def _validate_operator(probe: Mapping[str, Any]) -> None:
    _require(probe.get("status"), "completed", context="operator status")
    _require(probe.get("gate"), "PASS", context="operator gate")
    elapsed = _finite(
        probe.get("runtime_elapsed_seconds"), context="operator elapsed", nonnegative=True
    )
    allocated = _finite(
        probe.get("runtime_allocated_seconds"),
        context="operator allocation",
        nonnegative=True,
    )
    if elapsed > allocated or allocated != 300.0:
        raise ValueError("operator probe exceeded or changed its fixed allocation")
    evidence = probe.get("evidence", {})
    for key, expected in {
        "input_shape": [1, 32, 32, 32],
        "offset_shape": [1, 18, 32, 32],
        "mask_logits_shape": [1, 9, 32, 32],
        "mask_shape": [1, 9, 32, 32],
        "output_shape": [1, 32, 32, 32],
        "initial_offset_max_abs": 0.0,
        "initial_mask_logits_max_abs": 0.0,
        "initial_mask_min": 0.5,
        "initial_mask_max": 0.5,
        "all_values_finite": True,
        "all_enhancer_gradients_finite": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": True,
        "strict_deterministic_deform_backward_available": False,
        "actual_operator": "torchvision.ops.deform_conv.DeformConv2d",
        "ordinary_conv_fallback": False,
    }.items():
        _require(evidence.get(key), expected, context=f"operator {key}")
    for key in (
        "offset_predictor_gradient_l1",
        "mask_predictor_gradient_l1",
        "deform_weight_gradient_l1",
        "spatial_attention_gradient_l1",
    ):
        if _finite(evidence.get(key), context=f"operator {key}") <= 0:
            raise ValueError(f"operator {key} must be nonzero")


def _validate_tiny(tiny: Mapping[str, Any], review: Mapping[str, Any]) -> None:
    _require(tiny.get("status"), "completed", context="tiny status")
    _require(tiny.get("programmatic_gate"), "PASS", context="tiny programmatic gate")
    _require(tiny.get("steps_completed"), 500, context="tiny steps")
    _require(tiny.get("operator_probe_gate"), "PASS", context="tiny operator binding")
    structure = tiny.get("structure_probe", {})
    _require(structure.get("input_shape"), [1, 1, 512, 512], context="tiny input shape")
    _require(structure.get("shape_contract_passed"), True, context="tiny shape contract")
    _require(structure.get("all_values_finite"), True, context="tiny finite structure")
    _require(
        structure.get("channel_order"), ["PS1", "PS2", "FH1"], context="tiny channel order"
    )
    actual = structure.get("actual_shapes", {})
    for key, expected in {
        "feature": [1, 32, 128, 128],
        "ps_enhancer_input": [1, 32, 128, 128],
        "ps_enhancer_output": [1, 32, 128, 128],
        "ps_decoder_output": [1, 2, 128, 128],
        "fh_enhancer_input": [1, 32, 128, 128],
        "fh_enhancer_output": [1, 32, 128, 128],
        "fh_decoder_output": [1, 1, 128, 128],
        "output": [1, 3, 256, 256],
        "dsnt": [1, 3, 2],
    }.items():
        _require(actual.get(key), expected, context=f"tiny shape {key}")
    gradients = tiny.get("gradient_evidence", {})
    for key in (
        "all_dedicated_parameter_gradients_present",
        "all_required_gradients_finite",
        "all_required_nonzero",
    ):
        _require(gradients.get(key), True, context=f"tiny gradients {key}")
    metrics = tiny.get("eval_mode", {})
    _require(metrics.get("n_samples"), 4, context="tiny samples")
    _require(metrics.get("n_evaluable_aop"), 4, context="tiny evaluable AoP")
    _require(metrics.get("n_valid_aop"), 4, context="tiny valid AoP")
    _require(metrics.get("aop_invalid_prediction_count"), 0, context="tiny invalid AoP")
    _require(metrics.get("nonfinite_count"), 0, context="tiny nonfinite count")
    if _finite(metrics.get("MRE_ALL"), context="tiny MRE", nonnegative=True) > 5.0:
        raise ValueError("tiny MRE_ALL did not pass the fixed 5 px threshold")
    for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "aop_mae_deg"):
        _finite(metrics.get(name), context=f"tiny {name}", nonnegative=True)
    visualization = tiny.get("visualization", {})
    _require(visualization.get("visualization_count"), 4, context="tiny review count")
    _require(
        visualization.get("programmatic_check_passed"), True, context="tiny coordinate review"
    )
    _require(review.get("decision"), "PASS", context="tiny manual review")
    _require(
        review.get("coordinate_or_channel_mismatch_observed"),
        False,
        context="tiny coordinate/channel review",
    )


def _validate_determinism(value: Mapping[str, Any], *, context: str) -> None:
    for key, expected in {
        "seed": 42,
        "data_order_generator_seeded": True,
        "cudnn_deterministic": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": True,
        "strict_bitwise_determinism_claimed": False,
        "known_nondeterministic_operation": "DeformConv2d CUDA backward",
    }.items():
        _require(value.get(key), expected, context=f"{context}.{key}")


def _validate_formal(
    formal: Mapping[str, Any],
    key_metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    _require(formal.get("status"), "completed", context="formal status")
    _require(formal.get("epochs_completed"), 16, context="formal completed epochs")
    _require(formal.get("epochs_requested"), 16, context="formal requested epochs")
    _require(formal.get("partial"), False, context="formal partial flag")
    _require(
        tuple(formal.get("selection_order", ())), SELECTION_ORDER, context="selection order"
    )
    _require(formal.get("best_epoch"), 14, context="formal best epoch")
    elapsed = _finite(formal.get("runtime_elapsed_sec"), context="formal elapsed", nonnegative=True)
    allocated = _finite(
        formal.get("runtime_allocated_sec"), context="formal allocation", nonnegative=True
    )
    if elapsed > allocated or allocated != 9000.0:
        raise ValueError("formal run exceeded or changed its fixed allocation")
    for key in (
        "formal_allocation_exceeded",
        "aggregate_gpu_cap_exceeded",
        "training_subbudget_exhausted",
    ):
        _require(formal.get(key), False, context=f"formal {key}")
    _require(formal.get("within_runtime_allocation"), True, context="formal runtime status")
    _validate_determinism(formal.get("determinism_policy", {}), context="formal determinism")
    initialization = formal.get("initialization", {})
    _require(
        initialization.get("base_state_values_equal"),
        {"backbone": True, "ps_decoder": True, "fh_decoder": True},
        context="formal base initialization",
    )
    _require(
        initialization.get("base_parameter_storage_aliased"),
        False,
        context="formal base storage",
    )
    _require(
        initialization.get("h2_trainable_parameters"), 29332275, context="H2 parameters"
    )
    _require(
        initialization.get("h3_trainable_parameters"), 29372695, context="H3 parameters"
    )
    _require(
        initialization.get("additional_trainable_parameters"), 40420, context="new parameters"
    )
    _require(
        initialization.get("complete_function_equivalent"),
        False,
        context="complete-function equivalence",
    )
    selected = min(history, key=_selection_key)
    _require(selected["epoch"], 14, context="recomputed selected epoch")
    best_metrics = _metric_view(
        formal.get("best_validation_metrics", {}),
        context="formal best validation",
        expected_samples=100,
    )
    last_metrics = _metric_view(
        formal.get("last_validation_metrics", {}),
        context="formal last validation",
        expected_samples=100,
    )
    for name in METRIC_NAMES:
        _close(best_metrics[name], selected["validation"][name], context=f"formal best {name}")
        _close(last_metrics[name], history[-1]["validation"][name], context=f"formal last {name}")
    _require(key_metrics.get("status"), "completed", context="key metric audit")
    checkpoints = key_metrics.get("checkpoints", {})
    for endpoint, epoch, row in (
        ("best", 14, selected),
        ("last", 16, history[-1]),
    ):
        audit = checkpoints.get(endpoint, {})
        _require(audit.get("epoch"), epoch, context=f"{endpoint} audit epoch")
        train = _metric_view(
            audit.get("train", {}), context=f"{endpoint} train", expected_samples=300
        )
        validation = _metric_view(
            audit.get("validation", {}),
            context=f"{endpoint} validation",
            expected_samples=100,
        )
        for name in METRIC_NAMES:
            _close(train[name], row["train"][name], context=f"{endpoint} train {name}")
            _close(
                validation[name], row["validation"][name], context=f"{endpoint} validation {name}"
            )
        for flag in (
            "evaluation_state_unchanged",
            "checkpoint_epoch_matches_selection",
            "checkpoint_metrics_match_train_log",
            "recomputed_validation_matches_checkpoint",
            "full_resume_state_present",
        ):
            _require(audit.get(flag), True, context=f"{endpoint} audit {flag}")


def _validate_phase1b(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scope = value.get("data_scope", {})
    _require(scope.get("train_samples"), 300, context="Phase 1B train count")
    _require(scope.get("validation_samples"), 100, context="Phase 1B validation count")
    _require(scope.get("testing_frozen"), True, context="Phase 1B testing freeze")
    control = value.get("decoder_control", {})
    h1 = control.get("h1_shared", {})
    h2 = control.get("h2_split", {})
    _require(h1.get("status"), "completed", context="H1 status")
    _require(h1.get("epochs_completed"), 20, context="H1 epochs")
    _require(h2.get("status"), "budget_exhausted", context="H2 status")
    _require(h2.get("partial"), True, context="H2 partial flag")
    _require(h2.get("epochs_completed"), 16, context="H2 epochs")
    _require(
        control.get("parameter_counts"),
        {"h1_shared": 29318355, "h2_split": 29332275, "h2_minus_h1": 13920},
        context="H1/H2 parameters",
    )
    matched = control.get("matched_epoch16", {})
    return {
        "h1_best": _phase1b_metrics(h1.get("best", {}), context="H1 best"),
        "h2_best": _phase1b_metrics(h2.get("best", {}), context="H2 best"),
        "h1_e16": _phase1b_metrics(
            matched.get("h1_shared", {}), context="H1 epoch 16"
        ),
        "h2_e16": _phase1b_metrics(
            matched.get("h2_split", {}), context="H2 epoch 16"
        ),
        "h2_gap_e3": {
            name: _finite(
                h2.get("best_validation_minus_train", {}).get(name),
                context=f"H2 e3 gap {name}",
            )
            for name in METRIC_NAMES
        },
        "h2_gap_e16": {
            name: _finite(
                h2.get("last_observed_validation_minus_train", {}).get(name),
                context=f"H2 e16 gap {name}",
            )
            for name in METRIC_NAMES
        },
    }


def _public_hygiene(value: Any, *, context: str) -> None:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    if re.search(r"(?i)(?:^|[\s\"'(])(?:[a-z]:[\\/]|/users/|/home/)", text):
        raise ValueError(f"{context} contains a local absolute path")
    if re.search(r"(?i)\b[0-9a-f]{64}\b", text):
        raise ValueError(f"{context} contains a provenance hash")
    lowered = text.lower()
    for token in (
        "sha256",
        "fingerprint",
        "git_head",
        "started_at_utc",
        "finished_at_utc",
        "entry_sha",
        "sample_00",
        "sample_01",
        "sample_02",
        "sample_03",
        "private_predictions",
        "state_dict",
        ".pt\"",
    ):
        if token in lowered:
            raise ValueError(f"{context} contains forbidden public detail: {token}")


def _reference_points(
    phase1b: Mapping[str, Mapping[str, Any]], history: Sequence[Mapping[str, Any]]
) -> dict[str, dict[int, Mapping[str, Any]]]:
    return {
        "H1": {3: phase1b["h1_best"], 16: phase1b["h1_e16"]},
        "H2": {3: phase1b["h2_best"], 16: phase1b["h2_e16"]},
        "H3": {int(row["epoch"]): row["validation"] for row in history},
    }


def _save_png(figure: Any, path: Path) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, metadata={})
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as image:
        clean = image.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        clean.save(path, format="PNG", optimize=True)


def _write_curves(
    curve_root: Path,
    history: Sequence[Mapping[str, Any]],
    phase1b: Mapping[str, Mapping[str, Any]],
) -> None:
    points = _reference_points(phase1b, history)
    colors = {
        "MRE_PS1": "#2166ac",
        "MRE_PS2": "#b2182b",
        "MRE_FH1": "#1b7837",
        "MRE_ALL": "#542788",
        "aop_mae_deg": "#e08214",
    }
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.72), dpi=100)
    epochs = [int(row["epoch"]) for row in history]
    for name in METRIC_NAMES[:3]:
        axes[0].plot(
            epochs,
            [row["validation"][name] for row in history],
            color=colors[name],
            label=f"H3 {name.replace('MRE_', '')}",
        )
    for name in METRIC_NAMES[3:]:
        label = "H3 MRE_ALL" if name == "MRE_ALL" else "H3 AoP MAE"
        axes[1].plot(
            epochs,
            [row["validation"][name] for row in history],
            color=colors[name],
            label=label,
        )
    for model, marker in (("H1", "x"), ("H2", "D")):
        for name in METRIC_NAMES[:3]:
            axes[0].scatter(
                (3, 16),
                [points[model][epoch][name] for epoch in (3, 16)],
                color=colors[name],
                marker=marker,
                s=55,
                linewidths=1.5,
                alpha=0.85,
            )
        for name in METRIC_NAMES[3:]:
            axes[1].scatter(
                (3, 16),
                [points[model][epoch][name] for epoch in (3, 16)],
                color=colors[name],
                marker=marker,
                s=55,
                linewidths=1.5,
                alpha=0.85,
            )
    axes[0].set(title="Validation point errors", xlabel="Epoch", ylabel="MRE (px)")
    axes[1].set(title="Validation summary metrics", xlabel="Epoch", ylabel="Metric value")
    for axis in axes:
        axis.set_xlim(1, 16)
        axis.set_xticks((1, 3, 5, 10, 14, 16))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Phase 1C supervised H3 validation curves; H1 x / H2 diamond at e3 and e16")
    figure.tight_layout()
    _save_png(figure, curve_root / "validation_metrics.png")

    figure, axes = plt.subplots(1, 2, figsize=(16, 6.72), dpi=100)
    for name in METRIC_NAMES[:4]:
        axes[0].plot(
            epochs,
            [row["validation"][name] - row["train"][name] for row in history],
            color=colors[name],
            label=name.replace("MRE_", ""),
        )
    axes[1].plot(
        epochs,
        [row["validation"]["aop_mae_deg"] - row["train"]["aop_mae_deg"] for row in history],
        color=colors["aop_mae_deg"],
        label="H3 AoP gap",
    )
    axes[0].set(title="H3 validation - train point gaps", xlabel="Epoch", ylabel="MRE gap (px)")
    axes[1].set(title="H3 validation - train AoP gap", xlabel="Epoch", ylabel="AoP MAE gap (deg)")
    for axis in axes:
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_xlim(1, 16)
        axis.set_xticks((1, 3, 5, 10, 14, 16))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Phase 1C H3 full-split evaluation gaps")
    figure.tight_layout()
    _save_png(figure, curve_root / "h3_train_validation_gap.png")


def _build_aggregate(
    probe: Mapping[str, Any],
    tiny: Mapping[str, Any],
    formal: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    phase1b: Mapping[str, Mapping[str, Any]],
    phase1b_raw: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {int(row["epoch"]): row for row in history}
    h3_e3 = {"epoch": 3, **_rounded_metrics(rows[3]["validation"])}
    h3_best = {"epoch": 14, **_rounded_metrics(rows[14]["validation"])}
    h3_e16 = {"epoch": 16, **_rounded_metrics(rows[16]["validation"])}
    h1_e3 = {"epoch": 3, **_rounded_metrics(phase1b["h1_best"])}
    h2_e3 = {"epoch": 3, **_rounded_metrics(phase1b["h2_best"])}
    h1_e16 = {"epoch": 16, **_rounded_metrics(phase1b["h1_e16"])}
    h2_e16 = {"epoch": 16, **_rounded_metrics(phase1b["h2_e16"])}
    h3_gaps = {
        "epoch3": _gap(rows[3]),
        "selected_best_epoch14": _gap(rows[14]),
        "epoch16": _gap(rows[16]),
    }
    h2_gaps = {
        "epoch3": {name: round(phase1b["h2_gap_e3"][name], 6) for name in METRIC_NAMES},
        "epoch16": {name: round(phase1b["h2_gap_e16"][name], 6) for name in METRIC_NAMES},
    }
    probe_evidence = probe["evidence"]
    tiny_metrics = tiny["eval_mode"]
    initialization = formal["initialization"]
    historical = phase1b_raw.get("historical_context_only", {}).get("unet_B2", {})
    aggregate = {
        "schema_version": 1,
        "phase": "phase1c-supervised-specialized-enhancer-control",
        "scope": "PS deformable/spatial enhancement and FH ASPP-lite/SE enhancement on the H2 split decoder",
        "data_scope": {
            "train_samples": 300,
            "validation_samples": 100,
            "testing_frozen": True,
            "unlabeled_training_used": False,
        },
        "protocol": {
            "seed": 42,
            "epochs": 16,
            "batch_size": 1,
            "loss": "B2 engineering supervision: heatmap MSE + 10x coordinate SmoothL1 + distribution JS",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "precision": "FP32",
            "selection_order": list(SELECTION_ORDER),
            "classification": "enhanced supervised engineering reference",
        },
        "determinism": {
            "seed_and_data_order_fixed": True,
            "deterministic_algorithms_enabled": True,
            "warn_only": True,
            "strict_bitwise_reproducibility_claimed": False,
            "reason": "CUDA backward for DeformConv2d is a known nondeterministic operation in this environment.",
        },
        "architecture": {
            "model": "HRNetW32SpecializedHeatmap",
            "shape_contract": {
                "input": [1, 1, 512, 512],
                "backbone_feature": [1, 32, 128, 128],
                "heatmaps": [1, 3, 256, 256],
                "dsnt_coordinates": [1, 3, 2],
                "channel_order": ["PS1", "PS2", "FH1"],
            },
            "ps_enhancer": {
                "deformable_operator": "torchvision.ops.DeformConv2d",
                "offset_channels": 18,
                "mask_channels": 9,
                "initial_offset": 0.0,
                "initial_mask": 0.5,
                "spatial_attention_channels": 1,
                "fusion": "GELU(deformable feature) x sigmoid(spatial attention), residual, channel LayerNorm",
                "ordinary_convolution_fallback": False,
            },
            "fh_enhancer": {
                "aspp_lite_dilations": [1, 3, 6],
                "se_channels": "32 to 8 to 32",
                "fusion": "ASPP-lite + SE feature, residual, channel LayerNorm",
                "aspp_batch_normalization": False,
            },
            "branch_isolation_scope": (
                "PS-only and FH-only optimizer isolation was verified through the dedicated "
                "forward_ps and forward_fh diagnostic paths. The ordinary full forward computes "
                "both branches and was used for B2 training."
            ),
            "initialization_fairness": {
                "backbone_and_decoders_equal_to_fresh_h2": True,
                "parameter_storage_aliased": False,
                "complete_h3_function_equal_to_h2": False,
                "statement": "The backbone and split decoders start from identical values, but the new specialized enhancers make the complete function non-equivalent.",
            },
            "parameter_counts": {
                "h1_shared": 29318355,
                "h2_split": 29332275,
                "h3_specialized": 29372695,
                "h3_minus_h2": 40420,
                "ps_enhancer": int(initialization["h3_component_trainable_parameters"]["ps_enhancer"]),
                "fh_enhancer": int(initialization["h3_component_trainable_parameters"]["fh_enhancer"]),
            },
        },
        "gates": {
            "deformable_operator": {
                "status": "PASS",
                "actual_operator": "torchvision.ops.DeformConv2d",
                "offset_shape": list(probe_evidence["offset_shape"]),
                "mask_shape": list(probe_evidence["mask_shape"]),
                "all_values_finite": True,
                "offset_predictor_nonzero_gradient": probe_evidence["offset_predictor_gradient_l1"] > 0,
                "mask_predictor_nonzero_gradient": probe_evidence["mask_predictor_gradient_l1"] > 0,
                "deformable_weight_nonzero_gradient": probe_evidence["deform_weight_gradient_l1"] > 0,
                "ordinary_convolution_fallback": False,
                "evidence_scope": "synthetic CUDA operator probe",
            },
            "four_sample_learning": {
                "status": "PASS",
                "steps": 500,
                "programmatic_gate": "PASS",
                "manual_coordinate_and_channel_review": "PASS",
                "MRE_PS1": round(float(tiny_metrics["MRE_PS1"]), 6),
                "MRE_PS2": round(float(tiny_metrics["MRE_PS2"]), 6),
                "MRE_FH1": round(float(tiny_metrics["MRE_FH1"]), 6),
                "MRE_ALL": round(float(tiny_metrics["MRE_ALL"]), 6),
                "aop_mae_deg": round(float(tiny_metrics["aop_mae_deg"]), 6),
                "n_valid_aop": 4,
                "n_evaluable_aop": 4,
                "all_dedicated_parameter_gradients_present": True,
                "execution_history": TINY_GATE_HISTORY,
            },
        },
        "supervised_comparison": {
            "selected_best": {
                "h1_shared": h1_e3,
                "h2_split": h2_e3,
                "h3_specialized": h3_best,
                "delta_h3_minus_h2": _delta(h3_best, h2_e3),
                "delta_h3_minus_h1": _delta(h3_best, h1_e3),
                "comparability_note": "Selected nodes occur at e3, e3, and e14 respectively and use the same validation split; this is descriptive, not a matched-epoch causal estimate.",
            },
            "matched_epoch3": {
                "h1_shared": h1_e3,
                "h2_split": h2_e3,
                "h3_specialized": h3_e3,
                "delta_h3_minus_h2": _delta(h3_e3, h2_e3),
                "delta_h3_minus_h1": _delta(h3_e3, h1_e3),
            },
            "matched_epoch16": {
                "h1_shared": h1_e16,
                "h2_split": h2_e16,
                "h3_specialized": h3_e16,
                "delta_h3_minus_h2": _delta(h3_e16, h2_e16),
                "delta_h3_minus_h1": _delta(h3_e16, h1_e16),
            },
            "h3_train_validation_gaps": h3_gaps,
            "h2_gap_reference": h2_gaps,
            "matched_gap_delta_h3_minus_h2": {
                "epoch3": _delta(h3_gaps["epoch3"], h2_gaps["epoch3"]),
                "epoch16": _delta(h3_gaps["epoch16"], h2_gaps["epoch16"]),
            },
            "pointwise_answers": {
                "PS1": "H3 is lower than H2 at selected best, e3, and e16, but the combined two-enhancer intervention does not isolate a PS-module causal effect.",
                "PS2": "H3 is lower at selected best and e16, but higher at matched e3; the direction is not uniform.",
                "FH1": "H3 is lower at selected best and e3, but higher at matched e16; the direction is not uniform.",
                "MRE_ALL": "H3 is lower than H2 at all three reported comparisons for this single run.",
                "AoP_MAE": "H3 is lower at selected best and e3, but higher at matched e16.",
                "train_validation_gap": "The H3 MRE_ALL gap is larger at e3 and smaller at e16 than H2; no uniform gap reduction is established.",
            },
            "consistent_all_keypoint_improvement": False,
            "consistency_evidence": [
                "At matched e3, H3 PS2 is higher than H2.",
                "At matched e16, H3 FH1 and AoP MAE are higher than H2.",
            ],
        },
        "formal_run": {
            "status": "completed",
            "epochs_completed": 16,
            "epochs_requested": 16,
            "selected_epoch": 14,
            "elapsed_seconds": round(float(formal["runtime_elapsed_sec"]), 3),
            "allocated_seconds": 9000.0,
            "within_allocation": True,
            "aggregate_gpu_cap_exceeded": False,
            "best_and_last_recomputed": True,
        },
        "resources": {
            "trainable_parameters": 29372695,
            "additional_parameters_over_h2": 40420,
            "peak_memory_allocated_mib": round(
                float(formal["resource_measurements"]["peak_memory_allocated_bytes"])
                / 1024**2,
                3,
            ),
            "peak_memory_reserved_mib": round(
                float(formal["resource_measurements"]["peak_memory_reserved_bytes"])
                / 1024**2,
                3,
            ),
            "formal_elapsed_minutes": round(float(formal["runtime_elapsed_sec"]) / 60.0, 3),
            "new_gpu_cap_hours": 3.0,
            "aggregate_cap_exceeded": False,
        },
        "unlabeled_intake": UNLABELED_AUDIT,
        "historical_context_only": {
            "unet_B2": {
                "best_epoch": int(historical.get("best_epoch", 15)),
                **{
                    name: round(float(historical[name]), 6)
                    for name in METRIC_NAMES
                },
                "boundary": "Different architecture and budget; descriptive context only.",
            }
        },
        "conclusion": (
            "H3 passed the operator and four-sample gates and completed 16 epochs. Its selected "
            "validation node is better than H2 on all reported metrics, and MRE_ALL is lower at "
            "e3 and e16 as well; pointwise directions nevertheless conflict across matched epochs, "
            "so no consistent all-keypoint improvement is claimed."
        ),
        "limitations": [
            "Single seed and one validation split; no significance, stability, SOTA, or generalization claim.",
            "The selected nodes are chosen and described on the same validation split.",
            "DeformConv2d CUDA backward ran under warn-only determinism, so bitwise rerun identity is not claimed.",
            "H3 changes both PS and FH enhancement branches together; observed differences do not isolate either module causally.",
            "This remains an enhanced supervised engineering reference using B2, not the advisor's pure-MSE recipe and not the complete GeoEqui-LD method.",
            "No EMA teacher, pseudo-labels, unlabeled consistency loss, confidence mechanism, or semi-supervised training was used.",
        ],
    }
    return aggregate


def _sanitized_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "phase1c-supervised-specialized-enhancer-control",
        "data_scope": {
            "train_samples": 300,
            "validation_samples": 100,
            "testing_frozen": True,
            "unlabeled_training_used": False,
        },
        "reproducibility": {
            "seed": 42,
            "data_order_seeded": True,
            "deterministic_algorithms": "warn_only",
            "strict_bitwise_reproducibility_claimed": False,
            "precision": "float32",
            "amp_enabled": False,
        },
        "training": {
            "epochs": 16,
            "batch_size": 1,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 5.0,
            "foreach": False,
            "loss_weights": {
                "heatmap_mse": 1.0,
                "coordinate_smooth_l1": 10.0,
                "distribution_js": 1.0,
            },
            "selection_order": list(SELECTION_ORDER),
        },
        "geometry": {
            "input_size": [512, 512],
            "heatmap_size": [256, 256],
            "heatmap_sigma_px": 4.0,
            "dsnt_temperature": 0.05,
            "align_corners": True,
            "keypoint_order": ["PS1", "PS2", "FH1"],
        },
        "model": {
            "class": "HRNetW32SpecializedHeatmap",
            "backbone": "hrnet_w32",
            "pretrained": False,
            "feature_channels": 32,
            "ps_operator": "torchvision.ops.DeformConv2d",
            "ps_offset_channels": 18,
            "ps_mask_channels": 9,
            "ps_normalization": "LayerNorm2d",
            "fh_aspp_dilations": [1, 3, 6],
            "fh_se_hidden_channels": 8,
            "fh_normalization": "LayerNorm2d",
        },
        "resources": {
            "operator_probe_max_seconds": 300,
            "tiny_max_steps": 500,
            "tiny_max_seconds": 2400,
            "formal_max_seconds": 9000,
            "total_gpu_max_seconds": 10800,
        },
    }


def _table_row(label: str, value: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {int(value['epoch'])} | {value['MRE_PS1']:.3f} | "
        f"{value['MRE_PS2']:.3f} | {value['MRE_FH1']:.3f} | "
        f"{value['MRE_ALL']:.3f} | {value['aop_mae_deg']:.3f} |"
    )


def _comparison_markdown(aggregate: Mapping[str, Any]) -> str:
    comparison = aggregate["supervised_comparison"]
    selected = comparison["selected_best"]
    e3 = comparison["matched_epoch3"]
    e16 = comparison["matched_epoch16"]
    return f"""# Phase 1C：H1 / H2 / H3 监督对照

本轮只比较有标签监督工程参照。H1 是共享头，H2 是 PS/FH 独立头，H3 在 H2 基础上增加 PS 与 FH 专业增强模块；三者都不是完整半监督 GeoEqui-LD。误差越低越好。

## selected best

| 模型 | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
{_table_row('H1 共享头', selected['h1_shared'])}
{_table_row('H2 独立头', selected['h2_split'])}
{_table_row('H3 专业增强', selected['h3_specialized'])}

H3 的选择节点是 epoch 14，H1 与 H2 都是 epoch 3。H3 相对 H2 的差值为 PS1 {selected['delta_h3_minus_h2']['MRE_PS1']:+.3f} px、PS2 {selected['delta_h3_minus_h2']['MRE_PS2']:+.3f} px、FH1 {selected['delta_h3_minus_h2']['MRE_FH1']:+.3f} px、MRE_ALL {selected['delta_h3_minus_h2']['MRE_ALL']:+.3f} px、AoP MAE {selected['delta_h3_minus_h2']['aop_mae_deg']:+.3f}°。这一组数字全部更低，但节点轮次不同，且来自同一 validation 的选择与汇报，只能作为单次描述。

## matched epoch 3

| 模型 | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
{_table_row('H1 共享头', e3['h1_shared'])}
{_table_row('H2 独立头', e3['h2_split'])}
{_table_row('H3 专业增强', e3['h3_specialized'])}

H3 相对 H2：PS1 {e3['delta_h3_minus_h2']['MRE_PS1']:+.3f} px，PS2 {e3['delta_h3_minus_h2']['MRE_PS2']:+.3f} px，FH1 {e3['delta_h3_minus_h2']['MRE_FH1']:+.3f} px，MRE_ALL {e3['delta_h3_minus_h2']['MRE_ALL']:+.3f} px，AoP MAE {e3['delta_h3_minus_h2']['aop_mae_deg']:+.3f}°。这里 PS2 反而更高。

## matched epoch 16

| 模型 | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
{_table_row('H1 共享头', e16['h1_shared'])}
{_table_row('H2 独立头', e16['h2_split'])}
{_table_row('H3 专业增强', e16['h3_specialized'])}

H3 相对 H2：PS1 {e16['delta_h3_minus_h2']['MRE_PS1']:+.3f} px，PS2 {e16['delta_h3_minus_h2']['MRE_PS2']:+.3f} px，FH1 {e16['delta_h3_minus_h2']['MRE_FH1']:+.3f} px，MRE_ALL {e16['delta_h3_minus_h2']['MRE_ALL']:+.3f} px，AoP MAE {e16['delta_h3_minus_h2']['aop_mae_deg']:+.3f}°。这里 FH1 与 AoP MAE 反而更高。

## 逐点回答

- PS1：H3 在 selected best、epoch 3、epoch 16 都低于 H2；但 H3 同时加入两个增强分支，不能把差值直接解释成 PS 模块的独立因果收益。
- PS2：selected best 与 epoch 16 更低，epoch 3 更高，方向不统一。
- FH1：selected best 与 epoch 3 更低，epoch 16 更高，方向不统一。
- MRE_ALL：三组对照均低于 H2，是这次运行最稳定的正向信号。
- AoP MAE：selected best 与 epoch 3 更低，epoch 16 更高，不能称为全程改善。
- train–validation gap：H3 的 MRE_ALL gap 相对 H2 在 epoch 3 扩大 {comparison['matched_gap_delta_h3_minus_h2']['epoch3']['MRE_ALL']:+.3f} px，在 epoch 16 缩小 {comparison['matched_gap_delta_h3_minus_h2']['epoch16']['MRE_ALL']:+.3f} px；没有统一收窄。

所以准确结论是：H3 的 selected best 和 MRE_ALL 结果支持继续研究，但三个关键点没有在所有对齐轮次上一致改善。当前只有 seed 42，不做显著性、稳定胜出或 SOTA 声明。

曲线见 [validation_metrics.png](curves/validation_metrics.png) 与 [h3_train_validation_gap.png](curves/h3_train_validation_gap.png)。旧 U-Net B2 仅保留为历史量级参照，架构与预算不同，不用于证明 HRNet 或 H3 的结构优劣。
"""


def _architecture_markdown(aggregate: Mapping[str, Any]) -> str:
    gate = aggregate["gates"]
    architecture = aggregate["architecture"]
    resources = aggregate["resources"]
    tiny = gate["four_sample_learning"]
    return f"""# Phase 1C：PS/FH 专业特征增强结构

## 已实现并验证的结构

H3 保留 H2 的 HRNet-W32 主干和 PS/FH 独立解码器，在 32 通道、128×128 的最终高分辨率特征之后加入两个专属分支。输入为 `[B,1,512,512]`，拼接顺序固定为 `[PS1,PS2,FH1]`，输出热图为 `[B,3,256,256]`，DSNT 坐标为 `[B,3,2]`。

PS 分支真实调用 `torchvision.ops.DeformConv2d`。3×3 预测卷积输出 27 通道，其中 18 通道是 offset、9 通道经 sigmoid 后作为 modulation mask；没有普通卷积回退。预测层零初始化，因此初始 offset 为 0，初始 mask 为 0.5。可变形卷积结果经 GELU，与 1×1 空间注意力逐元素相乘，再与输入残差相加并做通道 LayerNorm。合成 CUDA 探针确认 offset 形状为 `{gate['deformable_operator']['offset_shape']}`、mask 形状为 `{gate['deformable_operator']['mask_shape']}`，offset 预测器、mask 预测器和可变形卷积权重均有非零有限梯度。

FH 分支使用 ASPP-lite：1×1、dilation 3 和 dilation 6 三路卷积，各输出 32 通道，拼成 96 通道后用 1×1 卷积压回 32 通道；ASPP 内没有新增 BatchNorm。SE 分支为全局池化后 `32→8→32`，产生 `[B,32,1,1]` 通道权重。ASPP 与 SE 特征相加，再做输入残差和通道 LayerNorm。

LayerNorm2d 使用 NCHW→NHWC，对最后 32 通道调用真正的 LayerNorm，再转回 NCHW；没有用 BatchNorm 或 GroupNorm 代替。

## 初始化、公平性和分支测试边界

基础主干与两个解码器从同一份 seed 42 的未训练 H2 参数复制，数值起点相同且不共享参数存储；PS/FH 增强模块使用自己的初始化。因此基础部分可对齐，但完整 H3 函数从一开始就不与 H2 等价。

H2 有 {architecture['parameter_counts']['h2_split']:,} 个可训练参数，H3 有 {architecture['parameter_counts']['h3_specialized']:,} 个，增加 {architecture['parameter_counts']['h3_minus_h2']:,}；其中 PS 增强器 {architecture['parameter_counts']['ps_enhancer']:,}，FH 增强器 {architecture['parameter_counts']['fh_enhancer']:,}。

“只更新一个专属分支”的隔离检查限定在 `forward_ps` 与 `forward_fh` 两条诊断前向路径：PS 诊断不更新 FH 专属参数，FH 诊断反之。常规完整前向会计算两个分支，正式 B2 训练使用的也是常规完整前向；不能把诊断路径结论扩大成“任意切片损失都天然隔离”。

## 门禁与确定性边界

四样本门禁跑满 {tiny['steps']} 步并通过：PS1 {tiny['MRE_PS1']:.3f} px、PS2 {tiny['MRE_PS2']:.3f} px、FH1 {tiny['MRE_FH1']:.3f} px、MRE_ALL {tiny['MRE_ALL']:.3f} px、AoP MAE {tiny['aop_mae_deg']:.3f}°，AoP 有效率 4/4。第一次尝试没有通过梯度审计；随后只做了一次最小门禁修复，把共享主干的要求限定为当前配置路径实际参与反传的参数，同时继续要求所有 PS/FH 专属参数都有有限梯度。模型、损失、500 步上限和 5 px 阈值都没有改变。

CUDA 可变形卷积反向在当前环境中不能保证严格确定性。训练固定 seed、数据顺序并启用了 deterministic algorithms，但采用 warn-only；因此不声称位级复现。算子结构与梯度门禁是合成 CUDA 检查，四样本和 16 轮结果则来自真实有标签 train/validation 运行。

正式 H3 可训练参数为 {resources['trainable_parameters']:,}，峰值已分配显存 {resources['peak_memory_allocated_mib']:.1f} MiB，峰值保留显存 {resources['peak_memory_reserved_mib']:.1f} MiB；16 轮用时 {resources['formal_elapsed_minutes']:.1f} 分钟，未超过本轮 3 小时新增 GPU 上限。
"""


def _unlabeled_markdown() -> str:
    return """# Phase 1C：无标签数据接入状态

本轮没有下载无标签数据，也没有进行无标签或半监督训练。只读审计结论是：当前没有完整、可训练的无标签池，实际可用于训练的无标签文件数为 0；配置中的无标签目录不存在，默认数据根和环境覆盖也都没有可用设置。

发现的归档只完成约 21.816%，不能当作数据池使用。项目记录同时出现 31,421 与 31,121 两种总量口径，另有 2,045 的 Example 子集记录；目前没有证据说明该子集能直接相加，因此不能靠猜测确定最终数量。

来源层面可以追溯到 Kaggle 与 Zenodo 记录，但许可文本或签名证据仍有冲突/缺失，尚未达到可训练、可公开复现的接入条件。由于 testing 在本轮完全冻结，无法检查无标签候选与 testing 的重叠；因此“与全部有标签分区无重叠”目前仍是未知状态，而不是已经通过。

后续接入前至少需要完成：

1. 固定无标签目录结构与允许文件格式。
2. 解压后重新核对实际文件数，解释 31,421、31,121 与 2,045 三种记录之间的关系。
3. 做损坏图、尺寸与解码检查。
4. 在获准的数据边界内进行内容哈希去重，并检查与全部有标签分区的重叠。
5. 明确数据来源、许可版本、许可文件及可验证签名。
6. 记录最终纳入与排除规则，再进入 Phase 2 数据接入。

无标签池缺失不影响本轮 H3 的监督结构验证，但会阻塞后续真正的半监督训练。
"""


def _summary_markdown(aggregate: Mapping[str, Any]) -> str:
    best = aggregate["supervised_comparison"]["selected_best"]["h3_specialized"]
    tiny = aggregate["gates"]["four_sample_learning"]
    return f"""# Phase 1C 小结

这轮把导师方案里的两个专业增强模块接到了现有 H2 独立解码器上：PS 分支是真实可变形卷积、modulation mask 与空间注意力，FH 分支是 ASPP-lite 与 SE。结构、梯度、保存加载、四样本学习门禁和正式训练链路均已通过。

四样本跑满 500 步，MRE_ALL 为 {tiny['MRE_ALL']:.3f} px，AoP MAE 为 {tiny['aop_mae_deg']:.3f}°，有效率 4/4。第一次门禁尝试暴露的是梯度审计范围过宽，不是模型学习失败；一次最小修复后重新从未训练初始化运行，模型、损失、步数和阈值都没改。

正式 H3 跑满 16/16 轮，按“惩罚 AoP→MRE_ALL→较早轮次”选到 epoch {best['epoch']}：PS1 {best['MRE_PS1']:.3f}、PS2 {best['MRE_PS2']:.3f}、FH1 {best['MRE_FH1']:.3f}、MRE_ALL {best['MRE_ALL']:.3f} px，AoP MAE {best['aop_mae_deg']:.3f}°。selected best 相对 H2 的三个点与总体指标都更低；但 matched epoch 3 的 PS2 更高，matched epoch 16 的 FH1 与 AoP MAE 更高，所以不能写成“所有关键点稳定改善”。更准确的说法是：当前单 seed、16 轮 validation 结果支持继续研究。

训练固定 seed 42 与数据顺序，并启用了 deterministic algorithms；DeformConv2d CUDA backward 只能 warn-only，因此不声称位级复现。H3 比 H2 增加 40,420 个可训练参数，正式运行约 {aggregate['resources']['formal_elapsed_minutes']:.1f} 分钟，预算内完成。

这一阶段仍是 B2 增强监督工程参照，不是导师原文的纯 MSE，也没有 EMA 教师、伪标签、置信度机制、无标签一致性损失或半监督结论。testing 始终冻结，公开报告只有脱敏配置、聚合 train/validation 指标和曲线。

无标签审计显示当前完整可训练文件数为 0；总量口径、归档完整性、许可和分区重叠都还没闭环。详情见 [UNLABELED_INTAKE.md](UNLABELED_INTAKE.md)。结构说明见 [SPECIALIZED_ARCHITECTURE.md](SPECIALIZED_ARCHITECTURE.md)，逐点对照见 [SPECIALIZED_COMPARISON.md](SPECIALIZED_COMPARISON.md)。
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probe = _read_json(args.operator_probe)
    tiny = _read_json(args.tiny_result)
    review = _read_json(args.tiny_review)
    formal = _read_json(args.formal_result)
    key_metrics = _read_json(args.key_metrics)
    phase1b_raw = _read_json(args.phase1b_aggregate)
    config = _read_yaml(args.protocol_config)
    history = _read_history(args.train_log)

    _validate_config(config)
    _validate_operator(probe)
    _validate_tiny(tiny, review)
    _validate_determinism(tiny.get("determinism_policy", {}), context="tiny determinism")
    _validate_formal(formal, key_metrics, history)
    phase1b = _validate_phase1b(phase1b_raw)

    aggregate = _build_aggregate(
        probe, tiny, formal, history, phase1b, phase1b_raw
    )
    sanitized = _sanitized_config()
    reports = {
        "SPECIALIZED_ARCHITECTURE.md": _architecture_markdown(aggregate),
        "SPECIALIZED_COMPARISON.md": _comparison_markdown(aggregate),
        "PHASE1C_SUMMARY.md": _summary_markdown(aggregate),
        "UNLABELED_INTAKE.md": _unlabeled_markdown(),
    }
    aggregate_text = json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    config_text = yaml.safe_dump(sanitized, allow_unicode=True, sort_keys=False)
    _public_hygiene(aggregate_text, context="aggregate report")
    _public_hygiene(config_text, context="sanitized config")
    for name, text in reports.items():
        _public_hygiene(text, context=name)

    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "aggregate_results.json").write_text(
        aggregate_text, encoding="utf-8"
    )
    (args.report_root / "sanitized_config.yaml").write_text(
        config_text, encoding="utf-8"
    )
    for name, text in reports.items():
        (args.report_root / name).write_text(text, encoding="utf-8")
    _write_curves(args.report_root / "curves", history, phase1b)
    print(json.dumps({"status": "PASS", "report_root": "reports/phase1c"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
