#!/usr/bin/env python
"""Validate and publish the locked Phase 0.6 long-budget comparison.

The validator constructs exactly three registered run directories and opens an
explicit file allowlist in each. It never discovers sibling directories. Public
artifacts are rebuilt from a small whitelist that excludes losses, filesystem
paths, data fingerprints, environment details, and provenance digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase06_long_budget.yaml"
CANONICAL_RUN_ROOT = REPOSITORY_ROOT / "runs" / "phase06"
CANONICAL_REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase06"

PHASE = "phase0.6-long-budget-fidelity"
VARIANTS = ("B0", "B1", "B2")
SEED = 42
EPOCHS = 200
MILESTONES = (20, 50, 100, 150, 200)
CHECKPOINT_SELECTION = ("aop_mae_deg", "MRE_ALL", "earlier_epoch")
METRIC_NAMES = ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
METRIC_LABELS = {
    "MRE_PS1": "PS1 MRE (px)",
    "MRE_PS2": "PS2 MRE (px)",
    "MRE_FH1": "FH1 MRE (px)",
    "MRE_ALL": "Overall MRE (px)",
    "aop_mae_deg": "AoP MAE (deg)",
}
COUNT_NAMES = (
    "n_samples",
    "n_valid_aop",
    "n_evaluable_aop",
    "aop_invalid_prediction_count",
)
METRIC_PAYLOAD_FIELDS = {
    "total_loss",
    "heatmap_mse",
    "coordinate_smooth_l1",
    "distribution_js",
    *METRIC_NAMES,
    *COUNT_NAMES,
    "decoder",
    "aop_mae_valid_deg",
}
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
PUBLIC_TRAINING_FIELDS = (
    "input_size_hw",
    "heatmap_size_hw",
    "sigma_heatmap_px",
    "align_corners",
    "dsnt_temperature",
    "keypoint_order",
    "base_channels",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "num_workers",
)


@dataclass(frozen=True)
class ValidatedRun:
    variant: str
    config: dict[str, Any]
    result: dict[str, Any]
    history: tuple[dict[str, Any], ...]
    environment: dict[str, Any]
    order_records: tuple[dict[str, Any], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate three locked 200-epoch runs and publish validation-only reports."
    )
    parser.add_argument("--protocol", type=Path, default=CANONICAL_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=CANONICAL_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=CANONICAL_REPORT_ROOT)
    return parser


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_forbidden_split_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
            if normalized != "testing_frozen" and tokens & {"test", "testing"}:
                raise PermissionError(f"{context} contains a forbidden split-derived key: {key}")
            _reject_forbidden_split_keys(item, context=context)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_forbidden_split_keys(item, context=context)


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path.name}")
    _reject_forbidden_split_keys(loaded, context=path.name)
    return loaded


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path.name}")
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


def _close(actual: Any, expected: Any, *, context: str) -> None:
    actual_value = _finite(actual, context=context)
    expected_value = _finite(expected, context=context)
    if not math.isclose(actual_value, expected_value, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{context} differs between artifacts")


def _require_mre_identity(actual: Any, expected: Any, *, context: str) -> None:
    """Allow only realistic serialization-scale rounding in the three-point mean."""

    actual_value = _finite(actual, context=context)
    expected_value = _finite(expected, context=context)
    if not math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{context} is not the three-keypoint arithmetic mean")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    project = protocol.get("project")
    selection = protocol.get("selection")
    contract = protocol.get("data_contract")
    training = protocol.get("training")
    execution = protocol.get("execution")
    optimizer = protocol.get("optimizer")
    variants = protocol.get("variants")
    if not all(
        isinstance(item, Mapping)
        for item in (project, selection, contract, training, execution, optimizer, variants)
    ):
        raise ValueError("Phase 0.6 protocol mappings are incomplete")
    assert isinstance(project, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(contract, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(variants, Mapping)
    _require_equal(project.get("phase"), PHASE, context="protocol phase")
    _require_equal(project.get("testing_frozen"), True, context="protocol testing freeze")
    if not _is_hex(project.get("parent_phase05_commit"), length=40):
        raise ValueError("Protocol parent Phase 0.5 commit is invalid")
    _require_equal(project.get("parent_phase05_tag"), "phase05-v0.1.0", context="freeze tag")
    _require_equal(
        dict(selection),
        {
            "split": "validation",
            "common_decoder": "dsnt",
            "checkpoint_selection": list(CHECKPOINT_SELECTION),
            "milestones": list(MILESTONES),
        },
        context="selection protocol",
    )
    _require_equal(contract.get("allowed_splits"), ["train", "validation"], context="splits")
    _require_equal(set(contract.get("forbidden_splits", [])), {"test", "testing"}, context="freeze")
    for role, count, fh1 in (("train", 300, "FH1"), ("validation", 100, "AOP Tangency")):
        _require_equal(
            contract.get(role),
            {
                "sample_count": count,
                "fingerprint_required": True,
                "source_columns": {"PS1": "PS1", "PS2": "PS2", "FH1": fh1},
            },
            context=f"{role} contract",
        )
    _require_equal(training.get("seed"), SEED, context="protocol seed")
    _require_equal(training.get("epochs"), EPOCHS, context="protocol epochs")
    _require_equal(training.get("batch_size"), 1, context="protocol batch size")
    _require_equal(training.get("learning_rate"), 0.001, context="protocol learning rate")
    _require_equal(execution.get("variants"), list(VARIANTS), context="variant matrix")
    _require_equal(execution.get("seed"), SEED, context="execution seed")
    _require_equal(execution.get("run_all_epochs"), True, context="full-budget execution")
    _require_equal(
        execution.get("compare_total_loss_across_variants"),
        False,
        context="cross-variant objective comparison",
    )
    _require_equal(set(variants), set(VARIANTS), context="variant definitions")
    expected_weights = {"B0": [1.0, 0.0, 0.0], "B1": [1.0, 10.0, 0.0], "B2": [1.0, 10.0, 1.0]}
    for variant in VARIANTS:
        definition = variants[variant]
        if not isinstance(definition, Mapping):
            raise ValueError(f"Protocol definition for {variant} is invalid")
        _require_equal(
            definition.get("weights"), expected_weights[variant], context=f"{variant} weights"
        )
        _require_equal(
            definition.get("validation_decoders"), ["dsnt"], context=f"{variant} decoder"
        )


def _expected_training(protocol: Mapping[str, Any], variant: str) -> dict[str, Any]:
    training = dict(protocol["training"])
    weights = protocol["variants"][variant]["weights"]
    for key, value in zip(
        ("heatmap_loss_weight", "coordinate_loss_weight", "distribution_loss_weight"),
        weights,
        strict=True,
    ):
        training[key] = float(value)
    return training


def _validate_data(value: Any, *, protocol: Mapping[str, Any], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    _require_exact_keys(value, {"train", "validation"}, context=context)
    for role in ("train", "validation"):
        fingerprint = value[role]
        if not isinstance(fingerprint, dict):
            raise ValueError(f"{context}.{role} must be a mapping")
        _require_exact_keys(
            fingerprint,
            {"sample_count", "labels_sha256", "aggregate_sha256", "source_columns"},
            context=f"{context}.{role}",
        )
        _require_equal(
            fingerprint["sample_count"],
            protocol["data_contract"][role]["sample_count"],
            context=f"{context}.{role}.sample_count",
        )
        for key in ("labels_sha256", "aggregate_sha256"):
            if not _is_hex(fingerprint[key], length=64):
                raise ValueError(f"{context}.{role}.{key} is invalid")
        _require_equal(
            fingerprint["source_columns"],
            protocol["data_contract"][role]["source_columns"],
            context=f"{context}.{role}.source_columns",
        )
    return value


def _validate_metric_payload(
    value: Any,
    *,
    expected_samples: int,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a metric mapping")
    _require_exact_keys(value, METRIC_PAYLOAD_FIELDS, context=context)
    for name in (
        "total_loss",
        "heatmap_mse",
        "coordinate_smooth_l1",
        "distribution_js",
        *METRIC_NAMES,
        "aop_mae_valid_deg",
    ):
        _finite(value[name], context=f"{context}.{name}", nonnegative=True)
    _require_equal(value["decoder"], "dsnt", context=f"{context}.decoder")
    for name in COUNT_NAMES:
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError(f"{context}.{name} must be a non-negative integer")
    _require_equal(value["n_samples"], expected_samples, context=f"{context}.n_samples")
    if value["n_evaluable_aop"] > expected_samples:
        raise ValueError(f"{context}.n_evaluable_aop exceeds the sample count")
    _require_equal(
        value["n_valid_aop"] + value["aop_invalid_prediction_count"],
        value["n_evaluable_aop"],
        context=f"{context} AoP counts",
    )
    expected_mre = sum(float(value[name]) for name in METRIC_NAMES[:3]) / 3.0
    _require_mre_identity(value["MRE_ALL"], expected_mre, context=f"{context}.MRE_ALL")
    return value


def _csv_int(value: Any, *, context: str) -> int:
    converted = _finite(value, context=context)
    integer = int(converted)
    if converted != integer:
        raise ValueError(f"{context} must be an integer")
    return integer


def _history_metrics(
    row: Mapping[str, Any], *, expected_samples: int, context: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "total_loss": _finite(row["val_total_loss"], context=f"{context}.total", nonnegative=True),
        "heatmap_mse": _finite(
            row["val_heatmap_mse"], context=f"{context}.heatmap", nonnegative=True
        ),
        "coordinate_smooth_l1": _finite(
            row["val_coordinate_smooth_l1"], context=f"{context}.coordinate", nonnegative=True
        ),
        "distribution_js": _finite(
            row["val_distribution_js"], context=f"{context}.distribution", nonnegative=True
        ),
        **{
            name: _finite(row[f"val_{name}"], context=f"{context}.{name}", nonnegative=True)
            for name in METRIC_NAMES
        },
        "n_samples": _csv_int(row["val_n_samples"], context=f"{context}.n_samples"),
        "decoder": row["val_decoder"],
        "n_valid_aop": _csv_int(row["val_n_valid_aop"], context=f"{context}.n_valid"),
        "n_evaluable_aop": _csv_int(row["val_n_evaluable_aop"], context=f"{context}.n_evaluable"),
        "aop_invalid_prediction_count": _csv_int(
            row["val_aop_invalid_prediction_count"], context=f"{context}.n_invalid"
        ),
        "aop_mae_valid_deg": _finite(
            row["val_aop_mae_valid_deg"], context=f"{context}.aop_valid", nonnegative=True
        ),
    }
    return _validate_metric_payload(payload, expected_samples=expected_samples, context=context)


def _read_history(path: Path, *, expected_samples: int, context: str) -> tuple[dict[str, Any], ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_equal(
            tuple(reader.fieldnames or ()), HISTORY_COLUMNS, context=f"{context} columns"
        )
        raw_rows = list(reader)
    _require_equal(len(raw_rows), EPOCHS, context=f"{context} row count")
    rows: list[dict[str, Any]] = []
    for expected_epoch, raw in enumerate(raw_rows, start=1):
        epoch = _csv_int(raw["epoch"], context=f"{context}.epoch")
        _require_equal(epoch, expected_epoch, context=f"{context} epoch sequence")
        for name in (
            "train_time_sec",
            "validation_time_sec",
            "epoch_time_sec",
            "train_total_loss",
            "train_heatmap_mse",
            "train_coordinate_smooth_l1",
            "train_distribution_js",
        ):
            _finite(raw[name], context=f"{context}.epoch{epoch}.{name}", nonnegative=True)
        _require_equal(
            _csv_int(raw["train_batches"], context=f"{context}.epoch{epoch}.train_batches"),
            300,
            context=f"{context}.epoch{epoch}.train_batches",
        )
        metrics = _history_metrics(
            raw,
            expected_samples=expected_samples,
            context=f"{context}.epoch{epoch}",
        )
        rows.append(
            {
                "epoch": epoch,
                **{name: float(metrics[name]) for name in METRIC_NAMES},
                "_payload": metrics,
            }
        )
    return tuple(rows)


def _compare_payload(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    for name in METRIC_PAYLOAD_FIELDS:
        if name == "decoder" or name in COUNT_NAMES:
            _require_equal(actual[name], expected[name], context=f"{context}.{name}")
        else:
            _close(actual[name], expected[name], context=f"{context}.{name}")


def _validate_order(path: Path, *, context: str) -> tuple[dict[str, Any], ...]:
    ledger = _read_json(path)
    _require_exact_keys(
        ledger,
        {"schema_version", "contains_filenames", "records"},
        context=f"{context} order ledger",
    )
    _require_equal(ledger["schema_version"], 1, context=f"{context} order schema")
    _require_equal(ledger["contains_filenames"], False, context=f"{context} filename policy")
    records = ledger["records"]
    if not isinstance(records, list):
        raise ValueError(f"{context} order records must be a list")
    _require_equal(len(records), EPOCHS, context=f"{context} order epoch count")
    validated: list[dict[str, Any]] = []
    for expected_epoch, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{context} order record {expected_epoch} is invalid")
        _require_exact_keys(
            record,
            {"epoch", "sample_count", "filename_order_sha256"},
            context=f"{context} order epoch {expected_epoch}",
        )
        _require_equal(record["epoch"], expected_epoch, context=f"{context} order epoch")
        _require_equal(record["sample_count"], 300, context=f"{context} order sample count")
        if not _is_hex(record["filename_order_sha256"], length=64):
            raise ValueError(f"{context} order digest at epoch {expected_epoch} is invalid")
        validated.append(dict(record))
    return tuple(validated)


def _validate_environment(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    expected = {
        "platform",
        "python",
        "torch",
        "device",
        "cuda_available",
        "torch_cuda",
        "cudnn",
    }
    if value.get("device") != "cpu":
        expected.add("gpu")
    _require_exact_keys(value, expected, context=context)
    for name in ("platform", "python", "torch", "device"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"{context}.{name} must be a non-empty string")
    if type(value["cuda_available"]) is not bool:
        raise ValueError(f"{context}.cuda_available must be boolean")
    if "gpu" in value and (not isinstance(value["gpu"], str) or not value["gpu"]):
        raise ValueError(f"{context}.gpu must be a non-empty string")
    return dict(value)


def _validate_resources(value: Any, *, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} resources must be a mapping")
    _require_exact_keys(
        value,
        {
            "training_runtime_sec",
            "evaluation_runtime_sec",
            "peak_gpu_allocated_mb",
            "peak_gpu_reserved_mb",
        },
        context=f"{context} resources",
    )
    for name in ("training_runtime_sec", "evaluation_runtime_sec"):
        _finite(value[name], context=f"{context}.{name}", nonnegative=True)
    for name in ("peak_gpu_allocated_mb", "peak_gpu_reserved_mb"):
        if value[name] is not None:
            _finite(value[name], context=f"{context}.{name}", nonnegative=True)


def _validate_run(
    run_root: Path,
    *,
    variant: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> ValidatedRun:
    # Construct one registered directory and one explicit file allowlist. Do not glob.
    run_dir = run_root / variant / "seed_42"
    paths = {
        "config_yaml": run_dir / "config.yaml",
        "config_json": run_dir / "config.json",
        "result": run_dir / "phase06_result.json",
        "metrics": run_dir / "metrics.json",
        "history": run_dir / "train_log.csv",
        "environment": run_dir / "environment.json",
        "order": run_dir / "training_order.json",
        "best": run_dir / "best.pt",
        "last": run_dir / "last.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {variant}/seed_42 run: missing {missing}")

    config_yaml = _read_yaml(paths["config_yaml"])
    config_json = _read_json(paths["config_json"])
    _require_equal(config_yaml, config_json, context=f"{variant} config YAML/JSON")
    config = config_json
    result = _read_json(paths["result"])
    metrics_file = _read_json(paths["metrics"])
    environment = _validate_environment(
        _read_json(paths["environment"]), context=f"{variant} environment"
    )
    history = _read_history(
        paths["history"],
        expected_samples=int(protocol["data_contract"]["validation"]["sample_count"]),
        context=f"{variant} history",
    )
    order_records = _validate_order(paths["order"], context=variant)

    _require_exact_keys(
        config,
        {
            "schema_version",
            "phase",
            "variant",
            "variant_description",
            "training",
            "optimizer",
            "selection",
            "testing_frozen",
            "data",
            "model",
            "order_audit",
            "provenance",
        },
        context=f"{variant} config",
    )
    _require_exact_keys(
        result,
        {
            "status",
            "phase",
            "variant",
            "seed",
            "epochs_completed",
            "selection_split",
            "selection_decoder",
            "testing_frozen",
            "best_epoch",
            "best_validation_metrics",
            "milestone_validation_metrics",
            "last_validation_metrics",
            "order_audit",
            "resources",
            "model",
            "provenance",
        },
        context=f"{variant} result",
    )
    _require_exact_keys(
        metrics_file,
        {
            "status",
            "selection_split",
            "checkpoint_metric",
            "selection_tiebreak",
            "best_epoch",
            "best_value",
            "best_validation_metrics",
            "last_validation_metrics",
            "best_checkpoint",
            "last_checkpoint",
        },
        context=f"{variant} metrics",
    )

    for artifact, name in ((config, "config"), (result, "result")):
        _require_equal(artifact.get("phase"), PHASE, context=f"{variant} {name} phase")
        _require_equal(artifact.get("variant"), variant, context=f"{variant} {name} variant")
        _require_equal(
            artifact.get("testing_frozen"), True, context=f"{variant} {name} testing freeze"
        )
    _require_equal(config["schema_version"], 1, context=f"{variant} schema")
    _require_equal(
        config["variant_description"],
        protocol["variants"][variant]["description"],
        context=f"{variant} description",
    )
    _require_equal(
        config["training"],
        _expected_training(protocol, variant),
        context=f"{variant} training lock",
    )
    _require_equal(config["optimizer"], protocol["optimizer"], context=f"{variant} optimizer")
    _require_equal(
        config["selection"],
        {
            "split": "validation",
            "common_decoder": "dsnt",
            "checkpoint_selection": list(CHECKPOINT_SELECTION),
            "milestones": list(MILESTONES),
        },
        context=f"{variant} selection",
    )
    _validate_data(config["data"], protocol=protocol, context=f"{variant} data")
    _require_equal(
        config["order_audit"],
        {
            "algorithm": "RandomSampler-compatible randperm",
            "generator_seed": SEED,
            "per_epoch_filename_order_sha256": True,
        },
        context=f"{variant} order audit",
    )

    model = config["model"]
    if not isinstance(model, dict):
        raise ValueError(f"{variant} model identity is missing")
    _require_exact_keys(
        model,
        {"class", "trainable_parameters", "initialization_sha256"},
        context=f"{variant} model",
    )
    _require_equal(model["class"], "HeatmapUNet", context=f"{variant} model class")
    _require_equal(
        model["trainable_parameters"],
        484171,
        context=f"{variant} HeatmapUNet parameter count",
    )
    if not _is_hex(model["initialization_sha256"], length=64):
        raise ValueError(f"{variant} initialization digest is invalid")
    _require_equal(result["model"], model, context=f"{variant} result model")

    config_provenance = config["provenance"]
    result_provenance = result["provenance"]
    if not isinstance(config_provenance, dict) or not isinstance(result_provenance, dict):
        raise ValueError(f"{variant} provenance is missing")
    _require_exact_keys(
        config_provenance,
        {
            "protocol_sha256",
            "git_commit",
            "git_dirty",
            "parent_phase05_commit",
            "parent_phase05_tag",
        },
        context=f"{variant} config provenance",
    )
    _require_exact_keys(
        result_provenance,
        {
            "protocol_sha256",
            "git_commit",
            "git_dirty",
            "parent_phase05_commit",
            "parent_phase05_tag",
            "best_checkpoint_sha256",
        },
        context=f"{variant} result provenance",
    )
    expected_parent = protocol["project"]["parent_phase05_commit"]
    expected_tag = protocol["project"]["parent_phase05_tag"]
    for provenance, name in ((config_provenance, "config"), (result_provenance, "result")):
        _require_equal(
            provenance["protocol_sha256"], protocol_sha256, context=f"{variant} {name} protocol"
        )
        if not _is_hex(provenance["git_commit"], length=40):
            raise ValueError(f"{variant} {name} Git commit is invalid")
        _require_equal(provenance["git_dirty"], False, context=f"{variant} {name} Git state")
        _require_equal(
            provenance["parent_phase05_commit"], expected_parent, context=f"{variant} parent commit"
        )
        _require_equal(
            provenance["parent_phase05_tag"], expected_tag, context=f"{variant} parent tag"
        )
    _require_equal(
        result_provenance["git_commit"],
        config_provenance["git_commit"],
        context=f"{variant} recorded commit",
    )
    checkpoint_digest = result_provenance["best_checkpoint_sha256"]
    if not _is_hex(checkpoint_digest, length=64):
        raise ValueError(f"{variant} best checkpoint digest is invalid")
    _require_equal(
        _sha256_file(paths["best"]), checkpoint_digest, context=f"{variant} best checkpoint"
    )

    _require_equal(result["status"], "completed", context=f"{variant} status")
    _require_equal(result["seed"], SEED, context=f"{variant} seed")
    _require_equal(result["epochs_completed"], EPOCHS, context=f"{variant} completed epochs")
    _require_equal(result["selection_split"], "validation", context=f"{variant} split")
    _require_equal(result["selection_decoder"], "dsnt", context=f"{variant} decoder")
    _require_equal(
        result["order_audit"],
        {"recorded_epochs": EPOCHS, "samples_per_epoch": 300, "contains_filenames": False},
        context=f"{variant} result order audit",
    )
    expected_samples = int(protocol["data_contract"]["validation"]["sample_count"])
    best_row = min(
        history,
        key=lambda row: (row["aop_mae_deg"], row["MRE_ALL"], row["epoch"]),
    )
    best_epoch = int(best_row["epoch"])
    _require_equal(result["best_epoch"], best_epoch, context=f"{variant} best epoch")
    decoder_metrics = result["best_validation_metrics"]
    if not isinstance(decoder_metrics, dict):
        raise ValueError(f"{variant} best validation metrics must be a mapping")
    _require_exact_keys(decoder_metrics, {"dsnt"}, context=f"{variant} decoder metrics")
    result_best = _validate_metric_payload(
        decoder_metrics["dsnt"], expected_samples=expected_samples, context=f"{variant} best"
    )
    result_last = _validate_metric_payload(
        result["last_validation_metrics"],
        expected_samples=expected_samples,
        context=f"{variant} last",
    )
    _compare_payload(result_best, best_row["_payload"], context=f"{variant} best versus history")
    _compare_payload(result_last, history[-1]["_payload"], context=f"{variant} last versus history")

    milestones = result["milestone_validation_metrics"]
    if not isinstance(milestones, dict):
        raise ValueError(f"{variant} milestone metrics must be a mapping")
    _require_exact_keys(
        milestones,
        {*(str(epoch) for epoch in MILESTONES), "best"},
        context=f"{variant} milestone metrics",
    )
    for label in (*map(str, MILESTONES), "best"):
        entry = milestones[label]
        if not isinstance(entry, dict):
            raise ValueError(f"{variant} milestone {label} is invalid")
        expected_epoch = best_epoch if label == "best" else int(label)
        _require_exact_keys(entry, {"epoch", *METRIC_NAMES}, context=f"{variant} milestone {label}")
        _require_equal(entry["epoch"], expected_epoch, context=f"{variant} milestone {label} epoch")
        expected_row = history[expected_epoch - 1]
        for metric in METRIC_NAMES:
            _close(
                entry[metric], expected_row[metric], context=f"{variant} milestone {label}.{metric}"
            )

    expected_metrics_identity = {
        "status": "completed",
        "selection_split": "validation",
        "checkpoint_metric": "aop_mae_deg",
        "selection_tiebreak": list(CHECKPOINT_SELECTION),
        "best_epoch": best_epoch,
    }
    for key, expected in expected_metrics_identity.items():
        _require_equal(metrics_file[key], expected, context=f"{variant} metrics.{key}")
    _close(metrics_file["best_value"], best_row["aop_mae_deg"], context=f"{variant} best value")
    metrics_best = _validate_metric_payload(
        metrics_file["best_validation_metrics"],
        expected_samples=expected_samples,
        context=f"{variant} metrics best",
    )
    metrics_last = _validate_metric_payload(
        metrics_file["last_validation_metrics"],
        expected_samples=expected_samples,
        context=f"{variant} metrics last",
    )
    _compare_payload(metrics_best, result_best, context=f"{variant} metrics/result best")
    _compare_payload(metrics_last, result_last, context=f"{variant} metrics/result last")
    for field, file_key in (("best_checkpoint", "best"), ("last_checkpoint", "last")):
        reported = Path(str(metrics_file[field]))
        if not reported.is_absolute():
            reported = run_dir / reported
        _require_equal(
            reported.resolve(strict=True),
            paths[file_key].resolve(strict=True),
            context=f"{variant} metrics.{field}",
        )
    _validate_resources(result["resources"], context=variant)
    return ValidatedRun(
        variant=variant,
        config=config,
        result=result,
        history=history,
        environment=environment,
        order_records=order_records,
    )


def _without_objective_weights(training: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(training)
    for name in ("heatmap_loss_weight", "coordinate_loss_weight", "distribution_loss_weight"):
        result.pop(name)
    return result


def _validate_cross_run_identity(runs: Mapping[str, ValidatedRun]) -> None:
    _require_equal(set(runs), set(VARIANTS), context="run matrix")
    reference = runs["B0"]
    for variant in VARIANTS[1:]:
        run = runs[variant]
        _require_equal(run.config["data"], reference.config["data"], context=f"{variant} data")
        _require_equal(
            _without_objective_weights(run.config["training"]),
            _without_objective_weights(reference.config["training"]),
            context=f"{variant} non-objective training settings",
        )
        _require_equal(
            run.config["optimizer"], reference.config["optimizer"], context=f"{variant} optimizer"
        )
        _require_equal(
            run.config["selection"], reference.config["selection"], context=f"{variant} selection"
        )
        _require_equal(
            run.config["model"],
            reference.config["model"],
            context=f"{variant} model and initialization",
        )
        _require_equal(run.environment, reference.environment, context=f"{variant} environment")
        _require_equal(
            run.config["provenance"],
            reference.config["provenance"],
            context=f"{variant} provenance",
        )
        _require_equal(
            run.order_records, reference.order_records, context=f"{variant} training order"
        )


def _public_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in METRIC_NAMES}


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    return float(row["aop_mae_deg"]), float(row["MRE_ALL"]), int(row["epoch"])


def _best_row(run: ValidatedRun) -> Mapping[str, Any]:
    return min(run.history, key=_selection_key)


def _primary_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    comparisons = [float(left[name]) - float(right[name]) for name in ("aop_mae_deg", "MRE_ALL")]
    if all(value <= 0 for value in comparisons) and any(value < 0 for value in comparisons):
        return "better_or_equal_on_both"
    if all(value >= 0 for value in comparisons) and any(value > 0 for value in comparisons):
        return "worse_or_equal_on_both"
    if all(value == 0 for value in comparisons):
        return "equal_on_both"
    return "mixed"


def _relative_reduction(reference: float, candidate: float) -> float | None:
    if reference == 0:
        return None
    return (reference - candidate) / reference


def _earliest_matching_epoch(
    candidate: ValidatedRun, reference_best: Mapping[str, Any]
) -> int | None:
    for row in candidate.history:
        if float(row["aop_mae_deg"]) <= float(reference_best["aop_mae_deg"]) and float(
            row["MRE_ALL"]
        ) <= float(reference_best["MRE_ALL"]):
            return int(row["epoch"])
    return None


def _effect_diagnostic(candidate: ValidatedRun, reference: ValidatedRun) -> dict[str, Any]:
    candidate_best = _best_row(candidate)
    reference_best = _best_row(reference)
    relation = _primary_relation(candidate_best, reference_best)
    reductions = {
        name: _relative_reduction(float(reference_best[name]), float(candidate_best[name]))
        for name in ("aop_mae_deg", "MRE_ALL")
    }
    final_benefit = relation == "better_or_equal_on_both" and any(
        value is not None and value >= 0.05 for value in reductions.values()
    )
    earliest = _earliest_matching_epoch(candidate, reference_best)
    speed_benefit = earliest is not None and earliest < int(reference_best["epoch"])
    if final_benefit:
        classification = "mainly_final_performance"
    elif speed_benefit:
        classification = "mainly_convergence_speed"
    elif relation == "mixed":
        classification = "mixed_evidence"
    else:
        classification = "no_clear_advantage"
    return {
        "candidate": candidate.variant,
        "reference": reference.variant,
        "best_relation": relation,
        "best_metric_delta_candidate_minus_reference": {
            name: float(candidate_best[name]) - float(reference_best[name])
            for name in ("aop_mae_deg", "MRE_ALL")
        },
        "relative_reduction_at_best": reductions,
        "candidate_best_epoch": int(candidate_best["epoch"]),
        "reference_best_epoch": int(reference_best["epoch"]),
        "earliest_epoch_matching_reference_best_on_both_primary_metrics": earliest,
        "classification": classification,
        "descriptive_rule": (
            "final: best dominates both primary metrics and improves at least one by 5%; "
            "speed: reaches both reference-best thresholds before the reference best epoch"
        ),
    }


def _b0_post20(runs: Mapping[str, ValidatedRun]) -> dict[str, Any]:
    b0 = runs["B0"]
    epoch20 = b0.history[19]
    post20_best = min(b0.history[20:], key=_selection_key)
    reductions = {
        name: _relative_reduction(float(epoch20[name]), float(post20_best[name]))
        for name in ("aop_mae_deg", "MRE_ALL")
    }
    meaningful = (
        reductions["aop_mae_deg"] is not None
        and reductions["aop_mae_deg"] >= 0.05
        and float(post20_best["MRE_ALL"]) <= float(epoch20["MRE_ALL"])
    )
    return {
        "epoch20": {"epoch": 20, "metrics": _public_metrics(epoch20)},
        "best_after_epoch20": {
            "epoch": int(post20_best["epoch"]),
            "metrics": _public_metrics(post20_best),
        },
        "relative_reduction_from_epoch20": reductions,
        "clearly_improved_by_descriptive_rule": meaningful,
        "descriptive_rule": (
            "post-20 best AoP MAE is at least 5% lower than epoch 20 and MRE_ALL does not worsen"
        ),
    }


def _catch_comparison(runs: Mapping[str, ValidatedRun]) -> dict[str, Any]:
    b0_best = _best_row(runs["B0"])
    b0_endpoint = runs["B0"].history[-1]
    return {
        target: {
            "best_within_200_relation": _primary_relation(b0_best, _best_row(runs[target])),
            "epoch200_relation": _primary_relation(b0_endpoint, runs[target].history[-1]),
            "best_metric_delta_B0_minus_target": {
                name: float(b0_best[name]) - float(_best_row(runs[target])[name])
                for name in ("aop_mae_deg", "MRE_ALL")
            },
            "epoch200_metric_delta_B0_minus_target": {
                name: float(b0_endpoint[name]) - float(runs[target].history[-1][name])
                for name in ("aop_mae_deg", "MRE_ALL")
            },
        }
        for target in ("B1", "B2")
    }


def _relation_phrase(relation: str) -> str:
    if relation in {"better_or_equal_on_both", "equal_on_both"}:
        return "两项主指标均不高于对方，可视为追上"
    if relation == "worse_or_equal_on_both":
        return "两项主指标均更高，未追上"
    return "两项主指标方向不一致，不能说已经追上"


def _is_caught(relation: str) -> bool:
    return relation in {"better_or_equal_on_both", "equal_on_both"}


def _b0_target_conclusion(target: str, comparison: Mapping[str, Any]) -> str:
    best_relation = str(comparison["best_within_200_relation"])
    endpoint_relation = str(comparison["epoch200_relation"])
    sentence = (
        f"对 {target}，预算内best：{_relation_phrase(best_relation)}；"
        f"epoch 200端点：{_relation_phrase(endpoint_relation)}"
    )
    if _is_caught(best_relation) != _is_caught(endpoint_relation):
        sentence += "；两个口径结论不同，前者回答预算内最优能力，后者回答末轮状态"
    return sentence


def _effect_answer(name: str, diagnostic: Mapping[str, Any]) -> str:
    classification = diagnostic["classification"]
    earliest = diagnostic["earliest_epoch_matching_reference_best_on_both_primary_metrics"]
    if classification == "mainly_final_performance":
        extra = f"；并且第 {earliest} 轮已同时达到参照方案的最佳双指标" if earliest else ""
        return f"{name}在这次单种子检查里更像是改变最终性能{extra}。"
    if classification == "mainly_convergence_speed":
        return f"{name}主要表现为加快收敛：第 {earliest} 轮已达到参照方案的最佳双指标。"
    if classification == "mixed_evidence":
        return f"{name}在两项主指标上的方向不一致，当前不能归为只改变最终性能或只加速收敛。"
    return f"{name}没有显示出足以归为最终性能改善或收敛加速的清楚优势。"


def _build_conclusions(
    runs: Mapping[str, ValidatedRun],
    *,
    catch: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    js: Mapping[str, Any],
) -> list[dict[str, str | int]]:
    b0_phrases = [_b0_target_conclusion(target, catch[target]) for target in ("B1", "B2")]
    b0_best = _best_row(runs["B0"])
    return [
        {
            "number": 1,
            "question": "B0在200轮后是否追上B1或B2？",
            "answer": ("；".join(b0_phrases) + "。这是seed 42的描述性结果。"),
        },
        {
            "number": 2,
            "question": "坐标项主要改变最终性能，还是主要加快收敛？",
            "answer": _effect_answer("坐标项（B1相对B0）", coordinate)
            + "这里只基于一个seed，不作统计显著性判断。",
        },
        {
            "number": 3,
            "question": "JS项主要改变最终性能，还是主要加快收敛？",
            "answer": _effect_answer("JS项（B2相对B1）", js)
            + "这里只基于一个seed，不作统计显著性判断。",
        },
        {
            "number": 4,
            "question": "按老师原文采用纯MSE是否仍然具备可行性？",
            "answer": (
                f"仅在工程/复现层面的监督基线意义上具备可行性。B0完整跑完200轮并"
                f"得到有限的validation指标，其best在第 {int(b0_best['epoch'])} 轮；"
                "这说明方案可训练、可复现，但不代表达到任何临床或实用性能阈值，"
                "也不等于它在当前指标上优于增强版本。"
            ),
        },
        {
            "number": 5,
            "question": "B1/B2是否仍只能被称为增强监督基线？",
            "answer": (
                "是。B1/B2只是在同一有标注训练流程中增加坐标项或JS项，"
                "没有EMA教师、伪标签或无标注一致性目标，因此仍是增强监督基线。"
            ),
        },
    ]


def _build_aggregate(
    runs: Mapping[str, ValidatedRun], *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    public_runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        run = runs[variant]
        best = _best_row(run)
        checkpoints = {
            str(epoch): {"epoch": epoch, "metrics": _public_metrics(run.history[epoch - 1])}
            for epoch in MILESTONES
        }
        checkpoints["best"] = {
            "epoch": int(best["epoch"]),
            "metrics": _public_metrics(best),
        }
        public_runs.append(
            {
                "variant": variant,
                "seed": SEED,
                "best_epoch": int(best["epoch"]),
                "checkpoints": checkpoints,
            }
        )
    catch = _catch_comparison(runs)
    coordinate = _effect_diagnostic(runs["B1"], runs["B0"])
    js = _effect_diagnostic(runs["B2"], runs["B1"])
    post20 = _b0_post20(runs)
    conclusions = _build_conclusions(
        runs,
        catch=catch,
        coordinate=coordinate,
        js=js,
    )
    return {
        "schema_version": 1,
        "phase": PHASE,
        "scope": "supervised train/validation long-budget fidelity check",
        "selection_split": "validation",
        "testing_frozen": True,
        "metric_units": {name: METRIC_LABELS[name] for name in METRIC_NAMES},
        "lower_is_better": list(METRIC_NAMES),
        "protocol": {
            "seed": SEED,
            "epochs": EPOCHS,
            "milestones": list(MILESTONES),
            "checkpoint_selection": list(CHECKPOINT_SELECTION),
            "training": {name: protocol["training"][name] for name in PUBLIC_TRAINING_FIELDS},
            "optimizer": dict(protocol["optimizer"]),
            "data_sample_counts": {"train": 300, "validation": 100},
            "variants": {
                variant: {
                    "description": protocol["variants"][variant]["description"],
                }
                for variant in VARIANTS
            },
        },
        "model": {
            "class": runs["B0"].config["model"]["class"],
            "trainable_parameters": runs["B0"].config["model"]["trainable_parameters"],
        },
        "runs": public_runs,
        "comparisons": {
            "B0_after_epoch20": post20,
            "B0_against_augmented": catch,
            "coordinate_term_B1_against_B0": coordinate,
            "JS_term_B2_against_B1": js,
        },
        "conclusions": conclusions,
        "integrity": {
            "validated_run_count": 3,
            "epochs_per_run": EPOCHS,
            "all_runs_share_data_identity": True,
            "all_runs_share_model_initialization": True,
            "all_runs_share_environment": True,
            "all_epoch_training_orders_match": True,
            "all_best_checkpoints_verified": True,
            "all_best_tuples_recomputed": True,
        },
        "limitations": [
            "validation-only; testing was not read or evaluated",
            "one seed; descriptive comparison only and no significance claim",
            "the same validation split selected and reported each best checkpoint",
            "not the semi-supervised GeoEqui-LD system",
        ],
    }


def _assert_public_payload_sanitized(payload: Any, *, context: str = "aggregate") -> None:
    forbidden_key_fragments = ("loss", "path", "hash", "fingerprint", "commit")
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).casefold()
            if "sha256" in normalized or any(
                fragment in normalized for fragment in forbidden_key_fragments
            ):
                raise ValueError(f"{context} contains non-public key {key!r}")
            _assert_public_payload_sanitized(value, context=f"{context}.{key}")
    elif isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            _assert_public_payload_sanitized(value, context=f"{context}[{index}]")
    elif isinstance(payload, str) and (_is_hex(payload, length=40) or _is_hex(payload, length=64)):
        raise ValueError(f"{context} contains a non-public digest")


def _format(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_epoch(value: int | None) -> str:
    return "未达到" if value is None else f"epoch {value}"


def _best_table(aggregate: Mapping[str, Any]) -> str:
    lines = [
        "| 方案 | best epoch | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in aggregate["runs"]:
        metrics = run["checkpoints"]["best"]["metrics"]
        lines.append(
            "| {variant} | {epoch} | {ps1} | {ps2} | {fh1} | {mre} | {aop} |".format(
                variant=run["variant"],
                epoch=run["best_epoch"],
                ps1=_format(metrics["MRE_PS1"]),
                ps2=_format(metrics["MRE_PS2"]),
                fh1=_format(metrics["MRE_FH1"]),
                mre=_format(metrics["MRE_ALL"]),
                aop=_format(metrics["aop_mae_deg"]),
            )
        )
    return "\n".join(lines)


def _milestone_table(run: Mapping[str, Any]) -> str:
    lines = [
        "| 位置 | epoch | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (*map(str, MILESTONES), "best"):
        entry = run["checkpoints"][label]
        metrics = entry["metrics"]
        display = "best" if label == "best" else f"epoch {label}"
        lines.append(
            "| {display} | {epoch} | {ps1} | {ps2} | {fh1} | {mre} | {aop} |".format(
                display=display,
                epoch=entry["epoch"],
                ps1=_format(metrics["MRE_PS1"]),
                ps2=_format(metrics["MRE_PS2"]),
                fh1=_format(metrics["MRE_FH1"]),
                mre=_format(metrics["MRE_ALL"]),
                aop=_format(metrics["aop_mae_deg"]),
            )
        )
    return "\n".join(lines)


def _conclusion_list(aggregate: Mapping[str, Any]) -> str:
    return "\n\n".join(
        f"{item['number']}. **{item['question']}**  \n{item['answer']}"
        for item in aggregate["conclusions"]
    )


def _render_summary(aggregate: Mapping[str, Any]) -> str:
    post20 = aggregate["comparisons"]["B0_after_epoch20"]
    reduction = post20["relative_reduction_from_epoch20"]
    improvement = (
        "达到预设的描述性改善口径"
        if post20["clearly_improved_by_descriptive_rule"]
        else "未达到预设的描述性改善口径"
    )
    return f"""# Phase 0.6 长预算检查小结

这轮没有改模型，也没有加新目标，只把 B0/B1/B2 在同一 seed、同一初始化和同一
数据顺序下完整跑到 200 轮。所有 checkpoint 都只按 validation 的 AoP MAE、
MRE_ALL、较早 epoch 依次选择；testing 没有读取或评估。

## 最佳 validation 结果

{_best_table(aggregate)}

MRE 单位是原图像素，AoP MAE 单位是度，均为越低越好。三种方案的 total loss
定义不同，因此本报告不横向比较 total loss。

## B0 在 20 轮以后

B0 从 epoch 20 到其 20 轮后的最佳点（epoch
{post20["best_after_epoch20"]["epoch"]}），AoP MAE 相对下降
{_format(100 * reduction["aop_mae_deg"] if reduction["aop_mae_deg"] is not None else None)}%，
MRE_ALL 相对下降
{_format(100 * reduction["MRE_ALL"] if reduction["MRE_ALL"] is not None else None)}%；
按“前者至少下降 5%，且后者不恶化”的事先写明口径，{improvement}。

## 五个结论

{_conclusion_list(aggregate)}

## 边界

这里只有 seed 42，一切比较都是描述性的，不作显著性结论。同一 validation
既用于选择 best，也用于报告指标，不是独立 holdout。B1/B2 仍只是增强监督基线，
本轮没有 HRNet、PS/FH 解耦、EMA、伪标签或半监督目标。

- [200 轮 validation 曲线](curves/validation_metrics.png)
- [逐里程碑记录](LONG_BUDGET_COMPARISON.md)
- [脱敏配置](sanitized_config.yaml)
- [机器可读汇总](aggregate_results.json)
"""


def _render_detailed(aggregate: Mapping[str, Any]) -> str:
    run_sections = "\n\n".join(
        f"### {run['variant']}\n\n{_milestone_table(run)}" for run in aggregate["runs"]
    )
    post20 = aggregate["comparisons"]["B0_after_epoch20"]
    coordinate = aggregate["comparisons"]["coordinate_term_B1_against_B0"]
    js = aggregate["comparisons"]["JS_term_B2_against_B1"]
    return f"""# Phase 0.6：200 轮监督忠实性比较

## 固定条件

| 项目 | 设置 |
|---|---|
| 数据 | train 300，validation 100 |
| 模型 | HeatmapUNet，{aggregate["model"]["trainable_parameters"]:,} 个可训练参数 |
| 随机性 | seed 42；三方案初始化、逐轮样本顺序和运行环境完全一致 |
| 预算 | batch size 1，200 epochs，Adam，lr 0.001 |
| 方案 | B0=MSE；B1=MSE+coordinate SmoothL1；B2=B1+JS |
| checkpoint | validation AoP MAE → MRE_ALL → 较早 epoch |
| 数据边界 | 只读 train/validation；testing 冻结 |

这里不比较三方案的 total loss 绝对值。

## epoch 20/50/100/150/200 与 best

{run_sections}

## B0 的 20 轮后变化

- epoch 20：MRE_ALL={_format(post20["epoch20"]["metrics"]["MRE_ALL"])} px，
  AoP MAE={_format(post20["epoch20"]["metrics"]["aop_mae_deg"])}°。
- 20 轮后的最佳点在 epoch {post20["best_after_epoch20"]["epoch"]}：
  MRE_ALL={_format(post20["best_after_epoch20"]["metrics"]["MRE_ALL"])} px，
  AoP MAE={_format(post20["best_after_epoch20"]["metrics"]["aop_mae_deg"])}°。
- “明显改善”仅作描述性判断：20 轮后的最佳 AoP MAE 至少低 5%，同时 MRE_ALL
  不恶化。本次判断为 **{"是" if post20["clearly_improved_by_descriptive_rule"] else "否"}**。

## 收敛与最终性能的拆分

- 坐标项：B1 best epoch={coordinate["candidate_best_epoch"]}，B0 best
  epoch={coordinate["reference_best_epoch"]}；B1 首次同时达到 B0 best 的 AoP MAE 与
  MRE_ALL 阈值：
  {_format_epoch(coordinate["earliest_epoch_matching_reference_best_on_both_primary_metrics"])}。
- JS 项：B2 best epoch={js["candidate_best_epoch"]}，B1 best
  epoch={js["reference_best_epoch"]}；B2 首次同时达到 B1 best 的两项阈值：
  {_format_epoch(js["earliest_epoch_matching_reference_best_on_both_primary_metrics"])}。

这里把“最终性能改变”定义为：best 在两项主指标上均不差，且至少一项相对改善 5%；
把“加快收敛”定义为：增强方案早于参照方案的 best epoch，同时达到参照 best 的
两项阈值。这只是透明的描述性口径，不是统计检验。

## 五个结论

{_conclusion_list(aggregate)}

## validation 曲线

![B0/B1/B2 validation curves](curves/validation_metrics.png)

图中只有 MRE_PS1、MRE_PS2、MRE_FH1、MRE_ALL 和 AoP MAE，没有绘制跨方案
total loss。

## 完整性与限制

- 三个运行均恰好 200 轮，epoch 连续为 1–200，五项 validation 指标逐轮有限。
- MRE_ALL 已按三个关键点 MRE 的算术均值逐轮复算。
- best 已按 AoP MAE、MRE_ALL、较早 epoch 的三元组从 CSV 重算，并与
  result、metrics 和 checkpoint 记录交叉核对。
- 三个运行的数据身份、模型初始化、代码版本、协议、环境以及 200 轮训练顺序一致；
  公开文件不包含这些内部摘要值、原始路径或数据指纹。
- testing 没有读取、选择或评估。
- 单 seed 不能支持显著性或稳定性结论；同一 validation 同时用于模型选择和报告。
- 本轮不是完整 GeoEqui-LD，没有 HRNet、解耦、EMA、伪标签或半监督目标。
"""


def _plot_validation_curves(runs: Mapping[str, ValidatedRun], destination: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=160, sharex=True)
    colors = {"B0": "#4C78A8", "B1": "#F58518", "B2": "#54A24B"}
    for axis, metric in zip(axes.flat, METRIC_NAMES, strict=False):
        for variant in VARIANTS:
            run = runs[variant]
            axis.plot(
                [row["epoch"] for row in run.history],
                [row[metric] for row in run.history],
                label=variant,
                color=colors[variant],
                linewidth=1.25,
            )
        for milestone in MILESTONES:
            axis.axvline(milestone, color="#BBBBBB", alpha=0.18, linewidth=0.7)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylabel("px" if metric != "aop_mae_deg" else "degrees")
        axis.grid(alpha=0.22)
    axes.flat[-1].set_visible(False)
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("epoch")
    axes.flat[0].legend(ncol=3)
    figure.suptitle("Phase 0.6 validation metrics (seed 42)", y=0.995)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)


def _sanitized_config(aggregate: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase": aggregate["phase"],
        "scope": aggregate["scope"],
        "selection_split": "validation",
        "testing_frozen": True,
        "data_sample_counts": aggregate["protocol"]["data_sample_counts"],
        "seed": SEED,
        "model": aggregate["model"],
        "training": aggregate["protocol"]["training"],
        "optimizer": aggregate["protocol"]["optimizer"],
        "checkpoint_selection": aggregate["protocol"]["checkpoint_selection"],
        "milestones": list(MILESTONES),
        "variants": {
            variant: {
                "description": protocol["variants"][variant]["description"],
                "weights": protocol["variants"][variant]["weights"],
            }
            for variant in VARIANTS
        },
    }


def _write_reports(
    aggregate: Mapping[str, Any],
    runs: Mapping[str, ValidatedRun],
    *,
    protocol: Mapping[str, Any],
    report_root: Path,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = report_root / "aggregate_results.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_root / "PHASE06_SUMMARY.md").write_text(_render_summary(aggregate), encoding="utf-8")
    (report_root / "LONG_BUDGET_COMPARISON.md").write_text(
        _render_detailed(aggregate), encoding="utf-8"
    )
    sanitized = _sanitized_config(aggregate, protocol)
    _assert_public_payload_sanitized(sanitized, context="sanitized config")
    (report_root / "sanitized_config.yaml").write_text(
        yaml.safe_dump(sanitized, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _plot_validation_curves(runs, report_root / "curves" / "validation_metrics.png")


def summarize_phase06(
    *,
    protocol_path: Path,
    run_root: Path,
    report_root: Path,
) -> dict[str, Any]:
    protocol = _read_yaml(protocol_path)
    _validate_protocol(protocol)
    protocol_sha256 = _sha256_file(protocol_path)
    runs = {
        variant: _validate_run(
            run_root,
            variant=variant,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )
        for variant in VARIANTS
    }
    _validate_cross_run_identity(runs)
    aggregate = _build_aggregate(runs, protocol=protocol)
    _assert_public_payload_sanitized(aggregate)
    _write_reports(aggregate, runs, protocol=protocol, report_root=report_root)
    return aggregate


def _require_canonical_path(actual: Path, expected: Path, *, context: str) -> None:
    if actual.resolve(strict=True) != expected.resolve(strict=True):
        raise PermissionError(f"{context} must use the canonical repository path")


def _validate_commit_object(commit: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    _require_equal(result.stdout.strip(), "commit", context="recorded Git object type")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_canonical_path(args.protocol, CANONICAL_PROTOCOL, context="Protocol")
    _require_canonical_path(args.run_root, CANONICAL_RUN_ROOT, context="Run root")
    if args.report_root.resolve(strict=False) != CANONICAL_REPORT_ROOT.resolve(strict=False):
        raise PermissionError("Report root must be reports/phase06")
    first_config = _read_yaml(CANONICAL_RUN_ROOT / "B0" / "seed_42" / "config.yaml")
    _validate_commit_object(str(first_config.get("provenance", {}).get("git_commit", "")))
    aggregate = summarize_phase06(
        protocol_path=args.protocol,
        run_root=args.run_root,
        report_root=args.report_root,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "validated_run_count": aggregate["integrity"]["validated_run_count"],
                "reports": [
                    "reports/phase06/PHASE06_SUMMARY.md",
                    "reports/phase06/LONG_BUDGET_COMPARISON.md",
                    "reports/phase06/aggregate_results.json",
                    "reports/phase06/sanitized_config.yaml",
                    "reports/phase06/curves/validation_metrics.png",
                ],
                "testing_frozen": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
