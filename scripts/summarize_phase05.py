#!/usr/bin/env python
"""Validate and publish the fixed Phase 0.5 validation-only ablation matrix.

The validator deliberately opens only the nine pre-registered run directories.
It never discovers or reads any other split or run directory, and the public
outputs are assembled from an explicit aggregate whitelist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase05_ablation.yaml"
CANONICAL_RUN_ROOT = REPOSITORY_ROOT / "runs" / "phase05"
CANONICAL_REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase05"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.training.config import SupervisedTrainingConfig  # noqa: E402

PHASE = "phase0.5-supervised-ablation"
SCREENING_SEED = 42
CONFIRMATION_SEEDS = (43, 44, 45)
SCREENING_VARIANTS = ("B0", "B1", "B2")
CONFIRMATION_VARIANTS = ("B1", "B2")
RUN_MATRIX = (
    *((variant, SCREENING_SEED) for variant in SCREENING_VARIANTS),
    *((variant, seed) for seed in CONFIRMATION_SEEDS for variant in CONFIRMATION_VARIANTS),
)
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
RESOURCE_NAMES = (
    "training_runtime_sec",
    "evaluation_runtime_sec",
    "peak_gpu_allocated_mb",
    "peak_gpu_reserved_mb",
)
LOSS_COMPONENT_NAMES = (
    "heatmap_mse",
    "coordinate_smooth_l1",
    "distribution_js",
)
CHECKPOINT_SELECTION = ("aop_mae_deg", "MRE_ALL", "earlier_epoch")
RETENTION_RULE = ("aop_mae_deg", "MRE_ALL", "simpler_objective")
HEX_DIGEST_LENGTH = 64
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


@dataclass(frozen=True)
class ValidatedRun:
    variant: str
    seed: int
    config: dict[str, Any]
    result: dict[str, Any]
    history: tuple[dict[str, Any], ...]
    result_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the fixed nine-run Phase 0.5 matrix and publish aggregate "
            "validation-only reports."
        )
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


def _is_digest(value: Any, *, length: int = HEX_DIGEST_LENGTH) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(character in "0123456789abcdef" for character in value)


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


def _read_json_mapping(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path.name}")
    _reject_forbidden_split_keys(loaded, context=path.name)
    return loaded


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path.name}")
    _reject_forbidden_split_keys(loaded, context=path.name)
    return loaded


def _require_equal(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch: expected {expected!r}, got {actual!r}")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _require_finite(value: Any, *, context: str, nonnegative: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be finite")
    if nonnegative and converted < 0:
        raise ValueError(f"{context} must be non-negative")
    return converted


def _require_close(actual: Any, expected: Any, *, context: str) -> None:
    actual_value = _require_finite(actual, context=context)
    expected_value = _require_finite(expected, context=context)
    if not math.isclose(actual_value, expected_value, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{context} differs between artifacts")


def _validate_protocol(protocol: Mapping[str, Any]) -> SupervisedTrainingConfig:
    project = protocol.get("project")
    selection = protocol.get("selection")
    data_contract = protocol.get("data_contract")
    variants = protocol.get("variants")
    if not all(
        isinstance(value, Mapping) for value in (project, selection, data_contract, variants)
    ):
        raise ValueError("Protocol is missing a required mapping")
    assert isinstance(project, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(data_contract, Mapping)
    assert isinstance(variants, Mapping)
    _require_exact_keys(
        protocol,
        {"project", "selection", "data_contract", "training", "variants"},
        context="protocol",
    )
    _require_exact_keys(
        project,
        {"name", "phase", "parent_phase0_commit", "parent_phase0_tag", "testing_frozen"},
        context="protocol.project",
    )
    _require_exact_keys(
        selection,
        {
            "split",
            "common_decoder",
            "checkpoint_selection",
            "variant_retention",
            "first_round_seed",
            "confirmation_seeds",
            "retain_after_first_round",
        },
        context="protocol.selection",
    )
    _require_exact_keys(
        data_contract,
        {"allowed_splits", "forbidden_splits", "train", "validation"},
        context="protocol.data_contract",
    )

    _require_equal(project.get("phase"), PHASE, context="protocol phase")
    _require_equal(project.get("testing_frozen"), True, context="protocol testing policy")
    _require_equal(selection.get("split"), "validation", context="selection split")
    _require_equal(selection.get("common_decoder"), "dsnt", context="selection decoder")
    _require_equal(
        tuple(selection.get("checkpoint_selection", ())),
        CHECKPOINT_SELECTION,
        context="checkpoint selection",
    )
    _require_equal(
        tuple(selection.get("variant_retention", ())),
        RETENTION_RULE,
        context="variant retention rule",
    )
    _require_equal(selection.get("first_round_seed"), 42, context="screening seed")
    _require_equal(
        tuple(selection.get("confirmation_seeds", ())),
        CONFIRMATION_SEEDS,
        context="confirmation seeds",
    )
    _require_equal(selection.get("retain_after_first_round"), 2, context="retained count")

    _require_equal(
        tuple(data_contract.get("allowed_splits", ())),
        ("train", "validation"),
        context="allowed data splits",
    )
    _require_equal(
        set(data_contract.get("forbidden_splits", ())),
        {"test", "testing"},
        context="frozen data splits",
    )
    expected_data = {
        "train": (300, {"PS1": "PS1", "PS2": "PS2", "FH1": "FH1"}),
        "validation": (
            100,
            {"PS1": "PS1", "PS2": "PS2", "FH1": "AOP Tangency"},
        ),
    }
    for split, (sample_count, source_columns) in expected_data.items():
        contract = data_contract.get(split)
        if not isinstance(contract, Mapping):
            raise ValueError(f"Missing {split} data contract")
        _require_exact_keys(
            contract,
            {"sample_count", "fingerprint_required", "source_columns"},
            context=f"protocol.data_contract.{split}",
        )
        _require_equal(contract.get("sample_count"), sample_count, context=f"{split} count")
        _require_equal(
            contract.get("fingerprint_required"),
            True,
            context=f"{split} fingerprint policy",
        )
        _require_equal(
            contract.get("source_columns"),
            source_columns,
            context=f"{split} source columns",
        )

    expected_variants = {
        "B0": ((1.0, 0.0, 0.0), ("dsnt", "argmax")),
        "B1": ((1.0, 10.0, 0.0), ("dsnt",)),
        "B2": ((1.0, 10.0, 1.0), ("dsnt",)),
    }
    _require_equal(set(variants), set(expected_variants), context="protocol variants")
    for variant, (weights, decoders) in expected_variants.items():
        declaration = variants[variant]
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Invalid declaration for {variant}")
        _require_exact_keys(
            declaration,
            {"description", "weights", "validation_decoders"},
            context=f"protocol.variants.{variant}",
        )
        _require_equal(
            tuple(float(value) for value in declaration.get("weights", ())),
            weights,
            context=f"{variant} weights",
        )
        _require_equal(
            tuple(declaration.get("validation_decoders", ())),
            decoders,
            context=f"{variant} decoders",
        )

    training = protocol.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Protocol is missing its training mapping")
    training_config = SupervisedTrainingConfig.from_mapping(training)
    _require_exact_keys(
        training,
        set(training_config.to_dict()),
        context="protocol.training",
    )
    expected_training = {
        "seed": 42,
        "device": "auto",
        "deterministic": True,
        "input_size_hw": (512, 512),
        "heatmap_size_hw": (256, 256),
        "sigma_heatmap_px": 4.0,
        "align_corners": True,
        "dsnt_temperature": 0.05,
        "keypoint_order": ("PS1", "PS2", "FH1"),
        "aop_vertex_index": 0,
        "aop_pubic_axis_other_index": 1,
        "aop_fetal_head_index": 2,
        "base_channels": 8,
        "batch_size": 1,
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "heatmap_loss_weight": 1.0,
        "coordinate_loss_weight": 10.0,
        "distribution_loss_weight": 1.0,
        "max_grad_norm": 5.0,
        "num_workers": 0,
        "checkpoint_metric": "aop_mae_deg",
    }
    actual_training = training_config.to_dict()
    for name, expected in expected_training.items():
        actual = actual_training[name]
        if isinstance(expected, tuple):
            actual = tuple(actual)
        _require_equal(actual, expected, context=f"training.{name}")
    return training_config


def _validate_selection(
    selection: Mapping[str, Any],
    *,
    protocol_sha256: str,
) -> str:
    _require_exact_keys(
        selection,
        {
            "schema_version",
            "phase",
            "testing_frozen",
            "git_commit",
            "protocol_sha256",
            "screening_seed",
            "confirmation_seeds",
            "rule",
            "selected_variants",
            "input_result_sha256",
        },
        context="selection manifest",
    )
    expected = {
        "schema_version": 1,
        "phase": PHASE,
        "testing_frozen": True,
        "screening_seed": SCREENING_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "rule": list(RETENTION_RULE),
    }
    for key, value in expected.items():
        _require_equal(selection.get(key), value, context=f"selection.{key}")
    _require_equal(
        selection.get("protocol_sha256"),
        protocol_sha256,
        context="selection protocol digest",
    )
    selected = selection.get("selected_variants")
    if not isinstance(selected, list) or len(selected) != 2 or set(selected) != {"B1", "B2"}:
        raise ValueError("Selection manifest must retain exactly B1 and B2")
    commit = selection.get("git_commit")
    if not _is_digest(commit, length=40):
        raise ValueError("Selection manifest has an invalid Git commit")
    input_digests = selection.get("input_result_sha256")
    if not isinstance(input_digests, Mapping) or set(input_digests) != set(SCREENING_VARIANTS):
        raise ValueError("Selection manifest has an invalid screening-result digest map")
    if not all(_is_digest(value) for value in input_digests.values()):
        raise ValueError("Selection manifest contains an invalid result digest")
    return str(commit)


def _expected_training(
    protocol_training: Mapping[str, Any],
    protocol_variants: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    expected = dict(protocol_training)
    expected["seed"] = seed
    weights = protocol_variants[variant]["weights"]
    expected["heatmap_loss_weight"] = float(weights[0])
    expected["coordinate_loss_weight"] = float(weights[1])
    expected["distribution_loss_weight"] = float(weights[2])
    return expected


def _validate_data_identity(
    data: Any,
    *,
    protocol: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {"train", "validation"}:
        raise ValueError(f"{context} must contain only train and validation identities")
    data_contract = protocol["data_contract"]
    for split in ("train", "validation"):
        identity = data[split]
        if not isinstance(identity, dict):
            raise ValueError(f"{context}.{split} must be a mapping")
        _require_exact_keys(
            identity,
            {"sample_count", "labels_sha256", "aggregate_sha256", "source_columns"},
            context=f"{context}.{split}",
        )
        _require_equal(
            identity.get("sample_count"),
            data_contract[split]["sample_count"],
            context=f"{context}.{split}.sample_count",
        )
        _require_equal(
            identity.get("source_columns"),
            data_contract[split]["source_columns"],
            context=f"{context}.{split}.source_columns",
        )
        for name in ("labels_sha256", "aggregate_sha256"):
            if not _is_digest(identity.get(name)):
                raise ValueError(f"{context}.{split}.{name} is not a SHA-256 digest")
    return data


def _validate_metric_payload(
    metrics: Any,
    *,
    decoder: str,
    expected_samples: int,
    context: str,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError(f"{context} must be a mapping")
    _require_exact_keys(metrics, METRIC_PAYLOAD_FIELDS, context=context)
    _require_equal(metrics.get("decoder"), decoder, context=f"{context}.decoder")
    for name in ("total_loss", *LOSS_COMPONENT_NAMES):
        _require_finite(metrics.get(name), context=f"{context}.{name}", nonnegative=True)
    for name in METRIC_NAMES:
        _require_finite(metrics.get(name), context=f"{context}.{name}", nonnegative=True)
    expected_mre_all = statistics.fmean(float(metrics[name]) for name in METRIC_NAMES[:3])
    if not math.isclose(
        float(metrics["MRE_ALL"]),
        expected_mre_all,
        rel_tol=1e-6,
        abs_tol=1e-5,
    ):
        raise ValueError(f"{context}.MRE_ALL is inconsistent with the three keypoint MREs")
    _require_finite(
        metrics.get("aop_mae_valid_deg"),
        context=f"{context}.aop_mae_valid_deg",
        nonnegative=True,
    )
    if float(metrics["aop_mae_deg"]) > 180.0 or float(metrics["aop_mae_valid_deg"]) > 180.0:
        raise ValueError(f"{context} has an AoP MAE outside [0, 180] degrees")
    for name in COUNT_NAMES:
        if type(metrics.get(name)) is not int:  # bool is intentionally rejected
            raise ValueError(f"{context}.{name} must be an integer")
    _require_equal(metrics["n_samples"], expected_samples, context=f"{context}.n_samples")
    _require_equal(
        metrics["n_evaluable_aop"],
        expected_samples,
        context=f"{context}.n_evaluable_aop",
    )
    valid = metrics["n_valid_aop"]
    evaluable = metrics["n_evaluable_aop"]
    invalid = metrics["aop_invalid_prediction_count"]
    if not 0 <= valid <= evaluable or invalid != evaluable - valid:
        raise ValueError(f"{context} has inconsistent AoP counts")
    expected_penalized = (valid * float(metrics["aop_mae_valid_deg"]) + invalid * 180.0) / evaluable
    if not math.isclose(
        float(metrics["aop_mae_deg"]),
        expected_penalized,
        rel_tol=1e-7,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{context} has inconsistent valid-only and penalized AoP MAE")
    return metrics


def _csv_int(value: Any, *, context: str) -> int:
    converted = _require_finite(value, context=context)
    if not converted.is_integer():
        raise ValueError(f"{context} must be an integer")
    return int(converted)


def _history_metric(
    row: Mapping[str, Any], *, context: str, expected_samples: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "total_loss": _require_finite(
            row["val_total_loss"], context=f"{context}.val_total_loss"
        ),
        **{
            name: _require_finite(row[f"val_{name}"], context=f"{context}.val_{name}")
            for name in LOSS_COMPONENT_NAMES
        },
        **{
            name: _require_finite(row[f"val_{name}"], context=f"{context}.val_{name}")
            for name in METRIC_NAMES
        },
    }
    payload["aop_mae_valid_deg"] = _require_finite(
        row["val_aop_mae_valid_deg"],
        context=f"{context}.val_aop_mae_valid_deg",
    )
    payload.update(
        {
            name: _csv_int(row[f"val_{name}"], context=f"{context}.val_{name}")
            for name in COUNT_NAMES
        }
    )
    payload["decoder"] = row["val_decoder"]
    return _validate_metric_payload(
        payload,
        decoder="dsnt",
        expected_samples=expected_samples,
        context=context,
    )


def _read_and_validate_history(
    path: Path,
    *,
    expected_samples: int,
    loss_weights: tuple[float, float, float],
    context: str,
) -> tuple[dict[str, Any], ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 20:
        raise ValueError(f"{context} must contain exactly 20 epoch rows")
    if tuple(reader.fieldnames or ()) != HISTORY_COLUMNS:
        raise ValueError(f"{context} must use the exact registered CSV column set")
    for expected_epoch, row in enumerate(rows, start=1):
        _require_equal(
            _csv_int(row["epoch"], context=f"{context}.epoch"),
            expected_epoch,
            context=f"{context}.epoch",
        )
        train_time = _require_finite(
            row["train_time_sec"],
            context=f"{context}.train_time_sec",
            nonnegative=True,
        )
        validation_time = _require_finite(
            row["validation_time_sec"],
            context=f"{context}.validation_time_sec",
            nonnegative=True,
        )
        epoch_time = _require_finite(
            row["epoch_time_sec"],
            context=f"{context}.epoch_time_sec",
            nonnegative=True,
        )
        if epoch_time + 1e-6 < train_time + validation_time:
            raise ValueError(f"{context} has an inconsistent epoch runtime")
        _require_equal(
            _csv_int(row["train_batches"], context=f"{context}.train_batches"),
            300,
            context=f"{context}.train_batches",
        )
        for split in ("train", "val"):
            components = tuple(
                _require_finite(
                    row[f"{split}_{name}"],
                    context=f"{context}.{split}_{name}",
                    nonnegative=True,
                )
                for name in LOSS_COMPONENT_NAMES
            )
            total = _require_finite(
                row[f"{split}_total_loss"],
                context=f"{context}.{split}_total_loss",
                nonnegative=True,
            )
            expected_total = sum(
                weight * component
                for weight, component in zip(loss_weights, components, strict=True)
            )
            if not math.isclose(total, expected_total, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError(f"{context} has an inconsistent {split} total loss")
            for name, weight, component in zip(
                LOSS_COMPONENT_NAMES,
                loss_weights,
                components,
                strict=True,
            ):
                if weight == 0.0 and not math.isclose(component, 0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"{context}.{split}_{name} must be zero when its weight is zero"
                    )
        _history_metric(
            row, context=f"{context}.epoch_{expected_epoch}", expected_samples=expected_samples
        )
    return tuple(rows)


def _compare_metric_artifacts(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for name in ("total_loss", *LOSS_COMPONENT_NAMES, *METRIC_NAMES, "aop_mae_valid_deg"):
        _require_close(actual.get(name), expected.get(name), context=f"{context}.{name}")
    for name in (*COUNT_NAMES, "decoder"):
        _require_equal(actual.get(name), expected.get(name), context=f"{context}.{name}")


def _validate_resources(
    resources: Any,
    *,
    history: Sequence[Mapping[str, Any]],
    context: str,
) -> dict[str, float]:
    if not isinstance(resources, dict) or set(resources) != set(RESOURCE_NAMES):
        raise ValueError(f"{context} must contain exactly the registered resource fields")
    normalized = {
        name: _require_finite(resources[name], context=f"{context}.{name}", nonnegative=True)
        for name in RESOURCE_NAMES
    }
    if normalized["training_runtime_sec"] <= 0 or normalized["evaluation_runtime_sec"] <= 0:
        raise ValueError(f"{context} runtimes must be positive")
    if normalized["peak_gpu_reserved_mb"] < normalized["peak_gpu_allocated_mb"]:
        raise ValueError(f"{context} GPU reserved memory is smaller than allocated memory")
    logged_runtime = sum(float(row["epoch_time_sec"]) for row in history)
    if normalized["training_runtime_sec"] + 1e-6 < logged_runtime:
        raise ValueError(f"{context} training runtime is shorter than the logged epochs")
    return normalized


def _validate_run(
    run_root: Path,
    *,
    variant: str,
    seed: int,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    commit: str,
) -> ValidatedRun:
    # Deliberately construct one allowed directory; never discover sibling directories.
    run_dir = run_root / variant / f"seed_{seed}"
    paths = {
        "config_yaml": run_dir / "config.yaml",
        "config_json": run_dir / "config.json",
        "result": run_dir / "phase05_result.json",
        "metrics": run_dir / "metrics.json",
        "history": run_dir / "train_log.csv",
        "best": run_dir / "best.pt",
        "last": run_dir / "last.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {variant}/seed_{seed} run: missing {missing}")

    config_yaml = _read_yaml_mapping(paths["config_yaml"])
    config_json = _read_json_mapping(paths["config_json"])
    _require_equal(config_yaml, config_json, context=f"{variant}/seed_{seed} config JSON/YAML")
    config = config_json
    result = _read_json_mapping(paths["result"])
    metrics_file = _read_json_mapping(paths["metrics"])
    _require_exact_keys(
        config,
        {
            "schema_version",
            "phase",
            "variant",
            "variant_description",
            "training",
            "selection",
            "testing_frozen",
            "data",
            "model",
            "provenance",
        },
        context=f"{variant}/seed_{seed} config",
    )
    _require_exact_keys(
        result,
        {
            "status",
            "phase",
            "variant",
            "seed",
            "selection_split",
            "selection_decoder",
            "testing_frozen",
            "best_epoch",
            "best_validation_metrics",
            "last_validation_metrics",
            "resources",
            "model",
            "provenance",
        },
        context=f"{variant}/seed_{seed} result",
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
        context=f"{variant}/seed_{seed} metrics",
    )
    expected_samples = int(protocol["data_contract"]["validation"]["sample_count"])
    history = _read_and_validate_history(
        paths["history"],
        expected_samples=expected_samples,
        loss_weights=tuple(float(value) for value in protocol["variants"][variant]["weights"]),
        context=f"{variant}/seed_{seed} history",
    )

    expected_identity = {
        "phase": PHASE,
        "variant": variant,
        "testing_frozen": True,
    }
    for key, expected in expected_identity.items():
        _require_equal(config.get(key), expected, context=f"config.{key}")
        _require_equal(result.get(key), expected, context=f"result.{key}")
    _require_equal(config.get("schema_version"), 1, context="config.schema_version")
    _require_equal(result.get("status"), "completed", context="result.status")
    _require_equal(result.get("seed"), seed, context="result.seed")
    _require_equal(result.get("selection_split"), "validation", context="result split")
    _require_equal(result.get("selection_decoder"), "dsnt", context="result decoder")
    _require_equal(
        config.get("variant_description"),
        protocol["variants"][variant]["description"],
        context="variant description",
    )
    _require_equal(
        config.get("training"),
        _expected_training(
            protocol["training"],
            protocol["variants"],
            variant=variant,
            seed=seed,
        ),
        context=f"{variant}/seed_{seed} training config",
    )
    _require_equal(
        config.get("selection"),
        {
            "split": "validation",
            "common_decoder": "dsnt",
            "checkpoint_selection": list(CHECKPOINT_SELECTION),
        },
        context=f"{variant}/seed_{seed} selection config",
    )
    _validate_data_identity(
        config.get("data"),
        protocol=protocol,
        context=f"{variant}/seed_{seed}.data",
    )

    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{variant}/seed_{seed} model identity is missing")
    _require_exact_keys(
        model,
        {"class", "trainable_parameters", "initialization_sha256"},
        context=f"{variant}/seed_{seed} model",
    )
    _require_equal(model.get("class"), "HeatmapUNet", context="model class")
    if type(model.get("trainable_parameters")) is not int or model["trainable_parameters"] <= 0:
        raise ValueError("Model trainable-parameter count must be a positive integer")
    if not _is_digest(model.get("initialization_sha256")):
        raise ValueError("Model initialization digest is invalid")
    _require_equal(result.get("model"), model, context=f"{variant}/seed_{seed} result model")

    config_provenance = config.get("provenance")
    result_provenance = result.get("provenance")
    if not isinstance(config_provenance, dict) or not isinstance(result_provenance, dict):
        raise ValueError(f"{variant}/seed_{seed} provenance is missing")
    _require_exact_keys(
        config_provenance,
        {"protocol_sha256", "git_commit", "git_dirty"},
        context=f"{variant}/seed_{seed} config provenance",
    )
    _require_exact_keys(
        result_provenance,
        {"protocol_sha256", "git_commit", "git_dirty", "best_checkpoint_sha256"},
        context=f"{variant}/seed_{seed} result provenance",
    )
    for provenance, context in (
        (config_provenance, "config provenance"),
        (result_provenance, "result provenance"),
    ):
        _require_equal(
            provenance.get("protocol_sha256"),
            protocol_sha256,
            context=f"{context}.protocol_sha256",
        )
        _require_equal(provenance.get("git_commit"), commit, context=f"{context}.git_commit")
        _require_equal(provenance.get("git_dirty"), False, context=f"{context}.git_dirty")
    checkpoint_digest = result_provenance.get("best_checkpoint_sha256")
    if not _is_digest(checkpoint_digest):
        raise ValueError(f"{variant}/seed_{seed} checkpoint digest is invalid")
    _require_equal(
        _sha256_file(paths["best"]),
        checkpoint_digest,
        context=f"{variant}/seed_{seed} best checkpoint digest",
    )

    decoder_metrics = result.get("best_validation_metrics")
    expected_decoders = {"dsnt", "argmax"} if variant == "B0" else {"dsnt"}
    if not isinstance(decoder_metrics, dict) or set(decoder_metrics) != expected_decoders:
        raise ValueError(f"{variant}/seed_{seed} has an unexpected decoder metric set")
    for decoder, payload in decoder_metrics.items():
        _validate_metric_payload(
            payload,
            decoder=decoder,
            expected_samples=expected_samples,
            context=f"{variant}/seed_{seed}.{decoder}",
        )
    _validate_metric_payload(
        result.get("last_validation_metrics"),
        decoder="dsnt",
        expected_samples=expected_samples,
        context=f"{variant}/seed_{seed}.last",
    )

    best_row = min(
        history,
        key=lambda row: (
            float(row["val_aop_mae_deg"]),
            float(row["val_MRE_ALL"]),
            int(row["epoch"]),
        ),
    )
    best_epoch = int(best_row["epoch"])
    _require_equal(result.get("best_epoch"), best_epoch, context="result.best_epoch")
    _compare_metric_artifacts(
        decoder_metrics["dsnt"],
        _history_metric(
            best_row,
            context=f"{variant}/seed_{seed}.best_history",
            expected_samples=expected_samples,
        ),
        context=f"{variant}/seed_{seed}.best",
    )
    _compare_metric_artifacts(
        result["last_validation_metrics"],
        _history_metric(
            history[-1],
            context=f"{variant}/seed_{seed}.last_history",
            expected_samples=expected_samples,
        ),
        context=f"{variant}/seed_{seed}.last",
    )

    expected_metrics_identity = {
        "status": "completed",
        "selection_split": "validation",
        "checkpoint_metric": "aop_mae_deg",
        "selection_tiebreak": list(CHECKPOINT_SELECTION),
        "best_epoch": best_epoch,
    }
    for key, expected in expected_metrics_identity.items():
        _require_equal(metrics_file.get(key), expected, context=f"metrics.json.{key}")
    _require_close(
        metrics_file.get("best_value"),
        best_row["val_aop_mae_deg"],
        context="metrics.json.best_value",
    )
    metrics_best = _validate_metric_payload(
        metrics_file.get("best_validation_metrics"),
        decoder="dsnt",
        expected_samples=expected_samples,
        context=f"{variant}/seed_{seed}.metrics_best",
    )
    metrics_last = _validate_metric_payload(
        metrics_file.get("last_validation_metrics"),
        decoder="dsnt",
        expected_samples=expected_samples,
        context=f"{variant}/seed_{seed}.metrics_last",
    )
    _compare_metric_artifacts(
        metrics_best,
        decoder_metrics["dsnt"],
        context=f"{variant}/seed_{seed}.metrics_best",
    )
    _compare_metric_artifacts(
        metrics_last,
        result["last_validation_metrics"],
        context=f"{variant}/seed_{seed}.metrics_last",
    )
    for field, expected_name in (("best_checkpoint", "best.pt"), ("last_checkpoint", "last.pt")):
        reported = Path(str(metrics_file.get(field, "")))
        if not reported.is_absolute():
            reported = run_dir / reported
        _require_equal(
            reported.resolve(strict=True),
            paths["best" if field == "best_checkpoint" else "last"].resolve(strict=True),
            context=f"metrics.json.{field}",
        )
        _require_equal(reported.name, expected_name, context=f"metrics.json.{field} name")

    _validate_resources(
        result.get("resources"),
        history=history,
        context=f"{variant}/seed_{seed}.resources",
    )
    return ValidatedRun(
        variant=variant,
        seed=seed,
        config=config,
        result=result,
        history=history,
        result_sha256=_sha256_file(paths["result"]),
    )


def _validate_cross_run_identity(
    runs: Mapping[tuple[str, int], ValidatedRun],
    *,
    selection: Mapping[str, Any],
) -> None:
    reference = runs[("B0", SCREENING_SEED)]
    reference_data = reference.config["data"]
    reference_model = reference.config["model"]
    for (variant, seed), run in runs.items():
        _require_equal(run.config["data"], reference_data, context=f"{variant}/seed_{seed} data")
        _require_equal(
            run.config["model"]["class"],
            reference_model["class"],
            context=f"{variant}/seed_{seed} model class",
        )
        _require_equal(
            run.config["model"]["trainable_parameters"],
            reference_model["trainable_parameters"],
            context=f"{variant}/seed_{seed} parameter count",
        )

    for seed in (SCREENING_SEED, *CONFIRMATION_SEEDS):
        variants = SCREENING_VARIANTS if seed == SCREENING_SEED else CONFIRMATION_VARIANTS
        initialization_digests = {
            runs[(variant, seed)].config["model"]["initialization_sha256"] for variant in variants
        }
        if len(initialization_digests) != 1:
            raise ValueError(f"Seed {seed} does not share one model initialization")

    per_seed_initialization = {
        seed: runs[("B0" if seed == SCREENING_SEED else "B1", seed)].config["model"][
            "initialization_sha256"
        ]
        for seed in (SCREENING_SEED, *CONFIRMATION_SEEDS)
    }
    if len(set(per_seed_initialization.values())) != len(per_seed_initialization):
        raise ValueError("Different registered seeds must use different model initializations")

    input_digests = selection["input_result_sha256"]
    for variant in SCREENING_VARIANTS:
        _require_equal(
            input_digests[variant],
            runs[(variant, SCREENING_SEED)].result_sha256,
            context=f"selection input digest for {variant}",
        )
    complexity = {variant: index for index, variant in enumerate(SCREENING_VARIANTS)}
    ranked = sorted(
        SCREENING_VARIANTS,
        key=lambda variant: (
            runs[(variant, SCREENING_SEED)].result["best_validation_metrics"]["dsnt"][
                "aop_mae_deg"
            ],
            runs[(variant, SCREENING_SEED)].result["best_validation_metrics"]["dsnt"]["MRE_ALL"],
            complexity[variant],
        ),
    )
    _require_equal(
        selection["selected_variants"],
        ranked[:2],
        context="selection result versus pre-registered ranking",
    )


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def _public_counts(metrics: Mapping[str, Any]) -> dict[str, int | float]:
    evaluable = int(metrics["n_evaluable_aop"])
    invalid = int(metrics["aop_invalid_prediction_count"])
    return {
        "n_samples": int(metrics["n_samples"]),
        "n_evaluable_aop": evaluable,
        "n_valid_aop": int(metrics["n_valid_aop"]),
        "aop_invalid_prediction_count": invalid,
        "aop_invalid_prediction_rate": invalid / evaluable,
    }


def _public_resources(resources: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(resources[name]) for name in RESOURCE_NAMES}


def _public_run(run: ValidatedRun) -> dict[str, Any]:
    metrics = run.result["best_validation_metrics"]["dsnt"]
    return {
        "variant": run.variant,
        "seed": run.seed,
        "best_epoch": int(run.result["best_epoch"]),
        "metrics": _public_metrics(metrics),
        "aop_counts": _public_counts(metrics),
        "resources": _public_resources(run.result["resources"]),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("Sample standard deviation requires at least two values")
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
    }


def _build_aggregate(
    runs: Mapping[tuple[str, int], ValidatedRun],
    *,
    protocol: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    screening_runs = [
        _public_run(runs[(variant, SCREENING_SEED)]) for variant in SCREENING_VARIANTS
    ]
    confirmation_runs = [
        _public_run(runs[(variant, seed)])
        for seed in CONFIRMATION_SEEDS
        for variant in CONFIRMATION_VARIANTS
    ]
    confirmation_summary: dict[str, Any] = {}
    for variant in CONFIRMATION_VARIANTS:
        confirmation_summary[variant] = {
            name: _summary(
                [
                    float(runs[(variant, seed)].result["best_validation_metrics"]["dsnt"][name])
                    for seed in CONFIRMATION_SEEDS
                ]
            )
            for name in METRIC_NAMES
        }
    paired_delta: dict[str, Any] = {}
    for name in METRIC_NAMES:
        by_seed = [
            {
                "seed": seed,
                "value": float(
                    runs[("B2", seed)].result["best_validation_metrics"]["dsnt"][name]
                    - runs[("B1", seed)].result["best_validation_metrics"]["dsnt"][name]
                ),
            }
            for seed in CONFIRMATION_SEEDS
        ]
        paired_delta[name] = {
            "by_seed": by_seed,
            **_summary([entry["value"] for entry in by_seed]),
        }

    b0_metrics = runs[("B0", SCREENING_SEED)].result["best_validation_metrics"]
    b0_decoder_comparison = {
        decoder: {
            "metrics": _public_metrics(b0_metrics[decoder]),
            "aop_counts": _public_counts(b0_metrics[decoder]),
        }
        for decoder in ("dsnt", "argmax")
    }
    public_training_fields = (
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
    return {
        "schema_version": 1,
        "phase": PHASE,
        "scope": "supervised train/validation methodology audit",
        "selection_split": "validation",
        "testing_frozen": True,
        "metric_units": {name: METRIC_LABELS[name] for name in METRIC_NAMES},
        "lower_is_better": list(METRIC_NAMES),
        "protocol": {
            "screening_seed": SCREENING_SEED,
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "checkpoint_selection": list(CHECKPOINT_SELECTION),
            "variant_retention": list(RETENTION_RULE),
            "training": {name: protocol["training"][name] for name in public_training_fields},
            "data_sample_counts": {"train": 300, "validation": 100},
            "variants": {
                variant: {
                    "description": protocol["variants"][variant]["description"],
                    "weights": protocol["variants"][variant]["weights"],
                }
                for variant in SCREENING_VARIANTS
            },
        },
        "model": {
            "class": runs[("B0", SCREENING_SEED)].config["model"]["class"],
            "trainable_parameters": runs[("B0", SCREENING_SEED)].config["model"][
                "trainable_parameters"
            ],
        },
        "screening": {
            "seed": SCREENING_SEED,
            "runs": screening_runs,
            "selected_variants": list(selection["selected_variants"]),
            "B0_decoder_comparison": b0_decoder_comparison,
        },
        "confirmation": {
            "seeds": list(CONFIRMATION_SEEDS),
            "runs": confirmation_runs,
            "mean_and_sample_sd": confirmation_summary,
            "paired_delta_B2_minus_B1": paired_delta,
        },
        "integrity": {
            "validated_run_count": len(RUN_MATRIX),
            "epochs_per_run": 20,
            "all_runs_share_data_identity": True,
            "same_seed_variants_share_initialization": True,
            "all_best_checkpoints_verified": True,
            "all_aop_counts_consistent": True,
        },
        "limitations": [
            "validation-only; testing was not read or evaluated",
            (
                "the same validation split selected each best epoch and supplied the "
                "reported metrics; it is not an independent holdout"
            ),
            "three confirmation seeds; no significance claim",
            "subject/group metadata was not available for a group-level split audit",
            "not the full semi-supervised GeoEqui-LD system",
        ],
    }


def _format_number(value: float) -> str:
    return f"{value:.3f}"


def _format_mean_sd(summary: Mapping[str, float]) -> str:
    return f"{summary['mean']:.3f} ± {summary['sample_sd']:.3f}"


def _run_table(runs: Sequence[Mapping[str, Any]], *, resources: bool = False) -> str:
    if resources:
        header = (
            "| 版本 | seed | 最佳轮次 | 训练(s) | 复评(s) | "
            "显存 allocated/reserved (MB) |\n"
            "|---|---:|---:|---:|---:|---:|"
        )
        rows = [
            "| {variant} | {seed} | {best_epoch} | {train:.1f} | {evaluation:.1f} | "
            "{allocated:.1f} / {reserved:.1f} |".format(
                variant=run["variant"],
                seed=run["seed"],
                best_epoch=run["best_epoch"],
                train=run["resources"]["training_runtime_sec"],
                evaluation=run["resources"]["evaluation_runtime_sec"],
                allocated=run["resources"]["peak_gpu_allocated_mb"],
                reserved=run["resources"]["peak_gpu_reserved_mb"],
            )
            for run in runs
        ]
    else:
        header = (
            "| 版本 | seed | 最佳轮次 | PS1 MRE | PS2 MRE | FH1 MRE | 总体 MRE | "
            "AoP MAE | 无效 AoP |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        rows = []
        for run in runs:
            metrics = run["metrics"]
            counts = run["aop_counts"]
            rows.append(
                "| {variant} | {seed} | {best_epoch} | {ps1:.3f} | {ps2:.3f} | "
                "{fh1:.3f} | {mre:.3f} | {aop:.3f}° | {invalid}/{evaluable} |".format(
                    variant=run["variant"],
                    seed=run["seed"],
                    best_epoch=run["best_epoch"],
                    ps1=metrics["MRE_PS1"],
                    ps2=metrics["MRE_PS2"],
                    fh1=metrics["MRE_FH1"],
                    mre=metrics["MRE_ALL"],
                    aop=metrics["aop_mae_deg"],
                    invalid=counts["aop_invalid_prediction_count"],
                    evaluable=counts["n_evaluable_aop"],
                )
            )
    return "\n".join((header, *rows))


def _confirmation_table(summary: Mapping[str, Any]) -> str:
    lines = [
        "| 版本 | PS1 MRE | PS2 MRE | FH1 MRE | 总体 MRE | AoP MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in CONFIRMATION_VARIANTS:
        values = summary[variant]
        lines.append(
            "| {variant} | {ps1} | {ps2} | {fh1} | {mre} | {aop}° |".format(
                variant=variant,
                ps1=_format_mean_sd(values["MRE_PS1"]),
                ps2=_format_mean_sd(values["MRE_PS2"]),
                fh1=_format_mean_sd(values["MRE_FH1"]),
                mre=_format_mean_sd(values["MRE_ALL"]),
                aop=_format_mean_sd(values["aop_mae_deg"]),
            )
        )
    return "\n".join(lines)


def _paired_table(paired: Mapping[str, Any]) -> str:
    lines = [
        "| 指标 | seed 43 | seed 44 | seed 45 | 平均 ± 样本SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in METRIC_NAMES:
        item = paired[name]
        values = [entry["value"] for entry in item["by_seed"]]
        lines.append(
            "| {label} | {v0:.3f} | {v1:.3f} | {v2:.3f} | {mean:.3f} ± {sd:.3f} |".format(
                label=METRIC_LABELS[name],
                v0=values[0],
                v1=values[1],
                v2=values[2],
                mean=item["mean"],
                sd=item["sample_sd"],
            )
        )
    return "\n".join(lines)


def _decoder_table(comparison: Mapping[str, Any]) -> str:
    lines = [
        "| 解码方式 | PS1 MRE | PS2 MRE | FH1 MRE | 总体 MRE | AoP MAE | 无效 AoP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for decoder in ("dsnt", "argmax"):
        item = comparison[decoder]
        metrics = item["metrics"]
        counts = item["aop_counts"]
        lines.append(
            "| {decoder} | {ps1:.3f} | {ps2:.3f} | {fh1:.3f} | {mre:.3f} | "
            "{aop:.3f}° | {invalid}/{evaluable} |".format(
                decoder=decoder,
                ps1=metrics["MRE_PS1"],
                ps2=metrics["MRE_PS2"],
                fh1=metrics["MRE_FH1"],
                mre=metrics["MRE_ALL"],
                aop=metrics["aop_mae_deg"],
                invalid=counts["aop_invalid_prediction_count"],
                evaluable=counts["n_evaluable_aop"],
            )
        )
    return "\n".join(lines)


def _result_interpretation(aggregate: Mapping[str, Any]) -> str:
    paired = aggregate["confirmation"]["paired_delta_B2_minus_B1"]
    mre_delta = float(paired["MRE_ALL"]["mean"])
    aop_delta = float(paired["aop_mae_deg"]["mean"])
    if mre_delta < 0 and aop_delta < 0:
        return (
            "三个新种子的配对结果里，B2 的总体 MRE 和 AoP MAE 平均都低于 B1；"
            "这支持 JS 项在当前监督基线中的作用，但只有 3 个种子，先不作显著性结论。"
        )
    if mre_delta > 0 and aop_delta > 0:
        return (
            "三个新种子的配对结果里，B2 的总体 MRE 和 AoP MAE 平均都高于 B1；"
            "当前结果没有显示 JS 项带来稳定收益。"
        )
    return (
        "三个新种子的配对结果在总体 MRE 和 AoP MAE 上方向不一致，"
        "因此当前只能说 JS 项的影响是混合的。"
    )


def _render_summary(aggregate: Mapping[str, Any]) -> str:
    screening = aggregate["screening"]
    confirmation = aggregate["confirmation"]
    return f"""# Phase 0.5 监督基线小结

这轮先只看一个问题：同一套小型 U-Net、同一份训练/验证数据和同样的 20 轮预算下，
给热图 MSE 逐步加上坐标 SmoothL1 与分布 JS，关键点定位会有什么变化。
这里没有接入半监督分支，也没有读取 testing。

## 运行设置

- 数据：train 300 张、validation 100 张；模型参数量 {aggregate["model"]["trainable_parameters"]:,}。
- seed 42 用于三种损失的首轮筛选；按预先写好的 AoP MAE → 总体 MRE →
  简单目标规则保留 {", ".join(screening["selected_variants"])}。
- seed 43/44/45 只复核保留下来的 B1、B2；下表的“±”是三个新种子的样本标准差。
- 所有最佳 checkpoint 都只由 validation 的 DSNT 指标选择。
  不同版本的 total loss 定义不同，所以不横向比较 total loss。

## seed 42 首轮

{_run_table(screening["runs"])}

## 新种子复核

{_confirmation_table(confirmation["mean_and_sample_sd"])}

B2 − B1 的同种子配对差如下，误差指标越小越好，因此负数表示 B2 更低。

{_paired_table(confirmation["paired_delta_B2_minus_B1"])}

{_result_interpretation(aggregate)}

## B0 的两种解码

{_decoder_table(screening["B0_decoder_comparison"])}

B0 的热图 MSE 能继续下降，但坐标误差仍然很大；同一个 checkpoint 换成 argmax 后，
各点和 AoP 的变化方向也不完全一致。这个现象说明“像素级热图接近”本身不能替代
坐标与几何指标，正好也是后两项监督值得单独检查的原因。

## 结果边界

这只是监督基线的方法审计，不是完整的 GeoEqui-LD：没有 HRNet、EMA 教师、
伪标签或半监督损失。当前也没有可用于受试者分组核验的 group metadata；
结果只来自 validation 和 3 个确认种子，不能据此声称统计显著、测试集性能或 SOTA。
而且同一 validation 同时用于选择最佳轮次和报告指标，不是独立 holdout。

曲线只画 validation 的 MRE 与 AoP MAE，不画也不比较不同目标下的 total loss：

- [seed 42 三版本曲线](curves/seed42_validation_metrics.png)
- [seed 43/44/45 均值曲线](curves/confirmation_validation_metrics.png)
- [机器可读汇总](aggregate_results.json)
"""


def _render_detailed(aggregate: Mapping[str, Any]) -> str:
    screening_runs = aggregate["screening"]["runs"]
    confirmation_runs = aggregate["confirmation"]["runs"]
    all_runs = [*screening_runs, *confirmation_runs]
    return f"""# Phase 0.5：监督损失消融记录

## 问题与范围

目标是把监督部分拆开核对：B0 为 heatmap MSE；B1 为 MSE + coordinate SmoothL1；
B2 再加入 distribution JS。除损失权重外，数据、初始化（同 seed）、模型和训练预算
均保持一致。整个流程只允许 train/validation，testing 保持冻结。

## 固定协议

| 项目 | 设置 |
|---|---|
| 输入 / 热图尺寸 | 512×512 / 256×256 |
| 模型 | HeatmapUNet，base channels=8，{aggregate["model"]["trainable_parameters"]:,} 个可训练参数 |
| 优化 | Adam，lr=0.001，weight decay=0.0001，gradient clip=5.0 |
| 预算 | batch size=1，20 epochs，每个版本每个 seed 完全相同 |
| checkpoint | validation DSNT；AoP MAE、总体 MRE、较早 epoch 依次打破平局 |
| 首轮 | seed 42：B0/B1/B2，保留 2 个 |
| 复核 | seed 43/44/45：B1/B2，同种子配对 |

## 全部最佳验证结果

MRE 单位为原图像素，AoP MAE 单位为度。AoP 表中的分母是 100 个可评估样本。

{_run_table(all_runs)}

## 确认轮汇总（seed 43/44/45）

{_confirmation_table(aggregate["confirmation"]["mean_and_sample_sd"])}

{_paired_table(aggregate["confirmation"]["paired_delta_B2_minus_B1"])}

{_result_interpretation(aggregate)}

## B0 解码检查

B0 使用同一个最佳 checkpoint 分别作 DSNT 与 argmax 解码：

{_decoder_table(aggregate["screening"]["B0_decoder_comparison"])}

这项对照只用于检查解码行为，不改变三版本筛选时统一使用 DSNT 的规则。

## 运行时间与显存

{_run_table(all_runs, resources=True)}

这里报告的是每次训练、最佳 checkpoint 复评时间，以及框架记录的峰值
allocated/reserved 显存；没有公开设备名称、系统路径或运行环境明细。

## 完整性检查

- 固定 9 次运行均为 20 轮，配置 YAML 与 JSON 完全一致。
- 所有运行的数据指纹在本地相同；公开汇总不包含指纹值。
- 同一 seed 的版本使用相同初始化；B0/B1/B2 的参数量一致。
- 最佳轮次由 20 轮 CSV 按预注册规则重新计算，并与 result/metrics 文件交叉核对。
- 每个 best checkpoint 的文件摘要已与结果记录核对；公开汇总不包含摘要值。
- AoP 的有效数、可评估数、无效预测数和惩罚后 MAE 相互一致。
- testing 没有参与读取、选择或评估。

## 曲线

![seed 42 validation curves](curves/seed42_validation_metrics.png)

![confirmation validation curves](curves/confirmation_validation_metrics.png)

图中只比较同定义的 validation MRE 与 AoP MAE。由于 B0/B1/B2 的 total loss
由不同项组成，不绘制跨版本 total loss 对比。

## 限制

这份结果仅覆盖监督损失消融，不包含 HRNet、EMA 教师、伪标签和半监督一致性训练。
确认轮只有 3 个 seed，且现有材料没有 subject/group metadata 可做分组拆分审计；
同一 validation 还同时用于选择最佳轮次和报告指标，并不是独立 holdout。
因此这里不做显著性、testing 性能或 SOTA 声明。
"""


def _plot_screening(
    runs: Mapping[tuple[str, int], ValidatedRun],
    destination: Path,
) -> None:
    colors = {"B0": "#6B7280", "B1": "#0072B2", "B2": "#D55E00"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, layout="constrained")
    for variant in SCREENING_VARIANTS:
        history = runs[(variant, SCREENING_SEED)].history
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(
            epochs,
            [float(row["val_MRE_ALL"]) for row in history],
            label=variant,
            color=colors[variant],
            linewidth=1.8,
        )
        axes[1].plot(
            epochs,
            [float(row["val_aop_mae_deg"]) for row in history],
            label=variant,
            color=colors[variant],
            linewidth=1.8,
        )
    axes[0].set(title="Seed 42", xlabel="Epoch", ylabel="Validation MRE (px)")
    axes[1].set(title="Seed 42", xlabel="Epoch", ylabel="Validation AoP MAE (deg)")
    for axis in axes:
        axis.set_xlim(1, 20)
        axis.set_xticks((1, 5, 10, 15, 20))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.savefig(destination, metadata={"Software": "matplotlib"})
    plt.close(figure)


def _plot_confirmation(
    runs: Mapping[tuple[str, int], ValidatedRun],
    destination: Path,
) -> None:
    colors = {"B1": "#0072B2", "B2": "#D55E00"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, layout="constrained")
    for variant in CONFIRMATION_VARIANTS:
        for axis, field in zip(axes, ("val_MRE_ALL", "val_aop_mae_deg"), strict=True):
            per_epoch = [
                [float(runs[(variant, seed)].history[index][field]) for seed in CONFIRMATION_SEEDS]
                for index in range(20)
            ]
            means = [statistics.fmean(values) for values in per_epoch]
            sample_sd = [statistics.stdev(values) for values in per_epoch]
            epochs = list(range(1, 21))
            lower = [mean - sd for mean, sd in zip(means, sample_sd, strict=True)]
            upper = [mean + sd for mean, sd in zip(means, sample_sd, strict=True)]
            axis.plot(epochs, means, label=variant, color=colors[variant], linewidth=1.8)
            axis.fill_between(epochs, lower, upper, color=colors[variant], alpha=0.14)
    axes[0].set(
        title="Seeds 43–45: mean ± sample SD",
        xlabel="Epoch",
        ylabel="Validation MRE (px)",
    )
    axes[1].set(
        title="Seeds 43–45: mean ± sample SD",
        xlabel="Epoch",
        ylabel="Validation AoP MAE (deg)",
    )
    for axis in axes:
        axis.set_xlim(1, 20)
        axis.set_xticks((1, 5, 10, 15, 20))
        axis.title.set_fontsize(11)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.savefig(destination, metadata={"Software": "matplotlib"})
    plt.close(figure)


def _write_reports(
    aggregate: Mapping[str, Any],
    runs: Mapping[tuple[str, int], ValidatedRun],
    *,
    report_root: Path,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    curves = report_root / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    (report_root / "aggregate_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_root / "PHASE05_SUMMARY.md").write_text(
        _render_summary(aggregate),
        encoding="utf-8",
    )
    (report_root / "SUPERVISED_ABLATION.md").write_text(
        _render_detailed(aggregate),
        encoding="utf-8",
    )
    _plot_screening(runs, curves / "seed42_validation_metrics.png")
    _plot_confirmation(runs, curves / "confirmation_validation_metrics.png")


def summarize_phase05(
    *,
    protocol_path: Path,
    run_root: Path,
    report_root: Path,
) -> dict[str, Any]:
    protocol = _read_yaml_mapping(protocol_path)
    _validate_protocol(protocol)
    protocol_sha256 = _sha256_file(protocol_path)
    selection_path = run_root / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Missing frozen Phase 0.5 selection manifest")
    selection = _read_json_mapping(selection_path)
    commit = _validate_selection(selection, protocol_sha256=protocol_sha256)
    runs = {
        (variant, seed): _validate_run(
            run_root,
            variant=variant,
            seed=seed,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            commit=commit,
        )
        for variant, seed in RUN_MATRIX
    }
    _validate_cross_run_identity(runs, selection=selection)
    aggregate = _build_aggregate(runs, protocol=protocol, selection=selection)
    _write_reports(aggregate, runs, report_root=report_root)
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
        raise PermissionError("Report root must be reports/phase05")
    selection = _read_json_mapping(CANONICAL_RUN_ROOT / "selection.json")
    _validate_commit_object(str(selection.get("git_commit", "")))
    aggregate = summarize_phase05(
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
                    "reports/phase05/aggregate_results.json",
                    "reports/phase05/PHASE05_SUMMARY.md",
                    "reports/phase05/SUPERVISED_ABLATION.md",
                    "reports/phase05/curves/seed42_validation_metrics.png",
                    "reports/phase05/curves/confirmation_validation_metrics.png",
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
