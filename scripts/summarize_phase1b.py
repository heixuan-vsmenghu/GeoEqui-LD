#!/usr/bin/env python
# ruff: noqa: E501
"""Validate fixed Phase 1B artifacts and publish an allowlist-only summary.

The script reads only named train/validation experiment artifacts. It never
discovers dataset files, opens medical images, or loads model weights. Public
outputs are rebuilt from selected aggregate fields, so local paths, hashes,
timestamps, environments, and per-sample predictions cannot be copied through.
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
RUN_ROOT = REPOSITORY_ROOT / "runs" / "phase1b"
REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase1b"
PHASE1A_ROOT = REPOSITORY_ROOT / "runs" / "phase1a" / "H1_shared_B2_seed42_20e"
PHASE1A_AGGREGATE = REPOSITORY_ROOT / "reports" / "phase1a" / "aggregate_results.json"
PHASE06_AGGREGATE = REPOSITORY_ROOT / "reports" / "phase06" / "aggregate_results.json"

METRIC_NAMES = ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
SELECTION_ORDER = ("aop_mae_deg", "MRE_ALL", "earlier_epoch")
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
        description="Validate Phase 1B train/validation artifacts and publish reports."
    )
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--phase1a-run-root", type=Path, default=PHASE1A_ROOT)
    parser.add_argument("--phase1a-aggregate", type=Path, default=PHASE1A_AGGREGATE)
    parser.add_argument("--phase06-aggregate", type=Path, default=PHASE06_AGGREGATE)
    return parser


def _require_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Required regular file is missing: {path.name}")


def _read_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path.name}")
    return value


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


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    converted = _finite(value, context=context)
    integer = int(converted)
    if converted != integer or integer < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return integer


def _require(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


def _close(actual: Any, expected: Any, *, context: str, tolerance: float = 1.0e-5) -> None:
    a = _finite(actual, context=context)
    b = _finite(expected, context=context)
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{context} differs between artifacts")


def _metric_view(value: Any, *, context: str, expected_samples: int = 100) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a metric mapping")
    required = {*METRIC_NAMES[:4], "n_samples", "n_valid_aop", "n_evaluable_aop"}
    if not required.issubset(value):
        raise ValueError(f"{context} is missing required metrics")
    result = {
        name: _finite(value[name], context=f"{context}.{name}", nonnegative=True)
        for name in METRIC_NAMES[:4]
    }
    aop_value = value.get("aop_mae_deg", value.get("aop_mae_valid_deg"))
    result["aop_mae_deg"] = _finite(
        aop_value, context=f"{context}.aop_mae_deg", nonnegative=True
    )
    result["n_samples"] = _integer(value["n_samples"], context=f"{context}.n_samples")
    result["n_valid_aop"] = _integer(value["n_valid_aop"], context=f"{context}.n_valid")
    result["n_evaluable_aop"] = _integer(
        value["n_evaluable_aop"], context=f"{context}.n_evaluable"
    )
    _require(result["n_samples"], expected_samples, context=f"{context}.n_samples")
    _require(result["n_evaluable_aop"], expected_samples, context=f"{context}.n_evaluable")
    _require(result["n_valid_aop"], expected_samples, context=f"{context}.n_valid")
    mean = sum(result[name] for name in METRIC_NAMES[:3]) / 3.0
    _close(result["MRE_ALL"], mean, context=f"{context}.MRE_ALL")
    return result


def _read_history(path: Path, *, expected_epochs: int) -> tuple[dict[str, Any], ...]:
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
        metrics = {
            "MRE_PS1": _finite(row["val_MRE_PS1"], context="history PS1", nonnegative=True),
            "MRE_PS2": _finite(row["val_MRE_PS2"], context="history PS2", nonnegative=True),
            "MRE_FH1": _finite(row["val_MRE_FH1"], context="history FH1", nonnegative=True),
            "MRE_ALL": _finite(row["val_MRE_ALL"], context="history MRE", nonnegative=True),
            "aop_mae_deg": _finite(
                row["val_aop_mae_deg"], context="history AoP", nonnegative=True
            ),
            "n_samples": _integer(row["val_n_samples"], context="history samples"),
            "n_valid_aop": _integer(row["val_n_valid_aop"], context="history valid AoP"),
            "n_evaluable_aop": _integer(
                row["val_n_evaluable_aop"], context="history evaluable AoP"
            ),
        }
        checked = _metric_view(metrics, context=f"history epoch {epoch}")
        history.append({"epoch": epoch, **checked})
    return tuple(history)


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    return (float(row["aop_mae_deg"]), float(row["MRE_ALL"]), int(row["epoch"]))


def _rounded_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "MRE_PS1": round(float(value["MRE_PS1"]), 6),
        "MRE_PS2": round(float(value["MRE_PS2"]), 6),
        "MRE_FH1": round(float(value["MRE_FH1"]), 6),
        "MRE_ALL": round(float(value["MRE_ALL"]), 6),
        "aop_mae_deg": round(
            float(
                value["aop_mae_deg"]
                if "aop_mae_deg" in value
                else value["aop_mae_valid_deg"]
            ),
            6,
        ),
        "n_valid_aop": int(value.get("n_valid_aop", 100)),
        "n_evaluable_aop": int(value.get("n_evaluable_aop", 100)),
    }


def _write_curve(
    path: Path,
    shared: Sequence[Mapping[str, Any]],
    split: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.72), dpi=100)
    colors = {"PS2": "#2c7fb8", "FH1": "#d95f0e", "ALL": "#5e3c99", "AoP": "#238b45"}
    for history, model, linestyle in (
        (shared, "H1 shared", "--"),
        (split, "H2 split", "-"),
    ):
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(epochs, [row["MRE_PS2"] for row in history], linestyle, color=colors["PS2"], label=f"{model} PS2")
        axes[0].plot(epochs, [row["MRE_FH1"] for row in history], linestyle, color=colors["FH1"], label=f"{model} FH1")
        axes[1].plot(epochs, [row["MRE_ALL"] for row in history], linestyle, color=colors["ALL"], label=f"{model} MRE_ALL")
        axes[1].plot(epochs, [row["aop_mae_deg"] for row in history], linestyle, color=colors["AoP"], label=f"{model} AoP MAE")
    axes[0].set(title="Validation point errors", xlabel="Epoch", ylabel="MRE (px)")
    axes[1].set(title="Validation summary metrics", xlabel="Epoch", ylabel="Metric value")
    for axis in axes:
        axis.set_xlim(1, 20)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Phase 1B supervised decoder control (H2 stops after epoch 16)")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, metadata={})
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as image:
        clean = image.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        clean.save(path, format="PNG", optimize=True)


def _public_hygiene(value: Any, *, context: str) -> None:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    if re.search(r"(?i)(?:^|[\s\"'(])(?:[a-z]:[\\/]|/users/|/home/)", text):
        raise ValueError(f"{context} contains a local absolute path")
    forbidden = (
        "sha256",
        "fingerprint",
        "private_predictions",
        "sample_00.png",
        "started_at_utc",
        "finished_at_utc",
        "state_dict",
        ".pt\"",
    )
    lowered = text.lower()
    for token in forbidden:
        if token in lowered:
            raise ValueError(f"{context} contains forbidden public detail: {token}")


def _read_sources(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bn": _read_json(args.report_root / "BN_DIAGNOSTICS_AGGREGATE.json"),
        "replay": _read_json(args.run_root / "H1_epoch1_replay" / "replay_result.json"),
        "tiny": _read_json(args.run_root / "P1B_split_tiny_B2" / "tiny_result.json"),
        "review": _read_json(args.run_root / "P1B_split_tiny_review" / "tiny_review.json"),
        "formal": _read_json(
            args.run_root / "H2_split_B2_seed42_20e" / "formal_result.json"
        ),
        "formal_config": _read_json(
            args.run_root / "H2_split_B2_seed42_20e" / "config.json"
        ),
        "key_metrics": _read_json(
            args.run_root / "H2_split_B2_seed42_20e" / "key_checkpoint_metrics.json"
        ),
        "ledger": _read_json(args.run_root / "gpu_budget.json"),
        "phase1a": _read_json(args.phase1a_aggregate),
        "phase06": _read_json(args.phase06_aggregate),
    }


def _validate_sources(
    sources: Mapping[str, Any],
    shared_history: Sequence[Mapping[str, Any]],
    split_history: Sequence[Mapping[str, Any]],
) -> None:
    bn = sources["bn"]
    _require(bn.get("status"), "completed", context="BN diagnostic status")
    _require(bn.get("testing_frozen"), True, context="BN testing policy")
    for endpoint in ("best", "last"):
        evidence = bn["endpoints"][endpoint]
        _metric_view(
            evidence["original_bn"]["train"],
            context=f"BN {endpoint} original train",
            expected_samples=300,
        )
        _metric_view(evidence["original_bn"]["validation"], context=f"BN {endpoint} original")
        _metric_view(
            evidence["train_images_bn_reestimated"]["validation"],
            context=f"BN {endpoint} re-estimated",
        )
        if not all(evidence["integrity"].values()):
            raise ValueError(f"BN {endpoint} integrity audit did not pass")

    replay = sources["replay"]
    _require(replay.get("comparison"), "PASS", context="H1 replay")
    _require(replay.get("strictly_comparable_for_phase1b"), True, context="H1 comparability")
    _require(
        replay.get("comparison_classification"),
        "frozen_shared_comparator",
        context="H1 classification",
    )
    if not all(replay.get("static_checks", {}).values()):
        raise ValueError("H1 replay static checks did not all pass")
    if not all(item.get("matches") for item in replay.get("metric_comparisons", {}).values()):
        raise ValueError("H1 replay metrics did not all match")

    tiny = sources["tiny"]
    _require(tiny.get("status"), "completed", context="tiny status")
    _require(tiny.get("programmatic_gate"), "PASS", context="tiny programmatic gate")
    _require(tiny.get("steps_completed"), 500, context="tiny steps")
    _require(tiny["initialization"].get("output_equivalent"), True, context="initial output")
    _require(
        tiny["initialization"].get("shared_trainable_parameters"),
        29318355,
        context="shared parameter count",
    )
    _require(
        tiny["initialization"].get("split_trainable_parameters"),
        29332275,
        context="split parameter count",
    )
    _require(
        tiny["initialization"].get("additional_trainable_parameters"),
        13920,
        context="additional parameter count",
    )
    _require(tiny["review" if "review" in tiny else "visualization"].get("programmatic_check_passed"), True, context="tiny coordinate checks")
    _require(sources["review"].get("decision"), "PASS", context="tiny manual review")
    _metric_view(tiny["eval_mode"], context="tiny eval", expected_samples=4)

    formal = sources["formal"]
    _require(formal.get("status"), "budget_exhausted", context="H2 status")
    _require(formal.get("partial"), True, context="H2 partial flag")
    _require(formal.get("epochs_completed"), 16, context="H2 completed epochs")
    _require(formal.get("epochs_requested"), 20, context="H2 requested epochs")
    _require(tuple(formal.get("selection_order", ())), SELECTION_ORDER, context="selection order")
    _require(
        formal.get("shared_reference_classification"),
        "frozen_shared_comparator",
        context="formal H1 classification",
    )
    elapsed = _finite(formal.get("runtime_elapsed_sec"), context="H2 elapsed", nonnegative=True)
    allocated = _finite(
        formal.get("runtime_allocated_sec"), context="H2 allocation", nonnegative=True
    )
    if not elapsed < allocated == 7200.0:
        raise ValueError("H2 elapsed time is not below its fixed 7200 s allocation")
    runtime = sources["formal_config"].get("runtime", {})
    _require(runtime.get("allocated_seconds"), 7200.0, context="formal allocation")
    _require(
        runtime.get("training_allocated_seconds"), 6600.0, context="training guard budget"
    )
    _require(
        runtime.get("post_evaluation_reserve_seconds"),
        600.0,
        context="post-evaluation reserve",
    )
    _require(len(shared_history), 20, context="H1 history")
    _require(len(split_history), 16, context="H2 history")
    best = min(split_history, key=_selection_key)
    _require(best["epoch"], formal.get("best_epoch"), context="H2 selected best epoch")
    for name in METRIC_NAMES:
        _close(best[name], formal["best_validation_metrics"][name], context=f"H2 best {name}")
        _close(split_history[-1][name], formal["last_validation_metrics"][name], context=f"H2 last {name}")

    key_metrics = sources["key_metrics"]
    for endpoint, epoch in (("best", 3), ("last", 16)):
        audit = key_metrics[endpoint]
        _require(audit.get("epoch"), epoch, context=f"key metric {endpoint} epoch")
        _metric_view(
            audit.get("train"), context=f"key metric {endpoint} train", expected_samples=300
        )
        _metric_view(audit.get("validation"), context=f"key metric {endpoint} validation")
        for flag in (
            "evaluation_state_unchanged",
            "checkpoint_epoch_matches_selection",
            "checkpoint_metrics_match_train_log",
            "recomputed_validation_matches_checkpoint",
            "full_resume_state_present",
        ):
            _require(audit.get(flag), True, context=f"key metric {endpoint} {flag}")

    ledger = sources["ledger"]
    _require(ledger.get("total_limit_seconds"), 10800.0, context="ledger total limit")
    _require(ledger.get("active_run"), None, context="ledger active run")
    runs = ledger.get("runs", [])
    _require(len(runs), 4, context="ledger run count")
    formal_entry = runs[-1]
    _require(formal_entry.get("name"), "H2_split_B2_seed42_20e", context="ledger H2 name")
    _require(formal_entry.get("status"), "budget_exhausted", context="ledger H2 status")
    _require(formal_entry["details"].get("epochs_completed"), 16, context="ledger H2 epochs")
    _require(formal_entry["details"].get("allocation_exceeded"), False, context="H2 allocation")
    _require(
        formal_entry["details"].get("aggregate_limit_exceeded"), False, context="total GPU cap"
    )

    h1 = sources["phase1a"]["formal_run"]
    _require(h1.get("status"), "completed", context="H1 status")
    _require(h1.get("epochs_completed"), 20, context="H1 epochs")
    _require(sources["phase1a"]["data_scope"].get("testing_frozen"), True, context="H1 testing")
    _require(sources["phase06"].get("testing_frozen"), True, context="U-Net testing")
    unet_runs = [run for run in sources["phase06"].get("runs", []) if run.get("variant") == "B2"]
    _require(len(unet_runs), 1, context="historical U-Net B2 run count")
    _require(
        unet_runs[0]["checkpoints"]["best"].get("epoch"), 15, context="historical U-Net best"
    )


def _build_aggregate(
    sources: Mapping[str, Any],
    shared_history: Sequence[Mapping[str, Any]],
    split_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bn = sources["bn"]
    formal = sources["formal"]
    h1 = sources["phase1a"]["formal_run"]
    tiny = sources["tiny"]
    ledger_runs = sources["ledger"]["runs"]
    h1_best = {"epoch": int(h1["best"]["epoch"]), **_rounded_metrics(h1["best"])}
    h1_last = {"epoch": int(h1["last"]["epoch"]), **_rounded_metrics(h1["last"])}
    h2_best = {"epoch": int(formal["best_epoch"]), **_rounded_metrics(formal["best_validation_metrics"])}
    h2_last = {"epoch": 16, **_rounded_metrics(formal["last_validation_metrics"])}
    h2_best_train_raw = sources["key_metrics"]["best"]["train"]
    h2_last_train_raw = sources["key_metrics"]["last"]["train"]
    h2_best_train = _rounded_metrics(h2_best_train_raw)
    h2_last_train = _rounded_metrics(h2_last_train_raw)
    h2_best_gap = {
        name: round(
            float(formal["best_validation_metrics"][name]) - float(h2_best_train_raw[name]), 6
        )
        for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
    }
    h2_last_gap = {
        name: round(
            float(formal["last_validation_metrics"][name]) - float(h2_last_train_raw[name]), 6
        )
        for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
    }
    deltas = {
        name: round(h2_best[name] - h1_best[name], 6)
        for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
    }
    h1_epoch16 = {"epoch": 16, **_rounded_metrics(shared_history[15])}
    h2_epoch16 = {"epoch": 16, **_rounded_metrics(split_history[15])}
    matched_epoch16_deltas = {
        name: round(h2_epoch16[name] - h1_epoch16[name], 6)
        for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
    }

    bn_endpoints: dict[str, Any] = {}
    for endpoint in ("best", "last"):
        item = bn["endpoints"][endpoint]
        original_train = _rounded_metrics(item["original_bn"]["train"])
        original = _rounded_metrics(item["original_bn"]["validation"])
        reestimated = _rounded_metrics(item["train_images_bn_reestimated"]["validation"])
        bn_endpoints[endpoint] = {
            "epoch": int(item["epoch"]),
            "original_bn_train": original_train,
            "original_bn_validation": original,
            "original_bn_validation_minus_train": {
                name: round(float(original[name]) - float(original_train[name]), 6)
                for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
            },
            "train_images_bn_reestimated": reestimated,
            "delta_reestimated_minus_original": {
                name: round(float(reestimated[name]) - float(original[name]), 6)
                for name in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
            },
        }

    unet_run = next(run for run in sources["phase06"]["runs"] if run["variant"] == "B2")
    unet = {
        "epoch": int(unet_run["checkpoints"]["best"]["epoch"]),
        **_rounded_metrics(unet_run["checkpoints"]["best"]["metrics"]),
    }
    elapsed_total = sum(float(run["elapsed_seconds"]) for run in ledger_runs)
    return {
        "schema_version": 1,
        "phase": "phase1b-supervised-decoder-control",
        "scope": "BatchNorm diagnostic and shared-versus-independent decoder control",
        "data_scope": {
            "train_samples": 300,
            "validation_samples": 100,
            "testing_frozen": True,
        },
        "protocol": {
            "seed": 42,
            "loss": "B2: heatmap MSE + coordinate SmoothL1 + distribution JS",
            "batch_size": 1,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "selection_order": list(SELECTION_ORDER),
            "shared_model": "HRNetW32SharedHeatmap",
            "split_model": "HRNetW32SplitHeatmap (PS 2-channel head + FH 1-channel head)",
        },
        "integrity": {
            "h1_epoch1_replay": "PASS",
            "h1_comparison_classification": "frozen_shared_comparator",
            "split_initial_output_equivalent_to_shared": True,
            "split_initial_max_absolute_output_difference": 0.0,
            "best_and_last_recomputed_validation_match": True,
            "evaluation_state_unchanged": True,
            "public_outputs_sanitized": True,
        },
        "bn_diagnostic": {
            "method": "Reset BN running statistics and make one cumulative train-image-only pass",
            "used_for_model_selection": False,
            "parameters_unchanged": True,
            "only_bn_running_statistics_changed": True,
            "endpoints": bn_endpoints,
            "interpretation": (
                "Re-estimation helped the epoch-20 endpoint but hurt the selected epoch-3 "
                "checkpoint overall; this supports BN-statistics sensitivity as a risk, not a sole cause."
            ),
        },
        "decoder_control": {
            "parameter_counts": {
                "h1_shared": int(tiny["initialization"]["shared_trainable_parameters"]),
                "h2_split": int(tiny["initialization"]["split_trainable_parameters"]),
                "h2_minus_h1": int(tiny["initialization"]["additional_trainable_parameters"]),
            },
            "tiny_gate": {
                "steps": 500,
                "programmatic_gate": "PASS",
                "manual_visual_review": "PASS",
                "MRE_ALL": round(float(tiny["eval_mode"]["MRE_ALL"]), 6),
                "aop_mae_deg": round(float(tiny["eval_mode"]["aop_mae_deg"]), 6),
                "n_valid_aop": int(tiny["eval_mode"]["n_valid_aop"]),
            },
            "h1_shared": {
                "status": "completed",
                "epochs_completed": 20,
                "best": h1_best,
                "last": h1_last,
            },
            "h2_split": {
                "status": "budget_exhausted",
                "training_subbudget_exhausted": True,
                "formal_allocation_exceeded": False,
                "aggregate_gpu_cap_exceeded": False,
                "partial": True,
                "epochs_completed": 16,
                "epochs_requested": 20,
                "formal_elapsed_seconds": round(float(formal["runtime_elapsed_sec"]), 3),
                "formal_allocation_seconds": 7200.0,
                "post_evaluation_reserve_seconds": 600.0,
                "training_guard_seconds": 6600.0,
                "ledger_closing_reserve_seconds": 120.0,
                "actual_formal_elapsed_below_allocation": True,
                "stop_explanation": (
                    "The 6600 s training guard stopped before epoch 17 so that post-training "
                    "evaluation could use its 600 s reserve inside the 7200 s formal allocation."
                ),
                "best": h2_best,
                "best_train": h2_best_train,
                "best_validation_minus_train": h2_best_gap,
                "last_observed": h2_last,
                "last_observed_train": h2_last_train,
                "last_observed_validation_minus_train": h2_last_gap,
            },
            "selected_best_delta_h2_minus_h1": deltas,
            "matched_epoch16": {
                "h1_shared": h1_epoch16,
                "h2_split": h2_epoch16,
                "delta_h2_minus_h1": matched_epoch16_deltas,
                "interpretation": (
                    "At the strictly matched epoch 16, H2 PS2 is 3.129 px higher, while FH1 "
                    "is 9.264 px lower; this mixed pointwise result does not establish that the "
                    "split decoder alleviates late-stage regression."
                ),
            },
            "pointwise_readout": {
                "PS2": (
                    "At the selected epoch-3 checkpoints, H2 is 3.661 px lower; this is a "
                    "single-seed validation difference, not proof of a general decoder benefit."
                ),
                "FH1": (
                    "At the selected epoch-3 checkpoints, H2 is 4.960 px higher; independent "
                    "decoding did not improve PS2 and FH1 together."
                ),
            },
        },
        "historical_context_only": {
            "unet_B2": {
                "best_epoch": unet["epoch"],
                "MRE_PS1": unet["MRE_PS1"],
                "MRE_PS2": unet["MRE_PS2"],
                "MRE_FH1": unet["MRE_FH1"],
                "MRE_ALL": unet["MRE_ALL"],
                "aop_mae_deg": unet["aop_mae_deg"],
                "boundary": "Different architecture and budget; descriptive context only.",
            }
        },
        "resources": {
            "total_phase_gpu_cap_seconds": 10800.0,
            "formal_allocation_seconds": 7200.0,
            "post_evaluation_reserve_seconds": 600.0,
            "training_guard_seconds": 6600.0,
            "ledger_closing_reserve_seconds": 120.0,
            "ledger_elapsed_seconds": round(elapsed_total, 3),
            "all_runs_closed": True,
            "allocation_or_total_limit_exceeded": False,
        },
        "conclusion": (
            "The independent PS/FH decoder is implemented and trainable, but this partial "
            "single-seed control does not show a uniform validation advantage over the shared head."
        ),
        "limitations": [
            "H2 completed 16 of 20 requested epochs; no epoch-20 endpoint exists.",
            "One seed and one validation split; no statistical significance claim.",
            "Best checkpoints are selected and reported on the same validation split.",
            "This is a supervised decoder control, not the complete GeoEqui-LD method.",
            "No EMA teacher, pseudo-labels, unlabeled consistency objective, or semi-supervised claim.",
        ],
    }


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    return (
        f"{metrics['MRE_PS1']:.3f} | {metrics['MRE_PS2']:.3f} | "
        f"{metrics['MRE_FH1']:.3f} | {metrics['MRE_ALL']:.3f} | "
        f"{metrics['aop_mae_deg']:.3f}"
    )


def _render_bn(aggregate: Mapping[str, Any]) -> str:
    best = aggregate["bn_diagnostic"]["endpoints"]["best"]
    last = aggregate["bn_diagnostic"]["endpoints"]["last"]
    return f"""# Phase 1B：BatchNorm 短诊断

这次只做了一个很窄的检查：固定 H1 的 best（epoch 3）和 last（epoch 20）权重，不训练参数，只用 300 张 train 图像重新累计一次 BatchNorm 运行统计，然后重新看 validation。原 checkpoint 没有被覆盖，重估结果也没有参与选模。

| checkpoint | BN 状态 | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE |
|---|---|---:|---:|---:|---:|---:|
| best / e3 | 原 BN | {_format_metrics(best['original_bn_validation'])} |
| best / e3 | train 图像重估 | {_format_metrics(best['train_images_bn_reestimated'])} |
| last / e20 | 原 BN | {_format_metrics(last['original_bn_validation'])} |
| last / e20 | train 图像重估 | {_format_metrics(last['train_images_bn_reestimated'])} |

## 原 BN 下的 validation–train 差距

| checkpoint | split | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE |
|---|---|---:|---:|---:|---:|---:|
| best / e3 | train | {_format_metrics(best['original_bn_train'])} |
| best / e3 | validation | {_format_metrics(best['original_bn_validation'])} |
| best / e3 | validation − train | {_format_metrics(best['original_bn_validation_minus_train'])} |
| last / e20 | train | {_format_metrics(last['original_bn_train'])} |
| last / e20 | validation | {_format_metrics(last['original_bn_validation'])} |
| last / e20 | validation − train | {_format_metrics(last['original_bn_validation_minus_train'])} |

e3 的 validation−train 差距在 PS2/FH1 上分别是 {best['original_bn_validation_minus_train']['MRE_PS2']:.3f}/{best['original_bn_validation_minus_train']['MRE_FH1']:.3f} px；到 e20 扩大为 {last['original_bn_validation_minus_train']['MRE_PS2']:.3f}/{last['original_bn_validation_minus_train']['MRE_FH1']:.3f} px。AoP 差距也从 {best['original_bn_validation_minus_train']['aop_mae_deg']:.3f}° 增至 {last['original_bn_validation_minus_train']['aop_mae_deg']:.3f}°。这描述了后期泛化差距，但仍不能单独确定成因。

数值方向并不一致。best checkpoint 重估后，PS2 从 {best['original_bn_validation']['MRE_PS2']:.3f} 变成 {best['train_images_bn_reestimated']['MRE_PS2']:.3f} px，变差 {best['delta_reestimated_minus_original']['MRE_PS2']:.3f} px；FH1 反而改善 {abs(best['delta_reestimated_minus_original']['MRE_FH1']):.3f} px。last checkpoint 的 PS2 和 FH1 则分别改善 {abs(last['delta_reestimated_minus_original']['MRE_PS2']):.3f} 和 {abs(last['delta_reestimated_minus_original']['MRE_FH1']):.3f} px，整体 MRE 与 AoP 也下降。

所以目前比较稳妥的说法是：batch size 1 下，BN 运行统计确实会影响 validation 表现，可能是波动来源之一；但一次重估既能改善某个端点，也会损害另一个端点，不能据此把 H1 的全部波动都归因于 BN。参数审计确认权重未变，重估阶段只有 BN 运行统计发生变化。

这项检查只使用 train 图像更新统计并在 validation 上汇总，testing 没有读取或评估。
"""


def _render_decoder(aggregate: Mapping[str, Any]) -> str:
    control = aggregate["decoder_control"]
    h1 = control["h1_shared"]
    h2 = control["h2_split"]
    delta = control["selected_best_delta_h2_minus_h1"]
    matched = control["matched_epoch16"]
    params = control["parameter_counts"]
    return f"""# Phase 1B：共享解码器与 PS/FH 独立解码器

这一轮只改了解码头：H1 用一个共享三通道热图头；H2 保留同一 HRNet-W32 主干，把输出拆成 PS 两通道头和 FH 一通道头，再按 `[PS1, PS2, FH1]` 拼回去使用原来的 B2 损失。可训练参数由 {params['h1_shared']:,} 增到 {params['h2_split']:,}，增加 {params['h2_minus_h1']:,}。H2 从同一个共享模型状态拆分初始化，初始化探针的最大输出差为 0，因而起点可对齐。H1 的 epoch 1 确定性重放也逐项一致，可以作为冻结的共享头参照。

四样本 tiny-overfit 跑满 500 步，MRE_ALL 为 {control['tiny_gate']['MRE_ALL']:.3f} px，AoP MAE 为 {control['tiny_gate']['aop_mae_deg']:.3f}°，4/4 AoP 有效；程序检查与受限叠加图人工检查均通过。

## validation 结果

| 模型 | 运行状态 | checkpoint | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| H1 共享头 | 完整 20/20 | best | {h1['best']['epoch']} | {_format_metrics(h1['best'])} |
| H1 共享头 | 完整 20/20 | last | {h1['last']['epoch']} | {_format_metrics(h1['last'])} |
| H2 独立头 | 部分 16/20 | best | {h2['best']['epoch']} | {_format_metrics(h2['best'])} |
| H2 独立头 | 部分 16/20 | last observed | {h2['last_observed']['epoch']} | {_format_metrics(h2['last_observed'])} |

## H2 的 train–validation 差距

| checkpoint | split | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---|---:|---:|---:|---:|---:|
| best / e3 | train | {_format_metrics(h2['best_train'])} |
| best / e3 | validation | {_format_metrics(h2['best'])} |
| best / e3 | validation − train | {_format_metrics(h2['best_validation_minus_train'])} |
| last / e16 | train | {_format_metrics(h2['last_observed_train'])} |
| last / e16 | validation | {_format_metrics(h2['last_observed'])} |
| last / e16 | validation − train | {_format_metrics(h2['last_observed_validation_minus_train'])} |

H2 e3 的 validation−train gap 是 PS2 {h2['best_validation_minus_train']['MRE_PS2']:.3f} px、FH1 {h2['best_validation_minus_train']['MRE_FH1']:.3f} px；e16 分别为 {h2['last_observed_validation_minus_train']['MRE_PS2']:.3f} 和 {h2['last_observed_validation_minus_train']['MRE_FH1']:.3f} px。也就是说 PS2 的 gap 到 e16 更大，而 FH1 略小；这里同样不能只看整体均值。

H2 正式运行的实际总用时为 {h2['formal_elapsed_seconds']:.1f} 秒，低于 7200 秒 formal allocation。这个 7200 秒内预留 600 秒做训练后的 best/last 复算，因此训练循环使用 6600 秒 guard；另外 ledger 对 Phase 1B 总 GPU 预算保留 120 秒 closing reserve。epoch 16 后触发的是 6600 秒 training guard，ledger 因而记为 `budget_exhausted`；它不表示 7200 秒 formal allocation 超限，也不表示超过 3 小时总 GPU 上限。这不是 20 轮完整结果，也不存在 H2 epoch 20 指标。

## 两个点分别怎么看

- PS2：两个方案的 selected best 都在 epoch 3，H2 为 {h2['best']['MRE_PS2']:.3f} px，H1 为 {h1['best']['MRE_PS2']:.3f} px，差值 {delta['MRE_PS2']:+.3f} px。这个点在本次 validation 上更好。
- FH1：同一对 selected best 下，H2 为 {h2['best']['MRE_FH1']:.3f} px，H1 为 {h1['best']['MRE_FH1']:.3f} px，差值 {delta['MRE_FH1']:+.3f} px，反而更差。

总体 MRE 在 H2 selected best 上低 {abs(delta['MRE_ALL']):.3f} px，但作为首要选择指标的 AoP MAE 高 {delta['aop_mae_deg']:.3f}°。因此这一轮没有得到“拆头以后 PS 和 FH 都更准”的证据。H2 的 epoch 16 端点在若干指标上看起来比 H1 epoch 20 好，但训练轮数不同，而且 H2 没跑满，不能把这组端点差直接解释为结构收益。

## 严格对齐到 epoch 16

| 模型 | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
| H1 共享头 | 16 | {_format_metrics(matched['h1_shared'])} |
| H2 独立头 | 16 | {_format_metrics(matched['h2_split'])} |
| H2 − H1 | 16 | {_format_metrics(matched['delta_h2_minus_h1'])} |

在严格 matched epoch 16 上，H2 的 PS2 反而高 {matched['delta_h2_minus_h1']['MRE_PS2']:.4f} px，FH1 低 {abs(matched['delta_h2_minus_h1']['MRE_FH1']):.4f} px，AoP MAE 几乎相同（差 {matched['delta_h2_minus_h1']['aop_mae_deg']:+.4f}°）。所以不能说独立头缓解了共享头的后期退步；证据仍然是点间方向混合。

曲线见 [validation_metrics.png](curves/validation_metrics.png)。本对照只有 seed 42，仍属于增强监督诊断，不是半监督方法结果。
"""


def _render_summary(aggregate: Mapping[str, Any]) -> str:
    h2 = aggregate["decoder_control"]["h2_split"]
    control = aggregate["decoder_control"]
    matched = control["matched_epoch16"]
    params = control["parameter_counts"]
    bn = aggregate["bn_diagnostic"]["endpoints"]
    unet = aggregate["historical_context_only"]["unet_B2"]
    return f"""# Phase 1B 小结

这轮做了两件事：先检查 H1 的 BatchNorm 运行统计是否会影响结果，再把共享三通道热图头拆成 PS 与 FH 两个独立头做监督对照。它们都是进入后续方法前的排查，不是完整 GeoEqui-LD。

BN 检查给出的信号是“有影响，但不能一口咬定是唯一原因”。原 BN 下，H1 的 validation−train MRE_ALL 差距从 e3 的 {bn['best']['original_bn_validation_minus_train']['MRE_ALL']:.3f} px 扩到 e20 的 {bn['last']['original_bn_validation_minus_train']['MRE_ALL']:.3f} px；PS2 差距由 {bn['best']['original_bn_validation_minus_train']['MRE_PS2']:.3f} 扩到 {bn['last']['original_bn_validation_minus_train']['MRE_PS2']:.3f} px，FH1 由 {bn['best']['original_bn_validation_minus_train']['MRE_FH1']:.3f} 扩到 {bn['last']['original_bn_validation_minus_train']['MRE_FH1']:.3f} px。重估 train 图像统计后，H1 的 epoch 20 validation 明显改善；同样操作放到 epoch 3 best 上，整体指标却变差。具体数字见 [BN_DIAGNOSTICS.md](BN_DIAGNOSTICS.md)。

独立解码器增加 {params['h2_minus_h1']:,} 个可训练参数（{params['h1_shared']:,} → {params['h2_split']:,}），并通过等价初始化、四样本 tiny-overfit 和 checkpoint 复算。正式 H2 原计划 20 轮，实际完成 {h2['epochs_completed']}/20；formal elapsed 为 {h2['formal_elapsed_seconds']:.1f} 秒，低于 7200 秒 formal allocation。运行器在这 7200 秒内给训练后复算留了 600 秒，所以训练循环使用 6600 秒 guard；ledger 另留 120 秒 closing reserve。`budget_exhausted` 表示 training guard 在下一轮前触发，不是 7200 秒实际超时，也不是 3 小时总上限超限；这里只按“16 轮部分结果”汇报。

在可比的 selected best（两者都是 epoch 3）上，H2 的 PS2 从 27.854 降到 24.193 px，FH1 却从 46.837 升到 51.797 px；MRE_ALL 小幅下降，AoP MAE 则升高。简单说，拆头后某些点有改善，但没有形成一致优势。更完整的逐点记录和曲线见 [DECODER_COMPARISON.md](DECODER_COMPARISON.md)。

H2 自身的 validation−train gap 在 e3 为 PS2 {h2['best_validation_minus_train']['MRE_PS2']:.3f}、FH1 {h2['best_validation_minus_train']['MRE_FH1']:.3f} px；到 e16 是 PS2 {h2['last_observed_validation_minus_train']['MRE_PS2']:.3f}、FH1 {h2['last_observed_validation_minus_train']['MRE_FH1']:.3f} px。PS2 gap 继续扩大，不能用 MRE_ALL 的变化代替逐点判断。

严格对齐到 epoch 16 后，H2 相对 H1 的 PS2 是 {matched['delta_h2_minus_h1']['MRE_PS2']:+.4f} px，FH1 是 {matched['delta_h2_minus_h1']['MRE_FH1']:+.4f} px，MRE_ALL 是 {matched['delta_h2_minus_h1']['MRE_ALL']:+.4f} px，AoP MAE 是 {matched['delta_h2_minus_h1']['aop_mae_deg']:+.4f}°。尤其 PS2 并未改善，所以不能说独立头缓解了后期退步。

旧 U-Net B2 的 epoch {unet['best_epoch']} best 逐点为 PS1 {unet['MRE_PS1']:.4f}、PS2 {unet['MRE_PS2']:.4f}、FH1 {unet['MRE_FH1']:.4f}、MRE_ALL {unet['MRE_ALL']:.4f} px，AoP MAE {unet['aop_mae_deg']:.5f}°。这里只保留为历史量级参照；它与 HRNet 的架构和预算不同，不拿来证明哪个结构更优。

## 当前可以下的结论

1. H1 的 validation 波动对 BN 运行统计敏感，但现有诊断不足以确认单一原因。
2. PS/FH 独立头在工程上可用，初始化和训练链路都通过；本次单 seed、16/20 轮结果没有证明它整体优于共享头。
3. H2 的 PS2 selected-best 结果更好，FH1 与 AoP selected-best 结果更差，后续若继续需要分别看点，而不能只报 MRE_ALL。
4. 这一阶段仍是有标注监督对照，没有 EMA、伪标签、无标签一致性或半监督结论。

所有公开数字都是 train/validation 聚合结果，testing 保持冻结；公开目录不含权重、逐样本预测、真实图像或本机路径。
"""


def _sanitized_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "phase1b-supervised-decoder-control",
        "data": {"train_samples": 300, "validation_samples": 100, "testing_frozen": True},
        "training": {
            "seed": 42,
            "batch_size": 1,
            "epochs_requested": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "Adam",
            "precision": "float32",
            "loss": "heatmap MSE + 10 * coordinate SmoothL1 + distribution JS",
        },
        "model": {
            "backbone": "HRNet-W32",
            "input_channels": 1,
            "shared_head": "one 3-channel heatmap decoder",
            "split_head": "PS 2-channel decoder + FH 1-channel decoder",
            "concatenation_order": ["PS1", "PS2", "FH1"],
            "pretrained": False,
        },
        "selection": ["validation AoP MAE", "validation MRE_ALL", "earlier epoch"],
        "resources": {
            "formal_allocation_seconds": 7200,
            "post_evaluation_reserve_seconds": 600,
            "training_guard_seconds": 6600,
            "total_phase_gpu_budget_seconds": 10800,
            "ledger_closing_reserve_seconds": 120,
        },
        "claim_boundary": "supervised diagnostic only; not the complete semi-supervised method",
    }


def publish(args: argparse.Namespace) -> dict[str, Any]:
    sources = _read_sources(args)
    shared_history = _read_history(args.phase1a_run_root / "train_log.csv", expected_epochs=20)
    split_history = _read_history(
        args.run_root / "H2_split_B2_seed42_20e" / "train_log.csv", expected_epochs=16
    )
    _validate_sources(sources, shared_history, split_history)
    aggregate = _build_aggregate(sources, shared_history, split_history)
    config = _sanitized_config()
    markdown = {
        "BN_DIAGNOSTICS.md": _render_bn(aggregate),
        "DECODER_COMPARISON.md": _render_decoder(aggregate),
        "PHASE1B_SUMMARY.md": _render_summary(aggregate),
    }
    _public_hygiene(aggregate, context="aggregate")
    _public_hygiene(config, context="sanitized config")
    for name, text in markdown.items():
        _public_hygiene(text, context=name)

    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "aggregate_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_root / "sanitized_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    for name, text in markdown.items():
        (args.report_root / name).write_text(text, encoding="utf-8")
    _write_curve(args.report_root / "curves" / "validation_metrics.png", shared_history, split_history)
    return aggregate


def main() -> None:
    args = build_parser().parse_args()
    aggregate = publish(args)
    h2 = aggregate["decoder_control"]["h2_split"]
    print(
        "Phase 1B reports published: "
        f"H2 {h2['epochs_completed']}/{h2['epochs_requested']} epochs, "
        f"status={h2['status']}."
    )


if __name__ == "__main__":
    main()
