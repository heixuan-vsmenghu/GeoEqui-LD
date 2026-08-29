#!/usr/bin/env python
"""Fit constant coordinates on train labels and evaluate them on validation only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase1a_b0_diagnostics.yaml"
CANONICAL_LOCAL_CONFIG = REPOSITORY_ROOT / "configs" / "phase05_local.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.diagnostics.phase1a import (  # noqa: E402
    assert_public_aggregate,
    load_phase1a_protocol,
    load_verified_splits,
    make_labeled_dataset,
    require_canonical_path,
    require_private_output_path,
    require_public_output_path,
    train_mean_coordinate_baseline,
)
from geoequi_ld.training.engine import write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the train-label-only mean baseline")
    parser.add_argument("--protocol", type=Path, default=CANONICAL_PROTOCOL)
    parser.add_argument("--local-config", type=Path, default=CANONICAL_LOCAL_CONFIG)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "phase1a" / "TRAIN_MEAN_BASELINE.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "phase1a" / "TRAIN_MEAN_BASELINE.md",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "phase1a" / "mean_baseline" / "details.json",
    )
    return parser


def _metric_text(value: object, suffix: str = "") -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "undefined"
    return f"{float(value):.6f}{suffix}"


def _markdown(public: dict[str, object]) -> str:
    metrics = public["validation_metrics"]
    assert isinstance(metrics, dict)
    return "\n".join(
        [
            "# Phase 1A train-mean coordinate baseline",
            "",
            "The three constant coordinates are fitted from the 300 train labels only. The 100 "
            "validation labels are used only by the frozen evaluator. No image pixels are model "
            "inputs.",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| MRE_PS1 | {_metric_text(metrics['MRE_PS1'], ' px')} |",
            f"| MRE_PS2 | {_metric_text(metrics['MRE_PS2'], ' px')} |",
            f"| MRE_FH1 | {_metric_text(metrics['MRE_FH1'], ' px')} |",
            f"| MRE_ALL | {_metric_text(metrics['MRE_ALL'], ' px')} |",
            f"| evaluable AoP | {metrics['n_evaluable_aop']} |",
            f"| valid predicted AoP | {metrics['n_valid_aop']} |",
            f"| valid predicted AoP ratio | {_metric_text(metrics['aop_valid_ratio'])} |",
            f"| invalid predicted AoP | {metrics['aop_invalid_prediction_count']} |",
            "| invalid predicted AoP ratio | "
            f"{_metric_text(metrics['aop_invalid_prediction_ratio'])} |",
            f"| valid-only AoP MAE | {_metric_text(metrics['aop_mae_valid_deg'], ' deg')} |",
            f"| penalized AoP score | {_metric_text(metrics['aop_mae_deg'], ' deg')} |",
            "",
            "Exact fitted coordinates, source paths, fingerprints, and identifiers remain in the "
            "Git-ignored private artifact.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = require_canonical_path(
        args.protocol, CANONICAL_PROTOCOL, context="Phase 1A protocol"
    )
    local_config = require_canonical_path(
        args.local_config, CANONICAL_LOCAL_CONFIG, context="Phase 1A local split config"
    )
    json_output = require_public_output_path(args.json_output, repository_root=REPOSITORY_ROOT)
    markdown_output = require_public_output_path(
        args.markdown_output, repository_root=REPOSITORY_ROOT
    )
    private_output = require_private_output_path(
        args.private_output, repository_root=REPOSITORY_ROOT
    )

    protocol = load_phase1a_protocol(protocol_path)
    verified = load_verified_splits(local_config, protocol)
    train_dataset = make_labeled_dataset(verified.specs["train"], protocol)
    validation_dataset = make_labeled_dataset(verified.specs["validation"], protocol)
    metrics, train_mean = train_mean_coordinate_baseline(train_dataset, validation_dataset)
    public: dict[str, object] = {
        "phase": "phase1a-b0-diagnostics",
        "status": "completed",
        "baseline": "train_mean_coordinate_baseline",
        "fit_split": "train",
        "evaluation_split": "validation",
        "data": {
            "train_count": len(train_dataset),
            "validation_count": len(validation_dataset),
            "fingerprints_verified": True,
        },
        "validation_metrics": metrics,
        "interpretation_boundary": (
            "This is a no-image reference, not a trained model and not an independent "
            "holdout result."
        ),
    }
    assert_public_aggregate(public)
    private = {
        "phase": "phase1a-b0-diagnostics",
        "baseline": "train_mean_coordinate_baseline",
        "local_config": str(local_config),
        "train_mean_coordinates": train_mean.tolist(),
        "fingerprints": {
            role: dict(value) for role, value in verified.fingerprints.items()
        },
        "validation_metrics": metrics,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_output, public)
    write_json(private_output, private)
    markdown_output.write_text(_markdown(public), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "public_report": str(markdown_output.relative_to(REPOSITORY_ROOT)),
                "private_details": str(private_output.relative_to(REPOSITORY_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
