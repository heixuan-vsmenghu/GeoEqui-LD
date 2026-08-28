#!/usr/bin/env python
"""Freeze the two retained variants after the seed-42 validation screen."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase05_ablation.yaml"
CANONICAL_RUN_ROOT = REPOSITORY_ROOT / "runs" / "phase05"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.training.ablation import VARIANTS, assert_variant_weights  # noqa: E402
from geoequi_ld.training.config import SupervisedTrainingConfig  # noqa: E402
from geoequi_ld.utils.hashing import sha256_file  # noqa: E402

METRIC_NAMES = (
    "MRE_PS1",
    "MRE_PS2",
    "MRE_FH1",
    "MRE_ALL",
    "aop_mae_deg",
    "aop_mae_valid_deg",
)
COUNT_NAMES = (
    "n_samples",
    "n_valid_aop",
    "n_evaluable_aop",
    "aop_invalid_prediction_count",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate all seed-42 runs and write an immutable local selection manifest."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _require_finite_metrics(metrics: Mapping[str, Any], *, decoder: str) -> None:
    if metrics.get("decoder") != decoder:
        raise ValueError(f"Decoder identity mismatch: expected {decoder}")
    for name in METRIC_NAMES:
        if not math.isfinite(float(metrics[name])):
            raise ValueError(f"Non-finite validation metric: {name}")
    counts = {name: int(metrics[name]) for name in COUNT_NAMES}
    if counts["n_samples"] != 100 or counts["n_evaluable_aop"] != 100:
        raise ValueError("Phase 0.5 expects exactly 100 evaluable validation samples")
    if not 0 <= counts["n_valid_aop"] <= counts["n_evaluable_aop"]:
        raise ValueError("Invalid AoP count range")
    expected_invalid = counts["n_evaluable_aop"] - counts["n_valid_aop"]
    if counts["aop_invalid_prediction_count"] != expected_invalid:
        raise ValueError("AoP invalid-prediction count is inconsistent")


def _load_run(variant: str, *, protocol_sha256: str, commit: str) -> dict[str, Any]:
    run_dir = CANONICAL_RUN_ROOT / variant / "seed_42"
    result_path = run_dir / "phase05_result.json"
    config_path = run_dir / "config.yaml"
    if not result_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing completed seed-42 run for {variant}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError(f"Invalid run config for {variant}")
    expected_identity = {
        "status": "completed",
        "phase": "phase0.5-supervised-ablation",
        "variant": variant,
        "seed": 42,
        "selection_split": "validation",
        "selection_decoder": "dsnt",
        "testing_frozen": True,
    }
    for key, expected in expected_identity.items():
        if result.get(key) != expected:
            raise ValueError(f"Run identity mismatch for {variant}: {key}")
    if config.get("phase") != expected_identity["phase"] or config.get("variant") != variant:
        raise ValueError(f"Config identity mismatch for {variant}")
    if config.get("testing_frozen") is not True:
        raise PermissionError(f"Run config does not freeze testing for {variant}")
    training = SupervisedTrainingConfig.from_mapping(config["training"])
    if training.seed != 42:
        raise ValueError(f"Config seed mismatch for {variant}")
    assert_variant_weights(training, variant)
    provenance = config["provenance"]
    if provenance.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Protocol hash mismatch for {variant}")
    if provenance.get("git_commit") != commit or provenance.get("git_dirty") is not False:
        raise ValueError(f"Git provenance mismatch for {variant}")
    result_provenance = result.get("provenance", {})
    if result_provenance.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Result protocol mismatch for {variant}")
    if result_provenance.get("git_commit") != commit:
        raise ValueError(f"Result commit mismatch for {variant}")
    if not isinstance(result_provenance.get("best_checkpoint_sha256"), str):
        raise ValueError(f"Missing checkpoint digest for {variant}")
    decoders = result.get("best_validation_metrics", {})
    _require_finite_metrics(decoders["dsnt"], decoder="dsnt")
    if variant == "B0":
        _require_finite_metrics(decoders["argmax"], decoder="argmax")
    return {
        "result": result,
        "config": config,
        "result_sha256": sha256_file(result_path),
    }


def _without_loss_weights(training: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "heatmap_loss_weight",
        "coordinate_loss_weight",
        "distribution_loss_weight",
    }
    return {key: value for key, value in training.items() if key not in ignored}


def _validate_common_identity(runs: Mapping[str, Mapping[str, Any]]) -> None:
    reference = runs["B0"]["config"]
    reference_training = _without_loss_weights(reference["training"])
    reference_data = reference["data"]
    reference_model = reference["model"]
    for variant, run in runs.items():
        config = run["config"]
        if _without_loss_weights(config["training"]) != reference_training:
            raise ValueError(f"Controlled training variables differ for {variant}")
        if config["data"] != reference_data:
            raise ValueError(f"Data identity differs for {variant}")
        if config["model"]["class"] != reference_model["class"]:
            raise ValueError(f"Model class differs for {variant}")
        if config["model"]["trainable_parameters"] != reference_model["trainable_parameters"]:
            raise ValueError(f"Parameter count differs for {variant}")
        if config["model"]["initialization_sha256"] != reference_model["initialization_sha256"]:
            raise ValueError(f"Seed-42 initialization differs for {variant}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.protocol.resolve(strict=True) != CANONICAL_PROTOCOL.resolve(strict=True):
        raise PermissionError("Selection accepts only the canonical Phase 0.5 protocol")
    if args.run_root.resolve(strict=True) != CANONICAL_RUN_ROOT.resolve(strict=True):
        raise PermissionError("Selection accepts only runs/phase05")
    if _git_output("status", "--porcelain"):
        raise RuntimeError("Selection requires a clean Git worktree")
    commit = _git_output("rev-parse", "HEAD")
    protocol = yaml.safe_load(CANONICAL_PROTOCOL.read_text(encoding="utf-8"))
    if protocol["selection"]["first_round_seed"] != 42:
        raise ValueError("Unexpected screening seed")
    protocol_sha256 = sha256_file(CANONICAL_PROTOCOL)
    runs = {
        variant: _load_run(variant, protocol_sha256=protocol_sha256, commit=commit)
        for variant in VARIANTS
    }
    _validate_common_identity(runs)
    complexity = {name: index for index, name in enumerate(VARIANTS)}
    ranked = sorted(
        VARIANTS,
        key=lambda variant: (
            float(runs[variant]["result"]["best_validation_metrics"]["dsnt"]["aop_mae_deg"]),
            float(runs[variant]["result"]["best_validation_metrics"]["dsnt"]["MRE_ALL"]),
            complexity[variant],
        ),
    )
    selected = ranked[: int(protocol["selection"]["retain_after_first_round"])]
    manifest = {
        "schema_version": 1,
        "phase": "phase0.5-supervised-ablation",
        "testing_frozen": True,
        "git_commit": commit,
        "protocol_sha256": protocol_sha256,
        "screening_seed": 42,
        "confirmation_seeds": protocol["selection"]["confirmation_seeds"],
        "rule": protocol["selection"]["variant_retention"],
        "selected_variants": selected,
        "input_result_sha256": {
            variant: runs[variant]["result_sha256"] for variant in VARIANTS
        },
    }
    destination = CANONICAL_RUN_ROOT / "selection.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise FileExistsError("Frozen selection manifest already exists with different content")
    else:
        destination.write_text(serialized, encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
