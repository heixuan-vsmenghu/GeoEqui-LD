#!/usr/bin/env python
"""Run the gated, bounded H2 split-decoder formal experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.training.phase1b_runners import run_phase1b_split_formal  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gated, bounded 20-epoch H2 split-decoder B2 run"
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase05_local.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase1b_decoder_control.yaml",
    )
    parser.add_argument("--tiny-artifact", type=Path, required=True)
    parser.add_argument("--tiny-review-artifact", type=Path, required=True)
    parser.add_argument("--h1-comparability-artifact", type=Path, required=True)
    parser.add_argument(
        "--h1-run-dir",
        type=Path,
        default=(
            REPOSITORY_ROOT / "runs" / "phase1a" / "H1_shared_B2_seed42_20e"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "phase1b" / "gpu_budget.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase1b_split_formal(
        local_config=args.local_config,
        phase1b_config=args.config,
        tiny_artifact=args.tiny_artifact,
        tiny_review_artifact=args.tiny_review_artifact,
        h1_comparability_artifact=args.h1_comparability_artifact,
        h1_run_dir=args.h1_run_dir,
        output_dir=args.output_dir,
        ledger_path=args.ledger,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "budget_exhausted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
