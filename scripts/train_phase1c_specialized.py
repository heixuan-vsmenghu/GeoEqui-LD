#!/usr/bin/env python
"""Run the gated, bounded Phase 1C H3 supervised control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.training.phase1c_runners import run_phase1c_formal  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gated Phase 1C H3 specialized B2 run, up to 16 epochs"
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase05_local.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase1c_specialized_enhancers.yaml",
    )
    parser.add_argument("--operator-probe-artifact", type=Path, required=True)
    parser.add_argument("--tiny-artifact", type=Path, required=True)
    parser.add_argument("--tiny-review-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "phase1c" / "gpu_budget.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase1c_formal(
        local_config=args.local_config,
        phase1c_config=args.config,
        operator_probe_artifact=args.operator_probe_artifact,
        tiny_artifact=args.tiny_artifact,
        tiny_review_artifact=args.tiny_review_artifact,
        output_dir=args.output_dir,
        ledger_path=args.ledger,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "budget_exhausted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
