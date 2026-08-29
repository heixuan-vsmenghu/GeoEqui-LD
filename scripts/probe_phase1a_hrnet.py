#!/usr/bin/env python
"""Run the fixed Phase 1A B3 HRNet structural/resource probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.training.phase1a_runners import run_b3_probe  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="B3: fixed 512px FP32 HRNet-W32 first-step and roundtrip probe"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase1a_hrnet_shared.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "phase1a" / "gpu_budget.json",
    )
    parser.add_argument("--max-seconds", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_seconds <= 0 or args.max_seconds > 900.0:
        raise ValueError("--max-seconds must be in (0, 900]")
    result = run_b3_probe(
        config_path=args.config,
        output_dir=args.output_dir,
        ledger_path=args.ledger,
        repository_root=REPOSITORY_ROOT,
        requested_seconds=args.max_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("gate") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
