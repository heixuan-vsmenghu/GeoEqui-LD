#!/usr/bin/env python
"""Run a preregistered Phase 1A four-sample A4 or B4 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.training.phase1a_runners import run_tiny_gate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A4 U-Net B0 or B4 HRNet B2 bounded four-sample gate"
    )
    parser.add_argument(
        "--gate",
        choices=("A4_unet_B0", "B4_hrnet_B2"),
        required=True,
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase05_local.yaml",
    )
    parser.add_argument(
        "--hrnet-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase1a_hrnet_shared.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--b3-artifact",
        type=Path,
        default=None,
        help="Required for B4; ignored for the independent A4 diagnostic",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "phase1a" / "gpu_budget.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_tiny_gate(
        gate=args.gate,
        local_config=args.local_config,
        hrnet_config=args.hrnet_config,
        output_dir=args.output_dir,
        ledger_path=args.ledger,
        repository_root=REPOSITORY_ROOT,
        b3_artifact=args.b3_artifact,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.gate == "A4_unet_B0":
        return 0 if result.get("diagnostic_completion") == "completed" else 2
    return 0 if result.get("gate") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
