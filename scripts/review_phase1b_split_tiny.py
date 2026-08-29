#!/usr/bin/env python
"""Bind an explicit human decision to the Phase 1B tiny overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.training.phase1b_runners import create_phase1b_tiny_review  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "After inspecting all four restricted overlays, bind a PASS/FAIL human decision "
            "to their hashes and the unchanged tiny result"
        )
    )
    parser.add_argument("--tiny-artifact", type=Path, required=True)
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
    parser.add_argument("--decision", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = create_phase1b_tiny_review(
        tiny_artifact=args.tiny_artifact,
        local_config=args.local_config,
        phase1b_config=args.config,
        decision=args.decision,
        note=args.note,
        output_dir=args.output_dir,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
