#!/usr/bin/env python
"""Run the fixed Phase 1A synthetic Gaussian/DSNT/AoP sanity matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase1a_b0_diagnostics.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.diagnostics.phase1a import (  # noqa: E402
    assert_public_aggregate,
    load_phase1a_protocol,
    require_canonical_path,
    require_public_output_path,
    run_synthetic_sanity,
)
from geoequi_ld.training.engine import write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1A synthetic heatmap decoding checks")
    parser.add_argument("--protocol", type=Path, default=CANONICAL_PROTOCOL)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "phase1a" / "HEATMAP_DECODE_SANITY.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "phase1a" / "HEATMAP_DECODE_SANITY.md",
    )
    return parser


def _markdown(result: dict[str, object]) -> str:
    cases = result["cases"]
    assert isinstance(cases, list)
    rows = []
    for case in cases:
        assert isinstance(case, dict)
        rows.append(
            "| {case_id} | {decoder} | {temperature:.3g} | {raw_heatmap_mse:.8f} | "
            "{probability_entropy_normalized:.6f} | {MRE_ALL:.3f} | {valid} | "
            "{aop_penalized_score_deg:.3f} |".format(
                **case,
                valid="yes" if case["aop_official_valid"] else "no",
            )
        )
    return "\n".join(
        [
            "# Phase 1A synthetic heatmap/DSNT sanity",
            "",
            "This report uses synthetic three-channel geometry only. It does not inspect real "
            "images and does not establish the cause of the saved B0 endpoint.",
            "",
            "H is the raw heatmap used for MSE. P is its spatial Softmax and sums to one. "
            "DSNT is the coordinate expectation under P.",
            "",
            "| case | decoder | T | raw MSE | normalized entropy | MRE px | AoP valid | "
            "penalized AoP score |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Raw heatmap MSE is computed before softmax. Amplitude and temperature are reported "
            "as a fixed diagnostic matrix, not searched for model selection.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = require_canonical_path(
        args.protocol, CANONICAL_PROTOCOL, context="Phase 1A protocol"
    )
    json_output = require_public_output_path(args.json_output, repository_root=REPOSITORY_ROOT)
    markdown_output = require_public_output_path(
        args.markdown_output, repository_root=REPOSITORY_ROOT
    )
    protocol = load_phase1a_protocol(protocol_path)
    result = run_synthetic_sanity(protocol)
    assert_public_aggregate(result)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_output, result)
    markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "case_count": len(result["cases"]),
                "json_report": str(json_output.relative_to(REPOSITORY_ROOT)),
                "markdown_report": str(markdown_output.relative_to(REPOSITORY_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
