#!/usr/bin/env python
"""Run the bounded Phase 1B H1 BatchNorm-state diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.diagnostics.phase1b_bn import (  # noqa: E402
    BN_DIAGNOSTIC_MAX_SECONDS,
    H1_EXPERIMENT_NAME,
    PHASE1B_TOTAL_GPU_SECONDS,
    Phase1BBudgetExceeded,
    require_fresh_phase1b_public_file,
    run_phase1b_bn_diagnostic,
    write_json_strict,
)
from geoequi_ld.training.budget import GpuBudgetLedger  # noqa: E402

RUN_NAME = "BN_short_diagnostic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen H1 best/last on train+validation and diagnose one fixed "
            "train-image-only BatchNorm statistics re-estimation"
        )
    )
    h1_root = REPOSITORY_ROOT / "runs" / "phase1a" / H1_EXPERIMENT_NAME
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase05_local.yaml",
    )
    parser.add_argument("--best-checkpoint", type=Path, default=h1_root / "best.pt")
    parser.add_argument("--last-checkpoint", type=Path, default=h1_root / "last.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--public-output",
        type=Path,
        help="Optional new aggregate-only JSON below reports/phase1b",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "phase1b" / "gpu_budget.json",
    )
    parser.add_argument(
        "--requested-seconds",
        type=float,
        default=BN_DIAGNOSTIC_MAX_SECONDS,
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def _require_phase1b_ledger(path: Path, *, repository_root: Path) -> Path:
    root = repository_root.resolve(strict=True)
    candidate = path.resolve(strict=False)
    canonical = (root / "runs" / "phase1b" / "gpu_budget.json").resolve(strict=False)
    if candidate != canonical:
        raise PermissionError(
            "All Phase 1B GPU work must use canonical runs/phase1b/gpu_budget.json"
        )
    return candidate


def execute_with_ledger(
    args: argparse.Namespace,
    *,
    runner: Callable[..., dict[str, Any]] = run_phase1b_bn_diagnostic,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one diagnostic and close the shared ledger on every terminal path."""

    requested = float(args.requested_seconds)
    if requested <= 0 or requested > BN_DIAGNOSTIC_MAX_SECONDS:
        raise ValueError("--requested-seconds must be in (0, 900]")
    public_output = None
    if args.public_output is not None:
        public_output = require_fresh_phase1b_public_file(
            args.public_output,
            repository_root=repository_root,
        )
    ledger_path = _require_phase1b_ledger(args.ledger, repository_root=repository_root)
    ledger = GpuBudgetLedger(
        ledger_path,
        total_limit_seconds=PHASE1B_TOTAL_GPU_SECONDS,
    )
    allocation = ledger.begin(RUN_NAME, requested_limit_seconds=requested)
    started = time.perf_counter()
    result: dict[str, Any] | None = None
    status = "failed"
    try:
        result = runner(
            local_config=args.local_config,
            best_checkpoint=args.best_checkpoint,
            last_checkpoint=args.last_checkpoint,
            output_dir=args.output_dir,
            repository_root=repository_root,
            device=torch.device(args.device),
            max_runtime_seconds=allocation,
        )
        status = "completed"
    except Phase1BBudgetExceeded:
        status = "budget_exhausted"
        raise
    except torch.cuda.OutOfMemoryError:
        status = "oom"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        elapsed = time.perf_counter() - started
        allocation_exceeded = elapsed > allocation
        if allocation_exceeded:
            status = "budget_exhausted"
        ledger_snapshot = ledger.finish(
            RUN_NAME,
            elapsed_seconds=elapsed,
            status=status,
            details={
                "diagnostic": "H1 best/last fixed BN re-estimation",
                "allocation_exceeded": allocation_exceeded,
            },
        )
    assert result is not None
    if elapsed > allocation:
        raise RuntimeError(
            f"BN diagnostic exceeded its ledger allocation: {elapsed:.3f}s > {allocation:.3f}s"
        )
    if float(ledger_snapshot["used_seconds"]) > PHASE1B_TOTAL_GPU_SECONDS:
        raise RuntimeError("Phase 1B aggregate GPU ledger exceeded 10800 seconds")
    result["gpu_budget"] = {
        "ledger_relative_path": "runs/phase1b/gpu_budget.json",
        "run_name": RUN_NAME,
        "allocated_seconds": allocation,
        "elapsed_seconds": elapsed,
        "total_limit_seconds": PHASE1B_TOTAL_GPU_SECONDS,
    }
    private_path = Path(args.output_dir) / "bn_diagnostics_full.json"
    write_json_strict(private_path, result)
    if public_output is not None:
        write_json_strict(public_output, result["public_aggregate"])
    return result, ledger_snapshot


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, ledger = execute_with_ledger(args)
    except Phase1BBudgetExceeded as error:
        print(str(error), file=sys.stderr)
        return 3
    except torch.cuda.OutOfMemoryError:
        print("Phase 1B BN diagnostic ran out of CUDA memory", file=sys.stderr)
        return 4
    print(
        json.dumps(
            {
                "status": result["status"],
                "private_result": str(Path(args.output_dir) / "bn_diagnostics_full.json"),
                "public_output": str(args.public_output) if args.public_output else None,
                "gpu_budget_used_seconds": ledger["used_seconds"],
                "gpu_budget_remaining_seconds": ledger["remaining_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
