from __future__ import annotations

from pathlib import Path

import pytest

from geoequi_ld.training.budget import (
    GpuBudgetLedger,
    WallClockBudget,
    require_fresh_output_directory,
)


def test_fresh_output_directory_refuses_existing_content_and_protected_root(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "runs" / "phase06"
    protected.mkdir(parents=True)
    with pytest.raises(PermissionError):
        require_fresh_output_directory(protected / "B0", protected_roots=(protected,))

    destination = require_fresh_output_directory(tmp_path / "runs" / "phase1a" / "new")
    assert destination.is_dir()
    (destination / "marker.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        require_fresh_output_directory(destination)


def test_wall_clock_budget_validates_inputs() -> None:
    with pytest.raises(ValueError):
        WallClockBudget.start(0)
    budget = WallClockBudget.start(10)
    assert 0 < budget.remaining_seconds() <= 10
    assert budget.can_start(0)
    with pytest.raises(ValueError):
        budget.can_start(-1)


def test_gpu_budget_ledger_reserves_and_accounts_serial_runs(tmp_path: Path) -> None:
    ledger = GpuBudgetLedger(tmp_path / "gpu_budget.json", total_limit_seconds=180)
    assert ledger.begin("B3", requested_limit_seconds=30, reserve_after_seconds=60) == 30
    with pytest.raises(RuntimeError):
        ledger.begin("B4", requested_limit_seconds=40)
    snapshot = ledger.finish("B3", elapsed_seconds=12.5, status="completed")
    assert snapshot["used_seconds"] == 12.5
    assert snapshot["remaining_seconds"] == 167.5

    allocation = ledger.begin("B4", requested_limit_seconds=160, reserve_after_seconds=20)
    assert allocation == 147.5
    final = ledger.finish("B4", elapsed_seconds=100, status="budget_exhausted")
    assert final["used_seconds"] == 112.5
    assert final["runs"][-1]["status"] == "budget_exhausted"
