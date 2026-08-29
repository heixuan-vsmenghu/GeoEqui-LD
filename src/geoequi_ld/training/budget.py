"""Fail-closed output and GPU-time budget helpers for bounded experiments."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require_fresh_output_directory(
    path: str | Path,
    *,
    protected_roots: tuple[str | Path, ...] = (),
) -> Path:
    """Create an empty run directory without ever reusing protected results."""

    destination = Path(path).resolve()
    for raw_root in protected_roots:
        root = Path(raw_root).resolve()
        if destination == root or root in destination.parents:
            raise PermissionError(f"Output directory is protected: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Output path is not a directory: {destination}")
        if next(destination.iterdir(), None) is not None:
            raise FileExistsError(f"Refusing to reuse non-empty output directory: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


@dataclass(frozen=True)
class WallClockBudget:
    """A monotonic, conservative deadline for a single bounded run."""

    limit_seconds: float
    started_monotonic: float

    @classmethod
    def start(cls, limit_seconds: float) -> WallClockBudget:
        if limit_seconds <= 0:
            raise ValueError("limit_seconds must be positive")
        return cls(limit_seconds=float(limit_seconds), started_monotonic=time.perf_counter())

    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.started_monotonic)

    def remaining_seconds(self) -> float:
        return max(0.0, self.limit_seconds - self.elapsed_seconds())

    def can_start(self, estimated_unit_seconds: float = 0.0) -> bool:
        if estimated_unit_seconds < 0:
            raise ValueError("estimated_unit_seconds must be non-negative")
        return self.remaining_seconds() >= estimated_unit_seconds


class GpuBudgetLedger:
    """Durable serial ledger for the Phase 1A aggregate GPU-time ceiling."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, *, total_limit_seconds: float) -> None:
        if total_limit_seconds <= 0:
            raise ValueError("total_limit_seconds must be positive")
        self.path = Path(path)
        self.total_limit_seconds = float(total_limit_seconds)

    def _new_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "total_limit_seconds": self.total_limit_seconds,
            "runs": [],
            "active_run": None,
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new_payload()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported GPU budget ledger schema")
        if float(payload.get("total_limit_seconds", -1)) != self.total_limit_seconds:
            raise ValueError("GPU budget limit disagrees with the existing ledger")
        if not isinstance(payload.get("runs"), list):
            raise ValueError("GPU budget ledger has an invalid runs field")
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def snapshot(self) -> dict[str, Any]:
        payload = self._load()
        used = sum(float(run["elapsed_seconds"]) for run in payload["runs"])
        return {
            **payload,
            "used_seconds": used,
            "remaining_seconds": max(0.0, self.total_limit_seconds - used),
        }

    def begin(
        self,
        name: str,
        *,
        requested_limit_seconds: float,
        reserve_after_seconds: float = 0.0,
    ) -> float:
        """Reserve a bounded serial run and return its actual allowed seconds."""

        if not name.strip():
            raise ValueError("Run name must be non-empty")
        if requested_limit_seconds <= 0 or reserve_after_seconds < 0:
            raise ValueError("Requested limit must be positive and reserve non-negative")
        payload = self._load()
        if payload.get("active_run") is not None:
            raise RuntimeError("Another GPU run is still marked active")
        used = sum(float(run["elapsed_seconds"]) for run in payload["runs"])
        available = self.total_limit_seconds - used - float(reserve_after_seconds)
        allocation = min(float(requested_limit_seconds), max(0.0, available))
        if allocation <= 0:
            raise RuntimeError("No aggregate GPU budget remains after the required reserve")
        payload["active_run"] = {
            "name": name,
            "allocated_seconds": allocation,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write(payload)
        return allocation

    def finish(
        self,
        name: str,
        *,
        elapsed_seconds: float,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if status not in {"completed", "budget_exhausted", "failed", "oom"}:
            raise ValueError(f"Unsupported run status: {status}")
        payload = self._load()
        active = payload.get("active_run")
        if not isinstance(active, dict) or active.get("name") != name:
            raise RuntimeError(f"Run is not the active ledger entry: {name}")
        entry = {
            **active,
            "elapsed_seconds": float(elapsed_seconds),
            "status": status,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "details": dict(details or {}),
        }
        payload["runs"].append(entry)
        payload["active_run"] = None
        self._write(payload)
        return self.snapshot()
