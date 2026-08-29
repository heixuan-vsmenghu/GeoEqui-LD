"""Bounded, fail-closed runners for the Phase 1B decoder control.

All data-derived outputs are restricted to ignored ``runs/phase1b`` or
``artifacts/phase1b`` directories.  The module only asks the frozen access
policy for train and validation; testing is never an accepted input.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from geoequi_ld.data.access_policy import load_phase05_local_splits
from geoequi_ld.models.dsnt import DSNT
from geoequi_ld.training.budget import GpuBudgetLedger, WallClockBudget
from geoequi_ld.training.checkpoints import restore_checkpoint, save_checkpoint
from geoequi_ld.training.engine import (
    evaluate_model,
    train_for_steps_bounded,
    train_one_epoch,
    write_history_csv,
    write_json,
)
from geoequi_ld.training.phase1a_runners import (
    TRAIN_SAMPLE_COUNT,
    VALIDATION_SAMPLE_COUNT,
    _dataset,
    b4_gate_passed,
    evaluate_tiny_mode,
    fingerprint_digest,
    load_verified_phase1a_data,
    phase1a_training_config,
    save_tiny_prediction_visualizations,
    select_preregistered_tiny_indices,
)
from geoequi_ld.training.phase1b_config import (
    Phase1BDecoderControlConfig,
    build_phase1b_adam,
    load_phase1b_decoder_config,
)
from geoequi_ld.training.runtime import make_generator, seed_everything

PHASE1B_SEED = 42
TINY_SAMPLE_COUNT = 4
REPLAY_COMPARISON_ATOL = 1.0e-6
REPLAY_COMPARISON_RTOL = 1.0e-6
MILESTONE_FORMAT_VERSION = 1
H1_BINDING_FILENAMES = (
    "config.json",
    "formal_result.json",
    "train_log.csv",
    "best.pt",
    "last.pt",
)
RUNTIME_SOURCE_FILENAMES = (
    "src/geoequi_ld/data/access_policy.py",
    "src/geoequi_ld/data/dataset.py",
    "src/geoequi_ld/data/heatmaps.py",
    "src/geoequi_ld/geometry/aop.py",
    "src/geoequi_ld/geometry/coordinates.py",
    "src/geoequi_ld/geometry/transforms.py",
    "src/geoequi_ld/metrics/keypoints.py",
    "src/geoequi_ld/models/decoding.py",
    "src/geoequi_ld/models/dsnt.py",
    "src/geoequi_ld/models/hrnet.py",
    "src/geoequi_ld/training/budget.py",
    "src/geoequi_ld/training/checkpoints.py",
    "src/geoequi_ld/training/config.py",
    "src/geoequi_ld/training/engine.py",
    "src/geoequi_ld/training/phase1a_config.py",
    "src/geoequi_ld/training/phase1a_runners.py",
    "src/geoequi_ld/training/phase1b_config.py",
    "src/geoequi_ld/training/phase1b_runners.py",
    "src/geoequi_ld/training/runtime.py",
)
TINY_OVERLAY_FILENAMES = tuple(f"sample_{index:02d}.png" for index in range(4))


def _contains_forbidden_component(path: str | Path) -> bool:
    return any(part.casefold() in {"test", "testing"} for part in Path(path).parts)


def require_phase1b_fresh_output(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Create one new private Phase 1B directory without touching older phases."""

    root = Path(repository_root).resolve()
    destination = Path(path).resolve()
    allowed = ((root / "runs" / "phase1b").resolve(), (root / "artifacts" / "phase1b").resolve())
    if not any(destination != base and destination.is_relative_to(base) for base in allowed):
        raise PermissionError(
            "Phase 1B output must be a child of runs/phase1b or artifacts/phase1b"
        )
    if _contains_forbidden_component(destination):
        raise PermissionError("Phase 1B refuses a test/testing output path")
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Output path is not a directory: {destination}")
        if next(destination.iterdir(), None) is not None:
            raise FileExistsError(f"Refusing to reuse non-empty output directory: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def _require_private_phase1b_file(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    root = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    allowed = ((root / "runs" / "phase1b").resolve(), (root / "artifacts" / "phase1b").resolve())
    if not any(candidate.is_relative_to(base) for base in allowed):
        raise PermissionError("Phase 1B artifact must stay below runs/phase1b or artifacts/phase1b")
    if _contains_forbidden_component(candidate):
        raise PermissionError("Phase 1B refuses a test/testing artifact path")
    if not candidate.is_file():
        raise FileNotFoundError(f"Phase 1B artifact does not exist: {candidate}")
    return candidate


def _require_frozen_h1_run(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    root = Path(repository_root).resolve()
    run = Path(path).resolve()
    allowed = (root / "runs" / "phase1a").resolve()
    if run == allowed or not run.is_relative_to(allowed):
        raise PermissionError("H1 reference must be a child of the frozen runs/phase1a directory")
    if _contains_forbidden_component(run):
        raise PermissionError("H1 reference cannot contain test/testing")
    missing = [name for name in H1_BINDING_FILENAMES if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen H1 reference is incomplete: {missing}")
    return run


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: str | Path, *, logical_name: str) -> dict[str, Any]:
    candidate = Path(path)
    return {
        "logical_name": logical_name,
        "size_bytes": candidate.stat().st_size,
        "sha256": _sha256_file(candidate),
    }


def _h1_reference_binding(h1_run: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": "H1_shared_B2_seed42_20e",
        "files": {
            name: _file_binding(h1_run / name, logical_name=name)
            for name in H1_BINDING_FILENAMES
        },
    }


def _runtime_source_binding(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    files = {
        name: _file_binding(root / Path(name), logical_name=name)
        for name in RUNTIME_SOURCE_FILENAMES
    }
    try:
        head_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "src/geoequi_ld"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        git_identity = {
            "head_tree": head_tree,
            "tracked_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "tracked_source_diff_size_bytes": len(diff),
        }
    except (OSError, subprocess.CalledProcessError):
        git_identity = {
            "head_tree": "unavailable",
            "tracked_source_diff_sha256": "unavailable",
            "tracked_source_diff_size_bytes": None,
        }
    return {"files": files, "git_source_identity": git_identity}


def _phase1b_ledger(
    path: str | Path,
    *,
    repository_root: str | Path,
    config: Phase1BDecoderControlConfig,
) -> GpuBudgetLedger:
    root = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    canonical = (root / "runs" / "phase1b" / "gpu_budget.json").resolve()
    if candidate != canonical:
        raise PermissionError(
            "All Phase 1B GPU work must use canonical runs/phase1b/gpu_budget.json"
        )
    return GpuBudgetLedger(
        candidate,
        total_limit_seconds=float(config.resources.total_gpu_max_seconds),
    )


def _finish_ledger(
    ledger: GpuBudgetLedger,
    name: str,
    *,
    started: float,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    before = ledger.snapshot()
    active = before.get("active_run")
    if active is None:
        return before
    if not isinstance(active, Mapping) or active.get("name") != name:
        raise RuntimeError(f"Unexpected active Phase 1B GPU ledger entry: {active}")
    allocation = float(active["allocated_seconds"])
    elapsed = time.perf_counter() - started
    aggregate_after = float(before["used_seconds"]) + elapsed
    allocation_exceeded = elapsed > allocation
    aggregate_exceeded = aggregate_after > float(before["total_limit_seconds"])
    effective_status = (
        "budget_exhausted"
        if allocation_exceeded or aggregate_exceeded
        else status
    )
    effective_details = {
        **dict(details or {}),
        "allocation_exceeded": allocation_exceeded,
        "aggregate_limit_exceeded": aggregate_exceeded,
    }
    snapshot = ledger.finish(
        name,
        elapsed_seconds=elapsed,
        status=effective_status,
        details=effective_details,
    )
    return snapshot


def _ledger_run_binding(
    snapshot: Mapping[str, Any],
    name: str,
    *,
    run_index: int | None = None,
) -> dict[str, Any]:
    runs = snapshot.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Phase 1B GPU ledger has no runs list")
    if run_index is None:
        matching = [
            (index, entry)
            for index, entry in enumerate(runs)
            if isinstance(entry, Mapping) and entry.get("name") == name
        ]
        if not matching:
            raise ValueError(f"No Phase 1B ledger entry exists for {name}")
        index, entry = matching[-1]
    else:
        if run_index < 0 or run_index >= len(runs):
            raise ValueError(f"Phase 1B ledger index is out of range: {run_index}")
        index, entry = run_index, runs[run_index]
        if not isinstance(entry, Mapping) or entry.get("name") != name:
            raise ValueError(f"Phase 1B ledger index {run_index} is not {name}")
    canonical_entry = json.dumps(
        entry,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "ledger_relative_path": "runs/phase1b/gpu_budget.json",
        "run_index": index,
        "entry": dict(entry),
        "entry_sha256": hashlib.sha256(canonical_entry).hexdigest(),
        "total_limit_seconds": float(snapshot["total_limit_seconds"]),
    }


def _formal_runtime_allocation_outcome(
    *,
    elapsed_seconds: float,
    allocated_seconds: float,
    ledger_binding: Mapping[str, Any],
) -> dict[str, bool]:
    """Separate a training sub-budget stop from a formal allocation overrun."""

    entry = ledger_binding.get("entry")
    if not isinstance(entry, Mapping):
        raise ValueError("Phase 1B formal ledger binding has no entry")
    details = entry.get("details")
    if not isinstance(details, Mapping):
        raise ValueError("Phase 1B formal ledger entry has no details")
    allocation_flag = details.get("allocation_exceeded")
    aggregate_flag = details.get("aggregate_limit_exceeded")
    if not isinstance(allocation_flag, bool) or not isinstance(aggregate_flag, bool):
        raise ValueError("Phase 1B formal ledger exceed flags must be boolean")

    ledger_elapsed = float(entry["elapsed_seconds"])
    ledger_allocation = float(entry["allocated_seconds"])
    formal_allocation_exceeded = (
        float(elapsed_seconds) > float(allocated_seconds)
        or ledger_elapsed > ledger_allocation
        or allocation_flag
    )
    aggregate_gpu_cap_exceeded = aggregate_flag
    return {
        "formal_allocation_exceeded": formal_allocation_exceeded,
        "aggregate_gpu_cap_exceeded": aggregate_gpu_cap_exceeded,
        "within_runtime_allocation": not (
            formal_allocation_exceeded or aggregate_gpu_cap_exceeded
        ),
    }


def _require_ledger_binding_current(
    binding: Mapping[str, Any],
    *,
    name: str,
    ledger_path: str | Path,
    repository_root: str | Path,
    config: Phase1BDecoderControlConfig,
    required_details: Mapping[str, Any],
) -> None:
    ledger = _phase1b_ledger(
        ledger_path,
        repository_root=repository_root,
        config=config,
    )
    snapshot = ledger.snapshot()
    try:
        run_index = int(binding["run_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError(f"Invalid GPU ledger run index for {name}") from error
    expected = _ledger_run_binding(snapshot, name, run_index=run_index)
    if dict(binding) != expected:
        raise PermissionError(f"Stale or invented GPU ledger binding for {name}")
    entry = expected["entry"]
    if entry.get("status") != "completed":
        raise PermissionError(f"Bound GPU ledger entry for {name} is not completed")
    details = entry.get("details")
    if not isinstance(details, Mapping) or any(
        details.get(key) != value for key, value in required_details.items()
    ):
        raise PermissionError(f"Bound GPU ledger details do not prove {name} outcome")
    if float(entry["elapsed_seconds"]) > float(entry["allocated_seconds"]):
        raise PermissionError(f"Recorded {name} elapsed time exceeds its allocation")
    if float(snapshot["used_seconds"]) > float(snapshot["total_limit_seconds"]):
        raise PermissionError("Phase 1B aggregate GPU ledger exceeds 10800 seconds")


def _tensor_state_digest(state: Mapping[str, Tensor]) -> str:
    """Hash a local tensor state without serializing private paths or values."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _runtime_environment(repository_root: str | Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unavailable", None
    cuda_available = torch.cuda.is_available()
    return {
        "git_head": head,
        "git_worktree_dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "timm": importlib.metadata.version("timm"),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "precision": "float32",
        "amp_enabled": False,
    }


def phase1b_training_config():
    """Return the locked B2 engine configuration used by H1 and H2."""

    return phase1a_training_config(gate=None)


def load_verified_phase1b_data(
    local_config: str | Path,
    *,
    repository_root: str | Path,
):
    """Require the actual split contract to equal the canonical local access policy."""

    if _contains_forbidden_component(local_config):
        raise PermissionError("Phase 1B local configuration cannot contain test/testing")
    verified = load_verified_phase1a_data(local_config)
    canonical_path = Path(repository_root).resolve() / "configs" / "phase05_local.yaml"
    if not canonical_path.is_file():
        raise FileNotFoundError("Canonical configs/phase05_local.yaml is required")
    canonical = load_phase05_local_splits(canonical_path)
    for role in ("train", "validation"):
        actual_spec = verified.specs[role]
        canonical_spec = canonical[role]
        equivalent = bool(
            actual_spec.image_dir.resolve() == canonical_spec.image_dir.resolve()
            and actual_spec.labels_csv.resolve() == canonical_spec.labels_csv.resolve()
            and actual_spec.fh1_column == canonical_spec.fh1_column
            and actual_spec.expected_fingerprint == canonical_spec.expected_fingerprint
        )
        if not equivalent:
            raise PermissionError(
                f"Phase 1B {role} split differs from canonical phase05_local access policy"
            )
    return verified


def _build_paired_initialization() -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Create split and shared models from one seed-42 shared state on CPU."""

    from geoequi_ld.models.hrnet import HRNetW32SharedHeatmap, HRNetW32SplitHeatmap

    seed_everything(PHASE1B_SEED, deterministic=True)
    shared = HRNetW32SharedHeatmap(align_corners=True)
    shared.eval()
    split = HRNetW32SplitHeatmap.from_shared(shared)
    split.eval()
    probe = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    with torch.inference_mode():
        shared_output = shared(probe)
        split_output = split(probe)
    maximum_difference = float((shared_output - split_output).abs().max())
    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    split_parameters = sum(parameter.numel() for parameter in split.parameters())
    metadata = {
        "seed": PHASE1B_SEED,
        "method": "HRNetW32SplitHeatmap.from_shared",
        "shared_state_sha256": _tensor_state_digest(shared.state_dict()),
        "split_state_sha256": _tensor_state_digest(split.state_dict()),
        "shared_trainable_parameters": shared_parameters,
        "split_trainable_parameters": split_parameters,
        "additional_trainable_parameters": split_parameters - shared_parameters,
        "probe_input_shape": list(probe.shape),
        "probe_output_shape": list(shared_output.shape),
        "maximum_absolute_output_difference": maximum_difference,
        "output_equivalence_atol": 1.0e-6,
        "output_equivalence_rtol": 1.0e-5,
        "output_equivalent": bool(
            torch.allclose(split_output, shared_output, rtol=1.0e-5, atol=1.0e-6)
        ),
    }
    if not metadata["output_equivalent"]:
        raise RuntimeError("Shared-to-split initialization failed the output equivalence check")
    return shared, split, metadata


def _save_model_only(path: str | Path, *, model: nn.Module, epoch: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MILESTONE_FORMAT_VERSION,
        "checkpoint_kind": "model_only_milestone",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def _tiny_programmatic_payload_valid(payload: Mapping[str, Any]) -> bool:
    metrics = payload.get("eval_mode")
    visualization = payload.get("visualization")
    initialization = payload.get("initialization")
    return bool(
        payload.get("schema_version") == 1
        and payload.get("gate_id") == "P1B_split_tiny_B2"
        and payload.get("status") == "completed"
        and payload.get("gate") == "PENDING_REVIEW"
        and payload.get("programmatic_gate") == "PASS"
        and payload.get("steps_completed") == 500
        and payload.get("max_steps") == 500
        and payload.get("within_total_allocation") is True
        and payload.get("augmentation") == "disabled"
        and payload.get("batch_size") == 1
        and payload.get("precision") == "float32"
        and payload.get("loss_application")
        == "concatenate_[PS1,PS2,FH1]_then_existing_B2_once"
        and isinstance(metrics, Mapping)
        and b4_gate_passed(metrics)
        and isinstance(visualization, Mapping)
        and visualization.get("programmatic_check_passed") is True
        and visualization.get("manual_review_status") == "pending"
        and visualization.get("visualization_count") == TINY_SAMPLE_COUNT
        and visualization.get("private_relative_directory") == "predictions"
        and isinstance(initialization, Mapping)
        and initialization.get("output_equivalent") is True
    )


def _tiny_overlay_binding(tiny_result_path: Path) -> dict[str, Any]:
    predictions = tiny_result_path.parent / "predictions"
    if not predictions.is_dir():
        raise FileNotFoundError("Phase 1B tiny predictions directory is missing")
    actual_pngs = sorted(path.name for path in predictions.glob("*.png"))
    if actual_pngs != list(TINY_OVERLAY_FILENAMES):
        raise PermissionError(
            f"Phase 1B tiny review requires exactly four canonical overlays: {actual_pngs}"
        )
    return {
        name: _file_binding(predictions / name, logical_name=f"predictions/{name}")
        for name in TINY_OVERLAY_FILENAMES
    }


def _require_current_tiny_context(
    payload: Mapping[str, Any],
    *,
    data_fingerprint_digest: str,
    phase1b_config: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    current = {
        "data_fingerprint_digest": data_fingerprint_digest,
        "protocol_config_binding": _file_binding(
            phase1b_config,
            logical_name="configs/phase1b_decoder_control.yaml",
        ),
        "environment": _runtime_environment(repository_root),
        "runtime_source_binding": _runtime_source_binding(repository_root),
    }
    mismatched = [name for name, value in current.items() if payload.get(name) != value]
    if mismatched:
        raise PermissionError(f"Phase 1B tiny evidence is stale for: {mismatched}")
    return current


def create_phase1b_tiny_review(
    *,
    tiny_artifact: str | Path,
    local_config: str | Path,
    phase1b_config: str | Path,
    decision: str,
    output_dir: str | Path,
    repository_root: str | Path,
    note: str = "",
) -> dict[str, Any]:
    """Record an explicit human decision bound to raw metrics and four overlays."""

    artifact = _require_private_phase1b_file(
        tiny_artifact,
        repository_root=repository_root,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not _tiny_programmatic_payload_valid(payload):
        raise PermissionError("Only a canonical PENDING_REVIEW tiny result can be reviewed")
    verified = load_verified_phase1b_data(
        local_config,
        repository_root=repository_root,
    )
    review_context = _require_current_tiny_context(
        payload,
        data_fingerprint_digest=fingerprint_digest(verified.fingerprints),
        phase1b_config=phase1b_config,
        repository_root=repository_root,
    )
    normalized_decision = decision.strip().upper()
    if normalized_decision not in {"PASS", "FAIL"}:
        raise ValueError("Tiny human review decision must be PASS or FAIL")
    output = require_phase1b_fresh_output(output_dir, repository_root=repository_root)
    review = {
        "schema_version": 1,
        "gate_id": "P1B_split_tiny_B2_human_review",
        "decision": normalized_decision,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": (
            "Four restricted target/prediction overlays and their PS1/PS2/FH1 heatmaps"
        ),
        "review_note": note,
        "review_context": review_context,
        "tiny_result_binding": _file_binding(
            artifact,
            logical_name="tiny_result.json",
        ),
        "overlay_bindings": _tiny_overlay_binding(artifact),
    }
    write_json(output / "tiny_review.json", review)
    return review


def require_passed_phase1b_tiny_artifact(
    path: str | Path,
    review_path: str | Path,
    *,
    data_fingerprint_digest: str,
    phase1b_config: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    artifact = _require_private_phase1b_file(path, repository_root=repository_root)
    review_artifact = _require_private_phase1b_file(
        review_path,
        repository_root=repository_root,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    review = json.loads(review_artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not _tiny_programmatic_payload_valid(payload):
        raise PermissionError(
            "Phase 1B formal training requires an unchanged PENDING_REVIEW tiny result"
        )
    current_context = _require_current_tiny_context(
        payload,
        data_fingerprint_digest=data_fingerprint_digest,
        phase1b_config=phase1b_config,
        repository_root=repository_root,
    )
    ledger_binding = payload.get("gpu_ledger_binding")
    if not isinstance(ledger_binding, Mapping):
        raise PermissionError("Phase 1B tiny result has no canonical GPU ledger binding")
    _require_ledger_binding_current(
        ledger_binding,
        name="P1B_split_tiny_B2",
        ledger_path=ledger_path,
        repository_root=repository_root,
        config=load_phase1b_decoder_config(phase1b_config),
        required_details={
            "gate": "PENDING_REVIEW",
            "programmatic_gate": "PASS",
            "steps_completed": 500,
        },
    )
    expected_result_binding = _file_binding(artifact, logical_name="tiny_result.json")
    expected_overlays = _tiny_overlay_binding(artifact)
    if not isinstance(review, Mapping) or not (
        review.get("schema_version") == 1
        and review.get("gate_id") == "P1B_split_tiny_B2_human_review"
        and review.get("decision") == "PASS"
        and review.get("review_scope")
        == "Four restricted target/prediction overlays and their PS1/PS2/FH1 heatmaps"
        and review.get("review_context") == current_context
        and review.get("tiny_result_binding") == expected_result_binding
        and review.get("overlay_bindings") == expected_overlays
    ):
        raise PermissionError(
            "Phase 1B formal training requires a PASS review bound to the exact tiny result "
            "and four overlay hashes"
        )
    return {
        "tiny_result": dict(payload),
        "human_review": dict(review),
        "bindings_recomputed": True,
    }


def run_phase1b_split_tiny(
    *,
    local_config: str | Path,
    phase1b_config: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Run the fixed four-sample, 500-step split-decoder implementation check."""

    protocol = load_phase1b_decoder_config(phase1b_config)
    output = require_phase1b_fresh_output(output_dir, repository_root=repository_root)
    verified = load_verified_phase1b_data(
        local_config,
        repository_root=repository_root,
    )
    config = phase1b_training_config()
    train_dataset = _dataset(verified.specs["train"], config)
    indices = select_preregistered_tiny_indices(len(train_dataset))
    subset = Subset(train_dataset, indices)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=make_generator(PHASE1B_SEED + 1),
        pin_memory=True,
    )
    evaluation_loader: DataLoader[dict[str, Any]] = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    if not torch.cuda.is_available():
        result = {
            "schema_version": 1,
            "gate_id": "P1B_split_tiny_B2",
            "gate": "FAIL",
            "status": "cuda_unavailable",
        }
        write_json(output / "tiny_result.json", result)
        return result

    shared, model, initialization = _build_paired_initialization()
    del shared
    ledger = _phase1b_ledger(
        ledger_path,
        repository_root=repository_root,
        config=protocol,
    )
    allocation = ledger.begin(
        "P1B_split_tiny_B2",
        requested_limit_seconds=float(protocol.resources.tiny_max_seconds),
    )
    diagnostic_reserve = min(120.0, allocation * 0.1)
    training_allocation = allocation - diagnostic_reserve
    if training_allocation <= 0:
        raise RuntimeError("No tiny-run budget remains after the diagnostic reserve")
    started = time.perf_counter()
    try:
        device = torch.device("cuda")
        model = model.to(device)
        optimizer = build_phase1b_adam(model.parameters(), protocol)
        dsnt = DSNT(temperature=0.05, align_corners=True).to(device)
        bounded = train_for_steps_bounded(
            model,
            train_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
            max_steps=protocol.resources.tiny_max_steps,
            max_runtime_seconds=training_allocation,
        )
        eval_metrics, private_rows = evaluate_tiny_mode(
            model,
            evaluation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            mode="eval",
        )
        visualization = save_tiny_prediction_visualizations(
            model,
            evaluation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            private_run_root=output,
            repository_root=repository_root,
        )
        coordinate_checks = visualization["programmatic_coordinate_checks"]
        programmatic_visual_pass = bool(
            visualization["visualization_count"] == TINY_SAMPLE_COUNT
            and coordinate_checks["all_coordinates_finite"]
            and coordinate_checks["all_targets_in_input_bounds"]
            and coordinate_checks["all_predictions_in_input_bounds"]
            and float(coordinate_checks["max_target_normalized_roundtrip_error_px"]) <= 1.0e-3
        )
        programmatic_pass = bool(
            bounded.status == "completed"
            and b4_gate_passed(eval_metrics)
            and programmatic_visual_pass
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "gate_id": "P1B_split_tiny_B2",
            "gate": "PENDING_REVIEW" if programmatic_pass else "FAIL",
            "programmatic_gate": "PASS" if programmatic_pass else "FAIL",
            "status": bounded.status,
            "steps_completed": bounded.steps_completed,
            "max_steps": protocol.resources.tiny_max_steps,
            "training_elapsed_seconds": bounded.elapsed_seconds,
            "training_allocated_seconds": training_allocation,
            "total_allocated_seconds": allocation,
            "initialization": initialization,
            "sample_selection": {
                "algorithm": "torch.randperm(train=300, seed=42)[:4]",
                "sample_count": TINY_SAMPLE_COUNT,
                "indices_omitted_from_gate_result": True,
            },
            "data_fingerprint_digest": fingerprint_digest(verified.fingerprints),
            "protocol_config_binding": _file_binding(
                phase1b_config,
                logical_name="configs/phase1b_decoder_control.yaml",
            ),
            "environment": _runtime_environment(repository_root),
            "runtime_source_binding": _runtime_source_binding(repository_root),
            "eval_mode": eval_metrics,
            "visualization": {
                **visualization,
                "programmatic_check_passed": programmatic_visual_pass,
                "manual_review_status": "pending",
            },
            "augmentation": "disabled",
            "batch_size": 1,
            "precision": "float32",
            "loss_application": "concatenate_[PS1,PS2,FH1]_then_existing_B2_once",
        }
        write_history_csv(output / "train_log.csv", bounded.history)
        write_json(
            output / "private_predictions.json",
            {
                "restricted_local_output": True,
                "selected_indices": list(indices),
                "eval_mode": private_rows,
            },
        )
        model.eval()
        save_checkpoint(
            output / "tiny.pt",
            model=model,
            optimizer=optimizer,
            epoch=0,
            config={
                "phase": "phase1b",
                "experiment_name": "P1B_split_tiny_B2",
                "testing_frozen": True,
                "training": config.to_dict(),
                "model": protocol.model.to_dict(),
            },
            seed=PHASE1B_SEED,
            metrics=eval_metrics,
            extra={"steps_completed": bounded.steps_completed, "status": bounded.status},
        )
        total_elapsed = time.perf_counter() - started
        result["total_elapsed_seconds"] = total_elapsed
        result["within_total_allocation"] = total_elapsed <= allocation
        if not result["within_total_allocation"]:
            result["gate"] = "FAIL"
            result["programmatic_gate"] = "FAIL"
        ledger_status = "completed" if bounded.status == "completed" else "budget_exhausted"
        ledger_snapshot = _finish_ledger(
            ledger,
            "P1B_split_tiny_B2",
            started=started,
            status=ledger_status,
            details={
                "gate": result["gate"],
                "programmatic_gate": result["programmatic_gate"],
                "steps_completed": bounded.steps_completed,
            },
        )
        result["gpu_ledger_binding"] = _ledger_run_binding(
            ledger_snapshot,
            "P1B_split_tiny_B2",
        )
        if result["gpu_ledger_binding"]["entry"]["status"] != "completed":
            result["status"] = "budget_exhausted"
            result["gate"] = "FAIL"
            result["programmatic_gate"] = "FAIL"
            result["within_total_allocation"] = False
        write_json(output / "tiny_result.json", result)
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "gate_id": "P1B_split_tiny_B2",
            "gate": "FAIL",
            "status": "oom",
            "adaptation_attempted": False,
        }
        write_json(output / "tiny_result.json", result)
        _finish_ledger(ledger, "P1B_split_tiny_B2", started=started, status="oom")
        return result
    except Exception:
        _finish_ledger(ledger, "P1B_split_tiny_B2", started=started, status="failed")
        raise


def _load_h1_reference_epoch1(h1_run: Path) -> dict[str, Any]:
    with (h1_run / "train_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or int(rows[0]["epoch"]) != 1:
        raise ValueError("Frozen H1 log must contain epochs 1 through 20")
    return rows[0]


def _expected_h1_static_contract(
    *,
    current_fingerprint_digest: str,
) -> dict[str, Any]:
    config = phase1b_training_config()
    return {
        "phase": "phase1a",
        "experiment_name": "H1_shared_B2_seed42_20e",
        "testing_frozen": True,
        "training": json.loads(json.dumps(config.to_dict())),
        "model": {
            "backbone": "hrnet_w32",
            "timm_version": "1.0.28",
            "pretrained": False,
            "in_channels": 1,
            "out_channels": 3,
            "feature_location": "",
            "out_indices": [1],
            "feature_channels": 32,
            "feature_reduction": 4,
            "decoder_channels": [32, 16],
            "decoder_normalization": "BatchNorm2d",
            "decoder_activation": "GELU",
            "interpolation_mode": "bilinear",
            "align_corners": True,
            "class": "HRNetW32SharedHeatmap",
        },
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "amsgrad": False,
            "foreach": False,
            "class": "Adam",
        },
        "data": {
            "train_count": TRAIN_SAMPLE_COUNT,
            "validation_count": VALIDATION_SAMPLE_COUNT,
            "fingerprint_digest": current_fingerprint_digest,
            "paths_embedded": False,
        },
    }


def audit_h1_comparability(
    *,
    h1_run_dir: str | Path,
    current_fingerprint_digest: str,
    repository_root: str | Path,
    replay_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Audit frozen H1 without claiming pairing unless epoch-1 replay also matches."""

    h1_run = _require_frozen_h1_run(h1_run_dir, repository_root=repository_root)
    stored_config = json.loads((h1_run / "config.json").read_text(encoding="utf-8"))
    formal = json.loads((h1_run / "formal_result.json").read_text(encoding="utf-8"))
    expected = _expected_h1_static_contract(
        current_fingerprint_digest=current_fingerprint_digest,
    )
    static_checks = {
        key: stored_config.get(key) == expected_value for key, expected_value in expected.items()
    }
    static_checks.update(
        {
            "twenty_epochs_completed": (
                formal.get("status") == "completed"
                and formal.get("epochs_completed") == 20
                and formal.get("epochs_requested") == 20
            ),
            "validation_selection_order": formal.get("selection_order")
            == ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
        }
    )
    replay_passed = False
    if replay_artifact is not None:
        try:
            validate_h1_replay_artifact(
                replay_artifact=replay_artifact,
                h1_run_dir=h1_run,
                current_fingerprint_digest=current_fingerprint_digest,
                repository_root=repository_root,
                ledger_path=Path(repository_root)
                / "runs"
                / "phase1b"
                / "gpu_budget.json",
                phase1b_config=Path(repository_root)
                / "configs"
                / "phase1b_decoder_control.yaml",
            )
            replay_passed = True
        except (PermissionError, ValueError, FileNotFoundError, KeyError, TypeError):
            replay_passed = False
    comparable = all(static_checks.values()) and replay_passed
    return {
        "schema_version": 1,
        "shared_reference": "H1_shared_B2_seed42_20e",
        "static_checks": static_checks,
        "epoch1_replay_passed": replay_passed,
        "comparison_classification": (
            "frozen_shared_comparator" if comparable else "historical_reference_only"
        ),
        "strictly_comparable_for_phase1b": comparable,
        "limitation": (
            None
            if comparable
            else (
                "Initialization/order/runtime equivalence is not accepted without "
                "a matching epoch-1 replay."
            )
        ),
    }


def _numeric_close(actual: float, expected: float) -> tuple[bool, float]:
    difference = abs(actual - expected)
    limit = REPLAY_COMPARISON_ATOL + REPLAY_COMPARISON_RTOL * abs(expected)
    return difference <= limit, difference


def _compare_replay_metrics(
    reference: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    actual = {
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in validation_metrics.items()},
    }
    comparisons: dict[str, Any] = {}
    required = (
        "train_total_loss",
        "train_heatmap_mse",
        "train_coordinate_smooth_l1",
        "train_distribution_js",
        "train_batches",
        "val_total_loss",
        "val_heatmap_mse",
        "val_coordinate_smooth_l1",
        "val_distribution_js",
        "val_MRE_PS1",
        "val_MRE_PS2",
        "val_MRE_FH1",
        "val_MRE_ALL",
        "val_n_samples",
        "val_n_valid_aop",
        "val_n_evaluable_aop",
        "val_aop_invalid_prediction_count",
        "val_aop_mae_valid_deg",
        "val_aop_mae_deg",
    )
    all_match = True
    for name in required:
        expected_value = float(reference[name])
        actual_value = float(actual[name])
        matches, difference = _numeric_close(actual_value, expected_value)
        comparisons[name] = {
            "matches": matches,
            "absolute_difference": difference,
            "reference": expected_value,
            "replay": actual_value,
        }
        all_match = all_match and matches
    return all_match, comparisons


def validate_h1_replay_artifact(
    *,
    replay_artifact: str | Path,
    h1_run_dir: str | Path,
    current_fingerprint_digest: str,
    repository_root: str | Path,
    ledger_path: str | Path,
    phase1b_config: str | Path,
) -> dict[str, Any]:
    """Recompute every H1 replay claim from frozen files and current runtime."""

    artifact = _require_private_phase1b_file(
        replay_artifact,
        repository_root=repository_root,
    )
    loaded = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise PermissionError("H1 replay artifact root must be a mapping")
    h1_run = _require_frozen_h1_run(h1_run_dir, repository_root=repository_root)
    static = audit_h1_comparability(
        h1_run_dir=h1_run,
        current_fingerprint_digest=current_fingerprint_digest,
        repository_root=repository_root,
    )
    train_metrics = loaded.get("train_metrics")
    validation_metrics = loaded.get("validation_metrics")
    if not isinstance(train_metrics, Mapping) or not isinstance(validation_metrics, Mapping):
        raise PermissionError("H1 replay artifact is missing raw train/validation metrics")
    reference = _load_h1_reference_epoch1(h1_run)
    replay_matches, recomputed_comparisons = _compare_replay_metrics(
        reference,
        train_metrics,
        validation_metrics,
    )
    expected_environment = _runtime_environment(repository_root)
    expected_sources = _runtime_source_binding(repository_root)
    expected_h1_binding = _h1_reference_binding(h1_run)
    expected_config_binding = _file_binding(
        phase1b_config,
        logical_name="configs/phase1b_decoder_control.yaml",
    )
    try:
        runtime_within_allocation = float(loaded["total_elapsed_seconds"]) <= float(
            loaded["total_allocated_seconds"]
        )
    except (KeyError, TypeError, ValueError):
        runtime_within_allocation = False
    required_claims = {
        "schema_version": loaded.get("schema_version") == 1,
        "gate_id": loaded.get("gate_id") == "H1_epoch1_deterministic_replay",
        "status": loaded.get("status") == "completed",
        "comparison": loaded.get("comparison") == "PASS",
        "steps": (
            loaded.get("steps_completed") == TRAIN_SAMPLE_COUNT
            and loaded.get("steps_requested") == TRAIN_SAMPLE_COUNT
        ),
        "tolerance": (
            loaded.get("atol") == REPLAY_COMPARISON_ATOL
            and loaded.get("rtol") == REPLAY_COMPARISON_RTOL
        ),
        "fingerprint": loaded.get("reference_fingerprint_digest")
        == current_fingerprint_digest,
        "static_checks": loaded.get("static_checks") == static["static_checks"]
        and all(static["static_checks"].values()),
        "metric_recomputation": replay_matches
        and loaded.get("metric_comparisons") == recomputed_comparisons,
        "h1_binding": loaded.get("h1_reference_binding") == expected_h1_binding,
        "config_binding": loaded.get("protocol_config_binding") == expected_config_binding,
        "environment": loaded.get("environment") == expected_environment,
        "runtime_sources": loaded.get("runtime_source_binding") == expected_sources,
        "environment_scope": (
            loaded.get("historical_environment_record_available") is False
            and loaded.get("environment_evidence_scope")
            == "same_current_environment_exact_epoch1_replay"
        ),
        "classification": (
            loaded.get("strictly_comparable_for_phase1b") is True
            and loaded.get("comparison_classification") == "frozen_shared_comparator"
        ),
        "runtime_allocation": runtime_within_allocation,
    }
    failures = [name for name, passed in required_claims.items() if not passed]
    if failures:
        raise PermissionError(
            "H1 replay evidence failed strict recomputation; a same-protocol shared rerun "
            f"is required. Failed checks: {failures}"
        )
    ledger_binding = loaded.get("gpu_ledger_binding")
    if not isinstance(ledger_binding, Mapping):
        raise PermissionError("H1 replay artifact has no canonical GPU ledger binding")
    _require_ledger_binding_current(
        ledger_binding,
        name="H1_epoch1_deterministic_replay",
        ledger_path=ledger_path,
        repository_root=repository_root,
        config=load_phase1b_decoder_config(phase1b_config),
        required_details={"comparison": "PASS", "steps_completed": 300},
    )
    return {
        "evidence_gate_id": "H1_epoch1_deterministic_replay",
        "epoch1_replay_comparison": "PASS",
        "strictly_comparable_for_phase1b": True,
        "comparison_classification": "frozen_shared_comparator",
        "recomputed_checks": required_claims,
        "environment_evidence_scope": "same_current_environment_exact_epoch1_replay",
        "limitation": (
            "The historical H1 run did not persist a complete environment record; runtime "
            "consistency is supported by exact epoch-1 replay in the same current environment."
        ),
    }


def run_h1_epoch1_replay(
    *,
    local_config: str | Path,
    phase1b_config: str | Path,
    h1_run_dir: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Replay exactly one H1 epoch to test initialization/order reproducibility."""

    protocol = load_phase1b_decoder_config(phase1b_config)
    h1_run = _require_frozen_h1_run(h1_run_dir, repository_root=repository_root)
    output = require_phase1b_fresh_output(output_dir, repository_root=repository_root)
    verified = load_verified_phase1b_data(
        local_config,
        repository_root=repository_root,
    )
    current_digest = fingerprint_digest(verified.fingerprints)
    static = audit_h1_comparability(
        h1_run_dir=h1_run,
        current_fingerprint_digest=current_digest,
        repository_root=repository_root,
    )
    if not all(static["static_checks"].values()):
        raise PermissionError("Frozen H1 static contract does not match the Phase 1B comparator")
    reference = _load_h1_reference_epoch1(h1_run)
    config = phase1b_training_config()
    train_dataset = _dataset(verified.specs["train"], config)
    validation_dataset = _dataset(verified.specs["validation"], config)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=make_generator(PHASE1B_SEED),
        pin_memory=True,
    )
    validation_loader: DataLoader[dict[str, Any]] = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    if not torch.cuda.is_available():
        result = {
            "schema_version": 1,
            "gate_id": "H1_epoch1_deterministic_replay",
            "status": "cuda_unavailable",
            "comparison": "INCOMPLETE",
        }
        write_json(output / "replay_result.json", result)
        return result
    ledger = _phase1b_ledger(
        ledger_path,
        repository_root=repository_root,
        config=protocol,
    )
    allocation = ledger.begin(
        "H1_epoch1_deterministic_replay",
        requested_limit_seconds=float(protocol.resources.replay_max_seconds),
    )
    evaluation_reserve = min(60.0, allocation * 0.1)
    training_allocation = allocation - evaluation_reserve
    started = time.perf_counter()
    try:
        from geoequi_ld.models.hrnet import HRNetW32SharedHeatmap

        seed_everything(PHASE1B_SEED, deterministic=True)
        device = torch.device("cuda")
        model = HRNetW32SharedHeatmap(align_corners=True).to(device)
        optimizer = build_phase1b_adam(model.parameters(), protocol)
        dsnt = DSNT(temperature=0.05, align_corners=True).to(device)
        bounded = train_for_steps_bounded(
            model,
            train_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
            max_steps=TRAIN_SAMPLE_COUNT,
            max_runtime_seconds=training_allocation,
        )
        if bounded.steps_completed == TRAIN_SAMPLE_COUNT:
            train_metrics = {
                "total_loss": sum(row["total_loss"] for row in bounded.history)
                / TRAIN_SAMPLE_COUNT,
                "heatmap_mse": sum(row["heatmap_mse"] for row in bounded.history)
                / TRAIN_SAMPLE_COUNT,
                "coordinate_smooth_l1": sum(
                    row["coordinate_smooth_l1"] for row in bounded.history
                )
                / TRAIN_SAMPLE_COUNT,
                "distribution_js": sum(row["distribution_js"] for row in bounded.history)
                / TRAIN_SAMPLE_COUNT,
                "batches": float(TRAIN_SAMPLE_COUNT),
            }
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                dsnt=dsnt,
                device=device,
                config=config,
            )
            matches, comparisons = _compare_replay_metrics(
                reference,
                train_metrics,
                validation_metrics,
            )
            comparison = "PASS" if matches else "FAIL"
        else:
            train_metrics = {}
            validation_metrics = {}
            comparisons = {}
            comparison = "INCOMPLETE"
        _save_model_only(output / "epoch_001_model_only.pt", model=model, epoch=1)
        write_history_csv(output / "replay_train_steps.csv", bounded.history)
        result = {
            "schema_version": 1,
            "gate_id": "H1_epoch1_deterministic_replay",
            "status": bounded.status,
            "comparison": comparison,
            "steps_completed": bounded.steps_completed,
            "steps_requested": TRAIN_SAMPLE_COUNT,
            "atol": REPLAY_COMPARISON_ATOL,
            "rtol": REPLAY_COMPARISON_RTOL,
            "reference_fingerprint_digest": current_digest,
            "static_checks": static["static_checks"],
            "strictly_comparable_for_phase1b": bool(
                comparison == "PASS" and all(static["static_checks"].values())
            ),
            "comparison_classification": (
                "frozen_shared_comparator"
                if comparison == "PASS" and all(static["static_checks"].values())
                else "historical_reference_only"
            ),
            "fixed_train_order": "DataLoader shuffle generator seed=42",
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "metric_comparisons": comparisons,
            "environment": _runtime_environment(repository_root),
            "runtime_source_binding": _runtime_source_binding(repository_root),
            "protocol_config_binding": _file_binding(
                phase1b_config,
                logical_name="configs/phase1b_decoder_control.yaml",
            ),
            "h1_reference_binding": _h1_reference_binding(h1_run),
            "historical_environment_record_available": False,
            "environment_evidence_scope": "same_current_environment_exact_epoch1_replay",
            "total_elapsed_seconds": time.perf_counter() - started,
            "total_allocated_seconds": allocation,
        }
        ledger_status = "completed" if bounded.status == "completed" else "budget_exhausted"
        ledger_snapshot = _finish_ledger(
            ledger,
            "H1_epoch1_deterministic_replay",
            started=started,
            status=ledger_status,
            details={"comparison": comparison, "steps_completed": bounded.steps_completed},
        )
        result["gpu_ledger_binding"] = _ledger_run_binding(
            ledger_snapshot,
            "H1_epoch1_deterministic_replay",
        )
        if result["gpu_ledger_binding"]["entry"]["status"] != "completed":
            result["status"] = "budget_exhausted"
            result["comparison"] = "INCOMPLETE"
            result["strictly_comparable_for_phase1b"] = False
            result["comparison_classification"] = "historical_reference_only"
        write_json(output / "replay_result.json", result)
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "gate_id": "H1_epoch1_deterministic_replay",
            "status": "oom",
            "comparison": "INCOMPLETE",
            "adaptation_attempted": False,
        }
        write_json(output / "replay_result.json", result)
        _finish_ledger(
            ledger,
            "H1_epoch1_deterministic_replay",
            started=started,
            status="oom",
        )
        return result
    except Exception:
        _finish_ledger(
            ledger,
            "H1_epoch1_deterministic_replay",
            started=started,
            status="failed",
        )
        raise


def _capture_resume_state(train_generator: torch.Generator) -> dict[str, Any]:
    """Capture every RNG stream needed to resume a full best/last checkpoint."""

    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "train_loader_generator_state": train_generator.get_state(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _fit_phase1b_supervised(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    dsnt: DSNT,
    device: torch.device,
    config: Any,
    output_dir: Path,
    checkpoint_config: Mapping[str, Any],
    max_runtime_seconds: float,
    milestone_epochs: Sequence[int],
    train_generator: torch.Generator,
) -> dict[str, Any]:
    """Train at epoch boundaries while retaining preregistered model-only milestones."""

    write_json(output_dir / "config.json", checkpoint_config)
    history: list[dict[str, Any]] = []
    best_selection = (math.inf, math.inf, math.inf)
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    started = time.perf_counter()
    budget = WallClockBudget.start(max_runtime_seconds)
    milestones = set(int(epoch) for epoch in milestone_epochs)
    for epoch in range(1, config.epochs + 1):
        if history:
            recent = history[-3:]
            estimate = sum(float(row["epoch_time_sec"]) for row in recent) / len(recent)
            if not budget.can_start(estimate):
                break
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        validation_seconds = time.perf_counter() - validation_started
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_time_sec": train_seconds,
            "validation_time_sec": validation_seconds,
            "epoch_time_sec": time.perf_counter() - epoch_started,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch in milestones:
            _save_model_only(
                output_dir / "milestones" / f"epoch_{epoch:03d}_model_only.pt",
                model=model,
                epoch=epoch,
            )
        resume_state = _capture_resume_state(train_generator)
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=checkpoint_config,
            seed=config.seed,
            metrics=validation_metrics,
            extra={
                "runtime_elapsed_sec": time.perf_counter() - started,
                "runtime_limit_sec": max_runtime_seconds,
                "resume_state": resume_state,
            },
        )
        selection = (
            float(validation_metrics["aop_mae_deg"]),
            float(validation_metrics["MRE_ALL"]),
            float(epoch),
        )
        if not all(math.isfinite(value) for value in selection[:2]):
            raise FloatingPointError("Validation selection metrics became NaN or Inf")
        if selection < best_selection:
            best_selection = selection
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=checkpoint_config,
                seed=config.seed,
                metrics=validation_metrics,
                extra={
                    "runtime_elapsed_sec": time.perf_counter() - started,
                    "runtime_limit_sec": max_runtime_seconds,
                    "resume_state": resume_state,
                },
            )
        write_history_csv(output_dir / "train_log.csv", history)
    if not history:
        raise RuntimeError("Phase 1B formal budget did not permit one complete epoch")
    return {
        "status": "completed" if len(history) == config.epochs else "budget_exhausted",
        "epochs_completed": len(history),
        "epochs_requested": config.epochs,
        "runtime_elapsed_sec": time.perf_counter() - started,
        "runtime_limit_sec": max_runtime_seconds,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "last_validation_metrics": {
            name.removeprefix("val_"): value
            for name, value in history[-1].items()
            if name.startswith("val_")
        },
    }


def _metric_close(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> bool:
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual) == str(expected)
    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except (TypeError, ValueError):
        return actual == expected
    return math.isclose(actual_number, expected_number, abs_tol=atol, rel_tol=rtol)


def _evaluate_key_checkpoints(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    output: Path,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    dsnt: DSNT,
    device: torch.device,
    config: Any,
) -> dict[str, Any]:
    with (output / "train_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError("Phase 1B train_log.csv is empty")
    best_row = min(
        history,
        key=lambda row: (
            float(row["val_aop_mae_deg"]),
            float(row["val_MRE_ALL"]),
            int(row["epoch"]),
        ),
    )
    expected_epochs = {"best": int(best_row["epoch"]), "last": int(history[-1]["epoch"])}
    rows_by_epoch = {int(row["epoch"]): row for row in history}
    results: dict[str, Any] = {}
    for name in ("best", "last"):
        payload = restore_checkpoint(
            output / f"{name}.pt",
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        expected_epoch = expected_epochs[name]
        if int(payload["epoch"]) != expected_epoch:
            raise RuntimeError(
                f"{name} checkpoint epoch {payload['epoch']} != train-log epoch {expected_epoch}"
            )
        extra = payload.get("extra")
        resume_state = extra.get("resume_state") if isinstance(extra, Mapping) else None
        required_resume = {
            "python_random_state",
            "numpy_random_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
            "train_loader_generator_state",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
        }
        if not isinstance(resume_state, Mapping) or set(resume_state) != required_resume:
            raise RuntimeError(f"{name} checkpoint lacks complete RNG/DataLoader resume state")
        before = _tensor_state_digest(model.state_dict())
        train_metrics = evaluate_model(
            model,
            train_loader,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        after = _tensor_state_digest(model.state_dict())
        if before != after:
            raise RuntimeError(
                "Eval-mode key-checkpoint evaluation modified persistent model state"
            )
        checkpoint_metrics = payload["metrics"]
        row = rows_by_epoch[expected_epoch]
        inconsistent: list[str] = []
        for metric_name, checkpoint_value in checkpoint_metrics.items():
            row_name = f"val_{metric_name}"
            if row_name not in row or not _metric_close(checkpoint_value, row[row_name]):
                inconsistent.append(f"checkpoint_vs_log:{metric_name}")
            if metric_name not in validation_metrics or not _metric_close(
                validation_metrics[metric_name],
                checkpoint_value,
            ):
                inconsistent.append(f"recomputed_vs_checkpoint:{metric_name}")
        if inconsistent:
            raise RuntimeError(
                f"{name} checkpoint validation provenance mismatch: {inconsistent}"
            )
        results[name] = {
            "epoch": expected_epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "evaluation_state_unchanged": True,
            "checkpoint_epoch_matches_selection": True,
            "checkpoint_metrics_match_train_log": True,
            "recomputed_validation_matches_checkpoint": True,
            "full_resume_state_present": True,
        }
    return results


def run_phase1b_split_formal(
    *,
    local_config: str | Path,
    phase1b_config: str | Path,
    tiny_artifact: str | Path,
    tiny_review_artifact: str | Path,
    h1_comparability_artifact: str | Path,
    h1_run_dir: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Run H2_split_B2_seed42_20e from the common untrained shared state."""

    protocol = load_phase1b_decoder_config(phase1b_config)
    verified = load_verified_phase1b_data(
        local_config,
        repository_root=repository_root,
    )
    current_fingerprint = fingerprint_digest(verified.fingerprints)
    tiny_evidence = require_passed_phase1b_tiny_artifact(
        tiny_artifact,
        tiny_review_artifact,
        data_fingerprint_digest=current_fingerprint,
        phase1b_config=phase1b_config,
        ledger_path=ledger_path,
        repository_root=repository_root,
    )
    comparability = validate_h1_replay_artifact(
        replay_artifact=h1_comparability_artifact,
        h1_run_dir=h1_run_dir,
        current_fingerprint_digest=current_fingerprint,
        repository_root=repository_root,
        ledger_path=ledger_path,
        phase1b_config=phase1b_config,
    )
    config = replace(phase1b_training_config(), epochs=20)
    train_dataset = _dataset(verified.specs["train"], config)
    validation_dataset = _dataset(verified.specs["validation"], config)
    if (
        len(train_dataset) != TRAIN_SAMPLE_COUNT
        or len(validation_dataset) != VALIDATION_SAMPLE_COUNT
    ):
        raise PermissionError("Phase 1B formal split run requires the verified 300/100 split")
    train_generator = make_generator(PHASE1B_SEED)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
        pin_memory=True,
    )
    train_evaluation_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    validation_loader: DataLoader[dict[str, Any]] = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    shared, model, initialization = _build_paired_initialization()
    tiny_initialization = tiny_evidence["tiny_result"].get("initialization")
    initialization_keys = (
        "seed",
        "method",
        "shared_state_sha256",
        "split_state_sha256",
        "shared_trainable_parameters",
        "split_trainable_parameters",
        "additional_trainable_parameters",
        "output_equivalent",
    )
    if not isinstance(tiny_initialization, Mapping) or any(
        tiny_initialization.get(name) != initialization.get(name)
        for name in initialization_keys
    ):
        raise PermissionError(
            "Phase 1B tiny and formal runs do not share the bound seed-42 initialization"
        )
    output = require_phase1b_fresh_output(output_dir, repository_root=repository_root)
    _save_model_only(output / "initial_shared_model_only.pt", model=shared, epoch=0)
    del shared
    if not torch.cuda.is_available():
        result = {
            "schema_version": 1,
            "experiment_name": protocol.experiment_name,
            "status": "cuda_unavailable",
            "partial": True,
        }
        write_json(output / "formal_result.json", result)
        return result
    ledger = _phase1b_ledger(
        ledger_path,
        repository_root=repository_root,
        config=protocol,
    )
    allocation = ledger.begin(
        protocol.experiment_name,
        requested_limit_seconds=float(protocol.resources.formal_max_seconds),
        reserve_after_seconds=float(protocol.resources.closing_reserve_seconds),
    )
    post_evaluation_reserve = min(600.0, allocation * 0.12)
    training_allocation = allocation - post_evaluation_reserve
    if training_allocation <= 0:
        raise RuntimeError("No formal training budget remains after key-checkpoint reserve")
    started = time.perf_counter()
    try:
        device = torch.device("cuda")
        model = model.to(device)
        optimizer = build_phase1b_adam(model.parameters(), protocol)
        dsnt = DSNT(temperature=0.05, align_corners=True).to(device)
        checkpoint_config = {
            "phase": "phase1b",
            "experiment_name": protocol.experiment_name,
            "testing_frozen": True,
            "training": config.to_dict(),
            "model": protocol.model.to_dict(),
            "optimizer": protocol.optimizer.to_dict(),
            "data": {
                "train_count": TRAIN_SAMPLE_COUNT,
                "validation_count": VALIDATION_SAMPLE_COUNT,
                "fingerprint_digest": current_fingerprint,
                "train_shuffle_generator_seed": PHASE1B_SEED,
                "paths_embedded": False,
            },
            "runtime": {
                "allocated_seconds": allocation,
                "training_allocated_seconds": training_allocation,
                "post_evaluation_reserve_seconds": post_evaluation_reserve,
                "total_phase_gpu_cap_seconds": protocol.resources.total_gpu_max_seconds,
            },
            "environment": _runtime_environment(repository_root),
            "runtime_source_binding": _runtime_source_binding(repository_root),
            "protocol_config_binding": _file_binding(
                phase1b_config,
                logical_name="configs/phase1b_decoder_control.yaml",
            ),
            "common_initialization": initialization,
            "h1_comparability": comparability,
            "tiny_gate_evidence": {
                "tiny_result_binding": tiny_evidence["human_review"][
                    "tiny_result_binding"
                ],
                "overlay_bindings": tiny_evidence["human_review"]["overlay_bindings"],
                "human_review_decision": "PASS",
            },
        }
        summary = _fit_phase1b_supervised(
            model,
            train_loader,
            validation_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
            output_dir=output,
            checkpoint_config=checkpoint_config,
            max_runtime_seconds=training_allocation,
            milestone_epochs=protocol.resources.milestone_epochs,
            train_generator=train_generator,
        )
        key_metrics = _evaluate_key_checkpoints(
            model,
            optimizer,
            output=output,
            train_loader=train_evaluation_loader,
            validation_loader=validation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        write_json(output / "key_checkpoint_metrics.json", key_metrics)
        total_elapsed = time.perf_counter() - started
        training_subbudget_exhausted = summary["status"] != "completed"
        result = {
            "schema_version": 1,
            "experiment_name": protocol.experiment_name,
            "status": summary["status"],
            "epochs_completed": summary["epochs_completed"],
            "epochs_requested": 20,
            "partial": training_subbudget_exhausted,
            "training_subbudget_exhausted": training_subbudget_exhausted,
            "selection_split": "validation",
            "selection_order": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
            "runtime_elapsed_sec": total_elapsed,
            "runtime_allocated_sec": allocation,
            "within_runtime_allocation": total_elapsed <= allocation,
            "formal_allocation_exceeded": total_elapsed > allocation,
            "aggregate_gpu_cap_exceeded": False,
            "best_epoch": summary["best_epoch"],
            "best_validation_metrics": summary["best_validation_metrics"],
            "last_validation_metrics": summary["last_validation_metrics"],
            "key_checkpoint_metrics": key_metrics,
            "initialization": initialization,
            "shared_reference_classification": (
                comparability["comparison_classification"]
            ),
        }
        if not result["within_runtime_allocation"]:
            result["status"] = "budget_exhausted"
            result["partial"] = True
        ledger_status = "completed" if result["status"] == "completed" else "budget_exhausted"
        ledger_snapshot = _finish_ledger(
            ledger,
            protocol.experiment_name,
            started=started,
            status=ledger_status,
            details={"epochs_completed": summary["epochs_completed"]},
        )
        result["gpu_ledger_binding"] = _ledger_run_binding(
            ledger_snapshot,
            protocol.experiment_name,
        )
        result.update(
            _formal_runtime_allocation_outcome(
                elapsed_seconds=total_elapsed,
                allocated_seconds=allocation,
                ledger_binding=result["gpu_ledger_binding"],
            )
        )
        if result["gpu_ledger_binding"]["entry"]["status"] != "completed":
            result["status"] = "budget_exhausted"
            result["partial"] = True
        if not result["within_runtime_allocation"]:
            result["status"] = "budget_exhausted"
            result["partial"] = True
        write_json(output / "formal_result.json", result)
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "experiment_name": protocol.experiment_name,
            "status": "oom",
            "partial": True,
            "adaptation_attempted": False,
        }
        write_json(output / "formal_result.json", result)
        _finish_ledger(ledger, protocol.experiment_name, started=started, status="oom")
        return result
    except Exception:
        _finish_ledger(ledger, protocol.experiment_name, started=started, status="failed")
        raise
