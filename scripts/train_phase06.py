#!/usr/bin/env python
"""Run one locked Phase 0.6 long-budget supervised fidelity check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.optim import Adam  # noqa: E402
from torch.utils.data import DataLoader, RandomSampler, Sampler  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase06_long_budget.yaml"
CANONICAL_LOCAL_CONFIG = REPOSITORY_ROOT / "configs" / "phase05_local.yaml"
PHASE05_IDENTITY_SNAPSHOT = REPOSITORY_ROOT / "runs" / "phase05" / "B0" / "seed_42" / "config.json"
PHASE05_FREEZE_RECORD = REPOSITORY_ROOT / "reports" / "phase06" / "PHASE05_FREEZE.json"
PHASE06_RUN_ROOT = REPOSITORY_ROOT / "runs" / "phase06"
PHASE06_IDENTITY_MANIFEST = PHASE06_RUN_ROOT / "identity.json"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.data.access_policy import (  # noqa: E402
    fingerprint_labeled_split,
    load_phase05_local_splits,
    verify_fingerprint,
)
from geoequi_ld.data.dataset import IUGCLabeledDataset  # noqa: E402
from geoequi_ld.models.dsnt import DSNT  # noqa: E402
from geoequi_ld.models.unet import HeatmapUNet, count_trainable_parameters  # noqa: E402
from geoequi_ld.training.ablation import (  # noqa: E402
    VARIANTS,
    apply_variant,
    assert_variant_weights,
    normalize_variant,
)
from geoequi_ld.training.checkpoints import restore_checkpoint  # noqa: E402
from geoequi_ld.training.config import SupervisedTrainingConfig  # noqa: E402
from geoequi_ld.training.engine import evaluate_model, fit_supervised, write_json  # noqa: E402
from geoequi_ld.training.runtime import (  # noqa: E402
    make_generator,
    resolve_device,
    seed_data_loader_worker,
    seed_everything,
)
from geoequi_ld.utils.hashing import sha256_file  # noqa: E402

PHASE05_FROZEN_COMMIT = "e351f3be10d5309bafe1ab9e2a1c69cc6744118c"
PHASE05_FROZEN_TAG = "phase05-v0.1.0"
PHASE05_FROZEN_SCOPE = (
    "reports/phase05",
    "configs/phase05_ablation.yaml",
    "configs/phase05_local.example.yaml",
    "configs/phase0_frozen_20e.yaml",
    "scripts/train_phase05.py",
    "scripts/select_phase05.py",
    "scripts/summarize_phase05.py",
    "src/geoequi_ld",
)
EXPECTED_MODEL_PARAMETERS = 484_171
MILESTONES = (20, 50, 100, 150, 200)
VALIDATION_METRICS = ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
EXPECTED_TRAINING: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "deterministic": True,
    "input_size_hw": [512, 512],
    "heatmap_size_hw": [256, 256],
    "sigma_heatmap_px": 4.0,
    "align_corners": True,
    "dsnt_temperature": 0.05,
    "keypoint_order": ["PS1", "PS2", "FH1"],
    "aop_vertex_index": 0,
    "aop_pubic_axis_other_index": 1,
    "aop_fetal_head_index": 2,
    "base_channels": 8,
    "batch_size": 1,
    "epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "heatmap_loss_weight": 1.0,
    "coordinate_loss_weight": 10.0,
    "distribution_loss_weight": 1.0,
    "max_grad_norm": 5.0,
    "num_workers": 0,
    "checkpoint_metric": "aop_mae_deg",
}
EXPECTED_OPTIMIZER: dict[str, Any] = {
    "class": "Adam",
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "amsgrad": False,
}
EXPECTED_EXECUTION: dict[str, Any] = {
    "variants": ["B0", "B1", "B2"],
    "seed": 42,
    "output_root": "runs/phase06",
    "local_config": "configs/phase05_local.yaml",
    "phase05_identity_snapshot": "runs/phase05/B0/seed_42/config.json",
    "identity_manifest": "runs/phase06/identity.json",
    "require_clean_worktree": True,
    "require_identical_initialization": True,
    "record_epoch_filename_order_sha256": True,
    "run_all_epochs": True,
    "compare_total_loss_across_variants": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pre-registered 200-epoch Phase 0.6 fidelity check. "
            "Only train and validation are accepted; testing is forbidden."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _load_protocol(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Phase 0.6 protocol must be a mapping")
    project = loaded.get("project")
    selection = loaded.get("selection")
    data_contract = loaded.get("data_contract")
    if not all(isinstance(value, dict) for value in (project, selection, data_contract)):
        raise ValueError("Protocol requires project, selection, and data_contract mappings")
    assert isinstance(project, dict)
    assert isinstance(selection, dict)
    assert isinstance(data_contract, dict)
    if project.get("phase") != "phase0.6-long-budget-fidelity":
        raise ValueError("Protocol phase must be phase0.6-long-budget-fidelity")
    if project.get("parent_phase05_commit") != PHASE05_FROZEN_COMMIT:
        raise ValueError("Protocol must reference the exact frozen Phase 0.5 commit")
    if project.get("parent_phase05_tag") != PHASE05_FROZEN_TAG:
        raise ValueError("Protocol must reference the exact frozen Phase 0.5 tag")
    if project.get("testing_frozen") is not True:
        raise PermissionError("Phase 0.6 requires testing_frozen=true")
    if project.get("frozen_scope") != list(PHASE05_FROZEN_SCOPE):
        raise ValueError("Protocol must freeze the exact registered Phase 0.5 scope")
    if selection != {
        "split": "validation",
        "common_decoder": "dsnt",
        "checkpoint_selection": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
        "milestones": list(MILESTONES),
    }:
        raise ValueError("Unexpected Phase 0.6 validation-selection protocol")
    if data_contract.get("allowed_splits") != ["train", "validation"]:
        raise PermissionError("Phase 0.6 permits only train and validation")
    if set(data_contract.get("forbidden_splits", [])) != {"test", "testing"}:
        raise PermissionError("Protocol must explicitly freeze test/testing")
    expected_data = {
        "train": {
            "sample_count": 300,
            "fingerprint_required": True,
            "source_columns": {"PS1": "PS1", "PS2": "PS2", "FH1": "FH1"},
        },
        "validation": {
            "sample_count": 100,
            "fingerprint_required": True,
            "source_columns": {
                "PS1": "PS1",
                "PS2": "PS2",
                "FH1": "AOP Tangency",
            },
        },
    }
    for role, expected in expected_data.items():
        if data_contract.get(role) != expected:
            raise ValueError(f"Unexpected Phase 0.6 {role} data contract")
    if loaded.get("training") != EXPECTED_TRAINING:
        raise ValueError("Phase 0.6 training settings do not match the long-budget lock")
    if loaded.get("optimizer") != EXPECTED_OPTIMIZER:
        raise ValueError("Phase 0.6 optimizer settings do not match the Adam lock")
    if loaded.get("model") != {
        "class": "HeatmapUNet",
        "trainable_parameters": EXPECTED_MODEL_PARAMETERS,
    }:
        raise ValueError("Phase 0.6 model settings do not match the HeatmapUNet lock")
    if loaded.get("execution") != EXPECTED_EXECUTION:
        raise ValueError("Phase 0.6 execution settings do not match the fidelity lock")
    expected_variants = {
        "B0": {
            "description": "heatmap MSE",
            "weights": [1.0, 0.0, 0.0],
            "validation_decoders": ["dsnt"],
        },
        "B1": {
            "description": "heatmap MSE + coordinate SmoothL1",
            "weights": [1.0, 10.0, 0.0],
            "validation_decoders": ["dsnt"],
        },
        "B2": {
            "description": "heatmap MSE + coordinate SmoothL1 + distribution JS",
            "weights": [1.0, 10.0, 1.0],
            "validation_decoders": ["dsnt"],
        },
    }
    if loaded.get("variants") != expected_variants:
        raise ValueError("Phase 0.6 requires the exact B0/B1/B2 objective lock")
    return loaded


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _git_state() -> tuple[str, bool]:
    commit = _git_output("rev-parse", "HEAD")
    dirty = bool(_git_output("status", "--porcelain"))
    return commit, dirty


def _validate_phase05_freeze() -> None:
    if _git_output("cat-file", "-t", PHASE05_FROZEN_TAG) != "tag":
        raise RuntimeError("Phase 0.5 freeze must be an annotated tag object")
    if _git_output("rev-parse", f"{PHASE05_FROZEN_TAG}^{{}}") != PHASE05_FROZEN_COMMIT:
        raise RuntimeError("The annotated Phase 0.5 tag does not peel to the frozen commit")
    difference = subprocess.run(
        ["git", "diff", "--quiet", PHASE05_FROZEN_TAG, "--", *PHASE05_FROZEN_SCOPE],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )
    if difference.returncode == 1:
        raise RuntimeError("A frozen Phase 0.5 implementation or report file has changed")
    if difference.returncode != 0:
        raise RuntimeError("Could not validate the frozen Phase 0.5 scope")
    untracked = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *PHASE05_FROZEN_SCOPE,
    )
    if untracked:
        raise RuntimeError("The frozen Phase 0.5 scope contains untracked or modified files")
    _validate_phase05_freeze_record()


def _validate_phase05_freeze_record_payload(
    value: Mapping[str, Any],
    *,
    tag_object: str,
    frozen_tree: str,
) -> None:
    expected = {
        "schema_version": 1,
        "phase": "phase0.5-supervised-ablation",
        "frozen_commit": PHASE05_FROZEN_COMMIT,
        "frozen_tree": frozen_tree,
        "annotated_tag": PHASE05_FROZEN_TAG,
        "annotated_tag_object": tag_object,
        "phase06_testing_policy": "frozen_no_read_no_evaluation",
        "note": (
            "Phase 0.6 is a train/validation-only 200-epoch supervised fidelity check; "
            "Phase 0.5 artifacts remain unchanged."
        ),
    }
    if dict(value) != expected:
        raise RuntimeError("PHASE05_FREEZE.json does not exactly match the Git freeze")


def _validate_phase05_freeze_record() -> None:
    _git_output("ls-files", "--error-unmatch", "reports/phase06/PHASE05_FREEZE.json")
    try:
        loaded = json.loads(PHASE05_FREEZE_RECORD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("PHASE05_FREEZE.json is missing or invalid") from error
    if not isinstance(loaded, dict):
        raise RuntimeError("PHASE05_FREEZE.json must contain a mapping")
    _validate_phase05_freeze_record_payload(
        loaded,
        tag_object=_git_output("rev-parse", PHASE05_FROZEN_TAG),
        frozen_tree=_git_output("rev-parse", f"{PHASE05_FROZEN_TAG}^{{tree}}"),
    )


def _validate_repository_state(protocol_path: Path, protocol: Mapping[str, Any]) -> str:
    if protocol_path.resolve(strict=True) != CANONICAL_PROTOCOL.resolve(strict=True):
        raise PermissionError("Phase 0.6 accepts only configs/phase06_long_budget.yaml")
    _git_output("ls-files", "--error-unmatch", "configs/phase06_long_budget.yaml")
    commit, dirty = _git_state()
    if dirty:
        raise RuntimeError("Formal Phase 0.6 runs require a clean Git worktree")
    _validate_phase05_freeze()
    project = protocol["project"]
    parent_commit = str(project["parent_phase05_commit"])
    parent_tag = str(project["parent_phase05_tag"])
    if _git_output("rev-parse", f"{parent_tag}^{{}}") != parent_commit:
        raise RuntimeError("The frozen Phase 0.5 tag does not match the protocol")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent_commit, commit],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return commit


def _validate_local_config_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    expected = CANONICAL_LOCAL_CONFIG.resolve(strict=False)
    if resolved != expected:
        raise PermissionError("Phase 0.6 requires exactly configs/phase05_local.yaml")
    return resolved


def _load_phase05_data_identity(path: Path) -> dict[str, dict[str, Any]]:
    if path.resolve(strict=True) != PHASE05_IDENTITY_SNAPSHOT.resolve(strict=True):
        raise PermissionError("Only the registered Phase 0.5 B0 configuration may be read")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("The frozen Phase 0.5 configuration must be a mapping")
    if (
        loaded.get("phase") != "phase0.5-supervised-ablation"
        or loaded.get("variant") != "B0"
        or loaded.get("testing_frozen") is not True
    ):
        raise ValueError("The frozen Phase 0.5 identity snapshot is not the registered B0 run")
    training = loaded.get("training")
    model = loaded.get("model")
    data = loaded.get("data")
    if not isinstance(training, dict) or not isinstance(model, dict) or not isinstance(data, dict):
        raise ValueError("The frozen Phase 0.5 identity snapshot is incomplete")
    if training.get("seed") != 42 or training.get("base_channels") != 8:
        raise ValueError("The frozen Phase 0.5 identity snapshot has drifted")
    if model.get("class") != "HeatmapUNet" or model.get("trainable_parameters") != 484_171:
        raise ValueError("The frozen Phase 0.5 model identity has drifted")
    if set(data) != {"train", "validation"}:
        raise PermissionError("The Phase 0.5 identity snapshot must contain only train/validation")
    result: dict[str, dict[str, Any]] = {}
    for role in ("train", "validation"):
        value = data.get(role)
        if not isinstance(value, dict) or set(value) != {
            "sample_count",
            "labels_sha256",
            "aggregate_sha256",
            "source_columns",
        }:
            raise ValueError(f"The frozen Phase 0.5 {role} identity is incomplete")
        for field in ("labels_sha256", "aggregate_sha256"):
            digest = value.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"The frozen Phase 0.5 {role} digest is invalid")
        result[role] = dict(value)
    return result


def _assert_phase05_data_identity(
    actual: Mapping[str, Mapping[str, Any]],
    frozen: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(actual) != {"train", "validation"} or set(frozen) != {"train", "validation"}:
        raise PermissionError("Data identity must contain exactly train and validation")
    for role in ("train", "validation"):
        if dict(actual[role]) != dict(frozen[role]):
            raise PermissionError(f"Phase 0.6 {role} data identity differs from frozen Phase 0.5")


def _validate_seed_and_output(*, variant: str, seed: int, output_dir: Path) -> Path:
    if seed != 42:
        raise ValueError("Phase 0.6 permits only seed 42")
    expected = (PHASE06_RUN_ROOT / variant / "seed_42").resolve(strict=False)
    if output_dir.resolve(strict=False) != expected:
        raise PermissionError(f"Output must be runs/phase06/{variant}/seed_42")
    return expected


def _ensure_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")


def _filename_order_sha256(filenames: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for filename in filenames:
        encoded = filename.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


class _TrainingOrderSampler(Sampler[int]):
    """Match RandomSampler while recording only a digest of each actual epoch order."""

    def __init__(
        self,
        filenames: Sequence[str],
        *,
        generator: torch.Generator,
        record_path: Path,
    ) -> None:
        self._filenames = tuple(filenames)
        self._sampler = RandomSampler(
            self._filenames,
            replacement=False,
            generator=generator,
        )
        self._record_path = record_path
        self.records: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._sampler)

    def __iter__(self) -> Iterator[int]:
        indices = list(self._sampler)
        ordered_names = [self._filenames[index] for index in indices]
        self.records.append(
            {
                "epoch": len(self.records) + 1,
                "sample_count": len(indices),
                "filename_order_sha256": _filename_order_sha256(ordered_names),
            }
        )
        payload = {
            "schema_version": 1,
            "contains_filenames": False,
            "records": self.records,
        }
        temporary = self._record_path.with_suffix(self._record_path.suffix + ".tmp")
        write_json(temporary, payload)
        os.replace(temporary, self._record_path)
        return iter(indices)


def _validation_loader(
    dataset: IUGCLabeledDataset,
    *,
    config: SupervisedTrainingConfig,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_loader_worker if config.num_workers else None,
        generator=make_generator(config.seed + 1),
        persistent_workers=config.num_workers > 0,
    )


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        payload["gpu"] = torch.cuda.get_device_name(device)
    return payload


def _identity_payload(
    *,
    commit: str,
    protocol_sha256: str,
    data: Mapping[str, Mapping[str, Any]],
    config: SupervisedTrainingConfig,
    model: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    common_training = config.to_dict()
    for field in (
        "heatmap_loss_weight",
        "coordinate_loss_weight",
        "distribution_loss_weight",
    ):
        common_training.pop(field)
    return {
        "schema_version": 1,
        "phase": "phase0.6-long-budget-fidelity",
        "git_commit": commit,
        "protocol_sha256": protocol_sha256,
        "data": {role: dict(value) for role, value in data.items()},
        "model": dict(model),
        "training_common": common_training,
        "optimizer": dict(optimizer),
        "environment": dict(environment),
    }


def _normalize_json(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if not isinstance(normalized, dict):  # pragma: no cover - defensive type lock
        raise TypeError("Identity payload must normalize to a mapping")
    return normalized


def _read_identity(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Phase 0.6 identity manifest is unreadable or incomplete") from error
    if not isinstance(loaded, dict):
        raise RuntimeError("Phase 0.6 identity manifest must be a mapping")
    return loaded


def _write_or_validate_identity(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_create: bool,
) -> str:
    expected = _normalize_json(payload)
    if path.exists():
        if not path.is_file():
            raise RuntimeError("Phase 0.6 identity manifest path is not a file")
        if _read_identity(path) != expected:
            raise RuntimeError("Phase 0.6 cross-run identity drift detected")
        return "validated"
    if not allow_create:
        raise PermissionError("B1/B2 require the write-once Phase 0.6 identity from B0")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(expected, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if _read_identity(path) != expected:
            raise RuntimeError("Phase 0.6 cross-run identity drift detected") from None
        return "validated"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "created"


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _milestone_metrics(rows: Sequence[Mapping[str, str]], best_epoch: int) -> dict[str, Any]:
    wanted = {*MILESTONES, best_epoch}
    result: dict[str, Any] = {}
    metric_names = ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL", "aop_mae_deg")
    for row in rows:
        epoch = int(row["epoch"])
        if epoch in wanted:
            key = "best" if epoch == best_epoch else str(epoch)
            result[key] = {
                "epoch": epoch,
                **{name: float(row[f"val_{name}"]) for name in metric_names},
            }
            if epoch == best_epoch and epoch in MILESTONES:
                result[str(epoch)] = dict(result[key])
    return result


def _finite_metric(value: Any, *, context: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{context} must be finite and non-negative")
    return converted


def _require_metric_close(actual: Any, expected: Any, *, context: str) -> None:
    actual_value = _finite_metric(actual, context=context)
    expected_value = _finite_metric(expected, context=context)
    if not math.isclose(actual_value, expected_value, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError(f"{context} differs between completed-run artifacts")


def _require_mre_reduction_close(actual: Any, expected: Any, *, context: str) -> None:
    actual_value = _finite_metric(actual, context=context)
    expected_value = _finite_metric(expected, context=context)
    if not math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{context} violates the three-keypoint MRE reduction identity")


def _validate_completed_run(
    rows: Sequence[Mapping[str, str]],
    *,
    summary: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    best_validation_metrics: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != 200:
        raise ValueError("Phase 0.6 train_log must contain exactly 200 rows")
    expected_epochs = list(range(1, 201))
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Phase 0.6 train_log has an invalid epoch column") from error
    if epochs != expected_epochs:
        raise ValueError("Phase 0.6 train_log epochs must be exactly 1..200")

    parsed: list[dict[str, float]] = []
    for epoch, row in zip(epochs, rows, strict=True):
        metrics = {
            name: _finite_metric(row.get(f"val_{name}"), context=f"epoch {epoch} {name}")
            for name in VALIDATION_METRICS
        }
        expected_mre = (
            metrics["MRE_PS1"] + metrics["MRE_PS2"] + metrics["MRE_FH1"]
        ) / 3.0
        _require_mre_reduction_close(
            metrics["MRE_ALL"],
            expected_mre,
            context=f"epoch {epoch} MRE_ALL mean identity",
        )
        parsed.append(metrics)

    recomputed_best_epoch = min(
        expected_epochs,
        key=lambda epoch: (
            parsed[epoch - 1]["aop_mae_deg"],
            parsed[epoch - 1]["MRE_ALL"],
            epoch,
        ),
    )
    if int(summary.get("best_epoch", -1)) != recomputed_best_epoch:
        raise ValueError("Summary best epoch differs from the recomputed checkpoint tuple")
    if summary.get("selection_split") != "validation":
        raise ValueError("Completed run was not selected on validation")
    if summary.get("checkpoint_metric") != "aop_mae_deg":
        raise ValueError("Completed run used an unexpected checkpoint metric")
    if summary.get("selection_tiebreak") != ["aop_mae_deg", "MRE_ALL", "earlier_epoch"]:
        raise ValueError("Completed run used an unexpected checkpoint tie-break")
    _require_metric_close(
        summary.get("best_value"),
        parsed[recomputed_best_epoch - 1]["aop_mae_deg"],
        context="summary best_value",
    )
    if int(checkpoint_payload.get("epoch", -1)) != recomputed_best_epoch:
        raise ValueError("Best checkpoint epoch differs from the recomputed checkpoint tuple")
    if checkpoint_payload.get("seed") != 42:
        raise ValueError("Best checkpoint seed is not the registered seed 42")
    if checkpoint_payload.get("config") != dict(run_config):
        raise ValueError("Best checkpoint configuration differs from the active run configuration")

    summary_best = summary.get("best_validation_metrics")
    checkpoint_best = checkpoint_payload.get("metrics")
    summary_last = summary.get("last_validation_metrics")
    if not all(
        isinstance(value, Mapping)
        for value in (summary_best, checkpoint_best, summary_last, best_validation_metrics)
    ):
        raise ValueError("Completed-run metric payloads are incomplete")
    assert isinstance(summary_best, Mapping)
    assert isinstance(checkpoint_best, Mapping)
    assert isinstance(summary_last, Mapping)
    best_row = parsed[recomputed_best_epoch - 1]
    last_row = parsed[-1]
    for name in VALIDATION_METRICS:
        for label, payload in (
            ("summary best", summary_best),
            ("checkpoint best", checkpoint_best),
            ("best checkpoint re-evaluation", best_validation_metrics),
        ):
            _require_metric_close(payload.get(name), best_row[name], context=f"{label} {name}")
        _require_metric_close(
            summary_last.get(name),
            last_row[name],
            context=f"summary last {name}",
        )

    milestones = _milestone_metrics(rows, recomputed_best_epoch)
    expected_milestones = {*(str(epoch) for epoch in MILESTONES), "best"}
    if set(milestones) != expected_milestones:
        raise ValueError("Completed run is missing a registered milestone or best snapshot")
    return {
        "validated_epochs": len(rows),
        "best_epoch": recomputed_best_epoch,
        "milestones": milestones,
    }


def _save_validation_curves(run_dir: Path) -> None:
    rows = _read_history(run_dir / "train_log.csv")
    epochs = [int(row["epoch"]) for row in rows]
    curves_dir = run_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    figure, (mre_axis, aop_axis) = plt.subplots(2, 1, figsize=(8, 7), dpi=150, sharex=True)
    for key in ("MRE_PS1", "MRE_PS2", "MRE_FH1", "MRE_ALL"):
        mre_axis.plot(epochs, [float(row[f"val_{key}"]) for row in rows], label=key)
    mre_axis.set_ylabel("validation MRE (px)")
    mre_axis.grid(alpha=0.25)
    mre_axis.legend(ncol=2)
    aop_axis.plot(
        epochs,
        [float(row["val_aop_mae_deg"]) for row in rows],
        color="#D55E00",
        label="AoP MAE",
    )
    aop_axis.set(xlabel="epoch", ylabel="validation AoP MAE (deg)")
    aop_axis.grid(alpha=0.25)
    aop_axis.legend()
    figure.tight_layout()
    figure.savefig(curves_dir / "validation_metrics.png")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variant_name = normalize_variant(args.variant)
    if args.protocol.resolve(strict=True) != CANONICAL_PROTOCOL.resolve(strict=True):
        raise PermissionError("Phase 0.6 accepts only the canonical repository protocol")
    protocol = _load_protocol(args.protocol)
    commit = _validate_repository_state(args.protocol, protocol)
    output_dir = _validate_seed_and_output(
        variant=variant_name,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if not args.verify_only:
        _ensure_empty_output(output_dir)

    _validate_local_config_path(args.local_config)
    frozen_phase05_data = _load_phase05_data_identity(PHASE05_IDENTITY_SNAPSHOT)
    local_splits = load_phase05_local_splits(args.local_config)
    data_contract = protocol["data_contract"]
    fingerprints: dict[str, dict[str, str | int]] = {}
    data_identity: dict[str, dict[str, Any]] = {}
    for role in ("train", "validation"):
        actual = fingerprint_labeled_split(local_splits[role])
        verify_fingerprint(actual, local_splits[role].expected_fingerprint, role=role)
        if data_contract[role].get("fingerprint_required") is not True:
            raise PermissionError(f"Public protocol must require a local fingerprint for {role}")
        if actual["sample_count"] != int(data_contract[role]["sample_count"]):
            raise PermissionError(f"Public sample-count contract failed for {role}")
        actual_columns = {
            "PS1": "PS1",
            "PS2": "PS2",
            "FH1": local_splits[role].fh1_column,
        }
        if actual_columns != data_contract[role].get("source_columns"):
            raise PermissionError(f"Source-column contract failed for {role}")
        fingerprints[role] = actual
        data_identity[role] = {**actual, "source_columns": actual_columns}
    _assert_phase05_data_identity(data_identity, frozen_phase05_data)

    base_config = SupervisedTrainingConfig.from_mapping(protocol["training"])
    config = apply_variant(replace(base_config, seed=args.seed), variant_name)
    assert_variant_weights(config, variant_name)
    declared_weights = tuple(
        float(value) for value in protocol["variants"][variant_name]["weights"]
    )
    actual_weights = (
        config.heatmap_loss_weight,
        config.coordinate_loss_weight,
        config.distribution_loss_weight,
    )
    if declared_weights != actual_weights:
        raise ValueError("Protocol variant weights disagree with the implementation lock")

    verified = {
        "status": "validated",
        "phase": "phase0.6-long-budget-fidelity",
        "variant": variant_name,
        "seed": config.seed,
        "epochs": config.epochs,
        "selection_split": "validation",
        "selection_decoder": "dsnt",
        "testing_frozen": True,
        "data": {
            role: {"sample_count": value["sample_count"], "fingerprint_verified": True}
            for role, value in fingerprints.items()
        },
        "git_commit": commit,
        "git_dirty": False,
    }
    if args.verify_only:
        print(json.dumps(verified, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    common_dataset = {
        "keypoint_order": config.keypoint_order,
        "input_size_hw": config.input_size_hw,
        "heatmap_size_hw": config.heatmap_size_hw,
        "sigma": config.sigma_heatmap_px,
        "align_corners": config.align_corners,
    }
    train_spec = local_splits["train"]
    validation_spec = local_splits["validation"]
    train_dataset = IUGCLabeledDataset(
        image_dir=train_spec.image_dir,
        labels_csv=train_spec.labels_csv,
        source_columns={"PS1": "PS1", "PS2": "PS2", "FH1": train_spec.fh1_column},
        **common_dataset,
    )
    validation_dataset = IUGCLabeledDataset(
        image_dir=validation_spec.image_dir,
        labels_csv=validation_spec.labels_csv,
        source_columns={
            "PS1": "PS1",
            "PS2": "PS2",
            "FH1": validation_spec.fh1_column,
        },
        **common_dataset,
    )
    train_generator = make_generator(config.seed)
    training_order_sampler = _TrainingOrderSampler(
        train_dataset.rows["Filename"].astype(str).tolist(),
        generator=train_generator,
        record_path=output_dir / "training_order.json",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=training_order_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_loader_worker if config.num_workers else None,
        generator=train_generator,
        persistent_workers=config.num_workers > 0,
    )
    validation_loader = _validation_loader(validation_dataset, config=config, device=device)

    model = HeatmapUNet(base_channels=config.base_channels).to(device)
    trainable_parameters = count_trainable_parameters(model)
    if trainable_parameters != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError(
            "HeatmapUNet parameter count drifted from the frozen 484171-parameter model"
        )
    initialization_sha256 = _state_dict_sha256(model)
    dsnt = DSNT(
        temperature=config.dsnt_temperature,
        align_corners=config.align_corners,
    ).to(device)
    optimizer_spec = protocol["optimizer"]
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=tuple(float(value) for value in optimizer_spec["betas"]),
        eps=float(optimizer_spec["eps"]),
        amsgrad=bool(optimizer_spec["amsgrad"]),
    )
    protocol_sha256 = sha256_file(args.protocol)
    environment = _environment(device)
    model_identity = {
        "class": "HeatmapUNet",
        "trainable_parameters": trainable_parameters,
        "initialization_sha256": initialization_sha256,
    }
    identity = _identity_payload(
        commit=commit,
        protocol_sha256=protocol_sha256,
        data=data_identity,
        config=config,
        model=model_identity,
        optimizer=optimizer_spec,
        environment=environment,
    )
    _write_or_validate_identity(
        PHASE06_IDENTITY_MANIFEST,
        identity,
        allow_create=variant_name == "B0",
    )
    run_config = {
        "schema_version": 1,
        "phase": "phase0.6-long-budget-fidelity",
        "variant": variant_name,
        "variant_description": VARIANTS[variant_name].description,
        "training": config.to_dict(),
        "optimizer": optimizer_spec,
        "selection": {
            "split": "validation",
            "common_decoder": "dsnt",
            "checkpoint_selection": protocol["selection"]["checkpoint_selection"],
            "milestones": list(MILESTONES),
        },
        "testing_frozen": True,
        "data": data_identity,
        "model": model_identity,
        "order_audit": {
            "algorithm": "RandomSampler-compatible randperm",
            "generator_seed": config.seed,
            "per_epoch_filename_order_sha256": True,
        },
        "provenance": {
            "protocol_sha256": protocol_sha256,
            "git_commit": commit,
            "git_dirty": False,
            "parent_phase05_commit": PHASE05_FROZEN_COMMIT,
            "parent_phase05_tag": PHASE05_FROZEN_TAG,
        },
    }
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(output_dir / "environment.json", environment)

    started = time.perf_counter()
    summary = fit_supervised(
        model,
        train_loader,
        validation_loader,
        optimizer,
        dsnt=dsnt,
        device=device,
        config=config,
        output_dir=output_dir,
        checkpoint_config=run_config,
    )
    training_runtime_sec = time.perf_counter() - started
    if len(training_order_sampler.records) != config.epochs:
        raise RuntimeError("Training-order ledger does not contain exactly 200 epochs")
    if any(record["sample_count"] != 300 for record in training_order_sampler.records):
        raise RuntimeError("Training-order ledger contains an incomplete epoch")
    peak_gpu_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )
    peak_gpu_reserved_mb = (
        torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
    )
    evaluation_started = time.perf_counter()
    checkpoint_payload = restore_checkpoint(
        summary["best_checkpoint"], model=model, map_location=device
    )
    best_validation_metrics = evaluate_model(
        model,
        validation_loader,
        dsnt=dsnt,
        device=device,
        config=config,
        decoder="dsnt",
    )
    evaluation_runtime_sec = time.perf_counter() - evaluation_started
    rows = _read_history(output_dir / "train_log.csv")
    integrity = _validate_completed_run(
        rows,
        summary=summary,
        checkpoint_payload=checkpoint_payload,
        best_validation_metrics=best_validation_metrics,
        run_config=run_config,
    )
    milestone_metrics = integrity["milestones"]
    _save_validation_curves(output_dir)
    best_checkpoint_sha256 = sha256_file(Path(summary["best_checkpoint"]))
    result = {
        "status": "completed",
        "phase": "phase0.6-long-budget-fidelity",
        "variant": variant_name,
        "seed": config.seed,
        "epochs_completed": len(rows),
        "selection_split": "validation",
        "selection_decoder": "dsnt",
        "testing_frozen": True,
        "best_epoch": summary["best_epoch"],
        "best_validation_metrics": {"dsnt": best_validation_metrics},
        "milestone_validation_metrics": milestone_metrics,
        "last_validation_metrics": summary["last_validation_metrics"],
        "order_audit": {
            "recorded_epochs": len(training_order_sampler.records),
            "samples_per_epoch": 300,
            "contains_filenames": False,
        },
        "resources": {
            "training_runtime_sec": training_runtime_sec,
            "evaluation_runtime_sec": evaluation_runtime_sec,
            "peak_gpu_allocated_mb": peak_gpu_allocated_mb,
            "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
        },
        "model": run_config["model"],
        "provenance": {
            **run_config["provenance"],
            "best_checkpoint_sha256": best_checkpoint_sha256,
        },
    }
    write_json(output_dir / "phase06_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
