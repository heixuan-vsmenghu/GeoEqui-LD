"""State-audited BatchNorm diagnostic for the frozen Phase 1A HRNet endpoints.

This module deliberately reuses the supervised evaluator and the verified
train/validation access policy.  It does not contain an alternative coordinate
decoder, metric implementation, or validation-time BatchNorm update path.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from geoequi_ld.data.dataset import IUGCLabeledDataset
from geoequi_ld.diagnostics.phase1a import assert_public_aggregate
from geoequi_ld.models.dsnt import DSNT
from geoequi_ld.training.budget import WallClockBudget, require_fresh_output_directory
from geoequi_ld.training.checkpoints import read_checkpoint
from geoequi_ld.training.config import SupervisedTrainingConfig
from geoequi_ld.training.engine import evaluate_model
from geoequi_ld.training.phase1a_runners import (
    TRAIN_SAMPLE_COUNT,
    VALIDATION_SAMPLE_COUNT,
    VerifiedData,
    load_verified_phase1a_data,
    phase1a_training_config,
)
from geoequi_ld.utils.hashing import sha256_file

PHASE1B_TOTAL_GPU_SECONDS: Final = 10_800.0
BN_DIAGNOSTIC_MAX_SECONDS: Final = 900.0
H1_EXPERIMENT_NAME: Final = "H1_shared_B2_seed42_20e"
H1_ENDPOINT_EPOCHS: Final = {"best": 3, "last": 20}
ALLOWED_SPLITS: Final = ("train", "validation")
BN_TYPES: Final = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)
PUBLIC_METRIC_KEYS: Final = (
    "MRE_PS1",
    "MRE_PS2",
    "MRE_FH1",
    "MRE_ALL",
    "n_samples",
    "n_valid_aop",
    "n_evaluable_aop",
    "aop_invalid_prediction_count",
    "aop_valid_ratio",
    "aop_invalid_prediction_ratio",
    "aop_mae_valid_deg",
    "aop_penalized_selection_score_deg",
)


class Phase1BBudgetExceeded(RuntimeError):
    """Raised when the fixed short diagnostic exhausts its wall-clock allocation."""


@dataclass(frozen=True)
class TensorRecord:
    """Exact digest and metadata for one state tensor."""

    shape: tuple[int, ...]
    dtype: str
    sha256: str
    requires_grad: bool | None = None


@dataclass(frozen=True)
class ModelStateSnapshot:
    """A compact exact snapshot of parameters and persistent buffers."""

    parameters: Mapping[str, TensorRecord]
    persistent_buffers: Mapping[str, TensorRecord]


def _contains_forbidden_component(path: str | Path) -> bool:
    return any(part.casefold() in {"test", "testing"} for part in Path(path).parts)


def require_fresh_phase1b_private_output(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Create a fresh ignored Phase 1B directory without touching older phases."""

    root = Path(repository_root).resolve(strict=True)
    destination = Path(path).resolve(strict=False)
    allowed_roots = (
        (root / "runs" / "phase1b").resolve(strict=False),
        (root / "artifacts" / "phase1b").resolve(strict=False),
    )
    if not any(
        destination.is_relative_to(allowed) and destination != allowed
        for allowed in allowed_roots
    ):
        raise PermissionError(
            "Private Phase 1B output must be below runs/phase1b or artifacts/phase1b"
        )
    if _contains_forbidden_component(destination):
        raise PermissionError("Phase 1B refuses a test/testing output path")
    protected = tuple(
        (root / relative).resolve(strict=False)
        for relative in (
            "runs/phase0",
            "runs/phase05",
            "runs/phase06",
            "runs/phase1a",
            "reports",
            "checkpoints",
        )
    )
    return require_fresh_output_directory(destination, protected_roots=protected)


def require_fresh_phase1b_public_file(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Validate a new aggregate-only file below ``reports/phase1b``."""

    root = Path(repository_root).resolve(strict=True)
    destination = Path(path).resolve(strict=False)
    public_root = (root / "reports" / "phase1b").resolve(strict=False)
    if not destination.is_relative_to(public_root) or destination == public_root:
        raise PermissionError("Public Phase 1B output must be a file below reports/phase1b")
    if _contains_forbidden_component(destination):
        raise PermissionError("Phase 1B refuses a test/testing public output path")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite public Phase 1B output: {destination}")
    return destination


def require_canonical_h1_checkpoint(
    path: str | Path,
    *,
    checkpoint_id: str,
    repository_root: str | Path,
) -> Path:
    """Accept only the reviewed Phase 1A H1 best/last checkpoint locations."""

    if checkpoint_id not in H1_ENDPOINT_EPOCHS:
        raise ValueError(f"Unsupported H1 endpoint: {checkpoint_id!r}")
    root = Path(repository_root).resolve(strict=True)
    expected = (
        root / "runs" / "phase1a" / H1_EXPERIMENT_NAME / f"{checkpoint_id}.pt"
    ).resolve(strict=True)
    actual = Path(path).resolve(strict=True)
    if actual != expected:
        raise PermissionError(f"{checkpoint_id} must use the canonical frozen H1 checkpoint")
    return actual


def require_canonical_local_config(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    root = Path(repository_root).resolve(strict=True)
    expected = (root / "configs" / "phase05_local.yaml").resolve(strict=True)
    actual = Path(path).resolve(strict=True)
    if actual != expected:
        raise PermissionError("Phase 1B must reuse configs/phase05_local.yaml")
    return actual


def freeze_checkpoint_copies(
    sources: Mapping[str, Path],
    *,
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Copy both H1 endpoints once and verify byte-identical immutable inputs.

    The copies are called frozen because the diagnostic never writes through
    them and checks their SHA-256 values again at the end.  File permissions are
    intentionally not changed, so a failed local run remains easy to clean up.
    """

    if set(sources) != set(H1_ENDPOINT_EPOCHS):
        raise ValueError("Exactly the H1 best and last checkpoints are required")
    destination_root = Path(output_dir) / "frozen_checkpoints"
    destination_root.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, Any]] = {}
    for checkpoint_id in H1_ENDPOINT_EPOCHS:
        source = Path(sources[checkpoint_id])
        source_hash = sha256_file(source)
        destination = destination_root / f"{checkpoint_id}.pt"
        shutil.copy2(source, destination)
        copy_hash = sha256_file(destination)
        if copy_hash != source_hash:
            raise OSError(f"Frozen checkpoint copy hash mismatch: {checkpoint_id}")
        records[checkpoint_id] = {
            "source_path": str(source),
            "copy_path": str(destination),
            "source_sha256_before": source_hash,
            "copy_sha256_before": copy_hash,
        }
    return records


def _tensor_record(tensor: Tensor, *, requires_grad: bool | None = None) -> TensorRecord:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
    return TensorRecord(
        shape=tuple(int(value) for value in cpu.shape),
        dtype=str(cpu.dtype),
        sha256=digest,
        requires_grad=requires_grad,
    )


def _named_persistent_buffers(model: nn.Module) -> dict[str, Tensor]:
    """Return only state-dict buffers, excluding explicitly non-persistent ones."""

    result: dict[str, Tensor] = {}
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        non_persistent = module._non_persistent_buffers_set
        for local_name, value in module._buffers.items():
            if value is None or local_name in non_persistent or id(value) in seen:
                continue
            seen.add(id(value))
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            result[full_name] = value
    return result


def capture_model_state(model: nn.Module) -> ModelStateSnapshot:
    """Hash every parameter and every persistent buffer exactly."""

    parameters = {
        name: _tensor_record(parameter, requires_grad=parameter.requires_grad)
        for name, parameter in model.named_parameters(remove_duplicate=True)
    }
    persistent_buffers = {
        name: _tensor_record(buffer)
        for name, buffer in _named_persistent_buffers(model).items()
    }
    return ModelStateSnapshot(parameters=parameters, persistent_buffers=persistent_buffers)


def _changed_names(
    before: Mapping[str, TensorRecord],
    after: Mapping[str, TensorRecord],
    *,
    kind: str,
) -> list[str]:
    if set(before) != set(after):
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        raise RuntimeError(f"{kind} state keys changed; missing={missing}, added={added}")
    return sorted(name for name in before if before[name] != after[name])


def audit_state_transition(
    before: ModelStateSnapshot,
    after: ModelStateSnapshot,
    *,
    allowed_buffer_mutations: Iterable[str] = (),
) -> dict[str, Any]:
    """Fail unless parameters stay exact and only declared buffers may change."""

    parameter_mutations = _changed_names(before.parameters, after.parameters, kind="parameter")
    if parameter_mutations:
        raise RuntimeError(f"Model parameters changed: {parameter_mutations[:8]}")
    buffer_mutations = _changed_names(
        before.persistent_buffers,
        after.persistent_buffers,
        kind="persistent buffer",
    )
    allowed = set(allowed_buffer_mutations)
    illegal = sorted(set(buffer_mutations) - allowed)
    if illegal:
        raise RuntimeError(f"Non-BatchNorm persistent buffers changed: {illegal[:8]}")
    return {
        "audited_parameter_count": len(before.parameters),
        "audited_persistent_buffer_count": len(before.persistent_buffers),
        "parameter_mutations": parameter_mutations,
        "persistent_buffer_mutations": buffer_mutations,
        "allowed_persistent_buffer_mutations": sorted(allowed),
        "all_parameters_unchanged": True,
        "only_allowed_persistent_buffers_changed": True,
    }


def batch_norm_running_buffer_names(model: nn.Module) -> tuple[str, ...]:
    """Return the exact persistent BN statistics allowed during re-estimation."""

    names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, BN_TYPES) or not module.track_running_stats:
            continue
        for local_name in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(module, local_name)
            if value is None:
                raise RuntimeError(f"Tracked BatchNorm has no {local_name}: {module_name}")
            names.append(f"{module_name}.{local_name}" if module_name else local_name)
    if not names:
        raise RuntimeError("The H1 model contains no tracked BatchNorm statistics")
    return tuple(names)


def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _assert_frozen_and_grad_free(model: nn.Module) -> None:
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    gradients = [name for name, value in model.named_parameters() if value.grad is not None]
    if trainable or gradients:
        raise RuntimeError(
            f"BN diagnostic must keep every parameter frozen and grad-free; "
            f"trainable={trainable[:8]}, gradients={gradients[:8]}"
        )


def _budgeted_loader(
    loader: Iterable[Mapping[str, Any]],
    budget: WallClockBudget | None,
) -> Iterable[Mapping[str, Any]]:
    for batch in loader:
        if budget is not None and budget.remaining_seconds() <= 0:
            raise Phase1BBudgetExceeded("Phase 1B BN diagnostic exhausted its allocation")
        yield batch


def _metrics_with_validity(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    evaluable = int(result["n_evaluable_aop"])
    valid = int(result["n_valid_aop"])
    invalid = int(result["aop_invalid_prediction_count"])
    result["aop_valid_ratio"] = valid / evaluable if evaluable else math.nan
    result["aop_invalid_prediction_ratio"] = invalid / evaluable if evaluable else math.nan
    penalized_score = float(result["aop_mae_deg"])
    # The canonical evaluator calls the checkpoint-selection score ``aop_mae_deg``.
    # Give the public Phase 1B diagnostic an unambiguous name because invalid AoP
    # predictions contribute a fixed 180-degree penalty to this value; it is not
    # the valid-prediction-only MAE reported above.
    result["aop_penalized_selection_score_deg"] = penalized_score
    result["selection_penalty_score_deg"] = penalized_score
    return result


def evaluate_model_without_state_change(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    *,
    dsnt: DSNT,
    device: torch.device,
    config: SupervisedTrainingConfig,
    budget: WallClockBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the canonical evaluator and prove eval changed no model state."""

    _assert_frozen_and_grad_free(model)
    model.eval()
    before = capture_model_state(model)
    metrics = evaluate_model(
        model,
        _budgeted_loader(data_loader, budget),
        dsnt=dsnt,
        device=device,
        config=config,
        decoder="dsnt",
    )
    after = capture_model_state(model)
    audit = audit_state_transition(before, after)
    _assert_frozen_and_grad_free(model)
    if any(module.training for module in model.modules()):
        raise RuntimeError("Canonical evaluation did not leave the whole model in eval mode")
    return _metrics_with_validity(metrics), audit


def reestimate_batch_norm_statistics(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    expected_samples: int,
    budget: WallClockBudget | None = None,
) -> dict[str, Any]:
    """Reset and cumulatively update BN statistics from training images once.

    The canonical dataset may materialize target fields, but this update loop
    consumes only ``image`` and ``filename`` and never supplies labels to the
    model.  The model as a whole remains in eval mode while only BatchNorm leaves
    are placed in train mode.  Original momentum values are restored after the
    image pass; the newly accumulated running statistics remain on this in-memory
    checkpoint copy.
    """

    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    freeze_all_parameters(model)
    _assert_frozen_and_grad_free(model)
    model.eval()
    batch_norms = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BN_TYPES)
    ]
    if not batch_norms:
        raise RuntimeError("Cannot re-estimate statistics without BatchNorm modules")
    allowed = batch_norm_running_buffer_names(model)
    before = capture_model_state(model)
    original_momenta = {name: module.momentum for name, module in batch_norms}
    order_digest = hashlib.sha256()
    samples_seen = 0
    try:
        with torch.inference_mode():
            for _, module in batch_norms:
                module.reset_running_stats()
                module.momentum = None
                module.train()
            non_bn_training = [
                name
                for name, module in model.named_modules()
                if not isinstance(module, BN_TYPES) and module.training
            ]
            if non_bn_training:
                raise RuntimeError(
                    f"Non-BatchNorm modules entered train mode: {non_bn_training[:8]}"
                )
            for raw_batch in _budgeted_loader(train_loader, budget):
                image = raw_batch.get("image")
                if (
                    not isinstance(image, Tensor)
                    or image.ndim != 4
                    or int(image.shape[0]) != 1
                ):
                    raise ValueError("BN re-estimation requires image batches of exactly one")
                filenames = raw_batch.get("filename")
                if not isinstance(filenames, list | tuple) or len(filenames) != 1:
                    raise ValueError("BN re-estimation requires one stable filename per batch")
                order_digest.update(str(filenames[0]).encode("utf-8"))
                order_digest.update(b"\0")
                model(image.to(device, non_blocking=device.type == "cuda"))
                samples_seen += 1
    finally:
        for name, module in batch_norms:
            module.momentum = original_momenta[name]
        model.eval()
    if samples_seen != expected_samples:
        raise PermissionError(
            f"BN re-estimation requires exactly {expected_samples} training images, "
            f"saw {samples_seen}"
        )
    after = capture_model_state(model)
    audit = audit_state_transition(before, after, allowed_buffer_mutations=allowed)
    if not audit["persistent_buffer_mutations"]:
        raise RuntimeError("BN re-estimation did not change any running statistics")
    _assert_frozen_and_grad_free(model)
    if any(module.training for module in model.modules()):
        raise RuntimeError("BN re-estimation did not restore the whole model to eval mode")
    counts = {
        name: int(module.num_batches_tracked.item())
        for name, module in batch_norms
        if module.num_batches_tracked is not None
    }
    if not counts:
        raise RuntimeError("No tracked BatchNorm counters were available after re-estimation")
    unexpected_counts = {
        name: count for name, count in counts.items() if count != expected_samples
    }
    if unexpected_counts:
        preview = dict(list(unexpected_counts.items())[:8])
        raise RuntimeError(
            "Every tracked BatchNorm leaf must execute exactly once per training image; "
            f"expected {expected_samples}, mismatches={preview}"
        )
    return {
        "method": "reset_running_stats_then_cumulative_average",
        "batch_size": 1,
        "shuffle": False,
        "random_augmentation": False,
        "source_split": "train",
        "labels_not_used_as_model_inputs_for_update": True,
        "validation_used_for_update": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "momentum_during_update": None,
        "original_momentum_restored": True,
        "samples_seen": samples_seen,
        "order_sha256": order_digest.hexdigest(),
        "batch_norm_module_count": len(batch_norms),
        "tracked_batch_norm_module_count": len(counts),
        "all_tracked_batch_norm_counts_equal_expected_samples": True,
        "num_batches_tracked_min": min(counts.values()),
        "num_batches_tracked_max": max(counts.values()),
        "state_transition": audit,
    }


def _dataset(
    verified: VerifiedData,
    role: str,
    config: SupervisedTrainingConfig,
) -> IUGCLabeledDataset:
    if role not in ALLOWED_SPLITS:
        raise PermissionError(f"Phase 1B refuses split {role!r}")
    spec = verified.specs[role]
    return IUGCLabeledDataset(
        image_dir=spec.image_dir,
        labels_csv=spec.labels_csv,
        source_columns={"PS1": "PS1", "PS2": "PS2", "FH1": spec.fh1_column},
        keypoint_order=config.keypoint_order,
        input_size_hw=config.input_size_hw,
        heatmap_size_hw=config.heatmap_size_hw,
        sigma=config.sigma_heatmap_px,
        align_corners=config.align_corners,
    )


def _validate_h1_payload(payload: Mapping[str, Any], *, checkpoint_id: str) -> None:
    required = {
        "format_version",
        "epoch",
        "seed",
        "config",
        "metrics",
        "model_state_dict",
        "optimizer_state_dict",
    }
    if not required.issubset(payload):
        raise ValueError(f"H1 {checkpoint_id} checkpoint is incomplete")
    if int(payload["format_version"]) != 1:
        raise ValueError(f"H1 {checkpoint_id} checkpoint format changed")
    if int(payload["epoch"]) != H1_ENDPOINT_EPOCHS[checkpoint_id] or int(payload["seed"]) != 42:
        raise ValueError(f"H1 {checkpoint_id} epoch/seed does not match the reviewed endpoint")
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise ValueError(f"H1 {checkpoint_id} config is not a mapping")
    if (
        config.get("phase") != "phase1a"
        or config.get("experiment_name") != H1_EXPERIMENT_NAME
        or config.get("testing_frozen") is not True
    ):
        raise PermissionError(f"H1 {checkpoint_id} provenance/testing contract mismatch")
    training = config.get("training")
    data = config.get("data")
    model = config.get("model")
    if not isinstance(training, Mapping) or not isinstance(data, Mapping) or not isinstance(
        model, Mapping
    ):
        raise ValueError(f"H1 {checkpoint_id} checkpoint lacks training/data/model provenance")
    expected_training = {
        "seed": 42,
        "batch_size": 1,
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "heatmap_loss_weight": 1.0,
        "coordinate_loss_weight": 10.0,
        "distribution_loss_weight": 1.0,
        "dsnt_temperature": 0.05,
        "align_corners": True,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError(f"H1 {checkpoint_id} training protocol drifted")
    if data.get("train_count") != 300 or data.get("validation_count") != 100:
        raise ValueError(f"H1 {checkpoint_id} data counts drifted")
    if model.get("class") != "HRNetW32SharedHeatmap" or model.get("pretrained") is not False:
        raise ValueError(f"H1 {checkpoint_id} model provenance drifted")


def _model_from_state(
    state: Mapping[str, Tensor],
    *,
    device: torch.device,
    model_factory: Callable[[], nn.Module],
) -> nn.Module:
    model = model_factory()
    model.load_state_dict(state, strict=True)
    model.to(device)
    freeze_all_parameters(model)
    model.eval()
    return model


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in PUBLIC_METRIC_KEYS}


def build_public_bn_aggregate(private_result: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the private state/hash record to publishable aggregate metrics."""

    endpoints = private_result.get("endpoints")
    if not isinstance(endpoints, Mapping) or set(endpoints) != set(H1_ENDPOINT_EPOCHS):
        raise ValueError("Private BN result must contain H1 best and last endpoints")
    public_endpoints: dict[str, Any] = {}
    for checkpoint_id in H1_ENDPOINT_EPOCHS:
        endpoint = endpoints[checkpoint_id]
        if not isinstance(endpoint, Mapping):
            raise ValueError(f"Endpoint result is invalid: {checkpoint_id}")
        original = endpoint["original_bn"]
        recalibrated = endpoint["train_images_bn_reestimated"]
        assert isinstance(original, Mapping)
        assert isinstance(recalibrated, Mapping)
        original_validation = original["validation_metrics"]
        recalibrated_validation = recalibrated["validation_metrics"]
        assert isinstance(original_validation, Mapping)
        assert isinstance(recalibrated_validation, Mapping)
        deltas = {
            key: float(recalibrated_validation[key]) - float(original_validation[key])
            for key in (
                "MRE_PS1",
                "MRE_PS2",
                "MRE_FH1",
                "MRE_ALL",
                "aop_mae_valid_deg",
                "aop_invalid_prediction_ratio",
                "aop_penalized_selection_score_deg",
            )
        }
        public_endpoints[checkpoint_id] = {
            "epoch": int(endpoint["epoch"]),
            "original_bn": {
                "train": _metric_subset(original["train_metrics"]),
                "validation": _metric_subset(original_validation),
            },
            "train_images_bn_reestimated": {
                "validation": _metric_subset(recalibrated_validation),
                "delta_vs_original_validation": deltas,
            },
            "integrity": {
                "original_train_eval_state_unchanged": True,
                "original_validation_eval_state_unchanged": True,
                "reestimated_validation_eval_state_unchanged": True,
                "all_parameters_unchanged": True,
                "only_bn_running_statistics_changed_during_reestimation": True,
                "source_checkpoint_unchanged": bool(endpoint["source_checkpoint_unchanged"]),
                "frozen_copy_unchanged": bool(endpoint["frozen_copy_unchanged"]),
            },
        }
    aggregate = {
        "schema_version": 1,
        "phase": "phase1b_bn_short_diagnostic",
        "experiment_name": H1_EXPERIMENT_NAME,
        "status": "completed",
        "testing_frozen": True,
        "selection_split": "validation",
        "bn_reestimation_used_for_model_selection": False,
        "bn_reestimation_protocol": {
            "source_split": "train",
            "sample_count": TRAIN_SAMPLE_COUNT,
            "batch_size": 1,
            "shuffle": False,
            "random_augmentation": False,
            "reset_running_stats": True,
            "momentum": None,
            "passes": 1,
            "labels_not_used_as_model_inputs_for_update": True,
        },
        "endpoints": public_endpoints,
        "claim_boundary": (
            "A one-pass fixed BN-statistics diagnostic; it does not replace the original "
            "checkpoint result or establish the sole cause of validation variation."
        ),
    }
    assert_public_aggregate(aggregate)
    return aggregate


def run_phase1b_bn_diagnostic(
    *,
    local_config: str | Path,
    best_checkpoint: str | Path,
    last_checkpoint: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
    device: torch.device,
    max_runtime_seconds: float,
    model_factory: Callable[[], nn.Module] | None = None,
) -> dict[str, Any]:
    """Evaluate original H1 endpoints and fixed train-image BN copies."""

    if max_runtime_seconds <= 0 or max_runtime_seconds > BN_DIAGNOSTIC_MAX_SECONDS:
        raise ValueError("BN diagnostic runtime must be in (0, 900] seconds")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Phase 1B but is unavailable")
    root = Path(repository_root).resolve(strict=True)
    canonical_config = require_canonical_local_config(local_config, repository_root=root)
    sources = {
        "best": require_canonical_h1_checkpoint(
            best_checkpoint, checkpoint_id="best", repository_root=root
        ),
        "last": require_canonical_h1_checkpoint(
            last_checkpoint, checkpoint_id="last", repository_root=root
        ),
    }
    verified = load_verified_phase1a_data(canonical_config)
    config = phase1a_training_config()
    train_dataset = _dataset(verified, "train", config)
    validation_dataset = _dataset(verified, "validation", config)
    if (
        len(train_dataset) != TRAIN_SAMPLE_COUNT
        or len(validation_dataset) != VALIDATION_SAMPLE_COUNT
    ):
        raise PermissionError("Phase 1B BN diagnostic requires the verified 300/100 split")
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader: DataLoader[dict[str, Any]] = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = require_fresh_phase1b_private_output(output_dir, repository_root=root)
    frozen = freeze_checkpoint_copies(sources, output_dir=output)
    budget = WallClockBudget.start(max_runtime_seconds)
    factory = model_factory
    if factory is None:
        from geoequi_ld.models.hrnet import HRNetW32SharedHeatmap

        def make_hrnet() -> nn.Module:
            return HRNetW32SharedHeatmap(align_corners=True)

        factory = make_hrnet
    dsnt = DSNT(temperature=0.05, align_corners=True).to(device)
    endpoints: dict[str, Any] = {}
    for checkpoint_id in H1_ENDPOINT_EPOCHS:
        copy_path = Path(frozen[checkpoint_id]["copy_path"])
        payload = read_checkpoint(copy_path, map_location="cpu")
        _validate_h1_payload(payload, checkpoint_id=checkpoint_id)
        raw_state = payload["model_state_dict"]
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"H1 {checkpoint_id} model state is invalid")
        state = {str(name): value for name, value in raw_state.items() if isinstance(value, Tensor)}
        if len(state) != len(raw_state):
            raise ValueError(f"H1 {checkpoint_id} model state contains non-tensors")
        payload.pop("optimizer_state_dict", None)

        original_model = _model_from_state(
            state,
            device=device,
            model_factory=factory,
        )
        original_train_metrics, original_train_audit = evaluate_model_without_state_change(
            original_model,
            train_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            budget=budget,
        )
        original_validation_metrics, original_validation_audit = (
            evaluate_model_without_state_change(
                original_model,
                validation_loader,
                dsnt=dsnt,
                device=device,
                config=config,
                budget=budget,
            )
        )
        del original_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        recalibrated_model = _model_from_state(
            state,
            device=device,
            model_factory=factory,
        )
        recalibration = reestimate_batch_norm_statistics(
            recalibrated_model,
            train_loader,
            device=device,
            expected_samples=TRAIN_SAMPLE_COUNT,
            budget=budget,
        )
        recalibrated_validation_metrics, recalibrated_validation_audit = (
            evaluate_model_without_state_change(
                recalibrated_model,
                validation_loader,
                dsnt=dsnt,
                device=device,
                config=config,
                budget=budget,
            )
        )
        del recalibrated_model, payload, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

        source_hash_after = sha256_file(sources[checkpoint_id])
        copy_hash_after = sha256_file(copy_path)
        source_unchanged = source_hash_after == frozen[checkpoint_id]["source_sha256_before"]
        copy_unchanged = copy_hash_after == frozen[checkpoint_id]["copy_sha256_before"]
        if not source_unchanged or not copy_unchanged:
            raise RuntimeError(f"Checkpoint bytes changed during diagnostic: {checkpoint_id}")
        endpoints[checkpoint_id] = {
            "epoch": H1_ENDPOINT_EPOCHS[checkpoint_id],
            "original_bn": {
                "train_metrics": original_train_metrics,
                "validation_metrics": original_validation_metrics,
                "train_evaluation_state_audit": original_train_audit,
                "validation_evaluation_state_audit": original_validation_audit,
            },
            "train_images_bn_reestimated": {
                "protocol_and_state_audit": recalibration,
                "validation_metrics": recalibrated_validation_metrics,
                "validation_evaluation_state_audit": recalibrated_validation_audit,
            },
            "source_checkpoint_unchanged": source_unchanged,
            "frozen_copy_unchanged": copy_unchanged,
            "source_sha256_after": source_hash_after,
            "copy_sha256_after": copy_hash_after,
        }

    result = {
        "schema_version": 1,
        "phase": "phase1b_bn_short_diagnostic",
        "status": "completed",
        "testing_frozen": True,
        "device": str(device),
        "max_runtime_seconds": max_runtime_seconds,
        "elapsed_seconds": budget.elapsed_seconds(),
        "data": {
            "train_count": len(train_dataset),
            "validation_count": len(validation_dataset),
            "fingerprints": {
                role: dict(verified.fingerprints[role]) for role in ALLOWED_SPLITS
            },
            "shuffle": False,
            "augmentation": "disabled",
        },
        "frozen_checkpoints": frozen,
        "endpoints": endpoints,
    }
    result["public_aggregate"] = build_public_bn_aggregate(result)
    return result


def write_json_strict(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write one new JSON file, converting non-finite floats to null."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite JSON output: {destination}")

    def safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): safe(nested) for key, nested in value.items()}
        if isinstance(value, list | tuple):
            return [safe(nested) for nested in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "__dataclass_fields__"):
            return safe(asdict(value))
        return value

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
