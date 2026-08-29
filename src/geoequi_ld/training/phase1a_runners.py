"""Fail-closed Phase 1A experiment runners.

This module deliberately keeps all data-derived outputs below ignored local
directories.  It never accepts a split name or path containing ``test`` or
``testing`` and delegates the exact train/validation fingerprint check to the
frozen Phase 0.5 access policy before constructing a dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from geoequi_ld.data.access_policy import (
    LabeledSplitSpec,
    fingerprint_labeled_split,
    load_phase05_local_splits,
    verify_fingerprint,
)
from geoequi_ld.data.dataset import IUGCLabeledDataset
from geoequi_ld.geometry.aop import compute_aop
from geoequi_ld.geometry.coordinates import normalized_to_pixel, pixel_to_normalized
from geoequi_ld.metrics.keypoints import summarize_keypoint_metrics
from geoequi_ld.models.dsnt import DSNT, spatial_softmax
from geoequi_ld.training.budget import (
    GpuBudgetLedger,
    require_fresh_output_directory,
)
from geoequi_ld.training.checkpoints import restore_checkpoint, save_checkpoint
from geoequi_ld.training.config import SupervisedTrainingConfig
from geoequi_ld.training.engine import (
    compute_supervised_losses,
    fit_supervised,
    train_for_steps_bounded,
    write_history_csv,
    write_json,
)
from geoequi_ld.training.runtime import make_generator, seed_everything

GateName = Literal["A4_unet_B0", "B4_hrnet_B2"]

PHASE1A_SEED = 42
TINY_SAMPLE_COUNT = 4
TRAIN_SAMPLE_COUNT = 300
VALIDATION_SAMPLE_COUNT = 100
TOTAL_GPU_LIMIT_SECONDS = 10_800.0
FORMAL_CLOSING_RESERVE_SECONDS = 900.0
PROTECTED_RESULT_ROOTS = ("runs/phase06", "reports", "checkpoints")


@dataclass(frozen=True)
class VerifiedData:
    specs: Mapping[str, LabeledSplitSpec]
    fingerprints: Mapping[str, Mapping[str, str | int]]


def _contains_forbidden_component(path: str | Path) -> bool:
    return any(part.casefold() in {"test", "testing"} for part in Path(path).parts)


def require_private_fresh_output(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Require a fresh output below ignored ``runs`` or ``artifacts`` only."""

    root = Path(repository_root).resolve()
    destination = Path(path).resolve()
    allowed_roots = ((root / "runs").resolve(), (root / "artifacts").resolve())
    if not any(
        destination == allowed or allowed in destination.parents for allowed in allowed_roots
    ):
        raise PermissionError("Phase 1A private output must stay below runs/ or artifacts/")
    if _contains_forbidden_component(destination):
        raise PermissionError("Phase 1A refuses a test/testing output path")
    protected = tuple((root / item).resolve() for item in PROTECTED_RESULT_ROOTS)
    return require_fresh_output_directory(destination, protected_roots=protected)


def require_private_existing_file(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    root = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    allowed_roots = ((root / "runs").resolve(), (root / "artifacts").resolve())
    if not any(candidate.is_relative_to(allowed) for allowed in allowed_roots):
        raise PermissionError("Gate artifacts must stay below runs/ or artifacts/")
    if _contains_forbidden_component(candidate):
        raise PermissionError("Phase 1A refuses a test/testing artifact path")
    if not candidate.is_file():
        raise FileNotFoundError(f"Gate artifact does not exist: {candidate}")
    return candidate


def load_verified_phase1a_data(local_config: str | Path) -> VerifiedData:
    """Load exactly train/validation and verify both before dataset access."""

    specs = load_phase05_local_splits(local_config)
    fingerprints: dict[str, Mapping[str, str | int]] = {}
    expected_counts = {"train": TRAIN_SAMPLE_COUNT, "validation": VALIDATION_SAMPLE_COUNT}
    for role in ("train", "validation"):
        spec = specs[role]
        actual = fingerprint_labeled_split(spec)
        verify_fingerprint(actual, spec.expected_fingerprint, role=role)
        if int(actual["sample_count"]) != expected_counts[role]:
            raise PermissionError(
                f"Phase 1A requires {expected_counts[role]} {role} samples, "
                f"fingerprint contains {actual['sample_count']}"
            )
        fingerprints[role] = actual
    return VerifiedData(specs=specs, fingerprints=fingerprints)


def fingerprint_digest(fingerprints: Mapping[str, Mapping[str, str | int]]) -> str:
    """Return a path-free digest of the verified train/validation contracts."""

    canonical = json.dumps(fingerprints, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _dataset(spec: LabeledSplitSpec, config: SupervisedTrainingConfig) -> IUGCLabeledDataset:
    source_columns = {"PS1": "PS1", "PS2": "PS2", "FH1": spec.fh1_column}
    return IUGCLabeledDataset(
        image_dir=spec.image_dir,
        labels_csv=spec.labels_csv,
        source_columns=source_columns,
        keypoint_order=config.keypoint_order,
        input_size_hw=config.input_size_hw,
        heatmap_size_hw=config.heatmap_size_hw,
        sigma=config.sigma_heatmap_px,
        align_corners=config.align_corners,
    )


def phase1a_training_config(*, gate: GateName | None = None) -> SupervisedTrainingConfig:
    """Materialize the locked B0/B2 settings in the legacy engine schema."""

    if gate == "A4_unet_B0":
        coordinate_weight, distribution_weight = 0.0, 0.0
    else:
        coordinate_weight, distribution_weight = 10.0, 1.0
    config = SupervisedTrainingConfig(
        seed=PHASE1A_SEED,
        device="cuda",
        deterministic=True,
        input_size_hw=(512, 512),
        heatmap_size_hw=(256, 256),
        sigma_heatmap_px=4.0,
        align_corners=True,
        dsnt_temperature=0.05,
        keypoint_order=("PS1", "PS2", "FH1"),
        aop_vertex_index=0,
        aop_pubic_axis_other_index=1,
        aop_fetal_head_index=2,
        base_channels=8,
        batch_size=1,
        epochs=20,
        learning_rate=0.001,
        weight_decay=0.0001,
        heatmap_loss_weight=1.0,
        coordinate_loss_weight=coordinate_weight,
        distribution_loss_weight=distribution_weight,
        max_grad_norm=5.0,
        num_workers=0,
        checkpoint_metric="aop_mae_deg",
    )
    config.validate()
    return config


def select_preregistered_tiny_indices(dataset_length: int) -> tuple[int, ...]:
    """Return the preregistered seed-42 sample indices, failing on split drift."""

    if dataset_length != TRAIN_SAMPLE_COUNT:
        raise PermissionError(
            f"Tiny gate is preregistered for {TRAIN_SAMPLE_COUNT} training samples"
        )
    return tuple(
        int(value)
        for value in torch.randperm(
            dataset_length,
            generator=make_generator(PHASE1A_SEED),
        )[:TINY_SAMPLE_COUNT]
    )


def _to_original_pixels(
    normalized: Tensor,
    original_sizes_hw: Tensor,
    *,
    align_corners: bool,
) -> Tensor:
    converted = []
    for points, size in zip(normalized, original_sizes_hw, strict=True):
        converted.append(
            normalized_to_pixel(
                points,
                (int(size[0].item()), int(size[1].item())),
                align_corners=align_corners,
            )
        )
    return torch.stack(converted)


def _batch_norm_modules(model: nn.Module) -> list[nn.Module]:
    batch_norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    return [module for module in model.modules() if isinstance(module, batch_norm_types)]


def _capture_batch_norm_buffers(model: nn.Module) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    state: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
            continue
        if (
            module.running_mean is None
            or module.running_var is None
            or module.num_batches_tracked is None
        ):
            continue
        state[name] = (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
    return state


def _restore_batch_norm_buffers(
    model: nn.Module,
    state: Mapping[str, tuple[Tensor, Tensor, Tensor]],
) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, (running_mean, running_var, batches) in state.items():
            module = modules[name]
            if not isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
                raise RuntimeError(f"BatchNorm module changed during tiny evaluation: {name}")
            assert module.running_mean is not None
            assert module.running_var is not None
            assert module.num_batches_tracked is not None
            module.running_mean.copy_(running_mean)
            module.running_var.copy_(running_var)
            module.num_batches_tracked.copy_(batches)


def _batch_norm_buffers_match(
    model: nn.Module,
    state: Mapping[str, tuple[Tensor, Tensor, Tensor]],
) -> bool:
    modules = dict(model.named_modules())
    for name, (running_mean, running_var, batches) in state.items():
        module = modules.get(name)
        if not isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
            return False
        if (
            module.running_mean is None
            or module.running_var is None
            or module.num_batches_tracked is None
        ):
            return False
        if not (
            torch.equal(module.running_mean, running_mean)
            and torch.equal(module.running_var, running_var)
            and torch.equal(module.num_batches_tracked, batches)
        ):
            return False
    return True


def evaluate_tiny_mode(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    dsnt: DSNT,
    device: torch.device,
    config: SupervisedTrainingConfig,
    mode: Literal["train", "eval"],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate real DSNT coordinates plus AoP and aggregate heatmap statistics."""

    model.train(mode == "train")
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    identifiers: list[str] = []
    raw_values: list[Tensor] = []
    probability_peaks: list[Tensor] = []
    private_rows: list[dict[str, Any]] = []
    total_losses: list[float] = []
    coordinate_error_count = 0
    nonfinite_count = 0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            heatmaps = batch["heatmaps"].to(device)
            target_normalized = batch["points_normalized"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            logits = model(image)
            losses = compute_supervised_losses(
                logits,
                heatmaps,
                target_normalized,
                valid_mask,
                dsnt=dsnt,
                heatmap_weight=config.heatmap_loss_weight,
                coordinate_weight=config.coordinate_loss_weight,
                distribution_weight=config.distribution_loss_weight,
            )
            total_losses.append(float(losses.total.cpu()))
            probabilities = spatial_softmax(logits, temperature=dsnt.temperature)
            normalized = dsnt(logits)
            finite = torch.isfinite(logits).all() & torch.isfinite(probabilities).all()
            finite = finite & torch.isfinite(normalized).all()
            if not bool(finite):
                nonfinite_count += int(image.shape[0])
            out_of_range = (normalized < -1.000001) | (normalized > 1.000001)
            coordinate_error_count += int(out_of_range.any(dim=(-1, -2)).sum().item())
            predicted_px = _to_original_pixels(
                normalized,
                batch["original_size_hw"].to(device),
                align_corners=config.align_corners,
            )
            target_px = batch["points_original_px"].to(device)
            predictions.append(predicted_px.cpu())
            targets.append(target_px.cpu())
            masks.append(batch["valid_mask"].cpu())
            raw_values.append(logits.detach().flatten().cpu())
            probability_peaks.append(probabilities.amax(dim=(-1, -2)).detach().flatten().cpu())
            names = [str(name) for name in batch["filename"]]
            identifiers.extend(names)
            for index, name in enumerate(names):
                private_rows.append(
                    {
                        "filename": name,
                        "predicted_original_xy": predicted_px[index].detach().cpu().tolist(),
                        "target_original_xy": target_px[index].detach().cpu().tolist(),
                    }
                )

    predicted = torch.cat(predictions)
    target = torch.cat(targets)
    valid_mask = torch.cat(masks).to(torch.bool)
    keypoint_metrics = summarize_keypoint_metrics(
        predicted,
        target,
        keypoint_names=config.keypoint_order,
        valid_mask=valid_mask,
    )
    predicted_aop, predicted_valid = compute_aop(
        predicted,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
        invalid="mask",
    )
    target_aop, target_valid = compute_aop(
        target,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
        invalid="mask",
    )
    evaluable = target_valid & valid_mask.all(dim=1)
    valid_aop = predicted_valid & evaluable
    angle_error = (predicted_aop - target_aop).abs()
    penalized = torch.where(predicted_valid, angle_error, torch.full_like(angle_error, 180.0))
    raw = torch.cat(raw_values)
    peaks = torch.cat(probability_peaks)
    metrics: dict[str, Any] = {
        "mode": mode,
        **keypoint_metrics,
        "mean_total_loss": sum(total_losses) / len(total_losses),
        "n_samples": int(predicted.shape[0]),
        "n_evaluable_aop": int(evaluable.sum()),
        "n_valid_aop": int(valid_aop.sum()),
        "aop_invalid_prediction_count": int((evaluable & ~predicted_valid).sum()),
        "aop_mae_deg": float(penalized[evaluable].mean()) if bool(evaluable.any()) else math.nan,
        "coordinate_error_count": coordinate_error_count,
        "nonfinite_count": nonfinite_count,
        "raw_heatmap_min": float(raw.min()),
        "raw_heatmap_max": float(raw.max()),
        "raw_heatmap_mean": float(raw.mean()),
        "raw_heatmap_std": float(raw.std(unbiased=False)),
        "probability_peak_mean": float(peaks.mean()),
        "probability_peak_min": float(peaks.min()),
    }
    return metrics, private_rows


def save_tiny_prediction_visualizations(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    dsnt: DSNT,
    device: torch.device,
    config: SupervisedTrainingConfig,
    private_run_root: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Save restricted overlays and three probability heatmaps per tiny sample."""

    repository = Path(repository_root).resolve()
    private_root = Path(private_run_root).resolve()
    allowed_roots = (
        (repository / "runs").resolve(),
        (repository / "artifacts").resolve(),
    )
    if not any(private_root.is_relative_to(allowed) for allowed in allowed_roots):
        raise PermissionError("Tiny visualizations must stay below ignored runs/ or artifacts/")
    if private_root.is_relative_to((repository / "runs" / "phase06").resolve()):
        raise PermissionError("Tiny visualizations cannot modify protected Phase 0.6 results")
    if _contains_forbidden_component(private_root):
        raise PermissionError("Tiny visualizations refuse a test/testing path")
    destination = (private_root / "predictions").resolve()
    if not destination.is_relative_to(private_root) or destination.parent != private_root:
        raise PermissionError("Tiny visualizations must stay in the private run predictions/")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    destination.mkdir(parents=False, exist_ok=False)
    model.eval()
    colors = ("#00E5FF", "#FFB000", "#FF4DA6")
    visualized = 0
    all_coordinates_finite = True
    all_targets_in_bounds = True
    all_predictions_in_bounds = True
    maximum_target_roundtrip_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            logits = model(image)
            probabilities = spatial_softmax(logits, temperature=dsnt.temperature)
            predicted_normalized = dsnt(logits)
            predicted_input = normalized_to_pixel(
                predicted_normalized,
                config.input_size_hw,
                align_corners=config.align_corners,
            )
            target_input = batch["points_input_px"].to(device)
            target_roundtrip = normalized_to_pixel(
                batch["points_normalized"].to(device),
                config.input_size_hw,
                align_corners=config.align_corners,
            )
            roundtrip_errors = torch.linalg.vector_norm(
                target_roundtrip - target_input,
                dim=-1,
            )
            maximum_target_roundtrip_error = max(
                maximum_target_roundtrip_error,
                float(roundtrip_errors.max().detach().cpu()),
            )
            finite = (
                torch.isfinite(predicted_input).all()
                & torch.isfinite(target_input).all()
                & torch.isfinite(probabilities).all()
            )
            all_coordinates_finite = all_coordinates_finite and bool(finite)
            height, width = config.input_size_hw
            target_in_bounds = (
                (target_input[..., 0] >= 0)
                & (target_input[..., 0] <= width - 1)
                & (target_input[..., 1] >= 0)
                & (target_input[..., 1] <= height - 1)
            )
            prediction_in_bounds = (
                (predicted_input[..., 0] >= 0)
                & (predicted_input[..., 0] <= width - 1)
                & (predicted_input[..., 1] >= 0)
                & (predicted_input[..., 1] <= height - 1)
            )
            all_targets_in_bounds = all_targets_in_bounds and bool(target_in_bounds.all())
            all_predictions_in_bounds = all_predictions_in_bounds and bool(
                prediction_in_bounds.all()
            )

            for batch_index in range(int(image.shape[0])):
                figure, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=140)
                axes[0].imshow(image[batch_index, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
                for keypoint_index, (name, color) in enumerate(
                    zip(config.keypoint_order, colors, strict=True)
                ):
                    target_x, target_y = target_input[batch_index, keypoint_index].detach().cpu()
                    prediction_x, prediction_y = (
                        predicted_input[batch_index, keypoint_index].detach().cpu()
                    )
                    axes[0].scatter(
                        float(target_x),
                        float(target_y),
                        c=color,
                        s=55,
                        marker="o",
                        edgecolors="black",
                    )
                    axes[0].scatter(
                        float(prediction_x),
                        float(prediction_y),
                        c=color,
                        s=70,
                        marker="x",
                        linewidths=2,
                    )
                    axes[0].text(
                        float(target_x) + 2,
                        float(target_y) - 2,
                        name,
                        color=color,
                        fontsize=8,
                    )
                axes[0].set_title("circle=target, x=DSNT prediction")
                axes[0].set_axis_off()
                for keypoint_index, name in enumerate(config.keypoint_order):
                    axes[keypoint_index + 1].imshow(
                        probabilities[batch_index, keypoint_index].detach().cpu(),
                        cmap="magma",
                    )
                    axes[keypoint_index + 1].set_title(f"predicted {name} probability")
                    axes[keypoint_index + 1].set_axis_off()
                figure.suptitle(
                    "RESTRICTED LOCAL DIAGNOSTIC - DO NOT COMMIT",
                    color="#A00000",
                    weight="bold",
                )
                figure.tight_layout()
                figure.savefig(
                    destination / f"sample_{visualized:02d}.png",
                    bbox_inches="tight",
                )
                plt.close(figure)
                visualized += 1

    (destination / "DO_NOT_COMMIT.txt").write_text(
        "Restricted medical-data derivative. Do not commit or upload without permission.\n",
        encoding="utf-8",
    )
    return {
        "visualization_count": visualized,
        "private_relative_directory": "predictions",
        "overlay_coordinate_system": "resized model-input pixels",
        "overlay_semantics": "circle=target points_input_px; x=DSNT prediction",
        "heatmap_semantics": "temperature-0.05 spatial-softmax probability per channel",
        "programmatic_coordinate_checks": {
            "all_coordinates_finite": all_coordinates_finite,
            "all_targets_in_input_bounds": all_targets_in_bounds,
            "all_predictions_in_input_bounds": all_predictions_in_bounds,
            "max_target_normalized_roundtrip_error_px": maximum_target_roundtrip_error,
        },
        "human_review_basis": (
            "Inspect target circles and DSNT crosses on the same resized input, then compare "
            "each cross with the corresponding PS1/PS2/FH1 probability-map peak."
        ),
    }


def b4_gate_passed(metrics: Mapping[str, Any]) -> bool:
    """Apply the strict preregistered B4 acceptance gate."""

    required_finite = (
        "MRE_PS1",
        "MRE_PS2",
        "MRE_FH1",
        "MRE_ALL",
        "aop_mae_deg",
        "mean_total_loss",
        "raw_heatmap_min",
        "raw_heatmap_max",
        "raw_heatmap_mean",
        "raw_heatmap_std",
        "probability_peak_mean",
        "probability_peak_min",
    )
    try:
        return bool(
            all(math.isfinite(float(metrics[name])) for name in required_finite)
            and float(metrics["MRE_ALL"]) <= 5.0
            and int(metrics["n_samples"]) == TINY_SAMPLE_COUNT
            and int(metrics["n_evaluable_aop"]) == TINY_SAMPLE_COUNT
            and int(metrics["n_valid_aop"]) == TINY_SAMPLE_COUNT
            and int(metrics["aop_invalid_prediction_count"]) == 0
            and int(metrics["coordinate_error_count"]) == 0
            and int(metrics["nonfinite_count"]) == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def a4_diagnostic_completed(metrics: Mapping[str, Any]) -> bool:
    """Return whether A4 produced an interpretable four-sample endpoint.

    A4 is a bounded diagnostic, not a promotion gate.  Completing the run and
    producing finite coordinates must therefore never be serialized as a
    scientific ``PASS``.
    """

    try:
        return bool(
            math.isfinite(float(metrics["MRE_ALL"]))
            and int(metrics["n_samples"]) == TINY_SAMPLE_COUNT
            and int(metrics["n_evaluable_aop"]) == TINY_SAMPLE_COUNT
            and int(metrics["coordinate_error_count"]) == 0
            and int(metrics["nonfinite_count"]) == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def a4_learning_outcome(metrics: Mapping[str, Any]) -> str:
    """Classify localization without turning A4 into a formal gate.

    Five pixels is reused only as a conservative four-sample localization
    check.  An endpoint inside that bound remains a candidate for raw-peak and
    overlay review; it is not automatically declared successful.
    """

    if not a4_diagnostic_completed(metrics):
        return "indeterminate"
    try:
        localized = (
            float(metrics["MRE_ALL"]) <= 5.0
            and int(metrics["n_valid_aop"]) == TINY_SAMPLE_COUNT
            and int(metrics["aop_invalid_prediction_count"]) == 0
        )
    except (KeyError, TypeError, ValueError):
        return "indeterminate"
    return "candidate_requires_peak_and_overlay_review" if localized else "not_learned"


def _ledger(path: str | Path, *, repository_root: str | Path) -> GpuBudgetLedger:
    root = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    allowed_roots = ((root / "runs").resolve(), (root / "artifacts").resolve())
    if not any(candidate.is_relative_to(allowed) for allowed in allowed_roots):
        raise PermissionError("GPU budget ledger must stay below runs/ or artifacts/")
    if _contains_forbidden_component(candidate):
        raise PermissionError("GPU ledger path cannot contain test/testing")
    return GpuBudgetLedger(candidate, total_limit_seconds=TOTAL_GPU_LIMIT_SECONDS)


def _finish_ledger(
    ledger: GpuBudgetLedger,
    name: str,
    *,
    started: float,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    ledger.finish(
        name,
        elapsed_seconds=time.perf_counter() - started,
        status=status,
        details=dict(details or {}),
    )


def run_b3_probe(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
    requested_seconds: float,
) -> dict[str, Any]:
    """Run one fixed FP32 HRNet-W32 structural/resource probe on CUDA."""

    if requested_seconds <= 0 or requested_seconds > 900.0:
        raise ValueError("B3 requested_seconds must be in (0, 900]")
    output = require_private_fresh_output(output_dir, repository_root=repository_root)
    from geoequi_ld.models.hrnet import HRNetW32SharedHeatmap
    from geoequi_ld.training.phase1a_config import (
        build_phase1a_adam,
        load_phase1a_hrnet_config,
    )

    config = load_phase1a_hrnet_config(config_path)
    if not torch.cuda.is_available():
        result = {
            "schema_version": 1,
            "gate_id": "B3",
            "gate": "FAIL",
            "status": "cuda_unavailable",
        }
        write_json(output / "b3_result.json", result)
        return result
    ledger = _ledger(ledger_path, repository_root=repository_root)
    allocation = ledger.begin("B3_structure_probe", requested_limit_seconds=requested_seconds)
    started = time.perf_counter()
    result: dict[str, Any]
    try:
        seed_everything(PHASE1A_SEED, deterministic=True)
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = HRNetW32SharedHeatmap(align_corners=True).to(device=device, dtype=torch.float32)
        dsnt = DSNT(temperature=0.05, align_corners=True).to(device)
        optimizer = build_phase1a_adam(model.parameters(), config)
        stage_input_shapes: list[list[int]] = []
        stage_output_shapes: list[list[int]] = []
        stage_input_grad_l1: list[float] = [0.0, 0.0, 0.0, 0.0]

        def stage_hook(_module: nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
            branches_in = inputs[0] if inputs and isinstance(inputs[0], list | tuple) else ()
            branches_out = outputs if isinstance(outputs, list | tuple) else ()
            if len(branches_in) != 4 or len(branches_out) != 4:
                raise RuntimeError("Final stage4 fusion did not expose four scales")
            if not stage_input_shapes:
                stage_input_shapes.extend([list(tensor.shape) for tensor in branches_in])
                stage_output_shapes.extend([list(tensor.shape) for tensor in branches_out])
                for index, tensor in enumerate(branches_in):
                    tensor.register_hook(
                        lambda gradient, i=index: stage_input_grad_l1.__setitem__(
                            i, float(gradient.detach().abs().sum().cpu())
                        )
                    )

        hook_handle = model.final_fusion_module.register_forward_hook(stage_hook)
        image = torch.rand((1, 1, 512, 512), dtype=torch.float32, device=device)
        points_hm = torch.tensor(
            [[80.0, 160.0], [160.0, 160.0], [105.0, 80.0]],
            dtype=torch.float32,
            device=device,
        )
        from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps

        target_heatmaps = generate_gaussian_heatmaps(
            points_hm,
            size_hw=(256, 256),
            sigma=4.0,
        ).unsqueeze(0)
        target_normalized = pixel_to_normalized(
            points_hm,
            (256, 256),
            align_corners=True,
        ).unsqueeze(0)
        valid = torch.ones((1, 3), dtype=torch.bool, device=device)
        backbone_parameter = next(model.backbone.parameters())
        decoder_parameter = next(model.decoder.parameters())
        backbone_before = backbone_parameter.detach().clone()
        decoder_before = decoder_parameter.detach().clone()

        first_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        dsnt_coordinates = dsnt(logits)
        losses = compute_supervised_losses(
            logits,
            target_heatmaps,
            target_normalized,
            valid,
            dsnt=dsnt,
            heatmap_weight=1.0,
            coordinate_weight=10.0,
            distribution_weight=1.0,
        )
        losses.total.backward()
        preclip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        backbone_grad_l1 = sum(
            float(parameter.grad.detach().abs().sum().cpu())
            for parameter in model.backbone.parameters()
            if parameter.grad is not None
        )
        decoder_grad_l1 = sum(
            float(parameter.grad.detach().abs().sum().cpu())
            for parameter in model.decoder.parameters()
            if parameter.grad is not None
        )
        optimizer.step()  # first step allocates Adam state and is part of the resource probe
        torch.cuda.synchronize(device)
        first_step_seconds = time.perf_counter() - first_started
        hook_handle.remove()
        backbone_delta = float((backbone_parameter.detach() - backbone_before).abs().sum().cpu())
        decoder_delta = float((decoder_parameter.detach() - decoder_before).abs().sum().cpu())

        # The first full step is the warm-up. Time one identical full step afterward.
        warm_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        warm_logits = model(image)
        warm_losses = compute_supervised_losses(
            warm_logits,
            target_heatmaps,
            target_normalized,
            valid,
            dsnt=dsnt,
            heatmap_weight=1.0,
            coordinate_weight=10.0,
            distribution_weight=1.0,
        )
        warm_losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        warmed_step_seconds = time.perf_counter() - warm_started

        batch_norm_modules = _batch_norm_modules(model)
        model.eval()
        batch_norm_eval_switched = bool(batch_norm_modules) and all(
            not module.training for module in batch_norm_modules
        )
        with torch.inference_mode():
            _ = model(image)  # eval warm-up
            torch.cuda.synchronize(device)
            eval_started = time.perf_counter()
            eval_logits = model(image)
            torch.cuda.synchronize(device)
            warmed_eval_seconds = time.perf_counter() - eval_started

        checkpoint_path = output / "b3_roundtrip.pt"
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=0,
            config={"phase": "phase1a", "gate": "B3", "protocol": asdict(config)},
            seed=PHASE1A_SEED,
            metrics={"total_loss": float(warm_losses.total.detach().cpu())},
        )
        saved_value = backbone_parameter.detach().clone()
        with torch.no_grad():
            backbone_parameter.add_(1.0)
        restore_checkpoint(checkpoint_path, model=model, optimizer=optimizer, map_location=device)
        checkpoint_parameter_roundtrip = bool(
            torch.equal(saved_value, backbone_parameter.detach())
        )
        model.eval()
        with torch.inference_mode():
            restored_logits = model(image)
        checkpoint_output_roundtrip = bool(torch.equal(eval_logits, restored_logits))
        model.train()
        batch_norm_train_switched = bool(batch_norm_modules) and all(
            module.training for module in batch_norm_modules
        )
        with torch.inference_mode():
            train_mode_logits = model(image)

        expected_scales = [
            [1, 32, 128, 128],
            [1, 64, 64, 64],
            [1, 128, 32, 32],
            [1, 256, 16, 16],
        ]
        elapsed = time.perf_counter() - started
        checks = {
            "fp32_batch1_512": image.dtype == torch.float32
            and list(image.shape) == [1, 1, 512, 512],
            "output_shape": list(logits.shape) == [1, 3, 256, 256],
            "dsnt_shape": list(dsnt_coordinates.shape) == [1, 3, 2],
            "stage4_four_scale_shapes": stage_output_shapes == expected_scales,
            "stage4_four_scale_input_gradients": len(stage_input_grad_l1) == 4
            and all(math.isfinite(value) and value > 0 for value in stage_input_grad_l1),
            "losses_finite": all(
                math.isfinite(float(value.detach().cpu()))
                for value in (
                    losses.total,
                    losses.heatmap_mse,
                    losses.coordinate_smooth_l1,
                    losses.distribution_js,
                )
            ),
            "gradients_finite": gradients_finite,
            "backbone_nonzero_gradient": backbone_grad_l1 > 0,
            "decoder_nonzero_gradient": decoder_grad_l1 > 0,
            "backbone_updated": backbone_delta > 0,
            "decoder_updated": decoder_delta > 0,
            "train_eval_finite": bool(torch.isfinite(train_mode_logits).all())
            and bool(torch.isfinite(eval_logits).all()),
            "batch_norm_eval_switched": batch_norm_eval_switched,
            "batch_norm_train_switched": batch_norm_train_switched,
            "checkpoint_parameter_roundtrip": checkpoint_parameter_roundtrip,
            "checkpoint_output_roundtrip": checkpoint_output_roundtrip,
            "within_allocated_time": elapsed <= allocation,
        }
        passed = all(bool(value) for value in checks.values())
        result = {
            "schema_version": 1,
            "gate_id": "B3",
            "gate": "PASS" if passed else "FAIL",
            "status": "completed",
            "checks": checks,
            "feature_contract": model.feature_contract.to_dict(),
            "stage4": {
                "input_shapes": stage_input_shapes,
                "output_shapes": stage_output_shapes,
                "input_gradient_l1": stage_input_grad_l1,
            },
            "losses": {
                "total": float(losses.total.detach().cpu()),
                "heatmap_mse": float(losses.heatmap_mse.detach().cpu()),
                "coordinate_smooth_l1": float(losses.coordinate_smooth_l1.detach().cpu()),
                "distribution_js": float(losses.distribution_js.detach().cpu()),
            },
            "gradients": {
                "preclip_global_norm": float(preclip_norm.detach().cpu()),
                "backbone_l1": backbone_grad_l1,
                "decoder_l1": decoder_grad_l1,
            },
            "batch_norm_module_count": len(batch_norm_modules),
            "timing_seconds": {
                "first_adam_step": first_step_seconds,
                "warmed_full_step": warmed_step_seconds,
                "warmed_eval_forward": warmed_eval_seconds,
                "total": elapsed,
                "allocated": allocation,
            },
            "cuda_memory_mb": {
                "peak_allocated": torch.cuda.max_memory_allocated(device) / (1024**2),
                "peak_reserved": torch.cuda.max_memory_reserved(device) / (1024**2),
            },
        }
        write_json(output / "b3_result.json", result)
        _finish_ledger(
            ledger,
            "B3_structure_probe",
            started=started,
            status="completed" if passed else "failed",
            details={"gate": result["gate"]},
        )
        return result
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "gate_id": "B3",
            "gate": "FAIL",
            "status": "oom",
            "error_type": type(error).__name__,
            "adaptation_attempted": False,
        }
        write_json(output / "b3_result.json", result)
        _finish_ledger(ledger, "B3_structure_probe", started=started, status="oom")
        return result
    except Exception as error:
        result = {
            "schema_version": 1,
            "gate_id": "B3",
            "gate": "FAIL",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(output / "b3_result.json", result)
        _finish_ledger(ledger, "B3_structure_probe", started=started, status="failed")
        raise


def run_tiny_gate(
    *,
    gate: GateName,
    local_config: str | Path,
    hrnet_config: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
    b3_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Run the deterministic four-sample A4 or B4 bounded gate."""

    limits = {
        "A4_unet_B0": (1000, 1200.0),
        "B4_hrnet_B2": (500, 2400.0),
    }
    if gate not in limits:
        raise ValueError(f"Unknown tiny gate: {gate}")
    if gate == "B4_hrnet_B2":
        if b3_artifact is None:
            raise PermissionError("B4 requires an explicit passed B3 artifact")
        require_passed_gate_artifact(
            b3_artifact,
            expected_gate_id="B3",
            repository_root=repository_root,
        )
    output = require_private_fresh_output(output_dir, repository_root=repository_root)
    verified = load_verified_phase1a_data(local_config)
    config = phase1a_training_config(gate=gate)
    train_dataset = _dataset(verified.specs["train"], config)
    indices = select_preregistered_tiny_indices(len(train_dataset))
    subset = Subset(train_dataset, indices)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=make_generator(PHASE1A_SEED + 1),
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
            "gate_id": gate,
            "gate": "FAIL" if gate == "B4_hrnet_B2" else "NOT_APPLICABLE",
            "status": "cuda_unavailable",
            "diagnostic_completion": (
                "not_run" if gate == "A4_unet_B0" else "not_applicable"
            ),
        }
        write_json(output / "tiny_gate_result.json", result)
        return result
    max_steps, requested_seconds = limits[gate]
    ledger = _ledger(ledger_path, repository_root=repository_root)
    allocation = ledger.begin(gate, requested_limit_seconds=requested_seconds)
    diagnostic_reserve = min(0.1 * allocation, 120.0 if gate == "A4_unet_B0" else 240.0)
    training_allocation = allocation - diagnostic_reserve
    if training_allocation <= 0:
        raise RuntimeError("No tiny-gate training budget remains after diagnostic reserve")
    started = time.perf_counter()
    try:
        seed_everything(PHASE1A_SEED, deterministic=True)
        device = torch.device("cuda")
        if gate == "A4_unet_B0":
            from geoequi_ld.models.unet import HeatmapUNet

            model: nn.Module = HeatmapUNet(base_channels=8).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                foreach=False,
            )
        else:
            from geoequi_ld.models.hrnet import HRNetW32SharedHeatmap
            from geoequi_ld.training.phase1a_config import (
                build_phase1a_adam,
                load_phase1a_hrnet_config,
            )

            hrnet = load_phase1a_hrnet_config(hrnet_config)
            model = HRNetW32SharedHeatmap(align_corners=True).to(device)
            optimizer = build_phase1a_adam(model.parameters(), hrnet)
        dsnt = DSNT(temperature=0.05, align_corners=True).to(device)
        bounded = train_for_steps_bounded(
            model,
            train_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
            max_steps=max_steps,
            max_runtime_seconds=training_allocation,
        )
        eval_metrics, eval_private = evaluate_tiny_mode(
            model,
            evaluation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            mode="eval",
        )
        batch_norm_before_train_diagnostic = _capture_batch_norm_buffers(model)
        train_metrics, train_private = evaluate_tiny_mode(
            model,
            evaluation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            mode="train",
        )
        _restore_batch_norm_buffers(model, batch_norm_before_train_diagnostic)
        batch_norm_buffers_restored = _batch_norm_buffers_match(
            model,
            batch_norm_before_train_diagnostic,
        )
        model.eval()
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
        visualization_programmatic_pass = bool(
            visualization["visualization_count"] == TINY_SAMPLE_COUNT
            and coordinate_checks["all_coordinates_finite"]
            and coordinate_checks["all_targets_in_input_bounds"]
            and coordinate_checks["all_predictions_in_input_bounds"]
            and float(coordinate_checks["max_target_normalized_roundtrip_error_px"])
            <= 1.0e-3
        )
        diagnostic_integrity = batch_norm_buffers_restored and visualization_programmatic_pass
        passed = b4_gate_passed(eval_metrics) and diagnostic_integrity
        a4_completion = a4_diagnostic_completed(eval_metrics) and diagnostic_integrity
        result = {
            "schema_version": 1,
            "gate_id": gate,
            "gate": (
                "PASS" if passed else "FAIL"
            )
            if gate == "B4_hrnet_B2"
            else "NOT_APPLICABLE",
            "status": bounded.status,
            "diagnostic_completion": (
                "completed" if a4_completion else "invalid"
            )
            if gate == "A4_unet_B0"
            else "not_applicable",
            "learning_outcome": (
                a4_learning_outcome(eval_metrics) if a4_completion else "indeterminate"
            )
            if gate == "A4_unet_B0"
            else "not_applicable",
            "steps_completed": bounded.steps_completed,
            "max_steps": max_steps,
            "training_elapsed_seconds": bounded.elapsed_seconds,
            "training_allocated_seconds": training_allocation,
            "total_allocated_seconds": allocation,
            "diagnostic_reserve_seconds": diagnostic_reserve,
            "sample_selection": {
                "algorithm": "torch.randperm(train=300, seed=42)[:4]",
                "sample_count": TINY_SAMPLE_COUNT,
                "indices_omitted_from_gate_result": True,
            },
            "data_fingerprint_digest": fingerprint_digest(verified.fingerprints),
            "eval_mode": eval_metrics,
            "train_mode": train_metrics,
            "batch_norm_buffers_restored_after_train_mode_diagnostic": (
                batch_norm_buffers_restored
            ),
            "checkpoint_saved_in_eval_mode": not model.training,
            "visualization": {
                **visualization,
                "programmatic_check_passed": visualization_programmatic_pass,
                "manual_review_status": "pending",
            },
            "augmentation": "disabled",
            "batch_size": 1,
            "precision": "float32",
        }
        write_history_csv(output / "train_log.csv", bounded.history)
        write_json(
            output / "private_predictions.json",
            {
                "restricted_local_output": True,
                "selected_indices": list(indices),
                "eval_mode": eval_private,
                "train_mode": train_private,
            },
        )
        save_checkpoint(
            output / "tiny_gate.pt",
            model=model,
            optimizer=optimizer,
            epoch=0,
            config={"phase": "phase1a", "gate": gate, "training": config.to_dict()},
            seed=PHASE1A_SEED,
            metrics=eval_metrics,
            extra={"steps_completed": bounded.steps_completed, "status": bounded.status},
        )
        total_elapsed = time.perf_counter() - started
        result["total_elapsed_seconds"] = total_elapsed
        result["within_total_allocation"] = total_elapsed <= allocation
        if not result["within_total_allocation"] and gate == "B4_hrnet_B2":
            result["gate"] = "FAIL"
        if not result["within_total_allocation"] and gate == "A4_unet_B0":
            result["diagnostic_completion"] = "invalid"
            result["learning_outcome"] = "indeterminate"
        write_json(output / "tiny_gate_result.json", result)
        ledger_status = "completed" if bounded.status == "completed" else "budget_exhausted"
        _finish_ledger(
            ledger,
            gate,
            started=started,
            status=ledger_status,
            details={"gate": result["gate"], "steps_completed": bounded.steps_completed},
        )
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "gate_id": gate,
            "gate": "FAIL" if gate == "B4_hrnet_B2" else "NOT_APPLICABLE",
            "status": "oom",
            "diagnostic_completion": (
                "not_run" if gate == "A4_unet_B0" else "not_applicable"
            ),
            "adaptation_attempted": False,
        }
        write_json(output / "tiny_gate_result.json", result)
        _finish_ledger(ledger, gate, started=started, status="oom")
        return result
    except Exception:
        _finish_ledger(ledger, gate, started=started, status="failed")
        raise


def require_passed_gate_artifact(
    path: str | Path,
    *,
    expected_gate_id: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate a private B3/B4 artifact before formal training."""

    artifact = require_private_existing_file(path, repository_root=repository_root)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gate artifact root must be an object")
    if payload.get("schema_version") != 1 or payload.get("gate_id") != expected_gate_id:
        raise PermissionError(f"Expected a schema-1 {expected_gate_id} gate artifact")
    if payload.get("gate") != "PASS":
        raise PermissionError(f"{expected_gate_id} did not pass")
    if expected_gate_id == "B3":
        if payload.get("status") != "completed":
            raise PermissionError("B3 PASS must have completed status")
        required_checks = {
            "fp32_batch1_512",
            "output_shape",
            "dsnt_shape",
            "stage4_four_scale_shapes",
            "stage4_four_scale_input_gradients",
            "losses_finite",
            "gradients_finite",
            "backbone_nonzero_gradient",
            "decoder_nonzero_gradient",
            "backbone_updated",
            "decoder_updated",
            "train_eval_finite",
            "batch_norm_eval_switched",
            "batch_norm_train_switched",
            "checkpoint_parameter_roundtrip",
            "checkpoint_output_roundtrip",
            "within_allocated_time",
        }
        checks = payload.get("checks")
        if not isinstance(checks, Mapping) or not required_checks.issubset(checks):
            raise PermissionError("B3 artifact is missing required structural checks")
        if not all(checks[name] is True for name in required_checks):
            raise PermissionError("B3 artifact contains a failed structural check")
        contract = payload.get("feature_contract")
        expected_contract = {
            "timm_version": "1.0.28",
            "backbone_name": "hrnet_w32",
            "feature_location": "",
            "out_indices": [1],
            "channels": [32],
            "reductions": [4],
        }
        if not isinstance(contract, Mapping) or any(
            contract.get(key) != value for key, value in expected_contract.items()
        ):
            raise PermissionError("B3 HRNet feature contract does not match Phase 1A")
        stage4 = payload.get("stage4")
        expected_shapes = [
            [1, 32, 128, 128],
            [1, 64, 64, 64],
            [1, 128, 32, 32],
            [1, 256, 16, 16],
        ]
        if not isinstance(stage4, Mapping) or stage4.get("output_shapes") != expected_shapes:
            raise PermissionError("B3 stage4 four-scale evidence is incomplete")
        input_gradients = stage4.get("input_gradient_l1")
        if not isinstance(input_gradients, list | tuple) or len(input_gradients) != 4:
            raise PermissionError("B3 stage4 gradient evidence is incomplete")
        try:
            gradients_valid = all(
                math.isfinite(float(value)) and float(value) > 0 for value in input_gradients
            )
        except (TypeError, ValueError):
            gradients_valid = False
        if not gradients_valid:
            raise PermissionError("B3 stage4 gradient evidence contains a failed branch")
    if expected_gate_id == "B4_hrnet_B2":
        if payload.get("status") not in {"completed", "budget_exhausted"}:
            raise PermissionError("B4 did not reach an allowed bounded terminal status")
        if (
            payload.get("within_total_allocation") is not True
            or payload.get("batch_norm_buffers_restored_after_train_mode_diagnostic") is not True
            or payload.get("checkpoint_saved_in_eval_mode") is not True
            or payload.get("augmentation") != "disabled"
            or payload.get("batch_size") != 1
            or payload.get("precision") != "float32"
        ):
            raise PermissionError("B4 execution-integrity evidence is incomplete")
        eval_metrics = payload.get("eval_mode")
        if not isinstance(eval_metrics, Mapping) or not b4_gate_passed(eval_metrics):
            raise PermissionError("B4 stored metrics no longer satisfy the strict gate")
        visualization = payload.get("visualization")
        if not isinstance(visualization, Mapping) or (
            visualization.get("programmatic_check_passed") is not True
            or visualization.get("manual_review_status") != "passed"
        ):
            raise PermissionError("B4 overlay review has not been explicitly passed")
    return payload


def run_formal_hrnet(
    *,
    local_config: str | Path,
    hrnet_config: str | Path,
    b3_artifact: str | Path,
    b4_artifact: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Run H1_shared_B2_seed42_20e only after strict B3+B4 acceptance."""

    require_passed_gate_artifact(
        b3_artifact,
        expected_gate_id="B3",
        repository_root=repository_root,
    )
    require_passed_gate_artifact(
        b4_artifact,
        expected_gate_id="B4_hrnet_B2",
        repository_root=repository_root,
    )
    output = require_private_fresh_output(output_dir, repository_root=repository_root)
    verified = load_verified_phase1a_data(local_config)
    from geoequi_ld.models.hrnet import HRNetW32SharedHeatmap
    from geoequi_ld.training.phase1a_config import (
        build_phase1a_adam,
        load_phase1a_hrnet_config,
    )

    phase_config = load_phase1a_hrnet_config(hrnet_config)
    config = phase1a_training_config()
    config = replace(config, epochs=20)
    train_dataset = _dataset(verified.specs["train"], config)
    validation_dataset = _dataset(verified.specs["validation"], config)
    if (
        len(train_dataset) != TRAIN_SAMPLE_COUNT
        or len(validation_dataset) != VALIDATION_SAMPLE_COUNT
    ):
        raise PermissionError("Formal H1 requires the verified 300/100 split")
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=make_generator(PHASE1A_SEED),
        pin_memory=True,
    )
    validation_loader: DataLoader[dict[str, Any]] = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Phase 1A requires CUDA; no CPU fallback is permitted")
    ledger = _ledger(ledger_path, repository_root=repository_root)
    allocation = ledger.begin(
        "H1_shared_B2_seed42_20e",
        requested_limit_seconds=7200.0,
        reserve_after_seconds=FORMAL_CLOSING_RESERVE_SECONDS,
    )
    started = time.perf_counter()
    checkpoint_config = {
        "phase": "phase1a",
        "experiment_name": "H1_shared_B2_seed42_20e",
        "testing_frozen": True,
        "training": config.to_dict(),
        "model": phase_config.model.to_dict(),
        "optimizer": phase_config.optimizer.to_dict(),
        "data": {
            "train_count": TRAIN_SAMPLE_COUNT,
            "validation_count": VALIDATION_SAMPLE_COUNT,
            "fingerprint_digest": fingerprint_digest(verified.fingerprints),
            "paths_embedded": False,
        },
        "runtime": {
            "allocated_seconds": allocation,
            "formal_cap_seconds": 7200.0,
            "closing_reserve_seconds": FORMAL_CLOSING_RESERVE_SECONDS,
        },
    }
    try:
        seed_everything(PHASE1A_SEED, deterministic=True)
        device = torch.device("cuda")
        model = HRNetW32SharedHeatmap(align_corners=True).to(device)
        optimizer = build_phase1a_adam(model.parameters(), phase_config)
        dsnt = DSNT(temperature=0.05, align_corners=True).to(device)
        summary = fit_supervised(
            model,
            train_loader,
            validation_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
            output_dir=output,
            checkpoint_config=checkpoint_config,
            max_runtime_seconds=allocation,
        )
        result = {
            "schema_version": 1,
            "experiment_name": "H1_shared_B2_seed42_20e",
            "status": summary["status"],
            "epochs_completed": summary["epochs_completed"],
            "epochs_requested": 20,
            "partial": summary["status"] != "completed",
            "selection_split": "validation",
            "selection_order": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
            "runtime_elapsed_sec": summary["runtime_elapsed_sec"],
            "runtime_allocated_sec": allocation,
            "best_epoch": summary["best_epoch"],
            "best_validation_metrics": summary["best_validation_metrics"],
        }
        write_json(output / "formal_result.json", result)
        ledger_status = "completed" if summary["status"] == "completed" else "budget_exhausted"
        _finish_ledger(
            ledger,
            "H1_shared_B2_seed42_20e",
            started=started,
            status=ledger_status,
            details={"epochs_completed": summary["epochs_completed"]},
        )
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "experiment_name": "H1_shared_B2_seed42_20e",
            "status": "oom",
            "partial": True,
            "adaptation_attempted": False,
        }
        write_json(output / "formal_result.json", result)
        _finish_ledger(
            ledger,
            "H1_shared_B2_seed42_20e",
            started=started,
            status="oom",
        )
        return result
    except Exception:
        _finish_ledger(
            ledger,
            "H1_shared_B2_seed42_20e",
            started=started,
            status="failed",
        )
        raise
