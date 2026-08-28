"""Minimal supervised training and evaluation for Phase 0."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from geoequi_ld.geometry.aop import compute_aop
from geoequi_ld.geometry.coordinates import normalized_to_pixel
from geoequi_ld.metrics.keypoints import absolute_angle_error, summarize_keypoint_metrics
from geoequi_ld.models.decoding import decode_heatmaps
from geoequi_ld.models.dsnt import DSNT, spatial_expectation, spatial_softmax
from geoequi_ld.training.checkpoints import save_checkpoint
from geoequi_ld.training.config import SupervisedTrainingConfig


@dataclass(frozen=True)
class SupervisedLosses:
    total: Tensor
    heatmap_mse: Tensor
    coordinate_smooth_l1: Tensor
    distribution_js: Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_float = mask.to(device=values.device, dtype=values.dtype)
    denominator = mask_float.sum()
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("A supervised batch contains no valid keypoints")
    return (values * mask_float).sum() / denominator


def compute_supervised_losses(
    heatmap_logits: Tensor,
    target_heatmaps: Tensor,
    target_normalized_xy: Tensor,
    valid_mask: Tensor,
    *,
    dsnt: DSNT,
    heatmap_weight: float,
    coordinate_weight: float,
    distribution_weight: float,
) -> SupervisedLosses:
    """Combine heatmap, continuous-coordinate, and distribution supervision.

    The MSE term matches the *raw* predicted heatmap to the unnormalized
    Gaussian target.  DSNT instead applies a temperature-scaled spatial
    softmax and the SmoothL1 term supervises its continuous coordinate.  A
    good heatmap MSE alone does not imply a good softmax expectation on a large
    grid, so the coordinate term must not be silently omitted when DSNT/AoP is
    part of the trainable path. Jensen-Shannon divergence additionally makes
    the spatial-softmax distribution resemble the normalized Gaussian target.
    """

    if heatmap_logits.shape != target_heatmaps.shape:
        predicted_shape = tuple(heatmap_logits.shape)
        target_shape = tuple(target_heatmaps.shape)
        raise ValueError(f"Heatmap shape mismatch: {predicted_shape} != {target_shape}")
    if target_normalized_xy.shape != (*heatmap_logits.shape[:2], 2):
        raise ValueError("target_normalized_xy must have shape [B,K,2]")
    if valid_mask.shape != heatmap_logits.shape[:2]:
        raise ValueError("valid_mask must have shape [B,K]")
    mask = valid_mask.to(device=heatmap_logits.device, dtype=torch.bool)
    per_keypoint_mse = (heatmap_logits - target_heatmaps).square().mean(dim=(-1, -2))
    heatmap_mse = _masked_mean(per_keypoint_mse, mask)

    zero = heatmap_logits.new_zeros(())
    predicted_probabilities: Tensor | None = None
    if coordinate_weight > 0 or distribution_weight > 0:
        predicted_probabilities = spatial_softmax(
            heatmap_logits,
            temperature=dsnt.temperature,
        )
    if coordinate_weight > 0:
        assert predicted_probabilities is not None
        predicted_normalized = spatial_expectation(
            predicted_probabilities,
            align_corners=dsnt.align_corners,
        )
        per_coordinate = F.smooth_l1_loss(
            predicted_normalized,
            target_normalized_xy,
            reduction="none",
        ).mean(dim=-1)
        coordinate_smooth_l1 = _masked_mean(per_coordinate, mask)
    else:
        coordinate_smooth_l1 = zero
    if distribution_weight > 0:
        assert predicted_probabilities is not None
        target_probabilities = target_heatmaps / target_heatmaps.sum(
            dim=(-1, -2),
            keepdim=True,
        ).clamp_min(1e-12)
        mixture = 0.5 * (predicted_probabilities + target_probabilities)
        eps = torch.finfo(heatmap_logits.dtype).eps
        log_mixture = mixture.clamp_min(eps).log()
        kl_prediction = (
            predicted_probabilities
            * (predicted_probabilities.clamp_min(eps).log() - log_mixture)
        ).sum(dim=(-1, -2))
        kl_target = (
            target_probabilities * (target_probabilities.clamp_min(eps).log() - log_mixture)
        ).sum(dim=(-1, -2))
        distribution_js = _masked_mean(0.5 * (kl_prediction + kl_target), mask)
    else:
        distribution_js = zero
    total = (
        heatmap_weight * heatmap_mse
        + coordinate_weight * coordinate_smooth_l1
        + distribution_weight * distribution_js
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Supervised loss became NaN or Inf")
    return SupervisedLosses(total, heatmap_mse, coordinate_smooth_l1, distribution_js)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _loss_accumulator() -> dict[str, float]:
    return {
        "total_loss": 0.0,
        "heatmap_mse": 0.0,
        "coordinate_smooth_l1": 0.0,
        "distribution_js": 0.0,
        "samples": 0.0,
    }


def _add_losses(accumulator: dict[str, float], losses: SupervisedLosses, batch_size: int) -> None:
    accumulator["total_loss"] += float(losses.total.detach().cpu()) * batch_size
    accumulator["heatmap_mse"] += float(losses.heatmap_mse.detach().cpu()) * batch_size
    coordinate_loss = float(losses.coordinate_smooth_l1.detach().cpu())
    accumulator["coordinate_smooth_l1"] += coordinate_loss * batch_size
    accumulator["distribution_js"] += float(losses.distribution_js.detach().cpu()) * batch_size
    accumulator["samples"] += batch_size


def _mean_losses(accumulator: Mapping[str, float]) -> dict[str, float]:
    samples = accumulator["samples"]
    if samples <= 0:
        raise ValueError("Data loader produced no samples")
    return {
        "total_loss": accumulator["total_loss"] / samples,
        "heatmap_mse": accumulator["heatmap_mse"] / samples,
        "coordinate_smooth_l1": accumulator["coordinate_smooth_l1"] / samples,
        "distribution_js": accumulator["distribution_js"] / samples,
    }


def _require_finite_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("A model gradient became NaN or Inf")


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    *,
    dsnt: DSNT,
    device: torch.device,
    config: SupervisedTrainingConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Train one epoch, optionally bounded for a quick diagnostic run."""

    model.train()
    accumulator = _loss_accumulator()
    batches = 0
    for raw_batch in data_loader:
        batch = _to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"])
        losses = compute_supervised_losses(
            logits,
            batch["heatmaps"],
            batch["points_normalized"],
            batch["valid_mask"],
            dsnt=dsnt,
            heatmap_weight=config.heatmap_loss_weight,
            coordinate_weight=config.coordinate_loss_weight,
            distribution_weight=config.distribution_loss_weight,
        )
        losses.total.backward()
        if config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        _require_finite_gradients(model)
        optimizer.step()
        batch_size = int(batch["image"].shape[0])
        _add_losses(accumulator, losses, batch_size)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    metrics = _mean_losses(accumulator)
    metrics["batches"] = float(batches)
    return metrics


def train_for_steps(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    *,
    dsnt: DSNT,
    device: torch.device,
    config: SupervisedTrainingConfig,
    max_steps: int,
) -> list[dict[str, float]]:
    """Repeat a tiny loader until exactly ``max_steps`` optimizer steps."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    history: list[dict[str, float]] = []
    steps = 0
    while steps < max_steps:
        for raw_batch in data_loader:
            batch = _to_device(raw_batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"])
            losses = compute_supervised_losses(
                logits,
                batch["heatmaps"],
                batch["points_normalized"],
                batch["valid_mask"],
                dsnt=dsnt,
                heatmap_weight=config.heatmap_loss_weight,
                coordinate_weight=config.coordinate_loss_weight,
                distribution_weight=config.distribution_loss_weight,
            )
            losses.total.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            _require_finite_gradients(model)
            optimizer.step()
            steps += 1
            history.append(
                {
                    "step": float(steps),
                    "total_loss": float(losses.total.detach().cpu()),
                    "heatmap_mse": float(losses.heatmap_mse.detach().cpu()),
                    "coordinate_smooth_l1": float(losses.coordinate_smooth_l1.detach().cpu()),
                    "distribution_js": float(losses.distribution_js.detach().cpu()),
                }
            )
            if steps >= max_steps:
                break
    return history


def _normalized_batch_to_original_pixels(
    points_normalized: Tensor,
    original_sizes_hw: Tensor,
    *,
    align_corners: bool,
) -> Tensor:
    if original_sizes_hw.ndim != 2 or original_sizes_hw.shape != (points_normalized.shape[0], 2):
        raise ValueError("original_size_hw must have shape [B,2]")
    converted = []
    for points, size in zip(points_normalized, original_sizes_hw, strict=True):
        size_hw = (int(size[0].item()), int(size[1].item()))
        converted.append(normalized_to_pixel(points, size_hw, align_corners=align_corners))
    return torch.stack(converted, dim=0)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    *,
    dsnt: DSNT,
    device: torch.device,
    config: SupervisedTrainingConfig,
    decoder: str = "dsnt",
) -> dict[str, float | int | str]:
    """Evaluate losses, original-512-style MRE, and unsigned AoP MAE."""

    model.eval()
    accumulator = _loss_accumulator()
    predicted_original: list[Tensor] = []
    target_original: list[Tensor] = []
    validity: list[Tensor] = []
    for raw_batch in data_loader:
        batch = _to_device(raw_batch, device)
        logits = model(batch["image"])
        losses = compute_supervised_losses(
            logits,
            batch["heatmaps"],
            batch["points_normalized"],
            batch["valid_mask"],
            dsnt=dsnt,
            heatmap_weight=config.heatmap_loss_weight,
            coordinate_weight=config.coordinate_loss_weight,
            distribution_weight=config.distribution_loss_weight,
        )
        batch_size = int(batch["image"].shape[0])
        _add_losses(accumulator, losses, batch_size)
        normalized = decode_heatmaps(
            logits,
            method=decoder,
            dsnt=dsnt,
            align_corners=config.align_corners,
        )
        predicted_px = _normalized_batch_to_original_pixels(
            normalized,
            batch["original_size_hw"],
            align_corners=config.align_corners,
        )
        predicted_original.append(predicted_px.detach().cpu())
        target_original.append(batch["points_original_px"].detach().cpu())
        validity.append(batch["valid_mask"].detach().cpu().to(dtype=torch.bool))

    losses_mean = _mean_losses(accumulator)
    predicted = torch.cat(predicted_original, dim=0)
    target = torch.cat(target_original, dim=0)
    valid_mask = torch.cat(validity, dim=0)
    metrics: dict[str, float | int] = {
        **losses_mean,
        **summarize_keypoint_metrics(
            predicted,
            target,
            keypoint_names=config.keypoint_order,
            valid_mask=valid_mask,
        ),
        "n_samples": int(predicted.shape[0]),
        "decoder": decoder,
    }

    predicted_aop, predicted_aop_valid = compute_aop(
        predicted,
        vertex_index=config.aop_vertex_index,
        pubic_axis_other_index=config.aop_pubic_axis_other_index,
        fetal_head_index=config.aop_fetal_head_index,
        output_unit="degrees",
        invalid="mask",
    )
    target_aop, target_aop_valid = compute_aop(
        target,
        vertex_index=config.aop_vertex_index,
        pubic_axis_other_index=config.aop_pubic_axis_other_index,
        fetal_head_index=config.aop_fetal_head_index,
        output_unit="degrees",
        invalid="mask",
    )
    required_indices = [
        config.aop_vertex_index,
        config.aop_pubic_axis_other_index,
        config.aop_fetal_head_index,
    ]
    evaluable_aop = target_aop_valid & valid_mask[:, required_indices].all(dim=1)
    angle_valid = predicted_aop_valid & evaluable_aop
    angle_errors = absolute_angle_error(predicted_aop, target_aop)
    finite_errors = angle_errors[angle_valid & torch.isfinite(angle_errors)]
    metrics["n_valid_aop"] = int(finite_errors.numel())
    metrics["n_evaluable_aop"] = int(evaluable_aop.sum().item())
    metrics["aop_invalid_prediction_count"] = int(
        (evaluable_aop & ~predicted_aop_valid).sum().item()
    )
    metrics["aop_mae_valid_deg"] = (
        float(finite_errors.mean().item()) if finite_errors.numel() else float("nan")
    )
    penalized_errors = torch.where(
        predicted_aop_valid,
        angle_errors,
        torch.full_like(angle_errors, 180.0),
    )
    evaluable_errors = penalized_errors[evaluable_aop & torch.isfinite(penalized_errors)]
    metrics["aop_mae_deg"] = (
        float(evaluable_errors.mean().item()) if evaluable_errors.numel() else float("nan")
    )
    return metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_history_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty training history")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_supervised(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    *,
    dsnt: DSNT,
    device: torch.device,
    config: SupervisedTrainingConfig,
    output_dir: str | Path,
    checkpoint_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train, select on validation only, and write best/last checkpoints."""

    config.validate()
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", checkpoint_config)
    history: list[dict[str, Any]] = []
    best_value = math.inf
    best_selection_key = (math.inf, math.inf, math.inf)
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
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
        train_time_sec = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        validation_time_sec = time.perf_counter() - validation_started
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_time_sec": train_time_sec,
            "validation_time_sec": validation_time_sec,
            "epoch_time_sec": time.perf_counter() - epoch_started,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        print(json.dumps(_json_safe(row), ensure_ascii=False), flush=True)
        metric_value = float(validation_metrics[config.checkpoint_metric])
        mre_value = float(validation_metrics["MRE_ALL"])
        if not math.isfinite(metric_value):
            raise FloatingPointError(
                f"Validation checkpoint metric {config.checkpoint_metric} is NaN or Inf"
            )
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=checkpoint_config,
            seed=config.seed,
            metrics=validation_metrics,
        )
        selection_key = (metric_value, mre_value, float(epoch))
        if selection_key < best_selection_key:
            best_value = metric_value
            best_selection_key = selection_key
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=checkpoint_config,
                seed=config.seed,
                metrics=validation_metrics,
            )
        write_history_csv(run_dir / "train_log.csv", history)

    summary = {
        "status": "completed",
        "selection_split": "validation",
        "checkpoint_metric": config.checkpoint_metric,
        "selection_tiebreak": [config.checkpoint_metric, "MRE_ALL", "earlier_epoch"],
        "best_epoch": best_epoch,
        "best_value": best_value,
        "best_validation_metrics": best_metrics,
        "last_validation_metrics": {
            key.removeprefix("val_"): value
            for key, value in history[-1].items()
            if key.startswith("val_")
        },
        "best_checkpoint": str((run_dir / "best.pt").resolve()),
        "last_checkpoint": str((run_dir / "last.pt").resolve()),
    }
    write_json(run_dir / "metrics.json", summary)
    return summary
