#!/usr/bin/env python
"""Overfit a deterministic 4--8 sample subset before formal training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.data.dataset import IUGCLabeledDataset  # noqa: E402
from geoequi_ld.geometry.coordinates import normalized_to_pixel  # noqa: E402
from geoequi_ld.models.dsnt import DSNT, spatial_softmax  # noqa: E402
from geoequi_ld.models.unet import HeatmapUNet, count_trainable_parameters  # noqa: E402
from geoequi_ld.training.checkpoints import save_checkpoint  # noqa: E402
from geoequi_ld.training.config import (  # noqa: E402
    SupervisedTrainingConfig,
    load_training_config,
)
from geoequi_ld.training.engine import (  # noqa: E402
    evaluate_model,
    train_for_steps,
    write_history_csv,
    write_json,
)
from geoequi_ld.training.runtime import (  # noqa: E402
    make_generator,
    resolve_device,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 0 tiny-sample overfit gate")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fh1-column", default="FH1")
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of unique samples (recommended 4--8)",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--heatmap-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--sigma", type=float, dest="sigma_heatmap_px")
    parser.add_argument("--temperature", type=float, dest="dsnt_temperature")
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--heatmap-loss-weight", type=float)
    parser.add_argument("--coordinate-loss-weight", type=float)
    parser.add_argument("--distribution-loss-weight", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    return parser


def _config(args: argparse.Namespace) -> SupervisedTrainingConfig:
    config = load_training_config(args.config) if args.config else SupervisedTrainingConfig()
    updates: dict[str, Any] = {"num_workers": 0}
    for name in (
        "seed",
        "device",
        "sigma_heatmap_px",
        "dsnt_temperature",
        "base_channels",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "heatmap_loss_weight",
        "coordinate_loss_weight",
        "distribution_loss_weight",
        "max_grad_norm",
    ):
        value = getattr(args, name)
        if value is not None:
            updates[name] = value
    if args.input_size is not None:
        updates["input_size_hw"] = tuple(args.input_size)
    if args.heatmap_size is not None:
        updates["heatmap_size_hw"] = tuple(args.heatmap_size)
    config = replace(config, **updates)
    config.validate()
    return config


def _parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


@torch.inference_mode()
def _save_prediction_visualizations(
    model: torch.nn.Module,
    dsnt: DSNT,
    tiny_dataset: Subset[Any],
    *,
    device: torch.device,
    config: SupervisedTrainingConfig,
    output_dir: Path,
) -> None:
    """Save restricted local overlays and predicted heatmaps for diagnosis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    loader = DataLoader(tiny_dataset, batch_size=1, shuffle=False, num_workers=0)
    colors = ("#00E5FF", "#FFB000", "#FF4DA6")
    for sample_number, batch in enumerate(loader):
        image = batch["image"].to(device)
        logits = model(image)
        probabilities = spatial_softmax(logits, temperature=dsnt.temperature)
        predicted_normalized = dsnt(logits)
        predicted_input = normalized_to_pixel(
            predicted_normalized,
            config.input_size_hw,
            align_corners=config.align_corners,
        )[0].cpu()
        target_input = batch["points_input_px"][0]

        figure, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=140)
        axes[0].imshow(image[0, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        for keypoint_index, (name, color) in enumerate(
            zip(config.keypoint_order, colors, strict=True)
        ):
            tx, ty = target_input[keypoint_index]
            px, py = predicted_input[keypoint_index]
            axes[0].scatter(float(tx), float(ty), c=color, s=55, marker="o", edgecolors="black")
            axes[0].scatter(float(px), float(py), c=color, s=70, marker="x", linewidths=2.0)
            axes[0].text(float(tx) + 5, float(ty) - 5, name, color=color, fontsize=8)
        axes[0].set_title("circle=target, x=prediction")
        axes[0].set_axis_off()

        for keypoint_index, name in enumerate(config.keypoint_order):
            heatmap = probabilities[0, keypoint_index].detach().cpu()
            axes[keypoint_index + 1].imshow(heatmap, cmap="magma")
            axes[keypoint_index + 1].set_title(f"predicted {name} probability")
            axes[keypoint_index + 1].set_axis_off()
        figure.suptitle(
            "RESTRICTED LOCAL DIAGNOSTIC — DO NOT COMMIT", color="#A00000", weight="bold"
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"sample_{sample_number:02d}.png", bbox_inches="tight")
        plt.close(figure)

    (output_dir / "DO_NOT_COMMIT.txt").write_text(
        "Restricted medical-data derivative. Do not commit or upload without permission.\n",
        encoding="utf-8",
    )


def _tiny_report(summary: dict[str, Any]) -> str:
    initial = summary["initial_metrics"]
    final = summary["final_metrics"]
    possible_issues: list[str] = []
    if not summary["reached_target_reduction"]:
        possible_issues.append("损失下降不足：检查损失缩放、学习率或模型容量。")
    if not summary["parameters_updated"]:
        possible_issues.append("参数没有更新：检查优化器、反向传播和冻结状态。")
    if not summary["gradient_observed"]:
        possible_issues.append("没有观察到非零梯度：检查 DSNT 和损失链路。")
    if not summary["all_reported_metrics_finite"] or not summary["all_gradients_finite"]:
        possible_issues.append("出现 NaN/Inf：检查坐标边界、AoP 退化向量和学习率。")
    if float(final["MRE_ALL"]) >= 0.5 * float(initial["MRE_ALL"]):
        possible_issues.append("定位误差下降不明显：检查坐标系统、标签顺序和热图解码。")
    if not possible_issues:
        possible_issues.append("本次门槛未发现上述结构性异常。")

    return "\n".join(
        [
            "# Tiny-overfit diagnostic",
            "",
            "> This run uses restricted local data. Prediction images and checkpoints "
            "must not be committed.",
            "",
            f"- Gate: **{'PASS' if summary['gate_passed'] else 'FAIL'}**",
            f"- Samples / steps: {summary['sample_count']} / {summary['steps']}",
            f"- Loss: {float(initial['total_loss']):.6f} → {float(final['total_loss']):.6f} "
            f"({float(summary['loss_reduction_percent']):.2f}% reduction)",
            f"- MRE_ALL: {float(initial['MRE_ALL']):.3f} px → {float(final['MRE_ALL']):.3f} px",
            f"- AoP MAE: {float(initial['aop_mae_deg']):.3f}° → {float(final['aop_mae_deg']):.3f}°",
            f"- Parameters updated: {summary['parameters_updated']}",
            f"- Non-zero gradient observed: {summary['gradient_observed']}",
            f"- Finite metrics / gradients: {summary['all_reported_metrics_finite']} / "
            f"{summary['all_gradients_finite']}",
            "",
            "## Diagnostic interpretation",
            "",
            *(f"- {issue}" for issue in possible_issues),
            "",
            "Visual overlays and heatmaps are stored under `predictions/` in this "
            "local run directory.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    config = _config(args)
    seed_everything(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    source_columns = {"PS1": "PS1", "PS2": "PS2", "FH1": args.fh1_column}
    dataset = IUGCLabeledDataset(
        image_dir=args.images,
        labels_csv=args.labels,
        source_columns=source_columns,
        keypoint_order=config.keypoint_order,
        input_size_hw=config.input_size_hw,
        heatmap_size_hw=config.heatmap_size_hw,
        sigma=config.sigma_heatmap_px,
        align_corners=config.align_corners,
    )
    if args.samples > len(dataset):
        raise ValueError(
            f"Requested {args.samples} samples, but the dataset contains {len(dataset)}"
        )
    permutation = torch.randperm(len(dataset), generator=make_generator(config.seed))
    indices = permutation[: args.samples].tolist()
    tiny_dataset = Subset(dataset, indices)
    loader: DataLoader[dict[str, Any]] = DataLoader(
        tiny_dataset,
        batch_size=min(config.batch_size, args.samples),
        shuffle=True,
        num_workers=0,
        generator=make_generator(config.seed + 1),
        pin_memory=device.type == "cuda",
    )
    evaluation_loader: DataLoader[dict[str, Any]] = DataLoader(
        tiny_dataset,
        batch_size=min(config.batch_size, args.samples),
        shuffle=False,
        num_workers=0,
    )
    model = HeatmapUNet(base_channels=config.base_channels).to(device)
    dsnt = DSNT(temperature=config.dsnt_temperature, align_corners=config.align_corners).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run_config = {
        "phase": "phase0_tiny_overfit",
        "training": config.to_dict(),
        "data": {
            "images": str(args.images.resolve()),
            "labels": str(args.labels.resolve()),
            "source_columns": source_columns,
            "selected_indices": indices,
            "sample_count": args.samples,
        },
        "model": {
            "class": "HeatmapUNet",
            "trainable_parameters": count_trainable_parameters(model),
        },
        "max_steps": args.max_steps,
        "resolved_device": str(device),
    }
    initial_metrics = evaluate_model(
        model,
        evaluation_loader,
        dsnt=dsnt,
        device=device,
        config=config,
    )
    before = _parameter_snapshot(model)
    history = train_for_steps(
        model,
        loader,
        optimizer,
        dsnt=dsnt,
        device=device,
        config=config,
        max_steps=args.max_steps,
    )
    final_metrics = evaluate_model(
        model,
        evaluation_loader,
        dsnt=dsnt,
        device=device,
        config=config,
    )
    after = _parameter_snapshot(model)
    parameter_delta_l1 = sum(
        float((new - old).abs().sum()) for old, new in zip(before, after, strict=True)
    )
    initial_loss = float(initial_metrics["total_loss"])
    final_loss = float(final_metrics["total_loss"])
    reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    all_finite = all(
        not isinstance(value, float) or math.isfinite(value)
        for metrics in (initial_metrics, final_metrics)
        for value in metrics.values()
    )
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_l1 = sum(float(gradient.detach().abs().sum().cpu()) for gradient in gradients)
    all_gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    mre_improved = float(final_metrics["MRE_ALL"]) < 0.5 * float(initial_metrics["MRE_ALL"])
    summary = {
        "status": "completed",
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "loss_reduction_fraction": reduction,
        "loss_reduction_percent": 100.0 * reduction,
        "target_reduction_percent": 90.0,
        "reached_target_reduction": reduction >= 0.90,
        "parameter_delta_l1": parameter_delta_l1,
        "parameters_updated": parameter_delta_l1 > 0,
        "gradient_l1": gradient_l1,
        "gradient_observed": gradient_l1 > 0,
        "all_gradients_finite": all_gradients_finite,
        "all_reported_metrics_finite": all_finite,
        "steps": args.max_steps,
        "sample_count": args.samples,
        "sample_indices": indices,
    }
    summary["gate_passed"] = bool(
        summary["reached_target_reduction"]
        and summary["parameters_updated"]
        and summary["gradient_observed"]
        and summary["all_gradients_finite"]
        and summary["all_reported_metrics_finite"]
        and mre_improved
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", run_config)
    write_history_csv(args.output_dir / "train_log.csv", history)
    write_json(args.output_dir / "metrics.json", summary)
    _save_prediction_visualizations(
        model,
        dsnt,
        tiny_dataset,
        device=device,
        config=config,
        output_dir=args.output_dir / "predictions",
    )
    (args.output_dir / "TINY_OVERFIT_REPORT.md").write_text(
        _tiny_report(summary),
        encoding="utf-8",
    )
    save_checkpoint(
        args.output_dir / "tiny_overfit.pt",
        model=model,
        optimizer=optimizer,
        epoch=0,
        config=run_config,
        seed=config.seed,
        metrics=final_metrics,
        extra={"optimizer_steps": args.max_steps, "sample_indices": indices},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
