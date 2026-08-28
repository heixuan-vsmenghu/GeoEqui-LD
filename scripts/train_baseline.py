#!/usr/bin/env python
"""Train the minimal supervised U-Net + DSNT coordinate baseline."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.optim import Adam
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.data.dataset import IUGCLabeledDataset  # noqa: E402
from geoequi_ld.geometry.coordinates import normalized_to_pixel  # noqa: E402
from geoequi_ld.models.dsnt import DSNT, spatial_softmax  # noqa: E402
from geoequi_ld.models.unet import HeatmapUNet, count_trainable_parameters  # noqa: E402
from geoequi_ld.training.checkpoints import restore_checkpoint  # noqa: E402
from geoequi_ld.training.config import (  # noqa: E402
    SupervisedTrainingConfig,
    load_training_config,
)
from geoequi_ld.training.engine import fit_supervised  # noqa: E402
from geoequi_ld.training.runtime import (  # noqa: E402
    make_generator,
    resolve_device,
    seed_data_loader_worker,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    description = (
        "Train the Phase 0 supervised heatmap/coordinate baseline. "
        "Test data is never accepted here."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML training configuration")
    parser.add_argument("--train-fh1-column", default="FH1")
    parser.add_argument("--val-fh1-column", default="FH1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--heatmap-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--sigma", type=float, dest="sigma_heatmap_px")
    parser.add_argument("--temperature", type=float, dest="dsnt_temperature")
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--heatmap-loss-weight", type=float)
    parser.add_argument("--coordinate-loss-weight", type=float)
    parser.add_argument("--distribution-loss-weight", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--checkpoint-metric", choices=("aop_mae_deg", "MRE_ALL"))
    return parser


def config_from_args(args: argparse.Namespace) -> SupervisedTrainingConfig:
    config = load_training_config(args.config) if args.config else SupervisedTrainingConfig()
    updates: dict[str, Any] = {}
    direct_fields = (
        "seed",
        "device",
        "deterministic",
        "sigma_heatmap_px",
        "dsnt_temperature",
        "base_channels",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "heatmap_loss_weight",
        "coordinate_loss_weight",
        "distribution_loss_weight",
        "max_grad_norm",
        "num_workers",
        "checkpoint_metric",
    )
    for name in direct_fields:
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


def _loader(
    dataset: IUGCLabeledDataset,
    *,
    config: SupervisedTrainingConfig,
    shuffle: bool,
    seed_offset: int,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_loader_worker if config.num_workers else None,
        generator=make_generator(config.seed + seed_offset),
        persistent_workers=config.num_workers > 0,
    )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _environment_text(device: torch.device) -> str:
    lines = [
        f"platform={platform.platform()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"torch={torch.__version__}",
        f"resolved_device={device}",
        f"cuda_available={torch.cuda.is_available()}",
        f"torch_cuda={torch.version.cuda}",
        f"cudnn={torch.backends.cudnn.version()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"gpu={torch.cuda.get_device_name(device)}")
    return "\n".join(lines) + "\n"


def _save_curves(run_dir: Path) -> None:
    with (run_dir / "train_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    curves_dir = run_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4), dpi=150)
    for key in ("train_total_loss", "val_total_loss", "val_distribution_js"):
        axis.plot(epochs, [float(row[key]) for row in rows], label=key)
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(curves_dir / "losses.png")
    plt.close(figure)

    figure, left = plt.subplots(figsize=(7, 4), dpi=150)
    right = left.twinx()
    left.plot(epochs, [float(row["val_MRE_ALL"]) for row in rows], color="#0072B2", label="MRE_ALL")
    right.plot(
        epochs, [float(row["val_aop_mae_deg"]) for row in rows], color="#D55E00", label="AoP MAE"
    )
    left.set_xlabel("epoch")
    left.set_ylabel("MRE (px)", color="#0072B2")
    right.set_ylabel("AoP MAE (deg)", color="#D55E00")
    left.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(curves_dir / "validation_metrics.png")
    plt.close(figure)


@torch.inference_mode()
def _save_validation_predictions(
    model: torch.nn.Module,
    dsnt: DSNT,
    data_loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    config: SupervisedTrainingConfig,
    output_dir: Path,
    count: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#00E5FF", "#FFB000", "#FF4DA6")
    model.eval()
    saved = 0
    for batch in data_loader:
        images = batch["image"].to(device)
        logits = model(images)
        probabilities = spatial_softmax(logits, temperature=dsnt.temperature)
        predicted = normalized_to_pixel(
            dsnt(logits),
            config.input_size_hw,
            align_corners=config.align_corners,
        ).cpu()
        for batch_index in range(images.shape[0]):
            figure, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=140)
            axes[0].imshow(images[batch_index, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
            for keypoint_index, (name, color) in enumerate(
                zip(config.keypoint_order, colors, strict=True)
            ):
                target_xy = batch["points_input_px"][batch_index, keypoint_index]
                predicted_xy = predicted[batch_index, keypoint_index]
                axes[0].scatter(float(target_xy[0]), float(target_xy[1]), c=color, s=50, marker="o")
                axes[0].scatter(
                    float(predicted_xy[0]), float(predicted_xy[1]), c=color, s=65, marker="x"
                )
                axes[0].text(
                    float(target_xy[0]) + 5, float(target_xy[1]) - 5, name, color=color, fontsize=8
                )
            axes[0].set_title("circle=target, x=prediction")
            axes[0].set_axis_off()
            for keypoint_index, name in enumerate(config.keypoint_order):
                axes[keypoint_index + 1].imshow(
                    probabilities[batch_index, keypoint_index].cpu(),
                    cmap="magma",
                )
                axes[keypoint_index + 1].set_title(f"{name} probability")
                axes[keypoint_index + 1].set_axis_off()
            figure.suptitle(
                "RESTRICTED VALIDATION DIAGNOSTIC — DO NOT COMMIT", color="#A00000", weight="bold"
            )
            figure.tight_layout()
            figure.savefig(output_dir / f"sample_{saved:02d}.png", bbox_inches="tight")
            plt.close(figure)
            saved += 1
            if saved >= count:
                (output_dir / "DO_NOT_COMMIT.txt").write_text(
                    "Restricted validation-data derivative. Do not commit or upload.\n",
                    encoding="utf-8",
                )
                return


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    seed_everything(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    train_columns = {"PS1": "PS1", "PS2": "PS2", "FH1": args.train_fh1_column}
    val_columns = {"PS1": "PS1", "PS2": "PS2", "FH1": args.val_fh1_column}
    common_dataset = {
        "keypoint_order": config.keypoint_order,
        "input_size_hw": config.input_size_hw,
        "heatmap_size_hw": config.heatmap_size_hw,
        "sigma": config.sigma_heatmap_px,
        "align_corners": config.align_corners,
    }
    train_dataset = IUGCLabeledDataset(
        image_dir=args.train_images,
        labels_csv=args.train_labels,
        source_columns=train_columns,
        **common_dataset,
    )
    validation_dataset = IUGCLabeledDataset(
        image_dir=args.val_images,
        labels_csv=args.val_labels,
        source_columns=val_columns,
        **common_dataset,
    )
    train_loader = _loader(train_dataset, config=config, shuffle=True, seed_offset=0, device=device)
    validation_loader = _loader(
        validation_dataset,
        config=config,
        shuffle=False,
        seed_offset=1,
        device=device,
    )

    model = HeatmapUNet(base_channels=config.base_channels).to(device)
    dsnt = DSNT(
        temperature=config.dsnt_temperature,
        align_corners=config.align_corners,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    run_config = {
        "phase": "phase0_supervised_baseline",
        "training": config.to_dict(),
        "data": {
            "train_images": str(args.train_images.resolve()),
            "train_labels": str(args.train_labels.resolve()),
            "validation_images": str(args.val_images.resolve()),
            "validation_labels": str(args.val_labels.resolve()),
            "train_source_columns": train_columns,
            "validation_source_columns": val_columns,
        },
        "model": {
            "class": "HeatmapUNet",
            "trainable_parameters": count_trainable_parameters(model),
        },
        "runtime": {
            "resolved_device": str(device),
            "torch_version": torch.__version__,
            "git_commit": _git_commit(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (args.output_dir / "environment.txt").write_text(
        _environment_text(device),
        encoding="utf-8",
    )
    summary = fit_supervised(
        model,
        train_loader,
        validation_loader,
        optimizer,
        dsnt=dsnt,
        device=device,
        config=config,
        output_dir=args.output_dir,
        checkpoint_config=run_config,
    )
    restore_checkpoint(summary["best_checkpoint"], model=model, map_location=device)
    _save_curves(args.output_dir)
    _save_validation_predictions(
        model,
        dsnt,
        validation_loader,
        device=device,
        config=config,
        output_dir=args.output_dir / "predictions",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
