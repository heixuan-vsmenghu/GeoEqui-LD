#!/usr/bin/env python
"""Evaluate a frozen Phase 0 checkpoint on one explicitly supplied split."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.data.dataset import IUGCLabeledDataset  # noqa: E402
from geoequi_ld.models.dsnt import DSNT  # noqa: E402
from geoequi_ld.models.unet import HeatmapUNet  # noqa: E402
from geoequi_ld.training.checkpoints import read_checkpoint, restore_checkpoint  # noqa: E402
from geoequi_ld.training.config import SupervisedTrainingConfig  # noqa: E402
from geoequi_ld.training.engine import evaluate_model, write_json  # noqa: E402
from geoequi_ld.training.runtime import resolve_device, seed_everything  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen supervised checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--split-name",
        required=True,
        help="For provenance only, e.g. validation or testing",
    )
    parser.add_argument("--fh1-column", default="FH1")
    parser.add_argument("--device", help="Override checkpoint device request")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", type=Path, help="Optional metrics JSON path")
    return parser


def _training_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    training = config.get("training", config)
    if not isinstance(training, Mapping):
        raise ValueError("Checkpoint training config must be a mapping")
    return training


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = read_checkpoint(args.checkpoint, map_location="cpu")
    config = SupervisedTrainingConfig.from_mapping(_training_mapping(payload))
    updates: dict[str, Any] = {}
    if args.device is not None:
        updates["device"] = args.device
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.num_workers is not None:
        updates["num_workers"] = args.num_workers
    config = replace(config, **updates)
    config.validate()
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
    loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    model = HeatmapUNet(base_channels=config.base_channels).to(device)
    restore_checkpoint(args.checkpoint, model=model, map_location="cpu")
    dsnt = DSNT(temperature=config.dsnt_temperature, align_corners=config.align_corners).to(device)
    metrics = evaluate_model(model, loader, dsnt=dsnt, device=device, config=config)
    result = {
        "split": args.split_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_seed": int(payload["seed"]),
        "metrics": metrics,
    }
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
