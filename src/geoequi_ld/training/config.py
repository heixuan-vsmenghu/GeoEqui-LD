"""Configuration for the Phase 0 supervised training loop."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    """Serializable settings for the minimal supervised baseline.

    Data paths intentionally do not live here.  Scripts add them to the run
    snapshot so checkpoints remain portable and this module never embeds a
    machine-specific absolute path.
    """

    seed: int = 42
    device: str = "auto"
    deterministic: bool = True
    input_size_hw: tuple[int, int] = (512, 512)
    heatmap_size_hw: tuple[int, int] = (256, 256)
    sigma_heatmap_px: float = 4.0
    align_corners: bool = True
    dsnt_temperature: float = 0.05
    keypoint_order: tuple[str, str, str] = ("PS1", "PS2", "FH1")
    aop_vertex_index: int = 0
    aop_pubic_axis_other_index: int = 1
    aop_fetal_head_index: int = 2
    base_channels: int = 16
    batch_size: int = 4
    epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    heatmap_loss_weight: float = 1.0
    coordinate_loss_weight: float = 10.0
    distribution_loss_weight: float = 1.0
    max_grad_norm: float | None = None
    num_workers: int = 0
    checkpoint_metric: str = "aop_mae_deg"

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if len(self.input_size_hw) != 2 or min(self.input_size_hw) <= 0:
            raise ValueError("input_size_hw must contain two positive integers")
        if len(self.heatmap_size_hw) != 2 or min(self.heatmap_size_hw) <= 0:
            raise ValueError("heatmap_size_hw must contain two positive integers")
        expected_heatmap = tuple(size // 2 for size in self.input_size_hw)
        if tuple(self.heatmap_size_hw) != expected_heatmap:
            raise ValueError(
                "HeatmapUNet produces half-resolution heatmaps; expected "
                f"{expected_heatmap}, got {self.heatmap_size_hw}"
            )
        if any(size % 16 for size in self.input_size_hw):
            raise ValueError("input_size_hw dimensions must be divisible by 16")
        if self.sigma_heatmap_px <= 0:
            raise ValueError("sigma_heatmap_px must be positive")
        if self.dsnt_temperature <= 0:
            raise ValueError("dsnt_temperature must be positive")
        if len(self.keypoint_order) != 3 or len(set(self.keypoint_order)) != 3:
            raise ValueError("Phase 0 requires three unique keypoint names")
        aop_indices = (
            self.aop_vertex_index,
            self.aop_pubic_axis_other_index,
            self.aop_fetal_head_index,
        )
        if sorted(aop_indices) != [0, 1, 2]:
            raise ValueError("AoP indices must be a permutation of 0, 1, 2")
        if self.base_channels <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("base_channels, batch_size, and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if (
            self.heatmap_loss_weight < 0
            or self.coordinate_loss_weight < 0
            or self.distribution_loss_weight < 0
        ):
            raise ValueError("loss weights must be non-negative")
        if (
            self.heatmap_loss_weight == 0
            and self.coordinate_loss_weight == 0
            and self.distribution_loss_weight == 0
        ):
            raise ValueError("at least one supervised loss weight must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.checkpoint_metric not in {"aop_mae_deg", "MRE_ALL"}:
            raise ValueError("checkpoint_metric must be 'aop_mae_deg' or 'MRE_ALL'")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SupervisedTrainingConfig:
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown training configuration fields: {unknown}")
        converted = dict(values)
        for name in ("input_size_hw", "heatmap_size_hw", "keypoint_order"):
            if name in converted:
                converted[name] = tuple(converted[name])
        config = cls(**converted)
        config.validate()
        return config


def load_training_config(path: str | Path) -> SupervisedTrainingConfig:
    """Load JSON/YAML, accepting either a flat mapping or ``training:`` block."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError("Training configuration must be a mapping")
    values = loaded.get("training", loaded)
    if not isinstance(values, Mapping):
        raise ValueError("The 'training' configuration value must be a mapping")
    return SupervisedTrainingConfig.from_mapping(values)
