"""Fail-closed configuration schema for the Phase 1A HRNet shared probe."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml
from torch import Tensor
from torch.optim import Adam

from geoequi_ld.models.hrnet import PINNED_TIMM_VERSION

_ConfigT = TypeVar("_ConfigT")


def _strict_values(
    cls: type[_ConfigT],
    values: Mapping[str, Any],
    *,
    tuple_fields: tuple[str, ...] = (),
    class_key: bool = False,
) -> dict[str, Any]:
    converted = dict(values)
    if class_key:
        if "class" not in converted:
            raise ValueError(f"{cls.__name__} requires an explicit 'class' field")
        if "class_name" in converted:
            raise ValueError("Use only the serialized 'class' field")
        converted["class_name"] = converted.pop("class")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(converted) - allowed)
    missing = sorted(allowed - set(converted))
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    if missing:
        raise ValueError(f"Missing {cls.__name__} fields: {missing}")
    for name in tuple_fields:
        converted[name] = tuple(converted[name])
    return converted


@dataclass(frozen=True)
class Phase1ATrainingConfig:
    seed: int
    device: str
    deterministic: bool
    input_size_hw: tuple[int, int]
    heatmap_size_hw: tuple[int, int]
    sigma_heatmap_px: float
    align_corners: bool
    dsnt_temperature: float
    keypoint_order: tuple[str, str, str]
    aop_vertex_index: int
    aop_pubic_axis_other_index: int
    aop_fetal_head_index: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    heatmap_loss_weight: float
    coordinate_loss_weight: float
    distribution_loss_weight: float
    max_grad_norm: float
    num_workers: int
    checkpoint_metric: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1ATrainingConfig:
        converted = _strict_values(
            cls,
            values,
            tuple_fields=("input_size_hw", "heatmap_size_hw", "keypoint_order"),
        )
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        expected: dict[str, Any] = {
            "seed": 42,
            "device": "auto",
            "deterministic": True,
            "input_size_hw": (512, 512),
            "heatmap_size_hw": (256, 256),
            "sigma_heatmap_px": 4.0,
            "align_corners": True,
            "dsnt_temperature": 0.05,
            "keypoint_order": ("PS1", "PS2", "FH1"),
            "aop_vertex_index": 0,
            "aop_pubic_axis_other_index": 1,
            "aop_fetal_head_index": 2,
            "batch_size": 1,
            "epochs": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "heatmap_loss_weight": 1.0,
            "coordinate_loss_weight": 10.0,
            "distribution_loss_weight": 1.0,
            "max_grad_norm": 5.0,
            "num_workers": 0,
            "checkpoint_metric": "aop_mae_deg",
        }
        actual = asdict(self)
        drift = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        if drift:
            raise ValueError(f"Phase 1A training protocol drifted: {drift}")


@dataclass(frozen=True)
class Phase1AModelConfig:
    class_name: str
    backbone: str
    timm_version: str
    pretrained: bool
    in_channels: int
    out_channels: int
    feature_location: str
    out_indices: tuple[int, ...]
    feature_channels: int
    feature_reduction: int
    decoder_channels: tuple[int, int]
    decoder_normalization: str
    decoder_activation: str
    interpolation_mode: str
    align_corners: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1AModelConfig:
        converted = _strict_values(
            cls,
            values,
            tuple_fields=("out_indices", "decoder_channels"),
            class_key=True,
        )
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        expected: dict[str, Any] = {
            "class_name": "HRNetW32SharedHeatmap",
            "backbone": "hrnet_w32",
            "timm_version": PINNED_TIMM_VERSION,
            "pretrained": False,
            "in_channels": 1,
            "out_channels": 3,
            "feature_location": "",
            "out_indices": (1,),
            "feature_channels": 32,
            "feature_reduction": 4,
            "decoder_channels": (32, 16),
            "decoder_normalization": "BatchNorm2d",
            "decoder_activation": "GELU",
            "interpolation_mode": "bilinear",
            "align_corners": True,
        }
        actual = asdict(self)
        drift = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        if drift:
            raise ValueError(f"Phase 1A HRNet model contract drifted: {drift}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class"] = payload.pop("class_name")
        return payload


@dataclass(frozen=True)
class Phase1AOptimizerConfig:
    class_name: str
    betas: tuple[float, float]
    eps: float
    amsgrad: bool
    foreach: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1AOptimizerConfig:
        converted = _strict_values(
            cls,
            values,
            tuple_fields=("betas",),
            class_key=True,
        )
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        if (
            self.class_name != "Adam"
            or self.betas != (0.9, 0.999)
            or self.eps != 1.0e-8
            or self.amsgrad
            or self.foreach
        ):
            raise ValueError(
                "Phase 1A requires Adam(betas=(0.9,0.999), eps=1e-8, "
                "amsgrad=False, foreach=False)"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class"] = payload.pop("class_name")
        return payload


@dataclass(frozen=True)
class Phase1AResourceConfig:
    precision: str
    amp_enabled: bool
    allow_input_resize: bool
    require_full_input_first_step_probe: bool
    tiny_max_steps: int
    tiny_max_seconds: int
    formal_max_seconds: int
    total_gpu_max_seconds: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1AResourceConfig:
        config = cls(**_strict_values(cls, values))
        config.validate()
        return config

    def validate(self) -> None:
        expected = {
            "precision": "float32",
            "amp_enabled": False,
            "allow_input_resize": False,
            "require_full_input_first_step_probe": True,
            "tiny_max_steps": 500,
            "tiny_max_seconds": 2400,
            "formal_max_seconds": 7200,
            "total_gpu_max_seconds": 10800,
        }
        actual = asdict(self)
        if actual != expected:
            raise ValueError(f"Phase 1A resource contract drifted: {actual}")


@dataclass(frozen=True)
class Phase1AHRNetConfig:
    schema_version: int
    experiment_name: str
    testing_frozen: bool
    training: Phase1ATrainingConfig
    model: Phase1AModelConfig
    optimizer: Phase1AOptimizerConfig
    resources: Phase1AResourceConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1AHRNetConfig:
        allowed = {
            "schema_version",
            "experiment_name",
            "testing_frozen",
            "training",
            "model",
            "optimizer",
            "resources",
        }
        unknown = sorted(set(values) - allowed)
        missing = sorted(allowed - set(values))
        if unknown or missing:
            raise ValueError(
                f"Phase 1A top-level fields invalid: missing={missing}, unknown={unknown}"
            )
        for section in ("training", "model", "optimizer", "resources"):
            if not isinstance(values[section], Mapping):
                raise ValueError(f"Phase 1A '{section}' section must be a mapping")
        config = cls(
            schema_version=int(values["schema_version"]),
            experiment_name=str(values["experiment_name"]),
            testing_frozen=bool(values["testing_frozen"]),
            training=Phase1ATrainingConfig.from_mapping(values["training"]),
            model=Phase1AModelConfig.from_mapping(values["model"]),
            optimizer=Phase1AOptimizerConfig.from_mapping(values["optimizer"]),
            resources=Phase1AResourceConfig.from_mapping(values["resources"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Phase 1A config schema_version must be 1")
        if self.experiment_name != "H1_shared_B2_seed42_20e":
            raise ValueError("Unexpected Phase 1A experiment_name")
        if not self.testing_frozen:
            raise PermissionError("Phase 1A must keep testing frozen")
        if self.training.align_corners != self.model.align_corners:
            raise ValueError("Training and model align_corners settings differ")


def load_phase1a_hrnet_config(path: str | Path) -> Phase1AHRNetConfig:
    """Load and validate the complete, path-free Phase 1A HRNet protocol."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 1A configuration does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    loaded = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError("Phase 1A configuration must be a mapping")
    return Phase1AHRNetConfig.from_mapping(loaded)


def build_phase1a_adam(
    parameters: Iterable[Tensor],
    config: Phase1AHRNetConfig,
) -> Adam:
    """Build the recorded low-peak-memory Adam implementation without AMP."""

    config.validate()
    return Adam(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        amsgrad=config.optimizer.amsgrad,
        foreach=config.optimizer.foreach,
    )
