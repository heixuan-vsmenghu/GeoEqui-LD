"""Strict protocol configuration for the Phase 1B decoder control."""

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
from geoequi_ld.training.phase1a_config import (
    Phase1AOptimizerConfig,
    Phase1ATrainingConfig,
)

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
        if "class" not in converted or "class_name" in converted:
            raise ValueError(f"{cls.__name__} requires only the serialized 'class' field")
        converted["class_name"] = converted.pop("class")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(converted) - allowed)
    missing = sorted(allowed - set(converted))
    if unknown or missing:
        raise ValueError(
            f"Invalid {cls.__name__} fields: missing={missing}, unknown={unknown}"
        )
    for name in tuple_fields:
        converted[name] = tuple(converted[name])
    return converted


@dataclass(frozen=True)
class Phase1BSplitModelConfig:
    class_name: str
    shared_initialization_class: str
    shared_initialization_method: str
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
    ps_out_channels: int
    fh_out_channels: int
    concatenation_order: tuple[str, str, str]
    interpolation_mode: str
    align_corners: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1BSplitModelConfig:
        config = cls(
            **_strict_values(
                cls,
                values,
                tuple_fields=("out_indices", "decoder_channels", "concatenation_order"),
                class_key=True,
            )
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected: dict[str, Any] = {
            "class_name": "HRNetW32SplitHeatmap",
            "shared_initialization_class": "HRNetW32SharedHeatmap",
            "shared_initialization_method": "from_shared",
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
            "ps_out_channels": 2,
            "fh_out_channels": 1,
            "concatenation_order": ("PS1", "PS2", "FH1"),
            "interpolation_mode": "bilinear",
            "align_corners": True,
        }
        actual = asdict(self)
        drift = {
            name: {"expected": expected_value, "actual": actual[name]}
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        }
        if drift:
            raise ValueError(f"Phase 1B split-model contract drifted: {drift}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class"] = payload.pop("class_name")
        return payload


@dataclass(frozen=True)
class Phase1BResourceConfig:
    precision: str
    amp_enabled: bool
    tiny_max_steps: int
    tiny_max_seconds: int
    replay_max_seconds: int
    formal_max_seconds: int
    total_gpu_max_seconds: int
    closing_reserve_seconds: int
    milestone_epochs: tuple[int, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1BResourceConfig:
        config = cls(
            **_strict_values(cls, values, tuple_fields=("milestone_epochs",))
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected = {
            "precision": "float32",
            "amp_enabled": False,
            "tiny_max_steps": 500,
            "tiny_max_seconds": 1200,
            "replay_max_seconds": 600,
            "formal_max_seconds": 7200,
            "total_gpu_max_seconds": 10800,
            "closing_reserve_seconds": 120,
            "milestone_epochs": (1, 3, 5, 10, 20),
        }
        actual = asdict(self)
        if actual != expected:
            raise ValueError(f"Phase 1B resource contract drifted: {actual}")


@dataclass(frozen=True)
class Phase1BDecoderControlConfig:
    schema_version: int
    experiment_name: str
    testing_frozen: bool
    training: Phase1ATrainingConfig
    model: Phase1BSplitModelConfig
    optimizer: Phase1AOptimizerConfig
    resources: Phase1BResourceConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1BDecoderControlConfig:
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
                f"Phase 1B top-level fields invalid: missing={missing}, unknown={unknown}"
            )
        for section in ("training", "model", "optimizer", "resources"):
            if not isinstance(values[section], Mapping):
                raise ValueError(f"Phase 1B '{section}' section must be a mapping")
        config = cls(
            schema_version=int(values["schema_version"]),
            experiment_name=str(values["experiment_name"]),
            testing_frozen=bool(values["testing_frozen"]),
            training=Phase1ATrainingConfig.from_mapping(values["training"]),
            model=Phase1BSplitModelConfig.from_mapping(values["model"]),
            optimizer=Phase1AOptimizerConfig.from_mapping(values["optimizer"]),
            resources=Phase1BResourceConfig.from_mapping(values["resources"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Phase 1B config schema_version must be 1")
        if self.experiment_name != "H2_split_B2_seed42_20e":
            raise ValueError("Unexpected Phase 1B experiment_name")
        if not self.testing_frozen:
            raise PermissionError("Phase 1B must keep testing frozen")
        if self.training.align_corners != self.model.align_corners:
            raise ValueError("Training and model align_corners settings differ")
        if self.training.keypoint_order != self.model.concatenation_order:
            raise ValueError("Training and split-decoder keypoint order differ")


def load_phase1b_decoder_config(path: str | Path) -> Phase1BDecoderControlConfig:
    """Load the path-free, fail-closed Phase 1B protocol."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 1B configuration does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    loaded = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError("Phase 1B configuration must be a mapping")
    return Phase1BDecoderControlConfig.from_mapping(loaded)


def build_phase1b_adam(
    parameters: Iterable[Tensor],
    config: Phase1BDecoderControlConfig,
) -> Adam:
    """Build the locked Adam implementation for the decoder control."""

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
