"""Fail-closed protocol configuration for Phase 1C specialized enhancers."""

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
from geoequi_ld.training.phase1a_config import Phase1AOptimizerConfig

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
class Phase1CTrainingConfig:
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
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1CTrainingConfig:
        config = cls(
            **_strict_values(
                cls,
                values,
                tuple_fields=("input_size_hw", "heatmap_size_hw", "keypoint_order"),
            )
        )
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
            "epochs": 16,
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
            name: {"expected": expected_value, "actual": actual[name]}
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        }
        if drift:
            raise ValueError(f"Phase 1C training contract drifted: {drift}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class Phase1CSpecializedModelConfig:
    class_name: str
    h2_initialization_class: str
    h3_initialization_method: str
    backbone: str
    timm_version: str
    torchvision_version: str
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
    concatenation_order: tuple[str, str, str]
    ps_deformable_operator: str
    ps_offset_channels: int
    ps_mask_channels: int
    ps_spatial_attention_channels: int
    ps_normalization: str
    fh_aspp_dilations: tuple[int, int, int]
    fh_se_hidden_channels: int
    fh_normalization: str
    interpolation_mode: str
    align_corners: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1CSpecializedModelConfig:
        config = cls(
            **_strict_values(
                cls,
                values,
                tuple_fields=(
                    "out_indices",
                    "decoder_channels",
                    "concatenation_order",
                    "fh_aspp_dilations",
                ),
                class_key=True,
            )
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected: dict[str, Any] = {
            "class_name": "HRNetW32SpecializedHeatmap",
            "h2_initialization_class": "HRNetW32SplitHeatmap",
            "h3_initialization_method": "from_split",
            "backbone": "hrnet_w32",
            "timm_version": PINNED_TIMM_VERSION,
            "torchvision_version": "0.20.1",
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
            "concatenation_order": ("PS1", "PS2", "FH1"),
            "ps_deformable_operator": "torchvision.ops.DeformConv2d",
            "ps_offset_channels": 18,
            "ps_mask_channels": 9,
            "ps_spatial_attention_channels": 1,
            "ps_normalization": "LayerNorm2d",
            "fh_aspp_dilations": (1, 3, 6),
            "fh_se_hidden_channels": 8,
            "fh_normalization": "LayerNorm2d",
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
            raise ValueError(f"Phase 1C specialized-model contract drifted: {drift}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class"] = payload.pop("class_name")
        return payload


@dataclass(frozen=True)
class Phase1CResourceConfig:
    precision: str
    amp_enabled: bool
    operator_probe_max_seconds: int
    tiny_max_steps: int
    tiny_max_seconds: int
    formal_max_seconds: int
    total_gpu_max_seconds: int
    closing_reserve_seconds: int
    post_evaluation_reserve_seconds: int
    milestone_epochs: tuple[int, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1CResourceConfig:
        config = cls(
            **_strict_values(cls, values, tuple_fields=("milestone_epochs",))
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected = {
            "precision": "float32",
            "amp_enabled": False,
            "operator_probe_max_seconds": 300,
            "tiny_max_steps": 500,
            "tiny_max_seconds": 2400,
            "formal_max_seconds": 9000,
            "total_gpu_max_seconds": 10800,
            "closing_reserve_seconds": 120,
            "post_evaluation_reserve_seconds": 600,
            "milestone_epochs": (1, 3, 5, 10, 16),
        }
        actual = asdict(self)
        if actual != expected:
            raise ValueError(f"Phase 1C resource contract drifted: {actual}")


@dataclass(frozen=True)
class Phase1CProtocolConfig:
    schema_version: int
    experiment_name: str
    testing_frozen: bool
    training: Phase1CTrainingConfig
    model: Phase1CSpecializedModelConfig
    optimizer: Phase1AOptimizerConfig
    resources: Phase1CResourceConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Phase1CProtocolConfig:
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
                f"Phase 1C top-level fields invalid: missing={missing}, unknown={unknown}"
            )
        for section in ("training", "model", "optimizer", "resources"):
            if not isinstance(values[section], Mapping):
                raise ValueError(f"Phase 1C '{section}' section must be a mapping")
        if values["testing_frozen"] is not True:
            raise PermissionError("Phase 1C testing_frozen must be the boolean true")
        config = cls(
            schema_version=int(values["schema_version"]),
            experiment_name=str(values["experiment_name"]),
            testing_frozen=values["testing_frozen"],
            training=Phase1CTrainingConfig.from_mapping(values["training"]),
            model=Phase1CSpecializedModelConfig.from_mapping(values["model"]),
            optimizer=Phase1AOptimizerConfig.from_mapping(values["optimizer"]),
            resources=Phase1CResourceConfig.from_mapping(values["resources"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Phase 1C config schema_version must be 1")
        if self.experiment_name != "H3_specialized_B2_seed42_16e":
            raise ValueError("Unexpected Phase 1C experiment_name")
        if not self.testing_frozen:
            raise PermissionError("Phase 1C must keep testing frozen")
        if self.training.align_corners != self.model.align_corners:
            raise ValueError("Training and model align_corners settings differ")
        if self.training.keypoint_order != self.model.concatenation_order:
            raise ValueError("Training and specialized-model keypoint order differ")


def load_phase1c_config(path: str | Path) -> Phase1CProtocolConfig:
    """Load the path-free Phase 1C protocol and reject any contract drift."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 1C configuration does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    loaded = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError("Phase 1C configuration must be a mapping")
    return Phase1CProtocolConfig.from_mapping(loaded)


def build_phase1c_adam(
    parameters: Iterable[Tensor],
    config: Phase1CProtocolConfig,
) -> Adam:
    """Build the locked Phase 1C Adam optimizer."""

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
