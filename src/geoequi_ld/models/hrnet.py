"""HRNet-W32 shared heatmap model for the Phase 1A supervised probe."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

import timm
from torch import Tensor, nn
from torch.nn import functional as F

PINNED_TIMM_VERSION: Final = "1.0.28"
HRNET_BACKBONE_NAME: Final = "hrnet_w32"
HRNET_FEATURE_LOCATION: Final = ""
HRNET_OUT_INDICES: Final = (1,)
HRNET_FEATURE_CHANNELS: Final = 32
HRNET_FEATURE_REDUCTION: Final = 4


class HRNetContractError(RuntimeError):
    """Raised when the installed timm HRNet interface drifts from Phase 1A."""


@dataclass(frozen=True)
class HRNetFeatureContract:
    """Auditable identity of the timm feature selected by this model."""

    timm_version: str
    backbone_name: str
    feature_location: str
    out_indices: tuple[int, ...]
    channels: tuple[int, ...]
    reductions: tuple[int, ...]
    final_fusion_module_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SharedHeatmapDecoder(nn.Module):
    """Decode the 1/4-resolution W32 branch into three 1/2-resolution logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = nn.GELU()
        self.output = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != HRNET_FEATURE_CHANNELS:
            raise HRNetContractError(
                "Shared decoder requires [B,32,H,W] final HRNet features, got "
                f"{tuple(features.shape)}"
            )
        features = self.act1(self.bn1(self.conv1(features)))
        features = self.act2(self.bn2(self.conv2(features)))
        return self.output(features)


def _installed_timm_version() -> str:
    try:
        return version("timm")
    except PackageNotFoundError as error:  # pragma: no cover - import already requires timm
        raise HRNetContractError("timm is not installed") from error


def _feature_values(feature_info: Any, method: str) -> tuple[int, ...]:
    getter = getattr(feature_info, method, None)
    if not callable(getter):
        raise HRNetContractError(f"timm feature_info has no callable {method}()")
    values = getter()
    if not isinstance(values, Sequence):
        raise HRNetContractError(f"timm feature_info.{method}() did not return a sequence")
    return tuple(int(value) for value in values)


class HRNetW32SharedHeatmap(nn.Module):
    """Use timm HRNet-W32's final fused high-resolution branch.

    The timm HRNet feature wrapper assigns feature index 0 to the first stem
    convolution.  With ``feature_location=''``, feature index 1 is branch 0 of
    the *final* stage4 output: 32 channels at input reduction 4.  Both the
    version and feature metadata are checked at construction time so a timm
    API change cannot silently substitute the stem or classification head.
    """

    def __init__(self, *, align_corners: bool = True) -> None:
        super().__init__()
        installed_version = _installed_timm_version()
        if installed_version != PINNED_TIMM_VERSION:
            raise HRNetContractError(
                f"Phase 1A requires timm=={PINNED_TIMM_VERSION}, got {installed_version}"
            )
        if not align_corners:
            raise ValueError("Phase 1A locks bilinear upsampling to align_corners=True")

        self.align_corners = align_corners
        self.backbone = timm.create_model(
            HRNET_BACKBONE_NAME,
            pretrained=False,
            in_chans=1,
            features_only=True,
            feature_location=HRNET_FEATURE_LOCATION,
            out_indices=HRNET_OUT_INDICES,
        )
        feature_info = getattr(self.backbone, "feature_info", None)
        if feature_info is None:
            raise HRNetContractError("timm HRNet-W32 did not expose feature_info")
        channels = _feature_values(feature_info, "channels")
        reductions = _feature_values(feature_info, "reduction")
        if channels != (HRNET_FEATURE_CHANNELS,) or reductions != (
            HRNET_FEATURE_REDUCTION,
        ):
            raise HRNetContractError(
                "Expected final HRNet-W32 branch metadata channels=(32,), "
                f"reductions=(4,), got channels={channels}, reductions={reductions}"
            )

        stage4 = getattr(self.backbone, "stage4", None)
        if not isinstance(stage4, nn.Sequential) or len(stage4) == 0:
            raise HRNetContractError("timm HRNet-W32 has no non-empty sequential stage4")
        self.feature_contract = HRNetFeatureContract(
            timm_version=installed_version,
            backbone_name=HRNET_BACKBONE_NAME,
            feature_location=HRNET_FEATURE_LOCATION,
            out_indices=HRNET_OUT_INDICES,
            channels=channels,
            reductions=reductions,
            final_fusion_module_path=f"backbone.stage4.{len(stage4) - 1}",
        )
        self.decoder = SharedHeatmapDecoder()

    @property
    def final_fusion_module(self) -> nn.Module:
        """Return the final multi-scale fusion module for hook-based audits."""

        stage4 = self.backbone.stage4
        return stage4[-1]

    def extract_high_resolution_features(self, inputs: Tensor) -> Tensor:
        """Return only the checked final stage4 high-resolution branch."""

        features = self.backbone(inputs)
        if not isinstance(features, list | tuple) or len(features) != 1:
            raise HRNetContractError(
                "timm HRNet-W32 must return exactly one selected feature tensor"
            )
        feature = features[0]
        if not isinstance(feature, Tensor) or feature.ndim != 4:
            raise HRNetContractError("Selected HRNet feature is not a 4D tensor")
        expected_spatial = tuple(size // HRNET_FEATURE_REDUCTION for size in inputs.shape[-2:])
        if feature.shape[1] != HRNET_FEATURE_CHANNELS or tuple(feature.shape[-2:]) != (
            expected_spatial
        ):
            raise HRNetContractError(
                "Selected feature is not the final 32-channel, reduction-4 HRNet branch: "
                f"got {tuple(feature.shape)}, expected spatial {expected_spatial}"
            )
        return feature

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,H,W] input, got {tuple(inputs.shape)}")
        if any(size < 32 or size % 32 for size in inputs.shape[-2:]):
            raise ValueError("Input height and width must be at least 32 and divisible by 32")
        features = self.extract_high_resolution_features(inputs)
        heatmaps = self.decoder(features)
        output_size = tuple(size // 2 for size in inputs.shape[-2:])
        output = F.interpolate(
            heatmaps,
            size=output_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if output.shape[1] != 3 or tuple(output.shape[-2:]) != output_size:
            raise HRNetContractError(
                f"Shared decoder output contract failed: got {tuple(output.shape)}"
            )
        return output


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters for report and checkpoint provenance."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
