"""HRNet-W32 heatmap models for the Phase 1A/1B supervised probes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

import timm
import torch
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


class SplitHeatmapDecoder(nn.Module):
    """Decode one task branch without sharing decoder parameters."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        if out_channels not in (1, 2):
            raise ValueError("A split decoder must emit one or two heatmap channels")
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = nn.GELU()
        self.output = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != HRNET_FEATURE_CHANNELS:
            raise HRNetContractError(
                "Split decoder requires [B,32,H,W] final HRNet features, got "
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


class HRNetW32SplitHeatmap(HRNetW32SharedHeatmap):
    """HRNet-W32 with independent PS and FH heatmap decoders.

    The backbone and feature contract are identical to
    :class:`HRNetW32SharedHeatmap`.  Only the decoder topology changes: the PS
    decoder emits ``[PS1, PS2]`` while the FH decoder emits ``[FH1]``.  Their
    outputs are concatenated before the same bilinear upsampling operation.
    """

    def __init__(self, *, align_corners: bool = True) -> None:
        super().__init__(align_corners=align_corners)
        del self.decoder
        self.ps_decoder = SplitHeatmapDecoder(out_channels=2)
        self.fh_decoder = SplitHeatmapDecoder(out_channels=1)

    @classmethod
    def from_shared(
        cls,
        shared: HRNetW32SharedHeatmap,
    ) -> HRNetW32SplitHeatmap:
        """Build an independent split model with the shared model's state."""

        split = cls(align_corners=shared.align_corners)
        reference_parameter = next(shared.parameters())
        split.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        return initialize_split_from_shared(shared, split)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,H,W] input, got {tuple(inputs.shape)}")
        if any(size < 32 or size % 32 for size in inputs.shape[-2:]):
            raise ValueError("Input height and width must be at least 32 and divisible by 32")

        features = self.extract_high_resolution_features(inputs)
        ps_heatmaps = self.ps_decoder(features)
        fh_heatmap = self.fh_decoder(features)
        heatmaps = torch.cat((ps_heatmaps, fh_heatmap), dim=1)
        output_size = tuple(size // 2 for size in inputs.shape[-2:])
        output = F.interpolate(
            heatmaps,
            size=output_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if output.shape[1] != 3 or tuple(output.shape[-2:]) != output_size:
            raise HRNetContractError(
                f"Split decoder output contract failed: got {tuple(output.shape)}"
            )
        return output


def _copy_module_modes(source: nn.Module, target: nn.Module) -> None:
    """Copy train/eval flags without mutating parameters or buffers."""

    source_modules = dict(source.named_modules())
    target_modules = dict(target.named_modules())
    if source_modules.keys() != target_modules.keys():
        raise HRNetContractError("Cannot copy module modes across different structures")
    for name, target_module in target_modules.items():
        target_module.training = source_modules[name].training


def _copy_shared_decoder_branch(
    shared: SharedHeatmapDecoder,
    target: SplitHeatmapDecoder,
    output_rows: slice,
) -> None:
    """Copy one independent decoder branch from selected shared output rows."""

    for module_name in ("conv1", "bn1", "conv2", "bn2"):
        source_module = getattr(shared, module_name)
        target_module = getattr(target, module_name)
        target_module.load_state_dict(source_module.state_dict(), strict=True)

    source_weight = shared.output.weight[output_rows]
    source_bias = shared.output.bias
    if source_bias is None or target.output.bias is None:
        raise HRNetContractError("Phase 1B output convolutions must include bias")
    source_bias = source_bias[output_rows]
    if source_weight.shape != target.output.weight.shape:
        raise HRNetContractError("Shared output rows do not match split decoder shape")
    with torch.no_grad():
        target.output.weight.copy_(source_weight)
        target.output.bias.copy_(source_bias)

    shared_parameters = dict(shared.named_parameters())
    for name, parameter in target.named_parameters():
        parameter.requires_grad_(shared_parameters[name].requires_grad)
    _copy_module_modes(shared, target)


def initialize_split_from_shared(
    shared: HRNetW32SharedHeatmap,
    split: HRNetW32SplitHeatmap,
) -> HRNetW32SplitHeatmap:
    """Copy one untrained shared initialization into independent PS/FH heads.

    The operation copies values into the already allocated split model.  It
    never aliases parameters or buffers, so subsequent optimisation of either
    decoder cannot directly mutate the other decoder or the source model.
    """

    if type(shared) is not HRNetW32SharedHeatmap:
        raise TypeError("shared must be an HRNetW32SharedHeatmap instance")
    if type(split) is not HRNetW32SplitHeatmap:
        raise TypeError("split must be an HRNetW32SplitHeatmap instance")
    if shared.align_corners != split.align_corners:
        raise HRNetContractError("Shared and split models use different upsampling settings")
    if shared.feature_contract != split.feature_contract:
        raise HRNetContractError("Shared and split models use different HRNet feature contracts")

    split.backbone.load_state_dict(shared.backbone.state_dict(), strict=True)
    shared_backbone_parameters = dict(shared.backbone.named_parameters())
    for name, parameter in split.backbone.named_parameters():
        parameter.requires_grad_(shared_backbone_parameters[name].requires_grad)
    _copy_module_modes(shared.backbone, split.backbone)
    _copy_shared_decoder_branch(shared.decoder, split.ps_decoder, slice(0, 2))
    _copy_shared_decoder_branch(shared.decoder, split.fh_decoder, slice(2, 3))
    split.training = shared.training
    return split


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters for report and checkpoint provenance."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
