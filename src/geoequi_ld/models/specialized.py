"""Phase 1C PS/FH-specialized feature enhancement for HRNet-W32."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d

from .hrnet import (
    HRNetContractError,
    HRNetW32SplitHeatmap,
    SplitHeatmapDecoder,
)

SPECIALIZED_FEATURE_CHANNELS: Final = 32
PS_OFFSET_CHANNELS: Final = 18
PS_MASK_CHANNELS: Final = 9
PS_OFFSET_MASK_CHANNELS: Final = PS_OFFSET_CHANNELS + PS_MASK_CHANNELS
PINNED_TORCHVISION_VERSION: Final = "0.20.1"


def _installed_torchvision_version() -> str:
    try:
        installed = version("torchvision")
    except PackageNotFoundError as error:  # pragma: no cover - import already requires it
        raise HRNetContractError("torchvision is not installed") from error
    base_version = installed.split("+", maxsplit=1)[0]
    if base_version != PINNED_TORCHVISION_VERSION:
        raise HRNetContractError(
            "Phase 1C requires torchvision=="
            f"{PINNED_TORCHVISION_VERSION} (build suffix allowed), got {installed}"
        )
    return installed


def _require_feature_tensor(features: Tensor, *, owner: str) -> None:
    if features.ndim != 4 or features.shape[1] != SPECIALIZED_FEATURE_CHANNELS:
        raise HRNetContractError(
            f"{owner} requires [B,{SPECIALIZED_FEATURE_CHANNELS},H,W] features, "
            f"got {tuple(features.shape)}"
        )


class LayerNorm2d(nn.Module):
    """Apply genuine per-pixel channel LayerNorm to an NCHW tensor.

    ``nn.LayerNorm`` operates on the final dimension.  The explicit NCHW to
    NHWC permutation therefore normalizes exactly the 32 channels at each
    spatial location; no BatchNorm or GroupNorm approximation is involved.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("LayerNorm2d channels must be positive")
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"LayerNorm2d expected [B,{self.channels},H,W], got {tuple(inputs.shape)}"
            )
        channels_last = inputs.permute(0, 2, 3, 1)
        normalized = self.norm(channels_last)
        return normalized.permute(0, 3, 1, 2).contiguous()


class PSFeatureEnhancer(nn.Module):
    """Enhance PS features with modulated deformable convolution and attention.

    The 27-channel predictor is deliberately zero-initialized.  Consequently,
    a newly constructed module starts with zero offsets, zero modulation-mask
    logits, and masks equal to ``sigmoid(0) == 0.5``.  The values remain fully
    learnable; no fixed offset or ordinary-convolution fallback is used.
    """

    def __init__(self, channels: int = SPECIALIZED_FEATURE_CHANNELS) -> None:
        super().__init__()
        if channels != SPECIALIZED_FEATURE_CHANNELS:
            raise ValueError(
                f"Phase 1C locks PSFeatureEnhancer to {SPECIALIZED_FEATURE_CHANNELS} channels"
            )
        self.channels = channels
        self.torchvision_version = _installed_torchvision_version()
        self.offset_mask = nn.Conv2d(
            channels,
            PS_OFFSET_MASK_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.deform = DeformConv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.deform_activation = nn.GELU()
        self.spatial_attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.norm = LayerNorm2d(channels)
        self.reset_offset_mask_parameters()

    def reset_offset_mask_parameters(self) -> None:
        """Restore the registered Phase 1C zero-offset/half-mask start."""

        nn.init.zeros_(self.offset_mask.weight)
        if self.offset_mask.bias is None:  # pragma: no cover - constructor fixes bias=True
            raise HRNetContractError("PS offset/mask predictor unexpectedly has no bias")
        nn.init.zeros_(self.offset_mask.bias)

    @property
    def initialization_summary(self) -> dict[str, Any]:
        """Return the explicit behavior of the specially initialized predictor."""

        return {
            "offset_mask_predictor": "zero_weight_and_bias",
            "initial_offset": 0.0,
            "initial_mask_logits": 0.0,
            "initial_mask_after_sigmoid": 0.5,
            "deformable_operator": "torchvision.ops.DeformConv2d",
            "torchvision_version": self.torchvision_version,
            "required_torchvision_base_version": PINNED_TORCHVISION_VERSION,
            "ordinary_conv_fallback": False,
        }

    def predict_offset_and_mask(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return offsets, sigmoid masks, and the unsquashed mask logits."""

        _require_feature_tensor(features, owner=type(self).__name__)
        prediction = self.offset_mask(features)
        offsets, mask_logits = torch.split(
            prediction,
            (PS_OFFSET_CHANNELS, PS_MASK_CHANNELS),
            dim=1,
        )
        masks = torch.sigmoid(mask_logits)
        expected_offset_shape = (
            features.shape[0],
            PS_OFFSET_CHANNELS,
            features.shape[2],
            features.shape[3],
        )
        expected_mask_shape = (
            features.shape[0],
            PS_MASK_CHANNELS,
            features.shape[2],
            features.shape[3],
        )
        if tuple(offsets.shape) != expected_offset_shape:
            raise HRNetContractError(
                f"PS deformable offsets have shape {tuple(offsets.shape)}, "
                f"expected {expected_offset_shape}"
            )
        if tuple(masks.shape) != expected_mask_shape:
            raise HRNetContractError(
                f"PS modulation masks have shape {tuple(masks.shape)}, "
                f"expected {expected_mask_shape}"
            )
        return offsets, masks, mask_logits

    def forward(self, features: Tensor) -> Tensor:
        _require_feature_tensor(features, owner=type(self).__name__)
        offsets, masks, _ = self.predict_offset_and_mask(features)
        deform_feature = self.deform(features, offsets, masks)
        attention = torch.sigmoid(self.spatial_attention(features))
        edge_feature = self.deform_activation(deform_feature) * attention
        return self.norm(features + edge_feature)


class FHFeatureEnhancer(nn.Module):
    """Enhance FH features with ASPP-lite and SE channel attention."""

    def __init__(self, channels: int = SPECIALIZED_FEATURE_CHANNELS) -> None:
        super().__init__()
        if channels != SPECIALIZED_FEATURE_CHANNELS:
            raise ValueError(
                f"Phase 1C locks FHFeatureEnhancer to {SPECIALIZED_FEATURE_CHANNELS} channels"
            )
        self.channels = channels
        self.aspp_d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.aspp_d3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3),
            nn.GELU(),
        )
        self.aspp_d6 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, dilation=6, padding=6),
            nn.GELU(),
        )
        self.aspp_projection = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GELU(),
        )
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Conv2d(channels, 8, kernel_size=1)
        self.se_activation = nn.ReLU()
        self.se_expand = nn.Conv2d(8, channels, kernel_size=1)
        self.norm = LayerNorm2d(channels)

    def channel_weights(self, features: Tensor) -> Tensor:
        """Return SE weights shaped ``[B,32,1,1]``."""

        _require_feature_tensor(features, owner=type(self).__name__)
        pooled = self.se_pool(features)
        return torch.sigmoid(
            self.se_expand(self.se_activation(self.se_reduce(pooled)))
        )

    def forward(self, features: Tensor) -> Tensor:
        _require_feature_tensor(features, owner=type(self).__name__)
        aspp_feature = self.aspp_projection(
            torch.cat(
                (
                    self.aspp_d1(features),
                    self.aspp_d3(features),
                    self.aspp_d6(features),
                ),
                dim=1,
            )
        )
        se_feature = features * self.channel_weights(features)
        semantic_feature = aspp_feature + se_feature
        return self.norm(features + semantic_feature)


def _storage_pointers(tensors: Iterable[Tensor]) -> set[int]:
    return {tensor.untyped_storage().data_ptr() for tensor in tensors}


def _parameters_and_buffers(module: nn.Module) -> Iterable[Tensor]:
    yield from module.parameters()
    yield from module.buffers()


def _copy_module_values_and_modes(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise HRNetContractError("Source and target parameter names differ during H3 setup")
    for name, parameter in target_parameters.items():
        parameter.requires_grad_(source_parameters[name].requires_grad)

    source_modules = dict(source.named_modules())
    target_modules = dict(target.named_modules())
    if source_modules.keys() != target_modules.keys():
        raise HRNetContractError("Source and target module names differ during H3 setup")
    for name, target_module in target_modules.items():
        target_module.training = source_modules[name].training


class HRNetW32SpecializedHeatmap(HRNetW32SplitHeatmap):
    """HRNet-W32 with PS/FH enhancers feeding the existing split decoders.

    Channel order remains ``[PS1, PS2, FH1]`` and logits are bilinearly
    upsampled from reduction 4 to reduction 2 with ``align_corners=True``.
    """

    def __init__(self, *, align_corners: bool = True) -> None:
        super().__init__(align_corners=align_corners)
        self.ps_enhancer = PSFeatureEnhancer()
        self.fh_enhancer = FHFeatureEnhancer()
        self.register_buffer(
            "_base_initialization_copied",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    @property
    def base_initialization_copied(self) -> bool:
        """Whether an H2 backbone/decoder copy was recorded in this state dict."""

        return bool(self._base_initialization_copied.item())

    @classmethod
    def from_split(cls, split: HRNetW32SplitHeatmap) -> HRNetW32SpecializedHeatmap:
        """Create H3 and copy an untrained H2 base without sharing storage."""

        specialized = cls(align_corners=split.align_corners)
        reference_parameter = next(split.parameters())
        specialized.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        return initialize_specialized_from_split(split, specialized)

    @property
    def initialization_summary(self) -> dict[str, Any]:
        return {
            "base_source": "HRNetW32SplitHeatmap",
            "backbone_and_decoders_copied": self.base_initialization_copied,
            "base_parameter_storage_aliased": False,
            "enhancers_have_own_parameter_storage": True,
            "complete_function_initially_equivalent_to_h2": False,
            "ps_enhancer": self.ps_enhancer.initialization_summary,
            "fh_enhancer": {
                "aspp_dilations": [1, 3, 6],
                "se_reduction": "32_to_8_to_32",
                "initialization": "pytorch_module_defaults",
            },
        }

    @staticmethod
    def _validate_image_inputs(inputs: Tensor) -> None:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,H,W] input, got {tuple(inputs.shape)}")
        if any(size < 32 or size % 32 for size in inputs.shape[-2:]):
            raise ValueError("Input height and width must be at least 32 and divisible by 32")

    def _upsample(self, heatmaps: Tensor, inputs: Tensor, channels: int) -> Tensor:
        output_size = tuple(size // 2 for size in inputs.shape[-2:])
        output = F.interpolate(
            heatmaps,
            size=output_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if output.shape[1] != channels or tuple(output.shape[-2:]) != output_size:
            raise HRNetContractError(
                f"Specialized decoder output contract failed: got {tuple(output.shape)}"
            )
        return output

    def forward_ps(self, inputs: Tensor) -> Tensor:
        """Evaluate only the PS path for branch-isolated diagnostics."""

        self._validate_image_inputs(inputs)
        features = self.extract_high_resolution_features(inputs)
        ps_heatmaps = self.ps_decoder(self.ps_enhancer(features))
        return self._upsample(ps_heatmaps, inputs, channels=2)

    def forward_fh(self, inputs: Tensor) -> Tensor:
        """Evaluate only the FH path for branch-isolated diagnostics."""

        self._validate_image_inputs(inputs)
        features = self.extract_high_resolution_features(inputs)
        fh_heatmap = self.fh_decoder(self.fh_enhancer(features))
        return self._upsample(fh_heatmap, inputs, channels=1)

    def forward(self, inputs: Tensor) -> Tensor:
        self._validate_image_inputs(inputs)
        features = self.extract_high_resolution_features(inputs)
        ps_heatmaps = self.ps_decoder(self.ps_enhancer(features))
        fh_heatmap = self.fh_decoder(self.fh_enhancer(features))
        heatmaps = torch.cat((ps_heatmaps, fh_heatmap), dim=1)
        return self._upsample(heatmaps, inputs, channels=3)


def initialize_specialized_from_split(
    split: HRNetW32SplitHeatmap,
    specialized: HRNetW32SpecializedHeatmap,
) -> HRNetW32SpecializedHeatmap:
    """Copy H2 backbone/decoders into H3 while retaining H3 enhancer initialization.

    This function copies tensor values and train/eval modes into already
    allocated modules.  It does not replace module objects, so H2 and H3 never
    share parameter or buffer storage.  The PS/FH enhancers are intentionally
    untouched and therefore keep their own Phase 1C initialization.
    """

    if type(split) is not HRNetW32SplitHeatmap:
        raise TypeError("split must be an HRNetW32SplitHeatmap instance")
    if type(specialized) is not HRNetW32SpecializedHeatmap:
        raise TypeError("specialized must be an HRNetW32SpecializedHeatmap instance")
    if split.align_corners != specialized.align_corners:
        raise HRNetContractError("H2 and H3 use different upsampling settings")
    if split.feature_contract != specialized.feature_contract:
        raise HRNetContractError("H2 and H3 use different HRNet feature contracts")
    if not isinstance(split.ps_decoder, SplitHeatmapDecoder) or not isinstance(
        split.fh_decoder, SplitHeatmapDecoder
    ):
        raise HRNetContractError("H2 does not expose the locked split decoder structure")

    _copy_module_values_and_modes(split.backbone, specialized.backbone)
    _copy_module_values_and_modes(split.ps_decoder, specialized.ps_decoder)
    _copy_module_values_and_modes(split.fh_decoder, specialized.fh_decoder)
    specialized.ps_enhancer.train(split.training)
    specialized.fh_enhancer.train(split.training)
    specialized.training = split.training

    source_storage = _storage_pointers(_parameters_and_buffers(split))
    copied_modules = (specialized.backbone, specialized.ps_decoder, specialized.fh_decoder)
    target_storage: set[int] = set()
    for module in copied_modules:
        target_storage.update(_storage_pointers(_parameters_and_buffers(module)))
    if not source_storage.isdisjoint(target_storage):
        raise HRNetContractError("H2 and H3 unexpectedly share parameter or buffer storage")
    specialized._base_initialization_copied.fill_(True)
    return specialized
