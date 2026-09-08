"""Strict GeoEqui-LD model and temperature-free DSNT from the advisor DOCX."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dsnt import spatial_expectation
from .hrnet import HRNetContractError, HRNetW32SplitHeatmap
from .specialized import FHFeatureEnhancer, PSFeatureEnhancer


def standard_spatial_softmax(heatmap_logits: Tensor) -> Tensor:
    """Apply an ordinary spatial softmax with no temperature parameter."""

    if heatmap_logits.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(heatmap_logits.shape)}")
    if not torch.is_floating_point(heatmap_logits):
        raise TypeError("DSNT input must use a floating dtype")
    if not bool(torch.isfinite(heatmap_logits).all()):
        raise ValueError("DSNT input contains NaN or Inf")
    batch, channels, height, width = heatmap_logits.shape
    probabilities = F.softmax(heatmap_logits.reshape(batch, channels, height * width), dim=-1)
    return probabilities.reshape(batch, channels, height, width)


class StandardDSNT(nn.Module):
    """DOCX DSNT: ordinary spatial Softmax followed by an ``[x,y]`` expectation."""

    def forward(self, heatmap_logits: Tensor) -> Tensor:
        probabilities = standard_spatial_softmax(heatmap_logits)
        return spatial_expectation(probabilities, align_corners=True)


class StrictGeoEquiLDModel(HRNetW32SplitHeatmap):
    """Advisor-DOCX HRNet-W32 with independent PS/FH enhancement and decoders.

    This neutral strict entry point intentionally does not inherit the H1/H2/H3
    experiment naming or copy an earlier checkpoint.  It starts from PyTorch/timm
    module initialization with ``pretrained=False`` and enforces the exact public
    tensor contract used by the strict pilot.
    """

    input_size_hw = (512, 512)
    output_size_hw = (256, 256)

    def __init__(self) -> None:
        super().__init__(align_corners=True)
        self.ps_enhancer = PSFeatureEnhancer()
        self.fh_enhancer = FHFeatureEnhancer()

    @staticmethod
    def _validate_inputs(inputs: Tensor) -> None:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,512,512], got {tuple(inputs.shape)}")
        if tuple(inputs.shape[-2:]) != StrictGeoEquiLDModel.input_size_hw:
            raise ValueError(f"Strict GeoEqui-LD requires 512x512 input, got {tuple(inputs.shape)}")

    def forward(self, inputs: Tensor) -> Tensor:
        self._validate_inputs(inputs)
        features = self.extract_high_resolution_features(inputs)
        expected_features = (inputs.shape[0], 32, 128, 128)
        if tuple(features.shape) != expected_features:
            raise HRNetContractError(
                f"Strict HRNet feature contract failed: {tuple(features.shape)} != "
                f"{expected_features}"
            )
        ps_heatmaps = self.ps_decoder(self.ps_enhancer(features))
        fh_heatmap = self.fh_decoder(self.fh_enhancer(features))
        heatmaps = torch.cat((ps_heatmaps, fh_heatmap), dim=1)
        output = F.interpolate(
            heatmaps,
            size=self.output_size_hw,
            mode="bilinear",
            align_corners=True,
        )
        expected_output = (inputs.shape[0], 3, *self.output_size_hw)
        if tuple(output.shape) != expected_output:
            raise HRNetContractError(
                f"Strict output contract failed: {tuple(output.shape)} != {expected_output}"
            )
        return output


__all__ = [
    "StandardDSNT",
    "StrictGeoEquiLDModel",
    "standard_spatial_softmax",
]
