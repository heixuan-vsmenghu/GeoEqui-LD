"""Differentiable spatial-to-numerical transform (DSNT)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def spatial_softmax(heatmap_logits: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Normalize every keypoint heatmap into a spatial probability map."""

    if heatmap_logits.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(heatmap_logits.shape)}")
    if not torch.is_floating_point(heatmap_logits):
        raise TypeError("DSNT input must use a floating dtype")
    if not torch.isfinite(heatmap_logits).all():
        raise ValueError("DSNT input contains NaN or Inf")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, channels, height, width = heatmap_logits.shape
    return F.softmax(
        heatmap_logits.reshape(batch, channels, height * width) / float(temperature),
        dim=-1,
    ).reshape(batch, channels, height, width)


def spatial_expectation(probabilities: Tensor, *, align_corners: bool = True) -> Tensor:
    """Return normalized ``[x,y]`` means from spatial probability maps."""

    if probabilities.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(probabilities.shape)}")
    if not torch.is_floating_point(probabilities):
        raise TypeError("DSNT probabilities must use a floating dtype")
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("DSNT probabilities must be finite and non-negative")
    _, _, height, width = probabilities.shape
    if height <= 1 or width <= 1:
        raise ValueError("DSNT requires both spatial dimensions to exceed one")

    if align_corners:
        x_axis = torch.linspace(
            -1.0, 1.0, width, device=probabilities.device, dtype=probabilities.dtype
        )
        y_axis = torch.linspace(
            -1.0, 1.0, height, device=probabilities.device, dtype=probabilities.dtype
        )
    else:
        x_axis = (
            2.0
            * (torch.arange(width, device=probabilities.device, dtype=probabilities.dtype) + 0.5)
            / width
        ) - 1.0
        y_axis = (
            2.0
            * (torch.arange(height, device=probabilities.device, dtype=probabilities.dtype) + 0.5)
            / height
        ) - 1.0

    expected_x = torch.sum(probabilities * x_axis.view(1, 1, 1, width), dim=(2, 3))
    expected_y = torch.sum(probabilities * y_axis.view(1, 1, height, 1), dim=(2, 3))
    return torch.stack((expected_x, expected_y), dim=-1)


class DSNT(nn.Module):
    """Convert heatmap logits ``[B,C,H,W]`` to normalized ``[x,y]`` means.

    A configurable temperature is included because Phase 0 trains heatmaps with
    MSE targets in ``[0,1]``; without sharpening, a spatial softmax over 65k
    pixels can remain almost uniform even around a correct peak.
    """

    def __init__(self, *, temperature: float = 0.05, align_corners: bool = True) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.align_corners = bool(align_corners)

    def forward(self, heatmap_logits: Tensor) -> Tensor:
        if heatmap_logits.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(heatmap_logits.shape)}")
        batch, channels, height, width = heatmap_logits.shape
        if height <= 1 or width <= 1:
            raise ValueError("DSNT requires both spatial dimensions to exceed one")

        probabilities = spatial_softmax(heatmap_logits, temperature=self.temperature)
        return spatial_expectation(probabilities, align_corners=self.align_corners)
