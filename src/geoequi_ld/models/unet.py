"""Small supervised U-Net heatmap baseline for Phase 0."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Phase 0 commonly runs with batch_size=1 on a 4 GB GPU. GroupNorm
        # avoids BatchNorm's train/eval running-statistics mismatch in the
        # tiny-overfit gate while retaining affine normalization parameters.
        group_count = min(8, out_channels)
        while out_channels % group_count:
            group_count -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(
                f"U-Net skip mismatch: upsampled {tuple(upsampled.shape)} "
                f"versus skip {tuple(skip.shape)}"
            )
        return self.conv(torch.cat((upsampled, skip), dim=1))


class HeatmapUNet(nn.Module):
    """Predict three half-resolution heatmap logits from grayscale images.

    A 512x512 input produces a 3x256x256 output as required by the Phase 0
    coordinate contract.  The final full-resolution decoder stage is omitted
    deliberately rather than followed by an implicit resize.
    """

    def __init__(
        self, *, in_channels: int = 1, out_channels: int = 3, base_channels: int = 16
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive")
        base = int(base_channels)
        self.encoder1 = ConvBlock(in_channels, base)
        self.encoder2 = ConvBlock(base, base * 2)
        self.encoder3 = ConvBlock(base * 2, base * 4)
        self.encoder4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up4 = UpBlock(base * 16, base * 8, base * 8)
        self.up3 = UpBlock(base * 8, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2, base * 2)
        self.head = nn.Conv2d(base * 2, out_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(inputs.shape)}")
        if inputs.shape[-2] % 16 or inputs.shape[-1] % 16:
            raise ValueError("Input height and width must be divisible by 16")
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        decoded4 = self.up4(bottleneck, enc4)
        decoded3 = self.up3(decoded4, enc3)
        decoded2 = self.up2(decoded3, enc2)
        return self.head(decoded2)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
