from __future__ import annotations

import torch

from geoequi_ld.models.unet import HeatmapUNet


def test_unet_maps_512_grayscale_to_three_256_heatmaps() -> None:
    model = HeatmapUNet(in_channels=1, out_channels=3, base_channels=4)
    with torch.inference_mode():
        output = model(torch.zeros((1, 1, 512, 512)))
    assert output.shape == (1, 3, 256, 256)
    assert torch.isfinite(output).all()


def test_unet_uses_batch_size_safe_normalization() -> None:
    model = HeatmapUNet(in_channels=1, out_channels=3, base_channels=4)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
