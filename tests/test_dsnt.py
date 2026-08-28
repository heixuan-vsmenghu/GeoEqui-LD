from __future__ import annotations

import torch

from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps
from geoequi_ld.geometry.coordinates import normalized_to_pixel
from geoequi_ld.models.dsnt import DSNT, spatial_softmax


def test_spatial_softmax_sums_to_one_per_channel() -> None:
    logits = torch.randn((2, 3, 9, 7))
    probabilities = spatial_softmax(logits, temperature=0.2)
    torch.testing.assert_close(probabilities.sum(dim=(-2, -1)), torch.ones((2, 3)))


def test_dsnt_recovers_centres_from_sharpened_gaussian_logits() -> None:
    target = torch.tensor([[32.0, 64.0], [127.5, 127.5], [220.0, 220.0]])
    logits = generate_gaussian_heatmaps(target, size_hw=(256, 256), sigma=4.0).unsqueeze(0)
    normalized = DSNT(temperature=0.05, align_corners=True)(logits)
    predicted = normalized_to_pixel(normalized, (256, 256), align_corners=True)
    assert normalized.shape == (1, 3, 2)
    torch.testing.assert_close(predicted[0], target, atol=0.02, rtol=0.0)


def test_dsnt_gradient_reaches_heatmap_logits() -> None:
    logits = torch.zeros((2, 3, 16, 16), dtype=torch.float32, requires_grad=True)
    logits.data[:, :, 4, 11] = 1.0
    output = DSNT(temperature=0.05)(logits)
    loss = output.square().sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
