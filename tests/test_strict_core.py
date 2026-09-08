"""Core-only excerpts from the existing strict component tests.

The DSNT and supervised MSE tests retain their original assertions. The HRNet
smoke test uses random initialization and synthetic input, without weights or data.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from geoequi_ld.models.strict import (
    StandardDSNT,
    StrictGeoEquiLDModel,
    standard_spatial_softmax,
)
from geoequi_ld.training.strict_losses import strict_supervised_heatmap_mse


def test_standard_dsnt_is_exact_ordinary_softmax_without_temperature() -> None:
    logits = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0]], [[-1.0, 0.0], [0.5, 2.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    with pytest.raises(TypeError):
        StandardDSNT(temperature=0.05)  # type: ignore[call-arg]
    dsnt = StandardDSNT()
    assert not hasattr(dsnt, "temperature")

    probabilities = standard_spatial_softmax(logits)
    expected_probabilities = F.softmax(logits.reshape(1, 2, 4), dim=-1).reshape_as(logits)
    torch.testing.assert_close(probabilities, expected_probabilities, rtol=0, atol=0)
    torch.testing.assert_close(
        probabilities.sum(dim=(-1, -2)),
        torch.ones((1, 2), dtype=logits.dtype),
    )

    x_axis = torch.tensor([-1.0, 1.0], dtype=logits.dtype).view(1, 1, 1, 2)
    y_axis = torch.tensor([-1.0, 1.0], dtype=logits.dtype).view(1, 1, 2, 1)
    expected = torch.stack(
        (
            (expected_probabilities * x_axis).sum(dim=(-1, -2)),
            (expected_probabilities * y_axis).sum(dim=(-1, -2)),
        ),
        dim=-1,
    )
    actual = dsnt(logits)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_strict_supervised_loss_is_only_masked_heatmap_mse() -> None:
    predicted = torch.tensor(
        [[[[1.0, 3.0]], [[10.0, 10.0]], [[2.0, 4.0]]]],
        requires_grad=True,
    )
    target = torch.zeros_like(predicted)
    valid = torch.tensor([[True, False, True]])
    loss = strict_supervised_heatmap_mse(predicted, target, valid)
    expected = torch.tensor(((1.0**2 + 3.0**2) / 2 + (2.0**2 + 4.0**2) / 2) / 2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad[:, 1]) == 0
    assert float(predicted.grad[:, (0, 2)].abs().sum()) > 0.0


def test_strict_real_hrnet_selected_feature_runs_on_small_synthetic_input() -> None:
    model = StrictGeoEquiLDModel().eval()
    assert model.feature_contract.backbone_name == "hrnet_w32"
    assert model.feature_contract.channels == (32,)
    assert model.feature_contract.reductions == (4,)
    with torch.inference_mode():
        features = model.extract_high_resolution_features(torch.zeros((1, 1, 64, 64)))
    assert features.shape == (1, 32, 16, 16)
    assert torch.isfinite(features).all()
