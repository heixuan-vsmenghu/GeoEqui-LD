from __future__ import annotations

import pytest

from geoequi_ld.training.ablation import VARIANTS, apply_variant, assert_variant_weights
from geoequi_ld.training.config import SupervisedTrainingConfig


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("B0", (1.0, 0.0, 0.0)),
        ("B1", (1.0, 10.0, 0.0)),
        ("B2", (1.0, 10.0, 1.0)),
    ],
)
def test_ablation_variant_locks_exact_weights(
    name: str,
    expected: tuple[float, float, float],
) -> None:
    config = apply_variant(SupervisedTrainingConfig(), name)
    actual = (
        config.heatmap_loss_weight,
        config.coordinate_loss_weight,
        config.distribution_loss_weight,
    )
    assert actual == expected
    assert_variant_weights(config, name)
    assert VARIANTS[name].name == name


def test_variant_identity_rejects_hand_edited_weights() -> None:
    with pytest.raises(ValueError, match="B0 weights"):
        assert_variant_weights(SupervisedTrainingConfig(), "B0")
