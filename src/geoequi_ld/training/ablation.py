"""Pre-registered Phase 0.5 supervised ablation identities."""

from __future__ import annotations

from dataclasses import dataclass, replace

from geoequi_ld.training.config import SupervisedTrainingConfig


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    heatmap_weight: float
    coordinate_weight: float
    distribution_weight: float


VARIANTS: dict[str, AblationVariant] = {
    "B0": AblationVariant("B0", "heatmap MSE", 1.0, 0.0, 0.0),
    "B1": AblationVariant("B1", "heatmap MSE + coordinate SmoothL1", 1.0, 10.0, 0.0),
    "B2": AblationVariant(
        "B2",
        "heatmap MSE + coordinate SmoothL1 + distribution JS",
        1.0,
        10.0,
        1.0,
    ),
}


def normalize_variant(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in VARIANTS:
        raise ValueError(f"Unknown Phase 0.5 variant: {value!r}")
    return normalized


def apply_variant(
    config: SupervisedTrainingConfig,
    variant_name: str,
) -> SupervisedTrainingConfig:
    """Return a config whose loss weights are locked to the named variant."""

    variant = VARIANTS[normalize_variant(variant_name)]
    configured = replace(
        config,
        heatmap_loss_weight=variant.heatmap_weight,
        coordinate_loss_weight=variant.coordinate_weight,
        distribution_loss_weight=variant.distribution_weight,
    )
    configured.validate()
    return configured


def assert_variant_weights(config: SupervisedTrainingConfig, variant_name: str) -> None:
    """Reject an accidental or hand-edited loss combination."""

    variant = VARIANTS[normalize_variant(variant_name)]
    actual = (
        config.heatmap_loss_weight,
        config.coordinate_loss_weight,
        config.distribution_loss_weight,
    )
    expected = (
        variant.heatmap_weight,
        variant.coordinate_weight,
        variant.distribution_weight,
    )
    if actual != expected:
        raise ValueError(f"{variant.name} weights must be {expected}, got {actual}")
