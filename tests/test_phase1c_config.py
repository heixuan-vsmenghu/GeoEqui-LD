from __future__ import annotations

from pathlib import Path

import pytest
import torch

from geoequi_ld.training.phase1c_config import (
    build_phase1c_adam,
    load_phase1c_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase1c_config_locks_specialized_architecture_and_budget() -> None:
    config = load_phase1c_config(ROOT / "configs" / "phase1c_specialized_enhancers.yaml")

    assert config.experiment_name == "H3_specialized_B2_seed42_16e"
    assert config.testing_frozen is True
    assert config.training.epochs == 16
    assert config.training.keypoint_order == ("PS1", "PS2", "FH1")
    assert config.training.coordinate_loss_weight == 10.0
    assert config.training.distribution_loss_weight == 1.0
    assert config.model.class_name == "HRNetW32SpecializedHeatmap"
    assert config.model.torchvision_version == "0.20.1"
    assert config.model.ps_deformable_operator == "torchvision.ops.DeformConv2d"
    assert (config.model.ps_offset_channels, config.model.ps_mask_channels) == (18, 9)
    assert config.model.ps_normalization == config.model.fh_normalization == "LayerNorm2d"
    assert config.model.fh_aspp_dilations == (1, 3, 6)
    assert config.resources.operator_probe_max_seconds == 300
    assert config.resources.tiny_max_steps == 500
    assert config.resources.tiny_max_seconds == 2400
    assert config.resources.formal_max_seconds == 9000
    assert config.resources.total_gpu_max_seconds == 10800
    assert config.resources.milestone_epochs == (1, 3, 5, 10, 16)


def test_phase1c_adam_is_locked_to_non_foreach() -> None:
    config = load_phase1c_config(ROOT / "configs" / "phase1c_specialized_enhancers.yaml")
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = build_phase1c_adam([parameter], config)

    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["weight_decay"] == 0.0001
    assert optimizer.defaults["foreach"] is False


def test_phase1c_config_rejects_contract_drift(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "phase1c_specialized_enhancers.yaml").read_text(
        encoding="utf-8"
    )
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(source.replace("epochs: 16", "epochs: 17"), encoding="utf-8")

    with pytest.raises(ValueError, match="training contract drifted"):
        load_phase1c_config(drifted)


@pytest.mark.parametrize("replacement", ['"false"', "false", "1"])
def test_phase1c_config_requires_literal_boolean_testing_freeze(
    tmp_path: Path,
    replacement: str,
) -> None:
    source = (ROOT / "configs" / "phase1c_specialized_enhancers.yaml").read_text(
        encoding="utf-8"
    )
    drifted = tmp_path / "testing-drift.yaml"
    drifted.write_text(
        source.replace("testing_frozen: true", f"testing_frozen: {replacement}"),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="boolean true"):
        load_phase1c_config(drifted)
