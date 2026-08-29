from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from geoequi_ld.training.phase1a_config import load_phase1a_hrnet_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase1a_hrnet_shared.yaml"


def _payload() -> dict[str, object]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "phase1a.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_phase1a_hrnet_config_locks_model_optimizer_and_resources() -> None:
    config = load_phase1a_hrnet_config(CONFIG_PATH)
    assert config.testing_frozen
    assert config.training.input_size_hw == (512, 512)
    assert config.training.heatmap_size_hw == (256, 256)
    assert config.training.batch_size == 1
    assert config.training.epochs == 20
    assert config.model.timm_version == "1.0.28"
    assert config.model.feature_location == ""
    assert config.model.out_indices == (1,)
    assert config.model.feature_channels == 32
    assert config.model.feature_reduction == 4
    assert config.model.decoder_channels == (32, 16)
    assert config.model.decoder_normalization == "BatchNorm2d"
    assert config.optimizer.class_name == "Adam"
    assert config.optimizer.foreach is False
    assert config.resources.precision == "float32"
    assert config.resources.amp_enabled is False
    assert config.resources.allow_input_resize is False
    assert config.resources.require_full_input_first_step_probe is True


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        ("model", "out_indices", [0], ValueError),
        ("model", "feature_location", "incre", ValueError),
        ("model", "pretrained", True, ValueError),
        ("training", "input_size_hw", [256, 256], ValueError),
        ("training", "batch_size", 2, ValueError),
        ("optimizer", "foreach", True, ValueError),
        ("resources", "precision", "float16", ValueError),
        ("resources", "amp_enabled", True, ValueError),
        ("resources", "allow_input_resize", True, ValueError),
        (None, "testing_frozen", False, PermissionError),
    ],
)
def test_phase1a_hrnet_config_rejects_protocol_drift(
    tmp_path: Path,
    section: str | None,
    key: str,
    value: object,
    error: type[Exception],
) -> None:
    payload = copy.deepcopy(_payload())
    target = payload if section is None else payload[section]
    assert isinstance(target, dict)
    target[key] = value
    with pytest.raises(error):
        load_phase1a_hrnet_config(_write_payload(tmp_path, payload))


def test_phase1a_hrnet_config_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _payload()
    model = payload["model"]
    assert isinstance(model, dict)
    model["silent_amp"] = True
    with pytest.raises(ValueError, match="Unknown Phase1AModelConfig fields"):
        load_phase1a_hrnet_config(_write_payload(tmp_path, payload))
