from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from geoequi_ld.geometry.coordinates import pixel_to_normalized
from geoequi_ld.models.dsnt import DSNT
from geoequi_ld.models.unet import HeatmapUNet, count_trainable_parameters
from geoequi_ld.training.phase1a_runners import (
    _batch_norm_buffers_match,
    _capture_batch_norm_buffers,
    _restore_batch_norm_buffers,
    a4_diagnostic_completed,
    a4_learning_outcome,
    b4_gate_passed,
    load_verified_phase1a_data,
    phase1a_training_config,
    require_passed_gate_artifact,
    require_private_fresh_output,
    run_tiny_gate,
    save_tiny_prediction_visualizations,
    select_preregistered_tiny_indices,
)
from scripts.probe_phase1a_hrnet import build_parser as build_b3_parser
from scripts.run_phase1a_tiny_gate import build_parser as build_tiny_parser
from scripts.train_phase1a_hrnet import build_parser as build_formal_parser


def _strict_metrics(*, mre_all: float = 5.0) -> dict[str, float | int]:
    return {
        "MRE_PS1": 1.0,
        "MRE_PS2": 2.0,
        "MRE_FH1": 3.0,
        "MRE_ALL": mre_all,
        "aop_mae_deg": 2.0,
        "mean_total_loss": 0.1,
        "raw_heatmap_min": -1.0,
        "raw_heatmap_max": 1.0,
        "raw_heatmap_mean": 0.0,
        "raw_heatmap_std": 0.5,
        "probability_peak_mean": 0.9,
        "probability_peak_min": 0.8,
        "n_samples": 4,
        "n_evaluable_aop": 4,
        "n_valid_aop": 4,
        "aop_invalid_prediction_count": 0,
        "coordinate_error_count": 0,
        "nonfinite_count": 0,
    }


def test_preregistered_tiny_indices_are_stable_and_fail_on_split_drift() -> None:
    first = select_preregistered_tiny_indices(300)
    second = select_preregistered_tiny_indices(300)
    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4
    assert all(0 <= index < 300 for index in first)
    with pytest.raises(PermissionError, match="preregistered"):
        select_preregistered_tiny_indices(299)


def test_locked_loss_recipes() -> None:
    b0 = phase1a_training_config(gate="A4_unet_B0")
    b2 = phase1a_training_config(gate="B4_hrnet_B2")
    assert (b0.heatmap_loss_weight, b0.coordinate_loss_weight, b0.distribution_loss_weight) == (
        1.0,
        0.0,
        0.0,
    )
    assert (b2.heatmap_loss_weight, b2.coordinate_loss_weight, b2.distribution_loss_weight) == (
        1.0,
        10.0,
        1.0,
    )
    assert b0.seed == b2.seed == 42
    assert b0.batch_size == b2.batch_size == 1
    assert b0.base_channels == b2.base_channels == 8
    assert count_trainable_parameters(HeatmapUNet(base_channels=b0.base_channels)) == 484_171


def test_private_output_is_fresh_ignored_and_protects_phase06(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    allowed = root / "runs" / "phase1a" / "B3"
    created = require_private_fresh_output(allowed, repository_root=root)
    assert created == allowed.resolve()
    (created / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        require_private_fresh_output(allowed, repository_root=root)
    with pytest.raises(PermissionError, match="protected"):
        require_private_fresh_output(root / "runs" / "phase06" / "new", repository_root=root)
    with pytest.raises(PermissionError, match="runs/ or artifacts"):
        require_private_fresh_output(root / "reports" / "phase1a", repository_root=root)
    with pytest.raises(PermissionError, match="test/testing"):
        require_private_fresh_output(
            root / "runs" / "phase1a" / "testing",
            repository_root=root,
        )


def test_b4_gate_is_strict_and_a4_remains_diagnostic() -> None:
    assert b4_gate_passed(_strict_metrics())
    assert not b4_gate_passed(_strict_metrics(mre_all=5.0001))
    invalid = _strict_metrics()
    invalid["n_valid_aop"] = 3
    assert not b4_gate_passed(invalid)
    nonfinite = _strict_metrics()
    nonfinite["MRE_PS1"] = float("nan")
    assert not b4_gate_passed(nonfinite)
    assert a4_diagnostic_completed(_strict_metrics(mre_all=100.0))
    assert a4_learning_outcome(_strict_metrics(mre_all=100.0)) == "not_learned"
    assert (
        a4_learning_outcome(_strict_metrics(mre_all=5.0))
        == "candidate_requires_peak_and_overlay_review"
    )


def test_train_mode_batch_norm_diagnostic_can_be_rolled_back() -> None:
    model = nn.Sequential(nn.BatchNorm2d(2), nn.Conv2d(2, 1, 1))
    state = _capture_batch_norm_buffers(model)
    model.train()
    with torch.inference_mode():
        model(torch.ones(1, 2, 4, 4))
    assert not _batch_norm_buffers_match(model, state)
    _restore_batch_norm_buffers(model, state)
    assert _batch_norm_buffers_match(model, state)


def test_local_config_rejects_testing_before_file_access(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        """phase: phase0.5
splits:
  train: {image_dir: missing, labels_csv: missing.csv, fh1_column: FH1, expected_fingerprint: {}}
  testing: {image_dir: missing, labels_csv: missing.csv, fh1_column: FH1, expected_fingerprint: {}}
""",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="testing"):
        load_verified_phase1a_data(local)


def test_synthetic_visualizations_stay_in_private_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.figure

    class FixedHeatmaps(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            template = torch.zeros(1, 3, 16, 16)
            template[0, 0, 8, 4] = 10
            template[0, 1, 8, 10] = 10
            template[0, 2, 3, 6] = 10
            self.register_buffer("template", template)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.template.expand(inputs.shape[0], -1, -1, -1)

    saved_paths: list[Path] = []

    def fake_savefig(
        _figure: matplotlib.figure.Figure,
        path: str | Path,
        **_kwargs: object,
    ) -> None:
        destination = Path(path)
        destination.write_bytes(b"synthetic-figure")
        saved_paths.append(destination.resolve())

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)
    config = replace(
        phase1a_training_config(gate="A4_unet_B0"),
        input_size_hw=(32, 32),
        heatmap_size_hw=(16, 16),
    )
    points_input = torch.tensor([[[8.0, 16.0], [20.0, 16.0], [12.0, 6.0]]])
    batch = {
        "filename": ["synthetic.png"],
        "image": torch.zeros(1, 1, 32, 32),
        "points_input_px": points_input,
        "points_normalized": pixel_to_normalized(
            points_input,
            (32, 32),
            align_corners=True,
        ),
    }
    run_root = tmp_path / "repo" / "runs" / "phase1a" / "tiny"
    run_root.mkdir(parents=True)
    result = save_tiny_prediction_visualizations(
        FixedHeatmaps(),
        [batch],  # type: ignore[arg-type]
        dsnt=DSNT(temperature=0.05, align_corners=True),
        device=torch.device("cpu"),
        config=config,
        private_run_root=run_root,
        repository_root=tmp_path / "repo",
    )
    predictions = (run_root / "predictions").resolve()
    assert result["visualization_count"] == 1
    assert saved_paths == [(predictions / "sample_00.png").resolve()]
    assert all(path.is_relative_to(predictions) for path in saved_paths)
    assert list(tmp_path.rglob("*.png")) == [predictions / "sample_00.png"]
    assert (predictions / "DO_NOT_COMMIT.txt").is_file()

    with pytest.raises(PermissionError, match="ignored runs/ or artifacts"):
        save_tiny_prediction_visualizations(
            FixedHeatmaps(),
            [batch],  # type: ignore[arg-type]
            dsnt=DSNT(temperature=0.05, align_corners=True),
            device=torch.device("cpu"),
            config=config,
            private_run_root=tmp_path / "repo" / "reports" / "phase1a",
            repository_root=tmp_path / "repo",
        )
    assert list(tmp_path.rglob("*.png")) == [predictions / "sample_00.png"]


def test_gate_artifacts_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    gate_dir = root / "runs" / "phase1a"
    gate_dir.mkdir(parents=True)
    b3 = gate_dir / "b3.json"
    b3_checks = {
        "fp32_batch1_512": True,
        "output_shape": True,
        "dsnt_shape": True,
        "stage4_four_scale_shapes": True,
        "stage4_four_scale_input_gradients": True,
        "losses_finite": True,
        "gradients_finite": True,
        "backbone_nonzero_gradient": True,
        "decoder_nonzero_gradient": True,
        "backbone_updated": True,
        "decoder_updated": True,
        "train_eval_finite": True,
        "batch_norm_eval_switched": True,
        "batch_norm_train_switched": True,
        "checkpoint_parameter_roundtrip": True,
        "checkpoint_output_roundtrip": True,
        "within_allocated_time": True,
    }
    b3.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": "B3",
                "gate": "PASS",
                "status": "completed",
                "checks": b3_checks,
                "feature_contract": {
                    "timm_version": "1.0.28",
                    "backbone_name": "hrnet_w32",
                    "feature_location": "",
                    "out_indices": [1],
                    "channels": [32],
                    "reductions": [4],
                },
                "stage4": {
                    "output_shapes": [
                        [1, 32, 128, 128],
                        [1, 64, 64, 64],
                        [1, 128, 32, 32],
                        [1, 256, 16, 16],
                    ],
                    "input_gradient_l1": [1.0, 2.0, 3.0, 4.0],
                },
            }
        ),
        encoding="utf-8",
    )
    assert require_passed_gate_artifact(
        b3,
        expected_gate_id="B3",
        repository_root=root,
    )["gate"] == "PASS"

    b4 = gate_dir / "b4.json"
    b4.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": "B4_hrnet_B2",
                "gate": "PASS",
                "status": "budget_exhausted",
                "eval_mode": _strict_metrics(mre_all=6.0),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="execution-integrity"):
        require_passed_gate_artifact(
            b4,
            expected_gate_id="B4_hrnet_B2",
            repository_root=root,
        )

    valid_b4 = {
        "schema_version": 1,
        "gate_id": "B4_hrnet_B2",
        "gate": "PASS",
        "status": "completed",
        "within_total_allocation": True,
        "batch_norm_buffers_restored_after_train_mode_diagnostic": True,
        "checkpoint_saved_in_eval_mode": True,
        "augmentation": "disabled",
        "batch_size": 1,
        "precision": "float32",
        "eval_mode": _strict_metrics(),
        "visualization": {
            "programmatic_check_passed": True,
            "manual_review_status": "passed",
        },
    }
    b4.write_text(json.dumps(valid_b4), encoding="utf-8")
    assert require_passed_gate_artifact(
        b4,
        expected_gate_id="B4_hrnet_B2",
        repository_root=root,
    )["gate"] == "PASS"
    valid_b4["eval_mode"] = _strict_metrics(mre_all=6.0)
    b4.write_text(json.dumps(valid_b4), encoding="utf-8")
    with pytest.raises(PermissionError, match="stored metrics"):
        require_passed_gate_artifact(
            b4,
            expected_gate_id="B4_hrnet_B2",
            repository_root=root,
        )
    valid_b4["eval_mode"] = _strict_metrics()
    valid_b4["batch_norm_buffers_restored_after_train_mode_diagnostic"] = False
    b4.write_text(json.dumps(valid_b4), encoding="utf-8")
    with pytest.raises(PermissionError, match="execution-integrity"):
        require_passed_gate_artifact(
            b4,
            expected_gate_id="B4_hrnet_B2",
            repository_root=root,
        )


def test_b4_cannot_start_without_an_explicit_passed_b3_artifact(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with pytest.raises(PermissionError, match="passed B3 artifact"):
        run_tiny_gate(
            gate="B4_hrnet_B2",
            local_config=root / "missing-local.yaml",
            hrnet_config=root / "missing-hrnet.yaml",
            output_dir=root / "runs" / "phase1a" / "b4",
            ledger_path=root / "runs" / "phase1a" / "ledger.json",
            repository_root=root,
        )
    assert not (root / "runs").exists()


def test_cli_parsers_lock_protocol_and_require_gates(tmp_path: Path) -> None:
    b3 = build_b3_parser().parse_args(["--output-dir", str(tmp_path / "runs" / "b3")])
    assert b3.max_seconds == 900.0

    tiny = build_tiny_parser().parse_args(
        [
            "--gate",
            "B4_hrnet_B2",
            "--b3-artifact",
            str(tmp_path / "runs" / "b3" / "b3_result.json"),
            "--output-dir",
            str(tmp_path / "runs" / "b4"),
        ]
    )
    assert tiny.gate == "B4_hrnet_B2"
    assert tiny.b3_artifact == tmp_path / "runs" / "b3" / "b3_result.json"

    formal = build_formal_parser().parse_args(
        [
            "--b3-artifact",
            "b3.json",
            "--b4-artifact",
            "b4.json",
            "--output-dir",
            str(tmp_path / "runs" / "formal"),
        ]
    )
    assert formal.b3_artifact == Path("b3.json")
    assert formal.b4_artifact == Path("b4.json")
