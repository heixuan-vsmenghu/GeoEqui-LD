from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from geoequi_ld.models import DSNT
from geoequi_ld.training.phase1c_config import load_phase1c_config
from geoequi_ld.training.phase1c_runners import (
    _fit_phase1c_supervised,
    _formal_allocation_outcome,
    _ledger_run_binding,
    _metric_with_rates,
    _operator_evidence_passed,
    _require_current_ledger_binding,
    _runtime_source_binding,
    _seed_phase1c,
    build_phase1c_initialization,
    require_phase1c_fresh_output,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase1c_seed_policy_keeps_determinism_enabled_warn_only() -> None:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        _seed_phase1c(42)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def test_operator_evidence_requires_real_deform_and_complete_gradients() -> None:
    evidence: dict[str, Any] = {
        "input_shape": [1, 32, 32, 32],
        "offset_shape": [1, 18, 32, 32],
        "mask_logits_shape": [1, 9, 32, 32],
        "mask_shape": [1, 9, 32, 32],
        "output_shape": [1, 32, 32, 32],
        "initial_offset_max_abs": 0.0,
        "initial_mask_logits_max_abs": 0.0,
        "initial_mask_min": 0.5,
        "initial_mask_max": 0.5,
        "offset_predictor_gradient_l1": 1.0,
        "mask_predictor_gradient_l1": 1.0,
        "deform_weight_gradient_l1": 1.0,
        "spatial_attention_gradient_l1": 1.0,
        "all_values_finite": True,
        "all_enhancer_gradients_finite": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": True,
        "strict_deterministic_deform_backward_available": False,
        "actual_operator": "torchvision.ops.deform_conv.DeformConv2d",
        "ordinary_conv_fallback": False,
    }

    assert _operator_evidence_passed(evidence)
    for name, replacement in (
        ("actual_operator", "torch.nn.modules.conv.Conv2d"),
        ("ordinary_conv_fallback", True),
        ("mask_predictor_gradient_l1", 0.0),
    ):
        drifted = {**evidence, name: replacement}
        assert not _operator_evidence_passed(drifted)


def test_runtime_source_binding_handles_clean_git_diff_as_text() -> None:
    binding = _runtime_source_binding(ROOT)

    assert binding["git_head"] != "unavailable"
    assert binding["tracked_source_diff_size_bytes"] >= 0
    assert len(binding["tracked_source_diff_sha256"]) == 64


def _metrics(value: float) -> dict[str, Any]:
    return {
        "total_loss": value,
        "heatmap_mse": value,
        "coordinate_smooth_l1": value,
        "distribution_js": value,
        "MRE_PS1": value + 1,
        "MRE_PS2": value + 2,
        "MRE_FH1": value + 3,
        "MRE_ALL": value + 2,
        "n_samples": 1,
        "decoder": "dsnt",
        "n_valid_aop": 1,
        "n_evaluable_aop": 1,
        "aop_invalid_prediction_count": 0,
        "aop_mae_valid_deg": value + 4,
        "aop_mae_deg": value + 4,
    }


def test_phase1c_output_guard_is_new_phase_only(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    destination = repository / "runs" / "phase1c" / "new_run"
    created = require_phase1c_fresh_output(destination, repository_root=repository)
    assert created == destination.resolve()
    with pytest.raises(FileExistsError, match="reuse"):
        (created / "marker").write_text("x", encoding="utf-8")
        require_phase1c_fresh_output(destination, repository_root=repository)
    with pytest.raises(PermissionError, match="Phase 1C output"):
        require_phase1c_fresh_output(
            repository / "runs" / "phase1b" / "wrong",
            repository_root=repository,
        )
    with pytest.raises(PermissionError, match="test/testing"):
        require_phase1c_fresh_output(
            repository / "runs" / "phase1c" / "testing",
            repository_root=repository,
        )


def test_seed42_initialization_copies_only_h2_base() -> None:
    h2, h3, metadata = build_phase1c_initialization()

    assert metadata["base_state_values_equal"] == {
        "backbone": True,
        "ps_decoder": True,
        "fh_decoder": True,
    }
    assert metadata["base_parameter_storage_aliased"] is False
    assert metadata["h2_trainable_parameters"] == 29_332_275
    assert metadata["h3_trainable_parameters"] == 29_372_695
    assert metadata["additional_trainable_parameters"] == 40_420
    assert metadata["complete_function_equivalent"] is False
    assert metadata["enhancer_initialization"] == h3.initialization_summary
    assert next(h2.backbone.parameters()).data_ptr() != next(h3.backbone.parameters()).data_ptr()


def test_metric_rates_keep_valid_and_penalized_aop_distinct() -> None:
    metrics = _metrics(1.0)
    metrics["n_valid_aop"] = 3
    metrics["n_evaluable_aop"] = 4
    metrics["aop_mae_valid_deg"] = 5.0
    metrics["aop_mae_deg"] = 48.75

    enriched = _metric_with_rates(metrics)

    assert enriched["aop_valid_rate"] == 0.75
    assert enriched["aop_mae_valid_deg"] == 5.0
    assert enriched["selection_aop_penalized_deg"] == 48.75


def test_formal_allocation_does_not_confuse_training_guard_stop_with_overrun() -> None:
    outcome = _formal_allocation_outcome(
        elapsed_seconds=8000.0,
        allocated_seconds=9000.0,
        ledger_binding={
            "entry": {
                "elapsed_seconds": 8000.1,
                "allocated_seconds": 9000.0,
                "details": {
                    "allocation_exceeded": False,
                    "aggregate_limit_exceeded": False,
                },
            }
        },
    )
    assert outcome == {
        "formal_allocation_exceeded": False,
        "aggregate_gpu_cap_exceeded": False,
        "within_runtime_allocation": True,
    }


def test_ledger_binding_supports_retry_but_binds_exact_completed_attempt(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    ledger_path = repository / "runs" / "phase1c" / "gpu_budget.json"
    ledger_path.parent.mkdir(parents=True)
    runs = [
        {
            "name": "P1C_deform_cuda_probe",
            "allocated_seconds": 300.0,
            "started_at_utc": "first",
            "elapsed_seconds": 2.0,
            "status": "failed",
            "finished_at_utc": "first-end",
            "details": {
                "gate": "FAIL",
                "allocation_exceeded": False,
                "aggregate_limit_exceeded": False,
            },
        },
        {
            "name": "P1C_deform_cuda_probe",
            "allocated_seconds": 300.0,
            "started_at_utc": "second",
            "elapsed_seconds": 2.0,
            "status": "completed",
            "finished_at_utc": "second-end",
            "details": {
                "gate": "PASS",
                "allocation_exceeded": False,
                "aggregate_limit_exceeded": False,
            },
        },
    ]
    payload = {
        "schema_version": 1,
        "total_limit_seconds": 10800.0,
        "runs": runs,
        "active_run": None,
    }
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    binding = _ledger_run_binding(payload, "P1C_deform_cuda_probe")
    config = load_phase1c_config(ROOT / "configs" / "phase1c_specialized_enhancers.yaml")

    assert binding["run_index"] == 1
    _require_current_ledger_binding(
        binding,
        name="P1C_deform_cuda_probe",
        ledger_path=ledger_path,
        repository_root=repository,
        config=config,
        required_details={"gate": "PASS"},
    )
    with pytest.raises(PermissionError, match="do not prove"):
        _require_current_ledger_binding(
            binding,
            name="P1C_deform_cuda_probe",
            ledger_path=ledger_path,
            repository_root=repository,
            config=config,
            required_details={"gate": "FAIL"},
        )


def test_fit_records_eval_mode_train_and_validation_every_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geoequi_ld.training.phase1c_runners as runners

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    evaluate_calls: list[object] = []
    histories: list[list[dict[str, Any]]] = []
    saved: list[str] = []

    monkeypatch.setattr(
        runners,
        "train_one_epoch",
        lambda *args, **kwargs: {
            "total_loss": 1.0,
            "heatmap_mse": 1.0,
            "coordinate_smooth_l1": 0.0,
            "distribution_js": 0.0,
            "batches": 1.0,
        },
    )

    def fake_evaluate(_model: nn.Module, loader: object, **_kwargs: Any) -> dict[str, Any]:
        evaluate_calls.append(loader)
        return _metrics(1.0 if loader == "train_eval" else 2.0)

    monkeypatch.setattr(runners, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(
        runners,
        "write_history_csv",
        lambda _path, rows: histories.append([dict(row) for row in rows]),
    )
    monkeypatch.setattr(
        runners,
        "save_checkpoint",
        lambda path, **_kwargs: saved.append(Path(path).name),
    )
    monkeypatch.setattr(
        runners,
        "_save_model_only",
        lambda path, **_kwargs: saved.append(Path(path).name),
    )
    config = SimpleNamespace(epochs=2, seed=42)

    summary = _fit_phase1c_supervised(
        model,
        "train_loader",  # type: ignore[arg-type]
        "train_eval",  # type: ignore[arg-type]
        "validation",  # type: ignore[arg-type]
        optimizer,
        dsnt=DSNT(temperature=0.05, align_corners=True),
        device=torch.device("cpu"),
        config=config,
        output_dir=tmp_path,
        checkpoint_config={"testing_frozen": True},
        max_runtime_seconds=3000.0,
        milestone_epochs=(1, 2),
        train_generator=torch.Generator().manual_seed(42),
    )

    assert summary["status"] == "completed"
    assert summary["epochs_completed"] == 2
    assert summary["per_epoch_train_metrics_semantics"] == (
        "post_epoch_eval_mode_full_train_split"
    )
    assert evaluate_calls == ["train_eval", "validation", "train_eval", "validation"]
    assert histories[-1][-1]["train_MRE_PS2"] == 3.0
    assert histories[-1][-1]["val_MRE_PS2"] == 4.0
    assert histories[-1][-1]["train_aop_valid_rate"] == 1.0
    assert histories[-1][-1]["val_selection_aop_penalized_deg"] == 6.0
    assert saved.count("last.pt") == 2
    assert "epoch_001_model_only.pt" in saved
    assert "epoch_002_model_only.pt" in saved


def test_fit_does_not_start_first_epoch_below_safety_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geoequi_ld.training.phase1c_runners as runners

    monkeypatch.setattr(
        runners,
        "train_one_epoch",
        lambda *args, **kwargs: pytest.fail("first epoch must not start"),
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="did not permit one complete epoch"):
        _fit_phase1c_supervised(
            model,
            "train_loader",  # type: ignore[arg-type]
            "train_eval",  # type: ignore[arg-type]
            "validation",  # type: ignore[arg-type]
            optimizer,
            dsnt=DSNT(temperature=0.05, align_corners=True),
            device=torch.device("cpu"),
            config=SimpleNamespace(epochs=1, seed=42),
            output_dir=tmp_path,
            checkpoint_config={"testing_frozen": True},
            max_runtime_seconds=1199.0,
            milestone_epochs=(1,),
            train_generator=torch.Generator().manual_seed(42),
        )
