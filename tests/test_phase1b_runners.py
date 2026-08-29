from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from geoequi_ld.training.checkpoints import save_checkpoint
from geoequi_ld.training.phase1b_config import load_phase1b_decoder_config
from geoequi_ld.training.phase1b_runners import (
    _capture_resume_state,
    _compare_replay_metrics,
    _evaluate_key_checkpoints,
    _expected_h1_static_contract,
    _finish_ledger,
    _formal_runtime_allocation_outcome,
    _ledger_run_binding,
    _phase1b_ledger,
    _save_model_only,
    audit_h1_comparability,
    create_phase1b_tiny_review,
    load_verified_phase1b_data,
    phase1b_training_config,
    require_passed_phase1b_tiny_artifact,
    require_phase1b_fresh_output,
    validate_h1_replay_artifact,
)
from scripts.review_phase1b_split_tiny import build_parser as build_review_parser
from scripts.run_phase1b_split_tiny import build_parser as build_tiny_parser
from scripts.train_phase1b_split import build_parser as build_formal_parser
from scripts.verify_phase1b_h1_replay import build_parser as build_replay_parser

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "phase1b_decoder_control.yaml"


def _strict_tiny_metrics() -> dict[str, float | int]:
    return {
        "MRE_PS1": 1.0,
        "MRE_PS2": 2.0,
        "MRE_FH1": 3.0,
        "MRE_ALL": 2.0,
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


def test_phase1b_config_locks_only_decoder_control() -> None:
    config = load_phase1b_decoder_config(CONFIG_PATH)
    assert config.experiment_name == "H2_split_B2_seed42_20e"
    assert config.testing_frozen
    assert config.training.keypoint_order == ("PS1", "PS2", "FH1")
    assert (
        config.training.heatmap_loss_weight,
        config.training.coordinate_loss_weight,
        config.training.distribution_loss_weight,
    ) == (1.0, 10.0, 1.0)
    assert config.model.class_name == "HRNetW32SplitHeatmap"
    assert config.model.ps_out_channels == 2
    assert config.model.fh_out_channels == 1
    assert config.model.shared_initialization_method == "from_shared"
    assert config.resources.milestone_epochs == (1, 3, 5, 10, 20)
    assert config.resources.formal_max_seconds == 7200
    assert config.resources.total_gpu_max_seconds == 10800
    assert config.optimizer.foreach is False


def test_phase1b_config_fails_closed_on_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["testing_frozen"] = False
    path = tmp_path / "phase1b.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="testing frozen"):
        load_phase1b_decoder_config(path)

    payload["testing_frozen"] = True
    payload["training"]["coordinate_loss_weight"] = 9.0
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol drifted"):
        load_phase1b_decoder_config(path)

    payload["training"]["coordinate_loss_weight"] = 10.0
    payload["model"]["extra_decoder"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_phase1b_decoder_config(path)


def test_phase1b_training_recipe_reuses_one_three_channel_b2() -> None:
    config = phase1b_training_config()
    assert config.seed == 42
    assert config.batch_size == 1
    assert config.epochs == 20
    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.0001
    assert config.max_grad_norm == 5.0
    assert (
        config.heatmap_loss_weight,
        config.coordinate_loss_weight,
        config.distribution_loss_weight,
    ) == (1.0, 10.0, 1.0)


def test_phase1b_private_output_cannot_touch_previous_phases(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    output = root / "runs" / "phase1b" / "tiny"
    assert require_phase1b_fresh_output(output, repository_root=root) == output.resolve()
    (output / "keep").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        require_phase1b_fresh_output(output, repository_root=root)
    with pytest.raises(PermissionError, match="runs/phase1b"):
        require_phase1b_fresh_output(
            root / "runs" / "phase1a" / "new",
            repository_root=root,
        )


def test_phase1b_data_contract_must_equal_canonical_local_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geoequi_ld.training.phase1b_runners as runners

    root = tmp_path / "repo"
    canonical_path = root / "configs" / "phase05_local.yaml"
    canonical_path.parent.mkdir(parents=True)
    canonical_path.write_text("canonical", encoding="utf-8")
    canonical_spec = SimpleNamespace(
        image_dir=root / "data" / "train",
        labels_csv=root / "data" / "train.csv",
        fh1_column="FH1",
        expected_fingerprint={"sample_count": 300},
    )
    validation_spec = SimpleNamespace(
        image_dir=root / "data" / "validation",
        labels_csv=root / "data" / "validation.csv",
        fh1_column="FH1",
        expected_fingerprint={"sample_count": 100},
    )
    verified = SimpleNamespace(
        specs={"train": canonical_spec, "validation": validation_spec},
        fingerprints={},
    )
    monkeypatch.setattr(runners, "load_verified_phase1a_data", lambda _path: verified)
    monkeypatch.setattr(
        runners,
        "load_phase05_local_splits",
        lambda _path: {"train": canonical_spec, "validation": validation_spec},
    )
    assert load_verified_phase1b_data(
        root / "equivalent.yaml",
        repository_root=root,
    ) is verified
    drifted = SimpleNamespace(**vars(canonical_spec))
    drifted.labels_csv = root / "data" / "other.csv"
    monkeypatch.setattr(
        runners,
        "load_phase05_local_splits",
        lambda _path: {"train": drifted, "validation": validation_spec},
    )
    with pytest.raises(PermissionError, match="canonical"):
        load_verified_phase1b_data(
            root / "equivalent.yaml",
            repository_root=root,
        )
    with pytest.raises(PermissionError, match="test/testing"):
        require_phase1b_fresh_output(
            root / "runs" / "phase1b" / "testing",
            repository_root=root,
        )


def test_split_tiny_review_binds_pending_result_and_four_overlays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geoequi_ld.training.phase1b_runners as runners

    root = tmp_path / "repo"
    artifact = root / "runs" / "phase1b" / "tiny" / "tiny_result.json"
    artifact.parent.mkdir(parents=True)
    predictions = artifact.parent / "predictions"
    predictions.mkdir()
    for index in range(4):
        (predictions / f"sample_{index:02d}.png").write_bytes(f"overlay-{index}".encode())
    payload = {
        "schema_version": 1,
        "gate_id": "P1B_split_tiny_B2",
        "gate": "PENDING_REVIEW",
        "programmatic_gate": "PASS",
        "status": "completed",
        "steps_completed": 500,
        "max_steps": 500,
        "within_total_allocation": True,
        "augmentation": "disabled",
        "batch_size": 1,
        "precision": "float32",
        "loss_application": "concatenate_[PS1,PS2,FH1]_then_existing_B2_once",
        "initialization": {"output_equivalent": True},
        "eval_mode": _strict_tiny_metrics(),
        "visualization": {
            "programmatic_check_passed": True,
            "manual_review_status": "pending",
            "visualization_count": 4,
            "private_relative_directory": "predictions",
        },
        "gpu_ledger_binding": {"bound": True},
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    context = {
        "data_fingerprint_digest": "data",
        "protocol_config_binding": {},
        "environment": {},
        "runtime_source_binding": {},
    }
    monkeypatch.setattr(runners, "_require_current_tiny_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(
        runners,
        "load_verified_phase1b_data",
        lambda *args, **kwargs: type("Verified", (), {"fingerprints": {}})(),
    )
    monkeypatch.setattr(runners, "fingerprint_digest", lambda _value: "data")
    monkeypatch.setattr(runners, "_require_ledger_binding_current", lambda *args, **kwargs: None)
    review_dir = root / "runs" / "phase1b" / "tiny_review"
    review = create_phase1b_tiny_review(
        tiny_artifact=artifact,
        local_config="local.yaml",
        phase1b_config=CONFIG_PATH,
        decision="PASS",
        output_dir=review_dir,
        repository_root=root,
    )
    assert review["decision"] == "PASS"
    evidence = require_passed_phase1b_tiny_artifact(
        artifact,
        review_dir / "tiny_review.json",
        data_fingerprint_digest="data",
        phase1b_config=CONFIG_PATH,
        ledger_path=root / "runs" / "phase1b" / "gpu_budget.json",
        repository_root=root,
    )
    assert evidence["bindings_recomputed"]

    (predictions / "sample_02.png").write_bytes(b"changed")
    with pytest.raises(PermissionError, match="overlay hashes"):
        require_passed_phase1b_tiny_artifact(
            artifact,
            review_dir / "tiny_review.json",
            data_fingerprint_digest="data",
            phase1b_config=CONFIG_PATH,
            ledger_path=root / "runs" / "phase1b" / "gpu_budget.json",
            repository_root=root,
        )

    payload["gate"] = "PASS"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="PENDING_REVIEW"):
        require_passed_phase1b_tiny_artifact(
            artifact,
            review_dir / "tiny_review.json",
            data_fingerprint_digest="data",
            phase1b_config=CONFIG_PATH,
            ledger_path=root / "runs" / "phase1b" / "gpu_budget.json",
            repository_root=root,
        )


def _write_synthetic_h1(root: Path, fingerprint: str) -> Path:
    run = root / "runs" / "phase1a" / "H1_shared_B2_seed42_20e"
    run.mkdir(parents=True)
    config = _expected_h1_static_contract(current_fingerprint_digest=fingerprint)
    config["runtime"] = {"allocated_seconds": 7200.0}
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "formal_result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "epochs_completed": 20,
                "epochs_requested": 20,
                "selection_order": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
            }
        ),
        encoding="utf-8",
    )
    metric_row = {
        "train_total_loss": 1.0,
        "train_heatmap_mse": 2.0,
        "train_coordinate_smooth_l1": 3.0,
        "train_distribution_js": 4.0,
        "train_batches": 300.0,
        "val_total_loss": 5.0,
        "val_heatmap_mse": 6.0,
        "val_coordinate_smooth_l1": 7.0,
        "val_distribution_js": 8.0,
        "val_MRE_PS1": 9.0,
        "val_MRE_PS2": 10.0,
        "val_MRE_FH1": 11.0,
        "val_MRE_ALL": 12.0,
        "val_n_samples": 100,
        "val_n_valid_aop": 100,
        "val_n_evaluable_aop": 100,
        "val_aop_invalid_prediction_count": 0,
        "val_aop_mae_valid_deg": 13.0,
        "val_aop_mae_deg": 13.0,
    }
    fieldnames = ["epoch", *metric_row]
    with (run / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({"epoch": epoch, **metric_row} for epoch in range(1, 21))
    (run / "best.pt").write_bytes(b"best")
    (run / "last.pt").write_bytes(b"last")
    return run


def test_h1_requires_matching_replay_before_strict_comparison(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    fingerprint = "current-data"
    h1 = _write_synthetic_h1(root, fingerprint)
    without_replay = audit_h1_comparability(
        h1_run_dir=h1,
        current_fingerprint_digest=fingerprint,
        repository_root=root,
    )
    assert all(without_replay["static_checks"].values())
    assert not without_replay["strictly_comparable_for_phase1b"]
    assert without_replay["comparison_classification"] == "historical_reference_only"

    replay = root / "runs" / "phase1b" / "replay" / "replay_result.json"
    replay.parent.mkdir(parents=True)
    replay.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": "H1_epoch1_deterministic_replay",
                "status": "completed",
                "comparison": "PASS",
                "reference_fingerprint_digest": fingerprint,
                "static_checks": {"invented": True},
                "strictly_comparable_for_phase1b": True,
            }
        ),
        encoding="utf-8",
    )
    with_replay = audit_h1_comparability(
        h1_run_dir=h1,
        current_fingerprint_digest=fingerprint,
        repository_root=root,
        replay_artifact=replay,
    )
    assert not with_replay["strictly_comparable_for_phase1b"]
    assert with_replay["comparison_classification"] == "historical_reference_only"
    with pytest.raises(PermissionError, match="raw train/validation metrics"):
        validate_h1_replay_artifact(
            replay_artifact=replay,
            h1_run_dir=h1,
            current_fingerprint_digest=fingerprint,
            repository_root=root,
            ledger_path=root / "runs" / "phase1b" / "gpu_budget.json",
            phase1b_config=CONFIG_PATH,
        )


def test_replay_metric_comparison_uses_preregistered_tolerance() -> None:
    train = {
        "total_loss": 1.0,
        "heatmap_mse": 2.0,
        "coordinate_smooth_l1": 3.0,
        "distribution_js": 4.0,
        "batches": 300.0,
    }
    validation = {
        "total_loss": 5.0,
        "heatmap_mse": 6.0,
        "coordinate_smooth_l1": 7.0,
        "distribution_js": 8.0,
        "MRE_PS1": 9.0,
        "MRE_PS2": 10.0,
        "MRE_FH1": 11.0,
        "MRE_ALL": 12.0,
        "n_samples": 100,
        "n_valid_aop": 100,
        "n_evaluable_aop": 100,
        "aop_invalid_prediction_count": 0,
        "aop_mae_valid_deg": 13.0,
        "aop_mae_deg": 13.0,
    }
    combined = {
        **{f"train_{key}": str(value) for key, value in train.items()},
        **{f"val_{key}": str(value) for key, value in validation.items()},
    }
    matches, details = _compare_replay_metrics(combined, train, validation)
    assert matches
    assert all(item["matches"] for item in details.values())
    combined["val_MRE_PS2"] = "10.1"
    matches, details = _compare_replay_metrics(combined, train, validation)
    assert not matches
    assert not details["val_MRE_PS2"]["matches"]


def test_h1_validator_recomputes_bindings_environment_and_each_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geoequi_ld.training.phase1b_runners as runners

    root = tmp_path / "repo"
    fingerprint = "current-data"
    h1 = _write_synthetic_h1(root, fingerprint)
    train = {
        "total_loss": 1.0,
        "heatmap_mse": 2.0,
        "coordinate_smooth_l1": 3.0,
        "distribution_js": 4.0,
        "batches": 300.0,
    }
    validation = {
        "total_loss": 5.0,
        "heatmap_mse": 6.0,
        "coordinate_smooth_l1": 7.0,
        "distribution_js": 8.0,
        "MRE_PS1": 9.0,
        "MRE_PS2": 10.0,
        "MRE_FH1": 11.0,
        "MRE_ALL": 12.0,
        "n_samples": 100,
        "n_valid_aop": 100,
        "n_evaluable_aop": 100,
        "aop_invalid_prediction_count": 0,
        "aop_mae_valid_deg": 13.0,
        "aop_mae_deg": 13.0,
    }
    with (h1 / "train_log.csv").open(encoding="utf-8", newline="") as handle:
        reference = next(csv.DictReader(handle))
    matches, comparisons = _compare_replay_metrics(reference, train, validation)
    assert matches
    static = audit_h1_comparability(
        h1_run_dir=h1,
        current_fingerprint_digest=fingerprint,
        repository_root=root,
    )
    environment = {"torch": "current"}
    sources = {"runner": {"sha256": "code"}}
    monkeypatch.setattr(runners, "_runtime_environment", lambda _root: environment)
    monkeypatch.setattr(runners, "_runtime_source_binding", lambda _root: sources)
    monkeypatch.setattr(runners, "_require_ledger_binding_current", lambda *args, **kwargs: None)
    artifact = root / "runs" / "phase1b" / "replay" / "replay_result.json"
    artifact.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "gate_id": "H1_epoch1_deterministic_replay",
        "status": "completed",
        "comparison": "PASS",
        "steps_completed": 300,
        "steps_requested": 300,
        "atol": 1.0e-6,
        "rtol": 1.0e-6,
        "reference_fingerprint_digest": fingerprint,
        "static_checks": static["static_checks"],
        "train_metrics": train,
        "validation_metrics": validation,
        "metric_comparisons": comparisons,
        "h1_reference_binding": runners._h1_reference_binding(h1),
        "protocol_config_binding": runners._file_binding(
            CONFIG_PATH,
            logical_name="configs/phase1b_decoder_control.yaml",
        ),
        "environment": environment,
        "runtime_source_binding": sources,
        "historical_environment_record_available": False,
        "environment_evidence_scope": "same_current_environment_exact_epoch1_replay",
        "strictly_comparable_for_phase1b": True,
        "comparison_classification": "frozen_shared_comparator",
        "total_elapsed_seconds": 10.0,
        "total_allocated_seconds": 600.0,
        "gpu_ledger_binding": {"bound": True},
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    validated = validate_h1_replay_artifact(
        replay_artifact=artifact,
        h1_run_dir=h1,
        current_fingerprint_digest=fingerprint,
        repository_root=root,
        ledger_path=root / "runs" / "phase1b" / "gpu_budget.json",
        phase1b_config=CONFIG_PATH,
    )
    assert validated["strictly_comparable_for_phase1b"]

    (h1 / "last.pt").write_bytes(b"mutated")
    with pytest.raises(PermissionError, match="h1_binding"):
        validate_h1_replay_artifact(
            replay_artifact=artifact,
            h1_run_dir=h1,
            current_fingerprint_digest=fingerprint,
            repository_root=root,
            ledger_path=root / "runs" / "phase1b" / "gpu_budget.json",
            phase1b_config=CONFIG_PATH,
        )


def test_model_only_milestone_has_no_optimizer_state(tmp_path: Path) -> None:
    model = nn.Conv2d(1, 3, 1)
    path = _save_model_only(tmp_path / "epoch_003_model_only.pt", model=model, epoch=3)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_kind"] == "model_only_milestone"
    assert payload["epoch"] == 3
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" not in payload


def test_phase1b_gpu_ledger_is_one_canonical_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config = load_phase1b_decoder_config(CONFIG_PATH)
    canonical = root / "runs" / "phase1b" / "gpu_budget.json"
    assert _phase1b_ledger(
        canonical,
        repository_root=root,
        config=config,
    ).path == canonical.resolve()
    with pytest.raises(PermissionError, match="canonical"):
        _phase1b_ledger(
            root / "runs" / "phase1b" / "alternate.json",
            repository_root=root,
            config=config,
        )
    ledger = _phase1b_ledger(canonical, repository_root=root, config=config)
    for attempt in range(2):
        ledger.begin("retryable", requested_limit_seconds=10.0)
        ledger.finish(
            "retryable",
            elapsed_seconds=1.0,
            status="completed",
            details={"attempt": attempt},
        )
    latest = _ledger_run_binding(ledger.snapshot(), "retryable")
    assert latest["run_index"] == 1
    assert len(latest["entry_sha256"]) == 64
    first = _ledger_run_binding(ledger.snapshot(), "retryable", run_index=0)
    assert first["entry"]["details"]["attempt"] == 0


def test_phase1b_ledger_closes_as_budget_exhausted_when_elapsed_exceeds_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geoequi_ld.training.phase1b_runners as runners

    root = tmp_path / "repo"
    config = load_phase1b_decoder_config(CONFIG_PATH)
    ledger = _phase1b_ledger(
        root / "runs" / "phase1b" / "gpu_budget.json",
        repository_root=root,
        config=config,
    )
    ledger.begin("bounded", requested_limit_seconds=1.0)
    monkeypatch.setattr(runners.time, "perf_counter", lambda: 2.0)
    _finish_ledger(
        ledger,
        "bounded",
        started=0.0,
        status="completed",
    )
    snapshot = ledger.snapshot()
    assert snapshot["active_run"] is None
    assert snapshot["runs"][-1]["status"] == "budget_exhausted"


def test_formal_runtime_allocation_distinguishes_training_subbudget_stop() -> None:
    binding = {
        "entry": {
            "status": "budget_exhausted",
            "elapsed_seconds": 6631.0,
            "allocated_seconds": 7200.0,
            "details": {
                "allocation_exceeded": False,
                "aggregate_limit_exceeded": False,
            },
        }
    }

    outcome = _formal_runtime_allocation_outcome(
        elapsed_seconds=6631.0,
        allocated_seconds=7200.0,
        ledger_binding=binding,
    )

    assert outcome == {
        "formal_allocation_exceeded": False,
        "aggregate_gpu_cap_exceeded": False,
        "within_runtime_allocation": True,
    }


@pytest.mark.parametrize(
    ("details", "elapsed", "expected_formal", "expected_aggregate"),
    [
        (
            {"allocation_exceeded": True, "aggregate_limit_exceeded": False},
            6631.0,
            True,
            False,
        ),
        (
            {"allocation_exceeded": False, "aggregate_limit_exceeded": True},
            6631.0,
            False,
            True,
        ),
        (
            {"allocation_exceeded": False, "aggregate_limit_exceeded": False},
            7200.1,
            True,
            False,
        ),
    ],
)
def test_formal_runtime_allocation_fails_on_actual_or_ledger_exceed_flags(
    details: dict[str, bool],
    elapsed: float,
    expected_formal: bool,
    expected_aggregate: bool,
) -> None:
    outcome = _formal_runtime_allocation_outcome(
        elapsed_seconds=elapsed,
        allocated_seconds=7200.0,
        ledger_binding={
            "entry": {
                "status": "budget_exhausted",
                "elapsed_seconds": min(elapsed, 7200.0),
                "allocated_seconds": 7200.0,
                "details": details,
            }
        },
    )
    assert outcome["formal_allocation_exceeded"] is expected_formal
    assert outcome["aggregate_gpu_cap_exceeded"] is expected_aggregate
    assert not outcome["within_runtime_allocation"]


def test_full_checkpoint_resume_state_captures_all_rng_streams() -> None:
    generator = torch.Generator().manual_seed(42)
    state = _capture_resume_state(generator)
    assert set(state) == {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "train_loader_generator_state",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "deterministic_algorithms",
    }
    torch.testing.assert_close(state["train_loader_generator_state"], generator.get_state())


def test_key_checkpoint_eval_recomputes_epoch_log_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geoequi_ld.training.phase1b_runners as runners

    model = nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    metrics = {"MRE_ALL": 1.0, "aop_mae_deg": 2.0, "decoder": "dsnt"}
    resume = _capture_resume_state(torch.Generator().manual_seed(42))
    for name, epoch in (("best", 1), ("last", 2)):
        save_checkpoint(
            tmp_path / f"{name}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config={"phase": "phase1b"},
            seed=42,
            metrics=metrics,
            extra={"resume_state": resume},
        )
    with (tmp_path / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "val_MRE_ALL", "val_aop_mae_deg", "val_decoder"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"epoch": 1, "val_MRE_ALL": 1.0, "val_aop_mae_deg": 2.0, "val_decoder": "dsnt"},
                {"epoch": 2, "val_MRE_ALL": 1.0, "val_aop_mae_deg": 2.0, "val_decoder": "dsnt"},
            ]
        )
    monkeypatch.setattr(runners, "evaluate_model", lambda *args, **kwargs: dict(metrics))
    result = _evaluate_key_checkpoints(
        model,
        optimizer,
        output=tmp_path,
        train_loader=None,  # type: ignore[arg-type]
        validation_loader=None,  # type: ignore[arg-type]
        dsnt=nn.Identity(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        config=None,
    )
    assert result["best"]["epoch"] == 1
    assert result["last"]["epoch"] == 2
    assert result["best"]["recomputed_validation_matches_checkpoint"]


def test_phase1b_cli_parsers_keep_artifacts_and_outputs_explicit(tmp_path: Path) -> None:
    tiny = build_tiny_parser().parse_args(["--output-dir", str(tmp_path / "tiny")])
    assert tiny.output_dir == tmp_path / "tiny"

    replay = build_replay_parser().parse_args(["--output-dir", str(tmp_path / "replay")])
    assert replay.h1_run_dir.name == "H1_shared_B2_seed42_20e"

    review = build_review_parser().parse_args(
        [
            "--tiny-artifact",
            "tiny.json",
            "--decision",
            "PASS",
            "--output-dir",
            str(tmp_path / "review"),
        ]
    )
    assert review.decision == "PASS"

    formal = build_formal_parser().parse_args(
        [
            "--tiny-artifact",
            "tiny.json",
            "--tiny-review-artifact",
            "review.json",
            "--h1-comparability-artifact",
            "comparison.json",
            "--output-dir",
            str(tmp_path / "formal"),
        ]
    )
    assert formal.tiny_artifact == Path("tiny.json")
    assert formal.tiny_review_artifact == Path("review.json")
    assert formal.h1_comparability_artifact == Path("comparison.json")
    assert formal.output_dir == tmp_path / "formal"


def test_resource_config_serializes_without_machine_paths() -> None:
    payload = asdict(load_phase1b_decoder_config(CONFIG_PATH).resources)
    serialized = json.dumps(payload)
    assert "10800" in serialized
    assert "\\" not in serialized
