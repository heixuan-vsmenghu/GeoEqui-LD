from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps
from geoequi_ld.diagnostics.phase1b_bn import (
    ModelStateSnapshot,
    TensorRecord,
    audit_state_transition,
    build_public_bn_aggregate,
    capture_model_state,
    evaluate_model_without_state_change,
    freeze_all_parameters,
    freeze_checkpoint_copies,
    reestimate_batch_norm_statistics,
    require_fresh_phase1b_private_output,
    require_fresh_phase1b_public_file,
)
from geoequi_ld.models.dsnt import DSNT
from geoequi_ld.training.config import SupervisedTrainingConfig
from scripts.diagnose_phase1b_bn import RUN_NAME, execute_with_ledger


class _ToyBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.register_buffer("calibration_guard", torch.tensor(7.0))

    def forward(self, image: Tensor) -> Tensor:
        return self.bn(self.conv(torch.nn.functional.avg_pool2d(image, 2)))


class _ToyReusedBNModel(_ToyBNModel):
    def forward(self, image: Tensor) -> Tensor:
        features = self.conv(torch.nn.functional.avg_pool2d(image, 2))
        return self.bn(self.bn(features))


def _config() -> SupervisedTrainingConfig:
    config = SupervisedTrainingConfig(
        input_size_hw=(32, 32),
        heatmap_size_hw=(16, 16),
        batch_size=1,
        epochs=1,
        base_channels=1,
        coordinate_loss_weight=10.0,
        distribution_loss_weight=1.0,
    )
    config.validate()
    return config


def _batches(count: int) -> list[dict[str, object]]:
    points_heatmap = torch.tensor([[[3.0, 5.0], [12.0, 5.0], [8.0, 12.0]]])
    heatmaps = generate_gaussian_heatmaps(
        points_heatmap.squeeze(0), size_hw=(16, 16), sigma=1.5
    ).unsqueeze(0)
    points_normalized = points_heatmap / 15.0 * 2.0 - 1.0
    points_original = points_heatmap / 15.0 * 31.0
    return [
        {
            "filename": [f"sample-{index}.png"],
            "image": torch.full((1, 1, 32, 32), 0.1 + index * 0.2),
            "heatmaps": heatmaps,
            "points_normalized": points_normalized,
            "points_original_px": points_original,
            "valid_mask": torch.ones((1, 3), dtype=torch.bool),
            "original_size_hw": torch.tensor([[32, 32]], dtype=torch.int64),
        }
        for index in range(count)
    ]


def _record(value: str) -> TensorRecord:
    return TensorRecord(shape=(1,), dtype="torch.float32", sha256=value)


def test_state_audit_rejects_parameter_and_non_bn_buffer_changes() -> None:
    before = ModelStateSnapshot(
        parameters={"weight": _record("a")},
        persistent_buffers={"guard": _record("b"), "bn.running_mean": _record("c")},
    )
    bad_parameter = ModelStateSnapshot(
        parameters={"weight": _record("changed")},
        persistent_buffers=before.persistent_buffers,
    )
    with pytest.raises(RuntimeError, match="parameters changed"):
        audit_state_transition(before, bad_parameter)
    bad_buffer = ModelStateSnapshot(
        parameters=before.parameters,
        persistent_buffers={"guard": _record("changed"), "bn.running_mean": _record("new")},
    )
    with pytest.raises(RuntimeError, match="Non-BatchNorm"):
        audit_state_transition(
            before,
            bad_buffer,
            allowed_buffer_mutations={"bn.running_mean"},
        )


def test_bn_reestimation_changes_only_running_statistics_and_uses_images() -> None:
    model = _ToyBNModel()
    freeze_all_parameters(model)
    guard_before = model.calibration_guard.clone()
    result = reestimate_batch_norm_statistics(
        model,
        _batches(3),
        device=torch.device("cpu"),
        expected_samples=3,
    )
    assert result["samples_seen"] == 3
    assert result["labels_not_used_as_model_inputs_for_update"] is True
    assert result["validation_used_for_update"] is False
    assert result["backward_called"] is False
    assert result["optimizer_step_called"] is False
    assert result["num_batches_tracked_min"] == 3
    assert result["num_batches_tracked_max"] == 3
    assert result["all_tracked_batch_norm_counts_equal_expected_samples"] is True
    assert result["state_transition"]["all_parameters_unchanged"] is True
    assert set(result["state_transition"]["persistent_buffer_mutations"]) == {
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
    }
    torch.testing.assert_close(model.calibration_guard, guard_before)
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()
    )
    assert all(not module.training for module in model.modules())


def test_bn_reestimation_rejects_a_reused_or_skipped_tracked_bn() -> None:
    model = _ToyReusedBNModel()
    freeze_all_parameters(model)
    with pytest.raises(RuntimeError, match="exactly once per training image"):
        reestimate_batch_norm_statistics(
            model,
            _batches(3),
            device=torch.device("cpu"),
            expected_samples=3,
        )


def test_canonical_evaluator_leaves_parameters_and_buffers_exact() -> None:
    model = _ToyBNModel()
    freeze_all_parameters(model)
    before = capture_model_state(model)
    metrics, audit = evaluate_model_without_state_change(
        model,
        _batches(2),
        dsnt=DSNT(temperature=0.05, align_corners=True),
        device=torch.device("cpu"),
        config=_config(),
    )
    after = capture_model_state(model)
    assert metrics["n_samples"] == 2
    assert metrics["selection_penalty_score_deg"] == metrics["aop_mae_deg"]
    assert metrics["aop_penalized_selection_score_deg"] == metrics["aop_mae_deg"]
    assert audit["parameter_mutations"] == []
    assert audit["persistent_buffer_mutations"] == []
    assert before == after


def test_checkpoint_freeze_copies_are_byte_identical(tmp_path: Path) -> None:
    sources = {"best": tmp_path / "best.pt", "last": tmp_path / "last.pt"}
    sources["best"].write_bytes(b"best checkpoint")
    sources["last"].write_bytes(b"last checkpoint")
    output = tmp_path / "output"
    output.mkdir()
    records = freeze_checkpoint_copies(sources, output_dir=output)
    for checkpoint_id in ("best", "last"):
        assert records[checkpoint_id]["source_sha256_before"] == records[checkpoint_id][
            "copy_sha256_before"
        ]
        assert Path(records[checkpoint_id]["copy_path"]).read_bytes() == sources[
            checkpoint_id
        ].read_bytes()


def test_phase1b_output_guards_are_fresh_private_and_aggregate_only(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "runs" / "phase1b").mkdir(parents=True)
    (root / "reports" / "phase1b").mkdir(parents=True)
    private = require_fresh_phase1b_private_output(
        root / "runs" / "phase1b" / "bn-short",
        repository_root=root,
    )
    assert private.is_dir()
    with pytest.raises(PermissionError, match="test/testing"):
        require_fresh_phase1b_private_output(
            root / "runs" / "phase1b" / "testing",
            repository_root=root,
        )
    public = require_fresh_phase1b_public_file(
        root / "reports" / "phase1b" / "bn.json",
        repository_root=root,
    )
    public.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        require_fresh_phase1b_public_file(public, repository_root=root)


def _metrics(
    value: float,
    *,
    invalid_predictions: int = 0,
) -> dict[str, float | int]:
    valid = 100 - invalid_predictions
    penalized = (value * valid + 180.0 * invalid_predictions) / 100
    return {
        "MRE_PS1": value,
        "MRE_PS2": value + 1,
        "MRE_FH1": value + 2,
        "MRE_ALL": value + 1,
        "n_samples": 100,
        "n_valid_aop": valid,
        "n_evaluable_aop": 100,
        "aop_invalid_prediction_count": invalid_predictions,
        "aop_valid_ratio": valid / 100,
        "aop_invalid_prediction_ratio": invalid_predictions / 100,
        "aop_mae_valid_deg": value,
        "aop_mae_deg": penalized,
        "aop_penalized_selection_score_deg": penalized,
    }


def test_public_bn_aggregate_has_no_private_paths_or_checkpoint_digests() -> None:
    endpoints = {}
    for checkpoint_id, epoch in (("best", 3), ("last", 20)):
        endpoints[checkpoint_id] = {
            "epoch": epoch,
            "original_bn": {
                "train_metrics": _metrics(1.0),
                "validation_metrics": _metrics(2.0),
            },
            "train_images_bn_reestimated": {"validation_metrics": _metrics(1.5)},
            "source_checkpoint_unchanged": True,
            "frozen_copy_unchanged": True,
        }
    aggregate = build_public_bn_aggregate({"endpoints": endpoints})
    text = str(aggregate).casefold()
    assert "checkpoint_path" not in text
    assert "sha256" not in text
    assert aggregate["bn_reestimation_used_for_model_selection"] is False


def test_public_bn_aggregate_distinguishes_valid_mae_from_penalized_score() -> None:
    endpoints = {}
    for checkpoint_id, epoch in (("best", 3), ("last", 20)):
        endpoints[checkpoint_id] = {
            "epoch": epoch,
            "original_bn": {
                "train_metrics": _metrics(2.0),
                "validation_metrics": _metrics(2.0),
            },
            "train_images_bn_reestimated": {
                "validation_metrics": _metrics(1.0, invalid_predictions=10)
            },
            "source_checkpoint_unchanged": True,
            "frozen_copy_unchanged": True,
        }

    aggregate = build_public_bn_aggregate({"endpoints": endpoints})
    validation = aggregate["endpoints"]["best"]["train_images_bn_reestimated"]
    metrics = validation["validation"]
    assert metrics["aop_mae_valid_deg"] == 1.0
    assert metrics["aop_invalid_prediction_ratio"] == 0.1
    assert metrics["aop_penalized_selection_score_deg"] == 18.9
    assert "aop_mae_deg" not in metrics
    assert (
        validation["delta_vs_original_validation"][
            "aop_penalized_selection_score_deg"
        ]
        == 16.9
    )


def test_cli_finishes_shared_gpu_ledger_when_runner_fails(tmp_path: Path) -> None:
    (tmp_path / "runs" / "phase1b").mkdir(parents=True)
    output = tmp_path / "runs" / "phase1b" / "failed"
    ledger = tmp_path / "runs" / "phase1b" / "gpu_budget.json"
    args = argparse.Namespace(
        requested_seconds=10.0,
        public_output=None,
        ledger=ledger,
        local_config=tmp_path / "local.yaml",
        best_checkpoint=tmp_path / "best.pt",
        last_checkpoint=tmp_path / "last.pt",
        output_dir=output,
        device="cpu",
    )

    def fail(**_: object) -> dict[str, object]:
        raise ValueError("intentional diagnostic failure")

    with pytest.raises(ValueError, match="intentional"):
        execute_with_ledger(args, runner=fail, repository_root=tmp_path)
    payload = __import__("json").loads(ledger.read_text(encoding="utf-8"))
    assert payload["active_run"] is None
    assert payload["runs"][-1]["name"] == RUN_NAME
    assert payload["runs"][-1]["status"] == "failed"
