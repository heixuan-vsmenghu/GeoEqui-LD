from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_FIXTURE = {"phase": "phase0.6-long-budget-fidelity", "seed": 42}
SCRIPT_PATH = ROOT / "scripts" / "train_phase06.py"
SPEC = importlib.util.spec_from_file_location("train_phase06", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_phase06 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_phase06)


def _protocol_payload() -> dict[str, object]:
    loaded = yaml.safe_load(
        (ROOT / "configs" / "phase06_long_budget.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def test_phase06_protocol_locks_long_budget_and_optimizer() -> None:
    protocol = train_phase06._load_protocol(
        ROOT / "configs" / "phase06_long_budget.yaml"
    )
    assert protocol["training"] == train_phase06.EXPECTED_TRAINING
    assert protocol["project"]["frozen_scope"] == list(
        train_phase06.PHASE05_FROZEN_SCOPE
    )
    assert protocol["optimizer"] == {
        "class": "Adam",
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "amsgrad": False,
    }
    assert protocol["training"]["aop_vertex_index"] == 0
    assert protocol["training"]["aop_pubic_axis_other_index"] == 1
    assert protocol["training"]["aop_fetal_head_index"] == 2
    assert protocol["selection"]["milestones"] == [20, 50, 100, 150, 200]
    assert protocol["model"] == {
        "class": "HeatmapUNet",
        "trainable_parameters": 484_171,
    }
    model = train_phase06.HeatmapUNet(base_channels=8)
    assert train_phase06.count_trainable_parameters(model) == 484_171
    assert protocol["execution"]["local_config"] == "configs/phase05_local.yaml"
    assert protocol["execution"]["identity_manifest"] == "runs/phase06/identity.json"
    assert all(
        variant["validation_decoders"] == ["dsnt"]
        for variant in protocol["variants"].values()
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        ("project", "testing_frozen", False, PermissionError),
        ("training", "seed", 43, ValueError),
        ("training", "epochs", 20, ValueError),
        ("optimizer", "eps", 1.0e-7, ValueError),
        ("execution", "compare_total_loss_across_variants", True, ValueError),
    ],
)
def test_phase06_protocol_rejects_drift(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    error: type[Exception],
) -> None:
    payload = _protocol_payload()
    payload[section][key] = value  # type: ignore[index]
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(error):
        train_phase06._load_protocol(path)


def test_phase06_seed_and_output_are_fail_closed() -> None:
    expected = ROOT / "runs" / "phase06" / "B1" / "seed_42"
    assert train_phase06._validate_seed_and_output(
        variant="B1", seed=42, output_dir=expected
    ) == expected.resolve(strict=False)
    with pytest.raises(ValueError, match="only seed 42"):
        train_phase06._validate_seed_and_output(
            variant="B1", seed=43, output_dir=expected
        )
    with pytest.raises(PermissionError, match="runs/phase06"):
        train_phase06._validate_seed_and_output(
            variant="B1", seed=42, output_dir=ROOT / "runs" / "phase05" / "B1" / "seed_42"
        )


def test_phase06_refuses_to_overwrite_nonempty_run_directory(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        train_phase06._ensure_empty_output(output)


def test_training_order_hash_is_deterministic_and_order_sensitive(tmp_path: Path) -> None:
    filenames = ["a.png", "b.png", "c.png", "d.png"]
    first = train_phase06._TrainingOrderSampler(
        filenames,
        generator=torch.Generator().manual_seed(42),
        record_path=tmp_path / "first.json",
    )
    second = train_phase06._TrainingOrderSampler(
        filenames,
        generator=torch.Generator().manual_seed(42),
        record_path=tmp_path / "second.json",
    )
    assert list(first) == list(second)
    assert first.records == second.records
    assert train_phase06._filename_order_sha256(filenames) != (
        train_phase06._filename_order_sha256(list(reversed(filenames)))
    )


def test_recording_sampler_matches_phase05_random_sampler_across_epochs(
    tmp_path: Path,
) -> None:
    dataset = list(range(9))
    phase05_generator = torch.Generator().manual_seed(42)
    phase05_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=phase05_generator,
        num_workers=0,
    )
    phase05_orders = [
        [int(batch.item()) for batch in phase05_loader]
        for _ in range(3)
    ]

    phase06_generator = torch.Generator().manual_seed(42)
    sampler = train_phase06._TrainingOrderSampler(
        [str(index) for index in dataset],
        generator=phase06_generator,
        record_path=tmp_path / "orders.json",
    )
    phase06_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        generator=phase06_generator,
        num_workers=0,
    )
    phase06_orders = [
        [int(batch.item()) for batch in phase06_loader]
        for _ in range(3)
    ]
    assert phase06_orders == phase05_orders


def test_phase06_public_protocol_contains_no_private_data_values() -> None:
    text = (ROOT / "configs" / "phase06_long_budget.yaml").read_text(encoding="utf-8")
    assert "labels_sha256" not in text
    assert "aggregate_sha256" not in text
    assert "image_dir" not in text
    assert "labels_csv" not in text


def test_local_config_path_is_exact_not_just_schema_compatible(tmp_path: Path) -> None:
    substitute = tmp_path / "phase05_local.yaml"
    substitute.write_text("phase: phase0.5\nsplits: {}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="exactly configs/phase05_local.yaml"):
        train_phase06._validate_local_config_path(substitute)


def test_self_consistent_but_different_data_identity_is_rejected() -> None:
    frozen = {
        "train": {
            "sample_count": 300,
            "labels_sha256": "a" * 64,
            "aggregate_sha256": "b" * 64,
            "source_columns": {"PS1": "PS1", "PS2": "PS2", "FH1": "FH1"},
        },
        "validation": {
            "sample_count": 100,
            "labels_sha256": "c" * 64,
            "aggregate_sha256": "d" * 64,
            "source_columns": {
                "PS1": "PS1",
                "PS2": "PS2",
                "FH1": "AOP Tangency",
            },
        },
    }
    actual = {role: dict(value) for role, value in frozen.items()}
    actual["train"]["labels_sha256"] = "e" * 64
    actual["train"]["aggregate_sha256"] = "f" * 64
    with pytest.raises(PermissionError, match="differs from frozen Phase 0.5"):
        train_phase06._assert_phase05_data_identity(actual, frozen)


def test_phase05_freeze_requires_annotated_tag_and_clean_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def clean_git_output(*arguments: str) -> str:
        if arguments[:2] == ("cat-file", "-t"):
            return "tag"
        if arguments[:2] == ("rev-parse", "phase05-v0.1.0^{}"):
            return train_phase06.PHASE05_FROZEN_COMMIT
        if arguments[0] == "status":
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(train_phase06, "_git_output", clean_git_output)
    monkeypatch.setattr(
        train_phase06.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(train_phase06, "_validate_phase05_freeze_record", lambda: None)
    train_phase06._validate_phase05_freeze()

    monkeypatch.setattr(
        train_phase06,
        "_git_output",
        lambda *args: "commit" if args[:2] == ("cat-file", "-t") else "",
    )
    with pytest.raises(RuntimeError, match="annotated tag"):
        train_phase06._validate_phase05_freeze()


def test_phase05_freeze_record_requires_exact_tag_object_and_tree() -> None:
    record = yaml.safe_load(
        (ROOT / "reports" / "phase06" / "PHASE05_FREEZE.json").read_text(
            encoding="utf-8"
        )
    )
    train_phase06._validate_phase05_freeze_record_payload(
        record,
        tag_object=record["annotated_tag_object"],
        frozen_tree=record["frozen_tree"],
    )
    drifted = {**record, "annotated_tag_object": "0" * 40}
    with pytest.raises(RuntimeError, match="exactly match"):
        train_phase06._validate_phase05_freeze_record_payload(
            drifted,
            tag_object=record["annotated_tag_object"],
            frozen_tree=record["frozen_tree"],
        )


def test_phase06_identity_manifest_is_write_once_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    payload = {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "protocol_sha256": "b" * 64,
        "data": {"train": {"sample_count": 300}, "validation": {"sample_count": 100}},
        "model": {"class": "HeatmapUNet", "trainable_parameters": 484_171},
        "training_common": {"seed": 42, "epochs": 200},
        "optimizer": {"class": "Adam"},
        "environment": {"torch": "2.5.1"},
    }
    assert train_phase06._write_or_validate_identity(
        path, payload, allow_create=True
    ) == "created"
    assert train_phase06._write_or_validate_identity(
        path, payload, allow_create=False
    ) == "validated"
    drifted = {**payload, "git_commit": "c" * 40}
    with pytest.raises(RuntimeError, match="identity drift"):
        train_phase06._write_or_validate_identity(path, drifted, allow_create=False)
    with pytest.raises(PermissionError, match="B1/B2 require"):
        train_phase06._write_or_validate_identity(
            tmp_path / "missing.json", payload, allow_create=False
        )


def test_identity_payload_excludes_only_expected_variant_loss_weights() -> None:
    protocol = train_phase06._load_protocol(
        ROOT / "configs" / "phase06_long_budget.yaml"
    )
    base = train_phase06.SupervisedTrainingConfig.from_mapping(protocol["training"])
    common = {
        "commit": "a" * 40,
        "protocol_sha256": "b" * 64,
        "data": {"train": {"sample_count": 300}, "validation": {"sample_count": 100}},
        "model": {
            "class": "HeatmapUNet",
            "trainable_parameters": 484_171,
            "initialization_sha256": "c" * 64,
        },
        "optimizer": protocol["optimizer"],
        "environment": {"torch": "2.5.1"},
    }
    b0 = train_phase06._identity_payload(
        config=train_phase06.apply_variant(base, "B0"), **common
    )
    b2 = train_phase06._identity_payload(
        config=train_phase06.apply_variant(base, "B2"), **common
    )
    assert b0 == b2
    assert not {
        "heatmap_loss_weight",
        "coordinate_loss_weight",
        "distribution_loss_weight",
    } & set(b0["training_common"])


def _completed_run_fixture(best_epoch: int = 137) -> tuple[
    list[dict[str, str]], dict[str, object], dict[str, object], dict[str, float]
]:
    rows: list[dict[str, str]] = []
    parsed: list[dict[str, float]] = []
    for epoch in range(1, 201):
        distance = abs(epoch - best_epoch)
        metrics = {
            "MRE_PS1": 10.0 + 0.1 * distance,
            "MRE_PS2": 20.0 + 0.1 * distance,
            "MRE_FH1": 30.0 + 0.1 * distance,
            "MRE_ALL": 20.0 + 0.1 * distance,
            "aop_mae_deg": 5.0 + 0.05 * distance,
        }
        parsed.append(metrics)
        rows.append(
            {
                "epoch": str(epoch),
                **{f"val_{name}": str(value) for name, value in metrics.items()},
            }
        )
    best = parsed[best_epoch - 1]
    summary: dict[str, object] = {
        "selection_split": "validation",
        "checkpoint_metric": "aop_mae_deg",
        "selection_tiebreak": ["aop_mae_deg", "MRE_ALL", "earlier_epoch"],
        "best_epoch": best_epoch,
        "best_value": best["aop_mae_deg"],
        "best_validation_metrics": dict(best),
        "last_validation_metrics": dict(parsed[-1]),
    }
    checkpoint: dict[str, object] = {
        "epoch": best_epoch,
        "seed": 42,
        "config": dict(RUN_CONFIG_FIXTURE),
        "metrics": dict(best),
    }
    return rows, summary, checkpoint, dict(best)


def test_completed_run_validation_recomputes_best_and_milestones() -> None:
    rows, summary, checkpoint, reevaluated = _completed_run_fixture()
    audit = train_phase06._validate_completed_run(
        rows,
        summary=summary,
        checkpoint_payload=checkpoint,
        best_validation_metrics=reevaluated,
        run_config=RUN_CONFIG_FIXTURE,
    )
    assert audit["validated_epochs"] == 200
    assert audit["best_epoch"] == 137
    assert set(audit["milestones"]) == {"20", "50", "100", "150", "200", "best"}


def test_completed_run_validation_rejects_history_and_artifact_drift() -> None:
    rows, summary, checkpoint, reevaluated = _completed_run_fixture()
    with pytest.raises(ValueError, match="exactly 200"):
        train_phase06._validate_completed_run(
            rows[:-1],
            summary=summary,
            checkpoint_payload=checkpoint,
            best_validation_metrics=reevaluated,
            run_config=RUN_CONFIG_FIXTURE,
        )

    broken_mean = [dict(row) for row in rows]
    broken_mean[9]["val_MRE_ALL"] = "999"
    with pytest.raises(ValueError, match="MRE_ALL mean identity"):
        train_phase06._validate_completed_run(
            broken_mean,
            summary=summary,
            checkpoint_payload=checkpoint,
            best_validation_metrics=reevaluated,
            run_config=RUN_CONFIG_FIXTURE,
        )

    wrong_summary = {**summary, "best_epoch": 136}
    with pytest.raises(ValueError, match="recomputed checkpoint tuple"):
        train_phase06._validate_completed_run(
            rows,
            summary=wrong_summary,
            checkpoint_payload=checkpoint,
            best_validation_metrics=reevaluated,
            run_config=RUN_CONFIG_FIXTURE,
        )

    wrong_reevaluation = {**reevaluated, "MRE_ALL": 99.0}
    with pytest.raises(ValueError, match="best checkpoint re-evaluation"):
        train_phase06._validate_completed_run(
            rows,
            summary=summary,
            checkpoint_payload=checkpoint,
            best_validation_metrics=wrong_reevaluation,
            run_config=RUN_CONFIG_FIXTURE,
        )


def test_mre_reduction_uses_rounding_tolerance_not_artifact_tolerance() -> None:
    expected_three_point_mean = 24.779
    reported_mre_all = expected_three_point_mean + 2.2888e-5
    train_phase06._require_mre_reduction_close(
        reported_mre_all,
        expected_three_point_mean,
        context="real-scale Phase 0.5 MRE reduction",
    )
    with pytest.raises(ValueError, match="differs between completed-run artifacts"):
        train_phase06._require_metric_close(
            reported_mre_all,
            expected_three_point_mean,
            context="cross-artifact equality",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 43, "registered seed 42"),
        ("config", {"phase": "drifted"}, "configuration differs"),
    ],
)
def test_completed_run_validation_rejects_checkpoint_identity_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    rows, summary, checkpoint, reevaluated = _completed_run_fixture()
    checkpoint[field] = value
    with pytest.raises(ValueError, match=message):
        train_phase06._validate_completed_run(
            rows,
            summary=summary,
            checkpoint_payload=checkpoint,
            best_validation_metrics=reevaluated,
            run_config=RUN_CONFIG_FIXTURE,
        )
