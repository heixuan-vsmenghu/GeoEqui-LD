from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps
from geoequi_ld.diagnostics.phase1a import (
    REQUIRED_CHECKPOINT_IDS,
    CheckpointSpec,
    HeatmapDiagnosticAccumulator,
    assert_public_aggregate,
    evaluate_coordinate_predictions,
    fixed_visualization_indices,
    load_checkpoint_specs,
    load_phase06_model,
    load_phase1a_protocol,
    require_private_output_path,
    require_public_output_path,
    run_synthetic_sanity,
    train_mean_coordinate_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "phase1a_b0_diagnostics.yaml"


class _RowsDataset:
    keypoint_order = ("PS1", "PS2", "FH1")
    source_columns = {"PS1": "PS1", "PS2": "PS2", "FH1": "FH1"}
    input_size_hw = (512, 512)

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.rows)


def _row(
    ps1: tuple[float, float],
    ps2: tuple[float, float],
    fh1: tuple[float, float],
) -> dict[str, str]:
    return {"PS1": str(ps1), "PS2": str(ps2), "FH1": str(fh1)}


def test_phase1a_protocol_locks_six_endpoints_and_privacy_roots() -> None:
    protocol = load_phase1a_protocol(PROTOCOL_PATH)
    rows = protocol["execution"]["checkpoints"]
    assert tuple(row["id"] for row in rows) == REQUIRED_CHECKPOINT_IDS
    assert protocol["data_contract"]["allowed_splits"] == ["train", "validation"]
    assert set(protocol["data_contract"]["forbidden_splits"]) == {"test", "testing"}
    assert protocol["reporting"]["public_payload"] == "aggregate_only"
    assert protocol["reporting"]["causal_claim_allowed"] is False


def test_synthetic_sanity_covers_gaussian_temperature_amplitude_and_flat_cases() -> None:
    result = run_synthetic_sanity(load_phase1a_protocol(PROTOCOL_PATH))
    cases = {case["case_id"]: case for case in result["cases"]}
    assert set(cases) == {
        "gaussian_argmax",
        "gaussian_dsnt_t1",
        "gaussian_dsnt_t0.05",
        "gaussian_amplitude_0.1_dsnt_t0.05",
        "gaussian_amplitude_0.01_dsnt_t0.05",
        "zero_heatmaps_dsnt_t0.05",
        "flat_heatmaps_dsnt_t0.05",
    }
    assert cases["gaussian_argmax"]["MRE_ALL"] == pytest.approx(0.0, abs=1e-6)
    assert cases["gaussian_dsnt_t0.05"]["MRE_ALL"] < cases["gaussian_dsnt_t1"]["MRE_ALL"]
    assert cases["gaussian_dsnt_t0.05"]["aop_official_valid"] is True
    for case_id in ("zero_heatmaps_dsnt_t0.05", "flat_heatmaps_dsnt_t0.05"):
        assert cases[case_id]["probability_entropy_normalized"] == pytest.approx(1.0, abs=1e-6)
        assert cases[case_id]["probability_sum_max_abs_error"] < 1e-6
        assert cases[case_id]["aop_official_valid"] is False
        assert cases[case_id]["aop_predicted_deg"] is None
        assert cases[case_id]["aop_mae_valid_deg"] is None
        assert cases[case_id]["aop_penalized_score_deg"] == pytest.approx(180.0)


def test_heatmap_diagnostics_identifies_zero_map_uniformity_ties_and_channel_collapse() -> None:
    points = torch.tensor([[[2.0, 5.0], [7.0, 5.0], [4.0, 2.0]]])
    targets = generate_gaussian_heatmaps(points, size_hw=(8, 10), sigma=1.0)
    logits = torch.zeros_like(targets)
    accumulator = HeatmapDiagnosticAccumulator(
        foreground_radius_px=2.0,
        probability_mass_radius_px=2.0,
        tie_atol=0.0,
        tie_rtol=0.0,
    )
    accumulator.add(logits, targets, points, torch.ones((1, 3), dtype=torch.bool))
    result = accumulator.finalize()
    assert result["overall"]["zero_map_mse_ratio"] == pytest.approx(1.0)
    assert result["overall"]["probability_entropy_normalized"]["mean"] == pytest.approx(1.0)
    assert result["overall"]["raw_peak_tie_count"]["mean"] == pytest.approx(80.0)
    assert result["overall"]["raw_peak_gap"]["mean"] == pytest.approx(0.0)
    assert all(
        summary["mean"] == pytest.approx(0.0)
        for summary in result["channel_pairwise_raw_mae"].values()
    )


def test_heatmap_diagnostics_rejects_nonfinite_checkpoint_output() -> None:
    points = torch.tensor([[[2.0, 5.0], [7.0, 5.0], [4.0, 2.0]]])
    targets = generate_gaussian_heatmaps(points, size_hw=(8, 10), sigma=1.0)
    logits = torch.zeros_like(targets)
    logits[0, 0, 0, 0] = torch.nan
    accumulator = HeatmapDiagnosticAccumulator()
    with pytest.raises(ValueError, match="NaN or Inf"):
        accumulator.add(logits, targets, points, torch.ones((1, 3), dtype=torch.bool))


def test_train_mean_baseline_fit_is_independent_of_validation_labels() -> None:
    train = _RowsDataset(
        [
            _row((10, 10), (30, 10), (20, 30)),
            _row((30, 10), (50, 10), (40, 30)),
        ]
    )
    validation_a = _RowsDataset([_row((20, 10), (40, 10), (30, 30))])
    validation_b = _RowsDataset([_row((500, 500), (490, 500), (500, 490))])
    _, mean_a = train_mean_coordinate_baseline(train, validation_a)  # type: ignore[arg-type]
    _, mean_b = train_mean_coordinate_baseline(train, validation_b)  # type: ignore[arg-type]
    expected = torch.tensor([[20.0, 10.0], [40.0, 10.0], [30.0, 30.0]])
    torch.testing.assert_close(mean_a, expected)
    torch.testing.assert_close(mean_b, expected)


def test_coordinate_evaluator_reports_undefined_valid_mae_and_180_penalty() -> None:
    target = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]])
    prediction = torch.zeros_like(target)
    metrics = evaluate_coordinate_predictions(
        prediction,
        target,
        torch.ones((1, 3), dtype=torch.bool),
    )
    assert metrics["n_evaluable_aop"] == 1
    assert metrics["n_valid_aop"] == 0
    assert metrics["aop_invalid_prediction_count"] == 1
    assert metrics["aop_valid_ratio"] == pytest.approx(0.0)
    assert metrics["aop_invalid_prediction_ratio"] == pytest.approx(1.0)
    assert math.isnan(float(metrics["aop_mae_valid_deg"]))
    assert metrics["aop_mae_deg"] == pytest.approx(180.0)


def test_visualization_indices_are_stable_label_independent_and_role_bounded() -> None:
    first = fixed_visualization_indices(100, role="validation", count=4, seed=42)
    second = fixed_visualization_indices(100, role="validation", count=4, seed=42)
    assert first == second
    assert len(first) == len(set(first)) == 4
    with pytest.raises(PermissionError):
        fixed_visualization_indices(100, role="testing", count=4, seed=42)


def test_public_aggregate_rejects_identifiers_real_coordinates_and_absolute_paths() -> None:
    assert_public_aggregate({"checkpoint_id": "B0_best", "MRE_ALL": 1.0})
    with pytest.raises(PermissionError, match="Sensitive field"):
        assert_public_aggregate({"filename": "private.jpg"})
    with pytest.raises(PermissionError, match="Sensitive field"):
        assert_public_aggregate({"train_mean_coordinates": [[1.0, 2.0]]})
    with pytest.raises(PermissionError, match="Absolute path"):
        windows_absolute = "C" + ":\\private\\artifact.json"
        assert_public_aggregate({"note": windows_absolute})


def test_output_guards_keep_public_and_private_material_separate(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    (root / "artifacts").mkdir()
    (root / "runs").mkdir()
    assert require_public_output_path(
        root / "reports" / "phase1a" / "aggregate.json", repository_root=root
    ).is_relative_to(root / "reports" / "phase1a")
    assert require_private_output_path(
        root / "artifacts" / "phase1a" / "details.json", repository_root=root
    ).is_relative_to(root / "artifacts")
    with pytest.raises(PermissionError):
        require_public_output_path(root / "artifacts" / "leak.json", repository_root=root)
    with pytest.raises(PermissionError):
        require_private_output_path(root / "reports" / "leak.json", repository_root=root)
    with pytest.raises(PermissionError):
        require_private_output_path(
            root / "runs" / "phase06" / "B0" / "best.pt", repository_root=root
        )


def test_checkpoint_matrix_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    phase06 = root / "runs" / "phase06"
    phase06.mkdir(parents=True)
    protocol = load_phase1a_protocol(PROTOCOL_PATH)
    rows = protocol["execution"]["checkpoints"]
    for row in rows:
        path = root / row["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    specs = load_checkpoint_specs(protocol, repository_root=root)
    assert tuple(spec.checkpoint_id for spec in specs) == REQUIRED_CHECKPOINT_IDS
    escaped = root / "outside.pt"
    escaped.write_bytes(b"not-allowed")
    rows[0]["relative_path"] = "outside.pt"
    with pytest.raises(PermissionError, match="checkpoint path"):
        load_checkpoint_specs(protocol, repository_root=root)


def test_checkpoint_matrix_requires_registered_sha256(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "runs" / "phase06").mkdir(parents=True)
    protocol = load_phase1a_protocol(PROTOCOL_PATH)
    for row in protocol["execution"]["checkpoints"]:
        path = root / row["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    protocol["execution"]["checkpoints"][0]["expected_sha256"] = None
    with pytest.raises(PermissionError, match="SHA-256"):
        load_checkpoint_specs(protocol, repository_root=root)


def test_checkpoint_hash_is_verified_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    spec = CheckpointSpec("B0_best", "B0", "best", 120, checkpoint, "0" * 64)
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before the hash check")

    monkeypatch.setattr(torch, "load", forbidden_load)
    protocol = load_phase1a_protocol(PROTOCOL_PATH)
    with pytest.raises(PermissionError, match="hash mismatch"):
        load_phase06_model(spec, device=torch.device("cpu"), protocol=protocol)
    assert called is False
