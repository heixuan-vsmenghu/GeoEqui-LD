"""Phase 1A diagnostics for the frozen Phase 0.6 supervised runs.

This module deliberately contains no optimizer or training loop.  It supports
three bounded questions only:

* whether the repository Gaussian targets and DSNT decoder agree on synthetic
  three-landmark geometry;
* how a train-label-only constant coordinate baseline performs on validation;
* what the six saved Phase 0.6 best/last checkpoints emit on the complete
  validation split.

The checkpoint comparison characterizes saved endpoints.  It cannot recover
the missing epoch-186/187 transition and therefore cannot establish the cause
of B0's late collapse.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from geoequi_ld.data.access_policy import (
    LabeledSplitSpec,
    fingerprint_labeled_split,
    load_phase05_local_splits,
    verify_fingerprint,
)
from geoequi_ld.data.dataset import IUGCLabeledDataset, parse_point
from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps
from geoequi_ld.geometry.aop import compute_aop
from geoequi_ld.geometry.coordinates import normalized_to_pixel, resize_points
from geoequi_ld.metrics.keypoints import absolute_angle_error, summarize_keypoint_metrics
from geoequi_ld.models.decoding import decode_heatmaps
from geoequi_ld.models.dsnt import DSNT, spatial_expectation, spatial_softmax
from geoequi_ld.models.unet import HeatmapUNet
from geoequi_ld.utils.hashing import sha256_file

KEYPOINT_NAMES = ("PS1", "PS2", "FH1")
REQUIRED_CHECKPOINT_IDS = (
    "B0_best",
    "B0_last",
    "B1_best",
    "B1_last",
    "B2_best",
    "B2_last",
)
OFFICIAL_AOP_EPS = 1e-8
INVALID_AOP_PENALTY_DEG = 180.0


@dataclass(frozen=True)
class CheckpointSpec:
    """A pre-registered local Phase 0.6 checkpoint endpoint."""

    checkpoint_id: str
    variant: str
    endpoint: str
    epoch: int
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class VerifiedSplits:
    """Exactly the approved train and validation split specifications."""

    specs: Mapping[str, LabeledSplitSpec]
    fingerprints: Mapping[str, Mapping[str, str | int]]


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def load_phase1a_protocol(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on the public Phase 1A diagnostic protocol."""

    protocol_path = Path(path)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Phase 1A protocol does not exist: {protocol_path}")
    loaded = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol = dict(_mapping(loaded, context="Phase 1A protocol"))
    project = _mapping(protocol.get("project"), context="project")
    data_contract = _mapping(protocol.get("data_contract"), context="data_contract")
    diagnostic = _mapping(protocol.get("diagnostic"), context="diagnostic")
    execution = _mapping(protocol.get("execution"), context="execution")

    if project.get("phase") != "phase1a-b0-diagnostics":
        raise ValueError("Protocol phase must be phase1a-b0-diagnostics")
    if project.get("testing_frozen") is not True:
        raise PermissionError("Phase 1A diagnostics require testing_frozen=true")
    if data_contract.get("allowed_splits") != ["train", "validation"]:
        raise PermissionError("Phase 1A permits exactly train and validation")
    if set(data_contract.get("forbidden_splits", [])) != {"test", "testing"}:
        raise PermissionError("Phase 1A must explicitly forbid test and testing")
    for role, expected_count in (("train", 300), ("validation", 100)):
        role_contract = _mapping(data_contract.get(role), context=f"data_contract.{role}")
        if role_contract.get("fingerprint_required") is not True:
            raise PermissionError(f"{role} must require fingerprint verification")
        if int(role_contract.get("sample_count", -1)) != expected_count:
            raise PermissionError(f"Unexpected {role} sample count contract")

    if tuple(diagnostic.get("keypoint_order", ())) != KEYPOINT_NAMES:
        raise ValueError("Diagnostic keypoint order must be PS1, PS2, FH1")
    if tuple(diagnostic.get("input_size_hw", ())) != (512, 512):
        raise ValueError("Phase 1A diagnostic input must be 512x512")
    if tuple(diagnostic.get("heatmap_size_hw", ())) != (256, 256):
        raise ValueError("Phase 1A diagnostic heatmaps must be 256x256")
    if float(diagnostic.get("sigma_heatmap_px", 0.0)) != 4.0:
        raise ValueError("Phase 1A diagnostic sigma must be 4 heatmap pixels")
    if float(diagnostic.get("dsnt_temperature", 0.0)) != 0.05:
        raise ValueError("Phase 1A diagnostic DSNT temperature must be 0.05")
    if diagnostic.get("align_corners") is not True:
        raise ValueError("Phase 1A diagnostics require align_corners=true")
    if int(diagnostic.get("foreground_radius_heatmap_px", 0)) <= 0:
        raise ValueError("foreground_radius_heatmap_px must be positive")
    if int(diagnostic.get("probability_mass_radius_heatmap_px", 0)) <= 0:
        raise ValueError("probability_mass_radius_heatmap_px must be positive")
    if int(diagnostic.get("visualization_train_count", 0)) <= 0:
        raise ValueError("visualization_train_count must be positive")
    if int(diagnostic.get("visualization_validation_count", 0)) <= 0:
        raise ValueError("visualization_validation_count must be positive")

    checkpoint_rows = execution.get("checkpoints")
    if not isinstance(checkpoint_rows, list):
        raise ValueError("execution.checkpoints must be a list")
    ids = [str(_mapping(row, context="checkpoint").get("id", "")) for row in checkpoint_rows]
    if tuple(ids) != REQUIRED_CHECKPOINT_IDS:
        raise ValueError("Protocol must list the six B0/B1/B2 best/last checkpoints in order")
    if str(execution.get("local_config")) != "configs/phase05_local.yaml":
        raise PermissionError("Phase 1A must reuse the canonical Phase 0.5 local split config")
    return protocol


def require_canonical_path(actual: str | Path, expected: str | Path, *, context: str) -> Path:
    """Require an existing input to resolve to its repository-canonical location."""

    actual_path = Path(actual).resolve(strict=True)
    expected_path = Path(expected).resolve(strict=True)
    if actual_path != expected_path:
        raise PermissionError(f"{context} must use the canonical repository path")
    return actual_path


def require_public_output_path(path: str | Path, *, repository_root: str | Path) -> Path:
    """Restrict publishable aggregates to reports/phase1a."""

    root = Path(repository_root).resolve(strict=True)
    public_root = (root / "reports" / "phase1a").resolve(strict=False)
    destination = Path(path).resolve(strict=False)
    if not destination.is_relative_to(public_root) or destination == public_root:
        raise PermissionError("Public Phase 1A output must be a file below reports/phase1a")
    return destination


def require_private_output_path(path: str | Path, *, repository_root: str | Path) -> Path:
    """Restrict sensitive diagnostics to new Git-ignored Phase 1A roots."""

    root = Path(repository_root).resolve(strict=True)
    destination = Path(path).resolve(strict=False)
    allowed_roots = (
        (root / "artifacts" / "phase1a").resolve(strict=False),
        (root / "runs" / "phase1a").resolve(strict=False),
    )
    if not any(
        destination.is_relative_to(allowed) and destination != allowed
        for allowed in allowed_roots
    ):
        raise PermissionError(
            "Private output must stay below artifacts/phase1a or runs/phase1a"
        )
    return destination


def load_checkpoint_specs(
    protocol: Mapping[str, Any], *, repository_root: str | Path
) -> tuple[CheckpointSpec, ...]:
    """Resolve only the six pre-registered checkpoint paths below runs/phase06."""

    root = Path(repository_root).resolve(strict=True)
    execution = _mapping(protocol.get("execution"), context="execution")
    rows = execution.get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError("execution.checkpoints must be a list")
    expected_matrix = {
        "B0_best": ("B0", "best", 120, "runs/phase06/B0/seed_42/best.pt"),
        "B0_last": ("B0", "last", 200, "runs/phase06/B0/seed_42/last.pt"),
        "B1_best": ("B1", "best", 194, "runs/phase06/B1/seed_42/best.pt"),
        "B1_last": ("B1", "last", 200, "runs/phase06/B1/seed_42/last.pt"),
        "B2_best": ("B2", "best", 15, "runs/phase06/B2/seed_42/best.pt"),
        "B2_last": ("B2", "last", 200, "runs/phase06/B2/seed_42/last.pt"),
    }
    phase06_root = (root / "runs" / "phase06").resolve(strict=True)
    specs: list[CheckpointSpec] = []
    for row_value in rows:
        row = _mapping(row_value, context="checkpoint")
        checkpoint_id = str(row.get("id", ""))
        if checkpoint_id not in expected_matrix:
            raise ValueError(f"Unexpected Phase 0.6 checkpoint id: {checkpoint_id}")
        expected_variant, expected_endpoint, expected_epoch, expected_relative = expected_matrix[
            checkpoint_id
        ]
        relative = Path(str(row.get("relative_path", "")))
        if relative.is_absolute():
            raise PermissionError("Checkpoint paths in the public protocol must be relative")
        if relative.as_posix() != expected_relative:
            raise PermissionError(f"Unexpected checkpoint path for {checkpoint_id}")
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_file() or not candidate.is_relative_to(phase06_root):
            raise PermissionError(f"Checkpoint escapes the frozen Phase 0.6 run root: {relative}")
        epoch = int(row.get("expected_epoch", -1))
        if epoch != expected_epoch:
            raise ValueError(f"Unexpected epoch for {checkpoint_id}: {epoch}")
        variant = str(row.get("variant"))
        endpoint = str(row.get("endpoint"))
        if variant != expected_variant or endpoint != expected_endpoint:
            raise ValueError(f"Checkpoint identity fields disagree for {checkpoint_id}")
        expected_hash = str(row.get("expected_sha256", ""))
        invalid_hash_character = any(
            character not in "0123456789abcdef" for character in expected_hash
        )
        if len(expected_hash) != 64 or invalid_hash_character:
            raise PermissionError(f"A lowercase SHA-256 is required for {checkpoint_id}")
        specs.append(
            CheckpointSpec(
                checkpoint_id=checkpoint_id,
                variant=variant,
                endpoint=endpoint,
                epoch=epoch,
                path=candidate,
                expected_sha256=expected_hash,
            )
        )
    if tuple(spec.checkpoint_id for spec in specs) != REQUIRED_CHECKPOINT_IDS:
        raise ValueError("Resolved checkpoint matrix is incomplete")
    return tuple(specs)


def load_verified_splits(
    local_config: str | Path, protocol: Mapping[str, Any]
) -> VerifiedSplits:
    """Load exactly train/validation and verify every registered fingerprint."""

    specs = load_phase05_local_splits(local_config)
    if set(specs) != {"train", "validation"}:
        raise PermissionError("Phase 1A accepts exactly train and validation")
    contract = _mapping(protocol.get("data_contract"), context="data_contract")
    fingerprints: dict[str, Mapping[str, str | int]] = {}
    for role in ("train", "validation"):
        spec = specs[role]
        actual = fingerprint_labeled_split(spec)
        verify_fingerprint(actual, spec.expected_fingerprint, role=role)
        role_contract = _mapping(contract.get(role), context=f"data_contract.{role}")
        if int(actual["sample_count"]) != int(role_contract["sample_count"]):
            raise PermissionError(f"{role} fingerprint count differs from the public protocol")
        expected_columns = _mapping(
            role_contract.get("source_columns"), context=f"data_contract.{role}.source_columns"
        )
        actual_columns = {"PS1": "PS1", "PS2": "PS2", "FH1": spec.fh1_column}
        if dict(expected_columns) != actual_columns:
            raise PermissionError(f"{role} source-column mapping differs from the protocol")
        fingerprints[role] = actual
    return VerifiedSplits(specs=specs, fingerprints=fingerprints)


def make_labeled_dataset(
    spec: LabeledSplitSpec, protocol: Mapping[str, Any]
) -> IUGCLabeledDataset:
    diagnostic = _mapping(protocol.get("diagnostic"), context="diagnostic")
    return IUGCLabeledDataset(
        image_dir=spec.image_dir,
        labels_csv=spec.labels_csv,
        source_columns={"PS1": "PS1", "PS2": "PS2", "FH1": spec.fh1_column},
        keypoint_order=tuple(diagnostic["keypoint_order"]),
        input_size_hw=tuple(diagnostic["input_size_hw"]),
        heatmap_size_hw=tuple(diagnostic["heatmap_size_hw"]),
        sigma=float(diagnostic["sigma_heatmap_px"]),
        align_corners=bool(diagnostic["align_corners"]),
    )


def points_from_labeled_dataset(dataset: IUGCLabeledDataset) -> Tensor:
    """Read landmark labels without decoding images."""

    points = torch.tensor(
        [
            [parse_point(row[dataset.source_columns[name]]) for name in dataset.keypoint_order]
            for _, row in dataset.rows.iterrows()
        ],
        dtype=torch.float32,
    )
    if tuple(points.shape) != (len(dataset), len(dataset.keypoint_order), 2):
        raise ValueError("Unexpected labeled coordinate shape")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("Labeled coordinates contain NaN or Inf")
    height, width = dataset.input_size_hw
    x, y = points.unbind(dim=-1)
    in_bounds = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    if not bool(in_bounds.all()):
        raise ValueError("Labeled coordinates fall outside the registered 512x512 frame")
    return points


def evaluate_coordinate_predictions(
    predicted_original_px: Tensor,
    target_original_px: Tensor,
    valid_mask: Tensor,
    *,
    keypoint_names: Sequence[str] = KEYPOINT_NAMES,
    vertex_index: int = 0,
    pubic_axis_other_index: int = 1,
    fetal_head_index: int = 2,
) -> dict[str, float | int]:
    """Apply the repository MRE and penalized-AoP evaluator to coordinates."""

    if predicted_original_px.shape != target_original_px.shape:
        raise ValueError("Predicted and target coordinate shapes differ")
    if valid_mask.shape != predicted_original_px.shape[:2]:
        raise ValueError("valid_mask shape differs from coordinate tensors")
    predicted = predicted_original_px.detach().cpu()
    target = target_original_px.detach().cpu()
    mask = valid_mask.detach().cpu().to(dtype=torch.bool)
    metrics: dict[str, float | int] = {
        **summarize_keypoint_metrics(
            predicted,
            target,
            keypoint_names=keypoint_names,
            valid_mask=mask,
        ),
        "n_samples": int(predicted.shape[0]),
    }
    predicted_aop, predicted_valid = compute_aop(
        predicted,
        vertex_index=vertex_index,
        pubic_axis_other_index=pubic_axis_other_index,
        fetal_head_index=fetal_head_index,
        output_unit="degrees",
        eps=OFFICIAL_AOP_EPS,
        invalid="mask",
    )
    target_aop, target_valid = compute_aop(
        target,
        vertex_index=vertex_index,
        pubic_axis_other_index=pubic_axis_other_index,
        fetal_head_index=fetal_head_index,
        output_unit="degrees",
        eps=OFFICIAL_AOP_EPS,
        invalid="mask",
    )
    required = [vertex_index, pubic_axis_other_index, fetal_head_index]
    evaluable = target_valid & mask[:, required].all(dim=1)
    angle_errors = absolute_angle_error(predicted_aop, target_aop)
    valid_errors = angle_errors[evaluable & predicted_valid & torch.isfinite(angle_errors)]
    metrics["n_valid_aop"] = int(valid_errors.numel())
    metrics["n_evaluable_aop"] = int(evaluable.sum().item())
    metrics["aop_invalid_prediction_count"] = int((evaluable & ~predicted_valid).sum().item())
    if metrics["n_evaluable_aop"]:
        metrics["aop_valid_ratio"] = metrics["n_valid_aop"] / metrics["n_evaluable_aop"]
        metrics["aop_invalid_prediction_ratio"] = (
            metrics["aop_invalid_prediction_count"] / metrics["n_evaluable_aop"]
        )
    else:
        metrics["aop_valid_ratio"] = float("nan")
        metrics["aop_invalid_prediction_ratio"] = float("nan")
    metrics["aop_mae_valid_deg"] = (
        float(valid_errors.mean().item()) if valid_errors.numel() else float("nan")
    )
    penalized = torch.where(
        predicted_valid,
        angle_errors,
        torch.full_like(angle_errors, INVALID_AOP_PENALTY_DEG),
    )
    evaluable_errors = penalized[evaluable & torch.isfinite(penalized)]
    metrics["aop_mae_deg"] = (
        float(evaluable_errors.mean().item()) if evaluable_errors.numel() else float("nan")
    )
    metrics["aop_invalid_penalty_deg"] = INVALID_AOP_PENALTY_DEG
    return metrics


def train_mean_coordinate_baseline(
    train_dataset: IUGCLabeledDataset,
    validation_dataset: IUGCLabeledDataset,
) -> tuple[dict[str, float | int], Tensor]:
    """Fit one coordinate triplet on train labels and evaluate only on validation."""

    train_points = points_from_labeled_dataset(train_dataset)
    validation_points = points_from_labeled_dataset(validation_dataset)
    mean_points = train_points.mean(dim=0)
    predictions = mean_points.unsqueeze(0).expand(validation_points.shape[0], -1, -1).clone()
    valid_mask = torch.ones(validation_points.shape[:2], dtype=torch.bool)
    metrics = evaluate_coordinate_predictions(predictions, validation_points, valid_mask)
    return metrics, mean_points


def fixed_visualization_indices(
    length: int, *, role: str, count: int, seed: int
) -> tuple[int, ...]:
    """Select a stable, label-independent subset without exposing filenames."""

    if role not in {"train", "validation"}:
        raise PermissionError("Visualization role must be train or validation")
    if length <= 0 or count <= 0 or count > length or seed < 0:
        raise ValueError("Invalid deterministic visualization selection request")
    ranked = sorted(
        range(length),
        key=lambda index: hashlib.sha256(f"{seed}:{role}:{index}".encode()).digest(),
    )
    return tuple(sorted(ranked[:count]))


def _normalized_entropy(probabilities: Tensor) -> Tensor:
    flat = probabilities.flatten(start_dim=-2).to(dtype=torch.float64)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(flat.dtype).tiny)
    eps = torch.finfo(flat.dtype).tiny
    entropy = -(flat * flat.clamp_min(eps).log()).sum(dim=-1)
    return entropy / math.log(flat.shape[-1])


def _synthetic_case(
    *,
    case_id: str,
    heatmaps: Tensor,
    targets: Tensor,
    target_points_heatmap: Tensor,
    input_size_hw: tuple[int, int],
    align_corners: bool,
    method: str,
    temperature: float,
) -> dict[str, Any]:
    probabilities = spatial_softmax(heatmaps, temperature=temperature)
    if method == "argmax":
        decoded_normalized = decode_heatmaps(
            heatmaps,
            method="argmax",
            dsnt=DSNT(temperature=temperature, align_corners=align_corners),
            align_corners=align_corners,
        )
    elif method == "dsnt":
        decoded_normalized = spatial_expectation(probabilities, align_corners=align_corners)
    else:
        raise ValueError(f"Unsupported synthetic decoder: {method}")
    heatmap_size_hw = tuple(int(value) for value in heatmaps.shape[-2:])
    decoded_heatmap = normalized_to_pixel(
        decoded_normalized, heatmap_size_hw, align_corners=align_corners
    )
    decoded_input = resize_points(
        decoded_heatmap,
        heatmap_size_hw,
        input_size_hw,
        align_corners=align_corners,
    )
    target_input = resize_points(
        target_points_heatmap,
        heatmap_size_hw,
        input_size_hw,
        align_corners=align_corners,
    )
    valid_mask = torch.ones(decoded_input.shape[:2], dtype=torch.bool)
    coordinate_metrics = evaluate_coordinate_predictions(decoded_input, target_input, valid_mask)
    predicted_aop, predicted_valid = compute_aop(
        decoded_input,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
        output_unit="degrees",
        eps=OFFICIAL_AOP_EPS,
        invalid="mask",
    )
    valid_aop_mae = float(coordinate_metrics["aop_mae_valid_deg"])
    return {
        "case_id": case_id,
        "decoder": method,
        "temperature": float(temperature),
        "raw_heatmap_mse": float((heatmaps - targets).square().mean().item()),
        "zero_map_mse": float(targets.square().mean().item()),
        "probability_sum_max_abs_error": float(
            (probabilities.sum(dim=(-1, -2)) - 1.0).abs().max().item()
        ),
        "probability_entropy_normalized": float(_normalized_entropy(probabilities).mean().item()),
        "decoded_heatmap_xy": decoded_heatmap.squeeze(0).tolist(),
        "coordinate_error_input_px": {
            name: float(coordinate_metrics[f"MRE_{name}"]) for name in KEYPOINT_NAMES
        },
        "MRE_ALL": float(coordinate_metrics["MRE_ALL"]),
        "aop_official_valid": bool(predicted_valid.item()),
        "aop_predicted_deg": float(predicted_aop.item()) if bool(predicted_valid.item()) else None,
        "aop_mae_valid_deg": valid_aop_mae if math.isfinite(valid_aop_mae) else None,
        "aop_penalized_score_deg": float(coordinate_metrics["aop_mae_deg"]),
        "aop_invalid_penalty_deg": INVALID_AOP_PENALTY_DEG,
    }


def run_synthetic_sanity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Run the fixed three-channel Gaussian/DSNT/AoP matrix on CPU."""

    diagnostic = _mapping(protocol.get("diagnostic"), context="diagnostic")
    heatmap_size = tuple(int(value) for value in diagnostic["heatmap_size_hw"])
    input_size = tuple(int(value) for value in diagnostic["input_size_hw"])
    sigma = float(diagnostic["sigma_heatmap_px"])
    align_corners = bool(diagnostic["align_corners"])
    temperatures = [float(value) for value in diagnostic["synthetic_temperatures"]]
    if temperatures != [1.0, 0.05]:
        raise ValueError("Synthetic temperatures must be pre-registered as [1.0, 0.05]")
    amplitudes = [float(value) for value in diagnostic["synthetic_amplitudes"]]
    if amplitudes != [0.1, 0.01]:
        raise ValueError("Synthetic amplitudes must be pre-registered as [0.1, 0.01]")
    points = torch.tensor(
        diagnostic["synthetic_points_heatmap_xy"], dtype=torch.float32
    ).unsqueeze(0)
    if tuple(points.shape) != (1, 3, 2):
        raise ValueError("Synthetic sanity requires exactly three [x,y] points")
    targets = generate_gaussian_heatmaps(points, size_hw=heatmap_size, sigma=sigma)
    target_aop, target_valid = compute_aop(
        points,
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
        invalid="mask",
    )
    if not bool(target_valid.item()):
        raise ValueError("Synthetic points must define valid AoP geometry")

    cases = [
        _synthetic_case(
            case_id="gaussian_argmax",
            heatmaps=targets,
            targets=targets,
            target_points_heatmap=points,
            input_size_hw=input_size,
            align_corners=align_corners,
            method="argmax",
            temperature=0.05,
        ),
        *[
            _synthetic_case(
                case_id=f"gaussian_dsnt_t{temperature:g}",
                heatmaps=targets,
                targets=targets,
                target_points_heatmap=points,
                input_size_hw=input_size,
                align_corners=align_corners,
                method="dsnt",
                temperature=temperature,
            )
            for temperature in temperatures
        ],
        *[
            _synthetic_case(
                case_id=f"gaussian_amplitude_{amplitude:g}_dsnt_t0.05",
                heatmaps=targets * amplitude,
                targets=targets,
                target_points_heatmap=points,
                input_size_hw=input_size,
                align_corners=align_corners,
                method="dsnt",
                temperature=0.05,
            )
            for amplitude in amplitudes
        ],
        _synthetic_case(
            case_id="zero_heatmaps_dsnt_t0.05",
            heatmaps=torch.zeros_like(targets),
            targets=targets,
            target_points_heatmap=points,
            input_size_hw=input_size,
            align_corners=align_corners,
            method="dsnt",
            temperature=0.05,
        ),
        _synthetic_case(
            case_id="flat_heatmaps_dsnt_t0.05",
            heatmaps=torch.full_like(targets, 0.25),
            targets=targets,
            target_points_heatmap=points,
            input_size_hw=input_size,
            align_corners=align_corners,
            method="dsnt",
            temperature=0.05,
        ),
    ]
    return {
        "phase": "phase1a-b0-diagnostics",
        "status": "synthetic_only",
        "synthetic_geometry_valid": True,
        "target_aop_deg": float(target_aop.item()),
        "heatmap_size_hw": list(heatmap_size),
        "input_size_hw": list(input_size),
        "sigma_heatmap_px": sigma,
        "cases": cases,
        "interpretation_boundary": (
            "These cases validate math and interfaces only; they do not establish the cause of "
            "the saved B0 endpoint."
        ),
    }


def _finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = torch.tensor([value for value in values if math.isfinite(value)], dtype=torch.float64)
    if not finite.numel():
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


class HeatmapDiagnosticAccumulator:
    """Aggregate raw-heatmap and DSNT-probability evidence without sample IDs."""

    def __init__(
        self,
        *,
        keypoint_names: Sequence[str] = KEYPOINT_NAMES,
        temperature: float = 0.05,
        foreground_radius_px: float = 12.0,
        probability_mass_radius_px: float = 12.0,
        tie_atol: float = 1e-8,
        tie_rtol: float = 1e-6,
    ) -> None:
        if temperature <= 0 or foreground_radius_px <= 0 or probability_mass_radius_px <= 0:
            raise ValueError("Diagnostic temperature and radii must be positive")
        if tie_atol < 0 or tie_rtol < 0:
            raise ValueError("Tie tolerances must be non-negative")
        self.keypoint_names = tuple(keypoint_names)
        self.temperature = float(temperature)
        self.foreground_radius_px = float(foreground_radius_px)
        self.probability_mass_radius_px = float(probability_mass_radius_px)
        self.tie_atol = float(tie_atol)
        self.tie_rtol = float(tie_rtol)
        self.overall: defaultdict[str, list[float]] = defaultdict(list)
        self.channels: dict[str, defaultdict[str, list[float]]] = {
            name: defaultdict(list) for name in self.keypoint_names
        }
        self.pairs: defaultdict[str, list[float]] = defaultdict(list)

    @staticmethod
    def _extend(destination: list[float], values: Tensor) -> None:
        destination.extend(float(value) for value in values.detach().cpu().flatten().tolist())

    def add(
        self,
        logits: Tensor,
        targets: Tensor,
        target_points_heatmap_xy: Tensor,
        valid_mask: Tensor,
    ) -> None:
        if logits.shape != targets.shape or logits.ndim != 4:
            raise ValueError("Diagnostic logits and targets must share [B,K,H,W] shape")
        if logits.shape[1] != len(self.keypoint_names):
            raise ValueError("Diagnostic channel count differs from keypoint names")
        if target_points_heatmap_xy.shape != (*logits.shape[:2], 2):
            raise ValueError("target_points_heatmap_xy has the wrong shape")
        if valid_mask.shape != logits.shape[:2]:
            raise ValueError("valid_mask has the wrong shape")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Checkpoint emitted NaN or Inf heatmaps")

        batch, channels, height, width = logits.shape
        mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        squared = (logits - targets).square()
        per_keypoint_mse = squared.mean(dim=(-1, -2))
        zero_map_mse = targets.square().mean(dim=(-1, -2))
        y_grid = torch.arange(height, device=logits.device, dtype=logits.dtype).view(
            1, 1, height, 1
        )
        x_grid = torch.arange(width, device=logits.device, dtype=logits.dtype).view(
            1, 1, 1, width
        )
        x = target_points_heatmap_xy[..., 0, None, None]
        y = target_points_heatmap_xy[..., 1, None, None]
        distance_squared = (x_grid - x).square() + (y_grid - y).square()
        foreground = distance_squared <= self.foreground_radius_px**2
        probability_region = distance_squared <= self.probability_mass_radius_px**2
        foreground_mse = (squared * foreground).sum(dim=(-1, -2)) / foreground.sum(
            dim=(-1, -2)
        ).clamp_min(1)
        background = ~foreground
        background_mse = (squared * background).sum(dim=(-1, -2)) / background.sum(
            dim=(-1, -2)
        ).clamp_min(1)

        probabilities = spatial_softmax(logits, temperature=self.temperature)
        probability_mass = (probabilities * probability_region).sum(dim=(-1, -2))
        entropy = _normalized_entropy(probabilities)
        flattened = logits.flatten(start_dim=-2)
        raw_max = flattened.max(dim=-1).values
        raw_min = flattened.min(dim=-1).values
        raw_std = flattened.std(dim=-1, unbiased=False)
        probability_peak = probabilities.flatten(start_dim=-2).max(dim=-1).values
        tie_count = torch.isclose(
            flattened,
            raw_max.unsqueeze(-1),
            atol=self.tie_atol,
            rtol=self.tie_rtol,
        ).sum(dim=-1)
        top_two = torch.topk(flattened, k=2, dim=-1).values
        peak_gap = top_two[..., 0] - top_two[..., 1]
        fields = {
            "heatmap_mse": per_keypoint_mse,
            "zero_map_mse": zero_map_mse,
            "foreground_mse": foreground_mse,
            "background_mse": background_mse,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_std": raw_std,
            "raw_peak_tie_count": tie_count.to(dtype=logits.dtype),
            "raw_peak_gap": peak_gap,
            "probability_peak": probability_peak,
            "probability_entropy_normalized": entropy,
            "probability_mass_near_truth": probability_mass,
        }
        for field, values in fields.items():
            self._extend(self.overall[field], values[mask])
            for channel_index, name in enumerate(self.keypoint_names):
                channel_valid = mask[:, channel_index]
                self._extend(
                    self.channels[name][field], values[:, channel_index][channel_valid]
                )
        for left in range(channels):
            for right in range(left + 1, channels):
                pair_name = f"{self.keypoint_names[left]}__{self.keypoint_names[right]}"
                pairwise_mae = (logits[:, left] - logits[:, right]).abs().mean(dim=(-1, -2))
                self._extend(self.pairs[pair_name], pairwise_mae)
        if batch <= 0:
            raise ValueError("Diagnostic batch cannot be empty")

    def finalize(self) -> dict[str, Any]:
        overall = {name: _finite_summary(values) for name, values in sorted(self.overall.items())}
        zero_mean = overall["zero_map_mse"]["mean"]
        mse_mean = overall["heatmap_mse"]["mean"]
        if not isinstance(zero_mean, float) or zero_mean <= 0 or not isinstance(mse_mean, float):
            raise ValueError("Cannot compute zero-map MSE ratio")
        return {
            "overall": {
                **overall,
                "zero_map_mse_ratio": mse_mean / zero_mean,
            },
            "channels": {
                channel: {
                    name: _finite_summary(values) for name, values in sorted(fields.items())
                }
                for channel, fields in self.channels.items()
            },
            "channel_pairwise_raw_mae": {
                name: _finite_summary(values)
                for name, values in sorted(self.pairs.items())
            },
            "definitions": {
                "foreground_radius_heatmap_px": self.foreground_radius_px,
                "probability_mass_radius_heatmap_px": self.probability_mass_radius_px,
                "dsnt_temperature": self.temperature,
                "peak_tie_atol": self.tie_atol,
                "peak_tie_rtol": self.tie_rtol,
            },
        }


class _RayAccumulator:
    def __init__(self) -> None:
        self.pubic: list[float] = []
        self.fetal: list[float] = []
        self.valid_count = 0
        self.total_count = 0
        self.reasons: defaultdict[str, int] = defaultdict(int)

    def add(self, points: Tensor) -> None:
        points_cpu = points.detach().cpu()
        pubic = torch.linalg.vector_norm(points_cpu[:, 1] - points_cpu[:, 0], dim=-1)
        fetal = torch.linalg.vector_norm(points_cpu[:, 2] - points_cpu[:, 0], dim=-1)
        _, valid = compute_aop(
            points_cpu,
            vertex_index=0,
            pubic_axis_other_index=1,
            fetal_head_index=2,
            eps=OFFICIAL_AOP_EPS,
            invalid="mask",
        )
        self.pubic.extend(float(value) for value in pubic.tolist())
        self.fetal.extend(float(value) for value in fetal.tolist())
        self.valid_count += int(valid.sum().item())
        self.total_count += int(valid.numel())
        finite = torch.isfinite(points_cpu).all(dim=(-1, -2))
        for index in range(points_cpu.shape[0]):
            if bool(valid[index]):
                continue
            reasons: list[str] = []
            if not bool(finite[index]):
                reasons.append("nonfinite_points")
            if float(pubic[index]) <= OFFICIAL_AOP_EPS:
                reasons.append("pubic_ray_zero")
            if float(fetal[index]) <= OFFICIAL_AOP_EPS:
                reasons.append("fetal_ray_zero")
            self.reasons["+".join(reasons) if reasons else "unknown"] += 1

    def finalize(self) -> dict[str, Any]:
        valid_ratio = self.valid_count / self.total_count if self.total_count else float("nan")
        return {
            "pubic_ray_length_original_px": _finite_summary(self.pubic),
            "fetal_ray_length_original_px": _finite_summary(self.fetal),
            "official_eps_original_px": OFFICIAL_AOP_EPS,
            "official_valid_count": self.valid_count,
            "official_invalid_count": self.total_count - self.valid_count,
            "official_valid_ratio": valid_ratio,
            "official_invalid_ratio": 1.0 - valid_ratio,
            "official_invalid_reason_counts": dict(sorted(self.reasons.items())),
        }


def _normalized_to_original_pixels(
    normalized: Tensor, original_sizes_hw: Tensor, *, align_corners: bool
) -> Tensor:
    converted = []
    for points, size in zip(normalized, original_sizes_hw, strict=True):
        converted.append(
            normalized_to_pixel(
                points,
                (int(size[0].item()), int(size[1].item())),
                align_corners=align_corners,
            )
        )
    return torch.stack(converted)


def _private_ray_record(points: Tensor) -> dict[str, Any]:
    points_cpu = points.detach().cpu()
    pubic = float(torch.linalg.vector_norm(points_cpu[1] - points_cpu[0]).item())
    fetal = float(torch.linalg.vector_norm(points_cpu[2] - points_cpu[0]).item())
    _, valid = compute_aop(
        points_cpu.unsqueeze(0),
        vertex_index=0,
        pubic_axis_other_index=1,
        fetal_head_index=2,
        eps=OFFICIAL_AOP_EPS,
        invalid="mask",
    )
    reasons: list[str] = []
    if not bool(torch.isfinite(points_cpu).all()):
        reasons.append("nonfinite_points")
    if pubic <= OFFICIAL_AOP_EPS:
        reasons.append("pubic_ray_zero")
    if fetal <= OFFICIAL_AOP_EPS:
        reasons.append("fetal_ray_zero")
    return {
        "pubic_ray_length_original_px": pubic,
        "fetal_ray_length_original_px": fetal,
        "official_valid": bool(valid.item()),
        "official_invalid_reasons": reasons,
    }


@torch.inference_mode()
def diagnose_model_on_validation(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    *,
    dsnt: DSNT,
    device: torch.device,
    protocol: Mapping[str, Any],
    visualization_indices: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one model forward per validation batch and aggregate both decoders."""

    diagnostic = _mapping(protocol.get("diagnostic"), context="diagnostic")
    align_corners = bool(diagnostic["align_corners"])
    accumulator = HeatmapDiagnosticAccumulator(
        keypoint_names=tuple(diagnostic["keypoint_order"]),
        temperature=float(diagnostic["dsnt_temperature"]),
        foreground_radius_px=float(diagnostic["foreground_radius_heatmap_px"]),
        probability_mass_radius_px=float(
            diagnostic["probability_mass_radius_heatmap_px"]
        ),
        tie_atol=float(diagnostic["peak_tie_atol"]),
        tie_rtol=float(diagnostic["peak_tie_rtol"]),
    )
    predictions: dict[str, list[Tensor]] = {"dsnt": [], "argmax": []}
    ray_accumulators = {"dsnt": _RayAccumulator(), "argmax": _RayAccumulator()}
    targets: list[Tensor] = []
    valid_masks: list[Tensor] = []
    selected = set(int(index) for index in visualization_indices)
    private_samples: list[dict[str, Any]] = []
    seen = 0
    model.eval()
    for raw_batch in data_loader:
        batch = {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in raw_batch.items()
        }
        logits = model(batch["image"])
        accumulator.add(
            logits,
            batch["heatmaps"],
            batch["points_heatmap_px"],
            batch["valid_mask"],
        )
        decoded_original: dict[str, Tensor] = {}
        for method in ("dsnt", "argmax"):
            normalized = decode_heatmaps(
                logits,
                method=method,
                dsnt=dsnt,
                align_corners=align_corners,
            )
            original = _normalized_to_original_pixels(
                normalized,
                batch["original_size_hw"],
                align_corners=align_corners,
            )
            decoded_original[method] = original
            predictions[method].append(original.detach().cpu())
            ray_accumulators[method].add(original)
        targets.append(batch["points_original_px"].detach().cpu())
        valid_masks.append(batch["valid_mask"].detach().cpu().to(dtype=torch.bool))

        filenames = list(raw_batch["filename"])
        for batch_index, filename in enumerate(filenames):
            dataset_index = seen + batch_index
            if dataset_index not in selected:
                continue
            sample = {
                "dataset_index": dataset_index,
                "filename": str(filename),
                "image": batch["image"][batch_index].detach().cpu(),
                "target_heatmaps": batch["heatmaps"][batch_index].detach().cpu(),
                "predicted_heatmaps": logits[batch_index].detach().cpu(),
                "target_points_original_px": batch["points_original_px"][batch_index]
                .detach()
                .cpu(),
                "target_points_heatmap_px": batch["points_heatmap_px"][batch_index]
                .detach()
                .cpu(),
                "predicted_points_original_px": {
                    method: decoded_original[method][batch_index].detach().cpu()
                    for method in ("dsnt", "argmax")
                },
                "ray_diagnostics": {
                    method: _private_ray_record(decoded_original[method][batch_index])
                    for method in ("dsnt", "argmax")
                },
            }
            private_samples.append(sample)
        seen += int(logits.shape[0])

    if seen <= 0:
        raise ValueError("Validation loader produced no samples")
    if any(index >= seen for index in selected):
        raise ValueError("A visualization index exceeds the validation dataset")
    target_all = torch.cat(targets)
    valid_all = torch.cat(valid_masks)
    decoder_metrics = {
        method: evaluate_coordinate_predictions(
            torch.cat(values),
            target_all,
            valid_all,
            keypoint_names=tuple(diagnostic["keypoint_order"]),
            vertex_index=int(diagnostic["aop_vertex_index"]),
            pubic_axis_other_index=int(diagnostic["aop_pubic_axis_other_index"]),
            fetal_head_index=int(diagnostic["aop_fetal_head_index"]),
        )
        for method, values in predictions.items()
    }
    public = {
        "validation_sample_count": seen,
        "decoder_metrics": decoder_metrics,
        "heatmap_diagnostics": accumulator.finalize(),
        "ray_diagnostics": {
            method: ray_accumulators[method].finalize() for method in ("dsnt", "argmax")
        },
    }
    return public, private_samples


def load_phase06_model(
    spec: CheckpointSpec,
    *,
    device: torch.device,
    protocol: Mapping[str, Any],
) -> tuple[HeatmapUNet, Mapping[str, Any], str]:
    """Safely load and validate one frozen Phase 0.6 checkpoint."""

    checkpoint_hash = sha256_file(spec.path)
    if checkpoint_hash != spec.expected_sha256:
        raise PermissionError(f"Checkpoint hash mismatch for {spec.checkpoint_id}")
    payload = torch.load(spec.path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload is not a mapping: {spec.checkpoint_id}")
    required = {"format_version", "epoch", "seed", "config", "metrics", "model_state_dict"}
    if not required.issubset(payload):
        raise ValueError(f"Checkpoint is missing required fields: {spec.checkpoint_id}")
    if int(payload["format_version"]) != 1:
        raise ValueError(f"Checkpoint format mismatch for {spec.checkpoint_id}")
    if int(payload["epoch"]) != spec.epoch or int(payload["seed"]) != 42:
        raise ValueError(f"Checkpoint epoch/seed mismatch for {spec.checkpoint_id}")
    checkpoint_config = _mapping(payload["config"], context="checkpoint.config")
    if checkpoint_config.get("phase") != "phase0.6-long-budget-fidelity":
        raise ValueError(f"Checkpoint phase mismatch for {spec.checkpoint_id}")
    if checkpoint_config.get("variant") != spec.variant:
        raise ValueError(f"Checkpoint variant mismatch for {spec.checkpoint_id}")
    if checkpoint_config.get("testing_frozen") is not True:
        raise PermissionError(
            f"Checkpoint does not preserve frozen testing for {spec.checkpoint_id}"
        )
    selection = _mapping(checkpoint_config.get("selection"), context="checkpoint.config.selection")
    if selection.get("split") != "validation" or selection.get("common_decoder") != "dsnt":
        raise ValueError(f"Checkpoint selection contract mismatch for {spec.checkpoint_id}")
    training = _mapping(checkpoint_config.get("training"), context="checkpoint.config.training")
    diagnostic = _mapping(protocol.get("diagnostic"), context="diagnostic")
    expected = {
        "seed": 42,
        "input_size_hw": tuple(diagnostic["input_size_hw"]),
        "heatmap_size_hw": tuple(diagnostic["heatmap_size_hw"]),
        "keypoint_order": tuple(diagnostic["keypoint_order"]),
        "align_corners": True,
        "dsnt_temperature": 0.05,
        "base_channels": 8,
    }
    sequence_fields = {"input_size_hw", "heatmap_size_hw", "keypoint_order"}
    for key, value in expected.items():
        actual = training.get(key)
        if key in sequence_fields and isinstance(actual, Sequence) and not isinstance(
            actual, str | bytes
        ):
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"Checkpoint training field {key} mismatch for {spec.checkpoint_id}")
    model = HeatmapUNet(base_channels=8).to(device)
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError(f"model_state_dict is invalid for {spec.checkpoint_id}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload, checkpoint_hash


_SENSITIVE_PUBLIC_KEYS = {
    "filename",
    "filenames",
    "sample_id",
    "sample_ids",
    "selected_indices",
    "dataset_index",
    "checkpoint_path",
    "image_path",
    "image_dir",
    "labels_csv",
    "target_points_original_px",
    "target_points_heatmap_px",
    "predicted_points_original_px",
    "train_mean_coordinates",
}


def assert_public_aggregate(payload: Mapping[str, Any]) -> None:
    """Reject identifiers, real coordinates, and absolute paths from public reports."""

    def visit(value: Any, *, location: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key)
                if key_text.casefold() in _SENSITIVE_PUBLIC_KEYS:
                    raise PermissionError(f"Sensitive field {key_text!r} at {location}")
                visit(nested, location=f"{location}.{key_text}")
        elif isinstance(value, list | tuple):
            for index, nested in enumerate(value):
                visit(nested, location=f"{location}[{index}]")
        elif isinstance(value, Path):
            raise PermissionError(f"Path object is forbidden in public payload at {location}")
        elif isinstance(value, str):
            if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
                raise PermissionError(f"Absolute path is forbidden in public payload at {location}")

    visit(payload, location="root")
