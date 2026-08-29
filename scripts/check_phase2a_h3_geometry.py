#!/usr/bin/env python
"""Read-only H3 two-view geometry-gradient wiring check for Phase 2A.

This is an interface diagnostic, not training and not a semi-supervised result.
It reads one or two images from the configured *train* image directory without
opening the label CSV, restores the frozen Phase 1C H3 checkpoint, performs one
backward pass, and never constructs an optimizer or calls ``optimizer.step``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.data.access_policy import load_phase05_local_splits  # noqa: E402
from geoequi_ld.data.dataset import IUGCUnlabeledDataset  # noqa: E402
from geoequi_ld.geometry.transforms import (  # noqa: E402
    make_similarity_transform,
    warp_image,
)
from geoequi_ld.geometry_consistency import geometry_consistency_loss  # noqa: E402
from geoequi_ld.models import DSNT, HRNetW32SpecializedHeatmap  # noqa: E402
from geoequi_ld.training.checkpoints import read_checkpoint  # noqa: E402
from geoequi_ld.utils.hashing import sha256_file  # noqa: E402
from geoequi_ld.utils.io import write_json  # noqa: E402

DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "runs"
    / "phase1c"
    / "H3_specialized_B2_seed42_16e"
    / "best.pt"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "phase2a" / "h3_geometry_gradient_check.json"
PHASE2A_SEED = 42
MAX_DIAGNOSTIC_SAMPLES = 2
REQUIRED_COMPONENTS = (
    "backbone",
    "ps_enhancer",
    "fh_enhancer",
    "ps_decoder",
    "fh_decoder",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only H3 geometry-gradient check on 1-2 train images; "
            "no labels, optimizer step, testing, or evaluation"
        )
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase05_local.yaml",
        help="Canonical train/validation access policy; only train.image_dir is accessed",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def require_private_output_path(
    output_json: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Require a fresh JSON below the gitignored ``runs/phase2a`` root."""

    repository = Path(repository_root).resolve()
    private_root = (repository / "runs" / "phase2a").resolve()
    output = Path(output_json).resolve()
    if output.suffix.casefold() != ".json":
        raise ValueError("Phase 2A H3 diagnostic output must use a .json suffix")
    if output == private_root or not output.is_relative_to(private_root):
        raise PermissionError("Output must stay below the private gitignored runs/phase2a root")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing diagnostic: {output}")
    return output


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_diagnostic(seed: int = PHASE2A_SEED) -> dict[str, Any]:
    """Apply the inherited Phase 1C deterministic-warn-only boundary."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "bitwise_reproducibility_claimed": False,
        "deformconv_cuda_boundary": "warn_only_inherited_from_phase1c",
    }


def load_train_images_without_labels(
    local_config: str | Path,
    *,
    sample_count: int,
) -> Tensor:
    """Load only train image pixels; the configured label CSV is never opened."""

    if sample_count < 1 or sample_count > MAX_DIAGNOSTIC_SAMPLES:
        raise ValueError(f"sample_count must be between 1 and {MAX_DIAGNOSTIC_SAMPLES}")
    splits = load_phase05_local_splits(local_config)
    train_spec = splits["train"]
    dataset = IUGCUnlabeledDataset(
        image_dir=train_spec.image_dir,
        input_size_hw=(512, 512),
    )
    if len(dataset) < sample_count:
        raise ValueError(f"Train image directory contains fewer than {sample_count} images")
    images = [dataset[index]["image"] for index in range(sample_count)]
    return torch.stack(images)


def explicit_view_transforms(
    batch_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return two fixed, invertible, orientation-preserving similarity views."""

    if batch_size < 1 or batch_size > MAX_DIAGNOSTIC_SAMPLES:
        raise ValueError(f"batch_size must be between 1 and {MAX_DIAGNOSTIC_SAMPLES}")
    transform1 = make_similarity_transform(
        0.95,
        (0.04, -0.03),
        dtype=dtype,
        device=device,
    )
    transform2 = make_similarity_transform(
        1.05,
        (-0.03, 0.02),
        dtype=dtype,
        device=device,
    )
    return (
        transform1.unsqueeze(0).expand(batch_size, -1, -1).clone(),
        transform2.unsqueeze(0).expand(batch_size, -1, -1).clone(),
    )


def component_gradient_evidence(model: nn.Module) -> dict[str, dict[str, int | float | bool]]:
    """Summarize gradients without requiring every HRNet parameter to be active."""

    evidence: dict[str, dict[str, int | float | bool]] = {}
    for component_name in REQUIRED_COMPONENTS:
        component = getattr(model, component_name, None)
        if not isinstance(component, nn.Module):
            raise TypeError(f"Model has no nn.Module component {component_name!r}")
        parameters = [parameter for parameter in component.parameters() if parameter.requires_grad]
        gradients = [parameter.grad for parameter in parameters]
        present = [gradient for gradient in gradients if gradient is not None]
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in present)
        gradient_l1 = sum(float(gradient.detach().abs().sum().cpu()) for gradient in present)
        evidence[component_name] = {
            "trainable_parameter_tensor_count": len(parameters),
            "gradient_present_tensor_count": len(present),
            "gradient_missing_tensor_count": len(parameters) - len(present),
            "all_present_gradients_finite": finite,
            "gradient_l1": gradient_l1,
            "finite_nonzero_gradient_reached": bool(present and finite and gradient_l1 > 0.0),
        }
    return evidence


def gradient_gate_passed(evidence: Mapping[str, Mapping[str, int | float | bool]]) -> bool:
    return bool(
        set(evidence) == set(REQUIRED_COMPONENTS)
        and all(
            component.get("finite_nonzero_gradient_reached") is True
            for component in evidence.values()
        )
    )


def _checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("H3 checkpoint config must be a mapping")
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("H3 checkpoint config must contain model metadata")
    if config.get("testing_frozen") is not True:
        raise PermissionError("H3 checkpoint does not preserve testing_frozen=true")
    if model_config.get("class") != "HRNetW32SpecializedHeatmap":
        raise ValueError("Checkpoint is not the frozen Phase 1C H3 model")
    return {
        "epoch": int(payload["epoch"]),
        "seed": int(payload["seed"]),
        "model_class": str(model_config["class"]),
        "testing_frozen": True,
        "optimizer_state_restored": False,
    }


def run_h3_geometry_check(
    *,
    local_config: str | Path,
    checkpoint: str | Path,
    output_json: str | Path,
    sample_count: int,
    requested_device: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Execute one eval-mode backward pass and atomically save private evidence."""

    started = time.perf_counter()
    output = require_private_output_path(output_json, repository_root=repository_root)
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    if checkpoint_path.suffix.casefold() not in {".pt", ".pth", ".ckpt"}:
        raise ValueError("Checkpoint must be a PyTorch checkpoint file")
    checkpoint_digest_before = sha256_file(checkpoint_path)
    checkpoint_size_before = checkpoint_path.stat().st_size

    deterministic_policy = seed_diagnostic()
    device = resolve_device(requested_device)
    images = load_train_images_without_labels(local_config, sample_count=sample_count).to(device)
    transform1, transform2 = explicit_view_transforms(
        images.shape[0],
        dtype=images.dtype,
        device=device,
    )
    view1 = warp_image(images, transform1, align_corners=True)
    view2 = warp_image(images, transform2, align_corners=True)

    model = HRNetW32SpecializedHeatmap(align_corners=True)
    payload = read_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_contract = _checkpoint_contract(payload)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    dsnt = DSNT(temperature=0.05, align_corners=True).to(device).eval()

    model.zero_grad(set_to_none=True)
    prediction1 = dsnt(model(view1))
    prediction2 = dsnt(model(view2))
    # This all-visible mask is synthetic diagnostic scaffolding.  It is not a
    # claim that these structures are truly visible in an unlabelled image.
    diagnostic_visibility = torch.ones(
        (images.shape[0], 3),
        dtype=torch.bool,
        device=device,
    )
    geometry = geometry_consistency_loss(
        prediction1,
        prediction2,
        transform1,
        transform2,
        visibility_view1=diagnostic_visibility,
        visibility_view2=diagnostic_visibility,
        image_size_hw=(512, 512),
        coordinate_weight=0.1,
    )
    geometry.total_loss.backward()
    gradients = component_gradient_evidence(model)
    gate_passed = gradient_gate_passed(gradients) and not geometry.no_valid_geometry

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    checkpoint_digest_after = sha256_file(checkpoint_path)
    checkpoint_size_after = checkpoint_path.stat().st_size
    checkpoint_unchanged = bool(
        checkpoint_digest_before == checkpoint_digest_after
        and checkpoint_size_before == checkpoint_size_after
    )
    if not checkpoint_unchanged:
        raise PermissionError("Frozen H3 checkpoint bytes changed during a read-only check")
    elapsed_seconds = time.perf_counter() - started

    result: dict[str, Any] = {
        "schema_version": 1,
        "phase": "phase2a",
        "status": "passed" if gate_passed else "failed_gradient_gate",
        "claim_scope": "train_image_geometry_interface_check_not_training_or_evaluation",
        "formal_semi_supervised_result": False,
        "testing_read": False,
        "testing_evaluated": False,
        "labels_read": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "model_mode": "eval_with_autograd_enabled",
        "device": str(device),
        "runtime": {
            "wall_seconds": elapsed_seconds,
            "gpu_used": device.type == "cuda",
            "cuda_available_in_locked_runtime": torch.cuda.is_available(),
            "gpu_budget_minutes_used": elapsed_seconds / 60.0 if device.type == "cuda" else 0.0,
        },
        "sample_count": int(images.shape[0]),
        "input_shape": list(images.shape),
        "heatmap_shape": [int(images.shape[0]), 3, 256, 256],
        "prediction_shape": list(prediction1.shape),
        "view_transforms": {
            "direction": "normalized_original_to_normalized_view",
            "orientation_preserving": True,
            "invertible": True,
            "view1": {"uniform_scale": 0.95, "translation_normalized_xy": [0.04, -0.03]},
            "view2": {"uniform_scale": 1.05, "translation_normalized_xy": [-0.03, 0.02]},
        },
        "visibility": {
            "strategy": "synthetic_all_visible_diagnostic_mask",
            "true_visibility_claimed": False,
            "ground_truth_used": False,
        },
        "geometry": {
            "angle_loss_degrees": float(geometry.angle_loss.detach().cpu()),
            "coordinate_loss_normalized": float(geometry.coordinate_loss.detach().cpu()),
            "total_loss": float(geometry.total_loss.detach().cpu()),
            "definition": "angle_degrees + 0.1 * normalized_coordinate_distance",
            "valid_point_count": geometry.valid_point_count,
            "valid_angle_count": geometry.valid_angle_count,
            "skip_reason": geometry.skip_reason,
        },
        "gradient_gate_passed": gate_passed,
        "component_gradients": gradients,
        "checkpoint": {
            **checkpoint_contract,
            "sha256_before": checkpoint_digest_before,
            "sha256_after": checkpoint_digest_after,
            "size_bytes_before": checkpoint_size_before,
            "size_bytes_after": checkpoint_size_after,
            "bytes_unchanged": checkpoint_unchanged,
            "checkpoint_written": False,
        },
        "determinism": deterministic_policy,
    }
    write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_h3_geometry_check(
        local_config=args.local_config,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        sample_count=args.sample_count,
        requested_device=args.device,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
