"""Bounded, fail-closed Phase 1C specialized-enhancer runners.

Only the verified train and validation splits are accepted.  Every data-derived
artifact stays below ignored ``runs/phase1c`` or ``artifacts/phase1c`` paths;
testing is neither an argument nor a discoverable role in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from geoequi_ld.models.dsnt import DSNT
from geoequi_ld.models.hrnet import (
    HRNetW32SharedHeatmap,
    HRNetW32SplitHeatmap,
    count_trainable_parameters,
)
from geoequi_ld.models.specialized import (
    HRNetW32SpecializedHeatmap,
    PSFeatureEnhancer,
)
from geoequi_ld.training.budget import GpuBudgetLedger, WallClockBudget
from geoequi_ld.training.checkpoints import save_checkpoint
from geoequi_ld.training.engine import (
    evaluate_model,
    train_for_steps_bounded,
    train_one_epoch,
    write_history_csv,
    write_json,
)
from geoequi_ld.training.phase1a_runners import (
    TRAIN_SAMPLE_COUNT,
    VALIDATION_SAMPLE_COUNT,
    _dataset,
    b4_gate_passed,
    evaluate_tiny_mode,
    fingerprint_digest,
    save_tiny_prediction_visualizations,
    select_preregistered_tiny_indices,
)
from geoequi_ld.training.phase1b_runners import (
    _capture_resume_state,
    _evaluate_key_checkpoints,
    _file_binding,
    _runtime_environment,
    _save_model_only,
    _tensor_state_digest,
    load_verified_phase1b_data,
)
from geoequi_ld.training.phase1c_config import (
    Phase1CProtocolConfig,
    build_phase1c_adam,
    load_phase1c_config,
)
from geoequi_ld.training.runtime import make_generator, seed_everything

PHASE1C_SEED = 42
TINY_SAMPLE_COUNT = 4
MILESTONE_FORMAT_VERSION = 1
EPOCH_START_SAFETY_SECONDS = 1200.0
TINY_OVERLAY_FILENAMES = tuple(f"sample_{index:02d}.png" for index in range(4))
RUNTIME_SOURCE_FILENAMES = (
    "src/geoequi_ld/data/access_policy.py",
    "src/geoequi_ld/data/dataset.py",
    "src/geoequi_ld/data/heatmaps.py",
    "src/geoequi_ld/geometry/aop.py",
    "src/geoequi_ld/geometry/coordinates.py",
    "src/geoequi_ld/geometry/transforms.py",
    "src/geoequi_ld/metrics/keypoints.py",
    "src/geoequi_ld/models/decoding.py",
    "src/geoequi_ld/models/dsnt.py",
    "src/geoequi_ld/models/hrnet.py",
    "src/geoequi_ld/models/specialized.py",
    "src/geoequi_ld/training/budget.py",
    "src/geoequi_ld/training/checkpoints.py",
    "src/geoequi_ld/training/config.py",
    "src/geoequi_ld/training/engine.py",
    "src/geoequi_ld/training/phase1a_runners.py",
    "src/geoequi_ld/training/phase1b_runners.py",
    "src/geoequi_ld/training/phase1c_config.py",
    "src/geoequi_ld/training/phase1c_runners.py",
    "src/geoequi_ld/training/runtime.py",
)


def _contains_forbidden_component(path: str | Path) -> bool:
    return any(part.casefold() in {"test", "testing"} for part in Path(path).parts)


def require_phase1c_fresh_output(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Create one fresh private Phase 1C output without touching older phases."""

    root = Path(repository_root).resolve()
    destination = Path(path).resolve()
    allowed = (
        (root / "runs" / "phase1c").resolve(),
        (root / "artifacts" / "phase1c").resolve(),
    )
    if not any(destination != base and destination.is_relative_to(base) for base in allowed):
        raise PermissionError(
            "Phase 1C output must be a child of runs/phase1c or artifacts/phase1c"
        )
    if _contains_forbidden_component(destination):
        raise PermissionError("Phase 1C refuses a test/testing output path")
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Output path is not a directory: {destination}")
        if next(destination.iterdir(), None) is not None:
            raise FileExistsError(f"Refusing to reuse non-empty output directory: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def _require_private_phase1c_file(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    root = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    allowed = (
        (root / "runs" / "phase1c").resolve(),
        (root / "artifacts" / "phase1c").resolve(),
    )
    if not any(candidate.is_relative_to(base) for base in allowed):
        raise PermissionError("Phase 1C artifact must stay below its private directories")
    if _contains_forbidden_component(candidate):
        raise PermissionError("Phase 1C refuses a test/testing artifact path")
    if not candidate.is_file():
        raise FileNotFoundError(f"Phase 1C artifact does not exist: {candidate}")
    return candidate


def _phase1c_ledger(
    path: str | Path,
    *,
    repository_root: str | Path,
    config: Phase1CProtocolConfig,
) -> GpuBudgetLedger:
    root = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    canonical = (root / "runs" / "phase1c" / "gpu_budget.json").resolve()
    if candidate != canonical:
        raise PermissionError(
            "All Phase 1C GPU work must use canonical runs/phase1c/gpu_budget.json"
        )
    return GpuBudgetLedger(
        candidate,
        total_limit_seconds=float(config.resources.total_gpu_max_seconds),
    )


def _finish_ledger(
    ledger: GpuBudgetLedger,
    name: str,
    *,
    started: float,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    before = ledger.snapshot()
    active = before.get("active_run")
    if active is None:
        return before
    if not isinstance(active, Mapping) or active.get("name") != name:
        raise RuntimeError(f"Unexpected active Phase 1C GPU ledger entry: {active}")
    allocation = float(active["allocated_seconds"])
    elapsed = time.perf_counter() - started
    aggregate_after = float(before["used_seconds"]) + elapsed
    allocation_exceeded = elapsed > allocation
    aggregate_exceeded = aggregate_after > float(before["total_limit_seconds"])
    effective_status = (
        "budget_exhausted" if allocation_exceeded or aggregate_exceeded else status
    )
    return ledger.finish(
        name,
        elapsed_seconds=elapsed,
        status=effective_status,
        details={
            **dict(details or {}),
            "allocation_exceeded": allocation_exceeded,
            "aggregate_limit_exceeded": aggregate_exceeded,
        },
    )


def _ledger_run_binding(
    snapshot: Mapping[str, Any],
    name: str,
    *,
    run_index: int | None = None,
) -> dict[str, Any]:
    runs = snapshot.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Phase 1C ledger has no run list")
    if run_index is None:
        matches = [
            (index, entry)
            for index, entry in enumerate(runs)
            if entry.get("name") == name
        ]
        if not matches:
            raise ValueError(f"Phase 1C ledger has no {name!r} entry")
        index, entry = matches[-1]
    else:
        if run_index < 0 or run_index >= len(runs):
            raise ValueError(f"Phase 1C ledger index is out of range: {run_index}")
        index, entry = run_index, runs[run_index]
        if not isinstance(entry, Mapping) or entry.get("name") != name:
            raise ValueError(f"Phase 1C ledger index {run_index} is not {name}")
    canonical = json.dumps(
        entry,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "ledger_relative_path": "runs/phase1c/gpu_budget.json",
        "run_index": index,
        "entry": dict(entry),
        "entry_sha256": hashlib.sha256(canonical).hexdigest(),
        "total_limit_seconds": float(snapshot["total_limit_seconds"]),
    }


def _require_current_ledger_binding(
    binding: Mapping[str, Any],
    *,
    name: str,
    ledger_path: str | Path,
    repository_root: str | Path,
    config: Phase1CProtocolConfig,
    required_details: Mapping[str, Any],
) -> None:
    ledger = _phase1c_ledger(
        ledger_path,
        repository_root=repository_root,
        config=config,
    )
    snapshot = ledger.snapshot()
    try:
        index = int(binding["run_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError(f"Invalid GPU ledger run index for {name}") from error
    expected = _ledger_run_binding(snapshot, name, run_index=index)
    if dict(binding) != expected:
        raise PermissionError(f"Stale or invented GPU ledger binding for {name}")
    entry = expected["entry"]
    if entry.get("status") != "completed":
        raise PermissionError(f"Bound GPU ledger entry for {name} is not completed")
    details = entry.get("details")
    if not isinstance(details, Mapping) or any(
        details.get(key) != value for key, value in required_details.items()
    ):
        raise PermissionError(f"Bound GPU ledger details do not prove {name} outcome")
    if float(entry["elapsed_seconds"]) > float(entry["allocated_seconds"]):
        raise PermissionError(f"Recorded {name} elapsed time exceeds its allocation")
    if float(snapshot["used_seconds"]) > float(snapshot["total_limit_seconds"]):
        raise PermissionError("Phase 1C aggregate GPU ledger exceeds 10800 seconds")


def _runtime_source_binding(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    files = {
        name: _file_binding(root / Path(name), logical_name=name)
        for name in RUNTIME_SOURCE_FILENAMES
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "src/geoequi_ld"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        head, diff = "unavailable", "unavailable"
    return {
        "files": files,
        "git_head": head,
        "tracked_source_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "tracked_source_diff_size_bytes": len(diff),
    }


def _module_state_equal(source: nn.Module, target: nn.Module) -> bool:
    source_state = source.state_dict()
    target_state = target.state_dict()
    return bool(
        source_state.keys() == target_state.keys()
        and all(torch.equal(source_state[name], target_state[name]) for name in source_state)
    )


def _storage_pointers(tensors: Iterable[Tensor]) -> set[int]:
    return {tensor.untyped_storage().data_ptr() for tensor in tensors}


def _parameters_and_buffers(module: nn.Module) -> Iterable[Tensor]:
    yield from module.parameters()
    yield from module.buffers()


def _component_parameter_counts(model: HRNetW32SpecializedHeatmap) -> dict[str, int]:
    return {
        "backbone": count_trainable_parameters(model.backbone),
        "ps_enhancer": count_trainable_parameters(model.ps_enhancer),
        "fh_enhancer": count_trainable_parameters(model.fh_enhancer),
        "ps_decoder": count_trainable_parameters(model.ps_decoder),
        "fh_decoder": count_trainable_parameters(model.fh_decoder),
        "total": count_trainable_parameters(model),
    }


def build_phase1c_initialization() -> tuple[
    HRNetW32SplitHeatmap,
    HRNetW32SpecializedHeatmap,
    dict[str, Any],
]:
    """Create H2 and H3 from the same untrained seed-42 shared base on CPU."""

    seed_everything(PHASE1C_SEED, deterministic=True)
    shared = HRNetW32SharedHeatmap(align_corners=True).eval()
    h2 = HRNetW32SplitHeatmap.from_shared(shared).eval()
    del shared
    h3 = HRNetW32SpecializedHeatmap.from_split(h2).eval()
    base_equal = {
        "backbone": _module_state_equal(h2.backbone, h3.backbone),
        "ps_decoder": _module_state_equal(h2.ps_decoder, h3.ps_decoder),
        "fh_decoder": _module_state_equal(h2.fh_decoder, h3.fh_decoder),
    }
    source_storage = _storage_pointers(_parameters_and_buffers(h2))
    copied_storage: set[int] = set()
    for module in (h3.backbone, h3.ps_decoder, h3.fh_decoder):
        copied_storage.update(_storage_pointers(_parameters_and_buffers(module)))
    probe = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        h2_output = h2(probe)
        h3_output = h3(probe)
    h2_parameters = count_trainable_parameters(h2)
    h3_parameters = count_trainable_parameters(h3)
    metadata = {
        "seed": PHASE1C_SEED,
        "method": "HRNetW32SpecializedHeatmap.from_split",
        "h2_backbone_state_sha256": _tensor_state_digest(h2.backbone.state_dict()),
        "h3_backbone_state_sha256": _tensor_state_digest(h3.backbone.state_dict()),
        "h2_ps_decoder_state_sha256": _tensor_state_digest(h2.ps_decoder.state_dict()),
        "h3_ps_decoder_state_sha256": _tensor_state_digest(h3.ps_decoder.state_dict()),
        "h2_fh_decoder_state_sha256": _tensor_state_digest(h2.fh_decoder.state_dict()),
        "h3_fh_decoder_state_sha256": _tensor_state_digest(h3.fh_decoder.state_dict()),
        "h3_complete_state_sha256": _tensor_state_digest(h3.state_dict()),
        "base_state_values_equal": base_equal,
        "base_parameter_storage_aliased": not source_storage.isdisjoint(copied_storage),
        "h2_trainable_parameters": h2_parameters,
        "h3_trainable_parameters": h3_parameters,
        "additional_trainable_parameters": h3_parameters - h2_parameters,
        "h3_component_trainable_parameters": _component_parameter_counts(h3),
        "probe_input_shape": list(probe.shape),
        "probe_output_shape": list(h3_output.shape),
        "complete_function_maximum_absolute_difference": float(
            (h3_output - h2_output).abs().max()
        ),
        "complete_function_equivalent": bool(
            torch.allclose(h3_output, h2_output, rtol=1.0e-5, atol=1.0e-6)
        ),
        "enhancer_initialization": h3.initialization_summary,
        "fairness_statement": (
            "The backbone and split decoders start from identical values, but the new "
            "specialized enhancers make the complete H3 function non-equivalent to H2."
        ),
    }
    if not all(base_equal.values()) or metadata["base_parameter_storage_aliased"]:
        raise RuntimeError("H2-to-H3 base initialization failed its value/storage contract")
    return h2, h3, metadata


def _gradient_sum(module: nn.Module) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        return 0.0
    return sum(float(gradient.abs().sum().detach().cpu()) for gradient in gradients)


def _finite_gradient_l1(gradient: Tensor | None) -> float:
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        return 0.0
    return float(gradient.abs().sum().detach().cpu())


def _all_parameter_gradients_finite(module: nn.Module) -> bool:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    return bool(
        parameters
        and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
    )


def _predictor_gradient_evidence(enhancer: PSFeatureEnhancer) -> dict[str, float]:
    gradient = enhancer.offset_mask.weight.grad
    bias_gradient = enhancer.offset_mask.bias.grad
    if gradient is None or bias_gradient is None:
        return {"offset": 0.0, "mask": 0.0}
    return {
        "offset": _finite_gradient_l1(gradient[:18])
        + _finite_gradient_l1(bias_gradient[:18]),
        "mask": _finite_gradient_l1(gradient[18:])
        + _finite_gradient_l1(bias_gradient[18:]),
    }


def _all_finite(values: Iterable[Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def run_phase1c_deform_operator_probe(
    *,
    phase1c_config: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Exercise real modulated DeformConv2d CUDA forward/backward deterministically."""

    protocol = load_phase1c_config(phase1c_config)
    output = require_phase1c_fresh_output(output_dir, repository_root=repository_root)
    if not torch.cuda.is_available():
        result = {"schema_version": 1, "status": "cuda_unavailable", "gate": "BLOCKED"}
        write_json(output / "operator_probe.json", result)
        return result
    ledger = _phase1c_ledger(
        ledger_path,
        repository_root=repository_root,
        config=protocol,
    )
    allocation = ledger.begin(
        "P1C_deform_cuda_probe",
        requested_limit_seconds=float(protocol.resources.operator_probe_max_seconds),
    )
    started = time.perf_counter()
    try:
        seed_everything(PHASE1C_SEED, deterministic=True)
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        enhancer = PSFeatureEnhancer().to(device).train()
        inputs = torch.randn((1, 32, 32, 32), device=device, dtype=torch.float32)
        offsets, masks, mask_logits = enhancer.predict_offset_and_mask(inputs)
        output_tensor = enhancer(inputs)
        weights = torch.linspace(
            -0.7,
            1.3,
            output_tensor.numel(),
            device=device,
            dtype=output_tensor.dtype,
        ).reshape_as(output_tensor)
        (output_tensor * weights).mean().backward()
        torch.cuda.synchronize(device)
        predictor_gradients = _predictor_gradient_evidence(enhancer)
        deform_gradient = _finite_gradient_l1(enhancer.deform.weight.grad)
        evidence = {
            "input_shape": list(inputs.shape),
            "offset_shape": list(offsets.shape),
            "mask_logits_shape": list(mask_logits.shape),
            "mask_shape": list(masks.shape),
            "output_shape": list(output_tensor.shape),
            "initial_offset_max_abs": float(offsets.abs().max().detach().cpu()),
            "initial_mask_logits_max_abs": float(mask_logits.abs().max().detach().cpu()),
            "initial_mask_min": float(masks.min().detach().cpu()),
            "initial_mask_max": float(masks.max().detach().cpu()),
            "offset_predictor_gradient_l1": predictor_gradients["offset"],
            "mask_predictor_gradient_l1": predictor_gradients["mask"],
            "deform_weight_gradient_l1": deform_gradient,
            "spatial_attention_gradient_l1": _gradient_sum(enhancer.spatial_attention),
            "all_values_finite": _all_finite(
                (offsets, mask_logits, masks, output_tensor)
            ),
            "all_enhancer_gradients_finite": _all_parameter_gradients_finite(enhancer),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "actual_operator": (
                f"{enhancer.deform.__class__.__module__}."
                f"{enhancer.deform.__class__.__name__}"
            ),
            "ordinary_conv_fallback": False,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        passed = bool(
            evidence["offset_shape"] == [1, 18, 32, 32]
            and evidence["mask_shape"] == [1, 9, 32, 32]
            and evidence["output_shape"] == [1, 32, 32, 32]
            and evidence["initial_offset_max_abs"] == 0.0
            and evidence["initial_mask_logits_max_abs"] == 0.0
            and evidence["initial_mask_min"] == evidence["initial_mask_max"] == 0.5
            and predictor_gradients["offset"] > 0
            and predictor_gradients["mask"] > 0
            and deform_gradient > 0
            and evidence["all_values_finite"]
            and evidence["all_enhancer_gradients_finite"]
            and evidence["deterministic_algorithms_enabled"]
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "gate_id": "P1C_deform_cuda_probe",
            "status": "completed",
            "gate": "PASS" if passed else "FAIL",
            "runtime_elapsed_seconds": time.perf_counter() - started,
            "runtime_allocated_seconds": allocation,
            "evidence": evidence,
            "protocol_config_binding": _file_binding(
                phase1c_config,
                logical_name="configs/phase1c_specialized_enhancers.yaml",
            ),
            "environment": _runtime_environment(repository_root),
            "runtime_source_binding": _runtime_source_binding(repository_root),
        }
        ledger_snapshot = _finish_ledger(
            ledger,
            "P1C_deform_cuda_probe",
            started=started,
            status="completed" if passed else "failed",
            details={"gate": result["gate"]},
        )
        result["gpu_ledger_binding"] = _ledger_run_binding(
            ledger_snapshot,
            "P1C_deform_cuda_probe",
        )
        if result["gpu_ledger_binding"]["entry"]["status"] != "completed":
            result["gate"] = "FAIL"
        write_json(output / "operator_probe.json", result)
        return result
    except Exception as error:
        ledger_snapshot = _finish_ledger(
            ledger,
            "P1C_deform_cuda_probe",
            started=started,
            status="failed",
            details={"error_type": type(error).__name__},
        )
        result = {
            "schema_version": 1,
            "gate_id": "P1C_deform_cuda_probe",
            "status": "blocked",
            "gate": "BLOCKED",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "ordinary_conv_fallback_attempted": False,
            "gpu_ledger_binding": _ledger_run_binding(
                ledger_snapshot,
                "P1C_deform_cuda_probe",
            ),
        }
        write_json(output / "operator_probe.json", result)
        return result


def _require_passed_operator_probe(
    artifact: str | Path,
    *,
    phase1c_config: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    protocol = load_phase1c_config(phase1c_config)
    path = _require_private_phase1c_file(artifact, repository_root=repository_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != "PASS" or payload.get("status") != "completed":
        raise PermissionError("Phase 1C requires a passing CUDA DeformConv2d probe")
    if payload.get("protocol_config_binding") != _file_binding(
        phase1c_config,
        logical_name="configs/phase1c_specialized_enhancers.yaml",
    ):
        raise PermissionError("Phase 1C operator probe config binding is stale")
    if payload.get("runtime_source_binding") != _runtime_source_binding(repository_root):
        raise PermissionError("Phase 1C operator probe source binding is stale")
    if payload.get("environment") != _runtime_environment(repository_root):
        raise PermissionError("Phase 1C operator probe environment binding is stale")
    binding = payload.get("gpu_ledger_binding")
    if not isinstance(binding, Mapping):
        raise PermissionError("Phase 1C operator probe has no ledger binding")
    _require_current_ledger_binding(
        binding,
        name="P1C_deform_cuda_probe",
        ledger_path=ledger_path,
        repository_root=repository_root,
        config=protocol,
        required_details={"gate": "PASS"},
    )
    return payload


def _structure_probe(
    model: HRNetW32SpecializedHeatmap,
    dsnt: DSNT,
    *,
    device: torch.device,
) -> dict[str, Any]:
    captured: dict[str, list[int]] = {}

    def capture(name: str):  # type: ignore[no-untyped-def]
        def hook(_module: nn.Module, arguments: tuple[Tensor, ...], value: Tensor) -> None:
            captured[f"{name}_input"] = list(arguments[0].shape)
            captured[f"{name}_output"] = list(value.shape)

        return hook

    hooks = (
        model.ps_enhancer.register_forward_hook(capture("ps_enhancer")),
        model.fh_enhancer.register_forward_hook(capture("fh_enhancer")),
        model.ps_decoder.register_forward_hook(capture("ps_decoder")),
        model.fh_decoder.register_forward_hook(capture("fh_decoder")),
    )
    model.eval()
    inputs = torch.zeros((1, 1, 512, 512), device=device, dtype=torch.float32)
    with torch.inference_mode():
        features = model.extract_high_resolution_features(inputs)
        output = model(inputs)
        coordinates = dsnt(output)
    for hook in hooks:
        hook.remove()
    expected = {
        "feature": [1, 32, 128, 128],
        "ps_enhancer_input": [1, 32, 128, 128],
        "ps_enhancer_output": [1, 32, 128, 128],
        "fh_enhancer_input": [1, 32, 128, 128],
        "fh_enhancer_output": [1, 32, 128, 128],
        "ps_decoder_input": [1, 32, 128, 128],
        "ps_decoder_output": [1, 2, 128, 128],
        "fh_decoder_input": [1, 32, 128, 128],
        "fh_decoder_output": [1, 1, 128, 128],
        "output": [1, 3, 256, 256],
        "dsnt": [1, 3, 2],
    }
    actual = {
        "feature": list(features.shape),
        **captured,
        "output": list(output.shape),
        "dsnt": list(coordinates.shape),
    }
    return {
        "input_shape": list(inputs.shape),
        "actual_shapes": actual,
        "expected_shapes": expected,
        "shape_contract_passed": actual == expected,
        "all_values_finite": _all_finite((features, output, coordinates)),
        "channel_order": ["PS1", "PS2", "FH1"],
    }


def _tiny_gradient_evidence(model: HRNetW32SpecializedHeatmap) -> dict[str, Any]:
    predictor = _predictor_gradient_evidence(model.ps_enhancer)
    evidence = {
        "backbone_gradient_l1": _gradient_sum(model.backbone),
        "ps_offset_predictor_gradient_l1": predictor["offset"],
        "ps_mask_predictor_gradient_l1": predictor["mask"],
        "ps_deform_weight_gradient_l1": _finite_gradient_l1(
            model.ps_enhancer.deform.weight.grad
        ),
        "ps_enhancer_gradient_l1": _gradient_sum(model.ps_enhancer),
        "ps_decoder_gradient_l1": _gradient_sum(model.ps_decoder),
        "fh_enhancer_gradient_l1": _gradient_sum(model.fh_enhancer),
        "fh_decoder_gradient_l1": _gradient_sum(model.fh_decoder),
    }
    evidence["all_required_gradients_finite"] = all(
        _all_parameter_gradients_finite(module)
        for module in (
            model.backbone,
            model.ps_enhancer,
            model.ps_decoder,
            model.fh_enhancer,
            model.fh_decoder,
        )
    )
    numeric = [value for value in evidence.values() if isinstance(value, int | float)]
    evidence["all_required_nonzero"] = bool(
        evidence["all_required_gradients_finite"]
        and all(math.isfinite(float(value)) and float(value) > 0 for value in numeric)
    )
    return evidence


def run_phase1c_tiny(
    *,
    local_config: str | Path,
    phase1c_config: str | Path,
    operator_probe_artifact: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Run the fixed four-sample H3 implementation gate from fresh initialization."""

    protocol = load_phase1c_config(phase1c_config)
    operator_probe = _require_passed_operator_probe(
        operator_probe_artifact,
        phase1c_config=phase1c_config,
        ledger_path=ledger_path,
        repository_root=repository_root,
    )
    verified = load_verified_phase1b_data(
        local_config,
        repository_root=repository_root,
    )
    config = protocol.training
    train_dataset = _dataset(verified.specs["train"], config)
    indices = select_preregistered_tiny_indices(len(train_dataset))
    subset = Subset(train_dataset, indices)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=make_generator(PHASE1C_SEED + 1),
        pin_memory=True,
    )
    evaluation_loader: DataLoader[dict[str, Any]] = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    output = require_phase1c_fresh_output(output_dir, repository_root=repository_root)
    if not torch.cuda.is_available():
        result = {"schema_version": 1, "status": "cuda_unavailable", "gate": "FAIL"}
        write_json(output / "tiny_result.json", result)
        return result
    h2, model, initialization = build_phase1c_initialization()
    del h2
    ledger = _phase1c_ledger(
        ledger_path,
        repository_root=repository_root,
        config=protocol,
    )
    allocation = ledger.begin(
        "P1C_specialized_tiny_B2",
        requested_limit_seconds=float(protocol.resources.tiny_max_seconds),
    )
    diagnostic_reserve = min(180.0, allocation * 0.1)
    training_allocation = allocation - diagnostic_reserve
    started = time.perf_counter()
    try:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        model = model.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer = build_phase1c_adam(model.parameters(), protocol)
        dsnt = DSNT(
            temperature=config.dsnt_temperature,
            align_corners=config.align_corners,
        ).to(device)
        structure = _structure_probe(model, dsnt, device=device)
        if not (
            structure["shape_contract_passed"] and structure["all_values_finite"]
        ):
            result = {
                "schema_version": 1,
                "gate_id": "P1C_specialized_tiny_B2",
                "gate": "FAIL",
                "programmatic_gate": "FAIL",
                "status": "structure_failed",
                "steps_completed": 0,
                "max_steps": protocol.resources.tiny_max_steps,
                "training_elapsed_seconds": 0.0,
                "training_allocated_seconds": training_allocation,
                "total_allocated_seconds": allocation,
                "initialization": initialization,
                "operator_probe_binding": _file_binding(
                    operator_probe_artifact,
                    logical_name="operator_probe.json",
                ),
                "operator_probe_gate": operator_probe["gate"],
                "structure_probe": structure,
                "data_fingerprint_digest": fingerprint_digest(verified.fingerprints),
                "protocol_config_binding": _file_binding(
                    phase1c_config,
                    logical_name="configs/phase1c_specialized_enhancers.yaml",
                ),
                "environment": _runtime_environment(repository_root),
                "runtime_source_binding": _runtime_source_binding(repository_root),
                "augmentation": "disabled",
                "batch_size": 1,
                "precision": "float32",
                "loss_application": (
                    "concatenate_[PS1,PS2,FH1]_then_existing_B2_once"
                ),
            }
            result["total_elapsed_seconds"] = time.perf_counter() - started
            result["within_total_allocation"] = (
                result["total_elapsed_seconds"] <= allocation
            )
            ledger_snapshot = _finish_ledger(
                ledger,
                "P1C_specialized_tiny_B2",
                started=started,
                status="completed",
                details={
                    "gate": "FAIL",
                    "programmatic_gate": "FAIL",
                    "steps_completed": 0,
                    "failure_reason": "structure_probe",
                },
            )
            result["gpu_ledger_binding"] = _ledger_run_binding(
                ledger_snapshot,
                "P1C_specialized_tiny_B2",
            )
            write_json(output / "tiny_result.json", result)
            return result
        bounded = train_for_steps_bounded(
            model,
            train_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
            max_steps=protocol.resources.tiny_max_steps,
            max_runtime_seconds=training_allocation,
        )
        torch.cuda.synchronize(device)
        gradients = _tiny_gradient_evidence(model)
        eval_metrics, private_rows = evaluate_tiny_mode(
            model,
            evaluation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            mode="eval",
        )
        visualization = save_tiny_prediction_visualizations(
            model,
            evaluation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            private_run_root=output,
            repository_root=repository_root,
        )
        coordinate_checks = visualization["programmatic_coordinate_checks"]
        visual_pass = bool(
            visualization["visualization_count"] == TINY_SAMPLE_COUNT
            and coordinate_checks["all_coordinates_finite"]
            and coordinate_checks["all_targets_in_input_bounds"]
            and coordinate_checks["all_predictions_in_input_bounds"]
            and float(coordinate_checks["max_target_normalized_roundtrip_error_px"])
            <= 1.0e-3
        )
        programmatic_pass = bool(
            bounded.status == "completed"
            and b4_gate_passed(eval_metrics)
            and visual_pass
            and structure["shape_contract_passed"]
            and structure["all_values_finite"]
            and gradients["all_required_nonzero"]
        )
        step_times = [float(row["step_time_sec"]) for row in bounded.history]
        measured_step_times = step_times[min(10, len(step_times)) :]
        if not measured_step_times:
            measured_step_times = step_times
        result: dict[str, Any] = {
            "schema_version": 1,
            "gate_id": "P1C_specialized_tiny_B2",
            "gate": "PENDING_REVIEW" if programmatic_pass else "FAIL",
            "programmatic_gate": "PASS" if programmatic_pass else "FAIL",
            "status": bounded.status,
            "steps_completed": bounded.steps_completed,
            "max_steps": protocol.resources.tiny_max_steps,
            "training_elapsed_seconds": bounded.elapsed_seconds,
            "training_allocated_seconds": training_allocation,
            "total_allocated_seconds": allocation,
            "initialization": initialization,
            "operator_probe_binding": _file_binding(
                operator_probe_artifact,
                logical_name="operator_probe.json",
            ),
            "operator_probe_gate": operator_probe["gate"],
            "structure_probe": structure,
            "gradient_evidence": gradients,
            "resource_measurements": {
                "trainable_parameters": count_trainable_parameters(model),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "mean_step_time_seconds_after_warmup": (
                    sum(measured_step_times) / len(measured_step_times)
                    if measured_step_times
                    else None
                ),
                "median_step_time_seconds_after_warmup": (
                    median(measured_step_times) if measured_step_times else None
                ),
            },
            "sample_selection": {
                "algorithm": "torch.randperm(train=300, seed=42)[:4]",
                "sample_count": TINY_SAMPLE_COUNT,
                "indices_omitted_from_gate_result": True,
            },
            "data_fingerprint_digest": fingerprint_digest(verified.fingerprints),
            "protocol_config_binding": _file_binding(
                phase1c_config,
                logical_name="configs/phase1c_specialized_enhancers.yaml",
            ),
            "environment": _runtime_environment(repository_root),
            "runtime_source_binding": _runtime_source_binding(repository_root),
            "eval_mode": eval_metrics,
            "visualization": {
                **visualization,
                "programmatic_check_passed": visual_pass,
                "manual_review_status": "pending",
            },
            "augmentation": "disabled",
            "batch_size": 1,
            "precision": "float32",
            "loss_application": "concatenate_[PS1,PS2,FH1]_then_existing_B2_once",
        }
        write_history_csv(output / "train_log.csv", bounded.history)
        write_json(
            output / "private_predictions.json",
            {
                "restricted_local_output": True,
                "selected_indices": list(indices),
                "eval_mode": private_rows,
            },
        )
        model.eval()
        save_checkpoint(
            output / "tiny.pt",
            model=model,
            optimizer=optimizer,
            epoch=0,
            config={
                "phase": "phase1c",
                "experiment_name": "P1C_specialized_tiny_B2",
                "testing_frozen": True,
                "training": config.to_dict(),
                "model": protocol.model.to_dict(),
            },
            seed=PHASE1C_SEED,
            metrics=eval_metrics,
            extra={"steps_completed": bounded.steps_completed, "status": bounded.status},
        )
        result["total_elapsed_seconds"] = time.perf_counter() - started
        result["within_total_allocation"] = result["total_elapsed_seconds"] <= allocation
        if not result["within_total_allocation"]:
            result["gate"] = result["programmatic_gate"] = "FAIL"
        ledger_status = "completed" if bounded.status == "completed" else "budget_exhausted"
        ledger_snapshot = _finish_ledger(
            ledger,
            "P1C_specialized_tiny_B2",
            started=started,
            status=ledger_status,
            details={
                "gate": result["gate"],
                "programmatic_gate": result["programmatic_gate"],
                "steps_completed": bounded.steps_completed,
            },
        )
        result["gpu_ledger_binding"] = _ledger_run_binding(
            ledger_snapshot,
            "P1C_specialized_tiny_B2",
        )
        if result["gpu_ledger_binding"]["entry"]["status"] != "completed":
            result["status"] = "budget_exhausted"
            result["gate"] = result["programmatic_gate"] = "FAIL"
            result["within_total_allocation"] = False
        write_json(output / "tiny_result.json", result)
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "gate_id": "P1C_specialized_tiny_B2",
            "gate": "FAIL",
            "status": "oom",
            "adaptation_attempted": False,
        }
        write_json(output / "tiny_result.json", result)
        _finish_ledger(
            ledger,
            "P1C_specialized_tiny_B2",
            started=started,
            status="oom",
        )
        return result
    except Exception:
        _finish_ledger(
            ledger,
            "P1C_specialized_tiny_B2",
            started=started,
            status="failed",
        )
        raise


def _overlay_bindings(tiny_result_path: Path) -> list[dict[str, Any]]:
    prediction_dir = tiny_result_path.parent / "predictions"
    actual_pngs = sorted(path.name for path in prediction_dir.glob("*.png"))
    if actual_pngs != sorted(TINY_OVERLAY_FILENAMES):
        raise PermissionError(
            "Phase 1C tiny review requires exactly the four preregistered overlays"
        )
    bindings: list[dict[str, Any]] = []
    for name in TINY_OVERLAY_FILENAMES:
        candidate = prediction_dir / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Phase 1C tiny overlay is missing: {name}")
        bindings.append(_file_binding(candidate, logical_name=f"predictions/{name}"))
    return bindings


def create_phase1c_tiny_review(
    *,
    tiny_artifact: str | Path,
    local_config: str | Path,
    phase1c_config: str | Path,
    decision: str,
    note: str,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Bind one explicit human review to the exact four private overlays."""

    if decision not in {"PASS", "FAIL"}:
        raise ValueError("Phase 1C tiny review decision must be PASS or FAIL")
    protocol = load_phase1c_config(phase1c_config)
    tiny_path = _require_private_phase1c_file(
        tiny_artifact,
        repository_root=repository_root,
    )
    tiny = json.loads(tiny_path.read_text(encoding="utf-8"))
    visualization = tiny.get("visualization")
    verified = load_verified_phase1b_data(local_config, repository_root=repository_root)
    if (
        tiny.get("gate_id") != "P1C_specialized_tiny_B2"
        or tiny.get("programmatic_gate") != "PASS"
        or tiny.get("gate") != "PENDING_REVIEW"
        or tiny.get("steps_completed") != protocol.resources.tiny_max_steps
        or tiny.get("data_fingerprint_digest") != fingerprint_digest(verified.fingerprints)
        or tiny.get("protocol_config_binding")
        != _file_binding(
            phase1c_config,
            logical_name="configs/phase1c_specialized_enhancers.yaml",
        )
        or tiny.get("runtime_source_binding") != _runtime_source_binding(repository_root)
        or tiny.get("environment") != _runtime_environment(repository_root)
        or tiny.get("loss_application")
        != "concatenate_[PS1,PS2,FH1]_then_existing_B2_once"
        or not isinstance(visualization, Mapping)
        or visualization.get("programmatic_check_passed") is not True
        or visualization.get("manual_review_status") != "pending"
        or visualization.get("visualization_count") != TINY_SAMPLE_COUNT
        or visualization.get("private_relative_directory") != "predictions"
    ):
        raise PermissionError("Phase 1C tiny result is stale or did not pass programmatically")
    output = require_phase1c_fresh_output(output_dir, repository_root=repository_root)
    review = {
        "schema_version": 1,
        "review_id": "P1C_specialized_tiny_manual_review",
        "decision": decision,
        "review_scope": "all_four_restricted_prediction_overlays",
        "review_note": note,
        "tiny_result_binding": _file_binding(tiny_path, logical_name="tiny_result.json"),
        "overlay_bindings": _overlay_bindings(tiny_path),
        "coordinate_or_channel_mismatch_observed": decision != "PASS",
    }
    write_json(output / "tiny_review.json", review)
    return review


def require_passed_phase1c_tiny(
    *,
    tiny_artifact: str | Path,
    review_artifact: str | Path,
    operator_probe_artifact: str | Path,
    local_config: str | Path,
    phase1c_config: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    tiny_path = _require_private_phase1c_file(
        tiny_artifact,
        repository_root=repository_root,
    )
    review_path = _require_private_phase1c_file(
        review_artifact,
        repository_root=repository_root,
    )
    tiny = json.loads(tiny_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    protocol = load_phase1c_config(phase1c_config)
    verified = load_verified_phase1b_data(local_config, repository_root=repository_root)
    metrics = tiny.get("eval_mode")
    visualization = tiny.get("visualization")
    if not isinstance(metrics, Mapping) or not b4_gate_passed(metrics):
        raise PermissionError("Phase 1C tiny metrics did not pass the fixed gate")
    required = bool(
        tiny.get("gate_id") == "P1C_specialized_tiny_B2"
        and tiny.get("gate") == "PENDING_REVIEW"
        and tiny.get("programmatic_gate") == "PASS"
        and tiny.get("status") == "completed"
        and tiny.get("steps_completed") == protocol.resources.tiny_max_steps
        and tiny.get("max_steps") == protocol.resources.tiny_max_steps
        and tiny.get("within_total_allocation") is True
        and tiny.get("augmentation") == "disabled"
        and tiny.get("batch_size") == 1
        and tiny.get("precision") == "float32"
        and tiny.get("loss_application")
        == "concatenate_[PS1,PS2,FH1]_then_existing_B2_once"
        and tiny.get("operator_probe_gate") == "PASS"
        and tiny.get("operator_probe_binding")
        == _file_binding(
            operator_probe_artifact,
            logical_name="operator_probe.json",
        )
        and isinstance(visualization, Mapping)
        and visualization.get("programmatic_check_passed") is True
        and visualization.get("manual_review_status") == "pending"
        and visualization.get("visualization_count") == TINY_SAMPLE_COUNT
        and visualization.get("private_relative_directory") == "predictions"
        and tiny.get("structure_probe", {}).get("shape_contract_passed") is True
        and tiny.get("structure_probe", {}).get("all_values_finite") is True
        and tiny.get("gradient_evidence", {}).get("all_required_nonzero") is True
        and tiny.get("data_fingerprint_digest") == fingerprint_digest(verified.fingerprints)
        and tiny.get("protocol_config_binding")
        == _file_binding(
            phase1c_config,
            logical_name="configs/phase1c_specialized_enhancers.yaml",
        )
        and tiny.get("runtime_source_binding") == _runtime_source_binding(repository_root)
        and tiny.get("environment") == _runtime_environment(repository_root)
    )
    if not required:
        raise PermissionError("Phase 1C tiny artifact is stale or incomplete")
    if (
        review.get("decision") != "PASS"
        or review.get("review_scope") != "all_four_restricted_prediction_overlays"
        or review.get("coordinate_or_channel_mismatch_observed") is not False
        or review.get("tiny_result_binding")
        != _file_binding(tiny_path, logical_name="tiny_result.json")
        or review.get("overlay_bindings") != _overlay_bindings(tiny_path)
    ):
        raise PermissionError("Phase 1C tiny human review is missing, stale, or failed")
    ledger_binding = tiny.get("gpu_ledger_binding")
    if not isinstance(ledger_binding, Mapping):
        raise PermissionError("Phase 1C tiny artifact has no ledger binding")
    _require_current_ledger_binding(
        ledger_binding,
        name="P1C_specialized_tiny_B2",
        ledger_path=ledger_path,
        repository_root=repository_root,
        config=protocol,
        required_details={
            "gate": "PENDING_REVIEW",
            "programmatic_gate": "PASS",
            "steps_completed": protocol.resources.tiny_max_steps,
        },
    )
    return {"tiny_result": tiny, "human_review": review}


def _metric_with_rates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    valid = int(metrics["n_valid_aop"])
    evaluable = int(metrics["n_evaluable_aop"])
    result["aop_valid_rate"] = valid / evaluable if evaluable else float("nan")
    result["selection_aop_penalized_deg"] = float(metrics["aop_mae_deg"])
    return result


def _fit_phase1c_supervised(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    train_evaluation_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    dsnt: DSNT,
    device: torch.device,
    config: Any,
    output_dir: Path,
    checkpoint_config: Mapping[str, Any],
    max_runtime_seconds: float,
    milestone_epochs: Sequence[int],
    train_generator: torch.Generator,
) -> dict[str, Any]:
    """Train H3 and recompute both train and validation metrics after every epoch."""

    write_json(output_dir / "config.json", checkpoint_config)
    history: list[dict[str, Any]] = []
    best_selection = (math.inf, math.inf, math.inf)
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    started = time.perf_counter()
    budget = WallClockBudget.start(max_runtime_seconds)
    milestones = set(int(epoch) for epoch in milestone_epochs)
    for epoch in range(1, config.epochs + 1):
        if history:
            recent = history[-3:]
            estimate = max(
                EPOCH_START_SAFETY_SECONDS,
                1.25 * max(float(row["epoch_time_sec"]) for row in recent),
            )
            if not budget.can_start(estimate):
                break
        epoch_started = time.perf_counter()
        optimization_started = time.perf_counter()
        optimization_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        optimization_seconds = time.perf_counter() - optimization_started
        train_eval_started = time.perf_counter()
        train_metrics = _metric_with_rates(
            evaluate_model(
                model,
                train_evaluation_loader,
                dsnt=dsnt,
                device=device,
                config=config,
            )
        )
        train_evaluation_seconds = time.perf_counter() - train_eval_started
        validation_started = time.perf_counter()
        validation_metrics = _metric_with_rates(
            evaluate_model(
                model,
                validation_loader,
                dsnt=dsnt,
                device=device,
                config=config,
            )
        )
        validation_seconds = time.perf_counter() - validation_started
        row: dict[str, Any] = {
            "epoch": epoch,
            "optimization_time_sec": optimization_seconds,
            "train_evaluation_time_sec": train_evaluation_seconds,
            "validation_time_sec": validation_seconds,
            "epoch_time_sec": time.perf_counter() - epoch_started,
        }
        row.update({f"optimization_{key}": value for key, value in optimization_metrics.items()})
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch in milestones:
            _save_model_only(
                output_dir / "milestones" / f"epoch_{epoch:03d}_model_only.pt",
                model=model,
                epoch=epoch,
            )
        resume_state = _capture_resume_state(train_generator)
        checkpoint_metrics = {
            name: value
            for name, value in validation_metrics.items()
            if name not in {"aop_valid_rate", "selection_aop_penalized_deg"}
        }
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=checkpoint_config,
            seed=config.seed,
            metrics=checkpoint_metrics,
            extra={
                "runtime_elapsed_sec": time.perf_counter() - started,
                "runtime_limit_sec": max_runtime_seconds,
                "resume_state": resume_state,
            },
        )
        selection = (
            float(validation_metrics["selection_aop_penalized_deg"]),
            float(validation_metrics["MRE_ALL"]),
            float(epoch),
        )
        if not all(math.isfinite(value) for value in selection[:2]):
            raise FloatingPointError("Phase 1C validation selection became NaN or Inf")
        if selection < best_selection:
            best_selection = selection
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=checkpoint_config,
                seed=config.seed,
                metrics=checkpoint_metrics,
                extra={
                    "runtime_elapsed_sec": time.perf_counter() - started,
                    "runtime_limit_sec": max_runtime_seconds,
                    "resume_state": resume_state,
                },
            )
        write_history_csv(output_dir / "train_log.csv", history)
    if not history:
        raise RuntimeError("Phase 1C formal budget did not permit one complete epoch")
    return {
        "status": "completed" if len(history) == config.epochs else "budget_exhausted",
        "epochs_completed": len(history),
        "epochs_requested": config.epochs,
        "runtime_elapsed_sec": time.perf_counter() - started,
        "runtime_limit_sec": max_runtime_seconds,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "last_validation_metrics": {
            name.removeprefix("val_"): value
            for name, value in history[-1].items()
            if name.startswith("val_")
        },
        "last_train_metrics": {
            name.removeprefix("train_"): value
            for name, value in history[-1].items()
            if name.startswith("train_")
        },
        "per_epoch_train_metrics_semantics": "post_epoch_eval_mode_full_train_split",
        "epoch_start_safety_seconds": EPOCH_START_SAFETY_SECONDS,
    }


def _initializations_match(
    tiny_initialization: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    keys = (
        "seed",
        "method",
        "h2_backbone_state_sha256",
        "h3_backbone_state_sha256",
        "h2_ps_decoder_state_sha256",
        "h3_ps_decoder_state_sha256",
        "h2_fh_decoder_state_sha256",
        "h3_fh_decoder_state_sha256",
        "h3_complete_state_sha256",
        "base_state_values_equal",
        "base_parameter_storage_aliased",
        "h2_trainable_parameters",
        "h3_trainable_parameters",
        "additional_trainable_parameters",
        "enhancer_initialization",
    )
    return all(tiny_initialization.get(name) == current.get(name) for name in keys)


def _formal_allocation_outcome(
    *,
    elapsed_seconds: float,
    allocated_seconds: float,
    ledger_binding: Mapping[str, Any],
) -> dict[str, bool]:
    entry = ledger_binding.get("entry")
    if not isinstance(entry, Mapping):
        raise ValueError("Phase 1C formal result has no bound ledger entry")
    details = entry.get("details")
    if not isinstance(details, Mapping):
        raise ValueError("Phase 1C formal ledger entry has no details")
    allocation_exceeded = bool(
        float(elapsed_seconds) > float(allocated_seconds)
        or float(entry["elapsed_seconds"]) > float(entry["allocated_seconds"])
        or details.get("allocation_exceeded") is True
    )
    aggregate_exceeded = details.get("aggregate_limit_exceeded") is True
    return {
        "formal_allocation_exceeded": allocation_exceeded,
        "aggregate_gpu_cap_exceeded": aggregate_exceeded,
        "within_runtime_allocation": not (allocation_exceeded or aggregate_exceeded),
    }


def run_phase1c_formal(
    *,
    local_config: str | Path,
    phase1c_config: str | Path,
    operator_probe_artifact: str | Path,
    tiny_artifact: str | Path,
    tiny_review_artifact: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Run the gated H3 seed-42, 16-epoch supervised control within 3 hours."""

    protocol = load_phase1c_config(phase1c_config)
    operator_probe = _require_passed_operator_probe(
        operator_probe_artifact,
        phase1c_config=phase1c_config,
        ledger_path=ledger_path,
        repository_root=repository_root,
    )
    tiny_evidence = require_passed_phase1c_tiny(
        tiny_artifact=tiny_artifact,
        review_artifact=tiny_review_artifact,
        operator_probe_artifact=operator_probe_artifact,
        local_config=local_config,
        phase1c_config=phase1c_config,
        ledger_path=ledger_path,
        repository_root=repository_root,
    )
    verified = load_verified_phase1b_data(
        local_config,
        repository_root=repository_root,
    )
    current_fingerprint = fingerprint_digest(verified.fingerprints)
    config = protocol.training
    train_dataset = _dataset(verified.specs["train"], config)
    validation_dataset = _dataset(verified.specs["validation"], config)
    if (
        len(train_dataset) != TRAIN_SAMPLE_COUNT
        or len(validation_dataset) != VALIDATION_SAMPLE_COUNT
    ):
        raise PermissionError("Phase 1C formal run requires the verified 300/100 split")
    train_generator = make_generator(PHASE1C_SEED)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
        pin_memory=True,
    )
    train_evaluation_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    validation_loader: DataLoader[dict[str, Any]] = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    h2, model, initialization = build_phase1c_initialization()
    tiny_initialization = tiny_evidence["tiny_result"].get("initialization")
    if not isinstance(tiny_initialization, Mapping) or not _initializations_match(
        tiny_initialization,
        initialization,
    ):
        raise PermissionError("Phase 1C tiny and formal runs do not share exact initialization")
    output = require_phase1c_fresh_output(output_dir, repository_root=repository_root)
    _save_model_only(output / "initial_h2_model_only.pt", model=h2, epoch=0)
    _save_model_only(output / "initial_h3_model_only.pt", model=model, epoch=0)
    del h2
    if not torch.cuda.is_available():
        result = {
            "schema_version": 1,
            "experiment_name": protocol.experiment_name,
            "status": "cuda_unavailable",
            "partial": True,
        }
        write_json(output / "formal_result.json", result)
        return result
    ledger = _phase1c_ledger(
        ledger_path,
        repository_root=repository_root,
        config=protocol,
    )
    allocation = ledger.begin(
        protocol.experiment_name,
        requested_limit_seconds=float(protocol.resources.formal_max_seconds),
        reserve_after_seconds=float(protocol.resources.closing_reserve_seconds),
    )
    post_evaluation_reserve = float(protocol.resources.post_evaluation_reserve_seconds)
    training_allocation = allocation - post_evaluation_reserve
    started = time.perf_counter()
    if training_allocation <= 0:
        ledger_snapshot = _finish_ledger(
            ledger,
            protocol.experiment_name,
            started=started,
            status="budget_exhausted",
            details={"epochs_completed": 0, "reason": "post_evaluation_reserve"},
        )
        result = {
            "schema_version": 1,
            "experiment_name": protocol.experiment_name,
            "status": "budget_exhausted",
            "epochs_completed": 0,
            "epochs_requested": config.epochs,
            "partial": True,
            "training_subbudget_exhausted": True,
            "runtime_elapsed_sec": time.perf_counter() - started,
            "runtime_allocated_sec": allocation,
            "training_guard_sec": max(0.0, training_allocation),
            "post_evaluation_reserve_sec": post_evaluation_reserve,
            "gpu_ledger_binding": _ledger_run_binding(
                ledger_snapshot,
                protocol.experiment_name,
            ),
            "within_runtime_allocation": True,
            "formal_allocation_exceeded": False,
            "aggregate_gpu_cap_exceeded": False,
        }
        write_json(output / "formal_result.json", result)
        return result
    try:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        model = model.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer = build_phase1c_adam(model.parameters(), protocol)
        dsnt = DSNT(
            temperature=config.dsnt_temperature,
            align_corners=config.align_corners,
        ).to(device)
        checkpoint_config = {
            "phase": "phase1c",
            "experiment_name": protocol.experiment_name,
            "testing_frozen": True,
            "training": config.to_dict(),
            "model": protocol.model.to_dict(),
            "optimizer": protocol.optimizer.to_dict(),
            "data": {
                "train_count": TRAIN_SAMPLE_COUNT,
                "validation_count": VALIDATION_SAMPLE_COUNT,
                "fingerprint_digest": current_fingerprint,
                "train_shuffle_generator_seed": PHASE1C_SEED,
                "paths_embedded": False,
            },
            "runtime": {
                "allocated_seconds": allocation,
                "training_allocated_seconds": training_allocation,
                "post_evaluation_reserve_seconds": post_evaluation_reserve,
                "total_phase_gpu_cap_seconds": protocol.resources.total_gpu_max_seconds,
            },
            "environment": _runtime_environment(repository_root),
            "runtime_source_binding": _runtime_source_binding(repository_root),
            "protocol_config_binding": _file_binding(
                phase1c_config,
                logical_name="configs/phase1c_specialized_enhancers.yaml",
            ),
            "common_initialization": initialization,
            "operator_probe_evidence": {
                "artifact_binding": _file_binding(
                    operator_probe_artifact,
                    logical_name="operator_probe.json",
                ),
                "gate": operator_probe["gate"],
            },
            "tiny_gate_evidence": {
                "tiny_result_binding": tiny_evidence["human_review"]["tiny_result_binding"],
                "overlay_bindings": tiny_evidence["human_review"]["overlay_bindings"],
                "human_review_decision": "PASS",
            },
            "per_epoch_train_metrics_semantics": "post_epoch_eval_mode_full_train_split",
        }
        summary = _fit_phase1c_supervised(
            model,
            train_loader,
            train_evaluation_loader,
            validation_loader,
            optimizer,
            dsnt=dsnt,
            device=device,
            config=config,
            output_dir=output,
            checkpoint_config=checkpoint_config,
            max_runtime_seconds=training_allocation,
            milestone_epochs=protocol.resources.milestone_epochs,
            train_generator=train_generator,
        )
        key_metrics = _evaluate_key_checkpoints(
            model,
            optimizer,
            output=output,
            train_loader=train_evaluation_loader,
            validation_loader=validation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
        )
        write_json(output / "key_checkpoint_metrics.json", key_metrics)
        torch.cuda.synchronize(device)
        total_elapsed = time.perf_counter() - started
        training_subbudget_exhausted = summary["status"] != "completed"
        result: dict[str, Any] = {
            "schema_version": 1,
            "experiment_name": protocol.experiment_name,
            "status": summary["status"],
            "epochs_completed": summary["epochs_completed"],
            "epochs_requested": config.epochs,
            "partial": training_subbudget_exhausted,
            "training_subbudget_exhausted": training_subbudget_exhausted,
            "selection_split": "validation",
            "selection_order": [
                "penalized_aop_mae_deg",
                "MRE_ALL",
                "earlier_epoch",
            ],
            "runtime_elapsed_sec": total_elapsed,
            "runtime_allocated_sec": allocation,
            "training_guard_sec": training_allocation,
            "post_evaluation_reserve_sec": post_evaluation_reserve,
            "epoch_start_safety_sec": summary["epoch_start_safety_seconds"],
            "best_epoch": summary["best_epoch"],
            "best_validation_metrics": summary["best_validation_metrics"],
            "last_validation_metrics": summary["last_validation_metrics"],
            "last_train_metrics": summary["last_train_metrics"],
            "per_epoch_train_metrics_semantics": summary[
                "per_epoch_train_metrics_semantics"
            ],
            "key_checkpoint_metrics": key_metrics,
            "initialization": initialization,
            "resource_measurements": {
                "trainable_parameters": count_trainable_parameters(model),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
        }
        ledger_status = "completed" if result["status"] == "completed" else "budget_exhausted"
        ledger_snapshot = _finish_ledger(
            ledger,
            protocol.experiment_name,
            started=started,
            status=ledger_status,
            details={"epochs_completed": summary["epochs_completed"]},
        )
        result["gpu_ledger_binding"] = _ledger_run_binding(
            ledger_snapshot,
            protocol.experiment_name,
        )
        result.update(
            _formal_allocation_outcome(
                elapsed_seconds=total_elapsed,
                allocated_seconds=allocation,
                ledger_binding=result["gpu_ledger_binding"],
            )
        )
        if result["gpu_ledger_binding"]["entry"]["status"] != "completed":
            result["status"] = "budget_exhausted"
            result["partial"] = True
        if not result["within_runtime_allocation"]:
            result["status"] = "budget_exhausted"
            result["partial"] = True
        write_json(output / "formal_result.json", result)
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {
            "schema_version": 1,
            "experiment_name": protocol.experiment_name,
            "status": "oom",
            "partial": True,
            "adaptation_attempted": False,
        }
        write_json(output / "formal_result.json", result)
        _finish_ledger(
            ledger,
            protocol.experiment_name,
            started=started,
            status="oom",
        )
        return result
    except Exception:
        _finish_ledger(
            ledger,
            protocol.experiment_name,
            started=started,
            status="failed",
        )
        raise
