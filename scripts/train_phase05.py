#!/usr/bin/env python
"""Run one locked Phase 0.5 supervised ablation on train/validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.optim import Adam  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase05_ablation.yaml"
PHASE05_RUN_ROOT = REPOSITORY_ROOT / "runs" / "phase05"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.data.access_policy import (  # noqa: E402
    fingerprint_labeled_split,
    load_phase05_local_splits,
    verify_fingerprint,
)
from geoequi_ld.data.dataset import IUGCLabeledDataset  # noqa: E402
from geoequi_ld.models.dsnt import DSNT  # noqa: E402
from geoequi_ld.models.unet import HeatmapUNet, count_trainable_parameters  # noqa: E402
from geoequi_ld.training.ablation import (  # noqa: E402
    VARIANTS,
    apply_variant,
    assert_variant_weights,
    normalize_variant,
)
from geoequi_ld.training.checkpoints import restore_checkpoint  # noqa: E402
from geoequi_ld.training.config import SupervisedTrainingConfig  # noqa: E402
from geoequi_ld.training.engine import evaluate_model, fit_supervised, write_json  # noqa: E402
from geoequi_ld.training.runtime import (  # noqa: E402
    make_generator,
    resolve_device,
    seed_data_loader_worker,
    seed_everything,
)
from geoequi_ld.utils.hashing import sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pre-registered Phase 0.5 loss ablation. "
            "This command accepts train and validation only; testing is forbidden."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _load_protocol(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Phase 0.5 protocol must be a mapping")
    project = loaded.get("project")
    selection = loaded.get("selection")
    data_contract = loaded.get("data_contract")
    if not isinstance(project, dict) or not isinstance(selection, dict):
        raise ValueError("Protocol requires project and selection mappings")
    if project.get("phase") != "phase0.5-supervised-ablation":
        raise ValueError("Protocol phase must be phase0.5-supervised-ablation")
    if project.get("testing_frozen") is not True:
        raise PermissionError("Phase 0.5 requires testing_frozen=true")
    if selection.get("split") != "validation" or selection.get("common_decoder") != "dsnt":
        raise PermissionError("Phase 0.5 selection must use validation with common DSNT decoding")
    if selection.get("checkpoint_selection") != [
        "aop_mae_deg",
        "MRE_ALL",
        "earlier_epoch",
    ]:
        raise ValueError("Unexpected checkpoint-selection rule")
    if selection.get("variant_retention") != [
        "aop_mae_deg",
        "MRE_ALL",
        "simpler_objective",
    ]:
        raise ValueError("Unexpected variant-retention rule")
    if selection.get("first_round_seed") != 42 or selection.get("confirmation_seeds") != [
        43,
        44,
        45,
    ]:
        raise ValueError("Unexpected Phase 0.5 seed protocol")
    if not isinstance(data_contract, dict):
        raise ValueError("Protocol requires a data_contract mapping")
    if data_contract.get("allowed_splits") != ["train", "validation"]:
        raise PermissionError("Phase 0.5 permits only train and validation")
    if set(data_contract.get("forbidden_splits", [])) != {"test", "testing"}:
        raise PermissionError("Protocol must explicitly freeze test/testing")
    return loaded


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _validate_repository_state(protocol_path: Path, protocol: Mapping[str, Any]) -> str:
    if protocol_path.resolve(strict=True) != CANONICAL_PROTOCOL.resolve(strict=True):
        raise PermissionError("Phase 0.5 accepts only configs/phase05_ablation.yaml")
    _git_output("ls-files", "--error-unmatch", "configs/phase05_ablation.yaml")
    commit, dirty = _git_state()
    if dirty:
        raise RuntimeError("Formal Phase 0.5 runs require a clean Git worktree")
    project = protocol["project"]
    parent_commit = str(project["parent_phase0_commit"])
    parent_tag = str(project["parent_phase0_tag"])
    if _git_output("rev-parse", f"{parent_tag}^{{}}") != parent_commit:
        raise RuntimeError("The frozen Phase 0 tag does not match the protocol")
    return commit


def _validate_seed_and_output(
    *,
    protocol: Mapping[str, Any],
    variant: str,
    seed: int,
    output_dir: Path,
    commit: str,
) -> Path:
    selection = protocol["selection"]
    first_seed = int(selection["first_round_seed"])
    confirmation_seeds = {int(value) for value in selection["confirmation_seeds"]}
    allowed = {first_seed, *confirmation_seeds}
    if seed not in allowed:
        raise ValueError(f"Seed {seed} is not pre-registered; allowed seeds are {sorted(allowed)}")
    if seed != first_seed:
        selection_path = PHASE05_RUN_ROOT / "selection.json"
        if not selection_path.is_file():
            raise PermissionError("Confirmation runs require a frozen seed-42 selection manifest")
        frozen_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if frozen_selection.get("testing_frozen") is not True:
            raise PermissionError("Selection manifest does not freeze testing")
        if frozen_selection.get("git_commit") != commit:
            raise RuntimeError("Selection manifest was produced by a different code commit")
        if variant not in frozen_selection.get("selected_variants", []):
            raise PermissionError(f"Variant {variant} was not retained after seed 42")
    expected_output = (PHASE05_RUN_ROOT / variant / f"seed_{seed}").resolve(strict=False)
    if output_dir.resolve(strict=False) != expected_output:
        raise PermissionError(f"Output must be runs/phase05/{variant}/seed_{seed}")
    return expected_output


def _loader(
    dataset: IUGCLabeledDataset,
    *,
    config: SupervisedTrainingConfig,
    shuffle: bool,
    seed_offset: int,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_loader_worker if config.num_workers else None,
        generator=make_generator(config.seed + seed_offset),
        persistent_workers=config.num_workers > 0,
    )


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return commit, dirty


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        payload["gpu"] = torch.cuda.get_device_name(device)
    return payload


def _save_curves(run_dir: Path) -> None:
    with (run_dir / "train_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    curves_dir = run_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4), dpi=150)
    for key in ("train_total_loss", "val_total_loss"):
        axis.plot(epochs, [float(row[key]) for row in rows], label=key)
    axis.set(xlabel="epoch", ylabel="loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(curves_dir / "losses.png")
    plt.close(figure)

    figure, left = plt.subplots(figsize=(7, 4), dpi=150)
    right = left.twinx()
    left.plot(epochs, [float(row["val_MRE_ALL"]) for row in rows], color="#0072B2")
    right.plot(epochs, [float(row["val_aop_mae_deg"]) for row in rows], color="#D55E00")
    left.set(xlabel="epoch", ylabel="MRE (px)")
    right.set_ylabel("AoP MAE (deg)")
    left.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(curves_dir / "validation_metrics.png")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variant_name = normalize_variant(args.variant)
    if args.protocol.resolve(strict=True) != CANONICAL_PROTOCOL.resolve(strict=True):
        raise PermissionError("Phase 0.5 accepts only the canonical repository protocol")
    protocol = _load_protocol(args.protocol)
    commit = _validate_repository_state(args.protocol, protocol)
    output_dir = _validate_seed_and_output(
        protocol=protocol,
        variant=variant_name,
        seed=args.seed,
        output_dir=args.output_dir,
        commit=commit,
    )
    local_splits = load_phase05_local_splits(args.local_config)
    data_contract = protocol["data_contract"]
    fingerprints: dict[str, dict[str, str | int]] = {}
    for role in ("train", "validation"):
        actual = fingerprint_labeled_split(local_splits[role])
        verify_fingerprint(actual, local_splits[role].expected_fingerprint, role=role)
        if data_contract[role].get("fingerprint_required") is not True:
            raise PermissionError(f"Public protocol must require a local fingerprint for {role}")
        if actual["sample_count"] != int(data_contract[role]["sample_count"]):
            raise PermissionError(f"Public sample-count contract failed for {role}")
        expected_columns = data_contract[role].get("source_columns")
        actual_columns = {
            "PS1": "PS1",
            "PS2": "PS2",
            "FH1": local_splits[role].fh1_column,
        }
        if actual_columns != expected_columns:
            raise PermissionError(f"Source-column contract failed for {role}")
        fingerprints[role] = actual

    base_config = SupervisedTrainingConfig.from_mapping(protocol["training"])
    config = apply_variant(replace(base_config, seed=args.seed), variant_name)
    assert_variant_weights(config, variant_name)
    declared_weights = tuple(
        float(value) for value in protocol["variants"][variant_name]["weights"]
    )
    actual_weights = (
        config.heatmap_loss_weight,
        config.coordinate_loss_weight,
        config.distribution_loss_weight,
    )
    if declared_weights != actual_weights:
        raise ValueError("Protocol variant weights disagree with the implementation lock")

    verified = {
        "status": "validated",
        "phase": "phase0.5-supervised-ablation",
        "variant": variant_name,
        "seed": config.seed,
        "selection_split": "validation",
        "selection_decoder": "dsnt",
        "testing_frozen": True,
        "data": {
            role: {"sample_count": value["sample_count"], "fingerprint_verified": True}
            for role, value in fingerprints.items()
        },
        "git_commit": commit,
        "git_dirty": False,
    }
    if args.verify_only:
        print(json.dumps(verified, ensure_ascii=False, indent=2))
        return 0
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    common_dataset = {
        "keypoint_order": config.keypoint_order,
        "input_size_hw": config.input_size_hw,
        "heatmap_size_hw": config.heatmap_size_hw,
        "sigma": config.sigma_heatmap_px,
        "align_corners": config.align_corners,
    }
    train_spec = local_splits["train"]
    validation_spec = local_splits["validation"]
    train_dataset = IUGCLabeledDataset(
        image_dir=train_spec.image_dir,
        labels_csv=train_spec.labels_csv,
        source_columns={"PS1": "PS1", "PS2": "PS2", "FH1": train_spec.fh1_column},
        **common_dataset,
    )
    validation_dataset = IUGCLabeledDataset(
        image_dir=validation_spec.image_dir,
        labels_csv=validation_spec.labels_csv,
        source_columns={
            "PS1": "PS1",
            "PS2": "PS2",
            "FH1": validation_spec.fh1_column,
        },
        **common_dataset,
    )
    train_loader = _loader(train_dataset, config=config, shuffle=True, seed_offset=0, device=device)
    validation_loader = _loader(
        validation_dataset,
        config=config,
        shuffle=False,
        seed_offset=1,
        device=device,
    )

    model = HeatmapUNet(base_channels=config.base_channels).to(device)
    initialization_sha256 = _state_dict_sha256(model)
    dsnt = DSNT(
        temperature=config.dsnt_temperature,
        align_corners=config.align_corners,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run_config = {
        "schema_version": 1,
        "phase": "phase0.5-supervised-ablation",
        "variant": variant_name,
        "variant_description": VARIANTS[variant_name].description,
        "training": config.to_dict(),
        "selection": {
            "split": "validation",
            "common_decoder": "dsnt",
            "checkpoint_selection": protocol["selection"]["checkpoint_selection"],
        },
        "testing_frozen": True,
        "data": {
            role: {
                **fingerprint,
                "source_columns": data_contract[role]["source_columns"],
            }
            for role, fingerprint in fingerprints.items()
        },
        "model": {
            "class": "HeatmapUNet",
            "trainable_parameters": count_trainable_parameters(model),
            "initialization_sha256": initialization_sha256,
        },
        "provenance": {
            "protocol_sha256": sha256_file(args.protocol),
            "git_commit": commit,
            "git_dirty": False,
        },
    }
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(output_dir / "environment.json", _environment(device))

    started = time.perf_counter()
    summary = fit_supervised(
        model,
        train_loader,
        validation_loader,
        optimizer,
        dsnt=dsnt,
        device=device,
        config=config,
        output_dir=output_dir,
        checkpoint_config=run_config,
    )
    training_runtime_sec = time.perf_counter() - started
    peak_gpu_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )
    peak_gpu_reserved_mb = (
        torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
    )
    evaluation_started = time.perf_counter()
    restore_checkpoint(summary["best_checkpoint"], model=model, map_location=device)
    decoder_metrics: dict[str, Any] = {
        "dsnt": evaluate_model(
            model,
            validation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            decoder="dsnt",
        )
    }
    if variant_name == "B0":
        decoder_metrics["argmax"] = evaluate_model(
            model,
            validation_loader,
            dsnt=dsnt,
            device=device,
            config=config,
            decoder="argmax",
        )
    evaluation_runtime_sec = time.perf_counter() - evaluation_started
    _save_curves(output_dir)
    best_checkpoint_sha256 = sha256_file(Path(summary["best_checkpoint"]))
    resources = {
        "training_runtime_sec": training_runtime_sec,
        "evaluation_runtime_sec": evaluation_runtime_sec,
        "peak_gpu_allocated_mb": peak_gpu_allocated_mb,
        "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
    }
    result = {
        "status": "completed",
        "phase": "phase0.5-supervised-ablation",
        "variant": variant_name,
        "seed": config.seed,
        "selection_split": "validation",
        "selection_decoder": "dsnt",
        "testing_frozen": True,
        "best_epoch": summary["best_epoch"],
        "best_validation_metrics": decoder_metrics,
        "last_validation_metrics": summary["last_validation_metrics"],
        "resources": resources,
        "model": run_config["model"],
        "provenance": {
            **run_config["provenance"],
            "best_checkpoint_sha256": best_checkpoint_sha256,
        },
    }
    write_json(output_dir / "phase05_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
