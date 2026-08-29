#!/usr/bin/env python
"""Characterize all six saved Phase 0.6 best/last checkpoint endpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "phase1a_b0_diagnostics.yaml"
CANONICAL_LOCAL_CONFIG = REPOSITORY_ROOT / "configs" / "phase05_local.yaml"
CANONICAL_SANITY_REPORT = REPOSITORY_ROOT / "reports" / "phase1a" / "HEATMAP_DECODE_SANITY.json"
CANONICAL_MEAN_REPORT = REPOSITORY_ROOT / "reports" / "phase1a" / "TRAIN_MEAN_BASELINE.json"
CANONICAL_A4_RESULT = REPOSITORY_ROOT / "runs" / "phase1a" / "A4_unet_B0" / "tiny_gate_result.json"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geoequi_ld.diagnostics.phase1a import (  # noqa: E402
    assert_public_aggregate,
    diagnose_model_on_validation,
    fixed_visualization_indices,
    load_checkpoint_specs,
    load_phase06_model,
    load_phase1a_protocol,
    load_verified_splits,
    make_labeled_dataset,
    require_canonical_path,
    require_private_output_path,
    require_public_output_path,
)
from geoequi_ld.models.dsnt import DSNT  # noqa: E402
from geoequi_ld.training.engine import write_json  # noqa: E402
from geoequi_ld.training.runtime import resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate B0/B1/B2 best/last heatmap and decoder diagnostics"
    )
    parser.add_argument("--protocol", type=Path, default=CANONICAL_PROTOCOL)
    parser.add_argument("--local-config", type=Path, default=CANONICAL_LOCAL_CONFIG)
    parser.add_argument("--device", default=None, help="Defaults to the protocol device (CPU)")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "phase1a" / "B0_CHECKPOINT_DIAGNOSTICS.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "phase1a" / "B0_DIAGNOSTICS.md",
    )
    parser.add_argument(
        "--private-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "phase1a" / "checkpoint_diagnostics",
    )
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Rebuild only the Markdown report from the fixed existing aggregate artifacts.",
    )
    return parser


def _loader(dataset: Any, *, batch_size: int, num_workers: int) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _private_sample_payload(sample: dict[str, Any], *, visualization: Path) -> dict[str, Any]:
    return {
        "split": sample["split"],
        "dataset_index": sample["dataset_index"],
        "filename": sample["filename"],
        "target_points_original_px": sample["target_points_original_px"].tolist(),
        "target_points_heatmap_px": sample["target_points_heatmap_px"].tolist(),
        "predicted_points_original_px": {
            method: value.tolist()
            for method, value in sample["predicted_points_original_px"].items()
        },
        "ray_diagnostics": sample["ray_diagnostics"],
        "visualization": str(visualization),
    }


def _save_visualization(sample: dict[str, Any], destination: Path) -> None:
    image = sample["image"].squeeze(0).numpy()
    predicted_heatmaps = sample["predicted_heatmaps"].numpy()
    target = sample["target_points_original_px"].numpy()
    predicted = {
        method: value.numpy() for method, value in sample["predicted_points_original_px"].items()
    }
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=130)
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].scatter(target[:, 0], target[:, 1], marker="x", c="lime", label="target")
    axes[0, 0].scatter(
        predicted["dsnt"][:, 0], predicted["dsnt"][:, 1], marker="o", c="red", label="DSNT"
    )
    axes[0, 0].scatter(
        predicted["argmax"][:, 0],
        predicted["argmax"][:, 1],
        marker="+",
        c="cyan",
        label="argmax",
    )
    axes[0, 0].set_title(f"{sample['split']} local diagnostic")
    axes[0, 0].legend(fontsize=7)
    for index, name in enumerate(("PS1", "PS2", "FH1")):
        axis = axes.flat[index + 1]
        axis.imshow(predicted_heatmaps[index], cmap="magma")
        axis.set_title(f"raw predicted {name}")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)


def _format_metric(value: object, suffix: str = "") -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "undefined"
    return f"{float(value):.4f}{suffix}"


def _read_json_mapping(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path.name}")
    return loaded


def _markdown(
    public: dict[str, Any],
    *,
    sanity: dict[str, Any],
    mean_baseline: dict[str, Any],
    a4_result: dict[str, Any],
) -> str:
    rows = []
    for checkpoint in public["checkpoints"]:
        dsnt = checkpoint["decoder_metrics"]["dsnt"]
        argmax = checkpoint["decoder_metrics"]["argmax"]
        heatmap = checkpoint["heatmap_diagnostics"]["overall"]
        rows.append(
            "| {id} | {epoch} | {ratio:.4f} | {dsnt_mre} | {dsnt_valid}/{n} | {argmax_mre} | "
            "{argmax_valid}/{n} |".format(
                id=checkpoint["checkpoint_id"],
                epoch=checkpoint["epoch"],
                ratio=heatmap["zero_map_mse_ratio"],
                dsnt_mre=_format_metric(dsnt["MRE_ALL"]),
                dsnt_valid=dsnt["n_valid_aop"],
                argmax_mre=_format_metric(argmax["MRE_ALL"]),
                argmax_valid=argmax["n_valid_aop"],
                n=dsnt["n_evaluable_aop"],
            )
        )
    checkpoints = {
        checkpoint["checkpoint_id"]: checkpoint for checkpoint in public["checkpoints"]
    }
    b0_best = checkpoints["B0_best"]
    b0_last = checkpoints["B0_last"]
    best_heatmap = b0_best["heatmap_diagnostics"]["overall"]
    last_heatmap = b0_last["heatmap_diagnostics"]["overall"]
    best_dsnt = b0_best["decoder_metrics"]["dsnt"]
    best_argmax = b0_best["decoder_metrics"]["argmax"]
    best_dsnt_rays = b0_best["ray_diagnostics"]["dsnt"]
    best_argmax_rays = b0_best["ray_diagnostics"]["argmax"]
    last_dsnt_rays = b0_last["ray_diagnostics"]["dsnt"]
    cases = {case["case_id"]: case for case in sanity["cases"]}
    mean_metrics = mean_baseline["validation_metrics"]
    a4_metrics = a4_result["eval_mode"]
    invalid_reasons = last_dsnt_rays["official_invalid_reason_counts"]
    invalid_reason_text = "、".join(
        f"{reason}={count}" for reason, count in invalid_reasons.items()
    )
    return "\n".join(
        [
            "# Phase 1A：B0 异常诊断",
            "",
            "这份诊断只使用 train 与完整的 100 张 validation。固定的小样本子集只用于"
            "本地看图，真实图像、逐样本坐标和 checkpoint 路径都没有进入公开报告。",
            "",
            "## 实际保存了什么",
            "",
            "Phase 0.6 实际留下的是 B0/B1/B2 各自的 best 和 last，共 6 个 checkpoint。"
            "best 仍按 validation 上的 `(AoP MAE, MRE_ALL, 较早 epoch)` 用 DSNT 选择；"
            "表里的 argmax 是对同一个已保存 checkpoint 重新解码，不是另选了一批模型。",
            "",
            "| checkpoint | epoch | MSE / zero-map MSE | DSNT MRE | DSNT AoP valid | "
            "argmax MRE | argmax AoP valid |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "没有保存 B0 从第 186 轮到第 187 轮附近的转折 checkpoint，因此现在能比较"
            "的是端点，不能倒推出崩溃究竟从哪一个 batch 或哪一种机制开始。",
            "",
            "## B0 best 与 last",
            "",
            f"B0_best 的 raw heatmap 还有微弱空间变化：平均空间标准差 "
            f"{best_heatmap['raw_std']['mean']:.6f}，峰值并列像素数均值 "
            f"{best_heatmap['raw_peak_tie_count']['mean']:.2f}。但 softmax 后的归一化熵已达 "
            f"{best_heatmap['probability_entropy_normalized']['mean']:.6f}，平均概率峰值只有 "
            f"{best_heatmap['probability_peak']['mean']:.6e}，真值邻域概率质量均值为 "
            f"{best_heatmap['probability_mass_near_truth']['mean']:.6f}。也就是说它不是严格"
            "常数图，但 DSNT 所见的概率仍然非常分散。",
            "",
            f"B0_last 则已经是严格的空间常数：raw 空间标准差 "
            f"{last_heatmap['raw_std']['mean']:.1f}、peak gap "
            f"{last_heatmap['raw_peak_gap']['mean']:.1f}，每个通道有 "
            f"{last_heatmap['raw_peak_tie_count']['mean']:.0f} 个并列峰值（即 256×256 全图）。"
            f"softmax 概率峰值 {last_heatmap['probability_peak']['mean']:.6e} 正好等于均匀"
            "分布的 1/65536，三点都被解到图像中心。",
            "",
            f"射线长度也符合这一点。B0_best 的 DSNT 耻骨射线/胎头射线均值分别为 "
            f"{best_dsnt_rays['pubic_ray_length_original_px']['mean']:.3f} px 和 "
            f"{best_dsnt_rays['fetal_ray_length_original_px']['mean']:.3f} px；argmax 对应为 "
            f"{best_argmax_rays['pubic_ray_length_original_px']['mean']:.3f} px 和 "
            f"{best_argmax_rays['fetal_ray_length_original_px']['mean']:.3f} px。到 B0_last，"
            f"两条 DSNT 射线都为 0，官方无效原因是 `{invalid_reason_text}`，所以 100/100 "
            "AoP 都无效；惩罚口径记为 180°。",
            "",
            "## DSNT 与 argmax 的边界",
            "",
            f"在 B0_best 上，argmax 的 MRE_ALL 为 {best_argmax['MRE_ALL']:.3f} px，低于 "
            f"DSNT 的 {best_dsnt['MRE_ALL']:.3f} px；但 argmax 的 AoP MAE 为 "
            f"{best_argmax['aop_mae_deg']:.3f}°，反而高于 DSNT 的 "
            f"{best_dsnt['aop_mae_deg']:.3f}°。这说明两个解码器看到的是热图的不同部分，"
            "不能只凭一个指标把 argmax 写成更好的模型。B0_last 的图完全平坦，两个解码"
            "都产生零长度射线，换解码器也救不回来。",
            "",
            "## 合成解码检查",
            "",
            f"标准高斯用 argmax 的 MRE 是 {cases['gaussian_argmax']['MRE_ALL']:.3f} px；"
            f"温度 0.05 的 DSNT 为 {cases['gaussian_dsnt_t0.05']['MRE_ALL']:.4f} px。"
            f"同一张高斯在温度 1 时变成 {cases['gaussian_dsnt_t1']['MRE_ALL']:.3f} px，"
            f"把振幅缩到 0.1 后，即使峰位置不变，温度 0.05 的 DSNT 也达到 "
            f"{cases['gaussian_amplitude_0.1_dsnt_t0.05']['MRE_ALL']:.3f} px。零热图和三张"
            "平坦热图都会把三个期望坐标解到同一点，AoP 按官方规则无效。这里分别检查"
            "的是 raw heatmap `H`、softmax 概率 `P` 和 `P` 下的坐标期望，不把三者混成"
            "一个量。",
            "",
            "## 不看图的均值参考与四样本诊断",
            "",
            f"只用 300 个 train 标签拟合三点均值，再在 validation 上直接输出常数坐标，"
            f"MRE_ALL 为 {mean_metrics['MRE_ALL']:.3f} px、AoP MAE 为 "
            f"{mean_metrics['aop_mae_deg']:.3f}°，100/100 AoP 有效。它不是图像模型，只是"
            "一个下限参照；B0_best 的 DSNT MRE 没有超过这个不看图的坐标均值。",
            "",
            f"A4 另外把轻量 U-Net 的纯 MSE 在固定 4 张 train 图上跑满 1000 步。程序"
            f"完整执行、数值有限，但 PS1/PS2/FH1 误差分别为 "
            f"{a4_metrics['MRE_PS1']:.3f}/{a4_metrics['MRE_PS2']:.3f}/"
            f"{a4_metrics['MRE_FH1']:.3f} px，整体 MRE {a4_metrics['MRE_ALL']:.3f} px；"
            "它只学到 PS1，没有通过三点学习判据。",
            "",
            "## 可以确定与仍是推测的部分",
            "",
            "**可以确定：** B0_best 的概率响应高度分散；B0_last 是严格常数图；last 的"
            "三点重合直接造成两条零长度射线和全部无效 AoP；合成检查证明当前 DSNT/坐标"
            "接口在足够尖锐的高斯上能正确工作，也证明低振幅或平坦响应会把期望拉向中心。",
            "",
            "**仍是推测：** 纯 MSE 的前景/背景不平衡很可能允许模型用接近零的背景取得"
            "较小平均损失，进而促成响应变平；但现有端点不能证明它是唯一原因，也不能"
            "定位训练转折。要回答因果问题，需要转折区间 checkpoint 或专门的受控实验，"
            "本报告不拿现有 6 个端点替代这部分证据。",
            "",
        ]
    )


def _markdown_context() -> dict[str, dict[str, Any]]:
    sanity = _read_json_mapping(CANONICAL_SANITY_REPORT)
    mean_baseline = _read_json_mapping(CANONICAL_MEAN_REPORT)
    a4_result = _read_json_mapping(CANONICAL_A4_RESULT)
    if sanity.get("status") != "synthetic_only":
        raise ValueError("Synthetic sanity artifact has an unexpected status")
    if mean_baseline.get("status") != "completed":
        raise ValueError("Train-mean baseline artifact has an unexpected status")
    if a4_result.get("status") != "completed" or a4_result.get("steps_completed") != 1000:
        raise ValueError("A4 diagnostic artifact is incomplete")
    return {
        "sanity": sanity,
        "mean_baseline": mean_baseline,
        "a4_result": a4_result,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_output = require_public_output_path(args.json_output, repository_root=REPOSITORY_ROOT)
    markdown_output = require_public_output_path(
        args.markdown_output, repository_root=REPOSITORY_ROOT
    )
    if args.render_existing:
        public = _read_json_mapping(json_output)
        assert_public_aggregate(public)
        markdown_output.write_text(
            _markdown(public, **_markdown_context()),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "rendered_existing",
                    "checkpoint_count": public.get("checkpoint_count"),
                    "public_report": str(markdown_output.relative_to(REPOSITORY_ROOT)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    protocol_path = require_canonical_path(
        args.protocol, CANONICAL_PROTOCOL, context="Phase 1A protocol"
    )
    local_config = require_canonical_path(
        args.local_config, CANONICAL_LOCAL_CONFIG, context="Phase 1A local split config"
    )
    private_root = require_private_output_path(
        args.private_root, repository_root=REPOSITORY_ROOT
    )
    protocol = load_phase1a_protocol(protocol_path)
    diagnostic = protocol["diagnostic"]
    device = resolve_device(args.device or str(diagnostic["device"]))
    verified = load_verified_splits(local_config, protocol)
    train_dataset = make_labeled_dataset(verified.specs["train"], protocol)
    validation_dataset = make_labeled_dataset(verified.specs["validation"], protocol)
    checkpoint_specs = load_checkpoint_specs(protocol, repository_root=REPOSITORY_ROOT)
    train_indices = fixed_visualization_indices(
        len(train_dataset),
        role="train",
        count=int(diagnostic["visualization_train_count"]),
        seed=int(diagnostic["visualization_seed"]),
    )
    validation_indices = fixed_visualization_indices(
        len(validation_dataset),
        role="validation",
        count=int(diagnostic["visualization_validation_count"]),
        seed=int(diagnostic["visualization_seed"]),
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=int(diagnostic["batch_size"]),
        num_workers=int(diagnostic["num_workers"]),
    )
    train_visual_loader = _loader(
        Subset(train_dataset, list(train_indices)),
        batch_size=int(diagnostic["batch_size"]),
        num_workers=int(diagnostic["num_workers"]),
    )
    dsnt = DSNT(
        temperature=float(diagnostic["dsnt_temperature"]),
        align_corners=bool(diagnostic["align_corners"]),
    ).to(device)

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for spec in checkpoint_specs:
        model, payload, checkpoint_hash = load_phase06_model(
            spec, device=device, protocol=protocol
        )
        aggregate, validation_samples = diagnose_model_on_validation(
            model,
            validation_loader,
            dsnt=dsnt,
            device=device,
            protocol=protocol,
            visualization_indices=validation_indices,
        )
        _, train_samples = diagnose_model_on_validation(
            model,
            train_visual_loader,
            dsnt=dsnt,
            device=device,
            protocol=protocol,
            visualization_indices=tuple(range(len(train_indices))),
        )
        for local_index, sample in enumerate(train_samples):
            sample["split"] = "train"
            sample["dataset_index"] = train_indices[local_index]
        for sample in validation_samples:
            sample["split"] = "validation"
        sample_payloads = []
        for sample in [*train_samples, *validation_samples]:
            destination = (
                private_root
                / spec.checkpoint_id
                / "visualizations"
                / f"{sample['split']}_{int(sample['dataset_index']):03d}.png"
            )
            _save_visualization(sample, destination)
            sample_payloads.append(_private_sample_payload(sample, visualization=destination))
        public_rows.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "variant": spec.variant,
                "endpoint": spec.endpoint,
                "epoch": spec.epoch,
                **aggregate,
            }
        )
        private_rows.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "checkpoint_path": str(spec.path),
                "checkpoint_sha256": checkpoint_hash,
                "epoch": int(payload["epoch"]),
                "seed": int(payload["seed"]),
                "recorded_metrics": dict(payload["metrics"]),
                "visualization_samples": sample_payloads,
            }
        )
        del model

    public: dict[str, Any] = {
        "phase": "phase1a-b0-diagnostics",
        "status": "completed",
        "scope": "six_saved_endpoints_complete_validation",
        "best_endpoint_selection_decoder": "dsnt",
        "argmax_checkpoint_selection": "none_same_saved_endpoints_redecoded",
        "data": {
            "train_count": len(train_dataset),
            "validation_count": len(validation_dataset),
            "fingerprints_verified": True,
        },
        "checkpoint_count": len(public_rows),
        "checkpoints": public_rows,
        "causal_conclusion": "not_established_missing_transition_checkpoints",
    }
    assert_public_aggregate(public)
    private = {
        "phase": "phase1a-b0-diagnostics",
        "local_config": str(local_config),
        "device": str(device),
        "fingerprints": {role: dict(value) for role, value in verified.fingerprints.items()},
        "fixed_train_indices": list(train_indices),
        "fixed_validation_indices": list(validation_indices),
        "checkpoints": private_rows,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    write_json(json_output, public)
    write_json(private_root / "details.json", private)
    markdown_output.write_text(
        _markdown(public, **_markdown_context()),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_count": len(public_rows),
                "validation_count": len(validation_dataset),
                "public_report": str(markdown_output.relative_to(REPOSITORY_ROOT)),
                "private_root": str(private_root.relative_to(REPOSITORY_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
