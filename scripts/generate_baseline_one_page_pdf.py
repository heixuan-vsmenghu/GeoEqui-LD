#!/usr/bin/env python
"""Generate the single-page advisor PDF for the official IUGC 2025 T10 baseline."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "runs"
    / "baseline_reproduction"
    / "BASELINE_REPRODUCTION_ONE_PAGE.pdf"
)
OFFICIAL_REPOSITORY = "https://github.com/0oTyTo0/IUGC2025"
OFFICIAL_COMMIT = "bc8fce2032c000c2569e916268ab918c0905ab4e"
METRIC_KEYS = (
    "MRE_PS1",
    "MRE_PS2",
    "MRE_FH1",
    "MRE_ALL",
    "AoP_absolute_error_deg",
)
OFFICIAL_METRICS = {
    "validation": {
        "MRE_PS1": 12.3408,
        "MRE_PS2": 21.5383,
        "MRE_FH1": 48.1807,
        "MRE_ALL": 27.35,
        "AoP_absolute_error_deg": 10.47,
    },
    "testing": {
        "MRE_PS1": 10.6720,
        "MRE_PS2": 15.6234,
        "MRE_FH1": 39.1866,
        "MRE_ALL": 21.83,
        "AoP_absolute_error_deg": 8.37,
    },
}
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_PATTERN = re.compile(
    rf"Epoch\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s*-\s*"
    rf"Training Loss:\s*(?P<loss>{FLOAT_PATTERN})\s*,\s*"
    rf"Coordinate Distance:\s*(?P<coord>{FLOAT_PATTERN})"
)


@dataclass(frozen=True)
class TrainingHistory:
    epochs: np.ndarray
    train_loss: np.ndarray
    coordinate_distance: np.ndarray
    declared_epochs: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--testing-metrics", type=Path, required=True)
    parser.add_argument("--good-image", type=Path, required=True)
    parser.add_argument("--median-image", type=Path, required=True)
    parser.add_argument("--poor-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--issue",
        action="append",
        default=[],
        help="Current issue to show; repeat for multiple items",
    )
    return parser


def parse_training_log(path: Path) -> TrainingHistory:
    """Parse the official script's per-epoch summary lines."""

    if not path.is_file():
        raise FileNotFoundError(f"Training log does not exist: {path}")
    by_epoch: dict[int, tuple[int, float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_PATTERN.search(line)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        by_epoch[epoch] = (
            int(match.group("total")),
            float(match.group("loss")),
            float(match.group("coord")),
        )
    if not by_epoch:
        raise ValueError(
            "No official epoch summaries found; expected 'Epoch N/T - Training Loss: ..., "
            "Coordinate Distance: ...'"
        )

    ordered_epochs = sorted(by_epoch)
    totals = {by_epoch[epoch][0] for epoch in ordered_epochs}
    if len(totals) != 1:
        raise ValueError(
            f"Training log contains inconsistent declared epoch totals: {sorted(totals)}"
        )
    history = TrainingHistory(
        epochs=np.asarray(ordered_epochs, dtype=np.int64),
        train_loss=np.asarray([by_epoch[epoch][1] for epoch in ordered_epochs], dtype=np.float64),
        coordinate_distance=np.asarray(
            [by_epoch[epoch][2] for epoch in ordered_epochs], dtype=np.float64
        ),
        declared_epochs=totals.pop(),
    )
    if not np.isfinite(history.train_loss).all() or not np.isfinite(
        history.coordinate_distance
    ).all():
        raise ValueError("Training log contains a non-finite loss or coordinate distance")
    return history


def load_metrics(path: Path, expected_split: str) -> dict[str, float | int]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics JSON does not exist: {path}")
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics JSON must contain an object: {path}")
    if payload.get("split") != expected_split:
        raise ValueError(
            f"Expected {expected_split!r} metrics but JSON declares {payload.get('split')!r}"
        )
    output: dict[str, float | int] = {}
    for key in METRIC_KEYS:
        value = float(payload[key])
        if not np.isfinite(value):
            raise ValueError(f"Metric {key} must be finite in {path}")
        output[key] = value
    count = int(payload["n_images"])
    if count <= 0:
        raise ValueError(f"n_images must be positive in {path}")
    output["n_images"] = count
    return output


def _configure_chinese_font() -> None:
    preferred = (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        }
    )


def _add_panel_title(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=6, color="#172554")


def _draw_settings(axis: plt.Axes, history: TrainingHistory) -> None:
    axis.axis("off")
    _add_panel_title(axis, "数据与训练设置")
    completed = int(history.epochs.max())
    lines = (
        "数据：导师指定的 Kaggle aspirexxx 当前公开包；训练 300，validation 100，testing 501",
        "输入：RGB 512×512；标签：3×64×64 Gaussian heatmap，σ=2",
        "模型：官方 HeatmapUNet；损失：heatmap MSE；解码：hard argmax",
        "训练：batch 4，150 epochs，Adam，lr=1e-4，weight decay=1e-4",
        "调度：StepLR(step=15, gamma=0.5)；seed=42",
        f"日志进度：epoch {completed}/{history.declared_epochs}",
    )
    axis.text(
        0.02,
        0.93,
        "\n".join(lines),
        transform=axis.transAxes,
        va="top",
        fontsize=7.25,
        linespacing=1.55,
        color="#1f2937",
    )
    axis.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            fill=False,
            linewidth=0.8,
            edgecolor="#cbd5e1",
        )
    )


def _draw_training_curve(axis: plt.Axes, history: TrainingHistory) -> None:
    _add_panel_title(axis, "训练曲线")
    loss_line = axis.plot(
        history.epochs,
        history.train_loss,
        color="#2563eb",
        linewidth=1.6,
        label="Train loss",
    )[0]
    axis.set_xlabel("Epoch", fontsize=7)
    axis.set_ylabel("Train loss", fontsize=7, color="#2563eb")
    axis.tick_params(axis="both", labelsize=6.5)
    axis.tick_params(axis="y", colors="#2563eb")
    axis.grid(alpha=0.22, linewidth=0.5)

    coordinate_axis = axis.twinx()
    coordinate_line = coordinate_axis.plot(
        history.epochs,
        history.coordinate_distance,
        color="#dc2626",
        linewidth=1.4,
        label="Coord distance",
    )[0]
    coordinate_axis.set_ylabel("Coord distance", fontsize=7, color="#dc2626")
    coordinate_axis.tick_params(axis="y", labelsize=6.5, colors="#dc2626")
    axis.legend(
        [loss_line, coordinate_line],
        ["Train loss", "Coord distance"],
        loc="upper right",
        fontsize=6.5,
        frameon=False,
    )


def _format_value(value: float, *, angle: bool = False) -> str:
    return f"{value:.2f}{'°' if angle else ''}"


def _draw_metric_table(
    axis: plt.Axes,
    validation: dict[str, float | int],
    testing: dict[str, float | int],
) -> None:
    axis.axis("off")
    _add_panel_title(axis, "官方结果 vs 本次复现")
    row_labels = ("PS1", "PS2", "FH1", "ALL", "AoP")
    table_rows: list[list[str]] = []
    for label, key in zip(row_labels, METRIC_KEYS, strict=True):
        angle = key == "AoP_absolute_error_deg"
        official_val = OFFICIAL_METRICS["validation"][key]
        reproduced_val = float(validation[key])
        official_test = OFFICIAL_METRICS["testing"][key]
        reproduced_test = float(testing[key])
        validation_delta = reproduced_val - official_val
        testing_delta = reproduced_test - official_test
        table_rows.append(
            [
                label,
                _format_value(official_val, angle=angle),
                f"{_format_value(reproduced_val, angle=angle)}\nΔ {validation_delta:+.2f}",
                _format_value(official_test, angle=angle),
                f"{_format_value(reproduced_test, angle=angle)}\nΔ {testing_delta:+.2f}",
            ]
        )
    table = axis.table(
        cellText=table_rows,
        colLabels=("指标", "官方 Val", "复现 Val", "官方 Test", "复现 Test"),
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.0, 1.0, 0.91),
        colWidths=(0.15, 0.20, 0.23, 0.20, 0.23),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.9)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold", color="#172554")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")


def _draw_case(axis: plt.Axes, image_path: Path, title: str) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Case visualization does not exist: {image_path}")
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    axis.imshow(image)
    axis.set_title(title, fontsize=8.5, fontweight="bold", color="#172554", pad=4)
    axis.axis("off")


def _draw_issues(axis: plt.Axes, issues: list[str], history: TrainingHistory) -> None:
    axis.axis("off")
    _add_panel_title(axis, "当前问题")
    displayed = list(issues)
    completed = int(history.epochs.max())
    if completed < history.declared_epochs:
        displayed.insert(0, f"训练日志仅到 epoch {completed}/{history.declared_epochs}。")
    if not displayed:
        displayed = [
            "官方仓库未公开独立评价脚本。",
            "论文未说明 Table 2 使用的 checkpoint 选择规则。",
        ]
    wrapped: list[str] = []
    for number, issue in enumerate(displayed, start=1):
        lines = textwrap.wrap(issue, width=28) or [issue]
        wrapped.append(f"{number}. {lines[0]}")
        wrapped.extend(f"   {line}" for line in lines[1:])
    axis.text(
        0.04,
        0.94,
        "\n".join(wrapped),
        transform=axis.transAxes,
        va="top",
        fontsize=7.4,
        linespacing=1.55,
        color="#1f2937",
    )
    axis.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            fill=False,
            linewidth=0.8,
            edgecolor="#cbd5e1",
        )
    )


def generate_pdf(
    *,
    output: Path,
    history: TrainingHistory,
    validation: dict[str, float | int],
    testing: dict[str, float | int],
    case_images: tuple[Path, Path, Path],
    issues: list[str],
) -> None:
    _configure_chinese_font()
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    figure.text(
        0.035,
        0.963,
        "IUGC 2025 — 官方 T10 UNet Heatmap Baseline 复现",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        va="top",
    )
    figure.text(
        0.035,
        0.918,
        f"来源：{OFFICIAL_REPOSITORY}    commit：{OFFICIAL_COMMIT}",
        fontsize=7.2,
        color="#475569",
        va="top",
    )
    figure.add_artist(
        plt.Line2D([0.035, 0.965], [0.895, 0.895], color="#94a3b8", linewidth=0.8)
    )

    settings_axis = figure.add_axes((0.035, 0.625, 0.275, 0.245))
    curve_axis = figure.add_axes((0.335, 0.625, 0.315, 0.245))
    metrics_axis = figure.add_axes((0.675, 0.625, 0.29, 0.245))
    _draw_settings(settings_axis, history)
    _draw_training_curve(curve_axis, history)
    _draw_metric_table(metrics_axis, validation, testing)

    case_positions = (
        (0.035, 0.075, 0.205, 0.47),
        (0.255, 0.075, 0.205, 0.47),
        (0.475, 0.075, 0.205, 0.47),
    )
    for image_path, title, position in zip(
        case_images,
        ("较好案例 / Good", "中等案例 / Median", "较差案例 / Poor"),
        case_positions,
        strict=True,
    ):
        _draw_case(figure.add_axes(position), image_path, title)
    _draw_issues(figure.add_axes((0.715, 0.075, 0.25, 0.47)), issues, history)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        dpi=180,
        metadata={"Title": "IUGC 2025 Official T10 Baseline Reproduction"},
    )
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    history = parse_training_log(args.training_log)
    validation = load_metrics(args.validation_metrics, "validation")
    testing = load_metrics(args.testing_metrics, "testing")
    generate_pdf(
        output=args.output,
        history=history,
        validation=validation,
        testing=testing,
        case_images=(args.good_image, args.median_image, args.poor_image),
        issues=args.issue,
    )
    print(f"Saved one-page advisor PDF: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
