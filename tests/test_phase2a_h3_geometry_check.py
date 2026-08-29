from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from scripts.check_phase2a_h3_geometry import (
    build_parser,
    component_gradient_evidence,
    explicit_view_transforms,
    gradient_gate_passed,
    load_train_images_without_labels,
    require_private_output_path,
)


def test_parser_locks_check_to_one_or_two_samples() -> None:
    parser = build_parser()
    assert parser.parse_args([]).sample_count == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["--sample-count", "3"])


def test_output_must_be_fresh_private_phase2a_json(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "phase2a" / "check.json"
    assert require_private_output_path(output, repository_root=tmp_path) == output.resolve()
    with pytest.raises(PermissionError, match="runs/phase2a"):
        require_private_output_path(tmp_path / "reports" / "check.json", repository_root=tmp_path)
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        require_private_output_path(output, repository_root=tmp_path)


def test_train_loader_does_not_open_label_csv(tmp_path: Path) -> None:
    train = tmp_path / "Training" / "Labeled cases"
    validation = tmp_path / "Validation"
    train.mkdir(parents=True)
    validation.mkdir()
    Image.new("L", (40, 30), color=127).save(train / "one.png")
    # Both label paths are deliberately absent. Loading train pixels must still
    # work because this diagnostic never reads either label file.
    config = tmp_path / "phase05_local.yaml"
    config.write_text(
        "\n".join(
            (
                "phase: phase0.5",
                "splits:",
                "  train:",
                f"    image_dir: '{train.as_posix()}'",
                f"    labels_csv: '{(train / 'missing.csv').as_posix()}'",
                "    fh1_column: FH1",
                "    expected_fingerprint: {sample_count: 1}",
                "  validation:",
                f"    image_dir: '{validation.as_posix()}'",
                f"    labels_csv: '{(validation / 'missing.csv').as_posix()}'",
                "    fh1_column: FH1",
                "    expected_fingerprint: {sample_count: 1}",
            )
        ),
        encoding="utf-8",
    )
    images = load_train_images_without_labels(config, sample_count=1)
    assert images.shape == (1, 1, 512, 512)
    assert torch.isfinite(images).all()


def test_explicit_views_are_invertible_orientation_preserving_similarities() -> None:
    transform1, transform2 = explicit_view_transforms(
        2,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert transform1.shape == (2, 3, 3)
    assert transform2.shape == (2, 3, 3)
    for transform in (transform1, transform2):
        linear = transform[:, :2, :2]
        assert torch.all(torch.linalg.det(linear) > 0)
        torch.testing.assert_close(
            linear.transpose(-1, -2) @ linear,
            torch.diagonal(
                linear.transpose(-1, -2) @ linear,
                dim1=-2,
                dim2=-1,
            ).mean(-1)[:, None, None]
            * torch.eye(2, dtype=torch.float64),
        )
        assert torch.isfinite(torch.linalg.inv(transform)).all()


class _GradientToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.ps_enhancer = nn.Linear(2, 2, bias=False)
        self.fh_enhancer = nn.Linear(2, 2, bias=False)
        self.ps_decoder = nn.Linear(2, 2, bias=False)
        self.fh_decoder = nn.Linear(2, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shared = self.backbone(inputs)
        ps = self.ps_decoder(self.ps_enhancer(shared))
        fh = self.fh_decoder(self.fh_enhancer(shared))
        return torch.cat((ps, fh), dim=-1)


def test_component_gradient_gate_requires_each_h3_path_finite_and_nonzero() -> None:
    model = _GradientToyModel()
    model(torch.tensor([[0.3, -0.7]])).sum().backward()
    evidence = component_gradient_evidence(model)
    assert gradient_gate_passed(evidence)
    assert all(row["finite_nonzero_gradient_reached"] is True for row in evidence.values())

    model.fh_decoder.weight.grad = None
    failed = component_gradient_evidence(model)
    assert not gradient_gate_passed(failed)
    assert failed["fh_decoder"]["finite_nonzero_gradient_reached"] is False
