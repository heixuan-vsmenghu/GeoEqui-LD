from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from geoequi_ld.data.heatmaps import generate_gaussian_heatmaps
from geoequi_ld.geometry.coordinates import pixel_to_normalized, resize_points
from geoequi_ld.models.dsnt import DSNT
from geoequi_ld.training.checkpoints import read_checkpoint, restore_checkpoint, save_checkpoint
from geoequi_ld.training.config import SupervisedTrainingConfig
from geoequi_ld.training.engine import (
    compute_supervised_losses,
    evaluate_model,
    train_for_steps,
)
from geoequi_ld.training.runtime import resolve_device, seed_everything


class SyntheticLandmarks(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, count: int = 2) -> None:
        self.count = count
        # Integer heatmap centres mapped through the align_corners=True
        # contract keep this synthetic evaluation analytically unambiguous.
        self.points_original = torch.tensor(
            [[3.0, 4.0], [12.0, 3.0], [10.0, 12.0]],
            dtype=torch.float32,
        ) * (31.0 / 15.0)
        self.points_heatmap = resize_points(
            self.points_original,
            (32, 32),
            (16, 16),
            align_corners=True,
        )
        self.points_normalized = pixel_to_normalized(
            self.points_original,
            (32, 32),
            align_corners=True,
        )
        self.heatmaps = generate_gaussian_heatmaps(
            self.points_heatmap,
            size_hw=(16, 16),
            sigma=1.0,
        )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "filename": f"synthetic_{index}.png",
            "image": torch.zeros((1, 32, 32), dtype=torch.float32),
            "heatmaps": self.heatmaps.clone(),
            "points_original_px": self.points_original.clone(),
            "points_input_px": self.points_original.clone(),
            "points_heatmap_px": self.points_heatmap.clone(),
            "points_normalized": self.points_normalized.clone(),
            "valid_mask": torch.ones(3, dtype=torch.bool),
            "aop_degrees": torch.tensor(float("nan")),
            "original_size_hw": torch.tensor((32, 32), dtype=torch.int64),
        }


class LearnedHeatmaps(nn.Module):
    def __init__(self, initial: torch.Tensor | None = None) -> None:
        super().__init__()
        value = torch.zeros((3, 16, 16)) if initial is None else initial.clone()
        self.logits = nn.Parameter(value)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(images.shape[0], -1, -1, -1)


def tiny_config() -> SupervisedTrainingConfig:
    config = SupervisedTrainingConfig(
        input_size_hw=(32, 32),
        heatmap_size_hw=(16, 16),
        sigma_heatmap_px=1.0,
        base_channels=2,
        batch_size=2,
        epochs=1,
        learning_rate=0.1,
        weight_decay=0.0,
        num_workers=0,
    )
    config.validate()
    return config


def test_supervised_loss_has_separate_heatmap_and_coordinate_gradients() -> None:
    sample = SyntheticLandmarks()[0]
    logits = torch.zeros((1, 3, 16, 16), requires_grad=True)
    losses = compute_supervised_losses(
        logits,
        sample["heatmaps"].unsqueeze(0),  # type: ignore[union-attr]
        sample["points_normalized"].unsqueeze(0),  # type: ignore[union-attr]
        sample["valid_mask"].unsqueeze(0),  # type: ignore[union-attr]
        dsnt=DSNT(temperature=0.05, align_corners=True),
        heatmap_weight=1.0,
        coordinate_weight=1.0,
        distribution_weight=1.0,
    )
    losses.total.backward()
    assert losses.heatmap_mse.item() > 0
    assert losses.coordinate_smooth_l1.item() > 0
    assert losses.distribution_js.item() > 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_heatmap_only_loss_skips_dsnt_and_js(monkeypatch: object) -> None:
    sample = SyntheticLandmarks()[0]
    logits = torch.zeros((1, 3, 16, 16), requires_grad=True)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("spatial_softmax must not run for B0")

    monkeypatch.setattr("geoequi_ld.training.engine.spatial_softmax", fail_if_called)  # type: ignore[attr-defined]
    losses = compute_supervised_losses(
        logits,
        sample["heatmaps"].unsqueeze(0),  # type: ignore[union-attr]
        sample["points_normalized"].unsqueeze(0),  # type: ignore[union-attr]
        sample["valid_mask"].unsqueeze(0),  # type: ignore[union-attr]
        dsnt=DSNT(temperature=0.05, align_corners=True),
        heatmap_weight=1.0,
        coordinate_weight=0.0,
        distribution_weight=0.0,
    )
    assert losses.coordinate_smooth_l1.item() == 0.0
    assert losses.distribution_js.item() == 0.0
    losses.total.backward()
    assert logits.grad is not None


def test_train_for_steps_updates_parameters_and_honours_limit() -> None:
    seed_everything(7)
    config = tiny_config()
    loader = DataLoader(SyntheticLandmarks(), batch_size=2, shuffle=False)
    model = LearnedHeatmaps()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    before = model.logits.detach().clone()
    history = train_for_steps(
        model,
        loader,
        optimizer,
        dsnt=DSNT(temperature=config.dsnt_temperature, align_corners=True),
        device=torch.device("cpu"),
        config=config,
        max_steps=3,
    )
    assert len(history) == 3
    assert history[-1]["step"] == 3
    assert not torch.equal(before, model.logits.detach())
    assert all(torch.isfinite(torch.tensor(row["total_loss"])) for row in history)


def test_evaluation_reports_original_pixel_mre_and_aop() -> None:
    config = tiny_config()
    dataset = SyntheticLandmarks()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = LearnedHeatmaps(dataset.heatmaps)
    metrics = evaluate_model(
        model,
        loader,
        dsnt=DSNT(temperature=config.dsnt_temperature, align_corners=True),
        device=torch.device("cpu"),
        config=config,
    )
    assert metrics["n_samples"] == 2
    assert metrics["n_valid_aop"] == 2
    assert float(metrics["MRE_ALL"]) < 0.05
    assert float(metrics["aop_mae_deg"]) < 0.05


def test_evaluation_supports_argmax_on_the_same_checkpoint() -> None:
    config = tiny_config()
    dataset = SyntheticLandmarks()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = LearnedHeatmaps(dataset.heatmaps)
    metrics = evaluate_model(
        model,
        loader,
        dsnt=DSNT(temperature=config.dsnt_temperature, align_corners=True),
        device=torch.device("cpu"),
        config=config,
        decoder="argmax",
    )
    assert metrics["decoder"] == "argmax"
    assert float(metrics["MRE_ALL"]) < 1e-5
    assert float(metrics["aop_mae_deg"]) < 1e-5


def test_checkpoint_round_trip_contains_required_provenance(tmp_path: Path) -> None:
    model = nn.Conv2d(1, 3, kernel_size=1)
    optimizer = Adam(model.parameters(), lr=1e-3)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=4,
        config={"training": tiny_config().to_dict()},
        seed=123,
        metrics={"MRE_ALL": 2.5, "aop_mae_deg": 1.25},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    payload = restore_checkpoint(path, model=model, optimizer=optimizer)
    assert payload["epoch"] == 4
    assert payload["seed"] == 123
    assert payload["config"]["training"]["dsnt_temperature"] == 0.05
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    assert read_checkpoint(path)["metrics"]["MRE_ALL"] == 2.5


def test_device_resolution_is_explicit() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
