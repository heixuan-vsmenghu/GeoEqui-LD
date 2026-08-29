from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from geoequi_ld.models.hrnet import (
    PINNED_TIMM_VERSION,
    HRNetW32SharedHeatmap,
    SharedHeatmapDecoder,
    count_trainable_parameters,
)
from geoequi_ld.training.checkpoints import restore_checkpoint, save_checkpoint
from geoequi_ld.training.phase1a_config import build_phase1a_adam, load_phase1a_hrnet_config

ROOT = Path(__file__).resolve().parents[1]


def _nonzero_finite_gradient_sum(model: nn.Module, name_fragment: str) -> float:
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name_fragment in name and parameter.grad is not None
    ]
    assert gradients, f"No gradients found for parameter group {name_fragment!r}"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return sum(float(gradient.abs().sum()) for gradient in gradients)


def test_shared_decoder_has_exact_phase1a_structure() -> None:
    decoder = SharedHeatmapDecoder()
    assert decoder.conv1.in_channels == 32
    assert decoder.conv1.out_channels == 32
    assert decoder.conv1.kernel_size == (3, 3)
    assert decoder.conv1.bias is None
    assert isinstance(decoder.bn1, nn.BatchNorm2d)
    assert isinstance(decoder.act1, nn.GELU)
    assert decoder.conv2.in_channels == 32
    assert decoder.conv2.out_channels == 16
    assert decoder.conv2.kernel_size == (3, 3)
    assert decoder.conv2.bias is None
    assert isinstance(decoder.bn2, nn.BatchNorm2d)
    assert isinstance(decoder.act2, nn.GELU)
    assert decoder.output.in_channels == 16
    assert decoder.output.out_channels == 3
    assert decoder.output.kernel_size == (1, 1)
    assert not any(isinstance(module, nn.Softmax) for module in decoder.modules())


def test_hrnet_small_cpu_stage4_gradient_first_step_and_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model = HRNetW32SharedHeatmap()
    assert model.feature_contract.timm_version == PINNED_TIMM_VERSION
    assert model.feature_contract.channels == (32,)
    assert model.feature_contract.reductions == (4,)
    assert model.feature_contract.out_indices == (1,)
    assert model.feature_contract.final_fusion_module_path == "backbone.stage4.2"
    assert count_trainable_parameters(model) == 29_318_355

    final_stage_outputs: list[list[torch.Tensor]] = []
    decoder_inputs: list[torch.Tensor] = []

    def capture_final_stage(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: list[torch.Tensor],
    ) -> None:
        final_stage_outputs.append(output)

    def capture_decoder_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        decoder_inputs.append(inputs[0])

    stage_handle = model.final_fusion_module.register_forward_hook(capture_final_stage)
    decoder_handle = model.decoder.register_forward_pre_hook(capture_decoder_input)
    model.train()
    assert all(module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    inputs = torch.randn((1, 1, 64, 64), dtype=torch.float32)
    config = load_phase1a_hrnet_config(ROOT / "configs" / "phase1a_hrnet_shared.yaml")
    optimizer = build_phase1a_adam(model.parameters(), config)
    assert optimizer.defaults["foreach"] is False
    before = model.decoder.output.weight.detach().clone()

    output = model(inputs)
    stage_handle.remove()
    decoder_handle.remove()
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()
    assert len(final_stage_outputs) == 1
    assert [tuple(value.shape) for value in final_stage_outputs[0]] == [
        (1, 32, 16, 16),
        (1, 64, 8, 8),
        (1, 128, 4, 4),
        (1, 256, 2, 2),
    ]
    assert len(decoder_inputs) == 1
    torch.testing.assert_close(decoder_inputs[0], final_stage_outputs[0][0])

    loss = output.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert _nonzero_finite_gradient_sum(model, "backbone.stage2.0.branches.1") > 0
    assert _nonzero_finite_gradient_sum(model, "backbone.stage3.3.branches.2") > 0
    assert _nonzero_finite_gradient_sum(model, "backbone.stage4.2.branches.3") > 0
    assert _nonzero_finite_gradient_sum(model, "decoder") > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    assert not torch.equal(before, model.decoder.output.weight.detach())
    output_parameter = model.decoder.output.weight
    assert output_parameter in optimizer.state
    assert {"step", "exp_avg", "exp_avg_sq"} <= set(optimizer.state[output_parameter])

    model.eval()
    assert all(
        not module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    )
    with torch.inference_mode():
        expected = model(inputs)
    assert expected.shape == output.shape
    assert torch.isfinite(expected).all()

    checkpoint = save_checkpoint(
        tmp_path / "hrnet_roundtrip.pt",
        model=model,
        optimizer=optimizer,
        epoch=1,
        config={
            "feature_contract": model.feature_contract.to_dict(),
            "model": config.model.to_dict(),
            "optimizer": config.optimizer.to_dict(),
        },
        seed=7,
        metrics={"MRE_ALL": 1.0, "aop_mae_deg": 1.0},
    )
    saved_step = float(optimizer.state[output_parameter]["step"])
    with torch.no_grad():
        output_parameter.add_(1.0)
    optimizer.state.clear()
    payload = restore_checkpoint(checkpoint, model=model, optimizer=optimizer)
    assert payload["config"]["optimizer"]["foreach"] is False
    assert float(optimizer.state[output_parameter]["step"]) == saved_step
    with torch.inference_mode():
        restored = model(inputs)
    torch.testing.assert_close(restored, expected)

    with pytest.raises(ValueError, match="grayscale"):
        model(torch.zeros((1, 3, 64, 64)))
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.zeros((1, 1, 48, 64)))
