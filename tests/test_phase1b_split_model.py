from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from geoequi_ld.models.hrnet import (
    HRNetW32SharedHeatmap,
    HRNetW32SplitHeatmap,
    SplitHeatmapDecoder,
    count_trainable_parameters,
    initialize_split_from_shared,
)
from geoequi_ld.training.checkpoints import restore_checkpoint, save_checkpoint


def _snapshot_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(actual: nn.Module, expected: dict[str, Tensor]) -> None:
    actual_state = actual.state_dict()
    assert actual_state.keys() == expected.keys()
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual_state[name], expected_value, rtol=0, atol=0)


def _storage_pointers(tensors: Iterable[Tensor]) -> set[int]:
    return {tensor.untyped_storage().data_ptr() for tensor in tensors}


def _parameters_and_buffers(module: nn.Module) -> Iterable[Tensor]:
    yield from module.parameters()
    yield from module.buffers()


def _gradient_sum(module: nn.Module) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return sum(float(gradient.abs().sum()) for gradient in gradients)


def _clone_parameters(module: nn.Module) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def _assert_parameters_equal(module: nn.Module, expected: dict[str, Tensor]) -> None:
    actual = dict(module.named_parameters())
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name], expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("out_channels", [1, 2])
def test_split_decoder_has_exact_phase1b_structure(out_channels: int) -> None:
    decoder = SplitHeatmapDecoder(out_channels)
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
    assert decoder.output.out_channels == out_channels
    assert decoder.output.kernel_size == (1, 1)
    assert not any(isinstance(module, nn.Softmax) for module in decoder.modules())


def test_split_decoder_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="one or two"):
        SplitHeatmapDecoder(3)


def test_shared_initialization_is_equivalent_independent_and_checkpointable(
    tmp_path: Path,
) -> None:
    torch.manual_seed(19)
    shared = HRNetW32SharedHeatmap().eval()
    split = HRNetW32SplitHeatmap().eval()
    shared_before = _snapshot_state(shared)

    returned = initialize_split_from_shared(shared, split)
    assert returned is split
    assert split.training is shared.training is False
    _assert_state_equal(shared, shared_before)
    assert split.feature_contract == shared.feature_contract
    assert split.feature_contract.final_fusion_module_path == "backbone.stage4.2"
    assert count_trainable_parameters(shared) == 29_318_355
    assert count_trainable_parameters(split) == 29_332_275

    torch.testing.assert_close(
        split.ps_decoder.output.weight,
        shared.decoder.output.weight[:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        split.fh_decoder.output.weight,
        shared.decoder.output.weight[2:3],
        rtol=0,
        atol=0,
    )
    source_storage = _storage_pointers(_parameters_and_buffers(shared))
    split_storage = _storage_pointers(_parameters_and_buffers(split))
    ps_storage = _storage_pointers(_parameters_and_buffers(split.ps_decoder))
    fh_storage = _storage_pointers(_parameters_and_buffers(split.fh_decoder))
    assert source_storage.isdisjoint(split_storage)
    assert ps_storage.isdisjoint(fh_storage)

    inputs = torch.randn((1, 1, 64, 64), dtype=torch.float32)
    shared_eval_before = _snapshot_state(shared)
    split_eval_before = _snapshot_state(split)
    final_stage_shapes: list[tuple[int, ...]] = []

    def capture_final_stage(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        outputs: list[Tensor],
    ) -> None:
        final_stage_shapes.extend(tuple(output.shape) for output in outputs)

    handle = split.final_fusion_module.register_forward_hook(capture_final_stage)
    with torch.inference_mode():
        shared_output = shared(inputs)
        split_output = split(inputs)
    handle.remove()
    assert split_output.shape == (1, 3, 32, 32)
    assert final_stage_shapes == [
        (1, 32, 16, 16),
        (1, 64, 8, 8),
        (1, 128, 4, 4),
        (1, 256, 2, 2),
    ]
    torch.testing.assert_close(split_output, shared_output, rtol=1e-5, atol=1e-6)
    _assert_state_equal(shared, shared_eval_before)
    _assert_state_equal(split, split_eval_before)

    optimizer = torch.optim.Adam(split.parameters(), lr=0.001, foreach=False)
    checkpoint = save_checkpoint(
        tmp_path / "phase1b_split_roundtrip.pt",
        model=split,
        optimizer=optimizer,
        epoch=0,
        config={"model": "HRNetW32SplitHeatmap"},
        seed=19,
        metrics={"MRE_ALL": 0.0},
    )
    with torch.no_grad():
        split.ps_decoder.output.weight.add_(1.0)
        split.fh_decoder.bn1.running_mean.add_(1.0)
    restore_checkpoint(checkpoint, model=split, optimizer=optimizer)
    with torch.inference_mode():
        restored_output = split(inputs)
    torch.testing.assert_close(restored_output, split_output, rtol=0, atol=0)


def test_task_losses_reach_backbone_but_not_the_other_decoder() -> None:
    torch.manual_seed(23)
    model = HRNetW32SplitHeatmap().eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    inputs = torch.randn((1, 1, 64, 64), dtype=torch.float32)

    fh_before = _clone_parameters(model.fh_decoder)
    ps_output = model(inputs)
    ps_output[:, :2].abs().mean().backward()
    assert _gradient_sum(model.backbone) > 0
    assert _gradient_sum(model.ps_decoder) > 0
    assert _gradient_sum(model.fh_decoder) == 0
    optimizer.step()
    _assert_parameters_equal(model.fh_decoder, fh_before)

    optimizer.zero_grad(set_to_none=True)
    ps_before = _clone_parameters(model.ps_decoder)
    fh_output = model(inputs)
    fh_output[:, 2:].abs().mean().backward()
    assert _gradient_sum(model.backbone) > 0
    assert _gradient_sum(model.fh_decoder) > 0
    assert _gradient_sum(model.ps_decoder) == 0
    optimizer.step()
    _assert_parameters_equal(model.ps_decoder, ps_before)
