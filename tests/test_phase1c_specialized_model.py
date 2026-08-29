from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d

from geoequi_ld.models import DSNT
from geoequi_ld.models.hrnet import (
    HRNetW32SharedHeatmap,
    HRNetW32SplitHeatmap,
    count_trainable_parameters,
    initialize_split_from_shared,
)
from geoequi_ld.models.specialized import (
    FHFeatureEnhancer,
    HRNetW32SpecializedHeatmap,
    LayerNorm2d,
    PSFeatureEnhancer,
    initialize_specialized_from_split,
)
from geoequi_ld.training.checkpoints import restore_checkpoint, save_checkpoint


def _gradient_sum(module: nn.Module) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return sum(float(gradient.abs().sum()) for gradient in gradients)


def _assert_no_gradients(module: nn.Module) -> None:
    assert all(parameter.grad is None for parameter in module.parameters())


def _storage_pointers(tensors: Iterable[Tensor]) -> set[int]:
    return {tensor.untyped_storage().data_ptr() for tensor in tensors}


def _parameters_and_buffers(module: nn.Module) -> Iterable[Tensor]:
    yield from module.parameters()
    yield from module.buffers()


def _parameter_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def _assert_parameters_unchanged(module: nn.Module, expected: dict[str, Tensor]) -> None:
    actual = dict(module.named_parameters())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)


class _SyntheticFeatureBackbone(nn.Module):
    """Cheap differentiable reduction-4 trunk for CPU-only wiring tests."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 32, kernel_size=1)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        return [self.projection(F.avg_pool2d(inputs, kernel_size=4, stride=4))]


def _lightweight_specialized_model() -> HRNetW32SpecializedHeatmap:
    model = HRNetW32SpecializedHeatmap()
    model.backbone = _SyntheticFeatureBackbone()
    return model


def _weighted_loss(output: Tensor) -> Tensor:
    weights = torch.linspace(
        -0.7,
        1.3,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    return (output * weights).mean()


def test_layer_norm_2d_matches_explicit_nhwc_channel_layer_norm() -> None:
    torch.manual_seed(7)
    layer = LayerNorm2d(32)
    inputs = torch.randn((2, 32, 9, 11), dtype=torch.float32)
    actual = layer(inputs)
    channels_last = inputs.permute(0, 2, 3, 1)
    channel_mean = channels_last.mean(dim=-1, keepdim=True)
    channel_variance = channels_last.var(dim=-1, unbiased=False, keepdim=True)
    manually_normalized = (channels_last - channel_mean) * torch.rsqrt(
        channel_variance + layer.norm.eps
    )
    expected = (
        manually_normalized * layer.norm.weight + layer.norm.bias
    ).permute(0, 3, 1, 2)
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
    assert actual.shape == inputs.shape
    assert not any(isinstance(module, nn.BatchNorm2d | nn.GroupNorm) for module in layer.modules())


def test_ps_enhancer_uses_real_modulated_deform_conv_and_all_paths_have_gradients() -> None:
    torch.manual_seed(11)
    enhancer = PSFeatureEnhancer()
    assert isinstance(enhancer.deform, DeformConv2d)
    assert enhancer.deform.in_channels == enhancer.deform.out_channels == 32
    assert enhancer.deform.kernel_size == (3, 3)
    assert enhancer.deform.stride == (1, 1)
    assert enhancer.deform.padding == (1, 1)
    assert enhancer.offset_mask.in_channels == 32
    assert enhancer.offset_mask.out_channels == 27
    assert enhancer.offset_mask.kernel_size == (3, 3)
    assert enhancer.spatial_attention.in_channels == 32
    assert enhancer.spatial_attention.out_channels == 1
    assert enhancer.spatial_attention.kernel_size == (1, 1)

    features = torch.randn((2, 32, 8, 10), dtype=torch.float32, requires_grad=True)
    offsets, masks, mask_logits = enhancer.predict_offset_and_mask(features)
    assert offsets.shape == (2, 18, 8, 10)
    assert masks.shape == mask_logits.shape == (2, 9, 8, 10)
    torch.testing.assert_close(offsets, torch.zeros_like(offsets), rtol=0, atol=0)
    torch.testing.assert_close(mask_logits, torch.zeros_like(mask_logits), rtol=0, atol=0)
    torch.testing.assert_close(masks, torch.full_like(masks, 0.5), rtol=0, atol=0)
    assert torch.isfinite(offsets).all()
    assert torch.isfinite(mask_logits).all()
    assert torch.isfinite(masks).all()
    assert enhancer.initialization_summary["ordinary_conv_fallback"] is False
    assert enhancer.initialization_summary["required_torchvision_base_version"] == "0.20.1"

    deform_arguments: list[tuple[Tensor, ...]] = []

    def capture_deform_arguments(
        _module: nn.Module,
        arguments: tuple[Tensor, ...],
    ) -> None:
        deform_arguments.append(arguments)

    deform_hook = enhancer.deform.register_forward_pre_hook(capture_deform_arguments)
    output = enhancer(features)
    deform_hook.remove()
    assert len(deform_arguments) == 1
    assert len(deform_arguments[0]) == 3
    assert deform_arguments[0][1].shape == (2, 18, 8, 10)
    assert deform_arguments[0][2].shape == (2, 9, 8, 10)
    torch.testing.assert_close(deform_arguments[0][2], masks, rtol=0, atol=0)
    attention = torch.sigmoid(enhancer.spatial_attention(features))
    assert attention.shape == (2, 1, 8, 10)
    assert torch.isfinite(attention).all()
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    _weighted_loss(output).backward()
    assert features.grad is not None and float(features.grad.abs().sum()) > 0
    predictor_gradient = enhancer.offset_mask.weight.grad
    assert predictor_gradient is not None
    assert float(predictor_gradient[:18].abs().sum()) > 0
    assert float(predictor_gradient[18:].abs().sum()) > 0
    assert enhancer.deform.weight.grad is not None
    assert float(enhancer.deform.weight.grad.abs().sum()) > 0
    assert _gradient_sum(enhancer.spatial_attention) > 0
    assert _gradient_sum(enhancer.norm) > 0
    for name, parameter in enhancer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_fh_enhancer_has_locked_aspp_se_structure_and_all_parameters_have_gradients() -> None:
    torch.manual_seed(13)
    enhancer = FHFeatureEnhancer()
    d1 = enhancer.aspp_d1[0]
    d3 = enhancer.aspp_d3[0]
    d6 = enhancer.aspp_d6[0]
    assert isinstance(d1, nn.Conv2d) and d1.kernel_size == (1, 1) and d1.dilation == (1, 1)
    assert isinstance(d3, nn.Conv2d) and d3.kernel_size == (3, 3) and d3.dilation == (3, 3)
    assert d3.padding == (3, 3)
    assert isinstance(d6, nn.Conv2d) and d6.kernel_size == (3, 3) and d6.dilation == (6, 6)
    assert d6.padding == (6, 6)
    assert enhancer.aspp_projection[0].in_channels == 96
    assert enhancer.aspp_projection[0].out_channels == 32
    assert enhancer.se_reduce.in_channels == 32 and enhancer.se_reduce.out_channels == 8
    assert enhancer.se_expand.in_channels == 8 and enhancer.se_expand.out_channels == 32
    assert not any(isinstance(module, nn.BatchNorm2d) for module in enhancer.modules())

    features = torch.randn((2, 32, 9, 11), dtype=torch.float32, requires_grad=True)
    weights = enhancer.channel_weights(features)
    assert weights.shape == (2, 32, 1, 1)
    assert bool(((weights >= 0) & (weights <= 1)).all())
    output = enhancer(features)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    _weighted_loss(output).backward()
    assert features.grad is not None and float(features.grad.abs().sum()) > 0
    for name, parameter in enhancer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0, name


def test_h3_copies_seed42_h2_base_without_aliasing_or_touching_enhancer_initialization() -> None:
    torch.manual_seed(42)
    shared = HRNetW32SharedHeatmap().eval()
    h2 = HRNetW32SplitHeatmap().eval()
    initialize_split_from_shared(shared, h2)
    h3 = HRNetW32SpecializedHeatmap.from_split(h2)

    assert h3.training is False
    assert all(not module.training for module in h3.modules())
    assert h3.feature_contract == h2.feature_contract
    for source, target in (
        (h2.backbone, h3.backbone),
        (h2.ps_decoder, h3.ps_decoder),
        (h2.fh_decoder, h3.fh_decoder),
    ):
        source_state = source.state_dict()
        target_state = target.state_dict()
        assert source_state.keys() == target_state.keys()
        for name in source_state:
            torch.testing.assert_close(source_state[name], target_state[name], rtol=0, atol=0)
        assert _storage_pointers(_parameters_and_buffers(source)).isdisjoint(
            _storage_pointers(_parameters_and_buffers(target))
        )

    assert count_trainable_parameters(h2) == 29_332_275
    assert count_trainable_parameters(h3) == 29_372_695
    assert count_trainable_parameters(h3) - count_trainable_parameters(h2) == 40_420
    summary = h3.initialization_summary
    assert summary["backbone_and_decoders_copied"] is True
    assert summary["base_parameter_storage_aliased"] is False
    assert summary["complete_function_initially_equivalent_to_h2"] is False
    assert summary["ps_enhancer"]["initial_offset"] == 0.0
    assert summary["ps_enhancer"]["initial_mask_after_sigmoid"] == 0.5
    assert torch.count_nonzero(h3.ps_enhancer.offset_mask.weight) == 0
    assert torch.count_nonzero(h3.ps_enhancer.offset_mask.bias) == 0


def test_specialized_model_512_contract_channel_order_dsnt_and_finite_modes() -> None:
    torch.manual_seed(17)
    model = _lightweight_specialized_model().eval()
    inputs = torch.randn((1, 1, 512, 512), dtype=torch.float32)
    captured_shapes: dict[str, tuple[int, ...]] = {}

    def capture_shape(name: str):  # type: ignore[no-untyped-def]
        def hook(_module: nn.Module, arguments: tuple[Tensor, ...], output: Tensor) -> None:
            captured_shapes[f"{name}_input"] = tuple(arguments[0].shape)
            captured_shapes[f"{name}_output"] = tuple(output.shape)

        return hook

    ps_hook = model.ps_enhancer.register_forward_hook(capture_shape("ps"))
    fh_hook = model.fh_enhancer.register_forward_hook(capture_shape("fh"))
    with torch.no_grad():
        eval_output = model(inputs)
    ps_hook.remove()
    fh_hook.remove()
    assert captured_shapes == {
        "ps_input": (1, 32, 128, 128),
        "ps_output": (1, 32, 128, 128),
        "fh_input": (1, 32, 128, 128),
        "fh_output": (1, 32, 128, 128),
    }
    assert eval_output.shape == (1, 3, 256, 256)
    assert torch.isfinite(eval_output).all()
    coordinates = DSNT(temperature=0.05, align_corners=True)(eval_output)
    assert coordinates.shape == (1, 3, 2)
    assert torch.isfinite(coordinates).all()

    model.train()
    with torch.no_grad():
        train_output = model(inputs)
    assert train_output.shape == eval_output.shape
    assert torch.isfinite(train_output).all()

    model.eval()
    with torch.no_grad():
        model.ps_decoder.output.weight.zero_()
        model.ps_decoder.output.bias.copy_(torch.tensor((1.0, 2.0)))
        model.fh_decoder.output.weight.zero_()
        model.fh_decoder.output.bias.fill_(3.0)
        ordered = model(torch.zeros_like(inputs))
    torch.testing.assert_close(ordered[:, 0], torch.ones_like(ordered[:, 0]))
    torch.testing.assert_close(ordered[:, 1], torch.full_like(ordered[:, 1], 2.0))
    torch.testing.assert_close(ordered[:, 2], torch.full_like(ordered[:, 2], 3.0))


def test_ps_fh_and_combined_losses_have_expected_gradient_routing() -> None:
    torch.manual_seed(23)
    model = _lightweight_specialized_model().train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
        foreach=False,
    )
    inputs = torch.randn((1, 1, 32, 32), dtype=torch.float32)

    fh_enhancer_before = _parameter_snapshot(model.fh_enhancer)
    fh_decoder_before = _parameter_snapshot(model.fh_decoder)
    _weighted_loss(model.forward_ps(inputs)).backward()
    assert _gradient_sum(model.backbone) > 0
    assert _gradient_sum(model.ps_enhancer) > 0
    assert _gradient_sum(model.ps_decoder) > 0
    _assert_no_gradients(model.fh_enhancer)
    _assert_no_gradients(model.fh_decoder)
    optimizer.step()
    _assert_parameters_unchanged(model.fh_enhancer, fh_enhancer_before)
    _assert_parameters_unchanged(model.fh_decoder, fh_decoder_before)

    optimizer.zero_grad(set_to_none=True)
    ps_enhancer_before = _parameter_snapshot(model.ps_enhancer)
    ps_decoder_before = _parameter_snapshot(model.ps_decoder)
    _weighted_loss(model.forward_fh(inputs)).backward()
    assert _gradient_sum(model.backbone) > 0
    assert _gradient_sum(model.fh_enhancer) > 0
    assert _gradient_sum(model.fh_decoder) > 0
    _assert_no_gradients(model.ps_enhancer)
    _assert_no_gradients(model.ps_decoder)
    optimizer.step()
    _assert_parameters_unchanged(model.ps_enhancer, ps_enhancer_before)
    _assert_parameters_unchanged(model.ps_decoder, ps_decoder_before)

    optimizer.zero_grad(set_to_none=True)
    _weighted_loss(model(inputs)).backward()
    assert _gradient_sum(model.backbone) > 0
    assert _gradient_sum(model.ps_enhancer) > 0
    assert _gradient_sum(model.ps_decoder) > 0
    assert _gradient_sum(model.fh_enhancer) > 0
    assert _gradient_sum(model.fh_decoder) > 0


def test_specialized_checkpoint_round_trip_preserves_output(tmp_path: Path) -> None:
    torch.manual_seed(29)
    model = _lightweight_specialized_model().eval()
    model._base_initialization_copied.fill_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn((1, 1, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        expected = model(inputs).clone()
    checkpoint = tmp_path / "h3.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=3,
        config={"phase": "phase1c", "testing_frozen": True},
        seed=42,
        metrics={"MRE_ALL": 1.0},
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
        model._base_initialization_copied.fill_(False)
    payload = restore_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        map_location="cpu",
    )
    with torch.no_grad():
        actual = model(inputs)
    assert payload["epoch"] == 3
    assert payload["seed"] == 42
    assert model.base_initialization_copied is True
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_specialized_initializer_rejects_wrong_source_type() -> None:
    source = HRNetW32SpecializedHeatmap()
    target = HRNetW32SpecializedHeatmap()
    with pytest.raises(TypeError, match="HRNetW32SplitHeatmap"):
        initialize_specialized_from_split(source, target)  # type: ignore[arg-type]
