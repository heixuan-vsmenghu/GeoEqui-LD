from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoequi_ld.models.strict import StrictGeoEquiLDModel


class _SyntheticFeatureBackbone(nn.Module):
    """Cheap reduction-4 feature source that preserves the strict tensor contract."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 32, kernel_size=1)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        reduced = F.avg_pool2d(inputs, kernel_size=4, stride=4)
        return [self.projection(reduced)]


def _synthetic_strict_model() -> StrictGeoEquiLDModel:
    model = StrictGeoEquiLDModel()
    model.backbone = _SyntheticFeatureBackbone()
    return model


def test_strict_model_synthetic_backbone_has_exact_docx_shapes_and_channel_order() -> None:
    torch.manual_seed(17)
    model = _synthetic_strict_model().eval()
    inputs = torch.randn((1, 1, 512, 512), dtype=torch.float32)
    observed: dict[str, tuple[int, ...]] = {}

    def capture(name: str):  # type: ignore[no-untyped-def]
        def hook(_module: nn.Module, arguments: tuple[Tensor, ...], output: Tensor) -> None:
            observed[f"{name}_input"] = tuple(arguments[0].shape)
            observed[f"{name}_output"] = tuple(output.shape)

        return hook

    ps_hook = model.ps_enhancer.register_forward_hook(capture("ps"))
    fh_hook = model.fh_enhancer.register_forward_hook(capture("fh"))
    with torch.inference_mode():
        output = model(inputs)
    ps_hook.remove()
    fh_hook.remove()

    assert observed == {
        "ps_input": (1, 32, 128, 128),
        "ps_output": (1, 32, 128, 128),
        "fh_input": (1, 32, 128, 128),
        "fh_output": (1, 32, 128, 128),
    }
    assert output.shape == (1, 3, 256, 256)
    assert torch.isfinite(output).all()

    with torch.no_grad():
        model.ps_decoder.output.weight.zero_()
        model.ps_decoder.output.bias.copy_(torch.tensor((1.0, 2.0)))
        model.fh_decoder.output.weight.zero_()
        model.fh_decoder.output.bias.fill_(3.0)
        ordered = model(torch.zeros_like(inputs))
    torch.testing.assert_close(ordered[:, 0], torch.ones_like(ordered[:, 0]))
    torch.testing.assert_close(ordered[:, 1], torch.full_like(ordered[:, 1], 2.0))
    torch.testing.assert_close(ordered[:, 2], torch.full_like(ordered[:, 2], 3.0))


def test_strict_model_rejects_non_grayscale_or_non_512_inputs() -> None:
    model = _synthetic_strict_model().eval()
    with pytest.raises(ValueError, match="grayscale"):
        model(torch.zeros((1, 3, 512, 512), dtype=torch.float32))
    with pytest.raises(ValueError, match="512x512"):
        model(torch.zeros((1, 1, 256, 256), dtype=torch.float32))


def test_strict_model_ps_and_fh_decoders_have_independent_parameter_storage() -> None:
    model = _synthetic_strict_model()
    ps_pointers = {
        parameter.untyped_storage().data_ptr() for parameter in model.ps_decoder.parameters()
    }
    fh_pointers = {
        parameter.untyped_storage().data_ptr() for parameter in model.fh_decoder.parameters()
    }
    assert ps_pointers
    assert fh_pointers
    assert ps_pointers.isdisjoint(fh_pointers)
