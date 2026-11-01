"""Compare saved features and autograd with the published encoder itself."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from medfe.model import MEDFEGenerator


@pytest.fixture(scope="module")
def published_encoder():
    path = Path(__file__).resolve().parents[1] / "legacy/models/Encoder.py"
    spec = importlib.util.spec_from_file_location("medfe_published_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Encoder


@pytest.mark.parametrize("batch,width", [(1, 256), (2, 320)])
def test_saved_encoder_features_and_gradients_match_published_encoder(
    published_encoder, batch, width
):
    torch.manual_seed(93)
    reference = published_encoder(6, 3, ngf=2, res_num=1, norm_layer=nn.InstanceNorm2d).double()
    model = MEDFEGenerator(base_channels=2, residual_blocks=1, pconv_layers=1).double()
    parameter_pairs = []
    for index, block in enumerate(model.encoder):
        original = getattr(reference, f"Encoder_{index + 1}").model
        block.layers.load_state_dict(original.state_dict())
        parameter_pairs.extend(zip(block.layers.parameters(), original.parameters(), strict=True))
    model.residual[0].layers.load_state_dict(reference.middle[0].conv_block.state_dict())
    parameter_pairs.extend(
        zip(model.residual.parameters(), reference.middle.parameters(), strict=True)
    )

    value = torch.randn(batch, 6, 256, width, dtype=torch.float64, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_()
    unchanged = value.detach().clone()
    raw_features = []
    handles = [
        block.register_forward_hook(
            lambda _module, _inputs, output: raw_features.append((output, output.detach().clone()))
        )
        for block in model.encoder
    ]
    try:
        actual = model._encode(value)
    finally:
        for handle in handles:
            handle.remove()
    expected = reference(reference_value)
    assert torch.equal(value.detach(), unchanged)
    for raw, snapshot in raw_features:
        assert torch.equal(raw.detach(), snapshot)

    for index, (current, original) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(current, original, atol=2e-12, rtol=2e-12)
        if index < 5:
            # This regression must distinguish activated skips from raw skips.
            assert (raw_features[index][1] < 0).any()
            assert not torch.equal(current.detach(), raw_features[index][1])

    # Every saved level participates, as it does through the mutual block and
    # decoder skips. Nonconstant directions avoid zero-sum InstanceNorm probes.
    directions = [torch.randn_like(feature) for feature in actual]
    actual_loss = sum(
        (feature * direction).mean() for feature, direction in zip(actual, directions)
    )
    expected_loss = sum(
        (feature * direction).mean() for feature, direction in zip(expected, directions)
    )
    actual_parameters, reference_parameters = zip(*parameter_pairs, strict=True)
    actual_gradients = torch.autograd.grad(actual_loss, (value, *actual_parameters))
    expected_gradients = torch.autograd.grad(
        expected_loss, (reference_value, *reference_parameters)
    )
    for current, original in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(current).all()
        torch.testing.assert_close(current, original, atol=2e-11, rtol=2e-10)
