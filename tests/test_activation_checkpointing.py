"""Activation replay must preserve MEDFE's full and auxiliary gradient paths."""

from contextlib import nullcontext

import pytest
import torch

import medfe.model as model_module
from medfe.model import MEDFEGenerator


def _inputs():
    generator = torch.Generator().manual_seed(124)
    image = torch.rand(2, 3, 256, 256, generator=generator) * 2 - 1
    mask = torch.zeros(2, 1, 256, 256, dtype=torch.bool)
    mask[0, :, 50:175, 65:190] = True
    mask[1, :, 90:210, 10:140] = True
    return image, mask


@pytest.mark.parametrize("precision", ["double", "bf16"])
@pytest.mark.parametrize("auxiliary_only", [False, True])
def test_activation_checkpoint_preserves_outputs_and_all_gradients(precision, auxiliary_only):
    torch.manual_seed(123)
    config = {"base_channels": 2, "residual_blocks": 1, "pconv_layers": 2}
    plain = MEDFEGenerator(**config).train()
    replay = MEDFEGenerator(**config, activation_checkpointing=True).train()
    replay.load_state_dict(plain.state_dict(), strict=True)
    assert plain.state_dict().keys() == replay.state_dict().keys()
    image, mask = _inputs()
    if precision == "double":
        plain, replay = plain.double(), replay.double()
        image = image.double()
    image = image.requires_grad_()
    replay_image = image.detach().clone().requires_grad_()
    replays = []
    handle = replay.mutual.register_forward_pre_hook(lambda _module, _inputs: replays.append(True))
    context = torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with context:
        expected = plain(image, mask)
        actual = replay(replay_image, mask)
        for first, second in zip(actual, expected, strict=True):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        names = ["structure", "texture"]
        if not auxiliary_only:
            names.extend(["prediction", "completed"])
        directions = {name: torch.randn_like(getattr(expected, name)) for name in names}
        expected_loss = sum(
            (getattr(expected, name).float() * directions[name].float()).mean() for name in names
        )
        actual_loss = sum(
            (getattr(actual, name).float() * directions[name].float()).mean() for name in names
        )
    expected_loss.backward()
    actual_loss.backward()
    handle.remove()
    assert len(replays) == 2  # The mutual block runs once forward and once in backward.
    torch.testing.assert_close(replay_image.grad, image.grad, atol=0, rtol=0)
    for (name, first), (other_name, second) in zip(
        replay.named_parameters(), plain.named_parameters(), strict=True
    ):
        assert name == other_name
        assert (first.grad is None) == (second.grad is None), name
        if first.grad is not None:
            assert torch.isfinite(first.grad).all(), name
            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
    assert replay.mutual.auxiliary[0].weight.grad.abs().sum() > 0
    assert replay.encoder[0].layers[0].weight.grad.abs().sum() > 0
    if auxiliary_only:
        assert all(parameter.grad is None for parameter in replay.decoder.parameters())
    else:
        assert all(parameter.grad is not None for parameter in replay.parameters())


@pytest.mark.parametrize("mode", ["disabled", "eval", "no_grad"])
def test_activation_checkpoint_is_skipped_outside_enabled_gradient_training(monkeypatch, mode):
    model = MEDFEGenerator(
        base_channels=2,
        residual_blocks=1,
        pconv_layers=1,
        activation_checkpointing=mode != "disabled",
    )
    if mode == "eval":
        model.eval()

    def forbidden(*args, **kwargs):
        pytest.fail("Activation checkpoint must only run for enabled gradient training")

    monkeypatch.setattr(model_module, "checkpoint", forbidden)
    context = torch.no_grad() if mode == "no_grad" else nullcontext()
    with context:
        model(*_inputs())


@pytest.mark.parametrize("value", [1, "true", None])
def test_activation_checkpoint_option_rejects_nonboolean_values(value):
    with pytest.raises(ValueError, match="activation_checkpointing"):
        MEDFEGenerator(activation_checkpointing=value)
