"""Mask, precision and gradient contracts for the SD-inspired RGB backbone."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from medfe.blocks import PartialConv2d
from medfe.model import MEDFEGenerator
from medfe.sd_backbone import PartialEncoderStage, PartialResidualBlock, StableGroupNorm


def _model(**kwargs):
    return MEDFEGenerator(
        backbone="sd_partial",
        base_channels=2,
        residual_blocks=1,
        pconv_layers=1,
        blocks_per_scale=1,
        **kwargs,
    )


def _inputs(batch=2, width=256):
    generator = torch.Generator().manual_seed(853)
    image = torch.rand(batch, 3, 256, width, generator=generator) * 2 - 1
    mask = torch.zeros(batch, 1, 256, width, dtype=torch.bool)
    mask[0, :, 30:180, 50:210] = True
    if batch > 1:
        mask[1, :, 90:240, 20:150] = True
    return image, mask


def _expand(known, stride=1):
    # An independent receptive-field reference: any valid sample makes a
    # partial-convolution output valid; out-of-image padding is invalid.
    return F.max_pool2d(known.float(), 3, stride, 1) > 0


@pytest.mark.parametrize("kind", ["residual", "encoder"])
def test_partial_blocks_propagate_receptive_field_masks(kind):
    torch.manual_seed(851)
    known = torch.zeros(2, 1, 31, 39, dtype=torch.bool)
    known[0, :, 3:6, 8:11] = True
    known[1, :, 17:22, 27:30] = True
    value = torch.randn(2, 4, 31, 39) * known
    if kind == "residual":
        module = PartialResidualBlock(4)
        expected = _expand(_expand(known))
    else:
        module = PartialEncoderStage(4, 8, blocks=2)
        expected = _expand(known, stride=2)
        for _ in range(4):
            expected = _expand(expected)
    result, actual = module(value, known)
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)
    assert torch.equal(
        result.masked_select(~actual), torch.zeros_like(result.masked_select(~actual))
    )


@pytest.mark.parametrize("kind", ["residual", "encoder"])
def test_partial_blocks_keep_all_holes_zero_despite_learned_bias(kind):
    module = PartialResidualBlock(4) if kind == "residual" else PartialEncoderStage(4, 8)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(0.73)
    known = torch.zeros(2, 1, 17, 23, dtype=torch.bool)
    result, valid = module(torch.randn(2, 4, 17, 23), known)
    assert not valid.any()
    assert torch.count_nonzero(result) == 0


def test_partial_residual_ignores_invalid_input_values_before_normalizing():
    torch.manual_seed(852)
    block = PartialResidualBlock(8)
    known = torch.rand(2, 1, 13, 19) > 0.65
    value = torch.randn(2, 8, 13, 19)
    alternate = torch.where(known, value, torch.randn_like(value) * 100)
    expected, expected_mask = block(value, known)
    actual, actual_mask = block(alternate, known)
    assert torch.equal(expected_mask, actual_mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_group_norm_uses_float32_statistics_for_bfloat16_and_preserves_double(dtype):
    torch.manual_seed(854)
    layer = StableGroupNorm(12)
    with torch.no_grad():
        layer.weight.uniform_(0.5, 1.5)
        layer.bias.uniform_(-0.5, 0.5)
    if dtype == torch.float64:
        layer = layer.double()
    value = torch.randn(2, 12, 7, 9, dtype=dtype, requires_grad=True)
    work = value if dtype == torch.float64 else value.float()
    expected = F.group_norm(work, layer.num_groups, layer.weight, layer.bias, layer.eps).to(dtype)
    actual = layer(value)
    assert actual.dtype == dtype
    assert layer.num_groups <= 32 and 12 % layer.num_groups == 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    direction = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * direction).sum(), value, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * direction).sum(), value)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_generator_retains_six_scales_and_rgb_composition_for_nonsquare_inputs():
    torch.manual_seed(855)
    model = _model().eval()
    image, mask = _inputs(width=320)
    with torch.no_grad():
        masked = torch.where(mask, model.fill, image)
        known = ~mask
        value = torch.cat((masked, known.to(image.dtype).expand(-1, 3, -1, -1)), dim=1)
        features, validity = model._encode_partial(value, known)
        out = model(image, mask)
    assert [tuple(value.shape[1:]) for value in features] == [
        (2, 128, 160),
        (4, 64, 80),
        (8, 32, 40),
        (16, 16, 20),
        (16, 8, 10),
        (16, 4, 5),
    ]
    assert validity.shape == (2, 1, 32, 40) and validity.dtype == torch.bool
    assert out.prediction.shape == out.completed.shape == image.shape
    assert out.structure.shape == out.texture.shape == (2, 3, 32, 40)
    known_rgb = known.expand_as(image)
    assert torch.equal(out.completed[known_rgb], image[known_rgb])
    assert torch.equal(out.completed[~known_rgb], out.prediction[~known_rgb])
    assert not any(isinstance(module, nn.ConvTranspose2d) for module in model.modules())
    assert any(isinstance(module, StableGroupNorm) for module in model.encoder.modules())
    for value in out:
        assert torch.isfinite(value).all()
        assert value.min() >= -1 and value.max() <= 1


def test_hidden_rgb_never_enters_partial_encoder_or_predictions():
    torch.manual_seed(856)
    model = _model().eval()
    image, mask = _inputs()
    other = torch.where(mask, -image, image)
    known = ~mask
    mask_channels = known.to(image.dtype).expand(-1, 3, -1, -1)
    with torch.no_grad():
        first_features, first_valid = model._encode_partial(
            torch.cat((image, mask_channels), 1), known
        )
        other_features, other_valid = model._encode_partial(
            torch.cat((other, mask_channels), 1), known
        )
        first_out = model(image, mask)
        other_out = model(other, mask)
    assert torch.equal(first_valid, other_valid)
    for first, second in zip(first_features, other_features, strict=True):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    for first, second in zip(first_out, other_out, strict=True):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.parametrize("all_holes", [False, True])
def test_generator_handles_empty_and_full_masks(all_holes):
    model = _model().eval()
    image, _ = _inputs(batch=1)
    mask = torch.full((1, 1, 256, 256), all_holes, dtype=torch.bool)
    with torch.no_grad():
        out = model(image, mask)
        if all_holes:
            features, valid = model._encode_partial(torch.randn(1, 6, 256, 256), ~mask)
            assert not valid.any()
            assert all(torch.count_nonzero(value) == 0 for value in features)
    assert all(torch.isfinite(value).all() for value in out)
    assert torch.equal(out.completed, out.prediction if all_holes else image)


def test_generator_samples_do_not_share_normalization_statistics():
    torch.manual_seed(857)
    model = _model().eval()
    image, mask = _inputs()
    with torch.no_grad():
        batch = model(image, mask)
        singles = [model(image[i : i + 1], mask[i : i + 1]) for i in range(2)]
    for field in batch._fields:
        expected = torch.cat([getattr(single, field) for single in singles])
        torch.testing.assert_close(getattr(batch, field), expected, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_all_training_branches_receive_finite_gradients(precision):
    torch.manual_seed(858)
    model = _model().train()
    image, mask = _inputs()
    context = torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with context:
        out = model(image, mask)
        loss = sum(
            (value.float() - torch.randn_like(value).float()).square().mean()
            for value in (out.prediction, out.structure, out.texture)
        )
    loss.backward()
    assert all(torch.isfinite(value).all() for value in out)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    # SD-style zero-initialized residual outputs intentionally block some
    # internal gradients on the first step. Each trainable branch must still
    # receive a nonzero learning signal somewhere.
    for prefix in ("encoder", "residual", "decoder", "mutual"):
        assert (
            sum(
                p.grad.abs().sum().item()
                for name, p in model.named_parameters()
                if name.startswith(prefix)
            )
            > 0
        ), prefix
    for module in model.modules():
        if isinstance(module, PartialConv2d):
            assert torch.equal(module.count_kernel, torch.ones_like(module.count_kernel))


@pytest.mark.parametrize("precision", ["double", "bf16"])
def test_activation_replay_preserves_partial_masks_outputs_and_parameter_gradients(precision):
    torch.manual_seed(859)
    plain, replay = _model().train(), _model(activation_checkpointing=True).train()
    replay.load_state_dict(plain.state_dict(), strict=True)
    image, mask = _inputs()
    if precision == "double":
        plain, replay = plain.double(), replay.double()
        image = image.double()
    first_image = image.requires_grad_()
    second_image = image.detach().clone().requires_grad_()
    context = torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with context:
        expected = plain(first_image, mask)
        actual = replay(second_image, mask)
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        fields = ("prediction", "structure", "texture")
        directions = {field: torch.randn_like(getattr(expected, field)) for field in fields}
        expected_loss = sum(
            (getattr(expected, field).float() * directions[field].float()).mean()
            for field in fields
        )
        actual_loss = sum(
            (getattr(actual, field).float() * directions[field].float()).mean() for field in fields
        )
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(first_image.grad, second_image.grad, atol=0, rtol=0)
    for (name, left), (other_name, right) in zip(
        plain.named_parameters(), replay.named_parameters(), strict=True
    ):
        assert name == other_name
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
