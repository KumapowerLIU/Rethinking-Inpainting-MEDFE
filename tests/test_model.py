import pytest
import torch

from medfe.blocks import PartialConv2d
from medfe.model import MEDFEGenerator, MEDFEOutput


@pytest.fixture
def model():
    torch.manual_seed(31)
    return MEDFEGenerator(base_channels=2, residual_blocks=1, pconv_layers=1)


def inputs(batch=2, width=256):
    torch.manual_seed(32)
    image = torch.rand(batch, 3, 256, width) * 2 - 1
    mask = torch.zeros(batch, 1, 256, width, dtype=torch.bool)
    mask[0, :, 60:180, 70:190] = True
    if batch > 1:
        mask[1, :, 90:220, 25:120] = True
    return image, mask


def test_generator_shapes_composition_and_no_input_mutation(model):
    image, mask = inputs(width=320)
    original_image, original_mask = image.clone(), mask.clone()
    out = model(image, mask)
    assert isinstance(out, MEDFEOutput)
    assert out.prediction.shape == out.completed.shape == out.masked.shape == image.shape
    assert out.structure.shape == out.texture.shape == (2, 3, 32, 40)
    assert torch.equal(image, original_image) and torch.equal(mask, original_mask)
    valid_rgb = (~mask).expand_as(image)
    assert torch.equal(out.completed[valid_rgb], image[valid_rgb])
    assert torch.equal(out.masked[valid_rgb], image[valid_rgb])
    assert torch.equal(out.completed[~valid_rgb], out.prediction[~valid_rgb])
    for value in out:
        assert torch.isfinite(value).all()
        assert value.min() >= -1 and value.max() <= 1


def test_generator_batch_independence_with_different_masks(model):
    model.eval()
    image, mask = inputs()
    with torch.inference_mode():
        batch = model(image, mask)
        single = [model(image[index : index + 1], mask[index : index + 1]) for index in range(2)]
    for field in MEDFEOutput._fields:
        expected = torch.cat([getattr(one, field) for one in single])
        torch.testing.assert_close(getattr(batch, field), expected, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("all_holes", [False, True])
def test_all_holes_and_no_holes_are_finite(model, all_holes):
    image, _ = inputs(batch=1)
    mask = torch.full((1, 1, 256, 256), all_holes, dtype=torch.bool)
    out = model(image, mask)
    for value in out:
        assert torch.isfinite(value).all()
    if all_holes:
        assert torch.equal(out.completed, out.prediction)
    else:
        assert torch.equal(out.completed, image)


def test_generator_all_branches_get_gradients_and_parameters_stay_finite(model):
    image, mask = inputs()
    out = model(image, mask)
    target = torch.randn_like(out.prediction)
    # Independent nonconstant supervision prevents zero gradients from a sum of IN outputs.
    loss = (out.prediction - target).square().mean()
    loss = loss + (out.structure - torch.randn_like(out.structure)).abs().mean()
    loss = loss + (out.texture - torch.randn_like(out.texture)).abs().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for prefix in [
        "encoder",
        "residual",
        "decoder",
        "mutual.texture_align",
        "mutual.structure_align",
        "mutual.streams.0",
        "mutual.streams.1",
        "mutual.streams.2",
        "mutual.equalization",
        "mutual.auxiliary",
    ]:
        total = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        )
        assert total > 0, prefix
    for module in model.modules():
        if isinstance(module, PartialConv2d):
            assert torch.equal(module.count_kernel, torch.ones_like(module.count_kernel))


def test_auxiliary_heads_are_differentiable_without_forward_state(model):
    image, mask = inputs(batch=1)
    out = model(image, mask)
    aux = out.structure.square().mean() + out.texture.square().mean()
    aux.backward()
    assert model.mutual.auxiliary[0].weight.grad.abs().sum() > 0
    assert model.encoder[0].layers[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert not any(
        hasattr(module, "loss") or hasattr(module, "target") for module in model.modules()
    )


@pytest.mark.parametrize("shape", [(1, 3, 128, 256), (1, 3, 256, 257), (1, 1, 256, 256)])
def test_invalid_image_geometry(model, shape):
    image = torch.zeros(shape)
    mask = torch.zeros(shape[0], 1, shape[2], shape[3], dtype=torch.bool)
    with pytest.raises(ValueError):
        model(image, mask)


def test_reject_nonboolean_or_wrong_batch_masks(model):
    image, mask = inputs()
    with pytest.raises(ValueError):
        model(image, mask.float())
    with pytest.raises(ValueError):
        model(image, mask[:1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_channels": 0},
        {"pconv_layers": 0},
        {"residual_blocks": 0},
        {"gaussian_sigma": float("nan")},
        {"init_gain": 0},
    ],
)
def test_invalid_architecture_arguments(kwargs):
    with pytest.raises(ValueError):
        MEDFEGenerator(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_full_generator_forward_backward_is_finite():
    model = MEDFEGenerator(base_channels=4, residual_blocks=1, pconv_layers=1).cuda()
    image, mask = inputs()
    image, mask = image.cuda(), mask.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(image, mask)
        loss = (
            (out.completed - image).abs().mean()
            + out.structure.abs().mean()
            + out.texture.abs().mean()
        )
    loss.backward()
    for value in out:
        assert torch.isfinite(value).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert torch.equal(out.completed[(~mask).expand_as(image)], image[(~mask).expand_as(image)])
