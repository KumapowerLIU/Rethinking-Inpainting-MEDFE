import pytest
import torch
import torch.nn.functional as F
from torch import nn

from medfe.blocks import (
    FeatureEqualization,
    PartialConv2d,
    StableReflectionPad2d,
    downsample_validity,
    gaussian_propagation,
    local_propagation,
)


def reflection_reference(value, padding):
    # An independent gather reference, rather than another F.pad invocation.
    def indices(size):
        index = torch.arange(-padding, size + padding, device=value.device)
        index = torch.where(index < 0, -index, index)
        return torch.where(index >= size, 2 * size - 2 - index, index)

    return value.index_select(-2, indices(value.shape[-2])).index_select(
        -1, indices(value.shape[-1])
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("padding", [1, 2])
def test_stable_reflection_forward_dtype_and_reference_gradient(dtype, padding):
    torch.manual_seed(91)
    value = torch.randn(2, 3, 4, 5).to(dtype).requires_grad_()
    actual = StableReflectionPad2d(padding)(value)
    expected_forward = reflection_reference(value.detach(), padding)
    assert actual.dtype == dtype and torch.equal(actual.detach(), expected_forward)
    # Dyadic gradients keep the independent accumulation reference exact.
    direction = ((torch.arange(actual.numel()).reshape(actual.shape) % 7) - 3).to(dtype) / 8
    actual.backward(direction)
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    reference_input = value.detach().to(work_dtype).requires_grad_()
    reflection_reference(reference_input, padding).backward(direction.to(work_dtype))
    assert torch.equal(value.grad, reference_input.grad.to(dtype))
    if dtype in (torch.float32, torch.float64):
        native_input = value.detach().requires_grad_()
        native = nn.ReflectionPad2d(padding)(native_input)
        native.backward(direction)
        assert torch.equal(actual.detach(), native.detach())
        assert torch.equal(value.grad, native_input.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_stable_reflection_preserves_native_forward_and_fp32_gradient(dtype):
    value = torch.randn(2, 3, 4, 5, device="cuda", dtype=dtype, requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        actual = StableReflectionPad2d(2)(value)
        native = nn.ReflectionPad2d(2)(value.detach())
    assert actual.dtype == dtype and torch.equal(actual.detach(), native)
    direction = (torch.arange(actual.numel(), device="cuda").reshape(actual.shape) % 7) - 3
    direction = direction.to(dtype) / 8
    actual.backward(direction)
    reference = value.detach().float().requires_grad_()
    reflection_reference(reference, 2).backward(direction.float())
    assert torch.equal(value.grad, reference.grad.to(dtype))


def dense_gaussian_reference(value, sigma):
    b, c, h, w = value.shape
    coords = torch.cartesian_prod(
        torch.arange(h, device=value.device),
        torch.arange(w, device=value.device),
    ).to(value.dtype)
    distances = (coords[:, None] - coords[None, :]).square().sum(-1)
    kernel = torch.exp(-distances / (2 * sigma**2))
    kernel = kernel / kernel.sum(-1, keepdim=True)
    return torch.einsum("pq,bcq->bcp", kernel, value.flatten(2)).reshape(b, c, h, w)


def unfolded_local_reference(value):
    b, c, h, w = value.shape
    query = torch.sigmoid(value)
    keys = F.unfold(query, kernel_size=3, padding=1).reshape(b, c, 9, h * w)
    patches = F.unfold(value, kernel_size=3, padding=1).reshape(b, c, 9, h * w)
    scores = (keys * query.reshape(b, c, 1, h * w)).mean(1)
    return (patches * scores.softmax(1).unsqueeze(1)).sum(2).reshape_as(value)


def loop_partial_reference(layer, value, known):
    kernel = layer.kernel_size
    x = F.pad(value * known, (layer.padding,) * 4)
    mask = F.pad(known, (layer.padding,) * 4)
    rows, valid_rows = [], []
    for y in range(0, x.shape[-2] - kernel + 1, layer.stride):
        row, valid_row = [], []
        for z in range(0, x.shape[-1] - kernel + 1, layer.stride):
            patch = x[:, :, y : y + kernel, z : z + kernel]
            count = mask[:, :, y : y + kernel, z : z + kernel].sum((-2, -1))
            dot = torch.einsum("bcij,ocij->bo", patch, layer.conv.weight)
            result = dot * (kernel**2 / count.clamp_min(1).to(value.dtype))
            if layer.conv.bias is not None:
                result = result + layer.conv.bias
            row.append(torch.where(count > 0, result, torch.zeros_like(result)))
            valid_row.append(count > 0)
        rows.append(torch.stack(row, -1))
        valid_rows.append(torch.stack(valid_row, -1))
    return torch.stack(rows, -2), torch.stack(valid_rows, -2)


@pytest.mark.parametrize("sigma", [0.4, 1.5, 7.0])
def test_gaussian_matches_dense_forward_and_gradient(sigma):
    torch.manual_seed(21)
    value = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    actual = gaussian_propagation(value, sigma)
    expected = dense_gaussian_reference(value, sigma)
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * direction).sum(), value)[0]
    ge = torch.autograd.grad((expected * direction).sum(), value)[0]
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(ga, ge, atol=2e-14, rtol=2e-14)


def test_gaussian_keeps_constant_channels_and_batch_independent():
    value = torch.tensor([1.0, 7.0, -2.0, 13.0]).reshape(2, 2, 1, 1).expand(-1, -1, 4, 7)
    actual = gaussian_propagation(value)
    torch.testing.assert_close(actual, value)
    for index in range(2):
        torch.testing.assert_close(
            actual[index : index + 1], gaussian_propagation(value[index : index + 1])
        )


def test_local_matches_unfold_forward_and_gradient():
    torch.manual_seed(22)
    value = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    actual = local_propagation(value)
    expected = unfolded_local_reference(value)
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * direction).sum(), value)[0]
    ge = torch.autograd.grad((expected * direction).sum(), value)[0]
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(ga, ge, atol=1e-14, rtol=1e-14)


def test_local_preserves_constant_channel_ratios_and_batch_independence():
    value = torch.tensor([1.0, 5.0, 2.0, 10.0]).reshape(2, 2, 1, 1).expand(-1, -1, 5, 7)
    result = local_propagation(value)
    torch.testing.assert_close(result[:, 1], result[:, 0] * 5)
    # Zero-padded boundary neighbors participate in the original local softmax.
    torch.testing.assert_close(result[:, :, 1:-1, 1:-1], value[:, :, 1:-1, 1:-1])
    for index in range(2):
        torch.testing.assert_close(
            result[index : index + 1], local_propagation(value[index : index + 1])
        )


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_partial_convolution_matches_loop_forward_and_gradients(stride, bias):
    torch.manual_seed(23)
    layer = PartialConv2d(2, 3, 3, stride=stride, padding=1, bias=bias).double()
    value = torch.randn(2, 2, 5, 6, dtype=torch.float64, requires_grad=True)
    known = torch.rand(2, 1, 5, 6) > 0.4
    actual, actual_mask = layer(value, known)
    expected, expected_mask = loop_partial_reference(layer, value, known)
    inputs = (value, *layer.parameters())
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * direction).sum(), inputs)
    ge = torch.autograd.grad((expected * direction).sum(), inputs)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    assert torch.equal(actual_mask, expected_mask)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=5e-14, rtol=5e-14)


def test_partial_convolution_all_known_interior_and_padding_normalization():
    layer = PartialConv2d(1, 1, 3, padding=1, bias=True)
    with torch.no_grad():
        layer.conv.weight.fill_(1)
        layer.conv.bias.fill_(2)
    value = torch.ones(2, 1, 5, 7)
    actual, valid = layer(value, torch.ones_like(value, dtype=torch.bool))
    torch.testing.assert_close(actual, torch.full_like(actual, 11))
    torch.testing.assert_close(actual[:, :, 1:-1, 1:-1], layer.conv(value)[:, :, 1:-1, 1:-1])
    assert valid.all()


def test_partial_convolution_all_holes_stay_zero_with_no_input_gradient():
    layer = PartialConv2d(2, 3, 3, padding=1)
    value = torch.randn(2, 2, 5, 7, requires_grad=True)
    actual, valid = layer(value, torch.zeros(2, 1, 5, 7, dtype=torch.bool))
    assert not actual.any() and not valid.any()
    actual.sum().backward()
    assert not value.grad.any()
    assert not layer.conv.weight.grad.any()


def test_count_kernel_is_buffer_and_conv_initialization_cannot_change_it():
    layer = PartialConv2d(4, 8, 5, padding=2)
    before = layer.count_kernel.clone()
    for module in layer.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0, 0.02)
    assert torch.equal(before, layer.count_kernel)
    assert "count_kernel" in dict(layer.named_buffers())
    assert "count_kernel" not in dict(layer.named_parameters())


def test_mask_propagation_all_holes_and_different_batch_masks():
    known = torch.zeros(2, 1, 256, 320, dtype=torch.bool)
    known[1, :, 120, 160] = True
    value = downsample_validity(known)
    assert value.shape == (2, 1, 32, 40)
    assert not value[0].any() and value[1].any()
    # Independent arithmetic reference: a count greater than zero at each level.
    expected = known.float()
    for _ in range(3):
        expected = (F.conv2d(expected, torch.ones(1, 1, 4, 4), stride=2, padding=1) > 0).float()
    assert torch.equal(value, expected.bool())
    assert downsample_validity(torch.ones_like(known)).all()


@pytest.mark.parametrize("sigma", [0, -1, float("inf"), float("nan")])
def test_invalid_gaussian_sigma(sigma):
    with pytest.raises(ValueError):
        gaussian_propagation(torch.ones(1, 2, 3, 4), sigma)


def test_feature_equalization_has_finite_parameter_and_input_gradients():
    torch.manual_seed(24)
    block = FeatureEqualization(8)
    value = torch.randn(2, 8, 5, 7, requires_grad=True)
    result = block(value)
    (result * torch.randn_like(result)).sum().backward()
    assert torch.isfinite(result).all() and torch.isfinite(value.grad).all()
    assert value.grad.abs().sum() > 0
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_operators_are_finite_and_keep_counting_buffers():
    block = FeatureEqualization(16).cuda()
    partial = PartialConv2d(16, 16, 3, padding=1, bias=False).cuda()
    x = torch.randn(2, 16, 8, 12, device="cuda", requires_grad=True)
    known = torch.rand(2, 1, 8, 12, device="cuda") > 0.5
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, _ = partial(x, known)
        y = block(y)
    y.float().square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert torch.equal(partial.count_kernel, torch.ones_like(partial.count_kernel))
