"""Batch chunking must preserve the native bilinear operation and its gradients."""

import pytest
import torch
from torch.nn import functional as F

import medfe.model as model_module
from medfe.model import _resize_bilinear


@pytest.mark.parametrize("batch", [9, 17])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
@pytest.mark.parametrize("source,target", [((4, 5), (9, 11)), ((11, 9), (4, 5))])
def test_chunked_bilinear_matches_native_forward_and_gradient(
    monkeypatch, batch, dtype, source, target
):
    torch.manual_seed(127)
    value = torch.randn(batch, 3, *source).to(dtype).requires_grad_()
    reference = value.detach().clone().requires_grad_()
    expected = F.interpolate(reference, size=target, mode="bilinear", align_corners=False)
    native = F.interpolate
    calls = []

    def observed(chunk, *args, **kwargs):
        calls.append((chunk.shape[0], kwargs["mode"], kwargs["align_corners"]))
        return native(chunk, *args, **kwargs)

    monkeypatch.setattr(model_module.F, "interpolate", observed)
    actual = _resize_bilinear(value, target)
    assert calls == [(8, "bilinear", False)] * (batch // 8) + [(batch % 8, "bilinear", False)]
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    direction = torch.randn_like(expected)
    actual.backward(direction)
    expected.backward(direction)
    torch.testing.assert_close(value.grad, reference.grad, atol=0, rtol=0)
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_chunked_bilinear_calls_have_no_empty_chunks(monkeypatch, batch):
    native = F.interpolate
    calls = []

    def observed(chunk, *args, **kwargs):
        calls.append(chunk.shape[0])
        return native(chunk, *args, **kwargs)

    monkeypatch.setattr(model_module.F, "interpolate", observed)
    value = torch.randn(batch, 2, 3, 4)
    _resize_bilinear(value, (6, 8))
    assert calls == ([batch] if batch <= 8 else [8, 8])
