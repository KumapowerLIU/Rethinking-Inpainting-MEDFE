"""Batch-safe MEDFE feature propagation and partial-convolution primitives.

These operators implement corrected paper semantics. They intentionally do not
reproduce the legacy mask threshold, randomized counting kernels, or the
position/channel reshape errors of the original research implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _statistics_dtype(value: Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


class StableReflectionPad2d(nn.ReflectionPad2d):
    """Preserve reflection values while accumulating AMP backward in FP32.

    Reflection is a value-copy operation, so promoting FP16/BF16 inputs to
    FP32, padding, then casting back leaves the forward tensor bitwise intact.
    The backward scatter instead accumulates repeated-position gradients in
    FP32 before a single cast to the input dtype. FP32/FP64 use the native
    operation unchanged. CUDA's FP32 scatter may still be nondeterministic;
    this improves accumulation precision, not a determinism guarantee.
    """

    def forward(self, value: Tensor) -> Tensor:
        if value.dtype not in (torch.float16, torch.bfloat16):
            return super().forward(value)
        with torch.autocast(device_type=value.device.type, enabled=False):
            result = F.pad(value.float(), self.padding, mode="reflect")
        return result.to(value.dtype)


class InstanceNorm2d(nn.InstanceNorm2d):
    """Instance normalization with FP32 statistics during mixed precision."""

    def forward(self, value: Tensor) -> Tensor:
        if value.dtype not in (torch.float16, torch.bfloat16):
            return super().forward(value)
        with torch.autocast(device_type=value.device.type, enabled=False):
            result = F.instance_norm(
                value.float(),
                None,
                None,
                None if self.weight is None else self.weight.float(),
                None if self.bias is None else self.bias.float(),
                use_input_stats=True,
                momentum=self.momentum,
                eps=self.eps,
            )
        return result.to(value.dtype)


def downsample_validity(known: Tensor, levels: int = 3) -> Tensor:
    """Propagate *known* pixels through encoder 4x4/stride-2 receptive fields.

    A feature is valid when its receptive field contains any known input.
    Padding contributes no known pixels. All-hole inputs therefore remain
    invalid. This replaces the legacy ``mean(mask) > 1`` operation, which
    discarded every hole and is not retained for checkpoint compatibility.
    """
    if known.ndim != 4 or known.shape[1] != 1 or known.dtype != torch.bool:
        raise ValueError("known must be a boolean Bx1xHxW tensor")
    if not isinstance(levels, int) or levels < 0:
        raise ValueError("levels must be a nonnegative integer")
    result = known
    for _ in range(levels):
        result = F.max_pool2d(result.float(), kernel_size=4, stride=2, padding=1) > 0
    return result


class PartialConv2d(nn.Module):
    """Partial convolution with one spatial validity mask per image.

    Normalization is ``kernel_area / valid_count``. The all-valid interior
    equals ordinary convolution; boundary pixels are normalized by the number
    of valid in-image samples. The fixed count kernel is a buffer, never a
    trainable/reinitialized convolution. Invalid outputs and their new masks
    are zero, including the all-hole case.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or stride < 1 or padding < 0:
            raise ValueError("invalid partial-convolution geometry")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer("count_kernel", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, value: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        if (
            value.ndim != 4
            or known.ndim != 4
            or known.shape[1] != 1
            or known.shape[0] != value.shape[0]
            or known.shape[2:] != value.shape[2:]
            or known.dtype != torch.bool
            or known.device != value.device
        ):
            raise ValueError("known must be boolean Bx1xHxW on the feature device")
        raw = self.conv(value * known.to(value.dtype))
        with torch.autocast(device_type=value.device.type, enabled=False):
            counts = F.conv2d(
                known.float(),
                self.count_kernel.float(),
                stride=self.stride,
                padding=self.padding,
            )
            next_known = counts > 0
            work = raw.to(_statistics_dtype(raw))
            scale = self.kernel_size**2 / counts.clamp_min(1).to(work.dtype)
            if self.conv.bias is None:
                normalized = work * scale
            else:
                bias = self.conv.bias.to(work.dtype).view(1, -1, 1, 1)
                normalized = (work - bias) * scale + bias
            result = torch.where(next_known, normalized, torch.zeros_like(normalized))
        return result.to(raw.dtype), next_known


class PartialConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.partial = PartialConv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = InstanceNorm2d(channels, affine=True)

    def forward(self, value: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        value, known = self.partial(value, known)
        return F.leaky_relu(self.norm(value), negative_slope=0.2), known


class PartialConvStream(nn.Module):
    def __init__(self, channels: int, kernel_size: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [PartialConvBlock(channels, kernel_size) for _ in range(layers)]
        )

    def forward(self, value: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            value, known = block(value, known)
        return value, known


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        pooled = value.mean(dim=(-2, -1), keepdim=True)
        gate = torch.sigmoid(self.expand(F.relu(self.reduce(pooled))))
        return value * gate


def gaussian_propagation(value: Tensor, sigma: float = 1.5) -> Tensor:
    """Full-support Gaussian spatial propagation, normalized at every position.

    Two separable matrix multiplies implement the same finite-image Gaussian
    weights as the dense H*W by H*W reference, including boundary normalization.
    No B*C*(H*W)^2 broadcast product or truncated filter is constructed.
    """
    if value.ndim != 4 or not value.is_floating_point():
        raise ValueError("value must be a floating-point BxCxHxW tensor")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    with torch.autocast(device_type=value.device.type, enabled=False):
        work = value.to(_statistics_dtype(value))
        y = torch.arange(value.shape[-2], device=value.device, dtype=work.dtype)
        x = torch.arange(value.shape[-1], device=value.device, dtype=work.dtype)
        gy = torch.exp(-(y[:, None] - y[None, :]).square() / (2 * sigma**2))
        gx = torch.exp(-(x[:, None] - x[None, :]).square() / (2 * sigma**2))
        gy = gy / gy.sum(dim=-1, keepdim=True)
        gx = gx / gx.sum(dim=-1, keepdim=True)
        result = torch.matmul(torch.matmul(gy, work), gx.transpose(0, 1))
    return result.to(value.dtype)


def local_propagation(value: Tensor) -> Tensor:
    """Original 3x3 BPA similarity: mean(sigmoid(q)*sigmoid(k)), softmax over 9.

    Values are the untransformed features. Zero padding matches the original
    neighborhood definition. Fixed nine-offset loops avoid materializing two
    B*C*9*H*W unfolded tensors. Each batch and channel retains its own identity.
    """
    if value.ndim != 4 or not value.is_floating_point():
        raise ValueError("value must be a floating-point BxCxHxW tensor")
    with torch.autocast(device_type=value.device.type, enabled=False):
        work = value.to(_statistics_dtype(value))
        query = torch.sigmoid(work)
        keys = F.pad(query, (1, 1, 1, 1))
        values = F.pad(work, (1, 1, 1, 1))
        height, width = work.shape[-2:]
        similarities = [
            (query * keys[:, :, dy : dy + height, dx : dx + width]).mean(dim=1)
            for dy in range(3)
            for dx in range(3)
        ]
        weights = torch.stack(similarities, dim=1).softmax(dim=1)
        result = torch.zeros_like(work)
        for index, (dy, dx) in enumerate((y, x) for y in range(3) for x in range(3)):
            result = result + (
                values[:, :, dy : dy + height, dx : dx + width] * weights[:, index : index + 1]
            )
    return result.to(value.dtype)


class FeatureEqualization(nn.Module):
    def __init__(self, channels: int, gaussian_sigma: float = 1.5) -> None:
        super().__init__()
        self.channel_attention = SqueezeExcitation(channels)
        self.gaussian_sigma = gaussian_sigma
        self.project = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = self.channel_attention(value)
        return self.project(
            torch.cat(
                (
                    gaussian_propagation(value, self.gaussian_sigma),
                    local_propagation(value),
                ),
                dim=1,
            )
        )
