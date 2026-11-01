"""SD-inspired residual building blocks for single-pass RGB inpainting.

These modules borrow residual blocks, GroupNorm, SiLU and resize-convolution
from the Stable Diffusion U-Net. They do not implement diffusion timesteps,
text conditioning, attention, or pretrained SD weight loading. Encoder masks
are boolean tensors with True for valid features, the inverse of MEDFE's
public hole mask.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import PartialConv2d


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class StableGroupNorm(nn.GroupNorm):
    """GroupNorm with at most 32 groups and FP32 mixed-precision statistics.

    The largest divisor of the channel count no greater than 32 is used, so
    narrow diagnostic models follow the same rule as full-width models.
    This is ordinary GroupNorm: zero-valued invalid positions participate in
    its statistics. The subsequent partial convolution masks its input again.
    FP64 inputs retain FP64 statistics for numerical reference checks.
    """

    def __init__(self, channels: int, *, eps: float = 1e-5, affine: bool = True) -> None:
        _positive_integer(channels, "channels")
        groups = next(group for group in range(min(32, channels), 0, -1) if channels % group == 0)
        super().__init__(groups, channels, eps=eps, affine=affine)

    def forward(self, value: Tensor) -> Tensor:
        if value.dtype not in (torch.float16, torch.bfloat16):
            return super().forward(value)
        with torch.autocast(device_type=value.device.type, enabled=False):
            result = F.group_norm(
                value.float(),
                self.num_groups,
                None if self.weight is None else self.weight.float(),
                None if self.bias is None else self.bias.float(),
                self.eps,
            )
        return result.to(value.dtype)


class PartialResidualBlock(nn.Module):
    """Pre-normalized residual block with two mask-updating partial convolutions.

    The identity branch is valid only on the incoming mask. The main branch
    expands that mask through both 3x3 receptive fields, and their union is
    returned. Invalid input values are removed before normalization; invalid
    output positions remain exactly zero even with learned affine biases.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        _positive_integer(channels, "channels")
        self.norm1 = StableGroupNorm(channels)
        self.conv1 = PartialConv2d(channels, channels, 3, padding=1)
        self.norm2 = StableGroupNorm(channels)
        self.conv2 = PartialConv2d(channels, channels, 3, padding=1)

    def reset_residual_output(self) -> None:
        """Zero the residual output after the enclosing model initializes weights."""
        nn.init.zeros_(self.conv2.conv.weight)
        if self.conv2.conv.bias is not None:
            nn.init.zeros_(self.conv2.conv.bias)

    def forward(self, value: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        identity = torch.where(known, value, torch.zeros_like(value))
        hidden, hidden_known = self.conv1(F.silu(self.norm1(identity)), known)
        hidden, hidden_known = self.conv2(F.silu(self.norm2(hidden)), hidden_known)
        next_known = known | hidden_known
        result = torch.where(next_known, identity + hidden, torch.zeros_like(hidden))
        return result, next_known


class PartialEncoderStage(nn.Module):
    """Halve spatial resolution, then refine features while propagating validity."""

    def __init__(self, inputs: int, outputs: int, blocks: int = 2) -> None:
        super().__init__()
        _positive_integer(inputs, "inputs")
        _positive_integer(outputs, "outputs")
        _positive_integer(blocks, "blocks")
        self.downsample = PartialConv2d(inputs, outputs, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([PartialResidualBlock(outputs) for _ in range(blocks)])

    def forward(self, value: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        value, known = self.downsample(value, known)
        for block in self.blocks:
            value, known = block(value, known)
        return value, known


class PartialBottleneck(nn.Module):
    """Refine the deepest encoder features without changing their resolution."""

    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        _positive_integer(channels, "channels")
        _positive_integer(blocks, "blocks")
        self.blocks = nn.ModuleList([PartialResidualBlock(channels) for _ in range(blocks)])

    def forward(self, value: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            value, known = block(value, known)
        return value, known


class DecoderResidualBlock(nn.Module):
    """Unconditioned SD-style residual block for already-completed feature maps."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        _positive_integer(inputs, "inputs")
        _positive_integer(outputs, "outputs")
        self.norm1 = StableGroupNorm(inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm2 = StableGroupNorm(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.skip = nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)

    def reset_residual_output(self) -> None:
        """Zero the residual output after the enclosing model initializes weights."""
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(value) + hidden


class ResizeDecoderStage(nn.Module):
    """Residual refinement followed by nearest-neighbor resize and 3x3 convolution.

    The final stage keeps an intermediate feature width of half its incoming
    channels, then projects normalized features to bounded RGB values. No
    transposed convolution or mask propagation is used in the decoder.
    """

    def __init__(self, inputs: int, outputs: int, blocks: int = 2, last: bool = False) -> None:
        super().__init__()
        _positive_integer(inputs, "inputs")
        _positive_integer(outputs, "outputs")
        _positive_integer(blocks, "blocks")
        if type(last) is not bool:
            raise ValueError("last must be a boolean")
        if last and (inputs < 2 or outputs != 3):
            raise ValueError("The final decoder stage requires inputs >=2 and three RGB outputs")
        hidden = inputs // 2 if last else outputs
        self.blocks = nn.Sequential(
            DecoderResidualBlock(inputs, hidden),
            *[DecoderResidualBlock(hidden, hidden) for _ in range(blocks - 1)],
        )
        self.upsample_conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.output = (
            nn.Sequential(
                StableGroupNorm(hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, outputs, 3, padding=1),
                nn.Tanh(),
            )
            if last
            else nn.Identity()
        )

    def forward(self, value: Tensor) -> Tensor:
        value = self.blocks(value)
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        return self.output(self.upsample_conv(value))
