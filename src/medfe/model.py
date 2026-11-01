"""Mutual encoder-decoder with corrected MEDFE feature equalization.

The default architecture retains the six-level 64-channel encoder/decoder,
four dilated residual blocks, and shared 3/5/7-kernel streams with five partial
convolutions each. This implementation uses explicit outputs instead of keeping
loss tensors/targets as mutable module state.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    FeatureEqualization,
    InstanceNorm2d,
    PartialConvStream,
    StableReflectionPad2d,
    downsample_validity,
)
from .sd_backbone import (
    DecoderResidualBlock,
    PartialBottleneck,
    PartialEncoderStage,
    PartialResidualBlock,
    ResizeDecoderStage,
)


class MEDFEOutput(NamedTuple):
    masked: Tensor
    prediction: Tensor
    completed: Tensor
    structure: Tensor
    texture: Tensor


class EncoderBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, first: bool = False, last: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        if not first:
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        layers.append(nn.Conv2d(inputs, outputs, 4, stride=2, padding=1))
        if not first and not last:
            layers.append(InstanceNorm2d(outputs, affine=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            StableReflectionPad2d(2),
            nn.Conv2d(channels, channels, 3, dilation=2, bias=False),
            InstanceNorm2d(channels),
            nn.ReLU(inplace=False),
            StableReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            InstanceNorm2d(channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.layers(value)


class DecoderBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, last: bool = False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(inputs, outputs, 4, stride=2, padding=1),
            nn.Tanh() if last else InstanceNorm2d(outputs, affine=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def _project(
    inputs: int,
    outputs: int,
    kernel: int = 1,
    stride: int = 1,
    padding: int = 0,
    activation: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=padding, bias=False),
        InstanceNorm2d(outputs),
    ]
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=False))
    return nn.Sequential(*layers)


def _resize_bilinear(value: Tensor, size: tuple[int, int]) -> Tensor:
    """Bound interpolation temporaries without changing per-image resizing.

    Deterministic CUDA bilinear interpolation can allocate several FP32
    corner tensors. Resizing independent groups of at most eight images
    bounds those temporaries while retaining the native operation and its
    deterministic backward, including for BF16 inputs.
    """
    if value.shape[0] <= 8:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)
    return torch.cat(
        [
            F.interpolate(chunk, size=size, mode="bilinear", align_corners=False)
            for chunk in value.split(8, dim=0)
        ],
        dim=0,
    )


class MutualFeatureBlock(nn.Module):
    def __init__(self, width: int, pconv_layers: int, gaussian_sigma: float) -> None:
        super().__init__()
        channels = width * 4
        self.texture_align = nn.ModuleList(
            [
                nn.Sequential(
                    _project(width, 2 * width, 4, 2, 1), _project(2 * width, channels, 4, 2, 1)
                ),
                _project(2 * width, channels, 4, 2, 1),
                _project(channels, channels),
            ]
        )
        self.structure_align = _project(8 * width, channels)
        # Both branches share alignment reduction, the three streams, and fusion.
        self.reduce = _project(3 * channels, channels)
        self.streams = nn.ModuleList(
            [PartialConvStream(channels, kernel, pconv_layers) for kernel in (3, 5, 7)]
        )
        self.fuse = _project(2 * channels, 2 * channels)
        self.equalization = FeatureEqualization(2 * channels, gaussian_sigma)
        self.texture_skip = nn.ModuleList(
            [
                _project(2 * channels, width),
                _project(2 * channels, 2 * width),
                _project(2 * channels, channels),
            ]
        )
        self.structure_skip = nn.ModuleList(
            [
                _project(8 * width, 8 * width, 4, 2, 1, activation=False),
                nn.Sequential(
                    _project(8 * width, 8 * width, 4, 2, 1),
                    _project(8 * width, 8 * width, 4, 2, 1, activation=False),
                ),
                nn.Sequential(
                    _project(8 * width, 8 * width, 4, 2, 1),
                    _project(8 * width, 8 * width, 4, 2, 1),
                    _project(8 * width, 8 * width, 4, 2, 1, activation=False),
                ),
            ]
        )
        # Shared projection preserves the original shared auxiliary decoder.
        self.auxiliary = nn.Sequential(nn.Conv2d(channels, 3, 1), nn.Tanh())

    def _fill(self, value: Tensor, known: Tensor) -> Tensor:
        return self.reduce(torch.cat([stream(value, known)[0] for stream in self.streams], 1))

    def forward(self, features: list[Tensor], known: Tensor) -> tuple[list[Tensor], Tensor, Tensor]:
        # The published PCconv applies this activation in addition to the
        # encoder's saved activation at levels 1--5.
        active = [F.leaky_relu(feature, 0.2) for feature in features]
        size = features[2].shape[-2:]
        texture = self.reduce(
            torch.cat([layer(value) for layer, value in zip(self.texture_align, active[:3])], dim=1)
        )
        structure = self.reduce(
            torch.cat(
                [self.structure_align(_resize_bilinear(value, size)) for value in active[3:]],
                dim=1,
            )
        )
        texture = self._fill(texture, known)
        structure = self._fill(structure, known)
        equalized = self.equalization(self.fuse(torch.cat((structure, texture), dim=1)))
        refined = [
            layer(_resize_bilinear(equalized, features[index].shape[-2:])) + features[index]
            for index, layer in enumerate(self.texture_skip)
        ]
        refined.extend(
            layer(equalized) + features[index + 3]
            for index, layer in enumerate(self.structure_skip)
        )
        return refined, self.auxiliary(structure), self.auxiliary(texture)


class MEDFEGenerator(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        residual_blocks: int = 4,
        pconv_layers: int = 5,
        gaussian_sigma: float = 1.5,
        init_gain: float = 0.02,
        activation_checkpointing: bool = False,
        backbone: str = "released",
        blocks_per_scale: int = 2,
    ) -> None:
        super().__init__()
        if (
            not isinstance(base_channels, int)
            or isinstance(base_channels, bool)
            or base_channels < 1
            or not isinstance(residual_blocks, int)
            or isinstance(residual_blocks, bool)
            or residual_blocks < 1
            or not isinstance(pconv_layers, int)
            or isinstance(pconv_layers, bool)
            or pconv_layers < 1
            or type(blocks_per_scale) is not int
            or blocks_per_scale < 1
        ):
            raise ValueError("channel width and block counts must be positive integers")
        if (
            not math.isfinite(gaussian_sigma)
            or gaussian_sigma <= 0
            or not math.isfinite(init_gain)
            or init_gain <= 0
        ):
            raise ValueError("Gaussian sigma and initialization gain must be finite and positive")
        if not isinstance(activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be a boolean")
        if backbone not in ("released", "sd_partial"):
            raise ValueError("backbone must be released or sd_partial")
        self.backbone = backbone
        self.activation_checkpointing = activation_checkpointing
        width = base_channels
        if backbone == "released":
            self.encoder = nn.ModuleList(
                [
                    EncoderBlock(6, width, first=True),
                    EncoderBlock(width, 2 * width),
                    EncoderBlock(2 * width, 4 * width),
                    EncoderBlock(4 * width, 8 * width),
                    EncoderBlock(8 * width, 8 * width),
                    EncoderBlock(8 * width, 8 * width, last=True),
                ]
            )
            self.residual = nn.Sequential(
                *[ResidualBlock(8 * width) for _ in range(residual_blocks)]
            )
            self.mutual = MutualFeatureBlock(width, pconv_layers, gaussian_sigma)
            self.decoder = nn.ModuleList(
                [
                    DecoderBlock(8 * width, 8 * width),
                    DecoderBlock(16 * width, 8 * width),
                    DecoderBlock(16 * width, 4 * width),
                    DecoderBlock(8 * width, 2 * width),
                    DecoderBlock(4 * width, width),
                    DecoderBlock(2 * width, 3, last=True),
                ]
            )
        else:
            channels = [6, width, 2 * width, 4 * width, 8 * width, 8 * width, 8 * width]
            self.encoder = nn.ModuleList(
                PartialEncoderStage(inputs, outputs, blocks=blocks_per_scale)
                for inputs, outputs in zip(channels[:-1], channels[1:], strict=True)
            )
            self.residual = PartialBottleneck(8 * width, residual_blocks)
            self.mutual = MutualFeatureBlock(width, pconv_layers, gaussian_sigma)
            self.decoder = nn.ModuleList(
                ResizeDecoderStage(inputs, outputs, blocks=blocks_per_scale, last=index == 5)
                for index, (inputs, outputs) in enumerate(
                    [
                        (8 * width, 8 * width),
                        (16 * width, 8 * width),
                        (16 * width, 4 * width),
                        (8 * width, 2 * width),
                        (4 * width, width),
                        (2 * width, 3),
                    ]
                )
            )
        self.register_buffer(
            "fill", (torch.tensor([123.0, 104.0, 117.0]) / 127.5 - 1).view(1, 3, 1, 1)
        )
        self.apply(lambda module: self._initialize(module, init_gain))
        if backbone == "sd_partial":
            # Keep SD-style residual branches initially zero without changing
            # the initialization or state-dict keys of the released generator.
            for module in self.modules():
                if isinstance(module, (PartialResidualBlock, DecoderResidualBlock)):
                    module.reset_residual_output()

    @staticmethod
    def _initialize(module: nn.Module, gain: float) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, 0, gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.InstanceNorm2d, nn.GroupNorm)) and module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _encode(self, value: Tensor) -> list[Tensor]:
        """Keep the published encoder's saved features without in-place aliases.

        In the original encoder, the next block's in-place LeakyReLU changes
        each saved feature from levels 1--5. Store those activated values
        explicitly, while the main chain keeps passing the unactivated output
        to the next block (which applies its own input activation). The sixth
        level enters the residual stack without that additional activation.
        """
        features: list[Tensor] = []
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            features.append(
                F.leaky_relu(value, 0.2) if index + 1 < len(self.encoder) else self.residual(value)
            )
        return features

    def _run_stage(self, layer: nn.Module, *values: Tensor):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(layer, *values, use_reentrant=False, preserve_rng_state=False)
        return layer(*values)

    def _encode_partial(self, value: Tensor, known: Tensor) -> tuple[list[Tensor], Tensor]:
        features, masks = [], []
        for layer in self.encoder:
            value, known = self._run_stage(layer, value, known)
            features.append(value)
            masks.append(known)
        value, _ = self._run_stage(self.residual, value, known)
        features[-1] = value
        # This is the propagated encoder mask at the mutual block's 1/8 scale,
        # not the old 4x4/stride-2 approximation. Partial residual convolutions
        # expand validity as well as the downsampling convolutions.
        return features, masks[2]

    def forward(self, image: Tensor, mask: Tensor) -> MEDFEOutput:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("image must be a floating-point Bx3xHxW tensor in [-1, 1]")
        if (
            mask.ndim != 4
            or mask.shape[1] != 1
            or mask.dtype != torch.bool
            or mask.shape[0] != image.shape[0]
            or mask.shape[2:] != image.shape[2:]
            or mask.device != image.device
        ):
            raise ValueError("mask must be boolean Bx1xHxW on the image device; True means hole")
        if image.shape[0] < 1 or any(size < 256 or size % 64 for size in image.shape[-2:]):
            raise ValueError("image height and width must each be at least 256 and divisible by 64")
        masked = torch.where(mask, self.fill.to(image.dtype), image)
        known = ~mask
        value = torch.cat((masked, known.to(image.dtype).expand(-1, 3, -1, -1)), dim=1)
        if self.backbone == "sd_partial":
            features, validity = self._encode_partial(value, known)
        else:
            features = self._encode(value)
            validity = downsample_validity(known, 3)
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            # Non-reentrant replay supports the nested feature lists and both
            # auxiliary outputs. It also restores the forward autocast state.
            # MutualFeatureBlock has no random operations or mutable statistics.
            features, structure, texture = checkpoint(
                self.mutual,
                features,
                validity,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            features, structure, texture = self.mutual(features, validity)
        run_decoder = self._run_stage if self.backbone == "sd_partial" else lambda f, x: f(x)
        value = run_decoder(self.decoder[0], features[-1])
        for index, layer in enumerate(self.decoder[1:]):
            value = run_decoder(layer, torch.cat((value, features[-2 - index]), dim=1))
        prediction = value
        completed = torch.where(mask, prediction.to(image.dtype), image)
        return MEDFEOutput(masked, prediction, completed, structure, texture)
