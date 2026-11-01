"""Real-ESRGAN's RGB U-Net discriminator and strict pretrained initialization.

Architecture adapted from xinntao/Real-ESRGAN, discriminator_arch.py,
Git blob 4b66ab1226d6793de846bc9828bbe427031a0e2d. The BasicSR registry
dependency is omitted; convolution names, operations and spectral-normalization
state keys are retained. Inputs use the upstream RGB [0, 1] convention.

BSD 3-Clause License
Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import hashlib
import io
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import parametrize, spectral_norm
from torch.nn.utils.parametrizations import spectral_norm as parametrized_spectral_norm
from torch.nn.utils.spectral_norm import SpectralNorm

from .weights import state_dict_sha256

REALESRGAN_D_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth"
)
REALESRGAN_D_ARCHITECTURE = "realesrgan_unet_sn_v1"
REALESRGAN_D_SHA256 = "afc291719bf0a8bd25ba2b4afbbc5fc0634382dac111d3e1d3a7240d32dfd33a"
SPECTRAL_NORM_POLICY = "fp32_once_before_each_d_and_g_phase_frozen_chunk_replay_v1"
PATCH_D_INITIALIZATION = "normal_0.02_zero_bias_v1"
PATCH_D_SPECTRAL_NORM = "torch_parametrizations_initial_power_iterations_15_v1"


def _patch_convolution(inputs: int, outputs: int) -> nn.Module:
    convolution = nn.Conv2d(inputs, outputs, 4, 2, 1)
    nn.init.normal_(convolution.weight, mean=0.0, std=0.02)
    nn.init.zeros_(convolution.bias)
    # PyTorch 2.6 estimates u/v with 15 iterations when registering SN.
    # Eval mode prevents its registration check from adding a 16th iteration.
    # Initialize the original weight first, so u/v describe that same weight.
    convolution.eval()
    parametrized_spectral_norm(convolution, n_power_iterations=1)
    return convolution.train()


class PatchDiscriminator(nn.Module):
    """Original MEDFE local discriminator: strided convolutions with SN.

    There are layers+2 convolutions, all stride two, and raw output logits.
    The legacy instance-norm choice only selected convolution biases; it did
    not insert normalization layers. This class preserves that architecture
    and the released normal(0, 0.02)/zero-bias initialization. The modern SN
    API initializes u/v with 15 power iterations; that initialization is not
    bitwise equivalent to the legacy hook's randomly initialized u/v.
    """

    def __init__(self, width: int = 64, layers: int = 3):
        super().__init__()
        if width < 1 or layers < 1:
            raise ValueError("Patch discriminator width/layers must be positive")
        self.width, self.layers = width, layers
        blocks = [_patch_convolution(3, width), nn.LeakyReLU(0.2, True)]
        previous = 1
        for index in range(1, layers + 1):
            current = min(2**index, 8)
            blocks.extend(
                [
                    _patch_convolution(width * previous, width * current),
                    nn.LeakyReLU(0.2, True),
                ]
            )
            previous = current
        blocks.append(_patch_convolution(width * previous, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, image: Tensor) -> Tensor:
        if (
            image.ndim != 4
            or image.shape[1] != 3
            or image.shape[0] < 1
            or not image.is_floating_point()
            or min(image.shape[-2:]) < 2 ** (self.layers + 2)
        ):
            raise ValueError("Patch discriminator expects RGB batches large enough for its strides")
        return self.model(image)


@torch.no_grad()
def prepare_discriminator(module: nn.Module) -> None:
    """Advance SN once, then freeze u/v during all forwards in this phase.

    Call once before D real/fake forwards, and once after the D optimizer step
    before G adversarial forwards. Keep float32 master parameters. Eval-mode
    hooks/parametrizations recompute weights without mutating SN buffers,
    making activation recomputation and splitting a batch into chunks safe.
    """
    module.eval()
    for child in module.modules():
        if isinstance(child, PatchDiscriminator):
            for convolution in child.model:
                if not isinstance(convolution, nn.Conv2d):
                    continue
                if not parametrize.is_parametrized(convolution, "weight"):
                    raise ValueError("Patch discriminator requires parametrized spectral norm")
                parametrization = convolution.parametrizations.weight
                weight = parametrization.original
                if weight.dtype != torch.float32:
                    raise ValueError(
                        "Spectral-normalized discriminators require FP32 master weights"
                    )
                parametrization.train()
                try:
                    with torch.autocast(weight.device.type, enabled=False):
                        # Calling the ParametrizationList directly also avoids
                        # a surrounding parametrize.cached() hiding this update.
                        parametrization()
                finally:
                    parametrization.eval()
        for hook in child._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm):
                weight = getattr(child, hook.name + "_orig")
                if weight.dtype != torch.float32:
                    raise ValueError(
                        "Spectral-normalized discriminators require FP32 master weights"
                    )
                with torch.autocast(weight.device.type, enabled=False):
                    hook.compute_weight(child, do_power_iteration=True)


class UNetDiscriminatorSN(nn.Module):
    """Upstream architecture, returning B x 1 x H x W raw realness logits.

    The official pretrained network has num_in_ch=3, num_feat=64 and skip
    connections enabled. Smaller widths support isolated diagnostic tests.
    Normalization to [0, 1] belongs at the training caller, not inside this
    module, so loading upstream weights preserves the original forward graph.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        if num_in_ch != 3 or num_feat < 1 or not isinstance(skip_connection, bool):
            raise ValueError("Real-ESRGAN D requires RGB, positive width and boolean skips")
        self.num_in_ch = num_in_ch
        self.num_feat = num_feat
        self.skip_connection = skip_connection
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Real-ESRGAN D expects floating B x 3 x H x W RGB images")
        if any(size < 8 or size % 8 for size in image.shape[-2:]):
            raise ValueError("Real-ESRGAN D image dimensions must be multiples of 8")
        x0 = F.leaky_relu(self.conv0(image), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        return self.conv9(out)


def discriminator_architecture(module: UNetDiscriminatorSN) -> dict:
    return {
        "architecture": REALESRGAN_D_ARCHITECTURE,
        "num_in_ch": module.num_in_ch,
        "num_feat": module.num_feat,
        "skip_connection": module.skip_connection,
        "input_range": [0, 1],
    }


def patch_discriminator_provenance(
    module: PatchDiscriminator, *, test_fixture: bool = False
) -> dict:
    """Describe the fresh Patch D contract, without hashes of evolving weights."""
    if not isinstance(module, PatchDiscriminator) or type(test_fixture) is not bool:
        raise ValueError("Patch provenance requires a PatchDiscriminator and boolean fixture flag")
    return {
        "architecture": "patch",
        "width": module.width,
        "layers": module.layers,
        "input_range": [-1, 1],
        "initialization": PATCH_D_INITIALIZATION,
        "spectral_norm": PATCH_D_SPECTRAL_NORM,
        "test_fixture": test_fixture,
    }


def validate_discriminator_provenance(
    provenance: dict, module: UNetDiscriminatorSN | PatchDiscriminator
) -> None:
    """Validate initialization metadata without comparing it to evolved weights."""
    if isinstance(module, PatchDiscriminator):
        if not isinstance(provenance, dict) or type(provenance.get("test_fixture")) is not bool:
            raise ValueError("Invalid Patch discriminator initialization provenance")
        expected = patch_discriminator_provenance(module, test_fixture=provenance["test_fixture"])
        if provenance != expected:
            raise ValueError("Invalid Patch discriminator initialization provenance")
        return
    expected = discriminator_architecture(module)
    keys = {*expected, "source", "checkpoint", "file_sha256", "initial_state_sha256", "param_key"}
    if (
        not isinstance(provenance, dict)
        or set(provenance) != keys
        or any(provenance[key] != value for key, value in expected.items())
        or provenance.get("param_key") != "params"
        or any(
            not isinstance(provenance.get(key), str) or not provenance[key]
            for key in ("source", "checkpoint")
        )
        or any(
            not isinstance(provenance.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None
            for key in ("file_sha256", "initial_state_sha256")
        )
    ):
        raise ValueError("Invalid Real-ESRGAN discriminator initialization provenance")


def load_realesrgan_weights(module: UNetDiscriminatorSN, path: str | Path) -> dict:
    """Strictly load an upstream ``params`` checkpoint and record its identity.

    A single byte snapshot is used for hashing and weights-only deserialization.
    This loader does not download assets or accept partial/mismatched tensors.
    Missing keys, shapes, dtypes or nonfinite weights fail before model mutation.
    """
    if (module.num_in_ch, module.num_feat, module.skip_connection) != (3, 64, True):
        raise ValueError("Production Real-ESRGAN D must use the official RGB width-64 architecture")
    path = Path(path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != REALESRGAN_D_SHA256:
        raise ValueError("Real-ESRGAN pretrained discriminator SHA256 mismatch")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not isinstance(state.get("params"), Mapping):
        raise ValueError("Real-ESRGAN discriminator checkpoint requires a params mapping")
    weights = state["params"]
    current = module.state_dict()
    if set(weights) != set(current):
        raise ValueError("Real-ESRGAN discriminator parameter keys do not match")
    for name, target in current.items():
        value = weights[name]
        if (
            not isinstance(value, Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid Real-ESRGAN discriminator tensor: {name}")
    provenance = {
        **discriminator_architecture(module),
        "source": REALESRGAN_D_URL,
        "checkpoint": str(path),
        "file_sha256": digest,
        "initial_state_sha256": state_dict_sha256(weights),
        "param_key": "params",
    }
    validate_discriminator_provenance(provenance, module)
    module.load_state_dict(weights, strict=True)
    return provenance
