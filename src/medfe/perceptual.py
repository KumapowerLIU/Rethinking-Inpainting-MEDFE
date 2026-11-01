"""Frozen FP32 perceptual metrics with explicit pretrained/fixture provenance."""

from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .weights import file_sha256, state_dict_sha256


def _backbone_cache(net: str, allow_download: bool) -> Path:
    """Check/download the official torchvision backbone before constructing it."""
    from torchvision.models import AlexNet_Weights, SqueezeNet1_1_Weights, VGG16_Weights

    choices = {
        "vgg": VGG16_Weights.IMAGENET1K_V1,
        "alex": AlexNet_Weights.IMAGENET1K_V1,
        "squeeze": SqueezeNet1_1_Weights.IMAGENET1K_V1,
    }
    if net not in choices:
        raise ValueError("LPIPS backbone must be vgg, alex or squeeze")
    weights = choices[net]
    name = Path(urlparse(weights.url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / name
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Pretrained {net} weights are not cached: {path}. "
                "Populate TORCH_HOME or explicitly enable pretrained-weight downloads."
            )
        weights.get_state_dict(progress=False, check_hash=True)
    expected_prefix = Path(name).stem.rsplit("-", 1)[-1]
    if not file_sha256(path).startswith(expected_prefix):
        raise ValueError(f"Cached pretrained {net} weights failed the official SHA256 prefix check")
    return path


def _validate_images(prediction: Tensor, target: Tensor) -> None:
    if (
        prediction.shape != target.shape
        or prediction.ndim != 4
        or prediction.shape[0] < 1
        or prediction.shape[1] != 3
        or prediction.device != target.device
        or not prediction.is_floating_point()
        or not target.is_floating_point()
    ):
        raise ValueError("Perceptual loss expects matching nonempty floating B x 3 x H x W tensors")


def _validate_state(module: nn.Module, state: dict) -> None:
    current = module.state_dict()
    if not isinstance(state, dict) or set(state) != set(current):
        raise ValueError("Perceptual checkpoint tensor keys differ")
    for key, expected in current.items():
        value = state[key]
        if (
            not isinstance(value, Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid perceptual checkpoint tensor: {key}")


class LPIPSLoss(nn.Module):
    """Calibrated LPIPS v0.1 on RGB [-1,1], averaged across the whole batch.

    A restoring instance cannot run until restore_checkpoint has loaded and
    verified its self-contained weights. fixture_network is exclusively for
    tests and is visibly marked in provenance; it is never called pretrained.
    """

    def __init__(
        self,
        net: str = "vgg",
        chunk_size: int = 8,
        allow_download: bool = False,
        restoring: bool = False,
        *,
        fixture_network: nn.Module | None = None,
    ):
        super().__init__()
        if net not in {"vgg", "alex", "squeeze"} or chunk_size < 1:
            raise ValueError("Invalid LPIPS backbone or chunk size")
        self.net, self.chunk_size = net, chunk_size
        self.fixture = fixture_network is not None
        self._restore_pending = restoring
        if self.fixture:
            self.package_version = "fixture"
            self.network = fixture_network
        else:
            try:
                import lpips
            except ImportError as error:
                raise ImportError(
                    "Install lpips==0.1.4 to use the production LPIPS loss"
                ) from error
            self.package_version = version("lpips")
            if self.package_version != "0.1.4":
                raise ValueError(
                    "Production LPIPS requires the audited lpips==0.1.4 implementation"
                )
            if not restoring:
                _backbone_cache(net, allow_download)
            self.network = lpips.LPIPS(
                net=net,
                version="0.1",
                pretrained=not restoring,
                pnet_rand=restoring,
                pnet_tune=False,
                eval_mode=True,
                verbose=False,
            )
        self.network.float().eval().requires_grad_(False)
        self.provenance = self._provenance(pretrained=not restoring)
        self.eval()

    def _provenance(self, *, pretrained: bool) -> dict:
        weights = self.network.state_dict()
        return {
            "kind": "lpips_v0.1",
            "package": "fixture" if self.fixture else "lpips",
            "package_version": self.package_version,
            "net": self.net,
            "version": "0.1",
            "fixture": self.fixture,
            "pretrained": pretrained and not self.fixture,
            "pnet_rand": not pretrained and not self.fixture,
            "pnet_tune": False,
            "normalize": False,
            "input_range": [-1, 1],
            "feature_weights_sha256": state_dict_sha256(
                {key: value for key, value in weights.items() if key.startswith("net.")}
            ),
            "metric_weights_sha256": state_dict_sha256(
                {key: value for key, value in weights.items() if not key.startswith("net.")}
            ),
        }

    def restore_checkpoint(self, state: dict, provenance: dict) -> None:
        self._restore_pending = True
        current = self._provenance(pretrained=True)
        hash_keys = {"feature_weights_sha256", "metric_weights_sha256"}
        if (
            not isinstance(provenance, dict)
            or set(provenance) != set(current)
            or any(provenance[key] != current[key] for key in set(current) - hash_keys)
        ):
            raise ValueError("LPIPS checkpoint provenance differs from this implementation")
        _validate_state(self, state)
        self.load_state_dict(state, strict=True)
        restored = self._provenance(pretrained=True)
        if restored != provenance:
            raise ValueError("LPIPS checkpoint weights do not match provenance")
        self.network.eval().requires_grad_(False)
        self.provenance = restored
        self._restore_pending = False

    def train(self, mode: bool = True):
        # Calibration dropout must remain disabled even when parent G trains.
        super().train(False)
        return self

    def _chunk(self, prediction: Tensor, target: Tensor) -> Tensor:
        with torch.autocast(prediction.device.type, enabled=False):
            return self.network(prediction.float(), target.detach().float(), normalize=False).mean()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self._restore_pending:
            raise RuntimeError("LPIPS is awaiting verified checkpoint weights")
        _validate_images(prediction, target)
        self.eval()
        total = prediction.new_zeros((), dtype=torch.float32)
        count = prediction.shape[0]
        for start in range(0, count, self.chunk_size):
            predicted_chunk = prediction[start : start + self.chunk_size]
            target_chunk = target[start : start + self.chunk_size]
            if torch.is_grad_enabled() and predicted_chunk.requires_grad:
                value = checkpoint(self._chunk, predicted_chunk, target_chunk, use_reentrant=False)
            else:
                value = self._chunk(predicted_chunk, target_chunk)
            total = total + value * (predicted_chunk.shape[0] / count)
        return total


def _vgg16_features() -> nn.Sequential:
    """Torchvision VGG16 feature architecture without allocating its classifier."""
    configuration = (
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    )
    blocks, channels = [], 3
    for width in configuration:
        if width == "M":
            blocks.append(nn.MaxPool2d(2, 2))
        else:
            blocks.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=False)])
            channels = width
    return nn.Sequential(*blocks)


class SharedVGG(nn.Module):
    """VGG16 perceptual/style profiles sharing each image feature pass.

    ``paper`` preserves the normalized objective used by the modern baseline.
    ``released`` uses the released training code's unnormalized RGB [-1,1]
    inputs and layer selections, with the erroneous relu5_2/5_3 wiring fixed.
    Both profiles freeze one VGG backbone and retain prediction gradients.
    """

    CONTENT = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    # Paper page 8, Eq. (8): style uses the same activation maps as perceptual.
    STYLE = CONTENT
    INDICES = {
        1: "relu1_1",
        6: "relu2_1",
        11: "relu3_1",
        18: "relu4_1",
        25: "relu5_1",
    }
    RELEASED_STYLE = ("relu2_2", "relu3_3", "relu4_3", "relu5_2")
    RELEASED_INDICES = {
        **INDICES,
        8: "relu2_2",
        15: "relu3_3",
        22: "relu4_3",
        27: "relu5_2",
    }

    def __init__(
        self,
        allow_download: bool = False,
        restoring: bool = False,
        *,
        profile: str = "paper",
        chunk_size: int = 8,
        fixture_features: nn.Sequential | None = None,
    ):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("VGG chunk size must be positive")
        if profile not in {"paper", "released"}:
            raise ValueError("VGG profile must be paper or released")
        self.profile = profile
        self.style_layers = self.STYLE if profile == "paper" else self.RELEASED_STYLE
        self.feature_indices = self.INDICES if profile == "paper" else self.RELEASED_INDICES
        self.chunk_size = chunk_size
        self.fixture = fixture_features is not None
        self._restore_pending = restoring
        self.network = fixture_features if self.fixture else _vgg16_features()
        if not isinstance(self.network, nn.Sequential) or len(self.network) <= max(
            self.feature_indices
        ):
            raise ValueError("VGG features must expose the canonical layer indices")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if not self.fixture and not restoring:
            path = _backbone_cache("vgg", allow_download)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            weights = {
                key.removeprefix("features."): value
                for key, value in saved.items()
                if key.startswith("features.")
            }
            _validate_state(self.network, weights)
            self.network.load_state_dict(weights, strict=True)
        self.network.float().eval().requires_grad_(False)
        self.provenance = self._provenance(pretrained=not restoring)
        self.eval()

    def _provenance(self, *, pretrained: bool) -> dict:
        return {
            "kind": "shared_vgg16_profiles_v2",
            "profile": self.profile,
            "fixture": self.fixture,
            "pretrained": pretrained and not self.fixture,
            "weights": "fixture" if self.fixture else "VGG16_Weights.IMAGENET1K_V1",
            "input_range": [-1, 1],
            "input_normalization": "imagenet" if self.profile == "paper" else "none",
            "mean": [0.485, 0.456, 0.406] if self.profile == "paper" else None,
            "std": [0.229, 0.224, 0.225] if self.profile == "paper" else None,
            "content_layers": list(self.CONTENT),
            "style_layers": list(self.style_layers),
            "gram_normalization": "C*H*W",
            "state_sha256": state_dict_sha256(self.state_dict()),
        }

    def restore_checkpoint(self, state: dict, provenance: dict) -> None:
        self._restore_pending = True
        current = self._provenance(pretrained=True)
        if (
            not isinstance(provenance, dict)
            or set(provenance) != set(current)
            or any(provenance[key] != current[key] for key in current if key != "state_sha256")
        ):
            raise ValueError("VGG checkpoint provenance differs from this implementation")
        _validate_state(self, state)
        for name in ("mean", "std"):
            if not torch.equal(state[name].to(getattr(self, name).device), getattr(self, name)):
                raise ValueError("VGG checkpoint normalization buffers differ from its profile")
        if state_dict_sha256(state) != provenance["state_sha256"]:
            raise ValueError("VGG checkpoint weights do not match provenance")
        self.load_state_dict(state, strict=True)
        self.network.eval().requires_grad_(False)
        self.provenance = self._provenance(pretrained=True)
        self._restore_pending = False

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def extract_features(self, image: Tensor) -> dict[str, Tensor]:
        if self._restore_pending:
            raise RuntimeError("VGG is awaiting verified checkpoint weights")
        value = image.float()
        if self.profile == "paper":
            value = ((value + 1) * 0.5 - self.mean) / self.std
        result = {}
        for index, layer in enumerate(self.network):
            value = layer(value)
            if index in self.feature_indices:
                result[self.feature_indices[index]] = value
            if index == max(self.feature_indices):
                break
        return result

    @staticmethod
    def gram(value: Tensor) -> Tensor:
        batch, channels, height, width = value.shape
        flat = value.reshape(batch, channels, height * width)
        return flat.bmm(flat.transpose(1, 2)) / (channels * height * width)

    def _chunk(self, prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(prediction.device.type, enabled=False):
            predicted = self.extract_features(prediction)
            with torch.no_grad():
                expected = self.extract_features(target.detach())
            perceptual = sum(F.l1_loss(predicted[name], expected[name]) for name in self.CONTENT)
            style = sum(
                F.l1_loss(self.gram(predicted[name]), self.gram(expected[name]))
                for name in self.style_layers
            )
        return perceptual, style

    def forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        if self._restore_pending:
            raise RuntimeError("VGG is awaiting verified checkpoint weights")
        _validate_images(prediction, target)
        if min(prediction.shape[-2:]) < 16:
            raise ValueError("VGG images must be at least 16 x 16")
        self.eval()
        perceptual = prediction.new_zeros((), dtype=torch.float32)
        style = prediction.new_zeros((), dtype=torch.float32)
        count = prediction.shape[0]
        for start in range(0, count, self.chunk_size):
            predicted_chunk, target_chunk = (
                value[start : start + self.chunk_size] for value in (prediction, target)
            )
            if torch.is_grad_enabled() and predicted_chunk.requires_grad:
                values = checkpoint(self._chunk, predicted_chunk, target_chunk, use_reentrant=False)
            else:
                values = self._chunk(predicted_chunk, target_chunk)
            weight = predicted_chunk.shape[0] / count
            perceptual = perceptual + values[0] * weight
            style = style + values[1] * weight
        return {"perceptual": perceptual, "style": style}

    def losses(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        return self(prediction, target)
