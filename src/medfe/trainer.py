"""One joint MEDFE generator; explicit D/G phases and one G backward pass."""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .checkpoint import (
    FORMAT,
    atomic_save,
    capture_rng,
    code_signature,
    load_checkpoint,
    restore_rng,
    semantics_for_model,
    valid_sha256,
    validate_progress,
)
from .config import Config
from .discriminators import (
    SPECTRAL_NORM_POLICY,
    PatchDiscriminator,
    UNetDiscriminatorSN,
    load_realesrgan_weights,
    patch_discriminator_provenance,
    prepare_discriminator,
    validate_discriminator_provenance,
)
from .losses import auxiliary_losses, ralsgan_discriminator, ralsgan_generator
from .model import MEDFEGenerator
from .perceptual import LPIPSLoss, SharedVGG
from .weights import state_dict_sha256


@contextmanager
def frozen_parameters(*modules: nn.Module):
    parameters = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def select_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def crop_batch(image: Tensor, positions: Tensor, size: int) -> Tensor:
    """Each sample has its own crop, shared between real/fake and both GAN phases."""
    return torch.stack(
        [
            sample[:, int(top) : int(top) + size, int(left) : int(left) + size]
            for sample, (top, left) in zip(image, positions)
        ]
    )


class Trainer:
    def __init__(
        self,
        config: Config,
        *,
        restoring: bool = False,
        test_fixture: bool = False,
        generator: nn.Module | None = None,
        global_d: nn.Module | None = None,
        local_d: nn.Module | None = None,
        lpips: nn.Module | None = None,
        vgg: nn.Module | None = None,
    ):
        config.validate()
        if not test_fixture and any(
            x is not None for x in (generator, global_d, local_d, lpips, vgg)
        ):
            raise ValueError("Injected test modules require test_fixture=True")
        self.config = copy.deepcopy(config)
        self.device = select_device(config.train.device)
        if config.train.memory_budget_gib:
            if self.device.type != "cuda":
                raise ValueError("A GPU memory budget requires a CUDA device")
            free, total = torch.cuda.mem_get_info(self.device)
            budget = int(config.train.memory_budget_gib * 1024**3)
            if budget > free - 2 * 1024**3:
                raise RuntimeError("Insufficient free GPU memory for budget plus 2 GiB headroom")
            torch.cuda.set_per_process_memory_fraction(budget / total, self.device)
        self.test_fixture = test_fixture
        self._ready = not restoring
        self._healthy = True
        self._step_in_progress = False
        self.global_step = 0
        self.completed_epochs = 0
        self.data_progress: dict[str, Any] | None = None
        self.source = code_signature()
        if config.train.amp == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        if config.train.amp == "bf16" and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("The selected CUDA device does not support BF16")
        if config.train.amp != "off" and self.device.type == "mps":
            raise ValueError("MPS training currently requires amp=off")
        self.generator = (
            generator if generator is not None else MEDFEGenerator(**asdict(config.model))
        ).to(self.device)
        dc = config.discriminator
        if global_d is not None:
            self.global_d = global_d
        elif dc.global_kind == "patch":
            self.global_d = PatchDiscriminator(width=dc.global_width)
        else:
            if not test_fixture and dc.global_width != 64:
                raise ValueError("Official Real-ESRGAN initialization requires width 64")
            self.global_d = UNetDiscriminatorSN(num_feat=dc.global_width)
        self.global_d_provenance: dict[str, Any] = {
            "architecture": dc.global_kind,
            "width": dc.global_width,
            "test_fixture": test_fixture,
        }
        if isinstance(self.global_d, PatchDiscriminator):
            self.global_d_provenance = patch_discriminator_provenance(
                self.global_d, test_fixture=test_fixture
            )
        if dc.global_kind == "realesrgan_unet_sn" and not restoring and not test_fixture:
            if not dc.pretrained_path:
                raise ValueError(
                    "Fresh Real-ESRGAN training requires discriminator.pretrained_path"
                )
            self.global_d_provenance = load_realesrgan_weights(self.global_d, dc.pretrained_path)
        self.global_d = self.global_d.to(self.device).eval()
        self.local_d = (
            (local_d if local_d is not None else PatchDiscriminator(width=dc.local_width))
            .to(self.device)
            .eval()
        )
        self.lpips = lpips
        self.vgg = vgg
        if config.loss.lpips and self.lpips is None:
            self.lpips = LPIPSLoss(
                net=config.loss.lpips_net,
                chunk_size=config.loss.lpips_chunk_size,
                allow_download=config.train.allow_weight_download,
                restoring=restoring,
            )
        if (config.loss.perceptual or config.loss.style) and self.vgg is None:
            self.vgg = SharedVGG(
                allow_download=config.train.allow_weight_download,
                restoring=restoring,
                profile=config.loss.vgg_profile,
                chunk_size=config.loss.vgg_chunk_size,
            )
        for metric in (self.lpips, self.vgg):
            if metric is not None:
                metric.to(self.device).float().eval().requires_grad_(False)
                if not test_fixture and metric.provenance.get("fixture", False):
                    raise ValueError("Production losses cannot use fixture features")
        optimizer_args = {"lr": config.train.lr, "betas": (config.train.beta1, config.train.beta2)}
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), **optimizer_args)
        self.d_optimizer = torch.optim.Adam(
            [*self.global_d.parameters(), *self.local_d.parameters()],
            **optimizer_args,
        )

        def multiplier(epoch):
            if config.train.schedule == "constant":
                return 1.0
            return max(
                0.0,
                1.0
                - max(0, epoch - config.train.decay_start)
                / (config.train.epochs - config.train.decay_start),
            )

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, multiplier)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, multiplier)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp == "fp16")

    def autocast(self):
        if self.config.train.amp == "off":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.train.amp == "bf16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def effective_adversarial_weight(self) -> float:
        """Weight for the next update, indexed by completed generator updates.

        The first N updates disable both GAN phases. With a ramp of M updates,
        the first enabled update uses 1/M of the configured coefficient, and
        the Mth reaches it. Zero/zero preserves the unscheduled objective.
        """
        elapsed = self.global_step - self.config.train.adversarial_warmup_steps
        if elapsed < 0:
            return 0.0
        ramp = self.config.train.adversarial_ramp_steps
        fraction = min(1.0, (elapsed + 1) / ramp) if ramp else 1.0
        return self.config.loss.adversarial * fraction

    def bind_data(self, fingerprint: str, size: int) -> None:
        if not valid_sha256(fingerprint) or type(size) is not int or size < 1:
            raise ValueError("Dataset requires a SHA256 identity and a positive size")
        if self.data_progress is None:
            if self.global_step != 0 or self.completed_epochs != 0:
                raise ValueError(
                    "Bind the dataset before training; probe updates cannot become a fresh run"
                )
            self.data_progress = {
                "fingerprint": fingerprint,
                "size": size,
                "epoch": self.completed_epochs,
                "samples_in_epoch": 0,
                "samples_seen": 0,
            }
        elif self.data_progress["fingerprint"] != fingerprint or self.data_progress["size"] != size:
            raise ValueError("Dataset changed since checkpoint; start a new experiment explicitly")

    def _logits(self, discriminator: nn.Module, image: Tensor, *, global_image: bool) -> Tensor:
        if discriminator.training:
            raise RuntimeError("Discriminator buffers must be frozen during phase replay")
        if global_image and self.config.discriminator.global_kind == "realesrgan_unet_sn":
            image = (image.float() + 1) * 0.5
        logits = []
        for chunk in image.split(self.config.discriminator.chunk_size):
            if torch.is_grad_enabled() and (
                chunk.requires_grad or any(p.requires_grad for p in discriminator.parameters())
            ):
                value = checkpoint(
                    discriminator, chunk, use_reentrant=False, preserve_rng_state=False
                )
            else:
                value = discriminator(chunk)
            logits.append(value.float())
        # RaLSGAN averages span every image and spatial logit, never per-chunk means.
        return torch.cat(logits)

    @staticmethod
    def _finite(loss: Tensor, name: str) -> None:
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"Non-finite {name}; the previous checkpoint remains valid")

    def _check_gradients(self, optimizer: torch.optim.Optimizer, name: str) -> None:
        self.scaler.unscale_(optimizer)
        gradients = [
            p.grad
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not gradients or not bool(
            torch.stack([torch.isfinite(g).all() for g in gradients]).all()
        ):
            raise FloatingPointError(f"Missing or non-finite {name} gradients")

    def step(self, image: Tensor, structure: Tensor, mask: Tensor) -> dict[str, float]:
        if not self._ready or not self._healthy:
            raise RuntimeError("Trainer must be initialized/restored successfully before stepping")
        if image.shape != structure.shape or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Aligned image and structure tensors must have shape B,3,H,W")
        if mask.shape != (image.shape[0], 1, *image.shape[-2:]) or mask.dtype != torch.bool:
            raise ValueError("Mask must be a per-sample B,1,H,W boolean hole mask")
        if not 1 <= image.shape[0] <= self.config.data.batch_size or any(
            size != self.config.data.image_size for size in image.shape[-2:]
        ):
            raise ValueError("Training images must match the configured square image_size")
        if self.data_progress is not None:
            remaining = self.data_progress["size"] - self.data_progress["samples_in_epoch"]
            if len(image) != min(self.config.data.batch_size, remaining):
                raise ValueError(
                    "Batch must match the next complete batch or the epoch's final remainder"
                )
        image = image.to(self.device, dtype=torch.float32, non_blocking=True)
        structure = structure.to(self.device, dtype=torch.float32, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        for name, tensor in (("image", image), ("structure", structure)):
            self._finite(tensor, name)
            if bool((tensor.abs() > 1.00001).any()):
                raise ValueError(f"{name} must use RGB [-1,1]")
        adversarial_weight = self.effective_adversarial_weight()
        adversarial_enabled = adversarial_weight > 0
        if adversarial_enabled:
            crop_size = self.config.discriminator.local_crop_size
            positions = torch.stack(
                (
                    torch.randint(image.shape[-2] - crop_size + 1, (len(image),)),
                    torch.randint(image.shape[-1] - crop_size + 1, (len(image),)),
                ),
                1,
            )
        self.generator.train()
        self.g_optimizer.zero_grad(set_to_none=True)
        if adversarial_enabled:
            self.d_optimizer.zero_grad(set_to_none=True)
        self._step_in_progress = True
        try:
            with self.autocast():
                output = self.generator(image, mask)
            d_global = d_local = d_loss = image.new_zeros(())
            if adversarial_enabled:
                with self.autocast():
                    real_crop = crop_batch(image, positions, crop_size)
                    fake_crop = crop_batch(output.prediction, positions, crop_size)
                    prepare_discriminator(self.global_d)
                    prepare_discriminator(self.local_d)
                    d_global = ralsgan_discriminator(
                        self._logits(self.global_d, image, global_image=True),
                        self._logits(self.global_d, output.prediction.detach(), global_image=True),
                    )
                    d_local = ralsgan_discriminator(
                        self._logits(self.local_d, real_crop, global_image=False),
                        self._logits(self.local_d, fake_crop.detach(), global_image=False),
                    )
                    d_loss = d_global + d_local
                self._finite(d_loss, "discriminator loss")
                self.scaler.scale(d_loss).backward()
                self._check_gradients(self.d_optimizer, "discriminator")
                self.scaler.step(self.d_optimizer)
                prepare_discriminator(self.global_d)
                prepare_discriminator(self.local_d)
            with frozen_parameters(self.global_d, self.local_d), self.autocast():
                adversarial = image.new_zeros(())
                if adversarial_enabled:
                    with torch.no_grad():
                        real_global = self._logits(self.global_d, image, global_image=True)
                        real_local = self._logits(self.local_d, real_crop, global_image=False)
                    g_global = ralsgan_generator(
                        real_global,
                        self._logits(self.global_d, output.prediction, global_image=True),
                    )
                    g_local = ralsgan_generator(
                        real_local,
                        self._logits(self.local_d, fake_crop, global_image=False),
                    )
                    adversarial = g_global + g_local
                aux = auxiliary_losses(output.structure, output.texture, structure, image)
                raw = {
                    "reconstruction": F.l1_loss(output.prediction.float(), image),
                    "structure": aux["structure"],
                    "texture": aux["texture"],
                    "adversarial": adversarial,
                    "lpips": image.new_zeros(()),
                    "perceptual": image.new_zeros(()),
                    "style": image.new_zeros(()),
                }
                if self.lpips is not None and self.config.loss.lpips:
                    raw["lpips"] = self.lpips(output.completed, image)
                if self.vgg is not None and (self.config.loss.perceptual or self.config.loss.style):
                    raw.update(self.vgg(output.prediction, image))
                weighted = {
                    name: value
                    * (
                        adversarial_weight
                        if name == "adversarial"
                        else getattr(self.config.loss, name)
                    )
                    for name, value in raw.items()
                }
                g_loss = sum(weighted.values())
                self._finite(g_loss, "generator loss")
                # Both auxiliary branches enter this sum exactly once.
                self.scaler.scale(g_loss).backward()
            self._check_gradients(self.g_optimizer, "generator")
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
            self.global_step += 1
            if self.data_progress is not None:
                self.data_progress["samples_in_epoch"] += len(image)
                self.data_progress["samples_seen"] += len(image)
            values = {
                "generator": g_loss,
                "discriminator": d_loss,
                "d_global": d_global,
                "d_local": d_local,
                **raw,
                **{name + "_weighted": value for name, value in weighted.items()},
                "hole_fraction": mask.float().mean(),
                "effective_adversarial_weight": image.new_tensor(adversarial_weight),
            }
            numbers = torch.stack([v.detach().float() for v in values.values()]).cpu().tolist()
            return dict(zip(values, numbers))
        except BaseException:
            self._healthy = False  # Never save a half-applied D/G update as resumable state.
            raise
        finally:
            self._step_in_progress = False

    def finish_epoch(self) -> None:
        if not self._ready or not self._healthy or self._step_in_progress:
            raise RuntimeError("Only a complete, healthy training step can finish an epoch")
        if self.data_progress is None or (
            self.data_progress["samples_in_epoch"] != self.data_progress["size"]
        ):
            raise ValueError("An epoch can finish only after all its samples were consumed")
        self.g_scheduler.step()
        if self.d_optimizer.state:
            self.d_scheduler.step()
        else:
            # Both use the same epoch policy. During GAN warmup no D optimizer
            # step has occurred: synchronize its schedule without inventing a
            # step or triggering PyTorch's scheduler-before-optimizer warning.
            self.d_scheduler.load_state_dict(self.g_scheduler.state_dict())
            for group, lr in zip(self.d_optimizer.param_groups, self.d_scheduler.get_last_lr()):
                group["lr"] = lr
        self.completed_epochs += 1
        self.data_progress.update(epoch=self.completed_epochs, samples_in_epoch=0)

    def save(self, path: str | Path) -> None:
        if not self._ready or not self._healthy or self._step_in_progress:
            raise RuntimeError("Only a complete, healthy training step can be checkpointed")
        validate_progress(
            self.data_progress,
            epoch=self.completed_epochs,
            step=self.global_step,
            batch_size=self.config.data.batch_size,
        )
        model_weights = {
            name: getattr(self, name).state_dict() for name in ("generator", "global_d", "local_d")
        }
        model_hashes = {name: state_dict_sha256(weights) for name, weights in model_weights.items()}
        state = {
            "format": FORMAT,
            "semantics": semantics_for_model(self.config.model),
            "test_fixture": self.test_fixture,
            "config": self.config.as_dict(),
            "torch_version": str(torch.__version__),
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "data_progress": copy.deepcopy(self.data_progress),
            "source": self.source,
            **model_weights,
            "model_state_sha256": model_hashes,
            "global_d_provenance": copy.deepcopy(self.global_d_provenance),
            "global_d_state_sha256": model_hashes["global_d"],
            "discriminator_spectral_norm_policy": SPECTRAL_NORM_POLICY,
            "lpips": self.lpips.state_dict() if self.lpips is not None else None,
            "lpips_provenance": self.lpips.provenance if self.lpips is not None else None,
            "vgg": self.vgg.state_dict() if self.vgg is not None else None,
            "vgg_provenance": self.vgg.provenance if self.vgg is not None else None,
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_scheduler": self.g_scheduler.state_dict(),
            "d_scheduler": self.d_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": capture_rng(),
        }
        atomic_save(state, path)

    def restore(self, path: str | Path) -> None:
        self._ready = False
        state = load_checkpoint(path, allow_fixture=self.test_fixture)
        saved = Config.from_dict(state["config"])
        if state["test_fixture"] != self.test_fixture:
            raise ValueError("Production and fixture checkpoint kinds differ")
        if self.config.resume_contract() != saved.resume_contract():
            raise ValueError("Resume configuration changes architecture, losses, or training order")
        if state["source"] != self.source:
            raise ValueError("Source changed since checkpoint; use the original source snapshot")
        progress = state["data_progress"]
        # Imported lazily so module-level tools remain usable without cached weights.
        if self.config.discriminator.global_kind == "realesrgan_unet_sn" and not self.test_fixture:
            provenance = state["global_d_provenance"]
            from .discriminators import REALESRGAN_D_SHA256, REALESRGAN_D_URL

            validate_discriminator_provenance(provenance, self.global_d)
            if (
                provenance["file_sha256"] != REALESRGAN_D_SHA256
                or provenance["source"] != REALESRGAN_D_URL
            ):
                raise ValueError("Checkpoint does not originate from the official Real-ESRGAN D")
        elif state["global_d_provenance"] != self.global_d_provenance:
            raise ValueError("Checkpoint discriminator architecture/fixture identity differs")
        # PyTorch load_state_dict may silently cast dtypes; validate the tensor
        # contract before mutating any of the trainable networks.
        for name in ("generator", "global_d", "local_d"):
            current = getattr(self, name).state_dict()
            incoming = state[name]
            if set(current) != set(incoming) or any(
                current[key].shape != incoming[key].shape
                or current[key].dtype != incoming[key].dtype
                for key in current
            ):
                raise ValueError(f"Checkpoint {name} tensor shape/dtype differs")
            if name == "generator":
                for key in current:
                    if key == "fill" or key.endswith(".count_kernel"):
                        if not torch.equal(current[key].cpu(), incoming[key]):
                            raise ValueError(f"Checkpoint changed fixed generator buffer: {key}")
        for name in ("generator", "global_d", "local_d"):
            getattr(self, name).load_state_dict(state[name], strict=True)
        for name in ("lpips", "vgg"):
            module = getattr(self, name)
            if (module is None) != (state[name] is None):
                raise ValueError(f"Checkpoint {name} objective differs")
            if module is not None:
                module.restore_checkpoint(state[name], state[name + "_provenance"])
        # A failed FP16 D/G half-step leaves per-optimizer scaler bookkeeping;
        # use a fresh scaler so restoring the same Trainer is also safe.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.train.amp == "fp16")
        for name in ("g_optimizer", "d_optimizer", "g_scheduler", "d_scheduler", "scaler"):
            getattr(self, name).load_state_dict(state[name])
        self.global_d_provenance = copy.deepcopy(state["global_d_provenance"])
        self.global_step = state["global_step"]
        self.completed_epochs = state["completed_epochs"]
        self.data_progress = copy.deepcopy(progress)
        self.global_d.eval()
        self.local_d.eval()
        restore_rng(state["rng"], self.device)
        self._healthy = True
        self._ready = True
