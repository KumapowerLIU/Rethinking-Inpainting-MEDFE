"""Validated, serializable experiment settings; no hidden argparse-only defaults."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    base_channels: int = 64
    residual_blocks: int = 4
    pconv_layers: int = 5
    gaussian_sigma: float = 1.5
    init_gain: float = 0.02
    activation_checkpointing: bool = False
    backbone: str = "released"
    blocks_per_scale: int = 2


@dataclass
class DataConfig:
    format: str = "paired"
    rtv_cache: str = ""
    train_manifest: str = ""
    val_manifest: str = ""
    image_size: int = 256
    batch_size: int = 4
    workers: int = 4
    mask_mode: str = "random"
    min_fraction: float = 0.1
    max_fraction: float = 0.4
    allow_repeated_records: bool = False


@dataclass
class LossConfig:
    reconstruction: float = 1.0
    lpips: float = 0.0
    perceptual: float = 0.1
    style: float = 250.0
    structure: float = 1.0
    texture: float = 1.0
    adversarial: float = 0.2
    lpips_net: str = "vgg"
    lpips_chunk_size: int = 8
    vgg_profile: str = "paper"
    vgg_chunk_size: int = 8


@dataclass
class DiscriminatorConfig:
    global_kind: str = "patch"
    global_width: int = 64
    local_width: int = 64
    local_crop_size: int = 64
    chunk_size: int = 8
    pretrained_path: str = ""


@dataclass
class TrainConfig:
    output: str = "runs/medfe"
    device: str = "auto"
    amp: str = "off"
    seed: int = 2019
    epochs: int = 60
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    schedule: str = "constant"
    decay_start: int = 20
    adversarial_warmup_steps: int = 0
    adversarial_ramp_steps: int = 0
    save_every: int = 100
    evaluate_every: int = 500
    log_every: int = 10
    evaluation_images: int = 64
    evaluation_batch_size: int = 4
    memory_budget_gib: float = 0.0
    allow_weight_download: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Config:
        sections = {
            "model": ModelConfig,
            "data": DataConfig,
            "loss": LossConfig,
            "discriminator": DiscriminatorConfig,
            "train": TrainConfig,
        }
        if not isinstance(value, dict) or set(value) - sections.keys():
            raise ValueError("Unknown or invalid configuration sections")
        parts = {}
        for name, kind in sections.items():
            raw = value.get(name, {})
            if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(kind)}:
                raise ValueError(f"Unknown or invalid {name} settings")
            parts[name] = kind(**raw)
        result = cls(**parts)
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Config:
        value = yaml.safe_load(Path(path).read_text())
        config = cls.from_dict(value)
        value = config.as_dict()
        for expression in overrides or []:
            key, separator, raw = expression.partition("=")
            parts = key.split(".")
            if not separator or len(parts) != 2:
                raise ValueError(f"Expected section.key=value: {expression}")
            section, name = parts
            if section not in value or name not in value[section]:
                raise ValueError(f"Unknown setting: {key}")
            expected = get_type_hints(type(getattr(config, section)))[name]
            if expected is str:
                parsed = raw  # Preserve YAML 1.1 strings such as 'off'.
            elif expected is float:
                try:
                    parsed = float(raw)
                except ValueError as error:
                    raise ValueError(f"Expected a number for {key}") from error
            else:
                parsed = yaml.safe_load(raw)
            value[section][name] = parsed
        return cls.from_dict(value)

    def validate(self) -> None:
        for section in fields(self):
            obj = getattr(self, section.name)
            for name, expected in get_type_hints(type(obj)).items():
                value = getattr(obj, name)
                valid = (
                    type(value) in (float, int) if expected is float else type(value) is expected
                )
                if not valid:
                    raise ValueError(f"{section.name}.{name} must be {expected.__name__}")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{section.name}.{name} must be finite")
        if self.model.base_channels < 2 or self.model.residual_blocks < 1:
            raise ValueError("Model width must be >=2 and residual_blocks >=1")
        if self.model.backbone not in {"released", "sd_partial"}:
            raise ValueError("model.backbone must be released or sd_partial")
        if self.model.blocks_per_scale < 1:
            raise ValueError("model.blocks_per_scale must be positive")
        if min(self.model.pconv_layers, self.model.gaussian_sigma, self.model.init_gain) <= 0:
            raise ValueError(
                "Partial-convolution depth, Gaussian sigma and init gain must be positive"
            )
        if self.data.image_size < 256 or self.data.image_size % 64:
            raise ValueError("image_size must be a multiple of 64 and at least 256")
        if self.data.batch_size < 1 or self.data.workers < 0:
            raise ValueError("batch_size must be positive and workers nonnegative")
        if self.data.format not in {"paired", "indexed_rtv"}:
            raise ValueError("data.format must be paired or indexed_rtv")
        if self.data.format == "indexed_rtv" and not self.data.rtv_cache.strip():
            raise ValueError("indexed_rtv requires data.rtv_cache")
        if self.data.allow_repeated_records and self.data.format != "indexed_rtv":
            raise ValueError("allow_repeated_records requires indexed_rtv")
        if self.train.memory_budget_gib < 0:
            raise ValueError("memory_budget_gib must be nonnegative")
        if self.train.evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive")
        if min(self.train.adversarial_warmup_steps, self.train.adversarial_ramp_steps) < 0:
            raise ValueError("Adversarial warmup and ramp steps must be nonnegative")
        if self.data.mask_mode not in {"random", "center", "file"}:
            raise ValueError("mask_mode must be random, center or file")
        if not 0 < self.data.min_fraction <= self.data.max_fraction < 1:
            raise ValueError("Require 0 < min_fraction <= max_fraction < 1")
        for name in (
            "reconstruction",
            "lpips",
            "perceptual",
            "style",
            "structure",
            "texture",
            "adversarial",
        ):
            if getattr(self.loss, name) < 0:
                raise ValueError("Loss coefficients must be nonnegative")
        if not any(
            getattr(self.loss, n) > 0
            for n in (
                "reconstruction",
                "lpips",
                "perceptual",
                "style",
                "structure",
                "texture",
                "adversarial",
            )
        ):
            raise ValueError("At least one generator loss must be enabled")
        if self.train.adversarial_warmup_steps and not any(
            getattr(self.loss, name) > 0
            for name in ("reconstruction", "lpips", "perceptual", "style", "structure", "texture")
        ):
            raise ValueError("Adversarial warmup requires a non-adversarial generator loss")
        if self.loss.lpips_net not in {"vgg", "alex", "squeeze"}:
            raise ValueError("Unsupported LPIPS network")
        if self.loss.vgg_profile not in {"paper", "released"}:
            raise ValueError("VGG profile must be paper or released")
        if (
            min(
                self.loss.lpips_chunk_size,
                self.loss.vgg_chunk_size,
                self.discriminator.chunk_size,
            )
            < 1
        ):
            raise ValueError("Chunk sizes must be positive")
        if self.discriminator.global_kind not in {"patch", "realesrgan_unet_sn"}:
            raise ValueError("Unsupported global discriminator")
        if min(self.discriminator.global_width, self.discriminator.local_width) < 1:
            raise ValueError("Discriminator widths must be positive")
        crop = self.discriminator.local_crop_size
        if crop < 32 or crop % 32 or crop > self.data.image_size:
            raise ValueError("local_crop_size must be a multiple of 32 between 32 and image_size")
        if re.fullmatch(r"auto|cpu|mps|cuda(?::[0-9]+)?", self.train.device) is None:
            raise ValueError("device must be auto, cpu, cuda[:index], or mps")
        if self.train.amp not in {"off", "bf16", "fp16"}:
            raise ValueError("amp must be off, bf16, or fp16")
        if self.train.seed < 0 or self.train.epochs < 1 or self.train.lr <= 0:
            raise ValueError("Invalid training seed, epochs, or learning rate")
        if not 0 <= self.train.beta1 < 1 or not 0 <= self.train.beta2 < 1:
            raise ValueError("Adam betas must be in [0,1)")
        if self.train.schedule not in {"constant", "linear"}:
            raise ValueError("schedule must be constant or linear")
        if self.train.decay_start < 0 or (
            self.train.schedule == "linear" and self.train.decay_start >= self.train.epochs
        ):
            raise ValueError("Linear decay must start before the last epoch")
        if (
            min(
                self.train.save_every,
                self.train.evaluate_every,
                self.train.log_every,
                self.train.evaluation_images,
            )
            < 1
        ):
            raise ValueError("Logging, saving and evaluation intervals/counts must be positive")

    def resume_contract(self) -> dict[str, Any]:
        """Only execution locations and observability may change on exact resume."""
        value = self.as_dict()
        for key in ("train_manifest", "val_manifest", "workers", "rtv_cache"):
            value["data"].pop(key)
        for key in (
            "output",
            "device",
            "save_every",
            "evaluate_every",
            "log_every",
            "evaluation_images",
            "evaluation_batch_size",
            "memory_budget_gib",
            "allow_weight_download",
        ):
            value["train"].pop(key)
        value["discriminator"].pop("pretrained_path")
        return value
