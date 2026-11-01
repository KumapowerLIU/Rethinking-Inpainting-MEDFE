"""Portable generator bundles; training resume still requires the full checkpoint."""

from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import atomic_save, semantics_for_model, valid_sha256
from .config import Config
from .weights import file_sha256, state_dict_sha256

FORMAT = "medfe_inference_v1"
INPUT_CONTRACT = {"rgb_range": [-1, 1], "mask": "true_is_hole", "known_pixels": "copied"}


def validate_bundle(state: dict) -> dict:
    required = {
        "format",
        "semantics",
        "config",
        "generator",
        "generator_sha256",
        "global_step",
        "completed_epochs",
        "source_checkpoint_sha256",
        "input_contract",
    }
    if set(state) != required or state["format"] != FORMAT:
        raise ValueError("Invalid MEDFE inference bundle fields")
    config = Config.from_dict(state["config"])
    if state["semantics"] != semantics_for_model(config.model):
        raise ValueError("Inference bundle backbone and semantics differ")
    if state["input_contract"] != INPUT_CONTRACT:
        raise ValueError("Unsupported inference input contract")
    for key in ("global_step", "completed_epochs"):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError(f"Invalid inference {key}")
    if not valid_sha256(state["source_checkpoint_sha256"]):
        raise ValueError("Invalid inference source checksum")
    weights = state["generator"]
    if (
        not isinstance(weights, dict)
        or not weights
        or any(
            not isinstance(k, str)
            or not isinstance(v, torch.Tensor)
            or not bool(torch.isfinite(v).all())
            for k, v in weights.items()
        )
    ):
        raise ValueError("Invalid inference generator tensors")
    if not valid_sha256(state["generator_sha256"]) or (
        state_dict_sha256(weights) != state["generator_sha256"]
    ):
        raise ValueError("Inference generator checksum mismatch")
    return state


def export_generator(checkpoint: str | Path, output: str | Path) -> dict:
    """Export validated FP32 G tensors without optimizers, D, metric weights or private paths."""
    from .evaluation import load_generator

    source, destination = Path(checkpoint), Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    before = source.stat()
    model, config, payload = load_generator(source, "cpu")
    weights = model.state_dict()
    if state_dict_sha256(weights) != state_dict_sha256(payload["generator"]):
        raise ValueError("Export would change generator tensor values or dtypes")
    # Evaluation paths are supplied by the caller; retain only public protocol settings.
    data = {
        key: getattr(config.data, key)
        for key in ("format", "image_size", "mask_mode", "min_fraction", "max_fraction")
    }
    if data["format"] == "indexed_rtv":
        data["rtv_cache"] = "data/indexed-rtv-cache"
    portable_config = {
        "model": asdict(config.model),
        "data": data,
        "train": {"seed": config.train.seed},
    }
    bundle = {
        "format": FORMAT,
        "semantics": semantics_for_model(config.model),
        "config": portable_config,
        "generator": weights,
        "generator_sha256": state_dict_sha256(weights),
        "global_step": payload["global_step"],
        "completed_epochs": payload["completed_epochs"],
        "source_checkpoint_sha256": file_sha256(source),
        "input_contract": dict(INPUT_CONTRACT),
    }
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("Checkpoint changed during export; use a stopped or pinned checkpoint")
    validate_bundle(bundle)
    atomic_save(bundle, destination)
    return {
        "output": str(destination.resolve()),
        "step": bundle["global_step"],
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "generator_sha256": bundle["generator_sha256"],
        "source_checkpoint_sha256": bundle["source_checkpoint_sha256"],
    }
