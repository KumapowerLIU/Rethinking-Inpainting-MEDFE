"""Atomic, self-contained training state and deterministic sample order."""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, ModelConfig
from .discriminators import SPECTRAL_NORM_POLICY
from .weights import state_dict_sha256

FORMAT = "medfe_training_v1"
# The v2 generator explicitly preserves the released encoder's activated
# saved features. Tensor shapes alone cannot distinguish it from v1. Old runs
# remain readable with their frozen source; never silently reinterpret them.
SEMANTICS = "medfe_joint_released_encoder_corrected_pconv_bpa_aux_v2"
SD_PARTIAL_SEMANTICS = "medfe_sd_partial_residual_gn_resize_encoder_validity_v1"


def semantics_for_model(model: ModelConfig) -> str:
    if model.backbone == "released":
        return SEMANTICS
    if model.backbone == "sd_partial":
        return SD_PARTIAL_SEMANTICS
    raise ValueError(f"Unknown generator backbone: {model.backbone}")


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_progress(progress: Any, *, epoch: int, step: int, batch_size: int) -> None:
    if progress is None:
        return
    keys = {"fingerprint", "size", "epoch", "samples_in_epoch", "samples_seen"}
    if (
        not isinstance(progress, dict)
        or set(progress) != keys
        or not valid_sha256(progress["fingerprint"])
        or any(type(progress[key]) is not int for key in keys - {"fingerprint"})
        or progress["size"] < 1
        or progress["epoch"] != epoch
        or not 0 <= progress["samples_in_epoch"] <= progress["size"]
        or progress["samples_seen"] != epoch * progress["size"] + progress["samples_in_epoch"]
    ):
        raise ValueError("Invalid dataset cursor in checkpoint")
    size, offset = progress["size"], progress["samples_in_epoch"]
    if offset != size and offset % batch_size:
        raise ValueError("Dataset cursor is not on a completed batch boundary")
    expected_steps = epoch * ((size + batch_size - 1) // batch_size)
    expected_steps += (offset + batch_size - 1) // batch_size
    if expected_steps != step:
        raise ValueError("Dataset cursor and optimizer step count disagree")


def _finite_tensors(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"Non-finite tensor in checkpoint {name}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite scalar in checkpoint {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tensors(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tensors(item, f"{name}.{index}")


def code_signature() -> dict[str, str]:
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    array_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [array_state[0], array_state[1].tolist(), *array_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    array = state["numpy"]
    np.random.set_state((array[0], np.asarray(array[1], dtype=np.uint32), *array[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint and runtime expose different CUDA device counts")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if device.type == "mps" and state.get("mps") is not None:
        torch.mps.set_rng_state(state["mps"].cpu())


def epoch_indices(size: int, seed: int, epoch: int, offset: int = 0) -> list[int]:
    if (
        any(type(value) is not int for value in (size, seed, epoch, offset))
        or size < 1
        or seed < 0
        or epoch < 0
        or not 0 <= offset <= size
    ):
        raise ValueError("Invalid epoch size or resume cursor")
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator).tolist()[offset:]


def atomic_save(state: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, *, allow_fixture: bool = False, allow_inference: bool = False
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if allow_inference and isinstance(state, dict) and state.get("format") == "medfe_inference_v1":
        from .inference import validate_bundle

        return validate_bundle(state)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("Not a modern MEDFE training checkpoint; legacy files need conversion")
    if state.get("semantics") not in {SEMANTICS, SD_PARTIAL_SEMANTICS}:
        raise ValueError("MEDFE checkpoint mathematics differ from this implementation")
    if type(state.get("test_fixture")) is not bool:
        raise ValueError("Checkpoint is missing its production/fixture designation")
    if state["test_fixture"] and not allow_fixture:
        raise ValueError("A diagnostic fixture cannot be used as a production model")
    if type(state.get("global_step")) is not int or state["global_step"] < 0:
        raise ValueError("Checkpoint has an invalid step")
    if type(state.get("completed_epochs")) is not int or state["completed_epochs"] < 0:
        raise ValueError("Checkpoint has an invalid epoch")
    required = {
        "config",
        "generator",
        "global_d",
        "local_d",
        "g_optimizer",
        "d_optimizer",
        "g_scheduler",
        "d_scheduler",
        "scaler",
        "rng",
        "data_progress",
        "source",
        "global_d_provenance",
        "global_d_state_sha256",
        "lpips",
        "lpips_provenance",
        "vgg",
        "vgg_provenance",
        "model_state_sha256",
        "discriminator_spectral_norm_policy",
    }
    if required - state.keys():
        raise ValueError(f"Incomplete MEDFE checkpoint: {sorted(required - state.keys())}")
    config = Config.from_dict(state["config"])
    if state["semantics"] != semantics_for_model(config.model):
        raise ValueError("MEDFE checkpoint mathematics differ from its configured backbone")
    validate_progress(
        state["data_progress"],
        epoch=state["completed_epochs"],
        step=state["global_step"],
        batch_size=config.data.batch_size,
    )
    if (
        not isinstance(state["source"], dict)
        or not state["source"]
        or any(
            not isinstance(name, str) or not valid_sha256(digest)
            for name, digest in state["source"].items()
        )
    ):
        raise ValueError("Invalid checkpoint source identity")
    if state["discriminator_spectral_norm_policy"] != SPECTRAL_NORM_POLICY:
        raise ValueError("Checkpoint discriminator normalization semantics differ")
    digests = state["model_state_sha256"]
    if not isinstance(digests, dict) or set(digests) != {"generator", "global_d", "local_d"}:
        raise ValueError("Checkpoint model hashes are incomplete")
    for name in ("generator", "global_d", "local_d"):
        weights = state[name]
        if (
            not isinstance(weights, dict)
            or not weights
            or any(
                not isinstance(key, str) or not isinstance(value, torch.Tensor)
                for key, value in weights.items()
            )
        ):
            raise ValueError(f"Invalid checkpoint weights for {name}")
        _finite_tensors(weights, name)
        if not valid_sha256(digests[name]) or state_dict_sha256(weights) != digests[name]:
            raise ValueError(f"Checkpoint {name} weight hash mismatch")
    if state["global_d_state_sha256"] != digests["global_d"]:
        raise ValueError("Checkpoint global discriminator hashes disagree")
    for name in ("g_optimizer", "d_optimizer", "lpips", "vgg"):
        _finite_tensors(state[name], name)
    return state
