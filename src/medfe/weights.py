"""Stable identities for checkpoint files and tensor state dictionaries."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash keys, shapes, dtypes and exact bytes, independent of tensor device."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError("State dictionaries must map string keys to tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
