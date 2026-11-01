"""Paired RGB/RTV datasets with reproducible geometry and content-checked caches.

JSONL rows contain unique ``id``, ``image``, ``structure``, optional ``mask`` and
optional ``image_sha256``, ``structure_sha256``, ``mask_sha256``. Relative paths
are resolved against the manifest directory. Syntax/IDs are checked eagerly;
file bytes, optional hashes, dimensions and value ranges are checked before each
sample is returned. Generated manifests include hashes for every RGB/RTV file.

File masks must align with the original, EXIF-oriented RGB geometry; white means
hole. They are thresholded at 127, resized with nearest-neighbour interpolation,
then receive the same crop/flip as RGB and structure. Random masks are generated
after the shared image transform and are independent per sample and epoch.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import Dataset

from .masks import center_mask, random_mask, sample_seed
from .rtv import RTV_VERSION, RTVConfig, rtv_smooth

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _read_bytes(path: Path, expected_hash: str | None = None) -> bytes:
    content = path.read_bytes()
    if expected_hash is not None and hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"Content hash mismatch: {path}")
    return content


def _rgb_from_bytes(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def _read_structure(path: Path, expected_hash: str | None) -> np.ndarray:
    content = _read_bytes(path, expected_hash)
    if path.suffix.lower() == ".npy":
        structure = np.load(io.BytesIO(content), allow_pickle=False)
        if not np.issubdtype(structure.dtype, np.floating):
            raise ValueError(f"NPY structure must be floating point in [0, 1]: {path}")
    else:
        structure = _rgb_from_bytes(content)
    if (
        structure.ndim != 3
        or structure.shape[-1] != 3
        or min(structure.shape[:2]) < 1
        or not np.isfinite(structure).all()
        or structure.min() < 0
        or structure.max() > 1
    ):
        raise ValueError(f"Invalid RGB structure image: {path}")
    return structure.astype(np.float32, copy=False)


@dataclass(frozen=True)
class _Record:
    id: str
    image: Path
    structure: Path
    mask: Path | None
    image_sha256: str | None
    structure_sha256: str | None
    mask_sha256: str | None


class InpaintingDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        seed: int = 2019,
        training: bool = True,
        mask_mode: str = "random",
        min_fraction: float = 0.1,
        max_fraction: float = 0.4,
    ):
        if type(image_size) is not int or image_size < 1:
            raise ValueError("image_size must be a positive integer")
        if type(seed) is not int or type(training) is not bool:
            raise ValueError("seed must be an integer and training must be bool")
        if mask_mode not in {"random", "file", "center"}:
            raise ValueError("mask_mode must be random, file, or center")
        if (
            not math.isfinite(min_fraction)
            or not math.isfinite(max_fraction)
            or not 0 < min_fraction <= max_fraction < 1
        ):
            raise ValueError("Require 0 < min_fraction <= max_fraction < 1")
        self.manifest = Path(manifest).resolve()
        raw = self.manifest.read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.image_size, self.seed, self.training = image_size, seed, training
        self.mask_mode = mask_mode
        self.min_fraction, self.max_fraction = min_fraction, max_fraction
        # A shared scalar also reaches persistent DataLoader workers after
        # set_epoch(), instead of leaving their private dataset copies at zero.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.records = []
        seen = set()
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            allowed = {
                "id",
                "image",
                "structure",
                "mask",
                "image_sha256",
                "structure_sha256",
                "mask_sha256",
            }
            if not isinstance(record, dict) or set(record) - allowed:
                raise ValueError(f"Invalid manifest record on line {number}")
            for key in ("id", "image", "structure"):
                if not isinstance(record.get(key), str) or not record[key].strip():
                    raise ValueError(f"Missing or invalid {key} on manifest line {number}")
            if record["id"] in seen:
                raise ValueError(f"Duplicate manifest id: {record['id']}")
            seen.add(record["id"])
            for key in ("image_sha256", "structure_sha256", "mask_sha256"):
                value = record.get(key)
                if value is not None and (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    raise ValueError(f"Invalid {key} on manifest line {number}")
            mask = record.get("mask")
            if mask is not None and (not isinstance(mask, str) or not mask.strip()):
                raise ValueError(f"Invalid mask path on manifest line {number}")
            if mask_mode == "file" and mask is None:
                raise ValueError("File mask mode requires a mask path in every record")
            if record.get("mask_sha256") is not None and mask is None:
                raise ValueError("mask_sha256 requires a mask path")

            def resolve(value):
                return (self.manifest.parent / value).resolve()

            self.records.append(
                _Record(
                    record["id"],
                    resolve(record["image"]),
                    resolve(record["structure"]),
                    resolve(mask) if mask is not None else None,
                    record.get("image_sha256"),
                    record.get("structure_sha256"),
                    record.get("mask_sha256"),
                )
            )
        if not self.records:
            raise ValueError("Manifest contains no samples")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._epoch.fill_(epoch)

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.records):
            raise IndexError(index)
        index = int(index)
        record = self.records[index]
        image = _rgb_from_bytes(_read_bytes(record.image, record.image_sha256))
        structure = _read_structure(record.structure, record.structure_sha256)
        if image.shape != structure.shape:
            raise ValueError(f"Image/structure dimensions differ for {record.id}")
        height, width = image.shape[:2]
        mask = None
        if self.mask_mode == "file":
            with Image.open(io.BytesIO(_read_bytes(record.mask, record.mask_sha256))) as opened:
                mask_array = np.asarray(ImageOps.exif_transpose(opened).convert("L")) > 127
            if mask_array.shape != (height, width):
                raise ValueError(f"File mask dimensions differ from RGB for {record.id}")
            mask = torch.from_numpy(mask_array.copy()).unsqueeze(0)
        # Stack the two RGB tensors before interpolation to guarantee one exact
        # geometric transform without quantizing floating-point RTV targets.
        pair = torch.from_numpy(np.concatenate((image, structure), axis=2).copy()).permute(2, 0, 1)
        scale = self.image_size / min(height, width)
        resized = (
            max(self.image_size, int(round(height * scale))),
            max(self.image_size, int(round(width * scale))),
        )
        pair = F.interpolate(
            pair.unsqueeze(0), size=resized, mode="bilinear", align_corners=False, antialias=True
        )[0]
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=resized, mode="nearest")[0].bool()
        epoch = self.epoch if self.training else 0
        rng = np.random.default_rng(sample_seed(self.seed, epoch, index, "transform"))
        if self.training:
            top = int(rng.integers(resized[0] - self.image_size + 1))
            left = int(rng.integers(resized[1] - self.image_size + 1))
            flip = bool(rng.integers(2))
        else:
            top = (resized[0] - self.image_size) // 2
            left = (resized[1] - self.image_size) // 2
            flip = False
        pair = pair[:, top : top + self.image_size, left : left + self.image_size]
        if mask is not None:
            mask = mask[:, top : top + self.image_size, left : left + self.image_size]
        if flip:
            pair = pair.flip(-1)
            if mask is not None:
                mask = mask.flip(-1)
        if self.mask_mode == "random":
            mask = random_mask(
                self.image_size,
                self.image_size,
                sample_seed(self.seed, epoch, index, "mask"),
                self.min_fraction,
                self.max_fraction,
            )
        elif self.mask_mode == "center":
            mask = center_mask(self.image_size, self.image_size)
        return {
            "image": pair[:3].contiguous() * 2 - 1,
            "structure": pair[3:].contiguous() * 2 - 1,
            "mask": mask.contiguous(),
            "id": record.id,
            "index": index,
        }


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _prepare_rtv_record(arguments):
    image_path, identifier, cache_root, config = arguments
    image_path, cache_root = Path(image_path), Path(cache_root)
    content = image_path.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    image = _rgb_from_bytes(content)
    cache = cache_root / config.fingerprint / f"{source_hash}.npy"
    metadata_path = cache.with_suffix(".json")
    reused = False
    if cache.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            reused = (
                metadata["source_sha256"] == source_hash
                and metadata["config_fingerprint"] == config.fingerprint
                and metadata["algorithm"] == RTV_VERSION
                and tuple(metadata["shape"]) == image.shape
                and isinstance(metadata["structure_sha256"], str)
                and len(metadata["structure_sha256"]) == 64
            )
            if reused:
                target = _read_structure(cache, metadata["structure_sha256"])
                reused = target.shape == image.shape
        except (OSError, ValueError, KeyError, TypeError):
            reused = False
    if not reused:
        target = rtv_smooth(image, config).astype(np.float32)
        stream = io.BytesIO()
        np.save(stream, target, allow_pickle=False)
        target_bytes = stream.getvalue()
        metadata = {
            "algorithm": RTV_VERSION,
            "parameters": config.as_dict(),
            "config_fingerprint": config.fingerprint,
            "source_sha256": source_hash,
            "structure_sha256": hashlib.sha256(target_bytes).hexdigest(),
            "shape": list(image.shape),
            "storage": "float32 NPY, RGB [0,1]",
            "solver": "SciPy direct sparse LU; MATLAB PCG is not bitwise equivalent",
        }
        _atomic_write(cache, target_bytes)
        _atomic_write(
            metadata_path, (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
        )
    return {
        "id": identifier,
        "image": str(image_path),
        "structure": str(cache),
        "image_sha256": source_hash,
        "structure_sha256": metadata["structure_sha256"],
    }, reused


def prepare_rtv_dataset(
    images_root: str | Path,
    cache_root: str | Path,
    manifest_path: str | Path,
    config: RTVConfig | None = None,
    workers: int = 0,
) -> dict:
    """Prepare content-addressed RTV targets and atomically publish JSONL pairs.

    IDs are source paths relative to ``images_root`` (including extension).
    Targets are lossless float32 NPY arrays; input files, algorithm version and
    RTV parameters all participate in cache identity. Damaged caches are rebuilt.
    No dataset downloads or resizing are performed by this preparation step.
    """
    config = RTVConfig() if config is None else config
    if not isinstance(config, RTVConfig):
        raise TypeError("config must be RTVConfig")
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    root, cache, destination = (
        Path(images_root).resolve(),
        Path(cache_root).resolve(),
        Path(manifest_path).resolve(),
    )
    if not root.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported RGB images found under {root}")
    arguments = [
        (str(path), path.relative_to(root).as_posix(), str(cache), config) for path in images
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_prepare_rtv_record, arguments))
    else:
        results = [_prepare_rtv_record(item) for item in arguments]
    records = []
    for record, _ in results:
        for key in ("image", "structure"):
            record[key] = Path(os.path.relpath(record[key], destination.parent)).as_posix()
        records.append(record)
    raw = ("\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n").encode()
    _atomic_write(destination, raw)
    reused = sum(reused for _, reused in results)
    return {
        "count": len(records),
        "manifest": str(destination),
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "generated": len(records) - reused,
        "reused": reused,
        "config_fingerprint": config.fingerprint,
        "rtv": config.as_dict(),
    }
