"""Existing TAR/TFRecord/ordinary images with content-checked RTV targets.

This protocol is intentionally distinct from paired full-resolution RTV data:
EXIF-oriented RGB is resized to the square training size using PIL BICUBIC,
then the actual RTV solver runs on the complete, unmasked image. Only a shared
horizontal flip and a fresh hole mask are applied after caching. No archive
extraction, image download, ordinary-blur target, or unrestricted pickle is used.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import stat
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL import __version__ as pillow_version
from torch.utils.data import Dataset

from .data import _atomic_write
from .masks import center_mask, random_mask, sample_seed
from .rtv import RTV_VERSION, RTVConfig, rtv_smooth

INDEXED_RTV_PROTOCOL = "medfe_indexed_exif_rgb_bicubic_square_then_rtv_v1"
MAX_RECORD_BYTES = 64 * 1024**2


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"Invalid indexed {name}: expected integer >= {minimum}")
    return value


def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    for count in range(10):
        if position >= len(data):
            raise ValueError("Truncated TFRecord protobuf varint")
        byte = data[position]
        position += 1
        if count == 9 and byte > 1:
            raise ValueError("Oversized TFRecord protobuf varint")
        value |= (byte & 127) << (7 * count)
        if byte < 128:
            return value, position
    raise ValueError("Invalid TFRecord protobuf varint")


def _fields(data: bytes):
    position = 0
    while position < len(data):
        tag, position = _varint(data, position)
        field, wire = tag >> 3, tag & 7
        if field < 1:
            raise ValueError("Invalid TFRecord protobuf field number")
        if wire == 0:
            value, position = _varint(data, position)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, position = _varint(data, position)
            else:
                size = 8 if wire == 1 else 4
            end = position + size
            if end > len(data):
                raise ValueError("Truncated TFRecord protobuf field")
            value, position = data[position:end], end
        else:
            raise ValueError(f"Unsupported TFRecord protobuf wire type {wire}")
        yield field, value


def _one_bytes(data: bytes, number: int) -> bytes:
    values = [value for field, value in _fields(data) if field == number]
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError(f"Expected one bytes/message protobuf field {number}")
    return values[0]


class _BytesOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("TFRecord pickle globals are forbidden")


def decode_tfrecord_image(payload: bytes) -> bytes:
    """Decode the established bytes-only state_dict/raw_image feature safely."""
    features = _one_bytes(payload, 1)
    matches = []
    for field, entry in _fields(features):
        if field != 1 or not isinstance(entry, bytes):
            raise ValueError("Invalid TFRecord Features map")
        if _one_bytes(entry, 1) == b"state_dict":
            feature = _one_bytes(entry, 2)
            matches.append(_one_bytes(_one_bytes(feature, 1), 1))
    if len(matches) != 1:
        raise ValueError("Expected one TFRecord state_dict feature")
    stream = io.BytesIO(matches[0])
    try:
        state = _BytesOnlyUnpickler(stream).load()
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid restricted TFRecord pickle: {error}") from error
    if stream.read(1) or not isinstance(state, dict):
        raise ValueError("TFRecord state_dict must be one complete dictionary")
    image = state.get("raw_image")
    if type(image) is not bytes or not 0 < len(image) <= MAX_RECORD_BYTES:
        raise ValueError("TFRecord raw_image must be bounded nonempty bytes")
    return image


@dataclass(frozen=True, slots=True)
class _Record:
    source: int
    format: str
    offset: int
    length: int
    name: str


def _stamp(path: str) -> tuple[int, int]:
    metadata = os.stat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Indexed source is not a regular file: {path}")
    return metadata.st_size, metadata.st_mtime_ns


def _index_fields(raw, number: int) -> tuple[str, str, int, int, str]:
    """Validate one row identically in the metadata and record-building passes."""
    if isinstance(raw, list) and len(raw) == 3:
        path, offset, length = raw
        image_format, name = "tfrecord", ""
        offset = _integer(offset, "TFRecord offset")
        length = _integer(length, "TFRecord length", 16)
        if length > MAX_RECORD_BYTES + 16:
            raise ValueError("TFRecord index exceeds the record size limit")
    elif isinstance(raw, dict) and raw.get("format") == "tar":
        if set(raw) != {"format", "path", "offset", "length", "name"}:
            raise ValueError(f"Unexpected TAR index keys on line {number}")
        path, image_format = raw["path"], "tar"
        offset, length = (
            _integer(raw["offset"], "TAR offset"),
            _integer(raw["length"], "TAR length", 1),
        )
        name = raw["name"]
        if not isinstance(name, str) or not name or length > MAX_RECORD_BYTES:
            raise ValueError("Invalid TAR member name or length")
    elif isinstance(raw, dict) and raw.get("format") == "image":
        if set(raw) != {"format", "path"}:
            raise ValueError(f"Unexpected image index keys on line {number}")
        path, image_format = raw["path"], "image"
        offset, length, name = 0, 0, ""
    else:
        raise ValueError(f"Expected TFRecord list, TAR or image dictionary on line {number}")
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError(f"Indexed source path must be absolute on line {number}")
    return os.path.normpath(path), image_format, offset, length, name


def _source_metadata(manifest: Path) -> tuple[dict[str, tuple[int, int]], bytes]:
    """Retain bounded parallel filesystem results instead of relying on its cache.

    Only distinct paths are retained, not the potentially millions of JSON
    records. Executor.map returns results (and raises errors) in input order,
    independent of completion order. The second pass verifies exact manifest
    bytes before accepting the dataset.
    """
    paths: dict[str, None] = {}
    digest = hashlib.sha256()
    with manifest.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if line.strip():
                path, *_ = _index_fields(json.loads(line), number)
                paths.setdefault(path, None)
    if not paths:
        return {}, digest.digest()
    with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
        metadata = dict(zip(paths, pool.map(_stamp, paths), strict=True))
    return metadata, digest.digest()


class IndexedRTVDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        seed: int = 2019,
        training: bool = True,
        mask_mode: str = "random",
        min_fraction: float = 0.1,
        max_fraction: float = 0.4,
        cache_root: str | Path = "data/indexed-rtv-cache",
        allow_repeated_records: bool = False,
    ):
        if type(image_size) is not int or image_size < 1:
            raise ValueError("image_size must be a positive integer")
        if type(seed) is not int or seed < 0 or type(training) is not bool:
            raise ValueError("seed must be nonnegative and training must be bool")
        if type(allow_repeated_records) is not bool:
            raise ValueError("allow_repeated_records must be a boolean")
        if mask_mode not in {"random", "center"}:
            raise ValueError("Indexed RTV supports random or center masks")
        if not all(math.isfinite(value) for value in (min_fraction, max_fraction)) or not (
            0 < min_fraction <= max_fraction < 1
        ):
            raise ValueError("Require 0 < min_fraction <= max_fraction < 1")
        self.manifest = Path(manifest).resolve()
        self.image_size, self.seed, self.training = image_size, seed, training
        self.mask_mode, self.min_fraction, self.max_fraction = mask_mode, min_fraction, max_fraction
        self.cache_root = Path(cache_root).resolve()
        self.allow_repeated_records = allow_repeated_records
        self.rtv_config = RTVConfig()
        self.protocol = {
            "protocol": INDEXED_RTV_PROTOCOL,
            "rtv_algorithm": RTV_VERSION,
            "rtv_parameters": self.rtv_config.as_dict(),
            "image_size": image_size,
            "resize": "EXIF transpose, RGB uint8, square PIL BICUBIC, float32 / 255",
            "pillow_version": pillow_version,
        }
        protocol_bytes = json.dumps(self.protocol, sort_keys=True).encode()
        self.protocol_fingerprint = hashlib.sha256(protocol_bytes).hexdigest()
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.records: list[_Record] = []
        self.sources: list[tuple[str, int, int]] = []
        metadata, scanned_manifest_hash = _source_metadata(self.manifest)
        source_ids, seen = {}, {}
        manifest_hash = hashlib.sha256()
        with self.manifest.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                manifest_hash.update(line)
                if not line.strip():
                    continue
                path, image_format, offset, length, name = _index_fields(json.loads(line), number)
                if path not in source_ids:
                    try:
                        size, mtime = metadata[path]
                    except KeyError as error:
                        raise ValueError(
                            "Indexed manifest changed while collecting source metadata"
                        ) from error
                    source_ids[path] = len(self.sources)
                    self.sources.append((path, size, mtime))
                source_id = source_ids[path]
                if image_format == "image":
                    length = self.sources[source_id][1]
                    if not 0 < length <= MAX_RECORD_BYTES:
                        raise ValueError("Image file must be nonempty and within record size limit")
                if offset + length > self.sources[source_id][1]:
                    raise ValueError(f"Indexed record extends beyond source on line {number}")
                identity = (source_id, offset)
                record = _Record(source_id, image_format, offset, length, name)
                if identity in seen:
                    if not allow_repeated_records:
                        raise ValueError(f"Duplicate indexed image on line {number}")
                    if seen[identity] != record:
                        raise ValueError(f"Conflicting repeated indexed image on line {number}")
                else:
                    seen[identity] = record
                # Repeated rows are intentional sampling weights, not duplicate
                # storage. Their distinct indices retain independent mask seeds.
                self.records.append(record)
        if manifest_hash.digest() != scanned_manifest_hash:
            raise ValueError("Indexed manifest changed while collecting source metadata")
        if not self.records:
            raise ValueError("Indexed manifest contains no images")
        self.manifest_sha256 = manifest_hash.hexdigest()
        identity = {
            "manifest_sha256": self.manifest_sha256,
            "protocol": self.protocol,
            "sources": self.sources,
        }
        if allow_repeated_records:
            # Preserve the exact historical fingerprint when repeats are off.
            # Sampling policy does not change the canonical RTV cache protocol.
            identity["allow_repeated_records"] = True
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self._epoch.fill_(epoch)

    def _read_image(self, record: _Record) -> tuple[bytes, str]:
        path, size, mtime = self.sources[record.source]
        expected = (size, mtime)
        if _stamp(path) != expected:
            raise ValueError(f"Indexed source changed since dataset initialization: {path}")
        with open(path, "rb") as stream:
            stream.seek(record.offset)
            contents = stream.read(record.length)
            current = os.fstat(stream.fileno())
        if (current.st_size, current.st_mtime_ns) != expected:
            raise ValueError(f"Indexed source changed while reading: {path}")
        if len(contents) != record.length:
            raise ValueError(f"Truncated indexed image: {path}:{record.offset}")
        if record.format == "image":
            return contents, path
        identifier = f"{path}:{record.offset}"
        if record.format == "tfrecord":
            # Header: uint64 length + uint32 masked CRC; footer: uint32 masked CRC.
            # Match indexed framing exactly. This legacy reader does not certify CRCs.
            length = struct.unpack("<Q", contents[:8])[0]
            if length + 16 != record.length or length > MAX_RECORD_BYTES:
                raise ValueError("TFRecord header length does not match its index")
            return decode_tfrecord_image(contents[12:-4]), identifier
        return contents, identifier + ":" + record.name

    def _target(self, image: np.ndarray) -> np.ndarray:
        source_hash = hashlib.sha256(image.tobytes(order="C")).hexdigest()
        directory = self.cache_root / self.protocol_fingerprint / source_hash[:2]
        path = directory / (source_hash + ".npy")
        metadata_path = path.with_suffix(".json")
        expected = {
            "protocol_fingerprint": self.protocol_fingerprint,
            "canonical_rgb_sha256": source_hash,
            "shape": list(image.shape),
            "dtype": "float32",
            "storage": "NPY RGB [0,1]",
        }
        try:
            metadata = json.loads(metadata_path.read_text())
            content = path.read_bytes()
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("RTV cache provenance mismatch")
            if hashlib.sha256(content).hexdigest() != metadata["structure_sha256"]:
                raise ValueError("RTV cache content hash mismatch")
            target = np.load(io.BytesIO(content), allow_pickle=False)
            if (
                target.shape != image.shape
                or target.dtype != np.float32
                or not np.isfinite(target).all()
                or target.min() < 0
                or target.max() > 1
            ):
                raise ValueError("Invalid RTV cache tensor")
            return target
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            target = rtv_smooth(image, self.rtv_config).astype(np.float32)
            contents = io.BytesIO()
            np.save(contents, target, allow_pickle=False)
            encoded = contents.getvalue()
            metadata = {
                **expected,
                "protocol": self.protocol,
                "structure_sha256": hashlib.sha256(encoded).hexdigest(),
            }
            _atomic_write(path, encoded)
            _atomic_write(metadata_path, (json.dumps(metadata, sort_keys=True) + "\n").encode())
            return target

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.records):
            raise IndexError(index)
        index = int(index)
        content, identifier = self._read_image(self.records[index])
        with Image.open(io.BytesIO(content)) as opened:
            resized = (
                ImageOps.exif_transpose(opened)
                .convert("RGB")
                .resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            )
            image = np.asarray(resized, dtype=np.float32) / 255.0
        structure = self._target(image)
        epoch = self.epoch if self.training else 0
        rng = np.random.default_rng(sample_seed(self.seed, epoch, index, "transform"))
        if self.training and bool(rng.integers(2)):
            image, structure = image[:, ::-1].copy(), structure[:, ::-1].copy()
        else:
            image, structure = image.copy(), structure.copy()
        if self.mask_mode == "random":
            mask = random_mask(
                self.image_size,
                self.image_size,
                sample_seed(self.seed, epoch, index, "mask"),
                self.min_fraction,
                self.max_fraction,
            )
        else:
            mask = center_mask(self.image_size, self.image_size)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1) * 2 - 1,
            "structure": torch.from_numpy(structure).permute(2, 0, 1) * 2 - 1,
            "mask": mask,
            "id": identifier,
            "index": index,
        }
