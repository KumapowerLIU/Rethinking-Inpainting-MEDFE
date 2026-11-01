import hashlib
import io
import json
import os
import pickle
import struct
import tarfile
import threading

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import medfe.indexed_data as indexed
from medfe.indexed_data import IndexedRTVDataset, decode_tfrecord_image
from medfe.rtv import rtv_smooth


def varint(value):
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    return bytes(output + bytes([value]))


def field(number, value):
    return varint(number << 3 | 2) + varint(len(value)) + value


def example(state):
    feature = field(1, field(1, pickle.dumps(state, protocol=4)))
    entry = field(1, b"state_dict") + field(2, feature)
    return field(1, field(1, entry))


def png_bytes(index=0):
    pixels = np.random.default_rng(index).integers(0, 256, (11, 17, 3), dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def write_tf(path, payload):
    # Framing matches TFRecord; CRC certification is not part of this legacy reader.
    content = struct.pack("<Q", len(payload)) + bytes(4) + payload + bytes(4)
    path.write_bytes(content)
    return [str(path), 0, len(content)]


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def mixed_manifest(tmp_path):
    first = write_tf(tmp_path / "images.tfrecord", example({"raw_image": png_bytes(0)}))
    archive = tmp_path / "images.tar"
    with tarfile.open(archive, "w") as stream:
        for index in (1, 2):
            payload = png_bytes(index)
            info = tarfile.TarInfo(f"category/{index}.png")
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
    rows = [first]
    with tarfile.open(archive) as stream:
        for member in stream:
            rows.append(
                {
                    "format": "tar",
                    "path": str(archive),
                    "offset": member.offset_data,
                    "length": member.size,
                    "name": member.name,
                }
            )
    return write_manifest(tmp_path / "mixed.jsonl", rows)


def dataset(manifest, **kwargs):
    return IndexedRTVDataset(
        manifest, image_size=16, cache_root=manifest.parent / "cache", **kwargs
    )


def test_real_tfrecord_tar_and_canonical_rtv_are_correct(mixed_manifest):
    values = dataset(mixed_manifest, training=False)
    assert len(values) == 3 and len(values.fingerprint) == 64
    for index in range(3):
        result = values[index]
        with Image.open(io.BytesIO(png_bytes(index))) as image:
            expected = (
                np.asarray(image.resize((16, 16), Image.Resampling.BICUBIC), dtype=np.float32) / 255
            )
        target = rtv_smooth(expected).astype(np.float32)
        torch.testing.assert_close(
            result["image"], torch.from_numpy(expected).permute(2, 0, 1) * 2 - 1, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result["structure"], torch.from_numpy(target).permute(2, 0, 1) * 2 - 1, rtol=0, atol=0
        )
        assert result["mask"].dtype == torch.bool
        assert result["index"] == index and result["id"]
    assert len(list((mixed_manifest.parent / "cache").rglob("*.npy"))) == 3


def test_cache_reuse_corruption_and_metadata_hash_are_checked(mixed_manifest, monkeypatch):
    original = indexed.rtv_smooth
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(indexed, "rtv_smooth", counted)
    values = dataset(mixed_manifest, training=False)
    first = values[0]
    assert len(calls) == 1
    assert torch.equal(values[0]["structure"], first["structure"])
    assert len(calls) == 1
    cache = next((mixed_manifest.parent / "cache").rglob("*.npy"))
    cache.write_bytes(b"truncated-cache")
    assert torch.equal(values[0]["structure"], first["structure"])
    assert len(calls) == 2
    metadata = json.loads(cache.with_suffix(".json").read_text())
    assert metadata["structure_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    metadata["shape"] = [1, 2, 3]
    cache.with_suffix(".json").write_text(json.dumps(metadata))
    values[0]
    assert len(calls) == 3
    values.set_epoch(5)
    values[0]
    assert len(calls) == 3  # No per-epoch copy of the same canonical target.


def test_epoch_flip_is_synchronized_and_validation_is_fixed(mixed_manifest):
    values = dataset(mixed_manifest)
    fixed = dataset(mixed_manifest, training=False)
    canonical = fixed[0]
    before = values[0]
    found = set()
    for epoch in range(8):
        values.set_epoch(epoch)
        result = values[0]
        flipped = torch.equal(result["image"], canonical["image"].flip(-1))
        found.add(flipped)
        expected = canonical["structure"].flip(-1) if flipped else canonical["structure"]
        assert torch.equal(result["structure"], expected)
        fixed.set_epoch(epoch)
        assert torch.equal(fixed[0]["image"], canonical["image"])
        assert torch.equal(fixed[0]["mask"], canonical["mask"])
    assert found == {False, True}
    assert not torch.equal(before["mask"], values[0]["mask"])


def test_workers_and_persistent_epoch_state_match(mixed_manifest):
    values = dataset(mixed_manifest)
    loader = DataLoader(values, batch_size=2, num_workers=2, persistent_workers=True)
    try:
        for epoch in (0, 2):
            values.set_epoch(epoch)
            batches = list(loader)
            for index in range(3):
                actual = values[index]
                for key in ("image", "structure", "mask"):
                    assert torch.equal(batches[index // 2][key][index % 2], actual[key])
    finally:
        del loader


def test_source_mutation_is_rejected_and_changes_new_fingerprint(mixed_manifest):
    values = dataset(mixed_manifest)
    path = mixed_manifest.parent / "images.tfrecord"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="source changed"):
        values[0]
    assert dataset(mixed_manifest).fingerprint != values.fingerprint


def test_fingerprint_includes_protocol_size_and_manifest(mixed_manifest, monkeypatch):
    first = dataset(mixed_manifest)
    other = IndexedRTVDataset(
        mixed_manifest, image_size=24, cache_root=mixed_manifest.parent / "cache"
    )
    assert other.fingerprint != first.fingerprint
    monkeypatch.setattr(indexed, "INDEXED_RTV_PROTOCOL", "next_protocol")
    assert dataset(mixed_manifest).fingerprint != first.fingerprint
    mixed_manifest.write_bytes(mixed_manifest.read_bytes() + b"\n")
    assert dataset(mixed_manifest).fingerprint != first.fingerprint


def test_restricted_pickle_forbids_globals_and_invalid_content():
    with pytest.raises(ValueError, match="globals are forbidden"):
        decode_tfrecord_image(example({"raw_image": Exception("must not construct")}))
    for state in ({"raw_image": "text"}, {}, [], {"raw_image": b""}):
        with pytest.raises(ValueError):
            decode_tfrecord_image(example(state))


@pytest.mark.parametrize(
    "payload", [b"", b"\x00", b"\x0a\xff", b"\x0a\x20x", b"\x0b", bytes([128]) * 11]
)
def test_malformed_protobuf_is_rejected(payload):
    with pytest.raises(ValueError):
        decode_tfrecord_image(payload)


def test_duplicate_and_out_of_bounds_records_rejected(mixed_manifest):
    rows = [json.loads(line) for line in mixed_manifest.read_text().splitlines()]
    with pytest.raises(ValueError, match="Duplicate"):
        dataset(write_manifest(mixed_manifest, [rows[0], rows[0]]))
    broken = list(rows[0])
    broken[2] += 1
    with pytest.raises(ValueError, match="beyond source"):
        dataset(write_manifest(mixed_manifest, [broken]))
    with pytest.raises(ValueError, match="absolute"):
        dataset(write_manifest(mixed_manifest, [["relative.tfrecord", 0, 16]]))


def test_header_length_mismatch_and_partial_payload_rejected(tmp_path):
    row = write_tf(tmp_path / "bad.tfrecord", example({"raw_image": png_bytes()}))
    contents = bytearray((tmp_path / "bad.tfrecord").read_bytes())
    contents[:8] = struct.pack("<Q", 1)
    (tmp_path / "bad.tfrecord").write_bytes(contents)
    values = dataset(write_manifest(tmp_path / "bad.jsonl", [row]))
    with pytest.raises(ValueError, match="header length"):
        values[0]


def test_returned_tensors_cannot_mutate_cache(mixed_manifest):
    values = dataset(mixed_manifest, training=False)
    original = values[0]["structure"].clone()
    values[0]["structure"].fill_(0)
    assert torch.equal(values[0]["structure"], original)


def test_csa_image_record_matches_archive_pixels_and_canonical_rtv(mixed_manifest):
    file_path = mixed_manifest.parent / "face.png"
    file_path.write_bytes(png_bytes(1))
    manifest = write_manifest(
        mixed_manifest.parent / "face.jsonl", [{"format": "image", "path": str(file_path)}]
    )
    images = dataset(manifest, training=False)
    archive = dataset(mixed_manifest, training=False)
    actual, expected = images[0], archive[1]
    assert actual["id"] == str(file_path)
    for key in ("image", "structure"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert len(list((manifest.parent / "cache").rglob("*.npy"))) == 1


def test_explicit_repeats_preserve_rows_masks_and_share_rtv_cache(tmp_path, monkeypatch):
    image = tmp_path / "face.png"
    image.write_bytes(png_bytes(5))
    row = {"format": "image", "path": str(image)}
    manifest = write_manifest(tmp_path / "repeated.jsonl", [row, row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        dataset(manifest)
    calls = []
    original = indexed.rtv_smooth

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(indexed, "rtv_smooth", counted)
    values = dataset(manifest, allow_repeated_records=True)
    assert len(values) == 3 and len(values.sources) == 1
    assert values.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    results = [values[index] for index in range(3)]
    assert [result["index"] for result in results] == [0, 1, 2]
    assert len({result["id"] for result in results}) == 1
    assert len(calls) == 1
    assert not torch.equal(results[0]["mask"], results[1]["mask"])
    for index, result in enumerate(results):
        expected_mask = indexed.random_mask(
            16, 16, indexed.sample_seed(2019, 0, index, "mask"), 0.1, 0.4
        )
        assert torch.equal(result["mask"], expected_mask)
    assert len(list((tmp_path / "cache").rglob("*.npy"))) == 1


def test_repeats_option_keeps_old_fingerprint_and_cache_protocol(mixed_manifest):
    default = dataset(mixed_manifest)
    disabled = dataset(mixed_manifest, allow_repeated_records=False)
    enabled = dataset(mixed_manifest, allow_repeated_records=True)
    old_identity = {
        "manifest_sha256": default.manifest_sha256,
        "protocol": default.protocol,
        "sources": default.sources,
    }
    historical = hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
    assert default.fingerprint == disabled.fingerprint == historical
    assert enabled.fingerprint != historical
    assert enabled.protocol_fingerprint == default.protocol_fingerprint


@pytest.mark.parametrize("field", ["length", "name", "format"])
def test_repeated_record_conflicts_are_rejected(mixed_manifest, field):
    rows = [json.loads(line) for line in mixed_manifest.read_text().splitlines()]
    original = rows[1]
    changed = dict(original)
    if field == "length":
        changed["length"] -= 1
    elif field == "name":
        changed["name"] = "conflicting/name.png"
    else:
        original = {"format": "image", "path": original["path"]}
        changed["offset"] = 0
    with pytest.raises(ValueError, match="Conflicting repeated"):
        dataset(write_manifest(mixed_manifest, [original, changed]), allow_repeated_records=True)


def test_csa_image_file_mutation_is_rejected(tmp_path):
    image = tmp_path / "face.png"
    image.write_bytes(png_bytes(3))
    manifest = write_manifest(tmp_path / "image.jsonl", [{"format": "image", "path": str(image)}])
    values = dataset(manifest)
    old = image.stat()
    os.utime(image, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="source changed"):
        values[0]


@pytest.mark.parametrize("kind", ["extra_key", "relative", "empty", "large", "directory"])
def test_csa_image_records_are_strict_and_bounded(tmp_path, monkeypatch, kind):
    image = tmp_path / "face.png"
    image.write_bytes(png_bytes())
    row = {"format": "image", "path": str(image)}
    if kind == "extra_key":
        row["offset"] = 0
    elif kind == "relative":
        row["path"] = "face.png"
    elif kind == "empty":
        image.write_bytes(b"")
    elif kind == "large":
        monkeypatch.setattr(indexed, "MAX_RECORD_BYTES", 4)
    else:
        row["path"] = str(tmp_path)
    with pytest.raises(ValueError):
        dataset(write_manifest(tmp_path / "invalid.jsonl", [row]))


@pytest.mark.parametrize("value", [1, "true", None])
def test_repeat_policy_requires_boolean(mixed_manifest, value):
    with pytest.raises(ValueError, match="boolean"):
        dataset(mixed_manifest, allow_repeated_records=value)


def test_parallel_metadata_preserves_serial_source_order_and_fingerprint(tmp_path, monkeypatch):
    paths = [tmp_path / f"face-{index}.png" for index in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(png_bytes(index))
    order = [paths[1], paths[0], paths[1], paths[2]]
    manifest = write_manifest(
        tmp_path / "parallel.jsonl", [{"format": "image", "path": str(p)} for p in order]
    )
    original_stamp = indexed._stamp
    expected_sources = [(str(path), *original_stamp(str(path))) for path in dict.fromkeys(order)]
    barrier = threading.Barrier(3)
    calls, lock = [], threading.Lock()

    def concurrent_stamp(path):
        with lock:
            calls.append(path)
        barrier.wait(timeout=5)
        return original_stamp(path)

    monkeypatch.setattr(indexed, "_stamp", concurrent_stamp)
    values = dataset(manifest, allow_repeated_records=True)
    assert sorted(calls) == sorted(str(path) for path in paths)
    assert values.sources == expected_sources
    assert [record.source for record in values.records] == [0, 1, 0, 2]
    expected_identity = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "protocol": values.protocol,
        "sources": expected_sources,
        "allow_repeated_records": True,
    }
    expected_hash = hashlib.sha256(
        json.dumps(expected_identity, sort_keys=True).encode()
    ).hexdigest()
    assert values.fingerprint == expected_hash


@pytest.mark.parametrize("change", ["whitespace", "new_source"])
def test_manifest_mutation_between_metadata_and_records_is_rejected(tmp_path, monkeypatch, change):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first.write_bytes(png_bytes(0))
    second.write_bytes(png_bytes(1))
    manifest = write_manifest(tmp_path / "mutable.jsonl", [{"format": "image", "path": str(first)}])
    original = indexed._source_metadata

    def mutate_after_metadata(path):
        result = original(path)
        if change == "whitespace":
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            write_manifest(path, [{"format": "image", "path": str(second)}])
        return result

    monkeypatch.setattr(indexed, "_source_metadata", mutate_after_metadata)
    with pytest.raises(ValueError, match="manifest changed"):
        dataset(manifest)


def test_stamp_checks_regular_file_with_one_stat(tmp_path, monkeypatch):
    image = tmp_path / "face.png"
    image.write_bytes(png_bytes())
    original_stat = os.stat
    expected = original_stat(image)
    calls = []

    def counted(path, *args, **kwargs):
        calls.append(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(indexed.os, "stat", counted)
    assert indexed._stamp(str(image)) == (expected.st_size, expected.st_mtime_ns)
    assert calls == [str(image)]
