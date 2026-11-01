import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from medfe.data import InpaintingDataset, prepare_rtv_dataset
from medfe.rtv import RTVConfig


def image_file(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path)
    return path


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def paired_manifest(tmp_path):
    y, x = np.indices((24, 40))
    pixels = np.stack((x * 6, y * 10, ((x + y) % 2) * 255), axis=-1).astype(np.uint8)
    rows = []
    for index in range(4):
        path = image_file(tmp_path / f"image-{index}.png", pixels)
        # Equal image and structure make any unsynchronized augmentation visible.
        rows.append(
            dict(
                id=f"id-{index}",
                image=path.name,
                structure=path.name,
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return write_manifest(tmp_path / "pairs.jsonl", rows)


def test_paired_transform_stays_aligned_and_preserves_targets(paired_manifest):
    dataset = InpaintingDataset(paired_manifest, image_size=16)
    first = dataset[0]
    assert first["image"].shape == first["structure"].shape == (3, 16, 16)
    assert first["mask"].shape == (1, 16, 16)
    assert first["mask"].dtype == torch.bool
    assert first["image"].dtype == torch.float32
    assert first["image"].min() >= -1 and first["image"].max() <= 1
    assert torch.equal(first["image"], first["structure"])
    assert first["id"] == "id-0" and first["index"] == 0
    assert dataset.fingerprint == hashlib.sha256(paired_manifest.read_bytes()).hexdigest()
    assert torch.equal(first["image"], dataset[0]["image"])
    assert torch.equal(first["mask"], dataset[0]["mask"])
    assert not torch.equal(first["mask"], dataset[1]["mask"])
    first["image"].fill_(0)
    assert not torch.equal(first["image"], dataset[0]["image"])


def test_validation_is_fixed_and_training_changes_with_epoch(paired_manifest):
    train = InpaintingDataset(paired_manifest, image_size=16)
    valid = InpaintingDataset(paired_manifest, image_size=16, training=False)
    train_before, valid_before = train[0], valid[0]
    train.set_epoch(4)
    valid.set_epoch(4)
    assert not torch.equal(train_before["mask"], train[0]["mask"])
    assert torch.equal(valid_before["mask"], valid[0]["mask"])
    assert torch.equal(valid_before["image"], valid[0]["image"])


def test_workers_and_persistent_worker_epochs_match_direct_reads(paired_manifest):
    dataset = InpaintingDataset(paired_manifest, image_size=16)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True)
    try:
        for epoch in (0, 2):
            dataset.set_epoch(epoch)
            batches = list(loader)
            for index in range(len(dataset)):
                batch, offset = batches[index // 2], index % 2
                expected = dataset[index]
                assert torch.equal(batch["image"][offset], expected["image"])
                assert torch.equal(batch["structure"][offset], expected["structure"])
                assert torch.equal(batch["mask"][offset], expected["mask"])
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_file_mask_geometry_and_nearest_binary_resize(tmp_path):
    y, x = np.indices((12, 20))
    values = (((x // 2 + y // 2) % 2) * 255).astype(np.uint8)
    path = image_file(tmp_path / "rgb.png", np.repeat(values[..., None], 3, axis=-1))
    mask = image_file(tmp_path / "mask.png", values)
    manifest = write_manifest(
        tmp_path / "pairs.jsonl",
        [dict(id="one", image=path.name, structure=path.name, mask=mask.name)],
    )
    dataset = InpaintingDataset(manifest, image_size=12, mask_mode="file")
    for epoch in range(4):
        dataset.set_epoch(epoch)
        item = dataset[0]
        # No resize is needed: exact RGB bits must still match its paired mask
        # after the independently sampled crop and optional flip.
        assert torch.equal(item["mask"], item["image"][:1] > 0)
        assert item["mask"].dtype == torch.bool
    resized = InpaintingDataset(manifest, image_size=18, mask_mode="file")[0]
    assert set(resized["mask"].unique().tolist()) <= {False, True}


def test_structure_npy_keeps_float_values(tmp_path):
    image_file(tmp_path / "rgb.png", np.full((9, 9, 3), 100, np.uint8))
    structure = np.full((9, 9, 3), 0.12345678, np.float32)
    np.save(tmp_path / "structure.npy", structure)
    manifest = write_manifest(
        tmp_path / "pairs.jsonl", [dict(id="npy", image="rgb.png", structure="structure.npy")]
    )
    item = InpaintingDataset(manifest, image_size=9)[0]
    torch.testing.assert_close(
        item["structure"], torch.full((3, 9, 9), 0.12345678 * 2 - 1), atol=1e-7, rtol=0
    )


def test_manifest_rejects_duplicate_ids_and_missing_file_masks(tmp_path):
    row = dict(id="same", image="a.png", structure="b.png")
    path = write_manifest(tmp_path / "pairs.jsonl", [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        InpaintingDataset(path)
    write_manifest(path, [row])
    with pytest.raises(ValueError, match="requires a mask"):
        InpaintingDataset(path, mask_mode="file")


def test_changed_image_contents_fail_hash_before_returning_a_sample(paired_manifest):
    dataset = InpaintingDataset(paired_manifest, image_size=16)
    image_file(paired_manifest.parent / "image-0.png", np.zeros((24, 40, 3), np.uint8))
    with pytest.raises(ValueError, match="hash mismatch"):
        dataset[0]


def test_bad_pair_dimensions_are_not_silently_resized(tmp_path):
    image_file(tmp_path / "a.png", np.zeros((8, 9, 3), np.uint8))
    image_file(tmp_path / "b.png", np.zeros((9, 8, 3), np.uint8))
    path = write_manifest(
        tmp_path / "pairs.jsonl", [dict(id="mismatch", image="a.png", structure="b.png")]
    )
    with pytest.raises(ValueError, match="dimensions differ"):
        InpaintingDataset(path)[0]


def test_preparation_cache_is_content_addressed_and_reproducible(tmp_path):
    images, cache, manifest = tmp_path / "images", tmp_path / "cache", tmp_path / "pairs.jsonl"
    image_file(
        images / "nested/a.png",
        np.random.default_rng(2).integers(0, 256, (7, 9, 3), dtype=np.uint8),
    )
    image_file(images / "b.jpg", np.full((8, 6, 3), 87, np.uint8))
    first = prepare_rtv_dataset(images, cache, manifest)
    raw = manifest.read_bytes()
    assert first["count"] == first["generated"] == 2 and first["reused"] == 0
    second = prepare_rtv_dataset(images, cache, manifest)
    assert second["reused"] == 2 and second["generated"] == 0
    assert manifest.read_bytes() == raw and second["fingerprint"] == first["fingerprint"]
    dataset = InpaintingDataset(manifest, image_size=8)
    assert dataset[0]["structure"].shape == (3, 8, 8)
    image_file(images / "b.jpg", np.full((8, 6, 3), 120, np.uint8))
    changed = prepare_rtv_dataset(images, cache, manifest)
    assert changed["generated"] == 1 and changed["reused"] == 1
    assert changed["fingerprint"] != first["fingerprint"]
    changed_parameters = prepare_rtv_dataset(images, cache, manifest, RTVConfig(sigma=2))
    assert changed_parameters["generated"] == 2
    assert changed_parameters["config_fingerprint"] != first["config_fingerprint"]


def test_corrupt_cache_is_rebuilt_and_manifest_remains_valid(tmp_path):
    images, cache, manifest = tmp_path / "images", tmp_path / "cache", tmp_path / "pairs.jsonl"
    image_file(images / "one.png", np.full((6, 8, 3), 100, np.uint8))
    prepare_rtv_dataset(images, cache, manifest)
    row = json.loads(manifest.read_text())
    target = (manifest.parent / row["structure"]).resolve()
    target.write_bytes(b"broken-npy")
    result = prepare_rtv_dataset(images, cache, manifest)
    assert result["generated"] == 1 and result["reused"] == 0
    assert torch.isfinite(InpaintingDataset(manifest, image_size=8)[0]["structure"]).all()


def test_failed_preparation_does_not_replace_existing_manifest(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "invalid.png").write_bytes(b"not-an-image")
    manifest = tmp_path / "existing.jsonl"
    manifest.write_bytes(b"existing evidence\n")
    with pytest.raises(OSError):
        prepare_rtv_dataset(images, tmp_path / "cache", manifest)
    assert manifest.read_bytes() == b"existing evidence\n"


def test_parallel_preparation_is_sorted_and_loadable(tmp_path):
    images = tmp_path / "images"
    for index in range(3):
        image_file(images / f"{index}.png", np.full((6, 7, 3), 30 + index * 50, np.uint8))
    manifest = tmp_path / "pairs.jsonl"
    result = prepare_rtv_dataset(images, tmp_path / "cache", manifest, workers=2)
    assert result["count"] == 3
    assert [row["id"] for row in map(json.loads, manifest.read_text().splitlines())] == [
        "0.png",
        "1.png",
        "2.png",
    ]
    assert len(InpaintingDataset(manifest, image_size=8)) == 3
