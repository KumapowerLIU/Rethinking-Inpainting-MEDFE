import hashlib
import json
import os
import signal
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from medfe.checkpoint import load_checkpoint
from medfe.config import Config
from medfe.data import InpaintingDataset
from medfe.runner import make_dataset, make_loader, run_lock, train


def small_config(tmp_path):
    rows = []
    for index in range(3):
        image = tmp_path / f"{index}.png"
        Image.fromarray(
            np.random.default_rng(index).integers(
                0,
                256,
                (260, 280, 3),
                dtype=np.uint8,
            )
        ).save(image)
        rows.append(
            {
                "id": str(index),
                "image": str(image),
                "structure": str(image),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return Config.from_dict(
        {
            "model": {"base_channels": 2, "residual_blocks": 1, "pconv_layers": 1},
            "data": {"train_manifest": str(manifest), "batch_size": 2, "workers": 0},
            "loss": {"lpips": 0, "perceptual": 0, "style": 0},
            "discriminator": {"global_width": 2, "local_width": 2},
            "train": {
                "output": str(tmp_path / "run"),
                "epochs": 2,
                "device": "cpu",
                "save_every": 1,
            },
        }
    )


def test_loader_resume_cursor_worker_independence_and_rng(tmp_path):
    config = small_config(tmp_path)
    dataset = make_dataset(config)
    initial = torch.get_rng_state().clone()
    batches = list(make_loader(dataset, config, 1))
    assert torch.equal(initial, torch.get_rng_state())
    assert [len(batch["image"]) for batch in batches] == [2, 1]
    config.data.workers = 2
    remaining = list(make_loader(dataset, config, 1, offset=2))
    assert remaining[0]["id"] == batches[1]["id"]
    for key in ("image", "structure", "mask"):
        assert torch.equal(remaining[0][key], batches[1][key])


def test_factory_preserves_paired_backend(tmp_path):
    config = small_config(tmp_path)
    dataset = make_dataset(config)
    assert isinstance(dataset, InpaintingDataset)
    assert dataset.training and len(dataset) == 3


def test_factory_selects_indexed_backend_for_training_and_validation(tmp_path, monkeypatch):
    # The adapter has independent IO/RTV tests. This boundary test verifies the
    # factory selects it and preserves split, transform and cache settings.
    class IndexedProbe:
        def __init__(self, manifest, **kwargs):
            self.manifest, self.settings = manifest, kwargs

    adapter = types.ModuleType("medfe.indexed_data")
    adapter.IndexedRTVDataset = IndexedProbe
    monkeypatch.setitem(sys.modules, "medfe.indexed_data", adapter)
    config = Config.from_dict(
        {
            "data": {
                "format": "indexed_rtv",
                "rtv_cache": str(tmp_path / "cache"),
                "train_manifest": "train-index.jsonl",
                "val_manifest": "val-index.jsonl",
                "min_fraction": 0.2,
                "max_fraction": 0.35,
            },
            "train": {"seed": 71},
        }
    )
    training = make_dataset(config)
    validation = make_dataset(config, validation=True)
    assert isinstance(training, IndexedProbe) and isinstance(validation, IndexedProbe)
    assert training.manifest == "train-index.jsonl"
    assert validation.manifest == "val-index.jsonl"
    assert training.settings["training"] is True
    assert validation.settings["training"] is False
    for dataset in (training, validation):
        assert dataset.settings["cache_root"] == str(tmp_path / "cache")
        assert dataset.settings["seed"] == 71
        assert dataset.settings["image_size"] == 256
        assert dataset.settings["mask_mode"] == "random"
        assert dataset.settings["min_fraction"] == 0.2
        assert dataset.settings["max_fraction"] == 0.35


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_nested_equal(first, second)
    else:
        assert left == right


def test_actual_training_resume_matches_uninterrupted_and_partial_batch(tmp_path):
    config = small_config(tmp_path)
    config.data.val_manifest = config.data.train_manifest
    config.train.evaluate_every = 1
    first = train(config, max_steps=1)
    assert first["step"] == 1 and first["data"]["samples_in_epoch"] == 2
    latest = Path(config.train.output) / "latest.pt"
    resumed = train(config, resume=latest, max_steps=3)
    saved = load_checkpoint(latest)
    assert resumed["step"] == 3
    assert saved["data_progress"]["samples_seen"] == 5
    assert saved["completed_epochs"] == 1
    assert len(list((Path(config.train.output) / "validation").glob("step-*"))) == 3
    config.train.output = str(tmp_path / "continuous")
    config.train.evaluate_every = 10  # Evaluation must not affect the next training update.
    train(config, max_steps=3)
    continuous = load_checkpoint(Path(config.train.output) / "latest.pt")
    for key in (
        "generator",
        "global_d",
        "local_d",
        "g_optimizer",
        "d_optimizer",
        "g_scheduler",
        "d_scheduler",
        "rng",
        "data_progress",
    ):
        assert_nested_equal(saved[key], continuous[key])
    assert (Path(config.train.output) / "source/medfe/model.py").exists()
    assert len(list(Path(config.train.output).glob("*.pt"))) == 1
    with pytest.raises(FileExistsError):
        train(config, max_steps=1)


def test_output_lock_rejects_concurrent_writer(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another trainer"):
            with run_lock(tmp_path):
                pytest.fail("Second writer acquired a lock")


def test_signal_and_worker_failure_save_last_complete_step(tmp_path, monkeypatch):
    import medfe.runner as runner

    config = small_config(tmp_path)
    original = runner.make_loader

    def interrupted_loader(*args, **kwargs):
        yield next(iter(original(*args, **kwargs)))
        os.kill(os.getpid(), signal.SIGTERM)
        raise RuntimeError("Worker exited after process-group termination")

    monkeypatch.setattr(runner, "make_loader", interrupted_loader)
    result = train(config)
    assert result["status"] == "stopped" and result["step"] == 1
    assert load_checkpoint(Path(config.train.output) / "latest.pt")["global_step"] == 1


def test_validation_uses_its_own_batch_size(tmp_path, monkeypatch):
    import medfe.evaluation as evaluation

    config = small_config(tmp_path)
    config.data.val_manifest = config.data.train_manifest
    config.train.evaluate_every = 1
    config.train.evaluation_batch_size = 1
    config.train.evaluation_images = 3
    original, seen = evaluation.evaluate, []

    def record_batch_size(*args, **kwargs):
        seen.append(kwargs["batch_size"])
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate", record_batch_size)
    result = train(config, max_steps=1)
    assert result["step"] == 1 and config.data.batch_size == 2
    assert seen == [1]
    report = Path(config.train.output) / "validation/step-00000001/metrics.json"
    assert json.loads(report.read_text())["images"] == 3


def test_cpu_memory_budget_is_rejected_before_network_allocation(monkeypatch):
    import medfe.trainer as trainer_module

    def unexpected_network(*args, **kwargs):
        pytest.fail("Memory budget must be validated before allocating the generator")

    monkeypatch.setattr(trainer_module, "MEDFEGenerator", unexpected_network)
    with pytest.raises(ValueError, match="[Cc][Uu][Dd][Aa]|memory_budget"):
        config = Config.from_dict({"train": {"device": "cpu", "memory_budget_gib": 32}})
        trainer_module.Trainer(config)


@pytest.mark.parametrize("free_gib,accepted", [(33, False), (34, True)])
def test_cuda_budget_preserves_headroom_before_allocation(monkeypatch, free_gib, accepted):
    import medfe.trainer as trainer_module

    # No CUDA access: stop the constructor before its first model allocation.
    class AllocationBoundary(Exception):
        pass

    calls = []

    def stop_at_allocation(*args, **kwargs):
        raise AllocationBoundary

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_gib * 2**30, 80 * 2**30))
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, device)),
    )
    monkeypatch.setattr(trainer_module, "MEDFEGenerator", stop_at_allocation)
    config = Config.from_dict({"train": {"device": "cuda:0", "memory_budget_gib": 32}})
    if accepted:
        with pytest.raises(AllocationBoundary):
            trainer_module.Trainer(config)
        assert calls == [(0.4, torch.device("cuda:0"))]
    else:
        with pytest.raises(RuntimeError, match="headroom"):
            trainer_module.Trainer(config)
        assert not calls
