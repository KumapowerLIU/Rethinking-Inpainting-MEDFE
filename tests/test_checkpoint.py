import copy
import random

import numpy as np
import pytest
import torch

from medfe.checkpoint import (
    FORMAT,
    SEMANTICS,
    atomic_save,
    capture_rng,
    epoch_indices,
    load_checkpoint,
    restore_rng,
    seed_everything,
    validate_progress,
)
from medfe.config import Config
from medfe.discriminators import SPECTRAL_NORM_POLICY
from medfe.weights import state_dict_sha256


def valid_state():
    models = {name: {"weight": torch.ones(2, 3)} for name in ("generator", "global_d", "local_d")}
    hashes = {name: state_dict_sha256(weights) for name, weights in models.items()}
    return {
        "format": FORMAT,
        "semantics": SEMANTICS,
        "test_fixture": True,
        "global_step": 0,
        "completed_epochs": 0,
        "config": Config().as_dict(),
        **models,
        "model_state_sha256": hashes,
        "global_d_state_sha256": hashes["global_d"],
        "source": {"fixture.py": "a" * 64},
        "data_progress": None,
        "global_d_provenance": {},
        "lpips": None,
        "lpips_provenance": None,
        "vgg": None,
        "vgg_provenance": None,
        "g_optimizer": {},
        "d_optimizer": {},
        "g_scheduler": {},
        "d_scheduler": {},
        "scaler": {},
        "rng": capture_rng(),
        "discriminator_spectral_norm_policy": SPECTRAL_NORM_POLICY,
    }


def test_atomic_checkpoint_and_fixture_gate(tmp_path):
    path = tmp_path / "new/latest.pt"
    atomic_save(valid_state(), path)
    with pytest.raises(ValueError, match="fixture"):
        load_checkpoint(path)
    assert load_checkpoint(path, allow_fixture=True)["global_step"] == 0
    assert not list(path.parent.glob("*.tmp"))


def test_failed_atomic_write_preserves_previous_file_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    atomic_save(valid_state(), path)
    previous = path.read_bytes()

    def fail(state, stream):
        stream.write(b"incomplete")
        raise OSError("disk interrupted")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="interrupted"):
        atomic_save(valid_state(), path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda x: x.update(format="legacy"), "modern"),
        (lambda x: x.update(semantics="other"), "mathematics"),
        (lambda x: x.update(test_fixture=1), "designation"),
        (lambda x: x.update(global_step=True), "step"),
        (lambda x: x.pop("rng"), "Incomplete"),
        (lambda x: x.update(source={"a.py": "wrong"}), "source"),
        (lambda x: x.update(discriminator_spectral_norm_policy="other"), "normalization"),
        (lambda x: x["generator"]["weight"].add_(1), "generator weight hash"),
        (lambda x: x["local_d"]["weight"].fill_(float("nan")), "Non-finite"),
        (
            lambda x: x["g_optimizer"].update(state={0: {"exp_avg": torch.tensor(float("inf"))}}),
            "Non-finite",
        ),
        (lambda x: x.update(global_d_state_sha256="b" * 64), "hashes disagree"),
    ],
)
def test_corrupted_checkpoint_is_rejected(tmp_path, mutation, match):
    state = valid_state()
    mutation(state)
    path = tmp_path / "bad.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match=match):
        load_checkpoint(path, allow_fixture=True)


def test_cursor_validates_partial_final_batch_and_epoch_boundary():
    progress = {
        "fingerprint": "a" * 64,
        "size": 5,
        "epoch": 0,
        "samples_in_epoch": 5,
        "samples_seen": 5,
    }
    validate_progress(progress, epoch=0, step=3, batch_size=2)
    progress.update(epoch=1, samples_in_epoch=0)
    validate_progress(progress, epoch=1, step=3, batch_size=2)
    for field, bad in [
        ("fingerprint", "z" * 64),
        ("size", True),
        ("samples_seen", 4),
        ("samples_in_epoch", 1),
        ("epoch", 2),
    ]:
        invalid = copy.deepcopy(progress)
        invalid[field] = bad
        with pytest.raises(ValueError):
            validate_progress(invalid, epoch=1, step=3, batch_size=2)
    with pytest.raises(ValueError, match="step count"):
        validate_progress(progress, epoch=1, step=4, batch_size=2)


def test_rng_restore_covers_python_numpy_and_torch():
    seed_everything(99)
    saved = capture_rng()
    expected = (random.random(), np.random.rand(5), torch.rand(5))
    seed_everything(100)
    restore_rng(saved, torch.device("cpu"))
    assert random.random() == expected[0]
    assert np.array_equal(np.random.rand(5), expected[1])
    assert torch.equal(torch.rand(5), expected[2])


def test_epoch_order_resume_is_a_permutation_suffix_without_global_rng_mutation():
    before = torch.get_rng_state().clone()
    order = epoch_indices(19, 2019, 4)
    assert sorted(order) == list(range(19))
    assert epoch_indices(19, 2019, 4, 7) == order[7:]
    assert epoch_indices(19, 2019, 5) != order
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError):
        epoch_indices(19, -1, 4)
