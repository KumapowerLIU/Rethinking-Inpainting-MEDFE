import copy
import random
import warnings

import numpy as np
import pytest
import torch
from torch import nn

import medfe.trainer as trainer_module
from medfe.checkpoint import seed_everything
from medfe.config import Config
from medfe.perceptual import LPIPSLoss, SharedVGG
from medfe.trainer import Trainer, frozen_parameters
from medfe.weights import state_dict_sha256


class FixtureDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, prediction, target, normalize=False):
        assert normalize is False
        return (prediction - target).square().mean((1, 2, 3), keepdim=True) * self.scale


def tiny_config():
    config = Config()
    config.model.base_channels = 2
    config.model.residual_blocks = config.model.pconv_layers = 1
    config.discriminator.global_width = config.discriminator.local_width = 2
    config.discriminator.chunk_size = 1
    config.train.device = "cpu"
    config.data.batch_size = 2
    config.data.workers = 0
    config.loss.perceptual = config.loss.style = 0
    return config


def batch(size=2):
    generator = torch.Generator().manual_seed(55 + size)
    image = torch.rand(size, 3, 256, 256, generator=generator) * 2 - 1
    structure = image * 0.8
    mask = torch.zeros(size, 1, 256, 256, dtype=torch.bool)
    mask[0, :, 70:200, 50:190] = True
    if size > 1:
        mask[1, :, 20:130, 110:220] = True
    return image, structure, mask


@pytest.fixture(autouse=True)
def stable_signature(monkeypatch):
    monkeypatch.setattr(trainer_module, "code_signature", lambda: {"fixture.py": "a" * 64})


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_actual_medfe_step_uses_one_g_backward_and_all_losses_once(monkeypatch):
    config = tiny_config()
    config.loss.reconstruction = 0.1
    config.loss.lpips, config.loss.perceptual, config.loss.style = 1, 0.2, 0.3
    config.loss.structure, config.loss.texture = 0.7, 0.8
    metric = LPIPSLoss(fixture_network=FixtureDistance())
    features = nn.Sequential(nn.Conv2d(3, 3, 1), *[nn.Identity() for _ in range(27)])
    trainer = Trainer(
        config, test_fixture=True, lpips=metric, vgg=SharedVGG(fixture_features=features)
    )
    original = batch()
    saved = tuple(value.clone() for value in original)
    calls = []
    backward = torch.Tensor.backward

    def observed(value, *args, **kwargs):
        calls.append(float(value.detach()))
        return backward(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", observed)
    logs = trainer.step(*original)
    assert len(calls) == 2  # One joint D scalar and one joint G scalar.
    assert calls[0] == pytest.approx(logs["discriminator"])
    assert calls[1] == pytest.approx(logs["generator"])
    weighted_sum = sum(value for key, value in logs.items() if key.endswith("_weighted"))
    assert logs["generator"] == pytest.approx(weighted_sum, abs=2e-6)
    for name in (
        "reconstruction",
        "lpips",
        "perceptual",
        "style",
        "structure",
        "texture",
        "adversarial",
    ):
        assert logs[name + "_weighted"] == pytest.approx(logs[name] * getattr(config.loss, name))
    for original_value, saved_value in zip(original, saved):
        assert torch.equal(original_value, saved_value)
    assert trainer.global_step == 1
    assert all(parameter.grad is None for parameter in metric.parameters())
    assert all(parameter.grad is None for parameter in trainer.vgg.parameters())
    assert trainer.global_d.training is trainer.local_d.training is False
    assert logs["effective_adversarial_weight"] == pytest.approx(config.loss.adversarial)


@pytest.mark.parametrize(
    "warmup,ramp,coefficient,expected",
    [
        (0, 0, 0.2, [0.2, 0.2, 0.2]),
        (2, 0, 0.2, [0, 0, 0.2, 0.2]),
        (0, 3, 0.3, [0.1, 0.2, 0.3, 0.3]),
        (2, 3, 0.3, [0, 0, 0.1, 0.2, 0.3, 0.3]),
        (2, 1, 0.2, [0, 0, 0.2, 0.2]),
        (2, 3, 0.0, [0, 0, 0, 0, 0, 0]),
    ],
)
def test_adversarial_weight_indexes_the_next_update(warmup, ramp, coefficient, expected):
    trainer = object.__new__(Trainer)
    trainer.config = tiny_config()
    trainer.config.train.adversarial_warmup_steps = warmup
    trainer.config.train.adversarial_ramp_steps = ramp
    trainer.config.loss.adversarial = coefficient
    for completed_steps, weight in enumerate(expected):
        trainer.global_step = completed_steps
        assert trainer.effective_adversarial_weight() == pytest.approx(weight)


@pytest.mark.parametrize("disable", ["coefficient", "warmup"])
def test_zero_gan_skips_every_d_path_and_preserves_d_state(monkeypatch, disable):
    config = tiny_config()
    config.loss.lpips = 1
    if disable == "coefficient":
        config.loss.adversarial = 0
    else:
        config.train.adversarial_warmup_steps = 2
    trainer = Trainer(config, test_fixture=True, lpips=LPIPSLoss(fixture_network=FixtureDistance()))
    before = {
        name: copy.deepcopy(getattr(trainer, name).state_dict())
        for name in ("global_d", "local_d", "d_optimizer")
    }
    g_before = state_dict_sha256(trainer.generator.state_dict())

    def forbidden(*args, **kwargs):
        pytest.fail("Zero GAN weight must not execute a discriminator path")

    monkeypatch.setattr(trainer_module, "prepare_discriminator", forbidden)
    monkeypatch.setattr(trainer, "_logits", forbidden)
    monkeypatch.setattr(trainer.d_optimizer, "step", forbidden)
    monkeypatch.setattr(trainer_module, "crop_batch", forbidden)
    logs = trainer.step(*batch())
    for name, state in before.items():
        assert_nested_equal(state, getattr(trainer, name).state_dict())
    assert state_dict_sha256(trainer.generator.state_dict()) != g_before
    assert logs["effective_adversarial_weight"] == 0
    for name in ("discriminator", "d_global", "d_local", "adversarial", "adversarial_weighted"):
        assert logs[name] == 0
    assert logs["lpips_weighted"] > 0
    assert logs["reconstruction_weighted"] > 0
    assert logs["structure_weighted"] > 0 and logs["texture_weighted"] > 0


@pytest.mark.parametrize("completed_steps", [1, 2])
def test_adversarial_ramp_resumes_exactly_at_boundaries(tmp_path, completed_steps):
    seed_everything(2019)
    config = tiny_config()
    config.train.adversarial_warmup_steps = 1
    config.train.adversarial_ramp_steps = 2
    trainer = Trainer(config)
    trainer.bind_data("a" * 64, 8)
    for _ in range(completed_steps):
        trainer.step(*batch())
    path = tmp_path / "boundary.pt"
    trainer.save(path)
    expected = trainer.step(*batch())
    expected_states = {
        name: copy.deepcopy(getattr(trainer, name).state_dict())
        for name in ("generator", "global_d", "local_d", "g_optimizer", "d_optimizer")
    }
    restored = Trainer(config, restoring=True)
    restored.restore(path)
    actual = restored.step(*batch())
    assert actual == expected
    assert actual["effective_adversarial_weight"] == pytest.approx(0.1 * completed_steps)
    for name, state in expected_states.items():
        assert_nested_equal(state, getattr(restored, name).state_dict())


def test_warmup_crosses_epochs_without_scheduler_warning_and_resumes_first_gan(tmp_path):
    seed_everything(2019)
    config = tiny_config()
    config.train.schedule, config.train.epochs, config.train.decay_start = "linear", 4, 0
    config.train.adversarial_warmup_steps = 2
    config.train.adversarial_ramp_steps = 2
    trainer = Trainer(config)
    trainer.bind_data("a" * 64, 2)
    d_before = {
        name: copy.deepcopy(getattr(trainer, name).state_dict()) for name in ("global_d", "local_d")
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(2):
            assert trainer.step(*batch())["effective_adversarial_weight"] == 0
            trainer.finish_epoch()
            assert not trainer.d_optimizer.state
            assert_nested_equal(trainer.g_scheduler.state_dict(), trainer.d_scheduler.state_dict())
            expected_lr = config.train.lr * (1 - (epoch + 1) / config.train.epochs)
            assert trainer.d_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert not any("lr_scheduler.step()" in str(item.message) for item in caught)
    for name, state in d_before.items():
        assert_nested_equal(state, getattr(trainer, name).state_dict())
    path = tmp_path / "warmup-boundary.pt"
    trainer.save(path)
    expected = trainer.step(*batch())
    assert expected["effective_adversarial_weight"] == pytest.approx(0.1)
    restored = Trainer(config, restoring=True)
    restored.restore(path)
    assert restored.step(*batch()) == expected
    for name in ("generator", "global_d", "local_d", "g_optimizer", "d_optimizer"):
        assert_nested_equal(
            getattr(trainer, name).state_dict(), getattr(restored, name).state_dict()
        )


def test_real_generator_discriminators_resume_next_step_exactly(tmp_path):
    seed_everything(2019)
    config = tiny_config()
    trainer = Trainer(config)
    trainer.bind_data("a" * 64, 6)
    trainer.step(*batch())
    path = tmp_path / "latest.pt"
    trainer.save(path)
    expected_logs = trainer.step(*batch())
    expected_rng = (random.random(), np.random.rand(3), torch.rand(3))
    expected_weights = {
        name: copy.deepcopy(getattr(trainer, name).state_dict())
        for name in ("generator", "global_d", "local_d", "g_optimizer", "d_optimizer")
    }
    restored = Trainer(config, restoring=True)
    restored.restore(path)
    restored.bind_data("a" * 64, 6)
    actual_logs = restored.step(*batch())
    assert actual_logs == expected_logs
    for name, expected in expected_weights.items():
        assert_nested_equal(getattr(restored, name).state_dict(), expected)
    assert random.random() == expected_rng[0]
    assert np.array_equal(np.random.rand(3), expected_rng[1])
    assert torch.equal(torch.rand(3), expected_rng[2])
    assert restored.data_progress == trainer.data_progress
    assert restored.global_step == trainer.global_step == 2


def test_failed_g_half_step_cannot_save_and_same_trainer_can_restore(tmp_path, monkeypatch):
    config = tiny_config()
    config.loss.lpips = 1
    trainer = Trainer(config, test_fixture=True, lpips=LPIPSLoss(fixture_network=FixtureDistance()))
    trainer.bind_data("a" * 64, 2)
    path = tmp_path / "latest.pt"
    trainer.save(path)
    original_bytes = path.read_bytes()
    original_d = state_dict_sha256(trainer.global_d.state_dict())

    def fail(*args, **kwargs):
        raise RuntimeError("injected G-phase failure")

    with monkeypatch.context() as patch:
        patch.setattr(trainer.lpips, "forward", fail)
        with pytest.raises(RuntimeError, match="G-phase"):
            trainer.step(*batch())
    assert state_dict_sha256(trainer.global_d.state_dict()) != original_d
    assert trainer.global_step == 0 and trainer.data_progress["samples_seen"] == 0
    with pytest.raises(RuntimeError, match="healthy"):
        trainer.save(path)
    with pytest.raises(RuntimeError):
        trainer.step(*batch())
    assert path.read_bytes() == original_bytes
    trainer.restore(path)
    assert state_dict_sha256(trainer.global_d.state_dict()) == original_d
    trainer.step(*batch())
    assert trainer.global_step == 1


def test_restoring_trainer_cannot_step_or_save_before_restore(tmp_path):
    trainer = Trainer(tiny_config(), restoring=True)
    with pytest.raises(RuntimeError):
        trainer.step(*batch())
    with pytest.raises(RuntimeError):
        trainer.save(tmp_path / "bad.pt")


def test_resume_rejects_changed_objective_source_data_and_fixture(tmp_path, monkeypatch):
    config = tiny_config()
    trainer = Trainer(config)
    trainer.bind_data("a" * 64, 3)
    path = tmp_path / "latest.pt"
    trainer.save(path)
    changed = copy.deepcopy(config)
    changed.loss.structure = 0.5
    with pytest.raises(ValueError, match="configuration"):
        Trainer(changed, restoring=True).restore(path)
    with monkeypatch.context() as patch:
        patch.setattr(trainer_module, "code_signature", lambda: {"fixture.py": "b" * 64})
        with pytest.raises(ValueError, match="Source changed"):
            Trainer(config, restoring=True).restore(path)
    with pytest.raises(ValueError, match="kinds differ"):
        Trainer(config, restoring=True, test_fixture=True).restore(path)
    restored = Trainer(config, restoring=True)
    restored.restore(path)
    with pytest.raises(ValueError, match="Dataset changed"):
        restored.bind_data("b" * 64, 3)


def test_resume_rejects_model_dtype_change_even_with_recomputed_hash(tmp_path):
    trainer = Trainer(tiny_config())
    path = tmp_path / "latest.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    key = next(iter(state["generator"]))
    state["generator"][key] = state["generator"][key].double()
    state["model_state_sha256"]["generator"] = state_dict_sha256(state["generator"])
    torch.save(state, path)
    with pytest.raises(ValueError, match="shape/dtype"):
        Trainer(tiny_config(), restoring=True).restore(path)


def test_fixed_partial_count_kernel_cannot_change_even_with_matching_hash(tmp_path):
    trainer = Trainer(tiny_config())
    path = tmp_path / "latest.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    key = next(key for key in state["generator"] if key.endswith(".count_kernel"))
    state["generator"][key].fill_(0.7)
    state["model_state_sha256"]["generator"] = state_dict_sha256(state["generator"])
    torch.save(state, path)
    with pytest.raises(ValueError, match="fixed generator buffer"):
        Trainer(tiny_config(), restoring=True).restore(path)


def test_dataset_tail_cursor_and_epoch_scheduler(tmp_path):
    config = tiny_config()
    config.train.schedule, config.train.epochs, config.train.decay_start = "linear", 3, 0
    trainer = Trainer(config)
    trainer.bind_data("a" * 64, 3)
    with pytest.raises(ValueError, match="next complete batch"):
        trainer.step(*batch(1))
    trainer.step(*batch())
    trainer.step(*batch(1))
    with pytest.raises(ValueError):
        trainer.step(*batch(1))
    trainer.finish_epoch()
    assert trainer.completed_epochs == 1
    assert trainer.data_progress["samples_in_epoch"] == 0
    assert trainer.data_progress["samples_seen"] == 3
    assert trainer.g_optimizer.param_groups[0]["lr"] == pytest.approx(config.train.lr * (2 / 3))
    trainer.save(tmp_path / "boundary.pt")


def test_frozen_parameter_flags_restore_after_exception():
    module = nn.Linear(2, 3)
    module.bias.requires_grad_(False)
    with pytest.raises(RuntimeError):
        with frozen_parameters(module):
            assert not any(parameter.requires_grad for parameter in module.parameters())
            raise RuntimeError("failure")
    assert module.weight.requires_grad and not module.bias.requires_grad


def test_fixture_injection_is_forbidden_in_production():
    with pytest.raises(ValueError, match="Injected test"):
        Trainer(tiny_config(), generator=nn.Identity())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fp16_scaler_restores_after_half_step_failure(tmp_path, monkeypatch):
    config = tiny_config()
    config.train.device, config.train.amp = "cuda", "fp16"
    config.loss.lpips = 1
    trainer = Trainer(config, test_fixture=True, lpips=LPIPSLoss(fixture_network=FixtureDistance()))
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=64.0)
    trainer.bind_data("a" * 64, 2)
    path = tmp_path / "latest.pt"
    trainer.save(path)
    with monkeypatch.context() as patch:
        patch.setattr(
            trainer.lpips, "forward", lambda *args: (_ for _ in ()).throw(RuntimeError("half-step"))
        )
        with pytest.raises(RuntimeError, match="half-step"):
            trainer.step(*batch())
    trainer.restore(path)
    trainer.step(*batch())
    assert trainer.global_step == 1
