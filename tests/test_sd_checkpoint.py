"""Architecture provenance must survive saving and reject incompatible resumes."""

import copy

import pytest
import torch

from medfe.checkpoint import SD_PARTIAL_SEMANTICS, SEMANTICS, load_checkpoint
from medfe.config import Config
from medfe.evaluation import load_generator
from medfe.trainer import Trainer


def tiny_config(backbone="sd_partial"):
    config = Config()
    config.model.backbone = backbone
    config.model.base_channels = 2
    config.model.blocks_per_scale = 1
    config.model.residual_blocks = config.model.pconv_layers = 1
    config.model.activation_checkpointing = True
    config.discriminator.global_width = config.discriminator.local_width = 2
    config.loss.perceptual = config.loss.style = 0
    config.train.device = "cpu"
    config.data.batch_size = 1
    return config


def inputs():
    rng = torch.Generator().manual_seed(76)
    image = torch.rand(1, 3, 256, 256, generator=rng) * 2 - 1
    mask = torch.zeros(1, 1, 256, 256, dtype=torch.bool)
    mask[:, :, 60:190, 80:200] = True
    return image, image * 0.8, mask


def test_sd_training_checkpoint_inference_and_exact_resume(tmp_path):
    config = tiny_config()
    trainer = Trainer(config)
    image, structure, mask = inputs()
    trainer.step(image, structure, mask)
    path = tmp_path / "sd.pt"
    trainer.save(path)
    model, saved_config, state = load_generator(path, "cpu")
    assert state["semantics"] == SD_PARTIAL_SEMANTICS
    assert saved_config.model.backbone == model.backbone == "sd_partial"
    with torch.no_grad():
        expected = trainer.generator.eval()(image, mask)
        actual = model(image, mask)
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    trainer.restore(path)
    expected_logs = trainer.step(image, structure, mask)
    expected_weights = copy.deepcopy(trainer.generator.state_dict())
    restored = Trainer(config, restoring=True)
    restored.restore(path)
    actual_logs = restored.step(image, structure, mask)
    assert restored.global_step == 2
    assert actual_logs == expected_logs
    for name, value in restored.generator.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("backbone", ["released", "sd_partial"])
def test_backbone_and_checkpoint_semantics_must_match(tmp_path, backbone):
    path = tmp_path / "model.pt"
    Trainer(tiny_config(backbone)).save(path)
    state = torch.load(path, weights_only=True)
    state["config"]["model"]["backbone"] = "released" if backbone == "sd_partial" else "sd_partial"
    torch.save(state, path)
    with pytest.raises(ValueError, match="mathematics.*backbone"):
        load_checkpoint(path)


def test_pre_backbone_config_loads_as_released_and_cannot_resume_as_sd(tmp_path):
    path = tmp_path / "released.pt"
    config = tiny_config("released")
    trainer = Trainer(config)
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    state["config"]["model"].pop("backbone")
    state["config"]["model"].pop("blocks_per_scale")
    torch.save(state, path)
    model, loaded_config, loaded = load_generator(path, "cpu")
    assert loaded["semantics"] == SEMANTICS
    assert loaded_config.model.backbone == model.backbone == "released"
    image, _, mask = inputs()
    with torch.no_grad():
        expected, actual = trainer.generator.eval()(image, mask), model(image, mask)
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture"):
        Trainer(tiny_config(), restoring=True).restore(path)


@pytest.mark.parametrize(
    "settings",
    [
        {"backbone": "sd"},
        {"blocks_per_scale": 0},
        {"blocks_per_scale": True},
    ],
)
def test_new_backbone_settings_are_validated(settings):
    with pytest.raises(ValueError):
        Config.from_dict({"model": settings})
