import json

import pytest
import torch

from medfe.checkpoint import load_checkpoint
from medfe.config import Config
from medfe.evaluation import load_generator
from medfe.inference import export_generator
from medfe.trainer import Trainer


@pytest.mark.parametrize("backbone", ["released", "sd_partial"])
def test_bundle_roundtrip_is_exact_offline_and_not_a_training_checkpoint(tmp_path, backbone):
    config = Config()
    config.model.backbone = backbone
    config.model.base_channels = 2
    config.model.residual_blocks = config.model.pconv_layers = 1
    config.discriminator.global_width = config.discriminator.local_width = 2
    config.loss.perceptual = config.loss.style = 0
    config.train.device = "cpu"
    config.data.train_manifest = "/private/training/path.jsonl"
    source, exported = tmp_path / "training.pt", tmp_path / "inference.pt"
    Trainer(config).save(source)
    report = export_generator(source, exported)
    original, _, _ = load_generator(source, "cpu")
    portable, _, payload = load_generator(exported, "cpu")
    assert "private" not in json.dumps(payload["config"])
    assert not any(key in payload for key in ("global_d", "g_optimizer", "lpips"))
    assert report["bytes"] < source.stat().st_size
    image = torch.rand(1, 3, 256, 256) * 2 - 1
    mask = torch.zeros(1, 1, 256, 256, dtype=torch.bool)
    mask[:, :, 40:180, 60:150] = True
    with torch.inference_mode():
        assert torch.equal(original(image, mask).completed, portable(image, mask).completed)
    with pytest.raises(ValueError, match="training checkpoint"):
        load_checkpoint(exported)
    with pytest.raises(FileExistsError):
        export_generator(source, exported)
    key = next(k for k in payload["generator"] if k.endswith("weight"))
    payload["generator"][key].flatten()[0] += 1
    torch.save(payload, exported)
    with pytest.raises(ValueError, match="checksum"):
        load_generator(exported, "cpu")


def test_lpips_only_does_not_construct_or_fetch_realesrgan(tmp_path, monkeypatch):
    from medfe import cli, perceptual

    class Metric:
        provenance = {"fixture": True}

        def __init__(self, **kwargs):
            assert kwargs == {"net": "vgg", "allow_download": True}

    monkeypatch.setattr(perceptual, "LPIPSLoss", Metric)
    monkeypatch.setattr(cli.urllib.request, "urlretrieve", lambda *a: pytest.fail("D download"))
    result = cli.prepare_weights(str(tmp_path), lpips_only=True)
    assert set(result) == {"lpips"}
    assert not (tmp_path / "RealESRGAN_x4plus_netD.pth").exists()
