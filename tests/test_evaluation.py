import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from medfe.checkpoint import capture_rng
from medfe.config import Config
from medfe.evaluation import evaluate, infer, load_generator
from medfe.trainer import Trainer
from medfe.weights import state_dict_sha256


class EvaluationDataset:
    training = False
    fingerprint = "test-fixed-validation"

    def __len__(self):
        return 3

    def __getitem__(self, index):
        random.random()
        np.random.rand()
        torch.rand(1)
        image = torch.ones(3, 256, 256)
        mask = torch.zeros(1, 256, 256, dtype=torch.bool)
        if index == 1:
            mask[:, :, :128] = True
        elif index == 2:
            mask[:] = True
        return {"image": image, "structure": image, "mask": mask, "id": f"sample-{index}"}


class EvaluationModel(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))
        self.child = nn.Dropout(0.5).eval()
        self.fail = fail
        self.was_inference_mode = False

    def forward(self, image, mask):
        self.was_inference_mode = torch.is_inference_mode_enabled()
        assert not self.training and not self.child.training
        if self.fail:
            raise RuntimeError("deliberate evaluation failure")
        return SimpleNamespace(
            masked=torch.where(mask, -0.5, image), completed=torch.where(mask, self.value, image)
        )


def assert_same_rng(first, second):
    assert first["python"] == second["python"]
    assert first["numpy"] == second["numpy"]
    assert torch.equal(first["torch"], second["torch"])
    assert len(first["cuda"]) == len(second["cuda"])
    for a, b in zip(first["cuda"], second["cuda"]):
        assert torch.equal(a, b)
    if first["mps"] is not None:
        assert torch.equal(first["mps"], second["mps"])


@pytest.fixture
def generator_checkpoint(tmp_path):
    config = Config()
    config.model.base_channels = 2
    config.model.residual_blocks = 1
    config.model.pconv_layers = 1
    config.discriminator.global_width = 2
    config.discriminator.local_width = 2
    config.loss.perceptual = config.loss.style = 0
    config.train.device = "cpu"
    trainer = Trainer(config)
    path = tmp_path / "model.pt"
    trainer.save(path)
    return path


def test_evaluation_metrics_empty_masks_preview_and_rng_preservation(tmp_path):
    model = EvaluationModel()
    before = capture_rng()
    result = evaluate(model, EvaluationDataset(), tmp_path / "eval", "cpu", batch_size=2)
    after = capture_rng()
    assert_same_rng(before, after)
    assert model.training and not model.child.training and model.was_inference_mode
    assert result["images"] == 3 and result["empty_mask_images"] == 1
    assert result["per_image"][0]["hole_l1"] is None
    assert result["per_image"][0]["hole_psnr"] is None
    assert result["per_image"][0]["psnr"] == 120
    assert result["hole_l1"] == 0.5
    assert result["l1"] == 0.25
    assert result["per_image"][1]["psnr"] == pytest.approx(9.0308998699)
    assert model.value.grad is None
    assert json.loads((tmp_path / "eval/metrics.json").read_text())["images"] == 3
    with Image.open(tmp_path / "eval/preview.png") as preview:
        assert preview.width == 3 * 256


def test_evaluation_restores_modes_and_rng_on_failure(tmp_path):
    model = EvaluationModel(fail=True)
    before = capture_rng()
    with pytest.raises(RuntimeError, match="deliberate"):
        evaluate(model, EvaluationDataset(), tmp_path / "failed", "cpu")
    assert_same_rng(before, capture_rng())
    assert model.training and not model.child.training


def test_evaluation_rejects_training_dataset_and_overwriting_results(tmp_path):
    dataset = EvaluationDataset()
    dataset.training = True
    with pytest.raises(ValueError, match="validation"):
        evaluate(EvaluationModel(), dataset, tmp_path / "training", "cpu")
    dataset.training = False
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("retained")
    with pytest.raises(FileExistsError):
        evaluate(EvaluationModel(), dataset, output, "cpu")
    assert (output / "keep.txt").read_text() == "retained"


def test_evaluation_all_empty_masks_have_null_hole_aggregates(tmp_path):
    result = evaluate(EvaluationModel(), EvaluationDataset(), tmp_path / "empty", "cpu", limit=1)
    assert result["hole_l1"] is None and result["hole_psnr"] is None
    assert result["l1"] == 0 and result["psnr"] == 120
    json.dumps(result, allow_nan=False)


def test_generator_loading_is_offline_and_ignores_current_source_hash(
    generator_checkpoint, monkeypatch
):
    payload = torch.load(generator_checkpoint, map_location="cpu", weights_only=True)
    payload["source"] = {"old-version.py": "0" * 64}
    torch.save(payload, generator_checkpoint)
    monkeypatch.setattr(
        torch.hub,
        "download_url_to_file",
        lambda *args, **kwargs: pytest.fail("Inference attempted a download"),
    )
    before = capture_rng()
    model, config, saved = load_generator(generator_checkpoint, "cpu")
    assert_same_rng(before, capture_rng())
    assert config.model.base_channels == 2 and saved["source"] == payload["source"]
    assert not model.training and all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize(
    "corruption",
    ["fixture", "keys", "nonfinite", "dtype", "metric_fixture", "fill", "count_kernel"],
)
def test_generator_loading_rejects_invalid_checkpoint(generator_checkpoint, corruption):
    payload = torch.load(generator_checkpoint, map_location="cpu", weights_only=True)
    if corruption == "fixture":
        payload["test_fixture"] = True
    elif corruption == "metric_fixture":
        payload["lpips_provenance"] = {"fixture": True}
    elif corruption in {"fill", "count_kernel"}:
        name = next(name for name in payload["generator"] if name.endswith(corruption))
        payload["generator"][name].add_(0.5)
    else:
        name = next(name for name in payload["generator"] if name.endswith("weight"))
        if corruption == "keys":
            del payload["generator"][name]
        elif corruption == "nonfinite":
            payload["generator"][name].flatten()[0] = float("nan")
        else:
            payload["generator"][name] = payload["generator"][name].double()
    payload["model_state_sha256"]["generator"] = state_dict_sha256(payload["generator"])
    torch.save(payload, generator_checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        load_generator(generator_checkpoint, "cpu")


@pytest.mark.parametrize("channels", [3, 4])
def test_infer_pads_crops_and_preserves_known_uint8_pixels(
    generator_checkpoint, tmp_path, channels
):
    original = np.random.default_rng(31).integers(0, 256, (65, 97, channels), dtype=np.uint8)
    mask = np.zeros((65, 97), dtype=np.uint8)
    mask[14:51, 29:61] = 255
    image_path, mask_path, output = (
        tmp_path / "input.png",
        tmp_path / "mask.png",
        tmp_path / "out.png",
    )
    Image.fromarray(original).save(image_path)
    Image.fromarray(mask).save(mask_path)
    before = capture_rng()
    result = infer(generator_checkpoint, image_path, mask_path, output, "cpu")
    assert_same_rng(before, capture_rng())
    actual = np.asarray(Image.open(output))
    assert actual.shape == original.shape
    assert np.array_equal(actual[mask == 0], original[mask == 0])
    if channels == 4:
        assert np.array_equal(actual[:, :, 3], original[:, :, 3])
    assert result["known_pixels_preserved"]
    assert result["original_size"] == [65, 97] and result["padded_size"] == [256, 256]


def test_infer_rejects_misaligned_mask_and_lossy_or_source_output(generator_checkpoint, tmp_path):
    image, mask = tmp_path / "input.png", tmp_path / "mask.png"
    Image.new("RGB", (97, 65)).save(image)
    Image.new("L", (96, 65)).save(mask)
    with pytest.raises(ValueError, match="dimensions"):
        infer(generator_checkpoint, image, mask, tmp_path / "out.png", "cpu")
    with pytest.raises(ValueError, match="PNG"):
        infer(generator_checkpoint, image, mask, tmp_path / "out.jpg", "cpu")
    with pytest.raises(ValueError, match="overwrite"):
        infer(generator_checkpoint, image, mask, image, "cpu")
