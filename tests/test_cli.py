import json

import numpy as np
import yaml
from PIL import Image

from medfe.cli import main


def test_prepare_train_resume_evaluate_infer_commands(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        array = np.random.default_rng(index).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        Image.fromarray(array).save(images / f"{index}.png")
    manifest = tmp_path / "data.jsonl"
    main(
        [
            "prepare-data",
            "--images-root",
            str(images),
            "--cache-root",
            str(tmp_path / "rtv"),
            "--manifest",
            str(manifest),
        ]
    )
    assert json.loads(capsys.readouterr().out)["count"] == 3
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"base_channels": 2, "residual_blocks": 1, "pconv_layers": 1},
                "data": {"train_manifest": str(manifest), "batch_size": 2, "workers": 0},
                "loss": {"perceptual": 0, "style": 0},
                "discriminator": {"global_width": 2, "local_width": 2},
                "train": {"output": str(tmp_path / "run"), "epochs": 2, "device": "cpu"},
            }
        )
    )
    main(["train", "--config", str(config), "--max-steps", "1"])
    assert json.loads(capsys.readouterr().out)["step"] == 1
    checkpoint = tmp_path / "run/latest.pt"
    main(["train", "--config", str(config), "--resume", str(checkpoint), "--max-steps", "2"])
    assert json.loads(capsys.readouterr().out)["completed_epochs"] == 1
    main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "evaluation"),
            "--limit",
            "2",
            "--device",
            "cpu",
        ]
    )
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["images"] == 2 and evaluated["checkpoint_step"] == 2
    assert (tmp_path / "evaluation/preview.png").exists()
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:20, 15:30] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    main(
        [
            "infer",
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(images / "0.png"),
            "--mask",
            str(tmp_path / "mask.png"),
            "--output",
            str(tmp_path / "result.png"),
            "--device",
            "cpu",
        ]
    )
    assert json.loads(capsys.readouterr().out)["known_pixels_preserved"]
    result = np.asarray(Image.open(tmp_path / "result.png"))
    original = np.asarray(Image.open(images / "0.png"))
    assert np.array_equal(result[mask == 0], original[mask == 0])
