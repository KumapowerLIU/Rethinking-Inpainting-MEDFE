"""Deterministic generator-only evaluation and geometry-preserving inference."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .checkpoint import capture_rng, load_checkpoint, restore_rng
from .config import Config
from .model import MEDFEGenerator
from .weights import file_sha256, state_dict_sha256


def _device(request: str | torch.device) -> torch.device:
    if str(request) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


@contextmanager
def _preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state, torch.device("cpu"))
        if state["cuda"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        if state.get("mps") is not None:
            torch.mps.set_rng_state(state["mps"].cpu())


@contextmanager
def _evaluation_context(model: nn.Module, device: torch.device):
    # Retain heterogeneous child modes as well as the parent's top-level flag.
    modes = [(module, module.training) for module in model.modules()]
    with _preserve_rng():
        try:
            model.eval()
            with torch.inference_mode(), torch.autocast(device.type, enabled=False):
                yield
        finally:
            for module, mode in modes:
                module.training = mode


def load_generator(
    checkpoint: str | Path, device: str | torch.device = "auto"
) -> tuple[MEDFEGenerator, Config, dict[str, Any]]:
    """Load a production G without constructing D/metrics or downloading assets.

    Training-source hashes need not match the current checkout for inference.
    Architecture/semantics, tensor keys, shapes, dtypes and finiteness do match.
    Constructing the generator does not advance the caller's random streams.
    """
    selected = _device(device)
    with _preserve_rng():
        payload = load_checkpoint(checkpoint, allow_inference=True)
        for name in ("global_d_provenance", "lpips_provenance", "vgg_provenance"):
            provenance = payload.get(name)
            if isinstance(provenance, dict) and (
                provenance.get("fixture") is True or provenance.get("test_fixture") is True
            ):
                raise ValueError(
                    "Diagnostic perceptual/discriminator fixtures cannot label a production G"
                )
        config = Config.from_dict(payload["config"])
        model = MEDFEGenerator(**asdict(config.model))
        expected, state = model.state_dict(), payload["generator"]
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Generator checkpoint keys do not match the configured architecture")
        for name, target in expected.items():
            value = state[name]
            if (
                not isinstance(value, Tensor)
                or value.shape != target.shape
                or value.dtype != target.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"Invalid generator checkpoint tensor: {name}")
            if name == "fill" or name.endswith(".count_kernel"):
                if not torch.equal(value, target):
                    raise ValueError(f"Checkpoint changed fixed generator buffer: {name}")
        model.load_state_dict(state, strict=True)
        model.to(selected).float().eval().requires_grad_(False)
    return model, config, payload


def _validate_batch(image: Tensor, mask: Tensor) -> None:
    if (
        image.ndim != 4
        or image.shape[1] != 3
        or image.shape[0] < 1
        or not image.is_floating_point()
        or not bool(torch.isfinite(image).all())
        or bool((image < -1).any())
        or bool((image > 1).any())
    ):
        raise ValueError("Evaluation RGB batches must be finite Bx3xHxW in [-1,1]")
    if (
        mask.shape != (image.shape[0], 1, *image.shape[-2:])
        or mask.dtype != torch.bool
        or mask.device != image.device
    ):
        raise ValueError("Evaluation masks must be matching boolean Bx1xHxW, True means hole")


def _image_metrics(prediction: Tensor, target: Tensor, mask: Tensor) -> dict[str, Any]:
    # Metrics use RGB [0,1]. PSNR's explicit epsilon keeps exact matches finite
    # (120 dB), and JSON serialization never emits NaN or Infinity.
    difference = (prediction.detach().cpu().double() - target.detach().cpu().double()) * 0.5
    mask = mask.detach().cpu()
    mse = float(difference.square().mean())
    hole = mask.expand_as(difference)
    has_hole = bool(hole.any())
    hole_mse = float(difference[hole].square().mean()) if has_hole else None
    return {
        "l1": float(difference.abs().mean()),
        "psnr": -10 * math.log10(max(mse, 1e-12)),
        "hole_l1": float(difference[hole].abs().mean()) if has_hole else None,
        "hole_psnr": -10 * math.log10(max(hole_mse, 1e-12)) if has_hole else None,
        "hole_fraction": float(mask.float().mean()),
    }


def _rgb_image(value: Tensor) -> Image.Image:
    rgb = ((value.detach().cpu().float().clamp(-1, 1) + 1) * 127.5).round()
    return Image.fromarray(rgb.to(torch.uint8).permute(1, 2, 0).numpy())


def _save_preview(rows: list[tuple[str, Tensor, Tensor, Tensor]], path: Path) -> None:
    width, height = rows[0][1].shape[-1], rows[0][1].shape[-2]
    heading, caption = 26, 22
    canvas = Image.new("RGB", (width * 3, heading + len(rows) * (height + caption)), "#20242a")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Masked input", "Completed", "Ground truth")):
        draw.text((column * width + 6, 6), label, fill="white")
    for index, (identifier, masked, completed, truth) in enumerate(rows):
        y = heading + index * (height + caption)
        for column, tensor in enumerate((masked, completed, truth)):
            canvas.paste(_rgb_image(tensor), (column * width, y))
        # A default bitmap font need not support arbitrary Unicode sample IDs.
        label = identifier.encode("ascii", errors="replace").decode("ascii")
        draw.text((6, y + height + 3), label, fill="white")
    canvas.save(path, format="PNG")


def evaluate(
    model: nn.Module,
    dataset,
    output_dir: str | Path,
    device: str | torch.device,
    limit: int = 64,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Evaluate the first fixed validation records, preserving modes and RNG.

    ``dataset`` follows InpaintingDataset and must have training=False. The
    model must already be on the requested device, so evaluation cannot move
    live optimizer parameters. Empty-hole images have null hole metrics and
    are excluded from hole-metric averages; full-image metrics still apply.
    """
    if type(limit) is not int or limit < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation limit and batch size must be positive integers")
    if len(dataset) < 1 or getattr(dataset, "training", False):
        raise ValueError("Evaluation requires a nonempty validation dataset (training=False)")
    selected = _device(device)
    if selected.type == "cuda" and selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    tensors = [*model.parameters(), *model.buffers()]
    if any(tensor.device != selected for tensor in tensors):
        raise ValueError("Move the model to the requested evaluation device before evaluating")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    count = min(limit, len(dataset))
    rows, previews = [], []
    with _evaluation_context(model, selected):
        loader = DataLoader(
            Subset(dataset, range(count)),
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            generator=torch.Generator().manual_seed(0),
        )
        for batch in loader:
            image = batch["image"].to(selected).float()
            mask = batch["mask"].to(selected)
            _validate_batch(image, mask)
            prediction = model(image, mask)
            completed = prediction.completed
            if completed.shape != image.shape or not bool(torch.isfinite(completed).all()):
                raise ValueError("Generator returned invalid completed images")
            known = ~mask.expand_as(image)
            if not torch.equal(completed[known], image[known]):
                raise ValueError("Generator modified known pixels")
            for item in range(image.shape[0]):
                identifier = str(batch["id"][item]) if "id" in batch else str(len(rows))
                row = {
                    "index": len(rows),
                    "id": identifier,
                    **_image_metrics(completed[item], image[item], mask[item]),
                }
                rows.append(row)
                if len(previews) < 8:
                    previews.append(
                        (
                            identifier,
                            prediction.masked[item].cpu(),
                            completed[item].cpu(),
                            image[item].cpu(),
                        )
                    )
    summary = {
        "images": len(rows),
        "empty_mask_images": sum(row["hole_l1"] is None for row in rows),
    }
    for key in ("l1", "psnr", "hole_l1", "hole_psnr", "hole_fraction"):
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    report = {
        **summary,
        "precision": "FP32",
        "metric_range": [0, 1],
        "psnr_mse_floor": 1e-12,
        "empty_mask_policy": "null hole metrics; excluded from hole averages",
        "dataset_fingerprint": getattr(dataset, "fingerprint", None),
        "per_image": rows,
        "output_dir": str(output.resolve()),
        "preview": str((output / "preview.png").resolve()),
    }
    _save_preview(previews, output / "preview.png")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def infer(
    checkpoint: str | Path,
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Inpaint original geometry, padding right/bottom without resizing masks.

    White mask pixels (>127) are holes. RGB is replicate-padded to dimensions
    >=256 divisible by64; padding is treated as known context. The result is
    cropped back, and known uint8 pixels (and any alpha channel) are explicitly
    copied from the oriented input before lossless PNG encoding.
    """
    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError("Inference output must be PNG to preserve known pixels exactly")
    if destination.resolve() in {Path(image_path).resolve(), Path(mask_path).resolve()}:
        raise ValueError("Inference output must not overwrite the source image or mask")
    with Image.open(image_path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        original = np.asarray(oriented.convert("RGBA" if has_alpha else "RGB")).copy()
    rgb = original[:, :, :3]
    with Image.open(mask_path) as opened:
        hole = np.asarray(ImageOps.exif_transpose(opened).convert("L")) > 127
    height, width = rgb.shape[:2]
    if hole.shape != (height, width):
        raise ValueError(
            "Image and mask dimensions must match exactly; masks are never independently resized"
        )
    padded_h, padded_w = (max(256, math.ceil(size / 64) * 64) for size in (height, width))
    selected = _device(device)
    model, _, payload = load_generator(checkpoint, selected)
    with _evaluation_context(model, selected):
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(selected).float()
        image = image / 127.5 - 1
        mask = torch.from_numpy(hole.copy()).unsqueeze(0).unsqueeze(0).to(selected)
        image = F.pad(image, (0, padded_w - width, 0, padded_h - height), mode="replicate")
        mask = F.pad(mask, (0, padded_w - width, 0, padded_h - height), value=False)
        completed = model(image, mask).completed[:, :, :height, :width]
        if not bool(torch.isfinite(completed).all()):
            raise ValueError("Generator returned nonfinite inference pixels")
        generated = np.asarray(_rgb_image(completed[0])).copy()
    generated[~hole] = rgb[~hole]
    if has_alpha:
        generated = np.concatenate((generated, original[:, :, 3:4]), axis=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(generated).save(destination, format="PNG")
    with Image.open(destination) as reopened:
        verified = np.asarray(reopened)
    if not np.array_equal(verified[~hole], original[~hole]):
        raise RuntimeError("PNG known-pixel preservation verification failed")
    return {
        "output": str(destination.resolve()),
        "checkpoint_step": payload["global_step"],
        "generator_sha256": state_dict_sha256(payload["generator"]),
        "input_sha256": file_sha256(image_path),
        "mask_sha256": file_sha256(mask_path),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "original_size": [height, width],
        "padded_size": [padded_h, padded_w],
        "hole_fraction": float(hole.mean()),
        "known_pixels_preserved": True,
        "precision": "FP32",
        "mask_convention": "white (>127) means hole",
    }
