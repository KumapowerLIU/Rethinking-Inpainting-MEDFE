"""Explicit data preparation, weight provisioning, training and inference commands."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


def prepare_weights(
    output: str, copy_discriminator: str | None = None, *, lpips_only: bool = False
) -> dict:
    if lpips_only:
        if copy_discriminator:
            raise ValueError("--lpips-only cannot be combined with --copy-discriminator")
        from .perceptual import LPIPSLoss

        root = Path(output).resolve()
        root.mkdir(parents=True, exist_ok=True)
        result = {"lpips": LPIPSLoss(net="vgg", allow_download=True).provenance}
        (root / "lpips-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    from .discriminators import (
        REALESRGAN_D_SHA256,
        REALESRGAN_D_URL,
        UNetDiscriminatorSN,
        load_realesrgan_weights,
    )
    from .perceptual import LPIPSLoss
    from .weights import file_sha256

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "RealESRGAN_x4plus_netD.pth"
    if not destination.exists():
        fd, temporary = tempfile.mkstemp(dir=root, suffix=".download")
        os.close(fd)
        try:
            if copy_discriminator:
                shutil.copyfile(copy_discriminator, temporary)
            else:
                urllib.request.urlretrieve(REALESRGAN_D_URL, temporary)
            if file_sha256(temporary) != REALESRGAN_D_SHA256:
                raise ValueError("Downloaded Real-ESRGAN D failed the pinned SHA256 check")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    provenance = load_realesrgan_weights(UNetDiscriminatorSN(), destination)
    metric = LPIPSLoss(net="vgg", allow_download=True)
    result = {
        "discriminator": provenance,
        "lpips": metric.provenance,
        "torch_home": os.environ.get("TORCH_HOME", "~/.cache/torch"),
    }
    (root / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Modern MEDFE image inpainting")
    commands = result.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare-data", help="Create content-checked RTV pairs")
    data.add_argument("--images-root", required=True)
    data.add_argument("--cache-root", required=True)
    data.add_argument("--manifest", required=True)
    data.add_argument("--workers", type=int, default=0)
    weights = commands.add_parser("prepare-weights", help="Explicitly provision pretrained weights")
    weights.add_argument("--output", default="weights")
    weights.add_argument("--copy-discriminator")
    weights.add_argument("--lpips-only", action="store_true", help="Skip Real-ESRGAN weights")
    export = commands.add_parser("export", help="Export a portable generator-only checkpoint")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    train.add_argument("--resume")
    train.add_argument("--max-steps", type=int, help="Stop at this absolute optimizer step")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--limit", type=int, default=64)
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.add_argument("--device", default="auto")
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--image", required=True)
    inference.add_argument("--mask", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--device", default="auto")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare-data":
        from .data import prepare_rtv_dataset

        result = prepare_rtv_dataset(
            args.images_root, args.cache_root, args.manifest, workers=args.workers
        )
    elif args.command == "prepare-weights":
        result = prepare_weights(args.output, args.copy_discriminator, lpips_only=args.lpips_only)
    elif args.command == "export":
        from .inference import export_generator

        result = export_generator(args.checkpoint, args.output)
    elif args.command == "train":
        from .config import Config
        from .runner import train

        config = Config.load(args.config, args.set)
        result = train(config, resume=args.resume, max_steps=args.max_steps)
    elif args.command == "evaluate":
        from .evaluation import evaluate, load_generator
        from .runner import make_dataset

        model, config, payload = load_generator(args.checkpoint, args.device)
        config.data.val_manifest = args.manifest
        dataset = make_dataset(config, validation=True)
        result = evaluate(
            model,
            dataset,
            args.output,
            next(model.parameters()).device,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        result["checkpoint_step"] = payload["global_step"]
    else:
        from .evaluation import infer

        result = infer(args.checkpoint, args.image, args.mask, args.output, args.device)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
