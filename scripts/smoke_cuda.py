"""Bounded full-width numerical smoke; synthetic inputs do not assess image quality."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from medfe.checkpoint import capture_rng, seed_everything
from medfe.config import Config
from medfe.masks import random_mask
from medfe.rtv import rtv_smooth
from medfe.trainer import Trainer
from medfe.weights import state_dict_sha256


def recursive_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(state_dict_sha256({"value": item}).encode())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def snapshot(trainer):
    def clone_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: clone_cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(clone_cpu(item) for item in value)
        return copy.deepcopy(value)

    return clone_cpu(
        {
            **{
                name: getattr(trainer, name).state_dict()
                for name in (
                    "generator",
                    "global_d",
                    "local_d",
                    "g_optimizer",
                    "d_optimizer",
                    "g_scheduler",
                    "d_scheduler",
                    "scaler",
                    "lpips",
                    "vgg",
                )
                if getattr(trainer, name) is not None
            },
            "rng": capture_rng(),
            "step": trainer.global_step,
            "cursor": trainer.data_progress,
        }
    )


def compare_next_step(expected, actual):
    """CUDA reflection-pad backward uses atomics; report a bounded numerical comparison."""
    maxima = {"model": 0.0, "optimizer": 0.0}
    counts = {"tensors": 0, "bitwise_equal": 0, "outside_tolerance": 0}
    failures = []

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            counts["tensors"] += 1
            counts["bitwise_equal"] += int(torch.equal(left, right))
            category = "optimizer" if "optimizer" in path else "model"
            exact_required = path.endswith(".step") or any(
                f".{name}." in path for name in ("lpips", "vgg", "rng")
            )
            if left.is_floating_point() and not exact_required:
                rtol, atol = (1e-3, 1e-5) if category == "optimizer" else (1e-4, 1e-6)
                close = torch.isclose(left, right, rtol=rtol, atol=atol)
                if not bool(close.all()):
                    counts["outside_tolerance"] += 1
                    if len(failures) < 20:
                        failures.append(
                            {
                                "path": path,
                                "max_abs": float((left - right).abs().max()),
                                "outside_fraction": float((~close).float().mean()),
                            }
                        )
                maxima[category] = max(maxima[category], float((left - right).abs().max()))
            else:
                assert torch.equal(left, right), path
        elif isinstance(left, dict):
            assert left.keys() == right.keys(), path
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right), path
            for index, (first, second) in enumerate(zip(left, right)):
                visit(first, second, f"{path}.{index}")
        else:
            assert left == right, path

    visit(expected, actual, "state")
    return {
        "max_absolute_difference": maxima,
        **counts,
        "first_failures": failures,
        "model_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "optimizer_tolerance": {"rtol": 1e-3, "atol": 1e-5},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/realesrgan-lpips.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--memory-gib", type=float, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires CUDA")
    free, total = torch.cuda.mem_get_info()
    budget = int(args.memory_gib * 1024**3)
    if budget > free - 2 * 1024**3 or budget <= 0:
        raise RuntimeError("Insufficient free memory for the requested bounded smoke")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = Config.load(
        args.config,
        [
            "train.device=cuda",
            "train.amp=bf16",
            "data.workers=0",
            f"data.batch_size={args.batch_size}",
            f"discriminator.pretrained_path={args.weights}",
            "discriminator.chunk_size=1",
            "loss.lpips_chunk_size=1",
        ],
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.train.seed)
    pixels = np.random.default_rng(77).random((args.batch_size, 256, 256, 3)).astype(np.float32)
    image = torch.from_numpy(pixels).permute(0, 3, 1, 2) * 2 - 1
    structure = torch.from_numpy(np.stack([rtv_smooth(value) for value in pixels]))
    structure = structure.permute(0, 3, 1, 2).float() * 2 - 1
    mask = torch.stack([random_mask(256, 256, seed=90 + i) for i in range(args.batch_size)])
    trainer = Trainer(config)
    trainer.bind_data("f" * 64, 4 * args.batch_size)
    first = trainer.step(image, structure, mask)
    print(json.dumps({"event": "first_step", **first}), flush=True)
    checkpoint = output / "latest.pt"
    trainer.save(checkpoint)
    saved_hash = recursive_hash(snapshot(trainer))
    next_losses = trainer.step(image, structure, mask)
    expected = snapshot(trainer)
    parameter_count = sum(value.numel() for value in trainer.generator.parameters())
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed = Trainer(config, restoring=True)
    resumed.restore(checkpoint)
    if recursive_hash(snapshot(resumed)) != saved_hash:
        raise AssertionError("Checkpoint restore must reproduce every saved state exactly")
    resumed_losses = resumed.step(image, structure, mask)
    actual = snapshot(resumed)
    comparison = compare_next_step(expected, actual)
    torch.testing.assert_close(
        torch.tensor(list(next_losses.values())),
        torch.tensor(list(resumed_losses.values())),
        rtol=1e-4,
        atol=1e-5,
    )
    expected_hash, actual_hash = recursive_hash(expected), recursive_hash(actual)
    report = {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "precision": "bf16",
        "generator_parameters": parameter_count,
        "first_step": first,
        "next_step": next_losses,
        "checkpoint_restore_exact": True,
        "next_step_state_exact": expected_hash == actual_hash,
        "next_step_losses_exact": next_losses == resumed_losses,
        "next_step_numerically_close": comparison["outside_tolerance"] == 0,
        "comparison": comparison,
        "determinism": "warn_only: CUDA reflection_pad2d backward uses nondeterministic atomics",
        "backend": {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "state_sha256": actual_hash,
        "input_kind": "synthetic RGB with actual RTV; numerical smoke only",
        "global_d_provenance": resumed.global_d_provenance,
        "lpips_provenance": resumed.lpips.provenance if resumed.lpips else None,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if comparison["outside_tolerance"]:
        raise AssertionError("Next-step state exceeded predefined tolerances; see report.json")


if __name__ == "__main__":
    main()
