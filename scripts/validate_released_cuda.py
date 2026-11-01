"""Bounded real-data capacity and resume check for the released MEDFE objective."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from smoke_cuda import compare_next_step, recursive_hash, snapshot

from medfe.checkpoint import seed_everything
from medfe.config import Config
from medfe.runner import make_dataset, make_loader
from medfe.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = Config.load(args.config)
    if config.loss.vgg_profile != "released" or config.discriminator.global_kind != "patch":
        raise ValueError("This check requires the released VGG objective and original Patch D")
    if config.loss.lpips or config.discriminator.pretrained_path:
        raise ValueError("The released baseline must not use LPIPS or pretrained D weights")
    if config.train.adversarial_warmup_steps or config.train.adversarial_ramp_steps:
        raise ValueError("The original GAN must participate from the first update")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = make_dataset(config)
    loader = make_loader(dataset, config, 0)
    batch = next(iter(loader))
    del loader
    seed_everything(config.train.seed)
    trainer = Trainer(config)
    # This diagnostic deliberately reuses one real batch; it does not advance
    # the production data cursor and its weights never initialize the new run.
    started = time.monotonic()
    first = trainer.step(batch["image"], batch["structure"], batch["mask"])
    torch.cuda.synchronize()
    first_seconds = time.monotonic() - started
    print(json.dumps({"event": "first_step", "seconds": first_seconds, **first}), flush=True)
    checkpoint = output / "diagnostic.pt"
    trainer.save(checkpoint)
    saved_hash = recursive_hash(snapshot(trainer))
    second = trainer.step(batch["image"], batch["structure"], batch["mask"])
    expected = snapshot(trainer)
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    vgg_provenance = trainer.vgg.provenance
    d_provenance = trainer.global_d_provenance
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed = Trainer(config, restoring=True)
    resumed.restore(checkpoint)
    if recursive_hash(snapshot(resumed)) != saved_hash:
        raise AssertionError("Restored training state must exactly match the saved state")
    repeated = resumed.step(batch["image"], batch["structure"], batch["mask"])
    actual = snapshot(resumed)
    comparison = compare_next_step(expected, actual)
    torch.testing.assert_close(
        torch.tensor(list(second.values())),
        torch.tensor(list(repeated.values())),
        rtol=1e-4,
        atol=1e-5,
    )
    if comparison["outside_tolerance"]:
        raise AssertionError(json.dumps(comparison))
    report = {
        "status": "passed",
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "batch_size": len(batch["image"]),
        "precision": config.train.amp,
        "dataset_size": len(dataset),
        "dataset_fingerprint": dataset.fingerprint,
        "memory_budget_gib": config.train.memory_budget_gib,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "first_step_seconds": first_seconds,
        "first_step": first,
        "second_step": second,
        "checkpoint_restore_exact": True,
        "next_step_state_exact": recursive_hash(expected) == recursive_hash(actual),
        "next_step_comparison": comparison,
        "vgg_provenance": vgg_provenance,
        "global_d_provenance": d_provenance,
        "production_initialization": "Fresh seed; diagnostic weights are not reused",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    checkpoint.unlink()
    print(json.dumps({"event": "passed", **report}), flush=True)


if __name__ == "__main__":
    main()
