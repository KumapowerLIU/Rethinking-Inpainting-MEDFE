"""Single-device training orchestration with an explicit resumable data cursor."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .checkpoint import epoch_indices, seed_everything
from .config import Config
from .data import InpaintingDataset
from .trainer import Trainer


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def make_dataset(config: Config, *, validation: bool = False) -> InpaintingDataset:
    dataset_type = InpaintingDataset
    extras = {}
    if config.data.format == "indexed_rtv":
        from .indexed_data import IndexedRTVDataset

        dataset_type = IndexedRTVDataset
        extras["cache_root"] = config.data.rtv_cache
        extras["allow_repeated_records"] = config.data.allow_repeated_records and not validation
    return dataset_type(
        config.data.val_manifest if validation else config.data.train_manifest,
        image_size=config.data.image_size,
        seed=config.train.seed,
        training=not validation,
        mask_mode=config.data.mask_mode,
        min_fraction=config.data.min_fraction,
        max_fraction=config.data.max_fraction,
        **extras,
    )


def make_loader(
    dataset: InpaintingDataset, config: Config, epoch: int, offset: int = 0
) -> DataLoader:
    dataset.set_epoch(epoch)
    # Loader worker seeds and iterator creation must not consume the model's RNG.
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=epoch_indices(len(dataset), config.train.seed, epoch, offset),
        num_workers=config.data.workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.train.seed + epoch),
        pin_memory=config.train.device.startswith("cuda")
        or (config.train.device == "auto" and torch.cuda.is_available()),
    )


@contextlib.contextmanager
def run_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another trainer owns {output}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def train(
    config: Config, *, resume: str | Path | None = None, max_steps: int | None = None
) -> dict:
    """Run until an absolute step limit, epoch completion, or a graceful signal."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(config.train.output).resolve()
    with run_lock(output):
        return _train_locked(config, output, resume, max_steps)


def _train_locked(config: Config, output: Path, resume, max_steps) -> dict:
    from .evaluation import evaluate

    latest = output / "latest.pt"
    if resume is None and set(output.iterdir()) - {output / ".lock"}:
        raise FileExistsError(f"Output is not empty: {output}; choose a new run or --resume")
    # Resume into a new directory is allowed, but never overwrite another run's state.
    if resume is not None and latest.exists() and latest != Path(resume).resolve():
        raise ValueError("An existing output must resume its own latest.pt")
    dataset = make_dataset(config)
    validation = make_dataset(config, validation=True) if config.data.val_manifest else None
    seed_everything(config.train.seed)
    trainer = Trainer(config, restoring=resume is not None)
    if resume is not None:
        trainer.restore(resume)
    trainer.bind_data(dataset.fingerprint, len(dataset))
    if not (output / "config.yaml").exists():
        (output / "config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
    if not (output / "source").exists():
        shutil.copytree(
            Path(__file__).parent,
            output / "source" / "medfe",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, request_stop)
    started = time.monotonic()
    status = "running"

    def state():
        return {
            "status": status,
            "step": trainer.global_step,
            "completed_epochs": trainer.completed_epochs,
            "data": trainer.data_progress,
            "checkpoint": str(latest),
            "elapsed_seconds": time.monotonic() - started,
            "validation_fingerprint": validation.fingerprint if validation else None,
        }

    try:
        with (output / "metrics.jsonl").open("a", buffering=1) as log:
            log.write(json.dumps({"event": "resume" if resume else "start", **state()}) + "\n")
            while trainer.completed_epochs < config.train.epochs:
                if stop_requested or (max_steps is not None and trainer.global_step >= max_steps):
                    break
                progress = trainer.data_progress
                if progress["samples_in_epoch"] == len(dataset):
                    trainer.finish_epoch()
                    continue
                loader = make_loader(
                    dataset, config, trainer.completed_epochs, progress["samples_in_epoch"]
                )
                for batch in loader:
                    if stop_requested:
                        break
                    before = time.monotonic()
                    losses = trainer.step(batch["image"], batch["structure"], batch["mask"])
                    record = {
                        "event": "step",
                        "step": trainer.global_step,
                        "epoch": trainer.completed_epochs,
                        "seconds": time.monotonic() - before,
                        **losses,
                    }
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    if trainer.data_progress["samples_in_epoch"] == len(dataset):
                        trainer.finish_epoch()
                    if trainer.global_step % config.train.log_every == 0:
                        print(json.dumps(record, allow_nan=False), flush=True)
                    if trainer.global_step % config.train.save_every == 0:
                        trainer.save(latest)
                    if validation is not None and (
                        trainer.global_step % config.train.evaluate_every == 0
                    ):
                        metrics = evaluate(
                            trainer.generator,
                            validation,
                            output / "validation" / f"step-{trainer.global_step:08d}",
                            trainer.device,
                            limit=config.train.evaluation_images,
                            batch_size=config.train.evaluation_batch_size,
                        )
                        log.write(
                            json.dumps(
                                {"event": "validation", "step": trainer.global_step, **metrics},
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    atomic_json(output / "state.json", state())
                    if stop_requested or (
                        max_steps is not None and trainer.global_step >= max_steps
                    ):
                        break
            status = "completed" if trainer.completed_epochs == config.train.epochs else "stopped"
            trainer.save(latest)
            atomic_json(output / "state.json", state())
            log.write(json.dumps({"event": status, **state()}) + "\n")
            return state()
    except BaseException as error:
        if stop_requested:
            try:
                trainer.save(latest)
            except Exception:
                # A partial D/G update (or failed write) cannot become a clean stop.
                pass
            else:
                status = "stopped"
                atomic_json(output / "state.json", state())
                return state()
        status = "failed"
        atomic_json(output / "state.json", {**state(), "error": str(error)})
        # Preserve the last complete checkpoint; the current D/G update may be partial.
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
