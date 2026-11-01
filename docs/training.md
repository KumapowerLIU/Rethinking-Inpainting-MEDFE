# Training and experiment profiles

## Current model

Use `configs/sd-partial-csa-mix-large-batch.yaml`. It selects the `sd_partial` backbone,
64 base channels, two residual blocks per encoder scale, four bottleneck blocks,
and five partial convolutions per fusion stream. Encoder and decoder are trained jointly
with MEDFE feature equalization. There is no separate coarse/refinement training stage.

The full-image raw prediction receives L1=1. Structure and texture heads each receive
L1=1 exactly once. The completed image receives VGG-LPIPS=1. Global and local PatchGANs
use RaLSGAN, with their generator terms summed and scaled by the scheduled GAN coefficient.
Separate VGG perceptual/style terms are zero. Both discriminators are randomly initialized;
Real-ESRGAN is not used by this checkpoint.

GAN coefficient at update `s`: `0.002 * clamp((s - 5000) / 10000, 0, 1)`.
The effective coefficient is logged. Full-batch relativistic means are preserved when
D and perceptual forwards are chunked. Chunk size 8 is not gradient accumulation;
the effective batch remains 128.

## Hardware and memory

The run trained on one **NVIDIA H100 80 GB HBM3**, batch **128**, 256² RGB,
BF16 autocast, activation checkpointing, and Adam `(lr=2e-4, betas=(0.5,0.999))`.
The Python allocator budget is **38 GiB**; observed process memory was about **33 GiB**.
A prior full D/G synthetic-input capacity check measured **25.925 GiB allocated** and
**30.551 GiB reserved**. These are different quantities: neither includes every driver
allocation, and neither promises capacity for arbitrary image sizes or software versions.

Validation uses FP32 and batch 4. The measured standalone 32-image inference at step120700
peaked below 1 GiB allocated with the full-width generator. This excludes the CPU checkpoint
load and should not be used to estimate training memory or high-resolution inference.

## Start, pause, resume

The README gives a 20-step startup check and a resume command. Set paths and batch before
starting a new experiment. Initial output directories must be empty. `--max-steps` is an
absolute optimizer-step ceiling; do not use a value lower than a restored step.

On SIGINT/SIGTERM the runner finishes the current complete update and saves `latest.pt`.
Do not use SIGKILL if you need the last in-progress update. Each save atomically replaces
`latest.pt`; the run does not retain a checkpoint for every step. Copy a stopped or pinned
checkpoint to a named archive when selecting an experiment.

Full checkpoints contain generator, both D networks, frozen metric weights, optimizers,
schedulers, scaler, RNG and sample cursor. Exact resume checks configuration, source files
and data fingerprints. Existing runs retain frozen `source/medfe` directories; run them
through that source when later code cleanup changes hashes. Do not bypass the checks just
to load an old run into a changed implementation. A generator-only export cannot resume training.

The paused run stopped at **120,994 complete updates** on 2026-11-01. It had completed
10 full passes plus 513,792 entries of the next pass, counting repeated face entries.
Its configured 60 epochs were a budget, not a convergence claim. A temporary GAN=0.2
branch was abandoned; this checkpoint resumed the preserved small-GAN lineage at step12,945.

## Profiles kept for comparison

| Config | Backbone | Reconstruction/perceptual objectives | GAN |
| --- | --- | --- | --- |
| `sd-partial-csa-mix-large-batch.yaml` | PartialConv residual | L1 + structure + texture + LPIPS, each1 | Patch D pair; ramps to0.002; face oversampling |
| `sd-partial-l1-lpips-large-batch.yaml` | Same modern backbone | Same losses | Same ramp; no deliberate duplicate entries |
| `released-large-batch.yaml` | Original-style encoder/deconvolution | L1=1, perceptual=0.2, style=250, auxiliaries1 | Patch D pair;0.2 |
| `paper.yaml` | Original-style encoder/deconvolution | Paper-oriented VGG/loss profile | Patch D pair;0.2 |
| `realesrgan-lpips.yaml` | Historical variant | L1/LPIPS/auxiliaries | Pretrained Real-ESRGAN global D |

Other `l1-*` configs preserve intermediate ablations. They are not the active checkpoint’s
recipe. The modern variant is not an exact reproduction of the 2020 architecture, masks,
training split or loss weights; [paper alignment](paper-alignment.md) records those distinctions.
