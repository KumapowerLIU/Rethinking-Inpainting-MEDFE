"""Deterministic per-sample hole masks: True means missing/unknown."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from PIL import Image, ImageDraw


def sample_seed(seed: int, epoch: int, index: int, stream: str = "mask") -> int:
    """Stable independent RNG streams, unaffected by workers or Python hashes."""
    value = f"medfe-v1|{seed}|{epoch}|{index}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _geometry(height: int, width: int) -> None:
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("Mask height and width must be positive integers")


def _rectangle(height: int, width: int, rng, lower: int, upper: int):
    # Sample among feasible rectangle geometries, so the coverage contract also
    # holds on non-square and small test images.
    possibilities = []
    for rh in range(1, height + 1):
        left = max(1, math.ceil(lower / rh))
        right = min(width, upper // rh)
        if left <= right:
            possibilities.append((rh, left, right))
    if not possibilities:
        raise ValueError("No rectangle fits the requested mask coverage")
    rh, low, high = possibilities[int(rng.integers(len(possibilities)))]
    rw = int(rng.integers(low, high + 1))
    y = int(rng.integers(height - rh + 1))
    x = int(rng.integers(width - rw + 1))
    result = np.zeros((height, width), dtype=bool)
    result[y : y + rh, x : x + rw] = True
    return result


def _brush(height: int, width: int, rng, target: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    short_side = min(height, width)
    for _ in range(24):
        x = float(rng.uniform(0, width))
        y = float(rng.uniform(0, height))
        thickness = max(1, int(rng.uniform(0.025, 0.14) * short_side))
        for _ in range(int(rng.integers(2, 7))):
            angle = float(rng.uniform(0, 2 * math.pi))
            length = float(rng.uniform(0.06, 0.38) * short_side)
            nx = float(np.clip(x + math.cos(angle) * length, 0, width - 1))
            ny = float(np.clip(y + math.sin(angle) * length, 0, height - 1))
            draw.line((x, y, nx, ny), fill=255, width=thickness)
            radius = thickness / 2
            draw.ellipse((nx - radius, ny - radius, nx + radius, ny + radius), fill=255)
            x, y = nx, ny
        if np.count_nonzero(canvas) >= target:
            break
    return np.asarray(canvas, dtype=np.uint8) > 0


def random_mask(
    height: int,
    width: int,
    seed: int,
    min_fraction: float = 0.1,
    max_fraction: float = 0.4,
    *,
    kind: str | None = None,
) -> torch.Tensor:
    """Sample a rectangle, brush, or union mask with bounded hole coverage.

    ``kind`` is optional for controlled comparisons; the default samples all
    three families. Rejection sampling preserves geometric strokes. A rectangle
    is a bounded fallback if a requested brush cannot satisfy a tiny geometry.
    """
    _geometry(height, width)
    if (
        not math.isfinite(min_fraction)
        or not math.isfinite(max_fraction)
        or not 0 < min_fraction <= max_fraction < 1
    ):
        raise ValueError("Require 0 < min_fraction <= max_fraction < 1")
    if type(seed) is not int:
        raise ValueError("Mask seed must be an integer")
    if kind not in {None, "rectangle", "brush", "union"}:
        raise ValueError("Mask kind must be rectangle, brush, union, or None")
    area = height * width
    lower, upper = math.ceil(min_fraction * area), math.floor(max_fraction * area)
    if lower > upper:
        raise ValueError("Mask coverage interval contains no integer pixel count")
    rng = np.random.default_rng(sample_seed(seed, 0, 0, "geometry"))
    family = kind or ("rectangle", "brush", "union")[int(rng.integers(3))]
    for _ in range(32):
        if family == "rectangle":
            mask = _rectangle(height, width, rng, lower, upper)
        else:
            target = int(rng.integers(lower, upper + 1))
            mask = _brush(height, width, rng, target if family == "brush" else max(1, target // 2))
            if family == "union":
                rectangle = _rectangle(height, width, rng, max(1, lower // 3), max(1, upper // 2))
                mask = mask | rectangle
        if lower <= int(mask.sum()) <= upper:
            return torch.from_numpy(mask.copy()).unsqueeze(0)
    return torch.from_numpy(_rectangle(height, width, rng, lower, upper)).unsqueeze(0)


def center_mask(height: int, width: int, fraction: float = 0.25) -> torch.Tensor:
    """Explicit fixed central-hole baseline; default is 128x128 on 256x256."""
    _geometry(height, width)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Center mask fraction must be between zero and one")
    rh = max(1, min(height, int(round(height * math.sqrt(fraction)))))
    rw = max(1, min(width, int(round(width * math.sqrt(fraction)))))
    mask = torch.zeros((1, height, width), dtype=torch.bool)
    top, left = (height - rh) // 2, (width - rw) // 2
    mask[:, top : top + rh, left : left + rw] = True
    return mask
