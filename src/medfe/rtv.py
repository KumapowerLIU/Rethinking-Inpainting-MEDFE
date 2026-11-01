"""RTV structure targets following the original MEDFE MATLAB preprocessing.

The equations, zero-padded Gaussian filtering, forward differences, lambda/2,
and iterative sigma schedule match ``legacy/data/Matlab/tsmooth.m`` (Li Xu,
CUHK, 2012; retained with its original non-commercial license).  We solve its
weighted four-neighbour linear system with SciPy's direct sparse LU solver.
This matches the MATLAB ``A \\ b`` branch, not bit-for-bit MATLAB's optional
incomplete-Cholesky PCG branch (tolerance 0.1, maximum 100 iterations).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu

RTV_VERSION = "medfe_rtv_matlab_direct_v1"


@dataclass(frozen=True)
class RTVConfig:
    lambda_: float = 0.015
    sigma: float = 3.0
    sharpness: float = 0.001
    iterations: int = 3

    def __post_init__(self) -> None:
        for name in ("lambda_", "sigma", "sharpness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"RTV {name} must be a finite number")
            if not math.isfinite(value) or value < 0 or (name != "lambda_" and value == 0):
                raise ValueError(f"Invalid RTV {name}: {value!r}")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("RTV iterations must be a positive integer")

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = {"algorithm": RTV_VERSION, "parameters": self.as_dict()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _gaussian_kernel(sigma: float) -> np.ndarray:
    # MATLAB round for positive values rounds half away from zero; bitor(..., 1)
    # forces odd support. scipy's default Gaussian truncation would be different.
    size = max(1, int(math.floor(5 * sigma + 0.5)) | 1)
    positions = np.arange(size, dtype=np.float64) - (size - 1) / 2
    values = np.exp(-(positions * positions) / (2 * sigma * sigma))
    values[values < np.finfo(np.float64).eps * values.max()] = 0
    return values / values.sum()


def _forward_differences(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)
    dx[:, :-1] = image[:, 1:] - image[:, :-1]
    dy[:-1] = image[1:] - image[:-1]
    return dx, dy


def _texture_weights(image: np.ndarray, sigma: float, sharpness: float):
    dx, dy = _forward_differences(image)
    total_variation = np.maximum(np.sqrt(dx * dx + dy * dy).mean(axis=2), sharpness)
    kernel = _gaussian_kernel(sigma)
    smooth = ndimage.convolve1d(image, kernel, axis=1, mode="constant", cval=0.0)
    smooth = ndimage.convolve1d(smooth, kernel, axis=0, mode="constant", cval=0.0)
    sx, sy = _forward_differences(smooth)
    wx = 1.0 / (total_variation * np.maximum(np.abs(sx).mean(axis=2), 0.001))
    wy = 1.0 / (total_variation * np.maximum(np.abs(sy).mean(axis=2), 0.001))
    wx[:, -1] = 0
    wy[-1] = 0
    return wx, wy


def _weighted_system(wx: np.ndarray, wy: np.ndarray, strength: float):
    """I + strength * weighted grid Laplacian, using row-major indexing."""
    height, width = wx.shape
    size = height * width
    indices = np.arange(size, dtype=np.int64).reshape(height, width)
    starts = np.concatenate((indices[:, :-1].ravel(), indices[:-1].ravel()))
    ends = np.concatenate((indices[:, 1:].ravel(), indices[1:].ravel()))
    weights = strength * np.concatenate((wx[:, :-1].ravel(), wy[:-1].ravel()))
    diagonal = np.ones(size, dtype=np.float64)
    np.add.at(diagonal, starts, weights)
    np.add.at(diagonal, ends, weights)
    rows = np.concatenate((np.arange(size), starts, ends))
    cols = np.concatenate((np.arange(size), ends, starts))
    values = np.concatenate((diagonal, -weights, -weights))
    return sparse.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()


def rtv_smooth(image: np.ndarray, config: RTVConfig | None = None) -> np.ndarray:
    """Return an HWC RGB float64 structure image from RGB floats in [0, 1].

    Boundary derivatives have zero forward differences; Gaussian filtering uses
    zero extension exactly as MATLAB ``conv2(..., 'same')``. Each iteration has
    the original RGB image as its right-hand side. No image-size normalization
    or ordinary-blur substitution is performed.
    """
    config = RTVConfig() if config is None else config
    if not isinstance(config, RTVConfig):
        raise TypeError("config must be RTVConfig")
    source = np.asarray(image)
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or min(source.shape[:2]) < 1
        or not np.issubdtype(source.dtype, np.floating)
    ):
        raise ValueError("RTV input must be nonempty HWC RGB floating point")
    if not np.isfinite(source).all() or source.min() < 0 or source.max() > 1:
        raise ValueError("RTV input must contain finite values in [0, 1]")
    source = source.astype(np.float64, copy=True)
    current = source.copy()
    sigma = float(config.sigma)
    if config.lambda_ == 0:
        return current
    for _ in range(config.iterations):
        wx, wy = _texture_weights(current, sigma, config.sharpness)
        matrix = _weighted_system(wx, wy, config.lambda_ / 2.0)
        current = splu(matrix).solve(source.reshape(-1, 3)).reshape(source.shape)
        if not np.isfinite(current).all():
            raise FloatingPointError("RTV sparse solve produced nonfinite values")
        sigma = max(sigma / 2.0, 0.5)
    # This positive weighted Laplacian obeys the discrete maximum principle;
    # clip only floating-point roundoff at the [0, 1] endpoints.
    if current.min() < -1e-8 or current.max() > 1 + 1e-8:
        raise FloatingPointError("RTV solution violates the expected RGB range")
    return np.clip(current, 0.0, 1.0)
