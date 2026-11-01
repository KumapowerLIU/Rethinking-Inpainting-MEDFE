import math

import numpy as np
import pytest
from scipy.linalg import solve

from medfe.rtv import RTVConfig, rtv_smooth


def dense_reference(image, config):
    """Independent small-grid oracle: explicit Gaussian and dense edge system."""
    height, width, channels = image.shape
    current = image.copy()
    sigma = config.sigma
    for _ in range(config.iterations):
        length = int(math.floor(5 * sigma + 0.5)) | 1
        radius = length // 2
        kernel = np.array(
            [math.exp(-k * k / (2 * sigma * sigma)) for k in range(-radius, radius + 1)]
        )
        kernel /= kernel.sum()
        blurred = np.zeros_like(current)
        for y in range(height):
            for x in range(width):
                for ky in range(-radius, radius + 1):
                    for kx in range(-radius, radius + 1):
                        if 0 <= y + ky < height and 0 <= x + kx < width:
                            blurred[y, x] += (
                                current[y + ky, x + kx] * kernel[ky + radius] * kernel[kx + radius]
                            )
        matrix = np.eye(height * width)
        for y in range(height):
            for x in range(width):
                dx = current[y, x + 1] - current[y, x] if x + 1 < width else np.zeros(channels)
                dy = current[y + 1, x] - current[y, x] if y + 1 < height else np.zeros(channels)
                total = max(np.sqrt(dx**2 + dy**2).mean(), config.sharpness)
                for ny, nx in ((y, x + 1), (y + 1, x)):
                    if ny >= height or nx >= width:
                        continue
                    inherent = max(np.abs(blurred[ny, nx] - blurred[y, x]).mean(), 0.001)
                    weight = config.lambda_ / (2 * total * inherent)
                    first, second = y * width + x, ny * width + nx
                    matrix[first, first] += weight
                    matrix[second, second] += weight
                    matrix[first, second] -= weight
                    matrix[second, first] -= weight
        # The independently assembled system is SPD. Use dense Cholesky, a
        # different algorithm from the production sparse LU, and verify its
        # residual before treating it as an oracle. Some older NumPy OpenBLAS
        # CPU kernels return incorrect multi-RHS LU solutions on this host.
        rhs = image.reshape(-1, channels)
        solution = solve(matrix, rhs, assume_a="pos")
        np.testing.assert_allclose(matrix @ solution, rhs, atol=2e-12, rtol=2e-12)
        current = solution.reshape(image.shape)
        sigma = max(sigma / 2, 0.5)
    return current


@pytest.mark.parametrize("shape", [(4, 7, 3), (7, 3, 3), (1, 5, 3)])
def test_rtv_matches_independent_dense_matlab_equations(shape):
    image = np.random.default_rng(42).uniform(0.05, 0.95, shape)
    config = RTVConfig(sigma=1.5, iterations=2)
    np.testing.assert_allclose(
        rtv_smooth(image, config), dense_reference(image, config), atol=2e-12, rtol=2e-12
    )


def test_constant_rgb_and_single_pixel_are_preserved():
    color = np.array([0.13, 0.51, 0.89])
    image = np.broadcast_to(color, (9, 13, 3)).copy()
    np.testing.assert_allclose(rtv_smooth(image), image, atol=5e-12, rtol=0)
    np.testing.assert_array_equal(rtv_smooth(color.reshape(1, 1, 3)), color.reshape(1, 1, 3))


def test_rtv_reduces_texture_and_preserves_the_large_edge():
    grid = np.indices((24, 40)).sum(0)
    image = np.where(np.indices((24, 40))[1] < 20, 0.2, 0.8)
    image = np.repeat((image + 0.065 * (-1.0) ** grid)[..., None], 3, axis=2)
    result = rtv_smooth(image)
    assert result[4:-4, 3:16].std() < image[4:-4, 3:16].std() * 0.25
    assert result[4:-4, 25:36].mean() - result[4:-4, 3:16].mean() > 0.5
    assert result.min() >= image.min() - 1e-10
    assert result.max() <= image.max() + 1e-10
    np.testing.assert_array_equal(rtv_smooth(image), result)


def test_zero_strength_is_identity_and_input_is_not_mutated():
    image = np.random.default_rng(7).random((6, 8, 3)).astype(np.float32)
    saved = image.copy()
    np.testing.assert_array_equal(rtv_smooth(image, RTVConfig(lambda_=0)), image)
    rtv_smooth(image)
    np.testing.assert_array_equal(image, saved)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma=0),
        dict(lambda_=-1),
        dict(sharpness=float("nan")),
        dict(iterations=0),
        dict(iterations=True),
    ],
)
def test_invalid_rtv_configuration(kwargs):
    with pytest.raises(ValueError):
        RTVConfig(**kwargs)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((5, 5), dtype=float),
        np.zeros((5, 5, 3), dtype=np.uint8),
        np.full((5, 5, 3), np.nan),
        np.full((5, 5, 3), 1.1),
    ],
)
def test_invalid_rtv_input(image):
    with pytest.raises(ValueError):
        rtv_smooth(image)


def test_rtv_fingerprint_records_all_parameters():
    base = RTVConfig()
    for config in [
        RTVConfig(lambda_=0.02),
        RTVConfig(sigma=2),
        RTVConfig(sharpness=0.01),
        RTVConfig(iterations=4),
    ]:
        assert config.fingerprint != base.fingerprint
    assert RTVConfig().fingerprint == base.fingerprint
