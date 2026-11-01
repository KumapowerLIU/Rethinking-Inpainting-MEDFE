import pytest
import torch

from medfe.masks import center_mask, random_mask, sample_seed


@pytest.mark.parametrize("kind", [None, "rectangle", "brush", "union"])
@pytest.mark.parametrize("shape", [(32, 32), (24, 43), (8, 9)])
def test_random_masks_have_bounded_coverage_and_deterministic_geometry(kind, shape):
    outputs = []
    for seed in range(12):
        value = random_mask(*shape, seed=seed, kind=kind)
        assert value.shape == (1, *shape)
        assert value.dtype == torch.bool
        assert 0.1 <= value.float().mean().item() <= 0.4
        assert torch.equal(value, random_mask(*shape, seed=seed, kind=kind))
        outputs.append(value)
    assert any(not torch.equal(outputs[0], value) for value in outputs[1:])


def test_center_mask_is_explicit_paper_baseline():
    mask = center_mask(256, 256)
    assert mask.sum() == 128 * 128
    assert mask[:, 64:192, 64:192].all()
    assert not mask[:, :64].any()


def test_epoch_index_and_rng_streams_are_independent():
    values = {
        sample_seed(2019, e, i, s)
        for e in range(2)
        for i in range(4)
        for s in ["mask", "transform"]
    }
    assert len(values) == 16
    assert sample_seed(2019, 0, 1) == sample_seed(2019, 0, 1)


@pytest.mark.parametrize("args", [(-1, 8, 1), (8, 0, 1), (1, 1, 1)])
def test_impossible_geometry_is_rejected(args):
    with pytest.raises(ValueError):
        random_mask(*args)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_fraction=0),
        dict(max_fraction=1),
        dict(min_fraction=0.5, max_fraction=0.2),
        dict(kind="noise"),
    ],
)
def test_invalid_coverage_or_family(kwargs):
    with pytest.raises(ValueError):
        random_mask(32, 32, 1, **kwargs)
