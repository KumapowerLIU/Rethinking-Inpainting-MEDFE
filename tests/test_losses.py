import pytest
import torch

from medfe.losses import auxiliary_losses, ralsgan_discriminator, ralsgan_generator


def test_ralsgan_matches_full_batch_formula_and_gradient():
    real = torch.tensor([[0.1, 0.7], [1.1, -0.2]])
    fake = torch.tensor([[-0.6, 0.2], [0.8, -1.3]], requires_grad=True)
    expected_d = (
        (real - fake.mean() - 1).square().mean() + (fake - real.mean() + 1).square().mean()
    ) / 2
    expected_g = (
        (real - fake.mean() + 1).square().mean() + (fake - real.mean() - 1).square().mean()
    ) / 2
    assert torch.equal(ralsgan_discriminator(real, fake), expected_d)
    actual = ralsgan_generator(real, fake)
    assert torch.equal(actual, expected_g)
    assert torch.equal(
        torch.autograd.grad(actual, fake, retain_graph=True)[0],
        torch.autograd.grad(expected_g, fake)[0],
    )
    # A per-chunk mean changes the relativistic objective.
    chunks = sum(ralsgan_generator(r, f) for r, f in zip(real.split(1), fake.split(1))) / 2
    assert not torch.allclose(actual, chunks)


def test_ralsgan_fp32_reductions_under_bf16_autocast():
    real = torch.randn(3, 1, 8, 8).bfloat16()
    fake = torch.randn_like(real).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        d, g = ralsgan_discriminator(real, fake), ralsgan_generator(real, fake)
    assert d.dtype == g.dtype == torch.float32
    g.backward()
    assert torch.isfinite(fake.grad).all()


def test_auxiliary_loss_has_no_backward_side_effect_and_unit_gradient():
    structure = torch.ones(2, 3, 4, 4, requires_grad=True)
    texture = torch.ones(2, 3, 4, 4, requires_grad=True)
    target = torch.zeros(2, 3, 8, 8, requires_grad=True)
    result = auxiliary_losses(structure, texture, target, target)
    assert set(result) == {"structure", "texture"}
    assert structure.grad is None and texture.grad is None
    sum(result.values()).backward()
    assert torch.equal(structure.grad, torch.full_like(structure, 1 / structure.numel()))
    assert torch.equal(texture.grad, torch.full_like(texture, 1 / texture.numel()))
    assert target.grad is None


@pytest.mark.parametrize("shape", [(0, 3, 8, 8), (2, 1, 8, 8), (2, 8, 8)])
def test_auxiliary_loss_rejects_invalid_rgb_batches(shape):
    value = torch.zeros(shape)
    with pytest.raises(ValueError):
        auxiliary_losses(value, value, value, value)


def test_empty_gan_logits_rejected():
    with pytest.raises(ValueError):
        ralsgan_generator(torch.empty(0), torch.ones(1))
