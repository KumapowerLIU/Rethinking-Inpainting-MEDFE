import ast
import copy
import hashlib
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations, parametrize, spectral_norm
from torch.utils.checkpoint import checkpoint

from medfe.discriminators import (
    REALESRGAN_D_SHA256,
    PatchDiscriminator,
    UNetDiscriminatorSN,
    load_realesrgan_weights,
    patch_discriminator_provenance,
    prepare_discriminator,
    validate_discriminator_provenance,
)
from medfe.losses import ralsgan_discriminator, ralsgan_generator
from medfe.weights import state_dict_sha256


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_local_discriminator_matches_legacy_structure():
    import importlib.util
    from functools import partial

    legacy_path = Path(__file__).parents[1] / "legacy/models/Discriminator.py"
    spec = importlib.util.spec_from_file_location("legacy_medfe_discriminator", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    original = legacy.NLayerDiscriminator(
        3, ndf=4, n_layers=3, norm_layer=partial(nn.InstanceNorm2d, affine=True)
    ).eval()
    current = PatchDiscriminator(width=4, layers=3).eval()
    # This mapping is an explicit test fixture, not a production migration:
    # the old hook state must not load silently into the new training contract.
    with pytest.raises(RuntimeError):
        current.load_state_dict(original.state_dict(), strict=True)
    translated = {
        key.replace("weight_orig", "parametrizations.weight.original")
        .replace("weight_u", "parametrizations.weight.0._u")
        .replace("weight_v", "parametrizations.weight.0._v"): value
        for key, value in original.state_dict().items()
    }
    current.load_state_dict(translated, strict=True)
    image = torch.randn(2, 3, 64, 64)
    assert torch.equal(current(image), original(image))
    assert current(image).shape == (2, 1, 2, 2)


def test_patch_uses_official_weight_and_bias_initialization():
    torch.manual_seed(17)
    model = PatchDiscriminator(width=16, layers=2)
    weights = []
    for convolution in model.model:
        if isinstance(convolution, nn.Conv2d):
            assert parametrize.is_parametrized(convolution, "weight")
            weights.append(convolution.parametrizations.weight.original.detach().flatten())
            assert torch.count_nonzero(convolution.bias) == 0
    weights = torch.cat(weights)
    assert abs(weights.mean()) < 0.001
    assert 0.019 < weights.std() < 0.021
    assert not any("weight_orig" in key for key in model.state_dict())


def test_patch_initial_sn_estimation_and_phase_iterations(monkeypatch):
    # The version-specific upstream method is observed only in this test.
    # Production updates use public parametrization/module APIs.
    calls = []
    original = parametrizations._SpectralNorm._power_method

    def record_iterations(self, weight, n_power_iterations):
        calls.append(n_power_iterations)
        return original(self, weight, n_power_iterations)

    monkeypatch.setattr(parametrizations._SpectralNorm, "_power_method", record_iterations)
    model = PatchDiscriminator(width=4, layers=1)
    assert calls == [15, 15, 15]  # No extra registration-time training forward.
    calls.clear()
    prepare_discriminator(model)  # Before D.
    assert calls == [1, 1, 1]
    calls.clear()
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    checkpoint(model, image, use_reentrant=False).mean().backward()
    assert calls == []  # Neither forward nor recomputation advances u/v.
    prepare_discriminator(model)  # Before G, after its D optimizer step.
    assert calls == [1, 1, 1]


def test_patch_phase_update_matches_one_power_iteration():
    model = PatchDiscriminator(width=4, layers=1)
    convolution = model.model[0]
    weight = convolution.parametrizations.weight.original.detach().flatten(1)
    sn = convolution.parametrizations.weight[0]
    expected_u = F.normalize(weight @ sn._v, dim=0, eps=1e-12)
    expected_v = F.normalize(weight.T @ expected_u, dim=0, eps=1e-12)
    prepare_discriminator(model)
    torch.testing.assert_close(sn._u, expected_u, rtol=0, atol=0)
    torch.testing.assert_close(sn._v, expected_v, rtol=0, atol=0)
    assert not any(module.training for module in model.modules())


def test_patch_provenance_rejects_legacy_and_incorrect_initialization():
    model = PatchDiscriminator(width=4, layers=1)
    provenance = patch_discriminator_provenance(model)
    validate_discriminator_provenance(provenance, model)
    assert provenance["initialization"] == "normal_0.02_zero_bias_v1"
    assert provenance["spectral_norm"] == "torch_parametrizations_initial_power_iterations_15_v1"
    assert provenance["test_fixture"] is False
    for changed in (
        {"architecture": "patch", "width": 4, "test_fixture": False},
        {**provenance, "initialization": "default"},
        {**provenance, "spectral_norm": "legacy_hook"},
        {**provenance, "test_fixture": 0},
        {**provenance, "layers": 3},
    ):
        with pytest.raises(ValueError, match="provenance"):
            validate_discriminator_provenance(changed, model)


@pytest.mark.parametrize("kind", ["local", "global"])
def test_spectral_norm_buffers_frozen_during_forwards_and_backward(kind):
    model = PatchDiscriminator(4, 1) if kind == "local" else UNetDiscriminatorSN(num_feat=4)
    prepare_discriminator(model)
    before = {key: value.clone() for key, value in model.named_buffers()}
    value = torch.randn(2, 3, 32, 32, requires_grad=True)
    checkpoint(model, value, use_reentrant=False).square().mean().backward()
    assert torch.isfinite(value.grad).all()
    assert all(torch.equal(before[k], v) for k, v in model.named_buffers())
    assert not any(module.training for module in model.modules())


def _step(model, real, fake, chunk):
    prepare_discriminator(model)

    def logits(value):
        return torch.cat(
            [checkpoint(model, part, use_reentrant=False) for part in value.split(chunk)], dim=0
        )

    loss = ralsgan_discriminator(logits(real), logits(fake.detach()))
    loss.backward()
    return loss


@pytest.mark.parametrize("kind", ["patch", "realesrgan"])
def test_chunked_gan_preserves_full_batch_parameter_gradients(kind):
    torch.manual_seed(11)
    full = PatchDiscriminator(4, 1) if kind == "patch" else UNetDiscriminatorSN(num_feat=4)
    chunked = copy.deepcopy(full)
    real, fake = torch.rand(3, 3, 32, 32), torch.rand(3, 3, 32, 32)
    a, b = _step(full, real, fake, 3), _step(chunked, real, fake, 1)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for first, second in zip(full.parameters(), chunked.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=3e-4, atol=3e-6)
    for first, second in zip(full.buffers(), chunked.buffers()):
        assert torch.equal(first, second)


@pytest.mark.parametrize("kind", ["patch", "realesrgan"])
def test_frozen_discriminator_preserves_generator_input_gradient(kind):
    model = PatchDiscriminator(4, 1) if kind == "patch" else UNetDiscriminatorSN(num_feat=4)
    model.requires_grad_(False)
    prepare_discriminator(model)
    before = state_dict_sha256(model.state_dict())
    prediction = torch.rand(2, 3, 32, 32, requires_grad=True)
    real = model(torch.rand_like(prediction))
    fake = checkpoint(model, prediction, use_reentrant=False)
    ralsgan_generator(real, fake).backward()
    assert prediction.grad.abs().sum() > 0 and torch.isfinite(prediction.grad).all()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert state_dict_sha256(model.state_dict()) == before


@pytest.mark.parametrize("kind", ["patch", "realesrgan"])
def test_prepare_rejects_low_precision_master_weights(kind):
    model = PatchDiscriminator(4, 1) if kind == "patch" else UNetDiscriminatorSN(num_feat=4)
    model.bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        prepare_discriminator(model)


def test_production_loader_rejects_unverified_weights_without_mutation(tmp_path):
    model = UNetDiscriminatorSN()
    before = state_dict_sha256(model.state_dict())
    path = tmp_path / "random.pth"
    torch.save({"params": model.state_dict()}, path)
    with pytest.raises(ValueError, match="SHA256"):
        load_realesrgan_weights(model, path)
    assert state_dict_sha256(model.state_dict()) == before


def test_official_discriminator_strict_load_if_local_asset_available():
    fallback = (
        Path(__file__).parents[2]
        / "artifacts/csa-stage2-realesrgan-launch-20261026"
        / "weights/RealESRGAN_x4plus_netD.pth"
    )
    path = Path(os.environ.get("MEDFE_REALESRGAN_D", str(fallback)))
    if not path.is_file():
        pytest.skip("Official checkpoint not supplied locally; tests never download assets")
    model = UNetDiscriminatorSN()
    provenance = load_realesrgan_weights(model, path)
    assert provenance["file_sha256"] == REALESRGAN_D_SHA256
    model.eval()
    image = torch.rand(1, 3, 16, 24, requires_grad=True)
    logits = model(image)
    assert logits.shape == (1, 1, 16, 24)
    logits.mean().backward()
    assert torch.isfinite(image.grad).all()


def test_state_hash_ignores_mapping_order_and_covers_buffers():
    state = {"b": torch.tensor(2.0), "a": torch.arange(4).reshape(2, 2)}
    assert state_dict_sha256(state) == state_dict_sha256(dict(reversed(list(state.items()))))
    changed = {**state, "b": torch.tensor(3.0)}
    assert state_dict_sha256(state) != state_dict_sha256(changed)


def test_official_architecture_output_and_all_gradients():
    source_path = Path(__file__).parent / "references/realesrgan_discriminator_arch.py.txt"
    source = source_path.read_bytes()
    assert (
        hashlib.sha256(source).hexdigest()
        == "7dadb69cb6649726096ef77d463fad9e733c59755b84d760f1d4e406da9a0b78"
    )
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "UNetDiscriminatorSN"
    )
    node.decorator_list = []  # BasicSR registration is not part of the architecture.
    namespace = {"nn": nn, "F": F, "spectral_norm": spectral_norm}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    upstream = namespace["UNetDiscriminatorSN"](num_in_ch=3).eval()
    current = UNetDiscriminatorSN().eval()
    upstream.load_state_dict(current.state_dict(), strict=True)
    a = torch.rand(1, 3, 16, 24, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    first, second = upstream(a), current(b)
    assert torch.equal(first, second)
    first.mean().backward()
    second.mean().backward()
    assert torch.equal(a.grad, b.grad)
    for reference, actual in zip(upstream.parameters(), current.parameters()):
        assert torch.equal(reference.grad, actual.grad)


@pytest.mark.parametrize("kind", ["patch", "realesrgan"])
def test_discriminator_adam_and_sn_buffers_resume_exactly(kind):
    torch.manual_seed(31)

    def factory():
        return PatchDiscriminator(4, 1) if kind == "patch" else UNetDiscriminatorSN(num_feat=4)

    original = factory()
    optimizer = torch.optim.Adam(original.parameters(), lr=2e-4)
    real, fake = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)

    def advance(model, opt):
        opt.zero_grad(set_to_none=True)
        loss = _step(model, real, fake, 1)
        opt.step()
        prepare_discriminator(model)  # The subsequent G phase advances SN once.
        return loss

    advance(original, optimizer)
    saved_model, saved_optimizer = (
        copy.deepcopy(original.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    expected_loss = advance(original, optimizer)
    restored = factory()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=2e-4)
    restored.load_state_dict(saved_model, strict=True)
    restored_optimizer.load_state_dict(saved_optimizer)
    actual_loss = advance(restored, restored_optimizer)
    assert torch.equal(expected_loss, actual_loss)
    assert state_dict_sha256(original.state_dict()) == state_dict_sha256(restored.state_dict())
