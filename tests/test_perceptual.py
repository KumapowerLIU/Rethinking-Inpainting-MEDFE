import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from medfe.perceptual import LPIPSLoss, SharedVGG
from medfe.weights import state_dict_sha256


class FixtureLPIPS(nn.Module):
    """Explicit tiny test metric, never presented as pretrained LPIPS."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.lin = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(4, 1, 1, bias=False))

    def forward(self, prediction, target, normalize=False):
        assert normalize is False
        return self.lin((self.net(prediction) - self.net(target)).square()).mean((1, 2, 3))


def fixture_vgg():
    layers = [nn.Identity() for _ in range(31)]
    layers[0] = nn.Conv2d(3, 3, 1)
    for index in (4, 9, 16, 23, 30):
        layers[index] = nn.MaxPool2d(2, 2)
    return nn.Sequential(*layers)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_lpips_chunking_matches_whole_batch_value_and_prediction_gradient():
    torch.manual_seed(21)
    a = LPIPSLoss(chunk_size=5, fixture_network=FixtureLPIPS())
    b = LPIPSLoss(chunk_size=2, fixture_network=copy.deepcopy(a.network))
    prediction = torch.rand(5, 3, 16, 16).requires_grad_()
    copied = prediction.detach().clone().requires_grad_()
    target = torch.rand_like(prediction)
    full, chunked = a(prediction, target), b(copied, target)
    full.backward()
    chunked.backward()
    torch.testing.assert_close(full, chunked)
    torch.testing.assert_close(prediction.grad, copied.grad)
    assert prediction.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in a.parameters())
    assert a.provenance["fixture"] and not a.provenance["pretrained"]


def test_lpips_stays_eval_and_fp32_under_autocast():
    metric = LPIPSLoss(fixture_network=FixtureLPIPS())
    metric.train()
    assert not any(module.training for module in metric.modules())
    prediction = torch.rand(2, 3, 16, 16).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = metric(prediction, torch.zeros_like(prediction))
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("kind", ["lpips", "vgg", "vgg_released"])
def test_self_contained_metric_restore_and_provenance(kind):
    if kind == "lpips":
        original = LPIPSLoss(fixture_network=FixtureLPIPS())
        restored = LPIPSLoss(restoring=True, fixture_network=FixtureLPIPS())
    else:
        profile = "released" if kind == "vgg_released" else "paper"
        original = SharedVGG(profile=profile, fixture_features=fixture_vgg())
        restored = SharedVGG(profile=profile, restoring=True, fixture_features=fixture_vgg())
    prediction, target = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    with pytest.raises(RuntimeError, match="awaiting"):
        restored(prediction, target)
    restored.restore_checkpoint(original.state_dict(), original.provenance)
    first, second = original(prediction, target), restored(prediction, target)
    if isinstance(first, dict):
        for name in first:
            assert torch.equal(first[name], second[name])
    else:
        assert torch.equal(first, second)
    assert restored.provenance == original.provenance


@pytest.mark.parametrize("kind", ["lpips", "vgg", "vgg_released"])
def test_changed_metric_weights_are_rejected_and_forward_remains_blocked(kind):
    if kind == "lpips":
        metric = LPIPSLoss(fixture_network=FixtureLPIPS())
    else:
        profile = "released" if kind == "vgg_released" else "paper"
        metric = SharedVGG(profile=profile, fixture_features=fixture_vgg())
    state = copy.deepcopy(metric.state_dict())
    key = next(key for key in state if "weight" in key)
    state[key].add_(0.5)
    with pytest.raises(ValueError, match="provenance"):
        metric.restore_checkpoint(state, metric.provenance)
    with pytest.raises(RuntimeError, match="awaiting"):
        metric(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))


def test_fixture_cannot_be_restored_as_production_metric():
    metric = LPIPSLoss(fixture_network=FixtureLPIPS())
    provenance = {**metric.provenance, "fixture": False, "pretrained": True}
    with pytest.raises(ValueError, match="provenance"):
        metric.restore_checkpoint(metric.state_dict(), provenance)


def test_vgg_correct_layer_shapes_and_imagenet_normalization():
    features = fixture_vgg()
    features[0] = nn.Identity()
    metric = SharedVGG(fixture_features=features)
    value = torch.zeros(1, 3, 32, 32)
    extracted = metric.extract_features(value)
    assert extracted["relu5_1"].shape[-2:] == (2, 2)
    assert set(extracted) == set(SharedVGG.CONTENT)
    expected = (value + 1) * 0.5
    expected = (expected - metric.mean) / metric.std
    assert torch.equal(extracted["relu1_1"], expected)


@pytest.mark.parametrize("profile", ["paper", "released"])
def test_vgg_two_losses_share_one_feature_pass_per_image(profile):
    metric = SharedVGG(profile=profile, chunk_size=8, fixture_features=fixture_vgg())
    calls = []
    handle = metric.network[0].register_forward_hook(lambda *args: calls.append(1))
    result = metric.losses(torch.ones(2, 3, 32, 32), torch.zeros(2, 3, 32, 32))
    handle.remove()
    assert len(calls) == 2
    assert set(result) == {"perceptual", "style"}


@pytest.mark.parametrize("profile", ["paper", "released"])
def test_vgg_chunks_preserve_losses_gradients_and_fp32(profile):
    full = SharedVGG(profile=profile, chunk_size=5, fixture_features=fixture_vgg())
    chunked = SharedVGG(profile=profile, chunk_size=2, fixture_features=copy.deepcopy(full.network))
    prediction = torch.rand(5, 3, 32, 32).requires_grad_()
    copied = prediction.detach().clone().requires_grad_()
    target = torch.rand_like(prediction).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a, b = full(prediction, target), chunked(copied, target)
    for key in a:
        assert a[key].dtype == b[key].dtype == torch.float32
        torch.testing.assert_close(a[key], b[key])
    sum(a.values()).backward()
    sum(b.values()).backward()
    torch.testing.assert_close(prediction.grad, copied.grad)
    assert target.grad is None
    assert all(parameter.grad is None for parameter in full.parameters())


def test_gram_matches_channel_spatial_normalization():
    assert (
        SharedVGG.STYLE
        == SharedVGG.CONTENT
        == ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    )
    value = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    expected = torch.tensor([[[5.0, 11.0], [11.0, 25.0]]]) / 4
    assert torch.equal(SharedVGG.gram(value), expected)


def test_vgg_fails_without_cached_pretrained_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not cached"):
        SharedVGG(allow_download=False)


def test_restore_vgg_skeleton_does_not_need_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    model = SharedVGG(restoring=True)
    assert not model.provenance["pretrained"]
    with pytest.raises(RuntimeError, match="awaiting"):
        model(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))


def test_released_vgg_uses_raw_rgb_and_correct_relu5_2():
    features = fixture_vgg()
    features[0] = nn.Identity()
    metric = SharedVGG(profile="released", fixture_features=features)
    value = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    extracted = metric.extract_features(value)
    assert torch.equal(extracted["relu1_1"], value)
    assert extracted["relu5_1"].shape[-2:] == (2, 2)
    assert extracted["relu5_2"].shape[-2:] == (2, 2)
    assert metric.provenance["profile"] == "released"
    assert metric.provenance["input_normalization"] == "none"
    assert metric.provenance["mean"] is metric.provenance["std"] is None
    assert metric.provenance["style_layers"] == ["relu2_2", "relu3_3", "relu4_3", "relu5_2"]


def _corrected_released_vgg_reference(network, prediction, target):
    """Explicit legacy blocks and equations, fixing only the relu5_2 wiring."""
    blocks = (
        ("relu1_1", 0, 2),
        ("relu1_2", 2, 4),
        ("relu2_1", 4, 7),
        ("relu2_2", 7, 9),
        ("relu3_1", 9, 12),
        ("relu3_2", 12, 14),
        ("relu3_3", 14, 16),
        ("relu4_1", 16, 19),
        ("relu4_2", 19, 21),
        ("relu4_3", 21, 23),
        ("relu5_1", 23, 26),
        ("relu5_2", 26, 28),
    )

    def features(value):
        result = {}
        for name, start, stop in blocks:
            value = network[start:stop](value)
            result[name] = value
        return result

    def gram(value):
        batch, channels, height, width = value.shape
        flattened = value.view(batch, channels, height * width)
        return flattened.bmm(flattened.transpose(1, 2)) / (height * width * channels)

    x, y = features(prediction), features(target.detach())
    perceptual = sum(
        F.l1_loss(x[name], y[name])
        for name in ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    )
    style = sum(
        F.l1_loss(gram(x[name]), gram(y[name]))
        for name in ("relu2_2", "relu3_3", "relu4_3", "relu5_2")
    )
    return {"perceptual": perceptual, "style": style}


@pytest.mark.parametrize("chunk_size", [2, 5])
def test_released_vgg_matches_corrected_legacy_losses_and_gradients(chunk_size):
    torch.manual_seed(113)
    layers = []
    for count in (2, 2, 3, 3, 3):
        for _ in range(count):
            conv = nn.Conv2d(3, 3, 1)
            with torch.no_grad():
                conv.weight.copy_(0.7 * torch.eye(3).reshape(3, 3, 1, 1) + 0.03)
                conv.bias.fill_(0.05)
            layers.extend((conv, nn.ReLU()))
        layers.append(nn.MaxPool2d(2, 2))
    metric = SharedVGG(
        profile="released", chunk_size=chunk_size, fixture_features=nn.Sequential(*layers)
    )
    prediction = (torch.rand(5, 3, 32, 32) * 2 - 1).requires_grad_()
    reference_prediction = prediction.detach().clone().requires_grad_()
    target = (torch.rand_like(prediction) * 2 - 1).requires_grad_()
    actual = metric(prediction, target)
    expected = _corrected_released_vgg_reference(metric.network, reference_prediction, target)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7)
    (0.2 * actual["perceptual"] + 250 * actual["style"]).backward()
    (0.2 * expected["perceptual"] + 250 * expected["style"]).backward()
    torch.testing.assert_close(prediction.grad, reference_prediction.grad, rtol=1e-5, atol=1e-7)
    assert prediction.grad.abs().sum() > 0
    assert target.grad is None
    assert all(parameter.grad is None for parameter in metric.parameters())


@pytest.mark.parametrize(
    "source_profile,target_profile", [("paper", "released"), ("released", "paper")]
)
def test_vgg_rejects_restore_between_objective_profiles(source_profile, target_profile):
    source = SharedVGG(profile=source_profile, fixture_features=fixture_vgg())
    target = SharedVGG(profile=target_profile, restoring=True, fixture_features=fixture_vgg())
    with pytest.raises(ValueError, match="provenance"):
        target.restore_checkpoint(source.state_dict(), source.provenance)
    with pytest.raises(RuntimeError, match="awaiting"):
        target(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))


@pytest.mark.parametrize(
    "change",
    [
        {"profile": "paper"},
        {"input_normalization": "imagenet"},
        {"style_layers": ["relu5_1"]},
        {"content_layers": ["relu1_1"]},
        {"mean": [0.485, 0.456, 0.406]},
    ],
)
def test_vgg_rejects_changed_objective_provenance(change):
    metric = SharedVGG(profile="released", fixture_features=fixture_vgg())
    with pytest.raises(ValueError, match="provenance"):
        metric.restore_checkpoint(metric.state_dict(), {**metric.provenance, **change})
    with pytest.raises(RuntimeError, match="awaiting"):
        metric(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))


def test_vgg_rejects_unknown_profile():
    with pytest.raises(ValueError, match="profile"):
        SharedVGG(profile="legacy_typo", fixture_features=fixture_vgg())


@pytest.mark.parametrize("name", ["mean", "std"])
def test_vgg_rejects_normalization_buffer_drift_even_with_matching_state_hash(name):
    metric = SharedVGG(fixture_features=fixture_vgg())
    state = copy.deepcopy(metric.state_dict())
    state[name].add_(0.1)
    provenance = {**metric.provenance, "state_sha256": state_dict_sha256(state)}
    with pytest.raises(ValueError, match="normalization buffers"):
        metric.restore_checkpoint(state, provenance)
    with pytest.raises(RuntimeError, match="awaiting"):
        metric(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))
