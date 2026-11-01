import pytest

from medfe.config import Config


@pytest.mark.parametrize(
    "section,name,value",
    [
        ("model", "base_channels", True),
        ("model", "base_channels", 1),
        ("model", "gaussian_sigma", float("nan")),
        ("model", "activation_checkpointing", 1),
        ("loss", "lpips", -1),
        ("loss", "style", float("inf")),
        ("loss", "vgg_profile", "unknown"),
        ("loss", "vgg_chunk_size", 0),
        ("loss", "vgg_chunk_size", True),
        ("train", "lr", 0),
        ("data", "batch_size", 0),
        ("data", "workers", -1),
        ("data", "image_size", 255),
        ("data", "mask_mode", "unknown"),
        ("data", "format", "unknown"),
        ("data", "allow_repeated_records", 1),
        ("data", "allow_repeated_records", True),
        ("data", "rtv_cache", 123),
        ("train", "memory_budget_gib", -1),
        ("train", "memory_budget_gib", float("nan")),
        ("train", "memory_budget_gib", float("inf")),
        ("train", "evaluation_batch_size", 0),
        ("train", "evaluation_batch_size", True),
        ("train", "adversarial_warmup_steps", -1),
        ("train", "adversarial_warmup_steps", True),
        ("train", "adversarial_warmup_steps", 1.5),
        ("train", "adversarial_ramp_steps", -1),
        ("train", "adversarial_ramp_steps", False),
        ("train", "adversarial_ramp_steps", 2.0),
        ("train", "device", "cudabroken"),
        ("train", "device", "cuda:-1"),
        ("train", "device", "cpu:bad"),
        ("discriminator", "local_crop_size", 48),
        ("train", "amp", "mixed"),
        ("train", "beta1", 1),
    ],
)
def test_invalid_settings_are_rejected(section, name, value):
    with pytest.raises(ValueError):
        Config.from_dict({section: {name: value}})


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda", "cuda:0", "cuda:12"])
def test_valid_device_syntax(device):
    assert Config.from_dict({"train": {"device": device}}).train.device == device


def test_yaml_roundtrip_and_typed_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  amp: 'off'\n  lr: 0.0002\n")
    config = Config.load(
        path,
        [
            "train.amp=off",
            "train.lr=1e-5",
            "data.batch_size=2",
            "train.allow_weight_download=false",
        ],
    )
    assert config.train.amp == "off" and config.train.lr == 1e-5
    assert config.data.batch_size == 2 and config.train.allow_weight_download is False
    assert Config.from_dict(config.as_dict()).as_dict() == config.as_dict()


@pytest.mark.parametrize(
    "value", [None, [], {"unknown": {}}, {"train": []}, {"model": {"typo": 1}}]
)
def test_unknown_or_invalid_sections(value):
    with pytest.raises(ValueError):
        Config.from_dict(value)


def test_resume_contract_only_ignores_execution_locations_and_observability():
    before, after = Config(), Config()
    after.train.output = "elsewhere"
    after.train.device = "cuda:1"
    after.data.workers = 0
    after.data.train_manifest = "relocated.jsonl"
    after.discriminator.pretrained_path = "no-longer-needed.pth"
    after.train.evaluate_every = 10
    after.data.rtv_cache = "relocated-cache"
    after.train.memory_budget_gib = 32.0
    after.train.evaluation_batch_size = 1
    assert before.resume_contract() == after.resume_contract()
    after.data.batch_size += 1
    assert before.resume_contract() != after.resume_contract()


def test_paired_defaults_preserve_existing_configuration():
    config = Config.from_dict({"data": {"train_manifest": "paired.jsonl"}})
    assert config.data.format == "paired" and config.data.rtv_cache == ""
    assert config.train.memory_budget_gib == 0
    assert config.train.evaluation_batch_size == 4


@pytest.mark.parametrize("cache", ["", "   "])
def test_indexed_format_requires_a_cache_destination(cache):
    with pytest.raises(ValueError, match="rtv_cache"):
        Config.from_dict({"data": {"format": "indexed_rtv", "rtv_cache": cache}})


def test_indexed_storage_can_move_but_data_semantics_cannot_change():
    first = Config.from_dict({"data": {"format": "indexed_rtv", "rtv_cache": "cache-a"}})
    second = Config.from_dict(first.as_dict())
    second.data.rtv_cache = "cache-b"
    second.train.memory_budget_gib = 24
    second.train.evaluation_batch_size = 2
    assert first.resume_contract() == second.resume_contract()
    second.data.format = "paired"
    assert first.resume_contract() != second.resume_contract()


def test_all_zero_objective_and_invalid_schedule_are_rejected():
    config = Config()
    for name in (
        "reconstruction",
        "lpips",
        "perceptual",
        "style",
        "structure",
        "texture",
        "adversarial",
    ):
        setattr(config.loss, name, 0)
    with pytest.raises(ValueError, match="At least one"):
        config.validate()
    config = Config()
    config.train.schedule = "linear"
    config.train.decay_start = config.train.epochs
    with pytest.raises(ValueError, match="decay"):
        config.validate()


def test_adversarial_schedule_defaults_and_resume_contract():
    before = Config()
    assert before.train.adversarial_warmup_steps == before.train.adversarial_ramp_steps == 0
    for name in ("adversarial_warmup_steps", "adversarial_ramp_steps"):
        after = Config.from_dict(before.as_dict())
        setattr(after.train, name, 5)
        after.validate()
        assert after.resume_contract() != before.resume_contract()
        assert Config.from_dict(after.as_dict()).as_dict() == after.as_dict()


def test_vgg_objective_profile_is_part_of_resume_contract():
    before = Config()
    after = Config.from_dict(before.as_dict())
    after.loss.vgg_profile = "released"
    after.validate()
    assert before.resume_contract() != after.resume_contract()
    assert Config.from_dict(after.as_dict()).loss.vgg_profile == "released"


def test_adversarial_only_warmup_has_no_objective_and_is_rejected():
    config = Config()
    for name in ("reconstruction", "lpips", "perceptual", "style", "structure", "texture"):
        setattr(config.loss, name, 0)
    config.train.adversarial_warmup_steps = 5
    with pytest.raises(ValueError, match="non-adversarial"):
        config.validate()
    config.train.adversarial_warmup_steps = 0
    config.validate()
