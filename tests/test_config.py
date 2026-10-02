"""Config loading: defaults, YAML, overrides, error messages. (No torch needed.)"""

import pytest
import yaml

from latentgen.config import Config, apply_overrides, from_dict, load_config, to_dict


def test_defaults_round_trip():
    cfg = Config()
    assert cfg.vqvae.model.codebook_size == 256
    assert cfg.vqvae.train.microbatches_per_step == 8
    rebuilt = from_dict(Config, to_dict(cfg))
    assert to_dict(rebuilt) == to_dict(cfg)


def test_overrides_are_yaml_typed():
    data = apply_overrides(
        {},
        [
            "vqvae.train.epochs=3",
            "data.image_dir=/x/y",
            "generate.keep_schedule=[0.1, 0.5]",
            "project.seed=null",
        ],
    )
    cfg = from_dict(Config, data)
    assert cfg.vqvae.train.epochs == 3 and isinstance(cfg.vqvae.train.epochs, int)
    assert cfg.data.image_dir == "/x/y"
    assert cfg.generate.keep_schedule == (0.1, 0.5)
    assert cfg.project.seed is None


def test_unknown_key_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown config key"):
        from_dict(Config, {"vqvae": {"model": {"hidden_sise": 1}}})


def test_bad_batch_multiple():
    cfg = from_dict(Config, {"vqvae": {"train": {"batch_size": 10, "microbatch_size": 4}}})
    with pytest.raises(ValueError, match="multiple"):
        _ = cfg.vqvae.train.microbatches_per_step


def test_shipped_configs_load(tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parent.parent / "configs"
    for name in (
        "ffhq512.yaml",
        "my_dataset.yaml",
        "smoke_test.yaml",
        "smoke_test_audio.yaml",
        "examples/hf_ffhq128.yaml",
        "examples/hf_imagenet256.yaml",
        "examples/hf_librispeech.yaml",
    ):
        cfg = load_config(root / name)
        assert isinstance(cfg, Config)
    # and a config round-trips through YAML
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(to_dict(load_config(root / "smoke_test.yaml"))))
    assert load_config(p).data.image_size == 64


def test_partial_sections_keep_the_stage_defaults():
    """`cgan: {train: {epochs: 5}}` must keep the GAN's plain-AdamW optimizer, not fall back to the generic default."""
    cfg = from_dict(Config, {"cgan": {"train": {"epochs": 5}}, "maskgit": {"train": {"microbatch_size": 8}}})
    assert cfg.cgan.train.epochs == 5
    assert cfg.cgan.train.optimizer.type == "adamw" and cfg.cgan.train.optimizer.betas == (0.5, 0.999)
    assert cfg.maskgit.train.optimizer.lr == 1e-4 and cfg.maskgit.train.optimizer.warmup_steps == 1000
    root = __import__("pathlib").Path(__file__).resolve().parent.parent / "configs"
    assert load_config(root / "my_dataset.yaml").cgan.train.optimizer.type == "adamw"


def test_scientific_notation_overrides_become_floats():
    cfg = from_dict(
        Config,
        apply_overrides(
            {},
            [
                "maskgit.train.optimizer.lr=2e-4",
                "vqvae.train.epochs=1e1",
                "generate.keep_schedule=[1e-2, 0.5]",
            ],
        ),
    )
    assert cfg.maskgit.train.optimizer.lr == 2e-4 and isinstance(cfg.maskgit.train.optimizer.lr, float)
    assert cfg.vqvae.train.epochs == 10 and isinstance(cfg.vqvae.train.epochs, int)
    assert cfg.generate.keep_schedule == (0.01, 0.5)
    with pytest.raises(TypeError, match="expects a number"):
        from_dict(Config, {"vqvae": {"train": {"epochs": "many"}}})


def test_none_defaults_coerce_from_the_annotation():
    cfg = from_dict(Config, apply_overrides({}, ["project.seed=1e3", "maskgit.model.grid_h=16"]))
    assert cfg.project.seed == 1000 and isinstance(cfg.project.seed, int)
    assert cfg.maskgit.model.grid_h == 16
    assert from_dict(Config, {"project": {"seed": None}}).project.seed is None
