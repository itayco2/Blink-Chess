import dataclasses

import pytest

from blink.model.config import ModelConfig, TrainConfig, config_from_dict, config_to_dict, load_config


def _write(tmp_path, text: str):
    path = tmp_path / "c.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_skeleton_config_loads_as_frozen_dataclasses(repo_root):
    cfg = load_config(repo_root / "configs" / "t.toml")
    assert (cfg.batch_size, cfg.steps, cfg.peak_lr, cfg.warmup_steps) == (256, 3000, 1e-3, 200)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.steps = 1


def test_a_config_round_trips_through_a_plain_dict():
    cfg = TrainConfig(model=ModelConfig(d_model=64, n_heads=2), steps=10, warmup_steps=2)
    assert config_from_dict(config_to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    "text",
    [
        "[model]\nwidth = 3\n",
        "[train]\nlearning_rate = 1.0\n",
        "[optimizer]\nlr = 1.0\n",
    ],
)
def test_unknown_keys_and_tables_are_refused(tmp_path, text):
    with pytest.raises(ValueError, match="unknown"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize(
    "overrides",
    [
        {"steps": 0},
        {"batch_size": -1},
        {"warmup_steps": 3000},
        {"cooldown_frac": 0.0},
        {"alpha": 1.5},
        {"tau": 0.0},
        {"ckpt_every_steps": 0},
        {"ckpt_every_steps": -1000},
        {"ckpt_every_minutes": -1.0},
        {"heartbeat_s": -1.0},
    ],
)
def test_invalid_training_values_are_refused(overrides):
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_a_non_positive_model_size_is_refused():
    with pytest.raises(ValueError):
        ModelConfig(n_layers=0)


def test_a_zero_checkpoint_cadence_is_refused_when_the_config_loads(tmp_path):
    path = _write(tmp_path, "[train]\nckpt_every_steps = 0\n")
    with pytest.raises(ValueError, match="train.ckpt_every_steps must be positive, got 0"):
        load_config(path)


def test_zero_minutes_turns_the_wall_clock_checkpoint_off_and_zero_seconds_beats_every_step():
    cfg = TrainConfig(ckpt_every_minutes=0.0, heartbeat_s=0.0)
    assert (cfg.ckpt_every_minutes, cfg.heartbeat_s) == (0.0, 0.0)
