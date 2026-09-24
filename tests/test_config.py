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


def test_a_config_inherits_its_base_file_and_overrides_single_keys(tmp_path):
    (tmp_path / "recipe.toml").write_text(
        "[model]\ngab = true\nffn_mult = 2\n\n"
        '[train]\nbatch_size = 1024\npeak_lr = 1e-3\nclip_norm = "auto"\nwarmup_steps = 10\n',
        encoding="utf-8",
    )
    sized = tmp_path / "sizes" / "m.toml"
    sized.parent.mkdir()
    sized.write_text(
        'base = "../recipe.toml"\n\n[model]\nd_model = 512\nn_heads = 16\n\n[train]\npeak_lr = 7e-4\n',
        encoding="utf-8",
    )
    cfg = load_config(sized)
    assert (cfg.model.d_model, cfg.model.n_heads, cfg.model.gab, cfg.model.ffn_mult) == (512, 16, True, 2)
    assert (cfg.batch_size, cfg.peak_lr, cfg.clip_norm) == (1024, 7e-4, "auto")


def test_a_base_that_includes_itself_is_refused(tmp_path):
    (tmp_path / "a.toml").write_text('base = "b.toml"\n', encoding="utf-8")
    (tmp_path / "b.toml").write_text('base = "a.toml"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.toml")


def test_clip_norm_is_a_positive_number_or_auto():
    assert TrainConfig(clip_norm="auto").clip_norm == "auto"
    assert TrainConfig(clip_norm=2.5).clip_norm == 2.5
    for bad in ("sometimes", 0.0, -1.0):
        with pytest.raises(ValueError, match="clip_norm"):
            TrainConfig(clip_norm=bad)


def test_auto_clip_needs_warmup_steps_to_measure():
    with pytest.raises(ValueError, match="warmup"):
        TrainConfig(clip_norm="auto", warmup_steps=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 1024, "micro_batch": 300},
        {"micro_batch": -1},
        {"micro_batch": "half"},
        {"child_frac": 1.0},
        {"child_frac": -0.1},
        {"vaa_subset": 0},
        {"vaa_sigma": -0.01},
        {"vaa_sigma_subset": -0.01},
        {"keep_every_hours": -1.0},
    ],
)
def test_invalid_recipe_values_are_refused(overrides):
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_micro_batches_and_the_root_child_split_follow_the_config():
    cfg = TrainConfig(batch_size=1024, micro_batch=256, child_frac=0.3)
    assert (cfg.roots_per_step, cfg.children_per_step) == (717, 307)
    assert cfg.accumulation(micro=256) == 4
    assert TrainConfig(batch_size=256).roots_per_step == 256


def test_the_epoch_floor_is_1658_samples_per_s_at_the_worst_case_96_hours():
    from blink.model.config import epoch_floor_samples_per_s

    assert round(epoch_floor_samples_per_s(train_roots=401_000_000, child_frac=0.3, hours=96)) == 1658
