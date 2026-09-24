import pytest

from blink.model.config import epoch_floor_samples_per_s, load_config

SHAPES = {  # name: (d_model, n_layers, n_heads, peak_lr), plan P4 and section 1
    "s": (256, 8, 8, 1e-3),
    "m": (512, 10, 16, 7e-4),
    "m12": (512, 12, 16, 7e-4),
    "l": (640, 12, 20, 5e-4),
}
PARAMETERS = {  # name: (non-GAB, total), measured by blink.model.transformer.parameter_report
    "s": (4_450_432, 5_048_448),
    "m": (21_878_528, 22_550_272),
    "m12": (26_075_008, 26_746_752),
    "l": (40_703_616, 41_412_224),
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_size_configs_have_the_planned_shapes_and_learning_rates(repo_root, name):
    cfg = load_config(repo_root / "configs" / f"{name}.toml")
    d_model, n_layers, n_heads, peak_lr = SHAPES[name]
    assert (cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads, cfg.peak_lr) == (
        d_model,
        n_layers,
        n_heads,
        peak_lr,
    )
    assert cfg.model.head_dim == 32 and cfg.model.ffn_mult == 2 and cfg.model.gab


@pytest.mark.parametrize("name", ["recipe", "s", "m", "m12", "l", "long"])
def test_every_recipe_config_carries_recipe_d(repo_root, name):
    cfg = load_config(repo_root / "configs" / f"{name}.toml")
    assert (cfg.batch_size, cfg.roots_per_step, cfg.children_per_step) == (1024, 717, 307)
    assert (cfg.micro_batch, cfg.clip_norm, cfg.rebalance) == ("auto", "auto", True)
    assert (cfg.warmup_steps, cfg.cooldown_frac, cfg.weight_decay, cfg.beta2) == (2000, 0.2, 0.1, 0.95)
    assert (cfg.alpha, cfg.tau, cfg.lambda_v, cfg.ema_max) == (0.5, 0.05, 1.0, 0.9999)
    assert (cfg.eval_every, cfg.vaa_subset, cfg.ckpt_every_minutes) == (2000, 2000, 30.0)


def test_s10m_is_s_on_roots_only_for_about_one_gpu_hour(repo_root):
    s10m = load_config(repo_root / "configs" / "s10m.toml")
    s = load_config(repo_root / "configs" / "s.toml")
    assert s10m.model == s.model and s10m.peak_lr == s.peak_lr
    assert s10m.child_frac == 0.0 and s10m.children_per_step == 0
    assert s10m.steps < s.steps / 5  # 1 GPU-h against the 6 GPU-h sweep run


def test_the_long_config_is_the_flagship_template(repo_root):
    cfg = load_config(repo_root / "configs" / "long.toml")
    assert cfg.film and cfg.vaa_checks and cfg.keep_every_hours == 12.0
    assert cfg.steps > 10 * load_config(repo_root / "configs" / "m.toml").steps


def test_only_s_passes_the_epoch_floor_at_the_p4_measured_eager_rates():
    """P4 measured 7,003 / 1,567 / 1,121 / 984 samples/s for S / M / M12 / L (eager, shared GPU)."""
    floor = epoch_floor_samples_per_s(train_roots=401_000_000, child_frac=0.3, hours=96)
    rates = {"s": 7003, "m": 1567, "m12": 1121, "l": 984}
    assert [name for name, rate in rates.items() if rate >= floor] == ["s"]


@pytest.mark.torch
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_parameter_counts_with_and_without_gab(repo_root, name):
    pytest.importorskip("torch")
    import dataclasses

    from blink.model.transformer import BlinkNet, parameter_report

    cfg = load_config(repo_root / "configs" / f"{name}.toml").model
    report = parameter_report(BlinkNet(cfg))
    plain = parameter_report(BlinkNet(dataclasses.replace(cfg, gab=False)))
    assert (report["non_gab"], report["total"]) == PARAMETERS[name]
    assert plain["total"] == report["non_gab"]
