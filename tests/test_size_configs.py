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


# M and M12 pin micro-batch 256 (the schedule review of 2026-09-25): the trainer's eager VRAM probe picks
# 256 anyway, and 512 compiled was projected to peak ~7.0 GB against a 6.15 GB budget. The flagship
# inherits M's pin. Every other size keeps the recipe's "auto".
MICRO = {"recipe": "auto", "s": "auto", "m": 256, "m12": 256, "l": "auto", "long": 256}
EVAL_EVERY = {"long": 4000}  # PR-5: the flagship's subset evals every 4,000 steps; 2,000 elsewhere


@pytest.mark.parametrize("name", ["recipe", "s", "m", "m12", "l", "long"])
def test_every_recipe_config_carries_recipe_d(repo_root, name):
    cfg = load_config(repo_root / "configs" / f"{name}.toml")
    assert (cfg.batch_size, cfg.roots_per_step, cfg.children_per_step) == (1024, 717, 307)
    assert (cfg.micro_batch, cfg.clip_norm, cfg.rebalance) == (MICRO[name], "auto", True)
    assert (cfg.warmup_steps, cfg.cooldown_frac, cfg.weight_decay, cfg.beta2) == (2000, 0.2, 0.1, 0.95)
    assert (cfg.alpha, cfg.tau, cfg.lambda_v, cfg.ema_max) == (0.5, 0.05, 1.0, 0.9999)
    assert (cfg.eval_every, cfg.vaa_subset, cfg.ckpt_every_minutes) == (
        EVAL_EVERY.get(name, 2000),
        2000,
        30.0,
    )


def test_the_flagship_keeps_every_check_checkpoint_and_metrics_cadence_pr5_leaves_unchanged(repo_root):
    """PR-5 changes only the subset-eval cadence: the checks, the 30-minute checkpoints, metrics every
    50 steps and the stop rules stay as they were."""
    cfg = load_config(repo_root / "configs" / "long.toml")
    assert (cfg.metrics_every, cfg.ckpt_every_minutes, cfg.keep_last, cfg.vaa_checks) == (50, 30.0, 3, True)
    text = (repo_root / "configs" / "long.toml").read_text(encoding="utf-8")
    assert "PR-5" in text


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


def test_m12_s_pin_comment_gives_m12_s_own_bench_numbers_not_m_s(repo_root):
    """M12's pin was copied from M's comment; M12 has its own measured rows (bench.json, 2026-09-24):
    512 compiled peaked at 7.05 GB against the 6.20 GB budget, and 256 runs at 2,290 samples/s."""
    text = (repo_root / "configs" / "m12.toml").read_text(encoding="utf-8")
    pin = text[text.index("micro_batch = 256") : text.index("steps =")]
    assert "7.05 GB" in pin and "6.20 GB" in pin and "2,290" in pin
    assert "projected" not in pin and "6.15 GB" not in pin
