"""`blink sweep ablations|sizes|choose` (plan P5 and P6), tested with fake arm results."""

import json
import tomllib
from pathlib import Path

import pytest

from blink import cli
from blink.train import sweep
from blink.train.supervise import Outcome

REPO = Path(__file__).resolve().parent.parent
ABLATIONS = REPO / "configs" / "ablations"
BASE = """
[model]
d_model = 64
n_layers = 1
n_heads = 2
head_dim = 32

[train]
batch_size = 1000
steps = 100
peak_lr = 0.001
warmup_steps = 10
"""


def test_every_ablation_arm_file_only_overrides_the_recipe():
    plan = tomllib.loads((ABLATIONS / "plan.toml").read_text(encoding="utf-8"))["plan"]
    expected = ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a10", "a11", "a12", "a15"]
    assert sorted(plan["arms"] + plan["held"]) == expected  # PF66: arms without code yet are held
    assert plan["arms"][:3] == ["a01", "a02", "a03"] and plan["held"][-1] == "a15"
    assert sorted(p.stem for p in ABLATIONS.glob("a*.toml")) == expected  # a09, a13, a14 are cut
    for name in expected:
        arm = sweep.load_arm(ABLATIONS / f"{name}.toml")
        assert arm.change and set(arm.overrides) <= {"model", "train"}
        assert "steps" not in arm.overrides.get("train", {})  # the sweep sets steps from the clock
    seeds = [sweep.load_arm(ABLATIONS / f"a0{i}.toml").overrides for i in (1, 2, 3)]
    assert seeds == [{"train": {"seed": i}} for i in (1, 2, 3)]
    assert sweep.load_arm(ABLATIONS / "a05.toml").overrides == {"train": {"alpha": 0.0}}
    assert sweep.load_arm(ABLATIONS / "a11.toml").peak_lr_scale == 0.5
    assert sweep.load_arm(ABLATIONS / "a12.toml").peak_lr_scale == 2.0
    assert sweep.load_arm(ABLATIONS / "a07.toml").judged_on == "games10k_top1"
    assert sweep.load_arm(ABLATIONS / "a15.toml").combine is True


def test_the_a10_arm_turns_the_s_recipe_into_a_valid_muon_config():
    """a10 was held until its code existed (PF66); its overrides now pass the trainer's own check."""
    from blink.model.config import config_from_dict, read_tables

    arm = sweep.load_arm(ABLATIONS / "a10.toml")
    assert arm.overrides == {"train": {"optimizer": "muon", "muon_adjust_lr_fn": "match_rms_adamw"}}
    merged = sweep.merged_config(read_tables(REPO / "configs" / "s.toml"), arm, steps=10_000)
    sweep.validate(merged)
    cfg = config_from_dict({**merged["train"], "model": merged["model"]})
    assert (cfg.optimizer, cfg.muon_adjust_lr_fn, cfg.weight_decay) == ("muon", "match_rms_adamw", 0.1)


def _noise(vaa=(0.500, 0.502, 0.498), top1=(0.300, 0.301, 0.299), **extra):
    results = {f"a0{i + 1}": {"vaa": v, "top1": t} for i, (v, t) in enumerate(zip(vaa, top1, strict=True))}
    for key, values in extra.items():
        for i, value in enumerate(values):
            results[f"a0{i + 1}"][key] = value
    return sweep.noise_floor(results, ("a01", "a02", "a03"))


def _arm(name="a04", judged_on="vaa", **kw):
    return sweep.Arm(name=name, change="test", judged_on=judged_on, **kw)


def test_the_noise_floor_is_the_mean_and_sample_sigma_of_a01_to_a03():
    floor = _noise()
    assert floor["vaa"]["d"] == pytest.approx(0.5) and floor["vaa"]["sigma"] == pytest.approx(0.002)
    assert floor["sigma_ok"] is True  # sigma_VAA 0.2 pt <= 1.0 pt
    assert _noise(vaa=(0.50, 0.53, 0.47))["sigma_ok"] is False


def test_the_adopt_rule_needs_vaa_above_d_plus_2_sigma_and_top1_above_d_minus_2_sigma():
    floor = _noise()  # D_vaa 0.500, sigma 0.002; D_top1 0.300, sigma 0.001
    assert sweep.decide(_arm(), {"vaa": 0.505, "top1": 0.2985}, floor)["adopt"] is True
    assert sweep.decide(_arm(), {"vaa": 0.503, "top1": 0.300}, floor)["adopt"] is False  # under D + 2 sigma
    assert sweep.decide(_arm(), {"vaa": 0.510, "top1": 0.297}, floor)["adopt"] is False  # top-1 fell too far
    missing = sweep.decide(_arm(), {"top1": 0.3}, floor)
    assert missing["adopt"] is False and "not judged" in missing["reason"]


def test_a07_is_judged_on_games10k_top1():
    floor = _noise(games10k_top1=(0.20, 0.21, 0.19))
    arm = _arm("a07", judged_on="games10k_top1")
    assert sweep.decide(arm, {"vaa": 0.40, "top1": 0.30, "games10k_top1": 0.23}, floor)["adopt"] is True
    assert sweep.decide(arm, {"vaa": 0.60, "top1": 0.30, "games10k_top1": 0.20}, floor)["adopt"] is False


def test_a08_must_not_lose_more_than_2_points_of_mate_preservation():
    floor = _noise(mate_preserving=(0.90, 0.91, 0.89))
    arm = _arm("a08", guard="mate_preserving")
    kept = {"vaa": 0.51, "top1": 0.30, "mate_preserving": 0.885}
    lost = {"vaa": 0.51, "top1": 0.30, "mate_preserving": 0.875}
    assert sweep.decide(arm, kept, floor)["adopt"] is True
    assert sweep.decide(arm, lost, floor)["adopt"] is False


def test_a15_combines_the_adopted_arms_and_must_stay_within_1_sigma_of_d():
    arms = {
        "a05": _arm("a05", overrides={"train": {"alpha": 0.0}}),
        "a11": _arm("a11", peak_lr_scale=0.5),
        "a06": _arm("a06", overrides={"model": {"gab": False}}),
    }
    combined = sweep.combine_adopted(
        arms, {"a05": {"adopt": True}, "a11": {"adopt": True}, "a06": {"adopt": False}}
    )
    assert combined.overrides == {"train": {"alpha": 0.0}} and combined.peak_lr_scale == 0.5
    assert combined.combined_from == ("a05", "a11")
    floor = _noise()
    assert sweep.recipe_verdict({"vaa": 0.4985}, floor, combined)["recipe"] == "D + a05 + a11"
    assert sweep.recipe_verdict({"vaa": 0.4970}, floor, combined)["recipe"] == "D"


def test_merged_configs_take_steps_from_the_clock_and_scale_the_learning_rate(tmp_path):
    base = tmp_path / "s.toml"
    base.write_text(BASE, encoding="utf-8")
    arm = _arm("a11", overrides={"train": {"seed": 2}}, peak_lr_scale=0.5)
    steps = sweep.steps_for(hours=1.5, samples_per_s=2000.0, batch_size=1000)
    assert steps == 10800
    merged = sweep.merged_config(tomllib.loads(BASE), arm, steps)
    assert merged["train"]["steps"] == 10800 and merged["train"]["seed"] == 2
    assert merged["train"]["peak_lr"] == pytest.approx(0.0005) and merged["model"]["d_model"] == 64
    out = tmp_path / "merged.toml"
    sweep.write_config(merged, out)
    assert tomllib.loads(out.read_text(encoding="utf-8")) == merged
    with pytest.raises(ValueError, match="warmup"):
        sweep.merged_config(tomllib.loads(BASE), arm, 10)


def _plan(tmp_path: Path, arms: dict[str, str], order: list[str]) -> Path:
    folder = tmp_path / "ablations"
    folder.mkdir()
    (tmp_path / "s.toml").write_text(BASE, encoding="utf-8")
    for name, body in arms.items():
        (folder / f"{name}.toml").write_text(f'[arm]\nchange = "{name}"\n{body}', encoding="utf-8")
    plan = folder / "plan.toml"
    plan.write_text(
        f'[plan]\nrecipe = "{(tmp_path / "s.toml").as_posix()}"\ndata = "{tmp_path.as_posix()}"\n'
        f'hours = 0.01\nsize = "s"\nsigma_arms = ["a01", "a02", "a03"]\narms = {json.dumps(order)}\n'
        'slip_cut = ["a10"]\n',
        encoding="utf-8",
    )
    return plan


ARMS = {
    "a01": "[train]\nseed = 1\n",
    "a02": "[train]\nseed = 2\n",
    "a03": "[train]\nseed = 3\n",
    "a05": "[train]\nalpha = 0.0\n",
    "a10": "[train]\nseed = 9\n",
    "a15": "combine = true\n",
}
ORDER = ["a01", "a02", "a03", "a05", "a10", "a15"]
FAKE_VAA = {
    "abl-a01": 0.500,
    "abl-a02": 0.502,
    "abl-a03": 0.498,
    "abl-a05": 0.51,
    "abl-a10": 0.49,
    "abl-a15": 0.52,
}


class FakeRunner:
    """Writes a final evals.jsonl row per run; can stop the sweep partway like a killed process."""

    def __init__(self, home: Path, stop_after: int | None = None):
        self.home, self.stop_after, self.requests = home, stop_after, []

    def __call__(self, request: sweep.RunRequest) -> Outcome:
        if self.stop_after is not None and len(self.requests) == self.stop_after:
            raise KeyboardInterrupt
        self.requests.append(request)
        config = tomllib.loads(request.config.read_text(encoding="utf-8"))
        run_dir = self.home / "runs" / request.run
        run_dir.mkdir(parents=True, exist_ok=True)
        row = {"step": config["train"]["steps"], "vaa": FAKE_VAA[request.run], "top1": 0.30}
        (run_dir / "evals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return Outcome("finished", "finished", 0)


def test_the_ablation_sweep_runs_arms_in_order_and_resumes_where_it_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, ARMS, ORDER))
    out = tmp_path / "home" / "eval" / "ablations.json"
    first = FakeRunner(tmp_path / "home", stop_after=2)
    with pytest.raises(KeyboardInterrupt):
        sweep.run_ablations(plan, out, rate=1000.0, runner=first, log=lambda _: None)
    state = json.loads(out.read_text(encoding="utf-8"))
    assert [state["arms"][a]["status"] for a in ("a01", "a02", "a03")] == ["finished", "finished", "running"]
    second = FakeRunner(tmp_path / "home")
    report = sweep.run_ablations(plan, out, rate=1000.0, runner=second, log=lambda _: None)
    assert [r.run for r in second.requests] == ["abl-a03", "abl-a05", "abl-a10", "abl-a15"]
    assert second.requests[0].resume is False  # a03 had no checkpoint yet: it starts again
    assert report["arms"]["a01"]["vaa"] == 0.5 and report["noise"]["vaa"]["sigma"] == pytest.approx(0.002)
    assert report["decisions"]["a05"]["adopt"] is True and report["decisions"]["a10"]["adopt"] is False
    a15 = tomllib.loads(second.requests[-1].config.read_text(encoding="utf-8"))
    assert a15["train"]["alpha"] == 0.0  # the combined winners
    assert report["recipe"]["recipe"] == "D + a05"


def test_the_slip_rule_drops_the_cut_arms_and_an_invalid_arm_does_not_stop_the_sweep(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    arms = {**ARMS, "a05": "[train]\nno_such_key = 1\n"}
    plan = sweep.load_plan(_plan(tmp_path, arms, ORDER))
    runner = FakeRunner(tmp_path / "home")
    out = tmp_path / "ablations.json"
    report = sweep.run_ablations(plan, out, rate=1000.0, runner=runner, log=lambda _: None, slip=True)
    assert "abl-a10" not in [r.run for r in runner.requests]
    assert report["arms"]["a10"]["status"] == "not tested: cut by the slip rule"
    assert report["arms"]["a05"]["status"].startswith("invalid: ")
    assert report["arms"]["a15"]["status"] == "not run: no arm was adopted"
    assert report["recipe"]["recipe"] == "D"


def _bench(rates: dict[str, float], p99: dict[str, float], budget: float = 5.5) -> dict:
    throughput = [
        {"size": s, "micro": 256, "compile": "off", "samples_per_s": r, "oom": False, "error": None,
         "peak_reserved_gb": 3.0, "parameters": {"s": 4e6, "m": 21e6, "m12": 31e6, "l": 60e6}[s]}
        for s, r in rates.items()
    ]  # fmt: skip
    play = [
        {"size": s, "rows": 219, "concurrency": c, "p99_ms": v, "p50_ms": v / 2}
        for s, v in p99.items()
        for c in (2, 5)
    ]
    return {"machine": {"vram_budget_gb": budget}, "throughput": throughput, "play": play}


def _choose(rates, vaa, p99=None, sigma=0.005):
    p99 = p99 or dict.fromkeys(rates, 40.0)
    sizes = {s: {"vaa": v} for s, v in vaa.items()}
    return sweep.choose(_bench(rates, p99), sizes, sigma, sweep.ChooseRules())


def test_choose_prefers_the_best_6h_vaa_among_sizes_that_pass_every_constraint():
    rates = {"s": 9000.0, "m": 3000.0, "m12": 2200.0, "l": 1200.0}
    choice = _choose(rates, {"s": 0.50, "m": 0.54, "m12": 0.56, "l": 0.60})
    assert choice["n_star"] == "m12"  # l fails the epoch floor: 1,200 < 1,658 samples/s
    assert choice["sizes"]["l"]["eligible"] is False and "epoch floor" in choice["sizes"]["l"]["reason"]
    assert choice["sizes"]["m"]["epochs"]["96"] == pytest.approx(3000.0 * 96 * 3600 / (401e6 / 0.7))


def test_choose_takes_m_when_the_best_is_within_2_sigma_of_m():
    rates = {"s": 9000.0, "m": 3000.0, "m12": 2200.0}
    assert _choose(rates, {"s": 0.50, "m": 0.54, "m12": 0.549})["n_star"] == "m"
    assert _choose(rates, {"s": 0.50, "m": 0.54, "m12": 0.551})["n_star"] == "m12"


def test_choose_takes_the_largest_passing_size_when_m_fails_the_floor():
    rates = {"s": 9000.0, "m": 1500.0, "m12": 1100.0}
    choice = _choose(rates, {"s": 0.50, "m": 0.54, "m12": 0.56})
    assert choice["n_star"] == "s" and "M fails the epoch floor" in choice["reason"]


def test_choose_drops_a_size_whose_value_mode_p99_is_over_100_ms():
    rates = {"s": 9000.0, "m": 3000.0, "m12": 2200.0}
    choice = _choose(rates, {"s": 0.50, "m": 0.54, "m12": 0.58}, p99={"s": 20.0, "m": 60.0, "m12": 130.0})
    assert choice["n_star"] == "m" and "p99" in choice["sizes"]["m12"]["reason"]


def test_the_epoch_floor_is_one_epoch_of_training_roots_in_96_hours():
    rules = sweep.ChooseRules()
    assert rules.epoch_floor == 1658.0
    assert rules.samples_per_epoch / (rules.t_long_hours * 3600) == pytest.approx(1658.0, abs=1.0)


def test_the_size_sweep_skips_a_conditional_size_below_the_epoch_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    for size in ("s", "m", "l"):
        (tmp_path / f"{size}.toml").write_text(BASE, encoding="utf-8")
    bench = _bench({"s": 9000.0, "m": 3000.0, "l": 1200.0}, {"s": 10.0, "m": 20.0, "l": 30.0})
    runner = SizeRunner(tmp_path / "home")
    setup = sweep.SizeSweep(
        sizes=("s", "m", "l"), conditional=("l",), hours=0.01, recipe=None, data=tmp_path, config_dir=tmp_path
    )
    out = tmp_path / "sweep.json"
    report = sweep.run_sizes(setup, bench, out, runner=runner, log=lambda _: None)
    assert [r.run for r in runner.requests] == ["size-s", "size-m"]
    assert report["sizes"]["l"]["status"].startswith("not run: fails the epoch floor")
    assert report["sizes"]["m"]["vaa"] == 0.54 and report["sizes"]["m"]["samples_per_s"] == 3000.0


class SizeRunner(FakeRunner):
    VAA = {"size-s": 0.50, "size-m": 0.54}

    def __call__(self, request):
        self.requests.append(request)
        run_dir = self.home / "runs" / request.run
        run_dir.mkdir(parents=True, exist_ok=True)
        row = {"step": 1, "vaa": self.VAA[request.run], "top1": 0.3, "value_ce": 3.1}
        (run_dir / "evals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return Outcome("finished", "finished", 0)


def test_sweep_choose_command_reads_bench_and_sweep_json(tmp_path, capsys):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(_bench({"s": 9000.0, "m": 3000.0}, {"s": 10.0, "m": 20.0})), encoding="utf-8")
    sizes = tmp_path / "sweep.json"
    sizes.write_text(json.dumps({"sizes": {"s": {"vaa": 0.50}, "m": {"vaa": 0.52}}}), encoding="utf-8")
    argv = ["sweep", "choose", "--bench", str(bench), "--sweep", str(sizes), "--sigma", "0.005"]
    assert cli.main(argv) == 0
    assert "N* = m" in capsys.readouterr().out
    assert json.loads(sizes.read_text(encoding="utf-8"))["choice"]["n_star"] == "m"


def test_sweep_ablations_dry_run_prints_each_arm_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = _plan(tmp_path, ARMS, ORDER)
    assert cli.main(["sweep", "ablations", "--plan", str(plan), "--rate", "1000", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "abl-a01" in out and "abl-a15" in out and "steps 36" in out


RECIPE = """
[model]
head_dim = 32

[train]
batch_size = 1000
warmup_steps = 10
child_frac = 0.3
"""
SIZE_WITH_BASE = """base = "recipe.toml"

[model]
d_model = 64
n_layers = 1
n_heads = 2

[train]
peak_lr = 0.001
steps = 100
"""


def _recipe_and_size(folder: Path, compile_mode: str = "off") -> Path:
    (folder / "recipe.toml").write_text(RECIPE + f'compile = "{compile_mode}"\n', encoding="utf-8")
    size = folder / "s.toml"
    size.write_text(SIZE_WITH_BASE, encoding="utf-8")
    return size


def test_an_arm_keeps_the_recipe_its_size_config_names_as_base(tmp_path, monkeypatch):
    """PF66: the sweep read configs/s.toml with plain tomllib, so `base = "recipe.toml"` was dropped
    and every arm trained TrainConfig defaults (batch 256, no children, no GAB-lite)."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    size = _recipe_and_size(tmp_path, "inductor")
    arm = _arm("a01", overrides={"train": {"seed": 1}})
    path, info = sweep._prepare("abl-a01", size, arm, 0.01, 1000.0, "ablations")
    written = tomllib.loads(path.read_text(encoding="utf-8"))["train"]
    assert (written["batch_size"], written["child_frac"], written["compile"]) == (1000, 0.3, "inductor")
    assert written["seed"] == 1 and info["steps"] == 36  # 0.01 h x 1,000/s / batch 1,000


def _two_mode_bench(path: Path, size: str = "s", off: float = 1000.0, inductor: float = 2000.0) -> Path:
    ok = {"size": size, "micro": 1024, "oom": False, "error": None, "peak_reserved_gb": 1.0}
    rows = [
        {**ok, "compile": "off", "samples_per_s": off},
        {**ok, "compile": "inductor", "samples_per_s": inductor},
    ]
    path.write_text(json.dumps({"throughput": rows}), encoding="utf-8")
    return path


@pytest.mark.parametrize(("mode", "steps"), [("off", "steps 36"), ("inductor", "steps 72")])
def test_ablation_steps_use_the_bench_row_of_the_recipes_compile_mode(
    tmp_path, monkeypatch, capsys, mode, steps
):
    """PF66: steps came from the fastest row of any compile mode while the recipe trained eager."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = _plan(tmp_path, ARMS, ORDER)
    _recipe_and_size(tmp_path, mode)  # replaces the plan's s.toml with one that names recipe.toml as base
    bench = _two_mode_bench(tmp_path / "bench.json")
    assert cli.main(["sweep", "ablations", "--plan", str(plan), "--bench", str(bench), "--dry-run"]) == 0
    assert steps in capsys.readouterr().out


def test_the_size_sweep_plans_each_size_at_its_own_compile_modes_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    (tmp_path / "recipe.toml").write_text(RECIPE + 'compile = "off"\n', encoding="utf-8")
    (tmp_path / "m.toml").write_text(SIZE_WITH_BASE, encoding="utf-8")
    bench = json.loads(
        _two_mode_bench(tmp_path / "bench.json", "m", 3000.0, 4500.0).read_text(encoding="utf-8")
    )
    setup = sweep.SizeSweep(
        sizes=("m",), conditional=(), hours=0.01, recipe=None, data=tmp_path, config_dir=tmp_path
    )
    runner = SizeRunner(tmp_path / "home")
    report = sweep.run_sizes(setup, bench, tmp_path / "sweep.json", runner=runner, log=lambda _: None)
    assert report["sizes"]["m"]["samples_per_s"] == 3000.0 and report["sizes"]["m"]["compile"] == "off"
    assert runner.requests[0].bench_rate == 3000.0
