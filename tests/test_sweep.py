"""`blink sweep ablations|sizes|choose` (plan P5 and P6), tested with fake arm results."""

import json
import tomllib
from pathlib import Path

import pytest

from blink import cli, heartbeat
from blink.train import nstar, size_sweep, sweep
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
    assert sorted(plan["arms"] + plan["held"]) == expected
    tonight = ["a01", "a02", "a03", "a04", "a05", "a11", "a12"]  # the seven arms PF66 started
    # the late arms run from the GPU gap, a06 only once `pytest -m cuda` passes; a15 combines, so last
    assert plan["arms"] == [*tonight, "a10", "a06", "a07", "a15"]
    assert plan["held"] == ["a08"]  # its guard needs child mate labels the mateset lacks
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


def test_a10_is_planned_at_the_s_muon_bench_row_which_is_s_with_a10s_optimizer():
    """Muon adds optimizer work to every step, so a10's clock budget comes from a bench row of its own
    config, not D's; configs/s-muon.toml is exactly what the sweep trains for a10, bar the step count."""
    from blink.model.config import read_tables

    arm = sweep.load_arm(ABLATIONS / "a10.toml")
    assert arm.bench_size == "s-muon"
    s_muon = read_tables(REPO / "configs" / "s-muon.toml")
    merged = sweep.merged_config(
        read_tables(REPO / "configs" / "s.toml"), arm, steps=s_muon["train"]["steps"]
    )
    assert s_muon == merged
    others = [p for p in ABLATIONS.glob("a*.toml") if p.stem != "a10"]
    assert all(sweep.load_arm(path).bench_size is None for path in others)


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
    """Writes a final evals.jsonl row per run; can stop the sweep partway like a killed process.

    `extra` adds metrics to a run's final row, as the trainer's last check row carries them."""

    def __init__(self, home: Path, stop_after: int | None = None, extra: dict[str, dict] | None = None):
        self.home, self.stop_after, self.requests = home, stop_after, []
        self.extra = extra or {}

    def __call__(self, request: sweep.RunRequest) -> Outcome:
        if self.stop_after is not None and len(self.requests) == self.stop_after:
            raise KeyboardInterrupt
        self.requests.append(request)
        config = tomllib.loads(request.config.read_text(encoding="utf-8"))
        run_dir = self.home / "runs" / request.run
        run_dir.mkdir(parents=True, exist_ok=True)
        row = {"step": config["train"]["steps"], "vaa": FAKE_VAA[request.run], "top1": 0.30}
        row.update(self.extra.get(request.run, {}))
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


def _steps_and_rates(runner: FakeRunner) -> dict[str, tuple[int, float | None]]:
    return {
        r.run: (tomllib.loads(r.config.read_text(encoding="utf-8"))["train"]["steps"], r.bench_rate)
        for r in runner.requests
    }


def test_an_arm_with_its_own_bench_row_is_planned_and_policed_at_that_rate(tmp_path, monkeypatch):
    """a10 at D's rate would get more steps than its hours hold, and the throughput stop rule
    (15% under the bench rate for 10 minutes) could end it before its cooldown."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, ARMS, ORDER))
    runner = FakeRunner(tmp_path / "home")
    out = tmp_path / "ablations.json"
    report = sweep.run_ablations(plan, out, 1000.0, runner, log=lambda _: None, arm_rates={"a10": 500.0})
    planned = _steps_and_rates(runner)
    assert planned["abl-a01"] == planned["abl-a05"] == (36, 1000.0)  # 0.01 h x 1,000/s / batch 1,000
    assert planned["abl-a10"] == (18, 500.0)
    assert (
        report["arms"]["a10"]["samples_per_s"] == 500.0 and report["arms"]["a01"]["samples_per_s"] == 1000.0
    )


def test_a_combined_arm_runs_at_the_slowest_rate_among_the_winners_it_combines(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, ARMS, ORDER))
    runner = FakeRunner(tmp_path / "home")
    rates = {"a05": 800.0, "a10": 500.0}  # a05 is adopted, a10 is not (FAKE_VAA)
    report = sweep.run_ablations(
        plan, tmp_path / "ablations.json", 1000.0, runner, lambda _: None, arm_rates=rates
    )
    assert report["recipe"]["recipe"] == "D + a05"
    assert _steps_and_rates(runner)["abl-a15"] == (28, 800.0)


def test_sweep_ablations_plans_an_arm_at_its_own_bench_row_and_refuses_a_missing_one(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = _plan(tmp_path, {**ARMS, "a10": 'bench_size = "s-muon"\n[train]\nseed = 9\n'}, ORDER)
    ok = {"micro": 1024, "oom": False, "error": None, "peak_reserved_gb": 1.0}
    rows = [
        {**ok, "size": size, "compile": mode, "samples_per_s": rate}
        for size, rate in (("s", 1000.0), ("s-muon", 500.0))
        for mode in ("off", "inductor", "cudagraphs")
    ]
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"throughput": rows}), encoding="utf-8")
    argv = ["sweep", "ablations", "--plan", str(plan), "--bench", str(bench), "--dry-run"]
    assert cli.main(argv) == 0
    lines = {line.split(":")[0]: line for line in capsys.readouterr().out.splitlines()}
    assert "steps 36;" in lines["abl-a01"] and "steps 18;" in lines["abl-a10"]

    bench.write_text(
        json.dumps({"throughput": [row for row in rows if row["size"] == "s"]}), encoding="utf-8"
    )
    assert cli.main(argv) != 0
    assert "a10" in (err := capsys.readouterr().err) and "s-muon" in err


def _check_row(games: float, kept: float, shortest: float = 0.6) -> dict[str, float]:
    return {"games10k_top1": games, "mate_preserving": kept, "shortest_mate": shortest, "check": "100%"}


def test_a07_and_a08_are_judged_on_the_games10k_and_mateset_metrics_of_their_last_check(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    arms = {
        **ARMS,
        "a07": 'judged_on = "games10k_top1"\n[train]\nrebalance = false\n',
        "a08": 'guard = "mate_preserving"\n[train]\nseed = 8\n',
    }
    plan = sweep.load_plan(_plan(tmp_path, arms, ["a01", "a02", "a03", "a07", "a08"]))
    extra = {
        "abl-a01": _check_row(0.40, 0.90),
        "abl-a02": _check_row(0.41, 0.91),
        "abl-a03": _check_row(0.39, 0.89),
        "abl-a07": _check_row(0.43, 0.90),  # VAA 0.49 is below D + 2 sigma, but a07 is judged on games10k
        "abl-a08": _check_row(0.40, 0.86),  # VAA passes; mate preservation fell 4 pt from D
    }
    monkeypatch.setitem(FAKE_VAA, "abl-a07", 0.49)
    monkeypatch.setitem(FAKE_VAA, "abl-a08", 0.51)
    runner = FakeRunner(tmp_path / "home", extra=extra)
    report = sweep.run_ablations(plan, tmp_path / "abl.json", rate=1000.0, runner=runner, log=lambda _: None)
    assert report["noise"]["games10k_top1"]["d"] == pytest.approx(0.40)
    assert report["noise"]["mate_preserving"]["sigma"] == pytest.approx(0.01)
    a07, a08 = report["decisions"]["a07"], report["decisions"]["a08"]
    assert a07["adopt"] is True and a07["metric"] == "games10k_top1" and a07["value"] == 0.43
    assert a08["adopt"] is False and "mate_preserving 0.8600 lost more than 2 pt" in a08["reason"]
    entry = report["arms"]["a08"]
    assert (entry["games10k_top1"], entry["mate_preserving"], entry["shortest_mate"]) == (0.40, 0.86, 0.6)


A07_A08 = {
    **ARMS,
    "a07": 'judged_on = "games10k_top1"\n[train]\nrebalance = false\n',
    "a08": 'guard = "mate_preserving"\n[train]\nseed = 8\n',
}
FROZEN_SEEDS = {"abl-a01": (0.40, 0.90), "abl-a02": (0.41, 0.91), "abl-a03": (0.39, 0.89)}


def _write_posthoc(home: Path, run: str, step: int, games: float, kept: float) -> None:
    """What `blink eval arm-metrics` writes for a finished run (blink.train.posthoc)."""
    metrics = {"games10k_top1": games, "mate_preserving": kept, "shortest_mate": 0.6}
    record = {"step": step, "checkpoint": f"ckpt_{step:09d}.pt", "inputs": {}, "metrics": metrics}
    (home / "runs" / run / "posthoc.json").write_text(json.dumps(record), encoding="utf-8")


def _frozen_commit_sweep(tmp_path: Path, monkeypatch) -> tuple[sweep.AblationPlan, Path, dict]:
    """a01-a03 finished by a commit whose checks had no games10k or mateset; a07 and a08 by one with them."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, A07_A08, ["a01", "a02", "a03", "a07", "a08"]))
    extra = {"abl-a07": _check_row(0.43, 0.90), "abl-a08": _check_row(0.40, 0.86)}
    monkeypatch.setitem(FAKE_VAA, "abl-a07", 0.49)
    monkeypatch.setitem(FAKE_VAA, "abl-a08", 0.51)
    out = tmp_path / "abl.json"
    report = sweep.run_ablations(
        plan, out, rate=1000.0, runner=FakeRunner(tmp_path / "home", extra=extra), log=lambda _: None
    )
    return plan, out, report


def test_arms_that_predate_the_metrics_leave_a07_and_a08_unjudged_until_scored_post_hoc(
    tmp_path, monkeypatch
):
    plan, out, report = _frozen_commit_sweep(tmp_path, monkeypatch)
    assert report["decisions"]["a07"]["reason"] == "not judged: games10k_top1 missing"
    home, step = tmp_path / "home", report["arms"]["a01"]["steps"]
    for run, (games, kept) in FROZEN_SEEDS.items():
        _write_posthoc(home, run, step, games, kept)
    again = sweep.run_ablations(plan, out, rate=1000.0, runner=FakeRunner(home), log=lambda _: None)
    assert again["noise"]["games10k_top1"]["d"] == pytest.approx(0.40)
    a07, a08 = again["decisions"]["a07"], again["decisions"]["a08"]
    assert a07["adopt"] is True and a07["value"] == 0.43
    assert a08["adopt"] is False and "mate_preserving 0.8600 lost more than 2 pt" in a08["reason"]
    a01 = again["arms"]["a01"]
    assert (a01["games10k_top1"], a01["metrics"]["mate_preserving"], a01["metrics"]["vaa"]) == (
        0.40,
        0.90,
        0.5,
    )
    assert json.loads(out.read_text(encoding="utf-8"))["decisions"]["a07"]["adopt"] is True
    evals = home / "runs" / "abl-a01" / "evals.jsonl"
    assert "games10k_top1" not in evals.read_text(encoding="utf-8")  # the history is never rewritten


def test_a_posthoc_record_of_another_step_is_not_merged_into_the_final_row(tmp_path):
    run_dir = tmp_path / "runs" / "abl-a01"
    run_dir.mkdir(parents=True)
    (run_dir / "evals.jsonl").write_text(json.dumps({"step": 100, "vaa": 0.5}) + "\n", encoding="utf-8")
    _write_posthoc(tmp_path, "abl-a01", 90, 0.4, 0.9)  # scored from an older checkpoint
    assert sweep.final_metrics(run_dir) == {"step": 100, "vaa": 0.5}
    _write_posthoc(tmp_path, "abl-a01", 100, 0.4, 0.9)
    merged = sweep.final_metrics(run_dir)
    assert (
        merged["games10k_top1"] == 0.4 and merged["vaa"] == 0.5 and merged["posthoc"] == "ckpt_000000100.pt"
    )


def test_rescore_scores_every_finished_arm_and_judges_a_fresh_read_without_writing_ablations_json(
    tmp_path, monkeypatch
):
    """Scoring takes minutes, and the sweep may record an arm meanwhile: rescore writes only posthoc.json
    files, so it can never put a stale snapshot of ablations.json back over the sweep's."""
    plan, out, report = _frozen_commit_sweep(tmp_path, monkeypatch)
    home, step = tmp_path / "home", report["arms"]["a01"]["steps"]
    state = json.loads(out.read_text(encoding="utf-8"))
    state["arms"]["a08"]["status"] = "running"  # a running arm is not scored
    out.write_text(json.dumps(state), encoding="utf-8")
    finished_a08 = json.dumps({**state, "arms": {**state["arms"], "a08": report["arms"]["a08"]}})
    scored, logs = [], []

    def scorer(run: str) -> None:
        scored.append(run)
        if run == "abl-a07":
            raise ValueError("abl-a07 has no checkpoint at its last step")
        _write_posthoc(home, run, step, *FROZEN_SEEDS[run])
        if run == "abl-a03":  # meanwhile the sweep records a08 finished
            out.write_text(finished_a08, encoding="utf-8")

    rescored = sweep.rescore_ablations(plan, out, scorer, log=logs.append)
    assert scored == ["abl-a01", "abl-a02", "abl-a03", "abl-a07"]
    assert any("a07: not rescored" in line and "last step" in line for line in logs)
    assert rescored["decisions"]["a07"]["adopt"] is True  # its own check row still holds games10k_top1
    assert "mate_preserving 0.8600 lost more than 2 pt" in rescored["decisions"]["a08"]["reason"]
    assert rescored["arms"]["a02"]["games10k_top1"] == 0.41
    assert out.read_text(encoding="utf-8") == finished_a08  # the sweep's record stands


WITH_A07 = ["a01", "a02", "a03", "a05", "a07", "a15"]


def _final_step(home: Path, run: str) -> int:
    rows = (home / "runs" / run / "evals.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(rows[-1])["step"]


def _seed_scorer(home: Path, scored: list[str], extra: dict[str, tuple] | None = None):
    """A stand-in for posthoc.score_run: the frozen-commit seeds' records (and `extra` runs')."""
    records = {**FROZEN_SEEDS, **(extra or {})}

    def scorer(run: str) -> None:
        scored.append(run)
        if run in records:
            _write_posthoc(home, run, _final_step(home, run), *records[run])

    return scorer


def test_the_sweep_scores_the_seed_arms_post_hoc_before_it_combines_the_winners(tmp_path, monkeypatch):
    """a01-a03 predate games10k_top1 and a07 is judged on it: the sweep scores the seeds itself while
    the GPU is idle between arms, so the unattended a15 does not run without a07."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, A07_A08, WITH_A07))
    monkeypatch.setitem(FAKE_VAA, "abl-a07", 0.49)
    home, scored = tmp_path / "home", []
    runner = FakeRunner(home, extra={"abl-a07": _check_row(0.43, 0.90)})
    report = sweep.run_ablations(
        plan, tmp_path / "abl.json", 1000.0, runner, log=lambda _: None, scorer=_seed_scorer(home, scored)
    )
    assert scored == ["abl-a01", "abl-a02", "abl-a03"]  # only the floor was missing the metric
    assert report["decisions"]["a07"]["adopt"] is True
    a15 = tomllib.loads(runner.requests[-1].config.read_text(encoding="utf-8"))["train"]
    assert (a15["alpha"], a15["rebalance"]) == (0.0, False)
    assert report["arms"]["a15"]["combined_from"] == ["a05", "a07"]
    assert report["recipe"]["recipe"] == "D + a05 + a07"


def test_a15_is_held_while_an_arm_holds_a_metric_the_seed_arms_lack(tmp_path, monkeypatch):
    """Scoring the seeds failed, so a07 cannot be judged: a15 waits instead of running without it, and a
    later sweep runs it once `blink sweep rescore` has scored them."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, A07_A08, WITH_A07))
    monkeypatch.setitem(FAKE_VAA, "abl-a07", 0.49)
    home, out, logs = tmp_path / "home", tmp_path / "abl.json", []

    def out_of_memory(run: str) -> None:
        raise RuntimeError("CUDA out of memory")

    runner = FakeRunner(home, extra={"abl-a07": _check_row(0.43, 0.90)})
    report = sweep.run_ablations(plan, out, 1000.0, runner, log=logs.append, scorer=out_of_memory)
    assert "abl-a15" not in [r.run for r in runner.requests]
    a15 = report["arms"]["a15"]
    assert a15["status"] == "held" and "a07 (games10k_top1)" in a15["reason"]
    assert "blink sweep rescore" in a15["reason"]
    assert any("a01: not rescored (CUDA out of memory)" in line for line in logs)
    assert any(line.startswith("a15: held (") for line in logs)
    assert report["recipe"] == {"recipe": "D", "reason": f"a15 is held: {a15['reason']}"}
    for run, (games, kept) in FROZEN_SEEDS.items():  # what `blink sweep rescore` writes
        _write_posthoc(home, run, _final_step(home, run), games, kept)
    again = FakeRunner(home)
    report = sweep.run_ablations(plan, out, 1000.0, again, log=lambda _: None)
    assert [r.run for r in again.requests] == ["abl-a15"]
    assert report["recipe"]["recipe"] == "D + a05 + a07"


def test_a15_held_for_the_floor_later_runs_at_the_own_bench_rate_of_the_winners_it_combines(
    tmp_path, monkeypatch
):
    """The CLI passes run_ablations both a10's own bench rate and the post-hoc scorer a07 needs: a15,
    held while a01-a03 lack games10k_top1, combines a07 and a10 once scored and runs at a10's rate."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, A07_A08, ["a01", "a02", "a03", "a07", "a10", "a15"]))
    monkeypatch.setitem(FAKE_VAA, "abl-a07", 0.49)
    monkeypatch.setitem(FAKE_VAA, "abl-a10", 0.52)
    home, out, rates = tmp_path / "home", tmp_path / "abl.json", {"a10": 500.0}

    def out_of_memory(run: str) -> None:
        raise RuntimeError("CUDA out of memory")

    first = FakeRunner(home, extra={"abl-a07": _check_row(0.43, 0.90)})
    report = sweep.run_ablations(
        plan, out, 1000.0, first, lambda _: None, arm_rates=rates, scorer=out_of_memory
    )
    planned = _steps_and_rates(first)
    assert planned["abl-a07"] == (36, 1000.0) and planned["abl-a10"] == (18, 500.0)
    assert "abl-a15" not in planned and report["arms"]["a15"]["status"] == "held"
    again, scored = FakeRunner(home), []
    report = sweep.run_ablations(
        plan, out, 1000.0, again, lambda _: None, arm_rates=rates, scorer=_seed_scorer(home, scored)
    )
    assert scored == ["abl-a01", "abl-a02", "abl-a03"]
    assert _steps_and_rates(again) == {"abl-a15": (18, 500.0)}  # the slowest winner's rate: a10's
    a15 = tomllib.loads(again.requests[0].config.read_text(encoding="utf-8"))["train"]
    assert (a15["rebalance"], a15["seed"]) == (False, 9)
    entry = report["arms"]["a15"]
    assert entry["combined_from"] == ["a07", "a10"] and entry["samples_per_s"] == 500.0
    assert report["recipe"]["recipe"] == "D + a07 + a10"


def test_a_guard_metric_the_arm_itself_lacks_does_not_hold_a15(tmp_path, monkeypatch):
    """a08's mate_preserving needs child mate labels today's mateset lacks, so a08's own row has none:
    no scoring of the seeds can judge a08, and a15 runs with the arms that were judged."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, A07_A08, ["a01", "a02", "a03", "a05", "a08", "a15"]))
    monkeypatch.setitem(FAKE_VAA, "abl-a08", 0.51)
    scored = []
    runner = FakeRunner(tmp_path / "home", extra={"abl-a08": {"games10k_top1": 0.40, "shortest_mate": 0.6}})
    report = sweep.run_ablations(
        plan, tmp_path / "abl.json", 1000.0, runner, log=lambda _: None, scorer=scored.append
    )
    assert scored == []
    assert report["decisions"]["a08"]["reason"] == "not judged: mate_preserving missing"
    assert report["arms"]["a15"]["combined_from"] == ["a05"] and report["recipe"]["recipe"] == "D + a05"


def test_a_rescore_after_a15_cannot_put_an_arm_it_never_trained_with_into_the_recipe(tmp_path, monkeypatch):
    """a07 trained without games10k, so a15 combined a05 alone. Scoring a07 and the seeds afterwards
    adopts a07, but a15's VAA vouches only for what it trained with: the recipe stays D until a15 reruns."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, A07_A08, WITH_A07))
    monkeypatch.setitem(FAKE_VAA, "abl-a07", 0.49)
    home, out = tmp_path / "home", tmp_path / "abl.json"
    report = sweep.run_ablations(plan, out, 1000.0, FakeRunner(home), log=lambda _: None)
    assert report["arms"]["a15"]["combined_from"] == ["a05"] and report["recipe"]["recipe"] == "D + a05"
    scorer = _seed_scorer(home, [], {"abl-a07": (0.43, 0.90)})
    judged = sweep.rescore_ablations(plan, out, scorer, log=lambda _: None)
    assert judged["decisions"]["a07"]["adopt"] is True
    assert judged["recipe"]["recipe"] == "D"
    stale = "a15 trained with [a05] but the adopted set is now [a05, a07]: rerun a15"
    assert judged["recipe"]["reason"].startswith(stale)
    again = FakeRunner(home)
    recorded = sweep.run_ablations(plan, out, 1000.0, again, log=lambda _: None)
    assert again.requests == [] and recorded["recipe"] == judged["recipe"]


def test_an_interrupted_a15_resumes_with_the_arms_it_started_with(tmp_path, monkeypatch):
    """Its checkpoints hold that combination's training: resuming them under a set a rescore changed
    meanwhile would train one a15 on two configs and record it as the new set."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, A07_A08, WITH_A07))
    monkeypatch.setitem(FAKE_VAA, "abl-a07", 0.49)
    home, out = tmp_path / "home", tmp_path / "abl.json"
    with pytest.raises(KeyboardInterrupt):  # killed as a15 started, with a07 not yet judged
        sweep.run_ablations(plan, out, 1000.0, FakeRunner(home, stop_after=5), log=lambda _: None)
    (home / "runs" / "abl-a15").mkdir()
    (home / "runs" / "abl-a15" / "ckpt_000000010.pt").write_bytes(b"")
    rescore = _seed_scorer(home, [], {"abl-a07": (0.43, 0.90)})
    for run in ("abl-a01", "abl-a02", "abl-a03", "abl-a07"):  # a rescore while a15 was down
        rescore(run)
    logs, second = [], FakeRunner(home)
    report = sweep.run_ablations(plan, out, 1000.0, second, log=logs.append)
    request = second.requests[0]
    assert request.run == "abl-a15" and request.resume is True
    config = tomllib.loads(request.config.read_text(encoding="utf-8"))["train"]
    assert config["alpha"] == 0.0 and "rebalance" not in config
    assert any("a15: resuming with the arms it started with (a05)" in line for line in logs)
    assert report["decisions"]["a07"]["adopt"] is True
    assert report["recipe"]["recipe"] == "D" and "rerun a15" in report["recipe"]["reason"]


def test_sweep_rescore_will_not_score_on_the_gpu_beside_a_live_training_run(tmp_path, monkeypatch, capsys):
    """The training arm sized its micro-batch to the free VRAM when it started; two more models and
    4,096-row chunks beside it could run it out of memory."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = str(_plan(tmp_path, ARMS, ORDER))
    out = tmp_path / "home" / "eval" / "ablations.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"arms": {}}), encoding="utf-8")
    run_dir = tmp_path / "home" / "runs" / "abl-a15"
    run_dir.mkdir(parents=True)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 5})
    assert cli.main(["sweep", "rescore", "--plan", plan]) == 2
    err = capsys.readouterr().err
    assert "abl-a15" in err and "--device cpu" in err
    assert cli.main(["sweep", "rescore", "--plan", plan, "--device", "cpu"]) == 1  # not refused
    assert "nothing to score" in capsys.readouterr().out  # no arm has finished


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
    return nstar.choose(_bench(rates, p99), sizes, sigma, nstar.ChooseRules())


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
    rules = nstar.ChooseRules()
    assert rules.epoch_floor == 1658.0
    assert rules.samples_per_epoch / (rules.t_long_hours * 3600) == pytest.approx(1658.0, abs=1.0)


def test_the_size_sweep_skips_a_conditional_size_below_the_epoch_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    for size in ("s", "m", "l"):
        (tmp_path / f"{size}.toml").write_text(BASE, encoding="utf-8")
    bench = _bench({"s": 9000.0, "m": 3000.0, "l": 1200.0}, {"s": 10.0, "m": 20.0, "l": 30.0})
    runner = SizeRunner(tmp_path / "home")
    setup = size_sweep.SizeSweep(
        sizes=("s", "m", "l"), conditional=("l",), hours=0.01, recipe=None, data=tmp_path, config_dir=tmp_path
    )
    out = tmp_path / "sweep.json"
    report = size_sweep.run_sizes(setup, bench, out, runner=runner, log=lambda _: None)
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
    """The command judges sizes at the compile mode the repo's recipe trains in (PF66)."""
    from blink.model.config import compile_mode, read_tables

    mode = compile_mode(read_tables(sweep.CONFIG_DIR / "recipe.toml"))
    data = _bench({"s": 9000.0, "m": 3000.0}, {"s": 10.0, "m": 20.0})
    data["throughput"] = [{**row, "compile": mode} for row in data["throughput"]]
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(data), encoding="utf-8")
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
    setup = size_sweep.SizeSweep(
        sizes=("m",), conditional=(), hours=0.01, recipe=None, data=tmp_path, config_dir=tmp_path
    )
    runner = SizeRunner(tmp_path / "home")
    report = size_sweep.run_sizes(setup, bench, tmp_path / "sweep.json", runner=runner, log=lambda _: None)
    assert report["sizes"]["m"]["samples_per_s"] == 3000.0 and report["sizes"]["m"]["compile"] == "off"
    assert runner.requests[0].bench_rate == 3000.0


PINNED_RECIPE = """
[model]
head_dim = 32

[train]
batch_size = 1024
micro_batch = "auto"
warmup_steps = 10
compile = "inductor"
"""


def _pinned_bench(m_rates: dict[int, float]) -> dict:
    ok = {"oom": False, "error": None, "peak_reserved_gb": 3.0, "compile": "inductor"}
    rows = [{**ok, "size": "m", "micro": micro, "samples_per_s": rate} for micro, rate in m_rates.items()]
    rows += [{**ok, "size": "s", "micro": 512, "samples_per_s": 9000.0}]
    rows += [{**ok, "size": "s", "micro": 1024, "samples_per_s": 9938.0}]
    return {"machine": {"vram_budget_gb": 6.2}, "throughput": rows, "play": []}


def test_the_size_sweep_plans_and_trains_a_size_at_its_pinned_micro_batch(tmp_path, monkeypatch):
    """M pins micro-batch 256: its run is planned and policed at the 256 row, and the recipe's "auto"
    (arm-style overrides merged over each size) must not undo the pin in the config it trains."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    (tmp_path / "recipe.toml").write_text(PINNED_RECIPE, encoding="utf-8")
    (tmp_path / "s.toml").write_text(SIZE_WITH_BASE, encoding="utf-8")
    pinned = SIZE_WITH_BASE.replace("steps = 100", "steps = 100\nmicro_batch = 256")
    (tmp_path / "m.toml").write_text(pinned, encoding="utf-8")
    setup = size_sweep.SizeSweep(
        sizes=("s", "m"),
        conditional=(),
        hours=0.01,
        recipe=tmp_path / "recipe.toml",
        data=tmp_path,
        config_dir=tmp_path,
    )
    runner = SizeRunner(tmp_path / "home")
    bench = _pinned_bench({256: 2695.0, 512: 2803.0})
    report = size_sweep.run_sizes(setup, bench, tmp_path / "sweep.json", runner=runner, log=lambda _: None)
    assert (report["sizes"]["m"]["micro"], report["sizes"]["m"]["samples_per_s"]) == (256, 2695.0)
    assert report["sizes"]["s"]["micro"] == 1024  # no pin: the fastest row that fits, as before
    trains = {r.run: tomllib.loads(r.config.read_text(encoding="utf-8"))["train"] for r in runner.requests}
    assert (trains["size-m"]["micro_batch"], trains["size-s"]["micro_batch"]) == (256, "auto")
    assert [r.bench_rate for r in runner.requests] == [9938.0, 2695.0]


def test_the_size_sweep_does_not_run_a_pinned_size_without_a_row_at_its_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    (tmp_path / "recipe.toml").write_text(PINNED_RECIPE, encoding="utf-8")
    pinned = SIZE_WITH_BASE.replace("steps = 100", "steps = 100\nmicro_batch = 256")
    (tmp_path / "m.toml").write_text(pinned, encoding="utf-8")
    setup = size_sweep.SizeSweep(
        sizes=("m",), conditional=(), hours=0.01, recipe=None, data=tmp_path, config_dir=tmp_path
    )
    runner = SizeRunner(tmp_path / "home")
    report = size_sweep.run_sizes(
        setup, _pinned_bench({512: 2803.0}), tmp_path / "sweep.json", runner=runner, log=lambda _: None
    )
    assert "micro-batch 256" in report["sizes"]["m"]["status"] and not runner.requests


def test_choose_judges_a_size_that_pins_its_micro_batch_at_that_rows_rate():
    bench = _bench({"s": 9000.0, "m": 1600.0}, {"s": 10.0, "m": 20.0})
    bench["throughput"].append({**bench["throughput"][1], "micro": 512, "samples_per_s": 3000.0})
    sizes = {"s": {"vaa": 0.50}, "m": {"vaa": 0.54}}
    assert nstar.choose(bench, sizes, 0.005, nstar.ChooseRules())["n_star"] == "m"
    pinned = nstar.choose(bench, sizes, 0.005, nstar.ChooseRules(), pins={"m": 256})
    assert pinned["n_star"] == "s" and pinned["sizes"]["m"]["failed"] == "floor"
    assert pinned["sizes"]["m"]["samples_per_s"] == 1600.0


def test_sweep_choose_command_judges_m_at_the_micro_batch_configs_m_toml_pins(tmp_path, capsys):
    from blink.model.config import compile_mode, micro_batch_pin, read_tables

    assert micro_batch_pin(read_tables(sweep.CONFIG_DIR / "m.toml")) == 256
    mode = compile_mode(read_tables(sweep.CONFIG_DIR / "recipe.toml"))
    data = _bench({"s": 9000.0, "m": 1600.0}, {"s": 10.0, "m": 20.0})
    data["throughput"].append({**data["throughput"][1], "micro": 512, "samples_per_s": 3000.0})
    data["throughput"] = [{**row, "compile": mode} for row in data["throughput"]]
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(data), encoding="utf-8")
    sizes = tmp_path / "sweep.json"
    sizes.write_text(json.dumps({"sizes": {"s": {"vaa": 0.50}, "m": {"vaa": 0.54}}}), encoding="utf-8")
    argv = ["sweep", "choose", "--bench", str(bench), "--sweep", str(sizes), "--sigma", "0.005"]
    assert cli.main(argv) == 0
    assert "N* = s" in capsys.readouterr().out  # m's pinned 256 row fails the epoch floor
