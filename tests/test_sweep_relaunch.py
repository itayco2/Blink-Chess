"""Relaunching `blink sweep ablations` partway through P5, and scoring arms beside it (plan P5).

The sweep is stopped and relaunched between its arms: to add a06 and a10 after the seven arms PF66
started, after the slip rule, after `blink sweep rescore`. A relaunch must plan every arm on the sample
budget its judges trained on, must not spend GPU time on an arm it will not run, and scoring on the
GPU must never share it with a trainer that is starting up.
"""

import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest
from test_sweep import ARMS, FAKE_VAA, FROZEN_SEEDS, ORDER, FakeRunner, _plan, _write_posthoc

from blink import cli, heartbeat
from blink.ops import launch
from blink.train import posthoc, sweep

REPO = Path(__file__).resolve().parent.parent
QUIET = {"log": lambda _: None}


def _ok_row(size: str, rate: float, mode: str = "off") -> dict:
    return {
        "size": size,
        "micro": 1024,
        "compile": mode,
        "samples_per_s": rate,
        "oom": False,
        "error": None,
        "peak_reserved_gb": 1.0,
    }


def _bench(path: Path, rates: dict[str, float]) -> Path:
    path.write_text(json.dumps({"throughput": [_ok_row(s, r) for s, r in rates.items()]}), encoding="utf-8")
    return path


def _muon_plan(tmp_path: Path) -> Path:
    return _plan(tmp_path, {**ARMS, "a10": 'bench_size = "s-muon"\n[train]\nseed = 9\n'}, ORDER)


def _steps(request: sweep.RunRequest) -> int:
    return tomllib.loads(request.config.read_text(encoding="utf-8"))["train"]["steps"]


# ---------------------------------------------------------------- the slip rule and own bench rows


def test_the_slip_rule_needs_no_bench_row_for_an_arm_it_cuts(tmp_path, monkeypatch, capsys):
    """--slip drops a10, so a10's missing s-muon row must not refuse the sweep: the rule exists to
    save GPU time, and benching the arm it drops would spend it."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = _muon_plan(tmp_path)
    bench = _bench(tmp_path / "bench.json", {"s": 1000.0})
    argv = ["sweep", "ablations", "--plan", str(plan), "--bench", str(bench), "--dry-run"]
    assert cli.main([*argv, "--slip"]) == 0
    lines = {line.split(":")[0]: line for line in capsys.readouterr().out.splitlines()}
    assert "cut by the slip rule" in lines["abl-a10"] and "steps" not in lines["abl-a10"]
    assert "steps 36;" in lines["abl-a01"]
    assert cli.main(argv) == 2  # without --slip a10 runs, and its row is still required
    assert "s-muon" in capsys.readouterr().err


def test_a_refused_bench_row_names_the_one_command_that_measures_it(tmp_path, monkeypatch, capsys):
    """`blink bench throughput` alone measures s, m, m12 and l in three compile modes and never s-muon."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    bench = _bench(tmp_path / "bench.json", {"s": 1000.0})
    argv = ["sweep", "ablations", "--plan", str(_muon_plan(tmp_path)), "--bench", str(bench), "--dry-run"]
    assert cli.main(argv) == 2
    assert "blink bench throughput --sizes s-muon --micro 1024 --compile off" in capsys.readouterr().err


def test_an_arm_that_already_finished_needs_no_bench_row(tmp_path, monkeypatch, capsys):
    """a10 finished at its s-muon rate, which ablations.json records: a relaunch (for a15, say) reads
    that rate for a15's plan and does not refuse when bench.json no longer holds the row."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan_path, home = _muon_plan(tmp_path), tmp_path / "home"
    plan, out = sweep.load_plan(plan_path), home / "eval" / "ablations.json"
    monkeypatch.setitem(FAKE_VAA, "abl-a10", 0.52)
    first = FakeRunner(home, stop_after=5)  # killed as a15 starts
    with pytest.raises(KeyboardInterrupt):
        sweep.run_ablations(plan, out, 1000.0, first, arm_rates={"a10": 500.0}, **QUIET)
    bench = _bench(tmp_path / "bench.json", {"s": 1000.0})
    argv = ["sweep", "ablations", "--plan", str(plan_path), "--bench", str(bench), "--dry-run"]
    assert cli.main(argv) == 0
    assert "abl-a10: finished" in capsys.readouterr().out
    again = FakeRunner(home)
    report = sweep.run_ablations(plan, out, 1000.0, again, **QUIET)  # the CLI passes no rate for a10
    assert [(r.run, _steps(r), r.bench_rate) for r in again.requests] == [("abl-a15", 18, 500.0)]
    assert report["arms"]["a15"]["combined_from"] == ["a05", "a10"]


# ---------------------------------------------------------------- one sample budget for the whole sweep


def test_a_relaunch_plans_later_arms_at_the_rate_the_seed_arms_trained_at(tmp_path, monkeypatch):
    """D's bench row re-measured partway through (10% faster now) must not give a05 more samples than
    the a01-a03 floor it is judged against: the first launch pins the plan rate in ablations.json."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan, home = sweep.load_plan(_plan(tmp_path, ARMS, ORDER)), tmp_path / "home"
    out = home / "eval" / "ablations.json"
    with pytest.raises(KeyboardInterrupt):
        sweep.run_ablations(plan, out, 1000.0, FakeRunner(home, stop_after=3), **QUIET)
    assert json.loads(out.read_text(encoding="utf-8"))["rate"] == 1000.0
    logs, again = [], FakeRunner(home)
    report = sweep.run_ablations(plan, out, 1100.0, again, log=logs.append)
    assert {r.run: (_steps(r), r.bench_rate) for r in again.requests}["abl-a05"] == (36, 1000.0)
    assert report["arms"]["a05"]["steps"] == report["arms"]["a01"]["steps"] == 36
    assert any("1,000 samples/s" in line and "1,100" in line for line in logs)


def test_a_sweep_recorded_before_the_pin_takes_its_rate_from_the_seed_arms_steps(tmp_path, monkeypatch):
    """The frozen PF66 sweep recorded each arm's steps but no rate: a06 relaunched by newer code after
    a re-bench still trains the seeds' step count."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan, home = sweep.load_plan(_plan(tmp_path, ARMS, ["a01", "a02", "a03", "a05"])), tmp_path / "home"
    out = home / "eval" / "ablations.json"
    out.parent.mkdir(parents=True)
    seeds = {
        f"a0{i}": {"name": f"a0{i}", "run": f"abl-a0{i}", "status": "finished", "steps": 36, "metrics": {}}
        for i in (1, 2, 3)
    }
    out.write_text(json.dumps({"arms": seeds}), encoding="utf-8")
    runner = FakeRunner(home)
    sweep.run_ablations(plan, out, 1100.0, runner, **QUIET)
    [request] = runner.requests
    assert request.run == "abl-a05" and _steps(request) == 36
    assert request.bench_rate == pytest.approx(1000.0, rel=1e-3)
    assert json.loads(out.read_text(encoding="utf-8"))["rate"] == pytest.approx(1000.0, rel=1e-3)


def test_an_arm_stopped_partway_resumes_with_the_steps_it_started_with(tmp_path, monkeypatch):
    """Its checkpoints hold a WSD schedule of that many steps; a new count would move its cooldown."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan, home = sweep.load_plan(_muon_plan(tmp_path)), tmp_path / "home"
    out = home / "eval" / "ablations.json"
    with pytest.raises(KeyboardInterrupt):  # killed as a10 trains
        sweep.run_ablations(
            plan, out, 1000.0, FakeRunner(home, stop_after=4), arm_rates={"a10": 500.0}, **QUIET
        )
    (home / "runs" / "abl-a10").mkdir(parents=True)
    (home / "runs" / "abl-a10" / "ckpt_000000009.pt").write_bytes(b"")
    again = FakeRunner(home)
    sweep.run_ablations(plan, out, 1000.0, again, arm_rates={"a10": 600.0}, **QUIET)  # s-muon re-benched
    request = again.requests[0]
    assert (request.run, request.resume, _steps(request), request.bench_rate) == ("abl-a10", True, 18, 500.0)


def test_a_frozen_arm_stopped_partway_keeps_its_recorded_steps(tmp_path, monkeypatch):
    """An entry the frozen sweep wrote has steps but no samples_per_s."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan, home = sweep.load_plan(_plan(tmp_path, ARMS, ["a01", "a02", "a03", "a05"])), tmp_path / "home"
    out = home / "eval" / "ablations.json"
    with pytest.raises(KeyboardInterrupt):
        sweep.run_ablations(plan, out, 1000.0, FakeRunner(home, stop_after=3), **QUIET)
    state = json.loads(out.read_text(encoding="utf-8"))
    del state["rate"]
    for entry in state["arms"].values():
        entry.pop("samples_per_s", None)
    state["arms"]["a05"]["steps"] = 40  # planned at a rate the seeds' steps no longer tell
    out.write_text(json.dumps(state), encoding="utf-8")
    (home / "runs" / "abl-a05").mkdir(parents=True)
    (home / "runs" / "abl-a05" / "ckpt_000000009.pt").write_bytes(b"")
    again = FakeRunner(home)
    sweep.run_ablations(plan, out, 1200.0, again, **QUIET)
    assert [(r.run, r.resume, _steps(r)) for r in again.requests] == [("abl-a05", True, 40)]


def test_the_bench_comment_measures_only_the_new_row():
    """Re-measuring s beside s-muon would replace D's row partway through the sweep."""
    text = (REPO / "configs" / "s-muon.toml").read_text(encoding="utf-8")
    assert "blink bench throughput --sizes s-muon --micro 1024" in text and "s,s-muon" not in text


# ---------------------------------------------------------------- a15 once an arm is adopted later


def test_a15_that_found_no_winner_runs_once_a_rescore_adopts_an_arm(tmp_path, monkeypatch):
    """a08's row lacked mate_preserving, so nothing was adopted and a15 did not run. Scoring it later
    adopts a08: the recipe must say a15 is due, and the next sweep must run it."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    arms = {**ARMS, "a08": 'guard = "mate_preserving"\n[train]\nseed = 8\n'}
    plan = sweep.load_plan(_plan(tmp_path, arms, ["a01", "a02", "a03", "a08", "a15"]))
    monkeypatch.setitem(FAKE_VAA, "abl-a08", 0.51)
    home, out = tmp_path / "home", tmp_path / "abl.json"
    runner = FakeRunner(home, extra={"abl-a08": {"games10k_top1": 0.40, "shortest_mate": 0.6}})
    report = sweep.run_ablations(plan, out, 1000.0, runner, **QUIET)
    assert report["arms"]["a15"]["status"] == "not run: no arm was adopted"
    step = report["arms"]["a01"]["steps"]

    def scorer(run: str) -> None:
        games, kept = {**FROZEN_SEEDS, "abl-a08": (0.40, 0.90)}[run]
        _write_posthoc(home, run, step, games, kept)

    judged = sweep.rescore_ablations(plan, out, scorer, **QUIET)
    assert judged["decisions"]["a08"]["adopt"] is True
    assert judged["recipe"]["recipe"] == "D"
    assert "a15" in judged["recipe"]["reason"] and "blink sweep ablations" in judged["recipe"]["reason"]
    again = FakeRunner(home)
    report = sweep.run_ablations(plan, out, 1000.0, again, **QUIET)
    assert [r.run for r in again.requests] == ["abl-a15"]
    assert report["arms"]["a15"]["combined_from"] == ["a08"] and report["recipe"]["recipe"] == "D + a08"


def test_a15_with_still_no_winner_stays_not_run(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, ARMS, ["a01", "a02", "a03", "a10", "a15"]))
    home, out = tmp_path / "home", tmp_path / "abl.json"
    sweep.run_ablations(plan, out, 1000.0, FakeRunner(home), **QUIET)
    again = FakeRunner(home)
    report = sweep.run_ablations(plan, out, 1000.0, again, **QUIET)
    assert again.requests == [] and report["arms"]["a15"]["status"] == "not run: no arm was adopted"
    assert report["recipe"] == {"recipe": "D", "reason": "no arm was adopted"}


# ---------------------------------------------------------------- post-hoc scoring and the GPU


def test_the_sweep_scores_the_seed_arms_in_a_child_process(tmp_path, monkeypatch):
    """Scored in the sweep's own process, the models' CUDA context and cached blocks would stay held
    while a15's trainer sizes its micro-batch from the free VRAM."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    calls, codes = [], iter([0, 2])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, check: calls.append(argv) or subprocess.CompletedProcess(argv, next(codes)),
    )
    monkeypatch.setattr(posthoc, "score_run", lambda *a, **k: pytest.fail("scored in the sweep's process"))
    seen = {}

    def fake_sweep(plan, out, rate, runner, log, slip, arm_rates, scorer):
        scorer("abl-a01")
        with pytest.raises(RuntimeError, match="exited 2"):
            scorer("abl-a02")
        seen["ok"] = True
        return {"recipe": {"recipe": "D"}, "decisions": {}, "noise": None}

    monkeypatch.setattr(sweep, "run_ablations", fake_sweep)
    assert (
        cli.main(["sweep", "ablations", "--plan", str(_plan(tmp_path, ARMS, ORDER)), "--rate", "1000"]) == 0
    )
    assert seen["ok"]
    argv = calls[0]
    assert argv[:5] == [sys.executable, "-m", "blink.cli", "eval", "arm-metrics"]
    assert argv[argv.index("--run") + 1] == "abl-a01" and argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--data") + 1] == str(tmp_path)  # the plan's pack, whose mateset.npz is scored


def _trainer(pid: int, *args: str) -> dict:
    cmdline = [r"C:\v\Scripts\python.exe", r"C:\v\Scripts\blink.exe", *args]
    return {"pid": pid, "cmdline": cmdline, "create_time": time.time() - 45}


def _processes(monkeypatch, *procs: dict) -> None:
    monkeypatch.setattr(launch, "blink_processes", lambda home=None: launch.ps_rows(procs, home or Path(".")))


def _finished_state(tmp_path: Path) -> str:
    plan = str(_plan(tmp_path, ARMS, ORDER))
    out = tmp_path / "home" / "eval" / "ablations.json"
    out.parent.mkdir(parents=True)
    entry = {"run": "abl-a01", "status": "finished", "metrics": {}}
    out.write_text(json.dumps({"arms": {"a01": {"name": "a01", **entry}}}), encoding="utf-8")
    return plan


def test_gpu_scoring_waits_for_a_trainer_that_has_not_beaten_yet(tmp_path, monkeypatch, capsys):
    """A trainer sizing its micro-batch from the free VRAM or compiling its first step writes no beat
    for minutes; its last beat is 45 s old, yet scoring beside it would shrink its micro-batch."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = _finished_state(tmp_path)
    run_dir = tmp_path / "home" / "runs" / "abl-a10"
    run_dir.mkdir(parents=True)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 0}, now=time.time() - 45)
    _processes(monkeypatch, _trainer(4242, "train", "--config", "c.toml", "--run", "abl-a10"))
    assert cli.main(["sweep", "rescore", "--plan", plan]) == 2
    err = capsys.readouterr().err
    assert "pid 4242 (blink train --run abl-a10)" in err and "--device cpu" in err
    assert cli.main(["eval", "arm-metrics", "--run", "abl-a01"]) == 2
    assert "pid 4242 (blink train" in capsys.readouterr().err
    _processes(monkeypatch, _trainer(77, "sweep", "ablations", "--plan", "p.toml"))  # between two arms
    assert cli.main(["eval", "arm-metrics", "--run", "abl-a01"]) == 2
    assert "pid 77 (blink sweep ablations)" in capsys.readouterr().err


def test_only_blink_jobs_that_train_hold_the_gpu():
    rows = launch.ps_rows(
        [
            _trainer(1, "supervise", "--run", "long", "--", "train", "--config", "c.toml"),
            _trainer(2, "data", "verify", "--pack", r"D:\blink\data\v1"),
            _trainer(3, "sweep", "ablations", "--dry-run"),
            {
                "pid": 4,
                "cmdline": [sys.executable, "-m", "blink.cli", "bench", "throughput"],
                "create_time": 0,
            },
            _trainer(5, "sweep", "rescore", "--device", "cuda"),
        ],
        Path("."),
    )
    assert [
        (row["pid"], launch.gpu_command(row["args"])) for row in rows if launch.gpu_command(row["args"])
    ] == [
        (1, "supervise"),
        (4, "bench throughput"),
    ]


# ---------------------------------------------------------------- what `blink sweep rescore` reports


def test_rescore_exits_non_zero_unless_every_finished_arm_was_scored(tmp_path, monkeypatch, capsys):
    """A scripted caller (a07 and a15 leave `held` only once the seeds are scored) must not read a
    total failure as success."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = _finished_state(tmp_path)
    assert cli.main(["sweep", "rescore", "--plan", plan, "--device", "cpu"]) == 1
    out = capsys.readouterr().out
    assert "a01: not rescored" in out and "posthoc.json files written" not in out
    assert "no arm was scored" in out


def test_rescore_with_no_finished_arm_says_so_and_exits_non_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = str(_plan(tmp_path, ARMS, ORDER))
    out = tmp_path / "home" / "eval" / "ablations.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"arms": {"a01": {"name": "a01", "status": "running"}}}), encoding="utf-8")
    assert cli.main(["sweep", "rescore", "--plan", plan, "--device", "cpu"]) == 1
    assert "no arm has finished" in capsys.readouterr().out


def test_rescore_lists_what_it_scored_and_what_it_could_not(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    plan = sweep.load_plan(_plan(tmp_path, ARMS, ["a01", "a02", "a03"]))
    out, home = tmp_path / "abl.json", tmp_path / "home"
    sweep.run_ablations(plan, out, 1000.0, FakeRunner(home), **QUIET)

    def scorer(run: str) -> None:
        if run == "abl-a02":
            raise ValueError("abl-a02 has no checkpoint")

    judged = sweep.rescore_ablations(plan, out, scorer, **QUIET)
    assert judged["scored"] == ["a01", "a03"] and judged["not_scored"] == {"a02": "abl-a02 has no checkpoint"}
