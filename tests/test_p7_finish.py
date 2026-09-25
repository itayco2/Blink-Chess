"""tools/p7_finish.py: PR-6's finish, the flagship's final 20% 1-sqrt cooldown branched from step c.

When Itay says the level is good enough, the run is re-planned so its last 20% cools down from the
current step: a branch runs/long-final of exactly ceil(c/4) cooldown steps from runs/long's checkpoint c.
runs/long is never touched, its own supervisor is ended only while it holds no trainer (paused by the
user, or already gone), and nothing runs while runs/long trains or while any other Blink run could take
the GPU beside the branch. A fake machine answers here.
"""

import json
import math
import sys
from pathlib import Path

import pytest
from test_p7_v2_driver import REPO

# isort: split
import p7_finish  # tools/: on sys.path once test_p7_v2_driver is imported

C = 123_457  # a pause step that is not a multiple of 4
K = math.ceil(C / 4)  # 30,865
PYTHON = ["C:/venv/python.exe", "-m", "blink.cli"]
FLAGSHIP = [
    "train",
    "--config",
    "configs/long.toml",
    "--run",
    "long",
    "--data",
    "D:/blink/data/v1",
    "--resume",
]
SUPERVISOR = [*PYTHON, "supervise", "--bench-size", "m", "--", *FLAGSHIP]
TRAINER = [*PYTHON, *FLAGSHIP]
SIZE_M = [*PYTHON, "supervise", "--", "train", "--run", "long", "--from-step", "47301", "--preview-steps",
          "11825", "--preview-name", "size-m"]  # fmt: skip


class FakeMachine:
    """`branch` is what the launched branch does once the flag is down: "trains" (its trainer writes
    config.json), "silent" (its supervisor runs, its trainer writes nothing) or "refused" (blink train
    exits 2 and the supervisor records the crash). While the flag is up its supervisor waits, paused."""

    def __init__(self, s, processes=(), launch_code=0, branch="trains") -> None:
        self.s, self.procs, self.launch_code = s, [dict(p) for p in processes], launch_code
        self.killed, self.runs, self.now, self.branch = [], [], 1_800_000_000.0, branch

    def processes(self):
        return [p for p in self.procs if p["pid"] not in self.killed]

    def kill_tree(self, pid):
        """The process and every descendant, as psutil ends them."""
        doomed = {pid}
        while grown := {p["pid"] for p in self.procs if p.get("ppid") in doomed} - doomed:
            doomed |= grown
        self.killed.extend(sorted(doomed - set(self.killed)))

    def run(self, step, args):
        self.runs.append((step, list(args)))
        branch = self.s.runs / "long-final"
        branch.mkdir(parents=True, exist_ok=True)
        record = {"state": "running", "status": "running", "started": self.now, "events": []}
        if (self.s.home / "PAUSE").exists():
            record |= {"state": "paused: user", "status": "paused: user"}
        elif self.branch == "trains":
            config = {"run": "long-final", "created": self.now + 40, "config": {}}
            (branch / "config.json").write_text(json.dumps(config), encoding="utf-8")
        elif self.branch == "refused":
            record["events"] = [{"time": self.now + 20, "event": "crash", "code": 2, "restart": 1}]
        (branch / "supervisor.json").write_text(json.dumps(record), encoding="utf-8")
        return self.launch_code, "launched p7-long-final: pid 77 (cmd.exe), python [78]\n"

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _home(tmp_path: Path, steps=(100_000, C), supervisor_state="paused: user", flag=True) -> Path:
    home = tmp_path / "home"
    run = home / "runs" / "long"
    run.mkdir(parents=True)
    for step in steps:
        (run / f"ckpt_{step:09d}.pt").write_bytes(b"")
    rows = [
        {"step": s, "time": 1000.0 + 0.36 * s, "phase": "train", "session": 1.0}
        for s in range(1000, C + 1, 1000)
    ]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    beat = {"state": supervisor_state, "step": steps[-1], "time": 1_799_999_990.0}
    (run / "heartbeat.json").write_text(json.dumps(beat), encoding="utf-8")
    record = {"state": supervisor_state, "status": supervisor_state, "started": 1_700_000_000.0}
    (run / "supervisor.json").write_text(json.dumps(record), encoding="utf-8")
    if flag:
        (home / "PAUSE").write_text("paused", encoding="utf-8")
    return home


def _settings(tmp_path: Path, **overrides) -> "p7_finish.Settings":
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    for name in ("long.toml", "m.toml", "recipe.toml"):
        (repo / "configs" / name).write_bytes((REPO / "configs" / name).read_bytes())
    home = tmp_path / "home" if (tmp_path / "home").is_dir() else _home(tmp_path)
    base = {"repo": repo, "python": Path("python.exe"), "home": home, "data": home / "data" / "v1"}
    return p7_finish.Settings(**{**base, **overrides})


def _record(s) -> dict:
    return json.loads((s.home / "eval" / "p7_finish.json").read_text(encoding="utf-8"))


def _supervisor(pid=41, ppid=1, cmdline=SUPERVISOR):
    return {"pid": pid, "ppid": ppid, "cmdline": list(cmdline)}


def test_the_branch_s_lr_schedule_is_the_recipe_s_1_sqrt_cooldown_from_exactly_c():
    """ceil(c/4) cooldown steps make the cooldown the final 20% of c + ceil(c/4), starting at c itself."""
    from blink.model.config import load_config
    from blink.train.preview import preview_config
    from blink.train.schedule import cooldown_start, wsd_lr

    def lr(step: int, cfg) -> float:
        return wsd_lr(step, cfg.peak_lr, cfg.warmup_steps, cfg.steps, cfg.cooldown_frac)

    flagship = load_config(REPO / "configs" / "long.toml")
    for c in (100_000, C, 331_758, 500_001):  # inside the flagship's stable phase (its cooldown: 528,862)
        k = p7_finish.cooldown_steps(c)
        branch = preview_config(flagship, c, k)
        total, peak = branch.steps, branch.peak_lr
        assert total == c + k and k == -(-c // 4) and abs(k - 0.2 * total) <= 1
        assert cooldown_start(total, branch.cooldown_frac) == c
        assert lr(c - 1, branch) == peak == lr(c - 1, flagship)  # the flagship's stable LR up to c
        assert lr(c, branch) == pytest.approx(peak * (1 - math.sqrt(1 / k)), rel=1e-12)
        half = c + k // 2
        assert lr(half, branch) == pytest.approx(peak * (1 - math.sqrt((k // 2 + 1) / k)), rel=1e-12)
        assert lr(total - 1, branch) == 0.0


def test_finish_branches_the_final_cooldown_from_the_step_the_pause_left(tmp_path, capsys):
    s = _settings(tmp_path)
    machine = FakeMachine(s, [_supervisor(41), _supervisor(42, ppid=41)])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_DONE
    assert machine.killed == [41, 42]  # the venv launcher's tree: its python child goes with it
    step, args = machine.runs[0]
    # no --config: the branch trains with the config runs/long's checkpoint holds
    branch = ["train", "--run", "long", "--data", str(s.data), "--from-step", str(C), "--preview-steps",
              str(K), "--preview-name", "long-final"]  # fmt: skip
    launch = ["ops", "launch", "--name", "p7-long-final", "--", "supervise", "--bench-size", "m", "--"]
    assert args == [*launch, *branch]
    record = _record(s)
    assert (record["at_step"], record["cooldown_steps"], record["total_steps"]) == (C, K, C + K)
    assert record["stopped_supervisor"] == [41] and record["command"].startswith(
        "python -m blink.cli ops launch"
    )
    assert record["training_hours"] == pytest.approx((C // 1000 - 1) * 360.0 / 3600)
    assert record["run"] == "long" and record["branch"] == "long-final" and "time" in record
    assert "Resume Blink" in capsys.readouterr().out  # the branch waits for the flag like any run


def test_with_the_flag_down_the_finish_is_done_once_long_final_s_trainer_has_written_its_config(tmp_path):
    _home(tmp_path, supervisor_state="stopped", flag=False)
    s = _settings(tmp_path)
    assert p7_finish.finish(s, FakeMachine(s, [], branch="trains")) == p7_finish.EXIT_DONE


def test_a_branch_whose_trainer_never_starts_is_a_failed_finish_not_a_launched_one(tmp_path, capsys):
    """The supervisor's first record says running before its child parsed its arguments: with the flag
    down only the trainer's config.json shows the branch was accepted."""
    _home(tmp_path, supervisor_state="stopped", flag=False)
    s = _settings(tmp_path)
    machine = FakeMachine(s, [], branch="silent")
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_FAILED
    err = capsys.readouterr().err
    assert "long-final" in err and "config.json" in err and "recorded" not in err
    assert machine.now - 1_800_000_000.0 >= p7_finish.VERIFY_S


def test_a_branch_whose_train_command_is_refused_fails_at_its_first_crash(tmp_path, capsys):
    _home(tmp_path, supervisor_state="stopped", flag=False)
    s = _settings(tmp_path)
    machine = FakeMachine(s, [], branch="refused")
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_FAILED
    assert "exited 2" in capsys.readouterr().err and machine.now - 1_800_000_000.0 < 60


def test_at_step_names_a_checkpoint_or_the_finish_refuses(tmp_path):
    s = _settings(tmp_path, at_step=110_000)
    machine = FakeMachine(s, [_supervisor()])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_REFUSED
    assert machine.killed == [] and machine.runs == []
    s = _settings(tmp_path, at_step=100_000)
    assert p7_finish.finish(s, FakeMachine(s, [_supervisor()])) == p7_finish.EXIT_DONE
    assert _record(s)["at_step"] == 100_000 and _record(s)["cooldown_steps"] == 25_000


def test_finish_refuses_while_runs_long_is_training(tmp_path, capsys):
    s = _settings(tmp_path)
    machine = FakeMachine(s, [_supervisor(41), {"pid": 50, "ppid": 41, "cmdline": TRAINER}])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_REFUSED
    assert "is training" in capsys.readouterr().err and machine.killed == [] and machine.runs == []
    assert not (s.home / "eval" / "p7_finish.json").exists()


def test_finish_never_stops_a_supervisor_that_could_start_a_trainer(tmp_path, capsys):
    """Only a supervisor paused by the user with the flag up holds no trainer and cannot start one."""
    s = _settings(tmp_path)
    (s.home / "PAUSE").unlink()  # Itay resumed a moment ago: the supervisor is restarting its trainer
    machine = FakeMachine(s, [_supervisor()])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_REFUSED
    assert "paused by the user" in capsys.readouterr().err and machine.killed == []


def test_a_stopped_flagship_with_no_supervisor_left_is_finished_without_stopping_anything(tmp_path):
    _home(tmp_path, supervisor_state="stopped", flag=False)
    s = _settings(tmp_path)
    machine = FakeMachine(s, [])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_DONE
    assert machine.killed == [] and _record(s)["stopped_supervisor"] == []


def test_a_fresh_running_heartbeat_refuses_even_when_no_trainer_process_is_seen(tmp_path):
    _home(tmp_path, supervisor_state="running", flag=False)
    s = _settings(tmp_path)
    beat = {"state": "running", "step": C, "time": 1_800_000_000.0 - 20}
    (s.runs / "long" / "heartbeat.json").write_text(json.dumps(beat), encoding="utf-8")
    assert p7_finish.finish(s, FakeMachine(s, [])) == p7_finish.EXIT_REFUSED


def test_the_final_cooldown_must_end_inside_long_toml_s_120_hour_upper_bound(tmp_path, capsys):
    s = _settings(tmp_path)
    steps = p7_finish.train_table(s.repo / "configs" / "long.toml")["steps"]
    late = steps - 10  # already inside its own planned cooldown: c + c/4 would pass the upper bound
    (s.runs / "long" / f"ckpt_{late:09d}.pt").write_bytes(b"")
    assert p7_finish.finish(s, FakeMachine(s, [_supervisor()])) == p7_finish.EXIT_REFUSED
    assert "upper bound" in capsys.readouterr().err


def test_a_second_finish_refuses_once_long_final_exists_and_says_how_to_resume_it(tmp_path, capsys):
    s = _settings(tmp_path)
    (s.runs / "long-final").mkdir()
    (s.runs / "long-final" / "config.json").write_text("{}", encoding="utf-8")
    assert p7_finish.finish(s, FakeMachine(s, [_supervisor()])) == p7_finish.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "long-final already exists" in err and "--resume" in err


def test_the_dry_run_prints_the_plan_and_the_command_and_changes_nothing(tmp_path, capsys):
    s = _settings(tmp_path, dry_run=True)
    machine = FakeMachine(s, [_supervisor(41)])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_DONE
    out = capsys.readouterr().out
    assert f"step {C:,}" in out and f"{K:,} cooldown steps" in out and f"{C + K:,}" in out
    assert "would stop runs/long's supervisor (pid 41)" in out and "--preview-name long-final" in out
    assert machine.killed == [] and machine.runs == [] and not (s.home / "eval" / "p7_finish.json").exists()


def test_a_failed_launch_is_reported_and_leaves_its_record_saying_so(tmp_path, capsys):
    s = _settings(tmp_path)
    assert p7_finish.finish(s, FakeMachine(s, [_supervisor()], launch_code=2)) == p7_finish.EXIT_FAILED
    assert _record(s)["launched"] is None and "ops launch exit 2" in capsys.readouterr().err


def test_the_served_run_is_read_as_blink_supervise_reads_it():
    assert p7_finish.served_run(p7_finish.blink_args(SUPERVISOR)) == "long"
    assert p7_finish.served_run(p7_finish.blink_args(TRAINER)) == "long"
    assert p7_finish.served_run(p7_finish.blink_args(SIZE_M)) == "size-m"
    assert p7_finish.served_run(["supervise", "--run", "x", "--", "train"]) == "x"
    assert p7_finish.served_run(["train", "--run", "long", "--from-step", "5", "--preview-steps", "2"]) == (
        "long-preview"
    )
    assert p7_finish.blink_args(["python.exe", "tools/p7_v2_driver.py"]) == []


def test_the_hours_are_blink_s_training_seconds_to_the_step():
    from blink.train.calibrate import training_seconds

    def row(i: int) -> dict:  # a restart after step 1,500 (500 s down), an eval window every 7th
        phase = "eval" if i % 7 == 0 else "train"
        return {"step": 50 * i, "time": 100.0 * i + (500 if i > 30 else 0), "phase": phase,
                "session": 1.0 if i <= 30 else 2.0}  # fmt: skip

    rows = [row(i) for i in range(1, 60)]
    assert p7_finish.training_hours(rows, 2000) == pytest.approx(training_seconds(rows[:40]) / 3600)
    assert p7_finish.training_hours(rows, 10**9) == pytest.approx(training_seconds(rows) / 3600)


def test_the_command_line_defaults_and_the_dry_run_flag():
    s = p7_finish.settings_from(["--dry-run", "--at-step", "5000"])
    assert (s.repo, s.home, s.at_step, s.dry_run) == (
        Path(r"C:\dev\blink-run"),
        Path(r"D:\blink"),
        5000,
        True,
    )
    assert s.data == Path(r"D:\blink") / "data" / "v1" and s.launch_name == "p7-long-final"
    assert sys.modules["p7_finish"] is p7_finish


PREVIEW = [*PYTHON, "supervise", "--", "train", "--run", "long", "--data", "D:/blink/data/v1",
           "--preview-cooldown", "3h", "--from-step", "331758"]  # fmt: skip
OTHERS = {
    "a paused long-preview supervisor": PREVIEW,
    "a size-m branch supervisor": SIZE_M,
    "a sweep": [*PYTHON, "sweep", "ablations", "--config", "configs/sweep.toml"],
    "a calibration": [*PYTHON, "train", "calibrate", "--config", "configs/long.toml"],
    "a throughput bench": [*PYTHON, "bench", "throughput", "--size", "m"],
}


@pytest.mark.parametrize("other", sorted(OTHERS))
def test_finish_refuses_while_another_blink_run_could_take_the_gpu(tmp_path, capsys, other):
    """Anything else that trains or starts training, paused or not, would train beside long-final once
    Resume Blink removes the flag: nothing is stopped and nothing launches while one is there."""
    s = _settings(tmp_path)
    machine = FakeMachine(s, [_supervisor(41), {"pid": 55, "ppid": 1, "cmdline": OTHERS[other]}])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "pid 55" in err and machine.killed == [] and machine.runs == []
    assert not (s.home / "eval" / "p7_finish.json").exists()


def test_a_dry_run_or_a_read_only_blink_command_beside_runs_long_does_not_refuse(tmp_path):
    s = _settings(tmp_path)
    harmless = [[*PYTHON, "sweep", "ablations", "--dry-run"], [*PYTHON, "eval", "strength", "--run", "long"],
                [*PYTHON, "status", "--live"], [*PYTHON, "ops", "ps"]]  # fmt: skip
    procs = [_supervisor(41), *({"pid": 60 + i, "ppid": 1, "cmdline": c} for i, c in enumerate(harmless))]
    machine = FakeMachine(s, procs)
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_DONE and machine.killed == [41]


def test_a_run_that_starts_after_the_stop_blocks_the_launch(tmp_path, capsys):
    """Looked at again just before the launch: runs/long's supervisor is gone, and nothing launches."""
    s = _settings(tmp_path)

    class Crowded(FakeMachine):
        def kill_tree(self, pid):
            super().kill_tree(pid)
            self.procs.append({"pid": 70, "ppid": 1, "cmdline": PREVIEW})

    machine = Crowded(s, [_supervisor(41)])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_FAILED
    err = capsys.readouterr().err
    assert "pid 70 (blink supervise, runs/long-preview)" in err and "nothing was launched" in err
    assert machine.runs == [] and not (s.home / "eval" / "p7_finish.json").exists()


def test_the_gpu_work_is_every_command_blink_ops_counts_and_every_sweep_action():
    from blink.ops.launch import GPU_COMMANDS, gpu_command

    for command in GPU_COMMANDS:
        assert p7_finish.gpu_work([*command, "--run", "x"]) and gpu_command([*command, "--run", "x"])
    assert p7_finish.gpu_work(["sweep", "sizes", "--resume"]) and p7_finish.gpu_work(["sweep", "choose"])
    assert not p7_finish.gpu_work(["sweep", "ablations", "--dry-run"]) and not p7_finish.gpu_work(["eval"])
    assert p7_finish.blink_args([r"C:\venv\Scripts\blink.exe", "sweep", "sizes"]) == ["sweep", "sizes"]


class Stubborn(FakeMachine):
    """A supervisor that outlives kill_tree (or one that another process restarted at once)."""

    def kill_tree(self, pid):
        self.killed.append(-pid)


def test_nothing_launches_while_runs_long_s_supervisor_is_still_there_after_the_stop(tmp_path, capsys):
    s = _settings(tmp_path)
    machine = Stubborn(s, [_supervisor(41)])
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_FAILED
    assert machine.runs == [] and "still" in capsys.readouterr().err


def test_the_stop_is_checked_again_just_before_it_happens(tmp_path, capsys):
    """Itay resumes between the look and the stop: the supervisor starts a trainer, so nothing is killed."""
    s = _settings(tmp_path)

    class Resumed(FakeMachine):
        def processes(self):
            if self.looks:
                (self.s.home / "PAUSE").unlink(missing_ok=True)
                self.procs.append({"pid": 50, "ppid": 41, "cmdline": TRAINER})
            self.looks += 1
            return super().processes()

    machine = Resumed(s, [_supervisor(41)])
    machine.looks = 0
    assert p7_finish.finish(s, machine) == p7_finish.EXIT_REFUSED
    assert machine.killed == [] and machine.runs == [] and "is training" in capsys.readouterr().err
