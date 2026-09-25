"""The user pause (BLINK_HOME/PAUSE): the flag check, the supervisor, the sweep and `blink status`.

Most supervisor tests run on fake clocks: a scripted child (FakeProc) answers poll() from the fake time,
and sleeping moves that time and fires the test's events (the flag going up or down). One test runs a
real child process through a real pause and resume.
"""

import json
import sys
import threading
import time
import timeit
from pathlib import Path

from blink import cli, heartbeat
from blink.train import status, supervise, sweep, userpause

# ---------------------------------------------------------------- fakes


class FakeClock:
    """time.time and time.sleep for a Supervisor: a sleep moves the clock, then fires every due event."""

    def __init__(self, start: float):
        self.now, self.events = start, []

    def time(self) -> float:
        return self.now

    def at(self, when: float, action) -> None:
        self.events.append((when, action))

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        for event in sorted(self.events, key=lambda e: e[0]):
            if event[0] <= self.now:
                self.events.remove(event)
                event[1]()


class FakeProc:
    """A child whose exit code comes from `script(proc)` on each poll (None: still running)."""

    pid = 2**31 - 1

    def __init__(self, argv: list[str], script):
        self.argv, self.script, self.code, self.seen = argv, script, None, None

    def poll(self):
        if self.code is None:
            self.code = self.script(self)
        return self.code

    def wait(self, timeout=None):
        self.code = 15 if self.code is None else self.code
        return self.code


class Spawner:
    """subprocess.Popen for a Supervisor: child i runs scripts[i] (the last script repeats)."""

    def __init__(self, clock: FakeClock, *scripts):
        self.clock, self.scripts, self.calls = clock, scripts, []

    def __call__(self, argv, env=None):
        self.calls.append((self.clock.now, list(argv)))
        return FakeProc(list(argv), self.scripts[min(len(self.calls), len(self.scripts)) - 1])


CFG = supervise.SuperviseConfig(
    interval_s=10.0,
    poll_s=1.0,
    heartbeat_stale_s=60.0,
    startup_grace_s=100.0,
    backoff_s=10_000.0,  # a user pause taken for a crash would wait this long before its restart
    pause_poll_s=5.0,
    pause_kill_s=10_000.0,
)
T0 = 1_000_000_000.0  # the fake clock starts far below real time: real beats never look older


def _supervisor(tmp_path, clock, spawner, cfg=CFG, argv=("trainer", "--run", "r")):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True, exist_ok=True)
    flag = tmp_path / userpause.FLAG_NAME
    sup = supervise.Supervisor(
        cfg,
        run_dir,
        list(argv),
        log=lambda _: None,
        pause_flag=flag,
        clock=clock.time,
        sleep=clock.sleep,
        spawn=spawner,
    )
    return sup, run_dir, flag


def _beating(clock, run_dir):
    """A training child: one heartbeat per poll, one step further each time, at the fake time."""

    def script(proc):
        proc.seen = (proc.seen or 100) + 1
        heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": proc.seen}, now=clock.now)
        return None

    return script


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "supervisor.json").read_text(encoding="utf-8"))


def _events(run_dir: Path) -> list[str]:
    return [e["event"] for e in _record(run_dir)["events"]]


# ---------------------------------------------------------------- the flag


def test_the_flag_lives_in_blink_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert userpause.flag_path() == tmp_path / "PAUSE"
    assert userpause.flag_path(tmp_path / "x") == tmp_path / "x" / "PAUSE"


def test_the_flag_watch_looks_at_the_disk_at_most_once_per_interval(tmp_path):
    now, looks = [0.0], []
    watch = userpause.FlagWatch(
        tmp_path / "PAUSE", 2.0, clock=lambda: now[0], exists=lambda p: looks.append(p) or True
    )
    assert watch.requested() is True  # the first question always looks
    now[0] = 1.9
    assert watch.requested() is False and len(looks) == 1
    now[0] = 2.0
    assert watch.requested() is True and len(looks) == 2


def test_the_per_step_flag_check_costs_a_clock_read_not_a_disk_look(tmp_path):
    """The trainer asks every step (about 100 ms at S); an unthrottled os.path.exists costs 77 us on D:
    while a run trains, the throttled question well under a microsecond (asserted under 10 us: 0.01%)."""
    watch = userpause.FlagWatch(tmp_path / "PAUSE", 2.0)
    watch.requested()
    calls = 100_000
    per_call = timeit.timeit(watch.requested, number=calls) / calls
    assert per_call < 10e-6, f"{per_call * 1e6:.2f} us per step"


def test_waiting_while_flagged_beats_and_returns_the_seconds_waited(tmp_path):
    flag = tmp_path / "PAUSE"
    flag.touch()
    clock, beats = FakeClock(0.0), []
    clock.at(12.0, flag.unlink)
    waited = userpause.wait_while_flagged(flag, 5.0, lambda: beats.append(clock.now), clock.sleep, clock.time)
    assert waited == 15.0 and beats == [0.0, 5.0, 10.0]
    assert userpause.wait_while_flagged(flag, 5.0, sleep=clock.sleep, clock=clock.time) == 0.0


# ---------------------------------------------------------------- the supervisor


def test_a_user_pause_is_no_crash_suspends_the_rules_and_extends_the_deadline_by_the_paused_time(tmp_path):
    clock = FakeClock(T0)
    seen = {}

    def pausing(proc):  # silent throughout; 300 s after the flag it has checkpointed and exits
        if not flag.exists():
            return None
        proc.seen = proc.seen or clock.now
        if clock.now - proc.seen < 300:
            return None
        (run_dir / "ckpt_000000100.pt").write_bytes(b"checkpoint")
        return supervise.EXIT_USER_PAUSE

    def look():
        beat = heartbeat.read(run_dir / "heartbeat.json")
        report = status.run_status(run_dir)
        seen.update(beat=beat, record=_record(run_dir), code=status.exit_code(report), badge=report.state)

    spawner = Spawner(clock, pausing, None)
    sup, run_dir, flag = _supervisor(tmp_path, clock, spawner)
    spawner.scripts = (pausing, _beating(clock, run_dir))
    clock.at(T0 + 50, flag.touch)
    clock.at(T0 + 700, look)
    clock.at(T0 + 1050, flag.unlink)
    outcome = sup.run(deadline_s=500.0)

    # without suspension the heartbeat rule fires at T0 + 100 (a silent child past its grace)
    assert outcome.state == "stopped" and outcome.status.startswith("stopped: wall_clock, 500 s")
    assert sup.paused_s == 1000.0 and clock.now == T0 + 1500  # the deadline moved by exactly the pause
    assert [t for t, _ in spawner.calls] == [T0, T0 + 1050]  # no 10,000 s crash backoff
    assert spawner.calls[1][1][-1] == "--resume" and sup.restarts == 0
    assert seen["beat"]["state"] == userpause.PAUSED_USER and seen["badge"] == userpause.PAUSED_USER
    assert seen["record"]["state"] == seen["record"]["status"] == userpause.PAUSED_USER
    assert seen["code"] == 0 and supervise.status_exit_code(seen["record"]) == 0
    events = _events(run_dir)
    assert {"user_pause", "user_pause_exit", "user_resume"} <= set(events) and "crash" not in events


def test_a_flag_up_when_supervise_starts_holds_the_child_until_it_goes(tmp_path):
    clock = FakeClock(T0)
    spawner = Spawner(clock, None)
    sup, run_dir, flag = _supervisor(tmp_path, clock, spawner)
    spawner.scripts = (_beating(clock, run_dir),)
    flag.touch()
    clock.at(T0 + 200, flag.unlink)
    outcome = sup.run(deadline_s=100.0)
    assert [t for t, _ in spawner.calls] == [T0 + 200] and "--resume" not in spawner.calls[0][1]
    assert outcome.status.startswith("stopped: wall_clock, 100 s") and clock.now == T0 + 300


def test_a_child_that_never_saw_the_flag_gets_a_fresh_startup_grace_when_it_goes(tmp_path):
    """A trainer still building its model (or waiting to start) when the flag goes up and down: the
    pause must not count against its startup grace, or the heartbeat rule stops it at once."""
    clock = FakeClock(T0)
    spawner = Spawner(clock, None)
    sup, run_dir, flag = _supervisor(tmp_path, clock, spawner)

    def slow_start(proc):
        if clock.now < T0 + 550:
            return None
        return 0 if clock.now >= T0 + 600 else _beating(clock, run_dir)(proc)

    spawner.scripts = (slow_start,)
    clock.at(T0 + 10, flag.touch)
    clock.at(T0 + 500, flag.unlink)
    outcome = sup.run()
    assert outcome.state == "finished" and sup.paused_s == 490.0 and len(spawner.calls) == 1


def test_a_child_that_ignores_the_flag_is_ended_after_the_kill_wait_and_resumed_later(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(supervise, "kill_tree", lambda pid, timeout_s: killed.append(pid))
    clock = FakeClock(T0)
    spawner = Spawner(clock, lambda proc: None, lambda proc: 0)
    cfg = supervise.SuperviseConfig(**{**CFG.__dict__, "pause_kill_s": 240.0})
    sup, run_dir, flag = _supervisor(tmp_path, clock, spawner, cfg=cfg, argv=("trainer", "--resume"))
    (run_dir / "ckpt_000000100.pt").write_bytes(b"checkpoint")
    clock.at(T0 + 10, flag.touch)
    clock.at(T0 + 400, flag.unlink)
    outcome = sup.run()
    assert outcome.state == "finished" and killed == [FakeProc.pid]
    assert [t for t, _ in spawner.calls] == [T0, T0 + 400] and spawner.calls[1][1][-1] == "--resume"
    assert "user_pause_kill" in _events(run_dir) and sup.restarts == 0


def test_the_p7_vaa_pause_still_ends_supervision_and_the_flag_does_not_lift_it(tmp_path):
    clock = FakeClock(T0)

    def fails_a_check_then_pauses(proc):
        if clock.now < T0 + 20:
            return None
        row = {"step": 100, "vaa": 0.41, "vaa_check_failed": True}
        (run_dir / "evals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        flag.touch()
        return supervise.EXIT_USER_PAUSE

    spawner = Spawner(clock, fails_a_check_then_pauses)
    sup, run_dir, flag = _supervisor(tmp_path, clock, spawner)
    outcome = sup.run()
    assert outcome.state == "paused" and outcome.status == "paused: P7-VAA"
    assert outcome.exit_code == supervise.EXIT_PAUSED and clock.now == T0 + 20
    flag.unlink()  # Resume Blink: nothing restarts, the gate is Itay's
    beat = heartbeat.read(run_dir / "heartbeat.json")
    assert beat["state"] == "paused" and beat["stopped"] == "paused: P7-VAA"
    record = _record(run_dir)
    assert record["pending_gate"] == "P7-VAA" and supervise.status_exit_code(record) == 1
    assert len(spawner.calls) == 1


FAKE_TRAINER = r"""
import json, sys, time
from pathlib import Path
from blink import heartbeat

args = sys.argv[1:]
run_dir, flag = Path(args[args.index("--run-dir") + 1]), Path(args[args.index("--flag") + 1])
calls = run_dir / "calls.jsonl"
with open(calls, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": args}) + "\n")
beat = lambda state: heartbeat.write(run_dir / "heartbeat.json", {"state": state, "step": 100})
if "--resume" in args:
    beat("running")
    sys.exit(0)
while not flag.exists():
    beat("running")
    time.sleep(0.02)
(run_dir / "ckpt_000000100.pt").write_bytes(b"checkpoint")
beat("paused: user")
sys.exit(75)
"""


def test_a_real_child_pauses_and_the_supervisor_resumes_it_without_a_backoff(tmp_path):
    run_dir, flag = tmp_path / "runs" / "fake", tmp_path / "PAUSE"
    run_dir.mkdir(parents=True)
    script = tmp_path / "fake_trainer.py"
    script.write_text(FAKE_TRAINER, encoding="utf-8")
    argv = [sys.executable, str(script), "--run-dir", str(run_dir), "--flag", str(flag)]
    cfg = supervise.SuperviseConfig(
        interval_s=0.1, poll_s=0.02, heartbeat_stale_s=5.0, backoff_s=30.0, pause_poll_s=0.05
    )
    seen = {}

    def gamer():
        deadline = time.time() + 30
        while not (run_dir / "calls.jsonl").exists() and time.time() < deadline:
            time.sleep(0.02)
        flag.touch()
        while time.time() < deadline:
            beat = heartbeat.read(run_dir / "heartbeat.json") or {}
            if beat.get("state") == userpause.PAUSED_USER and "supervisor_pid" in beat:
                break
            time.sleep(0.02)
        seen["code"] = status.exit_code(status.run_status(run_dir))
        seen["record"] = supervise.read_record(run_dir)
        flag.unlink()

    thread = threading.Thread(target=gamer)
    thread.start()
    started = time.time()
    outcome = supervise.supervise(cfg, run_dir, argv, log=lambda _: None, pause_flag=flag)
    thread.join()
    assert outcome.state == "finished" and time.time() - started < 20  # no 30 s crash backoff
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 2 and calls[1]["argv"][-1] == "--resume"
    assert seen["code"] == 0 and seen["record"]["state"] == userpause.PAUSED_USER
    record = supervise.read_record(run_dir)
    assert record["restarts"] == 0 and "crash" not in [e["event"] for e in record["events"]]


# ---------------------------------------------------------------- blink status


def _beat(run_dir: Path, state: str, age_s: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    heartbeat.write(
        run_dir / "heartbeat.json", {"state": state, "step": 7, "steps": 10}, now=time.time() - age_s
    )


def test_blink_status_tells_a_user_pause_from_a_stall(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    _beat(tmp_path / "runs" / "paused", userpause.PAUSED_USER, 3.0)
    assert cli.main(["status", "--run", "paused"]) == 0
    out = capsys.readouterr().out
    assert "PAUSED: USER" in out and "Resume Blink" in out
    _beat(tmp_path / "runs" / "gone", userpause.PAUSED_USER, 600.0)  # no supervisor beats it any more
    assert cli.main(["status", "--run", "gone"]) == 1
    _beat(tmp_path / "runs" / "stalled", "running", 600.0)
    assert cli.main(["status", "--run", "stalled"]) == 1


# ---------------------------------------------------------------- the sweep


BASE = "[model]\nd_model = 64\nn_layers = 1\nn_heads = 2\nhead_dim = 32\n\n[train]\nbatch_size = 1000\n"
SEEDS = ("a01", "a02", "a03")


def _plan(tmp_path: Path) -> sweep.AblationPlan:
    folder = tmp_path / "ablations"
    folder.mkdir()
    (tmp_path / "s.toml").write_text(
        BASE + "steps = 100\npeak_lr = 0.001\nwarmup_steps = 10\n", encoding="utf-8"
    )
    for i, name in enumerate(SEEDS, start=1):
        (folder / f"{name}.toml").write_text(
            f'[arm]\nchange = "seed"\n[train]\nseed = {i}\n', encoding="utf-8"
        )
    (folder / "plan.toml").write_text(
        f'[plan]\nrecipe = "{(tmp_path / "s.toml").as_posix()}"\ndata = "{tmp_path.as_posix()}"\n'
        f'hours = 0.01\nsize = "s"\nsigma_arms = {json.dumps(SEEDS)}\narms = {json.dumps(SEEDS)}\n',
        encoding="utf-8",
    )
    return sweep.load_plan(folder / "plan.toml")


def _final_row(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {"step": 36, "vaa": 0.5, "top1": 0.3}
    (run_dir / "evals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_an_arm_not_started_yet_waits_while_the_flag_is_up(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("BLINK_HOME", str(home))
    monkeypatch.setattr(userpause, "POLL_S", 0.02)
    flag, out, requests, seen = home / "PAUSE", home / "eval" / "ablations.json", [], {}
    home.mkdir()
    flag.touch()

    def runner(request):
        requests.append(request.run)
        _final_row(home / "runs" / request.run)
        return supervise.Outcome("finished", "finished", 0)

    def resume_later():
        time.sleep(0.3)
        seen.update(started=list(requests), recorded=out.exists())
        flag.unlink()

    thread = threading.Thread(target=resume_later)
    thread.start()
    report = sweep.run_ablations(_plan(tmp_path), out, 1000.0, runner, log=lambda _: None)
    thread.join()
    assert seen == {"started": [], "recorded": False}  # nothing planned, scored or run while paused
    assert requests == ["abl-a01", "abl-a02", "abl-a03"]
    assert all(report["arms"][name]["status"] == "finished" for name in SEEDS)


def test_a_user_pause_mid_arm_keeps_the_arm_running_and_continues_the_same_arm(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("BLINK_HOME", str(home))
    clock, flag, out, seen, supervisors = (
        FakeClock(T0),
        home / "PAUSE",
        home / "eval" / "ablations.json",
        {},
        {},
    )
    home.mkdir()

    def runner(request):
        run_dir = home / "runs" / request.run

        def pauses(proc):
            if not flag.exists():
                return None
            (run_dir / "ckpt_000000010.pt").write_bytes(b"checkpoint")
            return supervise.EXIT_USER_PAUSE

        def finishes(proc):
            _final_row(run_dir)
            return 0

        scripts = (pauses, finishes) if request.run == "abl-a02" else (finishes,)
        spawner = Spawner(clock, *scripts)
        argv = ["trainer", "--run", request.run] + (["--resume"] if request.resume else [])
        run_dir.mkdir(parents=True, exist_ok=True)
        sup = supervise.Supervisor(
            CFG,
            run_dir,
            argv,
            lambda _: None,
            pause_flag=flag,
            clock=clock.time,
            sleep=clock.sleep,
            spawn=spawner,
        )
        supervisors[request.run] = spawner
        if request.run == "abl-a02":
            clock.at(clock.now + 5, flag.touch)
            clock.at(clock.now + 100, lambda: seen.update(a02=json.loads(out.read_text(encoding="utf-8"))))
            clock.at(clock.now + 200, flag.unlink)
        return sup.run(request.deadline_s)

    report = sweep.run_ablations(_plan(tmp_path), out, 1000.0, runner, log=lambda _: None)
    assert seen["a02"]["arms"]["a02"]["status"] == "running"  # no final status while paused
    assert "a03" not in seen["a02"]["arms"]  # the sweep did not move on
    assert all(report["arms"][name]["status"] == "finished" for name in SEEDS)
    calls = supervisors["abl-a02"].calls
    assert len(calls) == 2 and calls[1][1][-1] == "--resume"  # the same arm, resumed
