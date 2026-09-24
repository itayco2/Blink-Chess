"""`blink supervise`: every P7 stop rule enforced by the supervisor itself, with no agent watching.

A fake trainer (a small script written into tmp_path) plays scripted scenarios: it beats its
heartbeat, writes crafted metrics.jsonl and evals.jsonl rows, hangs or exits. The supervisor runs it
as a real child process with every interval scaled down from seconds and minutes to fractions of a
second, and each test checks the verdict, the files it leaves and that the child is gone.
"""

import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import psutil
import pytest

from blink import cli, heartbeat
from blink.train import status, supervise

FAKE_TRAINER = r"""
import json, os, sys, time
from pathlib import Path
from blink import heartbeat

args = sys.argv[1:]
run_dir = Path(args[args.index("--run-dir") + 1])
scenario = json.loads(Path(args[args.index("--scenario") + 1]).read_text(encoding="utf-8"))
calls = run_dir / "calls.jsonl"
index = len(calls.read_text(encoding="utf-8").splitlines()) if calls.exists() else 0
with open(calls, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": args, "pid": os.getpid()}) + "\n")
plan = scenario["runs"][min(index, len(scenario["runs"]) - 1)]
step = 0


def beat(state="running"):
    heartbeat.write(run_dir / "heartbeat.json", {"kind": "train", "state": state, "step": step})


def append(name, row):
    row = {**row, "time": time.time()} if row.get("time") == "now" else row
    with open(run_dir / name, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


for action in plan:
    kind, value = next(iter(action.items()))
    if kind == "beat":
        beat(value)
    elif kind in ("metrics", "evals"):
        for row in value["rows"]:
            step = row.get("step", step)
            append(f"{kind}.jsonl", row)
            beat()
            time.sleep(value.get("every", 0.0))
    elif kind == "ckpt":
        (run_dir / f"ckpt_{value:09d}.pt").write_bytes(b"checkpoint")
    elif kind == "sleep":
        time.sleep(value)
    elif kind == "exit":
        beat("finished" if value == 0 else "crashed")
        sys.exit(value)
    elif kind == "hang":
        while True:
            if value == "beating":
                beat()
            time.sleep(0.05)
"""

FAST = supervise.SuperviseConfig(
    interval_s=0.1,
    poll_s=0.02,
    heartbeat_stale_s=0.6,
    startup_grace_s=1.5,
    slow_window_s=0.5,
    backoff_s=0.05,
    terminate_timeout_s=5.0,
)


def _metric_rows(steps, **fields):
    return [{"step": s, "loss_policy": 2.0, "loss_value": 1.0, "clip_frac": 0.0, **fields} for s in steps]


def _launch(tmp_path: Path, runs: list, cfg=FAST, timeout_s: float = 60.0):
    run_dir = tmp_path / "runs" / "fake"
    run_dir.mkdir(parents=True)
    script = tmp_path / "fake_trainer.py"
    script.write_text(FAKE_TRAINER, encoding="utf-8")
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    argv = [sys.executable, str(script), "--run-dir", str(run_dir), "--scenario", str(scenario)]
    outcome = supervise.supervise(cfg, run_dir, argv, log=lambda _: None, deadline_s=timeout_s)
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    return outcome, run_dir, calls


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _stopped_by(run_dir: Path) -> str:
    return json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))["stopped"]


BEAT = {"beat": "running"}
CKPT = {"ckpt": 100}
HANG = {"hang": "beating"}
CLIP_ROWS = _metric_rows(range(50, 1251, 50), clip_frac=0.5)
LOSS_ROWS = _metric_rows(range(50, 501, 50)) + [
    {"step": s, "loss_policy": 30.0, "loss_value": 5.0, "clip_frac": 0.0} for s in range(550, 1801, 50)
]
SLOW_ROWS = [{"step": 50 * i, "samples_per_s": 100.0, "time": "now", "phase": "train"} for i in range(1, 40)]
NAN_ROW = {"step": 250, "loss_policy": math.nan, "loss_value": 1.0, "clip_frac": 0.0}
STOP_CASES = {
    "heartbeat": [[BEAT, CKPT, {"hang": "silent"}]],
    "throughput": [[BEAT, CKPT, {"metrics": {"rows": SLOW_ROWS, "every": 0.05}}, HANG]],
    "clip": [[BEAT, CKPT, {"metrics": {"rows": CLIP_ROWS}}, HANG]],
    "loss": [[BEAT, CKPT, {"metrics": {"rows": LOSS_ROWS}}, HANG]],
    "nan": [[BEAT, CKPT, {"metrics": {"rows": [NAN_ROW]}}, {"exit": 1}]],
    "crash": [[BEAT, CKPT, {"exit": 1}]],
}


@pytest.mark.parametrize("rule", sorted(STOP_CASES))
def test_supervise_halts_the_trainer_on_each_stop_rule_without_an_agent(rule, tmp_path):
    cfg = supervise.SuperviseConfig(**{**FAST.__dict__, "bench_rate": 1000.0})
    outcome, run_dir, calls = _launch(tmp_path, STOP_CASES[rule], cfg=cfg)
    assert outcome.state == "stopped" and outcome.exit_code == supervise.EXIT_STOPPED
    assert outcome.status.startswith(f"stopped: {rule}, "), outcome.status
    assert _stopped_by(run_dir) == outcome.status
    assert all(_gone(call["pid"]) for call in calls)
    record = json.loads((run_dir / "supervisor.json").read_text(encoding="utf-8"))
    assert record["state"] == "stopped" and record["status"] == outcome.status
    assert (run_dir / "ckpt_000000100.pt").is_file()  # a stop never deletes checkpoints
    expected_calls = {"nan": 2, "crash": supervise.SuperviseConfig().max_restarts + 1}.get(rule, 1)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("rule", ["heartbeat", "clip", "nan", "crash", "vaa"])
def test_status_exits_nonzero_on_each_stop_rule(rule, tmp_path):
    vaa = [[BEAT, CKPT, {"evals": {"rows": [{"step": 100, "vaa_check_failed": True}]}}, HANG]]
    runs = STOP_CASES.get(rule) or vaa
    outcome, run_dir, _ = _launch(tmp_path, runs)
    report = status.run_status(run_dir)
    assert status.exit_code(report) == 1
    assert supervise.status_exit_code(supervise.read_record(run_dir)) == 1
    assert report.state in ("stopped", "paused")


def test_a_failed_vaa_check_pauses_the_run_and_keeps_checkpoints(tmp_path):
    marker = {"evals": {"rows": [{"step": 100, "vaa": 0.41, "vaa_check_failed": True}]}}
    outcome, run_dir, calls = _launch(tmp_path, [[BEAT, CKPT, marker, HANG]])
    assert outcome.state == "paused" and outcome.status == "paused: P7-VAA"
    assert outcome.exit_code == supervise.EXIT_PAUSED
    beat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["state"] == "paused" and beat["stopped"] == "paused: P7-VAA"
    assert _gone(calls[0]["pid"]) and (run_dir / "ckpt_000000100.pt").is_file()


def test_the_first_nan_rolls_back_once_with_half_the_learning_rate(tmp_path):
    first = [BEAT, CKPT, {"metrics": {"rows": [NAN_ROW]}}, {"exit": 1}]
    second = [BEAT, {"metrics": {"rows": _metric_rows([150, 200])}}, {"exit": 0}]
    outcome, run_dir, calls = _launch(tmp_path, [first, second])
    assert outcome.state == "finished" and outcome.exit_code == 0
    assert "--resume" not in calls[0]["argv"]
    assert calls[1]["argv"][-3:] == ["--resume", "--lr-scale", "0.5"]
    record = json.loads((run_dir / "supervisor.json").read_text(encoding="utf-8"))
    assert record["rollbacks"] == 1 and record["lr_scale"] == 0.5
    assert [e["event"] for e in record["events"]].count("rollback") == 1


def test_a_crash_resumes_after_the_backoff_and_a_clean_exit_finishes(tmp_path):
    outcome, run_dir, calls = _launch(tmp_path, [[BEAT, CKPT, {"exit": 1}], [BEAT, {"exit": 0}]])
    assert outcome.state == "finished"
    assert calls[1]["argv"][-1] == "--resume"
    assert json.loads((run_dir / "supervisor.json").read_text(encoding="utf-8"))["restarts"] == 1


def test_a_crash_before_any_checkpoint_restarts_without_resume(tmp_path):
    outcome, _, calls = _launch(tmp_path, [[BEAT, {"exit": 1}], [BEAT, {"exit": 0}]])
    assert outcome.state == "finished" and "--resume" not in calls[1]["argv"]


def test_a_run_with_checkpoints_is_refused_without_resume(tmp_path):
    run_dir = tmp_path / "runs" / "fake"
    run_dir.mkdir(parents=True)
    (run_dir / "ckpt_000000100.pt").write_bytes(b"checkpoint")
    (run_dir / "metrics.jsonl").write_text('{"step": 150}\n', encoding="utf-8")
    outcome = supervise.supervise(FAST, run_dir, [sys.executable, "-c", "pass"], log=lambda _: None)
    assert outcome.exit_code == supervise.EXIT_REFUSED and "--resume" in outcome.status
    assert (run_dir / "metrics.jsonl").read_text(encoding="utf-8") == '{"step": 150}\n'


def test_a_resumed_start_truncates_the_logs_to_the_checkpoint_like_the_trainer(tmp_path):
    run_dir = tmp_path / "runs" / "fake"
    run_dir.mkdir(parents=True)
    (run_dir / "ckpt_000000100.pt").write_bytes(b"checkpoint")
    rows = [{"step": 50}, {"step": 100}, {"step": 150, "loss_policy": math.nan}]
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    argv = [sys.executable, "-c", "pass", "--resume"]
    outcome = supervise.supervise(FAST, run_dir, argv, log=lambda _: None)
    assert outcome.state == "finished"
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [50, 100]


def test_slow_eval_and_checkpoint_windows_never_stop_a_healthy_run(tmp_path):
    windows = [{**row, "phase": "eval" if i % 2 else "ckpt"} for i, row in enumerate(SLOW_ROWS[:20])]
    runs = [[BEAT, {"metrics": {"rows": windows, "every": 0.05}}, {"exit": 0}]]
    cfg = supervise.SuperviseConfig(**{**FAST.__dict__, "bench_rate": 1000.0})
    outcome, _, _ = _launch(tmp_path, runs, cfg=cfg)
    assert outcome.state == "finished"


def test_a_disabled_rule_is_recorded_and_never_fires(tmp_path):
    cfg = supervise.SuperviseConfig(**{**FAST.__dict__, "disabled": ("clip",)})
    outcome, run_dir, _ = _launch(tmp_path, [[BEAT, {"metrics": {"rows": CLIP_ROWS}}, {"exit": 0}]], cfg=cfg)
    assert outcome.state == "finished"
    assert json.loads((run_dir / "supervisor.json").read_text(encoding="utf-8"))["disabled"] == ["clip"]


def test_clip_fraction_is_weighted_by_the_steps_each_row_covers():
    rows = [{"step": 1, "clip_frac": 1.0}] + _metric_rows(range(50, 1001, 50), clip_frac=0.1)
    assert supervise.clip_verdict(rows, supervise.SuperviseConfig()) is None
    hot = rows + [{"step": 1200, "clip_frac": 0.9}]
    verdict = supervise.clip_verdict(hot, supervise.SuperviseConfig())
    assert verdict.rule == "clip" and verdict.number == "26.0%"
    assert supervise.clip_verdict(rows[:10], supervise.SuperviseConfig()) is None  # under 1,000 steps seen


def test_a_loss_spike_must_last_over_1000_steps_and_never_moves_its_own_baseline():
    cfg = supervise.SuperviseConfig()
    calm = _metric_rows(range(50, 1001, 50))
    short = calm + [{"step": s, "loss_policy": 20.0, "loss_value": 1.0} for s in range(1050, 2001, 50)]
    assert supervise.loss_verdict(short, cfg) is None  # 950 steps above
    long = short + [{"step": 2050, "loss_policy": 20.0, "loss_value": 1.0}]
    assert supervise.loss_verdict(long, cfg).rule == "loss"
    recovered = short + [{"step": 2050, "loss_policy": 2.0, "loss_value": 1.0}]
    assert supervise.loss_verdict(recovered, cfg) is None


def test_throughput_counts_only_train_rows_and_resets_on_a_fast_one():
    cfg = supervise.SuperviseConfig(bench_rate=1000.0)
    slow = [{"step": i, "samples_per_s": 800.0, "_t": 60.0 * i} for i in range(12)]
    assert supervise.throughput_verdict(slow, cfg).rule == "throughput"
    reset = slow[:6] + [{"step": 6, "samples_per_s": 900.0, "_t": 360.0}] + slow[7:]
    assert supervise.throughput_verdict(reset, cfg) is None
    evals = [{**row, "phase": "eval"} for row in slow]
    assert supervise.throughput_verdict(evals, cfg) is None
    assert supervise.throughput_verdict(slow, supervise.SuperviseConfig()) is None  # no benchmark: off


def test_restart_argv_replaces_resume_and_lr_scale_flags():
    base = ["python", "-m", "blink.cli", "train", "--run", "x", "--resume", "--lr-scale", "0.5"]
    assert supervise.restart_argv(base, resume=True, lr_scale=0.25)[-4:] == [
        "x",
        "--resume",
        "--lr-scale",
        "0.25",
    ]
    assert supervise.restart_argv(base, resume=False, lr_scale=1.0) == base[:6]
    assert supervise.lr_scale_of(["train", "--lr-scale=0.5"]) == 0.5


def test_train_args_get_the_run_name_and_refuse_another_one():
    assert supervise.train_argv(["train", "--config", "c.toml"], "long")[-5:] == [
        "train",
        "--config",
        "c.toml",
        "--run",
        "long",
    ]
    with pytest.raises(ValueError, match="--run other"):
        supervise.train_argv(["train", "--run", "other"], "long")
    with pytest.raises(ValueError, match="train"):
        supervise.train_argv(["eval", "--run", "long"], "long")


def test_the_supervise_command_wraps_blink_train_and_adds_the_run_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    argv = ["supervise", "--run", "long", "--dry-run", "--", "train", "--config", "configs/t.toml"]
    assert cli.main(argv) == 0
    command, rule = capsys.readouterr().out.strip().splitlines()
    assert command.endswith("-m blink.cli train --config configs/t.toml --run long")
    assert rule == "throughput rule: off (no --bench-rate or --bench-size)"


def test_the_supervise_command_takes_the_run_name_from_the_train_command(tmp_path, monkeypatch, capsys):
    """The plan's P7 line is `blink supervise -- train --config configs/long.toml --run long`."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    argv = ["supervise", "--dry-run", "--", "train", "--config", "configs/long.toml", "--run", "long"]
    assert cli.main(argv) == 0
    assert "-m blink.cli train --config configs/long.toml --run long" in capsys.readouterr().out
    assert cli.main(["supervise", "--dry-run", "--", "train", "--config", "c.toml"]) == 2
    assert "--run" in capsys.readouterr().err


def _bench_json(path: Path) -> Path:
    ok = {"size": "s", "oom": False, "error": None}
    rows = [
        {**ok, "micro": 256, "compile": "off", "samples_per_s": 3000.0},
        {**ok, "micro": 512, "compile": "inductor", "samples_per_s": 4000.0},
    ]
    path.write_text(json.dumps({"throughput": rows}), encoding="utf-8")
    return path


def test_the_supervise_command_reads_its_benchmark_rate_from_bench_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    bench = _bench_json(tmp_path / "bench.json")
    common = ["supervise", "--run", "long", "--dry-run", "--bench", str(bench)]
    assert cli.main([*common, "--bench-size", "s", "--", "train"]) == 0
    expected = "throughput rule: floor 3,400 samples/s (85% of 4,000, size s in bench.json)"
    assert expected in capsys.readouterr().out
    assert cli.main([*common, "--bench-size", "m12", "--", "train"]) == 2
    assert "m12" in capsys.readouterr().err
    assert cli.main(["supervise", "--run", "long", "--dry-run", "--", "train"]) == 0
    assert "throughput rule: off" in capsys.readouterr().out


def test_the_supervise_command_refuses_a_mismatched_run_and_an_unknown_rule(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert cli.main(["supervise", "--run", "long", "--", "train", "--run", "short"]) == 2
    assert "--run short" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["supervise", "--run", "long", "--disable", "sleepy", "--", "train"])


def test_the_supervise_command_runs_a_child_to_the_end(tmp_path, monkeypatch, capsys):
    """A real `blink train` child that fails fast on a missing config: 3 crash resumes, then a stop."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    argv = [
        "supervise",
        "--run",
        "r",
        "--interval",
        "0.2",
        "--backoff",
        "0",
        "--",
        "train",
        "--config",
        "nope.toml",
    ]
    code = cli.main([*argv, "--data", str(tmp_path)])
    assert code == supervise.EXIT_STOPPED
    record = supervise.read_record(tmp_path / "runs" / "r")
    assert record["status"] == "stopped: crash, exit 2 after 3 restarts"
    assert record["launch_command"]


def test_a_momentarily_unreadable_heartbeat_is_not_a_stale_one(tmp_path):
    """Found by this suite: a read that raced the trainer's os.replace once looked like a 1 s old run."""
    run_dir = tmp_path / "runs" / "fake"
    run_dir.mkdir(parents=True)
    cfg = supervise.SuperviseConfig(heartbeat_stale_s=0.5, startup_grace_s=0.0)
    sup = supervise.Supervisor(cfg, run_dir, ["python"], log=lambda _: None)
    started = time.time() - 5.0
    sup.child = supervise._Child(types.SimpleNamespace(pid=0, poll=lambda: None), ["python"], started)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 3})
    sup._track_progress()
    (run_dir / "heartbeat.json").write_text("{half a jso", encoding="utf-8")  # mid-replace
    assert sup._check(time.time()) is None
    assert supervise.heartbeat_verdict(None, started, time.time(), True, cfg).rule == "heartbeat"


ORPHAN_PARENT = r"""
import os, subprocess, sys, time
from blink.train import supervise
print(supervise.bind_children_to_this_process() if sys.argv[1] == "bind" else "free", flush=True)
code = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)"
subprocess.Popen([sys.executable, "-c", code, sys.argv[2]])
print(os.getpid(), flush=True)
time.sleep(120)
"""


def _grandchild_after_parent_dies(tmp_path: Path, mode: str) -> tuple[str, int, bool]:
    script, pid_file = tmp_path / f"parent_{mode}.py", tmp_path / f"grandchild_{mode}.txt"
    script.write_text(ORPHAN_PARENT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(supervise.__file__).resolve().parents[2])}
    parent = subprocess.Popen(
        [sys.executable, str(script), mode, str(pid_file)], stdout=subprocess.PIPE, text=True, env=env
    )
    note, parent_pid = parent.stdout.readline().strip(), int(parent.stdout.readline())
    deadline = time.time() + 30
    while time.time() < deadline and not pid_file.exists():
        time.sleep(0.1)
    time.sleep(0.3)
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    psutil.Process(parent_pid).kill()  # TerminateProcess: the supervisor gets no chance to clean up
    time.sleep(3)
    alive = psutil.pid_exists(grandchild) and not _gone(grandchild)
    if alive:
        psutil.Process(grandchild).kill()
    parent.kill()
    parent.wait()
    return note, grandchild, alive


@pytest.mark.skipif(sys.platform != "win32", reason="kill-on-close job objects are a Windows feature")
def test_the_trainer_dies_with_its_supervisor_even_when_the_supervisor_is_killed(tmp_path):
    note, _, alive = _grandchild_after_parent_dies(tmp_path, "free")
    assert note == "free" and alive  # without the job, a killed supervisor leaves an orphan trainer
    note, _, alive = _grandchild_after_parent_dies(tmp_path, "bind")
    assert note.startswith("bound") and not alive


def test_a_locked_status_file_never_takes_the_supervision_down(tmp_path, monkeypatch):
    def locked(path, text):
        raise PermissionError(13, "held open by a reader", str(path))

    monkeypatch.setattr(supervise, "write_text_atomic", locked)
    monkeypatch.setattr(supervise.heartbeat, "write", lambda *a, **k: locked("heartbeat.json", ""))
    run_dir = tmp_path / "runs" / "fake"
    run_dir.mkdir(parents=True)
    lines = []
    outcome = supervise.supervise(
        FAST, run_dir, [sys.executable, "-c", "raise SystemExit(3)"], log=lines.append
    )
    assert outcome.status == "stopped: crash, exit 3 after 3 restarts"
    assert any("could not write supervisor.json" in line for line in lines)
    assert any("could not write heartbeat.json" in line for line in lines)
