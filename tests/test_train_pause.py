"""The trainer's side of the user pause: BLINK_HOME/PAUSE ends a supervised run at a step boundary (or
between eval chunks) with a checkpoint of that step, a run nothing would resume trains on, and any run
that starts while the flag is up builds nothing yet."""

import json
import threading
import time

import pytest
import zstandard

torch = pytest.importorskip("torch")

from train_helpers import FIXTURE, fixture_records, tiny_train_config  # noqa: E402

from blink import cli, heartbeat  # noqa: E402
from blink.train import loop, supervise, userpause  # noqa: E402
from blink.train.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint, step_of  # noqa: E402
from blink.train.source import InMemorySource  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"


@pytest.fixture(autouse=True)
def every_step(monkeypatch):
    """Production looks for the flag every few seconds; these runs take milliseconds a step."""
    monkeypatch.setattr(loop, "PAUSE_CHECK_S", 0.0)
    monkeypatch.setattr(userpause, "POLL_S", 0.02)


def _spec(run_dir, flag, **overrides) -> loop.RunSpec:
    """A run whose supervisor resumes it after a user pause (pause_exits), unless overridden."""
    fields = {"run_dir": run_dir, "world": WORLD, "device": "cpu", "pause_flag": flag, "pause_exits": True}
    return loop.RunSpec(**{**fields, **overrides})


def _flag_at(source, flag, step: int):
    """The source, raising the flag while it hands out the batch of step `step` + 1."""

    def batches(start_step: int):
        for index, batch in enumerate(source(start_step), start=start_step):
            if index == step:
                flag.touch()
            yield batch

    return batches


def _steps(path) -> list[int]:
    return [json.loads(line)["step"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_pause_checkpoints_its_step_boundary_and_the_resumed_run_equals_a_straight_one(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=40, ckpt_every_steps=20, warmup_steps=5, batch_size=16, eval_every=20)
    flag, quiet = tmp_path / "PAUSE", {"val": records, "log": lambda _: None}
    straight = loop.train(
        cfg, _spec(tmp_path / "straight", None), InMemorySource(records, 16, seed=3).batches, **quiet
    )

    run_dir = tmp_path / "paused"
    source = _flag_at(InMemorySource(records, 16, seed=3).batches, flag, 6)
    paused = loop.train(cfg, _spec(run_dir, flag), source, **quiet)
    assert paused.paused is True and paused.step == 7
    assert step_of(latest_checkpoint(run_dir)) == 7 and paused.checkpoint.name == "ckpt_000000007.pt"
    assert heartbeat.read(run_dir / "heartbeat.json")["state"] == userpause.PAUSED_USER

    flag.unlink()
    done = loop.train(
        cfg, _spec(run_dir, flag, resume=True), InMemorySource(records, 16, seed=3).batches, **quiet
    )
    assert done.paused is False and done.step == straight.step == 40
    a, b = (
        load_checkpoint(latest_checkpoint(tmp_path / "straight")),
        load_checkpoint(latest_checkpoint(run_dir)),
    )
    for name, tensor in a["model"].items():
        assert torch.equal(tensor, b["model"][name]), f"model.{name} differs after the pause"
    assert _steps(run_dir / "evals.jsonl") == [0, 20, 40]


def test_a_run_nothing_would_resume_trains_through_a_flag_raised_mid_run(tmp_path):
    """Plain `blink train` (the gap's calibrations, a job run by hand): an exit 75 would fail its caller,
    and a resumed calibration would time the pause into its rate. Only a supervised run stops."""
    records = fixture_records()
    cfg = tiny_train_config(steps=12, warmup_steps=2, batch_size=16, eval_every=6)
    flag, run_dir = tmp_path / "PAUSE", tmp_path / "run"
    source = _flag_at(InMemorySource(records, 16, seed=3).batches, flag, 4)
    done = loop.train(cfg, _spec(run_dir, flag, pause_exits=False), source, val=records, log=lambda _: None)
    assert flag.exists() and done.paused is False and done.step == 12
    assert heartbeat.read(run_dir / "heartbeat.json")["state"] == "finished"


def test_a_pause_between_eval_chunks_checkpoints_that_step_and_the_resume_scores_it_once(
    tmp_path, monkeypatch
):
    """The flag goes up during step 20's eval: the run checkpoints step 20 without its evals row, and the
    resumed run scores step 20 before training on, so no check is skipped or written twice."""
    records = fixture_records()
    cfg = tiny_train_config(steps=40, ckpt_every_steps=40, warmup_steps=5, batch_size=16, eval_every=10)
    flag, run_dir, real = tmp_path / "PAUSE", tmp_path / "run", loop.evals.evaluate

    def chunked(run, label=None, tick=None):  # a VAA check calls tick after every chunk
        if run.step == 20 and not (run_dir / "ckpt_000000020.pt").exists():
            flag.touch()
        if tick is not None:
            tick()
        return real(run, label, tick)

    monkeypatch.setattr(loop.evals, "evaluate", chunked)
    source = InMemorySource(records, 16, seed=1).batches
    paused = loop.train(cfg, _spec(run_dir, flag), source, val=records, log=lambda _: None)
    assert paused.paused and paused.step == 20
    assert [step_of(p) for p in list_checkpoints(run_dir)] == [20]
    assert _steps(run_dir / "evals.jsonl") == [0, 10]  # step 20's eval never finished

    flag.unlink()
    done = loop.train(cfg, _spec(run_dir, flag, resume=True), source, val=records, log=lambda _: None)
    assert done.step == 40 and _steps(run_dir / "evals.jsonl") == [0, 10, 20, 30, 40]


def test_a_run_that_starts_while_paused_builds_no_model_until_the_flag_goes(tmp_path, monkeypatch):
    records = fixture_records()
    cfg = tiny_train_config(steps=10, batch_size=16, warmup_steps=2)
    flag, run_dir, events = tmp_path / "PAUSE", tmp_path / "run", []
    flag.touch()
    real_build = loop._build
    monkeypatch.setattr(loop, "_build", lambda *a, **k: events.append("build") or real_build(*a, **k))

    def gamer_done():
        deadline = time.time() + 30
        while time.time() < deadline:
            beat = heartbeat.read(run_dir / "heartbeat.json") or {}
            if beat.get("state") == userpause.PAUSED_USER:
                break
            time.sleep(0.01)
        events.append(("waiting", (heartbeat.read(run_dir / "heartbeat.json") or {}).get("state")))
        flag.unlink()

    thread = threading.Thread(target=gamer_done)
    thread.start()
    spec = _spec(run_dir, flag, max_steps=3, pause_exits=False)  # waiting to start suits any run
    result = loop.train(cfg, spec, InMemorySource(records, 16, seed=1).batches, val=None)
    thread.join()
    assert events == [("waiting", userpause.PAUSED_USER), "build"] and result.step == 3


CONFIG = """
[model]
d_model = 64
n_layers = 1
n_heads = 2

[train]
batch_size = 16
steps = 30
warmup_steps = 5
metrics_every = 10
eval_every = 15
val_size = 64
ckpt_every_steps = 15
ckpt_every_minutes = 0.0
"""


def _raise_flag_at_step_7(tmp_path, monkeypatch, run: str) -> list[str]:
    """`blink train` argv for a tiny run whose flag goes up after step 7 (once: not again on resume)."""
    home = tmp_path / "home"
    monkeypatch.setenv("BLINK_HOME", str(home))
    config, raw = tmp_path / "tiny.toml", tmp_path / "evals.jsonl.zst"
    config.write_text(CONFIG, encoding="utf-8")
    raw.write_bytes(zstandard.ZstdCompressor().compress(FIXTURE.read_bytes()))
    real_after = loop._after_step

    def after(run_state, window, lr, end):
        real_after(run_state, window, lr, end)
        if run_state.step == 7 and not (home / "runs" / run / "ckpt_000000007.pt").exists():
            userpause.flag_path().touch()

    monkeypatch.setattr(loop, "_after_step", after)
    return ["train", "--config", str(config), "--run", run, "--source-raw", str(raw), "--device", "cpu"]


def test_blink_train_outside_a_supervisor_that_resumes_it_ignores_a_pause_mid_run(tmp_path, monkeypatch):
    monkeypatch.delenv(userpause.RESUMER_ENV, raising=False)
    argv = _raise_flag_at_step_7(tmp_path, monkeypatch, "plain")
    assert cli.main(argv) == 0
    assert userpause.flag_path().exists()
    assert heartbeat.read(tmp_path / "home" / "runs" / "plain" / "heartbeat.json")["state"] == "finished"


def test_blink_train_exits_with_the_user_pause_code_and_resumes_to_the_end(tmp_path, monkeypatch):
    monkeypatch.setenv(userpause.RESUMER_ENV, "1")  # what blink supervise sets for its child
    argv = _raise_flag_at_step_7(tmp_path, monkeypatch, "p")
    home = tmp_path / "home"
    assert cli.main(argv) == supervise.EXIT_USER_PAUSE
    run_dir = home / "runs" / "p"
    assert step_of(latest_checkpoint(run_dir)) == 7
    assert heartbeat.read(run_dir / "heartbeat.json")["state"] == userpause.PAUSED_USER
    userpause.flag_path().unlink()
    assert cli.main([*argv, "--resume"]) == 0
    assert heartbeat.read(run_dir / "heartbeat.json")["state"] == "finished"
