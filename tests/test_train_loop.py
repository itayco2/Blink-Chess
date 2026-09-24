import json
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, held_open, tiny_model_config, tiny_train_config  # noqa: E402

from blink.train import loop  # noqa: E402
from blink.train.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint, step_of  # noqa: E402
from blink.train.source import InMemorySource  # noqa: E402
from blink.train.world import WorldMismatch  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"


def _spec(run_dir, **overrides) -> loop.RunSpec:
    return loop.RunSpec(**{"run_dir": run_dir, "world": WORLD, "device": "cpu", **overrides})


def _records_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _repeat(batch: np.ndarray):
    def source(start_step: int):
        while True:
            yield batch

    return source


def test_a_tiny_model_memorises_one_batch(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(
        model=tiny_model_config(d_model=64, n_layers=1, n_heads=2),
        batch_size=len(records),
        steps=500,
        peak_lr=3e-3,
        warmup_steps=20,
        eval_every=100,
        val_size=len(records),
        ckpt_every_steps=500,
    )
    result = loop.train(cfg, _spec(tmp_path / "memorise"), _repeat(records), val=records, log=lambda _: None)
    assert result.step == 500
    assert result.last_eval["top1"] >= 0.99


def _state_after(cfg, run_dir, source, **spec_overrides):
    loop.train(cfg, _spec(run_dir, **spec_overrides), source, val=fixture_records(), log=lambda _: None)
    return load_checkpoint(latest_checkpoint(run_dir))


def test_resume_after_100_steps_equals_200_straight_steps_bitwise_on_cpu(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=200, ckpt_every_steps=100, warmup_steps=20, batch_size=16)
    straight = _state_after(cfg, tmp_path / "straight", InMemorySource(records, 16, seed=3).batches)

    split_dir = tmp_path / "split"
    first = _state_after(cfg, split_dir, InMemorySource(records, 16, seed=3).batches, max_steps=100)
    assert first["step"] == 100
    resumed = _state_after(cfg, split_dir, InMemorySource(records, 16, seed=3).batches, resume=True)

    assert resumed["step"] == straight["step"] == 200
    for key in ("model", "ema"):
        assert straight[key].keys() == resumed[key].keys()
        for name, tensor in straight[key].items():
            assert torch.equal(tensor, resumed[key][name]), f"{key}.{name} differs after resume"
    adam_straight = straight["optimizer"]["state"]
    adam_resumed = resumed["optimizer"]["state"]
    for index, slots in adam_straight.items():
        for slot, tensor in slots.items():
            assert torch.equal(tensor, adam_resumed[index][slot]), f"optimizer {index}.{slot} differs"


def test_resume_refuses_a_different_world(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=40, ckpt_every_steps=20, batch_size=16)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir, max_steps=20), _repeat(records[:16]), val=records, log=lambda _: None)
    with pytest.raises(WorldMismatch):
        loop.train(
            cfg,
            _spec(run_dir, resume=True, world="ffffffffffff"),
            _repeat(records[:16]),
            val=records,
            log=lambda _: None,
        )
    assert [step_of(p) for p in list_checkpoints(run_dir)] == [20]


def test_a_new_run_refuses_to_overwrite_existing_checkpoints(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=20, ckpt_every_steps=10, batch_size=16)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir), _repeat(records[:16]), val=records, log=lambda _: None)
    with pytest.raises(loop.RunExists):
        loop.train(cfg, _spec(run_dir), _repeat(records[:16]), val=records, log=lambda _: None)


def test_resume_without_a_checkpoint_is_refused(tmp_path):
    cfg = tiny_train_config(steps=20, batch_size=16)
    with pytest.raises(FileNotFoundError):
        loop.train(cfg, _spec(tmp_path / "empty", resume=True), _repeat(fixture_records()[:16]), val=None)


def test_metrics_and_evals_are_written_on_their_cadence(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=120, metrics_every=50, eval_every=60, batch_size=16, ckpt_every_steps=1000)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir), _repeat(records[:16]), val=records, log=lambda _: None)

    metrics = _records_jsonl(run_dir / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [1, 50, 100, 120]
    fields = {
        "step",
        "loss_policy",
        "loss_value",
        "lr",
        "grad_norm",
        "clip_frac",
        "samples_per_s",
        "gpu_mem_gb",
    }
    assert fields <= set(metrics[0])
    assert metrics[0]["loss_policy"] == pytest.approx(7.54, abs=0.1)
    assert metrics[0]["loss_value"] == pytest.approx(4.85, abs=0.05)

    evals = _records_jsonl(run_dir / "evals.jsonl")
    assert [e["step"] for e in evals] == [0, 60, 120]
    assert {"top1", "value_ce", "win_mae", "policy_ce", "ema_top1"} <= set(evals[0])
    assert [step_of(p) for p in list_checkpoints(run_dir)] == [120]

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["world"] == WORLD and config["config"]["steps"] == 120
    assert config["parameters"] > 0


def test_resume_drops_telemetry_written_after_the_checkpoint(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=100, metrics_every=10, eval_every=10, batch_size=16, ckpt_every_steps=50)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir, max_steps=70), _repeat(records[:16]), val=records, log=lambda _: None)
    for path in list_checkpoints(run_dir)[1:]:
        path.unlink()  # pretend the crash came before the step-70 checkpoint
    assert step_of(latest_checkpoint(run_dir)) == 50
    loop.train(cfg, _spec(run_dir, resume=True), _repeat(records[:16]), val=records, log=lambda _: None)
    steps = [m["step"] for m in _records_jsonl(run_dir / "metrics.jsonl")]
    assert steps == sorted(set(steps))
    assert steps[-1] == 100


def test_the_heartbeat_ends_with_the_final_state(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=30, batch_size=16, heartbeat_s=0.0)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir, max_steps=10), _repeat(records[:16]), val=records, log=lambda _: None)
    beat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["state"] == "stopped" and beat["step"] == 10 and beat["steps"] == 30
    loop.train(cfg, _spec(run_dir, resume=True), _repeat(records[:16]), val=records, log=lambda _: None)
    beat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["state"] == "finished" and beat["step"] == 30


def test_a_crash_is_written_to_the_heartbeat(tmp_path):
    cfg = tiny_train_config(steps=30, batch_size=16)
    run_dir = tmp_path / "run"

    def broken(start_step):
        yield fixture_records()[:16]
        raise RuntimeError("loader died")

    with pytest.raises(RuntimeError, match="loader died"):
        loop.train(cfg, _spec(run_dir), broken, val=fixture_records(), log=lambda _: None)
    beat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["state"] == "crashed" and "loader died" in beat["error"]


def test_weight_decay_applies_to_matrices_only():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = loop.build_optimizer(model, tiny_train_config(), "cpu")
    by_decay = {group["weight_decay"]: group["params"] for group in optimizer.param_groups}
    assert all(p.ndim >= 2 for p in by_decay[0.1])
    assert all(p.ndim < 2 for p in by_decay[0.0])


def test_a_batch_source_that_runs_dry_is_named_in_the_error(tmp_path):
    cfg = tiny_train_config(steps=30, batch_size=16)

    def short(start_step):
        yield from [fixture_records()[:16]] * 3

    with pytest.raises(RuntimeError, match="ran out of batches at step 3"):
        loop.train(cfg, _spec(tmp_path / "dry"), short, val=None, log=lambda _: None)


def test_an_empty_val_array_skips_evals(tmp_path):
    cfg = tiny_train_config(steps=5, warmup_steps=1, batch_size=16)
    run_dir = tmp_path / "noval"
    empty = fixture_records()[:0]
    loop.train(cfg, _spec(run_dir), _repeat(fixture_records()[:16]), val=empty, log=lambda _: None)
    assert not (run_dir / "evals.jsonl").exists()


@pytest.mark.cuda
def test_a_short_cuda_run_trains_in_bf16_and_resumes(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=40, warmup_steps=4, batch_size=32, ckpt_every_steps=20, eval_every=20)
    run_dir = tmp_path / "gpu"
    source = InMemorySource(records, 32, seed=1).batches
    quiet = {"val": records, "log": lambda _: None}
    first = loop.train(cfg, _spec(run_dir, device="cuda", max_steps=20), source, **quiet)
    assert first.step == 20 and first.last_metrics["gpu_mem_gb"] > 0
    done = loop.train(cfg, _spec(run_dir, device="cuda", resume=True), source, **quiet)
    assert done.step == 40
    assert done.last_eval["policy_ce"] < 7.54
    state = load_checkpoint(latest_checkpoint(run_dir))
    assert state["rng"]["cuda"] and state["step"] == 40


def test_resume_succeeds_while_the_dashboard_briefly_holds_the_logs_open(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(steps=40, metrics_every=10, eval_every=10, batch_size=16, ckpt_every_steps=20)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir, max_steps=30), _repeat(records[:16]), val=records, log=lambda _: None)
    assert step_of(latest_checkpoint(run_dir)) == 30
    list_checkpoints(run_dir)[-1].unlink()  # the resume rewinds to step 20 and truncates both logs
    with held_open(run_dir / "metrics.jsonl", run_dir / "evals.jsonl"):
        result = loop.train(
            cfg, _spec(run_dir, resume=True), _repeat(records[:16]), val=records, log=lambda _: None
        )
    assert result.step == 40
    assert [m["step"] for m in _records_jsonl(run_dir / "metrics.jsonl")] == [1, 10, 20, 30, 40]


def test_a_new_run_rewrites_a_stale_config_a_reader_is_holding_open(tmp_path):
    records = fixture_records()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")  # an attempt that died before step 1
    cfg = tiny_train_config(steps=20, batch_size=16, ckpt_every_steps=10)
    with held_open(run_dir / "config.json"):
        loop.train(cfg, _spec(run_dir, max_steps=1), _repeat(records[:16]), val=records, log=lambda _: None)
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["world"] == WORLD


def _sleepy(batch: np.ndarray, seconds: float):
    def source(start_step: int):
        while True:
            time.sleep(seconds)
            yield batch

    return source


def test_metrics_record_how_long_the_loop_waited_for_its_batches(tmp_path):
    records = fixture_records()
    cfg = tiny_train_config(
        steps=9, warmup_steps=2, metrics_every=3, batch_size=16, eval_every=1000, ckpt_every_steps=1000
    )
    for name, seconds in (("fast", 0.0), ("slow", 0.1)):
        loop.train(cfg, _spec(tmp_path / name), _sleepy(records[:16], seconds), val=None, log=lambda _: None)
    fast, slow = (_records_jsonl(tmp_path / name / "metrics.jsonl") for name in ("fast", "slow"))
    assert all(0.0 <= row["data_wait_frac"] <= 1.0 for row in fast + slow)
    assert all(row["time"] > 1_600_000_000 for row in fast + slow)
    # A window starts when the row before it is written, so share x window = seconds waited. The sleeps
    # bound that from below whatever the machine's load; a share alone would not be.
    slow_waits = _waits(slow)
    assert all(waited >= 0.95 * 0.1 * steps for waited, steps in slow_waits)
    assert sum(w for w, _ in _waits(fast)) < sum(w for w, _ in slow_waits)


def _waits(rows: list[dict]) -> list[tuple[float, int]]:
    """(seconds the loop waited, steps) of each metrics window after the first."""
    return [
        (row["data_wait_frac"] * (row["time"] - before["time"]), row["step"] - before["step"])
        for before, row in zip(rows, rows[1:], strict=False)
    ]
