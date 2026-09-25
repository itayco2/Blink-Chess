"""A check checkpoint (5/25/30/50/100%) survives every resume: a crash-resume and an in-place re-plan.

The trainer protects the check steps of its current plan from pruning. A resume whose plan moved (a
changed `steps`) or that cannot score checks (no valprobe) must still keep the checkpoints behind the
check rows already in evals.jsonl: the 30% one is where the preview branches from, and every one is a
point on the learning curve that FINDINGS cites.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_train_config  # noqa: E402

from blink.train import loop, vaa  # noqa: E402
from blink.train.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint, step_of  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"


def _spec(run_dir, **overrides) -> loop.RunSpec:
    return loop.RunSpec(**{"run_dir": run_dir, "world": WORLD, "device": "cpu", **overrides})


def _repeat(batch):
    def source(start_step):
        while True:
            yield batch

    return source


def _train(cfg, run_dir, probe, **spec) -> None:
    batch = fixture_records()[:16]
    loop.train(
        cfg, _spec(run_dir, **spec), _repeat(batch), val=fixture_records(), log=lambda _: None, probe=probe
    )


def _steps(run_dir) -> list[int]:
    return [step_of(p) for p in list_checkpoints(run_dir)]


def _check_rows(run_dir) -> dict[int, str]:
    lines = (run_dir / "evals.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["step"]: row["check"] for row in rows if "check" in row}


# checkpoints every 5 steps and only the last one kept: anything unprotected is pruned at once
PRUNING = {"batch_size": 16, "ckpt_every_steps": 5, "keep_last": 1, "eval_every": 1000, "warmup_steps": 1}


@pytest.fixture
def probe():
    return vaa.probe_from_roots(fixture_records()[:12])


def _first_leg(tmp_path, probe):
    """40 planned steps (checks at 2, 10, 12, 20 and 40), stopped at step 25."""
    cfg = tiny_train_config(steps=40, **PRUNING)
    run_dir = tmp_path / "long"
    _train(cfg, run_dir, probe, max_steps=25)
    assert _check_rows(run_dir) == {2: "5%", 10: "25%", 12: "30%", 20: "50%"}
    assert _steps(run_dir) == [2, 10, 12, 20, 25]
    return cfg, run_dir


def test_a_resume_that_replans_the_steps_keeps_every_earlier_check_checkpoint(tmp_path, probe):
    cfg, run_dir = _first_leg(tmp_path, probe)
    replanned = tiny_train_config(steps=60, **PRUNING)  # checks move to 3, 15, 18, 30 and 60
    assert set(vaa.check_steps(60)).isdisjoint({2, 10, 12, 20})
    _train(replanned, run_dir, probe, resume=True)
    assert _steps(run_dir) == [2, 10, 12, 20, 30, 60]
    assert load_checkpoint(latest_checkpoint(run_dir))["kept"] == [2, 10, 12, 20]


def test_a_crash_resume_that_cannot_score_checks_still_keeps_the_check_checkpoints(tmp_path, probe):
    cfg, run_dir = _first_leg(tmp_path, probe)
    _train(cfg, run_dir, None, resume=True)  # no valprobe this time: the plan has no check steps
    assert _steps(run_dir) == [2, 10, 12, 20, 40]


def test_a_resume_protects_only_check_rows_it_keeps_after_rewinding(tmp_path, probe):
    """A check row a dead attempt wrote past the resume step is cut with the logs, so it protects
    nothing: the checkpoint behind it does not exist in this attempt's history."""
    cfg, run_dir = _first_leg(tmp_path, probe)
    for path in list_checkpoints(run_dir):
        if step_of(path) > 12:
            path.unlink()  # the attempt died with its 30% checkpoint as the latest
    _train(cfg, run_dir, None, resume=True, max_steps=30)
    assert load_checkpoint(latest_checkpoint(run_dir))["kept"] == [2, 10, 12]
