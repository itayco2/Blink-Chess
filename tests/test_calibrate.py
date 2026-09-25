"""PR-5's calibration: R_true from the metrics rows' time stamps, then steps = floor(120 x 3600 x R / 1024).

The arithmetic is tested on synthetic metrics rows; one CPU run of `blink train calibrate` checks the
command end to end on a tiny config (the real 2,000-step run happens on the GPU before the P7 launch).
"""

import json
import math
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records  # noqa: E402

from blink import cli  # noqa: E402
from blink.model.config import config_to_dict, load_config  # noqa: E402
from blink.train import calibrate, film, vaa  # noqa: E402

pytestmark = pytest.mark.torch
BATCH = 1024


def _rows(rate: float = 2600.0, every: int = 50, last: int = 2000, start: float = 1000.0) -> list[dict]:
    """A 2,000-step run's metrics rows: slow compiled first steps, an eval window at step 1,000 and a
    checkpoint window at 1,500 (each slower by what ran in it), train windows at `rate` elsewhere.
    samples_per_s holds nonsense on purpose: R_true never reads it."""
    rows, now = [{"step": 1, "time": start, "phase": "train", "samples_per_s": 1.0}], start + 30.0
    for step in range(every, last + 1, every):
        phase = {1000: "eval", 1500: "ckpt"}.get(step, "train")
        extra = {"eval": 15.0, "ckpt": 1.5}.get(phase, 0.0)
        slow = 3.0 if step <= 500 else 1.0  # the first 500 steps run slower
        now += slow * (step - rows[-1]["step"]) * BATCH / rate + extra
        rows.append({"step": step, "time": now, "phase": phase, "samples_per_s": 99999.0})
    return rows


def test_r_true_is_samples_over_seconds_of_train_windows_after_step_500():
    rate = calibrate.true_rate(_rows(), BATCH)
    # 500->550 ... 1950->2000 is 30 intervals; the eval (950->1000) and ckpt (1450->1500) windows drop out
    assert (rate.intervals, rate.first_step, rate.last_step) == (28, 500, 2000)
    assert rate.samples == 28 * 50 * BATCH
    assert rate.samples_per_s == pytest.approx(2600.0, rel=1e-12)
    assert rate.seconds == pytest.approx(28 * 50 * BATCH / 2600.0, rel=1e-12)


def test_the_window_after_an_eval_counts_since_the_eval_is_charged_to_its_own_row():
    rows = _rows()
    for row in rows:
        if row["step"] >= 1050:
            row["time"] += 7.0  # the train window after the eval ran 7 s slower: counted, so R_true falls
    slower = calibrate.true_rate(rows, BATCH)
    assert slower.intervals == 28 and slower.seconds == pytest.approx(28 * 50 * BATCH / 2600.0 + 7.0)


def test_steps_is_the_floor_of_120_hours_at_r_true_over_the_batch():
    assert calibrate.flagship_steps(2621.3, BATCH) == 1_105_860  # 1,105,860.47 rounded down
    assert calibrate.flagship_steps(2600.0, BATCH) == math.floor(120 * 3600 * 2600.0 / 1024)
    rate = calibrate.true_rate(_rows(rate=2710.0), BATCH)
    assert calibrate.flagship_steps(rate.samples_per_s, BATCH) in (1_143_281, 1_143_280)  # float slack
    assert calibrate.T_LONG_HOURS == 120.0 and calibrate.SKIP_STEPS == 500


def test_r_true_refuses_rows_without_time_stamps_or_out_of_order_and_a_run_too_short():
    rows = _rows()
    rows[20].pop("time")
    with pytest.raises(ValueError, match="time stamp"):
        calibrate.true_rate(rows, BATCH)
    backwards = _rows()
    at = [r["step"] for r in backwards].index(1250)
    backwards[at]["time"] = backwards[at - 1]["time"] - 1.0
    with pytest.raises(ValueError, match="in order"):
        calibrate.true_rate(backwards, BATCH)
    with pytest.raises(ValueError, match="after step 500"):
        calibrate.true_rate(_rows(last=500), BATCH)


def test_everything_that_depends_on_steps_is_derived_from_it_at_run_time():
    facts = calibrate.derived(1_105_860, 0.2)
    assert facts["checks"] == {label: step for step, label in vaa.check_steps(1_105_860).items()}
    assert facts["preview_from_step"] == facts["checks"]["30%"] == 331_758
    assert facts["cooldown_start"] == 1_105_860 - round(0.2 * 1_105_860)
    assert facts["film_frames"] == len(film.frame_plan(1_105_860)) == 21


LONG_LIKE = """# a header comment that must survive
base = "m.toml"

[train]
steps = 661078          # replaced by `blink train calibrate --write`
eval_every = 4000       # PR-5
warmup_steps = 2000
ckpt_every_steps = 1000000
"""


def test_with_steps_replaces_only_the_train_tables_steps_line():
    new = calibrate.with_steps(LONG_LIKE, 1_105_860, "PR-5 note").splitlines()
    old = LONG_LIKE.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(new, old, strict=True)) if a != b]
    assert changed == [old.index("steps = 661078          # replaced by `blink train calibrate --write`")]
    assert new[changed[0]].startswith("steps = 1105860 ") and new[changed[0]].endswith("# PR-5 note")
    with pytest.raises(ValueError, match="steps"):
        calibrate.with_steps("[train]\nwarmup_steps = 5\n", 10, "x")
    crlf = calibrate.with_steps("[model]\r\nsteps = 1\r\n[train]\r\nsteps = 5\r\nseed = 1\r\n", 7, "n")
    assert crlf.split("\r\n") == ["[model]", "steps = 1", "[train]", f"{'steps = 7':<23} # n", "seed = 1", ""]


def test_write_steps_changes_the_real_long_config_in_steps_only(repo_root, tmp_path):
    for name in ("recipe.toml", "m.toml", "long.toml"):
        shutil.copy(repo_root / "configs" / name, tmp_path / name)
    before = load_config(tmp_path / "long.toml")
    old = calibrate.write_steps(tmp_path / "long.toml", 1_105_860, "PR-5: calibrated")
    after = load_config(tmp_path / "long.toml")
    assert old == before.steps and after.steps == 1_105_860
    expected = {**config_to_dict(before), "steps": 1_105_860}
    assert config_to_dict(after) == expected


# ---------------------------------------------------------------- the command, on CPU

TINY = """
[model]
d_model = 64
n_layers = 1
n_heads = 2

[train]
batch_size = 16
steps = 100000          # the flagship's plan: the calibration stops early
warmup_steps = 5
metrics_every = 50
eval_every = 100000
val_size = 64
ckpt_every_steps = 100000
ckpt_every_minutes = 0.0
film = true
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def shards(tmp_path, monkeypatch):
    import sys
    import types

    from test_train_command import FakeShardLoader

    fake = types.ModuleType("blink.data.loader")
    fake.ShardLoader = FakeShardLoader
    monkeypatch.setitem(sys.modules, "blink.data.loader", fake)
    root = tmp_path / "shards"
    root.mkdir()
    fixture_records().tofile(root / "train_000.bin")
    fixture_records()[:40].tofile(root / "val.bin")
    (root / "manifest.json").write_text(json.dumps({"train": 100}), encoding="utf-8")
    return root


def _metrics(run_dir) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def test_calibrate_trains_a_throwaway_run_and_writes_the_steps_it_measured(home, shards, tmp_path, capsys):
    config = tmp_path / "long.toml"
    config.write_text(TINY, encoding="utf-8")
    argv = ["train", "calibrate", "--config", str(config), "--steps", "600", "--data", str(shards)]
    assert cli.main([*argv, "--device", "cpu", "--run", "calib-t", "--write"]) == 0
    run_dir = home / "runs" / "calib-t"
    rate = calibrate.true_rate(_metrics(run_dir), 16)
    steps = calibrate.flagship_steps(rate.samples_per_s, 16)
    out = capsys.readouterr().out
    assert f"= {steps:,}" in out and "30%" in out
    assert load_config(config).steps == steps and "eval_every = 100000" in config.read_text(encoding="utf-8")
    record = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert (record["steps"], record["written"], record["intervals"]) == (steps, True, rate.intervals)
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    assert saved["steps"] == 100000 and saved["film"] is False  # the plan's schedule, no frames
    assert not (run_dir / "film").exists()


def test_calibrate_recomputes_from_a_finished_run_without_training(home, shards, tmp_path, capsys):
    config = tmp_path / "long.toml"
    config.write_text(TINY, encoding="utf-8")
    argv = ["train", "calibrate", "--config", str(config), "--steps", "600", "--data", str(shards)]
    assert cli.main([*argv, "--device", "cpu", "--run", "calib-t"]) == 0
    assert load_config(config).steps == 100000  # no --write: the config is untouched
    first = capsys.readouterr().out
    assert cli.main(["train", "calibrate", "--config", str(config), "--from-run", "calib-t"]) == 0
    again = capsys.readouterr().out
    result = [line for line in first.splitlines() if line.startswith(("R_true", "steps = floor"))]
    assert len(result) == 2 and all(line in again.splitlines() for line in result)


@pytest.mark.parametrize(
    "extra",
    [
        ["--steps", "500"],  # nothing left after the 500 steps R_true leaves out
        ["--resume"],
        ["--preview-steps", "5", "--from-step", "5"],
        ["--max-steps", "600"],
    ],
)
def test_calibrate_refuses_flags_that_do_not_fit_it(home, shards, tmp_path, extra):
    config = tmp_path / "long.toml"
    config.write_text(TINY, encoding="utf-8")
    argv = ["train", "calibrate", "--config", str(config), "--data", str(shards), "--device", "cpu"]
    assert cli.main([*argv, *extra]) == 2


def test_calibrate_only_flags_are_refused_by_a_plain_train(home, shards, tmp_path):
    config = tmp_path / "long.toml"
    config.write_text(TINY, encoding="utf-8")
    argv = ["train", "--config", str(config), "--run", "x", "--data", str(shards), "--device", "cpu"]
    assert cli.main([*argv, "--write"]) == 2
    assert cli.main(["train", "--config", str(config), "--data", str(shards)]) == 2  # no --run
    assert cli.main(["train", "--config", str(config), "--run", "x"]) == 2  # no data


@pytest.mark.cuda
def test_calibrate_runs_the_real_trainer_on_cuda(home, shards, tmp_path):
    config = tmp_path / "long.toml"
    config.write_text(TINY, encoding="utf-8")
    argv = ["train", "calibrate", "--config", str(config), "--steps", "600", "--data", str(shards)]
    assert cli.main([*argv, "--device", "cuda", "--run", "calib-cuda"]) == 0
    rows = _metrics(home / "runs" / "calib-cuda")
    assert rows[-1]["step"] == 600 and np.isfinite(calibrate.true_rate(rows, 16).samples_per_s)
