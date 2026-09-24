"""The Recipe D trainer end to end on CPU: phases, clip auto, children, LR scale, VAA, film, preview."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_mixed_training import children_from  # noqa: E402
from train_helpers import fixture_records, tiny_train_config  # noqa: E402

from blink.train import film, loop, preview, vaa  # noqa: E402
from blink.train.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint, step_of  # noqa: E402
from blink.train.source import InMemorySource, StepData, mixed_source  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"


def _spec(run_dir, **overrides) -> loop.RunSpec:
    return loop.RunSpec(**{"run_dir": run_dir, "world": WORLD, "device": "cpu", **overrides})


def _rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _repeat(batch):
    def source(start_step):
        while True:
            yield batch

    return source


def _mixed(seed: int = 3):
    records = fixture_records()
    roots = InMemorySource(records[:48], 12, seed=seed).batches
    children = InMemorySource(children_from(records[48:]), 4, seed=seed + 1).batches
    return mixed_source(roots, children, lambda r: (r["fen_hash"] % 7).astype(np.float32) / 3 + 0.2)


def _quiet(**kwargs):
    return {"val": fixture_records(), "log": lambda _: None, **kwargs}


def test_metrics_rows_carry_the_phase_of_their_window(tmp_path):
    cfg = tiny_train_config(steps=100, metrics_every=10, eval_every=40, ckpt_every_steps=50, batch_size=16)
    loop.train(cfg, _spec(tmp_path / "run"), _repeat(fixture_records()[:16]), **_quiet())
    phases = {row["step"]: row["phase"] for row in _rows(tmp_path / "run" / "metrics.jsonl")}
    assert phases[1] == phases[10] == phases[40] == "train"
    assert phases[50] == "eval" and phases[60] == "ckpt" and phases[90] == "eval"


def test_auto_clip_is_measured_over_the_warmup_then_fixed_and_logged(tmp_path):
    lines = []
    cfg = tiny_train_config(steps=40, warmup_steps=10, clip_norm="auto", metrics_every=5, batch_size=16)
    result = loop.train(
        cfg, _spec(tmp_path / "run"), _repeat(fixture_records()[:16]), **_quiet(log=lines.append)
    )
    rows = _rows(tmp_path / "run" / "metrics.jsonl")
    assert all(row["clip"] is None for row in rows if row["step"] < 10)
    assert all(row["clip_frac"] == 0.0 for row in rows if row["step"] <= 10)  # the warmup is never clipped
    assert all(row["clip"] == result.clip for row in rows if row["step"] >= 10)
    assert result.clip > 0 and any(line.startswith("clip auto: ") for line in lines)
    assert load_checkpoint(latest_checkpoint(tmp_path / "run"))["clip"]["value"] == result.clip


def _final_state(cfg, run_dir, source, **spec_overrides):
    loop.train(cfg, _spec(run_dir, **spec_overrides), source, **_quiet())
    return load_checkpoint(latest_checkpoint(run_dir))


def _assert_same_weights(a, b) -> None:
    for key in ("model", "ema"):
        for name, tensor in a[key].items():
            assert torch.equal(tensor, b[key][name]), f"{key}.{name} differs after resume"


def test_a_mixed_run_with_micro_batches_and_auto_clip_resumes_bitwise_mid_warmup(tmp_path):
    cfg = tiny_train_config(
        steps=40, warmup_steps=20, clip_norm="auto", batch_size=16, child_frac=0.25, micro_batch=8
    )
    straight = _final_state(cfg, tmp_path / "straight", _mixed())
    _final_state(cfg, tmp_path / "split", _mixed(), max_steps=10)
    resumed = _final_state(cfg, tmp_path / "split", _mixed(), resume=True)
    assert resumed["step"] == straight["step"] == 40
    assert resumed["clip"] == straight["clip"] and resumed["clip"]["value"] is not None
    _assert_same_weights(straight, resumed)


def test_lr_scale_on_resume_scales_the_schedule_and_is_kept_in_later_checkpoints(tmp_path):
    cfg = tiny_train_config(steps=60, warmup_steps=2, cooldown_frac=0.1, metrics_every=10, batch_size=16)
    run_dir, source = tmp_path / "run", _repeat(fixture_records()[:16])
    with pytest.raises(ValueError, match="lr_scale"):
        loop.train(cfg, _spec(tmp_path / "fresh", lr_scale=0.5), source, **_quiet())
    loop.train(cfg, _spec(run_dir, max_steps=20), source, **_quiet())
    loop.train(cfg, _spec(run_dir, resume=True, lr_scale=0.5, max_steps=40), source, **_quiet())
    loop.train(cfg, _spec(run_dir, resume=True, max_steps=50), source, **_quiet())
    lrs = {row["step"]: row["lr"] for row in _rows(run_dir / "metrics.jsonl")}
    assert lrs[20] == pytest.approx(1e-3) and lrs[30] == lrs[40] == lrs[50] == pytest.approx(5e-4)
    assert load_checkpoint(latest_checkpoint(run_dir))["lr_scale"] == 0.5


def test_lr_scale_sets_the_scale_so_a_restart_that_repeats_it_never_compounds(tmp_path):
    """The supervisor re-sends --lr-scale 0.5 on every restart after its NaN rollback: 0.5 stays 0.5."""
    cfg = tiny_train_config(steps=80, warmup_steps=2, cooldown_frac=0.1, metrics_every=10, batch_size=16)
    run_dir, source = tmp_path / "run", _repeat(fixture_records()[:16])
    loop.train(cfg, _spec(run_dir, max_steps=20), source, **_quiet())
    for end in (40, 60):
        loop.train(cfg, _spec(run_dir, resume=True, lr_scale=0.5, max_steps=end), source, **_quiet())
    loop.train(cfg, _spec(run_dir, resume=True, lr_scale=1.0, max_steps=70), source, **_quiet())
    lrs = {row["step"]: row["lr"] for row in _rows(run_dir / "metrics.jsonl")}
    assert lrs[30] == lrs[50] == lrs[60] == pytest.approx(5e-4)
    assert lrs[70] == pytest.approx(1e-3)


def test_subset_evals_score_the_ema_and_the_checks_score_raw_and_ema_on_the_full_probe(tmp_path):
    """The 2k-step evals score one model on the subset (half the cost of two); every rule reads the EMA."""
    probe = vaa.probe_from_roots(fixture_records()[:40])
    cfg = tiny_train_config(steps=40, eval_every=15, vaa_subset=10, keep_last=1, batch_size=16)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir), _repeat(fixture_records()[:16]), **_quiet(probe=probe))
    rows = {row["step"]: row for row in _rows(run_dir / "evals.jsonl")}
    assert sorted(rows) == [0, 2, 10, 12, 15, 20, 30, 40]
    checks = {step: row["check"] for step, row in rows.items() if "check" in row}
    assert checks == {2: "5%", 10: "25%", 12: "30%", 20: "50%", 40: "100%"}
    for step in (0, 15, 30):
        assert (rows[step]["vaa_set"], rows[step]["vaa_n"]) == ("subset", 10) and "vaa" not in rows[step]
        assert 0.0 <= rows[step]["ema_vaa"] <= 1.0
    for step in checks:
        assert (rows[step]["vaa_set"], rows[step]["vaa_n"], rows[step]["vaa_subset_n"]) == ("full", 40, 10)
        assert all(0.0 <= rows[step][key] <= 1.0 for key in ("vaa", "ema_vaa", "ema_vaa_subset"))
    assert [step_of(p) for p in list_checkpoints(run_dir)] == [2, 10, 12, 20, 40]


def test_the_heartbeat_keeps_beating_inside_an_eval(tmp_path, monkeypatch):
    beats = []
    real = loop.heartbeat.beat_once
    monkeypatch.setattr(
        loop.heartbeat, "beat_once", lambda path, payload: beats.append(payload) or real(path, payload)
    )
    probe = vaa.probe_from_roots(fixture_records()[:40])
    cfg = tiny_train_config(steps=4, warmup_steps=1, eval_every=4, heartbeat_s=0.0, batch_size=16)
    loop.train(cfg, _spec(tmp_path / "run"), _repeat(fixture_records()[:16]), **_quiet(probe=probe))
    during = [b for b in beats if b.get("phase") == "eval"]
    assert during and all(b["state"] == "running" for b in during)
    assert {b["step"] for b in during} >= {0, 4}


def _reference_run(runs_root, name: str, ema_vaa: float, roots: int) -> None:
    """A finished sweep run: full-valprobe check rows, and a subset row that the 5% rule must ignore."""
    ref = runs_root / name
    ref.mkdir(parents=True)
    config = {"config": dataclasses.asdict(tiny_train_config(steps=1000, batch_size=16))}
    (ref / "config.json").write_text(json.dumps(config), encoding="utf-8")
    subset = {"step": 50, "ema_vaa": 1.0 - ema_vaa, "vaa_set": "subset", "vaa_n": 10}
    rows = [subset] + [{"step": s, "ema_vaa": ema_vaa, "vaa_set": "full", "vaa_n": roots} for s in (100, 200)]
    (ref / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize(("reference_vaa", "fails"), [(1.0, True), (0.0, False)])
def test_the_5_percent_check_writes_vaa_check_failed_when_the_run_falls_behind(
    tmp_path, reference_vaa, fails
):
    probe = vaa.probe_from_roots(fixture_records()[:30])
    _reference_run(tmp_path, "s6h", reference_vaa, roots=int((np.diff(probe.child_offset) > 0).sum()))
    cfg = tiny_train_config(
        steps=40, eval_every=40, vaa_checks=True, vaa_sigma=0.0, vaa_reference="s6h", batch_size=16
    )
    loop.train(
        cfg, _spec(tmp_path / "long", max_steps=3), _repeat(fixture_records()[:16]), **_quiet(probe=probe)
    )
    row = next(r for r in _rows(tmp_path / "long" / "evals.jsonl") if r.get("check") == "5%")
    assert ("vaa_check_failed" in row) == fails
    if fails:
        assert row["vaa_check_failed"] is True and row["check_failure"]["reference"] == "s6h"


def test_film_frames_are_saved_at_the_planned_steps_in_one_world(tmp_path):
    cfg = tiny_train_config(steps=300, film=True, eval_every=300, batch_size=16, ckpt_every_steps=300)
    run_dir = tmp_path / "run"
    loop.train(cfg, _spec(run_dir), _repeat(fixture_records()[:16]), **_quiet())
    frames = film.list_frames(run_dir)
    assert [film.step_of(p) for p in frames] == [step for step, _ in film.frame_plan(300)]
    assert len(frames) == 21 and film.frames_world(frames) == WORLD
    final = torch.load(frames[-1], weights_only=True)
    assert final["kind"] == "final" and "raw" in final and final["samples"] == 300 * 16


def test_one_checkpoint_per_keep_every_hours_survives_pruning(tmp_path):
    cfg = tiny_train_config(steps=30, ckpt_every_steps=10, keep_last=1, keep_every_hours=1e-9, batch_size=16)
    loop.train(cfg, _spec(tmp_path / "run"), _repeat(fixture_records()[:16]), **_quiet())
    assert [step_of(p) for p in list_checkpoints(tmp_path / "run")] == [10, 20, 30]


def _digest(run_dir) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run_dir.iterdir()) if p.is_file()
    }


def test_a_preview_branch_cools_down_from_the_checkpoint_without_touching_the_main_run(tmp_path):
    cfg = tiny_train_config(steps=100, warmup_steps=5, ckpt_every_steps=30, metrics_every=10, batch_size=16)
    main, source = tmp_path / "long", _repeat(fixture_records()[:16])
    loop.train(cfg, _spec(main, max_steps=30), source, **_quiet())
    before = _digest(main)
    branch_cfg = preview.preview_config(cfg, from_step=30, steps=20)
    spec = _spec(tmp_path / "long-preview", init_from=main / "ckpt_000000030.pt", preview=True)
    result = loop.train(branch_cfg, spec, source, **_quiet())
    assert result.step == 50 and _digest(main) == before
    lrs = {row["step"]: row["lr"] for row in _rows(tmp_path / "long-preview" / "metrics.jsonl")}
    assert min(lrs) == 40 and lrs[40] < 1e-3 and lrs[50] == 0.0
    saved = json.loads((tmp_path / "long-preview" / "config.json").read_text(encoding="utf-8"))
    assert saved["branched_from"].endswith("ckpt_000000030.pt") and saved["world"] == WORLD


def test_a_step_data_source_trains_children_with_the_roots(tmp_path):
    cfg = tiny_train_config(steps=10, warmup_steps=2, batch_size=16, child_frac=0.25, metrics_every=5)
    seen = []

    def spy(start_step):
        for step in _mixed()(start_step):
            seen.append(step)
            yield step

    loop.train(cfg, _spec(tmp_path / "run"), spy, **_quiet())
    assert all(isinstance(s, StepData) and (len(s.roots), len(s.children)) == (12, 4) for s in seen)
    assert all(np.isfinite(r["loss_value"]) for r in _rows(tmp_path / "run" / "metrics.jsonl"))


@pytest.mark.cuda
def test_the_full_recipe_runs_on_cuda_with_a_measured_micro_batch(tmp_path):
    from train_helpers import tiny_model_config

    cfg = tiny_train_config(
        model=tiny_model_config(gab=True),
        steps=30,
        warmup_steps=10,
        batch_size=16,
        child_frac=0.25,
        micro_batch="auto",
        clip_norm="auto",
    )
    probe = vaa.probe_from_roots(fixture_records()[:20])
    result = loop.train(cfg, _spec(tmp_path / "gpu", device="cuda"), _mixed(), **_quiet(probe=probe))
    saved = json.loads((tmp_path / "gpu" / "config.json").read_text(encoding="utf-8"))
    assert saved["vram"]["budget_gb"] > 0 and saved["vram"]["micro_batch"] == result.micro_batch == 16
    assert result.clip is not None and "vaa" in result.last_eval
    assert saved["parameter_report"]["gab"] > 0
