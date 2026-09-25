"""The P7 checks under P6 v2 (EVAL.md PR-2), end to end on a tiny CPU trainer.

The flagship's first leg and the size-m branch run with vaa_reference = "" (no reference); the P6 v2
guard then sets vaa_reference = "size-m" and the flagship resumes. size-m is a cooldown branch, so it
has no stable-phase rows: the 5% check is recorded as skipped (the guard replaced it), and the 30%
preview must beat size-m's final VAA.
"""

import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_train_config  # noqa: E402

from blink.model.config import config_from_dict  # noqa: E402
from blink.train import loop, preview, vaa  # noqa: E402
from blink.train.checkpoint import checkpoint_name, latest_checkpoint, load_checkpoint  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"
TOTAL = 400  # the flagship's planned steps: 5% = 20, 25% = 100, 30% = 120
RUNG = 20  # the "6 h" rung: 5% of the flagship, as 6 h is of 120 h
RUNG_COOLDOWN = 4  # round(0.2 x 20)
RUNG_START = RUNG - RUNG_COOLDOWN  # 16: leg 1 stops here, before the first check


def _spec(run_dir, **overrides) -> loop.RunSpec:
    return loop.RunSpec(**{"run_dir": run_dir, "world": WORLD, "device": "cpu", **overrides})


def _rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _repeat(batch):
    def source(start_step):
        while True:
            yield batch

    return source


def _flagship_config(reference: str = ""):
    return tiny_train_config(
        steps=TOTAL,
        warmup_steps=5,
        batch_size=16,
        eval_every=8,
        metrics_every=8,
        ckpt_every_steps=1000,
        vaa_checks=True,
        vaa_sigma=0.01,
        vaa_reference=reference,
    )


def _train(cfg, spec, logs: list[str] | None = None):
    probe = vaa.probe_from_roots(fixture_records()[:30])
    log = logs.append if logs is not None else (lambda _: None)
    return loop.train(cfg, spec, _repeat(fixture_records()[:16]), val=fixture_records(), probe=probe, log=log)


@pytest.fixture(scope="module")
def v2_runs(tmp_path_factory):
    """Leg 1 to the rung's cooldown start, the size-m branch, then the flagship resumed past its 30% check
    with vaa_reference = "size-m", and its 30% preview branch. Returns (runs root, flagship logs)."""
    runs = tmp_path_factory.mktemp("runs")
    leg1 = _flagship_config()
    assert min(vaa.check_steps(TOTAL)) > RUNG_START  # leg 1 ends before the first check
    _train(leg1, _spec(runs / "long", max_steps=RUNG_START))
    branch = preview.preview_config(leg1, from_step=RUNG_START, steps=RUNG_COOLDOWN)
    source = runs / "long" / checkpoint_name(RUNG_START)
    _train(branch, _spec(runs / "size-m", init_from=source, preview=True))
    logs: list[str] = []
    resumed = _flagship_config(reference="size-m")
    _train(resumed, _spec(runs / "long", resume=True, max_steps=120), logs)
    at_30 = runs / "long" / checkpoint_name(120)  # the preview takes its config from the checkpoint (CLI)
    thirty = preview.preview_config(
        config_from_dict(load_checkpoint(at_30)["config"]), from_step=120, steps=10
    )
    _train(thirty, _spec(runs / "long-preview", init_from=at_30, preview=True))
    return runs, logs


def test_leg_1_writes_no_check_row_and_the_branch_starts_from_its_last_checkpoint(v2_runs):
    runs, logs = v2_runs
    early = [r for r in _rows(runs / "long" / "evals.jsonl") if r["step"] <= RUNG_START]
    assert early and not any("check" in r for r in early)
    branched = json.loads((runs / "size-m" / "config.json").read_text(encoding="utf-8"))["branched_from"]
    assert branched.endswith(checkpoint_name(RUNG_START))
    assert any(f"resumed from {checkpoint_name(RUNG_START)}" in line for line in logs)


def test_the_size_m_branch_cools_down_to_the_rung_s_end_and_skips_its_check_without_a_reference(v2_runs):
    runs, _ = v2_runs
    rows = _rows(runs / "size-m" / "evals.jsonl")
    final = rows[-1]
    assert final["step"] == RUNG and final["check"] == "preview" and final["vaa_set"] == "full"
    assert "no reference" in final["check_skipped"] and "vaa_check_failed" not in final
    assert all(r["step"] > RUNG_START for r in rows)  # nothing from the stable phase
    ref = vaa.load_reference(runs / "size-m")
    assert ref.cooldown_start == RUNG_START and ref.final() == final["ema_vaa"]


def test_the_resumed_flagship_skips_the_5_percent_check_against_the_branch_and_logs_it_as_skipped(v2_runs):
    runs, logs = v2_runs
    checks = {r["check"]: r for r in _rows(runs / "long" / "evals.jsonl") if "check" in r}
    five = checks["5%"]
    assert five["step"] == 20 and "size-m" in five["check_skipped"] and "vaa_check_failed" not in five
    assert "check_rule" in checks["25%"]  # 25% meets the 5% row's own EMA VAA, as always
    assert any("check 5% skipped" in line and "size-m" in line for line in logs)
    assert not any("check 5% passed" in line for line in logs)


def test_the_30_percent_preview_is_judged_against_size_m_s_final_vaa(v2_runs):
    runs, _ = v2_runs
    final = _rows(runs / "long-preview" / "evals.jsonl")[-1]
    size_m = vaa.load_reference(runs / "size-m").final()
    assert final["check"] == "preview" and "check_skipped" not in final
    assert final["check_threshold"] == size_m and "size-m" in final["check_rule"]
    assert ("vaa_check_failed" in final) == (final["ema_vaa"] <= size_m)


def test_the_resume_takes_the_reference_the_toml_names_now_and_checkpoints_carry_it(v2_runs):
    runs, logs = v2_runs
    assert any("config differs" in line and "vaa_reference" in line for line in logs)
    assert load_checkpoint(latest_checkpoint(runs / "long"))["config"]["vaa_reference"] == "size-m"


def test_a_reference_that_does_not_exist_yet_refuses_the_start(tmp_path):
    """Setting vaa_reference before the branch exists fails at startup, not at the 5% check."""
    cfg = dataclasses.replace(_flagship_config(reference="size-m"), steps=40)
    with pytest.raises(FileNotFoundError):
        _train(cfg, _spec(tmp_path / "long", max_steps=2))


def test_an_empty_reference_skips_the_5_percent_check_and_the_preview_gate():
    row = vaa.apply_check("5%", 0.1, [], None, 20 * 16, sigma=0.01, n=30)
    assert row == {"check": "5%", "check_skipped": "no reference run"}
    end = vaa.apply_check("preview", 0.1, [], None, 0, sigma=0.01)
    assert "vaa_check_failed" not in end and "no reference" in end["check_skipped"]


def test_the_rung_s_numbers_at_the_analysis_rate_give_a_branch_reference_with_no_stable_phase(tmp_path):
    """size-m at 2,803.05 samples/s: 59,126 steps, cooling from 47,301 over 11,825 (the analysis)."""
    run = tmp_path / "size-m"
    run.mkdir()
    base = _flagship_config()
    cfg = preview.preview_config(
        dataclasses.replace(base, steps=1_105_734, warmup_steps=2000), 47_301, 11_825
    )
    assert (cfg.steps, cfg.cooldown_frac) == (59_126, 11_825 / 59_126)
    (run / "config.json").write_text(json.dumps({"config": dataclasses.asdict(cfg)}), encoding="utf-8")
    rows = [
        {"step": s, "ema_vaa": 0.5, "vaa_set": "subset", "vaa_n": 2000} for s in (48_000, 52_000, 56_000)
    ] + [{"step": 59_126, "ema_vaa": 0.58, "vaa_set": "full", "vaa_n": 20_000, "check": "preview"}]
    (run / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ref = vaa.load_reference(run)
    assert ref.cooldown_start == 47_301 and ref.final() == 0.58
    five = int(np.floor(0.05 * 1_105_734 + 0.5))
    for sigma_subset in (0.0, 0.004):
        row = vaa.apply_check(
            "5%", 0.40, [], ref, five * 1024, 0.003, subset=(2000, 0.40), sigma_subset=sigma_subset, n=20_000
        )
        assert "vaa_check_failed" not in row and "size-m has no stable-phase" in row["check_skipped"]
