"""One evals.jsonl row: validation metrics, VAA on the valprobe, and the P7 check at check steps.

Every `eval_every` steps the row holds the fixed validation sample's policy and value metrics (raw
and EMA) and the EMA's VAA on the fixed first `vaa_subset` roots of the valprobe (ema_vaa, vaa_set
"subset"). Only the EMA is scored there: every rule reads the EMA, and one model's forward passes over
2,000 roots' 57,275 children cost 1.6% of 2,000 training steps at S (4.48 s against 281 s) and about
1.2% at M, where scoring raw and EMA cost about twice that (plan target: 1%). At a check step
(5, 25, 30, 50 and 100% of the planned steps) VAA runs on the full valprobe for the raw weights (vaa)
and the EMA (ema_vaa), and the same passes give the EMA's subset VAA (ema_vaa_subset). The 5% rule
compares root set with root set, each with its own noise floor: full rows with vaa_sigma, and subset
rows only once vaa_sigma_subset has been measured. With `vaa_checks` the check's rule is applied; a
failed rule sets `vaa_check_failed` True in the row, which the supervisor reads. A check row also
scores games10k top-1 and the mateset's mate rates when the run has those files (blink.train.checksets:
arms a07 and a08 are judged on them); when scoring them fails the row is written without them and the
log says why. `eval_s` records the row's cost so the overhead can be measured.
"""

import time
from collections.abc import Callable
from typing import Any

from blink.train import checksets, telemetry, vaa

EVAL_ROWS_PER_TRAIN_ROW = 2


def _val_metrics(run) -> dict[str, Any]:
    if run.val is None:
        return {}
    raw = telemetry.evaluate(run.model, run.val, run.cfg.alpha, run.cfg.tau)
    ema = telemetry.evaluate(run.ema.module, run.val, run.cfg.alpha, run.cfg.tau)
    return {**raw, **{f"ema_{k}": v for k, v in ema.items() if k != "n"}}


def check_chunk(micro: int) -> int:
    """Rows per no-grad forward pass for a run training `micro` rows at a time (no-grad rows cost far
    less VRAM). Post-hoc scoring uses the run's own value, so its rows match the check rows exactly."""
    return min(vaa.VAA_CHUNK, EVAL_ROWS_PER_TRAIN_ROW * micro)


def _chunk(run) -> int:
    return check_chunk(run.micro)


def _vaa_metrics(run, label: str | None, tick: Callable[[], None] | None) -> dict[str, Any]:
    if run.probe is None:
        return {}
    chunk = _chunk(run)
    subset = run.cfg.vaa_subset
    if label is None:
        ema = vaa.evaluate_vaa(run.ema.module, run.probe.subset(subset), run.device, chunk, tick=tick)
        return {"ema_vaa": ema["vaa"], "vaa_n": ema["n"], "vaa_set": "subset"}
    raw = vaa.evaluate_vaa(run.model, run.probe, run.device, chunk, tick=tick)
    ema = vaa.evaluate_vaa(run.ema.module, run.probe, run.device, chunk, subset=subset, tick=tick)
    return {
        "vaa": raw["vaa"],
        "ema_vaa": ema["vaa"],
        "vaa_n": raw["n"],
        "vaa_set": "full",
        "ema_vaa_subset": ema["vaa_subset"],
        "vaa_subset_n": ema["n_subset"],
    }


def _check(run, label: str, record: dict[str, Any]) -> dict[str, Any]:
    if not run.cfg.vaa_checks:
        return {"check": label}
    subset = (record["vaa_subset_n"], record["ema_vaa_subset"]) if "ema_vaa_subset" in record else None
    return vaa.apply_check(
        label,
        record["ema_vaa"],
        run.check_history,
        run.reference,
        record["samples"],
        run.cfg.vaa_sigma,
        subset=subset,
        sigma_subset=run.cfg.vaa_sigma_subset,
        n=record["vaa_n"],
    )


def _describe(record: dict[str, Any]) -> str:
    parts = [f"eval {record['step']}:"]
    if "top1" in record:
        parts.append(
            f"top-1 {record['top1']:.3f} (ema {record['ema_top1']:.3f}), "
            f"policy CE {record['policy_ce']:.3f}, value CE {record['value_ce']:.3f}, "
            f"win% MAE {record['win_mae']:.4f}"
        )
    if "vaa" in record:
        parts.append(f"VAA {record['vaa']:.3f} (ema {record['ema_vaa']:.3f}, {record['vaa_set']})")
    elif "ema_vaa" in record:
        parts.append(f"VAA ema {record['ema_vaa']:.3f} ({record['vaa_set']} of {record['vaa_n']})")
    for key, name in (("games10k_top1", "games10k top-1"), ("shortest_mate", "shortest mate")):
        if key in record:
            parts.append(f"{name} {record[key]:.3f} (ema {record['ema_' + key]:.3f})")
    if "mate_preserving" in record:
        parts.append(f"mate kept {record['mate_preserving']:.3f} (ema {record['ema_mate_preserving']:.3f})")
    if "vaa_check_failed" in record:
        parts.append(f"CHECK {record['check']} FAILED: {record['check_failure']}")
    elif "check" in record:
        parts.append(f"check {record['check']} passed")
    parts.append(f"{record['eval_s']:.1f} s")
    return " ".join(parts)


def _check_set_metrics(run, tick: Callable[[], None] | None) -> dict[str, Any]:
    """The games10k and mateset keys, or none when scoring them fails: an evaluation input or result
    never ends a run. Weights gone NaN, say, must reach the supervisor as the next metrics row's
    non-finite loss (its NaN rollback), not as a crash in a check."""
    try:
        return checksets.metrics(run, _chunk(run), tick)
    except Exception as exc:  # a CUDA OOM, a set that no longer fits the model: logged, never fatal
        run.log(
            f"step {run.step}: games10k and mateset not scored at this check ({type(exc).__name__}: {exc})"
        )
        return {}


def evaluate(run, label: str | None = None, tick: Callable[[], None] | None = None) -> None:
    """Write one evals row for the current step (nothing when the run has neither val nor probe).

    `tick` runs between VAA chunks (the trainer beats its heartbeat there during a long check)."""
    started = time.perf_counter()
    metrics = {**_val_metrics(run), **_vaa_metrics(run, label, tick)}
    if not metrics:
        return
    if label is not None:
        metrics.update(_check_set_metrics(run, tick))
    record = {"step": run.step, "samples": run.step * run.cfg.batch_size, **metrics}
    if label is not None:
        record.update(_check(run, label, record))
        run.check_history.append(record)
    record["eval_s"] = time.perf_counter() - started
    telemetry.append_jsonl(run.spec.run_dir / "evals.jsonl", record)
    run.last_eval = record
    run.log(_describe(record))
