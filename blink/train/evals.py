"""One evals.jsonl row: validation metrics, VAA on the valprobe, and the P7 check at check steps.

Every `eval_every` steps the row holds the fixed validation sample's policy and value metrics (raw
and EMA) and VAA on the fixed first-`vaa_subset`-roots of the valprobe. At a check step (5, 25, 30,
50 and 100% of the planned steps) VAA runs on the full valprobe, and with `vaa_checks` the check's
rule is applied; a failed rule writes a `vaa_check_failed` record into the row, which the supervisor
reads. `eval_s` records the row's cost so the overhead can be measured.
"""

import time
from typing import Any

from blink.train import telemetry, vaa


def _val_metrics(run) -> dict[str, Any]:
    if run.val is None:
        return {}
    raw = telemetry.evaluate(run.model, run.val, run.cfg.alpha, run.cfg.tau)
    ema = telemetry.evaluate(run.ema.module, run.val, run.cfg.alpha, run.cfg.tau)
    return {**raw, **{f"ema_{k}": v for k, v in ema.items() if k != "n"}}


def _vaa_metrics(run, label: str | None) -> dict[str, Any]:
    if run.probe is None:
        return {}
    probe = run.probe if label else run.probe.subset(run.cfg.vaa_subset)
    raw = vaa.evaluate_vaa(run.model, probe, run.device)
    ema = vaa.evaluate_vaa(run.ema.module, probe, run.device)
    return {
        "vaa": raw["vaa"],
        "ema_vaa": ema["vaa"],
        "vaa_n": raw["n"],
        "vaa_set": "full" if label else "subset",
    }


def _check(run, label: str, record: dict[str, Any]) -> dict[str, Any]:
    if not run.cfg.vaa_checks:
        return {"check": label}
    return vaa.apply_check(
        label, record["ema_vaa"], run.check_history, run.reference, record["samples"], run.cfg.vaa_sigma
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
    if "vaa_check_failed" in record:
        parts.append(f"CHECK {record['check']} FAILED: {record['vaa_check_failed']}")
    elif "check" in record:
        parts.append(f"check {record['check']} passed")
    parts.append(f"{record['eval_s']:.1f} s")
    return " ".join(parts)


def evaluate(run, label: str | None = None) -> None:
    """Write one evals row for the current step (nothing when the run has neither val nor probe)."""
    started = time.perf_counter()
    metrics = {**_val_metrics(run), **_vaa_metrics(run, label)}
    if not metrics:
        return
    record = {"step": run.step, "samples": run.step * run.cfg.batch_size, **metrics}
    if label is not None:
        record.update(_check(run, label, record))
        run.check_history.append(record)
    record["eval_s"] = time.perf_counter() - started
    telemetry.append_jsonl(run.spec.run_dir / "evals.jsonl", record)
    run.last_eval = record
    run.log(_describe(record))
