"""PR-2 (3)'s guard for tools/p7_v2_driver.py, and the noise floor configs/long.toml's vaa_sigma must be.

sigma_EMA is the sample standard deviation of the final (100% check) full-valprobe EMA VAA of D's three
seeds, a01-a03, whose runs ablations.json names. The guard pauses the flagship when size-m's final EMA
VAA is more than 2 sigma_EMA below the seeds' mean; the flagship's own 25% and 50% checks read the same
floor from long.toml's vaa_sigma, so the driver refuses to start unless that value is sigma_EMA to its
printed precision. Stdlib only, like the driver.
"""

import json
import statistics
import time
from pathlib import Path
from typing import Any

from p7_machine import StepFailed, parse_number, read_json, read_rows, train_literal, write_atomic

TOLERANCE = 1e-9  # float slack for a pre-registered comparison (blink.train.nstar)
PAUSED_VAA = "paused: P7-VAA"  # blink.train.supervise's pause status: gate P7-VAA


def guard_verdict(branch_vaa: float, arm_vaas: list[float], sigma_factor: float = 2.0) -> dict[str, Any]:
    """PR-2 (3): Delta = the branch's final EMA VAA - the arms' mean; it fails when Delta < -2 sigma_EMA,
    sigma_EMA being the arms' sample standard deviation."""
    mean, sigma = statistics.fmean(arm_vaas), statistics.stdev(arm_vaas)
    delta, threshold = branch_vaa - mean, -sigma_factor * sigma
    return {"mean": mean, "sigma_ema": sigma, "delta": delta, "threshold": threshold,
            "passed": delta >= threshold - TOLERANCE}  # fmt: skip


def final_full_row(run_dir: Path) -> dict[str, Any]:
    """A run's full-valprobe EMA VAA row at its last planned step."""
    planned = int(read_json(run_dir / "config.json")["config"]["steps"])
    rows = [r for r in read_rows(run_dir / "evals.jsonl") if r.get("vaa_set") == "full" and "ema_vaa" in r]
    if not rows or rows[-1]["step"] != planned:
        raise ValueError(f"{run_dir.name} has no full-valprobe EMA VAA at its last step {planned:,}")
    return rows[-1]


def seed_rows(s) -> dict[str, dict[str, Any]]:
    """The seeds' final full-valprobe rows, from the runs ablations.json names for them."""
    arms = read_json(s.home / "eval" / "ablations.json").get("arms", {})
    rows = {}
    for name in s.arms:
        entry = arms.get(name) or {}
        if not str(entry.get("status", "")).startswith("finished") or not entry.get("run"):
            raise ValueError(f"arm {name} has not finished in ablations.json")
        rows[name] = final_full_row(s.runs / entry["run"])
    return rows


def check_sigma(s) -> dict[str, Any]:
    """long.toml's vaa_sigma against the guard's sigma_EMA, to the literal's printed precision."""
    try:
        rows = seed_rows(s)
    except (OSError, ValueError, KeyError) as exc:
        why = f"sigma_EMA cannot be read, so vaa_sigma cannot be checked: {exc}"
        raise StepFailed("preflight", why) from exc
    vaas = [rows[arm]["ema_vaa"] for arm in s.arms]
    sigma = statistics.stdev(vaas)
    literal = train_literal(s.config_path, "vaa_sigma")
    try:
        value, eps = parse_number(literal) if literal else (None, 0.0)
    except ValueError:
        value, eps = None, 0.0
    record = {"long_toml": value, "literal": literal, "sigma_ema": sigma,
              "seeds": {arm: rows[arm]["ema_vaa"] for arm in s.arms}}  # fmt: skip
    if value is None or abs(value - sigma) > eps + TOLERANCE:
        seeds = ", ".join(f"{arm} {rows[arm]['ema_vaa']}" for arm in s.arms)
        raise StepFailed(
            "preflight",
            f"{s.config} vaa_sigma is {literal or 'unset'}, not sigma_EMA {sigma:.7f} (the sample sd of the "
            f"100% check EMA VAA of {seeds}): set vaa_sigma = {sigma:.7f} before anything trains",
        )
    return record


def guard(s) -> dict[str, Any]:
    """The guard's inputs from ablations.json, the arms' evals and the branch's, and its verdict."""
    rows = seed_rows(s)
    branch = final_full_row(s.runs / s.branch)
    probes = {row.get("vaa_n") for row in (*rows.values(), branch)}
    if len(probes) != 1:
        raise ValueError(f"the final rows scored different valprobes ({sorted(map(str, probes))} roots)")
    verdict = guard_verdict(branch["ema_vaa"], [rows[a]["ema_vaa"] for a in s.arms], s.sigma_factor)
    picked = {a: {"step": r["step"], "ema_vaa": r["ema_vaa"]} for a, r in rows.items()}
    return {
        "rule": "PR-2 (3): Delta = size-m final EMA VAA - mean final EMA VAA of a01-a03; pause if "
        f"Delta < -{s.sigma_factor:g} sigma_EMA (their sample sd); not equal GPU-hours, no scaling law",
        "branch": {"run": s.branch, "step": branch["step"], "ema_vaa": branch["ema_vaa"]},
        "arms": picked,
        "vaa_n": branch.get("vaa_n"),
        **verdict,
    }


def mark_paused(run_dir: Path, detail: str) -> None:
    """Gate P7-VAA as the supervisor sets it: `blink status` then reports the run as paused."""
    path = run_dir / "heartbeat.json"
    beat = read_json(path) if path.is_file() else {}
    record = {**beat, "state": "paused", "stopped": PAUSED_VAA, "detail": detail, "time": time.time()}
    write_atomic(path, json.dumps(record))
