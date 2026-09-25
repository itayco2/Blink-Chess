"""`blink sweep ablations`: the recipe ablations (plan P5), and the run plumbing the size sweep shares.

Ablations. Each arm is a TOML file of overrides on the recipe (configs/ablations/aNN.toml: an [arm]
table for the sweep plus [model] and [train] overrides). Arms run one after another, each through
`blink supervise`, for a fixed wall-clock budget: its steps are the hours times the size's measured
samples/s (bench.json) over the batch size, so the WSD schedule's own 20% cooldown ends the arm.
An arm whose change costs time per step (a10's Muon) names its own bench.json row ([arm] bench_size),
which then plans its steps and polices its throughput; a15 runs at the slowest rate of its winners.
The sweep is resumable: ablations.json records every arm, and a rerun skips finished arms and
resumes the one that was running. One sample budget holds for the whole sweep: the plan size's rate is
pinned in ablations.json when the sweep first plans (for a sweep that did not record it, the rate that
plans a01-a03's recorded steps), and an arm stopped partway resumes at the steps and rate it started
with, so re-measuring a bench row partway through moves no arm's schedule. Judging, after every arm
has cooled down:
- D and sigma are the mean and sample standard deviation of arms a01-a03 (Recipe D, seeds 1-3);
- adopt an arm when its metric >= D + 2 sigma and its policy top-1 >= D_top1 - 2 sigma_top1
  (a08 also must not lose more than 2 pt of mate preservation);
- a15 runs the adopted arms combined; the recipe becomes D plus them only if a15 >= D - 1 sigma.
VAA and top-1 are fractions in [0, 1], as evals.jsonl writes top1; 1 pt is 0.01. An arm's metrics are
its last evals row plus, for an arm trained before the checks scored games10k and the mateset, the
post-hoc record `blink sweep rescore` writes (blink.train.posthoc), read afresh whenever arms are judged.
The adopted set must be settled before a15 trains, since a15's VAA vouches only for what it trained
with: before a15 starts the sweep scores a01-a03 post hoc for any metric a finished arm holds and they
lack, and holds a15 if that fails; a15 stopped partway resumes with the arms it started with; and the
recipe stays D when the adopted set no longer matches the arms a15 recorded. Only the sweep writes
ablations.json; `blink sweep rescore` writes posthoc.json files only. A user pause (BLINK_HOME/PAUSE,
blink.train.userpause) holds an arm not yet started before it is planned, and pauses a running arm
inside its supervisor, so the arm stays "running" and continues once the flag is gone.

The size sweep (plan P6) is blink.train.size_sweep; N* is blink.train.nstar.
"""

import dataclasses
import json
import math
import time
import tomllib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blink import paths
from blink.model.config import compile_mode, config_from_dict, read_tables
from blink.train import posthoc, userpause
from blink.train.atomic import write_text_atomic
from blink.train.nstar import TOLERANCE
from blink.train.supervise import Outcome, checkpoint_steps, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
RERUN_STATES = ("pending", "running", "held")  # held: a15 waiting for arms it cannot judge yet
NO_WINNERS = "not run: no arm was adopted"  # a combined arm plans again once a rescore adopts one
SLIP_CUT = "not tested: cut by the slip rule"
DEADLINE_FACTOR = 1.5  # a supervised arm is stopped at 1.5x its planned hours plus the allowance
COMPILE_ALLOWANCE_S = 1800.0
TABLES = ("model", "train")
# the final check row's metrics an arm's entry repeats: what the arms are judged on (a07 games10k,
# a08 its mate_preserving guard) and what FINDINGS reports beside them
PICKED = ("vaa", "top1", "value_ce", "games10k_top1", "mate_preserving", "shortest_mate")
# what post-hoc scoring can raise without stopping a sweep: posthoc.NotFinished is a ValueError, a
# missing input an OSError, a CUDA out-of-memory a RuntimeError
SCORE_ERRORS = (ValueError, OSError, RuntimeError)

Log = Callable[[str], None]


@dataclass(frozen=True)
class Arm:
    name: str
    change: str
    judged_on: str = "vaa"
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_lr_scale: float = 1.0
    combine: bool = False  # a15: run the adopted arms together
    guard: str | None = None  # a08: a metric that must not fall more than 2 pt
    combined_from: tuple[str, ...] = ()
    bench_size: str | None = None  # its own bench.json throughput row; None: the plan's size


@dataclass(frozen=True)
class AblationPlan:
    recipe: Path
    data: Path
    hours: float
    size: str
    sigma_arms: tuple[str, ...]
    arms: tuple[Arm, ...]
    slip_cut: tuple[str, ...] = ()
    disable: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunRequest:
    run: str
    config: Path
    data: Path
    resume: bool
    bench_rate: float | None
    deadline_s: float
    disable: tuple[str, ...] = ()


Runner = Callable[[RunRequest], Outcome]


# ---------------------------------------------------------------- files


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def load_arm(path: Path) -> Arm:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(data) - {"arm", *TABLES})
    if unknown:
        raise ValueError(f"{path}: unknown tables {unknown}; an arm has [arm], [model] and [train]")
    meta = data.get("arm", {})
    return Arm(
        name=Path(path).stem,
        change=str(meta.get("change", "")),
        judged_on=str(meta.get("judged_on", "vaa")),
        overrides={t: dict(data[t]) for t in TABLES if t in data},
        peak_lr_scale=float(meta.get("peak_lr_scale", 1.0)),
        combine=bool(meta.get("combine", False)),
        guard=meta.get("guard"),
        bench_size=None if meta.get("bench_size") is None else str(meta["bench_size"]),
    )


def load_plan(path: Path) -> AblationPlan:
    path = Path(path)
    plan = tomllib.loads(path.read_text(encoding="utf-8"))["plan"]
    return AblationPlan(
        recipe=_resolve(plan["recipe"], REPO_ROOT),
        data=_resolve(plan["data"], paths.home() / "data"),
        hours=float(plan["hours"]),
        size=str(plan["size"]),
        sigma_arms=tuple(plan["sigma_arms"]),
        arms=tuple(load_arm(path.parent / f"{name}.toml") for name in plan["arms"]),
        slip_cut=tuple(plan.get("slip_cut", ())),
        disable=tuple(plan.get("disable", ())),
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} as a TOML value")


def write_config(config: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    lines = []
    for table in TABLES:
        lines.append(f"[{table}]")
        lines += [f"{key} = {_toml_value(value)}" for key, value in config.get(table, {}).items()]
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(Path(path), "\n".join(lines))


def steps_for(hours: float, samples_per_s: float, batch_size: int) -> int:
    """Optimizer steps that fill `hours` at the measured rate; the WSD cooldown is their last 20%."""
    return int(hours * 3600 * samples_per_s // batch_size)


def _rate_of_steps(steps: int, hours: float, batch_size: int) -> float:
    """The lowest samples/s that steps_for turns into `steps` (float rounding included)."""
    rate = steps * batch_size / (hours * 3600)
    while steps_for(hours, rate, batch_size) < steps:
        rate = math.nextafter(rate, math.inf)
    return rate


def merged_config(base: Mapping[str, Any], arm: Arm, steps: int) -> dict[str, dict[str, Any]]:
    """The recipe with the arm's overrides, its learning-rate scale and the clock's step count."""
    model = {**base.get("model", {}), **arm.overrides.get("model", {})}
    train = {**base.get("train", {}), **arm.overrides.get("train", {})}
    peak = float(train.get("peak_lr", 1e-3)) * arm.peak_lr_scale
    train = {**train, "steps": steps, "peak_lr": peak}
    warmup = int(train.get("warmup_steps", 0))
    if warmup >= steps:
        raise ValueError(f"warmup_steps {warmup} >= steps {steps}: the arm's clock budget is too short")
    return {"model": model, "train": train}


def validate(config: Mapping[str, Mapping[str, Any]]) -> None:
    """The trainer's own config check: unknown keys and bad values fail here, before any GPU time."""
    config_from_dict({**config.get("train", {}), "model": dict(config.get("model", {}))})


def final_metrics(run_dir: Path) -> dict[str, Any] | None:
    """The run's last evals row, with its post-hoc record (posthoc.json) merged when of the same step."""
    rows = read_jsonl(Path(run_dir) / "evals.jsonl")
    return posthoc.merge(rows[-1], posthoc.read(run_dir)) if rows else None


def _load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(Path(path), json.dumps({**state, "updated": time.time()}, indent=2, default=str) + "\n")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------- judging the arms


def _numbers(row: Mapping[str, Any]) -> set[str]:
    skip = {"step", "n"}
    return {
        k for k, v in row.items() if k not in skip and isinstance(v, int | float) and not isinstance(v, bool)
    }


def noise_floor(results: Mapping[str, Mapping[str, Any]], sigma_arms: Sequence[str]) -> dict[str, Any] | None:
    """D and sigma of every metric a01-a03 share: their mean and sample standard deviation."""
    rows = [results.get(name) for name in sigma_arms]
    if len(rows) < 2 or any(not row for row in rows):
        return None
    shared = set.intersection(*(_numbers(row) for row in rows))
    if "vaa" not in shared:
        return None
    floor: dict[str, Any] = {}
    for key in sorted(shared):
        values = [float(row[key]) for row in rows]
        mean = sum(values) / len(values)
        sigma = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        floor[key] = {"d": mean, "sigma": sigma}
    return {**floor, "arms": list(sigma_arms), "sigma_ok": floor["vaa"]["sigma"] <= 0.01 + TOLERANCE}


def _needed(arm: Arm) -> list[str]:
    """The metrics an arm's adopt rule reads: its own, policy top-1, and its guard if it has one."""
    return [arm.judged_on, "top1"] + ([arm.guard] if arm.guard else [])


def decide(arm: Arm, metrics: Mapping[str, Any], floor: Mapping[str, Any] | None) -> dict[str, Any]:
    """The pre-registered adopt rule for one arm."""
    metric = arm.judged_on
    if floor is None:
        return {"adopt": False, "reason": "not judged: the noise floor needs a01-a03 finished"}
    missing = [k for k in _needed(arm) if metrics.get(k) is None or k not in floor]
    if missing:
        return {"adopt": False, "reason": f"not judged: {', '.join(missing)} missing"}
    bar = floor[metric]["d"] + 2 * floor[metric]["sigma"]
    top_bar = floor["top1"]["d"] - 2 * floor["top1"]["sigma"]
    failed = []
    if metrics[metric] < bar - TOLERANCE:
        failed.append(f"{metric} {metrics[metric]:.4f} < D + 2 sigma {bar:.4f}")
    if metrics["top1"] < top_bar - TOLERANCE:
        failed.append(f"top1 {metrics['top1']:.4f} < D - 2 sigma {top_bar:.4f}")
    if arm.guard and metrics[arm.guard] < floor[arm.guard]["d"] - 0.02 - TOLERANCE:
        base = floor[arm.guard]["d"]
        failed.append(f"{arm.guard} {metrics[arm.guard]:.4f} lost more than 2 pt from {base:.4f}")
    reason = "; ".join(failed) if failed else f"{metric} {metrics[metric]:.4f} >= D + 2 sigma {bar:.4f}"
    return {"adopt": not failed, "reason": reason, "metric": metric, "value": metrics[metric], "bar": bar}


def combine_adopted(arms: Mapping[str, Arm], decisions: Mapping[str, Mapping], name: str = "a15") -> Arm:
    """One arm holding every adopted arm's overrides and learning-rate scale, in plan order."""
    chosen = [arm for arm_name, arm in arms.items() if decisions.get(arm_name, {}).get("adopt")]
    overrides: dict[str, dict[str, Any]] = {}
    for arm in chosen:
        for table, values in arm.overrides.items():
            overrides[table] = {**overrides.get(table, {}), **values}
    names = tuple(arm.name for arm in chosen)
    change = "combined winners: " + (", ".join(names) or "none")
    scale = math.prod(arm.peak_lr_scale for arm in chosen)
    return Arm(name, change, "vaa", overrides, scale, combine=True, combined_from=names)


def recipe_verdict(metrics: Mapping[str, Any] | None, floor: Mapping[str, Any] | None, combined: Arm) -> dict:
    """The frozen recipe: D plus the winners only if their combination stays >= D - 1 sigma."""
    if not combined.combined_from:
        return {"recipe": "D", "reason": "no arm was adopted"}
    if floor is None or not metrics or metrics.get("vaa") is None:
        return {"recipe": "D", "reason": f"{combined.name} was not judged"}
    bar = floor["vaa"]["d"] - floor["vaa"]["sigma"]
    vaa = metrics["vaa"]
    if vaa < bar - TOLERANCE:
        return {"recipe": "D", "reason": f"{combined.name} VAA {vaa:.4f} < D - 1 sigma {bar:.4f}"}
    return {
        "recipe": "D + " + " + ".join(combined.combined_from),
        "reason": f"{combined.name} VAA {vaa:.4f} >= D - 1 sigma {bar:.4f}",
        "overrides": combined.overrides,
        "peak_lr_scale": combined.peak_lr_scale,
    }


# ---------------------------------------------------------------- running arms and sizes


def batch_size_of(base: Mapping[str, Any], arm: Arm) -> int:
    from blink.model.config import TrainConfig

    train = {**base.get("train", {}), **arm.overrides.get("train", {})}
    return int(train.get("batch_size", TrainConfig().batch_size))


def _prepare(
    run: str, base_path: Path, arm: Arm, hours: float, rate: float, folder: str, steps: int | None = None
) -> tuple[Path, dict]:
    """Write the run's merged config under BLINK_HOME/eval/<folder>/ and describe it (`steps`, or the
    clock's at `rate`)."""
    base = read_tables(base_path)  # follows `base = "recipe.toml"` (PF66)
    steps = steps if steps is not None else steps_for(hours, rate, batch_size_of(base, arm))
    config = merged_config(base, arm, steps)
    validate(config)
    path = paths.home() / "eval" / folder / f"{run}.toml"
    write_config(config, path)
    return path, {"run": run, "config": str(path), "steps": steps, "peak_lr": config["train"]["peak_lr"]}


def _request(run: str, config: Path, data: Path, rate: float, hours: float, disable) -> RunRequest:
    resume = bool(checkpoint_steps(paths.home() / "runs" / run))
    deadline = hours * 3600 * DEADLINE_FACTOR + COMPILE_ALLOWANCE_S
    return RunRequest(run, config, data, resume, rate, deadline, tuple(disable))


def _finish_entry(entry: Mapping[str, Any], outcome: Outcome, run: str) -> dict[str, Any]:
    metrics = final_metrics(paths.home() / "runs" / run) or {}
    picked = {k: metrics.get(k) for k in PICKED}
    return {**entry, **picked, "status": outcome.status, "metrics": metrics, "finished": time.time()}


def _refreshed(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A finished arm's entry with its run's post-hoc record merged in, which `blink sweep rescore` may
    have written since the arm finished (even while this sweep runs other arms)."""
    if entry.get("status") != "finished" or not entry.get("run"):
        return dict(entry)
    record = posthoc.read(paths.home() / "runs" / entry["run"])
    metrics = posthoc.merge(entry.get("metrics") or {}, record)
    return {**entry, **{k: metrics.get(k) for k in PICKED}, "metrics": metrics}


def _execute(entry: dict[str, Any], request: RunRequest | None, runner: Runner, save, log: Log) -> dict:
    """Record the entry as running, run it, then record how it ended (saved before and after)."""
    save(entry)
    if request is None:
        reason = f" ({entry['reason']})" if entry.get("reason") else ""
        log(f"{entry.get('name')}: {entry['status']}{reason}")
        return entry
    log(f"{entry.get('name')}: {entry['steps']:,} steps, run {request.run}")
    done = _finish_entry(entry, runner(request), request.run)
    save(done)
    log(f"{entry.get('name')}: {done['status']}, VAA {done['vaa']}")
    return done


# ---------------------------------------------------------------- the ablation sweep


def _run_of(name: str, entry: Mapping[str, Any]) -> str:
    return entry.get("run") or f"abl-{name}"


def _finished_metrics(arms_state: Mapping[str, Mapping]) -> dict[str, dict[str, Any]]:
    """Each finished arm's final metrics, with any post-hoc record written since it finished."""
    return {n: _refreshed(e)["metrics"] for n, e in arms_state.items() if e.get("status") == "finished"}


def _judge(plan: AblationPlan, arms_state: Mapping[str, Mapping]) -> tuple[dict | None, dict[str, dict]]:
    finished = _finished_metrics(arms_state)
    floor = noise_floor(finished, plan.sigma_arms)
    decisions = {}
    for arm in plan.arms:
        if arm.name in plan.sigma_arms or arm.combine:
            continue
        status = arms_state.get(arm.name, {}).get("status", "not run")
        if status == "finished":
            decisions[arm.name] = decide(arm, finished[arm.name], floor)
        else:
            decisions[arm.name] = {"adopt": False, "reason": f"not judged: {status}"}
    return floor, decisions


def arm_compile_mode(plan: AblationPlan, arm: Arm) -> str:
    """The compile mode an arm trains in: the recipe's (with base) under the arm's own overrides."""
    tables = read_tables(plan.recipe)
    return compile_mode({"train": {**tables["train"], **arm.overrides.get("train", {})}})


def own_bench_rates(
    plan: AblationPlan, rate_of: Callable[[str, str], float], arms: Collection[str] | None = None
) -> dict[str, float]:
    """samples/s by arm name for every arm (of `arms`; all by default) that names its own bench row,
    from rate_of(size, mode)."""
    rates = {}
    for arm in plan.arms:
        if arm.bench_size is None or (arms is not None and arm.name not in arms):
            continue
        try:
            rates[arm.name] = rate_of(arm.bench_size, arm_compile_mode(plan, arm))
        except ValueError as exc:
            raise ValueError(
                f"arm {arm.name} is planned at its own bench row {arm.bench_size}: {exc}"
            ) from exc
    return rates


def arm_rate(arm: Arm, rate: float, rates: Mapping[str, float]) -> float:
    """The samples/s that plans an arm's steps and polices its throughput.

    Its own bench row's when it has one, else the plan size's. Planned at D's rate, an arm that is
    slower per step (a10's Muon) would run past its hours and trip the throughput stop rule before
    its cooldown. A combined arm takes the slowest rate among the winners it combines: an estimate
    for a mix no bench row measured, erring towards finishing early rather than overrunning.
    """
    if arm.combine:
        return min([rate, *(rates[name] for name in arm.combined_from if name in rates)])
    return rates.get(arm.name, rate)


def _recorded_rates(plan: AblationPlan, arms_state: Mapping[str, Mapping]) -> dict[str, float]:
    """The own-bench-row rates finished arms trained at, which a15 still plans from (no re-bench)."""
    done = {n: e for n, e in arms_state.items() if e.get("status") == "finished" and e.get("samples_per_s")}
    return {
        a.name: float(done[a.name]["samples_per_s"]) for a in plan.arms if a.bench_size and a.name in done
    }


def plan_rate(plan: AblationPlan, state: Mapping[str, Any], rate: float) -> float:
    """The plan size's samples/s for the whole sweep: ablations.json's pinned `rate`; for a sweep that
    did not record one (the frozen PF66 commit), a01-a03's own, or the rate that plans their recorded
    steps; else `rate`, the bench row read now. Re-benching the size partway through then cannot give
    a later arm (a06 after a01-a03) another sample budget than the floor it is judged against."""
    if state.get("rate"):
        return float(state["rate"])
    arms, base = state.get("arms", {}), read_tables(plan.recipe)
    for name in plan.sigma_arms:
        entry = arms.get(name, {})
        if entry.get("samples_per_s"):
            return float(entry["samples_per_s"])
        if entry.get("steps"):
            return _rate_of_steps(int(entry["steps"]), plan.hours, batch_size_of(base, Arm(name, "")))
    return rate


def _floor_gaps(plan: AblationPlan, finished: Mapping[str, Mapping]) -> dict[str, list[str]]:
    """{arm: metrics} for each finished arm whose own final row holds a metric its adopt rule reads but
    the a01-a03 noise floor lacks: the seeds predate it, and scoring them post hoc judges the arm.

    A metric the arm itself lacks is no gap (a08's mate_preserving, which today's mateset cannot score):
    no scoring of the seeds can judge that arm, and a15 goes ahead without it as before."""
    floor = noise_floor(finished, plan.sigma_arms)
    if floor is None:
        return {}
    gaps = {}
    for arm in plan.arms:
        metrics = finished.get(arm.name)
        if metrics is None or arm.name in plan.sigma_arms or arm.combine:
            continue
        lacking = [k for k in _needed(arm) if metrics.get(k) is not None and k not in floor]
        if lacking:
            gaps[arm.name] = lacking
    return gaps


def _score_arms(runs: Mapping[str, str], scorer: Callable[[str], Any], log: Log) -> dict[str, str]:
    """Score each arm's run post hoc; {arm: why} for each that could not be (logged and skipped)."""
    failed = {}
    for name, run in runs.items():
        try:
            scorer(run)
        except SCORE_ERRORS as exc:
            log(f"{name}: not rescored ({exc})")
            failed[name] = str(exc)
    return failed


def _settle_floor(plan: AblationPlan, arms_state: Mapping, scorer, log: Log) -> dict[str, list[str]]:
    """The floor gaps left after scoring post hoc each seed arm that lacks a gap's metric (the sweep's
    GPU is idle between arms, and score_run keeps a record it already has)."""
    finished = _finished_metrics(arms_state)
    gaps = _floor_gaps(plan, finished)
    if not gaps or scorer is None:
        return gaps
    wanted = sorted({k for keys in gaps.values() for k in keys})
    seeds = {
        n: _run_of(n, arms_state[n])
        for n in plan.sigma_arms
        if any(finished[n].get(k) is None for k in wanted)
    }
    log(f"scoring {', '.join(seeds)} post hoc for {', '.join(wanted)}, so {', '.join(gaps)} can be judged")
    _score_arms(seeds, scorer, log)
    return _floor_gaps(plan, _finished_metrics(arms_state))


def _held_reason(plan: AblationPlan, gaps: Mapping[str, Sequence[str]], name: str) -> str:
    arms = "; ".join(f"{arm} ({', '.join(keys)})" for arm, keys in gaps.items())
    return (
        f"not judged for a metric {', '.join(plan.sigma_arms)} lack: {arms}. `blink sweep rescore` "
        f"scores them post hoc, then `blink sweep ablations` runs {name}"
    )


def _resuming(arm: Arm, arms_state: Mapping[str, Mapping]) -> Mapping[str, Any] | None:
    """The entry of an arm stopped partway whose checkpoints remain (it resumes from them), else None."""
    entry = arms_state.get(arm.name, {})
    if entry.get("status") != "running":
        return None
    return entry if checkpoint_steps(paths.home() / "runs" / _run_of(arm.name, entry)) else None


def _started_with(arm: Arm, arms_state: Mapping[str, Mapping]) -> tuple[str, ...]:
    """The arms a combined arm was combining when it was stopped partway, if its checkpoints remain."""
    return tuple((_resuming(arm, arms_state) or {}).get("combined_from") or ())


def _budget(arm: Arm, rate: float, rates: Mapping[str, float], arms_state) -> tuple[float, int | None]:
    """(samples/s, steps; None for the clock's) an arm is planned at. An arm stopped partway resumes as
    it started: its checkpoints hold a WSD schedule of its recorded steps, which a re-measured bench
    row must not move (an entry of the frozen PF66 sweep has steps but no samples_per_s)."""
    entry = _resuming(arm, arms_state) or {}
    return float(entry.get("samples_per_s") or arm_rate(arm, rate, rates)), entry.get("steps")


def _plan_combined(plan: AblationPlan, arm: Arm, arms_state, scorer, log: Log) -> tuple[Arm, dict | None]:
    """(the combined arm, None) to run, or (arm, the entry saying why it does not run).

    A combined arm stopped partway resumes with the arms it started with: its checkpoints hold that
    set's training, and a rescore since may have changed the adopted set. Before it first starts it is
    held while a finished arm cannot be judged for a metric only the seeds lack."""
    arms = {a.name: a for a in plan.arms}
    started = _started_with(arm, arms_state)
    if started:
        adopted = combine_adopted(arms, _judge(plan, arms_state)[1], name=arm.name).combined_from
        now = "" if adopted == started else f"; the adopted set is now ({', '.join(adopted) or 'none'})"
        log(f"{arm.name}: resuming with the arms it started with ({', '.join(started)}){now}")
        return combine_adopted(arms, dict.fromkeys(started, {"adopt": True}), name=arm.name), None
    gaps = _settle_floor(plan, arms_state, scorer, log)
    if gaps:
        return arm, {"name": arm.name, "status": "held", "reason": _held_reason(plan, gaps, arm.name)}
    combined = combine_adopted(arms, _judge(plan, arms_state)[1], name=arm.name)
    if not combined.combined_from:
        return arm, {"name": arm.name, "status": NO_WINNERS}
    return combined, None


def _plan_arm(
    plan: AblationPlan,
    arm: Arm,
    rate: float,
    arm_rates: Mapping[str, float],
    arms_state,
    slip: bool,
    log: Log,
    scorer=None,
):
    """(entry, request) for one arm; request is None when the arm does not run, and entry says why."""
    if slip and arm.name in plan.slip_cut:
        return {"name": arm.name, "status": SLIP_CUT}, None
    to_run = arm
    if arm.combine:
        to_run, why = _plan_combined(plan, arm, arms_state, scorer, log)
        if why is not None:
            return why, None
    run = f"abl-{arm.name}"
    own, steps = _budget(to_run, rate, arm_rates, arms_state)
    try:
        config, info = _prepare(run, plan.recipe, to_run, plan.hours, own, "ablations", steps)
    except (ValueError, OSError) as exc:
        return {"name": arm.name, "status": f"invalid: {exc}"}, None
    entry = {
        **info,
        "name": arm.name,
        "change": to_run.change,
        "combined_from": list(to_run.combined_from),
        "samples_per_s": own,
    }
    request = _request(run, config, plan.data, own, plan.hours, plan.disable)
    return {**entry, "status": "running", "started": time.time()}, request


def run_ablations(
    plan: AblationPlan,
    out: Path,
    rate: float,
    runner: Runner,
    log: Log = print,
    slip: bool = False,
    arm_rates: Mapping[str, float] | None = None,
    scorer: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Run every pending arm in plan order through `runner`, recording each in `out` as it goes.

    `rate` is the plan size's samples/s (plan_rate pins it); `arm_rates` (by arm name) replaces it for
    arms with their own, and finished arms' recorded rates replace those. `scorer(run)` (a child
    `blink eval arm-metrics` in the CLI) scores seed arms post hoc before a15 is planned, when a
    finished arm is judged on a metric they predate; without one, a15 is held instead. While the user
    pause flag BLINK_HOME/PAUSE is up, no arm is planned, scored or started.
    """
    state = _load_state(out)
    pinned = plan_rate(plan, state, rate)
    if not math.isclose(pinned, rate, rel_tol=1e-3):
        seeds = ", ".join(plan.sigma_arms)
        log(f"planning at {pinned:,.0f} samples/s, the rate {seeds} were planned at, not {rate:,.0f}")
    state = {**state, "rate": pinned}
    arms_state = dict(state.get("arms", {}))
    own_rates = {**(arm_rates or {}), **_recorded_rates(plan, arms_state)}
    for arm in plan.arms:
        if not _to_plan(arm, arms_state):
            continue

        def save(entry: dict, name: str = arm.name) -> None:
            arms_state[name] = entry
            _save_state(out, {**state, "arms": arms_state})

        _hold_while_paused(arm.name, log)
        entry, request = _plan_arm(plan, arm, pinned, own_rates, arms_state, slip, log, scorer)
        _execute(entry, request, runner, save, log)
    return _report(plan, out, state, arms_state)


def _hold_while_paused(name: str, log: Log) -> None:
    flag = userpause.flag_path()
    if flag.exists():
        log(f"{name}: waiting to start while {flag} is up (paused by the user)")
        waited = userpause.wait_while_flagged(flag, userpause.POLL_S)
        log(f"{name}: the user pause is over after {waited:,.0f} s")


def _to_plan(arm: Arm, arms_state: Mapping[str, Mapping]) -> bool:
    """Whether a launch plans the arm: never recorded, stopped partway, held, or a combined arm that
    found no winner (a rescore since may have adopted one)."""
    status = arms_state.get(arm.name, {}).get("status", "pending")
    return status in RERUN_STATES or (arm.combine and status == NO_WINNERS)


def fresh_rate_arms(plan: AblationPlan, out: Path, slip: bool) -> set[str]:
    """The arms a launch now plans at a bench rate read now: those it plans, less the slip rule's
    cuts and any arm resuming at the rate it started with (finished arms' rates are recorded)."""
    arms_state = _load_state(out).get("arms", {})
    return {
        arm.name
        for arm in plan.arms
        if _to_plan(arm, arms_state)
        and not (slip and arm.name in plan.slip_cut)
        and not (_resuming(arm, arms_state) or {}).get("samples_per_s")
    }


def preview(
    plan: AblationPlan, out: Path, rate: float, arm_rates: Mapping[str, float], slip: bool
) -> list[str]:
    """`blink sweep ablations --dry-run`: what a launch now does with each arm, planned as it would be."""
    state = _load_state(out)
    arms_state, pinned, base = state.get("arms", {}), plan_rate(plan, state, rate), read_tables(plan.recipe)
    rates = {**arm_rates, **_recorded_rates(plan, arms_state)}
    lines = []
    for arm in plan.arms:
        if not _to_plan(arm, arms_state):
            lines.append(f"abl-{arm.name}: {arms_state[arm.name]['status']}")
        elif slip and arm.name in plan.slip_cut:
            lines.append(f"abl-{arm.name}: {SLIP_CUT}")
        elif arm.combine:
            lines.append(f"abl-{arm.name}: {arm.change}, chosen when it starts by the adopt rule")
        else:
            own, steps = _budget(arm, pinned, rates, arms_state)
            steps = steps or steps_for(plan.hours, own, batch_size_of(base, arm))
            peak = merged_config(base, arm, steps)["train"]["peak_lr"]
            lines.append(
                f"abl-{arm.name}: {arm.change}; steps {steps:,}; {own:,.0f} samples/s; "
                f"peak_lr {peak:g}; {arm.overrides}"
            )
    return lines


def _recipe(plan: AblationPlan, arms_state: Mapping, floor: Mapping | None, decisions: Mapping) -> dict:
    """The frozen recipe, from the combined arm as it trained. Its VAA vouches only for the arms it
    recorded (combined_from); when the adopted set has changed since (a rescore judged an arm anew),
    the recipe stays D until it reruns."""
    combine = next((arm for arm in plan.arms if arm.combine), None)
    if combine is None:
        return {"recipe": "D", "reason": "the plan has no combined arm"}
    entry = arms_state.get(combine.name, {})
    if entry.get("status") == "held":
        return {"recipe": "D", "reason": f"{combine.name} is held: {entry.get('reason')}"}
    adopted = combine_adopted({a.name: a for a in plan.arms}, decisions, name=combine.name)
    if entry.get("status") == NO_WINNERS and adopted.combined_from:
        winners = ", ".join(adopted.combined_from)
        why = f"{combine.name} did not run while no arm was adopted, and [{winners}] is now"
        return {"recipe": "D", "reason": f"{why}: `blink sweep ablations` runs {combine.name}"}
    if entry.get("status") != "finished" or not adopted.combined_from:
        return recipe_verdict(None, floor, adopted)
    trained = tuple(entry.get("combined_from") or ())
    if trained != adopted.combined_from:
        why = (
            f"{combine.name} trained with [{', '.join(trained)}] but the adopted set is now "
            f"[{', '.join(adopted.combined_from)}]: rerun {combine.name} (move runs/"
            f"{_run_of(combine.name, entry)} aside and delete its entry in ablations.json)"
        )
        return {"recipe": "D", "reason": why}
    return recipe_verdict(entry.get("metrics"), floor, adopted)


def judge_ablations(plan: AblationPlan, arms_state: Mapping[str, Mapping]) -> dict[str, Any]:
    """The arms (with any post-hoc records), the noise floor, every adopt decision and the recipe."""
    floor, decisions = _judge(plan, arms_state)
    return {
        "arms": {name: _refreshed(entry) for name, entry in arms_state.items()},
        "noise": floor,
        "decisions": decisions,
        "recipe": _recipe(plan, arms_state, floor, decisions),
    }


def _report(plan: AblationPlan, out: Path, state: Mapping, arms_state: Mapping) -> dict[str, Any]:
    report = _jsonable({**state, "plan": dataclasses.asdict(plan), **judge_ablations(plan, arms_state)})
    _save_state(out, report)
    return report


def rescore_ablations(plan: AblationPlan, out: Path, scorer: Callable[[str], Any], log: Log = print) -> dict:
    """Score every finished arm in `out` post hoc (`scorer(run)`, blink.train.posthoc.score_run in the
    CLI), then judge the plan from a fresh read of `out`; an arm that cannot be scored is logged. The
    result also lists the arms `scored` and, by arm, why the others were `not_scored`.

    Only posthoc.json files are written. Scoring takes minutes and a running sweep may record an arm
    meanwhile, so ablations.json stays the sweep's alone: its next report reads the records."""
    arms_state = _load_state(out).get("arms", {})
    finished = {n: _run_of(n, e) for n, e in arms_state.items() if e.get("status") == "finished"}
    failed = _score_arms(finished, scorer, log)
    judged = judge_ablations(plan, _load_state(out).get("arms", {}))
    scored = [name for name in finished if name not in failed]
    return _jsonable({**judged, "scored": scored, "not_scored": failed})


def supervised_runner(log: Log = print) -> Runner:
    """The real runner: `blink train` under `blink supervise`, with the size's benchmark rate."""
    from blink.train import supervise

    def run(request: RunRequest) -> Outcome:
        cfg = supervise.SuperviseConfig(bench_rate=request.bench_rate, disabled=request.disable)
        args = ["train", "--config", str(request.config), "--data", str(request.data)]
        argv = supervise.child_argv(
            supervise.train_argv(args + (["--resume"] if request.resume else []), request.run)
        )
        run_dir = paths.home() / "runs" / request.run
        return supervise.supervise(
            cfg, run_dir, argv, log=log, deadline_s=request.deadline_s, pause_flag=userpause.flag_path()
        )

    return run
