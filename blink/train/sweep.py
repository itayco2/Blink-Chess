"""`blink sweep ablations|sizes|choose`: the recipe ablations (plan P5) and the size sweep (plan P6).

Ablations. Each arm is a TOML file of overrides on the recipe (configs/ablations/aNN.toml: an [arm]
table for the sweep plus [model] and [train] overrides). Arms run one after another, each through
`blink supervise`, for a fixed wall-clock budget: its steps are the hours times the size's measured
samples/s (bench.json) over the batch size, so the WSD schedule's own 20% cooldown ends the arm.
An arm whose change costs time per step (a10's Muon) names its own bench.json row ([arm] bench_size),
which then plans its steps and polices its throughput; a15 runs at the slowest rate of its winners.
The sweep is resumable: ablations.json records every arm, and a rerun skips finished arms and
resumes the one that was running. Judging, after every arm has cooled down:
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
ablations.json; `blink sweep rescore` writes posthoc.json files only.

Sizes. S, M and M12 run for the same wall-clock hours; a conditional size (L) runs only if its
measured rate passes the epoch floor. `choose` then applies the pre-registered precedence:
epoch floor (>= 1,658 samples/s, one epoch of training roots in T_long = 96 h) > best 6 h VAA >
default M (a best VAA within 2 sigma of M's means M; if M fails the floor, the largest passing size).
"""

import dataclasses
import json
import math
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blink import paths
from blink.model.config import compile_mode, config_from_dict, read_tables
from blink.train import posthoc
from blink.train.atomic import write_text_atomic
from blink.train.bench import best_rates
from blink.train.supervise import Outcome, checkpoint_steps, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
SIZE_ORDER = ("t", "s", "m", "m12", "l")
RERUN_STATES = ("pending", "running", "held")  # held: a15 waiting for arms it cannot judge yet
DEADLINE_FACTOR = 1.5  # a supervised arm is stopped at 1.5x its planned hours plus the allowance
COMPILE_ALLOWANCE_S = 1800.0
TOLERANCE = 1e-9
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


@dataclass(frozen=True)
class ChooseRules:
    epoch_floor: float = 1658.0  # samples/s: one epoch of training roots in T_long hours
    t_long_hours: float = 96.0
    train_roots: int = 401_000_000
    root_frac: float = 0.7  # batches mix 70% roots and 30% children
    default: str = "m"
    p99_ms_max: float = 100.0
    p99_rows: int = 219
    p99_concurrency: tuple[int, ...] = (5, 2)
    sigma_factor: float = 2.0

    @property
    def samples_per_epoch(self) -> float:
        return self.train_roots / self.root_frac

    def epochs(self, samples_per_s: float, hours: float) -> float:
        return samples_per_s * hours * 3600 / self.samples_per_epoch


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
    run: str, base_path: Path, arm: Arm, hours: float, rate: float, folder: str
) -> tuple[Path, dict]:
    """Write the run's merged config under BLINK_HOME/eval/<folder>/ and describe it."""
    base = read_tables(base_path)  # follows `base = "recipe.toml"` (PF66)
    steps = steps_for(hours, rate, batch_size_of(base, arm))
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


def own_bench_rates(plan: AblationPlan, rate_of: Callable[[str, str], float]) -> dict[str, float]:
    """samples/s by arm name for every arm that names its own bench row, from rate_of(size, mode)."""
    rates = {}
    for arm in plan.arms:
        if arm.bench_size is None:
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


def _score_arms(runs: Mapping[str, str], scorer: Callable[[str], Any], log: Log) -> None:
    """Score each arm's run post hoc; an arm that cannot be scored is logged and skipped."""
    for name, run in runs.items():
        try:
            scorer(run)
        except SCORE_ERRORS as exc:
            log(f"{name}: not rescored ({exc})")


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


def _started_with(arm: Arm, arms_state: Mapping[str, Mapping]) -> tuple[str, ...]:
    """The arms a combined arm was combining when it was stopped partway, if its checkpoints remain."""
    entry = arms_state.get(arm.name, {})
    if entry.get("status") != "running":
        return ()
    if not checkpoint_steps(paths.home() / "runs" / _run_of(arm.name, entry)):
        return ()
    return tuple(entry.get("combined_from") or ())


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
        return arm, {"name": arm.name, "status": "not run: no arm was adopted"}
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
        return {"name": arm.name, "status": "not tested: cut by the slip rule"}, None
    to_run = arm
    if arm.combine:
        to_run, why = _plan_combined(plan, arm, arms_state, scorer, log)
        if why is not None:
            return why, None
    run = f"abl-{arm.name}"
    own = arm_rate(to_run, rate, arm_rates)
    try:
        config, info = _prepare(run, plan.recipe, to_run, plan.hours, own, "ablations")
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

    `rate` is the plan size's samples/s; `arm_rates` (by arm name) replaces it for arms with their own.
    `scorer(run)` (blink.train.posthoc.score_run in the CLI) scores seed arms post hoc before a15 is
    planned, when a finished arm is judged on a metric they predate; without one, a15 is held instead.
    """
    own_rates = dict(arm_rates or {})
    state = _load_state(out)
    arms_state = dict(state.get("arms", {}))
    for arm in plan.arms:
        if arms_state.get(arm.name, {}).get("status", "pending") not in RERUN_STATES:
            continue

        def save(entry: dict, name: str = arm.name) -> None:
            arms_state[name] = entry
            _save_state(out, {**state, "arms": arms_state})

        entry, request = _plan_arm(plan, arm, rate, own_rates, arms_state, slip, log, scorer)
        _execute(entry, request, runner, save, log)
    return _report(plan, out, state, arms_state)


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
    CLI), then judge the plan from a fresh read of `out`; an arm that cannot be scored is logged.

    Only posthoc.json files are written. Scoring takes minutes and a running sweep may record an arm
    meanwhile, so ablations.json stays the sweep's alone: its next report reads the records."""
    arms_state = _load_state(out).get("arms", {})
    finished = {n: _run_of(n, e) for n, e in arms_state.items() if e.get("status") == "finished"}
    _score_arms(finished, scorer, log)
    return _jsonable(judge_ablations(plan, _load_state(out).get("arms", {})))


# ---------------------------------------------------------------- the size sweep


@dataclass(frozen=True)
class SizeSweep:
    sizes: tuple[str, ...]
    conditional: tuple[str, ...]  # run only if the measured rate passes the epoch floor
    hours: float
    recipe: Path | None  # the frozen recipe (arm-style overrides), merged over each size config
    data: Path
    config_dir: Path = CONFIG_DIR
    disable: tuple[str, ...] = ()


def load_size_sweep(path: Path, sizes: Sequence[str] | None = None, hours: float | None = None) -> SizeSweep:
    table = tomllib.loads(Path(path).read_text(encoding="utf-8"))["sizes"]
    recipe = _resolve(table["recipe"], REPO_ROOT) if table.get("recipe") else None
    if recipe is not None and not recipe.is_file():
        raise FileNotFoundError(f"{recipe} is missing: it is frozen at the end of P5, before the size sweep")
    chosen = tuple(sizes) if sizes else tuple(table["sizes"]) + tuple(table.get("conditional", ()))
    return SizeSweep(
        sizes=chosen,
        conditional=tuple(table.get("conditional", ())),
        hours=float(hours if hours is not None else table["hours"]),
        recipe=recipe,
        data=_resolve(table["data"], paths.home() / "data"),
        disable=tuple(table.get("disable", ())),
    )


def p99_of(bench: Mapping[str, Any], size: str, rules: ChooseRules) -> dict[str, float | None]:
    """Value-mode p99 at L+1 rows for each concurrency the rules name (None when not measured)."""
    found = {
        row["concurrency"]: row.get("p99_ms")
        for row in bench.get("play", [])
        if row.get("size") == size and row.get("rows") == rules.p99_rows and not row.get("error")
    }
    return {str(c): found.get(c) for c in rules.p99_concurrency}


def _size_facts(size: str, row: Mapping[str, Any], bench: Mapping, rules: ChooseRules) -> dict[str, Any]:
    rate = row["samples_per_s"]
    return {
        "samples_per_s": rate,
        "micro": row.get("micro"),
        "compile": row.get("compile"),
        "peak_reserved_gb": row.get("peak_reserved_gb"),
        "parameters": row.get("parameters"),
        "epochs": {str(h): rules.epochs(rate, h) for h in (96, 120, 132)},
        "p99_ms": p99_of(bench, size, rules),
    }


def size_compile_mode(setup: SizeSweep, size: str, recipe: Arm) -> str:
    """The compile mode a size's run trains in: its config (with base) under the recipe's overrides."""
    tables = read_tables(setup.config_dir / f"{size}.toml")
    return compile_mode({"train": {**tables["train"], **recipe.overrides.get("train", {})}})


def _plan_size(setup: SizeSweep, size: str, bench: Mapping, rules: ChooseRules):
    """(entry, request) for one size; request is None when the size does not run, and entry says why."""
    recipe = load_arm(setup.recipe) if setup.recipe else Arm("D", "Recipe D as the size config writes it")
    try:
        mode = size_compile_mode(setup, size, recipe)
    except (ValueError, OSError) as exc:
        return {"name": size, "status": f"invalid: {exc}"}, None
    row = best_rates(bench, compile=mode).get(size)
    if row is None:
        why = f"not run: no bench.json row for {size} at compile {mode} fits the VRAM budget"
        return {"name": size, "status": why}, None
    rate = row["samples_per_s"]
    if size in setup.conditional and rate < rules.epoch_floor:
        why = f"not run: fails the epoch floor ({rate:,.0f} < {rules.epoch_floor:,.0f} samples/s)"
        return {"name": size, "status": why}, None
    run = f"size-{size}"
    try:
        config, info = _prepare(run, setup.config_dir / f"{size}.toml", recipe, setup.hours, rate, "sizes")
    except (ValueError, OSError) as exc:
        return {"name": size, "status": f"invalid: {exc}"}, None
    entry = {**_size_facts(size, row, bench, rules), **info, "name": size}
    request = _request(run, config, setup.data, rate, setup.hours, setup.disable)
    return {**entry, "status": "running", "started": time.time()}, request


def run_sizes(
    setup: SizeSweep,
    bench: Mapping,
    out: Path,
    runner: Runner,
    log: Log = print,
    rules: ChooseRules | None = None,
) -> dict[str, Any]:
    """Run every pending size for the same wall-clock hours, recording each in `out` (resumable)."""
    rules = rules or ChooseRules()
    state = _load_state(out)
    sizes_state = dict(state.get("sizes", {}))
    for size in setup.sizes:
        if sizes_state.get(size, {}).get("status", "pending") not in RERUN_STATES:
            continue

        def save(entry: dict, name: str = size) -> None:
            sizes_state[name] = entry
            _save_state(out, {**state, "sizes": sizes_state})

        entry, request = _plan_size(setup, size, bench, rules)
        _execute(entry, request, runner, save, log)
    report = _jsonable({**state, "sizes": sizes_state, "hours": setup.hours})
    _save_state(out, report)
    return report


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
        return supervise.supervise(cfg, run_dir, argv, log=log, deadline_s=request.deadline_s)

    return run


# ---------------------------------------------------------------- choosing N*


def load_rules(path: Path) -> ChooseRules:
    """The [choose] table of configs/sweep.toml; defaults are the plan's numbers."""
    table = tomllib.loads(Path(path).read_text(encoding="utf-8")).get("choose", {})
    return ChooseRules(**{k: tuple(v) if isinstance(v, list) else v for k, v in table.items()})


def _order(size: str, best: Mapping[str, Mapping]) -> tuple:
    known = SIZE_ORDER.index(size) if size in SIZE_ORDER else len(SIZE_ORDER)
    return known, best.get(size, {}).get("parameters") or 0, size


def _eligibility(row: Mapping | None, vaa: float | None, p99: Mapping, rules: ChooseRules) -> dict[str, Any]:
    entry: dict[str, Any] = {"vaa": vaa, "p99_ms": dict(p99), "eligible": False, "failed": None}
    if row is None:
        return {**entry, "failed": "vram", "reason": "no measured micro-batch >= 256 fits the VRAM budget"}
    rate = row["samples_per_s"]
    entry = {
        **entry,
        "samples_per_s": rate,
        "epochs": {str(h): rules.epochs(rate, h) for h in (96, 120, 132)},
    }
    if rate < rules.epoch_floor:
        why = f"fails the epoch floor: {rate:,.0f} < {rules.epoch_floor:,.0f} samples/s"
        return {**entry, "failed": "floor", "reason": why}
    missing = [c for c, v in p99.items() if v is None]
    over = {c: v for c, v in p99.items() if v is not None and v > rules.p99_ms_max}
    if missing:
        return {**entry, "failed": "p99", "reason": f"value-mode p99 not measured at concurrency {missing}"}
    if over:
        return {**entry, "failed": "p99", "reason": f"value-mode p99 over {rules.p99_ms_max:.0f} ms: {over}"}
    if vaa is None:
        return {**entry, "failed": "vaa", "reason": "no 6 h VAA"}
    return {**entry, "eligible": True, "reason": "passes every constraint"}


def choose(
    bench: Mapping, sizes: Mapping[str, Mapping], sigma: float, rules: ChooseRules, compile: str | None = None
) -> dict[str, Any]:
    """N* by the pre-registered precedence: epoch floor > best 6 h VAA > default M.

    `compile` is the mode the long run will train in; its rates decide the epoch floor.
    """
    best = best_rates(bench, compile=compile)
    ordered = sorted(sizes, key=lambda s: _order(s, best))
    entries = {
        s: _eligibility(best.get(s), sizes[s].get("vaa"), p99_of(bench, s, rules), rules) for s in ordered
    }
    eligible = [s for s in ordered if entries[s]["eligible"]]
    default = entries.get(rules.default)
    result = {"sigma": sigma, "rules": dataclasses.asdict(rules), "sizes": entries}
    if not eligible:
        return {**result, "n_star": None, "reason": "no size passes every constraint"}
    if default is not None and default["failed"] == "floor":
        return {
            **result,
            "n_star": eligible[-1],
            "reason": "M fails the epoch floor: the largest size that passes",
        }
    top = max(eligible, key=lambda s: entries[s]["vaa"])
    if default is None or not default["eligible"]:
        return {**result, "n_star": top, "reason": f"best 6 h VAA ({rules.default} is not eligible)"}
    margin = entries[top]["vaa"] - default["vaa"]
    if top != rules.default and margin <= rules.sigma_factor * sigma + TOLERANCE:
        return {**result, "n_star": rules.default, "reason": f"{top} is within 2 sigma of M ({margin:+.4f})"}
    return {**result, "n_star": top, "reason": f"best 6 h VAA ({entries[top]['vaa']:.4f})"}
