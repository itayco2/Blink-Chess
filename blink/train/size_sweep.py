"""`blink sweep sizes`: the size sweep (plan P6), each size at equal wall-clock hours.

S, M and M12 run for the same wall-clock hours with the frozen recipe; a conditional size (L) runs only
if its measured rate passes the epoch floor. Each size is planned and policed at bench.json's best row
in the compile mode it trains in, and runs through `blink supervise` like an ablation arm, with the
same resumable state file and run plumbing (blink.train.sweep). `blink sweep choose` then picks N*
(blink.train.nstar).
"""

import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink import paths
from blink.model.config import compile_mode, read_tables
from blink.train.bench import best_rates
from blink.train.nstar import ChooseRules, p99_of
from blink.train.sweep import (  # the run plumbing the ablation sweep and the size sweep share
    CONFIG_DIR,
    REPO_ROOT,
    RERUN_STATES,
    Arm,
    Log,
    Runner,
    _execute,
    _jsonable,
    _load_state,
    _prepare,
    _request,
    _resolve,
    _save_state,
    load_arm,
)


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
