"""Choosing N* after the size sweep (plan P6), by the pre-registered precedence: epoch floor
(>= 1,658 samples/s, one epoch of training roots in T_long = 96 h) > best 6 h VAA > default M (a best
VAA within 2 sigma of M's means M; if M fails the floor, the largest passing size). A size must also
fit the VRAM budget at micro-batch >= 256 and keep value-mode p99 <= 100 ms at L+1 rows.

The p99 is read only from bench play rows timed in the one play mode the rules name (p99_precision,
p99_compile; blink.play.fastmode), so a size is judged at the mode Blink will actually play in. The
defaults, fp32 uncompiled, are the plan's play runtime.
"""

import dataclasses
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink.play import fastmode
from blink.train.bench import best_rates, play_mode

SIZE_ORDER = ("t", "s", "m", "m12", "l")
TOLERANCE = 1e-9  # float slack for every pre-registered comparison (the sweep's too)


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
    p99_precision: str = fastmode.DEFAULT_PRECISION  # the play mode the p99 is judged in
    p99_compile: bool = False
    sigma_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.p99_precision not in fastmode.PRECISIONS:
            raise ValueError(
                f"p99_precision must be one of {fastmode.PRECISIONS}, got {self.p99_precision!r}"
            )
        if not isinstance(self.p99_compile, bool):
            raise ValueError(f"p99_compile must be true or false, got {self.p99_compile!r}")

    @property
    def play_mode(self) -> tuple[str, bool]:
        return self.p99_precision, self.p99_compile

    @property
    def samples_per_epoch(self) -> float:
        return self.train_roots / self.root_frac

    def epochs(self, samples_per_s: float, hours: float) -> float:
        return samples_per_s * hours * 3600 / self.samples_per_epoch


def load_rules(path: Path) -> ChooseRules:
    """The [choose] table of configs/sweep.toml; defaults are the plan's numbers."""
    table = tomllib.loads(Path(path).read_text(encoding="utf-8")).get("choose", {})
    return ChooseRules(**{k: tuple(v) if isinstance(v, list) else v for k, v in table.items()})


def p99_of(bench: Mapping[str, Any], size: str, rules: ChooseRules) -> dict[str, float | None]:
    """Value-mode p99 at L+1 rows for each concurrency the rules name, from rows timed in the rules'
    play mode only (None when not measured in it)."""
    found = {
        row["concurrency"]: row.get("p99_ms")
        for row in bench.get("play", [])
        if row.get("size") == size
        and row.get("rows") == rules.p99_rows
        and not row.get("error")
        and play_mode(row) == rules.play_mode
    }
    return {str(c): found.get(c) for c in rules.p99_concurrency}


def _order(size: str, best: Mapping[str, Mapping]) -> tuple:
    known = SIZE_ORDER.index(size) if size in SIZE_ORDER else len(SIZE_ORDER)
    return known, best.get(size, {}).get("parameters") or 0, size


def _eligibility(
    row: Mapping | None, vaa: float | None, p99: Mapping, rules: ChooseRules, pin: int | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {"vaa": vaa, "p99_ms": dict(p99), "eligible": False, "failed": None}
    if row is None:
        measured = (
            "no measured micro-batch >= 256" if pin is None else f"no row at its pinned micro-batch {pin}"
        )
        return {**entry, "failed": "vram", "reason": f"{measured} fits the VRAM budget"}
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
    mode = fastmode.describe(*rules.play_mode)
    if missing:
        why = f"value-mode p99 not measured at concurrency {missing} in {mode}"
        return {**entry, "failed": "p99", "reason": why}
    if over:
        why = f"value-mode p99 over {rules.p99_ms_max:.0f} ms in {mode}: {over}"
        return {**entry, "failed": "p99", "reason": why}
    if vaa is None:
        return {**entry, "failed": "vaa", "reason": "no 6 h VAA"}
    return {**entry, "eligible": True, "reason": "passes every constraint"}


def choose(
    bench: Mapping,
    sizes: Mapping[str, Mapping],
    sigma: float,
    rules: ChooseRules,
    compile: str | None = None,
    pins: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """N* by the pre-registered precedence: epoch floor > best 6 h VAA > default M.

    `compile` is the mode the long run will train in and `pins` the micro-batch each size's config
    pins (blink.train.size_sweep.size_micro_pin); the rates of those rows decide the epoch floor.
    """
    pins = pins or {}
    best = best_rates(bench, compile=compile, pins=pins)
    ordered = sorted(sizes, key=lambda s: _order(s, best))
    entries = {
        s: _eligibility(best.get(s), sizes[s].get("vaa"), p99_of(bench, s, rules), rules, pins.get(s))
        for s in ordered
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
