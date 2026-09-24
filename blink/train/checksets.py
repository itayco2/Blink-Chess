"""The two held-out sets a check row also scores: games10k (arm a07's metric) and the mateset (a08's guard).

They run only at the full-valprobe checks (5, 25, 30, 50 and 100% of the steps, or a preview's end),
never in the cheap 2k-step rows. Like vaa and top1, the raw weights write the plain keys and the EMA
writes ema_ keys: the sweep judges the plain vaa and top1 (blink.train.sweep.decide), so games10k_top1
and mate_preserving must come from the same model to be judged beside them.

Each file is loaded once per run, at the first check, and kept. A run without one (a skeleton pack, a
raw-source run, a machine without BLINK_HOME/data/games10k.npy) trains exactly as before: the metric
is not written, and the log says once what was skipped and why. An unreadable file (missing fields, an
empty file, half an .npz from a writer that was cut off) is logged the same way rather than stopping a
training run over an evaluation input; blink.train.evals likewise writes a check row without these
keys when scoring them fails.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from blink.train import games10k_eval, mateset_eval, telemetry

Log = Callable[[str], None]
_UNSET = object()


class Lazy[T]:
    """One file, read on the first `get` and kept for the run; what happened is logged once."""

    def __init__(
        self,
        name: str,
        path: Path | None,
        loader: Callable[[Path], T],
        metrics: str,
        describe: Callable[[T], str] = lambda _: "loaded",
    ) -> None:
        self.name, self.path, self.loader, self.metrics, self.describe = name, path, loader, metrics, describe
        self._value: Any = _UNSET

    def get(self, log: Log) -> T | None:
        if self._value is _UNSET:
            self._value = self._load(log)
        return self._value

    def _load(self, log: Log) -> T | None:
        if self.path is None:
            log(f"{self.name}: none for this run's data; {self.metrics} not scored")
            return None
        path = Path(self.path)
        if not path.is_file():
            log(f"{self.name}: {path} not found; {self.metrics} not scored in this run")
            return None
        try:
            value = self.loader(path)
        except Exception as exc:  # np.load: EOFError on an empty file, BadZipFile on half an .npz
            why = f"{type(exc).__name__}: {exc}"
            log(f"{self.name}: cannot use {path} ({why}); {self.metrics} not scored in this run")
            return None
        log(f"{self.name}: {self.describe(value)} from {path}")
        return value


def _describe_games(games: games10k_eval.GameSet) -> str:
    return f"{games.n:,} positions"


def _describe_mates(mates: mateset_eval.Mateset) -> str:
    text = f"{mates.probe.n_roots:,} roots, {len(mates.probe.child_board):,} children"
    if mates.child_mate_in is None:
        text += (
            f" (no {mateset_eval.CHILD_MATE_IN}, every legal move's mate status: "
            "mate_preserving is not scored, shortest_mate is)"
        )
    return text


def _games(path: Path | None) -> Lazy[games10k_eval.GameSet]:
    return Lazy("games10k", path, games10k_eval.load, "games10k_top1", _describe_games)


def _mates(path: Path | None) -> Lazy[mateset_eval.Mateset]:
    return Lazy("mateset", path, mateset_eval.load, "shortest_mate and mate_preserving", _describe_mates)


@dataclass(frozen=True)
class CheckSets:
    games10k: Lazy[games10k_eval.GameSet] = field(default_factory=lambda: _games(None))
    mateset: Lazy[mateset_eval.Mateset] = field(default_factory=lambda: _mates(None))


def for_run(games10k: Path | None, mateset: Path | None) -> CheckSets:
    return CheckSets(_games(games10k), _mates(mateset))


def _games_metrics(raw, ema, device, games: games10k_eval.GameSet, chunk: int, tick) -> dict[str, Any]:
    chunk = min(chunk, telemetry.EVAL_CHUNK)  # fp32 rows, as the val top1 runs them
    raw_top1 = games10k_eval.top1(raw, games, device, chunk, tick)
    ema_top1 = games10k_eval.top1(ema, games, device, chunk, tick)
    return {
        "games10k_top1": raw_top1["top1"],
        "ema_games10k_top1": ema_top1["top1"],
        "games10k_n": raw_top1["n"],
    }


def _mate_metrics(raw, ema, device, mates: mateset_eval.Mateset, chunk: int, tick) -> dict[str, Any]:
    raw_rates = mateset_eval.evaluate(raw, mates, device, chunk, tick=tick)
    ema_rates = mateset_eval.evaluate(ema, mates, device, chunk, tick=tick)
    rates = [key for key in ("shortest_mate", "mate_preserving") if key in raw_rates]
    return {
        **{k: raw_rates[k] for k in rates},
        **{f"ema_{k}": ema_rates[k] for k in rates},
        "mateset_n": raw_rates["n"],
    }


def score(
    raw: torch.nn.Module,
    ema: torch.nn.Module,
    device: torch.device,
    sets: CheckSets,
    log: Log,
    chunk: int,
    tick: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """The games10k and mateset keys for these weights: the raw ones write the plain keys, the EMA the
    ema_ keys (none for a set that is absent). Post-hoc scoring (blink.train.posthoc) calls this too."""
    out: dict[str, Any] = {}
    games = sets.games10k.get(log)
    if games is not None:
        out.update(_games_metrics(raw, ema, device, games, chunk, tick))
    mates = sets.mateset.get(log)
    if mates is not None:
        out.update(_mate_metrics(raw, ema, device, mates, chunk, tick))
    return out


def metrics(run, chunk: int, tick: Callable[[], None] | None = None) -> dict[str, Any]:
    """The games10k and mateset keys of a check row (none for a set this run does not have)."""
    return score(run.model, run.ema.module, run.device, run.sets, run.log, chunk, tick)
