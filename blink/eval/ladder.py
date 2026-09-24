"""E4 node ladder, E4b film checkpoints at the crossover, E6 ladder round robin (plan P8).

E4: the shipped model in its shipped mode against full-strength Stockfish 19 at 2^k nodes per move,
k = 4, 6, ..., 16 (7 rungs), 100 games per rung on the final slice, then 400 more (500 in all) at the
two rungs that bracket a 50% score. The crossover is where the score crosses 50%, interpolated on the
logit of the score against log2(nodes) between the bracketing rungs; it fills "about level with SF19 at
N nodes". Without a bracket the ladder reports a bound (above the top rung or below the bottom one).
E4b: 6 of the run's film checkpoints, evenly spaced from first to last, 200 games each against SF19 at
the crossover node count (dev slice): how the learning curve looks in games.
E6: random, material, linear, MLP, s10m and SF19 UCI_Elo 1320, every pair 200 games (final slice).
Every match is played by a `play` callable, so each block runs with any game budget and any backend.
"""

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

NODE_RUNGS = tuple(2**k for k in range(4, 17, 2))
RUNG_GAMES = 100
BRACKET_GAMES = 500
FILM_CHECKPOINTS = 6
FILM_GAMES = 200
LADDER_PLAYERS = ("random", "material", "linear", "mlp", "run:s10m", "SF1320")
LADDER_GAMES = 200
SCORE_CLAMP = 0.01

Report = dict
NodePlay = Callable[[int, int, int], Report]  # (nodes, games, openings to skip) -> report


def _logit(score: float) -> float:
    s = min(max(score, SCORE_CLAMP), 1 - SCORE_CLAMP)
    return math.log(s / (1 - s))


def bracketing(rungs: Sequence[Report]) -> tuple[int, int] | None:
    """The first adjacent pair of rungs (by nodes) where the score falls from >= 50% to below it."""
    ordered = sorted(rungs, key=lambda r: r["nodes"])
    for low, high in zip(ordered, ordered[1:], strict=False):
        if low["score"] is not None and high["score"] is not None and low["score"] >= 0.5 > high["score"]:
            return low["nodes"], high["nodes"]
    return None


def crossover(rungs: Sequence[Report]) -> dict:
    """The node count at which the score crosses 50%, or the bound the ladder reached."""
    pair = bracketing(rungs)
    if pair is None:
        scores = [r["score"] for r in sorted(rungs, key=lambda r: r["nodes"]) if r["score"] is not None]
        bound = "above the top rung" if scores and min(scores) >= 0.5 else "below the bottom rung"
        return {"nodes": None, "bound": bound, "bracket": None}
    by_nodes = {r["nodes"]: r for r in rungs}
    low, high = by_nodes[pair[0]], by_nodes[pair[1]]
    x0, x1 = math.log2(low["nodes"]), math.log2(high["nodes"])
    y0, y1 = _logit(low["score"]), _logit(high["score"])
    x = x0 + (y0 / (y0 - y1)) * (x1 - x0) if y0 != y1 else (x0 + x1) / 2
    return {"nodes": round(2**x), "bound": None, "bracket": list(pair)}


def _with_nodes(report: Report, nodes: int) -> Report:
    return {**report, "nodes": nodes}


def run_node_ladder(
    play: NodePlay,
    rung_games: int = RUNG_GAMES,
    bracket_games: int = BRACKET_GAMES,
    rungs: Sequence[int] = NODE_RUNGS,
    merge: Callable[[Report, Report], Report] | None = None,
) -> dict:
    """100 games per rung, then more at the two bracketing rungs; returns the rungs and the crossover."""
    from blink.eval.match import merge_reports

    merge = merge or merge_reports
    results = {nodes: _with_nodes(play(nodes, rung_games, 0), nodes) for nodes in rungs}
    pair = bracketing(list(results.values()))
    extra = max(0, bracket_games - rung_games)
    if pair is not None and extra:
        for nodes in pair:
            more = play(nodes, extra, rung_games // 2)
            results[nodes] = _with_nodes(merge(results[nodes], more), nodes)
    rows = [results[n] for n in sorted(results)]
    return {"rungs": rows, "crossover": crossover(rows), "games": sum(r["games"] for r in rows)}


def film_frames(run_dir: Path) -> list[Path]:
    """The run's film frames (runs/<name>/film/frame_<step>.pt) in step order."""
    return sorted((Path(run_dir) / "film").glob("frame_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def pick_checkpoints(frames: Sequence[Path], count: int = FILM_CHECKPOINTS) -> list[Path]:
    """`count` frames evenly spaced from the first to the last (all of them when there are fewer)."""
    if len(frames) <= count:
        return list(frames)
    picks = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[i] for i in picks]


def run_film_checkpoints(
    play: Callable[[str, int, int], Report], checkpoints: Sequence[Path], nodes: int, games: int = FILM_GAMES
) -> dict:
    """E4b: each checkpoint against SF19 at the crossover node count."""
    rows = [{**play(str(path), nodes, games), "checkpoint": str(path)} for path in checkpoints]
    return {"nodes": nodes, "checkpoints": rows, "games": sum(r["games"] for r in rows)}


def round_robin_pairs(players: Sequence[str]) -> list[tuple[str, str]]:
    return list(combinations(players, 2))


def run_round_robin(
    play: Callable[[str, str, int], Report],
    players: Sequence[str] = LADDER_PLAYERS,
    games: int = LADDER_GAMES,
) -> dict:
    """E6: every pair plays `games` games; the ratings come from Ordo over the PGNs."""
    rows = [play(a, b, games) for a, b in round_robin_pairs(players)]
    return {"pairs": rows, "games": sum(r["games"] for r in rows), "pgns": [r["pgn"] for r in rows]}
