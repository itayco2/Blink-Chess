"""E4 node ladder, E4b film checkpoints at the crossover, E6 ladder round robin (plan P8).

E4: the shipped model in its shipped mode against full-strength Stockfish 19 at 2^k nodes per move,
k = 4, 6, ..., 16 (7 rungs), 100 games per rung on the final slice, then 400 more (500 in all) at the
two rungs that bracket a 50% score. When those games move the bracket (a rung's score crosses 50%), the
new bracket's rungs are topped up to 500 as well, so the published crossover always rests on two rungs of
500 games. The crossover is where the score crosses 50%, interpolated on the logit of the score against
log2(nodes) between the bracketing rungs; it fills "about level with SF19 at N nodes". Without a bracket
the ladder reports a bound (above the top rung or below the bottom one).
E4b: 6 of the run's film checkpoints, evenly spaced from first to last, 200 games each against SF19 at
the crossover node count (dev slice): how the learning curve looks in games.
E6: random, material, linear, MLP, s10m and SF19 UCI_Elo 1320, every pair 200 games (final slice).
Every match is played by a `play` callable, so each block runs with any game budget and any backend.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
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
    """100 games per rung, then more at the bracketing rungs until the bracket's two rungs both have
    `bracket_games` (re-bracketing after each top-up); returns the rungs and the crossover.

    Each top-up gives at least one more rung its full count, so the loop ends within len(rungs) rounds;
    `short_bracket` lists any final bracketing rung still short (none, unless a match came back short)."""
    from blink.eval.match import merge_reports

    merge = merge or merge_reports
    results = {nodes: _with_nodes(play(nodes, rung_games, 0), nodes) for nodes in rungs}
    for _ in range(len(rungs)):
        short = _short_rungs(results, bracket_games)
        if not short:
            break
        for nodes in short:
            played = results[nodes]["games"]
            more = play(nodes, bracket_games - played, played // 2)
            results[nodes] = _with_nodes(merge(results[nodes], more), nodes)
    rows = [results[n] for n in sorted(results)]
    return {
        "rungs": rows,
        "crossover": crossover(rows),
        "short_bracket": _short_rungs(results, bracket_games),
        "games": sum(r["games"] for r in rows),
    }


def _short_rungs(results: dict[int, Report], bracket_games: int) -> list[int]:
    """The bracketing rungs (by nodes) with fewer than `bracket_games` games."""
    pair = bracketing(list(results.values())) or ()
    return [nodes for nodes in pair if results[nodes]["games"] < bracket_games]


def film_frames(run_dir: Path) -> list[Path]:
    """The run's film frames (runs/<name>/film/frame_<step>.pt) in step order."""
    return sorted((Path(run_dir) / "film").glob("frame_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def film_run_dir(film_run: str) -> Path:
    """A run's folder: a run name under BLINK_HOME/runs, or a folder given as a path."""
    from blink import paths

    given = Path(film_run)
    return given if given.is_dir() else paths.home() / "runs" / film_run


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


# ------------------------------------------------------------------------------ the blocks, in process


def _nodes_player(agent, out_dir: Path, book: str) -> NodePlay:
    """Blink against full-strength SF19 at a node budget, in process (deterministic: no clock on either)."""
    from blink.eval import fastchess, match

    def play(nodes: int, games: int, skip: int) -> Report:
        with match.stockfish_agent(fastchess.stockfish_exe(), nodes=nodes) as stockfish:
            return match.play_inprocess(agent, stockfish, games, book, out_dir, skip)

    return play


def e4_block(ctx, state: dict) -> dict:
    from blink.eval import match
    from blink.eval.orchestrate import shipped_mode

    mode = shipped_mode(ctx, state)
    agent = match.blink_agents(ctx.model, ctx.device, results_dir=ctx.results_dir)[mode]
    play = _nodes_player(agent, ctx.out_dir / "E4", "final")
    result = run_node_ladder(play, ctx.n(RUNG_GAMES), ctx.n(BRACKET_GAMES))
    return {**result, "mode": mode, "pgns": [p for r in result["rungs"] for p in r.get("pgns", [r["pgn"]])]}


def crossover_nodes(state: dict) -> tuple[int, str | None]:
    """E4's crossover, or the ladder's end it pressed against (flagged) when there was none."""
    cross = (state.get("E4") or {}).get("crossover") or {}
    if cross.get("nodes"):
        return int(cross["nodes"]), None
    if cross.get("bound") == "above the top rung":
        return NODE_RUNGS[-1], "no crossover: Blink beat the top rung"
    return NODE_RUNGS[0], "no crossover: Blink lost to the bottom rung" if cross else "E4 has not run"


def e4b_block(ctx, state: dict) -> dict:
    from blink.eval import match
    from blink.eval.orchestrate import shipped_mode
    from blink.play import factory

    if not ctx.film_run:
        return {"skipped": "no --film-run given", "games": 0, "pgns": []}
    mode, (nodes, flag) = shipped_mode(ctx, state), crossover_nodes(state)
    checkpoints = pick_checkpoints(film_frames(film_run_dir(ctx.film_run)))

    def play(selector: str, at: int, games: int) -> Report:
        evaluator = factory.load_evaluator(selector, device=ctx.device)
        agent = factory.make_agent(mode, evaluator, epsilon=match.read_epsilon(ctx.results_dir))
        agent = replace(agent, name=f"Blink-{mode}-{Path(selector).stem}")
        return _nodes_player(agent, ctx.out_dir / "E4b", "dev")(at, games, 0)

    result = run_film_checkpoints(play, checkpoints, nodes, ctx.n(FILM_GAMES))
    return {**result, "flag": flag, "pgns": [r["pgn"] for r in result["checkpoints"]]}


def _ladder_agent(name: str, ctx, state: dict):
    from blink.baselines.evaluator import baseline_agent
    from blink.eval import fastchess, match
    from blink.eval.anchors import anchor_tc
    from blink.eval.orchestrate import shipped_mode
    from blink.play import agents

    if name == "random":
        return agents.RandomAgent()
    if name in ("material", "linear", "mlp"):
        return baseline_agent(name, device=ctx.device)  # the P3 rungs, one agent wrapper and its rules
    if name.startswith("SF"):  # an anchor: st=0.1, or the self-check fallback's control like E5's anchors
        return match.stockfish_agent(fastchess.stockfish_exe(), elo=int(name[2:]), tc=anchor_tc(ctx, state))
    return match.blink_agents(name, ctx.device, results_dir=ctx.results_dir)[shipped_mode(ctx, state)]


def e6_block(ctx, state: dict) -> dict:
    from blink.eval import match

    agents, missing = {}, {}
    for name in LADDER_PLAYERS:
        try:
            agents[name] = _ladder_agent(name, ctx, state)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            missing[name] = str(exc)

    def play(a: str, b: str, games: int) -> Report:
        return match.play_inprocess(agents[a], agents[b], games, "final", ctx.out_dir / "E6")

    players = [p for p in LADDER_PLAYERS if p in agents]
    try:
        result = run_round_robin(play, players, ctx.n(LADDER_GAMES))
    finally:
        for agent in agents.values():
            getattr(agent, "close", lambda: None)()
    return {**result, "missing": missing}
