"""E5 anchors and E7 DeepMind 9M (plan P8), against Stockfish 19 UCI_Elo anchors.

E5: a 50-game locator on the dev slice against the anchor nearest a prior guess gives a rough rating
(anchor + the logistic Elo of the score, the score kept half a game inside 0 and 1). The final model then
plays 400 games (final slice) against each of 5 anchors at 100-point steps centred on the locator; inner
anchors should land in 25-75% and the outer ones are kept and flagged when they do not. Each 6 GPU-h size
and s10m, in each mode, gets the cheaper side row: the locator, then 200 games against the single nearest
anchor. E7: DM-9M gets the same locator and 5 centred anchors at 200 games each (1,000), and plays 1,000
games against Blink. The published Elo comes from Ordo over all final-slice PGNs (blink.eval.rating);
the numbers here only choose the anchors.

E0's SF self-check decides how the anchors play: st=0.1 when it passed, else the plan's fallback, 60+0.6,
with E5 shrunk to the final model in the shipped mode only (no second mode, no side rows). The verdict comes
from E0 in this run, or from <out>/E0.json when a block runs alone into the same --out folder.
"""

from collections.abc import Callable, Sequence

from blink.eval import rating
from blink.eval.rating import Anchor

LOCATOR_GAMES = 50
ANCHOR_GAMES = 400
SIDE_GAMES = 200
ANCHOR_COUNT = 5
DM_ANCHOR_GAMES = 200
BLINK_VS_DM_GAMES = 1000
DEFAULT_PRIOR = 1800
IN_BAND = (0.25, 0.75)
FALLBACK_TC = "60+0.6"  # E0 (3): the anchors' control when SF's st=0.1 self-check fails (plan P8)
LOCATOR_SUFFIX = "-locator"  # the only dev-slice games these blocks play are the locators'

Report = dict
AnchorPlay = Callable[[Anchor, int, str, int], Report]  # (anchor, games, book slice, openings to skip)


def locator_estimate(anchor: Anchor, report: Report) -> float:
    """The anchor's rating plus the logistic Elo of the score, clamped half a game inside 0% and 100%."""
    games = max(1, report["games"])
    score = min(max(report["score"] or 0.0, 0.5 / games), 1 - 0.5 / games)
    return anchor.rating + rating.logistic_elo(score)


def _flag(report: Report, anchor: Anchor) -> Report:
    score = report["score"]
    in_band = score is not None and IN_BAND[0] <= score <= IN_BAND[1]
    return {**report, "anchor": anchor.name, "anchor_rating": anchor.rating, "in_band": in_band}


def locate(play: AnchorPlay, grid: Sequence[Anchor], prior: float, games: int = LOCATOR_GAMES) -> dict:
    anchor = rating.nearest_anchor(prior, grid)
    report = play(anchor, games, "dev", 0)
    return {**_flag(report, anchor), "estimate": locator_estimate(anchor, report)}


def run_anchor_block(
    play: AnchorPlay,
    grid: Sequence[Anchor],
    prior: float = DEFAULT_PRIOR,
    locator_games: int = LOCATOR_GAMES,
    anchor_games: int = ANCHOR_GAMES,
    count: int = ANCHOR_COUNT,
) -> dict:
    """E5 for the final model in one mode: the locator, then `count` centred anchors on the final slice."""
    locator = locate(play, grid, prior, locator_games)
    chosen = rating.centred_anchors(locator["estimate"], grid, count)
    rows = [_flag(play(anchor, anchor_games, "final", 0), anchor) for anchor in chosen]
    inner = rows[1:-1] if len(rows) > 2 else rows
    return {
        "locator": locator,
        "anchors": rows,
        "inner_in_band": all(r["in_band"] for r in inner),
        "outer_flagged": [r["anchor"] for r in (rows[0], rows[-1]) if not r["in_band"]] if rows else [],
        "games": locator["games"] + sum(r["games"] for r in rows),
        "pgns": [r["pgn"] for r in [locator, *rows]],
    }


def run_side_row(
    play: AnchorPlay,
    grid: Sequence[Anchor],
    prior: float = DEFAULT_PRIOR,
    locator_games: int = LOCATOR_GAMES,
    games: int = SIDE_GAMES,
) -> dict:
    """A 6 GPU-h size or s10m in one mode: the locator, then the single nearest anchor on the final slice."""
    locator = locate(play, grid, prior, locator_games)
    anchor = rating.nearest_anchor(locator["estimate"], grid)
    row = _flag(play(anchor, games, "final", 0), anchor)
    return {
        "locator": locator,
        "anchors": [row],
        "games": locator["games"] + row["games"],
        "pgns": [locator["pgn"], row["pgn"]],
    }


def run_dm_block(
    play_anchor: AnchorPlay,
    play_blink: Callable[[int], Report],
    grid: Sequence[Anchor],
    prior: float = DEFAULT_PRIOR,
    locator_games: int = LOCATOR_GAMES,
    anchor_games: int = DM_ANCHOR_GAMES,
    blink_games: int = BLINK_VS_DM_GAMES,
) -> dict:
    """E7: DM-9M's gauntlet against 5 centred anchors, then Blink against DM-9M (a fishtest Elo too)."""
    gauntlet = run_anchor_block(play_anchor, grid, prior, locator_games, anchor_games)
    head = play_blink(blink_games)
    estimate = rating.elo_ci(head["penta"]).as_dict() if head.get("penta") else None
    return {
        "gauntlet": gauntlet,
        "blink_vs_dm": {**head, "elo": estimate},
        "games": gauntlet["games"] + head["games"],
        "pgns": [*gauntlet["pgns"], head["pgn"]],
    }


# ------------------------------------------------------------------------------ the blocks (fastchess)


def anchor_tc(ctx, state: dict) -> str | None:
    """None when the anchors play st=0.1; FALLBACK_TC when E0's SF self-check failed."""
    from blink.eval.orchestrate import earlier_report

    check = earlier_report(ctx, state, "E0").get("sf_selfcheck") or {}
    return FALLBACK_TC if check.get("passed") is False else None


def fastchess_player(ctx, selector: str, mode: str, subdir: str, anchor_tc: str | None = None) -> AnchorPlay:
    """The engine under test (Blink, or DM-9M for a dm selector) against one clocked UCI_Elo anchor,
    at st=0.1 or at `anchor_tc` (the self-check fallback); Blink keeps st=1 either way.

    Blink gets E2b's epsilon (results/epsilon.json) on its command line: the same value the in-process
    blocks read, so every game filed under one Blink name is played by one configuration. The locator's
    dev-slice games go to <subdir>-locator, so <subdir> holds final-slice games only."""
    from blink.eval import fastchess, match

    epsilon = match.read_epsilon(ctx.results_dir)

    def play(anchor: Anchor, games: int, book: str, skip: int) -> Report:
        first = fastchess.blink_engine(selector, mode, ctx.device, epsilon=epsilon)
        second = fastchess.stockfish_anchor(anchor.rating, fastchess.stockfish_exe())
        if anchor_tc:
            second = fastchess.with_tc(second, anchor_tc)
        folder = ctx.out_dir / (subdir if book == "final" else f"{subdir}{LOCATOR_SUFFIX}")
        gauntlet = fastchess.prepare_pair(first, second, games, book, folder, ctx.concurrency, skip=skip)
        return fastchess.match_report(fastchess.execute(gauntlet))

    return play


def _prior(state: dict) -> float:
    """Where the locator starts: the final model's E5 estimate for later rows, else the default."""
    final = (state.get("E5") or {}).get("final") or {}
    estimates = [block["locator"]["estimate"] for block in final.values()]
    return sum(estimates) / len(estimates) if estimates else DEFAULT_PRIOR


def e5_block(ctx, state: dict) -> dict:
    from blink.eval import fastchess
    from blink.eval.orchestrate import shipped_mode
    from blink.play.factory import MODES

    for mode in MODES:
        fastchess.check_distinct_names([ctx.model, *ctx.side_models], mode)
    tc = anchor_tc(ctx, state)
    modes, side_models = ((shipped_mode(ctx, state),), ()) if tc else (MODES, ctx.side_models)
    grid = rating.read_anchors()
    final = {
        mode: run_anchor_block(
            fastchess_player(ctx, ctx.model, mode, "E5", tc),
            grid,
            DEFAULT_PRIOR,
            ctx.n(LOCATOR_GAMES),
            ctx.n(ANCHOR_GAMES),
        )
        for mode in modes
    }
    prior = _prior({"E5": {"final": final}})
    side = {
        f"{selector}|{mode}": run_side_row(
            fastchess_player(ctx, selector, mode, "E5-side", tc),
            grid,
            prior,
            ctx.n(LOCATOR_GAMES),
            ctx.n(SIDE_GAMES),
        )
        for selector in side_models
        for mode in modes
    }
    blocks = [*final.values(), *side.values()]
    return {
        "anchor_tc": tc,
        "final": final,
        "side": side,
        "games": sum(b["games"] for b in blocks),
        "pgns": [p for b in blocks for p in b["pgns"]],
        "final_slice_pgns": [r["pgn"] for b in blocks for r in b["anchors"]],
    }


def e7_block(ctx, state: dict) -> dict:
    from blink.eval import match
    from blink.eval.orchestrate import shipped_mode
    from blink.reference import registry

    dm = "dm:9M"
    blink = match.blink_agents(ctx.model, ctx.device, results_dir=ctx.results_dir)[shipped_mode(ctx, state)]
    deepmind = registry.load_agent(dm, device=ctx.device)

    def play_blink(games: int) -> Report:
        return match.play_inprocess(blink, deepmind, games, "final", ctx.out_dir / "E7")

    tc = anchor_tc(ctx, state)
    result = run_dm_block(
        fastchess_player(ctx, dm, "policy", "E7", tc),
        play_blink,
        rating.read_anchors(),
        _prior(state),
        ctx.n(LOCATOR_GAMES),
        ctx.n(DM_ANCHOR_GAMES),
        ctx.n(BLINK_VS_DM_GAMES),
    )
    anchors_rows = result["gauntlet"]["anchors"]
    return {
        **result,
        "anchor_tc": tc,
        "final_slice_pgns": [*(r["pgn"] for r in anchors_rows), result["blink_vs_dm"]["pgn"]],
    }
