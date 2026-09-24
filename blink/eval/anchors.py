"""E5 anchors and E7 DeepMind 9M (plan P8), against Stockfish 19 UCI_Elo anchors.

E5: a 50-game locator on the dev slice against the anchor nearest a prior guess gives a rough rating
(anchor + the logistic Elo of the score, the score kept half a game inside 0 and 1). The final model then
plays 400 games (final slice) against each of 5 anchors at 100-point steps centred on the locator; inner
anchors should land in 25-75% and the outer ones are kept and flagged when they do not. Each 6 GPU-h size
and s10m, in each mode, gets the cheaper side row: the locator, then 200 games against the single nearest
anchor. E7: DM-9M gets the same locator and 5 centred anchors at 200 games each (1,000), and plays 1,000
games against Blink. The published Elo comes from Ordo over all final-slice PGNs (blink.eval.rating);
the numbers here only choose the anchors.
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
