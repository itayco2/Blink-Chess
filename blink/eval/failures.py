"""E9: Blink's failures, grouped into named classes (plan P8).

A failure is a final-slice game Blink lost or drew although Stockfish 19 at 1M nodes rated Blink at least
+3.00 at one of its turns. E9 reads the shipped engine's games against E5's centred anchors (its exact name,
the shipped mode, never the locator or side rows) and takes the games in turn from each anchor's PGN (the
first game of every file, then the second, ...), so the 50 failures it keeps span the anchors. Every
position where Blink was to move is labelled (cached, blink.eval.sflabel), and each failure gets one class:
- time forfeit or crash: the game ended on a clock, an illegal move or a crashed engine;
- missed a forced mate: Stockfish saw a mate for Blink at one of its turns and the game was not won;
- one-move blunder: a lost game in which one Blink move dropped Blink's score by 3.00 or more;
- slow collapse: a lost game with no single drop that large;
- a drawn game is named by how it ended: repetition, fifty-move rule, stalemate, insufficient material,
  or the 600-ply adjudication (no progress).
"""

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn

from blink.eval.sflabel import SfLabeler

ADVANTAGE_PAWNS = 3.0
BLUNDER_PAWNS = 3.0
MAX_FAILURES = 50
FORFEITS = {"time forfeit", "illegal move", "abandoned", "unterminated"}
DRAW_CLASSES = {
    "threefold repetition": "repetition while winning",
    "fifty-move rule": "fifty-move rule while winning",
    "stalemate": "stalemated the opponent",
    "insufficient material": "traded into insufficient material",
    "adjudication": "no progress: 600-ply adjudication",
}


@dataclass(frozen=True)
class Turn:
    ply: int
    fen: str
    pawns: float  # Stockfish's score for Blink, the side to move (a mate is +-100)
    mate: bool  # Stockfish sees a forced mate for Blink here


@dataclass(frozen=True)
class Failure:
    file: str
    game: int
    blink: str
    result: str
    ending: str
    peak: float
    worst_drop: float
    failure_class: str


def read_games(path: Path) -> Iterator[chess.pgn.Game]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            yield game


def blink_side(game: chess.pgn.Game, player: str, exact: bool = False) -> chess.Color | None:
    """Blink's colour: the side whose name is `player` (exact) or contains it (case-insensitive)."""
    white, black = game.headers.get("White", ""), game.headers.get("Black", "")
    if exact:
        is_white, is_black = white == player, black == player
    else:
        needle = player.lower()
        is_white, is_black = needle in white.lower(), needle in black.lower()
    if is_white and not is_black:
        return chess.WHITE
    if is_black and not is_white:
        return chess.BLACK
    return None


def blink_result(game: chess.pgn.Game, side: chess.Color) -> str:
    """win, draw, loss or unfinished, for Blink."""
    result = game.headers.get("Result", "*")
    if result == "1/2-1/2":
        return "draw"
    if result in ("1-0", "0-1"):
        return "win" if (result == "1-0") == (side == chess.WHITE) else "loss"
    return "unfinished"


def ending(game: chess.pgn.Game) -> str:
    """How the game ended, from its Termination tag and its final position (the arbiter's view)."""
    termination = game.headers.get("Termination", "").lower()
    if termination in FORFEITS or termination == "adjudication":
        return termination
    board = game.end().board()
    for test, name in (
        (board.is_checkmate, "checkmate"),
        (board.is_stalemate, "stalemate"),
        (board.is_insufficient_material, "insufficient material"),
        (lambda: board.halfmove_clock >= 100, "fifty-move rule"),
        (lambda: board.is_repetition(3), "threefold repetition"),
    ):
        if test():
            return name
    return "other"


def blink_turns(game: chess.pgn.Game, side: chess.Color, labeler: SfLabeler) -> list[Turn]:
    """Stockfish's score for Blink at every position where Blink was to move (book moves skipped)."""
    spots = []
    board = game.board()
    for node in game.mainline():
        if board.turn == side and node.comment.strip() != "book":
            spots.append((board.ply(), board.fen()))
        board.push(node.move)
    labels = labeler.label_many([(fen, None) for _, fen in spots])
    return [
        Turn(ply, fen, label.pawns, label.mate is not None and label.mate > 0)
        for (ply, fen), label in zip(spots, labels, strict=True)
    ]


def worst_drop(turns: Sequence[Turn]) -> float:
    """The largest fall in Blink's score from one of its turns to the next."""
    return max((a.pawns - b.pawns for a, b in zip(turns, turns[1:], strict=False)), default=0.0)


def classify(result: str, how: str, turns: Sequence[Turn]) -> str:
    if how in FORFEITS:
        return "time forfeit or crash"
    if any(t.mate for t in turns):
        return "missed a forced mate"
    if result == "loss":
        return "one-move blunder" if worst_drop(turns) >= BLUNDER_PAWNS else "slow collapse"
    return DRAW_CLASSES.get(how, "other draw")


def examine(
    game: chess.pgn.Game, number: int, source: str, player: str, labeler: SfLabeler, exact: bool = False
) -> Failure | None:
    side = blink_side(game, player, exact)
    if side is None:
        return None
    result = blink_result(game, side)
    if result not in ("loss", "draw"):
        return None
    turns = blink_turns(game, side, labeler)
    peak = max((t.pawns for t in turns), default=0.0)
    if peak < ADVANTAGE_PAWNS:
        return None
    how = ending(game)
    name = game.headers.get("White" if side == chess.WHITE else "Black", "?")
    return Failure(source, number, name, result, how, peak, worst_drop(turns), classify(result, how, turns))


def in_turn(pgns: Sequence[Path]) -> Iterator[tuple[Path, int, chess.pgn.Game]]:
    """(file, game number, game): the first game of every file, then the second of every file, and so on."""
    streams = [(Path(p), enumerate(read_games(p), start=1)) for p in pgns]
    while streams:
        still = []
        for path, games in streams:
            item = next(games, None)
            if item is not None:
                yield path, item[0], item[1]
                still.append((path, games))
        streams = still


def run_failures(
    pgns: Sequence[Path],
    labeler: SfLabeler,
    player: str = "blink",
    max_failures: int = MAX_FAILURES,
    max_examined: int | None = None,
    exact: bool = False,
) -> dict:
    """Up to `max_failures` failures, taking the files' games in turn, with class counts. `player` is
    Blink's exact name when `exact`, else a case-insensitive part of it.

    `max_examined` caps the games looked at (a smoke run's bound on Stockfish searches)."""
    found: list[Failure] = []
    examined = 0
    for path, number, game in in_turn(pgns):
        if len(found) >= max_failures or (max_examined is not None and examined >= max_examined):
            break
        examined += 1
        failure = examine(game, number, path.name, player, labeler, exact)
        if failure is not None:
            found.append(failure)
    return {
        "examined_games": examined,
        "failures": [asdict(f) for f in found],
        "classes": dict(Counter(f.failure_class for f in found).most_common()),
        "positions_searched": labeler.searched,
    }


def e9_pgns(ctx, state: dict, mode: str) -> list[Path]:
    """The shipped mode's E5 games against its centred anchors (final slice), from this run or from
    <out>/E5.json; refused when E5 recorded none or a file is gone."""
    from blink.eval.orchestrate import earlier_report

    block = ((earlier_report(ctx, state, "E5").get("final") or {}).get(mode)) or {}
    pgns = [Path(row["pgn"]) for row in block.get("anchors", [])]
    if not pgns:
        raise ValueError(
            f"E9 reads E5's final-slice anchor games in the shipped mode ({mode}) and none are recorded: "
            "run E5 first, in this run or into the same --out folder"
        )
    missing = [str(p) for p in pgns if not p.is_file()]
    if missing:
        raise ValueError(f"E9: E5's anchor games are missing: {', '.join(missing)}")
    return pgns


def e9_block(ctx, state: dict) -> dict:
    """The shipped engine's failures in its E5 anchor games, labelled by SF19 at 1M nodes."""
    from blink.eval import fastchess
    from blink.eval.orchestrate import shipped_mode
    from blink.eval.sflabel import SfLabeler

    mode = shipped_mode(ctx, state)
    player = fastchess.engine_name(ctx.model, mode)
    pgns = e9_pgns(ctx, state, mode)
    cap = MAX_FAILURES if ctx.games is None else min(MAX_FAILURES, ctx.games)
    with SfLabeler(1_000_000, exe=fastchess.stockfish_exe(), procs=ctx.sf_procs) as labeler:
        result = run_failures(
            pgns, labeler, player=player, max_failures=cap, max_examined=ctx.positions, exact=True
        )
    return {**result, "player": player, "source_pgns": [str(p) for p in pgns], "games": 0, "pgns": []}
