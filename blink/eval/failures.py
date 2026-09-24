"""E9: Blink's failures, grouped into named classes (plan P8).

A failure is a final-slice game Blink lost or drew although Stockfish 19 at 1M nodes rated Blink at least
+3.00 at one of its turns. Every position where Blink was to move is labelled (cached, blink.eval.sflabel);
the first 50 failures are kept and each gets one class:
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


def blink_side(game: chess.pgn.Game, player: str) -> chess.Color | None:
    white, black = game.headers.get("White", "").lower(), game.headers.get("Black", "").lower()
    if player in white and player not in black:
        return chess.WHITE
    if player in black and player not in white:
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
    turns = []
    board = game.board()
    for node in game.mainline():
        if board.turn == side and node.comment.strip() != "book":
            label = labeler.label(board.fen())
            turns.append(
                Turn(board.ply(), board.fen(), label.pawns, label.mate is not None and label.mate > 0)
            )
        board.push(node.move)
    return turns


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
    game: chess.pgn.Game, number: int, source: str, player: str, labeler: SfLabeler
) -> Failure | None:
    side = blink_side(game, player)
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


def run_failures(
    pgns: Sequence[Path], labeler: SfLabeler, player: str = "blink", max_failures: int = MAX_FAILURES
) -> dict:
    """The first `max_failures` failures in the PGNs, in file and game order, with class counts."""
    found: list[Failure] = []
    examined = 0
    for path in pgns:
        for number, game in enumerate(read_games(path), start=1):
            examined += 1
            failure = examine(game, number, Path(path).name, player.lower(), labeler)
            if failure is not None:
                found.append(failure)
            if len(found) >= max_failures:
                break
        if len(found) >= max_failures:
            break
    return {
        "examined_games": examined,
        "failures": [asdict(f) for f in found],
        "classes": dict(Counter(f.failure_class for f in found).most_common()),
        "positions_searched": labeler.searched,
    }
