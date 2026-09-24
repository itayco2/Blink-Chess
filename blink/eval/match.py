"""In-process matches between two agents, with python-chess as the arbiter.

Openings are played in book order, each once per colour (games 2k and 2k+1 share opening k).
A game ends by checkmate, or is drawn by rule (stalemate, insufficient material, halfmove clock 100,
a third repetition), or is adjudicated a draw after 600 engine plies (book plies excluded; this
mirrors fastchess `-maxmoves 300` and Lichess's forced draw). There is no resignation and no win
adjudication. An illegal move or a crash loses the game; a NoSearchViolation stops the match.

Every engine move carries a fastchess-style comment `{<score>/1 <seconds>s, n=<rows>}`, so
`blink audit no-search` reads these PGNs exactly as it reads fastchess's.
"""

import datetime
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn

from blink.eval.books import Opening
from blink.play.agents import Agent, Decision
from blink.play.budget import NoSearchViolation
from blink.uci import win_to_cp

MAX_ENGINE_PLIES = 600
HALFMOVE_DRAW = 100
DRAW = "1/2-1/2"


@dataclass(frozen=True)
class GameRecord:
    white: str
    black: str
    result: str
    termination: str  # the PGN Termination tag: normal | adjudication | illegal move | abandoned
    reason: str  # checkmate, stalemate, threefold repetition, ..., or what went wrong
    engine_plies: int
    opening: int
    illegal_by: str | None = None
    crashed_by: str | None = None


def rule_outcome(board: chess.Board) -> tuple[str, str] | None:
    """(result, reason) when the game is over by rule, else None. The arbiter, never an agent."""
    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), "checkmate"
    if board.is_stalemate():
        return DRAW, "stalemate"
    if board.is_insufficient_material():
        return DRAW, "insufficient material"
    if board.halfmove_clock >= HALFMOVE_DRAW:
        return DRAW, "fifty-move rule"
    if board.is_repetition(3):
        return DRAW, "threefold repetition"
    return None


def move_comment(decision: Decision, seconds: float) -> str:
    if decision.mate_now:
        score = "+M1"
    elif decision.win is not None:
        score = f"{win_to_cp(decision.win) / 100:+.2f}"
    else:
        score = "0.00"
    return f"{score}/1 {seconds:.3f}s, n={decision.n_rows}"


def _loss_for(color: chess.Color) -> str:
    return "0-1" if color == chess.WHITE else "1-0"


def _start(opening: Opening) -> tuple[chess.Board, chess.pgn.Game, chess.pgn.GameNode]:
    board = chess.Board(opening.fen)
    game = chess.pgn.Game()
    if opening.fen != chess.STARTING_FEN:
        game.setup(board)
    node: chess.pgn.GameNode = game
    for uci in opening.moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move, comment="book")
        board.push(move)
    return board, game, node


def play_game(
    white: Agent, black: Agent, opening: Opening, game_id: str, max_plies: int = MAX_ENGINE_PLIES
) -> tuple[chess.pgn.Game, GameRecord]:
    board, game, node = _start(opening)
    plies, extra = 0, {}
    while True:
        over = rule_outcome(board)
        if over is not None:
            result, termination, reason = over[0], "normal", over[1]
            break
        if plies >= max_plies:
            result, termination, reason = DRAW, "adjudication", f"{max_plies} engine plies"
            break
        player = white if board.turn == chess.WHITE else black
        started = time.perf_counter()
        try:
            decision = player.choose(board, game=game_id)
        except NoSearchViolation:
            raise
        except Exception as exc:  # a crash loses this game; the match goes on and counts it
            result, termination, reason = _loss_for(board.turn), "abandoned", f"crash: {exc!r}"
            extra = {"crashed_by": player.name}
            break
        if decision.move not in board.legal_moves:
            result, termination = _loss_for(board.turn), "illegal move"
            reason, extra = f"illegal move {decision.move.uci()}", {"illegal_by": player.name}
            break
        node = node.add_variation(
            decision.move, comment=move_comment(decision, time.perf_counter() - started)
        )
        board.push(decision.move)
        plies += 1
    record = GameRecord(white.name, black.name, result, termination, reason, plies, opening.number, **extra)
    _set_headers(game, record, len(opening.moves) + plies)
    return game, record


def _set_headers(game: chess.pgn.Game, record: GameRecord, ply_count: int) -> None:
    game.headers["Event"] = "Blink match"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["White"] = record.white
    game.headers["Black"] = record.black
    game.headers["Result"] = record.result
    game.headers["Termination"] = record.termination
    game.headers["EndReason"] = record.reason
    game.headers["PlyCount"] = str(ply_count)
    game.headers["BookIndex"] = str(record.opening)


def _score_for_a(record: GameRecord, a_is_white: bool) -> float:
    if record.result == DRAW:
        return 0.5
    white_won = record.result == "1-0"
    return 1.0 if white_won == a_is_white else 0.0


def summarize(records: Sequence[tuple[GameRecord, bool]], a_name: str, b_name: str) -> dict:
    scores = [_score_for_a(record, a_white) for record, a_white in records]
    return {
        "a": a_name,
        "b": b_name,
        "games": len(records),
        "a_wins": scores.count(1.0),
        "draws": scores.count(0.5),
        "a_losses": scores.count(0.0),
        "a_score": sum(scores) / len(scores) if scores else 0.0,
        "illegal_moves": sum(record.illegal_by is not None for record, _ in records),
        "crashes": sum(record.crashed_by is not None for record, _ in records),
        "adjudications": sum(record.termination == "adjudication" for record, _ in records),
        "reasons": dict(Counter(record.reason for record, _ in records)),
        "records": [asdict(record) for record, _ in records],
    }


def run_match(
    a: Agent,
    b: Agent,
    openings: Sequence[Opening],
    games: int,
    pgn_path: Path,
    max_plies: int = MAX_ENGINE_PLIES,
) -> dict:
    """Play `games` games, A as White in even games; append each game to pgn_path as it finishes."""
    if games > 2 * len(openings):
        raise ValueError(f"{games} games need {(games + 1) // 2} openings, got {len(openings)}")
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(games):
        a_white = index % 2 == 0
        white, black = (a, b) if a_white else (b, a)
        game, record = play_game(white, black, openings[index // 2], f"g{index + 1}", max_plies)
        game.headers["Round"] = str(index + 1)
        with open(pgn_path, "a", encoding="utf-8") as handle:
            print(game, file=handle, end="\n\n")
        records.append((record, a_white))
    return summarize(records, a.name, b.name)
