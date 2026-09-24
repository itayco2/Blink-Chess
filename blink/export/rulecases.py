"""site/tests/rules.json: what NSC-1's R2 and R3 say about every child, from python-chess (plan section 2).

The page plays one look with the bot's rule checks, ported to site/rules.js. This module is the Python
reference for that port: for each case (a FEN plus the moves played since), every checkmating child (R2)
and every rule-draw child with its reason (R3), by vocabulary index. It follows blink/play/rules.py:
- a mate is never a draw; then stalemate, insufficient material, a halfmove clock of 100, a third
  occurrence, in that order;
- a position's key is its placement, side to move, clean castling rights and the en-passant square
  only when an en-passant capture is legal;
- occurrences are counted from the game's own history, never with can_claim_threefold_repetition.
test_rule_cases_agree_with_the_play_areas_rules_module checks this against blink.play.rules itself.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from blink.board import moves

FORMAT_VERSION = 1
DRAW_DELTA = 0.10  # pre-registered in NSC-1: policy mode acts on draws only when |v(P) - 0.5| > delta
HALFMOVE_DRAW = 100

RepetitionKey = tuple[str, bool, int, int | None]


@dataclass(frozen=True)
class RuleCase:
    name: str
    fen: str
    moves: tuple[str, ...] = ()


def _case(name: str, fen: str, played: str = "") -> RuleCase:
    return RuleCase(name, fen, tuple(played.split()))


START = chess.STARTING_FEN
RULE_CASES: tuple[RuleCase, ...] = (
    _case("nothing applies", "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    _case("mate now on the back rank", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    _case("mate now for black", "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"),
    _case("two mates by promotion", "7k/5P2/6K1/8/8/8/8/8 w - - 0 1"),
    _case("scholar's mate from the start", START, "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6"),
    _case("a queen move that stalemates", "7k/8/8/6Q1/8/8/8/K7 w - - 0 1"),
    _case("king takes the last rook", "8/8/8/4k3/8/8/3r4/4K3 w - - 0 1"),
    _case("bishops left on one colour", "8/8/8/4k3/8/2b5/3N4/4K1B1 b - - 0 1"),
    _case("knight against knight is not a draw", "8/8/8/4k3/8/8/3r4/2n1K1N1 w - - 0 1"),
    _case("the hundredth halfmove, except a mate", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80"),
    _case("black repeats the start a third time", START, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1"),
    _case(
        "an unusable en-passant square is not part of the key",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "e2e4 e8e7 e1d1 e7e8 d1e1 e8e7 e1d1 e7e8",
    ),
    _case(
        "a capturable en-passant square is part of the key",
        "4k3/8/8/8/5p2/8/4P3/4K3 w - - 0 1",
        "e2e4 e8e7 e1d1 e7e8 d1e1 e8e7 e1d1 e7e8",
    ),
    _case(
        "castling rights are part of the key",
        "r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1",
        "h1h2 a8a7 h2h1 a7a8 h1h2 a8a7 h2h1",
    ),
)


def repetition_key(board: chess.Board) -> RepetitionKey:
    """Placement, side to move, clean castling rights, and the ep square only when a capture is legal."""
    ep = board.ep_square if board.has_legal_en_passant() else None
    return (board.board_fen(), board.turn, board.clean_castling_rights(), ep)


def history_counts(board: chess.Board) -> Counter:
    """How often each position occurred, the current one included, back to the last irreversible move."""
    walker = board.copy()
    counts = Counter([repetition_key(walker)])
    for _ in range(min(board.halfmove_clock, len(board.move_stack))):
        walker.pop()
        counts[repetition_key(walker)] += 1
    return counts


def rule_draw(child: chess.Board, counts: Counter) -> str | None:
    """R3: why the child is a draw by rule, or None. A mate is never a draw."""
    if child.is_checkmate():
        return None
    if child.is_stalemate():
        return "stalemate"
    if child.is_insufficient_material():
        return "insufficient material"
    if child.halfmove_clock >= HALFMOVE_DRAW:
        return "fifty-move rule"
    if counts[repetition_key(child)] + 1 >= 3:
        return "threefold repetition"
    return None


def replay(case: RuleCase) -> chess.Board:
    board = chess.Board(case.fen)
    for uci in case.moves:
        board.push_uci(uci)
    return board


def case_entry(case: RuleCase) -> dict[str, Any]:
    board = replay(case)
    counts = history_counts(board)
    mates, draws = [], []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        index = moves.encode_move(board, move)
        if child.is_checkmate():
            mates.append([index, move.uci()])
        reason = rule_draw(child, counts)
        if reason is not None:
            draws.append([index, move.uci(), reason])
    return {
        "name": case.name,
        "fen": case.fen,
        "moves": list(case.moves),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "legal": board.legal_moves.count(),
        "mates": sorted(mates),
        "draws": sorted(draws),
    }


def build(cases: tuple[RuleCase, ...] = RULE_CASES) -> dict[str, Any]:
    return {
        "about": (
            "NSC-1 R2 (mate now) and R3 (rule draws) for every child, from python-chess, written by "
            "`blink export rules`. site/tests/rules.test.mjs checks site/rules.js against it."
        ),
        "version": FORMAT_VERSION,
        "draw_delta": DRAW_DELTA,
        "halfmove_draw": HALFMOVE_DRAW,
        "cases": [case_entry(case) for case in cases],
    }


def render(data: dict[str, Any]) -> str:
    """One case per line, so a diff shows which case moved."""
    head = {key: item for key, item in data.items() if key != "cases"}
    rows = ",\n".join("    " + json.dumps(entry, separators=(",", ":")) for entry in data["cases"])
    return json.dumps(head, indent=2)[:-2] + ',\n  "cases": [\n' + rows + "\n  ]\n}\n"


def write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render(build()))
    return path
