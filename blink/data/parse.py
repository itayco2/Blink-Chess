"""One raw eval DB line -> one root record (or a Rejected with a reason code).

This is the reference parser: it uses python-chess for legality and castling, so it is exact but
slow. The P2 bulk packer may replace the hot path, and must match this one on every fixture row.
"""

import chess
import numpy as np
import orjson

from blink.board import encode, moves, value
from blink.data.record import NO_MOVE, NUM_ALTERNATIVES, ROOT_DTYPE

INT8_MAX = 127
INT16_MAX = 32767


class Rejected(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _score(pv: dict, turn: chess.Color) -> tuple[int, int]:
    """(cp, mate) from the side to move's view; cp is CP_NONE when the score is a mate."""
    sign = 1 if turn == chess.WHITE else -1
    if "mate" in pv:
        mate = max(-INT8_MAX, min(INT8_MAX, sign * int(pv["mate"])))
        return value.CP_NONE, mate
    if "cp" in pv:
        return max(-INT16_MAX, min(INT16_MAX, sign * int(pv["cp"]))), 0
    raise Rejected("no_score")


def _first_move(board: chess.Board, pv: dict) -> int:
    line = pv.get("line", "").split()
    if not line:
        raise Rejected("no_line")
    try:
        move = chess.Move.from_uci(moves.uci960_to_standard(board, line[0]))
    except ValueError as exc:
        raise Rejected("bad_uci", line[0]) from exc
    if move not in board.legal_moves:
        raise Rejected("illegal_best_move", line[0])
    return moves.encode_move(board, move)


def parse_line(line: bytes) -> np.void:
    try:
        row = orjson.loads(line)
        board = chess.Board(row["fen"] + " 0 1")
    except (orjson.JSONDecodeError, KeyError, ValueError) as exc:
        raise Rejected("bad_row", str(exc)[:80]) from exc
    evals = row.get("evals") or []
    if not evals or not evals[0].get("pvs"):
        raise Rejected("no_evals")
    top = evals[0]
    pvs = top["pvs"]

    rec = np.zeros((), dtype=ROOT_DTYPE)
    rec["board"] = encode.pack(encode.encode_board(board))
    rec["move"] = _first_move(board, pvs[0])
    rec["cp"], rec["mate"] = _score(pvs[0], board.turn)
    rec["depth"] = min(255, int(top.get("depth", 0)))
    rec["npv"] = min(255, len(pvs))
    rec["alt_move"] = NO_MOVE
    for slot, pv in enumerate(pvs[1 : 1 + NUM_ALTERNATIVES]):
        try:
            rec["alt_move"][slot] = _first_move(board, pv)
            rec["alt_cp"][slot], rec["alt_mate"][slot] = _score(pv, board.turn)
        except Rejected:
            rec["alt_move"][slot] = NO_MOVE
    rec["fen_hash"] = encode.key_hash(bytes(rec["board"]))
    return rec
