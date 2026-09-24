"""The sign check: is the value head's sign the right way round? (plan P1 done criteria)

For each ROOT record, value mode picks the child with the highest value for the mover. With the true
sign that value is 1 - the child's win probability (the child is scored from the opponent's side);
with a flipped sign it is the child's win probability itself. A correctly trained model's true-sign
top-1 (agreement with the record's best move) must be at least 3x the flipped-sign top-1. Policy
top-1 on the same rows is reported alongside. R2 and R3 are off: this measures the network only,
offline, so it batches many records into one call and is not a play-time decision.
"""

from pathlib import Path

import chess
import numpy as np

from blink.board import encode, moves
from blink.data.record import ROOT_DTYPE
from blink.play.evaluator import Evaluator

MAX_ROWS = 4096
PASS_RATIO = 3.0


def _piece_of_code() -> dict[int, chess.Piece]:
    table = {}
    for piece_type in chess.PIECE_TYPES:
        table[encode.OWN + piece_type - 1] = chess.Piece(piece_type, chess.WHITE)
        table[encode.OPP + piece_type - 1] = chess.Piece(piece_type, chess.BLACK)
    table[encode.OWN_CASTLING_ROOK] = chess.Piece(chess.ROOK, chess.WHITE)
    table[encode.OPP_CASTLING_ROOK] = chess.Piece(chess.ROOK, chess.BLACK)
    return table


PIECE_OF_CODE = _piece_of_code()


def decode_codes(codes: np.ndarray) -> chess.Board:
    """64 square codes to a board with the side to move as White (a position and its mirror are one input)."""
    board = chess.Board(None)
    board.turn = chess.WHITE
    rights = 0
    for square, code in enumerate(np.asarray(codes, dtype=np.uint8).tolist()):
        piece = PIECE_OF_CODE.get(code)
        if piece is not None:
            board.set_piece_at(square, piece)
        if code in (encode.OWN_CASTLING_ROOK, encode.OPP_CASTLING_ROOK):
            rights |= chess.BB_SQUARES[square]
        elif code == encode.EP_SQUARE:
            board.ep_square = square
    board.castling_rights = rights
    return board


def read_records(path: Path, limit: int | None = None) -> np.ndarray:
    """ROOT records from the front of a shard file (a sequential read, never a seek)."""
    return np.fromfile(path, dtype=ROOT_DTYPE, count=-1 if limit is None else limit)


def _children(codes: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    board = decode_codes(codes)
    pairs = []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        pairs.append((moves.encode_move(board, move), encode.encode_board(child)))
    pairs.sort(key=lambda pair: pair[0])
    return [index for index, _ in pairs], [child for _, child in pairs]


class _Batch:
    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.items: list[tuple[int, list[int], int]] = []  # (label move, child vocab indices, root row)

    def add(self, label: int, root: np.ndarray, indices: list[int], children: list[np.ndarray]) -> None:
        self.items.append((label, indices, len(self.rows)))
        self.rows.append(root)
        self.rows.extend(children)

    def score(self, evaluator: Evaluator, hits: np.ndarray) -> None:
        if not self.rows:
            return
        evaluation = evaluator.evaluate(np.stack(self.rows))
        win = evaluation.win_probability()
        for label, indices, root in self.items:
            legal = np.array(indices)
            child_win = win[root + 1 : root + 1 + len(indices)]
            hits[0] += legal[np.argmax(evaluation.policy_logits[root][legal])] == label
            hits[1] += legal[np.argmax(1.0 - child_win)] == label
            hits[2] += legal[np.argmax(child_win)] == label


def signcheck(evaluator: Evaluator, records: np.ndarray, max_rows: int = MAX_ROWS) -> dict:
    hits = np.zeros(3, dtype=np.int64)  # policy, true sign, flipped sign
    batch, n, skipped = _Batch(), 0, 0
    for record in records:
        root = encode.unpack(record["board"])
        indices, children = _children(root)
        if not indices:
            skipped += 1
            continue
        if batch.rows and len(batch.rows) + 1 + len(children) > max_rows:
            batch.score(evaluator, hits)
            batch = _Batch()
        batch.add(int(record["move"]), root, indices, children)
        n += 1
    batch.score(evaluator, hits)
    policy, true_sign, flipped = (float(h) / n if n else 0.0 for h in hits)
    return {
        "n": n,
        "skipped_terminal": skipped,
        "policy_top1": policy,
        "value_top1_true_sign": true_sign,
        "value_top1_flipped_sign": flipped,
        "ratio": true_sign / flipped if flipped else None,
        "passes_3x": true_sign > 0 and true_sign >= PASS_RATIO * flipped,
    }
