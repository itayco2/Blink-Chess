"""The P2 row parser: the frozen reference parser plus a Chess960 reason code and every PV's child.

parse.parse_line stays the one definition of a root record. This module only wraps it:
- a row whose castling rights need a king or rook off the standard squares (a Chess960 position), or
  whose best move is a Chess960 castle, is rejected with reason "chess960" instead of being packed as a
  standard position it is not (PF45);
- PVs beyond the fifth, which a root record has no room for, are read here so they still give children.
"""

import re
from collections import Counter
from typing import NamedTuple

import chess
import numpy as np
import orjson

from blink.data import children, parse
from blink.data.record import NUM_ALTERNATIVES, ROOT_DTYPE

CHESS960 = "chess960"
ERROR_SAMPLES = 3
FIRST_EXTRA_PV = 1 + NUM_ALTERNATIVES  # PVs from this index on are not in the root record
_FEN = re.compile(rb'"fen"\s*:\s*"([^\s"]+) [wb] ([^\s"]+)')
# what each standard right needs: (file, piece letter) on the back rank; Shredder letters map onto them
_STANDARD = {
    "K": ((4, "K"), (7, "R")),
    "Q": ((4, "K"), (0, "R")),
    "k": ((4, "k"), (7, "r")),
    "q": ((4, "k"), (0, "r")),
}
_SHREDDER = {"H": "K", "A": "Q", "h": "k", "a": "q"}


def _back_rank(placement: str, white: bool) -> str:
    """The 8 squares of rank 1 (white) or rank 8 (black), a to h, '.' for empty."""
    ranks = placement.split("/")
    row = ranks[-1] if white else ranks[0]
    return "".join("." * int(ch) if ch.isdigit() else ch for ch in row)


def chess960_rights(placement: str, rights: str) -> bool:
    """True when a castling right needs a king or rook that is not on its standard square."""
    if rights == "-":
        return False
    for right in rights:
        needs = _STANDARD.get(_SHREDDER.get(right, right))
        if needs is None:
            return True
        rank = _back_rank(placement, right.isupper())
        if len(rank) != 8 or any(rank[file] != piece for file, piece in needs):
            return True
    return False


def _chess960_castle(line: bytes) -> bool:
    """True when the row's best move is a legal Chess960 castle (king takes own rook)."""
    try:
        row = orjson.loads(line)
        board = chess.Board(row["fen"] + " 0 1", chess960=True)
        move = chess.Move.from_uci(row["evals"][0]["pvs"][0]["line"].split()[0])
    except (orjson.JSONDecodeError, KeyError, IndexError, ValueError):
        return False
    return move in board.legal_moves and board.is_castling(move)


def parse_root(line: bytes) -> np.void:
    """parse.parse_line, with Chess960 rows rejected as "chess960"."""
    fen = _FEN.search(line)
    if fen and chess960_rights(fen.group(1).decode("ascii"), fen.group(2).decode("ascii")):
        raise parse.Rejected(CHESS960, "castling rights off the standard squares")
    try:
        return parse.parse_line(line)
    except parse.Rejected as exc:
        if exc.reason == "illegal_best_move" and _chess960_castle(line):
            raise parse.Rejected(CHESS960, "best move is a Chess960 castle") from exc
        raise


def extra_pvs(line: bytes, root_index: int) -> list[children.ExtraPv]:
    """The usable PVs after the fifth, read with the reference parser's own move and score rules."""
    row = orjson.loads(line)
    pvs = row["evals"][0]["pvs"]
    if len(pvs) <= FIRST_EXTRA_PV:
        return []
    board = chess.Board(row["fen"] + " 0 1")
    out = []
    for pv in pvs[FIRST_EXTRA_PV:]:
        try:
            move = parse._first_move(board, pv)
            cp, mate = parse._score(pv, board.turn)
        except parse.Rejected:
            continue  # the same rule as a dropped alternative in the root record
        out.append(children.ExtraPv(root_index, move, cp, mate))
    return out


class ParsedRows(NamedTuple):
    roots: np.ndarray  # ROOT_DTYPE, in line order
    extras: list[children.ExtraPv]  # PVs beyond the fifth; .root indexes `roots`
    rejects: dict[str, int]
    errors: dict[str, int]  # anything but a documented Rejected: a parser bug, counted and sampled
    error_samples: list[str]


def parse_rows(lines: list[bytes]) -> ParsedRows:
    records = np.empty(len(lines), dtype=ROOT_DTYPE)
    kept = 0
    extras: list[children.ExtraPv] = []
    rejects: Counter = Counter()
    errors: Counter = Counter()
    samples: list[str] = []
    for line in lines:
        try:
            rec = parse_root(line)
            if rec["npv"] > FIRST_EXTRA_PV:
                extras.extend(extra_pvs(line, kept))
        except parse.Rejected as exc:
            rejects[exc.reason] += 1
            continue
        except Exception as exc:  # noqa: BLE001 - counted, sampled and reported, never dropped silently
            errors[type(exc).__name__] += 1
            if len(samples) < ERROR_SAMPLES:
                samples.append(f"{type(exc).__name__}: {exc} | {line[:160].decode('utf-8', 'replace')}")
            continue
        records[kept] = rec
        kept += 1
    return ParsedRows(records[:kept].copy(), extras, dict(rejects), dict(errors), samples)


def parse_children(line: bytes) -> np.ndarray:
    """The CHILD_DTYPE records of one raw line (none for a single-PV row). Raises Rejected like parse_root."""
    root = np.array([parse_root(line)], dtype=ROOT_DTYPE)
    extras = extra_pvs(line, 0) if root["npv"][0] > FIRST_EXTRA_PV else []
    return children.children_of(root, extras).records
