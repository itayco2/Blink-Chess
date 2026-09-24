"""valprobe: val roots with every legal child, the input of VAA (value-mode top-1 agreement with SF).

For each chosen val root: its packed board and Stockfish's best move, and one row per legal move with
the child's packed board (from the child's side to move), the move's vocabulary index, whether the move
is the PV 1 move or an alternative with exactly PV 1's score (child_is_best), and whether the child is
terminal (1: checkmate, the mover wins; 2: a rule draw, stalemate or insufficient material).
Roots are taken in file order (val_roots.bin is permuted), once per position, and only when their label
is deep (depth >= 18) or a mate, the same rule as the loader's default depth filter.
Arrays: root_board [N,32] u1, root_best [N] u2, child_offset [N+1] i8, child_board [M,32] u1,
child_move [M] u2, child_is_best [M] bool, child_terminal [M] i1, plus root_cp, root_mate, root_fen_hash.
"""

import os
from pathlib import Path

import chess
import numpy as np

from blink.board import encode, moves
from blink.board.value import CP_NONE
from blink.data import children
from blink.data.record import NO_MOVE, ROOT_DTYPE

SHALLOW_DEPTH = 18
NOT_TERMINAL, MATED, RULE_DRAW = 0, 1, 2
VAL_ROOTS = "val_roots.bin"
OUTPUT = "valprobe.npz"


def eligible(roots: np.ndarray) -> np.ndarray:
    return (roots["depth"] >= SHALLOW_DEPTH) | (roots["cp"] == CP_NONE)


def first_unique(roots: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Indices of the first n rows where mask holds, skipping repeated fen_hash values."""
    seen: set[int] = set()
    picked = []
    for index in np.flatnonzero(mask):
        key = int(roots["fen_hash"][index])
        if key in seen:
            continue
        seen.add(key)
        picked.append(index)
        if len(picked) == n:
            break
    return np.array(picked, dtype=np.int64)


def select(roots: np.ndarray, n: int) -> np.ndarray:
    return first_unique(roots, eligible(roots), n)


def _terminal(child: chess.Board) -> int:
    if not any(child.generate_legal_moves()):
        return MATED if child.is_check() else RULE_DRAW
    return RULE_DRAW if child.is_insufficient_material() else NOT_TERMINAL


def _best_moves(rec: np.void) -> set[int]:
    """PV 1's move plus every stored alternative whose score equals PV 1's exactly."""
    best = {int(rec["move"])}
    for slot in range(len(rec["alt_move"])):
        same = int(rec["alt_cp"][slot]) == int(rec["cp"]) and int(rec["alt_mate"][slot]) == int(rec["mate"])
        if int(rec["alt_move"][slot]) != NO_MOVE and same:
            best.add(int(rec["alt_move"][slot]))
    return best


def _children_of(rec: np.void) -> tuple[list[np.ndarray], list[int], list[bool], list[int]]:
    board = children.codes_to_board(encode.unpack(rec["board"]))
    best = _best_moves(rec)
    boards, move_ids, is_best, terminal = [], [], [], []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        index = moves.encode_move(board, move)
        boards.append(encode.pack(encode.encode_board(child)))
        move_ids.append(index)
        is_best.append(index in best)
        terminal.append(_terminal(child))
    return boards, move_ids, is_best, terminal


def probe_arrays(roots: np.ndarray) -> dict[str, np.ndarray]:
    """The valprobe arrays for these roots (all of them, in order)."""
    boards, move_ids, is_best, terminal, counts = [], [], [], [], []
    for rec in roots:
        b, m, best, term = _children_of(rec)
        boards += b
        move_ids += m
        is_best += best
        terminal += term
        counts.append(len(m))
    return {
        "root_board": np.ascontiguousarray(roots["board"], dtype=np.uint8).reshape(-1, 32),
        "root_best": roots["move"].astype(np.uint16),
        "root_cp": roots["cp"].astype(np.int16),
        "root_mate": roots["mate"].astype(np.int8),
        "root_fen_hash": roots["fen_hash"].astype(np.uint64),
        "child_offset": np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64),
        "child_board": np.array(boards, dtype=np.uint8).reshape(-1, 32),
        "child_move": np.array(move_ids, dtype=np.uint16),
        "child_is_best": np.array(is_best, dtype=bool),
        "child_terminal": np.array(terminal, dtype=np.int8),
    }


def build(roots: np.ndarray, n: int) -> dict[str, np.ndarray]:
    return probe_arrays(roots[select(roots, n)])


def read_val_roots(pack_dir: Path) -> np.ndarray:
    path = Path(pack_dir) / VAL_ROOTS
    if not path.is_file():
        raise FileNotFoundError(f"no {VAL_ROOTS} in {pack_dir}: run blink data bigpack first")
    return np.fromfile(path, dtype=ROOT_DTYPE)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """np.savez to a .tmp sibling, then os.replace, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def run(pack_dir: Path, n: int, builder=build, output: str = OUTPUT) -> dict:
    """Build from pack_dir/val_roots.bin and write pack_dir/<output>. Fewer than n roots is reported."""
    arrays = builder(read_val_roots(pack_dir), n)
    path = Path(pack_dir) / output
    save_npz(path, arrays)
    return {
        "requested": n,
        "written": len(arrays["root_best"]),
        "children": int(arrays["child_offset"][-1]),
        "path": str(path),
    }
