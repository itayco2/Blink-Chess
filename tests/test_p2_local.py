"""P2 on the real eval DB prefix and the skeleton probe (build machine only; skipped without them)."""

import chess
import numpy as np
import orjson
import pytest

from blink import paths
from blink.board import encode, moves
from blink.data import canon, frames, grouped, parse, rows, zst
from blink.data.record import ROOT_DTYPE

PREFIX = paths.home() / "data" / "raw" / "prefix-342M.jsonl.zst"
SKELETON = paths.home() / "data" / "skeleton"
needs_prefix = pytest.mark.skipif(not PREFIX.is_file(), reason=f"needs {PREFIX}")
needs_skeleton = pytest.mark.skipif(not (SKELETON / "manifest.json").is_file(), reason=f"needs {SKELETON}")


def first_frame_lines() -> list[bytes]:
    frame = next(iter(zst.FrameReader(PREFIX, 1)))
    return frames.split_text(zst.decompress_frame(frame.data)).lines


def _python_chess_children(line: bytes) -> list[int]:
    row = orjson.loads(line)
    board = chess.Board(row["fen"] + " 0 1")
    pvs = row["evals"][0]["pvs"]
    if len(pvs) < 2:
        return []
    out = []
    for pv in pvs:
        if "cp" not in pv and "mate" not in pv:
            continue
        move = chess.Move.from_uci(moves.uci960_to_standard(board, pv["line"].split()[0]))
        if move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            out.append(encode.position_hash(child))
    return out


@pytest.mark.local
@needs_prefix
def test_real_children_equal_python_chess_on_3000_prefix_lines():
    checked = 0
    for line in first_frame_lines()[:3000]:
        try:
            kids = rows.parse_children(line)
        except parse.Rejected:
            continue
        assert kids["fen_hash"].tolist() == _python_chess_children(line)
        checked += len(kids)
    assert checked > 3000


@pytest.mark.local
@needs_prefix
def test_real_canonical_epd_equals_python_chess_epd_on_3000_prefix_lines():
    checked = 0
    for line in first_frame_lines()[:3000]:
        try:
            record = rows.parse_root(line)
        except parse.Rejected:
            continue
        board = chess.Board(orjson.loads(line)["fen"] + " 0 1")
        assert canon.canonical_epd(encode.unpack(record["board"]), board.turn) == board.epd(), line[:80]
        checked += 1
    assert checked > 2900


@pytest.mark.local
@needs_prefix
def test_no_real_row_is_left_as_an_illegal_best_move_once_chess960_is_its_own_reason():
    got = rows.parse_rows(first_frame_lines()[:20_000])
    assert got.errors == {}
    assert set(got.rejects) <= {"chess960"} and got.rejects.get("chess960", 0) > 0


@pytest.mark.local
@needs_skeleton
def test_the_skeleton_probe_chooses_salt_1_and_salt_0_would_hold_out_a_giant_group():
    shards = sorted(SKELETON.glob("*.bin"))
    boards = np.concatenate([np.fromfile(p, dtype=ROOT_DTYPE) for p in shards])["board"]
    choice = grouped.choose_salt(boards)
    assert choice.salt == 1 and choice.largest_selected_roots <= choice.giant_threshold_roots
    assert 0.05 <= 100 * choice.selected_roots / choice.probe_roots <= 0.15
    zero = grouped.group_hashes(boards, 0)
    held = zero[zero % np.uint64(1000) == np.uint64(7)]
    assert np.unique(held, return_counts=True)[1].max() > choice.giant_threshold_roots
