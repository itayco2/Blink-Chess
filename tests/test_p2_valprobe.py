"""valprobe and mateset: val roots with every legal child, the best-move marks and terminal flags."""

import chess
import numpy as np
import pytest
from test_p2_fakes import board_line

from blink.board import encode, moves
from blink.board.value import CP_NONE
from blink.data import mateset, parse, valprobe
from blink.data.record import ROOT_DTYPE

BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
STALEMATE_IN_ONE = "k7/8/1Q6/8/8/8/8/7K w - - 0 1"  # b6c7 stalemates
BARE_KINGS_IN_ONE = "8/8/8/8/8/3k4/1r6/K7 w - - 0 1"  # a1b2 leaves king against king


def root(fen: str, pvs: list[tuple[str, dict]], depth: int = 30) -> np.ndarray:
    board = chess.Board(fen)
    return np.array([parse.parse_line(board_line(board, pvs, depth))], dtype=ROOT_DTYPE)


def probe_of(*roots: np.ndarray) -> dict:
    return valprobe.probe_arrays(np.concatenate(roots))


def children_of(arrays: dict, i: int) -> slice:
    return slice(int(arrays["child_offset"][i]), int(arrays["child_offset"][i + 1]))


def test_valprobe_lists_every_legal_child_with_its_move_index_and_board():
    arrays = probe_of(root(chess.STARTING_FEN, [("e2e4", {"cp": 30})]))
    board = chess.Board()
    legal = list(board.legal_moves)
    got = children_of(arrays, 0)
    assert got.stop - got.start == len(legal) == 20
    want = {}
    for move in legal:
        child = board.copy()
        child.push(move)
        want[moves.encode_move(board, move)] = encode.pack(encode.encode_board(child)).tobytes()
    assert {
        int(m): b.tobytes()
        for m, b in zip(arrays["child_move"][got], arrays["child_board"][got], strict=True)
    } == want
    assert int(arrays["root_best"][0]) == moves.encode_move(board, chess.Move.from_uci("e2e4"))


def test_child_is_best_marks_pv1_and_equal_score_alternatives_only():
    arrays = probe_of(
        root(chess.STARTING_FEN, [("e2e4", {"cp": 30}), ("d2d4", {"cp": 30}), ("g1f3", {"cp": 20})])
    )
    got = children_of(arrays, 0)
    board = chess.Board()
    best = {moves.encode_move(board, chess.Move.from_uci(u)) for u in ("e2e4", "d2d4")}
    marked = {int(m) for m in arrays["child_move"][got][arrays["child_is_best"][got]]}
    assert marked == best


def test_a_mate_score_alternative_is_best_only_with_the_same_mate():
    arrays = probe_of(root(BACK_RANK, [("a1a8", {"mate": 1}), ("a1a7", {"cp": 50})]))
    got = children_of(arrays, 0)
    assert int(arrays["child_is_best"][got].sum()) == 1


def test_terminal_flags_mark_checkmate_and_rule_draws():
    arrays = probe_of(
        root(BACK_RANK, [("a1a8", {"mate": 1})]),
        root(STALEMATE_IN_ONE, [("b6c7", {"cp": 0})]),
        root(BARE_KINGS_IN_ONE, [("a1b2", {"cp": 0})]),
    )
    flags = {}
    for i, (fen, uci) in enumerate(
        ((BACK_RANK, "a1a8"), (STALEMATE_IN_ONE, "b6c7"), (BARE_KINGS_IN_ONE, "a1b2"))
    ):
        board = chess.Board(fen)
        index = moves.encode_move(board, chess.Move.from_uci(uci))
        got = children_of(arrays, i)
        flags[uci] = int(arrays["child_terminal"][got][arrays["child_move"][got] == index][0])
    assert flags == {"a1a8": valprobe.MATED, "b6c7": valprobe.RULE_DRAW, "a1b2": valprobe.RULE_DRAW}
    back_rank = children_of(arrays, 0)  # only a1a8 ends the game; h2h3 and the rest do not
    assert (arrays["child_terminal"][back_rank] != valprobe.NOT_TERMINAL).sum() == 1
    stalemates = arrays["child_terminal"][children_of(arrays, 1)] == valprobe.RULE_DRAW
    assert 1 < stalemates.sum() < len(stalemates)  # Qc7 and every king move stalemate; Qb7+ does not


def test_the_arrays_have_the_interface_names_and_dtypes():
    arrays = probe_of(
        root(chess.STARTING_FEN, [("e2e4", {"cp": 30})]), root(BACK_RANK, [("a1a8", {"mate": 1})])
    )
    want = {
        "root_board": (np.uint8, (2, 32)),
        "root_best": (np.uint16, (2,)),
        "child_offset": (np.int64, (3,)),
        "child_board": (np.uint8, None),
        "child_move": (np.uint16, None),
        "child_is_best": (np.bool_, None),
        "child_terminal": (np.int8, None),
    }
    for name, (dtype, shape) in want.items():
        assert arrays[name].dtype == dtype, name
        if shape:
            assert arrays[name].shape == shape, name
    m = int(arrays["child_offset"][-1])
    assert arrays["child_board"].shape == (m, 32) and len(arrays["child_move"]) == m


def test_valprobe_keeps_deep_or_mate_roots_once_each_in_file_order():
    deep = root(chess.STARTING_FEN, [("e2e4", {"cp": 30})], depth=25)
    shallow = root(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", [("e7e5", {"cp": -30})], depth=12
    )
    shallow_mate = root(BACK_RANK, [("a1a8", {"mate": 1})], depth=5)
    roots = np.concatenate([deep, shallow, deep, shallow_mate])
    picked = valprobe.select(roots, n=10)
    assert picked.tolist() == [0, 3]
    assert valprobe.select(roots, n=1).tolist() == [0]


def test_mateset_takes_mates_in_2_to_5_for_the_side_to_move():
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    roots = np.concatenate(
        [
            root(fen, [("a1a8", {"mate": 1})]),
            root(fen, [("a1a7", {"mate": 3})]),
            root("6k1/5ppp/8/8/8/7P/5PP1/R5K1 w - - 0 1", [("a1a7", {"mate": 5})]),
            root(fen, [("a1a7", {"mate": 6})]),
            root("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1", [("g8f8", {"mate": 3})]),  # black is mated in 3
            root(fen, [("a1a7", {"cp": 900})]),
        ]
    )
    assert mateset.select(roots, n=10).tolist() == [1, 2]
    arrays = mateset.build(roots, n=10)
    assert arrays["mate_in"].tolist() == [3, 5] and arrays["mate_in"].dtype == np.int8
    assert arrays["root_board"].shape == (2, 32)


def test_run_writes_the_npz_beside_the_pack_and_reports_short_supply(tmp_path):
    roots = np.concatenate(
        [root(chess.STARTING_FEN, [("e2e4", {"cp": 30})]), root(BACK_RANK, [("a1a7", {"mate": 3})])]
    )
    roots.tofile(tmp_path / "val_roots.bin")
    summary = valprobe.run(tmp_path, n=5)
    assert summary == {
        "requested": 5,
        "written": 2,
        "children": 20 + len(list(chess.Board(BACK_RANK).legal_moves)),
        "path": str(tmp_path / "valprobe.npz"),
    }
    with np.load(tmp_path / "valprobe.npz") as data:
        assert set(data.files) >= {
            "root_board",
            "root_best",
            "child_offset",
            "child_is_best",
            "child_terminal",
        }
    assert mateset.run(tmp_path, n=5)["written"] == 1
    with np.load(tmp_path / "mateset.npz") as data:
        assert data["mate_in"].tolist() == [3]


def test_a_missing_val_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="val_roots.bin"):
        valprobe.run(tmp_path, n=5)


def test_mate_roots_keep_cp_none():
    arrays = probe_of(root(BACK_RANK, [("a1a8", {"mate": 1})]))
    assert int(arrays["root_cp"][0]) == CP_NONE and int(arrays["root_mate"][0]) == 1
