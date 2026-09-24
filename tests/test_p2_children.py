"""Children: the board after each PV's first move, scored from the child's side, built without python-chess.

The pure-numpy move application must equal python-chess (encode_board after push) on every legal move,
including castling, promotions, en passant captures and double pushes whose en-passant square is legal
only sometimes (a pinned pawn, the rank discovered check).
"""

import random

import chess
import numpy as np
import pytest
from test_p2_fakes import board_line

from blink.board import encode, moves, value
from blink.data import children, parse, rows
from blink.data.record import CHILD_DTYPE, NO_MOVE, ROOT_DTYPE

CRAFTED = [
    "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",  # both castles, both sides
    "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 0 1",
    "r3k2r/1P6/8/8/8/8/6p1/R3K2R w KQkq - 0 1",  # b7xa8 captures a castling rook while promoting
    "r3k2r/1P6/8/8/8/8/6p1/R3K2R b KQkq - 0 1",  # g2xh1 captures a castling rook while promoting
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  # en passant capture available
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",  # ep that would expose the king along the rank
    "7k/8/8/8/1b6/8/2P5/3K4 w - - 0 1",  # c2 pinned: no ep issue, but a pinned pusher
]
PINNED_EP = [
    # d2d4 lands next to the e4 pawn, but e4 is pinned on the long diagonal: no legal ep, no ep code
    ("8/1k6/8/8/4p3/8/3P4/4K2B w - - 0 1", "d2d4", False),
    # d2d4 lands next to e4; taking would clear rank 4 between the a4 rook and the h4 king: illegal
    ("8/8/8/8/R3p2k/8/3P4/4K3 w - - 0 1", "d2d4", False),
    # the same push with nothing pinning: the ep square is legal and is encoded
    ("8/8/8/8/4p2k/8/3P4/4K3 w - - 0 1", "d2d4", True),
    # black pushes c7c5 next to a white b5 pawn that is pinned on the b-file
    ("1r2k3/2p5/8/1P6/8/8/8/1K6 b - - 0 1", "c7c5", False),
    ("4k3/2p5/8/1P6/8/8/8/1K6 b - - 0 1", "c7c5", True),
]


def random_positions(n: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out = [chess.Board(fen) for fen in CRAFTED]
    board = chess.Board()
    while len(out) < n:
        legal = list(board.legal_moves)
        if not legal or board.ply() > 120:
            board = chess.Board()
            continue
        out.append(board.copy(stack=False))
        pawn_pushes = [m for m in legal if board.piece_type_at(m.from_square) == chess.PAWN]
        pool = pawn_pushes if pawn_pushes and rng.random() < 0.4 else legal
        board.push(rng.choice(pool))
    return out


def expected_and_actual(boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    parents, move_ids, expected, cases = [], [], [], []
    for board in boards:
        codes = encode.encode_board(board)
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            parents.append(codes)
            move_ids.append(moves.encode_move(board, move))
            expected.append(encode.encode_board(child))
            cases.append((board.fen(), move.uci()))
    actual = children.apply_moves(np.array(parents), np.array(move_ids))
    return np.array(expected), actual, cases


def test_child_codes_equal_python_chess_on_every_legal_move_of_600_positions():
    expected, actual, cases = expected_and_actual(random_positions(600, seed=3))
    wrong = [cases[i] for i in np.flatnonzero((expected != actual).any(axis=1))[:5]]
    assert not wrong
    assert len(cases) > 15_000


def test_child_epd_matches_python_chess_after_double_pushes_including_a_pinned_pawn():
    for fen, uci, ep_legal in PINNED_EP:
        board = chess.Board(fen)
        child = board.copy()
        child.push_uci(uci)
        assert child.has_legal_en_passant() is ep_legal, fen
        codes = children.apply_moves(
            encode.encode_board(board)[None], np.array([moves.encode_move(board, chess.Move.from_uci(uci))])
        )[0]
        mine = children.codes_to_board(codes)
        want = child if child.turn == chess.WHITE else child.mirror()
        assert mine.epd(en_passant="legal") == want.epd(en_passant="legal"), fen
        assert bool((codes == encode.EP_SQUARE).any()) is ep_legal


def test_every_double_push_child_in_random_games_matches_python_chess_epd():
    boards = random_positions(1500, seed=9)
    checked = 0
    for board in boards:
        for move in board.legal_moves:
            if board.piece_type_at(move.from_square) != chess.PAWN:
                continue
            if abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) != 2:
                continue
            child = board.copy(stack=False)
            child.push(move)
            codes = children.apply_moves(
                encode.encode_board(board)[None], np.array([moves.encode_move(board, move)])
            )[0]
            want = child if child.turn == chess.WHITE else child.mirror()
            assert children.codes_to_board(codes).epd(en_passant="legal") == want.epd(en_passant="legal")
            checked += 1
    assert checked > 1000


def test_codes_to_board_round_trips_through_encode_board():
    for board in random_positions(300, seed=5):
        codes = encode.encode_board(board)
        assert (encode.encode_board(children.codes_to_board(codes)) == codes).all()


def test_a_move_of_an_empty_square_is_refused():
    codes = encode.encode_board(chess.Board())
    empty_from = moves.FROM_TO.index((chess.E4, chess.E5))
    with pytest.raises(ValueError, match="own piece"):
        children.apply_moves(codes[None], np.array([empty_from]))


def _root(board: chess.Board, pvs: list[tuple[str, dict]], depth: int = 30) -> np.ndarray:
    return np.array([parse.parse_line(board_line(board, pvs, depth))], dtype=ROOT_DTYPE)


def test_a_mating_line_child_counts_down_and_a_mated_line_child_does_not():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")  # a1a8 mates
    pvs = [("a1a8", {"mate": 1}), ("a1a7", {"mate": 3}), ("g1f1", {"mate": -2}), ("h2h3", {"cp": 150})]
    kids = children.children_of(_root(board, pvs)).records
    assert kids["mate"].tolist() == [0, -2, 2, 0]
    assert kids["cp"].tolist() == [value.CP_NONE, value.CP_NONE, value.CP_NONE, -150]
    mated_now = children.codes_to_board(encode.unpack(kids["board"][0]))
    assert mated_now.is_checkmate()
    assert value.win_probability_array(kids["cp"][:1], kids["mate"][:1])[0] == 0.0
    root_w = value.win_probability(cp=150)
    assert value.win_probability_array(kids["cp"][3:], kids["mate"][3:])[0] == pytest.approx(1 - root_w)


def test_black_to_move_children_score_from_white_to_move():
    board = chess.Board("4k3/8/8/8/8/8/3p4/7K b - - 0 1")
    kids = children.children_of(_root(board, [("d2d1q", {"cp": -900}), ("e8e7", {"cp": -300})])).records
    assert kids["cp"].tolist() == [-900, -300]  # black was +900 and +300 for itself


def test_a_root_with_one_pv_has_no_children_and_a_root_with_n_pvs_has_n():
    board = chess.Board()
    one = _root(board, [("e2e4", {"cp": 30})])
    three = _root(board, [("e2e4", {"cp": 30}), ("d2d4", {"cp": 25}), ("g1f3", {"cp": 20})])
    got = children.children_of(np.concatenate([one, three]))
    assert len(got.records) == 3 and got.parent.tolist() == [1, 1, 1]
    assert got.records.dtype == CHILD_DTYPE


def test_child_depth_is_the_root_depth_and_fen_hash_is_the_colour_normalised_key_hash():
    board = chess.Board()
    kids = children.children_of(_root(board, [("e2e4", {"cp": 30}), ("d2d4", {"cp": 25})], depth=41))
    child = board.copy()
    child.push_uci("e2e4")
    assert kids.records["depth"].tolist() == [41, 41]
    assert int(kids.records["fen_hash"][0]) == encode.position_hash(child)
    assert int(kids.records["fen_hash"][0]) == encode.position_hash(child.mirror())


def test_a_dropped_alternative_makes_no_child():
    root = _root(chess.Board(), [("e2e4", {"cp": 30}), ("d2d4", {"cp": 25})])
    root["alt_move"][0][0] = NO_MOVE
    assert len(children.children_of(root).records) == 1


def test_pvs_beyond_the_fifth_still_make_children():
    board = chess.Board()
    firsts = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "f2f4", "a2a3"]
    line = board_line(board, [(uci, {"cp": 40 - i}) for i, uci in enumerate(firsts)])
    kids = rows.parse_children(line)
    expected = []
    for uci in firsts:
        child = board.copy()
        child.push_uci(uci)
        expected.append(encode.position_hash(child))
    assert kids["fen_hash"].tolist() == expected
    assert kids["cp"].tolist() == [-(40 - i) for i in range(7)]


def test_duplicate_children_keep_the_deepest_root_label():
    # 1.e4 e5 2.Nf3 and 1.Nf3 e5 2.e4 reach the same child; the deeper root's label must win
    a = chess.Board()
    for uci in ("e2e4", "e7e5"):
        a.push_uci(uci)
    b = chess.Board()
    for uci in ("g1f3", "e7e5"):
        b.push_uci(uci)
    shallow = _root(a, [("g1f3", {"cp": 40}), ("d2d4", {"cp": 35})], depth=22)
    deep = _root(b, [("e2e4", {"cp": 55}), ("d2d4", {"cp": 10})], depth=36)
    kids = children.children_of(np.concatenate([shallow, deep])).records
    kept, dropped = children.dedupe_deepest(kids)
    assert dropped == 1 and len(kept) == 3
    shared = kept[kept["fen_hash"] == kids["fen_hash"][0]]
    assert len(shared) == 1
    assert int(shared["depth"][0]) == 36 and int(shared["cp"][0]) == -55


def test_dedupe_keeps_the_first_label_when_depths_tie():
    recs = np.zeros(4, dtype=CHILD_DTYPE)
    recs["fen_hash"] = [5, 5, 7, 5]
    recs["depth"] = [30, 30, 12, 29]
    recs["cp"] = [1, 2, 3, 4]
    kept, dropped = children.dedupe_deepest(recs)
    assert dropped == 2
    assert kept["cp"].tolist() == [1, 3]
