"""Helpers for the P2 tests: hand-written eval DB lines with exact scores, and a tiny pzstd source.

Scores are given as the eval DB stores them, from White's point of view. The two tests at the bottom
check the helpers themselves against the reference parser.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import chess
import orjson
from data_fakes import lines_text, write_pzstd

from blink.board import encode, value
from blink.data import grouped, parse, split


def db_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])  # the eval DB drops the move counters


def db_uci(board: chess.Board, move: chess.Move) -> str:
    """The eval DB writes castling as king-takes-rook (e1h1)."""
    if board.is_castling(move):
        rook_file = 7 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 0
        rook = chess.square(rook_file, chess.square_rank(move.from_square))
        return chess.square_name(move.from_square) + chess.square_name(rook)
    return move.uci()


def db_line(fen: str, pvs: list[tuple[str, dict]], depth: int = 30) -> bytes:
    """One eval DB line: pvs are (raw uci first move, {"cp": x} or {"mate": m}) in White's view."""
    rows = [{**score, "line": uci} for uci, score in pvs]
    return orjson.dumps({"fen": fen, "evals": [{"pvs": rows, "knodes": 1000, "depth": depth}]})


def board_line(board: chess.Board, pvs: list[tuple[str, dict]], depth: int = 30) -> bytes:
    """db_line for a python-chess board, with moves given in standard uci (castling as e1g1)."""
    raw = [(db_uci(board, chess.Move.from_uci(uci)), score) for uci, score in pvs]
    return db_line(db_fen(board), raw, depth)


def write_source(path: Path, lines: list[bytes], frame_bytes: int = 4_000) -> Path:
    write_pzstd(path, lines_text(lines), frame_bytes)
    return path


def test_db_line_round_trips_through_the_reference_parser():
    board = chess.Board()
    rec = parse.parse_line(board_line(board, [("e2e4", {"cp": 30}), ("d2d4", {"cp": 25})], depth=33))
    assert int(rec["cp"]) == 30 and int(rec["depth"]) == 33 and int(rec["npv"]) == 2


def test_board_line_writes_castling_as_king_takes_rook():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    line = board_line(board, [("e1g1", {"mate": 3})])
    assert b'"line":"e1h1"' in line
    assert int(parse.parse_line(line)["cp"]) == value.CP_NONE


def test_a_found_transposition_really_shares_its_child():
    t = transposition("train", "train", seed=21)
    assert encode.position_hash(after(t.first, t.first_move)) == encode.position_hash(t.child)
    assert encode.position_hash(after(t.second, t.second_move)) == encode.position_hash(t.child)
    assert encode.position_hash(t.first) != encode.position_hash(t.second)


# --- leakage scenarios for the bigpack tests ---------------------------------------------------------

GROUPED_SALT = 0
MAX_WALK = 200_000


def split_of_board(board: chess.Board, salt: int = GROUPED_SALT) -> str:
    """The split bigpack gives a root: val/test_iid by hash first, then test_grouped by group."""
    name = split.split_of(encode.position_hash(board))
    if name == "train" and grouped.selected(encode.pack(encode.encode_board(board))[None], salt)[0]:
        return "test_grouped"
    return name


def after(board: chess.Board, move: chess.Move) -> chess.Board:
    child = board.copy(stack=False)
    child.push(move)
    return child


def walk(seed: int) -> Iterator[chess.Board]:
    """Positions of random games from ply 6 to 60 (so walks with different seeds never share a start)."""
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(MAX_WALK):
        legal = list(board.legal_moves)
        if not legal or board.ply() > 60:
            board = chess.Board()
            continue
        if board.ply() >= 6:
            yield board.copy(stack=False)
        board.push(rng.choice(legal))
    raise AssertionError(f"no position found in {MAX_WALK} plies")


def _quiet(board: chess.Board) -> list[chess.Move]:
    """Non-capturing, non-checking moves of knights, bishops, rooks and queens: they commute."""
    pieces = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    return [
        m
        for m in board.legal_moves
        if board.piece_type_at(m.from_square) in pieces
        and not board.is_capture(m)
        and not board.gives_check(m)
    ]


@dataclass(frozen=True)
class Transposition:
    first: chess.Board  # a root whose move `first_move` reaches `child`
    first_move: chess.Move
    second: chess.Board  # another root whose `second_move` reaches the same child
    second_move: chess.Move
    child: chess.Board


def _second_root(x, w1, w2, b1, r1, want_second) -> Transposition | None:
    if w2 == w1 or w2.from_square == w1.from_square or w2.to_square == w1.to_square:
        return None
    if w2 not in r1.legal_moves:
        return None
    x2 = after(x, w2)
    if b1 not in x2.legal_moves:
        return None
    r2 = after(x2, b1)
    if w1 not in r2.legal_moves or split_of_board(r2) != want_second:
        return None
    c1, c2 = after(r1, w2), after(r2, w1)
    if encode.position_hash(c1) != encode.position_hash(c2):
        return None
    return Transposition(r1, w2, r2, w1, c1)


def transposition(want_first: str, want_second: str, seed: int) -> Transposition:
    """Two roots in the wanted splits that share a child (X + w1 + b1 + w2 == X + w2 + b1 + w1)."""
    for x in walk(seed):
        quiet = _quiet(x)
        for w1 in quiet:
            x1 = after(x, w1)
            for b1 in _quiet(x1):
                r1 = after(x1, b1)
                if split_of_board(r1) != want_first:
                    continue
                for w2 in quiet:
                    if found := _second_root(x, w1, w2, b1, r1, want_second):
                        return found
    raise AssertionError("unreachable")


def root_where(seed: int, want: str = "train") -> chess.Board:
    return next(b for b in walk(seed) if split_of_board(b) == want)


def two_pv_line(
    board: chess.Board, first: chess.Move, depth: int = 30, cp: int = 30
) -> tuple[bytes, chess.Move]:
    """A 2-PV line whose PV 1 is `first`; returns it with the PV 2 move it chose."""
    second = next(m for m in board.legal_moves if m != first)
    sign = 1 if board.turn == chess.WHITE else -1  # the DB stores White's view
    pvs = [(first.uci(), {"cp": sign * cp}), (second.uci(), {"cp": sign * (cp - 20)})]
    return board_line(board, pvs, depth), second


def single_pv_line(board: chess.Board) -> bytes:
    return board_line(board, [(next(iter(board.legal_moves)).uci(), {"cp": 0})])


@dataclass
class LeakWorld:
    lines: list[bytes] = field(default_factory=list)
    blocklist: list[int] = field(default_factory=list)
    expect: dict[str, int] = field(default_factory=dict)


def _eval_transposition(world: LeakWorld, name: str, want: str, seed: int) -> None:
    t = transposition(want, "train", seed)
    line, second = two_pv_line(t.first, t.first_move)
    world.lines += [line, two_pv_line(t.second, t.second_move)[0]]
    world.expect[f"{name}_shared_child"] = encode.position_hash(t.child)
    other = after(t.first, second)
    if split_of_board(other) == "train" and any(other.legal_moves):  # a DB root equal to this eval child
        world.lines.append(single_pv_line(other))
        world.expect[f"{name}_child_as_train_root"] = encode.position_hash(other)


def _train_child(seed: int) -> tuple[chess.Board, chess.Move, chess.Board]:
    for root in walk(seed):
        if split_of_board(root) != "train":
            continue
        for move in root.legal_moves:
            child = after(root, move)
            if split_of_board(child) == "train" and any(child.legal_moves):
                return root, move, child
    raise AssertionError("unreachable")


def _grouped_child(world: LeakWorld, seed: int) -> None:
    for root in walk(seed):
        if split_of_board(root) != "train":
            continue
        for move in root.legal_moves:
            child = after(root, move)
            if grouped.selected(encode.pack(encode.encode_board(child))[None], GROUPED_SALT)[0]:
                world.lines.append(two_pv_line(root, move)[0])
                world.expect["grouped_child"] = encode.position_hash(child)
                return


def leak_world() -> LeakWorld:
    """Crafted rows for each leakage rule, plus two rejects. Scenario seeds are fixed, so it is stable."""
    world = LeakWorld()
    _eval_transposition(world, "val", "val", seed=1)
    _eval_transposition(world, "iid", "test_iid", seed=2)
    dup = transposition("train", "train", seed=3)
    world.lines.append(two_pv_line(dup.first, dup.first_move, depth=20, cp=40)[0])
    world.lines.append(two_pv_line(dup.second, dup.second_move, depth=35, cp=55)[0])
    world.expect["dup_child"] = encode.position_hash(dup.child)
    root, move, child = _train_child(seed=4)  # a child that is also a DB root
    world.lines += [two_pv_line(root, move)[0], single_pv_line(child)]
    world.expect["child_is_root"] = encode.position_hash(child)
    root, move, child = _train_child(seed=5)  # a blocklisted child
    world.lines.append(two_pv_line(root, move)[0])
    world.blocklist.append(encode.position_hash(child))
    world.expect["blocked_child"] = encode.position_hash(child)
    mirrored = root_where(seed=6)  # a root blocked through its colour mirror
    world.lines.append(two_pv_line(mirrored, next(iter(mirrored.legal_moves)))[0])
    world.blocklist.append(encode.position_hash(mirrored.mirror()))
    world.expect["mirror_root"] = encode.position_hash(mirrored)
    root, move, child = _train_child(seed=7)  # a child blocked through its colour mirror
    world.lines.append(two_pv_line(root, move)[0])
    world.blocklist.append(encode.position_hash(child.mirror()))
    world.expect["mirror_child"] = encode.position_hash(child)
    _grouped_child(world, seed=8)
    held_out = root_where(seed=9, want="test_grouped")
    world.lines.append(two_pv_line(held_out, next(iter(held_out.legal_moves)))[0])
    world.expect["grouped_root"] = encode.position_hash(held_out)
    world.lines.append(db_line("bqnbrkrn/pppppppp/8/8/8/8/PPPPPPPP/BQNBRKRN w KQkq -", [("e2e4", {"cp": 9})]))
    world.lines.append(b"not json")
    return world
