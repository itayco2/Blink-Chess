"""The leakage blocklist: every test position is removed from training before any split is drawn."""

import io

import chess
import numpy as np

from blink.board import encode
from blink.data import blocklist

DM_CSV = (
    "PuzzleId,Rating,PGN,Solution,FEN,Moves\n"
    "00MTG,669,1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Nc3 Be7 5. O-O O-O 6. h3 h6 7. d4 exd4 8. Nxd4 Nxd4 "
    "9. Qxd4 d6 10. f4 Be6 11. Nd5 Nxd5 12. exd5 Bf5 13. g4 Bh7 14. Qd2 Qd7 15. Qg2 Rae8 16. a4 a6 "
    "17. Ra3 Bh4 18. b4 b5 19. axb5 axb5 20. Bd3 Bxd3 21. Rxd3 Re7 22. g5 hxg5 23. fxg5 Rfe8 "
    "24. Bd2 Re2 25. Qf3 Qe7 26. Qh5,Bf2+ Rxf2 Rxf2 Kxf2,"
    "4r1k1/2p1qpp1/3p4/1p1P2PQ/1P5b/3R3P/2PBr3/5RK1 b - - 6 26,h4f2 f1f2 e2f2 g1f2\n"
)

LICHESS_CSV = (
    "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,DailyDate\n"
    "00008,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,"
    "1797,76,95,10183,crushing,https://lichess.org/787zsVup/black#48,,\n"
    "0000D,5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27,d3d6 f8d8 d6d8 f6d8,"
    "1468,75,96,37410,advantage,https://lichess.org/F8M8OS71#53,,\n"
    "0001X,5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27,d3d6 f8d8 d6d8 f6d8,"
    "1468,95,96,37410,advantage,https://lichess.org/F8M8OS71#53,,\n"
)


def line_positions(fen: str, uci_moves: str) -> list[chess.Board]:
    board = chess.Board(fen)
    out = [board.copy()]
    for uci in uci_moves.split():
        board.push_uci(uci)
        out.append(board.copy())
    return out


def test_every_position_on_a_puzzle_line_is_blocked_for_both_sides():
    hashes = blocklist.dm_puzzle_hashes(io.StringIO(DM_CSV), include_source_games=False)
    line = line_positions("4r1k1/2p1qpp1/3p4/1p1P2PQ/1P5b/3R3P/2PBr3/5RK1 b - - 6 26", "h4f2 f1f2 e2f2 g1f2")
    assert {encode.position_hash(b) for b in line} <= hashes
    assert {b.turn for b in line} == {chess.WHITE, chess.BLACK}


def test_a_colour_mirrored_blocklisted_position_is_blocked_too():
    hashes = blocklist.dm_puzzle_hashes(io.StringIO(DM_CSV), include_source_games=False)
    mirror = chess.Board("4r1k1/2p1qpp1/3p4/1p1P2PQ/1P5b/3R3P/2PBr3/5RK1 b - - 6 26").mirror()
    assert encode.position_hash(mirror) in hashes


def test_game_positions_before_ply_16_are_never_blocked():
    hashes = blocklist.dm_puzzle_hashes(io.StringIO(DM_CSV), include_source_games=True)
    board = chess.Board()
    for san in [
        "e4",
        "e5",
        "Nf3",
        "Nc6",
        "Bc4",
        "Nf6",
        "Nc3",
        "Be7",
        "O-O",
        "O-O",
        "h3",
        "h6",
        "d4",
        "exd4",
        "Nxd4",
        "Nxd4",
    ]:
        assert encode.position_hash(board) not in hashes
        board.push_san(san)
    assert encode.position_hash(board) in hashes  # the position after ply 16 is blocked


def test_lichess_band_puzzles_filter_on_deviation_and_skip_deepmind_ids():
    rows = blocklist.select_lichess_puzzles(
        io.StringIO(LICHESS_CSV),
        bands=[(1400, 1600), (1600, 1800), (1800, 2000)],
        per_band=5,
        max_deviation=80,
        min_plays=1000,
        exclude_ids={"00008"},
    )
    assert [r.puzzle_id for r in rows] == ["0000D"]  # 00008 is DeepMind's; 0001X has RD 95


def test_lichess_puzzle_lines_include_the_setup_move_and_every_later_position():
    rows = blocklist.select_lichess_puzzles(
        io.StringIO(LICHESS_CSV),
        bands=[(1400, 1600)],
        per_band=5,
        max_deviation=80,
        min_plays=1000,
        exclude_ids=set(),
    )
    hashes = blocklist.puzzle_line_hashes(rows)
    expected = line_positions("5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27", "d3d6 f8d8 d6d8 f6d8")
    assert {encode.position_hash(b) for b in expected} == hashes


def test_the_band_sample_is_deterministic_and_capped_per_band():
    many = (
        LICHESS_CSV.splitlines()[0]
        + "\n"
        + "".join(
            f"P{i:04d},5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27,d3d6 f8d8,1500,50,90,2000,x,u,,\n"
            for i in range(40)
        )
    )
    first = blocklist.select_lichess_puzzles(
        io.StringIO(many),
        bands=[(1400, 1600)],
        per_band=7,
        max_deviation=80,
        min_plays=1000,
        exclude_ids=set(),
    )
    second = blocklist.select_lichess_puzzles(
        io.StringIO(many),
        bands=[(1400, 1600)],
        per_band=7,
        max_deviation=80,
        min_plays=1000,
        exclude_ids=set(),
    )
    assert len(first) == 7 and [r.puzzle_id for r in first] == [r.puzzle_id for r in second]


def test_held_out_games_block_only_positions_after_ply_16():
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 *\n"
    games = blocklist.read_games(io.StringIO(pgn), limit=10)
    hashes = blocklist.game_hashes(games, skip_plies=16)
    board = chess.Board()
    plies = [m for m in games[0].mainline_moves()]
    for move in plies[:16]:
        assert encode.position_hash(board) not in hashes
        board.push(move)
    assert encode.position_hash(board) in hashes


def test_games_are_split_at_each_event_and_sampled_deterministically():
    one = '[Event "a"]\n[Site "https://lichess.org/AAA"]\n\n1. e4 e5 *\n\n'
    two = '[Event "b"]\n[Site "https://lichess.org/BBB"]\n\n1. d4 d5 *\n\n'
    texts = list(blocklist.iter_game_texts(io.StringIO(one + two)))
    assert len(texts) == 2 and texts[0].startswith('[Event "a"]') and texts[1].startswith('[Event "b"]')
    picked = blocklist.read_games(io.StringIO(one + two), limit=10, every=1)
    assert [g.headers["Site"] for g in picked] == ["https://lichess.org/AAA", "https://lichess.org/BBB"]


def test_non_standard_variants_are_skipped():
    variant = '[Event "a"]\n[Site "x"]\n[Variant "Chess960"]\n\n1. e4 e5 *\n\n'
    assert blocklist.read_games(io.StringIO(variant), limit=10) == []


def test_an_empty_blocklist_contains_nothing():
    empty = np.array([], dtype=np.uint64)
    assert blocklist.contains(empty, np.array([1, 2], dtype=np.uint64)).tolist() == [False, False]


def test_the_saved_blocklist_is_sorted_unique_uint64(tmp_path):
    path = tmp_path / "blocklist.npy"
    blocklist.save({5, 3, 9}, path)
    arr = np.load(path)
    assert arr.dtype == np.uint64 and arr.tolist() == [3, 5, 9]
    assert blocklist.contains(arr, np.array([9, 4], dtype=np.uint64)).tolist() == [True, False]
