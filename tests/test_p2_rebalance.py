"""Rebalancing: 48 buckets, weights p_games / p_evaldb clipped to [0.2, 5] and renormalised until stable."""

import chess
import numpy as np
import pytest
from data_fakes import write_pzstd

from blink.board import encode
from blink.board.value import CP_NONE
from blink.data import rebalance
from blink.data.record import CHILD_DTYPE, ROOT_DTYPE

GAME = (
    '[Event "Rated Blitz game"]\n[Site "https://lichess.org/{site}"]\n[Variant "Standard"]\n\n'
    "1. e4 {{ [%eval 0.2] }} 1... e5 {{ [%eval 0.25] }} 2. Nf3 {{ [%eval 0.3] }} 2... Nc6 {{ [%eval #-3] }} "
    "3. Ng1 {{ [%eval 0.1] }} 3... Nb8 {{ [%eval 0.2] }} 4. Nf3 {{ [%eval 0.3] }} 1-0\n\n"
)
PLAIN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/plain1"]

1. d4 d5 2. c4 1-0

"""


def rec_from(fen: str, cp: int = 0, mate: int = 0, dtype=ROOT_DTYPE) -> np.ndarray:
    rec = np.zeros(1, dtype=dtype)
    rec["board"] = encode.pack(encode.encode_board(chess.Board(fen)))
    rec["cp"], rec["mate"] = cp, mate
    return rec


def test_buckets_split_pieces_castling_and_decisiveness_into_48():
    start = rec_from(chess.STARTING_FEN, cp=20)
    no_castling = rec_from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", cp=20)
    decisive = rec_from(chess.STARTING_FEN, cp=900)
    mate = rec_from(chess.STARTING_FEN, cp=CP_NONE, mate=-2)
    endgame = rec_from("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", cp=20)
    got = rebalance.bucket_of(np.concatenate([start, no_castling, decisive, mate, endgame]))
    assert got.tolist()[0] != got.tolist()[1]  # castling rights move a position to another bucket
    assert got[2] == got[3] != got[0]  # a +9 and a mate are both in the most decisive band
    assert got[4] // 8 == 0 and got[0] // 8 == rebalance.PIECE_BANDS - 1
    assert rebalance.NUM_BUCKETS == 48 and got.min() >= 0 and got.max() < 48


def test_children_bucket_like_roots():
    root = rec_from(chess.STARTING_FEN, cp=20)
    child = rec_from(chess.STARTING_FEN, cp=20, dtype=CHILD_DTYPE)
    assert rebalance.bucket_of(root).tolist() == rebalance.bucket_of(child).tolist()


def test_weights_for_looks_up_each_records_bucket():
    recs = np.concatenate([rec_from(chess.STARTING_FEN, cp=20), rec_from("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")])
    weights = np.arange(48, dtype=np.float64) / 10
    got = rebalance.weights_for(recs, weights)
    assert got.dtype == np.float32
    assert got.tolist() == pytest.approx((weights[rebalance.bucket_of(recs)]).tolist())


@pytest.mark.parametrize("seed", range(6))
def test_rebalancing_weights_stay_in_0_2_to_5_after_normalisation_with_mean_within_0_02_of_one(seed):
    rng = np.random.default_rng(seed)
    evaldb = rng.integers(0, 10_000, 48) * (rng.random(48) < 0.9)
    games = rng.integers(0, 10_000, 48) * (rng.random(48) < 0.7)
    games[rng.integers(48)] = 10**7  # one bucket dominates the games
    weights = rebalance.table(games, evaldb)
    p_evaldb = evaldb / evaldb.sum()
    assert weights.min() >= 0.2 - 1e-9 and weights.max() <= 5 + 1e-9
    assert abs(float((p_evaldb * weights).sum()) - 1) <= 0.02


def test_a_table_needs_counts_on_both_sides():
    with pytest.raises(ValueError, match="games"):
        rebalance.table(np.zeros(48), np.ones(48))
    with pytest.raises(ValueError, match="48"):
        rebalance.table(np.ones(47), np.ones(47))


def _source(tmp_path, text: str):
    path = tmp_path / "games.pgn.zst"
    write_pzstd(path, text.encode("utf-8"), frame_bytes=300)
    return path


def test_the_games_histogram_counts_each_unique_position_once(tmp_path):
    once = rebalance.games_histogram(_source(tmp_path, GAME.format(site="a1") + PLAIN), heldout_sites=set())
    twice = rebalance.games_histogram(
        _source(tmp_path, GAME.format(site="a1") + GAME.format(site="b2")), heldout_sites=set()
    )
    # 7 evaluated positions, but 3... Nb8 and 4. Nf3 repeat the positions after 1... e5 and 2. Nf3: 5 unique
    assert once.counts.sum() == 5 == twice.counts.sum()
    assert (once.counts == twice.counts).all()
    assert once.games_with_eval == 1 and twice.games_with_eval == 2
    assert once.games_seen == 2


def test_held_out_games_are_left_out_of_the_games_histogram(tmp_path):
    source = _source(tmp_path, GAME.format(site="keep1") + GAME.format(site="held9").replace("e4", "d4", 1))
    got = rebalance.games_histogram(source, heldout_sites={"held9"})
    assert got.heldout_skipped == 1 and got.games_with_eval == 1
    assert got.counts.sum() == 5


def test_heldout_sites_are_read_from_the_site_tags(tmp_path):
    pgn = tmp_path / "heldout.pgn"
    pgn.write_text(GAME.format(site="abcd1234") + GAME.format(site="zz99"), encoding="utf-8")
    assert rebalance.heldout_sites(pgn) == {"abcd1234", "zz99"}
