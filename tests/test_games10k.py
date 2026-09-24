"""games10k: real-game positions (never in training) labelled by Stockfish, the honest external test set."""

import io
import os
from pathlib import Path

import chess
import numpy as np
import pytest

from blink.board import encode, moves
from blink.data import games10k
from blink.data.record import ROOT_DTYPE

PGN = (
    '[Event "a"]\n[Site "https://lichess.org/AAA"]\n\n'
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 "
    "10. d4 Nbd7 11. c4 c6 *\n\n"
)
STOCKFISH = Path(r"D:\tools\stockfish\stockfish-windows-x86-64-universal.exe")


def test_candidates_are_unique_positions_from_ply_16_on():
    positions = games10k.candidate_positions(io.StringIO(PGN), skip_plies=16)
    assert len(positions) == 7  # the positions after plies 16..22 of a 22-ply game
    assert len({encode.position_hash(b) for b in positions}) == 7


def test_the_sample_is_deterministic_and_capped():
    positions = games10k.candidate_positions(io.StringIO(PGN), skip_plies=16)
    first = games10k.sample(positions, n=4)
    assert len(first) == 4
    assert [b.fen() for b in first] == [b.fen() for b in games10k.sample(positions, n=4)]


def test_a_label_becomes_a_root_record_from_the_side_to_move_view():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1")
    rec = games10k.to_record(board, best=chess.Move.from_uci("g8f8"), cp=-450, mate=None, depth=22)
    assert rec.dtype == ROOT_DTYPE
    assert int(rec["cp"]) == -450 and int(rec["depth"]) == 22
    assert moves.decode_move(board, int(rec["move"])) == chess.Move.from_uci("g8f8")
    assert np.array_equal(encode.unpack(rec["board"]), encode.encode_board(board))


@pytest.mark.local
@pytest.mark.skipif(not STOCKFISH.exists(), reason="Stockfish 19 is not unpacked here")
def test_stockfish_labels_a_mate_in_one_with_mate_plus_one():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    [(best, cp, mate, depth)] = games10k.label_with_stockfish([board.fen()], STOCKFISH, nodes=20_000)
    assert chess.Move.from_uci(best) == chess.Move.from_uci("d1d8") and mate == 1 and cp is None
    assert depth > 0


def test_workers_are_spawn_safe_and_skip_nothing_when_resumed(tmp_path):
    done = {"a": ("e2e4", 10, None, 12)}
    todo = games10k.remaining(["a", "b"], done)
    assert todo == ["b"]
    assert 1 <= games10k.default_procs(cpu_count=4) == 3  # leaves one core free on a 4-core CI runner
    assert games10k.default_procs(cpu_count=12) == 5  # capped at 5 on the 12-thread build machine
    assert games10k.default_procs(cpu_count=None) == 1
    assert os.cpu_count() is None or games10k.default_procs() <= os.cpu_count()
