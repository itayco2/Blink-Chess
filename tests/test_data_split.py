"""The split rule: val and test_iid by blake2b-8 of the colour-normalised key, the same in any process."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import chess
import numpy as np

from blink.board import encode
from blink.data import split


def _positions() -> list[str]:
    board = chess.Board()
    fens = []
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"):
        board.push_uci(uci)
        fens.append(board.fen())
    return fens


def splits_in_this_process(fens: list[str]) -> tuple[int, list[int], list[str]]:
    hashes = [encode.position_hash(chess.Board(fen)) for fen in fens]
    hashes += list(range(0, 5000, 7))
    return os.getpid(), hashes, [split.split_of(h) for h in hashes]


def test_split_is_identical_in_two_spawned_processes():
    fens = _positions()
    here = splits_in_this_process(fens)
    seen = []
    for _ in range(2):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            seen.append(pool.submit(splits_in_this_process, fens).result())
    assert len({here[0], seen[0][0], seen[1][0]}) == 3  # three different processes
    assert seen[0][1:] == seen[1][1:] == here[1:]


def test_split_of_follows_the_mod_1000_rule():
    assert [split.split_of(h) for h in (0, 1, 2, 3, 4, 999, 1000, 1001, 1002, 2003, 2004)] == [
        "val",
        "val",
        "test_iid",
        "test_iid",
        "train",
        "train",
        "val",
        "val",
        "test_iid",
        "test_iid",
        "train",
    ]
    assert split.split_of(2**64 - 1) == ("val" if (2**64 - 1) % 1000 in (0, 1) else "train")


def test_split_codes_match_split_of_on_uint64_hashes():
    rng = np.random.default_rng(3)
    hashes = rng.integers(0, 2**64 - 1, size=20_000, dtype=np.uint64, endpoint=True)
    codes = split.split_codes(hashes)
    names = [split.SPLITS[code] for code in codes]
    assert names == [split.split_of(int(h)) for h in hashes]
    share = {name: names.count(name) / len(names) for name in split.SPLITS}
    assert abs(share["val"] - 0.002) < 0.001 and abs(share["test_iid"] - 0.002) < 0.001
