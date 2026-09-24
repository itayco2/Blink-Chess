"""Deterministic positions for parity checks: seeded random games, encoded as the network sees them."""

import random

import chess
import numpy as np

from blink.board import encode

MAX_PLIES = 160
KEEP_PROBABILITY = 0.15


def random_fens(n: int, seed: int, max_plies: int = MAX_PLIES) -> list[str]:
    """n distinct FENs sampled from seeded random games, each with at least one legal move."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = random.Random(seed)
    seen: set[str] = set()
    fens: list[str] = []
    while len(fens) < n:
        board = chess.Board()
        for _ in range(max_plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            fen = board.fen()
            if fen not in seen and rng.random() < KEEP_PROBABILITY:
                seen.add(fen)
                fens.append(fen)
                if len(fens) == n:
                    break
            board.push(rng.choice(legal))
    return fens


def encode_fens(fens: list[str]) -> np.ndarray:
    """int64 [N, 64] square codes, the network's input dtype."""
    codes = np.zeros((len(fens), 64), dtype=np.int64)
    for row, fen in enumerate(fens):
        codes[row] = encode.encode_board(chess.Board(fen))
    return codes
