"""The leakage blocklist: position hashes that must never reach training.

Three sources, all hashed on the colour-normalised key (so a mirror twin is blocked too):
1. every position on DeepMind's 10K puzzle lines, plus their source-game positions from ply 16 on;
2. the Lichess rating-band test puzzles (the setup move and every later position);
3. held-out games from the 2026-08 Lichess month, from ply 16 on.
The first 16 plies are left alone: opening positions are shared by thousands of games, and blocking them
would remove openings rather than leaked test positions.
"""

import csv
import hashlib
import io
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np

from blink.board import encode

SKIP_PLIES = 16
BANDS = tuple((lo, lo + 200) for lo in range(400, 2800, 200))


@dataclass(frozen=True)
class LichessPuzzle:
    puzzle_id: str
    fen: str
    moves: str
    rating: int
    deviation: int
    plays: int
    themes: str


def _line_hashes(fen: str, uci_moves: str) -> set[int]:
    board = chess.Board(fen)
    hashes = {encode.position_hash(board)}
    for uci in uci_moves.split():
        board.push_uci(uci)
        hashes.add(encode.position_hash(board))
    return hashes


def _game_hashes(game: chess.pgn.Game, skip_plies: int) -> set[int]:
    board = game.board()
    hashes = set()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        if ply >= skip_plies:
            hashes.add(encode.position_hash(board))
    return hashes


def dm_puzzle_hashes(fh: TextIO, include_source_games: bool = True, skip_plies: int = SKIP_PLIES) -> set[int]:
    hashes: set[int] = set()
    for row in csv.DictReader(fh):
        hashes |= _line_hashes(row["FEN"], row["Moves"])
        if include_source_games and row.get("PGN"):
            game = chess.pgn.read_game(io.StringIO(row["PGN"]))
            if game is not None:
                hashes |= _game_hashes(game, skip_plies)
    return hashes


def dm_puzzle_ids(fh: TextIO) -> set[str]:
    return {row["PuzzleId"] for row in csv.DictReader(fh)}


def _sample_key(puzzle_id: str) -> bytes:
    return hashlib.blake2b(puzzle_id.encode("utf-8"), digest_size=8).digest()


def select_lichess_puzzles(
    fh: TextIO,
    bands: Iterable[tuple[int, int]] = BANDS,
    per_band: int = 500,
    max_deviation: int = 80,
    min_plays: int = 1000,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> list[LichessPuzzle]:
    """A deterministic sample: per rating band, the first `per_band` eligible puzzles by hashed id."""
    bands = list(bands)
    pools: dict[tuple[int, int], list[LichessPuzzle]] = {band: [] for band in bands}
    for row in csv.DictReader(fh):
        if row["PuzzleId"] in exclude_ids:
            continue
        rating, deviation, plays = int(row["Rating"]), int(row["RatingDeviation"]), int(row["NbPlays"])
        if deviation > max_deviation or plays < min_plays:
            continue
        band = next((b for b in bands if b[0] <= rating < b[1]), None)
        if band is None:
            continue
        pools[band].append(
            LichessPuzzle(row["PuzzleId"], row["FEN"], row["Moves"], rating, deviation, plays, row["Themes"])
        )
    selected = []
    for band in bands:
        selected += sorted(pools[band], key=lambda p: _sample_key(p.puzzle_id))[:per_band]
    return selected


def puzzle_line_hashes(puzzles: Iterable[LichessPuzzle]) -> set[int]:
    hashes: set[int] = set()
    for puzzle in puzzles:
        hashes |= _line_hashes(puzzle.fen, puzzle.moves)
    return hashes


def read_games(fh: TextIO, limit: int, every: int = 1) -> list[chess.pgn.Game]:
    """Up to `limit` standard games, keeping one in `every` by hashed Site so the pick is deterministic."""
    games = []
    for text in iter_game_texts(fh):
        if len(games) >= limit:
            break
        site = _SITE.search(text)
        key = site.group(1) if site else text[:200]
        if int.from_bytes(_sample_key(key), "little") % every or _NON_STANDARD.search(text):
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        if game is not None and not game.errors:
            games.append(game)
    return games


_SITE = re.compile(r'^\[Site "([^"]*)"\]', re.MULTILINE)
_NON_STANDARD = re.compile(r'^\[Variant "(?!Standard")', re.MULTILINE)


def iter_game_texts(fh: TextIO) -> Iterable[str]:
    """Split a PGN stream into one text per game, at each [Event line. Streams; never loads the file."""
    buffer: list[str] = []
    for line in fh:
        if line.startswith("[Event ") and any(not b.startswith("[") and b.strip() for b in buffer):
            yield "".join(buffer)
            buffer = []
        buffer.append(line)
    if buffer:
        yield "".join(buffer)


def game_hashes(games: Iterable[chess.pgn.Game], skip_plies: int = SKIP_PLIES) -> set[int]:
    hashes: set[int] = set()
    for game in games:
        hashes |= _game_hashes(game, skip_plies)
    return hashes


def save(hashes: Iterable[int], path: Path) -> np.ndarray:
    arr = np.unique(np.fromiter(hashes, dtype=np.uint64))
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
    return arr


def contains(sorted_hashes: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Vectorised membership against a sorted unique uint64 array."""
    queries = np.asarray(queries, dtype=np.uint64)
    if len(sorted_hashes) == 0:
        return np.zeros(queries.shape, dtype=bool)
    idx = np.minimum(np.searchsorted(sorted_hashes, queries), len(sorted_hashes) - 1)
    return sorted_hashes[idx] == queries


def write_stats(path: Path, stats: dict) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
