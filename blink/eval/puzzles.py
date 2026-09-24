# Copyright 2025 DeepMind Technologies Limited
# Modifications copyright 2026 Itay Cohen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Puzzle scoring: DeepMind's scorer, ported exactly, and Blink's runner around it.

The scorer is `evaluate_puzzle_from_pandas_row` and `evaluate_puzzle_from_board` from
google-deepmind/searchless_chess `src/puzzles.py` (Apache-2.0). Modifications, all outside the
scoring logic: the row is any mapping (a csv.DictReader row instead of a pandas Series), the Engine
protocol keeps only `play`, absl flags and `main` are dropped, and the code is reformatted to this
repo's style. A puzzle is solved only if every one of the solver's moves equals the solution, except
that any move that checkmates is accepted (Lichess counts every mate in one as correct).

Everything below the "Blink runner" line is new: reading the CSV, rating bands, the Wilson interval,
and the per-puzzle CSV plus JSON summary written by `blink eval puzzles`.
"""

import csv
import io
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import chess
import chess.pgn

from blink import paths
from blink.play.agents import Agent


class Engine(Protocol):
    def play(self, board: chess.Board) -> chess.Move:
        """Returns the best legal move from a given board."""


def evaluate_puzzle_from_pandas_row(
    puzzle: Mapping[str, str],
    engine: Engine,
) -> bool:
    """Returns True if the `engine` solves the puzzle and False otherwise."""
    game = chess.pgn.read_game(io.StringIO(puzzle["PGN"]))
    if game is None:
        raise ValueError(f"Failed to read game from PGN {puzzle['PGN']}.")
    board = game.end().board()
    return evaluate_puzzle_from_board(
        board=board,
        moves=puzzle["Moves"].split(" "),
        engine=engine,
    )


def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine: Engine,
) -> bool:
    """Returns True if the `engine` solves the puzzle and False otherwise."""
    for move_idx, move in enumerate(moves):
        # According to https://database.lichess.org/#puzzles, the FEN is the
        # position before the opponent makes their move. The position to present to
        # the player is after applying the first move to that FEN. The second move
        # is the beginning of the solution.
        if move_idx % 2 == 1:
            predicted_move = engine.play(board=board).uci()
            # Lichess puzzles consider all mate-in-1 moves as correct, so we need to
            # check if the `predicted_move` results in a checkmate if it differs from
            # the solution.
            if move != predicted_move:
                board.push(chess.Move.from_uci(predicted_move))
                return board.is_checkmate()
        board.push(chess.Move.from_uci(move))
    return True


# ------------------------------------------------------------------------------ Blink runner

PUZZLE_SETS = {"dm10k": ("downloads", "puzzles.csv")}
REQUIRED_COLUMNS = ("PuzzleId", "Rating", "PGN", "Moves")
BANDS = ((1000, "<1000"), (1500, "1000-1500"), (2000, "1500-2000"), (2500, "2000-2500"))
TOP_BAND = "2500+"
Z_95 = 1.959964


def band_of(rating: int) -> str:
    return next((name for upper, name in BANDS if rating < upper), TOP_BAND)


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def resolve_set(name: str) -> Path:
    """A named set under BLINK_HOME, or a path to a CSV with the scorer's columns."""
    if name in PUZZLE_SETS:
        return paths.home().joinpath(*PUZZLE_SETS[name])
    return Path(name)


def read_puzzles(path: Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Rows of a puzzle CSV, front to back, refusing files without the scorer's columns."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks the columns {missing}; its header is {reader.fieldnames}")
        for count, row in enumerate(reader):
            if limit is not None and count >= limit:
                return
            yield row


def board_from_pgn(pgn: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError(f"unreadable PGN {pgn[:60]!r}")
    return game.end().board()


class AgentEngine:
    """DeepMind's Engine protocol on top of a Blink agent; counts any illegal move it is handed."""

    def __init__(self, agent: Agent, game: str = "") -> None:
        self.agent = agent
        self.game = game
        self.illegal = 0

    def play(self, board: chess.Board) -> chess.Move:
        move = self.agent.choose(board, game=self.game).move
        if move not in board.legal_moves:
            self.illegal += 1
        return move


def _summary(results: list[dict], mode: str, source: Path) -> dict:
    solved = sum(r["correct"] for r in results)
    low, high = wilson(solved, len(results))
    bands = {}
    for _, name in (*BANDS, (None, TOP_BAND)):
        members = [r for r in results if r["band"] == name]
        hits = sum(r["correct"] for r in members)
        bands[name] = {
            "n": len(members),
            "correct": hits,
            "accuracy": hits / len(members) if members else 0.0,
        }
    return {
        "set": str(source),
        "mode": mode,
        "n": len(results),
        "correct": solved,
        "accuracy": solved / len(results) if results else 0.0,
        "wilson95": [low, high],
        "illegal_moves": sum(r["illegal"] for r in results),
        "bands": bands,
    }


def run_puzzle_set(
    source: Path, agent: Agent, mode: str, out_dir: Path, limit: int | None = None, label: str = "set"
) -> dict:
    """Score `agent` on the puzzles in `source`; write puzzles_<label>_<mode>.csv and .json to out_dir."""
    results = []
    for row in read_puzzles(source, limit):
        engine = AgentEngine(agent, game=row["PuzzleId"])
        correct = evaluate_puzzle_from_pandas_row(puzzle=row, engine=engine)
        rating = int(row["Rating"])
        results.append(
            {
                "puzzle_id": row["PuzzleId"],
                "rating": rating,
                "band": band_of(rating),
                "correct": int(correct),
                "illegal": engine.illegal,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"puzzles_{label}_{mode}"
    with open(stem.with_suffix(".csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["puzzle_id", "rating", "band", "correct", "illegal"])
        writer.writeheader()
        writer.writerows(results)
    summary = _summary(results, mode, source)
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
