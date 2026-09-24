"""`blink film pick`: a story score for each of the 200 film candidates, and a short list of 5 for G9.

The candidates are 200 of the Lichess band puzzles (all in the leakage blocklist), drawn by a fixed
hash of their PuzzleId so the pool never depends on file order or on any model. Every frame of the
run scores all 200 in one network call. A candidate's story is:
  - final correctness: the last frame's top-1 is the puzzle's solution (weight 2);
  - top-1 changes between consecutive frames: the network visibly changes its mind, capped at 5 so
    a flickering position cannot win on noise (weight 1, scaled to [0, 1]);
  - the win% swing: max minus min of the mean win% across frames (weight 1).
Ties break by PuzzleId. Itay picks 1 of the top 5 at G9.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess

from blink.film import extract
from blink.train.atomic import write_text_atomic

CANDIDATES = 200
SALT = b"blink-film"
W_FINAL = 2.0
W_CHANGES = 1.0
W_SWING = 1.0
MAX_CHANGES = 5


@dataclass(frozen=True)
class Story:
    candidate: extract.FilmPosition
    top1: tuple[str, ...]
    wins: tuple[float, ...]
    changes: int
    final_correct: bool
    swing: float
    score: float


def _draw_key(puzzle_id: str) -> bytes:
    return hashlib.blake2b(SALT + puzzle_id.encode("utf-8"), digest_size=8).digest()


def draw_candidates(bands_csv: Path, n: int = CANDIDATES) -> list[extract.FilmPosition]:
    puzzles = extract.read_band_puzzles(bands_csv)
    return sorted(puzzles, key=lambda p: _draw_key(p.puzzle_id))[:n]


def story(candidate: extract.FilmPosition, top1: tuple[str, ...], wins: tuple[float, ...]) -> Story:
    changes = sum(a != b for a, b in zip(top1, top1[1:], strict=False))
    final_correct = bool(top1) and top1[-1] == candidate.solution
    swing = (max(wins) - min(wins)) if wins else 0.0
    score = W_FINAL * final_correct + W_CHANGES * min(changes, MAX_CHANGES) / MAX_CHANGES + W_SWING * swing
    return Story(candidate, tuple(top1), tuple(wins), changes, final_correct, swing, score)


def sort_key(item: Story) -> tuple[float, str]:
    return (-item.score, item.candidate.puzzle_id)


def rank(
    candidates: list[extract.FilmPosition],
    sources: list[extract.FrameSource],
    device: str = "cpu",
    loader: Callable = extract.load_frame,
) -> list[Story]:
    """Every candidate's story over every frame (one network call per frame), best first."""
    boards = [chess.Board(c.fen) for c in candidates]
    top1: list[list[str]] = [[] for _ in candidates]
    wins: list[list[float]] = [[] for _ in candidates]
    for source in sources:
        evaluator, _ = loader(source, device)
        for i, pred in enumerate(extract.predict(evaluator, boards)):
            top1[i].append(extract.top_moves(pred["legal"], 1)[0]["move"])
            wins[i].append(pred["win"])
    stories = [story(c, tuple(t), tuple(w)) for c, t, w in zip(candidates, top1, wins, strict=True)]
    return sorted(stories, key=sort_key)


def format_top(stories: list[Story], n: int) -> str:
    lines = ["rank puzzle  rating  score  final  changes  swing  top-1 by frame (first, last)  solution"]
    for i, s in enumerate(stories[:n], start=1):
        lines.append(
            f"{i:>4} {s.candidate.puzzle_id:<7} {s.candidate.rating:>6}  {s.score:5.2f}  "
            f"{'yes' if s.final_correct else 'no':<5}  {s.changes:>7}  {s.swing:5.2f}  "
            f"{s.top1[0] if s.top1 else '-'} -> {s.top1[-1] if s.top1 else '-'}  {s.candidate.solution}"
        )
    return "\n".join(lines)


def write_ranking(stories: list[Story], out: Path, run: str, frames: int) -> Path:
    ranked = [
        {
            "puzzle_id": s.candidate.puzzle_id,
            "rating": s.candidate.rating,
            "fen": s.candidate.fen,
            "solution": s.candidate.solution,
            "themes": s.candidate.themes,
            "score": round(s.score, 4),
            "final_correct": s.final_correct,
            "changes": s.changes,
            "swing": round(s.swing, 4),
            "top1": list(s.top1),
        }
        for s in stories
    ]
    payload = {"run": run, "frames": frames, "weights": [W_FINAL, W_CHANGES, W_SWING], "ranked": ranked}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(payload, indent=1) + "\n")
    return out
