"""The E8 conversion set: endgames.epd positions that Stockfish 19 rates won by at least +5.00.

endgames.epd is read front to back. Each position is screened by SF19 at 1M nodes; one scored at least
+5.00 (or a forced mate) for either side is confirmed at 10M nodes, and kept only if the same side is
still at least +5.00 there. The screen stops at 700 kept positions: the first 200 are the dev set (E2b,
the epsilon choice) and the next 500 the final set (E8). The winning side is the side Blink plays; it is
the side to move or its opponent, whichever Stockfish rates +5.00. This is a proxy for "won" (no
tablebase is used, plan section 1), and both searches are cached, so an interrupted screen resumes
without repeating a search.
"""

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from blink import paths
from blink.eval.sflabel import SfLabel, SfLabeler

SCREEN_NODES = 1_000_000
CONFIRM_NODES = 10_000_000
THRESHOLD_PAWNS = 5.0
WANT = 700
DEV_COUNT = 200
BATCH_PER_PROC = 16
EPD_NAME = "endgames.epd"


def epd_path() -> Path:
    return paths.home() / "books" / EPD_NAME


def out_dir() -> Path:
    return paths.home() / "eval" / "endgames"


def read_positions(path: Path, limit: int | None = None) -> Iterator[tuple[int, str]]:
    """(1-based line number, full FEN) for each position, front to back; EPD opcodes are dropped."""
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if limit is not None and number > limit:
                return
            text = line.strip()
            if not text:
                continue
            fields = text.split()
            fen = " ".join(fields[:6]) if len(fields) >= 6 and fields[4].isdigit() else " ".join(fields[:4])
            yield number, chess.Board(fen).fen()


def winner(label: SfLabel, side_to_move: chess.Color) -> chess.Color | None:
    """The side Stockfish rates at least +5.00 (or mating), or None."""
    pawns = label.pawns
    if pawns >= THRESHOLD_PAWNS:
        return side_to_move
    if pawns <= -THRESHOLD_PAWNS:
        return not side_to_move
    return None


@dataclass(frozen=True)
class Endgame:
    line: int  # line number in endgames.epd
    fen: str
    winner: str  # "white" or "black": the side Blink plays
    screen_pawns: float  # the winner's score at 1M nodes (mate = 100)
    confirm_pawns: float  # the winner's score at 10M nodes

    @property
    def blink_color(self) -> chess.Color:
        return chess.WHITE if self.winner == "white" else chess.BLACK


def _for_winner(label: SfLabel, side_to_move: chess.Color, side: chess.Color) -> float:
    return label.pawns if side == side_to_move else -label.pawns


def screen_one(line: int, fen: str, screen: SfLabeler, confirm: SfLabeler) -> Endgame | None:
    first = screen.label(fen)
    side = winner(first, chess.Board(fen).turn)
    return None if side is None else _confirmed(line, fen, first, confirm.label(fen), side)


def _confirmed(line: int, fen: str, first: SfLabel, second: SfLabel, side: chess.Color) -> Endgame | None:
    side_to_move = chess.Board(fen).turn
    if winner(second, side_to_move) != side:
        return None
    return Endgame(
        line,
        fen,
        "white" if side == chess.WHITE else "black",
        _for_winner(first, side_to_move, side),
        _for_winner(second, side_to_move, side),
    )


@dataclass(frozen=True)
class ScreenResult:
    kept: tuple[Endgame, ...]
    screened: int
    passed_screen: int  # positions at +5.00 at 1M nodes, each confirmed at 10M

    @property
    def dev(self) -> tuple[Endgame, ...]:
        return self.kept[:DEV_COUNT]

    @property
    def final(self) -> tuple[Endgame, ...]:
        return self.kept[DEV_COUNT:WANT]


def _batches(positions: Iterator[tuple[int, str]], size: int) -> Iterator[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    for item in positions:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def screen(
    positions: Iterator[tuple[int, str]],
    screen_labeler: SfLabeler,
    confirm_labeler: SfLabeler,
    want: int = WANT,
    progress: Callable[[int, int], None] | None = None,
) -> ScreenResult:
    """Screen positions in order until `want` are kept (or the positions run out), a batch at a time so
    the labelers can search on several processes; a position after the `want`-th keep is not counted."""
    kept: list[Endgame] = []
    screened = passed = 0
    for batch in _batches(positions, BATCH_PER_PROC * max(screen_labeler.procs, confirm_labeler.procs)):
        firsts = screen_labeler.label_many([(fen, None) for _, fen in batch])
        sides = [winner(label, chess.Board(fen).turn) for label, (_, fen) in zip(firsts, batch, strict=True)]
        seconds = iter(
            confirm_labeler.label_many(
                [(fen, None) for (_, fen), side in zip(batch, sides, strict=True) if side is not None]
            )
        )
        for (line, fen), first, side in zip(batch, firsts, sides, strict=True):
            screened += 1
            if side is not None:
                passed += 1
                found = _confirmed(line, fen, first, next(seconds), side)
                kept += [found] if found is not None else []
            if progress is not None:
                progress(screened, len(kept))
            if len(kept) >= want:
                return ScreenResult(tuple(kept), screened, passed)
    return ScreenResult(tuple(kept), screened, passed)


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_sets(result: ScreenResult, folder: Path) -> dict:
    """dev.jsonl and final.jsonl (one Endgame per line) and endgames.json with the counts."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (("dev", result.dev), ("final", result.final)):
        _write(folder / f"{name}.jsonl", "".join(json.dumps(asdict(e)) + "\n" for e in rows))
    summary = {
        "screened": result.screened,
        "passed_screen": result.passed_screen,
        "kept": len(result.kept),
        "dev": len(result.dev),
        "final": len(result.final),
        "screen_nodes": SCREEN_NODES,
        "confirm_nodes": CONFIRM_NODES,
        "threshold_pawns": THRESHOLD_PAWNS,
        "complete": len(result.kept) >= WANT,
    }
    _write(folder / "endgames.json", json.dumps(summary, indent=2))
    return summary


def read_set(folder: Path, name: str) -> list[Endgame]:
    path = folder / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no {path.name} in {folder}: run `blink eval endgames` first")
    return [Endgame(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
