"""The E8 conversion set: endgames.epd positions that Stockfish 19 rates won by at least +5.00.

endgames.epd is read front to back. Each position is screened by SF19 at 1M nodes; one scored at least
+5.00 (or a forced mate) for either side is confirmed at 10M nodes, and kept only if the same side is
still at least +5.00 there. The screen stops at 700 kept positions: the first 200 are the dev set (E2b,
the epsilon choice) and the next 500 the final set (E8). The winning side is the side Blink plays; it is
the side to move or its opponent, whichever Stockfish rates +5.00. This is a proxy for "won" (no
tablebase is used, plan section 1), and both searches are cached, so an interrupted screen resumes
without repeating a search.

endgames.epd repeats some positions with other move counters (22 of them). A position is screened only the
first time it appears (placement, side to move, castling and en passant; the counters ignored), so no
position sits in both the dev set, which chooses epsilon, and the final set, which is published.
endgames.json records the repeats skipped and the dev/final overlap, which E2b and E8 refuse unless 0.

PR-4 (EVAL.md section 5) adds looks at lines 1,000, 5,000 and 20,000, then every 20,000: the screen stops
when endgames.epd is declared unable to supply 700 (blink.eval.endgame_looks). endgames.json then also
records each source and its sha256, the harness commit, the counts at each look, any declaration and the
branch: "epd" (screening endgames.epd), "epd-declared" (the declaration was made) or "fallback" (PR-4's
fallback source, blink.eval.endgame_sources: only after the declaration and with Itay's OK). The fallback
writes only where the declaration is recorded, and its endgames.json keeps that epd-declared summary as
`declared_by`: the looks that justified the switch outlive the sets they replace.
"""

import json
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from blink import paths
from blink.eval.endgame_looks import Look, LookPlan, LookTracker, describe
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


def position_key(fen: str) -> str:
    """A position without its move counters: placement, side to move, castling and en passant."""
    return " ".join(fen.split()[:4])


def overlap(first: Iterable[Endgame], second: Iterable[Endgame]) -> int:
    """How many positions (counters ignored) appear in both sets."""
    return len({position_key(e.fen) for e in first} & {position_key(e.fen) for e in second})


def first_sightings(positions: Iterator[tuple[int, str]], repeats: list[int]) -> Iterator[tuple[int, str]]:
    """Each position the first time it appears; the lines of later repeats are appended to `repeats`."""
    seen: set[str] = set()
    for line, fen in positions:
        key = position_key(fen)
        if key in seen:
            repeats.append(line)
            continue
        seen.add(key)
        yield line, fen


@dataclass(frozen=True)
class ScreenResult:
    kept: tuple[Endgame, ...]
    screened: int
    passed_screen: int  # positions at +5.00 at 1M nodes, each confirmed at 10M
    repeats_skipped: int = 0  # lines holding a position already seen (other move counters)
    looks: tuple[Look, ...] = ()  # PR-4's looks, when the screen was given a plan
    declaration: str | None = None  # why the source was declared unable to supply 700 (PR-4), if it was
    dev_size: int | None = None  # how many of `kept` are the dev set (None: the first DEV_COUNT)

    @property
    def dev(self) -> tuple[Endgame, ...]:
        return self.kept[: self._dev_size]

    @property
    def final(self) -> tuple[Endgame, ...]:
        return self.kept[self._dev_size : self._dev_size + WANT - DEV_COUNT]

    @property
    def _dev_size(self) -> int:
        return DEV_COUNT if self.dev_size is None else self.dev_size

    @property
    def complete(self) -> bool:
        return len(self.dev) == DEV_COUNT and len(self.final) == WANT - DEV_COUNT


def _batches(positions: Iterator[tuple[int, str]], size: int) -> Iterator[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    for item in positions:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _labelled(
    batch: list[tuple[int, str]], screen_labeler: SfLabeler, confirm_labeler: SfLabeler
) -> Iterator[tuple[int, str, SfLabel, chess.Color | None, SfLabel | None]]:
    """(line, fen, 1M-node label, the side at +5.00 or None, its 10M-node label or None) per position:
    the batch's screens in one label_many call, then its confirms in another."""
    firsts = screen_labeler.label_many([(fen, None) for _, fen in batch])
    sides = [winner(label, chess.Board(fen).turn) for label, (_, fen) in zip(firsts, batch, strict=True)]
    seconds = iter(
        confirm_labeler.label_many(
            [(fen, None) for (_, fen), side in zip(batch, sides, strict=True) if side is not None]
        )
    )
    for (line, fen), first, side in zip(batch, firsts, sides, strict=True):
        yield line, fen, first, side, None if side is None else next(seconds)


def screen(
    positions: Iterator[tuple[int, str]],
    screen_labeler: SfLabeler,
    confirm_labeler: SfLabeler,
    want: int = WANT,
    progress: Callable[[int, int], None] | None = None,
    looks: LookPlan | None = None,
    on_look: Callable[[Look], None] | None = None,
) -> ScreenResult:
    """Screen positions in order until `want` are kept (or the positions run out), a batch at a time so
    the labelers can search on several processes; a position after the `want`-th keep is not counted,
    and a repeat of a position already seen is skipped. With a look plan (PR-4, blink.eval.endgame_looks)
    each look is taken as its line passes and handed to `on_look`, and a look that declares, or the file
    ending short of 700 kept, stops the screen with the declaration."""
    kept: list[Endgame] = []
    repeats: list[int] = []
    screened = passed = last_line = 0
    tracker = LookTracker(looks, on_look) if looks is not None else None

    def result(declaration: str | None = None) -> ScreenResult:
        taken = tuple(tracker.looks) if tracker else ()
        return ScreenResult(tuple(kept), screened, passed, len(repeats), taken, declaration)

    def look(line: int) -> str | None:
        declaring = tracker.reach(line, screened, passed, len(kept)) if tracker else None
        return describe(declaring) if declaring else None

    size = BATCH_PER_PROC * max(screen_labeler.procs, confirm_labeler.procs)
    for batch in _batches(first_sightings(positions, repeats), size):
        for line, fen, first, side, second in _labelled(batch, screen_labeler, confirm_labeler):
            declaration = look(line - 1)
            if declaration:
                return result(declaration)
            screened, last_line = screened + 1, line
            if side is not None:
                passed += 1
                found = _confirmed(line, fen, first, second, side)
                kept += [found] if found is not None else []
            if progress is not None:
                progress(screened, len(kept))
            declaration = look(line)
            if declaration or len(kept) >= want:
                return result(declaration)
    last_line = max([last_line, *repeats])
    declaration = look(last_line)
    return result(declaration or (tracker.ended(last_line, len(kept)) if tracker else None))


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def combine(dev: ScreenResult, final: ScreenResult) -> ScreenResult:
    """The fallback's two screens as one result: dev's kept positions, then final's."""
    return ScreenResult(
        dev.kept + final.kept,
        dev.screened + final.screened,
        dev.passed_screen + final.passed_screen,
        dev.repeats_skipped + final.repeats_skipped,
        dev_size=len(dev.kept),
    )


def screen_record(set_name: str, source: dict, result: ScreenResult, limit: int | None) -> dict:
    """One screen's entry in endgames.json: its source, its counts, its looks and any declaration."""
    return {
        "set": set_name,
        "source": source,
        "limit": limit,
        "screened": result.screened,
        "passed_screen": result.passed_screen,
        "repeats_skipped": result.repeats_skipped,
        "kept": len(result.kept),
        "looks": [asdict(look) for look in result.looks],
        "declaration": result.declaration,
    }


def _summary(folder: Path) -> dict | None:
    path = folder / "endgames.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def recorded_branch(folder: Path) -> str | None:
    """The branch the folder's endgames.json records (None without one, or from before branches)."""
    return (_summary(folder) or {}).get("branch")


def _declares(summary: object) -> bool:
    """An epd-declared summary with its declaration and a hashed endgames.epd screen behind it."""
    if (
        not isinstance(summary, dict)
        or summary.get("branch") != "epd-declared"
        or not summary.get("declaration")
    ):
        return False
    sources = [screen.get("source") or {} for screen in summary.get("screens") or []]
    return any(source.get("kind") == EPD_NAME and source.get("sha256") for source in sources)


def declaration_record(folder: Path) -> dict | None:
    """endgames.epd's PR-4 declaration as the folder's endgames.json records it: the epd-declared summary
    itself, or the one a fallback summary carries as `declared_by` (None when neither is there). The
    fallback carries it forward, so its looks, sha256, declaration and harness commit outlive the sets."""
    summary = _summary(folder)
    if summary is not None and summary.get("branch") == "fallback":
        summary = summary.get("declared_by")
    return summary if _declares(summary) else None


def write_sets(result: ScreenResult, folder: Path, record: dict | None = None) -> dict:
    """dev.jsonl and final.jsonl (one Endgame per line) and endgames.json with the counts, the declaration
    and `record` (the branch, the sources and the harness commit)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (("dev", result.dev), ("final", result.final)):
        _write(folder / f"{name}.jsonl", "".join(json.dumps(asdict(e)) + "\n" for e in rows))
    summary = {
        "screened": result.screened,
        "passed_screen": result.passed_screen,
        "repeats_skipped": result.repeats_skipped,
        "kept": len(result.kept),
        "dev": len(result.dev),
        "final": len(result.final),
        "overlap_positions": overlap(result.dev, result.final),
        "screen_nodes": SCREEN_NODES,
        "confirm_nodes": CONFIRM_NODES,
        "threshold_pawns": THRESHOLD_PAWNS,
        "complete": result.complete,
        "declaration": result.declaration,
        **(record or {}),
    }
    _write(folder / "endgames.json", json.dumps(summary, indent=2))
    return summary


def read_set(folder: Path, name: str) -> list[Endgame]:
    path = folder / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no {path.name} in {folder}: run `blink eval endgames` first")
    return [Endgame(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
