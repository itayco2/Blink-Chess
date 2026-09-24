"""Opening books, always read front to back (BLINK_HOME sits on a spinning disk: no random access).

The frozen protocol plays 8moves_v3.pgn sequentially: openings 1-10,000 are the dev slice and
10,001-34,700 the final slice. A match or gauntlet uses openings start, start+1, ... in order, each
once per colour, and may never run past the end of its slice.
"""

from dataclasses import dataclass
from pathlib import Path

import chess.pgn

from blink import paths

BOOK_NAME = "8moves_v3.pgn"
SLICES = {"dev": (1, 10_000), "final": (10_001, 34_700)}


@dataclass(frozen=True)
class Opening:
    number: int  # 1-based position in the book file
    fen: str  # the starting position (the standard one unless the game carries a FEN tag)
    moves: tuple[str, ...]  # the book moves in UCI


def book_file() -> Path:
    return paths.home() / "books" / BOOK_NAME


def read_openings(path: Path, start: int = 1, count: int | None = None) -> list[Opening]:
    """Openings number start, start+1, ... (at most `count`), skipping earlier games without parsing them."""
    out: list[Opening] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for _ in range(start - 1):
            if not chess.pgn.skip_game(handle):
                return out
        number = start
        while count is None or len(out) < count:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            moves = tuple(move.uci() for move in game.mainline_moves())
            out.append(Opening(number, game.board().fen(), moves))
            number += 1
    return out


def resolve(book: str) -> tuple[Path, int, int | None]:
    """(file, first opening, last opening or None): `dev` and `final` slice 8moves_v3; else a PGN path."""
    if book in SLICES:
        first, last = SLICES[book]
        return book_file(), first, last
    return Path(book), 1, None


def openings_for(book: str, pairs: int) -> list[Opening]:
    """The first `pairs` openings of a slice (or file), refusing to run past the slice's end."""
    path, first, last = resolve(book)
    if last is not None and first + pairs - 1 > last:
        raise ValueError(f"{pairs} openings run past the end of the {book} slice ({first}-{last})")
    openings = read_openings(path, first, pairs)
    if len(openings) < pairs:
        raise ValueError(f"{path} holds only {len(openings)} openings from {first}; {pairs} were asked for")
    return openings
