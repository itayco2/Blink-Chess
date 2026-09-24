"""Opening books, always read front to back (BLINK_HOME sits on a spinning disk: no random access).

The frozen protocol plays 8moves_v3.pgn sequentially: openings 1-10,000 are the dev slice and
10,001-34,700 the final slice. A match or gauntlet uses openings start, start+1, ... in order, each
once per colour, and may never run past the end of its slice.

`write_slices` (run once, by `blink eval books`) pre-splits the book into BLINK_HOME/books/dev.pgn and
final.pgn: byte-exact copies of the source games, with the sha256 of the source and of both slices in
books.json. Once a slice file exists, `resolve` plays it from its first game instead of skipping into
8moves_v3.pgn; the openings are the same either way. A later run finds the files already written and
checks their hashes; a file that no longer matches its recorded sha256 is refused, never rewritten.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import chess.pgn

from blink import paths

BOOK_NAME = "8moves_v3.pgn"
SLICES = {"dev": (1, 10_000), "final": (10_001, 34_700)}
MANIFEST = "books.json"
GAME_START = re.compile(rb"(?:^|(?<=\n))\[Event ")
RESULT_TOKEN = re.compile(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$")


@dataclass(frozen=True)
class Opening:
    number: int  # 1-based position in the book file
    fen: str  # the starting position (the standard one unless the game carries a FEN tag)
    moves: tuple[str, ...]  # the book moves in UCI


def books_dir() -> Path:
    return paths.home() / "books"


def book_file() -> Path:
    return books_dir() / BOOK_NAME


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
    """(file, first opening, last opening or None) for `dev`, `final` or a PGN path.

    A slice plays its own pre-split file from game 1 when it exists, else 8moves_v3.pgn from its offset."""
    if book in SLICES:
        first, last = SLICES[book]
        own = books_dir() / f"{book}.pgn"
        if own.is_file():
            return own, 1, last - first + 1
        return book_file(), first, last
    return Path(book), 1, None


def openings_for(book: str, pairs: int, skip: int = 0) -> list[Opening]:
    """`pairs` openings of a slice (or file) after the first `skip`, refusing to run past the slice's end."""
    path, first, last = resolve(book)
    if last is not None and first + skip + pairs - 1 > last:
        raise ValueError(
            f"{skip + pairs} openings run past the end of the {book} slice ({first}-{last} of {path.name})"
        )
    openings = read_openings(path, first + skip, pairs)
    if len(openings) < pairs:
        raise ValueError(
            f"{path} holds only {len(openings)} openings from {first + skip}; {pairs} were asked for"
        )
    return openings


# ------------------------------------------------------------------------------ the pre-split slices


def split_games(data: bytes) -> list[bytes]:
    """The raw bytes of each game: a game starts at an `[Event ` tag at the start of a line."""
    starts = [m.start() for m in GAME_START.finditer(data)]
    if not starts:
        return []
    starts[0] = 0  # anything before the first tag (a byte-order mark) stays with game 1
    return [data[a:b] for a, b in zip(starts, [*starts[1:], len(data)], strict=True)]


def _movetext(game: bytes) -> str:
    """The game's moves with whitespace collapsed and the result token dropped (an identity key)."""
    text = game.decode("utf-8", errors="replace").replace("\r\n", "\n")
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text
    return RESULT_TOKEN.sub("", " ".join(body.split()))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_once(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _check_existing(out_dir: Path, manifest: dict, source_sha: str, slices: dict) -> dict:
    if manifest["source"]["sha256"] != source_sha:
        raise ValueError(f"{out_dir / MANIFEST} was written from another book (source sha256 differs)")
    for name in slices:
        data = (out_dir / manifest[name]["file"]).read_bytes()
        if _sha256(data) != manifest[name]["sha256"]:
            raise ValueError(f"{out_dir / manifest[name]['file']} no longer matches its recorded sha256")
    return manifest


def write_slices(source: Path, out_dir: Path, slices: dict[str, tuple[int, int]] = SLICES) -> dict:
    """Write <out_dir>/<slice>.pgn for each slice and books.json with every sha256; once only."""
    data = Path(source).read_bytes()
    source_sha = _sha256(data)
    manifest_path = Path(out_dir) / MANIFEST
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _check_existing(Path(out_dir), manifest, source_sha, slices)
    games = split_games(data)
    needed = max(last for _, last in slices.values())
    if len(games) < needed:
        raise ValueError(f"{source} holds {len(games)} games; the slices need {needed}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": {"file": Path(source).name, "sha256": source_sha, "games": len(games)}}
    keys = {}
    for name, (first, last) in slices.items():
        chunk = games[first - 1 : last]
        payload = b"".join(chunk)
        _write_once(Path(out_dir) / f"{name}.pgn", payload)
        keys[name] = {_movetext(game) for game in chunk}
        manifest[name] = {
            "file": f"{name}.pgn",
            "first": first,
            "last": last,
            "openings": len(chunk),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
    names = list(slices)
    manifest["overlap_move_sequences"] = len(keys[names[0]] & keys[names[-1]]) if len(names) > 1 else 0
    _write_once(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))
    return manifest
