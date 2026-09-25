"""Where the E8/E2b endgames come from, and what endgames.json records about it (PR-4, EVAL.md section 5).

endgames.epd is the plan's source (section 1). Its record is the path, its sha256, its last line and its
unique positions (the Poisson looks project onto them; blink.eval.endgame_looks).

The fallback source is PR-4's, used only when endgames.epd has been declared unable to supply 700 and
only with Itay's OK, given before any conversion game; it runs only through `blink eval endgames --source
fallback`. It reads the v1 pack's held-out roots front to back:
- dev (200, E2b): val_roots.bin endgames whose group is not held out as test_grouped (the pack gives val
  precedence over test_grouped, so a val root can sit in a held-out group; blink.data.grouped.selected
  with the pack's salt finds them);
- final (500, E8): test_grouped_roots.bin endgames;
- an endgame is lichess's Divider one: at most 6 queens, rooks, bishops and knights in total;
- prefilter, for both: the Lichess label is at least +5.00 for either side (|cp| >= 500) or a forced mate
  (a mate label other than "checkmated now"); every candidate then meets the unchanged SF19 rule.
A candidate's `line` is its 1-based record index in the file, and its FEN is the packed board's
colour-normalised twin (White to move, move counters 0 1): the side Stockfish rates +5.00 is Blink's side.
"""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from blink.board import encode
from blink.board.value import CP_NONE
from blink.data import grouped
from blink.data.children import codes_to_board
from blink.data.record import ROOT_DTYPE

MAX_MAJORS_AND_MINORS = 6  # lichess's Divider: an endgame has at most 6 queens, rooks, bishops, knights
LICHESS_CP = 500  # the prefilter: at least +5.00 for either side
FALLBACK_SPLITS = {"dev": "val", "final": "test_grouped"}
MANIFEST = "manifest.json"
_MAJOR_AND_MINOR_CODES = np.array(
    [
        *(base + piece - 1 for base in (encode.OWN, encode.OPP) for piece in range(chess.KNIGHT, chess.KING)),
        encode.OWN_CASTLING_ROOK,
        encode.OPP_CASTLING_ROOK,
    ],
    dtype=np.uint8,
)


def _sha256(path: Path) -> str:
    from blink.eval.orchestrate import sha256_file

    return sha256_file(path)


def harness_commit(repo: Path | None = None) -> dict:
    """The git commit the harness runs from, and whether tracked files differ from it (None when the
    package is not in a git checkout)."""
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True
        )
    except OSError:
        return {"commit": None, "dirty": None}
    if head.returncode != 0:
        return {"commit": None, "dirty": None}
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def describe_epd(path: Path) -> dict:
    """endgames.epd's record: path, sha256, last line, unique positions (counters aside) and repeats."""
    from blink.eval.endgames import first_sightings, read_positions

    repeats: list[int] = []
    last = positions = 0
    for line, _ in first_sightings(read_positions(path), repeats):
        last, positions = line, positions + 1
    return {
        "kind": "endgames.epd",
        "path": str(path),
        "sha256": _sha256(path),
        "lines": max([last, *repeats]),
        "positions": positions,
        "repeats": len(repeats),
    }


def majors_and_minors(codes: np.ndarray) -> np.ndarray:
    """[N, 64] square codes -> the queens, rooks, bishops and knights of both sides on each board."""
    return np.isin(codes, _MAJOR_AND_MINOR_CODES).sum(axis=1)


def candidates(roots: np.ndarray, salt: int | None) -> np.ndarray:
    """0-based indices of the Divider endgames with a decisive Lichess label, front to back; with a salt,
    those whose group test_grouped holds out are left out."""
    cp = roots["cp"].astype(np.int32)
    is_mate = cp == CP_NONE
    decisive = np.where(is_mate, roots["mate"] != 0, np.abs(cp) >= LICHESS_CP)
    keep = decisive & (majors_and_minors(encode.unpack(roots["board"])) <= MAX_MAJORS_AND_MINORS)
    if salt is not None and keep.any():
        chosen = np.flatnonzero(keep)
        keep[chosen[grouped.selected(roots["board"][chosen], salt)]] = False
    return np.flatnonzero(keep)


@dataclass(frozen=True)
class FallbackSource:
    set_name: str  # "dev" or "final"
    split: str  # "val" or "test_grouped"
    path: Path
    salt: int | None  # val: test_grouped's salt, whose held-out groups are left out; test_grouped: None
    manifest_sha256: str | None  # the pack manifest's sha256 of the file, when it records one

    def read(self, limit: int | None = None) -> np.ndarray:
        """The file's first `limit` root records (all of them without a limit)."""
        return np.fromfile(self.path, dtype=ROOT_DTYPE, count=-1 if limit is None else limit)

    def positions(self, roots: np.ndarray) -> Iterator[tuple[int, str]]:
        """(1-based record index, FEN) of each candidate, front to back."""
        for index in candidates(roots, self.salt):
            yield int(index) + 1, codes_to_board(encode.unpack(roots["board"][index])).fen()

    def describe(self, roots: np.ndarray) -> dict:
        digest = _sha256(self.path)
        if self.manifest_sha256 is not None and digest != self.manifest_sha256:
            raise ValueError(
                f"{self.path.name} hashes to {digest}, but the pack's manifest records {self.manifest_sha256}"
            )
        return {
            "kind": "fallback",
            "set": self.set_name,
            "split": self.split,
            "path": str(self.path),
            "sha256": digest,
            "records": len(roots),
            "candidates": len(candidates(roots, self.salt)),
            "salt": self.salt,
        }


def fallback_sources(pack_dir: Path) -> dict[str, FallbackSource]:
    """PR-4's two fallback sources in the pack folder: dev from val, final from test_grouped."""
    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST} in {pack_dir}: the fallback needs the v1 pack")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    salt = int(manifest["grouped"]["salt"])
    shards = manifest.get("shards", {})
    sources = {}
    for set_name, split in FALLBACK_SPLITS.items():
        path = pack_dir / f"{split}_roots.bin"
        if not path.is_file():
            raise FileNotFoundError(f"no {path.name} in {pack_dir}: the fallback needs the v1 pack")
        recorded = shards.get(path.name, {}).get("sha256")
        sources[set_name] = FallbackSource(set_name, split, path, salt if split == "val" else None, recorded)
    return sources
