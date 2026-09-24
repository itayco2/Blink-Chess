"""Build the P2 blocklist and the held-out evaluation sets from the downloaded files.

Run: python -m blink.data.blocklist_build
Writes BLINK_HOME/data/blocklist_v1.npy, BLINK_HOME/data/blocklist_v1.json (counts and overlaps),
BLINK_HOME/eval/lichess_bands.csv (12 bands x 500) and BLINK_HOME/eval/heldout_games.pgn (2,000 games).
"""

import csv
import hashlib
import io
import time
from pathlib import Path

import zstandard

from blink import paths
from blink.data import blocklist

HELDOUT_GAMES = 2000
HELDOUT_EVERY = 50  # keep one game in 50 by hashed Site, so held-out games spread across the prefix


def _zst_text(path: Path) -> io.TextIOWrapper:
    raw = open(path, "rb")  # noqa: SIM115 - closed with the wrapper
    reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True, closefd=True)
    return io.TextIOWrapper(reader, encoding="utf-8", newline="")


def _write_puzzles(puzzles: list[blocklist.LichessPuzzle], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "NbPlays", "Themes"])
        for p in puzzles:
            writer.writerow([p.puzzle_id, p.fen, p.moves, p.rating, p.deviation, p.plays, p.themes])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(home: Path) -> dict:
    downloads, data, evaldir = home / "downloads", home / "data", home / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with open(downloads / "puzzles.csv", encoding="utf-8", newline="") as fh:
        dm_hashes = blocklist.dm_puzzle_hashes(fh)
    with open(downloads / "puzzles.csv", encoding="utf-8", newline="") as fh:
        dm_ids = blocklist.dm_puzzle_ids(fh)

    with _zst_text(downloads / "lichess_db_puzzle.csv.zst") as fh:
        bands = blocklist.select_lichess_puzzles(fh, exclude_ids=dm_ids)
    _write_puzzles(bands, evaldir / "lichess_bands.csv")
    band_hashes = blocklist.puzzle_line_hashes(bands)

    with _zst_text(downloads / "lichess_db_standard_rated_2026-08.prefix300M.pgn.zst") as fh:
        games = blocklist.read_games(fh, limit=HELDOUT_GAMES, every=HELDOUT_EVERY)
    (evaldir / "heldout_games.pgn").write_text("\n\n".join(str(g) for g in games) + "\n", encoding="utf-8")
    game_hashes = blocklist.game_hashes(games)

    union = dm_hashes | band_hashes | game_hashes
    arr = blocklist.save(union, data / "blocklist_v1.npy")
    stats = {
        "dm_puzzle_positions": len(dm_hashes),
        "dm_puzzle_ids": len(dm_ids),
        "lichess_band_puzzles": len(bands),
        "lichess_band_positions": len(band_hashes),
        "heldout_games": len(games),
        "heldout_game_positions": len(game_hashes),
        "overlap_dm_and_bands": len(dm_hashes & band_hashes),
        "overlap_dm_and_games": len(dm_hashes & game_hashes),
        "overlap_bands_and_games": len(band_hashes & game_hashes),
        "blocklist_size": int(arr.size),
        "blocklist_sha256": _sha256(data / "blocklist_v1.npy"),
        "skip_plies": blocklist.SKIP_PLIES,
        "bands": [list(b) for b in blocklist.BANDS],
        "seconds": round(time.time() - started, 1),
    }
    blocklist.write_stats(data / "blocklist_v1.json", stats)
    return stats


if __name__ == "__main__":
    import json

    print(json.dumps(build(paths.home()), indent=2))
