"""--source-raw: train before the data area's shards exist.

The first N lines of a raw eval-DB .zst are parsed with the reference parser (blink.data.parse) into
ROOT_DTYPE records and cached as one .npy under BLINK_HOME/data/raw-cache, keyed by the raw file's
name and size and N, so the next run starts in seconds. The read is one sequential pass from the
start of the file (D: is a spinning disk). A truncated final frame, as in a partial download, ends
the stream at the last complete line.

Splits follow the plan's hash rule on fen_hash (colour-normalised): val is h % 1000 in {0, 1},
test_iid is {2, 3}, everything else is train.
"""

import hashlib
import io
import json
import multiprocessing
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zstandard

from blink.data.parse import Rejected, parse_line
from blink.data.record import ROOT_DTYPE

PARSER_VERSION = 1
READ_BUFFER = 1 << 20
CHUNK_LINES = 5000
VAL_BUCKETS = (0, 1)
TEST_BUCKETS = (2, 3)


def read_lines(path: Path, max_lines: int) -> Iterator[bytes]:
    """Up to max_lines complete lines (each ending in a newline) from the start of a .zst."""
    with open(path, "rb") as handle:
        reader = zstandard.ZstdDecompressor().stream_reader(handle, read_across_frames=True)
        buffered = io.BufferedReader(reader, buffer_size=READ_BUFFER)
        for _ in range(max_lines):
            try:
                line = buffered.readline()
            except zstandard.ZstdError:
                return  # a damaged tail: stop at the last complete line
            if not line.endswith(b"\n"):
                return  # end of stream, or the unterminated tail of a truncated frame
            yield line


def _parse_chunk(lines: list[bytes]) -> tuple[np.ndarray, dict[str, int]]:
    records, rejected = [], {}
    for line in lines:
        try:
            records.append(parse_line(line))
        except Rejected as exc:
            rejected[exc.reason] = rejected.get(exc.reason, 0) + 1
    array = np.stack(records) if records else np.zeros(0, dtype=ROOT_DTYPE)
    return array, rejected


def _chunks(lines: Iterable[bytes], size: int) -> Iterator[list[bytes]]:
    chunk: list[bytes] = []
    for line in lines:
        chunk.append(line)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def parse_lines(
    lines: Iterable[bytes], workers: int, chunk_lines: int = CHUNK_LINES
) -> tuple[np.ndarray, dict[str, int]]:
    """Parse in order (spawned worker processes when workers > 1). Returns records and reject counts."""
    chunks = _chunks(lines, chunk_lines)
    if workers <= 1:
        results = list(map(_parse_chunk, chunks))
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = list(pool.imap(_parse_chunk, chunks))
    rejected: dict[str, int] = {}
    for _, counts in results:
        for reason, n in counts.items():
            rejected[reason] = rejected.get(reason, 0) + n
    arrays = [array for array, _ in results]
    return (np.concatenate(arrays) if arrays else np.zeros(0, dtype=ROOT_DTYPE)), rejected


@dataclass(frozen=True)
class RawData:
    records: np.ndarray
    sha1: str  # of the record bytes: this data's identity in the world id
    cache: Path
    built: bool


def cache_path(raw: Path, max_lines: int, cache_dir: Path) -> Path:
    stem = raw.name.split(".")[0]
    return cache_dir / f"{stem}-{raw.stat().st_size}-{max_lines}-v{PARSER_VERSION}.npy"


def _save(cache: Path, records: np.ndarray, sidecar: dict) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, records, allow_pickle=False)
    os.replace(tmp, cache)
    meta = cache.with_suffix(".json")
    meta.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8", newline="")


def load_or_build(
    raw: Path, max_lines: int, cache_dir: Path, workers: int, log: Callable[[str], None] = print
) -> RawData:
    if not raw.is_file():
        raise FileNotFoundError(f"--source-raw: {raw} does not exist")
    cache = cache_path(raw, max_lines, cache_dir)
    if cache.exists():
        records = np.load(cache, allow_pickle=False)
        log(f"source-raw: reusing {cache} ({len(records):,} records)")
        return RawData(records, hashlib.sha1(records.tobytes()).hexdigest(), cache, built=False)
    started = time.perf_counter()
    lines_read = [0]

    def counted() -> Iterator[bytes]:
        for line in read_lines(raw, max_lines):
            lines_read[0] += 1
            yield line

    records, rejected = parse_lines(counted(), workers)
    elapsed = time.perf_counter() - started
    sha1 = hashlib.sha1(records.tobytes()).hexdigest()
    n_lines = lines_read[0]
    sidecar = {
        "raw": raw.name,
        "raw_bytes": raw.stat().st_size,
        "max_lines": max_lines,
        "lines_read": n_lines,
        "records": len(records),
        "rejected": rejected,
        "parser_version": PARSER_VERSION,
        "sha1": sha1,
        "seconds": round(elapsed, 2),
        "lines_per_s": round(n_lines / max(elapsed, 1e-9), 1),
    }
    _save(cache, records, sidecar)
    log(f"source-raw: parsed {n_lines:,} lines into {len(records):,} records in {elapsed:.1f} s")
    return RawData(records, sha1, cache, built=True)


def split(records: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bucket = records["fen_hash"] % np.uint64(1000)
    is_val = np.isin(bucket, VAL_BUCKETS)
    is_test = np.isin(bucket, TEST_BUCKETS)
    return records[~(is_val | is_test)], records[is_val], records[is_test]
