"""The skeleton pack: N frames of the eval DB to permuted root shards plus a manifest.

Frames decode and parse in spawned workers (blink.data.frames). Records are filtered against an
optional blocklist, routed by split (blink.data.split), and train records go to shard fen_hash % K.
Each shard is permuted in RAM with a seed of its own, written to .tmp and moved into place; the
manifest goes last, so a directory with a manifest is a complete pack.

This pack holds every record in RAM, which suits the P1 skeleton (10 frames are about 40 MB).
The full P2 pack routes through on-disk buckets instead.
"""

import hashlib
import json
import os
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from blink.data import frames, parse, split, zst
from blink.data.record import ROOT_DTYPE

DEFAULT_SEED = 20260924
FORMAT = "blink-pack-v1"
MANIFEST = "manifest.json"
TIMING = "timing.json"
ERROR_SAMPLES = 3
REPLACE_RETRIES = 5


@dataclass(frozen=True)
class PackConfig:
    source: Path
    out: Path
    shards: int
    frames: int | None = None
    workers: int = 1
    seed: int = DEFAULT_SEED
    blocklist: Path | None = None
    overwrite: bool = False


class ParsedLines(NamedTuple):
    records: np.ndarray  # ROOT_DTYPE, in line order
    rejects: dict[str, int]  # documented parse.Rejected reasons
    errors: dict[str, int]  # anything else, by exception type: a parser bug, never silently dropped
    error_samples: list[str]


class Collected(NamedTuple):
    records: np.ndarray
    lines: int
    rejects: Counter
    errors: Counter
    error_samples: list[str]
    frames: int
    end: str | None
    timing: dict


def parse_lines(lines: list[bytes]) -> ParsedLines:
    """parse.parse_line over lines. Runs in a spawned worker, so it is a module-level function."""
    records = np.empty(len(lines), dtype=ROOT_DTYPE)
    kept = 0
    rejects: Counter = Counter()
    errors: Counter = Counter()
    samples: list[str] = []
    for line in lines:
        try:
            records[kept] = parse.parse_line(line)
            kept += 1
        except parse.Rejected as exc:
            rejects[exc.reason] += 1
        except Exception as exc:  # noqa: BLE001 - counted, sampled and reported, never dropped
            errors[type(exc).__name__] += 1
            if len(samples) < ERROR_SAMPLES:
                samples.append(f"{type(exc).__name__}: {exc} | {line[:160].decode('utf-8', 'replace')}")
    return ParsedLines(records[:kept].copy(), dict(rejects), dict(errors), samples)


def write_atomic(path: Path, data: bytes) -> None:
    """Write `path` via a .tmp sibling and os.replace, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a scanner or tail holds the target open (Windows)
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(0.1)


def load_blocklist(path: Path | None) -> tuple[np.ndarray, dict | None]:
    """A sorted unique uint64 array of blocked fen_hash values, and its manifest entry."""
    if path is None:
        return np.empty(0, dtype=np.uint64), None
    data = Path(path).read_bytes()
    blocked = np.load(path, allow_pickle=False)
    if blocked.ndim != 1 or blocked.dtype != np.uint64:
        raise ValueError(f"{path}: a blocklist is a 1-D uint64 .npy of fen_hash values, got {blocked.dtype}")
    unique = np.unique(blocked)
    return unique, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "entries": len(unique)}


def collect(cfg: PackConfig) -> Collected:
    reader = zst.FrameReader(cfg.source, cfg.frames)
    chunks, rejects, errors, samples = [], Counter(), Counter(), []
    lines = decode_s = parse_s = compressed = decompressed = 0
    start = time.perf_counter()
    for out in frames.run_frames(reader, parse_lines, cfg.workers):
        parsed: ParsedLines = out.result
        chunks.append(parsed.records)
        rejects.update(parsed.rejects)
        errors.update(parsed.errors)
        samples.extend(parsed.error_samples[: ERROR_SAMPLES - len(samples)])
        lines += out.lines
        decode_s += out.decode_s
        parse_s += out.work_s
        compressed += out.compressed
        decompressed += out.decompressed
    timing = {
        "wall_s": time.perf_counter() - start,
        "decode_s": decode_s,
        "parse_s": parse_s,
        "compressed_bytes": compressed,
        "decompressed_bytes": decompressed,
        "workers": cfg.workers,
    }
    records = np.concatenate(chunks) if chunks else np.empty(0, dtype=ROOT_DTYPE)
    return Collected(records, lines, rejects, errors, samples, reader.frames_read, reader.end, timing)


def route(records: np.ndarray, shards: int) -> dict[str, np.ndarray]:
    """{file name: records} for train_000..train_{K-1}, val and test_iid, each in arrival order."""
    codes = split.split_codes(records["fen_hash"])
    train = records[codes == split.TRAIN_CODE]
    shard_of = train["fen_hash"] % np.uint64(shards)
    routed = {f"train_{i:03d}.bin": train[shard_of == i] for i in range(shards)}
    routed["val.bin"] = records[codes == split.VAL_CODE]
    routed["test_iid.bin"] = records[codes == split.TEST_IID_CODE]
    return routed


def permute(records: np.ndarray, seed: int, name: str) -> np.ndarray:
    """A new array in a fixed order that depends only on the seed and the shard's name."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])
    return records[rng.permutation(len(records))]


def _split_of_shard(name: str) -> str:
    return "train" if name.startswith("train_") else name.removesuffix(".bin")


def write_shards(out: Path, routed: dict[str, np.ndarray], seed: int) -> dict[str, dict]:
    entries = {}
    for name in sorted(routed):
        data = permute(routed[name], seed, name).tobytes()
        write_atomic(out / name, data)
        entries[name] = {
            "split": _split_of_shard(name),
            "records": len(routed[name]),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return entries


def _split_totals(shards: dict[str, dict]) -> dict[str, int]:
    totals = dict.fromkeys(split.SPLITS, 0)
    for entry in shards.values():
        totals[entry["split"]] += entry["records"]
    return totals


def _prepare_out(out: Path, overwrite: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in out.iterdir() if p.name == MANIFEST or p.suffix == ".bin"]
    if existing and not overwrite:
        raise FileExistsError(
            f"{out} already holds a pack ({len(existing)} files); pass --overwrite to replace it"
        )


def _remove_stale(out: Path, keep: set[str]) -> None:
    for path in out.iterdir():
        if (path.suffix == ".bin" and path.name not in keep) or path.suffix == ".tmp":
            path.unlink()


def pack(cfg: PackConfig) -> dict:
    """Pack cfg.frames frames of cfg.source into cfg.out. Returns the manifest it wrote."""
    if cfg.shards < 1:
        raise ValueError(f"shards must be at least 1, got {cfg.shards}")
    _prepare_out(cfg.out, cfg.overwrite)
    blocked, blocklist_entry = load_blocklist(cfg.blocklist)
    got = collect(cfg)
    is_blocked = np.isin(got.records["fen_hash"], blocked)
    kept = got.records[~is_blocked]
    write_start = time.perf_counter()
    shards = write_shards(cfg.out, route(kept, cfg.shards), cfg.seed)
    manifest = {
        "format": FORMAT,
        "record_bytes": ROOT_DTYPE.itemsize,
        "source": {"path": str(cfg.source), "bytes": Path(cfg.source).stat().st_size},
        "frames": got.frames,
        "end": got.end,
        "lines": got.lines,
        "parsed": len(got.records),
        "rejects": dict(sorted(got.rejects.items())),
        "errors": dict(sorted(got.errors.items())),
        "error_samples": got.error_samples,
        "blocklist": blocklist_entry,
        "dropped_blocklisted": int(is_blocked.sum()),
        "records_written": len(kept),
        "duplicate_fen_hashes": int(len(kept) - len(np.unique(kept["fen_hash"]))),
        "splits": _split_totals(shards),
        "split_rule": split.SPLIT_RULE,
        "shard_rule": f"train shard = fen_hash % {cfg.shards}",
        "seed": cfg.seed,
        "shards": shards,
    }
    write_atomic(cfg.out / MANIFEST, json.dumps(manifest, indent=1).encode("utf-8"))
    _remove_stale(cfg.out, set(shards))
    timing = {**got.timing, "write_s": time.perf_counter() - write_start, "lines": got.lines}
    write_atomic(cfg.out / TIMING, json.dumps(timing, indent=1).encode("utf-8"))
    return manifest


def read_timing(pack_dir: Path) -> dict:
    return json.loads((Path(pack_dir) / TIMING).read_text(encoding="utf-8"))


def read_manifest(pack_dir: Path) -> dict:
    return json.loads((Path(pack_dir) / MANIFEST).read_text(encoding="utf-8"))


def split_paths(pack_dir: Path, split_name: str) -> list[Path]:
    """The shard files of one split, in name order, as the manifest lists them."""
    shards = read_manifest(pack_dir)["shards"]
    return [Path(pack_dir) / name for name in sorted(shards) if shards[name]["split"] == split_name]
