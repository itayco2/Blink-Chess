"""The v1 pack: every root and child of the eval DB, as 256 root and 256 child train shards plus eval files.

D: is a spinning disk, so the pack is two sequential passes (PF14).

Pass 1 reads the pzstd source once, front to back. Spawned workers decode whole frames, parse roots and
their children (blink.data.rows, blink.data.children) and give each root a split: val and test_iid by
fen_hash, then test_grouped by group (blink.data.grouped); children inherit their root's split, and a
train child whose own group is held out is dropped. The parent routes train roots and children by
fen_hash % 256 into 256 root and 256 child bucket files through ~8 MB buffers, so a child and a root with
the same position always share a bucket. Val and test roots and children go to their own files, which
are filtered, de-duplicated and written at the end of pass 1 (manifest status "pass2").

Pass 2 turns each bucket pair into train_rNNN.bin and train_cNNN.bin, dropping: blocklisted positions;
train roots equal to any val or test root or child; children equal to a val or test root or child;
children that are also DB roots (the root's own, deeper eval wins); duplicate children (the deepest
label wins). Each shard is permuted with a fixed seed, written as .tmp then os.replace, and recorded with
its sha256 in the manifest before its bucket pair is deleted. A resume skips recorded shards; pass 1
always restarts from zero; a source whose size or ETag changed stops the resume.
"""

import functools
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from blink.board.value import win_probability_array
from blink.data import children, frames, grouped, pack, rebalance, rows, split, zst
from blink.data.blocklist import contains
from blink.data.record import CHILD_DTYPE, ROOT_DTYPE

FORMAT = "blink-bigpack-v1"
MANIFEST = "manifest.json"
BUCKETS = 256
BUFFER_BYTES = 8 << 20
BUCKET_DIR = "buckets"
EVAL_HASHES = "eval_hashes.npy"
DEFAULT_SEED = pack.DEFAULT_SEED
SPLITS = ("train", "val", "test_iid", "test_grouped")
TRAIN, VAL, TEST_IID, TEST_GROUPED = range(4)
EVAL_SPLITS = SPLITS[1:]
KINDS = ("roots", "children")
DTYPES = {"roots": ROOT_DTYPE, "children": CHILD_DTYPE}
DERIVED = ("verify.json", "valprobe.npz", "mateset.npz", "data_stats.json")
DELETE_RETRIES = 10
_ETAG = re.compile(r"^etag:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


class SourceChanged(RuntimeError):
    """The source file is not the one pass 1 read: stop and ask before mixing two uploads."""


@dataclass(frozen=True)
class BigPackConfig:
    source: Path
    out: Path
    salt: int  # the test_grouped salt, chosen on a probe (blink.data.grouped.choose_salt)
    workers: int = 1
    limit_frames: int | None = None
    blocklist: Path | None = None
    seed: int = DEFAULT_SEED
    buckets: int = BUCKETS
    buffer_bytes: int = BUFFER_BYTES
    resume: bool = False
    overwrite: bool = False
    grouped_probe: dict | None = None  # the SaltChoice record, kept in the manifest


def source_identity(path: Path) -> dict:
    """Path, size and (when a `<name>.head.txt` from the download sits beside it) the HTTP ETag."""
    path = Path(path)
    head = path.with_name(path.name + ".head.txt")
    etag = None
    if head.is_file():
        found = _ETAG.search(head.read_text(encoding="utf-8"))
        etag = found.group(1) if found else None
    return {"path": str(path), "bytes": path.stat().st_size, "etag": etag}


# --- pass 1: workers ------------------------------------------------------------------------------------


class Pass1Chunk(NamedTuple):
    roots: tuple[np.ndarray, ...]  # by split code (TRAIN, VAL, TEST_IID, TEST_GROUPED)
    children: tuple[np.ndarray, ...]
    rejects: dict[str, int]
    errors: dict[str, int]
    error_samples: list[str]
    grouped_children_dropped: int  # train children whose own group is held out


def assign_splits(roots: np.ndarray, salt: int) -> np.ndarray:
    """uint8 split codes: val and test_iid by fen_hash (blink.data.split), then test_grouped by group."""
    codes = split.split_codes(roots["fen_hash"])
    train = np.flatnonzero(codes == split.TRAIN_CODE)
    if len(train):
        codes[train[grouped.selected(roots["board"][train], salt)]] = TEST_GROUPED
    return codes


def pass1_lines(lines: list[bytes], salt: int) -> Pass1Chunk:
    """Parse, make children and split some lines. Module-level so spawned workers can run it."""
    parsed = rows.parse_rows(lines)
    kids = children.children_of(parsed.roots, parsed.extras)
    root_split = assign_splits(parsed.roots, salt)
    child_split = root_split[kids.parent]
    held = np.zeros(len(kids.records), dtype=bool)
    train_kids = np.flatnonzero(child_split == TRAIN)
    if len(train_kids):
        held[train_kids] = grouped.selected(kids.records["board"][train_kids], salt)
    return Pass1Chunk(
        tuple(parsed.roots[root_split == code] for code in range(len(SPLITS))),
        tuple(kids.records[~held & (child_split == code)] for code in range(len(SPLITS))),
        parsed.rejects,
        parsed.errors,
        parsed.error_samples,
        int(held.sum()),
    )


# --- pass 1: the parent ---------------------------------------------------------------------------------


class BucketWriter:
    """Append-only files, each with an in-RAM buffer written out in one sequential write when full."""

    def __init__(self, paths: list[Path], buffer_bytes: int) -> None:
        self.paths = paths
        self.buffer_bytes = buffer_bytes
        self.pending: list[list[np.ndarray]] = [[] for _ in paths]
        self.held = [0] * len(paths)
        self.flush_s = 0.0
        self.flushed_bytes = 0
        for path in paths:
            path.write_bytes(b"")

    def add(self, index: int, records: np.ndarray) -> None:
        if not len(records):
            return
        self.pending[index].append(records)
        self.held[index] += records.nbytes
        if self.held[index] >= self.buffer_bytes:
            self._flush(index)

    def add_routed(self, records: np.ndarray, bucket_ids: np.ndarray) -> None:
        """Route records to bucket_ids (arrival order kept inside each bucket)."""
        if not len(records):
            return
        bucket_ids = bucket_ids.astype(np.int64)
        order = np.argsort(bucket_ids, kind="stable")
        counts = np.bincount(bucket_ids, minlength=len(self.paths))
        starts = np.concatenate([[0], np.cumsum(counts)])
        ordered = records[order]
        for bucket in np.flatnonzero(counts):
            self.add(int(bucket), ordered[starts[bucket] : starts[bucket + 1]])

    def _flush(self, index: int) -> None:
        data = b"".join(chunk.tobytes() for chunk in self.pending[index])
        start = time.perf_counter()
        with open(self.paths[index], "ab") as handle:
            handle.write(data)
        self.flush_s += time.perf_counter() - start
        self.flushed_bytes += len(data)
        self.pending[index], self.held[index] = [], 0

    def close(self) -> None:
        for index, pending in enumerate(self.pending):
            if pending:
                self._flush(index)


class _Pass1Tally:
    def __init__(self) -> None:
        self.lines = self.grouped_children_dropped = 0
        self.roots, self.children = Counter(), Counter()
        self.rejects, self.errors = Counter(), Counter()
        self.samples: list[str] = []
        self.decode_s = self.work_s = 0.0
        self.compressed = self.decompressed = 0

    def add(self, out: frames.FrameOutput) -> None:
        chunk: Pass1Chunk = out.result
        self.lines += out.lines
        for code, name in enumerate(SPLITS):
            self.roots[name] += len(chunk.roots[code])
            self.children[name] += len(chunk.children[code])
        self.rejects.update(chunk.rejects)
        self.errors.update(chunk.errors)
        self.samples.extend(chunk.error_samples[: pack.ERROR_SAMPLES - len(self.samples)])
        self.grouped_children_dropped += chunk.grouped_children_dropped
        self.decode_s += out.decode_s
        self.work_s += out.work_s
        self.compressed += out.compressed
        self.decompressed += out.decompressed


def _writers(cfg: BigPackConfig) -> dict[str, BucketWriter]:
    folder = cfg.out / BUCKET_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return {
        "roots": BucketWriter([folder / f"r{b:03d}.bin" for b in range(cfg.buckets)], cfg.buffer_bytes),
        "children": BucketWriter([folder / f"c{b:03d}.bin" for b in range(cfg.buckets)], cfg.buffer_bytes),
        "eval_roots": BucketWriter([folder / f"{s}_roots.part" for s in EVAL_SPLITS], cfg.buffer_bytes),
        "eval_children": BucketWriter([folder / f"{s}_children.part" for s in EVAL_SPLITS], cfg.buffer_bytes),
    }


def _route(chunk: Pass1Chunk, writers: dict[str, BucketWriter], buckets: int) -> None:
    for kind, arrays in (("roots", chunk.roots), ("children", chunk.children)):
        train = arrays[TRAIN]
        writers[kind].add_routed(train, train["fen_hash"] % np.uint64(buckets))
        for index, code in enumerate((VAL, TEST_IID, TEST_GROUPED)):
            writers[f"eval_{kind}"].add(index, arrays[code])


def run_pass1(cfg: BigPackConfig) -> dict:
    """Read the source once and fill the bucket and eval part files. Returns the pass-1 numbers."""
    writers = _writers(cfg)
    reader = zst.FrameReader(cfg.source, cfg.limit_frames)
    work = functools.partial(pass1_lines, salt=cfg.salt)
    tally = _Pass1Tally()
    start = time.perf_counter()
    for out in frames.run_frames(reader, work, cfg.workers):
        _route(out.result, writers, cfg.buckets)
        tally.add(out)
    for writer in writers.values():
        writer.close()
    wall = time.perf_counter() - start
    flush_s = sum(w.flush_s for w in writers.values())
    flushed = sum(w.flushed_bytes for w in writers.values())
    timing = {
        "wall_s": wall,
        "lines_per_s": tally.lines / wall if wall else 0.0,
        "workers": cfg.workers,
        "decode_s": tally.decode_s,
        "work_s": tally.work_s,
        "compressed_bytes": tally.compressed,
        "decompressed_bytes": tally.decompressed,
        "flush_s": flush_s,
        "flushed_bytes": flushed,
        "flush_mb_per_s": flushed / 1e6 / flush_s if flush_s else 0.0,
    }
    return {
        "frames": reader.frames_read,
        "end": reader.end,
        "lines": tally.lines,
        "parsed_roots": sum(tally.roots.values()),
        "rejects": dict(sorted(tally.rejects.items())),
        "errors": dict(sorted(tally.errors.items())),
        "error_samples": tally.samples,
        "pass1": {
            "roots": {name: tally.roots[name] for name in SPLITS},
            "children": {name: tally.children[name] for name in SPLITS},
            "grouped_children_dropped": tally.grouped_children_dropped,
            "timing": timing,
        },
    }


# --- shards ---------------------------------------------------------------------------------------------


def write_shard(out: Path, name: str, records: np.ndarray, split_name: str, kind: str, seed: int) -> dict:
    """Permute, write atomically, and describe one shard for the manifest."""
    data = pack.permute(records, seed, name).tobytes()
    pack.write_atomic(out / name, data)
    wins = win_probability_array(records["cp"], records["mate"])
    return {
        "split": split_name,
        "kind": kind,
        "records": len(records),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bucket_counts": np.bincount(rebalance.bucket_of(records), minlength=rebalance.NUM_BUCKETS).tolist(),
        "mean_win": float(wins.mean()) if len(records) else None,
    }


def _read(path: Path, dtype: np.dtype) -> np.ndarray:
    return np.fromfile(path, dtype=dtype)


def finalize_eval(cfg: BigPackConfig, blocked: np.ndarray) -> tuple[dict, dict]:
    """Write the val and test files; save every val/test hash for the pass-2 cross filters."""
    folder = cfg.out / BUCKET_DIR
    roots = {s: _read(folder / f"{s}_roots.part", ROOT_DTYPE) for s in EVAL_SPLITS}
    kids = {s: _read(folder / f"{s}_children.part", CHILD_DTYPE) for s in EVAL_SPLITS}
    root_hashes = np.unique(np.concatenate([r["fen_hash"] for r in roots.values()]))
    every = np.unique(np.concatenate([root_hashes] + [k["fen_hash"] for k in kids.values()]))
    entries, dropped = {}, Counter()
    for name in EVAL_SPLITS:
        r, k = roots[name], kids[name]
        r_blocked = contains(blocked, r["fen_hash"])
        k_blocked = contains(blocked, k["fen_hash"])
        k_root = contains(root_hashes, k["fen_hash"]) & ~k_blocked
        kept, duplicates = children.dedupe_deepest(k[~(k_blocked | k_root)])
        dropped.update(
            {
                f"{name}_roots_blocklisted": int(r_blocked.sum()),
                f"{name}_children_blocklisted": int(k_blocked.sum()),
                f"{name}_children_that_are_roots": int(k_root.sum()),
                f"{name}_children_duplicate": duplicates,
            }
        )
        for kind, records in (("roots", r[~r_blocked]), ("children", kept)):
            file = f"{name}_{kind}.bin"
            entries[file] = write_shard(cfg.out, file, records, name, kind, cfg.seed)
    tmp = folder / ("tmp_" + EVAL_HASHES)
    np.save(tmp, every)
    os.replace(tmp, folder / EVAL_HASHES)
    return entries, dict(dropped)


def pack_bucket(bucket: int, cfg: BigPackConfig, blocked: np.ndarray, eval_hashes: np.ndarray) -> dict:
    """One bucket pair to its two train shards, with every pass-2 filter. Returns their manifest entries."""
    start = time.perf_counter()
    folder = cfg.out / BUCKET_DIR
    roots = _read(folder / f"r{bucket:03d}.bin", ROOT_DTYPE)
    kids = _read(folder / f"c{bucket:03d}.bin", CHILD_DTYPE)
    db_roots = np.unique(roots["fen_hash"])
    r_blocked = contains(blocked, roots["fen_hash"])
    r_eval = contains(eval_hashes, roots["fen_hash"]) & ~r_blocked
    k_blocked = contains(blocked, kids["fen_hash"])
    k_eval = contains(eval_hashes, kids["fen_hash"]) & ~k_blocked
    k_root = contains(db_roots, kids["fen_hash"]) & ~k_blocked & ~k_eval
    kept, duplicates = children.dedupe_deepest(kids[~(k_blocked | k_eval | k_root)])
    names = (f"train_r{bucket:03d}.bin", f"train_c{bucket:03d}.bin")
    root_entry = write_shard(cfg.out, names[0], roots[~(r_blocked | r_eval)], "train", "roots", cfg.seed)
    child_entry = write_shard(cfg.out, names[1], kept, "train", "children", cfg.seed)
    root_entry["dropped"] = {
        "roots_blocklisted": int(r_blocked.sum()),
        "roots_equal_to_eval": int(r_eval.sum()),
    }
    child_entry["dropped"] = {
        "children_blocklisted": int(k_blocked.sum()),
        "children_equal_to_eval": int(k_eval.sum()),
        "children_that_are_roots": int(k_root.sum()),
        "children_duplicate": duplicates,
    }
    root_entry["input_records"], child_entry["input_records"] = len(roots), len(kids)
    root_entry["seconds"] = time.perf_counter() - start
    return {names[0]: root_entry, names[1]: child_entry}


# --- manifest, resume, pass 2 ---------------------------------------------------------------------------


def read_manifest(pack_dir: Path) -> dict:
    return json.loads((Path(pack_dir) / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(pack_dir: Path, manifest: dict) -> dict:
    text = json.dumps(manifest, indent=1)
    pack.write_atomic(Path(pack_dir) / MANIFEST, text.encode("utf-8"))
    return json.loads(text)


def shard_paths(pack_dir: Path, split_name: str, kind: str) -> list[Path]:
    """The files of one split and kind ("roots" or "children"), in name order."""
    shards = read_manifest(pack_dir)["shards"]
    names = sorted(n for n, e in shards.items() if e["split"] == split_name and e["kind"] == kind)
    return [Path(pack_dir) / name for name in names]


def _unlink(path: Path) -> None:
    for attempt in range(DELETE_RETRIES):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:  # a scanner holds it for a moment (Windows)
            if attempt == DELETE_RETRIES - 1:
                raise
            time.sleep(0.2)


def _sum_counts(rows_: list[list[int]]) -> list[int]:
    total = np.zeros(rebalance.NUM_BUCKETS, dtype=np.int64)
    for row in rows_:
        total += np.asarray(row, dtype=np.int64)
    return total.tolist()


def _sum_dropped(entries: list[dict]) -> dict:
    total: Counter = Counter()
    for entry in entries:
        total.update(entry.get("dropped", {}))
    return dict(sorted(total.items()))


def _complete(cfg: BigPackConfig, manifest: dict) -> dict:
    shards = manifest["shards"]
    train = [e for n, e in shards.items() if n.startswith("train_")]
    splits = {
        k: {
            s: sum(e["records"] for e in shards.values() if (e["split"], e["kind"]) == (s, k)) for s in SPLITS
        }
        for k in KINDS
    }
    hist = {k: _sum_counts([e["bucket_counts"] for e in train if e["kind"] == k]) for k in KINDS}
    seconds = sum(e.get("seconds", 0.0) for e in train)
    inputs = sum(e.get("input_records", 0) for e in train)
    shutil.rmtree(cfg.out / BUCKET_DIR)
    return write_manifest(
        cfg.out,
        {
            **manifest,
            "status": "complete",
            "splits": splits,
            "records_written": sum(sum(v.values()) for v in splits.values()),
            "evaldb_hist": hist,
            "pass2": {
                "dropped": _sum_dropped(train),
                "timing": {
                    "seconds": seconds,
                    "input_records": inputs,
                    "records_per_s": inputs / seconds if seconds else 0.0,
                },
            },
        },
    )


def run_pass2(cfg: BigPackConfig, manifest: dict, blocked: np.ndarray) -> dict:
    folder = cfg.out / BUCKET_DIR
    if not (folder / EVAL_HASHES).is_file():
        raise FileNotFoundError(
            f"{folder / EVAL_HASHES} is gone, so the unfinished pass 2 in {cfg.out} cannot resume; "
            "pass --overwrite to pack again"
        )
    eval_hashes = np.load(folder / EVAL_HASHES)
    for bucket in range(cfg.buckets):
        names = (f"train_r{bucket:03d}.bin", f"train_c{bucket:03d}.bin")
        if not all(name in manifest["shards"] for name in names):
            entries = pack_bucket(bucket, cfg, blocked, eval_hashes)
            manifest = write_manifest(cfg.out, {**manifest, "shards": {**manifest["shards"], **entries}})
        _unlink(folder / f"r{bucket:03d}.bin")
        _unlink(folder / f"c{bucket:03d}.bin")
    return _complete(cfg, manifest)


def _settings(cfg: BigPackConfig, blocklist_entry: dict | None) -> dict:
    return {
        "salt": cfg.salt,
        "seed": cfg.seed,
        "buckets": cfg.buckets,
        "limit_frames": cfg.limit_frames,
        "blocklist_sha256": blocklist_entry["sha256"] if blocklist_entry else None,
    }


def _check_resume(manifest: dict, cfg: BigPackConfig, identity: dict, blocklist_entry: dict | None) -> None:
    old = manifest["source"]
    for field in ("bytes", "etag"):
        if old[field] != identity[field]:
            raise SourceChanged(
                f"the source {field} changed since pass 1 ({old[field]} -> {identity[field]}) at "
                f"{identity['path']}: a new upload cannot be mixed into this pack; stop and ask"
            )
    wanted = _settings(cfg, blocklist_entry)
    if manifest["settings"] != wanted:
        raise ValueError(f"resume with other settings: pack has {manifest['settings']}, asked {wanted}")


def _clear(out: Path) -> None:
    """Remove a previous pack's files from `out` (only names this module writes)."""
    if (out / BUCKET_DIR).exists():
        shutil.rmtree(out / BUCKET_DIR)
    for path in out.iterdir():
        if path.name in (MANIFEST, *DERIVED) or path.suffix in (".bin", ".tmp"):
            _unlink(path)


def _has_pack_files(out: Path) -> bool:
    return any(p.name == MANIFEST or p.suffix == ".bin" or p.name == BUCKET_DIR for p in out.iterdir())


def _start(cfg: BigPackConfig, identity: dict, blocked: np.ndarray, blocklist_entry: dict | None) -> dict:
    """Pass 1 from zero, then the eval files; writes the manifest with status "pass2"."""
    _clear(cfg.out)
    numbers = run_pass1(cfg)
    start = time.perf_counter()
    entries, eval_dropped = finalize_eval(cfg, blocked)
    numbers["pass1"]["timing"]["eval_files_s"] = time.perf_counter() - start
    manifest = {
        "format": FORMAT,
        "status": "pass2",
        "record_bytes": {"roots": ROOT_DTYPE.itemsize, "children": CHILD_DTYPE.itemsize},
        "source": identity,
        "settings": _settings(cfg, blocklist_entry),
        "seed": cfg.seed,
        "buckets": cfg.buckets,
        "blocklist": blocklist_entry,
        "grouped": {"salt": cfg.salt, "rule": grouped.RULE, "probe": cfg.grouped_probe},
        "split_rule": split.SPLIT_RULE,
        "shard_rule": f"train roots and children: fen_hash % {cfg.buckets}; a child takes its root's split",
        **numbers,
        "eval_dropped": eval_dropped,
        "shards": entries,
    }
    return write_manifest(cfg.out, manifest)


def bigpack(cfg: BigPackConfig) -> dict:
    """Build (or finish, with resume) the pack in cfg.out. Returns the final manifest."""
    if cfg.buckets < 1 or cfg.buffer_bytes < 1:
        raise ValueError(f"need buckets >= 1 and buffer_bytes >= 1, got {cfg.buckets} and {cfg.buffer_bytes}")
    cfg.out.mkdir(parents=True, exist_ok=True)
    identity = source_identity(cfg.source)
    blocked, blocklist_entry = pack.load_blocklist(cfg.blocklist)
    manifest = read_manifest(cfg.out) if (cfg.out / MANIFEST).is_file() else None
    if manifest is not None and manifest.get("format") != FORMAT and not cfg.overwrite:
        raise FileExistsError(
            f"{cfg.out} holds a {manifest.get('format')!r} pack, not {FORMAT!r}; choose another --out "
            "(or --overwrite to delete it)"
        )
    if manifest is not None and not cfg.overwrite:
        if not cfg.resume:
            raise FileExistsError(
                f"{cfg.out} holds a pack with status {manifest['status']!r}; pass --resume to finish or keep "
                "it, or --overwrite to replace it"
            )
        _check_resume(manifest, cfg, identity, blocklist_entry)
        if manifest["status"] == "complete":
            return manifest
        return run_pass2(cfg, manifest, blocked)
    if manifest is None and _has_pack_files(cfg.out) and not (cfg.resume or cfg.overwrite):
        raise FileExistsError(
            f"{cfg.out} holds an unfinished pass 1; pass --resume or --overwrite to restart it"
        )
    return run_pass2(cfg, _start(cfg, identity, blocked, blocklist_entry), blocked)


def write_rebalance(pack_dir: Path, games: rebalance.GamesHistogram) -> dict:
    """Fill manifest["rebalance"] from a games histogram and the packed train roots. Returns the block."""
    manifest = read_manifest(pack_dir)
    if manifest.get("status") != "complete":
        raise ValueError(f"{pack_dir} is not a complete pack (status {manifest.get('status')!r})")
    evaldb = manifest["evaldb_hist"]["roots"]
    block = {
        "buckets": rebalance.NUM_BUCKETS,
        "weights": rebalance.table(games.counts, evaldb).tolist(),
        "definition": rebalance.DEFINITION,
        "p_games": [int(c) for c in games.counts],
        "p_evaldb": evaldb,
        "games": {k: v for k, v in games.as_dict().items() if k != "counts"},
    }
    return write_manifest(pack_dir, {**manifest, "rebalance": block})["rebalance"]
