"""`blink data ladder10m`: the pack s10m trains on, holding exactly the ladder's fixed roots (plan P3, P5).

s10m (ladder rung 4) trains the frozen recipe on the positions the linear and MLP rungs saw. This
writes them as a v1-layout pack the trainer reads unchanged:
- train_rNNN.bin: the fixed set (blink.data.fixedset), each file the used front part of the source's
  train_rNNN.bin in the same order, so fixed_train_set on this pack returns the same records with the
  same fen_hash sha256 as the baselines record.
- train_cNNN.bin: the source's train children of those roots. bigpack routes a child by its own
  fen_hash, not by its root's, so the children of one root lie in any of the source's child shards.
  They are found by rebuilding every root's PV children (blink.data.children: PVs 1-5, the ones a
  root record keeps) and keeping the source child records with those hashes, which keeps bigpack's
  filters (blocklist, eval leakage, children that are DB roots) and its deepest labels. Here a child
  sits in the shard of the first fixed root that implies it, so train_cNNN.bin holds the children of
  train_rNNN.bin's roots. Children of PVs beyond the fifth cannot be rebuilt from a root record; the
  roots that had them are counted in the manifest.
- val_roots.bin (required), valprobe.npz and mateset.npz, copied from the source.
- manifest.json, written last: the source's rebalance table, blocklist, split rule and grouped salt
  (what the trainer's world id reads), the fixed set's description, the child counts, and every
  file's records, bytes and sha256. It carries no timestamp, so rebuilding the same set gives the same
  manifest and the same world.

The source is read front to back, one file at a time (PF14): the used root shards, then every child
shard once.
"""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from blink.data import bigpack, children, fixedset, pack
from blink.data.blocklist import contains
from blink.data.record import CHILD_DTYPE, NUM_ALTERNATIVES, ROOT_DTYPE

FORMAT = "blink-ladder-v1"
DEFAULT_POSITIONS = 10_000_000
REBUILD_CHUNK = 500_000  # roots per children_of call, bounding the RAM of the unpacked boards
KEPT_PVS = 1 + NUM_ALTERNATIVES
REQUIRED_EVAL = "val_roots.bin"
EVAL_FILES = (REQUIRED_EVAL, "valprobe.npz", "mateset.npz")
SOURCE_KEYS = ("record_bytes", "blocklist", "split_rule", "grouped", "rebalance")
SHARD_RULE = (
    "roots: train_rNNN.bin is the fixed set's front part of the source's train_rNNN.bin, in its order; "
    "children: train_cNNN.bin holds the source's train children (PVs 1-5) of those roots, each child "
    "with the first root that implies it"
)

Log = Callable[[str], None]


@dataclass(frozen=True)
class LadderConfig:
    pack: Path
    out: Path
    positions: int = DEFAULT_POSITIONS
    overwrite: bool = False


class Implied(NamedTuple):
    """The distinct children the fixed roots imply, sorted by fen_hash."""

    hashes: np.ndarray  # uint64, sorted and unique
    rank: np.ndarray  # int64: where each first appears when the roots are walked in order, PVs in order
    shard: np.ndarray  # int64: the root shard (index into the used shards) of the root that first implies it
    rebuilt: int  # children rebuilt, repeats included


# ---------------------------------------------------------------- checks


def check_source(pack_dir: Path) -> dict[str, Any]:
    """The source manifest, or an error saying what to build first."""
    if not (pack_dir / bigpack.MANIFEST).is_file():
        raise FileNotFoundError(
            f"{pack_dir} has no manifest.json: build the v1 pack first (blink data bigpack)"
        )
    manifest = bigpack.read_manifest(pack_dir)
    if manifest.get("format") != bigpack.FORMAT:
        raise ValueError(f"{pack_dir} is not a v1 pack (format {manifest.get('format')!r})")
    if manifest.get("status") != "complete":
        raise ValueError(f"{pack_dir} is not a complete pack (status {manifest.get('status')!r})")
    if not manifest.get("rebalance"):
        raise ValueError(
            f"{pack_dir} has no rebalance table yet, so s10m would train unweighted: "
            f"run `blink data rebalance --pack {pack_dir}` first"
        )
    if not (pack_dir / REQUIRED_EVAL).is_file():
        raise FileNotFoundError(f"{pack_dir} has no {REQUIRED_EVAL}")
    return manifest


def _ours(path: Path) -> bool:
    name = path.name
    return name in (bigpack.MANIFEST, *EVAL_FILES) or name.endswith(".tmp") or name.startswith("train_")


def prepare_out(cfg: LadderConfig) -> None:
    """Refuse the source itself, another kind of pack, or a used folder without overwrite."""
    out = Path(cfg.out)
    if out.resolve() == Path(cfg.pack).resolve():
        raise ValueError(f"--out {out} is the source pack itself")
    if out.exists() and any(out.iterdir()):
        if (out / bigpack.MANIFEST).is_file() and bigpack.read_manifest(out).get("format") != FORMAT:
            raise FileExistsError(f"{out} holds another kind of pack; choose another --out")
        if not cfg.overwrite:
            raise FileExistsError(f"{out} is not empty; choose another --out or pass --overwrite")
        for path in out.iterdir():
            if path.is_file() and _ours(path):
                path.unlink()
    out.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- the children of the fixed roots


def implied_children(roots: np.ndarray, shard_of_root: np.ndarray, chunk: int = REBUILD_CHUNK) -> Implied:
    """Every child the roots' records imply, once each, with the root shard of its first root."""
    hashes, parents = [], []
    for start in range(0, len(roots), chunk):
        kids = children.children_of(roots[start : start + chunk])
        hashes.append(kids.records["fen_hash"])
        parents.append(kids.parent + start)
    every = np.concatenate(hashes) if hashes else np.zeros(0, dtype=np.uint64)
    parent = np.concatenate(parents) if parents else np.zeros(0, dtype=np.int64)
    unique, first = np.unique(every, return_index=True)
    return Implied(unique, first.astype(np.int64), shard_of_root[parent[first]].astype(np.int64), len(every))


def collect_children(child_shards: list[Path], implied: Implied, log: Log) -> tuple[np.ndarray, np.ndarray]:
    """The source's records of the implied children, and each one's index into `implied`."""
    found, where = [], []
    for number, path in enumerate(child_shards, start=1):
        records = np.fromfile(path, dtype=CHILD_DTYPE)
        kept = records[contains(implied.hashes, records["fen_hash"])]
        found.append(kept)
        where.append(np.searchsorted(implied.hashes, kept["fen_hash"]))
        if number % 16 == 0 or number == len(child_shards):
            held = sum(map(len, found))
            log(f"ladder: scanned {number} of {len(child_shards)} child shards, {held:,} children found")
    kids = np.concatenate(found) if found else np.zeros(0, dtype=CHILD_DTYPE)
    index = np.concatenate(where) if where else np.zeros(0, dtype=np.int64)
    if len(np.unique(kids["fen_hash"])) != len(kids):
        raise ValueError("the source child shards hold a child twice; verify the pack (blink data verify)")
    return kids, index


# ---------------------------------------------------------------- writing


def _entry(data: bytes, **fields: Any) -> dict[str, Any]:
    return {**fields, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write(out: Path, name: str, records: np.ndarray, kind: str, **fields: Any) -> dict[str, Any]:
    data = records.tobytes()
    pack.write_atomic(out / name, data)
    return _entry(data, split="train", kind=kind, records=len(records), **fields)


def shard_sizes(pack_dir: Path, used: list[str], positions: int) -> list[int]:
    """Records taken from each used root shard: all of each but the last, and the rest of the set."""
    whole = [os.path.getsize(pack_dir / name) // ROOT_DTYPE.itemsize for name in used[:-1]]
    return [*whole, positions - sum(whole)]


def write_roots(out: Path, roots: np.ndarray, used: list[str], sizes: list[int]) -> dict[str, dict]:
    entries, start = {}, 0
    for name, size in zip(used, sizes, strict=True):
        entries[name] = _write(out, name, roots[start : start + size], "roots", source_records=size)
        start += size
    return entries


def write_children(out: Path, kids: np.ndarray, rank: np.ndarray, shard: np.ndarray, used: list[str]) -> dict:
    """Each root shard's children, in the order their roots and PVs imply them; empty shards are skipped."""
    order = np.argsort(rank, kind="stable")
    kids, shard = kids[order], shard[order]
    entries = {}
    for index, root_name in enumerate(used):
        mine = kids[shard == index]
        if len(mine):
            name = root_name.replace("train_r", "train_c", 1)
            entries[name] = _write(out, name, mine, "children", roots_shard=root_name)
    return entries


def copy_eval_files(source: Path, out: Path) -> dict[str, dict]:
    entries = {}
    for name in EVAL_FILES:
        if (source / name).is_file():
            data = (source / name).read_bytes()
            pack.write_atomic(out / name, data)
            entries[name] = _entry(data, source=str(source / name))
    return entries


# ---------------------------------------------------------------- the command


def build(cfg: LadderConfig, log: Log = print) -> dict[str, Any]:
    """Write the ladder pack into cfg.out and return its manifest."""
    source_dir, out = Path(cfg.pack), Path(cfg.out)
    source = check_source(source_dir)
    roots, description = fixedset.fixed_train_set(source_dir, cfg.positions)
    prepare_out(cfg)
    used = description["shards"]
    sizes = shard_sizes(source_dir, used, cfg.positions)
    log(f"ladder: {cfg.positions:,} roots from {len(used)} root shards, {description['fen_hash_sha256']}")
    implied = implied_children(roots, np.repeat(np.arange(len(used)), sizes))
    log(f"ladder: {implied.rebuilt:,} children rebuilt, {len(implied.hashes):,} distinct")
    kids, index = collect_children(bigpack.shard_paths(source_dir, "train", "children"), implied, log)
    shards = write_roots(out, roots, used, sizes)
    shards |= write_children(out, kids, implied.rank[index], implied.shard[index], used)
    manifest = {
        "format": FORMAT,
        "status": "complete",
        "source": {
            "dir": str(source_dir),
            "format": source["format"],
            "manifest_sha256": hashlib.sha256((source_dir / bigpack.MANIFEST).read_bytes()).hexdigest(),
        },
        "fixed_set": description,
        "shard_rule": SHARD_RULE,
        "children": {
            "rebuilt": implied.rebuilt,
            "distinct": len(implied.hashes),
            "found": len(kids),
            "not_in_pack": len(implied.hashes) - len(kids),
            "roots_with_pvs_beyond_the_fifth": int((roots["npv"] > KEPT_PVS).sum()),
        },
        **{key: source[key] for key in SOURCE_KEYS if key in source},
        "shards": shards,
        "eval_files": copy_eval_files(source_dir, out),
    }
    check_written(out, description)
    return bigpack.write_manifest(out, manifest)


def check_written(out: Path, description: dict[str, Any]) -> None:
    """Read the written roots back as the baselines would: the same set, or no manifest is written."""
    _, again = fixedset.fixed_train_set(out, description["positions"])
    if again["fen_hash_sha256"] != description["fen_hash_sha256"]:
        raise RuntimeError(f"{out}: the written roots are not the fixed set ({again['fen_hash_sha256']})")


def summary(manifest: dict[str, Any], out: Path) -> str:
    fixed, kids, shards = manifest["fixed_set"], manifest["children"], manifest["shards"]
    kinds = [entry["kind"] for entry in shards.values()]
    return "\n".join(
        [
            f"roots       {fixed['positions']:,} from {', '.join(fixed['shards'])}",
            f"fixed set   fen_hash sha256 {fixed['fen_hash_sha256']}",
            f"children    {kids['found']:,} of {kids['distinct']:,} implied ({kids['not_in_pack']:,} are not "
            f"train children of the pack; {kids['roots_with_pvs_beyond_the_fifth']:,} roots had more PVs)",
            f"files       {kinds.count('roots')} root and {kinds.count('children')} child shards; "
            f"{', '.join(manifest['eval_files'])}",
            f"wrote {Path(out) / bigpack.MANIFEST}",
        ]
    )
