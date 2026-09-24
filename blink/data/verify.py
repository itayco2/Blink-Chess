"""verify: one front-to-back scan of a finished v1 pack, written to verify.json.

Checked on every train record (roots and children): sha256 against the manifest, the right bucket,
no hit in the blocklist or in val, test_iid, test_grouped (roots and children), no child equal to a train
root, no duplicate child. Checked on a random sample (1% by default): the fen_hash is the board's hash,
the best and alternative moves are legal, the position is valid for python-chess, and no record belongs
to a held-out group. Measured: PV monotonicity (side-to-move win% never rises from PV 1 down; a sign flip
drives it to about 50%), split fractions, shard sizes and each root shard's mean win%.
When valprobe.npz exists, the share of its children that are also train positions is reported.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blink.board import encode, moves
from blink.board.value import win_probability_array
from blink.data import bigpack, children, grouped, pack
from blink.data.blocklist import contains
from blink.data.record import CHILD_DTYPE, NO_MOVE, ROOT_DTYPE

REPORT = "verify.json"
SPLIT_TARGETS_PCT = {"val": (0.200, 0.01), "test_iid": (0.200, 0.01), "test_grouped": (0.10, 0.05)}
MONOTONE_MIN_PCT = 99.9
SHARD_SIZE_DEV_PCT = 2.0
MEAN_WIN_DEV_PT = 0.5
TIE = 1e-12


@dataclass(frozen=True)
class VerifyConfig:
    pack_dir: Path
    blocklist: Path | None = None  # default: the blocklist the manifest names
    legality_sample: float = 0.01
    seed: int = 0


def _check(value, limit: str, ok: bool) -> dict:
    return {"value": value, "limit": limit, "ok": bool(ok)}


def monotone_counts(roots: np.ndarray) -> tuple[int, int]:
    """(roots with an alternative, roots whose win% never rises from PV 1 through the alternatives)."""
    last = win_probability_array(roots["cp"], roots["mate"])
    ok = np.ones(len(roots), dtype=bool)
    multi = np.zeros(len(roots), dtype=bool)
    for slot in range(roots["alt_move"].shape[1]):
        present = roots["alt_move"][:, slot] != NO_MOVE
        wins = win_probability_array(roots["alt_cp"][:, slot], roots["alt_mate"][:, slot])
        ok &= ~present | (wins <= last + TIE)
        last = np.where(present, wins, last)
        multi |= present
    return int(multi.sum()), int((ok & multi).sum())


def split_checks(roots_by_split: dict[str, int]) -> dict:
    total = sum(roots_by_split.values())
    out = {}
    for name, (target, tolerance) in SPLIT_TARGETS_PCT.items():
        pct = 100.0 * roots_by_split[name] / total if total else 0.0
        out[f"{name}_pct"] = _check(pct, f"{target} +- {tolerance}", abs(pct - target) <= tolerance)
    return out


def _deviation_pct(sizes: list[int]) -> float:
    mean = float(np.mean(sizes)) if len(sizes) else 0.0
    return 100.0 * float(np.max(np.abs(np.asarray(sizes) - mean))) / mean if mean else 0.0


def balance_checks(root_sizes: list[int], child_sizes: list[int], mean_wins: list, global_win: float) -> dict:
    wins = [w for w in mean_wins if w is not None]
    win_dev = 100.0 * max((abs(w - global_win) for w in wins), default=0.0)
    root_dev, child_dev = _deviation_pct(root_sizes), _deviation_pct(child_sizes)
    return {
        "root_shard_size_dev_pct": _check(
            root_dev, f"<= {SHARD_SIZE_DEV_PCT}", root_dev <= SHARD_SIZE_DEV_PCT
        ),
        "child_shard_size_dev_pct": _check(
            child_dev, f"<= {SHARD_SIZE_DEV_PCT}", child_dev <= SHARD_SIZE_DEV_PCT
        ),
        "shard_mean_win_dev_pt": _check(win_dev, f"<= {MEAN_WIN_DEV_PT}", win_dev <= MEAN_WIN_DEV_PT),
    }


COUNTS = (
    "train_roots",
    "train_children",
    "blocklist_hits",
    "eval_hits",
    "children_equal_roots",
    "duplicate_children",
    "wrong_bucket",
    "sha256_mismatches",
    "fen_hash_mismatches",
    "grouped_leaks",
    "sampled_roots",
    "sampled_children",
    "legal_best",
    "alts",
    "legal_alts",
    "valid",
    "multi_pv",
    "monotone",
)


class _Tally:
    def __init__(self) -> None:
        self.n = dict.fromkeys(COUNTS, 0)
        self.root_sizes: list[int] = []
        self.child_sizes: list[int] = []
        self.mean_wins: list = []
        self.win_sum = 0.0

    def add(self, key: str, count) -> None:
        self.n[key] += int(count)


def _read_checked(path: Path, entry: dict, dtype: np.dtype, tally: _Tally) -> np.ndarray:
    data = path.read_bytes()
    tally.add("sha256_mismatches", hashlib.sha256(data).hexdigest() != entry["sha256"])
    return np.frombuffer(data, dtype=dtype)


def _sample_roots(recs: np.ndarray, tally: _Tally) -> None:
    for rec in recs:
        board = children.codes_to_board(encode.unpack(rec["board"]))
        legal = moves.legal_mask(board)
        tally.add("legal_best", legal[int(rec["move"])])
        alts = [int(m) for m in rec["alt_move"] if int(m) != NO_MOVE]
        tally.add("alts", len(alts))
        tally.add("legal_alts", sum(bool(legal[m]) for m in alts))
        tally.add("valid", board.is_valid())


def _sample_children(recs: np.ndarray, tally: _Tally) -> None:
    for rec in recs:
        tally.add("valid", children.codes_to_board(encode.unpack(rec["board"])).is_valid())


def _sample(recs: np.ndarray, kind: str, bucket: int, cfg: VerifyConfig, salt: int, tally: _Tally) -> None:
    rng = np.random.default_rng([cfg.seed, bucket, 0 if kind == "roots" else 1])
    chosen = recs[rng.random(len(recs)) < cfg.legality_sample]
    tally.add(f"sampled_{kind}", len(chosen))
    tally.add("fen_hash_mismatches", (children.hash_boards(chosen["board"]) != chosen["fen_hash"]).sum())
    tally.add("grouped_leaks", grouped.selected(chosen["board"], salt).sum() if len(chosen) else 0)
    (_sample_roots if kind == "roots" else _sample_children)(chosen, tally)


class _Context(dict):
    """blocked, eval_hashes, salt, buckets, valprobe (sorted child hashes or None), found (bool array)."""


def _scan_records(recs: np.ndarray, kind: str, bucket: int, ctx: _Context, cfg: VerifyConfig, tally) -> None:
    hashes = recs["fen_hash"]
    tally.add(f"train_{kind}", len(recs))
    tally.add("blocklist_hits", contains(ctx["blocked"], hashes).sum())
    tally.add("eval_hits", contains(ctx["eval_hashes"], hashes).sum())
    tally.add("wrong_bucket", (hashes % np.uint64(ctx["buckets"]) != np.uint64(bucket)).sum())
    if ctx["valprobe"] is not None and len(ctx["valprobe"]) and len(hashes):
        at = np.minimum(np.searchsorted(ctx["valprobe"], hashes), len(ctx["valprobe"]) - 1)
        ctx["found"][at[ctx["valprobe"][at] == hashes]] = True
    _sample(recs, kind, bucket, cfg, ctx["salt"], tally)


def _scan_bucket(cfg: VerifyConfig, shards: dict, bucket: int, ctx: _Context, tally: _Tally) -> None:
    root_name, child_name = f"train_r{bucket:03d}.bin", f"train_c{bucket:03d}.bin"
    roots = _read_checked(cfg.pack_dir / root_name, shards[root_name], ROOT_DTYPE, tally)
    kids = _read_checked(cfg.pack_dir / child_name, shards[child_name], CHILD_DTYPE, tally)
    _scan_records(roots, "roots", bucket, ctx, cfg, tally)
    _scan_records(kids, "children", bucket, ctx, cfg, tally)
    tally.add("children_equal_roots", np.isin(kids["fen_hash"], roots["fen_hash"]).sum())
    tally.add("duplicate_children", len(kids) - len(np.unique(kids["fen_hash"])))
    multi, mono = monotone_counts(roots)
    tally.add("multi_pv", multi)
    tally.add("monotone", mono)
    wins = win_probability_array(roots["cp"], roots["mate"])
    tally.root_sizes.append(len(roots))
    tally.child_sizes.append(len(kids))
    tally.mean_wins.append(float(wins.mean()) if len(wins) else None)
    tally.win_sum += float(wins.sum())


def _eval_hashes(cfg: VerifyConfig, shards: dict, tally: _Tally) -> tuple[np.ndarray, dict[str, int]]:
    hashes, roots_by_split = [], {}
    for name, entry in sorted(shards.items()):
        if entry["split"] == "train":
            continue
        recs = _read_checked(cfg.pack_dir / name, entry, bigpack.DTYPES[entry["kind"]], tally)
        hashes.append(recs["fen_hash"])
    for split_name in bigpack.SPLITS:
        roots_by_split[split_name] = sum(
            e["records"] for e in shards.values() if e["split"] == split_name and e["kind"] == "roots"
        )
    return np.unique(np.concatenate(hashes)) if hashes else np.empty(0, np.uint64), roots_by_split


def _valprobe_hashes(pack_dir: Path) -> np.ndarray | None:
    path = pack_dir / "valprobe.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return np.unique(children.hash_boards(data["child_board"]))


def _pct(part: int, whole: int) -> float | None:
    return 100.0 * part / whole if whole else None


def _checks(tally: _Tally, roots_by_split: dict[str, int]) -> dict:
    n = tally.n
    zero = {
        "train_blocklist_hits": n["blocklist_hits"],
        "train_eval_hits": n["eval_hits"],
        "train_children_equal_train_roots": n["children_equal_roots"],
        "train_duplicate_children": n["duplicate_children"],
        "wrong_bucket": n["wrong_bucket"],
        "sha256_mismatches": n["sha256_mismatches"],
        "fen_hash_mismatches": n["fen_hash_mismatches"],
        "grouped_leaks_in_sample": n["grouped_leaks"],
    }
    checks = {name: _check(value, "== 0", value == 0) for name, value in zero.items()}
    sampled = n["sampled_roots"] + n["sampled_children"]
    for name, part, whole in (
        ("best_move_legal_pct", n["legal_best"], n["sampled_roots"]),
        ("alt_move_legal_pct", n["legal_alts"], n["alts"]),
        ("position_valid_pct", n["valid"], sampled),
    ):
        pct = _pct(part, whole)
        checks[name] = _check(pct, "== 100", pct is None or pct == 100.0)
    mono = _pct(n["monotone"], n["multi_pv"])
    checks["pv_monotone_pct"] = _check(
        mono, f">= {MONOTONE_MIN_PCT}", mono is None or mono >= MONOTONE_MIN_PCT
    )
    global_win = tally.win_sum / n["train_roots"] if n["train_roots"] else 0.5
    checks.update(split_checks(roots_by_split))
    checks.update(balance_checks(tally.root_sizes, tally.child_sizes, tally.mean_wins, global_win))
    return checks


def verify(cfg: VerifyConfig) -> dict:
    """Scan cfg.pack_dir and write verify.json there. Returns the report (report["ok"] is the verdict)."""
    start = time.perf_counter()
    manifest = bigpack.read_manifest(cfg.pack_dir)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"{cfg.pack_dir} holds a pack with status {manifest.get('status')!r}, not 'complete'"
        )
    listed = (manifest.get("blocklist") or {}).get("path")
    blocklist_path = cfg.blocklist or (Path(listed) if listed else None)
    blocked, blocklist_entry = pack.load_blocklist(blocklist_path)
    tally = _Tally()
    eval_hashes, roots_by_split = _eval_hashes(cfg, manifest["shards"], tally)
    vp = _valprobe_hashes(cfg.pack_dir)
    ctx = _Context(
        blocked=blocked,
        eval_hashes=eval_hashes,
        salt=manifest["grouped"]["salt"],
        buckets=manifest["buckets"],
        valprobe=vp,
        found=np.zeros(len(vp) if vp is not None else 0, dtype=bool),
    )
    for bucket in range(manifest["buckets"]):
        _scan_bucket(cfg, manifest["shards"], bucket, ctx, tally)
    checks = _checks(tally, roots_by_split)
    report = {
        "pack": str(cfg.pack_dir),
        "manifest_sha256": hashlib.sha256((cfg.pack_dir / bigpack.MANIFEST).read_bytes()).hexdigest(),
        "blocklist": blocklist_entry,
        "legality_sample": cfg.legality_sample,
        "ok": all(c["ok"] for c in checks.values()),
        "checks": checks,
        "counts": tally.n,
        "roots_by_split": roots_by_split,
        "valprobe": None
        if vp is None
        else {"children": len(vp), "children_in_train": int(ctx["found"].sum())},
        "seconds": time.perf_counter() - start,
    }
    text = json.dumps(report, indent=1)
    pack.write_atomic(Path(cfg.pack_dir) / REPORT, text.encode("utf-8"))
    return json.loads(text)
