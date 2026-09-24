"""The gate between the full pack and any training on it (PF66). Read-only; exit 0 = pass, 1 = fail.

    python tools/pack_gate.py D:/blink/data/v1 --status D:/blink/logs/pack-v1.status.json

It reads verify.json, data_stats.json and manifest.json and fails on any leak into train, parity
below 100%, a manifest changed after verify (the run WORLD id is the manifest's sha1), a missing
rebalance table, missing shards or files, leftover buckets/ (pass 2 unfinished), or a driver that
did not finish. It never rewrites anything, so it is safe to run while training reads the pack.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ZERO_CHECKS = (
    "train_blocklist_hits",
    "train_eval_hits",
    "grouped_leaks_in_sample",
    "sha256_mismatches",
    "wrong_bucket",
    "train_children_equal_train_roots",
    "train_duplicate_children",
)
MIN_SAMPLED_CHILDREN = 1_000_000
SHARDS = 256
REBALANCE_BINS = 48
REQUIRED_FILES = ("val_roots.bin", "valprobe.npz", "mateset.npz")


def _load(path: Path, found: list[str], encoding: str = "utf-8") -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding=encoding))
    except (OSError, ValueError) as exc:
        found.append(f"cannot read {path}: {exc}")
        return {}


def _driver(status: Path, found: list[str]) -> None:
    record = _load(status, found, "utf-8-sig")  # PowerShell writes it with a BOM
    if record and not (record.get("state") == "done" and record.get("exit_code") == 0):
        found.append(f"driver did not finish: {record}")


def _verify(verify: dict[str, Any], manifest_raw: bytes, manifest: dict[str, Any], found: list[str]) -> None:
    checks, counts = verify.get("checks", {}), verify.get("counts", {})
    for name in ZERO_CHECKS:
        check = checks.get(name) or {}
        if check.get("value") != 0 or not check.get("ok"):
            found.append(f"leak check {name} = {check or 'missing'}")
    parity = checks.get("canonical_epd_parity_pct") or {}
    if parity.get("value") != 100.0 or not parity.get("ok"):
        found.append(f"canonical_epd_parity_pct = {parity or 'missing'}")
    if counts.get("epd_mismatches") != 0:
        found.append(f"counts.epd_mismatches = {counts.get('epd_mismatches')}")
    if counts.get("sampled_children", 0) < MIN_SAMPLED_CHILDREN:
        found.append(f"sampled_children {counts.get('sampled_children')} < {MIN_SAMPLED_CHILDREN:,}")
    if verify.get("ok") is not True:
        failed = sorted(name for name, check in checks.items() if not check.get("ok"))
        found.append(f"verify.ok is not true: {failed}")
    if verify.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
        found.append("manifest.json changed after verify")
    if (verify.get("blocklist") or {}).get("sha256") != (manifest.get("blocklist") or {}).get("sha256"):
        found.append("verify used a different blocklist than the manifest names")
    splits = manifest.get("splits", {})
    expected = ((splits.get("roots") or {}).get("train"), (splits.get("children") or {}).get("train"))
    if (counts.get("train_roots"), counts.get("train_children")) != expected:
        found.append(
            f"verify scanned {counts.get('train_roots')}/{counts.get('train_children')} != {expected}"
        )


def _layout(pack: Path, manifest: dict[str, Any], stats: dict[str, Any], found: list[str]) -> None:
    if stats.get("verify") != {"ok": True, "failed": []} or stats.get("errors"):
        found.append(f"data_stats verify = {stats.get('verify')}, errors = {stats.get('errors')}")
    if manifest.get("status") != "complete":
        found.append(f"manifest status = {manifest.get('status')}")
    if len((manifest.get("rebalance") or {}).get("weights") or []) != REBALANCE_BINS:
        found.append("manifest has no 48-bin rebalance table")
    for kind in ("r", "c"):
        count = len(list(pack.glob(f"train_{kind}*.bin")))
        if count != SHARDS:
            found.append(f"{count} train_{kind}*.bin shards, expected {SHARDS}")
    found.extend(f"missing {name}" for name in REQUIRED_FILES if not (pack / name).is_file())
    if (pack / "buckets").exists():
        found.append("buckets/ still present: pass 2 did not finish")


def problems(pack: Path, status: Path | None = None) -> list[str]:
    """Every reason the pack must not be trained on; empty means it passes."""
    pack, found = Path(pack), []
    if status is not None:
        _driver(Path(status), found)
    try:
        manifest_raw = (pack / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
    except (OSError, ValueError) as exc:
        return [*found, f"cannot read {pack / 'manifest.json'}: {exc}"]
    _verify(_load(pack / "verify.json", found), manifest_raw, manifest, found)
    _layout(pack, manifest, _load(pack / "data_stats.json", found), found)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack", help="the pack directory, e.g. D:/blink/data/v1")
    parser.add_argument("--status", help="the driver's status JSON (state must be done)")
    args = parser.parse_args(argv)
    found = problems(Path(args.pack), Path(args.status) if args.status else None)
    print("GATE " + ("FAIL" if found else "PASS"), *(f"  {line}" for line in found), sep="\n", flush=True)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
