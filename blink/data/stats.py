"""data_stats.json: the pack's headline numbers in one small file (the source of results/data_stats.json)."""

import json
from pathlib import Path

from blink.data import bigpack, pack, verify

OUTPUT = "data_stats.json"


def _verdict(pack_dir: Path) -> dict | None:
    path = Path(pack_dir) / verify.REPORT
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {"ok": report["ok"], "failed": sorted(n for n, c in report["checks"].items() if not c["ok"])}


def summary(pack_dir: Path) -> dict:
    m = bigpack.read_manifest(pack_dir)
    pass1 = m["pass1"]
    parsed = m["parsed_roots"]
    return {
        "source": m["source"],
        "frames": m["frames"],
        "end": m["end"],
        "lines": m["lines"],
        "parsed_roots": parsed,
        "rejects": m["rejects"],
        "reject_share": (m["lines"] - parsed) / m["lines"] if m["lines"] else 0.0,
        "errors": m["errors"],
        "children_per_root": sum(pass1["children"].values()) / parsed if parsed else 0.0,
        "roots": m["splits"]["roots"],
        "children": m["splits"]["children"],
        "grouped_children_dropped": pass1["grouped_children_dropped"],
        "pass2_dropped": m["pass2"]["dropped"],
        "eval_dropped": m["eval_dropped"],
        "disk_bytes": sum(e["bytes"] for e in m["shards"].values()),
        "pass1_lines_per_s": pass1["timing"]["lines_per_s"],
        "pass1_flush_mb_per_s": pass1["timing"]["flush_mb_per_s"],
        "pass2_records_per_s": m["pass2"]["timing"]["records_per_s"],
        "grouped_salt": m["grouped"]["salt"],
        "rebalance": m.get("rebalance", {}).get("weights"),
        "verify": _verdict(pack_dir),
    }


def write(pack_dir: Path, out: Path | None = None) -> dict:
    got = summary(pack_dir)
    text = json.dumps(got, indent=1)
    pack.write_atomic(Path(out) if out else Path(pack_dir) / OUTPUT, text.encode("utf-8"))
    return json.loads(text)
