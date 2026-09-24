"""tools/pack_gate.py: the pass/fail gate between the full pack and any training on it (PF66)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("pack_gate", REPO / "tools" / "pack_gate.py")
pack_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack_gate)

LEAKS = pack_gate.ZERO_CHECKS


def _pack(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal pack that passes every check, and the driver's status file."""
    pack = tmp_path / "v1"
    pack.mkdir()
    for kind in ("r", "c"):
        for i in range(256):
            (pack / f"train_{kind}{i:03d}.bin").write_bytes(b"")
    for name in ("val_roots.bin", "valprobe.npz", "mateset.npz"):
        (pack / name).write_bytes(b"x")
    manifest = {
        "status": "complete",
        "blocklist": {"sha256": "b" * 64},
        "splits": {"roots": {"train": 10}, "children": {"train": 15}},
        "rebalance": {"weights": [1.0] * 48},
    }
    raw = json.dumps(manifest).encode("utf-8")
    (pack / "manifest.json").write_bytes(raw)
    checks = {name: {"value": 0, "ok": True} for name in LEAKS}
    checks["canonical_epd_parity_pct"] = {"value": 100.0, "ok": True}
    verify = {
        "ok": True,
        "checks": checks,
        "counts": {
            "epd_mismatches": 0,
            "sampled_children": 1_000_000,
            "train_roots": 10,
            "train_children": 15,
        },
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "blocklist": {"sha256": "b" * 64},
    }
    (pack / "verify.json").write_text(json.dumps(verify), encoding="utf-8")
    (pack / "data_stats.json").write_text(
        json.dumps({"verify": {"ok": True, "failed": []}}), encoding="utf-8"
    )
    status = tmp_path / "status.json"
    status.write_text(
        chr(0xFEFF) + json.dumps({"state": "done", "step": "stats", "exit_code": 0}), encoding="utf-8"
    )
    return pack, status


def test_a_complete_clean_pack_passes_the_gate(tmp_path):
    pack, status = _pack(tmp_path)
    assert pack_gate.problems(pack, status) == []
    assert pack_gate.main([str(pack), "--status", str(status)]) == 0


@pytest.mark.parametrize("leak", ["train_blocklist_hits", "train_eval_hits", "grouped_leaks_in_sample"])
def test_any_leak_into_train_fails_the_gate(tmp_path, leak):
    pack, status = _pack(tmp_path)
    verify = json.loads((pack / "verify.json").read_text(encoding="utf-8"))
    verify["checks"][leak] = {"value": 3, "ok": False}
    (pack / "verify.json").write_text(json.dumps(verify), encoding="utf-8")
    assert any(leak in p for p in pack_gate.problems(pack, status))
    assert pack_gate.main([str(pack), "--status", str(status)]) == 1


def test_parity_below_100_percent_or_too_few_samples_fails(tmp_path):
    pack, status = _pack(tmp_path)
    verify = json.loads((pack / "verify.json").read_text(encoding="utf-8"))
    verify["checks"]["canonical_epd_parity_pct"] = {"value": 99.99, "ok": False}
    verify["counts"]["sampled_children"] = 6182
    (pack / "verify.json").write_text(json.dumps(verify), encoding="utf-8")
    found = pack_gate.problems(pack, status)
    assert any("canonical_epd_parity_pct" in p for p in found) and any("sampled_children" in p for p in found)


def test_a_manifest_rewritten_after_verify_fails(tmp_path):
    """The WORLD id is the manifest's sha1: a rewrite after verify would put runs in another world."""
    pack, status = _pack(tmp_path)
    (pack / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    assert any("manifest" in p for p in pack_gate.problems(pack, status))


def test_an_unfinished_or_failed_driver_fails(tmp_path):
    pack, status = _pack(tmp_path)
    status.write_text(json.dumps({"state": "running", "step": "verify", "exit_code": 0}), encoding="utf-8")
    assert any("driver" in p for p in pack_gate.problems(pack, status))
    (pack / "buckets").mkdir()
    assert any("buckets" in p for p in pack_gate.problems(pack, status))
