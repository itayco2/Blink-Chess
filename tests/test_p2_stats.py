"""stats and the manifest's rebalance block, on a small real bigpack."""

import json

import numpy as np
import pytest
from data_fakes import synthetic_lines
from test_p2_fakes import GROUPED_SALT, write_source

from blink.data import bigpack, rebalance, stats


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    src = write_source(tmp_path_factory.mktemp("src") / "db.jsonl.zst", synthetic_lines(400, seed=17), 25_000)
    out = tmp_path_factory.mktemp("v1")
    bigpack.bigpack(
        bigpack.BigPackConfig(source=src, out=out, salt=GROUPED_SALT, buckets=4, buffer_bytes=4096)
    )
    return out


def _games(counts: np.ndarray) -> rebalance.GamesHistogram:
    return rebalance.GamesHistogram(counts, 10, 5, 1, int(counts.sum()) + 3, 2, "truncated")


def test_the_rebalance_block_has_the_interface_keys_and_a_valid_table(tmp_path, pack_dir):
    pack = tmp_path / "copy"
    pack.mkdir()
    for path in pack_dir.iterdir():
        (pack / path.name).write_bytes(path.read_bytes())
    counts = np.arange(1, 49, dtype=np.int64)
    block = bigpack.write_rebalance(pack, _games(counts))
    manifest = bigpack.read_manifest(pack)
    assert manifest["rebalance"] == block
    assert (
        block["buckets"] == 48 and len(block["weights"]) == 48 and block["definition"] == rebalance.DEFINITION
    )
    evaldb = np.array(manifest["evaldb_hist"]["roots"], dtype=np.float64)
    weights = np.array(block["weights"])
    assert weights.min() >= 0.2 and weights.max() <= 5
    assert abs(float((evaldb / evaldb.sum() * weights).sum()) - 1) <= 0.02
    assert block["p_games"] == counts.tolist() and block["games"]["games_with_eval"] == 5


def test_stats_summarises_the_pack_in_one_file(pack_dir):
    got = stats.write(pack_dir)
    on_disk = json.loads((pack_dir / stats.OUTPUT).read_text(encoding="utf-8"))
    assert got == on_disk
    manifest = bigpack.read_manifest(pack_dir)
    assert got["lines"] == manifest["lines"] and got["roots"] == manifest["splits"]["roots"]
    assert got["children_per_root"] == pytest.approx(
        sum(manifest["pass1"]["children"].values()) / manifest["parsed_roots"]
    )
    assert got["disk_bytes"] == sum(e["bytes"] for e in manifest["shards"].values())
    assert got["pass1_lines_per_s"] > 0 and got["verify"] is None


def test_stats_includes_the_verify_verdict_when_present(tmp_path, pack_dir):
    pack = tmp_path / "copy"
    pack.mkdir()
    for path in pack_dir.iterdir():
        (pack / path.name).write_bytes(path.read_bytes())
    (pack / "verify.json").write_text(
        json.dumps({"ok": False, "checks": {"a": {"ok": False}}}), encoding="utf-8"
    )
    assert stats.write(pack)["verify"] == {"ok": False, "failed": ["a"]}
