"""Packing: frames to parsed roots, blocklist filter, split routing, permuted shards and a manifest."""

import hashlib
import json

import numpy as np
import pytest
from data_fakes import BAD_LINES, fixture_lines, lines_text, synthetic_lines, write_pzstd

from blink.data import pack, parse, split
from blink.data.record import ROOT_DTYPE

SHARDS = 4


@pytest.fixture(scope="module")
def good_lines() -> list[bytes]:
    return fixture_lines() + synthetic_lines(1400, seed=11)


@pytest.fixture(scope="module")
def source(tmp_path_factory, good_lines):
    lines = list(good_lines)
    for i, bad in enumerate(BAD_LINES):
        lines.insert(50 + 300 * i, bad)
    path = tmp_path_factory.mktemp("src") / "db.jsonl.zst"
    write_pzstd(path, lines_text(lines), frame_bytes=40_000)
    return path


@pytest.fixture(scope="module")
def good_hashes(good_lines) -> np.ndarray:
    return np.array([int(parse.parse_line(line)["fen_hash"]) for line in good_lines], dtype=np.uint64)


def _read_all(out) -> np.ndarray:
    return np.concatenate([np.fromfile(p, dtype=ROOT_DTYPE) for p in sorted(out.glob("*.bin"))])


def _pack(source, out, **kwargs) -> dict:
    cfg = pack.PackConfig(source=source, out=out, shards=SHARDS, **{"workers": 1, "seed": 7, **kwargs})
    return pack.pack(cfg)


def test_no_blocklisted_hash_survives_packing(tmp_path, source, good_hashes):
    blocked = np.concatenate([good_hashes[::5], np.array([1, 2, 3], dtype=np.uint64)])
    blocklist = tmp_path / "blocklist.npy"
    np.save(blocklist, blocked)
    manifest = _pack(source, tmp_path / "out", blocklist=blocklist)
    written = _read_all(tmp_path / "out")
    assert not np.isin(written["fen_hash"], blocked).any()
    assert manifest["dropped_blocklisted"] == int(np.isin(good_hashes, blocked).sum()) > 0
    assert manifest["blocklist"]["entries"] == len(np.unique(blocked))
    assert len(written) == len(good_hashes) - manifest["dropped_blocklisted"]


def test_manifest_counts_sum_to_records_written(tmp_path, source, good_lines):
    out = tmp_path / "out"
    manifest = _pack(source, out)
    shards = manifest["shards"]
    assert sum(entry["records"] for entry in shards.values()) == manifest["records_written"]
    assert sum(manifest["splits"].values()) == manifest["records_written"] == len(good_lines)
    assert manifest["lines"] == manifest["parsed"] + sum(manifest["rejects"].values())
    assert manifest["parsed"] == manifest["records_written"] + manifest["dropped_blocklisted"]
    assert manifest["rejects"] == {"bad_row": 2, "no_evals": 1, "illegal_best_move": 1}
    assert manifest["errors"] == {}
    for name, entry in shards.items():
        data = (out / name).read_bytes()
        assert len(data) == entry["bytes"] == entry["records"] * ROOT_DTYPE.itemsize
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["split_rule"] == split.SPLIT_RULE and manifest["seed"] == 7


def test_every_record_lands_in_its_split_and_train_shard(tmp_path, source):
    out = tmp_path / "out"
    manifest = _pack(source, out)
    assert set(manifest["shards"]) == {f"train_{i:03d}.bin" for i in range(SHARDS)} | {
        "val.bin",
        "test_iid.bin",
    }
    for name, entry in manifest["shards"].items():
        recs = np.fromfile(out / name, dtype=ROOT_DTYPE)
        assert {split.split_of(int(h)) for h in recs["fen_hash"]} <= {entry["split"]}
        if entry["split"] == "train":
            assert set((recs["fen_hash"] % SHARDS).tolist()) <= {int(name[6:9])}
    assert manifest["splits"]["val"] + manifest["splits"]["test_iid"] > 0


def test_packing_is_deterministic_and_independent_of_worker_count(tmp_path, source):
    one = _pack(source, tmp_path / "one", workers=1)
    two = _pack(source, tmp_path / "two", workers=2)
    assert {k: v["sha256"] for k, v in one["shards"].items()} == {
        k: v["sha256"] for k, v in two["shards"].items()
    }


def test_each_shard_is_permuted_by_the_seed(tmp_path, source, good_hashes):
    _pack(source, tmp_path / "a")
    _pack(source, tmp_path / "b", seed=8)
    a = np.fromfile(tmp_path / "a" / "train_000.bin", dtype=ROOT_DTYPE)
    b = np.fromfile(tmp_path / "b" / "train_000.bin", dtype=ROOT_DTYPE)
    in_file_order = good_hashes[(good_hashes % SHARDS == 0) & (split.split_codes(good_hashes) == 0)]
    assert sorted(a["fen_hash"].tolist()) == sorted(b["fen_hash"].tolist()) == sorted(in_file_order.tolist())
    assert a["fen_hash"].tolist() != in_file_order.tolist()
    assert a["fen_hash"].tolist() != b["fen_hash"].tolist()


def test_a_frame_limit_packs_only_that_many_frames(tmp_path, source):
    manifest = _pack(source, tmp_path / "out", frames=2)
    assert manifest["frames"] == 2 and manifest["end"] == "limit"
    assert 0 < manifest["lines"] < 1000


def test_packing_into_an_existing_pack_needs_overwrite_and_removes_stale_shards(tmp_path, source):
    out = tmp_path / "out"
    pack.pack(pack.PackConfig(source=source, out=out, shards=6, workers=1))
    with pytest.raises(FileExistsError, match="--overwrite"):
        _pack(source, out)
    manifest = _pack(source, out, overwrite=True)
    assert sorted(p.name for p in out.glob("*.bin")) == sorted(manifest["shards"])
    assert not list(out.glob("*.tmp"))


def test_a_blocklist_must_be_a_flat_uint64_array(tmp_path, source):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="uint64"):
        _pack(source, tmp_path / "out", blocklist=bad)


def test_split_paths_lists_the_shards_of_a_split_from_the_manifest(tmp_path, source):
    out = tmp_path / "out"
    _pack(source, out)
    assert pack.split_paths(out, "train") == [out / f"train_{i:03d}.bin" for i in range(SHARDS)]
    assert pack.split_paths(out, "val") == [out / "val.bin"]


def test_an_unexpected_parser_exception_is_counted_not_swallowed(monkeypatch):
    def explode(line):
        raise RuntimeError("boom")

    monkeypatch.setattr(pack.parse, "parse_line", explode)
    parsed = pack.parse_lines([b"{}", b"{}"])
    assert parsed.errors == {"RuntimeError": 2}
    assert len(parsed.records) == 0 and parsed.error_samples
