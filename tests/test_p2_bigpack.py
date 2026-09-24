"""bigpack: two HDD-friendly passes from the pzstd eval DB to the v1 pack, with every leakage filter."""

import hashlib
import json

import numpy as np
import pytest
from data_fakes import synthetic_lines
from test_p2_fakes import GROUPED_SALT, leak_world, write_source

from blink.data import bigpack, grouped, split
from blink.data.record import CHILD_DTYPE, ROOT_DTYPE

BUCKETS = 8


@pytest.fixture(scope="module")
def world():
    return leak_world()


@pytest.fixture(scope="module")
def source(tmp_path_factory, world):
    lines = synthetic_lines(500, seed=5) + world.lines
    return write_source(tmp_path_factory.mktemp("src") / "db.jsonl.zst", lines, frame_bytes=20_000)


@pytest.fixture(scope="module")
def blocklist(tmp_path_factory, world):
    path = tmp_path_factory.mktemp("block") / "blocklist.npy"
    np.save(path, np.unique(np.array(world.blocklist + [1, 2, 3], dtype=np.uint64)))
    return path


def config(source, out, blocklist=None, **kwargs) -> bigpack.BigPackConfig:
    base = {"salt": GROUPED_SALT, "workers": 1, "buckets": BUCKETS, "buffer_bytes": 4096, "seed": 7}
    return bigpack.BigPackConfig(source=source, out=out, blocklist=blocklist, **{**base, **kwargs})


@pytest.fixture(scope="module")
def packed(tmp_path_factory, source, blocklist):
    out = tmp_path_factory.mktemp("v1")
    return out, bigpack.bigpack(config(source, out, blocklist))


def read(out, name, dtype) -> np.ndarray:
    return np.fromfile(out / name, dtype=dtype)


def train(out) -> tuple[np.ndarray, np.ndarray]:
    roots = np.concatenate([read(out, f"train_r{b:03d}.bin", ROOT_DTYPE) for b in range(BUCKETS)])
    kids = np.concatenate([read(out, f"train_c{b:03d}.bin", CHILD_DTYPE) for b in range(BUCKETS)])
    return roots, kids


def train_hashes(out) -> set[int]:
    roots, kids = train(out)
    return set(roots["fen_hash"].tolist()) | set(kids["fen_hash"].tolist())


def test_children_of_val_and_test_roots_never_reach_train(packed, world):
    out, _ = packed
    everything = train_hashes(out)
    for name, eval_file in (("val", "val_children.bin"), ("iid", "test_iid_children.bin")):
        shared = world.expect[f"{name}_shared_child"]
        assert shared in set(read(out, eval_file, CHILD_DTYPE)["fen_hash"].tolist())
        assert shared not in everything  # a train root reaches it too, and loses it
    for name in ("val", "test_iid", "test_grouped"):
        kids = read(out, f"{name}_children.bin", CHILD_DTYPE)
        assert len(kids) and not (set(kids["fen_hash"].tolist()) & everything)


def test_a_train_root_equal_to_a_val_or_test_child_is_dropped(packed, world):
    out, manifest = packed
    roots, _ = train(out)
    for name in ("val", "iid"):
        assert world.expect[f"{name}_child_as_train_root"] not in set(roots["fen_hash"].tolist())
    assert manifest["pass2"]["dropped"]["roots_equal_to_eval"] >= 2


def test_a_child_that_is_also_a_db_root_is_dropped(packed, world):
    out, manifest = packed
    roots, kids = train(out)
    target = world.expect["child_is_root"]
    assert target in set(roots["fen_hash"].tolist())
    assert target not in set(kids["fen_hash"].tolist())
    assert manifest["pass2"]["dropped"]["children_that_are_roots"] >= 1


def test_a_transposed_child_is_packed_once_with_the_deepest_label(packed, world):
    _, kids = train(packed[0])
    hit = kids[kids["fen_hash"] == np.uint64(world.expect["dup_child"])]
    assert len(hit) == 1 and int(hit["depth"][0]) == 35 and int(hit["cp"][0]) == -55


def test_a_blocklisted_child_never_reaches_train(packed, world):
    out, manifest = packed
    assert world.expect["blocked_child"] not in train_hashes(out)
    assert manifest["pass2"]["dropped"]["children_blocklisted"] >= 1


def test_a_colour_mirrored_blocklisted_position_never_reaches_train(packed, world):
    everything = train_hashes(packed[0])
    assert world.expect["mirror_root"] not in everything
    assert world.expect["mirror_child"] not in everything


def test_a_child_in_a_held_out_group_never_reaches_train(packed, world):
    out, manifest = packed
    roots, kids = train(out)
    assert world.expect["grouped_child"] not in train_hashes(out)
    assert not grouped.selected(roots["board"], GROUPED_SALT).any()
    assert not grouped.selected(kids["board"], GROUPED_SALT).any()
    assert manifest["pass1"]["grouped_children_dropped"] >= 1
    held = read(out, "test_grouped_roots.bin", ROOT_DTYPE)
    assert world.expect["grouped_root"] in set(held["fen_hash"].tolist())


def test_every_record_lands_in_its_split_and_bucket(packed):
    out, manifest = packed
    for b in range(BUCKETS):
        for kind, dtype in (("r", ROOT_DTYPE), ("c", CHILD_DTYPE)):
            recs = read(out, f"train_{kind}{b:03d}.bin", dtype)
            assert (recs["fen_hash"] % np.uint64(BUCKETS) == np.uint64(b)).all()
    codes = {"val": split.VAL_CODE, "test_iid": split.TEST_IID_CODE}
    for name, code in codes.items():
        roots = read(out, f"{name}_roots.bin", ROOT_DTYPE)
        assert len(roots) and (split.split_codes(roots["fen_hash"]) == code).all()
    held = read(out, "test_grouped_roots.bin", ROOT_DTYPE)
    assert grouped.selected(held["board"], GROUPED_SALT).all()
    assert manifest["grouped"]["salt"] == GROUPED_SALT and manifest["buckets"] == BUCKETS


def test_manifest_counts_sum_to_records_written(packed):
    out, m = packed
    shards = m["shards"]
    assert m["status"] == "complete" and len(shards) == 2 * BUCKETS + 6
    for name, entry in shards.items():
        data = (out / name).read_bytes()
        itemsize = ROOT_DTYPE.itemsize if entry["kind"] == "roots" else CHILD_DTYPE.itemsize
        assert len(data) == entry["bytes"] == entry["records"] * itemsize
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert m["lines"] == m["parsed_roots"] + sum(m["rejects"].values()) + sum(m["errors"].values())
    assert m["rejects"] == {"chess960": 1, "bad_row": 1}
    assert m["parsed_roots"] == sum(m["pass1"]["roots"].values())
    written = {(e["split"], e["kind"]): 0 for e in shards.values()}
    for entry in shards.values():
        written[(entry["split"], entry["kind"])] += entry["records"]
    assert written == {(s, k): m["splits"][k][s] for s in bigpack.SPLITS for k in ("roots", "children")}
    dropped = m["pass2"]["dropped"]
    roots_dropped = dropped["roots_blocklisted"] + dropped["roots_equal_to_eval"]
    assert m["pass1"]["roots"]["train"] == m["splits"]["roots"]["train"] + roots_dropped
    kids_dropped = sum(v for k, v in dropped.items() if k.startswith("children_"))
    assert m["pass1"]["children"]["train"] == m["splits"]["children"]["train"] + kids_dropped
    assert sum(m["evaldb_hist"]["roots"]) == m["splits"]["roots"]["train"]
    assert sum(m["evaldb_hist"]["children"]) == m["splits"]["children"]["train"]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == m


def test_the_bucket_files_are_gone_once_the_pack_is_complete(packed):
    out, _ = packed
    assert not (out / bigpack.BUCKET_DIR).exists()
    assert not list(out.glob("*.tmp"))


def test_shard_paths_lists_the_files_of_one_split_and_kind(packed):
    out, _ = packed
    paths = bigpack.shard_paths(out, "train", "children")
    assert [p.name for p in paths] == [f"train_c{b:03d}.bin" for b in range(BUCKETS)]
    assert [p.name for p in bigpack.shard_paths(out, "val", "roots")] == ["val_roots.bin"]


def test_resume_after_a_crash_in_pass_2_writes_the_same_shards(
    tmp_path, source, blocklist, packed, monkeypatch
):
    real = bigpack.pack_bucket
    calls = {"n": 0}

    def crash_on_the_fourth(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt("power cut")
        return real(*args, **kwargs)

    monkeypatch.setattr(bigpack, "pack_bucket", crash_on_the_fourth)
    out = tmp_path / "v1"
    with pytest.raises(KeyboardInterrupt):
        bigpack.bigpack(config(source, out, blocklist))
    half = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert half["status"] == "pass2" and len(half["shards"]) == 6 + 2 * 3
    assert not (out / bigpack.BUCKET_DIR / "r000.bin").exists()  # recorded shards free their buckets
    assert (out / bigpack.BUCKET_DIR / "r003.bin").exists()
    with pytest.raises(FileExistsError, match="--resume"):
        bigpack.bigpack(config(source, out, blocklist))
    monkeypatch.setattr(bigpack, "pack_bucket", real)
    done = bigpack.bigpack(config(source, out, blocklist, resume=True))
    straight = packed[1]
    assert {k: v["sha256"] for k, v in done["shards"].items()} == {
        k: v["sha256"] for k, v in straight["shards"].items()
    }


def test_a_crash_in_pass_1_restarts_pass_1_from_zero(tmp_path, source, blocklist, packed, monkeypatch):
    real = bigpack.pass1_lines

    def crash(*args, **kwargs):
        raise KeyboardInterrupt("power cut")

    monkeypatch.setattr(bigpack, "pass1_lines", crash)
    out = tmp_path / "v1"
    with pytest.raises(KeyboardInterrupt):
        bigpack.bigpack(config(source, out, blocklist))
    assert not (out / "manifest.json").exists()
    monkeypatch.setattr(bigpack, "pass1_lines", real)
    done = bigpack.bigpack(config(source, out, blocklist, resume=True))
    assert done["splits"] == packed[1]["splits"]


def test_a_changed_source_stops_the_resume_and_writes_nothing(tmp_path, source, blocklist, monkeypatch):
    real = bigpack.pack_bucket
    monkeypatch.setattr(bigpack, "pack_bucket", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    moved = tmp_path / "db.jsonl.zst"
    moved.write_bytes(source.read_bytes())
    out = tmp_path / "v1"
    with pytest.raises(KeyboardInterrupt):
        bigpack.bigpack(config(moved, out, blocklist))
    monkeypatch.setattr(bigpack, "pack_bucket", real)
    before = (out / "manifest.json").read_bytes()
    with open(moved, "ab") as handle:
        handle.write(b"\0" * 12)  # the file grew: a new upload behind the same name
    with pytest.raises(bigpack.SourceChanged, match="bytes"):
        bigpack.bigpack(config(moved, out, blocklist, resume=True))
    moved.write_bytes(source.read_bytes())
    (tmp_path / "db.jsonl.zst.head.txt").write_text('ETag: "new-etag"\n', encoding="utf-8")
    with pytest.raises(bigpack.SourceChanged, match="etag"):
        bigpack.bigpack(config(moved, out, blocklist, resume=True))
    assert (out / "manifest.json").read_bytes() == before
    assert not list(out.glob("train_r*.bin"))


def test_the_source_identity_reads_the_etag_beside_the_file(tmp_path):
    raw = tmp_path / "db.jsonl.zst"
    raw.write_bytes(b"abc")
    assert bigpack.source_identity(raw) == {"path": str(raw), "bytes": 3, "etag": None}
    (tmp_path / "db.jsonl.zst.head.txt").write_text(
        'Content-Length: 3\nETag: "6aa21979-52475bec9"\n', encoding="utf-8"
    )
    assert bigpack.source_identity(raw)["etag"] == '"6aa21979-52475bec9"'


def test_a_complete_pack_is_kept_unless_overwrite_is_given(tmp_path, source):
    out = tmp_path / "v1"
    first = bigpack.bigpack(config(source, out, limit_frames=2))
    with pytest.raises(FileExistsError, match="--overwrite"):
        bigpack.bigpack(config(source, out, limit_frames=2))
    assert bigpack.bigpack(config(source, out, limit_frames=2, resume=True)) == first
    again = bigpack.bigpack(config(source, out, limit_frames=3, overwrite=True))
    assert again["frames"] == 3 and first["frames"] == 2


def test_a_directory_holding_another_pack_format_is_refused_even_with_resume(tmp_path, source):
    out = tmp_path / "skeleton"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"format": "blink-pack-v1"}), encoding="utf-8")
    (out / "train_000.bin").write_bytes(b"")
    for extra in ({}, {"resume": True}):
        with pytest.raises(FileExistsError, match="blink-pack-v1"):
            bigpack.bigpack(config(source, out, **extra))
    assert (out / "train_000.bin").exists()


def test_limit_frames_reads_only_that_many_frames(tmp_path, source, packed):
    got = bigpack.bigpack(config(source, tmp_path / "v1", limit_frames=1))
    whole = bigpack.read_manifest(packed[0])
    assert got["frames"] == 1 and got["end"] == "limit" and whole["end"] == "eof"
    assert 0 < got["lines"] < whole["lines"] / whole["frames"] * 2


def test_two_spawned_workers_pack_the_same_bytes_as_one(tmp_path, source, blocklist, packed):
    two = bigpack.bigpack(config(source, tmp_path / "v1", blocklist, workers=2))
    assert {k: v["sha256"] for k, v in two["shards"].items()} == {
        k: v["sha256"] for k, v in packed[1]["shards"].items()
    }


def test_bigpack_split_codes_extend_the_split_module_codes():
    for code, name in enumerate(split.SPLITS):
        assert bigpack.SPLITS[code] == name
    assert (bigpack.TRAIN, bigpack.VAL, bigpack.TEST_IID) == (
        split.TRAIN_CODE,
        split.VAL_CODE,
        split.TEST_IID_CODE,
    )
    assert bigpack.SPLITS[bigpack.TEST_GROUPED] == "test_grouped" == bigpack.SPLITS[-1]
