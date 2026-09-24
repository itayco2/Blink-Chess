"""`blink data ladder10m`: a pack of exactly the ladder's fixed roots and their children, for s10m."""

import hashlib
import json
import random

import chess
import numpy as np
import pytest
from data_fakes import synthetic_lines
from test_p2_fakes import GROUPED_SALT, board_line, walk, write_source

from blink import cli
from blink.data import bigpack, children, fixedset, ladder, rebalance, valprobe
from blink.data.record import CHILD_DTYPE, ROOT_DTYPE

BUCKETS = 4
WIDE_PV_ROOTS = 6  # roots with 7 PVs: their 6th and 7th children cannot be rebuilt from the root record


def _wide_lines(seed: int) -> list[bytes]:
    """Eval DB lines with 7 PVs, from positions of a walk no synthetic_lines game reaches."""
    rng, lines = random.Random(seed), []
    for board in walk(seed):
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        if len(legal) >= 7 and board.ply() >= 20 and rng.random() < 0.2:
            pvs = [(move.uci(), {"cp": rng.randint(-300, 300)}) for move in legal[:7]]
            lines.append(board_line(board, pvs))
        if len(lines) == WIDE_PV_ROOTS:
            return lines
    raise AssertionError("walk ended early")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A complete v1 pack with a rebalance table and a valprobe, as the ladder command needs."""
    lines = synthetic_lines(1600, seed=41) + _wide_lines(seed=9001)
    raw = write_source(tmp_path_factory.mktemp("src") / "db.jsonl.zst", lines, frame_bytes=40_000)
    out = tmp_path_factory.mktemp("v1")
    cfg = bigpack.BigPackConfig(source=raw, out=out, salt=GROUPED_SALT, buckets=BUCKETS, buffer_bytes=4096)
    bigpack.bigpack(cfg)
    games = rebalance.GamesHistogram(
        counts=np.arange(1, rebalance.NUM_BUCKETS + 1, dtype=np.int64),
        games_seen=10,
        games_with_eval=9,
        heldout_skipped=1,
        positions=1200,
        frames=1,
        end="eof",
    )
    bigpack.write_rebalance(out, games)
    valprobe.run(out, 20)
    return out


def _roots(pack, names) -> np.ndarray:
    return np.concatenate([np.fromfile(pack / name, dtype=ROOT_DTYPE) for name in names])


def _kids(pack, prefix: str = "train_c") -> np.ndarray:
    return np.concatenate([np.fromfile(p, dtype=CHILD_DTYPE) for p in sorted(pack.glob(f"{prefix}*.bin"))])


@pytest.fixture(scope="module")
def positions(source) -> int:
    """All of the first root shard and a few records of the second: the set ends mid-shard."""
    first = np.fromfile(source / "train_r000.bin", dtype=ROOT_DTYPE)
    return len(first) + 7


@pytest.fixture(scope="module")
def ladder_pack(source, positions, tmp_path_factory):
    out = tmp_path_factory.mktemp("ladder") / "ladder10m"
    manifest = ladder.build(
        ladder.LadderConfig(pack=source, out=out, positions=positions), log=lambda _: None
    )
    return out, manifest


def test_the_ladder_roots_are_the_fixed_set_in_the_same_order_with_the_same_sha(
    source, positions, ladder_pack
):
    out, manifest = ladder_pack
    expected, description = fixedset.fixed_train_set(source, positions)
    got, got_description = fixedset.fixed_train_set(out, positions)
    assert got.tobytes() == expected.tobytes()
    assert got_description["fen_hash_sha256"] == description["fen_hash_sha256"]
    assert manifest["fixed_set"] == description
    assert sorted(p.name for p in out.glob("train_r*.bin")) == ["train_r000.bin", "train_r001.bin"]
    assert sum(len(np.fromfile(p, dtype=ROOT_DTYPE)) for p in out.glob("train_r*.bin")) == positions


def test_the_baselines_record_the_set_with_the_same_rule():
    pytest.importorskip("torch")
    from blink.baselines import train as baseline_train

    assert baseline_train.fixed_train_set is fixedset.fixed_train_set
    assert baseline_train.root_shards is fixedset.root_shards


def test_the_ladder_children_are_the_source_children_of_exactly_those_roots(source, positions, ladder_pack):
    out, manifest = ladder_pack
    roots, _ = fixedset.fixed_train_set(source, positions)
    implied = set(children.children_of(roots).records["fen_hash"].tolist())
    source_kids = _kids(source)
    by_hash = {int(h): source_kids[i].tobytes() for i, h in enumerate(source_kids["fen_hash"])}
    expected = implied & set(by_hash)
    got = _kids(out)
    assert len(got) == len(expected) > 0
    assert set(got["fen_hash"].tolist()) == expected
    assert all(record.tobytes() == by_hash[int(record["fen_hash"])] for record in got)
    assert manifest["children"]["found"] == len(expected)
    assert manifest["children"]["not_in_pack"] == len(implied) - len(expected)


def test_each_ladder_child_shard_holds_the_children_of_its_root_shard(ladder_pack):
    out, _ = ladder_pack
    earlier: set[int] = set()
    for root_shard in sorted(out.glob("train_r*.bin")):
        kid_shard = out / root_shard.name.replace("train_r", "train_c")
        implied = set(
            children.children_of(np.fromfile(root_shard, dtype=ROOT_DTYPE)).records["fen_hash"].tolist()
        )
        kids = (
            set(np.fromfile(kid_shard, dtype=CHILD_DTYPE)["fen_hash"].tolist())
            if kid_shard.exists()
            else set()
        )
        assert kids <= implied and not kids & earlier  # a child two roots share sits with the first
        earlier |= implied


def test_bigpack_routes_a_child_by_its_own_hash_not_by_its_roots_shard(source):
    """Why the ladder scans every child shard: the children of train_r000's roots are spread over all."""
    roots = np.fromfile(source / "train_r000.bin", dtype=ROOT_DTYPE)
    implied = set(children.children_of(roots).records["fen_hash"].tolist())
    holders = {
        shard.name
        for shard in source.glob("train_c*.bin")
        if implied & set(np.fromfile(shard, dtype=CHILD_DTYPE)["fen_hash"].tolist())
    }
    assert len(holders) > 1
    assert all(h % BUCKETS == int(n[-7:-4]) for n in holders for h in implied & _hashes(source / n))


def _hashes(path) -> set[int]:
    return set(np.fromfile(path, dtype=CHILD_DTYPE)["fen_hash"].tolist())


def test_roots_with_more_than_five_pvs_are_counted_since_their_extra_children_are_left_out(source, tmp_path):
    total = sum(len(np.fromfile(p, dtype=ROOT_DTYPE)) for p in source.glob("train_r*.bin"))
    manifest = ladder.build(
        ladder.LadderConfig(pack=source, out=tmp_path / "all", positions=total), log=lambda _: None
    )
    assert manifest["children"]["roots_with_pvs_beyond_the_fifth"] >= 1
    assert manifest["children"]["found"] <= len(_kids(source))


def test_the_manifest_keeps_the_rebalance_table_and_what_the_world_id_reads(source, ladder_pack):
    out, manifest = ladder_pack
    original = bigpack.read_manifest(source)
    assert manifest["rebalance"] == original["rebalance"]
    for key in ("blocklist", "split_rule", "grouped", "record_bytes"):
        assert manifest[key] == original[key]
    assert manifest["status"] == "complete" and manifest["format"] == ladder.FORMAT
    assert (
        manifest["source"]["manifest_sha256"]
        == hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    )
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_every_file_is_listed_with_its_records_and_sha256(ladder_pack):
    out, manifest = ladder_pack
    listed = set(manifest["shards"]) | set(manifest["eval_files"])
    on_disk = {p.name for p in out.iterdir() if p.name != "manifest.json"}
    assert listed == on_disk
    for name, entry in {**manifest["shards"], **manifest["eval_files"]}.items():
        data = (out / name).read_bytes()
        assert entry["bytes"] == len(data) and entry["sha256"] == hashlib.sha256(data).hexdigest()
    for entry in manifest["shards"].values():
        size = CHILD_DTYPE.itemsize if entry["kind"] == "children" else ROOT_DTYPE.itemsize
        assert entry["records"] * size == entry["bytes"] and entry["split"] == "train"


def test_the_val_and_probe_files_the_trainer_reads_are_copied(source, ladder_pack):
    out, _ = ladder_pack
    for name in ("val_roots.bin", "valprobe.npz"):
        assert (out / name).read_bytes() == (source / name).read_bytes()
    assert bigpack.shard_paths(out, "train", "roots") == sorted(out.glob("train_r*.bin"))
    assert bigpack.shard_paths(out, "train", "children") == sorted(out.glob("train_c*.bin"))


@pytest.mark.torch
def test_the_trainer_draws_weighted_roots_and_children_from_the_ladder_pack(source, ladder_pack):
    pytest.importorskip("torch")
    from argparse import Namespace

    from blink.commands import train_data
    from blink.model.config import TrainConfig

    out, manifest = ladder_pack
    cfg = TrainConfig(batch_size=20, child_frac=0.25, val_size=8)
    plan = train_data.plan(Namespace(source_raw=None, data=str(out), valprobe=None), cfg)
    step = next(plan.source(0))
    assert (len(step.roots), len(step.children)) == (15, 5)
    assert step.root_weight is not None and step.child_weight is not None
    assert plan.description["child_shards"] == sum(
        e["kind"] == "children" for e in manifest["shards"].values()
    )
    val = np.fromfile(out / "val_roots.bin", dtype=ROOT_DTYPE)
    assert plan.probe is not None and plan.val.tobytes() == val[:8].tobytes()
    source_plan = train_data.plan(Namespace(source_raw=None, data=str(source), valprobe=None), cfg)
    assert plan.world != source_plan.world  # its own data world: a run cannot resume across the two


def _pack_copy(source, tmp_path, edit=None):
    copy = tmp_path / "copy"
    copy.mkdir()
    for path in source.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    if edit is not None:
        manifest = bigpack.read_manifest(copy)
        edit(manifest)
        bigpack.write_manifest(copy, manifest)
    return copy


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m.update(status="pass2"), "not a complete"),
        (lambda m: m.pop("rebalance"), "blink data rebalance"),
        (lambda m: m.update(format="blink-pack-v0"), "not a v1 pack"),
    ],
)
def test_an_unfinished_or_unbalanced_pack_is_refused(source, tmp_path, edit, message):
    copy = _pack_copy(source, tmp_path, edit)
    with pytest.raises(ValueError, match=message):
        ladder.build(ladder.LadderConfig(pack=copy, out=tmp_path / "out", positions=10), log=lambda _: None)
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_more_positions_than_the_pack_holds_is_refused(source, tmp_path):
    with pytest.raises(ValueError, match="fewer than the 1,000,000"):
        ladder.build(
            ladder.LadderConfig(pack=source, out=tmp_path / "o", positions=1_000_000), log=lambda _: None
        )


def test_an_existing_output_is_kept_unless_overwrite_is_given(source, tmp_path):
    cfg = ladder.LadderConfig(pack=source, out=tmp_path / "o", positions=10)
    first = ladder.build(cfg, log=lambda _: None)
    with pytest.raises(FileExistsError, match="--overwrite"):
        ladder.build(cfg, log=lambda _: None)
    again = ladder.build(ladder.LadderConfig(pack=source, out=tmp_path / "o", positions=10, overwrite=True))
    assert again["fixed_set"] == first["fixed_set"]
    with pytest.raises(ValueError, match="the source pack itself"):
        ladder.build(ladder.LadderConfig(pack=source, out=source, positions=10, overwrite=True))


def test_the_command_writes_the_pack_and_prints_the_set(source, positions, tmp_path, capsys):
    out = tmp_path / "ladder10m"
    argv = ["data", "ladder10m", "--pack", str(source), "--positions", str(positions), "--out", str(out)]
    assert cli.main(argv) == 0
    printed = capsys.readouterr().out
    manifest = bigpack.read_manifest(out)
    assert manifest["fixed_set"]["fen_hash_sha256"][:16] in printed and str(out) in printed
    assert cli.main(argv) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert (
        cli.main(["data", "ladder10m", "--pack", str(tmp_path / "none"), "--out", str(tmp_path / "x")]) == 2
    )
    assert "manifest.json" in capsys.readouterr().err
