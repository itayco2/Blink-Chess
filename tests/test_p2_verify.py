"""verify: a full scan of the pack for leakage, legality, label sanity and balance."""

import hashlib
import json

import chess
import numpy as np
import pytest
from data_fakes import synthetic_lines
from test_p2_fakes import GROUPED_SALT, write_source

from blink.board import encode, moves
from blink.board.value import CP_NONE
from blink.data import bigpack, children, valprobe, verify
from blink.data.record import CHILD_DTYPE, NO_MOVE, ROOT_DTYPE

BUCKETS = 4


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    src = write_source(tmp_path_factory.mktemp("src") / "db.jsonl.zst", synthetic_lines(700, seed=13), 30_000)
    out = tmp_path_factory.mktemp("v1")
    cfg = bigpack.BigPackConfig(source=src, out=out, salt=GROUPED_SALT, buckets=BUCKETS, buffer_bytes=2048)
    bigpack.bigpack(cfg)
    return out


def copy_pack(src, dst):
    dst.mkdir()
    for path in src.iterdir():
        (dst / path.name).write_bytes(path.read_bytes())
    return dst


def run(pack, **kwargs) -> dict:
    return verify.verify(verify.VerifyConfig(pack_dir=pack, **{"legality_sample": 1.0, **kwargs}))


def test_a_clean_pack_has_no_leak_and_only_legal_labels(pack_dir):
    report = run(pack_dir)
    checks = report["checks"]
    for name in ("train_blocklist_hits", "train_eval_hits", "train_children_equal_train_roots"):
        assert checks[name]["value"] == 0 and checks[name]["ok"], name
    for name in ("train_duplicate_children", "wrong_bucket", "sha256_mismatches", "fen_hash_mismatches"):
        assert checks[name]["value"] == 0, name
    assert checks["best_move_legal_pct"]["value"] == 100.0
    assert checks["alt_move_legal_pct"]["value"] == 100.0
    assert checks["position_valid_pct"]["value"] == 100.0
    assert checks["canonical_epd_parity_pct"]["value"] == 100.0 and checks["canonical_epd_parity_pct"]["ok"]
    assert report["counts"]["sampled_roots"] == report["counts"]["train_roots"] > 0
    assert json.loads((pack_dir / "verify.json").read_text(encoding="utf-8")) == report


def _retag(pack, name: str) -> None:
    """Record a tampered shard's new size and sha in the manifest, so only the planted fault shows."""
    manifest = bigpack.read_manifest(pack)
    data = (pack / name).read_bytes()
    entry = manifest["shards"][name]
    itemsize = 68 if entry["kind"] == "roots" else 44
    entry.update(bytes=len(data), records=len(data) // itemsize, sha256=hashlib.sha256(data).hexdigest())
    bigpack.write_manifest(pack, manifest)


def test_a_val_root_smuggled_into_train_is_found(pack_dir, tmp_path):
    pack = copy_pack(pack_dir, tmp_path / "bad")
    val = np.fromfile(pack / "val_roots.bin", dtype=ROOT_DTYPE)
    victim = val[:1].copy()
    name = f"train_r{int(victim['fen_hash'][0] % np.uint64(BUCKETS)):03d}.bin"
    with open(pack / name, "ab") as handle:
        handle.write(victim.tobytes())
    _retag(pack, name)
    report = run(pack)
    assert report["checks"]["train_eval_hits"]["value"] == 1
    assert not report["checks"]["train_eval_hits"]["ok"] and not report["ok"]


def test_a_blocklisted_train_record_is_found(pack_dir, tmp_path):
    roots = np.fromfile(pack_dir / "train_r001.bin", dtype=ROOT_DTYPE)
    blocklist = tmp_path / "block.npy"
    np.save(blocklist, np.unique(roots["fen_hash"][:3]))
    report = run(pack_dir, blocklist=blocklist)
    assert report["checks"]["train_blocklist_hits"]["value"] == 3


def test_an_illegal_best_move_and_a_bad_hash_are_found(pack_dir, tmp_path):
    pack = copy_pack(pack_dir, tmp_path / "bad")
    roots = np.fromfile(pack / "train_r002.bin", dtype=ROOT_DTYPE)
    board = children.codes_to_board(encode.unpack(roots["board"][0]))
    legal = moves.legal_mask(board)
    roots["move"][0] = int(np.flatnonzero(~legal)[0])  # a vocabulary move that is illegal here
    roots["fen_hash"][1] ^= np.uint64(4 * 12345)  # stays in bucket 2, no longer the board's hash
    roots.tofile(pack / "train_r002.bin")
    _retag(pack, "train_r002.bin")
    checks = run(pack)["checks"]
    assert checks["best_move_legal_pct"]["value"] < 100.0
    assert checks["fen_hash_mismatches"]["value"] == 1


def _plant_dead_ep_square(records: np.ndarray) -> int:
    """Mark an en-passant square no pawn can use on the first record that has room. Returns its index."""
    codes = encode.unpack(records["board"])
    for index, row in enumerate(codes):
        for square in range(40, 48):  # rank 6 from the side to move: where an ep target sits
            if row[square] == encode.EMPTY and row[square - 8] != encode.OPP + chess.PAWN - 1:
                row[square] = encode.EP_SQUARE
                records["board"][index] = encode.pack(row)
                return index
    raise AssertionError("no record with room for a planted ep square")


def test_a_board_python_chess_would_write_differently_fails_the_round_trip(pack_dir, tmp_path):
    clean = run(pack_dir)
    assert clean["checks"]["board_round_trip_pct"]["value"] == 100.0
    assert clean["counts"]["round_trip_mismatches"] == 0
    pack = copy_pack(pack_dir, tmp_path / "bad")
    kids = np.fromfile(pack / "train_c003.bin", dtype=CHILD_DTYPE)
    _plant_dead_ep_square(kids)
    kids.tofile(pack / "train_c003.bin")
    _retag(pack, "train_c003.bin")
    report = run(pack)
    assert report["counts"]["round_trip_mismatches"] == 1
    assert report["checks"]["board_round_trip_pct"]["value"] < 100.0
    assert not report["checks"]["board_round_trip_pct"]["ok"]


def _plant_off_corner_castling_rook(records: np.ndarray) -> int:
    """Turn an own rook off a1 and h1 into a castling rook on the first record that has one."""
    codes = encode.unpack(records["board"])
    for index, row in enumerate(codes):
        own_rooks = np.flatnonzero(row == encode.OWN + chess.ROOK - 1)
        rooks = [s for s in own_rooks if s not in (chess.A1, chess.H1)]
        if rooks:
            row[rooks[0]] = encode.OWN_CASTLING_ROOK
            records["board"][index] = encode.pack(row)
            return index
    raise AssertionError("no record with an own rook off the corners")


@pytest.mark.parametrize("plant", [_plant_dead_ep_square, _plant_off_corner_castling_rook])
def test_a_packed_board_whose_canonical_epd_differs_from_python_chess_epd_fails_the_parity(
    pack_dir, tmp_path, plant
):
    pack = copy_pack(pack_dir, tmp_path / "bad")
    roots = np.fromfile(pack / "train_r002.bin", dtype=ROOT_DTYPE)
    plant(roots)
    roots.tofile(pack / "train_r002.bin")
    _retag(pack, "train_r002.bin")
    report = run(pack)
    assert report["counts"]["epd_mismatches"] == 1
    assert report["checks"]["canonical_epd_parity_pct"]["value"] < 100.0
    assert not report["checks"]["canonical_epd_parity_pct"]["ok"] and not report["ok"]


def test_a_changed_shard_fails_its_sha256(pack_dir, tmp_path):
    pack = copy_pack(pack_dir, tmp_path / "bad")
    data = bytearray((pack / "train_c000.bin").read_bytes())
    data[5] ^= 1
    (pack / "train_c000.bin").write_bytes(bytes(data))
    assert run(pack)["checks"]["sha256_mismatches"]["value"] == 1


def test_pv_monotonicity_counts_roots_whose_alternatives_never_beat_pv1():
    roots = np.zeros(4, dtype=ROOT_DTYPE)
    roots["alt_move"] = NO_MOVE
    roots["cp"] = [50, 50, 50, CP_NONE]
    roots["mate"] = [0, 0, 0, 3]
    roots["alt_move"][:3, :2] = 1
    roots["alt_cp"][0, :2] = [40, 40]  # monotone, with a tie
    roots["alt_cp"][1, :2] = [60, 10]  # PV 2 beats PV 1
    roots["alt_cp"][2, :2] = [20, 30]  # PV 3 beats PV 2
    assert verify.monotone_counts(roots) == (3, 1)


@pytest.mark.parametrize(
    ("fen", "valid", "impossible"),
    [
        (chess.STARTING_FEN, True, False),
        ("7K/PPPPPPPP/PPPPPPPP/8/8/8/8/7k w - - 0 1", True, True),  # 16 pawns: composed, still playable
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", True, False),
        ("4k3/8/8/8/8/8/4q3/4K3 b - - 0 1", False, False),  # White is in check with Black to move
    ],
)
def test_impossible_material_is_counted_apart_from_invalid_positions(fen, valid, impossible):
    assert verify.classify(chess.Board(fen)) == (valid, impossible)


def test_the_valprobe_children_found_in_train_are_counted(pack_dir, tmp_path):
    pack = copy_pack(pack_dir, tmp_path / "vp")
    roots = np.fromfile(pack / "train_r000.bin", dtype=ROOT_DTYPE)
    valprobe.save_npz(pack / "valprobe.npz", valprobe.probe_arrays(roots[roots["npv"] >= 2][:5]))
    info = run(pack)["valprobe"]
    assert 0 < info["children_in_train"] < info["children"]  # their PV children were packed as train


def test_split_and_balance_checks_use_the_plan_limits():
    splits = {"train": 998_000, "val": 2_000, "test_iid": 2_150, "test_grouped": 400}
    checks = verify.split_checks(splits)
    assert (
        checks["val_pct"]["ok"] and not checks["test_iid_pct"]["ok"] and not checks["test_grouped_pct"]["ok"]
    )
    sizes = verify.balance_checks([100, 101, 99], [50, 50, 52], [0.50, 0.503, 0.499], 0.5)
    assert sizes["root_shard_size_dev_pct"]["ok"] and not sizes["child_shard_size_dev_pct"]["ok"]
    assert sizes["shard_mean_win_dev_pt"]["value"] == pytest.approx(0.3)


def test_verify_refuses_an_unfinished_pack(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "pass2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="pass2"):
        run(tmp_path)
