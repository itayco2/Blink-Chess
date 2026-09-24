"""The P2 data commands from the command line: bigpack, rebalance, valprobe, mateset, verify, stats."""

import json
import shutil

import numpy as np
import pytest
from data_fakes import synthetic_lines, write_pzstd
from test_p2_fakes import write_source

from blink import cli
from blink.data import bigpack

GAME = (
    '[Event "Rated Blitz game"]\n[Site "https://lichess.org/{site}"]\n\n'
    "1. e4 {{ [%eval 0.2] }} 1... e5 {{ [%eval 0.25] }} 2. Nf3 {{ [%eval 0.3] }} 1-0\n\n"
)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return write_source(
        tmp_path_factory.mktemp("src") / "db.jsonl.zst", synthetic_lines(3000, seed=23), 60_000
    )


@pytest.fixture(scope="module")
def blocklist(tmp_path_factory):
    path = tmp_path_factory.mktemp("block") / "blocklist.npy"
    np.save(path, np.array([1, 2, 3], dtype=np.uint64))
    return path


def bigpack_argv(source, out, blocklist, *extra) -> list[str]:
    return [
        "data",
        "bigpack",
        "--out",
        str(out),
        "--source",
        str(source),
        "--workers",
        "1",
        "--blocklist",
        str(blocklist),
        "--buckets",
        "4",
        *extra,
    ]


@pytest.fixture(scope="module")
def packed(tmp_path_factory, source, blocklist):
    out = tmp_path_factory.mktemp("v1")
    assert cli.main(bigpack_argv(source, out, blocklist, "--salt", "0")) == 0
    return out


def test_bigpack_prints_throughput_and_writes_a_complete_manifest(packed, capsys):
    manifest = bigpack.read_manifest(packed)
    assert manifest["status"] == "complete" and manifest["grouped"]["salt"] == 0
    assert manifest["buckets"] == 4 and len(manifest["shards"]) == 2 * 4 + 6


def test_valprobe_mateset_verify_and_stats_run_on_the_pack(tmp_path, packed, capsys):
    pack = tmp_path / "copy"
    pack.mkdir()
    for path in packed.iterdir():
        (pack / path.name).write_bytes(path.read_bytes())
    assert cli.main(["data", "valprobe", "--pack", str(pack), "--n", "50"]) == 0
    assert cli.main(["data", "mateset", "--pack", str(pack), "--n", "50"]) == 0
    assert (pack / "valprobe.npz").is_file() and (pack / "mateset.npz").is_file()
    code = cli.main(["data", "verify", "--pack", str(pack), "--legality-sample", "0.5"])
    report = json.loads((pack / "verify.json").read_text(encoding="utf-8"))
    assert code == (0 if report["ok"] else 1)
    assert report["checks"]["train_eval_hits"]["value"] == 0
    assert cli.main(["data", "stats", "--pack", str(pack)]) == 0
    out = capsys.readouterr().out
    assert "valprobe" in out and "verify" in out and "children per root" in out


def test_bigpack_picks_the_salt_on_a_probe_pack_when_none_is_given(tmp_path, source, blocklist):
    probe = tmp_path / "probe"
    assert (
        cli.main(
            ["data", "pack", "--shards", "2", "--out", str(probe), "--source", str(source), "--workers", "1"]
        )
        == 0
    )
    out = tmp_path / "v1"
    argv = bigpack_argv(source, out, blocklist, "--salt-probe", str(probe), "--giant-share", "0.01")
    assert cli.main(argv) == 0
    grouped = bigpack.read_manifest(out)["grouped"]
    assert grouped["probe"]["probe_roots"] > 0 and grouped["salt"] == grouped["probe"]["salt"]


def test_bigpack_without_a_salt_or_probe_is_refused(tmp_path, source, blocklist, capsys):
    argv = bigpack_argv(source, tmp_path / "v1", blocklist, "--salt-probe", str(tmp_path / "none"))
    assert cli.main(argv) == 2
    assert "--salt" in capsys.readouterr().err


def test_bigpack_refuses_a_missing_source_and_a_finished_pack(tmp_path, packed, source, blocklist, capsys):
    assert cli.main(bigpack_argv(tmp_path / "nope.zst", tmp_path / "v1", blocklist, "--salt", "0")) == 2
    assert cli.main(bigpack_argv(source, packed, blocklist, "--salt", "0")) == 2
    err = capsys.readouterr().err
    assert "nope.zst" in err and "--overwrite" in err


def test_a_changed_source_during_resume_exits_2_and_says_stop_and_ask(
    tmp_path, source, blocklist, capsys, monkeypatch
):
    moved = tmp_path / "db.jsonl.zst"
    moved.write_bytes(source.read_bytes())
    out = tmp_path / "v1"

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(bigpack, "pack_bucket", crash)
    with pytest.raises(KeyboardInterrupt):
        cli.main(bigpack_argv(moved, out, blocklist, "--salt", "0"))
    monkeypatch.undo()
    with open(moved, "ab") as handle:
        handle.write(b"x")
    assert cli.main(bigpack_argv(moved, out, blocklist, "--salt", "0", "--resume")) == 2
    assert "stop and ask" in capsys.readouterr().err


def test_a_resume_whose_buckets_were_deleted_exits_2_with_a_message(
    tmp_path, source, blocklist, capsys, monkeypatch
):
    out = tmp_path / "v1"

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(bigpack, "pack_bucket", crash)
    with pytest.raises(KeyboardInterrupt):
        cli.main(bigpack_argv(source, out, blocklist, "--salt", "0"))
    monkeypatch.undo()
    shutil.rmtree(out / bigpack.BUCKET_DIR)
    assert cli.main(bigpack_argv(source, out, blocklist, "--salt", "0", "--resume")) == 2
    assert "--overwrite" in capsys.readouterr().err


def test_rebalance_writes_the_manifest_block_and_keeps_the_games_histogram(tmp_path, packed, capsys):
    pack = tmp_path / "copy"
    pack.mkdir()
    for path in packed.iterdir():
        (pack / path.name).write_bytes(path.read_bytes())
    games = tmp_path / "games.pgn.zst"
    write_pzstd(games, "".join(GAME.format(site=f"g{i}") for i in range(3)).encode("utf-8"), 200)
    heldout = tmp_path / "heldout.pgn"
    heldout.write_text(GAME.format(site="g1"), encoding="utf-8")
    argv = ["data", "rebalance", "--pack", str(pack), "--games", str(games), "--heldout", str(heldout)]
    assert cli.main([*argv, "--workers", "1"]) == 0
    block = bigpack.read_manifest(pack)["rebalance"]
    assert block["buckets"] == 48 and block["games"]["heldout_skipped"] == 1
    saved = pack / "games_hist.json"
    assert saved.is_file()
    assert cli.main(["data", "rebalance", "--pack", str(pack), "--games-hist", str(saved)]) == 0
    assert bigpack.read_manifest(pack)["rebalance"] == block


def test_valprobe_on_a_directory_without_val_roots_exits_2(tmp_path, capsys):
    assert cli.main(["data", "valprobe", "--pack", str(tmp_path), "--n", "5"]) == 2
    assert "val_roots.bin" in capsys.readouterr().err
