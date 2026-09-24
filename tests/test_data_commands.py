"""`blink data probe` and `blink data pack` from the command line."""

import json

import pytest
from data_fakes import BAD_LINES, fixture_lines, lines_text, synthetic_lines, write_pzstd

from blink import cli
from blink.data import parse


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("cmd") / "db.jsonl.zst"
    write_pzstd(path, lines_text(fixture_lines() + synthetic_lines(300, seed=9) + BAD_LINES), 25_000)
    return path


def test_data_probe_writes_probe_json_and_prints_a_summary(tmp_path, source, capsys):
    out = tmp_path / "probe.json"
    code = cli.main(
        ["data", "probe", "--frames", "3", "--source", str(source), "--out", str(out), "--workers", "1"]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["frames"] == 3 and report["errors"] == {}
    printed = capsys.readouterr().out
    assert "lines/s" in printed and str(out) in printed


def test_data_pack_writes_shards_and_a_manifest(tmp_path, source, capsys):
    out = tmp_path / "skeleton"
    argv = ["data", "pack", "--shards", "3", "--out", str(out), "--source", str(source), "--workers", "1"]
    assert cli.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records_written"] == 400 and manifest["end"] == "eof"
    assert sorted(p.name for p in out.glob("*.bin")) == sorted(manifest["shards"])
    assert "records" in capsys.readouterr().out


def test_data_commands_refuse_a_missing_source(tmp_path, capsys):
    missing = tmp_path / "nope.jsonl.zst"
    assert cli.main(["data", "probe", "--source", str(missing), "--out", str(tmp_path / "p.json")]) == 2
    assert (
        cli.main(["data", "pack", "--source", str(missing), "--shards", "2", "--out", str(tmp_path / "o")])
        == 2
    )
    assert "nope.jsonl.zst" in capsys.readouterr().err


def test_data_pack_refuses_an_existing_pack_without_overwrite(tmp_path, source, capsys):
    out = tmp_path / "skeleton"
    argv = ["data", "pack", "--frames", "1", "--shards", "2", "--out", str(out), "--source", str(source)]
    assert cli.main([*argv, "--workers", "1"]) == 0
    assert cli.main([*argv, "--workers", "1"]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert cli.main([*argv, "--workers", "1", "--overwrite"]) == 0


def test_a_parser_error_makes_probe_and_pack_exit_non_zero(tmp_path, source, monkeypatch):
    def explode(line):
        raise IndexError("parser bug")

    monkeypatch.setattr(parse, "parse_line", explode)
    probe_argv = ["data", "probe", "--frames", "1", "--source", str(source), "--workers", "1"]
    assert cli.main([*probe_argv, "--out", str(tmp_path / "p.json")]) == 1
    pack_argv = ["data", "pack", "--frames", "1", "--shards", "2", "--source", str(source), "--workers", "1"]
    assert cli.main([*pack_argv, "--out", str(tmp_path / "o")]) == 1


def test_default_workers_leave_two_threads_free_and_cap_at_ten():
    from blink.commands import data

    assert data.default_workers(12) == 10
    assert data.default_workers(4) == 2
    assert data.default_workers(1) == 1
