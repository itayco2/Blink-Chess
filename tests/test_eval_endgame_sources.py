"""Where the E8/E2b endgames come from: endgames.epd's provenance, and PR-4's fallback from the v1 pack."""

import hashlib
import json
import subprocess

import chess
import numpy as np
import pytest

from blink import cli
from blink.board import encode
from blink.data import games10k, grouped
from blink.data.children import codes_to_board
from blink.eval import endgame_looks, endgame_sources, endgames, sflabel

ROOK_WHITE = "8/8/4k3/8/8/8/3RK3/8 w - - 0 1"  # 1 major or minor, White to move
ROOK_BLACK_LOSES = "8/8/4k3/8/8/8/3RK3/8 b - - 0 1"  # the same material, Black to move: another group
QUEEN = "8/8/8/3k4/8/8/3QK3/8 w - - 0 1"
PAWN = "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"
SIX = "rnb1k3/8/8/8/8/8/8/R1BQK3 w - - 0 1"  # exactly 6 queens, rooks, bishops and knights
SEVEN = "rnb1k3/8/8/8/8/8/8/R1BQKB2 w - - 0 1"  # 7: not a Divider endgame
START = chess.STARTING_FEN
REAL_LABELER = sflabel.SfLabeler  # the fakes below wrap it; a test may install a fake twice


def root(fen, cp=None, mate=None):
    board = chess.Board(fen)
    return games10k.to_record(board, next(iter(board.legal_moves)), cp, mate, 20)


def normalised(rec) -> str:
    return codes_to_board(encode.unpack(rec["board"])).fen()


def salt_holding_out(first, others) -> int:
    """A salt under which `first`'s group is test_grouped's and none of `others`' groups is."""
    boards = np.stack([first["board"], *[o["board"] for o in others]])
    for salt in range(200_000):
        chosen = grouped.selected(boards, salt)
        if chosen[0] and not chosen[1:].any():
            return salt
    raise AssertionError("no salt found")


def write_pack(folder, val, test_grouped, salt):
    folder.mkdir(parents=True, exist_ok=True)
    np.array(val).tofile(folder / "val_roots.bin")
    np.array(test_grouped).tofile(folder / "test_grouped_roots.bin")
    (folder / "manifest.json").write_text(json.dumps({"grouped": {"salt": salt}}), encoding="utf-8")
    return folder


@pytest.fixture
def pack(tmp_path):
    """val: 8 roots, 3 fallback candidates (records 4, 6, 8); test_grouped: 3 roots, candidates 1 and 3."""
    held_out = root(ROOK_WHITE, cp=900)  # a decisive val endgame in a test_grouped group: left out of dev
    val = [
        root(START, cp=900),  # 1: 14 majors and minors
        held_out,  # 2
        root(PAWN, cp=120),  # 3: not decisive
        root(ROOK_BLACK_LOSES, cp=-600),  # 4: decisive for White (cp is the side to move's)
        root(QUEEN, mate=0),  # 5: checkmated-now encoding, not a forced mate
        root(QUEEN, mate=5),  # 6: a forced mate
        root(SEVEN, cp=900),  # 7: 7 majors and minors
        root(SIX, cp=700),  # 8: exactly 6
    ]
    salt = salt_holding_out(held_out, [val[3], val[5], val[7]])
    test_grouped = [root(ROOK_WHITE, cp=900), root(START, cp=-900), root(QUEEN, cp=-500)]
    return write_pack(tmp_path / "pack", val, test_grouped, salt), val, test_grouped, salt


def test_the_divider_count_matches_static_phase_on_the_packed_board():
    castled = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"  # castling rooks have their own codes
    for fen in (START, ROOK_WHITE, SIX, SEVEN, castled):
        board = chess.Board(fen)
        codes = encode.encode_board(board)[None]
        expected = chess.popcount(board.occupied & ~(board.kings | board.pawns))
        assert endgame_sources.majors_and_minors(codes).tolist() == [expected]


def test_the_fallback_reads_val_endgames_outside_test_groupeds_groups_then_test_grouped(pack):
    folder, val, test_grouped, salt = pack
    sources = endgame_sources.fallback_sources(folder)
    dev, final = sources["dev"], sources["final"]
    assert (dev.split, dev.salt, final.split, final.salt) == ("val", salt, "test_grouped", None)
    dev_roots, final_roots = dev.read(), final.read()
    assert list(dev.positions(dev_roots)) == [(i, normalised(val[i - 1])) for i in (4, 6, 8)]
    assert list(final.positions(final_roots)) == [(i, normalised(test_grouped[i - 1])) for i in (1, 3)]
    assert [line for line, _ in dev.positions(dev.read(limit=6))] == [4, 6]
    described = dev.describe(dev_roots)
    digest = hashlib.sha256((folder / "val_roots.bin").read_bytes()).hexdigest()
    assert (described["path"], described["sha256"]) == (str(folder / "val_roots.bin"), digest)
    assert (described["records"], described["candidates"], described["salt"]) == (8, 3, salt)
    assert described["kind"] == "fallback" and described["set"] == "dev"


def test_a_pack_file_that_no_longer_matches_its_manifest_is_refused(pack):
    folder, *_ = pack
    manifest = {"grouped": {"salt": 1}, "shards": {"val_roots.bin": {"sha256": "0" * 64}}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dev = endgame_sources.fallback_sources(folder)["dev"]
    with pytest.raises(ValueError, match="val_roots.bin"):
        dev.describe(dev.read())


def test_a_missing_pack_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        endgame_sources.fallback_sources(tmp_path)


def test_endgames_epd_is_described_by_its_sha256_lines_and_unique_positions(tmp_path):
    epd = tmp_path / "e.epd"
    epd.write_text(f"{ROOK_WHITE}\n{QUEEN}\n{ROOK_WHITE.replace(' 0 1', ' 3 9')}\n", encoding="utf-8")
    described = endgame_sources.describe_epd(epd)
    assert described["sha256"] == hashlib.sha256(epd.read_bytes()).hexdigest()
    assert (described["lines"], described["positions"], described["repeats"]) == (3, 2, 1)


def test_the_harness_commit_is_this_checkout_s_head():
    found = endgame_sources.harness_commit()
    assert len(found["commit"]) == 40 and isinstance(found["dirty"], bool)


def test_the_harness_commit_reads_git_without_taking_the_index_lock(monkeypatch):
    """A plain `git status` may write index.lock and collide with a merge or commit in the same checkout."""
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="c" * 40 + "\n", stderr="")

    monkeypatch.setattr(endgame_sources.subprocess, "run", run)
    assert endgame_sources.harness_commit()["commit"] == "c" * 40
    assert any("status" in argv for argv in commands)
    assert all(argv[:2] == ["git", "--no-optional-locks"] for argv in commands)


# ------------------------------------------------------------------------------ `blink eval endgames`


class Killed(Exception):
    pass


def fake_labelers(monkeypatch, tmp_path, pawns=9.0, calls=None, kill_after=None):
    """Every position +`pawns` for its side to move; cached per node budget under tmp_path/cache."""

    def analyse(board, nodes, move):
        if kill_after is not None and len(calls) >= kill_after:
            raise Killed  # taskkill /F, as far as the screen can tell
        if calls is not None:
            calls.append((nodes, board.fen()))
        return sflabel.SfLabel(int(100 * pawns), None, 20, None)

    def fake(nodes, exe=None, cache_path=None, analyse_=None, procs=1):
        return REAL_LABELER(nodes, cache_path=tmp_path / "cache" / f"c{nodes}.jsonl", analyse=analyse)

    monkeypatch.setattr(sflabel, "SfLabeler", fake)


def endgames_args(out, *extra):
    return ["eval", "endgames", "--out", str(out), "--screen-nodes", "10", "--confirm-nodes", "20", *extra]


def summary_of(out):
    return json.loads((out / "endgames.json").read_text(encoding="utf-8"))


def declare(out, tmp_path, monkeypatch):
    """endgames.epd screened into `out` and declared unable to supply 700 at its look at line 1 (labellers
    faked by the caller): the record the fallback must find there before it runs."""
    epd = tmp_path / "declared.epd"
    epd.write_text(f"{ROOK_WHITE}\n{QUEEN}\n", encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(endgame_looks, "FIRST_LOOKS", (1,))
        assert cli.main(endgames_args(out, "--epd", str(epd))) == 0
    declared = summary_of(out)
    assert declared["branch"] == "epd-declared"
    return declared


def test_the_fallback_runs_only_by_its_flag_and_records_both_sources(pack, tmp_path, monkeypatch, capsys):
    folder, val, test_grouped, _ = pack
    fake_labelers(monkeypatch, tmp_path)
    monkeypatch.setattr(endgames, "DEV_COUNT", 2)
    monkeypatch.setattr(endgames, "WANT", 4)
    out = tmp_path / "out"
    declare(out, tmp_path, monkeypatch)
    assert cli.main(endgames_args(out, "--source", "fallback", "--data", str(folder))) == 0
    printed = capsys.readouterr().out
    assert "Itay's OK" in printed
    assert [e.line for e in endgames.read_set(out, "dev")] == [4, 6]
    assert [e.line for e in endgames.read_set(out, "final")] == [1, 3]
    summary = summary_of(out)
    assert summary["branch"] == "fallback" and summary["complete"] and summary["declaration"] is None
    dev, final = summary["screens"]
    assert (dev["set"], dev["source"]["split"], final["set"], final["source"]["split"]) == (
        "dev",
        "val",
        "final",
        "test_grouped",
    )
    tg_digest = hashlib.sha256((folder / "test_grouped_roots.bin").read_bytes()).hexdigest()
    assert final["source"]["sha256"] == tg_digest and len(summary["harness"]["commit"]) == 40
    assert summary["harness"]["head_changed_during_run"] is False


def test_the_fallback_keeps_the_declaration_it_follows_in_endgames_json(pack, tmp_path, monkeypatch, capsys):
    """The fallback's sets replace endgames.epd's in the folder E2b and E8 read, but the looks, sha256,
    declaration and harness commit that justified the switch stay in its endgames.json, even on a re-run."""
    folder, *_ = pack
    fake_labelers(monkeypatch, tmp_path)
    out = tmp_path / "out"
    declared = declare(out, tmp_path, monkeypatch)
    args = endgames_args(out, "--source", "fallback", "--data", str(folder))
    assert cli.main(args) == 0
    summary = summary_of(out)
    assert summary["branch"] == "fallback" and summary["declared_by"] == declared
    (screen,) = summary["declared_by"]["screens"]
    epd_digest = hashlib.sha256((tmp_path / "declared.epd").read_bytes()).hexdigest()
    assert (screen["source"]["kind"], screen["source"]["sha256"]) == ("endgames.epd", epd_digest)
    assert [look["line"] for look in screen["looks"]] == [1]
    assert summary["declared_by"]["declaration"].startswith("line 1: 1 kept of 1 screened")
    assert len(summary["declared_by"]["harness"]["commit"]) == 40
    assert "follows endgames.epd's declaration: line 1: 1 kept" in capsys.readouterr().out
    assert cli.main(args) == 0  # a re-run carries endgames.epd's record forward, not the first fallback's
    assert summary_of(out)["declared_by"] == declared


@pytest.mark.parametrize(
    ("heads", "changed"),
    [(("a" * 40, "a" * 40), False), (("a" * 40, "b" * 40), True), ((None, None), None)],
    ids=["unchanged", "merged-mid-run", "no-checkout"],
)
def test_the_harness_commit_is_read_before_the_first_search(tmp_path, monkeypatch, heads, changed):
    """The screen runs for hours from a checkout that main is merged into: endgames.json records the commit
    read before the first search, and whether HEAD moved before the sets were written."""
    epd = tmp_path / "e.epd"
    epd.write_text(f"{ROOK_WHITE}\n{QUEEN}\n", encoding="utf-8")
    events = []
    fake_labelers(monkeypatch, tmp_path, calls=events)
    reads = iter(heads)

    def harness_commit(repo=None):
        events.append("harness")
        return {"commit": next(reads), "dirty": False}

    monkeypatch.setattr(endgame_sources, "harness_commit", harness_commit)
    out = tmp_path / "out"
    assert cli.main(endgames_args(out, "--epd", str(epd))) == 0
    assert events[0] == events[-1] == "harness" and events.count("harness") == 2 and len(events) > 2
    expected = {"commit": heads[0], "dirty": False, "head_changed_during_run": changed}
    assert summary_of(out)["harness"] == expected


def test_without_the_flag_the_screen_reads_endgames_epd_and_records_its_looks(tmp_path, monkeypatch):
    epd = tmp_path / "e.epd"
    fens = [ROOK_WHITE, QUEEN, SIX, PAWN, ROOK_BLACK_LOSES]
    epd.write_text("".join(f + "\n" for f in fens), encoding="utf-8")
    fake_labelers(monkeypatch, tmp_path)
    monkeypatch.setattr(endgame_looks, "FIRST_LOOKS", (2, 4))
    out = tmp_path / "out"
    assert cli.main(endgames_args(out, "--epd", str(epd))) == 0
    summary = summary_of(out)
    (screen,) = summary["screens"]
    assert screen["source"]["sha256"] == hashlib.sha256(epd.read_bytes()).hexdigest()
    assert (screen["source"]["kind"], screen["source"]["positions"]) == ("endgames.epd", 5)
    assert [look["line"] for look in screen["looks"]] == [2]  # the look at line 2 declares: 2 kept of 5
    assert summary["branch"] == "epd-declared" and summary["declaration"].startswith("line 2: 2 kept")
    assert summary["kept"] == 2 and not summary["complete"]


def test_a_limit_is_not_the_end_of_the_file(tmp_path, monkeypatch):
    epd = tmp_path / "e.epd"
    epd.write_text("".join(f + "\n" for f in (ROOK_WHITE, QUEEN, PAWN)), encoding="utf-8")
    fake_labelers(monkeypatch, tmp_path)
    out = tmp_path / "out"
    assert cli.main(endgames_args(out, "--epd", str(epd), "--limit", "2")) == 0
    assert summary_of(out)["branch"] == "epd" and summary_of(out)["screens"][0]["limit"] == 2
    assert cli.main(endgames_args(out, "--epd", str(epd))) == 0
    assert summary_of(out)["declaration"] == "the file ended at line 3 with 3 kept (fewer than 700)"


def test_endgames_epd_never_replaces_the_fallbacks_sets(pack, tmp_path, monkeypatch, capsys):
    folder, *_ = pack
    epd = tmp_path / "e.epd"
    epd.write_text(ROOK_WHITE + "\n", encoding="utf-8")
    fake_labelers(monkeypatch, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "endgames.json").write_text(json.dumps({"branch": "fallback"}), encoding="utf-8")
    assert cli.main(endgames_args(out, "--epd", str(epd))) == 2
    assert "fallback" in capsys.readouterr().err and summary_of(out) == {"branch": "fallback"}


@pytest.mark.parametrize(
    "recorded",
    [None, {"branch": "epd"}, {"branch": "epd-declared"}, {"branch": "fallback"}],
    ids=["nothing", "epd", "declared-without-its-record", "fallback-without-declared_by"],
)
def test_the_fallback_needs_endgames_epds_declaration_recorded_in_the_folder(
    pack, tmp_path, monkeypatch, capsys, recorded
):
    folder, *_ = pack
    fake_labelers(monkeypatch, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    if recorded is not None:
        (out / "endgames.json").write_text(json.dumps(recorded), encoding="utf-8")
    assert cli.main(endgames_args(out, "--source", "fallback", "--data", str(folder))) == 2
    assert "no PR-4 declaration" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == (["endgames.json"] if recorded else [])
    if recorded is not None:
        assert summary_of(out) == recorded


def test_a_killed_screen_restarts_from_its_cache_and_ends_with_the_same_sets(tmp_path, monkeypatch):
    """The GPU-gap script kills the screen before its measurements: the restart searches nothing the
    killed run cached and writes what an unbroken run writes."""
    epd = tmp_path / "e.epd"
    fens = [ROOK_WHITE, QUEEN, SIX, PAWN, ROOK_BLACK_LOSES, START, SEVEN]
    epd.write_text("".join(f + "\n" for f in fens), encoding="utf-8")
    whole = tmp_path / "whole"
    fake_labelers(monkeypatch, whole)
    assert cli.main(endgames_args(whole / "out", "--epd", str(epd))) == 0
    killed = tmp_path / "killed"
    first = []
    fake_labelers(monkeypatch, killed, calls=first, kill_after=5)
    with pytest.raises(Killed):
        cli.main(endgames_args(killed / "out", "--epd", str(epd)))
    assert len(first) == 5 and not (killed / "out" / "endgames.json").exists()
    again = []
    fake_labelers(monkeypatch, killed, calls=again)
    assert cli.main(endgames_args(killed / "out", "--epd", str(epd))) == 0
    assert not set(first) & set(again) and len(first) + len(again) == len(json_lines(whole / "cache"))
    assert summary_of(killed / "out") == summary_of(whole / "out")
    for name in ("dev", "final"):
        assert endgames.read_set(killed / "out", name) == endgames.read_set(whole / "out", name)


def json_lines(folder):
    return [line for path in sorted(folder.glob("*.jsonl")) for line in path.read_text().splitlines() if line]
