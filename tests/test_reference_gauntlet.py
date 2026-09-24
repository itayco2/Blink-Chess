"""DM-9M in `blink gauntlet` (plan E7): DeepMind's engine under its own name, its own moves audited."""

import json
from pathlib import Path

import pytest

from blink import cli, paths
from blink.eval import fastchess, nosearch
from blink.play.factory import ModelUnavailable
from blink.reference import gauntlet as dm_gauntlet

REAL_WEIGHTS = paths.home() / "dm" / "9M-params.npz"
TOOLS_PRESENT = fastchess.fastchess_exe().is_file() and fastchess.stockfish_exe().is_file()
BOOK_PRESENT = fastchess.books.book_file().is_file()

# DM-9M plays 1. e4 (20 legal moves at the start) and 2. Nf3 (29 legal after 1. e4 e5). SF1320's node
# counts are far above L+1, so auditing the anchor by mistake would show up as violations.
FAKE_PGN = """[Event "gauntlet"]
[White "DM-9M"]
[Black "SF1320"]
[Result "*"]
[Termination "normal"]

1. e4 {+0.30/1 0.020s, n=20} e5 {+0.10/9 0.100s, n=5000} 2. Nf3 {+0.20/1 0.020s, n=29}
Nc6 {+0.10/9 0.100s, n=5000} *
"""
FAKE_LOG = "Games: 1, Wins: 0, Losses: 0, Draws: 1, Points: 0.5 (50.00 %)\n"


def prepare(tmp_path: Path, model: str = "dm:9M", tc: str | None = None) -> fastchess.Gauntlet:
    return dm_gauntlet.prepare_gauntlet(
        model=model,
        mode="policy",
        device="cpu",
        anchor=1320,
        games=2,
        book=str(tmp_path / "book.pgn"),
        out_dir=tmp_path,
        concurrency=1,
        tc=tc,
    )


@pytest.fixture
def fake_fastchess(monkeypatch) -> list[list[str]]:
    """fastchess replaced by a writer of one finished DM-9M game and its report line."""
    commands: list[list[str]] = []

    def run_fastchess(command, log: Path) -> int:
        commands.append(list(command))
        pgn = Path(command[command.index("-pgnout") + 1].split("=", 1)[1])
        pgn.write_text(FAKE_PGN, encoding="utf-8")
        log.write_text(FAKE_LOG, encoding="utf-8")
        return 0

    monkeypatch.setattr(fastchess, "run_fastchess", run_fastchess)
    return commands


def test_a_dm_gauntlet_runs_deepminds_engine_under_its_own_name(tmp_path):
    spec = prepare(tmp_path).blink
    assert spec.name == "DM-9M"
    assert spec.args == ("-m", "blink.uci", "--model=dm:9M", "--device=cpu")
    assert (spec.st, spec.timemargin_ms) == (fastchess.BLINK_ST, fastchess.BLINK_MARGIN_MS)


def test_the_ema_parameters_get_their_own_engine_name(tmp_path):
    spec = prepare(tmp_path, model="dm:9M:ema").blink
    assert spec.name == "DM-9M-ema"
    assert "--model=dm:9M:ema" in spec.args


def test_a_dm_gauntlet_pgn_is_named_after_deepminds_engine(tmp_path):
    pgn = prepare(tmp_path).plan.pgn_out
    assert pgn.parent == tmp_path.resolve()
    assert pgn.name.startswith("DM-9M_vs_SF1320_") and pgn.suffix == ".pgn"


def test_the_fastchess_command_never_calls_deepmind_blink(tmp_path):
    command = prepare(tmp_path).command()
    assert "name=DM-9M" in command
    assert not any(token.startswith("name=Blink") or "--mode=" in token for token in command)


def test_a_cutechess_time_control_applies_to_deepminds_engine_too(tmp_path):
    spec = prepare(tmp_path, tc="10+0.1").blink
    assert (spec.tc, spec.st) == ("10+0.1", None)


def test_a_malformed_dm_selector_is_a_clear_refusal_even_in_a_dry_run(tmp_path):
    with pytest.raises(ModelUnavailable, match=r"dm:9M\[:ema\]"):
        prepare(tmp_path, model="dm:270M")


def test_a_dm_gauntlet_audits_deepminds_moves_not_blinks(tmp_path, fake_fastchess):
    report = dm_gauntlet.execute(prepare(tmp_path))
    audit = report["audit"]
    assert (audit["decisions"], audit["players"], audit["violations"]) == (2, {"DM-9M": 2}, [])
    assert audit["compliant"] and audit["histogram"] == {20: 1, 29: 1}
    assert nosearch.audit([Path(report["pgn"])])["decisions"] == 0  # the default "blink" filter misses DM
    written = json.loads(Path(report["pgn"]).with_suffix(".nosearch.json").read_text(encoding="utf-8"))
    assert written["players"] == {"DM-9M": 2}


def test_blink_gauntlet_dry_run_with_a_dm_selector_prints_deepminds_engine(tmp_path, capsys):
    args = ["gauntlet", "--model", "dm:9M", "--device", "cpu", "--games", "2", "--dry-run"]
    assert cli.main([*args, "--book", str(tmp_path / "book.pgn"), "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "name=DM-9M" in printed and "--model=dm:9M" in printed
    assert "Blink-" not in printed and "--mode=" not in printed


def test_blink_gauntlet_with_a_dm_selector_reports_deepminds_moves(
    monkeypatch, tmp_path, fake_fastchess, capsys
):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    (tmp_path / "dm").mkdir()
    (tmp_path / "dm" / "9M-params.npz").write_bytes(b"")
    args = ["gauntlet", "--model", "dm:9M", "--device", "cpu", "--games", "2"]
    assert cli.main([*args, "--book", str(tmp_path / "book.pgn"), "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("DM-9M vs SF1320: games 1")
    assert "DM-9M forfeits {}" in printed and "no-search decisions 2, violations 0" in printed
    assert "Blink" not in printed


@pytest.mark.local
@pytest.mark.torch
@pytest.mark.skipif(
    not (TOOLS_PRESENT and BOOK_PRESENT and REAL_WEIGHTS.is_file()),
    reason="fastchess, Stockfish, the book or the converted 9M is missing",
)
def test_a_two_game_dm_gauntlet_against_stockfish_is_clean(tmp_path):
    report = dm_gauntlet.execute(
        dm_gauntlet.prepare_gauntlet(
            model="dm:9M",
            mode="policy",
            device="cpu",
            anchor=1320,
            games=2,
            book="dev",
            out_dir=tmp_path,
            concurrency=2,
            max_moves=15,
        )
    )
    audit = report["audit"]
    assert (report["returncode"], report["blink"], report["summary"]["games"]) == (0, "DM-9M", 2)
    assert audit["violations"] == [] and audit["decisions"] > 10
    assert set(audit["players"]) == {"DM-9M"}
    assert audit["value_mode_full_batches"] == 0  # one row per legal move: L rows, never L+1
    assert report["blink_forfeits"] == {}
