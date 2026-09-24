"""PF60: a dm selector runs under fastchess as DM-9M[-ema], audited with --engine dm; node-limited SF."""

from pathlib import Path

import pytest

from blink.eval import fastchess, nosearch
from blink.play.factory import ModelUnavailable

DM_PGN = """[Event "gauntlet"]
[White "DM-9M"]
[Black "SF1320"]
[Result "*"]
[Termination "normal"]

1. e4 {+0.30/1 0.020s, n=20} e5 {+0.10/9 0.100s, n=5000} 2. Nf3 {+0.20/1 0.020s, n=29} *
"""


def test_a_dm_selector_runs_under_the_fastchess_name_dm_9m():
    """PF60: the dry run used to print name=Blink-policy-dm_9M."""
    spec = fastchess.blink_engine("dm:9M", mode="policy", device="cuda")
    assert spec.name == "DM-9M"
    assert spec.args == ("-m", "blink.uci", "--model=dm:9M", "--device=cuda")
    assert fastchess.blink_engine("dm:9M:ema", mode="value", device="cpu").name == "DM-9M-ema"


def test_a_malformed_dm_selector_is_one_clear_refusal():
    with pytest.raises(ModelUnavailable, match="dm:9M"):
        fastchess.engine_name("dm:270M", "policy")


def test_blink_names_are_unchanged():
    assert fastchess.engine_name("run:skeleton:ema", "value") == "Blink-value-run_skeleton_ema"


def test_the_audit_filter_is_dm_for_deepminds_engine_and_blink_otherwise():
    assert fastchess.audit_engine("DM-9M") == "dm"
    assert fastchess.audit_engine("DM-9M-ema") == "dm"
    assert fastchess.audit_engine("Blink-value-ship") == "blink"


def test_a_dm_gauntlet_through_fastchess_audits_deepminds_own_moves(tmp_path, monkeypatch):
    def run_fastchess(command, log: Path) -> int:
        pgn = Path(command[command.index("-pgnout") + 1].split("=", 1)[1])
        pgn.write_text(DM_PGN, encoding="utf-8")
        log.write_text("Games: 1, Wins: 0, Losses: 0, Draws: 1, Points: 0.5 (50.00 %)\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(fastchess, "run_fastchess", run_fastchess)
    (tmp_path / "book.pgn").write_text("", encoding="utf-8")
    gauntlet = fastchess.prepare_gauntlet(
        model="dm:9M", mode="policy", device="cpu", anchor=1320, games=2,
        book=str(tmp_path / "book.pgn"), out_dir=tmp_path,
    )  # fmt: skip
    assert gauntlet.plan.pgn_out.name.startswith("DM-9M_vs_SF1320_")
    report = fastchess.execute(gauntlet)
    assert report["blink"] == "DM-9M"
    assert report["audit"]["players"] == {"DM-9M": 2} and report["audit"]["compliant"]
    assert nosearch.audit([gauntlet.plan.pgn_out])["decisions"] == 0  # the old "blink" filter saw nothing


def test_stockfish_at_a_node_budget_is_full_strength_with_no_clock():
    spec = fastchess.stockfish_nodes(1024, Path("sf.exe"))
    tokens = spec.fastchess_args()
    assert spec.name == "SF19-n1024"
    assert "nodes=1024" in tokens
    assert not any(t.startswith(("st=", "tc=", "timemargin=")) for t in tokens)
    assert not any("UCI_LimitStrength" in t or "UCI_Elo" in t for t in tokens)
    assert "option.Threads=1" in tokens and "option.Hash=16" in tokens
