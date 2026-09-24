"""`blink audit no-search`: rebuild the rows-per-move histogram from PGNs alone and check NSC-1."""

import json
from dataclasses import replace
from pathlib import Path

import chess

from blink.eval import books, match, nosearch
from blink.play import agents
from blink.play.oracles import RandomLogitEvaluator

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fastchess_blink_vs_sf.pgn"
FIRST_BLINK_MOVE = "9. Kf1 {+0.53/1 0.001s, n=5}"


def audit_text(tmp_path: Path, text: str) -> dict:
    path = tmp_path / "games.pgn"
    path.write_text(text, encoding="utf-8")
    return nosearch.audit([path])


def test_the_nosearch_audit_reads_node_counts_from_a_fastchess_pgn():
    report = nosearch.audit([FIXTURE])
    assert report["games"] == 2
    assert report["decisions"] == 57
    assert report["violations"] == []
    assert report["missing_counts"] == 0
    assert report["value_mode_full_batches"] == report["decisions"]
    assert report["max_rows"] <= report["max_legal"] + 1
    assert sum(report["histogram"].values()) == 57
    assert report["players"] == {"Blink-value-random": 57}
    assert report["terminations"] == {"normal": 2}


def test_the_audit_flags_more_rows_than_legal_moves_plus_one(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8").replace(
        FIRST_BLINK_MOVE, FIRST_BLINK_MOVE.replace("n=5", "n=6")
    )
    report = audit_text(tmp_path, text)
    assert [(v["ply"], v["rows"], v["legal"], v["rule"]) for v in report["violations"]] == [
        (16, 6, 4, "rows > L+1")
    ]


def test_the_audit_flags_zero_rows_without_a_mate(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8").replace(
        FIRST_BLINK_MOVE, FIRST_BLINK_MOVE.replace("n=5", "n=0")
    )
    assert [v["rule"] for v in audit_text(tmp_path, text)["violations"]] == ["0 rows without a mate"]


def test_the_audit_flags_a_blink_move_without_a_node_count(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8").replace(FIRST_BLINK_MOVE, "9. Kf1 {+0.53/1 0.001s}")
    report = audit_text(tmp_path, text)
    assert report["missing_counts"] == 1
    assert [v["rule"] for v in report["violations"]] == ["no node count"]


def test_the_audit_counts_terminations_and_forfeits_per_engine(tmp_path):
    games = [
        ("Blink-policy", "SF1320", "0-1", "time forfeit"),
        ("SF1320", "Blink-policy", "1/2-1/2", "adjudication"),
        ("SF1320", "Blink-policy", "1-0", "illegal move"),
    ]
    move = "1. e4 {+0.10/1 0.001s, n=1} *"
    text = "\n\n".join(
        f'[White "{w}"]\n[Black "{b}"]\n[Result "{r}"]\n[Termination "{t}"]\n\n{move}' for w, b, r, t in games
    )
    report = audit_text(tmp_path, text + "\n")
    assert report["terminations"] == {"time forfeit": 1, "adjudication": 1, "illegal move": 1}
    assert report["forfeits"] == {"Blink-policy": {"time forfeit": 1, "illegal move": 1}}
    assert report["adjudications"] == 1


def test_the_audit_reads_a_blink_match_pgn(tmp_path):
    blink = replace(agents.ValueAgent(RandomLogitEvaluator()), name="Blink-value-random")
    pgn = tmp_path / "match.pgn"
    fools = books.Opening(1, chess.STARTING_FEN, ("f2f3", "e7e5"))
    match.run_match(blink, agents.RandomAgent(), [fools], games=2, pgn_path=pgn, max_plies=30)
    report = nosearch.audit([pgn])
    assert report["violations"] == []
    assert report["decisions"] >= 15
    assert report["value_mode_full_batches"] + report["histogram"].get(0, 0) == report["decisions"]


def test_a_directory_is_audited_file_by_file_and_the_report_is_json(tmp_path):
    (tmp_path / "a.pgn").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "b.pgn").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    report = nosearch.audit(nosearch.pgn_files(tmp_path))
    out = nosearch.write_report(report, tmp_path / "out" / "nosearch.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert (loaded["games"], loaded["decisions"], loaded["files"]) == (4, 114, 2)
    assert loaded["compliant"] is True


PREFIX_NAMES = """[White "Blink-value-ship"]
[Black "Blink-value-ship-rules-off"]
[Result "*"]

1. e4 {+0.30/1 0.020s, n=1} e5 {+0.10/1 0.020s, n=1} 2. Nf3 {+0.20/1 0.020s, n=1} *

[White "DM-9M-ema"]
[Black "DM-9M"]
[Result "*"]

1. d4 {+0.30/1 0.020s, n=20} d5 {+0.10/1 0.020s} *
"""


def test_an_exact_audit_never_counts_a_player_whose_name_extends_another(tmp_path):
    """A substring filter would file Blink-value-ship-rules-off's moves under Blink-value-ship."""
    path = tmp_path / "g.pgn"
    path.write_text(PREFIX_NAMES, encoding="utf-8")
    names = ["Blink-value-ship", "Blink-value-ship-rules-off", "DM-9M", "DM-9M-ema"]
    audits = nosearch.audit_each([path], names)
    assert {n: a["players"] for n, a in audits.items()} == {
        "Blink-value-ship": {"Blink-value-ship": 2},
        "Blink-value-ship-rules-off": {"Blink-value-ship-rules-off": 1},
        "DM-9M": {"DM-9M": 1},
        "DM-9M-ema": {"DM-9M-ema": 1},
    }
    assert audits["Blink-value-ship"]["games"] == 1
    assert not audits["DM-9M"]["compliant"] and audits["DM-9M-ema"]["compliant"]
    assert nosearch.audit([path], engine="dm-9m")["players"] == {"DM-9M-ema": 1, "DM-9M": 1}


def test_searchless_players_are_blink_and_deepmind():
    assert nosearch.is_searchless("Blink-value-ship") and nosearch.is_searchless("DM-9M")
    assert not any(nosearch.is_searchless(n) for n in ("SF1800", "SF19-n256", "Material", "Random"))
