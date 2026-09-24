"""The P8 commands: eval books, endgames, sprt, static, block, all, and rate."""

import json

import chess
import pytest

from blink import cli
from blink.eval import books, endgames, rating, sflabel

GAME = '[Event "?"]\r\n[Result "*"]\r\n\r\n1. {moves} *\r\n\r\n'
MOVES = ["e4 e5", "d4 d5", "c4 c5", "Nf3 Nf6", "g3 g6", "b3 b6"]


@pytest.mark.parametrize("command", ["books", "endgames", "sprt", "static", "block", "all"])
def test_every_p8_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["eval", command, "--help"])
    assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_blink_rate_has_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["rate", "--help"])
    assert "--anchors" in capsys.readouterr().out


def test_eval_books_writes_both_slices_once(tmp_path, monkeypatch, capsys):
    source = tmp_path / "8moves_v3.pgn"
    source.write_bytes("".join(GAME.format(moves=m) for m in MOVES).encode("utf-8"))
    monkeypatch.setattr(books, "SLICES", {"dev": (1, 2), "final": (3, 6)})
    args = ["eval", "books", "--source", str(source), "--out", str(tmp_path / "out")]
    assert cli.main(args) == 0
    printed = capsys.readouterr().out
    assert "dev.pgn: openings 1-2 (2)" in printed and "final.pgn: openings 3-6 (4)" in printed
    assert cli.main(args) == 0  # a second run checks the hashes and writes nothing
    (tmp_path / "out" / "final.pgn").write_bytes(b"changed")
    assert cli.main(args) == 2


def test_eval_sprt_runs_between_two_tiny_agents(tmp_path, capsys):
    pgn = tmp_path / "s.pgn"
    book = tmp_path / "book.pgn"
    book.write_bytes("".join(GAME.format(moves=m) for m in MOVES).encode("utf-8"))
    args = ["eval", "sprt", "--a", "material", "--b", "random", "--games", "4", "--book", str(book)]
    assert cli.main([*args, "--max-plies", "40", "--out", str(pgn)]) == 0
    printed = capsys.readouterr().out
    assert "Material vs Random: 4 games" in printed and "verdict cap" in printed
    written = json.loads(pgn.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["games"] == 4 and sum(written["penta"]) == 2


def test_eval_endgames_screens_with_cached_labels(tmp_path, monkeypatch, capsys):
    epd = tmp_path / "e.epd"
    won = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
    epd.write_text(f"{won}\n{chess.STARTING_FEN}\n", encoding="utf-8")

    real = sflabel.SfLabeler
    pawns = {chess.Board(won).fen(): 9.0}

    def analyse(board, nodes, move):
        return sflabel.SfLabel(int(100 * pawns.get(board.fen(), 0.2)), None, 20, None)

    def fake(nodes, exe=None, cache_path=None, analyse_=None, procs=1):
        return real(nodes, cache_path=tmp_path / f"c{nodes}.jsonl", analyse=analyse)

    monkeypatch.setattr(sflabel, "SfLabeler", fake)
    args = ["eval", "endgames", "--epd", str(epd), "--out", str(tmp_path / "out"), "--screen-nodes", "10"]
    assert cli.main([*args, "--confirm-nodes", "20"]) == 0
    assert "screened 2 positions: 1 at +5.00" in capsys.readouterr().out
    assert len(endgames.read_set(tmp_path / "out", "dev")) == 1


def test_eval_all_dry_run_prints_the_protocol_and_the_game_table(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    protocol = tmp_path / "EVAL.md"
    protocol.write_text("# EVAL\n", encoding="utf-8")
    args = ["eval", "all", "--model", "ship", "--protocol", str(protocol), "--dry-run", "--games", "2"]
    assert cli.main(args) == 0
    printed = capsys.readouterr().out
    assert '"frozen": false' in printed and "| E3 |" in printed and "live training runs: none" in printed


def test_a_block_that_needs_the_mode_says_so_in_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    protocol = tmp_path / "EVAL.md"
    protocol.write_text("# EVAL\n", encoding="utf-8")
    args = [
        "eval",
        "block",
        "E4",
        "--model",
        "random",
        "--protocol",
        str(protocol),
        "--out",
        str(tmp_path / "o"),
    ]
    assert cli.main([*args, "--device", "cpu"]) == 2
    assert "shipped mode is unknown" in capsys.readouterr().err


def test_blink_rate_reports_fitted_and_unfittable_players(tmp_path, monkeypatch, capsys):
    pgn = tmp_path / "g.pgn"
    pgn.write_text('[White "A"]\n[Black "SF1320"]\n[Result "0-1"]\n\n1. e4 0-1\n', encoding="utf-8")

    def fake_run(files, anchors, out, simulations):
        out.mkdir(parents=True, exist_ok=True)
        tally = rating.tally_players(files)
        return rating.OrdoFit((), (), rating.unfittable(tally, anchors), tally, ("ordo",), {})

    monkeypatch.setattr(rating, "run_ordo", fake_run)
    assert cli.main(["rate", "--pgn", str(pgn), "--out", str(tmp_path / "o")]) == 0
    printed = capsys.readouterr().out
    assert "A" in printed and "not rated: all losses (0/1)" in printed
    assert json.loads((tmp_path / "o" / "rating.json").read_text(encoding="utf-8"))["excluded"] == {
        "A": "all losses"
    }
