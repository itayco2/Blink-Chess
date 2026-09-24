"""The commands `blink match`, `blink gauntlet`, `blink eval puzzles|signcheck`, `blink audit no-search`."""

import csv
import json
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest

from blink import cli
from blink.board import encode, moves
from blink.data.record import ROOT_DTYPE

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fastchess_blink_vs_sf.pgn"


def write_book(path: Path) -> Path:
    games = ["1. e4 e5 2. Nf3 Nc6", "1. d4 d5 2. c4 e6", "1. c4 c5"]
    path.write_text(
        "\n\n".join(f'[Event "?"]\n[Result "*"]\n\n{g} *' for g in games) + "\n", encoding="utf-8"
    )
    return path


def count_games(path: Path) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in iter(lambda: chess.pgn.read_game(handle), None))


@pytest.mark.parametrize("command", [["match"], ["gauntlet"], ["eval", "puzzles"], ["eval", "signcheck"]])
def test_every_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([*command, "--help"])
    assert exc.value.code == 0
    assert "--model" in capsys.readouterr().out or command == ["match"]


def test_audit_no_search_has_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["audit", "no-search", "--help"])
    assert "--pgn" in capsys.readouterr().out


def test_blink_match_writes_a_pgn_and_a_summary(tmp_path, capsys):
    pgn = tmp_path / "m.pgn"
    args = [
        "match",
        "--a",
        "random",
        "--b",
        "material",
        "--games",
        "4",
        "--book",
        str(write_book(tmp_path / "b.pgn")),
    ]
    assert cli.main([*args, "--out", str(pgn), "--max-plies", "60"]) == 0
    assert count_games(pgn) == 4
    summary = json.loads(pgn.with_suffix(".json").read_text(encoding="utf-8"))
    assert (summary["games"], summary["a"], summary["b"]) == (4, "Random", "Material")
    assert "Random vs Material" in capsys.readouterr().out


def test_a_random_net_blink_match_then_the_audit_is_clean(tmp_path, capsys):
    pgn = tmp_path / "blink.pgn"
    book = str(write_book(tmp_path / "b.pgn"))
    args = ["match", "--a", "random-net", "--mode", "value", "--b", "random", "--games", "2", "--book", book]
    assert cli.main([*args, "--out", str(pgn), "--max-plies", "40", "--device", "cpu"]) == 0
    out = tmp_path / "nosearch.json"
    assert cli.main(["audit", "no-search", "--pgn", str(tmp_path), "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["compliant"] is True
    assert list(report["players"]) == ["Blink-value-random-net"]


def test_the_audit_command_fails_on_a_violation(tmp_path):
    bad = tmp_path / "bad.pgn"
    bad.write_text(FIXTURE.read_text(encoding="utf-8").replace("n=5}", "n=99}", 1), encoding="utf-8")
    assert cli.main(["audit", "no-search", "--pgn", str(bad), "--out", str(tmp_path / "ns.json")]) == 1


def test_blink_eval_puzzles_writes_both_modes(tmp_path, capsys):
    source = tmp_path / "set.csv"
    with open(source, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PuzzleId", "Rating", "PGN", "Moves"])
        writer.writerow(["s1", "650", "1. e4 e5 2. Bc4 Nc6 3. Qh5", "g8f6 h5f7"])
        writer.writerow(["s2", "1700", "1. e4 e5", "g1f3 b8c6"])
    args = ["eval", "puzzles", "--set", str(source), "--model", "random", "--mode", "both", "--limit", "2"]
    assert cli.main([*args, "--out", str(tmp_path / "out"), "--device", "cpu"]) == 0
    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == [
        "puzzles_set_random_policy.csv",
        "puzzles_set_random_policy.json",
        "puzzles_set_random_value.csv",
        "puzzles_set_random_value.json",
    ]
    assert "<1000" in capsys.readouterr().out


def test_blink_eval_signcheck_reads_a_val_file(tmp_path, capsys):
    board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    records = np.zeros(2, dtype=ROOT_DTYPE)
    records["board"] = encode.pack(encode.encode_board(board))
    records["move"] = moves.encode_move(board, chess.Move.from_uci("d1d5"))
    records.tofile(tmp_path / "val.bin")
    out = tmp_path / "signcheck.json"
    code = cli.main(["eval", "signcheck", "--model", "random", "--data", str(tmp_path), "--out", str(out)])
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n"] == 2
    assert code == (0 if result["passes_3x"] else 1)


def test_blink_gauntlet_dry_run_prints_the_fastchess_command(tmp_path, capsys):
    args = ["gauntlet", "--model", "random", "--mode", "policy", "--anchor", "1320", "--games", "10"]
    assert cli.main([*args, "--dry-run", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "st=1" in printed and "st=0.1" in printed and "option.UCI_Elo=1320" in printed
    assert "-maxmoves 300" in printed


def test_a_missing_model_loader_is_one_clear_line_not_a_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "blink.model.loading", None)
    code = cli.main(["gauntlet", "--model", "run:skeleton", "--out", str(tmp_path)])
    assert code == 2
    assert "blink.model.loading" in capsys.readouterr().err


def test_missing_inputs_are_one_clear_line_and_exit_2(tmp_path, capsys):
    assert cli.main(["eval", "puzzles", "--set", str(tmp_path / "none.csv"), "--model", "random"]) == 2
    assert cli.main(["eval", "signcheck", "--model", "random", "--data", str(tmp_path)]) == 2
    assert capsys.readouterr().out.count("does not exist") == 2
