"""`blink match` plays the whole P3 ladder: the learned rungs by selector, and every pair in a round robin."""

import json
import os
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn
import pytest

from blink import cli
from blink.commands import play as play_command
from blink.eval import roundrobin
from blink.play import agents, factory, rules
from blink.play.oracles import MaterialEvaluator

REPO = Path(__file__).resolve().parent.parent
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
STALEMATE_TRAP = "k7/2K5/8/1P6/8/8/8/8 w - - 0 1"  # b5-b6 stalemates; no mate in one


def write_book(path: Path) -> Path:
    games = ["1. e4 e5 2. Nf3 Nc6", "1. d4 d5 2. c4 e6", "1. c4 c5"]
    path.write_text(
        "\n\n".join(f'[Event "?"]\n[Result "*"]\n\n{g} *' for g in games) + "\n", encoding="utf-8"
    )
    return path


def count_games(path: Path) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in iter(lambda: chess.pgn.read_game(handle), None))


def side(selector: str, epsilon: float = rules.DEFAULT_EPSILON) -> agents.Agent:
    return play_command.side_agent(selector, "policy", "cpu", 0, epsilon)


@pytest.fixture
def ladder_home(tmp_path, monkeypatch) -> Path:
    """BLINK_HOME with untrained linear and MLP rungs where `blink baselines train` writes them."""
    pytest.importorskip("torch")
    from blink.baselines import models

    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    for kind in models.KINDS:
        path = tmp_path / "runs" / f"baseline-{kind}" / "model.pt"
        models.save(path, models.build(kind), kind, {"note": "untrained"})
    return tmp_path


# ---------------------------------------------------------------- the selectors


def test_the_match_plays_material_through_the_one_value_agent_every_baseline_uses():
    material = side("material")
    assert material == agents.ValueAgent(MaterialEvaluator(), name="Material")
    assert material == factory.material_agent()


@pytest.mark.torch
@pytest.mark.parametrize("selector", ["material", "linear", "mlp", "baseline-file"])
def test_every_ladder_side_in_a_match_uses_the_same_agent_wrapper_and_rules(selector, ladder_home):
    if selector == "baseline-file":
        selector = "baseline:" + str(ladder_home / "runs" / "baseline-mlp" / "model.pt")
    agent = side(selector)
    assert isinstance(agent, agents.ValueAgent)
    mate = agent.choose(chess.Board(MATE_IN_ONE))
    assert mate.move == chess.Move.from_uci("a1a8") and mate.n_calls == 0 and "R2" in mate.rules
    trap = chess.Board(STALEMATE_TRAP)
    decision = agent.choose(trap)
    assert decision.n_calls == 1 and decision.n_rows == trap.legal_moves.count() + 1
    assert "R1" in decision.rules and "R3" in decision.rules


@pytest.mark.torch
def test_the_match_sides_equal_the_baselines_own_agents_apart_from_epsilon(ladder_home):
    from blink.baselines import evaluator

    assert side("material") == evaluator.baseline_agent("material")
    for kind, name in (("linear", "Linear"), ("mlp", "MLP")):
        agent = side(kind, epsilon=0.02)
        assert agent.name == name and agent.epsilon == 0.02
        assert type(agent.evaluator) is evaluator.BaselineEvaluator
    assert side("material", epsilon=0.02).epsilon == 0.02


@pytest.mark.torch
def test_a_missing_baseline_is_one_clear_line_and_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    book = str(write_book(tmp_path / "b.pgn"))
    argv = ["match", "--a", "random", "--b", "linear", "--games", "2", "--book", book, "--device", "cpu"]
    assert cli.main([*argv, "--out", str(tmp_path / "m.pgn")]) == 2
    err = capsys.readouterr().err
    assert "baseline-linear" in err and "blink baselines train" in err


@pytest.mark.torch
def test_a_match_against_a_baseline_file_names_the_rung(ladder_home, tmp_path, capsys):
    pgn = tmp_path / "m.pgn"
    weights = ladder_home / "runs" / "baseline-linear" / "model.pt"
    argv = ["match", "--a", "random", "--b", f"baseline:{weights}", "--games", "2"]
    argv += ["--book", str(write_book(tmp_path / "b.pgn")), "--device", "cpu", "--max-plies", "20"]
    assert cli.main([*argv, "--out", str(pgn)]) == 0
    summary = json.loads(pgn.with_suffix(".json").read_text(encoding="utf-8"))
    assert (summary["a"], summary["b"], summary["games"]) == ("Random", "Linear", 2)


def test_random_and_material_sides_import_no_torch():
    code = (
        "import sys; from blink.commands import play; "
        "[play.side_agent(s, 'policy', 'cpu', 0, 0.0) for s in ('material', 'random')]; "
        "print('torch' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


# ---------------------------------------------------------------- the round robin


def test_the_cross_table_scores_each_side_from_its_own_view():
    summaries = {
        (0, 1): {"games": 4, "a_wins": 3, "draws": 1, "a_losses": 0},
        (0, 2): {"games": 4, "a_wins": 0, "draws": 0, "a_losses": 4},
        (1, 2): {"games": 4, "a_wins": 1, "draws": 2, "a_losses": 1},
    }
    table = roundrobin.cross_table(["x", "y", "z"], summaries)
    assert table["x"] == {"y": 0.875, "z": 0.0}
    assert table["y"] == {"x": 0.125, "z": 0.5}
    assert table["z"] == {"x": 1.0, "y": 0.5}
    scores = roundrobin.total_scores(["x", "y", "z"], summaries)
    assert scores == {"x": 3.5 / 8, "y": 2.5 / 8, "z": 6 / 8}


def test_the_pairs_are_every_unordered_pair_in_the_order_given():
    assert roundrobin.pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_the_printed_table_has_one_row_per_side_and_a_dash_on_the_diagonal():
    table = {"random": {"material": 0.25}, "material": {"random": 0.75}}
    text = roundrobin.format_table(["random", "material"], table, {"random": 0.25, "material": 0.75})
    lines = text.splitlines()
    assert lines[0].split() == ["random", "material", "score"]
    assert lines[1].split() == ["random", "-", "25.0%", "25.0%"]
    assert lines[2].split() == ["material", "75.0%", "-", "75.0%"]


def _round_robin(tmp_path, sides: str, games: int = 2, *extra: str) -> list[str]:
    book = str(write_book(tmp_path / "b.pgn"))
    return ["match", "--round-robin", sides, "--games", str(games), "--book", book, "--device", "cpu", *extra]


def test_a_round_robin_plays_every_pair_on_the_same_openings(tmp_path, capsys):
    out = tmp_path / "rr"
    argv = _round_robin(tmp_path, "random,material,random-net", 4, "--max-plies", "16", "--mode", "value")
    assert cli.main([*argv, "--out", str(out)]) == 0
    record = json.loads((out / "round_robin.json").read_text(encoding="utf-8"))
    assert record["sides"] == ["random", "material", "random-net"]
    assert [(p["a"], p["b"]) for p in record["pairs"]] == [
        ("random", "material"),
        ("random", "random-net"),
        ("material", "random-net"),
    ]
    for pair in record["pairs"]:
        pgn = out / pair["pgn"]
        assert count_games(pgn) == 4 and pgn.with_suffix(".json").is_file()
        with open(pgn, encoding="utf-8") as handle:
            books = [chess.pgn.read_game(handle).headers["BookIndex"] for _ in range(4)]
        assert books == ["1", "1", "2", "2"]
    assert record["table"]["random"]["material"] == pytest.approx(1 - record["table"]["material"]["random"])
    printed = capsys.readouterr().out
    assert "random-net" in printed and "score" in printed and str(out / "round_robin.json") in printed


@pytest.mark.torch
def test_the_p3_round_robin_runs_the_four_rungs(ladder_home, tmp_path, capsys):
    out = tmp_path / "ladder"
    argv = _round_robin(tmp_path, "random,material,linear,mlp", 2, "--max-plies", "12")
    assert cli.main([*argv, "--out", str(out)]) == 0
    record = json.loads((out / "round_robin.json").read_text(encoding="utf-8"))
    assert len(record["pairs"]) == 6 and set(record["scores"]) == {"random", "material", "linear", "mlp"}
    names = {(p["a_name"], p["b_name"]) for p in record["pairs"]}
    assert ("Linear", "MLP") in names and ("Random", "Material") in names


@pytest.mark.parametrize(
    "extra",
    [
        ["--round-robin", "random"],
        ["--round-robin", "random,random"],
        ["--round-robin", "random,material", "--a", "random"],
        [],
    ],
)
def test_a_bad_round_robin_or_missing_sides_is_one_line_and_exit_2(extra, tmp_path, capsys):
    argv = ["match", *extra, "--games", "2", "--book", str(write_book(tmp_path / "b.pgn"))]
    assert cli.main([*argv, "--out", str(tmp_path / "rr")]) == 2
    assert "blink match:" in capsys.readouterr().err


def test_a_round_robin_never_appends_to_the_games_of_an_earlier_one(tmp_path, capsys):
    out = tmp_path / "rr"
    argv = [*_round_robin(tmp_path, "random,material"), "--max-plies", "6", "--out", str(out)]
    assert cli.main(argv) == 0
    capsys.readouterr()
    assert cli.main(argv) == 2
    assert "already holds" in capsys.readouterr().err
    assert count_games(out / "random_vs_material.pgn") == 2
