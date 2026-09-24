"""The UCI engine: handshake, positions, castling notation, node counts and no network at play time."""

import io
import json
import os
import random
import re
import socket
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np

from blink import uci
from blink.board import moves
from blink.play import agents
from blink.play.evaluator import Evaluation
from blink.play.oracles import RandomLogitEvaluator, one_hot_value

REPO = Path(__file__).resolve().parent.parent
INFO = re.compile(r"^info depth 1 nodes (\d+) score (cp -?\d+|mate 1) pv ([a-h][1-8][a-h][1-8][qrbn]?)$")
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"


class PreferMoves:
    """Policy logit +10 at the given vocab indices, a flat 0.5 value everywhere."""

    def __init__(self, *indices: int) -> None:
        self.indices = indices

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        policy = np.zeros((len(codes), moves.NUM_MOVES), dtype=np.float32)
        policy[:, list(self.indices)] = 10.0
        return Evaluation(policy, one_hot_value(np.full(len(codes), 0.5)))


def run_engine(lines: list[str], agent) -> list[str]:
    out = io.StringIO()
    engine = uci.UciEngine(lambda: agent, out)
    for line in lines:
        if not engine.handle(line):
            break
    return out.getvalue().splitlines()


def bestmove(lines: list[str]) -> str:
    return next(line.split()[1] for line in reversed(lines) if line.startswith("bestmove"))


def test_uci_handshake_position_go_bestmove():
    lines = run_engine(
        [
            "uci",
            "isready",
            "ucinewgame",
            "position startpos moves e2e4 e7e5",
            "go wtime 60000 btime 60000 winc 1000 binc 1000",
            "quit",
        ],
        agents.PolicyAgent(RandomLogitEvaluator()),
    )
    assert lines[0].startswith("id name Blink")
    assert lines.index("uciok") < lines.index("readyok")
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert chess.Move.from_uci(bestmove(lines)) in board.legal_moves


def test_uci_emits_standard_castling():
    for fen, castle in (
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
    ):
        index = moves.encode_move(chess.Board(fen), chess.Move.from_uci(castle))
        for agent in (agents.PolicyAgent(PreferMoves(index)), agents.ValueAgent(PreferMoves(index))):
            lines = run_engine([f"position fen {fen}", "go movetime 1000"], agent)
            assert bestmove(lines) == castle


def test_uci_accepts_king_takes_rook_castling_in_the_move_list():
    lines = run_engine(
        ["position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1h1", "go movetime 100"],
        agents.PolicyAgent(RandomLogitEvaluator()),
    )
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_uci("e1g1")
    assert chess.Move.from_uci(bestmove(lines)) in board.legal_moves


def test_uci_reports_rows_as_nodes():
    cases = (
        (agents.ValueAgent(RandomLogitEvaluator()), "position startpos", "21"),
        (agents.PolicyAgent(RandomLogitEvaluator()), "position startpos", "1"),
        (agents.ValueAgent(RandomLogitEvaluator()), f"position fen {MATE_IN_ONE}", "0"),
        (agents.PolicyAgent(RandomLogitEvaluator()), f"position fen {MATE_IN_ONE}", "0"),
    )
    for agent, position, nodes in cases:
        lines = run_engine([position, "go movetime 1000"], agent)
        info = INFO.match(lines[-2])
        assert info is not None, lines
        assert info.group(1) == nodes
        assert info.group(3) == bestmove(lines)
        assert (info.group(2) == "mate 1") == (nodes == "0")


def test_go_on_a_finished_game_answers_the_null_move():
    lines = run_engine(
        ["position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "go"], agents.PolicyAgent(RandomLogitEvaluator())
    )
    assert lines[-1] == "bestmove 0000"


def test_an_illegal_move_list_keeps_the_previous_position_and_says_why():
    lines = run_engine(
        ["position startpos moves e2e5", "go movetime 100"], agents.PolicyAgent(RandomLogitEvaluator())
    )
    assert lines[0].startswith("info string")
    assert chess.Move.from_uci(bestmove(lines)) in chess.Board().legal_moves


def random_game_positions(count: int, seed: int) -> list[tuple[str, list[str]]]:
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randrange(0, 60)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
        cut = rng.randrange(0, len(board.move_stack) + 1)
        start = chess.Board()
        for move in board.move_stack[:cut]:
            start.push(move)
        out.append((start.fen(), [m.uci() for m in board.move_stack[cut:]]))
    return out


def test_the_uci_engine_plays_50_positions_with_sockets_disabled(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network I/O at play time")

    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, no_network)
    positions = random_game_positions(50, seed=50)
    script = ["uci", "isready"]
    for fen, tail in positions:
        script += ["ucinewgame", f"position fen {fen} moves {' '.join(tail)}".rstrip(), "go movetime 1000"]
    out = io.StringIO()
    code = uci.main(
        ["--random", "--mode", "value", "--device", "cpu"], io.StringIO("\n".join(script) + "\nquit\n"), out
    )
    played = [line.split()[1] for line in out.getvalue().splitlines() if line.startswith("bestmove")]
    assert code == 0
    assert len(played) == 50
    for (fen, tail), move in zip(positions, played, strict=True):
        board = chess.Board(fen)
        for uci_move in tail:
            board.push_uci(uci_move)
        assert chess.Move.from_uci(move) in board.legal_moves


def test_the_decision_log_gets_one_json_line_per_move(tmp_path):
    log = tmp_path / "decisions.jsonl"
    script = ["uci", "isready", "ucinewgame", "position startpos", "go movetime 100"]
    script += ["position startpos moves e2e4", "go", "quit"]
    uci.main(
        ["--random", "--mode", "value", "--log", str(log)], io.StringIO("\n".join(script)), io.StringIO()
    )
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [(r["game"], r["ply"], r["n_rows"], r["n_calls"]) for r in records] == [
        ("g1", 0, 21, 1),
        ("g1", 1, 21, 1),
    ]


def test_a_missing_model_loader_fails_fast_with_a_clear_message(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "blink.model.loading", None)
    code = uci.main(["--model", "run:skeleton"], io.StringIO("uci\nquit\n"), io.StringIO())
    assert code == 2
    assert "blink.model.loading" in capsys.readouterr().err


def test_python_dash_m_blink_uci_speaks_uci_over_pipes():
    proc = subprocess.run(
        [sys.executable, "-m", "blink.uci", "--random", "--mode", "policy", "--device", "cpu"],
        input="uci\nisready\nposition startpos moves d2d4\ngo movetime 200\nquit\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    lines = proc.stdout.splitlines()
    assert proc.returncode == 0, proc.stderr
    assert "uciok" in lines and "readyok" in lines
    assert INFO.match(lines[-2]).group(1) == "1"
    assert bestmove(lines) in {
        m.uci() for m in chess.Board("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b").legal_moves
    }
