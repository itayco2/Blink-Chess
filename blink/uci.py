"""blink-uci: Blink as a UCI engine (`python -m blink.uci` or the `blink-uci` script).

Speaks the subset of UCI that fastchess and lichess-bot use: uci, isready (loads the model and
warms it up), ucinewgame, position startpos|fen ... [moves ...], go (any time arguments), stop, quit.
Each `go` is one decision. Before `bestmove` it prints

    info depth 1 nodes <n_rows> score cp <X> pv <move>

where nodes is the number of positions the network scored for this move (1 in policy mode, L+1 in
value mode, 0 when a mate in one was played by rule R2). fastchess writes that count into every PGN
(`-pgnout nodes=true`), which is how `blink audit no-search` proves compliance from the games alone.
Castling is always printed as the king's two-square move (e1g1).
"""

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import chess

from blink.board import value
from blink.play import factory, rules
from blink.play.agents import Agent, Decision, ValueAgent
from blink.reference import registry

ENGINE_NAME = "Blink"
AUTHOR = "Itay Cohen"
WARMUP_DECISIONS = 5  # timed decisions after one cold call; their max stands in for the p99 (R5)
WIN_FLOOR = 1e-6


def win_to_cp(win: float) -> int:
    """Invert the Lichess logistic: the centipawn score whose win probability is `win`."""
    clipped = min(max(win, WIN_FLOOR), 1.0 - WIN_FLOOR)
    return int(round(math.log(clipped / (1.0 - clipped)) / value.LICHESS_K))


def info_line(decision: Decision) -> str:
    if decision.mate_now:
        score = "mate 1"
    else:
        score = f"cp {win_to_cp(decision.win) if decision.win is not None else 0}"
    return f"info depth 1 nodes {decision.n_rows} score {score} pv {decision.move.uci()}"


def parse_position(args: Sequence[str]) -> chess.Board:
    """`startpos|fen <fields> [moves m1 m2 ...]`; raises ValueError on a bad FEN or an illegal move."""
    if not args:
        raise ValueError("position needs startpos or fen")
    split = list(args).index("moves") if "moves" in args else len(args)
    if args[0] == "startpos":
        board = chess.Board()
    elif args[0] == "fen":
        board = chess.Board(" ".join(args[1:split]))
    else:
        raise ValueError(f"position must start with startpos or fen, got {args[0]!r}")
    for uci_move in args[split + 1 :]:
        board.push_uci(uci_move)
    return board


def remaining_seconds(args: Sequence[str], turn: chess.Color) -> float | None:
    """The mover's clock from `go wtime/btime` (ms), or None under movetime, depth or infinite."""
    key = "wtime" if turn == chess.WHITE else "btime"
    tokens = list(args)
    if key in tokens and tokens.index(key) + 1 < len(tokens):
        return max(0.0, float(tokens[tokens.index(key) + 1]) / 1000.0)
    return None


def warm_up(agent: Agent) -> Agent:
    """One cold decision (CUDA context, kernels), then timed ones; value mode keeps their max as p99."""
    quiet = replace(agent, sink=None) if hasattr(agent, "sink") else agent
    board = chess.Board()
    quiet.choose(board)
    times = []
    for _ in range(WARMUP_DECISIONS):
        start = time.perf_counter()
        quiet.choose(board)
        times.append(time.perf_counter() - start)
    return replace(agent, p99_s=max(times)) if isinstance(agent, ValueAgent) else agent


class UciEngine:
    def __init__(self, agent_factory: Callable[[], Agent], out: TextIO, name: str = ENGINE_NAME) -> None:
        self._factory = agent_factory
        self._agent: Agent | None = None
        self._out = out
        self._name = name
        self._board = chess.Board()
        self._games = 0

    def send(self, line: str) -> None:
        self._out.write(line + "\n")
        self._out.flush()

    def ready(self) -> Agent:
        if self._agent is None:
            self._agent = warm_up(self._factory())
        return self._agent

    def handle(self, line: str) -> bool:
        """Handle one input line; False means quit."""
        tokens = line.split()
        if not tokens:
            return True
        command, args = tokens[0], tokens[1:]
        if command == "quit":
            return False
        if command == "uci":
            self.send(f"id name {self._name}")
            self.send(f"id author {AUTHOR}")
            self.send("uciok")
        elif command == "isready":
            self.ready()
            self.send("readyok")
        elif command == "ucinewgame":
            self.ready()
            self._games += 1
            self._board = chess.Board()
        elif command == "position":
            self._position(args)
        elif command == "go":
            self._go(args)
        elif command not in ("stop", "ponderhit", "setoption", "debug", "register"):
            self.send(f"info string unknown command {command}")
        return True

    def _position(self, args: Sequence[str]) -> None:
        try:
            self._board = parse_position(args)
        except ValueError as exc:
            self.send(f"info string ignored position ({exc}); keeping the previous one")

    def _go(self, args: Sequence[str]) -> None:
        agent = self.ready()
        board = self._board
        if not any(board.generate_legal_moves()):
            self.send("bestmove 0000")
            return
        game = f"g{max(self._games, 1)}"
        decision = agent.choose(board, remaining_s=remaining_seconds(args, board.turn), game=game)
        self.send(info_line(decision))
        self.send(f"bestmove {decision.move.uci()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blink-uci", description="Blink as a UCI engine (no search).")
    parser.add_argument(
        "--model", default="ship", help="run:<name>[:ema] | ship | release:<tag> | <path> | dm:9M[:ema]"
    )
    parser.add_argument("--mode", choices=factory.MODES, default="policy")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--random", action="store_true", help="a random-logit network, for harness tests")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random-logit network")
    parser.add_argument("--epsilon", type=float, default=rules.DEFAULT_EPSILON, help="R4 tie window")
    parser.add_argument("--log", type=Path, help="append one JSON line per decision to this file")
    parser.add_argument("--name", default=None, help="the name sent in `id name`")
    return parser


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    selector = "random" if args.random else args.model
    is_deepmind = registry.is_dm(selector)
    try:
        (registry.check_available if is_deepmind else factory.check_available)(selector)
    except factory.ModelUnavailable as exc:
        print(f"blink-uci: {exc}", file=sys.stderr)
        return 2
    sink = factory.JsonlSink(args.log) if args.log else None

    def make() -> Agent:
        if is_deepmind:  # DeepMind's released play logic: L rows per move, no Blink mode
            return registry.load_agent(selector, device=args.device, sink=sink)
        evaluator = factory.load_evaluator(selector, device=args.device, seed=args.seed)
        return factory.make_agent(args.mode, evaluator, epsilon=args.epsilon, sink=sink)

    name = args.name or (registry.parse(selector).name if is_deepmind else f"{ENGINE_NAME}-{args.mode}")
    engine = UciEngine(make, stdout, name=name)
    for line in iter(stdin.readline, ""):
        if not engine.handle(line.strip()):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
