"""G6 audit data: 50 DM-9M puzzles with the moves the model actually played (plan E0, PF69).

    python tools/g6_audit.py --out D:/blink/eval/puzzles/g6_audit.json

DM-9M missed its pre-registered 88.9 +- 1.0 on the 10K puzzles, so the plan asks Itay to hand-check
50 puzzles against the scorer. This picks them reproducibly (seed 0): 25 the scorer marked unsolved
and 25 it marked solved, spread over the rating bands, from the params run's per-puzzle CSV. It
replays each with DeepMind's scorer (blink.eval.puzzles) on the CPU, recording every move the model
chose, so each verdict can be checked on the Lichess puzzle page itself.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import chess

from blink import paths
from blink.eval import puzzles

SEED = 0
PER_VERDICT = 25


class RecordingEngine:
    """DeepMind's Engine protocol that records each move it plays (UCI), in order."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.played: list[str] = []

    def play(self, board: chess.Board) -> chess.Move:
        move = self.agent.choose(board).move
        self.played.append(move.uci())
        return move


def pick(rows: list[dict[str, str]], seed: int = SEED, per_verdict: int = PER_VERDICT) -> list[str]:
    """Puzzle ids: per_verdict unsolved and per_verdict solved, round-robin over the rating bands."""
    rng = random.Random(seed)
    chosen = []
    for verdict in ("0", "1"):
        by_band: dict[str, list[str]] = {}
        for row in rows:
            if row["correct"] == verdict:
                by_band.setdefault(row["band"], []).append(row["puzzle_id"])
        for ids in by_band.values():
            rng.shuffle(ids)
        bands = sorted(by_band)
        taken: list[str] = []
        while len(taken) < per_verdict and any(by_band[b] for b in bands):
            for band in bands:
                if by_band[band] and len(taken) < per_verdict:
                    taken.append(by_band[band].pop())
        chosen += taken
    return chosen


def _san_line(board: chess.Board, moves: list[str]) -> list[str]:
    board, out = board.copy(), []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        out.append(board.san(move) if move in board.legal_moves else f"{uci} (illegal)")
        board.push(move)
    return out


def audit_one(row: dict[str, str], agent, expected: str) -> dict:
    """Replay one puzzle with the real scorer and describe it for a person checking it on Lichess."""
    moves = row["Moves"].split(" ")
    start = puzzles.board_from_pgn(row["PGN"])
    engine = RecordingEngine(agent)
    solved = puzzles.evaluate_puzzle_from_board(board=start.copy(), moves=moves, engine=engine)
    shown = start.copy()
    shown.push(chess.Move.from_uci(moves[0]))  # the position the solver sees
    solution = moves[1:]
    return {
        "puzzle_id": row["PuzzleId"],
        "rating": int(row["Rating"]),
        "url": f"https://lichess.org/training/{row['PuzzleId']}",
        "fen": shown.fen(),
        "to_move": "White" if shown.turn == chess.WHITE else "Black",
        "opponent_first_move": _san_line(start, moves[:1])[0],
        "solution_san": _san_line(shown, solution),
        "model_moves_san": _san_line_played(shown, solution, engine.played),
        "scorer_verdict": "solved" if solved else "not solved",
        "csv_verdict": "solved" if expected == "1" else "not solved",
    }


def _san_line_played(shown: chess.Board, solution: list[str], played: list[str]) -> list[str]:
    """The model's moves in SAN, each on the board it was played from (opponent replies from the solution)."""
    board, out = shown.copy(), []
    for i, uci in enumerate(played):
        move = chess.Move.from_uci(uci)
        out.append(board.san(move) if move in board.legal_moves else f"{uci} (illegal)")
        if uci != solution[2 * i]:
            break
        board.push(move)
        if 2 * i + 1 < len(solution):
            board.push(chess.Move.from_uci(solution[2 * i + 1]))
    return out


def main(argv: list[str] | None = None) -> int:
    from blink.reference import registry

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv", default=str(paths.home() / "eval" / "puzzles" / "puzzles_dm10k_dm_9M_action-value.csv")
    )
    parser.add_argument("--model", default="dm:9M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(paths.home() / "eval" / "puzzles" / "g6_audit.json"))
    args = parser.parse_args(argv)
    with open(args.csv, encoding="utf-8", newline="") as handle:
        verdicts = list(csv.DictReader(handle))
    ids = pick(verdicts)
    expected = {r["puzzle_id"]: r["correct"] for r in verdicts}
    rows = {
        r["PuzzleId"]: r
        for r in puzzles.read_puzzles(puzzles.resolve_set("dm10k"))
        if r["PuzzleId"] in set(ids)
    }
    agent = registry.load_agent(args.model, device=args.device)
    items = [audit_one(rows[i], agent, expected[i]) for i in ids]
    mismatches = [i["puzzle_id"] for i in items if i["scorer_verdict"] != i["csv_verdict"]]
    record = {
        "model": args.model,
        "seed": SEED,
        "source_csv": args.csv,
        "items": items,
        "replay_mismatches": mismatches,
    }
    Path(args.out).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} puzzles -> {args.out}; replay vs CSV disagreements {len(mismatches)}: {mismatches}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
