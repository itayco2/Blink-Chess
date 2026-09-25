"""G6 by the Lichess API: each audited puzzle checked against Lichess's own record (PF69).

    python tools/g6_lichess_check.py --audit D:/blink/eval/puzzles/g6_audit.json \
        --out D:/blink/eval/puzzles/g6_lichess_check.json

Itay asked the agent to run gate G6 (he does not play chess). Instead of a person replaying 50 puzzles on
lichess.org, this reads each puzzle from Lichess's public API (GET /api/puzzle/<id>: the game PGN up to
the puzzle and the official solution) and checks, independently of our CSV and scorer:
  1. position: the board after Lichess's PGN equals the position our harness showed the model;
  2. solution: Lichess's solution equals our solution line, move for move;
  3. verdict: Lichess's rule (the solution move, or any move that checkmates) applied to the moves the
     model played gives the same solved / not solved as our scorer.
One request at a time with a pause between them; a 429 waits a minute and retries once.
"""

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import chess
import chess.pgn

from blink.lichess.snapshot import tls_context

API = "https://lichess.org/api/puzzle/{}"
PAUSE_S = 1.5
RATE_LIMIT_WAIT_S = 65
CONTEXT = tls_context()


def fetch(puzzle_id: str) -> dict:
    request = urllib.request.Request(API.format(puzzle_id), headers={"Accept": "application/json"})
    for attempt in range(2):
        try:
            # Windows ROOT-store trust only: the CA store holds an expired cross-cert (blink.lichess.snapshot)
            with urllib.request.urlopen(request, timeout=30, context=CONTEXT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                time.sleep(RATE_LIMIT_WAIT_S)
                continue
            raise
    raise RuntimeError("unreachable")


def lichess_position(record: dict) -> chess.Board:
    """The board the solver faces: Lichess's game PGN played to its end (it stops at the puzzle start)."""
    game = chess.pgn.read_game(io.StringIO(record["game"]["pgn"]))
    if game is None:
        raise ValueError("unreadable PGN in the API record")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def lichess_verdict(board: chess.Board, solution: list[str], played: list[str]) -> str:
    """Lichess's rule on the model's moves: each must be the solution move or a checkmate."""
    board = board.copy()
    for i, uci in enumerate(played):
        expected = solution[2 * i]
        if uci != expected:
            board.push(chess.Move.from_uci(uci))
            return "solved" if board.is_checkmate() else "not solved"
        board.push(chess.Move.from_uci(uci))
        if 2 * i + 1 < len(solution):
            board.push(chess.Move.from_uci(solution[2 * i + 1]))
    return "solved" if 2 * len(played) >= len(solution) else "not solved"


def played_uci(fen: str, solution_san: list[str], played_san: list[str]) -> tuple[list[str], list[str]]:
    """Our recorded SAN (solution and the model's line) back to UCI on our own board."""
    board = chess.Board(fen)
    solution = []
    for san in solution_san:
        move = board.parse_san(san)
        solution.append(move.uci())
        board.push(move)
    board, played = chess.Board(fen), []
    for i, san in enumerate(played_san):
        move = board.parse_san(san)
        played.append(move.uci())
        if move.uci() != solution[2 * i]:
            break
        board.push(move)
        if 2 * i + 1 < len(solution):
            board.push(chess.Move.from_uci(solution[2 * i + 1]))
    return solution, played


def check_one(item: dict, record: dict) -> dict:
    ours = chess.Board(item["fen"])
    theirs = lichess_position(record)
    our_solution, played = played_uci(item["fen"], item["solution_san"], item["model_moves_san"])
    their_solution = list(record["puzzle"]["solution"])
    verdict = lichess_verdict(theirs, their_solution, played)
    position_match = theirs.board_fen() == ours.board_fen() and theirs.turn == ours.turn
    return {
        "puzzle_id": item["puzzle_id"],
        "position_match": position_match,
        "solution_match": their_solution == our_solution,
        "our_verdict": item["scorer_verdict"],
        "lichess_verdict": verdict,
        "agree": position_match and their_solution == our_solution and verdict == item["scorer_verdict"],
        "lichess_rating_now": record["puzzle"].get("rating"),
        "our_rating": item["rating"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", default="D:/blink/eval/puzzles/g6_audit.json")
    parser.add_argument("--out", default="D:/blink/eval/puzzles/g6_lichess_check.json")
    args = parser.parse_args(argv)
    items = json.loads(Path(args.audit).read_text(encoding="utf-8"))["items"]
    rows, errors = [], []
    for n, item in enumerate(items):
        if n:
            time.sleep(PAUSE_S)
        try:
            rows.append(check_one(item, fetch(item["puzzle_id"])))
        except (urllib.error.URLError, ValueError, KeyError) as exc:
            errors.append({"puzzle_id": item["puzzle_id"], "error": f"{type(exc).__name__}: {exc}"})
    agree = sum(r["agree"] for r in rows)
    summary = {
        "checked": len(rows),
        "agree": agree,
        "position_mismatches": [r["puzzle_id"] for r in rows if not r["position_match"]],
        "solution_mismatches": [r["puzzle_id"] for r in rows if not r["solution_match"]],
        "verdict_mismatches": [r["puzzle_id"] for r in rows if r["lichess_verdict"] != r["our_verdict"]],
        "errors": errors,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if agree == len(items) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
