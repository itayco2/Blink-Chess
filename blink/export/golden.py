"""site/tests/golden.json: 50 positions the browser tokenizer and onnxruntime-web must reproduce (PF31).

For each FEN: the 64 square codes, every legal move as (vocabulary index, uci), the top-5 policy after a
softmax over legal moves only (one look), and the expected win probability for the side to move.
All 50 positions go through the network as one batch, so golden generation is one network call.
"""

import json
from pathlib import Path
from typing import Any

import chess
import numpy as np

from blink.board import encode, moves
from blink.export.evaluators import softmax
from blink.play.evaluator import Evaluator

TOP_K = 5
FORMAT_VERSION = 1

# Real move orders, castling and en-passant edge cases, promotions, checks, and CC0 eval DB rows
# (tests/fixtures/eval_lines_first100.jsonl). Tags are derived, never hand-written: see tags().
GOLDEN_FENS: tuple[str, ...] = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq d6 0 2",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pp1ppppp/8/8/2pPP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3",
    "rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQ1RK1 b - - 1 7",
    "r2q1rk1/1p1nbppp/p2pbn2/4p3/4P1P1/1NN1BP2/PPPQ3P/2KR1B1R b - g3 0 11",
    "rnbqk2r/pp2nppp/4p3/2ppP3/3P2Q1/P1P5/2P2PPP/R1B1KBNR b KQkq - 2 7",
    "r2qk2r/pp1n1ppp/2p1pn2/5b2/PbBP4/2N1PN2/1P2QPPP/R1B2RK1 b kq - 4 9",
    "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9",
    "r3k2r/pppqbppp/2npbn2/4p3/4P3/2NPBN2/PPPQBPPP/R3K2R w KQkq - 0 10",
    "r3k2r/pppqbppp/2npbn2/4p3/4P3/2NPBN2/PPPQBPPP/R3K2R b KQkq - 0 10",
    "4k3/8/8/8/2b5/8/8/R3K2R w KQ - 0 1",
    "r3k2r/8/8/8/8/8/8/3RK3 b kq - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w KQ - 0 1",
    "r3k2r/8/8/8/8/8/8/R4K1R w KQkq - 0 1",
    "r3k3/8/8/8/8/8/8/4K2R b Kkq - 0 1",
    "4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3pPp2/8/8/4K3 b - e3 0 1",
    "8/8/8/KPp4r/8/8/8/7k w - c6 0 2",
    "8/8/8/8/k2Pp2Q/8/8/7K b - d3 0 1",
    "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
    "r3k3/1P6/8/8/8/8/8/4K3 w q - 0 1",
    "3r4/2P1k3/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/6p1/4K2R b K - 0 1",
    "8/8/8/8/8/5k2/p7/4K3 b - - 0 1",
    "4k3/8/8/8/8/8/1p6/R3K3 b Q - 0 1",
    "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1",
    "4k3/4r3/8/8/8/8/8/4K3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PP1/3r2K1 w - - 0 1",
    "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - - 0 1",
    "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - - 0 1",
    "6k1/6p1/6N1/4K3/4N3/8/8/8 b - - 0 1",
    "8/8/2N2k2/8/1p2p3/p7/K7/8 b - - 0 1",
    "8/1r6/2R2pk1/6pp/3P4/6P1/5K1P/8 w - - 0 1",
    "1R4k1/3q1pp1/6n1/b2p2Pp/2pP2b1/p1P5/P1BQrPPB/5NK1 b - - 0 1",
    "1R6/3q1ppk/6n1/b2p2Pp/2pP2b1/p1P5/P1B1rPPB/2Q2NK1 b - - 0 1",
    "3r4/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 w - - 0 1",
    "1r2kb1r/pBp2ppp/4pn2/5b2/Q1pq4/6P1/PP1NPP1P/R1B2RK1 b k - 0 1",
    "r2k2r1/pppb1p1p/2p5/8/3Bn3/8/PPP2PPP/2KR1B1R b - - 0 1",
    "2r1r1k1/5ppp/8/8/Q7/8/5PPP/4R1K1 w - - 0 1",
    "rnbqkbnr/pp2pppp/2p5/3P4/3P4/8/PP2PPPP/RNBQKBNR b KQkq - 0 1",
    "8/4k3/8/4K3/8/4P3/8/8 b - - 0 1",
    "r1b2rk1/pp3ppp/1q2p3/2npP1N1/8/8/PPQ2PPP/R3RBK1 b - - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "k1K5/8/8/1P6/8/8/8/8 b - - 0 1",
    "r2qk2r/3n2p1/1pp1p3/3pPpb1/P2P1nBp/1NB4P/1PP2P2/R3QR1K w kq f6 0 1",
    "rnbqkb1r/pp2pp1p/5np1/2pp4/3P4/1P1BPN2/P1P2PPP/RNBQK2R b KQkq - 0 1",
)


def tags(board: chess.Board) -> list[str]:
    """What a position exercises, derived from python-chess."""
    legal = list(board.legal_moves)
    checks = {
        "black_to_move": board.turn == chess.BLACK,
        "castling": any(board.is_castling(move) for move in legal),
        "promotion": any(move.promotion for move in legal),
        "en_passant": board.has_legal_en_passant(),
        "ep_square_without_legal_capture": board.ep_square is not None and not board.has_legal_en_passant(),
        "castling_rights_without_king_or_rook": board.castling_rights != board.clean_castling_rights(),
        "check": board.is_check(),
    }
    return [name for name, hit in checks.items() if hit]


def _entry(fen: str, codes: np.ndarray, policy_logits: np.ndarray, win: float) -> dict[str, Any]:
    board = chess.Board(fen)
    legal = sorted((moves.encode_move(board, move), move.uci()) for move in board.legal_moves)
    indices = [index for index, _ in legal]
    probs = softmax(policy_logits[indices].astype(np.float64))
    order = np.argsort(-probs, kind="stable")[:TOP_K]
    top = [{"index": legal[k][0], "uci": legal[k][1], "prob": round(float(probs[k]), 8)} for k in order]
    return {
        "fen": fen,
        "tags": tags(board),
        "codes": codes.tolist(),
        "legal": [[index, uci] for index, uci in legal],
        "top5": top,
        "win": round(float(win), 8),
    }


def build(evaluator: Evaluator, model: dict[str, Any], fens: tuple[str, ...] = GOLDEN_FENS) -> dict[str, Any]:
    """The golden record for `fens` under one evaluator call."""
    codes = np.stack([encode.encode_board(chess.Board(fen)) for fen in fens])
    evaluation = evaluator.evaluate(codes)
    win = evaluation.win_probability()
    positions = [_entry(fen, codes[i], evaluation.policy_logits[i], win[i]) for i, fen in enumerate(fens)]
    return {"version": FORMAT_VERSION, "model": dict(model), "top_k": TOP_K, "positions": positions}


def write(data: dict[str, Any], path: Path) -> Path:
    """One position per line, so a diff shows which position moved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {key: item for key, item in data.items() if key != "positions"}
    rows = ",\n".join("    " + json.dumps(entry, separators=(",", ":")) for entry in data["positions"])
    body = json.dumps(head, indent=2)[:-2] + ',\n  "positions": [\n' + rows + "\n  ]\n}\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    return path
