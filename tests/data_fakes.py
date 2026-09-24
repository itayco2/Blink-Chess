"""Small fakes for the data tests: pzstd-like files and synthetic eval DB lines.

pzstd writes each zstd frame behind a 12-byte skippable frame whose payload is the compressed size of
the frame that follows, and cuts the text at fixed sizes, so lines cross frame boundaries.
"""

import random
import struct
from pathlib import Path

import chess
import orjson
import zstandard

SKIPPABLE_MAGIC = 0x184D2A50
FIXTURE = Path(__file__).parent / "fixtures" / "eval_lines_first100.jsonl"


def fixture_lines() -> list[bytes]:
    return FIXTURE.read_bytes().splitlines()


def write_pzstd(path: Path, text: bytes, frame_bytes: int) -> list[int]:
    """Write `text` as pzstd does, cut every `frame_bytes`. Returns each frame's end offset in the file."""
    compressor = zstandard.ZstdCompressor(level=3, write_content_size=False)
    ends = []
    with open(path, "wb") as handle:
        for start in range(0, len(text), frame_bytes):
            frame = compressor.compress(text[start : start + frame_bytes])
            handle.write(struct.pack("<III", SKIPPABLE_MAGIC, 4, len(frame)))
            handle.write(frame)
            ends.append(handle.tell())
    return ends


def lines_text(lines: list[bytes]) -> bytes:
    return b"".join(line + b"\n" for line in lines)


def _db_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])  # the eval DB drops the move counters


def _db_uci(board: chess.Board, move: chess.Move) -> str:
    """The eval DB writes castling as king-takes-rook (e1h1), like a chess960 engine."""
    if board.is_castling(move):
        rook_file = 7 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 0
        return chess.square_name(move.from_square) + chess.square_name(
            chess.square(rook_file, chess.square_rank(move.from_square))
        )
    return move.uci()


def synthetic_line(board: chess.Board, rng: random.Random) -> bytes:
    """One eval DB line for `board` with 1 to 3 PVs of random legal first moves and White-POV scores."""
    legal = list(board.legal_moves)
    rng.shuffle(legal)
    pvs = []
    for move in legal[: rng.randint(1, 3)]:
        pv: dict = {"line": _db_uci(board, move)}
        if rng.random() < 0.05:
            pv["mate"] = rng.choice([-3, -1, 1, 2, 5])
        else:
            pv["cp"] = rng.randint(-400, 400)
        pvs.append(pv)
    row = {"fen": _db_fen(board), "evals": [{"pvs": pvs, "knodes": 1000, "depth": rng.randint(12, 40)}]}
    return orjson.dumps(row)


def synthetic_lines(n: int, seed: int = 0) -> list[bytes]:
    """`n` lines from random games, including castling positions, with no repeated FEN."""
    rng = random.Random(seed)
    lines: list[bytes] = []
    seen: set[str] = set()
    board = chess.Board()
    while len(lines) < n:
        legal = list(board.legal_moves)
        if not legal or board.ply() > 80:
            board = chess.Board()
            continue
        fen = _db_fen(board)
        if fen not in seen:
            seen.add(fen)
            lines.append(synthetic_line(board, rng))
        board.push(rng.choice(legal))
    return lines


BAD_LINES = [
    b"not json at all",
    b'{"fen": "8/8/8/8/8/8/8/8 w - -", "evals": []}',
    b'{"evals": [{"pvs": [{"cp": 1, "line": "e2e4"}]}]}',
    b'{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", '
    b'"evals": [{"pvs": [{"cp": 10, "line": "e2e5"}], "depth": 20}]}',
]
