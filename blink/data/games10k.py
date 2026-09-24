"""games10k: 10,000 unique positions from the held-out games, labelled by Stockfish 19 at 1M nodes.

These positions come from real Lichess games (not from what people chose to analyse), and the blocklist
keeps them out of training, so they are the honest game-distribution test set.
Run: python -m blink.data.games10k [--n 10000] [--nodes 1000000] [--procs 5]
"""

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TextIO

import chess
import chess.engine
import chess.pgn
import numpy as np

from blink import paths
from blink.board import encode, moves, value
from blink.data import blocklist
from blink.data.record import NO_MOVE, ROOT_DTYPE

MAX_PROCS = 5
OUTPUT = "games10k.npy"  # ROOT_DTYPE records under BLINK_HOME/data, where the trainer's checks read it


def default_procs(cpu_count: int | None = os.cpu_count()) -> int:
    """Stockfish processes: one per core minus one kept free, at most 5 (5 on the 12-thread build box)."""
    return max(1, min(MAX_PROCS, (cpu_count or 2) - 1))


DEFAULT_PROCS = default_procs()
STOCKFISH = Path(r"D:\tools\stockfish\stockfish-windows-x86-64-universal.exe")
Label = tuple[str, int | None, int | None, int]  # best uci, cp, mate (side-to-move view), depth


def candidate_positions(fh: TextIO, skip_plies: int = blocklist.SKIP_PLIES) -> list[chess.Board]:
    """Unique positions (by colour-normalised hash) from ply `skip_plies` on, excluding finished games."""
    seen: set[int] = set()
    out = []
    for text in blocklist.iter_game_texts(fh):
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        board = game.board()
        for ply, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)
            if ply < skip_plies or board.is_game_over():
                continue
            h = encode.position_hash(board)
            if h not in seen:
                seen.add(h)
                out.append(board.copy(stack=False))
    return out


def sample(positions: Sequence[chess.Board], n: int) -> list[chess.Board]:
    """The first n positions ordered by a hash of their FEN: deterministic and spread over all games."""
    keyed = sorted(positions, key=lambda b: hashlib.blake2b(b.fen().encode(), digest_size=8).digest())
    return keyed[:n]


def to_record(board: chess.Board, best: chess.Move, cp: int | None, mate: int | None, depth: int) -> np.void:
    rec = np.zeros((), dtype=ROOT_DTYPE)
    rec["board"] = encode.pack(encode.encode_board(board))
    rec["move"] = moves.encode_move(board, best)
    rec["cp"], rec["mate"] = (value.CP_NONE, mate) if mate is not None else (cp, 0)
    rec["depth"] = min(255, depth)
    rec["npv"] = 1
    rec["alt_move"] = NO_MOVE
    rec["fen_hash"] = encode.key_hash(bytes(rec["board"]))
    return rec


def label_with_stockfish(fens: Sequence[str], engine_path: Path, nodes: int) -> list[Label]:
    """Label positions in one Stockfish process (Threads=1, Hash=64) at a fixed node budget."""
    out: list[Label] = []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        for fen in fens:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
            score = info["score"].pov(board.turn)
            best = info["pv"][0].uci()
            out.append((best, score.score(), score.mate(), int(info.get("depth", 0))))
    return out


def _label_chunk(args: tuple[list[str], str, int]) -> list[tuple[str, Label]]:
    fens, engine, nodes = args
    return list(zip(fens, label_with_stockfish(fens, Path(engine), nodes), strict=True))


def remaining(fens: Iterable[str], done: dict[str, Label]) -> list[str]:
    return [f for f in fens if f not in done]


def default_path() -> Path:
    """BLINK_HOME/data/games10k.npy: where `run` writes the set, and where training looks for it."""
    return paths.home() / "data" / OUTPUT


def run(n: int, nodes: int, procs: int, home: Path) -> dict:
    evaldir, data = home / "eval", home / "data"
    with open(evaldir / "heldout_games.pgn", encoding="utf-8") as fh:
        chosen = sample(candidate_positions(fh), n)
    fens = [b.fen() for b in chosen]
    partial = data / "games10k.partial.json"
    done: dict[str, Label] = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
    todo = remaining(fens, done)
    chunks = [(todo[i : i + 50], str(STOCKFISH), nodes) for i in range(0, len(todo), 50)]
    started = time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        for batch in pool.imap_unordered(_label_chunk, chunks):
            done.update({fen: tuple(lbl) for fen, lbl in batch})
            tmp = partial.with_name(partial.name + ".tmp")
            tmp.write_text(json.dumps(done), encoding="utf-8")
            os.replace(tmp, partial)
            print(
                f"games10k: {len(done):,}/{len(fens):,} labelled, {time.time() - started:.0f} s", flush=True
            )
    records = np.stack(
        [
            to_record(chess.Board(f), chess.Move.from_uci(done[f][0]), done[f][1], done[f][2], done[f][3])
            for f in fens
        ]
    )
    np.save(data / OUTPUT, records)
    (data / "games10k_fens.txt").write_text("\n".join(fens) + "\n", encoding="utf-8")
    return {
        "positions": len(fens),
        "nodes": nodes,
        "procs": procs,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    args = parser.parse_args()
    print(json.dumps(run(args.n, args.nodes, args.procs, paths.home()), indent=2))


if __name__ == "__main__":
    main()
