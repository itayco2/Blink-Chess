"""The probe: parse N frames of the eval DB and report what the data looks like.

Every line goes through the reference parser (blink.data.parse). One line in `check_every` is also
re-checked independently with python-chess: the decoded best move must be legal and must equal
python-chess's own reading of the raw UCI (which turns king-takes-rook castling into e1g1).
"""

import functools
import json
import re
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import orjson

from blink.board import encode, moves
from blink.board.value import CP_NONE
from blink.data import parse, zst
from blink.data.frames import run_frames
from blink.data.pack import ERROR_SAMPLES, write_atomic
from blink.data.record import NO_MOVE, NUM_ALTERNATIVES, ROOT_DTYPE

DEFAULT_CHECK_EVERY = 16
MOVE_REJECTS = ("illegal_best_move", "bad_uci")
SHALLOW_DEPTH = 18  # the loader drops non-mate roots below this depth (plan P2)
OWN_KING = encode.OWN + chess.KING - 1
CASTLING_MOVES = np.array(
    [moves.FROM_TO.index((chess.E1, chess.G1)), moves.FROM_TO.index((chess.E1, chess.C1))], dtype=np.uint16
)
FEN_FIELDS = re.compile(rb'"fen"\s*:\s*"([^\s"]+) ([wb]) ([^\s"]+)')  # placement, side to move, castling
# What each standard castling right needs on the board; anything else is Chess960 (X-FEN or Shredder).
STANDARD_RIGHTS = {
    "K": ((chess.E1, "K"), (chess.H1, "R")),
    "Q": ((chess.E1, "K"), (chess.A1, "R")),
    "k": ((chess.E8, "k"), (chess.H8, "r")),
    "q": ((chess.E8, "k"), (chess.A8, "r")),
}
COUNT_FIELDS = ("lines", "parsed", "checked", "legal", "canonical", "castling", "mate", "shallow_nonmate")
COUNT_FIELDS += ("white", "black", "alt_dropped", "chess960_castles", "nonstandard_castling")


@dataclass(frozen=True)
class ProbeCounts:
    lines: int = 0
    parsed: int = 0
    checked: int = 0  # lines re-checked with python-chess
    legal: int = 0  # checked lines whose decoded best move is legal
    canonical: int = 0  # checked lines whose best move equals python-chess's parse_uci of the raw move
    castling: int = 0  # parsed best moves that castle
    mate: int = 0
    shallow_nonmate: int = 0
    white: int = 0
    black: int = 0
    alt_dropped: int = 0  # PVs 2..5 whose first move the parser could not use
    chess960_castles: int = 0  # illegal_best_move rejects that are legal Chess960 castles
    nonstandard_castling: int = 0  # parsed lines whose castling rights a standard board drops
    rejects: Mapping[str, int] = field(default_factory=dict)
    errors: Mapping[str, int] = field(default_factory=dict)
    error_samples: tuple[str, ...] = ()
    npv: Mapping[int, int] = field(default_factory=dict)
    depth: Mapping[int, int] = field(default_factory=dict)


def merge(a: ProbeCounts, b: ProbeCounts) -> ProbeCounts:
    """A new ProbeCounts holding both. Neither input changes."""
    sums = {name: getattr(a, name) + getattr(b, name) for name in COUNT_FIELDS}
    maps = {
        name: dict(Counter(getattr(a, name)) + Counter(getattr(b, name)))
        for name in ("rejects", "errors", "npv", "depth")
    }
    samples = (a.error_samples + b.error_samples)[:ERROR_SAMPLES]
    return ProbeCounts(**sums, **maps, error_samples=samples)


def _check(line: bytes, rec: np.void) -> tuple[bool, bool]:
    """(legal, canonical) for one parsed line, decided by python-chess alone."""
    row = orjson.loads(line)
    board = chess.Board(row["fen"] + " 0 1")
    raw = row["evals"][0]["pvs"][0]["line"].split()[0]
    decoded = moves.decode_move(board, int(rec["move"]))
    try:
        canonical = board.parse_uci(raw)
    except ValueError:  # python-chess disagrees that the raw move is legal: counted as a mismatch
        return decoded in board.legal_moves, False
    return decoded in board.legal_moves, decoded == canonical


def _is_chess960_castle(line: bytes) -> bool:
    """True when a rejected best move is a legal Chess960 castle (king takes own rook)."""
    try:
        row = orjson.loads(line)
        board = chess.Board(row["fen"] + " 0 1", chess960=True)
        move = chess.Move.from_uci(row["evals"][0]["pvs"][0]["line"].split()[0])
    except (orjson.JSONDecodeError, KeyError, IndexError, ValueError):
        return False
    return move in board.legal_moves and board.is_castling(move)


def _squares(placement: str) -> dict[int, str]:
    squares = {}
    for rank_from_top, row in enumerate(placement.split("/")):
        file = 0
        for char in row:
            if char.isdigit():
                file += int(char)
            else:
                squares[chess.square(file, 7 - rank_from_top)] = char
                file += 1
    return squares


def _nonstandard_castling(placement: str, rights: str) -> bool:
    """True when a castling right needs a king or rook that is not on its standard square."""
    if rights == "-":
        return False
    squares = _squares(placement)
    for right in rights:
        needs = STANDARD_RIGHTS.get(right)
        if needs is None or any(squares.get(square) != piece for square, piece in needs):
            return True
    return False


def _record_stats(records: np.ndarray) -> dict:
    codes = encode.unpack(records["board"])
    is_mate = records["cp"] == CP_NONE
    expected_alts = np.minimum(records["npv"].astype(np.int64) - 1, NUM_ALTERNATIVES)
    present_alts = (records["alt_move"] != NO_MOVE).sum(axis=1)
    return {
        "castling": int(((codes[:, chess.E1] == OWN_KING) & np.isin(records["move"], CASTLING_MOVES)).sum()),
        "mate": int(is_mate.sum()),
        "shallow_nonmate": int(((records["depth"] < SHALLOW_DEPTH) & ~is_mate).sum()),
        "alt_dropped": int((expected_alts - present_alts).sum()),
        "npv": dict(Counter(records["npv"].tolist())),
        "depth": dict(Counter(records["depth"].tolist())),
    }


def probe_lines(lines: list[bytes], check_every: int = DEFAULT_CHECK_EVERY) -> ProbeCounts:
    """Probe statistics for some lines. Module-level, so spawned workers can run it."""
    recs, rejects, errors, samples = [], Counter(), Counter(), []
    checked = legal = canonical = white = chess960_castles = nonstandard = 0
    for i, line in enumerate(lines):
        try:
            rec = parse.parse_line(line)
        except parse.Rejected as exc:
            rejects[exc.reason] += 1
            chess960_castles += exc.reason == "illegal_best_move" and _is_chess960_castle(line)
            continue
        except Exception as exc:  # noqa: BLE001 - counted and reported; the probe then fails
            errors[type(exc).__name__] += 1
            if len(samples) < ERROR_SAMPLES:
                samples.append(f"{type(exc).__name__}: {exc} | {line[:160].decode('utf-8', 'replace')}")
            continue
        recs.append(rec)
        fen = FEN_FIELDS.search(line)
        if fen:
            white += fen.group(2) == b"w"
            nonstandard += _nonstandard_castling(fen.group(1).decode("ascii"), fen.group(3).decode("ascii"))
        if i % check_every == 0:
            ok_legal, ok_canonical = _check(line, rec)
            checked, legal, canonical = checked + 1, legal + ok_legal, canonical + ok_canonical
    stats = _record_stats(np.array(recs, dtype=ROOT_DTYPE)) if recs else {}
    return ProbeCounts(
        lines=len(lines),
        parsed=len(recs),
        checked=checked,
        legal=legal,
        canonical=canonical,
        white=white,
        black=len(recs) - white,
        chess960_castles=chess960_castles,
        nonstandard_castling=nonstandard,
        rejects=dict(rejects),
        errors=dict(errors),
        error_samples=tuple(samples),
        **stats,
    )


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def build_report(counts: ProbeCounts, source: Path, reader: zst.FrameReader, timing: dict) -> dict:
    move_rejects = sum(counts.rejects.get(reason, 0) for reason in MOVE_REJECTS)
    reached_move = counts.parsed + move_rejects  # lines whose best move was looked at
    lines = counts.lines
    return {
        "source": {"path": str(source), "bytes": Path(source).stat().st_size},
        "frames": reader.frames_read,
        "end": reader.end,
        "lines": lines,
        "parsed": counts.parsed,
        "rejects": dict(sorted(counts.rejects.items())),
        "reject_share": _share(lines - counts.parsed, lines),
        "illegal_best_moves_that_are_chess960_castles": counts.chess960_castles,
        "parsed_with_chess960_castling_rights": counts.nonstandard_castling,
        "errors": dict(sorted(counts.errors.items())),
        "error_samples": list(counts.error_samples),
        "best_move_legal_pct": _pct(counts.parsed, reached_move),
        "sample_check": {
            "checked": counts.checked,
            "legal_pct": _pct(counts.legal, counts.checked),
            "canonical_uci_pct": _pct(counts.canonical, counts.checked),
        },
        "castling_best_move_pct": _pct(counts.castling, counts.parsed),
        "alt_moves_dropped": counts.alt_dropped,
        "mate_share": _share(counts.mate, counts.parsed),
        "shallow_nonmate_share": _share(counts.shallow_nonmate, counts.parsed),
        "side_to_move": {
            "white": _share(counts.white, counts.parsed),
            "black": _share(counts.black, counts.parsed),
        },
        "npv_hist": {str(k): v for k, v in sorted(counts.npv.items())},
        "depth_hist": {str(k): v for k, v in sorted(counts.depth.items())},
        "throughput": timing,
    }


def probe(source: Path, frames: int | None, workers: int = 1, check_every: int = DEFAULT_CHECK_EVERY) -> dict:
    reader = zst.FrameReader(source, frames)
    work = functools.partial(probe_lines, check_every=check_every)
    counts = ProbeCounts()
    decode_s = parse_s = decompressed = 0.0
    start = time.perf_counter()
    for out in run_frames(reader, work, workers):
        counts = merge(counts, out.result)
        decode_s += out.decode_s
        parse_s += out.work_s
        decompressed += out.decompressed
    wall = time.perf_counter() - start
    timing = {
        "workers": workers,
        "wall_s": wall,
        "lines_per_s": counts.lines / wall if wall else 0.0,
        "decode_lines_per_worker_s": counts.lines / decode_s if decode_s else 0.0,
        "parse_lines_per_worker_s": counts.lines / parse_s if parse_s else 0.0,
        "decode_mb_per_worker_s": decompressed / 1e6 / decode_s if decode_s else 0.0,
        "decompressed_mb": decompressed / 1e6,
    }
    return build_report(counts, source, reader, timing)


def write_report(report: dict, out: Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_atomic(Path(out), json.dumps(report, indent=1).encode("utf-8"))


def summary(report: dict) -> str:
    t = report["throughput"]
    check = report["sample_check"]
    side = report["side_to_move"]
    return "\n".join(
        [
            f"source      {report['source']['path']} ({report['frames']} frames, end: {report['end']})",
            f"lines       {report['lines']:,} parsed {report['parsed']:,} rejects {report['rejects']}",
            f"errors      {report['errors'] or 'none'}",
            f"chess960    {report['illegal_best_moves_that_are_chess960_castles']:,} rejected best moves are "
            f"Chess960 castles; {report['parsed_with_chess960_castling_rights']:,} parsed lines have "
            f"castling rights a standard board drops",
            f"legal       best move {report['best_move_legal_pct']:.4f}%; sample of {check['checked']:,}: "
            f"legal {check['legal_pct']:.3f}%, canonical uci {check['canonical_uci_pct']:.3f}%",
            f"castling    {report['castling_best_move_pct']:.3f}% of best moves",
            f"mate        {report['mate_share']:.4f}  shallow non-mate (<{SHALLOW_DEPTH}) "
            f"{report['shallow_nonmate_share']:.4f}",
            f"to move     white {side['white']:.4f} black {side['black']:.4f}",
            f"npv         {report['npv_hist']}",
            f"throughput  {t['lines_per_s']:,.0f} lines/s wall over {t['wall_s']:.1f} s with {t['workers']} "
            f"workers; per worker: decode {t['decode_lines_per_worker_s']:,.0f} lines/s, "
            f"parse {t['parse_lines_per_worker_s']:,.0f} lines/s",
        ]
    )
