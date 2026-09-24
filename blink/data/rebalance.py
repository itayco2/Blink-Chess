"""Rebalancing: per-sample weights that move the eval DB's position mix toward real games.

The eval DB holds what people chose to analyse, which over-represents some phases (openings, wild
tactical positions) and under-represents others. Each record falls in one of 48 buckets:
    piece-count band (6) x any castling right (2) x |W - 0.5| band (4),
with W the Lichess win probability of the record's own label. The weight of a bucket is
p_games / p_evaldb, where p_games comes from the de-duplicated positions of %eval-annotated games in the
2026-08 prefix (held-out games excluded by Site) and p_evaldb from the packed train roots. Weights are
clipped to [0.2, 5] and renormalised (mean 1 under p_evaldb), repeated until stable.
"""

import io
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import chess
import chess.engine
import chess.pgn
import numpy as np

from blink.board import encode
from blink.board.value import CP_NONE, win_probability_array
from blink.data import children, frames, zst
from blink.data.blocklist import iter_game_texts
from blink.data.record import CHILD_DTYPE

PIECE_BAND_EDGES = (7, 13, 19, 25, 29)  # pieces incl. kings and pawns: 2-6, 7-12, 13-18, 19-24, 25-28, 29-32
W_BAND_EDGES = (0.05, 0.15, 0.30)  # |W - 0.5|: level, slight edge, clear edge, decisive (incl. mates)
PIECE_BANDS = len(PIECE_BAND_EDGES) + 1
W_BANDS = len(W_BAND_EDGES) + 1
NUM_BUCKETS = PIECE_BANDS * 2 * W_BANDS
CLIP_LOW, CLIP_HIGH = 0.2, 5.0
MEAN_TOLERANCE = 0.02
MAX_ROUNDS = 100
GAMES_PER_TASK = 200
INT16_MAX = 32767
DEFINITION = (
    "bucket = piece_band * 8 + any_castling * 4 + w_band; piece_band = digitize(pieces incl. kings and "
    f"pawns, {list(PIECE_BAND_EDGES)}); any_castling = a castling rook code (13 or 14) on the board; "
    f"w_band = digitize(|W - 0.5|, {list(W_BAND_EDGES)}) with W = win_probability_array(cp, mate) of the "
    "record's own label; weight = p_games / p_evaldb clipped to [0.2, 5] and renormalised to mean 1 under "
    "p_evaldb until stable; p_games = de-duplicated positions of %eval games in the 2026-08 prefix minus "
    "held-out games; p_evaldb = packed train roots"
)
_SITE = re.compile(r'^\[Site "([^"]*)"\]', re.MULTILINE)
_NON_STANDARD = re.compile(r'^\[Variant "(?!Standard")', re.MULTILINE)


def bucket_of(records: np.ndarray) -> np.ndarray:
    """The rebalancing bucket (0..47) of each ROOT_DTYPE or CHILD_DTYPE record."""
    codes = encode.unpack(records["board"])
    pieces = ((codes != encode.EMPTY) & (codes != encode.EP_SQUARE)).sum(axis=1)
    castling = ((codes == encode.OWN_CASTLING_ROOK) | (codes == encode.OPP_CASTLING_ROOK)).any(axis=1)
    edge = np.abs(win_probability_array(records["cp"], records["mate"]) - 0.5)
    piece_band = np.digitize(pieces, PIECE_BAND_EDGES)
    return (piece_band * 2 + castling) * W_BANDS + np.digitize(edge, W_BAND_EDGES)


def weights_for(records: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each record's weight from a 48-entry table (manifest["rebalance"]["weights"])."""
    table_ = np.asarray(weights, dtype=np.float32)
    if table_.shape != (NUM_BUCKETS,):
        raise ValueError(f"a rebalancing table has {NUM_BUCKETS} weights, got shape {table_.shape}")
    return table_[bucket_of(records)]


def table(games_counts: np.ndarray, evaldb_counts: np.ndarray) -> np.ndarray:
    """p_games / p_evaldb, clipped to [0.2, 5] and renormalised under p_evaldb until it stops moving."""
    games = np.asarray(games_counts, dtype=np.float64)
    evaldb = np.asarray(evaldb_counts, dtype=np.float64)
    if games.shape != (NUM_BUCKETS,) or evaldb.shape != (NUM_BUCKETS,):
        raise ValueError(f"need {NUM_BUCKETS} counts on each side, got {games.shape} and {evaldb.shape}")
    if games.sum() <= 0 or evaldb.sum() <= 0:
        raise ValueError(f"need positive totals: games {games.sum():.0f}, evaldb {evaldb.sum():.0f}")
    p_games, p_evaldb = games / games.sum(), evaldb / evaldb.sum()
    weights = np.divide(p_games, p_evaldb, out=np.ones(NUM_BUCKETS), where=p_evaldb > 0)
    for _ in range(MAX_ROUNDS):
        clipped = np.clip(weights, CLIP_LOW, CLIP_HIGH)
        weights = clipped / float((p_evaldb * clipped).sum())
        if weights.min() >= CLIP_LOW and weights.max() <= CLIP_HIGH:
            break
    weights = np.clip(weights, CLIP_LOW, CLIP_HIGH)
    mean = float((p_evaldb * weights).sum())
    if abs(mean - 1) > MEAN_TOLERANCE:
        raise ValueError(f"rebalancing did not settle: mean weight {mean:.4f} after {MAX_ROUNDS} rounds")
    return weights


def heldout_sites(pgn: Path) -> set[str]:
    """The game ids (last part of the Site URL) of a PGN file."""
    text = Path(pgn).read_text(encoding="utf-8")
    return {site.rsplit("/", 1)[-1] for site in _SITE.findall(text)}


def _text_lines(source: Path, reader: zst.FrameReader) -> Iterator[str]:
    """The complete lines of a pzstd text file; a partial last line of a cut download is dropped."""
    carry = b""
    for frame in reader:
        pieces = (carry + zst.decompress_frame(frame.data)).split(b"\n")
        carry = pieces.pop()
        for piece in pieces:
            yield piece.decode("utf-8", "replace") + "\n"


def _score(pov: chess.engine.PovScore, turn: chess.Color) -> tuple[int, int]:
    score = pov.pov(turn)
    mate = score.mate()
    if mate is not None:
        return CP_NONE, max(-127, min(127, mate))
    return max(-INT16_MAX, min(INT16_MAX, score.score())), 0


def positions_of_games(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(fen_hash, bucket) of every evaluated position of these PGN games. Runs in spawned workers."""
    boards, scores = [], []
    for text in texts:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None or game.errors:
            continue
        board = game.board()
        for node in game.mainline():
            board.push(node.move)
            pov = node.eval()
            if pov is not None:
                boards.append(encode.pack(encode.encode_board(board)))
                scores.append(_score(pov, board.turn))
    recs = np.zeros(len(boards), dtype=CHILD_DTYPE)
    if boards:
        recs["board"] = np.stack(boards)
        recs["cp"], recs["mate"] = np.array(scores, dtype=np.int32).T
    return children.hash_boards(recs["board"]), bucket_of(recs).astype(np.uint8)


class GamesHistogram(NamedTuple):
    counts: np.ndarray  # [48] unique evaluated positions per bucket
    games_seen: int
    games_with_eval: int  # used: evaluated, standard, not held out
    heldout_skipped: int
    positions: int  # evaluated positions before de-duplication
    frames: int
    end: str | None

    def as_dict(self) -> dict:
        return {**self._asdict(), "counts": self.counts.tolist(), "unique_positions": int(self.counts.sum())}


class _Tally:
    def __init__(self) -> None:
        self.seen = self.used = self.heldout = 0


def _eval_game_batches(
    texts: Iterable[str], heldout: set[str], max_games: int | None, tally: _Tally
) -> Iterator[list[str]]:
    batch: list[str] = []
    for text in texts:
        if max_games is not None and tally.used >= max_games:
            break
        tally.seen += 1
        site = _SITE.search(text)
        if site and site.group(1).rsplit("/", 1)[-1] in heldout:
            tally.heldout += 1
            continue
        if "%eval" not in text or _NON_STANDARD.search(text):
            continue
        tally.used += 1
        batch.append(text)
        if len(batch) >= GAMES_PER_TASK:
            yield batch
            batch = []
    if batch:
        yield batch


def games_histogram(
    source: Path, heldout_sites: set[str], max_games: int | None = None, workers: int = 1
) -> GamesHistogram:
    """p_games counts from a pzstd PGN file (a cut download is fine: it stops at the last whole frame)."""
    reader = zst.FrameReader(Path(source))
    tally = _Tally()
    batches = _eval_game_batches(
        iter_game_texts(_text_lines(source, reader)), heldout_sites, max_games, tally
    )
    hashes, buckets = [], []
    for got_hashes, got_buckets in frames.ordered_map(positions_of_games, batches, workers, 2 * workers):
        hashes.append(got_hashes)
        buckets.append(got_buckets)
    all_hashes = np.concatenate(hashes) if hashes else np.empty(0, dtype=np.uint64)
    all_buckets = np.concatenate(buckets) if buckets else np.empty(0, dtype=np.uint8)
    _, first = np.unique(all_hashes, return_index=True)
    counts = np.bincount(all_buckets[first], minlength=NUM_BUCKETS)
    return GamesHistogram(
        counts, tally.seen, tally.used, tally.heldout, len(all_hashes), reader.frames_read, reader.end
    )
