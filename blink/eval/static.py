"""E2, the static metrics (plan P8 and section 6, table 2), computed offline from the network alone.

Per root position, with Stockfish's stored PVs as the reference (up to 5, from the eval DB or games10k):
- policy top-1/3/5: the rank of SF's PV1 move among the legal moves by policy logit;
- legal mass: the softmax mass the policy puts on legal moves before masking;
- VAA: value mode's pick (1 - W of each child, a checkmating child first, a stalemate or insufficient-
  material child at 0.5, lowest vocabulary index on an exact tie) equals PV1 or an alternative with the
  identical score. Ties are counted as correct, and said so;
- near-best: the pick is one of the stored PVs within 5 win% points of PV1 (a lower bound: a move outside
  the stored PVs never counts);
- Kendall tau-b on SCORES: between the model's scores of the stored PV moves (policy logits, or value
  mode's 1 - W) and SF's win% of the same moves, per root with at least 2 PVs, averaged over roots. It is
  computed on the scores themselves, never on argsorts (DeepMind's formula ranks argsorts; PF37);
- Brier and ECE of the value head's W against SF's win% of the root, before and after one temperature
  fitted on val (it rescales the 128-bin distribution's log-probabilities);
- phase: lichess's Divider rules (scalachess Divider.scala, MIT) applied to the single position, with the
  side to move as White: endgame at <= 6 majors and minors, middlegame at <= 10, a sparse back rank or
  mixedness > 150, opening otherwise.
Also: win% regret against SF19 at 1M nodes on games10k (sflabel), mateset shortest-mate and mate-preserving
rates, puzzle accuracy per rating band with Wilson intervals, and the puzzle-rating equivalent: the
maximum-likelihood rating R with P(solve | puzzle rating r) = 1 / (1 + 10^((r - R) / 400)), with a
1,000-sample bootstrap interval. It is a puzzle-rating equivalent, never an Elo.
"""

import csv
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import chess
import numpy as np

from blink.board import encode, moves, value
from blink.data.children import codes_to_board
from blink.data.record import NO_MOVE, ROOT_DTYPE
from blink.eval import puzzles
from blink.eval.sflabel import SfLabeler
from blink.play.evaluator import Evaluator

CHUNK_ROWS = 4096
MATE_VALUE = 2.0
DRAW_VALUE = 0.5
NEAR_BEST_WIN = 0.05
ECE_BINS = 15
BOOTSTRAP = 1000
SEED = 0
PHASES = ("opening", "middlegame", "endgame")
LICHESS_BANDS = tuple(range(400, 2800, 200))


# ------------------------------------------------------------------------------ small statistics


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b between two score vectors (ties handled); nan when either is constant."""
    concordant = discordant = ties_x = ties_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else math.nan


def brier(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((pred - label) ** 2))


def ece(pred: np.ndarray, label: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error of predicted win% against SF win%, equal-width bins."""
    index = np.minimum(bins - 1, (pred * bins).astype(np.int64))
    gap = np.bincount(index, pred - label, minlength=bins)
    return float(np.abs(gap).sum() / max(1, len(pred)))


def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float], samples: int = BOOTSTRAP):
    """(low, high) of a 95% percentile bootstrap over the rows of `values` (fixed seed)."""
    rng = np.random.default_rng(SEED)
    n = len(values)
    if n == 0:
        return None
    draws = [stat(values[rng.integers(0, n, n)]) for _ in range(samples)]
    finite = np.array([d for d in draws if math.isfinite(d)])
    if len(finite) == 0:
        return None
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def scale_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.maximum(probs, 1e-12)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=-1, keepdims=True)


def fit_temperature(probs: np.ndarray, label_win: np.ndarray) -> float:
    """The T minimising the mean negative log-likelihood of the label's bin, by grid then golden section."""
    bins = np.minimum(value.NUM_BINS - 1, (label_win * value.NUM_BINS).astype(np.int64))

    def nll(log_t: float) -> float:
        scaled = scale_temperature(probs, math.exp(log_t))
        return float(-np.mean(np.log(np.maximum(scaled[np.arange(len(bins)), bins], 1e-12))))

    grid = np.linspace(-2.0, 2.0, 41)
    best = grid[int(np.argmin([nll(g) for g in grid]))]
    lo, hi = best - 0.1, best + 0.1
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        lo, hi = (lo, b) if nll(a) < nll(b) else (a, hi)
    return math.exp((lo + hi) / 2)


# ------------------------------------------------------------------------------ the Divider phase


def _mixedness_score(y: int, white: int, black: int) -> int:
    table = {
        (0, 1): 1 + y,
        (0, 2): 2 + (6 - y) if y < 6 else 0,
        (0, 3): 3 + (7 - y) if y < 7 else 0,
        (0, 4): 3 + (7 - y) if y < 7 else 0,
        (1, 0): 1 + (8 - y),
        (1, 1): 5 + abs(4 - y),
        (1, 2): 4 + (7 - y),
        (1, 3): 5 + (7 - y),
        (2, 0): 2 + (y - 2) if y > 2 else 0,
        (2, 1): 4 + (y - 1),
        (2, 2): 7,
        (3, 0): 3 + (y - 1) if y > 1 else 0,
        (3, 1): 5 + (y - 1),
        (4, 0): 3 + (y - 1) if y > 1 else 0,
    }
    return table.get((white, black), 0)


def mixedness(board: chess.Board) -> int:
    white, black = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    total = 0
    for i in range(49):
        region = 0x0303 << ((i % 7) + 8 * (i // 7))
        total += _mixedness_score(i // 7 + 1, chess.popcount(white & region), chess.popcount(black & region))
    return total


def phase(board: chess.Board) -> str:
    majors_minors = chess.popcount(board.occupied & ~(board.kings | board.pawns))
    if majors_minors <= 6:
        return "endgame"
    sparse = (
        chess.popcount(chess.BB_RANK_1 & board.occupied_co[chess.WHITE]) < 4
        or chess.popcount(chess.BB_RANK_8 & board.occupied_co[chess.BLACK]) < 4
    )
    if majors_minors <= 10 or sparse or mixedness(board) > 150:
        return "middlegame"
    return "opening"


# ------------------------------------------------------------------------------ roots and their PVs


@dataclass(frozen=True)
class Root:
    board: chess.Board  # the real position, or its colour-normalised twin (White to move)
    codes: np.ndarray  # uint8 [64]
    label: int  # SF PV1 move index
    pv_moves: tuple[int, ...]  # PV1 and the stored alternatives
    pv_win: tuple[float, ...]  # their win% for the side to move
    ties: frozenset[int]  # PV1 and every alternative with PV1's identical score


def _win(cp: int, mate: int) -> float:
    return float(value.win_probability_array(np.array([cp]), np.array([mate]))[0])


def root_from_record(record: np.void, board: chess.Board | None = None) -> Root:
    codes = encode.unpack(record["board"])
    pv, win, ties = [int(record["move"])], [_win(record["cp"], record["mate"])], {int(record["move"])}
    for slot in range(len(record["alt_move"])):
        move = int(record["alt_move"][slot])
        if move == NO_MOVE:
            continue
        cp, mate = int(record["alt_cp"][slot]), int(record["alt_mate"][slot])
        pv.append(move)
        win.append(_win(cp, mate))
        if cp == int(record["cp"]) and mate == int(record["mate"]):
            ties.add(move)
    return Root(
        board or codes_to_board(codes), codes, int(record["move"]), tuple(pv), tuple(win), frozenset(ties)
    )


def roots_from_records(records: np.ndarray, fens: Sequence[str] | None = None) -> list[Root]:
    boards = [chess.Board(f) for f in fens] if fens is not None else [None] * len(records)
    return [root_from_record(r, b) for r, b in zip(records, boards, strict=True)]


# ------------------------------------------------------------------------------ one pass of the network


@dataclass(frozen=True)
class RootOutcome:
    label_rank: int  # 0 = the policy's top legal move is PV1
    policy_pick: int
    legal_mass: float
    win: float  # the value head's W for the root
    win_scaled: float  # the same after the temperature
    label_win: float
    tau_policy: float
    near_best_policy: bool
    phase: str
    value_pick: int | None = None
    vaa: bool | None = None
    tau_value: float | None = None
    near_best_value: bool | None = None


def _near_best(root: Root, pick: int) -> bool:
    if pick not in root.pv_moves:
        return False
    return root.pv_win[root.pv_moves.index(pick)] >= root.pv_win[0] - NEAR_BEST_WIN


def _policy_outcome(root: Root, logits: np.ndarray, probs: np.ndarray, temperature: float) -> RootOutcome:
    legal = np.array(sorted(moves.encode_move(root.board, m) for m in root.board.legal_moves))
    order = legal[np.argsort(-logits[legal], kind="stable")]
    soft = np.exp(logits - logits.max())
    tau = (
        kendall_tau_b([logits[m] for m in root.pv_moves], root.pv_win) if len(root.pv_moves) > 1 else math.nan
    )
    return RootOutcome(
        label_rank=int(np.flatnonzero(order == root.label)[0]) if root.label in order else len(order),
        policy_pick=int(order[0]),
        legal_mass=float(soft[legal].sum() / soft.sum()),
        win=float(probs @ value.BIN_CENTERS),
        win_scaled=float(scale_temperature(probs[None], temperature)[0] @ value.BIN_CENTERS),
        label_win=root.pv_win[0],
        tau_policy=tau,
        near_best_policy=_near_best(root, int(order[0])),
        phase=phase(root.board),
    )


def _children(root: Root) -> tuple[list[int], list[np.ndarray], list[int]]:
    """Legal move indices (vocabulary order), child codes and terminal kinds (0, 1 mate, 2 rule draw)."""
    rows = []
    for move in root.board.legal_moves:
        child = root.board.copy(stack=False)
        child.push(move)
        kind = (
            1
            if child.is_checkmate()
            else 2
            if child.is_stalemate() or child.is_insufficient_material()
            else 0
        )
        rows.append((moves.encode_move(root.board, move), encode.encode_board(child), kind))
    rows.sort(key=lambda row: row[0])
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def value_scores(indices: list[int], child_win: np.ndarray, kinds: list[int]) -> np.ndarray:
    """Value mode's move values: 1 - W, a checkmating child first (lowest index), a rule draw at 0.5."""
    scores = 1.0 - np.asarray(child_win, dtype=np.float64)
    for i, kind in enumerate(kinds):
        if kind == 1:
            scores[i] = MATE_VALUE - indices[i] * 1e-6
        elif kind == 2:
            scores[i] = DRAW_VALUE
    return scores


def _with_value(outcome: RootOutcome, root: Root, indices: list[int], scores: np.ndarray) -> RootOutcome:
    pick = indices[int(np.argmax(scores))]
    by_move = dict(zip(indices, scores, strict=True))
    pairs = [(by_move[m], w) for m, w in zip(root.pv_moves, root.pv_win, strict=True) if m in by_move]
    tau = kendall_tau_b(*zip(*pairs, strict=True)) if len(pairs) > 1 else math.nan
    return replace(
        outcome, value_pick=pick, vaa=pick in root.ties, tau_value=tau, near_best_value=_near_best(root, pick)
    )


def evaluate_roots(
    evaluator: Evaluator,
    roots: Sequence[Root],
    temperature: float = 1.0,
    value_limit: int | None = 0,
    chunk_rows: int = CHUNK_ROWS,
) -> list[RootOutcome]:
    """Policy and value-head outcomes for every root; value mode for the first `value_limit` (None: all)."""
    outcomes: list[RootOutcome] = []
    for start in range(0, len(roots), chunk_rows):
        part = roots[start : start + chunk_rows]
        evaluation = evaluator.evaluate(np.stack([r.codes for r in part]))
        for j, root in enumerate(part):
            outcomes.append(
                _policy_outcome(root, evaluation.policy_logits[j], evaluation.value_probs[j], temperature)
            )
    limit = len(roots) if value_limit is None else min(value_limit, len(roots))
    return _value_pass(evaluator, roots, outcomes, limit, chunk_rows)


def _value_pass(evaluator, roots, outcomes, limit, chunk_rows) -> list[RootOutcome]:
    pending: list[tuple[int, list[int], list[np.ndarray], list[int]]] = []
    rows = 0

    def flush() -> None:
        if not pending:
            return
        win = evaluator.evaluate(np.stack([c for _, _, codes, _ in pending for c in codes])).win_probability()
        offset = 0
        for i, indices, codes, kinds in pending:
            scores = value_scores(indices, win[offset : offset + len(codes)], kinds)
            outcomes[i] = _with_value(outcomes[i], roots[i], indices, scores)
            offset += len(codes)
        pending.clear()

    for i in range(limit):
        indices, codes, kinds = _children(roots[i])
        if rows + len(codes) > chunk_rows:
            flush()
            rows = 0
        pending.append((i, indices, codes, kinds))
        rows += len(codes)
    flush()
    return outcomes


# ------------------------------------------------------------------------------ aggregation


def _rate(hits: Iterable[bool]) -> dict:
    hits = list(hits)
    n, k = len(hits), sum(hits)
    low, high = puzzles.wilson(k, n)
    return {"value": k / n if n else None, "n": n, "wilson95": [low, high]}


def _mean_ci(values: Iterable[float]) -> dict:
    data = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if len(data) == 0:
        return {"value": None, "n": 0, "ci95": None}
    return {"value": float(data.mean()), "n": len(data), "ci95": bootstrap_ci(data, np.mean)}


def _calibration(outs: Sequence[RootOutcome]) -> dict:
    pairs = np.array([(o.win, o.win_scaled, o.label_win) for o in outs], dtype=np.float64)
    if len(pairs) == 0:
        return {}
    pred, scaled, label = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    return {
        "brier": {
            "value": brier(pred, label),
            "n": len(pred),
            "ci95": bootstrap_ci(pairs, lambda s: brier(s[:, 0], s[:, 2])),
        },
        "ece_before": {
            "value": ece(pred, label),
            "ci95": bootstrap_ci(pairs, lambda s: ece(s[:, 0], s[:, 2])),
        },
        "ece_after": {
            "value": ece(scaled, label),
            "ci95": bootstrap_ci(pairs, lambda s: ece(s[:, 1], s[:, 2])),
        },
    }


def summarize(outs: Sequence[RootOutcome]) -> dict:
    valued = [o for o in outs if o.value_pick is not None]
    out = {
        "roots": len(outs),
        "value_roots": len(valued),
        **{f"top{k}": _rate(o.label_rank < k for o in outs) for k in (1, 3, 5)},
        "legal_mass": _mean_ci(o.legal_mass for o in outs),
        "near_best_policy": _rate(o.near_best_policy for o in outs),
        "tau_policy": _mean_ci(o.tau_policy for o in outs),
        "vaa": _rate(o.vaa for o in valued),
        "near_best_value": _rate(o.near_best_value for o in valued),
        "tau_value": _mean_ci(o.tau_value for o in valued),
        **_calibration(outs),
    }
    out["by_phase"] = {
        name: {
            "top1": _rate(o.label_rank == 0 for o in outs if o.phase == name),
            "vaa": _rate(o.vaa for o in valued if o.phase == name),
        }
        for name in PHASES
    }
    return out


# ------------------------------------------------------------------------------ win% regret on games10k


def regret(roots: Sequence[Root], picks: Sequence[int], labeler: SfLabeler) -> dict:
    """Mean win% given up against SF's best, the pick's win% from SF19 restricted to that move (cached).

    A pick equal to SF's best costs 0 without a search; a restricted search that scores the pick above
    SF's own best (search noise) also counts as 0. The searches run on the labeler's processes."""
    pairs = list(zip(roots, picks, strict=True))
    wanted = [(r.board.fen(), moves.decode_move(r.board, p).uci()) for r, p in pairs if p != r.label]
    labels = iter(labeler.label_many(wanted))
    losses = [
        0.0 if pick == root.label else max(0.0, root.pv_win[0] - next(labels).win) for root, pick in pairs
    ]
    return {**_mean_ci(losses), "searched": labeler.searched}


# ------------------------------------------------------------------------------ the mateset


def mate_rates(
    evaluator: Evaluator, arrays: dict[str, np.ndarray], labeler: SfLabeler | None, limit: int | None = None
) -> dict[str, dict]:
    """Per mode: the pick keeps the shortest mate (child_is_best), and, with SF, still mates at all."""
    n = len(arrays["root_best"]) if limit is None else min(limit, len(arrays["root_best"]))
    out: dict[str, dict] = {}
    picks = {"policy": [], "value": []}
    for i in range(n):
        lo, hi = int(arrays["child_offset"][i]), int(arrays["child_offset"][i + 1])
        indices = [int(m) for m in arrays["child_move"][lo:hi]]
        root_codes = encode.unpack(arrays["root_board"][i])
        rows = np.concatenate([root_codes[None], encode.unpack(arrays["child_board"][lo:hi])])
        evaluation = evaluator.evaluate(rows)
        logits = evaluation.policy_logits[0][indices]
        kinds = [int(k) for k in arrays["child_terminal"][lo:hi]]
        scores = value_scores(indices, evaluation.win_probability()[1:], kinds)
        order = sorted(range(len(indices)), key=lambda j: indices[j])
        policy_pick = max(order, key=lambda j: (logits[j], -indices[j]))
        value_pick = max(order, key=lambda j: (scores[j], -indices[j]))
        for mode, j in (("policy", policy_pick), ("value", value_pick)):
            picks[mode].append((i, j, lo))
    for mode, chosen in picks.items():
        shortest = [bool(arrays["child_is_best"][lo + j]) for _, j, lo in chosen]
        out[mode] = {"shortest": _rate(shortest), "preserving": None}
        if labeler is not None:
            keeps = [
                s or _still_mates(arrays, i, lo + j, labeler)
                for (i, j, lo), s in zip(chosen, shortest, strict=True)
            ]
            out[mode]["preserving"] = _rate(keeps)
    return out


def _still_mates(arrays: dict[str, np.ndarray], root: int, child: int, labeler: SfLabeler) -> bool:
    board = codes_to_board(encode.unpack(arrays["root_board"][root]))
    move = moves.decode_move(board, int(arrays["child_move"][child]))
    label = labeler.label(board.fen(), move)
    return label.mate is not None and label.mate > 0


# ------------------------------------------------------------------------------ puzzles by band


def band_accuracy(rows: Iterable[dict], band: Callable[[int], str]) -> dict[str, dict]:
    """Accuracy with a Wilson 95% interval for each rating band, from per-puzzle rows (rating, correct)."""
    groups: dict[str, list[bool]] = {}
    for row in rows:
        groups.setdefault(band(int(row["rating"])), []).append(bool(int(row["correct"])))
    return {name: _rate(hits) for name, hits in groups.items()}


def lichess_band(rating: int) -> str:
    low = max(b for b in LICHESS_BANDS if b <= rating) if rating >= LICHESS_BANDS[0] else LICHESS_BANDS[0]
    return f"{low}-{low + 200}"


def _loglik_slope(ratings: np.ndarray, correct: np.ndarray, r: float) -> float:
    p = 1 / (1 + 10 ** ((ratings - r) / 400))
    return float(np.sum(correct - p))


def puzzle_rating_mle(ratings: np.ndarray, correct: np.ndarray) -> float | None:
    """The R maximising sum log P(outcome | puzzle rating); None when all solved or none (unbounded)."""
    if correct.all() or not correct.any():
        return None
    lo, hi = float(ratings.min()) - 2000, float(ratings.max()) + 2000
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _loglik_slope(ratings, correct, mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def puzzle_rating_equivalent(rows: Sequence[dict], samples: int = BOOTSTRAP) -> dict:
    """The puzzle-rating equivalent (never an Elo) with a percentile bootstrap interval."""
    data = np.array([(int(r["rating"]), int(r["correct"])) for r in rows], dtype=np.float64)
    if len(data) == 0:
        return {"value": None, "n": 0, "ci95": None}
    estimate = puzzle_rating_mle(data[:, 0], data[:, 1] > 0)

    def stat(sample: np.ndarray) -> float:
        found = puzzle_rating_mle(sample[:, 0], sample[:, 1] > 0)
        return math.nan if found is None else found

    return {"value": estimate, "n": len(data), "ci95": bootstrap_ci(data, stat, samples)}


def read_puzzle_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


LICHESS_COLUMNS = ("PuzzleId", "FEN", "Moves", "Rating")


def read_lichess_puzzles(path: Path) -> list[dict]:
    """Rows of a Lichess puzzle CSV (the FEN is before the opponent's first move), refusing other files."""
    rows = read_puzzle_rows(path)
    missing = [c for c in LICHESS_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    return rows


def score_lichess_puzzles(rows: Iterable[dict], agent, game_prefix: str = "lb") -> list[dict]:
    """DeepMind's scorer on Lichess puzzle rows (FEN before the opponent's move, then Moves)."""
    out = []
    for row in rows:
        engine = puzzles.AgentEngine(agent, game=f"{game_prefix}-{row['PuzzleId']}")
        solved = puzzles.evaluate_puzzle_from_board(chess.Board(row["FEN"]), row["Moves"].split(), engine)
        out.append(
            {
                "puzzle_id": row["PuzzleId"],
                "rating": int(row["Rating"]),
                "correct": int(solved),
                "illegal": engine.illegal,
            }
        )
    return out


def first_per_band(rows: Iterable[dict], per_band: int) -> list[dict]:
    """The first `per_band` puzzles of each 200-point band, in file order."""
    seen: dict[str, int] = {}
    out = []
    for row in rows:
        name = lichess_band(int(row["Rating"]))
        if seen.get(name, 0) < per_band:
            seen[name] = seen.get(name, 0) + 1
            out.append(row)
    return out


# ------------------------------------------------------------------------------ E2, end to end


@dataclass(frozen=True)
class StaticInputs:
    test_iid: Path  # root records (.bin)
    val: Path | None = None  # root records for the temperature fit
    test_grouped: Path | None = None
    games10k: Path | None = None  # games10k.npy; its FENs in games10k_fens.txt beside it
    mateset: Path | None = None  # mateset.npz
    dm_puzzles: tuple[tuple[str, Path], ...] = ()  # (mode, per-puzzle CSV of `blink eval puzzles`)
    lichess_bands: Path | None = None  # lichess_bands.csv (PuzzleId, FEN, Moves, Rating, ...)


@dataclass(frozen=True)
class StaticLimits:
    test_roots: int = 1_000_000
    value_roots: int = 200_000
    val_roots: int = 50_000
    games10k: int = 10_000
    mateset: int = 2_000
    band_puzzles: int = 500  # per 200-point band


FULL_LIMITS = StaticLimits()


def read_roots(path: Path, limit: int) -> np.ndarray:
    """The first `limit` root records of a pack file (a sequential read from the front)."""
    return np.fromfile(path, dtype=ROOT_DTYPE, count=limit)


def _temperature(evaluator: Evaluator, inputs: StaticInputs, limits: StaticLimits) -> float:
    if inputs.val is None:
        return 1.0
    roots = roots_from_records(read_roots(inputs.val, limits.val_roots))
    probs, wins = [], []
    for start in range(0, len(roots), CHUNK_ROWS):
        part = roots[start : start + CHUNK_ROWS]
        probs.append(evaluator.evaluate(np.stack([r.codes for r in part])).value_probs)
        wins += [r.pv_win[0] for r in part]
    return fit_temperature(np.concatenate(probs), np.array(wins)) if roots else 1.0


def _root_set(evaluator, path: Path | None, limits: StaticLimits, temperature: float) -> dict | None:
    if path is None:
        return None
    roots = roots_from_records(read_roots(path, limits.test_roots))
    return summarize(evaluate_roots(evaluator, roots, temperature, limits.value_roots))


def _games10k(evaluator, inputs: StaticInputs, limits: StaticLimits, labeler, temperature) -> dict | None:
    if inputs.games10k is None:
        return None
    records = np.load(inputs.games10k)[: limits.games10k]
    fens = inputs.games10k.with_name("games10k_fens.txt").read_text(encoding="utf-8").split("\n")
    roots = roots_from_records(records, fens[: len(records)])
    outs = evaluate_roots(evaluator, roots, temperature, value_limit=None)
    out = {"summary": summarize(outs)}
    if labeler is not None:
        out["regret"] = {
            "policy": regret(roots, [o.policy_pick for o in outs], labeler),
            "value": regret(roots, [o.value_pick for o in outs], labeler),
        }
    return out


def _puzzles(inputs: StaticInputs, limits: StaticLimits, agents: dict) -> dict:
    out: dict[str, dict] = {}
    for mode, path in inputs.dm_puzzles:
        rows = read_puzzle_rows(path)
        out.setdefault("dm10k", {})[mode] = {
            "bands": band_accuracy(rows, puzzles.band_of),
            "puzzle_rating_equivalent": puzzle_rating_equivalent(rows),
        }
    if inputs.lichess_bands is not None:
        chosen = first_per_band(read_lichess_puzzles(inputs.lichess_bands), limits.band_puzzles)
        for mode, agent in agents.items():
            rows = score_lichess_puzzles(chosen, agent)
            out.setdefault("lichess_bands", {})[mode] = {
                "bands": band_accuracy(rows, lichess_band),
                "puzzle_rating_equivalent": puzzle_rating_equivalent(rows),
                "illegal_moves": sum(r["illegal"] for r in rows),
            }
    return out


def run_e2(
    evaluator: Evaluator,
    agents: dict,
    inputs: StaticInputs,
    limits: StaticLimits = FULL_LIMITS,
    labeler: SfLabeler | None = None,
) -> dict:
    """Every E2 number for one model: the dict behind its two DiagnosticsRows (and its intervals)."""
    temperature = _temperature(evaluator, inputs, limits)
    test_iid = _root_set(evaluator, inputs.test_iid, limits, temperature)
    grouped = _root_set(evaluator, inputs.test_grouped, limits, temperature)
    mates = None
    if inputs.mateset is not None:
        with np.load(inputs.mateset) as data:
            mates = mate_rates(evaluator, {k: data[k] for k in data.files}, labeler, limits.mateset)
    return {
        "temperature": temperature,
        "test_iid": test_iid,
        "test_grouped": grouped,
        "grouped_gap": _gap(test_iid, grouped),
        "games10k": _games10k(evaluator, inputs, limits, labeler, temperature),
        "mateset": mates,
        "puzzles": _puzzles(inputs, limits, agents),
        "limits": limits.__dict__,
    }


def _gap(iid: dict | None, grouped: dict | None) -> dict | None:
    """test_iid minus test_grouped, for top-1 and VAA: how much the random split flatters the model."""
    if not iid or not grouped:
        return None
    return {
        key: (iid[key]["value"] - grouped[key]["value"])
        if iid[key]["value"] is not None and grouped[key]["value"] is not None
        else None
        for key in ("top1", "vaa")
    }


def _value(entry: dict | None, *keys: str) -> float | None:
    for key in keys:
        if entry is None:
            return None
        entry = entry.get(key)
    return entry if isinstance(entry, (int, float)) or entry is None else None


def diagnostics_rows(e2: dict, agent: str) -> list:
    """The two DiagnosticsRows (policy, value) that results.json carries for this model.

    The schema never lets a number go out without what its interval is built from: each band's
    percentage carries its puzzle count (band_n), and a puzzle-rating equivalent whose bootstrap gave
    no interval is left out of the row (it stays in e2's own report).
    """
    from blink.report.results_schema import DiagnosticsRow

    iid, gap, games = e2["test_iid"] or {}, e2["grouped_gap"] or {}, e2["games10k"] or {}
    rows = []
    for mode in ("policy", "value"):
        dm = (e2["puzzles"].get("dm10k") or {}).get(mode) or {}
        scored = {name: b for name, b in (dm.get("bands") or {}).items() if b["value"] is not None}
        equiv = dm.get("puzzle_rating_equivalent") or {}
        ci = equiv.get("ci95")
        rows.append(
            DiagnosticsRow(
                agent=agent,
                mode=mode,
                top1=_value(iid, "top1", "value") if mode == "policy" else None,
                top3=_value(iid, "top3", "value") if mode == "policy" else None,
                top5=_value(iid, "top5", "value") if mode == "policy" else None,
                vaa=_value(iid, "vaa", "value") if mode == "value" else None,
                near_best=_value(iid, f"near_best_{mode}", "value"),
                kendall_tau_b=_value(iid, f"tau_{mode}", "value"),
                brier=_value(iid, "brier", "value"),
                ece_before=_value(iid, "ece_before", "value"),
                ece_after=_value(iid, "ece_after", "value"),
                regret_games10k=_value(games.get("regret"), mode, "value"),
                grouped_gap=gap.get("top1" if mode == "policy" else "vaa"),
                band_pct={name: 100 * b["value"] for name, b in scored.items()},
                band_n={name: b["n"] for name, b in scored.items()},
                mate_shortest=_value((e2["mateset"] or {}).get(mode), "shortest", "value"),
                mate_preserving=_value((e2["mateset"] or {}).get(mode), "preserving", "value"),
                puzzle_rating_equiv=equiv.get("value") if ci else None,
                puzzle_rating_ci=tuple(ci) if ci else None,
            )
        )
    return rows
