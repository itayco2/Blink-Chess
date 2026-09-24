"""The public hook and the pre-registered claim, both conditional on the shipped mode (plan section 1).

HOOK_EN and HOOK_HE are module attributes computed from results/results.json: they exist only once the
evaluation has named a shipped mode, so no document can quote a hook the evidence has not chosen.
`fill_claim` writes the pre-registered claim sentence with every blank taken from results/*.json and
refuses, naming each missing value, rather than print a sentence with a gap or a guess in it. The
wording is the plan's, in ASCII (+/- for the plus-minus sign).
"""

from pathlib import Path

from blink.report import results_schema as rs
from blink.report import scoreboard as sb

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
MODES = ("policy", "value")
LANGS = ("en", "he")
KWH_MIN_COVERAGE = 0.99  # the published kWh must cover at least 99% of the published GPU-hours

HOOKS = {
    ("policy", "en"): "Blink: a chess AI that never searches.",
    ("policy", "he"): "בלינק: בינה מלאכותית לשחמט שלא מחפשת אף פעם.",
    ("value", "en"): "Blink: a chess AI that looks exactly one move ahead, never two.",
    ("value", "he"): "בלינק: בינה מלאכותית לשחמט שמסתכלת מהלך אחד קדימה, אף פעם לא שניים.",
}
MODE_CLAUSES = {
    "value": (
        "looks exactly one move ahead, never two: each move is at most one batched forward pass that "
        "scores the position after every legal move once"
    ),
    "policy": "never searches: each move is at most one forward pass",
}
CLAIM = (
    "Blink is a {params}-parameter transformer that {mode_clause}. It never evaluates an opponent's reply, "
    "and it uses no opening book, tablebase or engine at play time. It was trained by supervised learning "
    "on {positions} positions drawn from the 409,710,113-position Lichess evaluation database (CC0). "
    "Training took {flagship_h} GPU-hours for the flagship run ({total_h} for the whole project), with no "
    "cloud GPU and no paid data: one home RTX 3070 ({kwh} GPU-board kWh). Against pinned Stockfish 19 "
    "UCI_Elo anchors it rates {elo} +/- {elo_ci} (95% CI, {games} games{extrapolated}). That is a "
    "CCRL-Blitz-anchored engine scale, not FIDE, and puts it about level with Stockfish 19 at {nodes} nodes "
    "per move. It solves {puzzles}% (Wilson 95% {puzzles_lo} to {puzzles_hi}) of DeepMind's 10K puzzles; no "
    "exact position (or its colour mirror) from any puzzle line, or from its source game after ply 16, was "
    "in training. As a Lichess BOT it is rated {rating} in blitz (RD {rd}, {n} rated games, {humans}% "
    "against humans, snapshot {date}). DeepMind's 9M model, re-measured in the same harness, rates "
    "{dm_elo} +/- {dm_ci}."
)
EXTRAPOLATED = f"; extrapolated below the {sb.ANCHOR_FLOOR} anchor"


class ClaimRefused(ValueError):
    """results/*.json cannot fill every blank of the claim; the message names each missing value."""


def hook(lang: str, mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}: the hook follows the shipped mode")
    if lang not in LANGS:
        raise ValueError(f"language {lang!r} is not one of {LANGS}")
    return HOOKS[(mode, lang)]


def shipped_mode(results_dir: Path | None = None) -> str | None:
    path = Path(results_dir if results_dir is not None else RESULTS_DIR) / "results.json"
    if not path.is_file():
        return None
    shipped = rs.from_json(path.read_text(encoding="utf-8")).shipped
    return shipped.mode if shipped else None


def __getattr__(name: str) -> str:
    if name in ("HOOK_EN", "HOOK_HE"):
        mode = shipped_mode()
        if mode is None:
            where = RESULTS_DIR / "results.json"
            raise AttributeError(f"claims.{name} depends on the shipped mode, and {where} names none yet")
        return hook(name[-2:].lower(), mode)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _dm_row(results: rs.Results) -> rs.StrengthRow | None:
    measured = [r for r in results.strength if r.kind == "reference" and r.elo is not None]
    return measured[0] if len(measured) == 1 else None


def _need(missing: list[str], label: str, value):
    if value is None:
        missing.append(label)
    return value


def _strength_blanks(row: rs.StrengthRow, missing: list[str]) -> dict:
    fields = ("params_total", "positions_seen", "elo", "elo_ci95", "elo_games", "sf_nodes_equiv")
    fields += ("dm_puzzles_pct", "dm_puzzles_ci")
    got = {f: _need(missing, f"shipped strength row {row.agent!r}: {f}", getattr(row, f)) for f in fields}
    if None in got.values():
        return {}
    return {
        "params": f"{got['params_total'] / 1e6:.1f}M",
        "positions": f"{got['positions_seen']:,}",
        "elo": f"{row.elo:.0f}",
        "elo_ci": f"{row.elo_ci95:.0f}",
        "games": f"{row.elo_games:,}",
        "extrapolated": EXTRAPOLATED if row.elo < sb.ANCHOR_FLOOR else "",
        "nodes": f"{row.sf_nodes_equiv:,}",
        "puzzles": f"{row.dm_puzzles_pct:.1f}",
        "puzzles_lo": f"{row.dm_puzzles_ci[0]:.1f}",
        "puzzles_hi": f"{row.dm_puzzles_ci[1]:.1f}",
    }


def _compute_blanks(compute: dict, missing: list[str]) -> dict:
    keys = ("flagship_gpu_hours", "total_gpu_hours", "gpu_board_kwh")
    got = {k: _need(missing, f"compute.json: {k}", compute.get(k)) for k in keys}
    coverage = compute.get("kwh_coverage") or 0.0
    if got["gpu_board_kwh"] is not None and coverage < KWH_MIN_COVERAGE:
        missing.append(f"compute.json: the kWh covers {100 * coverage:.1f}% of the GPU-hours (needs 99%)")
    if None in got.values():
        return {}
    return {
        "flagship_h": f"{got['flagship_gpu_hours']:.1f}",
        "total_h": f"{got['total_gpu_hours']:.1f}",
        "kwh": f"{got['gpu_board_kwh']:.1f}",
    }


def _lichess_blanks(snap: rs.LichessSnapshot | None, missing: list[str]) -> dict:
    if snap is None:
        missing.append("lichess.json (written by blink lichess snapshot at G12)")
        return {}
    if not snap.publishable:
        missing.append(
            f"lichess.json is not publishable yet ({snap.n} games, RD {snap.rd}; needs 200 and < 75)"
        )
        return {}
    if _need(missing, "lichess.json: human_share", snap.human_share) is None:
        return {}
    return {
        "rating": f"{snap.rating}",
        "rd": f"{snap.rd}",
        "n": f"{snap.n:,}",
        "humans": f"{100 * snap.human_share:.0f}",
        "date": snap.snapshot_date,
    }


def _load(results_dir: Path) -> sb.Bundle:
    try:
        return sb.load_bundle(results_dir)
    except sb.ScoreboardError as exc:
        raise ClaimRefused(str(exc)) from exc


def fill_claim(results_dir: Path | None = None) -> str:
    bundle = _load(Path(results_dir if results_dir is not None else RESULTS_DIR))
    missing: list[str] = []
    shipped = bundle.results.shipped
    row = sb.shipped_row(bundle.results)
    if shipped is None or row is None:
        raise ClaimRefused("results.json names no shipped model with a strength row")
    if shipped.mode not in MODES:
        raise ClaimRefused(f"results.json ships mode {shipped.mode!r}; the claim knows {MODES}")
    blanks = {"mode_clause": MODE_CLAUSES[shipped.mode]}
    blanks |= _strength_blanks(row, missing)
    blanks |= _compute_blanks(bundle.compute, missing)
    blanks |= _lichess_blanks(bundle.lichess, missing)
    dm = _dm_row(bundle.results)
    if dm is None:
        missing.append("results.json: exactly one re-measured DeepMind reference row with an Elo")
    else:
        blanks |= {"dm_elo": f"{dm.elo:.0f}", "dm_ci": f"{dm.elo_ci95:.0f}"}
    if missing:
        raise ClaimRefused("the claim needs values results/*.json does not hold: " + "; ".join(missing))
    return CLAIM.format(**blanks)
