"""Ratings: fishtest's Elo interval for pairwise results, and Ordo with fixed Stockfish 19 anchors.

Two tools, never mixed (plan P8 "Rating"):
- `elo_ci` is fishtest's `stat_util.get_elo`, ported exactly: a 3-count (losses, draws, wins) or a 5-count
  pentanomial (LL, LD+DL, LW+DD+WL, DW+WD, WW), zero counts regularised to 1e-3, a normal 95% interval on
  the mean score mapped through the logistic Elo curve (clamped to [1e-3, 1 - 1e-3], as fishtest does).
  It serves pairwise results: the SPRT, the node ladder, Blink against DM-9M.
- The Elo in the strength table comes only from Ordo: `ordo -P <pgn list> -m <anchors> -W -D -s 1000`,
  every SF19 UCI_Elo anchor fixed at its nominal rating (no loose anchors, no in-house Bradley-Terry).
  Ordo cannot place a player with all wins or all losses (its maximum-likelihood rating is infinite), so
  such a player is excluded from the fit and reported with its score instead of a number.
Ratings below 1320 (the lowest UCI_Elo) are extrapolated and labelled so.
"""

import csv
import io
import math
import re
import subprocess
import sys
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import chess.pgn

from blink import paths

REGULARIZE = 1e-3  # fishtest LLRcalc.regularize
SCORE_CLAMP = 1e-3  # fishtest stat_util.elo
LOWEST_UCI_ELO = 1320
ORDO_SIMULATIONS = 1000
ANCHORS_FILE = Path("configs") / "anchors.csv"
WHITE_ADVANTAGE = re.compile(r"White advantage = (-?[\d.]+) \+/- ([\d.]+)")
DRAW_RATE = re.compile(r"Draw rate \(equal opponents\) = ([\d.]+) % \+/- ([\d.]+)")
RESULT_POINTS = {"1-0": (1.0, 0.0), "0-1": (0.0, 1.0), "1/2-1/2": (0.5, 0.5)}


# ------------------------------------------------------------------------------ fishtest parity


def regularize(counts: Sequence[float]) -> tuple[float, ...]:
    """Zero counts become 1e-3 (fishtest LLRcalc.regularize)."""
    return tuple(REGULARIZE if c == 0 else float(c) for c in counts)


def logistic_score(elo: float) -> float:
    return 1 / (1 + 10 ** (-elo / 400))


def logistic_elo(score: float) -> float:
    """fishtest stat_util.elo: the score clamped to [1e-3, 1 - 1e-3], then -400 log10(1/x - 1)."""
    x = min(max(score, SCORE_CLAMP), 1 - SCORE_CLAMP)
    return -400 * math.log10(1 / x - 1)


def _check_counts(counts: Sequence[float]) -> None:
    if len(counts) not in (3, 5):
        raise ValueError(f"counts must hold 3 (L, D, W) or 5 (pentanomial) entries, got {len(counts)}")


def game_stats(counts: Sequence[float]) -> tuple[float, float, float]:
    """fishtest stat_util.stats: (games, mean score per game, variance of the score per game)."""
    k = len(counts)
    total = sum(counts)
    games = total * (k - 1) / 2.0
    mu = sum(c * (i / 2.0) for i, c in enumerate(counts)) / games
    unit_mu = (k - 1) / 2.0 * mu
    var = sum(c * (i / 2.0 - unit_mu) ** 2 for i, c in enumerate(counts)) / games
    return games, mu, var


@dataclass(frozen=True)
class EloEstimate:
    elo: float
    ci95: float  # half-width of the 95% interval, in Elo
    los: float  # likelihood of superiority
    games: int

    def as_dict(self) -> dict:
        return {"elo": self.elo, "elo_ci95": self.ci95, "los": self.los, "games": self.games}


def elo_ci(counts: Sequence[int]) -> EloEstimate:
    """fishtest get_elo on (L, D, W) or a pentanomial, plus the number of games it covers."""
    _check_counts(counts)
    games_raw = int(sum(counts) * (len(counts) - 1) / 2)
    games, mu, var = game_stats(regularize(counts))
    stdev = math.sqrt(var)
    spread = stdev / math.sqrt(games)
    normal = NormalDist()
    mu_min = mu + normal.inv_cdf(0.025) * spread
    mu_max = mu + normal.inv_cdf(0.975) * spread
    elo95 = (logistic_elo(mu_max) - logistic_elo(mu_min)) / 2.0
    los = normal.cdf((mu - 0.5) / spread)
    return EloEstimate(logistic_elo(mu), elo95, los, games_raw)


def wdl_counts(wins: int, draws: int, losses: int) -> tuple[int, int, int]:
    """The trinomial in fishtest's order: losses, draws, wins."""
    return losses, draws, wins


def pentanomial(scores: Sequence[float]) -> tuple[int, int, int, int, int]:
    """Game-pair counts from per-game scores of one player, games 2k and 2k+1 sharing an opening."""
    if len(scores) % 2:
        raise ValueError(f"{len(scores)} games do not form whole pairs (each opening once per colour)")
    counts = [0] * 5
    for first, second in zip(scores[::2], scores[1::2], strict=True):
        counts[round(2 * (first + second))] += 1
    return tuple(counts)


# ------------------------------------------------------------------------------ anchors


@dataclass(frozen=True)
class Anchor:
    name: str  # the fastchess engine name, SF<elo>
    rating: int  # Stockfish 19 UCI_Elo, fixed in the fit


def anchor_name(elo: int) -> str:
    return f"SF{elo}"


def read_anchors(path: Path | None = None) -> tuple[Anchor, ...]:
    """Rows of "SF<elo>",<elo> (Ordo's -m format), sorted by rating."""
    path = path or Path(__file__).resolve().parents[2] / ANCHORS_FILE
    rows = csv.reader(io.StringIO(Path(path).read_text(encoding="utf-8")))
    anchors = [Anchor(name.strip(), int(value)) for name, value in (r for r in rows if r)]
    return tuple(sorted(anchors, key=lambda a: a.rating))


def nearest_anchor(estimate: float, anchors: Sequence[Anchor]) -> Anchor:
    return min(anchors, key=lambda a: (abs(a.rating - estimate), a.rating))


def centred_anchors(estimate: float, anchors: Sequence[Anchor], count: int = 5) -> tuple[Anchor, ...]:
    """`count` consecutive anchors whose middle one is nearest the estimate, clamped to the grid's ends."""
    grid = sorted(anchors, key=lambda a: a.rating)
    middle = grid.index(nearest_anchor(estimate, grid))
    start = min(max(0, middle - count // 2), max(0, len(grid) - count))
    return tuple(grid[start : start + count])


# ------------------------------------------------------------------------------ Ordo


def ordo_exe() -> Path:
    root = Path(r"D:\tools") if sys.platform == "win32" else paths.home() / "tools"
    return root / "ordo" / "ordo-1.2.6-win" / "ordo-win64.exe"


@dataclass(frozen=True)
class OrdoRow:
    player: str
    rating: float
    error: float | None  # None for a fixed anchor
    points: float
    played: int
    percent: float

    @property
    def is_anchor(self) -> bool:
        return self.error is None

    @property
    def extrapolated(self) -> bool:
        return self.rating < LOWEST_UCI_ELO


def ordo_command(
    exe: Path, pgn_list: Path, anchors_csv: Path, out_stem: Path, simulations: int = ORDO_SIMULATIONS
) -> list[str]:
    return [
        str(exe),
        "-P",
        str(pgn_list),
        "-m",
        str(anchors_csv),
        "-W",
        "-D",
        "-s",
        str(simulations),
        "-c",
        str(out_stem.with_suffix(".csv")),
        "-o",
        str(out_stem.with_suffix(".txt")),
        "-q",
    ]


def _number(text: str) -> float | None:
    text = text.strip()
    return None if text in ("-", "----", "") else float(text)


def parse_ordo_csv(text: str) -> list[OrdoRow]:
    rows = []
    for record in csv.DictReader(io.StringIO(text)):
        rows.append(
            OrdoRow(
                player=record["PLAYER"],
                rating=float(record["RATING"]),
                error=_number(record["ERROR"]),
                points=float(record["POINTS"]),
                played=int(record["PLAYED"]),
                percent=float(record["(%)"]),
            )
        )
    return rows


def parse_ordo_text(text: str) -> dict[str, float]:
    out = {}
    if found := WHITE_ADVANTAGE.search(text):
        out["white_advantage"], out["white_advantage_error"] = float(found[1]), float(found[2])
    if found := DRAW_RATE.search(text):
        out["draw_rate_pct"], out["draw_rate_error"] = float(found[1]), float(found[2])
    return out


def strength_fields(row: OrdoRow) -> dict[str, float | int]:
    """The StrengthRow fields an Ordo row fills: elo, elo_ci95 (Ordo's 95% error) and elo_games."""
    if row.error is None:
        raise ValueError(f"{row.player} is a fixed anchor: it has no measured Elo")
    return {"elo": row.rating, "elo_ci95": row.error, "elo_games": row.played}


def tally_players(pgns: Iterable[Path], without: Collection[str] = ()) -> dict[str, dict[str, float]]:
    """Games and points per player, from the PGN headers alone (unfinished games are skipped, and so is
    every game of a player in `without`)."""
    tally: dict[str, dict[str, float]] = {}
    for path in pgns:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while (headers := chess.pgn.read_headers(handle)) is not None:
                points = RESULT_POINTS.get(headers.get("Result", "*"))
                names = (headers.get("White", "?"), headers.get("Black", "?"))
                if points is None or any(name in without for name in names):
                    continue
                for side, gained in zip(("White", "Black"), points, strict=True):
                    entry = tally.setdefault(headers.get(side, "?"), {"games": 0, "points": 0.0})
                    entry["games"] += 1
                    entry["points"] += gained
    return tally


def unfittable(tally: dict[str, dict[str, float]], anchors: Sequence[Anchor] = ()) -> dict[str, str]:
    """Non-anchor players whose maximum-likelihood rating is infinite (all wins or all losses)."""
    fixed = {a.name for a in anchors}
    out = {}
    for player, entry in sorted(tally.items()):
        if player in fixed:
            continue
        if entry["points"] == 0:
            out[player] = "all losses"
        elif entry["points"] == entry["games"]:
            out[player] = "all wins"
    return out


def exclusions(pgns: Sequence[Path], anchors: Sequence[Anchor]) -> dict[str, str]:
    """Unfittable players, found again after each round of removals until none is left: a player whose
    only draws or wins came against an excluded player is itself unfittable once those games go."""
    excluded: dict[str, str] = {}
    while True:
        found = unfittable(tally_players(pgns, excluded), anchors)
        if not found:
            return excluded
        excluded |= found


def anchors_present(anchors: Sequence[Anchor], tally: dict[str, dict[str, float]]) -> tuple[Anchor, ...]:
    """Ordo refuses an anchor that played no game, so only the ones that played are fixed."""
    return tuple(a for a in anchors if tally.get(a.name, {}).get("games", 0) > 0)


@dataclass(frozen=True)
class OrdoFit:
    rows: tuple[OrdoRow, ...]
    anchors: tuple[Anchor, ...]
    excluded: dict[str, str]  # player -> why Ordo could not place it
    tally: dict[str, dict[str, float]]
    command: tuple[str, ...]
    extras: dict[str, float]  # white advantage and draw rate, with their errors

    def row(self, player: str) -> OrdoRow:
        return next(r for r in self.rows if r.player == player)

    def as_dict(self) -> dict:
        return {
            "rows": [r.__dict__ | {"extrapolated": r.extrapolated} for r in self.rows],
            "anchors": [a.__dict__ for a in self.anchors],
            "excluded": self.excluded,
            "tally": self.tally,
            "command": list(self.command),
            **self.extras,
        }


def _write_inputs(
    workdir: Path, pgns: Sequence[Path], anchors: Sequence[Anchor], excluded
) -> tuple[Path, ...]:
    workdir.mkdir(parents=True, exist_ok=True)
    listing, anchors_csv, exclude = workdir / "pgns.txt", workdir / "anchors.csv", workdir / "exclude.txt"
    listing.write_text("".join(f"{Path(p).resolve()}\n" for p in pgns), encoding="utf-8")
    anchors_csv.write_text("".join(f'"{a.name}",{a.rating}\n' for a in anchors), encoding="utf-8")
    exclude.write_text("".join(f'"{name}"\n' for name in excluded), encoding="utf-8")
    return listing, anchors_csv, exclude


def run_ordo(
    pgns: Sequence[Path],
    anchors: Sequence[Anchor],
    workdir: Path,
    simulations: int = ORDO_SIMULATIONS,
    exe: Path | None = None,
) -> OrdoFit:
    """Fit every player in `pgns` with the anchors that played fixed: the rows, and who was left out."""
    tally = tally_players(pgns)
    excluded = exclusions(pgns, anchors)
    present = anchors_present(anchors, tally_players(pgns, excluded))
    listing, anchors_csv, exclude = _write_inputs(workdir, pgns, present, excluded)
    command = ordo_command(exe or ordo_exe(), listing, anchors_csv, workdir / "ordo", simulations)
    if excluded:
        command += ["-x", str(exclude), "--no-warnings"]
    fitted = [p for p in tally if p not in excluded and p not in {a.name for a in present}]
    if not present or not fitted:
        return OrdoFit((), present, excluded, tally, tuple(command), {})
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ordo failed ({proc.returncode}): {(proc.stdout + proc.stderr).strip()[-400:]}")
    rows = parse_ordo_csv((workdir / "ordo.csv").read_text(encoding="utf-8", errors="replace"))
    extras = parse_ordo_text((workdir / "ordo.txt").read_text(encoding="utf-8", errors="replace"))
    return OrdoFit(tuple(rows), present, excluded, tally, tuple(command), extras)
