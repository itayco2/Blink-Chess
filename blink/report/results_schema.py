"""The results/*.json contract shared by the evaluation suite (writer) and the report (reader).

Every public number lives in these files, and the README, the claims and the hook are generated from
them. Rows validate themselves, so a paper number can never sit in a measured column, and an Elo can
never appear without its interval and its game count.
"""

import dataclasses
import json
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
ROW_KINDS = ("ladder", "blink", "reference", "anchor")
PUBLISH_MIN_GAMES = 200
PUBLISH_MAX_RD = 75


@dataclass(frozen=True)
class StrengthRow:
    agent: str
    kind: str
    reproduce: str
    params_total: int | None = None
    params_non_gab: int | None = None
    positions_seen: int | None = None
    gpu_hours: float | None = None
    evals_per_move_median: float | None = None
    evals_per_move_max: int | None = None
    ms_per_move_p50: float | None = None
    elo: float | None = None
    elo_ci95: float | None = None
    elo_games: int | None = None
    sf_nodes_equiv: int | None = None
    dm_puzzles_pct: float | None = None
    dm_puzzles_ci: tuple[float, float] | None = None
    dm_puzzles_clean_pct: float | None = None
    dm_puzzles_clean_n: int | None = None
    paper_reported: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ROW_KINDS:
            raise ValueError(f"row kind {self.kind!r} is not one of {ROW_KINDS}")
        if self.paper_reported and self.elo is not None:
            raise ValueError(
                f"{self.agent}: a paper number goes in paper_reported, never in the measured elo"
            )
        if self.elo is not None and self.elo_ci95 is None:
            raise ValueError(f"{self.agent}: an Elo needs its 95% interval")
        if self.elo is not None and not self.elo_games:
            raise ValueError(f"{self.agent}: an Elo needs its number of games")


@dataclass(frozen=True)
class DiagnosticsRow:
    agent: str
    mode: str
    top1: float | None = None
    top3: float | None = None
    top5: float | None = None
    vaa: float | None = None
    near_best: float | None = None
    kendall_tau_b: float | None = None
    brier: float | None = None
    ece_before: float | None = None
    ece_after: float | None = None
    regret_games10k: float | None = None
    grouped_gap: float | None = None
    band_pct: dict[str, float] = field(default_factory=dict)
    mate_shortest: float | None = None
    mate_preserving: float | None = None
    conversion_pct: float | None = None
    puzzle_rating_equiv: float | None = None
    puzzle_rating_ci: tuple[float, float] | None = None


@dataclass(frozen=True)
class Shipped:
    agent: str
    mode: str
    sha: str


@dataclass(frozen=True)
class Results:
    strength: tuple[StrengthRow, ...]
    diagnostics: tuple[DiagnosticsRow, ...]
    shipped: Shipped | None
    eval_md_sha: str
    generated_at: str


@dataclass(frozen=True)
class LichessSnapshot:
    bot: str
    rating: int
    rd: int
    n: int
    snapshot_date: str
    human_share: float | None = None
    perf_vs_humans: float | None = None
    perf_vs_bots: float | None = None
    time_loss_rate: float | None = None
    abort_rate: float | None = None
    duplicate_rate: float | None = None

    @property
    def publishable(self) -> bool:
        return self.n >= PUBLISH_MIN_GAMES and self.rd < PUBLISH_MAX_RD


def _tuples(value):
    return tuple(value) if isinstance(value, list) else value


def _row(cls, data: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: _tuples(v) if k.endswith("_ci") else v for k, v in data.items() if k in names})


def to_json(results: Results) -> str:
    payload = {"schema_version": SCHEMA_VERSION, **dataclasses.asdict(results)}
    return json.dumps(payload, indent=2, sort_keys=True)


def from_json(text: str) -> Results:
    data = json.loads(text)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"results schema {data.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    shipped = data.get("shipped")
    return Results(
        strength=tuple(_row(StrengthRow, r) for r in data["strength"]),
        diagnostics=tuple(_row(DiagnosticsRow, r) for r in data["diagnostics"]),
        shipped=Shipped(**shipped) if shipped else None,
        eval_md_sha=data["eval_md_sha"],
        generated_at=data["generated_at"],
    )


def lichess_to_json(snapshot: LichessSnapshot) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "publishable": snapshot.publishable,
        **dataclasses.asdict(snapshot),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def lichess_from_json(text: str) -> LichessSnapshot:
    data = json.loads(text)
    return _row(LichessSnapshot, data)
