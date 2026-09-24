"""The README scoreboard, generated from results/*.json and nothing else (plan section 6).

`blink report scoreboard --write` puts the generated block between the README markers
<!-- scoreboard:start --> and <!-- scoreboard:end -->; `--check` fails when the README differs from
it by a single byte. The block holds, in order: the headline table (each number with its reproduce
command and what it does NOT prove), the no-search box, Table 1 (strength), Table 2 (ML diagnostics)
and the DeepMind puzzles by rating band. Every Elo cell carries its 95% interval and game count, every
percentage its Wilson 95% interval (and, where results.json stores a count rather than an interval,
its n), Elo below the lowest anchor (1320) is labelled extrapolated, paper numbers sit only in the
paper-reported column, and the text is ASCII (+/- rather than a plus-minus sign).
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from blink.eval.puzzles import wilson
from blink.report import compute as compute_mod
from blink.report import results_schema as rs
from blink.train.atomic import write_text_atomic

START = "<!-- scoreboard:start -->"
END = "<!-- scoreboard:end -->"
NEWLINE = "\n"
ANCHOR_FLOOR = 1320
DASH = "-"
TABLE1_HEADING = "### Table 1: strength"
TABLE2_HEADING = "### Table 2: ML diagnostics"
BANDS_HEADING = "### DeepMind puzzles by rating band"
ELO_CAVEAT = "a human or FIDE rating: it is CCRL-Blitz-anchored, with about +/-100 absolute error"
FRACTION_FIELDS = ("top1", "top3", "top5", "vaa", "near_best", "mate_shortest", "mate_preserving")
BAND_ORDER = ("<1000", "1000-1500", "1500-2000", "2000-2500", "2500+")


class ScoreboardError(ValueError):
    """The results cannot produce a scoreboard (a missing file, a unit mix-up, bad README markers)."""


@dataclass(frozen=True)
class Bundle:
    results: rs.Results
    lichess: rs.LichessSnapshot | None
    nosearch: dict
    compute: dict


def _read(folder: Path, name: str, hint: str) -> str:
    path = Path(folder) / name
    if not path.is_file():
        raise ScoreboardError(f"{path} is missing ({hint})")
    return path.read_text(encoding="utf-8")


NOSEARCH_KEYS = ("decisions", "games", "max_rows", "histogram", "violations", "compliant", "missing_counts")


def _parse(folder: Path) -> Bundle:
    results = rs.from_json(_read(folder, "results.json", "written by the evaluation suite, P8"))
    nosearch = json.loads(_read(folder, "nosearch.json", "run: uv run blink audit no-search"))
    _read(folder, "compute.json", "run: uv run blink report compute")
    compute = compute_mod.read_compute(Path(folder) / "compute.json")
    lichess_path = Path(folder) / "lichess.json"
    lichess = None
    if lichess_path.is_file():
        lichess = rs.lichess_from_json(lichess_path.read_text(encoding="utf-8"))
    return Bundle(results, lichess, nosearch, compute)


def load_bundle(folder: Path) -> Bundle:
    """results.json, nosearch.json and compute.json are required; lichess.json appears at G12."""
    try:
        bundle = _parse(Path(folder))
    except ScoreboardError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise ScoreboardError(f"{folder}: a results file does not match its schema: {exc}") from exc
    missing = [key for key in NOSEARCH_KEYS if key not in bundle.nosearch]
    if missing:
        raise ScoreboardError(f"nosearch.json lacks {missing} (rerun: uv run blink audit no-search)")
    _check_nosearch(bundle.nosearch)
    return bundle


def _check_nosearch(report: dict) -> None:
    """The box says 'at most legal+1 positions each': only a clean audit may say so."""
    problems = []
    if report["violations"]:
        problems.append(f"{len(report['violations']):,} violation(s)")
    if report["compliant"] is not True:
        problems.append("the audit is not compliant")
    if report["missing_counts"]:
        problems.append(f"{report['missing_counts']:,} moves without a node count")
    if not report["decisions"]:
        problems.append("no public moves were audited")
    if problems:
        raise ScoreboardError(f"nosearch.json: {'; '.join(problems)}: the no-search box cannot be published")


def shipped_row(results: rs.Results) -> rs.StrengthRow | None:
    """The strength row of the shipped model: '<agent> (<mode>)' first, then the bare agent name."""
    shipped = results.shipped
    if shipped is None:
        return None
    rows = {row.agent: row for row in results.strength}
    return rows.get(f"{shipped.agent} ({shipped.mode})") or rows.get(shipped.agent)


def _is_shipped_diag(results: rs.Results, row: rs.DiagnosticsRow) -> bool:
    shipped = results.shipped
    return shipped is not None and (row.agent, row.mode) == (shipped.agent, shipped.mode)


# ------------------------------------------------------------------------------------------ formatting


def elo_cell(row: rs.StrengthRow | None) -> str:
    if row is None or row.elo is None:
        return DASH
    text = f"{row.elo:.0f} +/- {row.elo_ci95:.0f} ({row.elo_games:,} games)"
    return f"{text}, extrapolated" if row.elo < ANCHOR_FLOOR else text


def pct_ci(pct: float | None, ci: tuple[float, float] | None) -> str:
    if pct is None:
        return DASH
    if ci is None:
        raise ScoreboardError(f"{pct}% has no 95% interval: a percentage is never printed without one")
    return f"{pct:.1f}% ({ci[0]:.1f} to {ci[1]:.1f})"


def pct_n(pct: float | None, n: int | None) -> str:
    """A percentage of n trials with its Wilson 95% interval and its n."""
    if pct is None:
        return DASH
    if not n:
        raise ScoreboardError(f"{pct}% has no n: its Wilson interval cannot be shown")
    low, high = wilson(round(pct / 100 * n), n)
    return f"{pct:.1f}% ({100 * low:.1f} to {100 * high:.1f}, n={n:,})"


def millions(n: int | None) -> str:
    return DASH if n is None else f"{n / 1e6:.2f}M"


def _num(value: float | int | None, fmt: str) -> str:
    return DASH if value is None else format(value, fmt)


def _positions(n: int | None) -> str:
    return DASH if n is None else f"{n / 1e6:.1f}M"


def _pct(fraction: float | None) -> str:
    return DASH if fraction is None else f"{100 * fraction:.1f}%"


def lichess_cell(snap: rs.LichessSnapshot | None, with_share: bool = False) -> str:
    if snap is None:
        return "rating accruing"
    if not snap.publishable:
        return f"rating accruing ({snap.n:,} games, RD {snap.rd})"
    if with_share:
        share = f", {_pct(snap.human_share)} vs humans" if snap.human_share is not None else ""
        return f"{snap.rating}, RD {snap.rd}, {snap.n:,} games{share}, {snap.snapshot_date}"
    return f"{snap.rating} +/- {2 * snap.rd} (2 RD), {snap.n:,} games, {snap.snapshot_date}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _bold(text: str, on: bool) -> str:
    return f"**{text}**" if on and text != DASH else text


# ------------------------------------------------------------------------------------------ sections


def _gpu_cell(compute: dict) -> str:
    flag, total, kwh = compute["flagship_gpu_hours"], compute["total_gpu_hours"], compute["gpu_board_kwh"]
    energy = DASH if kwh is None else f"{kwh:.1f}"
    coverage = compute.get("kwh_coverage") or 0.0
    partial = (
        f" (measured on {100 * coverage:.0f}% of the GPU-h)" if kwh is not None and coverage < 0.99 else ""
    )
    return (
        f"{_num(flag, '.1f')} flagship / {_num(total, '.1f')} total GPU-h, {energy} GPU-board kWh{partial}; "
        "no cloud GPU and no paid data: one home RTX 3070"
    )


def headline(bundle: Bundle) -> str:
    row = shipped_row(bundle.results)
    rows = [
        ["Elo vs Stockfish 19 UCI_Elo anchors, 95% CI (games)", elo_cell(row),
         "`uv run blink rate --model ship`", ELO_CAVEAT],
        ["Lichess BOT blitz (rating, RD, games, share vs humans, date)", lichess_cell(bundle.lichess, True),
         "`results/lichess.json` (snapshot at G12)",
         "strength against humans: the pool is mostly bots, it started at 3000, and it is not comparable "
         "with DeepMind's 2024 numbers"],
        ["DeepMind 10K puzzles, Wilson 95%", pct_ci(row.dm_puzzles_pct, row.dm_puzzles_ci) if row else DASH,
         "`uv run blink eval puzzles --model ship`",
         "playing strength; only exact positions (and colour mirrors) were kept out of training, "
         "near-duplicates remain"],
        ["GPU-hours (flagship / whole project), GPU-board kWh", _gpu_cell(bundle.compute),
         "`results/compute.json`",
         "zero electricity: kWh counts the GPU board only; whole-PC energy is estimated separately"],
    ]  # fmt: skip
    return table(["headline number", "measured", "reproduce", "what it does NOT prove"], rows)


def nosearch_box(report: dict) -> str:
    histogram = {int(k): v for k, v in report.get("histogram", {}).items()}
    violations = len(report.get("violations", []))
    one = histogram.get(1, 0)
    many = sum(v for k, v in histogram.items() if k >= 2)
    return (
        f"**No search.** Across {report['decisions']:,} public moves in {report['games']:,} games: at most "
        f"1 network call and legal+1 positions each ({violations:,} violations; largest batch "
        f"{report['max_rows']:,} rows). Rows per move: {histogram.get(0, 0):,} with 0 rows (R2 mate now), "
        f"{one:,} with 1 (one look), {many:,} with 2 or more (one look per move, legal+1). The positions "
        "per move are rebuilt from the PGNs alone by `uv run blink audit no-search`; the one-call limit is "
        "enforced inside the engine by EvalBudget, which raises on a second call in one decision.\n"
    )


TABLE1_HEADER = [
    "agent", "params (non-GAB / total)", "positions seen", "GPU-h", "network evals per move (median / max)",
    "ms per move p50", "Elo vs SF19 anchors, 95% CI (games)", "about SF19 at N nodes",
    "DeepMind puzzles, Wilson 95%", "clean subset (n)", "Lichess blitz (R +/- 2RD, N, date)",
    "paper-reported (scale named)", "reproduce",
]  # fmt: skip


def _strength_line(row: rs.StrengthRow, shipped: bool, lichess: rs.LichessSnapshot | None) -> list[str]:
    params = millions(row.params_total)
    if row.params_non_gab is not None:
        params = f"{millions(row.params_non_gab)} / {params}"
    evals = DASH
    if row.evals_per_move_median is not None:
        evals = f"{row.evals_per_move_median:g} / {_num(row.evals_per_move_max, 'd')}"
    clean = DASH
    if row.dm_puzzles_clean_pct is not None:
        clean = f"{row.dm_puzzles_clean_pct:.1f}% ({_num(row.dm_puzzles_clean_n, ',')})"
    return [
        _bold(row.agent, shipped), params, _positions(row.positions_seen), _num(row.gpu_hours, ".1f"), evals,
        _num(row.ms_per_move_p50, ".1f"),
        elo_cell(row), _num(row.sf_nodes_equiv, ","), pct_ci(row.dm_puzzles_pct, row.dm_puzzles_ci), clean,
        lichess_cell(lichess) if shipped else DASH, row.paper_reported or DASH, f"`{row.reproduce}`",
    ]  # fmt: skip


def strength_table(bundle: Bundle) -> str:
    shipped = shipped_row(bundle.results)
    rows = [_strength_line(r, r is shipped, bundle.lichess) for r in bundle.results.strength]
    return f"{TABLE1_HEADING}\n\n" + table(TABLE1_HEADER, rows)


def _check_fractions(row: rs.DiagnosticsRow) -> None:
    for name in FRACTION_FIELDS:
        value = getattr(row, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ScoreboardError(f"{row.agent} ({row.mode}): {name}={value} is not a fraction in [0, 1]")


def _diag_line(row: rs.DiagnosticsRow, shipped: bool) -> list[str]:
    _check_fractions(row)
    tops = DASH
    if row.top1 is not None:
        tops = (
            " / ".join(DASH if v is None else f"{100 * v:.1f}" for v in (row.top1, row.top3, row.top5)) + "%"
        )
    ece = (
        f"{_num(row.ece_before, '.3f')} / {_num(row.ece_after, '.3f')}"
        if row.ece_before is not None
        else DASH
    )
    mates = (
        f"{_pct(row.mate_shortest)} / {_pct(row.mate_preserving)}" if row.mate_shortest is not None else DASH
    )
    rating = DASH
    if row.puzzle_rating_equiv is not None:  # the schema requires its bootstrap CI
        ci = row.puzzle_rating_ci
        rating = f"{row.puzzle_rating_equiv:.0f} ({ci[0]:.0f} to {ci[1]:.0f})"
    return [
        _bold(row.agent, shipped), _bold(row.mode, shipped), tops, _pct(row.vaa), _pct(row.near_best),
        _num(row.kendall_tau_b, ".3f"), _num(row.brier, ".3f"), ece, _num(row.regret_games10k, ".3f"),
        _num(row.grouped_gap, ".3f"), mates, pct_n(row.conversion_pct, row.conversion_n), rating,
    ]  # fmt: skip


TABLE2_HEADER = [
    "agent", "mode", "top-1 / 3 / 5", "VAA (ties count)", "near-best", "Kendall tau-b (scores)", "Brier",
    "ECE before / after temperature", "win% regret (games10k)", "grouped vs random gap",
    "mate shortest / preserving", "conversion (proxy)", "puzzle-rating equivalent (CI)",
]  # fmt: skip


def diagnostics_table(bundle: Bundle) -> str:
    results = bundle.results
    rows = [_diag_line(r, _is_shipped_diag(results, r)) for r in results.diagnostics]
    return f"{TABLE2_HEADING}\n\n" + table(TABLE2_HEADER, rows)


def band_table(bundle: Bundle) -> str:
    results = bundle.results
    with_bands = [r for r in results.diagnostics if r.band_pct]
    keys = {k for r in with_bands for k in r.band_pct}
    bands = [b for b in BAND_ORDER if b in keys] + sorted(keys - set(BAND_ORDER))
    rows = [
        [_bold(r.agent, _is_shipped_diag(results, r)), _bold(r.mode, _is_shipped_diag(results, r))]
        + [pct_n(r.band_pct.get(b), r.band_n.get(b)) for b in bands]
        for r in with_bands
    ]
    return f"{BANDS_HEADING}\n\n" + table(["agent", "mode", *bands], rows)


def notes(results: rs.Results) -> str:
    return (
        "- Elo always names its pool: Stockfish 19 UCI_Elo anchors fitted by Ordo with fixed anchors "
        "(CCRL-Blitz-anchored, about +/-100 absolute error), not FIDE. Below 1320, the lowest anchor, a "
        "rating is extrapolated.\n"
        "- Paper numbers sit only in the paper-reported column. DeepMind's paper describes a Stockfish "
        "fallback its released code does not include; Blink's harness has none, and DM-9M is re-measured "
        "without it.\n"
        "- Time controls are asymmetric by design: Blink and DM-9M `st=1`, Stockfish `st=0.1`.\n"
        "- Kendall tau-b is computed on scores and is not comparable with DeepMind's. Validation metrics "
        "are optimistic; the grouped split shows by how much. The puzzle-rating equivalent is a logistic "
        "fit on puzzle ratings, never an Elo.\n"
        "- Generated by `uv run blink report scoreboard --write` from results/*.json "
        f"({results.generated_at}, EVAL.md sha {results.eval_md_sha[:12]}).\n"
    )


def render_block(bundle: Bundle) -> str:
    parts = [
        headline(bundle),
        nosearch_box(bundle.nosearch),
        strength_table(bundle),
        diagnostics_table(bundle),
        band_table(bundle),
        notes(bundle.results),
    ]
    return "\n".join(parts)


# ------------------------------------------------------------------------------------------ README


def _span(text: str) -> tuple[int, int]:
    """Start and end offsets of the block between the markers; ScoreboardError unless each appears once."""
    if text.count(START) != 1 or text.count(END) != 1:
        raise ScoreboardError(f"the README needs exactly one {START} and one {END} marker")
    begin = text.index(START) + len(START)
    end = text.index(END)
    if text[begin : begin + 1] != NEWLINE or end <= begin or text[end - 1] != NEWLINE:
        raise ScoreboardError(f"{START} and {END} must each sit on their own line, start first")
    return begin + 1, end


def write_readme(path: Path, block: str) -> None:
    text = Path(path).read_text(encoding="utf-8")
    begin, end = _span(text)
    write_text_atomic(Path(path), text[:begin] + block + text[end:])


def check_readme(path: Path, block: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        begin, end = _span(text)
    except ScoreboardError as exc:
        return [str(exc)]
    if text[begin:end] != block:
        return [
            f"{path}: the scoreboard block differs from results/*.json (run: blink report scoreboard --write)"
        ]
    return []


# ------------------------------------------------------------------------------------------ numbers in prose

_CODE = re.compile(r"```.*?```|`[^`\n]*`|<!--.*?-->|\]\([^)]*\)|https?://\S+", re.DOTALL)
_BLOCK = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
_DATE = re.compile(r"\b\d{4}([-/])\d{2}\1\d{2}\b")  # a snapshot date is a label, not a measurement
_PLUS_MINUS = re.compile(r"\+/-|±")
_BETWEEN_DIGITS = re.compile(r"(?<=\d)([-/])(?=\.?\d)")  # 75-85%, 1800-2200, 50/60: both ends count
# A number not glued to a word, a path or a hyphenated name (top-1, ply-16, P3-P6), with an optional
# minus sign that is not itself glued to one, and a leading-dot decimal (.031).
_NUMBER = re.compile(r"(?<![\w.,/:#-])(-?)(\d[\d,]*(?:\.\d+)?|\.\d+)(%|[MBK]\b)?")
_SCALE = {"M": 1e6, "B": 1e9, "K": 1e3}
SMALL_COUNT = 12  # non-negative integers up to this are counts and section numbers, not measurements


def _prose(text: str) -> str:
    text = _DATE.sub(" ", _CODE.sub(" ", text))
    return _BETWEEN_DIGITS.sub(r" \1 ", _PLUS_MINUS.sub(" +/- ", text))


def numbers_in(text: str) -> list[str]:
    """Number tokens a reader sees in prose (code, links, comments and dates removed), small counts left
    out: signed numbers, both ends of a range, the N of +/-N and leading-dot decimals included."""
    found = []
    for match in _NUMBER.finditer(_prose(text)):
        sign, digits, suffix = match.group(1), match.group(2).rstrip(","), match.group(3) or ""
        small = "," not in digits and "." not in digits and int(digits) <= SMALL_COUNT
        if small and not sign and not suffix:
            continue
        found.append(sign + digits + suffix)
    return found


def prose_outside_block(text: str) -> str:
    """A README without its generated scoreboard block (the block is checked byte-exact on its own)."""
    return _BLOCK.sub("\n", text)


def stray_numbers(text: str, values: list[float], allowed: Mapping[str, str]) -> list[str]:
    """Numbers in `text` that no results/*.json value states and no named contract constant explains."""
    return [n for n in numbers_in(text) if n not in allowed and not is_measured(n, values)]


def _numeric_leaves(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _numeric_leaves(v)]
    if isinstance(value, list | tuple):
        return [x for v in value for x in _numeric_leaves(v)]
    return []


def measured_values(folder: Path) -> list[float]:
    """Every number stored in results/*.json (strings, such as paper-reported text, are not measurements)."""
    values = []
    for path in sorted(Path(folder).glob("*.json")):
        values += _numeric_leaves(json.loads(path.read_text(encoding="utf-8")))
    return values


def is_measured(token: str, values: list[float]) -> bool:
    """True when `token`, as written (rounding, % of a fraction, an M/B/K suffix), states a measured value."""
    suffix = token[-1] if token[-1] in "%MBK" else ""
    digits = token[: len(token) - len(suffix)].replace(",", "")
    written = float(digits)
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    tolerance = 0.5 * 10**-decimals + 1e-9
    scales = [1 / _SCALE[suffix]] if suffix in _SCALE else [1.0, 100.0]
    return any(abs(round(v * s, decimals) - written) <= tolerance for v in values for s in scales)
