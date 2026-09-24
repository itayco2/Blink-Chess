"""A round robin: every pair of sides plays one in-process match on the same openings (plan P3).

Pair (i, j), i < j in the order given, is run_match(side_i, side_j), so side i is White in the even
games and every pair plays the same openings in the same order. A side is named in the table by the
selector it was given (two sides can share an agent name, as two MLP files do). Each pair's games go
to <out>/<a>_vs_<b>.pgn with that match's summary beside it as .json, and the cross table goes to
<out>/round_robin.json. A folder that already holds one of those files is refused, because a match
appends to its PGN and two round robins must never mix.
"""

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink.eval import match
from blink.eval.books import Opening
from blink.eval.fastchess import NAME_UNSAFE
from blink.play.agents import Agent
from blink.train.atomic import write_text_atomic

RECORD = "round_robin.json"
PAIR_FIELDS = ("games", "a_wins", "draws", "a_losses", "a_score", "illegal_moves", "crashes", "adjudications")

Summaries = dict[tuple[int, int], dict[str, Any]]


@dataclass(frozen=True)
class Entrant:
    label: str  # the selector as given: the table's row and column name
    agent: Agent


def pairs(n: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(n), 2))


def file_stem(label: str) -> str:
    return NAME_UNSAFE.sub("_", label).strip("_")


def pgn_name(a: str, b: str) -> str:
    return f"{file_stem(a)}_vs_{file_stem(b)}.pgn"


def _points(summary: dict[str, Any], for_a: bool) -> float:
    wins = summary["a_wins"] if for_a else summary["a_losses"]
    return wins + 0.5 * summary["draws"]


def cross_table(labels: Sequence[str], summaries: Summaries) -> dict[str, dict[str, float]]:
    """table[x][y]: x's score against y (wins plus half the draws, over the games), from x's view."""
    table: dict[str, dict[str, float]] = {label: {} for label in labels}
    for (i, j), summary in summaries.items():
        games = summary["games"]
        table[labels[i]][labels[j]] = _points(summary, True) / games
        table[labels[j]][labels[i]] = _points(summary, False) / games
    return table


def total_scores(labels: Sequence[str], summaries: Summaries) -> dict[str, float]:
    """Each side's points over all its games."""
    points = dict.fromkeys(labels, 0.0)
    games = dict.fromkeys(labels, 0)
    for (i, j), summary in summaries.items():
        for index, for_a in ((i, True), (j, False)):
            points[labels[index]] += _points(summary, for_a)
            games[labels[index]] += summary["games"]
    return {label: points[label] / games[label] if games[label] else 0.0 for label in labels}


def format_table(labels: Sequence[str], table: dict[str, dict[str, float]], scores: dict[str, float]) -> str:
    """Rows score against columns; the last column is each side's score over all its games."""
    width = max(8, *(len(label) for label in labels)) + 2
    header = " " * width + "".join(label.rjust(width) for label in labels) + "score".rjust(width)
    rows = [header]
    for row in labels:
        cells = ["-" if row == col else f"{100 * table[row][col]:.1f}%" for col in labels]
        rows.append(
            row.ljust(width)
            + "".join(c.rjust(width) for c in cells)
            + f"{100 * scores[row]:.1f}%".rjust(width)
        )
    return "\n".join(rows)


def check_labels(labels: Sequence[str]) -> None:
    """ValueError unless there are at least two sides and every side has its own file name."""
    if len(labels) < 2:
        raise ValueError(f"a round robin needs at least two sides, got {list(labels)}")
    stems = [file_stem(label) for label in labels]
    if len(set(stems)) != len(stems):
        raise ValueError(f"every side must appear once (and differ in file-name characters): {list(labels)}")


def check_out(out_dir: Path, labels: Sequence[str]) -> None:
    """FileExistsError when out_dir already holds this round robin's record or one of its PGNs."""
    names = [RECORD] + [pgn_name(labels[i], labels[j]) for i, j in pairs(len(labels))]
    taken = [name for name in names if (Path(out_dir) / name).exists()]
    if taken:
        raise FileExistsError(f"{out_dir} already holds {taken[0]}; choose another --out")


def _pair_row(a: Entrant, b: Entrant, pgn: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "a": a.label,
        "b": b.label,
        "a_name": a.agent.name,
        "b_name": b.agent.name,
        "pgn": pgn.name,
        **{field: summary[field] for field in PAIR_FIELDS},
    }


def run_round_robin(
    entrants: Sequence[Entrant],
    openings: Sequence[Opening],
    games: int,
    out_dir: Path,
    max_plies: int = match.MAX_ENGINE_PLIES,
    on_pair: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Play every pair, write each pair's PGN and summary and the round robin's record; return the record."""
    labels = [e.label for e in entrants]
    check_labels(labels)
    out_dir = Path(out_dir)
    check_out(out_dir, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: Summaries = {}
    rows = []
    for i, j in pairs(len(entrants)):
        a, b = entrants[i], entrants[j]
        pgn = out_dir / pgn_name(a.label, b.label)
        summary = match.run_match(a.agent, b.agent, openings, games, pgn, max_plies=max_plies)
        write_text_atomic(pgn.with_suffix(".json"), json.dumps(summary, indent=2))
        summaries[(i, j)] = summary
        rows.append(_pair_row(a, b, pgn, summary))
        if on_pair is not None:
            on_pair({**rows[-1], "summary": summary})
    record = {
        "sides": labels,
        "names": {e.label: e.agent.name for e in entrants},
        "games_per_pair": games,
        "pairs": rows,
        "table": cross_table(labels, summaries),
        "scores": total_scores(labels, summaries),
    }
    write_text_atomic(out_dir / RECORD, json.dumps(record, indent=2) + "\n")
    return record
