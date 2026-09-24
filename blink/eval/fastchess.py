"""Gauntlets against Stockfish 19 anchors, played by fastchess (plan P8 match rules).

Per-engine time controls: Blink `st=1 timemargin=500`; Stockfish `st=0.1 timemargin=100` with
UCI_LimitStrength=true, UCI_Elo=<anchor>, Threads=1, Hash=16. Openings are played sequentially from
the book slice, each once per colour (-repeat). No resignation and no win adjudication; the one draw
adjudication is `-maxmoves 300` (600 engine plies). `-pgnout nodes=true` writes Blink's per-move row
count into the PGN, and the no-search audit runs on that PGN as soon as the games finish.

A dm:9M[:ema] selector runs the same UCI process under DeepMind's own name, DM-9M or DM-9M-ema, with no
--mode (it has one, action-value), and its moves are audited with the engine filter "dm" (PF60): under a
Blink name its games would be filed as Blink's and the "blink" audit filter would see none of its moves.
Stockfish at a fixed node count (the E4 node ladder) is `SF19-n<nodes>`, full strength.
"""

import hashlib
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import blink
from blink import paths
from blink.eval import books, nosearch
from blink.play.factory import RANDOM_SELECTORS, ModelUnavailable
from blink.reference import registry

BLINK_ST, BLINK_MARGIN_MS = 1.0, 500
SF_ST, SF_MARGIN_MS = 0.1, 100
MAX_MOVES = 300
# fastchess waits 10 s for uciok/readyok by default; several torch+CUDA engines starting together on a
# loaded machine can need longer (PF55). Blink's strength does not depend on this wait.
STARTUP_MS = 60_000
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SUMMARY = re.compile(
    r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+), Points: ([\d.]+)",
)
ELO = re.compile(r"^Elo: .*$", re.MULTILINE)
PTNML = re.compile(r"Ptnml\(0-2\): \[(\d+), (\d+), (\d+), (\d+), (\d+)\]")
NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
NAME_TAG_MAX = 40  # a model tag longer than this keeps its head plus a hash of the whole selector
NAME_HASH_CHARS = 7


def tools_dir() -> Path:
    return Path(r"D:\tools") if sys.platform == "win32" else paths.home() / "tools"


def fastchess_exe() -> Path:
    return tools_dir() / "fastchess" / "fastchess-windows-x86-64" / "fastchess.exe"


def stockfish_exe() -> Path:
    return tools_dir() / "stockfish" / "stockfish-windows-x86-64-universal.exe"


@dataclass(frozen=True)
class EngineSpec:
    name: str
    cmd: str
    args: tuple[str, ...] = ()
    st: float | None = None
    tc: str | None = None
    timemargin_ms: int = 0
    options: tuple[tuple[str, str], ...] = ()
    nodes: int | None = None  # a fixed node budget per move (the node ladder), instead of a clock

    def fastchess_args(self) -> list[str]:
        out = ["-engine", f"cmd={self.cmd}", f"name={self.name}"]
        if self.args:
            out.append(f"args={' '.join(self.args)}")
        if self.nodes is not None:
            out.append(f"nodes={self.nodes}")
        if self.tc:
            out.append(f"tc={self.tc}")
        elif self.st is not None:
            out.append(f"st={self.st:g}")
        if self.tc or self.st is not None:
            out.append(f"timemargin={self.timemargin_ms}")
        out += [f"option.{key}={value}" for key, value in self.options]
        return out


def dm_name(selector: str) -> str:
    """DM-9M or DM-9M-ema; a malformed dm selector is refused like a missing model, in one line."""
    try:
        return registry.parse(selector).name
    except ValueError as exc:
        raise ModelUnavailable(str(exc)) from exc


def model_tag(model: str) -> str:
    """The selector made name-safe. A long one keeps its first characters plus a short hash of the whole
    selector, so two models whose tags share a head never play under one name (Ordo tallies by name)."""
    tag = NAME_UNSAFE.sub("_", model).strip("_")
    if len(tag) <= NAME_TAG_MAX:
        return tag
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:NAME_HASH_CHARS]
    return f"{tag[: NAME_TAG_MAX - NAME_HASH_CHARS - 1]}-{digest}"


def engine_name(model: str, mode: str) -> str:
    """The fastchess name of the engine under test: DM-9M[-ema] for a dm selector (PF60), else Blink's."""
    if registry.is_dm(model):
        return dm_name(model)
    return f"Blink-{mode}-{model_tag(model)}"


def check_distinct_names(selectors: Sequence[str], mode: str) -> None:
    """Refuse two different selectors that would play under one name: Ordo would merge their games."""
    seen: dict[str, str] = {}
    for selector in selectors:
        name = engine_name(selector, mode)
        first = seen.setdefault(name, selector)
        if first != selector:
            raise ValueError(
                f"{first!r} and {selector!r} would both play as {name}: give one a distinct path"
            )


def audit_engine(name: str) -> str:
    """The no-search audit's player filter for an engine name: "dm" for DeepMind's, "blink" otherwise."""
    return "dm" if name.lower().startswith("dm-") else nosearch.DEFAULT_ENGINE


def blink_engine(model: str, mode: str, device: str = "cuda") -> EngineSpec:
    """Blink (or DM-9M, for a dm selector) as `python -m blink.uci`, with this harness's interpreter."""
    if registry.is_dm(model):
        args = ("-m", "blink.uci", f"--model={model}", f"--device={device}")
    else:
        selector = ("--random",) if model in RANDOM_SELECTORS else (f"--model={model}",)
        args = ("-m", "blink.uci", *selector, f"--mode={mode}", f"--device={device}")
    return EngineSpec(
        engine_name(model, mode), sys.executable, args, st=BLINK_ST, timemargin_ms=BLINK_MARGIN_MS
    )


def stockfish_anchor(elo: int, exe: Path) -> EngineSpec:
    options = (("UCI_LimitStrength", "true"), ("UCI_Elo", str(elo)), ("Threads", "1"), ("Hash", "16"))
    return EngineSpec(f"SF{elo}", str(exe), st=SF_ST, timemargin_ms=SF_MARGIN_MS, options=options)


def stockfish_nodes(nodes: int, exe: Path) -> EngineSpec:
    """Full-strength Stockfish 19 with a fixed node budget per move (Threads=1, Hash=16)."""
    options = (("Threads", "1"), ("Hash", "16"))
    return EngineSpec(f"SF19-n{nodes}", str(exe), nodes=nodes, options=options)


def with_tc(engine: EngineSpec, tc: str) -> EngineSpec:
    """A cutechess-style time control (for example 10+0.1) in place of the fixed movetime."""
    return replace(engine, tc=tc, st=None)


@dataclass(frozen=True)
class GauntletPlan:
    games: int
    book: Path
    book_start: int
    concurrency: int
    pgn_out: Path
    max_moves: int = MAX_MOVES

    def __post_init__(self) -> None:
        if self.games <= 0 or self.games % 2:
            raise ValueError(
                f"games must be a positive even number (each opening once per colour), got {self.games}"
            )


def build_command(exe: Path, blink_spec: EngineSpec, anchor: EngineSpec, plan: GauntletPlan) -> list[str]:
    return [
        str(exe),
        *blink_spec.fastchess_args(),
        *anchor.fastchess_args(),
        "-openings",
        f"file={plan.book}",
        "format=pgn",
        "order=sequential",
        f"start={plan.book_start}",
        "-repeat",
        "-rounds",
        str(plan.games // 2),
        "-concurrency",
        str(plan.concurrency),
        "-maxmoves",
        str(plan.max_moves),
        "-startup-ms",
        str(STARTUP_MS),
        "-recover",
        "-pgnout",
        f"file={plan.pgn_out}",
        "nodes=true",
        "-report",
        "penta=true",
    ]


def parse_summary(output: str) -> dict | None:
    """The last `Games: ... Points:` line of fastchess's report, or None if there is none."""
    text = ANSI.sub("", output)
    found = SUMMARY.findall(text)
    if not found:
        return None
    games, wins, losses, draws, points = found[-1]
    elo = ELO.findall(text)
    penta = PTNML.findall(text)
    return {
        "games": int(games),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "points": float(points),
        "elo": elo[-1] if elo else None,
        "penta": [int(c) for c in penta[-1]] if penta else None,
    }


def _engine_env() -> dict[str, str]:
    """The harness's own blink package first on the path, so the engine runs the code under test."""
    root = str(Path(blink.__file__).resolve().parent.parent)
    inherited = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, inherited) if p), "PYTHONUTF8": "1"}


def _book_start(book: str, pairs: int, skip: int = 0) -> tuple[Path, int]:
    """The book file and the first opening to play, after `skip` openings of the slice."""
    path, first, last = books.resolve(book)
    if last is not None and first + skip + pairs - 1 > last:
        raise ValueError(f"{skip + pairs} openings run past the end of the {book} slice ({first}-{last})")
    return path, first + skip


def run_fastchess(command: Sequence[str], log: Path) -> int:
    """Run fastchess from the log's folder: it autosaves its tournament state as config.json in its cwd."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=_engine_env(),
            cwd=log.parent,
            check=False,
        )
    return proc.returncode


@dataclass(frozen=True)
class Gauntlet:
    blink: EngineSpec
    anchor: EngineSpec
    plan: GauntletPlan

    def command(self) -> list[str]:
        return build_command(fastchess_exe(), self.blink, self.anchor, self.plan)


def prepare_gauntlet(
    model: str,
    mode: str,
    device: str,
    anchor: int,
    games: int,
    book: str,
    out_dir: Path,
    concurrency: int = 5,
    max_moves: int = MAX_MOVES,
    tc: str | None = None,
) -> Gauntlet:
    """Blink against one SF19 anchor: engines, book slice and a fresh timestamped PGN under out_dir."""
    blink_spec = blink_engine(model, mode, device)
    anchor_spec = stockfish_anchor(anchor, stockfish_exe())
    if tc:
        blink_spec, anchor_spec = with_tc(blink_spec, tc), with_tc(anchor_spec, tc)
    return prepare_pair(blink_spec, anchor_spec, games, book, out_dir, concurrency, max_moves)


def _unique(path: Path) -> Path:
    """`path`, or path-2, path-3, ... when it exists: two runs never share a PGN."""
    candidate, number = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
        number += 1
    return candidate


def prepare_pair(
    first: EngineSpec,
    second: EngineSpec,
    games: int,
    book: str,
    out_dir: Path,
    concurrency: int = 5,
    max_moves: int = MAX_MOVES,
    skip: int = 0,
) -> Gauntlet:
    """Any two engines on a book slice (after `skip` of its openings), with a fresh timestamped PGN."""
    book_path, start = _book_start(book, games // 2, skip)
    book_path, out_dir = book_path.resolve(), out_dir.resolve()
    pgn = _unique(out_dir / f"{first.name}_vs_{second.name}_{time.strftime('%Y%m%d-%H%M%S')}.pgn")
    return Gauntlet(first, second, GauntletPlan(games, book_path, start, concurrency, pgn, max_moves))


def execute(gauntlet: Gauntlet) -> dict:
    """Run fastchess, then audit its PGN. Writes <pgn>.log, <pgn>.nosearch.json and returns the report."""
    pgn = gauntlet.plan.pgn_out
    command = gauntlet.command()
    started = time.perf_counter()
    returncode = run_fastchess(command, pgn.with_suffix(".log"))
    engine = audit_engine(gauntlet.blink.name)
    audit = nosearch.audit([pgn] if pgn.is_file() else [], engine=engine)
    nosearch.write_report(audit, pgn.with_suffix(".nosearch.json"))
    return {
        "blink": gauntlet.blink.name,
        "anchor": gauntlet.anchor.name,
        "command": command,
        "returncode": returncode,
        "seconds": round(time.perf_counter() - started, 1),
        "pgn": str(pgn),
        "summary": parse_summary(pgn.with_suffix(".log").read_text(encoding="utf-8", errors="replace")),
        "audit": audit,
        "blink_forfeits": audit["forfeits"].get(gauntlet.blink.name, {}),
        "anchor_forfeits": audit["forfeits"].get(gauntlet.anchor.name, {}),
    }


def match_report(report: dict) -> dict:
    """A fastchess report in the shape every P8 block reads: the first engine's W/D/L, score, pentanomial."""
    summary = report.get("summary") or {}
    games = summary.get("games", 0)
    return {
        "a": report["blink"],
        "b": report["anchor"],
        "games": games,
        "wins": summary.get("wins", 0),
        "draws": summary.get("draws", 0),
        "losses": summary.get("losses", 0),
        "score": summary.get("points", 0.0) / games if games else None,
        "penta": summary.get("penta"),
        "pgn": report["pgn"],
        "returncode": report["returncode"],
        "audit": {k: report["audit"][k] for k in ("decisions", "compliant", "forfeits", "adjudications")},
    }


def run_gauntlet(**kwargs) -> dict:
    """prepare_gauntlet(**kwargs), then execute it."""
    return execute(prepare_gauntlet(**kwargs))
