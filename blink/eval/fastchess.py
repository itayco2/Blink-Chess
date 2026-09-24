"""Gauntlets against Stockfish 19 anchors, played by fastchess (plan P8 match rules).

Per-engine time controls: Blink `st=1 timemargin=500`; Stockfish `st=0.1 timemargin=100` with
UCI_LimitStrength=true, UCI_Elo=<anchor>, Threads=1, Hash=16. Openings are played sequentially from
the book slice, each once per colour (-repeat). No resignation and no win adjudication; the one draw
adjudication is `-maxmoves 300` (600 engine plies). `-pgnout nodes=true` writes Blink's per-move row
count into the PGN, and the no-search audit runs on that PGN as soon as the games finish.
"""

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
from blink.play.factory import RANDOM_SELECTORS

BLINK_ST, BLINK_MARGIN_MS = 1.0, 500
SF_ST, SF_MARGIN_MS = 0.1, 100
MAX_MOVES = 300
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SUMMARY = re.compile(
    r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+), Points: ([\d.]+)",
)
ELO = re.compile(r"^Elo: .*$", re.MULTILINE)
NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


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

    def fastchess_args(self) -> list[str]:
        out = ["-engine", f"cmd={self.cmd}", f"name={self.name}"]
        if self.args:
            out.append(f"args={' '.join(self.args)}")
        out.append(f"tc={self.tc}" if self.tc else f"st={self.st:g}")
        out.append(f"timemargin={self.timemargin_ms}")
        out += [f"option.{key}={value}" for key, value in self.options]
        return out


def engine_name(model: str, mode: str) -> str:
    tag = NAME_UNSAFE.sub("_", model).strip("_")[:40]
    return f"Blink-{mode}-{tag}"


def blink_engine(model: str, mode: str, device: str = "cuda") -> EngineSpec:
    """Blink as `python -m blink.uci`, with the interpreter that runs this harness."""
    selector = ("--random",) if model in RANDOM_SELECTORS else (f"--model={model}",)
    args = ("-m", "blink.uci", *selector, f"--mode={mode}", f"--device={device}")
    return EngineSpec(
        engine_name(model, mode), sys.executable, args, st=BLINK_ST, timemargin_ms=BLINK_MARGIN_MS
    )


def stockfish_anchor(elo: int, exe: Path) -> EngineSpec:
    options = (("UCI_LimitStrength", "true"), ("UCI_Elo", str(elo)), ("Threads", "1"), ("Hash", "16"))
    return EngineSpec(f"SF{elo}", str(exe), st=SF_ST, timemargin_ms=SF_MARGIN_MS, options=options)


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
    return {
        "games": int(games),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "points": float(points),
        "elo": elo[-1] if elo else None,
    }


def _engine_env() -> dict[str, str]:
    """The harness's own blink package first on the path, so the engine runs the code under test."""
    root = str(Path(blink.__file__).resolve().parent.parent)
    inherited = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, inherited) if p), "PYTHONUTF8": "1"}


def _book_start(book: str, pairs: int) -> tuple[Path, int]:
    path, first, last = books.resolve(book)
    if last is not None and first + pairs - 1 > last:
        raise ValueError(f"{pairs} openings run past the end of the {book} slice ({first}-{last})")
    return path, first


def run_fastchess(command: Sequence[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(
            list(command), stdout=handle, stderr=subprocess.STDOUT, env=_engine_env(), check=False
        )
    return proc.returncode


def run_gauntlet(
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
) -> dict:
    """Play Blink against one SF19 anchor, then audit the PGN. Returns (and writes) a JSON report."""
    blink_spec = blink_engine(model, mode, device)
    anchor_spec = stockfish_anchor(anchor, stockfish_exe())
    if tc:
        blink_spec, anchor_spec = with_tc(blink_spec, tc), with_tc(anchor_spec, tc)
    book_path, start = _book_start(book, games // 2)
    pgn = out_dir / f"{blink_spec.name}_vs_{anchor_spec.name}_{time.strftime('%Y%m%d-%H%M%S')}.pgn"
    plan = GauntletPlan(games, book_path, start, concurrency, pgn, max_moves)
    command = build_command(fastchess_exe(), blink_spec, anchor_spec, plan)
    started = time.perf_counter()
    returncode = run_fastchess(command, pgn.with_suffix(".log"))
    audit = nosearch.audit([pgn]) if pgn.is_file() else nosearch.audit([])
    report = {
        "blink": blink_spec.name,
        "anchor": anchor_spec.name,
        "command": command,
        "returncode": returncode,
        "seconds": round(time.perf_counter() - started, 1),
        "pgn": str(pgn),
        "summary": parse_summary(pgn.with_suffix(".log").read_text(encoding="utf-8", errors="replace")),
        "audit": audit,
        "blink_forfeits": audit["forfeits"].get(blink_spec.name, {}),
        "anchor_forfeits": audit["forfeits"].get(anchor_spec.name, {}),
    }
    nosearch.write_report(audit, pgn.with_suffix(".nosearch.json"))
    return report
