"""`blink eval all --model ship --protocol EVAL.md`: the P8 blocks in the plan's order, each runnable alone.

Before anything runs: the per-block game-count table is printed; when the annotated tag eval-v1-frozen
exists, EVAL.md must be byte-identical to the tagged copy (else nothing runs); and a time-based block
refuses to start while any Blink training run's heartbeat is live (a busy GPU or CPU would bend both the
clocks and the anchors). After each block its report goes to <out>/<block>.json with the time forfeits and
adjudications of every engine in its PGNs and the no-search audit of every searchless player in them
(Blink-*, DM-*, each by its exact name; the full audits go to <out>/<block>.nosearch.json). The plan's
done-when gates the run can fail are listed as gate_failures, and `blink eval` then exits non-zero. At the
end blink.eval.publish writes results/results.json (Ordo over the final-slice PGNs for the Elo column,
E2 and E6's rungs for the diagnostics, each Blink row's size, training and play costs) and
results/nosearch.json (one audit of every searchless player over every PGN the blocks wrote).

Where the games are played: the fastchess blocks are the ones against clocked UCI_Elo anchors (E0's SF
self-check, E5 and DM-9M's E7 gauntlet), at concurrency 5 as in the plan. Every other match runs in process
with one model load (one CUDA context, PF58), on fastchess's clocks (blink.eval.match): a Blink or DM-9M
move over 1.5 s (st=1 timemargin=500) and a Stockfish move over 0.2 s at `go movetime 100` lose on time;
Stockfish at `go nodes N` has no clock. Blink's compute never depends on time (N4), so the clock only
checks that no move ran long.
A `games` override makes every match that long (and every SPRT cap), for smoke runs.

SF19 labels (E2's win% regret and mate-preserving searches, E9's failures) run on ctx.sf_procs Stockfish
processes, one thread each: 5 from `blink eval all` (P8's CPU budget), 3 at most while Blink trains
(blink.eval.sfbudget). The blocks run one at a time, so no time-based block runs beside them.

Blink plays every block after E2b with the epsilon E2b chose (results/epsilon.json), in process and under
fastchess alike (blink-uci gets it as --epsilon); a block refuses to start if that file changed mid-run.

The weights are pinned the same way: the model's weights file is hashed once when the run starts, every
block report records that sha256, a block refuses to start if the file no longer hashes to it, and each
fastchess engine starts blink-uci with --sha, so it refuses other weights. results.json names a shipped
model only with that pinned sha, and only if the file still matches it at the end.

Blink plays one fast play mode for the whole run (--precision, --compile; blink.play.fastmode), fp32
uncompiled by default: in process (load_evaluator) and under fastchess (blink-uci gets the flags) alike.
The mode's tag ends every Blink engine name (Blink-value-ship-bf16-compile), so games in two modes are
never rated as one player; every block report records the mode, and results.json's shipped record says
which one the rated games used (the Lichess bot must play it). The default mode adds nothing to a name.
E1's film frames are static training snapshots and stay fp32. A block run alone into an --out folder whose
reports were played in another mode is refused before anything runs (E9 would find no Blink in E5's games).
"""

import datetime
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import chess.pgn
import numpy as np

from blink import paths
from blink.eval.publish import FINAL_SLICE_LIST, build_results, final_slice_pgns, public_audit, write_pgn_list
from blink.eval.sfbudget import budgeted, training_processes
from blink.play import fastmode

BLOCK_ORDER = ("E0", "E1", "E2", "E2b", "E3", "E4", "E4b", "E5", "E6", "E7", "E8", "E9")
FROZEN_TAG = "eval-v1-frozen"
SELFCHECK_ANCHOR = 1800
SELFCHECK_TC = "120+1"
SELFCHECK_BAND = 0.07
DM_EXPECTED = (88.9, 1.0)  # DM-9M `params` on the 10K puzzles, % (arXiv v2 Table 1), pre-registered
# A loaded machine bends st=0.1: in a CPU-busy smoke, SF19 forfeited 4 of 24 E5 games on time.
BUSY_CPU_PCT = 25.0
# A smoke run's few games can leave Ordo's error simulations crawling (40 games: 20 simulations > 100 s).
SMOKE_ORDO_SIMULATIONS = 100
SMOKE_ORDO_TIMEOUT_S = 120
CPU_SAMPLE_S = 3.0
# The blocks after E2b whose Blink plays with the epsilon E2b chose (in process or under fastchess).
EPSILON_BLOCKS = frozenset({"E3", "E4", "E4b", "E5", "E6", "E7", "E8"})
# The blocks with no Blink player in the fast mode: E0 (DM-9M and SF only) and E1 (film frames, fp32).
MODE_FREE_BLOCKS = frozenset({"E0", "E1"})


@dataclass(frozen=True)
class BlockSpec:
    id: str
    title: str
    games: int  # the plan's game count (E3: its worst case, 6,000 per direction)
    time_based: bool
    book: str | None


BLOCKS = {
    b.id: b
    for b in (
        BlockSpec("E0", "harness checks: DM-9M puzzles, SF st=0.1 against itself at 120+1", 200, True, "dev"),
        BlockSpec("E1", "21 film frames, static", 0, False, None),
        BlockSpec("E2", "static metrics per mode", 0, False, None),
        BlockSpec("E2b", "epsilon: 200 dev endgames x 3, then 1,000 games", 1600, True, "dev"),
        BlockSpec("E3", "mode SPRT (value against policy), cap 6,000 per direction", 12000, False, "dev"),
        BlockSpec("E4", "node ladder 2^4-2^16, 100 per rung, 500 at the bracket", 1500, True, "final"),
        BlockSpec("E4b", "6 film checkpoints x 200 at the crossover", 1200, True, "dev"),
        BlockSpec("E5", "anchors: locators, 5 x 400 for the final model, side rows", 6100, True, "final"),
        BlockSpec("E6", "ladder round robin, 15 pairs x 200", 3000, True, "final"),
        BlockSpec("E7", "DM-9M: 1,000 against anchors, 1,000 against Blink", 2000, True, "final"),
        BlockSpec("E8", "conversion: 500 final endgames, rules on and off", 1000, True, "final"),
        BlockSpec("E9", "failure classes (SF19 at 1M nodes, no games)", 0, False, "final"),
    )
}


class ProtocolMismatch(RuntimeError):
    """EVAL.md differs from the copy frozen under the eval-v1-frozen tag."""


class TrainingLive(RuntimeError):
    """A time-based block was asked for while a training run's heartbeat is live."""


class MachineBusy(RuntimeError):
    """A time-based block was asked for while other work kept the CPU busy (clocks would bend)."""


class EpsilonChanged(RuntimeError):
    """results/epsilon.json changed after earlier blocks of this run played with another value."""


class WeightsChanged(RuntimeError):
    """The model's weights file changed after earlier blocks of this run played it."""


class PlayModeChanged(RuntimeError):
    """A report in the --out folder was played in another fast mode than this run plays."""


@dataclass(frozen=True)
class EvalContext:
    model: str
    device: str = "cuda"
    out_dir: Path = field(default_factory=lambda: paths.home() / "eval" / "p8")
    results_dir: Path = Path("results")
    protocol: Path = Path("EVAL.md")
    games: int | None = None  # smoke override: every match plays this many games
    positions: int | None = None  # smoke override for the static blocks and the SF-labelled sets
    concurrency: int = 5
    mode: str | None = None  # the shipped mode, when known before E3
    film_run: str | None = None  # runs/<name>/film for E1 and E4b
    side_models: tuple[str, ...] = ()  # 6 GPU-h sizes and s10m for E5's side rows
    data_dir: Path | None = None  # the pack with val/test roots, valprobe.npz and mateset.npz
    selfcheck_tc: str = SELFCHECK_TC  # E0: the slow side of SF's st=0.1 self-check
    sf_procs: int = 1  # Stockfish processes for SF19 labels (E2 regret and mate-preserving, E9); eval all: 5
    allow_busy_cpu: bool = False  # smoke runs only: start time-based blocks on a busy machine
    precision: str = fastmode.DEFAULT_PRECISION  # the fast play mode of every Blink player in the run
    compile: bool = False

    def __post_init__(self) -> None:
        fastmode.check(self.precision, self.device)  # bf16 off CUDA is refused, never played as fp32

    @property
    def play_mode(self) -> dict:
        """precision and compile, as blink_engine, blink_agents and engine_name take them."""
        return {"precision": self.precision, "compile": self.compile}

    @property
    def mode_tag(self) -> str:
        """'' for the default mode, else the tag that ends every Blink engine name of this run."""
        return fastmode.tag(self.precision, self.compile)

    def n(self, default: int) -> int:
        """A match length: the override when set (rounded up to whole pairs), else the plan's number."""
        if self.games is None:
            return default
        return max(2, self.games + self.games % 2)


# ------------------------------------------------------------------------------ before anything runs


def game_table(ids: Sequence[str], games: int | None = None) -> str:
    """The per-block game-count table, printed before a run starts."""
    lines = [
        "| block | games (plan) | games (this run) | time-based | book | what |",
        "|---|---|---|---|---|---|",
    ]
    total = 0
    for block_id in ids:
        spec = BLOCKS[block_id]
        this = spec.games if games is None or spec.games == 0 else _smoke_games(spec, games)
        total += this
        clock = "yes" if spec.time_based else "no"
        lines.append(
            f"| {spec.id} | {spec.games:,} | {this:,} | {clock} | {spec.book or '-'} | {spec.title} |"
        )
    lines.append(f"| total | | {total:,} | | | |")
    return "\n".join(lines)


def _smoke_games(spec: BlockSpec, games: int) -> int:
    """A rough count of a smoke run's games: matches per block times the override."""
    matches = {"E0": 1, "E2b": 4, "E3": 2, "E4": 9, "E4b": 6, "E5": 12, "E6": 15, "E7": 7, "E8": 2}
    return matches.get(spec.id, 1) * max(2, games + games % 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_file(selector: str) -> Path | None:
    """The weights file a selector loads; None for random and DeepMind selectors, a missing file, or no
    model loader (torch)."""
    try:
        from blink.model.loading import resolve_selector

        path, _ = resolve_selector(selector)
    except (ImportError, ValueError, FileNotFoundError):
        return None
    return Path(path) if Path(path).is_file() else None


def weights_sha(selector: str) -> str | None:
    """The sha256 of the weights file the selector names, or None when there is no file to hash."""
    path = weights_file(selector)
    return sha256_file(path) if path is not None else None


def frozen_protocol(repo: Path, relative: str = "EVAL.md", tag: str = FROZEN_TAG) -> bytes | None:
    """EVAL.md as frozen under the tag, or None when the tag does not exist yet."""
    proc = subprocess.run(["git", "show", f"{tag}:{relative}"], cwd=repo, capture_output=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def check_protocol(protocol: Path, repo: Path | None = None) -> dict:
    """The protocol's sha256, and whether it equals the frozen copy (None before the freeze)."""
    protocol = Path(protocol)
    frozen = frozen_protocol(repo or protocol.resolve().parent, protocol.name)
    current = protocol.read_bytes()
    report = {"path": str(protocol), "sha256": hashlib.sha256(current).hexdigest(), "tag": FROZEN_TAG}
    if frozen is None:
        return {**report, "frozen": False, "matches": None}
    if frozen.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
        raise ProtocolMismatch(f"{protocol} differs from its copy under {FROZEN_TAG}: nothing runs")
    return {**report, "frozen": True, "matches": True, "frozen_sha256": hashlib.sha256(frozen).hexdigest()}


def live_training_runs(runs_root: Path | None = None) -> list[str]:
    from blink.train.status import list_runs

    return [run.name for run in list_runs(runs_root or paths.home() / "runs") if run.live]


def cpu_load(seconds: float = CPU_SAMPLE_S) -> float:
    """The machine's CPU use in percent over the next few seconds."""
    import psutil

    return float(psutil.cpu_percent(interval=seconds))


def guard_time_based(
    block_id: str,
    runs_root: Path | None = None,
    allow_busy: bool = False,
    load: Callable[[], float] = cpu_load,
) -> float | None:
    """Refuse a time-based block while training is live, or while the CPU is busy (unless allowed).

    Returns the CPU use measured at the start, for the block's report (None for a block with no clock)."""
    if not BLOCKS[block_id].time_based:
        return None
    live = live_training_runs(runs_root)
    if live:
        raise TrainingLive(f"{block_id} is time-based and training is live ({', '.join(live)}): not started")
    busy = load()
    if busy > BUSY_CPU_PCT and not allow_busy:
        raise MachineBusy(
            f"{block_id} is time-based and the CPU is {busy:.0f}% busy "
            f"(limit {BUSY_CPU_PCT:.0f}%): not started"
        )
    return busy


def guard_epsilon(block_id: str, results_dir: Path, played: float | None) -> float | None:
    """The epsilon this block's Blink plays with (None for a block that does not use E2b's choice).

    Refuses the block when results/epsilon.json no longer holds the value earlier blocks of this run
    played with: the final Elo pools E5, E6 and E7, and one Blink name must be one configuration."""
    if block_id not in EPSILON_BLOCKS:
        return None
    from blink.eval.match import read_epsilon

    now = read_epsilon(results_dir)
    if played is not None and now != played:
        raise EpsilonChanged(
            f"{block_id}: {Path(results_dir) / 'epsilon.json'} now holds epsilon {now!r}, but earlier blocks "
            f"of this run played with {played!r}: not started"
        )
    return now


def guard_play_mode(ctx: EvalContext, block_id: str, report: dict) -> dict:
    """`report`, read from <out>/<block>.json, refused when it was played in another fast mode than `ctx`:
    its Blink games carry that mode's names. A report without the fields predates the modes (fp32)."""
    played = {
        "precision": report.get("precision", fastmode.DEFAULT_PRECISION),
        "compile": bool(report.get("compile")),
    }
    if block_id in MODE_FREE_BLOCKS or played == ctx.play_mode:
        return report
    raise PlayModeChanged(
        f"{block_id}: {Path(ctx.out_dir) / f'{block_id}.json'} was played {fastmode.describe(**played)}, but "
        f"this run plays {fastmode.describe(**ctx.play_mode)}: pass the mode it recorded, or another --out"
    )


def guard_weights(block_id: str, selector: str, pinned: str | None) -> None:
    """Refuse the block when the weights file no longer hashes to the sha pinned at the run's start: one
    Blink name must be one weights file across the multi-day run (say ship/blink.pt was replaced)."""
    if pinned is None:
        return
    now = weights_sha(selector)
    if now != pinned:
        raise WeightsChanged(
            f"{block_id}: the weights of {selector} now hash to {now or 'nothing'}, but earlier blocks of "
            f"this run played {pinned}: not started"
        )


# ------------------------------------------------------------------------------ forfeits and adjudications


def forfeit_table(pgns: Iterable[Path]) -> dict[str, dict[str, int]]:
    """Per engine: games, time forfeits, other forfeits (illegal move, crash, stall) and adjudications."""
    table: dict[str, Counter] = {}
    for path in pgns:
        if not Path(path).is_file():
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            while (headers := chess.pgn.read_headers(handle)) is not None:
                _count_game(table, headers)
    return {engine: dict(counts) for engine, counts in sorted(table.items())}


def _count_game(table: dict[str, Counter], headers: chess.pgn.Headers) -> None:
    white, black = headers.get("White", "?"), headers.get("Black", "?")
    termination, result = headers.get("Termination", "").lower(), headers.get("Result", "*")
    for engine in (white, black):
        counts = table.setdefault(
            engine, Counter({"games": 0, "time_forfeits": 0, "forfeits": 0, "adjudications": 0})
        )
        counts["games"] += 1
        counts["adjudications"] += termination == "adjudication"
    if result in ("1-0", "0-1") and termination not in ("", "normal", "adjudication"):
        loser = black if result == "1-0" else white
        key = "time_forfeits" if termination == "time forfeit" else "forfeits"
        table[loser][key] += 1


# ------------------------------------------------------------------------------ no-search audit, gates

AUDIT_KEYS = ("games", "decisions", "compliant", "missing_counts", "max_rows", "max_legal")


def audit_block(pgns: Sequence[Path], players: Iterable[str]) -> dict[str, dict]:
    """The no-search audit of every searchless player seated in these PGNs, each by its exact name."""
    from blink.eval import nosearch

    files = [Path(p) for p in pgns if Path(p).is_file()]
    return nosearch.audit_each(files, sorted(p for p in players if nosearch.is_searchless(p)))


def _audit_summary(audit: dict) -> dict:
    return {**{key: audit[key] for key in AUDIT_KEYS}, "violations": len(audit["violations"])}


SF_FORFEIT_BLOCKS = ("E0", "E8")  # the plan's done-when: E0 and E8 also fail on any Stockfish forfeit


def _selfcheck_failure(report: dict) -> list[str]:
    check = report.get("sf_selfcheck") or {}
    if check.get("passed") is not False:
        return []
    return [
        f"E0: the SF self-check failed (score {check.get('score')}, band 50% +- {SELFCHECK_BAND:.0%}, "
        f"SF forfeits {check.get('sf_forfeits') or 'none'}): the anchors fall back to 60+0.6 and E5 to the "
        "shipped mode only; record the slip in STATUS"
    ]


def _forfeit_failures(block_id: str, forfeits: dict) -> list[str]:
    failures = []
    for engine, counts in forfeits.items():
        on_time, other = counts.get("time_forfeits", 0), counts.get("forfeits", 0)
        if not on_time + other:
            continue
        if engine.startswith("Blink"):
            failures.append(
                f"{block_id}: {engine} lost {on_time} games on time and {other} to an illegal move or "
                "a crash (Blink forfeits must be 0)"
            )
        elif engine.startswith("SF") and block_id in SF_FORFEIT_BLOCKS:
            failures.append(f"{block_id}: {engine} forfeited {on_time} games on time and {other} otherwise")
    return failures


def gate_failures(state: dict) -> list[str]:
    """The plan's done-when gates this run failed (P8), one line each; empty when every gate holds."""
    failures = []
    for block_id in BLOCK_ORDER:
        report = state.get(block_id) or {}
        if block_id == "E0":
            failures += _selfcheck_failure(report)
        failures += _forfeit_failures(block_id, report.get("forfeits") or {})
        for player, audit in (report.get("nosearch") or {}).items():
            if not audit["compliant"]:
                failures.append(
                    f"{block_id}: the no-search audit of {player} is not compliant "
                    f"({audit['violations']} violations in {audit['decisions']} decisions)"
                )
    return failures


# ------------------------------------------------------------------------------ running blocks

Runner = Callable[[EvalContext, dict], dict]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)
    return path


def run_blocks(
    ctx: EvalContext,
    runners: Mapping[str, Runner],
    only: Sequence[str] | None = None,
    runs_root: Path | None = None,
    log: Callable[[str], None] = print,
    load: Callable[[], float] = cpu_load,
    processes: Callable[[], list[list[str]]] = training_processes,
) -> dict:
    """Run the chosen blocks in the plan's order; returns every block's report and its forfeit table.
    `processes` lists the live Blink processes' arguments, for each block's Stockfish budget."""
    ids = [b for b in BLOCK_ORDER if only is None or b in only]
    unknown = sorted(set(only or ()) - set(BLOCK_ORDER))
    if unknown:
        raise ValueError(f"unknown blocks {unknown}; the blocks are {', '.join(BLOCK_ORDER)}")
    for block_id in (b for b in BLOCK_ORDER if b not in ids):  # one --out folder, one fast mode
        earlier_report(ctx, {}, block_id)
    protocol = check_protocol(ctx.protocol)
    log(game_table(ids, ctx.games))
    pinned = weights_sha(ctx.model)
    log(f"weights of {ctx.model}: sha256 {pinned}" if pinned else f"{ctx.model}: no weights file to pin")
    state: dict = {"protocol": protocol, "started": _now(), "weights_sha": pinned}
    for block_id in ids:
        busy = guard_time_based(block_id, runs_root, ctx.allow_busy_cpu, load)
        epsilon = guard_epsilon(block_id, ctx.results_dir, state.get("epsilon"))
        guard_weights(block_id, ctx.model, pinned)
        log(f"{block_id}: {BLOCKS[block_id].title}")
        report = runners[block_id](budgeted(ctx, log, processes), state)
        pgns = [Path(p) for p in report.get("pgns", [])]
        forfeits = forfeit_table(pgns)
        audits = audit_block(pgns, forfeits)
        if audits:
            _write_json(ctx.out_dir / f"{block_id}.nosearch.json", audits)
        report = {
            **report,
            "forfeits": forfeits,
            "nosearch": {player: _audit_summary(audit) for player, audit in audits.items()},
            "cpu_pct_at_start": busy,
            "epsilon": epsilon,
            "weights_sha": pinned,
            **ctx.play_mode,
        }
        state[block_id] = report
        if epsilon is not None:
            state["epsilon"] = epsilon
        _write_json(ctx.out_dir / f"{block_id}.json", report)
        log(f"{block_id}: {report.get('games', 0):,} games, forfeits {report['forfeits'] or '{}'}")
    state["gate_failures"] = gate_failures(state)
    for line in state["gate_failures"]:
        log(f"done-when gate failed: {line}")
    return state


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def earlier_report(ctx: EvalContext, state: dict, block_id: str) -> dict:
    """A block's report from this run, or from <out>/<block>.json when an earlier run wrote it into the
    same --out folder (a block run alone), refused when played in another fast mode; {} when neither."""
    if state.get(block_id):
        return state[block_id]
    path = Path(ctx.out_dir) / f"{block_id}.json"
    if not path.is_file():
        return {}
    return guard_play_mode(ctx, block_id, json.loads(path.read_text(encoding="utf-8")))


def shipped_mode(ctx: EvalContext, state: dict) -> str:
    """The mode E3 chose, or the one given; a block that needs it refuses to guess."""
    chosen = (state.get("E3") or {}).get("mode") or ctx.mode
    if chosen is None:
        raise ValueError("the shipped mode is unknown: run E3 first or pass --mode policy|value")
    return chosen


# ------------------------------------------------------------------------------ E0, E1, E2, E3


def dm_puzzle_check(done: dict, expected: tuple[float, float] = DM_EXPECTED) -> dict:
    """E0 (2): DM-9M `params` against its pre-registered expectation; a miss calls for the G6 audit."""
    low, high = expected[0] - expected[1], expected[0] + expected[1]
    pct = 100 * done["accuracy"]
    return {
        "pct": pct,
        "expected": list(expected),
        "in_band": low <= pct <= high,
        "g6_needed": not low <= pct <= high,
    }


def selfcheck_verdict(report: dict) -> dict:
    """E0 (3): SF at st=0.1 scores 50% +- 7 against itself at the slow control, with no forfeit at all."""
    forfeits = (report.get("audit") or {}).get("forfeits") or {}
    within = report["score"] is not None and abs(report["score"] - 0.5) <= SELFCHECK_BAND
    return {
        "band": SELFCHECK_BAND,
        "within_band": within,
        "sf_forfeits": forfeits,
        "passed": within and not forfeits,
    }


def _dm_puzzles(ctx: EvalContext) -> dict:
    from blink.eval import puzzles
    from blink.play.factory import ModelUnavailable
    from blink.reference import registry

    out: dict = {}
    for selector in ("dm:9M", "dm:9M:ema"):
        try:
            agent = registry.load_agent(selector, device=ctx.device)
        except (ModelUnavailable, ImportError) as exc:
            out[selector] = {"skipped": str(exc)}
            continue
        label = f"dm10k_{selector.replace(':', '_')}"
        done = puzzles.run_puzzle_set(
            puzzles.resolve_set("dm10k"), agent, registry.MODE, ctx.out_dir / "E0", ctx.positions, label
        )
        out[selector] = {**done, **(dm_puzzle_check(done) if selector == "dm:9M" else {})}
    return out


def e0_block(ctx: EvalContext, state: dict) -> dict:
    """DM-9M (params, and params_ema when converted) on DeepMind's puzzles, and SF st=0.1 against itself
    at the calibration time control."""
    from blink.eval import fastchess

    out: dict = {"port_logits": "checked by test_the_dm_port_matches_saved_jax_logits (reference area)"}
    puzzles_by_selector = _dm_puzzles(ctx)
    out["dm_puzzles"] = puzzles_by_selector.get("dm:9M", {})
    out["dm_puzzles_ema"] = puzzles_by_selector.get("dm:9M:ema", {})
    exe = fastchess.stockfish_exe()
    quick = fastchess.stockfish_anchor(SELFCHECK_ANCHOR, exe)
    slow = fastchess.with_tc(fastchess.stockfish_anchor(SELFCHECK_ANCHOR, exe), ctx.selfcheck_tc)
    slow = replace(slow, name=f"SF{SELFCHECK_ANCHOR}-slow")
    games = ctx.n(BLOCKS["E0"].games)
    pair = fastchess.prepare_pair(quick, slow, games, "dev", ctx.out_dir / "E0", ctx.concurrency)
    report = fastchess.match_report(fastchess.execute(pair))
    out["sf_selfcheck"] = {**report, "slow_tc": ctx.selfcheck_tc, **selfcheck_verdict(report)}
    return {**out, "games": report["games"], "pgns": [report["pgn"]]}


def _pack_file(data_dir: Path, split: str) -> Path | None:
    """<split>_roots.bin (the v1 pack) or <split>.bin (the skeleton pack), whichever exists."""
    for name in (f"{split}_roots.bin", f"{split}.bin"):
        if (data_dir / name).is_file():
            return data_dir / name
    return None


def _scored_at(csv_path: Path, json_path: Path, mode: str, epsilon: float | None) -> bool:
    """A `blink eval puzzles` CSV E2 may use: policy mode always, value mode only at E2's epsilon."""
    if not csv_path.is_file():
        return False
    if mode != "value" or epsilon is None:
        return True
    recorded = (
        json.loads(json_path.read_text(encoding="utf-8")).get("epsilon") if json_path.is_file() else None
    )
    return recorded == epsilon


def static_inputs(ctx: EvalContext, label: str, epsilon: float | None = None):
    """E2's inputs: the pack's val and test roots and mateset, games10k, and the puzzle CSVs on disk
    (a value-mode CSV only when it was scored at `epsilon`, the epsilon E2's value agent plays)."""
    from blink.eval import puzzles, static

    data = ctx.data_dir or paths.home() / "data" / "v1"
    test_iid = _pack_file(data, "test_iid")
    if test_iid is None:
        raise FileNotFoundError(f"no test_iid roots in {data}")
    home = paths.home()
    folder = home / "eval" / "puzzles"
    found = {m: puzzles.output_paths(folder, f"dm10k_{label}", m) for m in ("policy", "value")}
    csvs = [(m, paths_[0]) for m, paths_ in found.items() if _scored_at(*paths_, m, epsilon)]
    optional = [data / "mateset.npz", home / "data" / "games10k.npy", home / "eval" / "lichess_bands.csv"]
    mateset, games10k, bands = (p if p.is_file() else None for p in optional)
    return static.StaticInputs(
        test_iid=test_iid,
        val=_pack_file(data, "val"),
        test_grouped=_pack_file(data, "test_grouped"),
        games10k=games10k,
        mateset=mateset,
        dm_puzzles=tuple(csvs),
        lichess_bands=bands,
    )


def static_limits(ctx: EvalContext):
    from blink.eval import static

    if ctx.positions is None:
        return static.FULL_LIMITS
    n = ctx.positions
    return static.StaticLimits(n, n, n, n, n, max(1, n // 12))


def e2_block(ctx: EvalContext, state: dict) -> dict:
    """E2 runs before E2b chooses epsilon: its value-mode puzzle numbers record the epsilon they used
    (value_epsilon), and results.json leaves them out if the shipped epsilon turns out different."""
    from blink.eval import fastchess, match, static
    from blink.eval.sflabel import SfLabeler

    epsilon = match.read_epsilon(ctx.results_dir)
    agents = match.blink_agents(ctx.model, ctx.device, epsilon=epsilon, **ctx.play_mode)
    label = fastchess.model_tag(ctx.model) + ctx.mode_tag  # as `blink eval puzzles` labels its files
    inputs, limits = static_inputs(ctx, label, epsilon), static_limits(ctx)
    with SfLabeler(1_000_000, exe=fastchess.stockfish_exe(), procs=ctx.sf_procs) as labeler:
        e2 = static.run_e2(agents["policy"].evaluator, agents, inputs, limits, labeler)
    rows = static.diagnostics_rows(e2, f"Blink-{label}")
    return {
        "e2": e2,
        "diagnostics": [r.__dict__ for r in rows],
        "value_epsilon": epsilon,
        "games": 0,
        "pgns": [],
    }


def _film_row(ctx: EvalContext, path: Path, val: list, probe: dict | None) -> dict:
    from blink.eval import puzzles, static
    from blink.play import factory

    evaluator = factory.load_evaluator(str(path), device=ctx.device)
    row = {
        "frame": path.name,
        "val_top1": static.summarize(static.evaluate_roots(evaluator, val))["top1"]["value"],
    }
    if probe is not None:
        row["vaa"] = static.mate_rates(evaluator, probe, None, ctx.positions)["value"]["shortest"]["value"]
    for mode in factory.MODES:
        agent = factory.make_agent(mode, evaluator)
        done = puzzles.run_puzzle_set(
            puzzles.resolve_set("dm10k"), agent, mode, ctx.out_dir / "E1", ctx.positions, path.stem
        )
        row[f"puzzles_{mode}"] = done["accuracy"]
    return row


def load_valprobe(ctx: EvalContext) -> dict | None:
    """The pack's valprobe.npz arrays (E1's frames and E6's rungs score VAA on it), or None without one."""
    path = (ctx.data_dir or paths.home() / "data" / "v1") / "valprobe.npz"
    if not path.is_file():
        return None
    with np.load(path) as arrays:
        return {k: arrays[k] for k in arrays.files}


def e1_block(ctx: EvalContext, state: dict) -> dict:
    """The 21 film frames, static only: val policy top-1, valprobe VAA and puzzles in both modes."""
    from blink.eval import ladder, static

    if not ctx.film_run:
        return {"skipped": "no --film-run given", "games": 0, "pgns": []}
    data = ctx.data_dir or paths.home() / "data" / "v1"
    val = static.roots_from_records(static.read_roots(_pack_file(data, "val"), ctx.positions or 50_000))
    probe = load_valprobe(ctx)
    frames = [_film_row(ctx, p, val, probe) for p in ladder.film_frames(ladder.film_run_dir(ctx.film_run))]
    film = _write_json(ctx.out_dir / "film.json", {"run": ctx.film_run, "frames": frames})
    return {"frames": frames, "film_json": str(film), "games": 0, "pgns": []}


def e3_block(ctx: EvalContext, state: dict) -> dict:
    """The pre-registered mode SPRT, in process with one model load, on the dev slice."""
    from blink.eval import books, match, sprt

    agents = match.blink_agents(ctx.model, ctx.device, results_dir=ctx.results_dir, **ctx.play_mode)
    config = sprt.SprtConfig(cap_games=ctx.n(sprt.MODE_SPRT.cap_games))
    pairs = config.cap_games // 2
    openings = books.openings_for("dev", 2 * pairs)
    forward_pgn, reverse_pgn = (  # fresh files: a re-run into the same --out never appends
        match.unique_path(ctx.out_dir / "E3" / "value_vs_policy.pgn"),
        match.unique_path(ctx.out_dir / "E3" / "policy_vs_value.pgn"),
    )
    choice = sprt.run_mode_choice(
        match.pair_player(agents["value"], agents["policy"], openings[:pairs], forward_pgn),
        match.pair_player(agents["policy"], agents["value"], openings[pairs:], reverse_pgn),
        config,
    )
    games = choice.forward.games + (choice.reverse.games if choice.reverse else 0)
    pgns = [str(forward_pgn)] + ([str(reverse_pgn)] if choice.reverse else [])
    return {"choice": choice.as_dict(), "mode": choice.mode, "games": games, "pgns": pgns}


def default_runners() -> dict[str, Runner]:
    """Every block's runner, imported only when a run starts (the modules pull in torch and engines)."""
    from blink.eval import anchors, conversion, failures, ladder

    return {
        "E0": e0_block,
        "E1": e1_block,
        "E2": e2_block,
        "E2b": conversion.e2b_block,
        "E3": e3_block,
        "E4": ladder.e4_block,
        "E4b": ladder.e4b_block,
        "E5": anchors.e5_block,
        "E6": ladder.e6_block,
        "E7": anchors.e7_block,
        "E8": conversion.e8_block,
        "E9": failures.e9_block,
    }


def _fit(pgns: Sequence[Path], ctx: EvalContext, ordo: Callable) -> tuple[object, str | None]:
    """Ordo over the final-slice PGNs; when Ordo refuses the pool, the tally alone (no Elo) and why."""
    from blink.eval import rating

    if not pgns:
        return None, None
    anchors = rating.read_anchors()
    smoke = ctx.games is not None
    simulations = SMOKE_ORDO_SIMULATIONS if smoke else rating.ORDO_SIMULATIONS
    timeout_s = SMOKE_ORDO_TIMEOUT_S if smoke else rating.ORDO_TIMEOUT_S
    try:
        return ordo(pgns, anchors, ctx.out_dir / "ordo", simulations=simulations, timeout_s=timeout_s), None
    except RuntimeError as exc:
        tally, left_out = rating.tally_players(pgns), rating.exclusions(pgns, anchors)
        return rating.OrdoFit((), (), left_out, tally, (), {}), str(exc)


def run_all(
    ctx: EvalContext,
    only: Sequence[str] | None = None,
    runners: Mapping[str, Runner] | None = None,
    runs_root: Path | None = None,
    log: Callable[[str], None] = print,
    ordo: Callable | None = None,
    load: Callable[[], float] = cpu_load,
) -> dict:
    """The blocks, then Ordo over the final-slice PGNs, then results/results.json (schema v1) and
    results/nosearch.json (blink.eval.publish); what either leaves out is listed in `notes`."""
    from blink.eval import rating
    from blink.report.results_schema import to_json

    state = run_blocks(ctx, runners or default_runners(), only, runs_root, log, load)
    pgns = [Path(p) for p in final_slice_pgns(state) if Path(p).is_file()]
    fit, ordo_error = _fit(pgns, ctx, ordo or rating.run_ordo)
    listing = write_pgn_list(pgns, ctx.results_dir / FINAL_SLICE_LIST)
    audits = public_audit(ctx, state)
    notes: list[str] = []
    results = build_results(state, ctx, fit, listing, notes, audits)
    path = ctx.results_dir / "results.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(to_json(results) + "\n")
    summary = {
        "results": str(path),
        "final_slice_pgns": str(listing),
        "ordo": fit.as_dict() if fit is not None else None,
        "ordo_error": ordo_error,
        "forfeits": {block: state[block]["forfeits"] for block in BLOCK_ORDER if block in state},
        "gate_failures": state["gate_failures"],
        "games": {block: state[block].get("games", 0) for block in BLOCK_ORDER if block in state},
        "notes": notes,
    }
    _write_json(ctx.out_dir / "summary.json", summary)
    for note in notes:
        log(f"results.json: {note}")
    log(f"results: {path}")
    return {"state": state, **summary}
