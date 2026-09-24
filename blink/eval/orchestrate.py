"""`blink eval all --model ship --protocol EVAL.md`: the P8 blocks in the plan's order, each runnable alone.

Before anything runs: the per-block game-count table is printed; when the annotated tag eval-v1-frozen
exists, EVAL.md must be byte-identical to the tagged copy (else nothing runs); and a time-based block
refuses to start while any Blink training run's heartbeat is live (a busy GPU or CPU would bend both the
clocks and the anchors). After each block its report goes to <out>/<block>.json with the time forfeits and
adjudications of every engine in its PGNs; at the end results/results.json is written through
blink.report.results_schema (Ordo over the final-slice PGNs for the Elo column, E2 for the diagnostics).

Where the games are played: the fastchess blocks are the ones against clocked UCI_Elo anchors (E0's SF
self-check, E5 and DM-9M's E7 gauntlet), at concurrency 5 as in the plan. Every other match runs in process
with one model load (one CUDA context, PF58): Blink needs no clock there because its compute never depends
on time (N4), and Stockfish gets `go movetime 100` (st=0.1) or `go nodes N` with the whole game.
A `games` override makes every match that long (and every SPRT cap), for smoke runs.
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

BLOCK_ORDER = ("E0", "E1", "E2", "E2b", "E3", "E4", "E4b", "E5", "E6", "E7", "E8", "E9")
FROZEN_TAG = "eval-v1-frozen"
SELFCHECK_ANCHOR = 1800
SELFCHECK_TC = "120+1"
SELFCHECK_BAND = 0.07
DM_EXPECTED = (88.9, 1.0)  # DM-9M `params` on the 10K puzzles, % (arXiv v2 Table 1), pre-registered
# A loaded machine bends st=0.1: in a CPU-busy smoke, SF19 forfeited 4 of 24 E5 games on time.
BUSY_CPU_PCT = 25.0
CPU_SAMPLE_S = 3.0


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
    sf_procs: int = 1  # Stockfish processes for SF19 labels (E2 regret, E9)
    allow_busy_cpu: bool = False  # smoke runs only: start time-based blocks on a busy machine

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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
) -> dict:
    """Run the chosen blocks in the plan's order; returns every block's report and its forfeit table."""
    ids = [b for b in BLOCK_ORDER if only is None or b in only]
    unknown = sorted(set(only or ()) - set(BLOCK_ORDER))
    if unknown:
        raise ValueError(f"unknown blocks {unknown}; the blocks are {', '.join(BLOCK_ORDER)}")
    protocol = check_protocol(ctx.protocol)
    log(game_table(ids, ctx.games))
    state: dict = {"protocol": protocol, "started": _now()}
    for block_id in ids:
        busy = guard_time_based(block_id, runs_root, ctx.allow_busy_cpu, load)
        log(f"{block_id}: {BLOCKS[block_id].title}")
        report = runners[block_id](ctx, state)
        forfeits = forfeit_table(Path(p) for p in report.get("pgns", []))
        report = {**report, "forfeits": forfeits, "cpu_pct_at_start": busy}
        state[block_id] = report
        _write_json(ctx.out_dir / f"{block_id}.json", report)
        log(f"{block_id}: {report.get('games', 0):,} games, forfeits {report['forfeits'] or '{}'}")
    return state


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


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


def static_inputs(ctx: EvalContext, label: str):
    """E2's inputs: the pack's val and test roots and mateset, games10k, and any puzzle CSVs on disk."""
    from blink.eval import static

    data = ctx.data_dir or paths.home() / "data" / "v1"
    test_iid = _pack_file(data, "test_iid")
    if test_iid is None:
        raise FileNotFoundError(f"no test_iid roots in {data}")
    home = paths.home()
    csvs = [(m, home / "eval" / "puzzles" / f"puzzles_dm10k_{label}_{m}.csv") for m in ("policy", "value")]
    optional = [data / "mateset.npz", home / "data" / "games10k.npy", home / "eval" / "lichess_bands.csv"]
    mateset, games10k, bands = (p if p.is_file() else None for p in optional)
    return static.StaticInputs(
        test_iid=test_iid,
        val=_pack_file(data, "val"),
        test_grouped=_pack_file(data, "test_grouped"),
        games10k=games10k,
        mateset=mateset,
        dm_puzzles=tuple((m, p) for m, p in csvs if p.is_file()),
        lichess_bands=bands,
    )


def static_limits(ctx: EvalContext):
    from blink.eval import static

    if ctx.positions is None:
        return static.FULL_LIMITS
    n = ctx.positions
    return static.StaticLimits(n, n, n, n, n, max(1, n // 12))


def e2_block(ctx: EvalContext, state: dict) -> dict:
    from blink.eval import fastchess, match, static
    from blink.eval.sflabel import SfLabeler

    agents = match.blink_agents(ctx.model, ctx.device)
    label = fastchess.NAME_UNSAFE.sub("_", ctx.model).strip("_")
    inputs, limits = static_inputs(ctx, label), static_limits(ctx)
    with SfLabeler(1_000_000, exe=fastchess.stockfish_exe(), procs=ctx.sf_procs) as labeler:
        e2 = static.run_e2(agents["policy"].evaluator, agents, inputs, limits, labeler)
    rows = static.diagnostics_rows(e2, f"Blink-{label}")
    return {"e2": e2, "diagnostics": [r.__dict__ for r in rows], "games": 0, "pgns": []}


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


def e1_block(ctx: EvalContext, state: dict) -> dict:
    """The 21 film frames, static only: val policy top-1, valprobe VAA and puzzles in both modes."""
    from blink.eval import ladder, static

    if not ctx.film_run:
        return {"skipped": "no --film-run given", "games": 0, "pgns": []}
    data = ctx.data_dir or paths.home() / "data" / "v1"
    val = static.roots_from_records(static.read_roots(_pack_file(data, "val"), ctx.positions or 50_000))
    probe = None
    if (data / "valprobe.npz").is_file():
        with np.load(data / "valprobe.npz") as arrays:
            probe = {k: arrays[k] for k in arrays.files}
    frames = [_film_row(ctx, p, val, probe) for p in ladder.film_frames(ladder.film_run_dir(ctx.film_run))]
    film = _write_json(ctx.out_dir / "film.json", {"run": ctx.film_run, "frames": frames})
    return {"frames": frames, "film_json": str(film), "games": 0, "pgns": []}


def e3_block(ctx: EvalContext, state: dict) -> dict:
    """The pre-registered mode SPRT, in process with one model load, on the dev slice."""
    from blink.eval import books, match, sprt

    agents = match.blink_agents(ctx.model, ctx.device)
    config = sprt.SprtConfig(cap_games=ctx.n(sprt.MODE_SPRT.cap_games))
    pairs = config.cap_games // 2
    openings = books.openings_for("dev", 2 * pairs)
    forward_pgn, reverse_pgn = (
        ctx.out_dir / "E3" / "value_vs_policy.pgn",
        ctx.out_dir / "E3" / "policy_vs_value.pgn",
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


# ------------------------------------------------------------------------------ results/results.json


def row_kind(agent: str) -> str:
    if agent.startswith("Blink"):
        return "blink"
    if agent.startswith("DM-"):
        return "reference"
    if agent[:2] == "SF" and agent[2:].isdigit():
        return "anchor"
    return "ladder"


def final_slice_pgns(state: dict) -> list[str]:
    """The games Ordo rates: E5's anchors and side rows, the E6 ladder and E7 (all on the final slice)."""
    e5, e6, e7 = (state.get(b) or {} for b in ("E5", "E6", "E7"))
    return [*e5.get("final_slice_pgns", []), *e6.get("pgns", []), *e7.get("final_slice_pgns", [])]


def _blink_puzzles(agent: str) -> dict:
    """What `blink eval puzzles --model <m>` wrote for a Blink-<mode>-<model> agent, if it ran."""
    _, mode, tag = agent.split("-", 2)
    path = paths.home() / "eval" / "puzzles" / f"puzzles_dm10k_{tag}_{mode}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _puzzle_fields(agent: str, state: dict) -> dict:
    """DeepMind's 10K puzzles for a row: DM-9M from E0, a Blink agent from `blink eval puzzles`."""
    kind = row_kind(agent)
    if kind == "reference":
        done = (state.get("E0") or {}).get("dm_puzzles") or {}
    elif kind == "blink" and agent.count("-") >= 2:
        done = _blink_puzzles(agent)
    else:
        return {}
    if "accuracy" not in done:
        return {}
    low, high = done["wilson95"]
    return {"dm_puzzles_pct": 100 * done["accuracy"], "dm_puzzles_ci": (100 * low, 100 * high)}


def strength_rows(fit, state: dict, shipped: str | None) -> tuple:
    from blink.report.results_schema import StrengthRow

    rows = []
    crossover = ((state.get("E4") or {}).get("crossover") or {}).get("nodes")
    for agent in sorted(fit.tally):
        kind = row_kind(agent)
        fitted = next((r for r in fit.rows if r.player == agent and not r.is_anchor), None)
        fields = (
            {"elo": fitted.rating, "elo_ci95": fitted.error, "elo_games": fitted.played} if fitted else {}
        )
        rows.append(
            StrengthRow(
                agent=agent,
                kind=kind,
                reproduce="uv run blink rate --pgn <eval out>/E5 --anchors configs/anchors.csv",
                sf_nodes_equiv=crossover if agent == shipped else None,
                **fields,
                **_puzzle_fields(agent, state),
            )
        )
    return tuple(rows)


def diagnostics_rows(state: dict, shipped_mode_name: str | None) -> tuple:
    from blink.report.results_schema import DiagnosticsRow

    e8 = state.get("E8") or {}
    conversion = (e8.get("rules_on") or {}).get("pct")
    rows = []
    for row in (state.get("E2") or {}).get("diagnostics", []):
        ci = row.get("puzzle_rating_ci")
        extra = {"conversion_pct": conversion} if row["mode"] == shipped_mode_name else {}
        rows.append(DiagnosticsRow(**{**row, "puzzle_rating_ci": tuple(ci) if ci else None, **extra}))
    return tuple(rows)


def weights_sha(selector: str) -> str:
    """The sha256 of the weights file the selector names, or a note saying why there is none."""
    try:
        from blink.model.loading import resolve_selector

        path, _ = resolve_selector(selector)
    except (ImportError, ValueError, FileNotFoundError) as exc:
        return f"unknown: {exc}"
    return sha256_file(path) if Path(path).is_file() else f"unknown: no file at {path}"


def build_results(state: dict, ctx: EvalContext, fit) -> object:
    from blink.eval.fastchess import engine_name
    from blink.report.results_schema import Results, Shipped

    mode = (state.get("E3") or {}).get("mode") or ctx.mode
    shipped_name = engine_name(ctx.model, mode) if mode else None
    shipped = Shipped(shipped_name, mode, weights_sha(ctx.model)) if mode else None
    return Results(
        strength=strength_rows(fit, state, shipped_name) if fit is not None else (),
        diagnostics=diagnostics_rows(state, mode),
        shipped=shipped,
        eval_md_sha=state["protocol"]["sha256"],
        generated_at=_now(),
    )


def run_all(
    ctx: EvalContext,
    only: Sequence[str] | None = None,
    runners: Mapping[str, Runner] | None = None,
    runs_root: Path | None = None,
    log: Callable[[str], None] = print,
    ordo: Callable | None = None,
    load: Callable[[], float] = cpu_load,
) -> dict:
    """The blocks, then Ordo over the final-slice PGNs, then results/results.json (schema v1)."""
    from blink.eval import rating
    from blink.report.results_schema import to_json

    state = run_blocks(ctx, runners or default_runners(), only, runs_root, log, load)
    pgns = [Path(p) for p in final_slice_pgns(state) if Path(p).is_file()]
    fit = (ordo or rating.run_ordo)(pgns, rating.read_anchors(), ctx.out_dir / "ordo") if pgns else None
    results = build_results(state, ctx, fit)
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.results_dir / "results.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(to_json(results) + "\n")
    summary = {
        "results": str(path),
        "ordo": fit.as_dict() if fit is not None else None,
        "forfeits": {block: state[block]["forfeits"] for block in BLOCK_ORDER if block in state},
        "games": {block: state[block].get("games", 0) for block in BLOCK_ORDER if block in state},
    }
    _write_json(ctx.out_dir / "summary.json", summary)
    log(f"results: {path}")
    return {"state": state, **summary}
