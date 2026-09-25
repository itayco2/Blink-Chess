"""What `blink eval all` publishes: results/results.json and results/nosearch.json (plan P8).

results.json goes through blink.report.results_schema: Ordo over the final-slice PGNs for the Elo column,
E2 and E6's rungs for the diagnostics, E0 and `blink eval puzzles` for DeepMind's puzzles, and the
shipped model with the sha pinned when the run started, and the epsilon and fast play mode (precision,
compile) its rated games used: the Lichess bot's check-config holds the bot to both. Every Blink
row also carries what it cost: parameters, positions seen, distinct training positions and GPU-hours
from its training run (runs/<name>/config.json, metrics.jsonl and the pack manifest; `ship` reads the
flagship named by --film-run), and rows and milliseconds per move from its own public moves.

nosearch.json is one exact-name audit of every searchless player over every PGN the blocks wrote, with
the list of PGNs it covers: the README's no-search box. Anything left out is said in the run's notes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from blink import paths

if TYPE_CHECKING:
    from blink.eval.orchestrate import EvalContext

FINAL_SLICE_LIST = "final_slice_pgns.txt"  # next to results.json: the exact PGNs its Elo was fitted on
NOSEARCH_FILE = "nosearch.json"  # next to results.json: the audit of every public move
_WEIGHTS_STEP = re.compile(r"^(?:ckpt|frame)_(\d+)\.pt$")  # checkpoint and film-frame names carry their step


def _orchestrate():
    """The orchestrator, imported when needed (it imports this module)."""
    from blink.eval import orchestrate

    return orchestrate


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
    """What `blink eval puzzles --model <m>` wrote for a Blink-<mode>-<tag> agent, if it ran (the CLI
    labels its files with the same fastchess.model_tag the engine name carries)."""
    from blink.eval import puzzles

    _, mode, tag = agent.split("-", 2)
    _, path = puzzles.output_paths(paths.home() / "eval" / "puzzles", f"dm10k_{tag}", mode)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _puzzle_fields(
    agent: str, state: dict, epsilon: float | None = None, notes: list[str] | None = None
) -> dict:
    """DeepMind's 10K puzzles for a row: DM-9M from E0, a Blink agent from `blink eval puzzles`.

    A value-mode score counts only when it was measured at `epsilon`, the one the rated games used."""
    kind = row_kind(agent)
    if kind == "reference":
        done = (state.get("E0") or {}).get("dm_puzzles") or {}
    elif kind == "blink" and agent.count("-") >= 2:
        done = _blink_puzzles(agent)
        stale = agent.split("-", 2)[1] == "value" and epsilon is not None and done.get("epsilon") != epsilon
        if done and stale:
            _note(notes, f"{agent}: its puzzles were scored at epsilon {done.get('epsilon')!r}, the rated "
                  f"games at {epsilon!r}; left out (rerun `blink eval puzzles` after E2b)")  # fmt: skip
            return {}
    else:
        return {}
    if "accuracy" not in done:
        return {}
    low, high = done["wilson95"]
    return {"dm_puzzles_pct": 100 * done["accuracy"], "dm_puzzles_ci": (100 * low, 100 * high)}


def reproduce_command(listing: Path) -> str:
    return f"uv run blink rate --pgn-list {Path(listing).as_posix()} --anchors configs/anchors.csv"


def _note(notes: list[str] | None, text: str) -> None:
    if notes is not None:
        notes.append(text)


PLAY_COSTS = ("evals_per_move_median", "evals_per_move_max", "ms_per_move_p50")


def play_costs(audit: dict | None) -> dict:
    """Rows and milliseconds per move, from a player's own public moves (its no-search audit)."""
    return {key: audit[key] for key in PLAY_COSTS if (audit or {}).get(key) is not None}


def run_dir_of(selector: str, ctx: EvalContext) -> Path | None:
    """The training run behind a selector: run:<name>, a path under runs/<name>/, or, for the shipped
    weights (ship, release:<tag>), the flagship run named by --film-run."""
    runs = paths.home() / "runs"
    kind, _, rest = selector.partition(":")
    if kind == "run":
        return runs / rest.partition(":")[0]
    if selector == "ship" or kind == "release":
        return runs / ctx.film_run if ctx.film_run else None
    parts = Path(selector).parts
    if "runs" in parts[:-1]:
        return runs / parts[parts.index("runs") + 1]
    return None


def _trained_steps(selector: str, usage) -> int | None:
    """The step of the weights: a checkpoint or film frame names it, else the run's last logged step."""
    from blink.report.compute import RunCompute

    path = _orchestrate().weights_file(selector)
    found = _WEIGHTS_STEP.match(path.name) if path is not None else None
    if found:
        return int(found.group(1))
    return usage.steps if isinstance(usage, RunCompute) else None


def model_facts(selector: str, ctx: EvalContext) -> dict:
    """params_total and params_non_gab (the trainer's parameter_report), positions_seen (steps x batch
    rows), training_positions (distinct train roots, results_schema.training_positions) and gpu_hours
    (blink.report.compute) for one Blink model, from its run's own files; {} without a run to read."""
    from blink.report import compute
    from blink.report import results_schema as rs

    run = run_dir_of(selector, ctx)
    if run is None or not (run / "config.json").is_file():
        return {}
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    report, train, data = (config.get(k) or {} for k in ("parameter_report", "config", "data"))
    usage = compute.run_compute(run)
    steps, batch = _trained_steps(selector, usage), train.get("batch_size")
    roots = data.get("roots_per_step") or (
        batch - round(train.get("child_frac", 0.0) * batch) if batch else None
    )
    facts = {
        "params_total": report.get("total", config.get("parameters")),
        "params_non_gab": report.get("non_gab"),
        "gpu_hours": usage.gpu_hours if isinstance(usage, compute.RunCompute) else None,
        "positions_seen": steps * batch if steps and batch else None,
    }
    manifest = Path(data["dir"]) / "manifest.json" if data.get("dir") else None
    if steps and roots and manifest is not None and manifest.is_file():
        pack = json.loads(manifest.read_text(encoding="utf-8"))
        facts["training_positions"] = rs.training_positions(pack, steps * roots)
    return {key: value for key, value in facts.items() if value is not None}


def blink_selectors(ctx: EvalContext) -> dict[str, str]:
    """Each Blink engine name the run can rate, mapped to its selector: the model under test, E5's side
    models and E6's Blink rung, in both modes, named in the run's fast play mode as they played."""
    from blink.eval.fastchess import engine_name
    from blink.eval.ladder import LADDER_PLAYERS
    from blink.play.factory import MODES

    selectors = [ctx.model, *ctx.side_models, *(p for p in LADDER_PLAYERS if p.startswith("run:"))]
    return {engine_name(s, mode, **ctx.play_mode): s for s in selectors for mode in MODES}


def row_costs(ctx: EvalContext, agents: Iterable[str], audits: Mapping[str, dict], notes) -> dict[str, dict]:
    """Per strength row: its play costs (searchless players) and, for a Blink model, its model facts."""
    selectors = blink_selectors(ctx)
    costs = {}
    for agent in agents:
        facts = model_facts(selectors[agent], ctx) if agent in selectors else {}
        if agent in selectors and not facts:
            _note(
                notes, f"{agent}: no training run to read its size and training from (ship needs --film-run)"
            )
        costs[agent] = {**facts, **play_costs(audits.get(agent))}
    return costs


def strength_rows(
    fit,
    state: dict,
    shipped: str | None,
    reproduce: str,
    epsilon: float | None = None,
    notes: list[str] | None = None,
    costs: Mapping[str, dict] | None = None,
) -> tuple:
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
                reproduce=reproduce,
                sf_nodes_equiv=crossover if agent == shipped else None,
                **fields,
                **_puzzle_fields(agent, state, epsilon, notes),
                **(costs or {}).get(agent, {}),
            )
        )
    return tuple(rows)


PUZZLE_DIAGNOSTICS = {"band_pct": {}, "band_n": {}, "puzzle_rating_equiv": None, "puzzle_rating_ci": None}


def diagnostics_rows(
    state: dict, shipped_mode_name: str | None, epsilon: float | None = None, notes: list[str] | None = None
) -> tuple:
    """E2's rows, then E6's rungs (each one's valprobe VAA, the film's ladder milestones); the shipped
    mode's E2 row also carries E8's rules-on conversion and its game count.

    E2 runs before E2b: when its value agent played another epsilon than the rated games (`epsilon`),
    the value row's puzzle numbers (bands, puzzle-rating equivalent) are left out, not mislabelled."""
    from blink.report.results_schema import DiagnosticsRow

    rules_on = (state.get("E8") or {}).get("rules_on") or {}
    conversion = {"conversion_pct": rules_on.get("pct"), "conversion_n": rules_on.get("n")}
    e2 = state.get("E2") or {}
    stale = epsilon is not None and e2.get("value_epsilon", epsilon) != epsilon
    rows = []
    for row in e2.get("diagnostics", []):
        ci = row.get("puzzle_rating_ci")
        extra = conversion if row["mode"] == shipped_mode_name else {}
        dropped = PUZZLE_DIAGNOSTICS if stale and row["mode"] == "value" else {}
        if dropped:
            _note(notes, f"E2 scored {row['agent']}'s value-mode puzzles at epsilon {e2['value_epsilon']!r}, "
                  f"the rated games used {epsilon!r}: left out (rerun E2 after E2b)")  # fmt: skip
        fields = {**row, "puzzle_rating_ci": tuple(ci) if ci else None, **extra, **dropped}
        rows.append(DiagnosticsRow(**fields))
    rows += [DiagnosticsRow(**row) for row in (state.get("E6") or {}).get("diagnostics", [])]
    return tuple(rows)


def run_epsilon(state: dict, ctx: EvalContext) -> float | None:
    """The epsilon this run's rated games were played with: the one the blocks after E2b recorded, else
    results/epsilon.json when it exists; None before E2b has run."""
    from blink.eval.match import read_epsilon

    if state.get("epsilon") is not None:
        return state["epsilon"]
    return read_epsilon(ctx.results_dir) if (Path(ctx.results_dir) / "epsilon.json").is_file() else None


def shipped_record(
    state: dict, ctx: EvalContext, name: str, mode: str, epsilon: float | None, notes: list[str] | None
):
    """The shipped model, with the sha pinned when the run started and the fast play mode its rated
    games used; None (and a note) when the weights could not be hashed or no longer match the pin, so
    the bot's --sha never trusts a guess."""
    from blink.report.results_schema import Shipped

    pinned = state.get("weights_sha")
    if not pinned:
        _note(notes, f"no shipped model: the weights of {ctx.model} could not be hashed (no sha to pin)")
        return None
    if _orchestrate().weights_sha(ctx.model) != pinned:
        _note(notes, f"no shipped model: the weights of {ctx.model} changed after the run pinned {pinned}")
        return None
    return Shipped(name, mode, pinned, epsilon if mode == "value" else None, **ctx.play_mode)


def build_results(
    state: dict,
    ctx: EvalContext,
    fit,
    listing: Path | None = None,
    notes: list[str] | None = None,
    audits: Mapping[str, dict] | None = None,
) -> object:
    from blink.eval.fastchess import engine_name
    from blink.report.results_schema import Results

    mode = (state.get("E3") or {}).get("mode") or ctx.mode
    shipped_name = engine_name(ctx.model, mode, **ctx.play_mode) if mode else None
    epsilon = run_epsilon(state, ctx)
    shipped = shipped_record(state, ctx, shipped_name, mode, epsilon, notes) if mode else None
    reproduce = reproduce_command(listing or ctx.results_dir / FINAL_SLICE_LIST)
    costs = row_costs(ctx, sorted(fit.tally), audits or {}, notes) if fit is not None else {}
    strength = (
        strength_rows(fit, state, shipped_name, reproduce, epsilon, notes, costs) if fit is not None else ()
    )
    return Results(
        strength=strength,
        diagnostics=diagnostics_rows(state, mode, epsilon, notes),
        shipped=shipped,
        eval_md_sha=state["protocol"]["sha256"],
        generated_at=_orchestrate()._now(),
    )


def public_pgns(ctx: EvalContext, state: dict) -> list[Path]:
    """Every PGN a block of this evaluation wrote (this run's reports, or <out>/<block>.json for a block
    that ran alone into the same --out), once each, in plan order."""
    listed: list[str] = []
    for block_id in _orchestrate().BLOCK_ORDER:
        for pgn in _orchestrate().earlier_report(ctx, state, block_id).get("pgns", []):
            if str(pgn) not in listed:
                listed.append(str(pgn))
    return [Path(p) for p in listed if Path(p).is_file()]


def public_audit(ctx: EvalContext, state: dict) -> dict[str, dict]:
    """results/nosearch.json: one exact-name audit of every searchless player over every public PGN,
    with the PGN list it covers; returns each player's own report (rows and ms per move)."""
    from blink.eval import nosearch

    files = public_pgns(ctx, state)
    players = sorted(p for p in _orchestrate().forfeit_table(files) if nosearch.is_searchless(p))
    together, each = nosearch.audit_public(files, players)
    payload = {**together, "pgns": [str(p) for p in files], "players_audited": players}
    _orchestrate()._write_json(Path(ctx.results_dir) / NOSEARCH_FILE, payload)
    return each


def write_pgn_list(pgns: Sequence[Path], path: Path) -> Path:
    """The PGNs Ordo rated, one path per line: `blink rate --pgn-list <this file>` refits exactly them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(f"{p}\n" for p in pgns))
    return path
