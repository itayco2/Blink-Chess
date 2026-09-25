"""From `blink eval all` to every public reader: results.json as orchestrate writes it, not a fixture.

The report tests read tests/fixtures/report, which is written by hand. These tests feed run_all's real
output (fake block runners and a fake Ordo fit, real result-building code) to the claim, the scoreboard,
the no-search box and the film's ladder milestones, so a field no writer produces fails here.
"""

import hashlib
import json
import shlex
from pathlib import Path

import pytest
from test_report_fixtures import lichess

from blink import cli
from blink.eval import orchestrate, rating
from blink.film import extract
from blink.report import claims, scoreboard
from blink.report import compute as compute_mod
from blink.report import results_schema as rs

EPSILON = 1 / 256
PARAMS = {"total": 9_437_184, "gab": 12_288, "non_gab": 9_424_896, "blocks": 9_000_000, "static_bias": 0}
TRAIN_ROOTS = 1_500_000
BATCH, ROOTS_PER_STEP, STEPS = 1024, 717, 2000
WEIGHTS = b"shipped weights"

GAMES = """[Event "E5"]
[White "Blink-value-ship"]
[Black "SF1800"]
[Result "1-0"]
[Termination "normal"]

1. e4 {+0.30/1 0.021s, n=21} e5 {+0.10/20 0.100s, n=90000} 2. Qh5 {+0.40/1 0.019s, n=30}
Nc6 {+0.10/20 0.100s, n=90000} 3. Bc4 {+0.40/1 0.035s, n=35} Nf6 {+0.10/20 0.100s, n=90000}
4. Qxf7# {+1.00/1 0.001s, n=0} 1-0

[Event "E7"]
[White "SF1800"]
[Black "DM-9M"]
[Result "1/2-1/2"]
[Termination "adjudication"]

1. d4 {+0.10/20 0.100s, n=90000} d5 {+0.00/1 0.050s, n=21} 1/2-1/2
"""


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def flagship(home: Path, pack: Path) -> Path:
    """runs/long as the trainer leaves it: config.json (parameter report, pack), metrics with power."""
    run = home / "runs" / "long"
    config = {
        "run": "long",
        "parameters": PARAMS["total"],
        "parameter_report": PARAMS,
        "device": "cuda",
        "data": {"source": "shards", "dir": str(pack), "roots_per_step": ROOTS_PER_STEP},
        "branched_from": None,
        "config": {"batch_size": BATCH, "child_frac": 0.3, "steps": STEPS},
    }
    _write_json(run / "config.json", config)
    rows = [{"step": s, "samples_per_s": 1024.0, "gpu_power_w": 200.0} for s in (1000, STEPS)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return run


def fake_fit(pgns, anchors, workdir, **options):
    rows = (
        rating.OrdoRow("Blink-value-ship", 1712.5, 41.0, 250.0, 400, 62.5),
        rating.OrdoRow("DM-9M", 2210.0, 60.0, 90.0, 200, 45.0),
        rating.OrdoRow("Material", 900.0, 70.0, 20.0, 400, 10.0),
        rating.OrdoRow("SF1800", 1800.0, None, 150.0, 400, 37.5),
    )
    tally = {r.player: {"games": r.played, "points": r.points} for r in rows}
    return rating.OrdoFit(rows, (rating.Anchor("SF1800", 1800),), {}, tally, ("ordo",), {})


def rung_rows() -> list[dict]:
    """What E6 writes for its rungs: each one's VAA on the valprobe, named as the rung plays."""
    return [
        {"agent": "Material", "mode": "value", "vaa": 0.08},
        {"agent": "Random", "mode": "value", "vaa": 0.03},
    ]


@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    """A full `blink eval all --model ship --film-run long` through fake runners; returns the results dir."""
    home = tmp_path / "home"
    monkeypatch.setenv("BLINK_HOME", str(home))
    pack = _write_json(
        tmp_path / "pack" / "manifest.json", {"splits": {"roots": {"train": TRAIN_ROOTS}}}
    ).parent
    flagship(home, pack)
    weights = tmp_path / "blink.pt"
    weights.write_bytes(WEIGHTS)
    monkeypatch.setattr(orchestrate, "weights_file", lambda selector: weights if selector == "ship" else None)
    results = tmp_path / "results"
    _write_json(results / "epsilon.json", {"epsilon": EPSILON})
    puzzles = {"accuracy": 0.801, "wilson95": [0.793, 0.809], "epsilon": EPSILON}
    _write_json(home / "eval" / "puzzles" / "puzzles_dm10k_ship_value.json", puzzles)
    pgn = tmp_path / "games" / "final.pgn"
    pgn.parent.mkdir()
    pgn.write_text(GAMES, encoding="utf-8")
    extra = {
        "E0": {"dm_puzzles": {"accuracy": 0.861, "wilson95": [0.854, 0.868]}},
        "E2": {
            "diagnostics": [{"agent": "Blink-ship", "mode": "value", "vaa": 0.6}],
            "value_epsilon": EPSILON,
        },
        "E3": {"mode": "value"},
        "E4": {"crossover": {"nodes": 2048}},
        "E5": {"final_slice_pgns": [str(pgn)], "pgns": [str(pgn)]},
        "E6": {"diagnostics": rung_rows()},
        "E8": {"rules_on": {"pct": 91.0, "n": 500}},
    }

    def runner(block):
        return lambda context, state: {"games": 2, "pgns": [], **extra.get(block, {})}

    protocol = tmp_path / "EVAL.md"
    protocol.write_text("# EVAL\n", encoding="utf-8")
    ctx = orchestrate.EvalContext(
        model="ship", out_dir=tmp_path / "out", results_dir=results, protocol=protocol, film_run="long"
    )
    runners = {block: runner(block) for block in orchestrate.BLOCK_ORDER}
    out = orchestrate.run_all(
        ctx, runners=runners, runs_root=tmp_path, log=lambda s: None, ordo=fake_fit, load=lambda: 0.0
    )
    compute_mod.write_compute(compute_mod.project_compute(home / "runs", "long"), results / "compute.json")
    (results / "lichess.json").write_text(rs.lichess_to_json(lichess()) + "\n", encoding="utf-8")
    return results, out


def _results(folder: Path) -> rs.Results:
    return rs.from_json((folder / "results.json").read_text(encoding="utf-8"))


def test_the_shipped_row_carries_the_models_size_training_and_play_costs(evaluated):
    folder, _ = evaluated
    row = scoreboard.shipped_row(_results(folder))
    assert (row.params_total, row.params_non_gab) == (PARAMS["total"], PARAMS["non_gab"])
    assert row.positions_seen == STEPS * BATCH
    assert row.training_positions == rs.training_positions(
        {"splits": {"roots": {"train": TRAIN_ROOTS}}}, STEPS * ROOTS_PER_STEP
    )
    assert row.gpu_hours == pytest.approx(STEPS * BATCH / 1024.0 / 3600)
    assert (row.evals_per_move_median, row.evals_per_move_max) == (25.5, 35)  # 0, 21, 30, 35 (R2's mate: 0)
    assert row.ms_per_move_p50 == pytest.approx(20.0)


def test_the_claim_fills_from_what_eval_all_writes(evaluated):
    folder, _ = evaluated
    claim = claims.fill_claim(folder)
    assert "9.4M-parameter transformer" in claim
    assert (
        f"{rs.training_positions({'splits': {'roots': {'train': TRAIN_ROOTS}}}, STEPS * ROOTS_PER_STEP):,}"
        in claim
    )
    assert "1712 +/- 41" in claim and "2,048 nodes" in claim and "80.1%" in claim


def test_the_no_search_box_audits_every_public_game_eval_all_played(evaluated):
    folder, _ = evaluated
    audit = json.loads((folder / "nosearch.json").read_text(encoding="utf-8"))
    assert audit["compliant"] and audit["decisions"] == 5  # Blink's 4 moves and DM-9M's 1
    assert audit["players_audited"] == ["Blink-value-ship", "DM-9M"]
    assert [Path(p).name for p in audit["pgns"]] == ["final.pgn"]
    assert "Across 5 public moves" in scoreboard.nosearch_box(scoreboard.load_bundle(folder).nosearch)


def test_the_scoreboard_renders_from_eval_all_and_its_elo_reproduce_command_runs(evaluated):
    folder, _ = evaluated
    bundle = scoreboard.load_bundle(folder)
    block = scoreboard.render_block(bundle)
    command = scoreboard.shipped_row(bundle.results).reproduce
    assert f"`{command}`" in scoreboard.headline(bundle)
    assert "--model ship`" not in block.split("uv run blink eval puzzles")[0]
    tokens = shlex.split(command)
    assert tokens[:3] == ["uv", "run", "blink"]
    args = cli.build_parser().parse_args(tokens[3:])  # the command parses; `rate --model` never did
    assert [Path(p).name for p in args.pgn_list] == ["final_slice_pgns.txt"]


def test_film_ladder_milestones_resolve_against_what_eval_all_writes(evaluated):
    folder, _ = evaluated
    rungs = extract.ladder_rungs(folder, [("passed material", "Material")])
    assert [(r["agent"], r["metric"], r["threshold"]) for r in rungs] == [("Material", "vaa", 0.08)]


def test_the_shipped_sha_is_the_pinned_weights(evaluated):
    folder, _ = evaluated
    shipped = _results(folder).shipped
    assert shipped.sha == hashlib.sha256(WEIGHTS).hexdigest() and shipped.epsilon == EPSILON
