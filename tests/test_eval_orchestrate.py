"""`blink eval all`: plan order, the game table, the frozen protocol, the training guard, forfeits."""

import json
import subprocess
import time
from pathlib import Path

import pytest

from blink.eval import orchestrate, rating
from blink.report import results_schema

PLAN_TOTAL = 28_600


def IDLE():  # noqa: N802 - a constant-like fake of the CPU probe
    return 0.0


def ctx(tmp_path, **kwargs):
    protocol = tmp_path / "EVAL.md"
    if not protocol.exists():
        protocol.write_text("# EVAL\n", encoding="utf-8")
    base = {
        "model": "run:x",
        "out_dir": tmp_path / "out",
        "results_dir": tmp_path / "results",
        "protocol": protocol,
    }
    return orchestrate.EvalContext(**{**base, **kwargs})


def recorder(calls, extra=None):
    def make(block):
        def run(context, state):
            calls.append(block)
            return {"games": 2, "pgns": [], **(extra or {}).get(block, {})}

        return run

    return {block: make(block) for block in orchestrate.BLOCK_ORDER}


def test_the_blocks_run_in_the_plans_order_and_each_can_run_alone(tmp_path):
    calls = []
    orchestrate.run_blocks(
        ctx(tmp_path), recorder(calls), runs_root=tmp_path / "runs", log=lambda s: None, load=IDLE
    )
    assert calls == list(orchestrate.BLOCK_ORDER)
    calls.clear()
    orchestrate.run_blocks(
        ctx(tmp_path), recorder(calls), only=["E8", "E3"], runs_root=tmp_path, log=lambda s: None, load=IDLE
    )
    assert calls == ["E3", "E8"]
    assert json.loads((tmp_path / "out" / "E8.json").read_text(encoding="utf-8"))["games"] == 2
    with pytest.raises(ValueError, match="E10"):
        orchestrate.run_blocks(ctx(tmp_path), recorder([]), only=["E10"], runs_root=tmp_path, load=IDLE)


def test_the_game_table_lists_every_block_with_the_plans_counts():
    table = orchestrate.game_table(orchestrate.BLOCK_ORDER)
    assert f"| total | | {PLAN_TOTAL:,} |" in table
    assert "| E5 | 6,100 | 6,100 | yes | final |" in table
    smoke = orchestrate.game_table(["E4"], games=2)
    assert "| E4 | 1,500 | 18 |" in smoke


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_a_changed_eval_md_is_refused_once_the_frozen_tag_exists(tmp_path):
    protocol = tmp_path / "EVAL.md"
    protocol.write_text("# EVAL v1\nfrozen rules\n", encoding="utf-8")
    assert orchestrate.check_protocol(protocol, tmp_path)["frozen"] is False
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "add", "EVAL.md")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "v1")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "tag", "-a", "eval-v1-frozen", "-m", "frozen")
    report = orchestrate.check_protocol(protocol, tmp_path)
    assert report["frozen"] and report["matches"]
    protocol.write_text("# EVAL v1\nquietly edited rules\n", encoding="utf-8")
    with pytest.raises(orchestrate.ProtocolMismatch):
        orchestrate.check_protocol(protocol, tmp_path)


def live_run(runs_root, state="running"):
    run = runs_root / "long"
    run.mkdir(parents=True)
    (run / "heartbeat.json").write_text(json.dumps({"state": state, "time": time.time()}), encoding="utf-8")


def test_time_based_blocks_refuse_to_start_while_a_training_heartbeat_is_live(tmp_path):
    live_run(tmp_path / "runs")
    calls = []
    with pytest.raises(orchestrate.TrainingLive, match="E0"):
        orchestrate.run_blocks(
            ctx(tmp_path), recorder(calls), runs_root=tmp_path / "runs", log=lambda s: None, load=IDLE
        )
    assert calls == []
    orchestrate.run_blocks(
        ctx(tmp_path), recorder(calls), only=["E2", "E3"], runs_root=tmp_path / "runs", load=IDLE
    )
    assert calls == ["E2", "E3"]


def test_a_finished_run_does_not_block_time_based_blocks(tmp_path):
    live_run(tmp_path / "runs", state="finished")
    assert orchestrate.guard_time_based("E5", tmp_path / "runs", load=IDLE) == 0.0


PGN = """[White "Blink-value-ship"]
[Black "SF1800"]
[Result "0-1"]
[Termination "time forfeit"]

1. e4 e5 0-1

[White "SF1800"]
[Black "Blink-value-ship"]
[Result "1/2-1/2"]
[Termination "adjudication"]

1. e4 e5 1/2-1/2

[White "SF1900"]
[Black "Blink-value-ship"]
[Result "0-1"]
[Termination "illegal move"]

1. e4 e5 0-1

[White "Blink-value-ship"]
[Black "SF1900"]
[Result "1-0"]
[Termination "normal"]

1. e4 e5 1-0
"""


def test_the_pgn_audit_counts_time_forfeits_and_adjudications_per_engine(tmp_path):
    pgn = tmp_path / "g.pgn"
    pgn.write_text(PGN, encoding="utf-8")
    table = orchestrate.forfeit_table([pgn, tmp_path / "missing.pgn"])
    assert table["Blink-value-ship"] == {"games": 4, "time_forfeits": 1, "forfeits": 0, "adjudications": 1}
    assert table["SF1900"] == {"games": 2, "time_forfeits": 0, "forfeits": 1, "adjudications": 0}
    assert table["SF1800"]["adjudications"] == 1 and table["SF1800"]["time_forfeits"] == 0


def test_each_block_report_carries_its_forfeit_table(tmp_path):
    pgn = tmp_path / "g.pgn"
    pgn.write_text(PGN, encoding="utf-8")
    extra = {"E5": {"pgns": [str(pgn)]}}
    state = orchestrate.run_blocks(
        ctx(tmp_path), recorder([], extra), only=["E5"], runs_root=tmp_path, load=IDLE
    )
    assert state["E5"]["forfeits"]["Blink-value-ship"]["time_forfeits"] == 1


DIAGNOSTICS = [
    {"agent": "Blink-run_x", "mode": "policy", "top1": 0.5, "puzzle_rating_ci": [1500.0, 1700.0]},
    {"agent": "Blink-run_x", "mode": "value", "vaa": 0.6},
]


def fake_fit(pgns, anchors, workdir, **options):
    rows = (
        rating.OrdoRow("Blink-value-run_x", 1712.5, 41.0, 250.0, 400, 62.5),
        rating.OrdoRow("DM-9M", 2210.0, 60.0, 90.0, 200, 45.0),
        rating.OrdoRow("SF1800", 1800.0, None, 150.0, 400, 37.5),
    )
    tally = {r.player: {"games": r.played, "points": r.points} for r in rows}
    tally["Random"] = {"games": 200, "points": 0.0}
    return rating.OrdoFit(
        rows, (rating.Anchor("SF1800", 1800),), {"Random": "all losses"}, tally, ("ordo",), {}
    )


def run_all(tmp_path):
    pgn = tmp_path / "final.pgn"
    pgn.write_text(PGN, encoding="utf-8")
    extra = {
        "E0": {"dm_puzzles": {"accuracy": 0.861, "wilson95": [0.838, 0.881]}},
        "E2": {"diagnostics": DIAGNOSTICS},
        "E3": {"mode": "value"},
        "E4": {"crossover": {"nodes": 2048}},
        "E5": {"final_slice_pgns": [str(pgn)]},
        "E8": {"rules_on": {"pct": 91.0}},
    }
    return orchestrate.run_all(
        ctx(tmp_path),
        runners=recorder([], extra),
        runs_root=tmp_path,
        log=lambda s: None,
        ordo=fake_fit,
        load=IDLE,
    )


def test_run_all_writes_results_json_through_the_schema(tmp_path):
    out = run_all(tmp_path)
    results = results_schema.from_json((tmp_path / "results" / "results.json").read_text(encoding="utf-8"))
    rows = {r.agent: r for r in results.strength}
    blink = rows["Blink-value-run_x"]
    assert (blink.elo, blink.elo_ci95, blink.elo_games, blink.kind) == (1712.5, 41.0, 400, "blink")
    assert blink.sf_nodes_equiv == 2048
    assert rows["SF1800"].kind == "anchor" and rows["SF1800"].elo is None
    assert rows["Random"].elo is None and rows["Random"].kind == "ladder"
    assert rows["DM-9M"].dm_puzzles_pct == pytest.approx(86.1)
    value_row = next(r for r in results.diagnostics if r.mode == "value")
    assert value_row.conversion_pct == 91.0 and results.shipped.mode == "value"
    assert results.eval_md_sha == out["state"]["protocol"]["sha256"]
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["games"]["E5"] == 2


def test_paper_numbers_never_sit_in_the_measured_elo_column(tmp_path):
    run_all(tmp_path)
    results = results_schema.from_json((tmp_path / "results" / "results.json").read_text(encoding="utf-8"))
    assert all(row.paper_reported is None for row in results.strength if row.elo is not None)
    with pytest.raises(ValueError, match="paper"):
        results_schema.StrengthRow(
            agent="DM-270M", kind="reference", reproduce="-", elo=2895.0, elo_ci95=1.0, elo_games=1,
            paper_reported="2895 Lichess blitz vs humans (2024)",
        )  # fmt: skip


def test_the_shipped_mode_is_never_guessed(tmp_path):
    with pytest.raises(ValueError, match="shipped mode"):
        orchestrate.shipped_mode(ctx(tmp_path), {})
    assert orchestrate.shipped_mode(ctx(tmp_path, mode="policy"), {}) == "policy"
    assert orchestrate.shipped_mode(ctx(tmp_path, mode="policy"), {"E3": {"mode": "value"}}) == "value"


def test_the_smoke_override_rounds_to_whole_pairs(tmp_path):
    assert ctx(tmp_path).n(400) == 400
    assert ctx(tmp_path, games=3).n(400) == 4
    assert ctx(tmp_path, games=1).n(400) == 2


def test_pack_files_are_found_in_both_layouts(tmp_path):
    (tmp_path / "test_iid.bin").write_bytes(b"")
    (tmp_path / "val_roots.bin").write_bytes(b"")
    assert orchestrate._pack_file(tmp_path, "test_iid") == tmp_path / "test_iid.bin"
    assert orchestrate._pack_file(tmp_path, "val") == tmp_path / "val_roots.bin"
    assert orchestrate._pack_file(tmp_path, "test_grouped") is None


def test_every_block_has_a_runner():
    assert set(orchestrate.default_runners()) == set(orchestrate.BLOCK_ORDER)
    assert Path(orchestrate.__file__).name == "orchestrate.py"


def test_time_based_blocks_refuse_to_start_on_a_busy_cpu_unless_allowed(tmp_path):
    calls = []
    with pytest.raises(orchestrate.MachineBusy, match="E5"):
        orchestrate.run_blocks(
            ctx(tmp_path), recorder(calls), only=["E5"], runs_root=tmp_path, load=lambda: 80.0
        )
    assert calls == []
    allowed = ctx(tmp_path, allow_busy_cpu=True)
    state = orchestrate.run_blocks(
        allowed, recorder(calls), only=["E5"], runs_root=tmp_path, load=lambda: 80.0
    )
    assert calls == ["E5"] and state["E5"]["cpu_pct_at_start"] == 80.0
    state = orchestrate.run_blocks(
        ctx(tmp_path), recorder(calls), only=["E3"], runs_root=tmp_path, load=lambda: 99.0
    )
    assert state["E3"]["cpu_pct_at_start"] is None  # E3 has no clock: no probe, no refusal


def test_dm_9m_puzzles_outside_88_9_plus_or_minus_1_call_for_the_g6_audit():
    assert orchestrate.dm_puzzle_check({"accuracy": 0.889})["in_band"]
    missed = orchestrate.dm_puzzle_check({"accuracy": 0.861})
    assert missed["g6_needed"] and missed["pct"] == pytest.approx(86.1)


def test_the_sf_self_check_fails_on_any_forfeit_even_at_50_percent():
    clean = orchestrate.selfcheck_verdict({"score": 0.52, "audit": {"forfeits": {}}})
    assert clean["passed"]
    forfeited = orchestrate.selfcheck_verdict(
        {"score": 0.5, "audit": {"forfeits": {"SF1800": {"time forfeit": 1}}}}
    )
    assert forfeited["within_band"] and not forfeited["passed"]
    assert not orchestrate.selfcheck_verdict({"score": 0.60, "audit": {"forfeits": {}}})["passed"]


def test_a_blink_row_takes_its_puzzle_score_from_blink_eval_puzzles(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    folder = tmp_path / "eval" / "puzzles"
    folder.mkdir(parents=True)
    done = {"accuracy": 0.8, "wilson95": [0.79, 0.81]}
    (folder / "puzzles_dm10k_ship_value.json").write_text(json.dumps(done), encoding="utf-8")
    fields = orchestrate._puzzle_fields("Blink-value-ship", {})
    assert fields["dm_puzzles_pct"] == pytest.approx(80.0)
    assert fields["dm_puzzles_ci"] == pytest.approx((79.0, 81.0))
    assert orchestrate._puzzle_fields("Blink-policy-ship", {}) == {}


def test_a_refused_ordo_pool_still_writes_results_json_without_elo(tmp_path):
    def refuse(pgns, anchors, workdir, **options):
        raise RuntimeError("ordo refused the pool")

    pgn = tmp_path / "final.pgn"
    pgn.write_text(PGN, encoding="utf-8")
    out = orchestrate.run_all(
        ctx(tmp_path),
        runners=recorder([], {"E5": {"final_slice_pgns": [str(pgn)]}}),
        runs_root=tmp_path,
        log=lambda s: None,
        ordo=refuse,
        load=IDLE,
    )
    assert out["ordo_error"] == "ordo refused the pool"
    results = results_schema.from_json((tmp_path / "results" / "results.json").read_text(encoding="utf-8"))
    assert {r.agent for r in results.strength} == {"Blink-value-ship", "SF1800", "SF1900"}
    assert all(r.elo is None for r in results.strength)


def test_a_smoke_run_asks_ordo_for_fewer_simulations_and_a_short_timeout(tmp_path):
    seen = {}

    def spy(pgns, anchors, workdir, **options):
        seen.update(options)
        return fake_fit(pgns, anchors, workdir)

    pgn = tmp_path / "final.pgn"
    pgn.write_text(PGN, encoding="utf-8")
    runners = recorder([], {"E5": {"final_slice_pgns": [str(pgn)]}})
    orchestrate.run_all(
        ctx(tmp_path, games=2), runners=runners, runs_root=tmp_path, log=lambda s: None, ordo=spy, load=IDLE
    )
    assert seen == {"simulations": 100, "timeout_s": 120}
    orchestrate.run_all(
        ctx(tmp_path), runners=runners, runs_root=tmp_path, log=lambda s: None, ordo=spy, load=IDLE
    )
    assert seen == {"simulations": 1000, "timeout_s": 1800}


def test_blocks_record_the_epsilon_they_played_with_and_refuse_a_changed_one(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "epsilon.json").write_text(json.dumps({"epsilon": 1 / 256}), encoding="utf-8")
    runners = recorder([])

    def e3_rewrites_epsilon(context, state):
        (results / "epsilon.json").write_text(json.dumps({"epsilon": 0.0}), encoding="utf-8")
        return {"games": 2, "pgns": []}

    state = orchestrate.run_blocks(
        ctx(tmp_path), runners, only=["E2", "E3"], runs_root=tmp_path, log=lambda s: None, load=IDLE
    )
    assert state["E3"]["epsilon"] == 1 / 256 and state["E2"]["epsilon"] is None
    with pytest.raises(orchestrate.EpsilonChanged, match="E4"):
        orchestrate.run_blocks(
            ctx(tmp_path),
            {**runners, "E3": e3_rewrites_epsilon},
            only=["E3", "E4"],
            runs_root=tmp_path,
            log=lambda s: None,
            load=IDLE,
        )
