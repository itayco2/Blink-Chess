"""Every engine carries both the chosen epsilon and the chosen fast play mode, and its name tells them apart.

The fastplay merge met main's epsilon plumbing: blink-uci under fastchess (E5, `blink gauntlet`), the
in-process blocks, E9's reading of E5's games, results.json's shipped record and `blink eval puzzles`.
Default engine commands and names are main's, byte for byte. Nothing here loads torch or a GPU: the
model loader and fastchess are replaced where a test would reach them.
"""

import hashlib
import json

import pytest
from test_eval_failures import e5_report, e9_context, fake_e9
from test_eval_orchestrate import DIAGNOSTICS, IDLE, PGN, fake_fit, pin_weights, puzzle_home, recorder
from test_eval_orchestrate import ctx as eval_ctx

from blink import cli, uci
from blink.eval import anchors, failures, fastchess, match, orchestrate, publish, rating
from blink.play import factory
from blink.play.oracles import RandomLogitEvaluator
from blink.report import results_schema

SHA = hashlib.sha256(b"blink weights").hexdigest()
EPSILON = 1 / 256
ANCHOR = rating.Anchor("SF1800", 1800)


# ------------------------------------------------------------------------------ names and commands


def test_default_engine_commands_and_names_are_mains_byte_for_byte():
    spec = fastchess.blink_engine("ship", "value", "cuda", epsilon=EPSILON, sha=SHA)
    assert spec.args == (
        "-m",
        "blink.uci",
        "--model=ship",
        "--mode=value",
        "--device=cuda",
        "--epsilon=0.00390625",
        f"--sha={SHA}",
    )
    assert spec.name == "Blink-value-ship"
    assert fastchess.engine_name("ship", "policy") == "Blink-policy-ship"
    dm = fastchess.blink_engine("dm:9M", "policy", "cuda", epsilon=EPSILON, sha=SHA)
    assert dm.args == ("-m", "blink.uci", "--model=dm:9M", "--device=cuda") and dm.name == "DM-9M"


def test_an_engine_command_carries_the_epsilon_the_sha_and_the_fast_mode_together():
    spec = fastchess.blink_engine(
        "ship", "value", "cuda", epsilon=EPSILON, sha=SHA, precision="bf16", compile=True
    )
    assert spec.args[-4:] == ("--epsilon=0.00390625", f"--sha={SHA}", "--precision=bf16", "--compile")
    assert spec.name == "Blink-value-ship-bf16-compile"
    parsed = uci.build_parser().parse_args(list(spec.args[2:]))  # what blink-uci itself reads
    assert (parsed.epsilon, parsed.sha, parsed.precision, parsed.compile) == (EPSILON, SHA, "bf16", True)


def test_every_setting_that_changes_play_gives_its_own_name():
    names = {
        fastchess.engine_name("ship", mode, precision, compiled)
        for mode in ("policy", "value")
        for precision in ("fp32", "bf16")
        for compiled in (False, True)
    }
    assert len(names) == 8
    assert "Blink-value-ship-compile" in names and "Blink-policy-ship-bf16" in names
    long = "run:" + "x" * 60  # a hashed long tag keeps the fast mode's tag after it
    assert fastchess.engine_name(long, "value", "bf16") == fastchess.engine_name(long, "value") + "-bf16"
    assert fastchess.engine_name("dm:9M", "policy") == "DM-9M"


def test_two_selectors_sharing_a_fast_name_are_still_refused():
    fastchess.check_distinct_names(["ship", "run:s10m"], "value", "bf16", True)
    with pytest.raises(ValueError, match="Blink-policy-run_a_b-bf16"):
        fastchess.check_distinct_names(["run:a b", "run_a_b"], "policy", "bf16")


def test_deepminds_engine_has_no_fast_mode():
    with pytest.raises(ValueError, match="Blink models only"):
        fastchess.blink_engine("dm:9M", "policy", "cuda", precision="bf16")


@pytest.mark.parametrize("given", [True, False])
def test_blink_gauntlet_passes_both_the_epsilon_and_the_fast_mode(tmp_path, capsys, given):
    results = tmp_path / "results"
    results.mkdir()
    (results / "epsilon.json").write_text(json.dumps({"epsilon": EPSILON}), encoding="utf-8")
    base = ["gauntlet", "--model", "random", "--mode", "value", "--games", "10", "--out", str(tmp_path)]
    epsilon = ["--epsilon", str(EPSILON)] if given else ["--results-dir", str(results)]
    assert cli.main([*base, *epsilon, "--precision", "bf16", "--compile", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "--epsilon=0.00390625 --precision=bf16 --compile" in printed
    assert "name=Blink-value-random-bf16-compile" in printed


# ------------------------------------------------------------------------------ the evaluation


def test_an_eval_run_refuses_bf16_off_cuda_before_anything_runs(tmp_path, capsys):
    with pytest.raises(ValueError, match="CUDA only"):
        orchestrate.EvalContext(model="ship", device="cpu", precision="bf16", out_dir=tmp_path)
    argv = ["eval", "all", "--model", "ship", "--device", "cpu", "--precision", "bf16", "--dry-run"]
    assert cli.main([*argv, "--out", str(tmp_path)]) == 2
    assert "CUDA only" in capsys.readouterr().err


def fast_ctx(tmp_path, **kwargs):
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "epsilon.json").write_text(json.dumps({"epsilon": EPSILON}), encoding="utf-8")
    base = {"model": "ship", "device": "cuda", "out_dir": tmp_path / "out", "results_dir": results}
    return orchestrate.EvalContext(**{**base, "precision": "bf16", "compile": True, **kwargs})


def captured_engines(monkeypatch) -> list:
    """fastchess stubbed at prepare_pair: the first engine of every match, nothing started."""
    seen = []

    def prepare_pair(first, second, games, book, out_dir, concurrency=5, max_moves=0, skip=0):
        seen.append(first)
        return first

    monkeypatch.setattr(fastchess, "prepare_pair", prepare_pair)
    monkeypatch.setattr(fastchess, "execute", lambda gauntlet: gauntlet)
    monkeypatch.setattr(fastchess, "match_report", lambda spec: {"games": 2, "score": 0.5, "pgn": "g.pgn"})
    monkeypatch.setattr(orchestrate, "weights_sha", lambda selector: SHA)
    return seen


def test_e5s_engine_carries_e2bs_epsilon_the_pinned_sha_and_the_runs_fast_mode(tmp_path, monkeypatch):
    seen = captured_engines(monkeypatch)
    ctx = fast_ctx(tmp_path)
    anchors.fastchess_player(ctx, "ship", "value", "E5")(ANCHOR, 2, "final", 0)
    anchors.fastchess_player(ctx, "dm:9M", "policy", "E7")(ANCHOR, 2, "final", 0)
    blink, deepmind = seen
    assert blink.args[-4:] == ("--epsilon=0.00390625", f"--sha={SHA}", "--precision=bf16", "--compile")
    assert blink.name == "Blink-value-ship-bf16-compile"
    assert deepmind.name == "DM-9M" and not any("precision" in a or "compile" in a for a in deepmind.args)


def test_in_process_players_load_and_are_named_in_the_fast_mode(monkeypatch):
    calls = []

    def load(selector, device="cuda", seed=0, **mode):
        calls.append(mode)
        return RandomLogitEvaluator()

    monkeypatch.setattr(factory, "load_evaluator", load)
    agents = match.blink_agents("run:x", "cuda", epsilon=EPSILON, precision="bf16", compile=True)
    assert calls == [{"precision": "bf16", "compile": True}]
    assert {m: a.name for m, a in agents.items()} == {
        "policy": "Blink-policy-run_x-bf16-compile",
        "value": "Blink-value-run_x-bf16-compile",
    }
    assert agents["value"].epsilon == EPSILON
    default = match.blink_agents("run:x", "cpu", epsilon=EPSILON)
    assert calls[-1] == {"precision": "fp32", "compile": False}
    assert default["value"].name == "Blink-value-run_x"


def test_e9_reads_e5s_games_under_the_fast_name_they_were_played_under(tmp_path, monkeypatch):
    seen = fake_e9(monkeypatch)
    e5, _ = e5_report(tmp_path)
    failures.e9_block(e9_context(tmp_path, device="cuda", precision="bf16"), {"E5": e5})
    assert seen["player"] == "Blink-value-ship-bf16"


def run_all(tmp_path, monkeypatch, **mode):
    pin_weights(tmp_path, monkeypatch)
    pgn = tmp_path / "final.pgn"
    pgn.write_text(PGN, encoding="utf-8")
    extra = {
        "E2": {"diagnostics": DIAGNOSTICS},
        "E3": {"mode": "value"},
        "E5": {"final_slice_pgns": [str(pgn)]},
    }
    return orchestrate.run_all(
        eval_ctx(tmp_path, device="cuda", **mode),
        runners=recorder([], extra),
        runs_root=tmp_path,
        log=lambda s: None,
        ordo=fake_fit,
        load=IDLE,
    )


def test_results_json_ships_the_fast_mode_the_rated_games_used_under_its_name(tmp_path, monkeypatch):
    out = run_all(tmp_path, monkeypatch, precision="bf16", compile=True)
    text = (tmp_path / "results" / "results.json").read_text(encoding="utf-8")
    shipped = results_schema.from_json(text).shipped
    assert shipped.agent == "Blink-value-run_x-bf16-compile"
    assert (shipped.precision, shipped.compile) == ("bf16", True)
    reports = [out["state"][block] for block in orchestrate.BLOCK_ORDER]
    assert all((r["precision"], r["compile"]) == ("bf16", True) for r in reports)  # every block says


def test_a_default_run_ships_fp32_uncompiled_under_mains_name(tmp_path, monkeypatch):
    run_all(tmp_path, monkeypatch)
    shipped = results_schema.from_json((tmp_path / "results" / "results.json").read_text("utf-8")).shipped
    assert (shipped.agent, shipped.precision, shipped.compile) == ("Blink-value-run_x", "fp32", False)


def test_a_results_json_from_before_the_fast_fields_reads_as_fp32_uncompiled():
    payload = {
        "schema_version": results_schema.SCHEMA_VERSION,
        "strength": [],
        "diagnostics": [],
        "shipped": {"agent": "Blink-value-ship", "mode": "value", "sha": SHA, "epsilon": EPSILON},
        "eval_md_sha": "e" * 12,
        "generated_at": "2026-10-08T12:00:00+03:00",
    }
    shipped = results_schema.from_json(json.dumps(payload)).shipped
    assert (shipped.precision, shipped.compile) == ("fp32", False)


def test_fast_mode_puzzles_are_filed_where_results_json_looks_for_that_name(tmp_path, monkeypatch):
    puzzle_home(tmp_path, monkeypatch)
    argv = ["eval", "puzzles", "--model", "ship", "--device", "cuda", "--limit", "2", "--precision", "bf16"]
    assert cli.main(argv) == 0
    written = sorted(p.name for p in (tmp_path / "eval" / "puzzles").iterdir())
    assert "puzzles_dm10k_ship-bf16_value.json" in written, written
    fast = fastchess.engine_name("ship", "value", "bf16")
    assert publish._puzzle_fields(fast, {}, epsilon=EPSILON)
    assert publish._puzzle_fields(fastchess.engine_name("ship", "value"), {}, epsilon=EPSILON) == {}


def test_puzzles_refuse_bf16_off_cuda(tmp_path, monkeypatch, capsys):
    puzzle_home(tmp_path, monkeypatch)
    argv = ["eval", "puzzles", "--model", "ship", "--device", "cpu", "--precision", "bf16"]
    assert cli.main(argv) == 2
    assert "CUDA only" in capsys.readouterr().err
