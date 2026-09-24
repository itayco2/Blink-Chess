"""E0's SF self-check decides how the anchors play (plan P8): st=0.1, or 60+0.6 with E5 in the shipped mode
only; a failed check, an SF forfeit in E0 or E8 and any Blink forfeit are failed done-when gates."""

import json
from types import SimpleNamespace

from blink.eval import anchors, fastchess, ladder, match, orchestrate, rating

FAILED = {"E0": {"sf_selfcheck": {"passed": False, "score": 0.61, "sf_forfeits": {}}}}
PASSED = {"E0": {"sf_selfcheck": {"passed": True, "score": 0.52, "sf_forfeits": {}}}}


def context(tmp_path, **kwargs):
    base = {
        "model": "ship",
        "out_dir": tmp_path / "out",
        "results_dir": tmp_path / "results",
        "side_models": ("run:s10m",),
        "device": "cpu",
    }
    return orchestrate.EvalContext(**{**base, **kwargs})


def recording_player(monkeypatch):
    calls = []

    def fastchess_player(ctx, selector, mode, subdir, anchor_tc=None):
        def play(anchor, games, book, skip):
            calls.append((selector, mode, subdir, anchor_tc, book))
            score = rating.logistic_score(2000 - anchor.rating)
            return {
                "games": games,
                "score": score,
                "pgn": f"{selector}-{mode}-{anchor.name}.pgn",
                "penta": None,
            }

        return play

    monkeypatch.setattr(anchors, "fastchess_player", fastchess_player)
    return calls


def test_a_passed_self_check_plays_st_0_1_anchors_in_both_modes_with_side_rows(tmp_path, monkeypatch):
    calls = recording_player(monkeypatch)
    out = anchors.e5_block(context(tmp_path, games=2), {**PASSED, "E3": {"mode": "value"}})
    assert {c[3] for c in calls} == {None}
    assert set(out["final"]) == {"policy", "value"} and set(out["side"]) == {
        "run:s10m|policy",
        "run:s10m|value",
    }
    assert out["anchor_tc"] is None


def test_a_failed_self_check_moves_the_anchors_to_60_0_6_and_e5_to_the_shipped_mode(tmp_path, monkeypatch):
    calls = recording_player(monkeypatch)
    out = anchors.e5_block(context(tmp_path, games=2), {**FAILED, "E3": {"mode": "value"}})
    assert {c[3] for c in calls} == {"60+0.6"}
    assert {(c[0], c[1]) for c in calls} == {("ship", "value")}
    assert set(out["final"]) == {"value"} and out["side"] == {} and out["anchor_tc"] == "60+0.6"


def test_a_block_run_alone_reads_e0_from_the_same_out_folder(tmp_path, monkeypatch):
    calls = recording_player(monkeypatch)
    ctx = context(tmp_path, games=2, mode="policy")
    ctx.out_dir.mkdir(parents=True)
    (ctx.out_dir / "E0.json").write_text(json.dumps(FAILED["E0"]), encoding="utf-8")
    anchors.e5_block(ctx, {})
    assert {(c[1], c[3]) for c in calls} == {("policy", "60+0.6")}


def test_the_fastchess_anchor_takes_the_fallback_control_and_blink_keeps_st_1(tmp_path, monkeypatch):
    seen = []

    def prepare_pair(first, second, games, book, out_dir, concurrency, skip=0):
        seen.append((first, second))
        return SimpleNamespace(pgn=str(out_dir / "g.pgn"))

    monkeypatch.setattr(fastchess, "prepare_pair", prepare_pair)
    monkeypatch.setattr(fastchess, "execute", lambda gauntlet: gauntlet)
    monkeypatch.setattr(fastchess, "match_report", lambda r: {"games": 2, "score": 0.5, "pgn": r.pgn})
    play = anchors.fastchess_player(context(tmp_path), "ship", "value", "E5", anchor_tc="60+0.6")
    play(rating.Anchor("SF1800", 1800), 2, "final", 0)
    ((first, second),) = seen
    assert (second.tc, second.st, second.timemargin_ms) == ("60+0.6", None, 100)
    assert (first.st, first.tc) == (1.0, None)


def test_e7s_deepmind_gauntlet_uses_the_fallback_anchors(tmp_path, monkeypatch):
    seen = {}

    def fastchess_player(ctx, selector, mode, subdir, anchor_tc=None):
        seen["dm"] = (selector, anchor_tc)
        return lambda *args: None

    monkeypatch.setattr(anchors, "fastchess_player", fastchess_player)
    monkeypatch.setattr(
        match, "blink_agents", lambda *a, **k: {"value": SimpleNamespace(name="Blink-value-ship")}
    )
    monkeypatch.setattr("blink.reference.registry.load_agent", lambda *a, **k: SimpleNamespace(name="DM-9M"))
    head = {"games": 2, "pgn": "h.pgn", "penta": None}
    fake = {"gauntlet": {"anchors": []}, "blink_vs_dm": head, "games": 2, "pgns": []}
    monkeypatch.setattr(anchors, "run_dm_block", lambda *args: fake)
    anchors.e7_block(context(tmp_path), {**FAILED, "E3": {"mode": "value"}})
    assert seen["dm"] == ("dm:9M", "60+0.6")


def test_e6s_stockfish_anchor_follows_the_self_check(tmp_path):
    ctx = context(tmp_path)
    fallback = ladder._ladder_agent("SF1320", ctx, FAILED)
    assert fallback.clock == match.Clock(base_s=60.0, inc_s=0.6, margin_s=0.1) and fallback.name == "SF1320"
    assert ladder._ladder_agent("SF1320", ctx, PASSED).clock == match.Clock(move_s=0.1, margin_s=0.1)


def test_the_done_when_gates_name_every_failure():
    no_forfeit = {"games": 2, "time_forfeits": 0, "forfeits": 0, "adjudications": 0}
    forfeit = {**no_forfeit, "time_forfeits": 1}
    state = {
        "E0": {**FAILED["E0"], "forfeits": {"SF1800": forfeit, "SF1800-slow": no_forfeit}},
        "E5": {"forfeits": {"SF1500": forfeit, "Blink-value-ship": no_forfeit}},
        "E8": {"forfeits": {"SF19": {**no_forfeit, "forfeits": 1}}},
    }
    failures = orchestrate.gate_failures(state)
    assert len(failures) == 3
    assert any(f.startswith("E0: the SF self-check failed") and "60+0.6" in f for f in failures)
    assert any(f.startswith("E0: SF1800") for f in failures) and any(
        f.startswith("E8: SF19") for f in failures
    )
    assert orchestrate.gate_failures({"E0": PASSED["E0"], "E5": {"forfeits": {"SF1500": forfeit}}}) == []


def test_a_failed_gate_makes_blink_eval_exit_non_zero(capsys):
    from blink.commands import evaluate

    assert (
        evaluate._run_guarded("blink eval all", lambda: {"gate_failures": ["E0: the SF self-check failed"]})
        == 1
    )
    assert "done-when gate failed: E0: the SF self-check failed" in capsys.readouterr().err
    assert evaluate._run_guarded("blink eval all", lambda: {"gate_failures": []}) == 0


def test_the_fallback_control_is_the_plans():
    assert anchors.FALLBACK_TC == "60+0.6"
    assert (
        fastchess.with_tc(fastchess.stockfish_anchor(1800, fastchess.stockfish_exe()), "60+0.6").tc
        == "60+0.6"
    )
