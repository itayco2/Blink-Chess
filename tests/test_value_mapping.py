"""Arm a08: the DeepMind value mapping (no cp clamp, every mate = 1.0) for the value head's targets."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from blink.board import value as board_value
from blink.board.value import CP_NONE, LICHESS_K
from blink.model import value_mapping
from blink.model.config import VALUE_MAPPINGS, TrainConfig, config_from_dict, load_config, read_tables

REPO = Path(__file__).resolve().parents[1]


def _logistic(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-LICHESS_K * cp))


def _deepmind(cp: list[int], mate: list[int]) -> np.ndarray:
    return value_mapping.deepmind_win_probability_array(np.array(cp), np.array(mate))


def _lichess(cp: list[int], mate: list[int]) -> np.ndarray:
    return board_value.win_probability_array(np.array(cp), np.array(mate))


# ---------------------------------------------------------------- config


def test_the_value_mapping_defaults_to_todays_lichess_targets():
    assert TrainConfig().value_mapping == "lichess"
    assert VALUE_MAPPINGS == ("lichess", "deepmind")


@pytest.mark.parametrize("bad", ["stockfish", "DeepMind", "", 1, None])
def test_an_unknown_value_mapping_is_refused(bad):
    with pytest.raises(ValueError, match="value_mapping"):
        TrainConfig(value_mapping=bad)


def test_the_value_mapping_loads_from_the_train_table(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[train]\nvalue_mapping = "deepmind"\n', encoding="utf-8")
    assert load_config(path).value_mapping == "deepmind"


def test_the_a08_arm_loads_over_the_recipe_at_s():
    from blink.train import sweep

    arm = sweep.load_arm(REPO / "configs" / "ablations" / "a08.toml")
    merged = sweep.merged_config(read_tables(REPO / "configs" / "s.toml"), arm, steps=10_000)
    sweep.validate(merged)
    cfg = config_from_dict({**merged["train"], "model": merged["model"]})
    assert cfg.value_mapping == "deepmind" and cfg.model.gab is True


def test_a08_is_not_judged_until_the_a01_a03_noise_floor_carries_mate_preserving_too():
    """a08's guard is mate_preserving, which no evals.jsonl row carries yet, so a08 stays held in
    configs/ablations/plan.toml: run now, it would spend its 1.5 GPU-h and end "not judged". Scoring
    a08 alone would not help, because the floor arms a01-a03 must carry the metric as well."""
    from blink.train import sweep

    arm = sweep.load_arm(REPO / "configs" / "ablations" / "a08.toml")
    rows = {f"a0{i}": {"vaa": 0.500 + i / 1000, "top1": 0.300} for i in (1, 2, 3)}
    floor = sweep.noise_floor(rows, ("a01", "a02", "a03"))
    scored = {"vaa": 0.60, "top1": 0.31, "mate_preserving": 0.95}
    unscored = {"vaa": 0.60, "top1": 0.31}
    assert arm.guard == "mate_preserving" and "mate_preserving" not in floor
    for metrics in (scored, unscored):
        verdict = sweep.decide(arm, metrics, floor)
        assert verdict == {"adopt": False, "reason": "not judged: mate_preserving missing"}


# ---------------------------------------------------------------- the mapping


@pytest.mark.parametrize("cp", [0, 1, -1, 100, -250, 500, 999, -1000, 1000])
def test_deepmind_is_the_lichess_logistic_inside_the_clamp(cp):
    got = _deepmind([cp], [0])[0]
    assert got == pytest.approx(_logistic(cp), abs=1e-15)
    assert got == _lichess([cp], [0])[0]


@pytest.mark.parametrize("cp", [1001, 1500, -1500, 3000, -32767, 32767])
def test_deepmind_does_not_clamp_the_centipawns(cp):
    got = _deepmind([cp], [0])[0]
    assert got == pytest.approx(_logistic(cp), abs=1e-15)
    clamped = _lichess([cp], [0])[0]
    assert abs(got - 0.5) > abs(clamped - 0.5)  # further from even than Lichess's +-1000 ceiling


def test_hand_picked_values_match_the_deepmind_formula():
    # utils.centipawns_to_win_probability: 0.5 + 0.5 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
    for cp, expected in ((1500, 0.99602251), (-3000, 1.5946743e-5), (2000, 0.99936684), (-1200, 0.011908816)):
        assert _deepmind([cp], [0])[0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("mate", [1, 2, 5, 10, 15, 23, 127])
def test_every_mate_for_the_side_to_move_is_a_certain_win(mate):
    assert _deepmind([CP_NONE], [mate])[0] == 1.0
    assert _lichess([CP_NONE], [mate])[0] < 1.0


@pytest.mark.parametrize("mate", [-1, -2, -5, -10, -16, -128])
def test_every_mate_against_the_side_to_move_is_a_certain_loss(mate):
    assert _deepmind([CP_NONE], [mate])[0] == 0.0
    assert _lichess([CP_NONE], [mate])[0] > 0.0


def test_a_side_already_checkmated_has_lost_under_both_mappings():
    assert _deepmind([CP_NONE], [0])[0] == _lichess([CP_NONE], [0])[0] == 0.0


def test_deepmind_differs_from_lichess_only_beyond_the_clamp_and_on_mates():
    cp = np.concatenate([np.arange(-1500, 1501), np.full(40, CP_NONE)]).astype(np.int16)
    mate = np.concatenate([np.zeros(3001), np.arange(-20, 20)]).astype(np.int8)
    deepmind, lichess = value_mapping.deepmind_win_probability_array(cp, mate), _lichess(cp, mate)
    is_mate = cp == CP_NONE
    beyond = ~is_mate & (np.abs(cp.astype(np.int32)) > board_value.CP_CLAMP)
    differs = deepmind != lichess
    assert np.array_equal(differs, beyond | (is_mate & (mate != 0)))
    assert np.all((deepmind >= 0.0) & (deepmind <= 1.0))


def test_the_mapping_is_chosen_by_name_and_lichess_is_the_frozen_contract():
    records = _mixed_records()
    frozen = board_value.win_probability_array(records["cp"], records["mate"])
    got = value_mapping.win_probability_array(records["cp"], records["mate"], "lichess")
    assert np.array_equal(got, frozen)
    deepmind = value_mapping.win_probability_array(records["cp"], records["mate"], "deepmind")
    assert np.array_equal(
        deepmind, value_mapping.deepmind_win_probability_array(records["cp"], records["mate"])
    )
    with pytest.raises(ValueError, match="value_mapping"):
        value_mapping.win_probability_array(records["cp"], records["mate"], "stockfish")


def _mixed_records() -> np.ndarray:
    from train_helpers import fixture_records

    return fixture_records()


# ---------------------------------------------------------------- training targets


torch_tests = pytest.mark.torch


@torch_tests
def test_the_value_target_follows_the_mapping_and_the_policy_target_keeps_lichess_w():
    pytest.importorskip("torch")
    from blink.train.batch import make_batch

    records = _mixed_records()
    lichess, deepmind = make_batch(records, "cpu"), make_batch(records, "cpu", value_mapping="deepmind")
    assert lichess.w_value is lichess.w_best
    assert np.array_equal(deepmind.w_best.numpy(), lichess.w_best.numpy())
    assert np.array_equal(deepmind.w_alt.numpy(), lichess.w_alt.numpy())
    expected = value_mapping.deepmind_win_probability_array(records["cp"], records["mate"]).astype(np.float32)
    assert np.array_equal(deepmind.w_value.numpy(), expected)


@torch_tests
def test_the_child_value_target_follows_the_mapping():
    pytest.importorskip("torch")
    from test_mixed_training import children_from

    from blink.train.batch import make_child_batch

    children = children_from(_mixed_records())
    batch = make_child_batch(children, "cpu", value_mapping="deepmind")
    expected = value_mapping.deepmind_win_probability_array(children["cp"], children["mate"]).astype(
        np.float32
    )
    assert np.array_equal(batch.w.numpy(), expected)
    default = make_child_batch(children, "cpu")
    lichess = board_value.win_probability_array(children["cp"], children["mate"]).astype(np.float32)
    assert np.array_equal(default.w.numpy(), lichess)


@torch_tests
def test_only_the_value_loss_of_mates_and_scores_beyond_the_clamp_changes():
    torch = pytest.importorskip("torch")
    from test_mixed_training import children_from

    from blink.model import losses
    from blink.train.batch import make_batch, make_child_batch

    records = _mixed_records()
    roots_rec, child_rec = records[:60], children_from(records[60:])
    generator = torch.Generator().manual_seed(0)
    policy = torch.randn(len(records), 1880, generator=generator)
    values = torch.randn(len(records), 128, generator=generator)
    out = {}
    for mapping in VALUE_MAPPINGS:
        roots = make_batch(roots_rec, "cpu", value_mapping=mapping)
        children = make_child_batch(child_rec, "cpu", value_mapping=mapping)
        out[mapping] = losses.mixed_losses(policy, values, roots, children.w, alpha=0.5, tau=0.05)
    assert torch.equal(out["lichess"][0], out["deepmind"][0])  # the policy CE is untouched
    cp = np.concatenate([roots_rec["cp"], child_rec["cp"]]).astype(np.int32)
    mate = np.concatenate([roots_rec["mate"], child_rec["mate"]])
    moved = ((cp == CP_NONE) & (mate != 0)) | ((cp != CP_NONE) & (np.abs(cp) > board_value.CP_CLAMP))
    changed = (out["lichess"][1] != out["deepmind"][1]).numpy()
    assert moved.any() and np.array_equal(changed, moved)


@torch_tests
def test_the_validation_sample_is_scored_against_the_runs_own_value_targets():
    torch = pytest.importorskip("torch")
    from train_helpers import tiny_model_config

    from blink.model.transformer import BlinkNet
    from blink.train import telemetry

    records = _mixed_records()
    torch.manual_seed(0)
    model = BlinkNet(tiny_model_config())
    with torch.no_grad():
        model.value.out.weight.normal_(std=0.05)  # a trained value head: non-uniform bins
    scores = {}
    for mapping in VALUE_MAPPINGS:
        val = telemetry.make_val_set(records, "cpu", value_mapping=mapping)
        expected = value_mapping.win_probability_array(records["cp"], records["mate"], mapping)
        assert np.array_equal(val.batch.w_value.numpy(), expected.astype(np.float32))
        scores[mapping] = telemetry.evaluate(model, val, alpha=0.5, tau=0.05)
    assert scores["deepmind"]["policy_ce"] == scores["lichess"]["policy_ce"]
    assert scores["deepmind"]["top1"] == scores["lichess"]["top1"]
    assert scores["deepmind"]["value_ce"] != scores["lichess"]["value_ce"]
    assert scores["deepmind"]["win_mae"] != scores["lichess"]["win_mae"]


@torch_tests
def test_a_tiny_a08_run_trains_on_cpu_on_its_own_value_targets(tmp_path):
    pytest.importorskip("torch")
    from test_mixed_training import children_from
    from train_helpers import fixture_records, tiny_train_config

    from blink.train import loop
    from blink.train.source import InMemorySource, mixed_source

    records = fixture_records()
    rows = {}
    for mapping in VALUE_MAPPINGS:
        cfg = tiny_train_config(
            steps=12,
            warmup_steps=3,
            batch_size=16,
            child_frac=0.25,
            micro_batch=8,
            metrics_every=1,
            eval_every=1000,
            ckpt_every_steps=1000,
            value_mapping=mapping,
        )
        roots = InMemorySource(records[:48], 12, seed=3).batches
        children = InMemorySource(children_from(records[48:]), 4, seed=4).batches
        run_dir = tmp_path / mapping
        spec = loop.RunSpec(run_dir=run_dir, world="0123456789ab", device="cpu")
        result = loop.train(cfg, spec, mixed_source(roots, children), val=records, log=lambda _: None)
        assert result.step == 12
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        metrics = [json.loads(line) for line in lines]
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert saved["config"]["value_mapping"] == mapping
        assert all(np.isfinite(row["loss_policy"]) and np.isfinite(row["loss_value"]) for row in metrics)
        assert np.isfinite(result.last_eval["value_ce"])
        rows[mapping] = metrics
    lichess, deepmind = rows["lichess"], rows["deepmind"]
    # step 1 runs the same initial weights on the same rows, and the policy target is unchanged
    assert deepmind[0]["loss_policy"] == lichess[0]["loss_policy"]
    # the zero-initialised value head scores ln 128 on any target at step 1; from step 2 on it has
    # learned from different targets
    assert deepmind[1]["loss_value"] != lichess[1]["loss_value"]
