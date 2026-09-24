"""The check rows also score games10k (a07's metric) and the mateset (a08's guard), for raw and EMA.

Each file is loaded once per run, at the first check; a run without them trains as before and says
once what it did not score. The sweep reads the last check row, so a07 and a08 can now be judged.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_train_config  # noqa: E402

from blink.board.value import CP_NONE  # noqa: E402
from blink.data import mateset, valprobe  # noqa: E402
from blink.train import checksets, games10k_eval, loop, mateset_eval, sweep, vaa  # noqa: E402
from blink.train.source import InMemorySource  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"
CHECKS = {2: "5%", 10: "25%", 12: "30%", 20: "50%", 40: "100%"}
GAMES_KEYS = {"games10k_top1", "ema_games10k_top1", "games10k_n"}
MATE_KEYS = {"shortest_mate", "ema_shortest_mate", "mate_preserving", "ema_mate_preserving", "mateset_n"}


def _games_file(tmp_path, n: int = 20):
    path = tmp_path / "games10k.npy"
    np.save(path, fixture_records()[:n])
    return path


def _mateset_file(tmp_path, labelled: bool = True):
    roots = fixture_records()[:12].copy()
    roots["cp"], roots["mate"] = CP_NONE, 3
    arrays = mateset.build(roots, n=100)
    if labelled:
        arrays["child_mate_in"] = np.where(arrays["child_is_best"], 3, 0).astype(np.int8)
    path = tmp_path / "mateset.npz"
    valprobe.save_npz(path, arrays)
    return path


def _train(tmp_path, max_steps: int | None = None, **spec):
    run_dir, logs = tmp_path / "run", []
    probe = vaa.probe_from_roots(fixture_records()[:20])
    cfg = tiny_train_config(steps=40, eval_every=15, vaa_subset=10, keep_last=1, batch_size=16)
    run_spec = loop.RunSpec(run_dir=run_dir, world=WORLD, device="cpu", max_steps=max_steps, **spec)
    source = InMemorySource(fixture_records()[:48], 16, seed=3).batches
    loop.train(cfg, run_spec, source, val=fixture_records(), log=logs.append, probe=probe)
    rows = [json.loads(line) for line in (run_dir / "evals.jsonl").read_text(encoding="utf-8").splitlines()]
    return {row["step"]: row for row in rows}, logs


def test_check_rows_carry_games10k_top1_and_the_mate_rates_for_raw_and_ema(tmp_path):
    rows, _ = _train(tmp_path, games10k=_games_file(tmp_path), mateset=_mateset_file(tmp_path))
    assert {step: row.get("check") for step, row in rows.items() if "check" in row} == CHECKS
    for step in CHECKS:
        assert set(rows[step]) >= GAMES_KEYS | MATE_KEYS
        assert (rows[step]["games10k_n"], rows[step]["mateset_n"]) == (20, 12)
        rates = (GAMES_KEYS | MATE_KEYS) - {"games10k_n", "mateset_n"}
        assert all(0.0 <= rows[step][key] <= 1.0 for key in rates)
    for step in (0, 15, 30):  # the 2k-step rows stay as cheap as before
        assert not (GAMES_KEYS | MATE_KEYS) & set(rows[step])


def test_each_file_is_loaded_once_per_run_and_only_when_a_check_comes(tmp_path, monkeypatch):
    calls = []
    for module in (games10k_eval, mateset_eval):
        real = module.load
        monkeypatch.setattr(module, "load", lambda path, real=real: calls.append(path.name) or real(path))
    files = {"games10k": _games_file(tmp_path), "mateset": _mateset_file(tmp_path)}
    _train(tmp_path / "early", max_steps=1, **files)
    assert calls == []  # no check before step 2: nothing is read
    _train(tmp_path / "full", **files)
    assert sorted(calls) == ["games10k.npy", "mateset.npz"]  # five checks, one load each


def test_a_run_without_the_files_trains_and_says_once_what_it_skipped(tmp_path):
    rows, logs = _train(tmp_path, games10k=tmp_path / "absent.npy")
    assert 40 in rows and rows[40]["check"] == "100%" and "vaa" in rows[40]
    assert not (GAMES_KEYS | MATE_KEYS) & set(rows[40])
    games = [line for line in logs if line.startswith("games10k:")]
    mates = [line for line in logs if line.startswith("mateset:")]
    assert len(games) == 1 and "absent.npy" in games[0] and "not found" in games[0]
    assert len(mates) == 1 and "mate_preserving" in mates[0]


def test_a_mateset_without_child_mate_in_scores_shortest_mate_and_names_what_is_missing(tmp_path):
    rows, logs = _train(tmp_path, mateset=_mateset_file(tmp_path, labelled=False))
    assert {"shortest_mate", "ema_shortest_mate", "mateset_n"} <= set(rows[40])
    assert "mate_preserving" not in rows[40] and "ema_mate_preserving" not in rows[40]
    missing = [line for line in logs if "child_mate_in" in line]
    assert len(missing) == 1 and "mate_preserving" in missing[0]


def test_the_sweep_judges_a07_and_a08_on_what_the_last_check_row_holds(tmp_path):
    files = {"games10k": _games_file(tmp_path), "mateset": _mateset_file(tmp_path)}
    _train(tmp_path, **files)
    final = sweep.final_metrics(tmp_path / "run")
    assert final["check"] == "100%" and set(final) >= GAMES_KEYS | MATE_KEYS
    floor = sweep.noise_floor({name: final for name in ("a01", "a02", "a03")}, ("a01", "a02", "a03"))
    a07 = sweep.decide(sweep.Arm("a07", "no rebalancing", judged_on="games10k_top1"), final, floor)
    assert "not judged" not in a07["reason"] and a07["value"] == final["games10k_top1"]
    a08 = sweep.Arm("a08", "DeepMind mapping", guard="mate_preserving")
    assert "not judged" not in sweep.decide(a08, final, floor)["reason"]
    lost = sweep.decide(a08, {**final, "mate_preserving": final["mate_preserving"] - 0.03}, floor)
    assert lost["adopt"] is False and "mate_preserving" in lost["reason"]


def test_the_lazy_set_logs_an_unreadable_file_once_and_keeps_training(tmp_path):
    broken = tmp_path / "games10k.npy"
    np.save(broken, np.zeros(3, dtype=np.int64))
    logs = []
    lazy = checksets.Lazy("games10k", broken, games10k_eval.load, "games10k_top1")
    assert lazy.get(logs.append) is None and lazy.get(logs.append) is None
    assert len(logs) == 1 and "games10k_top1" in logs[0] and "board" in logs[0]


def _half_written_mateset(tmp_path):
    whole = _mateset_file(tmp_path).read_bytes()
    path = tmp_path / "half.npz"
    path.write_bytes(whole[: len(whole) // 2])
    return path


@pytest.mark.parametrize(
    ("name", "loader", "damaged"),
    [
        ("games10k", games10k_eval.load, lambda tmp: (tmp / "empty.npy", b"")),
        ("mateset", mateset_eval.load, lambda tmp: (tmp / "empty.npz", b"")),
        ("mateset", mateset_eval.load, lambda tmp: (_half_written_mateset(tmp), None)),
    ],
    ids=["empty games10k", "empty mateset", "truncated mateset"],
)
def test_the_lazy_set_logs_an_empty_or_truncated_file_and_keeps_training(tmp_path, name, loader, damaged):
    """np.load raises EOFError on an empty file and zipfile.BadZipFile on half an .npz: neither is an
    OSError or a ValueError, and either would have ended the run at its first check."""
    path, content = damaged(tmp_path)
    if content is not None:
        path.write_bytes(content)
    logs = []
    lazy = checksets.Lazy(name, path, loader, "its metrics")
    assert lazy.get(logs.append) is None and lazy.get(logs.append) is None
    assert len(logs) == 1 and "cannot use" in logs[0] and "its metrics not scored" in logs[0]


def test_a_check_set_that_fails_to_score_leaves_the_row_and_the_run_going(tmp_path, monkeypatch):
    """An evaluation input or result never ends a run: the row keeps its VAA without the set's keys."""

    def out_of_memory(run, chunk, tick=None):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(checksets, "metrics", out_of_memory)
    rows, logs = _train(tmp_path, games10k=_games_file(tmp_path), mateset=_mateset_file(tmp_path))
    assert {step for step, row in rows.items() if "check" in row} == set(CHECKS)
    assert all("vaa" in rows[step] and not (GAMES_KEYS | MATE_KEYS) & set(rows[step]) for step in CHECKS)
    skipped = [line for line in logs if "CUDA out of memory" in line]
    assert len(skipped) == len(CHECKS) and "not scored" in skipped[0]


def test_weights_that_go_nan_before_a_check_reach_the_supervisor_as_a_non_finite_loss(tmp_path, monkeypatch):
    """The mateset's tie-break found no child near a NaN best value and raised ValueError at the 5% check:
    the supervisor took that for an ordinary crash and resumed at the same LR instead of rolling back.
    The check now scores what it can, and the next metrics row raises FloatingPointError as before."""
    real = loop._train_step

    def diverging(run, data, lr):
        out = real(run, data, lr)
        if run.step == 1:  # this step makes run.step 2: the 5% check
            with torch.no_grad():
                for parameter in run.model.parameters():
                    parameter.fill_(float("nan"))
        return out

    monkeypatch.setattr(loop, "_train_step", diverging)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        _train(tmp_path, games10k=_games_file(tmp_path), mateset=_mateset_file(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / "run" / "evals.jsonl").read_text().splitlines()]
    assert {row["step"]: row.get("check") for row in rows}[2] == "5%"


@pytest.mark.cuda
def test_a_cuda_run_scores_both_sets_at_its_checks(tmp_path):
    """On CUDA, games10k runs in fp32 chunks and the mateset in VAA's bf16 chunks, as on the GPU box."""
    run_dir = tmp_path / "gpu"
    probe = vaa.probe_from_roots(fixture_records()[:20])
    cfg = tiny_train_config(steps=20, eval_every=20, vaa_subset=10, batch_size=16)
    files = {"games10k": _games_file(tmp_path), "mateset": _mateset_file(tmp_path)}
    spec = loop.RunSpec(run_dir=run_dir, world=WORLD, device="cuda", **files)
    source = InMemorySource(fixture_records()[:48], 16, seed=3).batches
    loop.train(cfg, spec, source, val=fixture_records(), log=lambda _: None, probe=probe)
    final = sweep.final_metrics(run_dir)
    assert final["check"] == "100%" and set(final) >= GAMES_KEYS | MATE_KEYS
    assert (final["games10k_n"], final["mateset_n"]) == (20, 12)
