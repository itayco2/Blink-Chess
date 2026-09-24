"""Post-hoc arm metrics: `blink eval arm-metrics` and `blink sweep rescore` (blink.train.posthoc).

Arms a01-a05, a11 and a12 ran from a commit whose checks did not score games10k or the mateset, yet
a07 and a08 are judged against the a01-a03 noise floor of exactly those metrics. A finished run's
final checkpoint is scored afterwards as its last check row would have scored it, into posthoc.json,
which sweep.final_metrics merges; evals.jsonl is never rewritten.
"""

import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_train_config  # noqa: E402

from blink import cli  # noqa: E402
from blink.board.value import CP_NONE  # noqa: E402
from blink.data import mateset, valprobe  # noqa: E402
from blink.train import loop, posthoc, sweep, vaa  # noqa: E402
from blink.train.source import InMemorySource  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"
STEPS = 20
METRIC_KEYS = {
    "games10k_top1",
    "ema_games10k_top1",
    "games10k_n",
    "shortest_mate",
    "ema_shortest_mate",
    "mate_preserving",
    "ema_mate_preserving",
    "mateset_n",
}


def _inputs(folder):
    """A 20-position games10k and a pack whose 12-root mateset labels every child's mate-in."""
    games = folder / "games10k.npy"
    np.save(games, fixture_records()[:20])
    pack = folder / "pack"
    pack.mkdir()
    roots = fixture_records()[:12].copy()
    roots["cp"], roots["mate"] = CP_NONE, 3
    arrays = mateset.build(roots, n=100)
    arrays["child_mate_in"] = np.where(arrays["child_is_best"], 3, 0).astype(np.int8)
    valprobe.save_npz(pack / mateset.OUTPUT, arrays)
    return games, pack


def _train(run_dir, pack, **files):
    cfg = tiny_train_config(steps=STEPS, eval_every=10, vaa_subset=10, keep_last=1, batch_size=16)
    spec = loop.RunSpec(run_dir=run_dir, world=WORLD, device="cpu", data={"dir": str(pack)}, **files)
    source = InMemorySource(fixture_records()[:48], 16, seed=3).batches
    probe = vaa.probe_from_roots(fixture_records()[:20])
    loop.train(cfg, spec, source, val=fixture_records(), log=lambda _: None, probe=probe)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """Two finished tiny runs: `frozen` scored neither set at its checks, `current` scored both."""
    root = tmp_path_factory.mktemp("finished")
    games, pack = _inputs(root)
    _train(root / "frozen", pack)
    _train(root / "current", pack, games10k=games, mateset=pack / mateset.OUTPUT)
    return root


@pytest.fixture
def home(tmp_path, monkeypatch, finished):
    """BLINK_HOME with games10k in its data folder and copies of both runs (each test may change them)."""
    home = tmp_path / "home"
    monkeypatch.setenv("BLINK_HOME", str(home))
    (home / "data").mkdir(parents=True)
    shutil.copy(finished / "games10k.npy", home / "data" / "games10k.npy")
    for name in ("frozen", "current"):
        shutil.copytree(finished / name, home / "runs" / f"abl-{name}")
    return home


def _arm_metrics(run: str, *extra: str) -> int:
    return cli.main(["eval", "arm-metrics", "--run", run, "--device", "cpu", *extra])


def test_arm_metrics_scores_the_final_checkpoint_into_posthoc_json_and_leaves_evals_alone(home, finished):
    run_dir = home / "runs" / "abl-frozen"
    history = (run_dir / "evals.jsonl").read_bytes()
    assert not METRIC_KEYS & set(sweep.final_metrics(run_dir))
    assert _arm_metrics("abl-frozen", "--data", str(finished / "pack")) == 0
    record = json.loads((run_dir / posthoc.POSTHOC).read_text(encoding="utf-8"))
    assert record["step"] == STEPS and record["checkpoint"] == "ckpt_000000020.pt"
    assert set(record["metrics"]) == METRIC_KEYS
    assert record["metrics"]["games10k_n"] == 20 and record["metrics"]["mateset_n"] == 12
    assert (run_dir / "evals.jsonl").read_bytes() == history
    final = sweep.final_metrics(run_dir)
    assert final["check"] == "100%" and set(final) >= METRIC_KEYS and "vaa" in final


def test_the_posthoc_scores_are_the_ones_the_last_check_row_wrote(home, finished):
    """Same weights, same functions, same chunks: the two routes agree exactly, so arms scored either
    way share one noise floor."""
    run_dir = home / "runs" / "abl-current"
    row = sweep.final_metrics(run_dir)
    games, mates = home / "data" / "games10k.npy", finished / "pack" / mateset.OUTPUT
    record = posthoc.score_run(run_dir, games, mates, device="cpu", log=lambda _: None)
    assert record["metrics"] == {key: row[key] for key in METRIC_KEYS}


def test_rescoring_is_idempotent_and_new_inputs_or_force_score_again(home, finished, monkeypatch):
    run_dir = home / "runs" / "abl-frozen"
    games, mates = home / "data" / "games10k.npy", finished / "pack" / mateset.OUTPUT
    first = posthoc.score_run(run_dir, games, mates, device="cpu", log=lambda _: None)
    loads, real = [], posthoc.load_weights
    monkeypatch.setattr(posthoc, "load_weights", lambda *a, **k: loads.append(a) or real(*a, **k))
    logs = []
    assert posthoc.score_run(run_dir, games, mates, device="cpu", log=logs.append) == first
    assert loads == [] and any("already scored" in line for line in logs)
    np.save(games, fixture_records()[:10])  # games10k was rebuilt
    second = posthoc.score_run(run_dir, games, mates, device="cpu", log=lambda _: None)
    assert len(loads) == 1 and second["metrics"]["games10k_n"] == 10
    posthoc.score_run(run_dir, games, mates, device="cpu", log=lambda _: None, force=True)
    assert len(loads) == 2


def test_a_run_without_its_final_checkpoint_is_refused(home, finished, capsys):
    run_dir = home / "runs" / "abl-frozen"
    (run_dir / "ckpt_000000020.pt").unlink()  # as if it stopped before its last step
    assert _arm_metrics("abl-frozen", "--data", str(finished / "pack")) == 2
    assert "of 20" in capsys.readouterr().err and not (run_dir / posthoc.POSTHOC).exists()
    (home / "runs" / "abl-empty").mkdir()
    assert _arm_metrics("abl-empty") == 2
    assert "no checkpoint" in capsys.readouterr().err
    assert _arm_metrics("../abl-frozen") == 2


def test_arm_metrics_reads_the_runs_own_pack_and_refuses_when_there_is_nothing_to_score(
    home, finished, tmp_path, capsys
):
    assert _arm_metrics("abl-frozen") == 0  # --data defaults to the pack in the run's config.json
    record = json.loads((home / "runs" / "abl-frozen" / posthoc.POSTHOC).read_text(encoding="utf-8"))
    assert record["inputs"]["mateset"]["path"] == str(finished / "pack" / mateset.OUTPUT)
    assert "abl-frozen" in capsys.readouterr().out
    (home / "data" / "games10k.npy").unlink()
    assert _arm_metrics("abl-current", "--data", str(tmp_path / "no-pack")) == 2
    assert "nothing to score" in capsys.readouterr().err


def _ablation_state(home, out, arms: dict[str, str]) -> None:
    entries = {}
    for arm, source in arms.items():
        run = f"abl-{arm}"
        shutil.copytree(home / "runs" / f"abl-{source}", home / "runs" / run)
        metrics = sweep.final_metrics(home / "runs" / run)
        entries[arm] = {"name": arm, "run": run, "status": "finished", "metrics": metrics}
    out.write_text(json.dumps({"arms": entries}), encoding="utf-8")


def _rescore_plan(folder, pack) -> str:
    folder.mkdir()
    for arm in ("a01", "a02", "a03"):
        (folder / f"{arm}.toml").write_text(
            f'[arm]\nchange = "D"\n[train]\nseed = {arm[-1]}\n', encoding="utf-8"
        )
    (folder / "a07.toml").write_text('[arm]\nchange = "x"\njudged_on = "games10k_top1"\n', encoding="utf-8")
    (folder / "a08.toml").write_text('[arm]\nchange = "y"\nguard = "mate_preserving"\n', encoding="utf-8")
    plan = folder / "plan.toml"
    plan.write_text(
        f'[plan]\nrecipe = "recipe.toml"\ndata = "{pack.as_posix()}"\nhours = 1.5\nsize = "s"\n'
        'sigma_arms = ["a01", "a02", "a03"]\narms = ["a01", "a02", "a03", "a07", "a08"]\n',
        encoding="utf-8",
    )
    return str(plan)


def test_sweep_rescore_scores_every_finished_arm_so_a07_and_a08_are_judged(home, finished, tmp_path, capsys):
    out = home / "eval" / "ablations.json"
    out.parent.mkdir()
    _ablation_state(home, out, {"a01": "frozen", "a02": "frozen", "a03": "frozen", "a07": "current"})
    plan = _rescore_plan(tmp_path / "ablations", finished / "pack")
    assert sweep.load_plan(plan).arms[3].judged_on == "games10k_top1"
    assert cli.main(["sweep", "rescore", "--plan", plan, "--device", "cpu"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "not judged" not in report["decisions"]["a07"]["reason"]  # the floor has games10k_top1 now
    assert report["decisions"]["a08"]["reason"] == "not judged: not run"
    assert report["noise"]["mate_preserving"]["sigma"] == pytest.approx(0.0)
    assert report["arms"]["a01"]["games10k_top1"] is not None
    for arm in ("a01", "a02", "a03", "a07"):
        assert (home / "runs" / f"abl-{arm}" / posthoc.POSTHOC).is_file()
    assert "a07:" in capsys.readouterr().out
    assert cli.main(["sweep", "rescore", "--plan", plan, "--device", "cpu"]) == 0  # nothing to redo
    assert "already scored" in capsys.readouterr().out
