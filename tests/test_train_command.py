import json

import numpy as np
import pytest
import zstandard

torch = pytest.importorskip("torch")

from train_helpers import FIXTURE, fixture_records  # noqa: E402

from blink import cli  # noqa: E402
from blink.data.record import CHILD_DTYPE, ROOT_DTYPE  # noqa: E402

pytestmark = pytest.mark.torch

CONFIG = """
[model]
d_model = 64
n_layers = 1
n_heads = 2

[train]
batch_size = 16
steps = 30
warmup_steps = 5
metrics_every = 10
eval_every = 15
val_size = 64
ckpt_every_steps = 15
ckpt_every_minutes = 0.0
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "tiny.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def raw(tmp_path):
    path = tmp_path / "evals.jsonl.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(FIXTURE.read_bytes()))
    return path


def test_train_from_a_raw_zst_writes_a_run_and_a_cache(home, config, raw, capsys):
    argv = [
        "train",
        "--config",
        str(config),
        "--run",
        "smoke",
        "--source-raw",
        str(raw),
        "--max-lines",
        "100",
    ]
    assert cli.main([*argv, "--device", "cpu", "--workers", "1"]) == 0
    run_dir = home / "runs" / "smoke"
    assert (run_dir / "ckpt_000000030.pt").exists()
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["data"]["source"] == "raw" and saved["data"]["records"] == len(fixture_records())
    assert list((home / "data" / "raw-cache").glob("*.npy"))
    assert "finished" in capsys.readouterr().out


def test_resume_through_the_cli_continues_to_the_end(home, config, raw):
    base = ["train", "--config", str(config), "--run", "r", "--source-raw", str(raw), "--max-lines", "100"]
    assert cli.main([*base, "--device", "cpu", "--workers", "1", "--max-steps", "15"]) == 0
    assert cli.main([*base, "--device", "cpu", "--workers", "1"]) == 2  # exists, no --resume
    assert cli.main([*base, "--device", "cpu", "--workers", "1", "--resume"]) == 0
    beat = json.loads((home / "runs" / "r" / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["state"] == "finished" and beat["step"] == 30


def test_resume_with_different_data_is_refused_as_a_different_world(home, config, raw):
    base = ["train", "--config", str(config), "--run", "w", "--source-raw", str(raw), "--device", "cpu"]
    assert cli.main([*base, "--max-lines", "100", "--workers", "1", "--max-steps", "15"]) == 0
    assert cli.main([*base, "--max-lines", "90", "--workers", "1", "--resume"]) == 2


class FakeShardLoader:
    """The data area's ShardLoader interface: batch b of the stream starts at start_batch."""

    calls: list[dict] = []

    def __init__(self, paths, batch_size, seed, loop=True, start_batch=0, dtype=ROOT_DTYPE):
        self.records = np.concatenate([np.fromfile(p, dtype=dtype) for p in paths])
        self.batch_size, self.start_batch = batch_size, start_batch
        names = [p.name for p in paths]
        FakeShardLoader.calls.append({"names": names, "batch_size": batch_size, "dtype": dtype})

    def __iter__(self):
        per_epoch = len(self.records) // self.batch_size
        batch = self.start_batch
        while True:
            i = batch % per_epoch
            yield self.records[i * self.batch_size : (i + 1) * self.batch_size]
            batch += 1


@pytest.fixture
def fake_loader(monkeypatch):
    import sys
    import types

    FakeShardLoader.calls = []
    fake = types.ModuleType("blink.data.loader")
    fake.ShardLoader = FakeShardLoader
    monkeypatch.setitem(sys.modules, "blink.data.loader", fake)
    return FakeShardLoader


def test_train_from_a_skeleton_shard_directory_uses_the_shard_loader(home, config, tmp_path, fake_loader):
    shards = tmp_path / "shards"
    shards.mkdir()
    fixture_records().tofile(shards / "train_000.bin")
    fixture_records()[:40].tofile(shards / "val.bin")
    (shards / "manifest.json").write_text(json.dumps({"train": 100}), encoding="utf-8")
    argv = ["train", "--config", str(config), "--run", "shards", "--data", str(shards), "--device", "cpu"]
    assert cli.main(argv) == 0
    saved = json.loads((home / "runs" / "shards" / "config.json").read_text(encoding="utf-8"))
    assert saved["data"]["train_shards"] == 1 and saved["data"]["val_records"] == 40
    assert saved["data"]["child_shards"] == 0 and fake_loader.calls[0]["names"] == ["train_000.bin"]


def _v1_pack(root, weights: list[float] | None = None) -> None:
    from test_mixed_training import children_from

    from blink.train import vaa

    root.mkdir()
    records = fixture_records()
    records[:60].tofile(root / "train_r000.bin")
    records[60:].tofile(root / "train_r001.bin")
    children_from(records).tofile(root / "train_c000.bin")
    records[:40].tofile(root / "val_roots.bin")
    vaa.save_probe(root / "valprobe.npz", vaa.probe_from_roots(records[:12]))
    manifest = {"world": "v1v1v1v1v1v1"}
    if weights is not None:
        manifest["rebalance"] = {"buckets": 48, "weights": weights, "definition": "test"}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


MIXED_CONFIG = CONFIG.replace("warmup_steps = 5", "warmup_steps = 5\nchild_frac = 0.25")


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_v1_pack_mixes_root_and_child_shards_with_the_manifest_weights(
    home, tmp_path, fake_loader, monkeypatch
):
    from blink.data import rebalance

    calls = []

    def weights_for(records, weights):
        calls.append((records.dtype, len(weights)))
        return np.full(len(records), 2.0, dtype=np.float32)

    # Spy on the real module (P2 built it): replacing the module would hide the names the CLI's
    # other areas import from it, such as bigpack's GamesHistogram.
    monkeypatch.setattr(rebalance, "weights_for", weights_for)
    _v1_pack(tmp_path / "v1", weights=[1.0] * 48)
    config = tmp_path / "mixed.toml"
    config.write_text(MIXED_CONFIG, encoding="utf-8")
    argv = [
        "train",
        "--config",
        str(config),
        "--run",
        "v1",
        "--data",
        str(tmp_path / "v1"),
        "--device",
        "cpu",
    ]
    assert cli.main(argv) == 0
    saved = json.loads((home / "runs" / "v1" / "config.json").read_text(encoding="utf-8"))
    assert saved["world"] == "v1v1v1v1v1v1"
    assert (saved["data"]["roots_per_step"], saved["data"]["children_per_step"]) == (12, 4)
    assert saved["data"]["rebalance"] and saved["data"]["child_shards"] == 1
    assert {c["dtype"] for c in fake_loader.calls} == {ROOT_DTYPE, CHILD_DTYPE}
    assert (ROOT_DTYPE, 48) in calls and (CHILD_DTYPE, 48) in calls
    evals = _jsonl(home / "runs" / "v1" / "evals.jsonl")
    assert all("ema_vaa" in row for row in evals) and "vaa" in evals[-1] and evals[-1]["vaa_n"] == 12


def _pack_mateset(root) -> None:
    from blink.board.value import CP_NONE
    from blink.data import mateset, valprobe

    roots = fixture_records()[:8].copy()
    roots["cp"], roots["mate"] = CP_NONE, 2
    valprobe.save_npz(root / "mateset.npz", mateset.build(roots, n=8))


def test_a_v1_pack_run_scores_the_packs_mateset_and_blink_home_games10k_at_its_checks(
    home, tmp_path, fake_loader
):
    _v1_pack(tmp_path / "v1")
    _pack_mateset(tmp_path / "v1")
    (home / "data").mkdir(parents=True)
    np.save(home / "data" / "games10k.npy", fixture_records()[:30])
    config = tmp_path / "mixed.toml"
    config.write_text(MIXED_CONFIG, encoding="utf-8")
    argv = [
        "train",
        "--config",
        str(config),
        "--run",
        "v1",
        "--data",
        str(tmp_path / "v1"),
        "--device",
        "cpu",
    ]
    assert cli.main(argv) == 0
    evals = _jsonl(home / "runs" / "v1" / "evals.jsonl")
    final = evals[-1]
    assert final["check"] == "100%" and final["games10k_n"] == 30 and final["mateset_n"] == 8
    assert {"games10k_top1", "ema_games10k_top1", "shortest_mate", "ema_shortest_mate"} <= set(final)
    assert "games10k_top1" not in evals[0]  # step 0 is not a check


def test_games10k_can_be_pointed_elsewhere_and_a_raw_source_has_no_pack_mateset(home, config, raw, tmp_path):
    from blink.commands import train_data
    from blink.model.config import load_config

    elsewhere = tmp_path / "held-out.npy"
    args = cli.build_parser().parse_args(
        ["train", "--config", str(config), "--run", "g", "--source-raw", str(raw), "--max-lines", "100"]
    )
    args.workers = 1
    plan = train_data.plan(args, load_config(config))
    assert plan.games10k == home / "data" / "games10k.npy" and plan.mateset is None
    args.games10k = str(elsewhere)
    assert train_data.plan(args, load_config(config)).games10k == elsewhere


def test_a_pack_without_a_valprobe_says_vaa_will_not_be_recorded(tmp_path, capsys):
    from argparse import Namespace

    from blink.commands import train_data
    from blink.model.config import TrainConfig

    _v1_pack(tmp_path / "v1", weights=[1.0] * 48)
    (tmp_path / "v1" / "valprobe.npz").unlink()
    cfg = TrainConfig(batch_size=20, child_frac=0.25, val_size=8)
    plan = train_data.plan(Namespace(source_raw=None, data=str(tmp_path / "v1"), valprobe=None), cfg)
    assert plan.probe is None
    assert "valprobe: none, so VAA will not be recorded" in capsys.readouterr().out


def test_a_v1_world_names_the_packs_blocklist_sha_and_grouped_salt():
    """WORLD = sha1(contract, manifest sha, blocklist sha, split rule and salt): the v1 keys feed it."""
    import hashlib

    from blink.commands.train_data import pack_world
    from blink.train.world import NO_BLOCKLIST, world_id

    v1 = {"blocklist": {"sha256": "ab" * 32}, "grouped": {"salt": 7}, "split_rule": "rule"}
    raw = json.dumps(v1).encode("utf-8")
    sha = hashlib.sha1(raw).hexdigest()
    assert pack_world(raw, v1) == world_id(sha, "ab" * 32, "rule; grouped salt 7")
    skeleton = {"blocklist": None, "split_rule": "rule"}  # the P1 layout keeps its world
    assert pack_world(raw, skeleton) == world_id(sha, NO_BLOCKLIST, "rule")
    assert pack_world(raw, {**v1, "world": "fixedworld12"}) == "fixedworld12"


def test_a_config_with_children_is_refused_on_a_pack_without_child_shards(
    home, tmp_path, fake_loader, capsys
):
    shards = tmp_path / "shards"
    shards.mkdir()
    fixture_records().tofile(shards / "train_000.bin")
    (shards / "manifest.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "mixed.toml"
    config.write_text(MIXED_CONFIG, encoding="utf-8")
    argv = ["train", "--config", str(config), "--run", "x", "--data", str(shards), "--device", "cpu"]
    assert cli.main(argv) == 2
    assert "child_frac = 0" in capsys.readouterr().err


def test_lr_scale_needs_resume_and_a_positive_factor(home, config, raw):
    base = ["train", "--config", str(config), "--run", "s", "--source-raw", str(raw), "--device", "cpu"]
    assert cli.main([*base, "--lr-scale", "0.5"]) == 2
    assert cli.main([*base, "--resume", "--lr-scale", "0"]) == 2


def test_resume_with_lr_scale_through_the_cli_halves_the_learning_rate(home, config, raw):
    base = ["train", "--config", str(config), "--run", "h", "--source-raw", str(raw), "--max-lines", "100"]
    base += ["--device", "cpu", "--workers", "1"]
    assert cli.main([*base, "--max-steps", "15"]) == 0
    assert cli.main([*base, "--resume", "--lr-scale", "0.5", "--max-steps", "20"]) == 0
    lr = {row["step"]: row["lr"] for row in _jsonl(home / "runs" / "h" / "metrics.jsonl")}
    assert lr[20] == pytest.approx(0.5 * 1e-3)


def test_preview_cooldown_branches_into_name_preview_and_leaves_the_main_run_alone(home, config, raw):
    base = ["train", "--run", "long", "--source-raw", str(raw), "--max-lines", "100", "--device", "cpu"]
    base += ["--workers", "1"]
    assert cli.main([*base, "--config", str(config), "--max-steps", "15"]) == 0
    main = home / "runs" / "long"
    rows = [{"step": s, "samples_per_s": 160.0, "phase": "train"} for s in (5, 10, 15)]
    (main / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in main.iterdir() if p.is_file()}
    assert cli.main([*base, "--preview-cooldown", "2s", "--from-step", "30"]) == 2  # no such checkpoint
    assert cli.main([*base, "--preview-cooldown", "2s", "--from-step", "15"]) == 0
    assert {p.name: p.read_bytes() for p in main.iterdir() if p.is_file()} == before
    preview = json.loads((home / "runs" / "long-preview" / "config.json").read_text(encoding="utf-8"))
    assert preview["config"]["steps"] == 15 + 20 and preview["branched_from"].endswith("ckpt_000000015.pt")
    beat = json.loads((home / "runs" / "long-preview" / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["state"] == "finished" and beat["step"] == 35


def test_a_shard_directory_without_a_manifest_is_refused(home, config, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["train", "--config", str(config), "--run", "x", "--data", str(empty)]) == 2


def test_a_run_name_that_is_a_path_is_refused(home, config, raw):
    argv = ["train", "--config", str(config), "--run", "../escape", "--source-raw", str(raw)]
    assert cli.main(argv) == 2


def test_status_prints_the_run_and_exits_by_its_health(home, config, raw, capsys):
    argv = ["train", "--config", str(config), "--run", "st", "--source-raw", str(raw), "--max-lines", "100"]
    assert cli.main([*argv, "--device", "cpu", "--workers", "1"]) == 0
    capsys.readouterr()
    assert cli.main(["status", "--run", "st"]) == 0
    assert "FINISHED" in capsys.readouterr().out
    assert cli.main(["status", "--run", "missing"]) == 1
