import json

import numpy as np
import pytest
import zstandard

torch = pytest.importorskip("torch")

from train_helpers import FIXTURE, fixture_records  # noqa: E402

from blink import cli  # noqa: E402
from blink.data.record import ROOT_DTYPE  # noqa: E402

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


def test_train_from_a_shard_directory_uses_the_shard_loader(home, config, tmp_path, monkeypatch):
    import sys
    import types

    shards = tmp_path / "shards"
    shards.mkdir()
    fixture_records().tofile(shards / "train_000.bin")
    fixture_records()[:40].tofile(shards / "val.bin")
    (shards / "manifest.json").write_text(json.dumps({"train": 100}), encoding="utf-8")

    class FakeShardLoader:
        def __init__(self, paths, batch_size, seed, loop=True):
            self.records = np.concatenate([np.fromfile(p, dtype=ROOT_DTYPE) for p in paths])
            self.batch_size = batch_size

        def __iter__(self):
            while True:
                for i in range(0, len(self.records) - self.batch_size + 1, self.batch_size):
                    yield self.records[i : i + self.batch_size]

    fake = types.ModuleType("blink.data.loader")
    fake.ShardLoader = FakeShardLoader
    monkeypatch.setitem(sys.modules, "blink.data.loader", fake)
    argv = ["train", "--config", str(config), "--run", "shards", "--data", str(shards), "--device", "cpu"]
    assert cli.main(argv) == 0
    saved = json.loads((home / "runs" / "shards" / "config.json").read_text(encoding="utf-8"))
    assert saved["data"] == {"source": "shards", "dir": str(shards), "train_shards": 1, "val_records": 40}


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
