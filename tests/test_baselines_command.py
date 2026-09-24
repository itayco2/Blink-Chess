"""`blink baselines train` from the command line."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from blink import cli  # noqa: E402
from blink.board import encode  # noqa: E402
from blink.data.record import ROOT_DTYPE  # noqa: E402

pytestmark = pytest.mark.torch


def _write_records(path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    codes = np.zeros((n, 64), dtype=np.uint8)
    codes[:, 4], codes[:, 60] = encode.OWN + 5, encode.OPP + 5
    codes[:, 12] = rng.choice([encode.EMPTY, encode.OWN + 4], size=n)
    records = np.zeros(n, dtype=ROOT_DTYPE)
    records["board"] = encode.pack(codes)
    records["cp"] = np.where(codes[:, 12] == encode.OWN + 4, 900, 0)
    records.tofile(path)


def test_baselines_train_writes_the_run_under_blink_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    data = tmp_path / "data"
    data.mkdir()
    _write_records(data / "train_000.bin", 300, seed=0)
    _write_records(data / "val.bin", 60, seed=1)
    code = cli.main(
        [
            "baselines",
            "train",
            "--kind",
            "linear",
            "--positions",
            "256",
            "--data",
            str(data),
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--batch-size",
            "64",
        ]
    )
    assert code == 0
    run = tmp_path / "home" / "runs" / "baseline-linear"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["train_set"]["positions"] == 256 and (run / "model.pt").is_file()
    assert "val win% MAE" in capsys.readouterr().out


def test_baselines_train_refuses_a_missing_data_dir_with_one_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    code = cli.main(
        ["baselines", "train", "--kind", "mlp", "--data", str(tmp_path / "nope"), "--device", "cpu"]
    )
    assert code == 2
    assert "no train root shards" in capsys.readouterr().err
