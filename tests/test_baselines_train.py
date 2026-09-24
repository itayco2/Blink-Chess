"""Training the linear and MLP rungs: soft BCE on win%, early stop on val, at most 5 epochs."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from blink.baselines import features, models  # noqa: E402
from blink.baselines import train as baseline_train  # noqa: E402
from blink.board import encode, value  # noqa: E402
from blink.data.record import ROOT_DTYPE  # noqa: E402
from blink.play.oracles import material_balance  # noqa: E402

pytestmark = pytest.mark.torch


def _random_positions(n: int, seed: int) -> np.ndarray:
    """Codes of toy boards: 2 to 8 pieces of either colour on random squares, mostly pawns."""
    rng = np.random.default_rng(seed)
    codes = np.zeros((n, 64), dtype=np.uint8)
    for row in codes:
        count = rng.integers(2, 9)
        squares = rng.choice(64, size=count, replace=False)
        kinds = rng.choice(5, size=count, p=[0.5, 0.15, 0.15, 0.12, 0.08])
        row[squares] = np.where(rng.random(count) < 0.5, encode.OWN, encode.OPP) + kinds
    return codes


def _material_labels(codes: np.ndarray) -> np.ndarray:
    return np.array([value.win_probability(cp=100 * int(b)) for b in material_balance(codes)], np.float32)


def _records(codes: np.ndarray, cp: np.ndarray) -> np.ndarray:
    records = np.zeros(len(codes), dtype=ROOT_DTYPE)
    records["board"] = encode.pack(codes)
    records["cp"] = cp
    records["fen_hash"] = np.arange(len(codes), dtype=np.uint64) * 7919
    return records


def test_a_linear_model_learns_a_toy_material_rule():
    codes = _random_positions(12000, seed=1)
    win = _material_labels(codes)
    boards = encode.pack(codes)
    cfg = baseline_train.BaselineConfig(kind="linear", epochs=5, batch_size=256, lr=0.1, device="cpu")
    result = baseline_train.fit(
        cfg, boards[:10000], win[:10000], boards[10000:], win[10000:], log=lambda _: None
    )
    assert result.val_mae < 0.1 * result.initial["val_mae"]
    assert result.val_mae < 0.03
    weights = result.model.linear.weight.detach().numpy().reshape(features.NUM_PLANES, 64).mean(axis=1)
    own = [
        weights[p] for p in (features.OWN_PAWN, features.OWN_KNIGHT, features.OWN_ROOK, features.OWN_QUEEN)
    ]
    opp = [
        weights[p] for p in (features.OPP_PAWN, features.OPP_KNIGHT, features.OPP_ROOK, features.OPP_QUEEN)
    ]
    assert own == sorted(own) and own[0] > 0, own
    assert opp == sorted(opp, reverse=True) and opp[0] < 0, opp


def test_early_stopping_keeps_the_best_epoch_and_never_runs_more_than_five():
    codes = _random_positions(300, seed=2)
    win = _material_labels(codes)
    boards = encode.pack(codes)
    cfg = baseline_train.BaselineConfig(kind="mlp", epochs=5, batch_size=32, lr=0.01, device="cpu")
    result = baseline_train.fit(cfg, boards[:200], win[:200], boards[200:], win[200:], log=lambda _: None)
    epochs = [row["epoch"] for row in result.history]
    assert len(epochs) <= baseline_train.MAX_EPOCHS
    best = min(result.history, key=lambda row: row["val_bce"])
    assert result.best_epoch == best["epoch"]
    assert result.val_mae == pytest.approx(best["val_mae"])
    with pytest.raises(ValueError, match="at most 5"):
        baseline_train.BaselineConfig(kind="mlp", epochs=6)
    with pytest.raises(ValueError, match="linear"):
        baseline_train.BaselineConfig(kind="cnn")


def test_the_fixed_train_set_is_the_first_n_records_of_the_shards_in_name_order(tmp_path):
    codes = _random_positions(30, seed=3)
    records = _records(codes, np.arange(30, dtype=np.int16))
    records[20:].tofile(tmp_path / "train_r001.bin")
    records[:20].tofile(tmp_path / "train_r000.bin")
    records[:5].tofile(tmp_path / "train_c000.bin")  # children are never part of the set
    taken, description = baseline_train.fixed_train_set(tmp_path, 25)
    assert taken["cp"].tolist() == list(range(25))
    assert description["shards"] == ["train_r000.bin", "train_r001.bin"]
    assert description["positions"] == 25
    again, again_description = baseline_train.fixed_train_set(tmp_path, 25)
    assert again_description["fen_hash_sha256"] == description["fen_hash_sha256"]


def test_the_skeleton_layout_also_works_and_too_few_records_is_an_explicit_error(tmp_path):
    codes = _random_positions(10, seed=4)
    _records(codes, np.zeros(10, dtype=np.int16)).tofile(tmp_path / "train_000.bin")
    taken, _ = baseline_train.fixed_train_set(tmp_path, 10)
    assert len(taken) == 10
    with pytest.raises(ValueError, match="only 10"):
        baseline_train.fixed_train_set(tmp_path, 11)


def test_train_baseline_writes_model_and_metrics_under_the_run_dir(tmp_path):
    codes = _random_positions(160, seed=5)
    balance = material_balance(codes)
    records = _records(codes, (100 * balance).astype(np.int16))
    data = tmp_path / "data"
    data.mkdir()
    records[:120].tofile(data / "train_000.bin")
    records[120:].tofile(data / "val.bin")
    out = tmp_path / "run"
    cfg = baseline_train.BaselineConfig(kind="linear", positions=100, epochs=2, batch_size=32, device="cpu")
    metrics = baseline_train.train_baseline(cfg, data, out, log=lambda _: None)
    assert (out / "model.pt").is_file()
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "linear" and saved["train_set"]["positions"] == 100
    assert saved["val"]["positions"] == 40
    assert saved == metrics
    model, kind, meta = models.load(out / "model.pt")
    assert kind == "linear" and meta["val_mae"] == pytest.approx(saved["val_mae"])
