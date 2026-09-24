import chess
import numpy as np
import orjson
import pytest

torch = pytest.importorskip("torch")

from train_helpers import FIXTURE, fixture_records, held_open, tiny_model_config  # noqa: E402

from blink.board import encode, moves  # noqa: E402
from blink.board.encode import unpack  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train import telemetry  # noqa: E402

pytestmark = pytest.mark.torch


def _fixture_boards() -> list[chess.Board]:
    return [chess.Board(orjson.loads(line)["fen"] + " 0 1") for line in FIXTURE.read_bytes().splitlines()]


def test_the_legal_mask_rebuilt_from_square_codes_matches_python_chess():
    for board in _fixture_boards():
        rebuilt = telemetry.legal_mask_from_codes(encode.encode_board(board))
        assert np.array_equal(rebuilt, moves.legal_mask(board)), board.fen()


def test_the_best_move_of_every_fixture_record_is_legal_in_the_rebuilt_board():
    records = fixture_records()
    masks = np.stack([telemetry.legal_mask_from_codes(codes) for codes in unpack(records["board"])])
    assert masks[np.arange(len(records)), records["move"]].all()


def test_a_fresh_network_scores_chance_level_on_the_val_set():
    records = fixture_records()
    val = telemetry.make_val_set(records, "cpu")
    result = telemetry.evaluate(BlinkNet(tiny_model_config()), val, alpha=0.5, tau=0.05)
    assert result["n"] == len(records)
    assert result["policy_ce"] == pytest.approx(7.54, abs=0.1)
    assert result["value_ce"] == pytest.approx(4.85, abs=0.05)
    assert 0.0 <= result["top1"] <= 0.2
    assert result["win_mae"] == pytest.approx(np.abs(0.5 - val.batch.w_best.numpy()).mean(), abs=1e-5)


def test_truncate_after_keeps_lines_up_to_the_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for step in (1, 50, 100, 150):
        telemetry.append_jsonl(path, {"step": step})
    with open(path, "ab") as handle:
        handle.write(b'{"step": 200')  # a torn last line from a crash
    telemetry.truncate_after(path, 100)
    assert path.read_text(encoding="utf-8") == '{"step": 1}\n{"step": 50}\n{"step": 100}\n'


def test_truncating_a_missing_file_is_a_no_op(tmp_path):
    telemetry.truncate_after(tmp_path / "missing.jsonl", 10)
    assert not (tmp_path / "missing.jsonl").exists()


def test_truncating_succeeds_while_the_dashboard_briefly_holds_the_log_open(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for step in (50, 100, 150):
        telemetry.append_jsonl(path, {"step": step})
    with held_open(path):
        telemetry.truncate_after(path, 100)
    assert path.read_text(encoding="utf-8") == '{"step": 50}\n{"step": 100}\n'


class _Clock:
    """A stand-in for the time module: perf_counter and time move only when the test says so."""

    def __init__(self, now: float) -> None:
        self.now = now

    def perf_counter(self) -> float:
        return self.now

    def time(self) -> float:
        return 1_700_000_000.0 + self.now


def _step(window: telemetry.MetricWindow) -> None:
    one = torch.tensor(1.0)
    window.add(one, one, torch.tensor(0.5), clip_norm=1.0, samples=16)


def test_a_metrics_row_reports_the_share_of_its_window_spent_waiting_for_batches(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(telemetry, "time", clock)
    window = telemetry.MetricWindow(torch.device("cpu"))
    clock.now = 101.0
    window.waited(0.25)
    _step(window)
    window.waited(0.75)
    _step(window)
    clock.now = 104.0
    row = window.flush(step=2, lr=1e-3)
    assert row["data_wait_frac"] == pytest.approx(1.0 / 4.0)
    assert row["time"] == pytest.approx(1_700_000_104.0)
    clock.now = 106.0
    _step(window)
    assert window.flush(step=3, lr=1e-3)["data_wait_frac"] == 0.0  # each window counts only its own wait


def test_the_wait_share_never_exceeds_one(monkeypatch):
    clock = _Clock(5.0)
    monkeypatch.setattr(telemetry, "time", clock)
    window = telemetry.MetricWindow(torch.device("cpu"))
    window.waited(3.0)
    clock.now = 6.0
    assert window.flush(step=1, lr=0.0)["data_wait_frac"] == 1.0
