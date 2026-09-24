import pytest
from train_helpers import held_open

from blink.train import atomic


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(atomic, "RETRY_SLEEP_S", 0.0)


def _flaky_replace(monkeypatch, failures: int, error: OSError) -> list[int]:
    real_replace = atomic.os.replace
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) <= failures:
            raise error
        real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    return calls


def test_writing_text_atomically_replaces_the_file_and_leaves_no_temp_behind(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("old\n", encoding="utf-8")
    atomic.write_text_atomic(path, "new\n")
    assert path.read_text(encoding="utf-8") == "new\n"
    assert [p.name for p in tmp_path.iterdir()] == ["config.json"]


def test_a_target_locked_by_a_reader_is_retried_until_the_reader_lets_go(tmp_path, monkeypatch, no_sleep):
    calls = _flaky_replace(monkeypatch, failures=3, error=PermissionError(13, "held by the dashboard"))
    atomic.write_text_atomic(tmp_path / "metrics.jsonl", '{"step": 1}\n')
    assert len(calls) == 4
    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == '{"step": 1}\n'


def test_a_target_locked_for_good_raises_after_the_last_retry(tmp_path, monkeypatch, no_sleep):
    calls = _flaky_replace(monkeypatch, failures=10**6, error=PermissionError(13, "locked for good"))
    with pytest.raises(PermissionError):
        atomic.write_text_atomic(tmp_path / "evals.jsonl", "x\n")
    assert len(calls) == atomic.REPLACE_RETRIES


def test_errors_other_than_a_lock_are_raised_at_once(tmp_path, monkeypatch, no_sleep):
    calls = _flaky_replace(monkeypatch, failures=10**6, error=OSError(28, "no space left on device"))
    with pytest.raises(OSError, match="no space"):
        atomic.write_text_atomic(tmp_path / "config.json", "x\n")
    assert len(calls) == 1


def test_a_file_a_reader_holds_open_is_replaced_once_the_reader_closes_it(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n', encoding="utf-8")
    with held_open(path):
        atomic.write_text_atomic(path, '{"step": 1}\n')
    assert path.read_text(encoding="utf-8") == '{"step": 1}\n'
