import json

from blink import heartbeat


def test_write_heartbeat_writes_json_with_a_timestamp(tmp_path):
    target = tmp_path / "heartbeat.json"
    heartbeat.write(target, {"phase": "P0", "step": 3}, now=1_700_000_000.0)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == {"phase": "P0", "step": 3, "time": 1_700_000_000.0}


def test_write_heartbeat_replaces_the_file_and_leaves_no_temp_behind(tmp_path):
    target = tmp_path / "heartbeat.json"
    heartbeat.write(target, {"step": 1}, now=1.0)
    heartbeat.write(target, {"step": 2}, now=2.0)
    assert json.loads(target.read_text(encoding="utf-8"))["step"] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["heartbeat.json"]


def test_write_heartbeat_does_not_mutate_the_payload(tmp_path):
    payload = {"step": 1}
    heartbeat.write(tmp_path / "hb.json", payload, now=5.0)
    assert payload == {"step": 1}


def test_age_is_measured_from_the_file_timestamp(tmp_path):
    target = tmp_path / "hb.json"
    heartbeat.write(target, {}, now=100.0)
    assert heartbeat.age_seconds(target, now=130.0) == 30.0


def test_a_missing_heartbeat_has_no_age(tmp_path):
    assert heartbeat.age_seconds(tmp_path / "missing.json", now=1.0) is None


def test_write_retries_when_windows_briefly_locks_the_target(tmp_path, monkeypatch):
    """A watcher holding the file without delete-sharing makes os.replace raise PermissionError."""
    real_replace = heartbeat.os.replace
    failures = {"left": 2}

    def flaky_replace(src, dst):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError(32, "The process cannot access the file")
        real_replace(src, dst)

    monkeypatch.setattr(heartbeat.os, "replace", flaky_replace)
    heartbeat.write(tmp_path / "hb.json", {"step": 7}, now=1.0, retry_sleep=0.0)
    assert heartbeat.read(tmp_path / "hb.json")["step"] == 7


def test_write_gives_up_after_its_retries_and_cleans_its_temp_file(tmp_path, monkeypatch):
    def always_locked(src, dst):
        raise PermissionError(32, "locked")

    monkeypatch.setattr(heartbeat.os, "replace", always_locked)
    try:
        heartbeat.write(tmp_path / "hb.json", {}, now=1.0, retry_sleep=0.0)
    except PermissionError:
        assert list(tmp_path.iterdir()) == []
        return
    raise AssertionError("expected PermissionError after the retries")


def test_a_missed_beat_is_reported_not_fatal(tmp_path, monkeypatch):
    def always_locked(src, dst):
        raise PermissionError(32, "locked")

    monkeypatch.setattr(heartbeat.os, "replace", always_locked)
    assert heartbeat.beat_once(tmp_path / "hb.json", {"beat": 1}, retry_sleep=0.0) is False


def test_reading_a_locked_heartbeat_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "hb.json"
    heartbeat.write(target, {}, now=1.0)

    def locked(self, *args, **kwargs):
        raise PermissionError(32, "locked")

    monkeypatch.setattr(type(target), "read_text", locked)
    assert heartbeat.read(target) is None
