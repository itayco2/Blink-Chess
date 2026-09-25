"""tools/p7_machine.py: PowerShell output that can always be read, and the one-instance lock.

A detached driver reads process command lines through PowerShell: a byte the console code page cannot
decode once turned the whole stream into None (PF39), so the output is sent as UTF-8, decoded with
backslashreplace, and no output at all is no lines. Two drivers at once would race on long.toml and the
runs, so logs/p7v2.lock admits one; a lock left by a dead process, or by a pid Windows gave to another
process since, is replaced.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_p7_v2_driver import driver

# isort: split
import p7_machine  # tools/: on sys.path once test_p7_v2_driver is imported


class Recorded:
    def __init__(self, stdout):
        self.stdout, self.calls = stdout, []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=self.stdout, stderr="")


def _host(tmp_path) -> p7_machine.Host:
    return p7_machine.Host(tmp_path, Path(sys.executable), tmp_path, tmp_path)


def test_command_lines_are_decoded_as_utf8_with_backslashreplace(tmp_path, monkeypatch):
    fake = Recorded("python.exe -m blink.cli train --run long\n\n  \nC:\\caf\\xe9 python.exe x\n")
    monkeypatch.setattr(p7_machine.subprocess, "run", fake)
    lines = _host(tmp_path).command_lines()
    assert lines == ["python.exe -m blink.cli train --run long", "C:\\caf\\xe9 python.exe x"]
    argv, kwargs = fake.calls[0]
    assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "backslashreplace" and kwargs["text"]
    assert argv[-1].startswith("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8")


def test_no_output_at_all_is_no_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(p7_machine.subprocess, "run", Recorded(None))
    assert _host(tmp_path).command_lines() == [] and p7_machine.decode_lines(None) == []


def test_the_endgame_screen_stop_reads_its_output_the_same_way(tmp_path, monkeypatch):
    fake = Recorded(None)
    monkeypatch.setattr(p7_machine.subprocess, "run", fake)
    _host(tmp_path).stop_endgame_screen()
    assert fake.calls[0][1]["errors"] == "backslashreplace"


# ---------------------------------------------------------------- the lock


def _identity(table):
    return lambda pid: table.get(pid)


def test_the_lock_is_created_with_its_pid_and_create_time_and_refuses_a_second_instance(tmp_path):
    path = tmp_path / "p7v2.lock"
    alive = _identity({101: 5.0, 202: 7.0})
    p7_machine.acquire_lock(path, pid=101, identity=alive)
    held = json.loads(path.read_text(encoding="utf-8"))
    assert (held["pid"], held["create_time"]) == (101, 5.0)
    with pytest.raises(p7_machine.LockHeld, match="pid 101"):
        p7_machine.acquire_lock(path, pid=202, identity=alive)
    p7_machine.release_lock(path, pid=202)  # not its lock: left alone
    assert path.is_file()
    p7_machine.release_lock(path, pid=101)
    assert not path.exists()


def test_a_stale_lock_is_replaced_when_its_pid_is_dead_or_reused_by_another_process(tmp_path):
    path = tmp_path / "p7v2.lock"
    for gone in ({}, {101: 9.0}):  # 101 died; then 101 is a new process started at 9.0, not 5.0
        path.write_text(json.dumps({"pid": 101, "create_time": 5.0}), encoding="utf-8")
        os.utime(path, (1.0, 1.0))
        p7_machine.acquire_lock(path, pid=202, identity=_identity({**gone, 202: 7.0}))
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == 202
        path.unlink()


def test_an_unreadable_lock_is_its_owner_s_while_young_and_stale_once_old(tmp_path):
    path = tmp_path / "p7v2.lock"
    path.write_text("", encoding="utf-8")  # its owner has created it and not yet written it
    with pytest.raises(p7_machine.LockHeld):
        p7_machine.acquire_lock(path, pid=202, identity=_identity({202: 7.0}))
    os.utime(path, (1.0, 1.0))
    p7_machine.acquire_lock(path, pid=202, identity=_identity({202: 7.0}))


def test_the_real_identity_tells_this_process_from_a_dead_pid():
    assert p7_machine.process_identity(os.getpid()) is not None
    assert p7_machine.process_identity(2**31 - 7) is None


def test_the_driver_refuses_while_another_live_instance_holds_the_lock(tmp_path, capsys):
    home = tmp_path / "home"
    (home / "logs").mkdir(parents=True)
    p7_machine.acquire_lock(home / "logs" / "p7v2.lock")  # this very process: alive
    assert driver.main(["--home", str(home), "--repo", str(tmp_path)]) == driver.EXIT_REFUSED
    assert "holds" in capsys.readouterr().err and not (home / "logs" / "p7v2.status.json").exists()
