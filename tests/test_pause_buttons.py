"""The Pause, Resume and Status buttons (deploy/pause/), the `blink ops ps` flag line and the installer.

On Windows the buttons run for real in cmd.exe, against a temporary BLINK_HOME and a fake nvidia-smi
placed first on PATH. The installer is only ever pointed at temporary folders here: Itay's desktop is
written only when a person runs `blink ops install-pause-buttons`.
"""

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from blink import cli
from blink.ops import buttons

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="the buttons are cmd.exe scripts")
PAUSE, RESUME, STATUS = "Pause Blink.cmd", "Resume Blink.cmd", "Blink Status.cmd"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _press(button: str, home: Path, bin_dir: Path | None = None, wait_s: int = 5):
    """The button in cmd.exe; Blink Status's python is this interpreter, run from this checkout."""
    env = {**os.environ, "BLINK_HOME": str(home), "BLINK_PAUSE_WAIT_S": str(wait_s)}
    env["BLINK_PYTHON"] = sys.executable
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["cmd.exe", "/d", "/c", str(buttons.BUTTONS_DIR / button)],
        env=env,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _fake_nvidia_smi(tmp_path: Path, pids: list[int]) -> Path:
    """An nvidia-smi that lists these PIDs as GPU compute apps, whatever it is asked."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    lines = "".join(f"echo {pid}\r\n" for pid in pids)
    (bin_dir / "nvidia-smi.cmd").write_bytes(("@echo off\r\n" + lines).encode("ascii"))
    return bin_dir


def _sleeper(*argv: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *argv])


def _end(proc: subprocess.Popen) -> None:
    """Kill the sleeper and the interpreter the venv launcher started for it."""
    try:
        family = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        family = []
    for member in family:
        with contextlib.suppress(psutil.Error):
            member.kill()
    proc.kill()
    proc.wait()


def test_the_buttons_are_plain_ascii_and_say_what_the_plan_says():
    pause = (buttons.BUTTONS_DIR / PAUSE).read_text(encoding="ascii")
    resume = (buttons.BUTTONS_DIR / RESUME).read_text(encoding="ascii")
    assert (
        "nvidia-smi --query-compute-apps" in pause
        and "Blink is paused and the GPU is free. Have fun." in pause
    )
    assert 'set "BLINK_PAUSE_WAIT_S=300"' in pause  # up to 5 minutes
    assert "Blink will resume within a minute." in resume
    assert all('"%BLINK_HOME%\\PAUSE"' in text for text in (pause, resume))


def test_the_status_button_is_plain_ascii_and_only_reads(tmp_path):
    status = (buttons.BUTTONS_DIR / STATUS).read_text(encoding="ascii")
    assert r'set "BLINK_HOME=D:\blink"' in status
    assert r'set "BLINK_PYTHON=C:\dev\blink-chess\.venv\Scripts\python.exe"' in status
    assert '"%BLINK_PYTHON%" -m blink.cli status --live' in status
    assert "--query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader" in status
    assert '"%BLINK_PYTHON%" -m blink.cli eval strength --show --last 1' in status
    assert "del " not in status and "> " not in status.replace("2>nul", "").replace(">nul", "")
    assert status.rstrip().splitlines()[-2:] == ["pause", "exit /b 0"]


def test_the_installer_copies_all_three_buttons_with_crlf_line_ends(tmp_path, capsys):
    desk = tmp_path / "Desktop"
    desk.mkdir()
    assert buttons.BUTTONS == (PAUSE, RESUME, STATUS)
    assert cli.main(["ops", "install-pause-buttons", "--to", str(desk)]) == 0
    for name in (PAUSE, RESUME, STATUS):
        copied = (desk / name).read_bytes()
        source = (buttons.BUTTONS_DIR / name).read_bytes().replace(b"\r\n", b"\n")
        assert copied == source.replace(b"\n", b"\r\n") and b"\r\n" in copied
    assert str(desk) in capsys.readouterr().out


def test_the_installer_defaults_to_the_home_desktop_and_never_creates_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert buttons.default_target() == tmp_path / "Desktop"
    assert cli.main(["ops", "install-pause-buttons", "--dry-run"]) == 0
    assert str(tmp_path / "Desktop") in capsys.readouterr().out and not (tmp_path / "Desktop").exists()
    assert cli.main(["ops", "install-pause-buttons"]) == 2  # no Desktop folder: refused, not created
    assert "Desktop" in capsys.readouterr().err and not (tmp_path / "Desktop").exists()


def test_ops_ps_says_when_the_pause_flag_is_up(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert cli.main(["ops", "ps"]) == 0
    assert "user pause" not in capsys.readouterr().out
    (tmp_path / "PAUSE").touch()
    assert cli.main(["ops", "ps"]) == 0
    assert "user pause: " in capsys.readouterr().out
    assert cli.main(["ops", "ps", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["user_pause"].startswith("user pause: ")


@on_windows
def test_resume_removes_the_flag_and_says_blink_will_resume(tmp_path):
    (tmp_path / "PAUSE").write_text("paused", encoding="utf-8")
    done = _press(RESUME, tmp_path)
    assert "Blink will resume within a minute." in done.stdout and not (tmp_path / "PAUSE").exists()
    assert "not paused" in _press(RESUME, tmp_path).stdout


@on_windows
def test_pause_creates_the_flag_and_reports_a_free_gpu_when_no_blink_process_holds_it(tmp_path):
    other = _sleeper()  # a GPU user that is not Blink (a game, say)
    try:
        done = _press(PAUSE, tmp_path, _fake_nvidia_smi(tmp_path, [other.pid]))
    finally:
        _end(other)
    assert (tmp_path / "PAUSE").is_file()
    assert "Blink is paused and the GPU is free. Have fun." in done.stdout, done.stdout + done.stderr
    assert done.returncode == 0


@on_windows
def test_pause_names_a_blink_process_still_on_the_gpu_when_the_wait_runs_out(tmp_path):
    trainer = _sleeper("-m", "blink.cli", "train", "--run", "abl-fake")
    try:
        done = _press(PAUSE, tmp_path, _fake_nvidia_smi(tmp_path, [trainer.pid]), wait_s=0)
    finally:
        _end(trainer)
    assert f"pid {trainer.pid}" in done.stdout and "abl-fake" in done.stdout, done.stdout + done.stderr
    assert "GPU is free" not in done.stdout and done.returncode == 1
    assert (tmp_path / "PAUSE").is_file()  # still paused: the trainer stops at its next step


def _fake_gpu_line(tmp_path: Path) -> Path:
    """An nvidia-smi that prints one --query-gpu line: 37% use, 61 C, 180.50 W."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi.cmd").write_bytes(b"@echo off\r\necho 37 %%, 61, 180.50 W\r\n")
    return bin_dir


@on_windows
def test_status_shows_the_live_run_the_gpu_the_pause_flag_and_the_last_strength_check(tmp_path):
    from blink import heartbeat

    run = tmp_path / "runs" / "long"
    run.mkdir(parents=True)
    heartbeat.write(run / "heartbeat.json", {"state": "running", "step": 120_000, "steps": 1_105_860})
    (tmp_path / "PAUSE").write_text("paused", encoding="utf-8")
    check = {"kind": "strength", "run": "long", "step": 100_000, "hours": 9.9, "time": 0.0, "puzzles": 2000}
    scores = {"accuracy": 0.8, "wilson95": [0.78, 0.82]}
    check |= {"value": scores, "policy": scores, "dm9m": {"accuracy": 0.866, "wilson95": [0.85, 0.88]}}
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "strength_checks.jsonl").write_text(json.dumps(check) + "\n", encoding="utf-8")
    done = _press(STATUS, tmp_path, _fake_gpu_line(tmp_path))
    out = done.stdout
    assert "long: LIVE, step 120000/1105860" in out, out + done.stderr
    assert "37 %, 61, 180.50 W" in out
    assert "The pause flag is up" in out
    assert "100,000" in out and "DM-9M" in out and done.returncode == 0


@on_windows
def test_status_says_the_flag_is_down_and_no_run_is_live_on_an_idle_home(tmp_path):
    done = _press(STATUS, tmp_path, _fake_gpu_line(tmp_path))
    out = done.stdout
    assert "no live run" in out and "The pause flag is down" in out, out + done.stderr
    assert "no strength checks yet" in out and done.returncode == 0
