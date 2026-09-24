"""`blink ops launch` and `blink ops ps`: fully detached jobs via Win32_Process.Create (PF38)."""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from blink import cli, heartbeat
from blink.ops import launch

HOME = Path("D:/blink-test-home")


def _plan(args=("supervise", "--run", "long", "--", "train", "--config", "configs/long.toml"), **kw):
    return launch.plan_launch(
        "long", list(args), home=HOME, python=r"C:\venv\Scripts\python.exe", cwd="C:\\repo", **kw
    )


def test_launch_command_runs_detached_through_win32_process_create():
    plan = _plan()
    script = plan.script
    assert "Invoke-CimMethod -ClassName Win32_Process -MethodName Create" in script
    assert "ShowWindow" in script and "CurrentDirectory = 'C:\\repo'" in script
    line = plan.command_line
    assert line.startswith('cmd.exe /d /s /c "') and line.endswith('"')
    assert 'set "PYTHONUTF8=1"&& ' in line
    assert 'set "UV_CACHE_DIR=D:\\uv-cache"&& ' in line
    assert f'set "BLINK_HOME={HOME}"&& ' in line
    assert "C:\\venv\\Scripts\\python.exe -m blink.cli supervise --run long -- train --config" in line
    assert f'1>>"{HOME / "logs" / "long.out"}" 2>>"{HOME / "logs" / "long.err"}"' in line
    assert plan.heartbeat == HOME / "runs" / "long" / "heartbeat.json"


def test_the_environment_is_set_for_the_launched_process_only(monkeypatch):
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    before = dict(os.environ)
    _plan()
    assert dict(os.environ) == before


def test_the_encoded_command_decodes_to_the_script():
    plan = _plan()
    decoded = base64.b64decode(launch.encode_command(plan.script)).decode("utf-16-le")
    assert decoded == plan.script
    argv = launch.powershell_argv(plan.script)
    assert argv[0].lower().endswith("powershell.exe") and "-EncodedCommand" in argv and "-NoProfile" in argv


@pytest.mark.parametrize("bad", ["a&b", "50%", 'say "hi"', "x|y", "a>b", "a<b", "up^", "two\nlines"])
def test_launch_refuses_arguments_cmd_would_reinterpret(bad):
    with pytest.raises(ValueError, match="cmd.exe"):
        _plan(args=("train", "--config", bad))


def test_launch_quotes_arguments_with_spaces_and_single_quotes_survive_powershell():
    plan = _plan(args=("train", "--config", r"D:\my configs\it's.toml"))
    assert '"D:\\my configs\\it\'s.toml"' in plan.command_line
    assert "it''s.toml" in plan.script  # doubled inside the PowerShell single-quoted string


def test_launch_names_must_be_run_names():
    with pytest.raises(ValueError, match="name"):
        launch.plan_launch("..\\evil", ["train"], home=HOME)


def test_launch_parses_the_pid_and_names_wmi_failures(tmp_path):
    plan = launch.plan_launch("probe", ["heartbeat-probe", "--out", str(tmp_path / "hb.json")], home=tmp_path)

    def ok(argv, **_):
        return subprocess.CompletedProcess(argv, 0, stdout="0 4242\n", stderr="")

    result = launch.launch(plan, runner=ok, find_children=lambda pid: [4243])
    assert result.pid == 4242 and result.python_pids == (4243,)
    record = json.loads((tmp_path / "logs" / "probe.launch.json").read_text(encoding="utf-8"))
    assert record["pid"] == 4242 and record["heartbeat"] == str(tmp_path / "hb.json")

    def denied(argv, **_):
        return subprocess.CompletedProcess(argv, 0, stdout="2 0\n", stderr="")

    with pytest.raises(launch.LaunchError, match="access denied"):
        launch.launch(plan, runner=denied, find_children=lambda pid: [])


def test_ps_rows_show_each_blink_process_with_its_heartbeat(tmp_path):
    beat = tmp_path / "runs" / "long" / "heartbeat.json"
    beat.parent.mkdir(parents=True)
    heartbeat.write(beat, {"state": "running", "step": 5000}, now=1000.0)
    processes = [
        {
            "pid": 7,
            "create_time": 400.0,
            "cmdline": ["python.exe", "-m", "blink.cli", "train", "--run", "long"],
        },
        {"pid": 8, "create_time": 900.0, "cmdline": ["notepad.exe", "blink.txt"]},
        {
            "pid": 9,
            "create_time": 950.0,
            "cmdline": [r"C:\v\Scripts\blink.exe", "heartbeat-probe", "--out", "x"],
        },
    ]
    rows = launch.ps_rows(processes, home=tmp_path, now=1012.0)
    assert [row["pid"] for row in rows] == [7, 9]
    assert rows[0]["run"] == "long" and rows[0]["heartbeat"] == "12 s ago, running, step 5000"
    assert rows[1]["heartbeat"] == "no heartbeat file"
    text = launch.format_ps(rows)
    assert "long" in text and "12 s ago" in text


def test_ps_finds_a_real_blink_process(tmp_path):
    out = tmp_path / "hb.json"
    argv = [sys.executable, "-m", "blink.cli", "heartbeat-probe", "--out", str(out), "--minutes", "0.05"]
    child = subprocess.Popen(argv, cwd=Path(launch.__file__).resolve().parents[2])
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not out.exists():
            time.sleep(0.1)
        pids = {row["pid"] for row in launch.blink_processes()}
        assert child.pid in pids or any(p.pid in pids for p in _children(child.pid))
    finally:
        child.kill()
        child.wait()


def _children(pid: int):
    import psutil

    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []


@pytest.mark.local
@pytest.mark.skipif(sys.platform != "win32", reason="Win32_Process.Create exists only on Windows")
def test_a_detached_launch_really_beats_outside_this_process(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    out = tmp_path / "probe.json"
    args = ["heartbeat-probe", "--out", str(out), "--minutes", "0.0834", "--interval", "1"]
    result = launch.launch(launch.plan_launch("probe-test", args))
    assert result.pid > 0
    beats, deadline = set(), time.time() + 90
    while time.time() < deadline:
        record = heartbeat.read(out)
        if record:
            beats.add(record["beat"])
            if record.get("done"):
                break
        time.sleep(0.3)
    assert len(beats) >= 3, (beats, (tmp_path / "logs" / "probe-test.err").read_text(encoding="utf-8"))
    assert heartbeat.read(out).get("done") is True
    assert heartbeat.read(out)["pid"] != os.getpid()


def test_ops_launch_dry_run_keeps_the_nested_double_dash_for_supervise(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    argv = ["ops", "launch", "--name", "long", "--dry-run", "--", "supervise", "--run", "long", "--", "train"]
    assert cli.main(argv) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("cmd.exe /d /s /c") and "-m blink.cli supervise --run long -- train 1>>" in line


def test_ops_launch_refuses_an_unsafe_argument_with_one_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert cli.main(["ops", "launch", "--name", "x", "--", "train", "--config", "a&b"]) == 2
    assert "cmd.exe" in capsys.readouterr().err


def test_ops_ps_prints_a_table_or_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert cli.main(["ops", "ps"]) == 0
    assert cli.main(["ops", "ps", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"processes"' in out
