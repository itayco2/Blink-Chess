"""`blink lichess pause` and `resume-note`: the agent's only action on a running bot is stopping its PID."""

import ast
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from blink import cli
from blink.lichess import pause, snapshot

BOT_ROOT = "D:/blink-bot"
LAUNCHER = pause.ProcessRow(
    pid=100,
    ppid=10,
    create_time=1.0,
    cmdline=(
        "D:\\blink-bot\\venv\\Scripts\\python.exe",
        "lichess-bot.py",
        "--config",
        "D:\\blink-bot\\config.yml",
    ),
    exe="D:\\blink-bot\\venv\\Scripts\\python.exe",
    cwd="D:\\blink-bot\\lichess-bot",
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def a_bot(pid: int = 4242) -> pause.BotProcess:
    return pause.BotProcess(pid=pid, ppid=1, create_time=1.0, cmdline=("python.exe", "lichess-bot.py"))


def recording_deps(tmp_path, live: list, events: list, clock: Clock, found=None) -> pause.PauseDeps:
    answers = iter(live)

    def is_playing() -> bool:
        assert (tmp_path / "PAUSED").exists(), "the flag must be written before the first poll"
        answer = next(answers)
        events.append(("poll", answer))
        if isinstance(answer, Exception):
            raise answer
        return answer

    def find_bot():
        events.append(("find",))
        return (a_bot(),) if found is None else found

    def stop_tree(proc):
        events.append(("stop", proc.pid))
        return (proc.pid, proc.pid + 1)

    def sleep(seconds: float) -> None:
        events.append(("sleep", seconds))
        clock.sleep(seconds)

    return pause.PauseDeps(
        is_playing=is_playing,
        find_bot=find_bot,
        stop_tree=stop_tree,
        sleep=sleep,
        clock=clock.time,
        now=lambda: "2026-10-08T10:00:00+00:00",
        log=lambda line: None,
    )


def test_lichess_pause_waits_for_no_live_game_before_stopping_the_bot_pid(tmp_path):
    events: list = []
    deps = recording_deps(tmp_path, [True, True, False], events, Clock())
    record = pause.pause("BlinkBot", deps, tmp_path, timeout_s=600, poll_s=15, reason="GPU window")
    assert events == [
        ("poll", True),
        ("sleep", 15),
        ("poll", True),
        ("sleep", 15),
        ("poll", False),
        ("find",),
        ("stop", 4242),
    ]
    assert (record.polls, record.waited_s, record.live_game_at_stop, record.timed_out) == (
        3,
        30.0,
        False,
        False,
    )
    saved = json.loads((tmp_path / "pause.json").read_text(encoding="utf-8"))
    assert saved["processes"][0]["pid"] == 4242 and saved["processes"][0]["stopped"] == [4242, 4243]
    assert saved["reason"] == "GPU window" and saved["bot"] == "BlinkBot"
    assert (tmp_path / "PAUSED").exists()


def test_the_pause_stops_the_bot_anyway_when_the_game_outlasts_the_timeout(tmp_path):
    events: list = []
    deps = recording_deps(tmp_path, [True] * 10, events, Clock())
    record = pause.pause("BlinkBot", deps, tmp_path, timeout_s=60, poll_s=15, reason="stop rule")
    assert [e for e in events if e[0] == "poll"] == [("poll", True)] * 5  # t = 0, 15, 30, 45, 60
    assert events[-1] == ("stop", 4242)
    assert record.timed_out and record.live_game_at_stop


def test_an_api_error_while_waiting_counts_as_a_live_game(tmp_path):
    events: list = []
    live = [snapshot.ApiError("HTTP 503"), False]
    record = pause.pause("BlinkBot", recording_deps(tmp_path, live, events, Clock()), tmp_path, 600, 15, "x")
    assert events[:3] == [("poll", live[0]), ("sleep", 15), ("poll", False)]
    assert record.polls == 2 and not record.live_game_at_stop


def test_a_pause_with_no_bot_process_running_still_leaves_the_flag(tmp_path):
    events: list = []
    record = pause.pause(
        "BlinkBot", recording_deps(tmp_path, [False], events, Clock(), found=()), tmp_path, 60, 15, "x"
    )
    assert record.processes == () and ("stop", 4242) not in events
    assert (tmp_path / "PAUSED").exists()


def test_only_the_root_lichess_bot_process_under_the_bot_folder_is_found():
    base_python = pause.ProcessRow(
        pid=101,
        ppid=100,
        create_time=1.1,
        cmdline=("C:\\Python312\\python.exe", "lichess-bot.py", "--config", "D:\\blink-bot\\config.yml"),
        exe="C:\\Python312\\python.exe",
        cwd="D:\\blink-bot\\lichess-bot",
    )
    elsewhere = pause.ProcessRow(
        pid=200,
        ppid=10,
        create_time=2.0,
        cmdline=("python.exe", "C:\\other\\lichess-bot.py"),
        exe="C:\\Python312\\python.exe",
        cwd="C:\\other",
    )
    engine = pause.ProcessRow(
        pid=102,
        ppid=101,
        create_time=1.2,
        cmdline=("C:/dev/blink-chess/.venv/Scripts/blink-uci.exe", "--model=ship"),
        exe="C:/dev/blink-chess/.venv/Scripts/blink-uci.exe",
        cwd="D:\\blink-bot\\lichess-bot",
    )
    found = pause.find_bot_processes(BOT_ROOT, rows=[LAUNCHER, base_python, elsewhere, engine])
    assert [p.pid for p in found] == [100]


def test_an_editor_with_the_bot_script_open_is_never_taken_for_the_bot():
    editor = pause.ProcessRow(
        pid=300,
        ppid=10,
        create_time=3.0,
        cmdline=("C:\\Windows\\notepad.exe", "D:\\blink-bot\\lichess-bot\\lichess-bot.py"),
        exe="C:\\Windows\\notepad.exe",
        cwd="D:\\blink-bot\\lichess-bot",
    )
    assert pause.find_bot_processes(BOT_ROOT, rows=[editor, LAUNCHER]) == (
        pause.BotProcess(LAUNCHER.pid, LAUNCHER.ppid, LAUNCHER.create_time, LAUNCHER.cmdline),
    )


def test_the_bot_folder_match_ignores_case_and_slash_direction():
    assert pause.is_under("d:/BLINK-BOT/venv/Scripts/python.exe", "D:\\blink-bot")
    assert not pause.is_under("D:\\blink-bot-old\\python.exe", "D:\\blink-bot")
    assert not pause.is_under(None, "D:\\blink-bot")


def _spawn_tree():
    """A python child that starts a sleeping grandchild and prints its pid."""
    code = (
        "import subprocess, sys, time;"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']);"
        "print(g.pid, flush=True); time.sleep(120)"
    )
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    grandchild = int(child.stdout.readline())
    return child, grandchild


def _as_bot(pid: int, create_time: float) -> pause.BotProcess:
    return pause.BotProcess(pid=pid, ppid=0, create_time=create_time, cmdline=())


def _family(pid: int) -> set[int]:
    """The process and everything below it (on Windows a venv python.exe is a launcher with a child)."""
    root = psutil.Process(pid)
    return {pid, *(p.pid for p in root.children(recursive=True))}


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _kill_all(pids: set[int]) -> None:
    """Kill these processes and anything started below them since they were listed."""
    everyone = set(pids)
    for pid in pids:
        with contextlib.suppress(psutil.Error):
            everyone |= _family(pid)
    for pid in everyone:
        with contextlib.suppress(psutil.Error):
            psutil.Process(pid).kill()


def test_stop_tree_stops_the_bot_and_every_process_below_it():
    child, grandchild = _spawn_tree()
    family = _family(child.pid)
    try:
        stopped = set(
            pause.stop_tree(_as_bot(child.pid, psutil.Process(child.pid).create_time()), grace_s=10)
        )
        family |= stopped
        assert {child.pid, grandchild} <= stopped
        child.wait(timeout=10)
        deadline = time.monotonic() + 10
        while not all(_gone(pid) for pid in family) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert all(_gone(pid) for pid in family)
    finally:
        _kill_all(family)


def test_stop_tree_refuses_a_pid_that_now_belongs_to_another_process():
    child, grandchild = _spawn_tree()
    family = _family(child.pid)
    try:
        assert pause.stop_tree(_as_bot(child.pid, create_time=12345.0), grace_s=1) == ()
        assert child.poll() is None and psutil.pid_exists(grandchild)
    finally:
        _kill_all(family)


def test_resume_note_only_deletes_the_flag_and_says_itay_restarts_the_bot(tmp_path):
    flag = tmp_path / "PAUSED"
    flag.write_text("paused", encoding="utf-8")
    note = pause.resume_note(flag)
    assert not flag.exists()
    assert "Itay" in note and "start-bot.ps1" in note
    assert "no PAUSED flag" in pause.resume_note(flag)


def test_the_cli_pause_writes_the_flag_and_the_record_under_blink_home(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    stopped: list[int] = []

    def fake_deps(name, api, root):
        assert str(root).replace("\\", "/").lower() == "d:/blink-bot"
        return pause.PauseDeps(
            is_playing=lambda: False,
            find_bot=lambda: (a_bot(77),),
            stop_tree=lambda proc: stopped.append(proc.pid) or (proc.pid,),
            sleep=lambda s: None,
            log=print,
        )

    monkeypatch.setattr(pause, "default_deps", fake_deps)
    monkeypatch.setattr(snapshot, "default_api", lambda: object())
    assert cli.main(["lichess", "pause", "--bot", "BlinkBot", "--reason", "D15 GPU window"]) == 0
    assert stopped == [77]
    assert (tmp_path / "lichess" / "PAUSED").exists()
    assert (
        json.loads((tmp_path / "lichess" / "pause.json").read_text(encoding="utf-8"))["reason"]
        == "D15 GPU window"
    )
    assert "pid 77" in capsys.readouterr().out


def test_the_cli_resume_note_removes_the_flag(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    (tmp_path / "lichess").mkdir()
    (tmp_path / "lichess" / "PAUSED").write_text("paused", encoding="utf-8")
    assert cli.main(["lichess", "resume-note"]) == 0
    assert not (tmp_path / "lichess" / "PAUSED").exists()
    assert "Itay" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["", "a", "../x", "Blink Bot", "x" * 31])
def test_a_bad_bot_name_is_refused_before_anything_happens(monkeypatch, tmp_path, name):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert cli.main(["lichess", "pause", "--bot", name]) == 2
    assert not (tmp_path / "lichess" / "PAUSED").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["lichess", "pause", "--bot", "BlinkBot", "--poll", "0"],
        ["lichess", "check", "--bot", "BlinkBot", "--window", "0"],
        ["lichess", "snapshot", "--bot", "BlinkBot", "--max-games", "-1"],
    ],
)
def test_non_positive_poll_window_or_game_counts_are_refused_by_the_parser(argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2


def test_a_pause_that_cannot_write_its_flag_is_refused_cleanly(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "home"
    blocker.write_text("a file where the BLINK_HOME folder should be", encoding="utf-8")
    monkeypatch.setenv("BLINK_HOME", str(blocker))
    monkeypatch.setattr(snapshot, "default_api", lambda: object())
    never = pause.PauseDeps(
        is_playing=lambda: pytest.fail("polled without a flag"),
        find_bot=lambda: pytest.fail("looked for the bot without a flag"),
        stop_tree=lambda proc: pytest.fail("stopped the bot without a flag"),
    )
    monkeypatch.setattr(pause, "default_deps", lambda name, api, root: never)
    assert cli.main(["lichess", "pause", "--bot", "BlinkBot"]) == 2
    assert "blink lichess pause" in capsys.readouterr().err


def _code_names(path) -> set[str]:
    """Every name, attribute and non-docstring string constant a module's code uses."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            found.add(node.value)
    return found


def test_no_lichess_module_reads_a_process_environment_or_the_token_file():
    """psutil's Process.environ() would hand over the bot's token; the plan forbids reading it."""
    for path in Path(pause.__file__).parent.glob("*.py"):
        used = _code_names(path)
        assert not {"environ", "getenv", "environb", "putenv"} & used, path.name
        assert not any("dpapi" in text.lower() for text in used), path.name
