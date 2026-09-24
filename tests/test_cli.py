import os
import subprocess
import sys

import pytest

from blink import cli

NON_ASCII = "σ → בלינק"  # sigma, arrow, and "Blink" in Hebrew


def _env_without_utf8_overrides() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONUTF8", "PYTHONIOENCODING"):
        env.pop(key, None)
    return env


def _run_redirected(code: str, tmp_path) -> tuple[int, bytes]:
    out = tmp_path / "out.txt"
    with open(out, "wb") as handle:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=handle,
            stderr=subprocess.PIPE,
            env=_env_without_utf8_overrides(),
        )
    return proc.returncode, out.read_bytes()


def test_cli_survives_non_ascii_output_when_redirected(tmp_path):
    code = f"from blink.cli import configure_stdio; configure_stdio(); print({NON_ASCII!r})"
    returncode, data = _run_redirected(code, tmp_path)
    assert returncode == 0
    assert data.decode("utf-8").strip() == NON_ASCII


def _windows_ansi_code_page_is_utf8() -> bool:
    """True when Windows' "Use Unicode UTF-8 for worldwide language support" is on (ACP 65001)."""
    if sys.platform != "win32":
        return False
    import ctypes

    return ctypes.windll.kernel32.GetACP() == 65001


@pytest.mark.skipif(sys.platform != "win32", reason="the ANSI code page problem is Windows-only")
@pytest.mark.skipif(
    _windows_ansi_code_page_is_utf8(), reason="this machine's ANSI code page is already UTF-8"
)
def test_without_the_reconfigure_a_redirected_print_crashes_on_windows(tmp_path):
    """PF39: this is the failure configure_stdio exists to prevent."""
    returncode, _ = _run_redirected(f"print({NON_ASCII!r})", tmp_path)
    assert returncode != 0


def test_help_lists_the_p0_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for command in ("doctor", "gate", "heartbeat-probe"):
        assert command in text
