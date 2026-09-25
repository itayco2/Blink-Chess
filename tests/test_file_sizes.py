"""The repo's rule: every Python file stays under 800 lines (split one before it gets there)."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIMIT = 800


def test_every_python_file_of_blink_tools_and_tests_is_under_800_lines():
    files = [p for folder in ("blink", "tools", "tests") for p in (REPO / folder).rglob("*.py")]
    long = {
        str(p.relative_to(REPO)): n
        for p in files
        if (n := len(p.read_text(encoding="utf-8").splitlines())) >= LIMIT
    }
    assert files
    assert long == {}
