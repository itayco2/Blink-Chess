"""Every file the film and report commands write goes through a .tmp sibling and a replace.

A crash half way through a write (a full disk, a killed process) must leave the previous file whole,
never a torn JSON or a README cut off in the middle of its scoreboard.
"""

from pathlib import Path

import pytest

from blink.film import extract, pick, render
from blink.report import compute, scoreboard

OLD_JSON = '{"old": true}\n'
OLD_README = f"# Blink\n\n{scoreboard.START}\nold scoreboard\n{scoreboard.END}\n\nthe rest\n"

WRITERS = {
    "film.json": (OLD_JSON, lambda path: extract.write_film({"frames": [], "note": "new"}, path)),
    "candidates.json": (OLD_JSON, lambda path: pick.write_ranking([], path, "long", 21)),
    "compute.json": (
        OLD_JSON,
        lambda path: compute.write_compute({"schema_version": compute.SCHEMA_VERSION}, path),
    ),
    "film_en.json": (OLD_JSON, lambda path: render.write_sidecar({"out": "new"}, path.with_suffix(".mp4"))),
    "README.md": (OLD_README, lambda path: scoreboard.write_readme(path, "new scoreboard\n")),
}


def _crash_half_way(self, data, encoding=None, errors=None, newline=None):
    """Path.write_text that writes half its text, then fails like a full disk."""
    with open(self, "w", encoding=encoding, errors=errors, newline=newline) as handle:
        handle.write(data[: len(data) // 2])
    raise OSError(28, "no space left on device")


def _existing(tmp_path: Path, name: str) -> tuple[Path, str]:
    old, _ = WRITERS[name]
    target = tmp_path / name
    target.write_text(old, encoding="utf-8", newline="\n")
    return target, old


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_a_crash_half_way_through_a_write_leaves_the_previous_file_whole(tmp_path, monkeypatch, name):
    target, old = _existing(tmp_path, name)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "write_text", _crash_half_way)
        with pytest.raises(OSError, match="no space"):
            WRITERS[name][1](target)
    assert target.read_text(encoding="utf-8") == old


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_each_writer_replaces_its_file_with_lf_endings_and_leaves_no_temp_behind(tmp_path, name):
    target, old = _existing(tmp_path, name)
    WRITERS[name][1](target)
    data = target.read_bytes()
    assert data.decode("utf-8") != old and b"\r\n" not in data and data.endswith(b"\n")
    assert [p.name for p in tmp_path.iterdir()] == [name]
