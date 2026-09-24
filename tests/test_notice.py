"""NOTICE (plan section 3, licences): every Apache-2.0 port, python-chess, every vendored licence file."""

import re
from pathlib import Path

import pytest

# An Apache-2.0 file carries the licence header as a comment at the start of a line.
APACHE_HEADER = re.compile(r"^# Licensed under the Apache License, Version 2\.0", re.MULTILINE)
KNOWN_APACHE_PORTS = {"blink/eval/puzzles.py", "blink/reference/deepmind.py"}


@pytest.fixture(scope="module")
def notice(repo_root: Path) -> str:
    return (repo_root / "NOTICE").read_text(encoding="utf-8")


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def test_the_notice_lists_every_apache_licensed_file(notice, repo_files, repo_root):
    apache = {
        _relative(path, repo_root)
        for path in repo_files
        if path.suffix == ".py" and APACHE_HEADER.search(path.read_text(encoding="utf-8"))
    }
    assert apache >= KNOWN_APACHE_PORTS
    assert sorted(name for name in apache if name not in notice) == []


def test_the_notice_credits_deepmind_under_the_apache_licence(notice):
    assert "google-deepmind/searchless_chess" in notice
    assert "DeepMind Technologies Limited" in notice
    assert "Apache License, Version 2.0" in notice


def test_the_notice_discloses_python_chess_as_an_unvendored_gpl_dependency(notice, repo_root):
    assert '"chess==' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "python-chess" in notice and "GPL-3.0-or-later" in notice and "not vendored" in notice


def test_the_notice_points_at_every_vendored_licence_file(notice, repo_files, repo_root):
    vendored = {
        _relative(path, repo_root)
        for path in repo_files
        if path.name.upper().startswith("LICENSE") and path.parent != repo_root
    }
    assert vendored  # cm-chessboard and the cburnett pieces
    assert sorted(name for name in vendored if name not in notice) == []
