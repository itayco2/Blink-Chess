"""House rules every tracked file must follow. Each test names the rule it locks."""

import re
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".js", ".mjs"}

# Built from code points so this file never contains the characters it bans.
MACHINE_PUNCTUATION = {
    chr(0x2014): "em dash",
    chr(0x2013): "en dash",
    chr(0x2026): "ellipsis character",
    chr(0x201C): "curly double quote",
    chr(0x201D): "curly double quote",
    chr(0x2018): "curly single quote",
    chr(0x2019): "curly single quote",
}
EMOJI = re.compile(f"[{chr(0x1F300)}-{chr(0x1FAFF)}{chr(0x2600)}-{chr(0x27BF)}]")
TOKEN = re.compile(r"(lip|lio)_[A-Za-z0-9]{16,}")
LOCAL_USER_PATH = "C:" + "\\" + "Users" + "\\"
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _texts(repo_files: list[Path]) -> list[tuple[Path, str]]:
    """Every tracked file that decodes as UTF-8, whatever its name: .env, .log, LICENSE, .gitignore.

    Filtering by extension left extensionless and .env-style files unscanned, which are exactly
    where secrets leak (security review, 2026-09-24).
    """
    texts = []
    for path in repo_files:
        try:
            texts.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue  # binary files (images, fonts) cannot hold these text patterns
    return texts


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_machine_punctuation_in_authored_files(repo_files):
    offenders = []
    for path, text in _texts(repo_files):
        for char, name in MACHINE_PUNCTUATION.items():
            if char in text:
                offenders.append(f"{path.name}: {name}")
        if EMOJI.search(text):
            offenders.append(f"{path.name}: emoji")
    assert offenders == []


def resolve_link(doc: Path, target: str, repo_root: Path) -> Path:
    """Where a markdown link points. A leading slash means the repo root, as GitHub renders it."""
    bare = target.split("#", 1)[0]
    if bare.startswith("/"):
        return repo_root / bare.lstrip("/")
    return doc.parent / bare


def test_root_relative_links_resolve_against_the_repo_not_the_disk(repo_root):
    doc = repo_root / "deploy" / "lichess" / "RUNBOOK.md"
    assert resolve_link(doc, "/PREFLIGHT.md", repo_root) == repo_root / "PREFLIGHT.md"
    assert resolve_link(doc, "start-bot.template.ps1#top", repo_root) == doc.parent / "start-bot.template.ps1"


def test_every_relative_link_in_docs_resolves(repo_files, repo_root):
    broken = []
    for path in (p for p in repo_files if p.suffix.lower() == ".md"):
        for target in MARKDOWN_LINK.findall(_read(path)):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not resolve_link(path, target, repo_root).exists():
                broken.append(f"{path.name} -> {target}")
    assert broken == []


def test_no_tracked_file_exceeds_512_kb(repo_files):
    too_big = [f"{p.name}: {p.stat().st_size} B" for p in repo_files if p.stat().st_size > 512 * 1024]
    assert too_big == []


def test_no_source_file_exceeds_800_lines(repo_files):
    too_long = []
    for path in (p for p in repo_files if p.suffix.lower() in SOURCE_SUFFIXES):
        count = _read(path).count("\n")
        if count > 800:
            too_long.append(f"{path.name}: {count} lines")
    assert too_long == []


def test_no_lichess_token_in_tracked_files(repo_files):
    leaks = [p.name for p, text in _texts(repo_files) if TOKEN.search(text)]
    assert leaks == []


def test_the_token_scan_sees_extensionless_and_env_files(tmp_path):
    for name in (".env", "bot.log", "NOTES"):
        (tmp_path / name).write_text("token lip_" + "A" * 20, encoding="utf-8")
    scanned = {p.name for p, text in _texts(list(tmp_path.iterdir())) if TOKEN.search(text)}
    assert scanned == {".env", "bot.log", "NOTES"}


def test_no_tracked_file_contains_a_local_user_path(repo_files):
    leaks = [p.name for p, text in _texts(repo_files) if LOCAL_USER_PATH in text]
    assert leaks == []
