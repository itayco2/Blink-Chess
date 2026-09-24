"""The document contract: every public number and the hook come from results/*.json (plan P12).

These pass on the work-in-progress README and turn strict once results/results.json exists: until the
evaluation writes it, the tests that need it skip and say so. The placeholder scan is strict today.
"""

import json
import re
from pathlib import Path

import pytest

from blink.report import claims
from blink.report import scoreboard as sb

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
README = REPO_ROOT / "README.md"
HOW_IT_WORKS = REPO_ROOT / "HOW-IT-WORKS.md"
PREFLIGHT = REPO_ROOT / "PREFLIGHT.md"
POST_DRAFT = REPO_ROOT / "post-draft.md"  # gitignored: Itay's LinkedIn draft, local only
BOT_BIO = REPO_ROOT / "deploy" / "lichess" / "bio.txt"  # the bot bio draft Itay pastes at G12
PUBLIC_DOCS = (
    "README.md",
    "HOW-IT-WORKS.md",
    "FINDINGS.md",
    "PREFLIGHT.md",
    "PROGRESS.md",
    "EVAL.md",
    "NOTICE",
)
PLACEHOLDERS = re.compile(
    r"\b(TODO|TBD|FIXME|XXX|PLACEHOLDER|UNVERIFIED)\b|lorem ipsum|(?<![\w_])__(?![\w_])"
)
FIRST_WORDS = 200
FILM_RUN = "long"  # the flagship run whose film is published (skeleton demo films are not checked)
LINK_TEXTS = ("Play", "Lichess", "How it learned")
PF_ROW = re.compile(r"\bPF(\d{2})\b")
PENDING = "results/results.json does not exist yet; this contract turns strict once the evaluation writes it"

REQUIRED_HEADINGS = (
    "## Scoreboard",
    sb.TABLE1_HEADING,
    sb.TABLE2_HEADING,
    sb.BANDS_HEADING,
    "## What this does not prove",
    "## Run the tests",
    "## Credits and licences",
)

# Numbers HOW-IT-WORKS may state that are fixed by the contract or by arithmetic, not measured.
NOT_MEASURED = {
    "1880": "the move vocabulary",
    "128": "value bins",
    "64": "board squares",
    "16": "square codes, and the ply-16 blocklist cut",
    "409,710,113": "lines in the Lichess eval DB (the data, not a result)",
    "7.54": "ln 1880, the loss of a network that knows nothing",
    "4.85": "ln 128",
    "1320": "the lowest Stockfish UCI_Elo anchor",
    "3000": "the rating a new Lichess BOT starts at",
    "3070": "the GPU's name",
    "21": "film frames",
}


def _strict() -> None:
    if not (RESULTS_DIR / "results.json").is_file():
        pytest.skip(PENDING)


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _rendered_hooks(lang: str) -> set[str]:
    """The hook the published film burned in, from render's sidecar (empty if not rendered here)."""
    from blink import paths

    sidecar = paths.home() / "film" / FILM_RUN / f"film-{lang}.json"
    return {json.loads(sidecar.read_text(encoding="utf-8"))["hook"]} if sidecar.is_file() else set()


def _words(text: str) -> list[str]:
    return re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL).split()


def test_readme_scoreboard_is_byte_equal_to_the_generated_block():
    _strict()
    block = sb.render_block(sb.load_bundle(RESULTS_DIR))
    assert sb.check_readme(README, block) == []


def test_the_first_200_words_hold_the_headline_links_and_caveat():
    _strict()
    head = " ".join(_words(_readme())[:FIRST_WORDS])
    bundle = sb.load_bundle(RESULTS_DIR)
    assert bundle.results.shipped is not None, "results.json names no shipped model"
    assert claims.hook("en", bundle.results.shipped.mode) in head
    for label in LINK_TEXTS:
        assert re.search(r"\[[^\]]*" + re.escape(label) + r"[^\]]*\]\(", head), f"no {label!r} link"
    assert sb.elo_cell(sb.shipped_row(bundle.results)) in head
    assert sb.ELO_CAVEAT in head


def test_every_number_in_how_it_works_is_in_results():
    """The MEASURED pattern: each HOW-IT-WORKS number is in results/*.json or is a named contract constant."""
    _strict()
    if not HOW_IT_WORKS.is_file():
        pytest.skip("HOW-IT-WORKS.md is not written yet (P12)")
    measured = sb.measured_values(RESULTS_DIR)
    stray = [
        number
        for number in sb.numbers_in(HOW_IT_WORKS.read_text(encoding="utf-8"))
        if number not in NOT_MEASURED and not sb.is_measured(number, measured)
    ]
    assert stray == []


def test_required_headings_appear_in_order():
    _strict()
    lines = _readme().splitlines()
    title = f"# {claims.HOOK_EN}"
    assert lines[0] == title
    positions = [lines.index(h) if h in lines else -1 for h in REQUIRED_HEADINGS]
    missing = [h for h, pos in zip(REQUIRED_HEADINGS, positions, strict=True) if pos < 0]
    assert missing == []
    assert positions == sorted(positions)


def test_no_placeholder_words_in_public_docs():
    offenders = []
    for name in PUBLIC_DOCS:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if PLACEHOLDERS.search(line):
                offenders.append(f"{name}:{number}: {line.strip()[:80]}")
    assert offenders == []


def test_the_placeholder_scan_catches_blanks_but_not_dunder_names():
    assert PLACEHOLDERS.search("it rates __ +/- __")
    assert PLACEHOLDERS.search("TBD")
    assert not PLACEHOLDERS.search("blink/film/__init__.py and window.__ready")


def test_preflight_rows_named_by_tests_exist():
    named: dict[str, set[str]] = {}
    for path in sorted((REPO_ROOT / "tests").glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for row in PF_ROW.findall(path.read_text(encoding="utf-8")):
            named.setdefault(f"PF{row}", set()).add(path.name)
    rows = {
        f"PF{n}" for n in re.findall(r"^\| PF(\d{2}) \|", PREFLIGHT.read_text(encoding="utf-8"), re.MULTILINE)
    }
    missing = sorted(
        f"{row} (named in {', '.join(sorted(files))})" for row, files in named.items() if row not in rows
    )
    if missing and not (RESULTS_DIR / "results.json").is_file():
        pytest.skip(f"PREFLIGHT rows not written yet: {'; '.join(missing)}. {PENDING}")
    assert missing == []


def test_hook_en_is_identical_in_readme_film_en_and_bot_bio():
    _strict()
    from blink.film import render

    hook = claims.HOOK_EN
    assert _readme().splitlines()[0] == f"# {hook}"
    mode = sb.load_bundle(RESULTS_DIR).results.shipped.mode
    assert render.hook_for("en", mode) == hook
    assert _rendered_hooks("en") <= {hook}
    assert BOT_BIO.is_file(), f"{BOT_BIO.relative_to(REPO_ROOT)} (the bot bio draft) is missing"
    assert BOT_BIO.read_text(encoding="utf-8").splitlines()[0] == hook


@pytest.mark.local
def test_hook_he_is_identical_in_film_he_and_post_draft():
    if not POST_DRAFT.is_file():
        pytest.skip("post-draft.md is gitignored and absent on this machine")
    _strict()
    from blink.film import render

    hook = claims.HOOK_HE
    mode = sb.load_bundle(RESULTS_DIR).results.shipped.mode
    assert render.hook_for("he", mode) == hook
    assert _rendered_hooks("he") <= {hook}
    assert POST_DRAFT.read_text(encoding="utf-8").splitlines()[0] == hook
