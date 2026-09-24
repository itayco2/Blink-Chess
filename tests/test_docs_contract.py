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

# Public prose the MEASURED scan reads: README.md outside its generated scoreboard block (the block is
# checked byte-exact against results/*.json on its own), and every other public document with prose.
MEASURED_DOCS = ("README.md", "HOW-IT-WORKS.md", "FINDINGS.md", "PROGRESS.md")

# Numbers the public prose may state that are fixed by the contract or by arithmetic, not measured.
NOT_MEASURED = {
    "1880": "the move vocabulary",
    "128": "value bins",
    "64": "board squares",
    "16": "square codes, and the ply-16 blocklist cut",
    "409,710,113": "lines in the Lichess eval DB (the data, not a result)",
    "409.7M": "the same database size, in millions",
    "10K": "the name of DeepMind's 10K-puzzle set",
    "2.14": "the torch version",
    "7.54": "ln 1880, the loss of a network that knows nothing",
    "7.539": "ln 1880 to three decimals",
    "4.85": "ln 128",
    "4.852": "ln 128 to three decimals",
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
    if not sidecar.is_file():
        return set()
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    assert not report.get("preview"), f"{sidecar} is a preview render, made before a mode shipped"
    return {report["hook"]}


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


def _strays(doc: Path, results_dir: Path) -> list[str]:
    """Numbers in a public document's prose that neither results/*.json nor NOT_MEASURED explains."""
    prose = sb.prose_outside_block(doc.read_text(encoding="utf-8"))
    return sb.stray_numbers(prose, sb.measured_values(results_dir), NOT_MEASURED)


@pytest.mark.parametrize("name", MEASURED_DOCS)
def test_every_number_in_the_public_prose_is_in_results(name):
    """The MEASURED pattern: each number in README (outside the block), HOW-IT-WORKS, FINDINGS and PROGRESS
    is in results/*.json or is a named contract constant."""
    _strict()
    doc = REPO_ROOT / name
    if not doc.is_file():
        pytest.skip(f"{name} is not written yet")
    assert _strays(doc, RESULTS_DIR) == []


def test_the_measured_scan_reads_the_readme_story_and_skips_only_the_generated_block(tmp_path):
    from test_report_fixtures import FIXTURE_DIR

    readme = tmp_path / "README.md"
    lines = [
        "# Hook",
        "",
        "It solved 80.1% of the 10K puzzles, and lost by -35 Elo at 63-67% accuracy.",
        "",
        sb.START,
        "| 1851 +/- 36 (4,101 games) |",
        sb.END,
        "",
        "## What this does not prove",
        "",
        "+/-113.",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _strays(readme, FIXTURE_DIR) == ["-35", "63", "67%", "113"]


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
