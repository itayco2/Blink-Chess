"""pages.yml: the GitHub Pages deploy, gated, pinned, and fed by npm and the model-v1 release (P10, PF32).

PyYAML is not a dependency, so the workflow is read as text: a job is the block under its two-space
key in `jobs:`, and a step is the block that starts at its `- ` line.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "pages.yml"
USES = re.compile(r"^\s*(?:- )?uses: (\S+)(?:\s+# (\S+))?\s*$", re.MULTILINE)
PINNED = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")
UPLOAD_PAGES_ARTIFACT = "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9"
DEPLOY_PAGES = "actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346"


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(text: str, name: str) -> str:
    match = re.search(rf"^  {name}:\n(.*?)(?=^  [\w-]+:\n|\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, f"no job {name}"
    return match.group(1)


def _steps(job: str) -> list[str]:
    return re.split(r"^      - ", job, flags=re.MULTILINE)[1:]


def _step(job: str, needle: str) -> tuple[int, str]:
    return next((i, step) for i, step in enumerate(_steps(job)) if needle in step)


def test_pages_workflow_fetches_ort_and_model_and_tracks_neither(text, repo_files, repo_root):
    from blink.site import layout

    build = _job(text, "build")
    npm, _ = _step(build, "npm ci --prefix site")
    staged, stage_step = _step(build, "blink site stage")
    fetched, fetch_step = _step(build, "gh release download")
    assert npm < staged < fetched, "npm fetches ORT and chess.js, stage copies them, then the model lands"
    assert "blink site stage --out _site" in stage_step
    assert 'gh release download model-v1 -p "*.onnx" -p "model.json" -D _site/models' in fetch_step
    routes = {entry.route: (entry.package, entry.files, entry.licence) for entry in layout.NPM_FILES}
    wasm_entry = ("ort.wasm.bundle.min.mjs", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs")
    assert routes["vendor/ort/"] == ("onnxruntime-web", wasm_entry, None)
    assert routes["vendor/chess.js/"] == ("chess.js", ("chess.js",), "LICENSE")
    tracked = [path.relative_to(repo_root).as_posix() for path in repo_files]
    assert not [
        p for p in tracked if p.startswith(("site/vendor/ort/", "site/models/", "site/node_modules/"))
    ]
    assert not [p for p in tracked if p.endswith((".onnx", ".wasm")) or p == "site/vendor/chess.js/chess.js"]
    ignored = set((repo_root / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {"*.onnx", "site/models/", "site/node_modules/"} <= ignored


def test_pages_deploys_only_main_and_only_when_the_repository_variable_enables_it(text):
    guard = "    if: ${{ vars.PAGES_ENABLED == 'true' && github.ref == 'refs/heads/main' }}\n"
    assert guard in _job(text, "build")
    assert "    needs: build\n" in _job(text, "deploy")
    assert "    needs: deploy\n" in _job(text, "smoke")
    triggers = text[text.index("\non:\n") : text.index("\npermissions:\n")]
    assert "workflow_dispatch:" in triggers and "    branches: [main]\n" in triggers
    # a release runs on refs/tags/<tag>: github-pages admits only main, and it would build the tag's commit
    assert "release:" not in triggers and "tags:" not in triggers
    assert "gh workflow run pages.yml --ref main" in text[: text.index("\non:\n")], "G8's deploy step"


def test_every_action_is_pinned_by_a_full_commit_sha_with_its_tag_named(text):
    uses = USES.findall(text)
    assert uses and all(PINNED.match(action) and tag.startswith("v") for action, tag in uses), uses
    assert (UPLOAD_PAGES_ARTIFACT, "v5.0.0") in uses
    assert (DEPLOY_PAGES, "v5.0.1") in uses


def test_the_deploy_job_alone_can_write_pages_with_an_oidc_token(text):
    top = text[: text.index("\njobs:\n")]
    assert "permissions:\n  contents: read\n" in top
    assert "concurrency:\n  group: pages\n  cancel-in-progress: false\n" in top
    deploy = _job(text, "deploy")
    assert "    permissions:\n      pages: write\n      id-token: write\n" in deploy
    assert "      name: github-pages\n" in deploy
    assert "page_url: ${{ steps.deployment.outputs.page_url }}" in deploy
    assert "pages: write" not in _job(text, "build") and "pages: write" not in _job(text, "smoke")
    _, upload = _step(_job(text, "build"), "actions/upload-pages-artifact")
    assert "path: _site" in upload


def test_the_post_deploy_smoke_loads_selftest_and_requires_blink_self_test_ok(text):
    smoke = _job(text, "smoke")
    _, command = _step(smoke, "blink site smoke")
    assert '--url "${{ needs.deploy.outputs.page_url }}?selftest=1"' in command
    assert "--selftest" in command
    assert "uv sync --locked --no-group train --no-group compile" in smoke
    assert "playwright install --with-deps msedge" in smoke, "Edge is installed only if the runner lacks it"


def test_the_release_download_uses_the_workflow_token_and_no_secret(text):
    _, step = _step(_job(text, "build"), "gh release download")
    assert "GH_TOKEN: ${{ github.token }}" in step
    assert "secrets." not in text
