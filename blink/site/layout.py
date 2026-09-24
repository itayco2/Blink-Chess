"""What the deployed page is made of: the files under site/, minus build inputs, plus a few npm files.

onnxruntime-web and chess.js come from npm, pinned to an exact version and sha512 integrity in
site/package-lock.json, and are never tracked. ORT's wasm is 14 MB; chess.js's ESM build is one
3,368-line file, over the 800 lines test_no_source_file_exceeds_800_lines allows a tracked source file.
cm-chessboard, whose modules are small, is vendored in site/vendor instead (blink site vendor).
`blink site serve` maps the npm files in from site/node_modules and `blink site stage` copies them into
the deploy tree, both from NPM_FILES, so the local page and the published one load the same files.
"""

from dataclasses import dataclass
from pathlib import Path

NODE_MODULES = "node_modules"
# Build inputs, tests and the model directory: never part of the page itself.
NOT_DEPLOYED = frozenset({NODE_MODULES, "tests", "models", "package.json", "package-lock.json"})


@dataclass(frozen=True)
class NpmFiles:
    route: str  # the directory the page loads them from, e.g. "vendor/ort/"
    package: str  # the package's directory under node_modules
    source: str  # the directory inside the package that holds `files`
    files: tuple[str, ...]
    licence: str | None = None  # the package's licence file, published beside the files as LICENSE


NPM_FILES: tuple[NpmFiles, ...] = (
    # MIT; the package ships no LICENSE file, and each file carries Microsoft's MIT banner.
    NpmFiles(
        "vendor/ort/",
        "onnxruntime-web",
        "dist",
        ("ort.wasm.bundle.min.mjs", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"),
    ),
    NpmFiles("vendor/chess.js/", "chess.js", "dist/esm", ("chess.js",), licence="LICENSE"),  # BSD-2-Clause
)


def npm_sources(site_dir: Path) -> dict[str, Path]:
    """Published path (e.g. "vendor/chess.js/chess.js") -> its file under site/node_modules."""
    out: dict[str, Path] = {}
    for entry in NPM_FILES:
        package = site_dir / NODE_MODULES / entry.package
        for name in entry.files:
            out[entry.route + name] = package / entry.source / name
        if entry.licence:
            out[entry.route + "LICENSE"] = package / entry.licence
    return out


def is_npm_route(relative: str) -> bool:
    return relative.startswith(tuple(entry.route for entry in NPM_FILES))


def is_deployed(relative: str) -> bool:
    """False for a path under site/ that the page never ships (node_modules, tests, package files)."""
    return relative.split("/", 1)[0] not in NOT_DEPLOYED
