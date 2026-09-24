"""`blink site stage --out DIR [--model M]`: the exact tree GitHub Pages serves, as plain static files.

The page's own files under site/ (minus layout.NOT_DEPLOYED), the npm files in layout.NPM_FILES, and,
with --model, models/model.onnx plus its card. pages.yml runs `npm ci --prefix site` and then this, so
the deploy can never drift from what `blink site serve` and its Edge smoke test load locally.
"""

import shutil
from pathlib import Path

from blink.site import layout, serve

NPM_CI_HINT = "run npm ci --prefix site"


class StageError(RuntimeError):
    pass


def page_sources(site_dir: Path) -> dict[str, Path]:
    """Published path -> file, for every file under site/ that the page ships."""
    out = {}
    for path in sorted(site_dir.rglob("*")):
        relative = path.relative_to(site_dir).as_posix()
        if path.is_file() and layout.is_deployed(relative):
            out[relative] = path
    return out


def npm_sources(site_dir: Path) -> dict[str, Path]:
    sources = layout.npm_sources(site_dir)
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise StageError(f"missing {missing[0]} ({len(missing)} npm files in all): {NPM_CI_HINT}")
    return sources


def _check_out(site_dir: Path, out: Path) -> None:
    if out.resolve().is_relative_to(site_dir.resolve()):
        raise StageError(f"{out} is inside {site_dir}: stage into a directory outside the site")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise StageError(f"{out} exists and is not empty: stage into a new or empty directory")


def _write(target: Path, source: Path | bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        target.write_bytes(source)
    else:
        shutil.copyfile(source, target)


def stage(site_dir: Path, out: Path, model: Path | None = None) -> list[Path]:
    """Copy the deployable page into `out` (new or empty); returns the written files."""
    _check_out(site_dir, out)
    sources: dict[str, Path | bytes] = {**page_sources(site_dir), **npm_sources(site_dir)}
    if model is not None:
        onnx = serve.resolve_model(model)
        sources[f"models/{serve.MODEL_FILE}"] = onnx
        sources[f"models/{serve.CARD_FILE}"] = serve.model_card(onnx)
    written = []
    for relative, source in sources.items():
        _write(out / relative, source)
        written.append(out / relative)
    return written
