"""Where Blink keeps its heavy state.

The repo and its venv live on the fast NVMe drive (C:). Everything large (data shards, runs,
games, films) lives under BLINK_HOME, which defaults to D:\\blink on Windows because D: is the
drive with space. D: is a spinning disk, so readers of this state must stream it sequentially.
"""

import os
import sys
from collections.abc import Mapping
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from types import MappingProxyType

SUBDIRS = (
    "data",
    "runs",
    "games",
    "eval",
    "ship",
    "export",
    "film",
    "logs",
    "dm",
    "books",
    "lichess",
    "downloads",
)
WINDOWS_DEFAULT = r"D:\blink"


def default_home(platform: str = sys.platform, env: Mapping[str, str] = os.environ) -> PurePath:
    """BLINK_HOME if set, else D:\\blink on Windows and ~/.blink elsewhere."""
    pure = PureWindowsPath if platform == "win32" else PurePosixPath
    override = env.get("BLINK_HOME")
    if override:
        return pure(override)
    if platform == "win32":
        return pure(WINDOWS_DEFAULT)
    return pure(env.get("HOME", "~")) / ".blink"


def layout(home: PurePath) -> Mapping[str, PurePath]:
    """The named subdirectories under a home, as a read-only mapping."""
    return MappingProxyType({name: home / name for name in SUBDIRS})


def home() -> Path:
    """The concrete BLINK_HOME for this machine."""
    return Path(default_home())


def ensure_layout(root: Path | None = None) -> Mapping[str, Path]:
    """Create every subdirectory (idempotent) and return the concrete layout."""
    base = root if root is not None else home()
    dirs = {name: Path(path) for name, path in layout(base).items()}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return MappingProxyType(dirs)
