"""`blink ops install-pause-buttons`: "Pause Blink.cmd" and "Resume Blink.cmd" onto the desktop.

The buttons live in deploy/pause/ (blink.train.userpause says what they do). The installer copies both
into a folder that already exists, by default the user's Desktop, and never creates one; only a person
runs it, since it writes that desktop. Each copy gets CRLF line ends, which cmd.exe needs, whatever
line ends the checkout has.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUTTONS_DIR = REPO_ROOT / "deploy" / "pause"
BUTTONS = ("Pause Blink.cmd", "Resume Blink.cmd")


def default_target() -> Path:
    return Path.home() / "Desktop"


def install(target: Path, source: Path = BUTTONS_DIR) -> list[Path]:
    """Copy both buttons into `target` (replacing older copies); the paths written."""
    target = Path(target)
    if not target.is_dir():
        raise FileNotFoundError(f"{target} is not a folder; name one with --to")
    written = []
    for name in BUTTONS:
        text = (source / name).read_bytes().replace(b"\r\n", b"\n")
        (target / name).write_bytes(text.replace(b"\n", b"\r\n"))
        written.append(target / name)
    return written
