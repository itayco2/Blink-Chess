"""Atomic file replacement that survives Windows' sharing rules.

On Windows os.replace raises PermissionError while another process holds the target, or the freshly
written temp file, open without delete-sharing: the live dashboard polling metrics.jsonl, a tail, an
editor, an antivirus scan of a file just written. Such holds last milliseconds, so the replace is
retried for up to about two seconds before the error is let through. Any other OSError is raised at
once. Torch-free, so every writer in blink.train can share it.
"""

import os
import time
from pathlib import Path

REPLACE_RETRIES = 10
RETRY_SLEEP_S = 0.2


def replace_with_retry(tmp: Path, path: Path) -> None:
    """os.replace(tmp, path), retried while a reader or scanner briefly holds either file."""
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP_S)


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text to path.tmp, then replace path with it; readers never see a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    replace_with_retry(tmp, path)
