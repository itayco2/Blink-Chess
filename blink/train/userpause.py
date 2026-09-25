"""The user pause: BLINK_HOME/PAUSE makes Blink training let go of the GPU until the file is removed.

Itay games on the training PC. "Pause Blink.cmd" (deploy/pause/) creates the flag. The trainer notices
it at its next step boundary, or between two eval chunks, writes a checkpoint at that step and exits
with supervise.EXIT_USER_PAUSE. The supervisor counts that as no crash: it evaluates no stop rule while
paused, stops the run's wall-clock deadline counting, and restarts the trainer with --resume once
"Resume Blink.cmd" removes the flag. A trainer, supervisor or sweep arm that would start while the flag
is up waits first, before it takes any GPU memory. Heartbeats say "paused: user" meanwhile, which
`blink status` and `blink ops ps` show. The P7-VAA pause is another thing entirely: a pending gate only
Itay clears, which removing this flag never lifts.

The trainer asks at every step, so the check is throttled: one os.path.exists (77 us on D: while a run
trains) at most every few seconds, and a clock read (about 50 ns) otherwise. Torch-free.
"""

import os
import time
from collections.abc import Callable
from pathlib import Path

from blink import paths

FLAG_NAME = "PAUSE"
PAUSED_USER = "paused: user"  # the heartbeat state (and supervisor.json state) of a user pause
POLL_S = 5.0  # how often a paused process looks for the flag again


def flag_path(home: Path | None = None) -> Path:
    """BLINK_HOME/PAUSE: the file the Pause and Resume buttons create and remove."""
    return Path(home if home is not None else paths.home()) / FLAG_NAME


class FlagWatch:
    """Whether the flag is up, asking the disk at most once every `every_s` seconds.

    Mutable by nature: it remembers when it last looked. Between looks it answers False at the cost of a
    clock read, so the trainer can ask at every step without slowing it."""

    def __init__(
        self,
        path: Path,
        every_s: float,
        clock: Callable[[], float] = time.monotonic,
        exists: Callable[[Path], bool] = os.path.exists,
    ):
        self.path, self.every_s, self.clock, self.exists = Path(path), every_s, clock, exists
        self.next_look = float("-inf")

    def requested(self) -> bool:
        now = self.clock()
        if now < self.next_look:
            return False
        self.next_look = now + self.every_s
        return bool(self.exists(self.path))


def wait_while_flagged(
    path: Path,
    poll_s: float,
    on_wait: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Block while the flag exists, calling `on_wait` (a heartbeat, say) before each look; the seconds
    it waited (0.0 when the flag was not up)."""
    started = clock()
    while Path(path).exists():
        if on_wait is not None:
            on_wait()
        sleep(poll_s)
    return clock() - started


def describe(path: Path, now: float | None = None) -> str | None:
    """One line for `blink ops ps` while the flag is up, else None."""
    try:
        since = Path(path).stat().st_mtime
    except OSError:
        return None
    minutes = ((time.time() if now is None else now) - since) / 60
    return (
        f"user pause: {path} has been up for {minutes:.0f} min; Blink training waits (holding no GPU) "
        "until Resume Blink removes it"
    )
