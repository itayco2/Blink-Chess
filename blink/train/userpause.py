"""The user pause: BLINK_HOME/PAUSE makes Blink training let go of the GPU until the file is removed.

Itay games on the training PC. "Pause Blink.cmd" (deploy/pause/) creates the flag. A trainer under
`blink supervise` (every sweep arm, the flagship) notices it at its next step boundary, or between two
eval chunks, writes a checkpoint at that step and exits with supervise.EXIT_USER_PAUSE. The supervisor
counts that as no crash: it evaluates no stop rule while paused, stops the run's wall-clock deadline
counting, and restarts the trainer with --resume once "Resume Blink.cmd" removes the flag and the GPU
has the free memory the run measured at its start, less RESUME_SLACK_GB (a game still open, or only
minimized, holds several GB: the restart would measure that GPU and train at a smaller micro-batch or
spill, and a throughput stop is final for an arm). A trainer, supervisor or sweep arm that would start
while the flag is up waits first, before it takes any GPU memory. Heartbeats say "paused: user"
meanwhile, which `blink status` and `blink ops ps` show.

Only a trainer whose resumer says so (RESUMER_ENV, set by a supervisor watching the flag, or by the P6
v2 driver for the calibration it reruns as a fresh run after a pause) stops mid-run. Anything else
trains on through a pause: a plain `blink train` (a calibration or a job run by hand) would fail its
caller with exit 75, a resumed calibration would time the pause into its rate, and an older supervisor
would count exit 75 as a crash. The Pause button names such a process as one that could not free the
GPU. The P7-VAA pause is another thing entirely: a pending gate only Itay clears,
which removing this flag never lifts.

The trainer asks at every step, so the check is throttled: one os.path.exists (77 us on D: while a run
trains) at most every few seconds, and a clock read (about 50 ns) otherwise. Torch-free.
"""

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from blink import paths

FLAG_NAME = "PAUSE"
PAUSED_USER = "paused: user"  # the heartbeat state (and supervisor.json state) of a user pause
POLL_S = 5.0  # how often a paused process looks for the flag again
RESUMER_ENV = "BLINK_PAUSE_RESUMER"  # "1": this process's resumer restarts it after a user pause
# A resumed run needs the free VRAM it measured at its start less this. abl-a01..a05 measured 6.95 GB
# free at micro-batch 1024 (peak 5.85 GB): 1 GB less still leaves micro-batch 512, above P5's
# throughput floor, and normal desktop drift fits in it, while a game holds several GB. nvidia-smi
# reads a little more free than the trainer's own measure, which comes after its CUDA context.
RESUME_SLACK_GB = 1.0
MIB_PER_GB = 1024


def flag_path(home: Path | None = None) -> Path:
    """BLINK_HOME/PAUSE: the file the Pause and Resume buttons create and remove."""
    return Path(home if home is not None else paths.home()) / FLAG_NAME


def resumer_present(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process runs under a resumer: a supervisor that restarts it with --resume after a user
    pause, or the P6 v2 driver that reruns its calibration. Only then may a trainer stop for the flag
    mid-run."""
    return (os.environ if environ is None else environ).get(RESUMER_ENV) == "1"


def resume_need_gb(run_dir: Path) -> float | None:
    """The free GPU memory (GB) a paused run needs before it restarts: what it measured at its start
    (config.json's vram.free_gb) less RESUME_SLACK_GB. None when it measured none (not started yet,
    a CPU run)."""
    try:
        record = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
        free = record["vram"]["free_gb"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(free, int | float):
        return None
    return float(free) - RESUME_SLACK_GB


def gpu_free_gb(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> float | None:
    """Free memory on the first GPU (this PC has one) in GB, from nvidia-smi, which takes no GPU memory
    of its own; None when it cannot be read. No console window opens for it, even over a game."""
    argv = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    hidden = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows only; 0 elsewhere
    try:
        done = run(argv, capture_output=True, text=True, timeout=30, creationflags=hidden)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        return float(done.stdout.split()[0]) / MIB_PER_GB
    except (IndexError, ValueError):
        return None


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
