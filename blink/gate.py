"""`blink gate`: the under-a-minute check that runs before every phase.

It prints one ok/WARN/FAIL/skip line per check and names the fix for every failure.
Things that are simply not there yet (a download in progress, a tool not unpacked) are skips.
"""

import sys
from pathlib import Path

from blink import doctor, paths
from blink.checks import CheckResult

EVAL_DB_BYTES = 22_086_532_809
DM_PUZZLES_BYTES = 4_705_735
C_FLOOR = 12 * doctor.GB
D_FLOOR = 240 * doctor.GB


def check_python() -> CheckResult:
    version = sys.version_info
    if (version.major, version.minor) != (3, 12):
        return CheckResult("python", "FAIL", sys.version.split()[0], fix="uv python pin 3.12 && uv sync")
    return CheckResult("python", "ok", sys.version.split()[0])


def check_file(name: str, path: Path, expected_bytes: int | None, why_missing: str) -> CheckResult:
    if not path.exists():
        return CheckResult(name, "skip", f"{path} not present ({why_missing})")
    size = path.stat().st_size
    if expected_bytes is not None and size > expected_bytes:
        return CheckResult(
            name,
            "FAIL",
            f"{size:,} B is larger than the expected {expected_bytes:,} B, so it is corrupt (PF42)",
            fix=f"delete {path} and re-download it",
        )
    if expected_bytes is not None and size < expected_bytes:
        return CheckResult(name, "skip", f"{size:,} of {expected_bytes:,} B ({why_missing})")
    return CheckResult(name, "ok", f"{path} ({size:,} B)")


def check_layout(home: Path) -> CheckResult:
    missing = [name for name, sub in paths.layout(home).items() if not Path(sub).is_dir()]
    if missing:
        return CheckResult(
            "BLINK_HOME", "FAIL", f"{home} lacks {missing}", fix="uv run blink doctor --create-layout"
        )
    return CheckResult("BLINK_HOME", "ok", str(home))


def run() -> list[CheckResult]:
    home = paths.home()
    version, cuda = doctor.torch_facts()
    results = [
        check_python(),
        doctor.check_torch_build(version, cuda),
        check_layout(home),
        *doctor.disk_checks(sys.platform, C_FLOOR, D_FLOOR),
    ]
    results.append(
        check_file(
            "eval DB",
            home / "data" / "raw" / "lichess_db_eval.jsonl.zst",
            EVAL_DB_BYTES,
            "download in progress",
        )
    )
    results.append(
        check_file(
            "DeepMind puzzles",
            home / "downloads" / "puzzles.csv",
            DM_PUZZLES_BYTES,
            "download pending",
        )
    )
    tools = Path(r"D:\tools") if sys.platform == "win32" else home / "tools"
    results.append(
        check_file(
            "stockfish",
            tools / "stockfish" / "stockfish-windows-x86-64-universal.exe",
            None,
            "not unpacked yet",
        )
    )
    return results
