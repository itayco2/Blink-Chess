"""The result type shared by `blink doctor` and `blink gate`, and how results print."""

from collections.abc import Sequence
from dataclasses import dataclass

STATUSES = ("ok", "WARN", "FAIL", "skip")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    fix: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")
        if self.status == "FAIL" and not self.fix:
            raise ValueError(f"check {self.name!r} failed without naming a fix")


def format_result(result: CheckResult) -> str:
    line = f"{result.status:<4} {result.name}: {result.detail}"
    if result.fix:
        line += f"  -> fix: {result.fix}"
    return line


def format_results(results: Sequence[CheckResult]) -> str:
    return "\n".join(format_result(r) for r in results)


def exit_code(results: Sequence[CheckResult]) -> int:
    """1 if anything failed; warnings and skips never fail the gate."""
    return 1 if any(r.status == "FAIL" for r in results) else 0
