"""PR-4's looks at the endgame screen (EVAL.md section 5): when endgames.epd is declared unable to supply 700.

The screen reads endgames.epd front to back and looks at lines 1,000, 5,000 and 20,000, then every 20,000
lines. At a look it projects the file's total of kept positions from the positions screened so far:
kept x (unique positions in the file) / (positions screened). The file is declared unable to supply 700
if and only if the one-sided 95% Poisson upper bound on that projection is below 700 (the exact bound on
the Poisson mean given `kept`, scaled the same way), or the file ends with fewer than 700 kept. Over
157,824 unique positions that means kept <= 0, <= 14 and <= 73 at the first three looks. A look counts
every line up to and including its own, whatever the labelling batches; a repeated position counts as a
line but not as a screened position.

A plan without a file total (the fallback source) only records its looks: the rule is endgames.epd's.
"""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

FIRST_LOOKS = (1_000, 5_000, 20_000)
LOOK_EVERY = 20_000
CONFIDENCE = 0.95
NEED = 700  # positions the E8/E2b set needs (blink.eval.endgames.WANT)
_BISECTIONS = 200


def _log_cdf(kept: int, mean: float) -> float:
    """log P(X <= kept) for X ~ Poisson(mean), summed in log space (the mean can reach the hundreds)."""
    terms = [-mean + i * math.log(mean) - math.lgamma(i + 1) for i in range(kept + 1)]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def poisson_upper(kept: int, confidence: float = CONFIDENCE) -> float:
    """The exact one-sided upper confidence bound on a Poisson mean after observing `kept`: the mean at
    which P(X <= kept) = 1 - confidence (for kept = 0, -ln(1 - confidence))."""
    if kept == 0:
        return -math.log(1 - confidence)
    target = math.log(1 - confidence)
    low, high = float(kept), kept + 10.0 + 10 * math.sqrt(kept)
    for _ in range(_BISECTIONS):
        middle = (low + high) / 2
        low, high = (middle, high) if _log_cdf(kept, middle) > target else (low, middle)
    return (low + high) / 2


def look_lines(end: int, first: tuple[int, ...] = FIRST_LOOKS, every: int = LOOK_EVERY) -> Iterator[int]:
    """The look lines up to `end`: `first`, then every `every` lines after the last of them."""
    yield from (line for line in first if line <= end)
    line = first[-1] + every
    while line <= end:
        yield line
        line += every


@dataclass(frozen=True)
class LookPlan:
    positions: int | None  # unique positions in the whole source (None: record the looks, never declare)
    last_line: int  # the source's last line: reading it means the file ended
    need: int = NEED
    first: tuple[int, ...] = FIRST_LOOKS
    every: int = LOOK_EVERY


@dataclass(frozen=True)
class Look:
    line: int
    screened: int  # positions screened on lines up to `line` (repeats skipped)
    passed_screen: int  # of them, at +5.00 after the 1M-node search
    kept: int  # of them, confirmed at 10M nodes
    projected_total: float | None  # kept x positions / screened
    upper95_total: float | None  # the one-sided 95% Poisson upper bound on it
    need: int
    declares: bool


def look_at(plan: LookPlan, line: int, screened: int, passed: int, kept: int) -> Look:
    """PR-4's look at `line`: it declares when the upper bound on the projected file total is below
    plan.need. The screen's LookTracker calls this at each look line; it is the rule's only form."""
    if plan.positions is None or screened == 0:
        return Look(line, screened, passed, kept, None, None, plan.need, False)
    scale = plan.positions / screened
    upper = poisson_upper(kept) * scale
    return Look(line, screened, passed, kept, kept * scale, upper, plan.need, upper < plan.need)


def describe(look: Look) -> str:
    counts = f"line {look.line:,}: {look.kept} kept of {look.screened:,} screened"
    if look.upper95_total is None:
        return f"{counts} ({look.passed_screen} at +5.00 after the 1M-node search)"
    verdict = "<" if look.declares else ">="
    return (
        f"{counts}; projected file total {look.projected_total:.1f}, one-sided 95% Poisson upper bound "
        f"{look.upper95_total:.1f} {verdict} {look.need}"
    )


class LookTracker:
    """Takes a plan's looks as the screen passes their lines, in order."""

    def __init__(self, plan: LookPlan, on_look: Callable[[Look], None] | None = None) -> None:
        self.plan = plan
        self.on_look = on_look
        self.looks: list[Look] = []
        self._lines = look_lines(plan.last_line, plan.first, plan.every)
        self._next = next(self._lines, None)

    def reach(self, line: int, screened: int, passed: int, kept: int) -> Look | None:
        """Take every look at or before `line` (the screen has counted every position up to it); the
        first that declares, if any."""
        declaring = None
        while self._next is not None and self._next <= line:
            look = look_at(self.plan, self._next, screened, passed, kept)
            self.looks.append(look)
            if self.on_look is not None:
                self.on_look(look)
            declaring = declaring or (look if look.declares else None)
            self._next = next(self._lines, None)
        return declaring

    def ended(self, last_line: int, kept: int) -> str | None:
        """The end-of-file declaration: the source's last line was read with fewer than `need` kept."""
        if self.plan.positions is None or last_line < self.plan.last_line or kept >= self.plan.need:
            return None
        return f"the file ended at line {last_line:,} with {kept} kept (fewer than {self.plan.need})"
