"""The pre-registered mode choice: a pentanomial logistic SPRT with fishtest parity (plan section 1, E3).

`llr_logistic` is fishtest's `LLRcalc.LLR_logistic` ported exactly: the generalised log-likelihood ratio
of the observed pentanomial for the expected score s1 = L(elo1) against s0 = L(elo0), each alternative
being the multinomial maximum-likelihood fit with that expectation (the secular equation, solved here by
bisection where fishtest calls scipy's brentq). Wald's bounds: accept H0 at LLR <= log(beta / (1 - alpha)),
H1 at LLR >= log((1 - beta) / alpha).

The rule (EVAL.md section 2): value against policy with elo0 = 0, elo1 = 20, alpha = beta = 0.05, cap 6,000
games on the dev slice.
- H1: value ships.
- H0: the reverse SPRT (policy against value) with the same bounds; H1 there: policy ships; H0 again:
  "statistically tied", policy ships as the stricter claim.
- A cap without a decision: the higher point estimate ships (an exact tie ships policy).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from blink.eval import rating

PairScores = tuple[float, float]  # the first player's score in the two games of one opening
SECULAR_ITERATIONS = 200


@dataclass(frozen=True)
class SprtConfig:
    elo0: float = 0.0
    elo1: float = 20.0
    alpha: float = 0.05
    beta: float = 0.05
    cap_games: int = 6000

    def __post_init__(self) -> None:
        if self.cap_games <= 0 or self.cap_games % 2:
            raise ValueError(f"the cap must be a positive even number of games (pairs), got {self.cap_games}")
        if not self.elo0 < self.elo1:
            raise ValueError(f"elo0 must be below elo1, got {self.elo0} and {self.elo1}")


MODE_SPRT = SprtConfig()


def bounds(alpha: float, beta: float) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def _pdf(counts: Sequence[float]) -> tuple[float, list[tuple[float, float]]]:
    """fishtest results_to_pdf: (N, [(i / (k - 1), count_i / N)]) after regularisation."""
    regular = rating.regularize(counts)
    total = sum(regular)
    k = len(regular)
    return total, [(i / (k - 1), c / total) for i, c in enumerate(regular)]


def _mean(pdf: Sequence[tuple[float, float]]) -> float:
    return sum(p * a for a, p in pdf)


def _secular(pdf: Sequence[tuple[float, float]]) -> float:
    """The root x of sum_i p_i a_i / (1 + x a_i) = 0 on (-1/max a, -1/min a); f falls in x."""
    values = [a for a, _ in pdf]
    low_a, high_a = min(values), max(values)
    if low_a * high_a >= 0:
        raise ValueError("the secular equation needs support on both sides of zero")
    epsilon = 1e-9
    lo, hi = -1 / high_a + epsilon, -1 / low_a - epsilon

    def f(x: float) -> float:
        return sum(p * a / (1 + x * a) for a, p in pdf)

    for _ in range(SECULAR_ITERATIONS):
        mid = (lo + hi) / 2
        value = f(mid)
        if value == 0 or hi - lo < 1e-16:
            return mid
        lo, hi = (mid, hi) if value > 0 else (lo, mid)
    return (lo + hi) / 2


def _mle_expected(pdf: Sequence[tuple[float, float]], s: float) -> list[tuple[float, float]]:
    """fishtest MLE_expected: the closest distribution (in likelihood) with expectation s."""
    x = _secular([(a - s, p) for a, p in pdf])
    return [(a, p / (1 + x * (a - s))) for a, p in pdf]


def llr_logistic(elo0: float, elo1: float, counts: Sequence[int]) -> float:
    """fishtest LLR_logistic for a trinomial (L, D, W) or a pentanomial."""
    if len(counts) not in (3, 5):
        raise ValueError(f"counts must hold 3 or 5 entries, got {len(counts)}")
    total, pdf = _pdf(counts)
    s0, s1 = rating.logistic_score(elo0), rating.logistic_score(elo1)
    pdf0, pdf1 = _mle_expected(pdf, s0), _mle_expected(pdf, s1)
    jumps = [
        (math.log(p1) - math.log(p0), p) for (_, p0), (_, p1), (_, p) in zip(pdf0, pdf1, pdf, strict=True)
    ]
    return total * _mean(jumps)


@dataclass(frozen=True)
class SprtState:
    penta: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)

    @property
    def games(self) -> int:
        return 2 * sum(self.penta)

    def with_pair(self, first: float, second: float) -> "SprtState":
        index = round(2 * (first + second))
        return SprtState(tuple(c + (i == index) for i, c in enumerate(self.penta)))


@dataclass(frozen=True)
class SprtResult:
    penta: tuple[int, ...]
    llr: float
    lower: float
    upper: float
    verdict: str | None  # "H1", "H0", or None when the cap stopped it
    capped: bool
    elo: float  # fishtest get_elo on the pentanomial: the first player's Elo over the second
    elo_ci95: float
    games: int

    def as_dict(self) -> dict:
        return {**self.__dict__, "penta": list(self.penta)}


def _result(state: SprtState, config: SprtConfig) -> SprtResult:
    lower, upper = bounds(config.alpha, config.beta)
    llr = llr_logistic(config.elo0, config.elo1, state.penta) if state.games else 0.0
    verdict = "H1" if llr >= upper else "H0" if llr <= lower else None
    capped = verdict is None and state.games >= config.cap_games
    estimate = rating.elo_ci(state.penta) if state.games else rating.EloEstimate(0.0, math.inf, 0.5, 0)
    return SprtResult(
        state.penta, llr, lower, upper, verdict, capped, estimate.elo, estimate.ci95, state.games
    )


def run_sprt(play_pair: Callable[[int], PairScores], config: SprtConfig = MODE_SPRT) -> SprtResult:
    """Play pair 0, 1, 2, ... (each: one opening, both colours) until a bound is crossed or the cap."""
    state = SprtState()
    index = 0
    while True:
        result = _result(state, config)
        if result.verdict is not None or result.capped:
            return result
        state = state.with_pair(*play_pair(index))
        index += 1


@dataclass(frozen=True)
class ModeChoice:
    mode: str  # "value" or "policy"
    rule: str  # forward H1 | reverse H1 | tied | cap
    forward: SprtResult
    reverse: SprtResult | None = None

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "rule": self.rule,
            "forward": self.forward.as_dict(),
            "reverse": None if self.reverse is None else self.reverse.as_dict(),
        }


def needs_reverse(forward: SprtResult) -> bool:
    return forward.verdict == "H0"


def choose_mode(forward: SprtResult, reverse: SprtResult | None = None) -> ModeChoice:
    """The pre-registered rule. `forward` is value against policy; `reverse` policy against value."""
    if forward.verdict == "H1":
        return ModeChoice("value", "forward H1", forward)
    if forward.verdict is None:
        return ModeChoice("value" if forward.elo > 0 else "policy", "cap", forward)
    if reverse is None:
        raise ValueError("the forward SPRT accepted H0: the reverse SPRT (policy against value) must run")
    if reverse.verdict == "H1":
        return ModeChoice("policy", "reverse H1", forward, reverse)
    if reverse.verdict == "H0":
        return ModeChoice("policy", "tied", forward, reverse)
    return ModeChoice("policy" if reverse.elo >= 0 else "value", "cap", forward, reverse)


def run_mode_choice(
    value_vs_policy: Callable[[int], PairScores],
    policy_vs_value: Callable[[int], PairScores],
    config: SprtConfig = MODE_SPRT,
) -> ModeChoice:
    """The forward SPRT, then the reverse one only after an H0, then the rule."""
    forward = run_sprt(value_vs_policy, config)
    reverse = run_sprt(policy_vs_value, config) if needs_reverse(forward) else None
    return choose_mode(forward, reverse)
