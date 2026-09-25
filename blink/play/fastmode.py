"""The opt-in fast play modes: names, checks and flags shared by everything that plays or times Blink.

The play runtime is fp32 with no compile, and that stays the default everywhere. Two options trade
it for latency (P6 drops a size whose value-mode p99 at 219 rows is over 100 ms):

- precision "bf16": the trunk runs under bf16 autocast on CUDA, and both heads run in fp32 on its
  output cast to fp32 (blink.model.evaluator). bf16 needs CUDA: on the CPU it is refused, never
  quietly played as fp32, so a measurement is never labelled with a mode it did not run.
- compile: the trunk is wrapped with torch.compile(dynamic=True) for inference only. It needs a
  working inductor backend (CUDA with triton-windows here); the first calls of a process compile.

This module is torch-free, so the UCI engine, fastchess and `blink --help` can name the modes.
"""

import argparse

PRECISIONS = ("fp32", "bf16")
DEFAULT_PRECISION = "fp32"
_ON = frozenset({"on", "true", "yes", "1"})
_OFF = frozenset({"off", "false", "no", "0"})


def check(precision: str, device: str) -> None:
    """ValueError for an unknown precision, or for bf16 on anything but CUDA."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "bf16" and str(device).split(":", 1)[0] != "cuda":
        raise ValueError(f"precision bf16 plays on CUDA only, not on {device}: the CPU plays fp32")


def refusal(precision: str, compile: bool, device: str, deepmind: bool = False) -> str | None:
    """Why this mode cannot play here, or None. DeepMind's port always plays its own fp32 model."""
    if deepmind and not is_default(precision, compile):
        return "--precision and --compile apply to Blink models only, not to a dm: selector"
    try:
        check(precision, device)
    except ValueError as exc:
        return str(exc)
    return None


def is_default(precision: str, compile: bool) -> bool:
    return precision == DEFAULT_PRECISION and not compile


def tag(precision: str, compile: bool) -> str:
    """'' for the default mode, else '-bf16', '-compile' or '-bf16-compile' (engine and file names)."""
    parts = ([precision] if precision != DEFAULT_PRECISION else []) + (["compile"] if compile else [])
    return "".join(f"-{part}" for part in parts)


def describe(precision: str, compile: bool) -> str:
    """'fp32', 'bf16', 'fp32 compiled' or 'bf16 compiled' (log lines and reasons)."""
    return precision + (" compiled" if compile else "")


def uci_args(precision: str, compile: bool) -> tuple[str, ...]:
    """blink-uci flags for this mode; none for the default, so default engine commands never change."""
    precision_args = [] if precision == DEFAULT_PRECISION else [f"--precision={precision}"]
    return tuple(precision_args + (["--compile"] if compile else []))


def switch(text: str) -> bool:
    """`--compile` alone is on; `--compile=on|off` (also true/false, yes/no, 1/0) for lichess-bot's
    engine_options, which passes every option as --key=value."""
    lowered = str(text).strip().lower()
    if lowered in _ON:
        return True
    if lowered in _OFF:
        return False
    raise argparse.ArgumentTypeError(f"expected on or off, got {text!r}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """--precision fp32|bf16 and --compile[=on|off], both off by default."""
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default=DEFAULT_PRECISION,
        help="bf16: the trunk under bf16 autocast, both heads in fp32 (CUDA only; default fp32)",
    )
    parser.add_argument(
        "--compile",
        nargs="?",
        const=True,
        default=False,
        type=switch,
        metavar="on|off",
        help="torch.compile the trunk with dynamic shapes (inference only; --compile or --compile=on|off)",
    )
