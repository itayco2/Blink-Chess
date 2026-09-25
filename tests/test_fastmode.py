"""The opt-in fast play modes: names, checks and flags (torch-free)."""

import argparse

import pytest

from blink.play import fastmode


def test_the_default_mode_is_fp32_uncompiled_and_adds_no_flag_or_name():
    assert fastmode.DEFAULT_PRECISION == "fp32"
    assert fastmode.is_default("fp32", False)
    assert fastmode.uci_args("fp32", False) == ()
    assert fastmode.tag("fp32", False) == ""
    assert fastmode.describe("fp32", False) == "fp32"


@pytest.mark.parametrize(
    ("precision", "compile", "args", "tag"),
    [
        ("bf16", False, ("--precision=bf16",), "-bf16"),
        ("fp32", True, ("--compile",), "-compile"),
        ("bf16", True, ("--precision=bf16", "--compile"), "-bf16-compile"),
    ],
)
def test_a_fast_mode_has_its_own_flags_and_name(precision, compile, args, tag):
    assert not fastmode.is_default(precision, compile)
    assert fastmode.uci_args(precision, compile) == args
    assert fastmode.tag(precision, compile) == tag


def test_bf16_needs_cuda_and_an_unknown_precision_is_refused():
    fastmode.check("bf16", "cuda")
    fastmode.check("bf16", "cuda:0")
    fastmode.check("fp32", "cpu")
    with pytest.raises(ValueError, match="CUDA only"):
        fastmode.check("bf16", "cpu")
    with pytest.raises(ValueError, match="precision must be one of"):
        fastmode.check("fp16", "cuda")


def test_the_refusal_names_the_reason_or_is_none():
    assert fastmode.refusal("fp32", False, "cpu") is None
    assert fastmode.refusal("bf16", True, "cuda") is None
    assert "CUDA only" in fastmode.refusal("bf16", False, "cpu")
    assert fastmode.refusal("fp32", False, "cuda", deepmind=True) is None
    assert "Blink models only" in fastmode.refusal("fp32", True, "cuda", deepmind=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    fastmode.add_arguments(parser)
    return parser


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("fp32", False)),
        (["--precision", "bf16"], ("bf16", False)),
        (["--compile"], ("fp32", True)),
        (["--compile=on", "--precision=bf16"], ("bf16", True)),
        (["--compile=off"], ("fp32", False)),
        (["--compile=True"], ("fp32", True)),
    ],
)
def test_the_flags_parse_bare_or_with_a_value_as_lichess_bot_passes_them(argv, expected):
    args = _parser().parse_args(argv)
    assert (args.precision, args.compile) == expected


@pytest.mark.parametrize("argv", [["--precision", "fp16"], ["--compile=maybe"]])
def test_bad_flag_values_are_refused(argv):
    with pytest.raises(SystemExit):
        _parser().parse_args(argv)
