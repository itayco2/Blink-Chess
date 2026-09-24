"""The fast-mode flags reach the evaluator: the play factory, blink-uci, fastchess and `blink gauntlet`.

Every default stays what it was (fp32, no compile, the same engine names and commands); these tests
replace the model loader, so nothing here loads torch or touches a GPU.
"""

import importlib.machinery
import io
import sys
import types

import pytest

from blink import cli, uci
from blink.eval import fastchess
from blink.play import factory
from blink.play.oracles import RandomLogitEvaluator

LOADER = "blink.model.loading"


@pytest.fixture
def fake_loader(monkeypatch):
    """blink.model.loading replaced by a module whose load_evaluator records its arguments."""
    calls = []

    def load_evaluator(selector, device="cuda", **kwargs):
        calls.append({"selector": selector, "device": device, **kwargs})
        return RandomLogitEvaluator(0)

    module = types.ModuleType(LOADER)
    module.__spec__ = importlib.machinery.ModuleSpec(LOADER, None)
    module.load_evaluator = load_evaluator
    monkeypatch.setitem(sys.modules, LOADER, module)
    return calls


def test_the_factory_passes_the_mode_to_the_model_loader(fake_loader):
    factory.load_evaluator("run:x", device="cuda", precision="bf16", compile=True)
    factory.load_evaluator("run:x", device="cpu")
    assert fake_loader == [
        {"selector": "run:x", "device": "cuda", "precision": "bf16", "compile": True},
        {"selector": "run:x", "device": "cpu", "precision": "fp32", "compile": False},
    ]


def test_the_factory_refuses_bf16_off_cuda_before_loading(fake_loader):
    with pytest.raises(ValueError, match="CUDA only"):
        factory.load_evaluator("run:x", device="cpu", precision="bf16")
    assert fake_loader == []


def test_the_random_network_has_no_mode_of_its_own(fake_loader):
    evaluator = factory.load_evaluator("random", device="cuda", precision="bf16", compile=True)
    assert isinstance(evaluator, RandomLogitEvaluator) and fake_loader == []


def _run_uci(argv: list[str]) -> tuple[int, list[str]]:
    out = io.StringIO()
    code = uci.main(argv, io.StringIO("uci\nisready\nposition startpos\ngo movetime 100\nquit\n"), out)
    return code, out.getvalue().splitlines()


def test_the_uci_flags_reach_the_evaluator_and_name_the_engine(fake_loader):
    code, lines = _run_uci(["--model", "run:x", "--mode", "value", "--precision", "bf16", "--compile"])
    assert code == 0
    assert fake_loader == [{"selector": "run:x", "device": "cuda", "precision": "bf16", "compile": True}]
    assert "id name Blink-value-bf16-compile" in lines
    assert any(line.startswith("bestmove") for line in lines)


def test_the_uci_defaults_are_fp32_uncompiled_under_the_usual_name(fake_loader):
    code, lines = _run_uci(["--model", "run:x", "--mode", "policy", "--compile=off"])
    assert code == 0
    assert fake_loader == [{"selector": "run:x", "device": "cuda", "precision": "fp32", "compile": False}]
    assert "id name Blink-policy" in lines


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--model", "run:x", "--device", "cpu", "--precision", "bf16"], "CUDA only"),
        (["--model", "dm:9M", "--compile"], "Blink models only"),
    ],
)
def test_the_uci_engine_refuses_a_mode_it_cannot_play_before_any_traffic(fake_loader, capsys, argv, message):
    code, lines = _run_uci(argv)
    assert code == 2 and lines == []
    assert message in capsys.readouterr().err and fake_loader == []


def test_a_default_blink_engine_is_unchanged():
    spec = fastchess.blink_engine("run:skeleton", mode="value", device="cuda")
    assert spec.args == ("-m", "blink.uci", "--model=run:skeleton", "--mode=value", "--device=cuda")
    assert spec.name == "Blink-value-run_skeleton"


def test_a_fast_blink_engine_carries_its_flags_and_its_mode_in_the_name():
    spec = fastchess.blink_engine("run:skeleton", "value", "cuda", precision="bf16", compile=True)
    assert spec.args[-2:] == ("--precision=bf16", "--compile")
    assert spec.name == "Blink-value-run_skeleton-bf16-compile"
    options = dict(token.split("=", 1) for token in spec.fastchess_args() if "=" in token)
    assert options["args"].endswith("--device=cuda --precision=bf16 --compile")


def test_a_fast_blink_engine_on_the_cpu_is_refused():
    with pytest.raises(ValueError, match="CUDA only"):
        fastchess.blink_engine("run:skeleton", "value", "cpu", precision="bf16")


def test_blink_gauntlet_dry_run_passes_the_mode_to_the_engine(tmp_path, capsys):
    args = ["gauntlet", "--model", "random", "--mode", "value", "--games", "10", "--out", str(tmp_path)]
    assert cli.main([*args, "--precision", "bf16", "--compile", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "--precision=bf16 --compile" in printed and "name=Blink-value-random-bf16-compile" in printed


def test_blink_gauntlet_refuses_a_fast_mode_it_cannot_play(tmp_path, capsys):
    base = ["gauntlet", "--games", "10", "--out", str(tmp_path), "--dry-run"]
    assert cli.main([*base, "--model", "random", "--device", "cpu", "--precision", "bf16"]) == 2
    assert "CUDA only" in capsys.readouterr().err
    assert cli.main([*base, "--model", "dm:9M", "--compile"]) == 2
    assert "Blink models only" in capsys.readouterr().err
