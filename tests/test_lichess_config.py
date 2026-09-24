"""The Lichess bot config must switch off every lookup lichess-bot can make for the engine (N3)."""

import copy
import importlib
import json
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from blink import uci
from blink.lichess import config_check

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "deploy" / "lichess" / "config.template.yml"


def load_template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def console_scripts() -> dict[str, str]:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def lichess_bot_flags(engine_options: dict) -> list[str]:
    """lichess-bot appends each engine_options entry to the engine command as --key=value."""
    return [f"--{key}={value}" for key, value in engine_options.items()]


def test_the_bot_engine_is_a_console_script_the_project_installs():
    engine = load_template()["engine"]
    script = PurePosixPath(engine["name"])
    assert script.suffix == ".exe"
    assert PurePosixPath(engine["dir"]).parts[-2:] == (".venv", "Scripts")
    target = console_scripts().get(script.stem)
    assert target is not None, f"pyproject.toml [project.scripts] has no {script.stem!r}"
    module_name, _, function_name = target.partition(":")
    assert getattr(importlib.import_module(module_name), function_name) is uci.main


def test_the_bot_engine_options_are_flags_blink_uci_accepts():
    engine = load_template()["engine"]
    args = uci.build_parser().parse_args(lichess_bot_flags(engine["engine_options"]))
    assert (args.model, args.mode, args.device) == ("ship", "policy", "cuda")
    assert args.log == Path("D:/blink/lichess/decisions.jsonl")
    assert args.random is False


def test_lichess_config_disables_every_lookup():
    config = load_template()
    assert config_check.lookup_problems(config) == []
    engine = config["engine"]
    assert engine["polyglot"]["enabled"] is False
    assert all(engine["online_moves"][s]["enabled"] is False for s in config_check.ONLINE_SOURCES)
    assert all(engine["lichess_bot_tbs"][t]["enabled"] is False for t in config_check.LOCAL_TABLEBASES)
    assert engine["draw_or_resign"]["resign_for_egtb_minus_two"] is False
    assert engine["draw_or_resign"]["offer_draw_for_egtb_zero"] is False
    for section in ("engine", "correspondence"):
        assert (config[section]["ponder"], config[section]["uci_ponder"]) == (False, False)
    assert not any("syzygy" in option.lower() for option in engine["uci_options"])


@pytest.mark.parametrize(
    ("path", "value", "problem"),
    [
        (("engine", "polyglot", "enabled"), True, "engine.polyglot.enabled"),
        (
            ("engine", "online_moves", "online_egtb", "enabled"),
            True,
            "engine.online_moves.online_egtb.enabled",
        ),
        (("engine", "lichess_bot_tbs", "gaviota", "enabled"), True, "engine.lichess_bot_tbs.gaviota.enabled"),
        (
            ("engine", "draw_or_resign", "resign_for_egtb_minus_two"),
            True,
            "engine.draw_or_resign.resign_for_egtb_minus_two",
        ),
        (("correspondence", "uci_ponder"), True, "correspondence.uci_ponder"),
        (("engine", "uci_options", "SyzygyPath"), "./syzygy/", "engine.uci_options.SyzygyPath"),
    ],
)
def test_each_lookup_switched_on_is_named(path, value, problem):
    config = copy.deepcopy(load_template())
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert problem in config_check.lookup_problems(config)


def test_a_missing_switch_counts_as_on_because_lichess_bot_defaults_some_to_true():
    config = copy.deepcopy(load_template())
    del config["engine"]["draw_or_resign"]["offer_draw_for_egtb_zero"]
    assert config_check.lookup_problems(config) == ["engine.draw_or_resign.offer_draw_for_egtb_zero"]


def test_the_template_never_holds_a_token():
    config = load_template()
    assert "token" not in config
    assert "lip_" not in TEMPLATE.read_text(encoding="utf-8")
