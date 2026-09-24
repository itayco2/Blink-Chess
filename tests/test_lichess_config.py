"""The Lichess bot config must switch off every lookup lichess-bot can make for the engine (N3)."""

import copy
import json
from pathlib import Path

import pytest

from blink.lichess import config_check

TEMPLATE = Path(__file__).resolve().parent.parent / "deploy" / "lichess" / "config.template.yml"


def load_template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


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
