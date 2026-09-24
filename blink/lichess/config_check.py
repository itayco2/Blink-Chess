"""Checks that a lichess-bot config lets the engine alone choose every move (NSC-1, N3).

lichess-bot can answer a move itself from a polyglot book, online sources (chessdb, the Lichess
cloud analysis, the opening explorer, online tablebases) or local Syzygy/Gaviota tablebases, can
resign or offer draws on tablebase results, and can ponder. Every one of those switches must be
present and false: lichess-bot's own defaults turn some of them on (resign_for_egtb_minus_two and
offer_draw_for_egtb_zero default to true in its sample config), so a missing key counts as on.
The engine must also not be handed a SyzygyPath UCI option.
"""

from collections.abc import Mapping

ONLINE_SOURCES = ("chessdb_book", "lichess_cloud_analysis", "lichess_opening_explorer", "online_egtb")
LOCAL_TABLEBASES = ("syzygy", "gaviota")
EGTB_ACTIONS = ("resign_for_egtb_minus_two", "offer_draw_for_egtb_zero")
PONDER_SECTIONS = ("engine", "correspondence")


def _switch_off(config: Mapping, path: tuple[str, ...]) -> bool:
    node: object = config
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return False
        node = node[key]
    return node is False


def required_off_switches() -> list[tuple[str, ...]]:
    paths = [("engine", "polyglot", "enabled")]
    paths += [("engine", "online_moves", source, "enabled") for source in ONLINE_SOURCES]
    paths += [("engine", "lichess_bot_tbs", tb, "enabled") for tb in LOCAL_TABLEBASES]
    paths += [("engine", "draw_or_resign", action) for action in EGTB_ACTIONS]
    paths += [(section, key) for section in PONDER_SECTIONS for key in ("ponder", "uci_ponder")]
    return paths


def lookup_problems(config: Mapping) -> list[str]:
    """Dotted names of every lookup that is on or not explicitly off; empty when the config is clean."""
    problems = [".".join(path) for path in required_off_switches() if not _switch_off(config, path)]
    options = config.get("engine", {}).get("uci_options") or {}
    problems += [f"engine.uci_options.{name}" for name in options if "syzygy" in str(name).lower()]
    return problems
