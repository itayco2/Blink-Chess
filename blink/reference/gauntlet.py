"""DM-9M in `blink gauntlet` (plan E7): fastchess's gauntlet with DeepMind's engine under its own name.

blink.eval.fastchess builds the engine under test as Blink (`Blink-<mode>-<model>`, with `--mode=`) and
audits the players whose name contains "blink". For a dm selector that would file DeepMind's games and
moves under a Blink name, and its no-search audit would see no decisions at all. This module keeps
fastchess's gauntlet (anchor, book slice, time control, PGN folder) and changes only the engine:

    DM-9M[-ema]   python -m blink.uci --model=dm:9M[:ema] --device=<d>   (no --mode: it has one)

with the PGN named after it (`DM-9M_vs_SF1320_<time>.pgn`), and audits DM-9M's own moves: one row per
legal move, L rows per decision, inside the no-search bound of L+1.
"""

from dataclasses import replace

from blink.eval import fastchess, nosearch
from blink.play.factory import ModelUnavailable
from blink.reference import registry


def engine_name(selector: str) -> str:
    """DM-9M or DM-9M-ema; a malformed selector is refused like a missing model, in one line."""
    try:
        return registry.parse(selector).name
    except ValueError as exc:
        raise ModelUnavailable(str(exc)) from exc


def dm_engine(blink_spec: fastchess.EngineSpec, selector: str, device: str) -> fastchess.EngineSpec:
    """The same UCI process, time control and margin as Blink, under DeepMind's name and selector."""
    args = ("-m", "blink.uci", f"--model={selector}", f"--device={device}")
    return replace(blink_spec, name=engine_name(selector), args=args)


def prepare_gauntlet(model: str, device: str, **kwargs) -> fastchess.Gauntlet:
    """fastchess.prepare_gauntlet for a dm selector, with DeepMind's engine and a PGN named after it."""
    name = engine_name(model)
    base = fastchess.prepare_gauntlet(model=model, device=device, **kwargs)
    pgn = base.plan.pgn_out
    renamed = pgn.with_name(name + pgn.name.removeprefix(base.blink.name))
    return replace(base, blink=dm_engine(base.blink, model, device), plan=replace(base.plan, pgn_out=renamed))


def execute(gauntlet: fastchess.Gauntlet) -> dict:
    """fastchess.execute, then the no-search audit of the engine under test's own moves.

    fastchess.execute audits players named "blink", which finds no DeepMind move, so the audit and its
    <pgn>.nosearch.json are redone for gauntlet.blink.name. Forfeits come from the game endings, which the
    player filter does not touch, so blink_forfeits stands as fastchess reported it.
    """
    report = fastchess.execute(gauntlet)
    pgn = gauntlet.plan.pgn_out
    audit = nosearch.audit([pgn] if pgn.is_file() else [], engine=gauntlet.blink.name)
    nosearch.write_report(audit, pgn.with_suffix(".nosearch.json"))
    return {**report, "audit": audit}
