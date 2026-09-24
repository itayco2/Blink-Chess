"""fastchess gauntlets against Stockfish 19 anchors, with per-engine time controls (plan P8 match rules)."""

import sys
from pathlib import Path

import pytest

from blink.eval import fastchess

FASTCHESS_OUTPUT = """Failed to get console mode. Error code: 6
Started game 1 of 2 (Blink-policy-random vs SF1320)
\x1b[1;91mFinished game 1\x1b[0m (Blink-policy-random vs SF1320): 0-1 {Black mates}
--------------------------------------------------
Results of Blink-policy-random vs SF1320 (1/move - 0.1/move, NULL - 1t, NULL - 16MB, 8moves_v3.pgn):
Elo: -inf +/- nan, nElo: -inf +/- nan
LOS: 0.00 %, DrawRatio: 0.00 %, PairsRatio: 0.00
Games: 2, Wins: 0, Losses: 1, Draws: 1, Points: 0.5 (25.00 %)
Ptnml(0-2): [1, 0, 0, 0, 0], WL/DD Ratio: -nan
--------------------------------------------------
Finished match
"""


def plan(tmp_path: Path, games: int = 20) -> fastchess.GauntletPlan:
    return fastchess.GauntletPlan(
        games=games, book=Path("book.pgn"), book_start=1, concurrency=2, pgn_out=tmp_path / "g.pgn"
    )


def option_values(tokens: list[str]) -> dict[str, str]:
    return dict(token.split("=", 1) for token in tokens if "=" in token)


def engine_blocks(command: list[str]) -> list[list[str]]:
    starts = [i for i, token in enumerate(command) if token == "-engine"]
    ends = [*starts[1:], next(i for i, t in enumerate(command) if t.startswith("-") and i > starts[-1] + 1)]
    return [command[s + 1 : e] for s, e in zip(starts, ends, strict=True)]


def test_the_gauntlet_command_uses_per_engine_time_controls(tmp_path):
    blink = fastchess.blink_engine("run:skeleton", mode="policy", device="cuda")
    anchor = fastchess.stockfish_anchor(1320, Path("sf.exe"))
    command = fastchess.build_command(Path("fastchess.exe"), blink, anchor, plan(tmp_path))
    blink_opts, sf_opts = (option_values(block) for block in engine_blocks(command))
    assert (blink_opts["st"], blink_opts["timemargin"]) == ("1", "500")
    assert (sf_opts["st"], sf_opts["timemargin"]) == ("0.1", "100")
    assert sf_opts["option.UCI_LimitStrength"] == "true"
    assert sf_opts["option.UCI_Elo"] == "1320"
    assert (sf_opts["option.Threads"], sf_opts["option.Hash"]) == ("1", "16")
    assert blink_opts["args"] == "-m blink.uci --model=run:skeleton --mode=policy --device=cuda"
    assert blink_opts["cmd"] == sys.executable
    tail = command[command.index("-openings") :]
    assert tail[:5] == ["-openings", "file=book.pgn", "format=pgn", "order=sequential", "start=1"]
    for flag, value in (("-rounds", "10"), ("-maxmoves", "300"), ("-concurrency", "2")):
        assert command[command.index(flag) + 1] == value
    assert "-repeat" in command and "-recover" in command
    pgnout = command.index("-pgnout")
    assert command[pgnout + 1 : pgnout + 3] == [f"file={tmp_path / 'g.pgn'}", "nodes=true"]
    assert "-resign" not in command and "-draw" not in command


def test_the_gauntlet_gives_engines_a_minute_to_start():
    """PF55: fastchess waits 10 s for uciok/readyok by default, and a loaded machine starting several
    torch+CUDA engines at once needed longer, so 3 of 20 skeleton games were scored as crashes."""
    blink = fastchess.blink_engine("run:skeleton", mode="policy", device="cuda")
    anchor = fastchess.stockfish_anchor(1320, Path("sf.exe"))
    command = fastchess.build_command(Path("fastchess.exe"), blink, anchor, plan(Path("out")))
    assert int(command[command.index("-startup-ms") + 1]) >= 60_000


def test_a_random_blink_is_the_uci_engine_with_the_random_flag():
    blink = fastchess.blink_engine("random", mode="policy", device="cpu")
    assert blink.args == ("-m", "blink.uci", "--random", "--mode=policy", "--device=cpu")
    assert blink.name == "Blink-policy-random"


def test_a_cutechess_time_control_replaces_both_movetimes(tmp_path):
    blink = fastchess.with_tc(fastchess.blink_engine("random", "policy", "cpu"), "10+0.1")
    tokens = option_values(blink.fastchess_args())
    assert tokens["tc"] == "10+0.1"
    assert "st" not in tokens


def test_odd_game_counts_are_refused(tmp_path):
    with pytest.raises(ValueError, match="even"):
        plan(tmp_path, games=3)


def test_the_fastchess_summary_is_parsed():
    assert fastchess.parse_summary(FASTCHESS_OUTPUT) == {
        "games": 2,
        "wins": 0,
        "losses": 1,
        "draws": 1,
        "points": 0.5,
        "elo": "Elo: -inf +/- nan, nElo: -inf +/- nan",
        "penta": [1, 0, 0, 0, 0],
    }
    assert fastchess.parse_summary("nothing here") is None


TOOLS_PRESENT = fastchess.fastchess_exe().is_file() and fastchess.stockfish_exe().is_file()
BOOK_PRESENT = fastchess.books.book_file().is_file()


@pytest.mark.local
@pytest.mark.skipif(
    not (TOOLS_PRESENT and BOOK_PRESENT), reason="fastchess, Stockfish or the book is missing"
)
def test_a_two_game_gauntlet_against_stockfish_is_clean(tmp_path):
    report = fastchess.run_gauntlet(
        model="random",
        mode="value",
        device="cpu",
        anchor=1320,
        games=2,
        book="dev",
        out_dir=tmp_path,
        concurrency=2,
        max_moves=15,
    )
    assert report["returncode"] == 0
    assert report["summary"]["games"] == 2
    assert report["audit"]["violations"] == []
    assert report["audit"]["decisions"] > 10
    assert report["blink_forfeits"] == {}


def test_fastchess_runs_in_the_output_folder_so_its_autosave_lands_there(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(fastchess.subprocess, "run", fake_run)
    assert fastchess.run_fastchess(["fastchess.exe"], tmp_path / "games" / "g.log") == 0
    assert seen["cwd"] == tmp_path / "games"
    assert seen["env"]["PYTHONPATH"].split(fastchess.os.pathsep)[0] == str(
        Path(fastchess.blink.__file__).parents[1]
    )


def test_blink_under_fastchess_plays_with_the_epsilon_it_is_given():
    """E2b's epsilon must reach blink-uci; without --epsilon it falls back to 0."""
    spec = fastchess.blink_engine("ship", "value", "cuda", epsilon=1 / 256)
    assert spec.args[-1] == "--epsilon=0.00390625"
    assert not any(a.startswith("--epsilon") for a in fastchess.blink_engine("ship", "value").args)
    dm = fastchess.blink_engine("dm:9M", "policy", "cuda", epsilon=1 / 256)
    assert not any(a.startswith("--epsilon") for a in dm.args)


def test_the_uci_engine_parses_the_epsilon_flag_exactly():
    from blink import uci

    spec = fastchess.blink_engine("ship", "value", "cpu", epsilon=1 / 128)
    args = uci.build_parser().parse_args(list(spec.args[2:]))
    assert args.epsilon == 1 / 128
