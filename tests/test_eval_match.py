"""In-process matches: sequential openings once per colour, draws by rule, 600-ply adjudication, PGN out."""

import re
from dataclasses import replace

import chess
import chess.pgn
import pytest

from blink.eval import books, match
from blink.play import agents
from blink.play.budget import NoSearchViolation
from blink.play.oracles import RandomLogitEvaluator

START = chess.STARTING_FEN
OPEN_E4 = books.Opening(1, START, ("e2e4", "e7e5"))
OPEN_D4 = books.Opening(2, START, ("d2d4", "d7d5"))
NODES = re.compile(r"n=(\d+)")


def read_games(path) -> list[chess.pgn.Game]:
    games = []
    with open(path, encoding="utf-8") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            games.append(game)
    return games


class Cheater:
    name = "Cheater"

    def choose(self, board, remaining_s=None, game=""):
        return agents.Decision(chess.Move.from_uci("e2e5"))


class Crasher:
    name = "Crasher"

    def choose(self, board, remaining_s=None, game=""):
        raise RuntimeError("out of memory")


class Violator:
    name = "Violator"

    def choose(self, board, remaining_s=None, game=""):
        raise NoSearchViolation("a second network call in one decision")


def test_a_match_plays_each_opening_once_per_colour(tmp_path):
    pgn = tmp_path / "m.pgn"
    summary = match.run_match(
        agents.RandomAgent(seed=1), agents.MaterialAgent(seed=2), [OPEN_E4, OPEN_D4], games=4, pgn_path=pgn
    )
    games = read_games(pgn)
    assert [g.headers["White"] for g in games] == ["Random", "Material", "Random", "Material"]
    first_moves = [[m.uci() for m in g.mainline_moves()][:2] for g in games]
    assert first_moves == [["e2e4", "e7e5"]] * 2 + [["d2d4", "d7d5"]] * 2
    assert all(node.comment == "book" for g in games for node in list(g.mainline())[:2])
    assert summary["games"] == 4
    assert summary["a_wins"] + summary["draws"] + summary["a_losses"] == 4
    assert (summary["illegal_moves"], summary["crashes"]) == (0, 0)


def test_the_match_pgn_carries_node_counts_in_move_comments(tmp_path):
    blink = replace(agents.PolicyAgent(RandomLogitEvaluator()), name="Blink-policy-random")
    pgn = tmp_path / "nodes.pgn"
    match.run_match(blink, agents.RandomAgent(), [OPEN_E4], games=2, pgn_path=pgn, max_plies=40)
    counted = 0
    for game in read_games(pgn):
        for node in game.mainline():
            mover = (
                game.headers["White"] if node.parent.board().turn == chess.WHITE else game.headers["Black"]
            )
            if node.comment == "book" or mover != blink.name:
                continue
            counted += 1
            assert NODES.search(node.comment).group(1) in {"0", "1"}
    assert counted >= 20


def test_600_engine_plies_are_adjudicated_a_draw():
    game, record = match.play_game(
        agents.RandomAgent(), agents.RandomAgent(seed=5), OPEN_E4, "g1", max_plies=10
    )
    assert (record.result, record.termination, record.engine_plies) == ("1/2-1/2", "adjudication", 10)
    assert game.headers["PlyCount"] == "12"
    assert match.MAX_ENGINE_PLIES == 600


def test_draws_by_rule_end_the_game_before_any_engine_move():
    bare_kings = books.Opening(1, "8/8/8/4k3/8/8/8/4K3 w - - 0 1", ())
    _, record = match.play_game(agents.RandomAgent(), agents.RandomAgent(), bare_kings, "g1")
    assert (record.result, record.reason, record.engine_plies) == ("1/2-1/2", "insufficient material", 0)
    shuffle = books.Opening(1, START, ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"))
    _, record = match.play_game(agents.RandomAgent(), agents.RandomAgent(), shuffle, "g2")
    assert (record.result, record.reason) == ("1/2-1/2", "threefold repetition")


def test_a_checkmate_ends_the_game_for_the_mating_side():
    fools = books.Opening(1, START, ("f2f3", "e7e5", "g2g4"))
    blink = agents.PolicyAgent(RandomLogitEvaluator())
    game, record = match.play_game(agents.RandomAgent(), blink, fools, "g1")
    assert (record.result, record.reason, record.engine_plies) == ("0-1", "checkmate", 1)
    assert game.end().comment.endswith("n=0")


def test_an_illegal_move_loses_the_game():
    game, record = match.play_game(Cheater(), agents.RandomAgent(), books.Opening(1, START, ()), "g1")
    assert (record.result, record.termination, record.illegal_by) == ("0-1", "illegal move", "Cheater")
    assert game.headers["Termination"] == "illegal move"


def test_a_crashing_agent_loses_and_the_match_goes_on(tmp_path):
    summary = match.run_match(
        Crasher(), agents.RandomAgent(), [OPEN_E4], games=2, pgn_path=tmp_path / "c.pgn"
    )
    assert (summary["crashes"], summary["a_losses"]) == (2, 2)
    assert "out of memory" in (tmp_path / "c.pgn").read_text(encoding="utf-8")


def test_a_no_search_violation_is_never_swallowed(tmp_path):
    with pytest.raises(NoSearchViolation):
        match.run_match(Violator(), agents.RandomAgent(), [OPEN_E4], games=1, pgn_path=tmp_path / "v.pgn")


def test_the_pgn_is_written_game_by_game(tmp_path):
    pgn = tmp_path / "grow.pgn"
    match.run_match(
        agents.RandomAgent(), agents.MaterialAgent(), [OPEN_E4], games=2, pgn_path=pgn, max_plies=6
    )
    assert pgn.read_text(encoding="utf-8").count("[Event ") == 2
    assert len(read_games(pgn)) == 2


def write_book(path, games: list[str]) -> None:
    path.write_text("\n\n".join(f'[Event "?"]\n[Result "*"]\n\n{moves} *' for moves in games) + "\n", "utf-8")


def test_books_read_openings_front_to_back_from_a_start_index(tmp_path):
    book = tmp_path / "book.pgn"
    write_book(book, ["1. e4 e5", "1. d4 d5", "1. c4 c5", "1. Nf3 Nf6", "1. g3 g6"])
    openings = books.read_openings(book, start=2, count=2)
    assert [(o.number, o.moves) for o in openings] == [(2, ("d2d4", "d7d5")), (3, ("c2c4", "c7c5"))]
    assert len(books.read_openings(book, start=4)) == 2


def test_dev_and_final_book_slices_are_disjoint():
    dev, final = books.SLICES["dev"], books.SLICES["final"]
    assert dev == (1, 10_000)
    assert final == (10_001, 34_700)
    assert dev[1] < final[0]


def test_a_request_beyond_the_slice_is_refused(tmp_path):
    with pytest.raises(ValueError, match="dev slice"):
        books.openings_for("dev", 10_001)
