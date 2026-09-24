"""E9: losses and draws Blink should have won, grouped into named failure classes."""

import io

import chess
import chess.pgn

from blink.eval import failures, sflabel


def game_pgn(moves, white="Blink-value", black="SF1800", result="1/2-1/2", termination="normal", fen=None):
    board = chess.Board(fen) if fen else chess.Board()
    game = chess.pgn.Game()
    if fen:
        game.setup(board)
    node = game
    for uci in moves:
        node = node.add_variation(chess.Move.from_uci(uci))
    game.headers.update({"White": white, "Black": black, "Result": result, "Termination": termination})
    return str(game)


def labeler(tmp_path, pawns_by_ply=None, mate_at=None, default=5.0):
    def analyse(board, nodes, move):
        ply = board.ply()
        if mate_at is not None and ply == mate_at:
            return sflabel.SfLabel(None, 2, 20, None)
        pawns = (pawns_by_ply or {}).get(ply, default)
        return sflabel.SfLabel(int(pawns * 100), None, 20, None)

    return sflabel.SfLabeler(1, cache_path=tmp_path / "c.jsonl", analyse=analyse)


SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]


def write(tmp_path, *games):
    path = tmp_path / "g.pgn"
    path.write_text("\n\n".join(games) + "\n", encoding="utf-8")
    return path


def test_a_draw_by_repetition_while_plus_5_is_a_repetition_failure(tmp_path):
    pgn = write(tmp_path, game_pgn(SHUFFLE))
    out = failures.run_failures([pgn], labeler(tmp_path))
    assert out["classes"] == {"repetition while winning": 1}
    (failure,) = out["failures"]
    assert (
        failure["result"] == "draw" and failure["ending"] == "threefold repetition" and failure["peak"] == 5.0
    )


def test_a_draw_never_better_than_plus_3_is_not_a_failure(tmp_path):
    pgn = write(tmp_path, game_pgn(SHUFFLE))
    assert failures.run_failures([pgn], labeler(tmp_path, default=1.0))["failures"] == []


def test_a_loss_after_one_bad_move_is_a_one_move_blunder(tmp_path):
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    pgn = write(tmp_path, game_pgn(moves, result="0-1"))
    out = failures.run_failures([pgn], labeler(tmp_path, {0: 4.0, 2: -2.0}))
    assert out["classes"] == {"one-move blunder": 1}
    assert out["failures"][0]["worst_drop"] == 6.0


def test_a_loss_with_a_mate_on_the_board_is_a_missed_forced_mate(tmp_path):
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    pgn = write(tmp_path, game_pgn(moves, result="0-1"))
    out = failures.run_failures([pgn], labeler(tmp_path, {2: 0.0}, mate_at=0))
    assert out["classes"] == {"missed a forced mate": 1}


def test_forfeits_and_adjudications_get_their_own_classes(tmp_path):
    lost_on_time = game_pgn(
        ["e2e4", "e7e5"], white="SF1800", black="Blink-policy", result="1-0", termination="time forfeit"
    )
    adjudicated = game_pgn(["e2e4", "e7e5", "g1f3"], result="1/2-1/2", termination="adjudication")
    out = failures.run_failures([write(tmp_path, lost_on_time, adjudicated)], labeler(tmp_path))
    assert out["classes"] == {"time forfeit or crash": 1, "no progress: 600-ply adjudication": 1}


def test_wins_and_games_without_blink_are_skipped_and_the_cap_holds(tmp_path):
    won = game_pgn(["f2f3", "e7e5", "g2g4", "d8h4"], white="SF1800", black="Blink-value", result="0-1")
    stranger = game_pgn(SHUFFLE, white="SF1320", black="SF1400")
    drawn = game_pgn(SHUFFLE)
    out = failures.run_failures(
        [write(tmp_path, won, stranger, drawn, drawn)], labeler(tmp_path), max_failures=1
    )
    assert out["examined_games"] == 3 and len(out["failures"]) == 1


def test_book_moves_are_not_blinks_turns(tmp_path):
    game = chess.pgn.read_game(io.StringIO(game_pgn(["e2e4", "e7e5", "g1f3"])))
    game.variations[0].comment = "book"
    turns = failures.blink_turns(game, chess.WHITE, labeler(tmp_path))
    assert [t.ply for t in turns] == [2]
