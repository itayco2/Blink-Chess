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


def write_named(tmp_path, name, *games):
    path = tmp_path / name
    path.write_text("\n\n".join(games) + "\n", encoding="utf-8")
    return path


def test_failures_are_taken_in_turn_from_each_file_not_all_from_the_first(tmp_path):
    first = write_named(tmp_path, "a.pgn", *[game_pgn(SHUFFLE, black="SF1800")] * 3)
    second = write_named(tmp_path, "b.pgn", *[game_pgn(SHUFFLE, black="SF1900")] * 3)
    out = failures.run_failures([first, second], labeler(tmp_path), max_failures=2)
    assert [(f["file"], f["game"]) for f in out["failures"]] == [("a.pgn", 1), ("b.pgn", 1)]
    assert out["examined_games"] == 2


def test_an_exact_player_never_picks_up_a_name_that_contains_it(tmp_path):
    games = [
        game_pgn(SHUFFLE, white="Blink-value-ship-rules-off"),
        game_pgn(SHUFFLE, white="Blink-value-ship"),
    ]
    pgn = write_named(tmp_path, "g.pgn", *games)
    out = failures.run_failures([pgn], labeler(tmp_path), player="Blink-value-ship", exact=True)
    assert [(f["blink"], f["game"]) for f in out["failures"]] == [("Blink-value-ship", 2)]


def e9_context(tmp_path, **kwargs):
    from blink.eval import orchestrate

    base = {"model": "ship", "out_dir": tmp_path / "out", "mode": "value"}
    return orchestrate.EvalContext(**{**base, **kwargs})


def e5_report(tmp_path):
    pgns = {}
    for mode in ("policy", "value"):
        for anchor in ("SF1800", "SF1900"):
            pgns[mode, anchor] = write_named(tmp_path, f"{mode}_{anchor}.pgn", game_pgn(SHUFFLE))
    final = {
        mode: {
            "locator": {"pgn": str(tmp_path / f"{mode}_locator.pgn")},
            "anchors": [{"pgn": str(pgns[mode, a])} for a in ("SF1800", "SF1900")],
        }
        for mode in ("policy", "value")
    }
    return {"final": final, "side": {}}, pgns


def fake_e9(monkeypatch):
    seen = {}

    class Labeler:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def run_failures(pgns, labeler, player="blink", max_failures=50, max_examined=None, exact=False):
        seen.update(pgns=list(pgns), player=player, exact=exact)
        return {"examined_games": 0, "failures": [], "classes": {}, "positions_searched": 0}

    monkeypatch.setattr(sflabel, "SfLabeler", Labeler)
    monkeypatch.setattr(failures, "run_failures", run_failures)
    return seen


def test_e9_reads_only_the_shipped_engines_final_slice_anchor_games(tmp_path, monkeypatch):
    seen = fake_e9(monkeypatch)
    e5, pgns = e5_report(tmp_path)
    out = failures.e9_block(e9_context(tmp_path), {"E5": e5})
    assert seen == {
        "pgns": [pgns["value", "SF1800"], pgns["value", "SF1900"]],
        "player": "Blink-value-ship",
        "exact": True,
    }
    assert out["player"] == "Blink-value-ship" and len(out["source_pgns"]) == 2


def test_e9_run_alone_reads_e5_from_the_same_out_folder_and_refuses_without_it(tmp_path, monkeypatch):
    import json

    import pytest

    seen = fake_e9(monkeypatch)
    ctx = e9_context(tmp_path, mode="policy")
    with pytest.raises(ValueError, match="E5"):
        failures.e9_block(ctx, {})
    e5, pgns = e5_report(tmp_path)
    ctx.out_dir.mkdir(parents=True)
    (ctx.out_dir / "E5.json").write_text(json.dumps(e5), encoding="utf-8")
    failures.e9_block(ctx, {})
    assert seen["pgns"] == [pgns["policy", "SF1800"], pgns["policy", "SF1900"]]
    pgns["policy", "SF1900"].unlink()
    with pytest.raises(ValueError, match="policy_SF1900.pgn"):
        failures.e9_block(ctx, {})
