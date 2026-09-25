"""PR-4's looks at the endgame screen (EVAL.md section 5): the Poisson declaration rule and its lines."""

import math

import chess
import pytest

from blink.eval import endgame_looks as looks
from blink.eval import endgames, sflabel

FILE_POSITIONS = 157_824  # endgames.epd's unique positions (22 repeats among its 157,846 lines)


def test_the_upper_bound_is_the_exact_one_sided_95_percent_poisson_bound():
    assert looks.poisson_upper(0) == pytest.approx(-math.log(0.05), rel=1e-9)
    for kept in (1, 14, 73, 700):
        mu = looks.poisson_upper(kept)
        cdf = sum(math.exp(-mu + i * math.log(mu) - math.lgamma(i + 1)) for i in range(kept + 1))
        assert cdf == pytest.approx(0.05, rel=1e-6)


@pytest.mark.parametrize(
    ("screened", "last_declaring"), [(1_000, 0), (5_000, 14), (20_000, 73), (19_999, 73)]
)
def test_pr4_declares_at_kept_0_14_and_73_at_the_first_three_looks(screened, last_declaring):
    """Through look_at, which the screen's LookTracker calls at each look line."""
    plan = looks.LookPlan(FILE_POSITIONS, 157_846)
    assert looks.look_at(plan, screened, screened, last_declaring, last_declaring).declares
    assert not looks.look_at(plan, screened, screened, last_declaring + 1, last_declaring + 1).declares


def test_nothing_screened_or_no_known_total_never_declares():
    assert not looks.look_at(looks.LookPlan(FILE_POSITIONS, 157_846), 1_000, 0, 0, 0).declares
    assert looks.look_at(looks.LookPlan(None, 10), 5, 5, 0, 0).upper95_total is None
    assert not hasattr(looks, "declares")  # one rule, the one the screen calls


def test_looks_fall_at_lines_1000_5000_and_20000_then_every_20000():
    assert list(looks.look_lines(157_846)) == [1_000, 5_000, *range(20_000, 157_846, 20_000)]
    assert list(looks.look_lines(4_999)) == [1_000]


def test_a_look_records_the_counts_the_projection_and_its_bound():
    plan = looks.LookPlan(FILE_POSITIONS, 157_846)
    look = looks.look_at(plan, 5_000, 5_000, 40, 14)
    assert (look.line, look.screened, look.passed_screen, look.kept) == (5_000, 5_000, 40, 14)
    assert look.projected_total == pytest.approx(14 * FILE_POSITIONS / 5_000)
    assert look.upper95_total == pytest.approx(looks.poisson_upper(14) * FILE_POSITIONS / 5_000)
    assert look.declares and "690.8 < 700" in looks.describe(look)


# ------------------------------------------------------------------------------ the screen takes its looks


def distinct_positions(count: int) -> list[str]:
    """`count` distinct legal king-and-rook against king positions, White to move."""
    found = []
    for rook in chess.SQUARES:
        for white_king in chess.SQUARES:
            for black_king in chess.SQUARES:
                if len({rook, white_king, black_king}) < 3:
                    continue
                board = chess.Board.empty()
                board.set_piece_at(rook, chess.Piece(chess.ROOK, chess.WHITE))
                board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
                if board.is_valid():
                    found.append(board.fen())
                    if len(found) == count:
                        return found
    raise AssertionError("not enough positions")


def labeler(tmp_path, name, won, calls=None):
    """A fake SF19: +9.00 for the side to move in the `won` FENs, +0.20 elsewhere."""

    def analyse(board, nodes, move):
        if calls is not None:
            calls.append((name, board.fen()))
        return sflabel.SfLabel(900 if board.fen() in won else 20, None, 20, None)

    return sflabel.SfLabeler(1, cache_path=tmp_path / f"{name}.jsonl", analyse=analyse)


def numbered(fens):
    return iter(enumerate(fens, start=1))


def test_a_look_counts_only_the_lines_up_to_it_whatever_the_batch(tmp_path):
    fens = distinct_positions(20)
    won = {fens[2], fens[11]}  # lines 3 and 12; one batch of 16 straddles the look at line 10
    plan = looks.LookPlan(None, 20, first=(10,), every=10)
    result = endgames.screen(
        numbered(fens), labeler(tmp_path, "s", won), labeler(tmp_path, "c", won), looks=plan
    )
    assert [(k.line, k.screened, k.kept) for k in result.looks] == [(10, 10, 1), (20, 20, 2)]
    assert result.declaration is None and result.screened == 20


def test_a_declaring_look_stops_the_screen_and_says_why(tmp_path):
    fens = distinct_positions(30)
    seen = []
    plan = looks.LookPlan(positions=100, last_line=30, first=(10,), every=100)
    result = endgames.screen(
        numbered(fens),
        labeler(tmp_path, "s", set()),
        labeler(tmp_path, "c", set()),
        looks=plan,
        on_look=seen.append,
    )
    assert result.screened == 10 and len(result.looks) == 1 and seen == list(result.looks)
    assert result.looks[0].declares and result.declaration.startswith("line 10: 0 kept of 10 screened")


def test_a_look_that_does_not_declare_lets_the_screen_run_on(tmp_path):
    fens = distinct_positions(30)
    plan = looks.LookPlan(positions=100, last_line=30, first=(10,), every=100, need=20)
    result = endgames.screen(
        numbered(fens), labeler(tmp_path, "s", set()), labeler(tmp_path, "c", set()), looks=plan
    )
    assert not result.looks[0].declares and result.screened == 30


def test_the_file_ending_short_of_700_kept_is_a_declaration_but_a_limit_is_not(tmp_path):
    fens = distinct_positions(12)
    won = {fens[0]}
    plan = looks.LookPlan(12, 12, first=(100,), every=100)
    ended = endgames.screen(
        numbered(fens), labeler(tmp_path, "s", won), labeler(tmp_path, "c", won), looks=plan
    )
    assert ended.declaration == "the file ended at line 12 with 1 kept (fewer than 700)"
    limited = endgames.screen(
        numbered(fens[:5]), labeler(tmp_path, "s", won), labeler(tmp_path, "c", won), looks=plan
    )
    assert limited.declaration is None and len(limited.kept) == 1


def test_a_look_at_a_repeated_line_is_taken_before_the_next_position(tmp_path):
    fens = distinct_positions(12)
    fens[9] = fens[1].replace(" 0 1", " 0 40")  # line 10 repeats line 2 with other counters
    won = {fens[0]}
    plan = looks.LookPlan(None, 12, first=(10,), every=100)
    result = endgames.screen(
        numbered(fens), labeler(tmp_path, "s", won), labeler(tmp_path, "c", won), looks=plan
    )
    assert [(k.line, k.screened, k.kept) for k in result.looks] == [(10, 9, 1)]
    assert result.repeats_skipped == 1


def test_a_plan_without_a_file_total_only_records_its_looks(tmp_path):
    fens = distinct_positions(12)
    plan = looks.LookPlan(None, 12, first=(10,), every=100)
    result = endgames.screen(
        numbered(fens), labeler(tmp_path, "s", set()), labeler(tmp_path, "c", set()), looks=plan
    )
    assert result.declaration is None and result.screened == 12 and not result.looks[0].declares


def test_the_screen_without_looks_is_unchanged(tmp_path):
    fens = distinct_positions(5)
    result = endgames.screen(
        numbered(fens), labeler(tmp_path, "s", set(fens)), labeler(tmp_path, "c", set(fens))
    )
    assert len(result.kept) == 5 and result.looks == () and result.declaration is None
