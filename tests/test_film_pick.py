"""`blink film pick`: the story score over the 200 film candidates, short-listing 5 for Itay (G9)."""

import json

import chess
import numpy as np
import pytest

from blink.board import moves
from blink.film import extract, pick
from blink.play.evaluator import Evaluation

HEADER = "PuzzleId,FEN,Moves,Rating,RatingDeviation,NbPlays,Themes\n"
START = chess.STARTING_FEN


def _csv(tmp_path, n: int):
    rows = "".join(f"p{i:04d},{START},e2e4 e7e5,{1000 + i},70,2000,opening\n" for i in range(n))
    path = tmp_path / "bands.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def _candidate(puzzle_id="abc", solution="e7e5"):
    return extract.position_from_row(
        {"PuzzleId": puzzle_id, "FEN": START, "Moves": f"e2e4 {solution}", "Rating": "1500", "Themes": ""}
    )


def test_the_200_candidates_are_a_fixed_hash_draw_from_the_band_puzzles(tmp_path):
    path = _csv(tmp_path, 300)
    drawn = pick.draw_candidates(path, 200)
    assert len(drawn) == 200 and len({c.puzzle_id for c in drawn}) == 200
    assert [c.puzzle_id for c in drawn] == [c.puzzle_id for c in pick.draw_candidates(path, 200)]
    assert [c.puzzle_id for c in drawn] != sorted(c.puzzle_id for c in drawn)  # hashed, not file order
    assert len(pick.draw_candidates(_csv(tmp_path, 50), 200)) == 50


def test_the_story_score_rewards_a_late_correct_find_with_a_win_swing():
    candidate = _candidate()
    found = pick.story(candidate, ("d7d5", "g8f6", "e7e5", "e7e5"), (0.50, 0.40, 0.70, 0.90))
    assert found.final_correct and found.changes == 2 and found.swing == pytest.approx(0.5)
    assert found.score == pytest.approx(pick.W_FINAL + pick.W_CHANGES * 2 / pick.MAX_CHANGES + 0.5)
    missed = pick.story(candidate, ("e7e5", "e7e5", "e7e5", "d7d5"), (0.5, 0.5, 0.5, 0.5))
    assert not missed.final_correct and missed.score < found.score


def test_top1_changes_are_capped_so_a_flickering_position_cannot_win_on_noise():
    candidate = _candidate()
    flicker = pick.story(candidate, ("a7a6", "b7b6") * 5 + ("e7e5",), (0.5,) * 11)
    assert flicker.changes == 10
    assert flicker.score == pytest.approx(pick.W_FINAL + pick.W_CHANGES)


class _FakeEvaluator:
    """Frame f always puts all its policy mass on the f-th legal move (sorted by vocabulary index)."""

    def __init__(self, frame: int) -> None:
        self.frame = frame
        self.calls = 0

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        self.calls += 1
        n = len(codes)
        logits = np.zeros((n, moves.NUM_MOVES), dtype=np.float32)
        board = chess.Board()
        board.push_uci("e2e4")
        legal = np.flatnonzero(moves.legal_mask(board))
        logits[:, legal[self.frame % len(legal)]] = 20.0
        bins = np.zeros((n, 128), dtype=np.float32)
        bins[:, min(127, 40 + 30 * self.frame)] = 1.0
        return Evaluation(policy_logits=logits, value_probs=bins)


def test_rank_scores_every_candidate_on_every_frame_with_one_call_per_frame(tmp_path):
    candidates = [_candidate(f"c{i}") for i in range(3)]
    sources = [
        extract.FrameSource(step, "ema", "film", tmp_path / f"{step}.pt", "model") for step in (0, 1, 2)
    ]
    made = []

    def loader(source, device):
        made.append(_FakeEvaluator(source.step))
        return made[-1], {"world": "w", "kind": "ema", "samples": 0}

    ranked = pick.rank(candidates, sources, device="cpu", loader=loader)
    assert [e.calls for e in made] == [1, 1, 1]
    assert len(ranked) == 3 and all(len(s.top1) == 3 for s in ranked)
    assert [s.candidate.puzzle_id for s in ranked] == ["c0", "c1", "c2"]  # equal scores: by puzzle id
    assert ranked[0].changes == 2 and ranked[0].swing > 0.4


def test_the_short_list_prints_and_writes_the_top_n(tmp_path):
    stories = [pick.story(_candidate(f"c{i}"), ("d7d5", "e7e5"), (0.4, 0.4 + i / 10)) for i in range(7)]
    ranked = sorted(stories, key=pick.sort_key)
    text = pick.format_top(ranked, 5)
    assert text.count("\n") == 5 and text.splitlines()[1].split()[1] == "c6"
    out = pick.write_ranking(ranked, tmp_path / "candidates.json", "long", 21)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run"] == "long" and data["frames"] == 21 and len(data["ranked"]) == 7
    assert data["ranked"][0]["puzzle_id"] == "c6" and "score" in data["ranked"][0]
