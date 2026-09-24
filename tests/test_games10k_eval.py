"""games10k top-1 (arm a07's metric): the policy's legal-masked agreement with Stockfish on real games."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_model_config  # noqa: E402

from blink.board import encode, moves  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train import games10k_eval  # noqa: E402
from blink.train.telemetry import legal_mask_from_codes  # noqa: E402

pytestmark = pytest.mark.torch


class HighestIndexNet(torch.nn.Module):
    """A stand-in network whose policy prefers the highest vocabulary index (1879 is rarely legal)."""

    def forward(self, tokens):
        policy = torch.arange(moves.NUM_MOVES, dtype=torch.float32).expand(len(tokens), -1)
        return policy, torch.zeros(len(tokens), 128)


def _legal(records: np.ndarray) -> np.ndarray:
    return np.stack([legal_mask_from_codes(codes) for codes in encode.unpack(records["board"])])


def _records_the_net_gets_right(k: int) -> np.ndarray:
    """Fixture positions labelled so the stand-in's legal argmax matches the first k and misses the rest."""
    records = fixture_records()[:10].copy()
    legal = _legal(records)
    highest = np.array([np.flatnonzero(row)[-1] for row in legal])
    lowest = np.array([np.flatnonzero(row)[0] for row in legal])
    assert np.all(highest != lowest)
    records["move"] = np.where(np.arange(len(records)) < k, highest, lowest)
    return records


def test_games10k_top1_is_the_share_whose_legal_argmax_is_stockfish_s_move():
    games = games10k_eval.from_records(_records_the_net_gets_right(k=4))
    result = games10k_eval.top1(HighestIndexNet(), games, torch.device("cpu"), chunk=3)
    assert result == {"top1": pytest.approx(0.4), "n": 10}  # unmasked, the argmax would be 1879 everywhere


def test_games10k_loads_the_records_the_labeller_saves_and_names_a_wrong_file(tmp_path):
    records = _records_the_net_gets_right(k=10)
    np.save(tmp_path / "games10k.npy", records)
    games = games10k_eval.load(tmp_path / "games10k.npy")
    assert games.n == 10 and np.array_equal(games.best, records["move"])
    assert np.array_equal(games.legal, _legal(records)) and games.board.shape == (10, 32)
    np.save(tmp_path / "plain.npy", np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match="board"):
        games10k_eval.load(tmp_path / "plain.npy")


def test_a_real_model_scores_games10k_in_bounded_chunks_and_keeps_its_mode():
    model = BlinkNet(tiny_model_config()).train()
    games = games10k_eval.from_records(fixture_records()[:10])
    result = games10k_eval.top1(model, games, torch.device("cpu"), chunk=4)
    assert result["n"] == 10 and 0.0 <= result["top1"] <= 1.0
    assert model.training  # scoring switched to eval mode and back
