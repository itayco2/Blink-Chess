"""film.json: every frame of a run, run once on one never-seen puzzle position (plan P11)."""

import json

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from blink.board import encode  # noqa: E402
from blink.data import blocklist  # noqa: E402
from blink.film import extract  # noqa: E402
from blink.model.config import ModelConfig, TrainConfig, config_to_dict  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train import checkpoint, film  # noqa: E402

pytestmark = pytest.mark.torch

# Two real Lichess band puzzles (CC0): the first move is the opponent's setup move.
BANDS_CSV = (
    "PuzzleId,FEN,Moves,Rating,RatingDeviation,NbPlays,Themes\n"
    "mX46Y,2kr3r/ppqnbpp1/4p3/1P1p2Np/N2P2n1/P6P/2PB1PP1/R2Q1RK1 w - - 2 16,g5f7 c7h2,581,80,1519,"
    "mate mateIn1 middlegame oneMove\n"
    "uRVPg,r2q1rk1/ppp2ppp/1n6/3nP1P1/4Q3/2N5/PP4P1/R1B1K2R b KQ - 0 16,d5c3 e4h7,576,79,1072,"
    "kingsideAttack mate mateIn1 middlegame oneMove\n"
)
TINY = TrainConfig(
    model=ModelConfig(d_model=32, n_layers=1, n_heads=1, head_dim=32, ffn_mult=1), batch_size=8, steps=1000
)
WORLD = "0123456789ab"


def _bands(tmp_path):
    path = tmp_path / "bands.csv"
    path.write_text(BANDS_CSV, encoding="utf-8")
    return path


def _weights(seed: int) -> dict:
    torch.manual_seed(seed)
    return BlinkNet(TINY.model).state_dict()


def _film_run(tmp_path, steps, world=WORLD, name="run"):
    run = tmp_path / "runs" / name
    for step in steps:
        kind = "init" if step == 0 else "ema"
        film.save_frame(run, step, kind, _weights(step), world, config_to_dict(TINY), samples=step * 8)
    return run


def _blocklist(tmp_path, position: extract.FilmPosition, blocked: bool = True):
    board = chess.Board(position.setup_fen)
    hashes = {encode.position_hash(board)}
    for uci in position.line:
        board.push_uci(uci)
        hashes.add(encode.position_hash(board))
    if not blocked:
        hashes = {h ^ 1 for h in hashes}
    path = tmp_path / "blocklist.npy"
    blocklist.save(hashes, path)
    return path


def _position(tmp_path, puzzle_id="mX46Y"):
    return extract.film_position(_bands(tmp_path), puzzle_id)


def test_the_film_position_is_the_solvers_board_after_the_setup_move(tmp_path):
    position = _position(tmp_path)
    assert position.setup_move == "g5f7" and position.solution == "c7h2"
    assert position.fen == "2kr3r/ppqnbNp1/4p3/1P1p3p/N2P2n1/P6P/2PB1PP1/R2Q1RK1 b - - 0 16"
    assert position.side_to_move == "black" and position.rating == 581
    with pytest.raises(extract.FilmError, match="nope"):
        extract.film_position(_bands(tmp_path), "nope")


def test_film_json_has_21_frames_with_top5_and_128_bins(tmp_path):
    steps = [step for step, _ in film.frame_plan(100_000)]
    run = _film_run(tmp_path, steps)
    position = _position(tmp_path)
    result = extract.extract(run, position, _blocklist(tmp_path, position))
    frames = result["frames"]
    assert len(frames) == 21 and [f["index"] for f in frames] == list(range(1, 22))
    assert [f["step"] for f in frames] == steps
    legal = chess.Board(position.fen).legal_moves.count()
    for frame in frames:
        assert len(frame["top5"]) == 5 and not frame["interpolated"]
        probs = [m["p"] for m in frame["top5"]]
        assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0 + 1e-6
        assert len(frame["legal"]) == legal and sum(frame["legal"].values()) == pytest.approx(1.0, abs=1e-5)
        assert len(frame["value_bins"]) == 128 and sum(frame["value_bins"]) == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= frame["win"] <= 1.0
    assert result["world"] == WORLD and result["position"]["solution"] == "c7h2"


def test_film_frames_sort_numerically_and_share_one_world(tmp_path):
    run = _film_run(tmp_path, [100_000, 9, 0, 1_000_000_000, 10, 250])
    position = _position(tmp_path)
    bl = _blocklist(tmp_path, position)
    result = extract.extract(run, position, bl, expect=None)
    assert [f["step"] for f in result["frames"]] == [0, 9, 10, 250, 100_000, 1_000_000_000]
    film.save_frame(run, 11, "ema", _weights(11), "ffffffffffff", config_to_dict(TINY))
    with pytest.raises(extract.FilmError, match="2 worlds"):
        extract.extract(run, position, bl, expect=None)


def test_the_film_position_was_never_in_training(tmp_path):
    """The position and its whole puzzle line must be in the leakage blocklist, which training never sees."""
    run = _film_run(tmp_path, [0, 250])
    position = _position(tmp_path)
    with pytest.raises(extract.FilmError, match="blocklist"):
        extract.extract(run, position, _blocklist(tmp_path, position, blocked=False), expect=None)
    path = _blocklist(tmp_path, position)
    proof = extract.extract(run, position, path, expect=None)["never_in_training"]
    assert proof["position_blocked"] and proof["line_blocked"]
    assert proof["blocklist"] == "blocklist.npy" and len(proof["blocklist_sha256"]) == 64


def test_a_run_with_the_wrong_frame_count_is_refused_unless_padded(tmp_path):
    run = _film_run(tmp_path, [0, 250, 500])
    position = _position(tmp_path)
    with pytest.raises(extract.FilmError, match="3 frames"):
        extract.extract(run, position, _blocklist(tmp_path, position))


def test_padding_interpolates_between_measured_frames_and_labels_every_interpolated_one(tmp_path):
    run = _film_run(tmp_path, [0, 250, 500, 1000])
    position = _position(tmp_path)
    result = extract.extract(run, position, _blocklist(tmp_path, position), pad_to=21)
    frames = result["frames"]
    measured = [f for f in frames if not f["interpolated"]]
    assert len(frames) == 21 and [f["step"] for f in measured] == [0, 250, 500, 1000]
    assert [f["index"] for f in frames] == list(range(1, 22))
    steps = [f["step"] for f in frames]
    assert steps == sorted(steps)
    first, second = measured[0], measured[1]
    between = [f for f in frames if 0 < f["step"] < 250]
    assert between and all(f["origin"] == "interpolated" for f in between)
    for frame in between:
        for move, p in frame["legal"].items():
            lo, hi = sorted((first["legal"][move], second["legal"][move]))
            assert lo - 1e-9 <= p <= hi + 1e-9
    assert "17 of 21 frames are interpolated" in result["note"]


def test_a_run_without_film_frames_uses_its_ema_checkpoints_and_a_reproduced_init(tmp_path):
    run = tmp_path / "runs" / "skel"
    run.mkdir(parents=True)
    for step in (1000, 2000):
        state = {"step": step, "model": _weights(step), "ema": _weights(step + 1), "world": WORLD}
        checkpoint.save_checkpoint(run, step, {**state, "config": config_to_dict(TINY)})
    sources = extract.frame_sources(run)
    assert [(s.step, s.origin, s.weights) for s in sources] == [
        (0, "reproduced-init", "model"),
        (1000, "checkpoint", "ema"),
        (2000, "checkpoint", "ema"),
    ]
    evaluator, meta = extract.load_frame(sources[0], "cpu")
    torch.manual_seed(TINY.seed)
    expected = BlinkNet(TINY.model).state_dict()
    for name, tensor in evaluator.model.state_dict().items():
        assert torch.equal(tensor, expected[name])
    assert meta["world"] == WORLD and meta["samples"] == 0
    evaluator, meta = extract.load_frame(sources[1], "cpu")
    assert torch.equal(
        evaluator.model.state_dict()["policy.query.weight"], _weights(1001)["policy.query.weight"]
    )
    assert meta["samples"] == 1000 * TINY.batch_size


def test_gpu_hours_per_frame_come_from_the_run_telemetry(tmp_path):
    run = _film_run(tmp_path, [0, 250, 500])
    rows = [{"step": 250, "samples_per_s": 250 * 8 / 36.0}, {"step": 500, "samples_per_s": 250 * 8 / 72.0}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"config": {"batch_size": 8}}), encoding="utf-8")
    position = _position(tmp_path)
    frames = extract.extract(run, position, _blocklist(tmp_path, position), expect=None)["frames"]
    assert [f["gpu_hours"] for f in frames] == pytest.approx([0.0, 0.01, 0.03])
    assert [f["positions_seen"] for f in frames] == [0, 2000, 4000]


def test_predictions_are_legal_move_probabilities_decoded_in_the_real_board_frame(tmp_path):
    torch.manual_seed(0)
    from blink.model.evaluator import TorchEvaluator

    evaluator = TorchEvaluator(BlinkNet(TINY.model), "cpu")
    board = chess.Board(_position(tmp_path).fen)
    (pred,) = extract.predict(evaluator, [board])
    assert set(pred["legal"]) == {m.uci() for m in board.legal_moves}
    assert len(pred["value_bins"]) == 128 and pred["win"] == pytest.approx(
        float(np.dot(pred["value_bins"], np.linspace(0.5 / 128, 1 - 0.5 / 128, 128))), abs=1e-6
    )


def test_write_film_writes_utf8_json_with_lf_endings(tmp_path):
    out = extract.write_film({"frames": [], "note": "ok"}, tmp_path / "film" / "film.json")
    assert b"\r\n" not in out.read_bytes() and json.loads(out.read_text(encoding="utf-8"))["note"] == "ok"


def test_blink_film_pick_then_extract_writes_candidates_and_a_padded_film(tmp_path, monkeypatch, capsys):
    from blink import cli

    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    _film_run(tmp_path, [0, 250, 500], name="demo")
    bands = _bands(tmp_path)
    position = _position(tmp_path)
    bl = _blocklist(tmp_path, position)
    assert cli.main(["film", "pick", "--run", "demo", "--top", "2", "--bands", str(bands)]) == 0
    ranked = json.loads((tmp_path / "film" / "demo" / "candidates.json").read_text(encoding="utf-8"))
    assert {r["puzzle_id"] for r in ranked["ranked"]} == {"mX46Y", "uRVPg"} and ranked["frames"] == 3
    args = [
        "film",
        "extract",
        "--run",
        "demo",
        "--puzzle",
        "mX46Y",
        "--bands",
        str(bands),
        "--blocklist",
        str(bl),
    ]
    assert cli.main([*args, "--pad-to", "21"]) == 0
    written = extract.read_film(tmp_path / "film" / "demo" / "film.json")
    assert len(written["frames"]) == 21 and written["measured_frames"] == 3
    assert "21 frames (3 measured, 18 interpolated)" in capsys.readouterr().out


def test_the_checkpoint_at_the_planned_last_step_is_labelled_the_final_weights(tmp_path):
    run = tmp_path / "runs" / "skel"
    run.mkdir(parents=True)
    for step in (500, TINY.steps):
        state = {"step": step, "model": _weights(step), "ema": _weights(step), "world": WORLD}
        checkpoint.save_checkpoint(run, step, {**state, "config": config_to_dict(TINY)})
    kinds = [extract.load_frame(s, "cpu")[1]["kind"] for s in extract.frame_sources(run)]
    assert kinds == ["init", "ema", "final"]


def test_milestones_mark_the_first_eval_step_whose_ema_top1_passes_each_rung(tmp_path):
    run = _film_run(tmp_path, [0, 250, 500])
    rows = [{"step": 0, "ema_top1": 0.05}, {"step": 250, "ema_top1": 0.21}, {"step": 500, "ema_top1": 0.30}]
    (run / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rungs = [("passed the MLP", 0.2), ("passed a random mover", 0.03), ("never reached", 0.9)]
    assert extract.find_milestones(run, rungs) == [
        {"label": "passed a random mover", "step": 0},
        {"label": "passed the MLP", "step": 250},
    ]
    position = _position(tmp_path)
    film_json = extract.extract(run, position, _blocklist(tmp_path, position), expect=None, milestones=rungs)
    assert [m["step"] for m in film_json["milestones"]] == [0, 250]
    with pytest.raises(extract.FilmError, match="LABEL=TOP1"):
        extract.parse_milestone("no equals sign")
    assert extract.parse_milestone("passed the MLP=0.21") == ("passed the MLP", 0.21)


def test_gpu_hours_of_a_branched_run_start_at_its_branch_step(tmp_path):
    """A preview branched at step 250 bills 250-500 as its own GPU time, not steps 0-500."""
    run = tmp_path / "runs" / "long-preview"
    run.mkdir(parents=True)
    rows = [{"step": 500, "samples_per_s": 250 * 8 / 36.0}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = {
        "config": {"batch_size": 8},
        "branched_from": str(tmp_path / "runs" / "long" / "ckpt_000000250.pt"),
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert extract.gpu_hours_by_step(run) == [(500, pytest.approx(0.01))]
