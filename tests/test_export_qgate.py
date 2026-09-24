"""`blink export qgate`: the int8 model must play like the fp32 one before the page gets it (P10)."""

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import chess
import numpy as np
import pytest

from blink import cli
from blink.board import encode, moves, value
from blink.play.agents import Decision
from blink.play.evaluator import Evaluation

REPO = Path(__file__).resolve().parent.parent


def _report(**overrides):
    from blink.export import qgate

    return dataclasses.replace(qgate.example(), **overrides)


def _band(n: int, fp32: int, int8: int):
    from blink.export import qgate

    return qgate.PuzzleTally(n=n, fp32_correct=fp32, int8_correct=int8)


def test_quantization_gate_rejects_below_99_percent_agreement():
    from blink.export import qgate

    agreement = dataclasses.replace(qgate.example().agreement, top1_agreement=0.9899)
    failing = _report(agreement=agreement)
    assert not failing.passed
    assert failing.failures() == ["top-1 agreement 98.99% on 10,000 positions is below 99%"]
    passing = _report(agreement=dataclasses.replace(agreement, top1_agreement=0.99))
    assert passing.passed and passing.failures() == []


def test_the_gate_rejects_a_puzzle_drop_over_half_a_point_overall_or_in_any_band():
    report = _report(overall=_band(1000, 300, 294), bands=(("<1000", _band(400, 200, 199)),))
    assert report.failures() == ["puzzles dropped 0.60 pt overall (30.00% -> 29.40%), more than 0.5 pt"]
    report = _report(overall=_band(1000, 300, 300), bands=(("2500+", _band(100, 20, 19)),))
    assert report.failures() == ["puzzles dropped 1.00 pt in band 2500+ (20.00% -> 19.00%), more than 0.5 pt"]
    better = _report(
        overall=_band(1000, 300, 310), bands=(("2500+", _band(100, 20, 25)),), puzzle_set_rows=1000
    )
    assert better.passed, "int8 solving more puzzles is never a drop"


def test_the_gate_rejects_a_mean_win_change_over_one_point():
    from blink.export import qgate

    agreement = dataclasses.replace(qgate.example().agreement, mean_abs_dwin_pt=1.01)
    assert _report(agreement=agreement).failures() == ["mean |dwin%| 1.01 pt is above 1 pt"]


def test_the_gate_report_carries_its_thresholds_and_verdict_for_the_card():
    report = _report(agreement=dataclasses.replace(_report().agreement, top1_agreement=0.5))
    data = report.to_dict()
    assert data["thresholds"] == {
        "min_top1_agreement": 0.99,
        "max_puzzle_drop_pt": 0.5,
        "max_mean_abs_dwin_pt": 1.0,
    }
    assert data["passed"] is False and data["failures"] == report.failures()
    assert data["puzzles"]["overall"]["drop_pt"] == pytest.approx(report.overall.drop_pt)
    assert set(data["puzzles"]["bands"]) == {name for name, _ in report.bands}
    assert data["batch_size"] == 1 and data["puzzles"]["set_rows"] == report.puzzle_set_rows
    assert data["exploratory"] is False and data["deviations"] == []
    assert json.loads(json.dumps(data)) == data


def test_the_example_report_is_the_pre_registered_sample():
    from blink.export import qgate

    report = qgate.example()
    assert report.deviations() == [] and not report.exploratory and report.passed
    assert report.agreement.positions == qgate.GATE_POSITIONS and report.agreement.batch_size == 1
    assert report.overall.n == report.puzzle_set_rows == sum(t.n for _, t in report.bands)


def _off_sample(name: str):
    from blink.export import qgate

    agreement = qgate.example().agreement
    bands = qgate.example().bands
    return {
        "python runtime": ({"runtime": "onnxruntime 1.30.0, CPU, 1 thread"}, "not onnxruntime-web"),
        "random positions": ({"positions_source": "random"}, "positions are random, not games10k"),
        "fewer positions": (
            {"agreement": dataclasses.replace(agreement, positions=64)},
            "64 positions, not 10,000",
        ),
        "batched": (
            {"agreement": dataclasses.replace(agreement, batch_size=256)},
            "batch 256, not the page's 1",
        ),
        "dm10k": ({"puzzle_set": "puzzles.csv"}, "puzzles from puzzles.csv, not lichess_bands.csv"),
        "part of the set": ({"puzzle_set_rows": 6000}, "5,281 of the set's 6,000 puzzles"),
        "an empty band": (
            {"bands": (*bands[:4], ("2500+", _band(0, 0, 0)))},
            "band 2500+ has no puzzles",
        ),
    }[name]


@pytest.mark.parametrize(
    "name",
    [
        "python runtime",
        "random positions",
        "fewer positions",
        "batched",
        "dm10k",
        "part of the set",
        "an empty band",
    ],
)
def test_a_run_off_the_pre_registered_sample_is_exploratory_and_never_passes(name):
    overrides, reason = _off_sample(name)
    report = _report(**overrides)
    assert report.failures() == [], "the thresholds alone would pass"
    assert report.exploratory and not report.passed
    assert any(reason in text for text in report.deviations()), report.deviations()
    data = report.to_dict()
    assert data["exploratory"] is True and data["passed"] is False
    assert data["deviations"] == report.deviations()


def test_an_empty_puzzle_set_is_exploratory_not_a_pass():
    empty = tuple((name, _band(0, 0, 0)) for name, _ in _report().bands)
    report = _report(overall=_band(0, 0, 0), bands=empty, puzzle_set_rows=0)
    assert report.failures() == [] and not report.passed
    assert "band <1000 has no puzzles" in report.deviations()


class FixedEvaluator:
    """Fixed policy logits per row and a one-hot value bin per row."""

    def __init__(self, policy: np.ndarray, bins: list[int]) -> None:
        self.policy = policy.astype(np.float32)
        self.bins = bins
        self.calls = 0

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        rows = slice(self.calls, self.calls + len(codes))
        self.calls += len(codes)
        probs = np.zeros((len(codes), value.NUM_BINS), dtype=np.float32)
        probs[np.arange(len(codes)), self.bins[rows]] = 1.0
        return Evaluation(self.policy[rows], probs)


def test_agreement_is_the_legal_masked_top1_and_the_win_change_is_in_points():
    from blink.export import qgate

    n = 4
    masks = np.zeros((n, moves.NUM_MOVES), dtype=bool)
    masks[:, [10, 20]] = True
    fp32 = np.zeros((n, moves.NUM_MOVES))
    fp32[:, 10], fp32[:, 20], fp32[:, 5] = 2.0, 1.0, 9.0  # index 5 is illegal and never counts
    int8 = fp32.copy()
    int8[3, 20] = 3.0  # the last row flips to the second move
    result = qgate.measure_agreement(
        FixedEvaluator(fp32, [64] * n),
        FixedEvaluator(int8, [64, 64, 65, 64]),
        np.zeros((n, 64), dtype=np.uint8),
        masks,
        batch_size=3,
    )
    assert result.positions == 4 and result.batch_size == 3
    assert result.top1_agreement == 0.75
    assert result.mean_abs_dwin_pt == pytest.approx(100 / 128 / 4)
    assert result.max_abs_dwin_pt == pytest.approx(100 / 128)
    flipped = np.exp([2.0, 1.0]) / np.exp([2.0, 1.0]).sum()
    assert result.disagreement_margin_p50_pt == pytest.approx(100 * (flipped[0] - flipped[1]))


class RecordingEvaluator(FixedEvaluator):
    """A FixedEvaluator that remembers how many rows each call carried."""

    def __init__(self, policy: np.ndarray, bins: list[int]) -> None:
        super().__init__(policy, bins)
        self.sizes: list[int] = []

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        self.sizes.append(len(codes))
        return super().evaluate(codes)


def test_agreement_runs_one_row_per_call_as_the_page_does():
    # int8's dynamic activation scale covers the whole batch, so only batch 1 gives the page's logits
    from blink.export import qgate

    n = 5
    masks = np.ones((n, moves.NUM_MOVES), dtype=bool)
    logits = np.random.default_rng(0).normal(size=(n, moves.NUM_MOVES))
    fp32, int8 = RecordingEvaluator(logits, [64] * n), RecordingEvaluator(logits, [64] * n)
    result = qgate.measure_agreement(fp32, int8, np.zeros((n, 64), dtype=np.uint8), masks)
    assert fp32.sizes == int8.sizes == [1] * n
    assert result.batch_size == qgate.PAGE_BATCH == 1 and result.top1_agreement == 1.0


class ConstantEvaluator:
    """Zero policy logits and one fixed value bin for every row: enough for run()'s wiring."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        self.sizes.append(len(codes))
        probs = np.zeros((len(codes), value.NUM_BINS), dtype=np.float32)
        probs[:, 64] = 1.0
        return Evaluation(np.zeros((len(codes), moves.NUM_MOVES), dtype=np.float32), probs)


def _puzzle_csv(path: Path, rows: list[dict]) -> Path:
    lines = ["PuzzleId,FEN,Moves,Rating"] + [
        f"{p['PuzzleId']},{p['FEN']},{p['Moves']},{p['Rating']}" for p in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_run_counts_the_whole_puzzle_set_so_a_limited_run_is_exploratory(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from blink.export import qgate

    full, quantized = ConstantEvaluator(), ConstantEvaluator()

    @contextmanager
    def fake_evaluators(runtime, fp32, int8):
        yield "onnxruntime-web 1.30.0, wasm, 1 thread (Node v22.14.0)", full, quantized

    monkeypatch.setattr(qgate, "evaluators", fake_evaluators)
    csv_path = _puzzle_csv(tmp_path / "lichess_bands.csv", PUZZLES[:3])
    report = qgate.run(tmp_path, tmp_path, "web", "random", str(csv_path), puzzle_limit=2, position_limit=6)
    assert report.agreement.positions == 6 and report.agreement.batch_size == 1
    assert set(full.sizes) == set(quantized.sizes) == {1}, "every call is one row, as on the page"
    assert report.overall.n == 2 and report.puzzle_set_rows == 3
    deviations = report.deviations()
    assert "2 of the set's 3 puzzles were measured" in deviations
    assert "positions are random, not games10k" in deviations
    assert report.exploratory and not report.passed


class ScriptedAgent:
    """Plays the puzzle's solution except in the puzzles it is told to miss."""

    def __init__(self, solutions: dict[str, str], miss: set[str]) -> None:
        self.solutions = solutions
        self.miss = miss

    def choose(self, board: chess.Board, remaining_s=None, game: str = "") -> Decision:
        uci = self.solutions[game]
        if game in self.miss:
            uci = next(m.uci() for m in board.legal_moves if m.uci() != uci and not board.gives_check(m))
        return Decision(chess.Move.from_uci(uci), 1, 1, ("R1",), None, None)


PUZZLES = [
    {"PuzzleId": "a", "FEN": chess.STARTING_FEN, "Moves": "e2e4 e7e5", "Rating": "900"},
    {"PuzzleId": "b", "FEN": chess.STARTING_FEN, "Moves": "d2d4 d7d5", "Rating": "950"},
    {"PuzzleId": "c", "FEN": chess.STARTING_FEN, "Moves": "g1f3 g8f6", "Rating": "2600"},
    {"PuzzleId": "d", "Rating": "1200", "PGN": "1. e4 e5", "Moves": "g1f3 b8c6"},
]


def test_puzzle_tallies_are_paired_per_band_and_read_fen_or_pgn_rows():
    from blink.export import qgate

    solutions = {row["PuzzleId"]: row["Moves"].split()[1] for row in PUZZLES}
    overall, bands = qgate.measure_puzzles(
        ScriptedAgent(solutions, miss={"c"}), ScriptedAgent(solutions, miss={"a", "c"}), PUZZLES
    )
    assert overall == qgate.PuzzleTally(n=4, fp32_correct=3, int8_correct=2)
    assert dict(bands)["<1000"] == qgate.PuzzleTally(n=2, fp32_correct=2, int8_correct=1)
    assert dict(bands)["2500+"] == qgate.PuzzleTally(n=1, fp32_correct=0, int8_correct=0)
    assert dict(bands)["1000-1500"] == qgate.PuzzleTally(n=1, fp32_correct=1, int8_correct=1)
    assert dict(bands)["<1000"].drop_pt == pytest.approx(50.0)


def _write_games10k(path: Path, boards: list[chess.Board]) -> None:
    from blink.data.record import ROOT_DTYPE

    records = np.zeros(len(boards), dtype=ROOT_DTYPE)
    for row, board in enumerate(boards):
        records[row]["board"] = encode.pack(encode.encode_board(board))
    np.save(path, records)


def test_gate_positions_from_games10k_carry_the_legal_moves_of_each_side_to_move(tmp_path):
    from blink.export import qgate

    boards = [chess.Board(), chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 2")]
    boards.append(chess.Board("r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 2"))
    _write_games10k(tmp_path / "games10k.npy", boards)
    codes, masks = qgate.gate_positions(str(tmp_path / "games10k.npy"), n=10_000)
    assert codes.shape == (3, 64) and masks.shape == (3, moves.NUM_MOVES)
    for row, board in enumerate(boards):
        assert np.array_equal(codes[row], encode.encode_board(board))
        assert np.array_equal(masks[row], moves.legal_mask(board)), board.fen()


def test_random_gate_positions_are_seeded_and_legal():
    from blink.export import qgate

    codes, masks = qgate.gate_positions("random", n=20)
    again, _ = qgate.gate_positions("random", n=20)
    assert codes.shape == (20, 64) and np.array_equal(codes, again)
    assert masks.any(axis=1).all()


def _fake_run(report):
    def run(fp32, int8, runtime, position_set, puzzle_set, puzzle_limit=None, position_limit=None):
        return report

    return run


@pytest.mark.parametrize("passed", [True, False])
def test_qgate_writes_its_report_and_the_card_numbers_and_exits_on_the_verdict(
    tmp_path, monkeypatch, capsys, passed
):
    from blink.export import qgate
    from blink.site import card

    export = tmp_path / "skeleton"
    (export / "int8").mkdir(parents=True)
    (export / "model.onnx").write_bytes(b"fp32")
    (export / "int8" / "model.onnx").write_bytes(b"int8")
    card.write(card.example(), export / "int8" / "model.json")
    agreement = dataclasses.replace(qgate.example().agreement, top1_agreement=0.995 if passed else 0.95)
    monkeypatch.setattr(qgate, "run", _fake_run(_report(agreement=agreement)))
    code = cli.main(["export", "qgate", "--model", str(export)])
    assert code == (0 if passed else 1)
    written = json.loads((export / "int8" / "qgate.json").read_text(encoding="utf-8"))
    assert written["passed"] is passed
    gate = card.read(export / "int8" / "model.json")["quantization"]["gate"]
    assert gate["top1_agreement"] == (0.995 if passed else 0.95)
    out = capsys.readouterr().out
    assert ("ok: int8 passes" in out) is passed and ("FAIL: top-1 agreement" in out) is not passed


def test_an_exploratory_qgate_writes_its_report_but_never_stamps_the_card(tmp_path, monkeypatch, capsys):
    from blink.export import qgate
    from blink.site import card

    export = tmp_path / "skeleton"
    (export / "int8").mkdir(parents=True)
    (export / "model.onnx").write_bytes(b"fp32")
    (export / "int8" / "model.onnx").write_bytes(b"int8")
    card_path = card.write(card.example(), export / "int8" / "model.json")
    before = card_path.read_text(encoding="utf-8")
    monkeypatch.setattr(qgate, "run", _fake_run(_report(positions_source="random")))
    code = cli.main(["export", "qgate", "--model", str(export), "--positions", "random"])
    assert code == 1
    written = json.loads((export / "int8" / "qgate.json").read_text(encoding="utf-8"))
    assert written["exploratory"] is True and written["passed"] is False
    assert card_path.read_text(encoding="utf-8") == before, "an exploratory run leaves the card alone"
    out = capsys.readouterr().out
    assert "EXPLORATORY: positions are random, not games10k" in out
    assert "ok: int8 passes" not in out and "card not stamped" in out


def test_qgate_refuses_a_missing_int8_file_and_names_the_command_that_writes_it(tmp_path, capsys):
    (tmp_path / "model.onnx").write_bytes(b"fp32")
    assert cli.main(["export", "qgate", "--model", str(tmp_path)]) == 2
    assert "blink export quantize --int8" in capsys.readouterr().err


def _web_ready() -> bool:
    return shutil.which("node") is not None and (REPO / "site" / "node_modules" / "onnxruntime-web").is_dir()


@pytest.mark.torch
@pytest.mark.local
def test_the_web_runtime_reproduces_onnxruntime_on_the_fp32_model(tmp_path):
    pytest.importorskip("onnxruntime")
    if not _web_ready():
        pytest.skip("node or site/node_modules is absent (run npm ci --prefix site)")
    from blink.export import onnx as export_onnx
    from blink.export import positions, qgate, standin

    path = export_onnx.export(standin.build(seed=0), tmp_path / "model.onnx")
    codes = positions.encode_fens(positions.random_fens(40, seed=5)).astype(np.uint8)
    with qgate.WebRuntime([path]) as web:
        web_eval = web.evaluator(0).evaluate(codes)
        single = web.evaluator(0).evaluate(codes[:1])
    python_eval = qgate.SessionEvaluator(path).evaluate(codes)
    assert np.abs(web_eval.policy_logits - python_eval.policy_logits).max() < 1e-4
    assert np.abs(web_eval.value_probs - python_eval.value_probs).max() < 1e-5
    assert np.array_equal(single.policy_logits[0], web_eval.policy_logits[0])


@pytest.mark.torch
@pytest.mark.local
def test_the_gate_reads_int8_logits_exactly_as_the_page_does_one_row_per_look(tmp_path):
    pytest.importorskip("onnxruntime")
    if not _web_ready():
        pytest.skip("node or site/node_modules is absent (run npm ci --prefix site)")
    from blink.export import onnx as export_onnx
    from blink.export import positions, qgate, quantize, standin

    fp32 = export_onnx.export(standin.build(seed=0), tmp_path / "model.onnx")
    int8 = quantize.quantize_int8(fp32, tmp_path / "int8" / "model.onnx")
    codes = positions.encode_fens(positions.random_fens(40, seed=5)).astype(np.uint8)
    with qgate.WebRuntime([int8]) as web:
        page = np.concatenate([web.run(0, codes[i : i + 1])[0] for i in range(len(codes))])
        gate = qgate.evaluate_rows(web.evaluator(0), codes).policy_logits
        batched = web.run(0, codes)[0]
    assert np.array_equal(gate, page), "the gate's int8 logits are the page's ([1, 64] per look)"
    assert not np.array_equal(batched, page), "why: one dynamic activation scale spans the whole batch"


def test_the_web_runtime_names_node_when_it_is_missing(tmp_path):
    from blink.export import qgate

    with pytest.raises(qgate.GateError, match="node"):
        qgate.WebRuntime([tmp_path / "model.onnx"], node=str(tmp_path / "no-node.exe"))


def test_the_node_side_of_the_web_runtime_uses_the_wasm_entry_with_one_thread():
    text = (REPO / "site" / "tests" / "ortpipe.mjs").read_text(encoding="utf-8")
    assert 'from "onnxruntime-web/wasm"' in text
    assert "ort.env.wasm.numThreads = 1;" in text
    assert 'executionProviders: ["wasm"]' in text
    assert subprocess.run(["git", "check-ignore", "-q", "site/tests/ortpipe.mjs"], cwd=REPO).returncode == 1


@pytest.mark.local
def test_a_model_the_web_runtime_cannot_load_is_an_error_that_carries_nodes_message(tmp_path):
    from blink.export import qgate

    if not _web_ready():
        pytest.skip("node or site/node_modules is absent (run npm ci --prefix site)")
    junk = tmp_path / "model.onnx"
    junk.write_bytes(b"not a model")
    with pytest.raises(qgate.GateError, match=r"the web runtime stopped \(exit 1\): .*Error"):
        qgate.WebRuntime([junk])


def test_a_missing_puzzle_set_or_an_unknown_runtime_is_a_gate_error(tmp_path):
    from blink.export import qgate

    with pytest.raises(qgate.GateError, match="no puzzle set"):
        qgate.read_puzzle_rows(tmp_path / "missing.csv")
    with pytest.raises(qgate.GateError, match="runtime"), qgate.evaluators("gpu", tmp_path, tmp_path):
        pass


@pytest.mark.torch
def test_the_gate_runs_end_to_end_on_the_python_runtime(tmp_path):
    pytest.importorskip("onnxruntime")
    from blink.export import onnx as export_onnx
    from blink.export import qgate, quantize, standin

    fp32 = export_onnx.export(standin.build(seed=0), tmp_path / "model.onnx")
    int8 = quantize.quantize_int8(fp32, tmp_path / "int8" / "model.onnx")
    csv_path = _puzzle_csv(tmp_path / "bands.csv", PUZZLES[:3])
    report = qgate.run(fp32, int8, "python", "random", str(csv_path), position_limit=64)
    assert report.runtime.startswith("onnxruntime 1.30.0, CPU, 1 thread")
    assert report.agreement.positions == 64 and report.agreement.batch_size == 1
    assert 0.0 <= report.agreement.top1_agreement <= 1.0
    assert report.overall.n == 3 and report.puzzle_set == "bands.csv" and report.puzzle_set_rows == 3
    assert [name for name, _ in report.bands] == ["<1000", "1000-1500", "1500-2000", "2000-2500", "2500+"]
    assert report.exploratory and not report.passed, "the Python runtime on random positions is a try-out"
