"""`blink export qgate`: the int8 browser model must play like its fp32 source (P10).

Three pre-registered checks, each comparing int8 with fp32 on the same runtime:
- top-1 agreement >= 99% on 10,000 positions (games10k: real-game positions kept out of training),
  legal-masked, since that is the move one look plays;
- the puzzle drop is at most 0.5 pt overall and in every rating band: one look with rules R1-R3 under
  DeepMind's scorer, paired (both models see the same puzzles), on the held-out band set
  BLINK_HOME/eval/lichess_bands.csv by default, so the DeepMind test puzzles are never used to choose;
- the mean |dwin%| is at most 1 pt, win% being the expected value over the 128 bins.

The default runtime is the browser's own: onnxruntime-web's WASM backend on one thread, run in Node by
site/tests/ortpipe.mjs. onnxruntime's Python CPU build runs other int8 kernels, whose logits differ from
onnxruntime-web's by up to about 1 on the skeleton, so only the web runtime measures what the page plays.
Every check also feeds one position per call, as the page does ([1, 64] per look): DynamicQuantizeLinear
picks one activation scale per tensor, so an int8 row's logits change with the rows batched beside it
(by up to 0.2 on the stand-in), while fp32's do not.

Only the pre-registered sample can pass: the web runtime, the 10,000 games10k positions at batch 1 and the
whole band set with no empty band. Any other run (--runtime python, --positions random, a limit) is
exploratory: GateReport.deviations() names how it departs, passed is false, and `blink export qgate`
writes qgate.json but leaves the model card alone.
"""

import csv
import json
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np

from blink import paths
from blink.board import encode, moves, value
from blink.data import canon
from blink.eval import puzzles
from blink.export import positions
from blink.export.evaluators import softmax
from blink.play.agents import Agent, PolicyAgent
from blink.play.evaluator import Evaluation, Evaluator

MIN_TOP1_AGREEMENT = 0.99
MAX_PUZZLE_DROP_PT = 0.5
MAX_MEAN_ABS_DWIN_PT = 1.0
GATE_POSITIONS = 10_000
GATE_SEED = 11
PAGE_BATCH = 1  # rows per network call on the page; int8's activation scale spans the whole batch
GATE_POSITION_SET = "games10k"
GATE_PUZZLE_FILE = "lichess_bands.csv"
WEB_RUNTIME = "onnxruntime-web"  # the web runtime's label starts with this
PUZZLE_SETS = {"bands": ("eval", GATE_PUZZLE_FILE), "dm10k": ("downloads", "puzzles.csv")}
POSITION_SETS = {GATE_POSITION_SET: ("data", "games10k.npy")}
RUNTIMES = ("web", "python")
ORTPIPE = Path(__file__).resolve().parents[2] / "site" / "tests" / "ortpipe.mjs"
MAGIC = b"BLNK"
ERROR = 0xFFFFFFFF
EPS = 1e-9


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class PuzzleTally:
    n: int
    fp32_correct: int
    int8_correct: int

    @property
    def fp32_pct(self) -> float:
        return 100 * self.fp32_correct / self.n if self.n else 0.0

    @property
    def int8_pct(self) -> float:
        return 100 * self.int8_correct / self.n if self.n else 0.0

    @property
    def drop_pt(self) -> float:
        return self.fp32_pct - self.int8_pct

    def to_dict(self) -> dict:
        extra = {"fp32_pct": self.fp32_pct, "int8_pct": self.int8_pct, "drop_pt": self.drop_pt}
        return {**asdict(self), **extra}


@dataclass(frozen=True)
class Agreement:
    positions: int
    batch_size: int  # rows per network call; the page's is PAGE_BATCH
    top1_agreement: float
    mean_abs_dwin_pt: float
    max_abs_dwin_pt: float
    disagreement_margin_p50_pt: (
        float | None
    )  # fp32's probability lead of its move over int8's, where they differ


@dataclass(frozen=True)
class GateReport:
    runtime: str
    positions_source: str
    puzzle_set: str
    agreement: Agreement
    overall: PuzzleTally
    bands: tuple[tuple[str, PuzzleTally], ...]
    puzzle_set_rows: int  # puzzles in the whole set file, whatever the limit

    def deviations(self) -> list[str]:
        """How this run departs from the pre-registered sample; any departure makes it exploratory."""
        a = self.agreement
        checks = (
            (
                self.runtime.startswith(WEB_RUNTIME),
                f"runtime is {self.runtime}, not {WEB_RUNTIME} (the page's)",
            ),
            (
                self.positions_source == GATE_POSITION_SET,
                f"positions are {self.positions_source}, not {GATE_POSITION_SET}",
            ),
            (
                a.positions == GATE_POSITIONS,
                f"agreement on {a.positions:,} positions, not {GATE_POSITIONS:,}",
            ),
            (
                a.batch_size == PAGE_BATCH,
                f"agreement measured at batch {a.batch_size}, not the page's {PAGE_BATCH}",
            ),
            (self.puzzle_set == GATE_PUZZLE_FILE, f"puzzles from {self.puzzle_set}, not {GATE_PUZZLE_FILE}"),
            (
                self.overall.n == self.puzzle_set_rows,
                f"{self.overall.n:,} of the set's {self.puzzle_set_rows:,} puzzles were measured",
            ),
        )
        out = [reason for ok, reason in checks if not ok]
        return out + [f"band {name} has no puzzles" for name, tally in self.bands if tally.n == 0]

    @property
    def exploratory(self) -> bool:
        return bool(self.deviations())

    def failures(self) -> list[str]:
        """The thresholds this run misses on its own sample; deviations() judges the sample itself."""
        out = []
        a = self.agreement
        if a.top1_agreement < MIN_TOP1_AGREEMENT:
            out.append(
                f"top-1 agreement {100 * a.top1_agreement:.2f}% on {a.positions:,} positions "
                f"is below {100 * MIN_TOP1_AGREEMENT:g}%"
            )
        for where, tally in (("overall", self.overall), *((f"in band {n}", t) for n, t in self.bands)):
            if tally.drop_pt > MAX_PUZZLE_DROP_PT + EPS:
                out.append(
                    f"puzzles dropped {tally.drop_pt:.2f} pt {where} ({tally.fp32_pct:.2f}% -> "
                    f"{tally.int8_pct:.2f}%), more than {MAX_PUZZLE_DROP_PT:g} pt"
                )
        if a.mean_abs_dwin_pt > MAX_MEAN_ABS_DWIN_PT + EPS:
            out.append(f"mean |dwin%| {a.mean_abs_dwin_pt:.2f} pt is above {MAX_MEAN_ABS_DWIN_PT:g} pt")
        return out

    @property
    def passed(self) -> bool:
        return not self.failures() and not self.exploratory

    def to_dict(self) -> dict:
        return {
            "runtime": self.runtime,
            "positions_source": self.positions_source,
            **asdict(self.agreement),
            "puzzles": {
                "set": self.puzzle_set,
                "set_rows": self.puzzle_set_rows,
                "overall": self.overall.to_dict(),
                "bands": {name: tally.to_dict() for name, tally in self.bands},
            },
            "thresholds": {
                "min_top1_agreement": MIN_TOP1_AGREEMENT,
                "max_puzzle_drop_pt": MAX_PUZZLE_DROP_PT,
                "max_mean_abs_dwin_pt": MAX_MEAN_ABS_DWIN_PT,
            },
            "failures": self.failures(),
            "deviations": self.deviations(),
            "exploratory": self.exploratory,
            "passed": self.passed,
        }


def example() -> GateReport:
    """A passing report with plausible numbers: the card fixture and the tests start from it."""
    bands = (
        ("<1000", PuzzleTally(1007, 700, 699)),
        ("1000-1500", PuzzleTally(1243, 500, 499)),
        ("1500-2000", PuzzleTally(1257, 300, 300)),
        ("2000-2500", PuzzleTally(1409, 150, 149)),
        ("2500+", PuzzleTally(365, 20, 20)),
    )
    agreement = Agreement(
        positions=GATE_POSITIONS,
        batch_size=PAGE_BATCH,
        top1_agreement=0.9931,
        mean_abs_dwin_pt=0.21,
        max_abs_dwin_pt=2.4,
        disagreement_margin_p50_pt=3.1,
    )
    return GateReport(
        runtime=f"{WEB_RUNTIME} 1.30.0, wasm, 1 thread (Node v22.14.0)",
        positions_source=GATE_POSITION_SET,
        puzzle_set=GATE_PUZZLE_FILE,
        agreement=agreement,
        overall=PuzzleTally(5281, 1670, 1667),
        bands=bands,
        puzzle_set_rows=5281,
    )


# ----------------------------------------------------------------------------- measurements


def _masked_top1(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.where(masks, logits, -np.inf).argmax(axis=1)


def _disagreement_margin(logits: np.ndarray, masks: np.ndarray, int8_pick: np.ndarray) -> np.ndarray:
    probs = softmax(np.where(masks, logits.astype(np.float64), -np.inf))
    rows = np.arange(len(logits))
    return 100 * (probs.max(axis=1) - probs[rows, int8_pick])


def evaluate_rows(evaluator: Evaluator, codes: np.ndarray, batch_size: int = PAGE_BATCH) -> Evaluation:
    """Every row of `codes`, `batch_size` rows per network call (the page's one by default)."""
    parts = [
        evaluator.evaluate(codes[start : start + batch_size]) for start in range(0, len(codes), batch_size)
    ]
    policy = np.concatenate([part.policy_logits for part in parts])
    return Evaluation(policy_logits=policy, value_probs=np.concatenate([part.value_probs for part in parts]))


def measure_agreement(
    fp32: Evaluator, int8: Evaluator, codes: np.ndarray, masks: np.ndarray, batch_size: int = PAGE_BATCH
) -> Agreement:
    """Legal-masked top-1 agreement and the change in win% (in points), both models at `batch_size`."""
    a, b = evaluate_rows(fp32, codes, batch_size), evaluate_rows(int8, codes, batch_size)
    pick_a, pick_b = _masked_top1(a.policy_logits, masks), _masked_top1(b.policy_logits, masks)
    differ = pick_a != pick_b
    change = 100 * np.abs(a.win_probability() - b.win_probability())
    margin = _disagreement_margin(a.policy_logits[differ], masks[differ], pick_b[differ])
    return Agreement(
        positions=len(codes),
        batch_size=batch_size,
        top1_agreement=float((~differ).mean()),
        mean_abs_dwin_pt=float(change.mean()),
        max_abs_dwin_pt=float(change.max()),
        disagreement_margin_p50_pt=float(np.median(margin)) if len(margin) else None,
    )


def _puzzle_board(row: dict) -> chess.Board:
    return puzzles.board_from_pgn(row["PGN"]) if row.get("PGN") else chess.Board(row["FEN"])


def _solved(agent: Agent, row: dict) -> bool:
    engine = puzzles.AgentEngine(agent, game=row["PuzzleId"])
    return puzzles.evaluate_puzzle_from_board(_puzzle_board(row), row["Moves"].split(" "), engine)


def measure_puzzles(
    fp32: Agent, int8: Agent, rows: Sequence[dict]
) -> tuple[PuzzleTally, tuple[tuple[str, PuzzleTally], ...]]:
    """Paired puzzle results, overall and per rating band (DeepMind's scorer, one move per call)."""
    names = [name for _, name in puzzles.BANDS] + [puzzles.TOP_BAND]
    counts = {name: [0, 0, 0] for name in ["overall", *names]}
    for row in rows:
        hits = (1, int(_solved(fp32, row)), int(_solved(int8, row)))
        for key in ("overall", puzzles.band_of(int(row["Rating"]))):
            counts[key] = [x + y for x, y in zip(counts[key], hits, strict=True)]
    tallies = {name: PuzzleTally(*count) for name, count in counts.items()}
    return tallies["overall"], tuple((name, tallies[name]) for name in names)


def _frame_board(codes: np.ndarray) -> chess.Board:
    return chess.Board(canon.canonical_epd(codes))


def gate_positions(source: str, n: int = GATE_POSITIONS) -> tuple[np.ndarray, np.ndarray]:
    """uint8 codes [N, 64] and legal masks [N, 1880]: games10k (or an .npy of root records), or random."""
    if source == "random":
        boards = [chess.Board(fen) for fen in positions.random_fens(n, seed=GATE_SEED)]
        codes = np.stack([encode.encode_board(board) for board in boards]).astype(np.uint8)
        return codes, np.stack([moves.legal_mask(board) for board in boards])
    path = paths.home().joinpath(*POSITION_SETS[source]) if source in POSITION_SETS else Path(source)
    if not path.is_file():
        raise GateError(f"no position set at {path} (use --positions random, or build games10k)")
    codes = encode.unpack(np.load(path)["board"][:n])
    # the side-to-move frame has White to move and the same vocabulary indices as the real position
    return codes, np.stack([moves.legal_mask(_frame_board(row)) for row in codes])


def resolve_puzzle_set(name: str) -> Path:
    return paths.home().joinpath(*PUZZLE_SETS[name]) if name in PUZZLE_SETS else Path(name)


def read_puzzle_rows(path: Path) -> list[dict]:
    """Every puzzle row, in DeepMind's (PGN) or Lichess's (FEN) layout, front to back."""
    if not path.is_file():
        raise GateError(f"no puzzle set at {path} (use --puzzles bands, dm10k or a CSV path)")
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        if not {"PuzzleId", "Rating", "Moves"} <= header or not header & {"PGN", "FEN"}:
            raise GateError(
                f"{path} needs PuzzleId, Rating, Moves and PGN or FEN; its header is {sorted(header)}"
            )
        return list(reader)


# ----------------------------------------------------------------------------- runtimes


def _evaluation(policy: np.ndarray, values: np.ndarray) -> Evaluation:
    probs = softmax(values.astype(np.float64)).astype(np.float32)
    return Evaluation(policy_logits=policy.astype(np.float32), value_probs=probs)


class SessionEvaluator:
    """onnxruntime's Python CPU build on one thread (the machine is shared with other jobs)."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        feed = {"tokens": np.asarray(codes, dtype=np.int64)}
        return _evaluation(*self.session.run(["policy_logits", "value_logits"], feed))


class WebRuntime:
    """onnxruntime-web (WASM, one thread) in a Node child process, one session per model.

    Frames on stdin: uint32 model index, uint32 n, then n x 64 uint8 codes. Replies on stdout: uint32 n,
    then float32 policy [n, 1880] and float32 value logits [n, 128]; n = 0xFFFFFFFF carries an error.
    """

    def __init__(self, models: Sequence[Path], node: str = "node", script: Path = ORTPIPE) -> None:
        exe = shutil.which(node)
        if exe is None:
            raise GateError(f"node was not found ({node}): the web runtime runs onnxruntime-web in Node")
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - closed in close(), after the child exits
        command = [exe, str(script), *(str(path) for path in models)]
        self._proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr
        )
        try:
            self.label = self._handshake()
        except BaseException:
            self.close()
            raise

    def _handshake(self) -> str:
        if self._read(len(MAGIC)) != MAGIC:
            raise GateError(f"the web runtime did not start: {self._stderr_text()}")
        (length,) = struct.unpack("<I", self._read(4))
        info = json.loads(self._read(length).decode("utf-8"))
        return f"{WEB_RUNTIME} {info['ort']}, wasm, 1 thread (Node {info['node']})"

    def _stderr_text(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace").strip()[-2000:] or "no output"

    def _read(self, size: int) -> bytes:
        data = self._proc.stdout.read(size)
        if len(data) != size:
            self._proc.wait(timeout=10)
            raise GateError(f"the web runtime stopped (exit {self._proc.returncode}): {self._stderr_text()}")
        return data

    def run(self, index: int, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        self._proc.stdin.write(struct.pack("<II", index, len(codes)) + codes.tobytes())
        self._proc.stdin.flush()
        (n,) = struct.unpack("<I", self._read(4))
        if n == ERROR:
            (length,) = struct.unpack("<I", self._read(4))
            raise GateError(f"onnxruntime-web: {self._read(length).decode('utf-8', errors='replace')}")
        policy = np.frombuffer(self._read(n * moves.NUM_MOVES * 4), dtype="<f4").reshape(n, moves.NUM_MOVES)
        values = np.frombuffer(self._read(n * value.NUM_BINS * 4), dtype="<f4").reshape(n, value.NUM_BINS)
        return policy, values

    def evaluator(self, index: int) -> "WebEvaluator":
        return WebEvaluator(self, index)

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc.stdout.close()
        self._stderr.close()

    def __enter__(self) -> "WebRuntime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass(frozen=True)
class WebEvaluator:
    runtime: WebRuntime
    index: int

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        return _evaluation(*self.runtime.run(self.index, codes))


@contextmanager
def evaluators(runtime: str, fp32: Path, int8: Path) -> Iterator[tuple[str, Evaluator, Evaluator]]:
    """(runtime label, fp32 evaluator, int8 evaluator) on the chosen runtime."""
    if runtime == "python":
        import onnxruntime as ort

        yield f"onnxruntime {ort.__version__}, CPU, 1 thread", SessionEvaluator(fp32), SessionEvaluator(int8)
        return
    if runtime != "web":
        raise GateError(f"runtime must be one of {RUNTIMES}, got {runtime!r}")
    with WebRuntime([fp32, int8]) as web:
        yield web.label, web.evaluator(0), web.evaluator(1)


def run(
    fp32: Path,
    int8: Path,
    runtime: str = "web",
    position_set: str = "games10k",
    puzzle_set: str = "bands",
    puzzle_limit: int | None = None,
    position_limit: int | None = None,
) -> GateReport:
    """Measure the three checks on one runtime, one row per call; the verdict is GateReport.passed."""
    codes, masks = gate_positions(position_set, position_limit or GATE_POSITIONS)
    puzzle_path = resolve_puzzle_set(puzzle_set)
    every_row = read_puzzle_rows(puzzle_path)
    rows = every_row if puzzle_limit is None else every_row[:puzzle_limit]
    with evaluators(runtime, fp32, int8) as (label, full, quantized):
        agreement = measure_agreement(full, quantized, codes, masks)
        overall, bands = measure_puzzles(PolicyAgent(full), PolicyAgent(quantized), rows)
    return GateReport(label, position_set, puzzle_path.name, agreement, overall, bands, len(every_row))
