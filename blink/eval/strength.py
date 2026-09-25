"""PR-6's strength check (EVAL.md section 5): a run's latest EMA checkpoint on the first 2,000 DeepMind
puzzles, beside DeepMind 9M on the same puzzles, so Itay can decide when the flagship is good enough.

`blink eval strength --run long` scores run:<run>:ema, the EMA weights of the run's latest complete
checkpoint (one being written is a .tmp, which no reader lists), with the puzzle scorer exactly as the
v0 probe did:

    blink eval puzzles --model run:<run>:ema --mode both --device cpu --limit 2000 --epsilon 0
                       --out <BLINK_HOME>/eval/puzzles/checks/<run>-<step>

in a child process on the CPU only (CUDA_VISIBLE_DEVICES empty, at most 4 torch threads) at below-normal
priority, so it runs beside the trainer or while the PC is Itay's. The child names the checkpoint it
loaded (its "weights" line): when the run saved a newer one between the look and the load, the check is
filed under the step that was scored. Then the check records value and policy accuracy with Wilson 95%
intervals, a paired comparison with DM-9M's per-puzzle results on the same puzzles (both solved, only
Blink, only DM, neither; <BLINK_HOME>/eval/puzzles/puzzles_dm10k_dm_9M_action-value.csv, 86.6% on the
first 2,000), the step and the run's training hours up to it (blink.train.calibrate.training_seconds:
train-phase intervals, none that spans a pause, resume or restart). One JSON row goes to
<BLINK_HOME>/eval/strength_checks.jsonl, and the trend prints as a table with the DM-9M line.

PR-6 moves PR-3's parity and soak to the first check after 24 training hours: once a run has them and
strength_checks.jsonl holds no PR-3 record for it, the command says so and prints the commands;
`--record-pr3 PARITY_JSON` records the parity report once it has run. Torch-free.
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import blink
from blink.eval import fastchess, puzzles
from blink.play import fastmode
from blink.train import calibrate, status
from blink.train.supervise import checkpoint_steps

CHECK_PUZZLES = 2000  # PR-6: the first 2,000 DeepMind puzzles (dm10k)
MODES = ("value", "policy")
KIND, PR3_KIND = "strength", "pr3"
PR3_HOURS = 24.0  # PR-6: PR-3's parity and soak run on the first check after 24 training hours
TORCH_THREADS = 4
DM_CSV = ("eval", "puzzles", "puzzles_dm10k_dm_9M_action-value.csv")
WEIGHTS = re.compile(r"^weights (.+) \((ema|model)\)\s*$", re.MULTILINE)  # blink eval puzzles prints it
CHECKPOINT = re.compile(r"^ckpt_(\d+)\.pt$")
Scorer = Callable[[str, Path, int], Path | None]  # (run, out folder, puzzles) -> the weights it scored


# ---------------------------------------------------------------- the scorer, a CPU child


def selector(run: str) -> str:
    return f"run:{run}:ema"


def scorer_argv(run: str, out_dir: Path, limit: int) -> list[str]:
    """`blink eval puzzles` exactly as the v0 probe ran it (value mode at epsilon 0, as before E2b)."""
    return [sys.executable, "-m", "blink.cli", "eval", "puzzles", "--model", selector(run), "--mode", "both",
            "--device", "cpu", "--limit", str(limit), "--epsilon", "0", "--out", str(out_dir)]  # fmt: skip


def scorer_env(base: dict[str, str]) -> dict[str, str]:
    """No GPU to see, and at most TORCH_THREADS threads (fewer when the caller already asked for fewer)."""
    try:
        threads = min(TORCH_THREADS, max(1, int(base.get("OMP_NUM_THREADS") or TORCH_THREADS)))
    except ValueError:
        threads = TORCH_THREADS
    threads_env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    return {**base, "CUDA_VISIBLE_DEVICES": "", **threads_env, "PYTHONUTF8": "1"}


def priority_kwargs() -> dict[str, Any]:
    """Below-normal priority from the start, so the venv launcher's child inherits it too."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(10)}


def child_scorer(home: Path) -> Scorer:
    """The real scorer: `blink eval puzzles` in a child process run from this blink's own checkout."""

    def score(run: str, out_dir: Path, limit: int) -> Path | None:
        done = subprocess.run(
            scorer_argv(run, out_dir, limit),
            cwd=Path(blink.__file__).resolve().parents[1],
            env=scorer_env({**os.environ, "BLINK_HOME": str(home)}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            **priority_kwargs(),
        )
        print(done.stdout or "", end="", flush=True)
        if done.returncode not in (0, 1):  # 1: some illegal move, counted in the report
            raise RuntimeError(f"blink eval puzzles exited {done.returncode}: {(done.stderr or '')[-400:]}")
        found = WEIGHTS.findall(done.stdout or "")
        return Path(found[-1][0]) if found else None

    return score


# ---------------------------------------------------------------- results and pairing


def results_csv(out_dir: Path, run: str, mode: str) -> Path:
    """The per-puzzle CSV `blink eval puzzles` writes for this run's EMA selector and mode."""
    label = f"dm10k_{fastchess.model_tag(selector(run))}"
    return puzzles.output_paths(out_dir, label, mode)[0]


def read_results(path: Path) -> dict[str, int]:
    """puzzle_id -> 1 when solved, in file order."""
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["puzzle_id"]: int(row["correct"]) for row in csv.DictReader(handle)}


def dm_results(home: Path, limit: int) -> dict[str, int]:
    """DeepMind 9M's per-puzzle results on the first `limit` puzzles of the same set."""
    path = Path(home).joinpath(*DM_CSV)
    if not path.is_file():
        raise FileNotFoundError(f"no DM-9M per-puzzle results at {path} (blink eval puzzles --model dm:9M)")
    return dict(list(read_results(path).items())[:limit])


def summary(results: dict[str, int]) -> dict[str, Any]:
    correct, n = sum(results.values()), len(results)
    return {"correct": correct, "n": n, "accuracy": correct / n if n else 0.0,
            "wilson95": list(puzzles.wilson(correct, n))}  # fmt: skip


def paired(blink_results: dict[str, int], dm: dict[str, int]) -> dict[str, int]:
    """The two players puzzle by puzzle: both solved, only Blink, only DM, neither."""
    if set(blink_results) != set(dm):
        raise ValueError("Blink and DM-9M were not scored on the same puzzles: the pairing needs them")
    pairs = [(blink_results[pid], dm[pid]) for pid in dm]
    return {"both": pairs.count((1, 1)), "only_blink": pairs.count((1, 0)),
            "only_dm": pairs.count((0, 1)), "neither": pairs.count((0, 0))}  # fmt: skip


def training_hours(run_dir: Path, step: int) -> float:
    """The run's training hours up to `step`, from its metrics rows' time stamps (PR-5's intervals)."""
    rows = [row for row in status.read_rows(Path(run_dir) / "metrics.jsonl") if int(row["step"]) <= step]
    return calibrate.training_seconds(rows) / 3600


def _illegal(out_dir: Path, run: str, mode: str) -> int | None:
    path = results_csv(out_dir, run, mode).with_suffix(".json")
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["illegal_moves"])
    except (OSError, ValueError, KeyError):
        return None


def check_row(home: Path, run: str, step: int, out_dir: Path, limit: int, when: float) -> dict[str, Any]:
    dm = dm_results(home, limit)
    row = {"kind": KIND, "run": run, "step": step, "time": when, "puzzles": limit,
           "hours": training_hours(Path(home) / "runs" / run, step), "out": str(out_dir)}  # fmt: skip
    for mode in MODES:
        results = read_results(results_csv(out_dir, run, mode))
        row[mode] = {**summary(results), "paired_with_dm9m": paired(results, dm),
                     "illegal_moves": _illegal(out_dir, run, mode)}  # fmt: skip
    return {**row, "dm9m": summary(dm)}


# ---------------------------------------------------------------- the check


def checks_path(home: Path) -> Path:
    return Path(home) / "eval" / "strength_checks.jsonl"


def read_checks(home: Path) -> list[dict[str, Any]]:
    return status.read_rows(checks_path(home))


def append_check(home: Path, row: dict[str, Any]) -> None:
    path = checks_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row) + "\n")


def _scored_step(run_dir: Path, step: int, out_dir: Path, weights: Path | None) -> tuple[int, Path]:
    """The step the scorer loaded, and the folder filed under it (moved when a newer checkpoint landed
    between the look and the load)."""
    if weights is None:
        if checkpoint_steps(run_dir)[-1] != step:
            raise RuntimeError(f"runs/{run_dir.name} saved a newer checkpoint while step {step:,} was being "
                               "scored, and the scorer did not say which it loaded: check again")  # fmt: skip
        return step, out_dir
    match = CHECKPOINT.match(Path(weights).name)
    if match is None or Path(weights).parent.name != run_dir.name:
        raise RuntimeError(f"the scorer loaded {weights}, not a checkpoint of runs/{run_dir.name}")
    scored = int(match.group(1))
    if scored == step:
        return step, out_dir
    moved = out_dir.with_name(f"{run_dir.name}-{scored}")
    if moved.exists():
        raise RuntimeError(f"step {scored:,} was scored, but {moved} already holds a check of it")
    out_dir.rename(moved)
    return scored, moved


def run_check(
    run: str,
    home: Path,
    scorer: Scorer,
    limit: int = CHECK_PUZZLES,
    again: bool = False,
    now: Callable[[], float] = time.time,
) -> dict[str, Any] | None:
    """Score the run's latest checkpoint and append the check; None when that step was already checked
    (unless `again`)."""
    run_dir = Path(home) / "runs" / run
    steps = checkpoint_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"runs/{run} has no checkpoint to score")
    done = any(r.get("kind") == KIND and r.get("run") == run and r.get("step") == steps[-1]
               for r in read_checks(home))  # fmt: skip
    if done and not again:
        return None
    out_dir = Path(home) / "eval" / "puzzles" / "checks" / f"{run}-{steps[-1]}"
    step, out_dir = _scored_step(run_dir, steps[-1], out_dir, scorer(run, out_dir, limit))
    row = check_row(home, run, step, out_dir, limit, now())
    append_check(home, row)
    return row


# ---------------------------------------------------------------- the trend and PR-3


def _pct(entry: dict[str, Any]) -> str:
    low, high = entry["wilson95"]
    return f"{100 * entry['accuracy']:5.1f}% ({100 * low:.1f}-{100 * high:.1f})"


def trend(rows: list[dict[str, Any]], run: str | None = None, last: int | None = None) -> list[str]:
    """The checks of `run` (or of every run), oldest first, then DM-9M on the same puzzles."""
    checks = [r for r in rows if r.get("kind") == KIND and (run is None or r.get("run") == run)]
    if not checks:
        return [f"no strength checks{'' if run is None else f' of runs/{run}'} yet"]
    shown = checks[-last:] if last else checks
    head = f"strength checks on the first {checks[-1]['puzzles']:,} DeepMind puzzles (PR-6)"
    lines = [head, f"{'run':<12}{'step':>11}{'hours':>8}  {'value (Wilson 95%)':<22}  policy"]
    for r in shown:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["time"]))
        lines.append(f"{r['run']:<12}{r['step']:>11,}{r['hours']:>8.1f}  {_pct(r['value']):<22}  "
                     f"{_pct(r['policy'])}  {when}")  # fmt: skip
    return [*lines, f"{'DM-9M':<31}  {_pct(checks[-1]['dm9m'])}  (action-value, the same puzzles)"]


def pr3_due(rows: list[dict[str, Any]], run: str, hours: float) -> bool:
    recorded = any(r.get("kind") == PR3_KIND and r.get("run") == run for r in rows)
    return hours >= PR3_HOURS and not recorded


def pr3_lines(run: str, step: int, hours: float) -> list[str]:
    """PR-3's parity and soak (EVAL.md section 5) on this check's weights, with the flagship paused."""
    blink_cli = "python -m blink.cli"
    fast = "--precision bf16 --compile"
    parity = fastchess.NAME_UNSAFE.sub("_", selector(run)).strip("_") + fastmode.tag("bf16", True)
    return [
        f"PR-3 parity and soak are due: runs/{run} has {hours:.1f} training hours at step {step:,} (PR-6: on "
        "the first check after 24 h). With the run paused at this step and the GPU otherwise idle:",
        f"  {blink_cli} bench parity --model {selector(run)} --positions 20000 {fast} --device cuda",
        "    (value mode ships bf16 + compile iff move agreement >= 99% and |dVAA| <= 2 sigma_EMA)",
        f"  {blink_cli} bench play --sizes configs/long.toml --rows 219 --concurrency 5,4,3,2 --iters 5000 "
        f"{fast} --device cuda",
        "    (in the ship mode: without the fast flags if parity failed; a level passes at max <= 1,000 ms "
        "and no move over 1,500 ms)",
        f"  then: {blink_cli} eval strength --run {run} --record-pr3 <BLINK_HOME>/eval/parity/{parity}.json",
    ]


def pr3_record(run: str, parity: Path, when: float) -> dict[str, Any]:
    """The PR-3 record for strength_checks.jsonl, from `blink bench parity`'s report on this run."""
    report = json.loads(Path(parity).read_text(encoding="utf-8"))
    if report.get("model") != selector(run):
        raise ValueError(f"{parity} scored {report.get('model')!r}, not {selector(run)}")
    kept = ("precision", "compile", "policy_top1_agreement", "value_choice_agreement", "max_abs_dwin_pct")
    return {"kind": PR3_KIND, "run": run, "parity": str(parity), "time": when,
            **{key: report[key] for key in kept if key in report}}  # fmt: skip
