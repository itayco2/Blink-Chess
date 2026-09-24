"""VAA: value-mode top-1 agreement with Stockfish's PV1 over the valprobe, and the P7 check rules.

Value mode scores every legal child of a root once and plays the child worst for the opponent: the
move value is 1 - W(child), where W is the child's expected win probability for its own side to move.
The play rules decide terminal children without the network: a checkmating child always wins (R2,
lowest vocabulary index first) and a rule-draw child is worth 0.5 (R3). A root counts as correct when
the chosen child is Stockfish's PV1 move or an alternative with the identical score (ties counted as
correct, as the plan states). Forward passes run in bf16 on CUDA in chunks of at most VAA_CHUNK rows,
so no single submission nears the 2 s Windows TDR limit.

The valprobe file (written by the data area, interface 4) holds root_board, root_best, child_offset,
child_board, child_move, child_is_best and child_terminal (0 none, 1 checkmate, 2 rule draw).
"""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from blink.board import encode, moves, value
from blink.train.telemetry import board_from_codes

VAA_CHUNK = 4096
MATE_VALUE = 2.0  # above any 1 - W; lower vocabulary indices rank higher among mates (R2)
DRAW_VALUE = 0.5
NONE, CHECKMATE, RULE_DRAW = 0, 1, 2
CHECK_FRACS = ((0.05, "5%"), (0.25, "25%"), (0.30, "30%"), (0.50, "50%"), (1.0, "100%"))


@dataclass(frozen=True)
class Probe:
    root_board: np.ndarray  # uint8 [N, 32]
    root_best: np.ndarray  # uint16 [N]
    child_offset: np.ndarray  # int64 [N + 1]
    child_board: np.ndarray  # uint8 [M, 32]
    child_move: np.ndarray  # uint16 [M], the move from the root
    child_is_best: np.ndarray  # bool [M]
    child_terminal: np.ndarray  # int8 [M]

    def __post_init__(self) -> None:
        n, m = len(self.root_board), len(self.child_board)
        offsets = self.child_offset
        if len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != m or np.any(np.diff(offsets) < 0):
            raise ValueError(f"child_offset must rise from 0 to {m} over {n + 1} entries")
        for name in ("child_move", "child_is_best", "child_terminal"):
            if len(getattr(self, name)) != m:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {m}")

    @property
    def n_roots(self) -> int:
        return len(self.root_board)

    def subset(self, n: int) -> "Probe":
        """The first n roots and their children (the fixed subset every 2k-step eval scores)."""
        if n >= self.n_roots:
            return self
        end = int(self.child_offset[n])
        return Probe(
            self.root_board[:n],
            self.root_best[:n],
            self.child_offset[: n + 1],
            self.child_board[:end],
            self.child_move[:end],
            self.child_is_best[:end],
            self.child_terminal[:end],
        )


def load_probe(path: Path) -> Probe:
    with np.load(path, allow_pickle=False) as data:
        missing = [f.name for f in fields(Probe) if f.name not in data.files]
        if missing:
            raise ValueError(f"{path} lacks {', '.join(missing)}")
        return Probe(**{f.name: data[f.name] for f in fields(Probe)})


def save_probe(path: Path, probe: Probe) -> None:
    np.savez(path, **{f.name: getattr(probe, f.name) for f in fields(Probe)})


def _terminal(child: chess.Board) -> int:
    if child.is_checkmate():
        return CHECKMATE
    if child.is_stalemate() or child.is_insufficient_material():
        return RULE_DRAW
    return NONE


def _ties(record: np.void) -> set[int]:
    """The best move and every alternative with the identical (cp, mate) score."""
    best = {int(record["move"])}
    for move, cp, mate in zip(record["alt_move"], record["alt_cp"], record["alt_mate"], strict=True):
        if move != 0xFFFF and cp == record["cp"] and mate == record["mate"]:
            best.add(int(move))
    return best


def probe_from_roots(records: np.ndarray) -> Probe:
    """A probe built from root records with python-chess (every legal child, ties marked)."""
    boards, moves_out, best_out, terminal, counts = [], [], [], [], []
    for record, codes in zip(records, encode.unpack(records["board"]), strict=True):
        board, ties = board_from_codes(codes), _ties(record)
        legal = list(board.legal_moves)
        counts.append(len(legal))
        for move in legal:
            index = moves.encode_move(board, move)
            child = board.copy(stack=False)
            child.push(move)
            boards.append(encode.pack(encode.encode_board(child)))
            moves_out.append(index)
            best_out.append(index in ties)
            terminal.append(_terminal(child))
    return Probe(
        root_board=records["board"].copy(),
        root_best=records["move"].copy(),
        child_offset=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        child_board=np.stack(boards) if boards else np.zeros((0, 32), np.uint8),
        child_move=np.asarray(moves_out, dtype=np.uint16),
        child_is_best=np.asarray(best_out, dtype=bool),
        child_terminal=np.asarray(terminal, dtype=np.int8),
    )


@torch.no_grad()
def child_win_probability(
    model: torch.nn.Module, boards: np.ndarray, device: torch.device, chunk: int = VAA_CHUNK
) -> np.ndarray:
    """W for each packed board, for its own side to move, in bounded bf16 chunks on CUDA."""
    centers = torch.as_tensor(value.BIN_CENTERS, dtype=torch.float32, device=device)
    codes = encode.unpack(boards)
    parts = []
    for start in range(0, len(codes), chunk):
        tokens = torch.from_numpy(codes[start : start + chunk].astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, logits = model(tokens)
        parts.append(torch.softmax(logits.float(), dim=-1) @ centers)
    return torch.cat(parts).double().cpu().numpy() if parts else np.zeros(0)


def move_values(w_child: np.ndarray, probe: Probe) -> np.ndarray:
    values = 1.0 - w_child
    mates = probe.child_terminal == CHECKMATE
    values[mates] = MATE_VALUE - probe.child_move[mates] * 1e-6
    values[probe.child_terminal == RULE_DRAW] = DRAW_VALUE
    return values


def agreement(values: np.ndarray, probe: Probe) -> tuple[float, int]:
    """Share of roots (with at least one child) whose highest-value child is a best move."""
    counts = np.diff(probe.child_offset)
    has_children = counts > 0
    root_id = np.repeat(np.arange(probe.n_roots), counts)
    order = np.lexsort((-values, root_id))  # stable: the first child wins an exact tie
    chosen = order[probe.child_offset[:-1][has_children]]
    n = int(has_children.sum())
    return (float(probe.child_is_best[chosen].mean()) if n else math.nan), n


def evaluate_vaa(
    model: torch.nn.Module,
    probe: Probe,
    device: torch.device,
    chunk: int = VAA_CHUNK,
    subset: int | None = None,
) -> dict[str, Any]:
    """{"vaa", "n"} over the probe; with `subset`, also {"vaa_subset", "n_subset"} over its first roots,
    read off the same forward passes (the subset's children are the first children of the probe)."""
    was_training = model.training
    model.eval()
    try:
        w_child = child_win_probability(model, probe.child_board, device, chunk)
    finally:
        model.train(was_training)
    values = move_values(w_child, probe)
    vaa, n = agreement(values, probe)
    out = {"vaa": vaa, "n": n}
    if subset is not None:
        part = probe.subset(subset)
        out["vaa_subset"], out["n_subset"] = agreement(values[: len(part.child_board)], part)
    return out


def check_steps(total: int, preview: bool = False) -> dict[int, str]:
    """The steps of the full-valprobe checks; a preview branch only checks its own end."""
    if preview:
        return {total: "preview"}
    return {max(1, math.floor(frac * total + 0.5)): label for frac, label in CHECK_FRACS}


@dataclass(frozen=True)
class Reference:
    """The evals of the run a long run is checked against (the N* 6 h sweep run)."""

    name: str
    rows: tuple[dict[str, Any], ...]
    cooldown_start: int

    def stable_row(self, samples: int) -> dict[str, Any] | None:
        """Its latest stable-phase eval row (with an EMA VAA) with at most `samples` samples seen."""
        stable = [r for r in self.rows if r["step"] < self.cooldown_start and "ema_vaa" in r]
        if not stable:
            return None
        seen = [r for r in stable if r["samples"] <= samples]
        return seen[-1] if seen else stable[0]

    def stable_at(self, samples: int) -> float | None:
        """Its EMA VAA at the latest stable-phase eval with at most `samples` samples seen."""
        row = self.stable_row(samples)
        return None if row is None else row["ema_vaa"]

    def final(self) -> float | None:
        rows = [r for r in self.rows if "ema_vaa" in r]
        return rows[-1]["ema_vaa"] if rows else None


def load_reference(run_dir: Path) -> Reference:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    evals = run_dir / "evals.jsonl"
    if not evals.is_file():
        raise FileNotFoundError(f"reference run {run_dir.name} has no evals.jsonl")
    rows = [json.loads(line) for line in evals.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row.setdefault("samples", row["step"] * config["batch_size"])
    start = config["steps"] - int(round(config["cooldown_frac"] * config["steps"]))
    return Reference(run_dir.name, tuple(sorted(rows, key=lambda r: r["step"])), start)


def _failed(label: str, ema_vaa: float, threshold: float, rule: str, **extra: Any) -> dict[str, Any]:
    """A failed check: `vaa_check_failed` is True (the supervisor pauses on it), the detail beside it."""
    detail = {"check": label, "ema_vaa": ema_vaa, "threshold": threshold, "rule": rule, **extra}
    return {
        "check": label,
        "check_rule": rule,
        "check_threshold": threshold,
        "vaa_check_failed": True,
        "check_failure": detail,
    }


def _five_percent(
    ema_vaa: float, reference: Reference | None, samples: int, sigma: float, subset: tuple[int, float] | None
) -> dict[str, Any]:
    """EMA VAA >= the reference's stable-phase EMA VAA at equal samples - 2 sigma, on the same roots:
    a reference subset row is met with this check's own subset VAA when the subsets are the same size."""
    row = None if reference is None else reference.stable_row(samples)
    if row is None:
        return {"check": "5%", "check_skipped": "no reference run with stable-phase VAA"}
    own, roots = ema_vaa, "full"
    if row.get("vaa_set") == "subset" and subset is not None and subset[0] == row.get("vaa_n"):
        own, roots = subset[1], "subset"
    threshold = row["ema_vaa"] - 2 * sigma
    rule = f"ema_vaa ({roots}) >= {reference.name} stable VAA at equal samples - 2 sigma"
    if own < threshold:
        return _failed("5%", own, threshold, rule, reference=reference.name)
    return {"check": "5%", "check_rule": rule, "check_threshold": threshold}


def apply_check(
    label: str,
    ema_vaa: float,
    history: list[dict[str, Any]],
    reference: Reference | None,
    samples: int,
    sigma: float,
    subset: tuple[int, float] | None = None,
) -> dict[str, Any]:
    """The fields a check adds to its eval row; a failure sets `vaa_check_failed` True.

    `subset` is (roots, EMA VAA) of this check's first `vaa_subset` roots, for the 5% rule."""
    if label == "5%":
        return _five_percent(ema_vaa, reference, samples, sigma, subset)
    if label in ("25%", "50%"):
        previous = [row for row in history if "check" in row and "ema_vaa" in row]
        if not previous:
            return {"check": label, "check_skipped": "no previous check"}
        prior = previous[-1]
        threshold, rule = prior["ema_vaa"] - 2 * sigma, "ema_vaa >= the previous check's ema_vaa - 2 sigma"
        if ema_vaa < threshold:
            return _failed(label, ema_vaa, threshold, rule, previous_check=prior["check"])
        return {"check": label, "check_rule": rule, "check_threshold": threshold}
    if label == "preview":
        final = None if reference is None else reference.final()
        if final is None:
            return {"check": label, "check_skipped": "no reference run with a final VAA"}
        rule = f"ema_vaa > {reference.name} final VAA"
        if ema_vaa <= final:
            return _failed(label, ema_vaa, final, rule, reference=reference.name)
        return {"check": label, "check_rule": rule, "check_threshold": final}
    return {"check": label}
