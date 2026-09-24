"""What a run writes for the dashboard: metrics.jsonl, evals.jsonl, and the fixed validation sample.

Validation top-1 is legal-masked (the argmax over legal moves only, as a player would move); the
unmasked argmax is logged beside it. Legal moves are rebuilt from the 64 square codes alone: in the
side-to-move frame the mover is White, castling rights sit on the rook squares and the en-passant
code marks the target square, so python-chess can list the legal moves of that frame directly.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from blink.board import encode, moves, value
from blink.board.encode import unpack
from blink.model.losses import compute_losses
from blink.train.atomic import write_text_atomic
from blink.train.batch import Batch, make_batch

EVAL_CHUNK = 1024


def board_from_codes(codes: np.ndarray) -> chess.Board:
    """The side-to-move frame as a python-chess board with White to move."""
    board = chess.Board(None)
    rights = chess.BB_EMPTY
    for square, code in enumerate(int(c) for c in codes):
        if encode.OWN <= code < encode.OPP:
            board.set_piece_at(square, chess.Piece(code - encode.OWN + 1, chess.WHITE))
        elif encode.OPP <= code < encode.OWN_CASTLING_ROOK:
            board.set_piece_at(square, chess.Piece(code - encode.OPP + 1, chess.BLACK))
        elif code in (encode.OWN_CASTLING_ROOK, encode.OPP_CASTLING_ROOK):
            color = chess.WHITE if code == encode.OWN_CASTLING_ROOK else chess.BLACK
            board.set_piece_at(square, chess.Piece(chess.ROOK, color))
            rights |= chess.BB_SQUARES[square]
        elif code == encode.EP_SQUARE:
            board.ep_square = square
    board.castling_rights = rights
    board.turn = chess.WHITE
    return board


def legal_mask_from_codes(codes: np.ndarray) -> np.ndarray:
    return moves.legal_mask(board_from_codes(codes))


@dataclass(frozen=True)
class ValSet:
    batch: Batch
    legal: torch.Tensor  # bool [N, 1880]


def make_val_set(records: np.ndarray, device: str | torch.device) -> ValSet:
    masks = np.stack([legal_mask_from_codes(codes) for codes in unpack(records["board"])])
    return ValSet(batch=make_batch(records, device), legal=torch.from_numpy(masks).to(device))


def _forward_in_chunks(model: torch.nn.Module, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    parts = [model(tokens[i : i + EVAL_CHUNK]) for i in range(0, len(tokens), EVAL_CHUNK)]
    return torch.cat([p for p, _ in parts]).float(), torch.cat([v for _, v in parts]).float()


@torch.no_grad()
def evaluate(model: torch.nn.Module, val: ValSet, alpha: float, tau: float) -> dict[str, float]:
    """Policy CE, value CE, legal-masked and unmasked top-1, and win% MAE on the fixed sample (fp32)."""
    was_training = model.training
    model.eval()
    try:
        policy, value_logits = _forward_in_chunks(model, val.batch.tokens)
    finally:
        model.train(was_training)
    loss_policy, loss_value = compute_losses(policy, value_logits, val.batch, alpha, tau)
    best = val.batch.move
    masked = policy.masked_fill(~val.legal, float("-inf"))
    centers = torch.as_tensor(value.BIN_CENTERS, dtype=torch.float32, device=policy.device)
    win = torch.softmax(value_logits, dim=-1) @ centers
    return {
        "n": len(best),
        "policy_ce": loss_policy.item(),
        "value_ce": loss_value.item(),
        "top1": (masked.argmax(-1) == best).float().mean().item(),
        "top1_unmasked": (policy.argmax(-1) == best).float().mean().item(),
        "win_mae": (win - val.batch.w_best).abs().mean().item(),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def truncate_after(path: Path, step: int) -> None:
    """Keep only complete lines whose step is <= step (a resume rewinds the log to its checkpoint)."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        try:
            if json.loads(line)["step"] <= step:
                kept.append(line)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    write_text_atomic(path, "".join(kept))  # the live dashboard may be reading this very file


class MetricWindow:
    """Sums per-step losses on the device (no host sync) until the next metrics line is written."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._reset()

    def _reset(self) -> None:
        zero = torch.zeros((), device=self.device)
        self.loss_policy, self.loss_value, self.grad_norm, self.clipped = zero, zero, zero, zero
        self.steps, self.samples, self.started = 0, 0, time.perf_counter()

    def add(self, loss_policy, loss_value, grad_norm, clip_norm: float, samples: int) -> None:
        self.loss_policy = self.loss_policy + loss_policy
        self.loss_value = self.loss_value + loss_value
        self.grad_norm = self.grad_norm + grad_norm
        self.clipped = self.clipped + (grad_norm > clip_norm).float()
        self.steps += 1
        self.samples += samples

    def flush(self, step: int, lr: float) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        n = max(self.steps, 1)
        on_cuda = self.device.type == "cuda"
        record = {
            "step": step,
            "loss_policy": self.loss_policy.item() / n,
            "loss_value": self.loss_value.item() / n,
            "lr": lr,
            "grad_norm": self.grad_norm.item() / n,
            "clip_frac": self.clipped.item() / n,
            "samples_per_s": self.samples / elapsed,
            "gpu_mem_gb": torch.cuda.max_memory_reserved(self.device) / 2**30 if on_cuda else 0.0,
        }
        self._reset()
        return record
