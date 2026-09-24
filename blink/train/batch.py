"""ROOT_DTYPE and CHILD_DTYPE records -> tensors on the training device.

Win probabilities come from blink.board.value.win_probability_array on the CPU (float64), then
move to the device as float32. Board codes travel as uint8 and widen to int64 on the device.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from blink.board.encode import unpack
from blink.board.value import win_probability_array
from blink.data.record import CHILD_DTYPE, NO_MOVE, ROOT_DTYPE


@dataclass(frozen=True)
class Batch:
    tokens: torch.Tensor  # int64 [B, 64]
    move: torch.Tensor  # int64 [B]
    alt_move: torch.Tensor  # int64 [B, 4], NO_MOVE where absent
    alt_valid: torch.Tensor  # bool [B, 4]
    w_best: torch.Tensor  # float32 [B]
    w_alt: torch.Tensor  # float32 [B, 4]

    def __len__(self) -> int:
        return self.move.shape[0]


def make_batch(records: np.ndarray, device: str | torch.device) -> Batch:
    if records.dtype != ROOT_DTYPE:
        raise TypeError(f"expected ROOT_DTYPE records, got {records.dtype}")
    codes = unpack(records["board"])
    alt_move = records["alt_move"].astype(np.int64)
    arrays = {
        "tokens": torch.from_numpy(np.ascontiguousarray(codes)),
        "move": torch.from_numpy(records["move"].astype(np.int64)),
        "alt_move": torch.from_numpy(alt_move),
        "alt_valid": torch.from_numpy(alt_move != NO_MOVE),
        "w_best": torch.from_numpy(win_probability_array(records["cp"], records["mate"]).astype(np.float32)),
        "w_alt": torch.from_numpy(
            win_probability_array(records["alt_cp"], records["alt_mate"]).astype(np.float32)
        ),
    }
    moved = {name: tensor.to(device, non_blocking=True) for name, tensor in arrays.items()}
    return Batch(**{**moved, "tokens": moved["tokens"].long()})


@dataclass(frozen=True)
class ChildBatch:
    tokens: torch.Tensor  # int64 [C, 64]
    w: torch.Tensor  # float32 [C]: the child's win probability for its own side to move

    def __len__(self) -> int:
        return self.w.shape[0]


def make_child_batch(records: np.ndarray, device: str | torch.device) -> ChildBatch:
    if records.dtype != CHILD_DTYPE:
        raise TypeError(f"expected CHILD_DTYPE records, got {records.dtype}")
    codes = torch.from_numpy(np.ascontiguousarray(unpack(records["board"])))
    w = torch.from_numpy(win_probability_array(records["cp"], records["mate"]).astype(np.float32))
    return ChildBatch(tokens=codes.to(device, non_blocking=True).long(), w=w.to(device, non_blocking=True))


def take(batch: Batch | ChildBatch, rows: slice) -> Batch | ChildBatch:
    """The same kind of batch holding only `rows`."""
    fields = {f.name: getattr(batch, f.name)[rows] for f in dataclasses.fields(batch)}
    return type(batch)(**fields)
