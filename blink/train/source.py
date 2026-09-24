"""Batch sources for the trainer: a callable start_step -> iterator of one optimizer step's records.

A step is a StepData: root records (policy and value loss), child records (value loss only) and the
per-sample rebalancing weights of each. A source may also yield a bare ROOT_DTYPE array, which is a
roots-only step with unit weights. Taking the start step makes resume exact: the batches after a
resume are the batches the straight run would have seen. InMemorySource seeks in O(1); the shard
source asks the data area's ShardLoader to start at that batch.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blink.data.record import CHILD_DTYPE, ROOT_DTYPE

Weigher = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class StepData:
    roots: np.ndarray  # ROOT_DTYPE
    children: np.ndarray  # CHILD_DTYPE, possibly empty
    root_weight: np.ndarray | None = None  # float32 per root; None = all ones
    child_weight: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.roots) + len(self.children)


NO_CHILDREN = np.zeros(0, dtype=CHILD_DTYPE)
BatchSource = Callable[[int], Iterator[np.ndarray | StepData]]


def as_step_data(item: np.ndarray | StepData) -> StepData:
    if isinstance(item, StepData):
        return item
    if item.dtype != ROOT_DTYPE:
        raise TypeError(f"a bare batch must be ROOT_DTYPE records, got {item.dtype}")
    return StepData(roots=item, children=NO_CHILDREN)


class InMemorySource:
    """Epoch-wise shuffles of an in-memory record array; epoch e uses the permutation seeded (seed, e)."""

    def __init__(self, records: np.ndarray, batch_size: int, seed: int) -> None:
        if batch_size <= 0 or batch_size > len(records):
            raise ValueError(f"batch_size must be in 1..{len(records)}, got {batch_size}")
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.steps_per_epoch = len(records) // batch_size

    def _permutation(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(len(self.records))

    def batches(self, start_step: int = 0) -> Iterator[np.ndarray]:
        epoch, offset = divmod(start_step, self.steps_per_epoch)
        while True:
            order = self._permutation(epoch)
            for i in range(offset, self.steps_per_epoch):
                yield self.records[order[i * self.batch_size : (i + 1) * self.batch_size]]
            epoch, offset = epoch + 1, 0


def shard_source(
    paths: list[Path], batch_size: int, seed: int, dtype: np.dtype = ROOT_DTYPE
) -> Callable[[int], Iterator[np.ndarray]]:
    """Adapt the data area's ShardLoader; batch b of the stream is optimizer step b's records."""

    def batches(start_step: int) -> Iterator[np.ndarray]:
        from blink.data.loader import ShardLoader

        extra = {} if dtype == ROOT_DTYPE else {"dtype": dtype}  # the P1 loader streams roots only
        return iter(ShardLoader(paths, batch_size, seed, loop=True, start_batch=start_step, **extra))

    return batches


def mixed_source(
    roots: Callable[[int], Iterator[np.ndarray]],
    children: Callable[[int], Iterator[np.ndarray]] | None = None,
    weigher: Weigher | None = None,
) -> BatchSource:
    """Zip a root stream and a child stream (each already sized per step) into StepData, weighted."""

    def steps(start_step: int) -> Iterator[StepData]:
        child_batches = None if children is None else children(start_step)
        for root_batch in roots(start_step):
            child_batch = NO_CHILDREN if child_batches is None else next(child_batches, None)
            if child_batch is None:
                raise RuntimeError("the child stream ran out before the root stream")
            yield StepData(
                roots=root_batch,
                children=child_batch,
                root_weight=None if weigher is None else weigher(root_batch),
                child_weight=None if weigher is None or not len(child_batch) else weigher(child_batch),
            )

    return steps
