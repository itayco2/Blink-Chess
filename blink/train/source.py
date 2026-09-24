"""Batch sources for the trainer: a callable start_step -> iterator of ROOT_DTYPE batches.

Taking the start step makes resume exact: the batches after a resume are the batches the straight
run would have seen. InMemorySource seeks in O(1); the shard source skips forward by iterating.
"""

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

BatchSource = Callable[[int], Iterator[np.ndarray]]


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


def shard_source(paths: list[Path], batch_size: int, seed: int) -> BatchSource:
    """Adapt the data area's ShardLoader. Resume skips start_step batches, reading them sequentially."""

    def batches(start_step: int) -> Iterator[np.ndarray]:
        from blink.data.loader import ShardLoader

        loader = ShardLoader(paths, batch_size, seed, loop=True)
        return itertools.islice(iter(loader), start_step, None)

    return batches
