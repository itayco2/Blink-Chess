"""ShardLoader: batches of root records from shard files, read the way a spinning disk likes.

Each epoch visits the shards in an order drawn from (seed, epoch). Each shard is read front to back in
one sequential pass, permuted in RAM with a stream of its own, and cut into batches; a record left
over at the end of one shard starts the next batch with the following shard, so every batch is full.
A background thread reads and permutes the next shard while the current one is served (PF14).

The batches form one deterministic stream: batch b is records [b*B, (b+1)*B) of the concatenated
epochs, so `start_batch` resumes a run exactly where it stopped without reading the skipped shards.
"""

import contextlib
import os
import queue
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from blink.data.record import ROOT_DTYPE

READ_CHUNK = 64 << 20  # bytes per readinto call; one file handle reads the whole shard in order
QUEUE_POLL_S = 0.1
ORDER_STREAM, SHUFFLE_STREAM = 0, 1


def _open(path: Path):
    return open(path, "rb", buffering=0)  # noqa: SIM115 - the caller closes it


def read_shard(path: Path, size: int) -> np.ndarray:
    """The whole shard in one front-to-back pass of sequential readinto calls, as root records."""
    raw = np.empty(size, dtype=np.uint8)
    view = memoryview(raw)
    got = 0
    with _open(path) as handle:
        while got < size:
            n = handle.readinto(view[got : got + READ_CHUNK])
            if not n:
                raise OSError(f"{path}: ended after {got:,} of {size:,} bytes")
            got += n
    return raw.view(ROOT_DTYPE)


class _Step(NamedTuple):
    epoch: int
    shard: int
    skip: int  # records of this shard already consumed before start_batch


class _Failure(NamedTuple):
    error: BaseException


_DONE = object()


def _offer(out: queue.Queue, item: object, stop: threading.Event) -> bool:
    """Put `item` unless the consumer has gone. A plain put() would block a finished thread forever."""
    while not stop.is_set():
        try:
            out.put(item, timeout=QUEUE_POLL_S)
            return True
        except queue.Full:
            continue
    return False


class ShardLoader:
    """Iterable of ROOT_DTYPE arrays of exactly batch_size records."""

    def __init__(
        self,
        paths: Sequence[Path],
        batch_size: int,
        seed: int,
        loop: bool = True,
        start_batch: int = 0,
    ) -> None:
        if not paths:
            raise ValueError("ShardLoader got no shard paths")
        if batch_size < 1 or seed < 0 or start_batch < 0:
            raise ValueError(
                "need batch_size >= 1, seed >= 0, start_batch >= 0; "
                f"got batch_size={batch_size}, seed={seed}, start_batch={start_batch}"
            )
        self.paths = tuple(Path(p) for p in paths)
        self.batch_size = batch_size
        self.seed = seed
        self.loop = loop
        self.start_batch = start_batch
        self.sizes = tuple(self._records_in(p) for p in self.paths)
        self.num_records = sum(self.sizes)
        if self.num_records == 0:
            raise ValueError(f"the {len(self.paths)} shards hold no records")

    @staticmethod
    def _records_in(path: Path) -> int:
        size = os.path.getsize(path)
        if size % ROOT_DTYPE.itemsize:
            raise ValueError(f"{path}: {size:,} B is not a whole number of {ROOT_DTYPE.itemsize} B records")
        return size // ROOT_DTYPE.itemsize

    def shard_order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, ORDER_STREAM, epoch]).permutation(len(self.paths))

    def _shuffle(self, records: np.ndarray, step: _Step) -> np.ndarray:
        rng = np.random.default_rng([self.seed, SHUFFLE_STREAM, step.epoch, step.shard])
        return records[rng.permutation(len(records))]

    def _schedule(self) -> Iterator[_Step]:
        """The shards to read, in stream order, starting at the one that holds start_batch."""
        position = self.start_batch * self.batch_size
        epoch, offset = divmod(position, self.num_records)
        while self.loop or epoch == 0:
            for shard in self.shard_order(epoch):
                size = self.sizes[shard]
                if offset < size:
                    yield _Step(epoch, int(shard), offset)
                offset = max(0, offset - size)
            epoch += 1

    def _load(self, step: _Step) -> np.ndarray:
        records = read_shard(self.paths[step.shard], self.sizes[step.shard] * ROOT_DTYPE.itemsize)
        return self._shuffle(records, step)[step.skip :]

    def _prefetch(self, out: queue.Queue, stop: threading.Event) -> None:
        try:
            for step in self._schedule():
                if not _offer(out, self._load(step), stop):
                    return
            _offer(out, _DONE, stop)
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer, which raises it
            _offer(out, _Failure(exc), stop)

    def _shards(self) -> Iterator[np.ndarray]:
        """Permuted shards in stream order, each read by the prefetch thread one shard ahead."""
        out: queue.Queue = queue.Queue(maxsize=1)
        stop = threading.Event()
        thread = threading.Thread(target=self._prefetch, args=(out, stop), name="shard-prefetch", daemon=True)
        thread.start()
        try:
            while (item := out.get()) is not _DONE:
                if isinstance(item, _Failure):
                    raise item.error
                yield item
        finally:
            stop.set()
            thread.join()

    def __iter__(self) -> Iterator[np.ndarray]:
        with contextlib.closing(self._shards()) as shards:  # stops the thread however iteration ends
            yield from _cut(shards, self.batch_size)


def _cut(shards: Iterator[np.ndarray], size: int) -> Iterator[np.ndarray]:
    """Consecutive `size`-record slices of the concatenated shards; a final partial slice is dropped."""
    carry = np.empty(0, dtype=ROOT_DTYPE)
    for records in shards:
        if len(carry):
            need = size - len(carry)
            if len(records) < need:
                carry = np.concatenate([carry, records])
                continue
            yield np.concatenate([carry, records[:need]])
            records = records[need:]
        whole = len(records) // size
        for i in range(whole):
            yield records[i * size : (i + 1) * size]
        carry = records[whole * size :]
