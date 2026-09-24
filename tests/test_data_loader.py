"""ShardLoader: seeded shard order per epoch, one sequential read per shard, batches cut in RAM."""

import itertools
import threading
import time

import numpy as np
import pytest

from blink.data import loader
from blink.data.loader import ShardLoader
from blink.data.record import ROOT_DTYPE

SIZES = (37, 50, 0, 23, 64)


@pytest.fixture
def shards(tmp_path):
    paths = []
    for i, size in enumerate(SIZES):
        recs = np.zeros(size, dtype=ROOT_DTYPE)
        recs["fen_hash"] = np.arange(size, dtype=np.uint64) + 1000 * i
        path = tmp_path / f"train_{i:03d}.bin"
        recs.tofile(path)
        paths.append(path)
    return paths


def _all_ids() -> set[int]:
    return {1000 * i + j for i, size in enumerate(SIZES) for j in range(size)}


def _ids(batches) -> list[int]:
    return [int(h) for batch in batches for h in batch["fen_hash"]]


class SpyFile:
    """Wraps a real file and records every read: where it started and how many bytes it returned."""

    def __init__(self, path, log):
        self._file = open(path, "rb", buffering=0)  # noqa: SIM115 - closed by __exit__
        self._log = log
        log.append(("open", str(path)))

    def readinto(self, buffer):
        start = self._file.tell()
        n = self._file.readinto(buffer)
        self._log.append(("read", start, n))
        return n

    def seek(self, *args):
        self._log.append(("seek", args))
        return self._file.seek(*args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._file.close()


def test_loader_reads_each_shard_front_to_back_once(shards, monkeypatch):
    log: list = []
    monkeypatch.setattr(loader, "_open", lambda path: SpyFile(path, log))
    monkeypatch.setattr(loader, "READ_CHUNK", 1000)  # several reads per shard, so order is visible
    list(ShardLoader(shards, batch_size=16, seed=1, loop=False))
    opened = [entry[1] for entry in log if entry[0] == "open"]
    non_empty = [str(p) for p, size in zip(shards, SIZES, strict=True) if size]
    assert sorted(opened) == sorted(non_empty)  # once each; the empty shard is never opened
    assert not [entry for entry in log if entry[0] == "seek"]
    for path, size in zip(shards, SIZES, strict=True):
        if not size:
            continue
        at = log.index(("open", str(path)))
        reads = list(itertools.takewhile(lambda e: e[0] == "read", log[at + 1 :]))
        ends = list(itertools.accumulate(n for *_, n in reads))
        assert [start for _, start, _ in reads] == [0, *ends[:-1]]  # each read starts where the last ended
        assert sum(n for *_, n in reads) == size * ROOT_DTYPE.itemsize


def test_batches_are_root_records_of_exactly_batch_size(shards):
    batches = list(itertools.islice(ShardLoader(shards, batch_size=16, seed=1), 30))
    assert all(batch.dtype == ROOT_DTYPE and len(batch) == 16 for batch in batches)


def test_one_pass_without_loop_yields_each_record_once_and_drops_the_last_partial_batch(shards):
    ids = _ids(ShardLoader(shards, batch_size=16, seed=1, loop=False))
    total = sum(SIZES)
    assert len(ids) == total // 16 * 16 == len(set(ids))
    assert set(ids) <= _all_ids()


def test_the_same_seed_gives_the_same_batches_and_another_seed_does_not(shards):
    def first(seed):
        return _ids(itertools.islice(ShardLoader(shards, batch_size=16, seed=seed), 20))

    assert first(5) == first(5)
    assert first(5) != first(6)


def test_loop_cycles_epochs_each_a_fresh_permutation(shards):
    total = sum(SIZES)
    n_batches = -(-2 * total // 16)
    ids = _ids(itertools.islice(ShardLoader(shards, batch_size=16, seed=3), n_batches))
    first, second = ids[:total], ids[total : 2 * total]
    assert set(first) == set(second) == _all_ids()
    assert first != second


def test_records_are_shuffled_within_a_shard(shards):
    ids = _ids(ShardLoader([shards[4]], batch_size=8, seed=2, loop=False))
    assert sorted(ids) == list(range(4000, 4064))
    assert ids != sorted(ids)


def test_start_batch_resumes_the_exact_stream(shards):
    whole = list(itertools.islice(ShardLoader(shards, batch_size=16, seed=4), 25))
    for start in (0, 3, 11, 17):
        again = ShardLoader(shards, batch_size=16, seed=4, start_batch=start)
        resumed = list(itertools.islice(again, 25 - start))
        assert _ids(resumed) == _ids(whole[start:])


def test_start_batch_past_the_end_of_a_single_pass_yields_nothing(shards):
    assert list(ShardLoader(shards, batch_size=16, seed=4, loop=False, start_batch=100)) == []


def test_a_shard_that_is_not_whole_records_is_refused(tmp_path):
    bad = tmp_path / "train_000.bin"
    bad.write_bytes(b"\0" * (ROOT_DTYPE.itemsize + 1))
    with pytest.raises(ValueError, match="train_000.bin"):
        ShardLoader([bad], batch_size=4, seed=0)


def test_a_loader_without_records_is_refused(tmp_path):
    empty = tmp_path / "val.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="no records"):
        ShardLoader([empty], batch_size=4, seed=0)
    with pytest.raises(ValueError, match="no shard"):
        ShardLoader([], batch_size=4, seed=0)
    with pytest.raises(ValueError, match="batch_size=0"):
        ShardLoader([empty], batch_size=0, seed=0)
    with pytest.raises(ValueError, match="seed=-1"):
        ShardLoader([empty], batch_size=4, seed=-1)
    with pytest.raises(ValueError, match="start_batch=-3"):
        ShardLoader([empty], batch_size=4, seed=0, start_batch=-3)


def test_the_prefetch_thread_stops_when_the_consumer_stops(shards):
    before = threading.active_count()
    batches = iter(ShardLoader(shards, batch_size=16, seed=1))
    next(batches)
    assert threading.active_count() == before + 1
    batches.close()
    assert threading.active_count() == before


def test_closing_early_never_waits_on_a_prefetch_that_already_finished(shards):
    """Two shards, one pass: the thread ends up blocked handing over its end marker. Close must not hang."""
    batches = iter(ShardLoader(shards[:2], batch_size=4, seed=1, loop=False))
    next(batches)
    time.sleep(0.5)  # let the thread load the second shard and reach the end-marker hand-off
    closer = threading.Thread(target=batches.close, daemon=True)
    closer.start()
    closer.join(timeout=5)
    assert not closer.is_alive()


def test_a_read_error_in_the_prefetch_thread_reaches_the_consumer(shards, monkeypatch):
    def broken(path):
        raise OSError(f"disk gone: {path}")

    monkeypatch.setattr(loader, "_open", broken)
    with pytest.raises(OSError, match="disk gone"):
        next(iter(ShardLoader(shards, batch_size=16, seed=1)))


def test_num_records_counts_every_shard(shards):
    assert ShardLoader(shards, batch_size=16, seed=1).num_records == sum(SIZES)
