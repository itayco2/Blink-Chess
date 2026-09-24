import itertools
import sys
import types

import numpy as np
import pytest

from blink.data.record import ROOT_DTYPE
from blink.train.source import InMemorySource, shard_source


def _records(n: int) -> np.ndarray:
    records = np.zeros(n, dtype=ROOT_DTYPE)
    records["fen_hash"] = np.arange(n)
    return records


def _hashes(batches) -> list[list[int]]:
    return [batch["fen_hash"].tolist() for batch in batches]


def test_seeking_to_a_step_equals_skipping_to_it():
    source = InMemorySource(_records(40), batch_size=8, seed=5)
    straight = _hashes(itertools.islice(source.batches(0), 23))
    assert _hashes(itertools.islice(source.batches(17), 6)) == straight[17:]


def test_an_epoch_visits_every_record_once_in_a_new_order_each_epoch():
    source = InMemorySource(_records(40), batch_size=8, seed=5)
    first, second = (list(itertools.islice(source.batches(start), 5)) for start in (0, 5))
    assert sorted(np.concatenate(first)["fen_hash"].tolist()) == list(range(40))
    assert sorted(np.concatenate(second)["fen_hash"].tolist()) == list(range(40))
    assert _hashes(first) != _hashes(second)


def test_every_batch_is_full_and_of_root_dtype():
    batches = list(itertools.islice(InMemorySource(_records(21), batch_size=8, seed=0).batches(0), 7))
    assert all(len(b) == 8 and b.dtype == ROOT_DTYPE for b in batches)


def test_a_batch_larger_than_the_data_is_refused():
    with pytest.raises(ValueError):
        InMemorySource(_records(4), batch_size=8, seed=0)


def test_the_shard_source_skips_to_the_resume_step(monkeypatch, tmp_path):
    calls = {}

    class FakeShardLoader:
        def __init__(self, paths, batch_size, seed, loop=True):
            calls.update(paths=paths, batch_size=batch_size, seed=seed, loop=loop)

        def __iter__(self):
            for step in itertools.count():
                batch = _records(calls["batch_size"])
                batch["fen_hash"] = step
                yield batch

    fake = types.ModuleType("blink.data.loader")
    fake.ShardLoader = FakeShardLoader
    monkeypatch.setitem(sys.modules, "blink.data.loader", fake)
    paths = [tmp_path / "train_000.bin"]
    batches = list(itertools.islice(shard_source(paths, batch_size=4, seed=9)(3), 2))
    assert [int(b["fen_hash"][0]) for b in batches] == [3, 4]
    assert calls == {"paths": paths, "batch_size": 4, "seed": 9, "loop": True}
