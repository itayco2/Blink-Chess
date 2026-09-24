import numpy as np
import pytest
import zstandard
from train_helpers import FIXTURE, fixture_records

from blink.data.record import ROOT_DTYPE
from blink.train import atomic, rawsource

SKIPPABLE = (0x184D2A50).to_bytes(4, "little") + (4).to_bytes(4, "little") + b"meta"


def _lines() -> list[bytes]:
    return FIXTURE.read_bytes().splitlines(keepends=True)


def _write_zst(path, lines: list[bytes], frames: int = 3, truncate: int = 0) -> None:
    """pzstd-style: a skippable header before each independent frame, optionally cut short."""
    compressor = zstandard.ZstdCompressor()
    size = -(-len(lines) // frames)
    blob = b"".join(
        SKIPPABLE + compressor.compress(b"".join(lines[i : i + size])) for i in range(0, len(lines), size)
    )
    path.write_bytes(blob[: len(blob) - truncate] if truncate else blob)


def test_reading_spans_frames_with_skippable_headers_and_stops_at_max_lines(tmp_path):
    raw = tmp_path / "evals.jsonl.zst"
    _write_zst(raw, _lines())
    assert list(rawsource.read_lines(raw, max_lines=1000)) == _lines()
    assert list(rawsource.read_lines(raw, max_lines=37)) == _lines()[:37]


def test_a_truncated_last_frame_ends_the_stream_at_the_last_complete_line(tmp_path):
    raw = tmp_path / "partial.jsonl.zst"
    _write_zst(raw, _lines(), frames=2, truncate=40)
    got = list(rawsource.read_lines(raw, max_lines=1000))
    assert 50 <= len(got) < len(_lines())
    assert got == _lines()[: len(got)]
    assert all(line.endswith(b"\n") for line in got)


def test_parsing_matches_the_reference_parser_and_counts_rejects(tmp_path):
    lines = _lines()[:30] + [b'{"fen": "not a fen"}\n', b'{"fen": "8/8/8/8/8/8/8/K6k w - -", "evals": []}\n']
    records, rejected = rawsource.parse_lines(lines, workers=1)
    assert records.dtype == ROOT_DTYPE
    assert np.array_equal(records, fixture_records()[:30])
    assert rejected == {"bad_row": 1, "no_evals": 1}


def test_parallel_parsing_equals_serial_parsing():
    serial, _ = rawsource.parse_lines(_lines(), workers=1)
    parallel, _ = rawsource.parse_lines(_lines(), workers=2, chunk_lines=17)
    assert np.array_equal(serial, parallel)


def test_the_parsed_cache_is_built_once_and_then_reused(tmp_path, monkeypatch):
    raw = tmp_path / "evals.jsonl.zst"
    _write_zst(raw, _lines())
    cache_dir = tmp_path / "cache"
    first = rawsource.load_or_build(raw, max_lines=60, cache_dir=cache_dir, workers=1, log=lambda _: None)
    assert first.built and len(first.records) == 60 and first.cache.exists()

    def must_not_parse(*args, **kwargs):
        raise AssertionError("the cache should have been reused")

    monkeypatch.setattr(rawsource, "parse_lines", must_not_parse)
    second = rawsource.load_or_build(raw, max_lines=60, cache_dir=cache_dir, workers=1, log=lambda _: None)
    assert not second.built
    assert np.array_equal(first.records, second.records)
    assert first.sha1 == second.sha1 and len(first.sha1) == 40


def test_a_different_line_count_gets_its_own_cache(tmp_path):
    raw = tmp_path / "evals.jsonl.zst"
    _write_zst(raw, _lines())
    a = rawsource.load_or_build(raw, max_lines=20, cache_dir=tmp_path, workers=1, log=lambda _: None)
    b = rawsource.load_or_build(raw, max_lines=40, cache_dir=tmp_path, workers=1, log=lambda _: None)
    assert a.cache != b.cache and len(b.records) == 40


def test_the_hash_split_sends_0_1_to_val_2_3_to_test_and_the_rest_to_train():
    records = np.zeros(2000, dtype=ROOT_DTYPE)
    records["fen_hash"] = np.arange(2000, dtype=np.uint64)
    train, val, test = rawsource.split(records)
    assert sorted(val["fen_hash"] % 1000) == [0, 0, 1, 1]
    assert sorted(test["fen_hash"] % 1000) == [2, 2, 3, 3]
    assert len(train) == 1992 and not np.isin(train["fen_hash"] % 1000, [0, 1, 2, 3]).any()


def test_a_missing_raw_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        rawsource.load_or_build(tmp_path / "nope.zst", 10, tmp_path, workers=1, log=lambda _: None)


def test_the_cache_is_saved_even_if_a_scanner_briefly_locks_the_fresh_file(tmp_path, monkeypatch):
    raw = tmp_path / "evals.jsonl.zst"
    _write_zst(raw, _lines())
    real_replace = atomic.os.replace
    locks = {"left": 2}

    def scanned(src, dst):
        if locks["left"]:
            locks["left"] -= 1
            raise PermissionError(32, "an antivirus scan holds the file")
        real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", scanned)
    monkeypatch.setattr(atomic, "RETRY_SLEEP_S", 0.0)
    built = rawsource.load_or_build(
        raw, max_lines=30, cache_dir=tmp_path / "c", workers=1, log=lambda _: None
    )
    assert built.built and locks["left"] == 0
    assert np.array_equal(np.load(built.cache, allow_pickle=False), built.records)
    assert not list((tmp_path / "c").glob("*.tmp"))
