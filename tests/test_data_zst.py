"""pzstd frames: walking the skippable headers, decoding frames apart, and stitching lines back."""

import struct

import pytest
from data_fakes import SKIPPABLE_MAGIC, lines_text, synthetic_lines, write_pzstd

from blink.data import frames, zst

LONG_LINE = b"x" * 2500  # longer than two whole frames, so it has frames with no newline at all


def _lines() -> list[bytes]:
    body = [b'{"n": %d, "pad": "%s"}' % (i, b"p" * (i % 37)) for i in range(400)]
    return body[:150] + [LONG_LINE] + body[150:]


def _stitched(reader: zst.FrameReader, workers: int = 1) -> list[bytes]:
    return [line for out in frames.run_frames(reader, list, workers) for line in out.result]


def test_a_multi_frame_zst_with_skippable_headers_reproduces_every_line(tmp_path):
    lines = _lines()
    path = tmp_path / "db.jsonl.zst"
    ends = write_pzstd(path, lines_text(lines), frame_bytes=1000)
    assert len(ends) > 10
    reader = zst.FrameReader(path)
    assert _stitched(reader) == lines
    assert reader.end == "eof"
    assert reader.frames_read == len(ends)


def test_frames_decoded_in_spawned_workers_stitch_in_file_order(tmp_path):
    lines = _lines()
    path = tmp_path / "db.jsonl.zst"
    write_pzstd(path, lines_text(lines), frame_bytes=1000)
    assert _stitched(zst.FrameReader(path), workers=2) == lines


def test_frame_offsets_point_at_zstd_frames_after_their_headers(tmp_path):
    path = tmp_path / "db.jsonl.zst"
    ends = write_pzstd(path, lines_text(_lines()), frame_bytes=1000)
    got = list(zst.FrameReader(path))
    starts = [0] + ends[:-1]
    assert [frame.offset for frame in got] == [start + zst.HEADER.size for start in starts]
    assert all(frame.data[:4] == struct.pack("<I", zst.ZSTD_MAGIC) for frame in got)


def test_reading_a_partial_download_stops_at_the_last_complete_frame(tmp_path):
    lines = _lines()
    text = lines_text(lines)
    full = tmp_path / "full.jsonl.zst"
    ends = write_pzstd(full, text, frame_bytes=1000)
    data = full.read_bytes()
    for cut, complete in ((ends[6] + 5, 7), (ends[7] - 3, 7), (ends[6], 7)):
        part = tmp_path / f"part-{cut}.jsonl.zst"
        part.write_bytes(data[:cut])
        reader = zst.FrameReader(part)
        got = _stitched(reader)
        assert reader.frames_read == complete
        prefix = text[: complete * 1000]
        assert got == prefix.split(b"\n")[:-1]  # the line crossing into the missing frame is dropped
        assert reader.end == ("eof" if cut == ends[6] else "truncated")


def test_a_frame_limit_drops_the_line_that_continues_past_it(tmp_path):
    lines = _lines()
    text = lines_text(lines)
    path = tmp_path / "db.jsonl.zst"
    write_pzstd(path, text, frame_bytes=1000)
    reader = zst.FrameReader(path, limit=3)
    assert _stitched(reader) == text[:3000].split(b"\n")[:-1]
    assert reader.end == "limit"


def test_an_unterminated_last_fragment_is_treated_as_a_partial_line(tmp_path):
    """A download cut at a frame boundary looks like a file without its last newline: never a line."""
    lines = _lines()
    path = tmp_path / "db.jsonl.zst"
    write_pzstd(path, lines_text(lines)[:-1], frame_bytes=1000)
    reader = zst.FrameReader(path)
    assert _stitched(reader) == lines[:-1]
    assert reader.end == "eof"
    assert list(zst.stream_lines(path)) == lines[:-1]


def test_a_file_that_is_not_pzstd_is_refused_with_its_offset(tmp_path):
    path = tmp_path / "plain.zst"
    path.write_bytes(b"\x28\xb5\x2f\xfd" + b"\0" * 64)
    with pytest.raises(zst.FormatError, match="offset 0"):
        list(zst.FrameReader(path))


def test_a_skippable_header_must_be_followed_by_a_zstd_frame(tmp_path):
    path = tmp_path / "bad.zst"
    path.write_bytes(struct.pack("<III", SKIPPABLE_MAGIC, 4, 8) + b"\0" * 8)
    with pytest.raises(zst.FormatError, match="offset 12"):
        list(zst.FrameReader(path))


def test_the_sequential_stream_reads_across_frames(tmp_path):
    lines = _lines()
    path = tmp_path / "db.jsonl.zst"
    write_pzstd(path, lines_text(lines), frame_bytes=1000)
    assert list(zst.stream_lines(path, chunk_size=777)) == lines


def test_the_sequential_stream_stops_at_the_last_complete_line_of_a_truncated_file(tmp_path):
    lines = _lines()
    path = tmp_path / "db.jsonl.zst"
    ends = write_pzstd(path, lines_text(lines), frame_bytes=1000)
    path.write_bytes(path.read_bytes()[: ends[6] + 40])
    got = list(zst.stream_lines(path, chunk_size=777))
    complete = lines_text(lines)[:7000].split(b"\n")[:-1]
    assert got == lines[: len(got)]  # only whole lines, in order
    assert len(got) >= len(complete)


def test_split_text_keeps_fragments_for_the_neighbouring_frames():
    piece = frames.split_text(b"end of a line\nwhole one\nwhole two\nstart of a")
    assert piece.head == b"end of a line"
    assert piece.lines == [b"whole one", b"whole two"]
    assert piece.tail == b"start of a"
    assert piece.has_newline
    middle = frames.split_text(b"no newline here")
    assert not middle.has_newline and middle.head == b"no newline here" and middle.lines == []


def test_decompress_frame_handles_frames_without_a_content_size():
    import zstandard

    text = b"abc\n" * 10_000
    frame = zstandard.ZstdCompressor(write_content_size=False).compress(text)
    assert zst.decompress_frame(frame) == text


def test_run_frames_reports_bytes_and_line_counts_per_frame(tmp_path):
    lines = synthetic_lines(50)
    path = tmp_path / "db.jsonl.zst"
    ends = write_pzstd(path, lines_text(lines), frame_bytes=2048)
    outs = list(frames.run_frames(zst.FrameReader(path), len, workers=1))
    assert sum(out.lines for out in outs) == 50
    assert sum(out.result for out in outs) == 50
    assert sum(out.compressed for out in outs) == ends[-1] - zst.HEADER.size * len(ends)
    assert sum(out.decompressed for out in outs) == len(lines_text(lines))
