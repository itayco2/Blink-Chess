"""Reading the pzstd-compressed eval DB, front to back, with plain sequential reads.

pzstd writes every zstd frame behind a 12-byte skippable frame: magic 0x184D2A50, payload size 4,
then the compressed size of the next frame as u32 LE. Each frame decompresses to about 32 MiB and the
text is cut at fixed sizes, so lines cross frame boundaries (blink.data.frames stitches them).

A partial download ends inside a frame; FrameReader stops cleanly before it. An unterminated last
fragment is never a line: the eval DB ends every line with a newline, and a download cut exactly at
a frame boundary would otherwise pass off half a line as a whole one.
"""

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import zstandard

SKIPPABLE_MAGIC = 0x184D2A50
ZSTD_MAGIC = 0xFD2FB528
HEADER = struct.Struct("<III")  # skippable magic, payload size (always 4), next frame's compressed size
PAYLOAD_SIZE = 4
# decompress() with a size bound is 15x faster than decompressobj() on a 32 MiB frame (PF07).
MAX_FRAME_OUTPUT = 128 << 20
READ_CHUNK = 8 << 20


class FormatError(ValueError):
    """The file is not laid out as pzstd writes it."""


class Frame(NamedTuple):
    offset: int  # file offset of the zstd frame itself (just after its skippable header)
    data: bytes  # the compressed frame


class FrameReader:
    """Iterates the complete frames of a pzstd file in order, reading it once front to back.

    After iteration, `end` says why it stopped: "eof" (the file ended on a frame boundary),
    "truncated" (the last frame or header is incomplete: a partial download) or "limit".
    """

    def __init__(self, path: Path, limit: int | None = None) -> None:
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be at least 1 frame, got {limit}")
        self.path = Path(path)
        self.limit = limit
        self.frames_read = 0
        self.end: str | None = None

    def __iter__(self) -> Iterator[Frame]:
        self.frames_read, self.end = 0, None
        with open(self.path, "rb", buffering=READ_CHUNK) as handle:
            offset = 0
            while True:
                if self.limit is not None and self.frames_read >= self.limit:
                    self.end = "limit"
                    return
                header = handle.read(HEADER.size)
                if not header:
                    self.end = "eof"
                    return
                if len(header) < HEADER.size:
                    self.end = "truncated"
                    return
                size = _frame_size(header, offset)
                data = handle.read(size)
                if len(data) < size:
                    self.end = "truncated"
                    return
                _check_zstd_magic(data, offset + HEADER.size)
                yield Frame(offset + HEADER.size, data)
                self.frames_read += 1
                offset += HEADER.size + size


def _frame_size(header: bytes, offset: int) -> int:
    magic, payload, size = HEADER.unpack(header)
    if magic != SKIPPABLE_MAGIC or payload != PAYLOAD_SIZE:
        raise FormatError(
            f"expected a pzstd skippable header at offset {offset}, found magic {magic:#010x} "
            f"payload {payload}; this reader needs pzstd output"
        )
    return size


def _check_zstd_magic(data: bytes, offset: int) -> None:
    if len(data) < 4 or struct.unpack_from("<I", data)[0] != ZSTD_MAGIC:
        raise FormatError(f"expected a zstd frame at offset {offset}")


def decompress_frame(data: bytes) -> bytes:
    """One whole zstd frame to its text. pzstd frames carry no content size, so the output is bounded."""
    try:
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=MAX_FRAME_OUTPUT)
    except zstandard.ZstdError:
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)  # frames above the bound


def stream_lines(path: Path, chunk_size: int = READ_CHUNK) -> Iterator[bytes]:
    """Every complete line of a zstd file, in order, with one sequential stream across all frames.

    read_across_frames=True matters: the default stops after the first frame (PF07). A truncated file
    ends the stream quietly, and the unterminated fragment it leaves is dropped.
    """
    decompressor = zstandard.ZstdDecompressor()
    with (
        open(path, "rb", buffering=READ_CHUNK) as raw,
        decompressor.stream_reader(raw, read_size=READ_CHUNK, read_across_frames=True) as reader,
    ):
        carry = b""
        while chunk := reader.read(chunk_size):
            pieces = (carry + chunk).split(b"\n")
            carry = pieces.pop()
            yield from (piece for piece in pieces if piece)
