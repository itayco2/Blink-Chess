"""Frame-parallel decoding: each worker decodes whole frames, the parent stitches the seams.

A frame's text is `head \\n line \\n ... line \\n tail`: the head finishes a line begun in earlier frames
and the tail starts one finished later. Workers run `fn` on the complete lines of their frame and
send back only its result plus the two fragments; the parent joins tail(i-1) + head(i) into the seam
line and runs `fn` on it itself. Results come back in file order, whatever the worker count.

Workers are spawned (the only start method on Windows), so `fn` must be a module-level function.
"""

import functools
import multiprocessing
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import Any, NamedTuple

import zstandard

from blink.data import zst


class FrameText(NamedTuple):
    head: bytes  # before the first newline: the end of a line begun in earlier frames
    lines: list[bytes]  # complete lines wholly inside this frame (blank lines are not lines)
    tail: bytes  # after the last newline: the start of a line finished in later frames
    has_newline: bool  # False when the frame is one piece of a line longer than the frame


class FrameOutput(NamedTuple):
    result: Any  # fn(lines) for this piece
    lines: int  # how many lines fn saw
    compressed: int  # compressed bytes of the frame (0 for a seam line)
    decompressed: int  # decompressed bytes of the frame (0 for a seam line)
    decode_s: float  # seconds spent decompressing and splitting
    work_s: float  # seconds spent in fn


def split_text(text: bytes) -> FrameText:
    first = text.find(b"\n")
    if first < 0:
        return FrameText(text, [], b"", False)
    last = text.rfind(b"\n")
    inner = text[first + 1 : last].split(b"\n") if last > first else []
    return FrameText(text[:first], [line for line in inner if line], text[last + 1 :], True)


def decode_frame(data: bytes) -> FrameText:
    return split_text(zst.decompress_frame(data))


def _work_on_frame(fn: Callable[[list[bytes]], Any], frame: zst.Frame) -> tuple[FrameText, FrameOutput]:
    start = time.perf_counter()
    try:
        text = zst.decompress_frame(frame.data)
    except zstandard.ZstdError as exc:
        raise zst.FormatError(f"cannot decompress the frame at offset {frame.offset}: {exc}") from None
    piece = split_text(text)
    decoded = time.perf_counter()
    result = fn(piece.lines)
    out = FrameOutput(
        result=result,
        lines=len(piece.lines),
        compressed=len(frame.data),
        decompressed=len(text),
        decode_s=decoded - start,
        work_s=time.perf_counter() - decoded,
    )
    return piece._replace(lines=[]), out  # the lines stay in the worker; only fn's result travels


def ordered_map(
    fn: Callable[[Any], Any], items: Iterable[Any], workers: int, max_pending: int
) -> Iterator[Any]:
    """map(fn, items) in order on a spawned pool, with at most `max_pending` items in flight.

    Pool.imap would drain `items` as fast as it can read them, pulling a 22 GB file into RAM;
    here the next item is read only when a result has been handed back.
    """
    if workers <= 1:
        yield from map(fn, items)
        return
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        pending: deque = deque()
        for item in items:
            pending.append(pool.apply_async(fn, (item,)))
            if len(pending) >= max_pending:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()


def run_frames(
    frames: Iterable[zst.Frame],
    fn: Callable[[list[bytes]], Any],
    workers: int = 1,
    max_pending: int | None = None,
) -> Iterator[FrameOutput]:
    """fn over every complete line of `frames`, in file order: each seam line, then its frame's lines.

    The fragment left after the last frame is dropped: it is either the start of a line in a frame
    not read (a limit or a partial download) or an unterminated last line (see blink.data.zst).
    """
    work = functools.partial(_work_on_frame, fn)
    pending = max_pending if max_pending is not None else 2 * max(1, workers)
    carry = b""
    for piece, out in ordered_map(work, frames, workers, pending):
        if not piece.has_newline:
            carry += piece.head
            yield out
            continue
        seam = carry + piece.head
        if seam:
            start = time.perf_counter()
            result = fn([seam])
            yield FrameOutput(result, 1, 0, 0, 0.0, time.perf_counter() - start)
        yield out
        carry = piece.tail
