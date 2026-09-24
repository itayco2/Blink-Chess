"""WORLD: which data universe a run was trained in.

WORLD = sha1(contract hash, manifest sha, blocklist sha, split rule and salt)[:12]. It is written to
each run's config.json and into every checkpoint, and --resume refuses a mismatch, so a run can
never silently continue on a different vocabulary, record format, split or data pack.
"""

import hashlib

from blink.board import encode, moves, value
from blink.data import record

SPLIT_RULE = "fen_hash%1000:val{0,1}:test_iid{2,3}"
NO_BLOCKLIST = "none"


class WorldMismatch(RuntimeError):
    pass


def contract_hash() -> str:
    """sha1 of every frozen-contract constant that a trained checkpoint depends on."""
    parts = (
        moves.FROM_TO,
        moves.PROMO_PAIRS,
        moves.PROMO_PIECES,
        (encode.EMPTY, encode.OWN, encode.OPP, encode.OWN_CASTLING_ROOK, encode.OPP_CASTLING_ROOK),
        (encode.EP_SQUARE, encode.NUM_CODES),
        record.ROOT_DTYPE.descr,
        record.CHILD_DTYPE.descr,
        (value.NUM_BINS, value.SIGMA, value.LICHESS_K, value.CP_CLAMP, value.CP_NONE),
        tuple(value.mate_to_cp(m) for m in range(1, 13)),
    )
    return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()


def world_id(
    manifest_sha: str,
    blocklist_sha: str = NO_BLOCKLIST,
    split_rule: str = SPLIT_RULE,
    contract: str | None = None,
) -> str:
    fields = (contract or contract_hash(), manifest_sha, blocklist_sha, split_rule)
    return hashlib.sha1("\n".join(fields).encode("utf-8")).hexdigest()[:12]


def require_same_world(found: str, expected: str) -> None:
    if found != expected:
        raise WorldMismatch(
            f"the checkpoint was trained in world {found} but this data is world {expected}; "
            "refusing to resume (start a new run name instead)"
        )
