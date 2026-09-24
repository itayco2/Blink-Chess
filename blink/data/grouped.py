"""test_grouped: whole families of positions held out of training, so the test shows generalisation.

A group is a pawn structure plus a material signature, read from the packed side-to-move board (so a
position and its colour mirror share a group). A group is held out when
    blake2b-8(salt, own pawn bitboard, opponent pawn bitboard, material signature) % 1000 == 7.
Opening structures hold a large share of all positions, and selecting one of them would turn the test
set into "the Italian game". The salt is therefore chosen on a probe: the first salt under which no
selected group holds more than 0.01% of the probe's roots. The salt enters WORLD via the manifest.
"""

import hashlib
from dataclasses import asdict, dataclass

import chess
import numpy as np

from blink.board import encode

MODULUS = 1000
REMAINDER = 7
GIANT_SHARE = 1e-4  # a group holding more than 0.01% of probe roots must never be selected
MAX_SALTS = 10_000
_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
RULE = (
    "test_grouped: blake2b-8(salt u64 LE, own pawn bitboard u64 LE, opponent pawn bitboard u64 LE, "
    "counts of own then opponent P N B R Q as u8, castling rooks counted as rooks) % 1000 == 7, "
    "on the side-to-move board; val and test_iid (by fen_hash) take precedence"
)


def _counts(codes: np.ndarray, base: int, castling_rook: int) -> list[np.ndarray]:
    out = []
    for piece in _PIECES:
        hit = codes == base + piece - 1
        if piece == chess.ROOK:
            hit |= codes == castling_rook
        out.append(hit.sum(axis=1).astype(np.uint8))
    return out


def group_keys(boards: np.ndarray) -> np.ndarray:
    """[N, 32] packed boards -> [N, 26] uint8 keys: two pawn bitboards and ten piece counts."""
    codes = encode.unpack(np.asarray(boards, dtype=np.uint8).reshape(-1, 32))
    own_pawns = np.packbits(codes == encode.OWN + chess.PAWN - 1, axis=1, bitorder="little")
    opp_pawns = np.packbits(codes == encode.OPP + chess.PAWN - 1, axis=1, bitorder="little")
    counts = _counts(codes, encode.OWN, encode.OWN_CASTLING_ROOK)
    counts += _counts(codes, encode.OPP, encode.OPP_CASTLING_ROOK)
    return np.concatenate([own_pawns, opp_pawns, np.stack(counts, axis=1)], axis=1)


def _unique_keys(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.ascontiguousarray(group_keys(boards))
    rows = keys.view(np.dtype((np.void, keys.shape[1]))).ravel()
    unique, inverse = np.unique(rows, return_inverse=True)
    return unique, inverse.ravel()


def _hash_keys(unique: np.ndarray, salt: int) -> np.ndarray:
    prefix = int(salt).to_bytes(8, "little")
    digests = b"".join(hashlib.blake2b(prefix + key.tobytes(), digest_size=8).digest() for key in unique)
    return np.frombuffer(digests, dtype="<u8").astype(np.uint64)


def group_hashes(boards: np.ndarray, salt: int) -> np.ndarray:
    """The salted group hash of each packed board (uint64). Each distinct group is hashed once."""
    unique, inverse = _unique_keys(boards)
    return _hash_keys(unique, salt)[inverse]


def selected(boards: np.ndarray, salt: int) -> np.ndarray:
    """True where the board's group is held out as test_grouped under this salt."""
    return group_hashes(boards, salt) % np.uint64(MODULUS) == np.uint64(REMAINDER)


@dataclass(frozen=True)
class SaltChoice:
    salt: int
    tried: int  # salts looked at, the chosen one included
    probe_roots: int
    groups: int
    selected_groups: int
    selected_roots: int
    largest_selected_roots: int
    giant_threshold_roots: float
    rule: str = RULE

    def as_dict(self) -> dict:
        return asdict(self)


def choose_salt(
    boards: np.ndarray, giant_share: float = GIANT_SHARE, max_tries: int = MAX_SALTS
) -> SaltChoice:
    """The first salt (0, 1, ...) selecting at least one group and none above giant_share of `boards`."""
    unique, inverse = _unique_keys(boards)
    sizes = np.bincount(inverse, minlength=len(unique))
    threshold = giant_share * len(inverse)
    if sizes.min() > threshold:
        raise ValueError(
            f"no salt can work: every one of {len(unique):,} groups holds more than {threshold:.2f} of "
            f"{len(inverse):,} probe roots (the probe is too small for giant_share={giant_share})"
        )
    for salt in range(max_tries):
        chosen = _hash_keys(unique, salt) % np.uint64(MODULUS) == np.uint64(REMAINDER)
        if not chosen.any():
            continue  # a salt that holds nothing out is no test split
        largest = int(sizes[chosen].max())
        if largest <= threshold:
            return SaltChoice(
                salt=salt,
                tried=salt + 1,
                probe_roots=len(inverse),
                groups=len(unique),
                selected_groups=int(chosen.sum()),
                selected_roots=int(sizes[chosen].sum()),
                largest_selected_roots=largest,
                giant_threshold_roots=threshold,
            )
    raise ValueError(
        f"no salt in 0..{max_tries - 1} keeps every selected group at or under {threshold:.1f} of "
        f"{len(inverse):,} probe roots; the probe is dominated by giant groups"
    )
