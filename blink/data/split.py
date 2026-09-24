"""Which split a position belongs to, from h = blake2b-8 of its colour-normalised key.

h is stable across processes and machines (never Python's salted hash(), PF11), and a position and
its colour mirror share it, so mirror twins always land in the same split.
"""

import numpy as np

MODULUS = 1000
VAL = frozenset({0, 1})
TEST_IID = frozenset({2, 3})
SPLITS = ("train", "val", "test_iid")  # index = split code
TRAIN_CODE, VAL_CODE, TEST_IID_CODE = range(3)
SPLIT_RULE = "h = blake2b-8(colour-normalised key); h % 1000 in {0,1}: val; in {2,3}: test_iid; else train"


def split_of(fen_hash: int) -> str:
    bucket = fen_hash % MODULUS
    if bucket in VAL:
        return "val"
    if bucket in TEST_IID:
        return "test_iid"
    return "train"


def split_codes(hashes: np.ndarray) -> np.ndarray:
    """Vectorised split_of: uint8 codes indexing SPLITS."""
    bucket = np.asarray(hashes, dtype=np.uint64) % np.uint64(MODULUS)
    codes = np.full(bucket.shape, TRAIN_CODE, dtype=np.uint8)
    codes[np.isin(bucket, np.array(sorted(VAL), dtype=np.uint64))] = VAL_CODE
    codes[np.isin(bucket, np.array(sorted(TEST_IID), dtype=np.uint64))] = TEST_IID_CODE
    return codes
