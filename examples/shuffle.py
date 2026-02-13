"""Salted shuffling, factor encoding, and faux UUIDs.

Deterministic and fully reversible.  Uses a balanced Feistel network
with SHA-256 round keys derived from a user-supplied salt.
"""

import hashlib
import math
import uuid as _uuid


def _derive_round_key(salt: bytes, round_idx: int) -> bytes:
    """Derive a per-round key from the salt.  Precomputed once per call."""
    return hashlib.sha256(salt + round_idx.to_bytes(4, "little")).digest()


def _feistel_round(val: int, round_key: bytes) -> int:
    data = val.to_bytes(16, "little")
    return int.from_bytes(hashlib.sha256(round_key + data).digest()[:8], "little")


def _shuffle_in_range(value: int, n: int, salt: bytes, *, reverse: bool = False) -> int:
    """Bijective permutation on [0, n) using balanced Feistel + cycle-walking."""
    if n <= 1:
        return value

    # Balanced split: a^2 >= n, so both halves are in [0, a)
    a = math.isqrt(n - 1) + 1

    NUM_ROUNDS = 6
    round_keys = [_derive_round_key(salt, k) for k in range(NUM_ROUNDS)]

    x = value
    while True:
        left = x // a
        right = x % a

        rounds = range(NUM_ROUNDS) if not reverse else range(NUM_ROUNDS - 1, -1, -1)
        for k in rounds:
            if not reverse:
                left = (left + _feistel_round(right, round_keys[k])) % a
                left, right = right, left
            else:
                left, right = right, left
                left = (left - _feistel_round(right, round_keys[k])) % a

        x = left * a + right
        if x < n:
            return x


def _to_salt(salt: str) -> bytes:
    return salt.encode("utf-8") if salt else b""


def shuffle_encode(value: int, *, min_digits: int = 3, salt: str = "") -> str:
    """Shuffle an integer, returning a zero-padded string.

    All values with fewer than min_digits digits are shuffled together
    in the range [0, 10^min_digits).  Larger values are shuffled within
    their own digit-count bracket [10^(d-1), 10^d).

    Different salts produce completely different permutations.
    """
    if value < 0:
        raise ValueError("negative values not supported")

    s = _to_salt(salt)
    d = max(len(str(value)), min_digits)

    if d <= min_digits:
        shuffled = _shuffle_in_range(value, 10**d, s)
    else:
        lo = 10 ** (d - 1)
        shuffled = _shuffle_in_range(value - lo, 9 * lo, s) + lo

    return str(shuffled).zfill(d)


def shuffle_decode(shuffled_str: str, *, min_digits: int = 3, salt: str = "") -> int:
    """Reverse a shuffle.  The string length determines the range.

    min_digits and salt must match the values used at encode time.
    """
    s = _to_salt(salt)
    d = len(shuffled_str)
    n = int(shuffled_str)

    if d <= min_digits:
        return _shuffle_in_range(n, 10**d, s, reverse=True)
    else:
        lo = 10 ** (d - 1)
        return _shuffle_in_range(n - lo, 9 * lo, s, reverse=True) + lo


_N128 = 2**128


def int_to_faux_uuid(value: int, *, salt: str = "") -> str:
    """Shuffle an integer into a UUID-formatted 128-bit value.

    Any integer in [0, 2^128) is mapped to a different 128-bit value
    via a salted Feistel permutation, then formatted as a UUID string.
    Different salts produce completely different mappings.
    """
    if not 0 <= value < _N128:
        raise ValueError(f"value must be in [0, 2^128), got {value}")
    shuffled = _shuffle_in_range(value, _N128, _to_salt(salt))
    return str(_uuid.UUID(int=shuffled))


def faux_uuid_to_int(uuid_str: str, *, salt: str = "") -> int:
    """Reverse a faux UUID back to the original integer.

    The salt must match the one used at encode time.
    """
    n = _uuid.UUID(uuid_str).int
    return _shuffle_in_range(n, _N128, _to_salt(salt), reverse=True)


def factor_encode(value: int, *, factor: int) -> int:
    """Encode an integer by multiplying it with a factor."""
    if value < 0:
        raise ValueError("negative values not supported")
    if factor < 2:
        raise ValueError("factor must be >= 2")
    return value * factor


def factor_decode(encoded: int, *, factor: int) -> int:
    """Decode a factor-encoded integer.

    Raises ValueError if the value is not divisible by the factor,
    which indicates a hallucinated or corrupted ID.
    """
    if factor < 2:
        raise ValueError("factor must be >= 2")
    if encoded % factor != 0:
        raise ValueError(
            f"invalid ID: {encoded} is not divisible by the factor"
        )
    return encoded // factor
