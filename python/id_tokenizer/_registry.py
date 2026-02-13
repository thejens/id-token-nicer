"""Pluggable UUID registry for text substitution."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@runtime_checkable
class UuidRegistry(Protocol):
    def store(self, uuid: str) -> str: ...
    def fetch(self, key: str) -> str | None: ...
    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# Digit-preserving Feistel shuffle (keeps keys short)
# ---------------------------------------------------------------------------

_NUM_ROUNDS = 6
_SALT = b"id-tokenizer-registry"


def _derive_round_key(round_idx: int) -> bytes:
    return hashlib.sha256(_SALT + round_idx.to_bytes(4, "little")).digest()


_ROUND_KEYS = [_derive_round_key(k) for k in range(_NUM_ROUNDS)]


def _feistel_round(val: int, round_key: bytes) -> int:
    data = val.to_bytes(16, "little")
    return int.from_bytes(hashlib.sha256(round_key + data).digest()[:8], "little")


def _shuffle_in_range(value: int, n: int) -> int:
    """Bijective permutation on [0, n) via balanced Feistel + cycle-walking."""
    if n <= 1:
        return value
    a = math.isqrt(n - 1) + 1
    x = value
    while True:
        left, right = x // a, x % a
        for k in range(_NUM_ROUNDS):
            left = (left + _feistel_round(right, _ROUND_KEYS[k])) % a
            left, right = right, left
        x = left * a + right
        if x < n:
            return x


_MIN_DIGITS = 3


def _suffix_for(counter: int, *, factor: int, mix: bool) -> str:
    """Turn a 1-based counter into a key suffix.

    Composition order: factor first (multiply), then digit-preserving
    Feistel shuffle.  Factor makes random guesses ~(1/factor) likely to
    look valid.  Mix makes the sequence non-monotonic so the next key is
    unpredictable.  With the defaults (factor=97, mix=True) the first
    ~10 keys are 3 digits, the next ~93 are 4 digits.
    """
    n = counter
    if factor > 1:
        n *= factor
    if mix:
        d = max(len(str(n)), _MIN_DIGITS)
        if d <= _MIN_DIGITS:
            shuffled = _shuffle_in_range(n, 10**d)
        else:
            lo = 10 ** (d - 1)
            shuffled = _shuffle_in_range(n - lo, 9 * lo) + lo
        return str(shuffled).zfill(d)
    return str(n)


class MemoryRegistry:
    """In-memory UUID registry backed by two dicts.

    Parameters
    ----------
    factor : int
        If > 1, multiply the internal counter by this value before
        using it as the key suffix.  A randomly guessed suffix has
        only a 1/factor chance of passing a divisibility check.
    mix : bool
        If True, apply a 32-bit Feistel permutation after the
        (optional) factor step so keys are non-sequential.
    """

    def __init__(self, *, factor: int = 97, mix: bool = True) -> None:
        self._uuid_to_key: dict[str, str] = {}
        self._key_to_uuid: dict[str, str] = {}
        self._counter = 1
        self._factor = factor
        self._mix = mix

    def store(self, uuid: str) -> str:
        normalized = uuid.lower()
        if normalized in self._uuid_to_key:
            return self._uuid_to_key[normalized]
        suffix = _suffix_for(self._counter, factor=self._factor, mix=self._mix)
        key = f"UUID_{suffix}"
        self._counter += 1
        self._uuid_to_key[normalized] = key
        self._key_to_uuid[key] = normalized
        return key

    def fetch(self, key: str) -> str | None:
        return self._key_to_uuid.get(key)

    def __len__(self) -> int:
        return len(self._key_to_uuid)


class FileRegistry:
    """JSON-file-backed UUID registry.

    Parameters
    ----------
    factor : int
        If > 1, multiply the internal counter by this value before
        using it as the key suffix.
    mix : bool
        If True, apply a 32-bit Feistel permutation so keys are
        non-sequential.
    """

    def __init__(self, path: str | Path, *, factor: int = 97, mix: bool = True) -> None:
        self._path = Path(path)
        self._uuid_to_key: dict[str, str] = {}
        self._key_to_uuid: dict[str, str] = {}
        self._factor = factor
        self._mix = mix
        self._counter = 1
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for key, uuid in data.get("mappings", {}).items():
                normalized = uuid.lower()
                self._key_to_uuid[key] = normalized
                self._uuid_to_key[normalized] = key
            # Counter = number of stored mappings + 1 (works regardless of
            # key style since the suffix is no longer the raw counter).
            self._counter = len(self._key_to_uuid) + 1

    def store(self, uuid: str) -> str:
        normalized = uuid.lower()
        if normalized in self._uuid_to_key:
            return self._uuid_to_key[normalized]
        suffix = _suffix_for(self._counter, factor=self._factor, mix=self._mix)
        key = f"UUID_{suffix}"
        self._counter += 1
        self._uuid_to_key[normalized] = key
        self._key_to_uuid[key] = normalized
        self._flush()
        return key

    def fetch(self, key: str) -> str | None:
        return self._key_to_uuid.get(key)

    def __len__(self) -> int:
        return len(self._key_to_uuid)

    def _flush(self) -> None:
        self._path.write_text(json.dumps({"mappings": self._key_to_uuid}, indent=2) + "\n")
