"""Pluggable UUID registry for text substitution."""

from __future__ import annotations

import json
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


class MemoryRegistry:
    """In-memory UUID registry backed by two dicts."""

    def __init__(self) -> None:
        self._uuid_to_key: dict[str, str] = {}
        self._key_to_uuid: dict[str, str] = {}
        self._counter = 1

    def store(self, uuid: str) -> str:
        normalized = uuid.lower()
        if normalized in self._uuid_to_key:
            return self._uuid_to_key[normalized]
        key = f"UUID_{self._counter}"
        self._counter += 1
        self._uuid_to_key[normalized] = key
        self._key_to_uuid[key] = normalized
        return key

    def fetch(self, key: str) -> str | None:
        return self._key_to_uuid.get(key)

    def __len__(self) -> int:
        return len(self._key_to_uuid)


class FileRegistry:
    """JSON-file-backed UUID registry."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._uuid_to_key: dict[str, str] = {}
        self._key_to_uuid: dict[str, str] = {}
        self._counter = 1
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for key, uuid in data.get("mappings", {}).items():
                normalized = uuid.lower()
                self._key_to_uuid[key] = normalized
                self._uuid_to_key[normalized] = key
            if self._key_to_uuid:
                max_num = max(
                    int(k.split("_", 1)[1]) for k in self._key_to_uuid
                )
                self._counter = max_num + 1

    def store(self, uuid: str) -> str:
        normalized = uuid.lower()
        if normalized in self._uuid_to_key:
            return self._uuid_to_key[normalized]
        key = f"UUID_{self._counter}"
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
