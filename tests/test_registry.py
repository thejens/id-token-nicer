"""Tests for UUID registry and text substitution."""

from __future__ import annotations

import json

import pytest

from id_tokenizer import FileRegistry, MemoryRegistry, UuidRegistry
from id_tokenizer._substitution import substitute, restore


# ---------------------------------------------------------------------------
# MemoryRegistry
# ---------------------------------------------------------------------------

class TestMemoryRegistry:
    def test_store_returns_key(self) -> None:
        r = MemoryRegistry()
        assert r.store("550e8400-e29b-41d4-a716-446655440000") == "UUID_1"

    def test_idempotent(self) -> None:
        r = MemoryRegistry()
        k1 = r.store("550e8400-e29b-41d4-a716-446655440000")
        k2 = r.store("550e8400-e29b-41d4-a716-446655440000")
        assert k1 == k2
        assert len(r) == 1

    def test_case_normalization(self) -> None:
        r = MemoryRegistry()
        k1 = r.store("550E8400-E29B-41D4-A716-446655440000")
        k2 = r.store("550e8400-e29b-41d4-a716-446655440000")
        assert k1 == k2

    def test_fetch(self) -> None:
        r = MemoryRegistry()
        key = r.store("550e8400-e29b-41d4-a716-446655440000")
        assert r.fetch(key) == "550e8400-e29b-41d4-a716-446655440000"

    def test_fetch_missing(self) -> None:
        r = MemoryRegistry()
        assert r.fetch("UUID_999") is None

    def test_len(self) -> None:
        r = MemoryRegistry()
        assert len(r) == 0
        r.store("550e8400-e29b-41d4-a716-446655440000")
        r.store("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        assert len(r) == 2

    def test_protocol(self) -> None:
        assert isinstance(MemoryRegistry(), UuidRegistry)


# ---------------------------------------------------------------------------
# FileRegistry
# ---------------------------------------------------------------------------

class TestFileRegistry:
    def test_persistence(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r1 = FileRegistry(path)
        r1.store("550e8400-e29b-41d4-a716-446655440000")

        r2 = FileRegistry(path)
        assert r2.fetch("UUID_1") == "550e8400-e29b-41d4-a716-446655440000"
        assert len(r2) == 1

    def test_counter_recovery(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r1 = FileRegistry(path)
        r1.store("550e8400-e29b-41d4-a716-446655440000")
        r1.store("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

        r2 = FileRegistry(path)
        key = r2.store("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert key == "UUID_3"

    def test_nonexistent_file(self, tmp_path) -> None:
        path = tmp_path / "does_not_exist.json"
        r = FileRegistry(path)
        assert len(r) == 0
        r.store("550e8400-e29b-41d4-a716-446655440000")
        assert path.exists()

    def test_json_format(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r = FileRegistry(path)
        r.store("550e8400-e29b-41d4-a716-446655440000")
        data = json.loads(path.read_text())
        assert "mappings" in data
        assert data["mappings"]["UUID_1"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_protocol(self, tmp_path) -> None:
        assert isinstance(FileRegistry(tmp_path / "p.json"), UuidRegistry)


# ---------------------------------------------------------------------------
# substitute()
# ---------------------------------------------------------------------------

class TestSubstitute:
    def test_single_uuid(self) -> None:
        text = "User 550e8400-e29b-41d4-a716-446655440000 logged in"
        result, reg = substitute(text)
        assert result == "User ${UUID_1} logged in"
        assert len(reg) == 1

    def test_multiple_uuids(self) -> None:
        text = "a=550e8400-e29b-41d4-a716-446655440000 b=6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        result, reg = substitute(text)
        assert "${UUID_1}" in result
        assert "${UUID_2}" in result
        assert len(reg) == 2

    def test_duplicate_uuids(self) -> None:
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        text = f"{uuid} and {uuid}"
        result, reg = substitute(text)
        assert result == "${UUID_1} and ${UUID_1}"
        assert len(reg) == 1

    def test_no_uuids(self) -> None:
        text = "Hello, world!"
        result, reg = substitute(text)
        assert result == "Hello, world!"
        assert len(reg) == 0

    def test_existing_registry(self) -> None:
        reg = MemoryRegistry()
        reg.store("550e8400-e29b-41d4-a716-446655440000")
        text = "New: 6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        result, reg = substitute(text, registry=reg)
        assert "${UUID_2}" in result
        assert len(reg) == 2

    def test_uppercase_uuid(self) -> None:
        text = "ID=550E8400-E29B-41D4-A716-446655440000"
        result, reg = substitute(text)
        assert result == "ID=${UUID_1}"
        assert reg.fetch("UUID_1") == "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# restore()
# ---------------------------------------------------------------------------

class TestRestore:
    def test_roundtrip(self) -> None:
        original = "User 550e8400-e29b-41d4-a716-446655440000 did thing"
        substituted, reg = substitute(original)
        restored = restore(substituted, reg)
        assert restored == original.lower().replace(
            "user", "User"
        )  # UUIDs are normalized to lowercase
        # More precise: the UUID portion is lowercase
        assert "550e8400-e29b-41d4-a716-446655440000" in restored

    def test_unknown_placeholders_left_unchanged(self) -> None:
        reg = MemoryRegistry()
        text = "Value is ${UUID_99}"
        assert restore(text, reg) == "Value is ${UUID_99}"

    def test_mixed_known_unknown(self) -> None:
        reg = MemoryRegistry()
        reg.store("550e8400-e29b-41d4-a716-446655440000")
        text = "${UUID_1} and ${UUID_99}"
        result = restore(text, reg)
        assert "550e8400-e29b-41d4-a716-446655440000" in result
        assert "${UUID_99}" in result


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestRoundtripIntegration:
    def test_llm_simulation(self) -> None:
        """Simulate: user prompt with UUIDs → substitute → LLM replies using
        placeholders → restore → original UUIDs are back."""
        prompt = (
            "Compare order 550e8400-e29b-41d4-a716-446655440000 "
            "with order 6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        )
        sanitized, reg = substitute(prompt)

        # LLM never sees real UUIDs
        assert "550e8400" not in sanitized
        assert "6ba7b810" not in sanitized

        # Simulate LLM response using placeholders
        llm_response = f"Order {sanitized.split('order ')[1].split(' with')[0]} is newer than {sanitized.split('order ')[2]}."

        # A simpler approach: the LLM echoes back the placeholders
        llm_response = "The first order ${UUID_1} was placed before ${UUID_2}."

        restored = restore(llm_response, reg)
        assert "550e8400-e29b-41d4-a716-446655440000" in restored
        assert "6ba7b810-9dad-11d1-80b4-00c04fd430c8" in restored
        assert "${UUID_" not in restored
