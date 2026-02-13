"""Tests for UUID registry and text substitution."""

from __future__ import annotations

import json
import re

import pytest

from id_tokenizer import FileRegistry, MemoryRegistry, UuidRegistry
from id_tokenizer._substitution import substitute, restore


# ---------------------------------------------------------------------------
# MemoryRegistry
# ---------------------------------------------------------------------------

class TestMemoryRegistry:
    def test_store_returns_key_plain(self) -> None:
        r = MemoryRegistry(factor=0, mix=False)
        assert r.store("550e8400-e29b-41d4-a716-446655440000") == "UUID_1"

    def test_store_returns_key_default(self) -> None:
        r = MemoryRegistry()
        key = r.store("550e8400-e29b-41d4-a716-446655440000")
        assert key.startswith("UUID_")
        assert key != "UUID_1"

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

    def test_factor_only(self) -> None:
        r = MemoryRegistry(factor=97, mix=False)
        k1 = r.store("550e8400-e29b-41d4-a716-446655440000")
        k2 = r.store("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        assert k1 == "UUID_97"
        assert k2 == "UUID_194"

    def test_mix_keys_non_sequential(self) -> None:
        r = MemoryRegistry(factor=0, mix=True)
        keys = [
            r.store(f"550e8400-e29b-41d4-a716-44665544{i:04d}")
            for i in range(5)
        ]
        suffixes = [int(k.split("_", 1)[1]) for k in keys]
        assert suffixes != sorted(suffixes)

    def test_default_keys_short(self) -> None:
        """Default keys (factor=97, mix=True) should be 3-4 digits."""
        r = MemoryRegistry()
        keys = [
            r.store(f"550e8400-e29b-41d4-a716-44665544{i:04d}")
            for i in range(10)
        ]
        for key in keys:
            suffix = key.split("_", 1)[1]
            assert 3 <= len(suffix) <= 4, f"suffix {suffix!r} not 3-4 digits"

    def test_factor_mix_combined(self) -> None:
        r = MemoryRegistry(factor=97, mix=True)
        k1 = r.store("550e8400-e29b-41d4-a716-446655440000")
        k2 = r.store("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        assert r.fetch(k1) == "550e8400-e29b-41d4-a716-446655440000"
        assert r.fetch(k2) == "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        assert k1 != "UUID_1"
        assert k2 != "UUID_2"

    def test_factor_mix_idempotent(self) -> None:
        r = MemoryRegistry(factor=97, mix=True)
        k1 = r.store("550e8400-e29b-41d4-a716-446655440000")
        k2 = r.store("550e8400-e29b-41d4-a716-446655440000")
        assert k1 == k2
        assert len(r) == 1


# ---------------------------------------------------------------------------
# FileRegistry
# ---------------------------------------------------------------------------

class TestFileRegistry:
    def test_persistence_plain(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r1 = FileRegistry(path, factor=0, mix=False)
        r1.store("550e8400-e29b-41d4-a716-446655440000")

        r2 = FileRegistry(path, factor=0, mix=False)
        assert r2.fetch("UUID_1") == "550e8400-e29b-41d4-a716-446655440000"
        assert len(r2) == 1

    def test_persistence_default(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r1 = FileRegistry(path)
        key = r1.store("550e8400-e29b-41d4-a716-446655440000")

        r2 = FileRegistry(path)
        assert r2.fetch(key) == "550e8400-e29b-41d4-a716-446655440000"
        assert len(r2) == 1

    def test_counter_recovery_plain(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r1 = FileRegistry(path, factor=0, mix=False)
        r1.store("550e8400-e29b-41d4-a716-446655440000")
        r1.store("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

        r2 = FileRegistry(path, factor=0, mix=False)
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
        r = FileRegistry(path, factor=0, mix=False)
        r.store("550e8400-e29b-41d4-a716-446655440000")
        data = json.loads(path.read_text())
        assert "mappings" in data
        assert data["mappings"]["UUID_1"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_protocol(self, tmp_path) -> None:
        assert isinstance(FileRegistry(tmp_path / "p.json"), UuidRegistry)

    def test_factor_only_persistence(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r1 = FileRegistry(path, factor=97, mix=False)
        k1 = r1.store("550e8400-e29b-41d4-a716-446655440000")
        assert k1 == "UUID_97"

        r2 = FileRegistry(path, factor=97, mix=False)
        assert r2.fetch("UUID_97") == "550e8400-e29b-41d4-a716-446655440000"
        k2 = r2.store("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        assert k2 == "UUID_194"

    def test_factor_mix_counter_recovery(self, tmp_path) -> None:
        path = tmp_path / "reg.json"
        r1 = FileRegistry(path, factor=97, mix=True)
        r1.store("550e8400-e29b-41d4-a716-446655440000")
        r1.store("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

        r2 = FileRegistry(path, factor=97, mix=True)
        assert len(r2) == 2
        k3 = r2.store("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert r2.fetch(k3) == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert len(r2) == 3


# ---------------------------------------------------------------------------
# substitute()
# ---------------------------------------------------------------------------

class TestSubstitute:
    def test_single_uuid_plain(self) -> None:
        text = "User 550e8400-e29b-41d4-a716-446655440000 logged in"
        result, reg = substitute(text, factor=0, mix=False)
        assert result == "User ${UUID_1} logged in"
        assert len(reg) == 1

    def test_single_uuid_default(self) -> None:
        text = "User 550e8400-e29b-41d4-a716-446655440000 logged in"
        result, reg = substitute(text)
        assert "${UUID_" in result
        assert "550e8400" not in result
        assert len(reg) == 1

    def test_multiple_uuids(self) -> None:
        text = "a=550e8400-e29b-41d4-a716-446655440000 b=6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        result, reg = substitute(text)
        assert len(reg) == 2
        # Both placeholders present and distinct
        placeholders = re.findall(r"\$\{UUID_\d+\}", result)
        assert len(placeholders) == 2
        assert placeholders[0] != placeholders[1]

    def test_duplicate_uuids(self) -> None:
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        text = f"{uuid} and {uuid}"
        result, reg = substitute(text)
        placeholders = re.findall(r"\$\{UUID_\d+\}", result)
        assert len(placeholders) == 2
        assert placeholders[0] == placeholders[1]
        assert len(reg) == 1

    def test_no_uuids(self) -> None:
        text = "Hello, world!"
        result, reg = substitute(text)
        assert result == "Hello, world!"
        assert len(reg) == 0

    def test_existing_registry(self) -> None:
        reg = MemoryRegistry()
        k1 = reg.store("550e8400-e29b-41d4-a716-446655440000")
        text = "New: 6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        result, reg = substitute(text, registry=reg)
        assert len(reg) == 2
        assert f"${{{k1}}}" not in result  # first UUID not in this text

    def test_uppercase_uuid(self) -> None:
        text = "ID=550E8400-E29B-41D4-A716-446655440000"
        result, reg = substitute(text, factor=0, mix=False)
        assert result == "ID=${UUID_1}"
        assert reg.fetch("UUID_1") == "550e8400-e29b-41d4-a716-446655440000"

    def test_factor_only(self) -> None:
        text = "User 550e8400-e29b-41d4-a716-446655440000 logged in"
        result, reg = substitute(text, factor=97, mix=False)
        assert result == "User ${UUID_97} logged in"
        assert len(reg) == 1

    def test_factor_mix(self) -> None:
        text = "a=550e8400-e29b-41d4-a716-446655440000 b=6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        result, reg = substitute(text, factor=97, mix=True)
        assert "${UUID_1}" not in result
        assert "${UUID_2}" not in result
        restored = restore(result, reg)
        assert "550e8400-e29b-41d4-a716-446655440000" in restored
        assert "6ba7b810-9dad-11d1-80b4-00c04fd430c8" in restored


# ---------------------------------------------------------------------------
# restore()
# ---------------------------------------------------------------------------

class TestRestore:
    def test_roundtrip(self) -> None:
        original = "User 550e8400-e29b-41d4-a716-446655440000 did thing"
        substituted, reg = substitute(original)
        restored = restore(substituted, reg)
        assert "550e8400-e29b-41d4-a716-446655440000" in restored

    def test_unknown_placeholders_left_unchanged(self) -> None:
        reg = MemoryRegistry()
        text = "Value is ${UUID_99}"
        assert restore(text, reg) == "Value is ${UUID_99}"

    def test_mixed_known_unknown(self) -> None:
        reg = MemoryRegistry()
        key = reg.store("550e8400-e29b-41d4-a716-446655440000")
        text = f"${{{key}}} and ${{UUID_99999}}"
        result = restore(text, reg)
        assert "550e8400-e29b-41d4-a716-446655440000" in result
        assert "${UUID_99999}" in result


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

        assert "550e8400" not in sanitized
        assert "6ba7b810" not in sanitized

        # Extract the placeholders the LLM would see
        placeholders = re.findall(r"\$\{UUID_\d+\}", sanitized)
        assert len(placeholders) == 2

        # Simulate LLM echoing back the placeholders
        llm_response = f"The first order {placeholders[0]} was placed before {placeholders[1]}."

        restored = restore(llm_response, reg)
        assert "550e8400-e29b-41d4-a716-446655440000" in restored
        assert "6ba7b810-9dad-11d1-80b4-00c04fd430c8" in restored
        assert "${UUID_" not in restored
