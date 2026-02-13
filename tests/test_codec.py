"""Comprehensive tests for id_tokenizer Python bindings."""

import uuid
import random
import pytest
from id_tokenizer import Codec, encode_uuid, decode_uuid, encode_int, decode_int

VOCAB_SIZES = [2048, 4096, 8192, 16384, 32768]


class TestCodecClass:
    """Test the Codec class interface."""

    def test_constructor(self) -> None:
        c = Codec()
        assert c.vocab_size == 2048
        assert c.bits_per_word == 11

    def test_constructor_all_sizes(self) -> None:
        expected_bpw = {2048: 11, 4096: 12, 8192: 13, 16384: 14, 32768: 15}
        for vs, bpw in expected_bpw.items():
            c = Codec(vocab_size=vs)
            assert c.vocab_size == vs
            assert c.bits_per_word == bpw

    def test_invalid_vocab_size(self) -> None:
        with pytest.raises(ValueError):
            Codec(vocab_size=1024)

    def test_repr(self) -> None:
        assert repr(Codec(4096)) == 'Codec(vocab_size=4096, style="memorable")'
        assert repr(Codec(2048, style="token")) == 'Codec(vocab_size=2048, style="token")'


class TestUuidRoundtrip:
    """Round-trip: random UUIDs -> encode -> decode -> same UUID."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_roundtrip_10k(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        for _ in range(10_000):
            u = str(uuid.uuid4())
            phrase = c.uuid_to_words(u)
            assert c.words_to_uuid(phrase) == u

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_known_uuid(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        u = "550e8400-e29b-41d4-a716-446655440000"
        assert c.words_to_uuid(c.uuid_to_words(u)) == u

    def test_uuid_word_counts(self) -> None:
        expected = {2048: 12, 4096: 11, 8192: 11, 16384: 10, 32768: 9}
        for vs, n in expected.items():
            c = Codec(vs)
            assert c.uuid_word_count() == n


class TestMixingAvalanche:
    """Adjacent inputs should produce completely different word sequences."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_adjacent_uuids_differ(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        base = "550e8400-e29b-41d4-a716-44665544"
        phrases = [c.uuid_to_words(f"{base}{i:04x}") for i in range(10)]
        # All phrases should be unique
        assert len(set(phrases)) == 10
        # Adjacent phrases should share very few words
        for i in range(len(phrases) - 1):
            w1 = set(phrases[i].split("-"))
            w2 = set(phrases[i + 1].split("-"))
            shared = len(w1 & w2)
            total = len(w1)
            assert shared < total / 2, f"Too many shared words: {shared}/{total}"

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_sequential_u32_differ(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        # Fixed-width u32 encoding uses mixing
        from id_tokenizer import encode_u32
        phrases = [encode_u32(i, vocab_size) for i in range(10)]
        assert len(set(phrases)) == 10


class TestIntVariableLength:
    """Test variable-length integer encoding."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_roundtrip(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        for val in [0, 1, 42, 255, 1000, 65535, 2**32 - 1, 2**64 - 1, 2**128 - 1]:
            phrase = c.int_to_words(val)
            assert c.words_to_int(phrase) == val

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_compact(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        # Small values should use fewer words
        small_words = len(c.int_to_words(42).split("-"))
        large_words = len(c.int_to_words(2**128 - 1).split("-"))
        assert small_words < large_words

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_zero_is_one_word(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        assert len(c.int_to_words(0).split("-")) == 1

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_random_1000(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        for _ in range(1000):
            val = random.randint(0, 2**64 - 1)
            assert c.words_to_int(c.int_to_words(val)) == val

    def test_module_level_functions(self) -> None:
        phrase = encode_int(42)
        assert decode_int(phrase) == 42


class TestDeterminism:
    """Same input always produces the same output."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_uuid_deterministic(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        u = "550e8400-e29b-41d4-a716-446655440000"
        results = {c.uuid_to_words(u) for _ in range(100)}
        assert len(results) == 1

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_int_deterministic(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        results = {c.int_to_words(42) for _ in range(100)}
        assert len(results) == 1


class TestMutation:
    """Checksum should catch mutations."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_single_char_mutation_detection(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        detected = 0
        total = 0
        rng = random.Random(42)

        for _ in range(1000):
            u = str(uuid.uuid4())
            phrase = c.uuid_to_words(u)
            words = phrase.split("-")

            word_idx = rng.randint(0, len(words) - 1)
            word = words[word_idx]
            if len(word) < 2:
                continue

            char_idx = rng.randint(0, len(word) - 1)
            old_char = word[char_idx]
            new_char = chr(((ord(old_char) - ord("a") + rng.randint(1, 25)) % 26) + ord("a"))
            words[word_idx] = word[:char_idx] + new_char + word[char_idx + 1:]

            total += 1
            try:
                result = c.words_to_uuid("-".join(words))
                if result != u:
                    detected += 1
            except ValueError:
                detected += 1

        rate = detected / total
        assert rate > 0.90, f"Detection rate too low: {rate:.4f}"


class TestErrors:
    """Test error handling."""

    def test_wrong_word_count(self) -> None:
        c = Codec()
        with pytest.raises(ValueError, match="expected.*words"):
            c.words_to_uuid("one-two-three")

    def test_nil_uuid_rejected(self) -> None:
        c = Codec()
        phrase = c.uuid_to_words("00000000-0000-0000-0000-000000000000")
        with pytest.raises(ValueError, match="invalid UUID"):
            c.words_to_uuid(phrase)

    def test_empty_int_phrase(self) -> None:
        c = Codec()
        with pytest.raises(ValueError, match="empty"):
            c.words_to_int("")


class TestSeparators:
    """Test separator handling."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_whitespace_separator(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        u = "550e8400-e29b-41d4-a716-446655440000"
        phrase = c.uuid_to_words(u)
        spaced = phrase.replace("-", " ")
        assert c.words_to_uuid(spaced) == u

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_case_insensitive(self, vocab_size: int) -> None:
        c = Codec(vocab_size)
        u = "550e8400-e29b-41d4-a716-446655440000"
        phrase = c.uuid_to_words(u)
        assert c.words_to_uuid(phrase.upper()) == u


class TestModuleFunctions:
    """Test backward-compat module-level functions."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_encode_decode_uuid(self, vocab_size: int) -> None:
        u = "550e8400-e29b-41d4-a716-446655440000"
        phrase = encode_uuid(u, vocab_size=vocab_size)
        assert decode_uuid(phrase, vocab_size=vocab_size) == u


class TestTokenStyle:
    """Test token-optimized word list mode."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_uuid_roundtrip(self, vocab_size: int) -> None:
        c = Codec(vocab_size, style="token")
        for _ in range(1000):
            u = str(uuid.uuid4())
            phrase = c.uuid_to_words(u)
            assert c.words_to_uuid(phrase) == u

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_int_roundtrip(self, vocab_size: int) -> None:
        c = Codec(vocab_size, style="token")
        for val in [0, 1, 42, 255, 1000, 65535, 2**32 - 1, 2**64 - 1, 2**128 - 1]:
            phrase = c.int_to_words(val)
            assert c.words_to_int(phrase) == val

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_different_words_from_memorable(self, vocab_size: int) -> None:
        m = Codec(vocab_size, style="memorable")
        t = Codec(vocab_size, style="token")
        u = "550e8400-e29b-41d4-a716-446655440000"
        assert m.uuid_to_words(u) != t.uuid_to_words(u)

    def test_style_property(self) -> None:
        assert Codec(2048, style="token").style == "token"
        assert Codec(2048).style == "memorable"

    def test_module_functions_with_style(self) -> None:
        u = "550e8400-e29b-41d4-a716-446655440000"
        phrase = encode_uuid(u, style="token")
        assert decode_uuid(phrase, style="token") == u

    def test_invalid_style(self) -> None:
        with pytest.raises(ValueError, match="style"):
            Codec(2048, style="invalid")
