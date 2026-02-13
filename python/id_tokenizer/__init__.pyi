"""Type stubs for id_tokenizer (native Rust module via PyO3)."""

from typing import Optional

class Codec:
    """A codec instance bound to a specific vocabulary size and style."""

    vocab_size: int
    bits_per_word: int
    style: str

    def __init__(
        self, vocab_size: int = 2048, style: str = "memorable"
    ) -> None: ...
    def int_to_words(self, value: int) -> str: ...
    def words_to_int(self, phrase: str) -> int: ...
    def uuid_to_words(self, uuid_str: str) -> str: ...
    def words_to_uuid(self, phrase: str) -> str: ...
    def uuid_word_count(self) -> int: ...
    def word_count(self, payload_bytes: int) -> int: ...

def encode_uuid(
    uuid_str: str,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> str: ...
def decode_uuid(
    phrase: str,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> str: ...
def encode_int(
    value: int,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> str: ...
def decode_int(
    phrase: str,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> int: ...
def encode_u128(
    value: int,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> str: ...
def decode_u128(
    phrase: str,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> int: ...
def encode_u64(
    value: int,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> str: ...
def decode_u64(
    phrase: str,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> int: ...
def encode_u32(
    value: int,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> str: ...
def decode_u32(
    phrase: str,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> int: ...
def encode_bytes(
    data: bytes,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> str: ...
def decode_bytes(
    phrase: str,
    expected_len: int,
    vocab_size: Optional[int] = None,
    style: Optional[str] = None,
) -> bytes: ...
def word_count(
    payload_bytes: int,
    vocab_size: Optional[int] = None,
) -> int: ...
def mix_value(value: int, bits: int) -> int: ...
def unmix_value(value: int, bits: int) -> int: ...
