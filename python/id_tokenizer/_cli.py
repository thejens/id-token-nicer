"""CLI entry point for id-tokenizer (Python).

Thin wrapper around the Rust library exposed via PyO3.

Usage:
  id-tokenizer encode 550e8400-e29b-41d4-a716-446655440000
  id-tokenizer decode "nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh"
  echo 42 | id-tokenizer encode int
"""

import argparse
import sys
import uuid as _uuid

from . import (
    Codec,
    decode_int,
    decode_u32,
    decode_u64,
    decode_u128,
    decode_uuid,
    encode_u32,
    encode_u64,
    encode_u128,
)

KNOWN_TYPES = ("uuid", "int", "u32", "u64", "u128")

VOCAB_CHOICES = (2048, 4096, 8192, 16384, 32768)


def _parse_vocab(s: str) -> int:
    v = int(s)
    if v not in VOCAB_CHOICES:
        raise argparse.ArgumentTypeError(
            f"vocab must be one of {VOCAB_CHOICES}, got {v}"
        )
    return v


def _is_uuid(s: str) -> bool:
    try:
        _uuid.UUID(s)
        return True
    except ValueError:
        return False


def _detect_type(value: str) -> str:
    return "uuid" if _is_uuid(value) else "int"


def _do_encode(value: str, kind: str, vocab: int, style: str) -> str:
    c = Codec(vocab_size=vocab, style=style)
    match kind:
        case "uuid":
            return c.uuid_to_words(value)
        case "int":
            return c.int_to_words(int(value))
        case "u32":
            return encode_u32(int(value), vocab_size=vocab, style=style)
        case "u64":
            return encode_u64(int(value), vocab_size=vocab, style=style)
        case "u128":
            return encode_u128(int(value), vocab_size=vocab, style=style)
        case _:
            raise ValueError(f"unknown type: {kind}")


def _do_decode(phrase: str, kind: str, vocab: int, style: str) -> str:
    match kind:
        case "uuid":
            return decode_uuid(phrase, vocab_size=vocab, style=style)
        case "int":
            return str(decode_int(phrase, vocab_size=vocab, style=style))
        case "u32":
            return str(decode_u32(phrase, vocab_size=vocab, style=style))
        case "u64":
            return str(decode_u64(phrase, vocab_size=vocab, style=style))
        case "u128":
            return str(decode_u128(phrase, vocab_size=vocab, style=style))
        case _:
            raise ValueError(f"unknown type: {kind}")


def _try_decode_auto(phrase: str, vocab: int, style: str) -> str:
    try:
        return decode_uuid(phrase, vocab_size=vocab, style=style)
    except ValueError:
        pass
    return str(decode_int(phrase, vocab_size=vocab, style=style))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="id-tokenizer",
        description="Encode UUIDs and integers as word phrases",
    )
    parser.add_argument(
        "command",
        choices=["encode", "decode"],
        help="encode or decode",
    )
    parser.add_argument(
        "type",
        nargs="?",
        help="uuid, int, u32, u64, u128 (auto-detected if omitted)",
    )
    parser.add_argument(
        "value",
        nargs="?",
        help="value to encode/decode (reads stdin if omitted)",
    )
    parser.add_argument(
        "--vocab", "-v",
        type=_parse_vocab,
        default=2048,
        help="vocabulary size (default: 2048)",
    )
    parser.add_argument(
        "--style", "-s",
        choices=["memorable", "token"],
        default="memorable",
        help="word list style (default: memorable)",
    )
    args = parser.parse_args(argv)

    # Disambiguate positional args: [type] [value] vs [value]
    explicit_type = None
    inline_value = None
    if args.type is not None and args.type in KNOWN_TYPES:
        explicit_type = args.type
        inline_value = args.value
    elif args.type is not None:
        # args.type is actually the value
        inline_value = args.type
        if args.value is not None:
            parser.error(f"unknown type '{args.type}'. Known types: {', '.join(KNOWN_TYPES)}")

    def _process(value: str) -> None:
        if args.command == "encode":
            kind = explicit_type or _detect_type(value)
            print(_do_encode(value, kind, args.vocab, args.style))
        else:
            if explicit_type:
                print(_do_decode(value, explicit_type, args.vocab, args.style))
            else:
                print(_try_decode_auto(value, args.vocab, args.style))

    try:
        if inline_value:
            _process(inline_value)
        else:
            for line in sys.stdin:
                value = line.strip()
                if value:
                    _process(value)
    except (ValueError, OverflowError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
