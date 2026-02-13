#!/usr/bin/env python3
"""Encode UUIDs or integers from stdin to word phrases.

Usage:
  # Encode UUIDs (one per line)
  echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py
  cat uuids.txt | python examples/encode.py --type uuid --vocab 4096

  # Encode integers
  seq 100 105 | python examples/encode.py --type int

  # Token-optimized output (fewer LLM tokens)
  echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py --style token

  # Fixed-width integer encoding with bit-mixing
  seq 0 9 | python examples/encode.py --type u64
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import tiktoken

from id_tokenizer import Codec
from shuffle import factor_encode, int_to_faux_uuid, shuffle_encode

_VALID_VOCABS = {2048, 4096, 8192, 16384, 32768}


def _parse_vocab(v: int) -> int:
    """Accept either a vocab size (2048) or a power of 2 (11)."""
    if 11 <= v <= 15:
        return 1 << v
    if v in _VALID_VOCABS:
        return v
    raise SystemExit(f"error: vocab must be 2048..32768 or 11..15 (as power of 2)")

ENCODINGS = [
    ("cl100k_base", "GPT-4"),
    ("o200k_base", "GPT-4o"),
]


def _encode_numeric(value: str, typ: str, *, style: str, min_digits: int, salt: str, factor: int) -> str:
    import uuid as _uuid

    if typ == "uuid":
        n = int.from_bytes(_uuid.UUID(value).bytes)
    else:
        n = int(value)

    if style == "faux-uuid":
        return int_to_faux_uuid(n, salt=salt)
    if style == "shuffled":
        return shuffle_encode(n, min_digits=min_digits, salt=salt)
    if style == "factor":
        return str(factor_encode(n, factor=factor))

    return str(n)


def main():
    parser = argparse.ArgumentParser(description="Encode values to word phrases")
    parser.add_argument(
        "--type", "-t",
        choices=["uuid", "int", "u32", "u64", "u128"],
        default="uuid",
        help="Input type (default: uuid)",
    )
    parser.add_argument("--vocab", "-v", type=int, default=2048)
    parser.add_argument(
        "--style", "-s",
        choices=["memorable", "token", "numeric", "shuffled", "faux-uuid", "factor"],
        default="memorable",
    )
    parser.add_argument(
        "--separator",
        default="-",
        help="Word separator in output (default: -)",
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=3,
        help="Minimum digits for shuffled output (default: 3)",
    )
    parser.add_argument(
        "--salt",
        default="",
        help="Secret salt for shuffled style (must match at decode time)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=97,
        help="Factor for factor style (default: 97, a prime)",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Show token counts for input and output across tokenizers",
    )
    args = parser.parse_args()
    args.vocab = _parse_vocab(args.vocab)

    is_numeric = args.style in ("numeric", "shuffled", "faux-uuid", "factor")
    c = None if is_numeric else Codec(vocab_size=args.vocab, style=args.style)
    encs = (
        [(tiktoken.get_encoding(name), label) for name, label in ENCODINGS]
        if args.include_debug
        else []
    )

    for line in sys.stdin:
        value = line.strip()
        if not value:
            continue
        try:
            if is_numeric:
                phrase = _encode_numeric(
                    value, args.type,
                    style=args.style,
                    min_digits=args.min_digits,
                    salt=args.salt,
                    factor=args.factor,
                )
            else:
                match args.type:
                    case "uuid":
                        phrase = c.uuid_to_words(value)
                    case "int":
                        phrase = c.int_to_words(int(value))
                    case "u32":
                        from id_tokenizer import encode_u32
                        phrase = encode_u32(int(value), vocab_size=args.vocab, style=args.style)
                    case "u64":
                        from id_tokenizer import encode_u64
                        phrase = encode_u64(int(value), vocab_size=args.vocab, style=args.style)
                    case "u128":
                        from id_tokenizer import encode_u128
                        phrase = encode_u128(int(value), vocab_size=args.vocab, style=args.style)
                if args.separator != "-":
                    phrase = phrase.replace("-", args.separator)
            print(phrase)
            if encs:
                for enc, label in encs:
                    in_tokens = len(enc.encode(value))
                    out_tokens = len(enc.encode(phrase))
                    print(f"  {label}: {in_tokens} -> {out_tokens} tokens", file=sys.stderr)
        except (ValueError, OverflowError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
