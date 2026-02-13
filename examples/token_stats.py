#!/usr/bin/env python3
"""Show token consumption: raw value vs memorable vs token-optimized phrases.

Usage:
  echo "550e8400-e29b-41d4-a716-446655440000" | python examples/token_stats.py
  echo 12312312312 | python examples/token_stats.py
  echo "550e8400-e29b-41d4-a716-446655440000" | python examples/token_stats.py --vocab 4096

Requires: pip install tiktoken
"""

import argparse
import sys

import tiktoken
from id_tokenizer import Codec

ENCODINGS = [
    ("cl100k_base", "GPT-4"),
    ("o200k_base", "GPT-4o"),
]


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def main():
    parser = argparse.ArgumentParser(description="Compare token counts: raw vs encoded")
    parser.add_argument("--vocab", "-v", type=int, default=2048)
    parser.add_argument("value", nargs="?", help="Value to encode (reads stdin if omitted)")
    args = parser.parse_args()

    encs = [(tiktoken.get_encoding(name), label) for name, label in ENCODINGS]
    memorable = Codec(vocab_size=args.vocab, style="memorable")
    token = Codec(vocab_size=args.vocab, style="token")

    if args.value:
        values = [args.value]
    else:
        values = [line.strip() for line in sys.stdin if line.strip()]

    for value in values:
        # Auto-detect UUID vs int
        try:
            import uuid as _uuid
            _uuid.UUID(value)
            m_phrase = memorable.uuid_to_words(value)
            t_phrase = token.uuid_to_words(value)
        except ValueError:
            v = int(value)
            m_phrase = memorable.int_to_words(v)
            t_phrase = token.int_to_words(v)

        # Space-separated variants (tokenizers merge " word" into single tokens)
        m_spaced = m_phrase.replace("-", " ")
        t_spaced = t_phrase.replace("-", " ")

        print(f"  raw:              {value}")
        print(f"  memorable:        {m_phrase}")
        print(f"  token:            {t_phrase}")
        print(f"  token (spaced):   {t_spaced}")
        print()

        header = f"  {'':12s} {'raw':>4s} {'mem -':>6s} {'mem ␣':>6s} {'tok -':>6s} {'tok ␣':>6s}"
        print(header)
        for enc, label in encs:
            raw_n = count_tokens(value, enc)
            mem_h = count_tokens(m_phrase, enc)
            mem_s = count_tokens(m_spaced, enc)
            tok_h = count_tokens(t_phrase, enc)
            tok_s = count_tokens(t_spaced, enc)
            print(f"  {label:12s} {raw_n:4d} {mem_h:6d} {mem_s:6d} {tok_h:6d} {tok_s:6d}")
        print()


if __name__ == "__main__":
    main()
