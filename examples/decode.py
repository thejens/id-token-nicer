#!/usr/bin/env python3
"""Decode word phrases from stdin back to UUIDs or integers.

Usage:
  # Decode UUID phrases (one per line)
  echo "nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh" | python examples/decode.py
  cat phrases.txt | python examples/decode.py --type uuid --vocab 4096

  # Decode integers
  echo "fez" | python examples/decode.py --type int

  # Token-optimized phrases
  echo "nil-azzo-fun-also-oop-ized-eps-lsa-apk-hai-ived-ddb" | python examples/decode.py --style token

  # Roundtrip test
  echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py | python examples/decode.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from id_tokenizer import Codec
from shuffle import factor_decode, faux_uuid_to_int, shuffle_decode


def main():
    parser = argparse.ArgumentParser(description="Decode word phrases to values")
    parser.add_argument(
        "--type", "-t",
        choices=["uuid", "int", "u32", "u64", "u128"],
        default="uuid",
        help="Output type (default: uuid)",
    )
    parser.add_argument("--vocab", "-v", type=int, default=2048)
    parser.add_argument(
        "--style", "-s",
        choices=["memorable", "token", "numeric", "shuffled", "faux-uuid", "factor"],
        default="memorable",
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=3,
        help="Minimum digits used at encode time (default: 3)",
    )
    parser.add_argument(
        "--salt",
        default="",
        help="Secret salt (must match encode time)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=97,
        help="Factor for factor style (default: 97, must match encode time)",
    )
    args = parser.parse_args()

    is_numeric = args.style in ("numeric", "shuffled", "faux-uuid", "factor")
    c = None if is_numeric else Codec(vocab_size=args.vocab, style=args.style)

    for line in sys.stdin:
        phrase = line.strip()
        if not phrase:
            continue
        try:
            if is_numeric:
                if args.style == "faux-uuid":
                    print(faux_uuid_to_int(phrase, salt=args.salt))
                    continue
                if args.style == "factor":
                    print(factor_decode(int(phrase), factor=args.factor))
                    continue
                import uuid as _uuid
                if args.style == "shuffled":
                    n = shuffle_decode(phrase, min_digits=args.min_digits, salt=args.salt)
                else:
                    n = int(phrase)
                match args.type:
                    case "uuid":
                        print(_uuid.UUID(int=n))
                    case _:
                        print(n)
            else:
                match args.type:
                    case "uuid":
                        print(c.words_to_uuid(phrase))
                    case "int":
                        print(c.words_to_int(phrase))
                    case "u32":
                        from id_tokenizer import decode_u32
                        print(decode_u32(phrase, vocab_size=args.vocab, style=args.style))
                    case "u64":
                        from id_tokenizer import decode_u64
                        print(decode_u64(phrase, vocab_size=args.vocab, style=args.style))
                    case "u128":
                        from id_tokenizer import decode_u128
                        print(decode_u128(phrase, vocab_size=args.vocab, style=args.style))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
