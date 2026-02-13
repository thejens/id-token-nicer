#!/usr/bin/env python3
"""Batch encode/decode with tab-separated input/output.

Reads tab-separated lines: <id>\t<value>
Outputs:                    <id>\t<phrase>

Useful for processing CSV/TSV data where you want to keep row context.

Usage:
  # Encode a TSV of (row_id, uuid) pairs
  printf "row1\t550e8400-e29b-41d4-a716-446655440000\nrow2\tde305d54-75b4-431b-adb2-eb6b9e546014\n" \
    | python examples/batch.py encode uuid

  # Decode back
  printf "row1\tnil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh\n" \
    | python examples/batch.py decode uuid

  # Works with cut/paste for CSV pipelines
  cut -f1,3 data.tsv | python examples/batch.py encode uuid --style token
"""

import argparse
import sys

from id_tokenizer import Codec, encode_u32, encode_u64, encode_u128
from id_tokenizer import decode_u32, decode_u64, decode_u128


def main():
    parser = argparse.ArgumentParser(description="Batch encode/decode TSV data")
    parser.add_argument("action", choices=["encode", "decode"])
    parser.add_argument(
        "type",
        choices=["uuid", "int", "u32", "u64", "u128"],
    )
    parser.add_argument("--vocab", "-v", type=int, default=2048)
    parser.add_argument("--style", "-s", choices=["memorable", "token"], default="memorable")
    parser.add_argument("--delimiter", "-d", default="\t", help="Field delimiter (default: tab)")
    args = parser.parse_args()

    c = Codec(vocab_size=args.vocab, style=args.style)
    errors = 0

    for lineno, line in enumerate(sys.stdin, 1):
        line = line.rstrip("\n")
        if not line:
            continue

        parts = line.split(args.delimiter)
        if len(parts) < 2:
            print(f"error: line {lineno}: expected at least 2 fields", file=sys.stderr)
            errors += 1
            continue

        key, value = parts[0], parts[1]

        try:
            if args.action == "encode":
                match args.type:
                    case "uuid":
                        result = c.uuid_to_words(value)
                    case "int":
                        result = c.int_to_words(int(value))
                    case "u32":
                        result = encode_u32(int(value), vocab_size=args.vocab, style=args.style)
                    case "u64":
                        result = encode_u64(int(value), vocab_size=args.vocab, style=args.style)
                    case "u128":
                        result = encode_u128(int(value), vocab_size=args.vocab, style=args.style)
            else:
                match args.type:
                    case "uuid":
                        result = c.words_to_uuid(value)
                    case "int":
                        result = str(c.words_to_int(value))
                    case "u32":
                        result = str(decode_u32(value, vocab_size=args.vocab, style=args.style))
                    case "u64":
                        result = str(decode_u64(value, vocab_size=args.vocab, style=args.style))
                    case "u128":
                        result = str(decode_u128(value, vocab_size=args.vocab, style=args.style))
            print(f"{key}{args.delimiter}{result}")
        except (ValueError, OverflowError) as e:
            print(f"error: line {lineno}: {e}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"{errors} error(s)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
