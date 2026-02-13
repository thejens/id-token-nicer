# id-tokenizer

Encode UUIDs and integers as word sequences that LLMs can handle without hallucinating.

```
550e8400-e29b-41d4-a716-446655440000
  --> nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh
```

## Why

UUIDs are unfortunately quite long. A v4 UUID like `550e8400-e29b-41d4-a716-446655440000` is 36 characters of hex and dashes -- visually dense, easy to confuse, and expensive in LLM token budgets. But the actual information content is only 122 bits. What if we used a vocabulary of 2048 words (11 bits each) to describe them instead? That's 12 words for a UUID, and each word is something an LLM can reason about as a discrete unit rather than a string of hex nibbles.

This library exists because LLM agents have specific problems with identifiers:

**Agents hallucinate IDs.** When an LLM sees `550e8400-e29b-41d4-a716-446655440000` in a conversation, it may later reproduce it as `550e8400-e29b-41d4-a716-446655440001` -- a single character off, completely valid-looking, pointing at nothing. Word phrases like `nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh` are harder to hallucinate convincingly. Each word is a token the model has seen millions of times; inventing a plausible but wrong combination is much less likely than flipping a hex digit. And if the model does hallucinate a word, the checksum catches it. For plain integer IDs, factor encoding (multiply by a prime like 97) means a randomly invented number has only a ~1% chance of passing validation. None of these prevent an agent from *copying* a valid ID and using it in the wrong context -- that requires application-level checks -- but they catch outright fabrication.

**IDs eat tokens.** A raw UUID costs 18-23 tokens across GPT-4/4o tokenizers. Hyphenated word phrases are roughly token-neutral (~23 tokens) -- but each word is a discrete semantic unit the model can attend to, rather than a hex substring that might get split across token boundaries. Where tokens matter most, the numeric style converts UUIDs to plain integers (13 tokens, a ~40% reduction). And the token-optimized word list ensures that words never cost *more* than one token each, so the encoding is predictable: word count = token count.

**Sequential IDs leak patterns.** If an agent sees database rows 1, 2, 3, it will happily predict row 4 exists -- and it might be right, or it might hallucinate a row that was deleted. Feistel bit-mixing turns sequential inputs into scattered outputs, breaking the pattern without adding bits. The same mixing, applied with a secret salt, prevents leaking information like database sizes through URL parameters. You can even map integer IDs to faux UUIDs -- `1` becomes `7a7c5d13-e91d-5c3d-1715-ad59d15b33c8` -- giving integer-indexed databases a UUID-shaped public API without changing the underlying schema.

**Encoding and decoding must be fast.** Agents call tools in tight loops. Encoding is O(n) in the number of words (one hash per word, no search). Decoding is O(n) too -- each word hashes directly to its index, no dictionary lookup needed. The word lists are constructed so that `xxh64(word) & (vocab_size - 1) == slot_index`, making decode a pure arithmetic operation.

### Design goals

1. **Reduce hallucination** -- words are harder to hallucinate than hex; checksums catch errors; bloom filters catch typos early
2. **Predictable token cost** -- each word is exactly one token in GPT-4, GPT-4o, Gemma, Llama 3, and more; numeric mode cuts UUID tokens by ~40%
3. **O(n) encode/decode** -- no search, no lookup tables at decode time, just hashing
4. **Multiple modes** -- `memorable` for human-readable, `token` for LLM-optimized, `shuffled` for pattern-hiding
5. **Work for any ID type** -- UUIDs, integers, fixed-width IDs, arbitrary bytes
6. **Hide sequential patterns** -- Feistel mixing and salted shuffling prevent agents (and users) from guessing adjacent IDs
7. **Faux UUIDs** -- map integer IDs to UUID-formatted values with a secret salt, giving integer-indexed databases a UUID-shaped public API

## Features

- **UUID-aware encoding** -- strips RFC 4122 variant bits for a stronger checksum, validates version on decode
- **Variable-length integer encoding** -- `42` encodes to 1 word, `u64::MAX` to 7 words
- **Bit-mixing (Feistel network)** -- adjacent UUIDs and sequential IDs produce completely different phrases
- **Salted integer shuffling** -- digit-preserving permutation with a secret salt, so `1, 2, 3` become `597, 341, 148` and outsiders can't reverse it or infer database size
- **Factor encoding** -- multiply IDs by a prime (default 97) so hallucinated numbers fail a divisibility check; with factor 97, a random guess has ~1% chance of passing
- **Faux UUIDs** -- map any integer to a UUID-formatted string via salted 128-bit Feistel, giving integer-indexed databases UUID-shaped public IDs
- **5 vocabulary sizes** -- trade word count for vocabulary size (2048 to 32768 words)
- **2 word list styles** -- `memorable` (real English words) or `token` (LLM token-optimized)
- **Bloom filter** -- catches typos before the checksum is even checked
- **Profanity-filtered** -- curated against LDNOOBW + Google profanity lists
- **Rust core + Python bindings** -- all hashing, mixing, and lookup happens in Rust via PyO3

## Installation

### Python (from source)

Requires Rust 1.80+ and Python 3.9+.

```sh
pip install maturin
maturin develop --release
```

This installs both the `id_tokenizer` Python library and the `id-tokenizer` CLI command.

### Rust (CLI only)

```sh
cargo install --path .
```

## CLI

The `id-tokenizer` command is available after installation. It auto-detects UUIDs vs integers when the type is omitted.

```sh
# Encode a UUID
id-tokenizer encode 550e8400-e29b-41d4-a716-446655440000
# nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh

# Decode back
id-tokenizer decode "nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh"
# 550e8400-e29b-41d4-a716-446655440000

# Token-optimized style (single tokens in GPT-4, GPT-4o, Gemma, Llama 3)
id-tokenizer encode 550e8400-e29b-41d4-a716-446655440000 --style token
# nil-azzo-fun-also-oop-ized-eps-lsa-apk-hai-ived-ddb

# Variable-length integer
id-tokenizer encode int 42
# fez

# Fixed-width with bit-mixing (sequential IDs produce scattered phrases)
id-tokenizer encode u64 12345
# davy-mird-oak-twa-pan-tur-uji

# Larger vocab = fewer words
id-tokenizer encode 550e8400-e29b-41d4-a716-446655440000 --vocab 32768
# (9 words instead of 12)

# Pipe from stdin
echo "550e8400-e29b-41d4-a716-446655440000" | id-tokenizer encode
cat uuids.txt | id-tokenizer encode --style token
seq 100 105 | id-tokenizer encode int
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--vocab N` / `-v N` | Vocabulary size: 2048, 4096, 8192, 16384, 32768 | 2048 |
| `--style S` / `-s S` | Word list: `memorable` or `token` | memorable |

### Encoding types

| Type | Description | When to use |
|------|-------------|-------------|
| `uuid` | UUID with variant stripping + version validation | UUIDs (auto-detected) |
| `int` | Variable-length, compact, no zero-padding | Small to large integers (auto-detected) |
| `u32` | Fixed-width 32-bit with Feistel mixing | Database IDs where sequential pattern hiding matters |
| `u64` | Fixed-width 64-bit with Feistel mixing | 64-bit IDs |
| `u128` | Fixed-width 128-bit with Feistel mixing | 128-bit IDs |

## Python library

```python
from id_tokenizer import Codec

c = Codec(vocab_size=2048)  # or 4096, 8192, 16384, 32768

# UUID <-> words
phrase = c.uuid_to_words("550e8400-e29b-41d4-a716-446655440000")
# "nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh"
uuid_str = c.words_to_uuid(phrase)
# "550e8400-e29b-41d4-a716-446655440000"

# Integer <-> words (variable length, compact)
phrase = c.int_to_words(42)       # "fez"
val    = c.words_to_int(phrase)   # 42

# Token-optimized mode (for LLM contexts)
t = Codec(vocab_size=2048, style="token")
phrase = t.uuid_to_words("550e8400-e29b-41d4-a716-446655440000")
# Each word is a single token in GPT-4, GPT-4o, Gemma, and Llama 3

# Helpers
c.uuid_word_count()   # 12 for V2048, 9 for V32768
c.word_count(8)       # words needed for 8-byte payload
```

Module-level convenience functions are also available:

```python
from id_tokenizer import encode_uuid, decode_uuid, encode_int, decode_int

phrase = encode_uuid("550e8400-e29b-41d4-a716-446655440000", vocab_size=4096)
uuid_str = decode_uuid(phrase, vocab_size=4096)

phrase = encode_int(42)
val = decode_int(phrase)

# Fixed-width with bit-mixing
from id_tokenizer import encode_u64, decode_u64
phrase = encode_u64(12345)  # Feistel-mixed, fixed-width
val = decode_u64(phrase)    # 12345

# Arbitrary bytes
from id_tokenizer import encode_bytes, decode_bytes
phrase = encode_bytes(b"\x01\x02\x03\x04")
data = decode_bytes(phrase, expected_len=4)
```

## Rust library

```rust
use id_tokenizer::{Vocab, Style, encode_uuid, decode_uuid, encode_int, decode_int};

let phrase = encode_uuid(uuid_bytes, Vocab::V2048, Style::Memorable);
let bytes  = decode_uuid(&phrase, Vocab::V2048, Style::Memorable)?;

// Token-optimized
let phrase = encode_uuid(uuid_bytes, Vocab::V2048, Style::Token);

let phrase = encode_int(42u128, Vocab::V2048, Style::Memorable);  // "fez"
let val    = decode_int(&phrase, Vocab::V2048, Style::Memorable)?; // 42
```

## Word list styles

| Style | Description | Best for |
|-------|-------------|----------|
| `memorable` | Real English words optimized for human readability | Verbal communication, manual transcription |
| `token` | Words optimized to be single tokens in LLM tokenizers | LLM prompts, AI pipelines, context window efficiency |

### Token-optimized mode

The `token` style selects words that encode as **single tokens** in common LLM tokenizers. Words are sourced from the intersection of 4 tokenizers:

- **cl100k_base** (GPT-4, GPT-3.5-turbo) -- also covers Phi-4 (identical tokenizer)
- **o200k_base** (GPT-4o, GPT-4o-mini)
- **Gemma** (Gemini family)
- **Llama 3** -- cl100k is a subset of Llama 3, Qwen 2.5, and DeepSeek, so these are covered too

**Token coverage by vocabulary size:**

| Vocab | All 4 tokenizers | Some tokenizers | Dictionary | Generated | Total single-token |
|------:|-----------------:|----------------:|-----------:|----------:|-------------------:|
| 2048  | 2041             | 7               | 0          | 0         | **100.0%**         |
| 4096  | 3852             | 244             | 0          | 0         | **100.0%**         |
| 8192  | 6085             | 2066            | 35         | 6         | **99.5%**          |
| 16384 | 8029             | 7172            | 745        | 438       | **92.8%**          |
| 32768 | 9361             | 14707           | 3277       | 5423      | **73.4%**          |

"All 4 tokenizers" = single token in cl100k, o200k, Gemma, and Llama 3. "Some tokenizers" = single token in at least one. Words that fragment into 3+ tokens in hyphen context (`-word`) are rejected to ensure clean tokenization of hyphen-separated phrases.

**Token cost comparison (V2048, UUID):**

| Format | GPT-4 | GPT-4o | Notes |
|--------|------:|-------:|-------|
| Raw UUID | 18-23 | 18-22 | varies by UUID value |
| Words (hyphenated) | 23 | 22-23 | token-neutral, but each word = 1 token |
| Words (space-separated) | 18 | 15-17 | `" word"` merges into single tokens |
| Numeric (integer) | 13 | 13 | **~40% fewer tokens** |

The token-optimized word list doesn't primarily save tokens over raw UUIDs -- it makes the token cost **predictable** (exactly 1 token per word) and gives the model discrete semantic units to attend to instead of hex substrings split across BPE boundaries. Where raw token count matters most, the numeric or shuffled styles offer the real savings.

## Vocabulary sizes

| Vocab | Bits/word | UUID words | u64 words | u32 words | Check bits (UUID) |
|------:|----------:|-----------:|----------:|----------:|------------------:|
| 2048  | 11        | 12         | 7         | 4         | 4                 |
| 4096  | 12        | 11         | 6         | 3         | 4                 |
| 8192  | 13        | 11         | 6         | 3         | 15                |
| 16384 | 14        | 10         | 6         | 3         | 12                |
| 32768 | 15        | 9          | 5         | 3         | 7                 |

Larger vocabularies use fewer words per encoding but require more memory and contain longer/rarer words.

## Examples

All examples below show real outputs. The `examples/` directory has pipe-friendly scripts with additional styles beyond the core library.

### Word encoding (memorable)

Human-readable phrases built from real English words. Best when identifiers are spoken aloud, pasted in tickets, or shared in chat.

```sh
$ echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py
nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh

$ echo "nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh" | python examples/decode.py
550e8400-e29b-41d4-a716-446655440000
```

### Word encoding (token-optimized)

Every word is a single token in GPT-4, GPT-4o, Gemma, and Llama 3. Best when identifiers live inside LLM prompts and you want predictable, minimal token cost per word.

```sh
$ echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py --style token
nil-azzo-fun-also-oop-ized-eps-lsa-apk-hai-ived-ddb

$ echo "nil-azzo-fun-also-oop-ized-eps-lsa-apk-hai-ived-ddb" | python examples/decode.py --style token
550e8400-e29b-41d4-a716-446655440000
```

### Larger vocabularies = fewer words

Trade vocabulary size for shorter phrases. V4096 gives 11 words per UUID; V32768 gives 9.

```sh
$ echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py --vocab 4096
moxo-cocci-hyp-wem-wei-tiou-lors-ordu-waf-creak-cosh

$ echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py --vocab 32768
all-ecize-vejovis-minos-abb-allseed-heretic-signum-archhead
```

### Integer encoding

Variable-length: small numbers get fewer words. Best for encoding compact integer IDs.

```sh
$ echo "42" | python examples/encode.py -t int
fez

$ seq 100 105 | python examples/encode.py -t int
bale
call
axal
quop
rest
jur
```

### Fixed-width integers with bit mixing

The `u64` type applies a Feistel network so that sequential inputs produce scattered, unrelated phrases. Best when you want to hide sequential patterns in auto-incrementing database IDs.

```sh
$ seq 0 4 | python examples/encode.py -t u64
losh-hong-mica-gen-cup-yom-tad
teil-may-tiza-act-bad-path-hub
orna-peel-kulm-oft-fud-davy-asp
boat-ike-dub-vug-merl-eden-ared
hem-ima-blur-ant-tra-urn-leda
```

Note how 0, 1, 2, 3, 4 produce completely different phrases -- an observer can't tell these are adjacent IDs.

### Numeric style

Converts a UUID to its raw 128-bit integer. No words, just digits. Best when you need the smallest possible token footprint (~40% fewer tokens than hex UUIDs) and don't need human readability.

```sh
$ echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py --style numeric
113059749145936325402354257176981405696
```

### Shuffled integers

Digit-preserving Feistel permutation with a secret salt. A 3-digit number shuffles to another 3-digit number. Best for hiding auto-increment IDs in URLs or APIs without changing the column type.

```sh
$ seq 1 5 | python examples/encode.py -t int --style shuffled --salt mykey
152
895
213
729
848
```

The salt makes the mapping secret -- someone who knows the algorithm but not the salt can't reverse it:

```sh
$ seq 1 5 | python examples/encode.py -t int --style shuffled --salt different-key
998
062
746
356
225
```

Decoding requires the same salt:

```sh
$ echo "152" | python examples/decode.py -t int --style shuffled --salt mykey
1
```

### Minimum digit width

Small numbers (1-2 digits) are zero-padded into a wider shuffle range. Best when you want all IDs to be at least N digits long, e.g., for display consistency.

```sh
$ echo "5" | python examples/encode.py -t int --style shuffled --min-digits 6 --salt mykey
562083
```

### Faux UUIDs

Map any integer to a UUID-formatted string via salted 128-bit Feistel. Best for giving integer-indexed databases a UUID-shaped public API without changing the underlying schema.

```sh
$ seq 1 5 | python examples/encode.py -t int --style faux-uuid --salt mykey
7a7c5d13-e91d-5c3d-1715-ad59d15b33c8
409e750f-3a2d-df86-a50a-a9dfda7eec52
fbb70dd5-37ae-bb71-4e93-f7add771148b
f6bc37cc-2a15-25ad-eb28-510d60c8156b
d5896442-1c0a-3775-c4a7-95d0c02271f4

$ echo "7a7c5d13-e91d-5c3d-1715-ad59d15b33c8" \
    | python examples/decode.py --style faux-uuid --salt mykey
1
```

With the wrong salt, decode returns a different (wrong) integer -- it doesn't error, it just gives the wrong answer:

```sh
$ echo "7a7c5d13-e91d-5c3d-1715-ad59d15b33c8" \
    | python examples/decode.py --style faux-uuid --salt wrong-key
176507643592523770348173476994809259721
```

### Factor encoding

Multiply each ID by a prime number. On decode, a divisibility check catches hallucinated values. With the default factor of 97, a randomly guessed integer has only a ~1% chance of passing validation. Use a larger prime for stronger guarantees.

```sh
$ seq 1 5 | python examples/encode.py -t int --style factor
97
194
291
388
485

$ echo "291" | python examples/decode.py -t int --style factor
3

$ echo "292" | python examples/decode.py -t int --style factor
error: invalid ID: 292 is not divisible by the factor
```

Use `--factor` to choose a different prime:

```sh
$ seq 1 5 | python examples/encode.py -t int --style factor --factor 31
31
62
93
124
155
```

Note that this catches *invented* IDs but not *copied* ones -- an agent can still hallucinate by reusing a valid ID it saw earlier in the wrong context. The same limitation applies to word phrases: an agent can recombine words it knows are valid, or copy an entire phrase verbatim. Factor encoding catches random fabrication, not misattribution.

### Custom separator

Change the word separator for output formatting. Spaces save tokens in some tokenizers.

```sh
$ echo "550e8400-e29b-41d4-a716-446655440000" | python examples/encode.py --style token --separator " "
nil azzo fun also oop ized eps lsa apk hai ived ddb
```

### Token count debugging

The `--include-debug` flag prints token counts to stderr for GPT-4 and GPT-4o tokenizers. Useful for benchmarking token cost across styles.

```sh
$ echo "550e8400-e29b-41d4-a716-446655440000" \
    | python examples/encode.py --style memorable --include-debug
nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh
  GPT-4: 18 -> 23 tokens
  GPT-4o: 18 -> 22 tokens

$ echo "550e8400-e29b-41d4-a716-446655440000" \
    | python examples/encode.py --style token --include-debug
nil-azzo-fun-also-oop-ized-eps-lsa-apk-hai-ived-ddb
  GPT-4: 18 -> 23 tokens
  GPT-4o: 18 -> 23 tokens

$ echo "550e8400-e29b-41d4-a716-446655440000" \
    | python examples/encode.py --style token --separator " " --include-debug
nil azzo fun also oop ized eps lsa apk hai ived ddb
  GPT-4: 18 -> 18 tokens
  GPT-4o: 18 -> 17 tokens

$ echo "550e8400-e29b-41d4-a716-446655440000" \
    | python examples/encode.py --style numeric --include-debug
113059749145936325402354257176981405696
  GPT-4: 18 -> 13 tokens
  GPT-4o: 18 -> 13 tokens
```

Note: space-separated token-optimized words are actually cheaper (17-18 tokens) than hyphenated (23 tokens), because tokenizers merge `" word"` into a single token. Numeric is the cheapest at 13 tokens (~40% savings).

### Roundtrip verification

All styles are fully reversible. Pipe encode into decode to verify:

```sh
$ echo "550e8400-e29b-41d4-a716-446655440000" \
    | python examples/encode.py | python examples/decode.py
550e8400-e29b-41d4-a716-446655440000
```

### Rust CLI

The Rust CLI supports the core word encoding styles (memorable, token) and all type modes:

```sh
$ id-tokenizer encode 550e8400-e29b-41d4-a716-446655440000
nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh

$ id-tokenizer decode "nil-gol-bun-also-yuca-fret-hon-gunne-egg-drop-bice-cosh"
550e8400-e29b-41d4-a716-446655440000

$ id-tokenizer encode u64 12345
davy-mird-oak-twa-pan-tur-uji
```

## Design

### Hash function

**xxHash64** (seed 0) maps words to vocabulary slot indices: `slot = xxh64(word) & (vocab_size - 1)`. Decoding is pure math -- no table lookup needed. The word list is constructed so that each word's hash maps to its own index.

### Checksum

**FNV-1a 32-bit**, truncated to fit the available spare bits. The codec maximizes checksum width for the available capacity:

```
total_bits = word_count * bits_per_word
check_bits = total_bits - payload_bits     (minimum 4)
```

This gives 4-bit checksums (1-in-16 error detection) for tight fits and up to 15-bit checksums (1-in-32768) when there's room.

### Bit mixing

A **4-round Feistel network** using xxHash64 as the round function provides a bijective permutation that ensures adjacent inputs produce completely different outputs. Width-appropriate variants (128/64/32-bit) are used for each payload size. The mixing is applied to fixed-width encodings (UUID, u32, u64, u128) but not to variable-length `int_to_words` (where compactness is the priority).

### UUID optimization

For UUID encoding, the 2 variant bits (always `10` for RFC 4122) are stripped before encoding and restored on decode. This doesn't reduce word count but gives the checksum 2 extra bits of implicit validation. The version nibble (byte 6, high nibble) is also validated on decode, rejecting malformed UUIDs.

### Bloom filter

Each vocabulary + style combination has a pre-built bloom filter (2 bits per element, 4 hash functions) compiled into the binary. On decode, words are checked against the bloom filter before hashing, catching most typos with a fast bit-test before the more expensive checksum verification.

### Normalization

Decode accepts:
- Hyphen-separated (`word-word-word`) or whitespace-separated (`word word word`)
- Case-insensitive (normalized to lowercase)
- Words must contain only `[a-z]` after normalization

### Error types

| Error | Cause |
|-------|-------|
| `InvalidWordCount` | Wrong number of words for the expected payload size |
| `InvalidCharacter` | Word contains non-alphabetic characters |
| `NotInVocabulary` | Word rejected by bloom filter |
| `ChecksumMismatch` | Decoded checksum doesn't match payload |
| `InvalidUuid` | Decoded UUID has invalid version (0 or >8) |
| `EmptyPhrase` | Empty input to variable-length decode |
| `Overflow` | Decoded value exceeds u128 |

## Building from source

### Requirements

- Rust 1.80+ (2024 edition)
- Python 3.9+ (for bindings)
- [maturin](https://www.maturin.rs/) (for Python build)

### Build

```sh
# Python library + CLI (into active venv)
pip install maturin
maturin develop --release

# Rust-only (library + CLI binary)
cargo build --release

# Run tests
cargo test
pytest tests/
```

### Regenerating word lists

The word lists are generated once and committed. To regenerate (e.g., to change the blocklist or hash function):

```sh
pip install xxhash

# Memorable word lists (from /usr/share/dict/words)
python tools/gen_wordlist.py

# Token-optimized word lists (requires tiktoken + tokenizers)
pip install tiktoken tokenizers huggingface_hub
python tools/gen_tokenlist.py
```

## Project structure

```
src/
  lib.rs          Crate root, PyO3 module definition
  codec.rs        Core encode/decode logic, Vocab + Style enums
  mix.rs          Feistel network (32/64/128-bit)
  checksum.rs     FNV-1a variable-width checksum
  bloom.rs        Bloom filter membership check
  wordlist.rs     Generated: memorable word tables + bloom data
  tokenlist.rs    Generated: token-optimized word tables + bloom data
  python.rs       PyO3 bindings (Codec class + functions)
  main.rs         CLI binary
python/
  id_tokenizer/
    __init__.py     Re-exports native Rust module
    _cli.py         Python CLI entry point
tools/
  gen_wordlist.py         Memorable word list generator
  gen_tokenlist.py        Token-optimized word list generator
  blocklists/             Profanity filter lists (LDNOOBW, Google)
examples/
  encode.py               Pipe-friendly encoder with extra styles (numeric, shuffled)
  decode.py               Pipe-friendly decoder
  shuffle.py              Digit-preserving Feistel shuffle (salted)
  batch.py                TSV batch encode/decode
  token_stats.py          Token count comparison across styles
tests/
  test_codec.py           Python test suite (97 tests)
```
