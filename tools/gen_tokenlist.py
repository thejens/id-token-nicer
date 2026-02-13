#!/usr/bin/env python3
"""
Generate token-optimized word lists for what-12-words.

Prioritizes words that are single tokens across multiple LLM tokenizers.
The tiers are:
  1. Single token in ALL supported tokenizers (cl100k, o200k, Gemma, Llama 3)
  2. Single token in SOME tokenizers (union minus tier 1)
  3. Real English dictionary words (may be multi-token)
  4. Generated pronounceable CVCV sequences

The tier 1 intersection covers: GPT-4, GPT-4o, Gemma/Gemini, Llama 3,
Qwen 2.5, Phi-4, and DeepSeek (cl100k is a subset of Llama 3/Qwen/Phi/DeepSeek).

Each word must hash to its slot: xxh64(word, seed=0) & (M-1) == slot_index.

Requirements: pip install xxhash tiktoken tokenizers
"""

import hashlib
import sys
from pathlib import Path

import tiktoken
import xxhash


def load_blocklist() -> set[str]:
    blocklist: set[str] = set()
    blocklist_dir = Path(__file__).parent / "blocklists"
    for f in blocklist_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            w = line.strip().lower()
            if w and w.isalpha() and len(w) <= 8:
                blocklist.add(w)
    blocklist.update({"die", "dead", "kill", "hate", "nazi"})
    return blocklist


BLOCKLIST = load_blocklist()
HYPHEN_BAD: set[str] = set()  # populated at runtime with hyphen-fragmenting words


def xxh64(data: bytes, seed: int = 0) -> int:
    return xxhash.xxh64(data, seed=seed).intdigest()


def is_good_word(w: str) -> bool:
    return (
        w.isascii()
        and w.isalpha()
        and w.islower()
        and 3 <= len(w) <= 8
    )


def tiktoken_single_tokens(enc_name: str) -> set[str]:
    """Extract single-token words from a tiktoken encoding."""
    enc = tiktoken.get_encoding(enc_name)
    words = set()
    for token_id in range(enc.n_vocab):
        try:
            text = enc.decode([token_id])
        except Exception:
            continue
        if is_good_word(text) and text not in BLOCKLIST:
            words.add(text)
    return words


def tiktoken_hyphen_fragments(enc_name: str, words: set[str]) -> set[str]:
    """Find words that fragment into 3+ tokens when preceded by a hyphen."""
    enc = tiktoken.get_encoding(enc_name)
    bad = set()
    for w in words:
        tokens = enc.encode("-" + w)
        if len(tokens) >= 3:
            bad.add(w)
    return bad


def hf_tokenizer_single_tokens(repo_id: str) -> set[str]:
    """Extract single-token words from a HuggingFace tokenizer."""
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    words = set()
    vocab = tok.get_vocab()
    for token_text in vocab:
        # Strip leading special chars (Ġ for GPT-style, ▁ for sentencepiece)
        clean = token_text.lstrip("Ġ▁ ")
        if is_good_word(clean) and clean not in BLOCKLIST:
            words.add(clean)
    return words


def load_dictionary(path: str = "/usr/share/dict/words") -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            w = line.strip().lower()
            if w in seen:
                continue
            seen.add(w)
            if is_good_word(w) and w not in BLOCKLIST:
                words.append(w)
    return words


_HYPHEN_ENCS: list[tiktoken.Encoding] | None = None

def _get_hyphen_encs() -> list[tiktoken.Encoding]:
    global _HYPHEN_ENCS
    if _HYPHEN_ENCS is None:
        _HYPHEN_ENCS = [tiktoken.get_encoding(n) for n in ["cl100k_base", "o200k_base"]]
    return _HYPHEN_ENCS


def fragments_in_hyphen_context(word: str) -> bool:
    """Check if a word fragments into 3+ tokens when preceded by a hyphen."""
    for enc in _get_hyphen_encs():
        if len(enc.encode("-" + word)) >= 3:
            return True
    return False


def generate_fill_words(empty_slots: list[int], m: int, existing: set[str]) -> dict[int, str]:
    consonants = "bcdfghjklmnprstvwz"
    vowels = "aeiou"
    mask = m - 1
    needed = set(empty_slots)
    result: dict[int, str] = {}
    # Track fallback candidates for slots where all CVCV options fragment
    fallback: dict[int, str] = {}

    for length in range(4, 9):
        if not needed:
            break
        for trial in range(50_000_000):
            seed_bytes = (m * 10_000_000 + length * 5_000_000 + trial).to_bytes(8, "big")
            h = hashlib.sha256(seed_bytes).digest()
            chars: list[str] = []
            for i in range(length):
                byte = h[i % len(h)]
                if i % 2 == 0:
                    chars.append(consonants[byte % len(consonants)])
                else:
                    chars.append(vowels[byte % len(vowels)])
            word = "".join(chars)
            if word in existing:
                continue
            slot = xxh64(word.encode("utf-8"), seed=0) & mask
            if slot in needed:
                if fragments_in_hyphen_context(word):
                    if slot not in fallback:
                        fallback[slot] = word
                    continue
                result[slot] = word
                existing.add(word)
                needed.discard(slot)
                if not needed:
                    break

    # Use fallback (fragmenting) words for any remaining slots
    if needed and fallback:
        for slot in list(needed):
            if slot in fallback:
                result[slot] = fallback[slot]
                existing.add(fallback[slot])
                needed.discard(slot)

    if needed:
        raise RuntimeError(f"Could not fill {len(needed)} slots for M={m}")
    return result


def build_bloom(words: list[str], m_bits: int, k_hashes: int) -> list[int]:
    n_u64 = m_bits // 64
    bitset = [0] * n_u64
    for word in words:
        data = word.encode("utf-8")
        for i in range(k_hashes):
            h = xxh64(data, seed=i)
            bit = h % m_bits
            bitset[bit // 64] |= 1 << (bit % 64)
    return bitset


def build_tokenlist(
    tier1: set[str],
    tier2: set[str],
    tier3: list[str],
    m: int,
) -> tuple[list[str], dict[str, int]]:
    """Build a word list prioritizing single-token words."""
    mask = m - 1
    rep: list[str] = [""] * m
    used: set[str] = set()
    stats = {"tier1_all": 0, "tier2_some": 0, "tier3_dict": 0, "tier4_generated": 0}

    def try_fill(candidates: set[str] | list[str], tier_name: str):
        by_slot: dict[int, list[str]] = {}
        for w in candidates:
            if w in used or w in HYPHEN_BAD or fragments_in_hyphen_context(w):
                continue
            s = xxh64(w.encode("utf-8"), seed=0) & mask
            if rep[s] != "":
                continue
            by_slot.setdefault(s, []).append(w)
        for slot, words in by_slot.items():
            best = min(words, key=lambda w: (len(w), w))
            rep[slot] = best
            used.add(best)
            stats[tier_name] += 1

    try_fill(tier1, "tier1_all")
    try_fill(tier2, "tier2_some")
    try_fill(tier3, "tier3_dict")

    empty = [i for i in range(m) if rep[i] == ""]
    if empty:
        fill = generate_fill_words(empty, m, used)
        for slot, word in fill.items():
            rep[slot] = word
            stats["tier4_generated"] += 1

    for i in range(m):
        assert rep[i] != "", f"Slot {i} empty (M={m})"
        actual = xxh64(rep[i].encode("utf-8"), seed=0) & mask
        assert actual == i, f"Slot {i}: '{rep[i]}' hashes to {actual}"
    assert len(set(rep)) == m, "Duplicate words!"
    return rep, stats


def main():
    print("Loading tiktoken vocabularies...", file=sys.stderr)
    cl100k = tiktoken_single_tokens("cl100k_base")
    o200k = tiktoken_single_tokens("o200k_base")
    print(f"  cl100k_base (GPT-4):  {len(cl100k)}", file=sys.stderr)
    print(f"  o200k_base (GPT-4o):  {len(o200k)}", file=sys.stderr)

    print("Loading HuggingFace tokenizers...", file=sys.stderr)
    hf_tokenizers: dict[str, set[str]] = {}
    hf_sources = {
        "gemma": "Xenova/gemma-tokenizer",
        "llama3": "unsloth/Llama-3.3-70B-Instruct",
    }
    for name, repo in hf_sources.items():
        try:
            words = hf_tokenizer_single_tokens(repo)
            hf_tokenizers[name] = words
            print(f"  {name} ({repo}): {len(words)}", file=sys.stderr)
        except Exception as e:
            print(f"  {name}: FAILED ({e})", file=sys.stderr)

    # Remove words that fragment in hyphen context (-word -> 3+ tokens)
    # This ensures all words tokenize cleanly in hyphenated phrases
    print("\nFiltering hyphen-context fragments...", file=sys.stderr)
    all_candidates = cl100k | o200k
    for words in hf_tokenizers.values():
        all_candidates |= words
    hyphen_bad: set[str] = set()
    for enc_name in ["cl100k_base", "o200k_base"]:
        hyphen_bad |= tiktoken_hyphen_fragments(enc_name, all_candidates)
    if hyphen_bad:
        print(f"  rejecting {len(hyphen_bad)} words: {sorted(hyphen_bad)}", file=sys.stderr)
    cl100k -= hyphen_bad
    o200k -= hyphen_bad
    for name in hf_tokenizers:
        hf_tokenizers[name] -= hyphen_bad
    HYPHEN_BAD.update(hyphen_bad)

    # Tier 1: intersection of ALL tokenizers
    tier1 = cl100k & o200k
    for s in hf_tokenizers.values():
        tier1 &= s
    print(f"\n  tier1 (all {2 + len(hf_tokenizers)} tokenizers): {len(tier1)}", file=sys.stderr)

    # Tier 2: single token in any tokenizer, minus tier1
    union_all = cl100k | o200k
    for s in hf_tokenizers.values():
        union_all |= s
    tier2 = union_all - tier1
    print(f"  tier2 (any tokenizer):  {len(tier2)}", file=sys.stderr)

    print("\nLoading dictionary...", file=sys.stderr)
    dict_words = load_dictionary()
    print(f"  {len(dict_words)} dictionary words", file=sys.stderr)

    sizes = [2048, 4096, 8192, 16384, 32768]
    results: dict[int, tuple[list[str], list[int]]] = {}

    for m in sizes:
        print(f"\nBuilding token-optimized M={m}...", file=sys.stderr)
        rep, stats = build_tokenlist(tier1, tier2, dict_words, m)
        single_token = stats["tier1_all"] + stats["tier2_some"]
        print(f"    tier1 (all tokenizers):  {stats['tier1_all']}", file=sys.stderr)
        print(f"    tier2 (some tokenizers): {stats['tier2_some']}", file=sys.stderr)
        print(f"    tier3 (dictionary):      {stats['tier3_dict']}", file=sys.stderr)
        print(f"    tier4 (generated):       {stats['tier4_generated']}", file=sys.stderr)
        print(f"    total single-token:      {single_token}/{m} ({100*single_token/m:.1f}%)", file=sys.stderr)

        lengths = [len(w) for w in rep]
        print(f"    lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}", file=sys.stderr)

        bloom_bits = m * 2
        bloom = build_bloom(rep, bloom_bits, 4)
        results[m] = (rep, bloom)

    out_path = Path(__file__).parent.parent / "src" / "tokenlist.rs"
    print(f"\nWriting {out_path}...", file=sys.stderr)

    with open(out_path, "w") as f:
        f.write("// Auto-generated by tools/gen_tokenlist.py — do not edit\n//\n")
        f.write("// Token-optimized word lists. Tier 1 words are single tokens in:\n")
        f.write("//   cl100k_base (GPT-4), o200k_base (GPT-4o),\n")
        names = ", ".join(f"{n} ({r})" for n, r in hf_sources.items())
        f.write(f"//   {names}\n")
        f.write("// Also covers Qwen 2.5, Phi-4, DeepSeek (cl100k subsets).\n\n")

        for m in sizes:
            rep, bloom = results[m]
            bloom_bits = m * 2

            f.write(f"pub const TOKENLIST_{m}: [&str; {m}] = [\n")
            for word in rep:
                f.write(f'    "{word}",\n')
            f.write("];\n\n")

            n_u64 = bloom_bits // 64
            f.write(f"pub const TOKEN_BLOOM_{m}_BITS: usize = {bloom_bits};\n")
            f.write(f"pub const TOKEN_BLOOM_{m}_DATA: [u64; {n_u64}] = [\n")
            for val in bloom:
                f.write(f"    0x{val:016x},\n")
            f.write("];\n\n")

        f.write("pub const TOKEN_BLOOM_K: usize = 4;\n")

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
