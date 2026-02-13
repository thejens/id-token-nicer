#!/usr/bin/env python3
"""
Generate word lists for what-12-words at multiple vocabulary sizes (2048, 4096, 8192).

For each size M:
1. Hash each candidate word with xxHash64(seed=0) & (M-1) to get its slot
2. Pick the shortest real word per slot
3. Generate pronounceable fill words for any empty slots
4. Build a bloom filter for membership validation

Outputs src/wordlist.rs with all three word lists and bloom filters.

Requirements: pip install xxhash
"""

import hashlib
import sys
from pathlib import Path

import xxhash

def load_blocklist() -> set[str]:
    """Load profanity blocklist from downloaded lists + hardcoded extras."""
    blocklist: set[str] = set()
    blocklist_dir = Path(__file__).parent / "blocklists"
    for f in blocklist_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            w = line.strip().lower()
            if w and w.isalpha() and len(w) <= 8:
                blocklist.add(w)
    # Additional manual entries
    blocklist.update({
        "die", "dead", "kill", "hate", "nazi",
    })
    return blocklist


BLOCKLIST = load_blocklist()


def xxh64(data: bytes, seed: int = 0) -> int:
    return xxhash.xxh64(data, seed=seed).intdigest()


def is_good_word(w: str) -> bool:
    if not w.isascii() or not w.isalpha():
        return False
    w = w.lower()
    if len(w) < 3 or len(w) > 8:
        return False
    return all("a" <= c <= "z" for c in w)


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


def generate_fill_words(empty_slots: list[int], m: int, existing: set[str]) -> dict[int, str]:
    """Generate pronounceable words for all empty slots at once (batched for speed)."""
    consonants = "bcdfghjklmnprstvwz"
    vowels = "aeiou"
    mask = m - 1
    needed = set(empty_slots)
    result: dict[int, str] = {}

    for length in range(4, 9):
        if not needed:
            break
        for trial in range(5_000_000):
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
                result[slot] = word
                existing.add(word)
                needed.discard(slot)
                if not needed:
                    break

    if needed:
        raise RuntimeError(f"Could not fill {len(needed)} slots for M={m}: {sorted(needed)[:10]}...")
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


def build_wordlist(candidates: list[str], m: int) -> list[str]:
    mask = m - 1
    slots: dict[int, list[str]] = {i: [] for i in range(m)}
    for w in candidates:
        s = xxh64(w.encode("utf-8"), seed=0) & mask
        slots[s].append(w)

    rep: list[str] = [""] * m
    used: set[str] = set()
    filled = 0
    for i in range(m):
        for c in sorted(slots[i], key=lambda w: (len(w), w)):
            if c not in used:
                rep[i] = c
                used.add(c)
                filled += 1
                break

    empty = [i for i in range(m) if rep[i] == ""]
    print(f"    {filled}/{m} from dictionary, {len(empty)} to generate", file=sys.stderr)

    if empty:
        fill = generate_fill_words(empty, m, used)
        for slot, word in fill.items():
            rep[slot] = word

    # Verify
    for i in range(m):
        assert rep[i] != "", f"Slot {i} empty (M={m})"
        actual = xxh64(rep[i].encode("utf-8"), seed=0) & mask
        assert actual == i, f"Slot {i}: '{rep[i]}' hashes to {actual}"
    assert len(set(rep)) == m, "Duplicate words!"
    return rep


def main():
    print("Loading dictionary...", file=sys.stderr)
    candidates = load_dictionary()
    print(f"  {len(candidates)} candidate words", file=sys.stderr)

    sizes = [2048, 4096, 8192, 16384, 32768]
    results: dict[int, tuple[list[str], list[int]]] = {}

    for m in sizes:
        print(f"\nBuilding M={m} word list...", file=sys.stderr)
        rep = build_wordlist(candidates, m)
        lengths = [len(w) for w in rep]
        print(f"    lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}", file=sys.stderr)

        bloom_bits = m * 2  # ~2 bits per element
        bloom = build_bloom(rep, bloom_bits, 4)
        results[m] = (rep, bloom)

    out_path = Path(__file__).parent.parent / "src" / "wordlist.rs"
    print(f"\nWriting {out_path}...", file=sys.stderr)

    with open(out_path, "w") as f:
        f.write("// Auto-generated by tools/gen_wordlist.py — do not edit\n\n")

        for m in sizes:
            rep, bloom = results[m]
            bloom_bits = m * 2

            f.write(f"pub const WORDLIST_{m}: [&str; {m}] = [\n")
            for word in rep:
                f.write(f'    "{word}",\n')
            f.write("];\n\n")

            n_u64 = bloom_bits // 64
            f.write(f"pub const BLOOM_{m}_BITS: usize = {bloom_bits};\n")
            f.write(f"pub const BLOOM_{m}_DATA: [u64; {n_u64}] = [\n")
            for val in bloom:
                f.write(f"    0x{val:016x},\n")
            f.write("];\n\n")

        f.write("pub const BLOOM_K: usize = 4;\n")

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
