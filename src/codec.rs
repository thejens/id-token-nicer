use crate::bloom::probably_in_vocabulary;
use crate::checksum::fnv1a_checksum;
use crate::mix::{mix_value, unmix_value};
use crate::tokenlist;
use crate::wordlist::*;
use thiserror::Error;
use xxhash_rust::xxh64::xxh64;

/// Minimum checksum bits we require.
const MIN_CHECK_BITS: u32 = 4;

/// Fixed checksum bits for variable-length integer encoding.
const INT_CHECK_BITS: u32 = 4;

/// Vocabulary size — determines bits per word and word list used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vocab {
    V2048,
    V4096,
    V8192,
    V16384,
    V32768,
}

impl Vocab {
    pub fn size(self) -> usize {
        match self {
            Vocab::V2048 => 2048,
            Vocab::V4096 => 4096,
            Vocab::V8192 => 8192,
            Vocab::V16384 => 16384,
            Vocab::V32768 => 32768,
        }
    }

    pub fn bits_per_word(self) -> u32 {
        match self {
            Vocab::V2048 => 11,
            Vocab::V4096 => 12,
            Vocab::V8192 => 13,
            Vocab::V16384 => 14,
            Vocab::V32768 => 15,
        }
    }

    fn slot_mask(self) -> u64 {
        (self.size() as u64) - 1
    }

    fn wordlist(self, style: Style) -> &'static [&'static str] {
        match style {
            Style::Memorable => match self {
                Vocab::V2048 => &WORDLIST_2048,
                Vocab::V4096 => &WORDLIST_4096,
                Vocab::V8192 => &WORDLIST_8192,
                Vocab::V16384 => &WORDLIST_16384,
                Vocab::V32768 => &WORDLIST_32768,
            },
            Style::Token => match self {
                Vocab::V2048 => &tokenlist::TOKENLIST_2048,
                Vocab::V4096 => &tokenlist::TOKENLIST_4096,
                Vocab::V8192 => &tokenlist::TOKENLIST_8192,
                Vocab::V16384 => &tokenlist::TOKENLIST_16384,
                Vocab::V32768 => &tokenlist::TOKENLIST_32768,
            },
        }
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Vocab::V2048
    }
}

/// Word list style — determines whether to use memorable English words
/// or token-optimized words for LLM efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Style {
    /// Real English words optimized for memorability.
    #[default]
    Memorable,
    /// Words optimized to be single tokens in common LLM tokenizers
    /// (OpenAI cl100k_base and o200k_base).
    Token,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DecodeError {
    #[error("expected {expected} words, got {got}")]
    InvalidWordCount { expected: usize, got: usize },

    #[error("too many words: {got} (max {max})")]
    TooManyWords { max: usize, got: usize },

    #[error("empty phrase")]
    EmptyPhrase,

    #[error("invalid character in word: {word:?}")]
    InvalidCharacter { word: String },

    #[error("checksum mismatch")]
    ChecksumMismatch,

    #[error("word not in vocabulary: {word:?}")]
    NotInVocabulary { word: String },

    #[error("invalid UUID structure: {reason}")]
    InvalidUuid { reason: String },

    #[error("value overflow: decoded value exceeds u128")]
    Overflow,
}

/// Compute word count and actual checksum bits for a fixed-size payload.
fn layout(payload_bits: u32, bits_per_word: u32) -> (usize, u32) {
    let min_total = payload_bits + MIN_CHECK_BITS;
    let n_words = min_total.div_ceil(bits_per_word) as usize;
    let total_capacity = n_words as u32 * bits_per_word;
    let check_bits = total_capacity - payload_bits;
    (n_words, check_bits)
}

fn word_slot(word: &str, vocab: Vocab) -> u16 {
    (xxh64(word.as_bytes(), 0) & vocab.slot_mask()) as u16
}

fn normalize(word: &str) -> Result<String, DecodeError> {
    let w = word.trim().to_ascii_lowercase();
    if w.is_empty() || !w.bytes().all(|b| b.is_ascii_lowercase()) {
        return Err(DecodeError::InvalidCharacter {
            word: word.to_string(),
        });
    }
    Ok(w)
}

fn split_phrase(phrase: &str) -> Vec<&str> {
    phrase
        .split(|c: char| c == '-' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect()
}

// --- Wide integer (u128 + 16-bit overflow, up to 144 bits) ---

#[derive(Clone, Copy, Debug)]
struct Wide {
    hi: u16,
    lo: u128,
}

impl Wide {
    fn zero() -> Self {
        Self { hi: 0, lo: 0 }
    }

    fn from_u128(v: u128) -> Self {
        Self { hi: 0, lo: v }
    }

    fn shl(self, n: u32) -> Self {
        if n == 0 {
            return self;
        }
        if n >= 128 {
            Self {
                hi: (self.lo as u16) << (n - 128),
                lo: 0,
            }
        } else {
            Self {
                hi: ((self.hi as u128) << n | self.lo >> (128 - n)) as u16,
                lo: self.lo << n,
            }
        }
    }

    fn bitor(self, rhs: u64) -> Self {
        Self {
            hi: self.hi,
            lo: self.lo | rhs as u128,
        }
    }

    fn low_bits(self, n: u32) -> u64 {
        (self.lo & ((1u128 << n) - 1)) as u64
    }

    fn shr(self, n: u32) -> Self {
        if n == 0 {
            return self;
        }
        if n >= 128 {
            Self {
                hi: 0,
                lo: (self.hi as u128) >> (n - 128),
            }
        } else {
            Self {
                hi: (self.hi >> n) as u16,
                lo: self.lo >> n | (self.hi as u128) << (128 - n),
            }
        }
    }

    fn to_u128(self) -> u128 {
        self.lo
    }

    fn overflows_u128(self) -> bool {
        self.hi != 0
    }
}

fn pack_words(indices: &[u16], bits_per_word: u32) -> Wide {
    let mut x = Wide::zero();
    for &idx in indices {
        x = x.shl(bits_per_word).bitor(idx as u64);
    }
    x
}

fn unpack_words(mut x: Wide, count: usize, bits_per_word: u32) -> Vec<u16> {
    let mask = (1u64 << bits_per_word) - 1;
    let mut indices = vec![0u16; count];
    for i in (0..count).rev() {
        indices[i] = (x.low_bits(bits_per_word) & mask) as u16;
        x = x.shr(bits_per_word);
    }
    indices
}

fn pad_left(data: &[u8]) -> [u8; 16] {
    let mut buf = [0u8; 16];
    let n = data.len().min(16);
    buf[16 - n..].copy_from_slice(&data[..n]);
    buf
}

/// Checksum over a u128 value (big-endian, leading zeros stripped for consistency).
fn checksum_u128(value: u128, bits: u32) -> u16 {
    fnv1a_checksum(&value.to_be_bytes(), bits)
}

// --- Fixed-size encode/decode (byte arrays with known length) ---

fn encode_raw(payload: &[u8], payload_bits: u32, vocab: Vocab, style: Style) -> String {
    let bpw = vocab.bits_per_word();
    let (n_words, check_bits) = layout(payload_bits, bpw);

    // Mix payload for avalanche (adjacent values → different words)
    let payload_int = u128::from_be_bytes(pad_left(payload));
    let mixed = mix_value(payload_int, payload_bits);
    let mixed_bytes = mixed.to_be_bytes();
    let start = 16 - payload.len().min(16);
    let check = fnv1a_checksum(&mixed_bytes[start..], check_bits);
    let x = Wide::from_u128(mixed)
        .shl(check_bits)
        .bitor(check as u64);

    let indices = unpack_words(x, n_words, bpw);
    let wl = vocab.wordlist(style);

    indices
        .iter()
        .map(|&i| wl[i as usize])
        .collect::<Vec<_>>()
        .join("-")
}

fn decode_raw(
    phrase: &str,
    expected_bytes: usize,
    payload_bits: u32,
    vocab: Vocab,
    style: Style,
    validate_bloom: bool,
) -> Result<Vec<u8>, DecodeError> {
    let bpw = vocab.bits_per_word();
    let (expected_words, check_bits) = layout(payload_bits, bpw);
    let parts = split_phrase(phrase);

    if parts.len() != expected_words {
        return Err(DecodeError::InvalidWordCount {
            expected: expected_words,
            got: parts.len(),
        });
    }

    let mut indices = Vec::with_capacity(parts.len());
    for part in &parts {
        let w = normalize(part)?;
        if validate_bloom && !probably_in_vocabulary(&w, vocab.size(), style) {
            return Err(DecodeError::NotInVocabulary { word: w });
        }
        indices.push(word_slot(&w, vocab));
    }

    let x = pack_words(&indices, bpw);
    let check_in = x.low_bits(check_bits) as u16;
    let mixed_int = x.shr(check_bits).to_u128();

    let mixed_bytes = mixed_int.to_be_bytes();
    let start = 16 - expected_bytes;
    let check_calc = fnv1a_checksum(&mixed_bytes[start..], check_bits);
    if check_in != check_calc {
        return Err(DecodeError::ChecksumMismatch);
    }

    // Unmix to recover original payload
    let payload_int = unmix_value(mixed_int, payload_bits);
    let full = payload_int.to_be_bytes();
    Ok(full[start..].to_vec())
}

// --- Variable-length integer encoding (no zero-padding) ---

/// Encode a u128 with minimal words (no leading zero-words).
///
/// Uses variable word count based on value magnitude. No bit-mixing
/// (use encode_u32/u64/u128 for fixed-width with avalanche mixing).
pub fn encode_int(value: u128, vocab: Vocab, style: Style) -> String {
    let bpw = vocab.bits_per_word();
    let check = checksum_u128(value, INT_CHECK_BITS);
    let x = Wide::from_u128(value)
        .shl(INT_CHECK_BITS)
        .bitor(check as u64);

    let total_bits = if value == 0 {
        INT_CHECK_BITS
    } else {
        128 - value.leading_zeros() + INT_CHECK_BITS
    };
    let n_words = total_bits.div_ceil(bpw).max(1) as usize;

    let indices = unpack_words(x, n_words, bpw);
    let wl = vocab.wordlist(style);

    indices
        .iter()
        .map(|&i| wl[i as usize])
        .collect::<Vec<_>>()
        .join("-")
}

/// Decode a variable-length word phrase back to a u128.
///
/// The word count determines the payload capacity. No expected length needed.
pub fn decode_int(phrase: &str, vocab: Vocab, style: Style) -> Result<u128, DecodeError> {
    let bpw = vocab.bits_per_word();
    let parts = split_phrase(phrase);

    if parts.is_empty() {
        return Err(DecodeError::EmptyPhrase);
    }

    // Max words that fit in u128 + 4 check bits = 132 bits
    let max_words = (128 + INT_CHECK_BITS).div_ceil(bpw) as usize;
    if parts.len() > max_words {
        return Err(DecodeError::TooManyWords {
            max: max_words,
            got: parts.len(),
        });
    }

    let mut indices = Vec::with_capacity(parts.len());
    for part in &parts {
        let w = normalize(part)?;
        if !probably_in_vocabulary(&w, vocab.size(), style) {
            return Err(DecodeError::NotInVocabulary { word: w });
        }
        indices.push(word_slot(&w, vocab));
    }

    let x = pack_words(&indices, bpw);
    let check_in = x.low_bits(INT_CHECK_BITS) as u16;
    let payload = x.shr(INT_CHECK_BITS);

    if payload.overflows_u128() {
        return Err(DecodeError::Overflow);
    }

    let value = payload.to_u128();
    let check_calc = checksum_u128(value, INT_CHECK_BITS);
    if check_in != check_calc {
        return Err(DecodeError::ChecksumMismatch);
    }

    Ok(value)
}

// --- UUID-specific ---

fn strip_uuid_variant(uuid: &[u8; 16]) -> [u8; 16] {
    let mut stripped = *uuid;
    stripped[8] &= 0x3F;
    stripped
}

fn restore_uuid_variant(stripped: &mut [u8; 16]) {
    stripped[8] = (stripped[8] & 0x3F) | 0x80;
}

/// Encode a UUID (16 bytes) into words.
///
/// Strips the 2 known variant bits (always `10` for RFC 4122) before encoding,
/// giving the checksum more room. The variant is restored on decode.
pub fn encode_uuid(uuid: &[u8; 16], vocab: Vocab, style: Style) -> String {
    let stripped = strip_uuid_variant(uuid);
    encode_raw(&stripped, 128, vocab, style)
}

/// Decode words back to a UUID.
///
/// Validates checksum, restores variant bits, and checks UUID version.
pub fn decode_uuid(phrase: &str, vocab: Vocab, style: Style) -> Result<[u8; 16], DecodeError> {
    let bytes = decode_raw(phrase, 16, 128, vocab, style, true)?;
    let mut uuid: [u8; 16] = bytes.try_into().unwrap();
    restore_uuid_variant(&mut uuid);

    let version = uuid[6] >> 4;
    if version == 0 || version > 8 {
        return Err(DecodeError::InvalidUuid {
            reason: format!("invalid version nibble: {version}"),
        });
    }

    Ok(uuid)
}

// --- Fixed-width typed encode/decode ---

pub fn encode_u128(value: u128, vocab: Vocab, style: Style) -> String {
    encode_raw(&value.to_be_bytes(), 128, vocab, style)
}

pub fn decode_u128(phrase: &str, vocab: Vocab, style: Style) -> Result<u128, DecodeError> {
    let bytes = decode_raw(phrase, 16, 128, vocab, style, true)?;
    Ok(u128::from_be_bytes(bytes.try_into().unwrap()))
}

pub fn encode_u64(value: u64, vocab: Vocab, style: Style) -> String {
    encode_raw(&value.to_be_bytes(), 64, vocab, style)
}

pub fn decode_u64(phrase: &str, vocab: Vocab, style: Style) -> Result<u64, DecodeError> {
    let bytes = decode_raw(phrase, 8, 64, vocab, style, true)?;
    Ok(u64::from_be_bytes(bytes.try_into().unwrap()))
}

pub fn encode_u32(value: u32, vocab: Vocab, style: Style) -> String {
    encode_raw(&value.to_be_bytes(), 32, vocab, style)
}

pub fn decode_u32(phrase: &str, vocab: Vocab, style: Style) -> Result<u32, DecodeError> {
    let bytes = decode_raw(phrase, 4, 32, vocab, style, true)?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

pub fn encode_bytes(payload: &[u8], vocab: Vocab, style: Style) -> String {
    let payload_bits = (payload.len() * 8) as u32;
    let bpw = vocab.bits_per_word();
    let (n_words, check_bits) = layout(payload_bits, bpw);

    // No mixing for arbitrary bytes — just checksum and pack
    let check = fnv1a_checksum(payload, check_bits);
    let payload_int = u128::from_be_bytes(pad_left(payload));
    let x = Wide::from_u128(payload_int)
        .shl(check_bits)
        .bitor(check as u64);

    let indices = unpack_words(x, n_words, bpw);
    let wl = vocab.wordlist(style);

    indices
        .iter()
        .map(|&i| wl[i as usize])
        .collect::<Vec<_>>()
        .join("-")
}

pub fn decode_bytes(phrase: &str, expected_len: usize, vocab: Vocab, style: Style) -> Result<Vec<u8>, DecodeError> {
    let payload_bits = (expected_len * 8) as u32;
    let bpw = vocab.bits_per_word();
    let (expected_words, check_bits) = layout(payload_bits, bpw);
    let parts = split_phrase(phrase);

    if parts.len() != expected_words {
        return Err(DecodeError::InvalidWordCount {
            expected: expected_words,
            got: parts.len(),
        });
    }

    let mut indices = Vec::with_capacity(parts.len());
    for part in &parts {
        let w = normalize(part)?;
        if !probably_in_vocabulary(&w, vocab.size(), style) {
            return Err(DecodeError::NotInVocabulary { word: w });
        }
        indices.push(word_slot(&w, vocab));
    }

    let x = pack_words(&indices, bpw);
    let check_in = x.low_bits(check_bits) as u16;
    let payload_int = x.shr(check_bits).to_u128();

    let full = payload_int.to_be_bytes();
    let start = 16 - expected_len;
    let payload_bytes = &full[start..];

    let check_calc = fnv1a_checksum(payload_bytes, check_bits);
    if check_in != check_calc {
        return Err(DecodeError::ChecksumMismatch);
    }

    Ok(payload_bytes.to_vec())
}

/// Number of words for a given payload byte count and vocab.
pub fn word_count(payload_bytes: usize, vocab: Vocab) -> usize {
    let bits = (payload_bytes * 8) as u32;
    layout(bits, vocab.bits_per_word()).0
}

/// Number of words for UUID encoding with a given vocab.
pub fn uuid_word_count(vocab: Vocab) -> usize {
    layout(128, vocab.bits_per_word()).0
}

/// Number of checksum bits used for a given payload size and vocab.
pub fn check_bit_count(payload_bytes: usize, vocab: Vocab) -> u32 {
    let bits = (payload_bytes * 8) as u32;
    layout(bits, vocab.bits_per_word()).1
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_VOCABS: [Vocab; 5] = [
        Vocab::V2048,
        Vocab::V4096,
        Vocab::V8192,
        Vocab::V16384,
        Vocab::V32768,
    ];

    const ALL_STYLES: [Style; 2] = [Style::Memorable, Style::Token];

    #[test]
    fn test_layout_uuid() {
        assert_eq!(layout(128, 11), (12, 4));
        assert_eq!(layout(128, 12), (11, 4));
        assert_eq!(layout(128, 13), (11, 15));
        assert_eq!(layout(128, 14), (10, 12));  // 16384
        assert_eq!(layout(128, 15), (9, 7));    // 32768
    }

    #[test]
    fn test_uuid_roundtrip_all_vocabs() {
        let uuid: [u8; 16] = [
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4,
            0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00, 0x00,
        ];
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let phrase = encode_uuid(&uuid, vocab, style);
                let decoded = decode_uuid(&phrase, vocab, style).unwrap();
                assert_eq!(uuid, decoded, "failed for {vocab:?} {style:?}");
            }
        }
    }

    #[test]
    fn test_uuid_word_counts() {
        assert_eq!(uuid_word_count(Vocab::V2048), 12);
        assert_eq!(uuid_word_count(Vocab::V4096), 11);
        assert_eq!(uuid_word_count(Vocab::V8192), 11);
        assert_eq!(uuid_word_count(Vocab::V16384), 10);
        assert_eq!(uuid_word_count(Vocab::V32768), 9);
    }

    #[test]
    fn test_uuid_nil_rejected() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let phrase = encode_uuid(&[0u8; 16], vocab, style);
                assert!(decode_uuid(&phrase, vocab, style).is_err());
            }
        }
    }

    #[test]
    fn test_u128_roundtrip() {
        let val: u128 = 0xDEADBEEF_CAFEBABE_12345678_9ABCDEF0;
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let phrase = encode_u128(val, vocab, style);
                assert_eq!(decode_u128(&phrase, vocab, style).unwrap(), val);
            }
        }
    }

    #[test]
    fn test_u64_roundtrip() {
        let val: u64 = 0xDEADBEEFCAFEBABE;
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let phrase = encode_u64(val, vocab, style);
                assert_eq!(decode_u64(&phrase, vocab, style).unwrap(), val);
            }
        }
    }

    #[test]
    fn test_u32_roundtrip() {
        for val in [0u32, 1, 42, 0xDEADBEEF, u32::MAX] {
            for style in ALL_STYLES {
                for vocab in ALL_VOCABS {
                    let phrase = encode_u32(val, vocab, style);
                    assert_eq!(decode_u32(&phrase, vocab, style).unwrap(), val, "{val} {vocab:?} {style:?}");
                }
            }
        }
    }

    // --- Variable-length int tests ---

    #[test]
    fn test_int_roundtrip_small() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                for val in [0u128, 1, 42, 255, 1000, 65535] {
                    let phrase = encode_int(val, vocab, style);
                    assert_eq!(decode_int(&phrase, vocab, style).unwrap(), val, "{val} {vocab:?} {style:?}");
                }
            }
        }
    }

    #[test]
    fn test_int_roundtrip_large() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let val: u128 = 0xDEADBEEF_CAFEBABE_12345678_9ABCDEF0;
                let phrase = encode_int(val, vocab, style);
                assert_eq!(decode_int(&phrase, vocab, style).unwrap(), val);
            }
        }
    }

    #[test]
    fn test_int_compact_encoding() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let small = encode_int(42, vocab, style);
                let large = encode_int(u128::MAX, vocab, style);
                let small_words = split_phrase(&small).len();
                let large_words = split_phrase(&large).len();
                assert!(
                    small_words < large_words,
                    "{vocab:?}: small={small_words} words, large={large_words} words"
                );
            }
        }
    }

    #[test]
    fn test_int_zero_is_one_word() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let phrase = encode_int(0, vocab, style);
                assert_eq!(split_phrase(&phrase).len(), 1, "{vocab:?}");
            }
        }
    }

    #[test]
    fn test_int_max_u128() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let phrase = encode_int(u128::MAX, vocab, style);
                assert_eq!(decode_int(&phrase, vocab, style).unwrap(), u128::MAX);
            }
        }
    }

    #[test]
    fn test_int_deterministic() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                assert_eq!(encode_int(42, vocab, style), encode_int(42, vocab, style));
            }
        }
    }

    #[test]
    fn test_int_decode_empty() {
        assert!(matches!(
            decode_int("", Vocab::V2048, Style::Memorable),
            Err(DecodeError::EmptyPhrase)
        ));
    }

    // --- Existing tests ---

    #[test]
    fn test_deterministic() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                assert_eq!(encode_u64(42, vocab, style), encode_u64(42, vocab, style));
            }
        }
    }

    #[test]
    fn test_decode_wrong_word_count() {
        let result = decode_u32("one-two-three", Vocab::V2048, Style::Memorable);
        assert!(matches!(result, Err(DecodeError::InvalidWordCount { .. })));
    }

    #[test]
    fn test_decode_invalid_chars() {
        let result = decode_u32("abc-d3f-ghi-jkl", Vocab::V2048, Style::Memorable);
        assert!(matches!(result, Err(DecodeError::InvalidCharacter { .. })));
    }

    #[test]
    fn test_whitespace_separator() {
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let val: u32 = 12345;
                let phrase = encode_u32(val, vocab, style);
                let spaced = phrase.replace('-', "  ");
                assert_eq!(decode_u32(&spaced, vocab, style).unwrap(), val);
            }
        }
    }

    #[test]
    fn test_hash_matches_python() {
        use xxhash_rust::xxh64::xxh64;
        assert_eq!(xxh64(b"jilt", 0), 0xf21cbd4b7c63eabb);
        assert_eq!(xxh64(b"hello", 0), 0x26c7827d889f6da3);
        assert_eq!(xxh64(b"abc", 0), 0x44bc2cf5ad770999);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        for style in ALL_STYLES {
            for vocab in ALL_VOCABS {
                let phrase = encode_bytes(&data, vocab, style);
                assert_eq!(decode_bytes(&phrase, data.len(), vocab, style).unwrap(), data);
            }
        }
    }

    #[test]
    fn test_styles_produce_different_words() {
        let uuid: [u8; 16] = [
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4,
            0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00, 0x00,
        ];
        let memorable = encode_uuid(&uuid, Vocab::V2048, Style::Memorable);
        let token = encode_uuid(&uuid, Vocab::V2048, Style::Token);
        assert_ne!(memorable, token);
    }
}
