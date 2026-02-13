use crate::codec::Style;
use crate::tokenlist;
use crate::wordlist::*;
use xxhash_rust::xxh64::xxh64;

/// Check if a word is probably in the vocabulary for a given vocab size and style.
pub fn probably_in_vocabulary(word: &str, vocab_size: usize, style: Style) -> bool {
    let (bloom_bits, bloom_data) = bloom_for(vocab_size, style);
    let k = match style {
        Style::Memorable => BLOOM_K,
        Style::Token => tokenlist::TOKEN_BLOOM_K,
    };
    let data = word.as_bytes();
    for i in 0..k {
        let h = xxh64(data, i as u64);
        let bit = (h as usize) % bloom_bits;
        let word_idx = bit / 64;
        let bit_idx = bit % 64;
        if bloom_data[word_idx] & (1u64 << bit_idx) == 0 {
            return false;
        }
    }
    true
}

fn bloom_for(vocab_size: usize, style: Style) -> (usize, &'static [u64]) {
    match style {
        Style::Memorable => match vocab_size {
            2048 => (BLOOM_2048_BITS, &BLOOM_2048_DATA),
            4096 => (BLOOM_4096_BITS, &BLOOM_4096_DATA),
            8192 => (BLOOM_8192_BITS, &BLOOM_8192_DATA),
            16384 => (BLOOM_16384_BITS, &BLOOM_16384_DATA),
            32768 => (BLOOM_32768_BITS, &BLOOM_32768_DATA),
            _ => unreachable!("unsupported vocab size: {vocab_size}"),
        },
        Style::Token => match vocab_size {
            2048 => (tokenlist::TOKEN_BLOOM_2048_BITS, &tokenlist::TOKEN_BLOOM_2048_DATA),
            4096 => (tokenlist::TOKEN_BLOOM_4096_BITS, &tokenlist::TOKEN_BLOOM_4096_DATA),
            8192 => (tokenlist::TOKEN_BLOOM_8192_BITS, &tokenlist::TOKEN_BLOOM_8192_DATA),
            16384 => (tokenlist::TOKEN_BLOOM_16384_BITS, &tokenlist::TOKEN_BLOOM_16384_DATA),
            32768 => (tokenlist::TOKEN_BLOOM_32768_BITS, &tokenlist::TOKEN_BLOOM_32768_DATA),
            _ => unreachable!("unsupported vocab size: {vocab_size}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_words_pass_bloom() {
        for (words, size) in [
            (&WORDLIST_2048[..], 2048),
            (&WORDLIST_4096[..], 4096),
            (&WORDLIST_8192[..], 8192),
            (&WORDLIST_16384[..], 16384),
            (&WORDLIST_32768[..], 32768),
        ] {
            for word in words {
                assert!(
                    probably_in_vocabulary(word, size, Style::Memorable),
                    "missing '{word}' in bloom for M={size}"
                );
            }
        }
    }

    #[test]
    fn test_all_token_words_pass_bloom() {
        for (words, size) in [
            (&tokenlist::TOKENLIST_2048[..], 2048),
            (&tokenlist::TOKENLIST_4096[..], 4096),
            (&tokenlist::TOKENLIST_8192[..], 8192),
            (&tokenlist::TOKENLIST_16384[..], 16384),
            (&tokenlist::TOKENLIST_32768[..], 32768),
        ] {
            for word in words {
                assert!(
                    probably_in_vocabulary(word, size, Style::Token),
                    "missing '{word}' in token bloom for M={size}"
                );
            }
        }
    }
}
