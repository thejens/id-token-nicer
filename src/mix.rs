use xxhash_rust::xxh64::xxh64;

/// Round function for Feistel networks: xxHash64 with a round key.
fn feistel_round(x: u64, round_key: u64) -> u64 {
    xxh64(&x.to_le_bytes(), round_key)
}

/// 32-bit round function: hash and truncate.
fn feistel_round_32(x: u32, round_key: u64) -> u32 {
    xxh64(&x.to_le_bytes(), round_key) as u32
}

// --- 128-bit Feistel (4 rounds, 64-bit halves) ---

pub fn mix128(value: u128) -> u128 {
    let mut l = (value >> 64) as u64;
    let mut r = value as u64;
    l ^= feistel_round(r, 0);
    r ^= feistel_round(l, 1);
    l ^= feistel_round(r, 2);
    r ^= feistel_round(l, 3);
    ((l as u128) << 64) | (r as u128)
}

pub fn unmix128(value: u128) -> u128 {
    let mut l = (value >> 64) as u64;
    let mut r = value as u64;
    r ^= feistel_round(l, 3);
    l ^= feistel_round(r, 2);
    r ^= feistel_round(l, 1);
    l ^= feistel_round(r, 0);
    ((l as u128) << 64) | (r as u128)
}

// --- 64-bit Feistel (4 rounds, 32-bit halves) ---

pub fn mix64(value: u64) -> u64 {
    let mut l = (value >> 32) as u32;
    let mut r = value as u32;
    l ^= feistel_round_32(r, 10);
    r ^= feistel_round_32(l, 11);
    l ^= feistel_round_32(r, 12);
    r ^= feistel_round_32(l, 13);
    ((l as u64) << 32) | (r as u64)
}

pub fn unmix64(value: u64) -> u64 {
    let mut l = (value >> 32) as u32;
    let mut r = value as u32;
    r ^= feistel_round_32(l, 13);
    l ^= feistel_round_32(r, 12);
    r ^= feistel_round_32(l, 11);
    l ^= feistel_round_32(r, 10);
    ((l as u64) << 32) | (r as u64)
}

// --- 32-bit Feistel (4 rounds, 16-bit halves) ---

pub fn mix32(value: u32) -> u32 {
    let mut l = (value >> 16) as u16;
    let mut r = value as u16;
    l ^= xxh64(&r.to_le_bytes(), 20) as u16;
    r ^= xxh64(&l.to_le_bytes(), 21) as u16;
    l ^= xxh64(&r.to_le_bytes(), 22) as u16;
    r ^= xxh64(&l.to_le_bytes(), 23) as u16;
    ((l as u32) << 16) | (r as u32)
}

pub fn unmix32(value: u32) -> u32 {
    let mut l = (value >> 16) as u16;
    let mut r = value as u16;
    r ^= xxh64(&l.to_le_bytes(), 23) as u16;
    l ^= xxh64(&r.to_le_bytes(), 22) as u16;
    r ^= xxh64(&l.to_le_bytes(), 21) as u16;
    l ^= xxh64(&r.to_le_bytes(), 20) as u16;
    ((l as u32) << 16) | (r as u32)
}

/// Mix a u128 value within the given bit width.
/// Returns the mixed value with only the low `bits` populated.
pub fn mix_value(value: u128, bits: u32) -> u128 {
    match bits {
        0..=32 => mix32(value as u32) as u128,
        33..=64 => mix64(value as u64) as u128,
        _ => mix128(value),
    }
}

/// Unmix a u128 value within the given bit width.
pub fn unmix_value(value: u128, bits: u32) -> u128 {
    match bits {
        0..=32 => unmix32(value as u32) as u128,
        33..=64 => unmix64(value as u64) as u128,
        _ => unmix128(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_128() {
        for v in [0u128, 1, 42, u128::MAX, 0xDEADBEEF_CAFEBABE_12345678_9ABCDEF0] {
            assert_eq!(unmix128(mix128(v)), v);
        }
    }

    #[test]
    fn test_roundtrip_64() {
        for v in [0u64, 1, 42, u64::MAX, 0xDEADBEEFCAFEBABE] {
            assert_eq!(unmix64(mix64(v)), v);
        }
    }

    #[test]
    fn test_roundtrip_32() {
        for v in [0u32, 1, 42, u32::MAX, 0xDEADBEEF] {
            assert_eq!(unmix32(mix32(v)), v);
        }
    }

    #[test]
    fn test_roundtrip_value() {
        for v in [0u128, 1, 42, 1000, u32::MAX as u128, u64::MAX as u128, u128::MAX] {
            for bits in [32, 64, 128] {
                let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
                let original = v & mask;
                assert_eq!(unmix_value(mix_value(original, bits), bits), original);
            }
        }
    }

    #[test]
    fn test_bijective_128() {
        let inputs: Vec<u128> = (0..1000).map(|i| i as u128).collect();
        let outputs: std::collections::HashSet<u128> = inputs.iter().map(|&v| mix128(v)).collect();
        assert_eq!(outputs.len(), inputs.len());
    }

    #[test]
    fn test_bijective_64() {
        let inputs: Vec<u64> = (0..1000).collect();
        let outputs: std::collections::HashSet<u64> = inputs.iter().map(|&v| mix64(v)).collect();
        assert_eq!(outputs.len(), inputs.len());
    }

    #[test]
    fn test_bijective_32() {
        let inputs: Vec<u32> = (0..1000).collect();
        let outputs: std::collections::HashSet<u32> = inputs.iter().map(|&v| mix32(v)).collect();
        assert_eq!(outputs.len(), inputs.len());
    }

    #[test]
    fn test_avalanche_128() {
        let a = mix128(1000);
        let b = mix128(1001);
        assert!(
            (a ^ b).count_ones() > 40,
            "poor avalanche: {} bits differ",
            (a ^ b).count_ones()
        );
    }

    #[test]
    fn test_avalanche_64() {
        let a = mix64(1000);
        let b = mix64(1001);
        assert!(
            (a ^ b).count_ones() > 16,
            "poor avalanche: {} bits differ",
            (a ^ b).count_ones()
        );
    }
}
