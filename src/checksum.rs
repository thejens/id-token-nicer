/// FNV-1a 32-bit hash, returning only the low `bits` as a checksum.
/// Supports up to 16 bits of checksum.
pub fn fnv1a_checksum(data: &[u8], bits: u32) -> u16 {
    debug_assert!(bits > 0 && bits <= 16);
    let mask = (1u16 << bits) - 1;
    let mut h: u32 = 0x811c_9dc5; // FNV offset basis
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193); // FNV prime
    }
    (h as u16) & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"hello world";
        let c1 = fnv1a_checksum(data, 4);
        let c2 = fnv1a_checksum(data, 4);
        assert_eq!(c1, c2);
        assert!(c1 < 16);
    }

    #[test]
    fn test_fnv1a_various_widths() {
        let data = b"\x01\x02\x03\x04";
        assert!(fnv1a_checksum(data, 4) < 16);
        assert!(fnv1a_checksum(data, 6) < 64);
        assert!(fnv1a_checksum(data, 8) < 256);
        assert!(fnv1a_checksum(data, 12) < 4096);
        assert!(fnv1a_checksum(data, 15) < 32768);
    }
}
