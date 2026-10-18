//! CRC-32 (IEEE 802.3, reflected polynomial `0xEDB88320`), slicing-by-8.
//!
//! This is the checksum used by zlib, gzip and PNG. Slicing-by-8 (Kounavis &
//! Berry, 2008) processes eight bytes per step with eight lookup tables that
//! are computed at compile time.

const POLY: u32 = 0xEDB8_8320;

const TABLES: [[u32; 256]; 8] = build_tables();

const fn build_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0_u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }
    let mut t = 1;
    while t < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = tables[t - 1][i];
            tables[t][i] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            i += 1;
        }
        t += 1;
    }
    tables
}

/// Incremental CRC-32 state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Crc32(u32);

impl Crc32 {
    pub(crate) fn new() -> Self {
        Self(!0)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        let mut crc = self.0;
        let (chunks, tail) = bytes.as_chunks::<8>();
        for c in chunks {
            let lo = crc ^ u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            crc = TABLES[7][(lo & 0xFF) as usize]
                ^ TABLES[6][((lo >> 8) & 0xFF) as usize]
                ^ TABLES[5][((lo >> 16) & 0xFF) as usize]
                ^ TABLES[4][(lo >> 24) as usize]
                ^ TABLES[3][usize::from(c[4])]
                ^ TABLES[2][usize::from(c[5])]
                ^ TABLES[1][usize::from(c[6])]
                ^ TABLES[0][usize::from(c[7])];
        }
        for &b in tail {
            crc = (crc >> 8) ^ TABLES[0][((crc ^ u32::from(b)) & 0xFF) as usize];
        }
        self.0 = crc;
    }

    pub(crate) fn finish(self) -> u32 {
        !self.0
    }
}

/// CRC-32 of `bytes` in one call.
pub(crate) fn checksum(bytes: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(bytes);
    crc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bitwise reference implementation, one byte at a time.
    fn reference(bytes: &[u8]) -> u32 {
        let mut crc = !0_u32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ POLY
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[test]
    fn standard_check_value() {
        // The "check" value from the CRC catalog for CRC-32/ISO-HDLC.
        assert_eq!(checksum(b"123456789"), 0xCBF4_3926);
        assert_eq!(checksum(b""), 0);
        assert_eq!(
            checksum(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn matches_bitwise_reference_and_is_incremental() {
        let data: Vec<u8> = (0..1000_u32)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 3)) as u8)
            .collect();
        for len in [0, 1, 7, 8, 9, 63, 64, 65, 1000] {
            assert_eq!(checksum(&data[..len]), reference(&data[..len]), "len {len}");
        }
        for split in [0, 3, 8, 500, 999, 1000] {
            let mut crc = Crc32::new();
            crc.update(&data[..split]);
            crc.update(&data[split..]);
            assert_eq!(crc.finish(), checksum(&data), "split {split}");
        }
    }
}
