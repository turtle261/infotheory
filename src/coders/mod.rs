//! Entropy coders for rwkvzip.
//!
//! This module provides both Arithmetic Coding (AC) and rANS coders.
//!
//! # Coder Selection
//!
//! - **Arithmetic Coding (AC)**: Optimal compression ratio, slightly slower.
//!   Best for small files or maximum compression.
//! - **rANS**: Near-optimal compression with better throughput, especially
//!   with lane-interleaved encoding. Best for larger files.

pub mod ac;
pub mod rans;

/// Entropy coder type used by generic rate-coded compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoderType {
    /// Arithmetic coding: optimal compression ratio, slightly slower.
    #[default]
    AC,
    /// rANS coding: near-optimal compression with better throughput.
    RANS,
}

impl std::fmt::Display for CoderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoderType::AC => write!(f, "AC"),
            CoderType::RANS => write!(f, "rANS"),
        }
    }
}

/// Compute CRC32 checksum for data integrity verification.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

// Re-export main types
pub use ac::{
    ArithmeticDecoder, ArithmeticEncoder, CDF_TOTAL, p_min, quantize_pdf_to_cdf,
    quantize_pdf_to_cdf_inplace, softmax_pdf, softmax_pdf_floor, softmax_pdf_floor_inplace,
    softmax_pdf_inplace,
};

pub use rans::{
    ANS_BITS, ANS_HIGH, ANS_LOW, ANS_TOTAL, BLOCK_SIZE, BlockedRansDecoder, BlockedRansEncoder,
    Cdf, RansDecoder, RansEncoder, cdf_for_symbol, quantize_pdf_to_rans_cdf,
    quantize_pdf_to_rans_cdf_with_buffer,
};

// Interleaved multi-lane rANS types
pub use rans::{RANS_LANES, SimdRansDecoder, SimdRansEncoder};
