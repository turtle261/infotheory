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

#[inline]
pub(crate) fn quantize_pdf_to_integer_cdf_with_buffer(
    pdf: &[f64],
    total: u32,
    cdf_out: &mut [u32],
    _freq_buf: &mut [i64],
) {
    let n = pdf.len();
    assert!(cdf_out.len() > n, "cdf buffer too small");

    if n == 0 {
        cdf_out[0] = 0;
        return;
    }

    cdf_out[0] = 0;
    let scale = total as f64;
    let mut acc = 0.0f64;
    for i in 0..n {
        let p = pdf[i];
        if p.is_finite() && p > 0.0 {
            acc += p;
        }
        let mut next = (acc * scale) as u32;
        let min_next = cdf_out[i].saturating_add(1);
        let max_next = total.saturating_sub((n - i - 1) as u32);
        if next < min_next {
            next = min_next;
        } else if next > max_next {
            next = max_next;
        }
        cdf_out[i + 1] = next;
    }
    cdf_out[n] = total;
}

// Re-export main types
pub use ac::{
    ArithmeticDecoder, ArithmeticEncoder, CDF_TOTAL, p_min, quantize_pdf_to_cdf,
    quantize_pdf_to_cdf_inplace, quantize_pdf_to_cdf_with_buffer, softmax_pdf, softmax_pdf_floor,
    softmax_pdf_floor_inplace, softmax_pdf_inplace,
};

pub use rans::{
    ANS_BITS, ANS_HIGH, ANS_LOW, ANS_TOTAL, BLOCK_SIZE, BlockedRansDecoder, BlockedRansEncoder,
    Cdf, RansDecoder, RansEncoder, cdf_for_symbol, quantize_pdf_to_rans_cdf,
    quantize_pdf_to_rans_cdf_with_buffer,
};

// Interleaved multi-lane rANS types
pub use rans::{RANS_LANES, SimdRansDecoder, SimdRansEncoder};
