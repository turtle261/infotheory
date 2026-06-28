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
    assert!(
        (n as u32) <= total,
        "CDF total {total} must be >= symbol count {n} to guarantee positive widths"
    );

    let scale = total as f64;
    let mut acc = 0.0f64;
    let mut prev = 0u32;

    unsafe {
        *cdf_out.get_unchecked_mut(0) = 0;
        for i in 0..n {
            let p = *pdf.get_unchecked(i);
            if p.is_finite() && p > 0.0 {
                acc += p;
            }

            let next = (acc * scale) as u32;
            if next <= prev || next > total {
                quantize_pdf_to_integer_cdf_positive_width(pdf, total, cdf_out);
                return;
            }
            *cdf_out.get_unchecked_mut(i + 1) = next;
            prev = next;
        }
        *cdf_out.get_unchecked_mut(n) = total;
    }
}

#[inline]
pub(crate) fn quantize_pdf_to_integer_cdf_dense_positive_with_buffer(
    pdf: &[f64],
    total: u32,
    cdf_out: &mut [u32],
) {
    let n = pdf.len();
    assert!(cdf_out.len() > n, "cdf buffer too small");

    if n == 0 {
        cdf_out[0] = 0;
        return;
    }
    assert!(
        (n as u32) <= total,
        "CDF total {total} must be >= symbol count {n} to guarantee positive widths"
    );

    debug_assert!(pdf.iter().all(|&p| p.is_finite() && p > 0.0));

    let scale = total as f64;
    let mut acc = 0.0f64;
    let mut prev = 0u32;

    unsafe {
        *cdf_out.get_unchecked_mut(0) = 0;
        for i in 0..n {
            acc += *pdf.get_unchecked(i);

            let next = (acc * scale) as u32;
            if next <= prev || next > total {
                quantize_pdf_to_integer_cdf_positive_width_dense(pdf, total, cdf_out);
                return;
            }
            *cdf_out.get_unchecked_mut(i + 1) = next;
            prev = next;
        }
        *cdf_out.get_unchecked_mut(n) = total;
    }
}

#[inline]
fn quantize_pdf_to_integer_cdf_positive_width(pdf: &[f64], total: u32, cdf_out: &mut [u32]) {
    let n = pdf.len();
    let scale = total as f64;
    let remaining_extra = total - (n as u32);
    let mut acc = 0.0f64;
    let mut extra = 0u32;

    unsafe {
        *cdf_out.get_unchecked_mut(0) = 0;
        for i in 0..n {
            let p = *pdf.get_unchecked(i);
            if p.is_finite() && p > 0.0 {
                acc += p;
            }

            let raw_extra = ((acc * scale) as u32).saturating_sub((i as u32) + 1);
            let capped_extra = raw_extra.min(remaining_extra);
            if capped_extra > extra {
                extra = capped_extra;
            }
            *cdf_out.get_unchecked_mut(i + 1) = extra + (i as u32) + 1;
        }
        *cdf_out.get_unchecked_mut(n) = total;
    }
}

#[inline]
fn quantize_pdf_to_integer_cdf_positive_width_dense(pdf: &[f64], total: u32, cdf_out: &mut [u32]) {
    let n = pdf.len();
    let scale = total as f64;
    let remaining_extra = total - (n as u32);
    let mut acc = 0.0f64;
    let mut extra = 0u32;

    unsafe {
        *cdf_out.get_unchecked_mut(0) = 0;
        for i in 0..n {
            acc += *pdf.get_unchecked(i);

            let raw_extra = ((acc * scale) as u32).saturating_sub((i as u32) + 1);
            let capped_extra = raw_extra.min(remaining_extra);
            if capped_extra > extra {
                extra = capped_extra;
            }
            *cdf_out.get_unchecked_mut(i + 1) = extra + (i as u32) + 1;
        }
        *cdf_out.get_unchecked_mut(n) = total;
    }
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
