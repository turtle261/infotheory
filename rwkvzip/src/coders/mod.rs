//! Entropy coders for rwkvzip.
//!
//! This module provides both Arithmetic Coding (AC) and rANS coders.
//!
//! # Coder Selection
//!
//! - **Arithmetic Coding (AC)**: Optimal compression ratio, slightly slower.
//!   Best for small files or maximum compression.
//! - **rANS**: Near-optimal compression with better throughput, especially
//!   with SIMD on x86_64. Best for larger files.

pub mod ac;
pub mod rans;

// Re-export main types
pub use ac::{
    p_min, quantize_pdf_to_cdf, quantize_pdf_to_cdf_inplace, softmax_pdf, softmax_pdf_floor,
    softmax_pdf_floor_inplace, softmax_pdf_inplace, ArithmeticDecoder, ArithmeticEncoder,
    CDF_TOTAL,
};

pub use rans::{
    cdf_for_symbol, quantize_pdf_to_rans_cdf, quantize_pdf_to_rans_cdf_with_buffer,
    BlockedRansDecoder, BlockedRansEncoder, Cdf, RansDecoder, RansEncoder, ANS_BITS, ANS_HIGH,
    ANS_LOW, ANS_TOTAL, BLOCK_SIZE,
};

// SIMD types (x86_64 AVX2-enabled by default for this build)
pub use rans::{SimdRansDecoder, SimdRansEncoder, RANS_LANES};
