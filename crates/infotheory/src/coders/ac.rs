//! Arithmetic Coder implementation for rwkvzip.
//!
//! This implements a binary arithmetic coder with 32-bit precision, optimized for
//! neural network probability distributions. The implementation is mathematically
//! rigorous to ensure lossless compression.
//!
//! # Information-Theoretic Properties
//!
//! - Uses base-2 arithmetic for bitstream output
//! - 32-bit precision prevents underflow for typical neural network distributions
//! - Integer CDF quantization uses 30-bit total (2^30) to minimize quantization error
//! - Probability floor ensures no symbol has zero probability (critical for lossless)

use std::io::Write;
use wide::f32x8;
use wide::f64x4;

/// Total count for CDF quantization (2^30 for high precision)
pub const CDF_TOTAL: u32 = 1 << 30;

/// Divide `numerator` by `total`, using a bit shift whenever `total` is a
/// power of two.
///
/// `encode_counts`/`advance_counts` divide by the CDF `total`, which every
/// call site in this crate passes as [`CDF_TOTAL`] (`1 << 30`); a full u128
/// division there costs tens of cycles per call (there is no native 128-bit
/// divider), against a handful of cycles for `trailing_zeros` + shift. The
/// power-of-two check keeps the function correct for the fully general
/// `total` the public API contract allows, at the cost of one cheap branch
/// that is trivially predicted once `total` settles on `CDF_TOTAL`.
#[inline(always)]
fn div_by_total_u128(numerator: u128, total: u128) -> u128 {
    if total.is_power_of_two() {
        numerator >> total.trailing_zeros()
    } else {
        numerator / total
    }
}

/// Arithmetic coder precision in bits
const PRECISION: u32 = 32;

/// Base for arithmetic coding (binary)
const BASE: u64 = 2;

/// Returns the minimum probability floor for symbols.
/// P_MIN = 2 * 2^(-(PRECISION-2)) = 2^(-(PRECISION-3)) = 2^(-29)
#[inline]
pub fn p_min() -> f64 {
    // 2.0 * 2.0^(-(32-2)) = 2^(-29) ≈ 1.86e-9
    2.0f64.powi(-(PRECISION as i32 - 3))
}

/// Compute softmax PDF with probability floor.
///
/// # Arguments
/// * `logits` - Raw logits from the model
/// * `vocab_size` - Size of the vocabulary (256 for byte-level)
///
/// # Returns
/// Probability distribution with floor applied, normalized to sum to 1.
pub fn softmax_pdf_floor(logits: &[f32], vocab_size: usize) -> Vec<f64> {
    let mut result = vec![0f64; vocab_size];
    softmax_pdf_floor_inplace(logits, vocab_size, &mut result);
    result
}

/// In-place softmax from logits into a caller-provided `pdf_out` buffer.
///
/// `pdf_out.len()` must be at least `vocab_size`.
pub fn softmax_pdf_inplace(logits: &[f32], vocab_size: usize, pdf_out: &mut [f64]) {
    debug_assert!(pdf_out.len() >= vocab_size);

    let max = logits
        .iter()
        .take(vocab_size)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut sum = 0.0f64;
    for i in 0..vocab_size {
        let e = ((logits[i] - max) as f64).exp();
        pdf_out[i] = e;
        sum += e;
    }

    if sum > 0.0 {
        let inv = 1.0 / sum;
        for value in pdf_out.iter_mut().take(vocab_size) {
            *value *= inv;
        }
    } else {
        let inv = 1.0 / (vocab_size.max(1) as f64);
        for value in pdf_out.iter_mut().take(vocab_size) {
            *value = inv;
        }
    }
}

/// In-place version of softmax_pdf_floor to avoid allocations.
///
/// # Arguments
/// * `logits` - Raw logits from the model
/// * `vocab_size` - Size of the vocabulary
/// * `pdf_out` - Pre-allocated buffer for output PDF (length >= vocab_size)
pub fn softmax_pdf_floor_inplace(logits: &[f32], vocab_size: usize, pdf_out: &mut [f64]) {
    // Fast path for byte-level vocab using portable SIMD (`wide`).
    if vocab_size == 256 && logits.len() >= 256 && pdf_out.len() >= 256 {
        softmax_pdf_floor_wide_256(logits, pdf_out);
        return;
    }

    let p_min_val = p_min();

    // Find max for numerical stability
    let max = logits
        .iter()
        .take(vocab_size)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exp(x - max) and sum (reuse pdf_out as temp buffer)
    let mut sum = 0.0f64;
    for i in 0..vocab_size {
        let e = ((logits[i] - max) as f64).exp();
        pdf_out[i] = e;
        sum += e;
    }

    // Normalize and apply floor
    for value in pdf_out.iter_mut().take(vocab_size) {
        *value = (*value / sum).max(p_min_val);
    }

    // Re-normalize after floor application
    let norm: f64 = pdf_out[..vocab_size].iter().sum();
    for value in pdf_out.iter_mut().take(vocab_size) {
        *value /= norm;
    }
}

/// Portable SIMD softmax with probability floor for vocab_size=256.
#[inline]
fn softmax_pdf_floor_wide_256(logits: &[f32], pdf_out: &mut [f64]) {
    const N: usize = 256;
    debug_assert!(logits.len() >= N);
    debug_assert!(pdf_out.len() >= N);
    let p_min_val = p_min();

    #[inline(always)]
    unsafe fn load8(ptr: *const f32) -> f32x8 {
        ptr.cast::<f32x8>().read_unaligned()
    }

    let mut max_v = f32x8::splat(f32::NEG_INFINITY);
    for i in (0..N).step_by(8) {
        let v = unsafe { load8(logits.as_ptr().add(i)) };
        max_v = max_v.fast_max(v);
    }
    let mut max = f32::NEG_INFINITY;
    for x in max_v.to_array() {
        max = max.max(x);
    }
    let max_v = f32x8::splat(max);

    let mut sum4 = f64x4::ZERO;
    for (chunk_idx, out_chunk) in pdf_out[..N].chunks_exact_mut(8).enumerate() {
        let i = chunk_idx * 8;
        let centered = unsafe { load8(logits.as_ptr().add(i)) } - max_v;
        let exp_vals = centered.exp().to_array();
        let v0 = f64x4::new([
            exp_vals[0] as f64,
            exp_vals[1] as f64,
            exp_vals[2] as f64,
            exp_vals[3] as f64,
        ]);
        let v1 = f64x4::new([
            exp_vals[4] as f64,
            exp_vals[5] as f64,
            exp_vals[6] as f64,
            exp_vals[7] as f64,
        ]);
        sum4 += v0 + v1;

        let lanes0 = v0.to_array();
        let lanes1 = v1.to_array();
        out_chunk[..4].copy_from_slice(&lanes0);
        out_chunk[4..].copy_from_slice(&lanes1);
    }

    let sum_lanes = sum4.to_array();
    let sum = sum_lanes[0] + sum_lanes[1] + sum_lanes[2] + sum_lanes[3];

    let inv_sum = 1.0 / sum;
    let mut norm4 = f64x4::ZERO;
    let inv_sum4 = f64x4::splat(inv_sum);
    let min4 = f64x4::splat(p_min_val);

    for chunk in pdf_out[..N].chunks_exact_mut(4) {
        let vals = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mut v = vals * inv_sum4;
        v = v.max(min4);

        let lanes = v.to_array();
        chunk.copy_from_slice(&lanes);
        norm4 += v;
    }

    let norm_lanes = norm4.to_array();
    let norm = norm_lanes[0] + norm_lanes[1] + norm_lanes[2] + norm_lanes[3];

    let inv_norm = 1.0 / norm;
    let inv_norm4 = f64x4::splat(inv_norm);
    for chunk in pdf_out[..N].chunks_exact_mut(4) {
        let vals = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let out = vals * inv_norm4;
        let lanes = out.to_array();
        chunk.copy_from_slice(&lanes);
    }
}

/// Compute softmax PDF without floor (for entropy calculation).
pub fn softmax_pdf(logits: &[f32], vocab_size: usize) -> Vec<f64> {
    let max = logits
        .iter()
        .take(vocab_size)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut exps = vec![0f64; vocab_size];
    let mut sum = 0.0f64;
    for i in 0..vocab_size {
        let e = ((logits[i] - max) as f64).exp();
        exps[i] = e;
        sum += e;
    }

    if sum <= 0.0 {
        // Uniform distribution fallback
        let uniform = 1.0 / (vocab_size as f64);
        return vec![uniform; vocab_size];
    }

    let mut pdf = vec![0f64; vocab_size];
    for i in 0..vocab_size {
        pdf[i] = exps[i] / sum;
    }
    pdf
}

/// Quantize probability distribution to integer CDF.
///
/// The CDF is constructed to be monotonically non-decreasing with:
/// - `cdf[0] = 0`
/// - `cdf[vocab_size] = CDF_TOTAL`
///
/// # Arguments
/// * `pdf` - Probability distribution (must sum to ~1.0)
///
/// # Returns
/// Integer CDF with length vocab_size + 1
pub fn quantize_pdf_to_cdf(pdf: &[f64]) -> Vec<u32> {
    let mut cdf = vec![0u32; pdf.len() + 1];
    quantize_pdf_to_cdf_inplace(pdf, &mut cdf);
    cdf
}

/// Quantize PDF to integer CDF using a reusable output buffer.
///
/// `cdf_out` must have length at least `pdf.len() + 1`.
#[inline]
pub fn quantize_pdf_to_cdf_inplace(pdf: &[f64], cdf_out: &mut [u32]) {
    let mut unused_freq = [];
    super::quantize_pdf_to_integer_cdf_with_buffer(pdf, CDF_TOTAL, cdf_out, &mut unused_freq);
}

/// Quantize PDF to integer CDF using reusable output and frequency buffers.
#[inline]
pub fn quantize_pdf_to_cdf_with_buffer(pdf: &[f64], cdf_out: &mut [u32], freq_buf: &mut [i64]) {
    super::quantize_pdf_to_integer_cdf_with_buffer(pdf, CDF_TOTAL, cdf_out, freq_buf);
}

/// Binary arithmetic encoder.
pub struct ArithmeticEncoder<W: Write> {
    b_to_pm1: u64,
    b_to_pm2: u64,
    mask: u64,
    low: u64,
    high: u64,
    carry_run: u64,
    out: W,
    bit_buffer: u8,
    bit_count: u8,
    bytes_out: u64,
}

impl<W: Write> ArithmeticEncoder<W> {
    /// Create a new arithmetic encoder.
    pub fn new(out: W) -> Self {
        let b_to_pm1 = BASE.pow(PRECISION - 1);
        let b_to_pm2 = BASE.pow(PRECISION - 2);
        let mask = BASE.pow(PRECISION) - 1;
        Self {
            b_to_pm1,
            b_to_pm2,
            mask,
            low: 0,
            high: mask,
            carry_run: 0,
            out,
            bit_buffer: 0,
            bit_count: 0,
            bytes_out: 0,
        }
    }

    #[inline]
    fn write_byte(&mut self, byte: u8) -> anyhow::Result<()> {
        self.out.write_all(&[byte])?;
        self.bytes_out += 1;
        Ok(())
    }

    #[inline]
    fn put_bit_internal(&mut self, bit: u8) -> anyhow::Result<()> {
        self.bit_buffer = (self.bit_buffer << 1) | (bit & 1);
        self.bit_count += 1;
        if self.bit_count == 8 {
            let b = self.bit_buffer;
            self.write_byte(b)?;
            self.bit_buffer = 0;
            self.bit_count = 0;
        }
        Ok(())
    }

    #[inline]
    fn put_bit(&mut self, bit: u8) -> anyhow::Result<()> {
        self.put_bit_internal(bit)?;
        while self.carry_run > 0 {
            self.put_bit_internal((!bit) & 1)?;
            self.carry_run -= 1;
        }
        Ok(())
    }

    /// Encode a symbol using integer CDF bounds.
    ///
    /// # Arguments
    /// * `c_lo` - Lower CDF bound (cumulative probability before symbol)
    /// * `c_hi` - Upper CDF bound (cumulative probability including symbol)
    /// * `total` - Total CDF range (should be CDF_TOTAL)
    pub fn encode_counts(&mut self, c_lo: u64, c_hi: u64, total: u64) -> anyhow::Result<()> {
        let range = (self.high - self.low + 1) as u128;
        let total_u = total as u128;
        let c_lo_u = c_lo as u128;
        let c_hi_u = c_hi as u128;
        let low_u = self.low as u128;

        let new_low = low_u + div_by_total_u128(range * c_lo_u, total_u);
        let new_high = low_u + div_by_total_u128(range * c_hi_u, total_u) - 1;

        self.low = (new_low & (self.mask as u128)) as u64;
        self.high = (new_high & (self.mask as u128)) as u64;

        loop {
            if self.high < self.b_to_pm1 {
                self.put_bit(0)?;
            } else if self.low >= self.b_to_pm1 {
                self.put_bit(1)?;
                self.low -= self.b_to_pm1;
                self.high -= self.b_to_pm1;
            } else if self.low >= self.b_to_pm2 && self.high < self.b_to_pm2 * 3 {
                self.carry_run += 1;
                self.low -= self.b_to_pm2;
                self.high -= self.b_to_pm2;
            } else {
                break;
            }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
        }
        Ok(())
    }

    /// Encode a symbol given its PDF and symbol index.
    ///
    /// This is a convenience method that quantizes the PDF to CDF internally.
    pub fn encode_symbol(&mut self, pdf: &[f64], sym: usize) -> anyhow::Result<()> {
        let cdf = quantize_pdf_to_cdf(pdf);
        let c_lo = cdf[sym] as u64;
        let c_hi = cdf[sym + 1] as u64;
        self.encode_counts(c_lo, c_hi, CDF_TOTAL as u64)
    }

    /// Finish encoding and flush remaining bits.
    ///
    /// Returns the underlying writer.
    pub fn finish(mut self) -> anyhow::Result<W> {
        self.carry_run += 1;
        if self.low < self.b_to_pm2 {
            self.put_bit(0)?;
        } else {
            self.put_bit(1)?;
        }
        // Pad remaining bits
        if self.bit_count > 0 {
            let remaining = 8 - self.bit_count;
            for _ in 0..remaining {
                self.put_bit_internal(0)?;
            }
        }
        Ok(self.out)
    }

    /// Get the number of bytes written so far.
    #[inline]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_out
    }
}

/// Binary arithmetic decoder.
pub struct ArithmeticDecoder<'a> {
    b_to_pm1: u64,
    b_to_pm2: u64,
    mask: u64,
    low: u64,
    high: u64,
    code: u64,
    input: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> ArithmeticDecoder<'a> {
    /// Create a new arithmetic decoder from input bytes.
    pub fn new(input: &'a [u8]) -> anyhow::Result<Self> {
        let b_to_pm1 = BASE.pow(PRECISION - 1);
        let b_to_pm2 = BASE.pow(PRECISION - 2);
        let mask = BASE.pow(PRECISION) - 1;

        let mut s = Self {
            b_to_pm1,
            b_to_pm2,
            mask,
            low: 0,
            high: mask,
            code: 0,
            input,
            byte_pos: 0,
            bit_pos: 0,
        };

        // Initialize code register with first PRECISION bits
        for _ in 0..PRECISION {
            s.code = (s.code << 1) | (s.get_bit().unwrap_or(1) as u64);
        }

        Ok(s)
    }

    #[inline]
    fn get_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.input.len() {
            return None;
        }
        let byte = self.input[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }

    #[inline]
    fn advance_counts(&mut self, c_lo: u64, c_hi: u64, total: u32) {
        let range = (self.high - self.low + 1) as u128;
        let low_u = self.low as u128;
        let total_u128 = total as u128;
        let new_low = low_u + div_by_total_u128(range * (c_lo as u128), total_u128);
        let new_high = low_u + div_by_total_u128(range * (c_hi as u128), total_u128) - 1;

        self.low = new_low as u64;
        self.high = new_high as u64;

        loop {
            if self.high < self.b_to_pm1 {
                // Lower half; no offset adjustment is needed.
            } else if self.low >= self.b_to_pm1 {
                self.low -= self.b_to_pm1;
                self.high -= self.b_to_pm1;
                self.code -= self.b_to_pm1;
            } else if self.low >= self.b_to_pm2 && self.high < self.b_to_pm2 * 3 {
                self.low -= self.b_to_pm2;
                self.high -= self.b_to_pm2;
                self.code -= self.b_to_pm2;
            } else {
                break;
            }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
            self.code = ((self.code << 1) & self.mask) | (self.get_bit().unwrap_or(1) as u64);
        }
    }

    /// Decode a binary symbol from the two intervals `[0, split)` and
    /// `[split, total)`.
    ///
    /// This is equivalent to decoding with the CDF `[0, split, total]`, but it
    /// avoids the generic CDF search in bitwise arithmetic-coding hot paths.
    #[inline]
    pub(crate) fn decode_binary_counts(&mut self, split: u32, total: u32) -> anyhow::Result<u8> {
        debug_assert!(split > 0);
        debug_assert!(split < total);

        let total_u = total as u64;
        let range = self.high - self.low + 1;
        let value =
            (((self.code - self.low + 1) as u128 * (total_u as u128)) - 1) / (range as u128);
        let value_u = value as u32;

        if value_u < split {
            self.advance_counts(0, split as u64, total);
            Ok(0)
        } else {
            self.advance_counts(split as u64, total as u64, total);
            Ok(1)
        }
    }

    /// Decode a symbol using integer CDF.
    ///
    /// # Arguments
    /// * `cdf` - Cumulative distribution function (length = vocab_size + 1)
    /// * `total` - Total CDF range (should be CDF_TOTAL)
    ///
    /// # Returns
    /// The decoded symbol index.
    pub fn decode_symbol_counts(&mut self, cdf: &[u32], total: u32) -> anyhow::Result<usize> {
        let total_u = total as u64;
        let range = self.high - self.low + 1;
        let value =
            (((self.code - self.low + 1) as u128 * (total_u as u128)) - 1) / (range as u128);
        let value_u = value as u32;

        // Binary search for symbol `s` with `cdf[s] <= value < cdf[s+1]`
        let mut lo = 0usize;
        let mut hi = cdf.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid] <= value_u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let s = lo;
        let c_lo = cdf[s] as u64;
        let c_hi = cdf[s + 1] as u64;

        self.advance_counts(c_lo, c_hi, total);

        Ok(s)
    }

    /// Decode a symbol given a PDF.
    ///
    /// This is a convenience method that quantizes the PDF to CDF internally.
    pub fn decode_symbol(&mut self, pdf: &[f64]) -> anyhow::Result<usize> {
        let cdf = quantize_pdf_to_cdf(pdf);
        self.decode_symbol_counts(&cdf, CDF_TOTAL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_uniform() {
        // Test with uniform distribution
        let pdf = vec![0.25, 0.25, 0.25, 0.25];
        let symbols = vec![0, 1, 2, 3, 0, 1, 2, 3];

        // Encode
        let mut buf = Vec::new();
        let mut enc = ArithmeticEncoder::new(&mut buf);
        for &s in &symbols {
            enc.encode_symbol(&pdf, s).unwrap();
        }
        let buf = enc.finish().unwrap().to_vec();

        // Decode
        let mut dec = ArithmeticDecoder::new(&buf).unwrap();
        for &expected in &symbols {
            let got = dec.decode_symbol(&pdf).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_roundtrip_skewed() {
        // Test with skewed distribution
        let pdf = vec![0.7, 0.2, 0.05, 0.05];
        let symbols = vec![0, 0, 0, 1, 0, 2, 0, 3, 0, 0];

        // Encode
        let mut buf = Vec::new();
        let mut enc = ArithmeticEncoder::new(&mut buf);
        for &s in &symbols {
            enc.encode_symbol(&pdf, s).unwrap();
        }
        let buf = enc.finish().unwrap().to_vec();

        // Decode
        let mut dec = ArithmeticDecoder::new(&buf).unwrap();
        for &expected in &symbols {
            let got = dec.decode_symbol(&pdf).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_softmax_pdf_floor() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let pdf = softmax_pdf_floor(&logits, 4);

        // Check sum is ~1
        let sum: f64 = pdf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check all probabilities are >= floor
        let p_min_val = p_min();
        for &p in &pdf {
            assert!(p >= p_min_val);
        }
    }

    #[test]
    fn test_cdf_monotonic() {
        let pdf = vec![0.1, 0.2, 0.3, 0.4];
        let cdf = quantize_pdf_to_cdf(&pdf);

        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[4], CDF_TOTAL);

        // Check monotonicity
        for i in 1..cdf.len() {
            assert!(cdf[i] >= cdf[i - 1]);
        }
    }

    #[test]
    fn test_cdf_positive_width_for_tiny_positive_tail() {
        let tail = 1e-18;
        let head = 1.0 - (255.0 * tail);
        let mut pdf = vec![tail; 256];
        pdf[0] = head;
        let mut cdf = vec![0u32; 257];
        let mut freq = vec![0i64; 256];
        quantize_pdf_to_cdf_with_buffer(&pdf, &mut cdf, &mut freq);

        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[256], CDF_TOTAL);
        for i in 0..256 {
            assert!(cdf[i + 1] > cdf[i], "symbol {i} has zero-width interval");
        }
    }

    #[test]
    fn test_cdf_positive_width_when_mass_is_last_symbol() {
        let mut pdf = vec![0.0; 256];
        pdf[255] = 1.0;
        let mut cdf = vec![0u32; 257];
        let mut freq = vec![0i64; 256];
        quantize_pdf_to_cdf_with_buffer(&pdf, &mut cdf, &mut freq);

        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[255], 255);
        assert_eq!(cdf[256], CDF_TOTAL);
        for i in 0..256 {
            assert!(cdf[i + 1] > cdf[i], "symbol {i} has zero-width interval");
        }
    }

    #[test]
    #[should_panic]
    fn test_softmax_floor_256_short_logits_panics_safely() {
        // For vocab_size=256, short logits must not hit the SIMD fast path.
        // Safe fallback behavior is a normal Rust bounds panic in scalar code.
        let logits = vec![0.0f32; 255];
        let mut pdf_out = vec![0.0f64; 256];
        softmax_pdf_floor_inplace(&logits, 256, &mut pdf_out);
    }
}
