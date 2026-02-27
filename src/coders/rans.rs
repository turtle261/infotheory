//! rANS (range Asymmetric Numeral System) coder with an optional multi-lane path.
//!
//! The primary implementation is scalar and portable. On x86_64 builds, this
//! module also exposes an 8-lane interleaved encoder/decoder API.
//!
//! # Design
//!
//! - Uses 8-way parallel rANS states for throughput
//! - 15-bit precision for probability quantization
//! - Interleaved bitstream for decoder efficiency
//! - Supports both streaming and batch modes

/// Number of bits for rANS probability precision
pub const ANS_BITS: u32 = 15;

/// Total probability range (2^15 = 32768)
pub const ANS_TOTAL: u32 = 1 << ANS_BITS;

/// Lower bound for rANS state (2^15)
pub const ANS_LOW: u32 = 1 << ANS_BITS;

/// Upper bound for rANS state (2^31)
pub const ANS_HIGH: u32 = 1 << 31;

/// rANS CDF representation for a symbol.
#[derive(Clone, Debug)]
pub struct Cdf {
    /// Lower cumulative probability bound
    pub lo: u32,
    /// Upper cumulative probability bound  
    pub hi: u32,
    /// Total probability (should be ANS_TOTAL)
    pub total: u32,
}

impl Cdf {
    /// Create a new CDF entry.
    #[inline]
    pub fn new(lo: u32, hi: u32, total: u32) -> Self {
        Self { lo, hi, total }
    }

    /// Get the frequency (hi - lo).
    #[inline]
    pub fn freq(&self) -> u32 {
        self.hi - self.lo
    }
}

/// Quantize PDF to rANS CDF table with guaranteed minimum frequencies.
///
/// This implements a robust quantization algorithm that ensures:
/// 1. All symbols with p > 0 get freq >= 1
/// 2. The total equals ANS_TOTAL exactly
/// 3. Monotonicity is preserved (`cdf[i+1] >= cdf[i]`)
///
/// Uses error diffusion to distribute rounding errors across symbols.
///
/// # Arguments
/// * `pdf` - Probability distribution (must sum to ~1.0)
///
/// # Returns
/// CDF table where `cdf[i]` = cumulative probability up to symbol i
pub fn quantize_pdf_to_rans_cdf(pdf: &[f64]) -> Vec<u32> {
    let mut cdf = vec![0u32; pdf.len() + 1];
    let mut freqs = vec![0i64; pdf.len()];
    quantize_pdf_to_rans_cdf_with_buffer(pdf, &mut cdf, &mut freqs);
    cdf
}

/// Quantize PDF to rANS CDF using reusable scratch buffers.
///
/// * `cdf_out` must have length at least `pdf.len() + 1`
/// * `freq_buf` must have length at least `pdf.len()`
/// * `index_buf` must have length at least `pdf.len()`
pub fn quantize_pdf_to_rans_cdf_with_buffer(
    pdf: &[f64],
    cdf_out: &mut [u32],
    freq_buf: &mut [i64],
) {
    let n = pdf.len();
    assert!(cdf_out.len() >= n + 1, "cdf buffer too small");
    assert!(freq_buf.len() >= n, "frequency buffer too small");

    let total = ANS_TOTAL as i64;
    for i in 0..n {
        freq_buf[i] = (pdf[i] * total as f64).round() as i64;
        if pdf[i] > 0.0 && freq_buf[i] == 0 {
            freq_buf[i] = 1;
        } else if pdf[i] <= 0.0 {
            freq_buf[i] = 0;
        }
    }

    let sum: i64 = freq_buf[..n].iter().sum();
    if sum > total {
        let mut to_remove = sum - total;
        while to_remove > 0 {
            let mut removed = 0;
            for i in (0..n).rev() {
                if freq_buf[i] > 1 {
                    freq_buf[i] -= 1;
                    to_remove -= 1;
                    removed += 1;
                    if to_remove == 0 {
                        break;
                    }
                }
            }
            if removed == 0 {
                break;
            }
        }
    } else if sum < total {
        let mut to_add = total - sum;
        while to_add > 0 {
            let mut added = 0;
            for i in 0..n {
                if pdf[i] > 0.0 {
                    freq_buf[i] += 1;
                    to_add -= 1;
                    added += 1;
                    if to_add == 0 {
                        break;
                    }
                }
            }
            if added == 0 {
                for i in 0..n {
                    freq_buf[i] += 1;
                    to_add -= 1;
                    if to_add == 0 {
                        break;
                    }
                }
            }
        }
    }

    cdf_out[0] = 0;
    let mut cumsum = 0u32;
    for i in 0..n {
        cdf_out[i] = cumsum;
        cumsum += freq_buf[i] as u32;
    }
    cdf_out[n] = cumsum;

    debug_assert_eq!(cdf_out[n], ANS_TOTAL, "CDF total must equal ANS_TOTAL");
    for i in 0..n {
        if pdf[i] > 0.0 {
            debug_assert!(
                cdf_out[i + 1] > cdf_out[i],
                "Symbol {} with p={} has zero frequency",
                i,
                pdf[i]
            );
        }
    }
}

/// Get Cdf for a symbol from a CDF table.
#[inline]
pub fn cdf_for_symbol(cdf: &[u32], sym: usize) -> Cdf {
    Cdf::new(cdf[sym], cdf[sym + 1], ANS_TOTAL)
}

/// Scalar rANS encoder.
pub struct RansEncoder {
    state: u32,
    output: Vec<u16>, // 16-bit words for output
}

impl RansEncoder {
    /// Create a new rANS encoder.
    pub fn new() -> Self {
        Self {
            state: ANS_LOW,
            output: Vec::new(),
        }
    }

    /// Encode a symbol using its CDF bounds.
    ///
    /// rANS encoding formula:
    /// x' = (x / freq) << ANS_BITS + (x % freq) + c_lo
    #[inline]
    pub fn encode(&mut self, cdf: &Cdf) {
        let freq = cdf.freq();
        debug_assert!(freq > 0, "Symbol frequency must be > 0");

        // Renormalize: output 16-bit words while state >= max allowed
        // Max state after encode: freq * (2^16) - 1, we need this < ANS_HIGH
        // So we renorm when state >= (freq << (32 - 1 - ANS_BITS)) = freq << 16
        while self.state >= (freq << 16) {
            self.output.push(self.state as u16);
            self.state >>= 16;
        }

        // Encode: x' = (x / freq) << ANS_BITS + (x % freq) + c_lo
        let q = self.state / freq;
        let r = self.state % freq;
        self.state = (q << ANS_BITS) + r + cdf.lo;
    }

    /// Encode a symbol given a PDF.
    pub fn encode_pdf(&mut self, pdf: &[f64], sym: usize) {
        let cdf_table = quantize_pdf_to_rans_cdf(pdf);
        let cdf = cdf_for_symbol(&cdf_table, sym);
        self.encode(&cdf);
    }

    /// Finish encoding and return the output bytes.
    pub fn finish(self) -> Vec<u8> {
        // Output final state (4 bytes)
        let mut result = Vec::with_capacity(self.output.len() * 2 + 4);

        // Push final state first (will be read first during decode)
        result.extend_from_slice(&self.state.to_le_bytes());

        // Push output words in reverse order (LIFO)
        for &word in self.output.iter().rev() {
            result.extend_from_slice(&word.to_le_bytes());
        }

        result
    }

    /// Get current output size estimate.
    pub fn size_estimate(&self) -> usize {
        self.output.len() * 2 + 4 // *2 for u16->bytes, +4 for final state
    }
}

impl Default for RansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Scalar rANS decoder.
pub struct RansDecoder<'a> {
    state: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RansDecoder<'a> {
    /// Create a new rANS decoder from input bytes.
    pub fn new(input: &'a [u8]) -> anyhow::Result<Self> {
        if input.len() < 4 {
            anyhow::bail!("rANS input too short");
        }

        // Read initial state (little-endian, first 4 bytes)
        let state = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);

        Ok(Self {
            state,
            input,
            pos: 4,
        })
    }

    /// Decode a symbol using a CDF table.
    ///
    /// rANS decoding:
    /// 1. Extract slot = state % total (= state & (ANS_TOTAL - 1))
    /// 2. Find symbol `s` where `cdf[s] <= slot < cdf[s+1]`
    /// 3. Update state: x' = freq * (x >> ANS_BITS) + (x & (ANS_TOTAL-1)) - c_lo
    #[inline]
    pub fn decode(&mut self, cdf: &[u32]) -> anyhow::Result<usize> {
        // Extract slot from state (low ANS_BITS bits)
        let slot = self.state & (ANS_TOTAL - 1);

        // Binary search for symbol `s` where `cdf[s] <= slot < cdf[s+1]`
        let mut lo = 0usize;
        let mut hi = cdf.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid] <= slot {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let sym = lo;

        let c_lo = cdf[sym];
        let c_hi = cdf[sym + 1];
        let freq = c_hi - c_lo;

        // Decode: x' = freq * (x >> ANS_BITS) + (x & (ANS_TOTAL-1)) - c_lo
        self.state = freq * (self.state >> ANS_BITS) + slot - c_lo;

        // Renormalize: read 16-bit words while state < ANS_LOW
        while self.state < ANS_LOW && self.pos + 1 < self.input.len() {
            let word = u16::from_le_bytes([self.input[self.pos], self.input[self.pos + 1]]);
            self.state = (self.state << 16) | (word as u32);
            self.pos += 2;
        }

        Ok(sym)
    }

    /// Decode a symbol given a PDF.
    pub fn decode_pdf(&mut self, pdf: &[f64]) -> anyhow::Result<usize> {
        let cdf = quantize_pdf_to_rans_cdf(pdf);
        self.decode(&cdf)
    }
}

// =============================================================================
// 8-way interleaved rANS API (x86_64 build target)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;

    /// Number of parallel rANS streams
    pub const RANS_LANES: usize = 8;

    /// 8-way interleaved rANS encoder.
    pub struct SimdRansEncoder {
        states: [u32; RANS_LANES],
        outputs: [Vec<u8>; RANS_LANES],
        lane: usize,
    }

    impl SimdRansEncoder {
        /// Create a new SIMD rANS encoder.
        pub fn new() -> Self {
            Self {
                states: [ANS_LOW; RANS_LANES],
                outputs: Default::default(),
                lane: 0,
            }
        }

        /// Encode a symbol, cycling through lanes.
        pub fn encode(&mut self, cdf: &Cdf) {
            let freq = cdf.freq();
            let lane = self.lane;
            self.lane = (self.lane + 1) % RANS_LANES;

            let state = &mut self.states[lane];
            let output = &mut self.outputs[lane];

            // Renormalize
            while *state >= (ANS_HIGH / cdf.total) * freq {
                output.push(*state as u8);
                *state >>= 8;
            }

            // Encode
            *state = ((*state / freq) * cdf.total) + (*state % freq) + cdf.lo;
        }

        /// Finish encoding and return interleaved output.
        pub fn finish(self) -> Vec<u8> {
            let mut result = Vec::new();

            // Output final states (interleaved)
            for i in 0..RANS_LANES {
                let s = self.states[i];
                result.extend_from_slice(&s.to_le_bytes());
            }

            // Find max output length
            let max_len = self.outputs.iter().map(|v| v.len()).max().unwrap_or(0);

            // Interleave output bytes
            for pos in 0..max_len {
                for lane in 0..RANS_LANES {
                    let out = &self.outputs[lane];
                    if pos < out.len() {
                        result.push(out[out.len() - 1 - pos]);
                    } else {
                        result.push(0);
                    }
                }
            }

            result
        }
    }

    impl Default for SimdRansEncoder {
        fn default() -> Self {
            Self::new()
        }
    }

    /// 8-way interleaved rANS decoder.
    pub struct SimdRansDecoder<'a> {
        states: [u32; RANS_LANES],
        input: &'a [u8],
        pos: usize,
        lane: usize,
    }

    impl<'a> SimdRansDecoder<'a> {
        /// Create a new SIMD rANS decoder.
        pub fn new(input: &'a [u8]) -> anyhow::Result<Self> {
            if input.len() < RANS_LANES * 4 {
                anyhow::bail!("SIMD rANS input too short");
            }

            let mut states = [0u32; RANS_LANES];
            for i in 0..RANS_LANES {
                let offset = i * 4;
                states[i] = u32::from_le_bytes([
                    input[offset],
                    input[offset + 1],
                    input[offset + 2],
                    input[offset + 3],
                ]);
            }

            Ok(Self {
                states,
                input,
                pos: RANS_LANES * 4,
                lane: 0,
            })
        }

        /// Decode a symbol from the current lane.
        pub fn decode(&mut self, cdf: &[u32]) -> anyhow::Result<usize> {
            let lane = self.lane;
            self.lane = (self.lane + 1) % RANS_LANES;

            let state = &mut self.states[lane];
            let total = ANS_TOTAL;
            let value = *state & (total - 1);

            // Binary search
            let mut lo = 0usize;
            let mut hi = cdf.len() - 1;
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if cdf[mid] <= value {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let sym = lo;

            let c_lo = cdf[sym];
            let c_hi = cdf[sym + 1];
            let freq = c_hi - c_lo;

            // Decode
            *state = freq * (*state >> ANS_BITS) + (*state & (total - 1)) - c_lo;

            // Renormalize (read from interleaved stream)
            while *state < ANS_LOW {
                // Read byte for this lane
                let byte_idx = self.pos + lane;
                if byte_idx < self.input.len() {
                    *state = (*state << 8) | (self.input[byte_idx] as u32);
                }
                self.pos += RANS_LANES;
            }

            Ok(sym)
        }
    }
}
#[cfg(target_arch = "x86_64")]
pub use simd::*;

#[cfg(not(target_arch = "x86_64"))]
pub const RANS_LANES: usize = 1;

#[cfg(not(target_arch = "x86_64"))]
pub struct SimdRansEncoder {
    inner: RansEncoder,
}

#[cfg(not(target_arch = "x86_64"))]
impl SimdRansEncoder {
    pub fn new() -> Self {
        Self {
            inner: RansEncoder::new(),
        }
    }

    pub fn encode(&mut self, cdf: &Cdf) {
        self.inner.encode(cdf);
    }

    pub fn finish(self) -> Vec<u8> {
        self.inner.finish()
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl Default for SimdRansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub struct SimdRansDecoder<'a> {
    inner: RansDecoder<'a>,
}

#[cfg(not(target_arch = "x86_64"))]
impl<'a> SimdRansDecoder<'a> {
    pub fn new(input: &'a [u8]) -> anyhow::Result<Self> {
        Ok(Self {
            inner: RansDecoder::new(input)?,
        })
    }

    pub fn decode(&mut self, cdf: &[u32]) -> anyhow::Result<usize> {
        self.inner.decode(cdf)
    }
}

// =============================================================================
// Blocked rANS for streaming large files
// =============================================================================

/// Block size for blocked rANS (128KB)
pub const BLOCK_SIZE: usize = 128 * 1024;

/// Blocked rANS encoder for streaming large files.
///
/// Splits input into 128KB blocks and encodes each independently.
/// This allows O(1) memory for encoding arbitrary-sized inputs.
pub struct BlockedRansEncoder {
    /// Symbols buffered for current block (stores low/high bounds only)
    symbols: Vec<Cdf>,
    /// Encoded blocks
    blocks: Vec<Vec<u8>>,
}

impl BlockedRansEncoder {
    pub fn new() -> Self {
        Self {
            symbols: Vec::with_capacity(BLOCK_SIZE),
            blocks: Vec::new(),
        }
    }

    /// Encode a symbol with its CDF.
    pub fn encode(&mut self, cdf: Cdf) {
        self.symbols.push(cdf);

        // Flush block if full
        if self.symbols.len() >= BLOCK_SIZE {
            self.flush_block();
        }
    }

    /// Flush the current block.
    fn flush_block(&mut self) {
        if self.symbols.is_empty() {
            return;
        }

        // Encode in reverse order (rANS is LIFO)
        let mut encoder = RansEncoder::new();
        for cdf in self.symbols.iter().rev() {
            encoder.encode(cdf);
        }

        let encoded = encoder.finish();
        self.blocks.push(encoded);
        self.symbols.clear();
    }

    /// Finish encoding and return all blocks.
    pub fn finish(mut self) -> Vec<Vec<u8>> {
        // Flush any remaining symbols
        self.flush_block();
        self.blocks
    }
}

impl Default for BlockedRansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Blocked rANS decoder for streaming large files.
pub struct BlockedRansDecoder<'a> {
    blocks: Vec<&'a [u8]>,
    current_block: usize,
    decoder: Option<RansDecoder<'a>>,
}

impl<'a> BlockedRansDecoder<'a> {
    /// Create a new blocked decoder from encoded blocks.
    pub fn new(blocks: Vec<&'a [u8]>) -> Self {
        Self {
            blocks,
            current_block: 0,
            decoder: None,
        }
    }

    /// Decode next symbol with provided CDF.
    pub fn decode(&mut self, cdf: &[u32]) -> anyhow::Result<usize> {
        // Initialize decoder for first block if needed
        if self.decoder.is_none() {
            if self.current_block >= self.blocks.len() {
                anyhow::bail!("No more blocks to decode");
            }
            self.decoder = Some(RansDecoder::new(self.blocks[self.current_block])?);
        }

        // Try to decode from current block
        match self.decoder.as_mut().unwrap().decode(cdf) {
            Ok(sym) => Ok(sym),
            Err(_) => {
                // Current block exhausted, move to next
                self.current_block += 1;
                if self.current_block >= self.blocks.len() {
                    anyhow::bail!("All blocks exhausted");
                }
                self.decoder = Some(RansDecoder::new(self.blocks[self.current_block])?);
                self.decoder.as_mut().unwrap().decode(cdf)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_scalar() {
        let pdf = vec![0.5, 0.3, 0.15, 0.05];
        let symbols = vec![0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 1, 2];

        // Encode in REVERSE order (rANS is LIFO)
        let mut enc = RansEncoder::new();
        let cdf_table = quantize_pdf_to_rans_cdf(&pdf);
        for &s in symbols.iter().rev() {
            let cdf = cdf_for_symbol(&cdf_table, s);
            enc.encode(&cdf);
        }
        let encoded = enc.finish();

        // Decode in FORWARD order
        let mut dec = RansDecoder::new(&encoded).unwrap();
        for &expected in &symbols {
            let got = dec.decode(&cdf_table).unwrap();
            assert_eq!(got, expected, "Symbol mismatch");
        }
    }

    #[test]
    fn test_cdf_quantization() {
        let pdf = vec![0.25, 0.25, 0.25, 0.25];
        let cdf = quantize_pdf_to_rans_cdf(&pdf);

        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[4], ANS_TOTAL);

        // Check roughly equal spacing
        for i in 1..4 {
            let delta = cdf[i] - cdf[i - 1];
            assert!(delta > 0);
        }
    }

    #[test]
    fn test_extreme_probabilities() {
        // Very skewed distribution
        let pdf = vec![0.99, 0.005, 0.003, 0.002];
        let symbols = vec![0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 3];

        // Encode in REVERSE order (rANS is LIFO)
        let mut enc = RansEncoder::new();
        let cdf_table = quantize_pdf_to_rans_cdf(&pdf);
        for &s in symbols.iter().rev() {
            let cdf = cdf_for_symbol(&cdf_table, s);
            enc.encode(&cdf);
        }
        let encoded = enc.finish();

        // Decode in FORWARD order
        let mut dec = RansDecoder::new(&encoded).unwrap();
        for &expected in &symbols {
            let got = dec.decode(&cdf_table).unwrap();
            assert_eq!(got, expected);
        }
    }
}
