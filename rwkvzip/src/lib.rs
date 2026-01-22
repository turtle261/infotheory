// rwkvzip - High-performance neural network compressor using RWKV7.
//
// This library provides lossless compression by leveraging the RWKV7 language model's
// predictive capabilities to generate probability distributions, which are then
// compressed via entropy coding (arithmetic coding or rANS).
//
// # Architecture
//
// - **Byte-level compression**: Operates directly on raw bytes (vocab_size=256)
// - **Infinite context**: RWKV7's recurrent architecture maintains state indefinitely
// - **x86_64 optimized**: AVX2/FMA SIMD throughout, no external BLAS dependencies
// - **Correct-by-construction**: Information-theoretically sound implementation

use anyhow::{bail, Result};
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::sync::Arc;

pub mod coders;
pub mod rwkv7;

use coders::{
    quantize_pdf_to_cdf_inplace, quantize_pdf_to_rans_cdf_with_buffer, softmax_pdf_floor_inplace,
    softmax_pdf_inplace, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder,
    BlockedRansEncoder, Cdf, ANS_TOTAL, CDF_TOTAL,
};

pub use rwkv7::{Config, Model, ScratchBuffers, State};

// =============================================================================
// File Format Constants
// =============================================================================

/// File format magic number: "GPTZ" in little-endian (0x47505A54 as ASCII).
/// Used to identify valid rwkvzip compressed files.
pub const MAGIC: u32 = 0x5a505447;

/// File format version. Increment on breaking changes to ensure compatibility.
pub const VERSION: u8 = 2;

/// Vocabulary size for byte-level compression.
/// Each byte (0-255) is treated as a separate symbol.
pub const VOCAB_SIZE: usize = 256;

// =============================================================================
// Entropy Coder Selection
// =============================================================================

/// Entropy coder type for compression.
///
/// Both coders are lossless and produce equivalent results; the choice
/// affects compression speed and ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CoderType {
    /// Arithmetic coding: optimal compression ratio, slightly slower.
    /// Recommended for small files or when compression ratio is critical.
    #[default]
    AC,
    /// rANS coding: near-optimal compression with better throughput.
    /// Recommended for larger files where speed matters more.
    RANS,
}

struct CountingWriter {
    n: u64,
}

impl CountingWriter {
    #[inline]
    fn new() -> Self {
        Self { n: 0 }
    }

    #[inline]
    fn bytes_written(&self) -> u64 {
        self.n
    }
}

impl Write for CountingWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = buf.len();
        self.n = self.n.saturating_add(n as u64);
        Ok(n)
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl std::fmt::Display for CoderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoderType::AC => write!(f, "AC"),
            CoderType::RANS => write!(f, "rANS"),
        }
    }
}

// =============================================================================
// File Header
// =============================================================================

/// Header structure for compressed data files.
///
/// Layout (18 bytes total):
/// - magic: 4 bytes (little-endian u32)
/// - version: 1 byte
/// - coder: 1 byte (0=AC, 1=rANS)
/// - original_len: 8 bytes (little-endian u64)
/// - crc32: 4 bytes (little-endian u32)
#[derive(Debug, Clone)]
pub struct Header {
    /// Magic number for format identification (must be MAGIC).
    pub magic: u32,
    /// Format version for compatibility checking.
    pub version: u8,
    /// Coder type used (0=AC, 1=rANS).
    pub coder: u8,
    /// Original uncompressed data length in bytes.
    pub original_len: u64,
    /// CRC32 checksum of original data for integrity verification.
    pub crc32: u32,
}

impl Header {
    /// Total header size in bytes.
    pub const SIZE: usize = 4 + 1 + 1 + 8 + 4; // 18 bytes

    /// Create a new header for compressed data.
    pub fn new(coder: CoderType, original_len: u64, crc32: u32) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            coder: match coder {
                CoderType::AC => 0,
                CoderType::RANS => 1,
            },
            original_len,
            crc32,
        }
    }

    /// Serialize header to a writer (little-endian format).
    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.magic.to_le_bytes())?;
        w.write_all(&[self.version])?;
        w.write_all(&[self.coder])?;
        w.write_all(&self.original_len.to_le_bytes())?;
        w.write_all(&self.crc32.to_le_bytes())?;
        Ok(())
    }

    /// Deserialize header from a reader (little-endian format).
    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        let mut buf1 = [0u8; 1];

        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != MAGIC {
            bail!(
                "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                MAGIC,
                magic
            );
        }

        r.read_exact(&mut buf1)?;
        let version = buf1[0];
        if version > VERSION {
            bail!(
                "Unsupported version: {} (max supported: {})",
                version,
                VERSION
            );
        }

        r.read_exact(&mut buf1)?;
        let coder = buf1[0];

        r.read_exact(&mut buf8)?;
        let original_len = u64::from_le_bytes(buf8);

        r.read_exact(&mut buf4)?;
        let crc32 = u32::from_le_bytes(buf4);

        Ok(Self {
            magic,
            version,
            coder,
            original_len,
            crc32,
        })
    }

    /// Get the coder type from the header byte.
    pub fn coder_type(&self) -> CoderType {
        match self.coder {
            0 => CoderType::AC,
            _ => CoderType::RANS,
        }
    }
}

// =============================================================================
// CRC32 Checksum
// =============================================================================

/// Compute CRC32 checksum for data integrity verification.
///
/// Uses the crc32fast crate for hardware-accelerated computation.
pub fn crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

// =============================================================================
// Compressor
// =============================================================================

/// Main compressor/decompressor that combines RWKV7 inference with entropy coding.
///
/// The compressor maintains internal state and pre-allocated buffers to minimize
/// allocations during the compression/decompression hot path.
pub struct Compressor {
    /// RWKV7 model for generating probability distributions.
    pub model: Arc<Model>,
    /// Model state (recurrent hidden states).
    pub state: State,
    pub scratch: ScratchBuffers,
    /// Pre-allocated PDF buffer (eliminates allocations in compression loop).
    pub pdf_buffer: Vec<f64>,
    /// Reusable AC CDF buffer (vocab_size + 1 entries).
    pub cdf_buffer_ac: Vec<u32>,
    /// Reusable rANS CDF buffer (vocab_size + 1 entries).
    pub cdf_buffer_rans: Vec<u32>,
    /// Scratch frequencies for rANS quantization.
    pub rans_freq_buffer: Vec<i64>,
}

impl Clone for Compressor {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_from_model(self.model.clone());
        cloned.state = self.state.clone();
        cloned.pdf_buffer.clone_from(&self.pdf_buffer);
        cloned.cdf_buffer_ac.clone_from(&self.cdf_buffer_ac);
        cloned.cdf_buffer_rans.clone_from(&self.cdf_buffer_rans);
        cloned.rans_freq_buffer.clone_from(&self.rans_freq_buffer);
        cloned
    }
}

impl Compressor {
    /// Create a new compressor with the given model.
    ///
    /// # Arguments
    /// * `model_path` - Path to RWKV7 model weights (.safetensors format)
    ///
    /// # Returns
    /// A new Compressor ready for compression/decompression operations.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model = Arc::new(Model::load(model_path)?);
        Ok(Self::new_from_model(model))
    }

    pub fn load_model<P: AsRef<Path>>(model_path: P) -> Result<Arc<Model>> {
        Ok(Arc::new(Model::load(model_path)?))
    }

    pub fn new_from_model(model: Arc<Model>) -> Self {
        let state = model.new_state();
        let vocab_size = model.config().vocab_size;
        let scratch = ScratchBuffers::new(model.config());
        Self {
            model,
            state,
            scratch,
            pdf_buffer: vec![0.0f64; vocab_size],
            cdf_buffer_ac: vec![0u32; vocab_size + 1],
            cdf_buffer_rans: vec![0u32; vocab_size + 1],
            rans_freq_buffer: vec![0i64; vocab_size],
        }
    }

    /// Reset the model state to initial values.
    ///
    /// Call this between independent compression/decompression operations
    /// to ensure a clean state.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Get the vocabulary size (should always be 256 for byte-level).
    pub fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    /// Compress data using the specified entropy coder.
    ///
    /// # Arguments
    /// * `data` - Raw bytes to compress
    /// * `coder` - Entropy coder to use (AC or rANS)
    ///
    /// # Returns
    /// Compressed data including header with checksum.
    pub fn compress(&mut self, data: &[u8], coder: CoderType) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        self.compress_into(data, coder, &mut output)?;
        Ok(output)
    }

    pub fn compress_into<W: Write>(
        &mut self,
        data: &[u8],
        coder: CoderType,
        w: &mut W,
    ) -> Result<()> {
        self.state.reset();

        let checksum = crc32(data);
        let header = Header::new(coder, data.len() as u64, checksum);
        header.write(w)?;

        match coder {
            CoderType::AC => self.compress_ac(data, w)?,
            CoderType::RANS => self.compress_rans(data, w)?,
        }

        Ok(())
    }

    pub fn compress_chain_into<W: Write>(
        &mut self,
        parts: &[&[u8]],
        coder: CoderType,
        w: &mut W,
    ) -> Result<()> {
        self.state.reset();

        let mut total_len: u64 = 0;
        let mut hasher = crc32fast::Hasher::new();
        for p in parts {
            total_len = total_len.saturating_add(p.len() as u64);
            hasher.update(p);
        }
        let checksum = hasher.finalize();

        let header = Header::new(coder, total_len, checksum);
        header.write(w)?;

        let it = parts.iter().flat_map(|p| p.iter().copied());
        match coder {
            CoderType::AC => self.compress_ac_iter(it, w)?,
            CoderType::RANS => self.compress_rans_iter(it, w)?,
        }

        Ok(())
    }

    pub fn compress_size(&mut self, data: &[u8], coder: CoderType) -> Result<u64> {
        let mut w = CountingWriter::new();
        self.compress_into(data, coder, &mut w)?;
        Ok(w.bytes_written())
    }

    pub fn compress_size_chain(&mut self, parts: &[&[u8]], coder: CoderType) -> Result<u64> {
        let mut w = CountingWriter::new();
        self.compress_chain_into(parts, coder, &mut w)?;
        Ok(w.bytes_written())
    }

    /// Compress using arithmetic coding.
    fn compress_ac<W: Write>(&mut self, data: &[u8], output: &mut W) -> Result<()> {
        self.compress_ac_iter(data.iter().copied(), output)
    }

    fn compress_ac_iter<I, W: Write>(&mut self, data: I, output: &mut W) -> Result<()>
    where
        I: IntoIterator<Item = u8>,
    {
        let mut encoder = ArithmeticEncoder::new(output);
        let vocab_size = self.vocab_size();

        // Prime the model with a null byte to establish initial state
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);

        for byte in data {
            quantize_pdf_to_cdf_inplace(&self.pdf_buffer, &mut self.cdf_buffer_ac);
            let sym = byte as usize;
            let c_lo = self.cdf_buffer_ac[sym] as u64;
            let c_hi = self.cdf_buffer_ac[sym + 1] as u64;
            encoder.encode_counts(c_lo, c_hi, CDF_TOTAL as u64)?;

            // Update model state with actual byte for next prediction
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        let _ = encoder.finish()?;
        Ok(())
    }

    /// Compress using rANS coding with block-based encoding.
    fn compress_rans<W: Write>(&mut self, data: &[u8], output: &mut W) -> Result<()> {
        self.compress_rans_iter(data.iter().copied(), output)
    }

    fn compress_rans_iter<I, W: Write>(&mut self, data: I, output: &mut W) -> Result<()>
    where
        I: IntoIterator<Item = u8>,
    {
        let vocab_size = self.vocab_size();

        // Use blocked encoder (128KB blocks) for streaming large files
        let mut encoder = BlockedRansEncoder::new();

        // Prime the model with a null byte
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);

        for byte in data {
            quantize_pdf_to_rans_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_rans,
                &mut self.rans_freq_buffer,
            );
            let sym = byte as usize;
            let cdf = Cdf::new(
                self.cdf_buffer_rans[sym],
                self.cdf_buffer_rans[sym + 1],
                ANS_TOTAL,
            );
            encoder.encode(cdf);

            // Update model state
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        // Finish encoding and write blocks
        let blocks = encoder.finish();

        // Write block count
        output.write_all(&(blocks.len() as u32).to_le_bytes())?;

        // Write each block with length prefix
        for block in &blocks {
            output.write_all(&(block.len() as u32).to_le_bytes())?;
            output.write_all(block)?;
        }

        Ok(())
    }

    /// Decompress data.
    ///
    /// # Arguments
    /// * `data` - Compressed data (must include header)
    ///
    /// # Returns
    /// Original decompressed data. Returns error if checksum doesn't match.
    pub fn decompress(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let mut cursor = Cursor::new(data);
        let header = Header::read(&mut cursor)?;

        self.state.reset();

        let compressed = &data[Header::SIZE..];
        let result = match header.coder_type() {
            CoderType::AC => self.decompress_ac(compressed, header.original_len as usize)?,
            CoderType::RANS => self.decompress_rans(compressed, header.original_len as usize)?,
        };

        // Verify checksum for data integrity
        let actual_crc = crc32(&result);
        if actual_crc != header.crc32 {
            bail!(
                "CRC32 mismatch: expected 0x{:08X}, got 0x{:08X}",
                header.crc32,
                actual_crc
            );
        }

        Ok(result)
    }

    /// Decompress using arithmetic coding.
    fn decompress_ac(&mut self, compressed: &[u8], original_len: usize) -> Result<Vec<u8>> {
        let mut decoder = ArithmeticDecoder::new(compressed)?;
        let vocab_size = self.vocab_size();

        let mut result = Vec::with_capacity(original_len);

        // Prime with null byte (must match compression)
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);

        for _ in 0..original_len {
            quantize_pdf_to_cdf_inplace(&self.pdf_buffer, &mut self.cdf_buffer_ac);
            let sym = decoder.decode_symbol_counts(&self.cdf_buffer_ac, CDF_TOTAL)?;
            result.push(sym as u8);

            // Update model state with decoded byte
            let logits = self
                .model
                .forward(&mut self.scratch, sym as u32, &mut self.state);
            softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        Ok(result)
    }

    /// Decompress using rANS coding.
    fn decompress_rans(&mut self, compressed: &[u8], original_len: usize) -> Result<Vec<u8>> {
        // Read block count
        if compressed.len() < 4 {
            bail!("rANS data too short");
        }
        let block_count =
            u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]])
                as usize;

        // Read blocks
        let mut blocks = Vec::with_capacity(block_count);
        let mut pos = 4;

        for _ in 0..block_count {
            if pos + 4 > compressed.len() {
                bail!("Truncated block header");
            }
            let block_len = u32::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
            ]) as usize;
            pos += 4;

            if pos + block_len > compressed.len() {
                bail!("Truncated block data");
            }
            blocks.push(&compressed[pos..pos + block_len]);
            pos += block_len;
        }

        // Decode using blocked decoder
        let mut decoder = BlockedRansDecoder::new(blocks);
        let vocab_size = self.vocab_size();
        let mut result = Vec::with_capacity(original_len);

        // Prime with null byte
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);

        for _ in 0..original_len {
            quantize_pdf_to_rans_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_rans,
                &mut self.rans_freq_buffer,
            );
            let sym = decoder.decode(&self.cdf_buffer_rans)?;
            result.push(sym as u8);

            // Update model state
            let logits = self
                .model
                .forward(&mut self.scratch, sym as u32, &mut self.state);
            softmax_pdf_floor_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        Ok(result)
    }

    /// Calculate cross-entropy (bits per byte) for data without compression.
    ///
    /// This measures how well the model predicts the data, giving a theoretical
    /// lower bound on achievable compression. Useful for evaluating model quality.
    ///
    /// # Arguments
    /// * `data` - Data to analyze
    ///
    /// # Returns
    /// Average bits per byte (lower is better, 8.0 means no compression possible).
    pub fn cross_entropy(&mut self, data: &[u8]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        self.state.reset();
        let vocab_size = self.vocab_size();

        let mut total_bits = 0.0f64;

        // Prime with null byte
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);

        for &byte in data {
            let p = self.pdf_buffer[byte as usize];
            total_bits -= p.log2();
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        Ok(total_bits / (data.len() as f64))
    }

    pub fn cross_entropy_conditional_chain(
        &mut self,
        prefix_parts: &[&[u8]],
        data: &[u8],
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        self.state.reset();
        let vocab_size = self.vocab_size();

        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);

        for p in prefix_parts {
            for &byte in *p {
                let logits = self
                    .model
                    .forward(&mut self.scratch, byte as u32, &mut self.state);
                softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);
            }
        }

        let mut total_bits = 0.0f64;
        for &byte in data {
            let p = self.pdf_buffer[byte as usize];
            total_bits -= p.log2();
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        Ok(total_bits / (data.len() as f64))
    }

    pub fn cross_entropy_conditional(&mut self, prefix: &[u8], data: &[u8]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        self.state.reset();
        let vocab_size = self.vocab_size();

        // Prime with null byte
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);

        // Condition on prefix (update state, no scoring)
        for &byte in prefix {
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        let mut total_bits = 0.0f64;
        for &byte in data {
            let p = self.pdf_buffer[byte as usize];
            total_bits -= p.log2();
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        Ok(total_bits / (data.len() as f64))
    }

    pub fn joint_cross_entropy_aligned_min(&mut self, x: &[u8], y: &[u8]) -> Result<f64> {
        let n = x.len().min(y.len());
        if n == 0 {
            return Ok(0.0);
        }

        let h_xy = self.joint_cross_entropy_aligned_order(x, y, false)?;
        let h_yx = self.joint_cross_entropy_aligned_order(x, y, true)?;
        Ok(h_xy.min(h_yx))
    }

    fn joint_cross_entropy_aligned_order(&mut self, x: &[u8], y: &[u8], swap: bool) -> Result<f64> {
        let n = x.len().min(y.len());
        if n == 0 {
            return Ok(0.0);
        }

        self.state.reset();
        let vocab_size = self.vocab_size();

        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);

        let mut total_bits = 0.0f64;
        for i in 0..n {
            let a = if swap { y[i] } else { x[i] };
            let b = if swap { x[i] } else { y[i] };

            let pa = self.pdf_buffer[a as usize];
            total_bits -= pa.log2();
            let logits = self
                .model
                .forward(&mut self.scratch, a as u32, &mut self.state);
            softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);

            let pb = self.pdf_buffer[b as usize];
            total_bits -= pb.log2();
            let logits = self
                .model
                .forward(&mut self.scratch, b as u32, &mut self.state);
            softmax_pdf_inplace(logits, vocab_size, &mut self.pdf_buffer);
        }

        Ok(total_bits / (n as f64))
    }
}

// =============================================================================
// Compression Statistics
// =============================================================================

/// Statistics from a compression operation.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes (including header).
    pub compressed_size: usize,
    /// Compression ratio (original/compressed). Higher is better.
    pub ratio: f64,
    /// Bits per byte. Lower is better (theoretical minimum: ~0, maximum: 8).
    pub bits_per_byte: f64,
    /// Time taken in seconds.
    pub time_seconds: f64,
    /// Throughput in bytes per second.
    pub throughput: f64,
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} bytes -> {} bytes | ratio={:.3} | bits/byte={:.3} | time={:.2}s | {:.0} B/s",
            self.original_size,
            self.compressed_size,
            self.ratio,
            self.bits_per_byte,
            self.time_seconds,
            self.throughput,
        )
    }
}

/// Compress data and return both the compressed output and statistics.
///
/// This is a convenience function that wraps `Compressor::compress` with timing.
pub fn compress_with_stats(
    compressor: &mut Compressor,
    data: &[u8],
    coder: CoderType,
) -> Result<(Vec<u8>, CompressionStats)> {
    let start = std::time::Instant::now();
    let compressed = compressor.compress(data, coder)?;
    let elapsed = start.elapsed().as_secs_f64();

    let stats = CompressionStats {
        original_size: data.len(),
        compressed_size: compressed.len(),
        ratio: data.len() as f64 / compressed.len() as f64,
        bits_per_byte: (compressed.len() as f64 * 8.0) / data.len() as f64,
        time_seconds: elapsed,
        throughput: data.len() as f64 / elapsed,
    };

    Ok((compressed, stats))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = Header::new(CoderType::AC, 12345, 0xDEADBEEF);

        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();

        assert_eq!(buf.len(), Header::SIZE);

        let mut cursor = Cursor::new(&buf);
        let read_header = Header::read(&mut cursor).unwrap();

        assert_eq!(read_header.magic, MAGIC);
        assert_eq!(read_header.version, VERSION);
        assert_eq!(read_header.coder, 0);
        assert_eq!(read_header.original_len, 12345);
        assert_eq!(read_header.crc32, 0xDEADBEEF);
    }

    #[test]
    fn test_header_rans() {
        let header = Header::new(CoderType::RANS, 67890, 0xCAFEBABE);
        assert_eq!(header.coder, 1);
        assert_eq!(header.coder_type(), CoderType::RANS);
    }

    #[test]
    fn test_coder_type_display() {
        assert_eq!(format!("{}", CoderType::AC), "AC");
        assert_eq!(format!("{}", CoderType::RANS), "rANS");
    }

    #[test]
    fn test_crc32() {
        let data = b"Hello, World!";
        let c = crc32(data);
        assert_ne!(c, 0);
        // CRC32 should be deterministic
        assert_eq!(c, crc32(data));
    }

    #[test]
    fn test_crc32_different_data() {
        let c1 = crc32(b"Hello");
        let c2 = crc32(b"World");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_crc32_known_vector() {
        // Standard CRC-32 (ISO-HDLC) test vector.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_header_rejects_invalid_magic() {
        let mut buf = Vec::new();
        let header = Header::new(CoderType::AC, 1, 2);
        header.write(&mut buf).unwrap();
        // Corrupt magic.
        buf[0] ^= 0xFF;

        let mut cursor = Cursor::new(&buf);
        let err = Header::read(&mut cursor).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("Invalid magic number"));
    }
}
