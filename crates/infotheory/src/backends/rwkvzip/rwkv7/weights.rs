//! Native safetensors weight loading for RWKV7.
//!
//! This module provides a zero-dependency safetensors parser optimized for loading
//! RWKV7 model weights. The implementation directly parses the safetensors JSON
//! header and efficiently loads FP32 tensor data.
//!
//! # File Format
//!
//! Safetensors files have a simple structure:
//! - 8 bytes: header length (little-endian u64)
//! - N bytes: JSON header containing tensor metadata
//! - Remaining bytes: contiguous tensor data
//!
//! The JSON header maps tensor names to their dtype, shape, and data_offsets.

use ahash::{HashMap, HashMapExt};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::tensor::{Tensor1D, Tensor2D};

// =============================================================================
// Tensor Metadata
// =============================================================================

/// Parsed metadata for a single tensor from the safetensors header.
#[derive(Debug, Clone)]
struct TensorMeta {
    /// Shape dimensions (e.g., [256, 256] for a 256x256 matrix).
    shape: Vec<usize>,
    /// Byte offset where tensor data begins (relative to data section start).
    offset_start: usize,
    /// Byte offset where tensor data ends (exclusive).
    offset_end: usize,
}

// =============================================================================
// Weight Storage
// =============================================================================

/// Loaded tensor with data and shape information.
pub struct WeightTensor {
    /// Raw FP32 data in row-major order.
    data: Vec<f32>,
    /// Shape dimensions.
    shape: Vec<usize>,
}

impl WeightTensor {
    /// Get the raw data slice.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get the shape dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// View as 1D slice.
    #[inline]
    pub fn as_1d(&self) -> &[f32] {
        &self.data
    }

    /// Iterate over rows for 2D access.
    pub fn as_2d(&self, rows: usize, cols: usize) -> impl Iterator<Item = &[f32]> {
        debug_assert_eq!(rows * cols, self.data.len());
        (0..rows).map(move |r| &self.data[r * cols..(r + 1) * cols])
    }
}

// =============================================================================
// Weights Container
// =============================================================================

/// Container for all loaded RWKV7 model weights.
pub struct Weights {
    pub(crate) tensors: HashMap<String, WeightTensor>,
}

impl Weights {
    /// Load weights from a safetensors file.
    ///
    /// This function:
    /// 1. Reads and parses the JSON header to extract tensor metadata
    /// 2. Loads each tensor's FP32 data into memory
    /// 3. Returns a Weights container for efficient tensor lookup
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open weights file: {}", path.display()))?;

        // Read header length (8-byte little-endian u64)
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // Read JSON header
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header_str =
            std::str::from_utf8(&header_bytes).context("Invalid UTF-8 in safetensors header")?;

        // Parse tensor metadata from JSON header
        let metas = parse_safetensors_header(header_str)?;

        // Data section starts after the 8-byte length + header
        let data_offset = 8 + header_len;

        // Load all tensors
        let mut tensors = HashMap::with_capacity(metas.len());
        for (name, meta) in metas {
            // Seek to tensor data
            file.seek(SeekFrom::Start((data_offset + meta.offset_start) as u64))?;

            let byte_len = meta.offset_end - meta.offset_start;
            let mut raw_bytes = vec![0u8; byte_len];
            file.read_exact(&mut raw_bytes)?;

            // Convert bytes to f32 (little-endian)
            let data = bytes_to_f32(&raw_bytes);

            tensors.insert(
                name,
                WeightTensor {
                    data,
                    shape: meta.shape,
                },
            );
        }

        Ok(Self { tensors })
    }

    /// Get a tensor by name, returning None if not found.
    #[inline]
    pub fn get(&self, name: &str) -> Option<&WeightTensor> {
        self.tensors.get(name)
    }

    /// Get a tensor by name, or return an error if not found.
    pub fn require(&self, name: &str) -> Result<&WeightTensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("Missing required tensor: {}", name))
    }

    /// Get a tensor as a 1D aligned tensor.
    pub fn get_1d(&self, name: &str) -> Result<Tensor1D> {
        let t = self.require(name)?;
        Ok(Tensor1D::from_vec(t.data.clone()))
    }

    /// Get a tensor as a 2D aligned tensor.
    pub fn get_2d(&self, name: &str) -> Result<Tensor2D> {
        let t = self.require(name)?;
        match t.shape.len() {
            1 => Ok(Tensor2D::from_vec(t.data.clone(), 1, t.shape[0])),
            2 => Ok(Tensor2D::from_vec(t.data.clone(), t.shape[0], t.shape[1])),
            _ => bail!(
                "Expected 1D or 2D tensor for '{}', got shape {:?}",
                name,
                t.shape
            ),
        }
    }

    /// Iterate over all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Print summary of all loaded tensors (for debugging).
    pub fn print_summary(&self) {
        let mut names: Vec<_> = self.tensors.keys().collect();
        names.sort();
        for name in names {
            let t = &self.tensors[name];
            println!("  {} {:?} ({} params)", name, t.shape, t.numel());
        }
    }
}

// =============================================================================
// JSON Header Parser
// =============================================================================

/// Parse the safetensors JSON header to extract tensor metadata.
///
/// This is a minimal, hand-written parser that only extracts the fields we need:
/// - shape: array of dimension sizes
/// - data_offsets: [start, end] byte offsets
///
/// We skip dtype since we assume all tensors are FP32.
fn parse_safetensors_header(json: &str) -> Result<HashMap<String, TensorMeta>> {
    let bytes = json.as_bytes();
    let mut pos = 0;
    let mut metas = HashMap::new();

    // Skip whitespace and opening brace
    skip_whitespace(bytes, &mut pos);
    expect_char(bytes, &mut pos, b'{')?;

    loop {
        skip_whitespace(bytes, &mut pos);

        // Check for end of object
        if pos < bytes.len() && bytes[pos] == b'}' {
            break;
        }

        // Skip comma between entries
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
            skip_whitespace(bytes, &mut pos);
        }

        // Parse tensor name
        let name = parse_string(bytes, &mut pos)?;

        // Skip __metadata__ entries
        if name == "__metadata__" {
            skip_whitespace(bytes, &mut pos);
            expect_char(bytes, &mut pos, b':')?;
            skip_json_value(bytes, &mut pos)?;
            continue;
        }

        skip_whitespace(bytes, &mut pos);
        expect_char(bytes, &mut pos, b':')?;
        skip_whitespace(bytes, &mut pos);

        // Parse tensor info object
        let meta = parse_tensor_info(bytes, &mut pos)?;
        metas.insert(name, meta);
    }

    Ok(metas)
}

/// Parse a tensor info object: { "dtype": "...", "shape": [...], "data_offsets": [...] }
fn parse_tensor_info(bytes: &[u8], pos: &mut usize) -> Result<TensorMeta> {
    expect_char(bytes, pos, b'{')?;

    let mut shape: Option<Vec<usize>> = None;
    let mut offset_start: Option<usize> = None;
    let mut offset_end: Option<usize> = None;

    loop {
        skip_whitespace(bytes, pos);

        if *pos < bytes.len() && bytes[*pos] == b'}' {
            *pos += 1;
            break;
        }

        if *pos < bytes.len() && bytes[*pos] == b',' {
            *pos += 1;
            skip_whitespace(bytes, pos);
        }

        let key = parse_string(bytes, pos)?;
        skip_whitespace(bytes, pos);
        expect_char(bytes, pos, b':')?;
        skip_whitespace(bytes, pos);

        match key.as_str() {
            "shape" => {
                shape = Some(parse_int_array(bytes, pos)?);
            }
            "data_offsets" => {
                let offsets = parse_int_array(bytes, pos)?;
                if offsets.len() >= 2 {
                    offset_start = Some(offsets[0]);
                    offset_end = Some(offsets[1]);
                }
            }
            _ => {
                // Skip dtype and any other fields
                skip_json_value(bytes, pos)?;
            }
        }
    }

    Ok(TensorMeta {
        shape: shape.unwrap_or_default(),
        offset_start: offset_start.unwrap_or(0),
        offset_end: offset_end.unwrap_or(0),
    })
}

/// Parse a JSON string (expects opening quote at current position).
fn parse_string(bytes: &[u8], pos: &mut usize) -> Result<String> {
    expect_char(bytes, pos, b'"')?;

    let start = *pos;
    while *pos < bytes.len() && bytes[*pos] != b'"' {
        if bytes[*pos] == b'\\' {
            *pos += 1; // Skip escape character
        }
        *pos += 1;
    }
    let end = *pos;

    expect_char(bytes, pos, b'"')?;

    String::from_utf8(bytes[start..end].to_vec()).context("Invalid UTF-8 in JSON string")
}

/// Parse a JSON array of integers.
fn parse_int_array(bytes: &[u8], pos: &mut usize) -> Result<Vec<usize>> {
    expect_char(bytes, pos, b'[')?;

    let mut result = Vec::new();

    loop {
        skip_whitespace(bytes, pos);

        if *pos < bytes.len() && bytes[*pos] == b']' {
            *pos += 1;
            break;
        }

        if *pos < bytes.len() && bytes[*pos] == b',' {
            *pos += 1;
            skip_whitespace(bytes, pos);
        }

        result.push(parse_int(bytes, pos)?);
    }

    Ok(result)
}

/// Parse a single integer.
fn parse_int(bytes: &[u8], pos: &mut usize) -> Result<usize> {
    let start = *pos;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }

    if start == *pos {
        bail!("Expected integer at position {}", *pos);
    }

    let s = std::str::from_utf8(&bytes[start..*pos])?;
    s.parse().context("Failed to parse integer")
}

/// Skip a JSON value (string, number, object, array, boolean, null).
fn skip_json_value(bytes: &[u8], pos: &mut usize) -> Result<()> {
    skip_whitespace(bytes, pos);

    if *pos >= bytes.len() {
        return Ok(());
    }

    match bytes[*pos] {
        b'"' => {
            // String
            *pos += 1;
            while *pos < bytes.len() && bytes[*pos] != b'"' {
                if bytes[*pos] == b'\\' {
                    *pos += 1;
                }
                *pos += 1;
            }
            *pos += 1; // Skip closing quote
        }
        b'{' => {
            // Object
            let mut depth = 1;
            *pos += 1;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    b'"' => {
                        *pos += 1;
                        while *pos < bytes.len() && bytes[*pos] != b'"' {
                            if bytes[*pos] == b'\\' {
                                *pos += 1;
                            }
                            *pos += 1;
                        }
                    }
                    _ => {}
                }
                *pos += 1;
            }
        }
        b'[' => {
            // Array
            let mut depth = 1;
            *pos += 1;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    b'"' => {
                        *pos += 1;
                        while *pos < bytes.len() && bytes[*pos] != b'"' {
                            if bytes[*pos] == b'\\' {
                                *pos += 1;
                            }
                            *pos += 1;
                        }
                    }
                    _ => {}
                }
                *pos += 1;
            }
        }
        _ => {
            // Number, boolean, or null
            while *pos < bytes.len() && !matches!(bytes[*pos], b',' | b'}' | b']') {
                *pos += 1;
            }
        }
    }

    Ok(())
}

/// Skip whitespace characters.
#[inline]
fn skip_whitespace(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

/// Expect a specific character at the current position.
#[inline]
fn expect_char(bytes: &[u8], pos: &mut usize, expected: u8) -> Result<()> {
    if *pos >= bytes.len() || bytes[*pos] != expected {
        bail!(
            "Expected '{}' at position {}, found '{}'",
            expected as char,
            *pos,
            bytes.get(*pos).map(|&b| b as char).unwrap_or('\0')
        );
    }
    *pos += 1;
    Ok(())
}

// =============================================================================
// Byte Conversion
// =============================================================================

/// Convert a byte slice to FP32 values (little-endian).
///
/// This is a hot path during model loading, so we process 4 floats at a time
/// to help the compiler vectorize.
#[inline]
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let num_floats = bytes.len() / 4;
    let mut result = vec![0.0f32; num_floats];

    // Process in chunks of 4 floats (16 bytes) for better vectorization
    let chunks = num_floats / 4;
    for i in 0..chunks {
        let base = i * 16;
        result[i * 4] = f32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ]);
        result[i * 4 + 1] = f32::from_le_bytes([
            bytes[base + 4],
            bytes[base + 5],
            bytes[base + 6],
            bytes[base + 7],
        ]);
        result[i * 4 + 2] = f32::from_le_bytes([
            bytes[base + 8],
            bytes[base + 9],
            bytes[base + 10],
            bytes[base + 11],
        ]);
        result[i * 4 + 3] = f32::from_le_bytes([
            bytes[base + 12],
            bytes[base + 13],
            bytes[base + 14],
            bytes[base + 15],
        ]);
    }

    // Handle remaining floats
    for (offset, out) in result.iter_mut().skip(chunks * 4).enumerate() {
        let base = (chunks * 4 + offset) * 4;
        *out = f32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ]);
    }

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_weights() {
        // This test requires a model file - skip if not available
        let path =
            std::env::var("RWKV_MODEL").unwrap_or_else(|_| "rwkv-10m.safetensors".to_string());

        if !Path::new(&path).exists() {
            eprintln!("Skipping test: model file not found at {}", path);
            return;
        }

        let weights = Weights::load(&path).unwrap();

        // Verify expected tensors exist
        assert!(weights.get("model.embeddings.weight").is_some());
        assert!(weights.get("lm_head.weight").is_some());

        // Verify embedding shape (256 vocab, 256 hidden)
        let emb = weights.get("model.embeddings.weight").unwrap();
        assert_eq!(emb.shape.len(), 2);
        assert_eq!(emb.shape[0], 256);
        assert_eq!(emb.shape[1], 256);
    }

    #[test]
    fn test_bytes_to_f32() {
        // Test basic conversion
        let bytes = [0x00, 0x00, 0x80, 0x3F]; // 1.0 in little-endian
        let result = bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);

        // Test multiple values
        let bytes = [
            0x00, 0x00, 0x80, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
        ];
        let result = bytes_to_f32(&bytes);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }
}
