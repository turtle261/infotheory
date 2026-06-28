//! Native safetensors loader for Mamba checkpoints.
//!
//! Supports `F32`, `F16`, and `BF16` tensor payloads and converts all data to
//! runtime `f32` for deterministic CPU inference.

use ahash::{HashMap, HashMapExt};
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Parsed metadata for a tensor in a safetensors header.
#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: DType,
    shape: Vec<usize>,
    offset_start: usize,
    offset_end: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DType {
    F32,
    F16,
    BF16,
}

/// Loaded tensor as f32 data with explicit shape.
pub struct WeightTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl WeightTensor {
    /// Tensor payload as contiguous f32 slice.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Tensor shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Container of all loaded tensors.
pub struct Weights {
    tensors: HashMap<String, WeightTensor>,
}

impl Weights {
    /// Load all tensors from a safetensors checkpoint.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .with_context(|| format!("failed to open weights file: {}", path.display()))?;

        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header_str =
            std::str::from_utf8(&header_bytes).context("invalid UTF-8 in safetensors header")?;

        let metas = parse_safetensors_header(header_str)?;
        let data_offset = 8 + header_len;

        let mut tensors = HashMap::with_capacity(metas.len());
        for (name, meta) in metas {
            file.seek(SeekFrom::Start((data_offset + meta.offset_start) as u64))?;
            let byte_len = meta.offset_end.saturating_sub(meta.offset_start);
            let mut raw = vec![0u8; byte_len];
            file.read_exact(&mut raw)?;

            let data = bytes_to_f32(&raw, meta.dtype)
                .with_context(|| format!("failed to decode tensor '{name}' as {:?}", meta.dtype))?;

            if product(&meta.shape) != data.len() {
                bail!(
                    "tensor '{}' shape {:?} expects {} elems, decoded {}",
                    name,
                    meta.shape,
                    product(&meta.shape),
                    data.len()
                );
            }

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

    /// Optional tensor lookup.
    #[inline]
    pub fn get(&self, name: &str) -> Option<&WeightTensor> {
        self.tensors.get(name)
    }

    /// Required tensor lookup.
    pub fn require(&self, name: &str) -> Result<&WeightTensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing required tensor: {name}"))
    }

    /// Iterate tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

fn product(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>()
}

fn parse_safetensors_header(json: &str) -> Result<HashMap<String, TensorMeta>> {
    let bytes = json.as_bytes();
    let mut pos = 0usize;
    let mut metas = HashMap::new();

    skip_ws(bytes, &mut pos);
    expect(bytes, &mut pos, b'{')?;

    loop {
        skip_ws(bytes, &mut pos);
        if pos < bytes.len() && bytes[pos] == b'}' {
            break;
        }
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
            skip_ws(bytes, &mut pos);
        }

        let name = parse_string(bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);
        expect(bytes, &mut pos, b':')?;
        skip_ws(bytes, &mut pos);

        if name == "__metadata__" {
            skip_json_value(bytes, &mut pos)?;
            continue;
        }

        let meta = parse_tensor_info(bytes, &mut pos)?;
        metas.insert(name, meta);
    }

    Ok(metas)
}

fn parse_tensor_info(bytes: &[u8], pos: &mut usize) -> Result<TensorMeta> {
    expect(bytes, pos, b'{')?;

    let mut dtype: Option<DType> = None;
    let mut shape: Option<Vec<usize>> = None;
    let mut offs: Option<(usize, usize)> = None;

    loop {
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'}' {
            *pos += 1;
            break;
        }
        if *pos < bytes.len() && bytes[*pos] == b',' {
            *pos += 1;
            skip_ws(bytes, pos);
        }

        let key = parse_string(bytes, pos)?;
        skip_ws(bytes, pos);
        expect(bytes, pos, b':')?;
        skip_ws(bytes, pos);

        match key.as_str() {
            "dtype" => {
                let d = parse_string(bytes, pos)?;
                dtype = Some(match d.as_str() {
                    "F32" => DType::F32,
                    "F16" => DType::F16,
                    "BF16" => DType::BF16,
                    other => bail!("unsupported dtype '{other}' in safetensors"),
                });
            }
            "shape" => {
                shape = Some(parse_usize_array(bytes, pos)?);
            }
            "data_offsets" => {
                let arr = parse_usize_array(bytes, pos)?;
                if arr.len() != 2 {
                    bail!("data_offsets must contain [start, end]");
                }
                offs = Some((arr[0], arr[1]));
            }
            _ => {
                skip_json_value(bytes, pos)?;
            }
        }
    }

    let (offset_start, offset_end) = offs.context("missing data_offsets")?;
    Ok(TensorMeta {
        dtype: dtype.context("missing dtype")?,
        shape: shape.context("missing shape")?,
        offset_start,
        offset_end,
    })
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

fn expect(bytes: &[u8], pos: &mut usize, ch: u8) -> Result<()> {
    if *pos >= bytes.len() || bytes[*pos] != ch {
        bail!(
            "JSON parse error at byte {}: expected '{}'",
            *pos,
            ch as char
        );
    }
    *pos += 1;
    Ok(())
}

fn parse_string(bytes: &[u8], pos: &mut usize) -> Result<String> {
    expect(bytes, pos, b'"')?;
    let mut out = String::new();
    while *pos < bytes.len() {
        let c = bytes[*pos];
        *pos += 1;
        match c {
            b'"' => return Ok(out),
            b'\\' => {
                if *pos >= bytes.len() {
                    bail!("unterminated string escape");
                }
                let esc = bytes[*pos];
                *pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if *pos + 4 > bytes.len() {
                            bail!("invalid unicode escape");
                        }
                        let h = std::str::from_utf8(&bytes[*pos..*pos + 4])?;
                        *pos += 4;
                        let cp = u16::from_str_radix(h, 16)
                            .with_context(|| format!("invalid unicode escape \\u{h}"))?;
                        if let Some(ch) = char::from_u32(cp as u32) {
                            out.push(ch);
                        }
                    }
                    _ => bail!("unsupported escape in JSON string"),
                }
            }
            _ => out.push(c as char),
        }
    }
    bail!("unterminated JSON string")
}

fn parse_usize_array(bytes: &[u8], pos: &mut usize) -> Result<Vec<usize>> {
    expect(bytes, pos, b'[')?;
    let mut out = Vec::new();
    loop {
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b']' {
            *pos += 1;
            break;
        }
        if *pos < bytes.len() && bytes[*pos] == b',' {
            *pos += 1;
            skip_ws(bytes, pos);
        }
        out.push(parse_usize(bytes, pos)?);
    }
    Ok(out)
}

fn parse_usize(bytes: &[u8], pos: &mut usize) -> Result<usize> {
    let start = *pos;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if start == *pos {
        bail!("expected integer at byte {}", start);
    }
    let s = std::str::from_utf8(&bytes[start..*pos])?;
    Ok(s.parse::<usize>()?)
}

fn skip_json_value(bytes: &[u8], pos: &mut usize) -> Result<()> {
    skip_ws(bytes, pos);
    if *pos >= bytes.len() {
        bail!("unexpected end of JSON");
    }
    match bytes[*pos] {
        b'{' => {
            *pos += 1;
            let mut depth = 1usize;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'"' => {
                        let _ = parse_string(bytes, pos)?;
                    }
                    b'{' => {
                        depth += 1;
                        *pos += 1;
                    }
                    b'}' => {
                        depth -= 1;
                        *pos += 1;
                    }
                    _ => *pos += 1,
                }
            }
            Ok(())
        }
        b'[' => {
            *pos += 1;
            let mut depth = 1usize;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'"' => {
                        let _ = parse_string(bytes, pos)?;
                    }
                    b'[' => {
                        depth += 1;
                        *pos += 1;
                    }
                    b']' => {
                        depth -= 1;
                        *pos += 1;
                    }
                    _ => *pos += 1,
                }
            }
            Ok(())
        }
        b'"' => {
            let _ = parse_string(bytes, pos)?;
            Ok(())
        }
        _ => {
            while *pos < bytes.len() {
                let c = bytes[*pos];
                if c == b',' || c == b'}' || c == b']' {
                    break;
                }
                *pos += 1;
            }
            Ok(())
        }
    }
}

fn bytes_to_f32(raw: &[u8], dtype: DType) -> Result<Vec<f32>> {
    match dtype {
        DType::F32 => {
            if !raw.len().is_multiple_of(4) {
                bail!("F32 payload length {} not divisible by 4", raw.len());
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        DType::F16 => {
            if !raw.len().is_multiple_of(2) {
                bail!("F16 payload length {} not divisible by 2", raw.len());
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for c in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        DType::BF16 => {
            if !raw.len().is_multiple_of(2) {
                bail!("BF16 payload length {} not divisible by 2", raw.len());
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for c in raw.chunks_exact(2) {
                let hi = u16::from_le_bytes([c[0], c[1]]) as u32;
                out.push(f32::from_bits(hi << 16));
            }
            Ok(out)
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x03ff;

    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // subnormal
            let mut frac_u = frac as u32;
            let mut e = -14i32;
            while (frac_u & 0x0400) == 0 {
                frac_u <<= 1;
                e -= 1;
            }
            frac_u &= 0x03ff;
            let exp_f32 = ((e + 127) as u32) << 23;
            sign | exp_f32 | (frac_u << 13)
        }
    } else if exp == 0x1f {
        // inf / NaN
        sign | 0x7f80_0000 | ((frac as u32) << 13)
    } else {
        let exp_f32 = (((exp as i32) - 15 + 127) as u32) << 23;
        sign | exp_f32 | ((frac as u32) << 13)
    };

    f32::from_bits(bits)
}
