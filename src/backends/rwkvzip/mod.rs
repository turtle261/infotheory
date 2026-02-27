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
// - **Portable SIMD optimized**: `wide`-based kernels with ISA-specific codegen
// - **Correct-by-construction**: Information-theoretically sound implementation

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod rwkv7;
pub use crate::coders;

use crate::coders::{
    ANS_TOTAL, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder, BlockedRansEncoder,
    CDF_TOTAL, Cdf, quantize_pdf_to_cdf_inplace, quantize_pdf_to_rans_cdf_with_buffer,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnlineTrainMode {
    None,
    Sgd,
    Adam,
}

#[derive(Clone, Debug)]
pub struct OnlineConfig {
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub decay_rank: usize,
    pub a_rank: usize,
    pub v_rank: usize,
    pub g_rank: usize,
    pub seed: u64,
    pub train_mode: OnlineTrainMode,
    pub lr: f32,
    pub stride: usize,
}

impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            hidden: 256,
            layers: 6,
            intermediate: 1024,
            decay_rank: 32,
            a_rank: 32,
            v_rank: 32,
            g_rank: 64,
            seed: 0,
            train_mode: OnlineTrainMode::None,
            lr: 0.001,
            stride: 1,
        }
    }
}

impl OnlineConfig {
    pub fn to_rwkv_config(&self) -> Result<Config> {
        let hidden = self.hidden.max(64);
        if !hidden.is_multiple_of(64) {
            bail!("rwkv hidden must be a multiple of 64 (got {hidden})");
        }
        let num_heads = hidden / 64;
        let cfg = Config {
            vocab_size: 256,
            hidden_size: hidden,
            num_layers: self.layers.max(1),
            num_heads,
            head_dim: 64,
            intermediate_size: self.intermediate.max(1),
            layer_norm_eps: 1e-5,
            group_norm_eps: 64e-5,
            decay_low_rank: self.decay_rank.max(1),
            a_low_rank: self.a_rank.max(1),
            v_low_rank: self.v_rank.max(1),
            g_low_rank: self.g_rank.max(1),
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

#[derive(Clone, Debug)]
pub enum MethodSpec {
    File(PathBuf),
    Online(OnlineConfig),
}

#[derive(Clone, Debug)]
struct OnlineRuntime {
    cfg: OnlineConfig,
    canonical_cfg: String,
    tokens_processed: u64,
    out_bias: Vec<f32>,
    adam_m: Option<Vec<f32>>,
    adam_v: Option<Vec<f32>>,
    adam_t: usize,
}

#[derive(Clone)]
pub struct RuntimeSnapshot {
    state: State,
    pdf_buffer: Vec<f64>,
    online: Option<OnlineRuntime>,
}

impl OnlineRuntime {
    fn new(cfg: OnlineConfig, vocab_size: usize) -> Self {
        let use_adam = matches!(cfg.train_mode, OnlineTrainMode::Adam);
        Self {
            canonical_cfg: cfg_to_method_string(&cfg),
            cfg,
            tokens_processed: 0,
            out_bias: vec![0.0; vocab_size],
            adam_m: use_adam.then(|| vec![0.0; vocab_size]),
            adam_v: use_adam.then(|| vec![0.0; vocab_size]),
            adam_t: 0,
        }
    }
}

fn cfg_to_method_string(cfg: &OnlineConfig) -> String {
    let train = match cfg.train_mode {
        OnlineTrainMode::None => "none",
        OnlineTrainMode::Sgd => "sgd",
        OnlineTrainMode::Adam => "adam",
    };
    format!(
        "cfg:hidden={},layers={},intermediate={},decay_rank={},a_rank={},v_rank={},g_rank={},seed={},train={},lr={},stride={}",
        cfg.hidden,
        cfg.layers,
        cfg.intermediate,
        cfg.decay_rank,
        cfg.a_rank,
        cfg.v_rank,
        cfg.g_rank,
        cfg.seed,
        train,
        cfg.lr,
        cfg.stride.max(1),
    )
}

fn softmax_pdf_floor_with_bias(logits: &[f32], bias: Option<&[f32]>, pdf_out: &mut [f64]) {
    debug_assert_eq!(logits.len(), pdf_out.len());
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), logits.len());
    }
    if logits.is_empty() {
        return;
    }

    let mut max_logit = f32::NEG_INFINITY;
    for i in 0..logits.len() {
        let z = logits[i] + bias.map_or(0.0, |b| b[i]);
        if z > max_logit {
            max_logit = z;
        }
    }

    let mut sum = 0.0f64;
    for i in 0..logits.len() {
        let z = logits[i] + bias.map_or(0.0, |b| b[i]);
        let p = ((z - max_logit) as f64).exp();
        pdf_out[i] = p;
        sum += p;
    }

    let inv_sum = if sum.is_finite() && sum > 0.0 {
        1.0 / sum
    } else {
        1.0 / (logits.len() as f64)
    };

    let floor = 1e-12f64;
    let mut norm = 0.0f64;
    for p in pdf_out.iter_mut() {
        *p = (*p * inv_sum).max(floor);
        norm += *p;
    }
    let inv_norm = if norm.is_finite() && norm > 0.0 {
        1.0 / norm
    } else {
        1.0 / (logits.len() as f64)
    };
    for p in pdf_out.iter_mut() {
        *p *= inv_norm;
    }
}

fn parse_u64(v: &str, key: &str) -> Result<u64> {
    v.parse::<u64>()
        .with_context(|| format!("invalid integer value for '{key}': {v}"))
}

fn parse_usize(v: &str, key: &str) -> Result<usize> {
    v.parse::<usize>()
        .with_context(|| format!("invalid integer value for '{key}': {v}"))
}

fn parse_f32(v: &str, key: &str) -> Result<f32> {
    v.parse::<f32>()
        .with_context(|| format!("invalid float value for '{key}': {v}"))
}

fn parse_train_mode_token(v: &str) -> Result<OnlineTrainMode> {
    let code = v.trim().to_ascii_lowercase();
    match code.as_str() {
        "0" | "none" | "off" => Ok(OnlineTrainMode::None),
        "1" | "sgd" => Ok(OnlineTrainMode::Sgd),
        "2" | "adam" => Ok(OnlineTrainMode::Adam),
        other => bail!("unknown train mode '{other}'"),
    }
}

fn parse_cfg_positional(csv: &str) -> Result<OnlineConfig> {
    let vals: Vec<&str> = csv
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if vals.len() != 6 && vals.len() != 7 {
        bail!(
            "positional cfg format expects 6 or 7 values: hidden,intermediate,layers,train,seed,lr[,stride]"
        );
    }

    let cfg = OnlineConfig {
        hidden: parse_usize(vals[0], "hidden")?,
        intermediate: parse_usize(vals[1], "intermediate")?,
        layers: parse_usize(vals[2], "layers")?,
        train_mode: parse_train_mode_token(vals[3])?,
        seed: parse_u64(vals[4], "seed")?,
        lr: parse_f32(vals[5], "lr")?,
        stride: if vals.len() == 7 {
            parse_usize(vals[6], "stride")?
        } else {
            1
        },
        ..OnlineConfig::default()
    };
    Ok(cfg)
}

pub fn parse_method_spec(method: &str) -> Result<MethodSpec> {
    let trimmed = method.trim();
    if trimmed.is_empty() {
        bail!("empty rwkv method");
    }

    if let Some(path) = trimmed.strip_prefix("file:") {
        let p = PathBuf::from(path.trim());
        if p.as_os_str().is_empty() {
            bail!("empty file path in rwkv method");
        }
        return Ok(MethodSpec::File(p));
    }

    if let Some(cfg_s) = trimmed.strip_prefix("cfg:") {
        if !cfg_s.contains('=') {
            return Ok(MethodSpec::Online(parse_cfg_positional(cfg_s)?));
        }
        let mut cfg = OnlineConfig::default();
        for pair in cfg_s.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let (k, v) = pair
                .split_once('=')
                .with_context(|| format!("invalid cfg key/value pair '{pair}'"))?;
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim();
            match key.as_str() {
                "hidden" => cfg.hidden = parse_usize(val, "hidden")?,
                "layers" => cfg.layers = parse_usize(val, "layers")?,
                "intermediate" => cfg.intermediate = parse_usize(val, "intermediate")?,
                "decay_rank" => cfg.decay_rank = parse_usize(val, "decay_rank")?,
                "a_rank" => cfg.a_rank = parse_usize(val, "a_rank")?,
                "v_rank" => cfg.v_rank = parse_usize(val, "v_rank")?,
                "g_rank" => cfg.g_rank = parse_usize(val, "g_rank")?,
                "seed" => cfg.seed = parse_u64(val, "seed")?,
                "lr" => cfg.lr = parse_f32(val, "lr")?,
                "stride" => cfg.stride = parse_usize(val, "stride")?,
                "train" | "train_mode" => cfg.train_mode = parse_train_mode_token(val)?,
                other => bail!("unknown rwkv cfg key '{other}'"),
            }
        }
        return Ok(MethodSpec::Online(cfg));
    }

    let plain = PathBuf::from(trimmed);
    if plain.exists() {
        return Ok(MethodSpec::File(plain));
    }

    if trimmed.contains(',') {
        return Ok(MethodSpec::Online(parse_cfg_positional(trimmed)?));
    }

    bail!(
        "rwkv method must be 'file:<path>', 'cfg:<k=v,...>', positional cfg CSV, or an existing model path"
    );
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
    online: Option<OnlineRuntime>,
    source_model_path: Option<PathBuf>,
}

impl Clone for Compressor {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_from_model(self.model.clone());
        cloned.state = self.state.clone();
        cloned.pdf_buffer.clone_from(&self.pdf_buffer);
        cloned.cdf_buffer_ac.clone_from(&self.cdf_buffer_ac);
        cloned.cdf_buffer_rans.clone_from(&self.cdf_buffer_rans);
        cloned.rans_freq_buffer.clone_from(&self.rans_freq_buffer);
        cloned.online = self.online.clone();
        cloned.source_model_path = self.source_model_path.clone();
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
        let model_path = model_path.as_ref();
        let model = Arc::new(Model::load(model_path)?);
        let mut c = Self::new_from_model(model);
        c.source_model_path = Some(model_path.to_path_buf());
        c.maybe_load_sidecar()?;
        Ok(c)
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
            online: None,
            source_model_path: None,
        }
    }

    pub fn new_from_method(method: &str) -> Result<Self> {
        match parse_method_spec(method)? {
            MethodSpec::File(path) => Self::new(path),
            MethodSpec::Online(cfg) => {
                let rwcfg = cfg.to_rwkv_config()?;
                let model = Arc::new(Model::new_random(rwcfg, cfg.seed)?);
                let mut c = Self::new_from_model(model);
                c.online = Some(OnlineRuntime::new(cfg, VOCAB_SIZE));
                Ok(c)
            }
        }
    }

    /// Reset the model state to initial values.
    ///
    /// Call this between independent compression/decompression operations
    /// to ensure a clean state.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn reset_and_prime(&mut self) {
        self.state.reset();
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
    }

    pub fn snapshot_runtime(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            state: self.state.clone(),
            pdf_buffer: self.pdf_buffer.clone(),
            online: self.online.clone(),
        }
    }

    pub fn restore_runtime(&mut self, snapshot: &RuntimeSnapshot) {
        self.state = snapshot.state.clone();
        self.pdf_buffer.clone_from(&snapshot.pdf_buffer);
        self.online = snapshot.online.clone();
    }

    pub fn absorb_chain(&mut self, parts: &[&[u8]]) -> Result<()> {
        for part in parts {
            for &byte in *part {
                self.online_update_from_current_pdf(byte)?;
                let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
                let logits = self
                    .model
                    .forward(&mut self.scratch, byte as u32, &mut self.state);
                Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
            }
        }
        Ok(())
    }

    pub fn cross_entropy_from_current(&mut self, data: &[u8]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let mut total_bits = 0.0f64;
        for &byte in data {
            let p = self.pdf_buffer[byte as usize];
            total_bits -= p.log2();
            self.online_update_from_current_pdf(byte)?;
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
        }
        Ok(total_bits / (data.len() as f64))
    }

    pub fn is_online(&self) -> bool {
        self.online.is_some()
    }

    pub fn tokens_processed(&self) -> u64 {
        self.online.as_ref().map_or(0, |s| s.tokens_processed)
    }

    pub fn online_method_string(&self) -> Option<&str> {
        self.online.as_ref().map(|s| s.canonical_cfg.as_str())
    }

    /// Get the vocabulary size (should always be 256 for byte-level).
    pub fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    pub fn online_apply_logits_bias(&self, logits: &[f32], pdf_out: &mut [f64]) {
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        Self::logits_to_pdf(logits, bias, pdf_out);
    }

    pub fn logits_to_pdf(logits: &[f32], bias: Option<&[f32]>, pdf_out: &mut [f64]) {
        softmax_pdf_floor_with_bias(logits, bias, pdf_out);
    }

    pub fn online_bias_snapshot(&self) -> Option<Vec<f32>> {
        self.online.as_ref().map(|o| o.out_bias.clone())
    }

    pub fn online_update_from_pdf(&mut self, symbol: u8, pdf: &[f64]) -> Result<()> {
        let Some(online) = self.online.as_mut() else {
            return Ok(());
        };
        online.tokens_processed = online.tokens_processed.saturating_add(1);

        if matches!(online.cfg.train_mode, OnlineTrainMode::None) {
            return Ok(());
        }

        let stride = online.cfg.stride.max(1) as u64;
        if stride > 1 && (online.tokens_processed % stride) != 0 {
            return Ok(());
        }

        let lr = online.cfg.lr.max(0.0);
        if lr == 0.0 {
            return Ok(());
        }

        let n = online.out_bias.len().min(pdf.len());
        match online.cfg.train_mode {
            OnlineTrainMode::None => {}
            OnlineTrainMode::Sgd => {
                for (i, p_raw) in pdf.iter().enumerate().take(n) {
                    let p = (*p_raw).clamp(1e-12, 1.0) as f32;
                    let target = if i == symbol as usize { 1.0 } else { 0.0 };
                    let grad = target - p;
                    online.out_bias[i] += lr * grad;
                }
            }
            OnlineTrainMode::Adam => {
                online.adam_t = online.adam_t.saturating_add(1);
                let t = online.adam_t as i32;
                let b1 = 0.9f32;
                let b2 = 0.999f32;
                let eps = 1e-8f32;
                if let (Some(m), Some(v)) = (online.adam_m.as_mut(), online.adam_v.as_mut()) {
                    for i in 0..n {
                        let p = pdf[i].clamp(1e-12, 1.0) as f32;
                        let target = if i == symbol as usize { 1.0 } else { 0.0 };
                        let grad = target - p;
                        m[i] = b1 * m[i] + (1.0 - b1) * grad;
                        v[i] = b2 * v[i] + (1.0 - b2) * grad * grad;
                        let m_hat = m[i] / (1.0 - b1.powi(t));
                        let v_hat = v[i] / (1.0 - b2.powi(t));
                        online.out_bias[i] += lr * m_hat / (v_hat.sqrt() + eps);
                    }
                }
            }
        }

        Ok(())
    }

    fn online_update_from_current_pdf(&mut self, symbol: u8) -> Result<()> {
        if self.online.is_none() {
            return Ok(());
        }
        let pdf = self.pdf_buffer.clone();
        self.online_update_from_pdf(symbol, &pdf)
    }

    pub fn export_online<P: AsRef<Path>>(&self, model_path: P) -> Result<()> {
        let model_path = model_path.as_ref();
        self.model.save_safetensors(model_path)?;

        let sidecar = model_path.with_extension("json");
        let meta = if let Some(online) = &self.online {
            let train_mode = match online.cfg.train_mode {
                OnlineTrainMode::None => "none",
                OnlineTrainMode::Sgd => "sgd",
                OnlineTrainMode::Adam => "adam",
            };
            json!({
                "version": 1,
                "method": online.canonical_cfg,
                "training_mode": train_mode,
                "tokens_processed": online.tokens_processed,
                "config": {
                    "hidden": online.cfg.hidden,
                    "layers": online.cfg.layers,
                    "intermediate": online.cfg.intermediate,
                    "decay_rank": online.cfg.decay_rank,
                    "a_rank": online.cfg.a_rank,
                    "v_rank": online.cfg.v_rank,
                    "g_rank": online.cfg.g_rank,
                    "seed": online.cfg.seed,
                    "lr": online.cfg.lr,
                    "stride": online.cfg.stride.max(1),
                },
                "output_bias": online.out_bias,
            })
        } else {
            json!({
                "version": 1,
                "method": format!("file:{}", model_path.display()),
                "training_mode": "none",
                "tokens_processed": 0,
            })
        };

        fs::write(&sidecar, serde_json::to_vec_pretty(&meta)?)?;
        Ok(())
    }

    fn maybe_load_sidecar(&mut self) -> Result<()> {
        let Some(model_path) = &self.source_model_path else {
            return Ok(());
        };
        let sidecar = model_path.with_extension("json");
        if !sidecar.exists() {
            return Ok(());
        }
        let raw = fs::read(&sidecar)?;
        let v: serde_json::Value = serde_json::from_slice(&raw)?;
        let output_bias = v
            .get("output_bias")
            .and_then(|arr| arr.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect::<Vec<f32>>()
            });
        let method = v
            .get("method")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("file:{}", model_path.display()));
        let tokens = v
            .get("tokens_processed")
            .and_then(|t| t.as_u64())
            .unwrap_or(0);
        if let Some(mut out_bias) = output_bias {
            out_bias.resize(self.vocab_size(), 0.0);
            let mut cfg = OnlineConfig::default();
            if let Some(cfg_v) = v.get("config").and_then(|x| x.as_object()) {
                if let Some(x) = cfg_v.get("hidden").and_then(|x| x.as_u64()) {
                    cfg.hidden = x as usize;
                }
                if let Some(x) = cfg_v.get("layers").and_then(|x| x.as_u64()) {
                    cfg.layers = x as usize;
                }
                if let Some(x) = cfg_v.get("intermediate").and_then(|x| x.as_u64()) {
                    cfg.intermediate = x as usize;
                }
                if let Some(x) = cfg_v.get("decay_rank").and_then(|x| x.as_u64()) {
                    cfg.decay_rank = x as usize;
                }
                if let Some(x) = cfg_v.get("a_rank").and_then(|x| x.as_u64()) {
                    cfg.a_rank = x as usize;
                }
                if let Some(x) = cfg_v.get("v_rank").and_then(|x| x.as_u64()) {
                    cfg.v_rank = x as usize;
                }
                if let Some(x) = cfg_v.get("g_rank").and_then(|x| x.as_u64()) {
                    cfg.g_rank = x as usize;
                }
                if let Some(x) = cfg_v.get("seed").and_then(|x| x.as_u64()) {
                    cfg.seed = x;
                }
                if let Some(x) = cfg_v.get("lr").and_then(|x| x.as_f64()) {
                    cfg.lr = x as f32;
                }
                if let Some(x) = cfg_v.get("stride").and_then(|x| x.as_u64()) {
                    cfg.stride = (x as usize).max(1);
                }
            }
            cfg.train_mode = v
                .get("training_mode")
                .and_then(|x| x.as_str())
                .and_then(|s| parse_train_mode_token(s).ok())
                .unwrap_or(OnlineTrainMode::None);
            self.online = Some(OnlineRuntime {
                cfg,
                canonical_cfg: method,
                tokens_processed: tokens,
                out_bias,
                adam_m: None,
                adam_v: None,
                adam_t: 0,
            });
        }
        Ok(())
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

        // Prime the model with a null byte to establish initial state
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        for byte in data {
            quantize_pdf_to_cdf_inplace(&self.pdf_buffer, &mut self.cdf_buffer_ac);
            let sym = byte as usize;
            let c_lo = self.cdf_buffer_ac[sym] as u64;
            let c_hi = self.cdf_buffer_ac[sym + 1] as u64;
            encoder.encode_counts(c_lo, c_hi, CDF_TOTAL as u64)?;
            self.online_update_from_current_pdf(byte)?;

            // Update model state with actual byte for next prediction
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
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
        // Use blocked encoder (128KB blocks) for streaming large files
        let mut encoder = BlockedRansEncoder::new();

        // Prime the model with a null byte
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

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
            self.online_update_from_current_pdf(byte)?;

            // Update model state
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
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

        let mut result = Vec::with_capacity(original_len);

        // Prime with null byte (must match compression)
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        for _ in 0..original_len {
            quantize_pdf_to_cdf_inplace(&self.pdf_buffer, &mut self.cdf_buffer_ac);
            let sym = decoder.decode_symbol_counts(&self.cdf_buffer_ac, CDF_TOTAL)?;
            result.push(sym as u8);
            self.online_update_from_current_pdf(sym as u8)?;

            // Update model state with decoded byte
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, sym as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
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
        let mut result = Vec::with_capacity(original_len);

        // Prime with null byte
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        for _ in 0..original_len {
            quantize_pdf_to_rans_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_rans,
                &mut self.rans_freq_buffer,
            );
            let sym = decoder.decode(&self.cdf_buffer_rans)?;
            result.push(sym as u8);
            self.online_update_from_current_pdf(sym as u8)?;

            // Update model state
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, sym as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
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
        self.reset_and_prime();
        self.cross_entropy_from_current(data)
    }

    pub fn cross_entropy_conditional_chain(
        &mut self,
        prefix_parts: &[&[u8]],
        data: &[u8],
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        self.reset_and_prime();
        self.absorb_chain(prefix_parts)?;
        self.cross_entropy_from_current(data)
    }

    pub fn cross_entropy_conditional(&mut self, prefix: &[u8], data: &[u8]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        self.state.reset();

        // Prime with null byte
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        // Condition on prefix (update state, no scoring)
        for &byte in prefix {
            self.online_update_from_current_pdf(byte)?;
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
        }

        let mut total_bits = 0.0f64;
        for &byte in data {
            let p = self.pdf_buffer[byte as usize];
            total_bits -= p.log2();
            self.online_update_from_current_pdf(byte)?;
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
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

        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        let mut total_bits = 0.0f64;
        for i in 0..n {
            let a = if swap { y[i] } else { x[i] };
            let b = if swap { x[i] } else { y[i] };

            let pa = self.pdf_buffer[a as usize];
            total_bits -= pa.log2();
            self.online_update_from_current_pdf(a)?;
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, a as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

            let pb = self.pdf_buffer[b as usize];
            total_bits -= pb.log2();
            self.online_update_from_current_pdf(b)?;
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, b as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str, ext: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("infotheory_rwkvzip_{name}_{ts}.{ext}"))
    }

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

    #[test]
    fn test_parse_method_spec_file_and_cfg() {
        let p = temp_path("dummy", "bin");
        std::fs::write(&p, b"x").unwrap();

        match parse_method_spec(&format!("file:{}", p.display())).unwrap() {
            MethodSpec::File(got) => assert_eq!(got, p),
            _ => panic!("expected file method"),
        }

        match parse_method_spec(
            "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=1,train=none,lr=0.01,stride=2",
        )
        .unwrap()
        {
            MethodSpec::Online(cfg) => {
                assert_eq!(cfg.hidden, 64);
                assert_eq!(cfg.layers, 1);
                assert_eq!(cfg.seed, 1);
                assert_eq!(cfg.stride, 2);
            }
            _ => panic!("expected cfg method"),
        }

        match parse_method_spec("64,64,1,0,7,0.01,2").unwrap() {
            MethodSpec::Online(cfg) => {
                assert_eq!(cfg.hidden, 64);
                assert_eq!(cfg.intermediate, 64);
                assert_eq!(cfg.layers, 1);
                assert_eq!(cfg.seed, 7);
                assert_eq!(cfg.stride, 2);
            }
            _ => panic!("expected positional cfg method"),
        }

        // Backward-compatible plain existing path.
        match parse_method_spec(&p.display().to_string()).unwrap() {
            MethodSpec::File(got) => assert_eq!(got, p),
            _ => panic!("expected file method"),
        }

        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn test_parse_method_spec_rejects_unknown_cfg_key() {
        let err = parse_method_spec("cfg:hidden=64,wat=1").unwrap_err();
        assert!(format!("{err:#}").contains("unknown rwkv cfg key"));
    }

    #[test]
    fn test_online_export_reload_roundtrip() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=7,train=sgd,lr=0.01,stride=1";
        let data = b"rwkv online export/load deterministic sample";

        let mut c1 = Compressor::new_from_method(method).unwrap();
        let _ = c1.compress(data, CoderType::AC).unwrap();

        let model_path = temp_path("export", "safetensors");
        c1.export_online(&model_path).unwrap();
        let out1_after_export = c1.compress(data, CoderType::AC).unwrap();

        let mut c2 = Compressor::new(&model_path).unwrap();
        let out2 = c2.compress(data, CoderType::AC).unwrap();

        assert_eq!(out1_after_export, out2);
        assert!(model_path.with_extension("json").exists());

        std::fs::remove_file(&model_path).ok();
        std::fs::remove_file(model_path.with_extension("json")).ok();
    }

    #[test]
    fn test_runtime_snapshot_restores_online_state() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=9,train=sgd,lr=0.01,stride=1";
        let mut c = Compressor::new_from_method(method).unwrap();
        c.reset_and_prime();
        c.absorb_chain(&[b"prior context".as_slice()]).unwrap();
        let snap = c.snapshot_runtime();

        c.absorb_chain(&[b"snippet-a".as_slice()]).unwrap();
        let score_a = c.cross_entropy_from_current(b"query").unwrap();

        c.restore_runtime(&snap);
        c.absorb_chain(&[b"snippet-b".as_slice()]).unwrap();
        let score_b = c.cross_entropy_from_current(b"query").unwrap();

        c.restore_runtime(&snap);
        c.absorb_chain(&[b"snippet-b".as_slice()]).unwrap();
        let score_b_again = c.cross_entropy_from_current(b"query").unwrap();

        assert!((score_b - score_b_again).abs() < 1e-12);
        let _ = score_a;
    }
}
