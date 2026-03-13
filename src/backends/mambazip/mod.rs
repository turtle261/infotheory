#![allow(clippy::items_after_test_module)]

// mambazip - deterministic CPU-first Mamba-1 compressor/runtime.

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::backends::llm_policy::{
    self, LlmPolicy, OptimizerKind, PolicyAction, PolicyRuntime, split_method_policy_segments,
};
/// Mamba-1 model internals.
pub mod mamba1;

/// Shared entropy coders.
pub use crate::coders;
/// Backward-compatible coder re-export.
pub use crate::coders::CoderType;

use crate::coders::{
    ANS_TOTAL, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder, BlockedRansEncoder,
    CDF_TOTAL, Cdf, quantize_pdf_to_cdf_with_buffer, quantize_pdf_to_rans_cdf_with_buffer,
};

/// Mamba model config.
pub use mamba1::Config;
/// Mamba model.
pub use mamba1::Model;
/// Mamba reusable scratch buffers.
pub use mamba1::ScratchBuffers;
/// Mamba recurrent state.
pub use mamba1::State;

/// File format magic for mambazip payloads.
pub const MAGIC: u32 = 0x5a424d4d; // "MMBZ"
/// File format version.
pub const VERSION: u8 = 1;
/// Byte vocabulary size.
pub const VOCAB_SIZE: usize = 256;
const MAMBA_TRAIN_SCOPES: &[&str] = &[
    "embed",
    "layer_norm",
    "mixer_conv",
    "mixer_ssm",
    "mixer_proj",
    "head",
    "bias",
    "all",
    "none",
];

#[inline]
fn optimizer_sidecar_path(model_path: &Path) -> PathBuf {
    model_path.with_extension("opt.safetensors")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Online adaptation mode for Mamba output-bias updates.
pub enum OnlineTrainMode {
    /// Disable updates.
    None,
    /// SGD updates.
    Sgd,
    /// Adam updates.
    Adam,
}

#[derive(Clone, Debug)]
/// Online/runtime model construction config.
pub struct OnlineConfig {
    /// Hidden width.
    pub hidden: usize,
    /// Number of layers.
    pub layers: usize,
    /// Mamba inner width (`d_inner`).
    pub intermediate: usize,
    /// Mamba state width (`d_state`).
    pub state: usize,
    /// Mamba depthwise conv width (`d_conv`).
    pub conv: usize,
    /// Delta rank (`dt_rank`).
    pub dt_rank: usize,
    /// RNG seed for random init.
    pub seed: u64,
    /// Online training mode.
    pub train_mode: OnlineTrainMode,
    /// Learning rate.
    pub lr: f32,
    /// Update stride.
    pub stride: usize,
}

impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            hidden: 256,
            layers: 6,
            intermediate: 512,
            state: 16,
            conv: 4,
            dt_rank: 16,
            seed: 0,
            train_mode: OnlineTrainMode::None,
            lr: 0.001,
            stride: 1,
        }
    }
}

impl OnlineConfig {
    /// Convert to validated model config.
    pub fn to_mamba_config(&self) -> Result<Config> {
        let cfg = Config {
            vocab_size: VOCAB_SIZE,
            hidden_size: self.hidden.max(16),
            num_layers: self.layers.max(1),
            inner_size: self.intermediate.max(16),
            state_size: self.state.max(1),
            conv_kernel: self.conv.max(1),
            dt_rank: self.dt_rank.max(1),
            layer_norm_eps: 1e-5,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

#[derive(Clone, Debug)]
/// Parsed method specification.
pub enum MethodSpec {
    /// Load model from filesystem.
    File {
        path: PathBuf,
        policy: Option<LlmPolicy>,
    },
    /// Construct random model + online adaptation config.
    Online {
        cfg: OnlineConfig,
        policy: Option<LlmPolicy>,
    },
}

#[derive(Clone)]
struct OnlineRuntime {
    cfg: OnlineConfig,
    canonical_method: String,
    policy: Option<LlmPolicy>,
    policy_runtime: Option<PolicyRuntime>,
    needs_full_trace: bool,
    policy_stream_total: Option<u64>,
    policy_train_steps: u64,
    tokens_processed: u64,
    out_bias: Vec<f32>,
    adam_m: Option<Vec<f32>>,
    adam_v: Option<Vec<f32>>,
    full_adam: Option<mamba1::FullAdamState>,
    lm_head_adam_m: Option<Vec<f32>>,
    lm_head_adam_v: Option<Vec<f32>>,
    adam_t: usize,
}

#[derive(Clone)]
/// Snapshot of mutable runtime state.
pub struct RuntimeSnapshot {
    model: Arc<Model>,
    scratch: ScratchBuffers,
    state: State,
    pdf_buffer: Vec<f64>,
    online: Option<OnlineRuntime>,
}

impl OnlineRuntime {
    fn new(
        cfg: OnlineConfig,
        canonical_method: String,
        policy: Option<LlmPolicy>,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        let mut use_adam = matches!(cfg.train_mode, OnlineTrainMode::Adam);
        if let Some(pol) = &policy {
            use_adam = policy_uses_adam(pol) || use_adam;
        }
        let needs_full_trace = policy
            .as_ref()
            .map(policy_needs_full_trace)
            .unwrap_or(false);
        Self {
            canonical_method,
            cfg,
            policy,
            policy_runtime: None,
            needs_full_trace,
            policy_stream_total: None,
            policy_train_steps: 0,
            tokens_processed: 0,
            out_bias: vec![0.0; vocab_size],
            adam_m: use_adam.then(|| vec![0.0; vocab_size]),
            adam_v: use_adam.then(|| vec![0.0; vocab_size]),
            full_adam: None,
            lm_head_adam_m: use_adam.then(|| vec![0.0; vocab_size * hidden_size]),
            lm_head_adam_v: use_adam.then(|| vec![0.0; vocab_size * hidden_size]),
            adam_t: 0,
        }
    }

    fn prepare_policy_stream(&mut self, total_symbols: Option<u64>) -> Result<()> {
        self.policy_stream_total = total_symbols;
        self.policy_train_steps = 0;
        self.policy_runtime = match &self.policy {
            Some(p) => Some(PolicyRuntime::new(p.compile(total_symbols)?)),
            None => None,
        };
        Ok(())
    }

    #[inline]
    fn next_policy_action(&mut self) -> Result<Option<PolicyAction>> {
        if self.policy.is_none() {
            return Ok(None);
        }
        if self.policy_runtime.is_none() {
            self.prepare_policy_stream(None)?;
        }
        Ok(self.policy_runtime.as_mut().map(PolicyRuntime::next_action))
    }
}

fn policy_uses_adam(policy: &LlmPolicy) -> bool {
    use llm_policy::ScheduleRule;
    for rule in &policy.schedule {
        match rule {
            ScheduleRule::Interval(interval) => {
                if let PolicyAction::Train(train) = &interval.action
                    && matches!(train.optimizer, OptimizerKind::Adam)
                {
                    return true;
                }
            }
            ScheduleRule::Repeat(repeat) => {
                for seg in &repeat.pattern {
                    if let PolicyAction::Train(train) = &seg.action
                        && matches!(train.optimizer, OptimizerKind::Adam)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn scope_needs_full_trace(scope: &llm_policy::TrainScopeSet) -> bool {
    scope.all
        || scope.contains("embed")
        || scope.contains("layer_norm")
        || scope.contains("mixer_conv")
        || scope.contains("mixer_ssm")
        || scope.contains("mixer_proj")
}

fn policy_needs_full_trace(policy: &LlmPolicy) -> bool {
    use llm_policy::ScheduleRule;
    for rule in &policy.schedule {
        match rule {
            ScheduleRule::Interval(interval) => {
                if let PolicyAction::Train(train) = &interval.action
                    && scope_needs_full_trace(&train.scope)
                {
                    return true;
                }
            }
            ScheduleRule::Repeat(repeat) => {
                for seg in &repeat.pattern {
                    if let PolicyAction::Train(train) = &seg.action
                        && scope_needs_full_trace(&train.scope)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn cfg_to_method_string(cfg: &OnlineConfig) -> String {
    let train = match cfg.train_mode {
        OnlineTrainMode::None => "none",
        OnlineTrainMode::Sgd => "sgd",
        OnlineTrainMode::Adam => "adam",
    };
    format!(
        "cfg:hidden={},layers={},intermediate={},state={},conv={},dt_rank={},seed={},train={},lr={},stride={}",
        cfg.hidden,
        cfg.layers,
        cfg.intermediate,
        cfg.state,
        cfg.conv,
        cfg.dt_rank,
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
    if let Some(b) = bias {
        for i in 0..logits.len() {
            let z = logits[i] + b[i];
            if z > max_logit {
                max_logit = z;
            }
        }
    } else {
        for &z in logits {
            if z > max_logit {
                max_logit = z;
            }
        }
    }

    let mut sum = 0.0f64;
    if let Some(b) = bias {
        for i in 0..logits.len() {
            let p = ((logits[i] + b[i] - max_logit) as f64).exp();
            pdf_out[i] = p;
            sum += p;
        }
    } else {
        for i in 0..logits.len() {
            let p = ((logits[i] - max_logit) as f64).exp();
            pdf_out[i] = p;
            sum += p;
        }
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

    Ok(OnlineConfig {
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
    })
}

/// Parse a method string.
///
/// Supported formats:
/// - `file:/path/to/model.safetensors`
/// - `file:/path/to/model.safetensors;policy:...`
/// - `cfg:key=value,...[;policy:...]`
/// - positional `cfg` CSV
/// - existing model path
pub fn parse_method_spec(method: &str) -> Result<MethodSpec> {
    let (base, policy_segment) = split_method_policy_segments(method)?;
    let parse_policy = |s: &str| llm_policy::parse_policy_segment(s, MAMBA_TRAIN_SCOPES);
    let policy = policy_segment
        .as_deref()
        .map(parse_policy)
        .transpose()
        .context("failed to parse mamba policy segment")?;

    if let Some(path) = base.strip_prefix("file:") {
        let p = PathBuf::from(path.trim());
        if p.as_os_str().is_empty() {
            bail!("empty file path in mamba method");
        }
        if policy.as_ref().and_then(|p| p.load_from.as_ref()).is_some() {
            bail!("mamba method cannot use policy load_from together with file:<path>");
        }
        return Ok(MethodSpec::File { path: p, policy });
    }

    if let Some(cfg_s) = base.strip_prefix("cfg:") {
        if !cfg_s.contains('=') {
            return Ok(MethodSpec::Online {
                cfg: parse_cfg_positional(cfg_s)?,
                policy,
            });
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
                "state" | "d_state" => cfg.state = parse_usize(val, "state")?,
                "conv" | "d_conv" => cfg.conv = parse_usize(val, "conv")?,
                "dt_rank" => cfg.dt_rank = parse_usize(val, "dt_rank")?,
                "seed" => cfg.seed = parse_u64(val, "seed")?,
                "lr" => cfg.lr = parse_f32(val, "lr")?,
                "stride" => cfg.stride = parse_usize(val, "stride")?,
                "train" | "train_mode" => cfg.train_mode = parse_train_mode_token(val)?,
                other => bail!("unknown mamba cfg key '{other}'"),
            }
        }
        return Ok(MethodSpec::Online { cfg, policy });
    }

    let plain = PathBuf::from(base.trim());
    if plain.exists() {
        if policy.as_ref().and_then(|p| p.load_from.as_ref()).is_some() {
            bail!("mamba method cannot use policy load_from together with file path");
        }
        return Ok(MethodSpec::File {
            path: plain,
            policy,
        });
    }

    if base.contains(',') {
        return Ok(MethodSpec::Online {
            cfg: parse_cfg_positional(&base)?,
            policy,
        });
    }

    bail!(
        "mamba method must be 'file:<path>', 'cfg:<k=v,...>', positional cfg CSV, or an existing model path"
    );
}

/// Framing header for mambazip streams.
#[derive(Debug, Clone)]
pub struct Header {
    /// Magic number.
    pub magic: u32,
    /// Version byte.
    pub version: u8,
    /// Coder type (0=AC,1=rANS).
    pub coder: u8,
    /// Original length.
    pub original_len: u64,
    /// CRC32 checksum of original data.
    pub crc32: u32,
}

impl Header {
    /// Header serialized size.
    pub const SIZE: usize = 4 + 1 + 1 + 8 + 4;

    /// Construct a new header.
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

    /// Write header.
    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.magic.to_le_bytes())?;
        w.write_all(&[self.version])?;
        w.write_all(&[self.coder])?;
        w.write_all(&self.original_len.to_le_bytes())?;
        w.write_all(&self.crc32.to_le_bytes())?;
        Ok(())
    }

    /// Read header.
    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        let mut buf1 = [0u8; 1];

        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != MAGIC {
            bail!(
                "invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                MAGIC,
                magic
            );
        }

        r.read_exact(&mut buf1)?;
        let version = buf1[0];
        if version > VERSION {
            bail!(
                "unsupported version: {} (max supported: {})",
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

    /// Decode coder selection.
    pub fn coder_type(&self) -> CoderType {
        match self.coder {
            0 => CoderType::AC,
            _ => CoderType::RANS,
        }
    }
}

/// Compute CRC32 of data.
pub fn crc32(data: &[u8]) -> u32 {
    crate::coders::crc32(data)
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

/// Stateful compressor built around a Mamba model.
pub struct Compressor {
    /// Model weights.
    pub model: Arc<Model>,
    /// Recurrent state.
    pub state: State,
    /// Reusable scratch.
    pub scratch: ScratchBuffers,
    /// Reusable PDF buffer.
    pub pdf_buffer: Vec<f64>,
    cdf_buffer_ac: Vec<u32>,
    ac_freq_buffer: Vec<i64>,
    cdf_buffer_rans: Vec<u32>,
    rans_freq_buffer: Vec<i64>,
    online: Option<OnlineRuntime>,
    source_model_path: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_method_spec_accepts_cfg_and_positional() {
        let named = parse_method_spec(
            "cfg:hidden=64,layers=2,intermediate=96,state=8,conv=3,dt_rank=4,train=sgd,lr=0.01,stride=2;policy:schedule=0..100:infer",
        )
        .expect("named cfg");
        match named {
            MethodSpec::Online { cfg, .. } => {
                assert_eq!(cfg.hidden, 64);
                assert_eq!(cfg.layers, 2);
                assert_eq!(cfg.intermediate, 96);
                assert_eq!(cfg.state, 8);
                assert_eq!(cfg.conv, 3);
                assert_eq!(cfg.dt_rank, 4);
                assert!(matches!(cfg.train_mode, OnlineTrainMode::Sgd));
                assert_eq!(cfg.stride, 2);
            }
            _ => panic!("expected online cfg"),
        }

        let positional =
            parse_method_spec("cfg:64,96,2,adam,123,0.001,3;policy:schedule=0..100:infer")
                .expect("positional cfg");
        match positional {
            MethodSpec::Online { cfg, .. } => {
                assert_eq!(cfg.hidden, 64);
                assert_eq!(cfg.intermediate, 96);
                assert_eq!(cfg.layers, 2);
                assert!(matches!(cfg.train_mode, OnlineTrainMode::Adam));
                assert_eq!(cfg.seed, 123);
                assert_eq!(cfg.stride, 3);
            }
            _ => panic!("expected online cfg"),
        }
    }

    #[test]
    fn parse_method_spec_accepts_cfg_without_policy() {
        let spec = parse_method_spec("cfg:hidden=64,layers=2,intermediate=96").expect("cfg");
        match spec {
            MethodSpec::Online { cfg, policy } => {
                assert_eq!(cfg.hidden, 64);
                assert_eq!(cfg.layers, 2);
                assert_eq!(cfg.intermediate, 96);
                assert!(policy.is_none());
            }
            _ => panic!("expected online cfg"),
        }
    }

    #[test]
    fn canonical_method_omits_policy_when_absent() {
        let c = Compressor::new_from_method("cfg:hidden=64,layers=1,intermediate=96")
            .expect("online model");
        assert_eq!(
            c.online_method_string(),
            Some(
                "cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=0,train=none,lr=0.001,stride=1"
            )
        );
    }

    #[test]
    fn export_reload_roundtrip_reproducible() {
        let cfg = Config {
            vocab_size: 256,
            hidden_size: 32,
            num_layers: 2,
            inner_size: 48,
            state_size: 8,
            conv_kernel: 3,
            dt_rank: 4,
            layer_norm_eps: 1e-5,
        };
        let model = Arc::new(Model::new_random(cfg.clone(), 42).expect("random model"));
        let mut c1 = Compressor::new_from_model(model);
        c1.reset_and_prime();
        let _ = c1.cross_entropy_from_current(b"mamba test").expect("score");

        let base = std::env::temp_dir().join(format!(
            "infotheory_mamba_rt_{}_{}.safetensors",
            std::process::id(),
            c1.tokens_processed()
        ));
        c1.export_online(&base).expect("export");

        let mut c2 = Compressor::new(&base).expect("reload");
        c2.reset_and_prime();
        let h1 = c1.cross_entropy(b"abcabc").expect("h1");
        let h2 = c2.cross_entropy(b"abcabc").expect("h2");
        assert!((h1 - h2).abs() < 1e-9);

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(base.with_extension("json"));
    }

    #[test]
    fn online_training_updates_lm_head_weights() {
        let method = "cfg:hidden=64,layers=2,intermediate=96,state=8,conv=3,dt_rank=4,seed=11,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).expect("online model");
        c.reset_and_prime();
        let before = c.model.lm_head_weights()[0..64].to_vec();
        let _ = c
            .cross_entropy_from_current(b"online mamba weight update")
            .expect("score");
        let after = &c.model.lm_head_weights()[0..64];
        let mut changed = false;
        for i in 0..before.len() {
            if before[i].to_bits() != after[i].to_bits() {
                changed = true;
                break;
            }
        }
        assert!(
            changed,
            "expected LM-head weights to change under online training"
        );
    }

    #[test]
    fn online_training_scope_all_updates_non_head_params() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=7,train=adam,lr=0.002,stride=1;policy:schedule=0..100:train(scope=mixer_proj,opt=adam,lr=0.002,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).expect("online model");
        c.reset_and_prime();
        let before_head = c.model.lm_head_weights()[0..64].to_vec();
        let before_model = (*c.model).clone();
        let _ = c
            .cross_entropy_from_current(b"scope mixer_proj should train non-head mamba params")
            .expect("score");
        let after_head = &c.model.lm_head_weights()[0..64];
        let mut head_unchanged = true;
        for i in 0..before_head.len() {
            if before_head[i].to_bits() != after_head[i].to_bits() {
                head_unchanged = false;
                break;
            }
        }
        assert!(
            head_unchanged,
            "expected LM-head weights to remain unchanged under scope=mixer_proj"
        );

        // Compare logits from fresh state to detect non-head parameter movement.
        let mut s1 = before_model.new_state();
        let mut sc1 = ScratchBuffers::new(before_model.config());
        let mut s2 = c.model.new_state();
        let mut sc2 = ScratchBuffers::new(c.model.config());
        let logits_before = before_model.forward(&mut sc1, 0, &mut s1);
        let logits_after = c.model.forward(&mut sc2, 0, &mut s2);
        let mut changed = false;
        for idx in 0..logits_before.len().min(logits_after.len()) {
            if logits_before[idx].to_bits() != logits_after[idx].to_bits() {
                changed = true;
                break;
            }
        }
        assert!(
            changed,
            "expected non-head parameters to update under scope=mixer_proj"
        );
    }

    #[test]
    fn online_training_scope_all_bptt_gt_one_rejected() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=7,train=adam,lr=0.002,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.002,stride=1,bptt=2,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).expect("online model");
        c.reset_and_prime();
        let err = c.cross_entropy_from_current(b"abcd").unwrap_err();
        assert!(
            format!("{err:#}").contains("bptt=1"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn export_reload_roundtrip_preserves_full_adam_resume() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=17,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let data = b"mamba full adam export/reload deterministic continuation";
        let mut c1 = Compressor::new_from_method(method).expect("online model");
        let _ = c1.compress(data, CoderType::AC).expect("pre-train pass");

        let model_path = std::env::temp_dir().join(format!(
            "infotheory_mamba_full_adam_{}_{}.safetensors",
            std::process::id(),
            c1.tokens_processed()
        ));
        c1.export_online(&model_path).expect("export");
        assert!(model_path.with_extension("opt.safetensors").exists());

        let out1 = c1
            .compress(data, CoderType::AC)
            .expect("post-export compress");
        let mut c2 = Compressor::new(&model_path).expect("reload");
        let out2 = c2.compress(data, CoderType::AC).expect("reload compress");
        assert_eq!(out1, out2, "full-adam resume must be bit-identical");

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(model_path.with_extension("json"));
        let _ = std::fs::remove_file(model_path.with_extension("opt.safetensors"));
    }

    #[test]
    fn clone_keeps_full_training_trace_mode() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=18,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).expect("online model");
        let mut cloned = c.clone();
        cloned.reset_and_prime();
        let _ = cloned
            .cross_entropy_from_current(b"clone must preserve training-trace mode")
            .expect("full-training step should succeed after clone");
        c.reset_and_prime();
        let _ = c
            .cross_entropy_from_current(b"baseline run")
            .expect("baseline full-training step");
    }

    #[test]
    fn runtime_snapshot_restores_non_head_training_state() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=19,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).expect("online model");
        c.reset_and_prime();
        c.absorb_chain(&[b"prior context".as_slice()])
            .expect("prefix");
        let snap = c.snapshot_runtime();

        let _ = c
            .cross_entropy_from_current(b"mutate model before restore")
            .expect("mutation pass");

        c.restore_runtime(&snap);
        let score_a = c
            .cross_entropy_from_current(b"query after restore")
            .expect("score a");

        c.restore_runtime(&snap);
        let score_b = c
            .cross_entropy_from_current(b"query after restore")
            .expect("score b");

        assert!((score_a - score_b).abs() < 1e-12);
    }

    #[test]
    fn clone_preserves_non_head_training_trace() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=20,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).expect("online model");
        c.reset_and_prime();
        c.absorb_chain(&[b"clone trace prefix".as_slice()])
            .expect("prefix");

        let mut cloned = c.clone();
        let score = cloned
            .cross_entropy_from_current(b"clone trace query")
            .expect("cloned full-training step");
        assert!(score.is_finite());
    }
}

impl Clone for Compressor {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_from_model(self.model.clone());
        cloned.state = self.state.clone();
        cloned.pdf_buffer.clone_from(&self.pdf_buffer);
        cloned.cdf_buffer_ac.clone_from(&self.cdf_buffer_ac);
        cloned.ac_freq_buffer.clone_from(&self.ac_freq_buffer);
        cloned.cdf_buffer_rans.clone_from(&self.cdf_buffer_rans);
        cloned.rans_freq_buffer.clone_from(&self.rans_freq_buffer);
        cloned.scratch = self.scratch.clone();
        cloned.online = self.online.clone();
        cloned.source_model_path = self.source_model_path.clone();
        cloned
    }
}

impl Compressor {
    /// Create compressor by loading model path.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        let model = Arc::new(Model::load(model_path)?);
        let mut c = Self::new_from_model(model);
        c.source_model_path = Some(model_path.to_path_buf());
        c.maybe_load_sidecar()?;
        Ok(c)
    }

    /// Load model from path and wrap in Arc.
    pub fn load_model<P: AsRef<Path>>(model_path: P) -> Result<Arc<Model>> {
        Ok(Arc::new(Model::load(model_path)?))
    }

    /// Create compressor from preloaded model.
    pub fn new_from_model(model: Arc<Model>) -> Self {
        let state = model.new_state();
        let vocab_size = model.config().vocab_size;
        let scratch = ScratchBuffers::new(model.config());
        Self {
            model,
            state,
            scratch,
            pdf_buffer: vec![0.0; vocab_size],
            cdf_buffer_ac: vec![0u32; vocab_size + 1],
            ac_freq_buffer: vec![0i64; vocab_size],
            cdf_buffer_rans: vec![0u32; vocab_size + 1],
            rans_freq_buffer: vec![0i64; vocab_size],
            online: None,
            source_model_path: None,
        }
    }

    /// Create compressor from method string.
    pub fn new_from_method(method: &str) -> Result<Self> {
        match parse_method_spec(method)? {
            MethodSpec::File { path, policy } => {
                let mut c = Self::new(&path)?;
                if let Some(policy) = policy {
                    let canonical_method =
                        format!("file:{};policy:{}", path.display(), policy.canonical());
                    let hidden = c.model.config().hidden_size;
                    let mut online = c.online.take().unwrap_or_else(|| {
                        OnlineRuntime::new(
                            OnlineConfig::default(),
                            canonical_method.clone(),
                            Some(policy.clone()),
                            VOCAB_SIZE,
                            hidden,
                        )
                    });
                    online.canonical_method = canonical_method;
                    online.policy = Some(policy);
                    online.needs_full_trace = online
                        .policy
                        .as_ref()
                        .map(policy_needs_full_trace)
                        .unwrap_or(false);
                    c.online = Some(online);
                    c.scratch.set_capture_train_trace(
                        c.online.as_ref().is_some_and(|o| o.needs_full_trace),
                    );
                }
                Ok(c)
            }
            MethodSpec::Online { cfg, policy } => {
                let mcfg = cfg.to_mamba_config()?;
                let model = if let Some(load_from) =
                    policy.as_ref().and_then(|p| p.load_from.as_ref())
                {
                    let loaded = Arc::new(Model::load(load_from)?);
                    let loaded_cfg = loaded.config();
                    let shape_ok = loaded_cfg.vocab_size == mcfg.vocab_size
                        && loaded_cfg.hidden_size == mcfg.hidden_size
                        && loaded_cfg.num_layers == mcfg.num_layers
                        && loaded_cfg.inner_size == mcfg.inner_size
                        && loaded_cfg.state_size == mcfg.state_size
                        && loaded_cfg.conv_kernel == mcfg.conv_kernel
                        && loaded_cfg.dt_rank == mcfg.dt_rank;
                    if !shape_ok {
                        bail!(
                            "mamba policy load_from shape mismatch with cfg (strict match required)"
                        );
                    }
                    loaded
                } else {
                    Arc::new(Model::new_random(mcfg, cfg.seed)?)
                };
                let mut c = Self::new_from_model(model);
                let mut canonical_method = cfg_to_method_string(&cfg);
                if let Some(policy) = policy.as_ref() {
                    canonical_method.push_str(";policy:");
                    canonical_method.push_str(&policy.canonical());
                }
                c.online = Some(OnlineRuntime::new(
                    cfg,
                    canonical_method,
                    policy,
                    VOCAB_SIZE,
                    c.model.config().hidden_size,
                ));
                c.scratch
                    .set_capture_train_trace(c.online.as_ref().is_some_and(|o| o.needs_full_trace));
                Ok(c)
            }
        }
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    fn prepare_policy_stream(&mut self, total_symbols: Option<u64>) -> Result<()> {
        if let Some(online) = self.online.as_mut() {
            online.prepare_policy_stream(total_symbols)?;
        }
        Ok(())
    }

    /// Begin a policy stream with optional total symbol count.
    pub fn begin_online_policy_stream(&mut self, total_symbols: Option<u64>) -> Result<()> {
        self.prepare_policy_stream(total_symbols)
    }

    /// Reset and compute initial distribution.
    pub fn reset_and_prime(&mut self) {
        self.state.reset();
        self.refresh_current_pdf(0);
    }

    /// Capture runtime snapshot.
    pub fn snapshot_runtime(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            model: self.model.clone(),
            scratch: self.scratch.clone(),
            state: self.state.clone(),
            pdf_buffer: self.pdf_buffer.clone(),
            online: self.online.clone(),
        }
    }

    /// Restore runtime snapshot.
    pub fn restore_runtime(&mut self, snapshot: &RuntimeSnapshot) {
        self.model = snapshot.model.clone();
        self.scratch = snapshot.scratch.clone();
        self.state = snapshot.state.clone();
        self.pdf_buffer.clone_from(&snapshot.pdf_buffer);
        self.online = snapshot.online.clone();
    }

    /// Condition the model on a chain of prefixes.
    pub fn absorb_chain(&mut self, parts: &[&[u8]]) -> Result<()> {
        let total = parts
            .iter()
            .fold(0u64, |acc, part| acc.saturating_add(part.len() as u64));
        self.fit_chain(parts, Some(total))
    }

    /// Cross entropy from current runtime state.
    pub fn cross_entropy_from_current(&mut self, data: &[u8]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        self.prepare_policy_stream(Some(data.len() as u64))?;
        let mut total_bits = 0.0;
        for &byte in data {
            let p = self.pdf_buffer[byte as usize].max(1e-300);
            total_bits -= p.log2();
            self.online_update_from_current_pdf(byte)?;
            self.refresh_current_pdf(byte as u32);
        }
        Ok(total_bits / (data.len() as f64))
    }

    /// Fit on `fit_parts`, then reset stream state and score `data` without further adaptation.
    pub fn cross_entropy_frozen_plugin_chain(
        &mut self,
        fit_parts: &[&[u8]],
        data: &[u8],
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        if !self.can_adapt_online() {
            self.reset_and_prime();
            return self.cross_entropy_from_current(data);
        }
        self.reset_and_prime();
        let fit_total = fit_parts
            .iter()
            .fold(0u64, |acc, part| acc.saturating_add(part.len() as u64));
        self.fit_chain(fit_parts, Some(fit_total))?;
        self.reset_and_prime();

        let mut total_bits = 0.0;
        for &byte in data {
            total_bits -= self.pdf_buffer[byte as usize].max(1e-300).log2();
            self.advance_inference_only(byte);
        }
        Ok(total_bits / (data.len() as f64))
    }

    /// Whether online adaptation is enabled.
    pub fn is_online(&self) -> bool {
        self.online.is_some()
    }

    /// Returns `true` when the current online configuration can actually adapt parameters.
    pub fn can_adapt_online(&self) -> bool {
        let Some(online) = &self.online else {
            return false;
        };
        match &online.policy {
            Some(policy) => llm_policy::policy_can_train(policy),
            None => !matches!(online.cfg.train_mode, OnlineTrainMode::None),
        }
    }

    /// Tokens processed by online updater.
    pub fn tokens_processed(&self) -> u64 {
        self.online.as_ref().map_or(0, |s| s.tokens_processed)
    }

    /// Canonical online method string.
    pub fn online_method_string(&self) -> Option<&str> {
        self.online.as_ref().map(|s| s.canonical_method.as_str())
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    /// Convert logits to PDF with online bias if active.
    pub fn online_apply_logits_bias(&self, logits: &[f32], pdf_out: &mut [f64]) {
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        Self::logits_to_pdf(logits, bias, pdf_out);
    }

    /// Convert logits + optional bias into stable normalized PDF.
    pub fn logits_to_pdf(logits: &[f32], bias: Option<&[f32]>, pdf_out: &mut [f64]) {
        softmax_pdf_floor_with_bias(logits, bias, pdf_out);
    }

    #[inline]
    pub fn forward_to_pdf(&mut self, token: u32, pdf_out: &mut [f64]) {
        let logits = self
            .model
            .forward(&mut self.scratch, token, &mut self.state);
        let bias = self.online.as_ref().map(|o| o.out_bias.as_slice());
        Self::logits_to_pdf(logits, bias, pdf_out);
    }

    /// Snapshot online bias only.
    pub fn online_bias_snapshot(&self) -> Option<Vec<f32>> {
        self.online.as_ref().map(|o| o.out_bias.clone())
    }

    #[inline]
    pub fn online_bias_slice(&self) -> Option<&[f32]> {
        self.online.as_ref().map(|o| o.out_bias.as_slice())
    }

    /// Apply one online update using external PDF.
    pub fn online_update_from_pdf(&mut self, symbol: u8, pdf: &[f64]) -> Result<()> {
        self.online_update_with_pdf(symbol, pdf)
    }

    fn online_update_with_pdf(&mut self, symbol: u8, pdf: &[f64]) -> Result<()> {
        let Some(online) = self.online.as_mut() else {
            return Ok(());
        };
        online.tokens_processed = online.tokens_processed.saturating_add(1);

        let mut optimizer = match online.cfg.train_mode {
            OnlineTrainMode::None => OptimizerKind::Sgd,
            OnlineTrainMode::Sgd => OptimizerKind::Sgd,
            OnlineTrainMode::Adam => OptimizerKind::Adam,
        };
        let mut lr = online.cfg.lr.max(0.0);
        let mut stride = online.cfg.stride.max(1) as u64;
        let mut scope = mamba1::TrainScopeMask::default();
        let default_train = !matches!(online.cfg.train_mode, OnlineTrainMode::None);
        scope.head = default_train;
        scope.bias = default_train;
        let mut bptt = 1usize;
        let mut clip = 0.0f32;

        if let Some(action) = online.next_policy_action()? {
            match action {
                PolicyAction::Infer => {
                    scope = mamba1::TrainScopeMask::default();
                }
                PolicyAction::Train(train) => {
                    optimizer = train.optimizer;
                    lr = train.hyper.lr.max(0.0);
                    stride = train.hyper.stride.max(1) as u64;
                    clip = train.hyper.clip.max(0.0);
                    bptt = train.hyper.bptt.max(1);
                    if train.scope.all {
                        scope = mamba1::TrainScopeMask::all();
                    } else {
                        scope = mamba1::TrainScopeMask::default();
                        scope.embed = train.scope.contains("embed");
                        scope.layer_norm = train.scope.contains("layer_norm");
                        scope.mixer_conv = train.scope.contains("mixer_conv");
                        scope.mixer_ssm = train.scope.contains("mixer_ssm");
                        scope.mixer_proj = train.scope.contains("mixer_proj");
                        scope.head = train.scope.contains("head");
                        scope.bias = train.scope.contains("bias");
                    }
                }
            }
        }

        if !scope.trains_model_params() && !scope.bias {
            return Ok(());
        }
        online.policy_train_steps = online.policy_train_steps.saturating_add(1);
        if stride > 1 && (online.policy_train_steps % stride) != 0 {
            return Ok(());
        }
        if lr == 0.0 {
            return Ok(());
        }

        if bptt > 1
            && (scope.embed
                || scope.layer_norm
                || scope.mixer_conv
                || scope.mixer_ssm
                || scope.mixer_proj)
        {
            bail!("mamba full-parameter online training currently supports bptt=1");
        }

        if scope.trains_model_params() {
            self.scratch.set_capture_train_trace(true);
        }

        if matches!(optimizer, OptimizerKind::Adam) {
            if scope.bias && (online.adam_m.is_none() || online.adam_v.is_none()) {
                online.adam_m = Some(vec![0.0; online.out_bias.len()]);
                online.adam_v = Some(vec![0.0; online.out_bias.len()]);
            }
            if scope.trains_model_params() && online.full_adam.is_none() {
                online.full_adam = Some(self.model.as_ref().new_full_adam_state());
            }
        }

        let model = Arc::make_mut(&mut self.model);
        let OnlineRuntime {
            out_bias,
            adam_m,
            adam_v,
            full_adam,
            adam_t,
            ..
        } = online;
        model.online_train_step_bptt1(
            &mut self.scratch,
            &self.state,
            symbol,
            pdf,
            scope,
            optimizer,
            lr,
            clip,
            adam_t,
            full_adam.as_mut(),
            if scope.bias {
                Some(out_bias.as_mut_slice())
            } else {
                None
            },
            if scope.bias {
                adam_m.as_deref_mut()
            } else {
                None
            },
            if scope.bias {
                adam_v.as_deref_mut()
            } else {
                None
            },
        )?;

        Ok(())
    }

    #[inline]
    fn refresh_current_pdf(&mut self, token: u32) {
        let logits = self
            .model
            .forward(&mut self.scratch, token, &mut self.state);
        let bias = self.online.as_ref().map(|o| o.out_bias.as_slice());
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
    }

    fn fit_chain(&mut self, parts: &[&[u8]], total_symbols: Option<u64>) -> Result<()> {
        self.prepare_policy_stream(total_symbols)?;
        for part in parts {
            for &byte in *part {
                self.online_update_from_current_pdf(byte)?;
                self.refresh_current_pdf(byte as u32);
            }
        }
        Ok(())
    }

    #[inline]
    fn advance_inference_only(&mut self, symbol: u8) {
        self.refresh_current_pdf(symbol as u32);
    }

    fn online_update_from_current_pdf(&mut self, symbol: u8) -> Result<()> {
        let Some(online) = self.online.as_mut() else {
            return Ok(());
        };
        online.tokens_processed = online.tokens_processed.saturating_add(1);

        let mut optimizer = match online.cfg.train_mode {
            OnlineTrainMode::None => OptimizerKind::Sgd,
            OnlineTrainMode::Sgd => OptimizerKind::Sgd,
            OnlineTrainMode::Adam => OptimizerKind::Adam,
        };
        let mut lr = online.cfg.lr.max(0.0);
        let mut stride = online.cfg.stride.max(1) as u64;
        let mut scope = mamba1::TrainScopeMask::default();
        let default_train = !matches!(online.cfg.train_mode, OnlineTrainMode::None);
        scope.head = default_train;
        scope.bias = default_train;
        let mut bptt = 1usize;
        let mut clip = 0.0f32;

        if let Some(action) = online.next_policy_action()? {
            match action {
                PolicyAction::Infer => {
                    scope = mamba1::TrainScopeMask::default();
                }
                PolicyAction::Train(train) => {
                    optimizer = train.optimizer;
                    lr = train.hyper.lr.max(0.0);
                    stride = train.hyper.stride.max(1) as u64;
                    clip = train.hyper.clip.max(0.0);
                    bptt = train.hyper.bptt.max(1);
                    if train.scope.all {
                        scope = mamba1::TrainScopeMask::all();
                    } else {
                        scope = mamba1::TrainScopeMask::default();
                        scope.embed = train.scope.contains("embed");
                        scope.layer_norm = train.scope.contains("layer_norm");
                        scope.mixer_conv = train.scope.contains("mixer_conv");
                        scope.mixer_ssm = train.scope.contains("mixer_ssm");
                        scope.mixer_proj = train.scope.contains("mixer_proj");
                        scope.head = train.scope.contains("head");
                        scope.bias = train.scope.contains("bias");
                    }
                }
            }
        }

        if !scope.trains_model_params() && !scope.bias {
            return Ok(());
        }
        online.policy_train_steps = online.policy_train_steps.saturating_add(1);
        if stride > 1 && (online.policy_train_steps % stride) != 0 {
            return Ok(());
        }
        if lr == 0.0 {
            return Ok(());
        }

        if bptt > 1
            && (scope.embed
                || scope.layer_norm
                || scope.mixer_conv
                || scope.mixer_ssm
                || scope.mixer_proj)
        {
            bail!("mamba full-parameter online training currently supports bptt=1");
        }
        if scope.trains_model_params() {
            self.scratch.set_capture_train_trace(true);
        }
        if matches!(optimizer, OptimizerKind::Adam) {
            if scope.bias && (online.adam_m.is_none() || online.adam_v.is_none()) {
                online.adam_m = Some(vec![0.0; online.out_bias.len()]);
                online.adam_v = Some(vec![0.0; online.out_bias.len()]);
            }
            if scope.trains_model_params() && online.full_adam.is_none() {
                online.full_adam = Some(self.model.as_ref().new_full_adam_state());
            }
        }

        let model = Arc::make_mut(&mut self.model);
        let pdf = &self.pdf_buffer;
        let OnlineRuntime {
            out_bias,
            adam_m,
            adam_v,
            full_adam,
            adam_t,
            ..
        } = online;
        model.online_train_step_bptt1(
            &mut self.scratch,
            &self.state,
            symbol,
            pdf,
            scope,
            optimizer,
            lr,
            clip,
            adam_t,
            full_adam.as_mut(),
            if scope.bias {
                Some(out_bias.as_mut_slice())
            } else {
                None
            },
            if scope.bias {
                adam_m.as_deref_mut()
            } else {
                None
            },
            if scope.bias {
                adam_v.as_deref_mut()
            } else {
                None
            },
        )?;

        Ok(())
    }

    /// Export model and online sidecar.
    pub fn export_online<P: AsRef<Path>>(&self, model_path: P) -> Result<()> {
        let model_path = model_path.as_ref();
        self.model.save_safetensors(model_path)?;
        let opt_sidecar = optimizer_sidecar_path(model_path);

        let sidecar = model_path.with_extension("json");
        let meta = if let Some(online) = &self.online {
            if let Some(full_adam) = online.full_adam.as_ref() {
                self.model
                    .save_full_adam_safetensors(full_adam, &opt_sidecar)?;
            } else if opt_sidecar.exists() {
                let _ = fs::remove_file(&opt_sidecar);
            }
            let train_mode = match online.cfg.train_mode {
                OnlineTrainMode::None => "none",
                OnlineTrainMode::Sgd => "sgd",
                OnlineTrainMode::Adam => "adam",
            };
            json!({
                "version": 1,
                "method": online.canonical_method,
                "policy": online.policy.as_ref().map(LlmPolicy::canonical),
                "policy_cursor": online.policy_runtime.as_ref().map(PolicyRuntime::cursor).unwrap_or(0),
                "policy_stream_total": online.policy_stream_total,
                "policy_train_steps": online.policy_train_steps,
                "training_mode": train_mode,
                "tokens_processed": online.tokens_processed,
                "adam_t": online.adam_t,
                "has_full_adam": online.full_adam.is_some(),
                "config": {
                    "hidden": online.cfg.hidden,
                    "layers": online.cfg.layers,
                    "intermediate": online.cfg.intermediate,
                    "state": online.cfg.state,
                    "conv": online.cfg.conv,
                    "dt_rank": online.cfg.dt_rank,
                    "seed": online.cfg.seed,
                    "lr": online.cfg.lr,
                    "stride": online.cfg.stride.max(1),
                },
                "output_bias": online.out_bias,
                "adam_m": online.adam_m,
                "adam_v": online.adam_v,
                "lm_head_adam_m": online.lm_head_adam_m,
                "lm_head_adam_v": online.lm_head_adam_v,
            })
        } else {
            if opt_sidecar.exists() {
                let _ = fs::remove_file(&opt_sidecar);
            }
            json!({
                "version": 1,
                "method": format!("file:{}", model_path.display()),
                "training_mode": "none",
                "tokens_processed": 0,
            })
        };

        fs::write(sidecar, serde_json::to_vec_pretty(&meta)?)?;
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
        let parse_vec_f32 = |key: &str| -> Option<Vec<f32>> {
            v.get(key).and_then(|arr| arr.as_array()).map(|arr| {
                arr.iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect::<Vec<f32>>()
            })
        };
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
        let has_full_adam = v
            .get("has_full_adam")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let policy = v
            .get("policy")
            .and_then(|p| p.as_str())
            .and_then(|s| llm_policy::parse_policy_segment(s, MAMBA_TRAIN_SCOPES).ok());
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
                if let Some(x) = cfg_v.get("state").and_then(|x| x.as_u64()) {
                    cfg.state = x as usize;
                }
                if let Some(x) = cfg_v.get("conv").and_then(|x| x.as_u64()) {
                    cfg.conv = x as usize;
                }
                if let Some(x) = cfg_v.get("dt_rank").and_then(|x| x.as_u64()) {
                    cfg.dt_rank = x as usize;
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
            let needs_full_trace = policy
                .as_ref()
                .map(policy_needs_full_trace)
                .unwrap_or(false);

            self.online = Some(OnlineRuntime {
                cfg,
                canonical_method: method,
                policy,
                policy_runtime: None,
                needs_full_trace,
                policy_stream_total: v.get("policy_stream_total").and_then(|x| x.as_u64()),
                policy_train_steps: v
                    .get("policy_train_steps")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0),
                tokens_processed: tokens,
                out_bias,
                adam_m: parse_vec_f32("adam_m"),
                adam_v: parse_vec_f32("adam_v"),
                full_adam: None,
                lm_head_adam_m: parse_vec_f32("lm_head_adam_m"),
                lm_head_adam_v: parse_vec_f32("lm_head_adam_v"),
                adam_t: v.get("adam_t").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            });
            let opt_sidecar = optimizer_sidecar_path(model_path);
            if opt_sidecar.exists() {
                if let Some(online) = self.online.as_mut() {
                    online.full_adam = Some(self.model.load_full_adam_safetensors(&opt_sidecar)?);
                }
            } else if has_full_adam {
                bail!(
                    "missing optimizer sidecar '{}' required for exact online resume",
                    opt_sidecar.display()
                );
            }
            if let Some(cursor) = v.get("policy_cursor").and_then(|x| x.as_u64())
                && let Some(online) = self.online.as_mut()
                && online.policy.is_some()
            {
                let train_steps = online.policy_train_steps;
                online.prepare_policy_stream(online.policy_stream_total)?;
                online.policy_train_steps = train_steps;
                if let Some(rt) = online.policy_runtime.as_mut() {
                    rt.set_cursor(cursor);
                }
            }
            self.scratch
                .set_capture_train_trace(self.online.as_ref().is_some_and(|o| o.needs_full_trace));
        }
        Ok(())
    }

    /// Compress into writer.
    pub fn compress_into<W: Write>(
        &mut self,
        data: &[u8],
        coder: CoderType,
        w: &mut W,
    ) -> Result<()> {
        self.state.reset();
        self.prepare_policy_stream(Some(data.len() as u64))?;
        let checksum = crc32(data);
        let header = Header::new(coder, data.len() as u64, checksum);
        header.write(w)?;

        match coder {
            CoderType::AC => self.compress_ac_iter(data.iter().copied(), w)?,
            CoderType::RANS => self.compress_rans_iter(data.iter().copied(), w)?,
        }
        Ok(())
    }

    /// Compress a chain of byte slices into writer.
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
        self.prepare_policy_stream(Some(total_len))?;
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

    /// Return compressed size without output allocation.
    pub fn compress_size(&mut self, data: &[u8], coder: CoderType) -> Result<u64> {
        let mut w = CountingWriter::new();
        self.compress_into(data, coder, &mut w)?;
        Ok(w.bytes_written())
    }

    /// Return compressed size for chained inputs.
    pub fn compress_size_chain(&mut self, parts: &[&[u8]], coder: CoderType) -> Result<u64> {
        let mut w = CountingWriter::new();
        self.compress_chain_into(parts, coder, &mut w)?;
        Ok(w.bytes_written())
    }

    /// Compress to bytes.
    pub fn compress(&mut self, data: &[u8], coder: CoderType) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.compress_into(data, coder, &mut out)?;
        Ok(out)
    }

    fn compress_ac_iter<I, W: Write>(&mut self, data: I, output: &mut W) -> Result<()>
    where
        I: IntoIterator<Item = u8>,
    {
        let mut encoder = ArithmeticEncoder::new(output);

        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        for byte in data {
            quantize_pdf_to_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_ac,
                &mut self.ac_freq_buffer,
            );
            let sym = byte as usize;
            let lo = self.cdf_buffer_ac[sym] as u64;
            let hi = self.cdf_buffer_ac[sym + 1] as u64;
            encoder.encode_counts(lo, hi, CDF_TOTAL as u64)?;
            self.online_update_from_current_pdf(byte)?;

            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
        }

        let _ = encoder.finish()?;
        Ok(())
    }

    fn compress_rans_iter<I, W: Write>(&mut self, data: I, output: &mut W) -> Result<()>
    where
        I: IntoIterator<Item = u8>,
    {
        let mut encoder = BlockedRansEncoder::new();

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

            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
        }

        let blocks = encoder.finish();
        output.write_all(&(blocks.len() as u32).to_le_bytes())?;
        for block in &blocks {
            output.write_all(&(block.len() as u32).to_le_bytes())?;
            output.write_all(block)?;
        }
        Ok(())
    }

    /// Decompress bytes.
    pub fn decompress(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let mut cursor = Cursor::new(data);
        let header = Header::read(&mut cursor)?;

        self.state.reset();
        self.prepare_policy_stream(Some(header.original_len))?;
        let compressed = &data[Header::SIZE..];
        let result = match header.coder_type() {
            CoderType::AC => self.decompress_ac(compressed, header.original_len as usize)?,
            CoderType::RANS => self.decompress_rans(compressed, header.original_len as usize)?,
        };

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

    fn decompress_ac(&mut self, compressed: &[u8], original_len: usize) -> Result<Vec<u8>> {
        let mut decoder = ArithmeticDecoder::new(compressed)?;
        let mut result = Vec::with_capacity(original_len);

        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        for _ in 0..original_len {
            quantize_pdf_to_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_ac,
                &mut self.ac_freq_buffer,
            );
            let sym = decoder.decode_symbol_counts(&self.cdf_buffer_ac, CDF_TOTAL)?;
            let byte = sym as u8;
            result.push(byte);
            self.online_update_from_current_pdf(byte)?;

            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, byte as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
        }

        Ok(result)
    }

    fn decompress_rans(&mut self, compressed: &[u8], original_len: usize) -> Result<Vec<u8>> {
        if compressed.len() < 4 {
            bail!("rANS data too short");
        }
        let block_count =
            u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]])
                as usize;

        let mut blocks = Vec::with_capacity(block_count);
        let mut pos = 4usize;
        for _ in 0..block_count {
            if pos + 4 > compressed.len() {
                bail!("truncated rANS block header");
            }
            let len = u32::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
            ]) as usize;
            pos += 4;
            if pos + len > compressed.len() {
                bail!("truncated rANS block data");
            }
            blocks.push(&compressed[pos..pos + len]);
            pos += len;
        }

        let mut decoder = BlockedRansDecoder::new(blocks, original_len)?;
        let mut result = Vec::with_capacity(original_len);

        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        let logits = self.model.forward(&mut self.scratch, 0, &mut self.state);
        Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

        for _ in 0..original_len {
            quantize_pdf_to_rans_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_rans,
                &mut self.rans_freq_buffer,
            );
            let sym = decoder.decode(&self.cdf_buffer_rans)? as u8;
            result.push(sym);
            self.online_update_from_current_pdf(sym)?;

            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, sym as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);
        }

        Ok(result)
    }

    /// Cross entropy over whole sample.
    pub fn cross_entropy(&mut self, data: &[u8]) -> Result<f64> {
        self.reset_and_prime();
        self.cross_entropy_from_current(data)
    }

    /// Cross entropy conditioned on prefix chain.
    pub fn cross_entropy_conditional_chain(
        &mut self,
        prefix_parts: &[&[u8]],
        data: &[u8],
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let prefix_len = prefix_parts
            .iter()
            .fold(0usize, |acc, p| acc.saturating_add(p.len()));
        self.reset_and_prime();
        self.fit_chain(prefix_parts, Some((prefix_len + data.len()) as u64))?;

        let mut total_bits = 0.0;
        for &byte in data {
            total_bits -= self.pdf_buffer[byte as usize].max(1e-300).log2();
            self.online_update_from_current_pdf(byte)?;
            self.refresh_current_pdf(byte as u32);
        }
        Ok(total_bits / (data.len() as f64))
    }

    /// Cross entropy conditioned on one prefix.
    pub fn cross_entropy_conditional(&mut self, prefix: &[u8], data: &[u8]) -> Result<f64> {
        self.cross_entropy_conditional_chain(&[prefix], data)
    }

    /// Symmetric aligned joint cross entropy using the better ordering.
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

        self.reset_and_prime();
        self.prepare_policy_stream(Some((2 * n) as u64))?;

        let mut total_bits = 0.0;
        for idx in 0..n {
            let a = if swap { y[idx] } else { x[idx] };
            let b = if swap { x[idx] } else { y[idx] };

            total_bits -= self.pdf_buffer[a as usize].max(1e-300).log2();
            self.online_update_from_current_pdf(a)?;
            let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
            let logits = self
                .model
                .forward(&mut self.scratch, a as u32, &mut self.state);
            Self::logits_to_pdf(logits, bias, &mut self.pdf_buffer);

            total_bits -= self.pdf_buffer[b as usize].max(1e-300).log2();
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
