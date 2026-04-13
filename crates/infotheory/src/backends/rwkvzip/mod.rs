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

use crate::backends::llm_policy::{
    self, LlmPolicy, OptimizerKind, PolicyAction, PolicyRuntime, split_method_policy_segments,
};
/// RWKV7 model core (weights/state/kernels).
pub mod rwkv7;
/// Shared entropy coders used by rwkvzip containers.
pub use crate::coders;
/// Backward-compatible re-export of generic coder selection enum.
pub use crate::coders::CoderType;

use crate::coders::{
    ANS_TOTAL, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder, BlockedRansEncoder,
    CDF_TOTAL, Cdf, quantize_pdf_to_cdf_with_buffer, quantize_pdf_to_rans_cdf_with_buffer,
};

/// RWKV7 model configuration type.
pub use rwkv7::Config;
/// RWKV7 model type.
pub use rwkv7::Model;
/// RWKV7 temporary scratch buffers used during forward passes.
pub use rwkv7::ScratchBuffers;
/// RWKV7 recurrent state container.
pub use rwkv7::State;

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
/// Fast default full-parameter TBPTT window used when policy requests `bptt<=1`.
const DEFAULT_FULL_TBPTT_WINDOW: usize = 8;
const TBPTT_REPLAY_CHUNK: usize = 32;
fn optimizer_sidecar_path(model_path: &Path) -> PathBuf {
    model_path.with_extension("opt.safetensors")
}
const RWKV_TRAIN_SCOPES: &[&str] = &[
    "embed",
    "pre_norm",
    "attn_norm",
    "ffn_norm",
    "attn",
    "ffn",
    "head",
    "bias",
    "all",
    "none",
];

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Online adaptation mode for RWKV output-bias updates.
pub enum OnlineTrainMode {
    /// Disable online updates.
    None,
    /// SGD updates on output bias.
    Sgd,
    /// Adam updates on output bias.
    Adam,
}

#[derive(Clone, Debug)]
/// Configuration for online RWKV model instantiation and adaptation.
pub struct OnlineConfig {
    /// Hidden size (must be multiple of 64 after clamping).
    pub hidden: usize,
    /// Number of recurrent layers.
    pub layers: usize,
    /// Feed-forward intermediate size.
    pub intermediate: usize,
    /// Low-rank dimension for decay projection.
    pub decay_rank: usize,
    /// Low-rank dimension for key-like projection.
    pub a_rank: usize,
    /// Low-rank dimension for value-like projection.
    pub v_rank: usize,
    /// Low-rank dimension for gate projection.
    pub g_rank: usize,
    /// Random seed used for online-random model initialization.
    pub seed: u64,
    /// Online training mode.
    pub train_mode: OnlineTrainMode,
    /// Learning rate for online adaptation.
    pub lr: f32,
    /// Update stride (apply update every `stride` tokens).
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
    /// Convert to a validated RWKV model configuration.
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
/// Parsed RWKV method specification.
pub enum MethodSpec {
    /// Load a model from disk.
    File {
        /// Path to `.safetensors` model weights.
        path: PathBuf,
        /// Optional runtime training/inference policy.
        policy: Option<LlmPolicy>,
    },
    /// Build an online/random model from configuration.
    Online {
        /// Online model/training configuration.
        cfg: OnlineConfig,
        /// Optional runtime training/inference policy.
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
    full_adam: Option<rwkv7::FullAdamState>,
    lm_head_adam_m: Option<Vec<f32>>,
    lm_head_adam_v: Option<Vec<f32>>,
    adam_t: usize,
    full_tbptt: Option<FullTbpttRuntime>,
}

#[derive(Clone, Copy, Debug)]
struct FullTrainSettings {
    optimizer: OptimizerKind,
    lr: f32,
    scope: rwkv7::TrainScopeMask,
    bptt: usize,
    clip: f32,
}

impl FullTrainSettings {
    fn matches(
        self,
        optimizer: OptimizerKind,
        lr: f32,
        scope: rwkv7::TrainScopeMask,
        bptt: usize,
        clip: f32,
    ) -> bool {
        self.optimizer == optimizer
            && self.lr.to_bits() == lr.to_bits()
            && self.scope == scope
            && self.bptt == bptt
            && self.clip.to_bits() == clip.to_bits()
    }
}

#[derive(Clone)]
struct FullTbpttRuntime {
    pending_input_token: Option<u32>,
    pending_input_pre_state: Option<State>,
    segment_start_state: Option<State>,
    steps: Vec<(u32, u8)>,
    settings: Option<FullTrainSettings>,
}

#[derive(Clone)]
/// Snapshot of mutable runtime state used for reversible scoring.
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
            full_tbptt: needs_full_trace.then(|| FullTbpttRuntime {
                pending_input_token: None,
                pending_input_pre_state: None,
                segment_start_state: None,
                steps: Vec::new(),
                settings: None,
            }),
        }
    }

    fn prepare_policy_stream(&mut self, total_symbols: Option<u64>) -> Result<()> {
        self.policy_stream_total = total_symbols;
        self.policy_train_steps = 0;
        if let Some(tbptt) = self.full_tbptt.as_mut() {
            // Preserve the current predictive edge so the first symbol of the
            // new stream still trains against the already-primed distribution.
            tbptt.segment_start_state = None;
            tbptt.steps.clear();
            tbptt.settings = None;
        }
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

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn apply_online_lm_head_update(
    model: &mut Model,
    online: &mut OnlineRuntime,
    hidden: &[f32],
    symbol: u8,
    pdf: &[f64],
    lr: f32,
    optimizer: OptimizerKind,
    train_head: bool,
    train_bias: bool,
    clip: f32,
) {
    let h = hidden.len();
    if h == 0 {
        return;
    }

    let head = model.lm_head_weights_mut();
    let vocab_rows = head.len() / h;
    let n = online.out_bias.len().min(pdf.len()).min(vocab_rows);

    match optimizer {
        OptimizerKind::Sgd => {
            for (i, p_raw) in pdf.iter().enumerate().take(n) {
                let p = (*p_raw).clamp(1e-12, 1.0) as f32;
                let target = if i == symbol as usize { 1.0 } else { 0.0 };
                let mut grad = target - p;
                if clip > 0.0 {
                    grad = grad.clamp(-clip, clip);
                }
                if train_bias {
                    online.out_bias[i] += lr * grad;
                }

                if train_head {
                    let row_off = i * h;
                    for j in 0..h {
                        head[row_off + j] += lr * grad * hidden[j];
                    }
                }
            }
        }
        OptimizerKind::Adam => {
            online.adam_t = online.adam_t.saturating_add(1);
            let t = online.adam_t as i32;
            let b1 = 0.9f32;
            let b2 = 0.999f32;
            let eps = 1e-8f32;
            let bias_corr1 = 1.0 - b1.powi(t);
            let bias_corr2 = 1.0 - b2.powi(t);
            if online.adam_m.is_none() || online.adam_v.is_none() {
                online.adam_m = Some(vec![0.0; online.out_bias.len()]);
                online.adam_v = Some(vec![0.0; online.out_bias.len()]);
            }
            if online.lm_head_adam_m.is_none() || online.lm_head_adam_v.is_none() {
                online.lm_head_adam_m = Some(vec![0.0; vocab_rows * h]);
                online.lm_head_adam_v = Some(vec![0.0; vocab_rows * h]);
            }
            let bm = online.adam_m.as_mut().expect("adam_m initialized");
            let bv = online.adam_v.as_mut().expect("adam_v initialized");
            let hm = online
                .lm_head_adam_m
                .as_mut()
                .expect("lm_head_adam_m initialized");
            let hv = online
                .lm_head_adam_v
                .as_mut()
                .expect("lm_head_adam_v initialized");
            for i in 0..n {
                let p = pdf[i].clamp(1e-12, 1.0) as f32;
                let target = if i == symbol as usize { 1.0 } else { 0.0 };
                let mut grad = target - p;
                if clip > 0.0 {
                    grad = grad.clamp(-clip, clip);
                }

                if train_bias {
                    bm[i] = b1 * bm[i] + (1.0 - b1) * grad;
                    bv[i] = b2 * bv[i] + (1.0 - b2) * grad * grad;
                    let m_hat = bm[i] / bias_corr1;
                    let v_hat = bv[i] / bias_corr2;
                    online.out_bias[i] += lr * m_hat / (v_hat.sqrt() + eps);
                }

                if train_head {
                    let row_off = i * h;
                    for j in 0..h {
                        let idx = row_off + j;
                        let g = grad * hidden[j];
                        hm[idx] = b1 * hm[idx] + (1.0 - b1) * g;
                        hv[idx] = b2 * hv[idx] + (1.0 - b2) * g * g;
                        let m_hat_w = hm[idx] / bias_corr1;
                        let v_hat_w = hv[idx] / bias_corr2;
                        head[idx] += lr * m_hat_w / (v_hat_w.sqrt() + eps);
                    }
                }
            }
        }
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
        || scope.contains("pre_norm")
        || scope.contains("attn_norm")
        || scope.contains("ffn_norm")
        || scope.contains("attn")
        || scope.contains("ffn")
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

fn scope_from_train_action(train: &llm_policy::TrainAction) -> rwkv7::TrainScopeMask {
    if train.scope.all {
        return rwkv7::TrainScopeMask::all();
    }
    rwkv7::TrainScopeMask {
        embed: train.scope.contains("embed"),
        pre_norm: train.scope.contains("pre_norm"),
        attn_norm: train.scope.contains("attn_norm"),
        ffn_norm: train.scope.contains("ffn_norm"),
        attn: train.scope.contains("attn"),
        ffn: train.scope.contains("ffn"),
        head: train.scope.contains("head"),
        bias: train.scope.contains("bias"),
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

/// Parse a method string into a concrete RWKV method specification.
///
/// Supported formats:
/// - `file:/path/to/model.safetensors`
/// - `file:/path/to/model%3Bv1.safetensors`
/// - `file:/path/to/model.safetensors;policy:...`
/// - `cfg:key=value,...[;policy:...]`
/// - positional `cfg` CSV
/// - existing model path
///
/// `file:` methods percent-encode reserved delimiters inside the path segment.
pub fn parse_method_spec(method: &str) -> Result<MethodSpec> {
    let (base, policy_segment) = split_method_policy_segments(method)?;
    let parse_policy = |s: &str| llm_policy::parse_policy_segment(s, RWKV_TRAIN_SCOPES);
    let policy = policy_segment
        .as_deref()
        .map(parse_policy)
        .transpose()
        .context("failed to parse rwkv policy segment")?;

    if let Some(path) = base.strip_prefix("file:") {
        let p = llm_policy::parse_method_file_path(path.trim());
        if p.as_os_str().is_empty() {
            bail!("empty file path in rwkv method");
        }
        llm_policy::canonical_file_method_string(&p, policy.as_ref())?;
        if policy.as_ref().and_then(|p| p.load_from.as_ref()).is_some() {
            bail!("rwkv method cannot use policy load_from together with file:<path>");
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
        return Ok(MethodSpec::Online { cfg, policy });
    }

    let plain = PathBuf::from(base.trim());
    if plain.exists() {
        if policy.as_ref().and_then(|p| p.load_from.as_ref()).is_some() {
            bail!("rwkv method cannot use policy load_from together with file path");
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
        "rwkv method must be 'file:<path>', 'cfg:<k=v,...>', positional cfg CSV, or an existing model path"
    );
}

/// Convert a parsed method specification back into canonical method syntax.
pub fn canonical_method_string(spec: &MethodSpec) -> Result<String> {
    match spec {
        MethodSpec::File { path, policy } => {
            llm_policy::canonical_file_method_string(path, policy.as_ref())
        }
        MethodSpec::Online { cfg, policy } => {
            let mut method = cfg_to_method_string(cfg);
            if let Some(policy) = policy {
                method.push_str(";policy:");
                method.push_str(&policy.canonical());
            }
            Ok(method)
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
    crate::coders::crc32(data)
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
    /// Scratch buffers for model forward passes.
    pub scratch: ScratchBuffers,
    /// Pre-allocated PDF buffer (eliminates allocations in compression loop).
    pub pdf_buffer: Vec<f64>,
    /// Reusable AC CDF buffer (vocab_size + 1 entries).
    pub cdf_buffer_ac: Vec<u32>,
    /// Scratch frequencies for AC quantization.
    pub ac_freq_buffer: Vec<i64>,
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

    /// Load a model from disk and wrap it in `Arc`.
    pub fn load_model<P: AsRef<Path>>(model_path: P) -> Result<Arc<Model>> {
        Ok(Arc::new(Model::load(model_path)?))
    }

    /// Create a compressor from a preloaded model.
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
            ac_freq_buffer: vec![0i64; vocab_size],
            cdf_buffer_rans: vec![0u32; vocab_size + 1],
            rans_freq_buffer: vec![0i64; vocab_size],
            online: None,
            source_model_path: None,
        }
    }

    /// Create a compressor from a user method string.
    pub fn new_from_method(method: &str) -> Result<Self> {
        let spec = parse_method_spec(method)?;
        Self::new_from_method_spec(&spec)
    }

    /// Create a compressor from a parsed method specification.
    pub fn new_from_method_spec(spec: &MethodSpec) -> Result<Self> {
        match spec {
            MethodSpec::File { path, policy } => {
                let mut c = Self::new(path)?;
                if let Some(policy) = policy.as_ref() {
                    let canonical_method = canonical_method_string(spec)?;
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
                    online.policy = Some(policy.clone());
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
                let rwcfg = cfg.to_rwkv_config()?;
                let model = if let Some(load_from) =
                    policy.as_ref().and_then(|p| p.load_from.as_ref())
                {
                    let loaded = Arc::new(Model::load(load_from)?);
                    let loaded_cfg = loaded.config();
                    let shape_ok = loaded_cfg.vocab_size == rwcfg.vocab_size
                        && loaded_cfg.hidden_size == rwcfg.hidden_size
                        && loaded_cfg.num_layers == rwcfg.num_layers
                        && loaded_cfg.num_heads == rwcfg.num_heads
                        && loaded_cfg.head_dim == rwcfg.head_dim
                        && loaded_cfg.intermediate_size == rwcfg.intermediate_size
                        && loaded_cfg.decay_low_rank == rwcfg.decay_low_rank
                        && loaded_cfg.a_low_rank == rwcfg.a_low_rank
                        && loaded_cfg.v_low_rank == rwcfg.v_low_rank
                        && loaded_cfg.g_low_rank == rwcfg.g_low_rank;
                    if !shape_ok {
                        bail!(
                            "rwkv policy load_from shape mismatch with cfg (strict match required)"
                        );
                    }
                    loaded
                } else {
                    Arc::new(Model::new_random(rwcfg, cfg.seed)?)
                };
                let mut c = Self::new_from_model(model);
                let canonical_method = canonical_method_string(spec)?;
                c.online = Some(OnlineRuntime::new(
                    cfg.clone(),
                    canonical_method,
                    policy.clone(),
                    VOCAB_SIZE,
                    c.model.config().hidden_size,
                ));
                c.scratch
                    .set_capture_train_trace(c.online.as_ref().is_some_and(|o| o.needs_full_trace));
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
        self.clear_online_training_buffers();
    }

    fn prepare_policy_stream(&mut self, total_symbols: Option<u64>) -> Result<()> {
        if let Some(online) = self.online.as_mut() {
            online.prepare_policy_stream(total_symbols)?;
        }
        Ok(())
    }

    #[inline]
    fn effective_full_bptt(scope: rwkv7::TrainScopeMask, bptt: usize) -> usize {
        if scope.trains_non_head_params() && bptt <= 1 {
            DEFAULT_FULL_TBPTT_WINDOW
        } else {
            bptt.max(1)
        }
    }

    fn clear_online_training_buffers(&mut self) {
        if let Some(online) = self.online.as_mut()
            && let Some(tbptt) = online.full_tbptt.as_mut()
        {
            tbptt.pending_input_token = None;
            tbptt.pending_input_pre_state = None;
            tbptt.segment_start_state = None;
            tbptt.steps.clear();
            tbptt.settings = None;
        }
    }

    fn forward_with_online_record(&mut self, token: u32) {
        if let Some(online) = self.online.as_mut()
            && let Some(tbptt) = online.full_tbptt.as_mut()
        {
            tbptt.pending_input_token = Some(token);
            tbptt.pending_input_pre_state = Some(self.state.clone());
        }
        let _ = self
            .model
            .forward(&mut self.scratch, token, &mut self.state);
    }

    fn flush_full_tbptt_segment(&mut self) -> Result<()> {
        let extracted = {
            match self.online.as_mut() {
                Some(online) => match online.full_tbptt.as_mut() {
                    Some(tbptt) if !tbptt.steps.is_empty() => {
                        let settings = tbptt.settings.ok_or_else(|| {
                            anyhow::anyhow!("rwkv full tbptt settings are missing")
                        })?;
                        let start_state = tbptt.segment_start_state.clone().ok_or_else(|| {
                            anyhow::anyhow!("rwkv full tbptt segment start is missing")
                        })?;
                        let steps = tbptt.steps.clone();
                        tbptt.steps.clear();
                        tbptt.segment_start_state = None;
                        tbptt.settings = None;
                        let need_full_adam = matches!(settings.optimizer, OptimizerKind::Adam)
                            && settings.scope.trains_non_head_params()
                            && online.full_adam.is_none();
                        Some((settings, start_state, steps, need_full_adam))
                    }
                    _ => None,
                },
                None => None,
            }
        };
        let Some((settings, start_state, steps, need_full_adam)) = extracted else {
            return Ok(());
        };

        if need_full_adam {
            let full_adam = self.model.new_full_adam_state();
            if let Some(online) = self.online.as_mut() {
                online.full_adam = Some(full_adam);
            }
        }

        let model = Arc::make_mut(&mut self.model);
        let Some(online) = self.online.as_mut() else {
            return Ok(());
        };
        model.online_train_segment_tbptt(
            &mut self.scratch,
            &start_state,
            &steps,
            settings.scope,
            settings.optimizer,
            settings.lr,
            settings.clip,
            TBPTT_REPLAY_CHUNK,
            &mut online.adam_t,
            online.full_adam.as_mut(),
            if settings.scope.bias {
                Some(online.out_bias.as_mut_slice())
            } else {
                None
            },
            if settings.scope.bias {
                online.adam_m.as_deref_mut()
            } else {
                None
            },
            if settings.scope.bias {
                online.adam_v.as_deref_mut()
            } else {
                None
            },
            &mut self.state,
        )?;
        let bias = self.online.as_ref().map(|o| o.out_bias.as_slice());
        Self::logits_to_pdf(self.scratch.logits(), bias, &mut self.pdf_buffer);
        Ok(())
    }

    fn enqueue_full_tbptt_step(
        &mut self,
        settings: FullTrainSettings,
        target_symbol: u8,
    ) -> Result<()> {
        let should_flush = {
            let Some(online) = self.online.as_mut() else {
                return Ok(());
            };
            let Some(tbptt) = online.full_tbptt.as_mut() else {
                bail!("rwkv full-parameter online training requires trace-enabled tbptt runtime");
            };
            tbptt.settings.is_some_and(|prev| {
                !prev.matches(
                    settings.optimizer,
                    settings.lr,
                    settings.scope,
                    settings.bptt,
                    settings.clip,
                )
            }) && !tbptt.steps.is_empty()
        };
        if should_flush {
            self.flush_full_tbptt_segment()?;
        }

        let flush_now = {
            let Some(online) = self.online.as_mut() else {
                return Ok(());
            };
            let Some(tbptt) = online.full_tbptt.as_mut() else {
                bail!("rwkv full-parameter online training requires trace-enabled tbptt runtime");
            };
            let Some(input_token) = tbptt.pending_input_token.take() else {
                return Ok(());
            };
            let input_pre_state = tbptt
                .pending_input_pre_state
                .take()
                .ok_or_else(|| anyhow::anyhow!("rwkv full tbptt pending pre-state is missing"))?;
            if tbptt.steps.is_empty() {
                tbptt.segment_start_state = Some(input_pre_state);
            }
            tbptt.settings = Some(settings);
            tbptt.steps.push((input_token, target_symbol));
            tbptt.steps.len() >= settings.bptt.max(1)
        };
        if flush_now {
            self.flush_full_tbptt_segment()?;
        }
        Ok(())
    }

    /// Begin a policy stream with optional total symbol count.
    pub fn begin_online_policy_stream(&mut self, total_symbols: Option<u64>) -> Result<()> {
        self.finish_online_policy_stream()?;
        self.prepare_policy_stream(total_symbols)
    }

    /// Flush any pending TBPTT segment while preserving the current predictive state.
    pub fn finish_online_policy_stream(&mut self) -> Result<()> {
        self.flush_full_tbptt_segment()
    }

    /// Reset hidden state and TBPTT bookkeeping for a fresh stream.
    pub fn restart_online_policy_stream(&mut self, total_symbols: Option<u64>) -> Result<()> {
        self.finish_online_policy_stream()?;
        self.state.reset();
        self.clear_online_training_buffers();
        self.prepare_policy_stream(total_symbols)
    }

    /// Reset state and prime the first predictive distribution.
    pub fn reset_and_prime(&mut self) {
        self.state.reset();
        self.clear_online_training_buffers();
        self.refresh_current_pdf(0);
    }

    /// Capture runtime state for later restoration.
    pub fn snapshot_runtime(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            model: self.model.clone(),
            scratch: self.scratch.clone(),
            state: self.state.clone(),
            pdf_buffer: self.pdf_buffer.clone(),
            online: self.online.clone(),
        }
    }

    /// Restore previously captured runtime state.
    pub fn restore_runtime(&mut self, snapshot: &RuntimeSnapshot) {
        self.model = snapshot.model.clone();
        self.scratch = snapshot.scratch.clone();
        self.state = snapshot.state.clone();
        self.pdf_buffer.clone_from(&snapshot.pdf_buffer);
        self.online = snapshot.online.clone();
    }

    /// Absorb a sequence of byte slices as conditioning context.
    pub fn absorb_chain(&mut self, parts: &[&[u8]]) -> Result<()> {
        let total = parts
            .iter()
            .fold(0u64, |acc, part| acc.saturating_add(part.len() as u64));
        self.fit_chain(parts, Some(total))
    }

    /// Score bytes from the current predictive state.
    pub fn cross_entropy_from_current(&mut self, data: &[u8]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        self.begin_online_policy_stream(Some(data.len() as u64))?;
        let mut total_bits = 0.0f64;
        for &byte in data {
            total_bits -= self.pdf_buffer[byte as usize].log2();
            self.observe_symbol_from_current_pdf(byte)?;
        }
        self.finish_online_policy_stream()?;
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
            return self.cross_entropy(data);
        }
        self.finish_online_policy_stream()?;
        self.reset_and_prime();
        let fit_total = fit_parts
            .iter()
            .fold(0u64, |acc, part| acc.saturating_add(part.len() as u64));
        self.fit_chain(fit_parts, Some(fit_total))?;
        self.reset_and_prime();

        let mut total_bits = 0.0f64;
        for &byte in data {
            total_bits -= self.pdf_buffer[byte as usize].max(1e-300).log2();
            self.advance_inference_only(byte);
        }
        Ok(total_bits / (data.len() as f64))
    }

    /// Returns `true` when the compressor is in online-adaptation mode.
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

    /// Number of tokens processed by the online updater.
    pub fn tokens_processed(&self) -> u64 {
        self.online.as_ref().map_or(0, |s| s.tokens_processed)
    }

    /// Canonical method string for online mode, if enabled.
    pub fn online_method_string(&self) -> Option<&str> {
        self.online.as_ref().map(|s| s.canonical_method.as_str())
    }

    /// Get the vocabulary size (should always be 256 for byte-level).
    pub fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    /// Apply optional online bias to logits and emit normalized PDF.
    pub fn online_apply_logits_bias(&self, logits: &[f32], pdf_out: &mut [f64]) {
        let bias = self.online.as_ref().map(|s| s.out_bias.as_slice());
        Self::logits_to_pdf(logits, bias, pdf_out);
    }

    /// Convert logits (and optional bias) to a normalized PDF.
    pub fn logits_to_pdf(logits: &[f32], bias: Option<&[f32]>, pdf_out: &mut [f64]) {
        softmax_pdf_floor_with_bias(logits, bias, pdf_out);
    }

    #[inline]
    /// Forward one token and emit the resulting (optionally biased) PDF.
    pub fn forward_to_pdf(&mut self, token: u32, pdf_out: &mut [f64]) {
        self.forward_with_online_record(token);
        let bias = self.online.as_ref().map(|o| o.out_bias.as_slice());
        Self::logits_to_pdf(self.scratch.logits(), bias, pdf_out);
    }

    #[inline]
    /// Refresh the internal cached PDF buffer from `token`.
    pub fn forward_to_internal_pdf(&mut self, token: u32) {
        self.refresh_current_pdf(token);
    }

    #[inline]
    /// Copy the internal cached PDF into `pdf_out` (length must match vocab size).
    pub fn copy_current_pdf_to(&self, pdf_out: &mut [f64]) {
        assert_eq!(
            pdf_out.len(),
            self.pdf_buffer.len(),
            "rwkv pdf output length mismatch"
        );
        pdf_out.copy_from_slice(&self.pdf_buffer);
    }

    /// Snapshot current online output bias, if online mode is active.
    pub fn online_bias_snapshot(&self) -> Option<Vec<f32>> {
        self.online.as_ref().map(|o| o.out_bias.clone())
    }

    #[inline]
    /// Borrow online output bias vector when online mode is active.
    pub fn online_bias_slice(&self) -> Option<&[f32]> {
        self.online.as_ref().map(|o| o.out_bias.as_slice())
    }

    #[inline]
    fn refresh_current_pdf(&mut self, token: u32) {
        self.forward_with_online_record(token);
        let bias = self.online.as_ref().map(|o| o.out_bias.as_slice());
        Self::logits_to_pdf(self.scratch.logits(), bias, &mut self.pdf_buffer);
    }

    fn fit_chain(&mut self, parts: &[&[u8]], total_symbols: Option<u64>) -> Result<()> {
        self.begin_online_policy_stream(total_symbols)?;
        for part in parts {
            for &byte in *part {
                self.observe_symbol_from_current_pdf(byte)?;
            }
        }
        self.finish_online_policy_stream()?;
        Ok(())
    }

    #[inline]
    fn advance_inference_only(&mut self, symbol: u8) {
        self.refresh_current_pdf(symbol as u32);
    }

    fn resolve_online_train_action(
        online: &mut OnlineRuntime,
    ) -> Result<(OptimizerKind, f32, u64, rwkv7::TrainScopeMask, usize, f32)> {
        let mut optimizer = match online.cfg.train_mode {
            OnlineTrainMode::None => OptimizerKind::Sgd,
            OnlineTrainMode::Sgd => OptimizerKind::Sgd,
            OnlineTrainMode::Adam => OptimizerKind::Adam,
        };
        let mut lr = online.cfg.lr.max(0.0);
        let mut stride = online.cfg.stride.max(1) as u64;
        let mut scope = rwkv7::TrainScopeMask::default();
        let default_train = !matches!(online.cfg.train_mode, OnlineTrainMode::None);
        scope.head = default_train;
        scope.bias = default_train;
        let mut bptt = 1usize;
        let mut clip = 0.0f32;

        if let Some(action) = online.next_policy_action()? {
            match action {
                PolicyAction::Infer => {
                    scope = rwkv7::TrainScopeMask::default();
                }
                PolicyAction::Train(train) => {
                    optimizer = train.optimizer;
                    lr = train.hyper.lr.max(0.0);
                    stride = train.hyper.stride.max(1) as u64;
                    bptt = train.hyper.bptt.max(1);
                    clip = train.hyper.clip.max(0.0);
                    scope = scope_from_train_action(&train);
                }
            }
        }

        Ok((optimizer, lr, stride, scope, bptt, clip))
    }

    /// Apply one online update using externally supplied predictive PDF.
    pub fn online_update_from_pdf(&mut self, symbol: u8, pdf: &[f64]) -> Result<()> {
        self.online_update_with_pdf(symbol, pdf)
    }

    #[inline]
    /// Update online state from `pdf`, then advance model state with `symbol`.
    pub fn observe_symbol_from_pdf(&mut self, symbol: u8, pdf: &[f64]) -> Result<()> {
        self.online_update_with_pdf(symbol, pdf)?;
        self.refresh_current_pdf(symbol as u32);
        Ok(())
    }

    fn online_update_with_pdf(&mut self, symbol: u8, pdf: &[f64]) -> Result<()> {
        let (optimizer, lr, stride_hit, scope, bptt, clip) = {
            let Some(online) = self.online.as_mut() else {
                return Ok(());
            };
            online.tokens_processed = online.tokens_processed.saturating_add(1);
            let (optimizer, lr, stride, scope, bptt, clip) =
                Self::resolve_online_train_action(online)?;
            let mut stride_hit = false;
            if scope.trains_any_params() {
                online.policy_train_steps = online.policy_train_steps.saturating_add(1);
                stride_hit = stride <= 1 || (online.policy_train_steps % stride) == 0;
            }
            (optimizer, lr, stride_hit, scope, bptt, clip)
        };

        if !scope.trains_any_params() || !stride_hit || lr == 0.0 {
            self.flush_full_tbptt_segment()?;
            if let Some(online) = self.online.as_mut()
                && let Some(tbptt) = online.full_tbptt.as_mut()
            {
                tbptt.pending_input_token = None;
                tbptt.pending_input_pre_state = None;
            }
            return Ok(());
        }

        if matches!(optimizer, OptimizerKind::Adam)
            && let Some(online) = self.online.as_mut()
            && scope.bias
            && (online.adam_m.is_none() || online.adam_v.is_none())
        {
            online.adam_m = Some(vec![0.0; online.out_bias.len()]);
            online.adam_v = Some(vec![0.0; online.out_bias.len()]);
        }

        if !scope.trains_non_head_params() {
            let hidden = self.scratch.lm_head_input().to_vec();
            let pdf_snapshot = pdf.to_vec();
            self.flush_full_tbptt_segment()?;
            let Some(online) = self.online.as_mut() else {
                return Ok(());
            };
            if let Some(tbptt) = online.full_tbptt.as_mut() {
                tbptt.pending_input_token = None;
                tbptt.pending_input_pre_state = None;
            }
            let model = Arc::make_mut(&mut self.model);
            apply_online_lm_head_update(
                model,
                online,
                &hidden,
                symbol,
                &pdf_snapshot,
                lr,
                optimizer,
                scope.head,
                scope.bias,
                clip,
            );
            return Ok(());
        }

        let settings = FullTrainSettings {
            optimizer,
            lr,
            scope,
            bptt: Self::effective_full_bptt(scope, bptt),
            clip,
        };
        self.enqueue_full_tbptt_step(settings, symbol)
    }

    fn online_update_from_current_pdf(&mut self, symbol: u8) -> Result<()> {
        let pdf_snapshot = self.pdf_buffer.clone();
        self.online_update_with_pdf(symbol, &pdf_snapshot)
    }

    #[inline]
    /// Update online state using current internal PDF, then consume `symbol`.
    pub fn observe_symbol_from_current_pdf(&mut self, symbol: u8) -> Result<()> {
        self.online_update_from_current_pdf(symbol)?;
        self.refresh_current_pdf(symbol as u32);
        Ok(())
    }

    /// Export model weights and JSON sidecar metadata.
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
                    "decay_rank": online.cfg.decay_rank,
                    "a_rank": online.cfg.a_rank,
                    "v_rank": online.cfg.v_rank,
                    "g_rank": online.cfg.g_rank,
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
            let canonical_method = canonical_method_string(&MethodSpec::File {
                path: model_path.to_path_buf(),
                policy: None,
            })?;
            json!({
                "version": 1,
                "method": canonical_method,
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
        let default_method = canonical_method_string(&MethodSpec::File {
            path: model_path.to_path_buf(),
            policy: None,
        })?;
        let method = v
            .get("method")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string())
            .unwrap_or(default_method);
        let has_full_adam = v
            .get("has_full_adam")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let policy = v
            .get("policy")
            .and_then(|p| p.as_str())
            .and_then(|s| llm_policy::parse_policy_segment(s, RWKV_TRAIN_SCOPES).ok());
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
                full_tbptt: needs_full_trace.then(|| FullTbpttRuntime {
                    pending_input_token: None,
                    pending_input_pre_state: None,
                    segment_start_state: None,
                    steps: Vec::new(),
                    settings: None,
                }),
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

    /// Compress into an arbitrary writer.
    pub fn compress_into<W: Write>(
        &mut self,
        data: &[u8],
        coder: CoderType,
        w: &mut W,
    ) -> Result<()> {
        self.restart_online_policy_stream(Some(data.len() as u64))?;

        let checksum = crc32(data);
        let header = Header::new(coder, data.len() as u64, checksum);
        header.write(w)?;

        match coder {
            CoderType::AC => self.compress_ac(data, w)?,
            CoderType::RANS => self.compress_rans(data, w)?,
        }

        Ok(())
    }

    /// Compress a chain of byte slices into an arbitrary writer.
    pub fn compress_chain_into<W: Write>(
        &mut self,
        parts: &[&[u8]],
        coder: CoderType,
        w: &mut W,
    ) -> Result<()> {
        let mut total_len: u64 = 0;
        let mut hasher = crc32fast::Hasher::new();
        for p in parts {
            total_len = total_len.saturating_add(p.len() as u64);
            hasher.update(p);
        }
        let checksum = hasher.finalize();
        self.restart_online_policy_stream(Some(total_len))?;

        let header = Header::new(coder, total_len, checksum);
        header.write(w)?;

        let it = parts.iter().flat_map(|p| p.iter().copied());
        match coder {
            CoderType::AC => self.compress_ac_iter(it, w)?,
            CoderType::RANS => self.compress_rans_iter(it, w)?,
        }

        Ok(())
    }

    /// Return compressed byte size without materializing output bytes.
    pub fn compress_size(&mut self, data: &[u8], coder: CoderType) -> Result<u64> {
        let mut w = CountingWriter::new();
        self.compress_into(data, coder, &mut w)?;
        Ok(w.bytes_written())
    }

    /// Return compressed byte size for chained inputs.
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
        self.refresh_current_pdf(0);

        for byte in data {
            quantize_pdf_to_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_ac,
                &mut self.ac_freq_buffer,
            );
            let sym = byte as usize;
            let c_lo = self.cdf_buffer_ac[sym] as u64;
            let c_hi = self.cdf_buffer_ac[sym + 1] as u64;
            encoder.encode_counts(c_lo, c_hi, CDF_TOTAL as u64)?;
            self.observe_symbol_from_current_pdf(byte)?;
        }

        let _ = encoder.finish()?;
        self.finish_online_policy_stream()?;
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
        self.refresh_current_pdf(0);

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
            self.observe_symbol_from_current_pdf(byte)?;
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

        self.finish_online_policy_stream()?;
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

        self.restart_online_policy_stream(Some(header.original_len))?;

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
        self.refresh_current_pdf(0);

        for _ in 0..original_len {
            quantize_pdf_to_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_ac,
                &mut self.ac_freq_buffer,
            );
            let sym = decoder.decode_symbol_counts(&self.cdf_buffer_ac, CDF_TOTAL)?;
            result.push(sym as u8);
            self.observe_symbol_from_current_pdf(sym as u8)?;
        }

        self.finish_online_policy_stream()?;
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
        let mut decoder = BlockedRansDecoder::new(blocks, original_len)?;
        let mut result = Vec::with_capacity(original_len);

        // Prime with null byte
        self.refresh_current_pdf(0);

        for _ in 0..original_len {
            quantize_pdf_to_rans_cdf_with_buffer(
                &self.pdf_buffer,
                &mut self.cdf_buffer_rans,
                &mut self.rans_freq_buffer,
            );
            let sym = decoder.decode(&self.cdf_buffer_rans)?;
            result.push(sym as u8);
            self.observe_symbol_from_current_pdf(sym as u8)?;
        }

        self.finish_online_policy_stream()?;
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
        self.finish_online_policy_stream()?;
        self.reset_and_prime();
        self.cross_entropy_from_current(data)
    }

    /// Cross entropy conditioned on chained prefix slices.
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
        self.finish_online_policy_stream()?;
        self.reset_and_prime();
        self.fit_chain(prefix_parts, Some((prefix_len + data.len()) as u64))?;

        let mut total_bits = 0.0f64;
        for &byte in data {
            total_bits -= self.pdf_buffer[byte as usize].log2();
            self.observe_symbol_from_current_pdf(byte)?;
        }

        self.finish_online_policy_stream()?;
        Ok(total_bits / (data.len() as f64))
    }

    /// Cross entropy conditioned on a single prefix slice.
    pub fn cross_entropy_conditional(&mut self, prefix: &[u8], data: &[u8]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        self.finish_online_policy_stream()?;
        self.reset_and_prime();
        self.begin_online_policy_stream(Some((prefix.len() + data.len()) as u64))?;

        // Condition on prefix (update state, no scoring)
        for &byte in prefix {
            self.observe_symbol_from_current_pdf(byte)?;
        }

        let mut total_bits = 0.0f64;
        for &byte in data {
            total_bits -= self.pdf_buffer[byte as usize].log2();
            self.observe_symbol_from_current_pdf(byte)?;
        }

        self.finish_online_policy_stream()?;
        Ok(total_bits / (data.len() as f64))
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

        self.restart_online_policy_stream(Some((2 * n) as u64))?;

        self.refresh_current_pdf(0);

        let mut total_bits = 0.0f64;
        for i in 0..n {
            let a = if swap { y[i] } else { x[i] };
            let b = if swap { x[i] } else { y[i] };

            let pa = self.pdf_buffer[a as usize];
            total_bits -= pa.log2();
            self.observe_symbol_from_current_pdf(a)?;

            let pb = self.pdf_buffer[b as usize];
            total_bits -= pb.log2();
            self.observe_symbol_from_current_pdf(b)?;
        }

        self.finish_online_policy_stream()?;
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
            MethodSpec::File { path: got, .. } => assert_eq!(got, p),
            _ => panic!("expected file method"),
        }

        match parse_method_spec(
            "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=1,train=none,lr=0.01,stride=2;policy:schedule=0..100:infer",
        )
        .unwrap()
        {
            MethodSpec::Online { cfg, .. } => {
                assert_eq!(cfg.hidden, 64);
                assert_eq!(cfg.layers, 1);
                assert_eq!(cfg.seed, 1);
                assert_eq!(cfg.stride, 2);
            }
            _ => panic!("expected cfg method"),
        }

        match parse_method_spec("64,64,1,0,7,0.01,2;policy:schedule=0..100:infer").unwrap() {
            MethodSpec::Online { cfg, .. } => {
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
            MethodSpec::File { path: got, .. } => assert_eq!(got, p),
            _ => panic!("expected file method"),
        }

        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn parse_method_spec_accepts_raw_semicolons_in_file_paths() {
        match parse_method_spec("file:/tmp/rwkv;v1.safetensors").expect("file path parse") {
            MethodSpec::File { path, policy } => {
                assert_eq!(path, PathBuf::from("/tmp/rwkv;v1.safetensors"));
                assert!(policy.is_none());
            }
            _ => panic!("expected file method"),
        }
    }

    #[test]
    fn canonical_method_string_escapes_delimiter_bearing_file_paths() {
        let method = canonical_method_string(&MethodSpec::File {
            path: PathBuf::from("/tmp/rwkv;policy:model%v1.safetensors"),
            policy: None,
        })
        .expect("delimiter-bearing file path should canonicalize");
        assert_eq!(method, "file:/tmp/rwkv%3Bpolicy:model%25v1.safetensors");

        match parse_method_spec(&method).expect("canonical method should parse") {
            MethodSpec::File { path, policy } => {
                assert_eq!(path, PathBuf::from("/tmp/rwkv;policy:model%v1.safetensors"));
                assert!(policy.is_none());
            }
            _ => panic!("expected file method"),
        }
    }

    #[test]
    fn test_parse_method_spec_rejects_unknown_cfg_key() {
        let err =
            parse_method_spec("cfg:hidden=64,wat=1;policy:schedule=0..100:infer").unwrap_err();
        assert!(format!("{err:#}").contains("unknown rwkv cfg key"));
    }

    #[test]
    fn test_parse_method_spec_accepts_cfg_without_policy() {
        let spec = parse_method_spec("cfg:hidden=64,layers=1,intermediate=64").unwrap();
        match spec {
            MethodSpec::Online { cfg, policy } => {
                assert_eq!(cfg.hidden, 64);
                assert_eq!(cfg.layers, 1);
                assert_eq!(cfg.intermediate, 64);
                assert!(policy.is_none());
            }
            _ => panic!("expected cfg method"),
        }
    }

    #[test]
    fn test_canonical_method_omits_policy_when_absent() {
        let c = Compressor::new_from_method("cfg:hidden=64,layers=1,intermediate=64").unwrap();
        assert_eq!(
            c.online_method_string(),
            Some(
                "cfg:hidden=64,layers=1,intermediate=64,decay_rank=32,a_rank=32,v_rank=32,g_rank=64,seed=0,train=none,lr=0.001,stride=1"
            )
        );
    }

    #[test]
    fn test_online_export_reload_roundtrip() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=7,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.9)";
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
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=9,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.9)";
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

    #[test]
    fn test_runtime_snapshot_restores_non_head_training_state() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=15,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();
        c.reset_and_prime();
        c.absorb_chain(&[b"prior context".as_slice()]).unwrap();
        let snap = c.snapshot_runtime();

        let _ = c
            .cross_entropy_from_current(b"mutate model before restore")
            .unwrap();

        c.restore_runtime(&snap);
        let score_a = c
            .cross_entropy_from_current(b"query after restore")
            .unwrap();

        c.restore_runtime(&snap);
        let score_b = c
            .cross_entropy_from_current(b"query after restore")
            .unwrap();

        assert!((score_a - score_b).abs() < 1e-12);
    }

    #[test]
    fn test_online_training_updates_lm_head_weights() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=5,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();
        c.reset_and_prime();
        let before = c.model.lm_head_weights()[0..64].to_vec();
        let _ = c
            .cross_entropy_from_current(b"online rwkv weight update")
            .unwrap();
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
    fn test_cross_entropy_from_current_keeps_unique_model_arc() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=21,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();
        c.reset_and_prime();

        assert_eq!(Arc::strong_count(&c.model), 1);
        let before = Arc::as_ptr(&c.model);
        let _ = c.cross_entropy_from_current(b"arc uniqueness").unwrap();
        let after = Arc::as_ptr(&c.model);

        assert_eq!(Arc::strong_count(&c.model), 1);
        assert_eq!(before, after);
    }

    #[test]
    fn test_online_training_non_head_scope_updates_model_params() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=13,train=sgd,lr=0.005,stride=1;policy:schedule=0..100:train(scope=attn,opt=sgd,lr=0.005,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();

        let head_before = c.model.lm_head_weights()[0..64].to_vec();
        let before_path = temp_path("rwkv_non_head_before", "safetensors");
        let after_path = temp_path("rwkv_non_head_after", "safetensors");
        c.model.save_safetensors(&before_path).unwrap();

        c.reset_and_prime();
        let _ = c
            .cross_entropy_from_current(b"rwkv non head online update")
            .unwrap();
        c.model.save_safetensors(&after_path).unwrap();

        let head_after = &c.model.lm_head_weights()[0..64];
        for idx in 0..head_before.len() {
            assert_eq!(
                head_before[idx].to_bits(),
                head_after[idx].to_bits(),
                "lm-head changed under scope=attn at index {idx}"
            );
        }

        let before_bytes = std::fs::read(&before_path).unwrap();
        let after_bytes = std::fs::read(&after_path).unwrap();
        assert_ne!(
            before_bytes, after_bytes,
            "expected non-head params to change"
        );

        std::fs::remove_file(&before_path).ok();
        std::fs::remove_file(&after_path).ok();
    }

    #[test]
    fn test_online_training_scope_all_bptt_gt_one_supported() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=23,train=adam,lr=0.001,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=2,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();
        let before_path = temp_path("rwkv_tbptt_before", "safetensors");
        let after_path = temp_path("rwkv_tbptt_after", "safetensors");
        c.model.save_safetensors(&before_path).unwrap();
        c.reset_and_prime();
        let score = c.cross_entropy_from_current(b"abcdef").unwrap();
        assert!(score.is_finite());
        c.model.save_safetensors(&after_path).unwrap();
        let before_bytes = std::fs::read(&before_path).unwrap();
        let after_bytes = std::fs::read(&after_path).unwrap();
        assert_ne!(
            before_bytes, after_bytes,
            "expected tbptt training to update params"
        );
        std::fs::remove_file(&before_path).ok();
        std::fs::remove_file(&after_path).ok();
    }

    #[test]
    fn test_online_training_scope_all_bptt_one_uses_fast_default_window() {
        let method_bptt1 = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=27,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let method_bptt8 = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=27,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=8,clip=0,momentum=0.9)";
        let data = b"abcdefghij";

        let mut c1 = Compressor::new_from_method(method_bptt1).unwrap();
        let mut c8 = Compressor::new_from_method(method_bptt8).unwrap();

        let score1 = c1.cross_entropy(data).unwrap();
        let score8 = c8.cross_entropy(data).unwrap();
        assert!((score1 - score8).abs() < 1e-12);

        let bptt1_path = temp_path("rwkv_bptt1_fast_default", "safetensors");
        let bptt8_path = temp_path("rwkv_bptt8_fast_default", "safetensors");
        c1.model.save_safetensors(&bptt1_path).unwrap();
        c8.model.save_safetensors(&bptt8_path).unwrap();
        let bptt1_bytes = std::fs::read(&bptt1_path).unwrap();
        let bptt8_bytes = std::fs::read(&bptt8_path).unwrap();
        assert_eq!(bptt1_bytes, bptt8_bytes);
        std::fs::remove_file(&bptt1_path).ok();
        std::fs::remove_file(&bptt8_path).ok();
    }

    #[test]
    fn test_online_training_full_tbptt_updates_first_symbol_after_priming() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=33,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=8,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();
        let before_path = temp_path("rwkv_first_symbol_before", "safetensors");
        let after_path = temp_path("rwkv_first_symbol_after", "safetensors");
        c.model.save_safetensors(&before_path).unwrap();

        c.reset_and_prime();
        let score = c.cross_entropy_from_current(b"a").unwrap();
        assert!(score.is_finite());
        c.model.save_safetensors(&after_path).unwrap();

        let before_bytes = std::fs::read(&before_path).unwrap();
        let after_bytes = std::fs::read(&after_path).unwrap();
        assert_ne!(
            before_bytes, after_bytes,
            "expected the first symbol after priming to update params"
        );
        std::fs::remove_file(&before_path).ok();
        std::fs::remove_file(&after_path).ok();
    }

    #[test]
    fn test_online_export_reload_roundtrip_preserves_full_adam_resume() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let data = b"rwkv full-adam export/load deterministic continuation sample";

        let mut c1 = Compressor::new_from_method(method).unwrap();
        let _ = c1.compress(data, CoderType::AC).unwrap();

        let model_path = temp_path("rwkv_full_adam_export", "safetensors");
        let opt_path = optimizer_sidecar_path(&model_path);
        c1.export_online(&model_path).unwrap();
        assert!(
            opt_path.exists(),
            "expected optimizer sidecar to be exported"
        );
        let out1_after_export = c1.compress(data, CoderType::AC).unwrap();

        let mut c2 = Compressor::new(&model_path).unwrap();
        let out2 = c2.compress(data, CoderType::AC).unwrap();
        assert_eq!(out1_after_export, out2);

        std::fs::remove_file(&model_path).ok();
        std::fs::remove_file(model_path.with_extension("json")).ok();
        std::fs::remove_file(&opt_path).ok();
    }

    #[test]
    fn test_online_export_reload_missing_full_adam_sidecar_fails() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=41,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();
        let _ = c
            .compress(b"rwkv strict optimizer-sidecar requirement", CoderType::AC)
            .unwrap();

        let model_path = temp_path("rwkv_full_adam_missing_sidecar", "safetensors");
        let opt_path = optimizer_sidecar_path(&model_path);
        c.export_online(&model_path).unwrap();
        std::fs::remove_file(&opt_path).unwrap();

        let err = match Compressor::new(&model_path) {
            Ok(_) => panic!("expected missing optimizer sidecar to fail"),
            Err(err) => err,
        };
        assert!(format!("{err:#}").contains("missing optimizer sidecar"));

        std::fs::remove_file(&model_path).ok();
        std::fs::remove_file(model_path.with_extension("json")).ok();
    }

    #[test]
    fn test_clone_preserves_non_head_training_trace() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=43,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=1,clip=0,momentum=0.9)";
        let mut c = Compressor::new_from_method(method).unwrap();
        c.reset_and_prime();
        c.absorb_chain(&[b"clone trace prefix".as_slice()]).unwrap();

        let mut cloned = c.clone();
        let score = cloned
            .cross_entropy_from_current(b"clone trace query")
            .unwrap();
        assert!(score.is_finite());
    }
}
