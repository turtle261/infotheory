//! RWKV7 model implementation with portable SIMD-optimized inference.
//!
//! This is a high-performance implementation built on `wide` kernels.
//! Single-token inference is the primary use case (streaming compression).

use crate::backends::llm_policy::OptimizerKind;
use anyhow::{Context, Result, bail};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use wide::f32x8;

use super::kernel;
use super::profiling::{NullProfiler, ProfilerSink};
use super::tensor::Tensor1D;
use super::weights::Weights;

/// Model configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Vocabulary size (byte-level models use 256).
    pub vocab_size: usize,
    /// Hidden channel width `C`.
    pub hidden_size: usize,
    /// Number of RWKV blocks.
    pub num_layers: usize,
    /// Number of attention heads `H`.
    pub num_heads: usize,
    /// Per-head channel width `N` (currently fixed to 64 in kernels).
    pub head_dim: usize,
    /// Feed-forward intermediate width.
    pub intermediate_size: usize,
    /// Epsilon for layer normalization.
    pub layer_norm_eps: f32,
    /// Epsilon for group normalization (`64e-5` in reference).
    pub group_norm_eps: f32,

    /// Low-rank width for decay projection.
    pub decay_low_rank: usize, // w_lora
    /// Low-rank width for `a` projection.
    pub a_low_rank: usize,
    /// Low-rank width for `v` projection.
    pub v_low_rank: usize,
    /// Low-rank width for `g` projection.
    pub g_low_rank: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 256,
            num_layers: 12,
            num_heads: 4, // 256 / 64
            head_dim: 64,
            intermediate_size: 1024,
            layer_norm_eps: 1e-5,
            group_norm_eps: 64e-5,
            decay_low_rank: 32,
            a_low_rank: 32,
            v_low_rank: 32,
            g_low_rank: 64,
        }
    }
}

impl Config {
    /// Validate configuration invariants required by current kernels.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            bail!("rwkv7 vocab_size must be > 0");
        }
        if self.head_dim != 64 {
            bail!("rwkv7 head_dim must be 64 for current kernels");
        }
        if self.hidden_size != self.num_heads * self.head_dim {
            bail!(
                "rwkv7 hidden_size must equal num_heads * head_dim ({} != {} * {})",
                self.hidden_size,
                self.num_heads,
                self.head_dim
            );
        }
        if self.num_layers == 0 {
            bail!("rwkv7 num_layers must be > 0");
        }
        if self.intermediate_size == 0 {
            bail!("rwkv7 intermediate_size must be > 0");
        }
        Ok(())
    }
}

/// Per-layer state for RWKV7.
#[derive(Clone)]
pub struct LayerState {
    /// Previous token embedding for attention time-shift (hidden_size,)
    pub att_x_prev: Tensor1D,
    /// Attention state matrix (num_heads, head_dim, head_dim) = (H, N, N)
    pub att_state: Tensor1D, // Flat for SIMD access
    /// Previous token embedding for FFN time-shift (hidden_size,)
    pub ffn_x_prev: Tensor1D,
}

impl LayerState {
    fn new(cfg: &Config) -> Self {
        let state_size = cfg.num_heads * cfg.head_dim * cfg.head_dim;
        Self {
            att_x_prev: Tensor1D::zeros(cfg.hidden_size),
            att_state: Tensor1D::zeros(state_size),
            ffn_x_prev: Tensor1D::zeros(cfg.hidden_size),
        }
    }
}

/// Full model state.
#[derive(Clone)]
pub struct State {
    /// Per-layer recurrent state.
    pub layers: Vec<LayerState>,
    /// First layer's value output (for residual connection) - pre-allocated
    pub v_first: Tensor1D,
    /// Flag to indicate if v_first has been set
    pub v_first_set: bool,
}

impl State {
    /// Allocate a zero-initialized recurrent state for a model configuration.
    pub fn new(cfg: &Config) -> Self {
        Self {
            layers: (0..cfg.num_layers).map(|_| LayerState::new(cfg)).collect(),
            v_first: Tensor1D::zeros(cfg.hidden_size),
            v_first_set: false,
        }
    }

    /// Reset recurrent buffers to their initial (all-zero) state.
    pub fn reset(&mut self) {
        self.v_first_set = false;
        self.v_first.zero();
        for layer in &mut self.layers {
            layer.att_x_prev.zero();
            layer.att_state.zero();
            layer.ffn_x_prev.zero();
        }
    }
}

/// Weights for a single attention layer.
#[derive(Clone)]
struct AttentionWeights {
    // Token shift mixing factors
    x_r: Tensor1D,
    x_w: Tensor1D,
    x_k: Tensor1D,
    x_v: Tensor1D,
    x_a: Tensor1D,
    x_g: Tensor1D,

    // Packed r/k/v projections for parallel computation
    // Layout: [r_proj (C*C), k_proj (C*C), v_proj (C*C)]
    rkv_proj: Tensor1D,

    // Output projection (stored transposed for efficient gemv)
    o_proj: Tensor1D,

    // Low-rank W: w = tanh(x @ w1) @ w2 + w0
    w1: Tensor1D, // (C, D_w)
    w2: Tensor1D, // (D_w, C)
    w0: Tensor1D, // (C,)

    // Low-rank A: a = sigmoid(x @ a1 @ a2 + a0)
    a1: Tensor1D, // (C, D_a)
    a2: Tensor1D, // (D_a, C)
    a0: Tensor1D, // (C,)

    // Low-rank V (layers > 0): nu = sigmoid(x @ v1 @ v2 + v0)
    v1: Option<Tensor1D>, // (C, D_v)
    v2: Option<Tensor1D>, // (D_v, C)
    v0: Option<Tensor1D>, // (C,)

    // Low-rank G: g = sigmoid(x @ g1) @ g2
    g1: Tensor1D, // (C, D_g)
    g2: Tensor1D, // (D_g, C)

    // Key scaling
    k_k: Tensor1D, // (C,)
    k_a: Tensor1D, // (C,)
    r_k: Tensor1D, // (H, N)

    // Group norm for output
    g_norm_w: Tensor1D, // (C,)
    g_norm_b: Tensor1D, // (C,)
}

/// Weights for a single FFN layer.
#[derive(Clone)]
struct FfnWeights {
    x_k: Tensor1D,     // (C,) time shift mix
    key_w: Tensor1D,   // (C, I) -> relu(x @ W)^2
    value_w: Tensor1D, // (I, C)
}

/// Weights for a single block.
#[derive(Clone)]
struct BlockWeights {
    // Pre-norm (layer 0 only)
    pre_norm_w: Option<Tensor1D>,
    pre_norm_b: Option<Tensor1D>,

    // Attention norm
    attn_norm_w: Tensor1D,
    attn_norm_b: Tensor1D,

    // FFN norm
    ffn_norm_w: Tensor1D,
    ffn_norm_b: Tensor1D,

    attn: AttentionWeights,
    ffn: FfnWeights,
}

/// RWKV7 model.
#[derive(Clone)]
pub struct Model {
    cfg: Config,

    // Embeddings (vocab_size, hidden_size)
    embeddings: Tensor1D,

    // Output norm
    ln_out_w: Tensor1D,
    ln_out_b: Tensor1D,

    // LM head (vocab_size, hidden_size)
    lm_head: Tensor1D,

    // Layers
    blocks: Vec<BlockWeights>,
}

#[derive(Clone)]
struct AdamTensorState {
    m: Tensor1D,
    v: Tensor1D,
}

impl AdamTensorState {
    #[inline]
    fn new(len: usize) -> Self {
        Self {
            m: Tensor1D::zeros(len),
            v: Tensor1D::zeros(len),
        }
    }
}

#[derive(Clone)]
struct AttentionAdamState {
    x_r: AdamTensorState,
    x_w: AdamTensorState,
    x_k: AdamTensorState,
    x_v: AdamTensorState,
    x_a: AdamTensorState,
    x_g: AdamTensorState,
    rkv_proj: AdamTensorState,
    o_proj: AdamTensorState,
    w1: AdamTensorState,
    w2: AdamTensorState,
    w0: AdamTensorState,
    a1: AdamTensorState,
    a2: AdamTensorState,
    a0: AdamTensorState,
    v1: Option<AdamTensorState>,
    v2: Option<AdamTensorState>,
    v0: Option<AdamTensorState>,
    g1: AdamTensorState,
    g2: AdamTensorState,
    k_k: AdamTensorState,
    k_a: AdamTensorState,
    r_k: AdamTensorState,
    g_norm_w: AdamTensorState,
    g_norm_b: AdamTensorState,
}

#[derive(Clone)]
struct FfnAdamState {
    x_k: AdamTensorState,
    key_w: AdamTensorState,
    value_w: AdamTensorState,
}

#[derive(Clone)]
struct BlockAdamState {
    pre_norm_w: Option<AdamTensorState>,
    pre_norm_b: Option<AdamTensorState>,
    attn_norm_w: AdamTensorState,
    attn_norm_b: AdamTensorState,
    ffn_norm_w: AdamTensorState,
    ffn_norm_b: AdamTensorState,
    attn: AttentionAdamState,
    ffn: FfnAdamState,
}

#[derive(Clone)]
/// Adam moments for full-parameter RWKV online training.
pub struct FullAdamState {
    embeddings: AdamTensorState,
    ln_out_w: AdamTensorState,
    ln_out_b: AdamTensorState,
    lm_head: AdamTensorState,
    blocks: Vec<BlockAdamState>,
}

#[derive(Clone, Copy, Debug, Default)]
/// Train-scope mask for RWKV full-parameter online updates.
pub struct TrainScopeMask {
    pub embed: bool,
    pub pre_norm: bool,
    pub attn_norm: bool,
    pub ffn_norm: bool,
    pub attn: bool,
    pub ffn: bool,
    pub head: bool,
    pub bias: bool,
}

impl TrainScopeMask {
    #[inline]
    pub fn all() -> Self {
        Self {
            embed: true,
            pre_norm: true,
            attn_norm: true,
            ffn_norm: true,
            attn: true,
            ffn: true,
            head: true,
            bias: true,
        }
    }

    #[inline]
    pub fn trains_non_head_params(&self) -> bool {
        self.embed || self.pre_norm || self.attn_norm || self.ffn_norm || self.attn || self.ffn
    }

    #[inline]
    pub fn trains_any_params(&self) -> bool {
        self.trains_non_head_params() || self.head || self.bias
    }
}

struct AdamStep {
    lr: f32,
    clip: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
}

#[derive(Clone)]
struct LayerTrainTrace {
    x_in: Tensor1D,
    x_after_pre: Tensor1D,
    attn_norm: Tensor1D,
    att_x_prev_old: Tensor1D,
    ffn_x_prev_old: Tensor1D,
    att_state_old: Tensor1D,
    xr: Tensor1D,
    xw: Tensor1D,
    xk: Tensor1D,
    xv: Tensor1D,
    xa: Tensor1D,
    xg: Tensor1D,
    r: Tensor1D,
    k_pre: Tensor1D,
    k: Tensor1D,
    v_pre: Tensor1D,
    v: Tensor1D,
    nu: Tensor1D,
    w_hidden: Tensor1D,
    w_pre: Tensor1D,
    w_decay: Tensor1D,
    a_hidden: Tensor1D,
    a: Tensor1D,
    g_hidden: Tensor1D,
    g: Tensor1D,
    kk_pre: Tensor1D,
    kk: Tensor1D,
    y_wkv: Tensor1D,
    y_gn: Tensor1D,
    alpha: Tensor1D,
    y_head: Tensor1D,
    y_gate: Tensor1D,
    att_out: Tensor1D,
    x_after_attn: Tensor1D,
    ffn_norm: Tensor1D,
    ffn_xk: Tensor1D,
    ffn_pre: Tensor1D,
    ffn_k: Tensor1D,
    ffn_out: Tensor1D,
    x_out: Tensor1D,
    v_hidden: Tensor1D,
    uses_v_residual: bool,
}

impl LayerTrainTrace {
    fn new(cfg: &Config) -> Self {
        let c = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let state = cfg.num_heads * cfg.head_dim * cfg.head_dim;
        Self {
            x_in: Tensor1D::zeros(c),
            x_after_pre: Tensor1D::zeros(c),
            attn_norm: Tensor1D::zeros(c),
            att_x_prev_old: Tensor1D::zeros(c),
            ffn_x_prev_old: Tensor1D::zeros(c),
            att_state_old: Tensor1D::zeros(state),
            xr: Tensor1D::zeros(c),
            xw: Tensor1D::zeros(c),
            xk: Tensor1D::zeros(c),
            xv: Tensor1D::zeros(c),
            xa: Tensor1D::zeros(c),
            xg: Tensor1D::zeros(c),
            r: Tensor1D::zeros(c),
            k_pre: Tensor1D::zeros(c),
            k: Tensor1D::zeros(c),
            v_pre: Tensor1D::zeros(c),
            v: Tensor1D::zeros(c),
            nu: Tensor1D::zeros(c),
            w_hidden: Tensor1D::zeros(cfg.decay_low_rank),
            w_pre: Tensor1D::zeros(c),
            w_decay: Tensor1D::zeros(c),
            a_hidden: Tensor1D::zeros(cfg.a_low_rank),
            a: Tensor1D::zeros(c),
            g_hidden: Tensor1D::zeros(cfg.g_low_rank),
            g: Tensor1D::zeros(c),
            kk_pre: Tensor1D::zeros(c),
            kk: Tensor1D::zeros(c),
            y_wkv: Tensor1D::zeros(c),
            y_gn: Tensor1D::zeros(c),
            alpha: Tensor1D::zeros(cfg.num_heads),
            y_head: Tensor1D::zeros(c),
            y_gate: Tensor1D::zeros(c),
            att_out: Tensor1D::zeros(c),
            x_after_attn: Tensor1D::zeros(c),
            ffn_norm: Tensor1D::zeros(c),
            ffn_xk: Tensor1D::zeros(c),
            ffn_pre: Tensor1D::zeros(i),
            ffn_k: Tensor1D::zeros(i),
            ffn_out: Tensor1D::zeros(c),
            x_out: Tensor1D::zeros(c),
            v_hidden: Tensor1D::zeros(cfg.v_low_rank.max(1)),
            uses_v_residual: false,
        }
    }
}

/// Pre-allocated scratch buffers to avoid allocations in hot path.
#[derive(Clone)]
pub struct ScratchBuffers {
    x: Tensor1D,          // Current hidden state
    x_normed: Tensor1D,   // After layer norm
    xr: Tensor1D,         // Token-shifted for r
    xw: Tensor1D,         // Token-shifted for w
    xk: Tensor1D,         // Token-shifted for k
    xv: Tensor1D,         // Token-shifted for v
    xa: Tensor1D,         // Token-shifted for a
    xg: Tensor1D,         // Token-shifted for g
    r: Tensor1D,          // Receptance
    k: Tensor1D,          // Key
    v: Tensor1D,          // Value
    w_lora_tmp: Tensor1D, // Low-rank temp
    w_decay: Tensor1D,    // Decay factor
    a: Tensor1D,          // Gate a
    g: Tensor1D,          // Gate g
    kk: Tensor1D,         // Normalized key
    y: Tensor1D,          // WKV output
    att_out: Tensor1D,    // Attention output
    ffn_k: Tensor1D,      // FFN key
    ffn_out: Tensor1D,    // FFN output
    logits: Tensor1D,     // Output logits
    grad_x: Tensor1D,
    grad_x2: Tensor1D,
    grad_x3: Tensor1D,
    grad_x4: Tensor1D,
    grad_x5: Tensor1D,
    grad_x6: Tensor1D,
    grad_v_first: Tensor1D,
    grad_param: Tensor1D,
    grad_param2: Tensor1D,
    grad_saved: Tensor1D,
    grad_ffn: Tensor1D,
    grad_ffn2: Tensor1D,
    grad_low_rank: Tensor1D,
    grad_low_rank2: Tensor1D,
    grad_logits: Tensor1D,
    train_trace_layers: Vec<LayerTrainTrace>,
    train_token: usize,
    train_v_first: Tensor1D,
    train_trace_valid: bool,
    capture_train_trace: bool,
}

impl ScratchBuffers {
    /// Allocate reusable per-token scratch buffers sized for `cfg`.
    pub fn new(cfg: &Config) -> Self {
        let c = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let d_rank = cfg
            .decay_low_rank
            .max(cfg.a_low_rank)
            .max(cfg.v_low_rank)
            .max(cfg.g_low_rank)
            .max(64);
        let mut train_trace_layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            train_trace_layers.push(LayerTrainTrace::new(cfg));
        }

        Self {
            x: Tensor1D::zeros(c),
            x_normed: Tensor1D::zeros(c),
            xr: Tensor1D::zeros(c),
            xw: Tensor1D::zeros(c),
            xk: Tensor1D::zeros(c),
            xv: Tensor1D::zeros(c),
            xa: Tensor1D::zeros(c),
            xg: Tensor1D::zeros(c),
            r: Tensor1D::zeros(c),
            k: Tensor1D::zeros(c),
            v: Tensor1D::zeros(c),
            w_lora_tmp: Tensor1D::zeros(d_rank),
            w_decay: Tensor1D::zeros(c),
            a: Tensor1D::zeros(c),
            g: Tensor1D::zeros(c),
            kk: Tensor1D::zeros(c),
            y: Tensor1D::zeros(c),
            att_out: Tensor1D::zeros(c),
            ffn_k: Tensor1D::zeros(i),
            ffn_out: Tensor1D::zeros(c),
            logits: Tensor1D::zeros(v),
            grad_x: Tensor1D::zeros(c),
            grad_x2: Tensor1D::zeros(c),
            grad_x3: Tensor1D::zeros(c),
            grad_x4: Tensor1D::zeros(c),
            grad_x5: Tensor1D::zeros(c),
            grad_x6: Tensor1D::zeros(c),
            grad_v_first: Tensor1D::zeros(c),
            grad_param: Tensor1D::zeros(c),
            grad_param2: Tensor1D::zeros(c),
            grad_saved: Tensor1D::zeros(c),
            grad_ffn: Tensor1D::zeros(i),
            grad_ffn2: Tensor1D::zeros(i),
            grad_low_rank: Tensor1D::zeros(d_rank),
            grad_low_rank2: Tensor1D::zeros(d_rank),
            grad_logits: Tensor1D::zeros(v),
            train_trace_layers,
            train_token: 0,
            train_v_first: Tensor1D::zeros(c),
            train_trace_valid: false,
            capture_train_trace: false,
        }
    }

    /// Final normalized hidden state consumed by LM head.
    #[inline]
    pub fn lm_head_input(&self) -> &[f32] {
        self.x_normed.as_slice()
    }

    /// Restore LM-head input snapshot for reversible online updates.
    #[inline]
    pub fn set_lm_head_input(&mut self, value: &[f32]) {
        self.x_normed.as_mut_slice().copy_from_slice(value);
    }

    /// Enable or disable per-token training trace capture.
    #[inline]
    pub fn set_capture_train_trace(&mut self, enabled: bool) {
        self.capture_train_trace = enabled;
        if !enabled {
            self.train_trace_valid = false;
        }
    }

    /// Whether the current scratch contains a valid full-trace for the latest forward pass.
    #[inline]
    pub fn has_train_trace(&self) -> bool {
        self.train_trace_valid
    }
}

impl Model {
    fn tensor_from(weights: &Weights, name: &str) -> Result<Tensor1D> {
        Ok(Tensor1D::from_vec(weights.require(name)?.data().to_vec()))
    }

    fn optional_tensor_from(weights: &Weights, name: &str) -> Option<Tensor1D> {
        weights
            .get(name)
            .map(|tensor| Tensor1D::from_vec(tensor.data().to_vec()))
    }

    /// Load model from safetensors file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let weights = Weights::load(path.as_ref()).context("Failed to load model weights")?;

        // Infer config from weights
        let emb = weights.require("model.embeddings.weight")?;
        let vocab_size = emb.shape()[0];
        let hidden_size = emb.shape()[1];

        let num_heads = hidden_size / 64; // Assume head_dim=64
        let head_dim = 64;

        // Count layers by looking for layer weights
        let mut num_layers = 0;
        while weights
            .get(&format!("model.layers.{}.attn.r_proj.weight", num_layers))
            .is_some()
        {
            num_layers += 1;
        }

        // Get intermediate size from FFN
        let ffn_key = weights.require("model.layers.0.ffn.key.weight")?;
        let intermediate_size = ffn_key.shape()[0];

        // Get low-rank dimensions
        let w1 = weights.require("model.layers.0.attn.w_lora.lora.0.weight")?;
        let decay_low_rank = w1.shape()[0];

        let a1 = weights.require("model.layers.0.attn.a_lora.lora.0.weight")?;
        let a_low_rank = a1.shape()[0];

        let g1 = weights.require("model.layers.0.attn.g_lora.lora.0.weight")?;
        let g_low_rank = g1.shape()[0];

        // v_low_rank from layer 1 (layer 0 doesn't have it)
        let v_low_rank = if num_layers > 1 {
            if let Some(v1) = weights.get("model.layers.1.attn.v_lora.lora.0.weight") {
                v1.shape()[0]
            } else {
                32
            }
        } else {
            32
        };

        let cfg = Config {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            head_dim,
            intermediate_size,
            layer_norm_eps: 1e-5,
            group_norm_eps: 64e-5,
            decay_low_rank,
            a_low_rank,
            v_low_rank,
            g_low_rank,
        };

        // Load embeddings
        let embeddings = Self::tensor_from(&weights, "model.embeddings.weight")?;

        // Load output norm
        let ln_out_w = Self::tensor_from(&weights, "model.norm.weight")?;
        let ln_out_b = Self::tensor_from(&weights, "model.norm.bias")?;

        // Load LM head
        let lm_head = Self::tensor_from(&weights, "lm_head.weight")?;

        // Load blocks
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("model.layers.{}", i);

            // Pre-norm (layer 0 only)
            let (pre_norm_w, pre_norm_b) = if i == 0 {
                (
                    Some(Self::tensor_from(
                        &weights,
                        &format!("{}.pre_norm.weight", prefix),
                    )?),
                    Some(Self::tensor_from(
                        &weights,
                        &format!("{}.pre_norm.bias", prefix),
                    )?),
                )
            } else {
                (None, None)
            };

            // Norms
            let attn_norm_w = Self::tensor_from(&weights, &format!("{}.attn_norm.weight", prefix))?;
            let attn_norm_b = Self::tensor_from(&weights, &format!("{}.attn_norm.bias", prefix))?;
            let ffn_norm_w = Self::tensor_from(&weights, &format!("{}.ffn_norm.weight", prefix))?;
            let ffn_norm_b = Self::tensor_from(&weights, &format!("{}.ffn_norm.bias", prefix))?;

            // Attention weights
            // Load r/k/v projections and pack them contiguously
            let r_proj_data = weights
                .require(&format!("{}.attn.r_proj.weight", prefix))?
                .data();
            let k_proj_data = weights
                .require(&format!("{}.attn.k_proj.weight", prefix))?
                .data();
            let v_proj_data = weights
                .require(&format!("{}.attn.v_proj.weight", prefix))?
                .data();

            // Create packed RKV tensor: [r_proj, k_proj, v_proj]
            let proj_size = hidden_size * hidden_size;
            let mut rkv_proj = Tensor1D::zeros(3 * proj_size);
            rkv_proj.as_mut_slice()[0..proj_size].copy_from_slice(r_proj_data);
            rkv_proj.as_mut_slice()[proj_size..2 * proj_size].copy_from_slice(k_proj_data);
            rkv_proj.as_mut_slice()[2 * proj_size..3 * proj_size].copy_from_slice(v_proj_data);

            let attn = AttentionWeights {
                x_r: Self::tensor_from(&weights, &format!("{}.attn.x_r", prefix))?,
                x_w: Self::tensor_from(&weights, &format!("{}.attn.x_w", prefix))?,
                x_k: Self::tensor_from(&weights, &format!("{}.attn.x_k", prefix))?,
                x_v: Self::tensor_from(&weights, &format!("{}.attn.x_v", prefix))?,
                x_a: Self::tensor_from(&weights, &format!("{}.attn.x_a", prefix))?,
                x_g: Self::tensor_from(&weights, &format!("{}.attn.x_g", prefix))?,

                rkv_proj,
                o_proj: Self::tensor_from(&weights, &format!("{}.attn.o_proj.weight", prefix))?,

                w1: Self::tensor_from(&weights, &format!("{}.attn.w_lora.lora.0.weight", prefix))?,
                w2: Self::tensor_from(&weights, &format!("{}.attn.w_lora.lora.2.weight", prefix))?,
                w0: Self::tensor_from(&weights, &format!("{}.attn.w_lora.lora.2.bias", prefix))?,

                a1: Self::tensor_from(&weights, &format!("{}.attn.a_lora.lora.0.weight", prefix))?,
                a2: Self::tensor_from(&weights, &format!("{}.attn.a_lora.lora.2.weight", prefix))?,
                a0: Self::tensor_from(&weights, &format!("{}.attn.a_lora.lora.2.bias", prefix))?,

                v1: Self::optional_tensor_from(
                    &weights,
                    &format!("{}.attn.v_lora.lora.0.weight", prefix),
                ),
                v2: Self::optional_tensor_from(
                    &weights,
                    &format!("{}.attn.v_lora.lora.2.weight", prefix),
                ),
                v0: Self::optional_tensor_from(
                    &weights,
                    &format!("{}.attn.v_lora.lora.2.bias", prefix),
                ),

                g1: Self::tensor_from(&weights, &format!("{}.attn.g_lora.lora.0.weight", prefix))?,
                g2: Self::tensor_from(&weights, &format!("{}.attn.g_lora.lora.2.weight", prefix))?,

                k_k: Self::tensor_from(&weights, &format!("{}.attn.k_k", prefix))?,
                k_a: Self::tensor_from(&weights, &format!("{}.attn.k_a", prefix))?,
                r_k: Self::tensor_from(&weights, &format!("{}.attn.r_k", prefix))?,

                g_norm_w: Self::tensor_from(&weights, &format!("{}.attn.g_norm.weight", prefix))?,
                g_norm_b: Self::tensor_from(&weights, &format!("{}.attn.g_norm.bias", prefix))?,
            };

            // FFN weights
            let ffn = FfnWeights {
                x_k: Self::tensor_from(&weights, &format!("{}.ffn.x_k", prefix))?,
                key_w: Self::tensor_from(&weights, &format!("{}.ffn.key.weight", prefix))?,
                value_w: Self::tensor_from(&weights, &format!("{}.ffn.value.weight", prefix))?,
            };

            blocks.push(BlockWeights {
                pre_norm_w,
                pre_norm_b,
                attn_norm_w,
                attn_norm_b,
                ffn_norm_w,
                ffn_norm_b,
                attn,
                ffn,
            });
        }

        Ok(Self {
            cfg,
            embeddings,
            ln_out_w,
            ln_out_b,
            lm_head,
            blocks,
        })
    }

    /// Create a randomly initialized model for online-training workflows.
    pub fn new_random(cfg: Config, seed: u64) -> Result<Self> {
        cfg.validate()?;

        let mut rng = RwkvRng::new(seed);
        let c = cfg.hidden_size;
        let v = cfg.vocab_size;
        let i = cfg.intermediate_size;
        let d_w = cfg.decay_low_rank;
        let d_a = cfg.a_low_rank;
        let d_v = cfg.v_low_rank;
        let d_g = cfg.g_low_rank;

        let mut embeddings = Tensor1D::zeros(v * c);
        init_uniform(&mut embeddings, &mut rng, 0.02);

        let mut ln_out_w = Tensor1D::zeros(c);
        let mut ln_out_b = Tensor1D::zeros(c);
        init_const(&mut ln_out_w, 1.0);
        init_const(&mut ln_out_b, 0.0);

        let mut lm_head = Tensor1D::zeros(v * c);
        init_uniform(&mut lm_head, &mut rng, 0.02);

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let (pre_norm_w, pre_norm_b) = if layer_idx == 0 {
                let mut w = Tensor1D::zeros(c);
                let mut b = Tensor1D::zeros(c);
                init_const(&mut w, 1.0);
                init_const(&mut b, 0.0);
                (Some(w), Some(b))
            } else {
                (None, None)
            };

            let mut attn_norm_w = Tensor1D::zeros(c);
            let mut attn_norm_b = Tensor1D::zeros(c);
            init_const(&mut attn_norm_w, 1.0);
            init_const(&mut attn_norm_b, 0.0);

            let mut ffn_norm_w = Tensor1D::zeros(c);
            let mut ffn_norm_b = Tensor1D::zeros(c);
            init_const(&mut ffn_norm_w, 1.0);
            init_const(&mut ffn_norm_b, 0.0);

            let mut rkv_proj = Tensor1D::zeros(3 * c * c);
            init_uniform(&mut rkv_proj, &mut rng, 0.02);

            let mut o_proj = Tensor1D::zeros(c * c);
            init_uniform(&mut o_proj, &mut rng, 0.02);

            let mut w1 = Tensor1D::zeros(d_w * c);
            let mut w2 = Tensor1D::zeros(c * d_w);
            let mut w0 = Tensor1D::zeros(c);
            init_uniform(&mut w1, &mut rng, 0.02);
            init_uniform(&mut w2, &mut rng, 0.02);
            init_const(&mut w0, 0.0);

            let mut a1 = Tensor1D::zeros(d_a * c);
            let mut a2 = Tensor1D::zeros(c * d_a);
            let mut a0 = Tensor1D::zeros(c);
            init_uniform(&mut a1, &mut rng, 0.02);
            init_uniform(&mut a2, &mut rng, 0.02);
            init_const(&mut a0, 0.0);

            let (v1, v2, v0) = if layer_idx == 0 {
                (None, None, None)
            } else {
                let mut v1 = Tensor1D::zeros(d_v * c);
                let mut v2 = Tensor1D::zeros(c * d_v);
                let mut v0 = Tensor1D::zeros(c);
                init_uniform(&mut v1, &mut rng, 0.02);
                init_uniform(&mut v2, &mut rng, 0.02);
                init_const(&mut v0, 0.0);
                (Some(v1), Some(v2), Some(v0))
            };

            let mut g1 = Tensor1D::zeros(d_g * c);
            let mut g2 = Tensor1D::zeros(c * d_g);
            init_uniform(&mut g1, &mut rng, 0.02);
            init_uniform(&mut g2, &mut rng, 0.02);

            let mut x_r = Tensor1D::zeros(c);
            let mut x_w = Tensor1D::zeros(c);
            let mut x_k = Tensor1D::zeros(c);
            let mut x_v = Tensor1D::zeros(c);
            let mut x_a = Tensor1D::zeros(c);
            let mut x_g = Tensor1D::zeros(c);
            init_centered(&mut x_r, &mut rng, 0.5, 0.02);
            init_centered(&mut x_w, &mut rng, 0.5, 0.02);
            init_centered(&mut x_k, &mut rng, 0.5, 0.02);
            init_centered(&mut x_v, &mut rng, 0.5, 0.02);
            init_centered(&mut x_a, &mut rng, 0.5, 0.02);
            init_centered(&mut x_g, &mut rng, 0.5, 0.02);

            let mut k_k = Tensor1D::zeros(c);
            let mut k_a = Tensor1D::zeros(c);
            let mut r_k = Tensor1D::zeros(c);
            init_const(&mut k_k, 1.0);
            init_const(&mut k_a, 1.0);
            init_const(&mut r_k, 1.0);

            let mut g_norm_w = Tensor1D::zeros(c);
            let mut g_norm_b = Tensor1D::zeros(c);
            init_const(&mut g_norm_w, 1.0);
            init_const(&mut g_norm_b, 0.0);

            let attn = AttentionWeights {
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                rkv_proj,
                o_proj,
                w1,
                w2,
                w0,
                a1,
                a2,
                a0,
                v1,
                v2,
                v0,
                g1,
                g2,
                k_k,
                k_a,
                r_k,
                g_norm_w,
                g_norm_b,
            };

            let mut ffn_x_k = Tensor1D::zeros(c);
            init_centered(&mut ffn_x_k, &mut rng, 0.5, 0.02);
            let mut key_w = Tensor1D::zeros(i * c);
            let mut value_w = Tensor1D::zeros(c * i);
            init_uniform(&mut key_w, &mut rng, 0.02);
            init_uniform(&mut value_w, &mut rng, 0.02);

            let ffn = FfnWeights {
                x_k: ffn_x_k,
                key_w,
                value_w,
            };

            blocks.push(BlockWeights {
                pre_norm_w,
                pre_norm_b,
                attn_norm_w,
                attn_norm_b,
                ffn_norm_w,
                ffn_norm_b,
                attn,
                ffn,
            });
        }

        Ok(Self {
            cfg,
            embeddings,
            ln_out_w,
            ln_out_b,
            lm_head,
            blocks,
        })
    }

    /// Save model weights to a `.safetensors` file plus JSON sidecar config.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        #[derive(Clone)]
        struct TensorRec {
            name: String,
            shape: Vec<usize>,
            data: Vec<f32>,
        }

        let c = self.cfg.hidden_size;
        let v = self.cfg.vocab_size;
        let i = self.cfg.intermediate_size;
        let d_w = self.cfg.decay_low_rank;
        let d_a = self.cfg.a_low_rank;
        let d_v = self.cfg.v_low_rank;
        let d_g = self.cfg.g_low_rank;

        let mut recs = Vec::<TensorRec>::new();
        let push = |recs: &mut Vec<TensorRec>, name: String, shape: Vec<usize>, src: &Tensor1D| {
            recs.push(TensorRec {
                name,
                shape,
                data: src.as_slice().to_vec(),
            });
        };

        push(
            &mut recs,
            "model.embeddings.weight".to_string(),
            vec![v, c],
            &self.embeddings,
        );
        push(
            &mut recs,
            "model.norm.weight".to_string(),
            vec![c],
            &self.ln_out_w,
        );
        push(
            &mut recs,
            "model.norm.bias".to_string(),
            vec![c],
            &self.ln_out_b,
        );
        push(
            &mut recs,
            "lm_head.weight".to_string(),
            vec![v, c],
            &self.lm_head,
        );

        for (idx, b) in self.blocks.iter().enumerate() {
            let pfx = format!("model.layers.{idx}");
            if let (Some(w), Some(bias)) = (&b.pre_norm_w, &b.pre_norm_b) {
                push(&mut recs, format!("{pfx}.pre_norm.weight"), vec![c], w);
                push(&mut recs, format!("{pfx}.pre_norm.bias"), vec![c], bias);
            }

            push(
                &mut recs,
                format!("{pfx}.attn_norm.weight"),
                vec![c],
                &b.attn_norm_w,
            );
            push(
                &mut recs,
                format!("{pfx}.attn_norm.bias"),
                vec![c],
                &b.attn_norm_b,
            );
            push(
                &mut recs,
                format!("{pfx}.ffn_norm.weight"),
                vec![c],
                &b.ffn_norm_w,
            );
            push(
                &mut recs,
                format!("{pfx}.ffn_norm.bias"),
                vec![c],
                &b.ffn_norm_b,
            );

            let proj = b.attn.rkv_proj.as_slice();
            let proj_size = c * c;
            recs.push(TensorRec {
                name: format!("{pfx}.attn.r_proj.weight"),
                shape: vec![c, c],
                data: proj[0..proj_size].to_vec(),
            });
            recs.push(TensorRec {
                name: format!("{pfx}.attn.k_proj.weight"),
                shape: vec![c, c],
                data: proj[proj_size..2 * proj_size].to_vec(),
            });
            recs.push(TensorRec {
                name: format!("{pfx}.attn.v_proj.weight"),
                shape: vec![c, c],
                data: proj[2 * proj_size..3 * proj_size].to_vec(),
            });

            push(
                &mut recs,
                format!("{pfx}.attn.o_proj.weight"),
                vec![c, c],
                &b.attn.o_proj,
            );
            push(&mut recs, format!("{pfx}.attn.x_r"), vec![c], &b.attn.x_r);
            push(&mut recs, format!("{pfx}.attn.x_w"), vec![c], &b.attn.x_w);
            push(&mut recs, format!("{pfx}.attn.x_k"), vec![c], &b.attn.x_k);
            push(&mut recs, format!("{pfx}.attn.x_v"), vec![c], &b.attn.x_v);
            push(&mut recs, format!("{pfx}.attn.x_a"), vec![c], &b.attn.x_a);
            push(&mut recs, format!("{pfx}.attn.x_g"), vec![c], &b.attn.x_g);

            push(
                &mut recs,
                format!("{pfx}.attn.w_lora.lora.0.weight"),
                vec![d_w, c],
                &b.attn.w1,
            );
            push(
                &mut recs,
                format!("{pfx}.attn.w_lora.lora.2.weight"),
                vec![c, d_w],
                &b.attn.w2,
            );
            push(
                &mut recs,
                format!("{pfx}.attn.w_lora.lora.2.bias"),
                vec![c],
                &b.attn.w0,
            );

            push(
                &mut recs,
                format!("{pfx}.attn.a_lora.lora.0.weight"),
                vec![d_a, c],
                &b.attn.a1,
            );
            push(
                &mut recs,
                format!("{pfx}.attn.a_lora.lora.2.weight"),
                vec![c, d_a],
                &b.attn.a2,
            );
            push(
                &mut recs,
                format!("{pfx}.attn.a_lora.lora.2.bias"),
                vec![c],
                &b.attn.a0,
            );

            if let Some(v1) = &b.attn.v1 {
                push(
                    &mut recs,
                    format!("{pfx}.attn.v_lora.lora.0.weight"),
                    vec![d_v, c],
                    v1,
                );
            }
            if let Some(v2) = &b.attn.v2 {
                push(
                    &mut recs,
                    format!("{pfx}.attn.v_lora.lora.2.weight"),
                    vec![c, d_v],
                    v2,
                );
            }
            if let Some(v0) = &b.attn.v0 {
                push(
                    &mut recs,
                    format!("{pfx}.attn.v_lora.lora.2.bias"),
                    vec![c],
                    v0,
                );
            }

            push(
                &mut recs,
                format!("{pfx}.attn.g_lora.lora.0.weight"),
                vec![d_g, c],
                &b.attn.g1,
            );
            push(
                &mut recs,
                format!("{pfx}.attn.g_lora.lora.2.weight"),
                vec![c, d_g],
                &b.attn.g2,
            );

            push(&mut recs, format!("{pfx}.attn.k_k"), vec![c], &b.attn.k_k);
            push(&mut recs, format!("{pfx}.attn.k_a"), vec![c], &b.attn.k_a);
            push(&mut recs, format!("{pfx}.attn.r_k"), vec![c], &b.attn.r_k);
            push(
                &mut recs,
                format!("{pfx}.attn.g_norm.weight"),
                vec![c],
                &b.attn.g_norm_w,
            );
            push(
                &mut recs,
                format!("{pfx}.attn.g_norm.bias"),
                vec![c],
                &b.attn.g_norm_b,
            );

            push(&mut recs, format!("{pfx}.ffn.x_k"), vec![c], &b.ffn.x_k);
            push(
                &mut recs,
                format!("{pfx}.ffn.key.weight"),
                vec![i, c],
                &b.ffn.key_w,
            );
            push(
                &mut recs,
                format!("{pfx}.ffn.value.weight"),
                vec![c, i],
                &b.ffn.value_w,
            );
        }

        recs.sort_by(|a, b| a.name.cmp(&b.name));
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({}));
        for rec in &recs {
            let bytes = rec.data.len() * 4;
            header.insert(
                rec.name.clone(),
                json!({
                    "dtype": "F32",
                    "shape": rec.shape,
                    "data_offsets": [offset, offset + bytes]
                }),
            );
            offset += bytes;
        }
        let header_bytes = serde_json::to_vec(&header)?;
        let mut f = File::create(path.as_ref())?;
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(&header_bytes)?;
        for rec in &recs {
            for v in &rec.data {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Allocate zero-initialized Adam moments matching all trainable tensors.
    pub fn new_full_adam_state(&self) -> FullAdamState {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            blocks.push(BlockAdamState {
                pre_norm_w: b.pre_norm_w.as_ref().map(|t| AdamTensorState::new(t.len())),
                pre_norm_b: b.pre_norm_b.as_ref().map(|t| AdamTensorState::new(t.len())),
                attn_norm_w: AdamTensorState::new(b.attn_norm_w.len()),
                attn_norm_b: AdamTensorState::new(b.attn_norm_b.len()),
                ffn_norm_w: AdamTensorState::new(b.ffn_norm_w.len()),
                ffn_norm_b: AdamTensorState::new(b.ffn_norm_b.len()),
                attn: AttentionAdamState {
                    x_r: AdamTensorState::new(b.attn.x_r.len()),
                    x_w: AdamTensorState::new(b.attn.x_w.len()),
                    x_k: AdamTensorState::new(b.attn.x_k.len()),
                    x_v: AdamTensorState::new(b.attn.x_v.len()),
                    x_a: AdamTensorState::new(b.attn.x_a.len()),
                    x_g: AdamTensorState::new(b.attn.x_g.len()),
                    rkv_proj: AdamTensorState::new(b.attn.rkv_proj.len()),
                    o_proj: AdamTensorState::new(b.attn.o_proj.len()),
                    w1: AdamTensorState::new(b.attn.w1.len()),
                    w2: AdamTensorState::new(b.attn.w2.len()),
                    w0: AdamTensorState::new(b.attn.w0.len()),
                    a1: AdamTensorState::new(b.attn.a1.len()),
                    a2: AdamTensorState::new(b.attn.a2.len()),
                    a0: AdamTensorState::new(b.attn.a0.len()),
                    v1: b.attn.v1.as_ref().map(|t| AdamTensorState::new(t.len())),
                    v2: b.attn.v2.as_ref().map(|t| AdamTensorState::new(t.len())),
                    v0: b.attn.v0.as_ref().map(|t| AdamTensorState::new(t.len())),
                    g1: AdamTensorState::new(b.attn.g1.len()),
                    g2: AdamTensorState::new(b.attn.g2.len()),
                    k_k: AdamTensorState::new(b.attn.k_k.len()),
                    k_a: AdamTensorState::new(b.attn.k_a.len()),
                    r_k: AdamTensorState::new(b.attn.r_k.len()),
                    g_norm_w: AdamTensorState::new(b.attn.g_norm_w.len()),
                    g_norm_b: AdamTensorState::new(b.attn.g_norm_b.len()),
                },
                ffn: FfnAdamState {
                    x_k: AdamTensorState::new(b.ffn.x_k.len()),
                    key_w: AdamTensorState::new(b.ffn.key_w.len()),
                    value_w: AdamTensorState::new(b.ffn.value_w.len()),
                },
            });
        }
        FullAdamState {
            embeddings: AdamTensorState::new(self.embeddings.len()),
            ln_out_w: AdamTensorState::new(self.ln_out_w.len()),
            ln_out_b: AdamTensorState::new(self.ln_out_b.len()),
            lm_head: AdamTensorState::new(self.lm_head.len()),
            blocks,
        }
    }

    /// Save full-parameter Adam moments for exact online-training continuation.
    pub fn save_full_adam_safetensors<P: AsRef<Path>>(
        &self,
        adam: &FullAdamState,
        path: P,
    ) -> Result<()> {
        #[derive(Clone)]
        struct TensorRec {
            name: String,
            shape: Vec<usize>,
            data: Vec<f32>,
        }
        let c = self.cfg.hidden_size;
        let i = self.cfg.intermediate_size;
        let v = self.cfg.vocab_size;
        let h = self.cfg.num_heads;
        let n = self.cfg.head_dim;
        let d_w = self.cfg.decay_low_rank;
        let d_a = self.cfg.a_low_rank;
        let d_v = self.cfg.v_low_rank;
        let d_g = self.cfg.g_low_rank;
        let mut recs = Vec::<TensorRec>::new();
        let mut push_state = |name: &str, shape: Vec<usize>, st: &AdamTensorState| {
            recs.push(TensorRec {
                name: format!("{name}.m"),
                shape: shape.clone(),
                data: st.m.as_slice().to_vec(),
            });
            recs.push(TensorRec {
                name: format!("{name}.v"),
                shape,
                data: st.v.as_slice().to_vec(),
            });
        };

        push_state("opt.model.embeddings.weight", vec![v, c], &adam.embeddings);
        push_state("opt.model.norm.weight", vec![c], &adam.ln_out_w);
        push_state("opt.model.norm.bias", vec![c], &adam.ln_out_b);
        push_state("opt.lm_head.weight", vec![v, c], &adam.lm_head);
        for (idx, b) in adam.blocks.iter().enumerate() {
            let p = format!("opt.model.layers.{idx}");
            if let Some(st) = &b.pre_norm_w {
                push_state(&format!("{p}.pre_norm.weight"), vec![c], st);
            }
            if let Some(st) = &b.pre_norm_b {
                push_state(&format!("{p}.pre_norm.bias"), vec![c], st);
            }
            push_state(&format!("{p}.attn_norm.weight"), vec![c], &b.attn_norm_w);
            push_state(&format!("{p}.attn_norm.bias"), vec![c], &b.attn_norm_b);
            push_state(&format!("{p}.ffn_norm.weight"), vec![c], &b.ffn_norm_w);
            push_state(&format!("{p}.ffn_norm.bias"), vec![c], &b.ffn_norm_b);

            push_state(&format!("{p}.attn.x_r"), vec![c], &b.attn.x_r);
            push_state(&format!("{p}.attn.x_w"), vec![c], &b.attn.x_w);
            push_state(&format!("{p}.attn.x_k"), vec![c], &b.attn.x_k);
            push_state(&format!("{p}.attn.x_v"), vec![c], &b.attn.x_v);
            push_state(&format!("{p}.attn.x_a"), vec![c], &b.attn.x_a);
            push_state(&format!("{p}.attn.x_g"), vec![c], &b.attn.x_g);
            push_state(
                &format!("{p}.attn.rkv_proj"),
                vec![3, c, c],
                &b.attn.rkv_proj,
            );
            push_state(
                &format!("{p}.attn.o_proj.weight"),
                vec![c, c],
                &b.attn.o_proj,
            );
            push_state(
                &format!("{p}.attn.w_lora.lora.0.weight"),
                vec![d_w, c],
                &b.attn.w1,
            );
            push_state(
                &format!("{p}.attn.w_lora.lora.2.weight"),
                vec![c, d_w],
                &b.attn.w2,
            );
            push_state(&format!("{p}.attn.w_lora.lora.2.bias"), vec![c], &b.attn.w0);
            push_state(
                &format!("{p}.attn.a_lora.lora.0.weight"),
                vec![d_a, c],
                &b.attn.a1,
            );
            push_state(
                &format!("{p}.attn.a_lora.lora.2.weight"),
                vec![c, d_a],
                &b.attn.a2,
            );
            push_state(&format!("{p}.attn.a_lora.lora.2.bias"), vec![c], &b.attn.a0);
            if let Some(st) = &b.attn.v1 {
                push_state(&format!("{p}.attn.v_lora.lora.0.weight"), vec![d_v, c], st);
            }
            if let Some(st) = &b.attn.v2 {
                push_state(&format!("{p}.attn.v_lora.lora.2.weight"), vec![c, d_v], st);
            }
            if let Some(st) = &b.attn.v0 {
                push_state(&format!("{p}.attn.v_lora.lora.2.bias"), vec![c], st);
            }
            push_state(
                &format!("{p}.attn.g_lora.lora.0.weight"),
                vec![d_g, c],
                &b.attn.g1,
            );
            push_state(
                &format!("{p}.attn.g_lora.lora.2.weight"),
                vec![c, d_g],
                &b.attn.g2,
            );
            push_state(&format!("{p}.attn.k_k"), vec![c], &b.attn.k_k);
            push_state(&format!("{p}.attn.k_a"), vec![c], &b.attn.k_a);
            push_state(&format!("{p}.attn.r_k"), vec![h, n], &b.attn.r_k);
            push_state(
                &format!("{p}.attn.g_norm.weight"),
                vec![c],
                &b.attn.g_norm_w,
            );
            push_state(&format!("{p}.attn.g_norm.bias"), vec![c], &b.attn.g_norm_b);

            push_state(&format!("{p}.ffn.x_k"), vec![c], &b.ffn.x_k);
            push_state(&format!("{p}.ffn.key.weight"), vec![i, c], &b.ffn.key_w);
            push_state(&format!("{p}.ffn.value.weight"), vec![c, i], &b.ffn.value_w);
        }

        recs.sort_by(|a, b| a.name.cmp(&b.name));
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({}));
        for rec in &recs {
            let bytes = rec.data.len() * 4;
            header.insert(
                rec.name.clone(),
                json!({
                    "dtype": "F32",
                    "shape": rec.shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }

        let header_bytes = serde_json::to_vec(&header)?;
        let mut f = File::create(path)?;
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(&header_bytes)?;
        for rec in &recs {
            for v in &rec.data {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Load full-parameter Adam moments and validate tensor shapes.
    pub fn load_full_adam_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<FullAdamState> {
        let weights = Weights::load(path.as_ref()).with_context(|| {
            format!(
                "failed to load optimizer moments from {}",
                path.as_ref().display()
            )
        })?;
        let mut adam = self.new_full_adam_state();
        let load_state = |name: &str, st: &mut AdamTensorState| -> Result<()> {
            let m_name = format!("{name}.m");
            let v_name = format!("{name}.v");
            let m_t = weights
                .require(&m_name)
                .with_context(|| format!("missing optimizer tensor '{m_name}'"))?;
            let v_t = weights
                .require(&v_name)
                .with_context(|| format!("missing optimizer tensor '{v_name}'"))?;
            if m_t.data().len() != st.m.len() {
                bail!(
                    "optimizer tensor '{}' len {} != expected {}",
                    m_name,
                    m_t.data().len(),
                    st.m.len()
                );
            }
            if v_t.data().len() != st.v.len() {
                bail!(
                    "optimizer tensor '{}' len {} != expected {}",
                    v_name,
                    v_t.data().len(),
                    st.v.len()
                );
            }
            st.m.as_mut_slice().copy_from_slice(m_t.data());
            st.v.as_mut_slice().copy_from_slice(v_t.data());
            Ok(())
        };

        let c = self.cfg.hidden_size;
        let i = self.cfg.intermediate_size;
        let v = self.cfg.vocab_size;
        let h = self.cfg.num_heads;
        let n = self.cfg.head_dim;
        let _ = (c, i, v, h, n);
        load_state("opt.model.embeddings.weight", &mut adam.embeddings)?;
        load_state("opt.model.norm.weight", &mut adam.ln_out_w)?;
        load_state("opt.model.norm.bias", &mut adam.ln_out_b)?;
        load_state("opt.lm_head.weight", &mut adam.lm_head)?;
        for (idx, b) in adam.blocks.iter_mut().enumerate() {
            let p = format!("opt.model.layers.{idx}");
            if let Some(st) = b.pre_norm_w.as_mut() {
                load_state(&format!("{p}.pre_norm.weight"), st)?;
            }
            if let Some(st) = b.pre_norm_b.as_mut() {
                load_state(&format!("{p}.pre_norm.bias"), st)?;
            }
            load_state(&format!("{p}.attn_norm.weight"), &mut b.attn_norm_w)?;
            load_state(&format!("{p}.attn_norm.bias"), &mut b.attn_norm_b)?;
            load_state(&format!("{p}.ffn_norm.weight"), &mut b.ffn_norm_w)?;
            load_state(&format!("{p}.ffn_norm.bias"), &mut b.ffn_norm_b)?;
            load_state(&format!("{p}.attn.x_r"), &mut b.attn.x_r)?;
            load_state(&format!("{p}.attn.x_w"), &mut b.attn.x_w)?;
            load_state(&format!("{p}.attn.x_k"), &mut b.attn.x_k)?;
            load_state(&format!("{p}.attn.x_v"), &mut b.attn.x_v)?;
            load_state(&format!("{p}.attn.x_a"), &mut b.attn.x_a)?;
            load_state(&format!("{p}.attn.x_g"), &mut b.attn.x_g)?;
            load_state(&format!("{p}.attn.rkv_proj"), &mut b.attn.rkv_proj)?;
            load_state(&format!("{p}.attn.o_proj.weight"), &mut b.attn.o_proj)?;
            load_state(&format!("{p}.attn.w_lora.lora.0.weight"), &mut b.attn.w1)?;
            load_state(&format!("{p}.attn.w_lora.lora.2.weight"), &mut b.attn.w2)?;
            load_state(&format!("{p}.attn.w_lora.lora.2.bias"), &mut b.attn.w0)?;
            load_state(&format!("{p}.attn.a_lora.lora.0.weight"), &mut b.attn.a1)?;
            load_state(&format!("{p}.attn.a_lora.lora.2.weight"), &mut b.attn.a2)?;
            load_state(&format!("{p}.attn.a_lora.lora.2.bias"), &mut b.attn.a0)?;
            if let Some(st) = b.attn.v1.as_mut() {
                load_state(&format!("{p}.attn.v_lora.lora.0.weight"), st)?;
            }
            if let Some(st) = b.attn.v2.as_mut() {
                load_state(&format!("{p}.attn.v_lora.lora.2.weight"), st)?;
            }
            if let Some(st) = b.attn.v0.as_mut() {
                load_state(&format!("{p}.attn.v_lora.lora.2.bias"), st)?;
            }
            load_state(&format!("{p}.attn.g_lora.lora.0.weight"), &mut b.attn.g1)?;
            load_state(&format!("{p}.attn.g_lora.lora.2.weight"), &mut b.attn.g2)?;
            load_state(&format!("{p}.attn.k_k"), &mut b.attn.k_k)?;
            load_state(&format!("{p}.attn.k_a"), &mut b.attn.k_a)?;
            load_state(&format!("{p}.attn.r_k"), &mut b.attn.r_k)?;
            load_state(&format!("{p}.attn.g_norm.weight"), &mut b.attn.g_norm_w)?;
            load_state(&format!("{p}.attn.g_norm.bias"), &mut b.attn.g_norm_b)?;
            load_state(&format!("{p}.ffn.x_k"), &mut b.ffn.x_k)?;
            load_state(&format!("{p}.ffn.key.weight"), &mut b.ffn.key_w)?;
            load_state(&format!("{p}.ffn.value.weight"), &mut b.ffn.value_w)?;
        }
        Ok(adam)
    }

    /// Get model configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Create new state for this model.
    pub fn new_state(&self) -> State {
        State::new(&self.cfg)
    }

    /// Immutable LM-head weights, row-major `(vocab, hidden)`.
    #[inline]
    pub fn lm_head_weights(&self) -> &[f32] {
        self.lm_head.as_slice()
    }

    /// Mutable LM-head weights, row-major `(vocab, hidden)`.
    #[inline]
    pub fn lm_head_weights_mut(&mut self) -> &mut [f32] {
        self.lm_head.as_mut_slice()
    }

    /// Perform one exact bptt=1 online training step over the latest forward trace.
    #[allow(clippy::too_many_arguments)]
    pub fn online_train_step_bptt1(
        &mut self,
        scratch: &mut ScratchBuffers,
        state: &State,
        symbol: u8,
        pdf: &[f64],
        scope: TrainScopeMask,
        optimizer: OptimizerKind,
        lr: f32,
        clip: f32,
        adam_t: &mut usize,
        model_adam: Option<&mut FullAdamState>,
        out_bias: Option<&mut [f32]>,
        out_bias_adam_m: Option<&mut [f32]>,
        out_bias_adam_v: Option<&mut [f32]>,
    ) -> Result<()> {
        if !scope.trains_any_params() {
            return Ok(());
        }
        if scope.trains_non_head_params() && !scratch.train_trace_valid {
            bail!("rwkv full training trace is missing; run one forward step first");
        }
        let c = self.cfg.hidden_size;
        let h = self.cfg.num_heads;
        let n = self.cfg.head_dim;
        let i = self.cfg.intermediate_size;
        let d_w = self.cfg.decay_low_rank;
        let d_a = self.cfg.a_low_rank;
        let d_v = self.cfg.v_low_rank;
        let d_g = self.cfg.g_low_rank;
        let vocab = self.cfg.vocab_size.min(pdf.len());
        if vocab == 0 {
            return Ok(());
        }
        let mut adam_step = None::<AdamStep>;
        let mut model_adam = model_adam;
        if matches!(optimizer, OptimizerKind::Adam) {
            *adam_t = adam_t.saturating_add(1);
            let t = (*adam_t).max(1) as i32;
            let b1 = 0.9f32;
            let b2 = 0.999f32;
            adam_step = Some(AdamStep {
                lr,
                clip: clip.max(0.0),
                b1,
                b2,
                eps: 1e-8,
                bias_corr1: 1.0 - b1.powi(t),
                bias_corr2: 1.0 - b2.powi(t),
            });
            if scope.trains_non_head_params() && model_adam.is_none() {
                bail!("rwkv Adam full-training state is missing");
            }
        }

        scratch.grad_logits.zero();
        for idx in 0..vocab {
            let p = pdf[idx].clamp(1e-12, 1.0) as f32;
            let target = if idx == symbol as usize { 1.0 } else { 0.0 };
            let mut g = target - p;
            if clip > 0.0 {
                g = g.clamp(-clip, clip);
            }
            scratch.grad_logits[idx] = g;
        }

        if scope.bias
            && let Some(bias) = out_bias
        {
            match optimizer {
                OptimizerKind::Sgd => {
                    for idx in 0..bias.len().min(vocab) {
                        bias[idx] += lr * scratch.grad_logits[idx];
                    }
                }
                OptimizerKind::Adam => {
                    let cfg = adam_step.as_ref().expect("adam cfg initialized");
                    let Some(m) = out_bias_adam_m else {
                        bail!("rwkv Adam output-bias state is missing (m)");
                    };
                    let Some(vv) = out_bias_adam_v else {
                        bail!("rwkv Adam output-bias state is missing (v)");
                    };
                    let n = bias.len().min(vocab);
                    apply_adam_vec_update_raw(
                        &mut bias[0..n],
                        &scratch.grad_logits.as_slice()[0..n],
                        &mut m[0..n],
                        &mut vv[0..n],
                        cfg,
                    );
                }
            }
        }

        scratch.grad_x.zero();
        for row in 0..vocab {
            let g = scratch.grad_logits[row];
            if g == 0.0 {
                continue;
            }
            let row_off = row * c;
            for col in 0..c {
                scratch.grad_x[col] += self.lm_head[row_off + col] * g;
            }
        }

        if scope.head {
            match optimizer {
                OptimizerKind::Sgd => {
                    sgd_outer_update(
                        self.lm_head.as_mut_slice(),
                        vocab,
                        c,
                        &scratch.grad_logits.as_slice()[0..vocab],
                        scratch.x_normed.as_slice(),
                        lr,
                        clip,
                    );
                }
                OptimizerKind::Adam => {
                    let cfg = adam_step.as_ref().expect("adam cfg initialized");
                    let adam = model_adam.as_mut().expect("adam state exists");
                    apply_adam_outer_update(
                        self.lm_head.as_mut_slice(),
                        vocab,
                        c,
                        &scratch.grad_logits.as_slice()[0..vocab],
                        scratch.x_normed.as_slice(),
                        &mut adam.lm_head,
                        cfg,
                    );
                }
            }
        }

        let needs_backprop = scope.trains_non_head_params() || scope.head;
        if !needs_backprop {
            return Ok(());
        }
        layer_norm_backward(
            scratch.x.as_slice(),
            self.ln_out_w.as_slice(),
            scratch.grad_x.as_slice(),
            self.cfg.layer_norm_eps,
            scratch.grad_x2.as_mut_slice(),
            scratch.grad_x3.as_mut_slice(),
            scratch.grad_x4.as_mut_slice(),
        );
        if scope.head {
            match optimizer {
                OptimizerKind::Sgd => {
                    sgd_vec_update(
                        self.ln_out_w.as_mut_slice(),
                        scratch.grad_x3.as_slice(),
                        lr,
                        clip,
                    );
                    sgd_vec_update(
                        self.ln_out_b.as_mut_slice(),
                        scratch.grad_x4.as_slice(),
                        lr,
                        clip,
                    );
                }
                OptimizerKind::Adam => {
                    let cfg = adam_step.as_ref().expect("adam cfg initialized");
                    let adam = model_adam.as_mut().expect("adam state exists");
                    apply_adam_vec_update(
                        self.ln_out_w.as_mut_slice(),
                        scratch.grad_x3.as_slice(),
                        &mut adam.ln_out_w,
                        cfg,
                    );
                    apply_adam_vec_update(
                        self.ln_out_b.as_mut_slice(),
                        scratch.grad_x4.as_slice(),
                        &mut adam.ln_out_b,
                        cfg,
                    );
                }
            }
        }
        scratch.grad_x.copy_from_slice(scratch.grad_x2.as_slice());
        scratch.grad_v_first.zero();

        for layer_idx in (0..self.cfg.num_layers).rev() {
            let tr = &scratch.train_trace_layers[layer_idx];
            let block = &mut self.blocks[layer_idx];

            // FFN residual split: x_out = x_after_attn + ffn_out.
            scratch.grad_x2.copy_from_slice(scratch.grad_x.as_slice()); // d x_after_attn
            scratch.grad_x3.copy_from_slice(scratch.grad_x.as_slice()); // d ffn_out

            // ffn_out = value_w @ ffn_k
            scratch.grad_ffn.zero();
            for row in 0..c {
                let g = scratch.grad_x3[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * i;
                for col in 0..i {
                    scratch.grad_ffn[col] += block.ffn.value_w[off + col] * g;
                }
            }
            if scope.ffn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.ffn.value_w.as_mut_slice(),
                        c,
                        i,
                        scratch.grad_x3.as_slice(),
                        tr.ffn_k.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.ffn.value_w.as_mut_slice(),
                            c,
                            i,
                            scratch.grad_x3.as_slice(),
                            tr.ffn_k.as_slice(),
                            &mut adam.ffn.value_w,
                            cfg,
                        );
                    }
                }
            }

            // relu^2 backward
            for col in 0..i {
                let pre = tr.ffn_pre[col];
                scratch.grad_ffn2[col] = if pre > 0.0 {
                    scratch.grad_ffn[col] * (2.0 * pre)
                } else {
                    0.0
                };
            }

            // key_w backward
            scratch.grad_x4.zero();
            for row in 0..i {
                let g = scratch.grad_ffn2[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * c;
                for col in 0..c {
                    scratch.grad_x4[col] += block.ffn.key_w[off + col] * g;
                }
            }
            if scope.ffn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.ffn.key_w.as_mut_slice(),
                        i,
                        c,
                        scratch.grad_ffn2.as_slice(),
                        tr.ffn_xk.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.ffn.key_w.as_mut_slice(),
                            i,
                            c,
                            scratch.grad_ffn2.as_slice(),
                            tr.ffn_xk.as_slice(),
                            &mut adam.ffn.key_w,
                            cfg,
                        );
                    }
                }
            }

            // token_shift backward (ffn)
            for col in 0..c {
                let g = scratch.grad_x4[col];
                let mix = block.ffn.x_k[col];
                let base = tr.ffn_norm[col];
                let prev = tr.ffn_x_prev_old[col];
                scratch.grad_x5[col] = g * (1.0 - mix); // d ffn_norm
                scratch.grad_param[col] = g * (prev - base); // d x_k
            }
            if scope.ffn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.ffn.x_k.as_mut_slice(),
                        scratch.grad_param.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.ffn.x_k.as_mut_slice(),
                            scratch.grad_param.as_slice(),
                            &mut adam.ffn.x_k,
                            cfg,
                        );
                    }
                }
            }

            // ffn norm backward: d x_after_attn contribution.
            layer_norm_backward(
                tr.x_after_attn.as_slice(),
                block.ffn_norm_w.as_slice(),
                scratch.grad_x5.as_slice(),
                self.cfg.layer_norm_eps,
                scratch.grad_x4.as_mut_slice(),
                scratch.grad_x3.as_mut_slice(),
                scratch.grad_x6.as_mut_slice(),
            );
            if scope.ffn_norm {
                match optimizer {
                    OptimizerKind::Sgd => {
                        sgd_vec_update(
                            block.ffn_norm_w.as_mut_slice(),
                            scratch.grad_x3.as_slice(),
                            lr,
                            clip,
                        );
                        sgd_vec_update(
                            block.ffn_norm_b.as_mut_slice(),
                            scratch.grad_x6.as_slice(),
                            lr,
                            clip,
                        );
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.ffn_norm_w.as_mut_slice(),
                            scratch.grad_x3.as_slice(),
                            &mut adam.ffn_norm_w,
                            cfg,
                        );
                        apply_adam_vec_update(
                            block.ffn_norm_b.as_mut_slice(),
                            scratch.grad_x6.as_slice(),
                            &mut adam.ffn_norm_b,
                            cfg,
                        );
                    }
                }
            }
            for col in 0..c {
                scratch.grad_x2[col] += scratch.grad_x4[col];
            }

            // Attention residual split.
            scratch.grad_x.copy_from_slice(scratch.grad_x2.as_slice()); // d x_after_pre
            scratch.grad_x3.copy_from_slice(scratch.grad_x2.as_slice()); // d att_out

            // out_proj backward
            scratch.grad_x4.zero(); // d y_gate
            for row in 0..c {
                let g = scratch.grad_x3[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * c;
                for col in 0..c {
                    scratch.grad_x4[col] += block.attn.o_proj[off + col] * g;
                }
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.attn.o_proj.as_mut_slice(),
                        c,
                        c,
                        scratch.grad_x3.as_slice(),
                        tr.y_gate.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.attn.o_proj.as_mut_slice(),
                            c,
                            c,
                            scratch.grad_x3.as_slice(),
                            tr.y_gate.as_slice(),
                            &mut adam.attn.o_proj,
                            cfg,
                        );
                    }
                }
            }

            // Gate backward.
            for col in 0..c {
                let gy = scratch.grad_x4[col];
                scratch.grad_saved[col] = gy * tr.y_head[col]; // d g
                scratch.grad_x4[col] = gy * tr.g[col]; // d y_head
            }

            // Head-qk branch.
            scratch.grad_x2.zero(); // d r
            scratch.grad_x3.zero(); // d k_scaled
            scratch.grad_x6.zero(); // d v_final
            scratch.grad_param.zero(); // d r_k
            for head_idx in 0..h {
                let off = head_idx * n;
                let mut g_alpha = 0.0f32;
                for j in 0..n {
                    let g = scratch.grad_x4[off + j];
                    g_alpha += g * tr.v[off + j];
                    scratch.grad_x6[off + j] += g * tr.alpha[head_idx];
                }
                for j in 0..n {
                    let idx = off + j;
                    let rk = block.attn.r_k[idx];
                    let rv = tr.r[idx];
                    let kv = tr.k[idx];
                    let g = g_alpha * rk;
                    scratch.grad_x2[idx] += g * kv;
                    scratch.grad_x3[idx] += g * rv;
                    scratch.grad_param[idx] += g_alpha * rv * kv;
                }
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.r_k.as_mut_slice(),
                        scratch.grad_param.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.r_k.as_mut_slice(),
                            scratch.grad_param.as_slice(),
                            &mut adam.attn.r_k,
                            cfg,
                        );
                    }
                }
            }

            // GroupNorm backward.
            scratch.grad_x5.as_mut_slice()[0..c].copy_from_slice(&scratch.grad_x4.as_slice()[0..c]);
            group_norm_backward(
                tr.y_wkv.as_slice(),
                block.attn.g_norm_w.as_slice(),
                scratch.grad_x5.as_slice(),
                h,
                n,
                self.cfg.group_norm_eps,
                scratch.grad_x4.as_mut_slice(),     // d y_wkv
                scratch.grad_param.as_mut_slice(),  // d g_norm_w
                scratch.grad_param2.as_mut_slice(), // d g_norm_b
            );
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => {
                        sgd_vec_update(
                            block.attn.g_norm_w.as_mut_slice(),
                            scratch.grad_param.as_slice(),
                            lr,
                            clip,
                        );
                        sgd_vec_update(
                            block.attn.g_norm_b.as_mut_slice(),
                            scratch.grad_param2.as_slice(),
                            lr,
                            clip,
                        );
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.g_norm_w.as_mut_slice(),
                            scratch.grad_param.as_slice(),
                            &mut adam.attn.g_norm_w,
                            cfg,
                        );
                        apply_adam_vec_update(
                            block.attn.g_norm_b.as_mut_slice(),
                            scratch.grad_param2.as_slice(),
                            &mut adam.attn.g_norm_b,
                            cfg,
                        );
                    }
                }
            }

            // WKV kernel backward.
            scratch.grad_param.zero(); // d w_decay
            scratch.grad_x5.zero(); // d a
            scratch.grad_param2.zero(); // d kk
            let s_old = tr.att_state_old.as_slice();
            let s_new = state.layers[layer_idx].att_state.as_slice();
            for head_idx in 0..h {
                let off = head_idx * n;
                let s_head_old_off = head_idx * n * n;
                let s_head_new_off = head_idx * n * n;
                for row in 0..n {
                    let gy = scratch.grad_x4[off + row];
                    if gy == 0.0 {
                        continue;
                    }
                    let row_old_off = s_head_old_off + row * n;
                    let row_new_off = s_head_new_off + row * n;
                    let mut u = 0.0f32;
                    for col in 0..n {
                        u += s_old[row_old_off + col] * tr.kk[off + col];
                    }
                    let mut du = 0.0f32;
                    let v_i = tr.v[off + row];
                    for col in 0..n {
                        let idx = off + col;
                        let d_s = gy * tr.r[idx];
                        scratch.grad_x2[idx] += s_new[row_new_off + col] * gy; // d r
                        scratch.grad_param[idx] += d_s * s_old[row_old_off + col]; // d w
                        scratch.grad_x3[idx] += d_s * v_i; // d k_scaled
                        scratch.grad_x6[off + row] += d_s * tr.k[idx]; // d v
                        let d_kka = -d_s * u;
                        scratch.grad_param2[idx] += d_kka * tr.a[idx]; // d kk via kka
                        scratch.grad_x5[idx] += d_kka * tr.kk[idx]; // d a via kka
                        du += -d_s * (tr.kk[idx] * tr.a[idx]);
                    }
                    for col in 0..n {
                        let idx = off + col;
                        scratch.grad_param2[idx] += du * s_old[row_old_off + col];
                    }
                }
            }

            // k scaling + kk normalization backward.
            for col in 0..c {
                let gk = scratch.grad_x3[col];
                let scale = 1.0 + (tr.a[col] - 1.0) * block.attn.k_a[col];
                let d_scale = gk * tr.k_pre[col];
                scratch.grad_x3[col] = gk * scale; // d k_pre (scaled path)
                scratch.grad_x5[col] += d_scale * block.attn.k_a[col]; // d a
                scratch.grad_param[col] = d_scale * (tr.a[col] - 1.0); // d k_a
            }
            for head_idx in 0..h {
                let off = head_idx * n;
                l2_normalize_backward(
                    &tr.kk_pre.as_slice()[off..off + n],
                    &tr.kk.as_slice()[off..off + n],
                    &scratch.grad_param2.as_slice()[off..off + n],
                    1e-12,
                    &mut scratch.grad_x4.as_mut_slice()[off..off + n],
                );
            }
            for col in 0..c {
                let g = scratch.grad_x4[col];
                scratch.grad_x3[col] += g * block.attn.k_k[col]; // d k_pre from kk_pre
                scratch.grad_param2[col] = g * tr.k_pre[col]; // d k_k
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => {
                        sgd_vec_update(
                            block.attn.k_a.as_mut_slice(),
                            scratch.grad_param.as_slice(),
                            lr,
                            clip,
                        );
                        sgd_vec_update(
                            block.attn.k_k.as_mut_slice(),
                            scratch.grad_param2.as_slice(),
                            lr,
                            clip,
                        );
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.k_a.as_mut_slice(),
                            scratch.grad_param.as_slice(),
                            &mut adam.attn.k_a,
                            cfg,
                        );
                        apply_adam_vec_update(
                            block.attn.k_k.as_mut_slice(),
                            scratch.grad_param2.as_slice(),
                            &mut adam.attn.k_k,
                            cfg,
                        );
                    }
                }
            }

            // V-residual backward (layers > 0).
            scratch
                .grad_param2
                .copy_from_slice(scratch.grad_x6.as_slice()); // d v_final snapshot
            if layer_idx == 0 {
                for col in 0..c {
                    scratch.grad_x6[col] += scratch.grad_v_first[col];
                }
            } else if tr.uses_v_residual
                && let (Some(v1), Some(v2), Some(v0)) =
                    (&mut block.attn.v1, &mut block.attn.v2, &mut block.attn.v0)
            {
                for col in 0..c {
                    let gv = scratch.grad_param2[col];
                    let nu = tr.nu[col];
                    scratch.grad_x6[col] = gv * (1.0 - nu); // d v_pre
                    scratch.grad_x3[col] = gv * (scratch.train_v_first[col] - tr.v_pre[col]); // d nu
                    scratch.grad_v_first[col] += gv * nu; // d v_first
                }
                for col in 0..c {
                    let nu = tr.nu[col];
                    scratch.grad_x3[col] *= nu * (1.0 - nu); // d nu_pre
                }
                if scope.attn {
                    match optimizer {
                        OptimizerKind::Sgd => {
                            sgd_vec_update(v0.as_mut_slice(), scratch.grad_x3.as_slice(), lr, clip)
                        }
                        OptimizerKind::Adam => {
                            let cfg = adam_step.as_ref().expect("adam cfg initialized");
                            let adam = &mut model_adam.as_mut().expect("adam state exists").blocks
                                [layer_idx];
                            apply_adam_vec_update(
                                v0.as_mut_slice(),
                                scratch.grad_x3.as_slice(),
                                adam.attn.v0.as_mut().expect("adam v0 state"),
                                cfg,
                            );
                        }
                    }
                }
                if scope.attn {
                    match optimizer {
                        OptimizerKind::Sgd => sgd_outer_update(
                            v2.as_mut_slice(),
                            c,
                            d_v,
                            scratch.grad_x3.as_slice(),
                            &tr.v_hidden.as_slice()[0..d_v],
                            lr,
                            clip,
                        ),
                        OptimizerKind::Adam => {
                            let cfg = adam_step.as_ref().expect("adam cfg initialized");
                            let adam = &mut model_adam.as_mut().expect("adam state exists").blocks
                                [layer_idx];
                            apply_adam_outer_update(
                                v2.as_mut_slice(),
                                c,
                                d_v,
                                scratch.grad_x3.as_slice(),
                                &tr.v_hidden.as_slice()[0..d_v],
                                adam.attn.v2.as_mut().expect("adam v2 state"),
                                cfg,
                            );
                        }
                    }
                }
                scratch.grad_low_rank.zero();
                for row in 0..c {
                    let g = scratch.grad_x3[row];
                    if g == 0.0 {
                        continue;
                    }
                    let off = row * d_v;
                    for col in 0..d_v {
                        scratch.grad_low_rank[col] += v2[off + col] * g;
                    }
                }
                if scope.attn {
                    match optimizer {
                        OptimizerKind::Sgd => sgd_outer_update(
                            v1.as_mut_slice(),
                            d_v,
                            c,
                            &scratch.grad_low_rank.as_slice()[0..d_v],
                            tr.xv.as_slice(),
                            lr,
                            clip,
                        ),
                        OptimizerKind::Adam => {
                            let cfg = adam_step.as_ref().expect("adam cfg initialized");
                            let adam = &mut model_adam.as_mut().expect("adam state exists").blocks
                                [layer_idx];
                            apply_adam_outer_update(
                                v1.as_mut_slice(),
                                d_v,
                                c,
                                &scratch.grad_low_rank.as_slice()[0..d_v],
                                tr.xv.as_slice(),
                                adam.attn.v1.as_mut().expect("adam v1 state"),
                                cfg,
                            );
                        }
                    }
                }
                for col in 0..c {
                    let mut acc = 0.0f32;
                    for row in 0..d_v {
                        acc += v1[row * c + col] * scratch.grad_low_rank[row];
                    }
                    scratch.grad_x4[col] += acc; // add into d xv after projection transpose
                }
            }

            // R/K/V projection updates and input grads.
            let proj_size = c * c;
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => {
                        sgd_outer_update(
                            &mut block.attn.rkv_proj.as_mut_slice()[0..proj_size],
                            c,
                            c,
                            scratch.grad_x2.as_slice(),
                            tr.xr.as_slice(),
                            lr,
                            clip,
                        );
                        sgd_outer_update(
                            &mut block.attn.rkv_proj.as_mut_slice()[proj_size..2 * proj_size],
                            c,
                            c,
                            scratch.grad_x3.as_slice(),
                            tr.xk.as_slice(),
                            lr,
                            clip,
                        );
                        sgd_outer_update(
                            &mut block.attn.rkv_proj.as_mut_slice()[2 * proj_size..3 * proj_size],
                            c,
                            c,
                            scratch.grad_x6.as_slice(),
                            tr.xv.as_slice(),
                            lr,
                            clip,
                        );
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update_raw(
                            &mut block.attn.rkv_proj.as_mut_slice()[0..proj_size],
                            c,
                            c,
                            scratch.grad_x2.as_slice(),
                            tr.xr.as_slice(),
                            &mut adam.attn.rkv_proj.m.as_mut_slice()[0..proj_size],
                            &mut adam.attn.rkv_proj.v.as_mut_slice()[0..proj_size],
                            cfg,
                        );
                        apply_adam_outer_update_raw(
                            &mut block.attn.rkv_proj.as_mut_slice()[proj_size..2 * proj_size],
                            c,
                            c,
                            scratch.grad_x3.as_slice(),
                            tr.xk.as_slice(),
                            &mut adam.attn.rkv_proj.m.as_mut_slice()[proj_size..2 * proj_size],
                            &mut adam.attn.rkv_proj.v.as_mut_slice()[proj_size..2 * proj_size],
                            cfg,
                        );
                        apply_adam_outer_update_raw(
                            &mut block.attn.rkv_proj.as_mut_slice()[2 * proj_size..3 * proj_size],
                            c,
                            c,
                            scratch.grad_x6.as_slice(),
                            tr.xv.as_slice(),
                            &mut adam.attn.rkv_proj.m.as_mut_slice()[2 * proj_size..3 * proj_size],
                            &mut adam.attn.rkv_proj.v.as_mut_slice()[2 * proj_size..3 * proj_size],
                            cfg,
                        );
                    }
                }
            }
            scratch.grad_param.zero(); // d xr
            scratch.grad_param2.zero(); // d xk
            scratch.grad_x4.zero(); // d xv
            let proj = block.attn.rkv_proj.as_slice();
            for row in 0..c {
                let gr = scratch.grad_x2[row];
                let gk = scratch.grad_x3[row];
                let gv = scratch.grad_x6[row];
                let off = row * c;
                for col in 0..c {
                    scratch.grad_param[col] += proj[off + col] * gr;
                    scratch.grad_param2[col] += proj[proj_size + off + col] * gk;
                    scratch.grad_x4[col] += proj[2 * proj_size + off + col] * gv;
                }
            }

            // W low-rank backward.
            let inv_sqrt_e = 1.0 / std::f32::consts::E.sqrt();
            for col in 0..c {
                let sig = sigmoid(tr.w_pre[col]);
                let d_sig = scratch.grad_param[col] * (-inv_sqrt_e) * tr.w_decay[col];
                scratch.grad_param[col] = d_sig * sig * (1.0 - sig); // d w_pre
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.w0.as_mut_slice(),
                        scratch.grad_param.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.w0.as_mut_slice(),
                            scratch.grad_param.as_slice(),
                            &mut adam.attn.w0,
                            cfg,
                        );
                    }
                }
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.attn.w2.as_mut_slice(),
                        c,
                        d_w,
                        scratch.grad_param.as_slice(),
                        &tr.w_hidden.as_slice()[0..d_w],
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.attn.w2.as_mut_slice(),
                            c,
                            d_w,
                            scratch.grad_param.as_slice(),
                            &tr.w_hidden.as_slice()[0..d_w],
                            &mut adam.attn.w2,
                            cfg,
                        );
                    }
                }
            }
            scratch.grad_low_rank.zero();
            for row in 0..c {
                let g = scratch.grad_param[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * d_w;
                for col in 0..d_w {
                    scratch.grad_low_rank[col] += block.attn.w2[off + col] * g;
                }
            }
            for col in 0..d_w {
                let t = tr.w_hidden[col];
                scratch.grad_low_rank[col] *= 1.0 - t * t;
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.attn.w1.as_mut_slice(),
                        d_w,
                        c,
                        &scratch.grad_low_rank.as_slice()[0..d_w],
                        tr.xw.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.attn.w1.as_mut_slice(),
                            d_w,
                            c,
                            &scratch.grad_low_rank.as_slice()[0..d_w],
                            tr.xw.as_slice(),
                            &mut adam.attn.w1,
                            cfg,
                        );
                    }
                }
            }
            scratch.grad_x6.zero(); // d xw
            for row in 0..d_w {
                let g = scratch.grad_low_rank[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * c;
                for col in 0..c {
                    scratch.grad_x6[col] += block.attn.w1[off + col] * g;
                }
            }

            // A low-rank backward.
            for col in 0..c {
                let a = tr.a[col];
                scratch.grad_x5[col] *= a * (1.0 - a); // d a_pre
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.a0.as_mut_slice(),
                        scratch.grad_x5.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.a0.as_mut_slice(),
                            scratch.grad_x5.as_slice(),
                            &mut adam.attn.a0,
                            cfg,
                        );
                    }
                }
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.attn.a2.as_mut_slice(),
                        c,
                        d_a,
                        scratch.grad_x5.as_slice(),
                        &tr.a_hidden.as_slice()[0..d_a],
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.attn.a2.as_mut_slice(),
                            c,
                            d_a,
                            scratch.grad_x5.as_slice(),
                            &tr.a_hidden.as_slice()[0..d_a],
                            &mut adam.attn.a2,
                            cfg,
                        );
                    }
                }
            }
            scratch.grad_low_rank.zero();
            for row in 0..c {
                let g = scratch.grad_x5[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * d_a;
                for col in 0..d_a {
                    scratch.grad_low_rank[col] += block.attn.a2[off + col] * g;
                }
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.attn.a1.as_mut_slice(),
                        d_a,
                        c,
                        &scratch.grad_low_rank.as_slice()[0..d_a],
                        tr.xa.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.attn.a1.as_mut_slice(),
                            d_a,
                            c,
                            &scratch.grad_low_rank.as_slice()[0..d_a],
                            tr.xa.as_slice(),
                            &mut adam.attn.a1,
                            cfg,
                        );
                    }
                }
            }
            scratch.grad_x5.zero(); // d xa
            for row in 0..d_a {
                let g = scratch.grad_low_rank[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * c;
                for col in 0..c {
                    scratch.grad_x5[col] += block.attn.a1[off + col] * g;
                }
            }

            // G low-rank backward.
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.attn.g2.as_mut_slice(),
                        c,
                        d_g,
                        scratch.grad_saved.as_slice(),
                        &tr.g_hidden.as_slice()[0..d_g],
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.attn.g2.as_mut_slice(),
                            c,
                            d_g,
                            scratch.grad_saved.as_slice(),
                            &tr.g_hidden.as_slice()[0..d_g],
                            &mut adam.attn.g2,
                            cfg,
                        );
                    }
                }
            }
            scratch.grad_low_rank.zero();
            for row in 0..c {
                let g = scratch.grad_saved[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * d_g;
                for col in 0..d_g {
                    scratch.grad_low_rank[col] += block.attn.g2[off + col] * g;
                }
            }
            for col in 0..d_g {
                let sig = tr.g_hidden[col];
                scratch.grad_low_rank2[col] = scratch.grad_low_rank[col] * sig * (1.0 - sig);
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_outer_update(
                        block.attn.g1.as_mut_slice(),
                        d_g,
                        c,
                        &scratch.grad_low_rank2.as_slice()[0..d_g],
                        tr.xg.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_outer_update(
                            block.attn.g1.as_mut_slice(),
                            d_g,
                            c,
                            &scratch.grad_low_rank2.as_slice()[0..d_g],
                            tr.xg.as_slice(),
                            &mut adam.attn.g1,
                            cfg,
                        );
                    }
                }
            }
            scratch.grad_saved.zero(); // d xg
            for row in 0..d_g {
                let g = scratch.grad_low_rank2[row];
                if g == 0.0 {
                    continue;
                }
                let off = row * c;
                for col in 0..c {
                    scratch.grad_saved[col] += block.attn.g1[off + col] * g;
                }
            }

            // Token-shift backward for attention branches.
            scratch.grad_x3.zero(); // d attn_norm

            // x_r
            for col in 0..c {
                let g = scratch.grad_param[col];
                let mix = block.attn.x_r[col];
                let base = tr.attn_norm[col];
                let prev = tr.att_x_prev_old[col];
                scratch.grad_x3[col] += g * (1.0 - mix);
                scratch.grad_x2[col] = g * (prev - base);
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.x_r.as_mut_slice(),
                        scratch.grad_x2.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.x_r.as_mut_slice(),
                            scratch.grad_x2.as_slice(),
                            &mut adam.attn.x_r,
                            cfg,
                        );
                    }
                }
            }

            // x_w
            for col in 0..c {
                let g = scratch.grad_x6[col];
                let mix = block.attn.x_w[col];
                let base = tr.attn_norm[col];
                let prev = tr.att_x_prev_old[col];
                scratch.grad_x3[col] += g * (1.0 - mix);
                scratch.grad_x2[col] = g * (prev - base);
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.x_w.as_mut_slice(),
                        scratch.grad_x2.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.x_w.as_mut_slice(),
                            scratch.grad_x2.as_slice(),
                            &mut adam.attn.x_w,
                            cfg,
                        );
                    }
                }
            }

            // x_k
            for col in 0..c {
                let g = scratch.grad_param2[col];
                let mix = block.attn.x_k[col];
                let base = tr.attn_norm[col];
                let prev = tr.att_x_prev_old[col];
                scratch.grad_x3[col] += g * (1.0 - mix);
                scratch.grad_x2[col] = g * (prev - base);
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.x_k.as_mut_slice(),
                        scratch.grad_x2.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.x_k.as_mut_slice(),
                            scratch.grad_x2.as_slice(),
                            &mut adam.attn.x_k,
                            cfg,
                        );
                    }
                }
            }

            // x_v
            for col in 0..c {
                let g = scratch.grad_x4[col];
                let mix = block.attn.x_v[col];
                let base = tr.attn_norm[col];
                let prev = tr.att_x_prev_old[col];
                scratch.grad_x3[col] += g * (1.0 - mix);
                scratch.grad_x2[col] = g * (prev - base);
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.x_v.as_mut_slice(),
                        scratch.grad_x2.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.x_v.as_mut_slice(),
                            scratch.grad_x2.as_slice(),
                            &mut adam.attn.x_v,
                            cfg,
                        );
                    }
                }
            }

            // x_a
            for col in 0..c {
                let g = scratch.grad_x5[col];
                let mix = block.attn.x_a[col];
                let base = tr.attn_norm[col];
                let prev = tr.att_x_prev_old[col];
                scratch.grad_x3[col] += g * (1.0 - mix);
                scratch.grad_x2[col] = g * (prev - base);
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.x_a.as_mut_slice(),
                        scratch.grad_x2.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.x_a.as_mut_slice(),
                            scratch.grad_x2.as_slice(),
                            &mut adam.attn.x_a,
                            cfg,
                        );
                    }
                }
            }

            // x_g
            for col in 0..c {
                let g = scratch.grad_saved[col];
                let mix = block.attn.x_g[col];
                let base = tr.attn_norm[col];
                let prev = tr.att_x_prev_old[col];
                scratch.grad_x3[col] += g * (1.0 - mix);
                scratch.grad_x2[col] = g * (prev - base);
            }
            if scope.attn {
                match optimizer {
                    OptimizerKind::Sgd => sgd_vec_update(
                        block.attn.x_g.as_mut_slice(),
                        scratch.grad_x2.as_slice(),
                        lr,
                        clip,
                    ),
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn.x_g.as_mut_slice(),
                            scratch.grad_x2.as_slice(),
                            &mut adam.attn.x_g,
                            cfg,
                        );
                    }
                }
            }

            // attn norm backward.
            layer_norm_backward(
                tr.x_after_pre.as_slice(),
                block.attn_norm_w.as_slice(),
                scratch.grad_x3.as_slice(),
                self.cfg.layer_norm_eps,
                scratch.grad_x2.as_mut_slice(),
                scratch.grad_x4.as_mut_slice(),
                scratch.grad_x5.as_mut_slice(),
            );
            if scope.attn_norm {
                match optimizer {
                    OptimizerKind::Sgd => {
                        sgd_vec_update(
                            block.attn_norm_w.as_mut_slice(),
                            scratch.grad_x4.as_slice(),
                            lr,
                            clip,
                        );
                        sgd_vec_update(
                            block.attn_norm_b.as_mut_slice(),
                            scratch.grad_x5.as_slice(),
                            lr,
                            clip,
                        );
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_step.as_ref().expect("adam cfg initialized");
                        let adam =
                            &mut model_adam.as_mut().expect("adam state exists").blocks[layer_idx];
                        apply_adam_vec_update(
                            block.attn_norm_w.as_mut_slice(),
                            scratch.grad_x4.as_slice(),
                            &mut adam.attn_norm_w,
                            cfg,
                        );
                        apply_adam_vec_update(
                            block.attn_norm_b.as_mut_slice(),
                            scratch.grad_x5.as_slice(),
                            &mut adam.attn_norm_b,
                            cfg,
                        );
                    }
                }
            }
            for col in 0..c {
                scratch.grad_x[col] += scratch.grad_x2[col];
            }

            // pre_norm backward (layer 0 only).
            if layer_idx == 0
                && let (Some(w), Some(b)) = (&mut block.pre_norm_w, &mut block.pre_norm_b)
            {
                layer_norm_backward(
                    tr.x_in.as_slice(),
                    w.as_slice(),
                    scratch.grad_x.as_slice(),
                    self.cfg.layer_norm_eps,
                    scratch.grad_x2.as_mut_slice(),
                    scratch.grad_x3.as_mut_slice(),
                    scratch.grad_x4.as_mut_slice(),
                );
                if scope.pre_norm {
                    match optimizer {
                        OptimizerKind::Sgd => {
                            sgd_vec_update(w.as_mut_slice(), scratch.grad_x3.as_slice(), lr, clip);
                            sgd_vec_update(b.as_mut_slice(), scratch.grad_x4.as_slice(), lr, clip);
                        }
                        OptimizerKind::Adam => {
                            let cfg = adam_step.as_ref().expect("adam cfg initialized");
                            let adam = &mut model_adam.as_mut().expect("adam state exists").blocks
                                [layer_idx];
                            apply_adam_vec_update(
                                w.as_mut_slice(),
                                scratch.grad_x3.as_slice(),
                                adam.pre_norm_w.as_mut().expect("adam pre_norm_w"),
                                cfg,
                            );
                            apply_adam_vec_update(
                                b.as_mut_slice(),
                                scratch.grad_x4.as_slice(),
                                adam.pre_norm_b.as_mut().expect("adam pre_norm_b"),
                                cfg,
                            );
                        }
                    }
                }
                scratch.grad_x.copy_from_slice(scratch.grad_x2.as_slice());
            }
        }

        if scope.embed {
            let token_idx = scratch
                .train_token
                .min(self.cfg.vocab_size.saturating_sub(1));
            let off = token_idx * c;
            let row = &mut self.embeddings.as_mut_slice()[off..off + c];
            match optimizer {
                OptimizerKind::Sgd => {
                    sgd_vec_update(row, scratch.grad_x.as_slice(), lr, clip);
                }
                OptimizerKind::Adam => {
                    let cfg = adam_step.as_ref().expect("adam cfg initialized");
                    let adam = model_adam.as_mut().expect("adam state exists");
                    let m = &mut adam.embeddings.m.as_mut_slice()[off..off + c];
                    let v = &mut adam.embeddings.v.as_mut_slice()[off..off + c];
                    apply_adam_vec_update_raw(row, scratch.grad_x.as_slice(), m, v, cfg);
                }
            }
        }
        Ok(())
    }

    /// Forward pass for a single token.
    /// Returns logits for next token prediction.
    #[inline(never)]
    pub fn forward<'a>(
        &'a self,
        scratch: &'a mut ScratchBuffers,
        token: u32,
        state: &mut State,
    ) -> &'a [f32] {
        let mut sink = NullProfiler;
        self.forward_with_sink(scratch, token, state, &mut sink)
    }

    /// Forward pass that records per-layer timings through a custom sink.
    #[inline(never)]
    pub fn forward_with_profiler<'a, S: ProfilerSink>(
        &'a self,
        scratch: &'a mut ScratchBuffers,
        token: u32,
        state: &mut State,
        profiler: &mut S,
    ) -> &'a [f32] {
        self.forward_with_sink(scratch, token, state, profiler)
    }

    #[inline(never)]
    fn forward_with_sink<'a, S: ProfilerSink>(
        &'a self,
        scratch: &'a mut ScratchBuffers,
        token: u32,
        state: &mut State,
        profiler: &mut S,
    ) -> &'a [f32] {
        let c = self.cfg.hidden_size;
        let _h = self.cfg.num_heads;
        let _n = self.cfg.head_dim;
        let num_layers = self.cfg.num_layers;
        let token_idx = (token as usize).min(self.cfg.vocab_size.saturating_sub(1));

        // Get token embedding
        let emb_offset = token_idx * c;
        let emb_slice = &self.embeddings.as_slice()[emb_offset..emb_offset + c];
        scratch.x.as_mut_slice().copy_from_slice(emb_slice);
        if scratch.capture_train_trace {
            scratch.train_token = token_idx;
            scratch.train_trace_valid = true;
        } else {
            scratch.train_trace_valid = false;
        }

        profiler.begin_token();

        unsafe {
            // Process each layer (using index to avoid borrow conflicts)
            for layer_idx in 0..num_layers {
                if scratch.capture_train_trace {
                    scratch.train_trace_layers[layer_idx]
                        .x_in
                        .copy_from(&scratch.x);
                }
                // Pre-norm (layer 0 only)
                if let (Some(w), Some(b)) = (
                    &self.blocks[layer_idx].pre_norm_w,
                    &self.blocks[layer_idx].pre_norm_b,
                ) {
                    kernel::layer_norm_avx(
                        scratch.x.as_ptr(),
                        w.as_ptr(),
                        b.as_ptr(),
                        scratch.x.as_mut_ptr(),
                        c,
                        self.cfg.layer_norm_eps,
                    );
                }
                if scratch.capture_train_trace {
                    scratch.train_trace_layers[layer_idx]
                        .x_after_pre
                        .copy_from(&scratch.x);
                }

                // Attention norm
                kernel::layer_norm_avx(
                    scratch.x.as_ptr(),
                    self.blocks[layer_idx].attn_norm_w.as_ptr(),
                    self.blocks[layer_idx].attn_norm_b.as_ptr(),
                    scratch.x_normed.as_mut_ptr(),
                    c,
                    self.cfg.layer_norm_eps,
                );
                if scratch.capture_train_trace {
                    scratch.train_trace_layers[layer_idx]
                        .attn_norm
                        .copy_from(&scratch.x_normed);
                }

                let attn_start = Instant::now();
                let trace_ptr = if scratch.capture_train_trace {
                    Some(&mut scratch.train_trace_layers[layer_idx] as *mut LayerTrainTrace)
                } else {
                    None
                };
                self.attention_forward_impl(scratch, layer_idx, state, trace_ptr);
                profiler.record_attention(layer_idx, attn_start.elapsed());

                // Add attention residual: x = x + att_out
                kernel::add_avx(
                    scratch.x.as_ptr(),
                    scratch.att_out.as_ptr(),
                    scratch.x.as_mut_ptr(),
                    c,
                );
                if scratch.capture_train_trace {
                    scratch.train_trace_layers[layer_idx]
                        .x_after_attn
                        .copy_from(&scratch.x);
                }

                // FFN norm
                kernel::layer_norm_avx(
                    scratch.x.as_ptr(),
                    self.blocks[layer_idx].ffn_norm_w.as_ptr(),
                    self.blocks[layer_idx].ffn_norm_b.as_ptr(),
                    scratch.x_normed.as_mut_ptr(),
                    c,
                    self.cfg.layer_norm_eps,
                );
                if scratch.capture_train_trace {
                    scratch.train_trace_layers[layer_idx]
                        .ffn_norm
                        .copy_from(&scratch.x_normed);
                }

                let ffn_start = Instant::now();
                self.ffn_forward_impl(scratch, layer_idx, &mut state.layers[layer_idx], trace_ptr);
                profiler.record_ffn(layer_idx, ffn_start.elapsed());

                // Add FFN residual: x = x + ffn_out
                kernel::add_avx(
                    scratch.x.as_ptr(),
                    scratch.ffn_out.as_ptr(),
                    scratch.x.as_mut_ptr(),
                    c,
                );
                if scratch.capture_train_trace {
                    scratch.train_trace_layers[layer_idx]
                        .x_out
                        .copy_from(&scratch.x);
                }
            }

            // Output norm
            kernel::layer_norm_avx(
                scratch.x.as_ptr(),
                self.ln_out_w.as_ptr(),
                self.ln_out_b.as_ptr(),
                scratch.x_normed.as_mut_ptr(),
                c,
                self.cfg.layer_norm_eps,
            );

            // LM head: logits = x @ lm_head.T
            kernel::gemv_avx(
                self.lm_head.as_ptr(),
                scratch.x_normed.as_ptr(),
                scratch.logits.as_mut_ptr(),
                self.cfg.vocab_size,
                c,
            );
        }
        if scratch.capture_train_trace {
            scratch.train_v_first.copy_from(&state.v_first);
        }

        scratch.logits.as_slice()
    }

    #[inline(always)]
    unsafe fn attention_forward_impl(
        &self,
        scratch: &mut ScratchBuffers,
        layer_idx: usize,
        state: &mut State,
        trace: Option<*mut LayerTrainTrace>,
    ) {
        let attn = &self.blocks[layer_idx].attn;
        let layer_state = &mut state.layers[layer_idx];
        let c = self.cfg.hidden_size;
        let h = self.cfg.num_heads;
        let n = self.cfg.head_dim;
        let d_w = self.cfg.decay_low_rank;
        let d_a = self.cfg.a_low_rank;
        let d_g = self.cfg.g_low_rank;
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.att_x_prev_old.copy_from(&layer_state.att_x_prev);
            tr.att_state_old.copy_from(&layer_state.att_state);
        }

        kernel::token_shift_multi6_avx(
            scratch.x_normed.as_ptr(),
            layer_state.att_x_prev.as_ptr(),
            attn.x_r.as_ptr(),
            attn.x_w.as_ptr(),
            attn.x_k.as_ptr(),
            attn.x_v.as_ptr(),
            attn.x_a.as_ptr(),
            attn.x_g.as_ptr(),
            scratch.xr.as_mut_ptr(),
            scratch.xw.as_mut_ptr(),
            scratch.xk.as_mut_ptr(),
            scratch.xv.as_mut_ptr(),
            scratch.xa.as_mut_ptr(),
            scratch.xg.as_mut_ptr(),
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.xr.copy_from(&scratch.xr);
            tr.xw.copy_from(&scratch.xw);
            tr.xk.copy_from(&scratch.xk);
            tr.xv.copy_from(&scratch.xv);
            tr.xa.copy_from(&scratch.xa);
            tr.xg.copy_from(&scratch.xg);
        }

        // Update prev state for next token
        kernel::copy(
            scratch.x_normed.as_ptr(),
            layer_state.att_x_prev.as_mut_ptr(),
            c,
        );

        // r/k/v projections from packed matrix (sequential for better cache)
        // Packed layout: [r_proj (C*C), k_proj (C*C), v_proj (C*C)]
        let proj_size = c * c;
        kernel::gemv_avx(
            attn.rkv_proj.as_ptr(),
            scratch.xr.as_ptr(),
            scratch.r.as_mut_ptr(),
            c,
            c,
        );
        kernel::gemv_avx(
            attn.rkv_proj.as_ptr().add(proj_size),
            scratch.xk.as_ptr(),
            scratch.k.as_mut_ptr(),
            c,
            c,
        );
        kernel::gemv_avx(
            attn.rkv_proj.as_ptr().add(2 * proj_size),
            scratch.xv.as_ptr(),
            scratch.v.as_mut_ptr(),
            c,
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.r.copy_from(&scratch.r);
            tr.k_pre.copy_from(&scratch.k);
            tr.v_pre.copy_from(&scratch.v);
        }

        // w decay: w = exp(-sigmoid(tanh(xw @ w1) @ w2 + w0) / sqrt(e))
        // Step 1: tmp = xw @ w1.T (D_w output)
        kernel::gemv_avx(
            attn.w1.as_ptr(),
            scratch.xw.as_ptr(),
            scratch.w_lora_tmp.as_mut_ptr(),
            d_w,
            c,
        );
        // Step 2: tanh
        kernel::tanh_avx(
            scratch.w_lora_tmp.as_ptr(),
            scratch.w_lora_tmp.as_mut_ptr(),
            d_w,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.w_hidden.as_mut_slice()[0..d_w]
                .copy_from_slice(&scratch.w_lora_tmp.as_slice()[0..d_w]);
        }
        // Step 3: tmp @ w2.T + w0
        kernel::gemv_avx(
            attn.w2.as_ptr(),
            scratch.w_lora_tmp.as_ptr(),
            scratch.w_decay.as_mut_ptr(),
            c,
            d_w,
        );
        // Add bias w0
        kernel::add_avx(
            scratch.w_decay.as_ptr(),
            attn.w0.as_ptr(),
            scratch.w_decay.as_mut_ptr(),
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.w_pre.copy_from(&scratch.w_decay);
        }
        // Step 4: exp(-sigmoid(x) / sqrt(e))
        let inv_sqrt_e = 1.0 / std::f32::consts::E.sqrt();
        kernel::sigmoid_avx(scratch.w_decay.as_ptr(), scratch.w_decay.as_mut_ptr(), c);
        kernel::exp_neg_scaled_inplace(scratch.w_decay.as_mut_ptr(), inv_sqrt_e, c);
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.w_decay.copy_from(&scratch.w_decay);
        }

        // a = sigmoid(xa @ a1.T @ a2.T + a0)
        kernel::gemv_avx(
            attn.a1.as_ptr(),
            scratch.xa.as_ptr(),
            scratch.w_lora_tmp.as_mut_ptr(),
            d_a,
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.a_hidden.as_mut_slice()[0..d_a]
                .copy_from_slice(&scratch.w_lora_tmp.as_slice()[0..d_a]);
        }
        kernel::gemv_avx(
            attn.a2.as_ptr(),
            scratch.w_lora_tmp.as_ptr(),
            scratch.a.as_mut_ptr(),
            c,
            d_a,
        );
        kernel::add_avx(
            scratch.a.as_ptr(),
            attn.a0.as_ptr(),
            scratch.a.as_mut_ptr(),
            c,
        );
        kernel::sigmoid_avx(scratch.a.as_ptr(), scratch.a.as_mut_ptr(), c);
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.a.copy_from(&scratch.a);
        }

        // g = sigmoid(xg @ g1.T) @ g2.T
        kernel::gemv_avx(
            attn.g1.as_ptr(),
            scratch.xg.as_ptr(),
            scratch.w_lora_tmp.as_mut_ptr(),
            d_g,
            c,
        );
        kernel::sigmoid_avx(
            scratch.w_lora_tmp.as_ptr(),
            scratch.w_lora_tmp.as_mut_ptr(),
            d_g,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.g_hidden.as_mut_slice()[0..d_g]
                .copy_from_slice(&scratch.w_lora_tmp.as_slice()[0..d_g]);
        }
        kernel::gemv_avx(
            attn.g2.as_ptr(),
            scratch.w_lora_tmp.as_ptr(),
            scratch.g.as_mut_ptr(),
            c,
            d_g,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.g.copy_from(&scratch.g);
        }

        // Value residual (layer > 0)
        if layer_idx == 0 {
            // Copy v to v_first buffer (no allocation)
            state.v_first.copy_from(&scratch.v);
            state.v_first_set = true;
            if let Some(tr_ptr) = trace {
                let tr = &mut *tr_ptr;
                tr.uses_v_residual = false;
                tr.nu.zero();
                tr.v.copy_from(&scratch.v);
            }
        } else if state.v_first_set
            && let (Some(v1), Some(v2), Some(v0)) = (&attn.v1, &attn.v2, &attn.v0)
        {
            let d_v = self.cfg.v_low_rank;
            // nu = sigmoid(xv @ v1.T @ v2.T + v0)
            kernel::gemv_avx(
                v1.as_ptr(),
                scratch.xv.as_ptr(),
                scratch.w_lora_tmp.as_mut_ptr(),
                d_v,
                c,
            );
            if let Some(tr_ptr) = trace {
                let tr = &mut *tr_ptr;
                tr.v_hidden.as_mut_slice()[0..d_v]
                    .copy_from_slice(&scratch.w_lora_tmp.as_slice()[0..d_v]);
            }
            kernel::gemv_avx(
                v2.as_ptr(),
                scratch.w_lora_tmp.as_ptr(),
                scratch.att_out.as_mut_ptr(), // reuse as temp
                c,
                d_v,
            );
            kernel::add_avx(
                scratch.att_out.as_ptr(),
                v0.as_ptr(),
                scratch.att_out.as_mut_ptr(),
                c,
            );
            kernel::sigmoid_avx(scratch.att_out.as_ptr(), scratch.att_out.as_mut_ptr(), c);
            if let Some(tr_ptr) = trace {
                let tr = &mut *tr_ptr;
                tr.uses_v_residual = true;
                tr.nu.copy_from(&scratch.att_out);
            }
            // v = v + (v_first - v) * nu
            for i in 0..c {
                let nu = scratch.att_out[i];
                scratch.v[i] += (state.v_first[i] - scratch.v[i]) * nu;
            }
            if let Some(tr_ptr) = trace {
                let tr = &mut *tr_ptr;
                tr.v.copy_from(&scratch.v);
            }
        } else if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.uses_v_residual = false;
            tr.nu.zero();
            tr.v.copy_from(&scratch.v);
        }

        // kk = k * k_k, then L2 normalize per head
        kernel::mul_avx(
            scratch.k.as_ptr(),
            attn.k_k.as_ptr(),
            scratch.kk.as_mut_ptr(),
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.kk_pre.copy_from(&scratch.kk);
        }
        // Normalize per head
        for head in 0..h {
            let offset = head * n;
            kernel::l2_normalize_avx(
                scratch.kk.as_ptr().add(offset),
                scratch.kk.as_mut_ptr().add(offset),
                n,
                1e-12,
            );
        }
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.kk.copy_from(&scratch.kk);
        }

        // k = k * (1 + (a - 1) * k_a)
        for i in 0..c {
            let scale = 1.0 + (scratch.a[i] - 1.0) * attn.k_a[i];
            scratch.k[i] *= scale;
        }
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.k.copy_from(&scratch.k);
        }

        // WKV state update: S = S*w.T - S@kk*(kk*a).T + v*k.T; y = S@r
        kernel::rwkv7_wkv_update_avx(
            layer_state.att_state.as_mut_ptr(),
            scratch.w_decay.as_ptr(),
            scratch.k.as_ptr(),
            scratch.v.as_ptr(),
            scratch.kk.as_ptr(),
            scratch.a.as_ptr(),
            scratch.r.as_ptr(),
            scratch.y.as_mut_ptr(),
            h,
            n,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.y_wkv.copy_from(&scratch.y);
        }

        // Group norm
        kernel::group_norm_avx(
            scratch.y.as_ptr(),
            attn.g_norm_w.as_ptr(),
            attn.g_norm_b.as_ptr(),
            scratch.y.as_mut_ptr(),
            h,
            n,
            self.cfg.group_norm_eps,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.y_gn.copy_from(&scratch.y);
        }

        // Add head-qk term: y += ((r * k * r_k).sum_per_head) * v
        for head in 0..h {
            let offset = head * n;
            let mut alpha = 0.0f32;
            for j in 0..n {
                alpha += scratch.r[offset + j] * scratch.k[offset + j] * attn.r_k[head * n + j];
            }
            if let Some(tr_ptr) = trace {
                let tr = &mut *tr_ptr;
                tr.alpha[head] = alpha;
            }
            for j in 0..n {
                scratch.y[offset + j] += alpha * scratch.v[offset + j];
            }
        }
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.y_head.copy_from(&scratch.y);
        }

        // Apply gate: y = y * g
        kernel::mul_avx(
            scratch.y.as_ptr(),
            scratch.g.as_ptr(),
            scratch.y.as_mut_ptr(),
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.y_gate.copy_from(&scratch.y);
        }

        // Output projection: att_out = o_proj @ y
        kernel::gemv_avx(
            attn.o_proj.as_ptr(),
            scratch.y.as_ptr(),
            scratch.att_out.as_mut_ptr(),
            c,
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.att_out.copy_from(&scratch.att_out);
        }
    }

    #[inline(always)]
    unsafe fn ffn_forward_impl(
        &self,
        scratch: &mut ScratchBuffers,
        layer_idx: usize,
        layer_state: &mut LayerState,
        trace: Option<*mut LayerTrainTrace>,
    ) {
        let ffn = &self.blocks[layer_idx].ffn;
        let c = self.cfg.hidden_size;
        let i = self.cfg.intermediate_size;
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.ffn_x_prev_old.copy_from(&layer_state.ffn_x_prev);
        }

        // Token shift: xk = x_normed + x_k * (prev - x_normed)
        kernel::token_shift_avx(
            scratch.x_normed.as_ptr(),
            layer_state.ffn_x_prev.as_ptr(),
            ffn.x_k.as_ptr(),
            scratch.xk.as_mut_ptr(),
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.ffn_xk.copy_from(&scratch.xk);
        }

        // Update prev state
        kernel::copy(
            scratch.x_normed.as_ptr(),
            layer_state.ffn_x_prev.as_mut_ptr(),
            c,
        );

        // k = relu(xk @ key_w.T)^2
        kernel::gemv_avx(
            ffn.key_w.as_ptr(),
            scratch.xk.as_ptr(),
            scratch.ffn_k.as_mut_ptr(),
            i,
            c,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.ffn_pre.copy_from(&scratch.ffn_k);
        }
        kernel::relu_squared_avx(scratch.ffn_k.as_ptr(), scratch.ffn_k.as_mut_ptr(), i);
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.ffn_k.copy_from(&scratch.ffn_k);
        }

        // ffn_out = k @ value_w.T
        kernel::gemv_avx(
            ffn.value_w.as_ptr(),
            scratch.ffn_k.as_ptr(),
            scratch.ffn_out.as_mut_ptr(),
            c,
            i,
        );
        if let Some(tr_ptr) = trace {
            let tr = &mut *tr_ptr;
            tr.ffn_out.copy_from(&scratch.ffn_out);
        }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn layer_norm_backward(
    input: &[f32],
    weight: &[f32],
    grad_out: &[f32],
    eps: f32,
    grad_input: &mut [f32],
    grad_weight: &mut [f32],
    grad_bias: &mut [f32],
) {
    let n = input
        .len()
        .min(weight.len())
        .min(grad_out.len())
        .min(grad_input.len())
        .min(grad_weight.len())
        .min(grad_bias.len());
    if n == 0 {
        return;
    }
    let nf = n as f32;
    let mut mean = 0.0f32;
    for &x in &input[0..n] {
        mean += x;
    }
    mean /= nf;
    let mut var = 0.0f32;
    for &x in &input[0..n] {
        let d = x - mean;
        var += d * d;
    }
    var /= nf;
    let inv_std = (var + eps).sqrt().recip();
    let mut sum_gw = 0.0f32;
    let mut sum_gw_xhat = 0.0f32;
    for i in 0..n {
        let xhat = (input[i] - mean) * inv_std;
        let gw = grad_out[i] * weight[i];
        grad_weight[i] = grad_out[i] * xhat;
        grad_bias[i] = grad_out[i];
        sum_gw += gw;
        sum_gw_xhat += gw * xhat;
    }
    for i in 0..n {
        let xhat = (input[i] - mean) * inv_std;
        let gw = grad_out[i] * weight[i];
        grad_input[i] = (gw * nf - sum_gw - xhat * sum_gw_xhat) * inv_std / nf;
    }
}

#[allow(clippy::too_many_arguments)]
fn group_norm_backward(
    input: &[f32],
    weight: &[f32],
    grad_out: &[f32],
    num_groups: usize,
    group_size: usize,
    eps: f32,
    grad_input: &mut [f32],
    grad_weight: &mut [f32],
    grad_bias: &mut [f32],
) {
    let c = input
        .len()
        .min(weight.len())
        .min(grad_out.len())
        .min(grad_input.len())
        .min(grad_weight.len())
        .min(grad_bias.len());
    if c == 0 || num_groups == 0 || group_size == 0 {
        return;
    }
    grad_input[0..c].fill(0.0);
    grad_weight[0..c].fill(0.0);
    grad_bias[0..c].fill(0.0);
    let g = num_groups.min(c / group_size);
    let n = group_size as f32;
    for group in 0..g {
        let off = group * group_size;
        let end = (off + group_size).min(c);
        let len = end - off;
        if len == 0 {
            continue;
        }
        let mut mean = 0.0f32;
        for idx in off..end {
            mean += input[idx];
        }
        mean /= len as f32;
        let mut var = 0.0f32;
        for idx in off..end {
            let d = input[idx] - mean;
            var += d * d;
        }
        var /= len as f32;
        let inv_std = (var + eps).sqrt().recip();
        let mut sum_gw = 0.0f32;
        let mut sum_gw_xhat = 0.0f32;
        for idx in off..end {
            let xhat = (input[idx] - mean) * inv_std;
            let gw = grad_out[idx] * weight[idx];
            grad_weight[idx] += grad_out[idx] * xhat;
            grad_bias[idx] += grad_out[idx];
            sum_gw += gw;
            sum_gw_xhat += gw * xhat;
        }
        for idx in off..end {
            let xhat = (input[idx] - mean) * inv_std;
            let gw = grad_out[idx] * weight[idx];
            grad_input[idx] = (gw * n - sum_gw - xhat * sum_gw_xhat) * inv_std / n;
        }
    }
}

fn l2_normalize_backward(
    x: &[f32],
    y: &[f32],
    grad_out: &[f32],
    min_norm: f32,
    grad_input: &mut [f32],
) {
    let n = x
        .len()
        .min(y.len())
        .min(grad_out.len())
        .min(grad_input.len());
    if n == 0 {
        return;
    }
    let mut norm_sq = 0.0f32;
    for &v in &x[0..n] {
        norm_sq += v * v;
    }
    let norm_raw = norm_sq.sqrt();
    if norm_raw <= min_norm {
        let inv = min_norm.recip();
        for i in 0..n {
            grad_input[i] = grad_out[i] * inv;
        }
        return;
    }
    let norm = norm_raw;
    let mut dot = 0.0f32;
    for i in 0..n {
        dot += grad_out[i] * y[i];
    }
    let inv = norm.recip();
    for i in 0..n {
        grad_input[i] = (grad_out[i] - y[i] * dot) * inv;
    }
}

#[inline(always)]
fn sgd_vec_update(param: &mut [f32], grad: &[f32], lr: f32, clip: f32) {
    let n = param.len().min(grad.len());
    if n == 0 {
        return;
    }
    if clip > 0.0 {
        for i in 0..n {
            param[i] += lr * grad[i].clamp(-clip, clip);
        }
    } else {
        for i in 0..n {
            param[i] += lr * grad[i];
        }
    }
}

#[inline(always)]
fn sgd_outer_update(
    param: &mut [f32],
    rows: usize,
    cols: usize,
    left: &[f32],
    right: &[f32],
    lr: f32,
    clip: f32,
) {
    let rows = rows.min(left.len());
    let cols = cols.min(right.len());
    let n = param.len();
    if rows == 0 || cols == 0 || n == 0 {
        return;
    }
    for r in 0..rows {
        let g = left[r];
        let off = r * cols;
        if off >= n {
            break;
        }
        let row_cols = cols.min(n - off);
        if clip > 0.0 {
            for c in 0..row_cols {
                param[off + c] += lr * (g * right[c]).clamp(-clip, clip);
            }
        } else {
            for c in 0..row_cols {
                param[off + c] += lr * g * right[c];
            }
        }
    }
}

#[inline(always)]
fn apply_adam_vec_update(
    param: &mut [f32],
    grad: &[f32],
    adam: &mut AdamTensorState,
    step: &AdamStep,
) {
    let n = param
        .len()
        .min(grad.len())
        .min(adam.m.len())
        .min(adam.v.len());
    if n == 0 {
        return;
    }
    apply_adam_vec_update_raw(
        &mut param[0..n],
        &grad[0..n],
        &mut adam.m.as_mut_slice()[0..n],
        &mut adam.v.as_mut_slice()[0..n],
        step,
    );
}

#[inline(always)]
fn apply_adam_vec_update_raw(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: &AdamStep,
) {
    let n = param.len().min(grad.len()).min(m.len()).min(v.len());
    if n == 0 {
        return;
    }
    let b1 = step.b1;
    let b2 = step.b2;
    let one_b1 = 1.0 - b1;
    let one_b2 = 1.0 - b2;
    let inv_bc1 = 1.0 / step.bias_corr1;
    let inv_bc2 = 1.0 / step.bias_corr2;
    let do_clip = step.clip > 0.0;
    let clip = step.clip;
    if do_clip {
        for idx in 0..n {
            let g = grad[idx].clamp(-clip, clip);
            let mm = b1 * m[idx] + one_b1 * g;
            let vv = b2 * v[idx] + one_b2 * g * g;
            m[idx] = mm;
            v[idx] = vv;
            let m_hat = mm * inv_bc1;
            let v_hat = vv * inv_bc2;
            param[idx] += step.lr * m_hat / (v_hat.sqrt() + step.eps);
        }
        return;
    }
    let mut idx = 0usize;
    unsafe {
        let b1v = f32x8::splat(b1);
        let b2v = f32x8::splat(b2);
        let one_b1v = f32x8::splat(one_b1);
        let one_b2v = f32x8::splat(one_b2);
        let inv_bc1v = f32x8::splat(inv_bc1);
        let inv_bc2v = f32x8::splat(inv_bc2);
        let lrv = f32x8::splat(step.lr);
        let epsv = f32x8::splat(step.eps);
        while idx + 8 <= n {
            let gv = grad.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
            let mv = m.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
            let vv = v.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
            let mm = mv * b1v + gv * one_b1v;
            let vv2 = vv * b2v + (gv * gv) * one_b2v;
            m.as_mut_ptr().add(idx).cast::<f32x8>().write_unaligned(mm);
            v.as_mut_ptr().add(idx).cast::<f32x8>().write_unaligned(vv2);
            let pv = param.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
            let upd = ((mm * inv_bc1v) / ((vv2 * inv_bc2v).sqrt() + epsv)) * lrv;
            param
                .as_mut_ptr()
                .add(idx)
                .cast::<f32x8>()
                .write_unaligned(pv + upd);
            idx += 8;
        }
    }
    while idx < n {
        let g = grad[idx];
        let mm = b1 * m[idx] + one_b1 * g;
        let vv = b2 * v[idx] + one_b2 * g * g;
        m[idx] = mm;
        v[idx] = vv;
        let m_hat = mm * inv_bc1;
        let v_hat = vv * inv_bc2;
        param[idx] += step.lr * m_hat / (v_hat.sqrt() + step.eps);
        idx += 1;
    }
}

#[inline(always)]
fn apply_adam_outer_update(
    param: &mut [f32],
    rows: usize,
    cols: usize,
    left: &[f32],
    right: &[f32],
    adam: &mut AdamTensorState,
    step: &AdamStep,
) {
    let n = param.len().min(adam.m.len()).min(adam.v.len());
    if n == 0 {
        return;
    }
    apply_adam_outer_update_raw(
        &mut param[0..n],
        rows,
        cols,
        left,
        right,
        &mut adam.m.as_mut_slice()[0..n],
        &mut adam.v.as_mut_slice()[0..n],
        step,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn apply_adam_outer_update_raw(
    param: &mut [f32],
    rows: usize,
    cols: usize,
    left: &[f32],
    right: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: &AdamStep,
) {
    let rows = rows.min(left.len());
    let cols = cols.min(right.len());
    let n = param.len().min(m.len()).min(v.len());
    if rows == 0 || cols == 0 || n == 0 {
        return;
    }
    let b1 = step.b1;
    let b2 = step.b2;
    let one_b1 = 1.0 - b1;
    let one_b2 = 1.0 - b2;
    let inv_bc1 = 1.0 / step.bias_corr1;
    let inv_bc2 = 1.0 / step.bias_corr2;
    let do_clip = step.clip > 0.0;
    let clip = step.clip;
    let b1v = f32x8::splat(b1);
    let b2v = f32x8::splat(b2);
    let one_b1v = f32x8::splat(one_b1);
    let one_b2v = f32x8::splat(one_b2);
    let inv_bc1v = f32x8::splat(inv_bc1);
    let inv_bc2v = f32x8::splat(inv_bc2);
    let epsv = f32x8::splat(step.eps);
    let lrv = f32x8::splat(step.lr);
    for row in 0..rows {
        let g_row = left[row];
        let off = row * cols;
        if off >= n {
            break;
        }
        let row_cols = (n - off).min(cols);
        if do_clip {
            for col in 0..row_cols {
                let idx = off + col;
                let g = (g_row * right[col]).clamp(-clip, clip);
                let mm = b1 * m[idx] + one_b1 * g;
                let vv = b2 * v[idx] + one_b2 * g * g;
                m[idx] = mm;
                v[idx] = vv;
                let m_hat = mm * inv_bc1;
                let v_hat = vv * inv_bc2;
                param[idx] += step.lr * m_hat / (v_hat.sqrt() + step.eps);
            }
            continue;
        }
        let mut col = 0usize;
        unsafe {
            let g8 = f32x8::splat(g_row);
            while col + 8 <= row_cols {
                let idx = off + col;
                let rv = right.as_ptr().add(col).cast::<f32x8>().read_unaligned();
                let gv = g8 * rv;
                let mv = m.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
                let vv = v.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
                let mm = mv * b1v + gv * one_b1v;
                let vv2 = vv * b2v + (gv * gv) * one_b2v;
                m.as_mut_ptr().add(idx).cast::<f32x8>().write_unaligned(mm);
                v.as_mut_ptr().add(idx).cast::<f32x8>().write_unaligned(vv2);
                let pv = param.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
                let upd = ((mm * inv_bc1v) / ((vv2 * inv_bc2v).sqrt() + epsv)) * lrv;
                param
                    .as_mut_ptr()
                    .add(idx)
                    .cast::<f32x8>()
                    .write_unaligned(pv + upd);
                col += 8;
            }
        }
        while col < row_cols {
            let idx = off + col;
            let g = g_row * right[col];
            let mm = b1 * m[idx] + one_b1 * g;
            let vv = b2 * v[idx] + one_b2 * g * g;
            m[idx] = mm;
            v[idx] = vv;
            let m_hat = mm * inv_bc1;
            let v_hat = vv * inv_bc2;
            param[idx] += step.lr * m_hat / (v_hat.sqrt() + step.eps);
            col += 1;
        }
    }
}

struct RwkvRng {
    state: u64,
}

impl RwkvRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u32() as f32;
        v * (1.0 / (u32::MAX as f32))
    }
}

#[inline]
fn init_uniform(t: &mut Tensor1D, rng: &mut RwkvRng, scale: f32) {
    let s = t.as_mut_slice();
    for v in s {
        let r = rng.next_f32() - 0.5;
        *v = r * 2.0 * scale;
    }
}

#[inline]
fn init_centered(t: &mut Tensor1D, rng: &mut RwkvRng, center: f32, scale: f32) {
    let s = t.as_mut_slice();
    for v in s {
        let r = rng.next_f32() - 0.5;
        *v = center + r * 2.0 * scale;
    }
}

#[inline]
fn init_const(t: &mut Tensor1D, value: f32) {
    t.as_mut_slice().fill(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weighted_checksum(data: &[f32]) -> f64 {
        data.iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 + 1.0) * (v as f64))
            .sum()
    }

    #[test]
    fn test_config_default() {
        let cfg = Config::default();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.hidden_size, 256);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn test_forward_deterministic_snapshot() {
        let cfg = Config {
            vocab_size: 256,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 1,
            head_dim: 64,
            intermediate_size: 128,
            layer_norm_eps: 1e-5,
            group_norm_eps: 64e-5,
            decay_low_rank: 16,
            a_low_rank: 16,
            v_low_rank: 16,
            g_low_rank: 32,
        };
        cfg.validate().expect("valid test config");

        let model = Model::new_random(cfg.clone(), 0x1234_5678_9ABC_DEF0).expect("random model");
        let mut state = model.new_state();
        let mut scratch = ScratchBuffers::new(&cfg);
        let tokens = [0u32, 1, 7, 42, 255, 3, 128, 64, 17, 99];

        let mut probes = Vec::new();
        let mut last_logits = vec![0.0; 8];

        for &token in &tokens {
            let logits = model.forward(&mut scratch, token, &mut state);
            probes.push(logits[0]);
            probes.push(logits[1]);
            probes.push(logits[2]);
            probes.push(logits[42]);
            probes.push(logits[127]);
            probes.push(logits[255]);
            last_logits.copy_from_slice(&logits[0..8]);
        }

        let probe_checksum = weighted_checksum(&probes);
        let last_logits_checksum = weighted_checksum(&last_logits);
        let state_att_checksum = weighted_checksum(state.layers[0].att_state.as_slice());
        let state_prev_checksum = weighted_checksum(state.layers[1].att_x_prev.as_slice());
        let v_first_checksum = weighted_checksum(state.v_first.as_slice());

        let expected_probe_checksum = 25.674_967_924_598_604_f64;
        let expected_last_logits_checksum = 0.679_873_816_668_987_3_f64;
        let expected_state_att_checksum = 129.962_464_237_222_32_f64;
        let expected_state_prev_checksum = -231.326_208_570_972_08_f64;
        let expected_v_first_checksum = -1.921_361_377_462_744_7_f64;

        let tol = 2e-4_f64;
        assert!(
            (probe_checksum - expected_probe_checksum).abs() <= tol,
            "probe_checksum={probe_checksum}"
        );
        assert!(
            (last_logits_checksum - expected_last_logits_checksum).abs() <= tol,
            "last_logits_checksum={last_logits_checksum}"
        );
        assert!(
            (state_att_checksum - expected_state_att_checksum).abs() <= tol,
            "state_att_checksum={state_att_checksum}"
        );
        assert!(
            (state_prev_checksum - expected_state_prev_checksum).abs() <= tol,
            "state_prev_checksum={state_prev_checksum}"
        );
        assert!(
            (v_first_checksum - expected_v_first_checksum).abs() <= tol,
            "v_first_checksum={v_first_checksum}"
        );
    }
}
