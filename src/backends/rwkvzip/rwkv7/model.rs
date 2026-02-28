//! RWKV7 model implementation with portable SIMD-optimized inference.
//!
//! This is a high-performance implementation built on `wide` kernels.
//! Single-token inference is the primary use case (streaming compression).

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

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
struct FfnWeights {
    x_k: Tensor1D,     // (C,) time shift mix
    key_w: Tensor1D,   // (C, I) -> relu(x @ W)^2
    value_w: Tensor1D, // (I, C)
}

/// Weights for a single block.
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

/// Pre-allocated scratch buffers to avoid allocations in hot path.
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
}

impl ScratchBuffers {
    /// Allocate reusable per-token scratch buffers sized for `cfg`.
    pub fn new(cfg: &Config) -> Self {
        let c = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let d_w = cfg.decay_low_rank;

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
            w_lora_tmp: Tensor1D::zeros(d_w.max(64)), // Max of all low-rank dims
            w_decay: Tensor1D::zeros(c),
            a: Tensor1D::zeros(c),
            g: Tensor1D::zeros(c),
            kk: Tensor1D::zeros(c),
            y: Tensor1D::zeros(c),
            att_out: Tensor1D::zeros(c),
            ffn_k: Tensor1D::zeros(i),
            ffn_out: Tensor1D::zeros(c),
            logits: Tensor1D::zeros(v),
        }
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

    /// Get model configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Create new state for this model.
    pub fn new_state(&self) -> State {
        State::new(&self.cfg)
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

        // Get token embedding
        let emb_offset = token as usize * c;
        let emb_slice = &self.embeddings.as_slice()[emb_offset..emb_offset + c];
        scratch.x.as_mut_slice().copy_from_slice(emb_slice);

        profiler.begin_token();

        unsafe {
            // Process each layer (using index to avoid borrow conflicts)
            for layer_idx in 0..num_layers {
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

                // Attention norm
                kernel::layer_norm_avx(
                    scratch.x.as_ptr(),
                    self.blocks[layer_idx].attn_norm_w.as_ptr(),
                    self.blocks[layer_idx].attn_norm_b.as_ptr(),
                    scratch.x_normed.as_mut_ptr(),
                    c,
                    self.cfg.layer_norm_eps,
                );

                let attn_start = Instant::now();
                self.attention_forward_impl(scratch, layer_idx, state);
                profiler.record_attention(layer_idx, attn_start.elapsed());

                // Add attention residual: x = x + att_out
                kernel::add_avx(
                    scratch.x.as_ptr(),
                    scratch.att_out.as_ptr(),
                    scratch.x.as_mut_ptr(),
                    c,
                );

                // FFN norm
                kernel::layer_norm_avx(
                    scratch.x.as_ptr(),
                    self.blocks[layer_idx].ffn_norm_w.as_ptr(),
                    self.blocks[layer_idx].ffn_norm_b.as_ptr(),
                    scratch.x_normed.as_mut_ptr(),
                    c,
                    self.cfg.layer_norm_eps,
                );

                let ffn_start = Instant::now();
                self.ffn_forward_impl(scratch, layer_idx, &mut state.layers[layer_idx]);
                profiler.record_ffn(layer_idx, ffn_start.elapsed());

                // Add FFN residual: x = x + ffn_out
                kernel::add_avx(
                    scratch.x.as_ptr(),
                    scratch.ffn_out.as_ptr(),
                    scratch.x.as_mut_ptr(),
                    c,
                );
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

        scratch.logits.as_slice()
    }

    #[inline(always)]
    unsafe fn attention_forward_impl(
        &self,
        scratch: &mut ScratchBuffers,
        layer_idx: usize,
        state: &mut State,
    ) {
        let attn = &self.blocks[layer_idx].attn;
        let layer_state = &mut state.layers[layer_idx];
        let c = self.cfg.hidden_size;
        let h = self.cfg.num_heads;
        let n = self.cfg.head_dim;
        let d_w = self.cfg.decay_low_rank;
        let d_a = self.cfg.a_low_rank;
        let d_g = self.cfg.g_low_rank;

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
        // Step 4: exp(-sigmoid(x) / sqrt(e))
        let inv_sqrt_e = 1.0 / std::f32::consts::E.sqrt();
        kernel::sigmoid_avx(scratch.w_decay.as_ptr(), scratch.w_decay.as_mut_ptr(), c);
        kernel::exp_neg_scaled_inplace(scratch.w_decay.as_mut_ptr(), inv_sqrt_e, c);

        // a = sigmoid(xa @ a1.T @ a2.T + a0)
        kernel::gemv_avx(
            attn.a1.as_ptr(),
            scratch.xa.as_ptr(),
            scratch.w_lora_tmp.as_mut_ptr(),
            d_a,
            c,
        );
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
        kernel::gemv_avx(
            attn.g2.as_ptr(),
            scratch.w_lora_tmp.as_ptr(),
            scratch.g.as_mut_ptr(),
            c,
            d_g,
        );

        // Value residual (layer > 0)
        if layer_idx == 0 {
            // Copy v to v_first buffer (no allocation)
            state.v_first.copy_from(&scratch.v);
            state.v_first_set = true;
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
            // v = v + (v_first - v) * nu
            for i in 0..c {
                let nu = scratch.att_out[i];
                scratch.v[i] += (state.v_first[i] - scratch.v[i]) * nu;
            }
        }

        // kk = k * k_k, then L2 normalize per head
        kernel::mul_avx(
            scratch.k.as_ptr(),
            attn.k_k.as_ptr(),
            scratch.kk.as_mut_ptr(),
            c,
        );
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

        // k = k * (1 + (a - 1) * k_a)
        for i in 0..c {
            let scale = 1.0 + (scratch.a[i] - 1.0) * attn.k_a[i];
            scratch.k[i] *= scale;
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

        // Add head-qk term: y += ((r * k * r_k).sum_per_head) * v
        for head in 0..h {
            let offset = head * n;
            let mut alpha = 0.0f32;
            for j in 0..n {
                alpha += scratch.r[offset + j] * scratch.k[offset + j] * attn.r_k[head * n + j];
            }
            for j in 0..n {
                scratch.y[offset + j] += alpha * scratch.v[offset + j];
            }
        }

        // Apply gate: y = y * g
        kernel::mul_avx(
            scratch.y.as_ptr(),
            scratch.g.as_ptr(),
            scratch.y.as_mut_ptr(),
            c,
        );

        // Output projection: att_out = o_proj @ y
        kernel::gemv_avx(
            attn.o_proj.as_ptr(),
            scratch.y.as_ptr(),
            scratch.att_out.as_mut_ptr(),
            c,
            c,
        );
    }

    #[inline(always)]
    unsafe fn ffn_forward_impl(
        &self,
        scratch: &mut ScratchBuffers,
        layer_idx: usize,
        layer_state: &mut LayerState,
    ) {
        let ffn = &self.blocks[layer_idx].ffn;
        let c = self.cfg.hidden_size;
        let i = self.cfg.intermediate_size;

        // Token shift: xk = x_normed + x_k * (prev - x_normed)
        kernel::token_shift_avx(
            scratch.x_normed.as_ptr(),
            layer_state.ffn_x_prev.as_ptr(),
            ffn.x_k.as_ptr(),
            scratch.xk.as_mut_ptr(),
            c,
        );

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
        kernel::relu_squared_avx(scratch.ffn_k.as_ptr(), scratch.ffn_k.as_mut_ptr(), i);

        // ffn_out = k @ value_w.T
        kernel::gemv_avx(
            ffn.value_w.as_ptr(),
            scratch.ffn_k.as_ptr(),
            scratch.ffn_out.as_mut_ptr(),
            c,
            i,
        );
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
