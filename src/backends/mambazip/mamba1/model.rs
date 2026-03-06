//! Deterministic CPU Mamba-1 single-token inference runtime.

use super::kernel;
use super::tensor::Tensor1D;
use super::weights::{WeightTensor, Weights};
use crate::backends::llm_policy::OptimizerKind;
use anyhow::{Context, Result, bail};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use wide::f32x8;

/// Mamba-1 model configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Byte vocabulary size.
    pub vocab_size: usize,
    /// Model hidden width.
    pub hidden_size: usize,
    /// Number of Mamba blocks.
    pub num_layers: usize,
    /// Expanded mixer width (`d_inner`).
    pub inner_size: usize,
    /// SSM state width (`d_state`).
    pub state_size: usize,
    /// Depthwise convolution kernel width (`d_conv`).
    pub conv_kernel: usize,
    /// Delta projection rank (`dt_rank`).
    pub dt_rank: usize,
    /// RMSNorm epsilon.
    pub layer_norm_eps: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 256,
            num_layers: 6,
            inner_size: 512,
            state_size: 16,
            conv_kernel: 4,
            dt_rank: 16,
            layer_norm_eps: 1e-5,
        }
    }
}

impl Config {
    /// Validate shape invariants.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            bail!("mamba vocab_size must be > 0");
        }
        if self.hidden_size == 0 {
            bail!("mamba hidden_size must be > 0");
        }
        if self.num_layers == 0 {
            bail!("mamba num_layers must be > 0");
        }
        if self.inner_size == 0 {
            bail!("mamba inner_size must be > 0");
        }
        if self.state_size == 0 {
            bail!("mamba state_size must be > 0");
        }
        if self.conv_kernel == 0 {
            bail!("mamba conv_kernel must be > 0");
        }
        if self.dt_rank == 0 {
            bail!("mamba dt_rank must be > 0");
        }
        Ok(())
    }
}

#[derive(Clone)]
struct LayerState {
    conv: Tensor1D, // (inner * conv_kernel)
    conv_pos: usize,
    ssm: Tensor1D, // (inner * state)
}

impl LayerState {
    fn new(cfg: &Config) -> Self {
        Self {
            conv: Tensor1D::zeros(cfg.inner_size * cfg.conv_kernel),
            conv_pos: 0,
            ssm: Tensor1D::zeros(cfg.inner_size * cfg.state_size),
        }
    }

    fn reset(&mut self) {
        self.conv.zero();
        self.conv_pos = 0;
        self.ssm.zero();
    }
}

/// Recurrent runtime state for a Mamba model.
#[derive(Clone)]
pub struct State {
    layers: Vec<LayerState>,
}

impl State {
    /// Create a zero-initialized state.
    pub fn new(cfg: &Config) -> Self {
        Self {
            layers: (0..cfg.num_layers).map(|_| LayerState::new(cfg)).collect(),
        }
    }

    /// Reset all recurrent buffers.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }
}

#[derive(Clone)]
struct LayerWeights {
    norm_w: Tensor1D,
    norm_b: Option<Tensor1D>,

    in_proj_w: Tensor1D, // (2*inner, hidden)
    in_proj_b: Option<Tensor1D>,

    conv_w: Tensor1D, // (inner, conv_kernel)
    conv_b: Option<Tensor1D>,

    x_proj_w: Tensor1D, // (dt_rank + 2*state, inner)
    x_proj_b: Option<Tensor1D>,

    dt_proj_w: Tensor1D, // (inner, dt_rank)
    dt_proj_b: Tensor1D,

    // Stored SSM diagonal in official parameterization and cached runtime form.
    a_log: Tensor1D, // (inner, state)
    a: Tensor1D,     // (inner, state), equals -exp(a_log)
    d: Tensor1D,     // (inner)

    out_proj_w: Tensor1D, // (hidden, inner)
    out_proj_b: Option<Tensor1D>,
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
struct LayerAdamState {
    norm_w: AdamTensorState,
    norm_b: Option<AdamTensorState>,
    in_proj_w: AdamTensorState,
    in_proj_b: Option<AdamTensorState>,
    conv_w: AdamTensorState,
    conv_b: Option<AdamTensorState>,
    x_proj_w: AdamTensorState,
    x_proj_b: Option<AdamTensorState>,
    dt_proj_w: AdamTensorState,
    dt_proj_b: AdamTensorState,
    a: AdamTensorState,
    d: AdamTensorState,
    out_proj_w: AdamTensorState,
    out_proj_b: Option<AdamTensorState>,
}

#[derive(Clone)]
/// Adam moments for full-parameter online Mamba training.
pub struct FullAdamState {
    embeddings: AdamTensorState,
    final_norm_w: AdamTensorState,
    final_norm_b: Option<AdamTensorState>,
    lm_head: AdamTensorState,
    lm_head_b: Option<AdamTensorState>,
    layers: Vec<LayerAdamState>,
}

#[derive(Clone, Copy, Debug, Default)]
/// Train-scope mask for Mamba full-parameter online updates.
pub struct TrainScopeMask {
    pub embed: bool,
    pub layer_norm: bool,
    pub mixer_conv: bool,
    pub mixer_ssm: bool,
    pub mixer_proj: bool,
    pub head: bool,
    pub bias: bool,
}

impl TrainScopeMask {
    #[inline]
    pub fn all() -> Self {
        Self {
            embed: true,
            layer_norm: true,
            mixer_conv: true,
            mixer_ssm: true,
            mixer_proj: true,
            head: true,
            bias: true,
        }
    }

    #[inline]
    pub fn trains_model_params(&self) -> bool {
        self.embed
            || self.layer_norm
            || self.mixer_conv
            || self.mixer_ssm
            || self.mixer_proj
            || self.head
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
    h_in: Tensor1D,
    norm: Tensor1D,
    xz: Tensor1D,
    conv_pre: Tensor1D,
    conv_post: Tensor1D,
    proj: Tensor1D,
    dt_raw: Tensor1D,
    dt: Tensor1D,
    gate: Tensor1D,
    y_pre: Tensor1D,
    y: Tensor1D,
    out: Tensor1D,
    d_a: Tensor1D,
    ssm_prev: Tensor1D,
    conv_prev: Tensor1D,
    conv_pos_prev: usize,
}

impl LayerTrainTrace {
    fn new(cfg: &Config) -> Self {
        Self {
            h_in: Tensor1D::zeros(cfg.hidden_size),
            norm: Tensor1D::zeros(cfg.hidden_size),
            xz: Tensor1D::zeros(cfg.inner_size * 2),
            conv_pre: Tensor1D::zeros(cfg.inner_size),
            conv_post: Tensor1D::zeros(cfg.inner_size),
            proj: Tensor1D::zeros(cfg.dt_rank + 2 * cfg.state_size),
            dt_raw: Tensor1D::zeros(cfg.inner_size),
            dt: Tensor1D::zeros(cfg.inner_size),
            gate: Tensor1D::zeros(cfg.inner_size),
            y_pre: Tensor1D::zeros(cfg.inner_size),
            y: Tensor1D::zeros(cfg.inner_size),
            out: Tensor1D::zeros(cfg.hidden_size),
            d_a: Tensor1D::zeros(cfg.inner_size * cfg.state_size),
            ssm_prev: Tensor1D::zeros(cfg.inner_size * cfg.state_size),
            conv_prev: Tensor1D::zeros(cfg.inner_size * cfg.conv_kernel),
            conv_pos_prev: 0,
        }
    }
}

/// Mamba model weights and inference kernels.
#[derive(Clone)]
pub struct Model {
    cfg: Config,
    embeddings: Tensor1D, // (vocab, hidden)
    final_norm_w: Tensor1D,
    final_norm_b: Option<Tensor1D>,
    lm_head: Tensor1D, // (vocab, hidden)
    lm_head_b: Option<Tensor1D>,
    layers: Vec<LayerWeights>,
}

/// Preallocated temporary buffers for token forward passes.
#[derive(Clone)]
pub struct ScratchBuffers {
    h: Tensor1D,
    norm: Tensor1D,
    xz: Tensor1D,
    conv: Tensor1D,
    proj: Tensor1D,
    dt: Tensor1D,
    y: Tensor1D,
    out: Tensor1D,
    logits: Tensor1D,
    grad_h: Tensor1D,
    grad_norm: Tensor1D,
    grad_xz: Tensor1D,
    grad_conv: Tensor1D,
    grad_conv_pre: Tensor1D,
    grad_proj: Tensor1D,
    grad_dt_raw: Tensor1D,
    grad_u: Tensor1D,
    grad_b: Tensor1D,
    grad_c: Tensor1D,
    grad_ssm_d: Tensor1D,
    grad_ssm_a: Tensor1D,
    grad_ssm_a_log: Tensor1D,
    grad_conv_w: Tensor1D,
    grad_conv_b: Tensor1D,
    grad_y: Tensor1D,
    grad_out: Tensor1D,
    grad_logits: Tensor1D,
    grad_residual: Tensor1D,
    train_trace_layers: Vec<LayerTrainTrace>,
    train_h_final: Tensor1D,
    train_token: usize,
    train_trace_valid: bool,
    capture_train_trace: bool,
}

impl ScratchBuffers {
    /// Allocate scratch for config.
    pub fn new(cfg: &Config) -> Self {
        let mut train_trace_layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            train_trace_layers.push(LayerTrainTrace::new(cfg));
        }
        Self {
            h: Tensor1D::zeros(cfg.hidden_size),
            norm: Tensor1D::zeros(cfg.hidden_size),
            xz: Tensor1D::zeros(cfg.inner_size * 2),
            conv: Tensor1D::zeros(cfg.inner_size),
            proj: Tensor1D::zeros(cfg.dt_rank + 2 * cfg.state_size),
            dt: Tensor1D::zeros(cfg.inner_size),
            y: Tensor1D::zeros(cfg.inner_size),
            out: Tensor1D::zeros(cfg.hidden_size),
            logits: Tensor1D::zeros(cfg.vocab_size),
            grad_h: Tensor1D::zeros(cfg.hidden_size),
            grad_norm: Tensor1D::zeros(cfg.hidden_size),
            grad_xz: Tensor1D::zeros(cfg.inner_size * 2),
            grad_conv: Tensor1D::zeros(cfg.inner_size),
            grad_conv_pre: Tensor1D::zeros(cfg.inner_size),
            grad_proj: Tensor1D::zeros(cfg.dt_rank + 2 * cfg.state_size),
            grad_dt_raw: Tensor1D::zeros(cfg.inner_size),
            grad_u: Tensor1D::zeros(cfg.dt_rank),
            grad_b: Tensor1D::zeros(cfg.state_size),
            grad_c: Tensor1D::zeros(cfg.state_size),
            grad_ssm_d: Tensor1D::zeros(cfg.inner_size),
            grad_ssm_a: Tensor1D::zeros(cfg.inner_size * cfg.state_size),
            grad_ssm_a_log: Tensor1D::zeros(cfg.inner_size * cfg.state_size),
            grad_conv_w: Tensor1D::zeros(cfg.inner_size * cfg.conv_kernel),
            grad_conv_b: Tensor1D::zeros(cfg.inner_size),
            grad_y: Tensor1D::zeros(cfg.inner_size),
            grad_out: Tensor1D::zeros(cfg.hidden_size),
            grad_logits: Tensor1D::zeros(cfg.vocab_size),
            grad_residual: Tensor1D::zeros(cfg.hidden_size),
            train_trace_layers,
            train_h_final: Tensor1D::zeros(cfg.hidden_size),
            train_token: 0,
            train_trace_valid: false,
            capture_train_trace: false,
        }
    }

    /// Final normalized hidden state consumed by LM head.
    #[inline]
    pub fn lm_head_input(&self) -> &[f32] {
        self.norm.as_slice()
    }

    /// Restore LM-head input snapshot for reversible online updates.
    #[inline]
    pub fn set_lm_head_input(&mut self, value: &[f32]) {
        self.norm.as_mut_slice().copy_from_slice(value);
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
    /// Allocate zero-initialized Adam moments matching all trainable tensors.
    pub fn new_full_adam_state(&self) -> FullAdamState {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(LayerAdamState {
                norm_w: AdamTensorState::new(layer.norm_w.len()),
                norm_b: layer.norm_b.as_ref().map(|b| AdamTensorState::new(b.len())),
                in_proj_w: AdamTensorState::new(layer.in_proj_w.len()),
                in_proj_b: layer
                    .in_proj_b
                    .as_ref()
                    .map(|b| AdamTensorState::new(b.len())),
                conv_w: AdamTensorState::new(layer.conv_w.len()),
                conv_b: layer.conv_b.as_ref().map(|b| AdamTensorState::new(b.len())),
                x_proj_w: AdamTensorState::new(layer.x_proj_w.len()),
                x_proj_b: layer
                    .x_proj_b
                    .as_ref()
                    .map(|b| AdamTensorState::new(b.len())),
                dt_proj_w: AdamTensorState::new(layer.dt_proj_w.len()),
                dt_proj_b: AdamTensorState::new(layer.dt_proj_b.len()),
                a: AdamTensorState::new(layer.a_log.len()),
                d: AdamTensorState::new(layer.d.len()),
                out_proj_w: AdamTensorState::new(layer.out_proj_w.len()),
                out_proj_b: layer
                    .out_proj_b
                    .as_ref()
                    .map(|b| AdamTensorState::new(b.len())),
            });
        }

        FullAdamState {
            embeddings: AdamTensorState::new(self.embeddings.len()),
            final_norm_w: AdamTensorState::new(self.final_norm_w.len()),
            final_norm_b: self
                .final_norm_b
                .as_ref()
                .map(|b| AdamTensorState::new(b.len())),
            lm_head: AdamTensorState::new(self.lm_head.len()),
            lm_head_b: self
                .lm_head_b
                .as_ref()
                .map(|b| AdamTensorState::new(b.len())),
            layers,
        }
    }

    /// Load a model from safetensors checkpoint.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let weights = Weights::load(path.as_ref()).with_context(|| {
            format!(
                "failed to load model weights from {}",
                path.as_ref().display()
            )
        })?;

        if weights.get("backbone.embedding.weight").is_some() {
            Self::load_official(&weights)
        } else {
            Self::load_native(&weights)
        }
    }

    /// Build a deterministic random model for online-mode workflows.
    pub fn new_random(cfg: Config, seed: u64) -> Result<Self> {
        cfg.validate()?;

        let mut rng = MambaRng::new(seed);
        let v = cfg.vocab_size;
        let h = cfg.hidden_size;
        let i = cfg.inner_size;
        let s = cfg.state_size;
        let k = cfg.conv_kernel;
        let r = cfg.dt_rank;

        let mut embeddings = Tensor1D::zeros(v * h);
        init_uniform(&mut embeddings, &mut rng, 0.02);

        let mut final_norm_w = Tensor1D::zeros(h);
        init_const(&mut final_norm_w, 1.0);

        let mut lm_head = Tensor1D::zeros(v * h);
        init_uniform(&mut lm_head, &mut rng, 0.02);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let mut norm_w = Tensor1D::zeros(h);
            init_const(&mut norm_w, 1.0);

            let mut in_proj_w = Tensor1D::zeros((2 * i) * h);
            init_uniform(&mut in_proj_w, &mut rng, 0.02);
            let mut in_proj_b = Tensor1D::zeros(2 * i);
            init_const(&mut in_proj_b, 0.0);

            let mut conv_w = Tensor1D::zeros(i * k);
            init_uniform(&mut conv_w, &mut rng, 0.05);
            let mut conv_b = Tensor1D::zeros(i);
            init_const(&mut conv_b, 0.0);

            let mut x_proj_w = Tensor1D::zeros((r + 2 * s) * i);
            init_uniform(&mut x_proj_w, &mut rng, 0.02);

            let mut dt_proj_w = Tensor1D::zeros(i * r);
            init_uniform(&mut dt_proj_w, &mut rng, 0.02);
            let mut dt_proj_b = Tensor1D::zeros(i);
            init_const(&mut dt_proj_b, -2.0);

            let mut a_log = Tensor1D::zeros(i * s);
            init_const(&mut a_log, 0.0);
            let a = a_from_a_log_tensor(&a_log);
            let mut d = Tensor1D::zeros(i);
            init_const(&mut d, 1.0);

            let mut out_proj_w = Tensor1D::zeros(h * i);
            init_uniform(&mut out_proj_w, &mut rng, 0.02);
            let mut out_proj_b = Tensor1D::zeros(h);
            init_const(&mut out_proj_b, 0.0);

            layers.push(LayerWeights {
                norm_w,
                norm_b: None,
                in_proj_w,
                in_proj_b: Some(in_proj_b),
                conv_w,
                conv_b: Some(conv_b),
                x_proj_w,
                x_proj_b: None,
                dt_proj_w,
                dt_proj_b,
                a_log,
                a,
                d,
                out_proj_w,
                out_proj_b: Some(out_proj_b),
            });
        }

        Ok(Self {
            cfg,
            embeddings,
            final_norm_w,
            final_norm_b: None,
            lm_head,
            lm_head_b: None,
            layers,
        })
    }

    /// Save checkpoint to safetensors (native infotheory Mamba layout).
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        #[derive(Clone)]
        struct TensorRec {
            name: String,
            shape: Vec<usize>,
            data: Vec<f32>,
        }

        let c = self.cfg.hidden_size;
        let v = self.cfg.vocab_size;
        let i = self.cfg.inner_size;
        let s = self.cfg.state_size;
        let k = self.cfg.conv_kernel;
        let r = self.cfg.dt_rank;

        let mut recs: Vec<TensorRec> = Vec::new();

        let mut push = |name: String, shape: Vec<usize>, t: &Tensor1D| {
            recs.push(TensorRec {
                name,
                shape,
                data: t.as_slice().to_vec(),
            });
        };

        push(
            "model.embeddings.weight".to_string(),
            vec![v, c],
            &self.embeddings,
        );
        push("model.norm.weight".to_string(), vec![c], &self.final_norm_w);
        if let Some(b) = &self.final_norm_b {
            push("model.norm.bias".to_string(), vec![c], b);
        }
        push("lm_head.weight".to_string(), vec![v, c], &self.lm_head);
        if let Some(b) = &self.lm_head_b {
            push("lm_head.bias".to_string(), vec![v], b);
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            let pfx = format!("model.layers.{idx}.mixer");
            push(
                format!("model.layers.{idx}.norm.weight"),
                vec![c],
                &layer.norm_w,
            );
            if let Some(b) = &layer.norm_b {
                push(format!("model.layers.{idx}.norm.bias"), vec![c], b);
            }
            push(
                format!("{pfx}.in_proj.weight"),
                vec![2 * i, c],
                &layer.in_proj_w,
            );
            if let Some(b) = &layer.in_proj_b {
                push(format!("{pfx}.in_proj.bias"), vec![2 * i], b);
            }
            push(format!("{pfx}.conv1d.weight"), vec![i, 1, k], &layer.conv_w);
            if let Some(b) = &layer.conv_b {
                push(format!("{pfx}.conv1d.bias"), vec![i], b);
            }
            push(
                format!("{pfx}.x_proj.weight"),
                vec![r + 2 * s, i],
                &layer.x_proj_w,
            );
            if let Some(b) = &layer.x_proj_b {
                push(format!("{pfx}.x_proj.bias"), vec![r + 2 * s], b);
            }
            push(
                format!("{pfx}.dt_proj.weight"),
                vec![i, r],
                &layer.dt_proj_w,
            );
            push(format!("{pfx}.dt_proj.bias"), vec![i], &layer.dt_proj_b);
            push(format!("{pfx}.A_log"), vec![i, s], &layer.a_log);
            push(format!("{pfx}.D"), vec![i], &layer.d);
            push(
                format!("{pfx}.out_proj.weight"),
                vec![c, i],
                &layer.out_proj_w,
            );
            if let Some(b) = &layer.out_proj_b {
                push(format!("{pfx}.out_proj.bias"), vec![c], b);
            }
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
        let v = self.cfg.vocab_size;
        let i = self.cfg.inner_size;
        let s = self.cfg.state_size;
        let k = self.cfg.conv_kernel;
        let r = self.cfg.dt_rank;

        let mut recs: Vec<TensorRec> = Vec::new();
        let mut push_state = |name_prefix: &str, shape: Vec<usize>, st: &AdamTensorState| {
            recs.push(TensorRec {
                name: format!("{name_prefix}.m"),
                shape: shape.clone(),
                data: st.m.as_slice().to_vec(),
            });
            recs.push(TensorRec {
                name: format!("{name_prefix}.v"),
                shape,
                data: st.v.as_slice().to_vec(),
            });
        };

        push_state("opt.embeddings", vec![v, c], &adam.embeddings);
        push_state("opt.final_norm.weight", vec![c], &adam.final_norm_w);
        if let Some(b) = &adam.final_norm_b {
            push_state("opt.final_norm.bias", vec![c], b);
        }
        push_state("opt.lm_head.weight", vec![v, c], &adam.lm_head);
        if let Some(b) = &adam.lm_head_b {
            push_state("opt.lm_head.bias", vec![v], b);
        }

        for (idx, layer) in adam.layers.iter().enumerate() {
            let pfx = format!("opt.layers.{idx}");
            push_state(&format!("{pfx}.norm.weight"), vec![c], &layer.norm_w);
            if let Some(b) = &layer.norm_b {
                push_state(&format!("{pfx}.norm.bias"), vec![c], b);
            }
            push_state(
                &format!("{pfx}.in_proj.weight"),
                vec![2 * i, c],
                &layer.in_proj_w,
            );
            if let Some(b) = &layer.in_proj_b {
                push_state(&format!("{pfx}.in_proj.bias"), vec![2 * i], b);
            }
            push_state(
                &format!("{pfx}.conv1d.weight"),
                vec![i, 1, k],
                &layer.conv_w,
            );
            if let Some(b) = &layer.conv_b {
                push_state(&format!("{pfx}.conv1d.bias"), vec![i], b);
            }
            push_state(
                &format!("{pfx}.x_proj.weight"),
                vec![r + 2 * s, i],
                &layer.x_proj_w,
            );
            if let Some(b) = &layer.x_proj_b {
                push_state(&format!("{pfx}.x_proj.bias"), vec![r + 2 * s], b);
            }
            push_state(
                &format!("{pfx}.dt_proj.weight"),
                vec![i, r],
                &layer.dt_proj_w,
            );
            push_state(&format!("{pfx}.dt_proj.bias"), vec![i], &layer.dt_proj_b);
            push_state(&format!("{pfx}.A_log"), vec![i, s], &layer.a);
            push_state(&format!("{pfx}.D"), vec![i], &layer.d);
            push_state(
                &format!("{pfx}.out_proj.weight"),
                vec![c, i],
                &layer.out_proj_w,
            );
            if let Some(b) = &layer.out_proj_b {
                push_state(&format!("{pfx}.out_proj.bias"), vec![c], b);
            }
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

        let load_state = |name_prefix: &str, st: &mut AdamTensorState| -> Result<()> {
            let m_name = format!("{name_prefix}.m");
            let v_name = format!("{name_prefix}.v");
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

        load_state("opt.embeddings", &mut adam.embeddings)?;
        load_state("opt.final_norm.weight", &mut adam.final_norm_w)?;
        if let Some(st) = adam.final_norm_b.as_mut() {
            load_state("opt.final_norm.bias", st)?;
        }
        load_state("opt.lm_head.weight", &mut adam.lm_head)?;
        if let Some(st) = adam.lm_head_b.as_mut() {
            load_state("opt.lm_head.bias", st)?;
        }

        for (idx, layer) in adam.layers.iter_mut().enumerate() {
            let pfx = format!("opt.layers.{idx}");
            load_state(&format!("{pfx}.norm.weight"), &mut layer.norm_w)?;
            if let Some(st) = layer.norm_b.as_mut() {
                load_state(&format!("{pfx}.norm.bias"), st)?;
            }
            load_state(&format!("{pfx}.in_proj.weight"), &mut layer.in_proj_w)?;
            if let Some(st) = layer.in_proj_b.as_mut() {
                load_state(&format!("{pfx}.in_proj.bias"), st)?;
            }
            load_state(&format!("{pfx}.conv1d.weight"), &mut layer.conv_w)?;
            if let Some(st) = layer.conv_b.as_mut() {
                load_state(&format!("{pfx}.conv1d.bias"), st)?;
            }
            load_state(&format!("{pfx}.x_proj.weight"), &mut layer.x_proj_w)?;
            if let Some(st) = layer.x_proj_b.as_mut() {
                load_state(&format!("{pfx}.x_proj.bias"), st)?;
            }
            load_state(&format!("{pfx}.dt_proj.weight"), &mut layer.dt_proj_w)?;
            load_state(&format!("{pfx}.dt_proj.bias"), &mut layer.dt_proj_b)?;
            load_state(&format!("{pfx}.A_log"), &mut layer.a)?;
            load_state(&format!("{pfx}.D"), &mut layer.d)?;
            load_state(&format!("{pfx}.out_proj.weight"), &mut layer.out_proj_w)?;
            if let Some(st) = layer.out_proj_b.as_mut() {
                load_state(&format!("{pfx}.out_proj.bias"), st)?;
            }
        }

        Ok(adam)
    }

    /// Access model config.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Allocate a new recurrent state.
    #[inline]
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

    /// Forward a single byte token and return logits for next symbol.
    #[inline(never)]
    pub fn forward<'a>(
        &'a self,
        scratch: &'a mut ScratchBuffers,
        token: u32,
        state: &mut State,
    ) -> &'a [f32] {
        let c = self.cfg.hidden_size;
        let i = self.cfg.inner_size;
        let s = self.cfg.state_size;
        let r = self.cfg.dt_rank;

        let token_idx = (token as usize).min(self.cfg.vocab_size.saturating_sub(1));
        let emb_off = token_idx * c;
        if scratch.capture_train_trace {
            scratch.train_token = token_idx;
            scratch.train_trace_valid = true;
        } else {
            scratch.train_trace_valid = false;
        }
        scratch
            .h
            .as_mut_slice()
            .copy_from_slice(&self.embeddings.as_slice()[emb_off..emb_off + c]);

        for layer_idx in 0..self.cfg.num_layers {
            let layer = &self.layers[layer_idx];
            let st = &mut state.layers[layer_idx];
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.h_in.as_mut_slice().copy_from_slice(scratch.h.as_slice());
                tr.ssm_prev
                    .as_mut_slice()
                    .copy_from_slice(st.ssm.as_slice());
                tr.conv_prev
                    .as_mut_slice()
                    .copy_from_slice(st.conv.as_slice());
                tr.conv_pos_prev = st.conv_pos;
            }

            rms_norm(
                scratch.h.as_slice(),
                layer.norm_w.as_slice(),
                layer.norm_b.as_ref().map(Tensor1D::as_slice),
                self.cfg.layer_norm_eps,
                scratch.norm.as_mut_slice(),
            );
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.norm
                    .as_mut_slice()
                    .copy_from_slice(scratch.norm.as_slice());
            }

            unsafe {
                // SAFETY: row-major dimensions are validated at load time.
                kernel::gemv(
                    layer.in_proj_w.as_ptr(),
                    scratch.norm.as_ptr(),
                    scratch.xz.as_mut_ptr(),
                    i * 2,
                    c,
                );
            }
            if let Some(bias) = &layer.in_proj_b {
                for (dst, &b) in scratch.xz.as_mut_slice().iter_mut().zip(bias.as_slice()) {
                    *dst += b;
                }
            }
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.xz.as_mut_slice().copy_from_slice(scratch.xz.as_slice());
            }

            depthwise_conv_step(
                &scratch.xz.as_slice()[0..i],
                &layer.conv_w,
                layer.conv_b.as_ref(),
                self.cfg.conv_kernel,
                st,
                scratch.conv.as_mut_slice(),
            );
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.conv_pre
                    .as_mut_slice()
                    .copy_from_slice(scratch.conv.as_slice());
            }

            for idx in 0..i {
                scratch.conv[idx] = silu(scratch.conv[idx]);
            }
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.conv_post
                    .as_mut_slice()
                    .copy_from_slice(scratch.conv.as_slice());
            }

            unsafe {
                // SAFETY: row-major dimensions are validated at load time.
                kernel::gemv(
                    layer.x_proj_w.as_ptr(),
                    scratch.conv.as_ptr(),
                    scratch.proj.as_mut_ptr(),
                    r + 2 * s,
                    i,
                );
            }
            if let Some(bias) = &layer.x_proj_b {
                for (dst, &b) in scratch.proj.as_mut_slice().iter_mut().zip(bias.as_slice()) {
                    *dst += b;
                }
            }
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.proj
                    .as_mut_slice()
                    .copy_from_slice(scratch.proj.as_slice());
            }

            unsafe {
                // SAFETY: row-major dimensions are validated at load time.
                kernel::gemv(
                    layer.dt_proj_w.as_ptr(),
                    scratch.proj.as_ptr(),
                    scratch.dt.as_mut_ptr(),
                    i,
                    r,
                );
            }
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.dt_raw
                    .as_mut_slice()
                    .copy_from_slice(scratch.dt.as_slice());
            }

            let proj = scratch.proj.as_slice();
            let b_vec = &proj[r..r + s];
            let c_vec = &proj[r + s..r + 2 * s];
            let conv = scratch.conv.as_slice();
            let dt_raw = scratch.dt.as_slice();
            let xz = scratch.xz.as_slice();
            let d = layer.d.as_slice();
            let a = layer.a.as_slice();
            let dt_bias = layer.dt_proj_b.as_slice();
            let ssm = st.ssm.as_mut_slice();
            let b_ptr = b_vec.as_ptr();
            let c_ptr = c_vec.as_ptr();
            let a_ptr = a.as_ptr();
            let ssm_ptr = ssm.as_mut_ptr();

            for ch in 0..i {
                let x_ch = conv[ch];
                let dt = softplus(dt_raw[ch] + dt_bias[ch]);
                let gate = silu(xz[i + ch]);
                let x_dt = x_ch * dt;

                let mut y = d[ch] * x_ch;
                let ssm_row_off = ch * s;
                // SAFETY: offsets are in-bounds due to validated tensor shapes and loop ranges.
                let row_a = unsafe { a_ptr.add(ssm_row_off) };
                // SAFETY: offsets are in-bounds due to validated tensor shapes and loop ranges.
                let row_ssm = unsafe { ssm_ptr.add(ssm_row_off) };
                let mut j = 0usize;
                while j < s {
                    // SAFETY: j in [0, s), row pointers are valid for s elements.
                    let prev = unsafe { *row_ssm.add(j) };
                    // SAFETY: j in [0, s), row pointers are valid for s elements.
                    let d_a = (dt * unsafe { *row_a.add(j) }).exp();
                    if scratch.capture_train_trace {
                        scratch.train_trace_layers[layer_idx].d_a[ssm_row_off + j] = d_a;
                    }
                    // SAFETY: b/c vectors have length s by construction.
                    let next = prev * d_a + x_dt * unsafe { *b_ptr.add(j) };
                    // SAFETY: row pointer valid and unique for write.
                    unsafe { *row_ssm.add(j) = next };
                    // SAFETY: c vector has length s by construction.
                    y += next * unsafe { *c_ptr.add(j) };
                    j += 1;
                }
                if scratch.capture_train_trace {
                    let tr = &mut scratch.train_trace_layers[layer_idx];
                    tr.dt[ch] = dt;
                    tr.gate[ch] = gate;
                    tr.y_pre[ch] = y;
                }
                scratch.y[ch] = y * gate;
                if scratch.capture_train_trace {
                    scratch.train_trace_layers[layer_idx].y[ch] = scratch.y[ch];
                }
            }

            unsafe {
                // SAFETY: row-major dimensions are validated at load time.
                kernel::gemv(
                    layer.out_proj_w.as_ptr(),
                    scratch.y.as_ptr(),
                    scratch.out.as_mut_ptr(),
                    c,
                    i,
                );
            }
            if let Some(bias) = &layer.out_proj_b {
                for (dst, &b) in scratch.out.as_mut_slice().iter_mut().zip(bias.as_slice()) {
                    *dst += b;
                }
            }
            if scratch.capture_train_trace {
                let tr = &mut scratch.train_trace_layers[layer_idx];
                tr.out
                    .as_mut_slice()
                    .copy_from_slice(scratch.out.as_slice());
            }

            unsafe {
                // SAFETY: vector lengths match hidden size.
                kernel::add_inplace(scratch.h.as_mut_ptr(), scratch.out.as_ptr(), c);
            }
        }
        if scratch.capture_train_trace {
            scratch
                .train_h_final
                .as_mut_slice()
                .copy_from_slice(scratch.h.as_slice());
        }

        rms_norm(
            scratch.h.as_slice(),
            self.final_norm_w.as_slice(),
            self.final_norm_b.as_ref().map(Tensor1D::as_slice),
            self.cfg.layer_norm_eps,
            scratch.norm.as_mut_slice(),
        );

        unsafe {
            // SAFETY: row-major dimensions are validated at load time.
            kernel::gemv(
                self.lm_head.as_ptr(),
                scratch.norm.as_ptr(),
                scratch.logits.as_mut_ptr(),
                self.cfg.vocab_size,
                c,
            );
        }
        if let Some(bias) = &self.lm_head_b {
            for (dst, &b) in scratch
                .logits
                .as_mut_slice()
                .iter_mut()
                .zip(bias.as_slice())
            {
                *dst += b;
            }
        }

        scratch.logits.as_slice()
    }

    /// Exact single-step (`bptt=1`) online training for all Mamba parameters.
    ///
    /// This consumes the latest forward trace captured in `scratch` and applies
    /// one gradient step using the externally provided PDF/target symbol.
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
        if !scope.trains_model_params() && !scope.bias {
            return Ok(());
        }
        let needs_backprop = scope.embed
            || scope.layer_norm
            || scope.mixer_conv
            || scope.mixer_ssm
            || scope.mixer_proj;
        if needs_backprop && !scratch.train_trace_valid {
            bail!("mamba full training trace is missing; run one forward step first");
        }
        let c = self.cfg.hidden_size;
        let i = self.cfg.inner_size;
        let s = self.cfg.state_size;
        let r = self.cfg.dt_rank;
        let v = self.cfg.vocab_size.min(pdf.len());
        if v == 0 {
            return Ok(());
        }

        let mut adam_cfg = None::<AdamStep>;
        let mut model_adam = model_adam;
        if matches!(optimizer, OptimizerKind::Adam) {
            *adam_t = adam_t.saturating_add(1);
            let t = (*adam_t).max(1) as i32;
            let b1 = 0.9f32;
            let b2 = 0.999f32;
            adam_cfg = Some(AdamStep {
                lr,
                clip,
                b1,
                b2,
                eps: 1e-8,
                bias_corr1: 1.0 - b1.powi(t),
                bias_corr2: 1.0 - b2.powi(t),
            });
            if scope.trains_model_params() && model_adam.is_none() {
                bail!("mamba Adam full-training state is missing");
            }
        }

        // dL/d(logits + out_bias) for cross-entropy with one-hot target.
        scratch.grad_logits.zero();
        for tok in 0..v {
            let p = pdf[tok].clamp(1e-12, 1.0) as f32;
            let target = if tok == symbol as usize { 1.0 } else { 0.0 };
            let mut g = target - p;
            if clip > 0.0 {
                g = g.clamp(-clip, clip);
            }
            scratch.grad_logits[tok] = g;
        }

        if scope.bias
            && let Some(bias) = out_bias
        {
            let n = bias.len().min(v);
            let grad = &scratch.grad_logits.as_slice()[..n];
            match optimizer {
                OptimizerKind::Sgd => {
                    for idx in 0..n {
                        bias[idx] += lr * grad[idx];
                    }
                }
                OptimizerKind::Adam => {
                    let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                    let Some(m) = out_bias_adam_m else {
                        bail!("mamba Adam output-bias moments are missing");
                    };
                    let Some(vv) = out_bias_adam_v else {
                        bail!("mamba Adam output-bias moments are missing");
                    };
                    if m.len() < n || vv.len() < n {
                        bail!("mamba Adam output-bias moments have invalid shape");
                    }
                    for idx in 0..n {
                        let g = grad[idx];
                        m[idx] = cfg.b1 * m[idx] + (1.0 - cfg.b1) * g;
                        vv[idx] = cfg.b2 * vv[idx] + (1.0 - cfg.b2) * g * g;
                        let m_hat = m[idx] / cfg.bias_corr1;
                        let v_hat = vv[idx] / cfg.bias_corr2;
                        bias[idx] += cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
                    }
                }
            }
        }

        // Head backward/update. When scope.head is enabled we fuse grad_h and
        // parameter update in one pass to reduce memory traffic.
        scratch.grad_h.zero();
        let norm_in = scratch.norm.as_slice();
        if scope.head {
            match optimizer {
                OptimizerKind::Sgd => {
                    let head = self.lm_head.as_mut_slice();
                    let norm_ptr = norm_in.as_ptr();
                    let grad_h_ptr = scratch.grad_h.as_mut_slice().as_mut_ptr();
                    for tok in 0..v {
                        let g = scratch.grad_logits[tok];
                        let row_off = tok * c;
                        let mut j = 0usize;
                        unsafe {
                            let g8 = f32x8::splat(g);
                            let lr8 = f32x8::splat(lr);
                            while j + 8 <= c {
                                let idx = row_off + j;
                                let wv = head.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
                                let nv = norm_ptr.add(j).cast::<f32x8>().read_unaligned();
                                let ghv = grad_h_ptr.add(j).cast::<f32x8>().read_unaligned();
                                grad_h_ptr
                                    .add(j)
                                    .cast::<f32x8>()
                                    .write_unaligned(ghv + wv * g8);
                                head.as_mut_ptr()
                                    .add(idx)
                                    .cast::<f32x8>()
                                    .write_unaligned(wv + (g8 * nv) * lr8);
                                j += 8;
                            }
                        }
                        while j < c {
                            let idx = row_off + j;
                            let w_old = head[idx];
                            scratch.grad_h[j] += w_old * g;
                            head[idx] = w_old + lr * g * norm_in[j];
                            j += 1;
                        }
                    }
                    if let Some(b) = self.lm_head_b.as_mut() {
                        for tok in 0..v.min(b.len()) {
                            b[tok] += lr * scratch.grad_logits[tok];
                        }
                    }
                }
                OptimizerKind::Adam => {
                    let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                    let adam = model_adam.as_mut().expect("adam state exists");
                    let head = self.lm_head.as_mut_slice();
                    let hm = adam.lm_head.m.as_mut_slice();
                    let hv = adam.lm_head.v.as_mut_slice();
                    let norm_ptr = norm_in.as_ptr();
                    let grad_h_ptr = scratch.grad_h.as_mut_slice().as_mut_ptr();
                    let b1 = f32x8::splat(cfg.b1);
                    let b2 = f32x8::splat(cfg.b2);
                    let one_b1 = f32x8::splat(1.0 - cfg.b1);
                    let one_b2 = f32x8::splat(1.0 - cfg.b2);
                    let inv_bc1 = f32x8::splat(1.0 / cfg.bias_corr1);
                    let inv_bc2 = f32x8::splat(1.0 / cfg.bias_corr2);
                    let eps = f32x8::splat(cfg.eps);
                    let lr8 = f32x8::splat(cfg.lr);
                    for tok in 0..v {
                        let g = scratch.grad_logits[tok];
                        let row_off = tok * c;
                        let mut j = 0usize;
                        unsafe {
                            let g8 = f32x8::splat(g);
                            while j + 8 <= c {
                                let idx = row_off + j;
                                let wv = head.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
                                let nv = norm_ptr.add(j).cast::<f32x8>().read_unaligned();
                                let ghv = grad_h_ptr.add(j).cast::<f32x8>().read_unaligned();
                                grad_h_ptr
                                    .add(j)
                                    .cast::<f32x8>()
                                    .write_unaligned(ghv + wv * g8);

                                let gg = g8 * nv;
                                let hm_old = hm.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
                                let hv_old = hv.as_ptr().add(idx).cast::<f32x8>().read_unaligned();
                                let m = hm_old * b1 + gg * one_b1;
                                let vv = hv_old * b2 + (gg * gg) * one_b2;
                                hm.as_mut_ptr().add(idx).cast::<f32x8>().write_unaligned(m);
                                hv.as_mut_ptr().add(idx).cast::<f32x8>().write_unaligned(vv);
                                let upd = ((m * inv_bc1) / ((vv * inv_bc2).sqrt() + eps)) * lr8;
                                head.as_mut_ptr()
                                    .add(idx)
                                    .cast::<f32x8>()
                                    .write_unaligned(wv + upd);
                                j += 8;
                            }
                        }
                        while j < c {
                            let idx = row_off + j;
                            let w_old = head[idx];
                            scratch.grad_h[j] += w_old * g;
                            let gg = g * norm_in[j];
                            let m = cfg.b1 * hm[idx] + (1.0 - cfg.b1) * gg;
                            let vv = cfg.b2 * hv[idx] + (1.0 - cfg.b2) * gg * gg;
                            hm[idx] = m;
                            hv[idx] = vv;
                            let m_hat = m / cfg.bias_corr1;
                            let v_hat = vv / cfg.bias_corr2;
                            head[idx] = w_old + cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
                            j += 1;
                        }
                    }
                    if let (Some(b), Some(bm)) = (self.lm_head_b.as_mut(), adam.lm_head_b.as_mut())
                    {
                        let bm_m = bm.m.as_mut_slice();
                        let bm_v = bm.v.as_mut_slice();
                        for tok in 0..v.min(b.len()) {
                            let g = scratch.grad_logits[tok];
                            let m = cfg.b1 * bm_m[tok] + (1.0 - cfg.b1) * g;
                            let vv = cfg.b2 * bm_v[tok] + (1.0 - cfg.b2) * g * g;
                            bm_m[tok] = m;
                            bm_v[tok] = vv;
                            let m_hat = m / cfg.bias_corr1;
                            let v_hat = vv / cfg.bias_corr2;
                            b[tok] += cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
                        }
                    }
                }
            }
        } else {
            let head = self.lm_head.as_slice();
            for tok in 0..v {
                let g = scratch.grad_logits[tok];
                let row_off = tok * c;
                for j in 0..c {
                    scratch.grad_h[j] += head[row_off + j] * g;
                }
            }
        }

        if !needs_backprop {
            return Ok(());
        }

        // Final RMSNorm backward.
        rms_norm_backward(
            scratch.train_h_final.as_slice(),
            self.final_norm_w.as_slice(),
            scratch.grad_h.as_slice(),
            self.cfg.layer_norm_eps,
            scratch.grad_norm.as_mut_slice(),
            scratch.grad_out.as_mut_slice(),
        );
        if scope.layer_norm {
            match optimizer {
                OptimizerKind::Sgd => {
                    for idx in 0..c {
                        self.final_norm_w[idx] += lr * scratch.grad_out[idx];
                    }
                    if let Some(b) = self.final_norm_b.as_mut() {
                        for idx in 0..c.min(b.len()) {
                            b[idx] += lr * scratch.grad_h[idx];
                        }
                    }
                }
                OptimizerKind::Adam => {
                    let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                    let adam = model_adam.as_mut().expect("adam state exists");
                    apply_adam_vec_update(
                        self.final_norm_w.as_mut_slice(),
                        scratch.grad_out.as_slice(),
                        &mut adam.final_norm_w,
                        cfg,
                    );
                    if let (Some(b), Some(bm)) =
                        (self.final_norm_b.as_mut(), adam.final_norm_b.as_mut())
                    {
                        apply_adam_vec_update(b.as_mut_slice(), scratch.grad_h.as_slice(), bm, cfg);
                    }
                }
            }
        }
        scratch
            .grad_h
            .as_mut_slice()
            .copy_from_slice(scratch.grad_norm.as_slice());

        // Layerwise reverse-mode pass.
        for layer_idx in (0..self.cfg.num_layers).rev() {
            let tr = &scratch.train_trace_layers[layer_idx];
            let st_new = &state.layers[layer_idx];
            let layer = &mut self.layers[layer_idx];

            scratch
                .grad_out
                .as_mut_slice()
                .copy_from_slice(scratch.grad_h.as_slice());

            // grad_y = out_proj^T @ grad_out.
            unsafe {
                kernel::gemv_t(
                    layer.out_proj_w.as_ptr(),
                    scratch.grad_out.as_ptr(),
                    scratch.grad_y.as_mut_ptr(),
                    c,
                    i,
                );
            }

            if scope.mixer_proj {
                match optimizer {
                    OptimizerKind::Sgd => {
                        for row in 0..c {
                            let g = scratch.grad_out[row];
                            let off = row * i;
                            for col in 0..i {
                                layer.out_proj_w[off + col] += lr * g * tr.y[col];
                            }
                        }
                        if let Some(b) = layer.out_proj_b.as_mut() {
                            for row in 0..c.min(b.len()) {
                                b[row] += lr * scratch.grad_out[row];
                            }
                        }
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                        let adam_layer =
                            &mut model_adam.as_mut().expect("adam state exists").layers[layer_idx];
                        apply_adam_outer_update(
                            layer.out_proj_w.as_mut_slice(),
                            c,
                            i,
                            scratch.grad_out.as_slice(),
                            tr.y.as_slice(),
                            &mut adam_layer.out_proj_w,
                            cfg,
                        );
                        if let (Some(b), Some(bm)) =
                            (layer.out_proj_b.as_mut(), adam_layer.out_proj_b.as_mut())
                        {
                            apply_adam_vec_update(
                                b.as_mut_slice(),
                                scratch.grad_out.as_slice(),
                                bm,
                                cfg,
                            );
                        }
                    }
                }
            }

            scratch
                .grad_residual
                .as_mut_slice()
                .copy_from_slice(scratch.grad_out.as_slice());
            scratch.grad_xz.zero();
            scratch.grad_conv.zero();
            scratch.grad_proj.zero();
            scratch.grad_dt_raw.zero();
            scratch.grad_u.zero();
            scratch.grad_b.zero();
            scratch.grad_c.zero();
            scratch.grad_ssm_d.zero();
            scratch.grad_ssm_a.zero();
            scratch.grad_ssm_a_log.zero();
            scratch.grad_conv_w.zero();
            scratch.grad_conv_b.zero();

            // y path, selective scan, and SSM params.
            for ch in 0..i {
                let g_y = scratch.grad_y[ch];
                let gate = tr.gate[ch];
                let y_pre = tr.y_pre[ch];
                let g_y_pre = g_y * gate;
                let g_gate = g_y * y_pre;
                scratch.grad_xz[i + ch] = g_gate * silu_grad(tr.xz[i + ch]);

                let conv = tr.conv_post[ch];
                let dt = tr.dt[ch];
                let xdt = conv * dt;
                let mut g_xdt = 0.0f32;
                let mut g_dt = 0.0f32;

                let mut g_conv = g_y_pre * layer.d[ch];
                if scope.mixer_ssm {
                    scratch.grad_ssm_d[ch] = g_y_pre * conv;
                }

                let row = ch * s;
                for j in 0..s {
                    let idx = row + j;
                    let c_j = tr.proj[r + s + j];
                    let b_j = tr.proj[r + j];
                    let s_prev = tr.ssm_prev[idx];
                    let s_new = st_new.ssm[idx];
                    let a_ij = layer.a[idx];

                    let g_ssm_new = g_y_pre * c_j;
                    scratch.grad_c[j] += g_y_pre * s_new;
                    g_xdt += g_ssm_new * b_j;
                    scratch.grad_b[j] += g_ssm_new * xdt;

                    let d_a = tr.d_a[idx];
                    let g_d_a = g_ssm_new * s_prev;
                    g_dt += g_d_a * d_a * a_ij;
                    if scope.mixer_ssm {
                        scratch.grad_ssm_a[idx] = g_d_a * d_a * dt;
                    }
                }

                g_conv += g_xdt * dt;
                g_dt += g_xdt * conv;
                let dt_pre = tr.dt_raw[ch] + layer.dt_proj_b[ch];
                scratch.grad_dt_raw[ch] = g_dt * sigmoid(dt_pre);
                scratch.grad_conv[ch] = g_conv;
            }

            if scope.mixer_ssm {
                match optimizer {
                    OptimizerKind::Sgd => {
                        if clip > 0.0 {
                            for idx in 0..i {
                                layer.d[idx] += lr * scratch.grad_ssm_d[idx].clamp(-clip, clip);
                            }
                            for idx in 0..(i * s) {
                                let g_log =
                                    (scratch.grad_ssm_a[idx] * layer.a[idx]).clamp(-clip, clip);
                                layer.a_log[idx] += lr * g_log;
                            }
                        } else {
                            for idx in 0..i {
                                layer.d[idx] += lr * scratch.grad_ssm_d[idx];
                            }
                            for idx in 0..(i * s) {
                                let g_log = scratch.grad_ssm_a[idx] * layer.a[idx];
                                layer.a_log[idx] += lr * g_log;
                            }
                        }
                        sync_a_from_a_log(&mut layer.a, &layer.a_log);
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                        let adam_layer =
                            &mut model_adam.as_mut().expect("adam state exists").layers[layer_idx];
                        for idx in 0..(i * s) {
                            scratch.grad_ssm_a_log[idx] = scratch.grad_ssm_a[idx] * layer.a[idx];
                        }
                        apply_adam_vec_update(
                            layer.d.as_mut_slice(),
                            scratch.grad_ssm_d.as_slice(),
                            &mut adam_layer.d,
                            cfg,
                        );
                        apply_adam_vec_update(
                            layer.a_log.as_mut_slice(),
                            scratch.grad_ssm_a_log.as_slice(),
                            &mut adam_layer.a,
                            cfg,
                        );
                        sync_a_from_a_log(&mut layer.a, &layer.a_log);
                    }
                }
            }

            // dt_proj: grad_u = W^T grad_dt_raw, and parameter grads.
            unsafe {
                kernel::gemv_t(
                    layer.dt_proj_w.as_ptr(),
                    scratch.grad_dt_raw.as_ptr(),
                    scratch.grad_u.as_mut_ptr(),
                    i,
                    r,
                );
            }
            if scope.mixer_proj {
                match optimizer {
                    OptimizerKind::Sgd => {
                        for ch in 0..i {
                            let g = scratch.grad_dt_raw[ch];
                            let off = ch * r;
                            for kk in 0..r {
                                layer.dt_proj_w[off + kk] += lr * g * tr.proj[kk];
                            }
                            layer.dt_proj_b[ch] += lr * g;
                        }
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                        let adam_layer =
                            &mut model_adam.as_mut().expect("adam state exists").layers[layer_idx];
                        apply_adam_outer_update(
                            layer.dt_proj_w.as_mut_slice(),
                            i,
                            r,
                            scratch.grad_dt_raw.as_slice(),
                            &tr.proj.as_slice()[..r],
                            &mut adam_layer.dt_proj_w,
                            cfg,
                        );
                        apply_adam_vec_update(
                            layer.dt_proj_b.as_mut_slice(),
                            scratch.grad_dt_raw.as_slice(),
                            &mut adam_layer.dt_proj_b,
                            cfg,
                        );
                    }
                }
            }

            for kk in 0..r {
                scratch.grad_proj[kk] = scratch.grad_u[kk];
            }
            for j in 0..s {
                scratch.grad_proj[r + j] = scratch.grad_b[j];
                scratch.grad_proj[r + s + j] = scratch.grad_c[j];
            }

            // x_proj: grad_conv += W^T grad_proj.
            unsafe {
                kernel::gemv_t(
                    layer.x_proj_w.as_ptr(),
                    scratch.grad_proj.as_ptr(),
                    scratch.grad_conv_pre.as_mut_ptr(),
                    r + 2 * s,
                    i,
                );
                kernel::add_inplace(
                    scratch.grad_conv.as_mut_ptr(),
                    scratch.grad_conv_pre.as_ptr(),
                    i,
                );
            }
            if scope.mixer_proj {
                match optimizer {
                    OptimizerKind::Sgd => {
                        for row in 0..(r + 2 * s) {
                            let g = scratch.grad_proj[row];
                            let off = row * i;
                            for col in 0..i {
                                layer.x_proj_w[off + col] += lr * g * tr.conv_post[col];
                            }
                        }
                        if let Some(b) = layer.x_proj_b.as_mut() {
                            for row in 0..(r + 2 * s).min(b.len()) {
                                b[row] += lr * scratch.grad_proj[row];
                            }
                        }
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                        let adam_layer =
                            &mut model_adam.as_mut().expect("adam state exists").layers[layer_idx];
                        apply_adam_outer_update(
                            layer.x_proj_w.as_mut_slice(),
                            r + 2 * s,
                            i,
                            scratch.grad_proj.as_slice(),
                            tr.conv_post.as_slice(),
                            &mut adam_layer.x_proj_w,
                            cfg,
                        );
                        if let (Some(b), Some(bm)) =
                            (layer.x_proj_b.as_mut(), adam_layer.x_proj_b.as_mut())
                        {
                            apply_adam_vec_update(
                                b.as_mut_slice(),
                                scratch.grad_proj.as_slice(),
                                bm,
                                cfg,
                            );
                        }
                    }
                }
            }

            // conv + silu + in-proj(x branch)
            scratch.grad_conv_pre.zero();
            for ch in 0..i {
                scratch.grad_conv_pre[ch] = scratch.grad_conv[ch] * silu_grad(tr.conv_pre[ch]);
            }

            for ch in 0..i {
                let g = scratch.grad_conv_pre[ch];
                let base = ch * self.cfg.conv_kernel;
                let mut ring = tr.conv_pos_prev;
                let w0 = layer.conv_w[base];
                scratch.grad_xz[ch] += g * w0;
                for tap in 0..self.cfg.conv_kernel {
                    let val = if ring == tr.conv_pos_prev {
                        tr.xz[ch]
                    } else {
                        tr.conv_prev[base + ring]
                    };
                    if scope.mixer_conv {
                        scratch.grad_conv_w[base + tap] = g * val;
                    }
                    ring = if ring == 0 {
                        self.cfg.conv_kernel - 1
                    } else {
                        ring - 1
                    };
                }
                if scope.mixer_conv && layer.conv_b.is_some() {
                    scratch.grad_conv_b[ch] = g;
                }
            }

            if scope.mixer_conv {
                match optimizer {
                    OptimizerKind::Sgd => {
                        if clip > 0.0 {
                            for idx in 0..layer.conv_w.len() {
                                layer.conv_w[idx] +=
                                    lr * scratch.grad_conv_w[idx].clamp(-clip, clip);
                            }
                        } else {
                            for idx in 0..layer.conv_w.len() {
                                layer.conv_w[idx] += lr * scratch.grad_conv_w[idx];
                            }
                        }
                        if let Some(bias) = layer.conv_b.as_mut() {
                            if clip > 0.0 {
                                for idx in 0..bias.len().min(i) {
                                    bias[idx] += lr * scratch.grad_conv_b[idx].clamp(-clip, clip);
                                }
                            } else {
                                for idx in 0..bias.len().min(i) {
                                    bias[idx] += lr * scratch.grad_conv_b[idx];
                                }
                            }
                        }
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                        let adam_layer =
                            &mut model_adam.as_mut().expect("adam state exists").layers[layer_idx];
                        apply_adam_vec_update(
                            layer.conv_w.as_mut_slice(),
                            scratch.grad_conv_w.as_slice(),
                            &mut adam_layer.conv_w,
                            cfg,
                        );
                        if let (Some(bias), Some(bm)) =
                            (layer.conv_b.as_mut(), adam_layer.conv_b.as_mut())
                        {
                            apply_adam_vec_update(
                                bias.as_mut_slice(),
                                scratch.grad_conv_b.as_slice(),
                                bm,
                                cfg,
                            );
                        }
                    }
                }
            }

            // in_proj backward: grad_norm = W^T grad_xz
            unsafe {
                kernel::gemv_t(
                    layer.in_proj_w.as_ptr(),
                    scratch.grad_xz.as_ptr(),
                    scratch.grad_norm.as_mut_ptr(),
                    2 * i,
                    c,
                );
            }
            if scope.mixer_proj {
                match optimizer {
                    OptimizerKind::Sgd => {
                        for row in 0..(2 * i) {
                            let g = scratch.grad_xz[row];
                            let off = row * c;
                            for col in 0..c {
                                layer.in_proj_w[off + col] += lr * g * tr.norm[col];
                            }
                        }
                        if let Some(b) = layer.in_proj_b.as_mut() {
                            for row in 0..(2 * i).min(b.len()) {
                                b[row] += lr * scratch.grad_xz[row];
                            }
                        }
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                        let adam_layer =
                            &mut model_adam.as_mut().expect("adam state exists").layers[layer_idx];
                        apply_adam_outer_update(
                            layer.in_proj_w.as_mut_slice(),
                            2 * i,
                            c,
                            scratch.grad_xz.as_slice(),
                            tr.norm.as_slice(),
                            &mut adam_layer.in_proj_w,
                            cfg,
                        );
                        if let (Some(b), Some(bm)) =
                            (layer.in_proj_b.as_mut(), adam_layer.in_proj_b.as_mut())
                        {
                            apply_adam_vec_update(
                                b.as_mut_slice(),
                                scratch.grad_xz.as_slice(),
                                bm,
                                cfg,
                            );
                        }
                    }
                }
            }

            // layer norm backward + residual combine.
            rms_norm_backward(
                tr.h_in.as_slice(),
                layer.norm_w.as_slice(),
                scratch.grad_norm.as_slice(),
                self.cfg.layer_norm_eps,
                scratch.grad_h.as_mut_slice(),
                scratch.grad_out.as_mut_slice(),
            );
            if scope.layer_norm {
                match optimizer {
                    OptimizerKind::Sgd => {
                        for idx in 0..c {
                            layer.norm_w[idx] += lr * scratch.grad_out[idx];
                        }
                        if let Some(b) = layer.norm_b.as_mut() {
                            for idx in 0..c.min(b.len()) {
                                b[idx] += lr * scratch.grad_norm[idx];
                            }
                        }
                    }
                    OptimizerKind::Adam => {
                        let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                        let adam_layer =
                            &mut model_adam.as_mut().expect("adam state exists").layers[layer_idx];
                        apply_adam_vec_update(
                            layer.norm_w.as_mut_slice(),
                            scratch.grad_out.as_slice(),
                            &mut adam_layer.norm_w,
                            cfg,
                        );
                        if let (Some(b), Some(bm)) =
                            (layer.norm_b.as_mut(), adam_layer.norm_b.as_mut())
                        {
                            apply_adam_vec_update(
                                b.as_mut_slice(),
                                scratch.grad_norm.as_slice(),
                                bm,
                                cfg,
                            );
                        }
                    }
                }
            }

            for idx in 0..c {
                scratch.grad_h[idx] += scratch.grad_residual[idx];
            }
        }

        if scope.embed {
            let tok = scratch
                .train_token
                .min(self.cfg.vocab_size.saturating_sub(1));
            let row_off = tok * c;
            match optimizer {
                OptimizerKind::Sgd => {
                    for j in 0..c {
                        let g = if clip > 0.0 {
                            scratch.grad_h[j].clamp(-clip, clip)
                        } else {
                            scratch.grad_h[j]
                        };
                        self.embeddings[row_off + j] += lr * g;
                    }
                }
                OptimizerKind::Adam => {
                    let cfg = adam_cfg.as_ref().expect("adam cfg initialized");
                    let adam = model_adam.as_mut().expect("adam state exists");
                    let pm = adam.embeddings.m.as_mut_slice();
                    let pv = adam.embeddings.v.as_mut_slice();
                    let emb = self.embeddings.as_mut_slice();
                    for j in 0..c {
                        let idx = row_off + j;
                        let g = scratch.grad_h[j];
                        pm[idx] = cfg.b1 * pm[idx] + (1.0 - cfg.b1) * g;
                        pv[idx] = cfg.b2 * pv[idx] + (1.0 - cfg.b2) * g * g;
                        let m_hat = pm[idx] / cfg.bias_corr1;
                        let v_hat = pv[idx] / cfg.bias_corr2;
                        emb[idx] += cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
                    }
                }
            }
        }

        Ok(())
    }

    fn load_native(weights: &Weights) -> Result<Self> {
        let emb = weights.require("model.embeddings.weight")?;
        if emb.shape().len() != 2 {
            bail!("model.embeddings.weight must be rank-2");
        }
        let vocab_size = emb.shape()[0];
        let hidden_size = emb.shape()[1];

        let num_layers = count_layers(weights, "model.layers.", "mixer.in_proj.weight")?;
        if num_layers == 0 {
            bail!("no Mamba layers found in native checkpoint");
        }

        let first_in = weights.require("model.layers.0.mixer.in_proj.weight")?;
        let first_conv = weights.require("model.layers.0.mixer.conv1d.weight")?;
        let first_a = weights.require("model.layers.0.mixer.A_log")?;
        let first_dt = weights.require("model.layers.0.mixer.dt_proj.weight")?;

        let inner_size =
            infer_in_proj_inner(first_in, hidden_size, "model.layers.0.mixer.in_proj.weight")?;
        let conv_kernel =
            infer_conv_kernel(first_conv, inner_size, "model.layers.0.mixer.conv1d.weight")?;
        let state_size = infer_state_size(first_a, inner_size, "model.layers.0.mixer.A_log")?;
        let dt_rank = infer_dt_rank(first_dt, inner_size, "model.layers.0.mixer.dt_proj.weight")?;

        let cfg = Config {
            vocab_size,
            hidden_size,
            num_layers,
            inner_size,
            state_size,
            conv_kernel,
            dt_rank,
            layer_norm_eps: 1e-5,
        };
        cfg.validate()?;

        let embeddings = tensor_from(emb)?;
        let final_norm_w = tensor_from(weights.require("model.norm.weight")?)?;
        let final_norm_b = optional_tensor_from(weights, "model.norm.bias")?;
        let lm_head = if let Some(t) = weights.get("lm_head.weight") {
            tensor_from(t)?
        } else {
            embeddings.clone()
        };
        let lm_head_b = optional_tensor_from(weights, "lm_head.bias")?;

        let mut layers = Vec::with_capacity(num_layers);
        for idx in 0..num_layers {
            let root = format!("model.layers.{idx}");
            let mixer = format!("{root}.mixer");

            let norm_w = tensor_from(weights.require(&format!("{root}.norm.weight"))?)?;
            let norm_b = optional_tensor_from(weights, &format!("{root}.norm.bias"))?;

            let in_proj_w = tensor_from(weights.require(&format!("{mixer}.in_proj.weight"))?)?;
            let in_proj_b = optional_tensor_from(weights, &format!("{mixer}.in_proj.bias"))?;

            let conv_w = tensor_from_conv(
                weights.require(&format!("{mixer}.conv1d.weight"))?,
                inner_size,
            )?;
            let conv_b = optional_tensor_from(weights, &format!("{mixer}.conv1d.bias"))?;

            let x_proj_w = tensor_from(weights.require(&format!("{mixer}.x_proj.weight"))?)?;
            let x_proj_b = optional_tensor_from(weights, &format!("{mixer}.x_proj.bias"))?;

            let dt_proj_w = tensor_from(weights.require(&format!("{mixer}.dt_proj.weight"))?)?;
            let dt_proj_b = tensor_from(weights.require(&format!("{mixer}.dt_proj.bias"))?)?;

            let a_log = tensor_from(weights.require(&format!("{mixer}.A_log"))?)?;
            let a = a_from_a_log_tensor(&a_log);
            let d = tensor_from(weights.require(&format!("{mixer}.D"))?)?;

            let out_proj_w = tensor_from(weights.require(&format!("{mixer}.out_proj.weight"))?)?;
            let out_proj_b = optional_tensor_from(weights, &format!("{mixer}.out_proj.bias"))?;

            validate_layer_shapes(
                &cfg,
                idx,
                &norm_w,
                norm_b.as_ref(),
                &in_proj_w,
                in_proj_b.as_ref(),
                &conv_w,
                conv_b.as_ref(),
                &x_proj_w,
                x_proj_b.as_ref(),
                &dt_proj_w,
                &dt_proj_b,
                &a,
                &d,
                &out_proj_w,
                out_proj_b.as_ref(),
            )?;

            layers.push(LayerWeights {
                norm_w,
                norm_b,
                in_proj_w,
                in_proj_b,
                conv_w,
                conv_b,
                x_proj_w,
                x_proj_b,
                dt_proj_w,
                dt_proj_b,
                a_log,
                a,
                d,
                out_proj_w,
                out_proj_b,
            });
        }

        Ok(Self {
            cfg,
            embeddings,
            final_norm_w,
            final_norm_b,
            lm_head,
            lm_head_b,
            layers,
        })
    }

    fn load_official(weights: &Weights) -> Result<Self> {
        let emb = weights.require("backbone.embedding.weight")?;
        if emb.shape().len() != 2 {
            bail!("backbone.embedding.weight must be rank-2");
        }
        let vocab_size = emb.shape()[0];
        let hidden_size = emb.shape()[1];

        let num_layers = count_layers(weights, "backbone.layers.", "mixer.in_proj.weight")?;
        if num_layers == 0 {
            bail!("no Mamba layers found in official checkpoint");
        }

        let first_in = weights.require("backbone.layers.0.mixer.in_proj.weight")?;
        let first_conv = weights.require("backbone.layers.0.mixer.conv1d.weight")?;
        let first_a = weights.require("backbone.layers.0.mixer.A_log")?;
        let first_dt = weights.require("backbone.layers.0.mixer.dt_proj.weight")?;

        let inner_size = infer_in_proj_inner(
            first_in,
            hidden_size,
            "backbone.layers.0.mixer.in_proj.weight",
        )?;
        let conv_kernel = infer_conv_kernel(
            first_conv,
            inner_size,
            "backbone.layers.0.mixer.conv1d.weight",
        )?;
        let state_size = infer_state_size(first_a, inner_size, "backbone.layers.0.mixer.A_log")?;
        let dt_rank = infer_dt_rank(
            first_dt,
            inner_size,
            "backbone.layers.0.mixer.dt_proj.weight",
        )?;

        let cfg = Config {
            vocab_size,
            hidden_size,
            num_layers,
            inner_size,
            state_size,
            conv_kernel,
            dt_rank,
            layer_norm_eps: 1e-5,
        };
        cfg.validate()?;

        let embeddings = tensor_from(emb)?;
        let final_norm_w = tensor_from(weights.require("norm_f.weight")?)?;
        let final_norm_b = optional_tensor_from(weights, "norm_f.bias")?;
        let lm_head = if let Some(t) = weights.get("lm_head.weight") {
            tensor_from(t)?
        } else {
            embeddings.clone()
        };
        let lm_head_b = optional_tensor_from(weights, "lm_head.bias")?;

        let mut layers = Vec::with_capacity(num_layers);
        for idx in 0..num_layers {
            let root = format!("backbone.layers.{idx}");
            let mixer = format!("{root}.mixer");

            let norm_w = tensor_from(weights.require(&format!("{root}.norm.weight"))?)?;
            let norm_b = optional_tensor_from(weights, &format!("{root}.norm.bias"))?;

            let in_proj_w = tensor_from(weights.require(&format!("{mixer}.in_proj.weight"))?)?;
            let in_proj_b = optional_tensor_from(weights, &format!("{mixer}.in_proj.bias"))?;

            let conv_w = tensor_from_conv(
                weights.require(&format!("{mixer}.conv1d.weight"))?,
                inner_size,
            )?;
            let conv_b = optional_tensor_from(weights, &format!("{mixer}.conv1d.bias"))?;

            let x_proj_w = tensor_from(weights.require(&format!("{mixer}.x_proj.weight"))?)?;
            let x_proj_b = optional_tensor_from(weights, &format!("{mixer}.x_proj.bias"))?;

            let dt_proj_w = tensor_from(weights.require(&format!("{mixer}.dt_proj.weight"))?)?;
            let dt_proj_b = tensor_from(weights.require(&format!("{mixer}.dt_proj.bias"))?)?;

            let a_log = tensor_from(weights.require(&format!("{mixer}.A_log"))?)?;
            let a = a_from_a_log_tensor(&a_log);
            let d = tensor_from(weights.require(&format!("{mixer}.D"))?)?;

            let out_proj_w = tensor_from(weights.require(&format!("{mixer}.out_proj.weight"))?)?;
            let out_proj_b = optional_tensor_from(weights, &format!("{mixer}.out_proj.bias"))?;

            validate_layer_shapes(
                &cfg,
                idx,
                &norm_w,
                norm_b.as_ref(),
                &in_proj_w,
                in_proj_b.as_ref(),
                &conv_w,
                conv_b.as_ref(),
                &x_proj_w,
                x_proj_b.as_ref(),
                &dt_proj_w,
                &dt_proj_b,
                &a,
                &d,
                &out_proj_w,
                out_proj_b.as_ref(),
            )?;

            layers.push(LayerWeights {
                norm_w,
                norm_b,
                in_proj_w,
                in_proj_b,
                conv_w,
                conv_b,
                x_proj_w,
                x_proj_b,
                dt_proj_w,
                dt_proj_b,
                a_log,
                a,
                d,
                out_proj_w,
                out_proj_b,
            });
        }

        Ok(Self {
            cfg,
            embeddings,
            final_norm_w,
            final_norm_b,
            lm_head,
            lm_head_b,
            layers,
        })
    }
}

fn tensor_from(t: &WeightTensor) -> Result<Tensor1D> {
    Ok(Tensor1D::from_vec(t.data().to_vec()))
}

fn a_from_a_log_tensor(a_log: &Tensor1D) -> Tensor1D {
    let mut out = a_log.as_slice().to_vec();
    for v in &mut out {
        *v = -v.exp();
    }
    Tensor1D::from_vec(out)
}

#[inline(always)]
fn sync_a_from_a_log(a: &mut Tensor1D, a_log: &Tensor1D) {
    let dst = a.as_mut_slice();
    let src = a_log.as_slice();
    debug_assert_eq!(dst.len(), src.len());
    for idx in 0..dst.len().min(src.len()) {
        dst[idx] = -src[idx].exp();
    }
}

fn optional_tensor_from(weights: &Weights, name: &str) -> Result<Option<Tensor1D>> {
    match weights.get(name) {
        Some(t) => Ok(Some(tensor_from(t)?)),
        None => Ok(None),
    }
}

fn tensor_from_conv(t: &WeightTensor, inner_size: usize) -> Result<Tensor1D> {
    match t.shape() {
        [i, k] if *i == inner_size => Ok(Tensor1D::from_vec(t.data().to_vec())),
        [i, one, k] if *i == inner_size && *one == 1 => {
            let mut out = Vec::with_capacity(inner_size * k);
            let src = t.data();
            for ch in 0..inner_size {
                let off = ch * k;
                out.extend_from_slice(&src[off..off + k]);
            }
            Ok(Tensor1D::from_vec(out))
        }
        other => bail!("unexpected conv1d weight shape {:?}", other),
    }
}

fn count_layers(weights: &Weights, prefix: &str, suffix: &str) -> Result<usize> {
    let mut max_layer = None::<usize>;
    for name in weights.tensor_names() {
        let Some(rest) = name.strip_prefix(prefix) else {
            continue;
        };
        let Some((idx_s, tail)) = rest.split_once('.') else {
            continue;
        };
        if tail != suffix {
            continue;
        }
        let idx = idx_s
            .parse::<usize>()
            .with_context(|| format!("invalid layer index in tensor name '{name}'"))?;
        max_layer = Some(max_layer.map_or(idx, |m| m.max(idx)));
    }
    Ok(max_layer.map_or(0, |m| m + 1))
}

fn infer_in_proj_inner(t: &WeightTensor, hidden: usize, name: &str) -> Result<usize> {
    let shape = t.shape();
    if shape.len() != 2 {
        bail!("{name} must be rank-2, got {:?}", shape);
    }
    if shape[1] != hidden {
        bail!("{name} expected cols={}, got {}", hidden, shape[1]);
    }
    if !shape[0].is_multiple_of(2) {
        bail!("{name} first dim {} must be 2*d_inner", shape[0]);
    }
    Ok(shape[0] / 2)
}

fn infer_conv_kernel(t: &WeightTensor, inner: usize, name: &str) -> Result<usize> {
    let shape = t.shape();
    match shape {
        [i, k] if *i == inner => Ok(*k),
        [i, one, k] if *i == inner && *one == 1 => Ok(*k),
        _ => bail!("{name} shape {:?} incompatible with d_inner={inner}", shape),
    }
}

fn infer_state_size(t: &WeightTensor, inner: usize, name: &str) -> Result<usize> {
    let shape = t.shape();
    if shape.len() != 2 {
        bail!("{name} must be rank-2, got {:?}", shape);
    }
    if shape[0] != inner {
        bail!("{name} expected rows={}, got {}", inner, shape[0]);
    }
    Ok(shape[1])
}

fn infer_dt_rank(t: &WeightTensor, inner: usize, name: &str) -> Result<usize> {
    let shape = t.shape();
    if shape.len() != 2 {
        bail!("{name} must be rank-2, got {:?}", shape);
    }
    if shape[0] != inner {
        bail!("{name} expected rows={}, got {}", inner, shape[0]);
    }
    Ok(shape[1])
}

#[allow(clippy::too_many_arguments)]
fn validate_layer_shapes(
    cfg: &Config,
    idx: usize,
    norm_w: &Tensor1D,
    norm_b: Option<&Tensor1D>,
    in_proj_w: &Tensor1D,
    in_proj_b: Option<&Tensor1D>,
    conv_w: &Tensor1D,
    conv_b: Option<&Tensor1D>,
    x_proj_w: &Tensor1D,
    x_proj_b: Option<&Tensor1D>,
    dt_proj_w: &Tensor1D,
    dt_proj_b: &Tensor1D,
    a: &Tensor1D,
    d: &Tensor1D,
    out_proj_w: &Tensor1D,
    out_proj_b: Option<&Tensor1D>,
) -> Result<()> {
    let c = cfg.hidden_size;
    let i = cfg.inner_size;
    let s = cfg.state_size;
    let k = cfg.conv_kernel;
    let r = cfg.dt_rank;

    let check = |cond: bool, msg: String| -> Result<()> {
        if cond {
            Ok(())
        } else {
            bail!("layer {idx}: {msg}")
        }
    };

    check(
        norm_w.len() == c,
        format!("norm.weight len {} != hidden {c}", norm_w.len()),
    )?;
    if let Some(b) = norm_b {
        check(
            b.len() == c,
            format!("norm.bias len {} != hidden {c}", b.len()),
        )?;
    }

    check(
        in_proj_w.len() == (2 * i) * c,
        format!("in_proj.weight len {} != {}", in_proj_w.len(), (2 * i) * c),
    )?;
    if let Some(b) = in_proj_b {
        check(
            b.len() == 2 * i,
            format!("in_proj.bias len {} != {}", b.len(), 2 * i),
        )?;
    }

    check(
        conv_w.len() == i * k,
        format!("conv1d.weight len {} != {}", conv_w.len(), i * k),
    )?;
    if let Some(b) = conv_b {
        check(b.len() == i, format!("conv1d.bias len {} != {i}", b.len()))?;
    }

    check(
        x_proj_w.len() == (r + 2 * s) * i,
        format!(
            "x_proj.weight len {} != {}",
            x_proj_w.len(),
            (r + 2 * s) * i
        ),
    )?;
    if let Some(b) = x_proj_b {
        check(
            b.len() == r + 2 * s,
            format!("x_proj.bias len {} != {}", b.len(), r + 2 * s),
        )?;
    }

    check(
        dt_proj_w.len() == i * r,
        format!("dt_proj.weight len {} != {}", dt_proj_w.len(), i * r),
    )?;
    check(
        dt_proj_b.len() == i,
        format!("dt_proj.bias len {} != {i}", dt_proj_b.len()),
    )?;

    check(a.len() == i * s, format!("A len {} != {}", a.len(), i * s))?;
    check(d.len() == i, format!("D len {} != {i}", d.len()))?;

    check(
        out_proj_w.len() == c * i,
        format!("out_proj.weight len {} != {}", out_proj_w.len(), c * i),
    )?;
    if let Some(b) = out_proj_b {
        check(
            b.len() == c,
            format!("out_proj.bias len {} != {c}", b.len()),
        )?;
    }

    Ok(())
}

fn rms_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    debug_assert_eq!(input.len(), weight.len());
    debug_assert_eq!(input.len(), out.len());
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), input.len());
    }

    let mut mean_sq = 0.0f32;
    for &x in input {
        mean_sq += x * x;
    }
    mean_sq /= input.len().max(1) as f32;
    let inv = (mean_sq + eps).sqrt().recip();

    if let Some(b) = bias {
        for idx in 0..input.len() {
            out[idx] = input[idx] * inv * weight[idx] + b[idx];
        }
    } else {
        for idx in 0..input.len() {
            out[idx] = input[idx] * inv * weight[idx];
        }
    }
}

fn rms_norm_backward(
    input: &[f32],
    weight: &[f32],
    grad_out: &[f32],
    eps: f32,
    grad_input: &mut [f32],
    grad_weight: &mut [f32],
) {
    debug_assert_eq!(input.len(), weight.len());
    debug_assert_eq!(input.len(), grad_out.len());
    debug_assert_eq!(input.len(), grad_input.len());
    debug_assert_eq!(input.len(), grad_weight.len());

    let n = input.len().max(1) as f32;
    let mut mean_sq = 0.0f32;
    for &x in input {
        mean_sq += x * x;
    }
    mean_sq /= n;
    let inv = (mean_sq + eps).sqrt().recip();

    let mut s = 0.0f32;
    for idx in 0..input.len() {
        let gw = grad_out[idx] * weight[idx];
        grad_weight[idx] = grad_out[idx] * input[idx] * inv;
        s += gw * input[idx];
    }
    let coeff = -s * inv * inv * inv / n;
    for idx in 0..input.len() {
        grad_input[idx] = grad_out[idx] * weight[idx] * inv + input[idx] * coeff;
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
    let b1 = step.b1;
    let b2 = step.b2;
    let one_m_b1 = 1.0 - b1;
    let one_m_b2 = 1.0 - b2;
    let lr = step.lr;
    let eps = step.eps;
    let inv_bc1 = 1.0 / step.bias_corr1;
    let inv_bc2 = 1.0 / step.bias_corr2;
    let do_clip = step.clip > 0.0;
    let clip = step.clip;
    let m = adam.m.as_mut_slice();
    let v = adam.v.as_mut_slice();
    if do_clip {
        for idx in 0..n {
            let g = grad[idx].clamp(-clip, clip);
            let mm = b1 * m[idx] + one_m_b1 * g;
            let vv = b2 * v[idx] + one_m_b2 * g * g;
            m[idx] = mm;
            v[idx] = vv;
            let m_hat = mm * inv_bc1;
            let v_hat = vv * inv_bc2;
            param[idx] += lr * m_hat / (v_hat.sqrt() + eps);
        }
    } else {
        let mut idx = 0usize;
        unsafe {
            let b1v = f32x8::splat(b1);
            let b2v = f32x8::splat(b2);
            let one_b1v = f32x8::splat(one_m_b1);
            let one_b2v = f32x8::splat(one_m_b2);
            let inv_bc1v = f32x8::splat(inv_bc1);
            let inv_bc2v = f32x8::splat(inv_bc2);
            let lrv = f32x8::splat(lr);
            let epsv = f32x8::splat(eps);
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
            let mm = b1 * m[idx] + one_m_b1 * g;
            let vv = b2 * v[idx] + one_m_b2 * g * g;
            m[idx] = mm;
            v[idx] = vv;
            let m_hat = mm * inv_bc1;
            let v_hat = vv * inv_bc2;
            param[idx] += lr * m_hat / (v_hat.sqrt() + eps);
            idx += 1;
        }
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
    let rows = rows.min(left.len());
    let cols = cols.min(right.len());
    let n = param.len().min(adam.m.len()).min(adam.v.len());
    if rows == 0 || cols == 0 || n == 0 {
        return;
    }
    let b1 = step.b1;
    let b2 = step.b2;
    let one_m_b1 = 1.0 - b1;
    let one_m_b2 = 1.0 - b2;
    let lr = step.lr;
    let eps = step.eps;
    let inv_bc1 = 1.0 / step.bias_corr1;
    let inv_bc2 = 1.0 / step.bias_corr2;
    let do_clip = step.clip > 0.0;
    let clip = step.clip;
    let m = adam.m.as_mut_slice();
    let v = adam.v.as_mut_slice();
    let b1v = f32x8::splat(b1);
    let b2v = f32x8::splat(b2);
    let one_b1v = f32x8::splat(one_m_b1);
    let one_b2v = f32x8::splat(one_m_b2);
    let inv_bc1v = f32x8::splat(inv_bc1);
    let inv_bc2v = f32x8::splat(inv_bc2);
    let epsv = f32x8::splat(eps);
    let lrv = f32x8::splat(lr);
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
                let mm = b1 * m[idx] + one_m_b1 * g;
                let vv = b2 * v[idx] + one_m_b2 * g * g;
                m[idx] = mm;
                v[idx] = vv;
                let m_hat = mm * inv_bc1;
                let v_hat = vv * inv_bc2;
                param[idx] += lr * m_hat / (v_hat.sqrt() + eps);
            }
        } else {
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
                let mm = b1 * m[idx] + one_m_b1 * g;
                let vv = b2 * v[idx] + one_m_b2 * g * g;
                m[idx] = mm;
                v[idx] = vv;
                let m_hat = mm * inv_bc1;
                let v_hat = vv * inv_bc2;
                param[idx] += lr * m_hat / (v_hat.sqrt() + eps);
                col += 1;
            }
        }
    }
}

fn depthwise_conv_step(
    x: &[f32],
    conv_w: &Tensor1D,
    conv_b: Option<&Tensor1D>,
    conv_kernel: usize,
    state: &mut LayerState,
    out: &mut [f32],
) {
    let inner = x.len();
    debug_assert_eq!(out.len(), inner);
    debug_assert_eq!(conv_w.len(), inner * conv_kernel);

    let pos = state.conv_pos;
    let conv_state = state.conv.as_mut_slice();
    let weight = conv_w.as_slice();

    for ch in 0..inner {
        let base = ch * conv_kernel;
        conv_state[base + pos] = x[ch];

        let mut acc = conv_b.as_ref().map_or(0.0, |b| b[ch]);
        let mut ring_idx = pos;
        for tap in 0..conv_kernel {
            acc += conv_state[base + ring_idx] * weight[base + tap];
            ring_idx = if ring_idx == 0 {
                conv_kernel - 1
            } else {
                ring_idx - 1
            };
        }
        out[ch] = acc;
    }

    state.conv_pos = if pos + 1 == conv_kernel { 0 } else { pos + 1 };
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

struct MambaRng {
    state: u64,
}

impl MambaRng {
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
fn init_uniform(t: &mut Tensor1D, rng: &mut MambaRng, scale: f32) {
    for v in t.as_mut_slice() {
        let r = rng.next_f32() - 0.5;
        *v = r * 2.0 * scale;
    }
}

#[inline]
fn init_const(t: &mut Tensor1D, value: f32) {
    t.as_mut_slice().fill(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_log_prob(model: &Model, token: u32, target: u8) -> f32 {
        let mut state = model.new_state();
        let mut scratch = ScratchBuffers::new(model.config());
        let logits = model.forward(&mut scratch, token, &mut state);
        let mut max_logit = f32::NEG_INFINITY;
        for &logit in logits {
            max_logit = max_logit.max(logit);
        }
        let mut denom = 0.0f64;
        for &logit in logits {
            denom += ((logit - max_logit) as f64).exp();
        }
        let p = ((logits[target as usize] - max_logit) as f64).exp() / denom;
        p.max(1e-30).ln() as f32
    }

    #[test]
    fn forward_is_deterministic_for_same_input_and_state() {
        let cfg = Config {
            vocab_size: 256,
            hidden_size: 64,
            num_layers: 2,
            inner_size: 96,
            state_size: 8,
            conv_kernel: 4,
            dt_rank: 8,
            layer_norm_eps: 1e-5,
        };
        let model = Model::new_random(cfg.clone(), 1234).expect("random model");
        let mut s1 = model.new_state();
        let mut s2 = model.new_state();
        let mut b1 = ScratchBuffers::new(&cfg);
        let mut b2 = ScratchBuffers::new(&cfg);

        let seq = b"deterministic mamba";
        for &tok in seq {
            let l1 = model.forward(&mut b1, tok as u32, &mut s1).to_vec();
            let l2 = model.forward(&mut b2, tok as u32, &mut s2).to_vec();
            assert_eq!(l1.len(), l2.len());
            for (a, b) in l1.iter().zip(l2.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn online_embed_gradient_matches_finite_difference() {
        let cfg = Config {
            vocab_size: 256,
            hidden_size: 16,
            num_layers: 2,
            inner_size: 24,
            state_size: 4,
            conv_kernel: 3,
            dt_rank: 4,
            layer_norm_eps: 1e-5,
        };
        let token = 7u32;
        let target = 19u8;
        let lr = 1e-3f32;
        let eps = 1e-3f32;

        let model = Model::new_random(cfg.clone(), 99).expect("random model");
        let mut state = model.new_state();
        let mut scratch = ScratchBuffers::new(&cfg);
        scratch.set_capture_train_trace(true);
        let logits = model.forward(&mut scratch, token, &mut state);

        let mut pdf = vec![0.0f64; cfg.vocab_size];
        let mut max_logit = f32::NEG_INFINITY;
        for &logit in logits {
            max_logit = max_logit.max(logit);
        }
        let mut denom = 0.0f64;
        for &logit in logits {
            denom += ((logit - max_logit) as f64).exp();
        }
        for (idx, out) in pdf.iter_mut().enumerate() {
            *out = ((logits[idx] - max_logit) as f64).exp() / denom;
        }

        let base = model.clone();
        let mut trained = base.clone();
        let mut train_scratch = scratch.clone();
        trained
            .online_train_step_bptt1(
                &mut train_scratch,
                &state,
                target,
                &pdf,
                TrainScopeMask {
                    embed: true,
                    ..TrainScopeMask::default()
                },
                OptimizerKind::Sgd,
                lr,
                0.0,
                &mut 0usize,
                None,
                None,
                None,
                None,
            )
            .expect("training step");

        let param_idx = token as usize * cfg.hidden_size;
        let analytic = (trained.embeddings[param_idx] - base.embeddings[param_idx]) / lr;

        let mut plus = base.clone();
        plus.embeddings[param_idx] += eps;
        let mut minus = base.clone();
        minus.embeddings[param_idx] -= eps;
        let numeric = (target_log_prob(&plus, token, target)
            - target_log_prob(&minus, token, target))
            / (2.0 * eps);

        let diff = (analytic - numeric).abs();
        let scale = analytic.abs().max(numeric.abs()).max(1.0);
        assert!(
            diff <= 2e-2 * scale,
            "analytic={analytic} numeric={numeric} diff={diff}"
        );
    }
}
