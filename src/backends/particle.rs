//! Particle-latent filter ensemble rate backend.
//!
//! Implements a deterministic-by-default sequential particle model with latent
//! cells, selector/rule dynamics, online SGD, Bayesian particle weighting,
//! and resample+mutation.

use crate::ParticleSpec;
use crate::simd_math::{axpy_wide, dot_wide, logsumexp_wide, max_wide};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Deterministic hash utilities
// ---------------------------------------------------------------------------

/// Deterministic hash producing a u64 from four index components.
/// Uses a simple multiply-xor-shift chain seeded by `seed`.
#[inline]
fn det_hash(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut h = seed;
    h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(a);
    h ^= h >> 33;
    h = h.wrapping_mul(0x4cf5ad432745937f).wrapping_add(b);
    h ^= h >> 33;
    h = h.wrapping_mul(0x6c62272e07bb0142).wrapping_add(c);
    h ^= h >> 33;
    h
}

/// Map a hash to a deterministic f64 in [-1, 1].
#[inline]
fn hash_to_f64(h: u64) -> f64 {
    // Map to [0, 1) then scale to [-1, 1)
    let u = (h >> 11) as f64 / ((1u64 << 53) as f64);
    u * 2.0 - 1.0
}

/// Deterministic init value for a parameter, small magnitude.
#[inline]
fn init_param(seed: u64, layer: u64, row: u64, col: u64, scale: f64) -> f64 {
    hash_to_f64(det_hash(seed, layer, row, col)) * scale
}

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

#[inline]
fn clip(x: f64, limit: f64) -> f64 {
    x.clamp(-limit, limit)
}

fn softmax_inplace(xs: &mut [f64]) {
    let max_v = max_wide(xs);
    let mut sum = 0.0;
    for x in xs.iter_mut() {
        *x = (*x - max_v).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in xs.iter_mut() {
            *x *= inv;
        }
    }
}

fn log_softmax_with_floor(logits: &[f64], out: &mut [f64], min_prob: f64) {
    let max_v = max_wide(logits);
    let mut sum = 0.0;
    for &l in logits {
        sum += (l - max_v).exp();
    }
    let log_z = max_v + sum.ln();
    let log_floor = min_prob.ln();
    // First pass: compute raw log-softmax with floor
    let mut log_sum_exp_floor = f64::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        let lp = (l - log_z).max(log_floor);
        out[i] = lp;
        // accumulate for renormalization
        if lp > log_sum_exp_floor {
            let diff = log_sum_exp_floor - lp;
            if diff.is_finite() {
                log_sum_exp_floor = lp + (1.0 + diff.exp()).ln();
            } else {
                log_sum_exp_floor = lp;
            }
        } else {
            let diff = lp - log_sum_exp_floor;
            if diff.is_finite() {
                log_sum_exp_floor = log_sum_exp_floor + (1.0 + diff.exp()).ln();
            }
        }
    }
    // Renormalize so that sum(exp(out)) = 1
    if log_sum_exp_floor.is_finite() {
        for v in out.iter_mut() {
            *v -= log_sum_exp_floor;
        }
    }
}

// ---------------------------------------------------------------------------
// MLP layer: y = relu(W * x + b) or y = W * x + b
// ---------------------------------------------------------------------------

/// Dense layer parameters (row-major: weights[out_dim * in_dim]).
#[derive(Clone)]
struct DenseLayer {
    weights: Vec<f64>,
    bias: Vec<f64>,
    vel_weights: Vec<f64>,
    vel_bias: Vec<f64>,
    in_dim: usize,
    out_dim: usize,
}

impl DenseLayer {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weights: vec![0.0; out_dim * in_dim],
            bias: vec![0.0; out_dim],
            vel_weights: vec![0.0; out_dim * in_dim],
            vel_bias: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn init(&mut self, seed: u64, layer_id: u64, scale: f64) {
        for r in 0..self.out_dim {
            for c in 0..self.in_dim {
                self.weights[r * self.in_dim + c] =
                    init_param(seed, layer_id, r as u64, c as u64, scale);
            }
            self.bias[r] = 0.0;
        }
    }

    /// Forward: out = W * x + b (no activation).
    fn forward(&self, x: &[f64], out: &mut [f64]) {
        debug_assert!(x.len() >= self.in_dim);
        debug_assert!(out.len() >= self.out_dim);
        for r in 0..self.out_dim {
            let row = &self.weights[r * self.in_dim..(r + 1) * self.in_dim];
            out[r] = dot_wide(row, &x[..self.in_dim]) + self.bias[r];
        }
    }

    /// Forward with ReLU: out = max(0, W * x + b).
    fn forward_relu(&self, x: &[f64], out: &mut [f64]) {
        self.forward(x, out);
        for v in out[..self.out_dim].iter_mut() {
            *v = v.max(0.0);
        }
    }

    /// SGD update: weights -= lr * grad_out ⊗ x, bias -= lr * grad_out.
    /// Clips gradients and parameters.
    fn sgd_update(&mut self, grad_out: &[f64], x: &[f64], lr: f64, grad_clip: f64, momentum: f64) {
        if momentum == 0.0 {
            for r in 0..self.out_dim {
                let g = clip(grad_out[r], grad_clip);
                let row = &mut self.weights[r * self.in_dim..(r + 1) * self.in_dim];
                axpy_wide(row, -lr * g, &x[..self.in_dim]);
                self.bias[r] -= lr * g;
            }
            return;
        }

        for r in 0..self.out_dim {
            let g = clip(grad_out[r], grad_clip);
            for c in 0..self.in_dim {
                let idx = r * self.in_dim + c;
                let grad = g * x[c];
                self.vel_weights[idx] = momentum * self.vel_weights[idx] + grad;
                self.weights[idx] -= lr * self.vel_weights[idx];
            }
            self.vel_bias[r] = momentum * self.vel_bias[r] + g;
            self.bias[r] -= lr * self.vel_bias[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Per-cell selector + rules
// ---------------------------------------------------------------------------

/// Selector MLP for one cell: maps input → rule gate probabilities.
#[derive(Clone)]
struct CellSelector {
    hidden: DenseLayer, // in: 5*cell_dim → selector_hidden (relu)
    gate: DenseLayer,   // in: selector_hidden → num_rules (softmax)
}

/// One rule MLP for one cell: maps (input, noise) → cell delta.
#[derive(Clone)]
struct CellRule {
    hidden: DenseLayer, // in: 5*cell_dim + noise_dim → rule_hidden (relu)
    output: DenseLayer, // in: rule_hidden → cell_dim (linear)
}

/// All selector + rule params for one cell.
#[derive(Clone)]
struct CellParams {
    selector: CellSelector,
    rules: Vec<CellRule>,
}

// ---------------------------------------------------------------------------
// ParticleModel: shared parameters across the ensemble
// ---------------------------------------------------------------------------

/// The neural model parameters shared by all particles (cloned per-particle
/// for independent SGD, but initialized identically).
#[derive(Clone)]
struct ParticleModel {
    /// Byte embedding table: [256][cell_dim].
    embed: Vec<f64>,
    /// Per-cell selector + rules.
    cells: Vec<CellParams>,
    /// Readout layer: phi_dim → 256.
    readout: DenseLayer,
    /// Spec dimensions (cached).
    cell_dim: usize,
    num_cells: usize,
    noise_dim: usize,
    phi_dim: usize, // 5 * cell_dim (mean, max, stddev, second_last_emb, last_emb)
    selector_in_dim: usize, // 5 * cell_dim
}

impl ParticleModel {
    fn new(spec: &ParticleSpec) -> Self {
        let cell_dim = spec.cell_dim;
        let selector_in_dim = 5 * cell_dim;
        let rule_in_dim = 5 * cell_dim + spec.noise_dim;
        // phi = [mean_cells, max_cells, stddev_cells, ctx, last_emb]
        // The last_emb component is the direct embedding of the most recent byte.
        // This skip connection gives the readout immediate access to the nearest
        // predecessor without waiting for the slow cell warm-up.
        let phi_dim = 5 * cell_dim;

        let embed = vec![0.0; 256 * cell_dim];
        let cells = (0..spec.num_cells)
            .map(|_| CellParams {
                selector: CellSelector {
                    hidden: DenseLayer::new(selector_in_dim, spec.selector_hidden),
                    gate: DenseLayer::new(spec.selector_hidden, spec.num_rules),
                },
                rules: (0..spec.num_rules)
                    .map(|_| CellRule {
                        hidden: DenseLayer::new(rule_in_dim, spec.rule_hidden),
                        output: DenseLayer::new(spec.rule_hidden, cell_dim),
                    })
                    .collect(),
            })
            .collect();
        let readout = DenseLayer::new(phi_dim, 256);

        Self {
            embed,
            cells,
            readout,
            cell_dim,
            num_cells: spec.num_cells,
            noise_dim: spec.noise_dim,
            phi_dim,
            selector_in_dim,
        }
    }

    fn init(&mut self, seed: u64, spec: &ParticleSpec) {
        let scale = 0.1;
        // Use a larger scale for the embedding table so that the ctx and last_emb
        // components of phi carry significant signal from the first byte onward.
        // With scale=0.1 the effective cell-update magnitude is ~0.0005/byte,
        // meaning cells only become meaningful after thousands of bytes.
        // With embed_scale=0.3 the cell dynamics warm up ~3× faster.
        let embed_scale = 0.3;
        // Embedding table
        for i in 0..256 {
            for j in 0..self.cell_dim {
                self.embed[i * self.cell_dim + j] =
                    init_param(seed, 0, i as u64, j as u64, embed_scale);
            }
        }
        // Per-cell params
        for (ci, cp) in self.cells.iter_mut().enumerate() {
            let cell_seed = ci as u64 + 1;
            cp.selector.hidden.init(seed, cell_seed * 100 + 1, scale);
            cp.selector
                .gate
                .init(seed, cell_seed * 100 + 2, scale * 0.1);
            for (ri, rule) in cp.rules.iter_mut().enumerate() {
                let r_off = cell_seed * 100 + 10 + ri as u64;
                rule.hidden.init(seed, r_off * 10 + 1, scale);
                rule.output.init(seed, r_off * 10 + 2, scale * 0.5);
            }
        }
        // Readout: small init
        self.readout.init(seed, 9999, scale * 0.1);
        let _ = spec; // spec used for future extensions
    }
}

// ---------------------------------------------------------------------------
// ParticleState: per-particle latent state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ParticleState {
    /// Stable particle id for deterministic hash-noise generation.
    particle_id: u64,
    /// Latent cell values: [num_cells * cell_dim].
    cells: Vec<f64>,
    /// Context ring buffer (stores raw byte values).
    context: Vec<u8>,
    /// Write position in ring buffer.
    ctx_pos: usize,
    /// Number of bytes seen (for context length tracking).
    ctx_len: usize,
    /// Per-particle model parameters (for independent SGD).
    model: ParticleModel,
    /// Cached 256-way log-probabilities from last forward pass.
    cached_log_probs: [f64; 256],
    /// Whether cached_log_probs is valid.
    cache_valid: bool,
    // Scratch buffers (reused across forward passes to avoid allocation).
    scratch_ctx: Vec<f64>,
    scratch_mean_cells: Vec<f64>,
    scratch_p: Vec<f64>,
    scratch_sel_h: Vec<f64>,
    scratch_gate: Vec<f64>,
    scratch_rule_in: Vec<f64>,
    scratch_rule_h: Vec<f64>,
    scratch_delta_k: Vec<f64>,
    scratch_delta: Vec<f64>,
    scratch_phi: Vec<f64>,
    scratch_logits: Vec<f64>,
    // Backprop scratch buffers
    scratch_d_logits: Vec<f64>,
    scratch_d_phi: Vec<f64>,
    scratch_softmax: Vec<f64>,
    scratch_d_rule_out: Vec<f64>,
    scratch_d_rule_h: Vec<f64>,
    scratch_d_gate: Vec<f64>,
    scratch_d_gate_logits: Vec<f64>,
    scratch_d_sel_h: Vec<f64>,
    trace_history: VecDeque<StepTrace>,
}

#[derive(Clone)]
struct RuleTrace {
    rule_h: Vec<f64>,
    rule_out: Vec<f64>,
}

#[derive(Clone)]
struct CellTrace {
    p: Vec<f64>,
    sel_h: Vec<f64>,
    gate: Vec<f64>,
    rule_in: Vec<f64>,
    rules: Vec<RuleTrace>,
}

#[derive(Clone)]
struct StepTrace {
    cells: Vec<CellTrace>,
}

impl ParticleState {
    fn new(spec: &ParticleSpec, model: ParticleModel, particle_id: u64) -> Self {
        let cd = spec.cell_dim;
        let nc = spec.num_cells;
        let sel_in = 5 * cd;
        let rule_in = 5 * cd + spec.noise_dim;
        let phi_dim = model.phi_dim; // 5 * cell_dim
        Self {
            particle_id,
            cells: vec![0.0; nc * cd],
            context: vec![0; spec.context_window],
            ctx_pos: 0,
            ctx_len: 0,
            model,
            cached_log_probs: [0.0; 256],
            cache_valid: false,
            scratch_ctx: vec![0.0; cd],
            scratch_mean_cells: vec![0.0; cd],
            scratch_p: vec![0.0; sel_in],
            scratch_sel_h: vec![0.0; spec.selector_hidden],
            scratch_gate: vec![0.0; spec.num_rules],
            scratch_rule_in: vec![0.0; rule_in],
            scratch_rule_h: vec![0.0; spec.rule_hidden],
            scratch_delta_k: vec![0.0; cd],
            scratch_delta: vec![0.0; cd],
            scratch_phi: vec![0.0; phi_dim],
            scratch_logits: vec![0.0; 256],
            scratch_d_logits: vec![0.0; 256],
            scratch_d_phi: vec![0.0; phi_dim],
            scratch_softmax: vec![0.0; 256],
            scratch_d_rule_out: vec![0.0; cd],
            scratch_d_rule_h: vec![0.0; spec.rule_hidden],
            scratch_d_gate: vec![0.0; spec.num_rules],
            scratch_d_gate_logits: vec![0.0; spec.num_rules],
            scratch_d_sel_h: vec![0.0; spec.selector_hidden],
            trace_history: VecDeque::with_capacity(spec.bptt_depth.max(1)),
        }
    }

    /// Build context vector by mean-pooling byte embeddings from the ring buffer.
    fn build_ctx(&mut self) {
        let cd = self.model.cell_dim;
        self.scratch_ctx.iter_mut().for_each(|v| *v = 0.0);
        let len = self.ctx_len.min(self.context.len());
        if len == 0 {
            return;
        }
        let cw = self.context.len();
        // Exponential-decay pooling preserves order information by emphasizing
        // recent bytes while keeping fixed-size context.
        let decay = 0.90_f64;
        let mut weight_sum = 0.0_f64;
        for k in 0..len {
            let pos = (self.ctx_pos + cw - len + k) % cw;
            let byte = self.context[pos] as usize;
            let emb = &self.model.embed[byte * cd..(byte + 1) * cd];
            let age = (len - 1 - k) as i32;
            let w = decay.powi(age);
            weight_sum += w;
            for j in 0..cd {
                self.scratch_ctx[j] += emb[j] * w;
            }
        }
        if weight_sum > 0.0 {
            let inv = 1.0 / weight_sum;
            for v in &mut self.scratch_ctx {
                *v *= inv;
            }
        }
    }

    /// Compute mean of all cell vectors.
    fn compute_mean_cells(&mut self) {
        let cd = self.model.cell_dim;
        let nc = self.model.num_cells;
        self.scratch_mean_cells.iter_mut().for_each(|v| *v = 0.0);
        if nc == 0 {
            return;
        }
        let inv = 1.0 / nc as f64;
        for ci in 0..nc {
            let off = ci * cd;
            for j in 0..cd {
                self.scratch_mean_cells[j] += self.cells[off + j] * inv;
            }
        }
    }

    /// Build selector input p = concat(cell_i, left, right, ctx, mean_cells).
    fn build_selector_input(&mut self, cell_idx: usize) {
        let cd = self.model.cell_dim;
        let nc = self.model.num_cells.max(1);
        let off = cell_idx * cd;
        let left_idx = if nc <= 1 {
            cell_idx
        } else {
            (cell_idx + nc - 1) % nc
        };
        let right_idx = if nc <= 1 {
            cell_idx
        } else {
            (cell_idx + 1) % nc
        };
        let left_off = left_idx * cd;
        let right_off = right_idx * cd;
        self.scratch_p[..cd].copy_from_slice(&self.cells[off..off + cd]);
        self.scratch_p[cd..2 * cd].copy_from_slice(&self.cells[left_off..left_off + cd]);
        self.scratch_p[2 * cd..3 * cd].copy_from_slice(&self.cells[right_off..right_off + cd]);
        self.scratch_p[3 * cd..4 * cd].copy_from_slice(&self.scratch_ctx[..cd]);
        self.scratch_p[4 * cd..5 * cd].copy_from_slice(&self.scratch_mean_cells[..cd]);
    }

    /// Build rule input = concat(p, z) with deterministic hash-noise annealing.
    fn build_rule_input(
        &mut self,
        spec: &ParticleSpec,
        step_idx: u64,
        unroll_idx: usize,
        cell_idx: usize,
    ) {
        let sel_in = self.model.selector_in_dim;
        let nd = self.model.noise_dim;
        self.scratch_rule_in[..sel_in].copy_from_slice(&self.scratch_p[..sel_in]);
        if nd == 0 || !spec.enable_noise || spec.noise_scale <= 0.0 {
            for j in sel_in..sel_in + nd {
                self.scratch_rule_in[j] = 0.0;
            }
            return;
        }
        let anneal = if spec.noise_anneal_steps == 0 {
            1.0
        } else {
            let rem = spec.noise_anneal_steps.saturating_sub(step_idx as usize) as f64;
            rem / spec.noise_anneal_steps as f64
        };
        let scale = spec.noise_scale * anneal.max(0.0);
        for j in 0..nd {
            let h = det_hash(
                spec.seed ^ self.particle_id,
                step_idx,
                ((unroll_idx as u64) << 40) ^ ((cell_idx as u64) << 20) ^ j as u64,
                0xD1A6_51EED,
            );
            self.scratch_rule_in[sel_in + j] = hash_to_f64(h) * scale;
        }
    }

    /// Build featurize vector phi = concat(mean_cells, max_cells, stddev_cells, second_last_emb, last_emb).
    ///
    /// Components:
    ///   [0..cd]     mean_cells      — long-range context via latent cells
    ///   [cd..2cd]   max_cells       — long-range context peak
    ///   [2cd..3cd]  stddev_cells    — latent uncertainty
    ///   [3cd..4cd]  second_last_emb — direct embedding of x_{t-2} (skip connection)
    ///   [4cd..5cd]  last_emb        — direct embedding of x_{t-1} (skip connection)
    ///
    /// The two skip connections give the readout immediate access to the exact
    /// preceding two bytes, enabling bigram/trigram statistics to be learned
    /// from the very first observation.  We no longer include the blurred
    /// exponential-decay context average as a phi component; the selector/rule
    /// cell dynamics still use it (via scratch_ctx), but for prediction it is
    /// dominated by the direct embeddings.
    fn build_phi(&mut self) {
        let cd = self.model.cell_dim;
        let nc = self.model.num_cells;
        // mean_cells already in scratch_mean_cells
        self.scratch_phi[..cd].copy_from_slice(&self.scratch_mean_cells[..cd]);
        // max_cells
        for j in 0..cd {
            let mut mx = f64::NEG_INFINITY;
            for ci in 0..nc {
                let v = self.cells[ci * cd + j];
                if v > mx {
                    mx = v;
                }
            }
            self.scratch_phi[cd + j] = if mx.is_finite() { mx } else { 0.0 };
        }
        // stddev across cells (captures latent uncertainty)
        for j in 0..cd {
            let mean = self.scratch_mean_cells[j];
            let mut var = 0.0_f64;
            for ci in 0..nc {
                let d = self.cells[ci * cd + j] - mean;
                var += d * d;
            }
            self.scratch_phi[2 * cd + j] = (var / nc.max(1) as f64).sqrt();
        }
        // second_last_emb: direct embedding of x_{t-2}.
        let cw = self.context.len();
        if self.ctx_len >= 2 {
            // ctx_pos points to the NEXT write slot; walk back 2 bytes.
            let pos2 = (self.ctx_pos + cw - 2) % cw;
            let byte2 = self.context[pos2] as usize;
            self.scratch_phi[3 * cd..4 * cd]
                .copy_from_slice(&self.model.embed[byte2 * cd..(byte2 + 1) * cd]);
        } else {
            self.scratch_phi[3 * cd..4 * cd].fill(0.0);
        }
        // last_emb: direct embedding of x_{t-1}.
        if self.ctx_len >= 1 {
            let pos1 = (self.ctx_pos + cw - 1) % cw;
            let byte1 = self.context[pos1] as usize;
            self.scratch_phi[4 * cd..5 * cd]
                .copy_from_slice(&self.model.embed[byte1 * cd..(byte1 + 1) * cd]);
        } else {
            self.scratch_phi[4 * cd..5 * cd].fill(0.0);
        }
    }

    /// Full forward pass: update latent cells, compute log-probabilities.
    fn forward(&mut self, spec: &ParticleSpec, step_idx: u64) {
        self.build_ctx();
        self.compute_mean_cells();
        let capture_trace = spec.learning_rate_selector > 0.0 || spec.learning_rate_rule > 0.0;
        let mut step_trace = if capture_trace {
            Some(StepTrace {
                cells: Vec::with_capacity(self.model.num_cells),
            })
        } else {
            None
        };

        // Latent update (unroll_steps iterations)
        for unroll_idx in 0..spec.unroll_steps {
            for ci in 0..self.model.num_cells {
                self.build_selector_input(ci);

                // Selector: hidden = relu(W_sel * p + b_sel)
                self.model.cells[ci]
                    .selector
                    .hidden
                    .forward_relu(&self.scratch_p, &mut self.scratch_sel_h);
                // Gate: gate_logits = V_sel * h + c_sel, then softmax
                self.model.cells[ci]
                    .selector
                    .gate
                    .forward(&self.scratch_sel_h, &mut self.scratch_gate);
                softmax_inplace(&mut self.scratch_gate[..spec.num_rules]);

                // Build rule input
                self.build_rule_input(spec, step_idx, unroll_idx, ci);

                // Compute weighted delta
                let cd = self.model.cell_dim;
                self.scratch_delta[..cd].fill(0.0);
                let mut rule_traces = if capture_trace {
                    Some(Vec::with_capacity(spec.num_rules))
                } else {
                    None
                };
                for ki in 0..spec.num_rules {
                    let gate_k = self.scratch_gate[ki];
                    // Rule hidden
                    self.model.cells[ci].rules[ki]
                        .hidden
                        .forward_relu(&self.scratch_rule_in, &mut self.scratch_rule_h);
                    // Rule output
                    self.model.cells[ci].rules[ki]
                        .output
                        .forward(&self.scratch_rule_h, &mut self.scratch_delta_k);
                    if let Some(rt) = &mut rule_traces {
                        rt.push(RuleTrace {
                            rule_h: self.scratch_rule_h.clone(),
                            rule_out: self.scratch_delta_k[..cd].to_vec(),
                        });
                    }
                    for j in 0..cd {
                        self.scratch_delta[j] += gate_k * self.scratch_delta_k[j];
                    }
                }
                if let (Some(st), Some(rt)) = (&mut step_trace, rule_traces) {
                    st.cells.push(CellTrace {
                        p: self.scratch_p.clone(),
                        sel_h: self.scratch_sel_h.clone(),
                        gate: self.scratch_gate[..spec.num_rules].to_vec(),
                        rule_in: self.scratch_rule_in.clone(),
                        rules: rt,
                    });
                }

                // Update cell
                let off = ci * cd;
                for j in 0..cd {
                    self.cells[off + j] =
                        clip(self.cells[off + j] + self.scratch_delta[j], spec.state_clip);
                }
            }
            // Recompute mean_cells after each unroll step
            self.compute_mean_cells();
        }

        // Featurize
        self.build_phi();

        // Readout
        self.model
            .readout
            .forward(&self.scratch_phi, &mut self.scratch_logits);

        // Reuse exact softmax(logits) in SGD to avoid recomputing it.
        self.scratch_softmax.copy_from_slice(&self.scratch_logits);
        softmax_inplace(&mut self.scratch_softmax);

        // Log-softmax with floor
        log_softmax_with_floor(
            &self.scratch_logits,
            &mut self.cached_log_probs,
            spec.min_prob,
        );
        if let Some(st) = step_trace {
            self.trace_history.push_back(st);
            while self.trace_history.len() > spec.bptt_depth.max(1) {
                self.trace_history.pop_front();
            }
        }
        self.cache_valid = true;
    }

    fn apply_selector_rule_update_from_trace(
        &mut self,
        trace: &StepTrace,
        d_phi: &[f64],
        temporal: f64,
        spec: &ParticleSpec,
    ) {
        let cd = self.model.cell_dim;
        let nc = self.model.num_cells.max(1);
        let d_delta_scale = (1.0 / nc as f64) * temporal;

        for ci in 0..nc.min(trace.cells.len()) {
            let ct = &trace.cells[ci];

            // d_gate[k] = dot(d_delta, rule_out_k)
            self.scratch_d_gate[..spec.num_rules].fill(0.0);

            for ki in 0..spec.num_rules.min(ct.rules.len()) {
                let gate_k = ct.gate[ki];
                self.scratch_d_rule_out[..cd].fill(0.0);
                for j in 0..cd {
                    self.scratch_d_rule_out[j] = d_phi[j] * d_delta_scale * gate_k;
                }

                // output layer update
                self.model.cells[ci].rules[ki].output.sgd_update(
                    &self.scratch_d_rule_out,
                    &ct.rules[ki].rule_h,
                    spec.learning_rate_rule,
                    spec.grad_clip,
                    spec.optimizer_momentum,
                );

                // hidden layer update
                let rh = spec.rule_hidden;
                self.scratch_d_rule_h[..rh].fill(0.0);
                for r in 0..cd {
                    let g = clip(self.scratch_d_rule_out[r], spec.grad_clip);
                    if g.abs() < 1e-15 {
                        continue;
                    }
                    for c in 0..rh {
                        self.scratch_d_rule_h[c] +=
                            g * self.model.cells[ci].rules[ki].output.weights[r * rh + c];
                    }
                }
                for (j, h) in ct.rules[ki].rule_h.iter().enumerate().take(rh) {
                    if *h <= 0.0 {
                        self.scratch_d_rule_h[j] = 0.0;
                    }
                }
                self.model.cells[ci].rules[ki].hidden.sgd_update(
                    &self.scratch_d_rule_h[..rh],
                    &ct.rule_in,
                    spec.learning_rate_rule,
                    spec.grad_clip,
                    spec.optimizer_momentum,
                );

                for j in 0..cd {
                    self.scratch_d_gate[ki] += d_phi[j] * d_delta_scale * ct.rules[ki].rule_out[j];
                }
            }

            let dot_gd: f64 = (0..spec.num_rules.min(ct.gate.len()))
                .map(|k| ct.gate[k] * self.scratch_d_gate[k])
                .sum();
            self.scratch_d_gate_logits[..spec.num_rules].fill(0.0);
            for k in 0..spec.num_rules.min(ct.gate.len()) {
                self.scratch_d_gate_logits[k] = ct.gate[k] * (self.scratch_d_gate[k] - dot_gd);
            }

            self.model.cells[ci].selector.gate.sgd_update(
                &self.scratch_d_gate_logits[..spec.num_rules],
                &ct.sel_h,
                spec.learning_rate_selector,
                spec.grad_clip,
                spec.optimizer_momentum,
            );

            let sh = spec.selector_hidden;
            self.scratch_d_sel_h[..sh].fill(0.0);
            for r in 0..spec.num_rules.min(ct.gate.len()) {
                let g = clip(self.scratch_d_gate_logits[r], spec.grad_clip);
                if g.abs() < 1e-15 {
                    continue;
                }
                for c in 0..sh {
                    self.scratch_d_sel_h[c] +=
                        g * self.model.cells[ci].selector.gate.weights[r * sh + c];
                }
            }
            for (j, h) in ct.sel_h.iter().enumerate().take(sh) {
                if *h <= 0.0 {
                    self.scratch_d_sel_h[j] = 0.0;
                }
            }
            self.model.cells[ci].selector.hidden.sgd_update(
                &self.scratch_d_sel_h[..sh],
                &ct.p,
                spec.learning_rate_selector,
                spec.grad_clip,
                spec.optimizer_momentum,
            );
        }
    }

    /// Online SGD update after observing byte `y`.
    fn sgd_update(&mut self, y: u8, spec: &ParticleSpec) {
        // Readout gradient: d_logits = softmax - onehot(y)
        self.scratch_d_logits.copy_from_slice(&self.scratch_softmax);
        self.scratch_d_logits[y as usize] -= 1.0;

        // Clip readout gradients
        for v in self.scratch_d_logits.iter_mut() {
            *v = clip(*v, spec.grad_clip);
        }

        // Update readout: W -= lr * d_logits ⊗ phi, b -= lr * d_logits
        self.model.readout.sgd_update(
            &self.scratch_d_logits,
            &self.scratch_phi,
            spec.learning_rate_readout,
            spec.grad_clip,
            0.0,
        );

        // Backprop to phi: d_phi = readout.W^T * d_logits
        let phi_dim = self.model.phi_dim;
        self.scratch_d_phi[..phi_dim].fill(0.0);
        for r in 0..256 {
            let g = clip(self.scratch_d_logits[r], spec.grad_clip);
            if g.abs() < 1e-15 {
                continue;
            }
            let row_start = r * phi_dim;
            for c in 0..phi_dim {
                self.scratch_d_phi[c] += g * self.model.readout.weights[row_start + c];
            }
        }

        // Clip d_phi
        for v in self.scratch_d_phi[..phi_dim].iter_mut() {
            *v = clip(*v, spec.grad_clip);
        }

        if spec.learning_rate_selector > 0.0 || spec.learning_rate_rule > 0.0 {
            let depth = spec.bptt_depth.max(1).min(self.trace_history.len());
            let traces = std::mem::take(&mut self.trace_history);
            let d_phi = self.scratch_d_phi[..phi_dim].to_vec();
            let mut temporal = 1.0_f64;
            let temporal_decay = 0.7_f64;
            for idx in 0..depth {
                let hist_idx = traces.len() - 1 - idx;
                let trace = &traces[hist_idx];
                self.apply_selector_rule_update_from_trace(trace, &d_phi, temporal, spec);
                temporal *= temporal_decay;
            }
            self.trace_history = traces;
        }
    }

    /// Push a byte into the context ring buffer.
    fn push_context(&mut self, byte: u8) {
        self.context[self.ctx_pos] = byte;
        self.ctx_pos = (self.ctx_pos + 1) % self.context.len();
        self.ctx_len += 1;
    }
}

// ---------------------------------------------------------------------------
// ParticleRuntime: the ensemble predictor (public API)
// ---------------------------------------------------------------------------

/// Runtime for the particle-latent filter ensemble.
///
/// Implements [`crate::mixture::OnlineBytePredictor`] and provides
/// `pdf_next()` for compression compatibility.
pub struct ParticleRuntime {
    spec: ParticleSpec,
    particles: Vec<ParticleState>,
    log_weights: Vec<f64>,
    /// Cached mixture log-probabilities [256].
    mix_log_probs: [f64; 256],
    /// Cached mixture PDF [256].
    mix_pdf: Vec<f64>,
    /// Whether mix_log_probs is valid.
    cache_valid: bool,
    /// Step counter for deterministic hash.
    step_idx: u64,
    /// Scratch for logsumexp across particles.
    scratch_lse: Vec<f64>,
}

impl ParticleRuntime {
    #[inline]
    fn likelihood_beta(&self) -> f64 {
        // Early temperature prevents particle-weight collapse before experts
        // have adapted. Anneal to full Bayes update.
        const BETA_MIN: f64 = 0.35;
        const WARMUP_STEPS: u64 = 2048;
        if self.step_idx >= WARMUP_STEPS {
            1.0
        } else {
            BETA_MIN + (1.0 - BETA_MIN) * (self.step_idx as f64 / WARMUP_STEPS as f64)
        }
    }

    #[inline]
    fn diagnostics_enabled(&self) -> bool {
        self.spec.diagnostics_interval > 0
            && self.step_idx % self.spec.diagnostics_interval as u64 == 0
    }

    #[inline]
    fn weight_stats(&self) -> (f64, f64) {
        let mut sum_sq = 0.0;
        let mut max_w = 0.0;
        for &lw in &self.log_weights {
            let w = lw.exp();
            sum_sq += w * w;
            if w > max_w {
                max_w = w;
            }
        }
        let n_eff = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
        (n_eff, max_w)
    }

    fn weighted_prediction_kl_divergence(&self) -> f64 {
        // D = Σ_i α_i KL(p_i || p_mix), where α_i = softmax(log_weights).
        // This is the ensemble disagreement signal: zero means collapse.
        let n = self.particles.len();
        if n == 0 {
            return 0.0;
        }
        let log_z = logsumexp_wide(&self.log_weights);
        let mut mix_log_probs = [0.0_f64; 256];
        let mut scratch_lse = vec![0.0_f64; n];
        for v in 0..256 {
            for i in 0..n {
                scratch_lse[i] = self.log_weights[i] + self.particles[i].cached_log_probs[v];
            }
            mix_log_probs[v] = logsumexp_wide(&scratch_lse) - log_z;
        }
        let mut d = 0.0_f64;
        for (i, p) in self.particles.iter().enumerate() {
            let alpha = self.log_weights[i].exp();
            if alpha <= 0.0 {
                continue;
            }
            let mut kl_i = 0.0_f64;
            for v in 0..256 {
                let lp_i = p.cached_log_probs[v];
                let prob_i = lp_i.exp();
                kl_i += prob_i * (lp_i - mix_log_probs[v]);
            }
            d += alpha * kl_i.max(0.0);
        }
        d
    }

    fn log_diagnostics(
        &self,
        n_eff: f64,
        max_weight: f64,
        divergence: f64,
        beta: f64,
        will_resample: bool,
    ) {
        eprintln!(
            "[particle] step={} neff={:.3}/{:.0} max_w={:.3}% div_kl={:.6} beta={:.3} resample={}",
            self.step_idx,
            n_eff,
            self.particles.len() as f64,
            max_weight * 100.0,
            divergence,
            beta,
            will_resample
        );
    }

    fn diversify_initial_particles(&mut self) {
        let scale = 5e-3_f64;
        if self.particles.len() <= 1 {
            return;
        }
        for pi in 1..self.particles.len() {
            let p = &mut self.particles[pi];
            for (idx, v) in p.cells.iter_mut().enumerate() {
                let noise = hash_to_f64(det_hash(self.spec.seed, pi as u64, idx as u64, 1000));
                *v += noise * scale;
            }
            for (idx, v) in p.model.readout.bias.iter_mut().enumerate() {
                let noise = hash_to_f64(det_hash(self.spec.seed, pi as u64, idx as u64, 1001));
                *v += noise * scale;
            }
            for (idx, v) in p.model.readout.weights.iter_mut().enumerate() {
                let noise = hash_to_f64(det_hash(self.spec.seed, pi as u64, idx as u64, 1002));
                *v += noise * (scale * 0.5);
            }
        }
    }

    /// Create a new particle runtime from a spec.
    pub fn new(spec: &ParticleSpec) -> Self {
        let n = spec.num_particles;

        // Each particle is independently initialized with its own seed derived
        // from the master seed via a multiplicative hash spread.  This ensures
        // genuine parameter diversity: every embedding table, selector MLP, rule
        // MLP, and readout starts from a distinct random point in weight space.
        //
        // Without this, all particles are near-identical clones (only differing
        // by the tiny 5e-3 perturbation in diversify_initial_particles), and with
        // identical selector/rule MLPs they evolve identically regardless of
        // learning rate.  The ensemble then collapses to a single-particle model,
        // explaining why changing num_particles has no effect on compression.
        //
        // Using φ-like multiplicative spread (0x9e3779b9 ≈ 2^32/φ) ensures that
        // seeds for different particle indices are maximally well-separated in the
        // 64-bit hash space.
        let particles: Vec<ParticleState> = (0..n)
            .map(|pi| {
                let particle_seed = spec
                    .seed
                    .wrapping_add((pi as u64).wrapping_mul(0x9e3779b97f4a7c15u64));
                let mut model = ParticleModel::new(spec);
                model.init(particle_seed, spec);
                ParticleState::new(spec, model, pi as u64)
            })
            .collect();

        let log_w = -(n as f64).ln();
        let mut rt = Self {
            spec: spec.clone(),
            particles,
            log_weights: vec![log_w; n],
            mix_log_probs: [0.0; 256],
            mix_pdf: vec![0.0; 256],
            cache_valid: false,
            step_idx: 0,
            scratch_lse: vec![0.0; n],
        };
        rt.diversify_initial_particles();
        rt
    }

    /// Ensure all particles have valid cached log-probabilities.
    fn ensure_predictions(&mut self) {
        if self.cache_valid {
            return;
        }
        let spec = &self.spec;
        for p in &mut self.particles {
            if !p.cache_valid {
                p.forward(spec, self.step_idx);
            }
        }
        self.compute_mixture_log_probs();
        self.cache_valid = true;
    }

    /// Compute mixture log-probabilities by weighting particle predictions.
    fn compute_mixture_log_probs(&mut self) {
        let n = self.particles.len();
        // log_z = logsumexp(log_weights)
        let log_z = logsumexp_wide(&self.log_weights);

        for v in 0..256 {
            for i in 0..n {
                self.scratch_lse[i] = self.log_weights[i] + self.particles[i].cached_log_probs[v];
            }
            self.mix_log_probs[v] = logsumexp_wide(&self.scratch_lse) - log_z;
        }

        // Also compute PDF for compression
        let max_lp = max_wide(&self.mix_log_probs);
        let mut sum = 0.0;
        for v in 0..256 {
            let p = (self.mix_log_probs[v] - max_lp).exp();
            self.mix_pdf[v] = p;
            sum += p;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in &mut self.mix_pdf {
                *v *= inv;
            }
        }
    }

    /// Non-mutating log-probability query for a single symbol.
    pub fn peek_log_prob(&mut self, symbol: u8) -> f64 {
        self.ensure_predictions();
        self.mix_log_probs[symbol as usize]
    }

    /// Fill 256-way log-probabilities (non-mutating).
    pub fn fill_log_probs_cached(&mut self, out: &mut [f64; 256]) {
        self.ensure_predictions();
        *out = self.mix_log_probs;
    }

    /// Return 256-element PDF slice for compression.
    pub fn pdf_next(&mut self) -> &[f64] {
        self.ensure_predictions();
        &self.mix_pdf
    }

    /// Observe byte `y`: return ln(p(y)) then update ensemble state.
    pub fn step(&mut self, symbol: u8) -> f64 {
        self.ensure_predictions();
        let log_prob = self.mix_log_probs[symbol as usize];

        let n = self.particles.len();
        let spec = &self.spec;

        // (1) Weight update: logw_i += logq_i[y]
        let beta = self.likelihood_beta();
        for i in 0..n {
            self.log_weights[i] += beta * self.particles[i].cached_log_probs[symbol as usize];
        }
        // Normalize log-weights
        let log_z = logsumexp_wide(&self.log_weights);
        for w in &mut self.log_weights {
            *w -= log_z;
        }

        // (2) Forgetting
        if spec.forget_lambda > 0.0 {
            let uniform = -(n as f64).ln();
            for w in &mut self.log_weights {
                *w = (1.0 - spec.forget_lambda) * *w + spec.forget_lambda * uniform;
            }
            // Renormalize
            let log_z2 = logsumexp_wide(&self.log_weights);
            for w in &mut self.log_weights {
                *w -= log_z2;
            }
        }

        let (n_eff_before, max_w_before) = self.weight_stats();
        let will_resample = n_eff_before < self.spec.resample_threshold * n as f64;
        let should_log = self.diagnostics_enabled();
        let divergence = if should_log {
            self.weighted_prediction_kl_divergence()
        } else {
            0.0
        };

        // (3) Online SGD per particle
        for p in &mut self.particles {
            p.sgd_update(symbol, spec);
        }

        // (4) Push context byte
        for p in &mut self.particles {
            p.push_context(symbol);
        }

        if should_log {
            self.log_diagnostics(n_eff_before, max_w_before, divergence, beta, will_resample);
        }

        // (5) Resample check
        let _ = self.maybe_resample();

        // Invalidate caches
        for p in &mut self.particles {
            p.cache_valid = false;
        }
        self.cache_valid = false;
        self.step_idx += 1;

        log_prob
    }

    /// Check effective sample size and resample if needed.
    fn maybe_resample(&mut self) -> bool {
        let n = self.particles.len();
        if n <= 1 {
            return false;
        }

        // Compute Neff = 1 / Σ α_i^2 where α = softmax(logw)
        // = exp(-logsumexp(2*logw)) / exp(2*(-logsumexp(logw)))
        // But logw is already normalized, so logsumexp(logw) ≈ 0
        let mut sum_sq = 0.0;
        for &lw in &self.log_weights {
            let w = lw.exp();
            sum_sq += w * w;
        }
        let n_eff = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

        if n_eff >= self.spec.resample_threshold * n as f64 {
            return false;
        }

        // Deterministic systematic resampling with fixed offset 0.5/n
        let weights: Vec<f64> = self.log_weights.iter().map(|lw| lw.exp()).collect();
        let cdf: Vec<f64> = weights
            .iter()
            .scan(0.0, |acc, &w| {
                *acc += w;
                Some(*acc)
            })
            .collect();
        let total = *cdf.last().unwrap_or(&1.0);

        let step = total / n as f64;
        // Deterministic stratified offset from hash to avoid repeating the same
        // systematic pattern at every resample event.
        let u0 =
            ((det_hash(self.spec.seed, self.step_idx, 0, 0) >> 11) as f64) / ((1u64 << 53) as f64);
        let mut u = u0 * step;
        let mut indices = Vec::with_capacity(n);
        let mut j = 0;
        for _ in 0..n {
            while j < n - 1 && cdf[j] < u {
                j += 1;
            }
            indices.push(j);
            u += step;
        }

        // Clone selected particles
        let new_particles: Vec<ParticleState> = indices
            .iter()
            .map(|&idx| self.particles[idx].clone())
            .collect();
        self.particles = new_particles;

        // Mutate a fraction of particles
        let n_mutate = ((self.spec.mutate_fraction * n as f64).round() as usize).min(n);
        let mut mutated = vec![false; n];
        let mut picked = 0usize;
        let mut draw = 0u64;
        while picked < n_mutate && draw < (n * 8) as u64 {
            let mi = (det_hash(self.spec.seed ^ self.step_idx, draw, 0xA5A5, 0x5A5A) as usize) % n;
            if !mutated[mi] {
                self.mutate_particle(mi);
                mutated[mi] = true;
                picked += 1;
            }
            draw += 1;
        }

        // Reset log-weights to uniform
        let uniform = -(n as f64).ln();
        for w in &mut self.log_weights {
            *w = uniform;
        }
        true
    }

    /// Apply deterministic mutation: always perturb latent state; optionally
    /// perturb model parameters when explicitly enabled.
    fn mutate_particle(&mut self, particle_idx: usize) {
        let seed = self.spec.seed;
        let step = self.step_idx;
        let pi = particle_idx as u64;
        let scale = self.spec.mutate_scale;
        let state_clip = self.spec.state_clip;

        let p = &mut self.particles[particle_idx];
        let mut param_idx = 0u64;

        // Mutate cell states
        for v in p.cells.iter_mut() {
            let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 0)) * scale;
            *v = clip(*v + noise, state_clip);
            param_idx += 1;
        }

        if !self.spec.mutate_model_params {
            return;
        }

        let layer_scale = |vals: &[f64]| -> f64 {
            if vals.is_empty() {
                return 1.0;
            }
            let mut s = 0.0_f64;
            for &v in vals {
                s += v * v;
            }
            (s / vals.len() as f64).sqrt().max(1e-6)
        };

        // Mutate model parameters with layer-adaptive scale to avoid destructive jumps.
        let embed_layer = layer_scale(&p.model.embed);
        for v in p.model.embed.iter_mut() {
            let noise =
                hash_to_f64(det_hash(seed ^ step, pi, param_idx, 1)) * (scale * embed_layer);
            *v += noise;
            param_idx += 1;
        }

        // Cell params
        for cp in p.model.cells.iter_mut() {
            let sel_h_w = layer_scale(&cp.selector.hidden.weights);
            let sel_h_b = layer_scale(&cp.selector.hidden.bias);
            for v in cp.selector.hidden.weights.iter_mut() {
                let noise =
                    hash_to_f64(det_hash(seed ^ step, pi, param_idx, 2)) * (scale * sel_h_w);
                *v += noise;
                param_idx += 1;
            }
            for v in cp.selector.hidden.bias.iter_mut() {
                let noise =
                    hash_to_f64(det_hash(seed ^ step, pi, param_idx, 3)) * (scale * sel_h_b);
                *v += noise;
                param_idx += 1;
            }
            let sel_g_w = layer_scale(&cp.selector.gate.weights);
            let sel_g_b = layer_scale(&cp.selector.gate.bias);
            for v in cp.selector.gate.weights.iter_mut() {
                let noise =
                    hash_to_f64(det_hash(seed ^ step, pi, param_idx, 4)) * (scale * sel_g_w);
                *v += noise;
                param_idx += 1;
            }
            for v in cp.selector.gate.bias.iter_mut() {
                let noise =
                    hash_to_f64(det_hash(seed ^ step, pi, param_idx, 5)) * (scale * sel_g_b);
                *v += noise;
                param_idx += 1;
            }
            for rule in cp.rules.iter_mut() {
                let rule_h_w = layer_scale(&rule.hidden.weights);
                let rule_h_b = layer_scale(&rule.hidden.bias);
                let rule_o_w = layer_scale(&rule.output.weights);
                let rule_o_b = layer_scale(&rule.output.bias);
                for v in rule.hidden.weights.iter_mut() {
                    let noise =
                        hash_to_f64(det_hash(seed ^ step, pi, param_idx, 6)) * (scale * rule_h_w);
                    *v += noise;
                    param_idx += 1;
                }
                for v in rule.hidden.bias.iter_mut() {
                    let noise =
                        hash_to_f64(det_hash(seed ^ step, pi, param_idx, 7)) * (scale * rule_h_b);
                    *v += noise;
                    param_idx += 1;
                }
                for v in rule.output.weights.iter_mut() {
                    let noise =
                        hash_to_f64(det_hash(seed ^ step, pi, param_idx, 8)) * (scale * rule_o_w);
                    *v += noise;
                    param_idx += 1;
                }
                for v in rule.output.bias.iter_mut() {
                    let noise =
                        hash_to_f64(det_hash(seed ^ step, pi, param_idx, 9)) * (scale * rule_o_b);
                    *v += noise;
                    param_idx += 1;
                }
            }
        }

        // Readout
        let readout_w = layer_scale(&p.model.readout.weights);
        let readout_b = layer_scale(&p.model.readout.bias);
        for v in p.model.readout.weights.iter_mut() {
            let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 10)) * (scale * readout_w);
            *v += noise;
            param_idx += 1;
        }
        for v in p.model.readout.bias.iter_mut() {
            let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 11)) * (scale * readout_b);
            *v += noise;
            param_idx += 1;
        }
    }
}

impl Clone for ParticleRuntime {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec.clone(),
            particles: self.particles.clone(),
            log_weights: self.log_weights.clone(),
            mix_log_probs: self.mix_log_probs,
            mix_pdf: self.mix_pdf.clone(),
            cache_valid: self.cache_valid,
            step_idx: self.step_idx,
            scratch_lse: self.scratch_lse.clone(),
        }
    }
}

// Implement OnlineBytePredictor
impl crate::mixture::OnlineBytePredictor for ParticleRuntime {
    fn log_prob(&mut self, symbol: u8) -> f64 {
        self.peek_log_prob(symbol)
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        self.fill_log_probs_cached(out)
    }

    fn update(&mut self, symbol: u8) {
        self.step(symbol);
    }
}

// Safety: ParticleRuntime is Send because all internal state is owned Vec/f64.
unsafe impl Send for ParticleRuntime {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_spec() -> ParticleSpec {
        ParticleSpec {
            num_particles: 4,
            context_window: 8,
            unroll_steps: 1,
            num_cells: 2,
            cell_dim: 4,
            num_rules: 2,
            selector_hidden: 8,
            rule_hidden: 8,
            noise_dim: 2,
            ..ParticleSpec::default()
        }
    }

    #[test]
    fn pdf_sums_to_one() {
        let spec = default_spec();
        let mut rt = ParticleRuntime::new(&spec);
        let pdf = rt.pdf_next();
        let sum: f64 = pdf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "PDF sum = {sum}, expected ~1.0");
    }

    #[test]
    fn log_probs_finite_and_nonpositive() {
        let spec = default_spec();
        let mut rt = ParticleRuntime::new(&spec);
        let data = b"hello world";
        for &b in data.iter() {
            let lp = rt.peek_log_prob(b);
            assert!(lp.is_finite(), "log_prob not finite: {lp}");
            assert!(lp <= 0.0, "log_prob positive: {lp}");
            rt.step(b);
        }
    }

    #[test]
    fn deterministic_same_seed() {
        let spec = default_spec();
        let data = b"abcdefghij";

        let mut rt1 = ParticleRuntime::new(&spec);
        let mut rt2 = ParticleRuntime::new(&spec);

        for &b in data.iter() {
            let lp1 = rt1.step(b);
            let lp2 = rt2.step(b);
            assert!(
                (lp1 - lp2).abs() < 1e-12,
                "Mismatch at byte {b}: {lp1} vs {lp2}"
            );
        }
    }

    #[test]
    fn deterministic_with_hash_noise_enabled() {
        let spec = ParticleSpec {
            enable_noise: true,
            noise_scale: 0.15,
            noise_anneal_steps: 128,
            ..default_spec()
        };
        let data = b"particle noise determinism";

        let mut rt1 = ParticleRuntime::new(&spec);
        let mut rt2 = ParticleRuntime::new(&spec);

        for &b in data {
            let lp1 = rt1.step(b);
            let lp2 = rt2.step(b);
            assert!(
                (lp1 - lp2).abs() < 1e-12,
                "Hash-noise path non-deterministic at byte {b}: {lp1} vs {lp2}"
            );
        }
    }

    #[test]
    fn resample_forced() {
        let spec = ParticleSpec {
            resample_threshold: 1.0, // always resample
            ..default_spec()
        };
        let mut rt = ParticleRuntime::new(&spec);
        // Should not panic even with constant resampling
        for &b in b"test resampling works ok" {
            let lp = rt.step(b);
            assert!(lp.is_finite(), "log_prob not finite after resample: {lp}");
        }
    }

    #[test]
    fn mutation_determinism() {
        let spec = ParticleSpec {
            resample_threshold: 1.0,
            mutate_fraction: 1.0,
            ..default_spec()
        };
        let data = b"test mutation";

        let mut rt1 = ParticleRuntime::new(&spec);
        let mut rt2 = ParticleRuntime::new(&spec);

        for &b in data.iter() {
            let lp1 = rt1.step(b);
            let lp2 = rt2.step(b);
            assert!(
                (lp1 - lp2).abs() < 1e-12,
                "Mutation non-deterministic at byte {b}: {lp1} vs {lp2}"
            );
        }
    }

    #[test]
    fn empty_input_no_crash() {
        let spec = default_spec();
        let mut rt = ParticleRuntime::new(&spec);
        // Just verify we can get predictions without any input
        let lp = rt.peek_log_prob(0);
        assert!(lp.is_finite());
    }

    #[test]
    fn fill_log_probs_consistency() {
        let spec = default_spec();
        let mut rt = ParticleRuntime::new(&spec);
        rt.step(b'a');
        rt.step(b'b');

        let mut bulk = [0.0; 256];
        rt.fill_log_probs_cached(&mut bulk);

        for sym in 0..256u16 {
            let single = rt.peek_log_prob(sym as u8);
            assert!(
                (bulk[sym as usize] - single).abs() < 1e-12,
                "Mismatch for sym {sym}: bulk={} single={}",
                bulk[sym as usize],
                single
            );
        }
    }

    #[test]
    fn spec_validation() {
        let mut spec = ParticleSpec::default();
        assert!(spec.validate().is_ok());

        spec.num_particles = 0;
        assert!(spec.validate().is_err());
        spec.num_particles = 4;

        spec.resample_threshold = 0.0;
        assert!(spec.validate().is_err());
        spec.resample_threshold = 0.5;

        spec.min_prob = -1.0;
        assert!(spec.validate().is_err());
    }
}
