//! Particle-latent filter ensemble rate backend.
//!
//! Implements a deterministic-by-default sequential particle model with latent
//! cells, selector/rule dynamics, online SGD, Bayesian particle weighting,
//! and resample+mutation.

use crate::simd_math::{dot_wide, logsumexp_wide, max_wide};
use crate::ParticleSpec;

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
    in_dim: usize,
    out_dim: usize,
}

impl DenseLayer {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weights: vec![0.0; out_dim * in_dim],
            bias: vec![0.0; out_dim],
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
    fn sgd_update(
        &mut self,
        grad_out: &[f64],
        x: &[f64],
        lr: f64,
        grad_clip: f64,
    ) {
        for r in 0..self.out_dim {
            let g = clip(grad_out[r], grad_clip);
            for c in 0..self.in_dim {
                self.weights[r * self.in_dim + c] -= lr * g * x[c];
            }
            self.bias[r] -= lr * g;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-cell selector + rules
// ---------------------------------------------------------------------------

/// Selector MLP for one cell: maps input → rule gate probabilities.
#[derive(Clone)]
struct CellSelector {
    hidden: DenseLayer,   // in: 3*cell_dim → selector_hidden (relu)
    gate: DenseLayer,     // in: selector_hidden → num_rules (softmax)
}

/// One rule MLP for one cell: maps (input, noise) → cell delta.
#[derive(Clone)]
struct CellRule {
    hidden: DenseLayer,   // in: 3*cell_dim + noise_dim → rule_hidden (relu)
    output: DenseLayer,   // in: rule_hidden → cell_dim (linear)
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
    phi_dim: usize, // 3 * cell_dim
    selector_in_dim: usize, // 3 * cell_dim
}

impl ParticleModel {
    fn new(spec: &ParticleSpec) -> Self {
        let cell_dim = spec.cell_dim;
        let selector_in_dim = 3 * cell_dim;
        let rule_in_dim = 3 * cell_dim + spec.noise_dim;
        let phi_dim = 3 * cell_dim;

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
        // Embedding table
        for i in 0..256 {
            for j in 0..self.cell_dim {
                self.embed[i * self.cell_dim + j] =
                    init_param(seed, 0, i as u64, j as u64, scale);
            }
        }
        // Per-cell params
        for (ci, cp) in self.cells.iter_mut().enumerate() {
            let cell_seed = ci as u64 + 1;
            cp.selector.hidden.init(seed, cell_seed * 100 + 1, scale);
            cp.selector.gate.init(seed, cell_seed * 100 + 2, scale * 0.1);
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
    scratch_rule_outputs: Vec<f64>, // [num_rules * cell_dim]
}

impl ParticleState {
    fn new(spec: &ParticleSpec, model: ParticleModel) -> Self {
        let cd = spec.cell_dim;
        let nc = spec.num_cells;
        let sel_in = 3 * cd;
        let rule_in = 3 * cd + spec.noise_dim;
        Self {
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
            scratch_phi: vec![0.0; 3 * cd],
            scratch_logits: vec![0.0; 256],
            scratch_d_logits: vec![0.0; 256],
            scratch_d_phi: vec![0.0; 3 * cd],
            scratch_softmax: vec![0.0; 256],
            scratch_d_rule_out: vec![0.0; cd],
            scratch_d_rule_h: vec![0.0; spec.rule_hidden],
            scratch_d_gate: vec![0.0; spec.num_rules],
            scratch_d_gate_logits: vec![0.0; spec.num_rules],
            scratch_d_sel_h: vec![0.0; spec.selector_hidden],
            scratch_rule_outputs: vec![0.0; spec.num_rules * cd],
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
        let inv = 1.0 / len as f64;
        for k in 0..len {
            let pos = (self.ctx_pos + cw - len + k) % cw;
            let byte = self.context[pos] as usize;
            let emb = &self.model.embed[byte * cd..(byte + 1) * cd];
            for j in 0..cd {
                self.scratch_ctx[j] += emb[j] * inv;
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

    /// Build selector input p = concat(cell_i, ctx, mean_cells).
    fn build_selector_input(&mut self, cell_idx: usize) {
        let cd = self.model.cell_dim;
        let off = cell_idx * cd;
        self.scratch_p[..cd].copy_from_slice(&self.cells[off..off + cd]);
        self.scratch_p[cd..2 * cd].copy_from_slice(&self.scratch_ctx[..cd]);
        self.scratch_p[2 * cd..3 * cd].copy_from_slice(&self.scratch_mean_cells[..cd]);
    }

    /// Build rule input = concat(p, z) where z = 0 in deterministic mode.
    fn build_rule_input(&mut self) {
        let sel_in = self.model.selector_in_dim;
        let nd = self.model.noise_dim;
        self.scratch_rule_in[..sel_in].copy_from_slice(&self.scratch_p[..sel_in]);
        // z = 0 for deterministic mode
        for j in sel_in..sel_in + nd {
            self.scratch_rule_in[j] = 0.0;
        }
    }

    /// Build featurize vector phi = concat(mean_cells, max_cells, ctx).
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
        // ctx
        self.scratch_phi[2 * cd..3 * cd].copy_from_slice(&self.scratch_ctx[..cd]);
    }

    /// Full forward pass: update latent cells, compute log-probabilities.
    fn forward(&mut self, spec: &ParticleSpec) {
        self.build_ctx();
        self.compute_mean_cells();

        // Latent update (unroll_steps iterations)
        for _step in 0..spec.unroll_steps {
            for ci in 0..self.model.num_cells {
                self.build_selector_input(ci);

                // Selector: hidden = relu(W_sel * p + b_sel)
                self.model.cells[ci].selector.hidden.forward_relu(
                    &self.scratch_p,
                    &mut self.scratch_sel_h,
                );
                // Gate: gate_logits = V_sel * h + c_sel, then softmax
                self.model.cells[ci].selector.gate.forward(
                    &self.scratch_sel_h,
                    &mut self.scratch_gate,
                );
                softmax_inplace(&mut self.scratch_gate[..spec.num_rules]);

                // Build rule input
                self.build_rule_input();

                // Compute weighted delta
                let cd = self.model.cell_dim;
                self.scratch_delta[..cd].fill(0.0);
                for ki in 0..spec.num_rules {
                    let gate_k = self.scratch_gate[ki];
                    // Rule hidden
                    self.model.cells[ci].rules[ki].hidden.forward_relu(
                        &self.scratch_rule_in,
                        &mut self.scratch_rule_h,
                    );
                    // Rule output
                    self.model.cells[ci].rules[ki].output.forward(
                        &self.scratch_rule_h,
                        &mut self.scratch_delta_k,
                    );
                    for j in 0..cd {
                        self.scratch_delta[j] += gate_k * self.scratch_delta_k[j];
                    }
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
        self.model.readout.forward(&self.scratch_phi, &mut self.scratch_logits);

        // Log-softmax with floor
        log_softmax_with_floor(
            &self.scratch_logits,
            &mut self.cached_log_probs,
            spec.min_prob,
        );
        self.cache_valid = true;
    }

    /// Online SGD update after observing byte `y`.
    fn sgd_update(&mut self, y: u8, spec: &ParticleSpec) {
        // Compute softmax of logits
        self.scratch_softmax.copy_from_slice(&self.scratch_logits);
        softmax_inplace(&mut self.scratch_softmax);

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

        // Backprop through selector/rules (simplified: update using last step's cached inputs)
        // We update the selector and rule MLPs using the gradient signal from d_phi
        // propagated through the featurize operation. This is an approximation that
        // treats each cell's update independently.
        let cd = self.model.cell_dim;

        // d_phi components: [mean_cells, max_cells, ctx]
        // Gradient w.r.t. mean_cells is d_phi[0..cd] / num_cells (distributed to each cell)
        // Gradient w.r.t. ctx touches embeddings (we skip embedding SGD for stability)
        // For each cell, approximate gradient of cell state is d_phi[0..cd] / num_cells
        let nc = self.model.num_cells;
        let inv_nc = 1.0 / nc as f64;

        for ci in 0..nc {
            // Rebuild selector input for this cell
            self.build_selector_input(ci);
            self.build_rule_input();

            // Approximate gradient on cell delta = d_phi[0..cd] * inv_nc
            // (ignoring max_cells gradient for simplicity — conservative)
            let d_delta_scale = inv_nc;

            // Selector hidden
            self.model.cells[ci].selector.hidden.forward_relu(
                &self.scratch_p,
                &mut self.scratch_sel_h,
            );
            // Selector gate
            self.model.cells[ci].selector.gate.forward(
                &self.scratch_sel_h,
                &mut self.scratch_gate,
            );
            softmax_inplace(&mut self.scratch_gate[..spec.num_rules]);

            // For each rule, compute d_rule_output and update
            for ki in 0..spec.num_rules {
                let gate_k = self.scratch_gate[ki];
                // d_rule_out_k = d_delta * gate_k
                // We need to scale by d_delta which is d_phi[0..cd] * d_delta_scale
                self.scratch_d_rule_out[..cd].fill(0.0);
                for j in 0..cd {
                    self.scratch_d_rule_out[j] = self.scratch_d_phi[j] * d_delta_scale * gate_k;
                }

                // Rule hidden forward
                self.model.cells[ci].rules[ki].hidden.forward_relu(
                    &self.scratch_rule_in,
                    &mut self.scratch_rule_h,
                );
                self.model.cells[ci].rules[ki].output.forward(
                    &self.scratch_rule_h,
                    &mut self.scratch_delta_k,
                );
                let out_row = &mut self.scratch_rule_outputs[ki * cd..(ki + 1) * cd];
                out_row.copy_from_slice(&self.scratch_delta_k[..cd]);

                // Update rule output layer
                self.model.cells[ci].rules[ki].output.sgd_update(
                    &self.scratch_d_rule_out,
                    &self.scratch_rule_h,
                    spec.learning_rate_rule,
                    spec.grad_clip,
                );

                // Backprop to rule hidden
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
                // ReLU backward
                for (j, h) in self.scratch_rule_h.iter().enumerate().take(rh) {
                    if *h <= 0.0 {
                        self.scratch_d_rule_h[j] = 0.0;
                    }
                }
                // Update rule hidden layer
                self.model.cells[ci].rules[ki].hidden.sgd_update(
                    &self.scratch_d_rule_h[..rh],
                    &self.scratch_rule_in,
                    spec.learning_rate_rule,
                    spec.grad_clip,
                );
            }

            // Selector gradient (via gate → delta coupling)
            // d_gate[k] = dot(d_delta, rule_output_k) for each rule k
            self.scratch_d_gate[..spec.num_rules].fill(0.0);
            for ki in 0..spec.num_rules {
                let out_row = &self.scratch_rule_outputs[ki * cd..(ki + 1) * cd];
                for j in 0..cd {
                    self.scratch_d_gate[ki] += self.scratch_d_phi[j] * d_delta_scale * out_row[j];
                }
            }
            // Softmax backward: d_gate_logits = gate * (d_gate - dot(gate, d_gate))
            let dot_gd: f64 = (0..spec.num_rules)
                .map(|k| self.scratch_gate[k] * self.scratch_d_gate[k])
                .sum();
            self.scratch_d_gate_logits[..spec.num_rules].fill(0.0);
            for k in 0..spec.num_rules {
                self.scratch_d_gate_logits[k] =
                    self.scratch_gate[k] * (self.scratch_d_gate[k] - dot_gd);
            }

            // Update selector gate layer
            self.model.cells[ci].selector.gate.sgd_update(
                &self.scratch_d_gate_logits[..spec.num_rules],
                &self.scratch_sel_h,
                spec.learning_rate_selector,
                spec.grad_clip,
            );

            // Backprop through gate to selector hidden
            let sh = spec.selector_hidden;
            self.scratch_d_sel_h[..sh].fill(0.0);
            for r in 0..spec.num_rules {
                let g = clip(self.scratch_d_gate_logits[r], spec.grad_clip);
                if g.abs() < 1e-15 {
                    continue;
                }
                for c in 0..sh {
                    self.scratch_d_sel_h[c] +=
                        g * self.model.cells[ci].selector.gate.weights[r * sh + c];
                }
            }
            // ReLU backward
            for (j, h) in self.scratch_sel_h.iter().enumerate().take(sh) {
                if *h <= 0.0 {
                    self.scratch_d_sel_h[j] = 0.0;
                }
            }
            // Update selector hidden layer
            self.model.cells[ci].selector.hidden.sgd_update(
                &self.scratch_d_sel_h[..sh],
                &self.scratch_p,
                spec.learning_rate_selector,
                spec.grad_clip,
            );
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
    /// Create a new particle runtime from a spec.
    pub fn new(spec: &ParticleSpec) -> Self {
        let n = spec.num_particles;

        // Initialize model template
        let mut model_template = ParticleModel::new(spec);
        model_template.init(spec.seed, spec);

        // Create particles with independent model copies
        let particles: Vec<ParticleState> = (0..n)
            .map(|_| ParticleState::new(spec, model_template.clone()))
            .collect();

        let log_w = -(n as f64).ln();
        Self {
            spec: spec.clone(),
            particles,
            log_weights: vec![log_w; n],
            mix_log_probs: [0.0; 256],
            mix_pdf: vec![0.0; 256],
            cache_valid: false,
            step_idx: 0,
            scratch_lse: vec![0.0; n],
        }
    }

    /// Ensure all particles have valid cached log-probabilities.
    fn ensure_predictions(&mut self) {
        if self.cache_valid {
            return;
        }
        let spec = &self.spec;
        for p in &mut self.particles {
            if !p.cache_valid {
                p.forward(spec);
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
                self.scratch_lse[i] =
                    self.log_weights[i] + self.particles[i].cached_log_probs[v];
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
        for i in 0..n {
            self.log_weights[i] += self.particles[i].cached_log_probs[symbol as usize];
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

        // (3) Online SGD per particle
        for p in &mut self.particles {
            p.sgd_update(symbol, spec);
        }

        // (4) Push context byte
        for p in &mut self.particles {
            p.push_context(symbol);
        }

        // (5) Resample check
        self.maybe_resample();

        // Invalidate caches
        for p in &mut self.particles {
            p.cache_valid = false;
        }
        self.cache_valid = false;
        self.step_idx += 1;

        log_prob
    }

    /// Check effective sample size and resample if needed.
    fn maybe_resample(&mut self) {
        let n = self.particles.len();
        if n <= 1 {
            return;
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
            return;
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
        let mut u = 0.5 * step; // fixed offset
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
        let new_particles: Vec<ParticleState> =
            indices.iter().map(|&idx| self.particles[idx].clone()).collect();
        self.particles = new_particles;

        // Mutate a fraction of particles
        let n_mutate = ((self.spec.mutate_fraction * n as f64).round() as usize).min(n);
        for mi in 0..n_mutate {
            self.mutate_particle(mi);
        }

        // Reset log-weights to uniform
        let uniform = -(n as f64).ln();
        for w in &mut self.log_weights {
            *w = uniform;
        }
    }

    /// Apply deterministic hash-noise mutation to a particle's model parameters.
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

        // Mutate model parameters
        // Embedding
        for v in p.model.embed.iter_mut() {
            let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 1)) * scale;
            *v += noise;
            param_idx += 1;
        }

        // Cell params
        for cp in p.model.cells.iter_mut() {
            for v in cp.selector.hidden.weights.iter_mut() {
                let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 2)) * scale;
                *v += noise;
                param_idx += 1;
            }
            for v in cp.selector.hidden.bias.iter_mut() {
                let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 3)) * scale;
                *v += noise;
                param_idx += 1;
            }
            for v in cp.selector.gate.weights.iter_mut() {
                let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 4)) * scale;
                *v += noise;
                param_idx += 1;
            }
            for v in cp.selector.gate.bias.iter_mut() {
                let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 5)) * scale;
                *v += noise;
                param_idx += 1;
            }
            for rule in cp.rules.iter_mut() {
                for v in rule.hidden.weights.iter_mut() {
                    let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 6)) * scale;
                    *v += noise;
                    param_idx += 1;
                }
                for v in rule.hidden.bias.iter_mut() {
                    let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 7)) * scale;
                    *v += noise;
                    param_idx += 1;
                }
                for v in rule.output.weights.iter_mut() {
                    let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 8)) * scale;
                    *v += noise;
                    param_idx += 1;
                }
                for v in rule.output.bias.iter_mut() {
                    let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 9)) * scale;
                    *v += noise;
                    param_idx += 1;
                }
            }
        }

        // Readout
        for v in p.model.readout.weights.iter_mut() {
            let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 10)) * scale;
            *v += noise;
            param_idx += 1;
        }
        for v in p.model.readout.bias.iter_mut() {
            let noise = hash_to_f64(det_hash(seed ^ step, pi, param_idx, 11)) * scale;
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
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "PDF sum = {sum}, expected ~1.0"
        );
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
