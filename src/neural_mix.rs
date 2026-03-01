use crate::simd_math::{affine3_wide, axpy_wide, dot_wide, logsumexp_wide};

#[derive(Clone)]
struct NeuralStage1Entry {
    weights: Vec<f64>,
    bias: f64,
}

impl NeuralStage1Entry {
    fn new(width: usize) -> Self {
        Self {
            weights: vec![0.0; width],
            bias: 0.0,
        }
    }
}

#[derive(Clone)]
struct NeuralStage2Entry {
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl NeuralStage2Entry {
    fn new(width: usize) -> Self {
        Self {
            weights: vec![0.0; width],
            bias: vec![0.0; 256],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NeuralHistoryState {
    pub(crate) prev1: u8,
    pub(crate) prev2: u8,
    pub(crate) run_len: u16,
    pub(crate) has_history: bool,
}

/// Shared two-stage bytewise neural mixer core used by runtime and compression predictors.
#[derive(Clone)]
pub(crate) struct NeuralMixCore {
    stage1_tables: Vec<Vec<NeuralStage1Entry>>,
    stage2_table: Vec<NeuralStage2Entry>,
    stage1_lr: f64,
    stage2_lr: f64,
    update_skip_threshold: f64,
    history: NeuralHistoryState,
    expert_count: usize,
    stage1_out: Vec<f64>,
    energy: Vec<f64>,
    probs: Vec<f64>,
    errors: Vec<f64>,
}

impl NeuralMixCore {
    const STAGE1_CONTEXTS: usize = 3;
    const STAGE1_TABLE_SIZES: [usize; Self::STAGE1_CONTEXTS] = [1, 256, 1024];
    const STAGE2_TABLE_SIZE: usize = 512;

    pub(crate) fn new(
        expert_count: usize,
        prior_weights: &[f64],
        stage1_lr: f64,
        stage2_lr: f64,
        update_skip_threshold: f64,
    ) -> Self {
        debug_assert_eq!(prior_weights.len(), expert_count);
        let mut stage1_tables = Vec::with_capacity(Self::STAGE1_CONTEXTS);
        for (ctx_idx, table_size) in Self::STAGE1_TABLE_SIZES.iter().enumerate() {
            let mut table = Vec::with_capacity(*table_size);
            for _ in 0..*table_size {
                let mut entry = NeuralStage1Entry::new(expert_count);
                if ctx_idx == 0 {
                    entry.weights.clone_from_slice(prior_weights);
                }
                table.push(entry);
            }
            stage1_tables.push(table);
        }

        let mut stage2_table = Vec::with_capacity(Self::STAGE2_TABLE_SIZE);
        for _ in 0..Self::STAGE2_TABLE_SIZE {
            let mut entry = NeuralStage2Entry::new(Self::STAGE1_CONTEXTS);
            for w in &mut entry.weights {
                *w = 1.0 / (Self::STAGE1_CONTEXTS as f64);
            }
            stage2_table.push(entry);
        }

        Self {
            stage1_tables,
            stage2_table,
            stage1_lr,
            stage2_lr,
            update_skip_threshold,
            history: NeuralHistoryState::default(),
            expert_count,
            stage1_out: vec![0.0; Self::STAGE1_CONTEXTS * 256],
            energy: vec![0.0; 256],
            probs: vec![0.0; 256],
            errors: vec![0.0; 256],
        }
    }

    #[inline]
    pub(crate) fn history_state(&self) -> NeuralHistoryState {
        self.history
    }

    #[inline]
    pub(crate) fn evaluate(&mut self, expert_logits: &[f64]) {
        debug_assert_eq!(expert_logits.len(), self.expert_count * 256);
        let stage1_idx = self.stage1_context_indices();

        for (k, &ctx_i) in stage1_idx.iter().enumerate() {
            let entry = &self.stage1_tables[k][ctx_i];
            let row = &mut self.stage1_out[(k * 256)..((k + 1) * 256)];
            row.fill(entry.bias);
            for i in 0..self.expert_count {
                let feat = &expert_logits[(i * 256)..((i + 1) * 256)];
                axpy_wide(row, entry.weights[i], feat);
            }
            let log_z = logsumexp_wide(row);
            for v in row.iter_mut() {
                *v -= log_z;
            }
        }

        let stage2_idx = self.stage2_context_index();
        let entry2 = &self.stage2_table[stage2_idx];
        affine3_wide(
            &mut self.energy,
            &entry2.bias,
            [entry2.weights[0], entry2.weights[1], entry2.weights[2]],
            &self.stage1_out[0..256],
            &self.stage1_out[256..512],
            &self.stage1_out[512..768],
        );

        let log_z = logsumexp_wide(&self.energy);
        for b in 0..256usize {
            self.probs[b] = (self.energy[b] - log_z).exp();
        }
    }

    #[inline]
    pub(crate) fn prob(&self, symbol: u8) -> f64 {
        self.probs[symbol as usize]
    }

    #[cfg(feature = "backend-rwkv")]
    #[inline]
    pub(crate) fn probs(&self) -> &[f64] {
        &self.probs
    }

    #[cfg(feature = "backend-rwkv")]
    #[inline]
    pub(crate) fn probs_mut(&mut self) -> &mut [f64] {
        &mut self.probs
    }

    #[inline]
    pub(crate) fn update_weights(&mut self, expert_logits: &[f64], symbol: u8) {
        debug_assert_eq!(expert_logits.len(), self.expert_count * 256);
        let y = symbol as usize;
        let error_mag = (1.0 - self.probs[y]).abs();
        if error_mag <= self.update_skip_threshold {
            return;
        }

        for b in 0..256usize {
            self.errors[b] = if b == y { 1.0 } else { 0.0 } - self.probs[b];
        }

        let stage1_idx = self.stage1_context_indices();
        let stage2_idx = self.stage2_context_index();
        let old_stage2_weights = {
            let w = &self.stage2_table[stage2_idx].weights;
            [w[0], w[1], w[2]]
        };

        {
            let entry2 = &mut self.stage2_table[stage2_idx];
            for k in 0..Self::STAGE1_CONTEXTS {
                let grad = dot_wide(&self.errors, &self.stage1_out[(k * 256)..((k + 1) * 256)]);
                entry2.weights[k] = sanitize_weight(entry2.weights[k] + self.stage2_lr * grad);
            }
            for b in 0..256usize {
                entry2.bias[b] = sanitize_weight(entry2.bias[b] + self.stage2_lr * self.errors[b]);
            }
        }

        let grad_bias_base: f64 = self.errors.iter().sum();
        for (k, &ctx_i) in stage1_idx.iter().enumerate() {
            let v = old_stage2_weights[k];
            let entry = &mut self.stage1_tables[k][ctx_i];

            let grad_bias = grad_bias_base * v;
            entry.bias = sanitize_weight(entry.bias + self.stage1_lr * grad_bias);

            for i in 0..self.expert_count {
                let feat = &expert_logits[(i * 256)..((i + 1) * 256)];
                let mut grad = dot_wide(&self.errors, feat);
                grad *= v;
                entry.weights[i] = sanitize_weight(entry.weights[i] + self.stage1_lr * grad);
            }
        }
    }

    #[inline]
    pub(crate) fn update_history(&mut self, symbol: u8) {
        if self.history.has_history && symbol == self.history.prev1 {
            self.history.run_len = self.history.run_len.saturating_add(1).min(255);
        } else {
            self.history.run_len = 1;
        }
        self.history.prev2 = self.history.prev1;
        self.history.prev1 = symbol;
        self.history.has_history = true;
    }

    #[inline]
    fn stage1_context_indices(&self) -> [usize; Self::STAGE1_CONTEXTS] {
        if !self.history.has_history {
            return [0, 0, 0];
        }
        let run_bucket = (self.history.run_len.min(63) as usize) & 0x3f;
        let h = ((self.history.prev1 as usize) << 10)
            ^ ((self.history.prev2 as usize) << 2)
            ^ run_bucket
            ^ (((self.history.prev1 ^ self.history.prev2) as usize) << 5);
        [
            0,
            self.history.prev1 as usize,
            h % Self::STAGE1_TABLE_SIZES[2],
        ]
    }

    #[inline]
    fn stage2_context_index(&self) -> usize {
        if !self.history.has_history {
            return 0;
        }
        let run_bucket = (self.history.run_len.min(127) as usize) & 0x7f;
        let h = ((self.history.prev1 as usize) << 8) ^ (self.history.prev2 as usize) ^ run_bucket;
        h % Self::STAGE2_TABLE_SIZE
    }
}

#[cfg(feature = "backend-rwkv")]
#[inline]
fn clamp_prob(p: f64, min_prob: f64) -> f64 {
    if p.is_finite() {
        p.max(min_prob)
    } else {
        min_prob
    }
}

#[cfg(feature = "backend-rwkv")]
#[inline]
pub(crate) fn fill_log_probs_from_pdf_row(
    pdf_row: &[f64],
    min_prob: f64,
    out_log_probs: &mut [f64],
) {
    debug_assert!(pdf_row.len() >= 256);
    debug_assert!(out_log_probs.len() >= 256);
    for b in 0..256usize {
        out_log_probs[b] = clamp_prob(pdf_row[b], min_prob).ln();
    }
}

#[inline]
pub(crate) fn normalize_log_prob_row_and_make_logits(
    log_probs: &mut [f64],
    logits_out: &mut [f64],
    min_prob: f64,
) {
    debug_assert!(log_probs.len() >= 256);
    debug_assert!(logits_out.len() >= 256);
    let min_log = min_prob.ln();
    let row = &mut log_probs[..256];
    let log_z = logsumexp_wide(row);
    if !log_z.is_finite() {
        for b in 0..256usize {
            row[b] = min_log;
            logits_out[b] = min_log;
        }
        return;
    }
    for b in 0..256usize {
        let v = row[b] - log_z;
        let nv = if v.is_finite() { v } else { min_log };
        row[b] = nv;
        logits_out[b] = nv.max(min_log);
    }
}

#[inline]
fn sanitize_weight(w: f64) -> f64 {
    if w.is_finite() { w } else { 0.0 }
}
