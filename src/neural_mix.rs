#[derive(Clone)]
struct NeuralStage1Entry {
    logits: Vec<f64>,
}

impl NeuralStage1Entry {
    fn new(width: usize) -> Self {
        Self {
            logits: vec![0.0; width],
        }
    }
}

#[derive(Clone)]
struct NeuralStage2Entry {
    logits: Vec<f64>,
}

impl NeuralStage2Entry {
    fn new(width: usize) -> Self {
        Self {
            logits: vec![0.0; width],
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
    expert_probs: Vec<f64>,
    stage1_mix: Vec<f64>,
    stage1_probs: Vec<f64>,
    stage2_mix: Vec<f64>,
    expert_weights: Vec<f64>,
    mix_prob: f64,
    evaluated: bool,
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
                    for (dst, &p) in entry.logits.iter_mut().zip(prior_weights.iter()) {
                        let p = if p.is_finite() { p.max(1e-12) } else { 1e-12 };
                        *dst = p.ln();
                    }
                }
                table.push(entry);
            }
            stage1_tables.push(table);
        }

        let mut stage2_table = Vec::with_capacity(Self::STAGE2_TABLE_SIZE);
        for _ in 0..Self::STAGE2_TABLE_SIZE {
            let mut entry = NeuralStage2Entry::new(Self::STAGE1_CONTEXTS);
            for logit in &mut entry.logits {
                *logit = 0.0;
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
            expert_probs: vec![0.0; expert_count],
            stage1_mix: vec![0.0; Self::STAGE1_CONTEXTS * expert_count],
            stage1_probs: vec![0.0; Self::STAGE1_CONTEXTS],
            stage2_mix: vec![0.0; Self::STAGE1_CONTEXTS],
            expert_weights: vec![0.0; expert_count],
            mix_prob: 1.0 / 256.0,
            evaluated: false,
        }
    }

    #[inline]
    pub(crate) fn history_state(&self) -> NeuralHistoryState {
        self.history
    }

    #[inline]
    pub(crate) fn evaluate_symbol(&mut self, expert_log_probs: &[f64], min_prob: f64) -> f64 {
        debug_assert_eq!(expert_log_probs.len(), self.expert_count);
        let floor = min_prob.clamp(1e-12, 0.49);
        for i in 0..self.expert_count {
            let lp = expert_log_probs[i];
            let p = if lp.is_finite() { lp.exp() } else { floor };
            self.expert_probs[i] = p.max(floor).min(1.0 - floor);
        }

        self.compute_context_mixtures();

        let mut mix = 0.0;
        for k in 0..Self::STAGE1_CONTEXTS {
            let n = self.expert_count;
            let row = &self.stage1_mix[(k * n)..((k + 1) * n)];
            let mut p_k = 0.0;
            for i in 0..n {
                p_k += row[i] * self.expert_probs[i];
            }
            let p_k = p_k.max(floor).min(1.0 - floor);
            self.stage1_probs[k] = p_k;
            mix += self.stage2_mix[k] * p_k;
        }
        self.mix_prob = mix.max(floor).min(1.0 - floor);
        self.evaluated = true;
        self.mix_prob
    }

    #[inline]
    pub(crate) fn evaluate_expert_weights(&mut self) {
        self.compute_context_mixtures();
        self.evaluated = false;
    }

    #[inline]
    pub(crate) fn expert_weights(&self) -> &[f64] {
        &self.expert_weights
    }

    #[inline]
    pub(crate) fn update_weights_symbol(&mut self, expert_log_probs: &[f64], min_prob: f64) {
        debug_assert_eq!(expert_log_probs.len(), self.expert_count);
        if !self.evaluated {
            self.evaluate_symbol(expert_log_probs, min_prob);
        }
        let p_mix = self.mix_prob.max(1e-12);
        let error_mag = (1.0 - p_mix).abs();
        if error_mag <= self.update_skip_threshold {
            return;
        }

        let stage1_idx = self.stage1_context_indices();
        let stage2_idx = self.stage2_context_index();
        let old_stage2_mix = self.stage2_mix.clone();
        {
            let entry2 = &mut self.stage2_table[stage2_idx];
            for k in 0..Self::STAGE1_CONTEXTS {
                let grad = old_stage2_mix[k] * (self.stage1_probs[k] - p_mix) / p_mix;
                entry2.logits[k] = sanitize_weight(entry2.logits[k] + self.stage2_lr * grad);
            }
        }

        for (k, &ctx_i) in stage1_idx.iter().enumerate() {
            let entry = &mut self.stage1_tables[k][ctx_i];
            let r_k = old_stage2_mix[k];
            let p_k = self.stage1_probs[k];
            let n = self.expert_count;
            let row = &self.stage1_mix[(k * n)..((k + 1) * n)];
            for i in 0..n {
                let grad = r_k * row[i] * (self.expert_probs[i] - p_k) / p_mix;
                entry.logits[i] = sanitize_weight(entry.logits[i] + self.stage1_lr * grad);
            }
        }
        self.evaluated = false;
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

    #[inline]
    fn compute_context_mixtures(&mut self) {
        let stage1_idx = self.stage1_context_indices();
        let n = self.expert_count;
        self.expert_weights.fill(0.0);

        for (k, &ctx_i) in stage1_idx.iter().enumerate() {
            let entry = &self.stage1_tables[k][ctx_i];
            let row = &mut self.stage1_mix[(k * n)..((k + 1) * n)];
            softmax_into(&entry.logits, row);
        }

        let stage2_idx = self.stage2_context_index();
        let entry2 = &self.stage2_table[stage2_idx];
        softmax_into(&entry2.logits, &mut self.stage2_mix);

        for k in 0..Self::STAGE1_CONTEXTS {
            let n = self.expert_count;
            let row = &self.stage1_mix[(k * n)..((k + 1) * n)];
            let r_k = self.stage2_mix[k];
            for i in 0..n {
                self.expert_weights[i] += r_k * row[i];
            }
        }
    }
}

#[inline]
fn softmax_into(logits: &[f64], out: &mut [f64]) {
    debug_assert_eq!(logits.len(), out.len());
    if out.is_empty() {
        return;
    }
    let mut max_v = f64::NEG_INFINITY;
    for &v in logits {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        let u = 1.0 / (out.len() as f64);
        out.fill(u);
        return;
    }
    let mut sum = 0.0;
    for (dst, &v) in out.iter_mut().zip(logits.iter()) {
        let x = (v - max_v).exp();
        *dst = x;
        sum += x;
    }
    if sum <= 0.0 || !sum.is_finite() {
        let u = 1.0 / (out.len() as f64);
        out.fill(u);
        return;
    }
    let inv = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv;
    }
}

#[inline]
fn sanitize_weight(w: f64) -> f64 {
    if w.is_finite() { w } else { 0.0 }
}
