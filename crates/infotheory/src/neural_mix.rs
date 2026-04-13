use crate::backends::text_context::NeuralContextState;
pub(crate) use crate::backends::text_context::NeuralHistoryState;

/// Shared two-stage bytewise neural mixer core used by runtime and compression predictors.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct NeuralMixCore {
    stage1_tables: Vec<Vec<f64>>,
    stage2_table: Vec<[f64; Self::STAGE1_CONTEXTS]>,
    stage1_lr: f64,
    stage2_lr: f64,
    update_skip_threshold: f64,
    context: NeuralContextState,
    expert_count: usize,
    expert_probs: Vec<f64>,
    stage1_mix: Vec<f64>,
    stage1_probs: Vec<f64>,
    stage2_mix: Vec<f64>,
    expert_weights: Vec<f64>,
    mix_prob: f64,
    context_mixtures_valid: bool,
    evaluated: bool,
}

#[allow(dead_code)]
impl NeuralMixCore {
    const STAGE1_CONTEXTS: usize = 4;
    const STAGE1_TABLE_SIZES: [usize; Self::STAGE1_CONTEXTS] = [1, 256, 4096, 4096];
    const STAGE2_TABLE_SIZE: usize = 2048;

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
            let mut table = vec![0.0; table_size.saturating_mul(expert_count)];
            if ctx_idx == 0 && expert_count > 0 {
                for (dst, &p) in table[..expert_count].iter_mut().zip(prior_weights.iter()) {
                    let p = if p.is_finite() { p.max(1e-12) } else { 1e-12 };
                    *dst = p.ln();
                }
            }
            stage1_tables.push(table);
        }

        let stage2_table = vec![[0.0; Self::STAGE1_CONTEXTS]; Self::STAGE2_TABLE_SIZE];

        Self {
            stage1_tables,
            stage2_table,
            stage1_lr,
            stage2_lr,
            update_skip_threshold,
            context: NeuralContextState::default(),
            expert_count,
            expert_probs: vec![0.0; expert_count],
            stage1_mix: vec![0.0; Self::STAGE1_CONTEXTS * expert_count],
            stage1_probs: vec![0.0; Self::STAGE1_CONTEXTS],
            stage2_mix: vec![0.0; Self::STAGE1_CONTEXTS],
            expert_weights: vec![0.0; expert_count],
            mix_prob: 1.0 / 256.0,
            context_mixtures_valid: false,
            evaluated: false,
        }
    }

    #[inline]
    pub(crate) fn history_state(&self) -> NeuralHistoryState {
        self.context
    }

    #[inline]
    pub(crate) fn set_context_state(&mut self, context: NeuralContextState) {
        self.context = context;
        self.context_mixtures_valid = false;
        self.evaluated = false;
    }

    #[inline]
    fn stage1_row_bounds(&self, ctx_idx: usize) -> (usize, usize) {
        let start = ctx_idx * self.expert_count;
        (start, start + self.expert_count)
    }

    #[inline]
    pub(crate) fn evaluate_symbol(&mut self, expert_log_probs: &[f64], min_prob: f64) -> f64 {
        debug_assert_eq!(expert_log_probs.len(), self.expert_count);
        let floor = min_prob.clamp(1e-12, 0.49);
        for (dst, &lp) in self.expert_probs.iter_mut().zip(expert_log_probs.iter()) {
            let p = if lp.is_finite() { lp.exp() } else { floor };
            *dst = p.max(floor).min(1.0 - floor);
        }

        self.ensure_context_mixtures();

        let mut mix = 0.0;
        for k in 0..Self::STAGE1_CONTEXTS {
            let row = &self.stage1_mix[(k * self.expert_count)..((k + 1) * self.expert_count)];
            let mut p_k = 0.0;
            for (&weight, &expert_prob) in row.iter().zip(self.expert_probs.iter()) {
                p_k += weight * expert_prob;
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
        self.ensure_context_mixtures();
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
        let old_stage2_mix = [
            self.stage2_mix[0],
            self.stage2_mix[1],
            self.stage2_mix[2],
            self.stage2_mix[3],
        ];
        {
            let entry2 = &mut self.stage2_table[stage2_idx];
            for (k, logit) in entry2.iter_mut().enumerate() {
                let grad = old_stage2_mix[k] * (self.stage1_probs[k] - p_mix) / p_mix;
                *logit = sanitize_weight(*logit + self.stage2_lr * grad);
            }
        }

        for (k, &ctx_i) in stage1_idx.iter().enumerate() {
            let (start, end) = self.stage1_row_bounds(ctx_i);
            let entry = &mut self.stage1_tables[k][start..end];
            let r_k = old_stage2_mix[k];
            let p_k = self.stage1_probs[k];
            let row = &self.stage1_mix[(k * self.expert_count)..((k + 1) * self.expert_count)];
            for ((logit, &weight), &expert_prob) in entry
                .iter_mut()
                .zip(row.iter())
                .zip(self.expert_probs.iter())
            {
                let grad = r_k * weight * (expert_prob - p_k) / p_mix;
                *logit = sanitize_weight(*logit + self.stage1_lr * grad);
            }
        }
        self.evaluated = false;
        self.context_mixtures_valid = false;
    }

    #[inline]
    fn ensure_context_mixtures(&mut self) {
        if self.context_mixtures_valid {
            return;
        }
        self.compute_context_mixtures();
        self.context_mixtures_valid = true;
    }

    #[inline]
    fn stage1_context_indices(&self) -> [usize; Self::STAGE1_CONTEXTS] {
        if !self.context.has_history {
            return [0, 0, 0, 0];
        }
        [
            0,
            self.context.prev1 as usize,
            hash_fields(
                &[
                    self.context.prev1_class,
                    self.context.prev2_class,
                    self.context.word_len_bucket,
                    self.context.prev_word_class,
                    self.context.bracket_bucket,
                    self.context.quote_flags,
                    self.context.utf8_left,
                    self.context.sentence_boundary as u8,
                    self.context.paragraph_break as u8,
                ],
                Self::STAGE1_TABLE_SIZES[2],
            ),
            hash_fields(
                &[
                    self.context.repeat_len_bucket,
                    self.context.copied_last_byte as u8,
                    self.context.run_len.min(63) as u8,
                    self.context.prev1_class,
                    self.context.prev2_class,
                ],
                Self::STAGE1_TABLE_SIZES[3],
            ),
        ]
    }

    #[inline]
    fn stage2_context_index(&self) -> usize {
        if !self.context.has_history {
            return 0;
        }
        hash_fields(
            &[
                self.context.prev1,
                self.context.prev2,
                self.context.prev1_class,
                self.context.prev2_class,
                self.context.word_len_bucket,
                self.context.prev_word_class,
                self.context.bracket_bucket,
                self.context.quote_flags,
                self.context.utf8_left,
                self.context.repeat_len_bucket,
                self.context.copied_last_byte as u8,
                self.context.sentence_boundary as u8,
                self.context.paragraph_break as u8,
                self.context.run_len.min(127) as u8,
            ],
            Self::STAGE2_TABLE_SIZE,
        )
    }

    #[inline]
    fn compute_context_mixtures(&mut self) {
        let stage1_idx = self.stage1_context_indices();
        self.expert_weights.fill(0.0);

        for (k, &ctx_i) in stage1_idx.iter().enumerate() {
            let (start, end) = self.stage1_row_bounds(ctx_i);
            let entry = &self.stage1_tables[k][start..end];
            let row = &mut self.stage1_mix[(k * self.expert_count)..((k + 1) * self.expert_count)];
            softmax_into(entry, row);
        }

        let stage2_idx = self.stage2_context_index();
        let entry2 = &self.stage2_table[stage2_idx];
        softmax_into(entry2, &mut self.stage2_mix);

        for k in 0..Self::STAGE1_CONTEXTS {
            let row = &self.stage1_mix[(k * self.expert_count)..((k + 1) * self.expert_count)];
            let r_k = self.stage2_mix[k];
            for (expert_weight, &weight) in self.expert_weights.iter_mut().zip(row.iter()) {
                *expert_weight += r_k * weight;
            }
        }
    }
}

#[inline]
fn hash_fields(values: &[u8], modulo: usize) -> usize {
    let mut h = 0x9E37_79B9u32;
    for &value in values {
        h ^= value as u32;
        h = h.rotate_left(5).wrapping_mul(0x85EB_CA6B);
    }
    (h as usize) % modulo
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
