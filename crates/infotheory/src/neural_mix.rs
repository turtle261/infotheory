use crate::backends::text_context::NeuralContextState;
pub(crate) use crate::backends::text_context::NeuralHistoryState;
use crate::byte_prefix::{
    BYTE_SYMBOLS, BytePrefixCdf, MsbPrefixRange, advanced_prefix_code, normalize_pdf_mass_only,
};

/// Shared two-stage bytewise neural mixer core used by runtime and compression predictors.
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

/// Byte-history and MSB-prefix context for [`LogisticMixCore`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LogisticMixContext {
    pub(crate) history: NeuralContextState,
    pub(crate) bit_idx: u8,
    pub(crate) prefix: u16,
    pub(crate) match_len_bucket: u8,
    pub(crate) match_predicted_class: u8,
}

/// Lightweight match-state summary exposed to the logistic mixer context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LogisticMatchState {
    pub(crate) len_bucket: u8,
    pub(crate) predicted_class: u8,
}

impl LogisticMatchState {
    /// Merge two expert match summaries for logistic context selection.
    ///
    /// Prefers the longer match bucket; on a length tie, prefer a non-zero
    /// predicted byte class over an unset class.
    #[inline]
    pub(crate) fn merge_with(self, other: Self) -> Self {
        if other.len_bucket > self.len_bucket {
            other
        } else if other.len_bucket == self.len_bucket
            && other.predicted_class != 0
            && self.predicted_class == 0
        {
            Self {
                len_bucket: self.len_bucket,
                predicted_class: other.predicted_class,
            }
        } else {
            self
        }
    }
}

/// Fold expert match states with [`LogisticMatchState::merge_with`].
#[inline]
pub(crate) fn fold_logistic_match_states(
    states: impl IntoIterator<Item = LogisticMatchState>,
) -> LogisticMatchState {
    states.into_iter().fold(
        LogisticMatchState::default(),
        LogisticMatchState::merge_with,
    )
}

/// Borrowed scratch session for speculative byte-PDF materialization.
///
/// Runtime and compression adapters use the same session so the product of
/// mixed conditional bits has one implementation and one undo discipline.
pub(crate) struct LogisticBytePdfSession<'a> {
    pub(crate) logistic: &'a mut LogisticMixCore,
    pub(crate) expert_prefix_cdfs: &'a [Box<BytePrefixCdf>],
    pub(crate) history: NeuralContextState,
    pub(crate) match_state: LogisticMatchState,
    pub(crate) min_prob: f64,
    pub(crate) bit_probs: &'a mut [f64],
    pub(crate) ranges: &'a mut [MsbPrefixRange],
    pub(crate) undo: &'a mut LogisticMixUndo,
}

impl LogisticBytePdfSession<'_> {
    /// Return the current adaptive stretch-mixer probability of `symbol`.
    ///
    /// The requested byte alone is traversed. Mixer fitting after each
    /// speculative bit is visible to the later bits of that byte, exactly as
    /// on the adaptive native-prefix path, and the undo journal restores all
    /// fitted and transient mixer state before returning.
    pub(crate) fn score_adaptive(&mut self, symbol: u8) -> f64 {
        self.logistic.begin_undo(self.undo);
        let probability: f64 = self.score_path(symbol, LogisticByteScoringMode::Adaptive);
        self.logistic.restore_undo(self.undo);
        probability
    }

    /// Return the current frozen stretch-mixer probability of `symbol`.
    ///
    /// Frozen traversal changes the byte prefix used for context selection but
    /// never fits mixer weights. All transient context/cache state is restored
    /// before returning.
    pub(crate) fn score_frozen(&mut self, symbol: u8) -> f64 {
        self.logistic.begin_undo(self.undo);
        let probability: f64 = self.score_path(symbol, LogisticByteScoringMode::Frozen);
        self.logistic.restore_undo(self.undo);
        probability
    }

    /// Fill `out_pdf` with the current adaptive multi-expert byte PDF.
    ///
    /// Every byte is an MSB-first product of mixed bit probabilities. Mixer
    /// updates within each candidate path are rolled back through one reusable
    /// flat journal, so the mixer is unchanged and the hot traversal performs
    /// no per-row heap allocation after the journal reaches capacity.
    pub(crate) fn materialize_adaptive(&mut self, out_pdf: &mut [f64]) {
        self.validate_scratch(out_pdf);
        out_pdf.fill(0.0);
        for symbol in 0..=u8::MAX {
            out_pdf[symbol as usize] = self.score_adaptive(symbol);
        }
        normalize_pdf_mass_only(out_pdf);
    }

    /// Fill `out_pdf` with the current frozen multi-expert byte PDF.
    ///
    /// The mixer parameters remain fixed across every bit and candidate. One
    /// outer undo scope restores transient context/cache fields after the
    /// traversal; because no rows are fitted, the journal stays empty.
    pub(crate) fn materialize_frozen(&mut self, out_pdf: &mut [f64]) {
        self.validate_scratch(out_pdf);
        out_pdf.fill(0.0);
        self.logistic.begin_undo(self.undo);
        for symbol in 0..=u8::MAX {
            out_pdf[symbol as usize] = self.score_path(symbol, LogisticByteScoringMode::Frozen);
        }
        self.logistic.restore_undo(self.undo);
        normalize_pdf_mass_only(out_pdf);
    }

    #[inline]
    fn validate_scratch(&self, out_pdf: &[f64]) {
        let n = self.expert_prefix_cdfs.len();
        debug_assert_eq!(self.bit_probs.len(), n);
        debug_assert_eq!(self.ranges.len(), n);
        debug_assert_eq!(out_pdf.len(), BYTE_SYMBOLS);
        debug_assert!(logistic_stretch_mixer_active(n));
    }

    fn score_path(&mut self, symbol: u8, mode: LogisticByteScoringMode) -> f64 {
        let n: usize = self.expert_prefix_cdfs.len();
        self.ranges.fill(MsbPrefixRange::FULL);
        let mut prefix: u16 = 1;
        let mut probability: f64 = 1.0;
        for bit_idx in 0..8usize {
            for i in 0..n {
                self.bit_probs[i] =
                    self.ranges[i].prob_one(self.expert_prefix_cdfs[i].as_ref(), self.min_prob);
            }
            self.logistic.set_context(LogisticMixContext {
                history: self.history,
                bit_idx: bit_idx as u8,
                prefix,
                match_len_bucket: self.match_state.len_bucket,
                match_predicted_class: self.match_state.predicted_class,
            });
            let bit: bool = (symbol & (1u8 << (7 - bit_idx))) != 0;
            let p1: f64 = match mode {
                LogisticByteScoringMode::Adaptive => self.logistic.observe_bit_recording_undo(
                    self.bit_probs,
                    bit,
                    self.min_prob,
                    self.undo,
                ),
                LogisticByteScoringMode::Frozen => {
                    self.logistic.predict_bit(self.bit_probs, self.min_prob)
                }
            };
            probability *= if bit { p1 } else { 1.0 - p1 };
            for range in self.ranges.iter_mut() {
                range.observe(bit);
            }
            prefix = advanced_prefix_code(prefix, bit);
        }
        probability
    }
}

#[derive(Clone, Copy)]
enum LogisticByteScoringMode {
    Adaptive,
    Frozen,
}

/// Whether a logistic mixture should run its stretch-domain mixer.
///
/// A one-expert logistic mixture is defined to be identity of the sole expert on
/// every scoring and coding path (byte PDF, entropy rate, native bit-prefix, and
/// AC bitwise). Training [`LogisticMixCore`] in that case would make probabilities
/// path-dependent: byte updates would stay on the expert while bit/AC paths would
/// adapt unconstrained stretch weights. Multi-expert logistic mixtures always use
/// the mixer; empty mixtures never do.
#[inline]
pub(crate) fn logistic_stretch_mixer_active(expert_count: usize) -> bool {
    expert_count > 1
}

/// Smallest accepted logistic-learning rate.
///
/// This lower bound keeps a validated rate strictly positive while avoiding a
/// configuration whose updates round away in normal operating ranges.
pub(crate) const LOGISTIC_LEARNING_RATE_MIN: f64 = 1.0e-6;

/// Largest accepted logistic-learning rate.
pub(crate) const LOGISTIC_LEARNING_RATE_MAX: f64 = 1.0;

/// Validate the public learning-rate domain of the stretch-domain mixer.
///
/// The same predicate is used by typed specs and direct runtime construction,
/// so a labeled rate is never silently changed after validation.
pub(crate) fn validate_logistic_learning_rate(learning_rate: f64) -> Result<(), String> {
    if learning_rate.is_finite()
        && (LOGISTIC_LEARNING_RATE_MIN..=LOGISTIC_LEARNING_RATE_MAX).contains(&learning_rate)
    {
        Ok(())
    } else {
        Err("logistic mixture alpha (learning rate) must be finite and in [1e-6, 1.0]".to_string())
    }
}

/// Whether a logistic mixture may advertise a native MSB bit-prefix / recursive
/// AC bitwise path.
///
/// - Multi-expert: yes whenever there is at least one expert (bits are
///   synthesized from expert PDFs or native prefixes through the stretch mixer).
/// - One-expert: only if that expert itself exposes a native/recursive bit path
///   (identity of the expert; never the stretch mixer alone).
/// - Empty: never.
#[inline]
pub(crate) fn logistic_exposes_native_bit_path(
    expert_count: usize,
    sole_expert_has_native: bool,
) -> bool {
    if logistic_stretch_mixer_active(expert_count) {
        expert_count > 0
    } else {
        sole_expert_has_native
    }
}

/// Shared stretch-domain logistic mixer core used by runtime and compression predictors.
///
/// Invariant: predictions are made from bounded stretch-domain expert bit
/// probabilities and context-selected unconstrained weight rows. The global row
/// starts from the expert priors while all more specific rows start at zero, so
/// a fresh mixer is a prior-weighted geometric mixture rather than an already
/// overconfident stack of duplicated priors. Online updates are ordinary
/// logistic-regression gradient ascent on bit log-likelihood for each selected
/// row. We clip stretches and weights only to preserve finite arithmetic; the
/// row weights are not projected onto a simplex.
///
/// Callers must only train this core when [`logistic_stretch_mixer_active`] is
/// true for the surrounding expert count.
#[derive(Clone)]
pub(crate) struct LogisticMixCore {
    tables: Vec<Vec<f64>>,
    learning_rate: f64,
    expert_count: usize,
    context: LogisticMixContext,
    context_indices: [usize; Self::CONTEXTS],
    expert_stretches: Vec<f64>,
    last_prob_one: f64,
    last_valid: bool,
    #[cfg(test)]
    prediction_calls: usize,
}

#[derive(Clone, Default)]
pub(crate) struct LogisticMixUndo {
    entries: Vec<LogisticMixUndoEntry>,
    values: Vec<f64>,
    saved_context: LogisticMixContext,
    saved_context_indices: [usize; LogisticMixCore::CONTEXTS],
    saved_expert_stretches: Vec<f64>,
    saved_last_prob_one: f64,
    saved_last_valid: bool,
    active: bool,
}

#[derive(Clone, Copy)]
struct LogisticMixUndoEntry {
    table_idx: usize,
    start: usize,
    values_start: usize,
    values_len: usize,
}

impl LogisticMixCore {
    const CONTEXTS: usize = 6;
    const TABLE_SIZES: [usize; Self::CONTEXTS] = [1, 256, 4096, 4096, 1024, 2048];
    const STRETCH_CLIP: f64 = 16.0;
    const WEIGHT_CLIP: f64 = 16.0;

    /// Allocate a speculative journal sized for one eight-bit candidate path.
    pub(crate) fn new_undo(expert_count: usize) -> LogisticMixUndo {
        let row_updates: usize = 8 * Self::CONTEXTS;
        LogisticMixUndo {
            entries: Vec::with_capacity(row_updates),
            values: Vec::with_capacity(row_updates.saturating_mul(expert_count)),
            saved_expert_stretches: Vec::with_capacity(expert_count),
            ..LogisticMixUndo::default()
        }
    }

    pub(crate) fn new(expert_count: usize, prior_weights: &[f64], learning_rate: f64) -> Self {
        debug_assert_eq!(prior_weights.len(), expert_count);
        let mut tables = Vec::with_capacity(Self::CONTEXTS);
        for (ctx_idx, table_size) in Self::TABLE_SIZES.iter().enumerate() {
            let mut table = vec![0.0; table_size.saturating_mul(expert_count)];
            if ctx_idx == 0 && expert_count > 0 {
                let row = &mut table[..expert_count];
                for (dst, &prior) in row.iter_mut().zip(prior_weights.iter()) {
                    *dst = sanitize_logistic_weight(prior.max(0.0));
                }
            }
            tables.push(table);
        }

        // Construction is reachable only through validated specs or the direct
        // fallible `LogisticMixture` constructor. Keeping this assertion here
        // catches a crate-internal invariant violation without changing the
        // caller's labeled value in release builds.
        debug_assert!(
            validate_logistic_learning_rate(learning_rate).is_ok(),
            "validated logistic learning rate expected, got {learning_rate}"
        );
        let mut core = Self {
            tables,
            learning_rate,
            expert_count,
            context: LogisticMixContext::default(),
            context_indices: [0; Self::CONTEXTS],
            expert_stretches: vec![0.0; expert_count],
            last_prob_one: 0.5,
            last_valid: false,
            #[cfg(test)]
            prediction_calls: 0,
        };
        core.context_indices = core.compute_context_indices();
        core
    }

    pub(crate) fn reset_to_priors(&mut self, prior_weights: &[f64]) {
        debug_assert_eq!(prior_weights.len(), self.expert_count);
        for table in &mut self.tables {
            table.fill(0.0);
        }
        if self.expert_count > 0
            && let Some(global_table) = self.tables.first_mut()
        {
            for (dst, &prior) in global_table[..self.expert_count]
                .iter_mut()
                .zip(prior_weights.iter())
            {
                *dst = sanitize_logistic_weight(prior.max(0.0));
            }
        }
        self.context = LogisticMixContext::default();
        self.context_indices = self.compute_context_indices();
        self.expert_stretches.fill(0.0);
        self.last_prob_one = 0.5;
        self.last_valid = false;
        #[cfg(test)]
        {
            self.prediction_calls = 0;
        }
    }

    #[inline]
    pub(crate) fn set_context(&mut self, context: LogisticMixContext) {
        if self.context != context {
            self.context = context;
            self.context_indices = self.compute_context_indices();
            self.last_valid = false;
        }
    }

    pub(crate) fn predict_bit(&mut self, expert_prob_ones: &[f64], min_prob: f64) -> f64 {
        #[cfg(test)]
        {
            self.prediction_calls += 1;
        }
        debug_assert_eq!(expert_prob_ones.len(), self.expert_count);
        if self.expert_count == 0 {
            self.last_prob_one = 0.5;
            self.last_valid = true;
            return 0.5;
        }

        let floor = min_prob.clamp(1e-12, 0.49);
        for (dst, &p1) in self
            .expert_stretches
            .iter_mut()
            .zip(expert_prob_ones.iter())
        {
            *dst =
                stretch_probability_f64(p1, floor).clamp(-Self::STRETCH_CLIP, Self::STRETCH_CLIP);
        }

        let mut score = 0.0;
        for ctx_idx in 0..Self::CONTEXTS {
            let row = self.row(ctx_idx);
            for (&weight, &stretch) in row.iter().zip(self.expert_stretches.iter()) {
                score += weight * stretch;
            }
        }
        let p1 = squash(score).clamp(floor, 1.0 - floor);
        self.last_prob_one = p1;
        self.last_valid = true;
        p1
    }

    #[cfg(test)]
    pub(crate) fn prediction_calls(&self) -> usize {
        self.prediction_calls
    }

    pub(crate) fn update_last(&mut self, bit: bool) {
        if !self.last_valid || self.expert_count == 0 {
            return;
        }
        let target = if bit { 1.0 } else { 0.0 };
        let error = target - self.last_prob_one;
        let step = self.learning_rate * error / (Self::CONTEXTS as f64);
        let indices = self.context_indices;
        for (ctx_idx, &row_idx) in indices.iter().enumerate() {
            let expert_count = self.expert_count;
            let start = row_idx * expert_count;
            let end = start + expert_count;
            let row = &mut self.tables[ctx_idx][start..end];
            for (weight, &stretch) in row.iter_mut().zip(self.expert_stretches.iter()) {
                *weight = sanitize_logistic_weight(*weight + step * stretch);
            }
        }
        self.last_valid = false;
    }

    pub(crate) fn begin_undo(&self, undo: &mut LogisticMixUndo) {
        debug_assert!(!undo.active, "nested logistic undo scope is unsupported");
        undo.entries.clear();
        undo.values.clear();
        undo.saved_context = self.context;
        undo.saved_context_indices = self.context_indices;
        undo.saved_expert_stretches.clear();
        undo.saved_expert_stretches
            .extend_from_slice(&self.expert_stretches);
        undo.saved_last_prob_one = self.last_prob_one;
        undo.saved_last_valid = self.last_valid;
        undo.active = true;
    }

    pub(crate) fn restore_undo(&mut self, undo: &mut LogisticMixUndo) {
        debug_assert!(undo.active, "logistic undo restore without begin");
        for entry in undo.entries.iter().rev() {
            let table_end: usize = entry.start + entry.values_len;
            let values_end: usize = entry.values_start + entry.values_len;
            self.tables[entry.table_idx][entry.start..table_end]
                .copy_from_slice(&undo.values[entry.values_start..values_end]);
        }
        self.context = undo.saved_context;
        self.context_indices = undo.saved_context_indices;
        self.expert_stretches.clear();
        self.expert_stretches
            .extend_from_slice(&undo.saved_expert_stretches);
        self.last_prob_one = undo.saved_last_prob_one;
        self.last_valid = undo.saved_last_valid;
        undo.entries.clear();
        undo.values.clear();
        undo.active = false;
    }

    pub(crate) fn observe_bit_recording_undo(
        &mut self,
        expert_prob_ones: &[f64],
        bit: bool,
        min_prob: f64,
        undo: &mut LogisticMixUndo,
    ) -> f64 {
        let p1 = self.predict_bit(expert_prob_ones, min_prob);
        self.update_last_recording_undo(bit, undo);
        p1
    }

    fn update_last_recording_undo(&mut self, bit: bool, undo: &mut LogisticMixUndo) {
        if !self.last_valid || self.expert_count == 0 {
            return;
        }
        debug_assert!(undo.active, "logistic undo update without begin");
        let target = if bit { 1.0 } else { 0.0 };
        let error = target - self.last_prob_one;
        let step = self.learning_rate * error / (Self::CONTEXTS as f64);
        let indices = self.context_indices;
        for (ctx_idx, &row_idx) in indices.iter().enumerate() {
            let expert_count = self.expert_count;
            let start = row_idx * expert_count;
            let end = start + expert_count;
            let values_start: usize = undo.values.len();
            undo.values
                .extend_from_slice(&self.tables[ctx_idx][start..end]);
            undo.entries.push(LogisticMixUndoEntry {
                table_idx: ctx_idx,
                start,
                values_start,
                values_len: expert_count,
            });
            let row = &mut self.tables[ctx_idx][start..end];
            for (weight, &stretch) in row.iter_mut().zip(self.expert_stretches.iter()) {
                *weight = sanitize_logistic_weight(*weight + step * stretch);
            }
        }
        self.last_valid = false;
    }

    pub(crate) fn observe_bit(
        &mut self,
        expert_prob_ones: &[f64],
        bit: bool,
        min_prob: f64,
    ) -> f64 {
        let p1 = self.predict_bit(expert_prob_ones, min_prob);
        self.update_last(bit);
        p1
    }

    #[inline]
    fn row(&self, ctx_idx: usize) -> &[f64] {
        let row_idx = self.context_indices[ctx_idx];
        let start = row_idx * self.expert_count;
        &self.tables[ctx_idx][start..(start + self.expert_count)]
    }

    fn compute_context_indices(&self) -> [usize; Self::CONTEXTS] {
        let state = self.context.history;
        let prefix_lo = (self.context.prefix & 0xff) as u8;
        let prefix_hi = (self.context.prefix >> 8) as u8;
        if !state.has_history {
            return [
                0,
                0,
                0,
                0,
                hash_fields(
                    &[
                        self.context.match_len_bucket,
                        self.context.match_predicted_class,
                        self.context.bit_idx,
                        prefix_lo,
                        prefix_hi,
                    ],
                    Self::TABLE_SIZES[4],
                ),
                hash_fields(
                    &[self.context.bit_idx, prefix_lo, prefix_hi],
                    Self::TABLE_SIZES[5],
                ),
            ];
        }
        [
            0,
            state.prev1 as usize,
            hash_fields(&[state.prev2, state.prev1], Self::TABLE_SIZES[2]),
            hash_fields(
                &[
                    state.prev1_class,
                    state.prev2_class,
                    state.word_len_bucket,
                    state.prev_word_class,
                    state.bracket_bucket,
                    state.quote_flags,
                    state.utf8_left,
                    state.sentence_boundary as u8,
                    state.paragraph_break as u8,
                ],
                Self::TABLE_SIZES[3],
            ),
            hash_fields(
                &[
                    state.repeat_len_bucket,
                    state.copied_last_byte as u8,
                    state.run_len.min(63) as u8,
                    self.context.match_len_bucket,
                    self.context.match_predicted_class,
                ],
                Self::TABLE_SIZES[4],
            ),
            hash_fields(
                &[
                    self.context.bit_idx,
                    prefix_lo,
                    prefix_hi,
                    state.prev1,
                    state.prev2,
                    state.prev1_class,
                    state.prev2_class,
                    state.word_len_bucket,
                ],
                Self::TABLE_SIZES[5],
            ),
        ]
    }
}

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

    pub(crate) fn reset_to_priors(&mut self, prior_weights: &[f64]) {
        debug_assert_eq!(prior_weights.len(), self.expert_count);
        for table in &mut self.stage1_tables {
            table.fill(0.0);
        }
        if self.expert_count > 0
            && let Some(global_table) = self.stage1_tables.first_mut()
        {
            for (dst, &p) in global_table[..self.expert_count]
                .iter_mut()
                .zip(prior_weights.iter())
            {
                let p = if p.is_finite() { p.max(1e-12) } else { 1e-12 };
                *dst = p.ln();
            }
        }
        for row in &mut self.stage2_table {
            row.fill(0.0);
        }
        self.context = NeuralContextState::default();
        self.expert_probs.fill(0.0);
        self.stage1_mix.fill(0.0);
        self.stage1_probs.fill(0.0);
        self.stage2_mix.fill(0.0);
        self.expert_weights.fill(0.0);
        self.mix_prob = 1.0 / 256.0;
        self.context_mixtures_valid = false;
        self.evaluated = false;
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

#[inline]
fn sanitize_logistic_weight(w: f64) -> f64 {
    if w.is_finite() {
        w.clamp(-LogisticMixCore::WEIGHT_CLIP, LogisticMixCore::WEIGHT_CLIP)
    } else {
        0.0
    }
}

#[inline]
fn stretch_probability_f64(prob: f64, min_prob: f64) -> f64 {
    let p = if prob.is_finite() { prob } else { 0.5 };
    let p = p.clamp(min_prob, 1.0 - min_prob);
    (p / (1.0 - p)).ln()
}

#[inline]
fn squash(score: f64) -> f64 {
    let x = if score.is_finite() {
        score.clamp(-64.0, 64.0)
    } else {
        0.0
    };
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_prefix::{fill_prefix_cdf_from_pdf, zeroed_prefix_cdf_box};

    const TEST_FLOOR: f64 = 1e-12;

    // The production scratch representation deliberately boxes these large
    // fixed-size tables so moving the collection remains cheap and stable.
    #[allow(clippy::vec_box)]
    fn opposing_expert_cdfs() -> Vec<Box<BytePrefixCdf>> {
        let mut low = [0.0; BYTE_SYMBOLS];
        low[0] = 1.0;
        let mut high = [0.0; BYTE_SYMBOLS];
        high[BYTE_SYMBOLS - 1] = 1.0;

        let mut low_cdf = zeroed_prefix_cdf_box();
        fill_prefix_cdf_from_pdf(&mut low_cdf, &low, TEST_FLOOR);
        let mut high_cdf = zeroed_prefix_cdf_box();
        fill_prefix_cdf_from_pdf(&mut high_cdf, &high, TEST_FLOOR);
        vec![low_cdf, high_cdf]
    }

    fn frozen_reference_probability(
        core: &mut LogisticMixCore,
        cdfs: &[Box<BytePrefixCdf>],
        symbol: u8,
    ) -> f64 {
        let mut ranges = vec![MsbPrefixRange::FULL; cdfs.len()];
        let mut bit_probs = vec![0.5; cdfs.len()];
        let mut prefix: u16 = 1;
        let mut probability: f64 = 1.0;
        for bit_idx in 0..8usize {
            for i in 0..cdfs.len() {
                bit_probs[i] = ranges[i].prob_one(cdfs[i].as_ref(), TEST_FLOOR);
            }
            core.set_context(LogisticMixContext {
                bit_idx: bit_idx as u8,
                prefix,
                ..LogisticMixContext::default()
            });
            let bit: bool = (symbol & (1 << (7 - bit_idx))) != 0;
            let p1: f64 = core.predict_bit(&bit_probs, TEST_FLOOR);
            probability *= if bit { p1 } else { 1.0 - p1 };
            for range in &mut ranges {
                range.observe(bit);
            }
            prefix = advanced_prefix_code(prefix, bit);
        }
        probability
    }

    #[test]
    fn frozen_byte_scoring_never_fits_between_prefix_bits() {
        let cdfs = opposing_expert_cdfs();
        let mut core = LogisticMixCore::new(2, &[0.5, 0.5], 1.0);
        let original_tables = core.tables.clone();
        let mut reference_core = core.clone();
        let expected = frozen_reference_probability(&mut reference_core, &cdfs, u8::MAX);
        let mut bit_probs = vec![0.5; cdfs.len()];
        let mut ranges = vec![MsbPrefixRange::FULL; cdfs.len()];
        let mut undo = LogisticMixCore::new_undo(cdfs.len());
        let mut session = LogisticBytePdfSession {
            logistic: &mut core,
            expert_prefix_cdfs: &cdfs,
            history: NeuralContextState::default(),
            match_state: LogisticMatchState::default(),
            min_prob: TEST_FLOOR,
            bit_probs: &mut bit_probs,
            ranges: &mut ranges,
            undo: &mut undo,
        };

        let frozen = session.score_frozen(u8::MAX);
        let adaptive = session.score_adaptive(u8::MAX);

        assert!((frozen - expected).abs() < 1e-15);
        assert!(
            (adaptive - frozen).abs() > 1e-6,
            "test setup must distinguish adaptive within-byte fitting from frozen scoring"
        );
        assert_eq!(
            core.tables, original_tables,
            "speculative scoring must restore every fitted row"
        );
    }

    #[test]
    fn logistic_pdf_reuses_flat_undo_storage_and_preserves_mass() {
        let cdfs = opposing_expert_cdfs();
        let mut core = LogisticMixCore::new(2, &[0.5, 0.5], 0.03);
        let original_tables = core.tables.clone();
        let mut bit_probs = vec![0.5; cdfs.len()];
        let mut ranges = vec![MsbPrefixRange::FULL; cdfs.len()];
        let mut undo = LogisticMixCore::new_undo(cdfs.len());
        let mut pdf = [0.0; BYTE_SYMBOLS];

        {
            let mut session = LogisticBytePdfSession {
                logistic: &mut core,
                expert_prefix_cdfs: &cdfs,
                history: NeuralContextState::default(),
                match_state: LogisticMatchState::default(),
                min_prob: TEST_FLOOR,
                bit_probs: &mut bit_probs,
                ranges: &mut ranges,
                undo: &mut undo,
            };
            session.materialize_adaptive(&mut pdf);
        }
        let capacities = (
            undo.entries.capacity(),
            undo.values.capacity(),
            undo.saved_expert_stretches.capacity(),
        );
        {
            let mut session = LogisticBytePdfSession {
                logistic: &mut core,
                expert_prefix_cdfs: &cdfs,
                history: NeuralContextState::default(),
                match_state: LogisticMatchState::default(),
                min_prob: TEST_FLOOR,
                bit_probs: &mut bit_probs,
                ranges: &mut ranges,
                undo: &mut undo,
            };
            session.materialize_adaptive(&mut pdf);
        }

        assert_eq!(
            capacities,
            (
                undo.entries.capacity(),
                undo.values.capacity(),
                undo.saved_expert_stretches.capacity(),
            ),
            "a warmed PDF traversal must not grow any undo buffer"
        );
        assert!((pdf.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(pdf.iter().all(|probability| *probability > 0.0));
        assert_eq!(core.tables, original_tables);
    }
}
