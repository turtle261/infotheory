//! Stretch-domain logistic mixture runtime.
//!
//! The shared mixer law and byte-PDF materialization live in crate::neural_mix.
//! This module owns only the runtime adapter over ExpertState and its lifecycle.

use super::*;
use crate::byte_prefix::BytePrefixCdfScratch;
use crate::neural_mix::{
    LogisticBytePdfSession, LogisticMatchState, LogisticMixContext, LogisticMixCore,
    LogisticMixUndo, logistic_exposes_native_bit_path, logistic_stretch_mixer_active,
};

#[derive(Clone, Copy)]
enum LogisticPredictionMode {
    Adaptive,
    Frozen,
}

pub struct LogisticMixture {
    experts: Vec<ExpertState>,
    // A one-expert logistic mixture is exactly the expert, so allocating the
    // six context tables would both waste memory and obscure that invariant.
    logistic: Option<LogisticMixCore>,
    analyzer: TextContextAnalyzer,
    min_prob: f64,
    scratch_expert_logps: Vec<[f64; 256]>,
    scratch_prefix_cdfs: BytePrefixCdfScratch,
    scratch_ranges: Vec<MsbPrefixRange>,
    scratch_bit_probs: Vec<f64>,
    scratch_pdf: [f64; 256],
    speculative_undo: LogisticMixUndo,
    bitwise: MixtureBitPrefixState,
    active_match_state: LogisticMatchState,
    active_logp: f64,
    total_log_loss: f64,
}

impl Clone for LogisticMixture {
    fn clone(&self) -> Self {
        Self {
            experts: self.experts.clone(),
            logistic: self.logistic.clone(),
            analyzer: self.analyzer.clone(),
            min_prob: self.min_prob,
            // These buffers never carry observable state. Do not copy their
            // warmed allocations; the first scoring call sizes them for the
            // cloned expert set.
            scratch_expert_logps: Vec::new(),
            scratch_prefix_cdfs: Vec::new(),
            scratch_ranges: Vec::new(),
            scratch_bit_probs: Vec::new(),
            scratch_pdf: [0.0; 256],
            speculative_undo: LogisticMixUndo::default(),
            bitwise: self.bitwise.clone(),
            active_match_state: self.active_match_state,
            active_logp: self.active_logp,
            total_log_loss: self.total_log_loss,
        }
    }
}

impl LogisticMixture {
    /// Construct a per-bit stretch-domain logistic mixture.
    ///
    /// # Errors
    ///
    /// Returns an error unless `learning_rate` is finite and lies in the
    /// documented logistic-mixer domain. The accepted value is used exactly;
    /// it is never silently clamped or replaced.
    pub fn new(configs: &[ExpertConfig], learning_rate: f64) -> Result<Self, String> {
        crate::neural_mix::validate_logistic_learning_rate(learning_rate)?;
        let experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let logistic = if logistic_stretch_mixer_active(experts.len()) {
            let prior_weights = normalized_expert_prior_weights(&experts);
            Some(LogisticMixCore::new(
                experts.len(),
                &prior_weights,
                learning_rate,
            ))
        } else {
            None
        };
        Ok(Self {
            experts,
            logistic,
            analyzer: TextContextAnalyzer::new(),
            min_prob: DEFAULT_MIN_PROB,
            scratch_expert_logps: vec![[0.0; 256]; configs.len()],
            scratch_prefix_cdfs: Vec::new(),
            scratch_ranges: Vec::new(),
            scratch_bit_probs: vec![0.5; configs.len()],
            scratch_pdf: [0.0; 256],
            speculative_undo: LogisticMixCore::new_undo(configs.len()),
            bitwise: MixtureBitPrefixState::default(),
            active_match_state: LogisticMatchState::default(),
            active_logp: 0.0,
            total_log_loss: 0.0,
        })
    }

    #[inline]
    fn uses_stretch_mixer(&self) -> bool {
        debug_assert_eq!(
            self.logistic.is_some(),
            logistic_stretch_mixer_active(self.experts.len()),
            "LogisticMixture must allocate LogisticMixCore exactly for multi-expert mixtures"
        );
        self.logistic.is_some()
    }

    pub(super) fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if !self.uses_stretch_mixer() {
            return self.experts[0].log_prob(symbol);
        }
        self.score_stretch_symbol(symbol, LogisticPredictionMode::Adaptive)
            .ln()
    }

    pub(super) fn predict_log_prob_frozen(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if !self.uses_stretch_mixer() {
            return self.experts[0].log_prob_frozen(symbol);
        }
        self.score_stretch_symbol(symbol, LogisticPredictionMode::Frozen)
            .ln()
    }

    pub(super) fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        self.fill_log_probs_inner(out, LogisticPredictionMode::Adaptive);
    }

    pub(super) fn fill_log_probs_frozen(&mut self, out: &mut [f64; 256]) {
        self.fill_log_probs_inner(out, LogisticPredictionMode::Frozen);
    }

    fn fill_log_probs_inner(&mut self, out: &mut [f64; 256], mode: LogisticPredictionMode) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        if !self.uses_stretch_mixer() {
            match mode {
                LogisticPredictionMode::Adaptive => {
                    self.experts[0].predictor.fill_log_probs(out);
                }
                LogisticPredictionMode::Frozen => {
                    self.experts[0].fill_log_probs_frozen(out);
                }
            }
            return;
        }

        self.refresh_stretch_pdf(mode);
        for (slot, prob) in out.iter_mut().zip(self.scratch_pdf.iter()) {
            *slot = prob.ln();
        }
    }

    /// Return one multi-expert stretch-mixer leaf probability without paying
    /// for the other 255 candidate paths.
    fn score_stretch_symbol(&mut self, symbol: u8, mode: LogisticPredictionMode) -> f64 {
        debug_assert!(self.uses_stretch_mixer());
        self.prepare_expert_prefix_cdfs(mode);
        let history = self.analyzer.state();
        let match_state = aggregate_logistic_match_state(&mut self.experts);
        let logistic = self
            .logistic
            .as_mut()
            .expect("multi-expert LogisticMixture must own a LogisticMixCore");
        let mut session = LogisticBytePdfSession {
            logistic,
            expert_prefix_cdfs: &self.scratch_prefix_cdfs,
            history,
            match_state,
            min_prob: self.min_prob,
            bit_probs: &mut self.scratch_bit_probs,
            ranges: &mut self.scratch_ranges,
            undo: &mut self.speculative_undo,
        };
        match mode {
            LogisticPredictionMode::Adaptive => session.score_adaptive(symbol),
            LogisticPredictionMode::Frozen => session.score_frozen(symbol),
        }
    }

    /// Refresh the shared byte-PDF scratch buffer for a multi-expert stretch
    /// mixer, leaving both expert and mixer learning state unchanged.
    fn refresh_stretch_pdf(&mut self, mode: LogisticPredictionMode) {
        debug_assert!(self.uses_stretch_mixer());
        self.prepare_expert_prefix_cdfs(mode);

        let history = self.analyzer.state();
        let match_state = aggregate_logistic_match_state(&mut self.experts);
        let logistic = self
            .logistic
            .as_mut()
            .expect("multi-expert LogisticMixture must own a LogisticMixCore");
        let mut session = LogisticBytePdfSession {
            logistic,
            expert_prefix_cdfs: &self.scratch_prefix_cdfs,
            history,
            match_state,
            min_prob: self.min_prob,
            bit_probs: &mut self.scratch_bit_probs,
            ranges: &mut self.scratch_ranges,
            undo: &mut self.speculative_undo,
        };
        match mode {
            LogisticPredictionMode::Adaptive => session.materialize_adaptive(&mut self.scratch_pdf),
            LogisticPredictionMode::Frozen => session.materialize_frozen(&mut self.scratch_pdf),
        }
    }

    fn prepare_expert_prefix_cdfs(&mut self, mode: LogisticPredictionMode) {
        let n = self.experts.len();
        self.scratch_expert_logps.resize(n, [0.0; 256]);
        self.scratch_prefix_cdfs
            .resize_with(n, zeroed_prefix_cdf_box);
        self.scratch_ranges.resize(n, MsbPrefixRange::FULL);
        self.scratch_bit_probs.resize(n, 0.5);

        for i in 0..n {
            match mode {
                LogisticPredictionMode::Adaptive => self.experts[i]
                    .predictor
                    .fill_log_probs(&mut self.scratch_expert_logps[i]),
                LogisticPredictionMode::Frozen => {
                    self.experts[i].fill_log_probs_frozen(&mut self.scratch_expert_logps[i])
                }
            }
            fill_prefix_cdf_from_log_probs(
                &mut self.scratch_prefix_cdfs[i],
                &self.scratch_expert_logps[i],
                self.min_prob,
            );
        }
    }

    /// Log-probability (natural log) of the logistic mixture for `symbol`, then update.
    ///
    /// # Errors
    ///
    /// Prefix-capable custom experts can report a stepping error. Such errors
    /// are surfaced to the caller rather than converted into a panic. As with
    /// the underlying `OnlineBytePredictor` prefix contract, an error after
    /// an observed bit may leave that custom expert partially advanced; discard
    /// this mixture before continuing.
    pub fn step(&mut self, symbol: u8) -> Result<f64, String> {
        if self.experts.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }
        if !self.uses_stretch_mixer() {
            let expert = &mut self.experts[0];
            let logp = expert.log_prob_update(symbol);
            expert.cum_log_loss -= logp;
            self.total_log_loss -= logp;
            self.analyzer.update(symbol);
            return Ok(logp);
        }

        if !self.begin_prefix_step()? {
            return Err(
                "logistic mixture could not prepare a multi-expert byte-prefix step".to_string(),
            );
        }
        for bit_idx in 0..8usize {
            let bit = (symbol & (1u8 << (7 - bit_idx))) != 0;
            self.observe_known_prefix_bit(bit_idx, bit)?;
        }
        self.finish_prefix_step(symbol)?;
        Ok(self.active_logp)
    }

    /// Step the built-in `RateBackend` adapter.
    ///
    /// The public constructor validates its configuration, and built-in
    /// `RateBackendPredictor` experts obey their infallible byte-step contract.
    /// Therefore an error here can only indicate a crate-internal violation of
    /// the native-prefix protocol, not a recoverable user configuration error.
    pub(super) fn step_rate_backend(&mut self, symbol: u8) -> f64 {
        self.step(symbol).unwrap_or_else(|err| {
            panic!(
                "built-in logistic RateBackend prefix invariant violated while stepping byte {symbol}: {err}"
            )
        })
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    pub(super) fn begin_prefix_step(&mut self) -> Result<bool, String> {
        if self.experts.is_empty() {
            return Ok(false);
        }
        self.active_logp = 0.0;
        if !self.uses_stretch_mixer() {
            // Identity path: unit-weight convex bit mix over the sole expert so a
            // native bit session matches byte `step` / PDF scoring and never trains
            // LogisticMixCore.
            self.active_match_state = LogisticMatchState::default();
            let unit_weights: [f64; 1] = [1.0];
            return self.bitwise.begin(
                &mut self.experts,
                MixturePrefixPreparation::NativeIfAvailable {
                    weights: &unit_weights,
                },
            );
        }
        self.active_match_state = aggregate_logistic_match_state(&mut self.experts);
        self.bitwise
            .begin(&mut self.experts, MixturePrefixPreparation::AlwaysPrepare)
    }

    pub(super) fn prefix_prob_one(&mut self, bit_idx: usize) -> Result<f64, String> {
        if !self.uses_stretch_mixer() {
            return self.bitwise.prob_one(&mut self.experts, bit_idx);
        }
        let logistic = self
            .logistic
            .as_mut()
            .expect("multi-expert LogisticMixture must own a LogisticMixCore");
        self.bitwise.logistic_prob_one(
            &mut self.experts,
            logistic,
            &self.analyzer,
            self.active_match_state,
            bit_idx,
        )
    }

    pub(super) fn observe_prefix_bit(&mut self, bit_idx: usize, bit: bool) -> Result<(), String> {
        if !self.uses_stretch_mixer() {
            self.bitwise.observe(&mut self.experts, bit_idx, bit)?;
            return Ok(());
        }
        let logistic = self
            .logistic
            .as_mut()
            .expect("multi-expert LogisticMixture must own a LogisticMixCore");
        let p1 = self.bitwise.logistic_observe(
            &mut self.experts,
            logistic,
            &self.analyzer,
            self.active_match_state,
            bit_idx,
            bit,
        )?;
        self.active_logp += if bit {
            clamp_unit_prob(p1, self.min_prob).ln()
        } else {
            clamp_unit_prob(1.0 - p1, self.min_prob).ln()
        };
        Ok(())
    }

    pub(super) fn observe_known_prefix_bit(
        &mut self,
        bit_idx: usize,
        bit: bool,
    ) -> Result<f64, String> {
        if !self.uses_stretch_mixer() {
            return self.bitwise.observe_known(&mut self.experts, bit_idx, bit);
        }
        let logistic = self
            .logistic
            .as_mut()
            .expect("multi-expert LogisticMixture must own a LogisticMixCore");
        let p1 = self.bitwise.logistic_observe_known(
            &mut self.experts,
            logistic,
            &self.analyzer,
            self.active_match_state,
            bit_idx,
            bit,
        )?;
        self.active_logp += if bit {
            clamp_unit_prob(p1, self.min_prob).ln()
        } else {
            clamp_unit_prob(1.0 - p1, self.min_prob).ln()
        };
        Ok(p1)
    }

    pub(super) fn finish_prefix_step(&mut self, symbol: u8) -> Result<(), String> {
        self.bitwise.finish_adaptive(&mut self.experts, symbol)?;
        for idx in 0..self.experts.len() {
            self.experts[idx].cum_log_loss -= self.bitwise.logps[idx];
        }
        if !self.uses_stretch_mixer() {
            // Sole-expert native path does not accumulate mixer log-loss per bit;
            // the finished expert bit-product is the byte log-probability.
            self.active_logp = self.bitwise.logps.first().copied().unwrap_or(0.0);
            self.total_log_loss -= self.active_logp;
            self.analyzer.update(symbol);
            return Ok(());
        }
        self.total_log_loss -= self.active_logp;
        self.analyzer.update(symbol);
        self.set_idle_logistic_context();
        Ok(())
    }

    #[inline]
    fn clear_stream_state(&mut self) {
        self.analyzer = TextContextAnalyzer::new();
        self.bitwise.reset_inactive();
        self.active_match_state = LogisticMatchState::default();
        self.active_logp = 0.0;
        self.total_log_loss = 0.0;
        if let Some(logistic) = self.logistic.as_mut() {
            logistic.set_context(LogisticMixContext::default());
        }
    }

    pub(super) fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        reset_expert_frozen_stream(&mut self.experts, total_symbols)?;
        self.clear_stream_state();
        Ok(())
    }

    pub(super) fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_fresh_stream(&mut self.experts, total_symbols)?;
        let prior = reset_experts_to_priors(&mut self.experts);
        if let Some(logistic) = self.logistic.as_mut() {
            logistic.reset_to_priors(&prior);
        }
        self.clear_stream_state();
        Ok(())
    }

    pub(super) fn update_frozen(&mut self, symbol: u8) {
        for expert in &mut self.experts {
            expert.update_frozen(symbol);
        }
        self.analyzer.update(symbol);
        self.set_idle_logistic_context();
    }

    pub(super) fn logistic_match_state(&mut self) -> LogisticMatchState {
        aggregate_logistic_match_state(&mut self.experts)
    }

    fn set_idle_logistic_context(&mut self) {
        if let Some(logistic) = self.logistic.as_mut() {
            logistic.set_context(LogisticMixContext {
                history: self.analyzer.state(),
                bit_idx: 0,
                prefix: 1,
                match_len_bucket: 0,
                match_predicted_class: 0,
            });
        }
    }
}

#[derive(Clone)]
#[doc(hidden)]
// Prediction scratch is deliberately excluded: every scoring path fully
// overwrites it, so copying per-expert 256-way rows would add checkpoint cost
// without restoring any observable model state.
pub struct LogisticMixtureCheckpoint {
    experts: Vec<ExpertStateCheckpoint>,
    logistic: Option<LogisticMixCore>,
    analyzer: TextContextAnalyzer,
    bitwise: MixtureBitPrefixState,
    active_match_state: LogisticMatchState,
    active_logp: f64,
    total_log_loss: f64,
}

pub(super) struct LogisticMixtureLifecycleCheckpoint {
    experts: Vec<ExpertStateLifecycleCheckpoint>,
    logistic: Option<LogisticMixCore>,
    analyzer: TextContextAnalyzer,
    bitwise: MixtureBitPrefixState,
    active_match_state: LogisticMatchState,
    active_logp: f64,
    total_log_loss: f64,
}

impl LogisticMixture {
    pub(super) fn checkpoint(&mut self) -> Option<LogisticMixtureCheckpoint> {
        Some(LogisticMixtureCheckpoint {
            experts: checkpoint_experts(&mut self.experts)?,
            logistic: self.logistic.clone(),
            analyzer: self.analyzer.clone(),
            bitwise: self.bitwise.clone(),
            active_match_state: self.active_match_state,
            active_logp: self.active_logp,
            total_log_loss: self.total_log_loss,
        })
    }

    pub(super) fn restore_checkpoint(&mut self, checkpoint: &LogisticMixtureCheckpoint) {
        restore_experts(&mut self.experts, &checkpoint.experts);
        self.logistic = checkpoint.logistic.clone();
        self.analyzer = checkpoint.analyzer.clone();
        self.bitwise = checkpoint.bitwise.clone();
        self.active_match_state = checkpoint.active_match_state;
        self.active_logp = checkpoint.active_logp;
        self.total_log_loss = checkpoint.total_log_loss;
    }

    pub(super) fn discard_checkpoint(&mut self, checkpoint: LogisticMixtureCheckpoint) {
        discard_expert_checkpoints(&mut self.experts, checkpoint.experts);
    }

    pub(super) fn lifecycle_checkpoint(
        &mut self,
        expert_op: ExpertLifecycleOp,
    ) -> LogisticMixtureLifecycleCheckpoint {
        LogisticMixtureLifecycleCheckpoint {
            experts: lifecycle_checkpoint_experts(&mut self.experts, expert_op),
            logistic: self.logistic.clone(),
            analyzer: self.analyzer.clone(),
            bitwise: self.bitwise.clone(),
            active_match_state: self.active_match_state,
            active_logp: self.active_logp,
            total_log_loss: self.total_log_loss,
        }
    }

    pub(super) fn restore_lifecycle_checkpoint(
        &mut self,
        expert_op: ExpertLifecycleOp,
        checkpoint: LogisticMixtureLifecycleCheckpoint,
    ) {
        restore_lifecycle_experts(&mut self.experts, checkpoint.experts, expert_op);
        self.logistic = checkpoint.logistic;
        self.analyzer = checkpoint.analyzer;
        self.bitwise = checkpoint.bitwise;
        self.active_match_state = checkpoint.active_match_state;
        self.active_logp = checkpoint.active_logp;
        self.total_log_loss = checkpoint.total_log_loss;
    }

    pub(super) fn discard_lifecycle_checkpoint(
        &mut self,
        expert_op: ExpertLifecycleOp,
        checkpoint: LogisticMixtureLifecycleCheckpoint,
    ) {
        discard_lifecycle_experts(&mut self.experts, checkpoint.experts, expert_op);
    }

    pub(super) fn clear_checkpoints_if_supported(&mut self) {
        clear_expert_checkpoints(&mut self.experts);
    }

    pub(super) fn supports_frozen_reset(&self) -> bool {
        experts_support_frozen_reset(&self.experts)
    }

    pub(super) fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_stream(&mut self.experts, total_symbols)
    }

    pub(super) fn finish_stream(&mut self) -> Result<(), String> {
        finish_expert_stream(&mut self.experts)
    }

    pub(super) fn has_native_msb_byte_prefix(&self) -> bool {
        logistic_exposes_native_bit_path(
            self.experts.len(),
            experts_have_native_msb_byte_prefix(&self.experts),
        )
    }

    pub(super) fn supports_empty_native_msb_prefix_abort(&self) -> bool {
        logistic_exposes_native_bit_path(
            self.experts.len(),
            experts_have_abortable_native_msb_byte_prefix(&self.experts),
        )
    }

    pub(super) fn abort_empty_prefix(&mut self) -> Result<bool, String> {
        self.bitwise.abort_empty(&mut self.experts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct FavoredBytePredictor {
        favored: u8,
    }

    impl FavoredBytePredictor {
        fn fill(&self, out: &mut [f64; 256]) {
            let tail_probability: f64 = 1e-15;
            let favored_probability: f64 = 1.0 - 255.0 * tail_probability;
            out.fill(tail_probability.ln());
            out[self.favored as usize] = favored_probability.ln();
        }
    }

    impl OnlineBytePredictor for FavoredBytePredictor {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            let mut row = [0.0; 256];
            self.fill(&mut row);
            row[symbol as usize]
        }

        fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
            self.fill(out);
        }

        fn fill_log_probs_frozen(&mut self, out: &mut [f64; 256]) {
            self.fill(out);
        }

        fn update(&mut self, _symbol: u8) {}
    }

    fn opposing_configs() -> [ExpertConfig; 2] {
        [
            ExpertConfig::uniform("zero", || Box::new(FavoredBytePredictor { favored: 0 })),
            ExpertConfig::uniform("ff", || Box::new(FavoredBytePredictor { favored: u8::MAX })),
        ]
    }

    #[test]
    fn point_logistic_score_traverses_only_the_requested_byte() {
        let mut mixture = LogisticMixture::new(&opposing_configs(), 0.03).expect("mixture");
        let before = mixture
            .logistic
            .as_ref()
            .expect("multi-expert core")
            .prediction_calls();
        let _ = mixture.predict_log_prob(0x5a);
        let after_point = mixture
            .logistic
            .as_ref()
            .expect("multi-expert core")
            .prediction_calls();
        assert_eq!(after_point - before, 8);

        let mut row = [0.0; 256];
        mixture.fill_log_probs(&mut row);
        let after_pdf = mixture
            .logistic
            .as_ref()
            .expect("multi-expert core")
            .prediction_calls();
        assert_eq!(after_pdf - after_point, 256 * 8);
    }

    #[test]
    fn logistic_leaf_mass_and_frozen_scoring_contracts_are_preserved() {
        let mut mixture = LogisticMixture::new(&opposing_configs(), 1.0).expect("mixture");
        for _ in 0..64 {
            mixture.step(u8::MAX).expect("train mixer");
        }

        let target: u8 = 0;
        let point_logp: f64 = mixture.predict_log_prob(target);
        let mut stepped = mixture.clone();
        let step_logp: f64 = stepped.step(target).expect("step target");
        assert!(
            (point_logp - step_logp).abs() < 1e-12,
            "target-only adaptive score must match the native update path"
        );

        let mut adaptive_row = [0.0; 256];
        mixture.fill_log_probs(&mut adaptive_row);
        assert!((adaptive_row[target as usize] - point_logp).abs() < 1e-12);
        let adaptive_mass: f64 = adaptive_row.iter().map(|logp| logp.exp()).sum();
        assert!((adaptive_mass - 1.0).abs() < 1e-12);
        assert!(
            adaptive_row[target as usize].exp() < DEFAULT_MIN_PROB,
            "the byte adapter must not re-floor an already coherent low-mass leaf"
        );

        let frozen_point_logp: f64 = mixture.predict_log_prob_frozen(target);
        let mut frozen_row = [0.0; 256];
        mixture.fill_log_probs_frozen(&mut frozen_row);
        assert!((frozen_row[target as usize] - frozen_point_logp).abs() < 1e-12);
        let frozen_mass: f64 = frozen_row.iter().map(|logp| logp.exp()).sum();
        assert!((frozen_mass - 1.0).abs() < 1e-12);
        assert!(
            (frozen_point_logp - point_logp).abs() > 1e-6,
            "test setup must expose adaptive within-byte fitting in the non-frozen score"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn target_score_matches_native_prefix_step_with_native_experts() {
        let configs = [ExpertConfig::ctw("ctw-6", 6), ExpertConfig::ctw("ctw-8", 8)];
        let mut mixture = LogisticMixture::new(&configs, 0.03).expect("mixture");
        for &symbol in b"native expert point and prefix parity" {
            let point_logp: f64 = mixture.predict_log_prob(symbol);
            let step_logp: f64 = mixture.step(symbol).expect("step");
            assert!(
                (point_logp - step_logp).abs() < 1e-9,
                "symbol={symbol} point={point_logp} step={step_logp}"
            );
        }
    }
}
