//! Shared MSB-first byte-prefix state for mixture experts.
//!
//! It centralizes native-prefix setup, byte-PDF fallback, and the two distinct
//! preparation policies used by simplex and stretch-domain mixtures.

use super::*;
use crate::neural_mix::{LogisticMatchState, LogisticMixContext, LogisticMixCore};

#[derive(Clone, Default)]
#[doc(hidden)]
pub struct BytePrefixStepState {
    pub(super) kind: BytePrefixStepStateKind,
}

#[derive(Clone)]
pub(super) enum BytePrefixStepStateKind {
    Native,
    PdfPrefix {
        cdf: Box<BytePrefixCdf>,
        range: MsbPrefixRange,
    },
}

impl Default for BytePrefixStepStateKind {
    fn default() -> Self {
        Self::PdfPrefix {
            cdf: zeroed_prefix_cdf_box(),
            range: MsbPrefixRange::FULL,
        }
    }
}

impl BytePrefixStepState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(super) fn is_native(&self) -> bool {
        matches!(self.kind, BytePrefixStepStateKind::Native)
    }

    pub(super) fn prepare(
        &mut self,
        predictor: &mut dyn OnlineBytePredictor,
    ) -> Result<(), String> {
        if predictor.has_abortable_native_msb_byte_prefix()
            && predictor.begin_native_msb_byte_prefix()?
        {
            self.kind = BytePrefixStepStateKind::Native;
            return Ok(());
        }

        self.prepare_pdf_prefix(predictor)
    }

    fn prepare_pdf_prefix(
        &mut self,
        predictor: &mut dyn OnlineBytePredictor,
    ) -> Result<(), String> {
        let mut cdf = match std::mem::take(&mut self.kind) {
            BytePrefixStepStateKind::PdfPrefix { cdf, .. } => cdf,
            BytePrefixStepStateKind::Native => zeroed_prefix_cdf_box(),
        };
        let mut logps = [0.0f64; 256];
        predictor.fill_log_probs(&mut logps);
        fill_prefix_cdf_from_log_probs(&mut cdf, &logps, DEFAULT_MIN_PROB);
        self.kind = BytePrefixStepStateKind::PdfPrefix {
            cdf,
            range: MsbPrefixRange::FULL,
        };
        Ok(())
    }

    pub(super) fn prob_one(
        &mut self,
        predictor: &mut dyn OnlineBytePredictor,
        bit_idx: usize,
    ) -> Result<f64, String> {
        match &mut self.kind {
            BytePrefixStepStateKind::Native => predictor.native_msb_prefix_prob_one(bit_idx),
            BytePrefixStepStateKind::PdfPrefix { cdf, range } => {
                Ok(range.prob_one(cdf.as_ref(), DEFAULT_MIN_PROB))
            }
        }
    }

    pub(super) fn observe(
        &mut self,
        predictor: &mut dyn OnlineBytePredictor,
        bit_idx: usize,
        bit: bool,
    ) -> Result<(), String> {
        match &mut self.kind {
            BytePrefixStepStateKind::Native => {
                predictor.observe_native_msb_prefix_bit(bit_idx, bit)
            }
            BytePrefixStepStateKind::PdfPrefix { range, .. } => {
                range.observe(bit);
                Ok(())
            }
        }
    }

    pub(super) fn observe_known(
        &mut self,
        predictor: &mut dyn OnlineBytePredictor,
        bit_idx: usize,
        bit: bool,
    ) -> Result<f64, String> {
        match &mut self.kind {
            BytePrefixStepStateKind::Native => {
                predictor.observe_known_native_msb_prefix_bit(bit_idx, bit)
            }
            BytePrefixStepStateKind::PdfPrefix { cdf, range } => {
                let p1: f64 = range.prob_one(cdf.as_ref(), DEFAULT_MIN_PROB);
                range.observe(bit);
                Ok(p1)
            }
        }
    }

    pub(super) fn abort_empty(
        &mut self,
        predictor: &mut dyn OnlineBytePredictor,
    ) -> Result<(), String> {
        match &mut self.kind {
            BytePrefixStepStateKind::Native => {
                let aborted: bool = predictor.abort_empty_native_msb_byte_prefix()?;
                if !aborted {
                    return Err(
                        "native MSB-first byte-prefix abort reported no active empty prefix"
                            .to_string(),
                    );
                }
                *self = Self::new();
                Ok(())
            }
            BytePrefixStepStateKind::PdfPrefix { .. } => {
                *self = Self::new();
                Ok(())
            }
        }
    }

    pub(super) fn finish(
        &mut self,
        predictor: &mut dyn OnlineBytePredictor,
        symbol: u8,
    ) -> Result<(), String> {
        match &mut self.kind {
            BytePrefixStepStateKind::Native => predictor.finish_native_msb_byte_prefix(symbol),
            BytePrefixStepStateKind::PdfPrefix { .. } => {
                predictor.update(symbol);
                Ok(())
            }
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct MixtureBitPrefixState {
    pub(super) states: Vec<BytePrefixStepState>,
    pub(super) weights: Vec<f64>,
    pub(super) likelihoods: Vec<f64>,
    pub(super) bit_probs: Vec<f64>,
    pub(super) logps: Vec<f64>,
    pub(super) active: bool,
    pub(super) primed_bit_idx: Option<usize>,
    pub(super) expected_bit_idx: usize,
    pub(super) prefix: u16,
}

/// Select how a mixture prepares an MSB-first byte-prefix session.
///
/// Ordinary convex-style mixtures only enter a session when an expert offers
/// an abortable native prefix path; otherwise their caller uses a byte-PDF
/// fallback. The stretch-domain logistic mixer instead needs every expert's
/// conditional bit probability, so it always prepares each expert (native
/// where available, byte-PDF otherwise).
pub(super) enum MixturePrefixPreparation<'a> {
    /// Use normalized convex weights and require at least one native expert.
    NativeIfAvailable { weights: &'a [f64] },
    /// Prepare all experts for a non-convex per-bit mixer.
    AlwaysPrepare,
}

impl MixtureBitPrefixState {
    pub(super) fn reset_inactive(&mut self) {
        self.active = false;
        self.primed_bit_idx = None;
        self.expected_bit_idx = 0;
        self.prefix = 1;
    }

    fn validate_bit_idx(&self, bit_idx: usize) -> Result<(), String> {
        if bit_idx >= 8 {
            return Err(format!(
                "native MSB-first byte-prefix bit index {bit_idx} is out of range; expected 0..8"
            ));
        }
        if bit_idx != self.expected_bit_idx {
            return Err(format!(
                "native MSB-first byte-prefix bit index {bit_idx} violated sequential stepping; expected {}",
                self.expected_bit_idx
            ));
        }
        Ok(())
    }

    pub(super) fn begin(
        &mut self,
        experts: &mut [ExpertState],
        preparation: MixturePrefixPreparation<'_>,
    ) -> Result<bool, String> {
        let (always_prepare, weights) = match preparation {
            MixturePrefixPreparation::NativeIfAvailable { weights } => (false, Some(weights)),
            MixturePrefixPreparation::AlwaysPrepare => (true, None),
        };
        if !always_prepare
            && !experts
                .iter()
                .any(|expert| expert.predictor.has_abortable_native_msb_byte_prefix())
        {
            self.reset_inactive();
            return Ok(false);
        }

        let n: usize = experts.len();
        if n == 0 {
            self.reset_inactive();
            return Ok(false);
        }
        self.states.resize_with(n, BytePrefixStepState::new);
        self.weights.clear();
        if let Some(weights) = weights {
            self.weights.extend(weights.iter().copied());
            normalize_simplex_weights(&mut self.weights);
        }
        self.likelihoods.resize(n, 1.0);
        self.likelihoods.fill(1.0);
        self.bit_probs.resize(n, 0.5);
        self.logps.resize(n, 0.0);
        self.primed_bit_idx = None;
        self.expected_bit_idx = 0;
        self.prefix = 1;
        let mut native_started: bool = false;
        for idx in 0..n {
            let state = &mut self.states[idx];
            let expert = &mut experts[idx];
            if let Err(err) = state.prepare(expert.predictor.as_mut()) {
                let mut rollback_error: Option<String> = None;
                for rollback_idx in (0..idx).rev() {
                    if let Err(abort_err) = self.states[rollback_idx]
                        .abort_empty(experts[rollback_idx].predictor.as_mut())
                        && rollback_error.is_none()
                    {
                        rollback_error = Some(format!(
                            "expert {rollback_idx} empty-prefix rollback failed: {abort_err}"
                        ));
                    }
                }
                self.reset_inactive();
                return Err(match rollback_error {
                    Some(rollback_err) => {
                        format!("{err}; mixture prefix setup rollback also failed: {rollback_err}")
                    }
                    None => err,
                });
            }
            native_started |= state.is_native();
        }
        if !always_prepare && !native_started {
            self.reset_inactive();
            return Ok(false);
        }
        self.active = true;
        Ok(true)
    }

    pub(super) fn abort_empty(&mut self, experts: &mut [ExpertState]) -> Result<bool, String> {
        if !self.active {
            return Ok(false);
        }
        if self.expected_bit_idx != 0 {
            return Err(format!(
                "native MSB-first byte-prefix abort requires zero observed bits, got {}",
                self.expected_bit_idx
            ));
        }
        let mut rollback_error: Option<String> = None;
        for (index, (state, expert)) in self.states.iter_mut().zip(experts.iter_mut()).enumerate() {
            if let Err(err) = state.abort_empty(expert.predictor.as_mut())
                && rollback_error.is_none()
            {
                rollback_error = Some(format!("expert {index} empty-prefix abort failed: {err}"));
            }
        }
        self.reset_inactive();
        match rollback_error {
            Some(err) => Err(err),
            None => Ok(true),
        }
    }

    fn prime_bit_probs_if_needed(
        &mut self,
        experts: &mut [ExpertState],
        bit_idx: usize,
    ) -> Result<(), String> {
        self.validate_bit_idx(bit_idx)?;
        if self.primed_bit_idx == Some(bit_idx) {
            return Ok(());
        }
        // Index form required for coordinated access to per-expert state + scratch buffers
        // (same rationale as the allows in prob_one/observe below).
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let p1: f64 = self.states[idx].prob_one(experts[idx].predictor.as_mut(), bit_idx)?;
            self.bit_probs[idx] = p1;
        }
        self.primed_bit_idx = Some(bit_idx);
        Ok(())
    }

    pub(super) fn prob_one(
        &mut self,
        experts: &mut [ExpertState],
        bit_idx: usize,
    ) -> Result<f64, String> {
        debug_assert!(self.active);
        self.prime_bit_probs_if_needed(experts, bit_idx)?;
        let mut denom: f64 = 0.0;
        let mut numer: f64 = 0.0;
        // Index form required for parallel mutable access to multiple scratch buffers
        // alongside experts; iterators would require zip + tuple mutation which is less clear here.
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let p1: f64 = self.bit_probs[idx];
            let weighted_prefix: f64 = self.weights[idx] * self.likelihoods[idx];
            denom += weighted_prefix;
            numer += weighted_prefix * p1;
        }
        Ok(if denom.is_finite() && denom > 0.0 {
            (numer / denom).clamp(DEFAULT_MIN_PROB, 1.0 - DEFAULT_MIN_PROB)
        } else {
            // Invariant failure (introduced in bitwiseness bit-prefix state; tightened):
            // non-positive/NaN denom means internal expert weighting or priming
            // produced invalid state. Panic with context per AGENTS (contract violation).
            panic!(
                "MixtureBitPrefixState::prob_one: invalid weighted denom (must be finite > 0); \
                 this indicates a bug in prime_bit_probs_if_needed or expert likelihoods"
            )
        })
    }

    pub(super) fn observe(
        &mut self,
        experts: &mut [ExpertState],
        bit_idx: usize,
        bit: bool,
    ) -> Result<(), String> {
        debug_assert!(self.active);
        self.prime_bit_probs_if_needed(experts, bit_idx)?;
        // Index form clearest for coordinated mutation of likelihoods/states + experts[idx].
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let p1: f64 = self.bit_probs[idx];
            let pb: f64 = if bit { p1 } else { 1.0 - p1 };
            self.likelihoods[idx] = (self.likelihoods[idx] * pb).max(DEFAULT_MIN_PROB);
            self.states[idx].observe(experts[idx].predictor.as_mut(), bit_idx, bit)?;
        }
        self.expected_bit_idx += 1;
        self.prefix = advanced_prefix_code(self.prefix, bit);
        self.primed_bit_idx = None;
        Ok(())
    }

    pub(super) fn observe_known(
        &mut self,
        experts: &mut [ExpertState],
        bit_idx: usize,
        bit: bool,
    ) -> Result<f64, String> {
        debug_assert!(self.active);
        self.validate_bit_idx(bit_idx)?;
        let use_primed: bool = self.primed_bit_idx == Some(bit_idx);
        let mut denom: f64 = 0.0;
        let mut numer: f64 = 0.0;

        // Index form is clearest for coordinated mutation of likelihoods,
        // cached bit probabilities, per-expert prefix state, and experts[idx].
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let p1: f64 = if use_primed {
                let p1: f64 = self.bit_probs[idx];
                self.states[idx].observe(experts[idx].predictor.as_mut(), bit_idx, bit)?;
                p1
            } else {
                self.states[idx].observe_known(experts[idx].predictor.as_mut(), bit_idx, bit)?
            };
            let weighted_prefix: f64 = self.weights[idx] * self.likelihoods[idx];
            denom += weighted_prefix;
            numer += weighted_prefix * p1;
            let pb: f64 = if bit { p1 } else { 1.0 - p1 };
            self.likelihoods[idx] = (self.likelihoods[idx] * pb).max(DEFAULT_MIN_PROB);
        }
        self.expected_bit_idx += 1;
        self.prefix = advanced_prefix_code(self.prefix, bit);
        self.primed_bit_idx = None;

        Ok(if denom.is_finite() && denom > 0.0 {
            (numer / denom).clamp(DEFAULT_MIN_PROB, 1.0 - DEFAULT_MIN_PROB)
        } else {
            panic!(
                "MixtureBitPrefixState::observe_known: invalid weighted denom (must be finite > 0); \
                 this indicates a bug in known-bit prefix scoring or expert likelihoods"
            )
        })
    }

    pub(super) fn finish_adaptive(
        &mut self,
        experts: &mut [ExpertState],
        symbol: u8,
    ) -> Result<(), String> {
        debug_assert!(self.active);
        if self.expected_bit_idx != 8 {
            return Err(format!(
                "native MSB-first byte-prefix finish requires 8 observed bits, got {}",
                self.expected_bit_idx
            ));
        }
        // Index form clearest for coordinated mutation of likelihoods/logps/states + experts[idx].
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let lp: f64 = self.likelihoods[idx].max(DEFAULT_MIN_PROB).ln();
            self.logps[idx] = lp;
            self.states[idx].finish(experts[idx].predictor.as_mut(), symbol)?;
        }
        self.reset_inactive();
        Ok(())
    }

    fn logistic_context(
        &self,
        analyzer: &TextContextAnalyzer,
        match_state: LogisticMatchState,
        bit_idx: usize,
    ) -> LogisticMixContext {
        LogisticMixContext {
            history: analyzer.state(),
            bit_idx: bit_idx as u8,
            prefix: self.prefix,
            match_len_bucket: match_state.len_bucket,
            match_predicted_class: match_state.predicted_class,
        }
    }

    pub(super) fn logistic_prob_one(
        &mut self,
        experts: &mut [ExpertState],
        logistic: &mut LogisticMixCore,
        analyzer: &TextContextAnalyzer,
        match_state: LogisticMatchState,
        bit_idx: usize,
    ) -> Result<f64, String> {
        debug_assert!(self.active);
        self.prime_bit_probs_if_needed(experts, bit_idx)?;
        logistic.set_context(self.logistic_context(analyzer, match_state, bit_idx));
        Ok(logistic.predict_bit(&self.bit_probs[..experts.len()], DEFAULT_MIN_PROB))
    }

    pub(super) fn logistic_observe(
        &mut self,
        experts: &mut [ExpertState],
        logistic: &mut LogisticMixCore,
        analyzer: &TextContextAnalyzer,
        match_state: LogisticMatchState,
        bit_idx: usize,
        bit: bool,
    ) -> Result<f64, String> {
        debug_assert!(self.active);
        if self.primed_bit_idx != Some(bit_idx) {
            let _ = self.logistic_prob_one(experts, logistic, analyzer, match_state, bit_idx)?;
        }
        logistic.set_context(self.logistic_context(analyzer, match_state, bit_idx));
        let p_mix = logistic.observe_bit(&self.bit_probs[..experts.len()], bit, DEFAULT_MIN_PROB);
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let p1: f64 = self.bit_probs[idx];
            let pb: f64 = if bit { p1 } else { 1.0 - p1 };
            self.likelihoods[idx] = (self.likelihoods[idx] * pb).max(DEFAULT_MIN_PROB);
            self.states[idx].observe(experts[idx].predictor.as_mut(), bit_idx, bit)?;
        }
        self.expected_bit_idx += 1;
        self.prefix = advanced_prefix_code(self.prefix, bit);
        self.primed_bit_idx = None;
        Ok(p_mix)
    }

    pub(super) fn logistic_observe_known(
        &mut self,
        experts: &mut [ExpertState],
        logistic: &mut LogisticMixCore,
        analyzer: &TextContextAnalyzer,
        match_state: LogisticMatchState,
        bit_idx: usize,
        bit: bool,
    ) -> Result<f64, String> {
        debug_assert!(self.active);
        self.validate_bit_idx(bit_idx)?;
        let use_primed: bool = self.primed_bit_idx == Some(bit_idx);
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let p1: f64 = if use_primed {
                self.bit_probs[idx]
            } else {
                self.states[idx].prob_one(experts[idx].predictor.as_mut(), bit_idx)?
            };
            self.bit_probs[idx] = p1;
        }
        logistic.set_context(self.logistic_context(analyzer, match_state, bit_idx));
        let p_mix: f64 =
            logistic.observe_bit(&self.bit_probs[..experts.len()], bit, DEFAULT_MIN_PROB);
        #[allow(clippy::needless_range_loop)]
        for idx in 0..experts.len() {
            let p1: f64 = self.bit_probs[idx];
            let pb: f64 = if bit { p1 } else { 1.0 - p1 };
            self.likelihoods[idx] = (self.likelihoods[idx] * pb).max(DEFAULT_MIN_PROB);
            self.states[idx].observe(experts[idx].predictor.as_mut(), bit_idx, bit)?;
        }
        self.expected_bit_idx += 1;
        self.prefix = advanced_prefix_code(self.prefix, bit);
        self.primed_bit_idx = None;
        Ok(p_mix)
    }
}
