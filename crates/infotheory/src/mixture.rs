//! Online mixtures of probabilistic predictors (log-loss Hedge / Bayes, switching, MDL).
//!
//! This module provides a small, rigorously correct toolkit for sequential model mixing.
//! Predictors expose per-symbol log-probabilities, which allows principled Bayesian
//! mixture updates and clean information-theoretic accounting.
//!
//! ## Rate-Backend Mixtures
//!
//! The mixture primitives here power `RateBackend::Mixture`, enabling Bayes, fading Bayes,
//! switching, and MDL-style selectors to be used anywhere a rate backend is accepted.
#![cfg_attr(
    not(feature = "all-backends"),
    allow(
        dead_code,
        unused_imports,
        unused_variables,
        unused_mut,
        unreachable_code
    )
)]

#[cfg(test)]
use crate::api::MixtureSpec;
use crate::api::{MixtureKind, MixtureScheduleMode, RateBackend};
#[cfg(feature = "backend-calibrated")]
use crate::backends::calibration::CalibratorCore;
#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{
    ContextTree, ContextTreeLifecycleSnapshot, FacContextTree, FacContextTreeLifecycleSnapshot,
};
#[cfg(feature = "backend-match")]
use crate::backends::match_model::{MatchModel, MatchModelLifecycleSnapshot};
#[cfg(feature = "backend-ppmd")]
use crate::backends::ppmd::{PpmdLifecycleSnapshot, PpmdModel};
#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::{RosaPlus, RosaTx};
#[cfg(feature = "backend-sequitur")]
use crate::backends::sequitur::{SequiturCheckpoint, SequiturLifecycleSnapshot, SequiturModel};
#[cfg(feature = "backend-match")]
use crate::backends::sparse_match::SparseMatchModel;
use crate::backends::text_context::TextContextAnalyzer;
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
use crate::byte_prefix::{
    BytePrefixCdf, MsbPrefixRange, fill_prefix_cdf_from_log_probs, zeroed_prefix_cdf_box,
};
#[cfg(feature = "backend-mamba")]
use crate::mambazip;
use crate::neural_mix::{NeuralHistoryState, NeuralMixCore};
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip;
use crate::spec::CompiledRateBackend;
use std::sync::Arc;

/// Default minimum probability floor to avoid log(0).
pub const DEFAULT_MIN_PROB: f64 = 5.960_464_477_539_063e-8;

#[inline]
fn clamp_prob(p: f64, min_prob: f64) -> f64 {
    if p.is_finite() {
        p.max(min_prob)
    } else {
        min_prob
    }
}

#[inline]
fn clamp_unit_prob(p: f64, min_prob: f64) -> f64 {
    clamp_prob(p, min_prob).min(1.0 - min_prob)
}

#[inline]
fn logsumexp(xs: &[f64]) -> f64 {
    let mut max_v = f64::NEG_INFINITY;
    for &v in xs {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0;
    for &v in xs {
        sum += (v - max_v).exp();
    }
    max_v + sum.ln()
}

#[inline]
fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = if a > b { a } else { b };
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[inline]
fn logsumexp_weights(experts: &[ExpertState]) -> f64 {
    let mut max_v = f64::NEG_INFINITY;
    for e in experts {
        if e.log_weight > max_v {
            max_v = e.log_weight;
        }
    }
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0;
    for e in experts {
        sum += (e.log_weight - max_v).exp();
    }
    max_v + sum.ln()
}

fn normalize_simplex_weights(weights: &mut [f64]) {
    if weights.is_empty() {
        return;
    }
    let mut sum = 0.0;
    for weight in weights.iter_mut() {
        if !weight.is_finite() || *weight < 0.0 {
            *weight = 0.0;
        }
        sum += *weight;
    }
    if !sum.is_finite() || sum <= 0.0 {
        let uniform = 1.0 / (weights.len() as f64);
        weights.fill(uniform);
        return;
    }
    for weight in weights.iter_mut() {
        *weight /= sum;
    }
}

pub(crate) fn project_simplex_with_scratch(weights: &mut [f64], scratch: &mut Vec<f64>) {
    if weights.is_empty() {
        return;
    }

    scratch.clear();
    scratch.extend(
        weights
            .iter()
            .map(|&weight| if weight.is_finite() { weight } else { 0.0 }),
    );
    let sorted = scratch.as_mut_slice();
    sorted.sort_by(|a, b| b.total_cmp(a));

    let mut cumulative = 0.0;
    let mut rho = None;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += *value;
        let theta = (cumulative - 1.0) / ((index + 1) as f64);
        if *value > theta {
            rho = Some(index);
        }
    }

    let Some(rho_index) = rho else {
        let uniform = 1.0 / (weights.len() as f64);
        weights.fill(uniform);
        return;
    };

    let theta = (sorted.iter().take(rho_index + 1).sum::<f64>() - 1.0) / ((rho_index + 1) as f64);
    for weight in weights.iter_mut() {
        *weight = (*weight - theta).max(0.0);
    }
    normalize_simplex_weights(weights);
}

#[inline]
pub(crate) fn switching_alpha_for_update(
    schedule: MixtureScheduleMode,
    alpha: f64,
    processed_symbols: u64,
) -> f64 {
    match schedule {
        MixtureScheduleMode::Default => alpha.clamp(0.0, 1.0),
        MixtureScheduleMode::Theorem => 1.0 / ((processed_symbols + 2) as f64),
    }
}

#[inline]
pub(crate) fn convex_step_size_for_update(
    schedule: MixtureScheduleMode,
    alpha: f64,
    update_index: u64,
) -> f64 {
    let t = update_index.max(1) as f64;
    match schedule {
        MixtureScheduleMode::Default => alpha.max(1e-12) / t.sqrt(),
        MixtureScheduleMode::Theorem => DEFAULT_MIN_PROB / t.sqrt(),
    }
}

fn normalized_log_weights(log_weights: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut weights: Vec<f64> = log_weights.into_iter().collect();
    if weights.is_empty() {
        return Vec::new();
    }
    let max_log = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for w in &mut weights {
        *w = if max_log.is_finite() {
            (*w - max_log).exp()
        } else {
            0.0
        };
    }
    normalize_simplex_weights(&mut weights);
    weights
}

fn normalized_prior_weights(configs: &[ExpertConfig]) -> Vec<f64> {
    normalized_log_weights(configs.iter().map(|cfg| cfg.log_prior))
}

fn normalized_expert_prior_weights(experts: &[ExpertState]) -> Vec<f64> {
    normalized_log_weights(experts.iter().map(|expert| expert.log_prior))
}

#[cfg(feature = "backend-calibrated")]
#[inline]
fn reset_calibrated_wrapper_state(
    core: &mut CalibratorCore,
    bitwise: &mut BytePrefixStepState,
    pdf: &mut [f64; 256],
    valid: &mut bool,
) {
    core.reset_context();
    *bitwise = BytePrefixStepState::new();
    pdf.fill(1.0 / 256.0);
    *valid = false;
}

fn set_log_weights_from_linear(experts: &mut [ExpertState], weights: &[f64]) {
    for (expert, &weight) in experts.iter_mut().zip(weights.iter()) {
        expert.log_weight = if weight > 0.0 {
            weight.ln()
        } else {
            f64::NEG_INFINITY
        };
    }
}

/// Trait-object cloning companion for [`OnlineBytePredictor`].
pub trait OnlineBytePredictorClone {
    /// Clone this predictor as a trait object.
    ///
    /// This supports `Clone` for `Box<dyn OnlineBytePredictor>` via type erasure,
    /// so mixture experts can be duplicated without knowing their concrete type.
    fn clone_box(&self) -> Box<dyn OnlineBytePredictor>;
}

impl<T> OnlineBytePredictorClone for T
where
    T: 'static + OnlineBytePredictor + Clone,
{
    fn clone_box(&self) -> Box<dyn OnlineBytePredictor> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn OnlineBytePredictor> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Trait for online byte-level predictors that expose per-symbol log-probabilities.
pub trait OnlineBytePredictor: Send + OnlineBytePredictorClone {
    /// Whether this predictor supports frozen-conditioning resets.
    ///
    /// Predictors that return `false` may still support ordinary stream lifecycle
    /// hooks (`begin_stream`/`finish_stream`), but cannot provide plugin-entropy
    /// style frozen reset semantics.
    fn supports_frozen_reset(&self) -> bool {
        true
    }

    /// Optional stream-start hook.
    ///
    /// Predictors that require total symbol count (for example percent-based
    /// policy schedules) can initialize runtime state here.
    fn begin_stream(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
        Ok(())
    }

    /// Optional stream-finalization hook.
    fn finish_stream(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Capture a structural checkpoint when the concrete predictor supports it.
    fn checkpoint_if_supported(&mut self) -> Option<OnlineBytePredictorCheckpoint> {
        None
    }

    /// Restore a structural checkpoint created by [`Self::checkpoint_if_supported`].
    fn restore_checkpoint_if_supported(
        &mut self,
        _checkpoint: &OnlineBytePredictorCheckpoint,
    ) -> bool {
        false
    }

    /// Release a structural checkpoint that will never be restored.
    ///
    /// Predictors with compact rollback journals can use this to retire
    /// temporary checkpoints without clearing older checkpoints that may still
    /// be live elsewhere. Plain drop-based checkpoints can accept the default
    /// behavior: taking ownership of `_checkpoint` is already a successful
    /// discard.
    fn discard_checkpoint_if_supported(
        &mut self,
        _checkpoint: OnlineBytePredictorCheckpoint,
    ) -> bool {
        true
    }

    /// Clear compact checkpoint journals after all structural checkpoints expire.
    fn clear_checkpoints_if_supported(&mut self) {}

    /// Log-probability (natural log) of `symbol` given the current history.
    fn log_prob(&mut self, symbol: u8) -> f64;

    /// Bulk 256-way log-probabilities for the next byte.
    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        for (sym, slot) in out.iter_mut().enumerate() {
            *slot = self.log_prob(sym as u8);
        }
    }

    /// Whether this predictor can expose an MSB-first byte prefix natively.
    ///
    /// Native prefix stepping lets bitwise consumers query and condition on the
    /// bits of the next byte without first materializing all 256 byte
    /// probabilities. Predictors that return `false` remain fully supported
    /// through the generic byte-PDF prefix fallback.
    fn has_native_msb_byte_prefix(&self) -> bool {
        false
    }

    /// Prepare a native MSB-first byte-prefix step.
    ///
    /// Returns `true` when the predictor entered a native prefix state. Callers
    /// must then query bits in order, call [`Self::observe_native_msb_prefix_bit`]
    /// after each observed prefix bit, and finish with
    /// [`Self::finish_native_msb_byte_prefix`]. Implementations must leave the
    /// predictor unchanged when they return `Ok(false)` or `Err(_)`.
    fn begin_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
        Ok(false)
    }

    /// Abort an active native MSB-first byte-prefix step before any bits have
    /// been observed.
    ///
    /// This is used only when a caller restores to a checkpoint that was taken
    /// immediately after `begin_native_msb_byte_prefix`. Implementations must
    /// return an error rather than dropping observed prefix bits.
    fn abort_empty_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
        Ok(false)
    }

    /// Predict `P(bit = 1)` for the next MSB-first prefix bit.
    ///
    /// `bit_idx` must match the next unobserved bit in the active native
    /// prefix, counted MSB-first in `0..8`. Re-querying the current bit index
    /// before observing it is allowed.
    fn native_msb_prefix_prob_one(&mut self, _bit_idx: usize) -> Result<f64, String> {
        Err("native MSB-first byte-prefix prediction is unavailable".to_string())
    }

    /// Observe one MSB-first prefix bit inside an active native byte-prefix step.
    ///
    /// `bit_idx` must match the next unobserved bit in the active native
    /// prefix, counted MSB-first in `0..8`.
    fn observe_native_msb_prefix_bit(&mut self, _bit_idx: usize, _bit: bool) -> Result<(), String> {
        Err("native MSB-first byte-prefix stepping is unavailable".to_string())
    }

    /// Finish an active native byte-prefix step after all eight bits are known.
    fn finish_native_msb_byte_prefix(&mut self, _symbol: u8) -> Result<(), String> {
        Err("native MSB-first byte-prefix stepping is unavailable".to_string())
    }

    /// Log-probability (natural log) of `symbol`, then update the predictor.
    fn log_prob_update(&mut self, symbol: u8) -> f64 {
        let logp = self.log_prob(symbol);
        self.update(symbol);
        logp
    }

    /// Update the predictor with the observed `symbol`.
    fn update(&mut self, symbol: u8);

    /// Reset only dynamic conditioning state while preserving fitted parameters/statistics.
    ///
    /// Predictors with latent/posterior state may also preserve their learned
    /// parameter posterior here; "frozen" means no new parameter fitting during
    /// the score pass, not necessarily a static hidden-state belief.
    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        self.finish_stream()?;
        self.begin_stream(total_symbols)
    }

    /// Start a new stream in a way that preserves each predictor's semantic contract.
    ///
    /// This uses frozen-reset semantics when supported, and falls back to ordinary
    /// begin/finish stream hooks otherwise.
    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        if self.supports_frozen_reset() {
            self.reset_frozen(total_symbols)
        } else {
            self.begin_stream(total_symbols)
        }
    }

    /// Advance conditioning state without fitting or adapting parameters.
    ///
    /// For state-space or latent-variable models this may still update internal
    /// filtering/posterior state needed for correct sequential predictions.
    fn update_frozen(&mut self, symbol: u8) {
        self.update(symbol);
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct OnlineBytePredictorCheckpoint(OnlineBytePredictorCheckpointKind);

#[derive(Clone)]
enum OnlineBytePredictorCheckpointKind {
    RateBackend(RateBackendPredictorCheckpoint),
}

impl OnlineBytePredictorCheckpoint {
    fn rate_backend(checkpoint: RateBackendPredictorCheckpoint) -> Self {
        Self(OnlineBytePredictorCheckpointKind::RateBackend(checkpoint))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OnlineBytePredictorLifecycleOp {
    /// Start a possibly continuing stream.
    BeginStream,
    /// Start a fresh stream, preserving fitted predictor state where supported.
    BeginFreshStream,
    /// Reset transient conditioning state while preserving fitted state.
    ResetFrozen,
    /// Finish the current stream.
    FinishStream,
}

#[cfg(feature = "backend-ctw")]
fn validate_native_msb_prefix_bit_idx(next_bit_idx: usize, bit_idx: usize) -> Result<(), String> {
    if bit_idx >= 8 {
        return Err(format!(
            "native MSB-first byte-prefix bit index {bit_idx} is out of range; expected 0..8"
        ));
    }
    if bit_idx != next_bit_idx {
        return Err(format!(
            "native MSB-first byte-prefix bit index {bit_idx} violated sequential stepping; expected {next_bit_idx}"
        ));
    }
    Ok(())
}

#[cfg(feature = "backend-ctw")]
fn validate_native_msb_prefix_finish(next_bit_idx: usize) -> Result<(), String> {
    if next_bit_idx != 8 {
        return Err(format!(
            "native MSB-first byte-prefix finish requires 8 observed bits, got {next_bit_idx}"
        ));
    }
    Ok(())
}

#[derive(Clone)]
#[doc(hidden)]
pub struct BytePrefixStepState {
    kind: BytePrefixStepStateKind,
}

#[derive(Clone)]
enum BytePrefixStepStateKind {
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

impl Default for BytePrefixStepState {
    fn default() -> Self {
        Self {
            kind: BytePrefixStepStateKind::default(),
        }
    }
}

impl BytePrefixStepState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn is_native(&self) -> bool {
        matches!(self.kind, BytePrefixStepStateKind::Native)
    }

    fn prepare(&mut self, predictor: &mut dyn OnlineBytePredictor) -> Result<(), String> {
        if predictor.begin_native_msb_byte_prefix()? {
            self.kind = BytePrefixStepStateKind::Native;
            return Ok(());
        }

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

    fn prob_one(
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

    fn observe(
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

    fn abort_empty(&mut self, predictor: &mut dyn OnlineBytePredictor) -> Result<(), String> {
        match &mut self.kind {
            BytePrefixStepStateKind::Native => {
                predictor.abort_empty_native_msb_byte_prefix()?;
                *self = Self::new();
                Ok(())
            }
            BytePrefixStepStateKind::PdfPrefix { .. } => {
                *self = Self::new();
                Ok(())
            }
        }
    }

    fn finish(
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
struct MixtureBitPrefixState {
    states: Vec<BytePrefixStepState>,
    weights: Vec<f64>,
    likelihoods: Vec<f64>,
    bit_probs: Vec<f64>,
    logps: Vec<f64>,
    active: bool,
    primed_bit_idx: Option<usize>,
    expected_bit_idx: usize,
}

impl MixtureBitPrefixState {
    fn reset_inactive(&mut self) {
        self.active = false;
        self.primed_bit_idx = None;
        self.expected_bit_idx = 0;
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

    fn begin(&mut self, experts: &mut [ExpertState], weights: &[f64]) -> Result<bool, String> {
        if !experts
            .iter()
            .any(|expert| expert.predictor.has_native_msb_byte_prefix())
        {
            self.reset_inactive();
            return Ok(false);
        }

        let n: usize = experts.len();
        let mut native_checkpoints: Vec<ExpertTempCheckpoint> = Vec::new();
        for (idx, expert) in experts.iter_mut().enumerate() {
            if !expert.predictor.has_native_msb_byte_prefix() {
                continue;
            }
            let Some(checkpoint) = expert.predictor.checkpoint_if_supported() else {
                for checkpoint in native_checkpoints.drain(..) {
                    assert!(
                        experts[checkpoint.index]
                            .predictor
                            .discard_checkpoint_if_supported(checkpoint.checkpoint),
                        "native-prefix expert checkpoint could not be discarded",
                    );
                }
                self.reset_inactive();
                return Ok(false);
            };
            native_checkpoints.push(ExpertTempCheckpoint {
                index: idx,
                checkpoint,
            });
        }

        self.states.resize_with(n, BytePrefixStepState::new);
        self.weights.clear();
        self.weights.extend(weights.iter().copied());
        normalize_simplex_weights(&mut self.weights);
        self.likelihoods.resize(n, 1.0);
        self.likelihoods.fill(1.0);
        self.bit_probs.resize(n, 0.5);
        self.logps.resize(n, 0.0);
        self.primed_bit_idx = None;
        self.expected_bit_idx = 0;
        for (state, expert) in self.states.iter_mut().zip(experts.iter_mut()) {
            if let Err(err) = state.prepare(expert.predictor.as_mut()) {
                for checkpoint in native_checkpoints.drain(..) {
                    assert!(
                        experts[checkpoint.index]
                            .predictor
                            .restore_checkpoint_if_supported(&checkpoint.checkpoint),
                        "native-prefix expert checkpoint could not be restored",
                    );
                    assert!(
                        experts[checkpoint.index]
                            .predictor
                            .discard_checkpoint_if_supported(checkpoint.checkpoint),
                        "native-prefix expert checkpoint could not be discarded",
                    );
                }
                self.reset_inactive();
                return Err(err);
            }
        }
        for checkpoint in native_checkpoints.drain(..) {
            assert!(
                experts[checkpoint.index]
                    .predictor
                    .discard_checkpoint_if_supported(checkpoint.checkpoint),
                "native-prefix expert checkpoint could not be discarded",
            );
        }
        self.active = true;
        Ok(true)
    }

    fn abort_empty(&mut self, experts: &mut [ExpertState]) -> Result<bool, String> {
        if !self.active {
            return Ok(false);
        }
        if self.expected_bit_idx != 0 {
            return Err(format!(
                "native MSB-first byte-prefix abort requires zero observed bits, got {}",
                self.expected_bit_idx
            ));
        }
        for (state, expert) in self.states.iter_mut().zip(experts.iter_mut()) {
            state.abort_empty(expert.predictor.as_mut())?;
        }
        self.reset_inactive();
        Ok(true)
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

    fn prob_one(&mut self, experts: &mut [ExpertState], bit_idx: usize) -> Result<f64, String> {
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

    fn observe(
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
        self.primed_bit_idx = None;
        Ok(())
    }

    fn finish_adaptive(&mut self, experts: &mut [ExpertState], symbol: u8) -> Result<(), String> {
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
}

struct ExpertTempCheckpoint {
    index: usize,
    checkpoint: OnlineBytePredictorCheckpoint,
}

#[cfg(feature = "backend-rwkv")]
#[inline]
fn ensure_rwkv_primed(compressor: &mut rwkvzip::Compressor, primed: &mut bool) {
    if !*primed {
        compressor.reset_and_prime();
        *primed = true;
    }
}

#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{
    ctw_log_prob_msb, ctw_log_prob_update_lsb, ctw_log_prob_update_msb, ctw_symbol_bit_msb,
    fill_ctw_tree_log_probs, fill_fac_tree_log_probs,
};
/// A concrete online predictor backed by a `RateBackend` configuration.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum RateBackendPredictor {
    /// ROSA-Plus online suffix automaton.
    #[cfg(feature = "backend-rosa")]
    Rosa {
        /// ROSA model state.
        model: RosaPlus,
        /// Probability floor for numeric stability.
        min_prob: f64,
        /// Undo log for checkpointed updates and frozen-conditioning moves.
        checkpoint_journal: Vec<RosaPredictorUndo>,
        /// Number of active checkpoints currently recording into `checkpoint_journal`.
        checkpoint_depth: usize,
    },
    /// Local contiguous match predictor.
    #[cfg(feature = "backend-match")]
    Match {
        /// Match model state.
        model: MatchModel,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// Sparse/gapped local match predictor.
    #[cfg(feature = "backend-match")]
    SparseMatch {
        /// Sparse-match model state.
        model: SparseMatchModel,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// Bounded-memory PPMD-style predictor.
    #[cfg(feature = "backend-ppmd")]
    Ppmd {
        /// PPMD model state.
        model: PpmdModel,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// Exact online Sequitur grammar backend with predictive suffix contexts.
    #[cfg(feature = "backend-sequitur")]
    Sequitur {
        /// Sequitur model state.
        model: SequiturModel,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// AC-CTW with consumer-chosen symbol width interpreted MSB-first.
    #[cfg(feature = "backend-ctw")]
    Ctw {
        /// Single binary context tree.
        tree: ContextTree,
        /// Active bit-width per observed symbol.
        bits_per_symbol: usize,
        /// Probability floor for numeric stability.
        min_prob: f64,
        /// Compact rollback journal used while checkpoint scopes are active.
        checkpoint_journal: Vec<CtwUndoOp>,
        /// Number of active checkpoints that require journaling.
        checkpoint_depth: usize,
        /// In-flight native byte-prefix progress when stepping MSB-first bits.
        native_prefix_progress: Option<usize>,
    },
    /// Factorized CTW with width-dependent bit order.
    #[cfg(feature = "backend-ctw")]
    FacCtw {
        /// FAC-CTW tree stack for configured bit width.
        tree: FacContextTree,
        /// Active bit-width per symbol.
        bits_per_symbol: usize,
        /// Effective symbol bit order (`true` => MSB-first).
        ///
        /// FAC-CTW keeps legacy LSB-first behavior for non-byte symbol widths,
        /// but 8-bit symbols run MSB-first so byte-packed sessions can use the
        /// native prefix path consistently.
        msb_first: bool,
        /// Probability floor for numeric stability.
        min_prob: f64,
        /// Compact rollback journal used while checkpoint scopes are active.
        checkpoint_journal: Vec<FacCtwUndoOp>,
        /// Number of active checkpoints that require journaling.
        checkpoint_depth: usize,
        /// In-flight native byte-prefix progress when stepping MSB-first bits.
        native_prefix_progress: Option<usize>,
    },
    /// RWKV-7 neural predictor.
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        /// RWKV compressor/runtime state.
        compressor: rwkvzip::Compressor,
        /// Whether the first-token distribution has been primed.
        primed: bool,
        /// Scratch copy used for update API that borrows immutable PDF.
        pdf_scratch: Vec<f64>,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// Mamba-1 neural predictor.
    #[cfg(feature = "backend-mamba")]
    Mamba {
        /// Mamba compressor/runtime state.
        compressor: mambazip::Compressor,
        /// Whether the first-token distribution has been primed.
        primed: bool,
        /// Scratch copy used for update API that borrows immutable PDF.
        pdf_scratch: Vec<f64>,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// ZPAQ streaming rate model.
    #[cfg(feature = "backend-zpaq")]
    Zpaq {
        /// ZPAQ rate model state.
        model: ZpaqRateModel,
    },
    /// Online mixture over experts (Bayes, fading Bayes, switching, MDL).
    #[cfg(feature = "backend-mixture")]
    Mixture {
        /// Active mixture runtime.
        runtime: MixtureRuntime,
    },
    /// Particle-latent filter ensemble.
    #[cfg(feature = "backend-particle")]
    Particle {
        /// Particle runtime.
        runtime: crate::backends::particle::ParticleRuntime,
    },
    /// Calibrated wrapper around another predictor.
    #[cfg(feature = "backend-calibrated")]
    Calibrated {
        /// Wrapped predictor whose PDF is calibrated.
        base: Box<RateBackendPredictor>,
        /// Online calibrator state and context features.
        core: CalibratorCore,
        /// Reused byte-prefix state for the wrapped predictor.
        bitwise: BytePrefixStepState,
        /// Cached calibrated PDF.
        pdf: [f64; 256],
        /// Whether `pdf` currently matches wrapped state.
        valid: bool,
        /// Probability floor used for numerical stability.
        min_prob: f64,
    },
    /// Internal fallback variant used in ultra-minimal builds.
    Disabled {
        /// Human-readable failure reason.
        reason: String,
    },
}

#[derive(Clone)]
/// Checkpoint snapshot used for temporary predictor rollback.
///
/// Most backends use a full cloned predictor snapshot. Sequitur, CTW/FAC-CTW,
/// and ROSA use compact rollback markers to avoid cloning hot-path runtime
/// state.
pub enum RateBackendPredictorCheckpoint {
    /// Full predictor clone for backends without specialized checkpointing.
    ///
    /// Boxed so the enum stays pointer-sized: compact variants (`Ctw`, `Rosa`,
    /// `Sequitur`, etc.) are not forced to move ~4 KiB on the stack when passed
    /// through this type. The clone+heap cost is negligible versus `self.clone()`.
    /// Byte-prefix session buffering still boxes checkpoints out-of-line when
    /// stored in long-lived session state.
    Full(Box<RateBackendPredictor>),
    /// Compact ROSA journal marker for [`RateBackendPredictor::Rosa`].
    #[cfg(feature = "backend-rosa")]
    Rosa {
        /// Length of the rollback journal to restore when unwinding the checkpoint.
        journal_len: usize,
    },
    /// Compact Sequitur undo marker for [`RateBackendPredictor::Sequitur`].
    #[cfg(feature = "backend-sequitur")]
    Sequitur(SequiturCheckpoint),
    /// Compact CTW journal marker for [`RateBackendPredictor::Ctw`].
    #[cfg(feature = "backend-ctw")]
    Ctw {
        /// Length of the rollback journal to restore when unwinding the checkpoint.
        journal_len: usize,
        /// In-flight native prefix progress captured with the checkpoint.
        native_prefix_progress: Option<usize>,
    },
    /// Compact FAC-CTW journal marker for [`RateBackendPredictor::FacCtw`].
    #[cfg(feature = "backend-ctw")]
    FacCtw {
        /// Length of the rollback journal to restore when unwinding the checkpoint.
        journal_len: usize,
        /// In-flight native prefix progress captured with the checkpoint.
        native_prefix_progress: Option<usize>,
    },
    /// Composite checkpoint for calibrated predictors.
    #[cfg(feature = "backend-calibrated")]
    Calibrated(Box<CalibratedPredictorCheckpoint>),
    /// Lightweight calibrated byte-prefix start checkpoint.
    #[cfg(feature = "backend-calibrated")]
    CalibratedNativePrefixStart {
        /// Wrapped predictor checkpoint only; the SSE core uses its bounded
        /// active-byte undo log instead of cloning the full calibrated table.
        base: Box<RateBackendPredictorCheckpoint>,
        /// Whether the wrapped predictor itself had entered a native prefix
        /// session when the checkpoint was taken.
        base_prefix_active: bool,
    },
    /// Composite checkpoint for mixture predictors.
    #[cfg(feature = "backend-mixture")]
    Mixture(Box<MixtureRuntimeCheckpoint>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Internal AC-CTW journal event used to restore predictor state from checkpoints.
#[doc(hidden)]
pub enum CtwUndoOp {
    /// Symbol update applied in learning mode.
    LearnedSymbol,
    /// One native prefix bit applied in learning mode.
    LearnedBit,
    /// Symbol update applied in frozen/scoring mode.
    FrozenSymbol,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Internal FAC-CTW journal event used to restore predictor state from checkpoints.
#[doc(hidden)]
pub enum FacCtwUndoOp {
    /// Symbol update applied in learning mode.
    LearnedSymbol,
    /// One native prefix bit applied in learning mode.
    LearnedBit { bit_idx: usize },
    /// Symbol update applied in frozen/scoring mode.
    FrozenSymbol,
}

#[derive(Clone)]
#[cfg(feature = "backend-rosa")]
/// Internal ROSA journal event used to restore predictor state from checkpoints.
#[doc(hidden)]
pub enum RosaPredictorUndo {
    Learned(Box<RosaTx>),
    FrozenCursor { previous_last: i32 },
}

#[derive(Clone)]
#[cfg(feature = "backend-calibrated")]
/// Internal checkpoint payload for [`RateBackendPredictor::Calibrated`].
///
/// This stores wrapped predictor state plus calibrator caches so temporary
/// lookahead scoring can rollback without rebuilding runtime objects. The
/// learned SSE table is shared copy-on-write, so read-only lookahead and
/// checkpoint capture avoid multi-megabyte table copies while exact rollback is
/// preserved if later training mutates the table.
pub struct CalibratedPredictorCheckpoint {
    base: Box<RateBackendPredictorCheckpoint>,
    core: CalibratorCore,
    bitwise: BytePrefixStepState,
    pdf: [f64; 256],
    valid: bool,
}

enum RateBackendPredictorLifecycleCheckpoint {
    NotNeeded,
    Full(Box<RateBackendPredictor>),
    #[cfg(feature = "backend-ctw")]
    Ctw {
        tree: Option<ContextTreeLifecycleSnapshot>,
        native_prefix_progress: Option<usize>,
    },
    #[cfg(feature = "backend-ctw")]
    FacCtw {
        tree: Option<FacContextTreeLifecycleSnapshot>,
        native_prefix_progress: Option<usize>,
    },
    #[cfg(feature = "backend-ppmd")]
    Ppmd(PpmdLifecycleSnapshot),
    #[cfg(feature = "backend-match")]
    Match(MatchModelLifecycleSnapshot),
    #[cfg(feature = "backend-match")]
    SparseMatch(MatchModelLifecycleSnapshot),
    #[cfg(feature = "backend-sequitur")]
    Sequitur(SequiturLifecycleSnapshot),
    #[cfg(feature = "backend-calibrated")]
    Calibrated(Box<CalibratedPredictorLifecycleCheckpoint>),
    #[cfg(feature = "backend-mixture")]
    Mixture(Box<MixtureRuntimeLifecycleCheckpoint>),
}

#[cfg(feature = "backend-calibrated")]
struct CalibratedPredictorLifecycleCheckpoint {
    base: Box<RateBackendPredictorLifecycleCheckpoint>,
    core: CalibratorCore,
    bitwise: BytePrefixStepState,
    pdf: [f64; 256],
    valid: bool,
}

#[cfg(feature = "backend-ctw")]
fn restore_ctw_checkpoint(
    tree: &mut ContextTree,
    bits_per_symbol: usize,
    checkpoint_journal: &mut Vec<CtwUndoOp>,
    target_len: usize,
) {
    let bits = bits_per_symbol.clamp(1, 8);
    while checkpoint_journal.len() > target_len {
        match checkpoint_journal
            .pop()
            .expect("ctw checkpoint journal underflow")
        {
            CtwUndoOp::LearnedSymbol => {
                for _ in 0..bits {
                    tree.revert();
                }
            }
            CtwUndoOp::LearnedBit => {
                tree.revert();
            }
            CtwUndoOp::FrozenSymbol => {
                for _ in 0..bits {
                    tree.revert_history();
                }
            }
        }
    }
}

#[cfg(feature = "backend-ctw")]
fn restore_fac_ctw_checkpoint(
    tree: &mut FacContextTree,
    bits_per_symbol: usize,
    checkpoint_journal: &mut Vec<FacCtwUndoOp>,
    target_len: usize,
) {
    let bits = bits_per_symbol.clamp(1, 8);
    while checkpoint_journal.len() > target_len {
        match checkpoint_journal
            .pop()
            .expect("ctw checkpoint journal underflow")
        {
            FacCtwUndoOp::LearnedSymbol => {
                for bit_idx in (0..bits).rev() {
                    tree.revert(bit_idx);
                }
            }
            FacCtwUndoOp::LearnedBit { bit_idx } => {
                tree.revert(bit_idx);
            }
            FacCtwUndoOp::FrozenSymbol => {
                tree.revert_history(bits);
            }
        }
    }
}

impl RateBackendPredictor {
    /// Create a new online predictor from a compiled rate backend plan.
    pub fn try_from_compiled(backend: &CompiledRateBackend, min_prob: f64) -> Result<Self, String> {
        crate::runtime::build_rate_backend_predictor(backend, min_prob)
    }

    /// Create a new online predictor from a rate backend configuration.
    pub fn try_from_backend(backend: RateBackend, min_prob: f64) -> Result<Self, String> {
        let compiled = backend.compile().map_err(|err| err.to_string())?;
        Self::try_from_compiled(&compiled, min_prob)
    }

    /// Create a new online predictor from a rate backend configuration.
    pub fn from_backend(backend: RateBackend, min_prob: f64) -> Self {
        Self::try_from_backend(backend, min_prob)
            .unwrap_or_else(|err| panic!("failed to build RateBackendPredictor: {err}"))
    }

    /// Create a new online predictor from a compiled rate backend plan.
    pub fn from_compiled(backend: &CompiledRateBackend, min_prob: f64) -> Self {
        Self::try_from_compiled(backend, min_prob)
            .unwrap_or_else(|err| panic!("failed to build RateBackendPredictor: {err}"))
    }

    /// Human-readable default name for a backend.
    pub fn default_name(backend: &RateBackend) -> String {
        backend
            .compile()
            .map(|compiled| compiled.default_name())
            .unwrap_or_else(|_| {
                backend
                    .descriptor()
                    .map(|descriptor| format!("{}(invalid)", descriptor.canonical))
                    .unwrap_or_else(|_| "backend(invalid)".to_string())
            })
    }

    fn lifecycle_checkpoint(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
    ) -> RateBackendPredictorLifecycleCheckpoint {
        #[cfg(feature = "backend-ctw")]
        if matches!(
            self,
            RateBackendPredictor::Ctw {
                native_prefix_progress: Some(bits),
                ..
            } | RateBackendPredictor::FacCtw {
                native_prefix_progress: Some(bits),
                ..
            } if *bits > 0
        ) {
            return RateBackendPredictorLifecycleCheckpoint::Full(Box::new(self.clone()));
        }

        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa { .. } => match op {
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::BeginStream => {
                    RateBackendPredictorLifecycleCheckpoint::NotNeeded
                }
                OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::Full(Box::new(self.clone()))
                }
            },
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { model, .. } => match op {
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::BeginStream => {
                    RateBackendPredictorLifecycleCheckpoint::NotNeeded
                }
                OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::Match(model.lifecycle_snapshot())
                }
            },
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { model, .. } => match op {
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::BeginStream => {
                    RateBackendPredictorLifecycleCheckpoint::NotNeeded
                }
                OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::SparseMatch(model.lifecycle_snapshot())
                }
            },
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { model, .. } => match op {
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::BeginStream => {
                    RateBackendPredictorLifecycleCheckpoint::NotNeeded
                }
                OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::Ppmd(model.lifecycle_snapshot())
                }
            },
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => match op {
                OnlineBytePredictorLifecycleOp::FinishStream => {
                    RateBackendPredictorLifecycleCheckpoint::NotNeeded
                }
                OnlineBytePredictorLifecycleOp::BeginStream
                | OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::Sequitur(model.lifecycle_snapshot())
                }
            },
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                native_prefix_progress,
                ..
            } => match op {
                OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::Ctw {
                        tree: Some(tree.lifecycle_snapshot()),
                        native_prefix_progress: *native_prefix_progress,
                    }
                }
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::BeginStream => {
                    if native_prefix_progress.is_some() {
                        RateBackendPredictorLifecycleCheckpoint::Ctw {
                            tree: None,
                            native_prefix_progress: *native_prefix_progress,
                        }
                    } else {
                        RateBackendPredictorLifecycleCheckpoint::NotNeeded
                    }
                }
            },
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                native_prefix_progress,
                ..
            } => match op {
                OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::FacCtw {
                        tree: Some(tree.lifecycle_snapshot()),
                        native_prefix_progress: *native_prefix_progress,
                    }
                }
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::BeginStream => {
                    if native_prefix_progress.is_some() {
                        RateBackendPredictorLifecycleCheckpoint::FacCtw {
                            tree: None,
                            native_prefix_progress: *native_prefix_progress,
                        }
                    } else {
                        RateBackendPredictorLifecycleCheckpoint::NotNeeded
                    }
                }
            },
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { .. } => match op {
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::ResetFrozen => {
                    RateBackendPredictorLifecycleCheckpoint::NotNeeded
                }
                OnlineBytePredictorLifecycleOp::BeginStream
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::Full(Box::new(self.clone()))
                }
            },
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { .. } => match op {
                OnlineBytePredictorLifecycleOp::FinishStream
                | OnlineBytePredictorLifecycleOp::BeginStream => {
                    RateBackendPredictorLifecycleCheckpoint::NotNeeded
                }
                OnlineBytePredictorLifecycleOp::ResetFrozen
                | OnlineBytePredictorLifecycleOp::BeginFreshStream => {
                    RateBackendPredictorLifecycleCheckpoint::Full(Box::new(self.clone()))
                }
            },
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 { .. } => {
                RateBackendPredictorLifecycleCheckpoint::Full(Box::new(self.clone()))
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba { .. } => {
                RateBackendPredictorLifecycleCheckpoint::Full(Box::new(self.clone()))
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                pdf,
                valid,
                ..
            } => RateBackendPredictorLifecycleCheckpoint::Calibrated(Box::new(
                CalibratedPredictorLifecycleCheckpoint {
                    base: Box::new(base.lifecycle_checkpoint(op)),
                    core: core.clone(),
                    bitwise: bitwise.clone(),
                    pdf: *pdf,
                    valid: *valid,
                },
            )),
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                RateBackendPredictorLifecycleCheckpoint::Mixture(Box::new(
                    runtime.lifecycle_checkpoint(op),
                ))
            }
            RateBackendPredictor::Disabled { .. } => {
                RateBackendPredictorLifecycleCheckpoint::NotNeeded
            }
        }
    }

    fn restore_lifecycle_checkpoint(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
        checkpoint: RateBackendPredictorLifecycleCheckpoint,
    ) {
        match (self, checkpoint) {
            (_, RateBackendPredictorLifecycleCheckpoint::NotNeeded) => {}
            (slot, RateBackendPredictorLifecycleCheckpoint::Full(state)) => {
                *slot = *state;
            }
            #[cfg(feature = "backend-ctw")]
            (
                RateBackendPredictor::Ctw {
                    tree,
                    native_prefix_progress,
                    ..
                },
                RateBackendPredictorLifecycleCheckpoint::Ctw {
                    tree: tree_snapshot,
                    native_prefix_progress: prefix,
                },
            ) => {
                if let Some(snapshot) = tree_snapshot {
                    tree.restore_lifecycle_snapshot(snapshot);
                }
                *native_prefix_progress = prefix;
            }
            #[cfg(feature = "backend-ctw")]
            (
                RateBackendPredictor::FacCtw {
                    tree,
                    native_prefix_progress,
                    ..
                },
                RateBackendPredictorLifecycleCheckpoint::FacCtw {
                    tree: tree_snapshot,
                    native_prefix_progress: prefix,
                },
            ) => {
                if let Some(snapshot) = tree_snapshot {
                    tree.restore_lifecycle_snapshot(snapshot);
                }
                *native_prefix_progress = prefix;
            }
            #[cfg(feature = "backend-ppmd")]
            (
                RateBackendPredictor::Ppmd { model, .. },
                RateBackendPredictorLifecycleCheckpoint::Ppmd(snapshot),
            ) => model.restore_lifecycle_snapshot(snapshot),
            #[cfg(feature = "backend-match")]
            (
                RateBackendPredictor::Match { model, .. },
                RateBackendPredictorLifecycleCheckpoint::Match(snapshot),
            ) => model.restore_lifecycle_snapshot(snapshot),
            #[cfg(feature = "backend-match")]
            (
                RateBackendPredictor::SparseMatch { model, .. },
                RateBackendPredictorLifecycleCheckpoint::SparseMatch(snapshot),
            ) => model.restore_lifecycle_snapshot(snapshot),
            #[cfg(feature = "backend-sequitur")]
            (
                RateBackendPredictor::Sequitur { model, .. },
                RateBackendPredictorLifecycleCheckpoint::Sequitur(snapshot),
            ) => model.restore_lifecycle_snapshot(snapshot),
            #[cfg(feature = "backend-calibrated")]
            (
                RateBackendPredictor::Calibrated {
                    base,
                    core,
                    bitwise,
                    pdf,
                    valid,
                    ..
                },
                RateBackendPredictorLifecycleCheckpoint::Calibrated(checkpoint),
            ) => {
                base.restore_lifecycle_checkpoint(op, *checkpoint.base);
                *core = checkpoint.core;
                *bitwise = checkpoint.bitwise;
                *pdf = checkpoint.pdf;
                *valid = checkpoint.valid;
            }
            #[cfg(feature = "backend-mixture")]
            (
                RateBackendPredictor::Mixture { runtime },
                RateBackendPredictorLifecycleCheckpoint::Mixture(checkpoint),
            ) => runtime.restore_lifecycle_checkpoint(op, *checkpoint),
            #[cfg(feature = "backend-ctw")]
            (_, RateBackendPredictorLifecycleCheckpoint::Ctw { .. }) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-ctw")]
            (_, RateBackendPredictorLifecycleCheckpoint::FacCtw { .. }) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-ppmd")]
            (_, RateBackendPredictorLifecycleCheckpoint::Ppmd(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-match")]
            (_, RateBackendPredictorLifecycleCheckpoint::Match(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-match")]
            (_, RateBackendPredictorLifecycleCheckpoint::SparseMatch(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-sequitur")]
            (_, RateBackendPredictorLifecycleCheckpoint::Sequitur(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-calibrated")]
            (_, RateBackendPredictorLifecycleCheckpoint::Calibrated(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-mixture")]
            (_, RateBackendPredictorLifecycleCheckpoint::Mixture(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
        }
    }

    fn discard_lifecycle_checkpoint(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
        checkpoint: RateBackendPredictorLifecycleCheckpoint,
    ) {
        match (self, checkpoint) {
            (_, RateBackendPredictorLifecycleCheckpoint::NotNeeded)
            | (_, RateBackendPredictorLifecycleCheckpoint::Full(_)) => {}
            #[cfg(feature = "backend-calibrated")]
            (
                RateBackendPredictor::Calibrated { base, .. },
                RateBackendPredictorLifecycleCheckpoint::Calibrated(checkpoint),
            ) => base.discard_lifecycle_checkpoint(op, *checkpoint.base),
            #[cfg(feature = "backend-mixture")]
            (
                RateBackendPredictor::Mixture { runtime },
                RateBackendPredictorLifecycleCheckpoint::Mixture(checkpoint),
            ) => runtime.discard_lifecycle_checkpoint(op, *checkpoint),
            #[cfg(feature = "backend-ctw")]
            (
                RateBackendPredictor::Ctw { .. },
                RateBackendPredictorLifecycleCheckpoint::Ctw { .. },
            )
            | (
                RateBackendPredictor::FacCtw { .. },
                RateBackendPredictorLifecycleCheckpoint::FacCtw { .. },
            ) => {}
            #[cfg(feature = "backend-ppmd")]
            (
                RateBackendPredictor::Ppmd { .. },
                RateBackendPredictorLifecycleCheckpoint::Ppmd(_),
            ) => {}
            #[cfg(feature = "backend-match")]
            (
                RateBackendPredictor::Match { .. },
                RateBackendPredictorLifecycleCheckpoint::Match(_),
            )
            | (
                RateBackendPredictor::SparseMatch { .. },
                RateBackendPredictorLifecycleCheckpoint::SparseMatch(_),
            ) => {}
            #[cfg(feature = "backend-sequitur")]
            (
                RateBackendPredictor::Sequitur { .. },
                RateBackendPredictorLifecycleCheckpoint::Sequitur(_),
            ) => {}
            #[cfg(feature = "backend-ctw")]
            (_, RateBackendPredictorLifecycleCheckpoint::Ctw { .. })
            | (_, RateBackendPredictorLifecycleCheckpoint::FacCtw { .. }) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-ppmd")]
            (_, RateBackendPredictorLifecycleCheckpoint::Ppmd(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-match")]
            (_, RateBackendPredictorLifecycleCheckpoint::Match(_))
            | (_, RateBackendPredictorLifecycleCheckpoint::SparseMatch(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-sequitur")]
            (_, RateBackendPredictorLifecycleCheckpoint::Sequitur(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-calibrated")]
            (_, RateBackendPredictorLifecycleCheckpoint::Calibrated(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
            #[cfg(feature = "backend-mixture")]
            (_, RateBackendPredictorLifecycleCheckpoint::Mixture(_)) => {
                panic!("mismatched RateBackendPredictor lifecycle checkpoint variant")
            }
        }
    }

    pub(crate) fn checkpoint(&mut self) -> RateBackendPredictorCheckpoint {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                checkpoint_journal,
                checkpoint_depth,
                ..
            } => {
                *checkpoint_depth = checkpoint_depth.saturating_add(1);
                RateBackendPredictorCheckpoint::Rosa {
                    journal_len: checkpoint_journal.len(),
                }
            }
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => {
                RateBackendPredictorCheckpoint::Sequitur(model.checkpoint())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                *checkpoint_depth = checkpoint_depth.saturating_add(1);
                RateBackendPredictorCheckpoint::Ctw {
                    journal_len: checkpoint_journal.len(),
                    native_prefix_progress: *native_prefix_progress,
                }
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                *checkpoint_depth = checkpoint_depth.saturating_add(1);
                RateBackendPredictorCheckpoint::FacCtw {
                    journal_len: checkpoint_journal.len(),
                    native_prefix_progress: *native_prefix_progress,
                }
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                pdf,
                valid,
                ..
            } => RateBackendPredictorCheckpoint::Calibrated(Box::new(
                CalibratedPredictorCheckpoint {
                    base: Box::new(base.checkpoint()),
                    core: core.clone(),
                    bitwise: bitwise.clone(),
                    pdf: *pdf,
                    valid: *valid,
                },
            )),
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime
                .checkpoint()
                .map(|checkpoint| RateBackendPredictorCheckpoint::Mixture(Box::new(checkpoint)))
                .unwrap_or_else(|| RateBackendPredictorCheckpoint::Full(Box::new(self.clone()))),
            _ => RateBackendPredictorCheckpoint::Full(Box::new(self.clone())),
        }
    }

    pub(crate) fn native_prefix_start_checkpoint(&mut self) -> RateBackendPredictorCheckpoint {
        #[cfg(feature = "backend-calibrated")]
        if let RateBackendPredictor::Calibrated {
            base,
            core,
            bitwise,
            ..
        } = self
        {
            return RateBackendPredictorCheckpoint::CalibratedNativePrefixStart {
                base: Box::new(base.checkpoint()),
                base_prefix_active: core.byte_is_active() && bitwise.is_native(),
            };
        }

        self.checkpoint()
    }

    pub(crate) fn abort_empty_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
        match self {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                native_prefix_progress,
                ..
            } => match *native_prefix_progress {
                Some(0) => {
                    *native_prefix_progress = None;
                    Ok(true)
                }
                Some(bits) => Err(format!(
                    "native MSB-first byte-prefix abort requires zero observed bits, got {bits}"
                )),
                None => Ok(false),
            },
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                native_prefix_progress,
                ..
            } => match *native_prefix_progress {
                Some(0) => {
                    *native_prefix_progress = None;
                    Ok(true)
                }
                Some(bits) => Err(format!(
                    "native MSB-first byte-prefix abort requires zero observed bits, got {bits}"
                )),
                None => Ok(false),
            },
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                runtime.abort_empty_native_msb_byte_prefix()
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                ..
            } => {
                if !core.byte_is_active() {
                    return Ok(false);
                }
                core.validate_empty_byte()?;
                bitwise.abort_empty(base.as_mut())?;
                core.abort_empty_byte()?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn abandon_incomplete_native_msb_byte_prefix_for_lifecycle(
        &mut self,
    ) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                native_prefix_progress,
                ..
            }
            | RateBackendPredictor::FacCtw {
                native_prefix_progress,
                ..
            } => {
                *native_prefix_progress = None;
                Ok(())
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                let _ = runtime.abort_empty_native_msb_byte_prefix();
                Ok(())
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                core,
                bitwise,
                valid,
                ..
            } => {
                core.rollback_incomplete_byte()?;
                *bitwise = BytePrefixStepState::new();
                *valid = false;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn condition_native_msb_prefix_bit_for_rollback(
        &mut self,
        bit_idx: usize,
        bit: bool,
    ) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                core.condition_prefix_bit_for_rollback(bit_idx, bit)?;
                bitwise.observe(base.as_mut(), bit_idx, bit)?;
                *valid = false;
                Ok(())
            }
            _ => <Self as OnlineBytePredictor>::observe_native_msb_prefix_bit(self, bit_idx, bit),
        }
    }

    pub(crate) fn restore_checkpoint(&mut self, checkpoint: &RateBackendPredictorCheckpoint) {
        match (self, checkpoint) {
            #[cfg(feature = "backend-rosa")]
            (
                RateBackendPredictor::Rosa {
                    model,
                    checkpoint_journal,
                    ..
                },
                RateBackendPredictorCheckpoint::Rosa { journal_len },
            ) => {
                while checkpoint_journal.len() > *journal_len {
                    match checkpoint_journal
                        .pop()
                        .expect("rosa checkpoint journal underflow")
                    {
                        RosaPredictorUndo::Learned(tx) => model.rollback_tx(*tx),
                        RosaPredictorUndo::FrozenCursor { previous_last } => {
                            model.restore_conditioning_cursor(previous_last)
                        }
                    }
                }
            }
            #[cfg(feature = "backend-sequitur")]
            (
                RateBackendPredictor::Sequitur { model, .. },
                RateBackendPredictorCheckpoint::Sequitur(ck),
            ) => {
                model.restore(ck);
            }
            #[cfg(feature = "backend-ctw")]
            (
                RateBackendPredictor::Ctw {
                    tree,
                    bits_per_symbol,
                    checkpoint_journal,
                    native_prefix_progress,
                    ..
                },
                RateBackendPredictorCheckpoint::Ctw {
                    journal_len,
                    native_prefix_progress: checkpoint_progress,
                },
            ) => {
                restore_ctw_checkpoint(tree, *bits_per_symbol, checkpoint_journal, *journal_len);
                *native_prefix_progress = *checkpoint_progress;
            }
            #[cfg(feature = "backend-ctw")]
            (
                RateBackendPredictor::FacCtw {
                    tree,
                    bits_per_symbol,
                    checkpoint_journal,
                    native_prefix_progress,
                    ..
                },
                RateBackendPredictorCheckpoint::FacCtw {
                    journal_len,
                    native_prefix_progress: checkpoint_progress,
                },
            ) => {
                restore_fac_ctw_checkpoint(
                    tree,
                    *bits_per_symbol,
                    checkpoint_journal,
                    *journal_len,
                );
                *native_prefix_progress = *checkpoint_progress;
            }
            #[cfg(feature = "backend-calibrated")]
            (
                RateBackendPredictor::Calibrated {
                    base,
                    core,
                    bitwise,
                    pdf,
                    valid,
                    ..
                },
                RateBackendPredictorCheckpoint::Calibrated(ck),
            ) => {
                base.restore_checkpoint(&ck.base);
                *core = ck.core.clone();
                *bitwise = ck.bitwise.clone();
                *pdf = ck.pdf;
                *valid = ck.valid;
            }
            #[cfg(feature = "backend-calibrated")]
            (
                RateBackendPredictor::Calibrated {
                    base,
                    core,
                    bitwise,
                    valid,
                    ..
                },
                RateBackendPredictorCheckpoint::CalibratedNativePrefixStart {
                    base: checkpoint_base,
                    base_prefix_active,
                },
            ) => {
                core.rollback_incomplete_byte().unwrap_or_else(|err| {
                    panic!("calibrated native-prefix start restore failed: {err}")
                });
                *bitwise = BytePrefixStepState::new();
                *valid = false;
                base.restore_checkpoint(checkpoint_base);
                if *base_prefix_active {
                    base.abort_empty_native_msb_byte_prefix()
                        .unwrap_or_else(|err| {
                            panic!("calibrated base native-prefix abort failed: {err}")
                        });
                }
            }
            #[cfg(feature = "backend-mixture")]
            (
                RateBackendPredictor::Mixture { runtime },
                RateBackendPredictorCheckpoint::Mixture(ck),
            ) => {
                runtime.restore_checkpoint(ck);
            }
            (slot, RateBackendPredictorCheckpoint::Full(state)) => {
                *slot = state.as_ref().clone();
            }
            #[cfg(feature = "backend-rosa")]
            (_, RateBackendPredictorCheckpoint::Rosa { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-ctw")]
            (_, RateBackendPredictorCheckpoint::Ctw { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-ctw")]
            (_, RateBackendPredictorCheckpoint::FacCtw { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-calibrated")]
            (_, RateBackendPredictorCheckpoint::Calibrated(_)) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-calibrated")]
            (_, RateBackendPredictorCheckpoint::CalibratedNativePrefixStart { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-mixture")]
            (_, RateBackendPredictorCheckpoint::Mixture(_)) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-sequitur")]
            (_, RateBackendPredictorCheckpoint::Sequitur(_)) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
        }
    }

    pub(crate) fn clear_checkpoints_if_supported(&mut self) {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                checkpoint_journal,
                checkpoint_depth,
                ..
            } => {
                checkpoint_journal.clear();
                *checkpoint_depth = 0;
            }
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => model.clear_checkpoints(),
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                checkpoint_journal.clear();
                *checkpoint_depth = 0;
                *native_prefix_progress = None;
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                checkpoint_journal.clear();
                *checkpoint_depth = 0;
                *native_prefix_progress = None;
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated { base, .. } => {
                base.clear_checkpoints_if_supported();
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                runtime.clear_checkpoints_if_supported();
            }
            _ => {}
        }
    }

    pub(crate) fn discard_checkpoint(&mut self, checkpoint: RateBackendPredictorCheckpoint) {
        match (self, checkpoint) {
            #[cfg(feature = "backend-rosa")]
            (
                RateBackendPredictor::Rosa {
                    checkpoint_depth, ..
                },
                RateBackendPredictorCheckpoint::Rosa { .. },
            ) => {
                *checkpoint_depth = checkpoint_depth.saturating_sub(1);
            }
            #[cfg(feature = "backend-sequitur")]
            (
                RateBackendPredictor::Sequitur { .. },
                RateBackendPredictorCheckpoint::Sequitur(_),
            ) => {}
            #[cfg(feature = "backend-ctw")]
            (
                RateBackendPredictor::Ctw {
                    checkpoint_depth, ..
                },
                RateBackendPredictorCheckpoint::Ctw { .. },
            ) => {
                *checkpoint_depth = checkpoint_depth.saturating_sub(1);
            }
            #[cfg(feature = "backend-ctw")]
            (
                RateBackendPredictor::FacCtw {
                    checkpoint_depth, ..
                },
                RateBackendPredictorCheckpoint::FacCtw { .. },
            ) => {
                *checkpoint_depth = checkpoint_depth.saturating_sub(1);
            }
            #[cfg(feature = "backend-calibrated")]
            (
                RateBackendPredictor::Calibrated { base, .. },
                RateBackendPredictorCheckpoint::Calibrated(checkpoint),
            ) => {
                base.discard_checkpoint(*checkpoint.base);
            }
            #[cfg(feature = "backend-calibrated")]
            (
                RateBackendPredictor::Calibrated { base, .. },
                RateBackendPredictorCheckpoint::CalibratedNativePrefixStart {
                    base: checkpoint_base,
                    ..
                },
            ) => {
                base.discard_checkpoint(*checkpoint_base);
            }
            #[cfg(feature = "backend-mixture")]
            (
                RateBackendPredictor::Mixture { runtime },
                RateBackendPredictorCheckpoint::Mixture(checkpoint),
            ) => {
                runtime.discard_checkpoint(*checkpoint);
            }
            (_, RateBackendPredictorCheckpoint::Full(_)) => {}
            #[cfg(feature = "backend-rosa")]
            (_, RateBackendPredictorCheckpoint::Rosa { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-ctw")]
            (_, RateBackendPredictorCheckpoint::Ctw { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-ctw")]
            (_, RateBackendPredictorCheckpoint::FacCtw { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-calibrated")]
            (_, RateBackendPredictorCheckpoint::Calibrated(_)) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-calibrated")]
            (_, RateBackendPredictorCheckpoint::CalibratedNativePrefixStart { .. }) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-mixture")]
            (_, RateBackendPredictorCheckpoint::Mixture(_)) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
            #[cfg(feature = "backend-sequitur")]
            (_, RateBackendPredictorCheckpoint::Sequitur(_)) => {
                panic!("mismatched RateBackendPredictor checkpoint variant")
            }
        }
    }
}

impl OnlineBytePredictor for RateBackendPredictor {
    fn supports_frozen_reset(&self) -> bool {
        match self {
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { .. } => false,
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.supports_frozen_reset(),
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated { base, .. } => base.supports_frozen_reset(),
            _ => true,
        }
    }

    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.begin_fresh_stream(total_symbols),
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                pdf,
                valid,
                ..
            } => {
                base.begin_fresh_stream(total_symbols)?;
                reset_calibrated_wrapper_state(core, bitwise, pdf, valid);
                Ok(())
            }
            _ => {
                if self.supports_frozen_reset() {
                    self.reset_frozen(total_symbols)
                } else {
                    self.begin_stream(total_symbols)
                }
            }
        }
    }

    fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        self.finish_stream()?;
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                model,
                checkpoint_depth,
                ..
            } => {
                if *checkpoint_depth > 0 {
                    return Err(
                        "rosa lifecycle reset cannot run while prediction checkpoints are active"
                            .to_string(),
                    );
                }
                if let Some(total) = total_symbols {
                    let reserve = usize::try_from(total).unwrap_or(usize::MAX / 4);
                    model.reserve_for_stream(reserve);
                }
                Ok(())
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { .. } => Ok(()),
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { .. } => Ok(()),
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { .. } => Ok(()),
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => {
                if model.checkpoints_active() {
                    return Err(
                        "sequitur lifecycle begin_stream cannot run while prediction checkpoints are active"
                            .to_string(),
                    );
                }
                model.begin_stream(total_symbols);
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                native_prefix_progress,
                ..
            } => {
                *native_prefix_progress = None;
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                native_prefix_progress,
                ..
            } => {
                *native_prefix_progress = None;
                Ok(())
            }
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { model } => {
                model.begin_stream();
                Ok(())
            }
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { .. } => Ok(()),
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 { compressor, .. } => compressor
                .begin_online_policy_stream(total_symbols)
                .map_err(|e| e.to_string()),
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba { compressor, .. } => compressor
                .begin_online_policy_stream(total_symbols)
                .map_err(|e| e.to_string()),
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.begin_stream(total_symbols),
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                bitwise,
                valid,
                ..
            } => {
                *bitwise = BytePrefixStepState::new();
                *valid = false;
                base.begin_stream(total_symbols)
            }
            RateBackendPredictor::Disabled { reason } => Err(reason.clone()),
        }
    }

    fn finish_stream(&mut self) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa { .. } => Ok(()),
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { .. } => Ok(()),
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { .. } => Ok(()),
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { .. } => Ok(()),
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                native_prefix_progress,
                ..
            } => {
                *native_prefix_progress = None;
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                native_prefix_progress,
                ..
            } => {
                *native_prefix_progress = None;
                Ok(())
            }
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { .. } => Ok(()),
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { .. } => Ok(()),
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => {
                model.finish_stream();
                Ok(())
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 { compressor, .. } => compressor
                .finish_online_policy_stream()
                .map_err(|e| e.to_string()),
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba { compressor, .. } => compressor
                .finish_online_policy_stream()
                .map_err(|e| e.to_string()),
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.finish_stream(),
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                bitwise,
                valid,
                ..
            } => {
                *bitwise = BytePrefixStepState::new();
                *valid = false;
                base.finish_stream()
            }
            RateBackendPredictor::Disabled { .. } => Ok(()),
        }
    }

    fn checkpoint_if_supported(&mut self) -> Option<OnlineBytePredictorCheckpoint> {
        Some(OnlineBytePredictorCheckpoint::rate_backend(
            self.checkpoint(),
        ))
    }

    fn restore_checkpoint_if_supported(
        &mut self,
        checkpoint: &OnlineBytePredictorCheckpoint,
    ) -> bool {
        match &checkpoint.0 {
            OnlineBytePredictorCheckpointKind::RateBackend(checkpoint) => {
                self.restore_checkpoint(checkpoint);
                true
            }
        }
    }

    fn discard_checkpoint_if_supported(
        &mut self,
        checkpoint: OnlineBytePredictorCheckpoint,
    ) -> bool {
        match checkpoint.0 {
            OnlineBytePredictorCheckpointKind::RateBackend(checkpoint) => {
                self.discard_checkpoint(checkpoint);
                true
            }
        }
    }

    fn clear_checkpoints_if_supported(&mut self) {
        RateBackendPredictor::clear_checkpoints_if_supported(self);
    }

    fn log_prob(&mut self, symbol: u8) -> f64 {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                model, min_prob, ..
            } => {
                let p = clamp_prob(model.prob_for_last(symbol as u32), *min_prob);
                p.ln()
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { model, min_prob } => model.log_prob(symbol, *min_prob),
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { model, min_prob } => {
                model.log_prob(symbol, *min_prob)
            }
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { model, min_prob } => model.log_prob(symbol, *min_prob),
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, min_prob } => model.log_prob(symbol, *min_prob),
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                bits_per_symbol,
                min_prob,
                ..
            } => ctw_log_prob_msb(tree, symbol, *bits_per_symbol, *min_prob),
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                msb_first,
                min_prob,
                ..
            } => {
                let log_before = tree.get_log_block_probability();
                for i in 0..*bits_per_symbol {
                    let bit = if *msb_first {
                        ctw_symbol_bit_msb(symbol, *bits_per_symbol, i)
                    } else {
                        ((symbol >> i) & 1) == 1
                    };
                    tree.update(bit, i);
                }
                let log_after = tree.get_log_block_probability();
                for i in (0..*bits_per_symbol).rev() {
                    tree.revert(i);
                }
                let logp = log_after - log_before;
                if logp.is_finite() {
                    logp.max(min_prob.ln())
                } else {
                    min_prob.ln()
                }
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                ensure_rwkv_primed(compressor, primed);
                let p = clamp_prob(compressor.pdf_buffer[symbol as usize], *min_prob);
                p.ln()
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    mambazip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                let p = clamp_prob(compressor.pdf_buffer[symbol as usize], *min_prob);
                p.ln()
            }
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { model } => model.log_prob(symbol),
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.peek_log_prob(symbol),
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { runtime } => runtime.peek_log_prob(symbol),
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise: _,
                pdf,
                valid,
                min_prob,
            } => {
                if !*valid {
                    let mut base_logps = [0.0; 256];
                    base.fill_log_probs(&mut base_logps);
                    let mut base_pdf = [0.0; 256];
                    for (dst, &lp) in base_pdf.iter_mut().zip(base_logps.iter()) {
                        *dst = clamp_prob(lp.exp(), *min_prob);
                    }
                    core.apply_pdf(&base_pdf, pdf);
                    *valid = true;
                }
                pdf[symbol as usize].max(*min_prob).ln()
            }
            RateBackendPredictor::Disabled { .. } => f64::NEG_INFINITY,
        }
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                model, min_prob, ..
            } => {
                model.fill_probs_for_last_bytes(out);
                for slot in out.iter_mut() {
                    *slot = clamp_prob(*slot, *min_prob).ln();
                }
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { model, min_prob } => {
                let mut pdf = [0.0; 256];
                model.fill_pdf(&mut pdf);
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { model, min_prob } => {
                let mut pdf = [0.0; 256];
                model.fill_pdf(&mut pdf);
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { model, min_prob } => {
                let mut pdf = [0.0; 256];
                model.fill_pdf(&mut pdf);
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, min_prob } => {
                let mut pdf = [0.0; 256];
                model.fill_pdf(&mut pdf);
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                bits_per_symbol,
                min_prob,
                ..
            } => fill_ctw_tree_log_probs(tree, *bits_per_symbol, min_prob.ln(), out),
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                msb_first,
                min_prob,
                ..
            } => {
                fill_fac_tree_log_probs(tree, *bits_per_symbol, *msb_first, min_prob.ln(), out);
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                ensure_rwkv_primed(compressor, primed);
                for (slot, &p_raw) in out
                    .iter_mut()
                    .take(256)
                    .zip(compressor.pdf_buffer.iter().take(256))
                {
                    let p = clamp_prob(p_raw, *min_prob);
                    *slot = p.ln();
                }
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    mambazip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                for (slot, &p_raw) in out
                    .iter_mut()
                    .take(256)
                    .zip(compressor.pdf_buffer.iter().take(256))
                {
                    let p = clamp_prob(p_raw, *min_prob);
                    *slot = p.ln();
                }
            }
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { model } => {
                model.fill_log_probs(out);
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                runtime.fill_log_probs(out);
            }
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { runtime } => {
                runtime.fill_log_probs_cached(out);
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise: _,
                pdf,
                valid,
                min_prob,
            } => {
                if !*valid {
                    let mut base_logps = [0.0; 256];
                    base.fill_log_probs(&mut base_logps);
                    let mut base_pdf = [0.0; 256];
                    for (dst, &lp) in base_pdf.iter_mut().zip(base_logps.iter()) {
                        *dst = clamp_prob(lp.exp(), *min_prob);
                    }
                    core.apply_pdf(&base_pdf, pdf);
                    *valid = true;
                }
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            RateBackendPredictor::Disabled { .. } => out.fill(-(256.0f64).ln()),
        }
    }

    fn has_native_msb_byte_prefix(&self) -> bool {
        match self {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                bits_per_symbol, ..
            } => *bits_per_symbol == 8,
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                bits_per_symbol,
                msb_first,
                ..
            } => *bits_per_symbol == 8 && *msb_first,
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.has_native_msb_byte_prefix(),
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated { .. } => true,
            _ => false,
        }
    }

    fn begin_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
        match self {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                bits_per_symbol, ..
            } if *bits_per_symbol != 8 => Ok(false),
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                bits_per_symbol,
                msb_first,
                ..
            } if *bits_per_symbol != 8 || !*msb_first => Ok(false),
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                native_prefix_progress,
                ..
            } => {
                if native_prefix_progress.is_some() {
                    return Err(
                        "native MSB-first byte-prefix step is already active for this predictor"
                            .to_string(),
                    );
                }
                *native_prefix_progress = Some(0);
                Ok(true)
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                native_prefix_progress,
                ..
            } => {
                if native_prefix_progress.is_some() {
                    return Err(
                        "native MSB-first byte-prefix step is already active for this predictor"
                            .to_string(),
                    );
                }
                *native_prefix_progress = Some(0);
                Ok(true)
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.begin_native_msb_byte_prefix(),
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                core.begin_byte()?;
                if let Err(err) = bitwise.prepare(base.as_mut()) {
                    let _ = core.abort_empty_byte();
                    return Err(err);
                }
                *valid = false;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn abort_empty_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
        RateBackendPredictor::abort_empty_native_msb_byte_prefix(self)
    }

    fn native_msb_prefix_prob_one(&mut self, bit_idx: usize) -> Result<f64, String> {
        match self {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                min_prob,
                native_prefix_progress,
                ..
            } => {
                let next_bit_idx = native_prefix_progress
                    .as_ref()
                    .ok_or_else(|| "native MSB-first byte-prefix step is not active".to_string())?;
                validate_native_msb_prefix_bit_idx(*next_bit_idx, bit_idx)?;
                let p: f64 = tree.predict(true);
                Ok(p.clamp(*min_prob, 1.0 - *min_prob))
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                min_prob,
                native_prefix_progress,
                ..
            } => {
                let next_bit_idx = native_prefix_progress
                    .as_ref()
                    .ok_or_else(|| "native MSB-first byte-prefix step is not active".to_string())?;
                validate_native_msb_prefix_bit_idx(*next_bit_idx, bit_idx)?;
                let p: f64 = tree.predict_one(bit_idx);
                Ok(p.clamp(*min_prob, 1.0 - *min_prob))
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                runtime.native_msb_prefix_prob_one(bit_idx)
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                ..
            } => {
                let base_p1: f64 = bitwise.prob_one(base.as_mut(), bit_idx)?;
                core.predict_bit(bit_idx, base_p1)
            }
            _ => Err("native MSB-first byte-prefix prediction is unavailable".to_string()),
        }
    }

    fn observe_native_msb_prefix_bit(&mut self, bit_idx: usize, bit: bool) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                let next_bit_idx = native_prefix_progress
                    .as_mut()
                    .ok_or_else(|| "native MSB-first byte-prefix step is not active".to_string())?;
                validate_native_msb_prefix_bit_idx(*next_bit_idx, bit_idx)?;
                *next_bit_idx += 1;
                tree.update(bit);
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(CtwUndoOp::LearnedBit);
                }
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                let next_bit_idx = native_prefix_progress
                    .as_mut()
                    .ok_or_else(|| "native MSB-first byte-prefix step is not active".to_string())?;
                validate_native_msb_prefix_bit_idx(*next_bit_idx, bit_idx)?;
                *next_bit_idx += 1;
                tree.update_predicted(bit, bit_idx);
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(FacCtwUndoOp::LearnedBit { bit_idx });
                }
                Ok(())
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                runtime.observe_native_msb_prefix_bit(bit_idx, bit)
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                let base_p1: f64 = bitwise.prob_one(base.as_mut(), bit_idx)?;
                core.observe_bit_from_base(bit_idx, base_p1, bit)?;
                bitwise.observe(base.as_mut(), bit_idx, bit)?;
                *valid = false;
                Ok(())
            }
            _ => Err("native MSB-first byte-prefix stepping is unavailable".to_string()),
        }
    }

    fn finish_native_msb_byte_prefix(&mut self, symbol: u8) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                native_prefix_progress,
                ..
            } => {
                let next_bit_idx = native_prefix_progress
                    .as_ref()
                    .ok_or_else(|| "native MSB-first byte-prefix step is not active".to_string())?;
                validate_native_msb_prefix_finish(*next_bit_idx)?;
                *native_prefix_progress = None;
                let _ = symbol;
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                native_prefix_progress,
                ..
            } => {
                let next_bit_idx = native_prefix_progress
                    .as_ref()
                    .ok_or_else(|| "native MSB-first byte-prefix step is not active".to_string())?;
                validate_native_msb_prefix_finish(*next_bit_idx)?;
                *native_prefix_progress = None;
                let _ = symbol;
                Ok(())
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                runtime.finish_native_msb_byte_prefix(symbol)
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                core.validate_complete_byte()?;
                bitwise.finish(base.as_mut(), symbol)?;
                core.finish_byte()?;
                *valid = false;
                Ok(())
            }
            _ => Err("native MSB-first byte-prefix stepping is unavailable".to_string()),
        }
    }

    fn update(&mut self, symbol: u8) {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                model,
                checkpoint_journal,
                checkpoint_depth,
                ..
            } => {
                if *checkpoint_depth > 0 {
                    let mut tx = model.begin_tx();
                    model.train_sequence_tx(&mut tx, &[symbol]);
                    checkpoint_journal.push(RosaPredictorUndo::Learned(Box::new(tx)));
                } else {
                    model.train_byte(symbol);
                }
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { model, .. } => {
                model.update(symbol);
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { model, .. } => {
                model.update(symbol);
            }
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { model, .. } => {
                model.update(symbol);
            }
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => {
                model.update(symbol);
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                bits_per_symbol,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                debug_assert!(
                    native_prefix_progress.is_none(),
                    "ctw symbol update while native byte-prefix step is active"
                );
                *native_prefix_progress = None;
                for bit_idx in 0..(*bits_per_symbol).clamp(1, 8) {
                    tree.update(ctw_symbol_bit_msb(symbol, *bits_per_symbol, bit_idx));
                }
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(CtwUndoOp::LearnedSymbol);
                }
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                msb_first,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                debug_assert!(
                    native_prefix_progress.is_none(),
                    "fac-ctw symbol update while native byte-prefix step is active"
                );
                *native_prefix_progress = None;
                for i in 0..*bits_per_symbol {
                    let bit = if *msb_first {
                        ctw_symbol_bit_msb(symbol, *bits_per_symbol, i)
                    } else {
                        ((symbol >> i) & 1) == 1
                    };
                    tree.update(bit, i);
                }
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(FacCtwUndoOp::LearnedSymbol);
                }
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor, primed, ..
            } => {
                ensure_rwkv_primed(compressor, primed);
                compressor
                    .observe_symbol_from_current_pdf(symbol)
                    .unwrap_or_else(|e| panic!("rwkv online update failed: {e}"));
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor,
                primed,
                pdf_scratch,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    mambazip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                if pdf_scratch.len() != compressor.pdf_buffer.len() {
                    pdf_scratch.resize(compressor.pdf_buffer.len(), 0.0);
                }
                pdf_scratch.copy_from_slice(&compressor.pdf_buffer);
                compressor
                    .online_update_from_pdf(symbol, pdf_scratch)
                    .unwrap_or_else(|e| panic!("mamba online update failed: {e}"));
                let bias = compressor.online_bias_snapshot();
                let logits = compressor.model.forward(
                    &mut compressor.scratch,
                    symbol as u32,
                    &mut compressor.state,
                );
                mambazip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
            }
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { model } => {
                model.update(symbol);
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                let _ = runtime.step(symbol);
            }
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { runtime } => {
                runtime.step(symbol);
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                valid,
                min_prob,
                ..
            } => {
                let mut base_logps = [0.0; 256];
                base.fill_log_probs(&mut base_logps);
                let mut base_pdf = [0.0; 256];
                for (dst, &lp) in base_pdf.iter_mut().zip(base_logps.iter()) {
                    *dst = clamp_prob(lp.exp(), *min_prob);
                }
                core.observe_symbol_from_base_pdf(symbol, &base_pdf)
                    .unwrap_or_else(|err| {
                        panic!("calibrated SSE byte update violated prefix protocol: {err}")
                    });
                base.update(symbol);
                *valid = false;
            }
            RateBackendPredictor::Disabled { .. } => {}
        }
    }

    fn log_prob_update(&mut self, symbol: u8) -> f64 {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                model,
                min_prob,
                checkpoint_journal,
                checkpoint_depth,
            } => {
                let p = clamp_prob(model.prob_for_last(symbol as u32), *min_prob);
                if *checkpoint_depth > 0 {
                    let mut tx = model.begin_tx();
                    model.train_sequence_tx(&mut tx, &[symbol]);
                    checkpoint_journal.push(RosaPredictorUndo::Learned(Box::new(tx)));
                } else {
                    model.train_byte(symbol);
                }
                p.ln()
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                bits_per_symbol,
                min_prob,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
            } => {
                debug_assert!(
                    native_prefix_progress.is_none(),
                    "ctw symbol log_prob_update while native byte-prefix step is active"
                );
                *native_prefix_progress = None;
                let logp = ctw_log_prob_update_msb(tree, symbol, *bits_per_symbol, *min_prob);
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(CtwUndoOp::LearnedSymbol);
                }
                logp
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                msb_first,
                min_prob,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
            } => {
                debug_assert!(
                    native_prefix_progress.is_none(),
                    "fac-ctw symbol log_prob_update while native byte-prefix step is active"
                );
                *native_prefix_progress = None;
                let logp = if *msb_first {
                    let bits = (*bits_per_symbol).clamp(1, 8);
                    let mut acc = 0.0f64;
                    for bit_idx in 0..bits {
                        let bit = ctw_symbol_bit_msb(symbol, bits, bit_idx);
                        let p = tree.predict(bit, bit_idx).clamp(*min_prob, 1.0 - *min_prob);
                        acc += p.ln();
                        tree.update_predicted(bit, bit_idx);
                    }
                    acc
                } else {
                    ctw_log_prob_update_lsb(tree, symbol, *bits_per_symbol, *min_prob)
                };
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(FacCtwUndoOp::LearnedSymbol);
                }
                logp
            }
            _ => {
                let logp = self.log_prob(symbol);
                self.update(symbol);
                logp
            }
        }
    }

    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        #[cfg(feature = "backend-zpaq")]
        if matches!(self, RateBackendPredictor::Zpaq { .. }) {
            return Err("plugin entropy is not supported for zpaq rate backends".to_string());
        }
        #[cfg(feature = "backend-mixture")]
        if let RateBackendPredictor::Mixture { runtime } = self
            && !runtime.supports_frozen_reset()
        {
            return Err(
                "plugin entropy is not supported for mixture rate backends with non-resettable experts"
                    .to_string(),
            );
        }
        #[cfg(feature = "backend-calibrated")]
        if let RateBackendPredictor::Calibrated { base, .. } = self
            && !base.supports_frozen_reset()
        {
            return base.reset_frozen(total_symbols);
        }

        self.finish_stream()?;
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa { model, .. } => {
                if let Some(total) = total_symbols {
                    let reserve = usize::try_from(total).unwrap_or(usize::MAX / 4);
                    model.reserve_for_stream(reserve);
                }
                model.build_lm_full_bytes_no_finalize_endpos();
                model.reset_conditioning_cursor();
                Ok(())
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { model, .. } => {
                model.reset_history();
                Ok(())
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { model, .. } => {
                model.reset_history();
                Ok(())
            }
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { model, .. } => {
                model.reset_history();
                Ok(())
            }
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => {
                if model.checkpoints_active() {
                    return Err(
                        "sequitur lifecycle reset cannot run while prediction checkpoints are active"
                            .to_string(),
                    );
                }
                model.reset_frozen();
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                if *checkpoint_depth > 0 {
                    return Err(
                        "ctw lifecycle reset cannot run while prediction checkpoints are active"
                            .to_string(),
                    );
                }
                *native_prefix_progress = None;
                tree.truncate_history(0);
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                if *checkpoint_depth > 0 {
                    return Err(
                        "fac-ctw lifecycle reset cannot run while prediction checkpoints are active"
                            .to_string(),
                    );
                }
                *native_prefix_progress = None;
                tree.reset_history_only();
                Ok(())
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor, primed, ..
            } => {
                compressor.reset_and_prime();
                *primed = true;
                Ok(())
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor, primed, ..
            } => {
                compressor.reset_and_prime();
                *primed = true;
                Ok(())
            }
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { .. } => {
                Err("plugin entropy is not supported for zpaq rate backends".to_string())
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => runtime.reset_frozen(total_symbols),
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { runtime } => {
                runtime.reset_frozen_state();
                Ok(())
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                pdf,
                valid,
                ..
            } => {
                base.reset_frozen(total_symbols)?;
                reset_calibrated_wrapper_state(core, bitwise, pdf, valid);
                Ok(())
            }
            RateBackendPredictor::Disabled { reason } => Err(reason.clone()),
        }
    }

    fn update_frozen(&mut self, symbol: u8) {
        match self {
            #[cfg(feature = "backend-rosa")]
            RateBackendPredictor::Rosa {
                model,
                checkpoint_journal,
                checkpoint_depth,
                ..
            } => {
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(RosaPredictorUndo::FrozenCursor {
                        previous_last: model.conditioning_cursor(),
                    });
                }
                model.advance_conditioning_byte(symbol);
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::Match { model, .. } => {
                model.update_history_only(symbol);
            }
            #[cfg(feature = "backend-match")]
            RateBackendPredictor::SparseMatch { model, .. } => {
                model.update_history_only(symbol);
            }
            #[cfg(feature = "backend-ppmd")]
            RateBackendPredictor::Ppmd { model, .. } => {
                model.update_history_only(symbol);
            }
            #[cfg(feature = "backend-sequitur")]
            RateBackendPredictor::Sequitur { model, .. } => {
                model.update_frozen(symbol);
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                tree,
                bits_per_symbol,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                debug_assert!(
                    native_prefix_progress.is_none(),
                    "ctw frozen symbol update while native byte-prefix step is active"
                );
                *native_prefix_progress = None;
                let bits = (*bits_per_symbol).clamp(1, 8);
                let mut history_bits = [false; 8];
                for (bit_idx, slot) in history_bits.iter_mut().enumerate().take(bits) {
                    *slot = ctw_symbol_bit_msb(symbol, bits, bit_idx);
                }
                tree.update_history(&history_bits[..bits]);
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(CtwUndoOp::FrozenSymbol);
                }
            }
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                msb_first,
                checkpoint_journal,
                checkpoint_depth,
                native_prefix_progress,
                ..
            } => {
                debug_assert!(
                    native_prefix_progress.is_none(),
                    "fac-ctw frozen symbol update while native byte-prefix step is active"
                );
                *native_prefix_progress = None;
                let bits = (*bits_per_symbol).clamp(1, 8);
                let mut history_bits = [false; 8];
                for (idx, slot) in history_bits.iter_mut().enumerate().take(bits) {
                    *slot = if *msb_first {
                        ctw_symbol_bit_msb(symbol, bits, idx)
                    } else {
                        ((symbol >> idx) & 1) == 1
                    };
                }
                tree.update_history(&history_bits[..bits]);
                if *checkpoint_depth > 0 {
                    checkpoint_journal.push(FacCtwUndoOp::FrozenSymbol);
                }
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor, primed, ..
            } => {
                if !*primed {
                    compressor.reset_and_prime();
                    *primed = true;
                }
                compressor.forward_to_internal_pdf(symbol as u32);
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor, primed, ..
            } => {
                if !*primed {
                    compressor.reset_and_prime();
                    *primed = true;
                }
                let bias = compressor.online_bias_snapshot();
                let logits = compressor.model.forward(
                    &mut compressor.scratch,
                    symbol as u32,
                    &mut compressor.state,
                );
                mambazip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
            }
            #[cfg(feature = "backend-zpaq")]
            RateBackendPredictor::Zpaq { model } => {
                model.update(symbol);
            }
            #[cfg(feature = "backend-mixture")]
            RateBackendPredictor::Mixture { runtime } => {
                runtime.update_frozen(symbol);
            }
            #[cfg(feature = "backend-particle")]
            RateBackendPredictor::Particle { runtime } => {
                runtime.update_frozen(symbol);
            }
            #[cfg(feature = "backend-calibrated")]
            RateBackendPredictor::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                base.update_frozen(symbol);
                core.update_context_only(symbol);
                *bitwise = BytePrefixStepState::new();
                *valid = false;
            }
            RateBackendPredictor::Disabled { .. } => {}
        }
    }
}

#[derive(Clone)]
enum ExpertPredictor {
    Generic(Box<dyn OnlineBytePredictor>),
    RateBackend(Box<RateBackendPredictor>),
}

impl ExpertPredictor {
    fn generic(predictor: Box<dyn OnlineBytePredictor>) -> Self {
        Self::Generic(predictor)
    }

    fn rate_backend(predictor: RateBackendPredictor) -> Self {
        Self::RateBackend(Box::new(predictor))
    }

    fn as_mut(&mut self) -> &mut (dyn OnlineBytePredictor + 'static) {
        match self {
            Self::Generic(predictor) => predictor.as_mut(),
            Self::RateBackend(predictor) => predictor.as_mut(),
        }
    }

    fn into_box(self) -> Box<dyn OnlineBytePredictor> {
        match self {
            Self::Generic(predictor) => predictor,
            Self::RateBackend(predictor) => predictor,
        }
    }

    fn lifecycle_checkpoint(&mut self, op: OnlineBytePredictorLifecycleOp) -> ExpertLifecycleToken {
        match self {
            Self::Generic(_) => ExpertLifecycleToken::Full(self.clone()),
            Self::RateBackend(predictor) => {
                ExpertLifecycleToken::Compact(Box::new(predictor.lifecycle_checkpoint(op)))
            }
        }
    }

    fn restore_lifecycle(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
        token: ExpertLifecycleToken,
    ) {
        match (self, token) {
            (slot, ExpertLifecycleToken::Full(predictor)) => {
                *slot = predictor;
            }
            (Self::RateBackend(predictor), ExpertLifecycleToken::Compact(checkpoint)) => {
                predictor.restore_lifecycle_checkpoint(op, *checkpoint);
            }
            (Self::Generic(_), ExpertLifecycleToken::Compact(_)) => {
                panic!("generic expert received a compact rate-backend lifecycle checkpoint")
            }
        }
    }

    fn discard_lifecycle(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
        token: ExpertLifecycleToken,
    ) {
        match (self, token) {
            (_, ExpertLifecycleToken::Full(_)) => {}
            (Self::RateBackend(predictor), ExpertLifecycleToken::Compact(checkpoint)) => {
                predictor.discard_lifecycle_checkpoint(op, *checkpoint);
            }
            (Self::Generic(_), ExpertLifecycleToken::Compact(_)) => {
                panic!("generic expert received a compact rate-backend lifecycle checkpoint")
            }
        }
    }
}

impl std::ops::Deref for ExpertPredictor {
    type Target = dyn OnlineBytePredictor + 'static;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Generic(predictor) => predictor.as_ref(),
            Self::RateBackend(predictor) => predictor.as_ref(),
        }
    }
}

impl std::ops::DerefMut for ExpertPredictor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

/// Configuration for a mixture expert.
#[derive(Clone)]
pub struct ExpertConfig {
    /// Human-readable expert identifier.
    pub name: String,
    /// Log prior weight (natural log). Uniform priors can be `0.0`.
    pub log_prior: f64,
    builder: Arc<dyn Fn() -> ExpertPredictor + Send + Sync>,
}

impl ExpertConfig {
    /// Create a new expert config from a builder closure.
    pub fn new(
        name: impl Into<String>,
        log_prior: f64,
        builder: impl Fn() -> Box<dyn OnlineBytePredictor> + Send + Sync + 'static,
    ) -> Self {
        Self::new_with_predictor(name, log_prior, move || ExpertPredictor::generic(builder()))
    }

    fn new_with_predictor(
        name: impl Into<String>,
        log_prior: f64,
        builder: impl Fn() -> ExpertPredictor + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            log_prior,
            builder: Arc::new(builder),
        }
    }

    fn new_rate_backend(
        name: impl Into<String>,
        log_prior: f64,
        builder: impl Fn() -> RateBackendPredictor + Send + Sync + 'static,
    ) -> Self {
        Self::new_with_predictor(name, log_prior, move || {
            ExpertPredictor::rate_backend(builder())
        })
    }

    /// Uniform prior helper.
    pub fn uniform(
        name: impl Into<String>,
        builder: impl Fn() -> Box<dyn OnlineBytePredictor> + Send + Sync + 'static,
    ) -> Self {
        Self::new(name, 0.0, builder)
    }

    /// Expert from a `RateBackend` configuration. ROSA's `max_order` lives inside
    /// the [`RateBackend::RosaPlus`] variant.
    pub fn from_rate_backend(name: Option<String>, log_prior: f64, backend: RateBackend) -> Self {
        let name = name.unwrap_or_else(|| RateBackendPredictor::default_name(&backend));
        Self::new_rate_backend(name, log_prior, move || {
            RateBackendPredictor::from_backend(backend.clone(), DEFAULT_MIN_PROB)
        })
    }

    /// Expert from a compiled rate backend plan.
    pub fn from_compiled_rate_backend(
        name: Option<String>,
        log_prior: f64,
        backend: CompiledRateBackend,
    ) -> Self {
        let name = name.unwrap_or_else(|| backend.display_label());
        Self::new_rate_backend(name, log_prior, move || {
            RateBackendPredictor::from_compiled(&backend, DEFAULT_MIN_PROB)
        })
    }

    /// ROSA expert (uniform prior) with explicit `max_order`.
    pub fn rosa(name: impl Into<String>, max_order: i64) -> Self {
        let name = name.into();
        Self::new_rate_backend(name, 0.0, move || {
            RateBackendPredictor::from_backend(
                RateBackend::RosaPlus { max_order },
                DEFAULT_MIN_PROB,
            )
        })
    }

    /// CTW expert (uniform prior).
    pub fn ctw(name: impl Into<String>, depth: usize) -> Self {
        let name = name.into();
        Self::new_rate_backend(name, 0.0, move || {
            RateBackendPredictor::from_backend(RateBackend::Ctw { depth }, DEFAULT_MIN_PROB)
        })
    }

    /// FAC-CTW expert (uniform prior).
    pub fn fac_ctw(name: impl Into<String>, base_depth: usize, encoding_bits: usize) -> Self {
        let name = name.into();
        Self::new_rate_backend(name, 0.0, move || {
            RateBackendPredictor::from_backend(
                RateBackend::FacCtw {
                    base_depth,
                    num_percept_bits: encoding_bits,
                    encoding_bits,
                    msb_first: None,
                },
                DEFAULT_MIN_PROB,
            )
        })
    }

    /// RWKV-7 expert (uniform prior).
    #[cfg(feature = "backend-rwkv")]
    pub fn rwkv(name: impl Into<String>, method: impl Into<String>) -> Self {
        let name = name.into();
        let method = crate::rwkvzip::parse_method_spec(&method.into())
            .expect("rwkv expert method must be a valid RWKV method spec");
        Self::new_rate_backend(name, 0.0, move || {
            RateBackendPredictor::from_backend(
                RateBackend::Rwkv7Method {
                    method: method.clone(),
                },
                DEFAULT_MIN_PROB,
            )
        })
    }

    /// Mamba expert (uniform prior).
    #[cfg(feature = "backend-mamba")]
    pub fn mamba(name: impl Into<String>, method: impl Into<String>) -> Self {
        let name = name.into();
        let method = crate::mambazip::parse_method_spec(&method.into())
            .expect("mamba expert method must be a valid Mamba method spec");
        Self::new_rate_backend(name, 0.0, move || {
            RateBackendPredictor::from_backend(
                RateBackend::MambaMethod {
                    method: method.clone(),
                },
                DEFAULT_MIN_PROB,
            )
        })
    }

    /// ZPAQ expert (uniform prior).
    pub fn zpaq(name: impl Into<String>, method: impl Into<String>) -> Self {
        let name = name.into();
        let method = crate::api::ZpaqMethodSpec::literal(method.into());
        Self::new_rate_backend(name, 0.0, move || {
            RateBackendPredictor::from_backend(
                RateBackend::Zpaq {
                    method: method.clone(),
                },
                DEFAULT_MIN_PROB,
            )
        })
    }

    /// Expert name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Log prior weight (unnormalized).
    pub fn log_prior(&self) -> f64 {
        self.log_prior
    }

    /// Build a fresh predictor instance for evaluation or analysis.
    pub fn build_predictor(&self) -> Box<dyn OnlineBytePredictor> {
        (self.builder)().into_box()
    }

    fn build(&self) -> ExpertState {
        ExpertState {
            name: self.name.clone(),
            log_weight: self.log_prior,
            log_prior: self.log_prior,
            predictor: (self.builder)(),
            cum_log_loss: 0.0,
        }
    }
}

#[cfg(feature = "backend-mixture")]
pub(crate) fn expert_configs_from_compiled_mixture(
    backend: &CompiledRateBackend,
) -> Result<Vec<ExpertConfig>, String> {
    let crate::spec::core::RateBackendPlan::Mixture { experts, .. } = backend.plan() else {
        return Err("compiled backend is not a mixture backend".to_string());
    };
    experts
        .iter()
        .map(|expert| {
            let compiled =
                crate::spec::core::compiled_rate_backend_from_plan(expert.backend.clone())
                    .map_err(|err| err.to_string())?;
            Ok(ExpertConfig::from_compiled_rate_backend(
                expert.name.clone(),
                expert.log_prior,
                compiled,
            ))
        })
        .collect::<Result<Vec<_>, String>>()
}

#[cfg(feature = "backend-mixture")]
pub(crate) fn expert_configs_from_compiled_mixture_with_builder(
    backend: &CompiledRateBackend,
    builder: fn(&CompiledRateBackend, f64) -> Result<RateBackendPredictor, String>,
    min_prob: f64,
) -> Result<Vec<ExpertConfig>, String> {
    let crate::spec::core::RateBackendPlan::Mixture { experts, .. } = backend.plan() else {
        return Err("compiled backend is not a mixture backend".to_string());
    };
    experts
        .iter()
        .map(|expert| {
            let compiled =
                crate::spec::core::compiled_rate_backend_from_plan(expert.backend.clone())
                    .map_err(|err| err.to_string())?;
            // Validate once up front so mixture construction fails before we
            // commit any `ExpertConfig` values. The stored builder still has to
            // create a fresh predictor later because each runtime needs its own
            // independent expert state.
            builder(&compiled, min_prob).map(|_| ())?;
            let name = expert
                .name
                .clone()
                .unwrap_or_else(|| compiled.default_name());
            Ok(ExpertConfig::new_rate_backend(
                name,
                expert.log_prior,
                move || {
                    builder(&compiled, min_prob)
                        .expect("compiled mixture expert builder should succeed")
                },
            ))
        })
        .collect::<Result<Vec<_>, String>>()
}

enum ExpertLifecycleToken {
    Full(ExpertPredictor),
    Compact(Box<RateBackendPredictorLifecycleCheckpoint>),
}

struct ExpertStateLifecycleCheckpoint {
    log_weight: f64,
    log_prior: f64,
    cum_log_loss: f64,
    predictor: ExpertLifecycleToken,
}

#[derive(Clone)]
struct ExpertState {
    name: String,
    log_weight: f64,
    log_prior: f64,
    predictor: ExpertPredictor,
    cum_log_loss: f64,
}

impl ExpertState {
    #[inline]
    fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        self.predictor.begin_stream(total_symbols)
    }

    #[inline]
    fn finish_stream(&mut self) -> Result<(), String> {
        self.predictor.finish_stream()
    }

    #[inline]
    fn log_prob(&mut self, symbol: u8) -> f64 {
        self.predictor.log_prob(symbol)
    }

    #[inline]
    fn log_prob_update(&mut self, symbol: u8) -> f64 {
        self.predictor.log_prob_update(symbol)
    }

    #[inline]
    fn update(&mut self, symbol: u8) {
        self.predictor.update(symbol);
    }

    #[inline]
    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        self.predictor.reset_frozen(total_symbols)
    }

    #[inline]
    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        self.predictor.begin_fresh_stream(total_symbols)
    }

    #[inline]
    fn update_frozen(&mut self, symbol: u8) {
        self.predictor.update_frozen(symbol);
    }

    fn snapshot_lifecycle(&mut self, op: ExpertLifecycleOp) -> ExpertLifecycleToken {
        self.predictor.lifecycle_checkpoint(op.to_predictor_op())
    }

    fn restore_lifecycle(&mut self, op: ExpertLifecycleOp, token: ExpertLifecycleToken) {
        self.predictor
            .restore_lifecycle(op.to_predictor_op(), token);
    }

    fn discard_lifecycle(&mut self, op: ExpertLifecycleOp, token: ExpertLifecycleToken) {
        self.predictor
            .discard_lifecycle(op.to_predictor_op(), token);
    }

    fn snapshot_state_lifecycle(
        &mut self,
        op: ExpertLifecycleOp,
    ) -> ExpertStateLifecycleCheckpoint {
        ExpertStateLifecycleCheckpoint {
            log_weight: self.log_weight,
            log_prior: self.log_prior,
            cum_log_loss: self.cum_log_loss,
            predictor: self.snapshot_lifecycle(op),
        }
    }

    fn restore_state_lifecycle(
        &mut self,
        op: ExpertLifecycleOp,
        checkpoint: ExpertStateLifecycleCheckpoint,
    ) {
        self.log_weight = checkpoint.log_weight;
        self.log_prior = checkpoint.log_prior;
        self.cum_log_loss = checkpoint.cum_log_loss;
        self.restore_lifecycle(op, checkpoint.predictor);
    }

    fn discard_state_lifecycle(
        &mut self,
        op: ExpertLifecycleOp,
        checkpoint: ExpertStateLifecycleCheckpoint,
    ) {
        self.discard_lifecycle(op, checkpoint.predictor);
    }
}

fn reset_expert_losses(experts: &mut [ExpertState]) {
    for expert in experts {
        expert.cum_log_loss = 0.0;
    }
}

fn reset_experts_to_priors(experts: &mut [ExpertState]) -> Vec<f64> {
    let prior = normalized_expert_prior_weights(experts);
    set_log_weights_from_linear(experts, &prior);
    reset_expert_losses(experts);
    prior
}

fn apply_bayes_update_from_logps(
    experts: &mut [ExpertState],
    expert_logps: &[f64],
    scratch_mix: &mut Vec<f64>,
) -> f64 {
    let n = experts.len();
    scratch_mix.resize(n, 0.0);
    for idx in 0..n {
        scratch_mix[idx] = experts[idx].log_weight + expert_logps[idx];
    }
    let log_mix = logsumexp(&scratch_mix[..n]);
    for idx in 0..n {
        experts[idx].cum_log_loss -= expert_logps[idx];
        experts[idx].log_weight += expert_logps[idx] - log_mix;
    }
    log_mix
}

fn apply_fading_update_from_logps(
    experts: &mut [ExpertState],
    expert_logps: &[f64],
    scratch_mix: &mut Vec<f64>,
    decay: f64,
) -> f64 {
    let n = experts.len();
    scratch_mix.resize(n, 0.0);
    for idx in 0..n {
        scratch_mix[idx] = decay * experts[idx].log_weight;
    }
    let log_prior_norm = logsumexp(&scratch_mix[..n]);
    for idx in 0..n {
        scratch_mix[idx] += expert_logps[idx];
    }
    let log_evidence = logsumexp(&scratch_mix[..n]);
    for idx in 0..n {
        experts[idx].cum_log_loss -= expert_logps[idx];
        experts[idx].log_weight =
            decay * experts[idx].log_weight + expert_logps[idx] - log_evidence;
    }
    log_evidence - log_prior_norm
}

// The switching update coordinates expert weights, scratch buffers, schedule
// parameters, and the mutation counter in one hot-path pass; a config wrapper
// would obscure which state is read-only versus updated in place.
#[allow(clippy::too_many_arguments)]
fn apply_switching_update_from_logps(
    experts: &mut [ExpertState],
    expert_logps: &[f64],
    scratch_joint: &mut Vec<f64>,
    scratch_weights: &mut Vec<f64>,
    prior: &[f64],
    schedule: MixtureScheduleMode,
    alpha: f64,
    update_count: &mut u64,
) -> f64 {
    let n = experts.len();
    scratch_joint.resize(n, 0.0);
    scratch_weights.resize(n, 0.0);
    // Index form required for coordinated writes to two scratch vecs + experts.
    #[allow(clippy::needless_range_loop)]
    for idx in 0..n {
        experts[idx].cum_log_loss -= expert_logps[idx];
        scratch_joint[idx] = experts[idx].log_weight + expert_logps[idx];
    }
    let log_mix = logsumexp(&scratch_joint[..n]);
    #[allow(clippy::needless_range_loop)]
    for idx in 0..n {
        scratch_weights[idx] = (scratch_joint[idx] - log_mix).exp();
    }

    let alpha = switching_alpha_for_update(schedule, alpha, *update_count);
    *update_count = (*update_count).saturating_add(1);
    if n == 1 || alpha <= 0.0 {
        set_log_weights_from_linear(experts, scratch_weights);
        return log_mix;
    }

    let mut switch_out_sum = 0.0;
    let mut num_switch_targets = 0usize;
    for &prior_weight in prior {
        if prior_weight < 1.0 {
            num_switch_targets += 1;
        }
    }
    if num_switch_targets <= 1 {
        set_log_weights_from_linear(experts, scratch_weights);
        return log_mix;
    }

    for idx in 0..n {
        let denom = 1.0 - prior[idx];
        if denom > 0.0 {
            switch_out_sum += scratch_weights[idx] / denom;
        }
    }
    for idx in 0..n {
        let stay = (1.0 - alpha) * scratch_weights[idx];
        let switch_in = if prior[idx] > 0.0 {
            let denom = 1.0 - prior[idx];
            let switchable_mass = if denom > 0.0 {
                switch_out_sum - scratch_weights[idx] / denom
            } else {
                0.0
            };
            alpha * prior[idx] * switchable_mass
        } else {
            0.0
        };
        scratch_joint[idx] = stay + switch_in;
    }
    normalize_simplex_weights(scratch_joint);
    set_log_weights_from_linear(experts, scratch_joint);
    log_mix
}

fn mix_log_prob_convex(lambda: &[f64], logps: &[f64]) -> f64 {
    let mut mix = 0.0;
    for (weight, &logp) in lambda.iter().zip(logps.iter()) {
        if *weight > 0.0 {
            mix += *weight * logp.exp();
        }
    }
    clamp_prob(mix, DEFAULT_MIN_PROB).ln()
}

fn apply_convex_update_from_logps(
    experts: &mut [ExpertState],
    expert_logps: &[f64],
    lambda: &mut [f64],
    projection_scratch: &mut Vec<f64>,
    schedule: MixtureScheduleMode,
    alpha: f64,
    update_count: &mut u64,
) -> f64 {
    let log_mix = mix_log_prob_convex(lambda, expert_logps);
    for idx in 0..experts.len() {
        experts[idx].cum_log_loss -= expert_logps[idx];
    }
    *update_count = (*update_count).saturating_add(1);
    let step_size = convex_step_size_for_update(schedule, alpha, *update_count);
    for (weight, &logp) in lambda.iter_mut().zip(expert_logps.iter()) {
        let grad = -(logp - log_mix).exp();
        *weight -= step_size * grad;
    }
    project_simplex_with_scratch(lambda, projection_scratch);
    log_mix
}

fn apply_mdl_update_from_logps(
    experts: &mut [ExpertState],
    expert_logps: &[f64],
    best_idx: usize,
    last_best: &mut usize,
) -> f64 {
    for idx in 0..experts.len() {
        experts[idx].cum_log_loss -= expert_logps[idx];
    }
    *last_best = best_idx;
    expert_logps
        .get(best_idx)
        .copied()
        .unwrap_or(f64::NEG_INFINITY)
}

fn finish_neural_update_from_logps(
    mixture: &mut NeuralMixture,
    symbol: u8,
    logp: f64,
    update_weights: bool,
) {
    if update_weights {
        mixture
            .neural
            .update_weights_symbol(&mixture.scratch_expert_logps, mixture.min_prob);
    }
    mixture.total_log_loss -= logp;
    mixture.analyzer.update(symbol);
    mixture.neural.set_context_state(mixture.analyzer.state());
    mixture.invalidate_eval_cache();
}

/// Exponential-weights Bayes mixture (log-loss Hedge).
#[derive(Clone)]
pub struct BayesMixture {
    experts: Vec<ExpertState>,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

impl BayesMixture {
    /// Construct a normalized Bayes mixture from expert configs.
    pub fn new(configs: &[ExpertConfig]) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
        let norm = logsumexp(&log_priors);
        for e in &mut experts {
            e.log_weight -= norm;
        }
        Self {
            experts,
            scratch_logps: vec![0.0; configs.len()],
            scratch_mix: vec![0.0; configs.len()],
            bitwise: MixtureBitPrefixState::default(),
            cached_symbol: 0,
            cached_log_mix: f64::NEG_INFINITY,
            cache_valid: false,
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if self.cache_valid && self.cached_symbol == symbol {
            for expert in &mut self.experts {
                expert.update(symbol);
            }
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob_update(symbol);
            }
        }
        let log_mix = apply_bayes_update_from_logps(
            &mut self.experts,
            &self.scratch_logps,
            &mut self.scratch_mix,
        );
        self.cache_valid = false;
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_logps[i] = expert.log_prob(symbol);
            self.scratch_mix[i] = expert.log_weight + self.scratch_logps[i];
        }
        let log_mix = logsumexp(&self.scratch_mix);
        self.cached_symbol = symbol;
        self.cached_log_mix = log_mix;
        self.cache_valid = true;
        log_mix
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        out.fill(f64::NEG_INFINITY);
        let norm = logsumexp_weights(&self.experts);
        let mut row = [0.0f64; 256];
        for expert in &mut self.experts {
            expert.predictor.fill_log_probs(&mut row);
            let lw = expert.log_weight - norm;
            for b in 0..256 {
                out[b] = logsumexp2(out[b], lw + row[b]);
            }
        }
    }

    /// Posterior weights (normalized) over experts.
    pub fn posterior(&self) -> Vec<f64> {
        let norm = logsumexp_weights(&self.experts);
        self.experts
            .iter()
            .map(|e| (e.log_weight - norm).exp())
            .collect()
    }

    /// Index and log-loss (nats) of the current best expert.
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Index and posterior mass of the most likely expert.
    pub fn max_posterior(&self) -> (usize, f64) {
        let norm = logsumexp_weights(&self.experts);
        let mut best_idx = 0usize;
        let mut best_p = 0.0;
        for (i, e) in self.experts.iter().enumerate() {
            let p = (e.log_weight - norm).exp();
            if p > best_p {
                best_p = p;
                best_idx = i;
            }
        }
        (best_idx, best_p)
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert cumulative log-losses (nats) and names.
    pub fn expert_log_losses(&self) -> Vec<(String, f64)> {
        self.experts
            .iter()
            .map(|e| (e.name.clone(), e.cum_log_loss))
            .collect()
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }

    #[inline]
    fn clear_stream_state(&mut self) {
        self.cache_valid = false;
        self.total_log_loss = 0.0;
    }

    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        reset_expert_frozen_stream(&mut self.experts, total_symbols)?;
        self.clear_stream_state();
        Ok(())
    }

    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_fresh_stream(&mut self.experts, total_symbols)?;
        reset_experts_to_priors(&mut self.experts);
        self.clear_stream_state();
        Ok(())
    }

    fn update_frozen(&mut self, symbol: u8) {
        for expert in &mut self.experts {
            expert.update_frozen(symbol);
        }
        self.cache_valid = false;
    }
}

/// Exponential-weights Bayes mixture with exponential forgetting on weights.
///
/// This is a non-stationary control: weights are discounted each step by `decay`.
#[derive(Clone)]
pub struct FadingBayesMixture {
    experts: Vec<ExpertState>,
    decay: f64,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_predictive: f64,
    cached_log_evidence: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

impl FadingBayesMixture {
    /// Construct a fading Bayes mixture with decay in `[0, 1]`.
    pub fn new(configs: &[ExpertConfig], decay: f64) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
        let norm = logsumexp(&log_priors);
        for e in &mut experts {
            e.log_weight -= norm;
        }
        let decay = decay.clamp(0.0, 1.0);
        Self {
            experts,
            decay,
            scratch_logps: vec![0.0; configs.len()],
            scratch_mix: vec![0.0; configs.len()],
            bitwise: MixtureBitPrefixState::default(),
            cached_symbol: 0,
            cached_log_predictive: f64::NEG_INFINITY,
            cached_log_evidence: f64::NEG_INFINITY,
            cache_valid: false,
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the fading mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if self.cache_valid && self.cached_symbol == symbol {
            for expert in &mut self.experts {
                expert.update(symbol);
            }
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob_update(symbol);
            }
        }
        let log_predictive = apply_fading_update_from_logps(
            &mut self.experts,
            &self.scratch_logps,
            &mut self.scratch_mix,
            self.decay,
        );
        self.cache_valid = false;
        self.total_log_loss -= log_predictive;
        log_predictive
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_logps[i] = expert.log_prob(symbol);
            self.scratch_mix[i] = self.decay * expert.log_weight;
        }
        let log_prior_norm = logsumexp(&self.scratch_mix);
        for i in 0..self.experts.len() {
            self.scratch_mix[i] += self.scratch_logps[i];
        }
        let log_evidence = logsumexp(&self.scratch_mix);
        let log_predictive = log_evidence - log_prior_norm;
        self.cached_symbol = symbol;
        self.cached_log_predictive = log_predictive;
        self.cached_log_evidence = log_evidence;
        self.cache_valid = true;
        log_predictive
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        out.fill(f64::NEG_INFINITY);
        let mut decayed = Vec::with_capacity(self.experts.len());
        for expert in &self.experts {
            decayed.push(self.decay * expert.log_weight);
        }
        let norm = logsumexp(&decayed);
        let mut row = [0.0f64; 256];
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.predictor.fill_log_probs(&mut row);
            let lw = decayed[i] - norm;
            for b in 0..256 {
                out[b] = logsumexp2(out[b], lw + row[b]);
            }
        }
    }

    /// Posterior weights (normalized) over experts.
    pub fn posterior(&self) -> Vec<f64> {
        let norm = logsumexp_weights(&self.experts);
        self.experts
            .iter()
            .map(|e| (e.log_weight - norm).exp())
            .collect()
    }

    /// Index and log-loss (nats) of the current best expert (non-discounted loss).
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }

    #[inline]
    fn clear_stream_state(&mut self) {
        self.cache_valid = false;
        self.total_log_loss = 0.0;
    }

    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        reset_expert_frozen_stream(&mut self.experts, total_symbols)?;
        self.clear_stream_state();
        Ok(())
    }

    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_fresh_stream(&mut self.experts, total_symbols)?;
        reset_experts_to_priors(&mut self.experts);
        self.clear_stream_state();
        Ok(())
    }

    fn update_frozen(&mut self, symbol: u8) {
        for expert in &mut self.experts {
            expert.update_frozen(symbol);
        }
        self.cache_valid = false;
    }
}

/// Switching mixture: allows occasional switches between experts.
#[derive(Clone)]
pub struct SwitchingMixture {
    experts: Vec<ExpertState>,
    prior: Vec<f64>,
    alpha: f64,
    schedule: MixtureScheduleMode,
    scratch_logps: Vec<f64>,
    scratch_joint: Vec<f64>,
    scratch_weights: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
    update_count: u64,
}

impl SwitchingMixture {
    /// Construct a switching mixture.
    pub fn new(configs: &[ExpertConfig], alpha: f64, schedule: MixtureScheduleMode) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let prior = normalized_prior_weights(configs);
        set_log_weights_from_linear(&mut experts, &prior);
        Self {
            experts,
            prior,
            alpha,
            schedule,
            scratch_logps: vec![0.0; configs.len()],
            scratch_joint: vec![0.0; configs.len()],
            scratch_weights: vec![0.0; configs.len()],
            bitwise: MixtureBitPrefixState::default(),
            cached_symbol: 0,
            cached_log_mix: f64::NEG_INFINITY,
            cache_valid: false,
            total_log_loss: 0.0,
            update_count: 0,
        }
    }

    /// Log-probability (natural log) of the switching mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if self.cache_valid && self.cached_symbol == symbol {
            for expert in &mut self.experts {
                expert.update(symbol);
            }
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob_update(symbol);
            }
        }
        let log_mix = apply_switching_update_from_logps(
            &mut self.experts,
            &self.scratch_logps,
            &mut self.scratch_joint,
            &mut self.scratch_weights,
            &self.prior,
            self.schedule,
            self.alpha,
            &mut self.update_count,
        );
        self.cache_valid = false;
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for i in 0..self.experts.len() {
            let lp = self.experts[i].log_prob(symbol);
            self.scratch_logps[i] = lp;
            self.scratch_joint[i] = self.experts[i].log_weight + lp;
        }
        let log_mix = logsumexp(&self.scratch_joint);
        self.cached_symbol = symbol;
        self.cached_log_mix = log_mix;
        self.cache_valid = true;
        log_mix
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        out.fill(f64::NEG_INFINITY);
        let norm = logsumexp_weights(&self.experts);
        let mut row = [0.0f64; 256];
        for expert in &mut self.experts {
            expert.predictor.fill_log_probs(&mut row);
            let lw = expert.log_weight - norm;
            for b in 0..256 {
                out[b] = logsumexp2(out[b], lw + row[b]);
            }
        }
    }

    /// Posterior weights (normalized) over experts.
    pub fn posterior(&self) -> Vec<f64> {
        let norm = logsumexp_weights(&self.experts);
        self.experts
            .iter()
            .map(|e| (e.log_weight - norm).exp())
            .collect()
    }

    /// Index and log-loss (nats) of the current best expert.
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Index and posterior mass of the most likely expert.
    pub fn max_posterior(&self) -> (usize, f64) {
        let norm = logsumexp_weights(&self.experts);
        let mut best_idx = 0usize;
        let mut best_p = 0.0;
        for (i, e) in self.experts.iter().enumerate() {
            let p = (e.log_weight - norm).exp();
            if p > best_p {
                best_p = p;
                best_idx = i;
            }
        }
        (best_idx, best_p)
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert cumulative log-losses (nats) and names.
    pub fn expert_log_losses(&self) -> Vec<(String, f64)> {
        self.experts
            .iter()
            .map(|e| (e.name.clone(), e.cum_log_loss))
            .collect()
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }

    #[inline]
    fn clear_stream_state(&mut self) {
        self.cache_valid = false;
        self.total_log_loss = 0.0;
        self.update_count = 0;
    }

    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        reset_expert_frozen_stream(&mut self.experts, total_symbols)?;
        self.clear_stream_state();
        Ok(())
    }

    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_fresh_stream(&mut self.experts, total_symbols)?;
        set_log_weights_from_linear(&mut self.experts, &self.prior);
        reset_expert_losses(&mut self.experts);
        self.clear_stream_state();
        Ok(())
    }

    fn update_frozen(&mut self, symbol: u8) {
        for expert in &mut self.experts {
            expert.update_frozen(symbol);
        }
        self.cache_valid = false;
    }
}

/// Convex mixture with projected-simplex online updates.
#[derive(Clone)]
pub struct ConvexMixture {
    experts: Vec<ExpertState>,
    alpha: f64,
    schedule: MixtureScheduleMode,
    lambda: Vec<f64>,
    scratch_logps: Vec<f64>,
    projection_scratch: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
    update_count: u64,
}

impl ConvexMixture {
    /// Construct a convex mixture with prior-derived initial weights.
    pub fn new(configs: &[ExpertConfig], alpha: f64, schedule: MixtureScheduleMode) -> Self {
        Self {
            experts: configs.iter().map(|c| c.build()).collect(),
            alpha,
            schedule,
            lambda: normalized_prior_weights(configs),
            scratch_logps: vec![0.0; configs.len()],
            projection_scratch: Vec::with_capacity(configs.len()),
            bitwise: MixtureBitPrefixState::default(),
            cached_symbol: 0,
            cached_log_mix: f64::NEG_INFINITY,
            cache_valid: false,
            total_log_loss: 0.0,
            update_count: 0,
        }
    }

    fn mix_log_prob(&self, logps: &[f64]) -> f64 {
        mix_log_prob_convex(&self.lambda, logps)
    }

    /// Log-probability (natural log) of the convex mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }

        if self.cache_valid && self.cached_symbol == symbol {
            for expert in &mut self.experts {
                expert.update(symbol);
            }
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob_update(symbol);
            }
        }
        let log_mix = apply_convex_update_from_logps(
            &mut self.experts,
            &self.scratch_logps,
            &mut self.lambda,
            &mut self.projection_scratch,
            self.schedule,
            self.alpha,
            &mut self.update_count,
        );
        self.cache_valid = false;
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_logps[i] = expert.log_prob(symbol);
        }
        let log_mix = self.mix_log_prob(&self.scratch_logps);
        self.cached_symbol = symbol;
        self.cached_log_mix = log_mix;
        self.cache_valid = true;
        log_mix
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        out.fill(f64::NEG_INFINITY);
        let mut row = [0.0f64; 256];
        for (index, expert) in self.experts.iter_mut().enumerate() {
            expert.predictor.fill_log_probs(&mut row);
            let weight = self.lambda.get(index).copied().unwrap_or(0.0);
            if weight <= 0.0 {
                continue;
            }
            let log_weight = weight.ln();
            for byte in 0..256 {
                out[byte] = logsumexp2(out[byte], log_weight + row[byte]);
            }
        }
    }

    #[inline]
    fn clear_stream_state(&mut self) {
        self.cache_valid = false;
        self.total_log_loss = 0.0;
        self.update_count = 0;
    }

    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        reset_expert_frozen_stream(&mut self.experts, total_symbols)?;
        self.clear_stream_state();
        Ok(())
    }

    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_fresh_stream(&mut self.experts, total_symbols)?;
        self.lambda = reset_experts_to_priors(&mut self.experts);
        self.clear_stream_state();
        Ok(())
    }

    fn update_frozen(&mut self, symbol: u8) {
        for expert in &mut self.experts {
            expert.update_frozen(symbol);
        }
        self.cache_valid = false;
    }
}

/// MDL-style selector: predicts with the current best expert (by cumulative loss).
#[derive(Clone)]
pub struct MdlSelector {
    experts: Vec<ExpertState>,
    scratch_logps: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    total_log_loss: f64,
    last_best: usize,
    cached_symbol: u8,
    cached_best_idx: usize,
    cached_best_logp: f64,
    cache_valid: bool,
}

/// Bytewise neural mixer  (Loosely PAQ inspired)
///
/// This model is a context-conditioned two-stage gating network trained online
/// from per-symbol expert likelihoods:
/// 1) context-local first-stage expert gates,
/// 2) context-local second-stage meta-gate over stage-1 outputs,
/// 3) per-symbol SGD updates with optional tiny-error skip.
#[derive(Clone)]
pub struct NeuralMixture {
    experts: Vec<ExpertState>,
    neural: NeuralMixCore,
    analyzer: TextContextAnalyzer,
    min_prob: f64,
    scratch_expert_logps: Vec<f64>,
    scratch_mix_weights: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    eval_cache_valid: bool,
    eval_cache_full_valid: bool,
    eval_cache_history: NeuralHistoryState,
    eval_cache_symbol: u8,
    eval_cache_logp: f64,
    eval_cache_mix_logps: [f64; 256],
    eval_cache_expert_logps: Vec<[f64; 256]>,
    total_log_loss: f64,
}

impl NeuralMixture {
    /// Construct a neural mixture. `learning_rate` is taken from `MixtureSpec.alpha`.
    pub fn new(configs: &[ExpertConfig], learning_rate: f64) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let n = experts.len();

        let mut prior_weights = vec![0.0; n];
        if n > 0 {
            let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
            let norm = logsumexp(&log_priors);
            for (i, e) in experts.iter_mut().enumerate() {
                let p = (e.log_prior - norm).exp();
                prior_weights[i] = p;
            }
        }

        let base_lr = if learning_rate.is_finite() {
            learning_rate.abs().clamp(1e-6, 1.0)
        } else {
            0.03
        };
        let effective_lr = (base_lr * 25.0).clamp(1e-6, 1.0);
        let analyzer = TextContextAnalyzer::new();
        let mut neural =
            NeuralMixCore::new(n, &prior_weights, effective_lr * 0.5, effective_lr, 1e-5);
        neural.set_context_state(analyzer.state());
        let eval_cache_history = neural.history_state();

        Self {
            experts,
            neural,
            analyzer,
            min_prob: DEFAULT_MIN_PROB,
            scratch_expert_logps: vec![0.0; n],
            scratch_mix_weights: vec![0.0; n],
            bitwise: MixtureBitPrefixState::default(),
            eval_cache_valid: false,
            eval_cache_full_valid: false,
            eval_cache_history,
            eval_cache_symbol: 0,
            eval_cache_logp: f64::NEG_INFINITY,
            eval_cache_mix_logps: [f64::NEG_INFINITY; 256],
            eval_cache_expert_logps: vec![[f64::NEG_INFINITY; 256]; n],
            total_log_loss: 0.0,
        }
    }

    #[inline]
    fn invalidate_eval_cache(&mut self) {
        self.eval_cache_valid = false;
        self.eval_cache_full_valid = false;
    }

    fn sync_history_state(&mut self) -> NeuralHistoryState {
        let history = self.analyzer.state();
        if self.neural.history_state() != history {
            self.neural.set_context_state(history);
        }
        if self.eval_cache_history != history {
            self.invalidate_eval_cache();
            self.eval_cache_history = history;
        }
        history
    }

    fn ensure_full_evaluation(&mut self) {
        self.sync_history_state();
        if self.eval_cache_full_valid {
            return;
        }

        self.neural.evaluate_expert_weights();
        self.scratch_mix_weights
            .copy_from_slice(self.neural.expert_weights());
        let mut mix_pdf = [0.0f64; 256];
        for i in 0..self.experts.len() {
            let row = &mut self.eval_cache_expert_logps[i];
            self.experts[i].predictor.fill_log_probs(row);
            let w = self.scratch_mix_weights[i];
            for (dst, &lp) in mix_pdf.iter_mut().zip(row.iter()) {
                *dst += w * clamp_prob(lp.exp(), self.min_prob);
            }
        }

        let sum: f64 = mix_pdf.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            let uniform = (1.0f64 / 256.0).ln();
            self.eval_cache_mix_logps.fill(uniform);
        } else {
            let inv = 1.0 / sum;
            for (dst, &p_raw) in self.eval_cache_mix_logps.iter_mut().zip(mix_pdf.iter()) {
                let p = clamp_unit_prob(p_raw * inv, self.min_prob);
                *dst = p.ln();
            }
        }

        self.eval_cache_full_valid = true;
    }

    fn evaluate_symbol(&mut self, symbol: u8) -> f64 {
        let history = self.sync_history_state();
        if self.eval_cache_valid
            && self.eval_cache_history == history
            && self.eval_cache_symbol == symbol
        {
            return self.eval_cache_logp;
        }

        if self.eval_cache_full_valid && self.eval_cache_history == history {
            for (dst, row) in self
                .scratch_expert_logps
                .iter_mut()
                .zip(self.eval_cache_expert_logps.iter())
            {
                *dst = row[symbol as usize];
            }
            let logp = self.eval_cache_mix_logps[symbol as usize];
            self.eval_cache_valid = true;
            self.eval_cache_symbol = symbol;
            self.eval_cache_logp = logp;
            return logp;
        }

        let expert_count = self.experts.len();
        for i in 0..expert_count {
            self.scratch_expert_logps[i] = self.experts[i].log_prob(symbol);
        }
        let p = self
            .neural
            .evaluate_symbol(&self.scratch_expert_logps, self.min_prob);
        let logp = clamp_unit_prob(p, self.min_prob).ln();
        self.eval_cache_valid = true;
        self.eval_cache_history = history;
        self.eval_cache_symbol = symbol;
        self.eval_cache_logp = logp;
        logp
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if self.experts.len() == 1 {
            return self.experts[0].log_prob(symbol);
        }
        self.evaluate_symbol(symbol)
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        if self.experts.len() == 1 {
            self.experts[0].predictor.fill_log_probs(out);
            return;
        }
        self.ensure_full_evaluation();
        out.copy_from_slice(&self.eval_cache_mix_logps);
    }

    /// Log-probability (natural log) of the neural mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }

        if self.experts.len() == 1 {
            let expert = &mut self.experts[0];
            let logp = expert.log_prob_update(symbol);
            expert.cum_log_loss -= logp;
            self.total_log_loss -= logp;
            self.analyzer.update(symbol);
            self.neural.set_context_state(self.analyzer.state());
            self.invalidate_eval_cache();
            return logp;
        }

        let history = self.sync_history_state();
        let logp = if self.eval_cache_valid
            && self.eval_cache_history == history
            && self.eval_cache_symbol == symbol
        {
            let logp = self.eval_cache_logp;
            for i in 0..self.experts.len() {
                let expert = &mut self.experts[i];
                expert.cum_log_loss -= self.scratch_expert_logps[i];
                expert.update(symbol);
            }
            logp
        } else if self.eval_cache_full_valid && self.eval_cache_history == history {
            for i in 0..self.experts.len() {
                self.scratch_expert_logps[i] = self.eval_cache_expert_logps[i][symbol as usize];
            }
            let logp = self.eval_cache_mix_logps[symbol as usize];
            for i in 0..self.experts.len() {
                let expert = &mut self.experts[i];
                expert.cum_log_loss -= self.scratch_expert_logps[i];
                expert.update(symbol);
            }
            logp
        } else {
            for i in 0..self.experts.len() {
                let expert = &mut self.experts[i];
                self.scratch_expert_logps[i] = expert.log_prob_update(symbol);
                expert.cum_log_loss -= self.scratch_expert_logps[i];
            }
            let p = self
                .neural
                .evaluate_symbol(&self.scratch_expert_logps, self.min_prob);
            clamp_unit_prob(p, self.min_prob).ln()
        };
        finish_neural_update_from_logps(self, symbol, logp, true);
        logp
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    #[inline]
    fn clear_stream_state(&mut self) {
        self.analyzer = TextContextAnalyzer::new();
        self.neural.set_context_state(self.analyzer.state());
        self.invalidate_eval_cache();
        self.eval_cache_history = self.neural.history_state();
        self.total_log_loss = 0.0;
    }

    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        reset_expert_frozen_stream(&mut self.experts, total_symbols)?;
        self.clear_stream_state();
        Ok(())
    }

    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_fresh_stream(&mut self.experts, total_symbols)?;
        let prior = reset_experts_to_priors(&mut self.experts);
        self.neural.reset_to_priors(&prior);
        self.clear_stream_state();
        Ok(())
    }

    fn update_frozen(&mut self, symbol: u8) {
        for expert in &mut self.experts {
            expert.update_frozen(symbol);
        }
        self.analyzer.update(symbol);
        self.neural.set_context_state(self.analyzer.state());
        self.invalidate_eval_cache();
        self.eval_cache_history = self.neural.history_state();
    }
}

impl MdlSelector {
    /// Construct an MDL-style expert selector.
    pub fn new(configs: &[ExpertConfig]) -> Self {
        let experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let last_best = 0usize;
        Self {
            experts,
            scratch_logps: vec![0.0; configs.len()],
            bitwise: MixtureBitPrefixState::default(),
            total_log_loss: 0.0,
            last_best,
            cached_symbol: 0,
            cached_best_idx: 0,
            cached_best_logp: f64::NEG_INFINITY,
            cache_valid: false,
        }
    }

    /// Log-probability (natural log) of the MDL selector for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        let used_cache = self.cache_valid && self.cached_symbol == symbol;
        let best_idx = if used_cache {
            self.scratch_logps[self.cached_best_idx] = self.cached_best_logp;
            for (i, expert) in self.experts.iter_mut().enumerate() {
                if i == self.cached_best_idx {
                    continue;
                }
                self.scratch_logps[i] = expert.log_prob(symbol);
            }
            self.cached_best_idx
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob_update(symbol);
            }
            let mut best_idx = 0usize;
            let mut best_loss = f64::INFINITY;
            for (i, expert) in self.experts.iter().enumerate() {
                if expert.cum_log_loss < best_loss {
                    best_loss = expert.cum_log_loss;
                    best_idx = i;
                }
            }
            best_idx
        };
        self.cache_valid = false;
        for expert in &mut self.experts {
            if used_cache {
                expert.update(symbol);
            }
        }
        let logp = apply_mdl_update_from_logps(
            &mut self.experts,
            &self.scratch_logps,
            best_idx,
            &mut self.last_best,
        );
        self.total_log_loss -= logp;
        logp
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, expert) in self.experts.iter().enumerate() {
            if expert.cum_log_loss < best_loss {
                best_loss = expert.cum_log_loss;
                best_idx = i;
            }
        }
        let logp = self.experts[best_idx].log_prob(symbol);
        self.cached_symbol = symbol;
        self.cached_best_idx = best_idx;
        self.cached_best_logp = logp;
        self.cache_valid = true;
        logp
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, expert) in self.experts.iter().enumerate() {
            if expert.cum_log_loss < best_loss {
                best_loss = expert.cum_log_loss;
                best_idx = i;
            }
        }
        self.experts[best_idx].predictor.fill_log_probs(out);
    }

    /// Index of the current best expert.
    pub fn best_index(&self) -> usize {
        self.last_best
    }

    /// Index and log-loss (nats) of the current best expert.
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Total log-loss of the selector so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert cumulative log-losses (nats) and names.
    pub fn expert_log_losses(&self) -> Vec<(String, f64)> {
        self.experts
            .iter()
            .map(|e| (e.name.clone(), e.cum_log_loss))
            .collect()
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }

    #[inline]
    fn clear_stream_state(&mut self) {
        self.cache_valid = false;
        self.total_log_loss = 0.0;
    }

    fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        reset_expert_frozen_stream(&mut self.experts, total_symbols)?;
        self.clear_stream_state();
        Ok(())
    }

    fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        begin_expert_fresh_stream(&mut self.experts, total_symbols)?;
        reset_experts_to_priors(&mut self.experts);
        self.last_best = 0;
        self.clear_stream_state();
        Ok(())
    }

    fn update_frozen(&mut self, symbol: u8) {
        for expert in &mut self.experts {
            expert.update_frozen(symbol);
        }
        self.cache_valid = false;
    }
}

#[derive(Clone)]
struct ExpertStateCheckpoint {
    log_weight: f64,
    log_prior: f64,
    cum_log_loss: f64,
    predictor: OnlineBytePredictorCheckpoint,
}

fn checkpoint_experts(experts: &mut [ExpertState]) -> Option<Vec<ExpertStateCheckpoint>> {
    experts
        .iter_mut()
        .map(|expert| {
            expert
                .predictor
                .checkpoint_if_supported()
                .map(|predictor| ExpertStateCheckpoint {
                    log_weight: expert.log_weight,
                    log_prior: expert.log_prior,
                    cum_log_loss: expert.cum_log_loss,
                    predictor,
                })
        })
        .collect()
}

fn restore_experts(experts: &mut [ExpertState], checkpoints: &[ExpertStateCheckpoint]) {
    assert_eq!(
        experts.len(),
        checkpoints.len(),
        "mixture checkpoint expert count mismatch"
    );
    for (expert, checkpoint) in experts.iter_mut().zip(checkpoints.iter()) {
        expert.log_weight = checkpoint.log_weight;
        expert.log_prior = checkpoint.log_prior;
        expert.cum_log_loss = checkpoint.cum_log_loss;
        assert!(
            expert
                .predictor
                .restore_checkpoint_if_supported(&checkpoint.predictor),
            "mixture expert rejected its structural checkpoint"
        );
    }
}

fn clear_expert_checkpoints(experts: &mut [ExpertState]) {
    for expert in experts {
        expert.predictor.clear_checkpoints_if_supported();
    }
}

fn discard_expert_checkpoints(
    experts: &mut [ExpertState],
    checkpoints: Vec<ExpertStateCheckpoint>,
) {
    assert_eq!(
        experts.len(),
        checkpoints.len(),
        "mixture checkpoint expert count mismatch"
    );
    for (expert, checkpoint) in experts.iter_mut().zip(checkpoints.into_iter()) {
        assert!(
            expert
                .predictor
                .discard_checkpoint_if_supported(checkpoint.predictor),
            "mixture expert rejected checkpoint discard"
        );
    }
}

fn lifecycle_checkpoint_experts(
    experts: &mut [ExpertState],
    op: ExpertLifecycleOp,
) -> Vec<ExpertStateLifecycleCheckpoint> {
    experts
        .iter_mut()
        .map(|expert| expert.snapshot_state_lifecycle(op))
        .collect()
}

fn restore_lifecycle_experts(
    experts: &mut [ExpertState],
    checkpoints: Vec<ExpertStateLifecycleCheckpoint>,
    op: ExpertLifecycleOp,
) {
    assert_eq!(
        experts.len(),
        checkpoints.len(),
        "mixture lifecycle checkpoint expert count mismatch"
    );
    for (expert, checkpoint) in experts.iter_mut().zip(checkpoints.into_iter()) {
        expert.restore_state_lifecycle(op, checkpoint);
    }
}

fn discard_lifecycle_experts(
    experts: &mut [ExpertState],
    checkpoints: Vec<ExpertStateLifecycleCheckpoint>,
    op: ExpertLifecycleOp,
) {
    assert_eq!(
        experts.len(),
        checkpoints.len(),
        "mixture lifecycle checkpoint expert count mismatch"
    );
    for (expert, checkpoint) in experts.iter_mut().zip(checkpoints.into_iter()) {
        expert.discard_state_lifecycle(op, checkpoint);
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct BayesMixtureCheckpoint {
    experts: Vec<ExpertStateCheckpoint>,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct FadingBayesMixtureCheckpoint {
    experts: Vec<ExpertStateCheckpoint>,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_predictive: f64,
    cached_log_evidence: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct SwitchingMixtureCheckpoint {
    experts: Vec<ExpertStateCheckpoint>,
    scratch_logps: Vec<f64>,
    scratch_joint: Vec<f64>,
    scratch_weights: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
    update_count: u64,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct ConvexMixtureCheckpoint {
    experts: Vec<ExpertStateCheckpoint>,
    lambda: Vec<f64>,
    scratch_logps: Vec<f64>,
    projection_scratch: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
    update_count: u64,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct MdlSelectorCheckpoint {
    experts: Vec<ExpertStateCheckpoint>,
    scratch_logps: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    total_log_loss: f64,
    last_best: usize,
    cached_symbol: u8,
    cached_best_idx: usize,
    cached_best_logp: f64,
    cache_valid: bool,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct NeuralMixtureCheckpoint {
    experts: Vec<ExpertStateCheckpoint>,
    neural: NeuralMixCore,
    analyzer: TextContextAnalyzer,
    scratch_expert_logps: Vec<f64>,
    scratch_mix_weights: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    eval_cache_valid: bool,
    eval_cache_full_valid: bool,
    eval_cache_history: NeuralHistoryState,
    eval_cache_symbol: u8,
    eval_cache_logp: f64,
    eval_cache_mix_logps: [f64; 256],
    eval_cache_expert_logps: Vec<[f64; 256]>,
    total_log_loss: f64,
}

#[derive(Clone)]
#[doc(hidden)]
// Mirrors `MixtureRuntime`: the neural checkpoint owns inline 256-way
// probability caches, and boxing it would add allocation to normal checkpoint
// capture/restore without reducing resident runtime state.
#[allow(clippy::large_enum_variant)]
pub enum MixtureRuntimeCheckpoint {
    Bayes(BayesMixtureCheckpoint),
    Fading(FadingBayesMixtureCheckpoint),
    Switching(SwitchingMixtureCheckpoint),
    Convex(ConvexMixtureCheckpoint),
    Mdl(MdlSelectorCheckpoint),
    Neural(NeuralMixtureCheckpoint),
}

struct BayesMixtureLifecycleCheckpoint {
    experts: Vec<ExpertStateLifecycleCheckpoint>,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

struct FadingBayesMixtureLifecycleCheckpoint {
    experts: Vec<ExpertStateLifecycleCheckpoint>,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_predictive: f64,
    cached_log_evidence: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

struct SwitchingMixtureLifecycleCheckpoint {
    experts: Vec<ExpertStateLifecycleCheckpoint>,
    scratch_logps: Vec<f64>,
    scratch_joint: Vec<f64>,
    scratch_weights: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
    update_count: u64,
}

struct ConvexMixtureLifecycleCheckpoint {
    experts: Vec<ExpertStateLifecycleCheckpoint>,
    lambda: Vec<f64>,
    scratch_logps: Vec<f64>,
    projection_scratch: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
    update_count: u64,
}

struct MdlSelectorLifecycleCheckpoint {
    experts: Vec<ExpertStateLifecycleCheckpoint>,
    scratch_logps: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    total_log_loss: f64,
    last_best: usize,
    cached_symbol: u8,
    cached_best_idx: usize,
    cached_best_logp: f64,
    cache_valid: bool,
}

struct NeuralMixtureLifecycleCheckpoint {
    experts: Vec<ExpertStateLifecycleCheckpoint>,
    neural: NeuralMixCore,
    analyzer: TextContextAnalyzer,
    scratch_expert_logps: Vec<f64>,
    scratch_mix_weights: Vec<f64>,
    bitwise: MixtureBitPrefixState,
    eval_cache_valid: bool,
    eval_cache_full_valid: bool,
    eval_cache_history: NeuralHistoryState,
    eval_cache_symbol: u8,
    eval_cache_logp: f64,
    eval_cache_mix_logps: [f64; 256],
    eval_cache_expert_logps: Vec<[f64; 256]>,
    total_log_loss: f64,
}

#[allow(clippy::large_enum_variant)]
enum MixtureRuntimeLifecycleCheckpoint {
    Bayes(BayesMixtureLifecycleCheckpoint),
    Fading(FadingBayesMixtureLifecycleCheckpoint),
    Switching(SwitchingMixtureLifecycleCheckpoint),
    Convex(ConvexMixtureLifecycleCheckpoint),
    Mdl(MdlSelectorLifecycleCheckpoint),
    Neural(NeuralMixtureLifecycleCheckpoint),
}

// =============================================================================
// Mixture Runtime Helper (for RateBackend::Mixture)
// =============================================================================

/// Runtime wrapper over concrete mixture strategies.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum MixtureRuntime {
    /// Bayes mixture.
    Bayes(BayesMixture),
    /// Fading Bayes mixture.
    Fading(FadingBayesMixture),
    /// Switching mixture.
    Switching(SwitchingMixture),
    /// Convex mixture.
    Convex(ConvexMixture),
    /// MDL selector.
    Mdl(MdlSelector),
    /// Bytewise neural logistic mixer.
    Neural(NeuralMixture),
}

impl MixtureRuntime {
    pub(crate) fn checkpoint(&mut self) -> Option<MixtureRuntimeCheckpoint> {
        match self {
            MixtureRuntime::Bayes(m) => {
                Some(MixtureRuntimeCheckpoint::Bayes(BayesMixtureCheckpoint {
                    experts: checkpoint_experts(&mut m.experts)?,
                    scratch_logps: m.scratch_logps.clone(),
                    scratch_mix: m.scratch_mix.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_mix: m.cached_log_mix,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                }))
            }
            MixtureRuntime::Fading(m) => Some(MixtureRuntimeCheckpoint::Fading(
                FadingBayesMixtureCheckpoint {
                    experts: checkpoint_experts(&mut m.experts)?,
                    scratch_logps: m.scratch_logps.clone(),
                    scratch_mix: m.scratch_mix.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_predictive: m.cached_log_predictive,
                    cached_log_evidence: m.cached_log_evidence,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                },
            )),
            MixtureRuntime::Switching(m) => Some(MixtureRuntimeCheckpoint::Switching(
                SwitchingMixtureCheckpoint {
                    experts: checkpoint_experts(&mut m.experts)?,
                    scratch_logps: m.scratch_logps.clone(),
                    scratch_joint: m.scratch_joint.clone(),
                    scratch_weights: m.scratch_weights.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_mix: m.cached_log_mix,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                    update_count: m.update_count,
                },
            )),
            MixtureRuntime::Convex(m) => {
                Some(MixtureRuntimeCheckpoint::Convex(ConvexMixtureCheckpoint {
                    experts: checkpoint_experts(&mut m.experts)?,
                    lambda: m.lambda.clone(),
                    scratch_logps: m.scratch_logps.clone(),
                    projection_scratch: m.projection_scratch.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_mix: m.cached_log_mix,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                    update_count: m.update_count,
                }))
            }
            MixtureRuntime::Mdl(m) => Some(MixtureRuntimeCheckpoint::Mdl(MdlSelectorCheckpoint {
                experts: checkpoint_experts(&mut m.experts)?,
                scratch_logps: m.scratch_logps.clone(),
                bitwise: m.bitwise.clone(),
                total_log_loss: m.total_log_loss,
                last_best: m.last_best,
                cached_symbol: m.cached_symbol,
                cached_best_idx: m.cached_best_idx,
                cached_best_logp: m.cached_best_logp,
                cache_valid: m.cache_valid,
            })),
            MixtureRuntime::Neural(m) => {
                Some(MixtureRuntimeCheckpoint::Neural(NeuralMixtureCheckpoint {
                    experts: checkpoint_experts(&mut m.experts)?,
                    neural: m.neural.clone(),
                    analyzer: m.analyzer.clone(),
                    scratch_expert_logps: m.scratch_expert_logps.clone(),
                    scratch_mix_weights: m.scratch_mix_weights.clone(),
                    bitwise: m.bitwise.clone(),
                    eval_cache_valid: m.eval_cache_valid,
                    eval_cache_full_valid: m.eval_cache_full_valid,
                    eval_cache_history: m.eval_cache_history,
                    eval_cache_symbol: m.eval_cache_symbol,
                    eval_cache_logp: m.eval_cache_logp,
                    eval_cache_mix_logps: m.eval_cache_mix_logps,
                    eval_cache_expert_logps: m.eval_cache_expert_logps.clone(),
                    total_log_loss: m.total_log_loss,
                }))
            }
        }
    }

    pub(crate) fn restore_checkpoint(&mut self, checkpoint: &MixtureRuntimeCheckpoint) {
        match (self, checkpoint) {
            (MixtureRuntime::Bayes(m), MixtureRuntimeCheckpoint::Bayes(ck)) => {
                restore_experts(&mut m.experts, &ck.experts);
                m.scratch_logps = ck.scratch_logps.clone();
                m.scratch_mix = ck.scratch_mix.clone();
                m.bitwise = ck.bitwise.clone();
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_mix = ck.cached_log_mix;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
            }
            (MixtureRuntime::Fading(m), MixtureRuntimeCheckpoint::Fading(ck)) => {
                restore_experts(&mut m.experts, &ck.experts);
                m.scratch_logps = ck.scratch_logps.clone();
                m.scratch_mix = ck.scratch_mix.clone();
                m.bitwise = ck.bitwise.clone();
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_predictive = ck.cached_log_predictive;
                m.cached_log_evidence = ck.cached_log_evidence;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
            }
            (MixtureRuntime::Switching(m), MixtureRuntimeCheckpoint::Switching(ck)) => {
                restore_experts(&mut m.experts, &ck.experts);
                m.scratch_logps = ck.scratch_logps.clone();
                m.scratch_joint = ck.scratch_joint.clone();
                m.scratch_weights = ck.scratch_weights.clone();
                m.bitwise = ck.bitwise.clone();
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_mix = ck.cached_log_mix;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
                m.update_count = ck.update_count;
            }
            (MixtureRuntime::Convex(m), MixtureRuntimeCheckpoint::Convex(ck)) => {
                restore_experts(&mut m.experts, &ck.experts);
                m.lambda = ck.lambda.clone();
                m.scratch_logps = ck.scratch_logps.clone();
                m.projection_scratch = ck.projection_scratch.clone();
                m.bitwise = ck.bitwise.clone();
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_mix = ck.cached_log_mix;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
                m.update_count = ck.update_count;
            }
            (MixtureRuntime::Mdl(m), MixtureRuntimeCheckpoint::Mdl(ck)) => {
                restore_experts(&mut m.experts, &ck.experts);
                m.scratch_logps = ck.scratch_logps.clone();
                m.bitwise = ck.bitwise.clone();
                m.total_log_loss = ck.total_log_loss;
                m.last_best = ck.last_best;
                m.cached_symbol = ck.cached_symbol;
                m.cached_best_idx = ck.cached_best_idx;
                m.cached_best_logp = ck.cached_best_logp;
                m.cache_valid = ck.cache_valid;
            }
            (MixtureRuntime::Neural(m), MixtureRuntimeCheckpoint::Neural(ck)) => {
                restore_experts(&mut m.experts, &ck.experts);
                m.neural = ck.neural.clone();
                m.analyzer = ck.analyzer.clone();
                m.scratch_expert_logps = ck.scratch_expert_logps.clone();
                m.scratch_mix_weights = ck.scratch_mix_weights.clone();
                m.bitwise = ck.bitwise.clone();
                m.eval_cache_valid = ck.eval_cache_valid;
                m.eval_cache_full_valid = ck.eval_cache_full_valid;
                m.eval_cache_history = ck.eval_cache_history;
                m.eval_cache_symbol = ck.eval_cache_symbol;
                m.eval_cache_logp = ck.eval_cache_logp;
                m.eval_cache_mix_logps = ck.eval_cache_mix_logps;
                m.eval_cache_expert_logps = ck.eval_cache_expert_logps.clone();
                m.total_log_loss = ck.total_log_loss;
            }
            _ => panic!("mismatched MixtureRuntime checkpoint variant"),
        }
    }

    pub(crate) fn discard_checkpoint(&mut self, checkpoint: MixtureRuntimeCheckpoint) {
        match (self, checkpoint) {
            (MixtureRuntime::Bayes(m), MixtureRuntimeCheckpoint::Bayes(ck)) => {
                discard_expert_checkpoints(&mut m.experts, ck.experts);
            }
            (MixtureRuntime::Fading(m), MixtureRuntimeCheckpoint::Fading(ck)) => {
                discard_expert_checkpoints(&mut m.experts, ck.experts);
            }
            (MixtureRuntime::Switching(m), MixtureRuntimeCheckpoint::Switching(ck)) => {
                discard_expert_checkpoints(&mut m.experts, ck.experts);
            }
            (MixtureRuntime::Convex(m), MixtureRuntimeCheckpoint::Convex(ck)) => {
                discard_expert_checkpoints(&mut m.experts, ck.experts);
            }
            (MixtureRuntime::Mdl(m), MixtureRuntimeCheckpoint::Mdl(ck)) => {
                discard_expert_checkpoints(&mut m.experts, ck.experts);
            }
            (MixtureRuntime::Neural(m), MixtureRuntimeCheckpoint::Neural(ck)) => {
                discard_expert_checkpoints(&mut m.experts, ck.experts);
            }
            _ => panic!("mismatched MixtureRuntime checkpoint variant"),
        }
    }

    fn lifecycle_checkpoint(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
    ) -> MixtureRuntimeLifecycleCheckpoint {
        let expert_op = ExpertLifecycleOp::from_predictor_op(op);
        match self {
            MixtureRuntime::Bayes(m) => {
                MixtureRuntimeLifecycleCheckpoint::Bayes(BayesMixtureLifecycleCheckpoint {
                    experts: lifecycle_checkpoint_experts(&mut m.experts, expert_op),
                    scratch_logps: m.scratch_logps.clone(),
                    scratch_mix: m.scratch_mix.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_mix: m.cached_log_mix,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                })
            }
            MixtureRuntime::Fading(m) => {
                MixtureRuntimeLifecycleCheckpoint::Fading(FadingBayesMixtureLifecycleCheckpoint {
                    experts: lifecycle_checkpoint_experts(&mut m.experts, expert_op),
                    scratch_logps: m.scratch_logps.clone(),
                    scratch_mix: m.scratch_mix.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_predictive: m.cached_log_predictive,
                    cached_log_evidence: m.cached_log_evidence,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                })
            }
            MixtureRuntime::Switching(m) => {
                MixtureRuntimeLifecycleCheckpoint::Switching(SwitchingMixtureLifecycleCheckpoint {
                    experts: lifecycle_checkpoint_experts(&mut m.experts, expert_op),
                    scratch_logps: m.scratch_logps.clone(),
                    scratch_joint: m.scratch_joint.clone(),
                    scratch_weights: m.scratch_weights.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_mix: m.cached_log_mix,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                    update_count: m.update_count,
                })
            }
            MixtureRuntime::Convex(m) => {
                MixtureRuntimeLifecycleCheckpoint::Convex(ConvexMixtureLifecycleCheckpoint {
                    experts: lifecycle_checkpoint_experts(&mut m.experts, expert_op),
                    lambda: m.lambda.clone(),
                    scratch_logps: m.scratch_logps.clone(),
                    projection_scratch: m.projection_scratch.clone(),
                    bitwise: m.bitwise.clone(),
                    cached_symbol: m.cached_symbol,
                    cached_log_mix: m.cached_log_mix,
                    cache_valid: m.cache_valid,
                    total_log_loss: m.total_log_loss,
                    update_count: m.update_count,
                })
            }
            MixtureRuntime::Mdl(m) => {
                MixtureRuntimeLifecycleCheckpoint::Mdl(MdlSelectorLifecycleCheckpoint {
                    experts: lifecycle_checkpoint_experts(&mut m.experts, expert_op),
                    scratch_logps: m.scratch_logps.clone(),
                    bitwise: m.bitwise.clone(),
                    total_log_loss: m.total_log_loss,
                    last_best: m.last_best,
                    cached_symbol: m.cached_symbol,
                    cached_best_idx: m.cached_best_idx,
                    cached_best_logp: m.cached_best_logp,
                    cache_valid: m.cache_valid,
                })
            }
            MixtureRuntime::Neural(m) => {
                MixtureRuntimeLifecycleCheckpoint::Neural(NeuralMixtureLifecycleCheckpoint {
                    experts: lifecycle_checkpoint_experts(&mut m.experts, expert_op),
                    neural: m.neural.clone(),
                    analyzer: m.analyzer.clone(),
                    scratch_expert_logps: m.scratch_expert_logps.clone(),
                    scratch_mix_weights: m.scratch_mix_weights.clone(),
                    bitwise: m.bitwise.clone(),
                    eval_cache_valid: m.eval_cache_valid,
                    eval_cache_full_valid: m.eval_cache_full_valid,
                    eval_cache_history: m.eval_cache_history,
                    eval_cache_symbol: m.eval_cache_symbol,
                    eval_cache_logp: m.eval_cache_logp,
                    eval_cache_mix_logps: m.eval_cache_mix_logps,
                    eval_cache_expert_logps: m.eval_cache_expert_logps.clone(),
                    total_log_loss: m.total_log_loss,
                })
            }
        }
    }

    fn restore_lifecycle_checkpoint(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
        checkpoint: MixtureRuntimeLifecycleCheckpoint,
    ) {
        let expert_op = ExpertLifecycleOp::from_predictor_op(op);
        match (self, checkpoint) {
            (MixtureRuntime::Bayes(m), MixtureRuntimeLifecycleCheckpoint::Bayes(ck)) => {
                restore_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
                m.scratch_logps = ck.scratch_logps;
                m.scratch_mix = ck.scratch_mix;
                m.bitwise = ck.bitwise;
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_mix = ck.cached_log_mix;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
            }
            (MixtureRuntime::Fading(m), MixtureRuntimeLifecycleCheckpoint::Fading(ck)) => {
                restore_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
                m.scratch_logps = ck.scratch_logps;
                m.scratch_mix = ck.scratch_mix;
                m.bitwise = ck.bitwise;
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_predictive = ck.cached_log_predictive;
                m.cached_log_evidence = ck.cached_log_evidence;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
            }
            (MixtureRuntime::Switching(m), MixtureRuntimeLifecycleCheckpoint::Switching(ck)) => {
                restore_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
                m.scratch_logps = ck.scratch_logps;
                m.scratch_joint = ck.scratch_joint;
                m.scratch_weights = ck.scratch_weights;
                m.bitwise = ck.bitwise;
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_mix = ck.cached_log_mix;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
                m.update_count = ck.update_count;
            }
            (MixtureRuntime::Convex(m), MixtureRuntimeLifecycleCheckpoint::Convex(ck)) => {
                restore_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
                m.lambda = ck.lambda;
                m.scratch_logps = ck.scratch_logps;
                m.projection_scratch = ck.projection_scratch;
                m.bitwise = ck.bitwise;
                m.cached_symbol = ck.cached_symbol;
                m.cached_log_mix = ck.cached_log_mix;
                m.cache_valid = ck.cache_valid;
                m.total_log_loss = ck.total_log_loss;
                m.update_count = ck.update_count;
            }
            (MixtureRuntime::Mdl(m), MixtureRuntimeLifecycleCheckpoint::Mdl(ck)) => {
                restore_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
                m.scratch_logps = ck.scratch_logps;
                m.bitwise = ck.bitwise;
                m.total_log_loss = ck.total_log_loss;
                m.last_best = ck.last_best;
                m.cached_symbol = ck.cached_symbol;
                m.cached_best_idx = ck.cached_best_idx;
                m.cached_best_logp = ck.cached_best_logp;
                m.cache_valid = ck.cache_valid;
            }
            (MixtureRuntime::Neural(m), MixtureRuntimeLifecycleCheckpoint::Neural(ck)) => {
                restore_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
                m.neural = ck.neural;
                m.analyzer = ck.analyzer;
                m.scratch_expert_logps = ck.scratch_expert_logps;
                m.scratch_mix_weights = ck.scratch_mix_weights;
                m.bitwise = ck.bitwise;
                m.eval_cache_valid = ck.eval_cache_valid;
                m.eval_cache_full_valid = ck.eval_cache_full_valid;
                m.eval_cache_history = ck.eval_cache_history;
                m.eval_cache_symbol = ck.eval_cache_symbol;
                m.eval_cache_logp = ck.eval_cache_logp;
                m.eval_cache_mix_logps = ck.eval_cache_mix_logps;
                m.eval_cache_expert_logps = ck.eval_cache_expert_logps;
                m.total_log_loss = ck.total_log_loss;
            }
            _ => panic!("mismatched MixtureRuntime lifecycle checkpoint variant"),
        }
    }

    fn discard_lifecycle_checkpoint(
        &mut self,
        op: OnlineBytePredictorLifecycleOp,
        checkpoint: MixtureRuntimeLifecycleCheckpoint,
    ) {
        let expert_op = ExpertLifecycleOp::from_predictor_op(op);
        match (self, checkpoint) {
            (MixtureRuntime::Bayes(m), MixtureRuntimeLifecycleCheckpoint::Bayes(ck)) => {
                discard_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
            }
            (MixtureRuntime::Fading(m), MixtureRuntimeLifecycleCheckpoint::Fading(ck)) => {
                discard_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
            }
            (MixtureRuntime::Switching(m), MixtureRuntimeLifecycleCheckpoint::Switching(ck)) => {
                discard_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
            }
            (MixtureRuntime::Convex(m), MixtureRuntimeLifecycleCheckpoint::Convex(ck)) => {
                discard_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
            }
            (MixtureRuntime::Mdl(m), MixtureRuntimeLifecycleCheckpoint::Mdl(ck)) => {
                discard_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
            }
            (MixtureRuntime::Neural(m), MixtureRuntimeLifecycleCheckpoint::Neural(ck)) => {
                discard_lifecycle_experts(&mut m.experts, ck.experts, expert_op);
            }
            _ => panic!("mismatched MixtureRuntime lifecycle checkpoint variant"),
        }
    }

    pub(crate) fn clear_checkpoints_if_supported(&mut self) {
        match self {
            MixtureRuntime::Bayes(m) => clear_expert_checkpoints(&mut m.experts),
            MixtureRuntime::Fading(m) => clear_expert_checkpoints(&mut m.experts),
            MixtureRuntime::Switching(m) => clear_expert_checkpoints(&mut m.experts),
            MixtureRuntime::Convex(m) => clear_expert_checkpoints(&mut m.experts),
            MixtureRuntime::Mdl(m) => clear_expert_checkpoints(&mut m.experts),
            MixtureRuntime::Neural(m) => clear_expert_checkpoints(&mut m.experts),
        }
    }

    pub(crate) fn supports_frozen_reset(&self) -> bool {
        match self {
            MixtureRuntime::Bayes(m) => experts_support_frozen_reset(&m.experts),
            MixtureRuntime::Fading(m) => experts_support_frozen_reset(&m.experts),
            MixtureRuntime::Switching(m) => experts_support_frozen_reset(&m.experts),
            MixtureRuntime::Convex(m) => experts_support_frozen_reset(&m.experts),
            MixtureRuntime::Mdl(m) => experts_support_frozen_reset(&m.experts),
            MixtureRuntime::Neural(m) => experts_support_frozen_reset(&m.experts),
        }
    }

    pub(crate) fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        match self {
            MixtureRuntime::Bayes(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Fading(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Switching(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Convex(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Mdl(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Neural(m) => begin_expert_stream(&mut m.experts, total_symbols),
        }
    }

    pub(crate) fn begin_fresh_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        match self {
            MixtureRuntime::Bayes(m) => m.begin_fresh_stream(total_symbols),
            MixtureRuntime::Fading(m) => m.begin_fresh_stream(total_symbols),
            MixtureRuntime::Switching(m) => m.begin_fresh_stream(total_symbols),
            MixtureRuntime::Convex(m) => m.begin_fresh_stream(total_symbols),
            MixtureRuntime::Mdl(m) => m.begin_fresh_stream(total_symbols),
            MixtureRuntime::Neural(m) => m.begin_fresh_stream(total_symbols),
        }
    }

    pub(crate) fn finish_stream(&mut self) -> Result<(), String> {
        match self {
            MixtureRuntime::Bayes(m) => finish_expert_stream(&mut m.experts),
            MixtureRuntime::Fading(m) => finish_expert_stream(&mut m.experts),
            MixtureRuntime::Switching(m) => finish_expert_stream(&mut m.experts),
            MixtureRuntime::Convex(m) => finish_expert_stream(&mut m.experts),
            MixtureRuntime::Mdl(m) => finish_expert_stream(&mut m.experts),
            MixtureRuntime::Neural(m) => finish_expert_stream(&mut m.experts),
        }
    }

    pub(crate) fn reset_frozen(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        match self {
            MixtureRuntime::Bayes(m) => m.reset_frozen(total_symbols),
            MixtureRuntime::Fading(m) => m.reset_frozen(total_symbols),
            MixtureRuntime::Switching(m) => m.reset_frozen(total_symbols),
            MixtureRuntime::Convex(m) => m.reset_frozen(total_symbols),
            MixtureRuntime::Mdl(m) => m.reset_frozen(total_symbols),
            MixtureRuntime::Neural(m) => m.reset_frozen(total_symbols),
        }
    }

    /// Non-mutating log-probability (nats) for `symbol` at current state.
    pub(crate) fn peek_log_prob(&mut self, symbol: u8) -> f64 {
        match self {
            MixtureRuntime::Bayes(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Fading(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Switching(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Convex(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Mdl(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Neural(m) => m.predict_log_prob(symbol),
        }
    }

    /// Step the mixture and return log-probability (nats).
    pub(crate) fn step(&mut self, symbol: u8) -> f64 {
        match self {
            MixtureRuntime::Bayes(m) => m.step(symbol),
            MixtureRuntime::Fading(m) => m.step(symbol),
            MixtureRuntime::Switching(m) => m.step(symbol),
            MixtureRuntime::Convex(m) => m.step(symbol),
            MixtureRuntime::Mdl(m) => m.step(symbol),
            MixtureRuntime::Neural(m) => m.step(symbol),
        }
    }

    pub(crate) fn update_frozen(&mut self, symbol: u8) {
        match self {
            MixtureRuntime::Bayes(m) => m.update_frozen(symbol),
            MixtureRuntime::Fading(m) => m.update_frozen(symbol),
            MixtureRuntime::Switching(m) => m.update_frozen(symbol),
            MixtureRuntime::Convex(m) => m.update_frozen(symbol),
            MixtureRuntime::Mdl(m) => m.update_frozen(symbol),
            MixtureRuntime::Neural(m) => m.update_frozen(symbol),
        }
    }

    pub(crate) fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        match self {
            MixtureRuntime::Bayes(m) => m.fill_log_probs(out),
            MixtureRuntime::Fading(m) => m.fill_log_probs(out),
            MixtureRuntime::Switching(m) => m.fill_log_probs(out),
            MixtureRuntime::Convex(m) => m.fill_log_probs(out),
            MixtureRuntime::Mdl(m) => m.fill_log_probs(out),
            MixtureRuntime::Neural(m) => m.fill_log_probs(out),
        }
    }

    pub(crate) fn has_native_msb_byte_prefix(&self) -> bool {
        match self {
            MixtureRuntime::Bayes(m) => experts_have_native_msb_byte_prefix(&m.experts),
            MixtureRuntime::Fading(m) => experts_have_native_msb_byte_prefix(&m.experts),
            MixtureRuntime::Switching(m) => experts_have_native_msb_byte_prefix(&m.experts),
            MixtureRuntime::Convex(m) => experts_have_native_msb_byte_prefix(&m.experts),
            MixtureRuntime::Mdl(m) => experts_have_native_msb_byte_prefix(&m.experts),
            MixtureRuntime::Neural(m) => experts_have_native_msb_byte_prefix(&m.experts),
        }
    }

    pub(crate) fn begin_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
        match self {
            MixtureRuntime::Bayes(m) => {
                let weights: Vec<f64> = normalized_expert_log_weights(&m.experts);
                m.bitwise.begin(&mut m.experts, &weights)
            }
            MixtureRuntime::Fading(m) => {
                let weights: Vec<f64> = normalized_scaled_expert_log_weights(&m.experts, m.decay);
                m.bitwise.begin(&mut m.experts, &weights)
            }
            MixtureRuntime::Switching(m) => {
                let weights: Vec<f64> = normalized_expert_log_weights(&m.experts);
                m.bitwise.begin(&mut m.experts, &weights)
            }
            MixtureRuntime::Convex(m) => m.bitwise.begin(&mut m.experts, &m.lambda),
            MixtureRuntime::Mdl(m) => {
                let best_idx: usize = best_expert_index(&m.experts);
                let mut weights: Vec<f64> = vec![0.0; m.experts.len()];
                if let Some(slot) = weights.get_mut(best_idx) {
                    *slot = 1.0;
                }
                m.bitwise.begin(&mut m.experts, &weights)
            }
            MixtureRuntime::Neural(m) => {
                if m.experts.len() == 1 {
                    m.scratch_mix_weights.resize(1, 1.0);
                    m.scratch_mix_weights[0] = 1.0;
                } else {
                    m.sync_history_state();
                    m.neural.evaluate_expert_weights();
                    m.scratch_mix_weights
                        .copy_from_slice(m.neural.expert_weights());
                }
                m.bitwise.begin(&mut m.experts, &m.scratch_mix_weights)
            }
        }
    }

    pub(crate) fn abort_empty_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
        match self {
            MixtureRuntime::Bayes(m) => m.bitwise.abort_empty(&mut m.experts),
            MixtureRuntime::Fading(m) => m.bitwise.abort_empty(&mut m.experts),
            MixtureRuntime::Switching(m) => m.bitwise.abort_empty(&mut m.experts),
            MixtureRuntime::Convex(m) => m.bitwise.abort_empty(&mut m.experts),
            MixtureRuntime::Mdl(m) => m.bitwise.abort_empty(&mut m.experts),
            MixtureRuntime::Neural(m) => m.bitwise.abort_empty(&mut m.experts),
        }
    }

    pub(crate) fn native_msb_prefix_prob_one(&mut self, bit_idx: usize) -> Result<f64, String> {
        match self {
            MixtureRuntime::Bayes(m) => m.bitwise.prob_one(&mut m.experts, bit_idx),
            MixtureRuntime::Fading(m) => m.bitwise.prob_one(&mut m.experts, bit_idx),
            MixtureRuntime::Switching(m) => m.bitwise.prob_one(&mut m.experts, bit_idx),
            MixtureRuntime::Convex(m) => m.bitwise.prob_one(&mut m.experts, bit_idx),
            MixtureRuntime::Mdl(m) => m.bitwise.prob_one(&mut m.experts, bit_idx),
            MixtureRuntime::Neural(m) => m.bitwise.prob_one(&mut m.experts, bit_idx),
        }
    }

    pub(crate) fn observe_native_msb_prefix_bit(
        &mut self,
        bit_idx: usize,
        bit: bool,
    ) -> Result<(), String> {
        match self {
            MixtureRuntime::Bayes(m) => m.bitwise.observe(&mut m.experts, bit_idx, bit),
            MixtureRuntime::Fading(m) => m.bitwise.observe(&mut m.experts, bit_idx, bit),
            MixtureRuntime::Switching(m) => m.bitwise.observe(&mut m.experts, bit_idx, bit),
            MixtureRuntime::Convex(m) => m.bitwise.observe(&mut m.experts, bit_idx, bit),
            MixtureRuntime::Mdl(m) => m.bitwise.observe(&mut m.experts, bit_idx, bit),
            MixtureRuntime::Neural(m) => m.bitwise.observe(&mut m.experts, bit_idx, bit),
        }
    }

    pub(crate) fn finish_native_msb_byte_prefix(&mut self, symbol: u8) -> Result<(), String> {
        match self {
            MixtureRuntime::Bayes(m) => finish_bayes_native_prefix(m, symbol),
            MixtureRuntime::Fading(m) => finish_fading_native_prefix(m, symbol),
            MixtureRuntime::Switching(m) => finish_switching_native_prefix(m, symbol),
            MixtureRuntime::Convex(m) => finish_convex_native_prefix(m, symbol),
            MixtureRuntime::Mdl(m) => finish_mdl_native_prefix(m, symbol),
            MixtureRuntime::Neural(m) => finish_neural_native_prefix(m, symbol),
        }
    }
}

fn experts_have_native_msb_byte_prefix(experts: &[ExpertState]) -> bool {
    experts
        .iter()
        .any(|expert| expert.predictor.has_native_msb_byte_prefix())
}

fn normalized_expert_log_weights(experts: &[ExpertState]) -> Vec<f64> {
    let norm: f64 = logsumexp_weights(experts);
    experts
        .iter()
        .map(|expert| (expert.log_weight - norm).exp())
        .collect()
}

fn normalized_scaled_expert_log_weights(experts: &[ExpertState], scale: f64) -> Vec<f64> {
    let mut log_weights: Vec<f64> = experts
        .iter()
        .map(|expert| scale * expert.log_weight)
        .collect();
    let norm: f64 = logsumexp(&log_weights);
    for weight in &mut log_weights {
        *weight = (*weight - norm).exp();
    }
    log_weights
}

fn best_expert_index(experts: &[ExpertState]) -> usize {
    let mut best_idx: usize = 0;
    let mut best_loss: f64 = f64::INFINITY;
    for (idx, expert) in experts.iter().enumerate() {
        if expert.cum_log_loss < best_loss {
            best_loss = expert.cum_log_loss;
            best_idx = idx;
        }
    }
    best_idx
}

fn finish_bayes_native_prefix(m: &mut BayesMixture, symbol: u8) -> Result<(), String> {
    m.bitwise.finish_adaptive(&mut m.experts, symbol)?;
    let log_mix =
        apply_bayes_update_from_logps(&mut m.experts, &m.bitwise.logps, &mut m.scratch_mix);
    m.cache_valid = false;
    m.total_log_loss -= log_mix;
    Ok(())
}

fn finish_fading_native_prefix(m: &mut FadingBayesMixture, symbol: u8) -> Result<(), String> {
    m.bitwise.finish_adaptive(&mut m.experts, symbol)?;
    let log_predictive = apply_fading_update_from_logps(
        &mut m.experts,
        &m.bitwise.logps,
        &mut m.scratch_mix,
        m.decay,
    );
    m.cache_valid = false;
    m.total_log_loss -= log_predictive;
    Ok(())
}

fn finish_switching_native_prefix(m: &mut SwitchingMixture, symbol: u8) -> Result<(), String> {
    m.bitwise.finish_adaptive(&mut m.experts, symbol)?;
    let log_mix = apply_switching_update_from_logps(
        &mut m.experts,
        &m.bitwise.logps,
        &mut m.scratch_joint,
        &mut m.scratch_weights,
        &m.prior,
        m.schedule,
        m.alpha,
        &mut m.update_count,
    );
    m.cache_valid = false;
    m.total_log_loss -= log_mix;
    Ok(())
}

fn finish_convex_native_prefix(m: &mut ConvexMixture, symbol: u8) -> Result<(), String> {
    m.bitwise.finish_adaptive(&mut m.experts, symbol)?;
    let log_mix = apply_convex_update_from_logps(
        &mut m.experts,
        &m.bitwise.logps,
        &mut m.lambda,
        &mut m.projection_scratch,
        m.schedule,
        m.alpha,
        &mut m.update_count,
    );
    m.cache_valid = false;
    m.total_log_loss -= log_mix;
    Ok(())
}

fn finish_mdl_native_prefix(m: &mut MdlSelector, symbol: u8) -> Result<(), String> {
    let best_idx: usize = best_expert_index(&m.experts);
    m.bitwise.finish_adaptive(&mut m.experts, symbol)?;
    let logp =
        apply_mdl_update_from_logps(&mut m.experts, &m.bitwise.logps, best_idx, &mut m.last_best);
    m.cache_valid = false;
    m.total_log_loss -= logp;
    Ok(())
}

fn finish_neural_native_prefix(m: &mut NeuralMixture, symbol: u8) -> Result<(), String> {
    m.bitwise.finish_adaptive(&mut m.experts, symbol)?;
    m.scratch_expert_logps
        .copy_from_slice(&m.bitwise.logps[..m.experts.len()]);
    for idx in 0..m.experts.len() {
        m.experts[idx].cum_log_loss -= m.scratch_expert_logps[idx];
    }
    let logp: f64 = m
        .bitwise
        .weights
        .iter()
        .zip(m.bitwise.logps.iter())
        .map(|(&w, &lp)| w * lp.exp())
        .sum::<f64>()
        .max(m.min_prob)
        .ln();
    finish_neural_update_from_logps(m, symbol, logp, m.experts.len() > 1);
    m.eval_cache_history = m.neural.history_state();
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpertLifecycleOp {
    BeginStream,
    BeginFreshStream,
    ResetFrozen,
    FinishStream,
}

impl ExpertLifecycleOp {
    fn as_str(self) -> &'static str {
        match self {
            Self::BeginStream => "begin_stream",
            Self::BeginFreshStream => "begin_fresh_stream",
            Self::ResetFrozen => "reset_frozen",
            Self::FinishStream => "finish_stream",
        }
    }

    fn to_predictor_op(self) -> OnlineBytePredictorLifecycleOp {
        match self {
            Self::BeginStream => OnlineBytePredictorLifecycleOp::BeginStream,
            Self::BeginFreshStream => OnlineBytePredictorLifecycleOp::BeginFreshStream,
            Self::ResetFrozen => OnlineBytePredictorLifecycleOp::ResetFrozen,
            Self::FinishStream => OnlineBytePredictorLifecycleOp::FinishStream,
        }
    }

    fn from_predictor_op(op: OnlineBytePredictorLifecycleOp) -> Self {
        match op {
            OnlineBytePredictorLifecycleOp::BeginStream => Self::BeginStream,
            OnlineBytePredictorLifecycleOp::BeginFreshStream => Self::BeginFreshStream,
            OnlineBytePredictorLifecycleOp::ResetFrozen => Self::ResetFrozen,
            OnlineBytePredictorLifecycleOp::FinishStream => Self::FinishStream,
        }
    }
}

/// Apply a fallible lifecycle operation to every expert all-or-nothing.
///
/// Lifecycle hooks may mutate non-journaled state such as CTW history, neural
/// online-policy buffers, or wrapper caches before reporting an error. The
/// update checkpoints used for speculative byte/bit prediction are therefore not
/// sufficient here: rollback must restore the whole expert state that existed
/// before the lifecycle operation began. A full expert snapshot is faithful by
/// construction, and neural model weights are already `Arc`-shared by their
/// backends, so this does not deep-copy those parameters.
fn transact_expert_lifecycle(
    experts: &mut [ExpertState],
    op: ExpertLifecycleOp,
    mut apply: impl FnMut(&mut ExpertState) -> Result<(), String>,
) -> Result<(), String> {
    if experts.is_empty() {
        return Ok(());
    }
    let mut backups: Vec<(usize, ExpertLifecycleToken)> = Vec::with_capacity(experts.len());
    for idx in 0..experts.len() {
        let name = experts[idx].name.clone();
        let token = experts[idx].snapshot_lifecycle(op);
        backups.push((idx, token));
        if let Err(err) = apply(&mut experts[idx]) {
            for (restore_idx, token) in backups.into_iter().rev() {
                experts[restore_idx].restore_lifecycle(op, token);
            }
            return Err(format!(
                "mixture expert lifecycle {} failed for expert #{idx} '{name}': {err}",
                op.as_str()
            ));
        }
    }
    for (idx, token) in backups {
        experts[idx].discard_lifecycle(op, token);
    }
    Ok(())
}

fn begin_expert_stream(
    experts: &mut [ExpertState],
    total_symbols: Option<u64>,
) -> Result<(), String> {
    transact_expert_lifecycle(experts, ExpertLifecycleOp::BeginStream, |expert| {
        expert.begin_stream(total_symbols)
    })
}

fn begin_expert_fresh_stream(
    experts: &mut [ExpertState],
    total_symbols: Option<u64>,
) -> Result<(), String> {
    transact_expert_lifecycle(experts, ExpertLifecycleOp::BeginFreshStream, |expert| {
        expert.begin_fresh_stream(total_symbols)
    })
}

fn reset_expert_frozen_stream(
    experts: &mut [ExpertState],
    total_symbols: Option<u64>,
) -> Result<(), String> {
    transact_expert_lifecycle(experts, ExpertLifecycleOp::ResetFrozen, |expert| {
        expert.reset_frozen(total_symbols)
    })
}

fn experts_support_frozen_reset(experts: &[ExpertState]) -> bool {
    experts
        .iter()
        .all(|expert| expert.predictor.supports_frozen_reset())
}

fn finish_expert_stream(experts: &mut [ExpertState]) -> Result<(), String> {
    transact_expert_lifecycle(experts, ExpertLifecycleOp::FinishStream, |expert| {
        expert.finish_stream()
    })
}

#[cfg(test)]
pub(crate) fn build_mixture_runtime(
    spec: &MixtureSpec,
    experts: &[ExpertConfig],
) -> Result<MixtureRuntime, String> {
    spec.validate().map_err(|err| err.to_string())?;
    build_mixture_runtime_from_fields(spec.kind, spec.schedule, spec.alpha, spec.decay, experts)
}

#[cfg(feature = "backend-mixture")]
pub(crate) fn build_mixture_runtime_from_compiled(
    backend: &CompiledRateBackend,
    experts: &[ExpertConfig],
) -> Result<MixtureRuntime, String> {
    let crate::spec::core::RateBackendPlan::Mixture {
        kind,
        schedule,
        alpha,
        decay,
        ..
    } = backend.plan()
    else {
        return Err("compiled backend is not a mixture backend".to_string());
    };
    build_mixture_runtime_from_fields(*kind, *schedule, *alpha, *decay, experts)
}

fn build_mixture_runtime_from_fields(
    kind: MixtureKind,
    schedule: MixtureScheduleMode,
    alpha: f64,
    decay: Option<f64>,
    experts: &[ExpertConfig],
) -> Result<MixtureRuntime, String> {
    match kind {
        MixtureKind::Bayes => Ok(MixtureRuntime::Bayes(BayesMixture::new(experts))),
        MixtureKind::FadingBayes => {
            let decay = decay.ok_or_else(|| "fading Bayes mixture requires decay".to_string())?;
            Ok(MixtureRuntime::Fading(FadingBayesMixture::new(
                experts, decay,
            )))
        }
        MixtureKind::Switching => Ok(MixtureRuntime::Switching(SwitchingMixture::new(
            experts, alpha, schedule,
        ))),
        MixtureKind::Convex => Ok(MixtureRuntime::Convex(ConvexMixture::new(
            experts, alpha, schedule,
        ))),
        MixtureKind::Mdl => Ok(MixtureRuntime::Mdl(MdlSelector::new(experts))),
        MixtureKind::Neural => Ok(MixtureRuntime::Neural(NeuralMixture::new(experts, alpha))),
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct LifecycleMockPredict {
        state: usize,
        fail: Option<ExpertLifecycleOp>,
    }

    impl LifecycleMockPredict {
        fn apply(&mut self, op: ExpertLifecycleOp, next_state: usize) -> Result<(), String> {
            self.state = next_state;
            if self.fail == Some(op) {
                Err(format!("{} failed after mutation", op.as_str()))
            } else {
                Ok(())
            }
        }
    }

    impl OnlineBytePredictor for LifecycleMockPredict {
        fn begin_stream(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
            self.apply(ExpertLifecycleOp::BeginStream, 11)
        }

        fn begin_fresh_stream(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
            self.apply(ExpertLifecycleOp::BeginFreshStream, 22)
        }

        fn reset_frozen(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
            self.apply(ExpertLifecycleOp::ResetFrozen, 33)
        }

        fn finish_stream(&mut self) -> Result<(), String> {
            self.apply(ExpertLifecycleOp::FinishStream, 44)
        }

        fn log_prob(&mut self, _symbol: u8) -> f64 {
            self.state as f64
        }

        fn update(&mut self, _symbol: u8) {}
    }

    fn lifecycle_expert(name: &str, state: usize, fail: Option<ExpertLifecycleOp>) -> ExpertState {
        ExpertState {
            name: name.to_string(),
            log_weight: 0.0,
            log_prior: 0.0,
            predictor: ExpertPredictor::generic(Box::new(LifecycleMockPredict { state, fail })),
            cum_log_loss: 0.0,
        }
    }

    fn lifecycle_state(expert: &mut ExpertState) -> usize {
        expert.log_prob(0) as usize
    }

    fn assert_lifecycle_failure_rolls_back(op: ExpertLifecycleOp) {
        let mut experts: Vec<ExpertState> = vec![
            lifecycle_expert("mutated", 1, None),
            lifecycle_expert("failing", 2, Some(op)),
        ];

        let err = match op {
            ExpertLifecycleOp::BeginStream => begin_expert_stream(&mut experts, Some(8)),
            ExpertLifecycleOp::BeginFreshStream => begin_expert_fresh_stream(&mut experts, Some(8)),
            ExpertLifecycleOp::ResetFrozen => reset_expert_frozen_stream(&mut experts, Some(8)),
            ExpertLifecycleOp::FinishStream => finish_expert_stream(&mut experts),
        }
        .expect_err("second expert should fail after mutating");

        assert!(
            err.contains(op.as_str()),
            "error should name lifecycle operation: {err}"
        );
        assert!(
            err.contains("expert #1 'failing'"),
            "error should identify failing expert: {err}"
        );
        assert!(
            err.contains("failed after mutation"),
            "error should preserve source context: {err}"
        );
        assert_eq!(
            lifecycle_state(&mut experts[0]),
            1,
            "earlier successful expert mutation must be rolled back"
        );
        assert_eq!(
            lifecycle_state(&mut experts[1]),
            2,
            "failing expert mutation must be rolled back"
        );
    }

    #[test]
    fn lifecycle_begin_stream_failure_rolls_back_all_experts() {
        assert_lifecycle_failure_rolls_back(ExpertLifecycleOp::BeginStream);
    }

    #[test]
    fn lifecycle_begin_fresh_stream_failure_rolls_back_all_experts() {
        assert_lifecycle_failure_rolls_back(ExpertLifecycleOp::BeginFreshStream);
    }

    #[test]
    fn lifecycle_reset_frozen_failure_rolls_back_all_experts() {
        assert_lifecycle_failure_rolls_back(ExpertLifecycleOp::ResetFrozen);
    }

    #[test]
    fn lifecycle_finish_stream_failure_rolls_back_all_experts() {
        assert_lifecycle_failure_rolls_back(ExpertLifecycleOp::FinishStream);
    }

    struct CloneCountingLifecyclePredict {
        state: usize,
        fail: Option<ExpertLifecycleOp>,
        clones: Arc<AtomicUsize>,
    }

    impl Clone for CloneCountingLifecyclePredict {
        fn clone(&self) -> Self {
            self.clones.fetch_add(1, Ordering::Relaxed);
            Self {
                state: self.state,
                fail: self.fail,
                clones: self.clones.clone(),
            }
        }
    }

    impl OnlineBytePredictor for CloneCountingLifecyclePredict {
        fn begin_stream(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
            self.state = 99;
            if self.fail == Some(ExpertLifecycleOp::BeginStream) {
                Err("failed after mutation".to_string())
            } else {
                Ok(())
            }
        }

        fn log_prob(&mut self, _symbol: u8) -> f64 {
            self.state as f64
        }

        fn update(&mut self, _symbol: u8) {}
    }

    #[test]
    fn lifecycle_failure_snapshots_only_attempted_experts() {
        let clones = Arc::new(AtomicUsize::new(0));
        let mut experts = vec![
            ExpertState {
                name: "failing".to_string(),
                log_weight: 0.0,
                log_prior: 0.0,
                predictor: ExpertPredictor::generic(Box::new(CloneCountingLifecyclePredict {
                    state: 1,
                    fail: Some(ExpertLifecycleOp::BeginStream),
                    clones: clones.clone(),
                })),
                cum_log_loss: 0.0,
            },
            ExpertState {
                name: "unreached".to_string(),
                log_weight: 0.0,
                log_prior: 0.0,
                predictor: ExpertPredictor::generic(Box::new(CloneCountingLifecyclePredict {
                    state: 2,
                    fail: None,
                    clones: clones.clone(),
                })),
                cum_log_loss: 0.0,
            },
        ];

        begin_expert_stream(&mut experts, Some(1)).expect_err("first expert should fail");

        assert_eq!(
            clones.load(Ordering::Relaxed),
            1,
            "transaction must not snapshot experts after the first failure"
        );
        assert_eq!(lifecycle_state(&mut experts[0]), 1);
        assert_eq!(lifecycle_state(&mut experts[1]), 2);
    }
}

#[cfg(all(test, feature = "all-backends"))]
mod tests {
    use super::*;
    use crate::api::{CalibratedSpec, CalibrationContextKind};
    use std::sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct AlwaysPredict {
        byte: u8,
    }

    impl OnlineBytePredictor for AlwaysPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            if symbol == self.byte {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        }

        fn update(&mut self, _symbol: u8) {}
    }

    #[derive(Clone)]
    struct FixedProbPredict {
        prob_zero: f64,
    }

    impl OnlineBytePredictor for FixedProbPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            let p = if symbol == 0 {
                self.prob_zero
            } else {
                (1.0 - self.prob_zero) / 255.0
            };
            p.ln()
        }

        fn update(&mut self, _symbol: u8) {}
    }

    fn weighted_cfg(name: &'static str, weight: f64, prob_zero: f64) -> ExpertConfig {
        ExpertConfig::new(name, weight.ln(), move || {
            Box::new(FixedProbPredict { prob_zero })
        })
    }

    fn assert_weights_close(actual: &[f64], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-12,
                "{label}[{index}]: expected {e}, got {a}"
            );
        }
    }

    fn assert_expert_losses_reset(experts: &[ExpertState], label: &str) {
        for expert in experts {
            assert!(
                expert.cum_log_loss.abs() < 1e-12,
                "{label} expert '{}' loss should reset, got {}",
                expert.name,
                expert.cum_log_loss
            );
        }
    }

    #[derive(Clone)]
    struct NativeBitProbPredict {
        prob_one: f64,
    }

    impl OnlineBytePredictor for NativeBitProbPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            let p = if symbol == 0 {
                1.0 - self.prob_one
            } else {
                self.prob_one / 255.0
            };
            p.max(DEFAULT_MIN_PROB).ln()
        }

        fn update(&mut self, _symbol: u8) {}

        fn has_native_msb_byte_prefix(&self) -> bool {
            true
        }

        fn begin_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
            Ok(true)
        }

        fn native_msb_prefix_prob_one(&mut self, _bit_idx: usize) -> Result<f64, String> {
            Ok(self.prob_one)
        }

        fn observe_native_msb_prefix_bit(
            &mut self,
            _bit_idx: usize,
            _bit: bool,
        ) -> Result<(), String> {
            Ok(())
        }

        fn finish_native_msb_byte_prefix(&mut self, _symbol: u8) -> Result<(), String> {
            Ok(())
        }

        // Dummy checkpoint methods so that MixtureBitPrefixState::begin succeeds
        // for this mock (which advertises native MSB support). These are safe
        // no-ops: the test mock performs no real mutation, and begin discards
        // the captured checkpoints on the happy path after all experts prepare.
        fn checkpoint_if_supported(&mut self) -> Option<OnlineBytePredictorCheckpoint> {
            Some(OnlineBytePredictorCheckpoint::rate_backend(
                RateBackendPredictorCheckpoint::Full(Box::new(RateBackendPredictor::Disabled {
                    reason: "dummy_test_checkpoint".to_string(),
                })),
            ))
        }

        fn restore_checkpoint_if_supported(
            &mut self,
            _checkpoint: &OnlineBytePredictorCheckpoint,
        ) -> bool {
            true
        }

        fn discard_checkpoint_if_supported(
            &mut self,
            _checkpoint: OnlineBytePredictorCheckpoint,
        ) -> bool {
            true
        }
    }

    #[derive(Clone)]
    struct NativeWithoutCheckpointPredict {
        begin_calls: Arc<AtomicUsize>,
    }

    impl OnlineBytePredictor for NativeWithoutCheckpointPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            if symbol == 0 { 0.0 } else { f64::NEG_INFINITY }
        }

        fn update(&mut self, _symbol: u8) {}

        fn has_native_msb_byte_prefix(&self) -> bool {
            true
        }

        fn begin_native_msb_byte_prefix(&mut self) -> Result<bool, String> {
            self.begin_calls.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        }
    }

    #[test]
    fn mixture_bit_prefix_observe_without_prob_one_primes_current_bit() {
        let configs = [
            ExpertConfig::uniform("high", || Box::new(NativeBitProbPredict { prob_one: 0.9 })),
            ExpertConfig::uniform("low", || Box::new(NativeBitProbPredict { prob_one: 0.1 })),
        ];
        let mut experts: Vec<ExpertState> = configs.iter().map(ExpertConfig::build).collect();
        let mut bitwise = MixtureBitPrefixState::default();
        assert!(
            bitwise.begin(&mut experts, &[0.5, 0.5]).expect("begin"),
            "native prefix should activate when experts support native bit stepping"
        );

        // Drive full 8-bit sequence (per MixtureBitPrefixState expected_bit_idx contract
        // and finish_adaptive validation at 733) to exercise the "observe without prior
        // prob_one" priming path on every step, then finish. This satisfies the state
        // machine while preserving the original test intent (priming on observe-only
        // updates + final log-likelihood distinction). Single-observe + immediate finish
        // violated the sequential contract (now enforced post-mock fix).
        let symbol: u8 = 0xFF; // all-1s so high-p1 (0.9) expert has higher final likelihood than low-p1 (0.1) after 8 steps (preserves original distinction intent)
        for bit_idx in 0..8 {
            let bit = (symbol & (1u8 << (7 - bit_idx))) != 0;
            bitwise
                .observe(&mut experts, bit_idx, bit)
                .expect("observe without prior prob_one");
            if bit_idx == 0 {
                assert!(
                    (bitwise.likelihoods[0] - 0.9).abs() < 1e-12,
                    "high-prob expert likelihood should use freshly primed p1"
                );
                assert!(
                    (bitwise.likelihoods[1] - 0.1).abs() < 1e-12,
                    "low-prob expert likelihood should use freshly primed p1"
                );
                assert!(
                    bitwise.primed_bit_idx.is_none(),
                    "observe should clear priming for the next bit"
                );
            }
        }

        bitwise
            .finish_adaptive(&mut experts, symbol)
            .expect("finish adaptive");
        assert!(
            bitwise.logps[0] > bitwise.logps[1],
            "log-likelihoods should distinguish disagreeing experts after observe-only updates"
        );
    }

    #[test]
    fn mixture_bit_prefix_requires_monotone_bit_indices() {
        let configs = [ExpertConfig::ctw("left", 4), ExpertConfig::ctw("right", 5)];
        let mut experts: Vec<ExpertState> = configs.iter().map(ExpertConfig::build).collect();
        let mut bitwise = MixtureBitPrefixState::default();
        assert!(bitwise.begin(&mut experts, &[0.5, 0.5]).expect("begin"));

        let err = bitwise
            .prob_one(&mut experts, 1)
            .expect_err("bit 1 cannot be queried before bit 0 is observed");
        assert!(err.contains("expected 0"));

        bitwise
            .observe(&mut experts, 0, true)
            .expect("observe bit 0");
        let err = bitwise
            .observe(&mut experts, 0, false)
            .expect_err("duplicate observe must be rejected");
        assert!(err.contains("expected 1"));

        let err = bitwise
            .prob_one(&mut experts, 8)
            .expect_err("bit index 8 must be rejected");
        assert!(err.contains("out of range"));
    }

    #[test]
    fn mixture_bit_prefix_skips_native_mode_without_checkpoint_support() {
        let begin_calls = Arc::new(AtomicUsize::new(0));
        let shared = Arc::clone(&begin_calls);
        let configs = [ExpertConfig::uniform("native", move || {
            Box::new(NativeWithoutCheckpointPredict {
                begin_calls: Arc::clone(&shared),
            })
        })];
        let mut experts: Vec<ExpertState> = configs.iter().map(ExpertConfig::build).collect();
        let mut bitwise = MixtureBitPrefixState::default();

        assert!(
            !bitwise.begin(&mut experts, &[1.0]).expect("begin"),
            "native prefix should be disabled when rollback checkpoints are unavailable"
        );
        assert_eq!(
            begin_calls.load(Ordering::Relaxed),
            0,
            "unsupported native experts must not be entered speculatively"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn ctw_native_prefix_rejects_invalid_bit_indices() {
        let mut predictor =
            RateBackendPredictor::from_backend(RateBackend::Ctw { depth: 4 }, DEFAULT_MIN_PROB);
        assert!(
            predictor
                .begin_native_msb_byte_prefix()
                .expect("begin native prefix"),
            "ctw byte predictor should support native prefix stepping"
        );

        let err = predictor
            .native_msb_prefix_prob_one(8)
            .expect_err("bit index 8 must be rejected");
        assert!(err.contains("out of range"));

        let err = predictor
            .observe_native_msb_prefix_bit(1, true)
            .expect_err("bit 1 cannot be observed before bit 0");
        assert!(err.contains("expected 0"));

        predictor
            .native_msb_prefix_prob_one(0)
            .expect("query bit 0");
        predictor
            .observe_native_msb_prefix_bit(0, true)
            .expect("observe bit 0");
        let err = predictor
            .finish_native_msb_byte_prefix(0b1000_0000)
            .expect_err("finish must require a complete byte");
        assert!(err.contains("requires 8 observed bits"));
    }

    #[cfg(all(feature = "backend-calibrated", feature = "backend-ctw"))]
    #[test]
    fn calibrated_predictor_exposes_native_prefix_sse_updates() {
        let backend = RateBackend::Calibrated {
            spec: Arc::new(CalibratedSpec::new(
                RateBackend::Ctw { depth: 6 },
                CalibrationContextKind::TextRepeat,
            )),
        };
        let mut predictor = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);
        for &byte in b"calibrated predictor prefix history" {
            predictor.update(byte);
        }

        assert!(predictor.has_native_msb_byte_prefix());
        assert!(predictor.begin_native_msb_byte_prefix().expect("begin"));

        let symbol = b'Z';
        let mut logp = 0.0f64;
        for bit_idx in 0..8usize {
            let p1 = predictor
                .native_msb_prefix_prob_one(bit_idx)
                .expect("calibrated bit probability");
            assert!(p1 > 0.0 && p1 < 1.0 && p1.is_finite(), "p1={p1}");
            let bit = (symbol & (1u8 << (7 - bit_idx))) != 0;
            let p_bit = if bit { p1 } else { 1.0 - p1 };
            logp += p_bit.ln();
            predictor
                .observe_native_msb_prefix_bit(bit_idx, bit)
                .expect("observe calibrated bit");
        }
        predictor
            .finish_native_msb_byte_prefix(symbol)
            .expect("finish calibrated prefix");
        assert!(logp.is_finite());

        let mut row = [0.0f64; 256];
        predictor.fill_log_probs(&mut row);
        let mass: f64 = row.iter().map(|lp| lp.exp()).sum();
        assert!(
            (mass - 1.0).abs() < 1e-9,
            "calibrated byte distribution should normalize after bitwise update: {mass}"
        );
    }

    #[cfg(all(feature = "backend-calibrated", feature = "backend-match"))]
    #[test]
    fn calibrated_pdf_fallback_prefix_rejects_bad_lifecycle_without_resetting_adapter() {
        let backend = RateBackend::Calibrated {
            spec: Arc::new(CalibratedSpec::new(
                RateBackend::Match {
                    hash_bits: 18,
                    min_len: 3,
                    max_len: 64,
                    base_mix: 0.08,
                    confidence_scale: 1.0,
                },
                CalibrationContextKind::ByteClass,
            )),
        };
        let mut predictor = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);

        assert!(predictor.begin_native_msb_byte_prefix().expect("begin"));
        let finish_err = predictor
            .finish_native_msb_byte_prefix(0)
            .expect_err("prefix finish before 8 bits must be rejected");
        assert!(finish_err.contains("requires 8 observed bits, got 0"));
        assert!(
            predictor
                .abort_empty_native_msb_byte_prefix()
                .expect("empty abort after rejected finish"),
            "empty abort should close the still-active session"
        );

        assert!(predictor.begin_native_msb_byte_prefix().expect("begin"));
        predictor
            .native_msb_prefix_prob_one(0)
            .expect("predict first bit");
        predictor
            .observe_native_msb_prefix_bit(0, true)
            .expect("observe first bit");
        let abort_err = predictor
            .abort_empty_native_msb_byte_prefix()
            .expect_err("non-empty abort must be rejected before adapter reset");
        assert!(abort_err.contains("got 1"));

        match &predictor {
            RateBackendPredictor::Calibrated { bitwise, .. } => match &bitwise.kind {
                BytePrefixStepStateKind::PdfPrefix { range, .. } => {
                    assert_eq!(range.lo(), 128);
                    assert_eq!(range.hi(), 256);
                }
                BytePrefixStepStateKind::Native => {
                    panic!("match-backed calibrated predictor should use PDF prefix fallback")
                }
            },
            _ => panic!("expected calibrated predictor"),
        }

        for bit_idx in 1..8usize {
            predictor
                .native_msb_prefix_prob_one(bit_idx)
                .expect("predict remaining bit");
            predictor
                .observe_native_msb_prefix_bit(bit_idx, false)
                .expect("observe remaining bit");
        }
        predictor
            .finish_native_msb_byte_prefix(0x80)
            .expect("complete prefix should finish after rejected abort");
    }

    #[test]
    fn bayes_mixture_prefers_correct_expert() {
        let configs = vec![
            ExpertConfig::uniform("zero", || Box::new(AlwaysPredict { byte: 0 })),
            ExpertConfig::uniform("one", || Box::new(AlwaysPredict { byte: 1 })),
        ];
        let mut mix = BayesMixture::new(&configs);
        for _ in 0..10 {
            mix.step(0);
        }
        let post = mix.posterior();
        assert!(post[0] > 0.999);
        assert!(post[1] < 1e-6);
    }

    fn counting_cfg(name: &'static str, calls: Arc<AtomicUsize>) -> ExpertConfig {
        ExpertConfig::uniform(name, move || {
            Box::new(CountingPredict {
                calls: calls.clone(),
            })
        })
    }

    #[test]
    fn bayes_predict_then_step_reuses_cached_log_probs() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mix = BayesMixture::new(&[
            counting_cfg("c0", c0.clone()),
            counting_cfg("c1", c1.clone()),
        ]);
        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);
        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn fading_predict_then_step_reuses_cached_log_probs() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mix = FadingBayesMixture::new(
            &[
                counting_cfg("c0", c0.clone()),
                counting_cfg("c1", c1.clone()),
            ],
            0.95,
        );
        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);
        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn switching_predict_then_step_reuses_cached_log_probs() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mix = SwitchingMixture::new(
            &[
                counting_cfg("c0", c0.clone()),
                counting_cfg("c1", c1.clone()),
            ],
            0.05,
            MixtureScheduleMode::Default,
        );
        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);
        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn switching_mixture_matches_fixed_share_update_for_uniform_prior() {
        let configs = vec![weighted_cfg("a", 0.5, 0.8), weighted_cfg("b", 0.5, 0.3)];
        let alpha = 0.2;
        let mut mix = SwitchingMixture::new(&configs, alpha, MixtureScheduleMode::Default);

        let predicted = mix.predict_log_prob(0).exp();
        assert!((predicted - 0.55).abs() < 1e-12, "predicted={predicted}");

        let observed = mix.step(0).exp();
        assert!((observed - 0.55).abs() < 1e-12, "observed={observed}");

        let post = mix.posterior();
        let posterior_a = 0.5 * 0.8 / 0.55;
        let posterior_b = 0.5 * 0.3 / 0.55;
        let expected_a = (1.0 - alpha) * posterior_a + alpha * posterior_b;
        let expected_b = (1.0 - alpha) * posterior_b + alpha * posterior_a;
        assert!(
            (post[0] - expected_a).abs() < 1e-12 && (post[1] - expected_b).abs() < 1e-12,
            "expected [{expected_a}, {expected_b}], got {:?}",
            post
        );
    }

    #[test]
    fn switching_mixture_switches_according_to_prior_over_other_experts() {
        let configs = vec![
            weighted_cfg("a", 0.5, 0.75),
            weighted_cfg("b", 0.3, 0.25),
            weighted_cfg("c", 0.2, 0.60),
        ];
        let alpha = 0.15;
        let mut mix = SwitchingMixture::new(&configs, alpha, MixtureScheduleMode::Default);

        let _ = mix.step(0);
        let post = mix.posterior();

        let current = [0.5_f64, 0.3, 0.2];
        let likelihood = [0.75_f64, 0.25, 0.60];
        let mix_prob = current
            .iter()
            .zip(likelihood.iter())
            .map(|(w, p)| w * p)
            .sum::<f64>();
        let posterior = [
            current[0] * likelihood[0] / mix_prob,
            current[1] * likelihood[1] / mix_prob,
            current[2] * likelihood[2] / mix_prob,
        ];
        let prior = [0.5_f64, 0.3, 0.2];
        let mut expected = [0.0_f64; 3];
        for j in 0..3 {
            let stay = (1.0 - alpha) * posterior[j];
            let switch_in = alpha
                * prior[j]
                * (0..3)
                    .filter(|&k| k != j)
                    .map(|k| posterior[k] / (1.0 - prior[k]))
                    .sum::<f64>();
            expected[j] = stay + switch_in;
        }

        for i in 0..3 {
            assert!(
                (post[i] - expected[i]).abs() < 1e-12,
                "expert {i}: expected {} got {}",
                expected[i],
                post[i]
            );
        }
    }

    #[test]
    fn switching_theorem_schedule_uses_one_over_t() {
        assert!(
            (switching_alpha_for_update(MixtureScheduleMode::Theorem, 0.99, 0) - 0.5).abs() < 1e-12
        );
        assert!(
            (switching_alpha_for_update(MixtureScheduleMode::Theorem, 0.99, 1) - (1.0 / 3.0)).abs()
                < 1e-12
        );

        let configs = vec![weighted_cfg("a", 0.5, 0.8), weighted_cfg("b", 0.5, 0.3)];
        let mut mix = SwitchingMixture::new(&configs, 0.99, MixtureScheduleMode::Theorem);
        let _ = mix.step(0);
        let post = mix.posterior();
        let posterior_a = 0.5 * 0.8 / 0.55;
        let posterior_b = 0.5 * 0.3 / 0.55;
        let expected_a = 0.5 * posterior_a + 0.5 * posterior_b;
        let expected_b = expected_a;
        assert!((post[0] - expected_a).abs() < 1e-12);
        assert!((post[1] - expected_b).abs() < 1e-12);
    }

    #[test]
    fn convex_theorem_schedule_uses_paper_step_size() {
        let eta = convex_step_size_for_update(MixtureScheduleMode::Theorem, 9.0, 1);
        assert!((eta - DEFAULT_MIN_PROB).abs() < 1e-18);

        let configs = vec![weighted_cfg("a", 0.5, 0.8), weighted_cfg("b", 0.5, 0.3)];
        let mut mix = ConvexMixture::new(&configs, 9.0, MixtureScheduleMode::Theorem);
        let observed = mix.step(0).exp();
        assert!((observed - 0.55).abs() < 1e-12, "observed={observed}");

        let expected = [
            0.5 + eta * ((0.8 / 0.55) - 1.0),
            0.5 + eta * ((0.3 / 0.55) - 1.0),
        ];
        assert!((mix.lambda[0] - expected[0]).abs() < 1e-12);
        assert!((mix.lambda[1] - expected[1]).abs() < 1e-12);
    }

    #[test]
    fn mixture_begin_fresh_stream_resets_wrapper_state_to_priors() {
        let configs = vec![weighted_cfg("a", 0.7, 0.9), weighted_cfg("b", 0.3, 0.2)];
        let prior = [0.7_f64, 0.3_f64];

        let mut bayes = BayesMixture::new(&configs);
        let _ = bayes.step(0);
        assert!(bayes.posterior()[0] > prior[0]);
        bayes.begin_fresh_stream(Some(1)).expect("bayes fresh");
        assert_weights_close(&bayes.posterior(), &prior, "bayes posterior");
        assert_expert_losses_reset(&bayes.experts, "bayes");
        assert_eq!(bayes.total_log_loss(), 0.0);

        let mut fading = FadingBayesMixture::new(&configs, 0.8);
        let _ = fading.step(0);
        assert!(fading.posterior()[0] > prior[0]);
        fading.begin_fresh_stream(Some(1)).expect("fading fresh");
        assert_weights_close(&fading.posterior(), &prior, "fading posterior");
        assert_expert_losses_reset(&fading.experts, "fading");
        assert_eq!(fading.total_log_loss(), 0.0);

        let mut switching = SwitchingMixture::new(&configs, 0.15, MixtureScheduleMode::Default);
        let _ = switching.step(0);
        assert!(switching.posterior()[0] > prior[0]);
        switching
            .begin_fresh_stream(Some(1))
            .expect("switching fresh");
        assert_weights_close(&switching.posterior(), &prior, "switching posterior");
        assert_expert_losses_reset(&switching.experts, "switching");
        assert_eq!(switching.update_count, 0);
        assert_eq!(switching.total_log_loss(), 0.0);

        let mut convex = ConvexMixture::new(&configs, 0.2, MixtureScheduleMode::Default);
        let _ = convex.step(0);
        assert!(convex.lambda[0] > prior[0]);
        convex.begin_fresh_stream(Some(1)).expect("convex fresh");
        assert_weights_close(&convex.lambda, &prior, "convex lambda");
        assert_expert_losses_reset(&convex.experts, "convex");
        assert_eq!(convex.update_count, 0);
        assert_eq!(convex.total_log_loss, 0.0);

        let mut mdl = MdlSelector::new(&configs);
        let _ = mdl.step(0);
        assert!(mdl.experts.iter().any(|expert| expert.cum_log_loss > 0.0));
        mdl.begin_fresh_stream(Some(1)).expect("mdl fresh");
        assert_expert_losses_reset(&mdl.experts, "mdl");
        assert_eq!(mdl.best_index(), 0);
        assert_eq!(mdl.total_log_loss(), 0.0);

        let mut neural = NeuralMixture::new(&configs, 0.05);
        let mut fresh_neural = NeuralMixture::new(&configs, 0.05);
        let _ = neural.step(0);
        neural.begin_fresh_stream(Some(1)).expect("neural fresh");
        let reset_logp = neural.predict_log_prob(0);
        let fresh_logp = fresh_neural.predict_log_prob(0);
        assert!(
            (reset_logp - fresh_logp).abs() < 1e-12,
            "neural wrapper state should match a fresh wrapper after restart"
        );
        assert_expert_losses_reset(&neural.experts, "neural");
        assert_eq!(neural.total_log_loss(), 0.0);
    }

    #[test]
    fn mdl_predict_then_step_reuses_best_expert_log_prob() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mdl = MdlSelector::new(&[
            counting_cfg("c0", c0.clone()),
            counting_cfg("c1", c1.clone()),
        ]);
        let _ = mdl.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 1);
        let _ = mdl.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, 2);
    }

    #[test]
    fn neural_mixture_adapts_to_correct_symbol() {
        let configs = vec![
            ExpertConfig::uniform("zero", || Box::new(AlwaysPredict { byte: 0 })),
            ExpertConfig::uniform("one", || Box::new(AlwaysPredict { byte: 1 })),
        ];
        let mut mix = NeuralMixture::new(&configs, 0.05);

        let mut early = 0.0;
        let mut late = 0.0;
        for t in 0..200 {
            let lp = mix.step(0);
            if t < 20 {
                early -= lp;
            }
            if t >= 180 {
                late -= lp;
            }
        }

        let early_avg = early / 20.0;
        let late_avg = late / 20.0;
        assert!(
            late_avg < early_avg,
            "late_avg={late_avg} early_avg={early_avg}"
        );
        assert!(late_avg < 0.35, "late_avg={late_avg}");
    }

    #[derive(Clone)]
    struct CountingPredict {
        calls: Arc<AtomicUsize>,
    }

    impl OnlineBytePredictor for CountingPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if symbol == 0 { 0.0 } else { -20.0 }
        }

        fn update(&mut self, _symbol: u8) {}
    }

    #[derive(Clone)]
    struct CountingFillPredict {
        log_calls: Arc<AtomicUsize>,
        fill_calls: Arc<AtomicUsize>,
    }

    impl OnlineBytePredictor for CountingFillPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            self.log_calls.fetch_add(1, Ordering::Relaxed);
            if symbol == 0 { 0.0 } else { -20.0 }
        }

        fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
            self.fill_calls.fetch_add(1, Ordering::Relaxed);
            out.fill(-20.0);
            out[0] = 0.0;
        }

        fn update(&mut self, _symbol: u8) {}
    }

    #[derive(Clone)]
    struct BeginAwarePredict {
        seen_total: Arc<AtomicU64>,
        began: bool,
    }

    impl OnlineBytePredictor for BeginAwarePredict {
        fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
            let total = total_symbols.ok_or_else(|| "missing total symbols".to_string())?;
            self.seen_total.store(total, Ordering::Relaxed);
            self.began = true;
            Ok(())
        }

        fn log_prob(&mut self, _symbol: u8) -> f64 {
            if self.began { 0.0 } else { f64::NEG_INFINITY }
        }

        fn update(&mut self, _symbol: u8) {}
    }

    #[derive(Clone)]
    struct StatePreservingFreshPredict {
        learned: usize,
        began: bool,
    }

    impl OnlineBytePredictor for StatePreservingFreshPredict {
        fn begin_stream(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
            self.began = true;
            Ok(())
        }

        fn reset_frozen(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
            self.began = true;
            Ok(())
        }

        fn log_prob(&mut self, symbol: u8) -> f64 {
            if self.began && self.learned > 0 && symbol == b'K' {
                0.0
            } else {
                -12.0
            }
        }

        fn update(&mut self, symbol: u8) {
            if symbol == b'K' {
                self.learned += 1;
            }
        }
    }

    #[derive(Clone)]
    struct FailingNonResettableFreshPredict {
        learned: usize,
        begin_calls: Arc<AtomicUsize>,
    }

    impl OnlineBytePredictor for FailingNonResettableFreshPredict {
        fn supports_frozen_reset(&self) -> bool {
            false
        }

        fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
            self.begin_calls.fetch_add(1, Ordering::Relaxed);
            total_symbols
                .map(|_| ())
                .ok_or_else(|| "missing total symbols".to_string())
        }

        fn log_prob(&mut self, symbol: u8) -> f64 {
            if self.learned > 0 && symbol == b'Q' {
                0.0
            } else {
                -15.0
            }
        }

        fn update(&mut self, symbol: u8) {
            if symbol == b'Q' {
                self.learned += 1;
            }
        }
    }

    #[derive(Clone)]
    struct ResettingNonResettableFreshPredict {
        learned: usize,
    }

    impl OnlineBytePredictor for ResettingNonResettableFreshPredict {
        fn supports_frozen_reset(&self) -> bool {
            false
        }

        fn begin_stream(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
            self.learned = 0;
            Ok(())
        }

        fn log_prob(&mut self, symbol: u8) -> f64 {
            if self.learned > 0 && symbol == b'R' {
                0.0
            } else {
                -15.0
            }
        }

        fn update(&mut self, symbol: u8) {
            if symbol == b'R' {
                self.learned += 1;
            }
        }
    }

    fn assert_log_prob_update_matches_separate(label: &str, backend: RateBackend) {
        let mut separate = RateBackendPredictor::from_backend(backend.clone(), DEFAULT_MIN_PROB);
        let mut combined = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);
        let data = b"combined step check data";

        for &b in data {
            let logp_separate = separate.log_prob(b);
            separate.update(b);
            let logp_combined = combined.log_prob_update(b);
            let diff = (logp_separate - logp_combined).abs();
            assert!(
                diff <= 1e-12,
                "[{label}] symbol={b} separate={logp_separate} combined={logp_combined} diff={diff}"
            );

            let mut sep_row = [0.0; 256];
            let mut combo_row = [0.0; 256];
            separate.fill_log_probs(&mut sep_row);
            combined.fill_log_probs(&mut combo_row);
            for i in 0..256 {
                let diff = (sep_row[i] - combo_row[i]).abs();
                assert!(
                    diff <= 1e-12,
                    "row mismatch at {i}: {} vs {}",
                    sep_row[i],
                    combo_row[i]
                );
            }
        }
    }

    fn assert_fill_matches_symbol_queries(label: &str, backend: RateBackend) {
        let mut bulk = RateBackendPredictor::from_backend(backend.clone(), DEFAULT_MIN_PROB);
        let mut queried = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);
        let data = b"continuation consistency prompt";

        bulk.begin_stream(Some(data.len() as u64))
            .expect("bulk begin");
        queried
            .begin_stream(Some(data.len() as u64))
            .expect("query begin");
        for &b in data {
            bulk.update(b);
            queried.update(b);
        }

        let mut bulk_row = [0.0; 256];
        bulk.fill_log_probs(&mut bulk_row);
        for (sym, &bulk_logp) in bulk_row.iter().enumerate() {
            let queried_logp = queried.log_prob(sym as u8);
            let diff = (bulk_logp - queried_logp).abs();
            assert!(
                diff <= 1e-12,
                "[{label}] sym={sym} bulk={bulk_logp} queried={queried_logp} diff={diff}"
            );
        }
    }

    fn assert_fill_matches_symbol_queries_after_frozen_conditioning(
        label: &str,
        backend: RateBackend,
    ) {
        let fit = b"If a frog is green, dogs are red.\nIf a toad is green, cats are red.\n";
        let condition = b"If a cat is red, toads are \n";
        let total = (fit.len() + condition.len()) as u64;

        let mut bulk = RateBackendPredictor::from_backend(backend.clone(), DEFAULT_MIN_PROB);
        let mut queried = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);

        bulk.begin_stream(Some(total)).expect("bulk begin");
        queried.begin_stream(Some(total)).expect("query begin");
        for &b in fit {
            bulk.update(b);
            queried.update(b);
        }
        bulk.reset_frozen(Some(condition.len() as u64))
            .expect("bulk reset frozen");
        queried
            .reset_frozen(Some(condition.len() as u64))
            .expect("query reset frozen");
        for &b in condition {
            bulk.update_frozen(b);
            queried.update_frozen(b);
        }

        let mut bulk_row = [0.0; 256];
        bulk.fill_log_probs(&mut bulk_row);
        for (sym, &bulk_logp) in bulk_row.iter().enumerate() {
            let queried_logp = queried.log_prob(sym as u8);
            let diff = (bulk_logp - queried_logp).abs();
            assert!(
                diff <= 1e-12,
                "[{label}] frozen sym={sym} bulk={bulk_logp} queried={queried_logp} diff={diff}"
            );
        }
    }

    #[test]
    fn predictor_log_prob_update_matches_separate_update_for_rosa_backend() {
        assert_log_prob_update_matches_separate("rosa", RateBackend::RosaPlus { max_order: -1 });
    }

    #[test]
    fn predictor_log_prob_update_matches_separate_update_for_ctw_backend() {
        assert_log_prob_update_matches_separate("ctw", RateBackend::Ctw { depth: 6 });
    }

    #[test]
    fn predictor_log_prob_update_matches_separate_update_for_fac_ctw_backend() {
        assert_log_prob_update_matches_separate(
            "fac-ctw",
            RateBackend::FacCtw {
                base_depth: 6,
                num_percept_bits: 8,
                encoding_bits: 8,
                msb_first: None,
            },
        );
    }

    #[test]
    fn predictor_fill_matches_symbol_queries_for_rosa_backend() {
        assert_fill_matches_symbol_queries("rosa", RateBackend::RosaPlus { max_order: -1 });
    }

    #[test]
    fn predictor_fill_matches_symbol_queries_for_ctw_backend() {
        assert_fill_matches_symbol_queries("ctw", RateBackend::Ctw { depth: 6 });
    }

    #[test]
    fn predictor_fill_matches_symbol_queries_for_match_backend() {
        assert_fill_matches_symbol_queries(
            "match",
            RateBackend::Match {
                hash_bits: 18,
                min_len: 4,
                max_len: 64,
                base_mix: 0.02,
                confidence_scale: 1.0,
            },
        );
    }

    #[test]
    fn predictor_fill_matches_symbol_queries_for_ppmd_backend() {
        assert_fill_matches_symbol_queries(
            "ppmd",
            RateBackend::Ppmd {
                order: 8,
                memory_mb: 8,
            },
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn predictor_fill_matches_symbol_queries_for_rwkv_backend() {
        assert_fill_matches_symbol_queries(
            "rwkv7",
            RateBackend::Rwkv7Method {
                method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("rwkv method spec"),
            },
        );
    }

    #[test]
    fn predictor_fill_matches_symbol_queries_for_rosa_backend_after_frozen_conditioning() {
        assert_fill_matches_symbol_queries_after_frozen_conditioning(
            "rosa",
            RateBackend::RosaPlus { max_order: -1 },
        );
    }

    #[test]
    fn predictor_frozen_conditioning_reuses_match_fit_corpus() {
        let mut predictor = RateBackendPredictor::from_backend(
            RateBackend::Match {
                hash_bits: 20,
                min_len: 3,
                max_len: 32,
                base_mix: 0.02,
                confidence_scale: 1.0,
            },
            DEFAULT_MIN_PROB,
        );

        for &b in b"abcabcX" {
            predictor.update(b);
        }
        predictor
            .reset_frozen(Some(6))
            .expect("reset frozen for match backend");
        for &b in b"abcabc" {
            predictor.update_frozen(b);
        }
        let p_x = predictor.log_prob(b'X').exp();
        assert!(
            p_x > 0.01,
            "frozen conditioning should preserve fit corpus for match backend; p_x={p_x}"
        );
    }

    #[test]
    fn predictor_frozen_conditioning_reuses_sparse_match_fit_corpus() {
        let mut predictor = RateBackendPredictor::from_backend(
            RateBackend::SparseMatch {
                hash_bits: 20,
                min_len: 3,
                max_len: 32,
                gap_min: 0,
                gap_max: 2,
                base_mix: 0.02,
                confidence_scale: 1.0,
            },
            DEFAULT_MIN_PROB,
        );

        for &b in b"abcabcX" {
            predictor.update(b);
        }
        predictor
            .reset_frozen(Some(6))
            .expect("reset frozen for sparse-match backend");
        for &b in b"abcabc" {
            predictor.update_frozen(b);
        }
        let p_x = predictor.log_prob(b'X').exp();
        assert!(
            p_x > 0.01,
            "frozen conditioning should preserve fit corpus for sparse-match backend; p_x={p_x}"
        );
    }

    #[test]
    fn neural_predict_then_step_reuses_evaluation_cache() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let cfg0 = {
            let c = c0.clone();
            ExpertConfig::uniform("c0", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let cfg1 = {
            let c = c1.clone();
            ExpertConfig::uniform("c1", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let mut mix = NeuralMixture::new(&[cfg0, cfg1], 0.03);

        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);

        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn neural_predict_multiple_symbols_reuses_single_evaluation() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let cfg0 = {
            let c = c0.clone();
            ExpertConfig::uniform("c0", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let cfg1 = {
            let c = c1.clone();
            ExpertConfig::uniform("c1", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let mut mix = NeuralMixture::new(&[cfg0, cfg1], 0.03);

        let _ = mix.predict_log_prob(0);
        let after_first = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_first, 2);

        let _ = mix.predict_log_prob(1);
        let after_second = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_second, after_first + 2);
    }

    #[test]
    fn neural_fill_then_step_reuses_cached_full_rows() {
        let log0 = Arc::new(AtomicUsize::new(0));
        let log1 = Arc::new(AtomicUsize::new(0));
        let fill0 = Arc::new(AtomicUsize::new(0));
        let fill1 = Arc::new(AtomicUsize::new(0));
        let cfg0 = {
            let log_calls = log0.clone();
            let fill_calls = fill0.clone();
            ExpertConfig::uniform("c0", move || {
                Box::new(CountingFillPredict {
                    log_calls: log_calls.clone(),
                    fill_calls: fill_calls.clone(),
                })
            })
        };
        let cfg1 = {
            let log_calls = log1.clone();
            let fill_calls = fill1.clone();
            ExpertConfig::uniform("c1", move || {
                Box::new(CountingFillPredict {
                    log_calls: log_calls.clone(),
                    fill_calls: fill_calls.clone(),
                })
            })
        };
        let mut mix = NeuralMixture::new(&[cfg0, cfg1], 0.03);

        let mut row = [0.0; 256];
        mix.fill_log_probs(&mut row);
        assert_eq!(fill0.load(Ordering::Relaxed), 1);
        assert_eq!(fill1.load(Ordering::Relaxed), 1);
        assert_eq!(log0.load(Ordering::Relaxed), 0);
        assert_eq!(log1.load(Ordering::Relaxed), 0);

        let _ = mix.step(0);
        assert_eq!(fill0.load(Ordering::Relaxed), 1);
        assert_eq!(fill1.load(Ordering::Relaxed), 1);
        assert_eq!(log0.load(Ordering::Relaxed), 0);
        assert_eq!(log1.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn runtime_begin_stream_propagates_to_experts() {
        let seen_total = Arc::new(AtomicU64::new(0));
        let cfg = {
            let seen_total = seen_total.clone();
            ExpertConfig::uniform("begin-aware", move || {
                Box::new(BeginAwarePredict {
                    seen_total: seen_total.clone(),
                    began: false,
                })
            })
        };

        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![crate::MixtureExpertSpec {
                name: Some("begin-aware".to_string()),
                log_prior: 0.0,
                backend: RateBackend::Ctw { depth: 1 },
            }],
        );
        let mut runtime = build_mixture_runtime(&spec, &[cfg]).expect("runtime");
        runtime.begin_stream(Some(123)).expect("begin stream");
        let _ = runtime.step(0);
        assert_eq!(seen_total.load(Ordering::Relaxed), 123);
    }

    #[test]
    fn runtime_begin_fresh_stream_preserves_resettable_expert_state() {
        let build_calls = Arc::new(AtomicUsize::new(0));
        let cfg = {
            let build_calls = build_calls.clone();
            ExpertConfig::uniform("state-preserving", move || {
                build_calls.fetch_add(1, Ordering::Relaxed);
                Box::new(StatePreservingFreshPredict {
                    learned: 0,
                    began: false,
                })
            })
        };

        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![crate::MixtureExpertSpec {
                name: Some("state-preserving".to_string()),
                log_prior: 0.0,
                backend: RateBackend::Ctw { depth: 1 },
            }],
        );
        let mut runtime = build_mixture_runtime(&spec, &[cfg]).expect("runtime");
        runtime.begin_stream(Some(1)).expect("begin stream");
        let _ = runtime.step(b'K');

        runtime
            .begin_fresh_stream(Some(1))
            .expect("fresh stream restart");

        assert_eq!(
            build_calls.load(Ordering::Relaxed),
            1,
            "resettable experts must not be rebuilt for fresh stream restarts"
        );
        let logp = runtime.peek_log_prob(b'K');
        assert!(
            logp > -1.0,
            "fresh stream restart should preserve fitted expert state; logp={logp}"
        );
    }

    #[test]
    fn runtime_begin_fresh_stream_failure_preserves_existing_expert() {
        let build_calls = Arc::new(AtomicUsize::new(0));
        let begin_calls = Arc::new(AtomicUsize::new(0));
        let cfg = {
            let build_calls = build_calls.clone();
            let begin_calls = begin_calls.clone();
            ExpertConfig::uniform("non-resettable", move || {
                build_calls.fetch_add(1, Ordering::Relaxed);
                Box::new(FailingNonResettableFreshPredict {
                    learned: 0,
                    begin_calls: begin_calls.clone(),
                })
            })
        };

        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![crate::MixtureExpertSpec {
                name: Some("non-resettable".to_string()),
                log_prior: 0.0,
                backend: RateBackend::Ctw { depth: 1 },
            }],
        );
        let mut runtime = build_mixture_runtime(&spec, &[cfg]).expect("runtime");
        runtime.begin_stream(Some(1)).expect("begin stream");
        let _ = runtime.step(b'Q');

        let err = runtime
            .begin_fresh_stream(None)
            .expect_err("fresh stream restart should report the expert begin_stream failure");

        assert!(err.contains("missing total symbols"));
        assert_eq!(
            build_calls.load(Ordering::Relaxed),
            1,
            "failed fresh stream restarts must not rebuild or discard the old expert"
        );
        assert_eq!(begin_calls.load(Ordering::Relaxed), 2);
        let logp = runtime.peek_log_prob(b'Q');
        assert!(
            logp > -1.0,
            "old expert should remain usable after failed fresh restart; logp={logp}"
        );
    }

    #[test]
    fn runtime_begin_fresh_stream_failure_is_transactional_across_experts() {
        let failing_begin_calls = Arc::new(AtomicUsize::new(0));
        let resettable_cfg = ExpertConfig::uniform("resetting", || {
            Box::new(ResettingNonResettableFreshPredict { learned: 0 })
        });
        let failing_cfg = {
            let failing_begin_calls = failing_begin_calls.clone();
            ExpertConfig::uniform("failing", move || {
                Box::new(FailingNonResettableFreshPredict {
                    learned: 0,
                    begin_calls: failing_begin_calls.clone(),
                })
            })
        };

        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("resetting".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Ctw { depth: 1 },
                },
                crate::MixtureExpertSpec {
                    name: Some("failing".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Ctw { depth: 1 },
                },
            ],
        );
        let mut runtime =
            build_mixture_runtime(&spec, &[resettable_cfg, failing_cfg]).expect("runtime");
        runtime.begin_stream(Some(1)).expect("begin stream");
        let _ = runtime.step(b'R');
        let logp_before = runtime.peek_log_prob(b'R');
        assert!(
            logp_before > -1.0,
            "resetting expert should have learned prior to restart; logp={logp_before}"
        );

        let err = runtime
            .begin_fresh_stream(None)
            .expect_err("second expert should fail begin_fresh_stream");
        assert!(err.contains("missing total symbols"));
        assert_eq!(
            failing_begin_calls.load(Ordering::Relaxed),
            2,
            "failing expert begin should run once at initial begin and once at failed restart"
        );

        let logp_after = runtime.peek_log_prob(b'R');
        assert!(
            logp_after > -1.0,
            "failed fresh restart must restore earlier experts instead of leaving partial mutation; logp={logp_after}"
        );
    }

    fn ctw_checkpoint_depth(predictor: &RateBackendPredictor) -> usize {
        match predictor {
            #[cfg(feature = "backend-ctw")]
            RateBackendPredictor::Ctw {
                checkpoint_depth, ..
            } => *checkpoint_depth,
            _ => panic!("expected ctw predictor"),
        }
    }

    fn predictor_log_probs(predictor: &mut RateBackendPredictor) -> [f64; 256] {
        let mut row = [0.0f64; 256];
        predictor.fill_log_probs(&mut row);
        row
    }

    fn assert_log_prob_rows_close(actual: &[f64; 256], expected: &[f64; 256], label: &str) {
        for (symbol, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "{label}[{symbol}]: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn lifecycle_reset_rejects_active_ctw_prediction_checkpoint() {
        let mut predictor =
            RateBackendPredictor::from_backend(RateBackend::Ctw { depth: 4 }, DEFAULT_MIN_PROB);
        for &symbol in b"abracadabra ctw checkpoint guard" {
            predictor.update(symbol);
        }
        let mut baseline = predictor.clone();
        let checkpoint = predictor.checkpoint();

        let err = predictor
            .begin_fresh_stream(Some(0))
            .expect_err("ctw lifecycle reset must reject active compact prediction checkpoints");

        assert!(err.contains("prediction checkpoints"));
        assert_eq!(ctw_checkpoint_depth(&predictor), 1);
        let expected = predictor_log_probs(&mut baseline);
        let actual = predictor_log_probs(&mut predictor);
        assert_log_prob_rows_close(&actual, &expected, "ctw active-checkpoint lifecycle reject");
        predictor.discard_checkpoint(checkpoint);
    }

    #[test]
    fn ctw_lifecycle_with_active_prefix_uses_full_clone() {
        let mut predictor =
            RateBackendPredictor::from_backend(RateBackend::Ctw { depth: 4 }, DEFAULT_MIN_PROB);
        assert!(
            predictor
                .begin_native_msb_byte_prefix()
                .expect("begin native prefix")
        );
        predictor
            .observe_native_msb_prefix_bit(0, true)
            .expect("observe one prefix bit");

        let checkpoint =
            predictor.lifecycle_checkpoint(OnlineBytePredictorLifecycleOp::FinishStream);

        assert!(
            matches!(checkpoint, RateBackendPredictorLifecycleCheckpoint::Full(_)),
            "active native prefix with observed bits must not use compact lifecycle rollback"
        );
    }

    #[test]
    fn nested_mixture_lifecycle_failure_restores_wrapper_state() {
        let inner_cfgs = vec![
            weighted_cfg("zero-heavy", 1.0, 0.90),
            weighted_cfg("zero-light", 1.0, 0.10),
        ];
        let mut inner = MixtureRuntime::Bayes(BayesMixture::new(&inner_cfgs));
        inner.begin_stream(Some(1)).expect("inner begin");
        let _ = inner.step(0);
        let before = inner.peek_log_prob(0);

        let failing_begin_calls = Arc::new(AtomicUsize::new(0));
        let mut experts = vec![
            ExpertState {
                name: "nested".to_string(),
                log_weight: 0.0,
                log_prior: 0.0,
                predictor: ExpertPredictor::rate_backend(RateBackendPredictor::Mixture {
                    runtime: inner,
                }),
                cum_log_loss: 0.0,
            },
            ExpertState {
                name: "failing".to_string(),
                log_weight: 0.0,
                log_prior: 0.0,
                predictor: ExpertPredictor::generic(Box::new(FailingNonResettableFreshPredict {
                    learned: 0,
                    begin_calls: failing_begin_calls.clone(),
                })),
                cum_log_loss: 0.0,
            },
        ];

        let err = begin_expert_fresh_stream(&mut experts, None)
            .expect_err("outer lifecycle should fail on second expert");

        assert!(err.contains("missing total symbols"));
        let after = experts[0].log_prob(0);
        assert!(
            (after - before).abs() < 1e-12,
            "nested mixture wrapper state should be restored: before={before}, after={after}"
        );
    }

    #[test]
    fn zpaq_fill_log_probs_does_not_drift_history() {
        let backend = RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("1"),
        };
        let mut baseline = RateBackendPredictor::from_backend(backend.clone(), DEFAULT_MIN_PROB);
        let mut probe = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);

        let history = b"history for zpaq predictor";
        for &b in history {
            baseline.update(b);
            probe.update(b);
        }

        let mut row = [0.0f64; 256];
        probe.fill_log_probs(&mut row);

        let sym = b'k';
        let lp_base = baseline.log_prob(sym);
        let lp_probe = probe.log_prob(sym);
        assert!((lp_base - lp_probe).abs() < 1e-9);
        assert!((row[sym as usize] - lp_base).abs() < 1e-9);

        baseline.update(sym);
        probe.update(sym);
        let next = b'q';
        let next_base = baseline.log_prob(next);
        let next_probe = probe.log_prob(next);
        assert!((next_base - next_probe).abs() < 1e-9);
    }

    fn assert_checkpoint_roundtrip_restores_predictor(backend: RateBackend, history: &[u8]) {
        let mut predictor = RateBackendPredictor::from_backend(backend.clone(), DEFAULT_MIN_PROB);
        predictor.begin_stream(None).expect("begin stream");
        for &byte in history {
            predictor.update(byte);
        }

        let checkpoint = predictor.checkpoint();
        let mut baseline = predictor.clone();

        for &byte in b"speculative branch" {
            predictor.update(byte);
        }
        for &byte in b"frozen branch" {
            predictor.update_frozen(byte);
        }
        for &byte in b"tail" {
            predictor.update(byte);
        }

        predictor.restore_checkpoint(&checkpoint);
        predictor.clear_checkpoints_if_supported();

        let mut baseline_row = [0.0; 256];
        let mut restored_row = [0.0; 256];
        baseline.fill_log_probs(&mut baseline_row);
        predictor.fill_log_probs(&mut restored_row);
        for (expected, actual) in baseline_row.iter().zip(restored_row.iter()) {
            assert!(
                (expected - actual).abs() < 1e-12,
                "checkpoint restore drifted predictor state: expected={expected}, actual={actual}"
            );
        }
    }

    #[test]
    fn rosa_checkpoint_restores_exact_predictor_state() {
        assert_checkpoint_roundtrip_restores_predictor(
            RateBackend::RosaPlus { max_order: -1 },
            b"rosa checkpoint base history",
        );
    }

    #[test]
    fn rosa_checkpoint_uses_compact_journal_marker() {
        let mut predictor = RateBackendPredictor::from_backend(
            RateBackend::RosaPlus { max_order: -1 },
            DEFAULT_MIN_PROB,
        );
        match predictor.checkpoint() {
            RateBackendPredictorCheckpoint::Rosa { journal_len } => {
                assert_eq!(journal_len, 0);
            }
            _ => panic!("expected compact rosa checkpoint"),
        }
        predictor.clear_checkpoints_if_supported();
    }

    #[test]
    fn ctw_checkpoint_restores_mixed_learned_and_frozen_updates() {
        assert_checkpoint_roundtrip_restores_predictor(
            RateBackend::Ctw { depth: 8 },
            b"ctw checkpoint base history",
        );
    }

    #[test]
    fn fac_ctw_checkpoint_restores_mixed_learned_and_frozen_updates() {
        assert_checkpoint_roundtrip_restores_predictor(
            RateBackend::FacCtw {
                base_depth: 7,
                num_percept_bits: 6,
                encoding_bits: 6,
                msb_first: None,
            },
            b"fac-ctw checkpoint base history",
        );
    }

    #[test]
    fn ppmd_checkpoint_restores_mixed_learned_and_frozen_updates() {
        assert_checkpoint_roundtrip_restores_predictor(
            RateBackend::Ppmd {
                order: 8,
                memory_mb: 8,
            },
            b"ppmd checkpoint base history",
        );
    }

    #[test]
    fn calibrated_checkpoint_restores_wrapped_predictor_and_calibration_state() {
        assert_checkpoint_roundtrip_restores_predictor(
            RateBackend::Calibrated {
                spec: Arc::new(CalibratedSpec {
                    base: RateBackend::Ctw { depth: 6 },
                    context: crate::CalibrationContextKind::Text,
                    bins: 33,
                    learning_rate: 0.02,
                    bias_clip: 4.0,
                }),
            },
            b"calibrated checkpoint base history",
        );
    }

    fn assert_predictor_log_probs_normalize_to_one(backend: RateBackend) {
        let mut predictor = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);
        for &b in b"normalization corpus for ctw/fac predictor checks" {
            predictor.update(b);
        }
        let mut sum = 0.0f64;
        for sym in 0u8..=255u8 {
            sum += predictor.log_prob(sym).exp();
        }
        assert!(
            (sum - 1.0).abs() <= 1e-10,
            "probability mass drift: sum={sum}"
        );
    }

    #[test]
    fn ctw_predictor_symbol_probs_normalize() {
        assert_predictor_log_probs_normalize_to_one(RateBackend::Ctw { depth: 7 });
    }

    #[test]
    fn fac_ctw_predictor_symbol_probs_normalize() {
        assert_predictor_log_probs_normalize_to_one(RateBackend::FacCtw {
            base_depth: 7,
            num_percept_bits: 8,
            encoding_bits: 8,
            msb_first: None,
        });
    }

    #[test]
    fn expert_config_helpers_expose_names_priors_and_predictor_builders() {
        let cfg = ExpertConfig::from_rate_backend(
            Some("ctw-four".to_string()),
            -0.75,
            RateBackend::Ctw { depth: 4 },
        );
        assert_eq!(cfg.name(), "ctw-four");
        assert!((cfg.log_prior() + 0.75).abs() < 1e-12);
        let mut predictor = cfg.build_predictor();
        let logp = predictor.log_prob(b'a');
        assert!(logp.is_finite());

        let uniform = ExpertConfig::uniform("always-zero", || Box::new(AlwaysPredict { byte: 0 }));
        assert_eq!(uniform.name(), "always-zero");
        assert_eq!(uniform.log_prior(), 0.0);

        let zpaq = ExpertConfig::zpaq("zpaq-one", "1");
        assert_eq!(zpaq.name(), "zpaq-one");
        assert_eq!(zpaq.log_prior(), 0.0);
        let mut zpaq_predictor = zpaq.build_predictor();
        assert!(zpaq_predictor.log_prob(b'b').is_finite());
    }

    fn assert_runtime_variant_contracts(spec: MixtureSpec, expected_names: &[&str], symbol: u8) {
        let configs = vec![
            ExpertConfig::new(expected_names[0].to_string(), 0.75f64.ln(), move || {
                Box::new(FixedProbPredict { prob_zero: 0.8 })
            }),
            ExpertConfig::new(expected_names[1].to_string(), 0.25f64.ln(), move || {
                Box::new(FixedProbPredict { prob_zero: 0.35 })
            }),
        ];
        let mut runtime = build_mixture_runtime(&spec, &configs).expect("runtime should build");

        runtime.begin_stream(Some(8)).expect("begin stream");
        let peek = runtime.peek_log_prob(symbol);
        assert!(peek.is_finite());

        let mut row = [f64::NEG_INFINITY; 256];
        runtime.fill_log_probs(&mut row);
        let mass: f64 = row.iter().map(|lp| lp.exp()).sum();
        assert!(
            (mass - 1.0).abs() < 1e-8,
            "mixture runtime PDF must normalize; mass={mass}"
        );

        let stepped = runtime.step(symbol);
        assert!(stepped.is_finite());
        runtime.update_frozen(symbol.wrapping_add(1));
        runtime.finish_stream().expect("finish stream");
        runtime.reset_frozen(Some(3)).expect("reset frozen");
        runtime
            .begin_stream(Some(3))
            .expect("begin stream after reset");

        match &mut runtime {
            MixtureRuntime::Bayes(m) => {
                assert_eq!(m.expert_names(), expected_names);
                assert_eq!(m.expert_log_losses().len(), 2);
                assert_eq!(m.total_log_loss(), 0.0);
                let (_, posterior) = m.max_posterior();
                assert!((0.0..=1.0).contains(&posterior));
            }
            MixtureRuntime::Fading(m) => {
                assert_eq!(m.expert_names(), expected_names);
                assert_eq!(m.total_log_loss(), 0.0);
                let posterior_mass: f64 = m.posterior().into_iter().sum();
                assert!((posterior_mass - 1.0).abs() < 1e-10);
            }
            MixtureRuntime::Switching(m) => {
                assert_eq!(m.expert_names(), expected_names);
                assert_eq!(m.expert_log_losses().len(), 2);
                assert_eq!(m.total_log_loss(), 0.0);
                let (_, posterior) = m.max_posterior();
                assert!((0.0..=1.0).contains(&posterior));
            }
            MixtureRuntime::Convex(m) => {
                let weight_sum: f64 = m.lambda.iter().sum();
                assert!((weight_sum - 1.0).abs() < 1e-10);
                assert_eq!(m.total_log_loss, 0.0);
            }
            MixtureRuntime::Mdl(m) => {
                assert_eq!(m.expert_names(), expected_names);
                assert_eq!(m.expert_log_losses().len(), 2);
                assert_eq!(m.total_log_loss(), 0.0);
                assert!(m.best_index() < 2);
            }
            MixtureRuntime::Neural(m) => {
                assert_eq!(m.total_log_loss(), 0.0);
            }
        }
    }

    #[test]
    fn runtime_variants_support_stream_fill_and_reset_contracts() {
        assert_runtime_variant_contracts(
            MixtureSpec::new(
                MixtureKind::Bayes,
                vec![
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 }).with_name("left"),
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 5 }).with_name("right"),
                ],
            ),
            &["left", "right"],
            0,
        );

        assert_runtime_variant_contracts(
            MixtureSpec::new(
                MixtureKind::FadingBayes,
                vec![
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                        .with_name("fade-a"),
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 5 })
                        .with_name("fade-b"),
                ],
            )
            .with_decay(0.93),
            &["fade-a", "fade-b"],
            0,
        );

        assert_runtime_variant_contracts(
            MixtureSpec::new(
                MixtureKind::Switching,
                vec![
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                        .with_name("switch-a"),
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 5 })
                        .with_name("switch-b"),
                ],
            )
            .with_schedule(MixtureScheduleMode::Theorem),
            &["switch-a", "switch-b"],
            1,
        );

        assert_runtime_variant_contracts(
            MixtureSpec::new(
                MixtureKind::Convex,
                vec![
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                        .with_name("convex-a"),
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 5 })
                        .with_name("convex-b"),
                ],
            )
            .with_schedule(MixtureScheduleMode::Theorem)
            .with_alpha(1.25),
            &["convex-a", "convex-b"],
            0,
        );

        assert_runtime_variant_contracts(
            MixtureSpec::new(
                MixtureKind::Mdl,
                vec![
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 }).with_name("mdl-a"),
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 5 }).with_name("mdl-b"),
                ],
            ),
            &["mdl-a", "mdl-b"],
            0,
        );

        assert_runtime_variant_contracts(
            MixtureSpec::new(
                MixtureKind::Neural,
                vec![
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                        .with_name("neural-a"),
                    crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 5 })
                        .with_name("neural-b"),
                ],
            )
            .with_alpha(0.04),
            &["neural-a", "neural-b"],
            0,
        );
    }

    #[cfg(feature = "backend-mixture")]
    #[test]
    fn compiled_mixture_helpers_roundtrip_expert_configs_and_runtime() {
        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![
                crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 3 })
                    .with_name("compiled-ctw")
                    .with_log_prior(-0.5),
                crate::MixtureExpertSpec::new(RateBackend::RosaPlus { max_order: 7 })
                    .with_name("compiled-rosa")
                    .with_log_prior(-1.25),
            ],
        );
        let compiled = RateBackend::Mixture {
            spec: Arc::new(spec.clone()),
        }
        .compile()
        .expect("mixture backend should compile");

        let configs =
            expert_configs_from_compiled_mixture(&compiled).expect("compiled mixture configs");
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].name(), "compiled-ctw");
        assert_eq!(configs[1].name(), "compiled-rosa");
        assert!((configs[0].log_prior() + 0.5).abs() < 1e-12);
        assert!((configs[1].log_prior() + 1.25).abs() < 1e-12);
        assert!(
            configs
                .iter()
                .all(|cfg| cfg.build_predictor().log_prob(b'x').is_finite())
        );

        let mut runtime = build_mixture_runtime_from_compiled(&compiled, &configs)
            .expect("compiled mixture runtime should build");
        runtime.begin_stream(Some(4)).expect("begin stream");
        assert!(runtime.peek_log_prob(0).is_finite());
        let mut row = [f64::NEG_INFINITY; 256];
        runtime.fill_log_probs(&mut row);
        let mass: f64 = row.iter().map(|lp| lp.exp()).sum();
        assert!((mass - 1.0).abs() < 1e-8);
    }

    #[test]
    fn binary_token_mixture_builder_preserves_ctw_family_identity() {
        let ctw_backend = RateBackend::Ctw { depth: 4 }
            .compile()
            .expect("compiled ctw backend");
        let mixture_backend = RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![crate::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })],
            )),
        }
        .compile()
        .expect("compiled mixture backend");

        let mut direct = crate::runtime::build_rate_backend_binary_token_predictor(
            &ctw_backend,
            DEFAULT_MIN_PROB,
        )
        .expect("direct ctw bit predictor");
        let mut mixture = crate::runtime::build_rate_backend_binary_token_predictor(
            &mixture_backend,
            DEFAULT_MIN_PROB,
        )
        .expect("mixture bit predictor");

        direct.begin_stream(Some(9)).expect("begin direct stream");
        mixture.begin_stream(Some(9)).expect("begin mixture stream");

        for bit in [true, false, true, true, false, false, true, false, true] {
            let direct_p0 = direct.log_prob(0);
            let direct_p1 = direct.log_prob(1);
            let mixture_p0 = mixture.log_prob(0);
            let mixture_p1 = mixture.log_prob(1);
            assert!((direct_p0 - mixture_p0).abs() < 1e-12);
            assert!((direct_p1 - mixture_p1).abs() < 1e-12);

            direct.update(u8::from(bit));
            mixture.update(u8::from(bit));
        }
    }

    #[test]
    fn rate_backend_predictor_checkpoint_enum_stays_compact() {
        let checkpoint_size: usize = std::mem::size_of::<RateBackendPredictorCheckpoint>();
        assert!(
            checkpoint_size < 128,
            "RateBackendPredictorCheckpoint must stay pointer-sized after boxing Full; got {checkpoint_size}B"
        );
    }
}
