//! Predictive models for AIXI.
//!
//! This module defines the `Predictor` trait, which encapsulates the logic
//! for learning from history and predicting future symbols. Different implementations
//! provide different complexity vs performance trade-offs.

use crate::api::{CompiledRateBackend, RateBackend};
use crate::prediction::{binary_prediction_from_log_probs, binary_prediction_from_probs};
#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{ContextTree, FacContextTree};
#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::{RosaPlus, RosaTx};
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
#[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
use crate::error::{InfotheoryError, InfotheoryResult};
#[cfg(feature = "backend-mamba")]
use crate::mambazip::{Compressor as MambaCompressor, Model as MambaModel, State as MambaState};
use crate::mixture::{
    DEFAULT_MIN_PROB, OnlineBytePredictor, RateBackendPredictor, RateBackendPredictorCheckpoint,
};
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip::{Compressor as RwkvCompressor, Model as RwkvModel, State as RwkvState};
use crate::spec::SpecError;
use std::error::Error;
use std::fmt;
#[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
use std::sync::Arc;

/// Interface for an AIXI world model.
///
/// Predictors are mutated behind `&mut self` and cloned per worker during
/// parallel MCTS. They only need `Send`, not `Sync`, which avoids unsound
/// thread-sharing requirements for backends with thread-confined internals.
pub trait Predictor: Send {
    /// Incorporates a new symbol into the model's training history.
    fn update(&mut self, sym: bool);

    /// Incorporates a new symbol as committed training history without
    /// retaining rollback state when the predictor supports that optimization.
    fn commit_update(&mut self, sym: bool) {
        self.update(sym);
    }

    /// Appends a symbol to the model's interaction history without necessarily
    /// updating the training counts immediately (backend dependent).
    fn update_history(&mut self, sym: bool) {
        self.update(sym);
    }

    /// Appends a symbol as committed interaction history without retaining
    /// rollback state when the predictor supports that optimization.
    fn commit_update_history(&mut self, sym: bool) {
        self.update_history(sym);
    }

    /// Reverts the model to its state before the last `update`.
    fn revert(&mut self);

    /// Reverts the model to its state before the last `update_history`.
    fn pop_history(&mut self) {
        self.revert();
    }

    /// Begins an optional coarse rollback scope for simulation-heavy callers.
    ///
    /// Predictors that support this can avoid retaining per-symbol rollback state
    /// until the matching `rollback_scope` call.
    fn begin_rollback_scope(&mut self) {}

    /// Rolls back to the last scope opened with `begin_rollback_scope`.
    ///
    /// Returns `true` when a scope rollback was performed, allowing callers to skip
    /// per-symbol revert loops.
    fn rollback_scope(&mut self) -> bool {
        false
    }

    /// Predicts the probability of the next symbol being `sym`.
    fn predict_prob(&mut self, sym: bool) -> f64;

    /// Shorthand for `predict_prob(true)`.
    fn predict_one(&mut self) -> f64 {
        self.predict_prob(true)
    }

    /// Returns a human-readable name of the predictive model.
    fn model_name(&self) -> String;

    /// Creates a boxed clone of this predictor.
    fn boxed_clone(&self) -> Box<dyn Predictor>;

    /// Clear transient conditioning history while preserving all committed learning state.
    ///
    /// Called between independent teacher traces during warm-start to prevent the terminal
    /// conditioning context of one trace from influencing predictions at the start of the next.
    ///
    /// # Contract
    ///
    /// After a successful call, the predictor must behave as if it were freshly initialized
    /// with respect to context-dependent predictions (e.g. the sliding-window context suffix
    /// used to navigate a CTW tree is reset to empty). It must retain **all** committed
    /// learned model state induced by prior updates, while clearing only transient
    /// conditioning context. Implementations that clear learned counts (e.g. by calling
    /// a full `clear()`) violate this contract.
    ///
    /// # Errors
    ///
    /// Return `Err` only when the backend has no meaningful way to isolate conditioning
    /// state from learned parameters (e.g. a fully stateful streaming model where the
    /// two are inseparable). The default no-op is appropriate for backends whose context
    /// is already isolated or resets naturally.
    fn reset_conditioning_history(&mut self) -> Result<(), String> {
        Ok(())
    }
}


/// A predictor using the Action-Conditional CTW algorithm.
///
/// AC-CTW uses a single context tree for all bits in sequence.
/// For better type information exploitation, use `FacCtwPredictor`.
#[cfg(feature = "backend-ctw")]
pub struct CtwPredictor {
    tree: ContextTree,
}

#[cfg(feature = "backend-ctw")]
impl CtwPredictor {
    /// Creates a new `CtwPredictor` with the specified context depth.
    pub fn new(depth: usize) -> Self {
        Self {
            tree: ContextTree::new(depth),
        }
    }
}

#[cfg(feature = "backend-ctw")]
impl Predictor for CtwPredictor {
    fn update(&mut self, sym: bool) {
        self.tree.update(sym);
    }
    fn update_history(&mut self, sym: bool) {
        self.tree.update_history(&[sym]);
    }

    fn revert(&mut self) {
        self.tree.revert();
    }
    fn pop_history(&mut self) {
        self.tree.revert_history();
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        self.tree.predict(sym)
    }

    fn model_name(&self) -> String {
        format!("AC-CTW(d={})", self.tree.depth())
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        Box::new(Self {
            tree: self.tree.clone(),
        })
    }

    fn reset_conditioning_history(&mut self) -> Result<(), String> {
        self.tree.truncate_history(0);
        Ok(())
    }
}

/// A predictor using the Factorized Action-Conditional CTW (FAC-CTW) algorithm.
///
/// FAC-CTW uses k separate context trees (one per percept bit) with overlapping
/// context depths D+i-1 for bit i. This enables better exploitation of type
/// information within percepts, as described in Veness et al. (2011) Section 5.
///
/// This is the recommended CTW variant for MC-AIXI agents.
#[cfg(feature = "backend-ctw")]
pub struct FacCtwPredictor {
    tree: FacContextTree,
    /// Current bit index within a percept (cycles 0..num_bits).
    current_bit: usize,
    /// Total number of percept bits (k).
    num_bits: usize,
}

#[cfg(feature = "backend-ctw")]
impl FacCtwPredictor {
    /// Creates a new `FacCtwPredictor`.
    ///
    /// - `base_depth`: Context depth D for the first bit's tree
    /// - `num_percept_bits`: Total bits per percept (observation_bits + reward_bits)
    pub fn new(base_depth: usize, num_percept_bits: usize) -> Self {
        Self {
            tree: FacContextTree::new(base_depth, num_percept_bits),
            current_bit: 0,
            num_bits: num_percept_bits,
        }
    }
}

#[cfg(feature = "backend-ctw")]
impl Predictor for FacCtwPredictor {
    fn update(&mut self, sym: bool) {
        self.tree.update(sym, self.current_bit);
        self.current_bit = (self.current_bit + 1) % self.num_bits;
    }

    fn update_history(&mut self, sym: bool) {
        self.tree.update_history(&[sym]);
    }

    fn revert(&mut self) {
        // Revert to previous bit index
        self.current_bit = if self.current_bit == 0 {
            self.num_bits - 1
        } else {
            self.current_bit - 1
        };
        self.tree.revert(self.current_bit);
    }

    fn pop_history(&mut self) {
        self.tree.revert_history(1);
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        self.tree.predict(sym, self.current_bit)
    }

    fn model_name(&self) -> String {
        format!("FAC-CTW(D={}, k={})", self.tree.base_depth(), self.num_bits)
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        Box::new(Self {
            tree: self.tree.clone(),
            current_bit: self.current_bit,
            num_bits: self.num_bits,
        })
    }

    fn reset_conditioning_history(&mut self) -> Result<(), String> {
        self.tree.reset_history_only();
        self.current_bit = 0;
        Ok(())
    }
}

/// A predictor using the ROSA-Plus (Rapid Online Suffix Automaton + Witten-Bell Smoother) algorithm.
///
/// ROSA is a (practically) sub-quadratic suffix automaton based language model that
/// can handle very long contexts efficiently.
#[cfg(feature = "backend-rosa")]
pub struct RosaPredictor {
    model: RosaPlus,
    history: Vec<RosaTx>,
}

#[cfg(feature = "backend-rosa")]
impl RosaPredictor {
    /// Creates a new `RosaPredictor` with a maximum context length for the fallback LM.
    /// Note: deterministic ROSA uses the full SAM and is not capped by `max_order`.
    pub fn new(max_order: i64) -> Self {
        // Seed and EOT
        let mut model = RosaPlus::new(max_order, false, 0, 42);
        // Important: Pre-build the LM with full byte alphabet so we can incrementally update it.
        model.build_lm_full_bytes_no_finalize_endpos();
        Self {
            model,
            history: Vec::new(),
        }
    }
}

#[cfg(feature = "backend-rosa")]
impl Predictor for RosaPredictor {
    fn update(&mut self, sym: bool) {
        let mut tx = self.model.begin_tx();
        // Using 0u8 and 1u8 for bits
        let byte = if sym { 1u8 } else { 0u8 };

        // train_example_tx updates the tx object and the model
        self.model.train_sequence_tx(&mut tx, &[byte]);
        self.history.push(tx);
    }

    fn revert(&mut self) {
        if let Some(tx) = self.history.pop() {
            self.model.rollback_tx(tx);
        }
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        binary_prediction_from_probs(
            self.model.prob_for_last(0),
            self.model.prob_for_last(1),
            DEFAULT_MIN_PROB,
        )
        .prob(sym)
    }

    fn model_name(&self) -> String {
        "ROSA".to_string()
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        Box::new(Self {
            model: self.model.clone(),
            history: self.history.clone(),
        })
    }

    fn reset_conditioning_history(&mut self) -> Result<(), String> {
        // Preserve trained SAM/LM parameters while dropping transient cursor and
        // rollback journal state so independent traces start from empty context.
        self.model.reset_conditioning_cursor();
        self.history.clear();
        Ok(())
    }
}

/// A predictor using ZPAQ as a streaming rate model.
///
/// This maintains a full history so it can rebuild state on revert and handle
/// any misuse where `predict_prob` is called without a matching `update`.
#[cfg(feature = "backend-zpaq")]
pub struct ZpaqPredictor {
    method: String,
    min_prob: f64,
    model: ZpaqRateModel,
    history: Vec<u8>,
    pending: Option<(u8, f64)>,
}

#[cfg(feature = "backend-zpaq")]
impl ZpaqPredictor {
    /// Create a ZPAQ-backed predictor from a `method` and probability floor.
    pub fn new(method: String, min_prob: f64) -> Self {
        let model = ZpaqRateModel::new(method.clone(), min_prob);
        Self {
            method,
            min_prob,
            model,
            history: Vec::new(),
            pending: None,
        }
    }

    fn rebuild_from_history(&mut self) {
        self.model.reset();
        if !self.history.is_empty() {
            self.model.update_and_score(&self.history);
        }
    }

    fn log_prob_from_history(&self, symbol: u8) -> f64 {
        let mut tmp = ZpaqRateModel::new(self.method.clone(), self.min_prob);
        if !self.history.is_empty() {
            tmp.update_and_score(&self.history);
        }
        tmp.log_prob(symbol)
    }

    fn binary_log_prob_pair(&mut self, preferred_symbol: u8) -> (f64, f64) {
        let other_symbol = preferred_symbol ^ 1;
        let preferred_logp = match self.pending {
            Some((pending, logp)) if pending == preferred_symbol => logp,
            Some(_) => self.log_prob_from_history(preferred_symbol),
            None => {
                let logp = self.model.log_prob(preferred_symbol);
                self.pending = Some((preferred_symbol, logp));
                logp
            }
        };
        let other_logp = match self.pending {
            Some((pending, logp)) if pending == other_symbol => logp,
            _ => self.log_prob_from_history(other_symbol),
        };
        if preferred_symbol == 0 {
            (preferred_logp, other_logp)
        } else {
            (other_logp, preferred_logp)
        }
    }
}

#[cfg(feature = "backend-zpaq")]
impl Predictor for ZpaqPredictor {
    fn update(&mut self, sym: bool) {
        let byte = if sym { 1u8 } else { 0u8 };
        if let Some((pending, _)) = self.pending {
            if pending == byte {
                self.model.update(byte);
                self.pending = None;
                self.history.push(byte);
                return;
            }
            self.pending = None;
            self.rebuild_from_history();
        }
        self.model.update(byte);
        self.history.push(byte);
    }

    fn revert(&mut self) {
        if self.history.pop().is_some() {
            self.pending = None;
            self.rebuild_from_history();
        }
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        let preferred_symbol = if sym { 1u8 } else { 0u8 };
        let (logp0, logp1) = self.binary_log_prob_pair(preferred_symbol);
        binary_prediction_from_log_probs(logp0, logp1, self.min_prob).prob(sym)
    }

    fn model_name(&self) -> String {
        format!("ZPAQ({})", self.method)
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        Box::new(Self {
            method: self.method.clone(),
            min_prob: self.min_prob,
            model: self.model.clone(),
            history: self.history.clone(),
            pending: self.pending,
        })
    }
}

/// A generic bit-level predictor backed by any [`RateBackend`].
///
/// This adapter maps boolean symbols to bytes `{0,1}` and forwards them to the
/// workspace-wide rate backend abstraction. It prioritizes correctness and
/// backend coverage over rollback efficiency.
pub struct RateBackendBitPredictor {
    backend: CompiledRateBackend,
    min_prob: f64,
    predictor: RateBackendPredictor,
    journal: Vec<RateBackendJournalEntry>,
    rollback_scopes: Vec<RateBackendRollbackScope>,
}

/// Error returned while constructing or initializing a rate-backend bit predictor.
#[derive(Debug)]
#[non_exhaustive]
pub enum RateBackendBitPredictorError {
    /// Backend compilation failed.
    Compile(SpecError),
    /// ZPAQ-backed predictors cannot satisfy reversible bit-predictor semantics.
    UnsupportedZpaq,
    /// Runtime predictor construction failed.
    Runtime(String),
    /// Predictor stream initialization failed.
    StreamStart(String),
}

/// Error returned while constructing a predictor from a compiled rate backend.
#[derive(Debug)]
#[non_exhaustive]
pub enum PredictorBuildError {
    /// Bit-token adaptation failed while preparing the backend for binary symbols.
    BitAdaptation(SpecError),
    /// Generic bit-level predictor construction failed.
    BitPredictor(RateBackendBitPredictorError),
}

impl fmt::Display for PredictorBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitAdaptation(err) => write!(f, "{err}"),
            Self::BitPredictor(err) => write!(f, "{err}"),
        }
    }
}

impl Error for PredictorBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::BitAdaptation(err) => Some(err),
            Self::BitPredictor(err) => Some(err),
        }
    }
}

impl From<RateBackendBitPredictorError> for PredictorBuildError {
    fn from(value: RateBackendBitPredictorError) -> Self {
        Self::BitPredictor(value)
    }
}

impl fmt::Display for RateBackendBitPredictorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(err) => write!(f, "{err}"),
            Self::UnsupportedZpaq => {
                f.write_str("RateBackendBitPredictor does not support zpaq backends; use a non-zpaq rate_backend")
            }
            Self::Runtime(err) => write!(f, "{err}"),
            Self::StreamStart(err) => {
                write!(f, "failed to start RateBackend predictor stream: {err}")
            }
        }
    }
}

impl Error for RateBackendBitPredictorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Compile(err) => Some(err),
            _ => None,
        }
    }
}

impl From<SpecError> for RateBackendBitPredictorError {
    fn from(value: SpecError) -> Self {
        Self::Compile(value)
    }
}

/// Configuration for constructing a [`RateBackendBitPredictor`].
#[derive(Clone)]
#[non_exhaustive]
pub struct RateBackendBitPredictorConfig {
    /// Compiled backend used by the bit-level adapter.
    pub backend: CompiledRateBackend,
    /// Probability floor used when normalizing binary probabilities.
    pub min_prob: f64,
}

impl RateBackendBitPredictorConfig {
    /// Compile a rate backend into a bit-predictor configuration.
    pub fn compile(
        backend: RateBackend,
        min_prob: f64,
    ) -> Result<Self, RateBackendBitPredictorError> {
        let compiled = backend
            .compile()
            .map_err(RateBackendBitPredictorError::from)?;
        Ok(Self {
            backend: compiled,
            min_prob,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RateBackendJournalKind {
    Update,
    FrozenUpdate,
}

#[derive(Clone)]
struct RateBackendJournalEntry {
    kind: RateBackendJournalKind,
    checkpoint: RateBackendPredictorCheckpoint,
}

#[derive(Clone)]
struct RateBackendRollbackScope {
    checkpoint: RateBackendPredictorCheckpoint,
    journal_len: usize,
}

impl RateBackendBitPredictor {
    /// Create a new bit-level adapter.
    pub fn new(
        config: RateBackendBitPredictorConfig,
    ) -> Result<Self, RateBackendBitPredictorError> {
        let RateBackendBitPredictorConfig { backend, min_prob } = config;
        if backend.contains_zpaq() {
            return Err(RateBackendBitPredictorError::UnsupportedZpaq);
        }
        let mut predictor = crate::runtime::build_rate_backend_predictor(&backend, min_prob)
            .map_err(RateBackendBitPredictorError::Runtime)?;
        predictor
            .begin_stream(None)
            .map_err(RateBackendBitPredictorError::StreamStart)?;
        Ok(Self {
            backend,
            min_prob,
            predictor,
            journal: Vec::new(),
            rollback_scopes: Vec::new(),
        })
    }

    #[inline(always)]
    fn bit_to_byte(sym: bool) -> u8 {
        if sym { 1u8 } else { 0u8 }
    }

    fn clone_state(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            min_prob: self.min_prob,
            predictor: self.predictor.clone(),
            journal: self.journal.clone(),
            rollback_scopes: self.rollback_scopes.clone(),
        }
    }

    fn checkpoint(&mut self, kind: RateBackendJournalKind) -> RateBackendJournalEntry {
        RateBackendJournalEntry {
            kind,
            checkpoint: self.predictor.checkpoint(),
        }
    }

    fn restore_last(&mut self, expected_kind: RateBackendJournalKind) {
        assert!(
            self.rollback_scopes.is_empty(),
            "RateBackendBitPredictor per-symbol rollback inside active scope is unsupported"
        );
        let entry = self
            .journal
            .pop()
            .expect("RateBackendBitPredictor rollback underflow");
        assert_eq!(
            entry.kind, expected_kind,
            "RateBackendBitPredictor rollback kind mismatch: expected {expected_kind:?}, got {:?}",
            entry.kind
        );
        self.predictor.restore_checkpoint(&entry.checkpoint);
        if self.rollback_scopes.is_empty() && self.journal.is_empty() {
            self.predictor.clear_checkpoints_if_supported();
        }
    }
}

impl Predictor for RateBackendBitPredictor {
    fn update(&mut self, sym: bool) {
        if self.rollback_scopes.is_empty() {
            let checkpoint = self.checkpoint(RateBackendJournalKind::Update);
            self.journal.push(checkpoint);
        }
        self.predictor.update(Self::bit_to_byte(sym));
    }

    fn commit_update(&mut self, sym: bool) {
        self.predictor.update(Self::bit_to_byte(sym));
    }

    fn update_history(&mut self, sym: bool) {
        if self.rollback_scopes.is_empty() {
            let checkpoint = self.checkpoint(RateBackendJournalKind::FrozenUpdate);
            self.journal.push(checkpoint);
        }
        self.predictor.update_frozen(Self::bit_to_byte(sym));
    }

    fn commit_update_history(&mut self, sym: bool) {
        self.predictor.update_frozen(Self::bit_to_byte(sym));
    }

    fn revert(&mut self) {
        self.restore_last(RateBackendJournalKind::Update);
    }

    fn pop_history(&mut self) {
        self.restore_last(RateBackendJournalKind::FrozenUpdate);
    }

    fn begin_rollback_scope(&mut self) {
        let checkpoint = self.predictor.checkpoint();
        self.rollback_scopes.push(RateBackendRollbackScope {
            checkpoint,
            journal_len: self.journal.len(),
        });
    }

    fn rollback_scope(&mut self) -> bool {
        let Some(scope) = self.rollback_scopes.pop() else {
            return false;
        };
        self.predictor.restore_checkpoint(&scope.checkpoint);
        self.journal.truncate(scope.journal_len);
        if self.rollback_scopes.is_empty() && self.journal.is_empty() {
            self.predictor.clear_checkpoints_if_supported();
        }
        true
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        binary_prediction_from_log_probs(
            self.predictor.log_prob(0),
            self.predictor.log_prob(1),
            self.min_prob,
        )
        .prob(sym)
    }

    fn model_name(&self) -> String {
        format!("RateBackendBits({})", self.backend.default_name())
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        Box::new(self.clone_state())
    }

    fn reset_conditioning_history(&mut self) -> Result<(), String> {
        self.journal.clear();
        self.rollback_scopes.clear();
        self.predictor.reset_frozen(None)
    }
}

/// Build the predictor used by the MC-AIXI runtime from a compiled backend.
pub(crate) fn build_mc_aixi_predictor(
    backend: &CompiledRateBackend,
    #[allow(unused_variables)] percept_bits: usize,
) -> Result<Box<dyn Predictor>, PredictorBuildError> {
    match backend.canonical_spec() {
        #[cfg(feature = "backend-ctw")]
        RateBackend::FacCtw { base_depth, .. } => {
            Ok(Box::new(FacCtwPredictor::new(*base_depth, percept_bits)))
        }
        #[cfg(feature = "backend-ctw")]
        RateBackend::Ctw { depth } => Ok(Box::new(CtwPredictor::new(*depth))),
        #[cfg(feature = "backend-rosa")]
        RateBackend::RosaPlus { max_order } => Ok(Box::new(RosaPredictor::new(*max_order))),
        _ => Ok(Box::new(build_compiled_bit_predictor(backend)?)),
    }
}

/// Build the predictor used by the AIQI runtime from a compiled backend.
pub(crate) fn build_aiqi_predictor(
    backend: &CompiledRateBackend,
    #[allow(unused_variables)] return_bits: usize,
) -> Result<Box<dyn Predictor>, PredictorBuildError> {
    match backend.canonical_spec() {
        #[cfg(feature = "backend-ctw")]
        RateBackend::Ctw { depth } => Ok(Box::new(CtwPredictor::new(*depth))),
        #[cfg(feature = "backend-ctw")]
        RateBackend::FacCtw { base_depth, .. } => {
            Ok(Box::new(FacCtwPredictor::new(*base_depth, return_bits)))
        }
        _ => Ok(Box::new(build_compiled_bit_predictor(backend)?)),
    }
}

fn build_compiled_bit_predictor(
    backend: &CompiledRateBackend,
) -> Result<RateBackendBitPredictor, PredictorBuildError> {
    let bit_backend = if backend.supports_bit_token_adaptation() {
        backend
            .adapt_for_bit_tokens()
            .map_err(PredictorBuildError::BitAdaptation)?
    } else {
        backend.clone()
    };
    RateBackendBitPredictor::new(RateBackendBitPredictorConfig {
        backend: bit_backend,
        min_prob: DEFAULT_MIN_PROB,
    })
    .map_err(PredictorBuildError::BitPredictor)
}

#[cfg(feature = "backend-rwkv")]
use crate::coders::softmax_pdf_floor_inplace;

/// A predictor using the RWKV neural network architecture.
///
/// This provides a deep learning based world model for AIXI, allowing
/// the agent to leverage large pre-trained models for sequence prediction.
#[cfg(feature = "backend-rwkv")]
pub struct RwkvPredictor {
    compressor: RwkvCompressor,
    history: Vec<(RwkvState, Vec<f64>)>,
}

#[cfg(feature = "backend-rwkv")]
impl RwkvPredictor {
    /// Creates a new `RwkvPredictor` from an initialized `Model`.
    pub fn new(model: Arc<RwkvModel>) -> Self {
        let mut compressor = RwkvCompressor::new_from_model(model);
        let vocab_size = compressor.vocab_size();
        let logits = compressor
            .model
            .forward(&mut compressor.scratch, 0, &mut compressor.state);
        softmax_pdf_floor_inplace(logits, vocab_size, &mut compressor.pdf_buffer);

        Self {
            compressor,
            history: Vec::new(),
        }
    }

    /// Creates a new `RwkvPredictor` from a method string.
    pub fn from_method(method: &str) -> InfotheoryResult<Self> {
        let spec = crate::rwkvzip::parse_method_spec(method)
            .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?;
        Self::from_method_spec(&spec)
    }

    /// Creates a new `RwkvPredictor` from a parsed method spec.
    pub fn from_method_spec(method: &crate::rwkvzip::MethodSpec) -> InfotheoryResult<Self> {
        let mut compressor = RwkvCompressor::new_from_method_spec(method)
            .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?;
        compressor.forward_to_internal_pdf(0);
        Ok(Self {
            compressor,
            history: Vec::new(),
        })
    }
}

#[cfg(feature = "backend-rwkv")]
impl Predictor for RwkvPredictor {
    fn update(&mut self, sym: bool) {
        // Save current state and pdf
        self.history.push((
            self.compressor.state.clone(),
            self.compressor.pdf_buffer.clone(),
        ));

        let byte = if sym { 1u32 } else { 0u32 };
        let vocab_size = self.compressor.vocab_size();

        let logits = self.compressor.model.forward(
            &mut self.compressor.scratch,
            byte,
            &mut self.compressor.state,
        );
        softmax_pdf_floor_inplace(logits, vocab_size, &mut self.compressor.pdf_buffer);
    }

    fn revert(&mut self) {
        if let Some((state, pdf)) = self.history.pop() {
            self.compressor.state = state;
            self.compressor.pdf_buffer = pdf;
        }
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        binary_prediction_from_probs(
            self.compressor.pdf_buffer[0],
            self.compressor.pdf_buffer[1],
            DEFAULT_MIN_PROB,
        )
        .prob(sym)
    }

    fn model_name(&self) -> String {
        "RWKV".to_string()
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        Box::new(Self {
            compressor: self.compressor.clone(),
            history: self.history.clone(),
        })
    }
}

/// A predictor using the Mamba neural network architecture.
#[cfg(feature = "backend-mamba")]
pub struct MambaPredictor {
    compressor: MambaCompressor,
    history: Vec<(MambaState, Vec<f64>)>,
}

#[cfg(feature = "backend-mamba")]
impl MambaPredictor {
    /// Creates a new `MambaPredictor` from an initialized `Model`.
    pub fn new(model: Arc<MambaModel>) -> Self {
        let mut compressor = MambaCompressor::new_from_model(model);
        let logits = compressor
            .model
            .forward(&mut compressor.scratch, 0, &mut compressor.state)
            .to_vec();
        let bias = compressor.online_bias_snapshot();
        MambaCompressor::logits_to_pdf(&logits, bias.as_deref(), &mut compressor.pdf_buffer);

        Self {
            compressor,
            history: Vec::new(),
        }
    }

    /// Creates a new `MambaPredictor` from a method string.
    pub fn from_method(method: &str) -> InfotheoryResult<Self> {
        let spec = crate::mambazip::parse_method_spec(method)
            .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?;
        Self::from_method_spec(&spec)
    }

    /// Creates a new `MambaPredictor` from a parsed method spec.
    pub fn from_method_spec(method: &crate::mambazip::MethodSpec) -> InfotheoryResult<Self> {
        let mut compressor = MambaCompressor::new_from_method_spec(method)
            .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?;
        let mut pdf = vec![0.0f64; compressor.vocab_size()];
        compressor.forward_to_pdf(0, &mut pdf);
        compressor.pdf_buffer.clone_from(&pdf);
        Ok(Self {
            compressor,
            history: Vec::new(),
        })
    }
}

#[cfg(feature = "backend-mamba")]
impl Predictor for MambaPredictor {
    fn update(&mut self, sym: bool) {
        self.history.push((
            self.compressor.state.clone(),
            self.compressor.pdf_buffer.clone(),
        ));

        let byte = if sym { 1u32 } else { 0u32 };
        let logits = self
            .compressor
            .model
            .forward(
                &mut self.compressor.scratch,
                byte,
                &mut self.compressor.state,
            )
            .to_vec();
        let bias = self.compressor.online_bias_snapshot();
        MambaCompressor::logits_to_pdf(&logits, bias.as_deref(), &mut self.compressor.pdf_buffer);
    }

    fn revert(&mut self) {
        if let Some((state, pdf)) = self.history.pop() {
            self.compressor.state = state;
            self.compressor.pdf_buffer = pdf;
        }
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        binary_prediction_from_probs(
            self.compressor.pdf_buffer[0],
            self.compressor.pdf_buffer[1],
            DEFAULT_MIN_PROB,
        )
        .prob(sym)
    }

    fn model_name(&self) -> String {
        "Mamba".to_string()
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        Box::new(Self {
            compressor: self.compressor.clone(),
            history: self.history.clone(),
        })
    }
}

#[cfg(all(test, feature = "all-backends"))]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) {
        let diff = (a - b).abs();
        assert!(
            diff <= 1e-12,
            "expected probabilities to match exactly enough: left={a} right={b} diff={diff}"
        );
    }

    fn assert_binary_predictor_normalizes(mut predictor: Box<dyn Predictor>, label: &str) {
        for (step, &bit) in [false, true, true, false, true, false].iter().enumerate() {
            let p0 = predictor.predict_prob(false);
            let p1 = predictor.predict_prob(true);
            let sum = p0 + p1;
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "{label}: probabilities must sum to 1 at step {step}, got p0={p0}, p1={p1}, sum={sum}",
            );
            assert!(
                (0.0..=1.0).contains(&p0) && (0.0..=1.0).contains(&p1),
                "{label}: probabilities must stay in [0,1] at step {step}, got p0={p0}, p1={p1}",
            );
            predictor.commit_update(bit);
        }
    }

    fn predictor_signature(
        mut predictor: RateBackendBitPredictor,
        probe: &[bool],
    ) -> Vec<(f64, f64)> {
        let mut signature = Vec::with_capacity(probe.len());
        for &bit in probe {
            signature.push((predictor.predict_prob(false), predictor.predict_prob(true)));
            predictor.commit_update(bit);
        }
        signature
    }

    fn bit_predictor(backend: RateBackend) -> RateBackendBitPredictor {
        let config = RateBackendBitPredictorConfig::compile(backend, DEFAULT_MIN_PROB)
            .expect("rate backend bit predictor config should compile");
        RateBackendBitPredictor::new(config).expect("rate backend predictor should initialize")
    }

    #[test]
    fn committed_rate_backend_updates_do_not_grow_journal() {
        let mut predictor = bit_predictor(RateBackend::RosaPlus { max_order: 8 });

        for idx in 0..512usize {
            predictor.commit_update((idx & 1) == 0);
            predictor.commit_update_history((idx % 3) == 0);
        }

        assert!(
            predictor.journal.is_empty(),
            "committed history should not retain rollback snapshots"
        );
    }

    #[cfg(feature = "backend-rosa")]
    #[test]
    fn rosa_predictor_conditioning_reset_clears_cursor_and_rollback_history() {
        let mut predictor = RosaPredictor::new(8);
        for &bit in &[true, false, true, true, false] {
            predictor.commit_update(bit);
        }
        assert!(
            !predictor.history.is_empty(),
            "precondition: rollback journal should be populated after committed updates"
        );
        predictor.model.advance_conditioning_byte(1);
        predictor.model.advance_conditioning_byte(0);

        predictor
            .reset_conditioning_history()
            .expect("rosa conditioning reset should succeed");
        assert!(
            predictor.history.is_empty(),
            "conditioning reset must clear rollback journal state"
        );
        assert_eq!(
            predictor.model.conditioning_cursor(),
            0,
            "conditioning reset must return predictive cursor to root state"
        );
    }

    #[test]
    fn reversible_rate_backend_update_paths_round_trip_exactly() {
        let mut predictor = bit_predictor(RateBackend::RosaPlus { max_order: 8 });
        for &bit in &[true, false, true, true, false, false, true] {
            predictor.commit_update(bit);
        }

        let baseline_after_train = predictor.clone_state();
        predictor.update(true);
        predictor.update(false);
        predictor.revert();
        predictor.revert();
        assert_eq!(predictor.journal.len(), baseline_after_train.journal.len());

        let train_probe = [true, false, false, true, true, false];
        let got = predictor_signature(predictor.clone_state(), &train_probe);
        let want = predictor_signature(baseline_after_train.clone_state(), &train_probe);
        for ((got0, got1), (want0, want1)) in got.into_iter().zip(want.into_iter()) {
            approx_eq(got0, want0);
            approx_eq(got1, want1);
        }

        let baseline_after_history = baseline_after_train.clone_state();
        predictor.update_history(false);
        predictor.update_history(true);
        predictor.pop_history();
        predictor.pop_history();
        assert_eq!(
            predictor.journal.len(),
            baseline_after_history.journal.len()
        );

        let history_probe = [false, true, true, false, false, true];
        let got = predictor_signature(predictor.clone_state(), &history_probe);
        let want = predictor_signature(baseline_after_history, &history_probe);
        for ((got0, got1), (want0, want1)) in got.into_iter().zip(want.into_iter()) {
            approx_eq(got0, want0);
            approx_eq(got1, want1);
        }
    }

    #[test]
    fn long_committed_history_does_not_contaminate_clone_rollback_state() {
        let mut predictor = bit_predictor(RateBackend::RosaPlus { max_order: 8 });

        for idx in 0..2048usize {
            predictor.commit_update((idx & 7) < 3);
            predictor.commit_update_history((idx % 5) < 2);
        }
        assert!(predictor.journal.is_empty());

        let mut cloned = predictor.clone_state();
        assert!(
            cloned.journal.is_empty(),
            "clone state should only carry active reversible rollback depth"
        );

        let baseline = predictor_signature(predictor.clone_state(), &[true, false, true, false]);
        cloned.update(true);
        cloned.revert();
        cloned.update_history(false);
        cloned.pop_history();
        assert!(cloned.journal.is_empty());

        let after_round_trip = predictor_signature(cloned, &[true, false, true, false]);
        for ((got0, got1), (want0, want1)) in after_round_trip.into_iter().zip(baseline.into_iter())
        {
            approx_eq(got0, want0);
            approx_eq(got1, want1);
        }
    }

    #[test]
    fn rollback_scope_restores_simulation_state_without_growing_journal() {
        let mut predictor = bit_predictor(RateBackend::RosaPlus { max_order: 8 });
        for &bit in &[true, false, true, false, true] {
            predictor.commit_update(bit);
        }

        let baseline = predictor_signature(predictor.clone_state(), &[true, true, false, false]);
        predictor.begin_rollback_scope();
        for idx in 0..512usize {
            predictor.update((idx & 1) == 0);
            predictor.update_history((idx % 3) == 0);
        }
        assert!(
            predictor.journal.is_empty(),
            "scoped reversible updates should not retain per-bit snapshots"
        );
        assert!(predictor.rollback_scope(), "scope rollback should succeed");
        assert!(predictor.journal.is_empty());

        let after = predictor_signature(predictor, &[true, true, false, false]);
        for ((got0, got1), (want0, want1)) in after.into_iter().zip(baseline.into_iter()) {
            approx_eq(got0, want0);
            approx_eq(got1, want1);
        }
    }

    #[test]
    fn cloned_predictor_carries_only_active_scope_snapshots() {
        let mut predictor = bit_predictor(RateBackend::RosaPlus { max_order: 8 });
        for idx in 0..1024usize {
            predictor.commit_update((idx & 3) == 0);
        }

        predictor.begin_rollback_scope();
        for idx in 0..256usize {
            predictor.update((idx & 1) == 0);
        }
        let cloned = predictor.clone_state();
        assert!(
            cloned.journal.is_empty(),
            "scoped reversible updates should not leak per-bit journal state into clones"
        );
        assert_eq!(cloned.rollback_scopes.len(), 1);
    }

    #[test]
    fn generic_rate_backend_bit_predictors_normalize_binary_mass() {
        assert_binary_predictor_normalizes(
            Box::new(bit_predictor(RateBackend::RosaPlus { max_order: 8 })),
            "generic-rosa",
        );
        assert_binary_predictor_normalizes(
            Box::new(bit_predictor(RateBackend::Ppmd {
                order: 4,
                memory_mb: 8,
            })),
            "generic-ppmd",
        );
        assert_binary_predictor_normalizes(
            Box::new(bit_predictor(RateBackend::Match {
                hash_bits: 16,
                min_len: 2,
                max_len: 32,
                base_mix: 0.05,
                confidence_scale: 1.0,
            })),
            "generic-match",
        );
    }

    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn zpaq_predictor_normalizes_binary_mass() {
        assert_binary_predictor_normalizes(
            Box::new(ZpaqPredictor::new("1".to_string(), DEFAULT_MIN_PROB)),
            "zpaq",
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_predictor_normalizes_binary_mass() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
        let predictor = RwkvPredictor::from_method(method).expect("rwkv predictor");
        assert_binary_predictor_normalizes(Box::new(predictor), "rwkv");
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_predictor_normalizes_binary_mass() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=7,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
        let predictor = MambaPredictor::from_method(method).expect("mamba predictor");
        assert_binary_predictor_normalizes(Box::new(predictor), "mamba");
    }
}
