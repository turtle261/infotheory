//! Predictive models for AIXI.
//!
//! This module defines the `Predictor` trait, which encapsulates the logic
//! for learning from history and predicting future symbols. Different implementations
//! provide different complexity vs performance trade-offs.

use crate::ctw::{ContextTree, FacContextTree};
use crate::zpaq_rate::ZpaqRateModel;
use rosaplus::{RosaPlus, RosaTx};
use rwkvzip::{Compressor, Model, State};
use std::sync::Arc;

/// Interface for an AIXI world model.
///
/// A predictor must be able to update its internal state based on observed symbols,
/// revert its state for Monte Carlo simulations, and provide probabilities for
pub trait Predictor: Send + Sync {
    /// Incorporates a new symbol into the model's training history.
    fn update(&mut self, sym: bool);

    /// Appends a symbol to the model's interaction history without necessarily
    /// updating the training counts immediately (backend dependent).
    fn update_history(&mut self, sym: bool) {
        self.update(sym);
    }

    /// Reverts the model to its state before the last `update`.
    fn revert(&mut self);

    /// Reverts the model to its state before the last `update_history`.
    fn pop_history(&mut self) {
        self.revert();
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
}

/// A predictor using the Action-Conditional CTW algorithm.
///
/// AC-CTW uses a single context tree for all bits in sequence.
/// For better type information exploitation, use `FacCtwPredictor`.
pub struct CtwPredictor {
    tree: ContextTree,
}

impl CtwPredictor {
    /// Creates a new `CtwPredictor` with the specified context depth.
    pub fn new(depth: usize) -> Self {
        Self {
            tree: ContextTree::new(depth),
        }
    }
}

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
}

/// A predictor using the Factorized Action-Conditional CTW (FAC-CTW) algorithm.
///
/// FAC-CTW uses k separate context trees (one per percept bit) with overlapping
/// context depths D+i-1 for bit i. This enables better exploitation of type
/// information within percepts, as described in Veness et al. (2011) Section 5.
///
/// This is the recommended CTW variant for MC-AIXI agents.
pub struct FacCtwPredictor {
    tree: FacContextTree,
    /// Current bit index within a percept (cycles 0..num_bits).
    current_bit: usize,
    /// Total number of percept bits (k).
    num_bits: usize,
}

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
}

/// A predictor using the ROSA-Plus (Rapid Online Suffix Automaton + Witten-Bell Smoother) algorithm.
///
/// ROSA is a (practically) sub-quadratic suffix automaton based language model that
/// can handle very long contexts efficiently.
pub struct RosaPredictor {
    model: RosaPlus,
    history: Vec<RosaTx>,
}

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
        let p0 = self.model.prob_for_last(0);
        let p1 = self.model.prob_for_last(1);
        let denom = (p0 + p1).max(1e-12);
        if sym { p1 / denom } else { p0 / denom }
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
}

/// A predictor using ZPAQ as a streaming rate model.
///
/// This maintains a full history so it can rebuild state on revert and handle
/// any misuse where `predict_prob` is called without a matching `update`.
pub struct ZpaqPredictor {
    method: String,
    min_prob: f64,
    model: ZpaqRateModel,
    history: Vec<u8>,
    pending: Option<(u8, f64)>,
}

unsafe impl Sync for ZpaqPredictor {}

impl ZpaqPredictor {
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
}

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
        let byte = if sym { 1u8 } else { 0u8 };
        if let Some((pending, logp)) = self.pending {
            if pending == byte {
                return logp.exp();
            }
            return self.log_prob_from_history(byte).exp();
        }
        let logp = self.model.log_prob(byte);
        self.pending = Some((byte, logp));
        logp.exp()
    }

    fn model_name(&self) -> String {
        format!("ZPAQ({})", self.method)
    }

    fn boxed_clone(&self) -> Box<dyn Predictor> {
        let mut model = ZpaqRateModel::new(self.method.clone(), self.min_prob);
        if !self.history.is_empty() {
            model.update_and_score(&self.history);
        }
        Box::new(Self {
            method: self.method.clone(),
            min_prob: self.min_prob,
            model,
            history: self.history.clone(),
            pending: None,
        })
    }
}

use rwkvzip::coders::softmax_pdf_floor_inplace;

/// A predictor using the RWKV neural network architecture.
///
/// This provides a deep learning based world model for AIXI, allowing
/// the agent to leverage large pre-trained models for sequence prediction.
pub struct RwkvPredictor {
    compressor: Compressor,
    history: Vec<(State, Vec<f64>)>,
}

impl RwkvPredictor {
    /// Creates a new `RwkvPredictor` from an initialized `Model`.
    pub fn new(model: Arc<Model>) -> Self {
        let mut compressor = Compressor::new_from_model(model);
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
}

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
        let idx = if sym { 1 } else { 0 };
        // pdf_buffer contains probabilities
        self.compressor.pdf_buffer[idx]
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
