//! Predictive models for AIXI.
//!
//! This module defines the `Predictor` trait, which encapsulates the logic
//! for learning from history and predicting future symbols. Different implementations
//! provide different complexity vs performance trade-offs.

use crate::ctw::ContextTree;
use rosaplus::{RosaPlus, RosaTx};
use rwkvzip::{Compressor, Model, State};
use std::sync::Arc;

/// Interface for an AIXI world model.
///
/// A predictor must be able to update its internal state based on observed symbols,
/// revert its state for Monte Carlo simulations, and provide probabilities for
pub trait Predictor {
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
}

/// A predictor using the Context Tree Weighting (CTW) algorithm.
///
/// CTW is the classic model used in MC-AIXI, providing efficient
/// probability estimation for binary sequences.
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
        format!("CTW(d={})", self.tree.depth())
    }
}

/// A predictor using the ROSA (Record of Success Automaton) algorithm.
///
/// ROSA is a sub-quadratic suffix automaton based language model that
/// can handle very long contexts efficiently.
pub struct RosaPredictor {
    model: RosaPlus,
    history: Vec<RosaTx>,
}

impl RosaPredictor {
    /// Creates a new `RosaPredictor` with the specified maximum Markov order.
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
        self.model.train_example_tx(&mut tx, &[byte]);
        self.history.push(tx);
    }

    fn revert(&mut self) {
        if let Some(tx) = self.history.pop() {
            self.model.rollback_tx(tx);
        }
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        let byte = if sym { 1u32 } else { 0u32 };
        self.model.prob_for_last(byte)
    }

    fn model_name(&self) -> String {
        "ROSA".to_string()
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
}
