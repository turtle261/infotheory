//! Context Tree Weighting (CTW) algorithm.
//!
//! CTW is a powerful and efficient binary sequence prediction algorithm.
//! It maintains a tree of contexts and uses the Krichevsky-Trofimov (KT)
//! estimator at each node to calculate weighted probabilities.

use std::f64;

type Symbol = bool;

/// A node in the Context Tree.
///
/// Each node stores the log-probabilities and symbol counts for a
/// specific context.
#[derive(Clone, Debug)]
pub struct CtNode {
    /// Child nodes for context extensions (0 and 1).
    children: [Option<Box<CtNode>>; 2],
    /// Log-probability estimated by the KT-estimator.
    log_prob_kt: f64,
    /// Weighted log-probability (mix of KT and children's weighted probabilities).
    log_prob_weighted: f64,
    /// Counts of symbols (0 and 1) observed in this context.
    symbol_count: [u32; 2],
}

impl CtNode {
    /// Creates a new `CtNode`.
    pub fn new() -> Self {
        Self {
            children: [None, None],
            log_prob_kt: 0.0,
            log_prob_weighted: 0.0,
            symbol_count: [0, 0],
        }
    }

    /// Returns the total number of visits (symbol observations) at this node.
    pub fn visits(&self) -> u32 {
        self.symbol_count[0] + self.symbol_count[1]
    }

    /// Returns a reference to a child node.
    pub fn child(&self, sym: Symbol) -> Option<&CtNode> {
        self.children[sym as usize].as_deref()
    }

    /// Returns a mutable reference to a child node.
    pub fn child_mut(&mut self, sym: Symbol) -> Option<&mut CtNode> {
        self.children[sym as usize].as_deref_mut()
    }

    /// Ensures a child node exists and returns a mutable reference.
    pub fn ensure_child(&mut self, sym: Symbol) -> &mut CtNode {
        if self.children[sym as usize].is_none() {
            self.children[sym as usize] = Some(Box::new(CtNode::new()));
        }
        self.children[sym as usize].as_deref_mut().unwrap()
    }

    /// Calculates the log-multiplier for the KT-estimator when observing `sym`.
    fn log_kt_mul(&self, sym: Symbol) -> f64 {
        let sym_idx = sym as usize;
        let denominator = ((self.symbol_count[0] + self.symbol_count[1] + 1) as f64).ln();
        (self.symbol_count[sym_idx] as f64 + 0.5).ln() - denominator
    }

    /// Updates the KT-estimator log-probability.
    pub fn update_log_kt(&mut self, sym: Symbol, revert: bool) {
        let sym_idx = sym as usize;
        if !revert {
            self.log_prob_kt += self.log_kt_mul(sym);
            self.symbol_count[sym_idx] += 1;
        } else {
            if self.symbol_count[sym_idx] == 0 {
                return;
            }
            self.symbol_count[sym_idx] -= 1;
            self.log_prob_kt -= self.log_kt_mul(sym);
        }
    }

    /// Updates the weighted probability of the node after a symbol observation.
    pub fn update(&mut self, sym: Symbol, revert: bool) {
        self.update_log_kt(sym, revert);

        let log_prob_w0 = self.children[0]
            .as_ref()
            .map(|c| c.log_prob_weighted)
            .unwrap_or(0.0);
        let log_prob_w1 = self.children[1]
            .as_ref()
            .map(|c| c.log_prob_weighted)
            .unwrap_or(0.0);

        // Mix math: log_p_weighted = log((p_kt + p_w0*p_w1)/2)
        let mut prob_w01_kt_ratio = (log_prob_w0 + log_prob_w1 - self.log_prob_kt).exp();

        if prob_w01_kt_ratio > 1.0 {
            prob_w01_kt_ratio = (self.log_prob_kt - log_prob_w0 - log_prob_w1).exp();
            self.log_prob_weighted = log_prob_w0 + log_prob_w1;
        } else {
            self.log_prob_weighted = self.log_prob_kt;
        }

        if prob_w01_kt_ratio.is_nan() {
            prob_w01_kt_ratio = 0.0;
        }

        self.log_prob_weighted += prob_w01_kt_ratio.ln_1p() - std::f64::consts::LN_2;

        self.sanity_check();
    }

    /// Updates a leaf node (where weighted probability equals KT probability).
    pub fn update_leaf(&mut self, sym: Symbol, revert: bool) {
        self.update_log_kt(sym, revert);
        self.log_prob_weighted = self.log_prob_kt;
        self.sanity_check();
    }

    fn sanity_check(&mut self) {
        if self.log_prob_kt > 1.0e-10 {
            self.log_prob_kt = 0.0;
        }
        if self.log_prob_weighted > 1.0e-10 {
            self.log_prob_weighted = 0.0;
        }
    }
}

/// A Context Tree for binary sequence prediction.
pub struct ContextTree {
    /// Root node of the tree.
    root: Box<CtNode>,
    /// Recent symbol history used as context.
    history: Vec<Symbol>,
    /// Maximum depth of the context tree.
    max_depth: usize,
}

impl ContextTree {
    /// Creates a new `ContextTree` with the given depth.
    pub fn new(depth: usize) -> Self {
        Self {
            root: Box::new(CtNode::new()),
            history: Vec::new(),
            max_depth: depth,
        }
    }

    /// Resets the tree and history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.root = Box::new(CtNode::new());
    }

    /// Updates the tree with a new symbol.
    pub fn update(&mut self, sym: Symbol) {
        if self.history.len() >= self.max_depth {
            let history_len = self.history.len();
            let context = &self.history[history_len - self.max_depth..];

            Self::update_node(&mut self.root, context, sym, false, 0);
        }
        self.history.push(sym);
    }

    fn update_node(
        node: &mut CtNode,
        context: &[Symbol],
        sym: Symbol,
        revert: bool,
        depth: usize,
    ) -> bool {
        let max_depth = context.len();

        if depth == max_depth {
            node.update_leaf(sym, revert);
            return revert && node.visits() == 0;
        }

        let child_sym = context[max_depth - 1 - depth];

        let kill_child = {
            let child = node.ensure_child(child_sym);
            Self::update_node(child, context, sym, revert, depth + 1)
        };

        if kill_child {
            node.children[child_sym as usize] = None;
        }

        node.update(sym, revert);
        revert && node.visits() == 0
    }

    /// Appends symbols to the history without updating the tree.
    pub fn update_history(&mut self, symbols: &[Symbol]) {
        self.history.extend_from_slice(symbols);
    }

    /// Reverts the tree to its state before the last `update`.
    pub fn revert(&mut self) {
        if self.history.len() < self.max_depth + 1 {
            if self.history.len() > self.max_depth {
                let last_sym = *self.history.last().unwrap();
                self.history.pop();

                let history_len = self.history.len();
                let context = &self.history[history_len - self.max_depth..];
                Self::update_node(&mut self.root, context, last_sym, true, 0);
            } else {
                self.history.pop();
            }
        } else {
            let last_sym = *self.history.last().unwrap();
            self.history.pop();
            let history_len = self.history.len();
            let context = &self.history[history_len - self.max_depth..];
            Self::update_node(&mut self.root, context, last_sym, true, 0);
        }
    }

    /// Removes the last symbol from history.
    pub fn revert_history(&mut self) {
        self.history.pop();
    }

    /// Truncates the history to `new_size`.
    pub fn truncate_history(&mut self, new_size: usize) {
        if new_size < self.history.len() {
            self.history.truncate(new_size);
        }
    }

    /// Predicts the probability of the next symbol being `sym`.
    pub fn predict(&mut self, sym: Symbol) -> f64 {
        let log_prob_weighted = self.root.log_prob_weighted;
        self.update(sym);
        let log_prob_weighted_prime = self.root.log_prob_weighted;
        self.revert();
        (log_prob_weighted_prime - log_prob_weighted).exp()
    }

    /// Shorthand for predicting the probability of symbol `1` (`true`).
    pub fn predict_sym_prob(&mut self) -> f64 {
        self.predict(true)
    }

    /// Returns the total log-probability of the sequence observed so far.
    pub fn get_log_block_probability(&self) -> f64 {
        self.root.log_prob_weighted
    }

    /// Returns the configured maximum depth of the tree.
    pub fn depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the current length of the history.
    pub fn history_size(&self) -> usize {
        self.history.len()
    }
}
