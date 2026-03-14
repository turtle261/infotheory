//! Context Tree Weighting (CTW) and Factorized Action-Conditional CTW (FAC-CTW).
//!
//! This module implements both the standard CTW algorithm for binary sequence prediction
//! and the FAC-CTW variant described in Veness et al. (2011) for agent-based prediction.
//!
//! # Arena Allocator
//! For performance with deep trees (D > 64), nodes are stored in a flat arena using
//! indices rather than `Box` pointers, eliminating pointer chasing and improving cache locality.
//!
//! # Shared History Optimization (FAC-CTW)
//! FAC-CTW uses k trees that share the same base history. Rather than duplicating the
//! history k times, a single shared history is maintained with per-tree length tracking.

use std::f64;

type Symbol = bool;

#[inline(always)]
fn ensure_log_caches(log_int: &mut Vec<f64>, log_half: &mut Vec<f64>, upto: usize) {
    if upto < log_int.len() {
        return;
    }
    let start = log_int.len();
    log_int.reserve(upto + 1 - start);
    log_half.reserve(upto + 1 - start);
    for n in start..=upto {
        if n == 0 {
            log_int.push(f64::NEG_INFINITY);
        } else {
            log_int.push((n as f64).ln());
        }
        log_half.push((n as f64 + 0.5).ln());
    }
}

/// Index into the node arena. `NONE` indicates no child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIndex(u32);

impl NodeIndex {
    /// Sentinel value indicating the absence of a node.
    pub const NONE: NodeIndex = NodeIndex(u32::MAX);

    #[inline(always)]
    fn from_usize(idx: usize) -> Self {
        Self(u32::try_from(idx).expect("ctw node index overflow"))
    }

    /// Returns `true` when this is [`NodeIndex::NONE`].
    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns `true` when this points to a valid arena node.
    #[inline(always)]
    pub fn is_some(self) -> bool {
        self.0 != u32::MAX
    }

    /// Convert to a `usize` arena index.
    ///
    /// Caller must ensure this is not `NONE`.
    #[inline(always)]
    pub fn get(self) -> usize {
        self.0 as usize
    }
}

/// A node in the Context Tree, stored in an arena.
#[derive(Clone, Debug)]
pub struct CtNode {
    /// Child node indices for context extensions (0 and 1).
    pub children: [NodeIndex; 2],
    /// Log-probability estimated by the KT-estimator.
    pub log_prob_kt: f64,
    /// Weighted log-probability (mix of KT and children's weighted probabilities).
    pub log_prob_weighted: f64,
    /// Counts of symbols (0 and 1) observed in this context.
    pub symbol_count: [u32; 2],
}

impl CtNode {
    /// Create a zero-initialized CTW node.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            children: [NodeIndex::NONE, NodeIndex::NONE],
            log_prob_kt: 0.0,
            log_prob_weighted: 0.0,
            symbol_count: [0, 0],
        }
    }

    /// Returns the total number of visits (symbol observations) at this node.
    #[inline(always)]
    pub fn visits(&self) -> u32 {
        self.symbol_count[0] + self.symbol_count[1]
    }
}

impl Default for CtNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Arena allocator for context tree nodes.
#[derive(Clone, Debug)]
pub struct CtArena {
    nodes: Vec<CtNode>,
    free_list: Vec<NodeIndex>,
}

impl CtArena {
    /// Create an empty arena with a small default reserve.
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            free_list: Vec::new(),
        }
    }

    /// Create an empty arena with explicit node capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            free_list: Vec::new(),
        }
    }

    /// Allocate a node and return its index.
    #[inline(always)]
    pub fn alloc(&mut self) -> NodeIndex {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx.get()] = CtNode::new();
            idx
        } else {
            let idx = NodeIndex::from_usize(self.nodes.len());
            self.nodes.push(CtNode::new());
            idx
        }
    }

    /// Mark a node index as reusable.
    #[inline(always)]
    pub fn free(&mut self, idx: NodeIndex) {
        if idx.is_some() {
            self.free_list.push(idx);
        }
    }

    /// Immutable access to a node by index.
    #[inline(always)]
    pub fn get(&self, idx: NodeIndex) -> &CtNode {
        &self.nodes[idx.get()]
    }

    /// Mutable access to a node by index.
    #[inline(always)]
    pub fn get_mut(&mut self, idx: NodeIndex) -> &mut CtNode {
        &mut self.nodes[idx.get()]
    }

    /// Remove all nodes and reset allocator state.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free_list.clear();
    }

    /// Returns approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<CtNode>()
            + self.free_list.capacity() * std::mem::size_of::<NodeIndex>()
    }
}

impl Default for CtArena {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn history_symbol(history: &[Symbol], depth: usize) -> Symbol {
    let idx = history.len().wrapping_sub(depth + 1);
    if depth < history.len() {
        history[idx]
    } else {
        false
    }
}

#[inline(always)]
fn update_weighted_log_prob(node: &mut CtNode, log_prob_w0: f64, log_prob_w1: f64, is_leaf: bool) {
    if is_leaf {
        node.log_prob_weighted = node.log_prob_kt;
    } else {
        let mut prob_w01_kt_ratio = (log_prob_w0 + log_prob_w1 - node.log_prob_kt).exp();
        if prob_w01_kt_ratio > 1.0 {
            prob_w01_kt_ratio = (node.log_prob_kt - log_prob_w0 - log_prob_w1).exp();
            node.log_prob_weighted = log_prob_w0 + log_prob_w1;
        } else {
            node.log_prob_weighted = node.log_prob_kt;
        }

        if prob_w01_kt_ratio.is_nan() {
            prob_w01_kt_ratio = 0.0;
        }
        node.log_prob_weighted += prob_w01_kt_ratio.ln_1p() - std::f64::consts::LN_2;
    }

    if node.log_prob_kt > 1.0e-10 {
        node.log_prob_kt = 0.0;
    }
    if node.log_prob_weighted > 1.0e-10 {
        node.log_prob_weighted = 0.0;
    }
}

#[inline(always)]
fn predict_ratio_kt(node: &CtNode, sym_idx: usize) -> f64 {
    let total = (node.symbol_count[0] + node.symbol_count[1]) as f64;
    let sym_count = node.symbol_count[sym_idx] as f64;
    (sym_count + 0.5) / (total + 1.0)
}

#[inline(always)]
fn predict_ratio_internal(
    node: &CtNode,
    path_child_log_prob: f64,
    sibling_log_prob: f64,
    child_ratio: f64,
    sym_idx: usize,
) -> f64 {
    let kt_ratio = predict_ratio_kt(node, sym_idx);
    let delta = path_child_log_prob + sibling_log_prob - node.log_prob_kt;
    if delta >= 0.0 {
        let inv_rho = (-delta).exp();
        (kt_ratio * inv_rho + child_ratio) / (1.0 + inv_rho)
    } else {
        let rho = delta.exp();
        (kt_ratio + rho * child_ratio) / (1.0 + rho)
    }
}

/// A Context Tree for binary sequence prediction using arena allocation.
#[derive(Clone)]
pub struct ContextTree {
    arena: CtArena,
    root: NodeIndex,
    history: Vec<Symbol>,
    max_depth: usize,
    path_nodes: Vec<NodeIndex>,
    path_symbols: Vec<Symbol>,
    log_int: Vec<f64>,
    log_half: Vec<f64>,
}

impl ContextTree {
    #[inline(always)]
    fn root_visits(&self) -> usize {
        self.arena.get(self.root).visits() as usize
    }

    /// Creates a new `ContextTree` with the given depth.
    pub fn new(depth: usize) -> Self {
        let mut arena = CtArena::with_capacity(1024.min(1 << depth.min(16)));
        let root = arena.alloc();
        Self {
            arena,
            root,
            history: Vec::new(),
            max_depth: depth,
            path_nodes: vec![NodeIndex::NONE; depth + 1],
            path_symbols: vec![false; depth],
            log_int: vec![f64::NEG_INFINITY],
            log_half: vec![(0.5f64).ln()],
        }
    }

    /// Resets the tree and history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.arena.clear();
        self.root = self.arena.alloc();
        self.path_nodes.fill(NodeIndex::NONE);
        self.path_symbols.fill(false);
    }

    /// Updates the tree with a new symbol.
    #[inline]
    pub fn update(&mut self, sym: Symbol) {
        // Cache bounds are determined by per-tree symbol counts, not history size.
        // For update we need log_int[total_before + 1] where total_before <= root visits.
        let upto = self.root_visits() + 1;
        ensure_log_caches(&mut self.log_int, &mut self.log_half, upto);
        self.update_from_root(sym);
        self.history.push(sym);
    }

    /// Reverts the last symbol update.
    #[inline]
    pub fn revert(&mut self) {
        let Some(last_sym) = self.history.pop() else {
            return;
        };
        // Revert uses current counts before decrement, so root visits is sufficient.
        let upto = self.root_visits();
        ensure_log_caches(&mut self.log_int, &mut self.log_half, upto);
        self.revert_from_root(last_sym);
    }

    /// Appends symbols to the history without updating the tree (for action conditioning).
    #[inline]
    pub fn update_history(&mut self, symbols: &[Symbol]) {
        self.history.extend_from_slice(symbols);
    }

    /// Removes the last symbol from history without tree update.
    #[inline]
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
    #[inline]
    pub fn predict(&mut self, sym: Symbol) -> f64 {
        self.predict_from_root(sym)
    }

    /// Shorthand for predicting the probability of symbol `1` (`true`).
    #[inline]
    pub fn predict_sym_prob(&mut self) -> f64 {
        self.predict(true)
    }

    /// Returns the total log-probability of the sequence observed so far.
    #[inline]
    pub fn get_log_block_probability(&self) -> f64 {
        self.arena.get(self.root).log_prob_weighted
    }

    /// Returns the configured maximum depth of the tree.
    #[inline]
    pub fn depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the current length of the history.
    #[inline]
    pub fn history_size(&self) -> usize {
        self.history.len()
    }

    // --- Internal methods ---

    #[inline]
    fn update_from_root(&mut self, sym: Symbol) {
        let mut current = self.root;
        self.path_nodes[0] = current;
        let mut path_len = 1usize;

        for depth in 0..self.max_depth {
            let child_sym = history_symbol(&self.history, depth);
            self.path_symbols[depth] = child_sym;
            let child_idx = self.arena.get(current).children[child_sym as usize];
            let next = if child_idx.is_none() {
                let new_child = self.arena.alloc();
                self.arena.get_mut(current).children[child_sym as usize] = new_child;
                new_child
            } else {
                child_idx
            };
            current = next;
            self.path_nodes[path_len] = current;
            path_len += 1;
        }

        self.unwind_update(sym, path_len);
    }

    #[inline]
    fn predict_from_root(&mut self, sym: Symbol) -> f64 {
        let mut current = self.root;
        self.path_nodes[0] = current;
        let mut path_len = 1usize;
        let mut reached_max_depth = true;

        for depth in 0..self.max_depth {
            let child_sym = history_symbol(&self.history, depth);
            self.path_symbols[depth] = child_sym;
            let child_idx = self.arena.get(current).children[child_sym as usize];
            if child_idx.is_none() {
                reached_max_depth = false;
                break;
            }
            current = child_idx;
            self.path_nodes[path_len] = current;
            path_len += 1;
        }

        let sym_idx = sym as usize;
        let mut ratio = 0.5f64;
        for i in (0..path_len).rev() {
            let idx = self.path_nodes[i];
            let node = self.arena.get(idx);
            if i + 1 == path_len && reached_max_depth {
                ratio = predict_ratio_kt(node, sym_idx);
                continue;
            }

            let child_sym = self.path_symbols[i];
            let path_child = if i + 1 < path_len {
                self.path_nodes[i + 1]
            } else {
                NodeIndex::NONE
            };
            let path_child_log_prob = if path_child.is_some() {
                self.arena.get(path_child).log_prob_weighted
            } else {
                0.0
            };
            let sibling = node.children[(!child_sym) as usize];
            let sibling_log_prob = if sibling.is_some() {
                self.arena.get(sibling).log_prob_weighted
            } else {
                0.0
            };
            ratio =
                predict_ratio_internal(node, path_child_log_prob, sibling_log_prob, ratio, sym_idx);
        }
        ratio
    }

    #[inline]
    fn revert_from_root(&mut self, sym: Symbol) {
        let mut current = self.root;
        self.path_nodes[0] = current;
        let mut path_len = 1usize;

        for depth in 0..self.max_depth {
            let child_sym = history_symbol(&self.history, depth);
            self.path_symbols[depth] = child_sym;
            let child_idx = self.arena.get(current).children[child_sym as usize];
            if child_idx.is_none() {
                break;
            }
            current = child_idx;
            self.path_nodes[path_len] = current;
            path_len += 1;
        }

        self.unwind_revert(sym, path_len);
    }

    #[inline(always)]
    fn unwind_update(&mut self, sym: Symbol, path_len: usize) {
        let sym_idx = sym as usize;
        for i in (0..path_len).rev() {
            let idx = self.path_nodes[i];
            let is_leaf = i + 1 == path_len;
            let (log_prob_w0, log_prob_w1) = if is_leaf {
                (0.0, 0.0)
            } else {
                let node = self.arena.get(idx);
                let child0 = node.children[0];
                let child1 = node.children[1];
                let w0 = if child0.is_some() {
                    self.arena.get(child0).log_prob_weighted
                } else {
                    0.0
                };
                let w1 = if child1.is_some() {
                    self.arena.get(child1).log_prob_weighted
                } else {
                    0.0
                };
                (w0, w1)
            };

            let node = self.arena.get_mut(idx);
            let total_before = (node.symbol_count[0] + node.symbol_count[1]) as usize;
            let sym_before = node.symbol_count[sym_idx] as usize;
            node.log_prob_kt += self.log_half[sym_before] - self.log_int[total_before + 1];
            node.symbol_count[sym_idx] += 1;
            update_weighted_log_prob(node, log_prob_w0, log_prob_w1, is_leaf);
        }
    }

    #[inline(always)]
    fn unwind_revert(&mut self, sym: Symbol, path_len: usize) {
        let sym_idx = sym as usize;
        for i in (0..path_len).rev() {
            let idx = self.path_nodes[i];
            let is_leaf = i + 1 == path_len;
            let (log_prob_w0, log_prob_w1) = if is_leaf {
                (0.0, 0.0)
            } else {
                let node = self.arena.get(idx);
                let child0 = node.children[0];
                let child1 = node.children[1];
                let w0 = if child0.is_some() {
                    self.arena.get(child0).log_prob_weighted
                } else {
                    0.0
                };
                let w1 = if child1.is_some() {
                    self.arena.get(child1).log_prob_weighted
                } else {
                    0.0
                };
                (w0, w1)
            };

            {
                let node = self.arena.get_mut(idx);
                let total = (node.symbol_count[0] + node.symbol_count[1]) as usize;
                let sym_count = node.symbol_count[sym_idx] as usize;
                if sym_count > 0 && total > 0 {
                    node.log_prob_kt -= self.log_half[sym_count - 1] - self.log_int[total];
                    node.symbol_count[sym_idx] -= 1;
                }
                update_weighted_log_prob(node, log_prob_w0, log_prob_w1, is_leaf);
            }

            if i > 0 && self.arena.get(idx).visits() == 0 {
                let parent_idx = self.path_nodes[i - 1];
                let child_sym = self.path_symbols[i - 1];
                self.arena.get_mut(parent_idx).children[child_sym as usize] = NodeIndex::NONE;
                self.arena.free(idx);
            }
        }
    }
}

// Factorized Action-Conditional CTW (FAC-CTW)
// =============================================================================

/// Core tree structure without owned history (for use in shared-history FAC-CTW).
#[derive(Clone)]
struct ContextTreeCore {
    arena: CtArena,
    root: NodeIndex,
    max_depth: usize,
    path_nodes: Vec<NodeIndex>,
    path_symbols: Vec<Symbol>,
    log_int: Vec<f64>,
    log_half: Vec<f64>,
}

impl ContextTreeCore {
    #[inline(always)]
    fn root_visits(&self) -> usize {
        self.arena.get(self.root).visits() as usize
    }

    fn new(depth: usize) -> Self {
        let mut arena = CtArena::with_capacity(1024.min(1 << depth.min(16)));
        let root = arena.alloc();
        Self {
            arena,
            root,
            max_depth: depth,
            path_nodes: vec![NodeIndex::NONE; depth + 1],
            path_symbols: vec![false; depth],
            log_int: vec![f64::NEG_INFINITY],
            log_half: vec![(0.5f64).ln()],
        }
    }

    fn clear(&mut self) {
        self.arena.clear();
        self.root = self.arena.alloc();
        self.path_nodes.fill(NodeIndex::NONE);
        self.path_symbols.fill(false);
    }

    /// Update tree with symbol, using shared history for context.
    #[inline]
    fn update(&mut self, sym: Symbol, shared_history: &[Symbol]) {
        // Do not scale caches with global shared history length; only this tree's
        // own visit counts are needed for KT updates.
        let upto = self.root_visits() + 1;
        ensure_log_caches(&mut self.log_int, &mut self.log_half, upto);
        self.update_path(sym, shared_history);
    }

    /// Revert last update, using shared history for context.
    #[inline]
    fn revert(&mut self, last_sym: Symbol, shared_history: &[Symbol]) {
        // Revert uses counts prior to decrement.
        let upto = self.root_visits();
        ensure_log_caches(&mut self.log_int, &mut self.log_half, upto);
        self.revert_path(last_sym, shared_history);
    }

    /// Predict probability of sym using shared history.
    #[inline]
    fn predict(&mut self, sym: Symbol, shared_history: &[Symbol]) -> f64 {
        self.predict_path(sym, shared_history)
    }

    #[inline]
    fn get_log_block_probability(&self) -> f64 {
        self.arena.get(self.root).log_prob_weighted
    }

    #[inline]
    fn update_path(&mut self, sym: Symbol, shared_history: &[Symbol]) {
        let mut current = self.root;
        self.path_nodes[0] = current;
        let mut path_len = 1usize;

        for depth in 0..self.max_depth {
            let child_sym = history_symbol(shared_history, depth);
            self.path_symbols[depth] = child_sym;
            let child_idx = self.arena.get(current).children[child_sym as usize];
            let next = if child_idx.is_none() {
                let new_child = self.arena.alloc();
                self.arena.get_mut(current).children[child_sym as usize] = new_child;
                new_child
            } else {
                child_idx
            };
            current = next;
            self.path_nodes[path_len] = current;
            path_len += 1;
        }

        self.unwind_update(sym, path_len);
    }

    #[inline]
    fn predict_path(&mut self, sym: Symbol, shared_history: &[Symbol]) -> f64 {
        let mut current = self.root;
        self.path_nodes[0] = current;
        let mut path_len = 1usize;
        let mut reached_max_depth = true;

        for depth in 0..self.max_depth {
            let child_sym = history_symbol(shared_history, depth);
            self.path_symbols[depth] = child_sym;
            let child_idx = self.arena.get(current).children[child_sym as usize];
            if child_idx.is_none() {
                reached_max_depth = false;
                break;
            }
            current = child_idx;
            self.path_nodes[path_len] = current;
            path_len += 1;
        }

        let sym_idx = sym as usize;
        let mut ratio = 0.5f64;
        for i in (0..path_len).rev() {
            let idx = self.path_nodes[i];
            let node = self.arena.get(idx);
            if i + 1 == path_len && reached_max_depth {
                ratio = predict_ratio_kt(node, sym_idx);
                continue;
            }

            let child_sym = self.path_symbols[i];
            let path_child = if i + 1 < path_len {
                self.path_nodes[i + 1]
            } else {
                NodeIndex::NONE
            };
            let path_child_log_prob = if path_child.is_some() {
                self.arena.get(path_child).log_prob_weighted
            } else {
                0.0
            };
            let sibling = node.children[(!child_sym) as usize];
            let sibling_log_prob = if sibling.is_some() {
                self.arena.get(sibling).log_prob_weighted
            } else {
                0.0
            };
            ratio =
                predict_ratio_internal(node, path_child_log_prob, sibling_log_prob, ratio, sym_idx);
        }
        ratio
    }

    #[inline(always)]
    fn revert_path(&mut self, sym: Symbol, shared_history: &[Symbol]) {
        let mut current = self.root;
        self.path_nodes[0] = current;
        let mut path_len = 1usize;

        for depth in 0..self.max_depth {
            let child_sym = history_symbol(shared_history, depth);
            self.path_symbols[depth] = child_sym;
            let child_idx = self.arena.get(current).children[child_sym as usize];
            if child_idx.is_none() {
                break;
            }
            current = child_idx;
            self.path_nodes[path_len] = current;
            path_len += 1;
        }

        self.unwind_revert(sym, path_len);
    }

    #[inline(always)]
    fn unwind_update(&mut self, sym: Symbol, path_len: usize) {
        let sym_idx = sym as usize;
        for i in (0..path_len).rev() {
            let idx = self.path_nodes[i];
            let is_leaf = i + 1 == path_len;
            let (log_prob_w0, log_prob_w1) = if is_leaf {
                (0.0, 0.0)
            } else {
                let node = self.arena.get(idx);
                let child0 = node.children[0];
                let child1 = node.children[1];
                let w0 = if child0.is_some() {
                    self.arena.get(child0).log_prob_weighted
                } else {
                    0.0
                };
                let w1 = if child1.is_some() {
                    self.arena.get(child1).log_prob_weighted
                } else {
                    0.0
                };
                (w0, w1)
            };

            let node = self.arena.get_mut(idx);
            let total_before = (node.symbol_count[0] + node.symbol_count[1]) as usize;
            let sym_before = node.symbol_count[sym_idx] as usize;
            node.log_prob_kt += self.log_half[sym_before] - self.log_int[total_before + 1];
            node.symbol_count[sym_idx] += 1;
            update_weighted_log_prob(node, log_prob_w0, log_prob_w1, is_leaf);
        }
    }

    #[inline(always)]
    fn unwind_revert(&mut self, sym: Symbol, path_len: usize) {
        let sym_idx = sym as usize;
        for i in (0..path_len).rev() {
            let idx = self.path_nodes[i];
            let is_leaf = i + 1 == path_len;
            let (log_prob_w0, log_prob_w1) = if is_leaf {
                (0.0, 0.0)
            } else {
                let node = self.arena.get(idx);
                let child0 = node.children[0];
                let child1 = node.children[1];
                let w0 = if child0.is_some() {
                    self.arena.get(child0).log_prob_weighted
                } else {
                    0.0
                };
                let w1 = if child1.is_some() {
                    self.arena.get(child1).log_prob_weighted
                } else {
                    0.0
                };
                (w0, w1)
            };

            {
                let node = self.arena.get_mut(idx);
                let total = (node.symbol_count[0] + node.symbol_count[1]) as usize;
                let sym_count = node.symbol_count[sym_idx] as usize;
                if sym_count > 0 && total > 0 {
                    node.log_prob_kt -= self.log_half[sym_count - 1] - self.log_int[total];
                    node.symbol_count[sym_idx] -= 1;
                }
                update_weighted_log_prob(node, log_prob_w0, log_prob_w1, is_leaf);
            }

            if i > 0 && self.arena.get(idx).visits() == 0 {
                let parent_idx = self.path_nodes[i - 1];
                let child_sym = self.path_symbols[i - 1];
                self.arena.get_mut(parent_idx).children[child_sym as usize] = NodeIndex::NONE;
                self.arena.free(idx);
            }
        }
    }
}

/// Factorized Action-Conditional Context Tree Weighting.
///
/// FAC-CTW uses `k` separate context trees, one for each bit of the percept space.
/// Tree `i` (0-indexed) has depth `base_depth + i`, ensuring each percept bit is
/// dependent on the same portion of history while incorporating type information.
///
/// **Optimization**: All trees share a single history vector. Each tree tracks only
/// its effective history length, avoiding k-way duplication of history data.
///
/// Reference: Veness et al. (2011), Section 5, Equation 33-34.
#[derive(Clone)]
pub struct FacContextTree {
    /// Core trees without owned history.
    trees: Vec<ContextTreeCore>,
    /// Single shared history for all trees.
    shared_history: Vec<Symbol>,
    /// Base context depth D.
    base_depth: usize,
    /// Number of percept bits (k = l_X).
    num_bits: usize,
}

impl FacContextTree {
    /// Creates a new FAC-CTW with `base_depth` D and `num_percept_bits` k.
    ///
    /// Tree i will have depth D + i to ensure proper context overlap.
    pub fn new(base_depth: usize, num_percept_bits: usize) -> Self {
        let trees = (0..num_percept_bits)
            .map(|i| ContextTreeCore::new(base_depth + i))
            .collect();
        Self {
            trees,
            shared_history: Vec::new(),
            base_depth,
            num_bits: num_percept_bits,
        }
    }

    /// Returns the number of percept bits (k).
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Returns the base context depth (D).
    #[inline]
    pub fn base_depth(&self) -> usize {
        self.base_depth
    }

    /// Updates tree `bit_index` with symbol `sym` and updates subsequent trees' effective lengths.
    ///
    /// Call this in sequence for bits 0..k of each percept.
    #[inline]
    pub fn update(&mut self, sym: Symbol, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits);

        // Update the tree responsible for this bit
        self.trees[bit_index].update(sym, &self.shared_history);

        // Append to shared history
        self.shared_history.push(sym);
    }

    /// Predicts the probability of `sym` at `bit_index`.
    #[inline]
    pub fn predict(&mut self, sym: Symbol, bit_index: usize) -> f64 {
        debug_assert!(bit_index < self.num_bits);
        self.trees[bit_index].predict(sym, &self.shared_history)
    }

    /// Reverts the update at `bit_index`.
    #[inline]
    pub fn revert(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits);

        // Pop from shared history first to get the symbol
        let Some(last_sym) = self.shared_history.pop() else {
            return;
        };

        // Revert the tree responsible for this bit
        self.trees[bit_index].revert(last_sym, &self.shared_history);
    }

    /// Updates all trees' effective history lengths with action symbols (no KT update).
    #[inline]
    pub fn update_history(&mut self, symbols: &[Symbol]) {
        self.shared_history.extend_from_slice(symbols);
    }

    /// Reverts history from all trees.
    #[inline]
    pub fn revert_history(&mut self, count: usize) {
        let new_len = self.shared_history.len().saturating_sub(count);
        self.shared_history.truncate(new_len);
    }

    /// Reset only the shared conditioning history while preserving fitted trees.
    #[inline]
    pub fn reset_history_only(&mut self) {
        self.shared_history.clear();
    }

    /// Returns the combined log block probability (sum of all trees).
    #[inline]
    pub fn get_log_block_probability(&self) -> f64 {
        self.trees
            .iter()
            .map(|t| t.get_log_block_probability())
            .sum()
    }

    /// Clears all trees and shared history.
    pub fn clear(&mut self) {
        for tree in &mut self.trees {
            tree.clear();
        }
        self.shared_history.clear();
    }

    /// Returns approximate memory usage in bytes (including shared history).
    pub fn memory_usage(&self) -> usize {
        let tree_mem: usize = self.trees.iter().map(|t| t.arena.memory_usage()).sum();
        let history_mem = self.shared_history.capacity() * std::mem::size_of::<Symbol>();
        tree_mem + history_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "ctw node index overflow")]
    fn node_index_from_usize_rejects_overflow() {
        let _ = NodeIndex::from_usize((u32::MAX as usize) + 1);
    }

    #[test]
    fn ct_node_stays_compact() {
        assert_eq!(std::mem::size_of::<CtNode>(), 32);
    }

    fn assert_close(a: f64, b: f64) {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(diff <= 1e-12 * scale, "a={a} b={b} diff={diff}");
    }

    #[test]
    fn fac_ctw_history_consistency() {
        let mut fac = FacContextTree::new(4, 4);

        // Add action history
        fac.update_history(&[true, false, true]);
        assert_eq!(fac.shared_history.len(), 3);

        // Update percept bits
        fac.update(true, 0);
        fac.update(false, 1);
        assert_eq!(fac.shared_history.len(), 5);

        // Revert
        fac.revert(1);
        assert_eq!(fac.shared_history.len(), 4);

        fac.revert(0);
        assert_eq!(fac.shared_history.len(), 3);
    }

    #[test]
    fn fac_ctw_log_cache_tracks_tree_visits_not_shared_history() {
        let mut fac = FacContextTree::new(8, 8);
        let updates_per_tree = 512usize;

        for step in 0..updates_per_tree {
            let bit = (step & 1) == 1;
            for bit_idx in 0..8usize {
                fac.update(bit, bit_idx);
            }
        }

        assert_eq!(fac.shared_history.len(), updates_per_tree * 8);
        for tree in &fac.trees {
            let visits = tree.arena.get(tree.root).visits() as usize;
            assert_eq!(visits, updates_per_tree);
            assert!(
                tree.log_int.len() <= visits + 1,
                "log_int grew to {} for visits={visits}",
                tree.log_int.len()
            );
            assert!(
                tree.log_half.len() <= visits + 1,
                "log_half grew to {} for visits={visits}",
                tree.log_half.len()
            );
        }
    }

    #[test]
    fn context_tree_predict_preserves_state() {
        let mut tree = ContextTree::new(6);
        for &bit in &[true, false, true, true, false, false, true, false] {
            tree.update(bit);
        }
        let p0_before = tree.predict(false);
        let p1_before = tree.predict(true);
        let log_before = tree.get_log_block_probability();
        let history_before = tree.history.clone();
        let _ = tree.predict(true);

        assert_eq!(tree.history, history_before);
        assert_close(tree.get_log_block_probability(), log_before);
        assert_close(tree.predict(false), p0_before);
        assert_close(tree.predict(true), p1_before);
    }

    #[test]
    fn context_tree_predict_matches_update_ratio() {
        let mut tree = ContextTree::new(7);
        for &bit in &[true, false, true, false, true, true, false, true, false] {
            tree.update(bit);
        }
        for &sym in &[false, true] {
            let predicted = tree.predict(sym);
            let mut reference = tree.clone();
            let before = reference.get_log_block_probability();
            reference.update(sym);
            let after = reference.get_log_block_probability();
            assert_close(predicted, (after - before).exp());
        }
    }

    #[test]
    fn fac_ctw_predict_preserves_state() {
        let mut fac = FacContextTree::new(5, 8);
        for &byte in b"fac ctw state preservation" {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                fac.update(bit, bit_idx);
            }
        }
        let p0_before = fac.predict(false, 3);
        let p1_before = fac.predict(true, 3);
        let log_before = fac.get_log_block_probability();
        let history_before = fac.shared_history.clone();
        let _ = fac.predict(true, 3);

        assert_eq!(fac.shared_history, history_before);
        assert_close(fac.get_log_block_probability(), log_before);
        assert_close(fac.predict(false, 3), p0_before);
        assert_close(fac.predict(true, 3), p1_before);
    }

    #[test]
    fn fac_ctw_predict_matches_update_ratio() {
        let mut fac = FacContextTree::new(6, 8);
        for &byte in b"fac ctw exact predictive ratio" {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                fac.update(bit, bit_idx);
            }
        }
        for &sym in &[false, true] {
            let predicted = fac.predict(sym, 4);
            let mut reference = fac.clone();
            let before = reference.get_log_block_probability();
            reference.update(sym, 4);
            let after = reference.get_log_block_probability();
            assert_close(predicted, (after - before).exp());
        }
    }
}
