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

/// Index into the node arena. `NONE` indicates no child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIndex(u32);

impl NodeIndex {
    pub const NONE: NodeIndex = NodeIndex(u32::MAX);

    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline(always)]
    pub fn is_some(self) -> bool {
        self.0 != u32::MAX
    }

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

/// Arena allocator for context tree nodes.
#[derive(Clone, Debug)]
pub struct CtArena {
    nodes: Vec<CtNode>,
    free_list: Vec<NodeIndex>,
}

impl CtArena {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            free_list: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            free_list: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn alloc(&mut self) -> NodeIndex {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx.get()] = CtNode::new();
            idx
        } else {
            let idx = NodeIndex(self.nodes.len() as u32);
            self.nodes.push(CtNode::new());
            idx
        }
    }

    #[inline(always)]
    pub fn free(&mut self, idx: NodeIndex) {
        if idx.is_some() {
            self.free_list.push(idx);
        }
    }

    #[inline(always)]
    pub fn get(&self, idx: NodeIndex) -> &CtNode {
        &self.nodes[idx.get()]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, idx: NodeIndex) -> &mut CtNode {
        &mut self.nodes[idx.get()]
    }

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

/// A Context Tree for binary sequence prediction using arena allocation.
#[derive(Clone)]
pub struct ContextTree {
    arena: CtArena,
    root: NodeIndex,
    history: Vec<Symbol>,
    max_depth: usize,
    context_buf: Vec<Symbol>,
}

impl ContextTree {
    /// Creates a new `ContextTree` with the given depth.
    pub fn new(depth: usize) -> Self {
        let mut arena = CtArena::with_capacity(1024.min(1 << depth.min(16)));
        let root = arena.alloc();
        Self {
            arena,
            root,
            history: Vec::new(),
            max_depth: depth,
            context_buf: vec![false; depth],
        }
    }

    /// Resets the tree and history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.arena.clear();
        self.root = self.arena.alloc();
        self.context_buf.fill(false);
    }

    /// Updates the tree with a new symbol.
    #[inline]
    pub fn update(&mut self, sym: Symbol) {
        self.prepare_context();
        self.update_from_root(sym, false);
        self.history.push(sym);
    }

    /// Reverts the last symbol update.
    #[inline]
    pub fn revert(&mut self) {
        let Some(last_sym) = self.history.pop() else {
            return;
        };
        self.prepare_context();
        self.update_from_root(last_sym, true);
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
        let log_prob_before = self.arena.get(self.root).log_prob_weighted;
        self.update(sym);
        let log_prob_after = self.arena.get(self.root).log_prob_weighted;
        self.revert();
        (log_prob_after - log_prob_before).exp()
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

    #[inline(always)]
    fn prepare_context(&mut self) {
        self.context_buf.fill(false);
        let history_len = self.history.len();
        let copy_len = history_len.min(self.max_depth);
        if copy_len > 0 {
            self.context_buf[self.max_depth - copy_len..]
                .copy_from_slice(&self.history[history_len - copy_len..]);
        }
    }

    #[inline]
    fn update_from_root(&mut self, sym: Symbol, revert: bool) {
        self.update_node_iterative(self.root, sym, revert);
    }

    /// Iterative update to avoid deep recursion and enable better inlining.
    #[inline]
    fn update_node_iterative(&mut self, root_idx: NodeIndex, sym: Symbol, revert: bool) {
        let max_depth = self.max_depth;

        // Build path from root to leaf
        let mut path: Vec<NodeIndex> = Vec::with_capacity(max_depth + 1);
        path.push(root_idx);

        let mut current = root_idx;
        for depth in 0..max_depth {
            let child_sym = self.context_buf[max_depth - 1 - depth];
            let child_idx = self.arena.get(current).children[child_sym as usize];

            if revert {
                if child_idx.is_none() {
                    break;
                }
                current = child_idx;
            } else {
                let child = if child_idx.is_none() {
                    let new_child = self.arena.alloc();
                    self.arena.get_mut(current).children[child_sym as usize] = new_child;
                    new_child
                } else {
                    child_idx
                };
                current = child;
            }
            path.push(current);
        }

        // Update nodes from leaf to root
        let leaf_depth = path.len() - 1;
        for (i, &node_idx) in path.iter().enumerate().rev() {
            let is_leaf = i == leaf_depth;
            self.update_single_node(node_idx, sym, revert, is_leaf);

            // Clean up empty children during revert
            if revert && i > 0 {
                let parent_idx = path[i - 1];
                let depth = i - 1;
                let child_sym = self.context_buf[max_depth - 1 - depth];
                if self.arena.get(node_idx).visits() == 0 {
                    self.arena.get_mut(parent_idx).children[child_sym as usize] = NodeIndex::NONE;
                    self.arena.free(node_idx);
                }
            }
        }
    }

    #[inline(always)]
    fn update_single_node(&mut self, idx: NodeIndex, sym: Symbol, revert: bool, is_leaf: bool) {
        // Read child weighted probs BEFORE taking mutable borrow
        let (log_prob_w0, log_prob_w1) = if !is_leaf {
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
        } else {
            (0.0, 0.0)
        };

        let node = self.arena.get_mut(idx);

        // Update KT estimator
        let sym_idx = sym as usize;
        if !revert {
            node.log_prob_kt += log_kt_mul(node.symbol_count, sym);
            node.symbol_count[sym_idx] += 1;
        } else {
            let total = node.symbol_count[0] + node.symbol_count[1];
            if node.symbol_count[sym_idx] > 0 && total > 0 {
                let numerator = (node.symbol_count[sym_idx] as f64 - 0.5).ln();
                let denominator = (total as f64).ln();
                node.log_prob_kt -= numerator - denominator;
                node.symbol_count[sym_idx] -= 1;
            }
        }

        // Update weighted probability
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

        // Sanity check
        if node.log_prob_kt > 1.0e-10 {
            node.log_prob_kt = 0.0;
        }
        if node.log_prob_weighted > 1.0e-10 {
            node.log_prob_weighted = 0.0;
        }
    }
}

/// KT estimator log-multiplier calculation.
#[inline(always)]
fn log_kt_mul(counts: [u32; 2], sym: Symbol) -> f64 {
    let sym_idx = sym as usize;
    let denominator = ((counts[0] + counts[1] + 1) as f64).ln();
    (counts[sym_idx] as f64 + 0.5).ln() - denominator
}

// =============================================================================
// Factorized Action-Conditional CTW (FAC-CTW)
// =============================================================================

/// Core tree structure without owned history (for use in shared-history FAC-CTW).
#[derive(Clone)]
struct ContextTreeCore {
    arena: CtArena,
    root: NodeIndex,
    max_depth: usize,
    context_buf: Vec<Symbol>,
    /// The effective history length this tree sees (prefix of shared history).
    effective_history_len: usize,
}

impl ContextTreeCore {
    fn new(depth: usize) -> Self {
        let mut arena = CtArena::with_capacity(1024.min(1 << depth.min(16)));
        let root = arena.alloc();
        Self {
            arena,
            root,
            max_depth: depth,
            context_buf: vec![false; depth],
            effective_history_len: 0,
        }
    }

    fn clear(&mut self) {
        self.arena.clear();
        self.root = self.arena.alloc();
        self.context_buf.fill(false);
        self.effective_history_len = 0;
    }

    /// Prepares context buffer from shared history using this tree's effective length.
    #[inline(always)]
    fn prepare_context(&mut self, shared_history: &[Symbol]) {
        self.context_buf.fill(false);
        let history_len = self.effective_history_len;
        let copy_len = history_len.min(self.max_depth);
        if copy_len > 0 {
            // Use the tail of shared_history up to our effective length
            let start = history_len.saturating_sub(copy_len);
            self.context_buf[self.max_depth - copy_len..]
                .copy_from_slice(&shared_history[start..history_len]);
        }
    }

    /// Update tree with symbol, using shared history for context.
    #[inline]
    fn update(&mut self, sym: Symbol, shared_history: &[Symbol]) {
        self.prepare_context(shared_history);
        self.update_node_iterative(sym, false);
        self.effective_history_len += 1;
    }

    /// Revert last update, using shared history for context.
    #[inline]
    fn revert(&mut self, last_sym: Symbol, shared_history: &[Symbol]) {
        if self.effective_history_len == 0 {
            return;
        }
        self.effective_history_len -= 1;
        self.prepare_context(shared_history);
        self.update_node_iterative(last_sym, true);
    }

    /// Increment effective history length without tree update (for action conditioning).
    #[inline]
    fn extend_history(&mut self, count: usize) {
        self.effective_history_len += count;
    }

    /// Decrement effective history length without tree update.
    #[inline]
    fn shrink_history(&mut self, count: usize) {
        self.effective_history_len = self.effective_history_len.saturating_sub(count);
    }

    /// Predict probability of sym using shared history.
    #[inline]
    fn predict(&mut self, sym: Symbol, shared_history: &[Symbol]) -> f64 {
        let log_prob_before = self.arena.get(self.root).log_prob_weighted;
        self.update(sym, shared_history);
        let log_prob_after = self.arena.get(self.root).log_prob_weighted;
        // Manually revert without needing last symbol from history
        self.effective_history_len -= 1;
        self.prepare_context(shared_history);
        self.update_node_iterative(sym, true);
        (log_prob_after - log_prob_before).exp()
    }

    #[inline]
    fn get_log_block_probability(&self) -> f64 {
        self.arena.get(self.root).log_prob_weighted
    }

    #[inline]
    fn update_node_iterative(&mut self, sym: Symbol, revert: bool) {
        let max_depth = self.max_depth;

        let mut path: Vec<NodeIndex> = Vec::with_capacity(max_depth + 1);
        path.push(self.root);

        let mut current = self.root;
        for depth in 0..max_depth {
            let child_sym = self.context_buf[max_depth - 1 - depth];
            let child_idx = self.arena.get(current).children[child_sym as usize];

            if revert {
                if child_idx.is_none() {
                    break;
                }
                current = child_idx;
            } else {
                let child = if child_idx.is_none() {
                    let new_child = self.arena.alloc();
                    self.arena.get_mut(current).children[child_sym as usize] = new_child;
                    new_child
                } else {
                    child_idx
                };
                current = child;
            }
            path.push(current);
        }

        let leaf_depth = path.len() - 1;
        for (i, &node_idx) in path.iter().enumerate().rev() {
            let is_leaf = i == leaf_depth;
            self.update_single_node(node_idx, sym, revert, is_leaf);

            if revert && i > 0 {
                let parent_idx = path[i - 1];
                let depth = i - 1;
                let child_sym = self.context_buf[max_depth - 1 - depth];
                if self.arena.get(node_idx).visits() == 0 {
                    self.arena.get_mut(parent_idx).children[child_sym as usize] = NodeIndex::NONE;
                    self.arena.free(node_idx);
                }
            }
        }
    }

    #[inline(always)]
    fn update_single_node(&mut self, idx: NodeIndex, sym: Symbol, revert: bool, is_leaf: bool) {
        let (log_prob_w0, log_prob_w1) = if !is_leaf {
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
        } else {
            (0.0, 0.0)
        };

        let node = self.arena.get_mut(idx);

        let sym_idx = sym as usize;
        if !revert {
            node.log_prob_kt += log_kt_mul(node.symbol_count, sym);
            node.symbol_count[sym_idx] += 1;
        } else {
            let total = node.symbol_count[0] + node.symbol_count[1];
            if node.symbol_count[sym_idx] > 0 && total > 0 {
                let numerator = (node.symbol_count[sym_idx] as f64 - 0.5).ln();
                let denominator = (total as f64).ln();
                node.log_prob_kt -= numerator - denominator;
                node.symbol_count[sym_idx] -= 1;
            }
        }

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

        // Extend effective history length of all subsequent trees
        for tree in self.trees.iter_mut().skip(bit_index + 1) {
            tree.extend_history(1);
        }

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

        // Shrink effective history length of all subsequent trees
        for tree in self.trees.iter_mut().skip(bit_index + 1) {
            tree.shrink_history(1);
        }

        // Revert the tree responsible for this bit
        self.trees[bit_index].revert(last_sym, &self.shared_history);
    }

    /// Updates all trees' effective history lengths with action symbols (no KT update).
    #[inline]
    pub fn update_history(&mut self, symbols: &[Symbol]) {
        let count = symbols.len();
        for tree in &mut self.trees {
            tree.extend_history(count);
        }
        self.shared_history.extend_from_slice(symbols);
    }

    /// Reverts history from all trees.
    #[inline]
    pub fn revert_history(&mut self, count: usize) {
        for tree in &mut self.trees {
            tree.shrink_history(count);
        }
        let new_len = self.shared_history.len().saturating_sub(count);
        self.shared_history.truncate(new_len);
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
    fn arena_alloc_free() {
        let mut arena = CtArena::new();
        let a = arena.alloc();
        let b = arena.alloc();
        assert_ne!(a, b);
        arena.free(a);
        let c = arena.alloc();
        assert_eq!(a, c); // Reused from free list
    }

    #[test]
    fn context_tree_update_revert() {
        let mut tree = ContextTree::new(4);
        tree.update(true);
        tree.update(false);
        let log_p = tree.get_log_block_probability();
        tree.revert();
        tree.update(false);
        let log_p2 = tree.get_log_block_probability();
        assert!((log_p - log_p2).abs() < 1e-10);
    }

    #[test]
    fn context_tree_predict() {
        let mut tree = ContextTree::new(8);
        for _ in 0..100 {
            tree.update(true);
        }
        let p1 = tree.predict(true);
        let p0 = tree.predict(false);
        assert!(p1 > p0, "Should predict 1 more likely after many 1s");
        assert!((p1 + p0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fac_ctw_basic() {
        let mut fac = FacContextTree::new(8, 4);
        // Update 4 bits
        fac.update(true, 0);
        fac.update(false, 1);
        fac.update(true, 2);
        fac.update(false, 3);
        let log_p = fac.get_log_block_probability();
        assert!(log_p < 0.0);

        // Revert in reverse order
        fac.revert(3);
        fac.revert(2);
        fac.revert(1);
        fac.revert(0);
        let log_p_empty = fac.get_log_block_probability();
        assert!((log_p_empty - 0.0).abs() < 1e-10);
    }

    #[test]
    fn fac_ctw_predict_sums_to_one() {
        let mut fac = FacContextTree::new(4, 8);
        // Add some history
        fac.update_history(&[true, false, true, true, false, false, true, false]);
        for bit in 0..8 {
            let p0 = fac.predict(false, bit);
            let p1 = fac.predict(true, bit);
            assert!(
                (p0 + p1 - 1.0).abs() < 1e-10,
                "Probabilities should sum to 1 at bit {}: p0={}, p1={}", bit, p0, p1
            );
        }
    }

    #[test]
    fn fac_ctw_shared_history_memory() {
        let fac = FacContextTree::new(32, 64);
        // Shared history should be single allocation, not 64x
        // Initial memory: 64 trees each with arena + context_buf (~32-96 bytes)
        // This is dominated by arena pre-allocation, not history
        let mem = fac.memory_usage();
        // With shared history, we don't duplicate the Vec<Symbol> 64 times
        // 64 trees * ~32KB arena = ~2MB is reasonable initial overhead
        assert!(mem < 10_000_000, "Initial memory should be reasonable: {} bytes", mem);
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
        
        // Each tree i should have effective_history_len = 3 + updates seen
        assert_eq!(fac.trees[0].effective_history_len, 4); // saw 1 update
        assert_eq!(fac.trees[1].effective_history_len, 5); // saw 2 updates
        assert_eq!(fac.trees[2].effective_history_len, 5); // got extended
        assert_eq!(fac.trees[3].effective_history_len, 5); // got extended
        
        // Revert
        fac.revert(1);
        assert_eq!(fac.shared_history.len(), 4);
        assert_eq!(fac.trees[1].effective_history_len, 4);
        
        fac.revert(0);
        assert_eq!(fac.shared_history.len(), 3);
        assert_eq!(fac.trees[0].effective_history_len, 3);
    }
}
