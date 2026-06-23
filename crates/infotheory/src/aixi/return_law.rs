//! Shared return-label law evaluation for AIQI-style controllers.
//!
//! AIQI and warm-start controllers differ in how labels are produced and decoded,
//! but both perform the same action-selection subproblem: condition a binary
//! predictor on an action, evaluate an autoregressive distribution over
//! finite return labels, normalize valid label mass, and take an expectation
//! under a controller-specific decoder.

use crate::aixi::common::bits_for_cardinality;
use crate::aixi::model::Predictor;

const PROBABILITY_FLOOR: f64 = 1e-12;

/// Bit order used only for return-label codewords.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReturnLabelBitOrder {
    /// Most-significant bit first, so shallow trie prefixes select contiguous
    /// ordered value intervals.
    MsbFirst,
    /// Least-significant bit first. Kept for tests and comparisons against the
    /// former implementation; actions, observations, and rewards still use
    /// their existing LSB-first field encoders outside this module.
    #[cfg(test)]
    LsbFirst,
}

/// Fixed-width binary code for finite return labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReturnLabelCodec {
    bins: usize,
    bits: usize,
    order: ReturnLabelBitOrder,
}

impl ReturnLabelCodec {
    /// Build the value-monotone canonical return-label encoding.
    ///
    /// Labels are ordered by decoded value by the caller's semantic contract.
    /// With MSB-first natural binary codewords, every prefix denotes a
    /// contiguous interval of label indices, modulo the final invalid tail for
    /// non-power-of-two alphabets.
    pub(crate) fn value_monotone(bins: usize) -> Self {
        assert!(bins > 0, "return-label alphabet must be non-empty");
        Self {
            bins,
            bits: bits_for_cardinality(bins),
            order: ReturnLabelBitOrder::MsbFirst,
        }
    }

    /// Build the old LSB-first return-label encoding for regression checks.
    #[cfg(test)]
    pub(crate) fn lsb_first_for_test(bins: usize) -> Self {
        assert!(bins > 0, "return-label alphabet must be non-empty");
        Self {
            bins,
            bits: bits_for_cardinality(bins),
            order: ReturnLabelBitOrder::LsbFirst,
        }
    }

    /// Number of semantic return labels.
    pub(crate) fn bins(self) -> usize {
        self.bins
    }

    /// Fixed codeword width.
    pub(crate) fn bits(self) -> usize {
        self.bits
    }

    /// Encoding order.
    #[cfg(test)]
    pub(crate) fn order(self) -> ReturnLabelBitOrder {
        self.order
    }

    /// Return the encoded bit at stream depth `depth`.
    pub(crate) fn bit_at(self, label: u64, depth: usize) -> bool {
        debug_assert!(depth < self.bits);
        let shift = match self.order {
            ReturnLabelBitOrder::MsbFirst => self.bits - depth - 1,
            #[cfg(test)]
            ReturnLabelBitOrder::LsbFirst => depth,
        };
        ((label >> shift) & 1) == 1
    }

    /// Update a partially decoded label value with the next stream bit.
    fn append_to_partial_value(self, partial: u64, depth: usize, bit: bool) -> u64 {
        let shift = match self.order {
            ReturnLabelBitOrder::MsbFirst => self.bits - depth - 1,
            #[cfg(test)]
            ReturnLabelBitOrder::LsbFirst => depth,
        };
        if bit {
            partial | (1u64 << shift)
        } else {
            partial
        }
    }

    /// Return the valid label range under a trie prefix.
    ///
    /// The range is computed from the actual encoding relation, so it is valid
    /// for both MSB-first and LSB-first comparison experiments.
    #[cfg(test)]
    pub(crate) fn label_range_for_prefix(
        self,
        prefix_value: u64,
        depth: usize,
    ) -> Option<(u64, u64)> {
        debug_assert!(depth <= self.bits);
        if self.order == ReturnLabelBitOrder::MsbFirst {
            return self.msb_first_label_range_for_prefix(prefix_value, depth);
        }
        let mut min_label: Option<u64> = None;
        let mut max_label: Option<u64> = None;
        for label in 0..self.bins as u64 {
            if self.label_matches_prefix(label, prefix_value, depth) {
                min_label = Some(min_label.map_or(label, |current| current.min(label)));
                max_label = Some(max_label.map_or(label, |current| current.max(label)));
            }
        }
        min_label.zip(max_label)
    }

    #[cfg(test)]
    fn msb_first_label_range_for_prefix(
        self,
        prefix_value: u64,
        depth: usize,
    ) -> Option<(u64, u64)> {
        if depth > self.bits || self.bins == 0 {
            return None;
        }
        // `prefix_value` is indexed by trie depth; translate it into the
        // corresponding high-bit numeric interval for the MSB-first code.
        let mut base = 0u64;
        for idx in 0..depth {
            if ((prefix_value >> idx) & 1) == 1 {
                base |= 1u64 << (self.bits - idx - 1);
            }
        }
        let remaining_bits = self.bits.saturating_sub(depth);
        let suffix_mask = low_bits_mask(remaining_bits);
        let max_codeword = base | suffix_mask;
        let last_valid = (self.bins as u64).saturating_sub(1);
        if base > last_valid {
            None
        } else {
            Some((base, max_codeword.min(last_valid)))
        }
    }

    #[cfg(test)]
    fn label_matches_prefix(self, label: u64, prefix_value: u64, depth: usize) -> bool {
        for idx in 0..depth {
            if self.bit_at(label, idx) != (((prefix_value >> idx) & 1) == 1) {
                return false;
            }
        }
        true
    }

    /// Commit a label to the predictor's learned stream.
    pub(crate) fn push_label_commit(self, predictor: &mut dyn Predictor, label: u64) -> usize {
        for depth in 0..self.bits {
            predictor.commit_update(self.bit_at(label, depth));
        }
        self.bits
    }

    /// Add a label to transient conditioning history without committing it.
    pub(crate) fn push_label_history(self, predictor: &mut dyn Predictor, label: u64) -> usize {
        for depth in 0..self.bits {
            predictor.update_history(self.bit_at(label, depth));
        }
        self.bits
    }
}

/// How hypothetical return-prefix bits are applied while evaluating a label law.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReturnPrefixUpdate {
    /// Update the predictor exactly as if the hypothetical return bits were
    /// observed in the model stream, then roll the state back.
    Training,
    /// Update only transient conditioning history, then pop that history.
    #[cfg(test)]
    #[allow(dead_code)]
    FrozenHistory,
}

/// Complete exact evaluator variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReturnLawEvaluator {
    /// Former exhaustive codeword loop; retained as a test/reference comparator.
    #[cfg(test)]
    LeafByLeaf,
    /// Memoized shared-prefix trie evaluator, querying each reached internal
    /// prefix once.
    SharedPrefix,
}

/// Operational counters for return-label law evaluation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReturnLawEvalStats {
    /// Number of logical next-bit probability queries.
    pub logical_queries: usize,
    /// Number of hypothetical label-prefix bit advances.
    pub hypothetical_advances: usize,
    /// Number of per-symbol rollbacks/pops.
    pub rollbacks: usize,
    /// Number of valid label leaves whose mass was accumulated.
    pub valid_leaves: usize,
    /// Number of invalid code leaves skipped before normalization.
    pub invalid_leaves: usize,
}

/// Normalized return-label law plus evaluation counters.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReturnLawDistribution {
    /// Probabilities in semantic label-index order.
    pub probabilities: Vec<f64>,
    /// Whether zero or non-finite mass forced the uniform fallback.
    pub used_uniform_fallback: bool,
    /// Evaluation counters.
    pub stats: ReturnLawEvalStats,
}

/// Predict and normalize the return-label distribution under `predictor`.
pub(crate) fn predict_return_law(
    predictor: &mut dyn Predictor,
    codec: ReturnLabelCodec,
    prefix_update: ReturnPrefixUpdate,
    evaluator: ReturnLawEvaluator,
) -> ReturnLawDistribution {
    if codec.bins() == 1 {
        return ReturnLawDistribution {
            probabilities: vec![1.0],
            used_uniform_fallback: false,
            stats: ReturnLawEvalStats {
                valid_leaves: 1,
                ..ReturnLawEvalStats::default()
            },
        };
    }

    let mut masses = vec![0.0; codec.bins()];
    let mut stats = ReturnLawEvalStats::default();
    match evaluator {
        #[cfg(test)]
        ReturnLawEvaluator::LeafByLeaf => {
            predict_leaf_by_leaf(predictor, codec, prefix_update, &mut masses, &mut stats);
        }
        ReturnLawEvaluator::SharedPrefix => {
            predict_shared_prefix(
                predictor,
                codec,
                prefix_update,
                0,
                0,
                1.0,
                &mut masses,
                &mut stats,
            );
        }
    }
    let used_uniform_fallback = normalize_masses(&mut masses);
    ReturnLawDistribution {
        probabilities: masses,
        used_uniform_fallback,
        stats,
    }
}

#[cfg(test)]
fn low_bits_mask(bits: usize) -> u64 {
    if bits >= u64::BITS as usize {
        u64::MAX
    } else if bits == 0 {
        0
    } else {
        (1u64 << bits) - 1
    }
}

/// Compute an expectation from a normalized label law and semantic decoder.
pub(crate) fn expected_decoded_return(
    distribution: &[f64],
    mut decode_label: impl FnMut(u64) -> f64,
) -> f64 {
    distribution
        .iter()
        .enumerate()
        .map(|(label, probability)| decode_label(label as u64) * probability)
        .sum()
}

#[cfg(test)]
fn predict_leaf_by_leaf(
    predictor: &mut dyn Predictor,
    codec: ReturnLabelCodec,
    prefix_update: ReturnPrefixUpdate,
    masses: &mut [f64],
    stats: &mut ReturnLawEvalStats,
) {
    for (label, slot) in masses.iter_mut().enumerate() {
        let mut mass = 1.0f64;
        for depth in 0..codec.bits() {
            let bit = codec.bit_at(label as u64, depth);
            let q = predictor
                .predict_prob(bit)
                .clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR);
            stats.logical_queries = stats.logical_queries.saturating_add(1);
            mass *= q;
            apply_hypothetical_bit(predictor, bit, prefix_update, stats);
        }
        for _ in 0..codec.bits() {
            revert_hypothetical_bit(predictor, prefix_update, stats);
        }
        stats.valid_leaves = stats.valid_leaves.saturating_add(1);
        *slot = mass;
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_shared_prefix(
    predictor: &mut dyn Predictor,
    codec: ReturnLabelCodec,
    prefix_update: ReturnPrefixUpdate,
    depth: usize,
    partial_value: u64,
    prefix_mass: f64,
    masses: &mut [f64],
    stats: &mut ReturnLawEvalStats,
) {
    if depth == codec.bits() {
        if let Some(slot) = masses.get_mut(partial_value as usize) {
            *slot = prefix_mass;
            stats.valid_leaves = stats.valid_leaves.saturating_add(1);
        } else {
            stats.invalid_leaves = stats.invalid_leaves.saturating_add(1);
        }
        return;
    }

    let p_one = predictor
        .predict_one()
        .clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR);
    stats.logical_queries = stats.logical_queries.saturating_add(1);

    for (bit, child_mass) in [
        (false, prefix_mass * (1.0 - p_one)),
        (true, prefix_mass * p_one),
    ] {
        let child_value = codec.append_to_partial_value(partial_value, depth, bit);
        apply_hypothetical_bit(predictor, bit, prefix_update, stats);
        predict_shared_prefix(
            predictor,
            codec,
            prefix_update,
            depth + 1,
            child_value,
            child_mass,
            masses,
            stats,
        );
        revert_hypothetical_bit(predictor, prefix_update, stats);
    }
}

fn apply_hypothetical_bit(
    predictor: &mut dyn Predictor,
    bit: bool,
    prefix_update: ReturnPrefixUpdate,
    stats: &mut ReturnLawEvalStats,
) {
    match prefix_update {
        ReturnPrefixUpdate::Training => predictor.update(bit),
        #[cfg(test)]
        ReturnPrefixUpdate::FrozenHistory => predictor.update_history(bit),
    }
    stats.hypothetical_advances = stats.hypothetical_advances.saturating_add(1);
}

fn revert_hypothetical_bit(
    predictor: &mut dyn Predictor,
    prefix_update: ReturnPrefixUpdate,
    stats: &mut ReturnLawEvalStats,
) {
    match prefix_update {
        ReturnPrefixUpdate::Training => predictor.revert(),
        #[cfg(test)]
        ReturnPrefixUpdate::FrozenHistory => predictor.pop_history(),
    }
    stats.rollbacks = stats.rollbacks.saturating_add(1);
}

fn normalize_masses(masses: &mut [f64]) -> bool {
    let sum: f64 = masses.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        let uniform = 1.0 / masses.len() as f64;
        masses.fill(uniform);
        return true;
    }
    for mass in masses {
        *mass /= sum;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct UniformPredictor {
        history: Vec<bool>,
    }

    impl Predictor for UniformPredictor {
        fn update(&mut self, sym: bool) {
            self.history.push(sym);
        }

        fn update_history(&mut self, sym: bool) {
            self.history.push(sym);
        }

        fn revert(&mut self) {
            self.history.pop();
        }

        fn pop_history(&mut self) {
            self.history.pop();
        }

        fn predict_prob(&mut self, _sym: bool) -> f64 {
            0.5
        }

        fn model_name(&self) -> String {
            "uniform-test".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn value_monotone_prefixes_are_contiguous_for_power_of_two_labels() {
        let codec = ReturnLabelCodec::value_monotone(8);
        assert_eq!(codec.order(), ReturnLabelBitOrder::MsbFirst);
        assert_eq!(codec.label_range_for_prefix(0, 1), Some((0, 3)));
        assert_eq!(codec.label_range_for_prefix(1, 1), Some((4, 7)));
        assert_eq!(codec.label_range_for_prefix(0, 2), Some((0, 1)));
        assert_eq!(codec.label_range_for_prefix(2, 2), Some((2, 3)));
    }

    #[test]
    fn value_monotone_prefixes_stop_before_invalid_tail_for_non_power_of_two_labels() {
        let codec = ReturnLabelCodec::value_monotone(6);
        assert_eq!(codec.label_range_for_prefix(0, 1), Some((0, 3)));
        assert_eq!(codec.label_range_for_prefix(1, 1), Some((4, 5)));
        assert_eq!(codec.label_range_for_prefix(1, 2), Some((4, 5)));
        assert_eq!(codec.label_range_for_prefix(3, 2), None);
    }

    #[test]
    fn lsb_first_prefixes_interleave_values() {
        let codec = ReturnLabelCodec::lsb_first_for_test(8);
        assert_eq!(codec.label_range_for_prefix(0, 1), Some((0, 6)));
        assert_eq!(codec.label_range_for_prefix(1, 1), Some((1, 7)));
    }

    #[test]
    fn shared_prefix_matches_leaf_by_leaf_uniform_law_with_fewer_queries() {
        let codec = ReturnLabelCodec::value_monotone(8);
        let mut leaf_predictor = UniformPredictor::default();
        let mut shared_predictor = UniformPredictor::default();
        let leaf = predict_return_law(
            &mut leaf_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::LeafByLeaf,
        );
        let shared = predict_return_law(
            &mut shared_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );
        assert_eq!(leaf.probabilities, shared.probabilities);
        assert_eq!(leaf.stats.logical_queries, 24);
        assert_eq!(shared.stats.logical_queries, 7);
        assert_eq!(shared.stats.valid_leaves, 8);
    }
}
