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
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReturnLawDistribution {
    /// Probabilities in semantic label-index order.
    pub probabilities: Vec<f64>,
    /// Whether evaluation used the uniform-label fallback.
    pub used_uniform_fallback: bool,
    /// Evaluation counters.
    pub stats: ReturnLawEvalStats,
}

/// Exact expected decoded return plus evaluation counters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReturnLawExpectation {
    /// Expected value under the normalized valid-label law.
    pub value: f64,
    /// Whether zero or non-finite mass forced the uniform fallback.
    pub used_uniform_fallback: bool,
    /// Evaluation counters.
    pub stats: ReturnLawEvalStats,
}

/// Predict and normalize the return-label distribution under `predictor`.
///
/// With `ReturnPrefixUpdate::Training`, return-law descent deliberately uses
/// per-symbol rollback (`update`/`revert`) rather than
/// [`Predictor::begin_rollback_scope`]. Both policies preserve the same exact
/// trie semantics, but this evaluator opens one speculative branch per trie
/// edge, so scoped rollback would open one scope per edge rather than amortizing
/// a scope over a rollout. The built-in rate predictor's scopes are marker
/// based and measured cost-neutral here; per-symbol rollback remains the
/// clearer local contract for the return-law descent and matches the live
/// no-clone planning path.
///
/// Cost model: this is a depth-first descent that mutates and rolls back the
/// predictor once per reached trie edge, i.e. `hypothetical_advances` undo
/// operations total. Distribution and scalar-expectation callers share the
/// same descent engine and differ only in how valid leaves are accumulated.
/// Leaf masses are accumulated directly in `f64`; current callers keep return
/// label widths small enough that valid mass does not exhaust linear floating
/// point range before the explicit zero/non-finite-mass fallback applies.
///
/// This intentionally differs from MCTS rollout simulation, which can use
/// scoped predictor rollback because it opens one scope per rollout rather than
/// one scope per return-label trie edge.
#[cfg(test)]
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
    let stats = {
        let mut descent = ReturnLawDescent::new(predictor, codec, prefix_update);
        let mut sink = DistributionSink {
            masses: &mut masses,
        };
        descent.evaluate::<false>(evaluator, &mut sink);
        descent.stats
    };
    let used_uniform_fallback = normalize_masses(&mut masses);
    ReturnLawDistribution {
        probabilities: masses,
        used_uniform_fallback,
        stats,
    }
}

/// Predict the normalized expected decoded return without materializing a law vector.
///
/// For finite decoder values and finite nonzero valid mass, this is
/// mathematically equivalent to materializing the normalized label law and then
/// taking its dot product with `decode_label`, modulo floating-point
/// reassociation. The normal scalar path uses direct linear accumulation, which
/// is the intended regime for current finite-return alphabets. If any valid
/// leaf's linear mass underflows to exactly zero (only reachable for
/// pathologically deep return-label alphabets, and subsuming the degenerate
/// total-mass-zero case), the evaluator reruns the same balanced descent with
/// log-space accumulation so those negligible-but-nonzero leaves do not collapse
/// into a uniform midpoint fallback.
///
/// If the decoder itself produces a non-finite value, this scalar path uses the
/// uniform-label fallback shape, because a non-finite decoded expectation
/// cannot be repaired by label-law normalization alone.
///
/// Return-law descent deliberately uses balanced per-symbol rollback rather
/// than the scoped-simulation strategy used by MCTS rollouts: this evaluator
/// opens one speculative branch per return-label trie edge, so scoped rollback
/// would add per-edge scope management without reducing the number of
/// speculative updates.
pub(crate) fn predict_expected_return(
    predictor: &mut dyn Predictor,
    codec: ReturnLabelCodec,
    prefix_update: ReturnPrefixUpdate,
    evaluator: ReturnLawEvaluator,
    mut decode_label: impl FnMut(u64) -> f64,
) -> ReturnLawExpectation {
    if codec.bins() == 1 {
        return ReturnLawExpectation {
            value: decode_label(0),
            used_uniform_fallback: false,
            stats: ReturnLawEvalStats {
                valid_leaves: 1,
                ..ReturnLawEvalStats::default()
            },
        };
    }

    let mut sink = LinearExpectedReturnSink {
        decode_label: &mut decode_label,
        weighted_sum: 0.0,
        valid_mass: 0.0,
        saw_zero_mass: false,
        saw_non_finite_decode: false,
    };
    let stats = {
        let mut descent = ReturnLawDescent::new(predictor, codec, prefix_update);
        descent.evaluate::<false>(evaluator, &mut sink);
        descent.stats
    };

    if sink.valid_mass.is_finite()
        && sink.valid_mass > 0.0
        && sink.weighted_sum.is_finite()
        && !sink.saw_zero_mass
        && !sink.saw_non_finite_decode
    {
        return ReturnLawExpectation {
            value: sink.weighted_sum / sink.valid_mass,
            used_uniform_fallback: false,
            stats,
        };
    }

    // `valid_mass == 0.0` (every leaf underflowed) implies `saw_zero_mass`, so
    // the per-leaf flag alone gates the log-space rerun.
    if sink.saw_zero_mass && !sink.saw_non_finite_decode && sink.weighted_sum.is_finite() {
        return predict_expected_return_log_fallback(
            predictor,
            codec,
            prefix_update,
            evaluator,
            &mut decode_label,
        );
    }

    uniform_expectation(codec, stats, &mut decode_label)
}

fn predict_expected_return_log_fallback<F>(
    predictor: &mut dyn Predictor,
    codec: ReturnLabelCodec,
    prefix_update: ReturnPrefixUpdate,
    evaluator: ReturnLawEvaluator,
    decode_label: &mut F,
) -> ReturnLawExpectation
where
    F: FnMut(u64) -> f64,
{
    let mut sink = LogExpectedReturnSink {
        decode_label,
        log_valid_mass: None,
        log_positive_weighted_mass: None,
        log_negative_weighted_mass: None,
        saw_non_finite_decode: false,
    };
    let stats = {
        let mut descent = ReturnLawDescent::new(predictor, codec, prefix_update);
        descent.evaluate::<true>(evaluator, &mut sink);
        descent.stats
    };

    let Some(log_valid_mass) = sink.log_valid_mass else {
        return uniform_expectation(codec, stats, sink.decode_label);
    };

    if !log_valid_mass.is_finite() || sink.saw_non_finite_decode {
        return uniform_expectation(codec, stats, sink.decode_label);
    }

    let positive = sink
        .log_positive_weighted_mass
        .map_or(0.0, |log_sum| (log_sum - log_valid_mass).exp());
    let negative = sink
        .log_negative_weighted_mass
        .map_or(0.0, |log_sum| (log_sum - log_valid_mass).exp());

    ReturnLawExpectation {
        value: positive - negative,
        used_uniform_fallback: false,
        stats,
    }
}

fn uniform_expectation(
    codec: ReturnLabelCodec,
    stats: ReturnLawEvalStats,
    decode_label: &mut impl FnMut(u64) -> f64,
) -> ReturnLawExpectation {
    let uniform_sum: f64 = (0..codec.bins())
        .map(|label| decode_label(label as u64))
        .sum();
    ReturnLawExpectation {
        value: uniform_sum / codec.bins() as f64,
        used_uniform_fallback: true,
        stats,
    }
}

/// Predict the normalized expected semantic label without materializing a law vector.
///
/// This is the same trie evaluation as [`predict_expected_return`], specialized
/// for affine decoders of the form `offset + label * scale`. It avoids a
/// per-leaf affine evaluation by accumulating raw labels and letting callers
/// apply `offset + label * scale` once per action; the descent still pays the
/// trivial identity cast at each valid leaf.
pub(crate) fn predict_expected_label(
    predictor: &mut dyn Predictor,
    codec: ReturnLabelCodec,
    prefix_update: ReturnPrefixUpdate,
    evaluator: ReturnLawEvaluator,
) -> f64 {
    let expectation =
        predict_expected_return(predictor, codec, prefix_update, evaluator, |label| {
            label as f64
        });
    expectation.value
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
#[cfg(test)]
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

struct ReturnLawDescent<'a> {
    predictor: &'a mut dyn Predictor,
    codec: ReturnLabelCodec,
    prefix_update: ReturnPrefixUpdate,
    stats: ReturnLawEvalStats,
}

impl<'a> ReturnLawDescent<'a> {
    fn new(
        predictor: &'a mut dyn Predictor,
        codec: ReturnLabelCodec,
        prefix_update: ReturnPrefixUpdate,
    ) -> Self {
        Self {
            predictor,
            codec,
            prefix_update,
            stats: ReturnLawEvalStats::default(),
        }
    }

    fn evaluate<const TRACK_LOG: bool>(
        &mut self,
        evaluator: ReturnLawEvaluator,
        sink: &mut impl ReturnLawLeafSink,
    ) {
        match evaluator {
            #[cfg(test)]
            ReturnLawEvaluator::LeafByLeaf => self.descend_leaf_by_leaf::<TRACK_LOG>(sink),
            ReturnLawEvaluator::SharedPrefix => {
                self.descend_shared_prefix::<TRACK_LOG>(0, 0, 1.0, 0.0, sink);
            }
        }
    }

    #[cfg(test)]
    fn descend_leaf_by_leaf<const TRACK_LOG: bool>(&mut self, sink: &mut impl ReturnLawLeafSink) {
        for label in 0..self.codec.bins() {
            let mut mass = 1.0f64;
            let mut log_mass = 0.0f64;
            for depth in 0..self.codec.bits() {
                let bit = self.codec.bit_at(label as u64, depth);
                let q = self
                    .predictor
                    .predict_prob(bit)
                    .clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR);
                self.stats.logical_queries = self.stats.logical_queries.saturating_add(1);
                mass *= q;
                if TRACK_LOG {
                    log_mass += q.ln();
                }
                self.apply_hypothetical_bit(bit);
            }
            for _ in 0..self.codec.bits() {
                self.revert_hypothetical_bit();
            }
            sink.valid_leaf(label as u64, mass, log_mass, &mut self.stats);
        }
    }

    fn descend_shared_prefix<const TRACK_LOG: bool>(
        &mut self,
        depth: usize,
        partial_value: u64,
        prefix_mass: f64,
        prefix_log_mass: f64,
        sink: &mut impl ReturnLawLeafSink,
    ) {
        if depth == self.codec.bits() {
            if partial_value < self.codec.bins() as u64 {
                sink.valid_leaf(partial_value, prefix_mass, prefix_log_mass, &mut self.stats);
            } else {
                sink.invalid_leaf(&mut self.stats);
            }
            return;
        }

        let p_one = self
            .predictor
            .predict_one()
            .clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR);
        self.stats.logical_queries = self.stats.logical_queries.saturating_add(1);

        let zero_value = self
            .codec
            .append_to_partial_value(partial_value, depth, false);
        self.apply_hypothetical_bit(false);
        let p_zero = 1.0 - p_one;
        let zero_log_mass = if TRACK_LOG {
            prefix_log_mass + p_zero.ln()
        } else {
            0.0
        };
        self.descend_shared_prefix::<TRACK_LOG>(
            depth + 1,
            zero_value,
            prefix_mass * p_zero,
            zero_log_mass,
            sink,
        );
        self.revert_hypothetical_bit();

        let one_value = self
            .codec
            .append_to_partial_value(partial_value, depth, true);
        self.apply_hypothetical_bit(true);
        let one_log_mass = if TRACK_LOG {
            prefix_log_mass + p_one.ln()
        } else {
            0.0
        };
        self.descend_shared_prefix::<TRACK_LOG>(
            depth + 1,
            one_value,
            prefix_mass * p_one,
            one_log_mass,
            sink,
        );
        self.revert_hypothetical_bit();
    }

    fn apply_hypothetical_bit(&mut self, bit: bool) {
        match self.prefix_update {
            ReturnPrefixUpdate::Training => self.predictor.update(bit),
            #[cfg(test)]
            ReturnPrefixUpdate::FrozenHistory => self.predictor.update_history(bit),
        }
        self.stats.hypothetical_advances = self.stats.hypothetical_advances.saturating_add(1);
    }

    fn revert_hypothetical_bit(&mut self) {
        match self.prefix_update {
            ReturnPrefixUpdate::Training => self.predictor.revert(),
            #[cfg(test)]
            ReturnPrefixUpdate::FrozenHistory => self.predictor.pop_history(),
        }
        self.stats.rollbacks = self.stats.rollbacks.saturating_add(1);
    }
}

trait ReturnLawLeafSink {
    fn valid_leaf(&mut self, label: u64, mass: f64, log_mass: f64, stats: &mut ReturnLawEvalStats);

    fn invalid_leaf(&mut self, stats: &mut ReturnLawEvalStats) {
        stats.invalid_leaves = stats.invalid_leaves.saturating_add(1);
    }
}

#[cfg(test)]
struct DistributionSink<'a> {
    masses: &'a mut [f64],
}

#[cfg(test)]
impl ReturnLawLeafSink for DistributionSink<'_> {
    fn valid_leaf(
        &mut self,
        label: u64,
        mass: f64,
        _log_mass: f64,
        stats: &mut ReturnLawEvalStats,
    ) {
        // Defensive for test comparators: shared-prefix descent filters invalid
        // codeword tails before calling `valid_leaf`, while leaf-by-leaf always
        // iterates valid labels directly.
        if let Some(slot) = self.masses.get_mut(label as usize) {
            *slot = mass;
            stats.valid_leaves = stats.valid_leaves.saturating_add(1);
        } else {
            self.invalid_leaf(stats);
        }
    }
}

struct LinearExpectedReturnSink<'a, F>
where
    F: FnMut(u64) -> f64,
{
    decode_label: &'a mut F,
    weighted_sum: f64,
    valid_mass: f64,
    saw_zero_mass: bool,
    saw_non_finite_decode: bool,
}

impl<F> ReturnLawLeafSink for LinearExpectedReturnSink<'_, F>
where
    F: FnMut(u64) -> f64,
{
    fn valid_leaf(
        &mut self,
        label: u64,
        mass: f64,
        _log_mass: f64,
        stats: &mut ReturnLawEvalStats,
    ) {
        if mass == 0.0 {
            self.saw_zero_mass = true;
        }
        self.valid_mass += mass;
        let decoded = (self.decode_label)(label);
        if decoded.is_finite() {
            self.weighted_sum += decoded * mass;
        } else {
            self.saw_non_finite_decode = true;
        }
        stats.valid_leaves = stats.valid_leaves.saturating_add(1);
    }
}

struct LogExpectedReturnSink<'a, F>
where
    F: FnMut(u64) -> f64,
{
    decode_label: &'a mut F,
    log_valid_mass: Option<f64>,
    log_positive_weighted_mass: Option<f64>,
    log_negative_weighted_mass: Option<f64>,
    saw_non_finite_decode: bool,
}

impl<F> ReturnLawLeafSink for LogExpectedReturnSink<'_, F>
where
    F: FnMut(u64) -> f64,
{
    fn valid_leaf(
        &mut self,
        label: u64,
        _mass: f64,
        log_mass: f64,
        stats: &mut ReturnLawEvalStats,
    ) {
        self.log_valid_mass = Some(log_add_exp(self.log_valid_mass, log_mass));
        let decoded = (self.decode_label)(label);
        if !decoded.is_finite() {
            self.saw_non_finite_decode = true;
        } else if decoded > 0.0 {
            self.log_positive_weighted_mass = Some(log_add_exp(
                self.log_positive_weighted_mass,
                log_mass + decoded.ln(),
            ));
        } else if decoded < 0.0 {
            self.log_negative_weighted_mass = Some(log_add_exp(
                self.log_negative_weighted_mass,
                log_mass + (-decoded).ln(),
            ));
        }
        stats.valid_leaves = stats.valid_leaves.saturating_add(1);
    }
}

fn log_add_exp(current: Option<f64>, next: f64) -> f64 {
    let Some(current) = current else {
        return next;
    };
    if current >= next {
        current + (next - current).exp().ln_1p()
    } else {
        next + (current - next).exp().ln_1p()
    }
}

#[cfg(test)]
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

    #[derive(Clone, Default)]
    struct ScopedCountingPredictor {
        history: Vec<bool>,
        scopes: Vec<Vec<bool>>,
        begin_scope_calls: usize,
        rollback_scope_calls: usize,
        update_calls: usize,
        revert_calls: usize,
    }

    impl Predictor for ScopedCountingPredictor {
        fn update(&mut self, sym: bool) {
            self.update_calls = self.update_calls.saturating_add(1);
            self.history.push(sym);
        }

        fn revert(&mut self) {
            self.revert_calls = self.revert_calls.saturating_add(1);
            self.history.pop();
        }

        fn begin_rollback_scope(&mut self) {
            self.begin_scope_calls = self.begin_scope_calls.saturating_add(1);
            self.scopes.push(self.history.clone());
        }

        fn supports_rollback_scope(&self) -> bool {
            true
        }

        fn rollback_scope(&mut self) -> bool {
            self.rollback_scope_calls = self.rollback_scope_calls.saturating_add(1);
            let Some(history) = self.scopes.pop() else {
                return false;
            };
            self.history = history;
            true
        }

        fn predict_prob(&mut self, _sym: bool) -> f64 {
            0.5
        }

        fn model_name(&self) -> String {
            "scoped-counting-test".to_string()
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

    #[test]
    fn expected_return_matches_distribution_expectation_for_power_of_two_labels() {
        let codec = ReturnLabelCodec::value_monotone(16);
        let mut distribution_predictor = UniformPredictor::default();
        let mut expectation_predictor = UniformPredictor::default();
        let distribution = predict_return_law(
            &mut distribution_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );
        let expected_from_distribution =
            expected_decoded_return(&distribution.probabilities, |label| (label * label) as f64);
        let direct = predict_expected_return(
            &mut expectation_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
            |label| (label * label) as f64,
        );

        assert!((direct.value - expected_from_distribution).abs() < 1e-14);
        assert!(!direct.used_uniform_fallback);
        assert_eq!(direct.stats, distribution.stats);
    }

    #[test]
    fn expected_return_matches_distribution_expectation_for_sparse_code_tail() {
        let codec = ReturnLabelCodec::value_monotone(6);
        let mut distribution_predictor = UniformPredictor::default();
        let mut expectation_predictor = UniformPredictor::default();
        let distribution = predict_return_law(
            &mut distribution_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );
        let expected_from_distribution =
            expected_decoded_return(&distribution.probabilities, |label| label as f64 + 0.25);
        let direct = predict_expected_return(
            &mut expectation_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
            |label| label as f64 + 0.25,
        );

        assert!((direct.value - expected_from_distribution).abs() < 1e-14);
        assert!(!direct.used_uniform_fallback);
        assert_eq!(direct.stats, distribution.stats);
        assert_eq!(direct.stats.invalid_leaves, 2);
    }

    #[test]
    fn expected_return_matches_distribution_expectation_for_signed_decoder() {
        let codec = ReturnLabelCodec::value_monotone(8);
        let mut distribution_predictor = UniformPredictor::default();
        let mut expectation_predictor = UniformPredictor::default();
        let distribution = predict_return_law(
            &mut distribution_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );
        let expected_from_distribution =
            expected_decoded_return(&distribution.probabilities, |label| label as f64 - 3.5);
        let direct = predict_expected_return(
            &mut expectation_predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
            |label| label as f64 - 3.5,
        );

        assert!((direct.value - expected_from_distribution).abs() < 1e-14);
        assert!(!direct.used_uniform_fallback);
        assert_eq!(direct.stats, distribution.stats);
    }

    #[test]
    fn log_add_exp_retains_tiny_terms_without_linear_underflow() {
        let combined = log_add_exp(Some(-1000.0), -1000.0);
        assert!((combined - (-1000.0 + std::f64::consts::LN_2)).abs() < 1e-12);
    }

    #[test]
    fn expected_return_uses_uniform_fallback_for_non_finite_decoder_sum() {
        let codec = ReturnLabelCodec::value_monotone(4);
        let mut predictor = UniformPredictor::default();
        let direct = predict_expected_return(
            &mut predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
            |label| {
                if label == 0 {
                    f64::INFINITY
                } else {
                    label as f64
                }
            },
        );

        assert!(direct.used_uniform_fallback);
        assert!(direct.value.is_infinite());
        assert!(direct.value.is_sign_positive());
        assert_eq!(direct.stats.valid_leaves, 4);
    }

    #[test]
    fn shared_prefix_uses_per_symbol_rollbacks_for_return_law_training() {
        let codec = ReturnLabelCodec::value_monotone(4);
        let mut predictor = ScopedCountingPredictor::default();
        let law = predict_return_law(
            &mut predictor,
            codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );

        assert_eq!(law.probabilities, vec![0.25; 4]);
        assert_eq!(law.stats.logical_queries, 3);
        assert_eq!(law.stats.hypothetical_advances, 6);
        assert_eq!(law.stats.rollbacks, 6);
        assert_eq!(law.stats.valid_leaves, 4);
        assert_eq!(predictor.update_calls, 6);
        assert_eq!(predictor.revert_calls, 6);
        assert_eq!(predictor.begin_scope_calls, 0);
        assert_eq!(predictor.rollback_scope_calls, 0);
        assert!(predictor.history.is_empty());
        assert!(predictor.scopes.is_empty());
    }
}
