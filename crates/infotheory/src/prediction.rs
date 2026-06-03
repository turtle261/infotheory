//! Shared online prediction abstractions for byte and bit consumers.
//!
//! The crate keeps byte prediction first-class while also exposing a bit-native
//! layer for consumers whose natural symbol is a bit.  A byte model can be
//! queried as a bit model through a live prefix-mass view: the model remains a
//! byte model, but each bit query renormalizes over the surviving byte prefix.
//!
//! # Predictor Contract
//!
//! All `RateBackendPredictor` (and wrapper) implementations **must emit only
//! finite, non-negative** probabilities and log-probabilities. Callers (including
//! `BinaryPrediction` constructors, `BytePrefixMass`, mixture bit paths, and
//! entropy coders) may rely on this. Non-finite or negative outputs from a
//! predictor indicate an internal bug and are treated as contract violations
//! (surfaced via `panic!` with rich context in both debug and release builds for
//! `BinaryPrediction` constructors and the `binary_prediction_from_*` helpers).
//! Legitimate 0.5 / uniform policies remain only in the documented mathematical
//! cases below (measure-zero conditioning limits, not arithmetic corruption).
//!
//! Legitimate 0.5 / uniform policies (distinct from error masking):
//! - `BytePrefixMass::prediction` returns exact 0.5 when the current subtree
//!   mass is zero (conditioning on a measure-zero event under the byte model;
//!   the joint sequence prob is already zero; prevents NaN in coders).
//! - `from_raw_weights` (and thus public `from_pdf`/`from_log_probs`/`from_cdf`)
//!   falls back to uniform 1/256 when the *input row* has zero or non-finite
//!   total mass after sanitizing (construction-time robustness for invalid
//!   caller-provided PDFs; `from_log_probs` documents the all-invalid case).

/// Online byte-level predictor trait re-exported for prediction-oriented APIs.
pub use crate::mixture::OnlineBytePredictor;

/// Bit ordering used when factorizing byte symbols.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum BitOrder {
    /// Most-significant bit first, matching Infotheory's AC bitwise fast path.
    #[default]
    MsbFirst,
    /// Least-significant bit first.
    LsbFirst,
}

/// Semantic interpretation of a bit stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BitStreamSemantics {
    /// Bits are the fixed-width representation of byte symbols.
    ///
    /// This is a byte-native view: streams must begin and end on whole-byte
    /// boundaries, and any provided total bit count must therefore be a
    /// multiple of `8`.
    BytePacked {
        /// Bit ordering used when factorizing each byte symbol.
        order: BitOrder,
    },
    /// Bits are the actual modeled symbols, not a byte factorization.
    ///
    /// Native bit backends consume these symbols directly. Byte-native
    /// backends instead adapt this view to the literal byte symbols `0` and
    /// `1`, with probabilities renormalized over just those two outcomes.
    ///
    /// This therefore supports arbitrary non-multiple-of-8 lengths while
    /// preserving truthful capability metadata: native bit support remains
    /// distinguishable from byte-symbol adaptation.
    BinaryTokens,
}

impl Default for BitStreamSemantics {
    fn default() -> Self {
        // Generic bit sessions default to a byte-native view. Planner configs
        // use their own binary-token default because AIXI/AIQI interfaces are
        // commonly not byte-aligned.
        Self::BytePacked {
            order: BitOrder::MsbFirst,
        }
    }
}

/// Probability pair for the next binary symbol.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryPrediction {
    /// Probability of observing `0`.
    pub p0: f64,
    /// Probability of observing `1`.
    pub p1: f64,
}

impl BinaryPrediction {
    /// Construct a normalized binary prediction from `P(1)` with a numerical floor.
    pub fn from_prob_one(p1: f64, floor: f64) -> Self {
        let floor = binary_floor(floor);
        let p1 = if p1.is_finite() {
            p1
        } else {
            panic!(
                "RateBackendPredictor emitted non-finite p1 to BinaryPrediction::from_prob_one; \
                 this is now a hard contract violation (predictors must emit only finite \
                 non-negative values). See prediction.rs module docs and BinaryPrediction ctors."
            )
        };
        let p1 = p1.clamp(floor, 1.0 - floor);
        Self { p0: 1.0 - p1, p1 }
    }

    /// Construct an exact normalized binary prediction from `P(1)`.
    ///
    /// This preserves hard support semantics: exact `0` and `1` probabilities
    /// remain exact. Entropy coders should apply their own finite-count floor at
    /// the coding boundary rather than here.
    pub fn from_prob_one_exact(p1: f64) -> Self {
        let p1 = if p1.is_finite() {
            p1
        } else {
            panic!(
                "RateBackendPredictor emitted non-finite p1 to BinaryPrediction::from_prob_one_exact; \
                 this is now a hard contract violation (predictors must emit only finite \
                 non-negative values). See prediction.rs module docs and BinaryPrediction ctors."
            )
        };
        let p1 = p1.clamp(0.0, 1.0);
        Self { p0: 1.0 - p1, p1 }
    }

    /// Probability of `bit`.
    #[inline]
    pub fn prob(self, bit: bool) -> f64 {
        if bit { self.p1 } else { self.p0 }
    }
}

/// Minimal bit predictor interface for true binary consumers.
pub trait OnlineBitPredictor {
    /// Optional stream-start hook.
    fn begin_bit_stream(
        &mut self,
        _total_bits: Option<u64>,
        _semantics: BitStreamSemantics,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Optional stream-finalization hook.
    fn finish_bit_stream(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Predict the next bit without updating state.
    fn bit_prediction(&mut self) -> BinaryPrediction;

    /// Observe a bit while fitting/adapting.
    fn update_bit(&mut self, bit: bool);

    /// Observe a bit as conditioning only.
    fn update_bit_frozen(&mut self, bit: bool) {
        self.update_bit(bit);
    }
}

/// Live byte-prefix state used to query a 256-way byte PDF as conditional bits.
#[derive(Clone, Debug)]
pub struct BytePrefixMass {
    tree: [f64; 512],
    node: usize,
    order: BitOrder,
    bits_seen: u8,
    symbol: u8,
}

const BYTE_PREFIX_TREE_ROOT: usize = 1;
const BYTE_PREFIX_TREE_LEAF_BASE: usize = 256;

impl BytePrefixMass {
    /// Build a prefix-mass state from a byte PDF.
    ///
    /// Slices shorter than 256 are treated as zero-padded on the right
    /// (i.e. missing entries contribute 0 mass). This is an explicit
    /// construction-time contract for the public API (graceful handling of
    /// partial rows); see also the private `from_raw_weights` and module-level
    /// docs on legitimate uniform fallbacks.
    pub fn from_pdf(pdf: &[f64], order: BitOrder) -> Self {
        let mut weights = [0.0f64; 256];
        for (idx, weight) in weights.iter_mut().enumerate() {
            *weight = pdf.get(idx).copied().unwrap_or(0.0);
        }
        Self::from_raw_weights(weights, order)
    }

    /// Build a prefix-mass state from byte log-probabilities or log-weights.
    ///
    /// Finite entries are exponentiated after subtracting the maximum finite
    /// entry for numerical stability. Non-finite entries are treated as zero
    /// mass, and an all-invalid row therefore falls back to the same uniform
    /// distribution as [`Self::from_pdf`].
    ///
    /// The input slice is truncated to at most 256 entries (excess ignored);
    /// shorter slices are zero-padded (explicit contract, see [`Self::from_pdf`]).
    pub fn from_log_probs(log_probs: &[f64], order: BitOrder) -> Self {
        let log_probs = &log_probs[..log_probs.len().min(256)];
        let max_log = log_probs
            .iter()
            .copied()
            .filter(|lp| lp.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights = [0.0f64; 256];
        if max_log.is_finite() {
            for (weight, &lp) in weights.iter_mut().zip(log_probs.iter()) {
                *weight = if lp.is_finite() {
                    (lp - max_log).exp()
                } else {
                    0.0
                };
            }
        }
        Self::from_raw_weights(weights, order)
    }

    /// Build a prefix-mass state from a normalized byte CDF row.
    pub fn from_cdf(cdf: [f64; 257], order: BitOrder) -> Self {
        let mut weights = [0.0f64; 256];
        for (idx, weight) in weights.iter_mut().enumerate() {
            *weight = cdf[idx + 1] - cdf[idx];
        }
        Self::from_raw_weights(weights, order)
    }

    fn from_raw_weights(mut weights: [f64; 256], order: BitOrder) -> Self {
        let mut total = 0.0f64;
        for weight in &mut weights {
            *weight = if weight.is_finite() && *weight > 0.0 {
                *weight
            } else {
                0.0
            };
            total += *weight;
        }
        if !total.is_finite() || total <= 0.0 {
            weights.fill(1.0 / 256.0);
        } else {
            let inv = 1.0 / total;
            for weight in &mut weights {
                *weight *= inv;
            }
        }
        Self::from_normalized_pdf(weights, order)
    }

    fn from_normalized_pdf(pdf: [f64; 256], order: BitOrder) -> Self {
        let mut tree = [0.0f64; 512];
        for (symbol, &mass) in pdf.iter().enumerate() {
            let leaf = BYTE_PREFIX_TREE_LEAF_BASE + byte_prefix_leaf_offset(order, symbol as u8);
            tree[leaf] = mass;
        }
        for node in (1..BYTE_PREFIX_TREE_LEAF_BASE).rev() {
            tree[node] = tree[node * 2] + tree[node * 2 + 1];
        }
        Self {
            tree,
            node: BYTE_PREFIX_TREE_ROOT,
            order,
            bits_seen: 0,
            symbol: 0,
        }
    }

    /// Query the current conditional probability of the next bit.
    pub fn prediction(&self) -> BinaryPrediction {
        if self.is_complete() {
            return BinaryPrediction::from_prob_one_exact(0.5);
        }
        let total = self.tree[self.node];
        if !total.is_finite() || total <= 0.0 {
            // Legitimate policy: zero mass under the byte model means we are
            // conditioning on a measure-zero event for the prefix. The joint
            // probability of the observed sequence is already 0; returning the
            // max-entropy distribution (0.5) prevents NaN propagation into
            // arithmetic coders.
            return BinaryPrediction::from_prob_one_exact(0.5);
        }
        let one = self.tree[self.node * 2 + 1];
        let p1 = one / total;
        BinaryPrediction::from_prob_one_exact(p1)
    }

    /// Observe a bit, discarding the impossible sibling branch.
    pub fn observe(&mut self, bit: bool) {
        debug_assert!(
            !self.is_complete(),
            "BytePrefixMass::observe called after a full byte was already observed"
        );
        if self.is_complete() {
            return;
        }
        self.node = self.node * 2 + usize::from(bit);
        match self.order {
            BitOrder::MsbFirst => {
                self.symbol |= u8::from(bit) << (7 - self.bits_seen);
            }
            BitOrder::LsbFirst => {
                self.symbol |= u8::from(bit) << self.bits_seen;
            }
        }
        self.bits_seen = self.bits_seen.saturating_add(1);
    }

    /// Whether a full byte has been observed.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.bits_seen >= 8
    }

    /// Whether the current byte prefix has consumed at least one bit but has
    /// not completed a full byte yet.
    #[inline]
    pub fn has_partial_bits(&self) -> bool {
        self.bits_seen > 0 && !self.is_complete()
    }

    /// Current completed symbol. Meaningful once [`Self::is_complete`] is true.
    #[inline]
    pub fn symbol(&self) -> u8 {
        self.symbol
    }
}

#[inline]
fn byte_prefix_leaf_offset(order: BitOrder, symbol: u8) -> usize {
    match order {
        BitOrder::MsbFirst => usize::from(symbol),
        BitOrder::LsbFirst => usize::from(symbol.reverse_bits()),
    }
}

#[inline]
fn binary_floor(floor: f64) -> f64 {
    if floor.is_finite() {
        floor.clamp(1e-12, 0.499_999_999_999)
    } else {
        1e-12
    }
}

/// Convert a pair of raw probabilities into a normalized [`BinaryPrediction`].
///
/// Inputs must be finite and non-negative (predictor contract). When both masses
/// are exactly zero the conditioning event has measure zero under the model; the
/// maximum-entropy extension `P(1)=0.5` is returned via [`BinaryPrediction::from_prob_one_exact`].
#[inline]
pub(crate) fn binary_prediction_from_probs(p0: f64, p1: f64, floor: f64) -> BinaryPrediction {
    assert!(
        p0.is_finite() && p0 >= 0.0 && p1.is_finite() && p1 >= 0.0,
        "RateBackendPredictor emitted invalid probability to binary_prediction_from_probs: p0={p0}, p1={p1}; \
         Predictor contract violation (must emit only finite non-negative values)"
    );
    let sum: f64 = p0 + p1;
    if sum > 0.0 {
        BinaryPrediction::from_prob_one(p1 / sum, floor)
    } else {
        // Measure-zero conditioning limit: both symbol masses are exactly zero.
        BinaryPrediction::from_prob_one_exact(0.5)
    }
}

/// Convert a pair of natural-log probabilities into a normalized [`BinaryPrediction`].
///
/// Uses a log-max shift before exponentiating to avoid catastrophic underflow
/// when both `logp0` and `logp1` are very negative (e.g. deep inside a long
/// conditioning context).  Numerically, shifting by `max(logp0, logp1)` before
/// calling `exp` keeps the dominant term at `1.0` and the ratio exact.
///
/// `floor` is forwarded to [`BinaryPrediction::from_prob_one`] and clamped to
/// `[1e-12, 0.5)` so that exact zero/one log-probs are softened at the coding
/// boundary rather than silently propagating infinities.
#[inline]
pub(crate) fn binary_prediction_from_log_probs(
    logp0: f64,
    logp1: f64,
    floor: f64,
) -> BinaryPrediction {
    if logp0.is_nan() || logp1.is_nan() {
        panic!(
            "RateBackendPredictor emitted NaN log probability to binary_prediction_from_log_probs: \
             logp0={logp0}, logp1={logp1}; contract violation"
        );
    }
    let max_log: f64 = logp0.max(logp1);
    if max_log.is_infinite() {
        if max_log == f64::NEG_INFINITY {
            // Both log-probs are exactly -inf: measure-zero conditioning limit.
            return BinaryPrediction::from_prob_one_exact(0.5);
        }
        panic!(
            "RateBackendPredictor emitted +Inf log probability to binary_prediction_from_log_probs: \
             logp0={logp0}, logp1={logp1}; contract violation"
        );
    }
    // After the guards above, each logp is finite or exactly -inf; max_log is finite.
    // IEEE 754: (-inf) - finite = -inf, and exp(-inf) = 0.0 — no explicit branch needed.
    let p0: f64 = (logp0 - max_log).exp();
    let p1: f64 = (logp1 - max_log).exp();
    binary_prediction_from_probs(p0, p1, floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_pdf_for_test(mut pdf: [f64; 256]) -> [f64; 256] {
        let sum: f64 = pdf.iter().sum();
        for p in &mut pdf {
            *p /= sum;
        }
        pdf
    }

    fn assert_binary_prediction_close(actual: BinaryPrediction, expected: BinaryPrediction) {
        assert!((actual.p0 - expected.p0).abs() < 1e-12);
        assert!((actual.p1 - expected.p1).abs() < 1e-12);
    }

    fn bit_at(symbol: u8, order: BitOrder, bit_idx: u8) -> bool {
        match order {
            BitOrder::MsbFirst => ((symbol >> (7 - bit_idx)) & 1) == 1,
            BitOrder::LsbFirst => ((symbol >> bit_idx) & 1) == 1,
        }
    }

    fn extend_observed_prefix(observed: &mut u8, order: BitOrder, bit_idx: u8, bit: bool) {
        match order {
            BitOrder::MsbFirst => *observed |= u8::from(bit) << (7 - bit_idx),
            BitOrder::LsbFirst => *observed |= u8::from(bit) << bit_idx,
        }
    }

    fn prefix_matches(value: u8, observed: u8, order: BitOrder, bits_seen: u8) -> bool {
        for bit_idx in 0..bits_seen {
            if bit_at(value, order, bit_idx) != bit_at(observed, order, bit_idx) {
                return false;
            }
        }
        true
    }

    fn direct_prediction(
        pdf: &[f64; 256],
        order: BitOrder,
        observed: u8,
        bits_seen: u8,
    ) -> BinaryPrediction {
        if bits_seen >= 8 {
            return BinaryPrediction::from_prob_one_exact(0.5);
        }
        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        for (value, &mass) in pdf.iter().enumerate() {
            let value = value as u8;
            if !prefix_matches(value, observed, order, bits_seen) {
                continue;
            }
            if bit_at(value, order, bits_seen) {
                p1 += mass;
            } else {
                p0 += mass;
            }
        }
        let total = p0 + p1;
        let p1 = if total.is_finite() && total > 0.0 {
            p1 / total
        } else {
            0.5
        };
        BinaryPrediction::from_prob_one_exact(p1)
    }

    fn cdf_from_pdf(pdf: &[f64; 256]) -> [f64; 257] {
        let mut cdf = [0.0f64; 257];
        let mut acc = 0.0f64;
        for idx in 0..256usize {
            acc += pdf[idx];
            cdf[idx + 1] = acc;
        }
        cdf
    }

    #[test]
    fn byte_prefix_product_matches_symbol_probability_msb() {
        let mut pdf = [0.0f64; 256];
        for (idx, slot) in pdf.iter_mut().enumerate() {
            *slot = (idx + 1) as f64;
        }
        let pdf = normalize_pdf_for_test(pdf);

        let symbol = 0b1010_0110u8;
        let mut prefix = BytePrefixMass::from_pdf(&pdf, BitOrder::MsbFirst);
        let mut product = 1.0f64;
        for bit_idx in 0..8u8 {
            let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
            let pred = prefix.prediction();
            product *= pred.prob(bit);
            prefix.observe(bit);
        }
        assert!(prefix.is_complete());
        assert_eq!(prefix.symbol(), symbol);
        assert!((product - pdf[symbol as usize]).abs() < 1e-12);
    }

    #[test]
    fn byte_prefix_product_matches_symbol_probability_lsb() {
        let mut pdf = [0.0f64; 256];
        for (idx, slot) in pdf.iter_mut().enumerate() {
            *slot = (idx + 3) as f64;
        }
        let pdf = normalize_pdf_for_test(pdf);

        let symbol = 0b1010_0110u8;
        let mut prefix = BytePrefixMass::from_pdf(&pdf, BitOrder::LsbFirst);
        let mut product = 1.0f64;
        for bit_idx in 0..8u8 {
            let bit = ((symbol >> bit_idx) & 1) == 1;
            let pred = prefix.prediction();
            product *= pred.prob(bit);
            prefix.observe(bit);
        }
        assert!(prefix.is_complete());
        assert_eq!(prefix.symbol(), symbol);
        assert!((product - pdf[symbol as usize]).abs() < 1e-12);
    }

    #[test]
    fn byte_prefix_preserves_zero_mass_until_coder_boundary() {
        let mut pdf = [0.0f64; 256];
        pdf[0b1010_0000] = 0.25;
        pdf[0b1010_0001] = 0.75;

        let mut prefix = BytePrefixMass::from_pdf(&pdf, BitOrder::MsbFirst);
        for bit_idx in 0..4u8 {
            let bit = ((0b1010_0000u8 >> (7 - bit_idx)) & 1) == 1;
            let prediction = prefix.prediction();
            assert_eq!(prediction.prob(bit), 1.0);
            assert_eq!(prediction.prob(!bit), 0.0);
            prefix.observe(bit);
        }

        let impossible = prefix.prediction();
        assert_eq!(impossible.p1, 0.0);
        assert_eq!(impossible.p0, 1.0);
    }

    #[test]
    fn byte_prefix_msb_prediction_preserves_tiny_positive_tail_mass() {
        let eps = 5e-17;
        let mut pdf = [eps; 256];
        pdf[0] = 1.0 - (255.0 * eps);
        let pdf = normalize_pdf_for_test(pdf);

        let prediction = BytePrefixMass::from_pdf(&pdf, BitOrder::MsbFirst).prediction();
        let expected = pdf[128..256].iter().copied().sum::<f64>();
        assert!(expected > 0.0);
        assert!(prediction.p1 > 0.0);
        assert!((prediction.p1 - expected).abs() < 1e-18);
    }

    #[test]
    fn byte_prefix_lsb_prediction_preserves_tiny_positive_tail_mass() {
        let eps = 5e-17;
        let mut pdf = [eps; 256];
        pdf[0] = 1.0 - (255.0 * eps);
        let pdf = normalize_pdf_for_test(pdf);

        let prediction = BytePrefixMass::from_pdf(&pdf, BitOrder::LsbFirst).prediction();
        let expected = pdf
            .iter()
            .enumerate()
            .filter(|(idx, _)| (idx & 1) == 1)
            .map(|(_, &p)| p)
            .sum::<f64>();
        assert!(expected > 0.0);
        assert!(prediction.p1 > 0.0);
        assert!((prediction.p1 - expected).abs() < 1e-18);
    }

    #[test]
    fn byte_prefix_lsb_preserves_zero_mass_until_coder_boundary() {
        let mut pdf = [0.0f64; 256];
        pdf[0b0000_1010] = 0.25;
        pdf[0b1000_1010] = 0.75;

        let symbol = 0b0000_1010u8;
        let mut prefix = BytePrefixMass::from_pdf(&pdf, BitOrder::LsbFirst);
        for bit_idx in 0..7u8 {
            let bit = ((symbol >> bit_idx) & 1) == 1;
            let prediction = prefix.prediction();
            assert_eq!(prediction.prob(bit), 1.0);
            assert_eq!(prediction.prob(!bit), 0.0);
            prefix.observe(bit);
        }
    }

    #[test]
    fn byte_prefix_prediction_matches_direct_reference_for_both_orders() {
        let mut pdf = [0.0f64; 256];
        for (idx, slot) in pdf.iter_mut().enumerate() {
            *slot = ((idx * 37 + 11) % 257 + 1) as f64;
        }
        let pdf = normalize_pdf_for_test(pdf);

        for order in [BitOrder::MsbFirst, BitOrder::LsbFirst] {
            for symbol in 0u8..=255u8 {
                let mut prefix = BytePrefixMass::from_pdf(&pdf, order);
                let mut observed = 0u8;
                for bit_idx in 0..8u8 {
                    let expected = direct_prediction(&pdf, order, observed, bit_idx);
                    let actual = prefix.prediction();
                    assert_binary_prediction_close(actual, expected);

                    let bit = bit_at(symbol, order, bit_idx);
                    prefix.observe(bit);
                    extend_observed_prefix(&mut observed, order, bit_idx, bit);
                }
                assert!(prefix.is_complete());
                assert_eq!(prefix.symbol(), symbol);
            }
        }
    }

    #[test]
    fn byte_prefix_from_cdf_matches_from_pdf_for_both_orders() {
        let mut pdf = [0.0f64; 256];
        for (idx, slot) in pdf.iter_mut().enumerate() {
            *slot = ((idx * 19 + 7) % 193 + 1) as f64;
        }
        let pdf = normalize_pdf_for_test(pdf);
        let cdf = cdf_from_pdf(&pdf);

        for order in [BitOrder::MsbFirst, BitOrder::LsbFirst] {
            let symbol = 0b1010_0110u8;
            let mut from_pdf = BytePrefixMass::from_pdf(&pdf, order);
            let mut from_cdf = BytePrefixMass::from_cdf(cdf, order);
            for bit_idx in 0..8u8 {
                assert_binary_prediction_close(from_pdf.prediction(), from_cdf.prediction());
                let bit = bit_at(symbol, order, bit_idx);
                from_pdf.observe(bit);
                from_cdf.observe(bit);
            }
            assert_eq!(from_pdf.symbol(), symbol);
            assert_eq!(from_cdf.symbol(), symbol);
        }
    }

    #[test]
    fn byte_prefix_from_log_probs_matches_from_pdf_for_both_orders() {
        let mut pdf = [0.0f64; 256];
        for (idx, slot) in pdf.iter_mut().enumerate() {
            *slot = ((idx * 23 + 5) % 211 + 1) as f64;
        }
        let pdf = normalize_pdf_for_test(pdf);

        let mut log_probs = [f64::NEG_INFINITY; 256];
        for (dst, &mass) in log_probs.iter_mut().zip(pdf.iter()) {
            *dst = mass.ln() + 17.0;
        }

        for order in [BitOrder::MsbFirst, BitOrder::LsbFirst] {
            let symbol = 0b1010_0110u8;
            let mut from_pdf = BytePrefixMass::from_pdf(&pdf, order);
            let mut from_log_probs = BytePrefixMass::from_log_probs(&log_probs, order);
            for bit_idx in 0..8u8 {
                assert_binary_prediction_close(from_pdf.prediction(), from_log_probs.prediction());
                let bit = bit_at(symbol, order, bit_idx);
                from_pdf.observe(bit);
                from_log_probs.observe(bit);
            }
            assert_eq!(from_pdf.symbol(), symbol);
            assert_eq!(from_log_probs.symbol(), symbol);
        }
    }

    #[test]
    fn binary_prediction_from_probs_normalizes_and_floors() {
        let pred = binary_prediction_from_probs(2.0, 6.0, 1e-6);
        assert!((pred.p0 - 0.25).abs() < 1e-12);
        assert!((pred.p1 - 0.75).abs() < 1e-12);
    }

    #[test]
    fn binary_prediction_from_probs_both_zero_returns_exact_half() {
        let pred = binary_prediction_from_probs(0.0, 0.0, 1e-6);
        assert!((pred.p1 - 0.5).abs() < 1e-12);
        assert!((pred.p0 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn binary_prediction_from_log_probs_both_neg_inf_returns_exact_half() {
        let pred = binary_prediction_from_log_probs(f64::NEG_INFINITY, f64::NEG_INFINITY, 1e-6);
        assert!((pred.p1 - 0.5).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "invalid probability")]
    fn binary_prediction_from_probs_panics_on_nan() {
        let _ = binary_prediction_from_probs(f64::NAN, 0.5, 1e-6);
    }

    #[test]
    #[should_panic(expected = "invalid probability")]
    fn binary_prediction_from_probs_panics_on_negative() {
        let _ = binary_prediction_from_probs(-1.0, 0.5, 1e-6);
    }

    #[test]
    #[should_panic(expected = "NaN log probability")]
    fn binary_prediction_from_log_probs_panics_on_nan() {
        let _ = binary_prediction_from_log_probs(f64::NAN, -1.0, 1e-6);
    }

    #[test]
    #[should_panic(expected = "+Inf log probability")]
    fn binary_prediction_from_log_probs_panics_on_pos_inf() {
        let _ = binary_prediction_from_log_probs(0.0, f64::INFINITY, 1e-6);
    }

    #[test]
    fn byte_prefix_partial_bits_reports_only_in_progress_prefixes() {
        let pdf = [1.0 / 256.0; 256];
        let mut prefix = BytePrefixMass::from_pdf(&pdf, BitOrder::MsbFirst);
        assert!(!prefix.has_partial_bits());

        prefix.observe(true);
        assert!(prefix.has_partial_bits());

        for _ in 1..8u8 {
            prefix.observe(false);
        }
        assert!(prefix.is_complete());
        assert!(!prefix.has_partial_bits());
    }
}
