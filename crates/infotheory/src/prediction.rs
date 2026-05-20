//! Shared online prediction abstractions for byte and bit consumers.
//!
//! The crate keeps byte prediction first-class while also exposing a bit-native
//! layer for consumers whose natural symbol is a bit.  A byte model can be
//! queried as a bit model through a live prefix-mass view: the model remains a
//! byte model, but each bit query renormalizes over the surviving byte prefix.

/// Online byte-level predictor trait re-exported for prediction-oriented APIs.
pub use crate::mixture::OnlineBytePredictor;

/// Bit ordering used when factorizing byte symbols.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BitOrder {
    /// Most-significant bit first, matching Infotheory's AC bitwise fast path.
    MsbFirst,
    /// Least-significant bit first.
    LsbFirst,
}

impl Default for BitOrder {
    fn default() -> Self {
        Self::MsbFirst
    }
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
    /// Use this when the stream is genuinely bit-native, including arbitrary
    /// non-multiple-of-8 lengths.
    BinaryTokens,
}

impl Default for BitStreamSemantics {
    fn default() -> Self {
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
        let p1 = if p1.is_finite() { p1 } else { 0.5 };
        let p1 = p1.clamp(floor, 1.0 - floor);
        Self { p0: 1.0 - p1, p1 }
    }

    /// Construct an exact normalized binary prediction from `P(1)`.
    ///
    /// This preserves hard support semantics: exact `0` and `1` probabilities
    /// remain exact. Entropy coders should apply their own finite-count floor at
    /// the coding boundary rather than here.
    pub fn from_prob_one_exact(p1: f64) -> Self {
        let p1 = if p1.is_finite() { p1 } else { 0.5 };
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
    pdf: [f64; 256],
    lo: usize,
    hi: usize,
    order: BitOrder,
    bits_seen: u8,
    symbol: u8,
}

impl BytePrefixMass {
    /// Build a prefix-mass state from a byte PDF.
    pub fn from_pdf(pdf: &[f64], order: BitOrder) -> Self {
        let mut normalized = [0.0f64; 256];
        let mut acc = 0.0f64;
        for idx in 0..256usize {
            let p = pdf.get(idx).copied().unwrap_or(0.0);
            let p = if p.is_finite() && p > 0.0 { p } else { 0.0 };
            acc += p;
            normalized[idx] = p;
        }
        if !acc.is_finite() || acc <= 0.0 {
            normalized.fill(1.0 / 256.0);
        } else {
            let inv = 1.0 / acc;
            for slot in &mut normalized {
                *slot *= inv;
            }
        }
        Self::from_normalized_pdf(normalized, order)
    }

    /// Build a prefix-mass state from a normalized byte CDF row.
    pub fn from_cdf(cdf: [f64; 257], order: BitOrder) -> Self {
        let mut pdf = [0.0f64; 256];
        for idx in 0..256usize {
            pdf[idx] = (cdf[idx + 1] - cdf[idx]).max(0.0);
        }
        Self::from_pdf(&pdf, order)
    }

    fn from_normalized_pdf(pdf: [f64; 256], order: BitOrder) -> Self {
        Self {
            pdf,
            lo: 0,
            hi: 256,
            order,
            bits_seen: 0,
            symbol: 0,
        }
    }

    /// Query the current conditional probability of the next bit.
    pub fn prediction(&self) -> BinaryPrediction {
        if self.order == BitOrder::LsbFirst {
            return self.prediction_lsb();
        }
        let (zero_lo, zero_hi, one_lo, one_hi) = self.child_ranges();
        let zero = self.pdf[zero_lo..zero_hi].iter().copied().sum::<f64>();
        let one = self.pdf[one_lo..one_hi].iter().copied().sum::<f64>();
        let total = zero + one;
        if !total.is_finite() || total <= 0.0 {
            return BinaryPrediction::from_prob_one_exact(0.5);
        }
        let p1 = one / total;
        BinaryPrediction::from_prob_one_exact(p1)
    }

    /// Observe a bit, discarding the impossible sibling branch.
    pub fn observe(&mut self, bit: bool) {
        if self.order == BitOrder::MsbFirst {
            let (zero_lo, zero_hi, one_lo, one_hi) = self.child_ranges();
            if bit {
                self.lo = one_lo;
                self.hi = one_hi;
            } else {
                self.lo = zero_lo;
                self.hi = zero_hi;
            }
        }
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

    /// Current completed symbol. Meaningful once [`Self::is_complete`] is true.
    #[inline]
    pub fn symbol(&self) -> u8 {
        self.symbol
    }

    #[inline]
    fn child_ranges(&self) -> (usize, usize, usize, usize) {
        debug_assert_eq!(self.order, BitOrder::MsbFirst);
        let mid = (self.lo + self.hi) >> 1;
        (self.lo, mid, mid, self.hi)
    }

    fn prediction_lsb(&self) -> BinaryPrediction {
        let prefix_mask = if self.bits_seen == 0 {
            0usize
        } else {
            (1usize << self.bits_seen) - 1
        };
        let prefix = (self.symbol as usize) & prefix_mask;
        let next_mask = 1usize << self.bits_seen;
        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        for value in 0..256usize {
            if (value & prefix_mask) != prefix {
                continue;
            }
            if (value & next_mask) == 0 {
                p0 += self.pdf[value];
            } else {
                p1 += self.pdf[value];
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
/// Non-finite and negative inputs are treated as zero mass. If both sides are
/// degenerate, the result falls back to `0.5 / 0.5`.
#[inline]
pub(crate) fn binary_prediction_from_probs(p0: f64, p1: f64, floor: f64) -> BinaryPrediction {
    let p0 = if p0.is_finite() && p0 > 0.0 { p0 } else { 0.0 };
    let p1 = if p1.is_finite() && p1 > 0.0 { p1 } else { 0.0 };
    let sum = p0 + p1;
    let p1_norm = if sum.is_finite() && sum > 0.0 {
        p1 / sum
    } else {
        0.5
    };
    BinaryPrediction::from_prob_one(p1_norm, floor)
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
    let max_log = if logp0 > logp1 { logp0 } else { logp1 };
    if !max_log.is_finite() {
        return BinaryPrediction::from_prob_one(0.5, floor);
    }
    let p0 = if logp0.is_finite() { (logp0 - max_log).exp() } else { 0.0 };
    let p1 = if logp1.is_finite() { (logp1 - max_log).exp() } else { 0.0 };
    binary_prediction_from_probs(p0, p1, floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_prefix_product_matches_symbol_probability_msb() {
        let mut pdf = [0.0f64; 256];
        for (idx, slot) in pdf.iter_mut().enumerate() {
            *slot = (idx + 1) as f64;
        }
        let sum: f64 = pdf.iter().sum();
        for p in &mut pdf {
            *p /= sum;
        }

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
        let sum: f64 = pdf.iter().sum();
        for p in &mut pdf {
            *p /= sum;
        }

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
        let total: f64 = pdf.iter().sum();
        for p in &mut pdf {
            *p /= total;
        }

        let prediction = BytePrefixMass::from_pdf(&pdf, BitOrder::MsbFirst).prediction();
        let expected = pdf[128..256].iter().copied().sum::<f64>();
        assert!(expected > 0.0);
        assert!(prediction.p1 > 0.0);
        assert!((prediction.p1 - expected).abs() < 1e-18);
    }

    #[test]
    fn binary_prediction_from_probs_normalizes_and_floors() {
        let pred = binary_prediction_from_probs(2.0, 6.0, 1e-6);
        assert!((pred.p0 - 0.25).abs() < 1e-12);
        assert!((pred.p1 - 0.75).abs() < 1e-12);

        let degenerate = binary_prediction_from_probs(f64::NAN, -3.0, 1e-6);
        assert_eq!(degenerate, BinaryPrediction::from_prob_one(0.5, 1e-6));
    }
}
