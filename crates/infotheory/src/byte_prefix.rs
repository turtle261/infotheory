//! Shared MSB-first byte-prefix utilities for bitwise views of byte predictors.

/// Number of symbols in the byte alphabet.
pub(crate) const BYTE_SYMBOLS: usize = 256;
/// Length of a byte-symbol CDF row, including the leading zero.
pub(crate) const BYTE_CDF_LEN: usize = BYTE_SYMBOLS + 1;

/// Fixed-size CDF row over byte symbols.
pub(crate) type BytePrefixCdf = [f64; BYTE_CDF_LEN];

/// Reusable heap-stable CDF rows for a collection of experts.
///
/// A CDF row is 257 `f64`s (just over 2 KiB). Boxing each row keeps resizing
/// and reordering the expert scratch collection from copying that payload and
/// avoids placing it on transient stack frames.
#[allow(clippy::vec_box)]
pub(crate) type BytePrefixCdfScratch = Vec<Box<BytePrefixCdf>>;

/// Create a zeroed CDF row.
#[inline]
pub(crate) fn zeroed_prefix_cdf() -> BytePrefixCdf {
    [0.0; BYTE_CDF_LEN]
}

/// Allocate a reusable zeroed CDF row.
#[inline]
pub(crate) fn zeroed_prefix_cdf_box() -> Box<BytePrefixCdf> {
    Box::new(zeroed_prefix_cdf())
}

/// Active interval for an MSB-first walk through a byte-symbol CDF.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MsbPrefixRange {
    lo: usize,
    hi: usize,
}

impl MsbPrefixRange {
    /// The whole byte alphabet before any prefix bits have been observed.
    pub(crate) const FULL: Self = Self {
        lo: 0,
        hi: BYTE_SYMBOLS,
    };

    /// Return `P(next bit = 1)` under `cdf` for this prefix interval.
    #[inline]
    pub(crate) fn prob_one(self, cdf: &[f64], min_prob: f64) -> f64 {
        debug_assert!(
            min_prob.is_finite() && min_prob > 0.0 && min_prob < 0.5,
            "byte-prefix probability floor must be finite and in (0, 0.5)"
        );
        debug_assert!(
            self.lo < self.hi && self.hi <= BYTE_SYMBOLS,
            "MSB byte-prefix range invariant violated: [{}, {})",
            self.lo,
            self.hi
        );
        debug_assert!(
            cdf.len() >= BYTE_CDF_LEN,
            "byte-prefix CDF requires {BYTE_CDF_LEN} slots, got {}",
            cdf.len()
        );
        let mid: usize = (self.lo + self.hi) >> 1;
        let total: f64 = (cdf[self.hi] - cdf[self.lo]).max(min_prob);
        let one: f64 = (cdf[self.hi] - cdf[mid]).max(0.0);
        (one / total).clamp(min_prob, 1.0 - min_prob)
    }

    /// Advance the active interval by one observed MSB-first bit.
    #[inline]
    pub(crate) fn observe(&mut self, bit: bool) {
        let mid: usize = (self.lo + self.hi) >> 1;
        if bit {
            self.lo = mid;
        } else {
            self.hi = mid;
        }
    }

    /// Return the range that would result from observing `bit`.
    #[cfg(test)]
    #[inline]
    pub(crate) fn observed(mut self, bit: bool) -> Self {
        self.observe(bit);
        self
    }

    /// Current lower bound, useful when a backend owns the CDF lookup.
    #[cfg(test)]
    #[inline]
    pub(crate) fn lo(self) -> usize {
        self.lo
    }

    /// Current upper bound, useful when a backend owns the CDF lookup.
    #[cfg(test)]
    #[inline]
    pub(crate) fn hi(self) -> usize {
        self.hi
    }
}

impl Default for MsbPrefixRange {
    fn default() -> Self {
        Self::FULL
    }
}

/// Advance the ZPAQ/PAQ-style prefix code where root is `1`.
#[inline]
pub(crate) fn advanced_prefix_code(prefix: u16, bit: bool) -> u16 {
    (prefix << 1) | u16::from(bit)
}

/// Build a prefix CDF from a predictor-owned PDF whose validity is a caller invariant.
#[inline]
pub(crate) fn fill_prefix_cdf_from_pdf(cdf: &mut BytePrefixCdf, pdf: &[f64], min_prob: f64) {
    debug_assert_eq!(
        pdf.len(),
        BYTE_SYMBOLS,
        "byte-prefix CDF construction requires a full 256-element PDF"
    );
    debug_assert!(
        pdf.iter().all(|&p| p.is_finite() && p >= 0.0),
        "Predictor contract violation: predictor emitted non-finite or negative PDF mass"
    );
    cdf[0] = 0.0;
    for idx in 0..BYTE_SYMBOLS {
        cdf[idx + 1] = cdf[idx] + pdf[idx].max(min_prob);
    }
    debug_assert!(
        cdf[BYTE_SYMBOLS].is_finite() && cdf[BYTE_SYMBOLS] > 0.0,
        "Predictor contract violation: invalid prefix-CDF total ({})",
        cdf[BYTE_SYMBOLS]
    );
}

/// Build a prefix CDF from byte log probabilities.
#[inline]
pub(crate) fn fill_prefix_cdf_from_log_probs(
    cdf: &mut BytePrefixCdf,
    log_probs: &[f64; BYTE_SYMBOLS],
    min_prob: f64,
) {
    debug_assert!(
        min_prob.is_finite() && min_prob > 0.0,
        "byte-prefix log-probability floor must be positive and finite"
    );
    cdf[0] = 0.0;
    for (idx, &lp) in log_probs.iter().enumerate() {
        let p: f64 = if lp.is_finite() {
            lp.min(0.0).exp().clamp(min_prob, 1.0)
        } else {
            min_prob
        };
        cdf[idx + 1] = cdf[idx] + p;
    }
}

/// Build a normalized, defensive prefix CDF from externally supplied byte masses.
#[cfg(feature = "backend-calibrated")]
#[inline]
pub(crate) fn fill_normalized_prefix_cdf_from_pdf(cdf: &mut BytePrefixCdf, pdf: &[f64]) {
    cdf[0] = 0.0;
    for idx in 0..BYTE_SYMBOLS {
        let p: f64 = pdf.get(idx).copied().unwrap_or(0.0);
        cdf[idx + 1] = cdf[idx] + if p.is_finite() { p.max(0.0) } else { 0.0 };
    }
    let total: f64 = cdf[BYTE_SYMBOLS];
    if !total.is_finite() || total <= 0.0 {
        for (idx, slot) in cdf.iter_mut().enumerate() {
            *slot = (idx as f64) / (BYTE_SYMBOLS as f64);
        }
        return;
    }
    for slot in cdf.iter_mut().skip(1) {
        *slot /= total;
    }
    cdf[BYTE_SYMBOLS] = 1.0;
}

/// Normalize a PDF slice in-place, applying `min_prob` as the per-entry floor.
///
/// Each element is first replaced by `max(finite_value, min_prob)` (non-finite
/// entries are treated as zero before flooring). Callers must pass a non-empty
/// distribution. If the resulting total is non-positive or non-finite, non-empty
/// slices are filled with a uniform distribution.
///
/// The `min_prob` floor is a **semantic parameter**: coder-facing paths supply
/// [`crate::mixture::DEFAULT_MIN_PROB`] so every symbol gets at least one
/// quantization slot; SSE-internal intermediate outputs supply
/// [`f64::MIN_POSITIVE`] to avoid distorting concentrated mass distributions.
#[inline]
pub(crate) fn normalize_pdf(pdf: &mut [f64], min_prob: f64) {
    debug_assert!(
        min_prob.is_finite() && min_prob > 0.0,
        "normalize_pdf floor must be positive and finite"
    );
    debug_assert!(
        !pdf.is_empty(),
        "normalize_pdf requires a non-empty distribution"
    );
    if pdf.is_empty() {
        return;
    }
    let mut sum: f64 = 0.0;
    for p in pdf.iter_mut() {
        let v: f64 = if p.is_finite() { *p } else { 0.0 };
        *p = v.max(min_prob);
        sum += *p;
    }
    if !sum.is_finite() || sum <= 0.0 {
        let u: f64 = 1.0 / (pdf.len() as f64);
        pdf.fill(u);
        return;
    }
    let inv: f64 = 1.0 / sum;
    for p in pdf.iter_mut() {
        *p *= inv;
    }
}

/// Normalize a non-negative PDF slice without applying a per-entry floor.
///
/// This is appropriate for materializing probabilities that already come from
/// a coherent binary decision tree with floor-clamped conditional branches.
/// Applying a second symbol-level floor would change the tree semantics and
/// make the byte PDF disagree with the product of the same per-bit branches.
#[inline]
pub(crate) fn normalize_pdf_mass_only(pdf: &mut [f64]) {
    debug_assert!(
        !pdf.is_empty(),
        "normalize_pdf_mass_only requires a non-empty distribution"
    );
    if pdf.is_empty() {
        return;
    }
    let mut sum: f64 = 0.0;
    for p in pdf.iter_mut() {
        let v: f64 = if p.is_finite() { (*p).max(0.0) } else { 0.0 };
        *p = v;
        sum += v;
    }
    if !sum.is_finite() || sum <= 0.0 {
        let u: f64 = 1.0 / (pdf.len() as f64);
        pdf.fill(u);
        return;
    }
    let inv: f64 = 1.0 / sum;
    for p in pdf.iter_mut() {
        *p *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BYTE_SYMBOLS, BytePrefixCdf, MsbPrefixRange, fill_prefix_cdf_from_log_probs,
        fill_prefix_cdf_from_pdf, zeroed_prefix_cdf,
    };
    use crate::mixture::DEFAULT_MIN_PROB;

    #[test]
    #[cfg(debug_assertions)]
    fn fill_prefix_cdf_from_pdf_rejects_nan_pdf_in_debug() {
        let mut pdf = [1.0 / 256.0; 256];
        pdf[17] = f64::NAN;

        let result = std::panic::catch_unwind(|| {
            let mut cdf: BytePrefixCdf = zeroed_prefix_cdf();
            fill_prefix_cdf_from_pdf(&mut cdf, &pdf, f64::MIN_POSITIVE);
        });

        assert!(result.is_err());
    }

    #[test]
    #[cfg(debug_assertions)]
    fn fill_prefix_cdf_from_pdf_rejects_negative_pdf_in_debug() {
        let mut pdf = [1.0 / 256.0; 256];
        pdf[17] = -0.1;

        let result = std::panic::catch_unwind(|| {
            let mut cdf: BytePrefixCdf = zeroed_prefix_cdf();
            fill_prefix_cdf_from_pdf(&mut cdf, &pdf, f64::MIN_POSITIVE);
        });

        assert!(result.is_err());
    }

    #[test]
    fn msb_prefix_range_tracks_cdf_halves() {
        let mut cdf: BytePrefixCdf = zeroed_prefix_cdf();
        for (idx, slot) in cdf.iter_mut().enumerate() {
            *slot = idx as f64;
        }

        let mut range = MsbPrefixRange::FULL;
        assert_eq!(range.prob_one(&cdf, f64::MIN_POSITIVE), 0.5);
        range.observe(true);
        assert_eq!(range.lo(), 128);
        assert_eq!(range.hi(), 256);
        assert_eq!(range.prob_one(&cdf, f64::MIN_POSITIVE), 0.5);
        assert_eq!(range.observed(false), MsbPrefixRange { lo: 128, hi: 192 });
    }

    #[test]
    fn fill_prefix_cdf_from_log_probs_clamps_positive_log_probs_before_exp() {
        let mut log_probs: [f64; BYTE_SYMBOLS] = [-(BYTE_SYMBOLS as f64).ln(); BYTE_SYMBOLS];
        log_probs[17] = 710.0;
        log_probs[19] = f64::INFINITY;

        let mut cdf: BytePrefixCdf = zeroed_prefix_cdf();
        fill_prefix_cdf_from_log_probs(&mut cdf, &log_probs, DEFAULT_MIN_PROB);

        assert!(cdf.iter().all(|value| value.is_finite()));
        assert!(cdf[18] - cdf[17] <= 1.0);
        let observed_floor_mass: f64 = cdf[20] - cdf[19];
        let cdf_ulp_scale: f64 = f64::EPSILON * cdf[20].abs().max(1.0);
        assert!(
            (observed_floor_mass - DEFAULT_MIN_PROB).abs() <= 4.0 * cdf_ulp_scale,
            "observed floor mass {observed_floor_mass:e} differed from DEFAULT_MIN_PROB {DEFAULT_MIN_PROB:e}"
        );
    }
}
