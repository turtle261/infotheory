use crate::backends::match_model::{MatchModel, MatchModelLifecycleSnapshot};

#[derive(Clone, Debug)]
/// Gapped/sparse match predictor wrapper over [`MatchModel`].
pub struct SparseMatchModel {
    inner: MatchModel,
}

impl SparseMatchModel {
    /// Create a sparse match model with configurable gap range and mixing behavior.
    pub fn new(
        hash_bits: usize,
        min_len: usize,
        max_len: usize,
        gap_min: usize,
        gap_max: usize,
        base_mix: f64,
        confidence_scale: f64,
    ) -> Self {
        Self {
            inner: MatchModel::new(
                hash_bits,
                min_len,
                max_len,
                gap_min,
                gap_max,
                base_mix,
                confidence_scale,
            ),
        }
    }

    /// Fill `out` with the current normalized byte PDF.
    pub fn fill_pdf(&mut self, out: &mut [f64; 256]) {
        self.inner.fill_pdf(out);
    }

    /// Borrow the current normalized byte PDF.
    pub fn pdf(&mut self) -> &[f64; 256] {
        self.inner.pdf()
    }

    /// Borrow the cumulative distribution derived from the current PDF.
    pub fn cdf(&mut self) -> &[f64; 257] {
        self.inner.cdf()
    }

    /// Return `ln(max(P(symbol), min_prob))`.
    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.inner.log_prob(symbol, min_prob)
    }

    /// Observe one symbol and update sparse-match statistics.
    pub fn update(&mut self, symbol: u8) {
        self.inner.update(symbol);
    }

    /// Reset only conditioning history, preserving learned tables.
    pub fn reset_history(&mut self) {
        self.inner.reset_history();
    }

    pub(crate) fn lifecycle_snapshot(&self) -> MatchModelLifecycleSnapshot {
        self.inner.lifecycle_snapshot()
    }

    pub(crate) fn restore_lifecycle_snapshot(&mut self, snapshot: MatchModelLifecycleSnapshot) {
        self.inner.restore_lifecycle_snapshot(snapshot);
    }

    /// Advance history without updating learned sparse-match tables.
    pub fn update_history_only(&mut self, symbol: u8) {
        self.inner.update_history_only(symbol);
    }

    /// Length of the best sparse/gapped match used for the current distribution.
    pub fn match_len(&mut self) -> usize {
        self.inner.match_len()
    }

    /// Predicted next byte from the best sparse/gapped match, if any.
    pub fn predicted_byte(&mut self) -> Option<u8> {
        self.inner.predicted_byte()
    }
}
