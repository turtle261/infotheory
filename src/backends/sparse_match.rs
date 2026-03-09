use crate::backends::match_model::MatchModel;

#[derive(Clone, Debug)]
pub struct SparseMatchModel {
    inner: MatchModel,
}

impl SparseMatchModel {
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

    pub fn fill_pdf(&mut self, out: &mut [f64; 256]) {
        self.inner.fill_pdf(out);
    }

    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.inner.log_prob(symbol, min_prob)
    }

    pub fn update(&mut self, symbol: u8) {
        self.inner.update(symbol);
    }
}
