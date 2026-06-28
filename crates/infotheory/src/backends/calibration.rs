use crate::api::CalibrationContextKind;
use crate::backends::text_context::{NeuralContextState, TextContextAnalyzer};

#[derive(Clone, Debug)]
/// Lightweight online calibrator that rescales a base 256-way PDF by context/bin.
///
/// The calibrator keeps per-context logits over probability bins and applies an
/// exponential tilt `p' ∝ p * exp(w_bin)` followed by normalization.
pub struct CalibratorCore {
    analyzer: TextContextAnalyzer,
    context: CalibrationContextKind,
    bins: usize,
    learning_rate: f64,
    bias_clip: f64,
    weights: Vec<f64>,
    last_context: usize,
    last_bins: Vec<usize>,
}

impl CalibratorCore {
    /// Create a calibrator with bounded bin count and stable learning parameters.
    pub fn new(
        context: CalibrationContextKind,
        bins: usize,
        learning_rate: f64,
        bias_clip: f64,
    ) -> Self {
        let bins = bins.max(2);
        let weights = vec![0.0; context_cardinality(context) * bins];
        Self {
            analyzer: TextContextAnalyzer::new(),
            context,
            bins,
            learning_rate: learning_rate.max(1e-6),
            bias_clip: bias_clip.max(1e-6),
            weights,
            last_context: 0,
            last_bins: vec![0; 256],
        }
    }

    /// Apply the learned calibration transform to `base`, writing a normalized PDF to `out`.
    ///
    /// `base`/`out` are expected to be 256-byte distributions.
    pub fn apply_pdf(&mut self, base: &[f64], out: &mut [f64]) {
        let ctx = context_index(self.context, self.analyzer.state());
        self.last_context = ctx;
        let offset = ctx * self.bins;
        let mut sum = 0.0;
        for i in 0..256 {
            let p = base[i].clamp(1e-12, 1.0 - 1e-12);
            let bin = probability_bin(p, self.bins);
            self.last_bins[i] = bin;
            let w = self.weights[offset + bin];
            let adjusted = p * w.exp();
            out[i] = adjusted;
            sum += adjusted;
        }
        if !sum.is_finite() || sum <= 0.0 {
            let u = 1.0 / 256.0;
            out.fill(u);
            return;
        }
        let inv = 1.0 / sum;
        for value in out.iter_mut() {
            *value *= inv;
        }
    }

    /// Update the active bin weight from the observed symbol and calibrated distribution.
    pub fn update(&mut self, symbol: u8, calibrated_pdf: &[f64]) {
        let idx = self.last_context * self.bins + self.last_bins[symbol as usize];
        let q = calibrated_pdf[symbol as usize].clamp(1e-9, 1.0);
        self.weights[idx] = (self.weights[idx] + self.learning_rate * (1.0 - q))
            .clamp(-self.bias_clip, self.bias_clip);
        self.analyzer.update(symbol);
    }

    /// Reset only the dynamic context state while preserving fitted weights.
    pub fn reset_context(&mut self) {
        self.analyzer = TextContextAnalyzer::new();
        self.last_context = 0;
        self.last_bins.fill(0);
    }

    /// Advance context state without updating fitted calibration weights.
    pub fn update_context_only(&mut self, symbol: u8) {
        self.analyzer.update(symbol);
    }
}

fn context_cardinality(kind: CalibrationContextKind) -> usize {
    match kind {
        CalibrationContextKind::Global => 1,
        CalibrationContextKind::ByteClass => 8,
        CalibrationContextKind::Text => 256,
        CalibrationContextKind::Repeat => 64,
        CalibrationContextKind::TextRepeat => 512,
    }
}

fn context_index(kind: CalibrationContextKind, state: NeuralContextState) -> usize {
    match kind {
        CalibrationContextKind::Global => 0,
        CalibrationContextKind::ByteClass => state.prev1_class as usize,
        CalibrationContextKind::Text => hash_state(
            &[
                state.prev1_class,
                state.prev2_class,
                state.word_len_bucket,
                state.prev_word_class,
                state.bracket_bucket,
                state.quote_flags,
                state.utf8_left,
                state.sentence_boundary as u8,
                state.paragraph_break as u8,
            ],
            256,
        ),
        CalibrationContextKind::Repeat => hash_state(
            &[
                state.repeat_len_bucket,
                state.copied_last_byte as u8,
                (state.run_len.min(31) as u8),
            ],
            64,
        ),
        CalibrationContextKind::TextRepeat => hash_state(
            &[
                state.prev1_class,
                state.word_len_bucket,
                state.prev_word_class,
                state.bracket_bucket,
                state.quote_flags,
                state.repeat_len_bucket,
                state.copied_last_byte as u8,
                state.paragraph_break as u8,
            ],
            512,
        ),
    }
}

fn probability_bin(prob: f64, bins: usize) -> usize {
    let logit = (prob / (1.0 - prob)).ln();
    let scaled = ((logit + 24.0) / 24.0).clamp(0.0, 1.0);
    (scaled * ((bins - 1) as f64)).round() as usize
}

fn hash_state(values: &[u8], modulo: usize) -> usize {
    let mut h = 0x9E37_79B9u32;
    for &value in values {
        h ^= value as u32;
        h = h.rotate_left(5).wrapping_mul(0x85EB_CA6B);
    }
    (h as usize) % modulo
}
