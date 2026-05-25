//! ZPAQ-backed sequential rate model.
//!
//! This backend estimates `log p(x_t | x_{<t})` by measuring incremental
//! streaming compression growth under a streamable ZPAQ method.

#[cfg(feature = "backend-zpaq")]
use std::f64::consts::LN_2;

#[cfg(feature = "backend-zpaq")]
const DEFAULT_MIN_PROB: f64 = 5.960_464_477_539_063e-8;

#[cfg(feature = "backend-zpaq")]
mod imp {
    use super::{DEFAULT_MIN_PROB, LN_2};
    use zpaq_rs::StreamingCompressor;

    struct ZpaqStreaming {
        compressor: Option<StreamingCompressor>,
        last_bits: f64,
    }

    /// Compatibility no-op retained for callers that historically released a
    /// process-global active stream before running other ZPAQ operations.
    pub(crate) fn release_active_zpaq_rate_stream() {}

    /// Stateful ZPAQ-backed estimator of sequential symbol log-probabilities.
    pub struct ZpaqRateModel {
        stream: ZpaqStreaming,
        history: Vec<u8>,
        history_bits: f64,
        pending_symbol: Option<u8>,
        pending_bits: f64,
        min_prob: f64,
        method: String,
    }

    impl ZpaqRateModel {
        fn new_streaming_compressor(method: &str) -> StreamingCompressor {
            StreamingCompressor::new(method).unwrap_or_else(|e| {
                panic!("ZPAQ rate backend requires a streamable method; got '{method}': {e}")
            })
        }

        fn replace_stream_compressor(&mut self) {
            // Drop the previous compressor before constructing the replacement
            // so the transition remains strictly sequential.
            drop(self.stream.compressor.take());
            self.stream.compressor = Some(Self::new_streaming_compressor(self.method.as_str()));
            self.stream.last_bits = 0.0;
        }

        fn rebuild_stream_from_history(&mut self) {
            self.replace_stream_compressor();
            self.history_bits = 0.0;
            let history_len = self.history.len();
            for idx in 0..history_len {
                let symbol: u8 = self.history[idx];
                let (after, _) = self.encode_bits(symbol);
                self.history_bits = after;
            }
            self.pending_symbol = None;
            self.pending_bits = 0.0;
        }

        /// Create a new model with the provided streamable ZPAQ `method`.
        ///
        /// `min_prob` clamps very small probabilities for numerical stability.
        pub fn new(method: impl Into<String>, min_prob: f64) -> Self {
            let method = method.into();
            let min_prob = if min_prob.is_finite() && min_prob > 0.0 {
                min_prob
            } else {
                DEFAULT_MIN_PROB
            };
            zpaq_rs::validate_streaming_method(method.as_str()).unwrap_or_else(|e| {
                panic!("ZPAQ rate backend requires a streamable method; got '{method}': {e}")
            });

            Self {
                stream: ZpaqStreaming {
                    compressor: Some(Self::new_streaming_compressor(method.as_str())),
                    last_bits: 0.0,
                },
                history: Vec::new(),
                history_bits: 0.0,
                pending_symbol: None,
                pending_bits: 0.0,
                min_prob,
                method,
            }
        }

        /// Begin a fresh stream lifecycle.
        ///
        /// Newly constructed models are already at fresh-state, so the first
        /// call avoids redundant compressor reconstruction.
        pub fn begin_stream(&mut self) {
            if self.history.is_empty() && self.pending_symbol.is_none() && self.history_bits == 0.0
            {
                return;
            }
            self.reset();
        }

        /// Reset model state and clear any pending prediction cache.
        pub fn reset(&mut self) {
            self.replace_stream_compressor();
            self.history.clear();
            self.history_bits = 0.0;
            self.pending_symbol = None;
            self.pending_bits = 0.0;
        }

        fn log_prob_from_bits(min_prob: f64, bits: f64) -> f64 {
            let logp = -(bits * LN_2);
            logp.max(min_prob.ln())
        }

        fn log_prob_from_history(&self, symbol: u8) -> f64 {
            let mut compressor = Self::new_streaming_compressor(self.method.as_str());
            for &b in &self.history {
                compressor
                    .push(b)
                    .expect("zpaq streaming compression failed");
            }
            let before = compressor.bits();
            compressor
                .push(symbol)
                .expect("zpaq streaming compression failed");
            let bits = (compressor.bits() - before).max(0.0);
            Self::log_prob_from_bits(self.min_prob, bits)
        }

        fn encode_bits(&mut self, symbol: u8) -> (f64, f64) {
            let before = self.stream.last_bits;
            let compressor = self
                .stream
                .compressor
                .as_mut()
                .expect("zpaq stream compressor must be initialized");
            compressor
                .push(symbol)
                .expect("zpaq streaming compression failed");
            let after = compressor.bits();
            self.stream.last_bits = after;
            (after, (after - before).max(0.0))
        }

        /// Return `ln p(symbol | history)` under the current model state.
        ///
        /// This may cache the encoded-bit result for a matching immediate `update`.
        pub fn log_prob(&mut self, symbol: u8) -> f64 {
            if let Some(pending) = self.pending_symbol {
                if pending == symbol {
                    return Self::log_prob_from_bits(self.min_prob, self.pending_bits);
                }
                // We cannot rollback `StreamingCompressor`; rebuild to committed history.
                self.rebuild_stream_from_history();
            }

            let (_, bits) = self.encode_bits(symbol);
            self.pending_symbol = Some(symbol);
            self.pending_bits = bits;
            Self::log_prob_from_bits(self.min_prob, bits)
        }

        /// Fill 256-way log-probabilities for the current committed history without mutation.
        pub fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
            for (sym, slot) in out.iter_mut().enumerate() {
                *slot = self.log_prob_from_history(sym as u8);
            }
        }

        /// Advance model state with one observed symbol.
        pub fn update(&mut self, symbol: u8) {
            if let Some(pending) = self.pending_symbol
                && pending == symbol
            {
                self.history_bits += self.pending_bits;
                self.pending_symbol = None;
                self.pending_bits = 0.0;
                self.history.push(symbol);
                return;
            }
            if self.pending_symbol.is_some() {
                self.rebuild_stream_from_history();
            }
            let (after, _) = self.encode_bits(symbol);
            self.history_bits = after;
            self.pending_symbol = None;
            self.pending_bits = 0.0;
            self.history.push(symbol);
        }

        /// Score and consume an entire byte slice, returning total code length in bits.
        pub fn update_and_score(&mut self, data: &[u8]) -> f64 {
            if data.is_empty() {
                return 0.0;
            }
            if self.pending_symbol.is_some() {
                self.rebuild_stream_from_history();
            }
            let mut bits = 0.0;
            for &b in data {
                let (after, delta) = self.encode_bits(b);
                self.history_bits = after;
                bits += delta;
                self.history.push(b);
            }
            bits
        }
    }

    impl Clone for ZpaqRateModel {
        fn clone(&self) -> Self {
            let mut cloned = Self::new(self.method.clone(), self.min_prob);
            if !self.history.is_empty() {
                let _ = cloned.update_and_score(&self.history);
            }
            if let Some(symbol) = self.pending_symbol {
                let (_, bits) = cloned.encode_bits(symbol);
                cloned.pending_symbol = Some(symbol);
                cloned.pending_bits = bits;
            } else {
                cloned.pending_symbol = None;
                cloned.pending_bits = 0.0;
            }
            cloned
        }
    }

    /// Validate that `method` is streamable and accepted by the ZPAQ backend.
    pub fn validate_zpaq_rate_method(method: &str) -> Result<(), String> {
        zpaq_rs::validate_streaming_method(method).map_err(|e| e.to_string())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn zpaq_log_prob_update_matches_update_and_score() {
            let data = b"the quick brown fox jumps over the lazy dog";
            let mut model_a = ZpaqRateModel::new("1", 1e-9);
            let mut bits_a = 0.0;
            for &b in data {
                let logp = model_a.log_prob(b);
                bits_a += -logp / LN_2;
                model_a.update(b);
            }

            let mut model_b = ZpaqRateModel::new("1", 1e-9);
            let bits_b = model_b.update_and_score(data);

            let diff = (bits_a - bits_b).abs();
            assert!(diff < 1e-6, "bits mismatch: {bits_a} vs {bits_b}");
        }

        #[test]
        fn zpaq_update_and_score_keeps_raw_bit_deltas_when_floor_would_bind() {
            let data: Vec<u8> = (0u8..=255).collect();
            let mut raw_model = ZpaqRateModel::new("1", 0.5);
            let mut raw_bits = 0.0;
            for &symbol in &data {
                let (after, delta) = raw_model.encode_bits(symbol);
                raw_model.history_bits = after;
                raw_model.history.push(symbol);
                raw_bits += delta;
            }
            assert!(
                raw_bits > data.len() as f64,
                "test requires raw ZPAQ cost to exceed the 1-bit floor cap"
            );

            let mut scored_model = ZpaqRateModel::new("1", 0.5);
            let scored_bits = scored_model.update_and_score(&data);

            assert!(
                (scored_bits - raw_bits).abs() < 1e-9,
                "metric path must preserve raw ZPAQ bit growth: scored={scored_bits} raw={raw_bits}"
            );
        }

        #[test]
        fn zpaq_fill_log_probs_is_non_mutating() {
            let history = b"zpaq fill non mutating";
            let mut model_a = ZpaqRateModel::new("1", 1e-9);
            let mut model_b = ZpaqRateModel::new("1", 1e-9);
            for &b in history {
                model_a.update(b);
                model_b.update(b);
            }

            let mut row = [0.0f64; 256];
            model_b.fill_log_probs(&mut row);

            let sym = b'x';
            let lp_a = model_a.log_prob(sym);
            let lp_b = model_b.log_prob(sym);
            assert!((lp_a - lp_b).abs() < 1e-9, "lp_a={lp_a} lp_b={lp_b}");
            assert!((row[sym as usize] - lp_a).abs() < 1e-9);

            model_a.update(sym);
            model_b.update(sym);
            let next_sym = b'y';
            let lp_a2 = model_a.log_prob(next_sym);
            let lp_b2 = model_b.log_prob(next_sym);
            assert!((lp_a2 - lp_b2).abs() < 1e-9, "lp_a2={lp_a2} lp_b2={lp_b2}");
        }

        #[test]
        fn zpaq_fill_log_probs_preserves_pending_prediction_cache() {
            let history = b"zpaq fill preserves pending";
            let mut model_a = ZpaqRateModel::new("1", 1e-9);
            let mut model_b = ZpaqRateModel::new("1", 1e-9);
            for &b in history {
                model_a.update(b);
                model_b.update(b);
            }

            let probe = b'x';
            let lp_before = model_a.log_prob(probe);
            let mut row = [0.0f64; 256];
            model_a.fill_log_probs(&mut row);
            assert!(
                (row[probe as usize] - model_b.log_prob_from_history(probe)).abs() < 1e-9,
                "fill must score committed history, not speculative pending state"
            );

            let lp_after = model_a.log_prob(probe);
            assert!(
                (lp_before - lp_after).abs() < 1e-9,
                "fill must preserve the pending speculative cache: before={lp_before} after={lp_after}"
            );

            model_a.update(probe);
            model_b.update(probe);
            let next = b'y';
            let lp_a = model_a.log_prob(next);
            let lp_b = model_b.log_prob(next);
            assert!((lp_a - lp_b).abs() < 1e-9, "lp_a={lp_a} lp_b={lp_b}");
        }

        #[test]
        fn zpaq_clone_preserves_pending_prediction_state() {
            let mut model_a = ZpaqRateModel::new("1", 1e-9);
            for &b in b"clone preserves pending state" {
                model_a.update(b);
            }

            let probe = b'x';
            let lp_a = model_a.log_prob(probe);
            let mut model_b = model_a.clone();
            let lp_b = model_b.log_prob(probe);
            assert!((lp_a - lp_b).abs() < 1e-9, "lp_a={lp_a} lp_b={lp_b}");

            model_a.update(probe);
            model_b.update(probe);
            let next = b'y';
            let lp_a2 = model_a.log_prob(next);
            let lp_b2 = model_b.log_prob(next);
            assert!((lp_a2 - lp_b2).abs() < 1e-9, "lp_a2={lp_a2} lp_b2={lp_b2}");
        }

        #[test]
        fn zpaq_interleaved_models_match_separate_baselines() {
            let history_a = b"interleaved zpaq model A";
            let history_b = b"interleaved zpaq model B";
            let sequence_a = b"ABACABA";
            let sequence_b = b"XYZYZZX";

            let mut interleaved_a = ZpaqRateModel::new("1", 1e-9);
            let mut interleaved_b = ZpaqRateModel::new("1", 1e-9);
            let mut baseline_a = ZpaqRateModel::new("1", 1e-9);
            let mut baseline_b = ZpaqRateModel::new("1", 1e-9);

            for &b in history_a {
                interleaved_a.update(b);
                baseline_a.update(b);
            }
            for &b in history_b {
                interleaved_b.update(b);
                baseline_b.update(b);
            }

            for (&sym_a, &sym_b) in sequence_a.iter().zip(sequence_b.iter()) {
                let lp_interleaved_a = interleaved_a.log_prob(sym_a);
                let lp_baseline_a = baseline_a.log_prob(sym_a);
                assert!(
                    (lp_interleaved_a - lp_baseline_a).abs() < 1e-9,
                    "interleaving drifted model A: interleaved={lp_interleaved_a} baseline={lp_baseline_a}"
                );
                interleaved_a.update(sym_a);
                baseline_a.update(sym_a);

                let lp_interleaved_b = interleaved_b.log_prob(sym_b);
                let lp_baseline_b = baseline_b.log_prob(sym_b);
                assert!(
                    (lp_interleaved_b - lp_baseline_b).abs() < 1e-9,
                    "interleaving drifted model B: interleaved={lp_interleaved_b} baseline={lp_baseline_b}"
                );
                interleaved_b.update(sym_b);
                baseline_b.update(sym_b);
            }
        }

        #[test]
        fn zpaq_validate_method_is_non_intrusive_with_live_model() {
            let mut baseline = ZpaqRateModel::new("1", 1e-9);
            let mut probe = ZpaqRateModel::new("1", 1e-9);
            for &b in b"validate zpaq method while model is live" {
                baseline.update(b);
                probe.update(b);
            }

            validate_zpaq_rate_method("1").expect("streaming method should validate");

            let lp_baseline = baseline.log_prob(b'v');
            let lp_probe = probe.log_prob(b'v');
            assert!(
                (lp_baseline - lp_probe).abs() < 1e-9,
                "validation disturbed live model state: baseline={lp_baseline} probe={lp_probe}"
            );
        }
    }
}

#[cfg(not(feature = "backend-zpaq"))]
mod imp {
    #[derive(Clone)]
    pub struct ZpaqRateModel {
        min_log_prob: f64,
    }

    impl ZpaqRateModel {
        pub fn new(_method: impl Into<String>, min_prob: f64) -> Self {
            let min_prob = if min_prob.is_finite() && min_prob > 0.0 {
                min_prob
            } else {
                1e-12
            };
            Self {
                min_log_prob: min_prob.ln(),
            }
        }

        pub fn reset(&mut self) {}

        pub fn log_prob(&mut self, _symbol: u8) -> f64 {
            self.min_log_prob
        }

        pub fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
            out.fill(self.min_log_prob);
        }

        pub fn update(&mut self, _symbol: u8) {}

        pub fn update_and_score(&mut self, data: &[u8]) -> f64 {
            let bits_per_symbol = -self.min_log_prob / std::f64::consts::LN_2;
            bits_per_symbol * (data.len() as f64)
        }
    }

    pub fn validate_zpaq_rate_method(_method: &str) -> Result<(), String> {
        Err("zpaq backend disabled at compile time".to_string())
    }

    pub(crate) fn release_active_zpaq_rate_stream() {}
}

/// Stateful ZPAQ-based rate estimator.
pub use imp::ZpaqRateModel;
#[cfg(feature = "backend-zpaq")]
pub(crate) use imp::release_active_zpaq_rate_stream;
/// Validate that a ZPAQ method string is streamable and usable for rate modeling.
pub use imp::validate_zpaq_rate_method;
