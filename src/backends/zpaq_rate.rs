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
        compressor: StreamingCompressor,
        last_bits: f64,
    }

    /// Stateful ZPAQ-backed estimator of sequential symbol log-probabilities.
    pub struct ZpaqRateModel {
        stream: ZpaqStreaming,
        pending_symbol: Option<u8>,
        pending_bits: f64,
        min_prob: f64,
        method: String,
    }

    impl ZpaqRateModel {
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

            let compressor = StreamingCompressor::new(method.as_str()).unwrap_or_else(|e| {
                panic!("ZPAQ rate backend requires a streamable method; got '{method}': {e}")
            });

            Self {
                stream: ZpaqStreaming {
                    compressor,
                    last_bits: 0.0,
                },
                pending_symbol: None,
                pending_bits: 0.0,
                min_prob,
                method,
            }
        }

        /// Reset model state and clear any pending prediction cache.
        pub fn reset(&mut self) {
            let method = self.method.clone();
            let compressor = StreamingCompressor::new(method.as_str()).unwrap_or_else(|e| {
                panic!("ZPAQ rate backend requires a streamable method; got '{method}': {e}")
            });
            self.stream = ZpaqStreaming {
                compressor,
                last_bits: 0.0,
            };
            self.pending_symbol = None;
            self.pending_bits = 0.0;
        }

        fn encode_bits(&mut self, symbol: u8) -> f64 {
            let before = self.stream.last_bits;
            self.stream
                .compressor
                .push(symbol)
                .expect("zpaq streaming compression failed");
            let after = self.stream.compressor.bits();
            self.stream.last_bits = after;
            (after - before).max(0.0)
        }

        /// Return `ln p(symbol | history)` under the current model state.
        ///
        /// This may cache the encoded-bit result for a matching immediate `update`.
        pub fn log_prob(&mut self, symbol: u8) -> f64 {
            if let Some(pending) = self.pending_symbol {
                if pending == symbol {
                    let logp = -(self.pending_bits * LN_2);
                    return logp.max(self.min_prob.ln());
                }
                self.pending_symbol = None;
            }

            let bits = self.encode_bits(symbol);
            self.pending_symbol = Some(symbol);
            self.pending_bits = bits;
            let logp = -(bits * LN_2);
            logp.max(self.min_prob.ln())
        }

        /// Advance model state with one observed symbol.
        pub fn update(&mut self, symbol: u8) {
            if let Some(pending) = self.pending_symbol
                && pending == symbol
            {
                self.pending_symbol = None;
                return;
            }
            self.pending_symbol = None;
            self.pending_bits = 0.0;
            let _ = self.encode_bits(symbol);
        }

        /// Score and consume an entire byte slice, returning total code length in bits.
        pub fn update_and_score(&mut self, data: &[u8]) -> f64 {
            if data.is_empty() {
                return 0.0;
            }
            self.pending_symbol = None;
            self.pending_bits = 0.0;
            let mut bits = 0.0;
            for &b in data {
                bits += self.encode_bits(b);
            }
            bits
        }
    }

    /// Validate that `method` is streamable and accepted by the ZPAQ backend.
    pub fn validate_zpaq_rate_method(method: &str) -> Result<(), String> {
        StreamingCompressor::new(method)
            .map(|_| ())
            .map_err(|e| e.to_string())
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
    }
}

#[cfg(not(feature = "backend-zpaq"))]
mod imp {
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

        pub fn update(&mut self, _symbol: u8) {}

        pub fn update_and_score(&mut self, data: &[u8]) -> f64 {
            let bits_per_symbol = -self.min_log_prob / std::f64::consts::LN_2;
            bits_per_symbol * (data.len() as f64)
        }
    }

    pub fn validate_zpaq_rate_method(_method: &str) -> Result<(), String> {
        Err("zpaq backend disabled at compile time".to_string())
    }
}

/// Stateful ZPAQ-based rate estimator.
pub use imp::ZpaqRateModel;
/// Validate that a ZPAQ method string is streamable and usable for rate modeling.
pub use imp::validate_zpaq_rate_method;
