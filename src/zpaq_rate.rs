use std::f64::consts::LN_2;

use zpaq_rs::StreamingCompressor;

const DEFAULT_MIN_PROB: f64 = 5.960_464_477_539_063e-8;

struct ZpaqStreaming {
    compressor: StreamingCompressor,
    last_bits: f64,
}

pub struct ZpaqRateModel {
    stream: ZpaqStreaming,
    pending_symbol: Option<u8>,
    pending_bits: f64,
    min_prob: f64,
    method: String,
}

impl ZpaqRateModel {
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

    pub fn update(&mut self, symbol: u8) {
        if let Some(pending) = self.pending_symbol {
            if pending == symbol {
                self.pending_symbol = None;
                return;
            }
        }
        self.pending_symbol = None;
        self.pending_bits = 0.0;
        let _ = self.encode_bits(symbol);
    }

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

pub fn validate_zpaq_rate_method(method: &str) -> Result<(), String> {
    StreamingCompressor::new(method)
        .map(|_| ())
        .map_err(|e| e.to_string())
}
