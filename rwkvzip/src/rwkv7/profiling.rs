use std::time::Duration;

/// Timing data for a single transformer block.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerTiming {
    /// Accumulated attention time in nanoseconds.
    pub attention_ns: u64,
    /// Accumulated FFN time in nanoseconds.
    pub ffn_ns: u64,
}

/// Sink trait used by the model to surface per-layer timings without
/// committing to a particular profiler implementation.
pub trait ProfilerSink {
    #[inline(always)]
    fn begin_token(&mut self) {}

    #[inline(always)]
    fn record_attention(&mut self, _layer: usize, _duration: Duration) {}

    #[inline(always)]
    fn record_ffn(&mut self, _layer: usize, _duration: Duration) {}
}

/// No-op profiler used by default to keep the fast path branch-free.
pub struct NullProfiler;

impl ProfilerSink for NullProfiler {}

/// Collects wall-clock timings for each transformer block.
#[derive(Clone, Debug)]
pub struct LayerProfiler {
    layers: Vec<LayerTiming>,
    tokens: u64,
}

impl LayerProfiler {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerTiming::default(); num_layers],
            tokens: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.tokens = 0;
        self.layers.fill(LayerTiming::default());
    }

    #[inline]
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    #[inline]
    pub fn timings(&self) -> &[LayerTiming] {
        &self.layers
    }

    fn accumulate(target: &mut u64, duration: Duration) {
        let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
        *target = target.saturating_add(nanos);
    }
}

impl ProfilerSink for LayerProfiler {
    #[inline(always)]
    fn begin_token(&mut self) {
        self.tokens = self.tokens.saturating_add(1);
    }

    #[inline(always)]
    fn record_attention(&mut self, layer: usize, duration: Duration) {
        if let Some(entry) = self.layers.get_mut(layer) {
            Self::accumulate(&mut entry.attention_ns, duration);
        }
    }

    #[inline(always)]
    fn record_ffn(&mut self, layer: usize, duration: Duration) {
        if let Some(entry) = self.layers.get_mut(layer) {
            Self::accumulate(&mut entry.ffn_ns, duration);
        }
    }
}
