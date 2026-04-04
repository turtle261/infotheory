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
    /// Whether the caller should pay profiling overhead on the hot path.
    const ENABLED: bool = false;

    /// Start timing a new token forward pass.
    #[inline(always)]
    fn begin_token(&mut self) {}

    /// Record attention-kernel duration for `layer`.
    #[inline(always)]
    fn record_attention(&mut self, _layer: usize, _duration: Duration) {}

    /// Record feed-forward duration for `layer`.
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
    /// Create a layer profiler with `num_layers` counters.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerTiming::default(); num_layers],
            tokens: 0,
        }
    }

    #[inline]
    /// Reset token counter and all accumulated timings.
    pub fn reset(&mut self) {
        self.tokens = 0;
        self.layers.fill(LayerTiming::default());
    }

    #[inline]
    /// Number of tokens observed by this profiler.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    #[inline]
    /// Per-layer timing accumulators.
    pub fn timings(&self) -> &[LayerTiming] {
        &self.layers
    }

    fn accumulate(target: &mut u64, duration: Duration) {
        let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
        *target = target.saturating_add(nanos);
    }
}

impl ProfilerSink for LayerProfiler {
    const ENABLED: bool = true;

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
