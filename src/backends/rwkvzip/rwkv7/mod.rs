//! High-performance RWKV7 inference kernel for x86_64.
//!
//! This module provides a highly optimized RWKV7 implementation specifically
//! designed for x86_64 CPUs with AVX2/FMA support. No portability fallbacks.
//!
//! # Architecture
//!
//! - All matrix operations are SIMD-vectorized (AVX2 + FMA)
//! - State updates use hand-tuned kernel for N=64 head dimension
//! - Memory layout optimized for cache efficiency
//! - No external BLAS dependencies

mod kernel;
mod model;
mod profiling;
mod tensor;
mod weights;

pub use model::ScratchBuffers;
pub use model::{Config, Model, State};
pub use profiling::{LayerProfiler, LayerTiming, NullProfiler, ProfilerSink};
pub use tensor::{Tensor1D, Tensor2D, TensorView1D, TensorView2D};
pub use weights::Weights;
