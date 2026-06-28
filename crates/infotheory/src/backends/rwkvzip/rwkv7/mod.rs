//! High-performance RWKV7 inference kernel.
//!
//! This module provides a portable SIMD RWKV7 implementation using `wide` so
//! rustc/LLVM can pick the best ISA per target (x86_64, aarch64, wasm32 SIMD,
//! or scalar fallback on non-SIMD CPUs).
//!
//! # Architecture
//!
//! - Matrix/vector operations are vectorized via `wide` (`f32x8`)
//! - State updates are optimized for RWKV7 head dimension N=64
//! - Memory layout is cache-friendly and alignment-aware
//! - No external BLAS dependencies

mod kernel;
mod model;
mod profiling;
mod tensor;
mod weights;

pub use model::ScratchBuffers;
pub(crate) use model::TbpttReplayWorkspace;
pub use model::{Config, FullAdamState, Model, State, TrainScopeMask};
pub use profiling::{LayerProfiler, LayerTiming, NullProfiler, ProfilerSink};
pub use tensor::{Tensor1D, Tensor2D, TensorView1D, TensorView2D};
pub use weights::Weights;
