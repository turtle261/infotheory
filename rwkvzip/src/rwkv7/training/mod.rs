//! RWKV7 training (byte-level) implemented in Rust.
//!
//! This module trains a RWKV7 model that is *weight-compatible* with the
//! SIMD inference path in `crate::rwkv7::Model` by exporting the same
//! safetensors keys & shapes.
//!
//! Design goals:
//! - Correctness first: math matches `src/rwkv7/model.rs`
//! - GPU acceleration via libtorch (tch) when available
//! - Small-model friendliness (<10M params) with stable defaults

mod data;
mod export;
mod model;
mod model_fast;
mod train;
mod validate;
mod wkv_cuda;
mod wkv_fused;

pub use train::{train_enwik8, TrainConfig, TrainReport};
pub use wkv_cuda::cuda_wkv_available;
