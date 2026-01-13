//! Benchmarks for end-to-end compression.
//!
//! This module benchmarks the full compression pipeline including model inference
//! and entropy coding. These benchmarks require a model file.
//!
//! # Running
//!
//! ```sh
//! # Using --model argument:
//! cargo bench --bench compression -- --model ./rwkv-10m.safetensors
//!
//! # Or using environment variable:
//! RWKV_MODEL=./rwkv-10m.safetensors cargo bench --bench compression
//! ```

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use once_cell::sync::Lazy;
use rwkvzip::{CoderType, Compressor};

// =============================================================================
// Benchmark Configuration
// =============================================================================

static BENCH_ARGS: Lazy<BenchArgs> = Lazy::new(BenchArgs::from_env);

struct BenchArgs {
    model: PathBuf,
}

impl BenchArgs {
    fn from_env() -> Self {
        let mut args = std::env::args();
        let mut model_arg: Option<PathBuf> = None;

        while let Some(arg) = args.next() {
            if arg == "--model" {
                if let Some(path) = args.next() {
                    model_arg = Some(PathBuf::from(path));
                }
            }
        }

        // Fall back to environment variables
        if model_arg.is_none() {
            if let Ok(env_path) = std::env::var("RWKV_MODEL") {
                model_arg = Some(PathBuf::from(env_path));
            } else if let Ok(env_path) = std::env::var("RWKV_BENCH_MODEL") {
                model_arg = Some(PathBuf::from(env_path));
            }
        }

        let model = model_arg.expect(
            "Provide --model <path> or set RWKV_MODEL environment variable\n\
             Example: cargo bench --bench compression -- --model ./rwkv-10m.safetensors",
        );

        Self { model }
    }
}

// =============================================================================
// Test Data Generation
// =============================================================================

/// Generate reproducible test data of specified size.
/// Uses a simple PRNG for deterministic, compressible content.
fn generate_test_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut state: u32 = 0x12345678;

    for _ in 0..size {
        // Simple xorshift PRNG
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Bias towards ASCII-like values for compressibility
        let byte = ((state & 0x7F) + 32) as u8;
        data.push(byte);
    }

    data
}

/// Generate highly compressible data (repeated pattern).
fn generate_repetitive_data(size: usize) -> Vec<u8> {
    let pattern = b"Hello, World! This is a test pattern. ";
    pattern.iter().cycle().take(size).copied().collect()
}

/// Generate incompressible random data.
fn generate_random_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut state: u64 = 0xDEADBEEFCAFEBABE;

    for _ in 0..size {
        // Better quality PRNG for uniformity
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        data.push((state >> 56) as u8);
    }

    data
}

// =============================================================================
// Compression Benchmarks
// =============================================================================

fn bench_compress_ac(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut compressor = Compressor::new(&args.model).expect("Failed to load model");

    let mut group = c.benchmark_group("compress_ac");

    // Test different data sizes
    for size in [256, 512] {
        let data = generate_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                compressor.reset();
                black_box(compressor.compress(data, CoderType::AC).unwrap())
            })
        });
    }

    group.finish();
}

fn bench_compress_rans(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut compressor = Compressor::new(&args.model).expect("Failed to load model");

    let mut group = c.benchmark_group("compress_rans");

    for size in [256, 512] {
        let data = generate_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                compressor.reset();
                black_box(compressor.compress(data, CoderType::RANS).unwrap())
            })
        });
    }

    group.finish();
}

// =============================================================================
// Decompression Benchmarks
// =============================================================================

fn bench_decompress_ac(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut compressor = Compressor::new(&args.model).expect("Failed to load model");

    let mut group = c.benchmark_group("decompress_ac");

    for size in [256, 512] {
        let data = generate_test_data(size);
        let compressed = compressor.compress(&data, CoderType::AC).unwrap();

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    compressor.reset();
                    black_box(compressor.decompress(compressed).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn bench_decompress_rans(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut compressor = Compressor::new(&args.model).expect("Failed to load model");

    let mut group = c.benchmark_group("decompress_rans");

    for size in [256, 512] {
        let data = generate_test_data(size);
        let compressed = compressor.compress(&data, CoderType::RANS).unwrap();

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    compressor.reset();
                    black_box(compressor.decompress(compressed).unwrap())
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Roundtrip Benchmarks
// =============================================================================

fn bench_roundtrip(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut compressor = Compressor::new(&args.model).expect("Failed to load model");

    let mut group = c.benchmark_group("roundtrip");

    let size = 1024;
    let data = generate_test_data(size);

    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("ac", |b| {
        b.iter(|| {
            compressor.reset();
            let compressed = compressor.compress(&data, CoderType::AC).unwrap();
            compressor.reset();
            black_box(compressor.decompress(&compressed).unwrap())
        })
    });

    group.bench_function("rans", |b| {
        b.iter(|| {
            compressor.reset();
            let compressed = compressor.compress(&data, CoderType::RANS).unwrap();
            compressor.reset();
            black_box(compressor.decompress(&compressed).unwrap())
        })
    });

    group.finish();
}

// =============================================================================
// Data Type Comparison Benchmarks
// =============================================================================

fn bench_data_types(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut compressor = Compressor::new(&args.model).expect("Failed to load model");

    let mut group = c.benchmark_group("data_types");

    let size = 1024;

    // Test with different data characteristics
    let test_cases = [
        ("random", generate_random_data(size)),
        ("mixed", generate_test_data(size)),
        ("repetitive", generate_repetitive_data(size)),
    ];

    for (name, data) in &test_cases {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("ac", name), data, |b, data| {
            b.iter(|| {
                compressor.reset();
                black_box(compressor.compress(data, CoderType::AC).unwrap())
            })
        });
    }

    group.finish();
}

// =============================================================================
// Cross-Entropy Benchmark (Model Quality)
// =============================================================================

fn bench_cross_entropy(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut compressor = Compressor::new(&args.model).expect("Failed to load model");

    let mut group = c.benchmark_group("cross_entropy");

    let size = 1024;
    let data = generate_test_data(size);

    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("1024_bytes", |b| {
        b.iter(|| {
            compressor.reset();
            black_box(compressor.cross_entropy(&data).unwrap())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compress_ac,
    bench_compress_rans,
    bench_decompress_ac,
    bench_decompress_rans,
    bench_roundtrip,
    bench_data_types,
    bench_cross_entropy,
);

criterion_main!(benches);
