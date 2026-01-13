//! Benchmarks for entropy coders.
//!
//! This module benchmarks the two entropy coding backends (AC and rANS) as well as
//! the supporting operations (softmax, CDF quantization). These benchmarks help
//! identify performance bottlenecks in the entropy coding pipeline.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench coders
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rwkvzip::coders::{
    cdf_for_symbol, quantize_pdf_to_cdf, quantize_pdf_to_cdf_inplace, quantize_pdf_to_rans_cdf,
    quantize_pdf_to_rans_cdf_with_buffer, softmax_pdf_floor, softmax_pdf_floor_inplace,
    ArithmeticDecoder, ArithmeticEncoder, RansDecoder, RansEncoder,
};

// =============================================================================
// Arithmetic Coding Benchmarks
// =============================================================================

fn bench_ac_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("ac_encode");

    // Test with different distribution shapes
    let uniform_pdf: Vec<f64> = vec![0.25; 4];
    let skewed_pdf: Vec<f64> = vec![0.7, 0.2, 0.08, 0.02];
    let symbols: Vec<usize> = (0..1000).map(|i| i % 4).collect();

    group.throughput(Throughput::Elements(symbols.len() as u64));

    group.bench_function("uniform_dist", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            let mut enc = ArithmeticEncoder::new(&mut buf);
            for &s in &symbols {
                enc.encode_symbol(black_box(&uniform_pdf), s).unwrap();
            }
            enc.finish().unwrap();
        })
    });

    group.bench_function("skewed_dist", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            let mut enc = ArithmeticEncoder::new(&mut buf);
            for &s in &symbols {
                enc.encode_symbol(black_box(&skewed_pdf), s).unwrap();
            }
            enc.finish().unwrap();
        })
    });

    group.finish();
}

fn bench_ac_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("ac_decode");

    let pdf = vec![0.5, 0.3, 0.15, 0.05];
    let symbols: Vec<usize> = (0..1000).map(|i| i % 4).collect();

    // Pre-encode for decode benchmark
    let mut buf = Vec::new();
    let mut enc = ArithmeticEncoder::new(&mut buf);
    for &s in &symbols {
        enc.encode_symbol(&pdf, s).unwrap();
    }
    let encoded = enc.finish().unwrap().to_vec();

    group.throughput(Throughput::Elements(symbols.len() as u64));

    group.bench_function("1000_symbols", |b| {
        b.iter(|| {
            let mut dec = ArithmeticDecoder::new(black_box(&encoded)).unwrap();
            for _ in 0..symbols.len() {
                black_box(dec.decode_symbol(&pdf).unwrap());
            }
        })
    });

    group.finish();
}

// =============================================================================
// rANS Coding Benchmarks
// =============================================================================

fn bench_rans_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rans_encode");

    let pdf = vec![0.5, 0.3, 0.15, 0.05];
    let cdf = quantize_pdf_to_rans_cdf(&pdf);
    let symbols: Vec<usize> = (0..1000).map(|i| i % 4).collect();

    group.throughput(Throughput::Elements(symbols.len() as u64));

    group.bench_function("1000_symbols", |b| {
        b.iter(|| {
            let mut enc = RansEncoder::new();
            // rANS encodes in reverse order
            for &s in symbols.iter().rev() {
                let c = cdf_for_symbol(black_box(&cdf), s);
                enc.encode(&c);
            }
            black_box(enc.finish())
        })
    });

    group.finish();
}

fn bench_rans_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rans_decode");

    let pdf = vec![0.5, 0.3, 0.15, 0.05];
    let cdf = quantize_pdf_to_rans_cdf(&pdf);
    let symbols: Vec<usize> = (0..1000).map(|i| i % 4).collect();

    // Pre-encode for decode benchmark
    let mut enc = RansEncoder::new();
    for &s in symbols.iter().rev() {
        let c = cdf_for_symbol(&cdf, s);
        enc.encode(&c);
    }
    let encoded = enc.finish();

    group.throughput(Throughput::Elements(symbols.len() as u64));

    group.bench_function("1000_symbols", |b| {
        b.iter(|| {
            let mut dec = RansDecoder::new(black_box(&encoded)).unwrap();
            for _ in 0..symbols.len() {
                black_box(dec.decode(&cdf).unwrap());
            }
        })
    });

    group.finish();
}

// =============================================================================
// Softmax Benchmarks (AVX2/FMA accelerated)
// =============================================================================

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    // Test with realistic model logits (256 vocab for byte-level)
    let logits: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1) - 12.8).collect();

    group.throughput(Throughput::Elements(256));

    // Allocating version
    group.bench_function("256_vocab_alloc", |b| {
        b.iter(|| black_box(softmax_pdf_floor(black_box(&logits), 256)))
    });

    // In-place version (zero-allocation hot path)
    let mut pdf_buffer = vec![0.0f64; 256];
    group.bench_function("256_vocab_inplace", |b| {
        b.iter(|| {
            softmax_pdf_floor_inplace(black_box(&logits), 256, &mut pdf_buffer);
            black_box(pdf_buffer[0])
        })
    });

    group.finish();
}

// =============================================================================
// CDF Quantization Benchmarks
// =============================================================================

fn bench_cdf_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdf_quantization");

    // Create realistic PDF distribution (exponential-ish, like model outputs)
    let pdf: Vec<f64> = (0..256).map(|i| ((i as f64) / 256.0).exp()).collect();
    let sum: f64 = pdf.iter().sum();
    let pdf: Vec<f64> = pdf.iter().map(|p| p / sum).collect();

    group.throughput(Throughput::Elements(256));

    // AC CDF - allocating
    group.bench_function("ac_256_alloc", |b| {
        b.iter(|| black_box(quantize_pdf_to_cdf(black_box(&pdf))))
    });

    // AC CDF - in-place
    let mut cdf_buffer = vec![0u32; 257];
    group.bench_function("ac_256_inplace", |b| {
        b.iter(|| {
            quantize_pdf_to_cdf_inplace(black_box(&pdf), &mut cdf_buffer);
            black_box(cdf_buffer[0])
        })
    });

    // rANS CDF - allocating
    group.bench_function("rans_256_alloc", |b| {
        b.iter(|| black_box(quantize_pdf_to_rans_cdf(black_box(&pdf))))
    });

    // rANS CDF - in-place with scratch buffer
    let mut cdf_buffer = vec![0u32; 257];
    let mut freq_buffer = vec![0i64; 256];
    group.bench_function("rans_256_inplace", |b| {
        b.iter(|| {
            quantize_pdf_to_rans_cdf_with_buffer(
                black_box(&pdf),
                &mut cdf_buffer,
                &mut freq_buffer,
            );
            black_box(cdf_buffer[0])
        })
    });

    group.finish();
}

// =============================================================================
// Roundtrip Benchmarks (encode + decode)
// =============================================================================

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    // Test different data sizes
    for size in [100, 1000, 10000] {
        let pdf = vec![0.4, 0.3, 0.2, 0.1];
        let cdf = quantize_pdf_to_rans_cdf(&pdf);
        let symbols: Vec<usize> = (0..size).map(|i| i % 4).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("ac", size), &symbols, |b, symbols| {
            b.iter(|| {
                // Encode
                let mut buf = Vec::new();
                let mut enc = ArithmeticEncoder::new(&mut buf);
                for &s in symbols {
                    enc.encode_symbol(&pdf, s).unwrap();
                }
                let encoded = enc.finish().unwrap().to_vec();

                // Decode
                let mut dec = ArithmeticDecoder::new(&encoded).unwrap();
                for _ in 0..symbols.len() {
                    black_box(dec.decode_symbol(&pdf).unwrap());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("rans", size), &symbols, |b, symbols| {
            b.iter(|| {
                // Encode
                let mut enc = RansEncoder::new();
                for &s in symbols.iter().rev() {
                    let c = cdf_for_symbol(&cdf, s);
                    enc.encode(&c);
                }
                let encoded = enc.finish();

                // Decode
                let mut dec = RansDecoder::new(&encoded).unwrap();
                for _ in 0..symbols.len() {
                    black_box(dec.decode(&cdf).unwrap());
                }
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ac_encode,
    bench_ac_decode,
    bench_rans_encode,
    bench_rans_decode,
    bench_softmax,
    bench_cdf_quantization,
    bench_roundtrip,
);

criterion_main!(benches);
