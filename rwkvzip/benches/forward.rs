//! Micro-benchmarks for single-token RWKV7 inference.
//!
//! This module benchmarks the core forward pass, which is the performance-critical
//! component of the compression pipeline. Each byte requires one forward pass.
//!
//! # Running
//!
//! ```sh
//! # Basic benchmark:
//! cargo bench --bench forward -- --model ./rwkv-10m.safetensors
//!
//! # With per-layer profiling:
//! cargo bench --bench forward -- --model ./rwkv-10m.safetensors --profile-layers
//!
//! # Or using environment variable:
//! RWKV_MODEL=./rwkv-10m.safetensors cargo bench --bench forward
//! ```

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use once_cell::sync::Lazy;
use rwkvzip::rwkv7::{LayerProfiler, Model, ScratchBuffers};

// =============================================================================
// Benchmark Configuration
// =============================================================================

static BENCH_ARGS: Lazy<BenchArgs> = Lazy::new(BenchArgs::from_env);

struct BenchArgs {
    model: PathBuf,
    profile_layers: bool,
}

impl BenchArgs {
    fn from_env() -> Self {
        let mut args = std::env::args();
        let mut model_arg: Option<PathBuf> = None;
        let mut profile_layers = false;

        while let Some(arg) = args.next() {
            if arg == "--model" {
                if let Some(path) = args.next() {
                    model_arg = Some(PathBuf::from(path));
                }
            } else if arg == "--profile-layers" {
                profile_layers = true;
            }
        }

        // Fall back to environment variables
        if model_arg.is_none() {
            if let Ok(env_path) = std::env::var("RWKV_BENCH_MODEL") {
                model_arg = Some(PathBuf::from(env_path));
            } else if let Ok(env_path) = std::env::var("RWKV_MODEL") {
                model_arg = Some(PathBuf::from(env_path));
            }
        }

        let model = model_arg.expect(
            "Provide --model <path> or set RWKV_MODEL environment variable\n\
             Example: cargo bench --bench forward -- --model ./rwkv-10m.safetensors",
        );

        Self {
            model,
            profile_layers,
        }
    }
}

// =============================================================================
// Forward Pass Benchmark
// =============================================================================

fn bench_forward_token(c: &mut Criterion) {
    let args = &*BENCH_ARGS;
    let mut model = Model::load(&args.model).expect("Failed to load model for benchmark");
    let mut state = model.new_state();
    let mut scratch = ScratchBuffers::new(model.config());
    let mut profiler = args
        .profile_layers
        .then(|| LayerProfiler::new(model.config().num_layers));

    // Deterministic byte stream that exercises typical control flow.
    // Cycling through all 256 byte values ensures we hit all embedding rows.
    let tokens: Vec<u32> = (0..2048).map(|i| (i % 256) as u32).collect();
    let mut idx = 0usize;

    let mut group = c.benchmark_group("forward_token");
    group.throughput(Throughput::Elements(1));

    // Name includes model size for clarity in reports
    let config = model.config();
    let bench_name = format!("rwkv7_L{}_D{}", config.num_layers, config.hidden_size);

    group.bench_function(&bench_name, |b| {
        b.iter(|| {
            let token = tokens[idx];
            idx = (idx + 1) % tokens.len();
            let logits = if let Some(prof) = profiler.as_mut() {
                model.forward_with_profiler(&mut scratch, token, &mut state, prof)
            } else {
                model.forward(&mut scratch, token, &mut state)
            };
            // Return first logit to prevent dead code elimination
            black_box(logits[0])
        })
    });

    group.finish();

    // Print per-layer timings if profiling was enabled
    if let Some(prof) = profiler {
        let tokens = prof.tokens().max(1) as f64;
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!(
            "║  Per-layer timings ({:.0} tokens profiled)                      ║",
            tokens
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Layer │ Attention (ns) │ FFN (ns)    │ Total (ns)   ║");
        println!("╟────────┼────────────────┼─────────────┼──────────────╢");

        for (layer, timing) in prof.timings().iter().enumerate() {
            let attn = (timing.attention_ns as f64) / tokens;
            let ffn = (timing.ffn_ns as f64) / tokens;
            let total = attn + ffn;
            println!(
                "║  {:>4}  │ {:>12.1}  │ {:>9.1}  │ {:>10.1}  ║",
                layer, attn, ffn, total
            );
        }

        // Summary
        let total_attn: f64 = prof
            .timings()
            .iter()
            .map(|t| t.attention_ns as f64 / tokens)
            .sum();
        let total_ffn: f64 = prof
            .timings()
            .iter()
            .map(|t| t.ffn_ns as f64 / tokens)
            .sum();

        println!("╟────────┼────────────────┼─────────────┼──────────────╢");
        println!(
            "║  Total │ {:>12.1}  │ {:>9.1}  │ {:>10.1}  ║",
            total_attn,
            total_ffn,
            total_attn + total_ffn
        );
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\nNote: Timings are per-token averages in nanoseconds.");
    }
}

criterion_group!(benches, bench_forward_token);
criterion_main!(benches);
