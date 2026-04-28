use infotheory::api::{
    CompiledRateBackend, MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend,
    try_entropy_rate_backend,
};
use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

const ALPHA: f64 = 0.03;
const FADING_DECAY: f64 = 0.995;
const WARMUP_ITERS_DEFAULT: usize = 1;
const BENCH_ITERS_DEFAULT: usize = 2;
const EXPAND_FACTOR_DEFAULT: usize = 1;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn source_data(expand_factor: usize) -> Vec<u8> {
    let raw =
        std::fs::read("LICENSE-APACHE").expect("failed to read LICENSE-APACHE from repo root");
    assert!(!raw.is_empty(), "LICENSE-APACHE must not be empty");
    let mut expanded = Vec::with_capacity(raw.len() * expand_factor);
    for _ in 0..expand_factor {
        expanded.extend_from_slice(&raw);
    }
    expanded
}

fn make_experts() -> Vec<MixtureExpertSpec> {
    vec![
        MixtureExpertSpec::new(RateBackend::FacCtw {
            base_depth: 16,
            encoding_bits: 8,
            num_percept_bits: 8,
        })
        .with_name("fac"),
        MixtureExpertSpec::new(RateBackend::RosaPlus { max_order: -1 }).with_name("rosa"),
    ]
}

fn make_spec(kind: MixtureKind) -> MixtureSpec {
    let experts = make_experts();
    let spec = MixtureSpec::new(kind, experts).with_alpha(ALPHA);
    if matches!(kind, MixtureKind::FadingBayes) {
        spec.with_decay(FADING_DECAY)
    } else {
        spec
    }
}

fn bench_kind(
    name: &str,
    kind: MixtureKind,
    data: &[u8],
    warmup_iters: usize,
    bench_iters: usize,
) -> Duration {
    let backend = RateBackend::Mixture {
        spec: Arc::new(make_spec(kind)),
    }
    .compile()
    .expect("compile mixture backend");

    for _ in 0..warmup_iters {
        let h = entropy_rate_backend(data, &backend);
        black_box(h);
    }

    let start = Instant::now();
    let mut sink = 0.0;
    for _ in 0..bench_iters {
        let h = entropy_rate_backend(data, &backend);
        sink += h;
    }
    black_box(sink);
    let elapsed = start.elapsed();

    let secs = elapsed.as_secs_f64().max(1e-12);
    let bytes_total = (data.len() * bench_iters) as f64;
    let mib_per_sec = bytes_total / secs / (1024.0 * 1024.0);
    let ms_per_iter = (elapsed.as_secs_f64() * 1e3) / (bench_iters as f64);

    println!(
        "{name:>8}: {ms_per_iter:>9.3} ms/iter | {mib_per_sec:>9.3} MiB/s | n={} bytes",
        data.len()
    );

    elapsed
}

fn main() {
    let warmup_iters = env_usize("MIX_BENCH_WARMUP", WARMUP_ITERS_DEFAULT);
    let bench_iters = env_usize("MIX_BENCH_ITERS", BENCH_ITERS_DEFAULT);
    let expand_factor = env_usize("MIX_BENCH_EXPAND", EXPAND_FACTOR_DEFAULT);

    let data = source_data(expand_factor);
    println!(
        "Mixture backend benchmark on LICENSE-APACHE (expanded x{expand_factor}), warmup={warmup_iters}, iters={bench_iters}"
    );

    let benches = [
        ("bayes", MixtureKind::Bayes),
        ("fading", MixtureKind::FadingBayes),
        ("switch", MixtureKind::Switching),
        ("mdl", MixtureKind::Mdl),
        ("neural", MixtureKind::Neural),
    ];

    let mut total = Duration::ZERO;
    for (name, kind) in benches {
        total += bench_kind(name, kind, &data, warmup_iters, bench_iters);
    }
    println!("total elapsed: {:.3} s", total.as_secs_f64());
}
fn entropy_rate_backend(data: &[u8], backend: &CompiledRateBackend) -> f64 {
    try_entropy_rate_backend(data, backend).expect("entropy rate")
}
