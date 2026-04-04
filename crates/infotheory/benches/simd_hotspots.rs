use infotheory::api::{
    MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend, try_entropy_rate_backend,
};
use infotheory::coders::ac::softmax_pdf_floor_inplace;
use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const ALPHA: f64 = 0.03;
const WARMUP_ITERS_DEFAULT: usize = 2;
const BENCH_ITERS_DEFAULT: usize = 8;
const EXPAND_FACTOR_DEFAULT: usize = 2;

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
        MixtureExpertSpec {
            name: Some("fac".to_string()),
            log_prior: 0.0,
            max_order: -1,
            backend: RateBackend::FacCtw {
                base_depth: 16,
                encoding_bits: 8,
                num_percept_bits: 8,
            },
        },
        MixtureExpertSpec {
            name: Some("rosa".to_string()),
            log_prior: 0.0,
            max_order: -1,
            backend: RateBackend::RosaPlus,
        },
    ]
}

fn bench_neural_mixture(data: &[u8], warmup_iters: usize, bench_iters: usize) {
    let backend = RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(MixtureKind::Neural, make_experts()).with_alpha(ALPHA)),
    };

    for _ in 0..warmup_iters {
        let h = entropy_rate_backend(data, -1, &backend);
        black_box(h);
    }

    let start = Instant::now();
    let mut sink = 0.0;
    for _ in 0..bench_iters {
        sink += entropy_rate_backend(data, -1, &backend);
    }
    black_box(sink);
    let elapsed = start.elapsed().as_secs_f64();

    let ms_per_iter = (elapsed * 1e3) / (bench_iters as f64);
    let mib_per_s = ((data.len() * bench_iters) as f64) / elapsed / (1024.0 * 1024.0);

    println!(
        "neural_mixture      : {ms_per_iter:>9.3} ms/iter | {mib_per_s:>9.3} MiB/s | n={} bytes",
        data.len()
    );
}

fn bench_ac_softmax_floor_256(warmup_iters: usize, bench_iters: usize) {
    const N: usize = 256;
    let mut logits = vec![0.0f32; N];
    for (i, v) in logits.iter_mut().enumerate() {
        let x = i as f32;
        *v = (x * 0.073).sin() * 3.5 + (x * 0.019).cos() * 1.7;
    }

    let mut pdf = vec![0.0f64; N];
    for _ in 0..warmup_iters {
        softmax_pdf_floor_inplace(&logits, N, &mut pdf);
        black_box(pdf[0]);
    }

    let start = Instant::now();
    let mut checksum = 0.0f64;
    for iter in 0..bench_iters {
        logits[iter % N] += 0.001;
        softmax_pdf_floor_inplace(&logits, N, &mut pdf);
        checksum += pdf[(iter * 17) % N];
    }
    black_box(checksum);
    let elapsed = start.elapsed().as_secs_f64();

    let us_per_iter = (elapsed * 1e6) / (bench_iters as f64);
    let calls_per_s = (bench_iters as f64) / elapsed;
    let sum: f64 = pdf.iter().sum();

    println!(
        "ac_softmax_floor256 : {us_per_iter:>9.3} us/call | {calls_per_s:>9.1} calls/s | sum={sum:.12}"
    );
}

fn main() {
    let warmup_iters = env_usize("SIMD_BENCH_WARMUP", WARMUP_ITERS_DEFAULT);
    let bench_iters = env_usize("SIMD_BENCH_ITERS", BENCH_ITERS_DEFAULT);
    let expand_factor = env_usize("SIMD_BENCH_EXPAND", EXPAND_FACTOR_DEFAULT);

    println!(
        "SIMD hotspot baseline benchmark: warmup={warmup_iters}, iters={bench_iters}, expand={expand_factor}"
    );

    let data = source_data(expand_factor);
    bench_neural_mixture(&data, warmup_iters, bench_iters);
    bench_ac_softmax_floor_256(warmup_iters * 128, bench_iters * 20_000);
}
fn entropy_rate_backend(data: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
    try_entropy_rate_backend(data, max_order, backend).expect("entropy rate")
}
