use infotheory::api::{
    CompiledRateBackend, MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend,
    try_entropy_rate_backend,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const ALPHA: f64 = 0.03;
const WARMUP_ITERS: usize = 1;
const BENCH_ITERS: usize = 3;

fn data() -> Vec<u8> {
    std::fs::read("LICENSE-APACHE").expect("failed to read LICENSE-APACHE")
}

fn experts() -> Vec<MixtureExpertSpec> {
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

fn backend(kind: MixtureKind) -> RateBackend {
    RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(kind, experts()).with_alpha(ALPHA)),
    }
}

fn run_one(name: &str, kind: MixtureKind, bytes: &[u8]) {
    let backend = backend(kind).compile().expect("compile mixture backend");

    for _ in 0..WARMUP_ITERS {
        let h = entropy_rate_backend(bytes, &backend);
        black_box(h);
    }

    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..BENCH_ITERS {
        sum += entropy_rate_backend(bytes, &backend);
    }
    black_box(sum);
    let elapsed = start.elapsed().as_secs_f64();

    let ms = elapsed * 1e3 / (BENCH_ITERS as f64);
    let mib_s = ((bytes.len() * BENCH_ITERS) as f64) / elapsed / (1024.0 * 1024.0);
    let h = entropy_rate_backend(bytes, &backend);
    println!(
        "{name:>7}: {:>9.3} ms/iter | {:>8.3} MiB/s | H={:.9}",
        ms, mib_s, h
    );
}

fn main() {
    let bytes = data();
    println!(
        "Neural baseline benchmark on LICENSE-APACHE (n={})",
        bytes.len()
    );
    run_one("neural", MixtureKind::Neural, &bytes);
    run_one("switch", MixtureKind::Switching, &bytes);
    run_one("bayes", MixtureKind::Bayes, &bytes);
}
fn entropy_rate_backend(data: &[u8], backend: &CompiledRateBackend) -> f64 {
    try_entropy_rate_backend(data, backend).expect("entropy rate")
}
