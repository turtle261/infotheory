use infotheory::{MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend, entropy_rate_backend};
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

fn backend(kind: MixtureKind) -> RateBackend {
    RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(kind, experts()).with_alpha(ALPHA)),
    }
}

fn run_one(name: &str, kind: MixtureKind, bytes: &[u8]) {
    let backend = backend(kind);

    for _ in 0..WARMUP_ITERS {
        let h = entropy_rate_backend(bytes, -1, &backend);
        black_box(h);
    }

    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..BENCH_ITERS {
        sum += entropy_rate_backend(bytes, -1, &backend);
    }
    black_box(sum);
    let elapsed = start.elapsed().as_secs_f64();

    let ms = elapsed * 1e3 / (BENCH_ITERS as f64);
    let mib_s = ((bytes.len() * BENCH_ITERS) as f64) / elapsed / (1024.0 * 1024.0);
    let h = entropy_rate_backend(bytes, -1, &backend);
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
