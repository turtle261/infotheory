#![cfg(feature = "backend-mamba")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use infotheory::coders::CoderType;
use infotheory::compression::{FramingMode, compress_rate_bytes};
use infotheory::api::{RateBackend, try_entropy_rate_backend};
use std::time::Duration;

const DATA_LEN: usize = 16 * 1024;

fn bench_data() -> Vec<u8> {
    let seed = std::fs::read("README.md").expect("failed to read README.md from repo root");
    let mut out = Vec::with_capacity(DATA_LEN);
    while out.len() < DATA_LEN {
        out.extend_from_slice(&seed);
    }
    out.truncate(DATA_LEN);
    out
}

fn mamba_backend() -> RateBackend {
    RateBackend::MambaMethod {
        method: "cfg:hidden=128,layers=2,intermediate=256,state=16,conv=4,dt_rank=16,train=none,seed=7;policy:schedule=0..100:infer".to_string(),
    }
}

fn bench_mamba(c: &mut Criterion) {
    let data = bench_data();
    let backend = mamba_backend();

    let mut h_group = c.benchmark_group("mamba_entropy");
    h_group.throughput(Throughput::Bytes(data.len() as u64));
    h_group.bench_with_input(
        BenchmarkId::new("entropy_rate", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let h = entropy_rate_backend(d, -1, &backend);
                criterion::black_box(h)
            });
        },
    );
    h_group.finish();

    let mut c_group = c.benchmark_group("mamba_rate_coders");
    c_group.throughput(Throughput::Bytes(data.len() as u64));

    for (label, coder) in [("ac", CoderType::AC), ("rans", CoderType::RANS)] {
        c_group.bench_with_input(
            BenchmarkId::new(format!("{label}/compress"), data.len()),
            &data,
            |b, d| {
                b.iter(|| {
                    let out = compress_rate_bytes(d, &backend, -1, coder, FramingMode::Raw)
                        .expect("mamba rate compression benchmark failed");
                    criterion::black_box(out.len())
                });
            },
        );
    }
    c_group.finish();
}

criterion_group! {
    name = mamba_rate;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = bench_mamba
}
criterion_main!(mamba_rate);
fn entropy_rate_backend(data: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
    try_entropy_rate_backend(data, max_order, backend).expect("entropy rate")
}
