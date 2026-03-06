#![cfg(feature = "backend-mamba")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use infotheory::{RateBackend, entropy_rate_backend};
use std::time::Duration;

const DATA_LEN: usize = 64 * 1024;

fn bench_data() -> Vec<u8> {
    let seed = std::fs::read("README.md").expect("failed to read README.md");
    let mut out = Vec::with_capacity(DATA_LEN);
    while out.len() < DATA_LEN {
        out.extend_from_slice(&seed);
    }
    out.truncate(DATA_LEN);
    out
}

fn backend(method: &str) -> RateBackend {
    RateBackend::MambaMethod {
        method: method.to_string(),
    }
}

fn bench_mamba_online_train_full(c: &mut Criterion) {
    let data = bench_data();
    let infer = backend(
        "cfg:hidden=128,layers=2,intermediate=256,state=16,conv=4,dt_rank=16,seed=7,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer",
    );
    let train_full = backend(
        "cfg:hidden=128,layers=2,intermediate=256,state=16,conv=4,dt_rank=16,seed=7,train=adam,lr=0.001,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)",
    );

    let mut group = c.benchmark_group("mamba_online_train_full");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("infer", data.len()),
        &infer,
        |b, backend| {
            b.iter(|| {
                let h = entropy_rate_backend(&data, -1, backend);
                criterion::black_box(h)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("train_scope_head_bias", data.len()),
        &train_full,
        |b, backend| {
            b.iter(|| {
                let h = entropy_rate_backend(&data, -1, backend);
                criterion::black_box(h)
            });
        },
    );

    group.finish();
}

criterion_group! {
    name = mamba_online_train_full;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = bench_mamba_online_train_full
}
criterion_main!(mamba_online_train_full);
