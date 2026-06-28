#![cfg(feature = "backend-rwkv")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use infotheory::api::{CompiledRateBackend, RateBackend, try_entropy_rate_backend};
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
    RateBackend::Rwkv7Method {
        method: infotheory::rwkvzip::parse_method_spec(method)
            .expect("rwkv benchmark method must be valid"),
    }
}

fn bench_rwkv_online_train_full(c: &mut Criterion) {
    let data = bench_data();
    let infer = backend(
        "cfg:hidden=128,layers=2,intermediate=256,decay_rank=16,a_rank=16,v_rank=16,g_rank=16,seed=7,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer",
    )
    .compile()
    .expect("compile rwkv infer backend");
    let train_full = backend(
        "cfg:hidden=128,layers=2,intermediate=256,decay_rank=16,a_rank=16,v_rank=16,g_rank=16,seed=7,train=adam,lr=0.001,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)",
    )
    .compile()
    .expect("compile rwkv train backend");

    let mut group = c.benchmark_group("rwkv_online_train_full");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("infer", data.len()),
        &infer,
        |b, backend| {
            b.iter(|| {
                let h = entropy_rate_backend(&data, backend);
                criterion::black_box(h)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("train_scope_all", data.len()),
        &train_full,
        |b, backend| {
            b.iter(|| {
                let h = entropy_rate_backend(&data, backend);
                criterion::black_box(h)
            });
        },
    );

    group.finish();
}

criterion_group! {
    name = rwkv_online_train_full;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = bench_rwkv_online_train_full
}
criterion_main!(rwkv_online_train_full);
fn entropy_rate_backend(data: &[u8], backend: &CompiledRateBackend) -> f64 {
    try_entropy_rate_backend(data, backend).expect("entropy rate")
}
