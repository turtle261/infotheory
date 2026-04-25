use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use infotheory::api::{
    CompressionBackend, MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend,
    RateBackendSession, try_compress_size_backend, try_entropy_rate_backend,
};
use infotheory::coders::CoderType;
use infotheory::compression::FramingMode;
use std::sync::Arc;

fn short_bench_data() -> Vec<u8> {
    let seed = b"compiled-plan-overhead";
    let mut out = Vec::with_capacity(64);
    while out.len() < 64 {
        out.extend_from_slice(seed);
    }
    out.truncate(64);
    out
}

fn bench_case(c: &mut Criterion, label: &str, rate_backend: RateBackend, data: &[u8]) {
    let compiled_rate = rate_backend
        .compile()
        .expect("compile benchmark rate backend");
    let compression_backend = CompressionBackend::Rate {
        rate_backend: rate_backend.clone(),
        coder: CoderType::AC,
        framing: FramingMode::Framed,
    };
    let compiled_compression = compression_backend
        .compile()
        .expect("compile benchmark compression backend");

    let mut setup_group = c.benchmark_group("compiled_plan_overhead_setup");
    setup_group.bench_function(format!("{label}_rate_compiled_clone"), |b| {
        b.iter(|| {
            black_box(compiled_rate.clone());
        });
    });
    setup_group.bench_function(format!("{label}_rate_compile_each_call"), |b| {
        b.iter(|| {
            black_box(&rate_backend)
                .clone()
                .compile()
                .expect("compile-each-call rate setup benchmark");
        });
    });
    setup_group.bench_function(format!("{label}_compression_compiled_clone"), |b| {
        b.iter(|| {
            black_box(compiled_compression.clone());
        });
    });
    setup_group.bench_function(format!("{label}_compression_compile_each_call"), |b| {
        b.iter(|| {
            black_box(&compression_backend)
                .clone()
                .compile()
                .expect("compile-each-call compression setup benchmark");
        });
    });
    setup_group.finish();

    let mut entropy_group = c.benchmark_group("compiled_plan_overhead_entropy");
    entropy_group.throughput(Throughput::Bytes(data.len() as u64));
    entropy_group.bench_with_input(
        BenchmarkId::new(format!("{label}_compiled_reuse"), data.len()),
        data,
        |b, d| {
            b.iter(|| {
                try_entropy_rate_backend(black_box(d), -1, black_box(&compiled_rate))
                    .expect("compiled entropy benchmark");
            });
        },
    );
    entropy_group.bench_with_input(
        BenchmarkId::new(format!("{label}_compile_each_call"), data.len()),
        data,
        |b, d| {
            b.iter(|| {
                let compiled = black_box(&rate_backend)
                    .compile()
                    .expect("compile-each-call rate backend");
                try_entropy_rate_backend(black_box(d), -1, black_box(&compiled))
                    .expect("compile-each-call entropy benchmark");
            });
        },
    );
    entropy_group.finish();

    let mut compression_group = c.benchmark_group("compiled_plan_overhead_compression");
    compression_group.throughput(Throughput::Bytes(data.len() as u64));
    compression_group.bench_with_input(
        BenchmarkId::new(format!("{label}_compiled_reuse"), data.len()),
        data,
        |b, d| {
            b.iter(|| {
                try_compress_size_backend(black_box(d), black_box(&compiled_compression))
                    .expect("compiled compression benchmark");
            });
        },
    );
    compression_group.bench_with_input(
        BenchmarkId::new(format!("{label}_compile_each_call"), data.len()),
        data,
        |b, d| {
            b.iter(|| {
                let compiled = black_box(&compression_backend)
                    .compile()
                    .expect("compile-each-call compression backend");
                try_compress_size_backend(black_box(d), black_box(&compiled))
                    .expect("compile-each-call compression benchmark");
            });
        },
    );
    compression_group.finish();

    let mut session_group = c.benchmark_group("compiled_plan_overhead_session");
    session_group.throughput(Throughput::Bytes(data.len() as u64));
    session_group.bench_with_input(
        BenchmarkId::new(format!("{label}_compiled_reuse"), data.len()),
        data,
        |b, d| {
            b.iter(|| {
                let mut session =
                    RateBackendSession::from_backend(black_box(compiled_rate.clone()), -1, None)
                        .expect("compiled session benchmark");
                session.observe(black_box(d));
                let mut log_probs = [0.0; 256];
                session.fill_log_probs(&mut log_probs);
                black_box(log_probs);
            });
        },
    );
    session_group.bench_with_input(
        BenchmarkId::new(format!("{label}_compile_each_call"), data.len()),
        data,
        |b, d| {
            b.iter(|| {
                let mut session =
                    RateBackendSession::from_spec(black_box(rate_backend.clone()), -1, None)
                        .expect("compile-each-call session benchmark");
                session.observe(black_box(d));
                let mut log_probs = [0.0; 256];
                session.fill_log_probs(&mut log_probs);
                black_box(log_probs);
            });
        },
    );
    session_group.finish();
}

#[cfg(all(
    feature = "backend-mixture",
    feature = "backend-ctw",
    feature = "backend-match"
))]
fn nested_mixture_backend() -> RateBackend {
    RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(
            MixtureKind::Bayes,
            vec![
                {
                    let mut expert = MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 });
                    expert.name = Some("ctw".to_string());
                    expert.log_prior = 0.0;
                    expert.max_order = -1;
                    expert
                },
                {
                    let mut expert = MixtureExpertSpec::new(RateBackend::Match {
                        hash_bits: 18,
                        min_len: 4,
                        max_len: 64,
                        base_mix: 0.02,
                        confidence_scale: 1.0,
                    });
                    expert.name = Some("match".to_string());
                    expert.log_prior = -0.15;
                    expert.max_order = -1;
                    expert
                },
            ],
        )),
    }
}

fn bench_compiled_plan_overhead(c: &mut Criterion) {
    let data = short_bench_data();
    bench_case(c, "ctw", RateBackend::Ctw { depth: 8 }, &data);
    #[cfg(all(
        feature = "backend-mixture",
        feature = "backend-ctw",
        feature = "backend-match"
    ))]
    bench_case(c, "mixture", nested_mixture_backend(), &data);
}

criterion_group!(compiled_plan_overhead, bench_compiled_plan_overhead);
criterion_main!(compiled_plan_overhead);
