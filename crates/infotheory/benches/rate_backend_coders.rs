#![cfg(feature = "backend-rwkv")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use infotheory::api::{
    CompiledRateBackend, MixtureExpertSpec, MixtureKind, MixtureSpec, ParticleSpec, RateBackend,
};
use infotheory::coders::CoderType;
use infotheory::compression::{FramingMode, compress_rate_bytes};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

const DATA_LEN: usize = 10 * 1024;
const CTW_DEPTH: usize = 6;
const MIX_ALPHA: f64 = 0.03;
const MIX_DECAY: f64 = 0.995;
const RWKV_BENCH_METHOD: &str =
    "cfg:hidden=64,intermediate=64,layers=1,train=none,lr=0.0;policy:schedule=0..100:infer";

fn bench_data() -> &'static [u8] {
    static DATA: OnceLock<Vec<u8>> = OnceLock::new();
    DATA.get_or_init(|| {
        let seed = std::fs::read("LICENSE-APACHE")
            .expect("failed to read LICENSE-APACHE from repository root");
        assert!(!seed.is_empty(), "LICENSE-APACHE must not be empty");

        let mut out = Vec::with_capacity(DATA_LEN);
        while out.len() < DATA_LEN {
            out.extend_from_slice(&seed);
        }
        out.truncate(DATA_LEN);
        out
    })
}

fn particle_spec_from_example() -> ParticleSpec {
    let mut spec = ParticleSpec::default();
    spec.num_particles = 8;
    spec.context_window = 64;
    spec.unroll_steps = 2;
    spec.num_cells = 6;
    spec.cell_dim = 12;
    spec.num_rules = 2;
    spec.selector_hidden = 32;
    spec.rule_hidden = 32;
    spec.noise_dim = 8;
    spec.deterministic = true;
    spec.enable_noise = true;
    spec.noise_scale = 0.08;
    spec.noise_anneal_steps = 8192;
    spec.learning_rate_readout = 0.0003;
    spec.learning_rate_selector = 0.0;
    spec.learning_rate_rule = 0.0;
    spec.bptt_depth = 1;
    spec.optimizer_momentum = 0.05;
    spec.grad_clip = 1.0;
    spec.state_clip = 8.0;
    spec.forget_lambda = 0.0;
    spec.resample_threshold = 0.5;
    spec.mutate_fraction = 0.25;
    spec.mutate_scale = 0.01;
    spec.mutate_model_params = false;
    spec.diagnostics_interval = 0;
    spec.min_prob = 2f64.powi(-24);
    spec.seed = 42;
    spec
}

fn compile_rate_backend(backend: RateBackend) -> CompiledRateBackend {
    backend.compile().expect("compile benchmark rate backend")
}

fn individual_backends() -> Vec<(&'static str, CompiledRateBackend)> {
    vec![
        ("rosaplus-o-1", compile_rate_backend(RateBackend::RosaPlus)),
        (
            "ctw-d6",
            compile_rate_backend(RateBackend::Ctw { depth: CTW_DEPTH }),
        ),
        (
            "rwkv64x64",
            compile_rate_backend(RateBackend::Rwkv7Method {
                method: infotheory::rwkvzip::parse_method_spec(RWKV_BENCH_METHOD)
                    .expect("rwkv benchmark method must be valid"),
            }),
        ),
        (
            "particle-like-example",
            compile_rate_backend(RateBackend::Particle {
                spec: Arc::new(particle_spec_from_example()),
            }),
        ),
    ]
}

fn make_expert(name: &str, backend: RateBackend) -> MixtureExpertSpec {
    MixtureExpertSpec::new(backend).with_name(name)
}

fn mixture_backends() -> Vec<(&'static str, CompiledRateBackend)> {
    let rosa = make_expert("rosa", RateBackend::RosaPlus);
    let ctw = make_expert("ctw", RateBackend::Ctw { depth: CTW_DEPTH });
    let rwkv = make_expert(
        "rwkv64x64",
        RateBackend::Rwkv7Method {
            method: infotheory::rwkvzip::parse_method_spec(RWKV_BENCH_METHOD)
                .expect("rwkv benchmark method must be valid"),
        },
    );
    let particle = make_expert(
        "particle",
        RateBackend::Particle {
            spec: Arc::new(particle_spec_from_example()),
        },
    );

    let mk = |kind: MixtureKind, experts: Vec<MixtureExpertSpec>| {
        let mut spec = MixtureSpec::new(kind, experts).with_alpha(MIX_ALPHA);
        if matches!(kind, MixtureKind::FadingBayes) {
            spec = spec.with_decay(MIX_DECAY);
        }
        compile_rate_backend(RateBackend::Mixture {
            spec: Arc::new(spec),
        })
    };

    vec![
        (
            "mix-bayes-rosa-rwkv",
            mk(MixtureKind::Bayes, vec![rosa.clone(), rwkv.clone()]),
        ),
        (
            "mix-fading-rosa-particle",
            mk(
                MixtureKind::FadingBayes,
                vec![rosa.clone(), particle.clone()],
            ),
        ),
        (
            "mix-switch-ctw-rwkv",
            mk(MixtureKind::Switching, vec![ctw.clone(), rwkv.clone()]),
        ),
        (
            "mix-neural-all4",
            mk(MixtureKind::Neural, vec![rosa, ctw, rwkv, particle]),
        ),
    ]
}

fn bench_matrix(c: &mut Criterion) {
    let data = bench_data();

    let mut group = c.benchmark_group("rate_coders_individual");
    group.throughput(Throughput::Bytes(data.len() as u64));

    for (label, coder) in [("ac", CoderType::AC), ("rans", CoderType::RANS)] {
        for (backend_name, backend) in individual_backends() {
            group.bench_with_input(
                BenchmarkId::new(format!("{label}/{backend_name}"), data.len()),
                &backend,
                |b, rate_backend| {
                    b.iter(|| {
                        let out =
                            compress_rate_bytes(data, rate_backend, -1, coder, FramingMode::Raw)
                                .expect("compression benchmark failed");
                        criterion::black_box(out.len())
                    });
                },
            );
        }
    }
    group.finish();

    let mut mix_group = c.benchmark_group("rate_coders_mixtures");
    mix_group.throughput(Throughput::Bytes(data.len() as u64));

    for (label, coder) in [("ac", CoderType::AC), ("rans", CoderType::RANS)] {
        for (mix_name, backend) in mixture_backends() {
            mix_group.bench_with_input(
                BenchmarkId::new(format!("{label}/{mix_name}"), data.len()),
                &backend,
                |b, rate_backend| {
                    b.iter(|| {
                        let out =
                            compress_rate_bytes(data, rate_backend, -1, coder, FramingMode::Raw)
                                .expect("mixture compression benchmark failed");
                        criterion::black_box(out.len())
                    });
                },
            );
        }
    }
    mix_group.finish();
}

criterion_group! {
    name = rate_backend_coders;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = bench_matrix
}
criterion_main!(rate_backend_coders);
