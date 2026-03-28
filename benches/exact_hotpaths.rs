#![cfg(all(feature = "backend-rwkv", feature = "backend-mamba"))]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use infotheory::backends::ctw::FacContextTree;
use infotheory::backends::llm_policy::OptimizerKind;
use infotheory::coders::CoderType;
use infotheory::compression::{FramingMode, compress_rate_bytes};
use infotheory::mambazip::mamba1;
use infotheory::rwkvzip::rwkv7;
use infotheory::{MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend};
use std::sync::Arc;
use std::time::Duration;

const DATA_LEN: usize = 16 * 1024;

fn bench_data() -> Vec<u8> {
    let seeds = [
        "README.md",
        "LICENSE-APACHE",
        "Cargo.toml",
        "examples/two.json",
    ];
    let mut out = Vec::with_capacity(DATA_LEN);
    while out.len() < DATA_LEN {
        for path in seeds {
            let chunk = std::fs::read(path).expect("failed to read bench seed file");
            out.extend_from_slice(&chunk);
            if out.len() >= DATA_LEN {
                break;
            }
        }
    }
    out.truncate(DATA_LEN);
    out
}

fn softmax_pdf_from_logits(logits: &[f32], pdf: &mut Vec<f64>) {
    pdf.clear();
    pdf.reserve(logits.len());
    if logits.is_empty() {
        return;
    }
    let mut max_v = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        let x = ((v - max_v) as f64).exp();
        pdf.push(x);
        sum += x;
    }
    let inv = if sum > 0.0 && sum.is_finite() {
        1.0 / sum
    } else {
        1.0 / (logits.len() as f64)
    };
    for p in pdf.iter_mut() {
        *p *= inv;
    }
}

fn rwkv_cfg() -> rwkv7::Config {
    rwkv7::Config {
        vocab_size: 256,
        hidden_size: 64,
        num_layers: 1,
        num_heads: 1,
        head_dim: 64,
        intermediate_size: 64,
        layer_norm_eps: 1e-5,
        group_norm_eps: 64e-5,
        decay_low_rank: 16,
        a_low_rank: 16,
        v_low_rank: 16,
        g_low_rank: 16,
    }
}

fn mamba_cfg() -> mamba1::Config {
    mamba1::Config {
        vocab_size: 256,
        hidden_size: 64,
        num_layers: 1,
        inner_size: 128,
        state_size: 16,
        conv_kernel: 4,
        dt_rank: 16,
        layer_norm_eps: 1e-5,
    }
}

fn two_json_backend() -> RateBackend {
    let spec = MixtureSpec::new(
        MixtureKind::Neural,
        vec![
            MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Ctw { depth: 24 },
            },
            MixtureExpertSpec {
                name: Some("ppmd".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Ppmd {
                    order: 10,
                    memory_mb: 64,
                },
            },
            MixtureExpertSpec {
                name: Some("match".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Match {
                    hash_bits: 20,
                    min_len: 4,
                    max_len: 255,
                    base_mix: 0.02,
                    confidence_scale: 1.0,
                },
            },
        ],
    )
    .with_alpha(0.03);
    RateBackend::Mixture {
        spec: Arc::new(spec),
    }
}

fn bench_rwkv_direct(c: &mut Criterion) {
    let data = bench_data();
    let cfg = rwkv_cfg();
    let mut group = c.benchmark_group("rwkv_exact_hotpaths");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("forward_untraced", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let model = rwkv7::Model::new_random(cfg.clone(), 7).expect("random rwkv");
                let mut scratch = rwkv7::ScratchBuffers::new(&cfg);
                let mut state = rwkv7::State::new(&cfg);
                scratch.set_capture_train_trace(false);
                for &token in d.iter() {
                    let _ = model.forward(&mut scratch, token as u32, &mut state);
                }
                criterion::black_box(state)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("forward_traced", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let model = rwkv7::Model::new_random(cfg.clone(), 7).expect("random rwkv");
                let mut scratch = rwkv7::ScratchBuffers::new(&cfg);
                let mut state = rwkv7::State::new(&cfg);
                scratch.set_capture_train_trace(true);
                for &token in d.iter() {
                    let _ = model.forward(&mut scratch, token as u32, &mut state);
                }
                criterion::black_box((state, scratch.has_train_trace()))
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("train_scope_all", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let mut model = rwkv7::Model::new_random(cfg.clone(), 7).expect("random rwkv");
                let mut scratch = rwkv7::ScratchBuffers::new(&cfg);
                let mut state = rwkv7::State::new(&cfg);
                let mut adam = model.new_full_adam_state();
                let mut adam_t = 0usize;
                let mut pdf = Vec::with_capacity(cfg.vocab_size);
                scratch.set_capture_train_trace(true);
                for (idx, &token) in d.iter().enumerate() {
                    let logits = model.forward(&mut scratch, token as u32, &mut state);
                    softmax_pdf_from_logits(logits, &mut pdf);
                    let target = d[(idx + 1) % d.len()];
                    model
                        .online_train_step_bptt1(
                            &mut scratch,
                            &state,
                            target,
                            &pdf,
                            rwkv7::TrainScopeMask::all(),
                            OptimizerKind::Adam,
                            0.001,
                            1.0,
                            &mut adam_t,
                            Some(&mut adam),
                            None,
                            None,
                            None,
                        )
                        .expect("rwkv train");
                }
                criterion::black_box((state, adam_t))
            });
        },
    );

    group.finish();
}

fn bench_mamba_direct(c: &mut Criterion) {
    let data = bench_data();
    let cfg = mamba_cfg();
    let mut group = c.benchmark_group("mamba_exact_hotpaths");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("forward_untraced", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let model = mamba1::Model::new_random(cfg.clone(), 11).expect("random mamba");
                let mut scratch = mamba1::ScratchBuffers::new(&cfg);
                let mut state = mamba1::State::new(&cfg);
                scratch.set_capture_train_trace(false);
                for &token in d.iter() {
                    let _ = model.forward(&mut scratch, token as u32, &mut state);
                }
                criterion::black_box(state)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("forward_traced", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let model = mamba1::Model::new_random(cfg.clone(), 11).expect("random mamba");
                let mut scratch = mamba1::ScratchBuffers::new(&cfg);
                let mut state = mamba1::State::new(&cfg);
                scratch.set_capture_train_trace(true);
                for &token in d.iter() {
                    let _ = model.forward(&mut scratch, token as u32, &mut state);
                }
                criterion::black_box((state, scratch.has_train_trace()))
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("train_scope_all", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let mut model = mamba1::Model::new_random(cfg.clone(), 11).expect("random mamba");
                let mut scratch = mamba1::ScratchBuffers::new(&cfg);
                let mut state = mamba1::State::new(&cfg);
                let mut adam = model.new_full_adam_state();
                let mut adam_t = 0usize;
                let mut pdf = Vec::with_capacity(cfg.vocab_size);
                scratch.set_capture_train_trace(true);
                for (idx, &token) in d.iter().enumerate() {
                    let logits = model.forward(&mut scratch, token as u32, &mut state);
                    softmax_pdf_from_logits(logits, &mut pdf);
                    let target = d[(idx + 1) % d.len()];
                    model
                        .online_train_step_bptt1(
                            &mut scratch,
                            &state,
                            target,
                            &pdf,
                            mamba1::TrainScopeMask::all(),
                            OptimizerKind::Adam,
                            0.001,
                            0.0,
                            &mut adam_t,
                            Some(&mut adam),
                            None,
                            None,
                            None,
                        )
                        .expect("mamba train");
                }
                criterion::black_box((state, adam_t))
            });
        },
    );

    group.finish();
}

fn bench_ctw_direct(c: &mut Criterion) {
    let data = bench_data();
    let mut group = c.benchmark_group("ctw_exact_hotpaths");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("predict_update_fac", data.len()),
        &data,
        |b, d| {
            b.iter(|| {
                let mut tree = FacContextTree::new(32, 8);
                for &byte in d.iter() {
                    for bit_idx in 0..8usize {
                        let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                        let _ = tree.predict(bit, bit_idx);
                        tree.update(bit, bit_idx);
                    }
                }
                criterion::black_box(tree.get_log_block_probability())
            });
        },
    );

    group.finish();
}

fn bench_two_json_end_to_end(c: &mut Criterion) {
    let data = bench_data();
    let backend = two_json_backend();
    let mut group = c.benchmark_group("two_json_end_to_end");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_with_input(BenchmarkId::new("rate_ac", data.len()), &data, |b, d| {
        b.iter(|| {
            let encoded = compress_rate_bytes(d, &backend, -1, CoderType::AC, FramingMode::Raw)
                .expect("two.json compression bench failed");
            criterion::black_box(encoded.len())
        });
    });
    group.finish();
}

fn bench_exact_hotpaths(c: &mut Criterion) {
    bench_rwkv_direct(c);
    bench_mamba_direct(c);
    bench_ctw_direct(c);
    bench_two_json_end_to_end(c);
}

criterion_group! {
    name = exact_hotpaths;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = bench_exact_hotpaths
}
criterion_main!(exact_hotpaths);
