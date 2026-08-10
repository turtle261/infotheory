#![cfg(feature = "all-backends")]

use infotheory::api::{
    CalibratedSpec, CalibrationContextKind, MixtureExpertSpec, MixtureKind, MixtureScheduleMode,
    MixtureSpec, RateBackend, try_entropy_rate_backend as try_entropy_rate_backend_compiled,
};
use std::sync::Arc;

fn try_entropy_rate_backend(data: &[u8], backend: &RateBackend) -> Result<f64, String> {
    let compiled = backend.compile().map_err(|err| err.to_string())?;
    try_entropy_rate_backend_compiled(data, &compiled).map_err(|err| err.to_string())
}

#[test]
fn mixture_single_expert_matches_backend() {
    let data = b"abababababababababababababababab";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = try_entropy_rate_backend(data, &base).expect("base rate");

    let spec = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![MixtureExpertSpec::new(base.clone())],
    );
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = try_entropy_rate_backend(data, &mix_backend).expect("mix rate");

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn mixture_single_sequitur_expert_matches_backend() {
    let data = b"abcabcabcabcabcabc";
    let base = RateBackend::Sequitur { context_bytes: 32 };
    let base_rate = try_entropy_rate_backend(data, &base).expect("base rate");

    let spec = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![MixtureExpertSpec::new(base.clone()).with_name("sequitur")],
    );
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = try_entropy_rate_backend(data, &mix_backend).expect("mix rate");

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[cfg(feature = "backend-rwkv")]
#[test]
fn rwkv7_mixture_single_expert_matches_backend_with_tbptt() {
    let data = b"abcdefghij";
    let base = RateBackend::Rwkv7Method {
        method: infotheory::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=37,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=8,clip=0,momentum=0.9)").expect("rwkv method spec"),
    };
    let base_rate = try_entropy_rate_backend(data, &base).expect("base rate");

    let spec = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![MixtureExpertSpec::new(base.clone()).with_name("rwkv7")],
    );
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = try_entropy_rate_backend(data, &mix_backend).expect("mix rate");

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn mixture_recursive_expert_matches_backend() {
    let data = b"01010101010101010101010101010101";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = try_entropy_rate_backend(data, &base).expect("base rate");

    let inner = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![MixtureExpertSpec::new(base.clone()).with_name("ctw")],
    );
    let outer = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![
            MixtureExpertSpec::new(RateBackend::Mixture {
                spec: Arc::new(inner),
            })
            .with_name("inner"),
        ],
    );
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(outer),
    };
    let mix_rate = try_entropy_rate_backend(data, &mix_backend).expect("mix rate");

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn neural_mixture_single_expert_matches_backend() {
    let data = b"abababababababababababababababab";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = try_entropy_rate_backend(data, &base).expect("base rate");

    let spec = MixtureSpec::new(
        MixtureKind::Neural,
        vec![MixtureExpertSpec::new(base.clone())],
    )
    .with_alpha(0.05);
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = try_entropy_rate_backend(data, &mix_backend).expect("mix rate");

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn logistic_mixture_single_expert_matches_backend() {
    let data = b"abababababababababababababababab";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = try_entropy_rate_backend(data, &base).expect("base rate");

    let spec = MixtureSpec::new(
        MixtureKind::Logistic,
        vec![MixtureExpertSpec::new(base.clone())],
    )
    .with_alpha(0.03);
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = try_entropy_rate_backend(data, &mix_backend).expect("mix rate");

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn convex_mixture_single_expert_matches_backend() {
    let data = b"abababababababababababababababab";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = try_entropy_rate_backend(data, &base).expect("base rate");

    let spec = MixtureSpec::new(
        MixtureKind::Convex,
        vec![MixtureExpertSpec::new(base.clone())],
    )
    .with_alpha(1.25);
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = try_entropy_rate_backend(data, &mix_backend).expect("mix rate");

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn switching_theorem_schedule_backend_executes() {
    let data = b"abababababababababababababababab";
    let spec = MixtureSpec::new(
        MixtureKind::Switching,
        vec![
            MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 }).with_name("ctw"),
            MixtureExpertSpec::new(RateBackend::Match {
                hash_bits: 18,
                min_len: 3,
                max_len: 96,
                base_mix: 0.03,
                confidence_scale: 1.0,
            })
            .with_name("match"),
        ],
    )
    .with_schedule(MixtureScheduleMode::Theorem)
    .with_alpha(0.99);
    let backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let rate = try_entropy_rate_backend(data, &backend).expect("rate");
    assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
}

#[test]
fn convex_theorem_schedule_backend_executes() {
    let data = b"abababababababababababababababab";
    let spec = MixtureSpec::new(
        MixtureKind::Convex,
        vec![
            MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 }).with_name("ctw"),
            MixtureExpertSpec::new(RateBackend::FacCtw {
                base_depth: 8,
                num_percept_bits: 8,
                encoding_bits: 8,
                msb_first: None,
            })
            .with_name("fac"),
        ],
    )
    .with_schedule(MixtureScheduleMode::Theorem)
    .with_alpha(7.5);
    let backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let rate = try_entropy_rate_backend(data, &backend).expect("rate");
    assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
}

#[test]
fn neural_mixture_supports_nested_mixture_expert() {
    let data = b"abracadabra abracadabra abracadabra";
    let inner = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![
            MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 }).with_name("ctw"),
            MixtureExpertSpec::new(RateBackend::FacCtw {
                base_depth: 8,
                num_percept_bits: 8,
                encoding_bits: 8,
                msb_first: None,
            })
            .with_name("fac"),
        ],
    );

    let outer = MixtureSpec::new(
        MixtureKind::Neural,
        vec![
            MixtureExpertSpec::new(RateBackend::Mixture {
                spec: Arc::new(inner),
            })
            .with_name("nested"),
            MixtureExpertSpec::new(RateBackend::Zpaq {
                method: infotheory::api::ZpaqMethodSpec::literal("1"),
            })
            .with_name("zpaq"),
        ],
    )
    .with_alpha(0.03);

    let backend = RateBackend::Mixture {
        spec: Arc::new(outer),
    };
    let rate = try_entropy_rate_backend(data, &backend).expect("rate");
    assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
}

#[test]
fn logistic_mixture_supports_nested_mixture_expert() {
    let data = b"abracadabra abracadabra abracadabra";
    let inner = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![
            MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 }).with_name("ctw"),
            MixtureExpertSpec::new(RateBackend::Match {
                hash_bits: 18,
                min_len: 3,
                max_len: 64,
                base_mix: 0.03,
                confidence_scale: 1.0,
            })
            .with_name("match"),
        ],
    );

    let outer = MixtureSpec::new(
        MixtureKind::Logistic,
        vec![
            MixtureExpertSpec::new(RateBackend::Mixture {
                spec: Arc::new(inner),
            })
            .with_name("nested"),
            MixtureExpertSpec::new(RateBackend::SparseMatch {
                hash_bits: 18,
                min_len: 3,
                max_len: 64,
                gap_min: 1,
                gap_max: 3,
                base_mix: 0.04,
                confidence_scale: 1.0,
            })
            .with_name("sparse"),
        ],
    )
    .with_alpha(0.02);

    let backend = RateBackend::Mixture {
        spec: Arc::new(outer),
    };
    let rate = try_entropy_rate_backend(data, &backend).expect("rate");
    assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
}

#[test]
fn convex_mixture_supports_nested_mixture_expert() {
    let data = b"abracadabra abracadabra abracadabra";
    let inner = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![
            MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 }).with_name("ctw"),
            MixtureExpertSpec::new(RateBackend::FacCtw {
                base_depth: 8,
                num_percept_bits: 8,
                encoding_bits: 8,
                msb_first: None,
            })
            .with_name("fac"),
        ],
    );

    let outer = MixtureSpec::new(
        MixtureKind::Convex,
        vec![
            MixtureExpertSpec::new(RateBackend::Mixture {
                spec: Arc::new(inner),
            })
            .with_name("nested"),
            MixtureExpertSpec::new(RateBackend::Ppmd {
                order: 6,
                memory_mb: 8,
            })
            .with_name("ppmd"),
        ],
    )
    .with_alpha(1.25);

    let backend = RateBackend::Mixture {
        spec: Arc::new(outer),
    };
    let rate = try_entropy_rate_backend(data, &backend).expect("rate");
    assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
}

#[test]
fn new_backends_have_finite_entropy_rates() {
    let data = b"match match match sparse sparse sparse payload";
    let backends = [
        RateBackend::Match {
            hash_bits: 20,
            min_len: 4,
            max_len: 255,
            base_mix: 0.02,
            confidence_scale: 1.0,
        },
        RateBackend::SparseMatch {
            hash_bits: 19,
            min_len: 3,
            max_len: 64,
            gap_min: 1,
            gap_max: 2,
            base_mix: 0.05,
            confidence_scale: 1.0,
        },
        RateBackend::OrderNGram {
            order: 2,
            hash_bits: 12,
        },
        RateBackend::WordContext { hash_bits: 12 },
        RateBackend::Ppmd {
            order: 8,
            memory_mb: 8,
        },
    ];
    for backend in backends {
        let rate = try_entropy_rate_backend(data, &backend).expect("rate");
        assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
    }
}

#[test]
fn neural_mixture_supports_calibrated_expert() {
    let data = b"calibrated ctw expert payload calibrated ctw expert payload";
    let spec = MixtureSpec::new(
        MixtureKind::Neural,
        vec![
            MixtureExpertSpec::new(RateBackend::Calibrated {
                spec: Arc::new(CalibratedSpec::new(
                    RateBackend::Ctw { depth: 8 },
                    CalibrationContextKind::Text,
                )),
            })
            .with_name("cal"),
            MixtureExpertSpec::new(RateBackend::Match {
                hash_bits: 20,
                min_len: 4,
                max_len: 255,
                base_mix: 0.02,
                confidence_scale: 1.0,
            })
            .with_name("match"),
        ],
    )
    .with_alpha(0.03);
    let backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let rate = try_entropy_rate_backend(data, &backend).expect("rate");
    assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
}
