use infotheory::{MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend, entropy_rate_backend};
use std::sync::Arc;

#[test]
fn mixture_single_expert_matches_backend() {
    let data = b"abababababababababababababababab";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = entropy_rate_backend(data, -1, &base);

    let spec = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![MixtureExpertSpec {
            name: None,
            log_prior: 0.0,
            max_order: -1,
            backend: base.clone(),
        }],
    );
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = entropy_rate_backend(data, -1, &mix_backend);

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn mixture_recursive_expert_matches_backend() {
    let data = b"01010101010101010101010101010101";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = entropy_rate_backend(data, -1, &base);

    let inner = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![MixtureExpertSpec {
            name: Some("ctw".to_string()),
            log_prior: 0.0,
            max_order: -1,
            backend: base.clone(),
        }],
    );
    let outer = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![MixtureExpertSpec {
            name: Some("inner".to_string()),
            log_prior: 0.0,
            max_order: -1,
            backend: RateBackend::Mixture {
                spec: Arc::new(inner),
            },
        }],
    );
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(outer),
    };
    let mix_rate = entropy_rate_backend(data, -1, &mix_backend);

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn neural_mixture_single_expert_matches_backend() {
    let data = b"abababababababababababababababab";
    let base = RateBackend::Ctw { depth: 8 };
    let base_rate = entropy_rate_backend(data, -1, &base);

    let spec = MixtureSpec::new(
        MixtureKind::Neural,
        vec![MixtureExpertSpec {
            name: None,
            log_prior: 0.0,
            max_order: -1,
            backend: base.clone(),
        }],
    )
    .with_alpha(0.05);
    let mix_backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let mix_rate = entropy_rate_backend(data, -1, &mix_backend);

    assert!(
        (mix_rate - base_rate).abs() < 1e-6,
        "mix={mix_rate} base={base_rate}"
    );
}

#[test]
fn neural_mixture_supports_nested_mixture_expert() {
    let data = b"abracadabra abracadabra abracadabra";
    let inner = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![
            MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Ctw { depth: 8 },
            },
            MixtureExpertSpec {
                name: Some("fac".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::FacCtw {
                    base_depth: 8,
                    num_percept_bits: 8,
                    encoding_bits: 8,
                },
            },
        ],
    );

    let outer = MixtureSpec::new(
        MixtureKind::Neural,
        vec![
            MixtureExpertSpec {
                name: Some("nested".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Mixture {
                    spec: Arc::new(inner),
                },
            },
            MixtureExpertSpec {
                name: Some("zpaq".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Zpaq {
                    method: "1".to_string(),
                },
            },
        ],
    )
    .with_alpha(0.03);

    let backend = RateBackend::Mixture {
        spec: Arc::new(outer),
    };
    let rate = entropy_rate_backend(data, -1, &backend);
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
        RateBackend::Ppmd {
            order: 8,
            memory_mb: 8,
        },
    ];
    for backend in backends {
        let rate = entropy_rate_backend(data, -1, &backend);
        assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
    }
}

#[test]
fn neural_mixture_supports_calibrated_expert() {
    let data = b"calibrated ctw expert payload calibrated ctw expert payload";
    let spec = MixtureSpec::new(
        MixtureKind::Neural,
        vec![
            MixtureExpertSpec {
                name: Some("cal".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Calibrated {
                    spec: Arc::new(infotheory::CalibratedSpec {
                        base: RateBackend::Ctw { depth: 8 },
                        context: infotheory::CalibrationContextKind::Text,
                        bins: 33,
                        learning_rate: 0.02,
                        bias_clip: 4.0,
                    }),
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
    let backend = RateBackend::Mixture {
        spec: Arc::new(spec),
    };
    let rate = entropy_rate_backend(data, -1, &backend);
    assert!(rate.is_finite() && rate >= 0.0, "rate={rate}");
}
