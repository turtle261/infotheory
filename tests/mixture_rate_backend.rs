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
