//! ZPAQ rate-backend integration tests.
//!
//! Full appendix repro for cross-FFI preceding activity requires at least
//! `backend-zpaq,backend-ctw,backend-ppmd,backend-match` (or `all-backends`).
//! Narrow `backend-zpaq` slices still run ZPAQ settlement and first-symbol parity;
//! non-ZPAQ preceding coverage accumulates with enabled backend features.

#![cfg(feature = "backend-zpaq")]

use infotheory::api::{OnlineBytePredictor, RateBackend};
use infotheory::backends::zpaq_rate::ZpaqRateModel;
use infotheory::mixture::{DEFAULT_MIN_PROB, RateBackendPredictor};

#[test]
#[cfg(feature = "backend-zpaq")]
fn zpaq_rate_backend_compresses_copy_data() {
    use infotheory::api::try_entropy_rate_backend;

    let mut data = Vec::new();
    let pattern = b"copy-like-pattern-";
    for _ in 0..512 {
        data.extend_from_slice(pattern);
    }
    let backend = RateBackend::Zpaq {
        method: infotheory::api::ZpaqMethodSpec::literal("2"),
    };
    let backend = backend.compile().expect("compile zpaq rate backend");
    let rate = try_entropy_rate_backend(&data, &backend).expect("entropy rate");
    assert!(
        rate < 0.5,
        "expected low entropy rate for copy-like data, got {rate:.4}"
    );
}

#[allow(dead_code)]
fn exercise_preceding_backend(label: &str, backend: RateBackend) {
    let mut predictor = RateBackendPredictor::from_backend(backend, DEFAULT_MIN_PROB);
    predictor
        .begin_stream(Some(64))
        .unwrap_or_else(|err| panic!("{label} begin_stream failed: {err}"));
    let mut row = [0.0f64; 256];
    predictor.fill_log_probs(&mut row);
    for &byte in b"mixed preceding backend activity" {
        let _ = predictor.log_prob(byte);
        predictor.update(byte);
    }
    predictor
        .reset_frozen(Some(16))
        .unwrap_or_else(|err| panic!("{label} reset_frozen failed: {err}"));
    for &byte in b"conditioned tail" {
        predictor.update_frozen(byte);
    }
    predictor.fill_log_probs(&mut row);
    assert!(
        row.iter().all(|lp| lp.is_finite()),
        "{label} preceding activity produced non-finite log probabilities"
    );
}

#[cfg(feature = "backend-ctw")]
fn exercise_preceding_ctw() {
    exercise_preceding_backend("ctw", RateBackend::Ctw { depth: 8 });
}

#[cfg(feature = "backend-match")]
fn exercise_preceding_match() {
    exercise_preceding_backend(
        "match",
        RateBackend::Match {
            hash_bits: 12,
            min_len: 3,
            max_len: 16,
            base_mix: 0.02,
            confidence_scale: 1.0,
        },
    );
    exercise_preceding_backend(
        "sparse-match",
        RateBackend::SparseMatch {
            hash_bits: 12,
            min_len: 3,
            max_len: 16,
            gap_min: 0,
            gap_max: 2,
            base_mix: 0.02,
            confidence_scale: 1.0,
        },
    );
}

#[cfg(feature = "backend-ppmd")]
fn exercise_preceding_ppmd() {
    exercise_preceding_backend(
        "ppmd",
        RateBackend::Ppmd {
            order: 6,
            memory_mb: 8,
        },
    );
}

#[cfg(all(
    feature = "backend-mixture",
    feature = "backend-ctw",
    feature = "backend-ppmd"
))]
fn exercise_preceding_mixture() {
    use infotheory::api::{MixtureExpertSpec, MixtureKind, MixtureSpec};
    use std::sync::Arc;

    exercise_preceding_backend(
        "mixture",
        RateBackend::Mixture {
            spec: Arc::new(
                MixtureSpec::new(
                    MixtureKind::Bayes,
                    vec![
                        MixtureExpertSpec::new(RateBackend::Ctw { depth: 6 }).with_name("ctw"),
                        MixtureExpertSpec::new(RateBackend::Ppmd {
                            order: 4,
                            memory_mb: 8,
                        })
                        .with_name("ppmd"),
                    ],
                )
                .with_alpha(0.03),
            ),
        },
    );
}

#[test]
fn zpaq_restart_first_symbol_parity_after_preceding_activity() {
    #[cfg(feature = "backend-ctw")]
    exercise_preceding_ctw();
    #[cfg(feature = "backend-match")]
    exercise_preceding_match();
    #[cfg(feature = "backend-ppmd")]
    exercise_preceding_ppmd();
    #[cfg(all(
        feature = "backend-mixture",
        feature = "backend-ctw",
        feature = "backend-ppmd"
    ))]
    exercise_preceding_mixture();

    // ZPAQ preceding (exercises the settle path; always available under backend-zpaq):
    let _preceding_zpaq = ZpaqRateModel::new("1", 1e-9);

    let mut session = ZpaqRateModel::new("1", 1e-9);
    session.begin_stream();
    let mut warm = [0.0f64; 256];
    session.fill_log_probs(&mut warm);
    for &byte in b"zpaq history before restart" {
        session.update(byte);
    }

    session.begin_stream();
    let mut restarted = [0.0f64; 256];
    session.fill_log_probs(&mut restarted);

    let mut fresh = ZpaqRateModel::new("1", 1e-9);
    let mut expected = [0.0f64; 256];
    fresh.fill_log_probs(&mut expected);

    for (symbol, (&actual, &expected)) in restarted.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-9,
            "first-symbol parity after preceding + restart failed for symbol {symbol}; diff={}",
            actual - expected
        );
    }
}
