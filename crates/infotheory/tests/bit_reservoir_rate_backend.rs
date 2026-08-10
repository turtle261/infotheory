#![cfg(feature = "backend-bit-reservoir")]

use infotheory::api::{
    BitOrder, BitReservoirConfig, BitStreamSemantics, CompressionBackend, GenerationConfig,
    RateBackend, RateBackendBitSession, RateBackendSession, try_compress_bytes_backend,
    try_cross_entropy_rate_backend, try_decompress_bytes_backend, try_entropy_rate_backend,
};
use infotheory::coders::CoderType;
use infotheory::compression::FramingMode;

// Integration tests are outside the defining crate, so `BitReservoirConfig`'s
// `#[non_exhaustive]` contract intentionally rules out struct update syntax.
#[allow(clippy::field_reassign_with_default)]
fn small_config() -> BitReservoirConfig {
    let mut config = BitReservoirConfig::default();
    config.hidden = 16;
    config.delay_bits = 24;
    config.embedding_bits = 10;
    config.learning_rate = 0.02;
    config.learning_rate_decay = 0.0;
    config.weight_decay = 0.0;
    config.state_decay = 0.75;
    config.recurrent_scale = 0.35;
    config.input_scale = 0.8;
    config.phase_scale = 0.2;
    config.grad_clip = 1.0;
    config.seed = 7;
    config
}

fn small_backend() -> RateBackend {
    RateBackend::BitReservoir {
        config: small_config(),
    }
}

#[test]
fn bit_reservoir_json_and_shorthand_compile() {
    let json = serde_json::json!({
        "kind": "bit-reservoir",
        "hidden": 16,
        "delay_bits": 24,
        "embedding_bits": 10,
        "learning_rate": 0.02,
        "seed": 7,
    });
    let parsed = infotheory::spec::parse_rate_backend_json(
        &json,
        std::path::Path::new("."),
        infotheory::api::MAX_MIXTURE_NESTING,
    )
    .expect("parse bit-reservoir json");
    let compiled = parsed.compile().expect("compile parsed bit-reservoir");
    assert_eq!(compiled.canonical_name(), "bit-reservoir");
    assert!(compiled.supports_native_bit_prediction());
    assert!(compiled.supports_rate_coded_compression());
    assert!(!compiled.supports_reversible_bit_updates());

    let shorthand = infotheory::spec::compile_rate_backend_name_method(
        "bitreservoir",
        Some("h=16,d=24,eb=10,lr=0.02,seed=7"),
        &infotheory::spec::RateBackendShorthandOptions::default(),
    )
    .expect("compile shorthand bit-reservoir alias");
    assert_eq!(shorthand.canonical_name(), "bit-reservoir");
}

#[test]
fn bit_reservoir_entropy_and_byte_packed_bit_session_are_finite() {
    let backend = small_backend();
    let compiled = backend.compile().expect("compile bit-reservoir");
    let data = b"bitwise neural predictor smoke";
    let rate = try_entropy_rate_backend(data, &compiled).expect("bit-reservoir entropy");
    assert!(rate.is_finite() && rate >= 0.0);

    let semantics = BitStreamSemantics::BytePacked {
        order: BitOrder::MsbFirst,
    };
    let mut session =
        RateBackendBitSession::from_spec(backend, Some(16), semantics).expect("bit session");
    for &bit in &[true, false, true, false, false, true, true, false] {
        let prediction = session.step_bit(bit);
        assert!(prediction.p1.is_finite());
    }
    session.finish().expect("finish bit session");
}

#[test]
fn bit_reservoir_binary_tokens_match_one_bit_model_steps() {
    let config = small_config();
    let backend = RateBackend::BitReservoir {
        config: config.clone(),
    };
    let semantics = BitStreamSemantics::BinaryTokens;
    let bits = [true, false, true, true, false, false, true, false, true];
    let mut session = RateBackendBitSession::from_spec(backend, Some(bits.len() as u64), semantics)
        .expect("binary-token bit session");
    let mut direct = infotheory::backends::bit_reservoir::BitReservoirModel::new(config)
        .expect("direct bit-reservoir model");

    for &bit in &bits {
        let expected = direct.predict_prob_one();
        let prediction = session.step_bit(bit);
        assert!(
            (prediction.p1 - expected).abs() <= 1.0e-12,
            "binary-token prediction should consume one bit, got {} expected {}",
            prediction.p1,
            expected
        );
        direct.observe_bit(bit, true);
    }

    assert!(
        (session.predict_one() - direct.predict_prob_one()).abs() <= 1.0e-12,
        "binary-token session should stay aligned with direct one-bit updates"
    );
}

#[test]
fn bit_reservoir_byte_packed_frozen_prefix_predictions_match_no_learn_bits() {
    let mut config = small_config();
    config.embedding_bits = 8;
    config.learning_rate = 0.05;
    config.learning_rate_decay = 0.0;
    config.weight_decay = 0.0;

    let backend = RateBackend::BitReservoir {
        config: config.clone(),
    };
    let mut session = RateBackendBitSession::from_spec(
        backend,
        None,
        BitStreamSemantics::BytePacked {
            order: BitOrder::MsbFirst,
        },
    )
    .expect("byte-packed bit-reservoir session");
    let mut direct = infotheory::backends::bit_reservoir::BitReservoirModel::new(config)
        .expect("direct bit-reservoir model");

    let fit = b"trained sparse bit-reservoir state for frozen partial byte conditioning";
    for &byte in fit {
        direct.log_prob_update_byte(byte, true);
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            session.step_bit(bit);
        }
    }

    session.reset_frozen(None).expect("reset session frozen");
    direct.reset_state_only();

    let prefix_byte = 0b1011_0010u8;
    for bit_idx in 0..7usize {
        let actual_before = session.predict_one();
        let expected_before = direct.predict_prob_one();
        assert!(
            (actual_before - expected_before).abs() <= 1.0e-12,
            "frozen prefix prediction before bit {bit_idx} should match no-learn direct model: got {actual_before}, expected {expected_before}"
        );

        let bit = ((prefix_byte >> (7 - bit_idx)) & 1) == 1;
        session.condition_bit(bit);
        direct.observe_bit(bit, false);

        let actual_after = session.predict_one();
        let expected_after = direct.predict_prob_one();
        assert!(
            (actual_after - expected_after).abs() <= 1.0e-12,
            "frozen prefix prediction after bit {bit_idx} should not learn from conditioned bit: got {actual_after}, expected {expected_after}"
        );
    }
}

#[test]
fn bit_reservoir_entropy_matches_byte_packed_native_bit_session_log_loss() {
    let backend = small_backend();
    let data = b"bit-reservoir native bitwise mdl equivalence";
    let compiled = backend.clone().compile().expect("compile bit-reservoir");
    let entropy_bits = try_entropy_rate_backend(data, &compiled).expect("bit-reservoir entropy")
        * data.len() as f64;
    let mut session = RateBackendBitSession::from_spec(
        backend,
        Some((data.len() * 8) as u64),
        BitStreamSemantics::BytePacked {
            order: BitOrder::MsbFirst,
        },
    )
    .expect("byte-packed bit session");

    let mut bit_session_bits = 0.0f64;
    for &byte in data {
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            let prediction = session.step_bit(bit);
            let p = if bit { prediction.p1 } else { prediction.p0 };
            bit_session_bits -= p.ln() / std::f64::consts::LN_2;
        }
    }
    session.finish().expect("finish bit session");

    assert!(
        (entropy_bits - bit_session_bits).abs() <= 1.0e-10,
        "entropy bits {entropy_bits} must match native bit-session log loss {bit_session_bits}"
    );
}

#[test]
fn bit_reservoir_cross_entropy_uses_frozen_within_byte_scoring() {
    let mut config = small_config();
    config.embedding_bits = 8;
    config.learning_rate = 0.05;
    config.learning_rate_decay = 0.0;
    config.weight_decay = 0.0;
    let backend = RateBackend::BitReservoir {
        config: config.clone(),
    };
    let compiled = backend.compile().expect("compile bit-reservoir");
    let train = b"frozen plugin fit pass with sparse collisions";
    let test = b"score bytes";

    let cross_entropy_bits = try_cross_entropy_rate_backend(test, train, &compiled)
        .expect("cross entropy")
        * test.len() as f64;

    let mut direct = infotheory::backends::bit_reservoir::BitReservoirModel::new(config)
        .expect("direct bit-reservoir model");
    for &byte in train {
        direct.log_prob_update_byte(byte, true);
    }
    direct.reset_state_only();

    let mut direct_bits = 0.0f64;
    for &byte in test {
        direct_bits -= manual_frozen_byte_log_prob(&mut direct, byte) / std::f64::consts::LN_2;
    }

    assert!(
        (cross_entropy_bits - direct_bits).abs() <= 1.0e-10,
        "cross-entropy bits {cross_entropy_bits} must match frozen no-learn bit stepping {direct_bits}"
    );
}

#[test]
fn bit_reservoir_frozen_generation_uses_frozen_byte_row() {
    let mut config = small_config();
    config.embedding_bits = 8;
    config.learning_rate = 0.05;
    config.learning_rate_decay = 0.0;
    config.weight_decay = 0.0;

    let corpus = b"frozen generation should not score hypothetical within-byte learning";
    let backend = RateBackend::BitReservoir {
        config: config.clone(),
    };
    let compiled = backend.compile().expect("compile bit-reservoir");
    let mut session = RateBackendSession::from_backend(compiled, None).expect("session");
    session.observe(corpus);
    session.reset_frozen(None).expect("reset frozen");

    let mut direct = infotheory::backends::bit_reservoir::BitReservoirModel::new(config)
        .expect("direct bit-reservoir model");
    for &byte in corpus {
        direct.log_prob_update_byte(byte, true);
    }
    direct.reset_state_only();

    let mut expected = Vec::with_capacity(16);
    for _ in 0..16 {
        let mut row = [0.0f64; 256];
        direct.fill_byte_log_probs_frozen(&mut row, infotheory::mixture::DEFAULT_MIN_PROB);
        let byte = argmax_log_prob(&row);
        expected.push(byte);
        direct.update_byte(byte, false);
    }

    let generated = session.generate_bytes(16, GenerationConfig::greedy_frozen());
    assert_eq!(generated, expected);
}

#[test]
fn bit_reservoir_rate_ac_roundtrips() {
    let backend = CompressionBackend::Rate {
        rate_backend: small_backend(),
        coder: CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .expect("compile compression backend");
    let data = b"bit-reservoir framed arithmetic coding roundtrip";
    let encoded = try_compress_bytes_backend(data, &backend).expect("compress bit-reservoir");
    let decoded =
        try_decompress_bytes_backend(&encoded, &backend).expect("decompress bit-reservoir");
    assert_eq!(decoded, data);
}

fn manual_frozen_byte_log_prob(
    model: &mut infotheory::backends::bit_reservoir::BitReservoirModel,
    byte: u8,
) -> f64 {
    let mut logp = 0.0f64;
    for bit_idx in 0..8usize {
        let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
        let p1 = model.predict_prob_one();
        let p = if bit { p1 } else { 1.0 - p1 };
        logp += p.max(infotheory::mixture::DEFAULT_MIN_PROB).ln();
        model.observe_bit(bit, false);
    }
    logp
}

fn argmax_log_prob(row: &[f64; 256]) -> u8 {
    let mut best = 0usize;
    let mut best_logp = f64::NEG_INFINITY;
    for (idx, &logp) in row.iter().enumerate() {
        if logp > best_logp {
            best = idx;
            best_logp = logp;
        }
    }
    best as u8
}
