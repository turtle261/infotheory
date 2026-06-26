#[cfg(any(
    feature = "backend-ctw",
    feature = "backend-mixture",
    feature = "backend-zpaq",
    feature = "backend-calibrated",
    feature = "backend-rwkv",
    feature = "backend-mamba"
))]
use infotheory::api::{
    CompressionBackend, GenerationConfig, InfotheoryCtx, NcdVariant, RateBackend,
    RateBackendSession, try_compress_bytes_backend, try_decompress_bytes_backend,
};

#[cfg(feature = "backend-ctw")]
use infotheory::api::{
    BinaryPrediction, BitOrder, BitStreamSemantics, OnlineBitPredictor, RateBackendBitSession,
};

#[cfg(any(
    feature = "backend-ctw",
    feature = "backend-mixture",
    feature = "backend-zpaq",
    feature = "backend-calibrated",
    feature = "backend-rwkv",
    feature = "backend-mamba"
))]
fn assert_close(label: &str, left: f64, right: f64) {
    let diff = (left - right).abs();
    assert!(
        diff <= 1e-12,
        "{label} mismatch: left={left}, right={right}, diff={diff}"
    );
}

#[cfg(feature = "backend-ctw")]
fn assert_bit_prediction_close(label: &str, left: BinaryPrediction, right: BinaryPrediction) {
    assert_close(&format!("{label} p0"), left.p0, right.p0);
    assert_close(&format!("{label} p1"), left.p1, right.p1);
}

#[cfg(feature = "backend-ctw")]
fn assert_bit_sessions_predict_same(
    label: &str,
    compat_session: &mut RateBackendBitSession,
    compiled_session: &mut RateBackendBitSession,
) {
    let compat_prediction = compat_session.predict_bit();
    let compiled_prediction = compiled_session.predict_bit();
    assert_bit_prediction_close(label, compat_prediction, compiled_prediction);
}

#[cfg(feature = "backend-ctw")]
#[derive(Clone, Copy)]
enum BitSessionOp {
    Observe(bool),
    Condition(bool),
    Step(bool),
}

#[cfg(feature = "backend-ctw")]
fn apply_bit_session_op(
    label: &str,
    op: BitSessionOp,
    compat_session: &mut RateBackendBitSession,
    compiled_session: &mut RateBackendBitSession,
) {
    assert_bit_sessions_predict_same(
        &format!("{label} pre-update"),
        compat_session,
        compiled_session,
    );
    match op {
        BitSessionOp::Observe(bit) => {
            compat_session
                .try_observe_bit(bit)
                .unwrap_or_else(|err| panic!("{label} compat observe failed: {err}"));
            compiled_session
                .try_observe_bit(bit)
                .unwrap_or_else(|err| panic!("{label} compiled observe failed: {err}"));
        }
        BitSessionOp::Condition(bit) => {
            compat_session
                .try_condition_bit(bit)
                .unwrap_or_else(|err| panic!("{label} compat condition failed: {err}"));
            compiled_session
                .try_condition_bit(bit)
                .unwrap_or_else(|err| panic!("{label} compiled condition failed: {err}"));
        }
        BitSessionOp::Step(bit) => {
            let compat_prediction = compat_session
                .try_step_bit(bit)
                .unwrap_or_else(|err| panic!("{label} compat step failed: {err}"));
            let compiled_prediction = compiled_session
                .try_step_bit(bit)
                .unwrap_or_else(|err| panic!("{label} compiled step failed: {err}"));
            assert_bit_prediction_close(label, compat_prediction, compiled_prediction);
        }
    }
    assert_bit_sessions_predict_same(
        &format!("{label} post-update"),
        compat_session,
        compiled_session,
    );
}

#[cfg(any(
    feature = "backend-ctw",
    feature = "backend-mixture",
    feature = "backend-zpaq",
    feature = "backend-calibrated",
    feature = "backend-rwkv",
    feature = "backend-mamba"
))]
fn assert_ctx_parity(
    rate_backend: RateBackend,
    compression_backend: CompressionBackend,
    check_generation: bool,
) {
    let compiled_rate = rate_backend
        .compile()
        .unwrap_or_else(|err| panic!("compile rate backend: {err}"));
    let compiled_compression = compression_backend
        .compile()
        .unwrap_or_else(|err| panic!("compile compression backend: {err}"));

    let compiled_ctx = InfotheoryCtx::new(compiled_rate.clone(), compiled_compression.clone());
    let compat_ctx = InfotheoryCtx::from_specs(rate_backend.clone(), compression_backend.clone())
        .unwrap_or_else(|err| panic!("compat ctx: {err}"));

    let prompt = b"compiled plan parity prompt bytes";
    let train = b"compiled plan parity training data";
    let x = b"abcabc123compiled";
    let y = b"xyzxyz456compiled";

    assert_close(
        "entropy",
        compat_ctx.try_entropy_rate_bytes(prompt).unwrap(),
        compiled_ctx.try_entropy_rate_bytes(prompt).unwrap(),
    );
    assert_close(
        "cross-entropy",
        compat_ctx
            .try_cross_entropy_rate_bytes(prompt, train)
            .unwrap(),
        compiled_ctx
            .try_cross_entropy_rate_bytes(prompt, train)
            .unwrap(),
    );
    assert_close(
        "joint-entropy",
        compat_ctx.try_joint_entropy_rate_bytes(x, y).unwrap(),
        compiled_ctx.try_joint_entropy_rate_bytes(x, y).unwrap(),
    );
    assert_close(
        "ncd",
        compat_ctx.try_ncd_bytes(x, y, NcdVariant::Vitanyi).unwrap(),
        compiled_ctx
            .try_ncd_bytes(x, y, NcdVariant::Vitanyi)
            .unwrap(),
    );

    let enc_compat = try_compress_bytes_backend(prompt, &compat_ctx.compression_backend).unwrap();
    let enc_compiled =
        try_compress_bytes_backend(prompt, &compiled_ctx.compression_backend).unwrap();
    assert_eq!(enc_compat, enc_compiled, "compressed payload drift");
    let dec =
        try_decompress_bytes_backend(&enc_compiled, &compiled_ctx.compression_backend).unwrap();
    assert_eq!(dec, prompt, "decompressed payload mismatch");

    let mut compat_session =
        RateBackendSession::from_spec(rate_backend, Some(train.len() as u64)).unwrap();
    let mut compiled_session =
        RateBackendSession::from_backend(compiled_rate, Some(train.len() as u64)).unwrap();
    compat_session.observe(train);
    compiled_session.observe(train);
    let mut compat_logps = [0.0; 256];
    let mut compiled_logps = [0.0; 256];
    compat_session.fill_log_probs(&mut compat_logps);
    compiled_session.fill_log_probs(&mut compiled_logps);
    for (idx, (&left, &right)) in compat_logps.iter().zip(compiled_logps.iter()).enumerate() {
        let diff = (left - right).abs();
        assert!(
            diff <= 1e-12,
            "session log-prob drift at {idx}: left={left}, right={right}, diff={diff}"
        );
    }

    if check_generation {
        let mut cfg = GenerationConfig::default();
        cfg.seed = 7;
        let compat = compat_ctx
            .try_generate_bytes_with_config(prompt, 16, cfg)
            .unwrap();
        let compiled = compiled_ctx
            .try_generate_bytes_with_config(prompt, 16, cfg)
            .unwrap();
        assert_eq!(compat, compiled, "generation drift");
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn compiled_bit_session_matches_wrapper_for_binary_tokens_and_checkpoints() {
    let rate = RateBackend::Ctw { depth: 6 };
    let compiled_rate = rate
        .clone()
        .compile()
        .unwrap_or_else(|err| panic!("compile ctw rate backend: {err}"));
    let mut compat_session =
        RateBackendBitSession::from_spec(rate, Some(13), BitStreamSemantics::BinaryTokens)
            .expect("compat binary-token bit session");
    let mut compiled_session = RateBackendBitSession::from_backend(
        compiled_rate,
        Some(13),
        BitStreamSemantics::BinaryTokens,
    )
    .expect("compiled binary-token bit session");

    for (idx, op) in [
        BitSessionOp::Observe(true),
        BitSessionOp::Condition(false),
        BitSessionOp::Step(true),
        BitSessionOp::Observe(true),
        BitSessionOp::Condition(true),
        BitSessionOp::Step(false),
    ]
    .into_iter()
    .enumerate()
    {
        apply_bit_session_op(
            &format!("binary-token prefix op {idx}"),
            op,
            &mut compat_session,
            &mut compiled_session,
        );
    }

    let compat_checkpoint = compat_session.checkpoint();
    let compiled_checkpoint = compiled_session.checkpoint();

    for (idx, op) in [
        BitSessionOp::Step(true),
        BitSessionOp::Observe(false),
        BitSessionOp::Condition(false),
    ]
    .into_iter()
    .enumerate()
    {
        apply_bit_session_op(
            &format!("binary-token divergent op {idx}"),
            op,
            &mut compat_session,
            &mut compiled_session,
        );
    }

    compat_session
        .restore_checkpoint(&compat_checkpoint)
        .expect("restore compat binary-token checkpoint");
    compiled_session
        .restore_checkpoint(&compiled_checkpoint)
        .expect("restore compiled binary-token checkpoint");
    assert_bit_sessions_predict_same(
        "binary-token restored checkpoint",
        &mut compat_session,
        &mut compiled_session,
    );
    compat_session.clear_checkpoints_if_supported();
    compiled_session.clear_checkpoints_if_supported();

    OnlineBitPredictor::begin_bit_stream(
        &mut compat_session,
        Some(5),
        BitStreamSemantics::BinaryTokens,
    )
    .expect("compat binary-token stream restart");
    OnlineBitPredictor::begin_bit_stream(
        &mut compiled_session,
        Some(5),
        BitStreamSemantics::BinaryTokens,
    )
    .expect("compiled binary-token stream restart");

    for (idx, bit) in [true, false, true, true, false].into_iter().enumerate() {
        apply_bit_session_op(
            &format!("binary-token restarted op {idx}"),
            BitSessionOp::Step(bit),
            &mut compat_session,
            &mut compiled_session,
        );
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn compiled_bit_session_matches_wrapper_for_byte_packed_checkpoint_restore() {
    let rate = RateBackend::Ctw { depth: 6 };
    let compiled_rate = rate
        .clone()
        .compile()
        .unwrap_or_else(|err| panic!("compile ctw rate backend: {err}"));
    let semantics = BitStreamSemantics::BytePacked {
        order: BitOrder::MsbFirst,
    };
    let mut compat_session =
        RateBackendBitSession::from_spec(rate, Some(24), semantics).expect("compat bit session");
    let mut compiled_session =
        RateBackendBitSession::from_backend(compiled_rate, Some(24), semantics)
            .expect("compiled bit session");

    for (idx, bit) in [true, false, true, false, true, false, false, true]
        .into_iter()
        .enumerate()
    {
        apply_bit_session_op(
            &format!("byte-packed first byte bit {idx}"),
            BitSessionOp::Observe(bit),
            &mut compat_session,
            &mut compiled_session,
        );
    }

    for (idx, bit) in [false, true, true].into_iter().enumerate() {
        apply_bit_session_op(
            &format!("byte-packed checkpoint prefix bit {idx}"),
            BitSessionOp::Observe(bit),
            &mut compat_session,
            &mut compiled_session,
        );
    }

    let compat_checkpoint = compat_session.checkpoint();
    let compiled_checkpoint = compiled_session.checkpoint();

    for (idx, bit) in [true, true, false, false, false].into_iter().enumerate() {
        apply_bit_session_op(
            &format!("byte-packed divergent suffix bit {idx}"),
            BitSessionOp::Observe(bit),
            &mut compat_session,
            &mut compiled_session,
        );
    }

    compat_session
        .restore_checkpoint(&compat_checkpoint)
        .expect("restore compat byte-packed checkpoint");
    compiled_session
        .restore_checkpoint(&compiled_checkpoint)
        .expect("restore compiled byte-packed checkpoint");
    assert_bit_sessions_predict_same(
        "byte-packed restored mid-byte checkpoint",
        &mut compat_session,
        &mut compiled_session,
    );

    for (idx, bit) in [false, false, true, true, true].into_iter().enumerate() {
        apply_bit_session_op(
            &format!("byte-packed restored suffix bit {idx}"),
            BitSessionOp::Observe(bit),
            &mut compat_session,
            &mut compiled_session,
        );
    }

    for (idx, bit) in [true, true, false, false, true, false, true, false]
        .into_iter()
        .enumerate()
    {
        apply_bit_session_op(
            &format!("byte-packed frozen byte bit {idx}"),
            BitSessionOp::Condition(bit),
            &mut compat_session,
            &mut compiled_session,
        );
    }

    compat_session.finish().expect("compat byte-packed finish");
    compiled_session
        .finish()
        .expect("compiled byte-packed finish");
}

#[cfg(feature = "backend-ctw")]
#[test]
fn compiled_ctx_matches_wrapper_ctx_for_ctw_rate_and_rate_ac() {
    let rate = RateBackend::Ctw { depth: 8 };
    let compression = CompressionBackend::Rate {
        rate_backend: rate.clone(),
        coder: infotheory::coders::CoderType::AC,
        framing: infotheory::compression::FramingMode::Framed,
    };
    assert_ctx_parity(rate, compression, true);
}

#[cfg(all(
    feature = "backend-mixture",
    feature = "backend-ctw",
    feature = "backend-match"
))]
#[test]
fn compiled_ctx_matches_wrapper_ctx_for_mixture_backend() {
    let rate = RateBackend::Mixture {
        spec: std::sync::Arc::new(infotheory::api::MixtureSpec::new(
            infotheory::api::MixtureKind::Bayes,
            vec![
                {
                    let mut expert =
                        infotheory::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 });
                    expert.name = Some("ctw".to_string());
                    expert
                },
                {
                    let mut expert = infotheory::api::MixtureExpertSpec::new(RateBackend::Match {
                        hash_bits: 18,
                        min_len: 4,
                        max_len: 64,
                        base_mix: 0.02,
                        confidence_scale: 1.0,
                    });
                    expert.name = Some("match".to_string());
                    expert.log_prior = -0.2;
                    expert
                },
            ],
        )),
    };
    let compression = CompressionBackend::Rate {
        rate_backend: rate.clone(),
        coder: infotheory::coders::CoderType::RANS,
        framing: infotheory::compression::FramingMode::Framed,
    };
    assert_ctx_parity(rate, compression, false);
}

#[cfg(all(feature = "backend-calibrated", feature = "backend-ctw"))]
#[test]
fn compiled_ctx_matches_wrapper_ctx_for_calibrated_backend() {
    let rate = RateBackend::Calibrated {
        spec: std::sync::Arc::new(infotheory::api::CalibratedSpec::new(
            RateBackend::Ctw { depth: 8 },
            infotheory::api::CalibrationContextKind::Text,
        )),
    };
    let compression = CompressionBackend::Rate {
        rate_backend: rate.clone(),
        coder: infotheory::coders::CoderType::AC,
        framing: infotheory::compression::FramingMode::Framed,
    };
    assert_ctx_parity(rate, compression, false);
}

#[cfg(feature = "backend-zpaq")]
#[test]
fn compiled_ctx_matches_wrapper_ctx_for_zpaq_rate_and_compression() {
    let rate = RateBackend::Zpaq {
        method: infotheory::api::ZpaqMethodSpec::literal("1"),
    };
    let compression = CompressionBackend::zpaq("1");
    assert_ctx_parity(rate, compression, false);
}

#[cfg(feature = "backend-rwkv")]
#[test]
fn compiled_ctx_matches_wrapper_ctx_for_rwkv_backends() {
    let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
    let rate = RateBackend::Rwkv7Method {
        method: infotheory::rwkvzip::parse_method_spec(method).expect("rwkv method spec"),
    };
    let compression = CompressionBackend::Rwkv7 {
        method: infotheory::rwkvzip::parse_method_spec(method).expect("rwkv method spec"),
        coder: infotheory::coders::CoderType::AC,
    };
    assert_ctx_parity(rate, compression, false);
}

#[cfg(feature = "backend-mamba")]
#[test]
fn compiled_ctx_matches_wrapper_ctx_for_mamba_rate_backend() {
    let method = "cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=26,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
    let rate = RateBackend::MambaMethod {
        method: infotheory::mambazip::parse_method_spec(method).expect("mamba method spec"),
    };
    let compression = CompressionBackend::Rate {
        rate_backend: rate.clone(),
        coder: infotheory::coders::CoderType::AC,
        framing: infotheory::compression::FramingMode::Framed,
    };
    assert_ctx_parity(rate, compression, false);
}
