use super::plan_macros::expect_plan_ref;
use super::*;

#[cfg(feature = "backend-rosa")]
pub(super) fn build_predictor_rosa(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::RosaPlus { max_order },
        "rosa kernel used with non-rosa plan"
    );
    let mut model = RosaPlus::new(*max_order, false, 0, 42);
    model.build_lm_full_bytes_no_finalize_endpos();
    Ok(crate::mixture::RateBackendPredictor::Rosa {
        model,
        min_prob,
        checkpoint_journal: Vec::new(),
        checkpoint_depth: 0,
    })
}

#[cfg(not(feature = "backend-rosa"))]
pub(super) fn build_predictor_rosa(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(
        RateBackendKind::RosaPlus,
    ))
}

#[cfg(feature = "backend-match")]
pub(super) fn build_predictor_match(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Match {
            hash_bits,
            min_len,
            max_len,
            base_mix,
            confidence_scale,
        },
        "match kernel used with non-match plan"
    );
    Ok(crate::mixture::RateBackendPredictor::Match {
        model: MatchModel::new_contiguous(
            *hash_bits,
            *min_len,
            *max_len,
            *base_mix,
            *confidence_scale,
        ),
        min_prob,
    })
}

#[cfg(not(feature = "backend-match"))]
pub(super) fn build_predictor_match(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(RateBackendKind::Match))
}

#[cfg(feature = "backend-match")]
pub(super) fn build_predictor_sparse_match(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::SparseMatch {
            hash_bits,
            min_len,
            max_len,
            gap_min,
            gap_max,
            base_mix,
            confidence_scale,
        },
        "sparse-match kernel used with non-sparse-match plan"
    );
    Ok(crate::mixture::RateBackendPredictor::SparseMatch {
        model: SparseMatchModel::new(
            *hash_bits,
            *min_len,
            *max_len,
            *gap_min,
            *gap_max,
            *base_mix,
            *confidence_scale,
        ),
        min_prob,
    })
}

#[cfg(not(feature = "backend-match"))]
pub(super) fn build_predictor_sparse_match(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(
        RateBackendKind::SparseMatch,
    ))
}

#[cfg(feature = "backend-ppmd")]
pub(super) fn build_predictor_ppmd(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Ppmd { order, memory_mb },
        "ppmd kernel used with non-ppmd plan"
    );
    Ok(crate::mixture::RateBackendPredictor::Ppmd {
        model: PpmdModel::new(*order, *memory_mb),
        min_prob,
    })
}

#[cfg(not(feature = "backend-ppmd"))]
pub(super) fn build_predictor_ppmd(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(RateBackendKind::Ppmd))
}

#[cfg(feature = "backend-sequitur")]
pub(super) fn build_predictor_sequitur(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Sequitur { context_bytes },
        "sequitur kernel used with non-sequitur plan"
    );
    Ok(crate::mixture::RateBackendPredictor::Sequitur {
        model: SequiturModel::new(*context_bytes),
        min_prob,
    })
}

#[cfg(not(feature = "backend-sequitur"))]
pub(super) fn build_predictor_sequitur(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(
        RateBackendKind::Sequitur,
    ))
}

#[cfg(feature = "backend-ctw")]
pub(super) fn build_predictor_ctw(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Ctw { depth },
        "ctw kernel used with non-ctw plan"
    );
    Ok(crate::mixture::RateBackendPredictor::Ctw {
        tree: FacContextTree::new(*depth, 8),
        min_prob,
        checkpoint_journal: Vec::new(),
        checkpoint_depth: 0,
    })
}

#[cfg(not(feature = "backend-ctw"))]
pub(super) fn build_predictor_ctw(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(RateBackendKind::Ctw))
}

#[cfg(feature = "backend-ctw")]
pub(super) fn build_predictor_fac_ctw(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        },
        "fac-ctw kernel used with non-fac-ctw plan"
    );
    let bits_per_symbol = (*encoding_bits).clamp(1, 8);
    Ok(crate::mixture::RateBackendPredictor::FacCtw {
        tree: FacContextTree::new(*base_depth, bits_per_symbol),
        bits_per_symbol,
        min_prob,
        checkpoint_journal: Vec::new(),
        checkpoint_depth: 0,
    })
}

#[cfg(not(feature = "backend-ctw"))]
pub(super) fn build_predictor_fac_ctw(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(
        RateBackendKind::FacCtw,
    ))
}

#[cfg(feature = "backend-rwkv")]
pub(super) fn build_predictor_rwkv(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Rwkv7 { parsed_method, .. },
        "rwkv kernel used with non-rwkv plan"
    );
    let mut compressor = rwkvzip::Compressor::new_from_method_spec(parsed_method)
        .map_err(|e| format!("invalid rwkv method: {e}"))?;
    compressor.reset_and_prime();
    Ok(crate::mixture::RateBackendPredictor::Rwkv7 {
        pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
        compressor,
        primed: true,
        min_prob,
    })
}

#[cfg(not(feature = "backend-rwkv"))]
pub(super) fn build_predictor_rwkv(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(RateBackendKind::Rwkv7))
}

#[cfg(feature = "backend-mamba")]
pub(super) fn build_predictor_mamba(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Mamba { parsed_method, .. },
        "mamba kernel used with non-mamba plan"
    );
    let mut compressor = mambazip::Compressor::new_from_method_spec(parsed_method)
        .map_err(|e| format!("invalid mamba method: {e}"))?;
    let bias = compressor.online_bias_snapshot();
    let logits = compressor
        .model
        .forward(&mut compressor.scratch, 0, &mut compressor.state);
    mambazip::Compressor::logits_to_pdf(logits, bias.as_deref(), &mut compressor.pdf_buffer);
    Ok(crate::mixture::RateBackendPredictor::Mamba {
        pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
        compressor,
        primed: true,
        min_prob,
    })
}

#[cfg(not(feature = "backend-mamba"))]
pub(super) fn build_predictor_mamba(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(RateBackendKind::Mamba))
}

#[cfg(feature = "backend-zpaq")]
pub(super) fn build_predictor_zpaq(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Zpaq { method },
        "zpaq kernel used with non-zpaq plan"
    );
    Ok(crate::mixture::RateBackendPredictor::Zpaq {
        model: ZpaqRateModel::new(method.clone(), min_prob),
    })
}

#[cfg(not(feature = "backend-zpaq"))]
pub(super) fn build_predictor_zpaq(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(RateBackendKind::Zpaq))
}

#[cfg(feature = "backend-mixture")]
pub(super) fn build_predictor_mixture(
    backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let experts = crate::mixture::expert_configs_from_compiled_mixture(backend)?;
    let runtime = crate::mixture::build_mixture_runtime_from_compiled(backend, &experts)
        .map_err(|e| format!("MixtureSpec invalid: {e}"))?;
    Ok(crate::mixture::RateBackendPredictor::Mixture { runtime })
}

#[cfg(not(feature = "backend-mixture"))]
pub(super) fn build_predictor_mixture(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(
        RateBackendKind::Mixture,
    ))
}

#[cfg(feature = "backend-particle")]
pub(super) fn build_predictor_particle(
    backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Particle { spec },
        "particle kernel used with non-particle plan"
    );
    Ok(crate::mixture::RateBackendPredictor::Particle {
        runtime: ParticleRuntime::new(spec),
    })
}

#[cfg(not(feature = "backend-particle"))]
pub(super) fn build_predictor_particle(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(
        RateBackendKind::Particle,
    ))
}

#[cfg(feature = "backend-calibrated")]
pub(super) fn build_predictor_calibrated(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Calibrated {
            context,
            bins,
            learning_rate,
            bias_clip,
            base,
        },
        "calibrated kernel used with non-calibrated plan"
    );
    let base_backend = compile_calibrated_base_backend(base)?;
    Ok(crate::mixture::RateBackendPredictor::Calibrated {
        base: Box::new(build_rate_backend_predictor_via_kernel(
            &base_backend,
            min_prob,
        )?),
        core: CalibratorCore::new(*context, *bins, *learning_rate, *bias_clip),
        pdf: [1.0 / 256.0; 256],
        valid: false,
        min_prob,
    })
}

#[cfg(not(feature = "backend-calibrated"))]
pub(super) fn build_predictor_calibrated(
    _backend: &CompiledRateBackend,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(registry::rate_backend_feature_error(
        RateBackendKind::Calibrated,
    ))
}
