use super::*;

#[cfg(feature = "backend-rosa")]
pub(super) fn build_pdf_predictor_rosa(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::RosaPlus { max_order } = backend.plan() else {
        unreachable!("rosa kernel used with non-rosa plan")
    };
    Ok(crate::compression::RatePdfPredictor::Rosa(
        crate::compression::RosaPredictor::new(*max_order),
    ))
}

#[cfg(not(feature = "backend-rosa"))]
pub(super) fn build_pdf_predictor_rosa(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::RosaPlus))
}

#[cfg(feature = "backend-match")]
pub(super) fn build_pdf_predictor_match(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = backend.plan()
    else {
        unreachable!("match kernel used with non-match plan")
    };
    Ok(crate::compression::RatePdfPredictor::Match {
        model: MatchModel::new_contiguous(
            *hash_bits,
            *min_len,
            *max_len,
            *base_mix,
            *confidence_scale,
        ),
    })
}

#[cfg(not(feature = "backend-match"))]
pub(super) fn build_pdf_predictor_match(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Match))
}

#[cfg(feature = "backend-match")]
pub(super) fn build_pdf_predictor_sparse_match(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = backend.plan()
    else {
        unreachable!("sparse-match kernel used with non-sparse-match plan")
    };
    Ok(crate::compression::RatePdfPredictor::SparseMatch {
        model: SparseMatchModel::new(
            *hash_bits,
            *min_len,
            *max_len,
            *gap_min,
            *gap_max,
            *base_mix,
            *confidence_scale,
        ),
    })
}

#[cfg(not(feature = "backend-match"))]
pub(super) fn build_pdf_predictor_sparse_match(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!(
        "{}",
        rate_backend_feature_error(RateBackendKind::SparseMatch)
    )
}

#[cfg(feature = "backend-ppmd")]
pub(super) fn build_pdf_predictor_ppmd(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Ppmd { order, memory_mb } = backend.plan() else {
        unreachable!("ppmd kernel used with non-ppmd plan")
    };
    Ok(crate::compression::RatePdfPredictor::Ppmd {
        model: PpmdModel::new(*order, *memory_mb),
    })
}

#[cfg(not(feature = "backend-ppmd"))]
pub(super) fn build_pdf_predictor_ppmd(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Ppmd))
}

#[cfg(feature = "backend-sequitur")]
pub(super) fn build_pdf_predictor_sequitur(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Sequitur { context_bytes } = backend.plan() else {
        unreachable!("sequitur kernel used with non-sequitur plan")
    };
    Ok(crate::compression::RatePdfPredictor::Sequitur {
        model: SequiturModel::new(*context_bytes),
    })
}

#[cfg(not(feature = "backend-sequitur"))]
pub(super) fn build_pdf_predictor_sequitur(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Sequitur))
}

#[cfg(feature = "backend-ctw")]
pub(super) fn build_pdf_predictor_ctw(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Ctw { depth } = backend.plan() else {
        unreachable!("ctw kernel used with non-ctw plan")
    };
    Ok(crate::compression::RatePdfPredictor::Ctw(
        crate::compression::CtwPredictor::new_ctw(*depth),
    ))
}

#[cfg(not(feature = "backend-ctw"))]
pub(super) fn build_pdf_predictor_ctw(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Ctw))
}

#[cfg(feature = "backend-ctw")]
pub(super) fn build_pdf_predictor_fac_ctw(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits: _,
        encoding_bits,
    } = backend.plan()
    else {
        unreachable!("fac-ctw kernel used with non-fac-ctw plan")
    };
    Ok(crate::compression::RatePdfPredictor::FacCtw(
        crate::compression::CtwPredictor::new_fac(*base_depth, (*encoding_bits).clamp(1, 8)),
    ))
}

#[cfg(not(feature = "backend-ctw"))]
pub(super) fn build_pdf_predictor_fac_ctw(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::FacCtw))
}

#[cfg(feature = "backend-mamba")]
pub(super) fn build_pdf_predictor_mamba(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Mamba { parsed_method, .. } = backend.plan() else {
        unreachable!("mamba kernel used with non-mamba plan")
    };
    Ok(crate::compression::RatePdfPredictor::Mamba(
        crate::compression::MambaPredictor::from_method_spec(parsed_method)?,
    ))
}

#[cfg(not(feature = "backend-mamba"))]
pub(super) fn build_pdf_predictor_mamba(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Mamba))
}

#[cfg(feature = "backend-rwkv")]
pub(super) fn build_pdf_predictor_rwkv(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Rwkv7 { parsed_method, .. } = backend.plan() else {
        unreachable!("rwkv kernel used with non-rwkv plan")
    };
    Ok(crate::compression::RatePdfPredictor::Rwkv(
        crate::compression::RwkvPredictor::from_method_spec(parsed_method)?,
    ))
}

#[cfg(not(feature = "backend-rwkv"))]
pub(super) fn build_pdf_predictor_rwkv(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Rwkv7))
}

#[cfg(feature = "backend-zpaq")]
pub(super) fn build_pdf_predictor_zpaq(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Zpaq { method } = backend.plan() else {
        unreachable!("zpaq kernel used with non-zpaq plan")
    };
    Ok(crate::compression::RatePdfPredictor::Zpaq(
        crate::compression::ZpaqPredictor::new(method.clone()),
    ))
}

#[cfg(not(feature = "backend-zpaq"))]
pub(super) fn build_pdf_predictor_zpaq(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Zpaq))
}

#[cfg(feature = "backend-mixture")]
pub(super) fn build_pdf_predictor_mixture(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    Ok(crate::compression::RatePdfPredictor::Mixture(
        crate::compression::MixturePredictor::new_from_compiled(backend)?,
    ))
}

#[cfg(not(feature = "backend-mixture"))]
pub(super) fn build_pdf_predictor_mixture(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Mixture))
}

#[cfg(feature = "backend-particle")]
pub(super) fn build_pdf_predictor_particle(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Particle { spec } = backend.plan() else {
        unreachable!("particle kernel used with non-particle plan")
    };
    Ok(crate::compression::RatePdfPredictor::Particle(
        ParticleRuntime::new(spec),
    ))
}

#[cfg(not(feature = "backend-particle"))]
pub(super) fn build_pdf_predictor_particle(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Particle))
}

#[cfg(feature = "backend-calibrated")]
pub(super) fn build_pdf_predictor_calibrated(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = backend.plan()
    else {
        unreachable!("calibrated kernel used with non-calibrated plan")
    };
    let base_backend = crate::spec::core::compiled_rate_backend_from_plan(base.clone())
        .map_err(|err| anyhow::anyhow!("failed to compile calibrated base backend plan: {err}"))?;
    Ok(crate::compression::RatePdfPredictor::Calibrated {
        base: Box::new(build_rate_pdf_predictor_via_kernel(&base_backend)?),
        core: CalibratorCore::new(*context, *bins, *learning_rate, *bias_clip),
        pdf: vec![1.0 / 256.0; 256],
        valid: false,
    })
}

#[cfg(not(feature = "backend-calibrated"))]
pub(super) fn build_pdf_predictor_calibrated(
    _backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!(
        "{}",
        rate_backend_feature_error(RateBackendKind::Calibrated)
    )
}
