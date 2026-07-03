use super::*;

pub(crate) fn compile_rate_plan_rosa(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::RosaPlus { max_order } => Ok(RateBackendPlan::RosaPlus {
            max_order: *max_order,
        }),
        _ => unreachable!("rosa kernel used with non-rosa backend"),
    }
}

pub(crate) fn compile_rate_plan_match(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Match {
            hash_bits,
            min_len,
            max_len,
            base_mix,
            confidence_scale,
        } => Ok(RateBackendPlan::Match {
            hash_bits: *hash_bits,
            min_len: *min_len,
            max_len: *max_len,
            base_mix: *base_mix,
            confidence_scale: *confidence_scale,
        }),
        _ => unreachable!("match kernel used with non-match backend"),
    }
}

pub(crate) fn compile_rate_plan_sparse_match(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::SparseMatch {
            hash_bits,
            min_len,
            max_len,
            gap_min,
            gap_max,
            base_mix,
            confidence_scale,
        } => Ok(RateBackendPlan::SparseMatch {
            hash_bits: *hash_bits,
            min_len: *min_len,
            max_len: *max_len,
            gap_min: *gap_min,
            gap_max: *gap_max,
            base_mix: *base_mix,
            confidence_scale: *confidence_scale,
        }),
        _ => unreachable!("sparse-match kernel used with non-sparse-match backend"),
    }
}

pub(crate) fn compile_rate_plan_ppmd(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Ppmd { order, memory_mb } => Ok(RateBackendPlan::Ppmd {
            order: *order,
            memory_mb: *memory_mb,
        }),
        _ => unreachable!("ppmd kernel used with non-ppmd backend"),
    }
}

pub(crate) fn compile_rate_plan_sequitur(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Sequitur { context_bytes } => {
            if *context_bytes < 2 {
                return Err(SpecError::new("sequitur context_bytes must be >= 2"));
            }
            Ok(RateBackendPlan::Sequitur {
                context_bytes: *context_bytes,
            })
        }
        _ => unreachable!("sequitur kernel used with non-sequitur backend"),
    }
}

pub(crate) fn compile_rate_plan_ctw(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Ctw { depth } => Ok(RateBackendPlan::Ctw { depth: *depth }),
        _ => unreachable!("ctw kernel used with non-ctw backend"),
    }
}

pub(crate) fn compile_rate_plan_fac_ctw(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits,
            encoding_bits,
            msb_first,
        } => {
            if !(1..=8).contains(encoding_bits) {
                return Err(SpecError::new(format!(
                    "fac-ctw encoding_bits must be in 1..=8, got {encoding_bits}"
                )));
            }
            Ok(RateBackendPlan::FacCtw {
                base_depth: *base_depth,
                num_percept_bits: *num_percept_bits,
                encoding_bits: *encoding_bits,
                msb_first: msb_first.unwrap_or(*encoding_bits == 8),
            })
        }
        _ => unreachable!("fac-ctw kernel used with non-fac-ctw backend"),
    }
}

pub(crate) fn compile_rate_plan_zpaq(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Zpaq { method } => {
            crate::validate_zpaq_rate_method(method.value())
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackendPlan::Zpaq {
                method: method.value().to_string(),
            })
        }
        _ => unreachable!("zpaq kernel used with non-zpaq backend"),
    }
}

#[cfg(feature = "backend-bit-reservoir")]
pub(crate) fn compile_rate_plan_bit_reservoir(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::BitReservoir { config } => {
            config
                .validate()
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackendPlan::BitReservoir {
                config: config.clone(),
            })
        }
        _ => unreachable!("bit-reservoir kernel used with non-bit-reservoir backend"),
    }
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn compile_rate_plan_mamba(
    backend: &RateBackend,
    env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::MambaMethod { method } => {
            let parsed_method =
                crate::spec::normalize_mamba_method_spec_for_base_dir(env.base_dir(), method)?;
            let method = crate::mambazip::canonical_method_string(&parsed_method)
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackendPlan::Mamba {
                method,
                asset: mamba_asset_ref(&parsed_method),
                parsed_method,
            })
        }
        _ => unreachable!("mamba kernel used with non-mamba backend"),
    }
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compile_rate_plan_rwkv7(
    backend: &RateBackend,
    env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Rwkv7Method { method } => {
            let parsed_method =
                crate::spec::normalize_rwkv_method_spec_for_base_dir(env.base_dir(), method)?;
            let method = crate::rwkvzip::canonical_method_string(&parsed_method)
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackendPlan::Rwkv7 {
                method,
                asset: rwkv_asset_ref(&parsed_method),
                parsed_method,
            })
        }
        _ => unreachable!("rwkv7 kernel used with non-rwkv7 backend"),
    }
}

pub(crate) fn compile_rate_plan_mixture(
    backend: &RateBackend,
    env: &SpecEnvironment,
    depth: usize,
) -> SpecResult<RateBackendPlan> {
    let RateBackend::Mixture { spec } = backend else {
        unreachable!("mixture kernel used with non-mixture backend");
    };
    let experts: Vec<_> = spec
        .experts
        .iter()
        .map(|expert| {
            Ok(RateBackendPlanExpert {
                name: expert.name.clone(),
                log_prior: expert.log_prior,
                backend: Arc::new(build_rate_plan(&expert.backend, env, depth - 1)?),
            })
        })
        .collect::<SpecResult<_>>()?;
    let canonical = MixtureSpec {
        kind: spec.kind,
        schedule: spec.schedule,
        alpha: spec.alpha,
        decay: spec.decay,
        experts: experts
            .iter()
            .map(|expert| MixtureExpertSpec {
                name: expert.name.clone(),
                log_prior: expert.log_prior,
                backend: rate_plan_to_wrapper(expert.backend.as_ref()),
            })
            .collect(),
    };
    canonical
        .validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(RateBackendPlan::Mixture {
        kind: canonical.kind,
        schedule: canonical.schedule,
        alpha: canonical.alpha,
        decay: canonical.decay,
        experts: experts.into_boxed_slice(),
    })
}

pub(crate) fn compile_rate_plan_particle(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    let RateBackend::Particle { spec } = backend else {
        unreachable!("particle kernel used with non-particle backend");
    };
    spec.validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(RateBackendPlan::Particle {
        spec: spec.as_ref().clone(),
    })
}

pub(crate) fn compile_rate_plan_calibrated(
    backend: &RateBackend,
    env: &SpecEnvironment,
    depth: usize,
) -> SpecResult<RateBackendPlan> {
    let RateBackend::Calibrated { spec } = backend else {
        unreachable!("calibrated kernel used with non-calibrated backend");
    };
    Ok(RateBackendPlan::Calibrated {
        context: spec.context,
        bins: spec.bins,
        learning_rate: spec.learning_rate,
        bias_clip: spec.bias_clip,
        base: Arc::new(build_rate_plan(&spec.base, env, depth - 1)?),
    })
}

pub(crate) fn compile_compression_plan_zpaq(
    backend: &CompressionBackend,
    _env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    match backend {
        CompressionBackend::Zpaq { method, threads } => {
            crate::zpaq_compress_to_vec(&[], method.value()).map_err(|err| {
                SpecError::new(format!(
                    "invalid zpaq compression method '{}': {err}",
                    method.value()
                ))
            })?;
            Ok(CompressionBackendPlan::Zpaq {
                method: method.value().to_string(),
                threads: threads.get(),
            })
        }
        _ => unreachable!("zpaq compression kernel used with non-zpaq backend"),
    }
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compile_compression_plan_rwkv7(
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    match backend {
        CompressionBackend::Rwkv7 { method, coder } => {
            let parsed_method =
                crate::spec::normalize_rwkv_method_spec_for_base_dir(env.base_dir(), method)?;
            let method = crate::rwkvzip::canonical_method_string(&parsed_method)
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(CompressionBackendPlan::Rwkv7 {
                method,
                asset: rwkv_asset_ref(&parsed_method),
                parsed_method,
                coder: *coder,
            })
        }
        _ => unreachable!("rwkv7 compression kernel used with non-rwkv7 backend"),
    }
}

pub(crate) fn compile_compression_plan_rate(
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    match backend {
        CompressionBackend::Rate {
            rate_backend,
            coder,
            framing,
        } => Ok(CompressionBackendPlan::Rate {
            rate_backend: Arc::new(build_rate_plan(rate_backend, env, MAX_MIXTURE_NESTING)?),
            coder: *coder,
            framing: *framing,
        }),
        _ => unreachable!("rate compression kernel used with non-rate backend"),
    }
}
