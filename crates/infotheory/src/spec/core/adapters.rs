use super::*;

pub(crate) fn rate_plan_to_wrapper_rosa(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::RosaPlus { max_order } = plan else {
        unreachable!("rosa wrapper kernel used with non-rosa plan");
    };
    RateBackend::RosaPlus {
        max_order: *max_order,
    }
}

pub(crate) fn rate_plan_to_wrapper_match(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("match wrapper kernel used with non-match plan");
    };
    RateBackend::Match {
        hash_bits: *hash_bits,
        min_len: *min_len,
        max_len: *max_len,
        base_mix: *base_mix,
        confidence_scale: *confidence_scale,
    }
}

pub(crate) fn rate_plan_to_wrapper_sparse_match(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("sparse-match wrapper kernel used with non-sparse-match plan");
    };
    RateBackend::SparseMatch {
        hash_bits: *hash_bits,
        min_len: *min_len,
        max_len: *max_len,
        gap_min: *gap_min,
        gap_max: *gap_max,
        base_mix: *base_mix,
        confidence_scale: *confidence_scale,
    }
}

pub(crate) fn rate_plan_to_wrapper_ppmd(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd wrapper kernel used with non-ppmd plan");
    };
    RateBackend::Ppmd {
        order: *order,
        memory_mb: *memory_mb,
    }
}

pub(crate) fn rate_plan_to_wrapper_sequitur(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur wrapper kernel used with non-sequitur plan");
    };
    RateBackend::Sequitur {
        context_bytes: *context_bytes,
    }
}

pub(crate) fn rate_plan_to_wrapper_ctw(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw wrapper kernel used with non-ctw plan");
    };
    RateBackend::Ctw { depth: *depth }
}

pub(crate) fn rate_plan_to_wrapper_fac_ctw(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits,
        encoding_bits,
        msb_first,
    } = plan
    else {
        unreachable!("fac-ctw wrapper kernel used with non-fac-ctw plan");
    };
    RateBackend::FacCtw {
        base_depth: *base_depth,
        num_percept_bits: *num_percept_bits,
        encoding_bits: *encoding_bits,
        msb_first: Some(*msb_first),
    }
}

pub(crate) fn rate_plan_to_wrapper_zpaq(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq wrapper kernel used with non-zpaq plan");
    };
    RateBackend::Zpaq {
        method: crate::api::ZpaqMethodSpec::literal(method),
    }
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn rate_plan_to_wrapper_mamba(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Mamba { method, .. } = plan else {
        unreachable!("mamba wrapper kernel used with non-mamba plan");
    };
    RateBackend::MambaMethod {
        method: crate::mambazip::parse_method_spec(method)
            .expect("compiled mamba plan must retain a valid canonical method"),
    }
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn rate_plan_to_wrapper_rwkv7(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Rwkv7 { method, .. } = plan else {
        unreachable!("rwkv7 wrapper kernel used with non-rwkv7 plan");
    };
    RateBackend::Rwkv7Method {
        method: crate::rwkvzip::parse_method_spec(method)
            .expect("compiled rwkv plan must retain a valid canonical method"),
    }
}

pub(crate) fn rate_plan_to_wrapper_mixture(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Mixture {
        kind,
        schedule,
        alpha,
        decay,
        experts,
    } = plan
    else {
        unreachable!("mixture wrapper kernel used with non-mixture plan");
    };
    RateBackend::Mixture {
        spec: Arc::new(MixtureSpec {
            kind: *kind,
            schedule: *schedule,
            alpha: *alpha,
            decay: *decay,
            experts: experts
                .iter()
                .map(|expert| MixtureExpertSpec {
                    name: expert.name.clone(),
                    log_prior: expert.log_prior,
                    backend: rate_plan_to_wrapper(expert.backend.as_ref()),
                })
                .collect(),
        }),
    }
}

pub(crate) fn rate_plan_to_wrapper_particle(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle wrapper kernel used with non-particle plan");
    };
    RateBackend::Particle {
        spec: Arc::new(spec.clone()),
    }
}

pub(crate) fn rate_plan_to_wrapper_calibrated(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = plan
    else {
        unreachable!("calibrated wrapper kernel used with non-calibrated plan");
    };
    RateBackend::Calibrated {
        spec: Arc::new(CalibratedSpec {
            base: rate_plan_to_wrapper(base.as_ref()),
            context: *context,
            bins: *bins,
            learning_rate: *learning_rate,
            bias_clip: *bias_clip,
        }),
    }
}

pub(crate) fn compression_plan_to_wrapper_zpaq(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    let CompressionBackendPlan::Zpaq { method, threads } = plan else {
        unreachable!("zpaq compression wrapper kernel used with non-zpaq plan");
    };
    CompressionBackend::Zpaq {
        method: crate::api::ZpaqMethodSpec::literal(method),
        threads: std::num::NonZeroUsize::new(*threads)
            .expect("compiled zpaq compression plan must retain non-zero thread count"),
    }
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compression_plan_to_wrapper_rwkv7(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    let CompressionBackendPlan::Rwkv7 { method, coder, .. } = plan else {
        unreachable!("rwkv7 compression wrapper kernel used with non-rwkv7 plan");
    };
    CompressionBackend::Rwkv7 {
        method: crate::rwkvzip::parse_method_spec(method)
            .expect("compiled rwkv compression plan must retain a valid canonical method"),
        coder: *coder,
    }
}

pub(crate) fn compression_plan_to_wrapper_rate(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    let CompressionBackendPlan::Rate {
        rate_backend,
        coder,
        framing,
    } = plan
    else {
        unreachable!("rate compression wrapper kernel used with non-rate plan");
    };
    CompressionBackend::Rate {
        rate_backend: rate_plan_to_wrapper(rate_backend.as_ref()),
        coder: *coder,
        framing: *framing,
    }
}

pub(crate) fn rate_plan_contains_zpaq_false(_plan: &RateBackendPlan) -> bool {
    false
}

pub(crate) fn rate_plan_contains_zpaq_true(_plan: &RateBackendPlan) -> bool {
    true
}

pub(crate) fn rate_plan_contains_zpaq_mixture(plan: &RateBackendPlan) -> bool {
    let RateBackendPlan::Mixture { experts, .. } = plan else {
        unreachable!("mixture zpaq kernel used with non-mixture plan");
    };
    experts
        .iter()
        .any(|expert| rate_plan_contains_zpaq(expert.backend.as_ref()))
}

pub(crate) fn rate_plan_contains_zpaq_calibrated(plan: &RateBackendPlan) -> bool {
    let RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!("calibrated zpaq kernel used with non-calibrated plan");
    };
    rate_plan_contains_zpaq(base.as_ref())
}

pub(crate) fn rate_plan_display_label_rosa(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::RosaPlus { max_order } = plan else {
        unreachable!("rosa label kernel used with non-rosa plan");
    };
    format!("rosaplus(max_order={max_order})")
}

pub(crate) fn rate_plan_default_name_rosa(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::RosaPlus { max_order } = plan else {
        unreachable!("rosa default-name kernel used with non-rosa plan");
    };
    format!("rosa(mo={max_order})")
}

pub(crate) fn rate_plan_display_label_match(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("match label kernel used with non-match plan");
    };
    format!(
        "match(hash_bits={hash_bits},min_len={min_len},max_len={max_len},base_mix={base_mix},confidence_scale={confidence_scale})"
    )
}

pub(crate) fn rate_plan_default_name_match(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Match { .. } = plan else {
        unreachable!("match default-name kernel used with non-match plan");
    };
    "match".to_string()
}

pub(crate) fn rate_plan_display_label_sparse_match(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("sparse-match label kernel used with non-sparse-match plan");
    };
    format!(
        "sparse-match(hash_bits={hash_bits},min_len={min_len},max_len={max_len},gap_min={gap_min},gap_max={gap_max},base_mix={base_mix},confidence_scale={confidence_scale})"
    )
}

pub(crate) fn rate_plan_default_name_sparse_match(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::SparseMatch { .. } = plan else {
        unreachable!("sparse-match default-name kernel used with non-sparse-match plan");
    };
    "sparse-match".to_string()
}

pub(crate) fn rate_plan_display_label_ppmd(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd label kernel used with non-ppmd plan");
    };
    format!("ppmd(order={order},memory_mb={memory_mb})")
}

pub(crate) fn rate_plan_default_name_ppmd(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd default-name kernel used with non-ppmd plan");
    };
    format!("ppmd(o={order},m={memory_mb}MiB)")
}

pub(crate) fn rate_plan_display_label_sequitur(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur label kernel used with non-sequitur plan");
    };
    format!("sequitur(context_bytes={context_bytes})")
}

pub(crate) fn rate_plan_default_name_sequitur(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur default-name kernel used with non-sequitur plan");
    };
    format!("sequitur(ctx={context_bytes})")
}

pub(crate) fn rate_plan_display_label_ctw(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw label kernel used with non-ctw plan");
    };
    format!("ctw(depth={depth})")
}

pub(crate) fn rate_plan_default_name_ctw(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw default-name kernel used with non-ctw plan");
    };
    format!("ctw(d={depth})")
}

pub(crate) fn rate_plan_display_label_fac_ctw(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits,
        encoding_bits,
        msb_first,
    } = plan
    else {
        unreachable!("fac-ctw label kernel used with non-fac-ctw plan");
    };
    format!(
        "fac-ctw(base_depth={base_depth},num_percept_bits={num_percept_bits},encoding_bits={encoding_bits},msb_first={msb_first})"
    )
}

pub(crate) fn rate_plan_default_name_fac_ctw(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::FacCtw {
        base_depth,
        encoding_bits,
        msb_first,
        ..
    } = plan
    else {
        unreachable!("fac-ctw default-name kernel used with non-fac-ctw plan");
    };
    let order = if *msb_first { "msb" } else { "lsb" };
    format!("fac-ctw(d={base_depth},b={encoding_bits},{order})")
}

pub(crate) fn rate_plan_display_label_zpaq(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq label kernel used with non-zpaq plan");
    };
    format!("zpaq(method={method})")
}

pub(crate) fn rate_plan_default_name_zpaq(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq default-name kernel used with non-zpaq plan");
    };
    format!("zpaq(m={method})")
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn rate_plan_display_label_mamba(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Mamba { method, .. } = plan else {
        unreachable!("mamba label kernel used with non-mamba plan");
    };
    format!("mamba(method={method})")
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn rate_plan_default_name_mamba(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Mamba { method, .. } = plan else {
        unreachable!("mamba default-name kernel used with non-mamba plan");
    };
    format!("mamba({method})")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn rate_plan_display_label_rwkv7(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Rwkv7 { method, .. } = plan else {
        unreachable!("rwkv7 label kernel used with non-rwkv7 plan");
    };
    format!("rwkv7(method={method})")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn rate_plan_default_name_rwkv7(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Rwkv7 { method, .. } = plan else {
        unreachable!("rwkv7 default-name kernel used with non-rwkv7 plan");
    };
    format!("rwkv7({method})")
}

pub(crate) fn rate_plan_display_label_mixture(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Mixture { kind, .. } = plan else {
        unreachable!("mixture label kernel used with non-mixture plan");
    };
    match kind {
        MixtureKind::Bayes => "mixture:bayes".to_string(),
        MixtureKind::FadingBayes => "mixture:fading-bayes".to_string(),
        MixtureKind::Switching => "mixture:switching".to_string(),
        MixtureKind::Convex => "mixture:convex".to_string(),
        MixtureKind::Mdl => "mixture:mdl".to_string(),
        MixtureKind::Neural => "mixture:neural".to_string(),
    }
}

pub(crate) fn rate_plan_default_name_mixture(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Mixture { kind, .. } = plan else {
        unreachable!("mixture default-name kernel used with non-mixture plan");
    };
    let kind = match kind {
        MixtureKind::Bayes => "bayes",
        MixtureKind::FadingBayes => "fading",
        MixtureKind::Switching => "switch",
        MixtureKind::Convex => "convex",
        MixtureKind::Mdl => "mdl",
        MixtureKind::Neural => "neural",
    };
    format!("mix({kind})")
}

pub(crate) fn rate_plan_display_label_particle(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle label kernel used with non-particle plan");
    };
    format!(
        "particle(num_particles={},num_cells={})",
        spec.num_particles, spec.num_cells
    )
}

pub(crate) fn rate_plan_default_name_particle(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle default-name kernel used with non-particle plan");
    };
    format!("particle(n={},c={})", spec.num_particles, spec.num_cells)
}

pub(crate) fn rate_plan_display_label_calibrated(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        ..
    } = plan
    else {
        unreachable!("calibrated label kernel used with non-calibrated plan");
    };
    format!(
        "calibrated(context={context:?},bins={bins},learning_rate={learning_rate},bias_clip={bias_clip})"
    )
}

pub(crate) fn rate_plan_default_name_calibrated(plan: &RateBackendPlan) -> String {
    let RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!("calibrated default-name kernel used with non-calibrated plan");
    };
    format!(
        "calibrated({})",
        rate_backend_plan_default_name(base.as_ref())
    )
}

pub(crate) fn compression_plan_display_label_zpaq(plan: &CompressionBackendPlan) -> String {
    let CompressionBackendPlan::Zpaq { method, threads } = plan else {
        unreachable!("zpaq compression label kernel used with non-zpaq plan");
    };
    format!("zpaq(method={method},threads={threads})")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compression_plan_display_label_rwkv7(plan: &CompressionBackendPlan) -> String {
    let CompressionBackendPlan::Rwkv7 { method, coder, .. } = plan else {
        unreachable!("rwkv7 compression label kernel used with non-rwkv7 plan");
    };
    format!("rwkv7(coder={coder:?},method={method})")
}

pub(crate) fn compression_plan_display_label_rate(plan: &CompressionBackendPlan) -> String {
    let CompressionBackendPlan::Rate {
        rate_backend,
        coder,
        framing,
    } = plan
    else {
        unreachable!("rate compression label kernel used with non-rate plan");
    };
    format!(
        "{}(coder={coder:?},framing={framing:?})",
        crate::runtime::rate_backend_canonical_name(rate_backend.kind())
    )
}

pub(crate) fn encode_rate_payload_rosa(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::RosaPlus { max_order } = plan else {
        unreachable!("rosa encoder kernel used with non-rosa plan");
    };
    out.push(0);
    push_i64(out, *max_order);
}

pub(crate) fn encode_rate_payload_match(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("match encoder kernel used with non-match plan");
    };
    out.push(1);
    push_usize(out, *hash_bits);
    push_usize(out, *min_len);
    push_usize(out, *max_len);
    push_f64(out, *base_mix);
    push_f64(out, *confidence_scale);
}

pub(crate) fn encode_rate_payload_sparse_match(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("sparse-match encoder kernel used with non-sparse-match plan");
    };
    out.push(2);
    push_usize(out, *hash_bits);
    push_usize(out, *min_len);
    push_usize(out, *max_len);
    push_usize(out, *gap_min);
    push_usize(out, *gap_max);
    push_f64(out, *base_mix);
    push_f64(out, *confidence_scale);
}

pub(crate) fn encode_rate_payload_ppmd(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd encoder kernel used with non-ppmd plan");
    };
    out.push(3);
    push_usize(out, *order);
    push_usize(out, *memory_mb);
}

pub(crate) fn encode_rate_payload_sequitur(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur encoder kernel used with non-sequitur plan");
    };
    out.push(4);
    push_usize(out, *context_bytes);
}

pub(crate) fn encode_rate_payload_ctw(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw encoder kernel used with non-ctw plan");
    };
    out.push(5);
    push_usize(out, *depth);
}

pub(crate) fn encode_rate_payload_fac_ctw(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits,
        encoding_bits,
        msb_first,
    } = plan
    else {
        unreachable!("fac-ctw encoder kernel used with non-fac-ctw plan");
    };
    out.push(6);
    push_usize(out, *base_depth);
    push_usize(out, *num_percept_bits);
    push_usize(out, *encoding_bits);
    out.push(u8::from(*msb_first));
}

pub(crate) fn encode_rate_payload_zpaq(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq encoder kernel used with non-zpaq plan");
    };
    out.push(7);
    push_string(out, method);
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn encode_rate_payload_mamba(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Mamba { method, asset, .. } = plan else {
        unreachable!("mamba encoder kernel used with non-mamba plan");
    };
    out.push(8);
    push_string(out, method);
    push_asset_ref(out, asset.as_ref());
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn encode_rate_payload_rwkv7(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Rwkv7 { method, asset, .. } = plan else {
        unreachable!("rwkv7 encoder kernel used with non-rwkv7 plan");
    };
    out.push(9);
    push_string(out, method);
    push_asset_ref(out, asset.as_ref());
}

pub(crate) fn encode_rate_payload_mixture(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Mixture {
        kind,
        schedule,
        alpha,
        decay,
        experts,
    } = plan
    else {
        unreachable!("mixture encoder kernel used with non-mixture plan");
    };
    out.push(10);
    out.push(mixture_kind_tag(*kind));
    out.push(mixture_schedule_tag(*schedule));
    push_f64(out, *alpha);
    push_option_f64(out, *decay);
    push_varint(out, experts.len() as u64);
    for expert in experts.iter() {
        push_option_string(out, expert.name.as_deref());
        push_f64(out, expert.log_prior);
        encode_rate_backend_payload(expert.backend.as_ref(), out);
    }
}

pub(crate) fn encode_rate_payload_particle(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle encoder kernel used with non-particle plan");
    };
    out.push(11);
    encode_particle_spec(spec, out);
}

pub(crate) fn encode_rate_payload_calibrated(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = plan
    else {
        unreachable!("calibrated encoder kernel used with non-calibrated plan");
    };
    out.push(12);
    out.push(calibration_context_tag(*context));
    push_usize(out, *bins);
    push_f64(out, *learning_rate);
    push_f64(out, *bias_clip);
    encode_rate_backend_payload(base.as_ref(), out);
}

pub(crate) fn encode_compression_payload_zpaq(plan: &CompressionBackendPlan, out: &mut Vec<u8>) {
    let CompressionBackendPlan::Zpaq { method, threads } = plan else {
        unreachable!("zpaq compression encoder kernel used with non-zpaq plan");
    };
    out.push(0);
    push_string(out, method);
    push_usize(out, *threads);
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn encode_compression_payload_rwkv7(plan: &CompressionBackendPlan, out: &mut Vec<u8>) {
    let CompressionBackendPlan::Rwkv7 {
        method,
        asset,
        coder,
        ..
    } = plan
    else {
        unreachable!("rwkv7 compression encoder kernel used with non-rwkv7 plan");
    };
    out.push(1);
    push_string(out, method);
    push_asset_ref(out, asset.as_ref());
    out.push(coder_tag(*coder));
}

pub(crate) fn encode_compression_payload_rate(plan: &CompressionBackendPlan, out: &mut Vec<u8>) {
    let CompressionBackendPlan::Rate {
        rate_backend,
        coder,
        framing,
    } = plan
    else {
        unreachable!("rate compression encoder kernel used with non-rate plan");
    };
    out.push(2);
    out.push(coder_tag(*coder));
    out.push(framing_tag(*framing));
    encode_rate_backend_payload(rate_backend.as_ref(), out);
}
