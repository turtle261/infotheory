use super::*;

pub(super) fn annealer_progress(tune_started: Instant, time_budget_seconds: f64) -> f64 {
    annealer_progress_from_elapsed(tune_started.elapsed().as_secs_f64(), time_budget_seconds)
}

pub(super) fn annealer_progress_from_elapsed(
    elapsed_seconds: f64,
    time_budget_seconds: f64,
) -> f64 {
    if time_budget_seconds <= 0.0 {
        return 1.0;
    }
    (elapsed_seconds / time_budget_seconds).clamp(0.0, 1.0)
}

pub(super) fn annealer_temperature(progress: f64) -> f64 {
    let u = progress.clamp(0.0, 1.0);
    let temperature = ANNEALER_T_MIN_BITS * (ANNEALER_T0_BITS / ANNEALER_T_MIN_BITS).powf(1.0 - u);
    temperature.clamp(ANNEALER_T_MIN_BITS, ANNEALER_T0_BITS)
}

pub(super) fn annealer_active_radius(max_mutation_radius: usize, temperature: f64) -> usize {
    ((max_mutation_radius as f64) * temperature)
        .floor()
        .max(1.0) as usize
}

pub(super) fn annealer_runtime_path_name(profile: AnnealerKernelProfile) -> &'static str {
    match profile {
        AnnealerKernelProfile::ReversibleElementaryMetropolis => "reversible_elementary_metropolis",
        AnnealerKernelProfile::CompiledUniformMetropolisHastings => {
            "compiled_uniform_metropolis_hastings"
        }
    }
}

pub(super) fn annealer_acceptance_probability(
    profile: AnnealerKernelProfile,
    delta: f64,
    temperature: f64,
    proposal: &AnnealedProposal,
) -> Result<f64, String> {
    if proposal.forward_raw_action_count == 0 || proposal.forward_total_raw_actions == 0 {
        return Ok(0.0);
    }
    let metropolis = (-delta / temperature).exp();
    match profile {
        AnnealerKernelProfile::ReversibleElementaryMetropolis => {
            let forward = (proposal.forward_raw_action_count as u128)
                * (proposal.reverse_total_raw_actions as u128);
            let reverse = (proposal.reverse_raw_action_count as u128)
                * (proposal.forward_total_raw_actions as u128);
            if forward != reverse {
                return Err(
                    "compiled elementary proposal kernel failed reversibility check".to_string(),
                );
            }
            Ok(metropolis.clamp(0.0, 1.0))
        }
        AnnealerKernelProfile::CompiledUniformMetropolisHastings => {
            if proposal.reverse_raw_action_count == 0 || proposal.reverse_total_raw_actions == 0 {
                return Ok(0.0);
            }
            let hastings_ratio = ((proposal.reverse_raw_action_count as f64)
                * (proposal.forward_total_raw_actions as f64))
                / ((proposal.forward_raw_action_count as f64)
                    * (proposal.reverse_total_raw_actions as f64));
            Ok((metropolis * hastings_ratio).clamp(0.0, 1.0))
        }
    }
}

pub(super) fn sample_annealed_proposal(
    candidate: &crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
    max_mutation_radius: usize,
    active_radius: usize,
    env: &SpecEnvironment,
    rng: &mut RandomGenerator,
) -> Result<AnnealedProposalDraw, String> {
    let current_compiled = candidate
        .compile_in(env)
        .map_err(|err| format!("failed to compile current annealer candidate: {err}"))?;
    let current_canonical_bytes = current_compiled.canonical_bytes().as_slice().to_vec();
    let forward = compile_canonical_proposal_kernel(
        candidate,
        bounds,
        max_mutation_radius,
        active_radius,
        env,
        &current_canonical_bytes,
    )?;
    if forward.transitions.is_empty() {
        return Ok(AnnealedProposalDraw::Exhausted);
    }
    let Some(proposed) = forward.sample(rng) else {
        return Ok(AnnealedProposalDraw::SelfLoop);
    };
    let reverse = compile_canonical_proposal_kernel(
        &proposed.candidate,
        bounds,
        max_mutation_radius,
        active_radius,
        env,
        &proposed.candidate_canonical_bytes,
    )?;
    Ok(AnnealedProposalDraw::Proposal(AnnealedProposal {
        candidate: proposed.candidate.clone(),
        forward_raw_action_count: proposed.raw_action_count,
        forward_total_raw_actions: forward.total_raw_actions,
        reverse_raw_action_count: reverse
            .proposal_mass_to_canonical_bytes(&current_canonical_bytes),
        reverse_total_raw_actions: reverse.total_raw_actions,
    }))
}

pub(super) fn compile_canonical_proposal_kernel(
    candidate: &crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
    max_mutation_radius: usize,
    active_radius: usize,
    env: &SpecEnvironment,
    current_canonical_bytes: &[u8],
) -> Result<CanonicalProposalKernel, String> {
    let json = crate::spec::compression_backend_to_json_value(candidate)
        .map_err(|err| format!("failed to serialize candidate for proposal kernel: {err}"))?;
    let range_map = bounds
        .parameter_ranges
        .iter()
        .map(|range| (range.parameter.clone(), (range.min, range.max)))
        .collect::<BTreeMap<_, _>>();
    let mut leaves = collect_numeric_leaves(&json);
    if !range_map.is_empty() {
        leaves.retain(|leaf| range_map.contains_key(&leaf.path));
    }
    leaves.retain(|leaf| matches!(leaf.kind, NumericKind::Unsigned | NumericKind::Signed));
    let total_raw_actions = (leaves.len() as u64)
        .saturating_mul(2)
        .saturating_mul(max_mutation_radius.max(1) as u64);
    let mut transitions = BTreeMap::<Vec<u8>, CanonicalProposal>::new();
    for leaf in &leaves {
        let Some((min_bound, max_bound)) =
            integer_leaf_bounds(leaf.kind, range_map.get(&leaf.path).copied())
        else {
            continue;
        };
        for magnitude in 1..=max_mutation_radius.max(1) {
            for sign in [-1i128, 1i128] {
                if magnitude > active_radius {
                    continue;
                }
                let mut next_json = json.clone();
                let delta = sign * (magnitude as i128);
                if !apply_integer_descriptor(&mut next_json, leaf, min_bound, max_bound, delta) {
                    continue;
                }
                let parsed = match crate::spec::parse_compression_backend_json(
                    &next_json,
                    Path::new("."),
                    None,
                    crate::compression::FramingMode::Framed,
                ) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                if reject_candidate_local_external_artifacts(&parsed).is_err()
                    || validate_candidate_against_tune_bounds(&parsed, bounds).is_err()
                {
                    continue;
                }
                let compiled = match parsed.compile_in(env) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                let candidate_canonical_bytes = compiled.canonical_bytes().as_slice().to_vec();
                if candidate_canonical_bytes == current_canonical_bytes {
                    continue;
                }
                transitions
                    .entry(candidate_canonical_bytes.clone())
                    .and_modify(|proposal| {
                        proposal.raw_action_count = proposal.raw_action_count.saturating_add(1);
                    })
                    .or_insert(CanonicalProposal {
                        candidate: parsed,
                        candidate_canonical_bytes,
                        raw_action_count: 1,
                    });
            }
        }
    }
    Ok(CanonicalProposalKernel {
        transitions: transitions.into_values().collect(),
        total_raw_actions,
    })
}

pub(super) fn integer_leaf_bounds(
    kind: NumericKind,
    range: Option<(f64, f64)>,
) -> Option<(i128, i128)> {
    let (type_min, type_max) = match kind {
        NumericKind::Unsigned => (0i128, u64::MAX as i128),
        NumericKind::Signed => (i64::MIN as i128, i64::MAX as i128),
        NumericKind::Float => return None,
    };
    let (min, max) = match range {
        Some((min, max)) => (
            (min.ceil() as i128).clamp(type_min, type_max),
            (max.floor() as i128).clamp(type_min, type_max),
        ),
        None => (type_min, type_max),
    };
    (min <= max).then_some((min, max))
}

pub(super) fn apply_integer_descriptor(
    json: &mut Value,
    leaf: &NumericLeaf,
    min_bound: i128,
    max_bound: i128,
    delta: i128,
) -> bool {
    let Some(slot) = json.pointer_mut(&leaf.pointer) else {
        return false;
    };
    let current = match leaf.kind {
        NumericKind::Unsigned => slot.as_u64().map(i128::from),
        NumericKind::Signed => slot.as_i64().map(i128::from),
        NumericKind::Float => None,
    };
    let Some(current) = current else {
        return false;
    };
    let Some(next) = current.checked_add(delta) else {
        return false;
    };
    if next < min_bound || next > max_bound || next == current {
        return false;
    }
    match leaf.kind {
        NumericKind::Unsigned => {
            let Ok(value) = u64::try_from(next) else {
                return false;
            };
            *slot = Value::Number(serde_json::Number::from(value));
            true
        }
        NumericKind::Signed => {
            let Ok(value) = i64::try_from(next) else {
                return false;
            };
            *slot = Value::Number(serde_json::Number::from(value));
            true
        }
        NumericKind::Float => false,
    }
}

pub(super) fn collect_numeric_leaves(root: &Value) -> Vec<NumericLeaf> {
    let mut out = Vec::<NumericLeaf>::new();
    collect_numeric_leaves_inner(root, "", "", &mut out);
    out
}

fn collect_numeric_leaves_inner(
    value: &Value,
    path_prefix: &str,
    pointer_prefix: &str,
    out: &mut Vec<NumericLeaf>,
) {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let next_path = if path_prefix.is_empty() {
                    key.to_string()
                } else {
                    format!("{path_prefix}.{key}")
                };
                let escaped = key.replace('~', "~0").replace('/', "~1");
                let next_pointer = if pointer_prefix.is_empty() {
                    format!("/{escaped}")
                } else {
                    format!("{pointer_prefix}/{escaped}")
                };
                collect_numeric_leaves_inner(child, &next_path, &next_pointer, out);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                let next_path = format!("{path_prefix}[{index}]");
                let next_pointer = if pointer_prefix.is_empty() {
                    format!("/{index}")
                } else {
                    format!("{pointer_prefix}/{index}")
                };
                collect_numeric_leaves_inner(child, &next_path, &next_pointer, out);
            }
        }
        Value::Number(number) => {
            let kind = if number.as_u64().is_some() {
                Some(NumericKind::Unsigned)
            } else if number.as_i64().is_some() {
                Some(NumericKind::Signed)
            } else if number.as_f64().is_some() {
                Some(NumericKind::Float)
            } else {
                None
            };
            if let Some(kind) = kind {
                out.push(NumericLeaf {
                    path: path_prefix.to_string(),
                    pointer: pointer_prefix.to_string(),
                    kind,
                });
            }
        }
        _ => {}
    }
}
