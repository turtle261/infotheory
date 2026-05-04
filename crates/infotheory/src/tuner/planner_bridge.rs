use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_planner_family_controller(
    compiled: &crate::spec::CompiledTuneSpec,
    request: &TuneCommandRequest,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    verified_theorem: &VerifiedTheoremInputs,
    tune_started: Instant,
    baseline_eval: CandidateEvalResult,
    baseline_hash: String,
    baseline_bytes: Vec<u8>,
    baseline_key: CandidateCacheKey,
    cache: &mut HashMap<CandidateCacheKey, CandidateEvalResult>,
) -> Result<SearchSummary, String> {
    let contract =
        planner_controller_contract(compiled.controller(), compiled, dataset, verified_theorem)?;
    let reward_encoder = contract.reward_encoder(dataset, &baseline_eval, verified_theorem)?;
    let actions = compile_planner_mutation_actions(
        compiled.canonical_spec().baseline_candidate.clone(),
        &compiled.canonical_spec().bounds,
    )?;
    validate_theorem_planner_mutation_domain(&actions, &request.execution.theorem)?;
    let declared_actions = contract.interface.agent_actions.get();
    if actions.len() != declared_actions {
        return Err(format!(
            "action_alphabet_mismatch: compiled mutation alphabet has {} actions but controller.interface.agent_actions declares {}",
            actions.len(),
            declared_actions
        ));
    }
    let planner_run = compile_tuner_planner_run_spec(
        compiled.controller(),
        &contract,
        &reward_encoder,
        compiled,
        &SpecEnvironment::new(compiled.base_dir()),
    )?;
    let mut agent_runtime = build_tuner_planner_agent_runtime(
        compiled.controller(),
        &planner_run,
        &contract,
        encode_tuner_planner_percept(
            &contract.interface,
            Some(&baseline_eval),
            dataset.dataset_units,
            0,
            "baseline_initial_state",
            Some(&baseline_eval),
            Some(&baseline_bytes),
            Some(compiled.canonical_spec().eval_time_limit_seconds),
            false,
        )?,
    )?;
    let env = SpecEnvironment::new(compiled.base_dir());

    let mut best_candidate = compiled.canonical_spec().baseline_candidate.clone();
    let mut best_eval = baseline_eval.clone();
    let mut best_hash = baseline_hash;
    let mut best_bytes = baseline_bytes;
    let mut best_key = baseline_key;
    let mut current_candidate = best_candidate.clone();
    let mut current_eval = baseline_eval;
    let mut current_bytes = best_bytes.clone();
    let mut cache_hits: usize = 0;
    let mut cache_misses: usize = 1;
    let mut candidate_evaluations_executed: usize = 1;
    let mut proposals_attempted: usize = 0;
    let mut proposals_invalid: usize = 0;
    let mut self_loop_proposals: usize = 0;
    let mut successful_non_deployable: usize = 0;
    let mut final_best_move_reward: f64 = 0.0;
    let mut evaluations_seen: usize = 1;
    let max_evaluations = request.execution.max_evaluations.unwrap_or(usize::MAX);
    let mut stagnation_counter: usize = 0;
    let mut decision_steps: usize = 0;
    let mut warmstart_trace_refresh_merges: usize = 0;
    let total_rounds = if contract.warmstart_self_improvement {
        request.execution.self_improvement_rounds.max(1)
    } else {
        1
    };
    let trace_refresh_enabled = contract.warmstart_self_improvement
        && request.execution.warmstart_trace_refresh
        && total_rounds > 1;
    let mut refresh_teacher = contract
        .teacher
        .as_ref()
        .map(|teacher| teacher.traces.clone());
    let planner_return_bins = planner_return_bins(&planner_run);

    for round in 0..total_rounds {
        if trace_refresh_enabled
            && round > 0
            && let (Some(live_trace), Some(teacher)) = (
                agent_runtime.same_task_live_trace(),
                refresh_teacher.as_mut(),
            )
        {
            merge_warmstart_trace_deterministic(teacher, live_trace)?;
            agent_runtime.rebuild_warmstart_agent(&planner_run, teacher.clone())?;
            warmstart_trace_refresh_merges = warmstart_trace_refresh_merges.saturating_add(1);
        }
        let round_deadline_seconds = if contract.warmstart_self_improvement && total_rounds > 1 {
            Some(
                ((round + 1) as f64 / total_rounds as f64)
                    * compiled.canonical_spec().time_budget_seconds,
            )
        } else {
            None
        };
        loop {
            let elapsed_seconds = tune_started.elapsed().as_secs_f64();
            if evaluations_seen >= max_evaluations
                || elapsed_seconds >= compiled.canonical_spec().time_budget_seconds
                || round_deadline_seconds
                    .map(|deadline| elapsed_seconds >= deadline)
                    .unwrap_or(false)
            {
                break;
            }
            let action = agent_runtime.select_action();
            proposals_attempted = proposals_attempted.saturating_add(1);
            let action_index = usize::try_from(action)
                .map_err(|_| format!("planner action {action} does not fit usize"))?;
            if action_index >= actions.len() {
                proposals_invalid = proposals_invalid.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "invalid_action_index",
                    None,
                    None,
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            }
            let Some(proposed_candidate) =
                apply_planner_mutation_action(&current_candidate, &actions[action_index])?
            else {
                self_loop_proposals = self_loop_proposals.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "inapplicable_action",
                    None,
                    None,
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            };
            if reject_candidate_local_external_artifacts(&proposed_candidate).is_err()
                || validate_candidate_against_tune_bounds(
                    &proposed_candidate,
                    &compiled.canonical_spec().bounds,
                )
                .is_err()
            {
                proposals_invalid = proposals_invalid.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "candidate_out_of_bounds_or_external_artifact",
                    None,
                    None,
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            }
            let compiled_candidate = match proposed_candidate.compile_in(&env) {
                Ok(value) => value,
                Err(_) => {
                    proposals_invalid = proposals_invalid.saturating_add(1);
                    let percept = encode_tuner_planner_percept(
                        &contract.interface,
                        Some(&current_eval),
                        dataset.dataset_units,
                        0,
                        "candidate_compile_error",
                        None,
                        None,
                        None,
                        false,
                    )?;
                    agent_runtime.observe_transition(action, percept)?;
                    decision_steps = decision_steps.saturating_add(1);
                    continue;
                }
            };
            let candidate_bytes = compiled_candidate.canonical_bytes().as_slice().to_vec();
            let candidate_hash = crc32_hex(&candidate_bytes);
            let effective_limit = effective_eval_limit_seconds(
                compiled,
                tune_started,
                Some(compiled.canonical_spec().eval_time_limit_seconds),
                round_deadline_seconds,
            );
            if effective_limit <= 0.0 {
                break;
            }
            let candidate_profile = evaluator_profile.with_eval_time_limit(effective_limit);
            if candidate_bytes == current_bytes {
                self_loop_proposals = self_loop_proposals.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "self_loop_proposal",
                    None,
                    Some(&candidate_bytes),
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            }
            let cache_key = cache_key_for_candidate(
                compiled_candidate.canonical_bytes().as_slice(),
                &candidate_profile,
                &dataset.canonical_content_hash,
            )?;
            let candidate_eval = if let Some(cached) = cache.get(&cache_key) {
                cache_hits = cache_hits.saturating_add(1);
                cached.clone()
            } else {
                let evaluated = match evaluate_candidate(
                    &compiled_candidate,
                    dataset,
                    compiled_candidate.canonical_bytes().len(),
                    compiled.canonical_spec().min_throughput_bytes_per_second,
                    compiled.canonical_spec().max_memory_bytes,
                    effective_limit,
                    request.execution.rss_mode,
                    request.execution.evaluator_threads(),
                    verified_theorem.deterministic_table.as_ref(),
                ) {
                    Ok(value) => value,
                    Err(_) => error_eval_result(0.0, 0, effective_limit),
                };
                cache.insert(cache_key.clone(), evaluated.clone());
                cache_misses = cache_misses.saturating_add(1);
                candidate_evaluations_executed = candidate_evaluations_executed.saturating_add(1);
                evaluated
            };
            evaluations_seen = evaluations_seen.saturating_add(1);
            let incumbent_eval_before_step = current_eval.clone();
            let mut raw_improvement = 0.0f64;
            if !candidate_eval.deployable {
                successful_non_deployable = successful_non_deployable.saturating_add(1);
            } else {
                if key_less(
                    &candidate_eval,
                    &candidate_bytes,
                    &current_eval,
                    &current_bytes,
                ) {
                    raw_improvement =
                        (current_eval.objective_bits - candidate_eval.objective_bits).max(0.0);
                    current_candidate = proposed_candidate.clone();
                    current_eval = candidate_eval.clone();
                    current_bytes = candidate_bytes.clone();
                }
                if key_less(&candidate_eval, &candidate_bytes, &best_eval, &best_bytes) {
                    final_best_move_reward = raw_improvement;
                    best_candidate = proposed_candidate;
                    best_eval = candidate_eval.clone();
                    best_hash = candidate_hash;
                    best_bytes = candidate_bytes.clone();
                    best_key = cache_key;
                    stagnation_counter = 0;
                } else {
                    stagnation_counter = stagnation_counter.saturating_add(1);
                }
                if let Some(reset_after) = request.execution.stagnation_reset_evals
                    && stagnation_counter >= reset_after
                {
                    current_candidate = best_candidate.clone();
                    current_eval = best_eval.clone();
                    current_bytes = best_bytes.clone();
                    stagnation_counter = 0;
                }
            }
            let reward = reward_encoder.encode(raw_improvement)?;
            let diagnostic_token = if !candidate_eval.deployable {
                match candidate_eval.status {
                    CandidateEvalStatus::Timeout => "evaluator_timeout",
                    CandidateEvalStatus::Invalid => "evaluator_invalid",
                    CandidateEvalStatus::Error => "evaluator_error",
                    CandidateEvalStatus::Success => "nondeployable_candidate",
                }
            } else {
                "deployable_success"
            };
            let percept = encode_tuner_planner_percept(
                &contract.interface,
                Some(&incumbent_eval_before_step),
                dataset.dataset_units,
                reward,
                diagnostic_token,
                Some(&candidate_eval),
                Some(&candidate_bytes),
                Some(effective_limit),
                false,
            )?;
            agent_runtime.observe_transition(action, percept)?;
            decision_steps = decision_steps.saturating_add(1);
        }
    }

    Ok(SearchSummary {
        status: planner_completed_status(compiled.controller()),
        warning: None,
        best_candidate,
        best_candidate_crc32: best_hash,
        best_eval,
        cache_key_digest: best_key.digest_crc32(),
        cache_hits,
        cache_misses,
        candidate_evaluations_executed,
        non_warmup_candidate_results_seen: evaluations_seen,
        post_baseline_candidate_results_seen: evaluations_seen.saturating_sub(1),
        proposals_attempted,
        proposals_invalid,
        self_loop_proposals,
        successful_non_deployable,
        final_best_move_reward,
        controller_report: serde_json::json!({
            "kind": controller_kind_name(compiled.controller()),
            "runtime_path": planner_runtime_path_name(compiled.controller()),
            "planner_simulations_per_step": contract.planner_simulations_per_step,
            "simulations_per_decision_step": contract.planner_simulations_per_step,
            "decision_steps": decision_steps,
            "return_horizon": contract.return_horizon,
            "return_bins": planner_return_bins,
            "label_phase_period": contract.label_phase_period,
            "reward_semantics": contract.reward_semantics.name(),
            "reward_encoding": tuner_reward_encoding_name(&reward_encoder),
            "discount_factor": contract.discount_factor,
            "planner_run_controller_kind": planner_run.controller().kind_str(),
            "agent_runtime": planner_agent_runtime_name(compiled.controller()),
            "compiled_action_count": actions.len(),
            "compiled_action_paths": planner_action_paths(&actions),
            "declared_agent_actions": declared_actions,
            "observation_adapter": OBSERVATION_ADAPTER_DECLARATION,
            "scalar_representation": SCALAR_REPRESENTATION_DECLARATION,
            "warmstart_teacher_dataset": contract.teacher.as_ref().map(|value| serde_json::json!({
                "asset_id": value.asset_id.clone(),
                "resolved_path": value.resolved_path.clone(),
                "content_crc32": value.content_hash.clone(),
                "records": value.records,
            })),
            "warmstart_self_improvement_update": if contract.warmstart_self_improvement {
                Some("online_exact_h_step_delayed_label_update")
            } else {
                None::<&str>
            },
            "warmstart_trace_refresh_merges": warmstart_trace_refresh_merges,
        }),
    })
}

pub(super) fn planner_controller_contract(
    controller: &crate::spec::CompiledTuneController,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
) -> Result<PlannerControllerContract, String> {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(inner) => Ok(PlannerControllerContract {
            interface: inner.interface.clone(),
            planner_simulations_per_step: inner.planner_simulations_per_step,
            return_horizon: None,
            label_phase_period: None,
            discount_factor: 1.0,
            reward_semantics: PlannerRewardSemantics::ExactObjectiveDifference,
            clipping_interval: None,
            teacher: None,
            warmstart_self_improvement: false,
        }),
        crate::spec::CompiledTuneController::AiqiDiscounted(inner) => {
            if inner.max_improvement <= inner.min_improvement {
                return Err(
                    "normalized clipped reward contract requires max_improvement > min_improvement"
                        .to_string(),
                );
            }
            Ok(PlannerControllerContract {
                interface: inner.interface.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
                return_horizon: Some(inner.return_horizon),
                label_phase_period: None,
                discount_factor: inner.discount_factor,
                reward_semantics: PlannerRewardSemantics::NormalizedClippedImprovement,
                clipping_interval: Some((inner.min_improvement, inner.max_improvement)),
                teacher: None,
                warmstart_self_improvement: false,
            })
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(inner) => {
            let reward_certificate = verified_theorem.exact_reward_encoding.as_ref().ok_or_else(|| {
                "reward_encoding_unsafe: warm-start exact-J_H requires a verified exact_reward_encoding_certificate"
                    .to_string()
            })?;
            if !reward_certificate.is_identity_or_interval_encoding() {
                return Err(
                    "reward_encoding_unsafe: warm-start exact-J_H cannot use a non-identity finite_reward_map until teacher/live traces carry objective-difference labels or a verified decoder-based return encoder"
                        .to_string(),
                );
            }
            let teacher = load_warmstart_teacher_dataset(
                compiled,
                dataset,
                verified_theorem,
                &inner.warmstart_teacher_dataset_asset,
                &inner.interface,
                inner.return_horizon,
                inner.label_phase_period,
            )?;
            Ok(PlannerControllerContract {
                interface: inner.interface.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
                return_horizon: Some(inner.return_horizon),
                label_phase_period: Some(inner.label_phase_period),
                discount_factor: 1.0,
                reward_semantics: PlannerRewardSemantics::ExactObjectiveDifference,
                clipping_interval: None,
                teacher: Some(teacher),
                warmstart_self_improvement: true,
            })
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            Err("annealed controller does not use planner-family contract".to_string())
        }
    }
}

fn load_warmstart_teacher_dataset(
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
    asset_id: &str,
    interface: &crate::spec::TunePlannerInterfaceSpec,
    return_horizon: usize,
    label_phase_period: usize,
) -> Result<WarmstartTeacherDataset, String> {
    if asset_id == compiled.canonical_spec().input_asset {
        return Err(
            "warmstart_teacher_dataset_asset must be distinct from input_asset".to_string(),
        );
    }
    let binding = compiled
        .resolved_assets()
        .iter()
        .find(|entry| entry.id == asset_id)
        .ok_or_else(|| format!("unknown warmstart teacher dataset asset '{asset_id}'"))?;
    let AssetRef::Filesystem(path) = &binding.asset;
    let bytes = fs::read(path).map_err(|err| {
        format!(
            "failed to read warmstart_teacher_dataset_asset '{}': {err}",
            path.display()
        )
    })?;
    let hash = crc32_hex(&bytes);
    let traces = WarmStartExactJhTeacherDataset::from_json_slice(&bytes)
        .map_err(|err| format!("invalid warmstart_teacher_dataset_asset: {err}"))?;
    validate_warmstart_teacher_contract(
        compiled,
        dataset,
        verified_theorem,
        interface,
        return_horizon,
        label_phase_period,
        &traces,
    )?;
    let records = traces
        .traces
        .iter()
        .map(|trace| trace.transitions.len())
        .sum();
    Ok(WarmstartTeacherDataset {
        asset_id: asset_id.to_string(),
        resolved_path: path.to_string_lossy().to_string(),
        content_hash: hash,
        records,
        traces,
    })
}

pub(super) fn validate_warmstart_teacher_contract(
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
    interface: &crate::spec::TunePlannerInterfaceSpec,
    return_horizon: usize,
    label_phase_period: usize,
    teacher: &WarmStartExactJhTeacherDataset,
) -> Result<(), String> {
    let contract = &teacher.contract;
    if contract.schema_version != 1 {
        return Err("warmstart teacher schema_version must be 1".to_string());
    }
    let task_fingerprint = warmstart_task_fingerprint(compiled, dataset, verified_theorem)?;
    if contract.task_fingerprint != task_fingerprint {
        return Err(format!(
            "warmstart teacher task_fingerprint '{}' does not match current task '{}'",
            contract.task_fingerprint, task_fingerprint
        ));
    }
    if contract.action_alphabet_size != interface.agent_actions.get()
        || contract.observation_bits != interface.observation_bits
        || contract.observation_stream_len != interface.observation_stream_len.max(1)
        || contract.observation_key_mode
            != observation_key_mode_name(interface.observation_key_mode)
        || contract.reward_bits != interface.reward_bits
        || contract.return_horizon != return_horizon
        || contract.label_phase_period != label_phase_period
    {
        return Err(
            "warmstart teacher planner interface fingerprint does not match current controller"
                .to_string(),
        );
    }
    let expected_adapter_ref = OBSERVATION_ADAPTER_DECLARATION;
    let expected_adapter_hash = observation_adapter_content_hash()?;
    if contract.observation_adapter_spec_ref != expected_adapter_ref
        || contract.observation_adapter_content_crc32 != expected_adapter_hash
    {
        return Err(
            "warmstart teacher observation adapter fingerprint does not match current tuner observation adapter"
                .to_string(),
        );
    }
    let reward_cert = verified_theorem
        .exact_reward_encoding
        .as_ref()
        .ok_or_else(|| {
            "warmstart exact-J_H requires a verified exact_reward_encoding_certificate".to_string()
        })?;
    if contract.min_reward != 0
        || contract.max_reward != reward_cert.max_reward
        || contract.scalar_representation != reward_cert.scalar_representation
        || contract.exact_reward_encoding_certificate != reward_cert.base.content_hash
    {
        return Err("warmstart teacher reward/scalar fingerprint does not match verified exact reward encoder".to_string());
    }
    Ok(())
}

fn warmstart_task_fingerprint(
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
) -> Result<String, String> {
    let reward_hash = verified_theorem
        .exact_reward_encoding
        .as_ref()
        .map(|cert| cert.base.content_hash.as_str())
        .unwrap_or("unverified");
    let payload = serde_json::json!({
        "tune_canonical_crc32": crc32_hex(compiled.canonical_bytes().as_slice()),
        "dataset_crc32": dataset.canonical_content_hash,
        "dataset_kind": dataset_kind_name(dataset.kind),
        "bounds_crc32": bounds_hash(&compiled.canonical_spec().bounds)?,
        "controller_kind": controller_kind_name(compiled.controller()),
        "reward_certificate_crc32": reward_hash,
    });
    serde_json::to_vec(&payload)
        .map(|bytes| crc32_hex(&bytes))
        .map_err(|err| format!("failed to encode warmstart task fingerprint: {err}"))
}

fn compile_planner_mutation_actions(
    baseline: crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
) -> Result<Vec<PlannerMutationAction>, String> {
    let json = crate::spec::compression_backend_to_json_value(&baseline)
        .map_err(|err| format!("failed to serialize baseline for action compilation: {err}"))?;
    let range_map = bounds
        .parameter_ranges
        .iter()
        .map(|range| (range.parameter.clone(), (range.min, range.max)))
        .collect::<BTreeMap<_, _>>();
    let mut leaves = collect_numeric_leaves(&json);
    if !range_map.is_empty() {
        leaves.retain(|leaf| range_map.contains_key(&leaf.path));
    }
    let mut actions = Vec::<PlannerMutationAction>::new();
    for leaf in leaves {
        let deltas = match leaf.kind {
            NumericKind::Unsigned | NumericKind::Signed => [-1.0, 1.0],
            NumericKind::Float => [-0.05, 0.05],
        };
        for delta in deltas {
            actions.push(PlannerMutationAction::NumericStep {
                path: leaf.path.clone(),
                pointer: leaf.pointer.clone(),
                kind: leaf.kind,
                delta,
            });
        }
    }
    if actions.is_empty() {
        actions.push(PlannerMutationAction::Noop);
    }
    Ok(actions)
}

fn apply_planner_mutation_action(
    candidate: &crate::api::CompressionBackend,
    action: &PlannerMutationAction,
) -> Result<Option<crate::api::CompressionBackend>, String> {
    let PlannerMutationAction::NumericStep {
        path: _,
        pointer,
        kind,
        delta,
    } = action
    else {
        return Ok(None);
    };
    let mut json = crate::spec::compression_backend_to_json_value(candidate)
        .map_err(|err| format!("failed to serialize candidate for planner action: {err}"))?;
    let Some(slot) = json.pointer_mut(pointer) else {
        return Ok(None);
    };
    if !apply_numeric_delta(slot, *kind, *delta) {
        return Ok(None);
    }
    let mutated = crate::spec::parse_compression_backend_json(
        &json,
        Path::new("."),
        None,
        crate::compression::FramingMode::Framed,
    )
    .map_err(|err| format!("planner action produced unparsable candidate: {err}"))?;
    Ok(Some(mutated))
}

fn apply_numeric_delta(slot: &mut Value, kind: NumericKind, delta: f64) -> bool {
    match kind {
        NumericKind::Unsigned => {
            let Some(current) = slot.as_u64() else {
                return false;
            };
            let next = if delta >= 0.0 {
                current.saturating_add(delta.abs().ceil() as u64)
            } else {
                current.saturating_sub(delta.abs().ceil() as u64)
            };
            if next == current {
                return false;
            }
            *slot = Value::Number(serde_json::Number::from(next));
            true
        }
        NumericKind::Signed => {
            let Some(current) = slot.as_i64() else {
                return false;
            };
            let step = delta.abs().ceil() as i64;
            let next = if delta >= 0.0 {
                current.saturating_add(step)
            } else {
                current.saturating_sub(step)
            };
            if next == current {
                return false;
            }
            *slot = Value::Number(serde_json::Number::from(next));
            true
        }
        NumericKind::Float => {
            let Some(current) = slot.as_f64() else {
                return false;
            };
            let next = current + current.abs().max(1.0) * delta;
            if !next.is_finite() || (next - current).abs() <= f64::EPSILON {
                return false;
            }
            if let Some(number) = serde_json::Number::from_f64(next) {
                *slot = Value::Number(number);
                true
            } else {
                false
            }
        }
    }
}

fn planner_action_paths(actions: &[PlannerMutationAction]) -> Vec<String> {
    actions
        .iter()
        .map(|action| match action {
            PlannerMutationAction::NumericStep { path, delta, .. } => {
                format!("{path}:{delta:+}")
            }
            PlannerMutationAction::Noop => "noop".to_string(),
        })
        .collect()
}

pub(super) fn validate_theorem_planner_mutation_domain(
    actions: &[PlannerMutationAction],
    theorem: &TuneTheoremConfig,
) -> Result<(), String> {
    if !theorem_requests_exact_claims(theorem) {
        return Ok(());
    }
    if let Some(path) = actions.iter().find_map(|action| match action {
        PlannerMutationAction::NumericStep {
            path,
            kind: NumericKind::Float,
            ..
        } => Some(path.as_str()),
        _ => None,
    }) {
        return Err(format!(
            "theorem_finite_state_unsafe: planner mutation action '{path}' targets a floating-point leaf; exact theorem claims require an integer finite mutation grammar or a future certificate-enumerated finite float domain"
        ));
    }
    Ok(())
}

fn theorem_requests_exact_claims(theorem: &TuneTheoremConfig) -> bool {
    theorem.claim_exact_finite_mdp
        || theorem.claim_exact_observed_markov
        || theorem.claim_planner_convergence
}

pub(super) fn compile_tuner_planner_run_spec(
    controller: &crate::spec::CompiledTuneController,
    contract: &PlannerControllerContract,
    reward_encoder: &TunerRewardEncoder,
    compiled: &crate::spec::CompiledTuneSpec,
    env: &SpecEnvironment,
) -> Result<CompiledPlannerRunSpec, String> {
    let interface = PlannerInterfaceSpec {
        observation_bits: contract.interface.observation_bits,
        observation_stream_len: contract.interface.observation_stream_len,
        observation_key_mode: contract.interface.observation_key_mode,
        reward_bits: contract.interface.reward_bits,
        agent_actions: contract.interface.agent_actions,
        min_reward: reward_encoder.min_reward(),
        max_reward: reward_encoder.max_reward(),
        reward_offset: reward_encoder.reward_offset(),
    };
    let percept_bits = interface
        .observation_bits
        .saturating_mul(interface.observation_stream_len.max(1))
        .saturating_add(interface.reward_bits)
        .max(1);
    let controller_spec = match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(inner) => {
            ControllerSpec::McAixi(McAixiControllerSpec {
                predictor: RateBackend::FacCtw {
                    base_depth: TUNER_MCAIXI_FAC_CTW_BASE_DEPTH,
                    num_percept_bits: percept_bits,
                    encoding_bits: 1,
                },
                agent_horizon: TUNER_MCAIXI_HORIZON,
                num_simulations: inner.planner_simulations_per_step,
                mcts_strategy: MctsStrategy::RhoUct,
                exploration_exploitation_ratio: 1.4,
                discount_gamma: 1.0,
            })
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(inner) => {
            ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
                predictor: RateBackend::Ctw { depth: 8 },
                discount_gamma: inner.discount_factor,
                return_horizon: inner.return_horizon,
                return_bins: inner.return_bins,
                augmentation_period: inner.return_horizon,
                history_prune_keep_steps: None,
                baseline_exploration: 1.0e-12,
            })
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(inner) => {
            let return_bins =
                warmstart_return_bins(reward_encoder.max_reward(), inner.return_horizon)?;
            ControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 8 },
                return_horizon: inner.return_horizon,
                return_bins,
                label_phase_period: inner.label_phase_period,
                teacher_dataset_asset: inner.warmstart_teacher_dataset_asset.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
            })
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            return Err("annealed controller does not compile to planner-run agent".to_string());
        }
    };
    PlannerRunSpec {
        assets: compiled.canonical_spec().assets.clone(),
        environment: EnvironmentSpec::Builtin {
            builtin: BuiltinEnvironmentSpec::TunerBridge,
        },
        interface,
        controller: controller_spec,
        runtime: PlannerRuntimeSpec {
            random_seed: Some(compiled.canonical_spec().seed),
            learn_cycles: None,
            eval_cycles: None,
            terminate_lifetime: 1,
            log_every: 1,
            perf: false,
            vm_perf_only: false,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        },
    }
    .compile_in(env)
    .map_err(|err| format!("failed to compile tuner planner-run bridge: {err}"))
}

fn build_tuner_planner_agent_runtime(
    controller: &crate::spec::CompiledTuneController,
    planner_run: &CompiledPlannerRunSpec,
    contract: &PlannerControllerContract,
    initial_percept: PlannerEncodedPercept,
) -> Result<TunerPlannerAgentRuntime, String> {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => {
            Ok(TunerPlannerAgentRuntime::McAixi {
                agent: Agent::from_compiled_planner_run(planner_run)
                    .map_err(|err| err.to_string())?,
                prev_action: 0,
                prev_percept: initial_percept,
            })
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => {
            Ok(TunerPlannerAgentRuntime::AiqiDiscounted {
                agent: AiqiAgent::from_compiled_planner_run(planner_run)
                    .map_err(|err| err.to_string())?,
            })
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            let teacher = contract
                .teacher
                .as_ref()
                .ok_or_else(|| "warm-start controller missing teacher dataset".to_string())?;
            Ok(TunerPlannerAgentRuntime::WarmStartExactJh {
                agent: WarmStartExactJhAgent::from_compiled_planner_run(
                    planner_run,
                    teacher.traces.clone(),
                )
                .map_err(|err| err.to_string())?,
            })
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            Err("annealed controller does not instantiate planner-family agents".to_string())
        }
    }
}

pub(super) fn merge_warmstart_trace_deterministic(
    teacher: &mut WarmStartExactJhTeacherDataset,
    trace: WarmStartExactJhTeacherTrace,
) -> Result<(), String> {
    let key = warmstart_trace_key(&trace)?;
    let already_present = teacher
        .traces
        .iter()
        .map(warmstart_trace_key)
        .collect::<Result<BTreeSet<String>, String>>()?
        .contains(&key);
    if !already_present {
        teacher.traces.push(trace);
        teacher
            .traces
            .sort_by_key(|trace| warmstart_trace_key(trace).unwrap_or_default());
    }
    Ok(())
}

pub(super) fn warmstart_trace_key(trace: &WarmStartExactJhTeacherTrace) -> Result<String, String> {
    let value = serde_json::json!({
        "transitions": trace.transitions.iter().map(|transition| {
            serde_json::json!({
                "action": transition.action,
                "observations": transition.observations,
                "reward": transition.reward,
            })
        }).collect::<Vec<Value>>(),
    });
    serde_json::to_vec(&value)
        .map(|bytes| crc32_hex(&bytes))
        .map_err(|err| format!("failed to encode warm-start trace key: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_tuner_planner_percept(
    interface: &crate::spec::TunePlannerInterfaceSpec,
    incumbent_eval: Option<&CandidateEvalResult>,
    dataset_units: f64,
    reward: Reward,
    diagnostic_token: &str,
    candidate_eval: Option<&CandidateEvalResult>,
    candidate_bytes: Option<&[u8]>,
    effective_eval_limit_seconds: Option<f64>,
    terminal: bool,
) -> Result<PlannerEncodedPercept, String> {
    let raw = TunerRawObservation::from_runtime_step(
        incumbent_eval,
        dataset_units,
        candidate_eval,
        candidate_bytes,
        effective_eval_limit_seconds,
        diagnostic_token,
        terminal,
    );
    let stream_len = interface.observation_stream_len.max(1);
    let mut observations = Vec::with_capacity(stream_len);
    for index in 0..stream_len {
        observations.push(packed_observation_symbol(
            interface.observation_bits,
            index,
            raw.encoded_bytes(),
        ));
    }
    Ok(PlannerEncodedPercept {
        observations,
        reward,
    })
}

pub(super) struct TunerRawObservation {
    encoded: Vec<u8>,
}

impl TunerRawObservation {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_runtime_step(
        incumbent_eval: Option<&CandidateEvalResult>,
        dataset_units: f64,
        candidate_eval: Option<&CandidateEvalResult>,
        candidate_bytes: Option<&[u8]>,
        effective_eval_limit_seconds: Option<f64>,
        diagnostic_token: &str,
        terminal: bool,
    ) -> Self {
        let units = if dataset_units.is_finite() && dataset_units > 0.0 {
            dataset_units
        } else {
            1.0
        };
        let fail_flag = candidate_eval
            .map(|value| value.status != CandidateEvalStatus::Success || !value.deployable)
            .unwrap_or(true);
        let successful_eval =
            candidate_eval.filter(|value| value.status == CandidateEvalStatus::Success);
        let normalized_physical_size =
            successful_eval.map(|value| (value.compressed_bytes as f64) / units);
        let normalized_target_loss = successful_eval.and_then(|value| {
            if value.target_loss_bits.is_finite() {
                Some((value.target_loss_bits / units).max(0.0))
            } else {
                None
            }
        });
        let normalized_eval_time = match candidate_eval {
            Some(value) if value.status == CandidateEvalStatus::Timeout => Some(1.0),
            Some(value) if value.status == CandidateEvalStatus::Success => Some(
                normalize_eval_time(value.elapsed_seconds, effective_eval_limit_seconds),
            ),
            _ => None,
        };
        let physical_size_delta = match (incumbent_eval, successful_eval) {
            (Some(incumbent), Some(value))
                if incumbent.status == CandidateEvalStatus::Success
                    && incumbent.compressed_bytes > 0 =>
            {
                Some(
                    (incumbent.compressed_bytes as f64 - value.compressed_bytes as f64)
                        / incumbent.compressed_bytes as f64,
                )
            }
            _ => None,
        };
        let eval_time_delta = match (incumbent_eval, successful_eval) {
            (Some(incumbent), Some(value))
                if incumbent.status == CandidateEvalStatus::Success
                    && incumbent.elapsed_seconds.is_finite()
                    && value.elapsed_seconds.is_finite() =>
            {
                let denom = incumbent.elapsed_seconds.max(1.0e-9);
                Some((incumbent.elapsed_seconds - value.elapsed_seconds) / denom)
            }
            _ => None,
        };
        let signature_source = candidate_bytes.unwrap_or(diagnostic_token.as_bytes());
        let candidate_signature_crc32 = crc32_u32(signature_source);
        let mut encoded = Vec::<u8>::with_capacity(64);
        encoded.push(u8::from(fail_flag));
        push_optional_f64(&mut encoded, normalized_physical_size);
        push_optional_f64(&mut encoded, normalized_target_loss);
        push_optional_f64(&mut encoded, normalized_eval_time);
        push_optional_f64(&mut encoded, physical_size_delta);
        push_optional_f64(&mut encoded, eval_time_delta);
        encoded.extend_from_slice(&candidate_signature_crc32.to_le_bytes());
        encoded.push(u8::from(terminal));
        Self { encoded }
    }

    pub(super) fn encoded_bytes(&self) -> &[u8] {
        &self.encoded
    }
}

fn normalize_eval_time(elapsed_seconds: f64, effective_eval_limit_seconds: Option<f64>) -> f64 {
    if let Some(limit) = effective_eval_limit_seconds
        && limit > 0.0
    {
        return (elapsed_seconds / limit).clamp(0.0, 1.0);
    }
    if elapsed_seconds.is_finite() {
        elapsed_seconds.max(0.0)
    } else {
        1.0
    }
}

fn push_optional_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(number) => {
            out.push(1);
            out.extend_from_slice(&number.to_bits().to_le_bytes());
        }
        None => out.push(0),
    }
}

fn packed_observation_symbol(
    observation_bits: usize,
    index: usize,
    raw_payload: &[u8],
) -> PerceptVal {
    if observation_bits == 0 {
        return 0;
    }
    let offset = index.saturating_mul(std::mem::size_of::<u64>());
    let mut bytes = [0_u8; 8];
    if offset < raw_payload.len() {
        let available = (raw_payload.len() - offset).min(bytes.len());
        bytes[..available].copy_from_slice(&raw_payload[offset..offset + available]);
    } else {
        bytes[0] = 0xff;
    }
    let value = u64::from_le_bytes(bytes);
    if observation_bits >= 64 {
        value
    } else {
        value & ((1u64 << observation_bits) - 1)
    }
}

fn crc32_u32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

pub(super) fn max_nonnegative_reward_for_bits(reward_bits: usize) -> Result<Reward, String> {
    if reward_bits == 0 {
        return Err("reward_bits must be >= 1".to_string());
    }
    if reward_bits >= 63 {
        Ok(Reward::MAX)
    } else {
        Ok(((1u64 << reward_bits) - 1) as Reward)
    }
}

pub(super) fn exact_nonnegative_i64_from_f64(value: f64, label: &str) -> Result<Reward, String> {
    if !value.is_finite() {
        return Err(format!("{label} must be finite"));
    }
    if value < 0.0 {
        return Err(format!("{label} must be nonnegative"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(format!(
            "reward_encoding_unsafe: {label} must be exactly representable as a finite nonnegative integer"
        ));
    }
    if rounded > (Reward::MAX as f64) {
        return Err(format!(
            "reward_encoding_unsafe: {label} exceeds maximum representable reward"
        ));
    }
    Ok(rounded as Reward)
}

fn warmstart_return_bins(max_reward: Reward, return_horizon: usize) -> Result<usize, String> {
    let max_total = (max_reward as u128)
        .checked_mul(return_horizon as u128)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "warm-start exact-J_H return label range overflowed".to_string())?;
    usize::try_from(max_total)
        .map_err(|_| "warm-start exact-J_H return label range does not fit usize".to_string())
}

fn planner_return_bins(planner_run: &CompiledPlannerRunSpec) -> Option<usize> {
    match planner_run.controller() {
        crate::spec::CompiledPlannerController::AiqiDiscounted { return_bins, .. }
        | crate::spec::CompiledPlannerController::AiqiWarmstartExactJh { return_bins, .. } => {
            Some(*return_bins)
        }
        crate::spec::CompiledPlannerController::McAixi { .. } => None,
    }
}

fn planner_agent_runtime_name(controller: &crate::spec::CompiledTuneController) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => "aixi::agent::Agent",
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => "aixi::aiqi::AiqiAgent",
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            "aixi::warmstart::WarmStartExactJhAgent"
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => "annealer",
    }
}

fn tuner_reward_encoding_name(encoder: &TunerRewardEncoder) -> &'static str {
    match encoder {
        TunerRewardEncoder::ExactIntegerObjectiveDifference {
            objective_difference_to_symbol: Some(_),
            ..
        } => "exact_finite_reward_symbol_map",
        TunerRewardEncoder::ExactIntegerObjectiveDifference { .. } => {
            "exact_integer_objective_difference_interval"
        }
        TunerRewardEncoder::NormalizedClipped { .. } => "rounded_normalized_clipped_scalar",
    }
}

pub(super) fn normalized_clipped_improvement(
    raw_improvement: f64,
    min_improvement: f64,
    max_improvement: f64,
) -> Result<f64, String> {
    if !(min_improvement.is_finite()
        && max_improvement.is_finite()
        && max_improvement > min_improvement)
    {
        return Err(
            "normalized clipped improvement requires finite max_improvement > min_improvement"
                .to_string(),
        );
    }
    Ok(((raw_improvement - min_improvement) / (max_improvement - min_improvement)).clamp(0.0, 1.0))
}

pub(super) fn key_less(
    candidate_eval: &CandidateEvalResult,
    candidate_bytes: &[u8],
    incumbent_eval: &CandidateEvalResult,
    incumbent_bytes: &[u8],
) -> bool {
    candidate_eval.objective_bits < incumbent_eval.objective_bits
        || (candidate_eval.objective_bits == incumbent_eval.objective_bits
            && candidate_bytes < incumbent_bytes)
}
