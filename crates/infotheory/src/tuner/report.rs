use super::*;

pub(super) fn executor_controls_report(config: &TuneExecutionConfig) -> Value {
    let cgroup_peak_available = cgroup_peak_memory_bytes().is_some();
    let effective_measurement = match config.rss_mode {
        PeakMemoryMode::ProcessRssPeak => "process_rss_peak",
        PeakMemoryMode::BackendReported if cgroup_peak_available => "cgroup_peak_memory",
        PeakMemoryMode::BackendReported => "process_rss_peak_fallback",
        PeakMemoryMode::HybridStrictMax if cgroup_peak_available => {
            "max_process_rss_peak_cgroup_peak"
        }
        PeakMemoryMode::HybridStrictMax => "process_rss_peak_fallback",
    };
    serde_json::json!({
        "cpu_affinity": {
            "requested": config.cpu_affinity,
            "applied_to_current_process": config.cpu_affinity.is_some(),
        },
        "threads": {
            "requested": config.threads,
            "parent_controller_threads": 1usize,
            "evaluator_threads": config.evaluator_threads(),
            "rayon_global_pool_configured_in_parent": false,
            "worker_isolation_mode": "spawn_exec_worker",
            "evaluator_determinism": config.evaluator_determinism(),
        },
        "log_path": config.log_path,
        "diagnostic_chunk_bytes": config.diagnostic_chunk_bytes,
        "rss_mode": {
            "requested": peak_memory_mode_name(config.rss_mode),
            "effective_measurement": effective_measurement,
            "cgroup_peak_memory_available": cgroup_peak_available,
            "backend_peak_memory_report_available": cgroup_peak_available,
        },
    })
}

pub(super) fn diagnostic_chunking_report(
    dataset: &LoadedDataset,
    chunk_bytes: Option<usize>,
) -> Value {
    let charged_payload_bytes = dataset.raw_bytes.len();
    let Some(chunk_bytes) = chunk_bytes else {
        return serde_json::json!({
            "enabled": false,
            "chunk_bytes": null,
            "charged_payload_bytes": charged_payload_bytes,
            "chunk_count": 0usize,
            "last_chunk_bytes": 0usize,
            "affects_objective": false,
            "affects_canonical_candidate_identity": false,
        });
    };
    let chunk_count = if charged_payload_bytes == 0 {
        0usize
    } else {
        (charged_payload_bytes / chunk_bytes)
            + usize::from(!charged_payload_bytes.is_multiple_of(chunk_bytes))
    };
    let last_chunk_bytes = if charged_payload_bytes == 0 {
        0usize
    } else {
        let remainder = charged_payload_bytes % chunk_bytes;
        if remainder == 0 {
            chunk_bytes
        } else {
            remainder
        }
    };
    serde_json::json!({
        "enabled": true,
        "chunk_bytes": chunk_bytes,
        "charged_payload_bytes": charged_payload_bytes,
        "chunk_count": chunk_count,
        "last_chunk_bytes": last_chunk_bytes,
        "affects_objective": false,
        "affects_canonical_candidate_identity": false,
    })
}

pub(super) fn causal_profile_report(dataset: &LoadedDataset) -> Value {
    let Some(profile) = &dataset.causal_profile else {
        return serde_json::json!(null);
    };
    let domains = profile
        .domains
        .iter()
        .map(|(domain, support)| match support {
            CausalTargetDomain::ByteAlphabet => serde_json::json!({
                "domain": domain,
                "kind": "byte_alphabet",
                "normalization": "per_byte_symbol",
                "symbol_width_bytes": profile.byte_alphabet_symbol_width,
                "symbols": 256usize,
            }),
            CausalTargetDomain::EnumeratedPayloads { payloads } => serde_json::json!({
                "domain": domain,
                "kind": "enumerated_payloads",
                "normalization": "finite_payload_set",
                "payloads": payloads.len(),
            }),
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "channel_set": profile.channel_set.iter().collect::<Vec<_>>(),
        "domain_support_crc32": profile.domain_support_hash,
        "header_profile_crc32": profile.header_profile_hash,
        "collection_policy": profile.collection_policy,
        "action_alphabet_size": profile.action_alphabet_size,
        "percept_schema_channels": profile
            .percept_channels
            .iter()
            .map(|pair| {
                serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
            })
            .collect::<Vec<_>>(),
        "reward_encoding": serde_json::json!({
            "channel": profile.reward_channel.channel.as_str(),
            "domain": profile.reward_channel.domain.as_str(),
        }),
        "terminal_encoding": serde_json::json!({
            "channel": profile.terminal_channel.channel.as_str(),
            "domain": profile.terminal_channel.domain.as_str(),
        }),
        "event_grammar": serde_json::json!({
            "context_channels": profile.event_grammar.context_channels.iter().collect::<Vec<_>>(),
            "observe_target_no_score": profile
                .event_grammar
                .observe_target_no_score
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
            "target": profile
                .event_grammar
                .target
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
        }),
        "domains": domains,
        "structural_conditioning": "zero_cost_event_descriptor_tags_v1",
        "byte_alphabet_expansion_policy": "multi_byte_targets_expand_to_single_byte_events",
    })
}

pub(super) fn evaluator_execution_model(
    deterministic_table: Option<&VerifiedDeterministicEvaluatorTable>,
) -> &'static str {
    if deterministic_table.is_some() {
        "deterministic_table"
    } else if cfg!(unix) {
        "spawn_exec_worker_process_isolated_operational"
    } else {
        "unsupported_non_unix"
    }
}

pub(super) fn theorem_timing_basis(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
) -> &'static str {
    match theorem.timing_certification_tier {
        TimingCertificationTier::DeterministicTable if verified.deterministic_table.is_some() => {
            "verified_deterministic_evaluator_table"
        }
        TimingCertificationTier::RealTime if verified.determinism_deadline.is_some() => {
            "verified_real_time_deadline_certificate"
        }
        _ => "operational_only_uncertified",
    }
}

pub(super) fn dataset_kind_name(value: DatasetKind) -> &'static str {
    match value {
        DatasetKind::PassiveBytes => "passive_bytes",
        DatasetKind::InteractiveTrace => "interactive_trace",
        DatasetKind::CausalPrefixDataset => "causal_prefix_dataset",
    }
}

pub(super) fn objective_target_name(value: ObjectiveTarget) -> &'static str {
    match value {
        ObjectiveTarget::PassiveAc => "passive_ac",
        ObjectiveTarget::InteractiveCausalAc => "interactive_causal_ac",
        ObjectiveTarget::PlannerDeployableModel => "planner_deployable_model",
    }
}

pub(super) fn planner_deployability_report(
    enabled: bool,
    model_state_bytes: usize,
    eval_latency_seconds: f64,
    deployable_under_executor_limits: bool,
) -> Value {
    serde_json::json!({
        "enabled": enabled,
        "primary_score": "8L_B(z)+ell_D(z)",
        "diagnostics_are_secondary": true,
        "model_state_bytes": model_state_bytes,
        "snapshot_bytes": model_state_bytes,
        "clone_latency_seconds": 0.0,
        "update_latency_seconds": eval_latency_seconds.max(0.0),
        "restore_latency_seconds": 0.0,
        "sampling_support": true,
        "exact_log_probability_support": true,
        "deployable_under_executor_limits": deployable_under_executor_limits,
    })
}

pub(super) fn observation_key_mode_name(
    mode: crate::aixi::common::ObservationKeyMode,
) -> &'static str {
    match mode {
        crate::aixi::common::ObservationKeyMode::First => "first",
        crate::aixi::common::ObservationKeyMode::Last => "last",
        crate::aixi::common::ObservationKeyMode::StreamHash => "stream_hash",
        crate::aixi::common::ObservationKeyMode::FullStream => "full_stream",
    }
}

pub(super) fn controller_kind_name(
    controller: &crate::spec::CompiledTuneController,
) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => "annealed_hill_climbing",
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => "mc_aixi_fac_ctw",
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => "aiqi_discounted",
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => "aiqi_warmstart_exact_jh",
    }
}

pub(super) fn planner_completed_status(
    controller: &crate::spec::CompiledTuneController,
) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => "completed_mc_aixi_fac_ctw",
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => "completed_aiqi_discounted",
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            "completed_aiqi_warmstart_exact_jh"
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => "completed_annealed",
    }
}

pub(super) fn planner_runtime_path_name(
    controller: &crate::spec::CompiledTuneController,
) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => {
            "finite_mutation_agent_bridge_mcaixi_fac_ctw"
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => {
            "finite_mutation_agent_bridge_aiqi_discounted"
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            "finite_mutation_agent_bridge_aiqi_warmstart_exact_jh"
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            "reversible_elementary_metropolis"
        }
    }
}

pub(super) fn theorem_claims_report(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    dataset: &LoadedDataset,
    search: &SearchSummary,
) -> Value {
    serde_json::json!({
        "exact_finite_mdp": theorem_claim_status(
            theorem.claim_exact_finite_mdp,
            exact_finite_mdp_missing_prereqs(theorem, verified, controller, search),
            &[
                "Assumptions finite-Z/no-hidden-state are represented by finite compiled mutation alphabet",
                "Timing tier is theorem-admissible",
                "Verified determinism/deadline certificate is present unless verified deterministic_table is used",
            ],
        ),
        "exact_observed_markov": theorem_claim_status(
            theorem.claim_exact_observed_markov,
            exact_observed_markov_missing_prereqs(theorem, verified, controller, search),
            &[
                "All exact finite-MDP prerequisites hold",
                "Exact-state observation encoder reference is present",
                "Verified exact-state observation certificate is present",
            ],
        ),
        "planner_convergence": theorem_claim_status(
            theorem.claim_planner_convergence,
            planner_convergence_missing_prereqs(theorem, verified, controller, search),
            &[
                "Controller is MC-AIXI(FAC-CTW)",
                "Exact finite-MDP prerequisites hold",
                "Exact objective-difference reward semantics are active",
            ],
        ),
        "refs": {
            "proof_boundary": theorem_proof_boundary_report(verified),
            "timing_certification_tier": timing_tier_name(theorem.timing_certification_tier),
            "determinism_deadline_certificate": theorem.determinism_deadline_certificate,
            "observation_adapter_spec_ref": theorem.observation_adapter_spec_ref.as_deref().unwrap_or(OBSERVATION_ADAPTER_DECLARATION),
            "exact_state_encoder_spec_ref": theorem.exact_state_encoder_spec_ref,
            "exact_state_observation_basis": {
                "verified_certificate": verified.exact_state_observation.is_some(),
                "certificate": verified.exact_state_observation.as_ref().map(VerifiedExactStateObservationCertificate::to_json_value),
            },
            "scalar_representation_ref": theorem.scalar_representation_ref.as_deref().unwrap_or(SCALAR_REPRESENTATION_DECLARATION),
            "dataset_kind": dataset_kind_name(dataset.kind),
            "dataset_lowering_version": dataset.lowering_version,
            "target_domain_support_hash": dataset.target_domain_support_hash,
            "causal_header_profile_hash": dataset.causal_header_profile_hash,
            "verified": verified.to_json_value(),
        }
    })
}

fn theorem_proof_boundary_report(verified: &VerifiedTheoremInputs) -> Value {
    serde_json::json!({
        "finite_planner_state_certificate": certificate_boundary_kind(verified.finite_planner_state.as_ref()),
        "no_hidden_state_certificate": certificate_boundary_kind(verified.no_hidden_state.as_ref()),
        "determinism_deadline_certificate": certificate_boundary_kind(verified.determinism_deadline.as_ref()),
        "exact_reward_encoding_certificate": if verified.exact_reward_encoding.is_some() {
            "certified_by_checked_artifact"
        } else {
            "operational_only_uncertified"
        },
        "exact_state_observation_certificate": if verified.exact_state_observation.is_some() {
            "certified_by_checked_artifact"
        } else {
            "operational_only_uncertified"
        },
        "deterministic_evaluator_table": if verified.deterministic_table.is_some() {
            "certified_by_checked_artifact"
        } else {
            "operational_only_uncertified"
        },
        "generic_certificate_semantics": "content_hash_and_domain_context_checked_external_certificate",
    })
}

fn certificate_boundary_kind(value: Option<&VerifiedCertificate>) -> &'static str {
    if value.is_some() {
        "certified_by_external_certificate"
    } else {
        "operational_only_uncertified"
    }
}

fn theorem_claim_status(
    requested: bool,
    missing_prereqs: Vec<&'static str>,
    certified_basis: &[&'static str],
) -> Value {
    if !requested {
        serde_json::json!({
            "requested": false,
            "status": "disabled",
            "missing_prerequisites": [],
            "certified_basis": [],
        })
    } else if missing_prereqs.is_empty() {
        serde_json::json!({
            "requested": true,
            "status": "certified",
            "missing_prerequisites": [],
            "certified_basis": certified_basis,
        })
    } else {
        serde_json::json!({
            "requested": true,
            "status": "uncertified",
            "missing_prerequisites": missing_prereqs,
            "certified_basis": [],
        })
    }
}

pub(super) fn exact_finite_mdp_missing_prereqs(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    search: &SearchSummary,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    match controller {
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            missing.push("planner_family_controller");
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => {
            missing.push("exact_objective_difference_controller");
        }
        crate::spec::CompiledTuneController::McAixiFacCtw(_)
        | crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {}
    }
    if verified.finite_planner_state.is_none() {
        missing.push("verified_finite_planner_state_certificate");
    }
    if verified.no_hidden_state.is_none() {
        missing.push("verified_no_hidden_state_or_inert_state_certificate");
    }
    if verified.exact_reward_encoding.is_none() {
        missing.push("verified_exact_reward_encoding_certificate");
    }
    if !verified.timing_certified(theorem) {
        missing.push("theorem_certified_timing_or_deterministic_table");
    }
    if theorem.scalar_representation_ref.is_none() {
        missing.push("scalar_representation_ref");
    }
    if search.best_eval.objective_bits.is_finite() {
        missing
    } else {
        missing.push("finite_deployable_objective");
        missing
    }
}

fn exact_observed_markov_missing_prereqs(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    search: &SearchSummary,
) -> Vec<&'static str> {
    let mut missing = exact_finite_mdp_missing_prereqs(theorem, verified, controller, search);
    if theorem.exact_state_encoder_spec_ref.is_none() {
        missing.push("exact_state_encoder_spec_ref");
    }
    if verified.exact_state_observation.is_none() {
        missing.push("verified_exact_state_observation_certificate");
    }
    missing
}

fn planner_convergence_missing_prereqs(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    search: &SearchSummary,
) -> Vec<&'static str> {
    let mut missing = exact_finite_mdp_missing_prereqs(theorem, verified, controller, search);
    if !matches!(
        controller,
        crate::spec::CompiledTuneController::McAixiFacCtw(_)
    ) {
        missing.push("mc_aixi_fac_ctw_controller");
    }
    missing
}
