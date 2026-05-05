# Tuner V1 Traceability Checklist

This checklist maps the normative Tuner V1 implementation obligations in
`docs/infotheory-tuner-v1.tex` to executor code and regression coverage.

## Canonical And Executor Surfaces

- Canonical tune identity is parsed and compiled through `SpecDocument::Tune`
  and remains separate from executor controls in `TuneExecutionConfig`.
  Covered by `canonical_tune_document_rejects_unknown_nested_fields`,
  `tune_exec_config_rejects_unknown_fields_and_malformed_theorem`, and CLI
  precedence tests.
- `baseline_candidate` in canonical tune documents is validated as canonical
  compression-backend JSON with no alias/unknown nested fields. Implemented by
  `ensure_tune_baseline_candidate_is_canonical_json` and covered by
  `canonical_tune_document_rejects_unknown_nested_fields`.
- Executor-side theorem/certificate fields are strict JSON/CLI inputs and do
  not mutate canonical candidate bytes. Implemented in
  `TuneExecutionConfig::from_json_value`,
  `TuneExecutionConfig::apply_theorem_object`, and
  `parse_tune_command_args`.

## Objective And Evaluation Semantics

- The primary objective remains `8L_B(z) + ell_D(z)` for passive and
  interactive evaluator paths. Implemented by `evaluate_candidate_unbounded`
  and deterministic table routing in `VerifiedDeterministicEvaluatorTable`.
- Causal datasets must be canonical structured objects rather than ambiguous
  JSON payloads. Implemented by `parse_causal_header_profile` and covered by
  `structured_causal_dataset_rejects_missing_header_and_charged_history`.
- JSON object `input_asset` payloads fail closed unless they match canonical
  causal dataset kinds (`events` or `examples`/`prefixes`), rather than falling
  back to passive bytes. Implemented by `load_dataset`.
- Causal header typing and event grammar are strict canonical contracts.
  Implemented by `parse_causal_header_profile`, `parse_event_grammar_header`,
  `validate_header_profile_consistency`, and `parse_lowered_event`; covered by
  `causal_dataset_header_and_event_grammar_are_strict` and
  `causal_header_cross_checks_enforce_grammar_and_action_contracts`.
- Typed causal evaluator semantics are implemented through an explicit
  profile-fixed channel/domain/support adapter. Event descriptors are encoded
  into predictor state at zero target cost; byte domains are strict one-byte
  finite symbols with deterministic multi-byte event expansion; enumerated
  domains are scored by finite support normalization. Implemented by
  `CausalEvaluationProfile`, `expand_byte_alphabet_events`,
  `parse_causal_target_domains`, and `causal_target_loss_bits`; covered by
  `byte_alphabet_payloads_expand_to_single_byte_events`,
  `byte_alphabet_expansion_matches_chain_rule_loss`,
  `byte_alphabet_empty_payload_is_rejected`,
  `causal_dataset_domains_are_profile_fixed_and_support_checked`, and
  `causal_event_channel_and_domain_affect_skeleton_identity`.
- Causal-prefix histories replay prior targets only through zero-cost
  observation events; charged history targets are rejected by the same
  regression.
- `planner_deployable_model` is executor-side profile metadata and adds
  deployability diagnostics without changing canonical tune identity.
  Implemented by `ObjectiveTarget::PlannerDeployableModel` and
  `planner_deployability_report`; covered by
  `planner_deployable_model_flag_reports_objective_target_and_diagnostics`.
- Evaluator timing theorem claims are operational-only unless backed by a
  deterministic table or verified certificate basis. Implemented by
  `evaluator_execution_model`, `theorem_timing_basis`, and
  `theorem_claims_report`.
- `evaluator_execution_model: spawn_exec_worker_process_isolated_operational` records an
  operational Unix evaluator process model. It is not the same claim as theorem
  timing tier `isolated`; timing claims remain uncertified unless
  `theorem_timing_basis` names a verified real-time certificate or deterministic
  table. Covered by `real_time_timing_certificate_sets_verified_timing_basis`.
- The online isolated evaluator path is Unix-only. This is the documented
  operational target for process-isolated candidate evaluation; deterministic
  evaluator tables are the portable theorem/evaluation escape hatch on
  non-Unix targets.

## Exact Reward Encoding

- Exact-family planner controllers require a verified reward certificate before
  initialization. Implemented by `PlannerControllerContract::reward_encoder`
  and `TunerRewardEncoder::exact_integer_objective_difference`; covered by
  `exact_family_controller_rejects_missing_exact_reward_certificate`.
- Finite reward codomains are checkable injective maps. Interval
  `integer_objective_difference` remains a validated special case; explicit
  maps are parsed by `parse_finite_reward_map` and must declare
  `complete_nonnegative_interval_max` before exact controller initialization
  may use them. The interval is checked entry-by-entry rather than trusted as a
  bare claim. Covered by finite reward map unit tests,
  `exact_controller_rejects_finite_reward_map_without_complete_interval`, and
  `exact_reward_certificate_rejects_unrepresentable_reachable_reward`.
  Non-identity maps encode objective differences through the certified map,
  covered by
  `exact_finite_reward_map_encodes_objective_difference_not_symbol_arithmetic`.
- Warm-start exact-J_H accepts only interval or identity finite-map reward
  encodings until teacher/live traces carry objective-difference labels or a
  verified decoder-based return encoder. Non-identity finite maps are rejected
  for that controller path. Covered by
  `warmstart_exact_jh_rejects_nonidentity_finite_reward_map`.
- Non-canonical finite-map aliases (for example `reachable_rewards`) are
  rejected; canonical finite maps use only `values`. Covered by
  `finite_reward_map_rejects_reachable_rewards_alias`.
- Exact-objective controller families require finite-map certificates to
  declare `complete_nonnegative_interval_max` during certificate load.
  Implemented by `controller_requires_exact_objective_difference` and
  `load_exact_reward_certificate`.

## Observation Encoding

- Planner observations use the tex raw tuple order: fail flag, normalized
  physical size, normalized target loss, normalized evaluation time,
  physical-size delta, evaluation-time delta, candidate signature, terminal.
  Implemented by `TunerRawObservation::from_runtime_step` and
  `observation_adapter_spec_value`.
- Timeout, invalid, error, inapplicable, self-loop, nondeployable, and success
  diagnostics are stable planner-visible outcomes through field-specific
  sentinels, timeout `tilde_tau = 1`, and candidate/outcome signatures. Covered
  by `planner_percept_encoding_distinguishes_diagnostic_tokens` and raw
  observation sentinel tests.
- Exact-state observation certification is checkable, not boolean-attested.
  Implemented by `load_exact_state_observation_certificate` and
  `validate_exact_state_observation_artifact`; covered by positive theorem
  certification tests, `exact_state_observation_certificate_requires_injectivity_basis`,
  and
  `exact_state_observation_certificate_rejects_duplicate_state_ids`.

## Planner Bridge

- Planner-family tuner controllers use an internal `tuner_bridge` planner-run
  environment instead of a dummy public environment. Implemented by
  `compile_tuner_planner_run_spec`; public planner JSON rejects hand-written
  `tuner_bridge` specs. Covered by
  `planner_run_parser_rejects_internal_tuner_bridge_environment`.
- Public binary planner-run documents reject the same internal bridge marker.
  Covered by `planner_run_binary_rejects_internal_tuner_bridge_environment`.
- Planner action decoding is total over the declared finite mutation alphabet;
  exact theorem runs reject floating-point mutation leaves unless a future
  finite-domain certificate is introduced. Implemented by
  `compile_planner_mutation_actions` and
  `validate_theorem_planner_mutation_domain`.

## Warm-Start Exact J_H

- Warm-start teacher datasets are versioned and same-task fingerprinted,
  including action/observation/reward interface fields, observation key mode,
  observation adapter ref/hash, scalar representation, and exact reward
  certificate hash. Implemented by `WarmStartExactJhTeacherDataset`,
  `WarmStartExactJhTeacherContract`, and
  `validate_warmstart_teacher_contract`.
- Bounded same-task trace refresh rebuilds the warm-start agent from a
  deterministic merge of teacher traces and realized admissible live traces.
  Implemented by `WarmStartExactJhAgent::same_task_live_trace`,
  `merge_warmstart_trace_deterministic`, and
  `TunerPlannerAgentRuntime::rebuild_warmstart_agent`; covered by
  `warmstart_trace_refresh_merges_same_task_live_trace`.

## Executor Controls

- `rss_mode` is routed through explicit peak-memory measurement modes:
  process RSS peak, cgroup-reported peak where available, or a strict maximum
  of both. Implemented by `peak_memory_bytes`,
  `cgroup_v2_peak_memory_bytes`, and `cgroup_v1_peak_memory_bytes`. Reports
  expose the requested mode, effective measurement, and whether cgroup peak
  memory was available. Deterministic-table peak memory nondeployability is
  covered by `deterministic_table_peak_memory_can_make_baseline_nondeployable`.
- `diagnostic_chunk_bytes` is implemented as an executor-side diagnostic
  partition over charged target bytes. It is recorded in executor/evaluator
  profile and provenance, does not affect canonical candidate identity or the
  objective, and is covered by executor control/reporting tests. Empty theorem
  refs are rejected by strict JSON/CLI parsing.
- `max_evaluations` uses baseline-included, warmups-excluded, cache-hit
  included semantics for non-warmup admitted candidate results. Reports expose
  this explicitly via `baseline_counts_toward_max_evaluations`,
  `non_warmup_candidate_results_seen`, and
  `post_baseline_candidate_results_seen`, while fresh evaluator calls remain
  separately reported under cache/provenance counters.
- Unsupported theorem certificate URI schemes are configuration errors rather
  than silent absent certificates. Implemented in `load_certificate_value` and
  covered by `tune_executor_rejects_unsupported_certificate_uri_scheme`.

## Annealed Controller Schedule

- The annealer uses the tex log-linear schedule with elapsed-time progress and
  `T_min = 1e-3`. Implemented by `annealer_progress_from_elapsed` and
  `annealer_temperature`; covered by
  `annealer_schedule_matches_normative_log_linear_law`.

## Candidate-Local Asset Boundary

- Candidate-local `spec_path`, `base_path`, `model_path`, URL-like references,
  `file:` methods, and online-policy `load_from` paths are rejected rather than
  becoming tune-search candidates. Implemented by
  `reject_tune_candidate_local_external_refs` and
  `candidate_contains_external_artifact`; covered by
  `canonical_tune_document_rejects_candidate_local_external_assets`.

## Discounted AIQI Claim Boundary

- `aiqi_discounted` remains a normalized-clipped-improvement heuristic
  controller and cannot certify exact-J_H theorem claims. Covered by
  `discounted_aiqi_exact_theorem_claims_remain_uncertified_by_family`.

## Exact-State Observation Key Modes

- Exact-state observation certificates validate injectivity after applying the
  declared observation key projection, including `stream_hash`. Implemented by
  `project_observation_output`; covered by
  `exact_state_observation_projection_supports_stream_hash`.

## Binary Tune Documents

- The tune executor accepts canonical `.json` and binary `.itsd` tune documents
  through the same `SpecDocument::Tune` surface. Covered by
  `tune_cli_loads_binary_itsd_tune_document`.

## Theorem Claim Reporting

- The theorem report certifies claims only from verified certificate domain
  objects and deterministic table proofs, never from unchecked user booleans.
  Implemented by `VerifiedTheoremInputs`, certificate loaders, and
  `theorem_claims_report`; covered by theorem boundary integration tests.
- The theorem report also records the proof boundary for each certificate kind:
  mechanically checked artifacts, external certificates whose domain/provenance
  are checked but whose semantics are the proof boundary, and
  operational-only uncertified paths.

## Traceability Maintenance

- This checklist is CI-gated by `scripts/check_tuner_traceability.py`, which
  verifies that cited implementation symbols and regression test names still
  exist when the code moves.
