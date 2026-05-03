#!/usr/bin/env python3
"""Check that the Tuner V1 traceability checklist points at live symbols.

The checklist is intentionally human-readable, so this script keeps the gate
lightweight: it verifies the specific code symbols and regression names that
anchor each normative section still exist as definitions in the implementation.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TRACEABILITY = ROOT / "docs" / "tuner-v1-traceability.md"
TUNER = ROOT / "crates" / "infotheory" / "src" / "tuner.rs"
WARMSTART = ROOT / "crates" / "infotheory" / "src" / "aixi" / "warmstart.rs"
TUNER_TESTS = ROOT / "crates" / "infotheory" / "tests" / "tuner_integration.rs"
SPEC_TESTS = ROOT / "crates" / "infotheory" / "src" / "spec" / "document" / "tests.rs"
SPEC_PARSER = ROOT / "crates" / "infotheory" / "src" / "spec" / "document" / "parser.rs"


REQUIRED_REFS: tuple[tuple[str, Path], ...] = (
    ("TuneExecutionConfig::from_json_value", TUNER),
    ("apply_theorem_object", TUNER),
    ("parse_tune_command_args", TUNER),
    ("VerifiedDeterministicEvaluatorTable", TUNER),
    ("parse_causal_header_profile", TUNER),
    ("validate_header_profile_consistency", TUNER),
    ("load_dataset", TUNER),
    ("parse_lowered_event", TUNER),
    ("structured_causal_dataset_rejects_missing_header_and_charged_history", TUNER),
    ("causal_dataset_header_and_event_grammar_are_strict", TUNER),
    ("ObjectiveTarget::PlannerDeployableModel", TUNER),
    ("planner_deployability_report", TUNER),
    ("theorem_timing_basis", TUNER),
    ("theorem_claims_report", TUNER),
    ("PlannerControllerContract::reward_encoder", TUNER),
    ("TunerRewardEncoder::exact_integer_objective_difference", TUNER),
    ("parse_finite_reward_map", TUNER),
    ("finite_reward_map_rejects_reachable_rewards_alias", TUNER),
    ("complete_nonnegative_interval_max", TUNER),
    ("controller_requires_exact_objective_difference", TUNER),
    ("TunerRawObservation::from_runtime_step", TUNER),
    ("observation_adapter_spec_value", TUNER),
    ("load_exact_state_observation_certificate", TUNER),
    ("validate_exact_state_observation_artifact", TUNER),
    ("project_observation_output", TUNER),
    ("compile_tuner_planner_run_spec", TUNER),
    ("validate_theorem_planner_mutation_domain", TUNER),
    ("WarmStartExactJhTeacherDataset", WARMSTART),
    ("WarmStartExactJhTeacherContract", WARMSTART),
    ("validate_warmstart_teacher_contract", TUNER),
    ("WarmStartExactJhAgent::same_task_live_trace", WARMSTART),
    ("merge_warmstart_trace_deterministic", TUNER),
    ("TunerPlannerAgentRuntime::rebuild_warmstart_agent", TUNER),
    ("peak_memory_bytes", TUNER),
    ("annealer_progress_from_elapsed", TUNER),
    ("annealer_temperature", TUNER),
    ("cgroup_v2_peak_memory_bytes", TUNER),
    ("cgroup_v1_peak_memory_bytes", TUNER),
    ("VerifiedTheoremInputs", TUNER),
    ("canonical_tune_document_rejects_unknown_nested_fields", TUNER_TESTS),
    ("canonical_tune_document_rejects_candidate_local_external_assets", TUNER_TESTS),
    ("tune_exec_config_rejects_unknown_fields_and_malformed_theorem", TUNER_TESTS),
    ("tune_executor_rejects_unsupported_certificate_uri_scheme", TUNER_TESTS),
    ("exact_family_controller_rejects_missing_exact_reward_certificate", TUNER_TESTS),
    ("exact_controller_rejects_finite_reward_map_without_complete_interval", TUNER_TESTS),
    ("exact_reward_certificate_rejects_unrepresentable_reachable_reward", TUNER_TESTS),
    ("exact_finite_reward_map_encodes_objective_difference_not_symbol_arithmetic", TUNER),
    ("discounted_aiqi_exact_theorem_claims_remain_uncertified_by_family", TUNER),
    ("annealer_schedule_matches_normative_log_linear_law", TUNER),
    ("warmstart_exact_jh_rejects_nonidentity_finite_reward_map", TUNER_TESTS),
    ("planner_percept_encoding_distinguishes_diagnostic_tokens", TUNER),
    ("exact_state_observation_certificate_requires_injectivity_basis", TUNER_TESTS),
    ("exact_state_observation_projection_supports_stream_hash", TUNER),
    ("exact_state_observation_certificate_rejects_duplicate_state_ids", TUNER_TESTS),
    ("warmstart_trace_refresh_merges_same_task_live_trace", TUNER_TESTS),
    ("planner_deployable_model_flag_reports_objective_target_and_diagnostics", TUNER_TESTS),
    ("deterministic_table_peak_memory_can_make_baseline_nondeployable", TUNER_TESTS),
    ("real_time_timing_certificate_sets_verified_timing_basis", TUNER_TESTS),
    ("tune_cli_loads_binary_itsd_tune_document", TUNER_TESTS),
    ("planner_run_parser_rejects_internal_tuner_bridge_environment", SPEC_TESTS),
    ("planner_run_binary_rejects_internal_tuner_bridge_environment", SPEC_TESTS),
    ("reject_tune_candidate_local_external_refs", SPEC_PARSER),
    ("ensure_tune_baseline_candidate_is_canonical_json", SPEC_PARSER),
)


def normalize_ref(ref: str) -> str:
    return ref.split("::")[-1]


def regex_search(pattern: str, haystack: str) -> bool:
    return re.search(pattern, haystack, flags=re.MULTILINE | re.DOTALL) is not None


def has_test_function(name: str, haystack: str) -> bool:
    pattern = rf"#\s*\[\s*test\s*\][\s\S]{{0,512}}\bfn\s+{re.escape(name)}\s*\("
    return regex_search(pattern, haystack)


def has_free_item_or_field(name: str, haystack: str) -> bool:
    escaped = re.escape(name)
    patterns = (
        rf"\bfn\s+{escaped}\s*\(",
        rf"\bstruct\s+{escaped}\b",
        rf"\benum\s+{escaped}\b",
        rf"\b{escaped}\s*:",
    )
    return any(regex_search(pattern, haystack) for pattern in patterns)


def has_impl_method(type_name: str, method_name: str, haystack: str) -> bool:
    pattern = (
        rf"\bimpl(?:\s*<[^>]*>)?\s+{re.escape(type_name)}\b[^\{{]*\{{"
        rf"[\s\S]*?\bfn\s+{re.escape(method_name)}\s*\("
    )
    return regex_search(pattern, haystack)


def has_enum_variant(enum_name: str, variant_name: str, haystack: str) -> bool:
    pattern = rf"\benum\s+{re.escape(enum_name)}\b[^\{{]*\{{[\s\S]*?\b{re.escape(variant_name)}\b"
    return regex_search(pattern, haystack)


def ref_exists_as_definition(ref: str, path: Path, haystack: str) -> bool:
    leaf = normalize_ref(ref)
    if path in (TUNER_TESTS, SPEC_TESTS):
        return has_test_function(leaf, haystack)
    if "::" not in ref:
        return has_free_item_or_field(ref, haystack)
    type_name, member_name = ref.split("::", maxsplit=1)
    if member_name[:1].isupper():
        return has_enum_variant(type_name, member_name, haystack)
    return has_impl_method(type_name, member_name, haystack)


def main() -> int:
    doc = TRACEABILITY.read_text(encoding="utf-8")
    missing: list[str] = []
    for ref, path in REQUIRED_REFS:
        if ref not in doc:
            missing.append(f"{TRACEABILITY.relative_to(ROOT)} does not cite `{ref}`")
        haystack = path.read_text(encoding="utf-8")
        if not ref_exists_as_definition(ref, path, haystack):
            missing.append(f"{path.relative_to(ROOT)} does not define `{ref}`")
    if missing:
        print("Tuner traceability check failed:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"Checked {len(REQUIRED_REFS)} Tuner V1 traceability anchors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
