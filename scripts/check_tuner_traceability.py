#!/usr/bin/env python3
"""Check that the Tuner V1 traceability checklist points at live symbols.

The checklist is intentionally human-readable, so this script keeps the gate
lightweight: it verifies the specific code symbols and regression names that
anchor each normative section still exist as definitions in the implementation.
"""

from __future__ import annotations

from pathlib import Path
import sys

try:
    import tree_sitter
    import tree_sitter_rust
except ImportError as _exc:
    sys.exit(
        f"check_tuner_traceability: missing dependency: {_exc}\n"
        "Run with: uv run --no-project --with tree-sitter --with tree-sitter-rust "
        "python scripts/check_tuner_traceability.py"
    )


ROOT = Path(__file__).resolve().parents[1]
TRACEABILITY = ROOT / "docs" / "tuner-v1-traceability.md"
TUNER = ROOT / "crates" / "infotheory" / "src" / "tuner.rs"
TUNER_MODULE_DIR = ROOT / "crates" / "infotheory" / "src" / "tuner"
TUNER_SOURCES = (
    TUNER,
    TUNER_MODULE_DIR / "annealer.rs",
    TUNER_MODULE_DIR / "causal_dataset.rs",
    TUNER_MODULE_DIR / "certificates.rs",
    TUNER_MODULE_DIR / "config.rs",
    TUNER_MODULE_DIR / "eval.rs",
    TUNER_MODULE_DIR / "planner_bridge.rs",
    TUNER_MODULE_DIR / "report.rs",
    TUNER_MODULE_DIR / "tests.rs",
)
WARMSTART = ROOT / "crates" / "infotheory" / "src" / "aixi" / "warmstart.rs"
WARMSTART_CONTRACT = (
    ROOT / "crates" / "infotheory" / "src" / "aixi" / "warmstart_contract.rs"
)
PLANNER_RUNTIME = ROOT / "crates" / "infotheory" / "src" / "aixi" / "planner_runtime.rs"
TUNER_TESTS = ROOT / "crates" / "infotheory" / "tests" / "tuner_integration.rs"
SPEC_TESTS = ROOT / "crates" / "infotheory" / "src" / "spec" / "document" / "tests.rs"
SPEC_PARSER = ROOT / "crates" / "infotheory" / "src" / "spec" / "document" / "parser.rs"


REQUIRED_REFS: tuple[tuple[str, Path], ...] = (
    ("TuneExecutionConfig::from_json_value", TUNER),
    ("TuneExecutionConfig::apply_theorem_object", TUNER),
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
    (
        "CompiledTuneController::exact_objective_difference_controller",
        TUNER,
    ),
    ("TunerRawObservation::from_runtime_step", TUNER),
    ("observation_adapter_spec_value", TUNER),
    ("load_exact_state_observation_certificate", TUNER),
    ("validate_exact_state_observation_artifact", TUNER),
    ("project_observation_output", TUNER),
    ("compile_tuner_planner_run_spec", TUNER),
    ("validate_theorem_planner_mutation_domain", TUNER),
    ("warmstart_exact_jh_planner_task_fingerprint", WARMSTART_CONTRACT),
    ("WarmStartExactJhTeacherDataset", WARMSTART),
    ("WarmStartExactJhTeacherContract", WARMSTART),
    ("warmstart_teacher_trace_from_jsonl_path", WARMSTART),
    ("merge_warmstart_teacher_traces_deterministic", WARMSTART),
    ("validate_warmstart_exact_jh_teacher_contract", PLANNER_RUNTIME),
    ("load_warmstart_exact_jh_teacher_dataset", PLANNER_RUNTIME),
    ("WarmStartExactJhAgent::same_task_live_trace", WARMSTART),
    ("merge_warmstart_trace_deterministic", TUNER),
    ("TunerPlannerAgentRuntime::rebuild_warmstart_agent", TUNER),
    ("resolve_evaluator_runtime_profile", TUNER),
    ("ResolvedEvaluatorRuntimeProfile", TUNER),
    ("ResolvedMemoryAccountingKind", TUNER),
    ("EvaluatorWorkerCgroup", TUNER),
    ("resolve_required_tuner_eval_cgroup_parent", TUNER),
    ("annealer_progress_from_elapsed", TUNER),
    ("annealer_temperature", TUNER),
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
    ("warmstart_exact_jh_json_binary_and_compile_roundtrip", SPEC_TESTS),
    ("jsonl_trace_converter_rejects_malformed_and_inconsistent_records", WARMSTART),
    ("planner_deployable_model_flag_reports_objective_target_and_diagnostics", TUNER_TESTS),
    ("deterministic_table_peak_memory_can_make_baseline_nondeployable", TUNER_TESTS),
    ("real_time_timing_certificate_sets_verified_timing_basis", TUNER_TESTS),
    ("tune_cli_loads_binary_itsd_tune_document", TUNER_TESTS),
    ("planner_run_parser_rejects_internal_tuner_bridge_environment", SPEC_TESTS),
    ("planner_run_binary_rejects_internal_tuner_bridge_environment", SPEC_TESTS),
    ("reject_tune_candidate_local_external_refs", SPEC_PARSER),
    ("ensure_tune_baseline_candidate_is_canonical_json", SPEC_PARSER),
)


class AstIndex:
    def __init__(self) -> None:
        self.functions: set[str] = set()
        self.structs: set[str] = set()
        self.enums: set[str] = set()
        self.struct_fields: set[str] = set()
        self.enum_variants: set[str] = set()
        self.impl_methods: set[str] = set()
        self.test_functions: set[str] = set()

    def merge(self, other: "AstIndex") -> None:
        self.functions.update(other.functions)
        self.structs.update(other.structs)
        self.enums.update(other.enums)
        self.struct_fields.update(other.struct_fields)
        self.enum_variants.update(other.enum_variants)
        self.impl_methods.update(other.impl_methods)
        self.test_functions.update(other.test_functions)


def rust_language():
    language = tree_sitter_rust.language()
    if isinstance(language, tree_sitter.Language):
        return language
    return tree_sitter.Language(language)


def rust_parser():
    parser = tree_sitter.Parser()
    language = rust_language()
    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)
    return parser


def node_text(node) -> str:
    return node.text.decode("utf-8")


def normalized_attribute_text(node) -> str:
    return "".join(node_text(node).split())


def is_test_attribute(node) -> bool:
    return node.type == "attribute_item" and normalized_attribute_text(node) == "#[test]"


def function_has_test_attribute(node) -> bool:
    prev = node.prev_named_sibling
    while prev:
        if is_test_attribute(prev):
            return True
        if prev.type not in ("attribute_item", "line_comment", "block_comment"):
            return False
        prev = prev.prev_named_sibling
    return False


def impl_type_name(node) -> str | None:
    if node.child_by_field_name("trait") is not None:
        return None

    type_node = node.child_by_field_name("type")
    if type_node is None:
        return None

    type_text = "".join(node_text(type_node).split())
    if "<" in type_text:
        type_text = type_text.split("<", maxsplit=1)[0]
    if "::" in type_text:
        type_text = type_text.rsplit("::", maxsplit=1)[1]
    return type_text or None


def add_struct_fields(idx: AstIndex, node) -> None:
    body = node.child_by_field_name("body")
    if body is None or body.type != "field_declaration_list":
        return

    for child in body.named_children:
        if child.type != "field_declaration":
            continue
        field_name_node = child.child_by_field_name("name")
        if field_name_node is not None:
            idx.struct_fields.add(node_text(field_name_node))


def add_enum_variants(idx: AstIndex, node, enum_name: str) -> None:
    body = node.child_by_field_name("body")
    if body is None or body.type != "enum_variant_list":
        return

    for child in body.named_children:
        if child.type != "enum_variant":
            continue
        variant_name_node = child.child_by_field_name("name")
        if variant_name_node is not None:
            idx.enum_variants.add(f"{enum_name}::{node_text(variant_name_node)}")


def add_impl_methods(idx: AstIndex, node) -> None:
    type_name = impl_type_name(node)
    if type_name is None:
        return

    body = node.child_by_field_name("body")
    if body is None or body.type != "declaration_list":
        return

    for child in body.named_children:
        if child.type != "function_item":
            continue
        method_name_node = child.child_by_field_name("name")
        if method_name_node is not None:
            idx.impl_methods.add(f"{type_name}::{node_text(method_name_node)}")


def build_ast_index(source_code: bytes, source_name: str = "<memory>") -> AstIndex:
    parser = rust_parser()
    tree = parser.parse(source_code)
    if tree.root_node.has_error:
        raise ValueError(f"tree-sitter-rust parse error in {source_name}")
    idx = AstIndex()

    def visit_item(node) -> None:
        if node.type == "function_item":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = node_text(name_node)
                idx.functions.add(name)
                if function_has_test_attribute(node):
                    idx.test_functions.add(name)
        elif node.type == "struct_item":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                idx.structs.add(node_text(name_node))
            add_struct_fields(idx, node)
        elif node.type == "enum_item":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = node_text(name_node)
                idx.enums.add(name)
                add_enum_variants(idx, node, name)
        elif node.type == "impl_item":
            add_impl_methods(idx, node)
        elif node.type == "mod_item":
            body = node.child_by_field_name("body")
            if body is not None:
                visit_item_scope(body)

    def visit_item_scope(node) -> None:
        for child in node.named_children:
            visit_item(child)

    visit_item_scope(tree.root_node)
    return idx


def ref_exists_as_definition(ref: str, path: Path, idx: AstIndex) -> bool:
    if path in (TUNER_TESTS, SPEC_TESTS):
        return ref in idx.test_functions
    if "::" not in ref:
        return (
            ref in idx.functions
            or ref in idx.structs
            or ref in idx.enums
            or ref in idx.struct_fields
        )
    return ref in idx.enum_variants or ref in idx.impl_methods


def require_self_test(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"traceability self-test failed: {message}")


def run_self_tests() -> None:
    source = b"""
    #[test]
    fn real_test_anchor() {}

    struct Settings {
        peak_memory_bytes: u64,
    }

    fn peak_memory_bytes() {}

    enum ObjectiveTarget {
        PlannerDeployableModel,
    }

    impl Settings {
        fn from_json_value() {}
    }

    trait Decoder {
        fn trait_only() {}
    }

    impl Decoder for Settings {
        fn trait_only() {}
    }

    fn use_settings() {
        let x = Settings { peak_memory_bytes: 0 };
    }
    """
    idx = build_ast_index(source)
    require_self_test("real_test_anchor" in idx.test_functions, "failed to find #[test] function")
    require_self_test("peak_memory_bytes" in idx.struct_fields, "failed to find struct field")
    require_self_test("peak_memory_bytes" in idx.functions, "failed to find free function")
    require_self_test(
        "ObjectiveTarget::PlannerDeployableModel" in idx.enum_variants,
        "failed to find enum variant",
    )
    require_self_test(
        "Settings::from_json_value" in idx.impl_methods,
        "failed to find inherent impl method",
    )
    require_self_test(
        "Settings::trait_only" not in idx.impl_methods,
        "trait impl method was accepted as inherent method",
    )
    require_self_test(
        "from_json_value" not in idx.functions,
        "inherent impl method was accepted as free function",
    )

    source_no_def = b"""
    fn use_settings() {
        let x = Settings { peak_memory_bytes: 0 };
    }
    """
    idx_no_def = build_ast_index(source_no_def)
    require_self_test(
        "peak_memory_bytes" not in idx_no_def.struct_fields,
        "struct literal field initializer was accepted as a field definition",
    )
    require_self_test(
        not ref_exists_as_definition("peak_memory_bytes", TUNER, idx_no_def),
        "struct literal field initializer satisfied plain anchor lookup",
    )


def main() -> int:
    run_self_tests()

    doc = TRACEABILITY.read_text(encoding="utf-8")

    # Pre-parse indices
    indices: dict[Path, AstIndex] = {}
    for path in TUNER_SOURCES:
        indices[path] = build_ast_index(path.read_bytes(), str(path.relative_to(ROOT)))
    indices[WARMSTART] = build_ast_index(WARMSTART.read_bytes(), str(WARMSTART.relative_to(ROOT)))
    indices[WARMSTART_CONTRACT] = build_ast_index(
        WARMSTART_CONTRACT.read_bytes(), str(WARMSTART_CONTRACT.relative_to(ROOT))
    )
    indices[PLANNER_RUNTIME] = build_ast_index(
        PLANNER_RUNTIME.read_bytes(), str(PLANNER_RUNTIME.relative_to(ROOT))
    )
    indices[TUNER_TESTS] = build_ast_index(TUNER_TESTS.read_bytes(), str(TUNER_TESTS.relative_to(ROOT)))
    indices[SPEC_TESTS] = build_ast_index(SPEC_TESTS.read_bytes(), str(SPEC_TESTS.relative_to(ROOT)))
    indices[SPEC_PARSER] = build_ast_index(SPEC_PARSER.read_bytes(), str(SPEC_PARSER.relative_to(ROOT)))

    tuner_aggregate = AstIndex()
    for path in TUNER_SOURCES:
        tuner_aggregate.merge(indices[path])

    missing: list[str] = []
    for ref, path in REQUIRED_REFS:
        if ref not in doc:
            missing.append(f"{TRACEABILITY.relative_to(ROOT)} does not cite `{ref}`")

        idx = tuner_aggregate if path == TUNER else indices[path]
        if path == TUNER:
            target = "tuner implementation sources"
        else:
            target = str(path.relative_to(ROOT))
        if not ref_exists_as_definition(ref, path, idx):
            missing.append(f"{target} do not define `{ref}`")

    if missing:
        print("Tuner traceability check failed:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 1

    print(f"Checked {len(REQUIRED_REFS)} Tuner V1 traceability anchors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
