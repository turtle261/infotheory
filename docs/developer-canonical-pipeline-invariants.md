# Canonical Pipeline Invariants

This document defines the hard semantic boundary for spec/runtime flow in Infotheory.

## Normative Pipeline

The only valid semantic flow is:

1. Load/parse (JSON/binary wrapper form)
2. Validate/canonicalize
3. Lower/compile into runtime-ready plans
4. Execute via runtime builders

Executable runtime objects must be derived from canonical lowered plans, not from ad-hoc
raw strings or partially interpreted intermediate representations.

## Scope Boundaries For Refactor Series

In-scope:

- `spec/*` canonicalization, parsing, document, and plan-lowering boundaries
- `runtime/*` dispatch and runtime construction boundaries
- Rust API / CLI / Python route consistency
- converter alias parity with runtime registry authority

Out-of-scope:

- backend algorithmic mathematics
- vendor crate internals (`vendor/*`) except required feature wiring/compatibility

## Bypass Inventory (Tracked Targets)

The following internal areas are tracked as bypass-sensitive and must not be expanded:

- `crates/infotheory/src/spec/core.rs`
  - `compiled_rate_backend_from_plan` and `compiled_compression_backend_from_plan`
  - internal unchecked helpers must remain private and never be exposed outside `spec::core`
- `crates/infotheory/src/runtime/mod.rs`
  - runtime constructors for calibrated/rate wrappers must consume checked compiled plans
- `crates/infotheory/src/aixi/agent.rs` + `crates/infotheory/src/aixi/aiqi.rs`
  - planner spec construction must route through shared builder utilities and canonical spec
    compilation path

## Error Semantics

Document loading must remain format-directed and truthful:

- `.json` -> JSON parse only
- `.itsd` or magic envelope -> binary parse
- no JSON->binary blind fallback that obscures root parse errors

## Planner-Run Ownership Boundary

Planner-run execution has a single, layered ownership split. Each concern lives in exactly
one place and downstream layers must not re-implement upstream responsibilities.

- `crates/infotheory/src/aixi/planner_agent.rs` owns *execution*: controller construction
  (`PlannerControllerAgent`), the cycle/phase state machine (`PlannerRunSession`,
  `PlannerSchedule`, `PlannerPhase`), the observer trait (`PlannerCycleObserver`,
  `PlannerActionProvenance`), and per-cycle outcomes. Nothing here reads the filesystem or
  interprets CLI overlays.
- `crates/infotheory/src/aixi/planner_runtime.rs` owns *asset and environment binding*:
  resolving `planner_run` documents to compiled plans (`compile_planner_run_document`),
  building environments (`build_planner_environment`), and loading/validating warm-start
  teacher datasets against a compiled plan. This is the only layer that materializes a
  teacher dataset from a resolved asset.
- `crates/infotheory/src/cli/planner_run.rs` owns the *CLI adapter*: it reads the config
  path, rejects legacy non-canonical JSON and legacy interface reward-range fields, drives
  `run_compiled_planner_run`/`run_aixi_mode`, and renders trace/telemetry output
  (`AixiRunLogger`, `PlannerCliObserver`). `crates/infotheory/src/main.rs` only dispatches
  the `aixi` subcommand into this adapter; it must not host planner-run execution helpers,
  the run logger, or canonical-document shape checks.

The CLI layer depends on the planner-agent and planner-runtime facades; the dependency
never flows the other way. New planner-run surfaces (including warm-start variants) must
slot into this same three-layer split rather than re-deriving runtime behavior from raw
strings or partially interpreted JSON.

## Warm-Start Exact-J_H Invariants

### Task Fingerprint Binding

The warm-start task fingerprint
(`warmstart_exact_jh_planner_task_fingerprint` in
`crates/infotheory/src/aixi/warmstart_contract.rs`) is computed by *stripping the teacher
itself out of the task identity* before hashing. Concretely, the fingerprint payload
commits to:

- `planner_run_task_sha256`: a SHA-256 digest over the canonical planner-run JSON with the
  teacher removed from both `assets` and the controller's `teacher_dataset_asset` field;
- the controller kind and backend label;
- the teacher-contract schema version;
- `task_asset_content_sha256`: SHA-256 content commitments for every resolved asset *except* the
  teacher dataset asset.

Invariant: the fingerprint must be invariant to the teacher dataset asset's path and byte
content, and must bind to the content of every non-teacher (`task_input`) asset. This is
what lets a teacher dataset legitimately commit to the fingerprint of the task it teaches
without creating a self-reference. Any change that folds teacher path/content back into the
fingerprint, or that drops a non-teacher asset's content from it, is a regression and is
guarded by the fingerprint golden/asset-mutation tests in
`crates/infotheory/src/aixi/warmstart_contract.rs`.

### Validation Boundary

The teacher contract and the full teacher dataset are validated at distinct points:

- `validate_warmstart_exact_jh_teacher_contract` (in `planner_runtime.rs`) is
  *contract-only*: it checks the dataset's declared contract (interface, adapter, scalar,
  task fingerprint) against a compiled planner run via
  `validate_warmstart_teacher_against_compiled_planner_run`. It does not validate trace
  records.
- `load_warmstart_exact_jh_teacher_dataset` runs *full dataset validation*
  (`validate_warmstart_teacher_dataset_for_compiled_planner_run`) after parsing the asset
  bytes, i.e. contract plus every trace/transition, against the compiled plan.
- Tuner ingestion and the trace-refresh merge path likewise run full dataset validation
  after the relevant planner run is compiled; the trace-refresh merge counter
  (`warmstart_trace_refresh_merges`) increments only for structurally inserted traces. The
  self-improvement report marks `same_task_trace_refresh_rebuild` when same-task trace
  refresh is the active self-improvement mode; the actual warm-start agent rebuild remains
  insertion-aware and only happens when a merge changed the dataset.

Invariant: full dataset validation always happens against a *compiled* planner run, never
against raw JSON, and the contract-only check must never be silently substituted for full
dataset validation on a path that ingests teacher traces.
