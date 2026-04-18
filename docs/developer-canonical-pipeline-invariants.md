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
