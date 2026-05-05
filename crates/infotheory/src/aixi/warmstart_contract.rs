//! Shared helpers for warm-start exact-J_H planner contract metadata.
//!
//! These helpers are intentionally canonicalized in one place so planner-run
//! task identity and observation-key encoding names do not drift between runtime
//! and CLI validation paths.

use crate::aixi::common::ObservationKeyMode;
use crate::spec::CompiledPlannerRunSpec;
use crc32fast::Hasher;
use serde_json::json;

pub const WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION: u64 = 1;

/// Compute the stable planner-task fingerprint for an exact-J_H planner run.
///
/// The payload includes schema-affecting fields that define dataset applicability.
pub fn warmstart_exact_jh_planner_task_fingerprint(
    compiled: &CompiledPlannerRunSpec,
) -> Result<String, serde_json::Error> {
    let payload = json!({
        "planner_run_canonical_crc32": crc32_hex(compiled.canonical_bytes().as_slice()),
        "controller_kind": compiled.controller().kind_str(),
        "controller_backend": compiled.controller().backend_label(),
        "teacher_contract_schema_version": WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
    });
    serde_json::to_vec(&payload).map(|bytes| crc32_hex(&bytes))
}

/// Convert planner observation keying mode to the canonical contract string.
pub fn observation_key_mode_name(mode: ObservationKeyMode) -> &'static str {
    match mode {
        ObservationKeyMode::First => "first",
        ObservationKeyMode::Last => "last",
        ObservationKeyMode::StreamHash => "stream_hash",
        ObservationKeyMode::FullStream => "full_stream",
    }
}

fn crc32_hex(bytes: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    format!("{:08x}", hasher.finalize())
}
