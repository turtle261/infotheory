//! Shared helpers for warm-start exact-J_H planner contract metadata.
//!
//! These helpers are intentionally canonicalized in one place so planner-run
//! task identity and observation-key encoding names do not drift between runtime
//! and CLI validation paths.
//!
//! ## Standalone planner-run provenance (normative)
//!
//! For environments outside the tuner bridge, §3.2 of `docs/infotheory-tuner-v1.tex`
//! still requires observation adapter metadata and exact reward semantics to be
//! **hash-committed**: a versioned adapter declaration plus a CRC32 of a canonical
//! JSON spec object, a declared scalar representation string, and a CRC32 of a
//! small certificate object for the injective reward encoder on the channel.
//!
//! The legacy four identical `"standalone-planner-run"` sentinels are replaced here
//! by structured JSON → CRC32, matching the operational pattern used on the tuner
//! path (`observation_adapter_spec_value` + `observation_adapter_content_hash`).

use crate::aixi::common::{ObservationKeyMode, nonnegative_reward_encoding_bounds};
use crate::spec::{AssetRef, CompiledPlannerRunSpec};
use crc32fast::Hasher;
use serde_json::{Value, json};
use std::fs;

pub const WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION: u64 = 1;

/// Versioned declaration string stored in `observation_adapter_spec_ref` for standalone runs.
pub const WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF: &str =
    "direct-planner-percept-lsb-first-v1";

/// Declared canonical scalar type for instantaneous rewards on the standalone path.
pub const WARMSTART_STANDALONE_SCALAR_REPRESENTATION: &str =
    "nonnegative-integer-i64-instantaneous-reward-v1";

/// Structured observation adapter \(\eta_O\) for direct environment → planner percept.
///
/// Dimensions `(observation_bits, observation_stream_len, reward_bits)` are part of the
/// committed object so teachers cannot be mixed across incompatible interface shapes.
pub fn standalone_observation_adapter_spec_value(
    observation_bits: usize,
    observation_stream_len: usize,
    reward_bits: usize,
) -> Value {
    json!({
        "kind": WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF,
        "schema_version": 1,
        "observation_stream_len": observation_stream_len,
        "observation_bits_per_cell": observation_bits,
        "observation_cell_domain": "u64_values_bounded_by_observation_bits",
        "reward_bits": reward_bits,
        "reward_channel_offset": 0,
        "bit_order_within_field": "lsb_first",
        "reference_encoding": "crate::aixi::common::encode/encode_reward/decode_reward",
    })
}

/// CRC32 (hex, 8 lowercase hex digits) of [`standalone_observation_adapter_spec_value`].
pub fn standalone_observation_adapter_content_crc32(
    observation_bits: usize,
    observation_stream_len: usize,
    reward_bits: usize,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&standalone_observation_adapter_spec_value(
        observation_bits,
        observation_stream_len,
        reward_bits,
    ))?;
    Ok(crc32_hex(&bytes))
}

/// Certificate payload for identity-style \(\Omega_H\) on nonnegative integers in the reward channel.
pub fn standalone_exact_reward_encoding_certificate_value(reward_bits: usize) -> Value {
    let (_min, max_channel, _offset) = nonnegative_reward_encoding_bounds(reward_bits);
    json!({
        "kind": "standalone-identity-reward-encoder-v1",
        "schema_version": 1,
        "reward_bits": reward_bits,
        "min_instantaneous_reward": 0,
        "max_instantaneous_reward_channel": max_channel,
        "omega": "identity_on_representable_nonnegative_integers",
        "injectivity": "identity_is_injective_on_closed_interval_0_max_channel",
    })
}

/// CRC32 (hex) of [`standalone_exact_reward_encoding_certificate_value`].
pub fn standalone_exact_reward_encoding_certificate_hash(
    reward_bits: usize,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&standalone_exact_reward_encoding_certificate_value(
        reward_bits,
    ))?;
    Ok(crc32_hex(&bytes))
}

/// Pair `(observation_adapter_content_crc32, exact_reward_encoding_certificate)` for standalone teachers.
///
/// Callers building [`WarmStartExactJhTeacherContract`](crate::aixi::warmstart::WarmStartExactJhTeacherContract)
/// JSON should use this helper so adapter and reward hashes stay aligned with validation.
pub fn standalone_teacher_provenance_crc32_pair(
    observation_bits: usize,
    observation_stream_len: usize,
    reward_bits: usize,
) -> Result<(String, String), serde_json::Error> {
    Ok((
        standalone_observation_adapter_content_crc32(
            observation_bits,
            observation_stream_len,
            reward_bits,
        )?,
        standalone_exact_reward_encoding_certificate_hash(reward_bits)?,
    ))
}

/// Compute the stable planner-task fingerprint for an exact-J_H planner run.
///
/// The payload includes schema-affecting fields that define dataset applicability.
pub fn warmstart_exact_jh_planner_task_fingerprint(
    compiled: &CompiledPlannerRunSpec,
) -> Result<String, String> {
    let task_asset_content_crc32 = planner_task_asset_content_commitments(compiled)?;
    let payload = json!({
        "planner_run_canonical_crc32": crc32_hex(compiled.canonical_bytes().as_slice()),
        "controller_kind": compiled.controller().kind_str(),
        "controller_backend": compiled.controller().backend_label(),
        "teacher_contract_schema_version": WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
        "task_asset_content_crc32": task_asset_content_crc32,
    });
    serde_json::to_vec(&payload)
        .map(|bytes| crc32_hex(&bytes))
        .map_err(|err| format!("failed to encode warm-start task fingerprint payload: {err}"))
}

fn planner_task_asset_content_commitments(
    compiled: &CompiledPlannerRunSpec,
) -> Result<Vec<Value>, String> {
    let teacher_asset = match compiled.controller() {
        crate::spec::CompiledPlannerController::AiqiWarmstartExactJh {
            teacher_dataset_asset,
            ..
        } => Some(teacher_dataset_asset.as_str()),
        _ => None,
    };
    let mut commitments = Vec::<Value>::new();
    for binding in compiled.resolved_assets() {
        if teacher_asset == Some(binding.id.as_str()) {
            continue;
        }
        let AssetRef::Filesystem(path) = &binding.asset;
        let bytes = fs::read(path).map_err(|err| {
            format!(
                "failed to read task asset '{}' for warm-start task fingerprint: {err}",
                path.display()
            )
        })?;
        commitments.push(json!({
            "id": binding.id.as_str(),
            "content_crc32": crc32_hex(&bytes),
        }));
    }
    commitments.sort_by(|left, right| {
        left["id"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["id"].as_str().unwrap_or_default())
    });
    Ok(commitments)
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
