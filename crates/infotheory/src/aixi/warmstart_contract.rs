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
use crate::spec::{AssetRef, CanonicalJson, CompiledPlannerRunSpec, canonical_json_bytes};
use crc32fast::Hasher;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;

pub const WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION: u64 = 1;

/// SHA-256 task identity binding warm-start artifacts to one planner-run task.
///
/// Wire format is exactly 64 lowercase hexadecimal digits.
/// [`Self::parse_hex`] is strict: only that form is accepted, so malformed
/// fingerprints are unrepresentable.
///
/// Invariant: for every value produced by [`Self::from_payload_bytes`],
/// `parse_hex(&display(v)) == Some(v)`.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TaskFingerprint([u8; 32]);

impl TaskFingerprint {
    /// SHA-256 digest of the canonical task-fingerprint JSON payload bytes.
    pub fn from_payload_bytes(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        let mut output = [0_u8; 32];
        output.copy_from_slice(&digest);
        Self(output)
    }

    /// Parse the canonical 64-digit lowercase hex wire form.
    pub fn parse_hex(value: &str) -> Option<Self> {
        if value.len() != 64 {
            return None;
        }
        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_lower_hex_nibble(pair[0])?;
            let low = decode_lower_hex_nibble(pair[1])?;
            bytes[index] = (high << 4) | low;
        }
        Some(Self(bytes))
    }
}

impl fmt::Display for TaskFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for TaskFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskFingerprint({self})")
    }
}

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
    let bytes = canonical_json_bytes(&standalone_observation_adapter_spec_value(
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
    let bytes = canonical_json_bytes(&standalone_exact_reward_encoding_certificate_value(
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
/// The warm-start teacher asset reference itself is deliberately removed before
/// hashing to avoid making a same-task teacher depend on its own file path or
/// asset selector.
pub fn warmstart_exact_jh_planner_task_fingerprint(
    compiled: &CompiledPlannerRunSpec,
) -> Result<TaskFingerprint, String> {
    let task_asset_content_sha256 = planner_task_asset_content_commitments(compiled)?;
    let planner_run_task_sha256 = warmstart_planner_task_sha256(compiled)?;
    let payload = json!({
        "planner_run_task_sha256": planner_run_task_sha256,
        "controller_kind": compiled.controller().kind_str(),
        "controller_backend": compiled.controller().backend_label(),
        "teacher_contract_schema_version": WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
        "task_asset_content_sha256": task_asset_content_sha256,
    });
    canonical_json_bytes(&payload)
        .map(|bytes| TaskFingerprint::from_payload_bytes(&bytes))
        .map_err(|err| format!("failed to encode warm-start task fingerprint payload: {err}"))
}

fn warmstart_teacher_asset_id(compiled: &CompiledPlannerRunSpec) -> Option<&str> {
    match compiled.controller() {
        crate::spec::CompiledPlannerController::AiqiWarmstartExactJh {
            teacher_dataset_asset,
            ..
        } => Some(teacher_dataset_asset.as_str()),
        _ => None,
    }
}

fn warmstart_planner_task_sha256(compiled: &CompiledPlannerRunSpec) -> Result<String, String> {
    let teacher_asset = warmstart_teacher_asset_id(compiled);
    let mut value = compiled
        .canonical_spec()
        .to_canonical_json_value()
        .map_err(|err| format!("failed to encode planner task JSON: {err}"))?;
    if let Value::Object(root) = &mut value {
        if let Some(Value::Array(assets)) = root.get_mut("assets") {
            assets.retain(|asset| {
                asset
                    .get("id")
                    .and_then(Value::as_str)
                    .is_none_or(|id| Some(id) != teacher_asset)
            });
        }
        if let Some(Value::Object(controller)) = root.get_mut("controller") {
            controller.remove("teacher_dataset_asset");
        }
    }
    canonical_json_bytes(&value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|err| format!("failed to encode warm-start planner task payload: {err}"))
}

fn planner_task_asset_content_commitments(
    compiled: &CompiledPlannerRunSpec,
) -> Result<Vec<Value>, String> {
    let teacher_asset = warmstart_teacher_asset_id(compiled);
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
            "content_sha256": sha256_hex(&bytes),
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

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn decode_lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(all(test, feature = "backend-ctw"))]
mod tests {
    use super::{
        TaskFingerprint, standalone_teacher_provenance_crc32_pair,
        warmstart_exact_jh_planner_task_fingerprint,
    };
    use crate::aixi::common::{ActionAlphabet, ObservationKeyMode};
    use crate::api::{BitStreamSemantics, RateBackend};
    use crate::spec::{
        AssetBinding, BuiltinEnvironmentSpec, CanonicalJson, ControllerSpec, EnvironmentSpec,
        PlannerInterfaceSpec, PlannerRunSpec, PlannerRuntimeSpec, SpecDocument, SpecEnvironment,
        WarmStartExactJhControllerSpec,
    };
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "infotheory-warmstart-contract-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn action_alphabet(n: usize) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
    }

    fn write_asset(dir: &Path, name: &str, bytes: &[u8]) {
        std::fs::write(dir.join(name), bytes).expect("write asset");
    }

    fn sample_exact_jh_spec() -> PlannerRunSpec {
        PlannerRunSpec {
            assets: vec![
                AssetBinding {
                    id: "teacher".to_string(),
                    path: "teacher.json".to_string(),
                },
                AssetBinding {
                    id: "task_input".to_string(),
                    path: "task_input.bin".to_string(),
                },
            ],
            environment: EnvironmentSpec::Builtin {
                builtin: BuiltinEnvironmentSpec::CoinFlip,
            },
            interface: PlannerInterfaceSpec {
                observation_bits: 2,
                observation_stream_len: 1,
                observation_key_mode: ObservationKeyMode::FullStream,
                reward_bits: 2,
                agent_actions: action_alphabet(2),
            },
            controller: ControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: BitStreamSemantics::BinaryTokens,
                return_horizon: 1,
                return_bins: 4,
                label_phase_period: 1,
                teacher_dataset_asset: "teacher".to_string(),
                planner_simulations_per_step: 1,
            }),
            runtime: PlannerRuntimeSpec {
                random_seed: Some(7),
                learn_cycles: Some(1),
                eval_cycles: Some(1),
                terminate_lifetime: 2,
                log_every: 1,
                perf: false,
                vm_perf_only: false,
                explore_epsilon: 0.0,
                explore_gamma: 1.0,
            },
        }
    }

    fn compile_in_dir(spec: PlannerRunSpec, dir: &Path) -> crate::spec::CompiledPlannerRunSpec {
        let value = SpecDocument::PlannerRun(spec)
            .to_canonical_json_value()
            .expect("canonical json");
        let document =
            SpecDocument::parse_json_value(&value, dir).expect("parse warmstart planner run");
        let SpecDocument::PlannerRun(parsed) = document else {
            panic!("expected planner_run document");
        };
        parsed
            .compile_in(&SpecEnvironment::new(dir))
            .expect("compile warmstart planner run")
    }

    fn committed_non_teacher_asset_ids(
        compiled: &crate::spec::CompiledPlannerRunSpec,
    ) -> Vec<String> {
        let teacher_asset = match compiled.controller() {
            crate::spec::CompiledPlannerController::AiqiWarmstartExactJh {
                teacher_dataset_asset,
                ..
            } => Some(teacher_dataset_asset.as_str()),
            _ => None,
        };
        compiled
            .resolved_assets()
            .iter()
            .filter(|binding| Some(binding.id.as_str()) != teacher_asset)
            .map(|binding| binding.id.clone())
            .collect()
    }

    #[test]
    fn exact_jh_warmstart_fingerprint_golden_values_are_stable() {
        let (adapter_crc, reward_certificate) =
            standalone_teacher_provenance_crc32_pair(1, 1, 1).expect("standalone provenance");
        assert_eq!(adapter_crc, "30267f35");
        assert_eq!(reward_certificate, "e09613fc");

        let dir = temp_dir("exact-jh-fingerprint-golden");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        write_asset(
            &dir,
            "teacher.json",
            br#"{"schema_version":1,"contract":{},"traces":[]}"#,
        );
        write_asset(&dir, "task_input.bin", b"task-input-v1");

        let compiled = compile_in_dir(sample_exact_jh_spec(), &dir);
        let fingerprint = warmstart_exact_jh_planner_task_fingerprint(&compiled)
            .expect("task fingerprint")
            .to_string();
        assert_eq!(
            fingerprint,
            "34c801345a415878e31ccfc860ccefe77f22150721ab2989135fcefd3b7f2427"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn exact_jh_task_fingerprint_excludes_teacher_asset_path_and_content() {
        let dir = temp_dir("exact-jh-teacher-exclusion");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        write_asset(
            &dir,
            "teacher.json",
            br#"{"schema_version":1,"contract":{},"traces":[]}"#,
        );
        write_asset(&dir, "task_input.bin", b"task-input-v1");

        let compiled = compile_in_dir(sample_exact_jh_spec(), &dir);
        let baseline =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("baseline fingerprint");
        let committed = committed_non_teacher_asset_ids(&compiled);
        assert!(
            !committed.iter().any(|id| id == "teacher"),
            "teacher asset must be excluded from fingerprint commitments: {committed:?}"
        );
        assert!(
            committed.iter().any(|id| id == "task_input"),
            "non-teacher assets must remain committed: {committed:?}"
        );

        write_asset(
            &dir,
            "teacher.json",
            br#"{"schema_version":1,"contract":{"task_fingerprint":"mutated"},"traces":[]}"#,
        );
        let after_teacher_content = compile_in_dir(sample_exact_jh_spec(), &dir);
        let after_content = warmstart_exact_jh_planner_task_fingerprint(&after_teacher_content)
            .expect("fingerprint after teacher content mutation");
        assert_eq!(
            baseline, after_content,
            "mutating teacher asset bytes must not change task fingerprint"
        );

        let mut moved_teacher_spec = sample_exact_jh_spec();
        moved_teacher_spec.assets[0].path = "teacher_moved.json".to_string();
        write_asset(
            &dir,
            "teacher_moved.json",
            br#"{"schema_version":1,"contract":{},"traces":[]}"#,
        );
        let after_teacher_path = compile_in_dir(moved_teacher_spec, &dir);
        let after_path = warmstart_exact_jh_planner_task_fingerprint(&after_teacher_path)
            .expect("fingerprint after teacher path mutation");
        assert_eq!(
            baseline, after_path,
            "moving teacher asset path must not change task fingerprint"
        );

        write_asset(&dir, "task_input.bin", b"task-input-v2");
        let after_input_mutation = compile_in_dir(sample_exact_jh_spec(), &dir);
        let after_input = warmstart_exact_jh_planner_task_fingerprint(&after_input_mutation)
            .expect("fingerprint after non-teacher asset mutation");
        assert_ne!(
            baseline, after_input,
            "mutating committed non-teacher asset bytes must change task fingerprint"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn task_fingerprint_display_parse_round_trip() {
        let fingerprint = TaskFingerprint::from_payload_bytes(b"warm-start-task-payload");
        let wire = fingerprint.to_string();
        assert_eq!(wire.len(), 64);
        assert!(
            wire.bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        );
        assert_eq!(TaskFingerprint::parse_hex(&wire), Some(fingerprint));
    }

    #[test]
    fn task_fingerprint_parse_hex_rejects_non_canonical_forms() {
        assert!(TaskFingerprint::parse_hex("").is_none());
        let too_short = "0".repeat(63);
        let too_long = "0".repeat(65);
        assert!(TaskFingerprint::parse_hex(&too_short).is_none());
        assert!(TaskFingerprint::parse_hex(&too_long).is_none());
        assert!(
            TaskFingerprint::parse_hex(
                "ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
            )
            .is_none()
        );
        assert!(
            TaskFingerprint::parse_hex(
                "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"
            )
            .is_none()
        );
        assert!(
            TaskFingerprint::parse_hex(
                "000000000000000000000000000000000000000000000000000000000000000g"
            )
            .is_none()
        );
    }
}
