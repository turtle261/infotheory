#![cfg(all(feature = "tuner", feature = "backend-ctw"))]

use crc32fast::Hasher;
use infotheory::spec::{SpecDocument, SpecEnvironment};
use infotheory::tuner::{
    TimingCertificationTier, parse_tune_command_args, run_tune, run_tuner_eval_worker_from_env,
};
use serde_json::{Value, json};
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
#[cfg(feature = "cli")]
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct ControllerCase {
    kind: &'static str,
    status: &'static str,
    runtime_path: &'static str,
    agent_runtime: &'static str,
    planner_run_controller_kind: &'static str,
    reward_semantics: &'static str,
    needs_teacher: bool,
    controller: Value,
}

fn temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("infotheory_tuner_integration_{label}_{nanos}"));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn crc32_hex(bytes: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    format!("{:08x}", hasher.finalize())
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).expect("serialize json")).expect("write json");
}

fn write_tune_itsd(path: &Path, value: &Value, base_dir: &Path) {
    let document = SpecDocument::parse_json_value(value, base_dir).expect("parse tune json");
    fs::write(path, document.to_binary()).expect("write tune itsd");
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("read json")).expect("parse json")
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> &'a str {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{pointer} must be a string in {value}"))
}

fn u64_at(value: &Value, pointer: &str) -> u64 {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{pointer} must be an unsigned integer in {value}"))
}

fn f64_at(value: &Value, pointer: &str) -> f64 {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{pointer} must be a finite number in {value}"))
}

fn bool_at(value: &Value, pointer: &str) -> bool {
    value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("{pointer} must be a boolean in {value}"))
}

#[cfg(unix)]
#[test]
#[ignore = "libtest entrypoint for spawned tuner evaluator workers"]
fn __infotheory_tuner_eval_worker() {
    if std::env::var_os("INFOTHEORY_TUNER_EVAL_REQUEST_PATH").is_none()
        || std::env::var_os("INFOTHEORY_TUNER_EVAL_RESPONSE_PATH").is_none()
    {
        return;
    }
    run_tuner_eval_worker_from_env().expect("run tuner evaluator worker from env");
}

fn interface() -> Value {
    json!({
        "observation_bits": 8,
        "observation_stream_len": 1,
        "observation_key_mode": "full_stream",
        "reward_bits": 16,
        "agent_actions": 2,
    })
}

fn baseline_candidate() -> Value {
    json!({
        "kind": "rate-ac",
        "rate_backend": {
            "kind": "ctw",
            "depth": 8,
        },
        "framing": "framed",
    })
}

fn bounds() -> Value {
    json!({
        "allowed_backends": ["ctw"],
        "forbidden_backends": [],
        "parameter_ranges": [{
            "parameter": "rate_backend.depth",
            "min": 1.0,
            "max": 16.0,
        }],
        "max_experts": 2,
        "max_mixture_nesting_depth": 1,
        "min_experts": 1,
        "allow_duplicate_experts": false,
        "required_experts": [],
        "forbidden_expert_pairs": [],
    })
}

fn controller_cases() -> Vec<ControllerCase> {
    vec![
        ControllerCase {
            kind: "mc_aixi_fac_ctw",
            status: "completed_mc_aixi_fac_ctw",
            runtime_path: "finite_mutation_agent_bridge_mcaixi_fac_ctw",
            agent_runtime: "aixi::agent::Agent",
            planner_run_controller_kind: "mc_aixi",
            reward_semantics: "exact_objective_difference",
            needs_teacher: false,
            controller: json!({
                "kind": "mc_aixi_fac_ctw",
                "interface": interface(),
                "planner_simulations_per_step": 2,
            }),
        },
        ControllerCase {
            kind: "aiqi_discounted",
            status: "completed_aiqi_discounted",
            runtime_path: "finite_mutation_agent_bridge_aiqi_discounted",
            agent_runtime: "aixi::aiqi::AiqiAgent",
            planner_run_controller_kind: "aiqi_discounted",
            reward_semantics: "normalized_clipped_improvement",
            needs_teacher: false,
            controller: json!({
                "kind": "aiqi_discounted",
                "interface": interface(),
                "planner_simulations_per_step": 2,
                "return_horizon": 1,
                "return_bins": 2,
                "discount_factor": 0.5,
                "min_improvement": 0.0,
                "max_improvement": 1.0,
            }),
        },
        ControllerCase {
            kind: "aiqi_warmstart_exact_jh",
            status: "completed_aiqi_warmstart_exact_jh",
            runtime_path: "finite_mutation_agent_bridge_aiqi_warmstart_exact_jh",
            agent_runtime: "aixi::warmstart::WarmStartExactJhAgent",
            planner_run_controller_kind: "aiqi_warmstart_exact_jh",
            reward_semantics: "exact_objective_difference",
            needs_teacher: true,
            controller: json!({
                "kind": "aiqi_warmstart_exact_jh",
                "interface": interface(),
                "planner_simulations_per_step": 2,
                "return_horizon": 1,
                "warmstart_teacher_dataset_asset": "teacher",
                "label_phase_period": 1,
            }),
        },
    ]
}

fn tune_spec(
    dataset_path: &Path,
    output_path: &Path,
    report_path: &Path,
    controller: Value,
    teacher_path: Option<&Path>,
) -> Value {
    let mut assets = vec![json!({
        "id": "dataset",
        "path": path_string(dataset_path),
    })];
    if let Some(path) = teacher_path {
        assets.push(json!({
            "id": "teacher",
            "path": path_string(path),
        }));
    }
    json!({
        "schema_version": 1,
        "kind": "tune",
        "assets": assets,
        "input_asset": "dataset",
        "baseline_candidate": baseline_candidate(),
        "controller": controller,
        "bounds": bounds(),
        "eval_time_limit_seconds": 1.0,
        "time_budget_seconds": 5.0,
        "min_throughput_bytes_per_second": 1.0,
        "max_memory_bytes": 1099511627776u64,
        "output_config_path": path_string(output_path),
        "seed": 7,
        "report_path": path_string(report_path),
    })
}

fn write_passive_dataset(path: &Path) {
    fs::write(path, b"planner family passive integration dataset").expect("write passive dataset");
}

fn canonical_causal_dataset(codec_hash: &str, payload_key: &str, payload: Value) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("schema_version".to_string(), json!(1));
    object.insert("environment_id".to_string(), json!("test-env"));
    object.insert("environment_config_crc32".to_string(), json!("00000000"));
    object.insert("codec_hash".to_string(), json!(codec_hash));
    object.insert(
        "reset_convention".to_string(),
        json!("reset-before-episode"),
    );
    object.insert("action_alphabet".to_string(), json!({"size": 2}));
    object.insert(
        "percept_schema".to_string(),
        json!({
            "encoding": "bytes",
            "channels": [{"channel": "percept", "domain": "bytes"}]
        }),
    );
    object.insert(
        "reward_encoding".to_string(),
        json!({"encoding": "bytes", "channel": "reward", "domain": "binary"}),
    );
    object.insert(
        "terminal_encoding".to_string(),
        json!({"encoding": "bytes", "channel": "terminal", "domain": "binary"}),
    );
    object.insert("collection_policy".to_string(), json!("test-policy"));
    object.insert(
        "target_domains".to_string(),
        json!({
            "bytes": {"kind": "byte_alphabet"},
            "binary": {"kind": "enumerated_payloads", "payloads": [[0], [1]]}
        }),
    );
    object.insert(
        "event_grammar".to_string(),
        json!({
            "context_channels": ["action"],
            "observe_target_no_score": [
                {"channel": "percept", "domain": "bytes"},
                {"channel": "percept", "domain": "binary"},
                {"channel": "reward", "domain": "binary"},
                {"channel": "terminal", "domain": "binary"}
            ],
            "target": [
                {"channel": "percept", "domain": "bytes"},
                {"channel": "percept", "domain": "binary"},
                {"channel": "reward", "domain": "binary"},
                {"channel": "terminal", "domain": "binary"}
            ]
        }),
    );
    object.insert(payload_key.to_string(), payload);
    Value::Object(object)
}

fn write_teacher(path: &Path, task_fingerprint: &str, reward_cert_crc32: &str) {
    write_json(
        path,
        &json!({
            "schema_version": 1,
            "contract": {
                "task_fingerprint": task_fingerprint,
                "action_alphabet_size": 2,
                "observation_bits": 8,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "observation_adapter_spec_ref": "single-channel-conditional-byte-adapter-v1",
                "observation_adapter_content_crc32": observation_adapter_crc32(),
                "reward_bits": 16,
                "return_horizon": 1,
                "label_phase_period": 1,
                "scalar_representation": "scalar://finite-f64",
                "exact_reward_encoding_certificate": reward_cert_crc32
            },
            "traces": [{
                "transitions": [
                    {"action": 0, "observations": [0], "reward": 0},
                    {"action": 1, "observations": [1], "reward": 1}
                ]
            }]
        }),
    );
}

fn write_placeholder_teacher(path: &Path) {
    write_teacher(path, "placeholder", "placeholder");
}

fn timing_label(timing: TimingCertificationTier) -> &'static str {
    match timing {
        TimingCertificationTier::BestEffort => "best_effort",
        TimingCertificationTier::Isolated => "isolated",
        TimingCertificationTier::RealTime => "real_time",
        TimingCertificationTier::DeterministicTable => "deterministic_table",
        _ => unreachable!("test covers known timing variants"),
    }
}

fn timing_certifies(timing: TimingCertificationTier) -> bool {
    matches!(
        timing,
        TimingCertificationTier::RealTime | TimingCertificationTier::DeterministicTable
    )
}

fn feature_set() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "default-backends") {
        features.push("default-backends");
    }
    if cfg!(feature = "capability-default") {
        features.push("capability-default");
    }
    if cfg!(feature = "capability-statistical") {
        features.push("capability-statistical");
    }
    if cfg!(feature = "capability-neural") {
        features.push("capability-neural");
    }
    if cfg!(feature = "capability-archive") {
        features.push("capability-archive");
    }
    if cfg!(feature = "capability-vm") {
        features.push("capability-vm");
    }
    if cfg!(feature = "aixi") {
        features.push("aixi");
    }
    if cfg!(feature = "tuner") {
        features.push("tuner");
    }
    if cfg!(feature = "aixi-gameengine") {
        features.push("aixi-gameengine");
    }
    if cfg!(feature = "aixi-gameengine-physics") {
        features.push("aixi-gameengine-physics");
    }
    if cfg!(feature = "aixi-vm") {
        features.push("aixi-vm");
    }
    if cfg!(feature = "all-backends") {
        features.push("all-backends");
    }
    if cfg!(feature = "backend-rosa") {
        features.push("backend-rosa");
    }
    if cfg!(feature = "backend-ctw") {
        features.push("backend-ctw");
    }
    if cfg!(feature = "backend-match") {
        features.push("backend-match");
    }
    if cfg!(feature = "backend-ppmd") {
        features.push("backend-ppmd");
    }
    if cfg!(feature = "backend-sequitur") {
        features.push("backend-sequitur");
    }
    if cfg!(feature = "backend-mixture") {
        features.push("backend-mixture");
    }
    if cfg!(feature = "backend-particle") {
        features.push("backend-particle");
    }
    if cfg!(feature = "backend-calibrated") {
        features.push("backend-calibrated");
    }
    if cfg!(feature = "backend-mamba") {
        features.push("backend-mamba");
    }
    if cfg!(feature = "backend-rwkv") {
        features.push("backend-rwkv");
    }
    if cfg!(feature = "backend-zpaq") {
        features.push("backend-zpaq");
    }
    if cfg!(feature = "cli") {
        features.push("cli");
    }
    if cfg!(feature = "vm") {
        features.push("vm");
    }
    features
}

fn bounds_crc32() -> String {
    let value = json!({
        "allowed_backends": ["ctw"],
        "forbidden_backends": [],
        "parameter_ranges": [{
            "parameter": "rate_backend.depth",
            "min_bits": 1.0f64.to_bits(),
            "max_bits": 16.0f64.to_bits(),
        }],
        "max_experts": 2,
        "max_mixture_nesting_depth": 1,
        "min_experts": 1,
        "allow_duplicate_experts": false,
        "required_experts": [],
        "forbidden_expert_pairs": [],
    });
    crc32_hex(&serde_json::to_vec(&value).expect("bounds hash json"))
}

fn observation_adapter_crc32() -> String {
    let value = json!({
        "kind": "single-channel-conditional-byte-adapter-v1",
        "schema_version": 1,
        "stream": "fixed_len_packed_u64_little_endian",
        "fields": [
            "fail_flag",
            "normalized_physical_size",
            "normalized_target_loss",
            "normalized_eval_time",
            "physical_size_delta",
            "eval_time_delta",
            "candidate_signature_crc32",
            "terminal"
        ],
        "missing_sentinel": 255,
        "nonfinite_float_encoding": "forbidden_before_encoding",
        "delta_time_epsilon": 1.0e-9f64,
    });
    crc32_hex(&serde_json::to_vec(&value).expect("observation adapter hash json"))
}

fn dataset_crc32(path: &Path) -> String {
    crc32_hex(&fs::read(path).expect("read dataset for crc32"))
}

fn evaluator_profile_crc32_with_runtime_profile_and_worker(
    dataset_path: &Path,
    timing: TimingCertificationTier,
    deterministic_table_requested: bool,
    worker_executable_override: Option<&Path>,
) -> String {
    let (
        worker_executable_identity,
        resolved_memory_accounting_kind,
        resolved_memory_accounting_strict_theorem_facing,
        resolved_evaluator_cgroup_parent,
        backend_report_component_policy,
    ) = if deterministic_table_requested {
        (
            None::<String>,
            "deterministic_evaluator_table_row_peak_memory",
            true,
            None::<String>,
            "none_deterministic_table_row",
        )
    } else {
        let worker_executable = if let Some(path) = worker_executable_override {
            if !path.is_file() {
                panic!(
                    "evaluator_worker_executable '{}' does not resolve to a file",
                    path.display()
                );
            }
            path.to_path_buf()
        } else if let Some(path) = std::env::var_os("INFOTHEORY_TUNER_EVAL_WORKER_EXE") {
            let path = PathBuf::from(path);
            if !path.is_file() {
                panic!(
                    "INFOTHEORY_TUNER_EVAL_WORKER_EXE '{}' does not resolve to a file",
                    path.display()
                );
            }
            path
        } else if let Some(path) = std::env::var_os("CARGO_BIN_EXE_infotheory") {
            let path = PathBuf::from(path);
            if path.is_file() {
                path
            } else {
                std::env::current_exe().expect("resolve current executable for evaluator worker")
            }
        } else {
            std::env::current_exe().expect("resolve current executable for evaluator worker")
        };
        let worker_bytes = fs::read(&worker_executable).unwrap_or_else(|err| {
            panic!(
                "failed to read evaluator worker executable '{}' for profile hash: {err}",
                worker_executable.display()
            )
        });
        (
            Some(format!(
                "crc32:{}:bytes:{}",
                crc32_hex(&worker_bytes),
                worker_bytes.len()
            )),
            "unix_process_rss_fallback_explicit",
            false,
            None,
            "none",
        )
    };
    let value = json!({
        "dataset_kind": "passive_bytes",
        "objective_target": "passive_ac",
        "dataset_lowering_version": "passive-bytes-v1",
        "dataset_codec_hash": "passive-identity-bytes",
        "event_grammar_hash": "passive-target-only-byte-stream",
        "target_domain_support_hash": crc32_hex(b"passive-byte-alphabet"),
        "causal_header_profile_hash": crc32_hex(b"passive-none"),
        "target_size_function": "passive-bytes-len",
        "evaluator_interface_version": "typed-causal-evaluator-v1",
        "candidate_canonicalization_version": "bounds-v1",
        "warmup_baseline_runs": 0,
        "diagnostic_chunk_bytes": null,
        "effective_eval_time_limit_seconds_bits": 1.0f64.to_bits(),
        "evaluator_threads": 1,
        "worker_isolation_mode": "spawn_exec_worker",
        "worker_executable_identity": worker_executable_identity,
        "resolved_memory_accounting_kind": resolved_memory_accounting_kind,
        "resolved_memory_accounting_strict_theorem_facing": resolved_memory_accounting_strict_theorem_facing,
        "resolved_evaluator_cgroup_parent": resolved_evaluator_cgroup_parent,
        "backend_report_component_policy": backend_report_component_policy,
        "evaluator_determinism": "deterministic_under_h",
        "rss_mode": "process_rss_peak",
        "timing_certification_tier": timing_label(timing),
        "build_profile": option_env!("PROFILE").unwrap_or("unknown"),
        "feature_set": feature_set(),
    });
    let _ = dataset_path;
    crc32_hex(&serde_json::to_vec(&value).expect("profile hash json"))
}

fn compiled_tune_hashes(spec: &Value, dir: &Path) -> (String, String) {
    let document = SpecDocument::parse_json_value(spec, dir).expect("parse tune spec for certs");
    let SpecDocument::Tune(tune) = document else {
        panic!("expected tune spec");
    };
    let compiled = tune
        .compile_in(&SpecEnvironment::new(dir))
        .expect("compile tune spec for certs");
    (
        crc32_hex(compiled.canonical_bytes().as_slice()),
        crc32_hex(compiled.baseline_candidate().canonical_bytes().as_slice()),
    )
}

fn common_certificate(
    kind: &str,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
) -> Value {
    common_certificate_with_runtime_profile(kind, dataset_path, timing, controller_kind, false)
}

fn common_certificate_with_runtime_profile(
    kind: &str,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    deterministic_table_requested: bool,
) -> Value {
    common_certificate_with_runtime_profile_and_worker(
        kind,
        dataset_path,
        timing,
        controller_kind,
        deterministic_table_requested,
        None,
    )
}

fn common_certificate_with_runtime_profile_and_worker(
    kind: &str,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    deterministic_table_requested: bool,
    worker_executable_override: Option<&Path>,
) -> Value {
    json!({
        "schema_version": 1,
        "kind": kind,
        "dataset_crc32": dataset_crc32(dataset_path),
        "bounds_crc32": bounds_crc32(),
        "evaluator_profile_crc32": evaluator_profile_crc32_with_runtime_profile_and_worker(
            dataset_path,
            timing,
            deterministic_table_requested,
            worker_executable_override,
        ),
        "controller_kind": controller_kind,
        "action_alphabet_size": 2,
    })
}

fn write_common_certificate(
    path: &Path,
    kind: &str,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
) -> String {
    write_common_certificate_with_runtime_profile(
        path,
        kind,
        dataset_path,
        timing,
        controller_kind,
        false,
    )
}

fn write_common_certificate_with_runtime_profile(
    path: &Path,
    kind: &str,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    deterministic_table_requested: bool,
) -> String {
    let value = common_certificate_with_runtime_profile(
        kind,
        dataset_path,
        timing,
        controller_kind,
        deterministic_table_requested,
    );
    write_json(path, &value);
    crc32_hex(&fs::read(path).expect("read cert"))
}

fn write_exact_reward_certificate(
    path: &Path,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    max_reward: u64,
) -> String {
    write_exact_reward_certificate_with_runtime_profile(
        path,
        dataset_path,
        timing,
        controller_kind,
        max_reward,
        false,
    )
}

fn write_exact_reward_certificate_with_runtime_profile(
    path: &Path,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    max_reward: u64,
    deterministic_table_requested: bool,
) -> String {
    write_exact_reward_certificate_with_runtime_profile_and_worker(
        path,
        dataset_path,
        timing,
        controller_kind,
        max_reward,
        deterministic_table_requested,
        None,
    )
}

fn write_exact_reward_certificate_with_runtime_profile_and_worker(
    path: &Path,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    max_reward: u64,
    deterministic_table_requested: bool,
    worker_executable_override: Option<&Path>,
) -> String {
    let mut value = common_certificate_with_runtime_profile_and_worker(
        "exact_reward_encoding",
        dataset_path,
        timing,
        controller_kind,
        deterministic_table_requested,
        worker_executable_override,
    );
    let object = value.as_object_mut().expect("certificate object");
    object.insert(
        "encoding".to_string(),
        json!("integer_objective_difference"),
    );
    object.insert(
        "scalar_representation".to_string(),
        json!("scalar://finite-f64"),
    );
    object.insert("reward_bits".to_string(), json!(16));
    object.insert("max_reward".to_string(), json!(max_reward));
    write_json(path, &value);
    crc32_hex(&fs::read(path).expect("read reward cert"))
}

fn write_finite_reward_map_certificate(
    path: &Path,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    max_reward: u64,
    complete_nonnegative_interval_max: Option<u64>,
    values: Value,
) -> String {
    write_finite_reward_map_certificate_with_runtime_profile(
        path,
        dataset_path,
        timing,
        controller_kind,
        max_reward,
        complete_nonnegative_interval_max,
        values,
        false,
    )
}

fn write_finite_reward_map_certificate_with_runtime_profile(
    path: &Path,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    max_reward: u64,
    complete_nonnegative_interval_max: Option<u64>,
    values: Value,
    deterministic_table_requested: bool,
) -> String {
    let mut value = common_certificate_with_runtime_profile(
        "exact_reward_encoding",
        dataset_path,
        timing,
        controller_kind,
        deterministic_table_requested,
    );
    let object = value.as_object_mut().expect("certificate object");
    object.insert("encoding".to_string(), json!("finite_reward_map"));
    object.insert(
        "scalar_representation".to_string(),
        json!("scalar://finite-f64"),
    );
    object.insert("reward_bits".to_string(), json!(16));
    object.insert("max_reward".to_string(), json!(max_reward));
    if let Some(complete_max) = complete_nonnegative_interval_max {
        object.insert(
            "complete_nonnegative_interval_max".to_string(),
            json!(complete_max),
        );
    }
    object.insert("values".to_string(), values);
    write_json(path, &value);
    crc32_hex(&fs::read(path).expect("read reward map cert"))
}

fn write_observation_certificate(
    path: &Path,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    finite_planner_state_certificate_crc32: &str,
) -> String {
    write_observation_certificate_with_runtime_profile(
        path,
        dataset_path,
        timing,
        controller_kind,
        finite_planner_state_certificate_crc32,
        false,
    )
}

fn write_observation_certificate_with_runtime_profile(
    path: &Path,
    dataset_path: &Path,
    timing: TimingCertificationTier,
    controller_kind: &str,
    finite_planner_state_certificate_crc32: &str,
    deterministic_table_requested: bool,
) -> String {
    let mut value = common_certificate_with_runtime_profile(
        "exact_state_observation",
        dataset_path,
        timing,
        controller_kind,
        deterministic_table_requested,
    );
    let object = value.as_object_mut().expect("certificate object");
    object.insert("observation_key_mode".to_string(), json!("full_stream"));
    object.insert(
        "exact_state_encoder_spec_ref".to_string(),
        json!("encoder://full-state"),
    );
    object.insert(
        "observation_adapter_spec_ref".to_string(),
        json!("single-channel-conditional-byte-adapter-v1"),
    );
    object.insert(
        "observation_adapter_content_crc32".to_string(),
        json!(observation_adapter_crc32()),
    );
    object.insert(
        "finite_planner_state_certificate_crc32".to_string(),
        json!(finite_planner_state_certificate_crc32),
    );
    object.insert(
        "psi_h_outputs".to_string(),
        json!([
            {"state_id": "baseline", "observations": [1]},
            {"state_id": "terminal", "observations": [2]}
        ]),
    );
    write_json(path, &value);
    crc32_hex(&fs::read(path).expect("read observation cert"))
}

fn write_deterministic_table(
    path: &Path,
    dataset_path: &Path,
    controller_kind: &str,
    baseline_candidate_crc32: &str,
) -> String {
    write_deterministic_table_with_peak_memory(
        path,
        dataset_path,
        controller_kind,
        baseline_candidate_crc32,
        1,
    )
}

fn write_deterministic_table_with_peak_memory(
    path: &Path,
    dataset_path: &Path,
    controller_kind: &str,
    baseline_candidate_crc32: &str,
    peak_memory_bytes: u64,
) -> String {
    let mut value = common_certificate_with_runtime_profile(
        "deterministic_evaluator_table",
        dataset_path,
        TimingCertificationTier::DeterministicTable,
        controller_kind,
        true,
    );
    let object = value.as_object_mut().expect("certificate object");
    object.insert(
        "rows".to_string(),
        json!([{
            "candidate_crc32": baseline_candidate_crc32,
            "status": "success",
            "compressed_bytes": 16,
            "target_loss_bits": 128.0,
            "elapsed_seconds": 0.01,
            "peak_memory_bytes": peak_memory_bytes
        }]),
    );
    write_json(path, &value);
    crc32_hex(&fs::read(path).expect("read table cert"))
}

fn mismatch_marker_value(err: &str, marker: &str) -> Option<String> {
    let start = err.find(marker)? + marker.len();
    let rest = &err[start..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

fn derive_runtime_warmstart_task_fingerprint(
    args: &[String],
    teacher_path: &Path,
    reward_cert_crc32: &str,
) -> String {
    let request = parse_tune_command_args(args).expect("parse tune args for warmstart probe");
    write_teacher(teacher_path, "probe", reward_cert_crc32);
    let err = run_tune(&request)
        .expect_err("warmstart probe must fail before teacher task_fingerprint is corrected");
    if let Some(value) = mismatch_marker_value(&err, "current planner_run '") {
        return value;
    }
    panic!("warmstart probe must expose current planner_run fingerprint marker, got: {err}");
}

fn tune_args(
    spec_path: &Path,
    timing: TimingCertificationTier,
    max_evaluations: usize,
) -> Vec<String> {
    let args = vec![
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(spec_path),
        "--max-evaluations".to_string(),
        max_evaluations.to_string(),
        "--timing-tier".to_string(),
        timing_label(timing).to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--exact-state-encoder-spec-ref".to_string(),
        "encoder://full-state".to_string(),
        "--claim-exact-finite-mdp".to_string(),
        "--claim-exact-observed-markov".to_string(),
        "--claim-planner-convergence".to_string(),
    ];
    args
}

fn run_tune_case(
    dir: &Path,
    label: &str,
    dataset_path: &Path,
    teacher_path: &Path,
    case: &ControllerCase,
    timing: TimingCertificationTier,
    max_evaluations: usize,
) -> Value {
    let suffix = format!("{label}_{}_{}", case.kind, timing_label(timing));
    let spec_path = dir.join(format!("{suffix}.json"));
    let output_path = dir.join(format!("{suffix}_output.json"));
    let report_path = dir.join(format!("{suffix}_report.json"));
    let teacher = case.needs_teacher.then_some(teacher_path);
    if case.needs_teacher {
        write_placeholder_teacher(teacher_path);
    }
    let spec_value = tune_spec(
        dataset_path,
        &output_path,
        &report_path,
        case.controller.clone(),
        teacher,
    );
    write_json(&spec_path, &spec_value);
    let (_spec_crc32, _baseline_crc32) = compiled_tune_hashes(&spec_value, dir);
    let mut args = tune_args(&spec_path, timing, max_evaluations);
    if matches!(case.kind, "mc_aixi_fac_ctw" | "aiqi_warmstart_exact_jh") {
        let reward_cert_path = dir.join(format!("{suffix}_exact_reward.json"));
        let reward_cert_crc32 = write_exact_reward_certificate(
            &reward_cert_path,
            dataset_path,
            timing,
            case.kind,
            65_535,
        );
        args.push("--exact-reward-encoding-certificate".to_string());
        args.push(path_string(&reward_cert_path));
        if case.needs_teacher {
            let task_fingerprint =
                derive_runtime_warmstart_task_fingerprint(&args, teacher_path, &reward_cert_crc32);
            write_teacher(teacher_path, &task_fingerprint, &reward_cert_crc32);
        }
    }
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune");
    assert!(output_path.exists());
    read_json(&report_path)
}

fn assert_required_report_fields(report: &Value) {
    assert_eq!(str_at(report, "/kind"), "tune_report");
    assert_eq!(u64_at(report, "/schema_version"), 1);
    assert!(str_at(report, "/spec_crc32").len() == 8);
    assert!(str_at(report, "/evaluator_profile_crc32").len() == 8);
    assert!(str_at(report, "/provenance/bounds_crc32").len() == 8);
    assert!(str_at(report, "/input_asset/content_crc32").len() == 8);
    assert!(str_at(report, "/input_asset/lowered_skeleton_crc32").len() == 8);
    assert!(str_at(report, "/baseline/candidate_crc32").len() == 8);
    assert!(str_at(report, "/best/candidate_crc32").len() == 8);
    assert_eq!(
        str_at(report, "/cache/key_candidate_crc32"),
        str_at(report, "/best/candidate_crc32")
    );
    assert!(str_at(report, "/cache/key_digest_crc32").len() == 8);
    assert!(
        report["feature_set"]
            .as_array()
            .expect("feature_set array")
            .iter()
            .any(|item| item.as_str() == Some("tuner"))
    );
    assert!(u64_at(report, "/baseline/model_bytes") > 0);
    assert!(u64_at(report, "/best/model_bytes") > 0);
    assert_eq!(str_at(report, "/baseline/status"), "success");
    assert_eq!(str_at(report, "/best/status"), "success");
    assert!(f64_at(report, "/baseline/target_loss_bits").is_finite());
    assert!(f64_at(report, "/baseline/objective_bits").is_finite());
    assert!(f64_at(report, "/baseline/throughput_runtime_cap_seconds").is_finite());
    assert!(f64_at(report, "/baseline/effective_eval_time_limit_seconds").is_finite());
    assert!(u64_at(report, "/cache/candidate_evaluations_executed") >= 1);
    assert!(u64_at(report, "/search/fatal_evaluator_failures") <= 1);
    let counted_results = u64_at(report, "/search/candidate_result_counts/success_deployable")
        + u64_at(
            report,
            "/search/candidate_result_counts/success_non_deployable",
        )
        + u64_at(report, "/search/candidate_result_counts/timeout")
        + u64_at(report, "/search/candidate_result_counts/invalid")
        + u64_at(report, "/search/candidate_result_counts/error_recoverable");
    assert_eq!(
        counted_results,
        u64_at(report, "/search/non_warmup_candidate_results_seen")
    );
    let _external_asset_forbidden = u64_at(
        report,
        "/search/invalid_reason_counts/candidate_external_asset_forbidden",
    );
    assert!(matches!(
        str_at(report, "/provenance/canonical_code_certification/basis"),
        "structural_self_delimiting_binary_encoding_plus_tests"
    ));
    assert!(bool_at(
        report,
        "/provenance/canonical_code_certification/top_level_length_prefix"
    ));
    assert!(bool_at(
        report,
        "/provenance/canonical_code_certification/trailing_bytes_rejected"
    ));
    assert!(bool_at(
        report,
        "/baseline/physical_compressed_bytes_diagnostic_only"
    ));
    assert!(bool_at(
        report,
        "/best/physical_compressed_bytes_diagnostic_only"
    ));
    assert!(bool_at(report, "/output/output_written"));
    assert!(!str_at(report, "/output/output_config_path").is_empty());
    assert!(str_at(report, "/output/output_candidate_crc32").len() == 8);
}

#[test]
fn canonical_tune_document_rejects_executor_side_fields() {
    let dir = temp_dir("canonical_rejects_executor_fields");
    let dataset_path = dir.join("dataset.bin");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let base = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": 1,
        }),
        None,
    );
    for field in ["execution_profile", "theorem", "max_evaluations"] {
        let mut rejected = base.clone();
        rejected[field] = json!({});
        let err = match SpecDocument::parse_json_value(&rejected, &dir) {
            Ok(_) => panic!("canonical tune document accepted executor-side field {field}"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains(&format!("unknown tune field '{field}'")),
            "{err}"
        );
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn canonical_tune_document_rejects_unknown_nested_fields() {
    let dir = temp_dir("canonical_rejects_nested_fields");
    let dataset_path = dir.join("dataset.bin");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let base = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": 1,
        }),
        None,
    );
    let cases = [
        (
            "/assets/0",
            "unexpected_asset_field",
            "unknown tune.assets[0] field 'unexpected_asset_field'",
        ),
        (
            "/controller",
            "unexpected_controller_field",
            "unknown controller.annealed_hill_climbing field 'unexpected_controller_field'",
        ),
        (
            "/bounds",
            "unexpected_bounds_field",
            "unknown bounds field 'unexpected_bounds_field'",
        ),
        (
            "/bounds/parameter_ranges/0",
            "unexpected_range_field",
            "unknown bounds.parameter_ranges[0] field 'unexpected_range_field'",
        ),
    ];
    for (pointer, field, expected) in cases {
        let mut rejected = base.clone();
        rejected
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("{pointer} exists"))[field] = json!(true);
        let err = match SpecDocument::parse_json_value(&rejected, &dir) {
            Ok(_) => panic!("canonical tune document accepted nested field {field}"),
            Err(err) => err,
        };
        assert!(err.to_string().contains(expected), "{err}");
    }
    let mut rejected = base.clone();
    rejected["controller"] = json!({
        "kind": "mc_aixi_fac_ctw",
        "interface": {
            "observation_bits": 8,
            "observation_stream_len": 1,
            "observation_key_mode": "full_stream",
            "reward_bits": 8,
            "agent_actions": 2,
            "unexpected_interface_field": true,
        },
        "planner_simulations_per_step": 2,
    });
    let err = match SpecDocument::parse_json_value(&rejected, &dir) {
        Ok(_) => panic!("canonical tune document accepted nested interface field"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("unknown controller.interface field 'unexpected_interface_field'"),
        "{err}"
    );

    let mut rejected = base.clone();
    rejected["baseline_candidate"]["unexpected_candidate_field"] = json!(true);
    let err = match SpecDocument::parse_json_value(&rejected, &dir) {
        Ok(_) => panic!("canonical tune document accepted unknown baseline_candidate field"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("baseline_candidate must be canonical compression backend JSON"),
        "{err}"
    );

    let mut rejected = base.clone();
    rejected["baseline_candidate"]["rate_backend"]["unexpected_rate_backend_field"] = json!(true);
    let err = match SpecDocument::parse_json_value(&rejected, &dir) {
        Ok(_) => panic!("canonical tune document accepted unknown rate_backend field"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("baseline_candidate must be canonical compression backend JSON"),
        "{err}"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn canonical_tune_document_rejects_candidate_local_external_assets() {
    const EXTERNAL_ASSET_FORBIDDEN: &str = "candidate_external_asset_forbidden";
    let dir = temp_dir("canonical_rejects_candidate_external_assets");
    let dataset_path = dir.join("dataset.bin");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    for (field, value) in [
        ("model_path", json!("weights.safetensors")),
        ("spec_path", json!("nested/spec.json")),
        ("base_path", json!("nested/base.json")),
    ] {
        let mut candidate = baseline_candidate();
        candidate[field] = value;
        let spec = tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1,
            }),
            None,
        );
        let mut rejected = spec.clone();
        rejected["baseline_candidate"] = candidate;
        let err = match SpecDocument::parse_json_value(&rejected, &dir) {
            Ok(_) => panic!("candidate-local external asset field must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains(EXTERNAL_ASSET_FORBIDDEN), "{err}");
        assert!(
            err.to_string().contains("candidate-local external asset"),
            "{err}"
        );
    }
    let mut rejected = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": 1,
        }),
        None,
    );
    rejected["baseline_candidate"]["rate_backend"]["method"] =
        json!("online;policy:load_from=weights.safetensors");
    let err = match SpecDocument::parse_json_value(&rejected, &dir) {
        Ok(_) => panic!("policy load_from must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains(EXTERNAL_ASSET_FORBIDDEN), "{err}");
    assert!(err.to_string().contains("policy load_from"), "{err}");

    let mut rejected = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": 1,
        }),
        None,
    );
    rejected["baseline_candidate"]["rate_backend"]["method"] = json!({
        "kind": "file",
        "path": "weights.safetensors",
    });
    let err = match SpecDocument::parse_json_value(&rejected, &dir) {
        Ok(_) => panic!("method.path candidate-local asset must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains(EXTERNAL_ASSET_FORBIDDEN), "{err}");
    assert!(
        err.to_string()
            .contains("candidate-local external asset field"),
        "{err}"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn canonical_tune_document_requires_assets_array() {
    let dir = temp_dir("canonical_requires_assets");
    let dataset_path = dir.join("dataset.bin");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let base = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": 1,
        }),
        None,
    );
    let mut missing = base.clone();
    missing.as_object_mut().expect("object").remove("assets");
    let err = match SpecDocument::parse_json_value(&missing, &dir) {
        Ok(_) => panic!("canonical tune document accepted missing assets"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("tune.assets is required"), "{err}");
    let mut wrong_type = base;
    wrong_type["assets"] = json!({});
    let err = match SpecDocument::parse_json_value(&wrong_type, &dir) {
        Ok(_) => panic!("canonical tune document accepted non-array assets"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("tune.assets must be an array"),
        "{err}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tune_cli_loads_binary_itsd_tune_document() {
    let dir = temp_dir("tune_itsd_cli");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.itsd");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": 1,
        }),
        None,
    );
    write_tune_itsd(&spec_path, &spec_value, &dir);
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune from itsd");
    let report = read_json(&report_path);
    assert_eq!(str_at(&report, "/kind"), "tune_report");
    assert!(bool_at(&report, "/output/output_written"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tune_executor_accepts_supported_controls() {
    let supported = [
        vec!["--threads", "2"],
        vec!["--cpu-affinity", "0"],
        vec!["--evaluator-worker-executable", "/tmp/infotheory-worker"],
        vec![
            "--evaluator-cgroup-parent",
            "/sys/fs/cgroup/infotheory-tuner",
        ],
        vec!["--log-path", "tune.log"],
        vec!["--diagnostic-chunk-bytes", "4096"],
        vec!["--rss-mode", "process_rss_peak"],
        vec!["--rss-mode", "backend_reported"],
        vec!["--rss-mode", "hybrid_strict_max"],
        vec!["--planner-deployable-model"],
        vec!["--warmstart-trace-refresh"],
        vec![
            "--annealer-kernel-profile",
            "compiled_uniform_metropolis_hastings",
        ],
    ];
    for flags in supported {
        let mut args = vec![
            "infotheory".to_string(),
            "tune".to_string(),
            "spec.json".to_string(),
        ];
        args.extend(flags.iter().map(|flag| (*flag).to_string()));
        assert!(
            parse_tune_command_args(&args).is_ok(),
            "supported executor flags were rejected: {flags:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn run_tune_rejects_unusable_worker_executable() {
    let dir = temp_dir("invalid_worker_executable");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let missing_worker = dir.join("missing-worker-bin");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1
            }),
            None,
        ),
    );
    let args = vec![
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--evaluator-worker-executable".to_string(),
        path_string(&missing_worker),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("nonexistent worker executable must fail");
    assert!(
        err.contains("execution.evaluator_worker_executable")
            && err.contains("does not resolve to a file"),
        "{err}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn run_tune_strict_memory_mode_requires_delegated_cgroup_parent() {
    let dir = temp_dir("strict_memory_requires_cgroup");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1
            }),
            None,
        ),
    );
    let args = vec![
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--rss-mode".to_string(),
        "hybrid_strict_max".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request)
        .expect_err("strict memory-accounting mode without cgroup parent must fail");
    assert!(
        err.contains("strict memory-accounting mode") && err.contains("delegated cgroup-v2 parent"),
        "{err}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tune_executor_rejects_zero_diagnostic_chunk_bytes() {
    let args = vec![
        "infotheory".to_string(),
        "tune".to_string(),
        "spec.json".to_string(),
        "--diagnostic-chunk-bytes".to_string(),
        "0".to_string(),
    ];
    let err = parse_tune_command_args(&args).expect_err("zero chunk size should fail");
    assert!(err.contains("diagnostic_chunk_bytes must be >= 1"), "{err}");
}

#[test]
fn tune_exec_config_cli_overrides_are_order_independent() {
    let dir = temp_dir("exec_config_precedence");
    let exec_path = dir.join("exec.json");
    write_json(
        &exec_path,
        &json!({
            "max_evaluations": 1,
            "warmup_baseline_runs": 1,
            "theorem": {
                "timing_certification_tier": "isolated"
            }
        }),
    );
    let exec_arg = path_string(&exec_path);
    let before = [
        "infotheory",
        "tune",
        "spec.json",
        "--max-evaluations",
        "3",
        "--timing-tier",
        "real_time",
        "--exec-config",
        exec_arg.as_str(),
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let after = [
        "infotheory",
        "tune",
        "spec.json",
        "--exec-config",
        exec_arg.as_str(),
        "--max-evaluations",
        "3",
        "--timing-tier",
        "real_time",
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let before = parse_tune_command_args(&before).expect("parse before");
    let after = parse_tune_command_args(&after).expect("parse after");
    assert_eq!(before.execution.max_evaluations, Some(3));
    assert_eq!(after.execution.max_evaluations, Some(3));
    assert_eq!(before.execution.warmup_baseline_runs, 1);
    assert_eq!(after.execution.warmup_baseline_runs, 1);
    assert_eq!(
        before.execution.theorem.timing_certification_tier,
        TimingCertificationTier::RealTime
    );
    assert_eq!(
        after.execution.theorem.timing_certification_tier,
        TimingCertificationTier::RealTime
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tune_exec_config_rejects_unknown_fields_and_malformed_theorem() {
    let dir = temp_dir("exec_config_strict");
    let unknown_path = dir.join("unknown.json");
    write_json(
        &unknown_path,
        &json!({
            "max_evaluations": 1,
            "determinism_dealine_certificate": "typo"
        }),
    );
    let err = parse_tune_command_args(&[
        "infotheory".to_string(),
        "tune".to_string(),
        "spec.json".to_string(),
        "--exec-config".to_string(),
        path_string(&unknown_path),
    ])
    .expect_err("unknown key must be rejected");
    assert!(err.contains("unknown execution config field"), "{err}");

    let malformed_path = dir.join("malformed_theorem.json");
    write_json(
        &malformed_path,
        &json!({
            "max_evaluations": 1,
            "theorem": "not an object"
        }),
    );
    let err = parse_tune_command_args(&[
        "infotheory".to_string(),
        "tune".to_string(),
        "spec.json".to_string(),
        "--exec-config".to_string(),
        path_string(&malformed_path),
    ])
    .expect_err("malformed theorem value must be rejected");
    assert!(err.contains("field 'theorem' must be an object"), "{err}");

    let empty_cert_path = dir.join("empty_certificate.json");
    write_json(
        &empty_cert_path,
        &json!({
            "max_evaluations": 1,
            "theorem": {
                "finite_planner_state_certificate": ""
            }
        }),
    );
    let err = parse_tune_command_args(&[
        "infotheory".to_string(),
        "tune".to_string(),
        "spec.json".to_string(),
        "--exec-config".to_string(),
        path_string(&empty_cert_path),
    ])
    .expect_err("empty theorem certificate reference must be rejected");
    assert!(
        err.contains("finite_planner_state_certificate must be a non-empty string"),
        "{err}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tune_executor_rejects_unsupported_certificate_uri_scheme() {
    let dir = temp_dir("unsupported_certificate_uri");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1,
            }),
            None,
        ),
    );
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--determinism-deadline-certificate".to_string(),
        "https://example.invalid/deadline.json".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("unsupported certificate URI must fail");
    assert!(
        err.contains("unsupported theorem certificate reference scheme"),
        "{err}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn run_tune_records_log_controls() {
    let dir = temp_dir("executor_logging");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let log_path = dir.join("tune.log");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1,
            }),
            None,
        ),
    );
    let args = [
        "infotheory",
        "tune",
        path_string(&spec_path).as_str(),
        "--max-evaluations",
        "1",
        "--log-path",
        path_string(&log_path).as_str(),
        "--diagnostic-chunk-bytes",
        "2",
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune");
    let report = read_json(&report_path);
    assert_eq!(
        str_at(&report, "/provenance/executor_controls/log_path"),
        path_string(&log_path)
    );
    assert_eq!(
        u64_at(
            &report,
            "/provenance/executor_controls/diagnostic_chunk_bytes"
        ),
        2
    );
    assert!(bool_at(&report, "/provenance/diagnostic_chunking/enabled"));
    assert_eq!(
        u64_at(&report, "/provenance/diagnostic_chunking/chunk_count"),
        21
    );
    assert_eq!(
        u64_at(&report, "/input_asset/diagnostic_chunking/chunk_bytes"),
        2
    );
    assert!(!bool_at(
        &report,
        "/input_asset/diagnostic_chunking/affects_objective"
    ));
    let log = fs::read_to_string(&log_path).expect("read executor log");
    assert!(log.contains("\"event\":\"start\""), "{log}");
    assert!(log.contains("\"event\":\"finish\""), "{log}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn exact_family_controller_rejects_missing_exact_reward_certificate() {
    let dir = temp_dir("missing_exact_reward_certificate");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            mc_case.controller,
            None,
        ),
    );
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("exact controller must reject missing certificate");
    assert!(err.contains("reward_encoding_unsafe"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn exact_reward_certificate_rejects_unrepresentable_reachable_reward() {
    let dir = temp_dir("bad_exact_reward_certificate");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let reward_cert_path = dir.join("reward.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            mc_case.controller,
            None,
        ),
    );
    write_exact_reward_certificate(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
        1,
    );
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert_path),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("too-small reward certificate must fail");
    assert!(
        err.contains("exceeds verified exact reward maximum"),
        "{err}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn exact_controller_rejects_finite_reward_map_without_complete_interval() {
    let dir = temp_dir("incomplete_finite_reward_map");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let reward_cert_path = dir.join("reward_map.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        mc_case.controller,
        None,
    );
    write_json(&spec_path, &spec_value);
    let reward_values = (0..=1_000u64)
        .map(|objective_difference| {
            json!({
                "objective_difference": objective_difference,
                "symbol": objective_difference
            })
        })
        .collect::<Vec<_>>();
    write_finite_reward_map_certificate(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
        65_535,
        None,
        Value::Array(reward_values),
    );
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert_path),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("incomplete finite map must fail before runtime");
    assert!(err.contains("complete_nonnegative_interval_max"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn deterministic_table_certificates_can_certify_theorem_claims_when_used() {
    let dir = temp_dir("deterministic_table_certified");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        mc_case.controller,
        None,
    );
    write_json(&spec_path, &spec_value);
    let (_, baseline_candidate_crc32) = compiled_tune_hashes(&spec_value, &dir);

    let finite_cert = dir.join("finite.json");
    let no_hidden_cert = dir.join("no_hidden.json");
    let reward_cert = dir.join("reward.json");
    let observation_cert = dir.join("observation.json");
    let table_cert = dir.join("table.json");
    let finite_cert_crc32 = write_common_certificate_with_runtime_profile(
        &finite_cert,
        "finite_planner_state",
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "mc_aixi_fac_ctw",
        true,
    );
    write_common_certificate_with_runtime_profile(
        &no_hidden_cert,
        "no_hidden_state",
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "mc_aixi_fac_ctw",
        true,
    );
    write_exact_reward_certificate_with_runtime_profile(
        &reward_cert,
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "mc_aixi_fac_ctw",
        65_535,
        true,
    );
    write_observation_certificate_with_runtime_profile(
        &observation_cert,
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "mc_aixi_fac_ctw",
        &finite_cert_crc32,
        true,
    );
    write_deterministic_table(
        &table_cert,
        &dataset_path,
        "mc_aixi_fac_ctw",
        &baseline_candidate_crc32,
    );

    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--timing-tier".to_string(),
        "deterministic_table".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--finite-planner-state-certificate".to_string(),
        path_string(&finite_cert),
        "--no-hidden-state-certificate".to_string(),
        path_string(&no_hidden_cert),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert),
        "--exact-state-encoder-spec-ref".to_string(),
        "encoder://full-state".to_string(),
        "--exact-state-observation-certificate".to_string(),
        path_string(&observation_cert),
        "--deterministic-evaluator-table".to_string(),
        path_string(&table_cert),
        "--claim-exact-finite-mdp".to_string(),
        "--claim-exact-observed-markov".to_string(),
        "--claim-planner-convergence".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune");
    let report = read_json(&report_path);
    assert_eq!(
        str_at(&report, "/theorem_claims/exact_finite_mdp/status"),
        "certified"
    );
    assert_eq!(
        str_at(&report, "/theorem_claims/exact_observed_markov/status"),
        "certified"
    );
    assert_eq!(
        str_at(&report, "/theorem_claims/planner_convergence/status"),
        "certified"
    );
    assert!(
        report
            .pointer("/theorem_claims/refs/exact_state_observation_certified")
            .is_none(),
        "theorem refs must not expose unchecked user observation-certification booleans"
    );
    assert!(bool_at(
        &report,
        "/theorem_claims/refs/exact_state_observation_basis/verified_certificate"
    ));
    assert_eq!(
        str_at(&report, "/evaluator_execution_model"),
        "deterministic_table"
    );
    assert_eq!(
        str_at(&report, "/theorem_timing_basis"),
        "verified_deterministic_evaluator_table"
    );
    assert_eq!(u64_at(&report, "/baseline/compressed_bytes"), 16);
    assert_eq!(
        u64_at(
            &report,
            "/provenance/verified_theorem_inputs/deterministic_evaluator_table/rows"
        ),
        1
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn real_time_timing_certificate_sets_verified_timing_basis() {
    let dir = temp_dir("real_time_timing_certified");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        mc_case.controller,
        None,
    );
    write_json(&spec_path, &spec_value);

    let finite_cert = dir.join("finite.json");
    let no_hidden_cert = dir.join("no_hidden.json");
    let reward_cert = dir.join("reward.json");
    let observation_cert = dir.join("observation.json");
    let timing_cert = dir.join("timing.json");
    let finite_cert_crc32 = write_common_certificate(
        &finite_cert,
        "finite_planner_state",
        &dataset_path,
        TimingCertificationTier::RealTime,
        "mc_aixi_fac_ctw",
    );
    write_common_certificate(
        &no_hidden_cert,
        "no_hidden_state",
        &dataset_path,
        TimingCertificationTier::RealTime,
        "mc_aixi_fac_ctw",
    );
    write_exact_reward_certificate(
        &reward_cert,
        &dataset_path,
        TimingCertificationTier::RealTime,
        "mc_aixi_fac_ctw",
        65_535,
    );
    write_observation_certificate(
        &observation_cert,
        &dataset_path,
        TimingCertificationTier::RealTime,
        "mc_aixi_fac_ctw",
        &finite_cert_crc32,
    );
    write_common_certificate(
        &timing_cert,
        "determinism_deadline",
        &dataset_path,
        TimingCertificationTier::RealTime,
        "mc_aixi_fac_ctw",
    );

    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--timing-tier".to_string(),
        "real_time".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--finite-planner-state-certificate".to_string(),
        path_string(&finite_cert),
        "--no-hidden-state-certificate".to_string(),
        path_string(&no_hidden_cert),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert),
        "--exact-state-encoder-spec-ref".to_string(),
        "encoder://full-state".to_string(),
        "--exact-state-observation-certificate".to_string(),
        path_string(&observation_cert),
        "--determinism-deadline-certificate".to_string(),
        path_string(&timing_cert),
        "--claim-planner-convergence".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run real-time certified tune");
    let report = read_json(&report_path);
    assert_eq!(
        str_at(&report, "/theorem_timing_basis"),
        "verified_real_time_deadline_certificate"
    );
    assert_eq!(
        str_at(&report, "/theorem_claims/planner_convergence/status"),
        "uncertified"
    );
    assert!(
        report["theorem_claims"]["planner_convergence"]["missing_prerequisites"]
            .as_array()
            .expect("missing prereqs")
            .iter()
            .any(|item| item.as_str() == Some("strict_theorem_facing_memory_accounting"))
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn deterministic_table_peak_memory_can_make_baseline_nondeployable() {
    let dir = temp_dir("deterministic_table_memory_cap");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    let mut spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        mc_case.controller,
        None,
    );
    spec_value["max_memory_bytes"] = json!(1u64);
    write_json(&spec_path, &spec_value);
    let (_, baseline_candidate_crc32) = compiled_tune_hashes(&spec_value, &dir);
    let reward_cert = dir.join("reward.json");
    let table_cert = dir.join("table.json");
    write_exact_reward_certificate_with_runtime_profile(
        &reward_cert,
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "mc_aixi_fac_ctw",
        65_535,
        true,
    );
    write_deterministic_table_with_peak_memory(
        &table_cert,
        &dataset_path,
        "mc_aixi_fac_ctw",
        &baseline_candidate_crc32,
        2,
    );
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--timing-tier".to_string(),
        "deterministic_table".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert),
        "--deterministic-evaluator-table".to_string(),
        path_string(&table_cert),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("memory cap should reject baseline");
    assert!(err.contains("peak memory 2 bytes"), "{err}");
    let report = read_json(&report_path);
    assert!(!bool_at(&report, "/baseline/deployable"));
    assert_eq!(u64_at(&report, "/baseline/peak_memory_bytes"), 2);
    assert_eq!(u64_at(&report, "/baseline/max_memory_bytes"), 1);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn exact_state_observation_certificate_requires_injectivity_basis() {
    let dir = temp_dir("observation_injectivity_rejected");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        mc_case.controller,
        None,
    );
    write_json(&spec_path, &spec_value);

    let finite_cert = dir.join("finite.json");
    let reward_cert = dir.join("reward.json");
    let observation_cert = dir.join("bad_observation.json");
    let finite_cert_crc32 = write_common_certificate(
        &finite_cert,
        "finite_planner_state",
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
    );
    write_exact_reward_certificate(
        &reward_cert,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
        65_535,
    );
    let mut bad_observation = common_certificate(
        "exact_state_observation",
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
    );
    let object = bad_observation
        .as_object_mut()
        .expect("observation certificate object");
    object.insert("observation_key_mode".to_string(), json!("full_stream"));
    object.insert(
        "exact_state_encoder_spec_ref".to_string(),
        json!("encoder://full-state"),
    );
    object.insert(
        "observation_adapter_spec_ref".to_string(),
        json!("single-channel-conditional-byte-adapter-v1"),
    );
    object.insert(
        "observation_adapter_content_crc32".to_string(),
        json!(observation_adapter_crc32()),
    );
    object.insert(
        "finite_planner_state_certificate_crc32".to_string(),
        json!(finite_cert_crc32),
    );
    object.insert(
        "psi_h_outputs".to_string(),
        json!([
            {"state_id": "state-a", "observations": [7]},
            {"state_id": "state-b", "observations": [7]}
        ]),
    );
    write_json(&observation_cert, &bad_observation);

    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--finite-planner-state-certificate".to_string(),
        path_string(&finite_cert),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert),
        "--exact-state-encoder-spec-ref".to_string(),
        "encoder://full-state".to_string(),
        "--exact-state-observation-certificate".to_string(),
        path_string(&observation_cert),
        "--claim-exact-observed-markov".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("non-injective observation proof must fail");
    assert!(err.contains("not injective"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn exact_state_observation_certificate_rejects_duplicate_state_ids() {
    let dir = temp_dir("observation_duplicate_state_id");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        mc_case.controller,
        None,
    );
    write_json(&spec_path, &spec_value);

    let finite_cert = dir.join("finite.json");
    let reward_cert = dir.join("reward.json");
    let observation_cert = dir.join("duplicate_state_observation.json");
    let finite_cert_crc32 = write_common_certificate(
        &finite_cert,
        "finite_planner_state",
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
    );
    write_exact_reward_certificate(
        &reward_cert,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
        65_535,
    );
    let mut bad_observation = common_certificate(
        "exact_state_observation",
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "mc_aixi_fac_ctw",
    );
    let object = bad_observation
        .as_object_mut()
        .expect("observation certificate object");
    object.insert("observation_key_mode".to_string(), json!("full_stream"));
    object.insert(
        "exact_state_encoder_spec_ref".to_string(),
        json!("encoder://full-state"),
    );
    object.insert(
        "observation_adapter_spec_ref".to_string(),
        json!("single-channel-conditional-byte-adapter-v1"),
    );
    object.insert(
        "observation_adapter_content_crc32".to_string(),
        json!(observation_adapter_crc32()),
    );
    object.insert(
        "finite_planner_state_certificate_crc32".to_string(),
        json!(finite_cert_crc32),
    );
    object.insert(
        "psi_h_outputs".to_string(),
        json!([
            {"state_id": "duplicate", "observations": [7]},
            {"state_id": "duplicate", "observations": [8]}
        ]),
    );
    write_json(&observation_cert, &bad_observation);

    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--finite-planner-state-certificate".to_string(),
        path_string(&finite_cert),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert),
        "--exact-state-encoder-spec-ref".to_string(),
        "encoder://full-state".to_string(),
        "--exact-state-observation-certificate".to_string(),
        path_string(&observation_cert),
        "--claim-exact-observed-markov".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("duplicate observation state ids must fail");
    assert!(err.contains("duplicate state_id"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn run_tune_reports_per_candidate_timeout() {
    let dir = temp_dir("candidate_timeout");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let mut spec = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": 1,
        }),
        None,
    );
    spec["eval_time_limit_seconds"] = json!(0.000000001f64);
    write_json(&spec_path, &spec);
    let args = [
        "infotheory",
        "tune",
        path_string(&spec_path).as_str(),
        "--max-evaluations",
        "1",
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("timeout is not deployable");
    assert!(err.contains("timed out"), "{err}");
    let report = read_json(&report_path);
    assert_eq!(str_at(&report, "/baseline/status"), "timeout");
    assert_eq!(str_at(&report, "/best/status"), "timeout");
    assert!(!bool_at(&report, "/baseline/deployable"));
    assert_eq!(
        u64_at(&report, "/search/candidate_result_counts/timeout"),
        1
    );
    assert_eq!(
        u64_at(
            &report,
            "/search/candidate_result_counts/success_non_deployable"
        ),
        0
    );
    assert!(!output_path.exists());
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn run_tune_treats_worker_ok_false_as_unrecoverable_evaluator_failure() {
    let dir = temp_dir("worker_ok_false_fatal");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let worker_path = dir.join("synthetic-worker.sh");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1,
            }),
            None,
        ),
    );
    fs::write(
        &worker_path,
        br#"#!/bin/sh
if [ "${INFOTHEORY_TUNER_EVAL_WORKER_PING:-0}" = "1" ]; then
  exit 0
fi
printf '%s\n' '{"ok":false,"error":"synthetic worker setup failure"}' > "$INFOTHEORY_TUNER_EVAL_RESPONSE_PATH"
exit 0
"#,
    )
    .expect("write synthetic worker script");
    let mut permissions = fs::metadata(&worker_path)
        .expect("read worker metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&worker_path, permissions).expect("set worker script executable");

    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--evaluator-worker-executable".to_string(),
        path_string(&worker_path),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("worker ok:false must be fatal");
    assert!(
        err.contains("unrecoverable evaluator failure during baseline evaluation"),
        "{err}"
    );
    assert!(err.contains("synthetic worker setup failure"), "{err}");
    if report_path.exists() {
        let report = read_json(&report_path);
        assert_ne!(
            str_at(&report, "/search/termination_reason"),
            "baseline_not_deployable"
        );
    }
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn baseline_candidate_local_error_reports_baseline_not_deployable() {
    let dir = temp_dir("baseline_candidate_local_error");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let worker_path = dir.join("synthetic-worker.sh");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1,
            }),
            None,
        ),
    );
    fs::write(
        &worker_path,
        br#"#!/bin/sh
if [ "${INFOTHEORY_TUNER_EVAL_WORKER_PING:-0}" = "1" ]; then
  exit 0
fi
printf '%s\n' '{"ok":true,"status":"error","compressed_bytes":0,"elapsed_seconds":0.0,"effective_eval_time_limit_seconds":1.0,"throughput_bytes_per_second":0.0,"peak_memory_bytes":0,"target_loss_bits":null,"objective_bits":null,"deployable":false}' > "$INFOTHEORY_TUNER_EVAL_RESPONSE_PATH"
exit 0
"#,
    )
    .expect("write synthetic worker script");
    let mut permissions = fs::metadata(&worker_path)
        .expect("read worker metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&worker_path, permissions).expect("set worker script executable");

    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--evaluator-worker-executable".to_string(),
        path_string(&worker_path),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("baseline error status must be non-deployable");
    assert!(
        err.contains("baseline candidate is not deployable"),
        "unexpected error: {err}"
    );
    let report = read_json(&report_path);
    assert_eq!(
        str_at(&report, "/search/termination_reason"),
        "baseline_not_deployable"
    );
    assert_eq!(str_at(&report, "/baseline/status"), "error");
    assert_eq!(u64_at(&report, "/search/fatal_evaluator_failures"), 0);
    assert_eq!(
        u64_at(&report, "/search/candidate_result_counts/error_recoverable"),
        1
    );
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn baseline_fatal_inner_eval_error_reports_fatal_evaluator_failure() {
    let dir = temp_dir("baseline_fatal_inner_eval_error");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let worker_path = dir.join("synthetic-worker.sh");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1,
            }),
            None,
        ),
    );
    fs::write(
        &worker_path,
        br#"#!/bin/sh
if [ "${INFOTHEORY_TUNER_EVAL_WORKER_PING:-0}" = "1" ]; then
  exit 0
fi
printf '%s\n' '{"ok":false,"error":"synthetic inner fatal evaluator failure"}' > "$INFOTHEORY_TUNER_EVAL_RESPONSE_PATH"
exit 0
"#,
    )
    .expect("write synthetic worker script");
    let mut permissions = fs::metadata(&worker_path)
        .expect("read worker metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&worker_path, permissions).expect("set worker script executable");

    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--evaluator-worker-executable".to_string(),
        path_string(&worker_path),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("worker fatal must abort baseline");
    assert!(
        err.contains("unrecoverable evaluator failure during baseline evaluation"),
        "unexpected error: {err}"
    );
    assert!(
        err.contains("synthetic inner fatal evaluator failure"),
        "{err}"
    );
    assert!(
        !report_path.exists(),
        "fatal baseline evaluator failures should abort before report synthesis"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn run_tune_terminates_on_unrecoverable_evaluator_failure() {
    let dir = temp_dir("fatal_evaluator_failure");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    let table_cert_path = dir.join("deterministic_table.json");
    let reward_cert_path = dir.join("exact_reward.json");
    write_passive_dataset(&dataset_path);
    let mc_case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc_aixi controller case");
    let spec = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        mc_case.controller,
        None,
    );
    write_json(&spec_path, &spec);
    let (_spec_crc32, baseline_candidate_crc32) = compiled_tune_hashes(&spec, &dir);
    write_exact_reward_certificate_with_runtime_profile(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "mc_aixi_fac_ctw",
        65_535,
        true,
    );
    write_deterministic_table(
        &table_cert_path,
        &dataset_path,
        "mc_aixi_fac_ctw",
        &baseline_candidate_crc32,
    );

    let args = [
        "infotheory",
        "tune",
        path_string(&spec_path).as_str(),
        "--max-evaluations",
        "2",
        "--timing-tier",
        "deterministic_table",
        "--scalar-representation-ref",
        "scalar://finite-f64",
        "--exact-reward-encoding-certificate",
        path_string(&reward_cert_path).as_str(),
        "--deterministic-evaluator-table",
        path_string(&table_cert_path).as_str(),
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("missing deterministic-table row must be fatal");
    assert!(
        err.contains("unrecoverable evaluator failure"),
        "unexpected error: {err}"
    );

    let report = read_json(&report_path);
    assert_eq!(
        str_at(&report, "/status"),
        "terminated_unrecoverable_evaluator_failure"
    );
    assert_eq!(
        str_at(&report, "/search/termination_reason"),
        "terminated_unrecoverable_evaluator_failure"
    );
    assert_eq!(u64_at(&report, "/search/fatal_evaluator_failures"), 1);
    assert!(
        str_at(&report, "/search/fatal_evaluator_failure")
            .contains("deterministic evaluator table missing row"),
        "fatal diagnostic should retain deterministic table row failure context"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn run_tune_reports_compiled_uniform_mh_kernel() {
    let dir = temp_dir("compiled_uniform_mh");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_json(
        &spec_path,
        &tune_spec(
            &dataset_path,
            &output_path,
            &report_path,
            json!({
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1,
            }),
            None,
        ),
    );
    let args = [
        "infotheory",
        "tune",
        path_string(&spec_path).as_str(),
        "--max-evaluations",
        "3",
        "--annealer-kernel-profile",
        "compiled_uniform_metropolis_hastings",
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune");
    let report = read_json(&report_path);
    assert_eq!(
        str_at(&report, "/search/controller/runtime_path"),
        "compiled_uniform_metropolis_hastings"
    );
    assert_eq!(
        str_at(&report, "/search/controller/proposal_action_distribution"),
        "uniform_finite_bounded_numeric_elementary_descriptors"
    );
    assert!(bool_at(
        &report,
        "/search/controller/proposal_mass_accounting"
    ));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn run_tune_warmstart_self_improvement_reports_equal_split_deadlines() {
    let dir = temp_dir("warmstart_self_improvement_deadlines");
    let dataset_path = dir.join("dataset.bin");
    let teacher_path = dir.join("teacher.json");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_placeholder_teacher(&teacher_path);
    let warmstart = controller_cases()
        .into_iter()
        .find(|case| case.kind == "aiqi_warmstart_exact_jh")
        .expect("warmstart case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        warmstart.controller,
        Some(&teacher_path),
    );
    write_json(&spec_path, &spec_value);
    let (_spec_crc32, _) = compiled_tune_hashes(&spec_value, &dir);
    let reward_cert_path = dir.join("exact_reward.json");
    let reward_cert_crc32 = write_exact_reward_certificate(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "aiqi_warmstart_exact_jh",
        65_535,
    );
    let args = [
        "infotheory",
        "tune",
        path_string(&spec_path).as_str(),
        "--max-evaluations",
        "2",
        "--self-improvement-rounds",
        "3",
        "--scalar-representation-ref",
        "scalar://finite-f64",
        "--exact-reward-encoding-certificate",
        path_string(&reward_cert_path).as_str(),
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let task_fingerprint =
        derive_runtime_warmstart_task_fingerprint(&args, &teacher_path, &reward_cert_crc32);
    write_teacher(&teacher_path, &task_fingerprint, &reward_cert_crc32);
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune");
    let report = read_json(&report_path);
    assert!(!bool_at(
        &report,
        "/provenance/self_improvement_policy/same_task_trace_refresh_enabled"
    ));
    assert!(bool_at(
        &report,
        "/provenance/self_improvement_policy/online_delayed_label_update_enabled"
    ));
    assert_eq!(
        u64_at(&report, "/provenance/self_improvement_policy/rounds"),
        3
    );
    let deadlines =
        report["provenance"]["self_improvement_policy"]["deterministic_round_deadlines_seconds"]
            .as_array()
            .expect("deterministic deadline array");
    assert_eq!(deadlines.len(), 3);
    let expected = [5.0f64 / 3.0f64, 10.0f64 / 3.0f64, 5.0f64];
    for (index, expected_value) in expected.iter().enumerate() {
        let observed = deadlines[index]
            .as_f64()
            .unwrap_or_else(|| panic!("deadline[{index}] must be f64"));
        assert!((observed - expected_value).abs() <= 1.0e-9);
    }
    let realized =
        report["provenance"]["self_improvement_policy"]["realized_trace_counts_by_round"]
            .as_array()
            .expect("realized trace counts");
    assert_eq!(realized.len(), 3);
    let merges = report["provenance"]["self_improvement_policy"]["trace_refresh_merges_by_round"]
        .as_array()
        .expect("trace refresh merges");
    assert_eq!(merges.len(), 3);
    assert!(merges.iter().all(|value| value.as_u64() == Some(0)));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn warmstart_trace_refresh_merges_same_task_live_trace() {
    let dir = temp_dir("warmstart_trace_refresh");
    let dataset_path = dir.join("dataset.bin");
    let teacher_path = dir.join("teacher.json");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_placeholder_teacher(&teacher_path);
    let warmstart = controller_cases()
        .into_iter()
        .find(|case| case.kind == "aiqi_warmstart_exact_jh")
        .expect("warmstart case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        warmstart.controller,
        Some(&teacher_path),
    );
    write_json(&spec_path, &spec_value);
    let (_spec_crc32, _) = compiled_tune_hashes(&spec_value, &dir);
    let reward_cert_path = dir.join("exact_reward.json");
    let reward_cert_crc32 = write_exact_reward_certificate(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "aiqi_warmstart_exact_jh",
        65_535,
    );
    write_teacher(&teacher_path, "probe", "probe");
    let args = [
        "infotheory",
        "tune",
        path_string(&spec_path).as_str(),
        "--max-evaluations",
        "3",
        "--self-improvement-rounds",
        "3",
        "--warmstart-trace-refresh",
        "--scalar-representation-ref",
        "scalar://finite-f64",
        "--exact-reward-encoding-certificate",
        path_string(&reward_cert_path).as_str(),
    ]
    .iter()
    .map(|item| (*item).to_string())
    .collect::<Vec<_>>();
    let task_fingerprint =
        derive_runtime_warmstart_task_fingerprint(&args, &teacher_path, &reward_cert_crc32);
    write_teacher(&teacher_path, &task_fingerprint, &reward_cert_crc32);
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune");
    let report = read_json(&report_path);
    assert!(bool_at(
        &report,
        "/provenance/self_improvement_policy/same_task_trace_refresh_enabled"
    ));
    assert!(!bool_at(
        &report,
        "/provenance/self_improvement_policy/online_delayed_label_update_enabled"
    ));
    assert!(u64_at(&report, "/search/controller/warmstart_trace_refresh_merges") >= 1);
    let realized =
        report["provenance"]["self_improvement_policy"]["realized_trace_counts_by_round"]
            .as_array()
            .expect("realized trace counts");
    let merges = report["provenance"]["self_improvement_policy"]["trace_refresh_merges_by_round"]
        .as_array()
        .expect("trace refresh merges");
    assert_eq!(realized.len(), 3);
    assert_eq!(merges.len(), 3);
    assert!(merges.iter().any(|value| value.as_u64().unwrap_or(0) > 0));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn warmstart_teacher_fingerprint_mismatch_is_rejected() {
    let dir = temp_dir("warmstart_teacher_mismatch");
    let dataset_path = dir.join("dataset.bin");
    let teacher_path = dir.join("teacher.json");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_placeholder_teacher(&teacher_path);
    let warmstart = controller_cases()
        .into_iter()
        .find(|case| case.kind == "aiqi_warmstart_exact_jh")
        .expect("warmstart case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        warmstart.controller,
        Some(&teacher_path),
    );
    write_json(&spec_path, &spec_value);
    let reward_cert_path = dir.join("exact_reward.json");
    let reward_cert_crc32 = write_exact_reward_certificate(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "aiqi_warmstart_exact_jh",
        65_535,
    );
    write_teacher(
        &teacher_path,
        "mismatched-task-fingerprint",
        &reward_cert_crc32,
    );
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert_path),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("mismatched teacher must be rejected");
    assert!(err.contains("task_fingerprint"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn warmstart_teacher_observation_adapter_mismatch_is_rejected() {
    let dir = temp_dir("warmstart_teacher_observation_mismatch");
    let dataset_path = dir.join("dataset.bin");
    let teacher_path = dir.join("teacher.json");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_placeholder_teacher(&teacher_path);
    let warmstart = controller_cases()
        .into_iter()
        .find(|case| case.kind == "aiqi_warmstart_exact_jh")
        .expect("warmstart case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        warmstart.controller,
        Some(&teacher_path),
    );
    write_json(&spec_path, &spec_value);
    let (_spec_crc32, _) = compiled_tune_hashes(&spec_value, &dir);
    let reward_cert_path = dir.join("exact_reward.json");
    let reward_cert_crc32 = write_exact_reward_certificate(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::BestEffort,
        "aiqi_warmstart_exact_jh",
        65_535,
    );
    write_teacher(&teacher_path, "probe", "probe");
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert_path),
    ];
    let task_fingerprint =
        derive_runtime_warmstart_task_fingerprint(&args, &teacher_path, &reward_cert_crc32);
    write_teacher(&teacher_path, &task_fingerprint, &reward_cert_crc32);
    let mut teacher = read_json(&teacher_path);
    teacher["contract"]["observation_adapter_content_crc32"] = json!("00000000");
    write_json(&teacher_path, &teacher);
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("observation adapter mismatch must fail");
    assert!(err.contains("observation adapter fingerprint"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn warmstart_exact_jh_rejects_nonidentity_finite_reward_map() {
    let dir = temp_dir("warmstart_nonidentity_reward_map");
    let dataset_path = dir.join("dataset.bin");
    let teacher_path = dir.join("teacher.json");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    write_placeholder_teacher(&teacher_path);
    let warmstart = controller_cases()
        .into_iter()
        .find(|case| case.kind == "aiqi_warmstart_exact_jh")
        .expect("warmstart case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        warmstart.controller,
        Some(&teacher_path),
    );
    write_json(&spec_path, &spec_value);
    let (_spec_crc32, baseline_candidate_crc32) = compiled_tune_hashes(&spec_value, &dir);
    let document = SpecDocument::parse_json_value(&spec_value, &dir).expect("parse tune spec");
    let SpecDocument::Tune(tune) = document else {
        panic!("expected tune spec");
    };
    let compiled = tune
        .compile_in(&SpecEnvironment::new(&dir))
        .expect("compile tune spec");
    let baseline_objective = ((compiled.baseline_candidate_model_bytes() as u64) * 8) + 128;
    let reward_values = (0..=baseline_objective)
        .map(|objective_difference: u64| {
            let symbol = match objective_difference {
                1 => 2,
                2 => 1,
                other => other,
            };
            json!({
                "objective_difference": objective_difference,
                "symbol": symbol
            })
        })
        .collect::<Vec<_>>();
    let reward_cert_path = dir.join("reward_map.json");
    let reward_cert_crc32 = write_finite_reward_map_certificate_with_runtime_profile(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "aiqi_warmstart_exact_jh",
        65_535,
        Some(baseline_objective),
        Value::Array(reward_values),
        true,
    );
    write_teacher(&teacher_path, "placeholder", &reward_cert_crc32);
    let table_path = dir.join("table.json");
    write_deterministic_table(
        &table_path,
        &dataset_path,
        "aiqi_warmstart_exact_jh",
        &baseline_candidate_crc32,
    );
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--timing-tier".to_string(),
        "deterministic_table".to_string(),
        "--scalar-representation-ref".to_string(),
        "scalar://finite-f64".to_string(),
        "--exact-reward-encoding-certificate".to_string(),
        path_string(&reward_cert_path),
        "--deterministic-evaluator-table".to_string(),
        path_string(&table_path),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    let err = run_tune(&request).expect_err("nonidentity finite map must be rejected");
    assert!(err.contains("non-identity finite_reward_map"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn run_tune_planner_controller_timing_matrix_reports_dispatch_and_claim_gating() {
    let dir = temp_dir("controller_timing_matrix");
    let dataset_path = dir.join("dataset.bin");
    let teacher_path = dir.join("teacher.json");
    write_passive_dataset(&dataset_path);
    write_placeholder_teacher(&teacher_path);
    let timings = [
        TimingCertificationTier::BestEffort,
        TimingCertificationTier::Isolated,
        TimingCertificationTier::RealTime,
        TimingCertificationTier::DeterministicTable,
    ];
    for case in controller_cases() {
        for timing in timings {
            let report = run_tune_case(
                &dir,
                "passive",
                &dataset_path,
                &teacher_path,
                &case,
                timing,
                2,
            );
            assert_required_report_fields(&report);
            assert_eq!(str_at(&report, "/status"), case.status);
            assert_eq!(str_at(&report, "/search/controller/kind"), case.kind);
            assert_eq!(
                str_at(&report, "/search/controller/runtime_path"),
                case.runtime_path
            );
            assert_eq!(
                str_at(&report, "/search/controller/agent_runtime"),
                case.agent_runtime
            );
            assert_eq!(
                str_at(&report, "/search/controller/planner_run_controller_kind"),
                case.planner_run_controller_kind
            );
            assert_eq!(
                str_at(&report, "/search/controller/reward_semantics"),
                case.reward_semantics
            );
            assert_eq!(
                u64_at(&report, "/search/controller/compiled_action_count"),
                2
            );
            assert_eq!(
                u64_at(&report, "/search/controller/declared_agent_actions"),
                2
            );
            assert_eq!(
                str_at(&report, "/input_asset/dataset_kind"),
                "passive_bytes"
            );
            assert_eq!(
                str_at(&report, "/evaluator_profile/objective_target"),
                "passive_ac"
            );
            assert_eq!(
                str_at(&report, "/theorem_claims/exact_finite_mdp/status"),
                "uncertified"
            );
            assert_eq!(
                str_at(&report, "/theorem_claims/exact_observed_markov/status"),
                "uncertified"
            );
            assert_eq!(
                str_at(&report, "/theorem_claims/planner_convergence/status"),
                "uncertified"
            );
            assert!(
                report["theorem_claims"]["exact_finite_mdp"]["missing_prerequisites"]
                    .as_array()
                    .expect("missing prereqs")
                    .iter()
                    .any(|item| item.as_str() == Some("verified_finite_planner_state_certificate"))
            );
            assert!(
                report["theorem_claims"]["exact_finite_mdp"]["missing_prerequisites"]
                    .as_array()
                    .expect("missing prereqs")
                    .iter()
                    .any(|item| {
                        item.as_str() == Some("verified_no_hidden_state_or_inert_state_certificate")
                    })
            );
            if case.kind == "aiqi_discounted" {
                assert!(
                    report["theorem_claims"]["exact_finite_mdp"]["missing_prerequisites"]
                        .as_array()
                        .expect("missing prereqs")
                        .iter()
                        .any(|item| item.as_str()
                            == Some("verified_exact_reward_encoding_certificate"))
                );
                assert!(
                    report["theorem_claims"]["exact_finite_mdp"]["missing_prerequisites"]
                        .as_array()
                        .expect("missing prereqs")
                        .iter()
                        .any(|item| {
                            item.as_str() == Some("exact_objective_difference_controller")
                        })
                );
            }
            if !timing_certifies(timing) {
                assert!(
                    report["theorem_claims"]["exact_finite_mdp"]["missing_prerequisites"]
                        .as_array()
                        .expect("missing prereqs")
                        .iter()
                        .any(|item| {
                            item.as_str() == Some("theorem_certified_timing_or_deterministic_table")
                        })
                );
            }
            if case.kind != "mc_aixi_fac_ctw" {
                assert!(
                    report["theorem_claims"]["planner_convergence"]["missing_prerequisites"]
                        .as_array()
                        .expect("missing prereqs")
                        .iter()
                        .any(|item| item.as_str() == Some("mc_aixi_fac_ctw_controller"))
                );
            }
            if case.needs_teacher {
                assert!(
                    report["search"]["controller"]["warmstart_teacher_dataset"]["content_crc32"]
                        .as_str()
                        .is_some()
                );
            } else {
                assert!(report["search"]["controller"]["warmstart_teacher_dataset"].is_null());
            }
        }
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn run_tune_causal_dataset_modes_report_lowering_under_planner_execution() {
    let dir = temp_dir("causal_dataset_modes");
    let teacher_path = dir.join("teacher.json");
    write_placeholder_teacher(&teacher_path);
    let datasets = [
        (
            "interactive",
            canonical_causal_dataset(
                "test-interactive-codec",
                "events",
                json!([
                    {"kind": "context", "channel": "action", "bytes": [1]},
                    {"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [2]},
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [3, 4]}
                ]),
            ),
            "interactive_trace",
            "interactive-trace-events-v1",
            "interactive-trace-target-bytes",
            2u64,
            2u64,
            2.0f64,
        ),
        (
            "prefix",
            canonical_causal_dataset(
                "test-prefix-codec",
                "examples",
                json!([{
                    "history": [{"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [7]}],
                    "action": [1],
                    "channel": "percept",
                    "domain": "bytes",
                    "target": [8],
                    "weight": 2.0
                }]),
            ),
            "causal_prefix_dataset",
            "causal-prefix-examples-v1",
            "weighted-target-bytes-sum",
            1u64,
            1u64,
            2.0f64,
        ),
    ];
    let case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "aiqi_discounted")
        .expect("aiqi-discounted case");
    for (
        label,
        dataset,
        expected_kind,
        expected_lowering,
        expected_size_function,
        expected_charged_bytes,
        expected_target_events,
        expected_dataset_units,
    ) in datasets
    {
        let dataset_path = dir.join(format!("{label}.json"));
        write_json(&dataset_path, &dataset);
        let report = run_tune_case(
            &dir,
            label,
            &dataset_path,
            &teacher_path,
            &case,
            TimingCertificationTier::DeterministicTable,
            2,
        );
        assert_required_report_fields(&report);
        assert_eq!(str_at(&report, "/input_asset/dataset_kind"), expected_kind);
        assert_eq!(
            str_at(&report, "/input_asset/target_size_function"),
            expected_size_function
        );
        assert_eq!(
            str_at(&report, "/evaluator_profile/dataset_lowering_version"),
            expected_lowering
        );
        assert_eq!(
            str_at(&report, "/evaluator_profile/objective_target"),
            "interactive_causal_ac"
        );
        assert_eq!(
            u64_at(&report, "/input_asset/charged_target_bytes"),
            expected_charged_bytes
        );
        assert_eq!(
            u64_at(&report, "/input_asset/target_events"),
            expected_target_events
        );
        assert_eq!(
            f64_at(&report, "/input_asset/dataset_units"),
            expected_dataset_units
        );
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn planner_deployable_model_flag_reports_objective_target_and_diagnostics() {
    let dir = temp_dir("planner_deployable_report");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "aiqi_discounted")
        .expect("aiqi discounted case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        case.controller,
        None,
    );
    write_json(&spec_path, &spec_value);
    let args = [
        "infotheory".to_string(),
        "tune".to_string(),
        path_string(&spec_path),
        "--max-evaluations".to_string(),
        "1".to_string(),
        "--planner-deployable-model".to_string(),
    ];
    let request = parse_tune_command_args(&args).expect("parse tune args");
    run_tune(&request).expect("run tune");
    let report = read_json(&report_path);
    assert_eq!(
        str_at(&report, "/evaluator_profile/objective_target"),
        "planner_deployable_model"
    );
    assert!(bool_at(&report, "/search/planner_deployability/enabled"));
    assert!(bool_at(
        &report,
        "/search/planner_deployability/deployable_under_executor_limits"
    ));
    assert!(u64_at(&report, "/search/planner_deployability/model_state_bytes") > 0);
    let _ = fs::remove_dir_all(dir);
}

#[cfg(feature = "cli")]
#[test]
fn tune_cli_accepts_executor_flags_and_writes_report() {
    let dir = temp_dir("cli_smoke");
    let dataset_path = dir.join("dataset.bin");
    let spec_path = dir.join("spec.json");
    let output_path = dir.join("output.json");
    let report_path = dir.join("report.json");
    write_passive_dataset(&dataset_path);
    let case = controller_cases()
        .into_iter()
        .find(|case| case.kind == "mc_aixi_fac_ctw")
        .expect("mc-aixi case");
    let spec_value = tune_spec(
        &dataset_path,
        &output_path,
        &report_path,
        case.controller,
        None,
    );
    write_json(&spec_path, &spec_value);
    let reward_cert_path = dir.join("reward.json");
    write_exact_reward_certificate_with_runtime_profile_and_worker(
        &reward_cert_path,
        &dataset_path,
        TimingCertificationTier::DeterministicTable,
        "mc_aixi_fac_ctw",
        65_535,
        false,
        Some(Path::new(env!("CARGO_BIN_EXE_infotheory"))),
    );
    let spec_arg = path_string(&spec_path);
    let reward_cert_arg = path_string(&reward_cert_path);
    let output = Command::new(env!("CARGO_BIN_EXE_infotheory"))
        .args([
            "tune",
            spec_arg.as_str(),
            "--max-evaluations",
            "1",
            "--timing-tier",
            "deterministic_table",
            "--scalar-representation-ref",
            "scalar://finite-f64",
            "--exact-reward-encoding-certificate",
            reward_cert_arg.as_str(),
            "--claim-exact-finite-mdp",
        ])
        .output()
        .expect("run tune cli");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = read_json(&report_path);
    assert_eq!(u64_at(&report, "/execution_profile/max_evaluations"), 1);
    assert_eq!(
        str_at(&report, "/provenance/executor_controls/rss_mode/requested"),
        "process_rss_peak"
    );
    let effective_measurement = str_at(
        &report,
        "/provenance/executor_controls/rss_mode/effective_measurement",
    );
    assert_eq!(effective_measurement, "unix_process_rss_fallback_explicit");
    assert_eq!(
        str_at(&report, "/theorem_claims/exact_finite_mdp/status"),
        "uncertified"
    );
    assert_eq!(
        str_at(&report, "/evaluator_execution_model"),
        "spawn_exec_worker_process_isolated_operational"
    );
    assert_eq!(
        str_at(&report, "/theorem_timing_basis"),
        "operational_only_uncertified"
    );
    assert!(output_path.exists());
    let _ = fs::remove_dir_all(dir);
}
