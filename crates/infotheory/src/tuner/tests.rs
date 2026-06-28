use super::*;
#[cfg(feature = "backend-ctw")]
use crate::aixi::common::{ActionAlphabet, ObservationKeyMode};
use crate::aixi::warmstart::{
    WarmStartExactJhTeacherContract, WarmStartExactJhTeacherDataset, WarmStartExactJhTeacherTrace,
    WarmStartExactJhTransition,
};
use crate::aixi::warmstart_contract::TaskFingerprint;
use crate::api::CompressionBackend;
#[cfg(feature = "backend-ctw")]
use crate::api::RateBackend;
#[cfg(feature = "backend-ctw")]
use crate::compression::FramingMode;
#[cfg(feature = "backend-ctw")]
use crate::spec::{
    AiqiDiscountedTuneControllerSpec, AnnealedHillClimbingTuneControllerSpec, AssetBinding,
    McAixiFacCtwTuneControllerSpec, SpecDocument, TuneBoundsSpec, TuneControllerSpec,
    TunePlannerInterfaceSpec, TuneSpec, WarmStartExactJhTuneControllerSpec,
};
use crate::tuner::eval::ResolvedMemoryAccountingKind;
#[cfg(all(feature = "backend-ctw", feature = "backend-mixture"))]
use crate::tuner::planner_bridge::apply_planner_mutation_action;
#[cfg(feature = "backend-ctw")]
use std::time::{SystemTime, UNIX_EPOCH};

fn strict_mode_test_accounting_kind() -> ResolvedMemoryAccountingKind {
    #[cfg(target_os = "linux")]
    {
        ResolvedMemoryAccountingKind::StrictLinuxCgroupV2PeakMaxProcessRss
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        // Non-Linux targets cannot resolve strict cgroup-v2 accounting.
        ResolvedMemoryAccountingKind::UnixProcessRssFallbackExplicit
    }
    #[cfg(not(unix))]
    {
        ResolvedMemoryAccountingKind::DeterministicEvaluatorTable
    }
}

#[cfg(feature = "backend-ctw")]
fn temp_path(prefix: &str, suffix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("infotheory-tuner-{prefix}-{nanos}{suffix}"))
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

#[cfg(feature = "backend-ctw")]
fn sample_tune_spec(dataset_path: &str, output_path: &str, report_path: &str) -> TuneSpec {
    TuneSpec {
        assets: vec![AssetBinding {
            id: "dataset".to_string(),
            path: dataset_path.to_string(),
        }],
        input_asset: "dataset".to_string(),
        baseline_candidate: CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        },
        controller: TuneControllerSpec::AnnealedHillClimbing(
            AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: 1,
            },
        ),
        bounds: TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string()],
            forbidden_backends: Vec::new(),
            parameter_ranges: Vec::new(),
            max_experts: 2,
            max_mixture_nesting_depth: 1,
            min_experts: Some(1),
            allow_duplicate_experts: Some(false),
            required_experts: Vec::new(),
            forbidden_expert_pairs: Vec::new(),
        },
        eval_time_limit_seconds: 1.0,
        time_budget_seconds: 2.0,
        min_throughput_bytes_per_second: 1.0,
        max_memory_bytes: u64::MAX,
        output_config_path: output_path.to_string(),
        seed: 7,
        report_path: Some(report_path.to_string()),
    }
}

#[cfg(feature = "backend-ctw")]
fn action_alphabet(n: usize) -> ActionAlphabet {
    ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
}

#[cfg(feature = "backend-ctw")]
fn causal_dataset_value(codec_hash: &str, payload_key: &str, payload: Value) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("schema_version".to_string(), serde_json::json!(1));
    object.insert("environment_id".to_string(), serde_json::json!("test-env"));
    object.insert(
        "environment_config_crc32".to_string(),
        serde_json::json!("00000000"),
    );
    object.insert("codec_hash".to_string(), serde_json::json!(codec_hash));
    object.insert(
        "reset_convention".to_string(),
        serde_json::json!("reset-before-episode"),
    );
    object.insert(
        "action_alphabet".to_string(),
        serde_json::json!({"size": 2}),
    );
    object.insert(
        "percept_schema".to_string(),
        serde_json::json!({
            "encoding": "bytes",
            "channels": [{"channel": "percept", "domain": "bytes"}],
        }),
    );
    object.insert(
        "reward_encoding".to_string(),
        serde_json::json!({
            "encoding": "bytes",
            "channel": "reward",
            "domain": "binary",
        }),
    );
    object.insert(
        "terminal_encoding".to_string(),
        serde_json::json!({
            "encoding": "bytes",
            "channel": "terminal",
            "domain": "binary",
        }),
    );
    object.insert("collection_policy".to_string(), serde_json::json!("test"));
    object.insert(
        "target_domains".to_string(),
        serde_json::json!({
            "bytes": {"kind": "byte_alphabet"},
            "binary": {"kind": "enumerated_payloads", "payloads": [[0], [1]]}
        }),
    );
    object.insert(
        "event_grammar".to_string(),
        serde_json::json!({
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

#[cfg(feature = "backend-ctw")]
fn planner_interface_for_baseline(candidate: &CompressionBackend) -> TunePlannerInterfaceSpec {
    let json = crate::spec::compression_backend_to_json_value(candidate)
        .expect("baseline candidate must serialize");
    let actions = (collect_numeric_leaves(&json).len() * 2).max(1);
    TunePlannerInterfaceSpec {
        observation_bits: 8,
        observation_stream_len: 1,
        observation_key_mode: ObservationKeyMode::FullStream,
        reward_bits: 16,
        agent_actions: action_alphabet(actions),
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn synthesized_planner_bridge_inherits_tune_spec_environment() {
    let base_dir = temp_path("bridge-base", "");
    std::fs::create_dir_all(&base_dir).expect("create base dir");
    let output_path = base_dir.join("best.json");
    let report_path = base_dir.join("report.json");
    let mut spec = sample_tune_spec(
        "relative-dataset.bin",
        &output_path.to_string_lossy(),
        &report_path.to_string_lossy(),
    );
    spec.controller = TuneControllerSpec::AiqiDiscounted(AiqiDiscountedTuneControllerSpec {
        interface: TunePlannerInterfaceSpec {
            observation_bits: 8,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 8,
            agent_actions: action_alphabet(1),
        },
        planner_simulations_per_step: 1,
        return_horizon: 1,
        return_bins: 2,
        discount_factor: 0.5,
        min_improvement: 0.0,
        max_improvement: 1.0,
    });
    let env = SpecEnvironment::new(&base_dir);
    let compiled = spec.compile_in(&env).expect("compile tune spec");
    assert_eq!(compiled.base_dir(), base_dir.as_path());

    let dataset = LoadedDataset {
        kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        codec_hash: "passive-identity-bytes".to_string(),
        event_grammar_hash: "passive-target-only-byte-stream".to_string(),
        target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
        causal_header_profile_hash: crc32_hex(b"passive-none"),
        target_size_function: "passive-bytes-len",
        canonical_content_hash: crc32_hex(b"dataset"),
        lowered_skeleton_hash: crc32_hex(b"passive-bytes-target-only"),
        resolved_path: base_dir
            .join("relative-dataset.bin")
            .to_string_lossy()
            .to_string(),
        source_size_bytes: 7,
        raw_bytes: b"dataset".to_vec(),
        events: Vec::new(),
        causal_profile: None,
        dataset_units: 7.0,
        target_events: 1,
    };
    let verified = VerifiedTheoremInputs::default();
    let contract =
        planner_controller_contract(compiled.controller(), &compiled, &dataset, &verified)
            .expect("planner contract");
    let reward_encoder = contract
        .reward_encoder(
            &dataset,
            &CandidateEvalResult {
                status: CandidateEvalStatus::Success,
                compressed_bytes: 7,
                elapsed_seconds: 0.1,
                effective_eval_time_limit_seconds: 1.0,
                throughput_bytes_per_second: 70.0,
                peak_memory_bytes: 0,
                target_loss_bits: 7.0,
                objective_bits: 7.0,
                deployable: true,
            },
            &verified,
        )
        .expect("reward encoder");
    let planner_run = compile_tuner_planner_run_spec(
        compiled.controller(),
        &contract,
        &reward_encoder,
        &compiled,
        &env,
    )
    .expect("compile bridge");
    let binding = planner_run
        .resolved_assets()
        .iter()
        .find(|binding| binding.id == "dataset")
        .expect("dataset asset binding");
    let crate::spec::AssetRef::Filesystem(path) = &binding.asset;
    assert_eq!(path, &base_dir.join("relative-dataset.bin"));

    let _ = std::fs::remove_dir_all(base_dir);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn tuner_warmstart_task_fingerprint_binds_input_dataset_content_not_teacher_bytes() {
    let base_dir = temp_path("warmstart-fingerprint", "");
    std::fs::create_dir_all(&base_dir).expect("create base dir");
    let dataset_path = base_dir.join("dataset.bin");
    let teacher_path = base_dir.join("teacher.json");
    let teacher_alt_path = base_dir.join("teacher-alt.json");
    std::fs::write(&dataset_path, b"dataset-v1").expect("write dataset v1");
    std::fs::write(&teacher_path, b"teacher-v1").expect("write teacher v1");
    std::fs::write(&teacher_alt_path, b"teacher-v1-alt-path").expect("write alternate teacher");

    let output_path = base_dir.join("best.json");
    let report_path = base_dir.join("report.json");
    let mut spec = sample_tune_spec(
        "dataset.bin",
        &output_path.to_string_lossy(),
        &report_path.to_string_lossy(),
    );
    spec.assets.push(AssetBinding {
        id: "teacher".to_string(),
        path: "teacher.json".to_string(),
    });
    let interface = TunePlannerInterfaceSpec {
        observation_bits: 8,
        observation_stream_len: 1,
        observation_key_mode: ObservationKeyMode::FullStream,
        reward_bits: 8,
        agent_actions: action_alphabet(1),
    };
    spec.controller =
        TuneControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhTuneControllerSpec {
            interface: interface.clone(),
            planner_simulations_per_step: 1,
            return_horizon: 1,
            warmstart_teacher_dataset_asset: "teacher".to_string(),
            label_phase_period: 1,
        });

    let env = SpecEnvironment::new(&base_dir);
    let compiled = spec.compile_in(&env).expect("compile tune spec");
    let contract = PlannerControllerContract {
        interface,
        planner_simulations_per_step: 1,
        return_horizon: Some(1),
        label_phase_period: Some(1),
        discount_factor: 1.0,
        reward_semantics: PlannerRewardSemantics::ExactObjectiveDifference,
        clipping_interval: None,
        teacher: None,
        warmstart_self_improvement: true,
    };
    let reward_encoder = TunerRewardEncoder::ExactIntegerObjectiveDifference {
        max_reward: 3,
        objective_difference_to_symbol: None,
    };
    let planner_run = compile_tuner_planner_run_spec(
        compiled.controller(),
        &contract,
        &reward_encoder,
        &compiled,
        &env,
    )
    .expect("compile warmstart bridge");
    let fingerprint_v1 =
        crate::aixi::warmstart_contract::warmstart_exact_jh_planner_task_fingerprint(&planner_run)
            .expect("fingerprint v1");

    std::fs::write(&teacher_path, b"teacher-v2").expect("write teacher v2");
    let fingerprint_after_teacher_change =
        crate::aixi::warmstart_contract::warmstart_exact_jh_planner_task_fingerprint(&planner_run)
            .expect("fingerprint after teacher change");
    assert_eq!(
        fingerprint_v1, fingerprint_after_teacher_change,
        "teacher dataset bytes are intentionally excluded to avoid a circular task fingerprint"
    );

    let mut moved_teacher_spec = spec.clone();
    if let Some(binding) = moved_teacher_spec
        .assets
        .iter_mut()
        .find(|binding| binding.id == "teacher")
    {
        binding.path = "teacher-alt.json".to_string();
    }
    let moved_teacher_compiled = moved_teacher_spec
        .compile_in(&env)
        .expect("compile moved-teacher tune spec");
    let moved_teacher_planner_run = compile_tuner_planner_run_spec(
        moved_teacher_compiled.controller(),
        &contract,
        &reward_encoder,
        &moved_teacher_compiled,
        &env,
    )
    .expect("compile moved-teacher warmstart bridge");
    let fingerprint_after_teacher_path_change =
        crate::aixi::warmstart_contract::warmstart_exact_jh_planner_task_fingerprint(
            &moved_teacher_planner_run,
        )
        .expect("fingerprint after teacher path change");
    assert_eq!(
        fingerprint_v1, fingerprint_after_teacher_path_change,
        "teacher dataset asset path is intentionally excluded from same-task identity"
    );

    std::fs::write(&dataset_path, b"dataset-v2").expect("write dataset v2");
    let fingerprint_v2 =
        crate::aixi::warmstart_contract::warmstart_exact_jh_planner_task_fingerprint(&planner_run)
            .expect("fingerprint v2");
    assert_ne!(
        fingerprint_v1, fingerprint_v2,
        "same-path input dataset byte changes must invalidate same-task warm-start teachers"
    );

    let _ = std::fs::remove_dir_all(base_dir);
}

#[cfg(feature = "backend-ctw")]
fn write_test_exact_reward_certificate(
    path: &std::path::Path,
    dataset_path: &std::path::Path,
    bounds: &TuneBoundsSpec,
    controller_kind: &str,
) {
    let dataset = load_dataset(dataset_path).expect("load dataset for certificate");
    let execution = TuneExecutionConfig::default();
    let runtime_profile = resolve_evaluator_runtime_profile(&execution, false)
        .expect("resolve evaluator runtime profile for certificate");
    let evaluator_profile = EvaluatorProfile {
        dataset_kind: dataset.kind,
        objective_target: dataset.objective_target,
        dataset_lowering_version: dataset.lowering_version,
        dataset_codec_hash: dataset.codec_hash.clone(),
        event_grammar_hash: dataset.event_grammar_hash.clone(),
        target_domain_support_hash: dataset.target_domain_support_hash.clone(),
        causal_header_profile_hash: dataset.causal_header_profile_hash.clone(),
        target_size_function: dataset.target_size_function,
        evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
        candidate_canonicalization_version: "bounds-v1".to_string(),
        warmup_baseline_runs: 0,
        diagnostic_chunk_bytes: None,
        eval_time_limit_seconds: 1.0,
        evaluator_threads: execution.evaluator_threads(),
        worker_isolation_mode: "spawn_exec_worker",
        worker_executable_identity: runtime_profile.worker_executable_identity.clone(),
        resolved_memory_accounting_kind: runtime_profile.memory_accounting_kind.name(),
        resolved_memory_accounting_strict_theorem_facing: runtime_profile
            .strict_theorem_memory_certified(),
        resolved_evaluator_cgroup_parent: runtime_profile.resolved_cgroup_parent_string(),
        backend_report_component_policy: runtime_profile
            .memory_accounting_kind
            .backend_report_component_policy(),
        evaluator_determinism: execution.evaluator_determinism(),
        rss_mode: execution.rss_mode,
        timing_certification_tier: TimingCertificationTier::BestEffort,
        build_profile: option_env!("PROFILE").unwrap_or("unknown"),
        feature_set: compiled_feature_set(),
    };
    let reward_cert = serde_json::json!({
        "schema_version": 1,
        "kind": "exact_reward_encoding",
        "dataset_crc32": dataset.canonical_content_hash,
        "bounds_crc32": bounds_hash(bounds).expect("bounds hash"),
        "evaluator_profile_crc32": evaluator_profile.hash().expect("profile hash"),
        "controller_kind": controller_kind,
        "action_alphabet_size": 2,
        "encoding": "integer_objective_difference",
        "scalar_representation": SCALAR_REPRESENTATION_DECLARATION,
        "reward_bits": 16,
        "max_reward": 65_535u64,
    });
    std::fs::write(
        path,
        serde_json::to_vec(&reward_cert).expect("reward cert json"),
    )
    .expect("write reward cert");
}

#[test]
fn parse_tune_cli_args_and_theorem_flags() {
    let args = vec![
        "infotheory".to_string(),
        "tune".to_string(),
        "spec.json".to_string(),
        "--max-evaluations".to_string(),
        "12".to_string(),
        "--timing-tier".to_string(),
        "real_time".to_string(),
        "--claim-exact-finite-mdp".to_string(),
        "--evaluator-worker-executable".to_string(),
        "/tmp/infotheory-worker".to_string(),
        "--evaluator-cgroup-parent".to_string(),
        "/sys/fs/cgroup/infotheory-tuner".to_string(),
        "--emit-exact-reward-encoding-certificate".to_string(),
        "emit-reward-cert.json".to_string(),
    ];
    let parsed = parse_tune_command_args(&args).expect("parse tune args");
    assert_eq!(parsed.spec_path, "spec.json");
    assert_eq!(parsed.execution.max_evaluations, Some(12));
    assert_eq!(
        parsed.execution.theorem.timing_certification_tier,
        TimingCertificationTier::RealTime
    );
    assert!(parsed.execution.theorem.claim_exact_finite_mdp);
    assert_eq!(
        parsed.execution.evaluator_worker_executable.as_deref(),
        Some("/tmp/infotheory-worker")
    );
    assert_eq!(
        parsed.execution.evaluator_cgroup_parent.as_deref(),
        Some("/sys/fs/cgroup/infotheory-tuner")
    );
    assert_eq!(
        parsed.emit_exact_reward_encoding_certificate.as_deref(),
        Some("emit-reward-cert.json")
    );
}

#[test]
fn tune_execution_config_accepts_nested_theorem_json() {
    let value = serde_json::json!({
        "warmup_baseline_runs": 2,
        "planner_deployable_model": true,
        "theorem": {
            "claim_exact_observed_markov": true,
            "timing_certification_tier": "isolated"
        }
    });
    let cfg = TuneExecutionConfig::from_json_value(&value).expect("config parse");
    assert_eq!(cfg.warmup_baseline_runs, 2);
    assert!(cfg.planner_deployable_model);
    assert!(cfg.theorem.claim_exact_observed_markov);
    assert_eq!(
        cfg.theorem.timing_certification_tier,
        TimingCertificationTier::Isolated
    );
}

#[test]
fn tune_execution_config_reports_executor_profile_semantics() {
    let value = serde_json::json!({
        "max_evaluations": 7,
        "annealer_kernel_profile": "compiled_uniform_metropolis_hastings",
        "cpu_affinity": "0-1",
        "threads": 2,
        "evaluator_worker_executable": "/tmp/infotheory-worker",
        "evaluator_cgroup_parent": "/sys/fs/cgroup/infotheory-tuner",
        "warmup_baseline_runs": 1,
        "self_improvement_rounds": 3,
        "stagnation_reset_evals": 5,
        "log_path": "tune.log",
        "diagnostic_chunk_bytes": 4096,
        "rss_mode": "hybrid_strict_max",
        "planner_deployable_model": true,
        "warmstart_trace_refresh": true,
        "theorem": {
            "claim_exact_finite_mdp": true,
            "claim_exact_observed_markov": true,
            "claim_planner_convergence": true,
            "timing_certification_tier": "deterministic_table",
            "determinism_deadline_certificate": "cert://deadline",
            "observation_adapter_spec_ref": "adapter://single-channel",
            "exact_state_encoder_spec_ref": "state://encoder",
            "scalar_representation_ref": "scalar://finite-f64",
            "finite_planner_state_certificate": "cert://finite-state",
            "no_hidden_state_certificate": "cert://no-hidden",
            "exact_reward_encoding_certificate": "cert://reward",
            "exact_state_observation_certificate": "cert://observation",
            "deterministic_evaluator_table": "table://deterministic"
        }
    });
    let cfg = TuneExecutionConfig::from_json_value(&value).expect("config parse");

    assert_eq!(cfg.max_evaluations, Some(7));
    assert_eq!(
        cfg.annealer_kernel_profile,
        AnnealerKernelProfile::CompiledUniformMetropolisHastings
    );
    assert_eq!(cfg.evaluator_threads(), 2);
    assert_eq!(
        cfg.evaluator_determinism(),
        "requires_backend_determinism_when_threaded"
    );
    assert!(cfg.warmstart_trace_refresh);
    assert_eq!(
        cfg.theorem.timing_certification_tier,
        TimingCertificationTier::DeterministicTable
    );

    let profile = cfg.to_json_value();
    assert_eq!(profile["max_evaluations"], serde_json::json!(7));
    assert_eq!(
        profile["annealer_kernel_profile"],
        serde_json::json!("compiled_uniform_metropolis_hastings")
    );
    assert_eq!(
        profile["evaluator_determinism"],
        serde_json::json!("requires_backend_determinism_when_threaded")
    );
    assert_eq!(
        profile["evaluator_worker_executable"],
        serde_json::json!("/tmp/infotheory-worker")
    );
    assert_eq!(
        profile["evaluator_cgroup_parent"],
        serde_json::json!("/sys/fs/cgroup/infotheory-tuner")
    );
    assert_eq!(
        profile["theorem"]["deterministic_evaluator_table"],
        serde_json::json!("table://deterministic")
    );

    let controls = executor_controls_report(
        &cfg,
        &ResolvedEvaluatorRuntimeProfile {
            worker_executable: Some(std::path::PathBuf::from("/tmp/infotheory-worker")),
            worker_executable_identity: Some("crc32:00000000:bytes:0".to_string()),
            resolved_evaluator_cgroup_parent: Some(std::path::PathBuf::from(
                "/sys/fs/cgroup/infotheory-tuner",
            )),
            memory_accounting_kind: strict_mode_test_accounting_kind(),
        },
    );
    assert_eq!(
        controls["cpu_affinity"]["requested"],
        serde_json::json!("0-1")
    );
    assert_eq!(
        controls["threads"]["worker_isolation_mode"],
        serde_json::json!("spawn_exec_worker")
    );
    assert_eq!(
        controls["rss_mode"]["requested"],
        serde_json::json!("hybrid_strict_max")
    );
}

#[test]
fn tune_execution_config_rejects_empty_certificate_references() {
    for field in [
        "determinism_deadline_certificate",
        "observation_adapter_spec_ref",
        "exact_state_encoder_spec_ref",
        "scalar_representation_ref",
        "finite_planner_state_certificate",
        "no_hidden_state_certificate",
        "exact_reward_encoding_certificate",
        "exact_state_observation_certificate",
        "deterministic_evaluator_table",
    ] {
        let mut theorem = serde_json::Map::<String, Value>::new();
        theorem.insert(field.to_string(), serde_json::json!("   "));
        let value = serde_json::json!({
            "theorem": theorem
        });
        let err = TuneExecutionConfig::from_json_value(&value)
            .expect_err("empty theorem reference must be rejected");
        assert!(err.contains("must be a non-empty string"), "{field}: {err}");
    }
}

#[test]
fn tune_execution_config_rejects_empty_worker_executable() {
    let value = serde_json::json!({
        "evaluator_worker_executable": "   "
    });
    let err = TuneExecutionConfig::from_json_value(&value)
        .expect_err("empty evaluator_worker_executable must be rejected");
    assert!(err.contains("evaluator_worker_executable"), "{err}");
}

#[test]
fn tune_execution_config_rejects_empty_cgroup_parent() {
    let value = serde_json::json!({
        "evaluator_cgroup_parent": "   "
    });
    let err = TuneExecutionConfig::from_json_value(&value)
        .expect_err("empty evaluator_cgroup_parent must be rejected");
    assert!(err.contains("evaluator_cgroup_parent"), "{err}");
}

#[test]
fn run_tune_rejects_invalid_execution_config_direct_call() {
    let request = TuneCommandRequest {
        spec_path: "nonexistent-spec.json".to_string(),
        emit_exact_reward_encoding_certificate: None,
        execution: TuneExecutionConfig {
            threads: Some(0),
            ..TuneExecutionConfig::default()
        },
    };
    let err = run_tune(&request).expect_err("invalid execution config must fail at run_tune");
    assert!(err.contains("threads must be >= 1 when set"), "{err}");
}

#[cfg(feature = "backend-ctw")]
#[test]
fn tune_planner_interface_requires_explicit_observation_key_mode() {
    let mut value = SpecDocument::Tune(sample_tune_spec("dataset.bin", "out.json", "report.json"))
        .to_canonical_json_value()
        .expect("canonical tune json");
    value["controller"] = serde_json::json!({
        "kind": "mc_aixi_fac_ctw",
        "interface": {
            "observation_bits": 8,
            "observation_stream_len": 1,
            "reward_bits": 8,
            "agent_actions": 1
        },
        "planner_simulations_per_step": 1
    });
    let err = match SpecDocument::parse_json_value(&value, Path::new(".")) {
        Ok(_) => panic!("missing tune observation_key_mode must fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("controller.interface.observation_key_mode is required"),
        "{err}"
    );
}

#[test]
fn tune_execution_config_rejects_observation_certified_boolean() {
    let value = serde_json::json!({
        "theorem": {
            "exact_state_observation_certified": true
        }
    });
    let err = TuneExecutionConfig::from_json_value(&value)
        .expect_err("unchecked observation proof boolean must be rejected");
    assert!(err.contains("unknown execution config field"), "{err}");
}

fn passive_loaded_dataset(raw_bytes: Vec<u8>) -> LoadedDataset {
    LoadedDataset {
        kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        codec_hash: "codec".to_string(),
        event_grammar_hash: "none".to_string(),
        target_domain_support_hash: "none".to_string(),
        causal_header_profile_hash: "none".to_string(),
        target_size_function: "bytes",
        canonical_content_hash: "content".to_string(),
        lowered_skeleton_hash: "skeleton".to_string(),
        resolved_path: "dataset.bin".to_string(),
        source_size_bytes: raw_bytes.len(),
        dataset_units: raw_bytes.len() as f64,
        raw_bytes,
        events: Vec::new(),
        causal_profile: None,
        target_events: 0,
    }
}

#[test]
fn diagnostic_chunking_report_preserves_executor_only_contract() {
    let dataset = passive_loaded_dataset((0_u8..10).collect::<Vec<u8>>());

    let disabled = diagnostic_chunking_report(&dataset, None);
    assert_eq!(disabled["enabled"], serde_json::json!(false));
    assert_eq!(
        disabled["affects_canonical_candidate_identity"],
        serde_json::json!(false)
    );
    assert_eq!(disabled["chunk_count"], serde_json::json!(0));

    let enabled = diagnostic_chunking_report(&dataset, Some(4));
    assert_eq!(enabled["enabled"], serde_json::json!(true));
    assert_eq!(enabled["charged_payload_bytes"], serde_json::json!(10));
    assert_eq!(enabled["chunk_count"], serde_json::json!(3));
    assert_eq!(enabled["last_chunk_bytes"], serde_json::json!(2));
    assert_eq!(enabled["affects_objective"], serde_json::json!(false));
    assert_eq!(
        enabled["affects_canonical_candidate_identity"],
        serde_json::json!(false)
    );
}

#[test]
fn causal_profile_report_describes_domains_and_event_grammar() {
    let percept_channel = CausalChannelDomain {
        channel: "obs".to_string(),
        domain: "byte".to_string(),
    };
    let reward_channel = CausalChannelDomain {
        channel: "reward".to_string(),
        domain: "reward_symbols".to_string(),
    };
    let terminal_channel = CausalChannelDomain {
        channel: "terminal".to_string(),
        domain: "terminal_symbols".to_string(),
    };
    let mut domains = BTreeMap::<String, CausalTargetDomain>::new();
    domains.insert("byte".to_string(), CausalTargetDomain::ByteAlphabet);
    domains.insert(
        "reward_symbols".to_string(),
        CausalTargetDomain::EnumeratedPayloads {
            payloads: vec![vec![0], vec![1]],
        },
    );
    let mut channel_set = BTreeSet::<String>::new();
    channel_set.insert("obs".to_string());
    channel_set.insert("reward".to_string());
    channel_set.insert("terminal".to_string());
    let mut percept_channels = BTreeSet::<CausalChannelDomain>::new();
    percept_channels.insert(percept_channel.clone());
    let mut context_channels = BTreeSet::<String>::new();
    context_channels.insert("context".to_string());
    let mut observe_target_no_score = BTreeSet::<CausalChannelDomain>::new();
    observe_target_no_score.insert(percept_channel.clone());
    let mut target = BTreeSet::<CausalChannelDomain>::new();
    target.insert(reward_channel.clone());
    let profile = CausalEvaluationProfile {
        domains,
        channel_set,
        domain_support_hash: "domain-crc".to_string(),
        byte_alphabet_symbol_width: 1,
        header_profile_hash: "header-crc".to_string(),
        event_grammar: CausalEventGrammar {
            context_channels,
            observe_target_no_score,
            target,
        },
        action_alphabet_size: 3,
        collection_policy: "test-policy".to_string(),
        percept_channels,
        reward_channel,
        terminal_channel,
    };
    let mut dataset = passive_loaded_dataset(Vec::new());
    dataset.kind = DatasetKind::CausalPrefixDataset;
    dataset.causal_profile = Some(profile);

    let report = causal_profile_report(&dataset);
    assert_eq!(
        report["domain_support_crc32"],
        serde_json::json!("domain-crc")
    );
    assert_eq!(
        report["header_profile_crc32"],
        serde_json::json!("header-crc")
    );
    assert_eq!(report["action_alphabet_size"], serde_json::json!(3));
    assert_eq!(
        report["byte_alphabet_expansion_policy"],
        serde_json::json!("multi_byte_targets_expand_to_single_byte_events")
    );
    assert_eq!(
        report["reward_encoding"],
        serde_json::json!({"channel": "reward", "domain": "reward_symbols"})
    );
    assert_eq!(report["domains"].as_array().expect("domains").len(), 2);
}

#[test]
fn theorem_timing_and_evaluator_execution_models_report_verified_basis() {
    let deterministic_table = VerifiedDeterministicEvaluatorTable {
        base: VerifiedCertificate {
            ref_value: "table://deterministic".to_string(),
            content_hash: "table-crc".to_string(),
        },
        rows: HashMap::new(),
    };
    let verified_table = VerifiedTheoremInputs {
        deterministic_table: Some(deterministic_table),
        ..VerifiedTheoremInputs::default()
    };
    let table_theorem = TuneTheoremConfig {
        timing_certification_tier: TimingCertificationTier::DeterministicTable,
        ..TuneTheoremConfig::default()
    };
    assert_eq!(
        evaluator_execution_model(verified_table.deterministic_table.as_ref()),
        "deterministic_table"
    );
    assert_eq!(
        theorem_timing_basis(&table_theorem, &verified_table),
        "verified_deterministic_evaluator_table"
    );

    let verified_deadline = VerifiedTheoremInputs {
        determinism_deadline: Some(VerifiedCertificate {
            ref_value: "deadline://cert".to_string(),
            content_hash: "deadline-crc".to_string(),
        }),
        ..VerifiedTheoremInputs::default()
    };
    let real_time_theorem = TuneTheoremConfig {
        timing_certification_tier: TimingCertificationTier::RealTime,
        ..TuneTheoremConfig::default()
    };
    assert_eq!(
        theorem_timing_basis(&real_time_theorem, &verified_deadline),
        "verified_real_time_deadline_certificate"
    );
    assert_eq!(
        theorem_timing_basis(&real_time_theorem, &VerifiedTheoremInputs::default()),
        "operational_only_uncertified"
    );

    let deployability = planner_deployability_report(true, 128, -1.0, false);
    assert_eq!(
        deployability["update_latency_seconds"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        deployability["deployable_under_executor_limits"],
        serde_json::json!(false)
    );
}

#[test]
fn finite_reward_map_accepts_non_contiguous_injective_symbols() {
    let value = serde_json::json!({
        "values": [
            {"objective_difference": 0, "symbol": 0},
            {"objective_difference": 3, "symbol": 7}
        ]
    });
    let map = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
        .expect("finite reward map");
    assert_eq!(map.objective_difference_to_symbol.get(&0), Some(&0));
    assert_eq!(map.objective_difference_to_symbol.get(&3), Some(&7));
    assert_eq!(map.complete_nonnegative_interval_max, None);
}

#[test]
fn finite_reward_map_rejects_reachable_rewards_alias() {
    let value = serde_json::json!({
        "reachable_rewards": [
            {"objective_difference": 0, "symbol": 0},
            {"objective_difference": 1, "symbol": 1}
        ]
    });
    let err = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
        .expect_err("reachable_rewards alias must be rejected");
    assert!(err.contains("requires a 'values' array"), "{err}");
}

#[test]
fn finite_reward_map_rejects_duplicate_symbols() {
    let value = serde_json::json!({
        "values": [
            {"objective_difference": 0, "symbol": 1},
            {"objective_difference": 2, "symbol": 1}
        ]
    });
    let err = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
        .expect_err("duplicate symbol must fail");
    assert!(err.contains("duplicates reward symbol"), "{err}");
}

#[test]
fn finite_reward_map_rejects_incomplete_declared_interval() {
    let value = serde_json::json!({
        "complete_nonnegative_interval_max": 3,
        "values": [
            {"objective_difference": 0, "symbol": 0},
            {"objective_difference": 1, "symbol": 1},
            {"objective_difference": 3, "symbol": 3}
        ]
    });
    let err = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
        .expect_err("declared complete interval must contain every difference");
    assert!(err.contains("missing objective_difference 2"), "{err}");
}

#[test]
fn exact_finite_reward_map_encodes_objective_difference_not_symbol_arithmetic() {
    let encoder = TunerRewardEncoder::ExactIntegerObjectiveDifference {
        max_reward: 2,
        objective_difference_to_symbol: Some(BTreeMap::from([(0, 0), (1, 2), (2, 1)])),
    };
    assert_eq!(encoder.encode(1.0).expect("mapped reward"), 2);
    assert_eq!(encoder.encode(2.0).expect("mapped reward"), 1);
}

#[cfg(all(feature = "backend-ctw", target_os = "linux"))]
#[test]
fn cgroup_peak_reader_parses_fixture_file() {
    let path = temp_path("cgroup-memory-peak", ".txt");
    fs::write(&path, b"12345\n").expect("write cgroup fixture");
    assert_eq!(read_u64_from_file(&path).expect("parse cgroup peak"), 12345);
    fs::write(&path, b"max\n").expect("write cgroup sentinel fixture");
    let err = read_u64_from_file(&path).expect_err("max sentinel is not a measurement");
    assert!(err.contains("unbounded sentinel"), "{err}");
    let _ = fs::remove_file(path);
}

#[test]
fn executor_controls_report_reflects_requested_rss_mode() {
    let config = TuneExecutionConfig {
        rss_mode: PeakMemoryMode::HybridStrictMax,
        ..TuneExecutionConfig::default()
    };
    let report = executor_controls_report(
        &config,
        &ResolvedEvaluatorRuntimeProfile {
            worker_executable: Some(std::path::PathBuf::from("/tmp/worker")),
            worker_executable_identity: Some("crc32:11111111:bytes:1".to_string()),
            resolved_evaluator_cgroup_parent: Some(std::path::PathBuf::from("/sys/fs/cgroup/test")),
            memory_accounting_kind: strict_mode_test_accounting_kind(),
        },
    );
    assert_eq!(report["rss_mode"]["requested"], "hybrid_strict_max");
    let effective = report["rss_mode"]["effective_measurement"]
        .as_str()
        .expect("effective measurement");
    #[cfg(target_os = "linux")]
    assert_eq!(effective, "strict_linux_max_process_rss_cgroup_v2_peak");
    #[cfg(all(unix, not(target_os = "linux")))]
    assert_eq!(effective, "unix_process_rss_fallback_explicit");
    #[cfg(not(unix))]
    assert_eq!(effective, "deterministic_evaluator_table_row_peak_memory");
}

#[test]
fn executor_controls_report_uses_explicit_deterministic_table_provenance() {
    let config = TuneExecutionConfig {
        rss_mode: PeakMemoryMode::BackendReported,
        ..TuneExecutionConfig::default()
    };
    let report = executor_controls_report(
        &config,
        &ResolvedEvaluatorRuntimeProfile {
            worker_executable: None,
            worker_executable_identity: None,
            resolved_evaluator_cgroup_parent: None,
            memory_accounting_kind: ResolvedMemoryAccountingKind::DeterministicEvaluatorTable,
        },
    );
    assert_eq!(report["rss_mode"]["requested"], "backend_reported");
    assert_eq!(
        report["rss_mode"]["effective_measurement"],
        "deterministic_evaluator_table_row_peak_memory"
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn evaluator_profile_cache_key_changes_with_execution_profile_only() {
    let profile_a = EvaluatorProfile {
        dataset_kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        dataset_lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        dataset_codec_hash: "passive-identity-bytes".to_string(),
        event_grammar_hash: "passive-target-only-byte-stream".to_string(),
        target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
        causal_header_profile_hash: crc32_hex(b"passive-none"),
        target_size_function: "passive-bytes-len",
        evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
        candidate_canonicalization_version: "bounds-v1".to_string(),
        warmup_baseline_runs: 0,
        diagnostic_chunk_bytes: None,
        eval_time_limit_seconds: 1.0,
        evaluator_threads: 1,
        worker_isolation_mode: "spawn_exec_worker",
        worker_executable_identity: None,
        resolved_memory_accounting_kind: "unix_process_rss_fallback_explicit",
        resolved_memory_accounting_strict_theorem_facing: false,
        resolved_evaluator_cgroup_parent: None,
        backend_report_component_policy: "none",
        evaluator_determinism: "deterministic_under_h",
        rss_mode: PeakMemoryMode::ProcessRssPeak,
        timing_certification_tier: TimingCertificationTier::BestEffort,
        build_profile: "test",
        feature_set: vec!["test"],
    };
    let mut profile_b = profile_a.clone();
    profile_b.warmup_baseline_runs = 3;
    let mut profile_c = profile_a.clone();
    profile_c.eval_time_limit_seconds = 0.5;
    let mut profile_d = profile_a.clone();
    profile_d.diagnostic_chunk_bytes = Some(4096);

    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let candidate_bytes = candidate
        .compile()
        .expect("compile candidate")
        .canonical_bytes()
        .as_slice()
        .to_vec();
    let dataset_hash = crc32_hex(b"same-dataset");
    let key_a =
        cache_key_for_candidate(&candidate_bytes, &profile_a, &dataset_hash).expect("cache key a");
    let key_b =
        cache_key_for_candidate(&candidate_bytes, &profile_b, &dataset_hash).expect("cache key b");
    let key_c =
        cache_key_for_candidate(&candidate_bytes, &profile_c, &dataset_hash).expect("cache key c");
    let key_d =
        cache_key_for_candidate(&candidate_bytes, &profile_d, &dataset_hash).expect("cache key d");
    assert_ne!(key_a, key_b);
    assert_ne!(key_a, key_c);
    assert_ne!(key_a, key_d);
    assert_eq!(key_a.candidate_canonical_bytes, candidate_bytes);
    assert_eq!(
        key_b.candidate_canonical_bytes,
        key_a.candidate_canonical_bytes
    );
    assert_eq!(
        key_c.candidate_canonical_bytes,
        key_a.candidate_canonical_bytes
    );
    assert_eq!(key_b.dataset_identity, key_a.dataset_identity);
    assert_ne!(key_b.evaluator_profile_bytes, key_a.evaluator_profile_bytes);
    assert_ne!(key_c.evaluator_profile_bytes, key_a.evaluator_profile_bytes);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn deterministic_table_evaluation_enforces_exact_objective_formula() {
    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 4 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .expect("compile candidate");
    let candidate_crc32 = crc32_hex(candidate.canonical_bytes().as_slice());
    let model_bytes: usize = 17;
    let target_loss_bits = 23.5;
    let table = VerifiedDeterministicEvaluatorTable {
        base: VerifiedCertificate {
            ref_value: "test://deterministic-table".to_string(),
            content_hash: "00000000".to_string(),
        },
        rows: HashMap::from([(
            candidate_crc32,
            DeterministicEvaluatorRow {
                status: CandidateEvalStatus::Success,
                compressed_bytes: 3,
                target_loss_bits,
                elapsed_seconds: 0.25,
                peak_memory_bytes: 16,
            },
        )]),
    };
    let dataset = LoadedDataset {
        kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        codec_hash: "passive-identity-bytes".to_string(),
        event_grammar_hash: "passive-target-only-byte-stream".to_string(),
        target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
        causal_header_profile_hash: crc32_hex(b"passive-none"),
        target_size_function: "passive-bytes-len",
        canonical_content_hash: crc32_hex(b"dataset"),
        lowered_skeleton_hash: crc32_hex(b"passive-bytes-target-only"),
        resolved_path: "test://dataset".to_string(),
        source_size_bytes: 11,
        raw_bytes: b"hello world".to_vec(),
        events: Vec::new(),
        causal_profile: None,
        dataset_units: 11.0,
        target_events: 1,
    };

    let result = table
        .evaluate(&candidate, &dataset, model_bytes, 1.0, 1024, 1.0)
        .expect("deterministic table evaluation");
    assert_eq!(result.status, CandidateEvalStatus::Success);
    assert_eq!(result.target_loss_bits, target_loss_bits);
    assert_eq!(
        result.objective_bits,
        (model_bytes as f64 * 8.0) + target_loss_bits
    );
    assert_eq!(result.throughput_bytes_per_second, 44.0);
    assert!(result.deployable);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn deterministic_table_success_row_exceeding_effective_limit_is_timeout() {
    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 4 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .expect("compile candidate");
    let candidate_crc32 = crc32_hex(candidate.canonical_bytes().as_slice());
    let table = VerifiedDeterministicEvaluatorTable {
        base: VerifiedCertificate {
            ref_value: "test://deterministic-table".to_string(),
            content_hash: "00000000".to_string(),
        },
        rows: HashMap::from([(
            candidate_crc32,
            DeterministicEvaluatorRow {
                status: CandidateEvalStatus::Success,
                compressed_bytes: 3,
                target_loss_bits: 23.5,
                elapsed_seconds: 2.0,
                peak_memory_bytes: 16,
            },
        )]),
    };
    let dataset = LoadedDataset {
        kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        codec_hash: "passive-identity-bytes".to_string(),
        event_grammar_hash: "passive-target-only-byte-stream".to_string(),
        target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
        causal_header_profile_hash: crc32_hex(b"passive-none"),
        target_size_function: "passive-bytes-len",
        canonical_content_hash: crc32_hex(b"dataset"),
        lowered_skeleton_hash: crc32_hex(b"passive-bytes-target-only"),
        resolved_path: "test://dataset".to_string(),
        source_size_bytes: 11,
        raw_bytes: b"hello world".to_vec(),
        events: Vec::new(),
        causal_profile: None,
        dataset_units: 11.0,
        target_events: 1,
    };

    let effective_limit_seconds = 1.0;
    let result = table
        .evaluate(&candidate, &dataset, 17, 1.0, 1024, effective_limit_seconds)
        .expect("deterministic table evaluation");

    assert_eq!(result.status, CandidateEvalStatus::Timeout);
    assert_eq!(result.elapsed_seconds, 2.0);
    assert_eq!(
        result.effective_eval_time_limit_seconds,
        effective_limit_seconds
    );
    assert_eq!(result.peak_memory_bytes, 16);
    assert_eq!(result.target_loss_bits, f64::INFINITY);
    assert_eq!(result.objective_bits, f64::INFINITY);
    assert_eq!(result.throughput_bytes_per_second, 0.0);
    assert!(!result.deployable);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn candidate_bounds_validation_enforces_parameter_ranges() {
    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let bounds = TuneBoundsSpec {
        allowed_backends: vec!["ctw".to_string()],
        forbidden_backends: Vec::new(),
        parameter_ranges: vec![crate::spec::TuneParameterRangeSpec {
            parameter: "rate_backend.depth".to_string(),
            min: 4.0,
            max: 6.0,
        }],
        max_experts: 2,
        max_mixture_nesting_depth: 1,
        min_experts: Some(1),
        allow_duplicate_experts: Some(false),
        required_experts: Vec::new(),
        forbidden_expert_pairs: Vec::new(),
    };
    let err = validate_candidate_against_tune_bounds(&candidate, &bounds)
        .expect_err("depth out of range must fail");
    assert!(err.contains("rate_backend.depth"));
}

#[cfg(feature = "backend-ctw")]
#[test]
fn canonical_proposal_kernel_accounts_exact_integer_masses() {
    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 2 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let bounds = TuneBoundsSpec {
        allowed_backends: vec!["ctw".to_string()],
        forbidden_backends: Vec::new(),
        parameter_ranges: vec![crate::spec::TuneParameterRangeSpec {
            parameter: "rate_backend.depth".to_string(),
            min: 1.0,
            max: 3.0,
        }],
        max_experts: 2,
        max_mixture_nesting_depth: 1,
        min_experts: Some(1),
        allow_duplicate_experts: Some(false),
        required_experts: Vec::new(),
        forbidden_expert_pairs: Vec::new(),
    };
    let env = SpecEnvironment::new(".");
    let current = candidate.compile_in(&env).expect("compile current");
    let current_bytes = current.canonical_bytes().as_slice().to_vec();
    let kernel = compile_canonical_proposal_kernel(&candidate, &bounds, 1, 1, &env, &current_bytes)
        .expect("compile proposal kernel");
    assert_eq!(kernel.total_raw_actions, 2);
    assert_eq!(kernel.transitions.len(), 2);
    assert!(
        kernel
            .transitions
            .iter()
            .all(|proposal| proposal.raw_action_count == 1)
    );

    let lower = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 1 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let lower_bytes = lower
        .compile_in(&env)
        .expect("compile lower")
        .canonical_bytes()
        .as_slice()
        .to_vec();
    assert_eq!(kernel.proposal_mass_to_canonical_bytes(&lower_bytes), 1);

    let reverse = compile_canonical_proposal_kernel(&lower, &bounds, 1, 1, &env, &lower_bytes)
        .expect("compile reverse kernel");
    assert_eq!(reverse.total_raw_actions, 2);
    assert_eq!(reverse.proposal_mass_to_canonical_bytes(&current_bytes), 1);
    assert_eq!(reverse.transitions.len(), 1);
}

#[cfg(feature = "backend-match")]
#[test]
fn canonical_proposal_kernel_explores_bounded_float_only_search_space() {
    use crate::api::RateBackend;
    use crate::compression::FramingMode;
    use crate::spec::{TuneBoundsSpec, TuneParameterRangeSpec};

    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Match {
            hash_bits: 18,
            min_len: 4,
            max_len: 64,
            base_mix: 0.02,
            confidence_scale: 1.0,
        },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let bounds = TuneBoundsSpec {
        allowed_backends: vec!["match".to_string()],
        forbidden_backends: Vec::new(),
        parameter_ranges: vec![TuneParameterRangeSpec {
            parameter: "rate_backend.base_mix".to_string(),
            min: 0.01,
            max: 0.04,
        }],
        max_experts: 2,
        max_mixture_nesting_depth: 1,
        min_experts: Some(1),
        allow_duplicate_experts: Some(false),
        required_experts: Vec::new(),
        forbidden_expert_pairs: Vec::new(),
    };
    let env = SpecEnvironment::new(".");
    let current = candidate.compile_in(&env).expect("compile current");
    let current_bytes = current.canonical_bytes().as_slice().to_vec();
    let kernel = compile_canonical_proposal_kernel(&candidate, &bounds, 2, 2, &env, &current_bytes)
        .expect("compile float proposal kernel");

    assert_eq!(kernel.total_raw_actions, 4);
    assert!(
        !kernel.transitions.is_empty(),
        "bounded float-only search spaces must produce non-self proposals"
    );

    for proposal in &kernel.transitions {
        let CompressionBackend::Rate {
            rate_backend: RateBackend::Match { base_mix, .. },
            ..
        } = &proposal.candidate
        else {
            panic!("expected match proposal");
        };
        assert!((0.01..=0.04).contains(base_mix));

        let reverse = compile_canonical_proposal_kernel(
            &proposal.candidate,
            &bounds,
            2,
            2,
            &env,
            &proposal.candidate_canonical_bytes,
        )
        .expect("compile reverse float proposal kernel");
        assert_eq!(reverse.total_raw_actions, 4);
        assert_eq!(
            reverse.proposal_mass_to_canonical_bytes(&current_bytes),
            proposal.raw_action_count
        );
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn reversible_metropolis_acceptance_uses_objective_bits_temperature() {
    let proposal = AnnealedProposal {
        candidate: CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 1 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        },
        forward_raw_action_count: 1,
        forward_total_raw_actions: 2,
        reverse_raw_action_count: 1,
        reverse_total_raw_actions: 2,
    };
    let uphill = annealer_acceptance_probability(
        AnnealerKernelProfile::ReversibleElementaryMetropolis,
        3.0,
        2.0,
        &proposal,
    )
    .expect("reversible metropolis probability");
    assert!((uphill - (-1.5f64).exp()).abs() <= f64::EPSILON);
    let downhill = annealer_acceptance_probability(
        AnnealerKernelProfile::ReversibleElementaryMetropolis,
        -2.0,
        3.0,
        &proposal,
    )
    .expect("reversible metropolis probability");
    assert!((downhill - 1.0).abs() <= f64::EPSILON);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn metropolis_acceptance_envelope_sweep() {
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    use crate::tuner::AnnealedProposal;
    use crate::tuner::annealer::annealer_acceptance_probability;
    use crate::tuner::config::AnnealerKernelProfile;

    let proposal = AnnealedProposal {
        candidate: CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 1 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        },
        forward_raw_action_count: 1,
        forward_total_raw_actions: 2,
        reverse_raw_action_count: 1,
        reverse_total_raw_actions: 2,
    };

    let mut previous_probability: Option<f64> = None;
    for t in 1..=100 {
        let temp = t as f64;
        let delta = 2.0;
        let prob = annealer_acceptance_probability(
            AnnealerKernelProfile::ReversibleElementaryMetropolis,
            delta,
            temp,
            &proposal,
        )
        .unwrap();
        let expected = (-delta / temp).exp().clamp(0.0, 1.0);

        assert!(
            (prob - expected).abs() <= f64::EPSILON,
            "uphill Metropolis probability must equal exp(-delta / temperature): prob={prob}, expected={expected}, temperature={temp}"
        );
        if let Some(previous) = previous_probability {
            assert!(
                previous < prob,
                "uphill Metropolis probability must strictly increase with temperature: previous={previous}, current={prob}, temperature={temp}"
            );
        }
        previous_probability = Some(prob);

        let delta_neg = -2.0;
        let prob_neg = annealer_acceptance_probability(
            AnnealerKernelProfile::ReversibleElementaryMetropolis,
            delta_neg,
            temp,
            &proposal,
        )
        .unwrap();
        assert_eq!(prob_neg, 1.0, "Negative delta must always be accepted");
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn compiled_uniform_mh_uses_hastings_ratio_for_asymmetric_boundary_mass() {
    let proposal = AnnealedProposal {
        candidate: CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 1 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        },
        forward_raw_action_count: 1,
        forward_total_raw_actions: 2,
        reverse_raw_action_count: 1,
        reverse_total_raw_actions: 4,
    };
    let probability = annealer_acceptance_probability(
        AnnealerKernelProfile::CompiledUniformMetropolisHastings,
        1.0,
        1.0,
        &proposal,
    )
    .expect("mh probability");
    let expected = (-1.0f64).exp() * 0.5;
    assert!((probability - expected).abs() <= f64::EPSILON);
    let err = annealer_acceptance_probability(
        AnnealerKernelProfile::ReversibleElementaryMetropolis,
        1.0,
        1.0,
        &proposal,
    )
    .expect_err("default profile must reject asymmetric masses");
    assert!(err.contains("reversibility check"));
}

#[test]
fn key_less_uses_canonical_bytes_on_objective_ties() {
    let eval = CandidateEvalResult {
        status: CandidateEvalStatus::Success,
        compressed_bytes: 0,
        elapsed_seconds: 1.0,
        effective_eval_time_limit_seconds: 1.0,
        throughput_bytes_per_second: 1.0,
        peak_memory_bytes: 1,
        target_loss_bits: 1.0,
        objective_bits: 42.0,
        deployable: true,
    };
    let smaller = vec![0x01_u8, 0x02_u8];
    let larger = vec![0x01_u8, 0x03_u8];
    assert!(key_less(&eval, &smaller, &eval, &larger));
    assert!(!key_less(&eval, &larger, &eval, &smaller));
}

#[test]
fn key_less_requires_exact_objective_tie_before_byte_tiebreak() {
    let incumbent = CandidateEvalResult {
        status: CandidateEvalStatus::Success,
        compressed_bytes: 0,
        elapsed_seconds: 1.0,
        effective_eval_time_limit_seconds: 1.0,
        throughput_bytes_per_second: 1.0,
        peak_memory_bytes: 1,
        target_loss_bits: 1.0,
        objective_bits: 42.0,
        deployable: true,
    };
    let candidate = CandidateEvalResult {
        objective_bits: f64::from_bits(incumbent.objective_bits.to_bits() + 1),
        ..incumbent.clone()
    };
    let candidate_bytes = vec![0x01_u8, 0x00_u8];
    let incumbent_bytes = vec![0x01_u8, 0x01_u8];
    assert!(
        !key_less(&candidate, &candidate_bytes, &incumbent, &incumbent_bytes),
        "byte-order tiebreak must not apply unless objective bits are exactly equal"
    );
}

fn decode_observation_optional_f64(bytes: &[u8], field_index: usize) -> Option<f64> {
    let mut offset = 1usize;
    for index in 0..5 {
        let present = bytes[offset];
        offset += 1;
        if present == 1 {
            let mut raw = [0_u8; 8];
            raw.copy_from_slice(&bytes[offset..offset + 8]);
            let value = f64::from_bits(u64::from_le_bytes(raw));
            offset += 8;
            if index == field_index {
                return Some(value);
            }
        } else if index == field_index {
            return None;
        }
    }
    None
}

#[test]
fn raw_observation_timeout_sets_tau_one_and_invalid_uses_sentinel() {
    let incumbent = CandidateEvalResult {
        status: CandidateEvalStatus::Success,
        compressed_bytes: 10,
        elapsed_seconds: 0.5,
        effective_eval_time_limit_seconds: 1.0,
        throughput_bytes_per_second: 2.0,
        peak_memory_bytes: 1,
        target_loss_bits: 80.0,
        objective_bits: 100.0,
        deployable: true,
    };
    let timeout = timeout_eval_result(0.25, 1, 0.25);
    let timeout_observation = TunerRawObservation::from_runtime_step(
        Some(&incumbent),
        10.0,
        Some(&timeout),
        Some(b"candidate-timeout"),
        Some(0.25),
        "evaluator_timeout",
        false,
    );
    assert_eq!(
        decode_observation_optional_f64(timeout_observation.encoded_bytes(), 2),
        Some(1.0)
    );
    let invalid = CandidateEvalResult {
        status: CandidateEvalStatus::Invalid,
        compressed_bytes: 0,
        elapsed_seconds: 0.0,
        effective_eval_time_limit_seconds: 1.0,
        throughput_bytes_per_second: 0.0,
        peak_memory_bytes: 0,
        target_loss_bits: f64::INFINITY,
        objective_bits: f64::INFINITY,
        deployable: false,
    };
    let invalid_observation = TunerRawObservation::from_runtime_step(
        Some(&incumbent),
        10.0,
        Some(&invalid),
        Some(b"candidate-invalid"),
        Some(1.0),
        "evaluator_invalid",
        false,
    );
    assert_eq!(
        decode_observation_optional_f64(invalid_observation.encoded_bytes(), 2),
        None
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_percept_encoding_distinguishes_diagnostic_tokens() {
    let interface = TunePlannerInterfaceSpec {
        observation_bits: 16,
        observation_stream_len: 2,
        observation_key_mode: ObservationKeyMode::FullStream,
        reward_bits: 8,
        agent_actions: action_alphabet(2),
    };
    let current = vec![1_u8, 2, 3];
    let incumbent_eval = CandidateEvalResult {
        status: CandidateEvalStatus::Success,
        compressed_bytes: 12,
        elapsed_seconds: 0.25,
        effective_eval_time_limit_seconds: 1.0,
        throughput_bytes_per_second: 4.0,
        peak_memory_bytes: 1,
        target_loss_bits: 96.0,
        objective_bits: 128.0,
        deployable: true,
    };
    let inapplicable = encode_tuner_planner_percept(
        &interface,
        Some(&incumbent_eval),
        16.0,
        0,
        "inapplicable_action",
        None,
        None,
        None,
        false,
    )
    .expect("inapplicable percept");
    let invalid = encode_tuner_planner_percept(
        &interface,
        Some(&incumbent_eval),
        16.0,
        0,
        "invalid_action_index",
        None,
        None,
        None,
        false,
    )
    .expect("invalid percept");
    let nondeployable = encode_tuner_planner_percept(
        &interface,
        Some(&incumbent_eval),
        16.0,
        0,
        "nondeployable_candidate",
        Some(&CandidateEvalResult {
            status: CandidateEvalStatus::Invalid,
            compressed_bytes: 0,
            elapsed_seconds: 0.0,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: 0.0,
            peak_memory_bytes: 0,
            target_loss_bits: f64::INFINITY,
            objective_bits: f64::INFINITY,
            deployable: false,
        }),
        Some(&current),
        Some(1.0),
        false,
    )
    .expect("nondeployable percept");
    assert_ne!(inapplicable.observations, invalid.observations);
    assert_ne!(inapplicable.observations, nondeployable.observations);
    assert_ne!(invalid.observations, nondeployable.observations);
}

#[test]
fn theorem_claims_reject_float_planner_mutation_domains() {
    let actions = vec![
        PlannerMutationAction::NumericStep {
            path: "rate_backend.temperature".to_string(),
            pointer: "/rate_backend/temperature".to_string(),
            kind: NumericKind::Float,
            delta: 0.05,
            range: None,
        },
        PlannerMutationAction::Noop,
    ];
    let mut theorem = TuneTheoremConfig::default();
    validate_theorem_planner_mutation_domain(&actions, &theorem)
        .expect("operational run may use float mutation leaves");
    theorem.claim_exact_finite_mdp = true;
    let err = validate_theorem_planner_mutation_domain(&actions, &theorem)
        .expect_err("exact theorem claim must reject float mutation leaves");
    assert!(err.contains("theorem_finite_state_unsafe"), "{err}");
}

#[cfg(all(feature = "backend-ctw", feature = "backend-mixture"))]
#[test]
fn planner_float_mutation_uses_bounds_scale_for_small_positive_alpha() {
    use std::sync::Arc;

    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Mixture {
            spec: Arc::new(
                crate::api::MixtureSpec::new(
                    crate::api::MixtureKind::Neural,
                    vec![
                        crate::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                            .with_name("ctw"),
                    ],
                )
                .with_alpha(0.03),
            ),
        },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let action = PlannerMutationAction::NumericStep {
        path: "rate_backend.spec.alpha".to_string(),
        pointer: "/rate_backend/spec/alpha".to_string(),
        kind: NumericKind::Float,
        delta: -0.05,
        range: Some((0.005, 0.2)),
    };

    let mutated = apply_planner_mutation_action(&candidate, &action)
        .expect("planner mutation should decode")
        .expect("bounded float action should remain applicable");
    let json = crate::spec::compression_backend_to_json_value(&mutated)
        .expect("mutated candidate should serialize");
    let alpha = json
        .pointer("/rate_backend/spec/alpha")
        .and_then(Value::as_f64)
        .expect("mutated mixture alpha");

    assert!(alpha > 0.005, "alpha should remain within declared bounds");
    assert!(alpha < 0.03, "negative planner step should decrease alpha");
}

#[cfg(all(feature = "backend-ctw", feature = "backend-mixture"))]
#[test]
fn planner_unparsable_float_mutation_is_inapplicable_not_fatal() {
    use std::sync::Arc;

    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Mixture {
            spec: Arc::new(
                crate::api::MixtureSpec::new(
                    crate::api::MixtureKind::Neural,
                    vec![
                        crate::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                            .with_name("ctw"),
                    ],
                )
                .with_alpha(0.03),
            ),
        },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let action = PlannerMutationAction::NumericStep {
        path: "rate_backend.spec.alpha".to_string(),
        pointer: "/rate_backend/spec/alpha".to_string(),
        kind: NumericKind::Float,
        delta: -0.05,
        range: None,
    };

    let mutated = apply_planner_mutation_action(&candidate, &action)
        .expect("unparsable planner edit should not abort the run");
    assert!(
        mutated.is_none(),
        "invalid float planner edit should be treated as inapplicable"
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn theorem_claims_continue_as_uncertified_when_requested_prereqs_are_missing() {
    let dataset_path = temp_path("dataset-theorem-policy", ".bin");
    let output_path = temp_path("output-theorem-policy", ".json");
    let report_path = temp_path("report-theorem-policy", ".json");
    std::fs::write(&dataset_path, b"theorem policy dataset").expect("write dataset");
    let mut spec = sample_tune_spec(
        dataset_path.to_str().expect("dataset path"),
        output_path.to_str().expect("output path"),
        report_path.to_str().expect("report path"),
    );
    let interface = planner_interface_for_baseline(&spec.baseline_candidate);
    spec.controller = TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
        interface,
        planner_simulations_per_step: 2,
    });
    let best_candidate = spec.baseline_candidate.clone();
    let compiled = spec.compile().expect("compile tune spec");
    let dataset = load_dataset(&dataset_path).expect("load dataset");
    let search = SearchSummary {
        status: "completed_mc_aixi_fac_ctw",
        warning: None,
        fatal_evaluator_failure: None,
        fatal_evaluator_failures: 0,
        best_candidate,
        best_candidate_crc32: "00000000".to_string(),
        best_eval: CandidateEvalResult {
            status: CandidateEvalStatus::Success,
            compressed_bytes: 8,
            elapsed_seconds: 0.1,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: 10.0,
            peak_memory_bytes: 1,
            target_loss_bits: 64.0,
            objective_bits: 128.0,
            deployable: true,
        },
        cache_key_digest: "00000000".to_string(),
        cache_hits: 0,
        cache_misses: 1,
        candidate_evaluations_executed: 1,
        non_warmup_candidate_results_seen: 1,
        post_baseline_candidate_results_seen: 0,
        proposals_attempted: 0,
        proposals_invalid: 0,
        self_loop_proposals: 0,
        invalid_reason_counts: InvalidReasonCounts::default(),
        successful_non_deployable: 0,
        candidate_result_counts: CandidateResultCounts {
            success_deployable: 1,
            success_non_deployable: 0,
            timeout: 0,
            invalid: 0,
            error_recoverable: 0,
        },
        final_best_move_reward: 0.0,
        realized_trace_counts_by_round: None,
        trace_refresh_merges_by_round: None,
        controller_report: Value::Null,
    };
    let theorem = TuneTheoremConfig {
        claim_exact_finite_mdp: true,
        claim_exact_observed_markov: true,
        claim_planner_convergence: true,
        ..TuneTheoremConfig::default()
    };

    let report = theorem_claims_report(
        &theorem,
        &VerifiedTheoremInputs::default(),
        compiled.controller(),
        &dataset,
        &search,
        false,
    );
    for pointer in [
        "/exact_finite_mdp/status",
        "/exact_observed_markov/status",
        "/planner_convergence/status",
    ] {
        assert_eq!(
            report.pointer(pointer).and_then(Value::as_str),
            Some("uncertified")
        );
    }
    assert!(
        report["exact_observed_markov"]["missing_prerequisites"]
            .as_array()
            .expect("missing prerequisites")
            .iter()
            .any(|item| item.as_str() == Some("verified_exact_state_observation_certificate"))
    );

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(output_path);
    let _ = std::fs::remove_file(report_path);
}

#[test]
fn candidate_external_asset_references_are_rejected() {
    let candidate = CompressionBackend::zpaq("file:./candidate-model.zpaq");
    let err = reject_candidate_local_external_artifacts(&candidate)
        .expect_err("external file reference must fail");
    assert_eq!(
        err.reason,
        TuneInvalidReason::CandidateExternalAssetForbidden
    );
    assert!(
        err.diagnostic
            .contains(TuneInvalidReason::CandidateExternalAssetForbidden.as_str())
    );
    assert!(
        err.diagnostic
            .contains("candidate-local external filesystem/model path")
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn run_tune_writes_output_and_report_for_baseline_pass() {
    let dataset_path = temp_path("dataset", ".bin");
    let spec_path = temp_path("spec", ".json");
    let output_path = temp_path("output", ".json");
    let report_path = temp_path("report", ".json");
    std::fs::write(&dataset_path, b"hello baseline").expect("write dataset");

    let spec = sample_tune_spec(
        dataset_path.to_str().expect("dataset path"),
        output_path.to_str().expect("output path"),
        report_path.to_str().expect("report path"),
    );
    let spec_json = SpecDocument::Tune(spec)
        .to_canonical_json()
        .expect("spec json");
    std::fs::write(&spec_path, spec_json).expect("write spec");

    let request = TuneCommandRequest {
        spec_path: spec_path.to_string_lossy().to_string(),
        emit_exact_reward_encoding_certificate: None,
        execution: TuneExecutionConfig::default(),
    };
    run_tune(&request).expect("run tune");

    let output = std::fs::read_to_string(&output_path).expect("output exists");
    assert!(output.contains("\"kind\": \"rate-ac\""));
    let report = std::fs::read_to_string(&report_path).expect("report exists");
    assert!(report.contains("\"kind\": \"tune_report\""));

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(spec_path);
    let _ = std::fs::remove_file(output_path);
    let _ = std::fs::remove_file(report_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn run_tune_fails_when_baseline_not_deployable() {
    let dataset_path = temp_path("dataset", ".bin");
    let spec_path = temp_path("spec", ".json");
    let output_path = temp_path("output", ".json");
    let report_path = temp_path("report", ".json");
    std::fs::write(&dataset_path, vec![0u8; 4096]).expect("write dataset");

    let mut spec = sample_tune_spec(
        dataset_path.to_str().expect("dataset path"),
        output_path.to_str().expect("output path"),
        report_path.to_str().expect("report path"),
    );
    spec.min_throughput_bytes_per_second = f64::MAX;
    let spec_json = SpecDocument::Tune(spec)
        .to_canonical_json()
        .expect("spec json");
    std::fs::write(&spec_path, spec_json).expect("write spec");

    let request = TuneCommandRequest {
        spec_path: spec_path.to_string_lossy().to_string(),
        emit_exact_reward_encoding_certificate: None,
        execution: TuneExecutionConfig::default(),
    };
    let err = run_tune(&request).expect_err("non-deployable baseline must fail");
    assert!(err.contains("not deployable"));

    let report = std::fs::read_to_string(&report_path).expect("report exists");
    assert!(report.contains("\"status\": \"baseline_not_deployable\""));
    assert!(!output_path.exists());

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(spec_path);
    let _ = std::fs::remove_file(report_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn run_tune_can_emit_exact_reward_certificate_and_exit() {
    let dataset_path = temp_path("dataset-emit-reward", ".bin");
    let spec_path = temp_path("spec-emit-reward", ".json");
    let output_path = temp_path("output-emit-reward", ".json");
    let report_path = temp_path("report-emit-reward", ".json");
    let emitted_cert_path = temp_path("exact-reward-emitted", ".json");
    std::fs::write(&dataset_path, b"emit-reward-dataset").expect("write dataset");

    let mut spec = sample_tune_spec(
        dataset_path.to_str().expect("dataset path"),
        output_path.to_str().expect("output path"),
        report_path.to_str().expect("report path"),
    );
    spec.controller = TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
        interface: planner_interface_for_baseline(&spec.baseline_candidate),
        planner_simulations_per_step: 4,
    });
    let spec_json = SpecDocument::Tune(spec)
        .to_canonical_json()
        .expect("spec json");
    std::fs::write(&spec_path, spec_json).expect("write spec");

    let request = TuneCommandRequest {
        spec_path: spec_path.to_string_lossy().to_string(),
        emit_exact_reward_encoding_certificate: Some(
            emitted_cert_path.to_string_lossy().to_string(),
        ),
        execution: TuneExecutionConfig::default(),
    };
    run_tune(&request).expect("emit exact reward certificate");

    let cert_bytes = std::fs::read(&emitted_cert_path).expect("read emitted certificate");
    let cert: Value = serde_json::from_slice(&cert_bytes).expect("parse emitted certificate");
    assert_eq!(cert["kind"], serde_json::json!("exact_reward_encoding"));
    assert_eq!(
        cert["controller_kind"],
        serde_json::json!("mc_aixi_fac_ctw")
    );
    assert_eq!(
        cert["encoding"],
        serde_json::json!("integer_objective_difference")
    );
    assert_eq!(
        cert["scalar_representation"],
        serde_json::json!(SCALAR_REPRESENTATION_DECLARATION)
    );
    assert!(
        cert["dataset_crc32"].as_str().is_some(),
        "dataset_crc32 must be emitted"
    );
    assert!(
        cert["bounds_crc32"].as_str().is_some(),
        "bounds_crc32 must be emitted"
    );
    assert!(
        cert["evaluator_profile_crc32"].as_str().is_some(),
        "evaluator_profile_crc32 must be emitted"
    );
    assert!(
        !output_path.exists(),
        "emit mode should not run candidate evaluation or write output config"
    );
    assert!(
        !report_path.exists(),
        "emit mode should exit before tune report generation"
    );

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(spec_path);
    let _ = std::fs::remove_file(emitted_cert_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn run_tune_emit_exact_reward_certificate_rejects_non_exact_controller_family() {
    let dataset_path = temp_path("dataset-emit-reward-nonexact", ".bin");
    let spec_path = temp_path("spec-emit-reward-nonexact", ".json");
    let output_path = temp_path("output-emit-reward-nonexact", ".json");
    let report_path = temp_path("report-emit-reward-nonexact", ".json");
    let emitted_cert_path = temp_path("exact-reward-emitted-nonexact", ".json");
    std::fs::write(&dataset_path, b"emit-reward-dataset-nonexact").expect("write dataset");
    let spec = sample_tune_spec(
        dataset_path.to_str().expect("dataset path"),
        output_path.to_str().expect("output path"),
        report_path.to_str().expect("report path"),
    );
    let spec_json = SpecDocument::Tune(spec)
        .to_canonical_json()
        .expect("spec json");
    std::fs::write(&spec_path, spec_json).expect("write spec");

    let request = TuneCommandRequest {
        spec_path: spec_path.to_string_lossy().to_string(),
        emit_exact_reward_encoding_certificate: Some(
            emitted_cert_path.to_string_lossy().to_string(),
        ),
        execution: TuneExecutionConfig::default(),
    };
    let err = run_tune(&request).expect_err("non-exact family must be rejected");
    assert!(
        err.contains("exact reward-encoding certificate emission is only supported"),
        "{err}"
    );

    assert!(
        !emitted_cert_path.exists(),
        "rejected emitter path must not write a certificate"
    );

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(spec_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn structured_causal_dataset_objects_lower_into_charged_targets() {
    let dataset_path = temp_path("dataset-causal", ".json");
    std::fs::write(
        &dataset_path,
        causal_dataset_value(
            "test-codec",
            "events",
            serde_json::json!([
                {"kind": "context", "channel": "action", "bytes": [1]},
                {"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [2]},
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [3, 4]}
            ]),
        )
        .to_string(),
    )
    .expect("write dataset");

    let dataset = load_dataset(&dataset_path).expect("causal dataset lowers");
    assert_eq!(dataset.kind, DatasetKind::InteractiveTrace);
    assert_eq!(
        dataset.objective_target,
        ObjectiveTarget::InteractiveCausalAc
    );
    assert_eq!(dataset.lowering_version, INTERACTIVE_TRACE_LOWERING_VERSION);
    assert_eq!(dataset.codec_hash, "test-codec");
    assert_eq!(dataset.raw_bytes, vec![3, 4]);
    assert_eq!(dataset.target_events, 2);
    assert_eq!(dataset.dataset_units, 2.0);
    assert!(matches!(
        &dataset.events[0],
        LoweredCausalEvent::Context { channel, bytes }
            if channel == "action" && bytes == &[1]
    ));
    assert!(matches!(
        &dataset.events[1],
        LoweredCausalEvent::ObserveTargetNoScore {
            channel,
            domain,
            bytes,
        } if channel == "percept" && domain == "bytes" && bytes == &[2]
    ));
    assert!(matches!(
        &dataset.events[2],
        LoweredCausalEvent::Target {
            channel,
            domain,
            bytes,
            weight,
        } if channel == "percept" && domain == "bytes" && bytes == &[3] && *weight == 1.0
    ));
    assert!(matches!(
        &dataset.events[3],
        LoweredCausalEvent::Target {
            channel,
            domain,
            bytes,
            weight,
        } if channel == "percept" && domain == "bytes" && bytes == &[4] && *weight == 1.0
    ));

    let _ = std::fs::remove_file(dataset_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn causal_prefix_lowering_resets_examples_and_replays_targets_without_score() {
    let dataset_path = temp_path("dataset-prefix-semantics", ".json");
    std::fs::write(
        &dataset_path,
        causal_dataset_value(
            "test-prefix-codec",
            "examples",
            serde_json::json!([
                {
                    "history": [
                        {"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [7]}
                    ],
                    "action": [1],
                    "channel": "percept",
                    "domain": "bytes",
                    "target": [8],
                    "weight": 2.0
                },
                {
                    "action": [0],
                    "channel": "percept",
                    "domain": "bytes",
                    "target": [9]
                }
            ]),
        )
        .to_string(),
    )
    .expect("write causal-prefix dataset");

    let dataset = load_dataset(&dataset_path).expect("causal-prefix dataset lowers");
    assert_eq!(dataset.kind, DatasetKind::CausalPrefixDataset);
    assert_eq!(dataset.raw_bytes, vec![8, 9]);
    assert_eq!(dataset.target_events, 2);
    assert_eq!(dataset.dataset_units, 3.0);
    assert_eq!(
        dataset
            .events
            .iter()
            .filter(|event| matches!(event, LoweredCausalEvent::Reset))
            .count(),
        2
    );
    assert!(matches!(&dataset.events[0], LoweredCausalEvent::Reset));
    assert!(matches!(
        &dataset.events[1],
        LoweredCausalEvent::ObserveTargetNoScore { bytes, .. } if bytes == &[7]
    ));
    assert!(matches!(
        &dataset.events[2],
        LoweredCausalEvent::Context { channel, bytes }
            if channel == "action" && bytes == &[1]
    ));
    assert!(matches!(
        &dataset.events[3],
        LoweredCausalEvent::Target { bytes, weight, .. }
            if bytes == &[8] && *weight == 2.0
    ));
    assert!(matches!(&dataset.events[4], LoweredCausalEvent::Reset));
    assert!(matches!(
        &dataset.events[5],
        LoweredCausalEvent::Context { channel, bytes }
            if channel == "action" && bytes == &[0]
    ));
    assert!(matches!(
        &dataset.events[6],
        LoweredCausalEvent::Target { bytes, weight, .. }
            if bytes == &[9] && *weight == 1.0
    ));

    let _ = std::fs::remove_file(dataset_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn structured_causal_dataset_rejects_missing_header_and_charged_history() {
    let missing_header_path = temp_path("dataset-missing-causal-header", ".json");
    std::fs::write(
        &missing_header_path,
        serde_json::json!({
            "schema_version": 1,
            "events": [{"kind": "target", "bytes": [1]}]
        })
        .to_string(),
    )
    .expect("write missing-header dataset");
    let err = load_dataset(&missing_header_path).expect_err("header must be required");
    assert!(err.contains("environment_id is required"), "{err}");

    let malformed_structured_path = temp_path("dataset-malformed-structured", ".json");
    std::fs::write(
        &malformed_structured_path,
        serde_json::json!({
            "schema_version": 1,
            "codec_hash": "looks-structured"
        })
        .to_string(),
    )
    .expect("write malformed structured dataset");
    let err = load_dataset(&malformed_structured_path)
        .expect_err("structured object must not be passive");
    assert!(
        err.contains("must match a canonical tuner causal dataset kind"),
        "{err}"
    );

    let charged_history_path = temp_path("dataset-charged-history", ".json");
    std::fs::write(
        &charged_history_path,
        causal_dataset_value(
            "charged-history",
            "examples",
            serde_json::json!([{
                "history": [{"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [7]}],
                "action": [1],
                "channel": "percept",
                "domain": "bytes",
                "target": [8]
            }]),
        )
        .to_string(),
    )
    .expect("write charged-history dataset");
    let err = load_dataset(&charged_history_path).expect_err("charged history must fail");
    assert!(err.contains("observe_target_no_score"), "{err}");

    let _ = std::fs::remove_file(missing_header_path);
    let _ = std::fs::remove_file(malformed_structured_path);
    let _ = std::fs::remove_file(charged_history_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn causal_dataset_header_and_event_grammar_are_strict() {
    let invalid_header_path = temp_path("dataset-invalid-header-types", ".json");
    std::fs::write(
        &invalid_header_path,
        serde_json::json!({
            "schema_version": 1,
            "environment_id": 7,
            "environment_config_crc32": "00000000",
            "codec_hash": "codec",
            "reset_convention": "reset-before-episode",
            "action_alphabet": {"size": 2},
            "percept_schema": {"encoding": "bytes"},
            "reward_encoding": {"encoding": "bytes"},
            "terminal_encoding": {"encoding": "bytes"},
            "collection_policy": "test",
            "events": [{"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}]
        })
        .to_string(),
    )
    .expect("write invalid-header dataset");
    let err = load_dataset(&invalid_header_path).expect_err("invalid header type must fail");
    assert!(err.contains("environment_id is required"), "{err}");

    let alias_event_path = temp_path("dataset-alias-event-kind", ".json");
    std::fs::write(
        &alias_event_path,
        causal_dataset_value(
            "test-codec",
            "events",
            serde_json::json!([
                {"kind": "context", "channel": "action", "bytes": [1]},
                {"kind": "observe", "channel": "percept", "domain": "bytes", "bytes": [2]},
            ]),
        )
        .to_string(),
    )
    .expect("write alias-event dataset");
    let err = load_dataset(&alias_event_path).expect_err("alias event kind must fail");
    assert!(err.contains("unknown causal event kind"), "{err}");

    let missing_event_grammar_path = temp_path("dataset-missing-event-grammar", ".json");
    std::fs::write(
        &missing_event_grammar_path,
        serde_json::json!({
            "schema_version": 1,
            "environment_id": "env",
            "environment_config_crc32": "00000000",
            "codec_hash": "codec",
            "reset_convention": "reset-before-episode",
            "action_alphabet": {"size": 2},
            "percept_schema": {"encoding": "bytes", "channels": [{"channel": "percept", "domain": "bytes"}]},
            "reward_encoding": {"encoding": "bytes", "channel": "reward", "domain": "binary"},
            "terminal_encoding": {"encoding": "bytes", "channel": "terminal", "domain": "binary"},
            "collection_policy": "test",
            "target_domains": {
                "bytes": {"kind": "byte_alphabet"},
                "binary": {"kind": "enumerated_payloads", "payloads": [[0], [1]]}
            },
            "events": [{"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}]
        })
        .to_string(),
    )
    .expect("write missing-event-grammar dataset");
    let err =
        load_dataset(&missing_event_grammar_path).expect_err("missing event_grammar must fail");
    assert!(err.contains("requires event_grammar"), "{err}");

    let missing_domain_path = temp_path("dataset-missing-domain", ".json");
    std::fs::write(
        &missing_domain_path,
        causal_dataset_value(
            "test-codec",
            "events",
            serde_json::json!([
                {"kind": "context", "channel": "action", "bytes": [1]},
                {"kind": "target", "channel": "percept", "bytes": [2]},
            ]),
        )
        .to_string(),
    )
    .expect("write missing-domain dataset");
    let err = load_dataset(&missing_domain_path).expect_err("missing domain must fail");
    assert!(err.contains(".domain is required"), "{err}");

    let _ = std::fs::remove_file(invalid_header_path);
    let _ = std::fs::remove_file(alias_event_path);
    let _ = std::fs::remove_file(missing_event_grammar_path);
    let _ = std::fs::remove_file(missing_domain_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn byte_alphabet_payloads_expand_to_single_byte_events() {
    let dataset_path = temp_path("dataset-byte-alphabet-expand", ".json");
    std::fs::write(
        &dataset_path,
        causal_dataset_value(
            "byte-expand-codec",
            "events",
            serde_json::json!([
                {"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [3, 4]},
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [7, 8], "weight": 2.0}
            ]),
        )
        .to_string(),
    )
    .expect("write byte-alphabet expansion dataset");
    let dataset = load_dataset(&dataset_path).expect("dataset lowers");
    assert!(matches!(
        &dataset.events[0],
        LoweredCausalEvent::ObserveTargetNoScore { bytes, .. } if bytes == &[3]
    ));
    assert!(matches!(
        &dataset.events[1],
        LoweredCausalEvent::ObserveTargetNoScore { bytes, .. } if bytes == &[4]
    ));
    assert!(matches!(
        &dataset.events[2],
        LoweredCausalEvent::Target { bytes, weight, .. } if bytes == &[7] && *weight == 2.0
    ));
    assert!(matches!(
        &dataset.events[3],
        LoweredCausalEvent::Target { bytes, weight, .. } if bytes == &[8] && *weight == 2.0
    ));
    let _ = std::fs::remove_file(dataset_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn byte_alphabet_empty_payload_is_rejected() {
    let dataset_path = temp_path("dataset-byte-alphabet-empty", ".json");
    std::fs::write(
        &dataset_path,
        causal_dataset_value(
            "byte-empty-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": []}
            ]),
        )
        .to_string(),
    )
    .expect("write byte-alphabet empty payload dataset");
    let err = load_dataset(&dataset_path).expect_err("empty byte-alphabet payload must fail");
    assert!(err.contains("must contain at least one byte"), "{err}");
    let _ = std::fs::remove_file(dataset_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn causal_header_cross_checks_enforce_grammar_and_action_contracts() {
    let undeclared_descriptor_path = temp_path("dataset-undeclared-event-descriptor", ".json");
    std::fs::write(
        &undeclared_descriptor_path,
        causal_dataset_value(
            "descriptor-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "other", "domain": "bytes", "bytes": [1]}
            ]),
        )
        .to_string(),
    )
    .expect("write undeclared descriptor dataset");
    let err =
        load_dataset(&undeclared_descriptor_path).expect_err("undeclared descriptor must fail");
    assert!(
        err.contains("not declared in event_grammar.target"),
        "{err}"
    );

    let invalid_action_path = temp_path("dataset-invalid-action-context", ".json");
    std::fs::write(
        &invalid_action_path,
        causal_dataset_value(
            "invalid-action-codec",
            "events",
            serde_json::json!([
                {"kind": "context", "channel": "action", "bytes": [2]},
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}
            ]),
        )
        .to_string(),
    )
    .expect("write invalid action dataset");
    let err = load_dataset(&invalid_action_path).expect_err("invalid action context must fail");
    assert!(err.contains("outside action_alphabet.size"), "{err}");

    let mut invalid_grammar = causal_dataset_value(
        "invalid-grammar-codec",
        "events",
        serde_json::json!([
            {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}
        ]),
    );
    let grammar = invalid_grammar
        .get_mut("event_grammar")
        .and_then(Value::as_object_mut)
        .expect("event_grammar object");
    let targets = grammar
        .get_mut("target")
        .and_then(Value::as_array_mut)
        .expect("target grammar array");
    targets.push(serde_json::json!({"channel": "ghost", "domain": "ghost"}));
    let invalid_grammar_path = temp_path("dataset-invalid-grammar-domain", ".json");
    std::fs::write(
        &invalid_grammar_path,
        serde_json::to_string(&invalid_grammar).expect("invalid grammar json"),
    )
    .expect("write invalid grammar dataset");
    let err =
        load_dataset(&invalid_grammar_path).expect_err("grammar with undeclared domain must fail");
    assert!(
        err.contains("event_grammar references undeclared target domain"),
        "{err}"
    );

    let _ = std::fs::remove_file(undeclared_descriptor_path);
    let _ = std::fs::remove_file(invalid_action_path);
    let _ = std::fs::remove_file(invalid_grammar_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn byte_alphabet_expansion_matches_chain_rule_loss() {
    let dataset_expanded_from_multibyte = temp_path("dataset-byte-chain-multibyte", ".json");
    std::fs::write(
        &dataset_expanded_from_multibyte,
        causal_dataset_value(
            "chain-rule-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [65, 66]}
            ]),
        )
        .to_string(),
    )
    .expect("write multi-byte dataset");
    let dataset_explicit_singletons = temp_path("dataset-byte-chain-singletons", ".json");
    std::fs::write(
        &dataset_explicit_singletons,
        causal_dataset_value(
            "chain-rule-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [65]},
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [66]}
            ]),
        )
        .to_string(),
    )
    .expect("write singleton dataset");

    let dataset_a = load_dataset(&dataset_expanded_from_multibyte).expect("load dataset a");
    let dataset_b = load_dataset(&dataset_explicit_singletons).expect("load dataset b");
    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .expect("compile ctw candidate");
    let deadline = Instant::now() + Duration::from_secs(2);
    let (compressed_a, loss_a) =
        evaluate_candidate_causal_loss(&candidate, &dataset_a, deadline).expect("eval a");
    let (compressed_b, loss_b) =
        evaluate_candidate_causal_loss(&candidate, &dataset_b, deadline).expect("eval b");
    assert_eq!(compressed_a, compressed_b);
    assert!(
        (loss_a - loss_b).abs() < 1.0e-10,
        "loss_a={loss_a}, loss_b={loss_b}"
    );

    let _ = std::fs::remove_file(dataset_expanded_from_multibyte);
    let _ = std::fs::remove_file(dataset_explicit_singletons);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn causal_dataset_domains_are_profile_fixed_and_support_checked() {
    let dataset_path = temp_path("dataset-enumerated-domain", ".json");
    std::fs::write(
        &dataset_path,
        causal_dataset_value(
            "test-enumerated-codec",
            "events",
            serde_json::json!([
                {"kind": "context", "channel": "action", "bytes": [1]},
                {"kind": "target", "channel": "percept", "domain": "binary", "bytes": [1]}
            ]),
        )
        .to_string(),
    )
    .expect("write enumerated-domain dataset");
    let dataset = load_dataset(&dataset_path).expect("enumerated domain dataset lowers");
    let causal_profile = dataset.causal_profile.as_ref().expect("causal profile");
    assert!(causal_profile.domains.contains_key("binary"));
    assert_eq!(
        dataset.target_domain_support_hash,
        causal_profile.domain_support_hash
    );
    assert_ne!(
        dataset.target_domain_support_hash,
        crc32_hex(b"passive-byte-alphabet")
    );

    let out_of_support_path = temp_path("dataset-enumerated-domain-out", ".json");
    std::fs::write(
        &out_of_support_path,
        causal_dataset_value(
            "test-enumerated-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "percept", "domain": "binary", "bytes": [2]}
            ]),
        )
        .to_string(),
    )
    .expect("write out-of-support dataset");
    let err = load_dataset(&out_of_support_path).expect_err("out-of-support target fails");
    assert!(err.contains("outside target-domain support"), "{err}");

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(out_of_support_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn causal_event_channel_and_domain_affect_skeleton_identity() {
    let path_a = temp_path("dataset-channel-a", ".json");
    let path_b = temp_path("dataset-channel-b", ".json");
    std::fs::write(
        &path_a,
        causal_dataset_value(
            "same-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [7]}
            ]),
        )
        .to_string(),
    )
    .expect("write channel a");
    std::fs::write(
        &path_b,
        causal_dataset_value(
            "same-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "reward", "domain": "binary", "bytes": [1]}
            ]),
        )
        .to_string(),
    )
    .expect("write channel b");

    let dataset_a = load_dataset(&path_a).expect("load channel a");
    let dataset_b = load_dataset(&path_b).expect("load channel b");
    assert_ne!(dataset_a.event_grammar_hash, dataset_b.event_grammar_hash);
    assert_eq!(dataset_a.codec_hash, dataset_b.codec_hash);

    let _ = std::fs::remove_file(path_a);
    let _ = std::fs::remove_file(path_b);
}

#[test]
fn annealer_schedule_matches_normative_log_linear_law() {
    let mid = annealer_temperature(0.5);
    let expected_mid = ANNEALER_T_MIN_BITS * (ANNEALER_T0_BITS / ANNEALER_T_MIN_BITS).powf(0.5);
    assert_eq!(annealer_progress_from_elapsed(0.0, 10.0), 0.0);
    assert_eq!(annealer_progress_from_elapsed(5.0, 10.0), 0.5);
    assert_eq!(annealer_progress_from_elapsed(20.0, 10.0), 1.0);
    assert!((annealer_temperature(0.0) - ANNEALER_T0_BITS).abs() < 1.0e-12);
    assert!((annealer_temperature(1.0) - ANNEALER_T_MIN_BITS).abs() < 1.0e-12);
    assert!((mid - expected_mid).abs() < 1.0e-12);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn exact_state_observation_projection_supports_stream_hash() {
    let interface = PlannerInterfaceSpec {
        observation_bits: 3,
        observation_stream_len: 2,
        observation_key_mode: ObservationKeyMode::StreamHash,
        reward_bits: 16,
        agent_actions: action_alphabet(2),
    };
    let projected = project_observation_output("stream_hash", &[9, 2], interface.observation_bits)
        .expect("stream_hash projection");
    assert_eq!(projected, vec![130]);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn discounted_aiqi_exact_theorem_claims_remain_uncertified_by_family() {
    let controller =
        crate::spec::CompiledTuneController::AiqiDiscounted(AiqiDiscountedTuneControllerSpec {
            interface: TunePlannerInterfaceSpec {
                observation_bits: 8,
                observation_stream_len: 1,
                observation_key_mode: ObservationKeyMode::FullStream,
                reward_bits: 16,
                agent_actions: action_alphabet(2),
            },
            planner_simulations_per_step: 1,
            return_horizon: 1,
            return_bins: 2,
            discount_factor: 0.0,
            min_improvement: 0.0,
            max_improvement: 1.0,
        });
    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let search = SearchSummary {
        status: "test",
        warning: None,
        fatal_evaluator_failure: None,
        fatal_evaluator_failures: 0,
        best_candidate: candidate,
        best_candidate_crc32: "00000000".to_string(),
        best_eval: CandidateEvalResult {
            status: CandidateEvalStatus::Success,
            compressed_bytes: 1,
            elapsed_seconds: 0.1,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: 10.0,
            peak_memory_bytes: 1,
            target_loss_bits: 8.0,
            objective_bits: 16.0,
            deployable: true,
        },
        cache_key_digest: "00000000".to_string(),
        cache_hits: 0,
        cache_misses: 0,
        candidate_evaluations_executed: 1,
        non_warmup_candidate_results_seen: 1,
        post_baseline_candidate_results_seen: 0,
        proposals_attempted: 0,
        proposals_invalid: 0,
        self_loop_proposals: 0,
        invalid_reason_counts: InvalidReasonCounts::default(),
        successful_non_deployable: 0,
        candidate_result_counts: CandidateResultCounts {
            success_deployable: 1,
            success_non_deployable: 0,
            timeout: 0,
            invalid: 0,
            error_recoverable: 0,
        },
        final_best_move_reward: 0.0,
        realized_trace_counts_by_round: None,
        trace_refresh_merges_by_round: None,
        controller_report: Value::Null,
    };
    let theorem = TuneTheoremConfig {
        claim_exact_finite_mdp: true,
        scalar_representation_ref: Some(SCALAR_REPRESENTATION_DECLARATION.to_string()),
        ..TuneTheoremConfig::default()
    };
    let verified = VerifiedTheoremInputs {
        finite_planner_state: Some(VerifiedCertificate {
            ref_value: "finite.json".to_string(),
            content_hash: "00000000".to_string(),
        }),
        no_hidden_state: Some(VerifiedCertificate {
            ref_value: "hidden.json".to_string(),
            content_hash: "00000000".to_string(),
        }),
        exact_reward_encoding: Some(VerifiedExactRewardEncodingCertificate {
            base: VerifiedCertificate {
                ref_value: "reward.json".to_string(),
                content_hash: "00000000".to_string(),
            },
            max_reward: 65_535,
            reward_bits: 16,
            scalar_representation: SCALAR_REPRESENTATION_DECLARATION.to_string(),
            mode: VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval,
        }),
        exact_state_observation: None,
        determinism_deadline: None,
        deterministic_table: None,
    };
    let missing =
        exact_finite_mdp_missing_prereqs(&theorem, &verified, &controller, &search, false);
    assert!(missing.contains(&"exact_objective_difference_controller"));
}

#[test]
fn warmstart_trace_merge_is_content_deduplicated_and_structurally_ordered() {
    let mut teacher = WarmStartExactJhTeacherDataset::new(
        WarmStartExactJhTeacherContract {
            schema_version: 1,
            task_fingerprint: TaskFingerprint::parse_hex(
                "0102030401020304010203040102030401020304010203040102030401020304",
            )
            .expect("test fingerprint"),
            action_alphabet_size: 2,
            observation_bits: 1,
            observation_stream_len: 1,
            observation_key_mode: "first".to_string(),
            observation_adapter_spec_ref: String::new(),
            observation_adapter_content_crc32: String::new(),
            reward_bits: 1,
            return_horizon: 1,
            label_phase_period: 1,
            scalar_representation: String::new(),
            exact_reward_encoding_certificate: String::new(),
        },
        Vec::new(),
    );
    let high_structural_trace = WarmStartExactJhTeacherTrace {
        transitions: vec![WarmStartExactJhTransition {
            action: 1_u64,
            observations: vec![2],
            reward: 3,
        }],
    };
    let low_structural_trace = WarmStartExactJhTeacherTrace {
        transitions: vec![WarmStartExactJhTransition {
            action: 0_u64,
            observations: vec![1],
            reward: 1,
        }],
    };
    teacher.traces.push(high_structural_trace.clone());

    assert!(
        merge_warmstart_trace_deterministic(&mut teacher, low_structural_trace.clone())
            .expect("merge distinct trace")
    );
    assert_eq!(
        teacher.traces,
        vec![low_structural_trace.clone(), high_structural_trace]
    );
    assert_eq!(teacher.traces.len(), 2);

    assert!(
        !merge_warmstart_trace_deterministic(&mut teacher, low_structural_trace)
            .expect("duplicate merge remains idempotent")
    );
    assert_eq!(teacher.traces.len(), 2);
}

#[test]
fn warmstart_trace_refresh_merge_counter_counts_structural_inserts_only() {
    let mut teacher = WarmStartExactJhTeacherDataset::new(
        WarmStartExactJhTeacherContract {
            schema_version: 1,
            task_fingerprint: TaskFingerprint::parse_hex(
                "0102030401020304010203040102030401020304010203040102030401020304",
            )
            .expect("test fingerprint"),
            action_alphabet_size: 2,
            observation_bits: 1,
            observation_stream_len: 1,
            observation_key_mode: "first".to_string(),
            observation_adapter_spec_ref: String::new(),
            observation_adapter_content_crc32: String::new(),
            reward_bits: 1,
            return_horizon: 1,
            label_phase_period: 1,
            scalar_representation: String::new(),
            exact_reward_encoding_certificate: String::new(),
        },
        Vec::new(),
    );
    let live_trace = WarmStartExactJhTeacherTrace {
        transitions: vec![WarmStartExactJhTransition {
            action: 0_u64,
            observations: vec![1],
            reward: 0,
        }],
    };
    let mut warmstart_trace_refresh_merges: usize = 0;
    for _ in 0..2 {
        if merge_warmstart_trace_deterministic(&mut teacher, live_trace.clone())
            .expect("merge live trace")
        {
            warmstart_trace_refresh_merges = warmstart_trace_refresh_merges.saturating_add(1);
        }
    }
    assert_eq!(warmstart_trace_refresh_merges, 1);
    assert_eq!(teacher.traces.len(), 1);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn run_tune_annealed_reports_search_activity() {
    let dataset_path = temp_path("dataset-annealed", ".bin");
    let spec_path = temp_path("spec-annealed", ".json");
    let output_path = temp_path("output-annealed", ".json");
    let report_path = temp_path("report-annealed", ".json");
    std::fs::write(&dataset_path, b"annealed-search-dataset").expect("write dataset");

    let mut spec = sample_tune_spec(
        dataset_path.to_str().expect("dataset path"),
        output_path.to_str().expect("output path"),
        report_path.to_str().expect("report path"),
    );
    spec.bounds.parameter_ranges = vec![crate::spec::TuneParameterRangeSpec {
        parameter: "rate_backend.depth".to_string(),
        min: 1.0,
        max: 16.0,
    }];
    let spec_json = SpecDocument::Tune(spec)
        .to_canonical_json()
        .expect("spec json");
    std::fs::write(&spec_path, spec_json).expect("write spec");

    let request = TuneCommandRequest {
        spec_path: spec_path.to_string_lossy().to_string(),
        emit_exact_reward_encoding_certificate: None,
        execution: TuneExecutionConfig {
            max_evaluations: Some(3),
            ..TuneExecutionConfig::default()
        },
    };
    run_tune(&request).expect("run tune");

    let report = std::fs::read_to_string(&report_path).expect("report exists");
    assert!(report.contains("\"status\": \"completed_annealed\""));
    assert!(report.contains("\"proposals_attempted\":"));
    let report_json: Value = serde_json::from_str(&report).expect("report json");
    assert_eq!(
        report_json
            .pointer("/search/baseline_counts_toward_max_evaluations")
            .and_then(Value::as_bool),
        Some(true)
    );
    let non_warmup_results = report_json
        .pointer("/search/non_warmup_candidate_results_seen")
        .and_then(Value::as_u64)
        .expect("non_warmup_candidate_results_seen");
    let post_baseline_results = report_json
        .pointer("/search/post_baseline_candidate_results_seen")
        .and_then(Value::as_u64)
        .expect("post_baseline_candidate_results_seen");
    assert!((1..=3).contains(&non_warmup_results));
    assert_eq!(post_baseline_results + 1, non_warmup_results);
    let cache_calls = report_json
        .pointer("/cache/actual_evaluator_calls_excluding_warmups")
        .and_then(Value::as_u64)
        .expect("actual_evaluator_calls_excluding_warmups");
    assert_eq!(
        report_json
            .pointer("/cache/candidate_evaluations_executed")
            .and_then(Value::as_u64),
        Some(cache_calls)
    );

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(spec_path);
    let _ = std::fs::remove_file(output_path);
    let _ = std::fs::remove_file(report_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_family_controller_executes_runtime_path() {
    let dataset_path = temp_path("dataset-planner", ".bin");
    let spec_path = temp_path("spec-planner", ".json");
    let output_path = temp_path("output-planner", ".json");
    let report_path = temp_path("report-planner", ".json");
    std::fs::write(&dataset_path, b"planner-controller-dataset").expect("write dataset");

    let mut spec = sample_tune_spec(
        dataset_path.to_str().expect("dataset path"),
        output_path.to_str().expect("output path"),
        report_path.to_str().expect("report path"),
    );
    spec.controller = TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
        interface: planner_interface_for_baseline(&spec.baseline_candidate),
        planner_simulations_per_step: 8,
    });
    let bounds = spec.bounds.clone();
    let spec_json = SpecDocument::Tune(spec)
        .to_canonical_json()
        .expect("spec json");
    std::fs::write(&spec_path, spec_json).expect("write spec");
    let reward_cert_path = temp_path("reward-cert-planner", ".json");
    write_test_exact_reward_certificate(
        &reward_cert_path,
        &dataset_path,
        &bounds,
        "mc_aixi_fac_ctw",
    );

    let request = TuneCommandRequest {
        spec_path: spec_path.to_string_lossy().to_string(),
        emit_exact_reward_encoding_certificate: None,
        execution: TuneExecutionConfig {
            theorem: TuneTheoremConfig {
                exact_reward_encoding_certificate: Some(
                    reward_cert_path.to_string_lossy().to_string(),
                ),
                ..TuneTheoremConfig::default()
            },
            ..TuneExecutionConfig::default()
        },
    };
    run_tune(&request).expect("run tune");

    let report = std::fs::read_to_string(&report_path).expect("report exists");
    assert!(report.contains("\"status\": \"completed_mc_aixi_fac_ctw\""));
    assert!(report.contains("\"runtime_path\": \"finite_mutation_agent_bridge_mcaixi_fac_ctw\""));

    let _ = std::fs::remove_file(dataset_path);
    let _ = std::fs::remove_file(spec_path);
    let _ = std::fs::remove_file(output_path);
    let _ = std::fs::remove_file(report_path);
    let _ = std::fs::remove_file(reward_cert_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn executor_controls_are_excluded_from_canonical_tune_but_included_in_evaluator_profile() {
    let passive_dataset_path = temp_path("dataset-passive", ".bin");
    let trace_dataset_path = temp_path("dataset-trace", ".json");
    let prefix_dataset_path = temp_path("dataset-prefix", ".json");
    let output_path = temp_path("output-identity", ".json");
    let report_path = temp_path("report-identity", ".json");
    std::fs::write(&passive_dataset_path, b"identity-passive").expect("write passive");
    std::fs::write(
        &trace_dataset_path,
        causal_dataset_value(
            "identity-trace-codec",
            "events",
            serde_json::json!([
                {"kind": "context", "channel": "action", "bytes": [1]},
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [2, 3]}
            ]),
        )
        .to_string(),
    )
    .expect("write trace");
    std::fs::write(
        &prefix_dataset_path,
        causal_dataset_value(
            "identity-prefix-codec",
            "examples",
            serde_json::json!([
                {
                    "history": [{"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [7]}],
                    "action": [1],
                    "channel": "percept",
                    "domain": "bytes",
                    "target": [8],
                    "weight": 2.0
                }
            ]),
        )
        .to_string(),
    )
    .expect("write prefix");

    let dataset_paths = [
        passive_dataset_path.as_path(),
        trace_dataset_path.as_path(),
        prefix_dataset_path.as_path(),
    ];
    for dataset_path in dataset_paths {
        let mut base = sample_tune_spec(
            dataset_path.to_str().expect("dataset path"),
            output_path.to_str().expect("output path"),
            report_path.to_str().expect("report path"),
        );
        let interface = planner_interface_for_baseline(&base.baseline_candidate);
        let controllers = vec![
            TuneControllerSpec::AnnealedHillClimbing(AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: 1,
            }),
            TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
                interface: interface.clone(),
                planner_simulations_per_step: 2,
            }),
            TuneControllerSpec::AiqiDiscounted(AiqiDiscountedTuneControllerSpec {
                interface: interface.clone(),
                planner_simulations_per_step: 2,
                return_horizon: 1,
                return_bins: 2,
                discount_factor: 0.5,
                min_improvement: 0.0,
                max_improvement: 1.0,
            }),
            TuneControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhTuneControllerSpec {
                interface: interface.clone(),
                planner_simulations_per_step: 1,
                return_horizon: 1,
                warmstart_teacher_dataset_asset: "teacher".to_string(),
                label_phase_period: 1,
            }),
        ];
        for controller in controllers {
            base.controller = controller;
            base.assets.retain(|asset| asset.id == "dataset");
            if matches!(base.controller, TuneControllerSpec::AiqiWarmstartExactJh(_)) {
                base.assets.push(AssetBinding {
                    id: "teacher".to_string(),
                    path: passive_dataset_path.to_string_lossy().to_string(),
                });
            }
            let mut request_a = TuneCommandRequest {
                spec_path: "spec-a.json".to_string(),
                emit_exact_reward_encoding_certificate: None,
                execution: TuneExecutionConfig::default(),
            };
            let mut request_b = request_a.clone();
            request_b.spec_path = "spec-b.json".to_string();
            request_b.execution.max_evaluations = Some(1);
            request_b.execution.annealer_kernel_profile =
                AnnealerKernelProfile::CompiledUniformMetropolisHastings;
            request_b.execution.warmup_baseline_runs = 3;
            request_b.execution.diagnostic_chunk_bytes = Some(4096);
            request_b.execution.evaluator_cgroup_parent =
                Some("/sys/fs/cgroup/infotheory-tuner".to_string());
            request_b.execution.theorem.timing_certification_tier =
                TimingCertificationTier::RealTime;
            request_b.execution.theorem.determinism_deadline_certificate =
                Some("cert://deadline".to_string());

            let canonical_value_a = SpecDocument::Tune(base.clone())
                .to_canonical_json_value()
                .expect("canonical tune a");
            for field in [
                "max_evaluations",
                "annealer_kernel_profile",
                "cpu_affinity",
                "threads",
                "evaluator_worker_executable",
                "evaluator_cgroup_parent",
                "warmup_baseline_runs",
                "self_improvement_rounds",
                "stagnation_reset_evals",
                "log_path",
                "diagnostic_chunk_bytes",
                "rss_mode",
                "planner_deployable_model",
                "warmstart_trace_refresh",
                "theorem",
            ] {
                assert!(
                    canonical_value_a.get(field).is_none(),
                    "canonical tune document must not contain executor field '{field}'"
                );
            }
            let spec_path_a = temp_path("identity-spec-a", ".json");
            let spec_path_b = temp_path("identity-spec-b", ".json");
            let canonical_text = serde_json::to_vec(&canonical_value_a).expect("canonical json");
            std::fs::write(&spec_path_a, &canonical_text).expect("write spec a");
            std::fs::write(&spec_path_b, &canonical_text).expect("write spec b");
            request_a.spec_path = spec_path_a.to_string_lossy().to_string();
            request_b.spec_path = spec_path_b.to_string_lossy().to_string();
            let canonical_a = crate::spec::load_spec_document(&request_a.spec_path)
                .expect("load spec a")
                .validate()
                .expect("validate spec a")
                .canonical_bytes()
                .as_slice()
                .to_vec();
            let canonical_b = crate::spec::load_spec_document(&request_b.spec_path)
                .expect("load spec b")
                .validate()
                .expect("validate spec b")
                .canonical_bytes()
                .as_slice()
                .to_vec();
            assert_ne!(request_a.spec_path, request_b.spec_path);
            assert_ne!(request_a.execution, request_b.execution);
            assert_eq!(canonical_a, canonical_b);
            let _ = std::fs::remove_file(spec_path_a);
            let _ = std::fs::remove_file(spec_path_b);
            let loaded = load_dataset(dataset_path).expect("dataset mode loads");
            let profile_a = EvaluatorProfile {
                dataset_kind: loaded.kind,
                objective_target: loaded.objective_target,
                dataset_lowering_version: loaded.lowering_version,
                dataset_codec_hash: loaded.codec_hash.clone(),
                event_grammar_hash: loaded.event_grammar_hash.clone(),
                target_domain_support_hash: loaded.target_domain_support_hash.clone(),
                causal_header_profile_hash: loaded.causal_header_profile_hash.clone(),
                target_size_function: loaded.target_size_function,
                evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
                candidate_canonicalization_version: "bounds-v1".to_string(),
                warmup_baseline_runs: 0,
                diagnostic_chunk_bytes: None,
                eval_time_limit_seconds: base.eval_time_limit_seconds,
                evaluator_threads: 1,
                worker_isolation_mode: "spawn_exec_worker",
                worker_executable_identity: None,
                resolved_memory_accounting_kind: "unix_process_rss_fallback_explicit",
                resolved_memory_accounting_strict_theorem_facing: false,
                resolved_evaluator_cgroup_parent: None,
                backend_report_component_policy: "none",
                evaluator_determinism: "deterministic_under_h",
                rss_mode: PeakMemoryMode::ProcessRssPeak,
                timing_certification_tier: TimingCertificationTier::BestEffort,
                build_profile: "test",
                feature_set: vec!["test"],
            };
            let mut profile_b = profile_a.clone();
            profile_b.warmup_baseline_runs = request_b.execution.warmup_baseline_runs;
            profile_b.diagnostic_chunk_bytes = request_b.execution.diagnostic_chunk_bytes;
            profile_b.timing_certification_tier =
                request_b.execution.theorem.timing_certification_tier;
            assert_ne!(
                profile_a.hash().expect("profile a"),
                profile_b.hash().expect("profile b")
            );
        }
    }

    let _ = std::fs::remove_file(passive_dataset_path);
    let _ = std::fs::remove_file(trace_dataset_path);
    let _ = std::fs::remove_file(prefix_dataset_path);
}
// --- Group 1: Canonicalization and model-code properties ---

#[cfg(feature = "backend-ctw")]
fn sample_enabled_leaf_rate_backend_for_canonical_tests() -> Option<RateBackend> {
    crate::runtime::RATE_BACKEND_REGISTRY
        .iter()
        .filter(|descriptor| descriptor.enabled)
        .find_map(|descriptor| crate::runtime::default_rate_backend_spec(descriptor.kind))
}

#[cfg(feature = "backend-ctw")]
fn sample_roundtrip_compression_backends_for_canonical_tests() -> Vec<CompressionBackend> {
    let mut out = Vec::<CompressionBackend>::new();
    let leaf = sample_enabled_leaf_rate_backend_for_canonical_tests();
    for descriptor in crate::runtime::COMPRESSION_BACKEND_REGISTRY {
        if !descriptor.enabled {
            continue;
        }
        match descriptor.kind {
            crate::runtime::CompressionBackendKind::Zpaq => out.push(CompressionBackend::zpaq("5")),
            crate::runtime::CompressionBackendKind::RateAc => {
                if let Some(rate_backend) = leaf.clone() {
                    out.push(CompressionBackend::Rate {
                        rate_backend,
                        coder: crate::coders::CoderType::AC,
                        framing: FramingMode::Framed,
                    });
                }
            }
            crate::runtime::CompressionBackendKind::RateRans => {
                if let Some(rate_backend) = leaf.clone() {
                    out.push(CompressionBackend::Rate {
                        rate_backend,
                        coder: crate::coders::CoderType::RANS,
                        framing: FramingMode::Raw,
                    });
                }
            }
            #[cfg(feature = "backend-rwkv")]
            crate::runtime::CompressionBackendKind::Rwkv7 => {
                let opts = crate::spec::CompressionBackendShorthandOptions {
                    default_framing: crate::compression::FramingMode::Raw,
                    ..Default::default()
                };
                let rwkv = crate::spec::parse_compression_backend_name_method(
                    "rwkv7",
                    Some(
                        "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer",
                    ),
                    None,
                    &opts,
                )
                .expect("rwkv shorthand should parse");
                out.push(rwkv);
            }
            #[cfg(not(feature = "backend-rwkv"))]
            crate::runtime::CompressionBackendKind::Rwkv7 => {}
        }
    }
    out
}

#[cfg(feature = "backend-ctw")]
#[test]
fn canonical_code_roundtrip_and_idempotence_hold_on_enabled_corpus() {
    use std::path::Path;

    let corpus = sample_roundtrip_compression_backends_for_canonical_tests();
    assert!(
        !corpus.is_empty(),
        "at least one enabled compression backend must be available for canonicalization tests"
    );

    for candidate in corpus {
        let compiled = candidate.compile().expect("compile candidate");
        let canonical_bytes = compiled.canonical_bytes().as_slice().to_vec();
        let canonical_doc = SpecDocument::CompressionBackend(compiled.canonical_spec().clone());
        let canonical_doc_bytes = canonical_doc.to_binary();

        let reparsed = SpecDocument::from_binary(&canonical_doc_bytes, Path::new("."))
            .expect("parse canonical bytes");
        let recompiled = reparsed.compile().expect("compile reparsed document");
        let crate::spec::CompiledSpecDocument::CompressionBackend(recompiled_candidate) =
            recompiled
        else {
            panic!("expected compression backend document");
        };

        assert_eq!(
            canonical_bytes,
            recompiled_candidate.canonical_bytes().as_slice()
        );
        assert_eq!(
            canonical_doc.to_canonical_json().expect("canonical json"),
            SpecDocument::CompressionBackend(recompiled_candidate.canonical_spec().clone())
                .to_canonical_json()
                .expect("reparsed canonical json")
        );
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn canonical_code_image_is_injective_and_prefix_free_on_enabled_corpus() {
    let corpus = sample_roundtrip_compression_backends_for_canonical_tests();
    let mut image = std::collections::BTreeMap::<Vec<u8>, String>::new();

    for candidate in corpus {
        let compiled = candidate.compile().expect("compile candidate");
        let canonical_doc = SpecDocument::CompressionBackend(compiled.canonical_spec().clone());
        let bytes = compiled.canonical_bytes().as_slice().to_vec();
        let json = canonical_doc.to_canonical_json().expect("canonical json");

        if let Some(existing) = image.insert(bytes.clone(), json.clone()) {
            assert_eq!(
                existing, json,
                "equal canonical bytes must denote identical canonical candidate JSON"
            );
        }
    }

    let keys = image.keys().cloned().collect::<Vec<Vec<u8>>>();
    for i in 0..keys.len() {
        for j in 0..keys.len() {
            if i == j {
                continue;
            }
            assert!(
                !keys[i].starts_with(&keys[j]),
                "canonical code image must be prefix-free"
            );
        }
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn syntactic_aliases_canonicalize_to_same_model_code_length() {
    use crate::compression::FramingMode;
    let canonical_syntax = serde_json::json!({
        "kind": "rate-ac",
        "rate_backend": {"kind": "ctw", "depth": 16},
        "framing": "framed"
    });
    let alias_syntax = serde_json::json!({
        "kind": "rate-ac",
        "backend_spec": {"kind": "ctw"}
    });
    assert_ne!(canonical_syntax, alias_syntax);

    let z1_ast = crate::spec::parse_compression_backend_json(
        &canonical_syntax,
        std::path::Path::new("."),
        None,
        FramingMode::Framed,
    )
    .unwrap();
    let z2_ast = crate::spec::parse_compression_backend_json(
        &alias_syntax,
        std::path::Path::new("."),
        None,
        FramingMode::Framed,
    )
    .unwrap();
    let z1 = z1_ast.compile().unwrap();
    let z2 = z2_ast.compile().unwrap();
    assert_eq!(
        z1.canonical_bytes().as_slice(),
        z2.canonical_bytes().as_slice()
    );
    assert_eq!(z1.canonical_bytes().len(), z2.canonical_bytes().len());
    assert_eq!(
        8_usize * z1.canonical_bytes().len(),
        8_usize * z2.canonical_bytes().len()
    );
}

#[cfg(all(feature = "backend-ctw", feature = "backend-mixture"))]
#[test]
fn canonicalization_preserves_order_sensitivity_for_mixture_experts() {
    use std::sync::Arc;

    let left = CompressionBackend::Rate {
        rate_backend: RateBackend::Mixture {
            spec: Arc::new(crate::api::MixtureSpec::new(
                crate::api::MixtureKind::Bayes,
                vec![
                    crate::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                        .with_name("left"),
                    crate::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 6 })
                        .with_name("right"),
                ],
            )),
        },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let right = CompressionBackend::Rate {
        rate_backend: RateBackend::Mixture {
            spec: Arc::new(crate::api::MixtureSpec::new(
                crate::api::MixtureKind::Bayes,
                vec![
                    crate::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 6 })
                        .with_name("right"),
                    crate::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })
                        .with_name("left"),
                ],
            )),
        },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };

    let left_bytes = left
        .compile()
        .expect("compile left mixture")
        .canonical_bytes()
        .as_slice()
        .to_vec();
    let right_bytes = right
        .compile()
        .expect("compile right mixture")
        .canonical_bytes()
        .as_slice()
        .to_vec();
    assert_ne!(
        left_bytes, right_bytes,
        "expert order is semantic in order-sensitive mixture canonicalization paths"
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn canonicalization_is_insensitive_to_json_object_key_order() {
    use std::path::Path;

    let mut top_a = serde_json::Map::new();
    top_a.insert("kind".to_string(), serde_json::json!("rate-ac"));
    let mut rate_a = serde_json::Map::new();
    rate_a.insert("kind".to_string(), serde_json::json!("ctw"));
    rate_a.insert("depth".to_string(), serde_json::json!(8));
    top_a.insert("rate_backend".to_string(), Value::Object(rate_a));
    top_a.insert("framing".to_string(), serde_json::json!("framed"));

    let mut top_b = serde_json::Map::new();
    top_b.insert("framing".to_string(), serde_json::json!("framed"));
    let mut rate_b = serde_json::Map::new();
    rate_b.insert("depth".to_string(), serde_json::json!(8));
    rate_b.insert("kind".to_string(), serde_json::json!("ctw"));
    top_b.insert("rate_backend".to_string(), Value::Object(rate_b));
    top_b.insert("kind".to_string(), serde_json::json!("rate-ac"));

    let parsed_a = crate::spec::parse_compression_backend_json(
        &Value::Object(top_a),
        Path::new("."),
        None,
        FramingMode::Framed,
    )
    .expect("parse A");
    let parsed_b = crate::spec::parse_compression_backend_json(
        &Value::Object(top_b),
        Path::new("."),
        None,
        FramingMode::Framed,
    )
    .expect("parse B");

    let bytes_a = parsed_a
        .compile()
        .expect("compile A")
        .canonical_bytes()
        .as_slice()
        .to_vec();
    let bytes_b = parsed_b
        .compile()
        .expect("compile B")
        .canonical_bytes()
        .as_slice()
        .to_vec();
    assert_eq!(bytes_a, bytes_b);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn binary_canonical_deserializer_rejects_trailing_bytes() {
    use std::path::Path;

    let candidate = sample_roundtrip_compression_backends_for_canonical_tests()
        .into_iter()
        .next()
        .expect("candidate corpus must be non-empty");
    let compiled = candidate.compile().expect("compile candidate");
    let mut payload =
        SpecDocument::CompressionBackend(compiled.canonical_spec().clone()).to_binary();
    payload.extend_from_slice(&[0x00_u8, 0x01_u8]);

    let err = match SpecDocument::from_binary(&payload, Path::new(".")) {
        Ok(_) => panic!("trailing bytes must fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("unexpected trailing bytes"),
        "{err}"
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn cache_key_includes_effective_timeout() {
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    use crate::tuner::tests::temp_path;
    use crate::tuner::{EvaluatorProfile, cache_key_for_candidate};
    use std::time::Instant;
    let z = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 4 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .unwrap();

    let dataset_path = temp_path("dataset-cache-key", ".bin");
    std::fs::write(&dataset_path, b"test-data").unwrap();
    let dataset = crate::tuner::tests::load_dataset(&dataset_path).unwrap();
    let output_path = temp_path("cache-key-output", ".json");
    let report_path = temp_path("cache-key-report", ".json");
    let mut spec = sample_tune_spec(
        dataset_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
        report_path.to_str().unwrap(),
    );
    spec.eval_time_limit_seconds = 10.0;
    spec.time_budget_seconds = 60.0;
    let full_budget_compiled = spec.compile().unwrap();
    spec.time_budget_seconds = 9.999;
    let truncated_budget_compiled = spec.compile().unwrap();
    let tune_started = Instant::now();
    let full_effective_limit =
        effective_eval_limit_seconds(&full_budget_compiled, tune_started, Some(10.0), None);
    let truncated_effective_limit =
        effective_eval_limit_seconds(&truncated_budget_compiled, tune_started, Some(10.0), None);
    assert_eq!(full_effective_limit, 10.0);
    assert!(truncated_effective_limit > 0.0);
    assert!(truncated_effective_limit < full_effective_limit);

    let profile1 = EvaluatorProfile {
        dataset_kind: dataset.kind,
        objective_target: dataset.objective_target,
        dataset_lowering_version: dataset.lowering_version,
        dataset_codec_hash: dataset.codec_hash.clone(),
        event_grammar_hash: dataset.event_grammar_hash.clone(),
        target_domain_support_hash: dataset.target_domain_support_hash.clone(),
        causal_header_profile_hash: dataset.causal_header_profile_hash.clone(),
        target_size_function: dataset.target_size_function,
        evaluator_interface_version: crate::tuner::TUNER_EVALUATOR_INTERFACE_VERSION,
        candidate_canonicalization_version: "bounds-v1".to_string(),
        warmup_baseline_runs: 0,
        diagnostic_chunk_bytes: None,
        eval_time_limit_seconds: full_effective_limit,
        evaluator_threads: 1,
        worker_isolation_mode: "spawn_exec_worker",
        worker_executable_identity: None,
        resolved_memory_accounting_kind: "unix_process_rss_fallback_explicit",
        resolved_memory_accounting_strict_theorem_facing: false,
        resolved_evaluator_cgroup_parent: None,
        backend_report_component_policy: "none",
        evaluator_determinism: "deterministic_under_h",
        rss_mode: crate::tuner::PeakMemoryMode::ProcessRssPeak,
        timing_certification_tier: crate::tuner::TimingCertificationTier::BestEffort,
        build_profile: "unknown",
        feature_set: crate::tuner::compiled_feature_set(),
    };

    let mut profile2 = profile1.clone();
    profile2.eval_time_limit_seconds = truncated_effective_limit;

    let bytes = z.canonical_bytes().as_slice();
    let key1 = cache_key_for_candidate(bytes, &profile1, &dataset.canonical_content_hash).unwrap();
    let key2 = cache_key_for_candidate(bytes, &profile2, &dataset.canonical_content_hash).unwrap();
    assert_ne!(key1, key2);
    assert_eq!(
        key1.candidate_canonical_bytes,
        key2.candidate_canonical_bytes
    );
    assert_eq!(key1.dataset_identity, key2.dataset_identity);
    assert_ne!(key1.evaluator_profile_bytes, key2.evaluator_profile_bytes);
    assert_ne!(profile1.hash().unwrap(), profile2.hash().unwrap());
    let _ = std::fs::remove_file(dataset_path);
}

// --- Group 3: Deployability and objective-totalization properties ---

#[cfg(feature = "backend-ctw")]
#[test]
fn objective_totalizes_nondeployable_to_infinity_over_randomized_states() {
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    use crate::tuner::eval::evaluate_candidate;
    use crate::tuner::tests::temp_path;
    use crate::tuner::{
        CandidateEvalStatus, DeterministicEvaluatorRow, VerifiedDeterministicEvaluatorTable,
    };
    use std::collections::HashMap;

    let z1 = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 4 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .unwrap();
    let dataset_path = temp_path("dataset-dep", ".bin");
    std::fs::write(&dataset_path, b"1234567890123456").unwrap(); // 16 bytes
    let dataset = crate::tuner::tests::load_dataset(&dataset_path).unwrap();
    let crc = crate::tuner::crc32_hex(z1.canonical_bytes().as_slice());

    let mut rng = crate::tuner::RandomGenerator::new();
    let runtime_profile = crate::tuner::eval::ResolvedEvaluatorRuntimeProfile {
        worker_executable: None,
        worker_executable_identity: None,
        resolved_evaluator_cgroup_parent: None,
        memory_accounting_kind:
            crate::tuner::eval::ResolvedMemoryAccountingKind::DeterministicEvaluatorTable,
    };

    // Generative test over 100 random states
    for _ in 0..100 {
        let status = match rng.next_u64() % 4 {
            0 => CandidateEvalStatus::Success,
            1 => CandidateEvalStatus::Timeout,
            2 => CandidateEvalStatus::Invalid,
            _ => CandidateEvalStatus::Error,
        };
        // Generate random elapsed seconds between 0.001 and 10.0
        let elapsed = 0.001 + (rng.next_u64() as f64 / u64::MAX as f64) * 9.999;
        // Generate random peak memory up to 10MB
        let peak_mem = rng.next_u64() % 10_000_000;
        // fixed target_loss_bits for simplicity, it's valid
        let target_loss = 80.0;

        let mut rows = HashMap::new();
        rows.insert(
            crc.clone(),
            DeterministicEvaluatorRow {
                status: status.clone(),
                compressed_bytes: 10,
                elapsed_seconds: elapsed,
                peak_memory_bytes: peak_mem,
                target_loss_bits: target_loss,
            },
        );
        let table = VerifiedDeterministicEvaluatorTable {
            base: crate::tuner::VerifiedCertificate {
                ref_value: "ref".to_string(),
                content_hash: "hash".to_string(),
            },
            rows,
        };

        // Strict thresholds
        let min_tp = 50.0;
        let max_mem = 2000;

        let res = evaluate_candidate(
            &z1,
            &dataset,
            1,
            min_tp,
            max_mem,
            10.0,
            1,
            &runtime_profile,
            Some(&table),
        )
        .unwrap();

        let tp = 16.0 / elapsed;
        let is_deployable =
            matches!(status, CandidateEvalStatus::Success) && tp >= min_tp && peak_mem <= max_mem;

        assert_eq!(
            res.deployable, is_deployable,
            "deployable flag mismatch for status={:?}, tp={}, mem={}",
            status, tp, peak_mem
        );
        if is_deployable {
            assert!(
                res.objective_bits.is_finite(),
                "deployable candidate must have finite objective"
            );
        } else {
            assert_eq!(
                res.objective_bits,
                f64::INFINITY,
                "non-deployable candidate must totalize to infinity"
            );
        }
    }

    let _ = std::fs::remove_file(dataset_path);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn cache_key_is_exact_tuple_of_candidate_profile_and_dataset_identity() {
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    let z1 = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 4 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .unwrap();
    let z2 = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 5 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .unwrap();
    let profile1 = EvaluatorProfile {
        dataset_kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        dataset_lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        dataset_codec_hash: "passive-identity-bytes".to_string(),
        event_grammar_hash: "passive-target-only-byte-stream".to_string(),
        target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
        causal_header_profile_hash: crc32_hex(b"passive-none"),
        target_size_function: "passive-bytes-len",
        evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
        candidate_canonicalization_version: "bounds-v1".to_string(),
        warmup_baseline_runs: 0,
        diagnostic_chunk_bytes: None,
        eval_time_limit_seconds: 1.0,
        evaluator_threads: 1,
        worker_isolation_mode: "spawn_exec_worker",
        worker_executable_identity: None,
        resolved_memory_accounting_kind: "unix_process_rss_fallback_explicit",
        resolved_memory_accounting_strict_theorem_facing: false,
        resolved_evaluator_cgroup_parent: None,
        backend_report_component_policy: "none",
        evaluator_determinism: "deterministic_under_h",
        rss_mode: PeakMemoryMode::ProcessRssPeak,
        timing_certification_tier: TimingCertificationTier::BestEffort,
        build_profile: "test",
        feature_set: vec!["test"],
    };
    let mut profile2 = profile1.clone();
    profile2.eval_time_limit_seconds = 2.0;
    let dataset1 = crc32_hex(b"dataset-one");
    let dataset2 = crc32_hex(b"dataset-two");
    let z1_bytes = z1.canonical_bytes().as_slice();
    let z2_bytes = z2.canonical_bytes().as_slice();

    let key = cache_key_for_candidate(z1_bytes, &profile1, &dataset1).unwrap();
    let same = cache_key_for_candidate(z1_bytes, &profile1, &dataset1).unwrap();
    let changed_candidate = cache_key_for_candidate(z2_bytes, &profile1, &dataset1).unwrap();
    let changed_profile = cache_key_for_candidate(z1_bytes, &profile2, &dataset1).unwrap();
    let changed_dataset = cache_key_for_candidate(z1_bytes, &profile1, &dataset2).unwrap();

    assert_eq!(key, same);
    assert_ne!(key, changed_candidate);
    assert_ne!(key, changed_profile);
    assert_ne!(key, changed_dataset);
    assert_eq!(key.candidate_canonical_bytes, z1_bytes);
    assert_eq!(key.dataset_identity, dataset1);
    assert_ne!(
        key.evaluator_profile_bytes,
        changed_profile.evaluator_profile_bytes
    );
}

// --- Group 5: Causal evaluator semantic properties ---

#[cfg(feature = "backend-ctw")]
#[test]
fn observe_target_no_score_contributes_zero_bits() {
    // Property: the action context field (ObserveTargetNoScore) conditions the
    // predictor state but must itself contribute exactly zero bits to target_loss_bits.
    //
    // Pure semantic proof without heuristic numeric bounds:
    // Evaluate Dataset A: [Observe(0)], Dataset B: [Target(0)],
    // and Dataset C: [Observe(0), Target(0)].
    //
    // We assert:
    // - loss(A) == 0.0 (an ObserveTargetNoScore event literally costs 0.0 bits)
    // - loss(C) != loss(B), proving the replay event changed future predictor state.
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    use crate::tuner::eval::evaluate_candidate_causal_loss;
    use crate::tuner::tests::{causal_dataset_value, temp_path};

    let z = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    }
    .compile()
    .unwrap();

    let path_a = temp_path("causal-obs-a", ".json");
    std::fs::write(
        &path_a,
        causal_dataset_value(
            "json-causal-byte-events-v1",
            "events",
            serde_json::json!([
                { "kind": "observe_target_no_score", "channel": "percept", "domain": "binary", "bytes": [0] }
            ]),
        )
        .to_string(),
    )
    .unwrap();
    let path_b = temp_path("causal-target-b", ".json");
    std::fs::write(
        &path_b,
        causal_dataset_value(
            "json-causal-byte-events-v1",
            "events",
            serde_json::json!([
                { "kind": "target", "channel": "percept", "domain": "binary", "bytes": [0] }
            ]),
        )
        .to_string(),
    )
    .unwrap();
    let path_c = temp_path("causal-obs-c", ".json");
    std::fs::write(
        &path_c,
        causal_dataset_value(
            "json-causal-byte-events-v1",
            "events",
            serde_json::json!([
                { "kind": "observe_target_no_score", "channel": "percept", "domain": "binary", "bytes": [0] },
                { "kind": "target", "channel": "percept", "domain": "binary", "bytes": [0] }
            ]),
        )
        .to_string(),
    )
    .unwrap();

    let ds_a = crate::tuner::tests::load_dataset(&path_a).unwrap();
    let ds_b = crate::tuner::tests::load_dataset(&path_b).unwrap();
    let ds_c = crate::tuner::tests::load_dataset(&path_c).unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let res_a = evaluate_candidate_causal_loss(&z, &ds_a, deadline).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let res_b = evaluate_candidate_causal_loss(&z, &ds_b, deadline).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let res_c = evaluate_candidate_causal_loss(&z, &ds_c, deadline).unwrap();

    let loss_a: f64 = res_a.1;
    let loss_b: f64 = res_b.1;
    let loss_c: f64 = res_c.1;

    assert_eq!(
        loss_a, 0.0,
        "An isolated ObserveTargetNoScore event must contribute exactly 0.0 bits"
    );
    assert!(
        loss_b.is_finite() && loss_c.is_finite(),
        "charged target losses must be finite for the binary target-domain test"
    );
    assert_ne!(
        loss_b.to_bits(),
        loss_c.to_bits(),
        "ObserveTargetNoScore must update future predictor state; otherwise replay+target would equal target-only"
    );

    let _ = std::fs::remove_file(path_a);
    let _ = std::fs::remove_file(path_b);
    let _ = std::fs::remove_file(path_c);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn exact_mdl_map_equivalence_over_finite_semantic_class() {
    let candidates = [
        (16.0_f64, 2.0_f64.powi(-20)),
        (24.0_f64, 2.0_f64.powi(-6)),
        (32.0_f64, 2.0_f64.powi(-1)),
        (8.0_f64, 0.0_f64),
    ];
    let objectives = candidates
        .iter()
        .map(|(model_bits, likelihood)| {
            if *likelihood == 0.0 {
                f64::INFINITY
            } else {
                *model_bits - likelihood.log2()
            }
        })
        .collect::<Vec<_>>();
    let posterior_scores = candidates
        .iter()
        .map(|(model_bits, likelihood)| 2.0_f64.powf(-*model_bits) * *likelihood)
        .collect::<Vec<_>>();
    let prior_mass = candidates
        .iter()
        .map(|(model_bits, _)| 2.0_f64.powf(-*model_bits))
        .sum::<f64>();
    let normalized_scores = candidates
        .iter()
        .map(|(model_bits, likelihood)| (2.0_f64.powf(-*model_bits) / prior_mass) * *likelihood)
        .collect::<Vec<_>>();
    let argmin_objective = objectives
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap();
    let argmax_posterior = posterior_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap();
    let argmax_normalized = normalized_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap();
    let mixture_codelength = -posterior_scores.iter().sum::<f64>().log2();
    let best_objective = objectives[argmin_objective];

    assert_eq!(argmin_objective, argmax_posterior);
    assert_eq!(argmin_objective, argmax_normalized);
    assert!(mixture_codelength <= best_objective);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn finite_incumbent_oracle_enforces_monotone_key_and_objective() {
    fn eval(objective_bits: f64, deployable: bool) -> CandidateEvalResult {
        CandidateEvalResult {
            status: if deployable {
                CandidateEvalStatus::Success
            } else {
                CandidateEvalStatus::Timeout
            },
            compressed_bytes: 0,
            elapsed_seconds: 1.0,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: if deployable { 1.0 } else { 0.0 },
            peak_memory_bytes: 0,
            target_loss_bits: objective_bits,
            objective_bits,
            deployable,
        }
    }

    let mut best_eval = eval(10.0, true);
    let mut best_bytes = vec![0x20];
    let mut updates = vec![(best_eval.clone(), best_bytes.clone())];
    let candidates = vec![
        (eval(f64::INFINITY, false), vec![0x00]),
        (eval(12.0, true), vec![0x00]),
        (eval(8.0, true), vec![0xff]),
        (eval(8.0, true), vec![0x01]),
        (eval(9.0, true), vec![0x00]),
        (eval(5.0, true), vec![0x80]),
    ];

    for (candidate_eval, candidate_bytes) in candidates {
        let old_best_eval = best_eval.clone();
        let old_best_bytes = best_bytes.clone();
        let should_update = candidate_eval.deployable
            && key_less(&candidate_eval, &candidate_bytes, &best_eval, &best_bytes);

        if should_update {
            assert!(candidate_eval.objective_bits <= old_best_eval.objective_bits);
            if candidate_eval.objective_bits == old_best_eval.objective_bits {
                assert!(candidate_bytes < old_best_bytes);
            }
            best_eval = candidate_eval;
            best_bytes = candidate_bytes;
            updates.push((best_eval.clone(), best_bytes.clone()));
        } else {
            assert_eq!(
                best_eval.objective_bits.to_bits(),
                old_best_eval.objective_bits.to_bits()
            );
            assert_eq!(best_bytes, old_best_bytes);
        }
    }

    for pair in updates.windows(2) {
        let (previous_eval, previous_bytes) = &pair[0];
        let (next_eval, next_bytes) = &pair[1];
        assert!(key_less(
            next_eval,
            next_bytes,
            previous_eval,
            previous_bytes
        ));
        assert!(next_eval.objective_bits <= previous_eval.objective_bits);
    }

    let accepted_current_eval = eval(7.0, true);
    let accepted_current_bytes = vec![0x00];
    assert!(!key_less(
        &accepted_current_eval,
        &accepted_current_bytes,
        &best_eval,
        &best_bytes
    ));
    assert!(accepted_current_eval.objective_bits > best_eval.objective_bits);
    assert_eq!(best_eval.objective_bits, 5.0);
    assert_eq!(best_bytes, vec![0x80]);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn exact_reward_encoder_telescopes_incumbent_objective_decreases() {
    fn eval(objective_bits: f64, deployable: bool) -> CandidateEvalResult {
        CandidateEvalResult {
            status: if deployable {
                CandidateEvalStatus::Success
            } else {
                CandidateEvalStatus::Timeout
            },
            compressed_bytes: 0,
            elapsed_seconds: 1.0,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: if deployable { 1.0 } else { 0.0 },
            peak_memory_bytes: 0,
            target_loss_bits: objective_bits,
            objective_bits,
            deployable,
        }
    }

    let encoder = TunerRewardEncoder::ExactIntegerObjectiveDifference {
        max_reward: 20,
        objective_difference_to_symbol: None,
    };
    for reward in 0..=20 {
        assert_eq!(encoder.encode(reward as f64).unwrap(), reward);
    }

    let baseline_eval = eval(10.0, true);
    let baseline_bytes = vec![0x20];
    let mut best_eval = baseline_eval.clone();
    let mut best_bytes = baseline_bytes;
    let mut decoded_reward_sum = 0_i64;
    let candidates = vec![
        (eval(12.0, true), vec![0x00]),
        (eval(7.0, true), vec![0xff]),
        (eval(7.0, true), vec![0x01]),
        (eval(f64::INFINITY, false), vec![0x00]),
        (eval(4.0, true), vec![0x80]),
    ];

    for (candidate_eval, candidate_bytes) in candidates {
        let improves_best = candidate_eval.deployable
            && key_less(&candidate_eval, &candidate_bytes, &best_eval, &best_bytes);
        let raw_improvement = if improves_best {
            (best_eval.objective_bits - candidate_eval.objective_bits).max(0.0)
        } else {
            0.0
        };
        let reward = encoder.encode(raw_improvement).unwrap();
        decoded_reward_sum = decoded_reward_sum.saturating_add(reward);
        if improves_best {
            best_eval = candidate_eval;
            best_bytes = candidate_bytes;
        }
    }

    assert_eq!(
        decoded_reward_sum as f64,
        baseline_eval.objective_bits - best_eval.objective_bits
    );
    assert_eq!(decoded_reward_sum, 6);
    assert_eq!(best_eval.objective_bits, 4.0);
}

#[cfg(feature = "backend-ctw")]
#[test]
fn normalized_clipped_improvement_stays_in_unit_interval_and_rejects_degenerate_bounds() {
    assert_eq!(normalized_clipped_improvement(-1.0, 0.0, 2.0).unwrap(), 0.0);
    assert_eq!(normalized_clipped_improvement(1.0, 0.0, 2.0).unwrap(), 0.5);
    assert_eq!(normalized_clipped_improvement(3.0, 0.0, 2.0).unwrap(), 1.0);
    assert!(normalized_clipped_improvement(1.0, 2.0, 2.0).is_err());

    let encoder = TunerRewardEncoder::NormalizedClipped {
        min_improvement: 0.0,
        max_improvement: 2.0,
        max_reward: 10,
    };
    assert_eq!(encoder.encode(-1.0).unwrap(), 0);
    assert_eq!(encoder.encode(1.0).unwrap(), 5);
    assert_eq!(encoder.encode(3.0).unwrap(), 10);
}

// --- Group 6: Reversible elementary kernel properties ---

#[cfg(feature = "backend-ctw")]
#[test]
fn inactive_radius_moves_become_self_loops() {
    // Property: out-of-radius and boundary-crossing moves must not appear in the
    // transition kernel — the move is silently absent (a self-loop in MH terms),
    // never clipped to the nearest valid value.
    //
    // We test two distinct cases:
    //
    // Case 1 — active_radius=0: no move with any magnitude is within radius, so
    //   the entire transition map must be empty and sampling yields Exhausted.
    //
    // Case 2 — boundary self-loop: candidate is at the lower bound (depth=1), so
    //   the downward delta=-1 move (depth=0) violates the [1,16] bounds and must
    //   be ABSENT from transitions. The upward delta=+1 move (depth=2) is valid
    //   and must be PRESENT. This proves boundary moves self-loop rather than clip.
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    use crate::spec::TuneParameterRangeSpec;
    use crate::tuner::annealer::{
        apply_integer_descriptor, collect_numeric_leaves, compile_canonical_proposal_kernel,
        integer_leaf_bounds,
    };
    use crate::tuner::tests::sample_tune_spec;

    let mut bounds = sample_tune_spec("", "", "").bounds;
    bounds.parameter_ranges = vec![TuneParameterRangeSpec {
        parameter: "rate_backend.depth".to_string(),
        min: 1.0,
        max: 16.0,
    }];
    let env = crate::spec::SpecEnvironment::new(std::path::Path::new("."));

    // --- Case 1: active_radius = 0 yields an empty kernel (Exhausted). ---
    let candidate_mid = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 4 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let compiled_mid = candidate_mid.compile().unwrap();
    let bytes_mid = compiled_mid.canonical_bytes().as_slice();
    let kernel_zero_radius =
        compile_canonical_proposal_kernel(&candidate_mid, &bounds, 3, 0, &env, bytes_mid).unwrap();
    assert!(
        kernel_zero_radius.transitions.is_empty(),
        "active_radius=0 must produce an empty transition map"
    );
    let mut rng = crate::tuner::RandomGenerator::new();
    let draw = crate::tuner::annealer::sample_annealed_proposal(
        &candidate_mid,
        &bounds,
        3,
        0,
        &env,
        &mut rng,
    )
    .unwrap();
    assert!(
        matches!(draw, crate::tuner::AnnealedProposalDraw::Exhausted),
        "sampling from empty kernel must yield Exhausted"
    );

    // --- Case 2: boundary self-loop — candidate at lower bound (depth=1). ---
    // With active_radius=1 and max_mutation_radius=3, only magnitude-1 moves are
    // within radius.  depth=1 is the lower bound, so delta=-1 → depth=0 is out of
    // [1,16] and must be absent.  delta=+1 → depth=2 is valid and must be present.
    let candidate_lb = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 1 }, // at lower bound
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let compiled_lb = candidate_lb.compile().unwrap();
    let bytes_lb = compiled_lb.canonical_bytes().as_slice().to_vec();
    let kernel_lb =
        compile_canonical_proposal_kernel(&candidate_lb, &bounds, 3, 1, &env, &bytes_lb).unwrap();
    assert_eq!(
        kernel_lb.total_raw_actions, 6,
        "kernel raw action space must retain inactive and boundary self-loop descriptors"
    );
    assert_eq!(
        kernel_lb.transitions.len(),
        1,
        "at the lower bound with active radius 1, only the valid upward move may be emitted"
    );
    assert_eq!(
        kernel_lb.proposal_mass_to_canonical_bytes(&bytes_lb),
        0,
        "self-loops must not be emitted as explicit current-candidate transitions"
    );

    assert!(
        kernel_lb
            .transitions
            .iter()
            .all(|t| t.candidate_canonical_bytes != bytes_lb),
        "boundary and inactive self-loops must remain implicit, not emitted as clipped current transitions"
    );

    // The upward move (depth=2) MUST appear in transitions.
    let depth2_candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 2 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let depth2_bytes = depth2_candidate
        .compile()
        .unwrap()
        .canonical_bytes()
        .as_slice()
        .to_vec();
    let has_depth2 = kernel_lb
        .transitions
        .iter()
        .any(|t| t.candidate_canonical_bytes == depth2_bytes);
    assert!(
        has_depth2,
        "depth=2 (valid +1 from lower bound) must be present in transition kernel"
    );

    // Direct application: apply_integer_descriptor must return false for the
    // boundary-crossing delta, confirming no implicit clipping occurs.
    let json_lb = crate::spec::compression_backend_to_json_value(&candidate_lb).unwrap();
    let leaves = collect_numeric_leaves(&json_lb);
    let depth_leaf = leaves
        .iter()
        .find(|l| l.path.contains("depth"))
        .expect("depth leaf");
    let (min_b, max_b) = integer_leaf_bounds(depth_leaf.kind, Some((1.0, 16.0))).unwrap();
    let mut json_mut = json_lb.clone();
    let applied_down = apply_integer_descriptor(
        &mut json_mut,
        depth_leaf,
        depth_leaf.kind,
        min_b,
        max_b,
        1,
        -1,
    );
    assert!(
        !applied_down,
        "apply_integer_descriptor must return false for depth 1 + delta -1 (out of bounds)"
    );
    assert_eq!(
        json_mut, json_lb,
        "failed boundary move must leave the candidate JSON unchanged rather than clipping"
    );
    let mut json_mut2 = json_lb.clone();
    let applied_up = apply_integer_descriptor(
        &mut json_mut2,
        depth_leaf,
        depth_leaf.kind,
        min_b,
        max_b,
        1,
        1,
    );
    assert!(
        applied_up,
        "apply_integer_descriptor must return true for depth 1 + delta +1 (valid move)"
    );
}

#[cfg(feature = "backend-rosa")]
#[test]
fn signed_range_keeps_integer_kind_stable_across_zero_for_reversibility() {
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    use crate::spec::{TuneBoundsSpec, TuneParameterRangeSpec};
    use crate::tuner::annealer::{compile_canonical_proposal_kernel, sample_annealed_proposal};

    let candidate = CompressionBackend::Rate {
        rate_backend: RateBackend::RosaPlus { max_order: -1 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let bounds = TuneBoundsSpec {
        allowed_backends: vec!["rosaplus".to_string()],
        forbidden_backends: Vec::new(),
        parameter_ranges: vec![TuneParameterRangeSpec {
            parameter: "rate_backend.max_order".to_string(),
            min: -1.0,
            max: 8.0,
        }],
        max_experts: 2,
        max_mixture_nesting_depth: 1,
        min_experts: Some(1),
        allow_duplicate_experts: Some(false),
        required_experts: Vec::new(),
        forbidden_expert_pairs: Vec::new(),
    };
    let env = crate::spec::SpecEnvironment::new(".");

    let current = candidate.compile_in(&env).expect("compile current");
    let current_bytes = current.canonical_bytes().as_slice().to_vec();
    let kernel = compile_canonical_proposal_kernel(&candidate, &bounds, 1, 1, &env, &current_bytes)
        .expect("compile proposal kernel");
    assert!(
        !kernel.transitions.is_empty(),
        "kernel must include at least one transition from max_order=-1"
    );

    let target_zero = CompressionBackend::Rate {
        rate_backend: RateBackend::RosaPlus { max_order: 0 },
        coder: crate::coders::CoderType::AC,
        framing: FramingMode::Framed,
    };
    let target_zero_bytes = target_zero
        .compile_in(&env)
        .expect("compile max_order=0")
        .canonical_bytes()
        .as_slice()
        .to_vec();
    let zero_proposal = kernel
        .transitions
        .iter()
        .find(|proposal| proposal.candidate_canonical_bytes == target_zero_bytes)
        .expect("expected transition from max_order=-1 to max_order=0");

    let reverse =
        compile_canonical_proposal_kernel(&target_zero, &bounds, 1, 1, &env, &target_zero_bytes)
            .expect("compile reverse kernel");
    let reverse_mass = reverse.proposal_mass_to_canonical_bytes(&current_bytes);
    assert_eq!(
        reverse_mass, zero_proposal.raw_action_count,
        "reverse proposal mass must match forward raw action count across -1 <-> 0 boundary"
    );

    let mut rng = crate::tuner::RandomGenerator::new();
    let _ = sample_annealed_proposal(&candidate, &bounds, 1, 1, &env, &mut rng)
        .expect("proposal sampling should not fail on signed boundary transition");
}

#[cfg(feature = "backend-ctw")]
#[test]
fn finite_evaluator_table_oracle_selects_minimum_deployable_jh() {
    use crate::api::{CompressionBackend, RateBackend};
    use crate::compression::FramingMode;
    use crate::tuner::tests::temp_path;
    use crate::tuner::{
        CandidateEvalResult, CandidateEvalStatus, DeterministicEvaluatorRow,
        VerifiedDeterministicEvaluatorTable,
    };
    use std::collections::HashMap;

    let ds_path = temp_path("oracle", ".bin");
    std::fs::write(&ds_path, b"test").unwrap();
    let loaded = crate::tuner::tests::load_dataset(&ds_path).unwrap();

    let candidates = (1_usize..=8)
        .map(|depth| {
            (
                depth,
                CompressionBackend::Rate {
                    rate_backend: RateBackend::Ctw { depth },
                    coder: crate::coders::CoderType::AC,
                    framing: FramingMode::Framed,
                }
                .compile()
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let mut rows = HashMap::<String, DeterministicEvaluatorRow>::new();
    for (depth, candidate) in &candidates {
        let target_loss_bits = match depth {
            1 => 130.0,
            2 => 120.0,
            3 => 90.0,
            4 => 80.0,
            5 => 70.0,
            6 => 20.0,
            7 => 1.0,
            8 => 50.0,
            _ => unreachable!(),
        };
        let peak_memory_bytes = if *depth == 7 { 10_000 } else { 100 };
        rows.insert(
            crate::tuner::crc32_hex(candidate.canonical_bytes().as_slice()),
            DeterministicEvaluatorRow {
                status: CandidateEvalStatus::Success,
                compressed_bytes: 10,
                target_loss_bits,
                elapsed_seconds: 0.01,
                peak_memory_bytes,
            },
        );
    }
    let table = VerifiedDeterministicEvaluatorTable {
        base: crate::tuner::VerifiedCertificate {
            ref_value: "test://deterministic-table".to_string(),
            content_hash: "00000000".to_string(),
        },
        rows,
    };

    let max_memory_bytes: u64 = 1_000;
    let mut production_best: Option<(usize, CandidateEvalResult, Vec<u8>)> = None;
    let mut reference_best: Option<(usize, f64, Vec<u8>)> = None;
    for (depth, candidate) in &candidates {
        let model_bytes = candidate.canonical_bytes().len();
        let eval = table
            .evaluate(candidate, &loaded, model_bytes, 1.0, max_memory_bytes, 1.0)
            .unwrap();
        let candidate_bytes = candidate.canonical_bytes().as_slice().to_vec();
        let reference_objective = if eval.status == CandidateEvalStatus::Success
            && eval.throughput_bytes_per_second >= 1.0
            && eval.peak_memory_bytes <= max_memory_bytes
        {
            (model_bytes as f64 * 8.0) + eval.target_loss_bits
        } else {
            f64::INFINITY
        };

        if production_best
            .as_ref()
            .map(|(_, best_eval, best_bytes)| {
                key_less(&eval, &candidate_bytes, best_eval, best_bytes)
            })
            .unwrap_or(true)
        {
            production_best = Some((*depth, eval.clone(), candidate_bytes.clone()));
        }
        if reference_best
            .as_ref()
            .map(|(_, best_objective, best_bytes)| {
                reference_objective < *best_objective
                    || (reference_objective == *best_objective && candidate_bytes < *best_bytes)
            })
            .unwrap_or(true)
        {
            reference_best = Some((*depth, reference_objective, candidate_bytes));
        }
    }

    let production_best = production_best.unwrap();
    let reference_best = reference_best.unwrap();
    assert_eq!(production_best.0, reference_best.0);
    assert_eq!(production_best.0, 6);
    assert!(production_best.1.deployable);

    let _ = std::fs::remove_file(ds_path);
}
