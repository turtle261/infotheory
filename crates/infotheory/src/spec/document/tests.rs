//! Tests for canonical top-level specification documents.

use super::*;
use crate::aixi::common::ObservationKeyMode;
use crate::api::{CompressionBackend, RateBackend};

#[cfg(feature = "backend-ctw")]
fn sample_planner_run() -> PlannerRunSpec {
    PlannerRunSpec {
        assets: Vec::new(),
        environment: EnvironmentSpec::Builtin {
            builtin: BuiltinEnvironmentSpec::CoinFlip,
        },
        interface: PlannerInterfaceSpec {
            observation_bits: 1,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 1,
            agent_actions: 2,
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
        },
        controller: ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
            predictor: RateBackend::Ctw { depth: 8 },
            predictor_max_order: 8,
            discount_gamma: 0.99,
            return_horizon: 2,
            return_bins: 8,
            augmentation_period: 2,
            history_prune_keep_steps: None,
            baseline_exploration: 0.01,
        }),
        runtime: PlannerRuntimeSpec {
            random_seed: Some(7),
            learn_cycles: Some(4),
            eval_cycles: Some(2),
            terminate_lifetime: 4,
            log_every: 1,
            perf: false,
            vm_perf_only: false,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        },
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_json_roundtrip_is_stable() {
    let spec = sample_planner_run();
    let expected = spec.to_canonical_json().expect("json");
    let value = spec.to_canonical_json_value().expect("json value");
    let reparsed = SpecDocument::parse_json_value(&value, Path::new(".")).expect("parse");
    match reparsed {
        SpecDocument::PlannerRun(parsed) => {
            assert_eq!(parsed.to_canonical_json().expect("parsed json"), expected)
        }
        _ => panic!("expected planner run document"),
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_binary_roundtrip_is_stable() {
    let spec = sample_planner_run();
    let expected = spec.to_canonical_json().expect("json");
    let bytes = SpecDocument::PlannerRun(spec.clone()).to_binary();
    let reparsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary");
    match reparsed {
        SpecDocument::PlannerRun(parsed) => {
            assert_eq!(parsed.to_canonical_json().expect("parsed json"), expected)
        }
        _ => panic!("expected planner run document"),
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn staged_pipeline_matches_direct_planner_compile() {
    let spec = sample_planner_run();
    let value = SpecDocument::PlannerRun(spec.clone())
        .to_canonical_json_value()
        .expect("planner run json value");
    let parsed =
        SpecDocument::parse_json_value_staged(&value, Path::new(".")).expect("staged parse");
    let validated = parsed.validate().expect("staged validate");
    let compiled = validated.compile().expect("staged compile");
    let direct = spec.compile().expect("direct compile");

    match compiled {
        CompiledSpecDocument::PlannerRun(staged) => {
            assert_eq!(
                staged.canonical_bytes().as_slice(),
                direct.canonical_bytes().as_slice()
            );
        }
        _ => panic!("expected compiled planner-run document"),
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn staged_pipeline_supports_standalone_backend_documents() {
    let document = SpecDocument::RateBackend(RateBackend::Ctw { depth: 8 });
    let expected_json = document.to_canonical_json().expect("document json");
    let value = document
        .to_canonical_json_value()
        .expect("document json value");

    let parsed =
        SpecDocument::parse_json_value_staged(&value, Path::new(".")).expect("staged parse");
    let validated = parsed.validate().expect("staged validate");
    assert!(!validated.canonical_bytes().is_empty());
    let compiled = validated.compile().expect("staged compile");
    match compiled {
        CompiledSpecDocument::RateBackend(compiled_backend) => {
            let reparsed = SpecDocument::RateBackend(compiled_backend.canonical_spec().clone())
                .to_canonical_json()
                .expect("compiled canonical json");
            assert_eq!(reparsed, expected_json);
        }
        _ => panic!("expected compiled rate-backend document"),
    }

    let bytes = document.to_binary();
    let binary_parsed =
        SpecDocument::from_binary_staged(&bytes, Path::new(".")).expect("staged binary parse");
    let binary_compiled = binary_parsed
        .validate()
        .expect("staged binary validate")
        .compile()
        .expect("staged binary compile");
    match binary_compiled {
        CompiledSpecDocument::RateBackend(compiled_backend) => {
            assert!(matches!(
                compiled_backend.canonical_spec(),
                RateBackend::Ctw { depth: 8 }
            ));
        }
        _ => panic!("expected compiled rate-backend document from binary"),
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn standalone_backend_documents_roundtrip_without_embedded_json_fragments() {
    let rate = SpecDocument::RateBackend(RateBackend::Ctw { depth: 8 });
    let compression = SpecDocument::CompressionBackend(CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: crate::compression::FramingMode::Framed,
    });

    for doc in [rate, compression] {
        let expected = doc.to_canonical_json().expect("json");
        let bytes = doc.to_binary();
        assert!(
            !bytes
                .windows(br#""kind""#.len())
                .any(|window| window == br#""kind""#),
            "binary document should not embed canonical JSON object keys: {bytes:?}"
        );
        let reparsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary");
        assert_eq!(reparsed.to_canonical_json().expect("parsed json"), expected);
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_compile_exposes_compiled_predictor_and_action_bits() {
    let compiled = sample_planner_run()
        .compile()
        .expect("compiled planner run");
    assert_eq!(compiled.action_bits(), 1);
    match compiled.controller() {
        CompiledPlannerController::AiqiDiscounted { predictor, .. } => {
            assert!(matches!(
                predictor.canonical_spec(),
                RateBackend::Ctw { depth: 8 }
            ));
        }
        _ => panic!("expected compiled aiqi controller"),
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_compile_rejects_unrepresentable_reward_ranges() {
    let mut spec = sample_planner_run();
    spec.interface.reward_bits = 1;
    spec.interface.min_reward = 0;
    spec.interface.max_reward = 100;
    spec.interface.reward_offset = 0;
    let err = match spec.compile() {
        Ok(_) => panic!("unrepresentable rewards must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("reward_bits too small"), "{err}");
}

#[cfg(all(feature = "backend-ctw", feature = "backend-zpaq"))]
#[test]
fn planner_run_compile_rejects_mcaixi_predictors_with_zpaq_conditioning() {
    let mut spec = sample_planner_run();
    spec.controller = ControllerSpec::McAixi(McAixiControllerSpec {
        predictor: RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("1"),
        },
        predictor_max_order: 8,
        agent_horizon: 1,
        num_simulations: 1,
        exploration_exploitation_ratio: 1.0,
        discount_gamma: 1.0,
    });
    let err = match spec.compile() {
        Ok(_) => panic!("MC-AIXI zpaq backend must fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("reversible action conditioning"),
        "{err}"
    );
}

#[cfg(all(feature = "backend-ctw", feature = "backend-zpaq"))]
#[test]
fn planner_run_compile_rejects_aiqi_predictors_without_frozen_conditioning() {
    let mut spec = sample_planner_run();
    spec.controller = ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
        predictor: RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("1"),
        },
        predictor_max_order: 8,
        discount_gamma: 0.99,
        return_horizon: 2,
        return_bins: 8,
        augmentation_period: 2,
        history_prune_keep_steps: None,
        baseline_exploration: 0.01,
    });
    let err = match spec.compile() {
        Ok(_) => panic!("AIQI zpaq backend must fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("strict frozen conditioning"),
        "{err}"
    );
}

#[cfg(all(feature = "backend-ctw", feature = "vm"))]
fn sample_vm_planner_run() -> PlannerRunSpec {
    PlannerRunSpec {
        assets: vec![AssetBinding {
            id: "firecracker".to_string(),
            path: "dummy-firecracker.json".to_string(),
        }],
        environment: EnvironmentSpec::NyxVm(VmEnvironmentSpec {
            firecracker_config_asset: "firecracker".to_string(),
            instance_id: "vm-test".to_string(),
            shared_region_name: "shared".to_string(),
            shared_region_size: 4096,
            shared_memory_policy: SharedMemoryPolicySpec::Snapshot,
            step_timeout_ms: 100,
            boot_timeout_ms: 1_000,
            episode_steps: 4,
            step_cost: 0,
            observation_policy: VmObservationPolicySpec::OutputHash,
            observation_bits: 8,
            observation_stream_len: 16,
            observation_stream_mode: VmObservationStreamModeSpec::PadTruncate,
            observation_pad_byte: 0,
            reward_bits: 8,
            reward_policy: VmRewardPolicySpec::FromGuest,
            reward_shaping: None,
            action_source: VmRuntimeActionSourceSpec::Fuzz {
                seeds: vec!["seed".to_string()],
                encoding: VmPayloadEncodingSpec::Utf8,
                mutators: vec![VmFuzzMutatorSpec::FlipBit, VmFuzzMutatorSpec::SpliceSeed],
                min_len: 1,
                max_len: 16,
                dictionary: vec!["tok".to_string()],
                rng_seed: 7,
            },
            action_filter: None,
            action_prefix: "ACT ".to_string(),
            action_suffix: "\n".to_string(),
            obs_prefix: "OBS ".to_string(),
            rew_prefix: "REW ".to_string(),
            done_prefix: "DONE ".to_string(),
            data_prefix: "DATA ".to_string(),
            wire_encoding: VmPayloadEncodingSpec::Utf8,
            stats_backend: RateBackend::Ctw { depth: 8 },
            trace: None,
            debug_mode: false,
            crash_log: None,
        }),
        interface: PlannerInterfaceSpec {
            observation_bits: 8,
            observation_stream_len: 16,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 8,
            agent_actions: 1,
            min_reward: 0,
            max_reward: 255,
            reward_offset: 0,
        },
        controller: ControllerSpec::McAixi(McAixiControllerSpec {
            predictor: RateBackend::Ctw { depth: 8 },
            predictor_max_order: 8,
            agent_horizon: 1,
            num_simulations: 1,
            exploration_exploitation_ratio: 1.0,
            discount_gamma: 1.0,
        }),
        runtime: PlannerRuntimeSpec {
            random_seed: Some(7),
            learn_cycles: Some(1),
            eval_cycles: Some(0),
            terminate_lifetime: 1,
            log_every: 1,
            perf: false,
            vm_perf_only: false,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        },
    }
}

#[cfg(all(feature = "backend-ctw", feature = "vm"))]
#[test]
fn planner_run_compile_normalizes_vm_aliases_to_canonical_names() {
    let compiled = sample_vm_planner_run()
        .compile()
        .expect("vm planner run should compile");
    let EnvironmentSpec::NyxVm(vm) = &compiled.canonical_spec().environment else {
        panic!("expected vm environment");
    };
    assert_eq!(vm.observation_policy, VmObservationPolicySpec::OutputHash);
    assert_eq!(
        vm.observation_stream_mode,
        VmObservationStreamModeSpec::PadTruncate
    );
    assert_eq!(vm.wire_encoding, VmPayloadEncodingSpec::Utf8);
    match &vm.action_source {
        VmRuntimeActionSourceSpec::Fuzz {
            encoding, mutators, ..
        } => {
            assert_eq!(*encoding, VmPayloadEncodingSpec::Utf8);
            assert_eq!(
                mutators,
                &vec![VmFuzzMutatorSpec::FlipBit, VmFuzzMutatorSpec::SpliceSeed]
            );
        }
        other => panic!("expected fuzz action source, got {other:?}"),
    }
}

#[cfg(all(feature = "backend-ctw", feature = "vm"))]
#[test]
fn planner_run_binary_roundtrip_preserves_vm_fuzz_action_source_layout() {
    let spec = sample_vm_planner_run();
    let expected_json = SpecDocument::PlannerRun(spec.clone())
        .to_canonical_json()
        .expect("json");
    let bytes = SpecDocument::PlannerRun(spec.clone()).to_binary();
    let reparsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary");
    let reparsed_json = reparsed.to_canonical_json().expect("json");

    assert_eq!(reparsed_json, expected_json);

    let SpecDocument::PlannerRun(reparsed_spec) = reparsed else {
        panic!("expected planner_run document");
    };

    let EnvironmentSpec::NyxVm(vm) = reparsed_spec.environment else {
        panic!("expected vm environment");
    };
    match vm.action_source {
        VmRuntimeActionSourceSpec::Fuzz {
            seeds,
            encoding,
            mutators,
            min_len,
            max_len,
            dictionary,
            rng_seed,
        } => {
            assert_eq!(seeds, vec!["seed".to_string()]);
            assert_eq!(encoding, VmPayloadEncodingSpec::Utf8);
            assert_eq!(
                mutators,
                vec![VmFuzzMutatorSpec::FlipBit, VmFuzzMutatorSpec::SpliceSeed]
            );
            assert_eq!(min_len, 1);
            assert_eq!(max_len, 16);
            assert_eq!(dictionary, vec!["tok".to_string()]);
            assert_eq!(rng_seed, 7);
        }
        other => panic!("expected fuzz action source, got {other:?}"),
    }
}

#[cfg(all(feature = "backend-ctw", feature = "vm"))]
#[test]
fn planner_run_compile_rejects_unknown_vm_enum_names() {
    let unknown = sample_vm_planner_run();
    let mut value = unknown.to_canonical_json_value().expect("canonical json");
    value["environment"] = serde_json::json!({
        "kind": "nyx_vm",
        "firecracker_config_asset": "firecracker",
        "observation_policy": "nope"
    });
    let err = match SpecDocument::parse_json_value(&value, Path::new(".")) {
        Ok(_) => panic!("unknown observation policy must fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("unknown VM observation_policy"),
        "{err}"
    );

    let mut unknown_encoding_value = sample_vm_planner_run()
        .to_canonical_json_value()
        .expect("canonical json");
    unknown_encoding_value["environment"]["protocol"]["wire_encoding"] =
        serde_json::json!("base64");
    let err = match SpecDocument::parse_json_value(&unknown_encoding_value, Path::new(".")) {
        Ok(_) => panic!("unknown wire encoding must fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("unknown VM payload encoding"),
        "{err}"
    );

    let mut unknown_mutator_value = sample_vm_planner_run()
        .to_canonical_json_value()
        .expect("canonical json");
    unknown_mutator_value["environment"]["action_source"]["mutators"] =
        serde_json::json!(["invalid-mutator"]);
    let err = match SpecDocument::parse_json_value(&unknown_mutator_value, Path::new(".")) {
        Ok(_) => panic!("unknown mutator must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("unknown VM fuzz mutator"), "{err}");
}

#[cfg(feature = "backend-ctw")]
#[test]
fn tune_document_binary_roundtrip_is_stable() {
    let spec = TuneSpec {
        assets: vec![AssetBinding {
            id: "dataset".to_string(),
            path: "input.bin".to_string(),
        }],
        input_asset: "dataset".to_string(),
        baseline_candidate: CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Framed,
        },
        controller: TuneControllerSpec::AnnealedHillClimbing(
            AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: 2,
            },
        ),
        bounds: TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string()],
            forbidden_backends: vec!["zpaq".to_string()],
            parameter_ranges: vec![TuneParameterRangeSpec {
                parameter: "mixture.alpha".to_string(),
                min: 0.1,
                max: 0.5,
            }],
            max_experts: 4,
            max_mixture_nesting_depth: 2,
            min_experts: Some(1),
            allow_duplicate_experts: Some(false),
            required_experts: vec!["ctw".to_string()],
            forbidden_expert_pairs: vec![],
        },
        eval_time_limit_seconds: 1.0,
        time_budget_seconds: 10.0,
        min_throughput_bytes_per_second: 1024.0,
        max_memory_bytes: 1 << 20,
        output_config_path: "best.json".to_string(),
        seed: 7,
        report_path: Some("report.json".to_string()),
    };
    let expected = spec.to_canonical_json().expect("json");
    let bytes = SpecDocument::Tune(spec.clone()).to_binary();
    let reparsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary");
    match reparsed {
        SpecDocument::Tune(parsed) => {
            assert_eq!(parsed.to_canonical_json().expect("parsed json"), expected)
        }
        _ => panic!("expected tune document"),
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn tune_compile_model_bytes_ignore_outer_request_controls() {
    let base = TuneSpec {
        assets: vec![AssetBinding {
            id: "dataset".to_string(),
            path: "input.bin".to_string(),
        }],
        input_asset: "dataset".to_string(),
        baseline_candidate: CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Framed,
        },
        controller: TuneControllerSpec::AnnealedHillClimbing(
            AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: 2,
            },
        ),
        bounds: TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string()],
            forbidden_backends: vec![],
            parameter_ranges: vec![],
            max_experts: 4,
            max_mixture_nesting_depth: 2,
            min_experts: Some(1),
            allow_duplicate_experts: Some(false),
            required_experts: vec![],
            forbidden_expert_pairs: vec![],
        },
        eval_time_limit_seconds: 1.0,
        time_budget_seconds: 10.0,
        min_throughput_bytes_per_second: 1024.0,
        max_memory_bytes: 1 << 20,
        output_config_path: "best-a.json".to_string(),
        seed: 7,
        report_path: Some("report-a.json".to_string()),
    };
    let mut other = base.clone();
    other.output_config_path = "best-b.json".to_string();
    other.report_path = Some("report-b.json".to_string());

    let compiled_a = base.compile().expect("compiled tune a");
    let compiled_b = other.compile().expect("compiled tune b");
    assert_eq!(
        compiled_a.baseline_candidate().canonical_bytes().as_slice(),
        compiled_b.baseline_candidate().canonical_bytes().as_slice()
    );
    assert_eq!(
        compiled_a.baseline_candidate_model_bytes(),
        compiled_b.baseline_candidate_model_bytes()
    );
}

#[cfg(not(feature = "backend-ctw"))]
#[test]
fn planner_run_validation_reports_missing_backend_feature() {
    let spec = PlannerRunSpec {
        assets: Vec::new(),
        environment: EnvironmentSpec::Builtin {
            builtin: BuiltinEnvironmentSpec::CoinFlip,
        },
        interface: PlannerInterfaceSpec {
            observation_bits: 1,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 1,
            agent_actions: 2,
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
        },
        controller: ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
            predictor: RateBackend::Ctw { depth: 8 },
            predictor_max_order: 8,
            discount_gamma: 0.99,
            return_horizon: 2,
            return_bins: 8,
            augmentation_period: 2,
            history_prune_keep_steps: None,
            baseline_exploration: 0.01,
        }),
        runtime: PlannerRuntimeSpec {
            random_seed: Some(7),
            learn_cycles: Some(4),
            eval_cycles: Some(2),
            terminate_lifetime: 4,
            log_every: 1,
            perf: false,
            vm_perf_only: false,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        },
    };
    let err = spec.validate().expect_err("missing feature should fail");
    assert!(
        err.to_string()
            .contains("requires infotheory feature 'backend-ctw'"),
        "{err}"
    );
}
