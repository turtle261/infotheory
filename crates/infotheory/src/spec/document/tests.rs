//! Tests for canonical top-level specification documents.

use super::*;
use crate::aixi::common::{
    DEFAULT_RANDOM_SEED, MctsStrategy, ObservationKeyMode,
    parallel_uct_workers_one_warning_count_for_tests,
    reset_parallel_uct_workers_one_warning_for_tests,
};
#[cfg(feature = "backend-ctw")]
use crate::api::CompressionBackend;
use crate::api::RateBackend;
use crate::spec::CanonicalJson;
#[cfg(feature = "backend-ctw")]
use std::num::NonZeroUsize;

#[cfg(feature = "backend-ctw")]
fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("test fixture worker count must be non-zero")
}

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
fn sample_mc_aixi_planner_run(mcts_strategy: MctsStrategy) -> PlannerRunSpec {
    let mut spec = sample_planner_run();
    spec.controller = ControllerSpec::McAixi(McAixiControllerSpec {
        predictor: RateBackend::Ctw { depth: 8 },
        predictor_max_order: 8,
        agent_horizon: 2,
        num_simulations: 4,
        mcts_strategy,
        exploration_exploitation_ratio: 1.0,
        discount_gamma: 0.95,
    });
    spec
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_parser_accepts_canonical_builtin_names() {
    let names = [
        ("coin_flip", BuiltinEnvironmentSpec::CoinFlip),
        (
            "biased_rock_paper_scissor",
            BuiltinEnvironmentSpec::BiasedRockPaperScissor,
        ),
        ("kuhn_poker", BuiltinEnvironmentSpec::KuhnPoker),
        ("extended_tiger", BuiltinEnvironmentSpec::ExtendedTiger),
        ("tic_tac_toe", BuiltinEnvironmentSpec::TicTacToe),
        ("blackjack", BuiltinEnvironmentSpec::Blackjack),
        ("platformer", BuiltinEnvironmentSpec::Platformer),
    ];

    for (name, expected_builtin) in names {
        let mut value = sample_planner_run()
            .to_canonical_json_value()
            .expect("planner run json");
        value["environment"]["name"] = serde_json::Value::String(name.to_string());

        let parsed = SpecDocument::parse_json_value(&value, Path::new(".")).expect("parse");
        let SpecDocument::PlannerRun(planner_run) = parsed else {
            panic!("expected planner run document for builtin '{name}'");
        };
        match planner_run.environment {
            EnvironmentSpec::Builtin { builtin } => assert_eq!(builtin, expected_builtin),
            #[cfg(feature = "vm")]
            EnvironmentSpec::NyxVm(_) => {
                panic!("expected builtin environment for alias '{name}'");
            }
        }
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_parser_rejects_noncanonical_builtin_names() {
    for name in [
        "coin-flip",
        "biased_coinflip",
        "biased_rps",
        "kuhn-poker",
        "extended-poker",
        "extended_poker",
        "extended-tiger",
        "tic-tac-toe",
        "tictactoe",
    ] {
        let mut value = sample_planner_run()
            .to_canonical_json_value()
            .expect("planner run json");
        value["environment"]["name"] = serde_json::Value::String(name.to_string());

        let err = match SpecDocument::parse_json_value(&value, Path::new(".")) {
            Ok(_) => panic!("noncanonical builtin name '{name}' must be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("unknown builtin environment"),
            "unexpected parser error for '{name}': {err}"
        );
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_binary_roundtrip_covers_canonical_builtins() {
    let builtins = [
        BuiltinEnvironmentSpec::CoinFlip,
        BuiltinEnvironmentSpec::BiasedRockPaperScissor,
        BuiltinEnvironmentSpec::KuhnPoker,
        BuiltinEnvironmentSpec::ExtendedTiger,
        BuiltinEnvironmentSpec::TicTacToe,
        BuiltinEnvironmentSpec::Blackjack,
        BuiltinEnvironmentSpec::Platformer,
    ];

    for builtin in builtins {
        let mut spec = sample_planner_run();
        spec.environment = EnvironmentSpec::Builtin { builtin };
        let bytes = SpecDocument::PlannerRun(spec).to_binary();
        let parsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary parse");
        let SpecDocument::PlannerRun(parsed_run) = parsed else {
            panic!("expected planner run document for builtin {builtin:?}");
        };
        match parsed_run.environment {
            EnvironmentSpec::Builtin {
                builtin: parsed_builtin,
            } => assert_eq!(parsed_builtin, builtin),
            #[cfg(feature = "vm")]
            EnvironmentSpec::NyxVm(_) => {
                panic!("expected builtin environment for builtin {builtin:?}");
            }
        }
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
fn mc_aixi_missing_mcts_strategy_canonicalizes_to_explicit_rho_uct() {
    let spec = sample_mc_aixi_planner_run(MctsStrategy::RhoUct);
    let mut value = SpecDocument::PlannerRun(spec)
        .to_canonical_json_value()
        .expect("canonical json value");
    value["controller"]
        .as_object_mut()
        .expect("controller object")
        .remove("mcts_strategy");

    let parsed = SpecDocument::parse_json_value(&value, Path::new(".")).expect("parse");
    let SpecDocument::PlannerRun(parsed_run) = parsed else {
        panic!("expected planner run document");
    };
    let ControllerSpec::McAixi(inner) = &parsed_run.controller else {
        panic!("expected MC-AIXI controller");
    };
    assert_eq!(inner.mcts_strategy, MctsStrategy::RhoUct);

    let canonical = parsed_run
        .to_canonical_json_value()
        .expect("canonical json value");
    assert_eq!(
        canonical["controller"]["mcts_strategy"],
        serde_json::json!({ "kind": "rho_uct" })
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn mc_aixi_parallel_uct_binary_roundtrip_preserves_strategy() {
    let spec = sample_mc_aixi_planner_run(MctsStrategy::ParallelUct {
        workers: nz(16),
        bu_uct_m_max: Some(0.8),
    });
    let expected = SpecDocument::PlannerRun(spec.clone())
        .to_canonical_json_value()
        .expect("canonical json value");
    let bytes = SpecDocument::PlannerRun(spec).to_binary();
    let reparsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary parse");
    let SpecDocument::PlannerRun(parsed_run) = reparsed else {
        panic!("expected planner run document");
    };
    assert_eq!(
        parsed_run
            .to_canonical_json_value()
            .expect("canonical json value"),
        expected
    );
    let ControllerSpec::McAixi(inner) = parsed_run.controller else {
        panic!("expected MC-AIXI controller");
    };
    assert_eq!(
        inner.mcts_strategy,
        MctsStrategy::ParallelUct {
            workers: nz(16),
            bu_uct_m_max: Some(0.8),
        }
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn mc_aixi_parallel_uct_json_roundtrip_preserves_strategy() {
    let spec = sample_mc_aixi_planner_run(MctsStrategy::ParallelUct {
        workers: nz(16),
        bu_uct_m_max: None,
    });
    let expected = SpecDocument::PlannerRun(spec.clone())
        .to_canonical_json_value()
        .expect("canonical json value");
    let parsed =
        SpecDocument::parse_json_value(&expected, Path::new(".")).expect("canonical json parse");
    let SpecDocument::PlannerRun(parsed_run) = parsed else {
        panic!("expected planner run document");
    };
    assert_eq!(
        parsed_run
            .to_canonical_json_value()
            .expect("canonical json value"),
        expected
    );
    let ControllerSpec::McAixi(inner) = parsed_run.controller else {
        panic!("expected MC-AIXI controller");
    };
    assert_eq!(
        inner.mcts_strategy,
        MctsStrategy::ParallelUct {
            workers: nz(16),
            bu_uct_m_max: None,
        }
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn mc_aixi_parallel_uct_parser_rejects_zero_workers_in_canonical_json() {
    // `workers == 0` is type-prevented in `MctsStrategy::ParallelUct` itself
    // (`NonZeroUsize`), so the only surface where it can still be expressed
    // is the document layer. Verify the canonical-JSON parser rejects it
    // with a stable, label-prefixed error message.
    let mut spec = sample_mc_aixi_planner_run(MctsStrategy::ParallelUct {
        workers: nz(1),
        bu_uct_m_max: None,
    });
    // Take a valid canonical JSON value, then mutate `workers` to 0.
    let mut value = SpecDocument::PlannerRun(spec.clone())
        .to_canonical_json_value()
        .expect("canonical json value");
    value["controller"]["mcts_strategy"]["workers"] = serde_json::json!(0);
    let err = match SpecDocument::parse_json_value(&value, Path::new(".")) {
        Ok(_) => panic!("workers=0 must be rejected at parse time"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("controller.mcts_strategy.workers must be >= 1"),
        "{err}"
    );
    // Sanity check: an unrelated mutation (validating `bu_uct_m_max`) still
    // routes through the spec-pipeline validation layer.
    spec.controller = ControllerSpec::McAixi(McAixiControllerSpec {
        predictor: RateBackend::Ctw { depth: 8 },
        predictor_max_order: 8,
        agent_horizon: 2,
        num_simulations: 4,
        mcts_strategy: MctsStrategy::ParallelUct {
            workers: nz(4),
            bu_uct_m_max: Some(1.0),
        },
        exploration_exploitation_ratio: 1.0,
        discount_gamma: 0.95,
    });
    let err = match spec.compile() {
        Ok(_) => panic!("invalid bu_uct_m_max must fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("controller.mcts_strategy.bu_uct_m_max must be in (0, 1)"),
        "{err}"
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn mc_aixi_parallel_uct_rejects_invalid_bu_threshold() {
    for invalid in [0.0, 1.0, -0.25, 1.25] {
        let spec = sample_mc_aixi_planner_run(MctsStrategy::ParallelUct {
            workers: nz(4),
            bu_uct_m_max: Some(invalid),
        });
        let err = match spec.compile() {
            Ok(_) => panic!("invalid bu_uct_m_max={invalid} must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("controller.mcts_strategy.bu_uct_m_max must be in (0, 1)"),
            "{err}"
        );
    }
}

#[cfg(feature = "backend-ctw")]
#[test]
fn mc_aixi_parallel_uct_canonical_json_rejects_string_shorthand_for_rho_uct() {
    // The canonical schema requires the object form for every strategy. The
    // serializer always emits `{ "kind": "rho_uct" }` (or the parallel_uct
    // object), so the parser must symmetrically refuse string shorthands.
    let mut value = SpecDocument::PlannerRun(sample_mc_aixi_planner_run(MctsStrategy::RhoUct))
        .to_canonical_json_value()
        .expect("canonical json value");
    value["controller"]["mcts_strategy"] = serde_json::json!("rho_uct");
    let err = match SpecDocument::parse_json_value(&value, Path::new(".")) {
        Ok(_) => panic!("string shorthand must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("controller.mcts_strategy must be an object with a 'kind' field"),
        "{err}"
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn mc_aixi_parallel_uct_workers_one_warns_once_across_repeated_initialization() {
    reset_parallel_uct_workers_one_warning_for_tests();
    let spec = sample_mc_aixi_planner_run(MctsStrategy::ParallelUct {
        workers: nz(1),
        bu_uct_m_max: None,
    });

    spec.compile().expect("workers=1 should compile");
    assert_eq!(
        parallel_uct_workers_one_warning_count_for_tests(),
        1,
        "workers=1 should emit exactly one warning during first initialization"
    );

    spec.compile()
        .expect("workers=1 should keep compiling on subsequent initialization");
    assert_eq!(
        parallel_uct_workers_one_warning_count_for_tests(),
        1,
        "workers=1 warning must remain one-time across repeated initialization"
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_omitted_runtime_seed_canonicalizes_to_default_seed() {
    let mut spec = sample_planner_run();
    spec.runtime.random_seed = None;

    let compiled = spec.compile().expect("compile");
    assert_eq!(
        compiled.runtime().random_seed,
        Some(DEFAULT_RANDOM_SEED),
        "runtime.random_seed should canonicalize to deterministic default",
    );
    assert_eq!(
        compiled.canonical_spec().runtime.random_seed,
        Some(DEFAULT_RANDOM_SEED),
        "canonical spec should preserve the resolved default seed",
    );

    let canonical_value = compiled
        .canonical_spec()
        .to_canonical_json_value()
        .expect("canonical json value");
    assert_eq!(
        canonical_value["runtime"]["random_seed"],
        serde_json::Value::from(DEFAULT_RANDOM_SEED),
        "canonical JSON should expose resolved runtime.random_seed",
    );
}

#[cfg(feature = "backend-ctw")]
#[test]
fn planner_run_resolved_seed_survives_binary_roundtrip() {
    let mut spec = sample_planner_run();
    spec.runtime.random_seed = None;

    let compiled = spec.compile().expect("compile");
    let bytes = SpecDocument::PlannerRun(compiled.canonical_spec().clone()).to_binary();
    let parsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("from binary");
    let SpecDocument::PlannerRun(roundtripped) = parsed else {
        panic!("expected planner_run document");
    };
    let roundtripped_compiled = roundtripped.compile().expect("recompile");
    assert_eq!(
        roundtripped_compiled.runtime().random_seed,
        Some(DEFAULT_RANDOM_SEED),
    );
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
        mcts_strategy: MctsStrategy::RhoUct,
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
            mcts_strategy: MctsStrategy::RhoUct,
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
            builtin: BuiltinEnvironmentSpec::TicTacToe,
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

#[test]
fn builtin_environment_canonical_names_round_trip() {
    use BuiltinEnvironmentSpec::*;
    let cases: &[(BuiltinEnvironmentSpec, &str)] = &[
        (CoinFlip, "coin_flip"),
        (BiasedRockPaperScissor, "biased_rock_paper_scissor"),
        (KuhnPoker, "kuhn_poker"),
        (ExtendedTiger, "extended_tiger"),
        (TicTacToe, "tic_tac_toe"),
        (Blackjack, "blackjack"),
        (Platformer, "platformer"),
    ];
    for (variant, expected) in cases {
        assert_eq!(
            variant.canonical_name(),
            *expected,
            "canonical_name() for {variant:?}"
        );
    }
}

#[test]
fn spec_document_kind_str_matches_serialized_kind_field() {
    use crate::api::{CompressionBackend, RateBackend};
    use crate::coders::CoderType;
    use crate::compression::FramingMode;

    let rate = SpecDocument::RateBackend(RateBackend::Ctw { depth: 4 });
    assert_eq!(rate.kind_str(), "rate_backend");

    let compression = SpecDocument::CompressionBackend(CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 4 },
        coder: CoderType::AC,
        framing: FramingMode::Framed,
    });
    assert_eq!(compression.kind_str(), "compression_backend");
}

#[cfg(feature = "backend-ctw")]
#[test]
fn environment_spec_kind_str_is_stable() {
    let spec = EnvironmentSpec::Builtin {
        builtin: BuiltinEnvironmentSpec::CoinFlip,
    };
    assert_eq!(spec.kind_str(), "builtin");
}
