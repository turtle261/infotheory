//! Tests for canonical top-level specification documents.

use super::*;
#[cfg(any(feature = "backend-ctw", feature = "all-backends"))]
use crate::aixi::common::MctsStrategy;
use crate::aixi::common::{ActionAlphabet, ObservationKeyMode};
#[cfg(feature = "backend-ctw")]
use crate::aixi::common::{
    DEFAULT_RANDOM_SEED, parallel_uct_workers_one_warning_count_for_tests,
    reset_parallel_uct_workers_one_warning_for_tests,
};
#[cfg(feature = "backend-ctw")]
use crate::api::CompressionBackend;
use crate::api::RateBackend;
#[cfg(feature = "all-backends")]
use crate::api::{
    CalibratedSpec, CalibrationContextKind, MixtureExpertSpec, MixtureKind, MixtureScheduleMode,
    MixtureSpec, ParticleSpec,
};
#[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
use crate::backends::llm_policy::{
    LlmPolicy, OptimizerHyperParams, OptimizerKind, PolicyAction, PolicyRule, PositionExpr,
    RepeatRule, RepeatSegment, ScheduleRule, TrainAction, TrainScopeSet,
};
#[cfg(feature = "backend-ctw")]
use std::num::NonZeroUsize;
#[cfg(feature = "all-backends")]
use std::sync::Arc;

#[cfg(feature = "backend-ctw")]
fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("test fixture worker count must be non-zero")
}

fn action_alphabet(n: usize) -> ActionAlphabet {
    ActionAlphabet::try_from_usize(n).expect("test fixture action alphabet must be non-zero")
}

#[cfg(feature = "backend-ctw")]
fn sample_tune_spec() -> TuneSpec {
    TuneSpec {
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
    }
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
            agent_actions: action_alphabet(2),
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
        },
        controller: ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
            predictor: RateBackend::Ctw { depth: 8 },
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
fn planner_run_parser_rejects_zero_action_alphabet() {
    let mut value = sample_planner_run()
        .to_canonical_json_value()
        .expect("planner run json");
    value["interface"]["agent_actions"] = serde_json::json!(0);

    let err = match SpecDocument::parse_json_value(&value, Path::new(".")) {
        Ok(_) => panic!("agent_actions=0 must be rejected at parse time"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("interface.agent_actions must be >= 1"),
        "unexpected parser error: {err}"
    );
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
fn staged_document_and_compiled_planner_accessors_preserve_metadata() {
    let base_dir = Path::new("/tmp/infotheory-stage-planner");
    let spec = sample_planner_run();
    let value = SpecDocument::PlannerRun(spec.clone())
        .to_canonical_json_value()
        .expect("planner json value");
    let parsed =
        SpecDocument::parse_json_value_staged(&value, base_dir).expect("staged planner parse");

    assert_eq!(parsed.base_dir(), base_dir);
    assert!(matches!(parsed.document(), SpecDocument::PlannerRun(_)));
    assert!(matches!(
        parsed.clone().into_document(),
        SpecDocument::PlannerRun(_)
    ));

    let validated = parsed.validate().expect("staged planner validate");
    assert!(!validated.canonical_bytes().is_empty());
    let compiled_doc = validated.compile().expect("staged planner compile");
    assert_eq!(
        compiled_doc.canonical_bytes().as_slice(),
        validated.canonical_bytes().as_slice()
    );

    match compiled_doc {
        CompiledSpecDocument::PlannerRun(compiled) => {
            assert_eq!(
                compiled
                    .canonical_spec()
                    .to_canonical_json()
                    .expect("canonical json"),
                spec.compile()
                    .expect("direct planner compile")
                    .canonical_spec()
                    .to_canonical_json()
                    .expect("direct canonical json")
            );
            assert_eq!(compiled.resolved_assets().len(), 0);
            assert_eq!(compiled.interface().agent_actions.get(), 2);
            assert_eq!(compiled.runtime().random_seed, Some(7));
            assert_eq!(compiled.resolved_random_seed(), 7);
            assert_eq!(compiled.action_bits(), 1);
            assert_eq!(compiled.controller().kind_str(), "aiqi_discounted");
            assert_eq!(compiled.controller().backend_label(), "ctw(depth=8)");
        }
        _ => panic!("expected compiled planner document"),
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
fn tune_validation_and_compilation_accessors_surface_baseline_metadata() {
    let spec = sample_tune_spec();
    let validated = spec.validate().expect("validated tune spec");
    assert_eq!(validated.canonical_spec().input_asset, "dataset");
    assert!(!validated.canonical_bytes().is_empty());

    let compiled = validated.compile().expect("compiled tune spec");
    assert_eq!(compiled.canonical_spec().input_asset, "dataset");
    assert_eq!(compiled.resolved_assets().len(), 1);
    assert_eq!(compiled.resolved_assets()[0].id, "dataset");
    let crate::spec::AssetRef::Filesystem(path) = &compiled.resolved_assets()[0].asset;
    assert!(path.ends_with("input.bin"));
    assert!(matches!(
        compiled.controller(),
        CompiledTuneController::AnnealedHillClimbing(_)
    ));
    assert_eq!(compiled.candidate_canonicalization_version(), "bounds-v1");
    assert_eq!(
        compiled.baseline_candidate_model_bytes(),
        compiled.baseline_candidate().canonical_bytes().len()
    );
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

#[cfg(feature = "all-backends")]
#[test]
fn standalone_rate_backend_documents_cover_all_binary_backend_tags() {
    let rate_docs = vec![
        SpecDocument::RateBackend(RateBackend::RosaPlus { max_order: 32 }),
        SpecDocument::RateBackend(RateBackend::Match {
            hash_bits: 18,
            min_len: 2,
            max_len: 32,
            base_mix: 0.05,
            confidence_scale: 1.0,
        }),
        SpecDocument::RateBackend(RateBackend::SparseMatch {
            hash_bits: 18,
            min_len: 2,
            max_len: 32,
            gap_min: 1,
            gap_max: 4,
            base_mix: 0.05,
            confidence_scale: 1.0,
        }),
        SpecDocument::RateBackend(RateBackend::Ppmd {
            order: 6,
            memory_mb: 8,
        }),
        SpecDocument::RateBackend(RateBackend::Sequitur { context_bytes: 64 }),
        SpecDocument::RateBackend(RateBackend::Ctw { depth: 12 }),
        SpecDocument::RateBackend(RateBackend::FacCtw {
            base_depth: 10,
            num_percept_bits: 8,
            encoding_bits: 1,
        }),
        SpecDocument::RateBackend(RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("1"),
        }),
        SpecDocument::RateBackend(RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![MixtureExpertSpec::new(RateBackend::Ctw { depth: 6 })],
            )),
        }),
        SpecDocument::RateBackend(RateBackend::Mixture {
            spec: Arc::new(
                MixtureSpec::new(
                    MixtureKind::FadingBayes,
                    vec![MixtureExpertSpec::new(RateBackend::Match {
                        hash_bits: 16,
                        min_len: 2,
                        max_len: 16,
                        base_mix: 0.05,
                        confidence_scale: 1.0,
                    })],
                )
                .with_decay(0.97),
            ),
        }),
        SpecDocument::RateBackend(RateBackend::Mixture {
            spec: Arc::new(
                MixtureSpec::new(
                    MixtureKind::Switching,
                    vec![MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })],
                )
                .with_schedule(MixtureScheduleMode::Theorem),
            ),
        }),
        SpecDocument::RateBackend(RateBackend::Mixture {
            spec: Arc::new(
                MixtureSpec::new(
                    MixtureKind::Convex,
                    vec![MixtureExpertSpec::new(RateBackend::Ppmd {
                        order: 5,
                        memory_mb: 4,
                    })],
                )
                .with_alpha(1.25)
                .with_schedule(MixtureScheduleMode::Theorem),
            ),
        }),
        SpecDocument::RateBackend(RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Mdl,
                vec![MixtureExpertSpec::new(RateBackend::Sequitur {
                    context_bytes: 48,
                })],
            )),
        }),
        SpecDocument::RateBackend(RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Neural,
                vec![MixtureExpertSpec::new(RateBackend::FacCtw {
                    base_depth: 8,
                    num_percept_bits: 8,
                    encoding_bits: 1,
                })],
            )),
        }),
        SpecDocument::RateBackend(RateBackend::Particle {
            spec: Arc::new(ParticleSpec::default()),
        }),
        SpecDocument::RateBackend(RateBackend::Calibrated {
            spec: Arc::new(CalibratedSpec {
                context: CalibrationContextKind::Text,
                bins: 17,
                learning_rate: 0.05,
                bias_clip: 3.0,
                base: RateBackend::Ctw { depth: 8 },
            }),
        }),
    ];

    for doc in rate_docs {
        let expected = doc.to_canonical_json().expect("json");
        let reparsed = SpecDocument::from_binary(&doc.to_binary(), Path::new(".")).expect("binary");
        assert_eq!(reparsed.to_canonical_json().expect("parsed json"), expected);
    }
}

#[cfg(feature = "all-backends")]
#[test]
fn standalone_compression_backend_documents_cover_binary_coder_variants() {
    let docs = vec![
        SpecDocument::CompressionBackend(CompressionBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("5"),
        }),
        SpecDocument::CompressionBackend(CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Raw,
        }),
        SpecDocument::CompressionBackend(CompressionBackend::Rate {
            rate_backend: RateBackend::Mixture {
                spec: Arc::new(MixtureSpec::new(
                    MixtureKind::Bayes,
                    vec![MixtureExpertSpec::new(RateBackend::FacCtw {
                        base_depth: 8,
                        num_percept_bits: 8,
                        encoding_bits: 1,
                    })],
                )),
            },
            coder: crate::coders::CoderType::RANS,
            framing: crate::compression::FramingMode::Framed,
        }),
    ];

    for doc in docs {
        let expected = doc.to_canonical_json().expect("json");
        let reparsed = SpecDocument::from_binary(&doc.to_binary(), Path::new(".")).expect("binary");
        assert_eq!(reparsed.to_canonical_json().expect("parsed json"), expected);
    }
}

#[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
fn sample_llm_policy() -> LlmPolicy {
    LlmPolicy {
        load_from: Some("checkpoint;v1.safetensors".into()),
        schedule: vec![
            ScheduleRule::Interval(PolicyRule {
                start: PositionExpr::Bytes(0),
                end: PositionExpr::Percent(0.5),
                action: PolicyAction::Infer,
            }),
            ScheduleRule::Repeat(RepeatRule {
                start: PositionExpr::Bytes(10),
                end: PositionExpr::Bytes(200),
                period: PositionExpr::Bytes(6),
                pattern: vec![
                    RepeatSegment {
                        span: PositionExpr::Bytes(2),
                        action: PolicyAction::Train(TrainAction {
                            scope: TrainScopeSet {
                                all: false,
                                names: vec!["head".to_string(), "bias".to_string()],
                            },
                            optimizer: OptimizerKind::Adam,
                            hyper: OptimizerHyperParams {
                                lr: 0.01,
                                stride: 2,
                                bptt: 4,
                                clip: 1.5,
                                momentum: 0.9,
                            },
                        }),
                    },
                    RepeatSegment {
                        span: PositionExpr::Percent(0.5),
                        action: PolicyAction::Train(TrainAction {
                            scope: TrainScopeSet::all(),
                            optimizer: OptimizerKind::Sgd,
                            hyper: OptimizerHyperParams {
                                lr: 0.005,
                                stride: 1,
                                bptt: 1,
                                clip: 0.0,
                                momentum: 0.2,
                            },
                        }),
                    },
                ],
            }),
        ],
    }
}

#[cfg(feature = "backend-rwkv")]
#[test]
fn standalone_rwkv_method_documents_roundtrip_file_and_online_policies() {
    let file_doc = SpecDocument::RateBackend(RateBackend::Rwkv7Method {
        method: crate::rwkvzip::MethodSpec::File {
            path: "models/rwkv;demo.safetensors".into(),
            policy: Some(sample_llm_policy()),
        },
    });
    let online_doc = SpecDocument::CompressionBackend(CompressionBackend::Rwkv7 {
        method: crate::rwkvzip::MethodSpec::Online {
            cfg: crate::rwkvzip::OnlineConfig {
                hidden: 64,
                layers: 1,
                intermediate: 64,
                decay_rank: 8,
                a_rank: 8,
                v_rank: 8,
                g_rank: 8,
                seed: 17,
                train_mode: crate::rwkvzip::OnlineTrainMode::Adam,
                lr: 0.01,
                stride: 3,
            },
            policy: Some(sample_llm_policy()),
        },
        coder: crate::coders::CoderType::RANS,
    });

    for doc in [file_doc, online_doc] {
        let expected = doc.to_canonical_json().expect("json");
        let reparsed = SpecDocument::from_binary(&doc.to_binary(), Path::new(".")).expect("binary");
        assert_eq!(reparsed.to_canonical_json().expect("parsed json"), expected);
    }
}

#[cfg(feature = "backend-mamba")]
#[test]
fn standalone_mamba_method_documents_roundtrip_file_and_online_policies() {
    let file_doc = SpecDocument::RateBackend(RateBackend::MambaMethod {
        method: crate::mambazip::MethodSpec::File {
            path: "models/mamba;demo.safetensors".into(),
            policy: Some(sample_llm_policy()),
        },
    });
    let online_doc = SpecDocument::RateBackend(RateBackend::MambaMethod {
        method: crate::mambazip::MethodSpec::Online {
            cfg: crate::mambazip::OnlineConfig {
                hidden: 64,
                layers: 2,
                intermediate: 96,
                state: 8,
                conv: 4,
                dt_rank: 8,
                seed: 23,
                train_mode: crate::mambazip::OnlineTrainMode::Sgd,
                lr: 0.02,
                stride: 2,
            },
            policy: Some(sample_llm_policy()),
        },
    });

    for doc in [file_doc, online_doc] {
        let expected = doc.to_canonical_json().expect("json");
        let reparsed = SpecDocument::from_binary(&doc.to_binary(), Path::new(".")).expect("binary");
        assert_eq!(reparsed.to_canonical_json().expect("parsed json"), expected);
    }
}

#[cfg(feature = "all-backends")]
#[test]
fn planner_and_tune_documents_roundtrip_all_controller_variants() {
    let interface = PlannerInterfaceSpec {
        observation_bits: 2,
        observation_stream_len: 2,
        observation_key_mode: ObservationKeyMode::StreamHash,
        reward_bits: 2,
        agent_actions: action_alphabet(3),
        min_reward: 0,
        max_reward: 3,
        reward_offset: 0,
    };

    let planner_docs = vec![
        SpecDocument::PlannerRun(PlannerRunSpec {
            assets: vec![],
            environment: EnvironmentSpec::Builtin {
                builtin: BuiltinEnvironmentSpec::CoinFlip,
            },
            interface: interface.clone(),
            controller: ControllerSpec::McAixi(McAixiControllerSpec {
                predictor: RateBackend::FacCtw {
                    base_depth: 8,
                    num_percept_bits: 8,
                    encoding_bits: 1,
                },
                agent_horizon: 4,
                num_simulations: 12,
                mcts_strategy: MctsStrategy::ParallelUct {
                    workers: nz(3),
                    bu_uct_m_max: Some(0.5),
                },
                exploration_exploitation_ratio: 1.1,
                discount_gamma: 0.95,
            }),
            runtime: PlannerRuntimeSpec {
                random_seed: None,
                learn_cycles: Some(5),
                eval_cycles: Some(3),
                terminate_lifetime: 8,
                log_every: 2,
                perf: true,
                vm_perf_only: false,
                explore_epsilon: 0.2,
                explore_gamma: 0.9,
            },
        }),
        SpecDocument::PlannerRun(PlannerRunSpec {
            assets: vec![],
            environment: EnvironmentSpec::Builtin {
                builtin: BuiltinEnvironmentSpec::Blackjack,
            },
            interface: interface.clone(),
            controller: ControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhControllerSpec {
                predictor: RateBackend::Mixture {
                    spec: Arc::new(
                        MixtureSpec::new(
                            MixtureKind::Switching,
                            vec![
                                MixtureExpertSpec::new(RateBackend::Ctw { depth: 6 }),
                                MixtureExpertSpec::new(RateBackend::FacCtw {
                                    base_depth: 6,
                                    num_percept_bits: 8,
                                    encoding_bits: 1,
                                }),
                            ],
                        )
                        .with_schedule(MixtureScheduleMode::Theorem),
                    ),
                },
                return_horizon: 4,
                return_bins: 16,
                label_phase_period: 6,
                teacher_dataset_asset: "teacher".to_string(),
                planner_simulations_per_step: 9,
            }),
            runtime: PlannerRuntimeSpec {
                random_seed: Some(19),
                learn_cycles: None,
                eval_cycles: Some(4),
                terminate_lifetime: 7,
                log_every: 1,
                perf: false,
                vm_perf_only: false,
                explore_epsilon: 0.0,
                explore_gamma: 1.0,
            },
        }),
    ];

    for doc in planner_docs {
        let expected = doc.to_canonical_json().expect("json");
        let reparsed = SpecDocument::from_binary(&doc.to_binary(), Path::new(".")).expect("binary");
        assert_eq!(reparsed.to_canonical_json().expect("parsed json"), expected);
    }

    let bounds = TuneBoundsSpec {
        allowed_backends: vec!["ctw".to_string(), "fac-ctw".to_string()],
        forbidden_backends: vec!["zpaq".to_string()],
        parameter_ranges: vec![TuneParameterRangeSpec {
            parameter: "mixture.alpha".to_string(),
            min: 0.01,
            max: 0.5,
        }],
        max_experts: 4,
        max_mixture_nesting_depth: 2,
        min_experts: Some(1),
        allow_duplicate_experts: Some(false),
        required_experts: vec!["ctw".to_string()],
        forbidden_expert_pairs: vec![("ppmd".to_string(), "sequitur".to_string())],
    };

    let tune_docs = vec![
        SpecDocument::Tune(TuneSpec {
            assets: vec![],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 8 },
                coder: crate::coders::CoderType::AC,
                framing: crate::compression::FramingMode::Framed,
            },
            controller: TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
                interface: interface.clone(),
                planner_simulations_per_step: 10,
            }),
            bounds: bounds.clone(),
            eval_time_limit_seconds: 1.5,
            time_budget_seconds: 20.0,
            min_throughput_bytes_per_second: 2048.0,
            max_memory_bytes: 1 << 20,
            output_config_path: "mcaixi.json".to_string(),
            seed: 11,
            report_path: None,
        }),
        SpecDocument::Tune(TuneSpec {
            assets: vec![],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::FacCtw {
                    base_depth: 8,
                    num_percept_bits: 8,
                    encoding_bits: 1,
                },
                coder: crate::coders::CoderType::RANS,
                framing: crate::compression::FramingMode::Raw,
            },
            controller: TuneControllerSpec::AiqiDiscounted(AiqiDiscountedTuneControllerSpec {
                interface: interface.clone(),
                planner_simulations_per_step: 12,
                return_horizon: 5,
                return_bins: 16,
                discount_factor: 0.97,
            }),
            bounds: bounds.clone(),
            eval_time_limit_seconds: 2.0,
            time_budget_seconds: 30.0,
            min_throughput_bytes_per_second: 4096.0,
            max_memory_bytes: 1 << 21,
            output_config_path: "aiqi.json".to_string(),
            seed: 13,
            report_path: Some("aiqi-report.json".to_string()),
        }),
        SpecDocument::Tune(TuneSpec {
            assets: vec![AssetBinding {
                id: "teacher".to_string(),
                path: "teacher.bin".to_string(),
            }],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Match {
                    hash_bits: 16,
                    min_len: 2,
                    max_len: 16,
                    base_mix: 0.05,
                    confidence_scale: 1.0,
                },
                coder: crate::coders::CoderType::AC,
                framing: crate::compression::FramingMode::Framed,
            },
            controller: TuneControllerSpec::AiqiWarmstartExactJh(
                WarmStartExactJhTuneControllerSpec {
                    interface,
                    planner_simulations_per_step: 7,
                    return_horizon: 4,
                    warmstart_teacher_dataset_asset: "teacher".to_string(),
                    label_phase_period: 5,
                },
            ),
            bounds,
            eval_time_limit_seconds: 3.0,
            time_budget_seconds: 40.0,
            min_throughput_bytes_per_second: 1024.0,
            max_memory_bytes: 1 << 22,
            output_config_path: "warmstart.json".to_string(),
            seed: 17,
            report_path: Some("warmstart-report.json".to_string()),
        }),
    ];

    for doc in tune_docs {
        let expected = doc.to_canonical_json().expect("json");
        let reparsed = SpecDocument::from_binary(&doc.to_binary(), Path::new(".")).expect("binary");
        assert_eq!(reparsed.to_canonical_json().expect("parsed json"), expected);
    }
}

#[cfg(feature = "all-backends")]
#[test]
fn binary_spec_document_corruption_reports_precise_envelope_errors() {
    let rate_doc = SpecDocument::RateBackend(RateBackend::Ctw { depth: 8 });

    let mut bad_magic = rate_doc.to_binary();
    bad_magic[0] ^= 0x01;
    let err = match SpecDocument::from_binary(&bad_magic, Path::new(".")) {
        Ok(_) => panic!("corrupted magic must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("invalid spec document magic"));

    let mut bad_version = rate_doc.to_binary();
    bad_version[4] = 99;
    let err = match SpecDocument::from_binary(&bad_version, Path::new(".")) {
        Ok(_) => panic!("unknown version must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("unsupported spec document binary version")
    );

    let mut bad_doc_tag = rate_doc.to_binary();
    bad_doc_tag[5] = 99;
    let err = match SpecDocument::from_binary(&bad_doc_tag, Path::new(".")) {
        Ok(_) => panic!("unknown top-level tag must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("unknown spec document tag"));

    let mut bad_rate_tag = rate_doc.to_binary();
    bad_rate_tag[6] = 99;
    let err = match SpecDocument::from_binary(&bad_rate_tag, Path::new(".")) {
        Ok(_) => panic!("unknown rate backend tag must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("unknown rate backend tag"));

    let mut bad_compression_coder = SpecDocument::CompressionBackend(CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: crate::compression::FramingMode::Raw,
    })
    .to_binary();
    bad_compression_coder[7] = 99;
    let err = match SpecDocument::from_binary(&bad_compression_coder, Path::new(".")) {
        Ok(_) => panic!("unknown coder tag must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("unknown coder tag"));

    let mut bad_compression_framing = SpecDocument::CompressionBackend(CompressionBackend::Rate {
        rate_backend: RateBackend::Ctw { depth: 8 },
        coder: crate::coders::CoderType::AC,
        framing: crate::compression::FramingMode::Raw,
    })
    .to_binary();
    bad_compression_framing[8] = 99;
    let err = match SpecDocument::from_binary(&bad_compression_framing, Path::new(".")) {
        Ok(_) => panic!("unknown framing tag must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("unknown framing tag"));

    let mut bad_mixture_kind = SpecDocument::RateBackend(RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(
            MixtureKind::Bayes,
            vec![MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })],
        )),
    })
    .to_binary();
    bad_mixture_kind[7] = 99;
    let err = match SpecDocument::from_binary(&bad_mixture_kind, Path::new(".")) {
        Ok(_) => panic!("unknown mixture kind must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("unknown mixture kind tag"));

    let mut bad_mixture_schedule = SpecDocument::RateBackend(RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(
            MixtureKind::Switching,
            vec![MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })],
        )),
    })
    .to_binary();
    bad_mixture_schedule[8] = 99;
    let err = match SpecDocument::from_binary(&bad_mixture_schedule, Path::new(".")) {
        Ok(_) => panic!("unknown mixture schedule must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("unknown mixture schedule tag"));
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
            agent_actions: action_alphabet(1),
            min_reward: 0,
            max_reward: 255,
            reward_offset: 0,
        },
        controller: ControllerSpec::McAixi(McAixiControllerSpec {
            predictor: RateBackend::Ctw { depth: 8 },
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
    let spec = sample_tune_spec();
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
    let mut base = sample_tune_spec();
    base.bounds.forbidden_backends = vec![];
    base.bounds.parameter_ranges = vec![];
    base.bounds.required_experts = vec![];
    base.report_path = Some("report-a.json".to_string());
    base.output_config_path = "best-a.json".to_string();
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
            agent_actions: action_alphabet(2),
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
        },
        controller: ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
            predictor: RateBackend::Ctw { depth: 8 },
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
