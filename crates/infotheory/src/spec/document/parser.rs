//! JSON parsing for canonical top-level specification documents.

use super::{
    AiqiDiscountedControllerSpec, AiqiDiscountedTuneControllerSpec,
    AnnealedHillClimbingTuneControllerSpec, AssetBinding, ControllerSpec, EnvironmentSpec,
    McAixiControllerSpec, McAixiFacCtwTuneControllerSpec, PlannerInterfaceSpec, PlannerRunSpec,
    PlannerRuntimeSpec, SPEC_DOCUMENT_SCHEMA_VERSION, SpecDocument, SpecError, SpecResult,
    TuneBoundsSpec, TuneControllerSpec, TuneParameterRangeSpec, TuneSpec,
    WarmStartExactJhControllerSpec, WarmStartExactJhTuneControllerSpec,
    parse_compression_backend_json, parse_rate_backend_json,
};
use crate::aixi::common::MctsStrategy;
use std::num::NonZeroUsize;

#[cfg(feature = "vm")]
use super::{
    VmActionFilterSpec, VmEnvironmentSpec, VmRewardPolicySpec, VmRewardShapingSpec,
    VmRuntimeActionSourceSpec, VmTraceSpec,
};
use std::path::Path;

pub(super) fn parse_spec_document_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<SpecDocument> {
    let version = value["schema_version"].as_u64().unwrap_or(0);
    if version != SPEC_DOCUMENT_SCHEMA_VERSION as u64 {
        return Err(SpecError::new(format!(
            "unsupported spec document schema_version '{version}'"
        )));
    }
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("spec document kind is required"))?;
    match kind {
        "planner_run" => Ok(SpecDocument::PlannerRun(parse_planner_run_json_value(
            value, base_dir,
        )?)),
        "tune" => Ok(SpecDocument::Tune(parse_tune_spec_json_value(
            value, base_dir,
        )?)),
        "rate_backend" => Ok(SpecDocument::RateBackend(parse_rate_backend_json(
            &value["backend"],
            base_dir,
            crate::api::MAX_MIXTURE_NESTING,
        )?)),
        "compression_backend" => Ok(SpecDocument::CompressionBackend(
            parse_compression_backend_json(
                &value["backend"],
                base_dir,
                None,
                crate::compression::FramingMode::Framed,
            )?,
        )),
        other => Err(SpecError::new(format!(
            "unknown spec document kind '{other}'"
        ))),
    }
}

fn parse_planner_run_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<PlannerRunSpec> {
    Ok(PlannerRunSpec {
        assets: parse_asset_bindings(&value["assets"])?,
        environment: parse_environment_spec(&value["environment"], base_dir)?,
        interface: parse_interface_spec(&value["interface"])?,
        controller: parse_controller_spec(&value["controller"], base_dir)?,
        runtime: parse_runtime_spec(&value["runtime"])?,
    })
}

fn parse_tune_spec_json_value(value: &serde_json::Value, base_dir: &Path) -> SpecResult<TuneSpec> {
    Ok(TuneSpec {
        assets: parse_asset_bindings(&value["assets"])?,
        input_asset: required_string(&value["input_asset"], "input_asset")?,
        baseline_candidate: parse_compression_backend_json(
            &value["baseline_candidate"],
            base_dir,
            None,
            crate::compression::FramingMode::Framed,
        )?,
        controller: parse_tune_controller_spec(&value["controller"])?,
        bounds: parse_tune_bounds_spec(&value["bounds"])?,
        eval_time_limit_seconds: required_f64(
            &value["eval_time_limit_seconds"],
            "eval_time_limit_seconds",
        )?,
        time_budget_seconds: required_f64(&value["time_budget_seconds"], "time_budget_seconds")?,
        min_throughput_bytes_per_second: required_f64(
            &value["min_throughput_bytes_per_second"],
            "min_throughput_bytes_per_second",
        )?,
        max_memory_bytes: required_u64(&value["max_memory_bytes"], "max_memory_bytes")?,
        output_config_path: required_string(&value["output_config_path"], "output_config_path")?,
        seed: required_u64(&value["seed"], "seed")?,
        report_path: optional_string(&value["report_path"]),
    })
}

fn parse_asset_bindings(value: &serde_json::Value) -> SpecResult<Vec<AssetBinding>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            Ok(AssetBinding {
                id: required_string(&item["id"], "assets[].id")?,
                path: required_string(&item["path"], "assets[].path")?,
            })
        })
        .collect()
}

fn parse_environment_spec(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<EnvironmentSpec> {
    #[cfg(not(feature = "vm"))]
    let _ = base_dir;
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("environment.kind is required"))?;
    match kind {
        "builtin" => Ok(EnvironmentSpec::Builtin {
            builtin: parse_builtin_environment(
                value["name"]
                    .as_str()
                    .ok_or_else(|| SpecError::new("environment.name is required"))?,
            )?,
        }),
        #[cfg(feature = "vm")]
        "nyx_vm" => Ok(EnvironmentSpec::NyxVm(VmEnvironmentSpec {
            firecracker_config_asset: required_string(
                &value["firecracker_config_asset"],
                "environment.firecracker_config_asset",
            )?,
            instance_id: optional_string(&value["instance_id"])
                .unwrap_or_else(|| "aixi-nyx".to_string()),
            shared_region_name: optional_string(&value["shared_region_name"])
                .unwrap_or_else(|| "shared".to_string()),
            shared_region_size: value["shared_region_size"].as_u64().unwrap_or(4096) as usize,
            shared_memory_policy: parse_shared_memory_policy(
                value["shared_memory_policy"].as_str().unwrap_or("snapshot"),
            )?,
            step_timeout_ms: value["step_timeout_ms"].as_u64().unwrap_or(100),
            boot_timeout_ms: value["boot_timeout_ms"].as_u64().unwrap_or(30_000),
            episode_steps: value["episode_steps"].as_u64().unwrap_or(100) as usize,
            step_cost: value["step_cost"].as_i64().unwrap_or(0),
            observation_policy: super::canonicalize_vm_observation_policy_name(
                value["observation_policy"]
                    .as_str()
                    .unwrap_or("shared_memory"),
            )?,
            observation_bits: value["observation_bits"].as_u64().unwrap_or(8) as usize,
            observation_stream_len: value["observation_stream_len"].as_u64().unwrap_or(64) as usize,
            observation_stream_mode: super::canonicalize_vm_observation_stream_mode_name(
                value["observation_stream_mode"]
                    .as_str()
                    .unwrap_or("pad_truncate"),
            )?,
            observation_pad_byte: value["observation_pad_byte"].as_u64().unwrap_or(0) as u8,
            reward_bits: value["reward_bits"].as_u64().unwrap_or(8) as usize,
            reward_policy: parse_vm_reward_policy(&value["reward_policy"])?,
            reward_shaping: parse_optional_vm_reward_shaping(&value["reward_shaping"])?,
            action_source: parse_vm_action_source(&value["action_source"])?,
            action_filter: parse_optional_vm_action_filter(&value["action_filter"])?,
            action_prefix: value["protocol"]["action_prefix"]
                .as_str()
                .unwrap_or("ACT ")
                .to_string(),
            action_suffix: value["protocol"]["action_suffix"]
                .as_str()
                .unwrap_or("\n")
                .to_string(),
            obs_prefix: value["protocol"]["obs_prefix"]
                .as_str()
                .unwrap_or("OBS ")
                .to_string(),
            rew_prefix: value["protocol"]["rew_prefix"]
                .as_str()
                .unwrap_or("REW ")
                .to_string(),
            done_prefix: value["protocol"]["done_prefix"]
                .as_str()
                .unwrap_or("DONE ")
                .to_string(),
            data_prefix: value["protocol"]["data_prefix"]
                .as_str()
                .unwrap_or("DATA ")
                .to_string(),
            wire_encoding: super::canonicalize_vm_payload_encoding(
                value["protocol"]["wire_encoding"].as_str().unwrap_or("hex"),
                "environment.protocol.wire_encoding",
            )?,
            stats_backend: parse_rate_backend_json(
                &value["stats_backend"],
                base_dir,
                crate::api::MAX_MIXTURE_NESTING,
            )?,
            trace: parse_optional_vm_trace(&value["trace"])?,
            debug_mode: value["debug_mode"].as_bool().unwrap_or(false),
            crash_log: optional_string(&value["crash_log"]),
        })),
        #[cfg(not(feature = "vm"))]
        "nyx_vm" => Err(SpecError::new(
            "nyx_vm environment requires the 'vm' feature",
        )),
        other => Err(SpecError::new(format!(
            "unknown environment kind '{other}'"
        ))),
    }
}

fn parse_interface_spec(value: &serde_json::Value) -> SpecResult<PlannerInterfaceSpec> {
    Ok(PlannerInterfaceSpec {
        observation_bits: required_u64(&value["observation_bits"], "interface.observation_bits")?
            as usize,
        observation_stream_len: required_u64(
            &value["observation_stream_len"],
            "interface.observation_stream_len",
        )? as usize,
        observation_key_mode: parse_observation_key_mode(
            value["observation_key_mode"]
                .as_str()
                .unwrap_or("full_stream"),
        )?,
        reward_bits: required_u64(&value["reward_bits"], "interface.reward_bits")? as usize,
        agent_actions: required_u64(&value["agent_actions"], "interface.agent_actions")? as usize,
        min_reward: required_i64(&value["min_reward"], "interface.min_reward")?,
        max_reward: required_i64(&value["max_reward"], "interface.max_reward")?,
        reward_offset: required_i64(&value["reward_offset"], "interface.reward_offset")?,
    })
}

fn parse_controller_spec(value: &serde_json::Value, base_dir: &Path) -> SpecResult<ControllerSpec> {
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("controller.kind is required"))?;
    match kind {
        "mc_aixi" => Ok(ControllerSpec::McAixi(McAixiControllerSpec {
            predictor: parse_rate_backend_json(
                &value["predictor"],
                base_dir,
                crate::api::MAX_MIXTURE_NESTING,
            )?,
            agent_horizon: required_u64(&value["agent_horizon"], "controller.agent_horizon")?
                as usize,
            num_simulations: required_u64(&value["num_simulations"], "controller.num_simulations")?
                as usize,
            mcts_strategy: parse_mcts_strategy(
                value.get("mcts_strategy"),
                "controller.mcts_strategy",
            )?,
            exploration_exploitation_ratio: required_f64(
                &value["exploration_exploitation_ratio"],
                "controller.exploration_exploitation_ratio",
            )?,
            discount_gamma: required_f64(&value["discount_gamma"], "controller.discount_gamma")?,
        })),
        "aiqi_discounted" => Ok(ControllerSpec::AiqiDiscounted(
            AiqiDiscountedControllerSpec {
                predictor: parse_rate_backend_json(
                    &value["predictor"],
                    base_dir,
                    crate::api::MAX_MIXTURE_NESTING,
                )?,
                discount_gamma: required_f64(
                    &value["discount_gamma"],
                    "controller.discount_gamma",
                )?,
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                return_bins: required_u64(&value["return_bins"], "controller.return_bins")?
                    as usize,
                augmentation_period: required_u64(
                    &value["augmentation_period"],
                    "controller.augmentation_period",
                )? as usize,
                history_prune_keep_steps: value["history_prune_keep_steps"]
                    .as_u64()
                    .map(|n| n as usize),
                baseline_exploration: required_f64(
                    &value["baseline_exploration"],
                    "controller.baseline_exploration",
                )?,
            },
        )),
        "aiqi_warmstart_exact_jh" => Ok(ControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhControllerSpec {
                predictor: parse_rate_backend_json(
                    &value["predictor"],
                    base_dir,
                    crate::api::MAX_MIXTURE_NESTING,
                )?,
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                return_bins: required_u64(&value["return_bins"], "controller.return_bins")?
                    as usize,
                label_phase_period: required_u64(
                    &value["label_phase_period"],
                    "controller.label_phase_period",
                )? as usize,
                teacher_dataset_asset: required_string(
                    &value["teacher_dataset_asset"],
                    "controller.teacher_dataset_asset",
                )?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
            },
        )),
        other => Err(SpecError::new(format!("unknown controller kind '{other}'"))),
    }
}

fn parse_mcts_strategy(value: Option<&serde_json::Value>, label: &str) -> SpecResult<MctsStrategy> {
    // Absent field defaults to the canonical sequential planner so older
    // documents that predate `mcts_strategy` continue to parse cleanly.
    let Some(value) = value else {
        return Ok(MctsStrategy::RhoUct);
    };
    // The canonical, serializer-emitted form is always an object with a
    // `kind` field. Strings are not accepted: there is exactly one way to
    // spell each strategy in the schema.
    let Some(object) = value.as_object() else {
        return Err(SpecError::new(format!(
            "{label} must be an object with a 'kind' field"
        )));
    };
    let kind = object
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| SpecError::new(format!("{label}.kind is required")))?;
    match kind {
        "rho_uct" => Ok(MctsStrategy::RhoUct),
        "parallel_uct" => {
            let workers_raw = required_u64(&value["workers"], &format!("{label}.workers"))?;
            let workers = NonZeroUsize::new(workers_raw as usize)
                .ok_or_else(|| SpecError::new(format!("{label}.workers must be >= 1")))?;
            let bu_uct_m_max = match object.get("bu_uct_m_max") {
                Some(raw) if raw.is_null() => None,
                Some(raw) => Some(required_f64(raw, &format!("{label}.bu_uct_m_max"))?),
                None => None,
            };
            Ok(MctsStrategy::ParallelUct {
                workers,
                bu_uct_m_max,
            })
        }
        other => Err(SpecError::new(format!(
            "unknown MCTS strategy kind '{other}'"
        ))),
    }
}

fn parse_runtime_spec(value: &serde_json::Value) -> SpecResult<PlannerRuntimeSpec> {
    Ok(PlannerRuntimeSpec {
        random_seed: value["random_seed"].as_u64(),
        learn_cycles: value["learn_cycles"].as_u64().map(|n| n as usize),
        eval_cycles: value["eval_cycles"].as_u64().map(|n| n as usize),
        terminate_lifetime: value["terminate_lifetime"].as_u64().unwrap_or(20) as usize,
        log_every: value["log_every"].as_u64().unwrap_or(1) as usize,
        perf: value["perf"].as_bool().unwrap_or(false),
        vm_perf_only: value["vm_perf_only"].as_bool().unwrap_or(false),
        explore_epsilon: value["explore_epsilon"].as_f64().unwrap_or(0.0),
        explore_gamma: value["explore_gamma"].as_f64().unwrap_or(1.0),
    })
}

fn parse_tune_bounds_spec(value: &serde_json::Value) -> SpecResult<TuneBoundsSpec> {
    Ok(TuneBoundsSpec {
        allowed_backends: string_list(&value["allowed_backends"])?,
        forbidden_backends: string_list(&value["forbidden_backends"])?,
        parameter_ranges: parse_tune_parameter_ranges(&value["parameter_ranges"])?,
        max_experts: required_u64(&value["max_experts"], "bounds.max_experts")? as usize,
        max_mixture_nesting_depth: required_u64(
            &value["max_mixture_nesting_depth"],
            "bounds.max_mixture_nesting_depth",
        )? as usize,
        min_experts: value["min_experts"].as_u64().map(|n| n as usize),
        allow_duplicate_experts: value["allow_duplicate_experts"].as_bool(),
        required_experts: string_list(&value["required_experts"])?,
        forbidden_expert_pairs: pair_list(&value["forbidden_expert_pairs"])?,
    })
}

fn parse_tune_controller_spec(value: &serde_json::Value) -> SpecResult<TuneControllerSpec> {
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("controller.kind is required"))?;
    match kind {
        "annealed_hill_climbing" => Ok(TuneControllerSpec::AnnealedHillClimbing(
            AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: required_u64(
                    &value["max_mutation_radius"],
                    "controller.max_mutation_radius",
                )? as usize,
            },
        )),
        "mc_aixi_fac_ctw" => Ok(TuneControllerSpec::McAixiFacCtw(
            McAixiFacCtwTuneControllerSpec {
                interface: parse_interface_spec(&value["interface"])?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
            },
        )),
        "aiqi_discounted" => Ok(TuneControllerSpec::AiqiDiscounted(
            AiqiDiscountedTuneControllerSpec {
                interface: parse_interface_spec(&value["interface"])?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                return_bins: required_u64(&value["return_bins"], "controller.return_bins")?
                    as usize,
                discount_factor: required_f64(
                    &value["discount_factor"],
                    "controller.discount_factor",
                )?,
            },
        )),
        "aiqi_warmstart_exact_jh" => Ok(TuneControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhTuneControllerSpec {
                interface: parse_interface_spec(&value["interface"])?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                warmstart_teacher_dataset_asset: required_string(
                    &value["warmstart_teacher_dataset_asset"],
                    "controller.warmstart_teacher_dataset_asset",
                )?,
                label_phase_period: required_u64(
                    &value["label_phase_period"],
                    "controller.label_phase_period",
                )? as usize,
            },
        )),
        other => Err(SpecError::new(format!(
            "unknown tune controller kind '{other}'"
        ))),
    }
}

fn parse_tune_parameter_ranges(
    value: &serde_json::Value,
) -> SpecResult<Vec<TuneParameterRangeSpec>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            Ok(TuneParameterRangeSpec {
                parameter: required_string(
                    &item["parameter"],
                    "bounds.parameter_ranges[].parameter",
                )?,
                min: required_f64(&item["min"], "bounds.parameter_ranges[].min")?,
                max: required_f64(&item["max"], "bounds.parameter_ranges[].max")?,
            })
        })
        .collect()
}

#[cfg(feature = "vm")]
fn parse_vm_reward_policy(value: &serde_json::Value) -> SpecResult<VmRewardPolicySpec> {
    match value["kind"].as_str().unwrap_or("from_guest") {
        "from_guest" => Ok(VmRewardPolicySpec::FromGuest),
        "pattern" => Ok(VmRewardPolicySpec::Pattern {
            pattern: required_string(&value["pattern"], "environment.reward_policy.pattern")?,
            base_reward: value["base_reward"].as_i64().unwrap_or(0),
            bonus_reward: value["bonus_reward"].as_i64().unwrap_or(10),
        }),
        other => Err(SpecError::new(format!(
            "unknown vm reward policy '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_optional_vm_reward_shaping(
    value: &serde_json::Value,
) -> SpecResult<Option<VmRewardShapingSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    let spec = match value["kind"].as_str().unwrap_or("trace_entropy") {
        "entropy_reduction" => VmRewardShapingSpec::EntropyReduction {
            baseline_asset: required_string(
                &value["baseline_asset"],
                "environment.reward_shaping.baseline_asset",
            )?,
            scale: value["scale"].as_f64().unwrap_or(1.0),
            crash_bonus: value["crash_bonus"].as_i64(),
            timeout_bonus: value["timeout_bonus"].as_i64(),
        },
        "trace_entropy" => VmRewardShapingSpec::TraceEntropy {
            scale: value["scale"].as_f64().unwrap_or(1.0),
            normalize: value["normalize"].as_bool().unwrap_or(false),
        },
        other => {
            return Err(SpecError::new(format!(
                "unknown vm reward shaping '{other}'"
            )));
        }
    };
    Ok(Some(spec))
}

#[cfg(feature = "vm")]
fn parse_vm_action_source(value: &serde_json::Value) -> SpecResult<VmRuntimeActionSourceSpec> {
    match value["kind"].as_str().unwrap_or("literal") {
        "literal" => {
            let encoding = super::canonicalize_vm_payload_encoding(
                value["encoding"].as_str().unwrap_or("utf8"),
                "environment.action_source.encoding",
            )?;
            let actions = value["actions"]
                .as_array()
                .ok_or_else(|| SpecError::new("environment.action_source.actions is required"))?;
            let mut names = Vec::with_capacity(actions.len());
            let mut payloads = Vec::with_capacity(actions.len());
            for action in actions {
                names.push(optional_string(&action["name"]));
                payloads.push(required_string(
                    &action["payload"],
                    "environment.action_source.actions[].payload",
                )?);
            }
            Ok(VmRuntimeActionSourceSpec::Literal {
                names,
                payloads,
                encoding,
            })
        }
        "fuzz" => Ok(VmRuntimeActionSourceSpec::Fuzz {
            seeds: string_list(&value["seeds"])?,
            encoding: super::canonicalize_vm_payload_encoding(
                value["encoding"].as_str().unwrap_or("utf8"),
                "environment.action_source.encoding",
            )?,
            mutators: string_list(&value["mutators"])?
                .into_iter()
                .map(|name| super::canonicalize_vm_fuzz_mutator_name(&name))
                .collect::<SpecResult<Vec<_>>>()?,
            min_len: value["min_len"].as_u64().unwrap_or(1) as usize,
            max_len: value["max_len"].as_u64().unwrap_or(4096) as usize,
            dictionary: string_list(&value["dictionary"])?,
            rng_seed: value["rng_seed"].as_u64().unwrap_or(0),
        }),
        other => Err(SpecError::new(format!(
            "unknown vm action source '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_optional_vm_action_filter(
    value: &serde_json::Value,
) -> SpecResult<Option<VmActionFilterSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(VmActionFilterSpec {
        min_entropy: value["min_entropy"].as_f64(),
        max_entropy: value["max_entropy"].as_f64(),
        min_intrinsic_dependence: value["min_intrinsic_dependence"].as_f64(),
        min_novelty: value["min_novelty"].as_f64(),
        novelty_prior_asset: optional_string(&value["novelty_prior_asset"]),
        reject_reward: value["reject_reward"].as_i64(),
    }))
}

#[cfg(feature = "vm")]
fn parse_optional_vm_trace(value: &serde_json::Value) -> SpecResult<Option<VmTraceSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(VmTraceSpec {
        shared_region_name: optional_string(&value["shared_region_name"]),
        max_bytes: value["max_bytes"].as_u64().unwrap_or(1_000_000) as usize,
        reset_on_episode: value["reset_on_episode"].as_bool().unwrap_or(false),
    }))
}

fn string_list(value: &serde_json::Value) -> SpecResult<Vec<String>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(|text| text.to_string())
                .ok_or_else(|| SpecError::new("expected string list item"))
        })
        .collect()
}

fn pair_list(value: &serde_json::Value) -> SpecResult<Vec<(String, String)>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    let mut pairs = Vec::with_capacity(items.len());
    for item in items {
        let Some(pair) = item.as_array() else {
            return Err(SpecError::new(
                "forbidden_expert_pairs entries must be arrays",
            ));
        };
        if pair.len() != 2 {
            return Err(SpecError::new(
                "forbidden_expert_pairs entries must have length 2",
            ));
        }
        pairs.push((
            required_string(&pair[0], "forbidden_expert_pairs[][0]")?,
            required_string(&pair[1], "forbidden_expert_pairs[][1]")?,
        ));
    }
    Ok(pairs)
}

fn optional_string(value: &serde_json::Value) -> Option<String> {
    value.as_str().map(|text| text.to_string())
}

fn required_string(value: &serde_json::Value, label: &str) -> SpecResult<String> {
    value
        .as_str()
        .map(|text| text.to_string())
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_f64(value: &serde_json::Value, label: &str) -> SpecResult<f64> {
    value
        .as_f64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_u64(value: &serde_json::Value, label: &str) -> SpecResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_i64(value: &serde_json::Value, label: &str) -> SpecResult<i64> {
    value
        .as_i64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn parse_builtin_environment(name: &str) -> SpecResult<super::BuiltinEnvironmentSpec> {
    match name {
        "coin_flip" => Ok(super::BuiltinEnvironmentSpec::CoinFlip),
        "biased_rock_paper_scissor" => Ok(super::BuiltinEnvironmentSpec::BiasedRockPaperScissor),
        "kuhn_poker" => Ok(super::BuiltinEnvironmentSpec::KuhnPoker),
        "extended_tiger" => Ok(super::BuiltinEnvironmentSpec::ExtendedTiger),
        "ctw_test" => Err(SpecError::new(
            "builtin environment 'ctw_test' is no longer supported",
        )),
        "tic_tac_toe" => Ok(super::BuiltinEnvironmentSpec::TicTacToe),
        "blackjack" => Ok(super::BuiltinEnvironmentSpec::Blackjack),
        "platformer" => Ok(super::BuiltinEnvironmentSpec::Platformer),
        other => Err(SpecError::new(format!(
            "unknown builtin environment '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_shared_memory_policy(name: &str) -> SpecResult<super::SharedMemoryPolicySpec> {
    match name {
        "snapshot" => Ok(super::SharedMemoryPolicySpec::Snapshot),
        "preserve" => Ok(super::SharedMemoryPolicySpec::Preserve),
        other => Err(SpecError::new(format!(
            "unknown shared memory policy '{other}'"
        ))),
    }
}

fn parse_observation_key_mode(name: &str) -> SpecResult<crate::aixi::common::ObservationKeyMode> {
    match name {
        "first" => Ok(crate::aixi::common::ObservationKeyMode::First),
        "last" => Ok(crate::aixi::common::ObservationKeyMode::Last),
        "stream_hash" => Ok(crate::aixi::common::ObservationKeyMode::StreamHash),
        "full_stream" => Ok(crate::aixi::common::ObservationKeyMode::FullStream),
        other => Err(SpecError::new(format!(
            "unknown observation key mode '{other}'"
        ))),
    }
}
