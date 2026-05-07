//! Canonical JSON serialization for spec documents.

use super::{
    AssetBinding, ControllerSpec, EnvironmentSpec, PlannerInterfaceSpec, PlannerRunSpec,
    PlannerRuntimeSpec, SPEC_DOCUMENT_SCHEMA_VERSION, SpecDocument, SpecResult,
    compression_backend_to_json_value, rate_backend_to_json_value,
};
#[cfg(feature = "tuner")]
use super::{
    TuneBoundsSpec, TuneControllerSpec, TuneParameterRangeSpec, TunePlannerInterfaceSpec, TuneSpec,
};
use crate::aixi::common::MctsStrategy;

#[cfg(feature = "vm")]
use super::{
    VmActionFilterSpec, VmRewardPolicySpec, VmRewardShapingSpec, VmRuntimeActionSourceSpec,
    VmTraceSpec,
};

pub(super) fn spec_document_to_json_value(doc: &SpecDocument) -> SpecResult<serde_json::Value> {
    match doc {
        SpecDocument::PlannerRun(spec) => planner_run_to_json_value(spec),
        #[cfg(feature = "tuner")]
        SpecDocument::Tune(spec) => tune_spec_to_json_value(spec),
        SpecDocument::RateBackend(backend) => Ok(serde_json::json!({
            "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
            "kind": "rate_backend",
            "backend": rate_backend_to_json_value(backend)?,
        })),
        SpecDocument::CompressionBackend(backend) => Ok(serde_json::json!({
            "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
            "kind": "compression_backend",
            "backend": compression_backend_to_json_value(backend)?,
        })),
    }
}

pub(super) fn planner_run_to_json_value(spec: &PlannerRunSpec) -> SpecResult<serde_json::Value> {
    Ok(serde_json::json!({
        "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
        "kind": "planner_run",
        "assets": spec.assets.iter().map(asset_binding_to_json_value).collect::<Vec<_>>(),
        "environment": environment_spec_to_json_value(&spec.environment)?,
        "interface": interface_spec_to_json_value(&spec.interface),
        "controller": controller_spec_to_json_value(&spec.controller)?,
        "runtime": runtime_spec_to_json_value(&spec.runtime),
    }))
}

#[cfg(feature = "tuner")]
pub(super) fn tune_spec_to_json_value(spec: &TuneSpec) -> SpecResult<serde_json::Value> {
    Ok(serde_json::json!({
        "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
        "kind": "tune",
        "assets": spec.assets.iter().map(asset_binding_to_json_value).collect::<Vec<_>>(),
        "input_asset": spec.input_asset,
        "baseline_candidate": compression_backend_to_json_value(&spec.baseline_candidate)?,
        "controller": tune_controller_to_json_value(&spec.controller),
        "bounds": tune_bounds_to_json_value(&spec.bounds),
        "eval_time_limit_seconds": spec.eval_time_limit_seconds,
        "time_budget_seconds": spec.time_budget_seconds,
        "min_throughput_bytes_per_second": spec.min_throughput_bytes_per_second,
        "max_memory_bytes": spec.max_memory_bytes,
        "output_config_path": spec.output_config_path,
        "seed": spec.seed,
        "report_path": spec.report_path,
    }))
}

fn asset_binding_to_json_value(binding: &AssetBinding) -> serde_json::Value {
    serde_json::json!({
        "id": binding.id,
        "path": binding.path,
    })
}

fn environment_spec_to_json_value(spec: &EnvironmentSpec) -> SpecResult<serde_json::Value> {
    match spec {
        EnvironmentSpec::Builtin { builtin } => Ok(serde_json::json!({
            "kind": "builtin",
            "name": super::builtin_environment_name(*builtin),
        })),
        #[cfg(feature = "vm")]
        EnvironmentSpec::NyxVm(vm) => Ok(serde_json::json!({
            "kind": "nyx_vm",
            "firecracker_config_asset": vm.firecracker_config_asset,
            "instance_id": vm.instance_id,
            "shared_region_name": vm.shared_region_name,
            "shared_region_size": vm.shared_region_size,
            "shared_memory_policy": super::shared_memory_policy_name(vm.shared_memory_policy),
            "step_timeout_ms": vm.step_timeout_ms,
            "boot_timeout_ms": vm.boot_timeout_ms,
            "episode_steps": vm.episode_steps,
            "step_cost": vm.step_cost,
            "observation_policy": super::vm_observation_policy_name(vm.observation_policy),
            "observation_bits": vm.observation_bits,
            "observation_stream_len": vm.observation_stream_len,
            "observation_stream_mode": super::vm_observation_stream_mode_name(vm.observation_stream_mode),
            "observation_pad_byte": vm.observation_pad_byte,
            "reward_bits": vm.reward_bits,
            "reward_policy": vm_reward_policy_to_json_value(&vm.reward_policy),
            "reward_shaping": vm.reward_shaping.as_ref().map(vm_reward_shaping_to_json_value),
            "action_source": vm_action_source_to_json_value(&vm.action_source),
            "action_filter": vm.action_filter.as_ref().map(vm_action_filter_to_json_value),
            "protocol": {
                "action_prefix": vm.action_prefix,
                "action_suffix": vm.action_suffix,
                "obs_prefix": vm.obs_prefix,
                "rew_prefix": vm.rew_prefix,
                "done_prefix": vm.done_prefix,
                "data_prefix": vm.data_prefix,
                "wire_encoding": super::vm_payload_encoding_name(vm.wire_encoding),
            },
            "stats_backend": rate_backend_to_json_value(&vm.stats_backend)?,
            "trace": vm.trace.as_ref().map(vm_trace_to_json_value),
            "debug_mode": vm.debug_mode,
            "crash_log": vm.crash_log,
        })),
    }
}

fn interface_spec_to_json_value(spec: &PlannerInterfaceSpec) -> serde_json::Value {
    serde_json::json!({
        "observation_bits": spec.observation_bits,
        "observation_stream_len": spec.observation_stream_len,
        "observation_key_mode": super::observation_key_mode_name(spec.observation_key_mode),
        "reward_bits": spec.reward_bits,
        "agent_actions": spec.agent_actions.get(),
    })
}

#[cfg(feature = "tuner")]
fn tune_interface_spec_to_json_value(spec: &TunePlannerInterfaceSpec) -> serde_json::Value {
    serde_json::json!({
        "observation_bits": spec.observation_bits,
        "observation_stream_len": spec.observation_stream_len,
        "observation_key_mode": super::observation_key_mode_name(spec.observation_key_mode),
        "reward_bits": spec.reward_bits,
        "agent_actions": spec.agent_actions.get(),
    })
}

fn controller_spec_to_json_value(spec: &ControllerSpec) -> SpecResult<serde_json::Value> {
    match spec {
        ControllerSpec::McAixi(inner) => Ok(serde_json::json!({
            "kind": "mc_aixi",
            "predictor": rate_backend_to_json_value(&inner.predictor)?,
            "agent_horizon": inner.agent_horizon,
            "num_simulations": inner.num_simulations,
            "mcts_strategy": mcts_strategy_to_json_value(inner.mcts_strategy),
            "exploration_exploitation_ratio": inner.exploration_exploitation_ratio,
            "discount_gamma": inner.discount_gamma,
        })),
        ControllerSpec::AiqiDiscounted(inner) => Ok(serde_json::json!({
            "kind": "aiqi_discounted",
            "predictor": rate_backend_to_json_value(&inner.predictor)?,
            "discount_gamma": inner.discount_gamma,
            "return_horizon": inner.return_horizon,
            "return_bins": inner.return_bins,
            "augmentation_period": inner.augmentation_period,
            "history_prune_keep_steps": inner.history_prune_keep_steps,
            "baseline_exploration": inner.baseline_exploration,
        })),
        ControllerSpec::AiqiWarmstartExactJh(inner) => Ok(serde_json::json!({
            "kind": "aiqi_warmstart_exact_jh",
            "predictor": rate_backend_to_json_value(&inner.predictor)?,
            "return_horizon": inner.return_horizon,
            "return_bins": inner.return_bins,
            "label_phase_period": inner.label_phase_period,
            "teacher_dataset_asset": inner.teacher_dataset_asset,
            "planner_simulations_per_step": inner.planner_simulations_per_step,
        })),
    }
}

fn mcts_strategy_to_json_value(strategy: MctsStrategy) -> serde_json::Value {
    match strategy {
        MctsStrategy::RhoUct => serde_json::json!({
            "kind": "rho_uct",
        }),
        MctsStrategy::ParallelUct {
            workers,
            bu_uct_m_max,
        } => serde_json::json!({
            "kind": "parallel_uct",
            "workers": workers.get(),
            "bu_uct_m_max": bu_uct_m_max,
        }),
    }
}

fn runtime_spec_to_json_value(spec: &PlannerRuntimeSpec) -> serde_json::Value {
    serde_json::json!({
        "random_seed": spec.random_seed,
        "learn_cycles": spec.learn_cycles,
        "eval_cycles": spec.eval_cycles,
        "terminate_lifetime": spec.terminate_lifetime,
        "log_every": spec.log_every,
        "perf": spec.perf,
        "vm_perf_only": spec.vm_perf_only,
        "explore_epsilon": spec.explore_epsilon,
        "explore_gamma": spec.explore_gamma,
    })
}

#[cfg(feature = "tuner")]
fn tune_bounds_to_json_value(bounds: &TuneBoundsSpec) -> serde_json::Value {
    serde_json::json!({
        "allowed_backends": bounds.allowed_backends,
        "forbidden_backends": bounds.forbidden_backends,
        "parameter_ranges": bounds.parameter_ranges.iter().map(tune_parameter_range_to_json_value).collect::<Vec<_>>(),
        "max_experts": bounds.max_experts,
        "max_mixture_nesting_depth": bounds.max_mixture_nesting_depth,
        "min_experts": bounds.min_experts,
        "allow_duplicate_experts": bounds.allow_duplicate_experts,
        "required_experts": bounds.required_experts,
        "forbidden_expert_pairs": bounds.forbidden_expert_pairs.iter().map(|(a, b)| vec![a, b]).collect::<Vec<_>>(),
    })
}

#[cfg(feature = "tuner")]
fn tune_controller_to_json_value(spec: &TuneControllerSpec) -> serde_json::Value {
    match spec {
        TuneControllerSpec::AnnealedHillClimbing(inner) => serde_json::json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": inner.max_mutation_radius,
        }),
        TuneControllerSpec::McAixiFacCtw(inner) => serde_json::json!({
            "kind": "mc_aixi_fac_ctw",
            "interface": tune_interface_spec_to_json_value(&inner.interface),
            "planner_simulations_per_step": inner.planner_simulations_per_step,
        }),
        TuneControllerSpec::AiqiDiscounted(inner) => serde_json::json!({
            "kind": "aiqi_discounted",
            "interface": tune_interface_spec_to_json_value(&inner.interface),
            "planner_simulations_per_step": inner.planner_simulations_per_step,
            "return_horizon": inner.return_horizon,
            "return_bins": inner.return_bins,
            "discount_factor": inner.discount_factor,
            "min_improvement": inner.min_improvement,
            "max_improvement": inner.max_improvement,
        }),
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => serde_json::json!({
            "kind": "aiqi_warmstart_exact_jh",
            "interface": tune_interface_spec_to_json_value(&inner.interface),
            "planner_simulations_per_step": inner.planner_simulations_per_step,
            "return_horizon": inner.return_horizon,
            "warmstart_teacher_dataset_asset": inner.warmstart_teacher_dataset_asset,
            "label_phase_period": inner.label_phase_period,
        }),
    }
}

#[cfg(feature = "tuner")]
fn tune_parameter_range_to_json_value(range: &TuneParameterRangeSpec) -> serde_json::Value {
    serde_json::json!({
        "parameter": range.parameter,
        "min": range.min,
        "max": range.max,
    })
}

#[cfg(feature = "vm")]
fn vm_reward_policy_to_json_value(policy: &VmRewardPolicySpec) -> serde_json::Value {
    match policy {
        VmRewardPolicySpec::FromGuest => serde_json::json!({ "kind": "from_guest" }),
        VmRewardPolicySpec::Pattern {
            pattern,
            base_reward,
            bonus_reward,
        } => serde_json::json!({
            "kind": "pattern",
            "pattern": pattern,
            "base_reward": base_reward,
            "bonus_reward": bonus_reward,
        }),
    }
}

#[cfg(feature = "vm")]
fn vm_reward_shaping_to_json_value(spec: &VmRewardShapingSpec) -> serde_json::Value {
    match spec {
        VmRewardShapingSpec::EntropyReduction {
            baseline_asset,
            scale,
            crash_bonus,
            timeout_bonus,
        } => serde_json::json!({
            "kind": "entropy_reduction",
            "baseline_asset": baseline_asset,
            "scale": scale,
            "crash_bonus": crash_bonus,
            "timeout_bonus": timeout_bonus,
        }),
        VmRewardShapingSpec::TraceEntropy { scale, normalize } => serde_json::json!({
            "kind": "trace_entropy",
            "scale": scale,
            "normalize": normalize,
        }),
    }
}

#[cfg(feature = "vm")]
fn vm_action_source_to_json_value(spec: &VmRuntimeActionSourceSpec) -> serde_json::Value {
    match spec {
        VmRuntimeActionSourceSpec::Literal {
            names,
            payloads,
            encoding,
        } => serde_json::json!({
            "kind": "literal",
            "encoding": super::vm_payload_encoding_name(*encoding),
            "actions": payloads.iter().enumerate().map(|(idx, payload)| serde_json::json!({
                "name": names.get(idx).cloned().flatten(),
                "payload": payload,
            })).collect::<Vec<_>>(),
        }),
        VmRuntimeActionSourceSpec::Fuzz {
            seeds,
            encoding,
            mutators,
            min_len,
            max_len,
            dictionary,
            rng_seed,
        } => serde_json::json!({
            "kind": "fuzz",
            "encoding": super::vm_payload_encoding_name(*encoding),
            "seeds": seeds,
            "mutators": mutators.iter().map(|mutator| super::vm_fuzz_mutator_name(*mutator)).collect::<Vec<_>>(),
            "min_len": min_len,
            "max_len": max_len,
            "dictionary": dictionary,
            "rng_seed": rng_seed,
        }),
    }
}

#[cfg(feature = "vm")]
fn vm_action_filter_to_json_value(spec: &VmActionFilterSpec) -> serde_json::Value {
    serde_json::json!({
        "min_entropy": spec.min_entropy,
        "max_entropy": spec.max_entropy,
        "min_intrinsic_dependence": spec.min_intrinsic_dependence,
        "min_novelty": spec.min_novelty,
        "novelty_prior_asset": spec.novelty_prior_asset,
        "reject_reward": spec.reject_reward,
    })
}

#[cfg(feature = "vm")]
fn vm_trace_to_json_value(spec: &VmTraceSpec) -> serde_json::Value {
    serde_json::json!({
        "shared_region_name": spec.shared_region_name,
        "max_bytes": spec.max_bytes,
        "reset_on_episode": spec.reset_on_episode,
    })
}
