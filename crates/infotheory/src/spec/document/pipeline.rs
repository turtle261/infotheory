//! Canonical validation and compile pipeline for spec documents.

use super::{
    AssetBinding, CompiledPlannerController, CompiledPlannerRunSpec, ControllerSpec,
    EnvironmentSpec, PlannerInterfaceSpec, PlannerRunSpec, PlannerRuntimeSpec,
    ResolvedAssetBinding, SpecEnvironment, SpecError, SpecResult, ValidatedPlannerRunSpec,
};
#[cfg(feature = "tuner")]
use super::{
    CompiledTuneController, CompiledTuneSpec, TUNE_CANONICALIZATION_CLASSIFICATION_VERSION,
    TuneBoundsSpec, TuneControllerSpec, TunePlannerInterfaceSpec, TuneSpec, ValidatedTuneSpec,
};
use crate::aixi::common::{
    MctsStrategy, bits_for_cardinality, byte_packed_percept_bits, resolve_random_seed,
    validate_aiqi_byte_packed_alignment, validate_mc_aixi_byte_packed_alignment,
    warn_parallel_uct_workers_one_once,
};
#[cfg(feature = "aixi")]
use crate::aixi::warmstart::{
    WarmStartExactJhError, max_reward_from_exact_return_bins, reward_bounds_from_exact_return_bins,
};
use crate::spec::core::AssetRef;
use std::collections::HashMap;
#[cfg(feature = "aixi")]
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "vm")]
use super::{VmEnvironmentSpec, VmRewardShapingSpec, VmRuntimeActionSourceSpec};

fn resolve_asset_bindings(
    bindings: &[AssetBinding],
    base_dir: &Path,
) -> Arc<[ResolvedAssetBinding]> {
    bindings
        .iter()
        .map(|binding| ResolvedAssetBinding {
            id: binding.id.clone(),
            asset: AssetRef::Filesystem(super::super::resolve_spec_path(base_dir, &binding.path)),
        })
        .collect::<Vec<_>>()
        .into()
}

fn compile_planner_controller(
    spec: &ControllerSpec,
    env: &SpecEnvironment,
) -> SpecResult<CompiledPlannerController> {
    match spec {
        ControllerSpec::McAixi(inner) => Ok(CompiledPlannerController::McAixi {
            predictor: inner.predictor.validate_in(env)?.compile()?,
            bit_stream_semantics: inner.bit_stream_semantics,
            agent_horizon: inner.agent_horizon,
            num_simulations: inner.num_simulations,
            mcts_strategy: inner.mcts_strategy,
            exploration_exploitation_ratio: inner.exploration_exploitation_ratio,
            discount_gamma: inner.discount_gamma,
        }),
        ControllerSpec::AiqiDiscounted(inner) => Ok(CompiledPlannerController::AiqiDiscounted {
            predictor: inner.predictor.validate_in(env)?.compile()?,
            bit_stream_semantics: inner.bit_stream_semantics,
            discount_gamma: inner.discount_gamma,
            return_horizon: inner.return_horizon,
            return_bins: inner.return_bins,
            augmentation_period: inner.augmentation_period,
            history_prune_keep_steps: inner.history_prune_keep_steps,
            baseline_exploration: inner.baseline_exploration,
        }),
        #[cfg(feature = "aixi")]
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            Ok(CompiledPlannerController::AiqiWarmstartExactJh {
                predictor: inner.predictor.validate_in(env)?.compile()?,
                bit_stream_semantics: inner.bit_stream_semantics,
                return_horizon: inner.return_horizon,
                return_bins: inner.return_bins,
                label_phase_period: inner.label_phase_period,
                teacher_dataset_asset: inner.teacher_dataset_asset.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
            })
        }
    }
}

fn validate_mc_aixi_mcts_strategy(strategy: MctsStrategy) -> SpecResult<()> {
    match strategy {
        MctsStrategy::RhoUct => Ok(()),
        MctsStrategy::ParallelUct {
            workers,
            bu_uct_m_max,
        } => {
            // `workers` is type-enforced non-zero by `NonZeroUsize`; only the
            // `workers == 1` warning and the BU-UCT threshold range remain
            // checkable at this layer.
            if workers.get() == 1 {
                warn_parallel_uct_workers_one_once();
            }
            if bu_uct_m_max.is_some_and(|m_max| !(0.0 < m_max && m_max < 1.0)) {
                return Err(SpecError::new(
                    "controller.mcts_strategy.bu_uct_m_max must be in (0, 1)",
                ));
            }
            Ok(())
        }
    }
}

#[cfg(feature = "aixi")]
fn validate_warmstart_direct_evaluator_marker(
    planner_simulations_per_step: usize,
) -> SpecResult<()> {
    if planner_simulations_per_step != 1 {
        return Err(SpecError::new(
            "planner_simulations_per_step must be exactly 1 for warm-start exact-J_H direct evaluation",
        ));
    }
    Ok(())
}

#[cfg(feature = "aixi")]
fn validate_warmstart_exact_reward_channel(
    return_horizon: usize,
    return_bins: usize,
    reward_bits: usize,
) -> SpecResult<()> {
    let (return_horizon, return_bins) =
        nonzero_warmstart_exact_return_shape(return_horizon, return_bins)?;
    if let Err(err) = reward_bounds_from_exact_return_bins(return_horizon, return_bins, reward_bits)
    {
        return match err {
            WarmStartExactJhError::RewardEncoding(err) => {
                let max_reward = max_reward_from_exact_return_bins(return_horizon, return_bins)
                    .map_err(|err| SpecError::new(err.to_string()))?;
                Err(SpecError::new(format!(
                    "return_bins imply max_reward={max_reward} for return_horizon={}, \
                     but that reward range is not representable by reward_bits={reward_bits}: {err}",
                    return_horizon.get()
                )))
            }
            err => Err(SpecError::new(err.to_string())),
        };
    }
    Ok(())
}

#[cfg(feature = "aixi")]
fn validate_warmstart_exact_return_bins(
    return_horizon: usize,
    return_bins: usize,
) -> SpecResult<()> {
    let (return_horizon, return_bins) =
        nonzero_warmstart_exact_return_shape(return_horizon, return_bins)?;
    max_reward_from_exact_return_bins(return_horizon, return_bins)
        .map(|_| ())
        .map_err(|err| SpecError::new(err.to_string()))
}

#[cfg(feature = "aixi")]
fn nonzero_warmstart_exact_return_shape(
    return_horizon: usize,
    return_bins: usize,
) -> SpecResult<(NonZeroUsize, NonZeroUsize)> {
    let return_horizon = NonZeroUsize::new(return_horizon)
        .ok_or_else(|| SpecError::new("return_horizon must be >= 1"))?;
    let return_bins =
        NonZeroUsize::new(return_bins).ok_or_else(|| SpecError::new("return_bins must be >= 1"))?;
    Ok((return_horizon, return_bins))
}

#[cfg(feature = "tuner")]
fn compile_tune_controller(spec: &TuneControllerSpec) -> CompiledTuneController {
    match spec {
        TuneControllerSpec::AnnealedHillClimbing(inner) => {
            CompiledTuneController::AnnealedHillClimbing(inner.clone())
        }
        TuneControllerSpec::McAixiFacCtw(inner) => {
            CompiledTuneController::McAixiFacCtw(inner.clone())
        }
        TuneControllerSpec::AiqiDiscounted(inner) => {
            CompiledTuneController::AiqiDiscounted(inner.clone())
        }
        #[cfg(feature = "aixi")]
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => {
            CompiledTuneController::AiqiWarmstartExactJh(inner.clone())
        }
    }
}

pub(super) fn compile_planner_run_spec(
    spec: &PlannerRunSpec,
    base_dir: &Path,
) -> SpecResult<CompiledPlannerRunSpec> {
    let validated = spec.validate_in(&SpecEnvironment::new(base_dir))?;
    compile_validated_planner_run_spec(&validated)
}

pub(super) fn compile_validated_planner_run_spec(
    validated: &ValidatedPlannerRunSpec,
) -> SpecResult<CompiledPlannerRunSpec> {
    let env = SpecEnvironment::new(&validated.base_dir);
    Ok(CompiledPlannerRunSpec {
        canonical_spec: validated.canonical_spec.clone(),
        canonical_bytes: validated.canonical_bytes().clone(),
        resolved_assets: resolve_asset_bindings(
            &validated.canonical_spec().assets,
            &validated.base_dir,
        ),
        interface: validated.canonical_spec().interface.clone(),
        runtime: validated.canonical_spec().runtime.clone(),
        controller: compile_planner_controller(&validated.canonical_spec().controller, &env)?,
        action_bits: bits_for_cardinality(validated.canonical_spec().interface.agent_actions.get()),
    })
}

#[cfg(feature = "tuner")]
pub(super) fn compile_tune_spec(spec: &TuneSpec, base_dir: &Path) -> SpecResult<CompiledTuneSpec> {
    let validated = spec.validate_in(&SpecEnvironment::new(base_dir))?;
    compile_validated_tune_spec(&validated)
}

#[cfg(feature = "tuner")]
pub(super) fn compile_validated_tune_spec(
    validated: &ValidatedTuneSpec,
) -> SpecResult<CompiledTuneSpec> {
    let env = SpecEnvironment::new(&validated.base_dir);
    Ok(CompiledTuneSpec {
        canonical_spec: validated.canonical_spec.clone(),
        canonical_bytes: validated.canonical_bytes().clone(),
        base_dir: validated.base_dir.clone(),
        resolved_assets: resolve_asset_bindings(
            &validated.canonical_spec().assets,
            &validated.base_dir,
        ),
        baseline_candidate: validated
            .canonical_spec()
            .baseline_candidate
            .validate_in(&env)?
            .compile()?,
        controller: compile_tune_controller(&validated.canonical_spec().controller),
        candidate_canonicalization_version: TUNE_CANONICALIZATION_CLASSIFICATION_VERSION,
    })
}

pub(super) fn canonicalize_planner_run(
    spec: &PlannerRunSpec,
    env: &SpecEnvironment,
) -> SpecResult<PlannerRunSpec> {
    validate_asset_bindings(&spec.assets)?;
    let environment = canonicalize_environment_spec(&spec.environment, &spec.assets, env)?;
    let interface = canonicalize_interface_spec(&spec.interface)?;
    let controller =
        canonicalize_controller_spec(&spec.controller, &spec.assets, env, Some(&interface))?;
    let runtime = canonicalize_runtime_spec(&spec.runtime)?;
    Ok(PlannerRunSpec {
        assets: canonicalize_assets(&spec.assets),
        environment,
        interface,
        controller,
        runtime,
    })
}

#[cfg(feature = "tuner")]
pub(super) fn canonicalize_tune_spec(
    spec: &TuneSpec,
    env: &SpecEnvironment,
) -> SpecResult<TuneSpec> {
    validate_asset_bindings(&spec.assets)?;
    ensure_asset_exists(&spec.assets, &spec.input_asset)?;
    let baseline = spec
        .baseline_candidate
        .validate_in(env)?
        .canonical_spec()
        .clone();
    let controller = canonicalize_tune_controller(&spec.controller, &spec.assets, env)?;
    validate_tune_bounds(&spec.bounds)?;
    Ok(TuneSpec {
        assets: canonicalize_assets(&spec.assets),
        input_asset: spec.input_asset.trim().to_string(),
        baseline_candidate: baseline,
        controller,
        bounds: canonicalize_tune_bounds(&spec.bounds),
        eval_time_limit_seconds: finite_positive(
            spec.eval_time_limit_seconds,
            "eval_time_limit_seconds",
        )?,
        time_budget_seconds: finite_positive(spec.time_budget_seconds, "time_budget_seconds")?,
        min_throughput_bytes_per_second: finite_positive(
            spec.min_throughput_bytes_per_second,
            "min_throughput_bytes_per_second",
        )?,
        max_memory_bytes: nonzero_u64(spec.max_memory_bytes, "max_memory_bytes")?,
        output_config_path: spec.output_config_path.trim().to_string(),
        seed: spec.seed,
        report_path: clean_optional_string(spec.report_path.as_deref()),
    })
}

#[cfg(feature = "tuner")]
fn canonicalize_tune_controller(
    controller: &TuneControllerSpec,
    assets: &[AssetBinding],
    _env: &SpecEnvironment,
) -> SpecResult<TuneControllerSpec> {
    match controller {
        TuneControllerSpec::AnnealedHillClimbing(inner) => {
            if inner.max_mutation_radius == 0 {
                return Err(SpecError::new("max_mutation_radius must be >= 1"));
            }
            Ok(TuneControllerSpec::AnnealedHillClimbing(inner.clone()))
        }
        TuneControllerSpec::McAixiFacCtw(inner) => {
            canonicalize_tune_interface_spec(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            Ok(TuneControllerSpec::McAixiFacCtw(inner.clone()))
        }
        TuneControllerSpec::AiqiDiscounted(inner) => {
            canonicalize_tune_interface_spec(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 {
                return Err(SpecError::new("return_bins must be >= 1"));
            }
            if !(0.0..1.0).contains(&inner.discount_factor) {
                return Err(SpecError::new("discount_factor must be in [0, 1)"));
            }
            if !inner.min_improvement.is_finite() {
                return Err(SpecError::new("min_improvement must be finite"));
            }
            if !inner.max_improvement.is_finite() {
                return Err(SpecError::new("max_improvement must be finite"));
            }
            if inner.max_improvement <= inner.min_improvement {
                return Err(SpecError::new(
                    "max_improvement must be greater than min_improvement",
                ));
            }
            Ok(TuneControllerSpec::AiqiDiscounted(inner.clone()))
        }
        #[cfg(feature = "aixi")]
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => {
            canonicalize_tune_interface_spec(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            validate_warmstart_direct_evaluator_marker(inner.planner_simulations_per_step)?;
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.label_phase_period < inner.return_horizon {
                return Err(SpecError::new(
                    "label_phase_period must be >= return_horizon",
                ));
            }
            ensure_asset_exists(assets, &inner.warmstart_teacher_dataset_asset)?;
            Ok(TuneControllerSpec::AiqiWarmstartExactJh(inner.clone()))
        }
    }
}

#[cfg(feature = "tuner")]
fn canonicalize_tune_interface_spec(
    spec: &TunePlannerInterfaceSpec,
) -> SpecResult<TunePlannerInterfaceSpec> {
    if spec.observation_stream_len == 0 {
        return Err(SpecError::new("observation_stream_len must be >= 1"));
    }
    if spec.reward_bits == 0 {
        return Err(SpecError::new("reward_bits must be >= 1"));
    }
    Ok(spec.clone())
}

fn canonicalize_assets(bindings: &[AssetBinding]) -> Vec<AssetBinding> {
    let mut out = bindings.to_vec();
    out.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.path.cmp(&b.path)));
    out
}

fn validate_asset_bindings(bindings: &[AssetBinding]) -> SpecResult<()> {
    let mut seen = HashMap::<&str, &str>::new();
    for binding in bindings {
        let id = binding.id.trim();
        let path = binding.path.trim();
        if id.is_empty() {
            return Err(SpecError::new("asset id cannot be empty"));
        }
        if path.is_empty() {
            return Err(SpecError::new(format!(
                "asset '{}' path cannot be empty",
                binding.id
            )));
        }
        if let Some(previous) = seen.insert(id, path)
            && previous != path
        {
            return Err(SpecError::new(format!(
                "asset '{}' is bound to more than one path",
                binding.id
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "aixi")]
fn ensure_asset_exists(bindings: &[AssetBinding], id: &str) -> SpecResult<()> {
    if bindings.iter().any(|binding| binding.id == id) {
        Ok(())
    } else {
        Err(SpecError::new(format!("unknown asset id '{id}'")))
    }
}

fn canonicalize_interface_spec(spec: &PlannerInterfaceSpec) -> SpecResult<PlannerInterfaceSpec> {
    if spec.observation_stream_len == 0 {
        return Err(SpecError::new("observation_stream_len must be >= 1"));
    }
    if spec.reward_bits == 0 {
        return Err(SpecError::new("reward_bits must be >= 1"));
    }
    Ok(spec.clone())
}

fn canonicalize_controller_spec(
    spec: &ControllerSpec,
    #[cfg_attr(not(feature = "aixi"), allow(unused_variables))] assets: &[AssetBinding],
    env: &SpecEnvironment,
    interface: Option<&PlannerInterfaceSpec>,
) -> SpecResult<ControllerSpec> {
    match spec {
        ControllerSpec::McAixi(inner) => {
            if inner.agent_horizon == 0 {
                return Err(SpecError::new("agent_horizon must be >= 1"));
            }
            if inner.num_simulations == 0 {
                return Err(SpecError::new("num_simulations must be >= 1"));
            }
            validate_mc_aixi_mcts_strategy(inner.mcts_strategy)?;
            if inner.exploration_exploitation_ratio <= 0.0 {
                return Err(SpecError::new("exploration_exploitation_ratio must be > 0"));
            }
            if !(0.0..=1.0).contains(&inner.discount_gamma) {
                return Err(SpecError::new("discount_gamma must be in [0, 1]"));
            }
            if matches!(
                inner.bit_stream_semantics,
                crate::api::BitStreamSemantics::BytePacked { .. }
            ) && let Some(interface) = interface
            {
                let action_bits = interface.agent_actions.action_bits();
                let percept_bits = byte_packed_percept_bits(
                    interface.observation_bits,
                    interface.observation_stream_len,
                    interface.reward_bits,
                );
                validate_mc_aixi_byte_packed_alignment(action_bits, percept_bits)
                    .map_err(SpecError::new)?;
            }
            let validated_predictor = inner.predictor.validate_in(env)?;
            let predictor = validated_predictor.canonical_spec().clone();
            if validated_predictor.capabilities().contains_zpaq {
                return Err(SpecError::new(
                    "MC-AIXI strict generic rate_backend support requires reversible action conditioning; configured rate_backend contains zpaq which does not provide the reversible action conditioning required by \"A Monte-Carlo AIXI Approximation\"",
                ));
            }
            Ok(ControllerSpec::McAixi(super::McAixiControllerSpec {
                predictor,
                bit_stream_semantics: inner.bit_stream_semantics,
                agent_horizon: inner.agent_horizon,
                num_simulations: inner.num_simulations,
                mcts_strategy: inner.mcts_strategy,
                exploration_exploitation_ratio: inner.exploration_exploitation_ratio,
                discount_gamma: inner.discount_gamma,
            }))
        }
        ControllerSpec::AiqiDiscounted(inner) => {
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 {
                return Err(SpecError::new("return_bins must be >= 1"));
            }
            if inner.augmentation_period < inner.return_horizon {
                return Err(SpecError::new(
                    "augmentation_period must be >= return_horizon",
                ));
            }
            if !(0.0 < inner.discount_gamma && inner.discount_gamma < 1.0) {
                return Err(SpecError::new("discount_gamma must be in (0, 1)"));
            }
            if !(0.0 < inner.baseline_exploration && inner.baseline_exploration <= 1.0) {
                return Err(SpecError::new("baseline_exploration must be in (0, 1]"));
            }
            if matches!(
                inner.bit_stream_semantics,
                crate::api::BitStreamSemantics::BytePacked { .. }
            ) && let Some(interface) = interface
            {
                let action_bits = interface.agent_actions.action_bits();
                let percept_bits = byte_packed_percept_bits(
                    interface.observation_bits,
                    interface.observation_stream_len,
                    interface.reward_bits,
                );
                let return_bits = crate::aixi::common::bits_for_cardinality(inner.return_bins);
                validate_aiqi_byte_packed_alignment(action_bits, percept_bits, return_bits)
                    .map_err(SpecError::new)?;
            }
            let validated_predictor = inner.predictor.validate_in(env)?;
            let predictor = validated_predictor.canonical_spec().clone();
            if !validated_predictor
                .capabilities()
                .supports_frozen_conditioning
            {
                return Err(SpecError::new(
                    "AIQI strict mode requires frozen context updates; configured rate_backend contains zpaq which does not provide strict frozen conditioning",
                ));
            }
            Ok(ControllerSpec::AiqiDiscounted(
                super::AiqiDiscountedControllerSpec {
                    predictor,
                    bit_stream_semantics: inner.bit_stream_semantics,
                    discount_gamma: inner.discount_gamma,
                    return_horizon: inner.return_horizon,
                    return_bins: inner.return_bins,
                    augmentation_period: inner.augmentation_period,
                    history_prune_keep_steps: inner.history_prune_keep_steps,
                    baseline_exploration: inner.baseline_exploration,
                },
            ))
        }
        #[cfg(feature = "aixi")]
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            validate_warmstart_direct_evaluator_marker(inner.planner_simulations_per_step)?;
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 {
                return Err(SpecError::new("return_bins must be >= 1"));
            }
            validate_warmstart_exact_return_bins(inner.return_horizon, inner.return_bins)?;
            if let Some(interface) = interface {
                validate_warmstart_exact_reward_channel(
                    inner.return_horizon,
                    inner.return_bins,
                    interface.reward_bits,
                )?;
            }
            if inner.label_phase_period < inner.return_horizon {
                return Err(SpecError::new(
                    "label_phase_period must be >= return_horizon",
                ));
            }
            if matches!(
                inner.bit_stream_semantics,
                crate::api::BitStreamSemantics::BytePacked { .. }
            ) && let Some(interface) = interface
            {
                let action_bits = interface.agent_actions.action_bits();
                let percept_bits = byte_packed_percept_bits(
                    interface.observation_bits,
                    interface.observation_stream_len,
                    interface.reward_bits,
                );
                let return_bits = crate::aixi::common::bits_for_cardinality(inner.return_bins);
                validate_aiqi_byte_packed_alignment(action_bits, percept_bits, return_bits)
                    .map_err(SpecError::new)?;
            }
            let validated_predictor = inner.predictor.validate_in(env)?;
            let predictor = validated_predictor.canonical_spec().clone();
            let teacher_dataset_asset = inner.teacher_dataset_asset.trim();
            if teacher_dataset_asset.is_empty() {
                return Err(SpecError::new("teacher_dataset_asset cannot be empty"));
            }
            ensure_asset_exists(assets, teacher_dataset_asset)?;
            Ok(ControllerSpec::AiqiWarmstartExactJh(
                super::WarmStartExactJhControllerSpec {
                    predictor,
                    bit_stream_semantics: inner.bit_stream_semantics,
                    return_horizon: inner.return_horizon,
                    return_bins: inner.return_bins,
                    label_phase_period: inner.label_phase_period,
                    teacher_dataset_asset: teacher_dataset_asset.to_string(),
                    planner_simulations_per_step: inner.planner_simulations_per_step,
                },
            ))
        }
    }
}

#[cfg(feature = "vm")]
fn canonicalize_vm_action_source(
    source: &VmRuntimeActionSourceSpec,
) -> SpecResult<VmRuntimeActionSourceSpec> {
    match source {
        VmRuntimeActionSourceSpec::Literal {
            names,
            payloads,
            encoding,
        } => Ok(VmRuntimeActionSourceSpec::Literal {
            names: names.clone(),
            payloads: payloads.clone(),
            encoding: *encoding,
        }),
        VmRuntimeActionSourceSpec::Fuzz {
            seeds,
            encoding,
            mutators,
            min_len,
            max_len,
            dictionary,
            rng_seed,
        } => {
            if seeds.is_empty() {
                return Err(SpecError::new(
                    "environment.action_source.seeds must include at least one seed in fuzz mode",
                ));
            }
            if mutators.is_empty() {
                return Err(SpecError::new(
                    "environment.action_source.mutators must include at least one mutator in fuzz mode",
                ));
            }
            if min_len > max_len {
                return Err(SpecError::new(
                    "environment.action_source.min_len cannot exceed max_len",
                ));
            }
            Ok(VmRuntimeActionSourceSpec::Fuzz {
                seeds: seeds.clone(),
                encoding: *encoding,
                mutators: mutators.clone(),
                min_len: *min_len,
                max_len: *max_len,
                dictionary: dictionary.clone(),
                rng_seed: *rng_seed,
            })
        }
    }
}

#[cfg(feature = "vm")]
fn canonicalize_vm_environment_spec(
    vm: &VmEnvironmentSpec,
    assets: &[AssetBinding],
    env: &SpecEnvironment,
) -> SpecResult<VmEnvironmentSpec> {
    ensure_asset_exists(assets, &vm.firecracker_config_asset)?;
    vm.stats_backend.validate_in(env)?;
    if let Some(shape) = &vm.reward_shaping
        && let VmRewardShapingSpec::EntropyReduction { baseline_asset, .. } = shape
    {
        ensure_asset_exists(assets, baseline_asset)?;
    }
    if let Some(filter) = &vm.action_filter
        && let Some(asset) = &filter.novelty_prior_asset
    {
        ensure_asset_exists(assets, asset)?;
    }
    if vm.episode_steps == 0 {
        return Err(SpecError::new("environment.episode_steps must be >= 1"));
    }
    let mut canonical = vm.clone();
    canonical.action_source = canonicalize_vm_action_source(&vm.action_source)?;
    Ok(canonical)
}

pub(super) fn canonicalize_environment_spec(
    spec: &EnvironmentSpec,
    _assets: &[AssetBinding],
    _env: &SpecEnvironment,
) -> SpecResult<EnvironmentSpec> {
    match spec {
        EnvironmentSpec::Builtin { builtin } => Ok(EnvironmentSpec::Builtin { builtin: *builtin }),
        #[cfg(feature = "vm")]
        EnvironmentSpec::NyxVm(vm) => Ok(EnvironmentSpec::NyxVm(canonicalize_vm_environment_spec(
            vm, _assets, _env,
        )?)),
    }
}

fn canonicalize_runtime_spec(spec: &PlannerRuntimeSpec) -> SpecResult<PlannerRuntimeSpec> {
    if spec.terminate_lifetime == 0 {
        return Err(SpecError::new("terminate_lifetime must be >= 1"));
    }
    if spec.log_every == 0 {
        return Err(SpecError::new("log_every must be >= 1"));
    }
    if spec.explore_epsilon < 0.0 {
        return Err(SpecError::new("explore_epsilon must be >= 0"));
    }
    if spec.explore_gamma <= 0.0 {
        return Err(SpecError::new("explore_gamma must be > 0"));
    }
    let mut canonical = spec.clone();
    canonical.random_seed = Some(resolve_random_seed(spec.random_seed));
    Ok(canonical)
}

#[cfg(feature = "tuner")]
fn validate_tune_bounds(bounds: &TuneBoundsSpec) -> SpecResult<()> {
    if bounds.max_experts == 0 {
        return Err(SpecError::new("max_experts must be >= 1"));
    }
    if bounds.max_mixture_nesting_depth == 0 {
        return Err(SpecError::new("max_mixture_nesting_depth must be >= 1"));
    }
    if let Some(min_experts) = bounds.min_experts
        && min_experts > bounds.max_experts
    {
        return Err(SpecError::new("min_experts cannot exceed max_experts"));
    }
    for range in &bounds.parameter_ranges {
        if range.parameter.trim().is_empty() {
            return Err(SpecError::new(
                "bounds.parameter_ranges[].parameter cannot be empty",
            ));
        }
        if !range.min.is_finite() || !range.max.is_finite() {
            return Err(SpecError::new(
                "bounds.parameter_ranges must use finite min/max values",
            ));
        }
        if range.min > range.max {
            return Err(SpecError::new(
                "bounds.parameter_ranges min cannot exceed max",
            ));
        }
    }
    for name in bounds
        .allowed_backends
        .iter()
        .chain(bounds.forbidden_backends.iter())
        .chain(bounds.required_experts.iter())
    {
        if name.trim().is_empty() {
            return Err(SpecError::new(
                "bounds backend/expert names cannot be empty",
            ));
        }
    }
    for name in &bounds.allowed_backends {
        if bounds
            .forbidden_backends
            .iter()
            .any(|forbidden| forbidden == name)
        {
            return Err(SpecError::new(
                "allowed_backends and forbidden_backends cannot overlap",
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "tuner")]
fn canonicalize_tune_bounds(bounds: &TuneBoundsSpec) -> TuneBoundsSpec {
    let mut allowed = bounds.allowed_backends.clone();
    allowed.sort();
    allowed.dedup();
    let mut forbidden_backends = bounds.forbidden_backends.clone();
    forbidden_backends.sort();
    forbidden_backends.dedup();
    let mut required = bounds.required_experts.clone();
    required.sort();
    required.dedup();
    let mut parameter_ranges = bounds.parameter_ranges.clone();
    parameter_ranges.sort_by(|a, b| a.parameter.cmp(&b.parameter));
    parameter_ranges
        .dedup_by(|a, b| a.parameter == b.parameter && a.min == b.min && a.max == b.max);
    let mut forbidden_pairs = bounds
        .forbidden_expert_pairs
        .iter()
        .map(|(a, b)| {
            if a <= b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            }
        })
        .collect::<Vec<_>>();
    forbidden_pairs.sort();
    forbidden_pairs.dedup();
    TuneBoundsSpec {
        allowed_backends: allowed,
        forbidden_backends,
        parameter_ranges,
        max_experts: bounds.max_experts,
        max_mixture_nesting_depth: bounds.max_mixture_nesting_depth,
        min_experts: bounds.min_experts,
        allow_duplicate_experts: bounds.allow_duplicate_experts,
        required_experts: required,
        forbidden_expert_pairs: forbidden_pairs,
    }
}

#[cfg(feature = "tuner")]
fn finite_positive(value: f64, label: &str) -> SpecResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(SpecError::new(format!("{label} must be > 0")))
    }
}

#[cfg(feature = "tuner")]
fn nonzero_u64(value: u64, label: &str) -> SpecResult<u64> {
    if value > 0 {
        Ok(value)
    } else {
        Err(SpecError::new(format!("{label} must be > 0")))
    }
}

#[cfg(feature = "tuner")]
fn clean_optional_string(value: Option<&str>) -> Option<String> {
    value.and_then(|text| {
        let trimmed = text.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "backend-ctw")]
    use crate::aixi::common::MctsStrategy;
    use crate::aixi::common::{ActionAlphabet, ObservationKeyMode};
    #[cfg(all(feature = "backend-ctw", feature = "tuner"))]
    use crate::api::CompressionBackend;
    #[cfg(feature = "backend-ctw")]
    use crate::api::RateBackend;
    #[cfg(feature = "backend-ctw")]
    use std::num::NonZeroUsize;

    fn action_alphabet(n: usize) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
    }

    fn sample_interface() -> PlannerInterfaceSpec {
        PlannerInterfaceSpec {
            observation_bits: 8,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 8,
            agent_actions: action_alphabet(2),
        }
    }

    #[cfg(all(feature = "backend-ctw", feature = "tuner"))]
    fn sample_tune_interface() -> TunePlannerInterfaceSpec {
        TunePlannerInterfaceSpec {
            observation_bits: 8,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 8,
            agent_actions: action_alphabet(2),
        }
    }

    #[test]
    fn asset_binding_validation_and_sorting_are_stable() {
        let sorted = canonicalize_assets(&[
            AssetBinding {
                id: "b".to_string(),
                path: "b.bin".to_string(),
            },
            AssetBinding {
                id: "a".to_string(),
                path: "a.bin".to_string(),
            },
        ]);
        assert_eq!(sorted[0].id, "a");
        assert_eq!(sorted[1].id, "b");

        validate_asset_bindings(&[
            AssetBinding {
                id: "dataset".to_string(),
                path: "one.bin".to_string(),
            },
            AssetBinding {
                id: "dataset".to_string(),
                path: "one.bin".to_string(),
            },
        ])
        .expect("duplicate identical bindings are benign");

        let err = validate_asset_bindings(&[AssetBinding {
            id: " ".to_string(),
            path: "x".to_string(),
        }])
        .expect_err("blank asset id must fail");
        assert!(err.to_string().contains("asset id cannot be empty"));

        let err = validate_asset_bindings(&[AssetBinding {
            id: "dataset".to_string(),
            path: " ".to_string(),
        }])
        .expect_err("blank asset path must fail");
        assert!(
            err.to_string()
                .contains("asset 'dataset' path cannot be empty")
        );

        let err = validate_asset_bindings(&[
            AssetBinding {
                id: "dataset".to_string(),
                path: "one.bin".to_string(),
            },
            AssetBinding {
                id: "dataset".to_string(),
                path: "two.bin".to_string(),
            },
        ])
        .expect_err("conflicting asset bindings must fail");
        assert!(
            err.to_string()
                .contains("asset 'dataset' is bound to more than one path")
        );

        #[cfg(feature = "aixi")]
        {
            ensure_asset_exists(
                &[AssetBinding {
                    id: "dataset".to_string(),
                    path: "one.bin".to_string(),
                }],
                "dataset",
            )
            .expect("known asset id");
            let err = ensure_asset_exists(&[], "missing").expect_err("missing asset must fail");
            assert!(err.to_string().contains("unknown asset id 'missing'"));
        }
    }

    #[test]
    fn interface_runtime_and_scalar_validators_enforce_contracts() {
        canonicalize_interface_spec(&sample_interface()).expect("valid interface");

        let mut bad_interface = sample_interface();
        bad_interface.observation_stream_len = 0;
        let err = canonicalize_interface_spec(&bad_interface)
            .expect_err("zero observation stream length must fail");
        assert!(
            err.to_string()
                .contains("observation_stream_len must be >= 1")
        );

        bad_interface = sample_interface();
        bad_interface.reward_bits = 0;
        let err =
            canonicalize_interface_spec(&bad_interface).expect_err("zero reward bits must fail");
        assert!(err.to_string().contains("reward_bits must be >= 1"));

        let runtime = canonicalize_runtime_spec(&PlannerRuntimeSpec {
            random_seed: None,
            learn_cycles: None,
            eval_cycles: None,
            terminate_lifetime: 4,
            log_every: 2,
            perf: false,
            vm_perf_only: false,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        })
        .expect("valid runtime");
        assert_eq!(runtime.random_seed, Some(resolve_random_seed(None)));

        let err = canonicalize_runtime_spec(&PlannerRuntimeSpec {
            terminate_lifetime: 0,
            ..runtime.clone()
        })
        .expect_err("zero terminate_lifetime must fail");
        assert!(err.to_string().contains("terminate_lifetime must be >= 1"));

        let err = canonicalize_runtime_spec(&PlannerRuntimeSpec {
            log_every: 0,
            ..runtime.clone()
        })
        .expect_err("zero log_every must fail");
        assert!(err.to_string().contains("log_every must be >= 1"));

        let err = canonicalize_runtime_spec(&PlannerRuntimeSpec {
            explore_epsilon: -0.1,
            ..runtime.clone()
        })
        .expect_err("negative explore_epsilon must fail");
        assert!(err.to_string().contains("explore_epsilon must be >= 0"));

        let err = canonicalize_runtime_spec(&PlannerRuntimeSpec {
            explore_gamma: 0.0,
            ..runtime
        })
        .expect_err("non-positive explore_gamma must fail");
        assert!(err.to_string().contains("explore_gamma must be > 0"));

        #[cfg(feature = "tuner")]
        {
            assert_eq!(finite_positive(0.5, "x").expect("positive finite"), 0.5);
            assert!(finite_positive(f64::INFINITY, "x").is_err());
            assert_eq!(nonzero_u64(7, "y").expect("nonzero"), 7);
            assert!(nonzero_u64(0, "y").is_err());
            assert_eq!(
                clean_optional_string(Some("  trimmed  ")),
                Some("trimmed".to_string())
            );
            assert_eq!(clean_optional_string(Some("   ")), None);
            assert_eq!(clean_optional_string(None), None);
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_controller_validation_covers_mc_aixi_and_aiqi_contracts() {
        let env = SpecEnvironment::default();
        let workers = NonZeroUsize::new(2).expect("non-zero workers");

        canonicalize_controller_spec(
            &ControllerSpec::McAixi(super::super::McAixiControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                agent_horizon: 2,
                num_simulations: 8,
                mcts_strategy: MctsStrategy::ParallelUct {
                    workers,
                    bu_uct_m_max: Some(0.5),
                },
                exploration_exploitation_ratio: 1.0,
                discount_gamma: 0.8,
            }),
            &[],
            &env,
            None,
        )
        .expect("valid MC-AIXI controller");

        let err = match canonicalize_controller_spec(
            &ControllerSpec::McAixi(super::super::McAixiControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                agent_horizon: 0,
                num_simulations: 8,
                mcts_strategy: MctsStrategy::RhoUct,
                exploration_exploitation_ratio: 1.0,
                discount_gamma: 0.8,
            }),
            &[],
            &env,
            None,
        ) {
            Ok(_) => panic!("zero horizon must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("agent_horizon must be >= 1"));

        let err = match canonicalize_controller_spec(
            &ControllerSpec::AiqiDiscounted(super::super::AiqiDiscountedControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                discount_gamma: 1.0,
                return_horizon: 2,
                return_bins: 8,
                augmentation_period: 2,
                history_prune_keep_steps: None,
                baseline_exploration: 0.1,
            }),
            &[],
            &env,
            None,
        ) {
            Ok(_) => panic!("discount_gamma=1 must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("discount_gamma must be in (0, 1)"));
    }

    #[cfg(all(feature = "backend-ctw", feature = "aixi"))]
    #[test]
    fn planner_controller_validation_covers_warmstart_contract() {
        let env = SpecEnvironment::default();
        let warmstart_assets = vec![AssetBinding {
            id: "teacher-ds".to_string(),
            path: "teacher.json".to_string(),
        }];

        let warmstart = canonicalize_controller_spec(
            &ControllerSpec::AiqiWarmstartExactJh(super::super::WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                return_horizon: 2,
                return_bins: 5,
                label_phase_period: 3,
                teacher_dataset_asset: "  teacher-ds  ".to_string(),
                planner_simulations_per_step: 1,
            }),
            &warmstart_assets,
            &env,
            Some(&sample_interface()),
        )
        .expect("valid warmstart controller");
        match warmstart {
            ControllerSpec::AiqiWarmstartExactJh(inner) => {
                assert_eq!(inner.teacher_dataset_asset, "teacher-ds");
            }
            _ => panic!("expected warmstart controller"),
        }

        let err = match canonicalize_controller_spec(
            &ControllerSpec::AiqiWarmstartExactJh(super::super::WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                return_horizon: 4,
                return_bins: 8,
                label_phase_period: 4,
                teacher_dataset_asset: "teacher-ds".to_string(),
                planner_simulations_per_step: 1,
            }),
            &warmstart_assets,
            &env,
            Some(&sample_interface()),
        ) {
            Ok(_) => panic!("warmstart slack return bins must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("return_bins must be exactly H * max_reward + 1"),
            "{err}"
        );

        let err = match canonicalize_controller_spec(
            &ControllerSpec::AiqiWarmstartExactJh(super::super::WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                return_horizon: 2,
                return_bins: 513,
                label_phase_period: 3,
                teacher_dataset_asset: "teacher-ds".to_string(),
                planner_simulations_per_step: 1,
            }),
            &warmstart_assets,
            &env,
            Some(&PlannerInterfaceSpec {
                reward_bits: 8,
                ..sample_interface()
            }),
        ) {
            Ok(_) => panic!("warmstart reward_bits too narrow must fail at spec time"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("not representable by reward_bits=8"),
            "{err}"
        );

        let err = match canonicalize_controller_spec(
            &ControllerSpec::AiqiWarmstartExactJh(super::super::WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                return_horizon: 2,
                return_bins: 5,
                label_phase_period: 3,
                teacher_dataset_asset: "missing-teacher".to_string(),
                planner_simulations_per_step: 1,
            }),
            &warmstart_assets,
            &env,
            Some(&sample_interface()),
        ) {
            Ok(_) => panic!("warmstart missing teacher asset must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("unknown asset id 'missing-teacher'"),
            "{err}"
        );

        let err = match canonicalize_controller_spec(
            &ControllerSpec::AiqiWarmstartExactJh(super::super::WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                return_horizon: 2,
                return_bins: 5,
                label_phase_period: 3,
                teacher_dataset_asset: "  ".to_string(),
                planner_simulations_per_step: 1,
            }),
            &warmstart_assets,
            &env,
            Some(&sample_interface()),
        ) {
            Ok(_) => panic!("warmstart blank teacher asset must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("teacher_dataset_asset cannot be empty"),
            "{err}"
        );

        let err = match canonicalize_controller_spec(
            &ControllerSpec::AiqiWarmstartExactJh(super::super::WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 4 },
                bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
                return_horizon: 2,
                return_bins: 5,
                label_phase_period: 3,
                teacher_dataset_asset: "teacher-ds".to_string(),
                planner_simulations_per_step: 2,
            }),
            &warmstart_assets,
            &env,
            Some(&sample_interface()),
        ) {
            Ok(_) => panic!("warmstart non-direct planner marker must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("planner_simulations_per_step must be exactly 1"),
            "{err}"
        );
    }

    #[cfg(all(feature = "backend-ctw", feature = "tuner"))]
    #[test]
    fn tune_controller_and_bounds_validation_cover_semantic_errors() {
        let assets = vec![AssetBinding {
            id: "teacher".to_string(),
            path: "teacher.bin".to_string(),
        }];
        let env = SpecEnvironment::default();

        let annealed = canonicalize_tune_controller(
            &TuneControllerSpec::AnnealedHillClimbing(
                super::super::AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: 2,
                },
            ),
            &assets,
            &env,
        )
        .expect("valid annealed controller");
        assert!(matches!(
            annealed,
            TuneControllerSpec::AnnealedHillClimbing(_)
        ));

        canonicalize_tune_controller(
            &TuneControllerSpec::AiqiDiscounted(super::super::AiqiDiscountedTuneControllerSpec {
                interface: sample_tune_interface(),
                planner_simulations_per_step: 2,
                return_horizon: 2,
                return_bins: 3,
                discount_factor: 0.5,
                min_improvement: -1.0,
                max_improvement: 1.0,
            }),
            &assets,
            &env,
        )
        .expect("non-power-of-two bins are valid for discounted AIQI");

        let err = canonicalize_tune_controller(
            &TuneControllerSpec::AiqiDiscounted(super::super::AiqiDiscountedTuneControllerSpec {
                interface: sample_tune_interface(),
                planner_simulations_per_step: 2,
                return_horizon: 2,
                return_bins: 4,
                discount_factor: 0.5,
                min_improvement: f64::NAN,
                max_improvement: 1.0,
            }),
            &assets,
            &env,
        )
        .expect_err("NaN min_improvement must fail");
        assert!(err.to_string().contains("min_improvement must be finite"));

        let err = canonicalize_tune_controller(
            &TuneControllerSpec::AiqiDiscounted(super::super::AiqiDiscountedTuneControllerSpec {
                interface: sample_tune_interface(),
                planner_simulations_per_step: 2,
                return_horizon: 2,
                return_bins: 4,
                discount_factor: 0.5,
                min_improvement: f64::INFINITY,
                max_improvement: 1.0,
            }),
            &assets,
            &env,
        )
        .expect_err("infinite min_improvement must fail");
        assert!(err.to_string().contains("min_improvement must be finite"));

        let err = canonicalize_tune_controller(
            &TuneControllerSpec::AiqiDiscounted(super::super::AiqiDiscountedTuneControllerSpec {
                interface: sample_tune_interface(),
                planner_simulations_per_step: 2,
                return_horizon: 2,
                return_bins: 4,
                discount_factor: 0.5,
                min_improvement: -1.0,
                max_improvement: f64::NAN,
            }),
            &assets,
            &env,
        )
        .expect_err("NaN max_improvement must fail");
        assert!(err.to_string().contains("max_improvement must be finite"));

        let err = canonicalize_tune_controller(
            &TuneControllerSpec::AiqiDiscounted(super::super::AiqiDiscountedTuneControllerSpec {
                interface: sample_tune_interface(),
                planner_simulations_per_step: 2,
                return_horizon: 2,
                return_bins: 4,
                discount_factor: 0.5,
                min_improvement: -1.0,
                max_improvement: f64::INFINITY,
            }),
            &assets,
            &env,
        )
        .expect_err("infinite max_improvement must fail");
        assert!(err.to_string().contains("max_improvement must be finite"));

        validate_tune_bounds(&TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string()],
            forbidden_backends: vec!["zpaq".to_string()],
            parameter_ranges: vec![super::super::TuneParameterRangeSpec {
                parameter: "alpha".to_string(),
                min: 0.1,
                max: 0.2,
            }],
            max_experts: 4,
            max_mixture_nesting_depth: 2,
            min_experts: Some(1),
            allow_duplicate_experts: Some(false),
            required_experts: vec!["ctw".to_string()],
            forbidden_expert_pairs: vec![("zpaq".to_string(), "ctw".to_string())],
        })
        .expect("valid bounds");

        let canonical = canonicalize_tune_bounds(&TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string(), "ctw".to_string(), "rosa".to_string()],
            forbidden_backends: vec!["zpaq".to_string(), "zpaq".to_string()],
            parameter_ranges: vec![
                super::super::TuneParameterRangeSpec {
                    parameter: "beta".to_string(),
                    min: 0.2,
                    max: 0.4,
                },
                super::super::TuneParameterRangeSpec {
                    parameter: "alpha".to_string(),
                    min: 0.1,
                    max: 0.3,
                },
            ],
            max_experts: 4,
            max_mixture_nesting_depth: 2,
            min_experts: Some(1),
            allow_duplicate_experts: Some(false),
            required_experts: vec!["rosa".to_string(), "rosa".to_string()],
            forbidden_expert_pairs: vec![
                ("zpaq".to_string(), "ctw".to_string()),
                ("ctw".to_string(), "zpaq".to_string()),
            ],
        });
        assert_eq!(canonical.allowed_backends, vec!["ctw", "rosa"]);
        assert_eq!(canonical.forbidden_backends, vec!["zpaq"]);
        assert_eq!(canonical.required_experts, vec!["rosa"]);
        assert_eq!(
            canonical.forbidden_expert_pairs,
            vec![("ctw".to_string(), "zpaq".to_string())]
        );

        let err = validate_tune_bounds(&TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string()],
            forbidden_backends: vec!["ctw".to_string()],
            parameter_ranges: vec![],
            max_experts: 4,
            max_mixture_nesting_depth: 2,
            min_experts: Some(1),
            allow_duplicate_experts: None,
            required_experts: vec![],
            forbidden_expert_pairs: vec![],
        })
        .expect_err("overlapping allow/forbid bounds must fail");
        assert!(
            err.to_string()
                .contains("allowed_backends and forbidden_backends cannot overlap")
        );

        let compiled = compile_tune_controller(&TuneControllerSpec::AnnealedHillClimbing(
            super::super::AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: 2,
            },
        ));
        assert!(matches!(
            compiled,
            CompiledTuneController::AnnealedHillClimbing(_)
        ));

        let baseline = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 4 },
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Framed,
        };
        let tune = TuneSpec {
            assets: assets.clone(),
            input_asset: "teacher".to_string(),
            baseline_candidate: baseline,
            controller: TuneControllerSpec::AnnealedHillClimbing(
                super::super::AnnealedHillClimbingTuneControllerSpec {
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
            time_budget_seconds: 5.0,
            min_throughput_bytes_per_second: 1.0,
            max_memory_bytes: 1024,
            output_config_path: " out.json ".to_string(),
            seed: 9,
            report_path: Some(" report.json ".to_string()),
        };
        let compiled = compile_tune_spec(&tune, Path::new(".")).expect("compile tune spec");
        assert_eq!(compiled.canonical_spec().output_config_path, "out.json");
    }

    #[cfg(all(feature = "backend-ctw", feature = "tuner"))]
    #[test]
    fn tune_controller_validation_covers_warmstart_contract() {
        let assets = vec![AssetBinding {
            id: "teacher".to_string(),
            path: "teacher.bin".to_string(),
        }];
        let env = SpecEnvironment::default();

        let err = canonicalize_tune_controller(
            &TuneControllerSpec::AiqiWarmstartExactJh(
                super::super::WarmStartExactJhTuneControllerSpec {
                    interface: sample_tune_interface(),
                    planner_simulations_per_step: 1,
                    return_horizon: 2,
                    warmstart_teacher_dataset_asset: "missing".to_string(),
                    label_phase_period: 3,
                },
            ),
            &assets,
            &env,
        )
        .expect_err("missing teacher asset must fail");
        assert!(err.to_string().contains("unknown asset id 'missing'"));

        let err = canonicalize_tune_controller(
            &TuneControllerSpec::AiqiWarmstartExactJh(
                super::super::WarmStartExactJhTuneControllerSpec {
                    interface: sample_tune_interface(),
                    planner_simulations_per_step: 2,
                    return_horizon: 2,
                    warmstart_teacher_dataset_asset: "teacher".to_string(),
                    label_phase_period: 3,
                },
            ),
            &assets,
            &env,
        )
        .expect_err("warmstart tune non-direct planner marker must fail");
        assert!(
            err.to_string()
                .contains("planner_simulations_per_step must be exactly 1"),
            "{err}"
        );
    }
}
