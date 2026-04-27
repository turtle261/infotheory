//! Canonical validation and compile pipeline for spec documents.

use super::{
    AssetBinding, CompiledPlannerController, CompiledPlannerRunSpec, CompiledTuneController,
    CompiledTuneSpec, ControllerSpec, EnvironmentSpec, PlannerInterfaceSpec, PlannerRunSpec,
    PlannerRuntimeSpec, ResolvedAssetBinding, SpecEnvironment, SpecError, SpecResult,
    TUNE_CANONICALIZATION_CLASSIFICATION_VERSION, TuneBoundsSpec, TuneControllerSpec, TuneSpec,
    ValidatedPlannerRunSpec, ValidatedTuneSpec,
};
use crate::aixi::common::{
    MctsStrategy, bits_for_cardinality, resolve_random_seed, validate_reward_encoding_bounds,
    warn_parallel_uct_workers_one_once,
};
use crate::spec::core::AssetRef;
use std::collections::HashMap;
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
            predictor_max_order: inner.predictor_max_order,
            agent_horizon: inner.agent_horizon,
            num_simulations: inner.num_simulations,
            mcts_strategy: inner.mcts_strategy,
            exploration_exploitation_ratio: inner.exploration_exploitation_ratio,
            discount_gamma: inner.discount_gamma,
        }),
        ControllerSpec::AiqiDiscounted(inner) => Ok(CompiledPlannerController::AiqiDiscounted {
            predictor: inner.predictor.validate_in(env)?.compile()?,
            predictor_max_order: inner.predictor_max_order,
            discount_gamma: inner.discount_gamma,
            return_horizon: inner.return_horizon,
            return_bins: inner.return_bins,
            augmentation_period: inner.augmentation_period,
            history_prune_keep_steps: inner.history_prune_keep_steps,
            baseline_exploration: inner.baseline_exploration,
        }),
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            Ok(CompiledPlannerController::AiqiWarmstartExactJh {
                predictor: inner.predictor.validate_in(env)?.compile()?,
                predictor_max_order: inner.predictor_max_order,
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
            if let Some(m_max) = bu_uct_m_max {
                if !(0.0 < m_max && m_max < 1.0) {
                    return Err(SpecError::new(
                        "controller.mcts_strategy.bu_uct_m_max must be in (0, 1)",
                    ));
                }
            }
            Ok(())
        }
    }
}

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
        action_bits: bits_for_cardinality(validated.canonical_spec().interface.agent_actions),
    })
}

pub(super) fn compile_tune_spec(spec: &TuneSpec, base_dir: &Path) -> SpecResult<CompiledTuneSpec> {
    let validated = spec.validate_in(&SpecEnvironment::new(base_dir))?;
    compile_validated_tune_spec(&validated)
}

pub(super) fn compile_validated_tune_spec(
    validated: &ValidatedTuneSpec,
) -> SpecResult<CompiledTuneSpec> {
    let env = SpecEnvironment::new(&validated.base_dir);
    Ok(CompiledTuneSpec {
        canonical_spec: validated.canonical_spec.clone(),
        canonical_bytes: validated.canonical_bytes().clone(),
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
    let controller = canonicalize_controller_spec(&spec.controller, env)?;
    let runtime = canonicalize_runtime_spec(&spec.runtime)?;
    Ok(PlannerRunSpec {
        assets: canonicalize_assets(&spec.assets),
        environment,
        interface,
        controller,
        runtime,
    })
}

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

fn canonicalize_tune_controller(
    controller: &TuneControllerSpec,
    assets: &[AssetBinding],
    env: &SpecEnvironment,
) -> SpecResult<TuneControllerSpec> {
    match controller {
        TuneControllerSpec::AnnealedHillClimbing(inner) => {
            if inner.max_mutation_radius == 0 {
                return Err(SpecError::new("max_mutation_radius must be >= 1"));
            }
            Ok(TuneControllerSpec::AnnealedHillClimbing(inner.clone()))
        }
        TuneControllerSpec::McAixiFacCtw(inner) => {
            validate_planner_interface_for_tuning(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            Ok(TuneControllerSpec::McAixiFacCtw(inner.clone()))
        }
        TuneControllerSpec::AiqiDiscounted(inner) => {
            validate_planner_interface_for_tuning(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 || !inner.return_bins.is_power_of_two() {
                return Err(SpecError::new("return_bins must be a power of two"));
            }
            if !(0.0..1.0).contains(&inner.discount_factor) {
                return Err(SpecError::new("discount_factor must be in [0, 1)"));
            }
            Ok(TuneControllerSpec::AiqiDiscounted(inner.clone()))
        }
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => {
            validate_planner_interface_for_tuning(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.label_phase_period < inner.return_horizon {
                return Err(SpecError::new(
                    "label_phase_period must be >= return_horizon",
                ));
            }
            ensure_asset_exists(assets, &inner.warmstart_teacher_dataset_asset)?;
            let _ = env;
            Ok(TuneControllerSpec::AiqiWarmstartExactJh(inner.clone()))
        }
    }
}

fn validate_planner_interface_for_tuning(spec: &PlannerInterfaceSpec) -> SpecResult<()> {
    canonicalize_interface_spec(spec).map(|_| ())
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

fn ensure_asset_exists(bindings: &[AssetBinding], id: &str) -> SpecResult<()> {
    if bindings.iter().any(|binding| binding.id == id) {
        Ok(())
    } else {
        Err(SpecError::new(format!("unknown asset id '{id}'")))
    }
}

fn canonicalize_interface_spec(spec: &PlannerInterfaceSpec) -> SpecResult<PlannerInterfaceSpec> {
    if spec.agent_actions == 0 {
        return Err(SpecError::new("agent_actions must be >= 1"));
    }
    if spec.observation_stream_len == 0 {
        return Err(SpecError::new("observation_stream_len must be >= 1"));
    }
    if spec.reward_bits == 0 {
        return Err(SpecError::new("reward_bits must be >= 1"));
    }
    validate_reward_encoding_bounds(
        spec.min_reward,
        spec.max_reward,
        spec.reward_offset,
        spec.reward_bits,
    )
    .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(spec.clone())
}

fn canonicalize_controller_spec(
    spec: &ControllerSpec,
    env: &SpecEnvironment,
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
            let validated_predictor = inner.predictor.validate_in(env)?;
            let predictor = validated_predictor.canonical_spec().clone();
            if validated_predictor.capabilities().contains_zpaq {
                return Err(SpecError::new(
                    "MC-AIXI strict generic rate_backend support requires reversible action conditioning; configured rate_backend contains zpaq which does not provide the reversible action conditioning required by \"A Monte-Carlo AIXI Approximation\"",
                ));
            }
            Ok(ControllerSpec::McAixi(super::McAixiControllerSpec {
                predictor,
                predictor_max_order: inner.predictor_max_order,
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
            if inner.return_bins == 0 || !inner.return_bins.is_power_of_two() {
                return Err(SpecError::new("return_bins must be a power of two"));
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
                    predictor_max_order: inner.predictor_max_order,
                    discount_gamma: inner.discount_gamma,
                    return_horizon: inner.return_horizon,
                    return_bins: inner.return_bins,
                    augmentation_period: inner.augmentation_period,
                    history_prune_keep_steps: inner.history_prune_keep_steps,
                    baseline_exploration: inner.baseline_exploration,
                },
            ))
        }
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 {
                return Err(SpecError::new("return_bins must be >= 1"));
            }
            if inner.label_phase_period < inner.return_horizon {
                return Err(SpecError::new(
                    "label_phase_period must be >= return_horizon",
                ));
            }
            let validated_predictor = inner.predictor.validate_in(env)?;
            let predictor = validated_predictor.canonical_spec().clone();
            Ok(ControllerSpec::AiqiWarmstartExactJh(
                super::WarmStartExactJhControllerSpec {
                    predictor,
                    predictor_max_order: inner.predictor_max_order,
                    return_horizon: inner.return_horizon,
                    return_bins: inner.return_bins,
                    label_phase_period: inner.label_phase_period,
                    teacher_dataset_asset: inner.teacher_dataset_asset.trim().to_string(),
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

fn finite_positive(value: f64, label: &str) -> SpecResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(SpecError::new(format!("{label} must be > 0")))
    }
}

fn nonzero_u64(value: u64, label: &str) -> SpecResult<u64> {
    if value > 0 {
        Ok(value)
    } else {
        Err(SpecError::new(format!("{label} must be > 0")))
    }
}

fn clean_optional_string(value: Option<&str>) -> Option<String> {
    value.and_then(|text| {
        let trimmed = text.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}
