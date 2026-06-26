//! Compiled-spec bindings for planner-run documents.
//!
//! This module keeps filesystem asset loading and environment construction at
//! the edge of the planner runtime. Controller execution lives in
//! [`crate::aixi::planner_agent`].

use crate::aixi::common::ActionAlphabet;
use crate::aixi::environment::Environment;
#[cfg(feature = "aixi-gameengine")]
use crate::aixi::gameengine::build_builtin_environment as build_gameengine_builtin_environment;
#[cfg(feature = "vm")]
use crate::aixi::vm_nyx::{NyxVmConfig, NyxVmEnvironment};
use crate::aixi::warmstart::{
    WarmStartExactJhTeacherDataset, validate_warmstart_teacher_dataset_for_compiled_planner_run,
};
use crate::spec::{self, AssetRef, BuiltinEnvironmentSpec, CompiledPlannerRunSpec, SpecDocument};
use std::path::Path;

/// Load and validate a warm-start teacher dataset asset referenced by a compiled planner run.
pub fn load_warmstart_exact_jh_teacher_dataset(
    compiled: &CompiledPlannerRunSpec,
    asset_id: &str,
) -> anyhow::Result<WarmStartExactJhTeacherDataset> {
    let binding = compiled
        .resolved_assets()
        .iter()
        .find(|entry| entry.id == asset_id)
        .ok_or_else(|| anyhow::anyhow!("unknown warm-start teacher_dataset_asset '{asset_id}'"))?;
    let AssetRef::Filesystem(path) = &binding.asset;
    let bytes = std::fs::read(path).map_err(|err| {
        anyhow::anyhow!(
            "failed to read warm-start teacher_dataset_asset '{}': {err}",
            path.display()
        )
    })?;
    let teacher =
        WarmStartExactJhTeacherDataset::from_json_slice(&bytes).map_err(anyhow::Error::msg)?;
    validate_warmstart_teacher_dataset_for_compiled_planner_run(compiled, &teacher)
        .map_err(anyhow::Error::msg)?;
    Ok(teacher)
}

/// Validate a teacher dataset contract against a compiled planner run.
pub fn validate_warmstart_exact_jh_teacher_contract(
    compiled: &CompiledPlannerRunSpec,
    teacher: &WarmStartExactJhTeacherDataset,
) -> anyhow::Result<()> {
    crate::aixi::warmstart::validate_warmstart_teacher_against_compiled_planner_run(
        compiled,
        &teacher.contract,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))
}

/// Compile a planner_run document from a filesystem path.
pub fn compile_planner_run_document(
    path: &str,
    caller: &str,
) -> anyhow::Result<CompiledPlannerRunSpec> {
    let config_dir = Path::new(path).parent().unwrap_or(Path::new("."));
    let document = spec::load_spec_document(path).map_err(anyhow::Error::msg)?;
    match document {
        SpecDocument::PlannerRun(spec) => spec
            .compile_in(&spec::SpecEnvironment::new(config_dir))
            .map_err(anyhow::Error::msg),
        other => Err(anyhow::anyhow!(
            "{caller} expects a planner_run document, found kind '{}'",
            other.kind_str()
        )),
    }
}

fn build_builtin_environment(spec: BuiltinEnvironmentSpec) -> anyhow::Result<Box<dyn Environment>> {
    #[cfg(feature = "aixi-gameengine")]
    {
        build_gameengine_builtin_environment(spec).map_err(anyhow::Error::new)
    }
    #[cfg(not(feature = "aixi-gameengine"))]
    {
        Err(anyhow::anyhow!(
            "builtin environment '{}' requires feature 'aixi-gameengine'",
            spec.canonical_name()
        ))
    }
}

/// Build the environment declared by a compiled planner run.
pub fn build_planner_environment(
    compiled: &CompiledPlannerRunSpec,
) -> anyhow::Result<(Box<dyn Environment>, &'static str)> {
    #[allow(unreachable_patterns)]
    match &compiled.canonical_spec().environment {
        spec::EnvironmentSpec::Builtin { builtin } => Ok((
            build_builtin_environment(*builtin)?,
            builtin.canonical_name(),
        )),
        #[cfg(feature = "vm")]
        spec::EnvironmentSpec::NyxVm(vm) => {
            let config = NyxVmConfig::from_environment_spec(vm, compiled.resolved_assets())
                .map_err(anyhow::Error::msg)?;
            Ok((Box::new(NyxVmEnvironment::new(config)?), "vm"))
        }
        #[cfg(not(feature = "vm"))]
        other => Err(anyhow::anyhow!(
            "unsupported environment variant '{}' in this build",
            other.kind_str()
        )),
    }
}

/// Validate that the environment action alphabet matches the planner interface.
pub fn validate_action_alphabet(
    compiled: &CompiledPlannerRunSpec,
    env: &dyn Environment,
) -> anyhow::Result<()> {
    let actual: ActionAlphabet = env.get_num_actions();
    let expected: ActionAlphabet = compiled.interface().agent_actions;
    if actual != expected {
        return Err(anyhow::anyhow!(
            "action_alphabet_mismatch: planner interface declares {} actions but environment exposes {}",
            expected,
            actual
        ));
    }
    Ok(())
}

/// Validate the full environment interface against the compiled planner contract.
pub(crate) fn validate_environment_interface(
    compiled: &CompiledPlannerRunSpec,
    env: &dyn Environment,
) -> anyhow::Result<()> {
    validate_action_alphabet(compiled, env)?;
    let interface = compiled.interface();
    let expected_action_bits: usize = interface.agent_actions.action_bits();
    let actual_action_bits: usize = env.get_action_bits();
    if actual_action_bits != expected_action_bits {
        return Err(anyhow::anyhow!(
            "action_bits_mismatch: planner interface declares {} action bits for {} actions but environment exposes {}",
            expected_action_bits,
            interface.agent_actions,
            actual_action_bits
        ));
    }
    let actual_observation_bits: usize = env.get_observation_bits();
    if actual_observation_bits != interface.observation_bits {
        return Err(anyhow::anyhow!(
            "observation_bits_mismatch: planner interface declares {} observation bits but environment exposes {}",
            interface.observation_bits,
            actual_observation_bits
        ));
    }
    let actual_reward_bits: usize = env.get_reward_bits();
    if actual_reward_bits != interface.reward_bits {
        return Err(anyhow::anyhow!(
            "reward_bits_mismatch: planner interface declares {} reward bits but environment exposes {}",
            interface.reward_bits,
            actual_reward_bits
        ));
    }
    Ok(())
}
