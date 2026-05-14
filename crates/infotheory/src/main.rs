//! # InfoTheory CLI
//!
//! Command-line interface for the `infotheory` library.
//! Provides access to compression-based (NCD) and entropy-based (Shannon, ROSA, CTW)
//! estimators for files, as well as AIXI agents.
//!
//! ## Usage
//!
//! ### Single-file mode:
//! ```bash
//! infotheory <primitive> <file1> <file2>
//! ```
//!
//! ### Search mode:
//! ```bash
//! infotheory search <query> <target> [options]
//! ```
//!
//! ### AIXI Agent mode:
//! ```bash
//! infotheory aixi <config.json>
//! ```
//!
//! ### Batch JSON mode (for programmatic use):
//! ```bash
//! infotheory batch < input.json > output.json
//! echo '{"op":"metrics","text":"hello world"}' | infotheory batch
//! ```
//!
//! See `print_usage` for details on supported primitives.

mod cli;

use infotheory::aixi::agent::Agent;
use infotheory::aixi::aiqi::AiqiAgent;
#[cfg(test)]
use infotheory::aixi::common::ObservationKeyMode;
use infotheory::aixi::common::{
    ActionAlphabet, EXPLORE_RANDOM_SALT, RandomGenerator, resolve_random_seed,
};
use infotheory::aixi::environment::Environment;
#[cfg(feature = "aixi-gameengine")]
use infotheory::aixi::gameengine::build_builtin_environment as build_gameengine_builtin_environment;
#[cfg(all(test, feature = "vm"))]
use infotheory::aixi::vm_nyx::{
    FuzzMutator as NyxFuzzMutator, NyxActionFilter, NyxActionSource, NyxActionSpec, NyxFuzzConfig,
    NyxObservationPolicy, NyxObservationStreamMode, NyxProtocolConfig, NyxRewardPolicy,
    NyxRewardShaping, NyxTraceConfig, PayloadEncoding as NyxPayloadEncoding,
};
#[cfg(feature = "vm")]
use infotheory::aixi::vm_nyx::{NyxVmConfig, NyxVmEnvironment};
use infotheory::aixi::warmstart::{WarmStartExactJhAgent, WarmStartExactJhTeacherDataset};
use infotheory::api::*;
#[cfg(feature = "backend-mamba")]
use infotheory::mambazip;
#[cfg(feature = "backend-rwkv")]
use infotheory::rwkvzip;
#[cfg(feature = "backend-sequitur")]
use infotheory::sequitur::{CanonicalSymbol, SequiturModel};
use infotheory::spec::{
    self, AssetRef, BuiltinEnvironmentSpec, CompiledPlannerController, CompiledPlannerRunSpec,
    PlannerRuntimeSpec, SpecDocument,
};
#[cfg(all(test, feature = "vm"))]
use nyx_lite::SharedMemoryPolicy;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufWriter, IsTerminal, Read, Write};
use std::path::Path;
#[cfg(feature = "vm")]
use std::time::Instant;

#[cfg(not(feature = "vm"))]
use std::time::Instant;

#[cfg(all(test, feature = "all-backends"))]
use crate::cli::load_expert_spec;
#[cfg(all(test, feature = "vm"))]
use crate::cli::parse_vm_stats_backend;
use crate::cli::{
    CliBackendInvocation, CliBackendSourceFlags, build_ctx_invocation,
    file_roundtrip_compiled_backend, load_mixture_spec, maybe_export_online_model,
    parse_compression_backend, parse_rate_backend, read_file, read_stdin_all_for_generate,
    run_batch_mode, validate_obs_stream_len,
};
#[cfg(feature = "backend-sequitur")]
use crate::cli::{bytes_to_hex, parse_hex_bytes};
#[cfg(test)]
use crate::cli::{
    file_roundtrip_backend, parse_observation_key_mode, parse_observation_key_mode_for_env,
    parse_observation_key_mode_for_vm, parse_observation_key_mode_str,
    parse_observation_stream_len, parse_observation_stream_len_for_env,
    parse_observation_stream_len_for_vm, process_json_line, validate_observation_config,
};
#[cfg(feature = "backend-rosa")]
use infotheory::search;
#[cfg(feature = "tuner")]
use infotheory::tuner;

#[track_caller]
fn cli_unwrap<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    result.unwrap_or_else(|err| panic!("{context} failed: {err}"))
}

fn ncd_bytes_backend(
    x: &[u8],
    y: &[u8],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
) -> f64 {
    cli_unwrap(
        try_ncd_bytes_backend(x, y, backend, variant),
        "ncd_bytes_backend",
    )
}

fn intrinsic_dependence_bytes(data: &[u8]) -> f64 {
    cli_unwrap(
        try_intrinsic_dependence_bytes(data),
        "intrinsic_dependence_bytes",
    )
}

fn mutual_information_bytes(x: &[u8], y: &[u8]) -> f64 {
    cli_unwrap(
        try_mutual_information_bytes(x, y),
        "mutual_information_bytes",
    )
}

fn conditional_entropy_bytes(x: &[u8], y: &[u8]) -> f64 {
    cli_unwrap(
        try_conditional_entropy_bytes(x, y),
        "conditional_entropy_bytes",
    )
}

fn cross_entropy_bytes(test_data: &[u8], train_data: &[u8]) -> f64 {
    cli_unwrap(
        try_cross_entropy_bytes(test_data, train_data),
        "cross_entropy_bytes",
    )
}

fn joint_entropy_rate_bytes(x: &[u8], y: &[u8]) -> f64 {
    cli_unwrap(
        try_joint_entropy_rate_bytes(x, y),
        "joint_entropy_rate_bytes",
    )
}

fn resistance_to_transformation_bytes(x: &[u8], tx: &[u8]) -> f64 {
    cli_unwrap(
        try_resistance_to_transformation_bytes(x, tx),
        "resistance_to_transformation_bytes",
    )
}

fn ned_bytes(x: &[u8], y: &[u8]) -> f64 {
    cli_unwrap(try_ned_bytes(x, y), "ned_bytes")
}

fn ned_cons_bytes(x: &[u8], y: &[u8]) -> f64 {
    cli_unwrap(try_ned_cons_bytes(x, y), "ned_cons_bytes")
}

fn nte_bytes(x: &[u8], y: &[u8]) -> f64 {
    cli_unwrap(try_nte_bytes(x, y), "nte_bytes")
}

fn tvd_paths(x: &str, y: &str) -> f64 {
    cli_unwrap(try_tvd_paths(x, y), "tvd_paths")
}

fn nhd_paths(x: &str, y: &str) -> f64 {
    cli_unwrap(try_nhd_paths(x, y), "nhd_paths")
}

fn kl_divergence_paths(x: &str, y: &str) -> f64 {
    cli_unwrap(try_kl_divergence_paths(x, y), "kl_divergence_paths")
}

fn js_divergence_paths(x: &str, y: &str) -> f64 {
    cli_unwrap(try_js_divergence_paths(x, y), "js_divergence_paths")
}

struct AixiRunLogger {
    bits01: Option<BufWriter<File>>,
    jsonl: Option<BufWriter<File>>,
    flush_every: usize,
    step: usize,
}

impl AixiRunLogger {
    fn new(v: Option<&serde_json::Value>) -> anyhow::Result<Option<Self>> {
        let bits01_path = v.and_then(|value| value["trace_bits01_path"].as_str());
        let jsonl_path = v.and_then(|value| value["trace_jsonl_path"].as_str());
        if bits01_path.is_none() && jsonl_path.is_none() {
            return Ok(None);
        }

        let bits01 = if let Some(p) = bits01_path {
            let f = File::create(p)?;
            Some(BufWriter::new(f))
        } else {
            None
        };
        let jsonl = if let Some(p) = jsonl_path {
            let f = File::create(p)?;
            Some(BufWriter::new(f))
        } else {
            None
        };
        let flush_every = v
            .and_then(|value| value["trace_flush_every"].as_u64())
            .unwrap_or(1024) as usize;

        Ok(Some(Self {
            bits01,
            jsonl,
            flush_every,
            step: 0,
        }))
    }

    fn write_bits01(&mut self, bits: &[bool]) -> anyhow::Result<()> {
        if let Some(w) = self.bits01.as_mut() {
            for &b in bits {
                w.write_all(&[if b { 1u8 } else { 0u8 }])?;
            }
        }
        Ok(())
    }

    fn log_percept(
        &mut self,
        observations: &[u64],
        reward: i64,
        observation_bits: usize,
        reward_bits: usize,
        reward_offset: i64,
    ) -> anyhow::Result<()> {
        // Exact same bit encoding the agent uses internally.
        let mut bits = Vec::new();
        for &obs in observations {
            infotheory::aixi::common::encode(&mut bits, obs, observation_bits);
        }
        infotheory::aixi::common::encode_reward_offset(
            &mut bits,
            reward,
            reward_bits,
            reward_offset,
        );

        self.write_bits01(&bits)?;

        if let Some(w) = self.jsonl.as_mut() {
            let rec = serde_json::json!({
                "t": self.step,
                "kind": "percept",
                "observations": observations,
                "reward": reward,
            });
            writeln!(w, "{rec}")?;
        }
        Ok(())
    }

    fn log_action(&mut self, action: u64, action_bits: usize) -> anyhow::Result<()> {
        let mut bits = Vec::new();
        infotheory::aixi::common::encode(&mut bits, action, action_bits);
        self.write_bits01(&bits)?;

        if let Some(w) = self.jsonl.as_mut() {
            let rec = serde_json::json!({
                "t": self.step,
                "kind": "action",
                "action": action,
            });
            writeln!(w, "{rec}")?;
        }
        Ok(())
    }

    fn next_step(&mut self) -> anyhow::Result<()> {
        self.step = self.step.saturating_add(1);
        if self.flush_every > 0 && self.step.is_multiple_of(self.flush_every) {
            if let Some(w) = self.bits01.as_mut() {
                w.flush()?;
            }
            if let Some(w) = self.jsonl.as_mut() {
                w.flush()?;
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "all-backends"))]
fn parse_mixture_kind(kind: &str) -> anyhow::Result<MixtureKind> {
    infotheory::api::parse_mixture_kind_name(kind).map_err(anyhow::Error::msg)
}

#[cfg(all(test, feature = "all-backends"))]
fn parse_mixture_schedule(schedule: &str) -> anyhow::Result<MixtureScheduleMode> {
    infotheory::api::parse_mixture_schedule_name(schedule).map_err(anyhow::Error::msg)
}

#[cfg(all(test, feature = "all-backends"))]
fn parse_mixture_spec_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> anyhow::Result<MixtureSpec> {
    infotheory::spec::parse_mixture_spec_value(v, base_dir, depth).map_err(anyhow::Error::msg)
}

#[cfg(all(
    test,
    any(
        feature = "all-backends",
        feature = "backend-mamba",
        feature = "backend-rwkv"
    )
))]
fn parse_mixture_expert_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> anyhow::Result<MixtureExpertSpec> {
    infotheory::spec::parse_mixture_expert_value(v, base_dir, depth).map_err(anyhow::Error::msg)
}

fn is_canonical_spec_document(value: &serde_json::Value) -> bool {
    value["schema_version"].as_u64().is_some() && value["kind"].as_str().is_some()
}

fn builtin_environment_name(spec: BuiltinEnvironmentSpec) -> &'static str {
    spec.canonical_name()
}

fn legacy_planner_config_error(path: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "legacy aixi JSON configs are no longer executable; convert '{}' to a canonical planner_run document with top-level 'schema_version' and 'kind'",
        path
    )
}

#[derive(Clone, Copy)]
enum PlannerPhase {
    Learn,
    Eval,
}

struct PlannerRunSchedule {
    learn_cycles: usize,
    eval_cycles: usize,
    log_every: usize,
    perf: bool,
    vm_perf_only: bool,
    explore_epsilon: f64,
    explore_gamma: f64,
}

impl PlannerRunSchedule {
    fn from_runtime(runtime: &PlannerRuntimeSpec) -> Self {
        let terminate_lifetime = runtime.terminate_lifetime;
        let (learn_cycles, eval_cycles) = match (runtime.learn_cycles, runtime.eval_cycles) {
            (Some(learn), Some(eval)) => (learn, eval),
            (Some(learn), None) => (learn, 0usize),
            (None, Some(eval)) => (terminate_lifetime, eval),
            (None, None) => (terminate_lifetime, 0usize),
        };
        Self {
            learn_cycles,
            eval_cycles,
            log_every: runtime.log_every,
            perf: runtime.perf,
            vm_perf_only: runtime.vm_perf_only,
            explore_epsilon: runtime.explore_epsilon,
            explore_gamma: runtime.explore_gamma,
        }
    }

    fn extra_exploration(&self, step: usize) -> f64 {
        if self.explore_epsilon > 0.0 {
            (self.explore_epsilon * self.explore_gamma.powi(step as i32)).min(1.0)
        } else {
            0.0
        }
    }
}

struct PlannerExecutionContext {
    env: Box<dyn Environment>,
    observation_bits: usize,
    observation_stream_len: usize,
    reward_bits: usize,
    reward_offset: i64,
    agent_actions: ActionAlphabet,
    obs_stream: Vec<u64>,
    rew: i64,
    trace_logger: Option<AixiRunLogger>,
}

impl PlannerExecutionContext {
    fn new(
        compiled: &CompiledPlannerRunSpec,
        mut env: Box<dyn Environment>,
        cli_overlay: Option<&serde_json::Value>,
    ) -> anyhow::Result<Self> {
        let obs_stream = env.drain_observations();
        validate_obs_stream_len(
            compiled.interface().observation_stream_len,
            obs_stream.len(),
        )?;
        let rew = env.get_reward();
        Ok(Self {
            observation_bits: compiled.interface().observation_bits,
            observation_stream_len: compiled.interface().observation_stream_len,
            reward_bits: compiled.interface().reward_bits,
            reward_offset: 0,
            agent_actions: compiled.interface().agent_actions,
            trace_logger: AixiRunLogger::new(cli_overlay)?,
            env,
            obs_stream,
            rew,
        })
    }

    fn perform_action(&mut self, action: u64) -> anyhow::Result<i64> {
        self.env.perform_action(action);
        self.obs_stream = self.env.drain_observations();
        validate_obs_stream_len(self.observation_stream_len, self.obs_stream.len())?;
        self.rew = self.env.get_reward();
        Ok(self.rew)
    }
}

enum PlannerControllerRuntime {
    McAixi {
        agent: Agent,
        prev_action: u64,
        explore_rng: RandomGenerator,
    },
    AiqiDiscounted {
        agent: AiqiAgent,
    },
    WarmStartExactJh {
        agent: WarmStartExactJhAgent,
    },
}

impl PlannerControllerRuntime {
    fn from_compiled(compiled: &CompiledPlannerRunSpec) -> anyhow::Result<Self> {
        match compiled.controller() {
            CompiledPlannerController::McAixi { .. } => {
                let agent =
                    Agent::from_compiled_planner_run(compiled).map_err(anyhow::Error::msg)?;
                let explore_rng =
                    RandomGenerator::from_seed(resolve_random_seed(compiled.runtime().random_seed))
                        .fork_with(EXPLORE_RANDOM_SALT);
                Ok(Self::McAixi {
                    agent,
                    prev_action: 0,
                    explore_rng,
                })
            }
            CompiledPlannerController::AiqiDiscounted { .. } => Ok(Self::AiqiDiscounted {
                agent: AiqiAgent::from_compiled_planner_run(compiled)
                    .map_err(anyhow::Error::msg)?,
            }),
            CompiledPlannerController::AiqiWarmstartExactJh {
                teacher_dataset_asset,
                ..
            } => {
                let teacher =
                    load_warmstart_exact_jh_teacher_dataset(compiled, teacher_dataset_asset)?;
                Ok(Self::WarmStartExactJh {
                    agent: WarmStartExactJhAgent::from_compiled_planner_run(compiled, teacher)
                        .map_err(anyhow::Error::msg)?,
                })
            }
            other => Err(anyhow::anyhow!(
                "planner_run controller kind '{}' is not executable from the CLI",
                other.kind_str()
            )),
        }
    }

    fn run_cycle(
        &mut self,
        phase: PlannerPhase,
        step: usize,
        schedule: &PlannerRunSchedule,
        ctx: &mut PlannerExecutionContext,
    ) -> anyhow::Result<i64> {
        match self {
            Self::McAixi {
                agent,
                prev_action,
                explore_rng,
            } => {
                let obs_repr = agent.observation_repr_from_stream(&ctx.obs_stream);
                if schedule.log_every > 0 && step % schedule.log_every == 0 {
                    println!("Cycle {}: Obs={:?}, Rew={}", step, obs_repr, ctx.rew);
                }
                if let Some(logger) = ctx.trace_logger.as_mut() {
                    logger.log_percept(
                        &ctx.obs_stream,
                        ctx.rew,
                        ctx.observation_bits,
                        ctx.reward_bits,
                        ctx.reward_offset,
                    )?;
                }
                agent.model_update_percept_stream(&ctx.obs_stream, ctx.rew);

                let action = match phase {
                    PlannerPhase::Learn => {
                        let explore_p = schedule.extra_exploration(step);
                        if explore_p > 0.0 && explore_rng.gen_bool(explore_p) {
                            explore_rng.gen_range(ctx.agent_actions.get()) as u64
                        } else {
                            agent.get_planned_action(&ctx.obs_stream, ctx.rew, *prev_action)
                        }
                    }
                    PlannerPhase::Eval => {
                        agent.get_planned_action(&ctx.obs_stream, ctx.rew, *prev_action)
                    }
                };
                if schedule.log_every > 0 && step % schedule.log_every == 0 {
                    println!("Cycle {}: Planned Action={}", step, action);
                }
                if let Some(logger) = ctx.trace_logger.as_mut() {
                    logger.log_action(action, ctx.env.get_action_bits())?;
                }
                agent.model_update_action_external(action);
                let reward = ctx.perform_action(action)?;
                *prev_action = action;
                if let Some(logger) = ctx.trace_logger.as_mut() {
                    logger.next_step()?;
                }
                Ok(reward)
            }
            Self::AiqiDiscounted { agent } => {
                let action = match phase {
                    PlannerPhase::Learn => agent.get_planned_action_with_extra_exploration(
                        schedule.extra_exploration(step),
                    ),
                    PlannerPhase::Eval => agent.get_planned_action(),
                };
                if schedule.log_every > 0 && step % schedule.log_every == 0 {
                    println!(
                        "Cycle {}: Action={} Obs={:?} Rew={}",
                        step, action, ctx.obs_stream, ctx.rew
                    );
                }
                if let Some(logger) = ctx.trace_logger.as_mut() {
                    logger.log_action(action, ctx.env.get_action_bits())?;
                }
                let reward = ctx.perform_action(action)?;
                if let Some(logger) = ctx.trace_logger.as_mut() {
                    logger.log_percept(
                        &ctx.obs_stream,
                        ctx.rew,
                        ctx.observation_bits,
                        ctx.reward_bits,
                        ctx.reward_offset,
                    )?;
                    logger.next_step()?;
                }
                agent
                    .observe_transition(action, &ctx.obs_stream, ctx.rew)
                    .map_err(anyhow::Error::msg)?;
                Ok(reward)
            }
            Self::WarmStartExactJh { agent } => {
                let action = match phase {
                    PlannerPhase::Learn => agent.get_planned_action_with_extra_exploration(
                        schedule.extra_exploration(step),
                    ),
                    PlannerPhase::Eval => agent.get_planned_action(),
                };
                if schedule.log_every > 0 && step % schedule.log_every == 0 {
                    println!(
                        "Cycle {}: Action={} Obs={:?} Rew={}",
                        step, action, ctx.obs_stream, ctx.rew
                    );
                }
                if let Some(logger) = ctx.trace_logger.as_mut() {
                    logger.log_action(action, ctx.env.get_action_bits())?;
                }
                let reward = ctx.perform_action(action)?;
                if let Some(logger) = ctx.trace_logger.as_mut() {
                    logger.log_percept(
                        &ctx.obs_stream,
                        ctx.rew,
                        ctx.observation_bits,
                        ctx.reward_bits,
                        ctx.reward_offset,
                    )?;
                    logger.next_step()?;
                }
                agent
                    .observe_transition(action, &ctx.obs_stream, ctx.rew)
                    .map_err(anyhow::Error::msg)?;
                Ok(reward)
            }
        }
    }
}

/// Load a warm-start exact-J_H teacher dataset and validate its planner contract.
///
/// This enforces a stable planner-task boundary (fingerprint and schema) before
/// the data is used by runtime construction.
fn load_warmstart_exact_jh_teacher_dataset(
    compiled: &CompiledPlannerRunSpec,
    asset_id: &str,
) -> anyhow::Result<WarmStartExactJhTeacherDataset> {
    let binding = compiled
        .resolved_assets()
        .iter()
        .find(|entry| entry.id == asset_id)
        .ok_or_else(|| anyhow::anyhow!("unknown warm-start teacher_dataset_asset '{asset_id}'"))?;
    let path = match &binding.asset {
        AssetRef::Filesystem(path) => path,
        _ => {
            return Err(anyhow::anyhow!(
                "unsupported warm-start teacher_dataset_asset reference kind"
            ));
        }
    };
    let bytes = std::fs::read(path).map_err(|err| {
        anyhow::anyhow!(
            "failed to read warm-start teacher_dataset_asset '{}': {err}",
            path.display()
        )
    })?;
    let teacher =
        WarmStartExactJhTeacherDataset::from_json_slice(&bytes).map_err(anyhow::Error::msg)?;
    validate_warmstart_exact_jh_teacher_contract(compiled, &teacher)?;
    Ok(teacher)
}

/// Validate the parser-level contract fields against a concrete compiled planner run.
///
/// The contract comparison intentionally checks task identity, planner interface
/// dimensions, and planner execution invariants that affect trace encoding.
fn validate_warmstart_exact_jh_teacher_contract(
    compiled: &CompiledPlannerRunSpec,
    teacher: &WarmStartExactJhTeacherDataset,
) -> anyhow::Result<()> {
    infotheory::aixi::warmstart::validate_warmstart_teacher_against_compiled_planner_run(
        compiled,
        &teacher.contract,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))
}

fn controller_backend_label(controller: &CompiledPlannerController) -> String {
    controller.backend_label()
}

fn build_builtin_environment(spec: BuiltinEnvironmentSpec) -> anyhow::Result<Box<dyn Environment>> {
    #[cfg(feature = "aixi-gameengine")]
    {
        return build_gameengine_builtin_environment(spec).map_err(anyhow::Error::new);
    }
    #[cfg(not(feature = "aixi-gameengine"))]
    {
        Err(anyhow::anyhow!(
            "builtin environment '{}' requires feature 'aixi-gameengine'",
            builtin_environment_name(spec)
        ))
    }
}

#[cfg(feature = "vm")]
fn build_planner_environment(
    compiled: &CompiledPlannerRunSpec,
) -> anyhow::Result<(Box<dyn Environment>, &'static str)> {
    let environment = &compiled.canonical_spec().environment;
    match environment {
        spec::EnvironmentSpec::Builtin { builtin } => Ok((
            build_builtin_environment(*builtin)?,
            builtin_environment_name(*builtin),
        )),
        spec::EnvironmentSpec::NyxVm(vm) => {
            let config = NyxVmConfig::from_environment_spec(vm, compiled.resolved_assets())
                .map_err(anyhow::Error::msg)?;
            Ok((Box::new(NyxVmEnvironment::new(config)?), "vm"))
        }
        other => Err(anyhow::anyhow!(
            "unsupported environment variant '{}' in this build",
            other.kind_str()
        )),
    }
}

#[cfg(not(feature = "vm"))]
fn build_planner_environment(
    compiled: &CompiledPlannerRunSpec,
) -> anyhow::Result<(Box<dyn Environment>, &'static str)> {
    let environment = &compiled.canonical_spec().environment;
    match environment {
        spec::EnvironmentSpec::Builtin { builtin } => Ok((
            build_builtin_environment(*builtin)?,
            builtin_environment_name(*builtin),
        )),
        other => Err(anyhow::anyhow!(
            "unsupported environment variant '{}' in this build",
            other.kind_str()
        )),
    }
}

fn validate_action_alphabet(
    compiled: &CompiledPlannerRunSpec,
    env: &dyn Environment,
) -> anyhow::Result<()> {
    let actual = env.get_num_actions();
    let expected = compiled.interface().agent_actions;
    if actual != expected {
        return Err(anyhow::anyhow!(
            "action_alphabet_mismatch: planner interface declares {} actions but environment exposes {}",
            expected,
            actual
        ));
    }
    Ok(())
}

fn run_vm_perf_only(
    schedule: &PlannerRunSchedule,
    ctx: &mut PlannerExecutionContext,
) -> anyhow::Result<()> {
    let mut obs = ctx.obs_stream.first().copied().unwrap_or(0);
    let mut rew = ctx.rew;
    let start = Instant::now();
    for step in 0..schedule.learn_cycles {
        if schedule.log_every > 0 && step % schedule.log_every == 0 {
            println!("Cycle {}: Obs={}, Rew={}", step, obs, rew);
        }
        ctx.perform_action(0)?;
        obs = ctx.obs_stream.first().copied().unwrap_or(0);
        rew = ctx.rew;
    }
    if schedule.perf && schedule.learn_cycles > 0 {
        let elapsed = start.elapsed().as_secs_f64().max(1e-9);
        let cps = schedule.learn_cycles as f64 / elapsed;
        println!("Perf cycles/s: {:.2}", cps);
    }
    Ok(())
}

fn run_compiled_planner_run(
    compiled: &CompiledPlannerRunSpec,
    cli_overlay: Option<&serde_json::Value>,
) -> anyhow::Result<()> {
    let schedule = PlannerRunSchedule::from_runtime(compiled.runtime());
    let mut controller = PlannerControllerRuntime::from_compiled(compiled)?;
    let (mut env, env_name) = build_planner_environment(compiled)?;
    env.set_random_seed(resolve_random_seed(compiled.runtime().random_seed));
    validate_action_alphabet(compiled, env.as_ref())?;
    let mut ctx = PlannerExecutionContext::new(compiled, env, cli_overlay)?;

    match compiled.controller() {
        CompiledPlannerController::McAixi { .. } => println!(
            "Agent initialized with {} algorithm for {} environment.",
            controller_backend_label(compiled.controller()),
            env_name
        ),
        CompiledPlannerController::AiqiDiscounted { .. } => println!(
            "AIQI initialized ({}) for {} environment.",
            controller_backend_label(compiled.controller()),
            env_name
        ),
        // Other controller kinds are filtered out by
        // `PlannerControllerRuntime::from_compiled` returning Err before reaching
        // this point. We use `unreachable!` (with a helpful message via
        // `kind_str`) so any future bug that lets such a variant slip through
        // panics with a clear diagnostic rather than silently misbehaving.
        other => unreachable!(
            "PlannerControllerRuntime::from_compiled should reject controller kind '{}'",
            other.kind_str()
        ),
    }

    if schedule.vm_perf_only {
        return run_vm_perf_only(&schedule, &mut ctx);
    }

    let learn_start = Instant::now();
    let mut learn_total_reward = 0i64;
    for step in 0..schedule.learn_cycles {
        learn_total_reward +=
            controller.run_cycle(PlannerPhase::Learn, step, &schedule, &mut ctx)?;
    }
    if schedule.perf && schedule.learn_cycles > 0 {
        let elapsed = learn_start.elapsed().as_secs_f64().max(1e-9);
        let cps = schedule.learn_cycles as f64 / elapsed;
        println!("Learn cycles/s: {:.2}", cps);
    }

    if schedule.eval_cycles > 0 {
        let eval_start = Instant::now();
        let mut eval_total_reward = 0i64;
        for offset in 0..schedule.eval_cycles {
            let step = schedule.learn_cycles + offset;
            eval_total_reward +=
                controller.run_cycle(PlannerPhase::Eval, step, &schedule, &mut ctx)?;
        }
        if schedule.perf {
            let elapsed = eval_start.elapsed().as_secs_f64().max(1e-9);
            let cps = schedule.eval_cycles as f64 / elapsed;
            println!("Eval cycles/s: {:.2}", cps);
        }
        let avg = (eval_total_reward as f64) / (schedule.eval_cycles as f64);
        println!("Eval Total Reward: {}", eval_total_reward);
        println!("Eval Average Reward per Cycle: {:.6}", avg);
    }

    println!("Total Reward: {}", learn_total_reward);
    Ok(())
}

fn run_aixi_mode(config_path: &str) -> anyhow::Result<()> {
    let raw = std::fs::read(config_path)?;
    let json_overlay = serde_json::from_slice::<serde_json::Value>(&raw).ok();
    if let Some(value) = json_overlay.as_ref() {
        if !is_canonical_spec_document(value) {
            return Err(legacy_planner_config_error(config_path));
        }
    }

    let config_dir = Path::new(config_path).parent().unwrap_or(Path::new("."));
    let document = infotheory::spec::load_spec_document(config_path).map_err(anyhow::Error::msg)?;
    match document {
        SpecDocument::PlannerRun(spec) => {
            let compiled = spec
                .compile_in(&spec::SpecEnvironment::new(config_dir))
                .map_err(anyhow::Error::msg)?;
            run_compiled_planner_run(&compiled, json_overlay.as_ref())
        }
        other => Err(anyhow::anyhow!(
            "aixi expects a planner_run document, found kind '{}'",
            other.kind_str()
        )),
    }
}

#[cfg(feature = "tuner")]
fn run_tune_mode(args: &[String]) {
    match tuner::parse_tune_command_args(args).and_then(|request| tuner::run_tune(&request)) {
        Ok(()) => {}
        Err(err) => {
            eprintln!("Error: tune failed: {err}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "tuner"))]
fn run_tune_mode(_args: &[String]) {
    eprintln!("Error: 'tune' requires infotheory built with feature 'tuner'");
    std::process::exit(1);
}

#[cfg(feature = "tuner")]
fn run_tuner_eval_worker_mode() {
    if let Err(err) = tuner::run_tuner_eval_worker_from_env() {
        eprintln!("Error: tuner evaluator worker failed: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "tuner"))]
fn run_tuner_eval_worker_mode() {
    eprintln!("Error: tuner evaluator worker requires infotheory built with feature 'tuner'");
    std::process::exit(1);
}

#[cfg(feature = "backend-rosa")]
fn search_command(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Error: 'search' requires query and target path.");
        std::process::exit(1);
    }
    let query = &args[2];
    let target = &args[3];

    // Preserve the legacy behavior (and avoid extra parsing work) when no flags are given.
    if args.len() == 4 {
        if let Err(err) = search::run_search(query, target) {
            eprintln!("Error: search failed: {err}");
            std::process::exit(1);
        }
        return;
    }

    let mut opts = match search::SearchOptions::try_default() {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("Error: search defaults unavailable in this build: {err}");
            std::process::exit(1);
        }
    };
    let mut rate_backend = infotheory::search::DEFAULT_SEARCH_RATE_BACKEND_NAME.to_string();
    let compression_backend =
        infotheory::search::DEFAULT_SEARCH_COMPRESSION_BACKEND_NAME.to_string();
    let mut method: Option<String> = None;
    let mut expert_spec_path: Option<String> = None;
    let mut rate_backend_json_path: Option<String> = None;
    let mut compression_backend_json_path: Option<String> = None;
    let mut explicit_rate_backend_flag: bool = false;
    let explicit_compression_backend_flag: bool = false;
    let mut explicit_method_flag: bool = false;
    let mut stage2_prior_mode: Option<search::Stage2PriorMode> = None;

    let mut i = 4usize;
    while i < args.len() {
        match args[i].as_str() {
            "--level" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --level requires snippet|file");
                opts.granularity = if v == "snippet" {
                    search::SearchGranularity::Snippet
                } else {
                    search::SearchGranularity::File
                };
            }
            "--prior" => {
                i += 1;
                opts.universal_prior = args.get(i).cloned();
            }
            "--top-k" => {
                i += 1;
                opts.top_k = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--rate-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend requires a value");
                rate_backend = parse_rate_backend_flag_or_exit(v, "--rate-backend");
                explicit_rate_backend_flag = true;
            }
            "--rate-backend-json" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend-json requires a path");
                rate_backend_json_path = Some(v.clone());
            }
            "--compression-backend-json" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --compression-backend-json requires a path");
                compression_backend_json_path = Some(v.clone());
            }
            "--method" => {
                i += 1;
                method = args.get(i).cloned();
                explicit_method_flag = true;
            }
            "--expert-spec" => {
                i += 1;
                expert_spec_path = args.get(i).cloned();
                explicit_rate_backend_flag = true;
            }
            "--stage2-prior-mode" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    stage2_prior_mode = match v.as_str() {
                        "none" | "no-prior" => Some(search::Stage2PriorMode::Disable),
                        "summarize" | "summarize-prior" => Some(search::Stage2PriorMode::Summarize),
                        "use" | "use-prior" => Some(search::Stage2PriorMode::Use),
                        _ => Some(search::Stage2PriorMode::Use),
                    };
                }
            }
            _ => {
                i += 1;
            }
        }
        i += 1;
    }
    if let Some(mode) = stage2_prior_mode {
        opts.stage2_prior_mode = mode;
    }
    opts.ctx = build_ctx_invocation(CliBackendInvocation {
        rate_backend: &rate_backend,
        compression_backend: &compression_backend,
        method: method.as_deref(),
        expert_spec_path: expert_spec_path.as_deref(),
        rate_backend_json_path: rate_backend_json_path.as_deref(),
        compression_backend_json_path: compression_backend_json_path.as_deref(),
        flags: CliBackendSourceFlags {
            explicit_rate_backend: explicit_rate_backend_flag,
            explicit_compression_backend: explicit_compression_backend_flag,
            explicit_method: explicit_method_flag,
        },
    })
    .ctx;
    if let Err(err) = search::run_search_with_options(query, target, &opts) {
        eprintln!("Error: search failed: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "backend-rosa"))]
fn search_command(_args: &[String]) {
    eprintln!("Error: 'search' requires infotheory built with feature 'backend-rosa'");
    std::process::exit(1);
}

trait OptionExt<T> {
    fn unwrap_or_exit(self, msg: &str) -> T;
}
impl<T> OptionExt<T> for Option<T> {
    fn unwrap_or_exit(self, msg: &str) -> T {
        self.unwrap_or_else(|| {
            eprintln!("{}", msg);
            std::process::exit(1);
        })
    }
}

fn parse_rate_backend_flag_or_exit(value: &str, flag_name: &str) -> String {
    parse_rate_backend(value)
        .map(std::string::ToString::to_string)
        .unwrap_or_else(|| {
            eprintln!(
                "Error: {flag_name} expects a canonical backend name, got '{value}'. If '{value}' is a canonical RateBackend JSON document path, use --rate-backend-json. If '{value}' is a mixture spec path, use --rate-backend mixture --method <path>."
            );
            std::process::exit(1);
        })
}

fn parse_compression_backend_flag_or_exit(value: &str, flag_name: &str) -> String {
    parse_compression_backend(value)
        .map(std::string::ToString::to_string)
        .unwrap_or_else(|| {
            eprintln!(
                "Error: {flag_name} expects a canonical backend name, got '{value}'. Use --compression-backend-json for a canonical JSON spec path."
            );
            std::process::exit(1);
        })
}

#[cfg(feature = "backend-ctw")]
fn parse_ctw_profile_size(raw: &str, field: &str) -> anyhow::Result<usize> {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    let (digits, multiplier): (&str, usize) = if let Some(prefix) = lower.strip_suffix("kib") {
        (prefix, 1024)
    } else if let Some(prefix) = lower.strip_suffix("mib") {
        (prefix, 1024 * 1024)
    } else if let Some(prefix) = lower.strip_suffix("gib") {
        (prefix, 1024 * 1024 * 1024)
    } else if let Some(prefix) = lower.strip_suffix('k') {
        (prefix, 1_000)
    } else if let Some(prefix) = lower.strip_suffix('m') {
        (prefix, 1_000_000)
    } else if let Some(prefix) = lower.strip_suffix('g') {
        (prefix, 1_000_000_000)
    } else {
        (trimmed, 1)
    };
    let value = digits
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{field} must be a non-negative integer size, got '{raw}'"))?;
    value
        .checked_mul(multiplier)
        .ok_or_else(|| anyhow::anyhow!("{field} overflows usize: '{raw}'"))
}

#[cfg(feature = "backend-ctw")]
fn parse_ctw_profile_cutpoints(raw: &str) -> anyhow::Result<Vec<usize>> {
    let mut cutpoints = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_ctw_profile_size(part, "--cutpoints"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    cutpoints.sort_unstable();
    cutpoints.dedup();
    if cutpoints.is_empty() {
        anyhow::bail!("--cutpoints must contain at least one byte count");
    }
    Ok(cutpoints)
}

#[cfg(feature = "backend-ctw")]
fn default_ctw_profile_cutpoints(max_bytes: Option<usize>) -> Vec<usize> {
    let mut cutpoints = Vec::new();
    let mut next = 1_000_000usize;
    while next < 1_000_000_000usize {
        cutpoints.push(next);
        next = next.saturating_mul(2);
    }
    cutpoints.push(1_000_000_000usize);
    if let Some(max_bytes) = max_bytes {
        cutpoints.retain(|cutpoint| *cutpoint <= max_bytes);
        if cutpoints.last().copied() != Some(max_bytes) {
            cutpoints.push(max_bytes);
        }
    }
    cutpoints
}

#[cfg(all(feature = "backend-ctw", target_os = "linux"))]
fn ctw_profile_proc_memory_bytes() -> serde_json::Value {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return serde_json::json!(null);
    };
    let mut vm_rss_bytes = None;
    let mut vm_hwm_bytes = None;
    for line in status.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        let Ok(kib) = value.parse::<u64>() else {
            continue;
        };
        match key {
            "VmRSS:" => vm_rss_bytes = kib.checked_mul(1024),
            "VmHWM:" => vm_hwm_bytes = kib.checked_mul(1024),
            _ => {}
        }
    }
    serde_json::json!({
        "vm_rss_bytes": vm_rss_bytes,
        "vm_hwm_bytes": vm_hwm_bytes,
    })
}

#[cfg(all(feature = "backend-ctw", not(target_os = "linux")))]
fn ctw_profile_proc_memory_bytes() -> serde_json::Value {
    serde_json::json!(null)
}

#[cfg(feature = "backend-ctw")]
fn ctw_profile_tree_json(tree: &infotheory::ctw::FacContextTreeTreeTelemetry) -> serde_json::Value {
    serde_json::json!({
        "bit_index": tree.bit_index,
        "max_depth": tree.max_depth,
        "root_visits": tree.root_visits,
        "nodes_len": tree.nodes_len,
        "nodes_capacity": tree.nodes_capacity,
        "segments_len": tree.segments_len,
        "segments_capacity": tree.segments_capacity,
        "free_nodes_len": tree.free_nodes_len,
        "free_nodes_capacity": tree.free_nodes_capacity,
        "free_segments_len": tree.free_segments_len,
        "free_segments_capacity": tree.free_segments_capacity,
        "node_bytes": tree.node_bytes,
        "segment_bytes": tree.segment_bytes,
        "free_list_bytes": tree.free_list_bytes,
        "scratch_bytes": tree.scratch_bytes,
        "total_bytes": tree.total_bytes,
        "exact_segments": tree.exact_segments,
        "history_segments": tree.history_segments,
        "history_invert_segments": tree.history_invert_segments,
        "const_segments": tree.const_segments,
        "segment_bits": tree.segment_bits,
        "max_segment_len": tree.max_segment_len,
    })
}

#[cfg(feature = "backend-ctw")]
fn ctw_profile_snapshot_json(
    mode: &str,
    depth: usize,
    bytes_seen: usize,
    log_prob: Option<f64>,
    elapsed_seconds: f64,
    telemetry: &infotheory::ctw::FacContextTreeTelemetry,
) -> serde_json::Value {
    let bits = log_prob.map(|value| -value / std::f64::consts::LN_2);
    let bits_per_byte = bits.and_then(|value| {
        if bytes_seen == 0 {
            None
        } else {
            Some(value / bytes_seen as f64)
        }
    });
    serde_json::json!({
        "kind": "ctw_profile_snapshot",
        "mode": mode,
        "depth": depth,
        "bytes_seen": bytes_seen,
        "elapsed_seconds": elapsed_seconds,
        "log_probability": log_prob,
        "bits": bits,
        "bits_per_byte": bits_per_byte,
        "rss": ctw_profile_proc_memory_bytes(),
        "telemetry": {
            "base_depth": telemetry.base_depth,
            "num_bits": telemetry.num_bits,
            "shared_history_len_bits": telemetry.shared_history_len_bits,
            "shared_history_capacity_bits": telemetry.shared_history_capacity_bits,
            "shared_history_bytes": telemetry.shared_history_bytes,
            "shared_log_cache_bytes": telemetry.shared_log_cache_bytes,
            "tree_bytes": telemetry.tree_bytes,
            "total_bytes": telemetry.total_bytes,
            "nodes_len": telemetry.nodes_len,
            "nodes_capacity": telemetry.nodes_capacity,
            "segments_len": telemetry.segments_len,
            "segments_capacity": telemetry.segments_capacity,
            "free_nodes_len": telemetry.free_nodes_len,
            "free_segments_len": telemetry.free_segments_len,
            "exact_segments": telemetry.exact_segments,
            "history_segments": telemetry.history_segments,
            "history_invert_segments": telemetry.history_invert_segments,
            "const_segments": telemetry.const_segments,
            "segment_bits": telemetry.segment_bits,
            "trees": telemetry.trees.iter().map(ctw_profile_tree_json).collect::<Vec<_>>(),
        },
    })
}

#[cfg(feature = "backend-ctw")]
fn run_ctw_profile_mode(args: &[String]) {
    let result = (|| -> anyhow::Result<()> {
        let mut input_path: Option<String> = None;
        let mut depth: usize = 32;
        let mut max_bytes: Option<usize> = None;
        let mut reserve_symbols: Option<usize> = None;
        let mut cutpoints: Option<Vec<usize>> = None;
        let mut update_only = false;
        let mut i = 2usize;
        while i < args.len() {
            match args[i].as_str() {
                "--update-only" => {
                    update_only = true;
                }
                "--depth" => {
                    i += 1;
                    let raw = args
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("--depth requires a value"))?;
                    depth = raw
                        .parse::<usize>()
                        .map_err(|_| anyhow::anyhow!("--depth must be a non-negative integer"))?;
                }
                "--max-bytes" => {
                    i += 1;
                    let raw = args
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("--max-bytes requires a value"))?;
                    max_bytes = Some(parse_ctw_profile_size(raw, "--max-bytes")?);
                }
                "--reserve-symbols" => {
                    i += 1;
                    let raw = args
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("--reserve-symbols requires a value"))?;
                    reserve_symbols = Some(parse_ctw_profile_size(raw, "--reserve-symbols")?);
                }
                "--cutpoints" => {
                    i += 1;
                    let raw = args
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("--cutpoints requires a value"))?;
                    cutpoints = Some(parse_ctw_profile_cutpoints(raw)?);
                }
                flag if flag.starts_with("--") => {
                    anyhow::bail!("unknown ctw-profile option '{flag}'");
                }
                value => {
                    if input_path.is_some() {
                        anyhow::bail!("ctw-profile accepts exactly one input path or '-'");
                    }
                    input_path = Some(value.to_string());
                }
            }
            i += 1;
        }

        let input_path = input_path
            .ok_or_else(|| anyhow::anyhow!("ctw-profile requires an input path or '-'"))?;
        let mut cutpoints = cutpoints.unwrap_or_else(|| default_ctw_profile_cutpoints(max_bytes));
        cutpoints.sort_unstable();
        cutpoints.dedup();

        let mut tree = infotheory::ctw::FacContextTree::new(depth, 8);
        if let Some(symbols) = reserve_symbols {
            tree.reserve_for_symbols(symbols);
        }

        let stdin = io::stdin();
        let mut source: Box<dyn Read + '_> = if input_path == "-" {
            Box::new(stdin.lock())
        } else {
            Box::new(File::open(&input_path)?)
        };
        let mut reader = io::BufReader::with_capacity(1 << 20, &mut source);
        let mut out = BufWriter::new(io::stdout().lock());
        let started = Instant::now();
        let mut buf = [0u8; 1 << 20];
        let mut bytes_seen: usize = 0;
        let mut log_prob = 0.0f64;
        let mut cutpoint_index = 0usize;
        let mode = if update_only {
            "update_only"
        } else {
            "log_prob_update"
        };

        let initial = tree.telemetry();
        writeln!(
            out,
            "{}",
            ctw_profile_snapshot_json(
                mode,
                depth,
                bytes_seen,
                (!update_only).then_some(log_prob),
                0.0,
                &initial,
            )
        )?;

        'outer: loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &byte in &buf[..n] {
                if max_bytes.is_some_and(|limit| bytes_seen >= limit) {
                    break 'outer;
                }
                if update_only {
                    tree.update_byte_msb(byte);
                } else {
                    log_prob += tree.log_prob_update_byte_msb(byte);
                }
                bytes_seen = bytes_seen.saturating_add(1);
                while cutpoint_index < cutpoints.len() && bytes_seen >= cutpoints[cutpoint_index] {
                    let telemetry = tree.telemetry();
                    writeln!(
                        out,
                        "{}",
                        ctw_profile_snapshot_json(
                            mode,
                            depth,
                            bytes_seen,
                            (!update_only).then_some(log_prob),
                            started.elapsed().as_secs_f64(),
                            &telemetry,
                        )
                    )?;
                    out.flush()?;
                    cutpoint_index += 1;
                }
            }
        }

        if cutpoints.last().copied() != Some(bytes_seen) {
            let telemetry = tree.telemetry();
            writeln!(
                out,
                "{}",
                ctw_profile_snapshot_json(
                    mode,
                    depth,
                    bytes_seen,
                    (!update_only).then_some(log_prob),
                    started.elapsed().as_secs_f64(),
                    &telemetry,
                )
            )?;
        }
        out.flush()?;
        Ok(())
    })();

    if let Err(err) = result {
        eprintln!("Error: ctw-profile failed: {err:#}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "backend-ctw"))]
fn run_ctw_profile_mode(_args: &[String]) {
    eprintln!("Error: 'ctw-profile' requires infotheory built with feature 'backend-ctw'");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check for help flag early
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        print_usage();
        return;
    }

    if args.len() < 2 {
        print_usage();
        return;
    }

    let primitive = &args[1];
    if primitive == "__infotheory-tuner-eval-worker" {
        run_tuner_eval_worker_mode();
        return;
    }
    if primitive == "batch" {
        run_batch_mode();
        return;
    }
    if primitive == "tune" {
        run_tune_mode(&args);
        return;
    }
    if primitive == "ctw-profile" || primitive == "ctw_profile" {
        run_ctw_profile_mode(&args);
        return;
    }

    // Common positional and flag parsing.
    // Collect positionals only up to the first flag token, then parse flags separately.
    let mut file1: Option<String> = None;
    let mut file2: Option<String> = None;
    let mut pos_arg3: Option<String> = None;
    let mut flags_start = 2usize;

    if primitive != "search" && primitive != "aixi" {
        let mut positionals: Vec<String> = Vec::new();
        let mut i = 2usize;
        while i < args.len() {
            let tok = &args[i];
            if tok.starts_with('-') {
                break;
            }
            positionals.push(tok.clone());
            i += 1;
        }
        flags_start = i;
        file1 = positionals.first().cloned();
        file2 = positionals.get(1).cloned();
        pos_arg3 = positionals.get(2).cloned();
    }

    let mut rate_backend_str = "rosaplus".to_string();
    let mut compression_backend_str = "zpaq".to_string();
    let mut method_str: Option<String> = None;
    let mut expert_spec_path: Option<String> = None;
    let mut rate_backend_json_path: Option<String> = None;
    let mut compression_backend_json_path: Option<String> = None;
    let mut explicit_rate_backend_flag: bool = false;
    let mut explicit_compression_backend_flag: bool = false;
    let mut explicit_method_flag: bool = false;
    let mut model_export_path: Option<String> = None;
    let mut diagnostic_mixture_path: Option<String> = None;
    let mut diagnostic_out_prefix: Option<String> = None;
    let mut sequitur_debug_hexes: Vec<String> = Vec::new();
    #[cfg(feature = "backend-sequitur")]
    let mut sequitur_context_bytes: usize = 64;
    #[cfg(not(feature = "backend-sequitur"))]
    let _sequitur_context_bytes: usize = 64;
    #[cfg(feature = "backend-sequitur")]
    let mut sequitur_alphabet_prefix: usize = 4;
    #[cfg(not(feature = "backend-sequitur"))]
    let _sequitur_alphabet_prefix: usize = 4;
    let mut generate_len_bytes: usize = 8;
    let mut generate_config = GenerationConfig::default();
    let mut rate_backend_specified = false;

    let mut i = flags_start;
    while i < args.len() {
        match args[i].as_str() {
            "--rate-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend requires a value");
                rate_backend_str = parse_rate_backend_flag_or_exit(v, "--rate-backend");
                rate_backend_specified = true;
                explicit_rate_backend_flag = true;
            }
            "--rate-backend-json" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend-json requires a path");
                rate_backend_json_path = Some(v.clone());
                rate_backend_specified = true;
            }
            "--compression-backend-json" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --compression-backend-json requires a path");
                compression_backend_json_path = Some(v.clone());
            }
            "--compression-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --compression-backend requires a value");
                compression_backend_str =
                    parse_compression_backend_flag_or_exit(v, "--compression-backend");
                explicit_compression_backend_flag = true;
            }
            "--method" => {
                i += 1;
                method_str = args.get(i).cloned();
                explicit_method_flag = true;
            }
            "--expert-spec" => {
                i += 1;
                expert_spec_path = args.get(i).cloned();
                rate_backend_specified = true;
                explicit_rate_backend_flag = true;
            }
            "--model-export" => {
                i += 1;
                model_export_path = args.get(i).cloned();
            }
            "--rwkv-export" => {
                eprintln!("Error: --rwkv-export has been removed; use --model-export instead");
                std::process::exit(1);
            }
            "--mixture" => {
                i += 1;
                diagnostic_mixture_path = args.get(i).cloned();
            }
            "--out-prefix" => {
                i += 1;
                diagnostic_out_prefix = args.get(i).cloned();
            }
            "--hex" => {
                i += 1;
                if let Some(value) = args.get(i) {
                    sequitur_debug_hexes.push(value.clone());
                }
            }
            "--context-bytes" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --context-bytes requires a positive integer");
                let parsed = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: --context-bytes must be a positive integer, got '{raw}'");
                    std::process::exit(1);
                });
                #[cfg(feature = "backend-sequitur")]
                {
                    sequitur_context_bytes = parsed;
                }
                #[cfg(not(feature = "backend-sequitur"))]
                {
                    let _ = parsed;
                }
            }
            "--alphabet-prefix" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --alphabet-prefix requires a positive integer");
                let parsed = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: --alphabet-prefix must be a positive integer, got '{raw}'");
                    std::process::exit(1);
                });
                #[cfg(feature = "backend-sequitur")]
                {
                    sequitur_alphabet_prefix = parsed;
                }
                #[cfg(not(feature = "backend-sequitur"))]
                {
                    let _ = parsed;
                }
            }
            "--bytes" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --bytes requires a non-negative integer");
                generate_len_bytes = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: --bytes must be a non-negative integer, got '{raw}'");
                    std::process::exit(1);
                });
            }
            "--sample" => {
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--greedy" => {
                generate_config.strategy = GenerationStrategy::Greedy;
            }
            "--adaptive" => {
                generate_config.update_mode = GenerationUpdateMode::Adaptive;
            }
            "--seed" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --seed requires an unsigned integer");
                generate_config.seed = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("Error: --seed must be an unsigned integer, got '{raw}'");
                    std::process::exit(1);
                });
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--temperature" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --temperature requires a finite number");
                generate_config.temperature = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("Error: --temperature must be a finite number, got '{raw}'");
                    std::process::exit(1);
                });
                if !generate_config.temperature.is_finite() || generate_config.temperature < 0.0 {
                    eprintln!(
                        "Error: --temperature must be finite and non-negative, got '{}'",
                        generate_config.temperature
                    );
                    std::process::exit(1);
                }
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--top-k" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --top-k requires a non-negative integer");
                generate_config.top_k = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: --top-k must be a non-negative integer, got '{raw}'");
                    std::process::exit(1);
                });
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--top-p" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --top-p requires a number in (0, 1]");
                generate_config.top_p = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("Error: --top-p must be a number in (0, 1], got '{raw}'");
                    std::process::exit(1);
                });
                if !generate_config.top_p.is_finite()
                    || generate_config.top_p <= 0.0
                    || generate_config.top_p > 1.0
                {
                    eprintln!(
                        "Error: --top-p must be in (0, 1], got '{}'",
                        generate_config.top_p
                    );
                    std::process::exit(1);
                }
                generate_config.strategy = GenerationStrategy::Sample;
            }
            _ => {}
        }
        i += 1;
    }

    if primitive == "ac-log-loss" || primitive == "ac_log_loss" {
        let input_path = file1.unwrap_or_exit(
            "Error: 'ac-log-loss' requires <input> --mixture <spec.json> --out-prefix <prefix>",
        );
        let mixture_path = diagnostic_mixture_path
            .unwrap_or_exit("Error: 'ac-log-loss' requires --mixture <spec.json>");
        let out_prefix = diagnostic_out_prefix
            .unwrap_or_exit("Error: 'ac-log-loss' requires --out-prefix <prefix>");
        let spec = load_mixture_spec(&mixture_path).unwrap_or_else(|e| {
            eprintln!(
                "Error: failed to load mixture spec '{}': {}",
                mixture_path, e
            );
            std::process::exit(1);
        });
        let data = read_file(&input_path);
        match infotheory::diagnostics::run_ac_log_loss_mixture_bytes(&data, &spec, &out_prefix) {
            Ok(summary) => {
                println!(
                    "wrote {} rows to {}, nodes to {}, summary to {}",
                    summary.positions,
                    summary.trace_path.display(),
                    summary.nodes_path.display(),
                    summary.summary_path.display()
                );
            }
            Err(err) => {
                eprintln!("Error: AC log-loss diagnostic failed: {err:#}");
                std::process::exit(1);
            }
        }
        return;
    }

    if primitive == "sequitur-debug" || primitive == "sequitur_debug" {
        #[cfg(not(feature = "backend-sequitur"))]
        {
            eprintln!(
                "Error: 'sequitur-debug' requires infotheory built with feature 'backend-sequitur'"
            );
            std::process::exit(1);
        }
        #[cfg(feature = "backend-sequitur")]
        {
            let inputs = if !sequitur_debug_hexes.is_empty() {
                sequitur_debug_hexes
                    .iter()
                    .map(|raw_hex| {
                        parse_hex_bytes(raw_hex).unwrap_or_else(|e| {
                            eprintln!("Error: invalid --hex input for 'sequitur-debug': {e}");
                            std::process::exit(1);
                        })
                    })
                    .collect::<Vec<_>>()
            } else {
                let input_path =
                    file1.unwrap_or_exit("Error: 'sequitur-debug' requires <input> or --hex <hex>");
                vec![read_file(&input_path)]
            };
            let alphabet_prefix = sequitur_alphabet_prefix.clamp(1, 256);
            let cases = inputs
                .iter()
                .map(|data| {
                    let mut model = SequiturModel::new(sequitur_context_bytes);
                    let trace = model.predictive_trace(data, alphabet_prefix);
                    let rules = model
                        .canonical_grammar()
                        .rules
                        .iter()
                        .map(|rule| {
                            let rhs = rule
                                .rhs
                                .iter()
                                .map(|sym| match sym {
                                    CanonicalSymbol::Terminal(byte) => {
                                        serde_json::json!(*byte as i64)
                                    }
                                    CanonicalSymbol::NonTerminal(rule_id) => {
                                        serde_json::json!(-((*rule_id as i64) + 1))
                                    }
                                })
                                .collect::<Vec<_>>();
                            serde_json::json!({
                                "id": rule.id,
                                "rhs": rhs,
                            })
                        })
                        .collect::<Vec<_>>();
                    serde_json::json!({
                        "input_hex": bytes_to_hex(data),
                        "decoded_hex": bytes_to_hex(&model.decode()),
                        "rules": rules,
                        "trace": trace,
                    })
                })
                .collect::<Vec<_>>();
            let output = serde_json::json!({
                "context_bytes": sequitur_context_bytes,
                "alphabet_prefix": alphabet_prefix,
                "cases": cases,
            });
            println!(
                "{}",
                serde_json::to_string(&output).expect("sequitur debug json serialization")
            );
            return;
        }
    }

    let built_ctx = build_ctx_invocation(CliBackendInvocation {
        rate_backend: &rate_backend_str,
        compression_backend: &compression_backend_str,
        method: method_str.as_deref(),
        expert_spec_path: expert_spec_path.as_deref(),
        rate_backend_json_path: rate_backend_json_path.as_deref(),
        compression_backend_json_path: compression_backend_json_path.as_deref(),
        flags: CliBackendSourceFlags {
            explicit_rate_backend: explicit_rate_backend_flag,
            explicit_compression_backend: explicit_compression_backend_flag,
            explicit_method: explicit_method_flag,
        },
    });
    let ctx = built_ctx.ctx;
    set_default_ctx(ctx.clone());

    match primitive.as_str() {
        "aixi" => {
            if let Some(p) = args.get(2) {
                if let Err(e) = run_aixi_mode(p) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            } else {
                eprintln!("Error: 'aixi' requires config.json");
                std::process::exit(1);
            }
        }
        "search" => search_command(&args),
        "compress" => {
            let in_path = file1.unwrap_or_exit("Error: 'compress' requires <input> <output>");
            let out_path = file2.unwrap_or_exit("Error: 'compress' requires <input> <output>");
            let data = read_file(&in_path);
            let backend = file_roundtrip_compiled_backend(&ctx.compression_backend);
            let compressed = match infotheory::api::try_compress_bytes_backend(&data, &backend) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error: compression failed: {e}");
                    std::process::exit(1);
                }
            };
            if let Err(e) = std::fs::write(&out_path, &compressed) {
                eprintln!("Error: failed to write output '{}': {}", out_path, e);
                std::process::exit(1);
            }
            println!(
                "compressed {} bytes -> {} bytes",
                data.len(),
                compressed.len()
            );
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&data])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "decompress" => {
            let in_path = file1.unwrap_or_exit("Error: 'decompress' requires <input> <output>");
            let out_path = file2.unwrap_or_exit("Error: 'decompress' requires <input> <output>");
            let input = read_file(&in_path);
            let backend = file_roundtrip_compiled_backend(&ctx.compression_backend);
            let decoded = match infotheory::api::try_decompress_bytes_backend(&input, &backend) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error: decompression failed: {e}");
                    std::process::exit(1);
                }
            };
            if let Err(e) = std::fs::write(&out_path, &decoded) {
                eprintln!("Error: failed to write output '{}': {}", out_path, e);
                std::process::exit(1);
            }
            println!(
                "decompressed {} bytes -> {} bytes",
                input.len(),
                decoded.len()
            );
            if let Err(e) =
                maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&decoded])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "generate" => {
            let stdin_is_piped = !io::stdin().is_terminal();
            let file_path = match (file1.as_deref(), file2.as_deref()) {
                (Some(f), _) if !(stdin_is_piped && f.parse::<i64>().is_ok()) => Some(f),
                _ => None,
            };
            let input = if let Some(path) = file_path {
                read_file(path)
            } else {
                read_stdin_all_for_generate()
            };
            let generated = cli_unwrap(
                ctx.try_generate_bytes_with_config(&input, generate_len_bytes, generate_config),
                "generate_bytes_with_config",
            );
            if let Err(e) = io::stdout().write_all(&generated) {
                eprintln!("Error writing generated output: {e}");
                std::process::exit(1);
            }
            if let Err(e) = io::stdout().flush() {
                eprintln!("Error flushing generated output: {e}");
                std::process::exit(1);
            }
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&input])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "ncd" | "ncd_vitanyi" | "ncd_sym" | "ncd_sym_vitanyi" | "ncd_cons" | "ncd_sym_cons" => {
            let f1 = file1.unwrap_or_exit("Error: NCD requires two files");
            let f2 = file2.unwrap_or_exit("Error: NCD requires two files");
            let _method = pos_arg3.or(method_str).unwrap_or_else(|| "5".to_string());
            let variant = match primitive.as_str() {
                "ncd_sym" | "ncd_sym_vitanyi" => NcdVariant::SymVitanyi,
                "ncd_cons" => NcdVariant::Cons,
                "ncd_sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };
            let b1 = read_file(&f1);
            let b2 = read_file(&f2);
            println!(
                "{}",
                ncd_bytes_backend(&b1, &b2, &ctx.compression_backend, variant)
            );
            if let Err(e) =
                maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&b1, &b2])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "entropy" | "h" | "entropy_rate" | "h_rate" => {
            let f1 = file1.unwrap_or_exit("Error: 'h' requires a file");
            let data = read_file(&f1);
            // `h`/`entropy` -> empirical (zero-order, IID) Shannon entropy.
            // `h_rate`/`entropy_rate` -> algorithmic entropy rate via active rate backend.
            // The `rate_backend_specified` flag promotes `h`/`entropy` to the
            // algorithmic path so that `--rate-backend X h file` behaves intuitively.
            if !primitive.contains("rate") && !rate_backend_specified {
                println!("{}", empirical_entropy_bytes(&data));
            } else {
                println!(
                    "{}",
                    cli_unwrap(
                        ctx.try_entropy_rate_bytes(&data),
                        "InfotheoryCtx::try_entropy_rate_bytes",
                    )
                );
            }
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&data])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "id" => {
            let f1 = file1.unwrap_or_exit("Error: 'id' requires a file");
            let data = read_file(&f1);
            println!("{:.6}", intrinsic_dependence_bytes(&data));
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&data])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        other => {
            let f1 = file1.unwrap_or_exit("Error: requires two files");
            let f2 = file2.unwrap_or_exit("Error: requires two files");
            let b1 = read_file(&f1);
            let b2 = read_file(&f2);
            // For two-file primitives the `_rate_specified` flag promotes the
            // empirical helpers to algorithmic rate-backend variants whenever the
            // user explicitly requested a rate backend on the command line.
            let res = match other {
                "ned" if rate_backend_specified => ned_bytes(&b1, &b2),
                "ned" => empirical_ned_bytes(&b1, &b2),
                "ned_cons" if rate_backend_specified => ned_cons_bytes(&b1, &b2),
                "ned_cons" => empirical_ned_cons_bytes(&b1, &b2),
                "nte" if rate_backend_specified => nte_bytes(&b1, &b2),
                "nte" => empirical_nte_bytes(&b1, &b2),
                "mi" | "mutual_info" if rate_backend_specified => {
                    mutual_information_bytes(&b1, &b2)
                }
                "mi" | "mutual_info" => empirical_mutual_information_bytes(&b1, &b2),
                "ce" | "conditional_entropy" if rate_backend_specified => {
                    conditional_entropy_bytes(&b1, &b2)
                }
                "ce" | "conditional_entropy" => {
                    let h_xy = empirical_joint_entropy_bytes(&b1, &b2);
                    let h_y = empirical_entropy_bytes(&b2);
                    (h_xy - h_y).max(0.0)
                }
                "xe" | "cross_entropy" if rate_backend_specified => cross_entropy_bytes(&b1, &b2),
                "xe" | "cross_entropy" => empirical_cross_entropy_bytes(&b1, &b2),
                "joint_entropy" | "h_xy" if rate_backend_specified => {
                    joint_entropy_rate_bytes(&b1, &b2)
                }
                "joint_entropy" | "h_xy" => empirical_joint_entropy_bytes(&b1, &b2),
                "rt" | "resistance" if rate_backend_specified => {
                    resistance_to_transformation_bytes(&b1, &b2)
                }
                "rt" | "resistance" => empirical_resistance_to_transformation_bytes(&b1, &b2),
                "tvd" => tvd_paths(&f1, &f2),
                "nhd" => nhd_paths(&f1, &f2),
                "kl" | "kl_divergence" => kl_divergence_paths(&f1, &f2),
                "js" | "js_divergence" => js_divergence_paths(&f1, &f2),
                _ => {
                    eprintln!("Unknown primitive: {}", other);
                    print_usage();
                    return;
                }
            };
            println!("{}", res);
            if let Err(e) =
                maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&b1, &b2])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn print_usage() {
    let rate_backends = infotheory::backends::available_rate_backends()
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            if idx == 0 {
                format!("'{name}' (default)")
            } else {
                format!("'{name}'")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let compression_backends = infotheory::backends::available_compression_backends()
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            if idx == 0 {
                format!("'{name}' (default)")
            } else {
                format!("'{name}'")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    eprintln!(
        r#"InfoTheory CLI
Usage: infotheory <primitive> [args...] [options]

Primitives:
  Entropy & Information:
    h, entropy <file>                       Empirical (order-0/IID) Shannon entropy; with --rate-backend uses the rate backend
    h_rate, entropy_rate <file>             Algorithmic entropy rate via the active rate backend
    mi, mutual_info <f1> <f2>               Mutual information I(X;Y) (empirical; rate-backend if --rate-backend)
    xe, cross_entropy <f1> <f2>             Cross entropy (empirical; rate-backend if --rate-backend)
    ce, conditional_entropy <f1> <f2>       Conditional entropy H(X|Y) (empirical; rate-backend if --rate-backend)
    joint_entropy, h_xy <f1> <f2>           Joint entropy H(X,Y) (empirical; rate-backend if --rate-backend)
    id <file>                               Intrinsic dependence ID(X) using the active rate backend

  Distance & Divergence:
    ncd <f1> <f2> [method]                  Normalized Compression Distance (Vitanyi)
    ncd_sym, ncd_cons, ncd_sym_cons         NCD variants (Symmetric, Conservative, etc.)
    ned <f1> <f2>                           Normalized entropy distance (empirical; rate-backend if --rate-backend)
    nte <f1> <f2>                           Normalized transform effort (empirical; rate-backend if --rate-backend)
    kl, kl_divergence <f1> <f2>             Kullback-Leibler divergence (empirical histograms)
    js, js_divergence <f1> <f2>             Jensen-Shannon divergence (empirical histograms)
    tvd <f1> <f2>                           Total variation distance (empirical histograms)
    nhd <f1> <f2>                           Normalized Hellinger distance (empirical histograms)
    rt, resistance <f1> <f2>                Resistance to transformation

  Tools:
    search <query> <target> [options]       Search target using info-theoretic ranking
    aixi <config.json>                      Run AIXI agent
    tune <spec.json|spec.itsd> [options]    Run tuner with executor-side controls
    batch                                   Run in JSON-L batch mode
    generate [file]                         Generate continuation from file or piped stdin
    compress <in> <out>                     Compress file using selected compression backend
    decompress <in> <out>                   Decompress file using selected compression backend
    ctw-profile <input|-> [--depth N]       Emit FAC-CTW arena telemetry as JSONL
    ac-log-loss <input> --mixture <spec.json> --out-prefix <prefix>
                                          Emit exact AC/log-loss TSV diagnostics for a mixture
    sequitur-debug <input>|--hex <hex> [--hex <hex> ...]
                                          Emit canonical Sequitur grammar and bounded predictive traces

Options:
    --rate-backend <name>   Backend for rate estimation: {rate_backends}
  --compression-backend <name>
                          Backend for NCD/compression: {compression_backends}
  --method <val>          Method/config (e.g. '5' for zpaq, '16' for ctw, mixture spec path,
                          model method: file:/path/model.safetensors[;policy:...] or cfg:key=value,...[;policy:...])
  --rate-backend-json <path>
                          Load canonical RateBackend JSON (relative paths resolve against this file's directory).
                          Incompatible with --rate-backend and --expert-spec. When used with --method, the method applies to the compression backend shorthand.
  --compression-backend-json <path>
                          Load canonical CompressionBackend JSON (e.g. tuner output). Incompatible with
                          --compression-backend, --expert-spec, and --method. Optional --rate-backend-json must match the embedded rate model when the compression object includes one.
  --expert-spec <path>    Load one exact standalone expert JSON (same schema as a mixture 'experts' entry)
  --model-export <path>   Optional online model export path (.safetensors + .json sidecar)
  --mixture <path>        Mixture spec for 'ac-log-loss'
  --out-prefix <prefix>   Output prefix for 'ac-log-loss' TSVs
  --hex <hex>             Hex-encoded byte string for 'sequitur-debug' (repeatable)
  --context-bytes <n>     Sequitur context width (default: 64)
  --alphabet-prefix <n>   Prefix of predictive PDF to emit for 'sequitur-debug'
  --bytes <n>             Bytes to generate for 'generate' (default: 8)
  --sample                Use seeded sampling for generation
  --greedy                Force deterministic greedy generation
  --adaptive              Keep fitting on generated bytes instead of frozen continuation
  --seed <u64>            RNG seed for sampled generation
  --temperature <x>       Sampling temperature (default: 1.0)
  --top-k <n>             Sample only from the top-k bytes (0 disables)
  --top-p <p>             Nucleus sampling threshold in (0, 1]
  --exec-config <path>    Tune executor profile JSON (for `tune`)
  --max-evaluations <n>   Optional tuning evaluation cap (for `tune`)
  --annealer-kernel-profile <name>
                          Tune annealer profile: reversible_elementary_metropolis|compiled_uniform_metropolis_hastings
  --cpu-affinity <csv>    CPU affinity (comma-separated core ids, for `tune`)
  --threads <n>           Executor thread hint for tuning runs (for `tune`)
  --evaluator-worker-executable <path>
                          Explicit tuner evaluator worker executable path (for `tune`)
  --evaluator-cgroup-parent <path>
                          Delegated cgroup-v2 eval-parent (typically .../infotheory-tuner/evals)
  --warmup-baseline-runs <n>
                          Baseline warmup runs before normative baseline eval (for `tune`)
  --self-improvement-rounds <n>
                          Optional bounded online delayed-label update rounds (for `tune`)
  --stagnation-reset-evals <n>
                          Optional stagnation reset threshold (for `tune`)
  --log-path <path>       Optional JSONL executor event log output (for `tune`)
  --diagnostic-chunk-bytes <n>
                          Diagnostic report chunk size over charged target bytes (for `tune`)
  --rss-mode <mode>       Memory accounting mode:
                          process_rss_peak (explicit Unix RSS fallback) |
                          backend_reported (diagnostic backend component, RSS deployability) |
                          hybrid_strict_max (strict Linux cgroup-v2 + RSS max) (for `tune`)
  --planner-deployable-model
                          Use executor-side planner deployability diagnostics in the evaluator profile (for `tune`)
  --warmstart-trace-refresh
                          Rebuild warm-start exact-J_H from merged same-task live traces between rounds (for `tune`)
  --timing-tier <tier>    Theorem timing tier: best_effort|isolated|real_time|deterministic_table (for `tune`)
  --determinism-deadline-certificate <ref>
                          Determinism/deadline certification reference (for `tune`)
  --deterministic-evaluator-table <ref>
                          Verified deterministic evaluator table JSON path (for `tune`)
  --finite-planner-state-certificate <ref>
                          Verified finite planner-state certificate JSON path (for `tune`)
  --no-hidden-state-certificate <ref>
                          Verified no-hidden/inert-state certificate JSON path (for `tune`)
  --exact-reward-encoding-certificate <ref>
                          Verified exact reward encoding certificate JSON path (for `tune`)
  --emit-exact-reward-encoding-certificate <path>
                          Emit an exact reward encoding certificate JSON bound to resolved dataset/bounds/evaluator profile and exit (for `tune`)
  --exact-state-observation-certificate <ref>
                          Verified exact-state observation certificate JSON path (for `tune`)
  --observation-adapter-spec-ref <ref>
                          Observation adapter spec reference (for `tune`)
  --exact-state-encoder-spec-ref <ref>
                          Exact-state encoder specification reference (for `tune`)
  --scalar-representation-ref <ref>
                          Scalar representation specification reference (for `tune`)
  --claim-exact-finite-mdp
                          Request theorem-facing exact finite-MDP claim path (for `tune`)
  --claim-exact-observed-markov
                          Request theorem-facing exact observed-Markov claim path (for `tune`)
  --claim-planner-convergence
                          Request theorem-facing planner convergence claim path (for `tune`)

Examples:
  infotheory ncd file1.txt file2.txt --compression-backend zpaq --method 5
  infotheory ncd file1.txt file2.txt --compression-backend rate-ac --rate-backend ctw
  infotheory h file.txt --expert-spec ./expert.json
  infotheory h file.txt --rate-backend mamba --method "cfg:hidden=128,layers=2,intermediate=256,state=16,conv=4,train=adam,lr=0.001;policy:schedule=0..100:train(scope=head+bias,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)" --model-export ./mamba_online.safetensors
  infotheory h file.txt --rate-backend ctw --method 32
  infotheory h file.txt --rate-backend mixture --method mixture.json
  infotheory sequitur-debug --hex 616263616263 --alphabet-prefix 8
  infotheory search "encryption" ./src --prior "codebase context"
  cat prompt.txt | infotheory generate --rate-backend ctw --method 32 --bytes 8
  infotheory generate prompt.txt --rate-backend match --bytes 16 --sample --seed 7
  infotheory compress in.bin out.itc --compression-backend rate-ac --rate-backend mixture --method mixture.json
  infotheory decompress out.itc restored.bin --compression-backend rate-ac --rate-backend mixture --method mixture.json
  RAYON_NUM_THREADS=4 infotheory ac-log-loss corpus.bin --mixture configs/bench/mixture.json --out-prefix /tmp/mixture-diagnostic
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use infotheory::aixi::warmstart_contract::{
        WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF,
        WARMSTART_STANDALONE_SCALAR_REPRESENTATION, standalone_teacher_provenance_crc32_pair,
        warmstart_exact_jh_planner_task_fingerprint,
    };
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_TEST_PATH_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_path(prefix: &str, suffix: &str) -> PathBuf {
        let counter = TEMP_TEST_PATH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "{prefix}-{}-{nanos}-{counter}{suffix}",
            std::process::id()
        ))
    }

    fn canonical_test_path_string(path: &std::path::Path) -> String {
        path.to_string_lossy().replace('\\', "/")
    }

    #[cfg(feature = "backend-ctw")]
    fn action_alphabet(n: usize) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
    }

    #[cfg(feature = "backend-ctw")]
    fn sample_compiled_planner_run() -> CompiledPlannerRunSpec {
        let document = SpecDocument::parse_json_value(
            &json!({
                "schema_version": 1,
                "kind": "planner_run",
                "assets": [],
                "environment": {
                    "kind": "builtin",
                    "name": "coin_flip"
                },
                "interface": {
                    "observation_bits": 2,
                    "observation_stream_len": 1,
                    "observation_key_mode": "full_stream",
                    "reward_bits": 4,
                    "agent_actions": action_alphabet(2).get()
                },
                "controller": {
                    "kind": "aiqi_discounted",
                    "predictor": {
                        "kind": "ctw",
                        "depth": 4
                    },
                    "discount_gamma": 0.5,
                    "return_horizon": 2,
                    "return_bins": 8,
                    "augmentation_period": 2,
                    "baseline_exploration": 0.1
                },
                "runtime": {
                    "random_seed": 7,
                    "learn_cycles": 3,
                    "eval_cycles": 1,
                    "terminate_lifetime": 3,
                    "log_every": 1,
                    "perf": false,
                    "vm_perf_only": false,
                    "explore_epsilon": 0.25,
                    "explore_gamma": 0.5
                }
            }),
            Path::new("."),
        )
        .expect("sample planner document");
        let SpecDocument::PlannerRun(spec) = document else {
            panic!("expected planner_run document");
        };
        spec.compile().expect("sample planner run should compile")
    }

    #[cfg(feature = "backend-ctw")]
    fn sample_warmstart_compiled_planner_run(teacher_path: &Path) -> CompiledPlannerRunSpec {
        let document = SpecDocument::parse_json_value(
            &json!({
                "schema_version": 1,
                "kind": "planner_run",
                "assets": [{
                    "id": "teacher",
                    "path": teacher_path.to_string_lossy()
                }],
                "environment": {
                    "kind": "builtin",
                    "name": "coin_flip"
                },
                "interface": {
                    "observation_bits": 2,
                    "observation_stream_len": 1,
                    "observation_key_mode": "full_stream",
                    "reward_bits": 2,
                    "agent_actions": action_alphabet(2).get()
                },
                "controller": {
                    "kind": "aiqi_warmstart_exact_jh",
                    "predictor": {
                        "kind": "ctw",
                        "depth": 4
                    },
                    "return_horizon": 1,
                    "return_bins": 4,
                    "label_phase_period": 1,
                    "teacher_dataset_asset": "teacher",
                    "planner_simulations_per_step": 1
                },
                "runtime": {
                    "random_seed": 7,
                    "learn_cycles": 1,
                    "eval_cycles": 1,
                    "terminate_lifetime": 2,
                    "log_every": 1,
                    "perf": false,
                    "vm_perf_only": false,
                    "explore_epsilon": 0.0,
                    "explore_gamma": 1.0
                }
            }),
            Path::new("."),
        )
        .expect("sample warmstart planner document");
        let SpecDocument::PlannerRun(spec) = document else {
            panic!("expected planner_run document");
        };
        spec.compile()
            .expect("sample warmstart planner run should compile")
    }

    #[cfg(feature = "backend-ctw")]
    fn write_warmstart_teacher(
        path: &Path,
        task_fingerprint: &str,
        action_alphabet_size: usize,
        observation_bits: usize,
    ) {
        let reward_bits: usize = 2;
        let observation_stream_len: usize = 1;
        let (adapter_crc, reward_cert) = standalone_teacher_provenance_crc32_pair(
            observation_bits,
            observation_stream_len,
            reward_bits,
        )
        .expect("standalone teacher provenance crc pair");
        std::fs::write(
            path,
            serde_json::to_vec(&json!({
                "schema_version": 1,
                "contract": {
                    "task_fingerprint": task_fingerprint,
                    "action_alphabet_size": action_alphabet_size,
                    "observation_bits": observation_bits,
                    "observation_stream_len": observation_stream_len,
                    "observation_key_mode": "full_stream",
                    "observation_adapter_spec_ref": WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF,
                    "observation_adapter_content_crc32": adapter_crc,
                    "reward_bits": reward_bits,
                    "return_horizon": 1,
                    "label_phase_period": 1,
                    "scalar_representation": WARMSTART_STANDALONE_SCALAR_REPRESENTATION,
                    "exact_reward_encoding_certificate": reward_cert
                },
                "traces": [{
                    "transitions": [
                        {"action": 0, "observations": [1], "reward": 1}
                    ]
                }]
            }))
            .expect("serialize warmstart teacher"),
        )
        .expect("write warmstart teacher");
    }

    /// Mutate one `contract` string field in an on-disk warm-start teacher JSON file.
    #[cfg(feature = "backend-ctw")]
    fn mutate_warmstart_teacher_contract_field(path: &Path, field: &str, wrong: &str) {
        let bytes = std::fs::read(path).expect("read warmstart teacher");
        let mut doc: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse warmstart teacher json");
        let contract = doc
            .as_object_mut()
            .and_then(|root| root.get_mut("contract"))
            .and_then(|c| c.as_object_mut())
            .expect("teacher.contract object");
        contract.insert(field.to_string(), json!(wrong));
        std::fs::write(path, serde_json::to_vec(&doc).expect("serialize teacher"))
            .expect("write warmstart teacher");
    }

    #[cfg(feature = "backend-ctw")]
    #[derive(Clone, Copy)]
    struct CountingEnv {
        observation: u64,
        reward: i64,
        reward_bits: usize,
        action_bits: usize,
        observation_bits: usize,
    }

    #[cfg(feature = "backend-ctw")]
    impl Environment for CountingEnv {
        fn perform_action(&mut self, action: u64) {
            self.observation = self.observation.saturating_add(action + 1);
            self.reward = self.reward.saturating_add(1);
        }

        fn get_observation(&self) -> u64 {
            self.observation
        }

        fn get_reward(&self) -> i64 {
            self.reward
        }

        fn is_finished(&self) -> bool {
            false
        }

        fn get_observation_bits(&self) -> usize {
            self.observation_bits
        }

        fn get_reward_bits(&self) -> usize {
            self.reward_bits
        }

        fn get_action_bits(&self) -> usize {
            self.action_bits
        }
    }

    #[test]
    fn file_roundtrip_backend_keeps_zpaq_unchanged() {
        let b = CompressionBackend::zpaq("5");
        let out = file_roundtrip_backend(&b);
        assert!(matches!(out, CompressionBackend::Zpaq { method, .. } if method.value() == "5"));
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn file_roundtrip_backend_forces_rate_framed() {
        let b = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: infotheory::coders::CoderType::AC,
            framing: infotheory::compression::FramingMode::Raw,
        };
        let out = file_roundtrip_backend(&b);
        match out {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
    }

    #[test]
    fn process_json_line_rejects_invalid_json() {
        let out = process_json_line(r#"{"op":"metrics","text":"abc""#);
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("output should be json");
        assert!(
            parsed
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .contains("invalid json")
        );
    }

    #[test]
    fn canonical_spec_detection_and_legacy_error_messages_are_stable() {
        assert!(is_canonical_spec_document(&json!({
            "schema_version": 1,
            "kind": "planner_run"
        })));
        assert!(!is_canonical_spec_document(&json!({
            "schema_version": 1
        })));
        assert_eq!(
            builtin_environment_name(BuiltinEnvironmentSpec::CoinFlip),
            "coin_flip"
        );

        let err = legacy_planner_config_error("/tmp/legacy.json");
        let msg = err.to_string();
        assert!(msg.contains("legacy aixi JSON configs are no longer executable"));
        assert!(msg.contains("/tmp/legacy.json"));
        assert!(msg.contains("planner_run"));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_teacher_loader_accepts_matching_compiled_planner_contract() {
        let teacher_path = unique_temp_path("warmstart-teacher-matching", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("task fingerprint");
        write_warmstart_teacher(&teacher_path, &task_fingerprint, 2, 2);

        let teacher = load_warmstart_exact_jh_teacher_dataset(&compiled, "teacher")
            .expect("matching teacher contract must load");
        assert_eq!(teacher.contract.task_fingerprint, task_fingerprint);
        assert_eq!(teacher.traces.len(), 1);

        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_teacher_loader_rejects_mismatched_task_fingerprint() {
        let teacher_path = unique_temp_path("warmstart-teacher-task-mismatch", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        write_warmstart_teacher(&teacher_path, "different-task", 2, 2);

        let err = load_warmstart_exact_jh_teacher_dataset(&compiled, "teacher")
            .expect_err("mismatched teacher task must fail");
        assert!(err.to_string().contains("task_fingerprint"), "{err}");

        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_teacher_loader_rejects_action_alphabet_mismatch() {
        let teacher_path = unique_temp_path("warmstart-teacher-interface-mismatch", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("task fingerprint");
        write_warmstart_teacher(&teacher_path, &task_fingerprint, 3, 2);

        let err = load_warmstart_exact_jh_teacher_dataset(&compiled, "teacher")
            .expect_err("mismatched action alphabet must fail");
        assert!(
            err.to_string().contains("planner interface fingerprint"),
            "{err}"
        );

        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_teacher_loader_rejects_observation_adapter_spec_ref_mismatch() {
        let teacher_path = unique_temp_path("warmstart-teacher-adapter-ref", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("task fingerprint");
        write_warmstart_teacher(&teacher_path, &task_fingerprint, 2, 2);
        mutate_warmstart_teacher_contract_field(
            &teacher_path,
            "observation_adapter_spec_ref",
            "wrong-adapter-ref",
        );
        let err = load_warmstart_exact_jh_teacher_dataset(&compiled, "teacher")
            .expect_err("adapter spec ref mismatch must fail");
        assert!(
            err.to_string().contains("observation_adapter_spec_ref"),
            "{err}"
        );
        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_teacher_loader_rejects_observation_adapter_content_crc32_mismatch() {
        let teacher_path = unique_temp_path("warmstart-teacher-adapter-crc", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("task fingerprint");
        write_warmstart_teacher(&teacher_path, &task_fingerprint, 2, 2);
        mutate_warmstart_teacher_contract_field(
            &teacher_path,
            "observation_adapter_content_crc32",
            "deadbeef",
        );
        let err = load_warmstart_exact_jh_teacher_dataset(&compiled, "teacher")
            .expect_err("adapter crc mismatch must fail");
        assert!(
            err.to_string()
                .contains("observation_adapter_content_crc32"),
            "{err}"
        );
        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_teacher_loader_rejects_scalar_representation_mismatch() {
        let teacher_path = unique_temp_path("warmstart-teacher-scalar", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("task fingerprint");
        write_warmstart_teacher(&teacher_path, &task_fingerprint, 2, 2);
        mutate_warmstart_teacher_contract_field(
            &teacher_path,
            "scalar_representation",
            "wrong-scalar",
        );
        let err = load_warmstart_exact_jh_teacher_dataset(&compiled, "teacher")
            .expect_err("scalar representation mismatch must fail");
        assert!(err.to_string().contains("scalar_representation"), "{err}");
        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_teacher_loader_rejects_exact_reward_encoding_certificate_mismatch() {
        let teacher_path = unique_temp_path("warmstart-teacher-cert", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("task fingerprint");
        write_warmstart_teacher(&teacher_path, &task_fingerprint, 2, 2);
        mutate_warmstart_teacher_contract_field(
            &teacher_path,
            "exact_reward_encoding_certificate",
            "wrong-cert",
        );
        let err = load_warmstart_exact_jh_teacher_dataset(&compiled, "teacher")
            .expect_err("reward certificate mismatch must fail");
        assert!(
            err.to_string()
                .contains("exact_reward_encoding_certificate"),
            "{err}"
        );
        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn byte_and_path_metric_wrappers_match_basic_identities() {
        let x = b"banana bandana";
        let y = b"banana bandana";
        let z = b"entropy coding";
        let backend = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: infotheory::coders::CoderType::AC,
            framing: infotheory::compression::FramingMode::Raw,
        }
        .compile()
        .expect("ctw compression backend should compile");

        let ncd_same = ncd_bytes_backend(x, y, &backend, NcdVariant::Vitanyi);
        let ncd_diff = ncd_bytes_backend(x, z, &backend, NcdVariant::Vitanyi);
        assert!(ncd_same.is_finite() && ncd_diff.is_finite());
        assert!(
            ncd_same <= ncd_diff,
            "identical inputs should not rank farther apart"
        );
        assert!(
            ncd_same < 0.6,
            "identical inputs should remain relatively close; got {ncd_same}"
        );

        let id = intrinsic_dependence_bytes(x);
        assert!(id.is_finite());
        assert!(id >= 0.0);

        let mi = mutual_information_bytes(x, y);
        assert!(mi.is_finite());
        assert!(mi >= 0.0);

        let h_y_given_x = conditional_entropy_bytes(y, x);
        assert!(h_y_given_x.is_finite());
        assert!(h_y_given_x >= 0.0);

        let cross = cross_entropy_bytes(z, x);
        assert!(cross.is_finite());
        assert!(cross >= 0.0);

        let joint = joint_entropy_rate_bytes(x, y);
        assert!(joint.is_finite());
        assert!(joint >= 0.0);

        let resistance = resistance_to_transformation_bytes(x, y);
        assert!(resistance.is_finite());
        assert!(resistance >= 0.0);

        let ned = ned_bytes(x, y);
        let ned_cons = ned_cons_bytes(x, y);
        let nte = nte_bytes(x, y);
        assert!(ned.is_finite() && ned >= 0.0);
        assert!(ned_cons.is_finite() && ned_cons >= 0.0);
        assert!(nte.is_finite() && nte >= 0.0);

        let left = unique_temp_path("metric-left", ".txt");
        let right = unique_temp_path("metric-right", ".txt");
        let different = unique_temp_path("metric-different", ".txt");
        std::fs::write(&left, x).expect("write left file");
        std::fs::write(&right, y).expect("write right file");
        std::fs::write(&different, z).expect("write different file");

        let tvd_same = tvd_paths(left.to_str().expect("utf8"), right.to_str().expect("utf8"));
        let tvd_diff = tvd_paths(
            left.to_str().expect("utf8"),
            different.to_str().expect("utf8"),
        );
        assert!(tvd_same.is_finite() && tvd_diff.is_finite());
        assert!(tvd_same <= tvd_diff);

        let nhd_same = nhd_paths(left.to_str().expect("utf8"), right.to_str().expect("utf8"));
        let nhd_diff = nhd_paths(
            left.to_str().expect("utf8"),
            different.to_str().expect("utf8"),
        );
        assert!(nhd_same.is_finite() && nhd_diff.is_finite());
        assert!(nhd_same <= nhd_diff);

        let kl_same =
            kl_divergence_paths(left.to_str().expect("utf8"), right.to_str().expect("utf8"));
        let js_same =
            js_divergence_paths(left.to_str().expect("utf8"), right.to_str().expect("utf8"));
        assert!(kl_same.is_finite() && kl_same >= 0.0);
        assert!(js_same.is_finite() && js_same >= 0.0);

        let _ = std::fs::remove_file(left);
        let _ = std::fs::remove_file(right);
        let _ = std::fs::remove_file(different);
    }

    #[test]
    fn aixi_run_logger_handles_disabled_and_single_sink_modes() {
        assert!(
            AixiRunLogger::new(None)
                .expect("logger creation should succeed")
                .is_none()
        );

        let bits_path = unique_temp_path("aixi-trace-only-bits", ".bin");
        let jsonl_path = unique_temp_path("aixi-trace-only-jsonl", ".jsonl");

        let bits_overlay = json!({
            "trace_bits01_path": bits_path,
        });
        let mut bits_logger = AixiRunLogger::new(Some(&bits_overlay))
            .expect("bits logger")
            .expect("bits logger should be enabled");
        bits_logger.log_action(1, 1).expect("log action to bits");
        bits_logger.next_step().expect("advance bits step");
        drop(bits_logger);
        let bits = std::fs::read(&bits_path).expect("read bits trace");
        assert!(!bits.is_empty());

        let jsonl_overlay = json!({
            "trace_jsonl_path": jsonl_path,
            "trace_flush_every": 2
        });
        let mut jsonl_logger = AixiRunLogger::new(Some(&jsonl_overlay))
            .expect("jsonl logger")
            .expect("jsonl logger should be enabled");
        jsonl_logger
            .log_percept(&[3], 1, 2, 4, 1)
            .expect("log percept to jsonl");
        jsonl_logger.next_step().expect("advance jsonl step");
        drop(jsonl_logger);
        let jsonl = std::fs::read_to_string(&jsonl_path).expect("read jsonl trace");
        assert!(jsonl.contains("\"kind\":\"percept\""));

        let _ = std::fs::remove_file(bits_path);
        let _ = std::fs::remove_file(jsonl_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_run_schedule_derives_cycles_and_extra_exploration() {
        let compiled = sample_compiled_planner_run();
        let mut runtime = compiled.runtime().clone();
        runtime.learn_cycles = None;
        runtime.eval_cycles = Some(3);
        runtime.terminate_lifetime = 5;
        runtime.log_every = 2;
        runtime.perf = true;
        runtime.explore_epsilon = 0.4;
        runtime.explore_gamma = 0.5;
        let schedule = PlannerRunSchedule::from_runtime(&runtime);
        assert_eq!(schedule.learn_cycles, 5);
        assert_eq!(schedule.eval_cycles, 3);
        assert_eq!(schedule.log_every, 2);
        assert!(schedule.perf);
        assert!((schedule.extra_exploration(0) - 0.4).abs() < 1e-12);
        assert!((schedule.extra_exploration(2) - 0.1).abs() < 1e-12);

        let mut no_explore_runtime = runtime.clone();
        no_explore_runtime.learn_cycles = Some(1);
        no_explore_runtime.eval_cycles = None;
        no_explore_runtime.terminate_lifetime = 1;
        no_explore_runtime.explore_epsilon = 0.0;
        no_explore_runtime.explore_gamma = 0.25;
        let no_explore = PlannerRunSchedule::from_runtime(&no_explore_runtime);
        assert_eq!(no_explore.extra_exploration(99), 0.0);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_execution_context_and_vm_perf_only_update_environment_state() {
        let compiled = sample_compiled_planner_run();
        let env = Box::new(CountingEnv {
            observation: 1,
            reward: -1,
            reward_bits: 4,
            action_bits: 1,
            observation_bits: 2,
        });
        validate_action_alphabet(&compiled, env.as_ref()).expect("matching action alphabet");

        let mut ctx = PlannerExecutionContext::new(&compiled, env, None).expect("context");
        assert_eq!(ctx.obs_stream, vec![1]);
        assert_eq!(ctx.rew, -1);

        let reward = ctx.perform_action(0).expect("perform action");
        assert_eq!(reward, 0);
        assert_eq!(ctx.obs_stream, vec![2]);
        assert_eq!(ctx.rew, 0);

        let schedule = PlannerRunSchedule {
            learn_cycles: 2,
            eval_cycles: 0,
            log_every: 0,
            perf: false,
            vm_perf_only: true,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        };
        run_vm_perf_only(&schedule, &mut ctx).expect("vm perf only run");
        assert_eq!(ctx.obs_stream, vec![4]);
        assert_eq!(ctx.rew, 2);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn validate_action_alphabet_reports_mismatch() {
        let compiled = sample_compiled_planner_run();
        let env = CountingEnv {
            observation: 0,
            reward: 0,
            reward_bits: 4,
            action_bits: 2,
            observation_bits: 2,
        };
        let err = validate_action_alphabet(&compiled, &env)
            .expect_err("mismatched action bits must fail");
        assert!(err.to_string().contains("action_alphabet_mismatch"));
        assert!(err.to_string().contains("2 actions"));
        assert!(err.to_string().contains("4"));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn aixi_run_logger_writes_bits_and_jsonl_records() {
        let bits_path = unique_temp_path("aixi-trace-bits", ".bin");
        let jsonl_path = unique_temp_path("aixi-trace-jsonl", ".jsonl");
        let overlay = json!({
            "trace_bits01_path": bits_path,
            "trace_jsonl_path": jsonl_path,
            "trace_flush_every": 1
        });

        let mut logger = AixiRunLogger::new(Some(&overlay))
            .expect("logger setup")
            .expect("logger should be enabled");
        logger.log_action(1, 2).expect("log action");
        logger.log_percept(&[2], 1, 2, 4, 0).expect("log percept");
        logger.next_step().expect("advance step");
        drop(logger);

        let bits = std::fs::read(&bits_path).expect("read bits trace");
        assert!(!bits.is_empty());
        assert!(bits.iter().all(|byte| *byte == 0 || *byte == 1));

        let jsonl = std::fs::read_to_string(&jsonl_path).expect("read jsonl trace");
        assert!(jsonl.contains("\"kind\":\"action\""));
        assert!(jsonl.contains("\"kind\":\"percept\""));

        let _ = std::fs::remove_file(bits_path);
        let _ = std::fs::remove_file(jsonl_path);
    }

    #[test]
    fn process_json_line_parses_escaped_and_nested_json_correctly() {
        let line = r#"{
            "op":"metrics",
            "text":"hello\n\"json\"",
            "meta":{"op":"ncd"},
            "max_order":-1
        }"#;
        let out = process_json_line(line);
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("output should be json");
        if infotheory::api::RateBackend::try_default().is_ok() {
            assert!(parsed.get("h0").and_then(|v| v.as_f64()).unwrap_or(-1.0) >= 0.0);
            assert_eq!(parsed.get("len").and_then(|v| v.as_u64()), Some(12));
        } else {
            let err = parsed
                .get("error")
                .and_then(|v| v.as_str())
                .expect("backend-free build should return structured batch error");
            assert!(
                err.contains("metrics failed")
                    && err.contains("no default rate backend is available in this build"),
                "unexpected error: {err}"
            );
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn build_ctx_rwkv7_compression_accepts_cfg_method() {
        let ctx = crate::cli::build_ctx(
            "rosaplus",
            "rwkv7",
            Some(
                "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer",
            ),
            None,
        )
        .ctx;

        match ctx.compression_backend.canonical_spec() {
            CompressionBackend::Rate {
                rate_backend,
                coder,
                framing,
            } => {
                assert!(matches!(rate_backend, RateBackend::Rwkv7Method { .. }));
                assert_eq!(*coder, infotheory::coders::CoderType::AC);
                assert_eq!(*framing, infotheory::compression::FramingMode::Raw);
            }
            _ => panic!("expected rate-coded RWKV backend for cfg: method"),
        }
    }

    #[test]
    fn parse_backend_aliases_and_unknowns() {
        #[cfg(feature = "backend-rosa")]
        assert_eq!(parse_rate_backend("rosa"), Some("rosaplus"));
        #[cfg(feature = "backend-ctw")]
        assert_eq!(parse_rate_backend("fac-ctw"), Some("fac-ctw"));
        #[cfg(feature = "backend-match")]
        assert_eq!(parse_rate_backend("sparse-match"), Some("sparse-match"));
        #[cfg(feature = "backend-ppmd")]
        assert_eq!(parse_rate_backend("ppmd"), Some("ppmd"));
        #[cfg(feature = "backend-calibrated")]
        assert_eq!(parse_rate_backend("calibrated"), Some("calibrated"));
        assert_eq!(parse_rate_backend("facctw"), None);
        assert_eq!(parse_rate_backend("sparsematch"), None);
        assert_eq!(parse_rate_backend("ppm"), None);
        assert_eq!(parse_rate_backend("cal"), None);
        assert_eq!(parse_rate_backend("unknown"), None);

        assert_eq!(parse_compression_backend("unknown"), None);
        #[cfg(feature = "backend-zpaq")]
        assert_eq!(parse_compression_backend("zpaq"), Some("zpaq"));
        assert_eq!(parse_compression_backend("rate-ac"), Some("rate-ac"));
        assert_eq!(parse_compression_backend("rate-rans"), Some("rate-rans"));
        assert_eq!(parse_compression_backend("rate_ac"), None);
        assert_eq!(parse_compression_backend("raterans"), None);
        assert_eq!(parse_compression_backend("rate_rans"), None);
        #[cfg(feature = "backend-rwkv")]
        {
            assert_eq!(parse_compression_backend("rwkv7"), Some("rwkv7"));
        }
        assert_eq!(parse_compression_backend("rwkv"), None);
        #[cfg(feature = "backend-mamba")]
        {
            assert_eq!(parse_rate_backend("mamba"), Some("mamba"));
        }
        assert_eq!(parse_rate_backend("mamba1"), None);
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn parse_mixture_expert_supports_calibrated_and_match_backends() {
        let base_dir = Path::new(".");
        let expert = json!({
            "name": "cal-ctw",
            "kind": "calibrated",
            "context": "text",
            "bins": 33,
            "learning_rate": 0.02,
            "bias_clip": 4.0,
            "base": {
                "kind": "match"
            }
        });
        let parsed = parse_mixture_expert_value(&expert, base_dir, 4).expect("expert should parse");
        match parsed.backend {
            RateBackend::Calibrated { spec } => match spec.base {
                RateBackend::Match { .. } => {}
                _ => panic!("unexpected calibrated base"),
            },
            _ => panic!("expected calibrated backend"),
        }
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn parse_mixture_expert_supports_sequitur_backend() {
        let base_dir = Path::new(".");
        let expert = json!({
            "name": "sequitur",
            "kind": "sequitur",
            "context_bytes": 96
        });
        let parsed = parse_mixture_expert_value(&expert, base_dir, 4).expect("expert should parse");
        match parsed.backend {
            RateBackend::Sequitur { context_bytes } => assert_eq!(context_bytes, 96),
            _ => panic!("expected sequitur backend"),
        }
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn parse_mixture_expert_resolves_mamba_model_path_relative_to_base_dir() {
        let base_dir = unique_temp_path("infotheory-mamba-relpath", "");
        std::fs::create_dir_all(base_dir.join("weights")).expect("create temp dir");
        let rel_path = "weights/model;v1.safetensors";
        let expected = canonical_test_path_string(&base_dir.join(rel_path));
        let expert = json!({
            "name": "mamba-relative",
            "kind": "mamba",
            "model_path": rel_path
        });
        let err = match parse_mixture_expert_value(&expert, &base_dir, 4) {
            Ok(_) => panic!("missing model should return an error"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains(&expected),
            "error should mention resolved absolute model path. expected substring: {expected}, got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_mixture_expert_resolves_rwkv_model_path_relative_to_base_dir() {
        let base_dir = unique_temp_path("infotheory-rwkv-relpath", "");
        std::fs::create_dir_all(base_dir.join("weights")).expect("create temp dir");
        let rel_path = "weights/model;v1.safetensors";
        let expected = canonical_test_path_string(&base_dir.join(rel_path));
        let expert = json!({
            "name": "rwkv-relative",
            "kind": "rwkv7",
            "model_path": rel_path
        });
        let err = match parse_mixture_expert_value(&expert, &base_dir, 4) {
            Ok(_) => panic!("missing model should return an error"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains(&expected),
            "error should mention resolved absolute model path. expected substring: {expected}, got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn load_expert_spec_preserves_exact_ppmd_settings() {
        let expert_path = unique_temp_path("infotheory-expert-spec", ".json");
        std::fs::write(
            &expert_path,
            serde_json::to_vec(&json!({
                "name": "ppmd",
                "kind": "ppmd",
                "order": 12,
                "memory_mb": 256
            }))
            .expect("expert json"),
        )
        .expect("write temp expert spec");

        let parsed = load_expert_spec(expert_path.to_str().expect("utf8 path"))
            .expect("ppmd expert should load");
        match parsed.backend {
            RateBackend::Ppmd { order, memory_mb } => {
                assert_eq!(order, 12);
                assert_eq!(memory_mb, 256);
            }
            _ => panic!("expected ppmd backend"),
        }

        let _ = std::fs::remove_file(&expert_path);
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn build_ctx_loads_expert_spec_with_rosa_max_order() {
        let expert_path = unique_temp_path("infotheory-expert-spec-rosa", ".json");
        std::fs::write(
            &expert_path,
            serde_json::to_vec(&json!({
                "name": "rosa",
                "kind": "rosaplus",
                "max_order": 32
            }))
            .expect("expert json"),
        )
        .expect("write temp expert spec");

        let built = crate::cli::build_ctx(
            "rosaplus",
            "zpaq",
            None,
            Some(expert_path.to_string_lossy().as_ref()),
        );
        assert!(matches!(
            built.ctx.rate_backend.canonical_spec(),
            RateBackend::RosaPlus { max_order: 32 }
        ));

        let _ = std::fs::remove_file(&expert_path);
    }

    #[test]
    fn parse_observation_helpers_cover_vm_and_non_vm_cases() {
        let base = json!({
            "observation_stream_len": 3,
            "observation_key_mode": "stream_hash"
        });
        assert_eq!(parse_observation_stream_len(&base), 3);
        assert_eq!(
            parse_observation_key_mode(&base).expect("parse stream_hash mode"),
            ObservationKeyMode::StreamHash
        );
        assert_eq!(
            parse_observation_key_mode_str("full_stream").expect("parse full_stream"),
            ObservationKeyMode::FullStream
        );
        assert_eq!(
            parse_observation_key_mode_str("last").expect("parse last"),
            ObservationKeyMode::Last
        );
        assert!(parse_observation_key_mode_str("unknown").is_err());

        let vm = json!({
            "observation_stream_len": 2,
            "observation_key_mode": "full_stream",
            "vm_observation": {
                "stream_len": 2,
                "key_mode": "last"
            }
        });
        assert_eq!(
            parse_observation_stream_len_for_vm(&vm["vm_observation"]),
            2
        );
        assert_eq!(
            parse_observation_key_mode_for_vm(&vm["vm_observation"]).expect("parse vm mode"),
            ObservationKeyMode::Last
        );
        assert_eq!(parse_observation_stream_len_for_env(&vm, "vm"), 2);
        assert_eq!(
            parse_observation_key_mode_for_env(&vm, "vm").expect("parse vm env mode"),
            ObservationKeyMode::Last
        );
        assert_eq!(
            parse_observation_key_mode_for_env(&vm, "coin").expect("parse non-vm mode"),
            ObservationKeyMode::FullStream
        );

        let mismatch = json!({
            "observation_stream_len": 2,
            "vm_observation": {
                "stream_len": 3
            }
        });
        let err = validate_observation_config("vm", &mismatch, 2, ObservationKeyMode::FullStream)
            .expect_err("mismatched vm stream_len should fail");
        assert!(err.to_string().contains("conflicts"));

        let mismatch_mode = json!({
            "observation_key_mode": "full_stream",
            "vm_observation": {
                "key_mode": "last"
            }
        });
        let err = validate_observation_config("vm", &mismatch_mode, 1, ObservationKeyMode::Last)
            .expect_err("mismatched vm key mode should fail");
        assert!(err.to_string().contains("conflicts"));
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn parse_mixture_kind_and_spec_validation() {
        assert_eq!(
            parse_mixture_kind("bayes").expect("bayes kind"),
            MixtureKind::Bayes
        );
        assert_eq!(
            parse_mixture_kind("switching").expect("switching kind"),
            MixtureKind::Switching
        );
        assert_eq!(
            parse_mixture_kind("convex").expect("convex kind"),
            MixtureKind::Convex
        );
        assert_eq!(
            parse_mixture_kind("neural").expect("neural kind"),
            MixtureKind::Neural
        );
        assert!(parse_mixture_kind("bayes-mix").is_err());
        assert!(parse_mixture_kind("switch").is_err());
        assert!(parse_mixture_kind("nonsense").is_err());
        assert_eq!(
            parse_mixture_schedule("theorem").expect("theorem schedule"),
            MixtureScheduleMode::Theorem
        );
        assert!(parse_mixture_schedule("nonsense").is_err());

        let base_dir = Path::new(".");
        let missing_experts = json!({
            "kind": "bayes",
            "experts": []
        });
        assert!(parse_mixture_spec_value(&missing_experts, base_dir, 8).is_err());

        let fading_without_decay = json!({
            "kind": "fading",
            "experts": [
                {"name": "ctw-e", "kind": "ctw", "depth": 4}
            ]
        });
        assert!(parse_mixture_spec_value(&fading_without_decay, base_dir, 8).is_err());

        let valid = json!({
            "kind": "convex",
            "schedule": "theorem",
            "experts": [
                {"name": "ctw-e", "kind": "ctw", "depth": 8},
                {"name": "fac-e", "kind": "fac-ctw", "base_depth": 8, "encoding_bits": 8}
            ]
        });
        let spec = parse_mixture_spec_value(&valid, base_dir, 8).expect("valid mixture");
        assert_eq!(spec.schedule, MixtureScheduleMode::Theorem);
        assert_eq!(spec.experts.len(), 2);
        assert!(matches!(spec.kind, MixtureKind::Convex));

        let nested = json!({
            "kind": "convex",
            "alpha": 1.25,
            "experts": [
                {
                    "name": "nested",
                    "kind": "mixture",
                    "spec": {
                        "kind": "bayes",
                        "experts": [
                            {"name": "ctw-e", "kind": "ctw", "depth": 4}
                        ]
                    }
                },
                {"name": "match-e", "kind": "match", "hash_bits": 18}
            ]
        });
        let nested_spec = parse_mixture_spec_value(&nested, base_dir, 8).expect("nested mixture");
        assert!(matches!(nested_spec.kind, MixtureKind::Convex));
        assert_eq!(nested_spec.experts.len(), 2);
        match &nested_spec.experts[0].backend {
            RateBackend::Mixture { spec } => {
                assert!(matches!(spec.kind, MixtureKind::Bayes));
                assert_eq!(spec.experts.len(), 1);
            }
            _ => panic!("expected nested mixture backend"),
        }
    }

    #[cfg(all(feature = "vm", feature = "all-backends"))]
    #[test]
    fn parse_vm_stats_backend_supports_new_backends_and_rejects_unknowns() {
        let root = json!({
            "algorithm": "ctw",
            "ct_depth": 8,
            "observation_bits": 8,
            "reward_bits": 8
        });
        let base_dir = Path::new(".");

        let matched =
            parse_vm_stats_backend(&json!({"kind":"match","hash_bits":18}), &root, base_dir)
                .expect("match backend should parse");
        assert!(matches!(matched, RateBackend::Match { hash_bits: 18, .. }));

        let sparse = parse_vm_stats_backend(
            &json!({"kind":"sparse-match","gap_min":2,"gap_max":4}),
            &root,
            base_dir,
        )
        .expect("sparse-match backend should parse");
        assert!(matches!(
            sparse,
            RateBackend::SparseMatch {
                gap_min: 2,
                gap_max: 4,
                ..
            }
        ));

        let ppmd = parse_vm_stats_backend(&json!({"kind":"ppmd","order":12}), &root, base_dir)
            .expect("ppmd backend should parse");
        assert!(matches!(ppmd, RateBackend::Ppmd { order: 12, .. }));

        let sequitur = parse_vm_stats_backend(
            &json!({"kind":"sequitur","context_bytes":72}),
            &root,
            base_dir,
        )
        .expect("sequitur backend should parse");
        assert!(matches!(
            sequitur,
            RateBackend::Sequitur { context_bytes: 72 }
        ));

        let particle = parse_vm_stats_backend(
            &json!({
                "kind":"particle",
                "spec":{"num_particles":4,"num_cells":4,"cell_dim":8}
            }),
            &root,
            base_dir,
        )
        .expect("particle backend should parse");
        assert!(matches!(particle, RateBackend::Particle { .. }));

        let mixture = parse_vm_stats_backend(
            &json!({
                "kind":"mixture",
                "spec":{"kind":"bayes","experts":[{"kind":"match"}]}
            }),
            &root,
            base_dir,
        )
        .expect("mixture backend should parse");
        assert!(matches!(mixture, RateBackend::Mixture { .. }));

        let calibrated = parse_vm_stats_backend(
            &json!({
                "kind":"calibrated",
                "base":{"kind":"ctw","depth":8},
                "context":"text",
                "bins":17,
                "learning_rate":0.05,
                "bias_clip":3.0
            }),
            &root,
            base_dir,
        )
        .expect("calibrated backend should parse");
        assert!(matches!(calibrated, RateBackend::Calibrated { .. }));

        let err = match parse_vm_stats_backend(&json!("unknown-backend"), &root, base_dir) {
            Ok(_) => panic!("unknown backend should not silently fall back"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("unknown vm stats backend"),
            "unexpected error: {err}"
        );
    }

    #[cfg(all(feature = "vm", feature = "backend-ctw"))]
    #[test]
    fn parse_vm_stats_backend_preserves_fac_ctw_vm_defaults() {
        let root = json!({
            "algorithm": "fac-ctw",
            "ct_depth": 11,
            "observation_bits": 13,
            "reward_bits": 5
        });
        let parsed = parse_vm_stats_backend(&json!({"kind":"fac-ctw"}), &root, Path::new("."))
            .expect("fac-ctw backend should parse");
        match parsed {
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits,
                encoding_bits,
            } => {
                assert_eq!(base_depth, 32);
                assert_eq!(encoding_bits, 8);
                assert_eq!(num_percept_bits, 18);
            }
            _ => panic!("expected fac-ctw backend"),
        }
    }

    #[cfg(all(feature = "backend-ctw", feature = "tuner"))]
    #[test]
    fn run_aixi_mode_rejects_non_planner_spec_documents() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "infotheory-spec-kind-{}-{nanos}.json",
            std::process::id()
        ));
        let doc_value = json!({
            "schema_version": 1,
            "kind": "tune",
            "assets": [
                {
                    "id": "dataset",
                    "path": "input.bin"
                }
            ],
            "input_asset": "dataset",
            "baseline_candidate": {
                "kind": "rate-ac",
                "rate_backend": {
                    "kind": "ctw",
                    "depth": 8
                },
                "framing": "framed"
            },
            "controller": {
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1
            },
            "bounds": {
                "allowed_backends": ["ctw"],
                "forbidden_backends": [],
                "parameter_ranges": [],
                "max_experts": 2,
                "max_mixture_nesting_depth": 1,
                "min_experts": 1,
                "allow_duplicate_experts": false,
                "required_experts": [],
                "forbidden_expert_pairs": []
            },
            "eval_time_limit_seconds": 1.0,
            "time_budget_seconds": 2.0,
            "min_throughput_bytes_per_second": 1.0,
            "max_memory_bytes": 1024,
            "output_config_path": "best.json",
            "seed": 7,
            "report_path": null
        });
        let doc = infotheory::spec::SpecDocument::parse_json_value(&doc_value, Path::new("."))
            .expect("canonical tune document");
        std::fs::write(&path, doc.to_canonical_json().expect("canonical json"))
            .expect("write temp spec");

        let err = run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect_err("non planner spec should be rejected");
        assert!(err.to_string().contains("planner_run document"));

        let _ = std::fs::remove_file(path);
    }

    #[cfg(all(not(feature = "backend-ctw"), feature = "tuner"))]
    #[test]
    fn run_aixi_mode_surfaces_backend_validation_for_non_planner_documents() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "infotheory-canonical-non-planner-no-ctw-{nanos}.json"
        ));
        let doc_value = json!({
            "schema_version": 1,
            "kind": "tune",
            "assets": [
                {
                    "id": "dataset",
                    "path": "input.bin"
                }
            ],
            "input_asset": "dataset",
            "baseline_candidate": {
                "kind": "rate-ac",
                "rate_backend": {
                    "kind": "ctw",
                    "depth": 8
                },
                "framing": "framed"
            },
            "controller": {
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1
            },
            "bounds": {
                "allowed_backends": ["ctw"],
                "forbidden_backends": [],
                "parameter_ranges": [],
                "max_experts": 2,
                "max_mixture_nesting_depth": 1,
                "min_experts": 1,
                "allow_duplicate_experts": false,
                "required_experts": [],
                "forbidden_expert_pairs": []
            },
            "eval_time_limit_seconds": 1.0,
            "time_budget_seconds": 2.0,
            "min_throughput_bytes_per_second": 1.0,
            "max_memory_bytes": 1024,
            "output_config_path": "best.json",
            "seed": 7,
            "report_path": null
        });
        std::fs::write(
            &path,
            serde_json::to_vec(&doc_value).expect("serialize canonical json"),
        )
        .expect("write temp spec");

        let err = run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect_err("missing backend feature should be surfaced");
        assert!(
            err.to_string()
                .contains("requires infotheory feature 'backend-ctw'"),
            "{err}"
        );

        let _ = std::fs::remove_file(path);
    }

    #[cfg(all(feature = "backend-ctw", feature = "aixi-gameengine"))]
    #[test]
    fn run_aixi_mode_accepts_canonical_planner_run_documents() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "infotheory-planner-run-{}-{nanos}.json",
            std::process::id()
        ));
        let doc_value = json!({
            "schema_version": 1,
            "kind": "planner_run",
            "assets": [],
            "environment": {
                "kind": "builtin",
                "name": "coin_flip"
            },
            "interface": {
                "observation_bits": 1,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "reward_bits": 1,
                "agent_actions": 2
            },
            "controller": {
                "kind": "mc_aixi",
                "predictor": {
                    "kind": "ctw",
                    "depth": 8
                },
                "agent_horizon": 1,
                "num_simulations": 1,
                "mcts_strategy": {
                    "kind": "rho_uct"
                },
                "exploration_exploitation_ratio": 1.0,
                "discount_gamma": 1.0
            },
            "runtime": {
                "random_seed": 7,
                "learn_cycles": 1,
                "eval_cycles": 0,
                "terminate_lifetime": 1,
                "log_every": 1,
                "perf": false,
                "vm_perf_only": false,
                "explore_epsilon": 0.0,
                "explore_gamma": 1.0
            }
        });
        let doc = infotheory::spec::SpecDocument::parse_json_value(&doc_value, Path::new("."))
            .expect("canonical planner document");
        std::fs::write(&path, doc.to_canonical_json().expect("canonical json"))
            .expect("write temp planner spec");

        run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect("canonical planner_run document should execute");

        let _ = std::fs::remove_file(path);
    }

    #[cfg(all(feature = "backend-ctw", feature = "aixi-gameengine"))]
    #[test]
    fn run_aixi_mode_rejects_legacy_interface_reward_range_fields() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "infotheory-planner-run-invalid-reward-{}-{nanos}.json",
            std::process::id()
        ));
        let legacy_doc = json!({
            "schema_version": 1,
            "kind": "planner_run",
            "assets": [],
            "environment": {
                "kind": "builtin",
                "name": "coin_flip"
            },
            "interface": {
                "observation_bits": 1,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "reward_bits": 1,
                "agent_actions": 2,
                "min_reward": 0,
                "max_reward": 100,
                "reward_offset": 0
            },
            "controller": {
                "kind": "mc_aixi",
                "predictor": {
                    "kind": "ctw",
                    "depth": 8
                },
                "agent_horizon": 1,
                "num_simulations": 1,
                "mcts_strategy": {
                    "kind": "rho_uct"
                },
                "exploration_exploitation_ratio": 1.0,
                "discount_gamma": 1.0
            },
            "runtime": {
                "random_seed": 7,
                "learn_cycles": 1,
                "eval_cycles": 0,
                "terminate_lifetime": 1,
                "log_every": 1,
                "perf": false,
                "vm_perf_only": false,
                "explore_epsilon": 0.0,
                "explore_gamma": 1.0
            }
        });
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&legacy_doc).expect("legacy planner json"),
        )
        .expect("write temp planner spec");

        let err = run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect_err("legacy interface reward range fields should be rejected");
        assert!(
            err.to_string()
                .contains("unknown interface field 'min_reward'"),
            "{err}"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn run_aixi_mode_rejects_legacy_planner_json_documents() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "infotheory-legacy-planner-{}-{nanos}.json",
            std::process::id()
        ));
        let legacy = serde_json::json!({
            "environment": "coin-flip",
            "planner": "mc-aixi",
            "algorithm": "ctw",
            "ct_depth": 8,
            "agent_horizon": 1,
            "observation_bits": 1,
            "observation_stream_len": 1,
            "observation_key_mode": "full_stream",
            "reward_bits": 1,
            "agent_actions": 2,
            "num_simulations": 1,
            "discount_gamma": 1.0,
        });
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&legacy).expect("legacy planner json"),
        )
        .expect("write temp legacy planner config");

        let err = run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect_err("legacy planner json should be rejected");
        let message = err.to_string();
        assert!(message.contains("legacy aixi JSON configs are no longer executable"));
        assert!(message.contains("planner_run"));
        assert!(message.contains("schema_version"));
        assert!(message.contains("kind"));

        let _ = std::fs::remove_file(path);
    }
}
