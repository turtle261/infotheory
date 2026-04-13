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
//! infotheory <primitive> <file1> <file2> [method/max_order]
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
use infotheory::aixi::common::RandomGenerator;
use infotheory::aixi::environment::{
    BiasedRockPaperScissor, CoinFlip, CtwTest, Environment, ExtendedTiger, KuhnPoker, TicTacToe,
};
#[cfg(all(test, feature = "vm"))]
use infotheory::aixi::vm_nyx::{
    FuzzMutator as NyxFuzzMutator, NyxActionFilter, NyxActionSource, NyxActionSpec, NyxFuzzConfig,
    NyxObservationPolicy, NyxObservationStreamMode, NyxProtocolConfig, NyxRewardPolicy,
    NyxRewardShaping, NyxTraceConfig, PayloadEncoding as NyxPayloadEncoding,
};
#[cfg(feature = "vm")]
use infotheory::aixi::vm_nyx::{NyxVmConfig, NyxVmEnvironment};
use infotheory::api::*;
#[cfg(feature = "backend-mamba")]
use infotheory::mambazip;
#[cfg(feature = "backend-rwkv")]
use infotheory::rwkvzip;
#[cfg(feature = "backend-sequitur")]
use infotheory::sequitur::{CanonicalSymbol, SequiturModel};
use infotheory::spec::{
    self, BuiltinEnvironmentSpec, CompiledPlannerController, CompiledPlannerRunSpec,
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
    build_ctx, file_roundtrip_compiled_backend, load_mixture_spec, maybe_export_online_model,
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
#[cfg(test)]
use infotheory::aixi::common::ObservationKeyMode;
#[cfg(feature = "backend-rosa")]
use infotheory::search;

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

fn intrinsic_dependence_bytes(data: &[u8], max_order: i64) -> f64 {
    cli_unwrap(
        try_intrinsic_dependence_bytes(data, max_order),
        "intrinsic_dependence_bytes",
    )
}

fn mutual_information_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    cli_unwrap(
        try_mutual_information_bytes(x, y, max_order),
        "mutual_information_bytes",
    )
}

fn conditional_entropy_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    cli_unwrap(
        try_conditional_entropy_bytes(x, y, max_order),
        "conditional_entropy_bytes",
    )
}

fn cross_entropy_bytes(test_data: &[u8], train_data: &[u8], max_order: i64) -> f64 {
    cli_unwrap(
        try_cross_entropy_bytes(test_data, train_data, max_order),
        "cross_entropy_bytes",
    )
}

fn joint_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    cli_unwrap(
        try_joint_entropy_rate_bytes(x, y, max_order),
        "joint_entropy_rate_bytes",
    )
}

fn resistance_to_transformation_bytes(x: &[u8], tx: &[u8], max_order: i64) -> f64 {
    cli_unwrap(
        try_resistance_to_transformation_bytes(x, tx, max_order),
        "resistance_to_transformation_bytes",
    )
}

fn ned_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    cli_unwrap(try_ned_bytes(x, y, max_order), "ned_bytes")
}

fn ned_cons_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    cli_unwrap(try_ned_cons_bytes(x, y, max_order), "ned_cons_bytes")
}

fn nte_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    cli_unwrap(try_nte_bytes(x, y, max_order), "nte_bytes")
}

fn tvd_paths(x: &str, y: &str, max_order: i64) -> f64 {
    cli_unwrap(try_tvd_paths(x, y, max_order), "tvd_paths")
}

fn nhd_paths(x: &str, y: &str, max_order: i64) -> f64 {
    cli_unwrap(try_nhd_paths(x, y, max_order), "nhd_paths")
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
    match spec {
        BuiltinEnvironmentSpec::CoinFlip => "coin-flip",
        BuiltinEnvironmentSpec::CtwTest => "ctw-test",
        BuiltinEnvironmentSpec::ExtendedTiger => "extended-tiger",
        BuiltinEnvironmentSpec::TicTacToe => "tictactoe",
        BuiltinEnvironmentSpec::BiasedRockPaperScissor => "biased-rock-paper-scissor",
        BuiltinEnvironmentSpec::KuhnPoker => "kuhn-poker",
    }
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
    agent_actions: usize,
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
            reward_offset: compiled.interface().reward_offset,
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
}

impl PlannerControllerRuntime {
    fn from_compiled(compiled: &CompiledPlannerRunSpec) -> anyhow::Result<Self> {
        match compiled.controller() {
            CompiledPlannerController::McAixi { .. } => {
                let agent =
                    Agent::from_compiled_planner_run(compiled).map_err(anyhow::Error::msg)?;
                let explore_rng = if let Some(seed) = compiled.runtime().random_seed {
                    RandomGenerator::from_seed(seed).fork_with(0x4558_504c_4f52_455f)
                } else {
                    RandomGenerator::new()
                };
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
            CompiledPlannerController::AiqiWarmstartExactJh { .. } => Err(anyhow::anyhow!(
                "planner_run controller kind 'aiqi_warmstart_exact_jh' is not executable from the CLI yet"
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
                            explore_rng.gen_range(ctx.agent_actions) as u64
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
        }
    }
}

fn controller_backend_label(controller: &CompiledPlannerController) -> String {
    match controller {
        CompiledPlannerController::McAixi {
            predictor,
            predictor_max_order,
            ..
        }
        | CompiledPlannerController::AiqiDiscounted {
            predictor,
            predictor_max_order,
            ..
        }
        | CompiledPlannerController::AiqiWarmstartExactJh {
            predictor,
            predictor_max_order,
            ..
        } => predictor.display_label(*predictor_max_order),
    }
}

fn build_builtin_environment(spec: BuiltinEnvironmentSpec) -> Box<dyn Environment> {
    match spec {
        BuiltinEnvironmentSpec::CoinFlip => Box::new(CoinFlip::new(0.9)),
        BuiltinEnvironmentSpec::CtwTest => Box::new(CtwTest::new()),
        BuiltinEnvironmentSpec::ExtendedTiger => Box::new(ExtendedTiger::new()),
        BuiltinEnvironmentSpec::TicTacToe => Box::new(TicTacToe::new()),
        BuiltinEnvironmentSpec::BiasedRockPaperScissor => Box::new(BiasedRockPaperScissor::new()),
        BuiltinEnvironmentSpec::KuhnPoker => Box::new(KuhnPoker::new()),
    }
}

#[cfg(feature = "vm")]
fn build_planner_environment(
    compiled: &CompiledPlannerRunSpec,
) -> anyhow::Result<(Box<dyn Environment>, &'static str)> {
    match &compiled.canonical_spec().environment {
        spec::EnvironmentSpec::Builtin { builtin } => Ok((
            build_builtin_environment(*builtin),
            builtin_environment_name(*builtin),
        )),
        spec::EnvironmentSpec::NyxVm(vm) => {
            let config = NyxVmConfig::from_environment_spec(vm, compiled.resolved_assets())
                .map_err(anyhow::Error::msg)?;
            Ok((Box::new(NyxVmEnvironment::new(config)?), "vm"))
        }
    }
}

#[cfg(not(feature = "vm"))]
fn build_planner_environment(
    compiled: &CompiledPlannerRunSpec,
) -> anyhow::Result<(Box<dyn Environment>, &'static str)> {
    match &compiled.canonical_spec().environment {
        spec::EnvironmentSpec::Builtin { builtin } => Ok((
            build_builtin_environment(*builtin),
            builtin_environment_name(*builtin),
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
    if let Some(seed) = compiled.runtime().random_seed {
        env.set_random_seed(seed);
    }
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
        CompiledPlannerController::AiqiWarmstartExactJh { .. } => unreachable!(),
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
    match infotheory::spec::load_spec_document(config_path).map_err(anyhow::Error::msg)? {
        SpecDocument::PlannerRun(spec) => {
            let compiled = spec
                .compile_in(&spec::SpecEnvironment::new(config_dir))
                .map_err(anyhow::Error::msg)?;
            run_compiled_planner_run(&compiled, json_overlay.as_ref())
        }
        SpecDocument::Tune(_) => Err(anyhow::anyhow!(
            "aixi expects a planner_run document, found kind 'tune'"
        )),
        SpecDocument::RateBackend(_) => Err(anyhow::anyhow!(
            "aixi expects a planner_run document, found kind 'rate_backend'"
        )),
        SpecDocument::CompressionBackend(_) => Err(anyhow::anyhow!(
            "aixi expects a planner_run document, found kind 'compression_backend'"
        )),
    }
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

    let mut opts = search::SearchOptions::default();
    let mut rate_backend = infotheory::search::DEFAULT_SEARCH_RATE_BACKEND_NAME.to_string();
    let compression_backend =
        infotheory::search::DEFAULT_SEARCH_COMPRESSION_BACKEND_NAME.to_string();
    let mut method: Option<String> = None;
    let mut expert_spec_path: Option<String> = None;
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
            "--max-order" => {
                i += 1;
                opts.max_order = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1);
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
                rate_backend = parse_rate_backend(v)
                    .unwrap_or(infotheory::search::DEFAULT_SEARCH_RATE_BACKEND_NAME)
                    .to_string();
            }
            "--method" => {
                i += 1;
                method = args.get(i).cloned();
            }
            "--expert-spec" => {
                i += 1;
                expert_spec_path = args.get(i).cloned();
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
    opts.ctx = build_ctx(
        &rate_backend,
        &compression_backend,
        method.as_deref(),
        expert_spec_path.as_deref(),
    )
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
    if primitive == "batch" {
        run_batch_mode();
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
                rate_backend_str = parse_rate_backend(v).unwrap_or("rosaplus").to_string();
                rate_backend_specified = true;
            }
            "--ncd-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --compression-backend requires a value");
                compression_backend_str =
                    parse_compression_backend(v).unwrap_or("zpaq").to_string();
            }
            "--compression-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --compression-backend requires a value");
                compression_backend_str =
                    parse_compression_backend(v).unwrap_or("zpaq").to_string();
            }
            "--method" => {
                i += 1;
                method_str = args.get(i).cloned();
            }
            "--expert-spec" => {
                i += 1;
                expert_spec_path = args.get(i).cloned();
                rate_backend_specified = true;
            }
            "--model-export" | "--rwkv-export" => {
                i += 1;
                model_export_path = args.get(i).cloned();
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

    let built_ctx = build_ctx(
        &rate_backend_str,
        &compression_backend_str,
        method_str.as_deref(),
        expert_spec_path.as_deref(),
    );
    let ctx = built_ctx.ctx;
    let expert_spec_max_order = built_ctx.expert_spec_max_order;
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
            // Disambiguate positional args for `generate [file] [max_order]`.
            // When stdin is piped and the first positional looks like an integer,
            // treat it as max_order (not a file path).
            let stdin_is_piped = !io::stdin().is_terminal();
            let (file_path, explicit_max_order) = match (file1.as_deref(), file2.as_deref()) {
                // `generate <file> <max_order>` — both present
                (Some(f), Some(mo)) => (Some(f), mo.parse::<i64>().ok()),
                // `generate <arg>` — single positional:
                //   if stdin is piped and it parses as an integer, it's max_order
                //   otherwise it's a file path
                (Some(arg), None) if stdin_is_piped && arg.parse::<i64>().is_ok() => {
                    (None, arg.parse::<i64>().ok())
                }
                (Some(f), None) => (Some(f), None),
                // No positionals at all
                (None, _) => (None, None),
            };
            let max_order = explicit_max_order
                .or(pos_arg3.as_deref().and_then(|s| s.parse().ok()))
                .or(expert_spec_max_order)
                .unwrap_or(-1);
            let input = if let Some(path) = file_path {
                read_file(path)
            } else {
                read_stdin_all_for_generate()
            };
            let generated = cli_unwrap(
                ctx.try_generate_bytes_with_config(
                    &input,
                    generate_len_bytes,
                    max_order,
                    generate_config,
                ),
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
            let default_order = if primitive.contains("rate") || rate_backend_specified {
                expert_spec_max_order.unwrap_or(-1)
            } else {
                0
            };
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_order);
            let data = read_file(&f1);
            if max_order == 0 && !primitive.contains("rate") && !rate_backend_specified {
                println!("{}", marginal_entropy_bytes(&data));
            } else {
                println!(
                    "{}",
                    cli_unwrap(
                        ctx.try_entropy_rate_bytes(&data, max_order),
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
        "id" | "intrinsic_dep" => {
            let f1 = file1.unwrap_or_exit("Error: 'id' requires a file");
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(expert_spec_max_order.unwrap_or(-1));
            let data = read_file(&f1);
            println!("{:.6}", intrinsic_dependence_bytes(&data, max_order));
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&data])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        other => {
            let f1 = file1.unwrap_or_exit("Error: requires two files");
            let f2 = file2.unwrap_or_exit("Error: requires two files");
            let default_order = if rate_backend_specified {
                expert_spec_max_order.unwrap_or(-1)
            } else {
                0
            };
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_order);
            let b1 = read_file(&f1);
            let b2 = read_file(&f2);
            let res = match other {
                "ned" => ned_bytes(&b1, &b2, max_order),
                "ned_cons" => ned_cons_bytes(&b1, &b2, max_order),
                "nte" => nte_bytes(&b1, &b2, max_order),
                "mi" | "mutual_info" => mutual_information_bytes(&b1, &b2, max_order),
                "ce" | "conditional_entropy" => conditional_entropy_bytes(&b1, &b2, max_order),
                "xe" | "cross_entropy" => cross_entropy_bytes(&b1, &b2, max_order),
                "joint_entropy" | "h_xy" => {
                    if max_order == 0 {
                        joint_marginal_entropy_bytes(&b1, &b2)
                    } else {
                        joint_entropy_rate_bytes(&b1, &b2, max_order)
                    }
                }
                "rt" | "resistance" => resistance_to_transformation_bytes(&b1, &b2, max_order),
                "tvd" => tvd_paths(&f1, &f2, max_order),
                "nhd" => nhd_paths(&f1, &f2, max_order),
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
    h, entropy <file> [max_order]           Entropy (marginal if order=0, rate if >0)
    h_rate, entropy_rate <file> [max_order] Force entropy rate estimation
    mi, mutual_info <f1> <f2> [max_order]   Mutual Information I(X;Y)
    xe, cross_entropy <f1> <f2> [max_order] Cross Entropy H(X,Y) - H(Y)? (Check def)
    ce, conditional_entropy <f1> <f2>       Conditional Entropy H(X|Y)
    joint_entropy, h_xy <f1> <f2>           Joint Entropy H(X,Y)
    id, intrinsic_dep <file> [max_order]    Intrinsic Dependence

  Distance & Divergence:
    ncd <f1> <f2> [method]                  Normalized Compression Distance (Vitanyi)
    ncd_sym, ncd_cons, ncd_sym_cons         NCD variants (Symmetric, Conservative, etc.)
    ned <f1> <f2> [max_order]               Normalized Entropy Distance
    nte <f1> <f2> [max_order]               Normalized Transform Effort
    kl, kl_divergence <f1> <f2>             Kullback-Leibler Divergence
    js, js_divergence <f1> <f2>             Jensen-Shannon Divergence
    tvd <f1> <f2>                           Total Variation Distance
    nhd <f1> <f2>                           Normalized Hellinger Distance
    rt, resistance <f1> <f2>                Resistance to Transformation

  Tools:
    search <query> <target> [options]       Search target using info-theoretic ranking
    aixi <config.json>                      Run AIXI agent
    batch                                   Run in JSON-L batch mode
    generate [file] [max_order]             Generate continuation from file or piped stdin
    compress <in> <out>                     Compress file using selected compression backend
    decompress <in> <out>                   Decompress file using selected compression backend
    ac-log-loss <input> --mixture <spec.json> --out-prefix <prefix>
                                          Emit exact AC/log-loss TSV diagnostics for a mixture
    sequitur-debug <input>|--hex <hex> [--hex <hex> ...]
                                          Emit canonical Sequitur grammar and bounded predictive traces

Options:
    --rate-backend <name>   Backend for rate estimation: {rate_backends}
  --compression-backend <name>
                          Backend for NCD/compression: {compression_backends}
  --ncd-backend <name>    Deprecated alias for --compression-backend
  --method <val>          Method/config (e.g. '5' for zpaq, '16' for ctw, mixture spec path,
                          model method: file:/path/model.safetensors[;policy:...] or cfg:key=value,...[;policy:...])
  --expert-spec <path>    Load one exact standalone expert JSON (same schema as a mixture 'experts' entry)
  --model-export <path>   Optional online model export path (.safetensors + .json sidecar)
  --rwkv-export <path>    Backward-compatible alias for --model-export
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
    use serde_json::json;
    #[cfg(any(
        feature = "all-backends",
        feature = "backend-mamba",
        feature = "backend-rwkv"
    ))]
    use std::path::{Path, PathBuf};
    #[cfg(any(
        feature = "all-backends",
        feature = "backend-mamba",
        feature = "backend-rwkv"
    ))]
    use std::sync::atomic::{AtomicU64, Ordering};
    #[cfg(any(
        feature = "all-backends",
        feature = "backend-mamba",
        feature = "backend-rwkv"
    ))]
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(any(
        feature = "all-backends",
        feature = "backend-mamba",
        feature = "backend-rwkv"
    ))]
    static TEMP_TEST_PATH_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[cfg(any(
        feature = "all-backends",
        feature = "backend-mamba",
        feature = "backend-rwkv"
    ))]
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

    #[test]
    fn file_roundtrip_backend_keeps_zpaq_unchanged() {
        let b = CompressionBackend::Zpaq {
            method: "5".to_string(),
        };
        let out = file_roundtrip_backend(&b);
        assert!(matches!(out, CompressionBackend::Zpaq { method } if method == "5"));
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
        let ctx = build_ctx(
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
        assert_eq!(parse_rate_backend("facctw"), Some("fac-ctw"));
        #[cfg(feature = "backend-match")]
        assert_eq!(parse_rate_backend("sparsematch"), Some("sparse-match"));
        #[cfg(feature = "backend-ppmd")]
        assert_eq!(parse_rate_backend("ppm"), Some("ppmd"));
        #[cfg(feature = "backend-calibrated")]
        assert_eq!(parse_rate_backend("cal"), Some("calibrated"));
        assert_eq!(parse_rate_backend("unknown"), None);

        assert_eq!(parse_compression_backend("unknown"), None);
        #[cfg(feature = "backend-zpaq")]
        assert_eq!(parse_compression_backend("zpaq"), Some("zpaq"));
        assert_eq!(parse_compression_backend("rate_ac"), Some("rate-ac"));
        assert_eq!(parse_compression_backend("raterans"), Some("rate-rans"));
        #[cfg(feature = "backend-rwkv")]
        {
            assert_eq!(parse_compression_backend("rwkv"), Some("rwkv7"));
        }
        #[cfg(feature = "backend-mamba")]
        {
            assert_eq!(parse_rate_backend("mamba1"), Some("mamba"));
        }
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
        let expected = base_dir.join(rel_path).to_string_lossy().to_string();
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
        let expected = base_dir.join(rel_path).to_string_lossy().to_string();
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
    fn build_ctx_propagates_expert_spec_max_order_default() {
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

        let built = build_ctx(
            "rosaplus",
            "zpaq",
            None,
            Some(expert_path.to_str().expect("utf8 path")),
        );
        assert_eq!(built.expert_spec_max_order, Some(32));
        assert!(matches!(
            built.ctx.rate_backend.canonical_spec(),
            RateBackend::RosaPlus
        ));

        let _ = std::fs::remove_file(&expert_path);
    }

    #[test]
    fn parse_observation_helpers_cover_vm_and_non_vm_cases() {
        let base = json!({
            "observation_stream_len": 3,
            "observation_key_mode": "stream-hash"
        });
        assert_eq!(parse_observation_stream_len(&base), 3);
        assert_eq!(
            parse_observation_key_mode(&base),
            ObservationKeyMode::StreamHash
        );
        assert_eq!(
            parse_observation_key_mode_str("full"),
            ObservationKeyMode::FullStream
        );
        assert_eq!(
            parse_observation_key_mode_str("last"),
            ObservationKeyMode::Last
        );
        assert_eq!(
            parse_observation_key_mode_str("unknown"),
            ObservationKeyMode::First
        );

        let vm = json!({
            "observation_stream_len": 2,
            "observation_key_mode": "full",
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
            parse_observation_key_mode_for_vm(&vm["vm_observation"]),
            ObservationKeyMode::Last
        );
        assert_eq!(parse_observation_stream_len_for_env(&vm, "vm"), 2);
        assert_eq!(
            parse_observation_key_mode_for_env(&vm, "vm"),
            ObservationKeyMode::Last
        );
        assert_eq!(
            parse_observation_key_mode_for_env(&vm, "coin"),
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
            "observation_key_mode": "full",
            "vm_observation": {
                "key_mode": "last"
            }
        });
        let err = validate_observation_config("nyx", &mismatch_mode, 1, ObservationKeyMode::Last)
            .expect_err("mismatched vm key mode should fail");
        assert!(err.to_string().contains("conflicts"));
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn parse_mixture_kind_and_spec_validation() {
        assert_eq!(
            parse_mixture_kind("bayes-mix").expect("bayes alias"),
            MixtureKind::Bayes
        );
        assert_eq!(
            parse_mixture_kind("switch").expect("switch alias"),
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
            parse_vm_stats_backend(&json!({"name":"match","hash_bits":18}), &root, base_dir)
                .expect("match backend should parse");
        assert!(matches!(matched, RateBackend::Match { hash_bits: 18, .. }));

        let sparse = parse_vm_stats_backend(
            &json!({"name":"sparse-match","gap_min":2,"gap_max":4}),
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

        let ppmd = parse_vm_stats_backend(&json!({"name":"ppmd","order":12}), &root, base_dir)
            .expect("ppmd backend should parse");
        assert!(matches!(ppmd, RateBackend::Ppmd { order: 12, .. }));

        let sequitur = parse_vm_stats_backend(
            &json!({"name":"sequitur","context_bytes":72}),
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
                "name":"particle",
                "spec":{"num_particles":4,"num_cells":4,"cell_dim":8}
            }),
            &root,
            base_dir,
        )
        .expect("particle backend should parse");
        assert!(matches!(particle, RateBackend::Particle { .. }));

        let mixture = parse_vm_stats_backend(
            &json!({
                "name":"mixture",
                "spec":{"kind":"bayes","experts":[{"kind":"match"}]}
            }),
            &root,
            base_dir,
        )
        .expect("mixture backend should parse");
        assert!(matches!(mixture, RateBackend::Mixture { .. }));

        let calibrated = parse_vm_stats_backend(
            &json!({
                "name":"calibrated",
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
        let parsed = parse_vm_stats_backend(&json!({"name":"facctw"}), &root, Path::new("."))
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

    #[cfg(feature = "backend-ctw")]
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
        let doc = infotheory::spec::SpecDocument::Tune(infotheory::spec::TuneSpec {
            assets: vec![infotheory::spec::AssetBinding {
                id: "dataset".to_string(),
                path: "input.bin".to_string(),
            }],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 8 },
                coder: infotheory::coders::CoderType::AC,
                framing: infotheory::compression::FramingMode::Framed,
            },
            controller: infotheory::spec::TuneControllerSpec::AnnealedHillClimbing(
                infotheory::spec::AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: 1,
                },
            ),
            bounds: infotheory::spec::TuneBoundsSpec {
                allowed_backends: vec!["ctw".to_string()],
                forbidden_backends: vec![],
                parameter_ranges: vec![],
                max_experts: 2,
                max_mixture_nesting_depth: 1,
                min_experts: Some(1),
                allow_duplicate_experts: Some(false),
                required_experts: vec![],
                forbidden_expert_pairs: vec![],
            },
            eval_time_limit_seconds: 1.0,
            time_budget_seconds: 2.0,
            min_throughput_bytes_per_second: 1.0,
            max_memory_bytes: 1024,
            output_config_path: "best.json".to_string(),
            seed: 7,
            report_path: None,
        });
        std::fs::write(&path, doc.to_canonical_json().expect("canonical json"))
            .expect("write temp spec");

        let err = run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect_err("non planner spec should be rejected");
        assert!(err.to_string().contains("planner_run document"));

        let _ = std::fs::remove_file(path);
    }

    #[cfg(not(feature = "backend-ctw"))]
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
        let doc = infotheory::spec::SpecDocument::Tune(infotheory::spec::TuneSpec {
            assets: vec![infotheory::spec::AssetBinding {
                id: "dataset".to_string(),
                path: "input.bin".to_string(),
            }],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 8 },
                coder: infotheory::coders::CoderType::AC,
                framing: infotheory::compression::FramingMode::Framed,
            },
            controller: infotheory::spec::TuneControllerSpec::AnnealedHillClimbing(
                infotheory::spec::AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: 1,
                },
            ),
            bounds: infotheory::spec::TuneBoundsSpec {
                allowed_backends: vec!["ctw".to_string()],
                forbidden_backends: vec![],
                parameter_ranges: vec![],
                max_experts: 2,
                max_mixture_nesting_depth: 1,
                min_experts: Some(1),
                allow_duplicate_experts: Some(false),
                required_experts: vec![],
                forbidden_expert_pairs: vec![],
            },
            eval_time_limit_seconds: 1.0,
            time_budget_seconds: 2.0,
            min_throughput_bytes_per_second: 1.0,
            max_memory_bytes: 1024,
            output_config_path: "best.json".to_string(),
            seed: 7,
            report_path: None,
        });
        std::fs::write(&path, doc.to_canonical_json().expect("canonical json"))
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

    #[cfg(feature = "backend-ctw")]
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
        let doc = infotheory::spec::SpecDocument::PlannerRun(infotheory::spec::PlannerRunSpec {
            assets: Vec::new(),
            environment: infotheory::spec::EnvironmentSpec::Builtin {
                builtin: infotheory::spec::BuiltinEnvironmentSpec::CoinFlip,
            },
            interface: infotheory::spec::PlannerInterfaceSpec {
                observation_bits: 1,
                observation_stream_len: 1,
                observation_key_mode: ObservationKeyMode::FullStream,
                reward_bits: 1,
                agent_actions: 2,
                min_reward: 0,
                max_reward: 1,
                reward_offset: 0,
            },
            controller: infotheory::spec::ControllerSpec::McAixi(
                infotheory::spec::McAixiControllerSpec {
                    predictor: RateBackend::Ctw { depth: 8 },
                    predictor_max_order: 8,
                    agent_horizon: 1,
                    num_simulations: 1,
                    exploration_exploitation_ratio: 1.0,
                    discount_gamma: 1.0,
                },
            ),
            runtime: infotheory::spec::PlannerRuntimeSpec {
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
        });
        std::fs::write(&path, doc.to_canonical_json().expect("canonical json"))
            .expect("write temp planner spec");

        run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect("canonical planner_run document should execute");

        let _ = std::fs::remove_file(path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn run_aixi_mode_rejects_unrepresentable_reward_ranges_in_canonical_specs() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "infotheory-planner-run-invalid-reward-{}-{nanos}.json",
            std::process::id()
        ));
        let doc = infotheory::spec::SpecDocument::PlannerRun(infotheory::spec::PlannerRunSpec {
            assets: Vec::new(),
            environment: infotheory::spec::EnvironmentSpec::Builtin {
                builtin: infotheory::spec::BuiltinEnvironmentSpec::CoinFlip,
            },
            interface: infotheory::spec::PlannerInterfaceSpec {
                observation_bits: 1,
                observation_stream_len: 1,
                observation_key_mode: ObservationKeyMode::FullStream,
                reward_bits: 1,
                agent_actions: 2,
                min_reward: 0,
                max_reward: 100,
                reward_offset: 0,
            },
            controller: infotheory::spec::ControllerSpec::McAixi(
                infotheory::spec::McAixiControllerSpec {
                    predictor: RateBackend::Ctw { depth: 8 },
                    predictor_max_order: 8,
                    agent_horizon: 1,
                    num_simulations: 1,
                    exploration_exploitation_ratio: 1.0,
                    discount_gamma: 1.0,
                },
            ),
            runtime: infotheory::spec::PlannerRuntimeSpec {
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
        });
        std::fs::write(&path, doc.to_canonical_json().expect("canonical json"))
            .expect("write temp planner spec");

        let err = run_aixi_mode(path.to_str().expect("utf8 path"))
            .expect_err("invalid reward interface should be rejected");
        assert!(err.to_string().contains("reward_bits too small"), "{err}");

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
            "observation_key_mode": "full-stream",
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
