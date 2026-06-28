use infotheory::aixi::common::{Action, Reward, observation_repr_from_stream};
use infotheory::aixi::planner_agent::{
    PlannerActionProvenance, PlannerAgentError, PlannerControllerAgent, PlannerCycleObserver,
    PlannerEnvironment, PlannerPhase, PlannerRunSession, PlannerSchedule,
    build_planner_environment,
};
use infotheory::aixi::warmstart::{warmstart_jsonl_action_record, warmstart_jsonl_percept_record};
#[cfg(test)]
use infotheory::spec::BuiltinEnvironmentSpec;
use infotheory::spec::{self, CompiledPlannerController, CompiledPlannerRunSpec, SpecDocument};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[cfg(test)]
pub(crate) fn is_canonical_spec_document(value: &serde_json::Value) -> bool {
    is_canonical_spec_document_runtime(value)
}

#[cfg(test)]
pub(crate) fn builtin_environment_name(spec: BuiltinEnvironmentSpec) -> &'static str {
    spec.canonical_name()
}

fn is_canonical_spec_document_runtime(value: &serde_json::Value) -> bool {
    value["schema_version"].as_u64().is_some() && value["kind"].as_str().is_some()
}

fn legacy_interface_reward_range_field(value: &serde_json::Value) -> Option<&'static str> {
    let interface = value.get("interface")?.as_object()?;
    ["min_reward", "max_reward", "reward_offset"]
        .into_iter()
        .find(|field| interface.contains_key(*field))
}

#[cfg(test)]
pub(crate) fn legacy_planner_config_error(path: &str) -> anyhow::Error {
    legacy_planner_config_error_runtime(path)
}

fn legacy_planner_config_error_runtime(path: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "legacy aixi JSON configs are no longer executable; convert '{}' to a canonical planner_run document with top-level 'schema_version' and 'kind'",
        path
    )
}

fn legacy_interface_reward_range_error(path: &str, field: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "unknown interface field '{field}': legacy aixi reward-range fields are no longer accepted in planner_run documents; remove 'min_reward', 'max_reward', and 'reward_offset' from '{}' and use the canonical reward_bits-derived interface contract",
        path
    )
}

/// Planner telemetry logger for complete observable interaction traces.
///
/// The JSONL sink records normalized action and percept events. The `bits01`
/// sink records the same telemetry stream in the planner's bit encoding: each
/// action contributes `action_bits` bytes, and each percept contributes all
/// observations plus reward bytes. It is a trace representation, not a
/// codelength accounting surface for the predictor's actual online updates.
pub(crate) struct AixiRunLogger {
    bits01: Option<BufWriter<File>>,
    jsonl: Option<BufWriter<File>>,
    flush_every: usize,
    completed_steps: usize,
}

impl AixiRunLogger {
    pub(crate) fn new(v: Option<&serde_json::Value>) -> anyhow::Result<Option<Self>> {
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
            completed_steps: 0,
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

    pub(crate) fn flush(&mut self) -> anyhow::Result<()> {
        if let Some(w) = self.bits01.as_mut() {
            w.flush()?;
        }
        if let Some(w) = self.jsonl.as_mut() {
            w.flush()?;
        }
        Ok(())
    }

    pub(crate) fn log_percept(
        &mut self,
        step: usize,
        observations: &[u64],
        reward: i64,
        observation_bits: usize,
        reward_bits: usize,
        reward_offset: i64,
    ) -> anyhow::Result<()> {
        // Same symbol encoding as the controller, applied to the telemetry event.
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
            let rec = warmstart_jsonl_percept_record(step, observations, reward);
            writeln!(w, "{rec}")?;
        }
        Ok(())
    }

    pub(crate) fn log_action(
        &mut self,
        step: usize,
        action: Action,
        action_bits: usize,
        provenance: PlannerActionProvenance,
    ) -> anyhow::Result<()> {
        let mut bits = Vec::new();
        infotheory::aixi::common::encode(&mut bits, action, action_bits);
        self.write_bits01(&bits)?;

        if let Some(w) = self.jsonl.as_mut() {
            let rec = warmstart_jsonl_action_record(step, action, provenance);
            writeln!(w, "{rec}")?;
        }
        Ok(())
    }

    pub(crate) fn next_step(&mut self) -> anyhow::Result<()> {
        self.completed_steps = self.completed_steps.saturating_add(1);
        if self.flush_every > 0 && self.completed_steps.is_multiple_of(self.flush_every) {
            self.flush()?;
        }
        Ok(())
    }
}

fn controller_backend_label(controller: &CompiledPlannerController) -> String {
    controller.backend_label()
}

struct PlannerCliObserver {
    action_bits: usize,
    observation_bits: usize,
    reward_bits: usize,
    reward_offset: i64,
    trace_logger: Option<AixiRunLogger>,
}

impl PlannerCliObserver {
    fn new(
        compiled: &CompiledPlannerRunSpec,
        cli_overlay: Option<&serde_json::Value>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            action_bits: planner_action_bits(compiled),
            observation_bits: compiled.interface().observation_bits,
            reward_bits: compiled.interface().reward_bits,
            reward_offset: 0,
            trace_logger: AixiRunLogger::new(cli_overlay)?,
        })
    }

    fn flush(&mut self) -> Result<(), PlannerAgentError> {
        if let Some(logger) = self.trace_logger.as_mut() {
            logger.flush().map_err(planner_observer_error)?;
        }
        Ok(())
    }
}

fn planner_observer_error(err: anyhow::Error) -> PlannerAgentError {
    PlannerAgentError::Observer {
        reason: err.to_string(),
    }
}

impl PlannerCycleObserver for PlannerCliObserver {
    fn observe_percept(
        &mut self,
        step: usize,
        observations: &[u64],
        reward: Reward,
    ) -> Result<(), PlannerAgentError> {
        if let Some(logger) = self.trace_logger.as_mut() {
            logger
                .log_percept(
                    step,
                    observations,
                    reward,
                    self.observation_bits,
                    self.reward_bits,
                    self.reward_offset,
                )
                .map_err(planner_observer_error)?;
        }
        Ok(())
    }

    fn observe_action(
        &mut self,
        step: usize,
        action: Action,
        provenance: PlannerActionProvenance,
    ) -> Result<(), PlannerAgentError> {
        if let Some(logger) = self.trace_logger.as_mut() {
            logger
                .log_action(step, action, self.action_bits, provenance)
                .map_err(planner_observer_error)?;
        }
        Ok(())
    }

    fn end_cycle(&mut self, _step: usize) -> Result<(), PlannerAgentError> {
        if let Some(logger) = self.trace_logger.as_mut() {
            logger.next_step().map_err(planner_observer_error)?;
        }
        Ok(())
    }
}

fn planner_action_bits(compiled: &CompiledPlannerRunSpec) -> usize {
    compiled.interface().agent_actions.action_bits()
}

fn print_planner_cycle_log(
    compiled: &CompiledPlannerRunSpec,
    step: usize,
    outcome: &infotheory::aixi::planner_agent::PlannerCycleOutcome,
) {
    match compiled.controller() {
        CompiledPlannerController::McAixi { .. } => {
            let obs_repr = observation_repr_from_stream(
                compiled.interface().observation_key_mode,
                &outcome.pre_observations,
                compiled.interface().observation_bits,
            );
            println!(
                "Cycle {}: Obs={:?}, Rew={}",
                step, obs_repr, outcome.pre_reward
            );
            println!("Cycle {}: Planned Action={}", step, outcome.action);
        }
        CompiledPlannerController::AiqiDiscounted { .. }
        | CompiledPlannerController::AiqiWarmstartExactJh { .. } => {
            println!(
                "Cycle {}: Action={} Obs={:?} Rew={}",
                step, outcome.action, outcome.pre_observations, outcome.pre_reward
            );
        }
        _ => {
            println!(
                "Cycle {}: Action={} Obs={:?} Rew={}",
                step, outcome.action, outcome.pre_observations, outcome.pre_reward
            );
        }
    }
}

pub(crate) fn run_vm_perf_only(
    schedule: &PlannerSchedule,
    log_every: usize,
    perf: bool,
    env: &mut PlannerEnvironment,
) -> anyhow::Result<()> {
    let mut obs = env.observations().first().copied().unwrap_or(0);
    let mut rew = env.reward();
    let start = Instant::now();
    for step in 0..schedule.learn_cycles {
        if log_every > 0 && step % log_every == 0 {
            println!("Cycle {}: Obs={}, Rew={}", step, obs, rew);
        }
        env.perform_action(0)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        obs = env.observations().first().copied().unwrap_or(0);
        rew = env.reward();
    }
    if perf && schedule.learn_cycles > 0 {
        let elapsed = start.elapsed().as_secs_f64().max(1e-9);
        let cps = schedule.learn_cycles as f64 / elapsed;
        println!("Perf cycles/s: {:.2}", cps);
    }
    Ok(())
}

pub(crate) fn run_compiled_planner_run(
    compiled: &CompiledPlannerRunSpec,
    cli_overlay: Option<&serde_json::Value>,
) -> anyhow::Result<()> {
    let runtime = compiled.runtime();
    let schedule = PlannerSchedule::from_runtime(runtime);
    let (env, env_name) = build_planner_environment(compiled)?;

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
        CompiledPlannerController::AiqiWarmstartExactJh { .. } => println!(
            "Warm-start exact-J_H AIQI initialized ({}) for {} environment.",
            controller_backend_label(compiled.controller()),
            env_name
        ),
        other => println!(
            "Planner controller '{}' initialized ({}) for {} environment.",
            other.kind_str(),
            controller_backend_label(compiled.controller()),
            env_name
        ),
    }

    if runtime.vm_perf_only {
        let mut planner_env =
            PlannerEnvironment::new(compiled, env).map_err(|err| anyhow::anyhow!("{err}"))?;
        return run_vm_perf_only(&schedule, runtime.log_every, runtime.perf, &mut planner_env);
    }

    let controller =
        PlannerControllerAgent::from_compiled(compiled).map_err(|err| anyhow::anyhow!("{err}"))?;
    let mut session = PlannerRunSession::new(compiled, controller, env)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let mut observer = PlannerCliObserver::new(compiled, cli_overlay)?;
    let learn_start = Instant::now();
    let mut eval_start: Option<Instant> = None;
    let mut learn_perf_reported = false;
    let mut learn_total_reward: i64 = 0;
    let mut eval_total_reward: i64 = 0;

    while let Some(phase) = session.next_phase() {
        if phase == PlannerPhase::Eval && !learn_perf_reported {
            if runtime.perf && schedule.learn_cycles > 0 {
                let elapsed = learn_start.elapsed().as_secs_f64().max(1e-9);
                let cps = schedule.learn_cycles as f64 / elapsed;
                println!("Learn cycles/s: {:.2}", cps);
            }
            learn_perf_reported = true;
            eval_start = Some(Instant::now());
        }

        let step = session.next_step();
        let outcome = session
            .run_next_cycle(&mut observer)
            .map_err(|err| anyhow::anyhow!("{err}"))?
            .expect("PlannerRunSession next_phase promised a next cycle");
        if runtime.log_every > 0 && step.is_multiple_of(runtime.log_every) {
            print_planner_cycle_log(compiled, step, &outcome);
        }
        match phase {
            PlannerPhase::Learn => {
                learn_total_reward = learn_total_reward.saturating_add(outcome.reward);
            }
            PlannerPhase::Eval => {
                eval_total_reward = eval_total_reward.saturating_add(outcome.reward);
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "unsupported planner phase returned by PlannerRunSession"
                ));
            }
        }
    }

    observer.flush().map_err(|err| anyhow::anyhow!("{err}"))?;

    if !learn_perf_reported && runtime.perf && schedule.learn_cycles > 0 {
        let elapsed = learn_start.elapsed().as_secs_f64().max(1e-9);
        let cps = schedule.learn_cycles as f64 / elapsed;
        println!("Learn cycles/s: {:.2}", cps);
    }

    if schedule.eval_cycles > 0 {
        if runtime.perf {
            let elapsed = eval_start
                .unwrap_or_else(Instant::now)
                .elapsed()
                .as_secs_f64()
                .max(1e-9);
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

pub(crate) fn run_aixi_mode(config_path: &str) -> anyhow::Result<()> {
    let raw = std::fs::read(config_path)?;
    let json_overlay = serde_json::from_slice::<serde_json::Value>(&raw).ok();
    if let Some(value) = json_overlay.as_ref()
        && !is_canonical_spec_document_runtime(value)
    {
        return Err(legacy_planner_config_error_runtime(config_path));
    }
    if let Some(value) = json_overlay.as_ref()
        && let Some(field) = legacy_interface_reward_range_field(value)
    {
        return Err(legacy_interface_reward_range_error(config_path, field));
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

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "backend-ctw", feature = "aixi-gameengine"))]
    use super::run_compiled_planner_run;
    use super::{
        AixiRunLogger, builtin_environment_name, is_canonical_spec_document,
        legacy_planner_config_error, run_aixi_mode,
    };
    use infotheory::aixi::planner_agent::PlannerActionProvenance;
    use infotheory::spec::BuiltinEnvironmentSpec;
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

    #[cfg(all(feature = "backend-ctw", feature = "aixi-gameengine"))]
    fn compile_planner_run_value(
        value: &serde_json::Value,
    ) -> infotheory::spec::CompiledPlannerRunSpec {
        let doc =
            infotheory::spec::SpecDocument::parse_json_value(value, std::path::Path::new("."))
                .expect("parse planner_run JSON");
        let infotheory::spec::SpecDocument::PlannerRun(spec) = doc else {
            panic!("expected planner_run document");
        };
        spec.compile().expect("compile planner_run")
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
        bits_logger
            .log_action(0, 1, 1, PlannerActionProvenance::Greedy)
            .expect("log action to bits");
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
            .log_percept(0, &[3], 1, 2, 4, 1)
            .expect("log percept to jsonl");
        jsonl_logger.next_step().expect("advance jsonl step");
        drop(jsonl_logger);
        let jsonl = std::fs::read_to_string(&jsonl_path).expect("read jsonl trace");
        assert!(jsonl.contains("\"kind\":\"percept\""));

        let _ = std::fs::remove_file(bits_path);
        let _ = std::fs::remove_file(jsonl_path);
    }

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
        logger
            .log_action(0, 1, 2, PlannerActionProvenance::Exploratory)
            .expect("log action");
        logger
            .log_percept(0, &[2], 1, 2, 4, 0)
            .expect("log percept");
        logger.next_step().expect("advance step");
        drop(logger);

        let bits = std::fs::read(&bits_path).expect("read bits trace");
        assert!(!bits.is_empty());
        assert!(bits.iter().all(|byte| *byte == 0 || *byte == 1));

        let jsonl = std::fs::read_to_string(&jsonl_path).expect("read jsonl trace");
        assert!(jsonl.contains("\"kind\":\"action\""));
        assert!(jsonl.contains("\"provenance\":\"exploratory\""));
        assert!(jsonl.contains("\"kind\":\"percept\""));

        let _ = std::fs::remove_file(bits_path);
        let _ = std::fs::remove_file(jsonl_path);
    }

    #[cfg(all(feature = "backend-ctw", feature = "aixi-gameengine"))]
    #[test]
    fn mc_aixi_jsonl_trace_can_be_converted_to_warmstart_teacher_trace() {
        use infotheory::aixi::warmstart::{
            standalone_warmstart_teacher_contract_for_compiled_planner_run,
            warmstart_teacher_trace_from_jsonl_path,
        };

        let bits_path = unique_temp_path("mc-aixi-warmstart-trace", ".bits01");
        let jsonl_path = unique_temp_path("mc-aixi-warmstart-trace", ".jsonl");
        let teacher_path = unique_temp_path("mc-aixi-warmstart-teacher", ".json");
        let mc_aixi = compile_planner_run_value(&json!({
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
                    "depth": 4
                },
                "bit_stream_semantics": { "kind": "binary_tokens" },
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
                "learn_cycles": 2,
                "eval_cycles": 0,
                "terminate_lifetime": 2,
                "log_every": 1,
                "perf": false,
                "vm_perf_only": false,
                "explore_epsilon": 0.0,
                "explore_gamma": 1.0
            }
        }));
        let overlay = json!({
            "trace_bits01_path": bits_path,
            "trace_jsonl_path": jsonl_path,
            "trace_flush_every": 1
        });
        run_compiled_planner_run(&mc_aixi, Some(&overlay))
            .expect("MC-AIXI planner run should produce JSONL");

        let warmstart_target = compile_planner_run_value(&json!({
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
                "observation_bits": 1,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "reward_bits": 1,
                "agent_actions": 2
            },
            "controller": {
                "kind": "aiqi_warmstart_exact_jh",
                "predictor": {
                    "kind": "ctw",
                    "depth": 4
                },
                "return_horizon": 2,
                "return_bins": 3,
                "label_phase_period": 2,
                "teacher_dataset_asset": "teacher",
                "planner_simulations_per_step": 1
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
        }));
        let contract =
            standalone_warmstart_teacher_contract_for_compiled_planner_run(&warmstart_target)
                .expect("build standalone teacher contract");
        let trace = warmstart_teacher_trace_from_jsonl_path(&jsonl_path, &contract, 2)
            .expect("MC-AIXI JSONL should convert to a warm-start teacher trace");
        assert_eq!(trace.transitions.len(), 2);

        let jsonl = std::fs::read_to_string(&jsonl_path).expect("read JSONL trace");
        let records = jsonl
            .lines()
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<Result<Vec<_>, _>>()
            .expect("parse JSONL records");
        assert_eq!(records.len(), 5);
        let final_record = records.last().expect("terminal percept record");
        assert_eq!(final_record["kind"].as_str(), Some("percept"));
        assert_eq!(final_record["t"].as_u64(), Some(2));

        let bits = std::fs::read(&bits_path).expect("read bits01 trace");
        assert_eq!(
            bits.len(),
            8,
            "two MC-AIXI cycles should record p0,a0,p1,a1,p2 with 1-bit actions and 2-bit percepts"
        );

        let _ = std::fs::remove_file(bits_path);
        let _ = std::fs::remove_file(jsonl_path);
        let _ = std::fs::remove_file(teacher_path);
    }

    #[cfg(all(feature = "backend-ctw", feature = "tuner"))]
    #[test]
    fn run_aixi_mode_rejects_non_planner_spec_documents() {
        use infotheory::api::CanonicalJson;
        let path = unique_temp_path("infotheory-spec-kind", ".json");
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
        let doc =
            infotheory::spec::SpecDocument::parse_json_value(&doc_value, std::path::Path::new("."))
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
        let path = unique_temp_path("infotheory-canonical-non-planner-no-ctw", ".json");
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
        use infotheory::api::CanonicalJson;
        let path = unique_temp_path("infotheory-planner-run", ".json");
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
                "bit_stream_semantics": { "kind": "binary_tokens" },
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
        let doc =
            infotheory::spec::SpecDocument::parse_json_value(&doc_value, std::path::Path::new("."))
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
        let path = unique_temp_path("infotheory-planner-run-invalid-reward", ".json");
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
                "bit_stream_semantics": { "kind": "binary_tokens" },
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
        let message = err.to_string();
        assert!(
            message.contains("unknown interface field 'min_reward'")
                || message.contains("unknown interface field 'max_reward'"),
            "{message}"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn run_aixi_mode_rejects_legacy_planner_json_documents() {
        let path = unique_temp_path("infotheory-legacy-planner", ".json");
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
