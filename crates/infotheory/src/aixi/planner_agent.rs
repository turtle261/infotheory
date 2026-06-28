//! Public planner-agent substrate for AIXI-family controllers.
//!
//! The runtime surface is intentionally limited to executable base controllers:
//! MC-AIXI, discounted AIQI, and exact-\(J_H\) warm-start AIQI. Failed
//! meta-controller experiments are not part of this module.

use crate::aixi::agent::Agent;
use crate::aixi::aiqi::AiqiAgent;
use crate::aixi::common::{
    Action, EXPLORE_RANDOM_SALT, RandomGenerator, Reward, resolve_random_seed,
};
use crate::aixi::environment::Environment;
use crate::aixi::planner_runtime::validate_environment_interface;
use crate::aixi::warmstart::{
    WarmStartExactJhAgent, WarmStartExactJhError, WarmStartExactJhTeacherDataset,
};
use crate::spec::{CompiledPlannerController, CompiledPlannerRunSpec, PlannerRuntimeSpec};
use std::error::Error;
use std::fmt;

pub use crate::aixi::planner_runtime::{
    build_planner_environment, compile_planner_run_document,
    load_warmstart_exact_jh_teacher_dataset, validate_action_alphabet,
    validate_warmstart_exact_jh_teacher_contract,
};

/// Planner cycle phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PlannerPhase {
    /// Learning/exploration phase.
    Learn,
    /// Evaluation/greedy phase.
    Eval,
}

/// Planner learn/eval schedule derived from a runtime specification.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PlannerSchedule {
    /// Number of learning cycles.
    pub learn_cycles: usize,
    /// Number of evaluation cycles.
    pub eval_cycles: usize,
    /// Extra exploration probability at learning step zero.
    pub explore_epsilon: f64,
    /// Per-step exploration decay factor.
    pub explore_gamma: f64,
}

impl PlannerSchedule {
    /// Construct a schedule with zero extra exploration.
    pub fn new(learn_cycles: usize, eval_cycles: usize) -> Self {
        Self {
            learn_cycles,
            eval_cycles,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        }
    }

    /// Derive the semantic execution schedule from a compiled runtime spec.
    pub fn from_runtime(runtime: &PlannerRuntimeSpec) -> Self {
        let terminate_lifetime: usize = runtime.terminate_lifetime;
        let (learn_cycles, eval_cycles) = match (runtime.learn_cycles, runtime.eval_cycles) {
            (Some(learn), Some(eval)) => (learn, eval),
            (Some(learn), None) => (learn, 0usize),
            (None, Some(eval)) => (terminate_lifetime, eval),
            (None, None) => (terminate_lifetime, 0usize),
        };
        Self {
            learn_cycles,
            eval_cycles,
            explore_epsilon: runtime.explore_epsilon,
            explore_gamma: runtime.explore_gamma,
        }
    }

    /// Extra exploration probability at the given global planner step.
    pub fn extra_exploration(&self, step: usize) -> f64 {
        if self.explore_epsilon > 0.0 {
            let exponent = i32::try_from(step).unwrap_or(i32::MAX);
            (self.explore_epsilon * self.explore_gamma.powi(exponent)).min(1.0)
        } else {
            0.0
        }
    }
}

/// Stable action-provenance labels used by planner JSONL telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PlannerActionProvenance {
    /// Greedy controller action.
    Greedy,
    /// Exploratory controller action.
    Exploratory,
}

impl PlannerActionProvenance {
    /// Stable JSONL string representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Exploratory => "exploratory",
        }
    }

    /// Parse a normative JSONL provenance string.
    pub fn from_jsonl_str(value: &str) -> Result<Self, WarmStartExactJhError> {
        match value {
            "greedy" => Ok(Self::Greedy),
            "exploratory" => Ok(Self::Exploratory),
            other => Err(WarmStartExactJhError::InvalidTelemetry {
                reason: format!("unknown action provenance '{other}'"),
            }),
        }
    }
}

/// Outcome of one planner-environment cycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannerCycleOutcome {
    /// Observation stream visible at the decision point.
    pub pre_observations: Vec<u64>,
    /// Reward visible at the decision point.
    pub pre_reward: Reward,
    /// Selected action.
    pub action: Action,
    /// Post-action observation stream.
    pub observations: Vec<u64>,
    /// Post-action reward.
    pub reward: Reward,
    /// Provenance for the selected action.
    pub provenance: PlannerActionProvenance,
}

/// Observer strategy for ordered planner-cycle telemetry.
///
/// Implementations receive a controller's native event ordering, which is one of:
/// - *decision-percept*: the percept for step `t`, then action `t`; after the
///   final scheduled action, one terminal successor percept at step
///   `N = learn_cycles + eval_cycles`.
/// - *action-then-post-percept*: action `t`, then the reached percept, both at
///   step `t`, with no separate terminal percept.
///
/// Observers must tolerate either ordering and a percept reported at `step == N`.
/// Each concrete controller documents which ordering it produces.
pub trait PlannerCycleObserver {
    /// Observe a percept event for `step`.
    ///
    /// Under the decision-percept ordering, `step` may equal the total number of
    /// scheduled cycles `N` for the terminal successor percept; observers must
    /// accept this one-past-the-last-cycle index.
    fn observe_percept(
        &mut self,
        step: usize,
        observations: &[u64],
        reward: Reward,
    ) -> Result<(), PlannerAgentError>;

    /// Observe an action event for `step`.
    fn observe_action(
        &mut self,
        step: usize,
        action: Action,
        provenance: PlannerActionProvenance,
    ) -> Result<(), PlannerAgentError>;

    /// Mark the end of one complete cycle.
    fn end_cycle(&mut self, step: usize) -> Result<(), PlannerAgentError>;
}

/// No-op observer for executions that do not need telemetry.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullPlannerObserver;

impl PlannerCycleObserver for NullPlannerObserver {
    fn observe_percept(
        &mut self,
        _step: usize,
        _observations: &[u64],
        _reward: Reward,
    ) -> Result<(), PlannerAgentError> {
        Ok(())
    }

    fn observe_action(
        &mut self,
        _step: usize,
        _action: Action,
        _provenance: PlannerActionProvenance,
    ) -> Result<(), PlannerAgentError> {
        Ok(())
    }

    fn end_cycle(&mut self, _step: usize) -> Result<(), PlannerAgentError> {
        Ok(())
    }
}

impl PlannerCycleObserver for () {
    fn observe_percept(
        &mut self,
        _step: usize,
        _observations: &[u64],
        _reward: Reward,
    ) -> Result<(), PlannerAgentError> {
        Ok(())
    }

    fn observe_action(
        &mut self,
        _step: usize,
        _action: Action,
        _provenance: PlannerActionProvenance,
    ) -> Result<(), PlannerAgentError> {
        Ok(())
    }

    fn end_cycle(&mut self, _step: usize) -> Result<(), PlannerAgentError> {
        Ok(())
    }
}

/// Validate observation stream length.
pub fn validate_obs_stream_len(expected: usize, actual: usize) -> Result<(), PlannerAgentError> {
    if actual != expected {
        return Err(PlannerAgentError::EnvironmentInterface {
            reason: format!(
                "observation stream length mismatch: expected {expected}, got {actual}"
            ),
        });
    }
    Ok(())
}

/// Runtime environment state for planner episodes.
pub struct PlannerEnvironment {
    env: Box<dyn Environment>,
    observation_stream_len: usize,
    observations: Vec<u64>,
    reward: Reward,
}

impl PlannerEnvironment {
    /// Construct a validated planner environment state.
    pub fn new(
        compiled: &CompiledPlannerRunSpec,
        env: Box<dyn Environment>,
    ) -> Result<Self, PlannerAgentError> {
        Self::new_with_seed(
            compiled,
            env,
            resolve_random_seed(compiled.runtime().random_seed),
        )
    }

    /// Construct a validated planner environment state with an explicit seed.
    pub fn new_with_seed(
        compiled: &CompiledPlannerRunSpec,
        mut env: Box<dyn Environment>,
        random_seed: u64,
    ) -> Result<Self, PlannerAgentError> {
        validate_environment_interface(compiled, env.as_ref()).map_err(|err| {
            PlannerAgentError::EnvironmentInterface {
                reason: err.to_string(),
            }
        })?;
        env.set_random_seed(random_seed);
        let observation_stream_len: usize = compiled.interface().observation_stream_len;
        let observations = env.drain_observations();
        validate_obs_stream_len(observation_stream_len, observations.len())?;
        let reward: Reward = env.get_reward();
        Ok(Self {
            env,
            observation_stream_len,
            observations,
            reward,
        })
    }

    /// Current observation stream.
    pub fn observations(&self) -> &[u64] {
        &self.observations
    }

    /// Current reward.
    pub fn reward(&self) -> Reward {
        self.reward
    }

    /// Perform one action and update the current percept state.
    pub fn perform_action(&mut self, action: Action) -> Result<Reward, PlannerAgentError> {
        self.env.perform_action(action);
        self.observations = self.env.drain_observations();
        validate_obs_stream_len(self.observation_stream_len, self.observations.len())?;
        self.reward = self.env.get_reward();
        Ok(self.reward)
    }
}

/// Factory for constructing fresh environments for repeated episodes.
pub trait EnvironmentFactory {
    /// Build a fresh environment.
    fn build(&self) -> Result<Box<dyn Environment>, PlannerAgentError>;
}

/// Executable planner-agent abstraction.
pub trait PlannerAgent {
    /// Reseed controller-side stochastic state.
    ///
    /// This does not clear learned model state or retained history.
    fn reseed_for_episode(&mut self, _random_seed: u64) {}

    /// Start a fresh environment episode while preserving learned model state.
    ///
    /// Implementations should reset episode-local transient state such as a
    /// previous-action pointer or retained search tree. They should not discard
    /// learned predictor state unless the concrete controller documents that
    /// policy separately.
    fn reset_for_episode(&mut self, random_seed: u64) {
        self.reseed_for_episode(random_seed);
    }

    /// Execute one planner-environment cycle.
    fn run_cycle(
        &mut self,
        phase: PlannerPhase,
        step: usize,
        schedule: &PlannerSchedule,
        env: &mut PlannerEnvironment,
        observer: &mut dyn PlannerCycleObserver,
    ) -> Result<PlannerCycleOutcome, PlannerAgentError>;
}

/// MC-AIXI planner adapter.
///
/// Produces the decision-percept ordering (see [`PlannerCycleObserver`]).
pub struct McAixiPlannerAgent {
    agent: Agent,
    prev_action: Action,
    explore_rng: RandomGenerator,
}

impl McAixiPlannerAgent {
    /// Construct an MC-AIXI planner from a compiled planner-run spec.
    pub fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, PlannerAgentError> {
        let agent = Agent::from_compiled_planner_run(compiled)
            .map_err(|err| PlannerAgentError::McAixi(err.to_string()))?;
        let explore_rng: RandomGenerator =
            RandomGenerator::from_seed(resolve_random_seed(compiled.runtime().random_seed))
                .fork_with(EXPLORE_RANDOM_SALT);
        Ok(Self {
            agent,
            prev_action: 0,
            explore_rng,
        })
    }
}

impl PlannerAgent for McAixiPlannerAgent {
    fn reseed_for_episode(&mut self, random_seed: u64) {
        self.agent.reseed_random(random_seed);
        self.explore_rng = RandomGenerator::from_seed(random_seed).fork_with(EXPLORE_RANDOM_SALT);
    }

    fn reset_for_episode(&mut self, random_seed: u64) {
        self.reseed_for_episode(random_seed);
        self.prev_action = 0;
        self.agent.reset_planner_state();
    }

    fn run_cycle(
        &mut self,
        phase: PlannerPhase,
        step: usize,
        schedule: &PlannerSchedule,
        env: &mut PlannerEnvironment,
        observer: &mut dyn PlannerCycleObserver,
    ) -> Result<PlannerCycleOutcome, PlannerAgentError> {
        let pre_observations: Vec<u64> = env.observations().to_vec();
        let pre_reward: Reward = env.reward();
        observer.observe_percept(step, &pre_observations, pre_reward)?;
        self.agent
            .model_update_percept_stream(&pre_observations, pre_reward);
        let mut provenance = PlannerActionProvenance::Greedy;
        let action: Action = match phase {
            PlannerPhase::Learn => {
                let explore_p: f64 = schedule.extra_exploration(step);
                if explore_p > 0.0 && self.explore_rng.gen_bool(explore_p) {
                    provenance = PlannerActionProvenance::Exploratory;
                    self.explore_rng.gen_range(env.env.get_num_actions().get()) as u64
                } else {
                    self.agent
                        .get_planned_action(&pre_observations, pre_reward, self.prev_action)
                }
            }
            PlannerPhase::Eval => {
                self.agent
                    .get_planned_action(&pre_observations, pre_reward, self.prev_action)
            }
        };
        observer.observe_action(step, action, provenance)?;
        self.agent.model_update_action_external(action);
        let reward: Reward = env.perform_action(action)?;
        self.prev_action = action;
        observer.end_cycle(step)?;
        // The last post-action percept has no following decision cycle to emit it.
        let total_cycles: usize = schedule.learn_cycles.saturating_add(schedule.eval_cycles);
        if step.checked_add(1) == Some(total_cycles) {
            observer.observe_percept(total_cycles, env.observations(), reward)?;
        }
        Ok(PlannerCycleOutcome {
            pre_observations,
            pre_reward,
            action,
            observations: env.observations().to_vec(),
            reward,
            provenance,
        })
    }
}

/// Discounted AIQI planner adapter.
///
/// Produces the action-then-post-percept ordering (see [`PlannerCycleObserver`]).
pub struct AiqiDiscountedPlannerAgent {
    agent: AiqiAgent,
}

impl AiqiDiscountedPlannerAgent {
    /// Construct a discounted-AIQI planner from a compiled planner-run spec.
    pub fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, PlannerAgentError> {
        Ok(Self {
            agent: AiqiAgent::from_compiled_planner_run(compiled)
                .map_err(PlannerAgentError::Aiqi)?,
        })
    }
}

impl PlannerAgent for AiqiDiscountedPlannerAgent {
    fn reseed_for_episode(&mut self, random_seed: u64) {
        self.agent.reseed_random(random_seed);
    }

    fn run_cycle(
        &mut self,
        phase: PlannerPhase,
        step: usize,
        schedule: &PlannerSchedule,
        env: &mut PlannerEnvironment,
        observer: &mut dyn PlannerCycleObserver,
    ) -> Result<PlannerCycleOutcome, PlannerAgentError> {
        let pre_observations: Vec<u64> = env.observations().to_vec();
        let pre_reward: Reward = env.reward();
        let (action, explored) = match phase {
            PlannerPhase::Learn => self
                .agent
                .get_planned_action_with_extra_exploration_flag(schedule.extra_exploration(step)),
            PlannerPhase::Eval => (self.agent.get_planned_action(), false),
        };
        let provenance = if explored {
            PlannerActionProvenance::Exploratory
        } else {
            PlannerActionProvenance::Greedy
        };
        observer.observe_action(step, action, provenance)?;
        let reward: Reward = env.perform_action(action)?;
        observer.observe_percept(step, env.observations(), reward)?;
        self.agent
            .observe_transition(action, env.observations(), reward)
            .map_err(PlannerAgentError::Aiqi)?;
        observer.end_cycle(step)?;
        Ok(PlannerCycleOutcome {
            pre_observations,
            pre_reward,
            action,
            observations: env.observations().to_vec(),
            reward,
            provenance,
        })
    }
}

/// Exact-\(J_H\) warm-start planner adapter.
///
/// Produces the action-then-post-percept ordering (see [`PlannerCycleObserver`]).
pub struct WarmStartExactJhPlannerAgent {
    agent: WarmStartExactJhAgent,
}

impl WarmStartExactJhPlannerAgent {
    /// Construct a warm-start planner from a compiled planner-run spec and teacher data.
    pub fn from_compiled(
        compiled: &CompiledPlannerRunSpec,
        teacher: WarmStartExactJhTeacherDataset,
    ) -> Result<Self, PlannerAgentError> {
        Ok(Self {
            agent: WarmStartExactJhAgent::from_compiled_planner_run(compiled, teacher)
                .map_err(PlannerAgentError::WarmStart)?,
        })
    }
}

impl PlannerAgent for WarmStartExactJhPlannerAgent {
    fn reseed_for_episode(&mut self, random_seed: u64) {
        self.agent.reseed_random(random_seed);
    }

    fn run_cycle(
        &mut self,
        phase: PlannerPhase,
        step: usize,
        schedule: &PlannerSchedule,
        env: &mut PlannerEnvironment,
        observer: &mut dyn PlannerCycleObserver,
    ) -> Result<PlannerCycleOutcome, PlannerAgentError> {
        let pre_observations: Vec<u64> = env.observations().to_vec();
        let pre_reward: Reward = env.reward();
        let (action, explored) = match phase {
            PlannerPhase::Learn => self
                .agent
                .try_get_planned_action_with_extra_exploration_flag(
                    schedule.extra_exploration(step),
                )
                .map_err(PlannerAgentError::WarmStart)?,
            PlannerPhase::Eval => (
                self.agent
                    .try_get_planned_action()
                    .map_err(PlannerAgentError::WarmStart)?,
                false,
            ),
        };
        let provenance = if explored {
            PlannerActionProvenance::Exploratory
        } else {
            PlannerActionProvenance::Greedy
        };
        observer.observe_action(step, action, provenance)?;
        let reward: Reward = env.perform_action(action)?;
        observer.observe_percept(step, env.observations(), reward)?;
        self.agent
            .observe_transition(action, env.observations(), reward)
            .map_err(PlannerAgentError::WarmStart)?;
        observer.end_cycle(step)?;
        Ok(PlannerCycleOutcome {
            pre_observations,
            pre_reward,
            action,
            observations: env.observations().to_vec(),
            reward,
            provenance,
        })
    }
}

enum PlannerControllerAgentKind {
    McAixi(McAixiPlannerAgent),
    AiqiDiscounted(AiqiDiscountedPlannerAgent),
    WarmStartExactJh(WarmStartExactJhPlannerAgent),
}

/// Runtime controller selected from a compiled `planner_run`.
pub struct PlannerControllerAgent {
    inner: PlannerControllerAgentKind,
}

impl PlannerControllerAgent {
    /// Construct the executable base controller declared by `compiled`.
    pub fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, PlannerAgentError> {
        let inner = match compiled.controller() {
            CompiledPlannerController::McAixi { .. } => {
                PlannerControllerAgentKind::McAixi(McAixiPlannerAgent::from_compiled(compiled)?)
            }
            CompiledPlannerController::AiqiDiscounted { .. } => {
                PlannerControllerAgentKind::AiqiDiscounted(
                    AiqiDiscountedPlannerAgent::from_compiled(compiled)?,
                )
            }
            CompiledPlannerController::AiqiWarmstartExactJh {
                teacher_dataset_asset,
                ..
            } => {
                let teacher =
                    load_warmstart_exact_jh_teacher_dataset(compiled, teacher_dataset_asset)?;
                PlannerControllerAgentKind::WarmStartExactJh(
                    WarmStartExactJhPlannerAgent::from_compiled(compiled, teacher)?,
                )
            }
        };
        Ok(Self { inner })
    }

    /// Canonical controller kind label.
    pub fn controller_kind(&self) -> &'static str {
        match &self.inner {
            PlannerControllerAgentKind::McAixi(_) => "mc_aixi",
            PlannerControllerAgentKind::AiqiDiscounted(_) => "aiqi_discounted",
            PlannerControllerAgentKind::WarmStartExactJh(_) => "aiqi_warmstart_exact_jh",
        }
    }
}

impl PlannerAgent for PlannerControllerAgent {
    fn reseed_for_episode(&mut self, random_seed: u64) {
        match &mut self.inner {
            PlannerControllerAgentKind::McAixi(agent) => agent.reseed_for_episode(random_seed),
            PlannerControllerAgentKind::AiqiDiscounted(agent) => {
                agent.reseed_for_episode(random_seed);
            }
            PlannerControllerAgentKind::WarmStartExactJh(agent) => {
                agent.reseed_for_episode(random_seed);
            }
        }
    }

    fn reset_for_episode(&mut self, random_seed: u64) {
        match &mut self.inner {
            PlannerControllerAgentKind::McAixi(agent) => agent.reset_for_episode(random_seed),
            PlannerControllerAgentKind::AiqiDiscounted(agent) => {
                agent.reset_for_episode(random_seed);
            }
            PlannerControllerAgentKind::WarmStartExactJh(agent) => {
                agent.reset_for_episode(random_seed);
            }
        }
    }

    fn run_cycle(
        &mut self,
        phase: PlannerPhase,
        step: usize,
        schedule: &PlannerSchedule,
        env: &mut PlannerEnvironment,
        observer: &mut dyn PlannerCycleObserver,
    ) -> Result<PlannerCycleOutcome, PlannerAgentError> {
        match &mut self.inner {
            PlannerControllerAgentKind::McAixi(agent) => {
                agent.run_cycle(phase, step, schedule, env, observer)
            }
            PlannerControllerAgentKind::AiqiDiscounted(agent) => {
                agent.run_cycle(phase, step, schedule, env, observer)
            }
            PlannerControllerAgentKind::WarmStartExactJh(agent) => {
                agent.run_cycle(phase, step, schedule, env, observer)
            }
        }
    }
}

/// Executable planner-run session.
pub struct PlannerRunSession {
    agent: PlannerControllerAgent,
    environment: PlannerEnvironment,
    schedule: PlannerSchedule,
    next_step: usize,
}

impl PlannerRunSession {
    /// Construct a planner-run session from executable components.
    pub fn new(
        compiled: &CompiledPlannerRunSpec,
        mut agent: PlannerControllerAgent,
        env: Box<dyn Environment>,
    ) -> Result<Self, PlannerAgentError> {
        let environment = PlannerEnvironment::new(compiled, env)?;
        let schedule = PlannerSchedule::from_runtime(compiled.runtime());
        agent.reset_for_episode(resolve_random_seed(compiled.runtime().random_seed));
        Ok(Self {
            agent,
            environment,
            schedule,
            next_step: 0,
        })
    }

    /// Execution schedule for this session.
    pub fn schedule(&self) -> &PlannerSchedule {
        &self.schedule
    }

    /// Number of cycles already executed.
    pub fn next_step(&self) -> usize {
        self.next_step
    }

    /// Phase of the next scheduled cycle, or `None` when the session is complete.
    pub fn next_phase(&self) -> Option<PlannerPhase> {
        let total_cycles: usize = self
            .schedule
            .learn_cycles
            .saturating_add(self.schedule.eval_cycles);
        if self.next_step >= total_cycles {
            return None;
        }
        Some(if self.next_step < self.schedule.learn_cycles {
            PlannerPhase::Learn
        } else {
            PlannerPhase::Eval
        })
    }

    /// Whether all scheduled cycles have been executed.
    pub fn is_finished(&self) -> bool {
        self.next_phase().is_none()
    }

    /// Run the next scheduled cycle.
    pub fn run_next_cycle(
        &mut self,
        observer: &mut dyn PlannerCycleObserver,
    ) -> Result<Option<PlannerCycleOutcome>, PlannerAgentError> {
        let Some(phase) = self.next_phase() else {
            return Ok(None);
        };
        let outcome = self.agent.run_cycle(
            phase,
            self.next_step,
            &self.schedule,
            &mut self.environment,
            observer,
        )?;
        self.next_step = self.next_step.saturating_add(1);
        Ok(Some(outcome))
    }
}

/// Summary returned by [`run_episode`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlannerRunReport {
    /// Sum of rewards over learning cycles.
    pub learn_total_reward: Reward,
    /// Sum of rewards over evaluation cycles.
    pub eval_total_reward: Reward,
    /// Number of learning cycles executed.
    pub learn_cycles: usize,
    /// Number of evaluation cycles executed.
    pub eval_cycles: usize,
}

/// Run one complete environment episode with a supplied agent and environment factory.
///
/// The agent's learned model state is preserved across calls. Before the fresh
/// environment is used, the agent receives an episode-boundary reset so
/// controller-side randomness and episode-local transient state are aligned with
/// `random_seed`.
pub fn run_episode(
    agent: &mut dyn PlannerAgent,
    schedule: &PlannerSchedule,
    env_factory: &dyn EnvironmentFactory,
    random_seed: u64,
    compiled: &CompiledPlannerRunSpec,
) -> Result<PlannerRunReport, PlannerAgentError> {
    let env = env_factory.build()?;
    let mut environment = PlannerEnvironment::new_with_seed(compiled, env, random_seed)?;
    agent.reset_for_episode(random_seed);
    let mut observer = NullPlannerObserver;
    let mut learn_total_reward: Reward = 0;
    let mut eval_total_reward: Reward = 0;
    for step in 0..schedule.learn_cycles {
        let outcome = agent.run_cycle(
            PlannerPhase::Learn,
            step,
            schedule,
            &mut environment,
            &mut observer,
        )?;
        learn_total_reward = learn_total_reward.saturating_add(outcome.reward);
    }
    for offset in 0..schedule.eval_cycles {
        let step: usize = schedule.learn_cycles + offset;
        let outcome = agent.run_cycle(
            PlannerPhase::Eval,
            step,
            schedule,
            &mut environment,
            &mut observer,
        )?;
        eval_total_reward = eval_total_reward.saturating_add(outcome.reward);
    }
    Ok(PlannerRunReport {
        learn_total_reward,
        eval_total_reward,
        learn_cycles: schedule.learn_cycles,
        eval_cycles: schedule.eval_cycles,
    })
}

/// Planner-agent runtime error.
#[derive(Debug)]
#[non_exhaustive]
pub enum PlannerAgentError {
    /// MC-AIXI construction or execution error.
    McAixi(String),
    /// AIQI construction or execution error.
    Aiqi(crate::aixi::aiqi::AiqiError),
    /// Warm-start construction or execution error.
    WarmStart(WarmStartExactJhError),
    /// Environment interface mismatch.
    EnvironmentInterface {
        /// Human-readable reason.
        reason: String,
    },
    /// Environment construction failed.
    Environment {
        /// Human-readable reason.
        reason: String,
    },
    /// Observer or telemetry sink failed during cycle execution.
    Observer {
        /// Human-readable reason.
        reason: String,
    },
}

impl fmt::Display for PlannerAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::McAixi(err) => write!(f, "{err}"),
            Self::Aiqi(err) => write!(f, "{err}"),
            Self::WarmStart(err) => write!(f, "{err}"),
            Self::EnvironmentInterface { reason }
            | Self::Environment { reason }
            | Self::Observer { reason } => f.write_str(reason),
        }
    }
}

impl Error for PlannerAgentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Aiqi(err) => Some(err),
            Self::WarmStart(err) => Some(err),
            Self::McAixi(_)
            | Self::EnvironmentInterface { .. }
            | Self::Environment { .. }
            | Self::Observer { .. } => None,
        }
    }
}

#[cfg(all(test, feature = "backend-ctw"))]
mod tests {
    use super::*;
    use crate::aixi::common::{ActionAlphabet, PerceptVal};
    use crate::aixi::warmstart::{
        WarmStartExactJhTeacherDataset, WarmStartExactJhTeacherTrace, WarmStartExactJhTransition,
        standalone_warmstart_teacher_contract_for_compiled_planner_run,
    };
    use crate::spec::{SpecDocument, SpecEnvironment};
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
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

    fn action_alphabet(n: usize) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
    }

    #[test]
    fn extra_exploration_decay_does_not_wrap_after_i32_limit() {
        let schedule = PlannerSchedule {
            learn_cycles: 0,
            eval_cycles: 0,
            explore_epsilon: 0.5,
            explore_gamma: 0.5,
        };

        assert_eq!(schedule.extra_exploration(i32::MAX as usize + 1), 0.0);
    }

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
        spec.compile_in(&SpecEnvironment::new(Path::new(".")))
            .expect("sample warmstart planner run should compile")
    }

    fn write_matching_warmstart_teacher(path: &Path, compiled: &CompiledPlannerRunSpec) {
        let contract = standalone_warmstart_teacher_contract_for_compiled_planner_run(compiled)
            .expect("standalone warmstart teacher contract");
        let dataset = WarmStartExactJhTeacherDataset::new(
            contract,
            vec![WarmStartExactJhTeacherTrace::new(vec![
                WarmStartExactJhTransition::new(0, vec![1], 1),
            ])],
        );
        std::fs::write(
            path,
            serde_json::to_vec(&dataset.to_json_value()).expect("teacher JSON"),
        )
        .expect("write teacher dataset");
    }

    #[derive(Clone, Copy)]
    struct CountingEnv {
        observation: PerceptVal,
        reward: Reward,
    }

    impl Environment for CountingEnv {
        fn perform_action(&mut self, action: Action) {
            self.observation = (self.observation + action + 1) & 0b11;
            self.reward = (self.reward + 1).min(1);
        }

        fn get_observation(&self) -> PerceptVal {
            self.observation
        }

        fn get_reward(&self) -> Reward {
            self.reward
        }

        fn is_finished(&self) -> bool {
            false
        }

        fn get_observation_bits(&self) -> usize {
            2
        }

        fn get_reward_bits(&self) -> usize {
            2
        }

        fn get_action_bits(&self) -> usize {
            1
        }
    }

    struct SeedRecordingEnv {
        recorded_seed: Arc<AtomicU64>,
    }

    impl Environment for SeedRecordingEnv {
        fn perform_action(&mut self, _action: Action) {}

        fn get_observation(&self) -> PerceptVal {
            0
        }

        fn get_reward(&self) -> Reward {
            0
        }

        fn is_finished(&self) -> bool {
            false
        }

        fn get_observation_bits(&self) -> usize {
            2
        }

        fn get_reward_bits(&self) -> usize {
            2
        }

        fn get_action_bits(&self) -> usize {
            1
        }

        fn set_random_seed(&mut self, seed: u64) {
            self.recorded_seed.store(seed, Ordering::SeqCst);
        }
    }

    struct SeedRecordingFactory {
        recorded_seed: Arc<AtomicU64>,
    }

    impl EnvironmentFactory for SeedRecordingFactory {
        fn build(&self) -> Result<Box<dyn Environment>, PlannerAgentError> {
            Ok(Box::new(SeedRecordingEnv {
                recorded_seed: Arc::clone(&self.recorded_seed),
            }))
        }
    }

    struct SeedRecordingAgent {
        recorded_seed: Arc<AtomicU64>,
        reset_calls: Arc<AtomicU64>,
    }

    impl PlannerAgent for SeedRecordingAgent {
        fn reseed_for_episode(&mut self, random_seed: u64) {
            self.recorded_seed.store(random_seed, Ordering::SeqCst);
        }

        fn reset_for_episode(&mut self, random_seed: u64) {
            self.recorded_seed.store(random_seed, Ordering::SeqCst);
            self.reset_calls.fetch_add(1, Ordering::SeqCst);
        }

        fn run_cycle(
            &mut self,
            phase: PlannerPhase,
            step: usize,
            _schedule: &PlannerSchedule,
            env: &mut PlannerEnvironment,
            observer: &mut dyn PlannerCycleObserver,
        ) -> Result<PlannerCycleOutcome, PlannerAgentError> {
            let pre_observations = env.observations().to_vec();
            let pre_reward = env.reward();
            observer.observe_action(step, 0, PlannerActionProvenance::Greedy)?;
            let reward = env.perform_action(0)?;
            observer.observe_percept(step, env.observations(), reward)?;
            observer.end_cycle(step)?;
            assert_eq!(phase, PlannerPhase::Learn);
            Ok(PlannerCycleOutcome {
                pre_observations,
                pre_reward,
                action: 0,
                observations: env.observations().to_vec(),
                reward,
                provenance: PlannerActionProvenance::Greedy,
            })
        }
    }

    #[test]
    fn run_episode_uses_explicit_seed_for_environment_and_agent() {
        let compiled = sample_warmstart_compiled_planner_run(Path::new("teacher.json"));
        let env_seed = Arc::new(AtomicU64::new(u64::MAX));
        let agent_seed = Arc::new(AtomicU64::new(u64::MAX));
        let reset_calls = Arc::new(AtomicU64::new(0));
        let factory = SeedRecordingFactory {
            recorded_seed: Arc::clone(&env_seed),
        };
        let mut agent = SeedRecordingAgent {
            recorded_seed: Arc::clone(&agent_seed),
            reset_calls: Arc::clone(&reset_calls),
        };

        let report = run_episode(
            &mut agent,
            &PlannerSchedule::new(1, 0),
            &factory,
            99,
            &compiled,
        )
        .expect("seeded episode should run");

        assert_eq!(report.learn_cycles, 1);
        assert_eq!(env_seed.load(Ordering::SeqCst), 99);
        assert_eq!(agent_seed.load(Ordering::SeqCst), 99);
        assert_eq!(reset_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn planner_run_session_executes_warmstart_controller_from_compiled_spec() {
        let teacher_path = unique_temp_path("planner-agent-warmstart-teacher", ".json");
        let compiled = sample_warmstart_compiled_planner_run(&teacher_path);
        write_matching_warmstart_teacher(&teacher_path, &compiled);

        let controller =
            PlannerControllerAgent::from_compiled(&compiled).expect("warmstart controller");
        assert_eq!(controller.controller_kind(), "aiqi_warmstart_exact_jh");
        let env = Box::new(CountingEnv {
            observation: 1,
            reward: 0,
        });
        let mut session =
            PlannerRunSession::new(&compiled, controller, env).expect("planner session");
        assert_eq!(session.schedule().learn_cycles, 1);
        assert_eq!(session.schedule().eval_cycles, 1);
        assert_eq!(session.next_phase(), Some(PlannerPhase::Learn));

        let mut observer = NullPlannerObserver;
        let learn = session
            .run_next_cycle(&mut observer)
            .expect("learn cycle")
            .expect("learn outcome");
        assert_eq!(learn.pre_observations, vec![1]);
        assert!(learn.action < 2);
        assert_eq!(session.next_phase(), Some(PlannerPhase::Eval));

        let eval = session
            .run_next_cycle(&mut observer)
            .expect("eval cycle")
            .expect("eval outcome");
        assert!(eval.action < 2);
        assert!(session.is_finished());
        assert!(
            session
                .run_next_cycle(&mut observer)
                .expect("complete session")
                .is_none()
        );

        let _ = std::fs::remove_file(teacher_path);
    }
}

impl From<anyhow::Error> for PlannerAgentError {
    fn from(value: anyhow::Error) -> Self {
        Self::Environment {
            reason: value.to_string(),
        }
    }
}
