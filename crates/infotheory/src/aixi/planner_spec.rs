//! Shared planner-run spec builder utilities for AIXI/AIQI controllers.

use crate::aixi::common::{ObservationKeyMode, resolve_random_seed};
use crate::spec::{
    BuiltinEnvironmentSpec, ControllerSpec, EnvironmentSpec, PlannerInterfaceSpec, PlannerRunSpec,
    PlannerRuntimeSpec,
};

/// Canonical planner/environment interface settings shared by AIXI-family builders.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlannerInterfaceConfig {
    pub observation_bits: usize,
    pub observation_stream_len: usize,
    pub observation_key_mode: ObservationKeyMode,
    pub reward_bits: usize,
    pub agent_actions: usize,
    pub min_reward: i64,
    pub max_reward: i64,
    pub reward_offset: i64,
}

impl PlannerInterfaceConfig {
    fn into_spec(self) -> PlannerInterfaceSpec {
        PlannerInterfaceSpec {
            observation_bits: self.observation_bits,
            observation_stream_len: self.observation_stream_len.max(1),
            observation_key_mode: self.observation_key_mode,
            reward_bits: self.reward_bits,
            agent_actions: self.agent_actions,
            min_reward: self.min_reward,
            max_reward: self.max_reward,
            reward_offset: self.reward_offset,
        }
    }
}

/// Build the canonical planner-run spec used by AIXI-family agents.
///
/// The builtin environment is a minimal placeholder for programmatic
/// `AgentConfig`/`AiqiConfig` construction: these callers provide the actual
/// environment object at execution time, while the interface section below is
/// the authoritative contract the agent uses.
pub(crate) fn build_default_planner_run_spec(
    interface: PlannerInterfaceConfig,
    controller: ControllerSpec,
    random_seed: Option<u64>,
) -> PlannerRunSpec {
    PlannerRunSpec {
        assets: Vec::new(),
        environment: EnvironmentSpec::Builtin {
            builtin: BuiltinEnvironmentSpec::CoinFlip,
        },
        interface: interface.into_spec(),
        controller,
        runtime: PlannerRuntimeSpec {
            random_seed: Some(resolve_random_seed(random_seed)),
            learn_cycles: None,
            eval_cycles: None,
            terminate_lifetime: 1,
            log_every: 1,
            perf: false,
            vm_perf_only: false,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        },
    }
}
