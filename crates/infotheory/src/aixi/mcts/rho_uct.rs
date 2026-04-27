use super::{
    AgentSimulator, PerceptMap, PerceptOutcome, best_action_from_action_values,
    choose_uniform_unvisited, ensure_action_slots, prune_key, random_rollout,
};
use crate::aixi::common::{Action, PerceptVal, Reward};

/// Sequential `rho_uct` planner state.
///
/// The first-visit decision-node shortcut intentionally follows
/// `aixictwx.tex` Algorithm 2: when `T(h) = 0`, the planner performs a
/// rollout from `h` rather than forcing an action/chance expansion first.
pub struct RhoUctPlanner {
    root: Option<RhoUctNode>,
}

impl RhoUctPlanner {
    pub fn new() -> Self {
        Self {
            root: Some(RhoUctNode::new(false)),
        }
    }

    pub fn search(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: Action,
        samples: usize,
    ) -> Action {
        self.prune_tree(agent, prev_obs_stream, prev_rew, prev_act);

        let horizon = agent.horizon();
        let root = self.root.as_mut().expect("rho_uct root missing");
        for _ in 0..samples {
            agent.begin_simulation();
            root.sample(agent, horizon, horizon);
        }
        root.best_action(agent)
    }

    fn prune_tree(
        &mut self,
        agent: &dyn AgentSimulator,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: Action,
    ) {
        let Some(mut old_root) = self.root.take() else {
            self.root = Some(RhoUctNode::new(false));
            return;
        };

        let action_child = old_root
            .action_children
            .get_mut(prev_act as usize)
            .and_then(Option::take);

        let Some(mut chance_child) = action_child else {
            self.root = Some(RhoUctNode::new(false));
            return;
        };

        let key = prune_key(agent, prev_obs_stream, prev_rew);
        self.root = chance_child
            .percept_children
            .remove(&key)
            .or_else(|| Some(RhoUctNode::new(false)));
    }
}

impl Default for RhoUctPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct RhoUctNode {
    visits: u32,
    mean: f64,
    is_chance_node: bool,
    action_children: Vec<Option<RhoUctNode>>,
    percept_children: PerceptMap<RhoUctNode>,
}

impl RhoUctNode {
    fn new(is_chance_node: bool) -> Self {
        Self {
            visits: 0,
            mean: 0.0,
            is_chance_node,
            action_children: Vec::new(),
            percept_children: PerceptMap::default(),
        }
    }

    fn best_action(&self, agent: &mut dyn AgentSimulator) -> Action {
        best_action_from_action_values(
            self.action_children
                .iter()
                .enumerate()
                .filter_map(|(action_idx, child)| {
                    child.as_ref().map(|node| (action_idx, node.mean))
                }),
            agent.get_num_actions(),
            agent,
        )
    }

    fn sample(
        &mut self,
        agent: &mut dyn AgentSimulator,
        remaining_horizon: usize,
        total_horizon: usize,
    ) -> f64 {
        if remaining_horizon == 0 {
            agent.model_revert(total_horizon);
            return 0.0;
        }

        let reward = if self.is_chance_node {
            let (observations, immediate_reward) = agent.gen_percepts_and_update();
            let key = PerceptOutcome::new(observations, immediate_reward);
            let child = self
                .percept_children
                .entry(key)
                .or_insert_with(|| RhoUctNode::new(false));
            (immediate_reward as f64)
                + agent.discount_gamma() * child.sample(agent, remaining_horizon - 1, total_horizon)
        } else if self.visits == 0 {
            let reward = random_rollout(agent, remaining_horizon);
            agent.model_revert(total_horizon);
            reward
        } else {
            let (child, _action) = self.select_action(agent, remaining_horizon);
            child.sample(agent, remaining_horizon, total_horizon)
        };

        self.mean = (reward + (self.visits as f64) * self.mean) / ((self.visits + 1) as f64);
        self.visits += 1;
        reward
    }

    fn select_action(
        &mut self,
        agent: &mut dyn AgentSimulator,
        remaining_horizon: usize,
    ) -> (&mut RhoUctNode, Action) {
        let num_actions = agent.get_num_actions();
        ensure_action_slots(&mut self.action_children, num_actions);

        let action_idx = if let Some(unvisited) =
            choose_uniform_unvisited(agent, &self.action_children, num_actions)
        {
            self.action_children[unvisited] = Some(RhoUctNode::new(true));
            unvisited
        } else {
            let log_visits = (self.visits as f64).ln().max(0.0);
            let c = agent.get_explore_exploit_ratio().max(0.0);
            let mut best_score = -f64::INFINITY;
            let mut best_action = None;
            let mut num_maximal_actions = 0usize;

            for (action_idx, child) in self.action_children.iter().enumerate() {
                let Some(child) = child.as_ref() else {
                    continue;
                };
                let normalized_value = agent.norm_reward_for_horizon(child.mean, remaining_horizon);
                let exploration = if child.visits == 0 {
                    f64::INFINITY
                } else {
                    c * (log_visits / (child.visits as f64)).sqrt()
                };
                let score = normalized_value + exploration;
                debug_assert!(
                    score.is_finite(),
                    "rho_uct UCB score must be finite for visited action children"
                );
                match score.total_cmp(&best_score) {
                    std::cmp::Ordering::Greater => {
                        best_score = score;
                        best_action = Some(action_idx);
                        num_maximal_actions = 1;
                    }
                    std::cmp::Ordering::Equal => {
                        num_maximal_actions += 1;
                        if agent.gen_range(num_maximal_actions) == 0 {
                            best_action = Some(action_idx);
                        }
                    }
                    std::cmp::Ordering::Less => {}
                }
            }

            best_action.expect("rho_uct decision node must have a maximal action")
        };

        let action = action_idx as Action;
        agent.model_update_action(action);
        (
            self.action_children[action_idx]
                .as_mut()
                .expect("rho_uct action child missing"),
            action,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aixi::common::ObservationKeyMode;

    #[derive(Clone)]
    struct DummyAgent {
        num_actions: usize,
        obs_bits: usize,
        rew_bits: usize,
        horizon: usize,
        min_reward: Reward,
        max_reward: Reward,
        discount_gamma: f64,
        explore_exploit_ratio: f64,
        key_mode: ObservationKeyMode,
        last_range_end: usize,
        range_result: usize,
    }

    impl DummyAgent {
        fn new(obs_bits: usize, key_mode: ObservationKeyMode) -> Self {
            Self {
                num_actions: 4,
                obs_bits,
                rew_bits: 8,
                horizon: 5,
                min_reward: -1,
                max_reward: 1,
                discount_gamma: 1.0,
                explore_exploit_ratio: 1.0,
                key_mode,
                last_range_end: 0,
                range_result: 0,
            }
        }
    }

    impl AgentSimulator for DummyAgent {
        fn get_num_actions(&self) -> usize {
            self.num_actions
        }

        fn get_num_observation_bits(&self) -> usize {
            self.obs_bits
        }

        fn observation_key_mode(&self) -> ObservationKeyMode {
            self.key_mode
        }

        fn get_num_reward_bits(&self) -> usize {
            self.rew_bits
        }

        fn horizon(&self) -> usize {
            self.horizon
        }

        fn max_reward(&self) -> Reward {
            self.max_reward
        }

        fn min_reward(&self) -> Reward {
            self.min_reward
        }

        fn discount_gamma(&self) -> f64 {
            self.discount_gamma
        }

        fn get_explore_exploit_ratio(&self) -> f64 {
            self.explore_exploit_ratio
        }

        fn model_update_action(&mut self, _action: Action) {}

        fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
            0
        }

        fn model_revert(&mut self, _steps: usize) {}

        fn gen_range(&mut self, end: usize) -> usize {
            self.last_range_end = end;
            self.range_result.min(end.saturating_sub(1))
        }

        fn gen_f64(&mut self) -> f64 {
            0.0
        }

        fn boxed_clone_with_seed(&self, _seed: u64) -> Box<dyn AgentSimulator> {
            Box::new(self.clone())
        }
    }

    fn build_planner_with_key(
        agent: &DummyAgent,
        prev_act: Action,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        kept_mean: f64,
        kept_visits: u32,
    ) -> RhoUctPlanner {
        let mut old_root = RhoUctNode::new(false);
        old_root.action_children.resize(prev_act as usize + 1, None);

        let mut chance_child = RhoUctNode::new(true);
        let mut kept = RhoUctNode::new(false);
        kept.mean = kept_mean;
        kept.visits = kept_visits;

        let key = prune_key(agent, prev_obs_stream, prev_rew);
        chance_child.percept_children.insert(key, kept);
        old_root.action_children[prev_act as usize] = Some(chance_child);

        RhoUctPlanner {
            root: Some(old_root),
        }
    }

    #[test]
    fn prune_tree_keeps_matching_subtree() {
        let prev_act = 2u64;
        let prev_obs_stream = vec![9u64, 2u64, 7u64];
        let prev_rew: Reward = 3;

        let agent = DummyAgent::new(3, ObservationKeyMode::FullStream);
        let mut planner =
            build_planner_with_key(&agent, prev_act, &prev_obs_stream, prev_rew, 123.0, 7);

        planner.prune_tree(&agent, &prev_obs_stream, prev_rew, prev_act);

        let root = planner.root.as_ref().expect("rho_uct root should exist");
        assert!(!root.is_chance_node);
        assert_eq!(root.mean, 123.0);
        assert_eq!(root.visits, 7);
    }

    #[test]
    fn prune_tree_resets_when_action_missing() {
        let prev_act = 10u64;
        let prev_obs_stream = vec![1u64];
        let prev_rew: Reward = 0;

        let agent = DummyAgent::new(1, ObservationKeyMode::FullStream);
        let mut planner = RhoUctPlanner::new();

        planner.prune_tree(&agent, &prev_obs_stream, prev_rew, prev_act);

        let root = planner.root.as_ref().expect("rho_uct root");
        assert!(!root.is_chance_node);
        assert_eq!(root.visits, 0);
        assert_eq!(root.mean, 0.0);
    }

    #[test]
    fn prune_tree_resets_when_reward_mismatch_shares_observation_key() {
        let prev_act = 1u64;
        let prev_obs_stream = vec![4u64, 5u64];
        let kept_rew: Reward = -2;
        let requested_rew: Reward = 2;

        let agent = DummyAgent::new(6, ObservationKeyMode::FullStream);
        let mut planner =
            build_planner_with_key(&agent, prev_act, &prev_obs_stream, kept_rew, 77.0, 11);

        planner.prune_tree(&agent, &prev_obs_stream, requested_rew, prev_act);

        let root = planner.root.as_ref().expect("rho_uct root");
        assert_eq!(root.visits, 0);
        assert_eq!(root.mean, 0.0);
    }

    #[test]
    fn deeper_remaining_horizon_changes_ucb_normalization() {
        let mut root = RhoUctNode::new(false);
        root.visits = 32;
        root.action_children = vec![
            Some(RhoUctNode {
                visits: 8,
                mean: 2.0,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: PerceptMap::default(),
            }),
            Some(RhoUctNode {
                visits: 8,
                mean: 2.0,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: PerceptMap::default(),
            }),
        ];

        let mut shallow = DummyAgent::new(1, ObservationKeyMode::FullStream);
        shallow.horizon = 2;
        shallow.min_reward = -2;
        shallow.max_reward = 3;

        let mut deep = shallow.clone();
        deep.horizon = 8;

        let normalized_shallow =
            shallow.norm_reward_for_horizon(root.action_children[0].as_ref().unwrap().mean, 2);
        let normalized_deep =
            deep.norm_reward_for_horizon(root.action_children[0].as_ref().unwrap().mean, 8);
        assert!(
            normalized_shallow > normalized_deep,
            "same mean return should normalize differently when the remaining horizon changes"
        );
    }

    #[test]
    fn discounted_normalization_uses_remaining_horizon_bounds() {
        let mut agent = DummyAgent::new(1, ObservationKeyMode::FullStream);
        agent.min_reward = -1;
        agent.max_reward = 3;
        agent.discount_gamma = 0.5;

        let two_step = agent.norm_reward_for_horizon(1.0, 2);
        let four_step = agent.norm_reward_for_horizon(1.0, 4);
        assert_ne!(two_step, four_step);
    }

    #[test]
    fn undiscounted_normalization_is_action_equivalent_to_aixictwx_scaling() {
        let mut agent = DummyAgent::new(1, ObservationKeyMode::FullStream);
        agent.min_reward = -2;
        agent.max_reward = 3;
        agent.discount_gamma = 1.0;

        let remaining_horizon = 4usize;
        let lower_value = -1.5;
        let higher_value = 0.25;
        let normalized_order = agent
            .norm_reward_for_horizon(lower_value, remaining_horizon)
            .total_cmp(&agent.norm_reward_for_horizon(higher_value, remaining_horizon));
        let paper_range =
            (remaining_horizon as f64) * ((agent.max_reward - agent.min_reward) as f64);
        let paper_order = (lower_value / paper_range).total_cmp(&(higher_value / paper_range));

        assert_eq!(normalized_order, paper_order);
    }

    #[test]
    fn discounted_normalization_matches_discounted_finite_horizon_formula() {
        let mut agent = DummyAgent::new(1, ObservationKeyMode::FullStream);
        agent.min_reward = -1;
        agent.max_reward = 3;
        agent.discount_gamma = 0.5;

        let horizon = 4usize;
        let reward = 1.0;
        let discounted_sum = 1.0 + 0.5 + 0.25 + 0.125;
        let min_cumulative = -discounted_sum;
        let max_cumulative = 3.0 * discounted_sum;
        let expected = (reward - min_cumulative) / (max_cumulative - min_cumulative);

        assert!((agent.norm_reward_for_horizon(reward, horizon) - expected).abs() < 1e-12);
    }

    #[test]
    fn rho_uct_select_action_uses_remaining_horizon_ucb_scaling() {
        let mut root = RhoUctNode::new(false);
        root.visits = 16;
        root.action_children = vec![
            Some(RhoUctNode {
                visits: 4,
                mean: -1.5,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: PerceptMap::default(),
            }),
            Some(RhoUctNode {
                visits: 8,
                mean: -0.5,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: PerceptMap::default(),
            }),
        ];

        let mut agent = DummyAgent::new(1, ObservationKeyMode::FullStream);
        agent.num_actions = 2;
        agent.horizon = 5;
        agent.min_reward = -2;
        agent.max_reward = 3;
        agent.discount_gamma = 0.5;
        agent.explore_exploit_ratio = 0.25;
        agent.range_result = 0;

        let remaining_horizon = 3usize;
        let log_visits = (root.visits as f64).ln();
        let c = agent.get_explore_exploit_ratio();
        let left_score =
            agent.norm_reward_for_horizon(-1.5, remaining_horizon) + c * (log_visits / 4.0).sqrt();
        let right_score =
            agent.norm_reward_for_horizon(-0.5, remaining_horizon) + c * (log_visits / 8.0).sqrt();
        assert!(
            right_score > left_score,
            "expected action 1 to win under the rho_uct paper formula"
        );

        let (_child, action) = root.select_action(&mut agent, remaining_horizon);
        assert_eq!(action, 1);
    }

    #[test]
    fn rho_uct_select_action_breaks_exact_ties_uniformly_via_agent_rng() {
        let mut root = RhoUctNode::new(false);
        root.visits = 16;
        root.action_children = vec![
            Some(RhoUctNode {
                visits: 4,
                mean: 1.0,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: PerceptMap::default(),
            }),
            Some(RhoUctNode {
                visits: 4,
                mean: 1.0,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: PerceptMap::default(),
            }),
        ];

        let mut agent = DummyAgent::new(1, ObservationKeyMode::FullStream);
        agent.num_actions = 2;
        agent.min_reward = 0;
        agent.max_reward = 2;
        agent.range_result = 0;

        let (_child, action) = root.select_action(&mut agent, 2);
        assert_eq!(agent.last_range_end, 2);
        assert_eq!(action, 1);
    }

    #[derive(Clone)]
    struct DeterministicRewardAgent {
        last_action: Action,
        emit_reward: bool,
    }

    impl AgentSimulator for DeterministicRewardAgent {
        fn get_num_actions(&self) -> usize {
            2
        }

        fn get_num_observation_bits(&self) -> usize {
            1
        }

        fn get_num_reward_bits(&self) -> usize {
            1
        }

        fn horizon(&self) -> usize {
            1
        }

        fn max_reward(&self) -> Reward {
            1
        }

        fn min_reward(&self) -> Reward {
            0
        }

        fn get_explore_exploit_ratio(&self) -> f64 {
            0.0
        }

        fn model_update_action(&mut self, action: Action) {
            self.last_action = action;
            self.emit_reward = false;
        }

        fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
            if self.emit_reward {
                self.emit_reward = false;
                self.last_action
            } else {
                self.emit_reward = true;
                0
            }
        }

        fn model_revert(&mut self, _steps: usize) {
            self.emit_reward = false;
        }

        fn gen_range(&mut self, _end: usize) -> usize {
            0
        }

        fn gen_f64(&mut self) -> f64 {
            0.0
        }

        fn boxed_clone_with_seed(&self, _seed: u64) -> Box<dyn AgentSimulator> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn rho_uct_is_not_promoted_to_parallel_for_multiple_samples() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut planner = RhoUctPlanner::new();
        let action = planner.search(&mut agent, &[0], 0, 0, 8);
        let root = planner.root.as_ref().expect("rho_uct root");
        assert!(action < 2);
        assert_eq!(root.visits, 8);
        assert!(
            root.action_children
                .iter()
                .filter_map(Option::as_ref)
                .any(|child| child.visits > 0),
            "sequential rho_uct should accumulate completed action visits directly in the shared tree"
        );
    }
}
