use super::{
    AgentSimulator, PerceptMap, PerceptOutcome, best_action_from_action_values,
    choose_uniform_unvisited, ensure_action_slots, prune_key, random_rollout,
};
#[cfg(test)]
use crate::aixi::common::ActionAlphabet;
use crate::aixi::common::{Action, PerceptVal, Reward};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
#[cfg(test)]
use std::sync::{Arc, Mutex};

const PARALLEL_PLANNER_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Construction-time errors for [`ParallelUctPlanner`].
///
/// `workers == 0` is type-prevented at the API boundary by
/// [`NonZeroUsize`], so the only remaining failure mode is an out-of-range
/// `bu_uct_m_max`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelUctPlannerInitError {
    /// `bu_uct_m_max` was provided but is not strictly inside `(0, 1)`.
    InvalidBuUctMMax,
}

impl fmt::Display for ParallelUctPlannerInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBuUctMMax => write!(f, "parallel_uct bu_uct_m_max must be in (0, 1)"),
        }
    }
}

impl std::error::Error for ParallelUctPlannerInitError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelUctSearchError {
    PositiveSamplesRequirePositiveHorizon,
}

impl fmt::Display for ParallelUctSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositiveSamplesRequirePositiveHorizon => {
                write!(
                    f,
                    "parallel_uct requires agent.horizon() >= 1 when samples > 0"
                )
            }
        }
    }
}

impl std::error::Error for ParallelUctSearchError {}

/// Explicit parallel UCT planner state.
///
/// This backend implements explicit WU-UCT accounting and BU-core thresholding
/// plus grouped backpropagation over deterministic completion epochs. It does
/// not claim the full supplementary BU-UCT expansion scheduler.
pub struct ParallelUctPlanner {
    state: ParallelPlannerState,
    workers: NonZeroUsize,
    bu_uct_m_max: Option<f64>,
}

impl ParallelUctPlanner {
    /// Construct a parallel UCT planner.
    ///
    /// `workers` is type-enforced non-zero. `bu_uct_m_max == None` selects
    /// WU-UCT; `Some(x)` with `x \in (0, 1)` selects BU-UCT thresholding.
    pub fn new(
        workers: NonZeroUsize,
        bu_uct_m_max: Option<f64>,
    ) -> Result<Self, ParallelUctPlannerInitError> {
        if matches!(bu_uct_m_max, Some(m_max) if !(0.0 < m_max && m_max < 1.0)) {
            return Err(ParallelUctPlannerInitError::InvalidBuUctMMax);
        }
        Ok(Self {
            state: if bu_uct_m_max.is_some() {
                ParallelPlannerState::Bu(ParallelRuntime::new())
            } else {
                ParallelPlannerState::Wu(ParallelRuntime::new())
            },
            workers,
            bu_uct_m_max,
        })
    }

    pub fn search(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: Action,
        samples: usize,
    ) -> Result<Action, ParallelUctSearchError> {
        let horizon = agent.horizon();
        if samples > 0 && horizon == 0 {
            return Err(ParallelUctSearchError::PositiveSamplesRequirePositiveHorizon);
        }
        Ok(self.search_validated_with_horizon(
            agent,
            prev_obs_stream,
            prev_rew,
            prev_act,
            samples,
            horizon,
        ))
    }

    pub(crate) fn search_validated(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: Action,
        samples: usize,
    ) -> Action {
        let horizon = agent.horizon();
        debug_assert!(
            samples == 0 || horizon > 0,
            "parallel_uct validated search requires agent.horizon() >= 1 when samples > 0"
        );
        self.search_validated_with_horizon(
            agent,
            prev_obs_stream,
            prev_rew,
            prev_act,
            samples,
            horizon,
        )
    }

    fn search_validated_with_horizon(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: Action,
        samples: usize,
        horizon: usize,
    ) -> Action {
        let workers = self.workers.get();
        match &mut self.state {
            ParallelPlannerState::Wu(runtime) => search_runtime::<WuMode>(
                runtime,
                agent,
                prev_obs_stream,
                prev_rew,
                prev_act,
                samples,
                horizon,
                workers,
                None,
            ),
            ParallelPlannerState::Bu(runtime) => search_runtime::<BuMode>(
                runtime,
                agent,
                prev_obs_stream,
                prev_rew,
                prev_act,
                samples,
                horizon,
                workers,
                self.bu_uct_m_max,
            ),
        }
    }
}

fn search_runtime<M: ModeState>(
    runtime: &mut ParallelRuntime<M>,
    agent: &mut dyn AgentSimulator,
    prev_obs_stream: &[PerceptVal],
    prev_rew: Reward,
    prev_act: Action,
    samples: usize,
    horizon: usize,
    workers: usize,
    bu_uct_m_max: Option<f64>,
) -> Action {
    prune_tree(runtime, agent, prev_obs_stream, prev_rew, prev_act);

    debug_assert!(workers > 0);
    let logical_workers = workers.min(samples.max(1));
    let planner_seed = agent.gen_f64().to_bits();
    let gamma = agent.discount_gamma().clamp(0.0, 1.0);

    let mut dispatched = 0usize;
    if samples > 0 {
        let root_is_fresh = runtime.root.as_ref().is_some_and(|root| root.visits == 0);
        if root_is_fresh {
            let task_index = 0usize;
            let mut local_agent =
                agent.boxed_clone_with_seed(planner_task_seed(planner_seed, task_index));
            local_agent.begin_simulation();
            let bootstrap = bootstrap_root(runtime, local_agent.as_mut(), horizon, task_index);
            complete_update_batch(runtime, std::slice::from_ref(&bootstrap), gamma);
            dispatched = 1;
        }
    }

    while dispatched < samples {
        let batch_size = logical_workers.min(samples - dispatched);
        let mut pending = Vec::with_capacity(batch_size);

        for batch_index in 0..batch_size {
            let task_index = dispatched + batch_index;
            let mut local_agent =
                agent.boxed_clone_with_seed(planner_task_seed(planner_seed, task_index));
            local_agent.begin_simulation();

            let dispatch = {
                let root = runtime.root.as_mut().expect("parallel_uct root missing");
                dispatch_rollout::<M>(
                    root,
                    &mut runtime.next_node_id,
                    local_agent.as_mut(),
                    horizon,
                    workers,
                    bu_uct_m_max,
                )
            };
            pending.push(PendingRollout {
                task_index,
                agent: local_agent,
                remaining_horizon: dispatch.remaining_horizon,
                path: dispatch.path,
            });
        }

        let completed = pending
            .into_par_iter()
            .map(|mut task| CompletedRollout {
                task_index: task.task_index,
                path: task.path,
                tail_reward: random_rollout(task.agent.as_mut(), task.remaining_horizon),
            })
            .collect::<Vec<_>>();
        let mut completed = completed;
        completed.sort_by_key(|task| task.task_index);
        complete_update_batch(runtime, &completed, gamma);

        dispatched += batch_size;
    }

    if samples > 0 {
        let root = runtime.root.as_ref().expect("parallel_uct root missing");
        return best_action_after_positive_budget(root, agent);
    }

    best_action(runtime.root.as_ref(), agent)
}

fn planner_task_seed(planner_seed: u64, task_index: usize) -> u64 {
    planner_seed ^ ((task_index as u64).wrapping_mul(PARALLEL_PLANNER_SEED_SALT))
}

fn best_action<M: ModeState>(
    root: Option<&DecisionNode<M>>,
    agent: &mut dyn AgentSimulator,
) -> Action {
    let Some(root) = root else {
        return agent.gen_range(agent.get_num_actions().get()) as Action;
    };
    best_action_from_action_values(
        root.action_edges
            .iter()
            .enumerate()
            .filter_map(|(action_idx, edge)| {
                edge.as_ref()
                    .filter(|edge| edge.completed_n() > 0)
                    .map(|edge| (action_idx, edge.completed_q()))
            }),
        agent.get_num_actions(),
        agent,
    )
}

fn best_action_after_positive_budget<M: ModeState>(
    root: &DecisionNode<M>,
    agent: &mut dyn AgentSimulator,
) -> Action {
    debug_assert!(
        root.action_edges
            .iter()
            .filter_map(Option::as_ref)
            .any(|edge| edge.completed_n() > 0),
        "positive-budget parallel_uct search must leave at least one completed root edge"
    );

    best_action(Some(root), agent)
}

#[cfg(test)]
fn root_has_completed_edge<M: ModeState>(root: &DecisionNode<M>) -> bool {
    root.action_edges
        .iter()
        .filter_map(Option::as_ref)
        .any(|edge| edge.completed_n() > 0)
}

fn prune_tree<M: ModeState>(
    runtime: &mut ParallelRuntime<M>,
    agent: &dyn AgentSimulator,
    prev_obs_stream: &[PerceptVal],
    prev_rew: Reward,
    prev_act: Action,
) {
    let Some(mut old_root) = runtime.root.take() else {
        runtime.root = Some(runtime.fresh_decision_node());
        return;
    };

    let action_edge = old_root
        .action_edges
        .get_mut(prev_act as usize)
        .and_then(Option::take);
    let Some(mut action_edge) = action_edge else {
        runtime.root = Some(runtime.fresh_decision_node());
        return;
    };

    let key = prune_key(agent, prev_obs_stream, prev_rew);
    runtime.root = action_edge
        .chance_mut()
        .percept_children
        .remove(&key)
        .or_else(|| Some(runtime.fresh_decision_node()));
}

fn bootstrap_root<M: ModeState>(
    runtime: &mut ParallelRuntime<M>,
    agent: &mut dyn AgentSimulator,
    remaining_horizon: usize,
    task_index: usize,
) -> CompletedRollout {
    debug_assert!(remaining_horizon > 0);
    let root = runtime.root.as_mut().expect("parallel_uct root missing");
    let action_idx = choose_bootstrap_action(root, agent);

    agent.model_update_action(action_idx as Action);
    let (observations, immediate_reward) = agent.gen_percepts_and_update();
    let outcome = PerceptOutcome::new(observations, immediate_reward);

    let parent_node_id = root.id;
    let edge = root.action_edges[action_idx]
        .as_mut()
        .expect("parallel_uct bootstrap action edge missing");
    if !edge.chance().percept_children.contains_key(&outcome) {
        let id = runtime.next_node_id;
        runtime.next_node_id += 1;
        edge.chance_mut()
            .percept_children
            .insert(outcome.clone(), DecisionNode::new(id));
    }
    edge.on_incomplete_update();
    let child_node_id = edge
        .chance()
        .percept_children
        .get(&outcome)
        .expect("parallel_uct bootstrap percept child missing")
        .id;

    let path = vec![ParallelPathStep {
        parent_node_id,
        action_idx,
        child_node_id,
        outcome,
    }];
    let tail_reward = random_rollout(agent, remaining_horizon.saturating_sub(1));
    CompletedRollout {
        task_index,
        path,
        tail_reward,
    }
}

fn choose_bootstrap_action<M: ModeState>(
    root: &mut DecisionNode<M>,
    agent: &mut dyn AgentSimulator,
) -> usize {
    let num_actions = agent.get_num_actions();
    ensure_action_slots(&mut root.action_edges, num_actions.get());

    let mut unvisited = Vec::new();
    for action_idx in 0..num_actions.get() {
        match root.action_edges.get(action_idx).and_then(Option::as_ref) {
            None => unvisited.push(action_idx),
            Some(edge) if edge.completed_n() == 0 && edge.effective_visits() == 0 => {
                unvisited.push(action_idx);
            }
            Some(_) => {}
        }
    }

    let selected = unvisited[agent.gen_range(unvisited.len())];
    if root.action_edges[selected].is_none() {
        root.action_edges[selected] = Some(M::Edge::new());
    }
    selected
}

fn dispatch_rollout<M: ModeState>(
    node: &mut DecisionNode<M>,
    next_node_id: &mut u64,
    agent: &mut dyn AgentSimulator,
    remaining_horizon: usize,
    workers: usize,
    bu_uct_m_max: Option<f64>,
) -> DispatchRollout {
    let mut path = Vec::new();
    let remaining_horizon = dispatch_rollout_into::<M>(
        node,
        next_node_id,
        agent,
        remaining_horizon,
        workers,
        bu_uct_m_max,
        &mut path,
    );
    DispatchRollout {
        remaining_horizon,
        path,
    }
}

fn dispatch_rollout_into<M: ModeState>(
    node: &mut DecisionNode<M>,
    next_node_id: &mut u64,
    agent: &mut dyn AgentSimulator,
    remaining_horizon: usize,
    workers: usize,
    bu_uct_m_max: Option<f64>,
    path: &mut Vec<ParallelPathStep>,
) -> usize {
    if remaining_horizon == 0 || node.visits == 0 {
        return remaining_horizon;
    }

    let num_actions = agent.get_num_actions();
    ensure_action_slots(&mut node.action_edges, num_actions.get());

    let action_idx = if let Some(unvisited) =
        choose_uniform_unvisited(agent, &node.action_edges, num_actions.get())
    {
        node.action_edges[unvisited] = Some(M::Edge::new());
        unvisited
    } else {
        let Some(action_idx) =
            select_existing_action(node, agent, remaining_horizon, workers, bu_uct_m_max)
        else {
            return remaining_horizon;
        };
        action_idx
    };

    agent.model_update_action(action_idx as Action);
    let (observations, immediate_reward) = agent.gen_percepts_and_update();
    let outcome = PerceptOutcome::new(observations, immediate_reward);
    let parent_node_id = node.id;

    let edge = node.action_edges[action_idx]
        .as_mut()
        .expect("parallel_uct action edge missing");
    if !edge.chance().percept_children.contains_key(&outcome) {
        let id = *next_node_id;
        *next_node_id += 1;
        edge.chance_mut()
            .percept_children
            .insert(outcome.clone(), DecisionNode::new(id));
    }
    edge.on_incomplete_update();
    let child = edge
        .chance_mut()
        .percept_children
        .get_mut(&outcome)
        .expect("parallel_uct percept child missing after insertion");
    let child_node_id = child.id;

    path.push(ParallelPathStep {
        parent_node_id,
        action_idx,
        child_node_id,
        outcome,
    });

    dispatch_rollout_into::<M>(
        child,
        next_node_id,
        agent,
        remaining_horizon - 1,
        workers,
        bu_uct_m_max,
        path,
    )
}

fn select_existing_action<M: ModeState>(
    node: &DecisionNode<M>,
    agent: &mut dyn AgentSimulator,
    remaining_horizon: usize,
    workers: usize,
    bu_uct_m_max: Option<f64>,
) -> Option<usize> {
    let total_overline_n = node
        .action_edges
        .iter()
        .filter_map(Option::as_ref)
        .map(EdgeOps::effective_visits)
        .sum::<u32>();
    let log_total = ((total_overline_n.max(1)) as f64).ln().max(0.0);
    let c = agent.get_explore_exploit_ratio().max(0.0);

    let mut best_score = -f64::INFINITY;
    let mut best_action = None;
    let mut num_maximal_actions = 0usize;

    for (action_idx, edge) in node.action_edges.iter().enumerate() {
        let Some(edge) = edge.as_ref() else {
            continue;
        };
        let overline_n = edge.effective_visits();
        if overline_n == 0 || !M::edge_is_selectable(edge, workers, bu_uct_m_max) {
            continue;
        }

        let normalized_value = agent.norm_reward_for_horizon(edge.completed_q(), remaining_horizon);
        let exploration = c * ((2.0 * log_total) / (overline_n as f64)).sqrt();
        let score = normalized_value + exploration;
        debug_assert!(
            score.is_finite(),
            "parallel_uct UCB score must be finite for visited action edges"
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

    best_action
}

#[cfg(test)]
fn incomplete_update<M: ModeState>(runtime: &mut ParallelRuntime<M>, path: &[ParallelPathStep]) {
    if path.is_empty() {
        return;
    }
    let mut current = runtime.root.as_mut().expect("parallel_uct root missing");
    for step in path {
        let edge = current.action_edges[step.action_idx]
            .as_mut()
            .expect("parallel_uct action edge missing during incomplete_update");
        edge.on_incomplete_update();
        current = edge
            .chance_mut()
            .percept_children
            .get_mut(&step.outcome)
            .expect("parallel_uct percept child missing during incomplete_update");
    }
}

fn complete_update_batch<M: ModeState>(
    runtime: &mut ParallelRuntime<M>,
    completed: &[CompletedRollout],
    gamma: f64,
) {
    let mut epoch_state = M::EpochState::default();
    for task in completed {
        let root = runtime.root.as_mut().expect("parallel_uct root missing");
        complete_update_node::<M>(
            root,
            &task.path,
            0,
            task.tail_reward,
            gamma,
            &mut epoch_state,
        );
    }
}

fn complete_update_node<M: ModeState>(
    node: &mut DecisionNode<M>,
    path: &[ParallelPathStep],
    depth: usize,
    tail_reward: f64,
    gamma: f64,
    epoch_state: &mut M::EpochState,
) -> f64 {
    node.visits += 1;
    if depth == path.len() {
        return tail_reward;
    }

    let step = &path[depth];
    let edge = node.action_edges[step.action_idx]
        .as_mut()
        .expect("parallel_uct action edge missing during complete_update");
    let child = edge
        .chance_mut()
        .percept_children
        .get_mut(&step.outcome)
        .expect("parallel_uct percept child missing during complete_update");
    let downstream =
        complete_update_node::<M>(child, path, depth + 1, tail_reward, gamma, epoch_state);

    let reward = (step.outcome.reward() as f64) + gamma * downstream;
    M::complete_edge(
        edge,
        BuEpochKey {
            parent_node_id: step.parent_node_id,
            action_idx: step.action_idx,
            child_node_id: step.child_node_id,
        },
        reward,
        epoch_state,
    );
    reward
}

enum ParallelPlannerState {
    Wu(ParallelRuntime<WuMode>),
    Bu(ParallelRuntime<BuMode>),
}

struct ParallelRuntime<M: ModeState> {
    root: Option<DecisionNode<M>>,
    next_node_id: u64,
}

impl<M: ModeState> ParallelRuntime<M> {
    fn new() -> Self {
        Self {
            root: Some(DecisionNode::new(0)),
            next_node_id: 1,
        }
    }

    fn fresh_decision_node(&mut self) -> DecisionNode<M> {
        let id = self.next_node_id;
        self.next_node_id += 1;
        DecisionNode::new(id)
    }
}

trait ModeState: Copy {
    type Edge: EdgeOps<Self>;
    type EpochState: Default;

    fn edge_is_selectable(edge: &Self::Edge, workers: usize, bu_uct_m_max: Option<f64>) -> bool;

    fn complete_edge(
        edge: &mut Self::Edge,
        key: BuEpochKey,
        reward: f64,
        epoch_state: &mut Self::EpochState,
    );
}

trait EdgeOps<M: ModeState>: Clone {
    fn new() -> Self;
    fn chance(&self) -> &ChanceNode<M>;
    fn chance_mut(&mut self) -> &mut ChanceNode<M>;
    fn effective_visits(&self) -> u32;
    fn completed_q(&self) -> f64;
    fn completed_n(&self) -> u32;
    fn on_incomplete_update(&mut self);
}

#[derive(Clone, Copy)]
struct WuMode;

#[derive(Clone, Copy)]
struct BuMode;

#[derive(Clone)]
struct DecisionNode<M: ModeState> {
    id: u64,
    visits: u32,
    action_edges: Vec<Option<M::Edge>>,
}

impl<M: ModeState> DecisionNode<M> {
    fn new(id: u64) -> Self {
        Self {
            id,
            visits: 0,
            action_edges: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct ChanceNode<M: ModeState> {
    percept_children: PerceptMap<DecisionNode<M>>,
}

impl<M: ModeState> Default for ChanceNode<M> {
    fn default() -> Self {
        Self {
            percept_children: PerceptMap::default(),
        }
    }
}

#[derive(Clone)]
struct WuActionEdge {
    q: f64,
    n: u32,
    o: u32,
    child: ChanceNode<WuMode>,
}

impl EdgeOps<WuMode> for WuActionEdge {
    fn new() -> Self {
        Self {
            q: 0.0,
            n: 0,
            o: 0,
            child: ChanceNode::default(),
        }
    }

    fn chance(&self) -> &ChanceNode<WuMode> {
        &self.child
    }

    fn chance_mut(&mut self) -> &mut ChanceNode<WuMode> {
        &mut self.child
    }

    fn effective_visits(&self) -> u32 {
        self.n + self.o
    }

    fn completed_q(&self) -> f64 {
        self.q
    }

    fn completed_n(&self) -> u32 {
        self.n
    }

    fn on_incomplete_update(&mut self) {
        self.o += 1;
    }
}

impl ModeState for WuMode {
    type Edge = WuActionEdge;
    type EpochState = ();

    fn edge_is_selectable(_edge: &Self::Edge, _workers: usize, _bu_uct_m_max: Option<f64>) -> bool {
        true
    }

    fn complete_edge(
        edge: &mut Self::Edge,
        _key: BuEpochKey,
        reward: f64,
        _epoch_state: &mut Self::EpochState,
    ) {
        edge.o = edge.o.saturating_sub(1);
        edge.q = (reward + (edge.n as f64) * edge.q) / ((edge.n + 1) as f64);
        edge.n += 1;
    }
}

#[derive(Clone)]
struct BuActionEdge {
    q: f64,
    n: u32,
    o: u32,
    // Paper-style BU incomplete-occupancy statistic updated only on
    // `incomplete_update`; it is not a live mirror of the current `o` count.
    o_bar: f64,
    child: ChanceNode<BuMode>,
}

impl EdgeOps<BuMode> for BuActionEdge {
    fn new() -> Self {
        Self {
            q: 0.0,
            n: 0,
            o: 0,
            o_bar: 0.0,
            child: ChanceNode::default(),
        }
    }

    fn chance(&self) -> &ChanceNode<BuMode> {
        &self.child
    }

    fn chance_mut(&mut self) -> &mut ChanceNode<BuMode> {
        &mut self.child
    }

    fn effective_visits(&self) -> u32 {
        self.n + self.o
    }

    fn completed_q(&self) -> f64 {
        self.q
    }

    fn completed_n(&self) -> u32 {
        self.n
    }

    fn on_incomplete_update(&mut self) {
        self.o += 1;
        let overline_n = self.effective_visits();
        if overline_n > 0 {
            self.o_bar =
                (((overline_n - 1) as f64) * self.o_bar + (self.o as f64)) / (overline_n as f64);
        }
    }
}

impl ModeState for BuMode {
    type Edge = BuActionEdge;
    type EpochState = BuEpochState;

    fn edge_is_selectable(edge: &Self::Edge, workers: usize, bu_uct_m_max: Option<f64>) -> bool {
        let m_max = bu_uct_m_max.expect("BU mode requires bu_uct_m_max");
        edge.o_bar < m_max * (workers as f64)
    }

    fn complete_edge(
        edge: &mut Self::Edge,
        key: BuEpochKey,
        reward: f64,
        epoch_state: &mut Self::EpochState,
    ) {
        edge.o = edge.o.saturating_sub(1);
        // BU-core Part 1 intentionally keeps the paper-style one-sided `o_bar`
        // lifecycle: completion decrements live `o` but does not recompute
        // `o_bar`, so thresholding continues to use the accumulated incomplete
        // occupancy statistic rather than current live occupancy.
        edge.update_bu_epoch(key, reward, epoch_state);
    }
}

impl BuActionEdge {
    fn update_bu_epoch(&mut self, key: BuEpochKey, reward: f64, epoch_state: &mut BuEpochState) {
        use std::collections::hash_map::Entry;

        match epoch_state.groups.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(BuGroupStat {
                    mean: reward,
                    count: 1,
                });
                self.q = if self.n == 0 {
                    reward
                } else {
                    (((self.n as f64) * self.q) + reward) / ((self.n + 1) as f64)
                };
                self.n += 1;
            }
            Entry::Occupied(mut entry) => {
                let old_mean = entry.get().mean;
                let stat = entry.get_mut();
                stat.count += 1;
                stat.mean = old_mean + (reward - old_mean) / (stat.count as f64);
                if self.n > 0 {
                    self.q += (stat.mean - old_mean) / (self.n as f64);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BuEpochKey {
    parent_node_id: u64,
    action_idx: usize,
    child_node_id: u64,
}

#[derive(Clone, Copy)]
struct BuGroupStat {
    mean: f64,
    count: u32,
}

#[derive(Default)]
struct BuEpochState {
    groups: HashMap<BuEpochKey, BuGroupStat>,
}

struct DispatchRollout {
    remaining_horizon: usize,
    path: Vec<ParallelPathStep>,
}

struct PendingRollout {
    task_index: usize,
    agent: Box<dyn AgentSimulator>,
    remaining_horizon: usize,
    path: Vec<ParallelPathStep>,
}

struct CompletedRollout {
    task_index: usize,
    path: Vec<ParallelPathStep>,
    tail_reward: f64,
}

#[derive(Clone)]
struct ParallelPathStep {
    parent_node_id: u64,
    action_idx: usize,
    child_node_id: u64,
    outcome: PerceptOutcome,
}

#[cfg(test)]
type WuDecisionNode = DecisionNode<WuMode>;
#[cfg(test)]
type BuDecisionNode = DecisionNode<BuMode>;
#[cfg(test)]
type BuChanceNode = ChanceNode<BuMode>;

#[cfg(test)]
impl ParallelUctPlanner {
    fn wu_root(&self) -> &WuDecisionNode {
        match &self.state {
            ParallelPlannerState::Wu(runtime) => runtime.root.as_ref().expect("WU root"),
            ParallelPlannerState::Bu(_) => panic!("expected WU planner"),
        }
    }

    fn bu_root(&self) -> &BuDecisionNode {
        match &self.state {
            ParallelPlannerState::Bu(runtime) => runtime.root.as_ref().expect("BU root"),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }
    }

    fn set_wu_root(&mut self, root: WuDecisionNode) {
        match &mut self.state {
            ParallelPlannerState::Wu(runtime) => runtime.root = Some(root),
            ParallelPlannerState::Bu(_) => panic!("expected WU planner"),
        }
    }

    fn set_bu_root(&mut self, root: BuDecisionNode) {
        match &mut self.state {
            ParallelPlannerState::Bu(runtime) => runtime.root = Some(root),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aixi::mcts::RhoUctPlanner;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test helper: build a `NonZeroUsize` worker count, panicking if zero.
    ///
    /// `NonZeroUsize` is the type-enforced API for `ParallelUctPlanner::new`,
    /// so test fixtures opt into a tiny helper rather than repeating
    /// `NonZeroUsize::new(N).expect(..)` at every call site.
    fn workers(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test fixtures must use non-zero worker counts")
    }

    #[derive(Clone)]
    struct DeterministicRewardAgent {
        last_action: Action,
        emit_reward: bool,
    }

    impl AgentSimulator for DeterministicRewardAgent {
        fn get_num_actions(&self) -> ActionAlphabet {
            ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid")
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

        fn boxed_clone_with_seed(&self, seed: u64) -> Box<dyn AgentSimulator> {
            let _ = seed;
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct ThresholdProbeAgent {
        num_actions: usize,
        model_updates: usize,
        range_result: usize,
    }

    impl AgentSimulator for ThresholdProbeAgent {
        fn get_num_actions(&self) -> ActionAlphabet {
            ActionAlphabet::try_from_usize(self.num_actions)
                .expect("test fixture action alphabet must be valid")
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

        fn model_update_action(&mut self, _action: Action) {
            self.model_updates += 1;
        }

        fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
            0
        }

        fn model_revert(&mut self, _steps: usize) {}

        fn gen_range(&mut self, end: usize) -> usize {
            self.range_result.min(end.saturating_sub(1))
        }

        fn gen_f64(&mut self) -> f64 {
            0.0
        }

        fn boxed_clone_with_seed(&self, _seed: u64) -> Box<dyn AgentSimulator> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct CounterAgent {
        clone_count: Arc<AtomicUsize>,
        begin_count: Arc<AtomicUsize>,
        model_updates: Arc<AtomicUsize>,
        planning_horizon: usize,
        last_action: Action,
        emit_reward: bool,
    }

    impl CounterAgent {
        fn new_with_horizon(planning_horizon: usize) -> Self {
            Self {
                clone_count: Arc::new(AtomicUsize::new(0)),
                begin_count: Arc::new(AtomicUsize::new(0)),
                model_updates: Arc::new(AtomicUsize::new(0)),
                planning_horizon,
                last_action: 0,
                emit_reward: false,
            }
        }
    }

    impl AgentSimulator for CounterAgent {
        fn get_num_actions(&self) -> ActionAlphabet {
            ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid")
        }

        fn get_num_observation_bits(&self) -> usize {
            1
        }

        fn get_num_reward_bits(&self) -> usize {
            1
        }

        fn horizon(&self) -> usize {
            self.planning_horizon
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

        fn begin_simulation(&mut self) {
            self.begin_count.fetch_add(1, Ordering::SeqCst);
        }

        fn model_update_action(&mut self, action: Action) {
            self.last_action = action;
            self.emit_reward = false;
            self.model_updates.fetch_add(1, Ordering::SeqCst);
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
            self.clone_count.fetch_add(1, Ordering::SeqCst);
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct SeedRecordingAgent {
        recorded_seeds: Arc<Mutex<Vec<u64>>>,
        last_action: Action,
        emit_reward: bool,
        clone_seed: u64,
    }

    impl SeedRecordingAgent {
        fn new() -> Self {
            Self {
                recorded_seeds: Arc::new(Mutex::new(Vec::new())),
                last_action: 0,
                emit_reward: false,
                clone_seed: 0,
            }
        }
    }

    impl AgentSimulator for SeedRecordingAgent {
        fn get_num_actions(&self) -> ActionAlphabet {
            ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid")
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
                ((self.clone_seed ^ self.last_action) & 1) as u64
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
            0.25
        }

        fn boxed_clone_with_seed(&self, seed: u64) -> Box<dyn AgentSimulator> {
            self.recorded_seeds.lock().expect("seed list").push(seed);
            Box::new(Self {
                recorded_seeds: Arc::clone(&self.recorded_seeds),
                last_action: 0,
                emit_reward: false,
                clone_seed: seed,
            })
        }
    }

    fn step(
        parent_node_id: u64,
        action_idx: usize,
        child_node_id: u64,
        observation: u64,
        reward: Reward,
    ) -> ParallelPathStep {
        ParallelPathStep {
            parent_node_id,
            action_idx,
            child_node_id,
            outcome: PerceptOutcome::new(vec![observation], reward),
        }
    }

    fn two_step_wu_root() -> WuDecisionNode {
        WuDecisionNode {
            id: 10,
            visits: 2,
            action_edges: vec![
                Some(WuActionEdge {
                    q: 1.0,
                    n: 1,
                    o: 0,
                    child: ChanceNode {
                        percept_children: HashMap::from([(
                            PerceptOutcome::new(vec![0], 0),
                            WuDecisionNode {
                                id: 11,
                                visits: 2,
                                action_edges: vec![
                                    Some(WuActionEdge {
                                        q: 1.0,
                                        n: 1,
                                        o: 0,
                                        child: ChanceNode {
                                            percept_children: HashMap::from([(
                                                PerceptOutcome::new(vec![0], 0),
                                                WuDecisionNode::new(12),
                                            )]),
                                        },
                                    }),
                                    Some(WuActionEdge {
                                        q: 0.0,
                                        n: 1,
                                        o: 0,
                                        child: ChanceNode::default(),
                                    }),
                                ],
                            },
                        )]),
                    },
                }),
                Some(WuActionEdge {
                    q: 0.0,
                    n: 1,
                    o: 0,
                    child: ChanceNode::default(),
                }),
            ],
        }
    }

    fn retained_parent_for(next_root: WuDecisionNode) -> WuDecisionNode {
        WuDecisionNode {
            id: 9,
            visits: 1,
            action_edges: vec![Some(WuActionEdge {
                q: 0.0,
                n: 1,
                o: 0,
                child: ChanceNode {
                    percept_children: HashMap::from([(PerceptOutcome::new(vec![0], 0), next_root)]),
                },
            })],
        }
    }

    // NOTE: `workers == 0` is now type-enforced at the API boundary by
    // `ParallelUctPlanner::new` accepting `NonZeroUsize`, so a runtime test
    // analogous to the previous `planner_new_rejects_zero_workers_at_api_boundary`
    // is structurally impossible here and would not even compile.

    #[test]
    fn planner_new_rejects_invalid_bu_threshold_at_api_boundary() {
        for invalid in [0.0, 1.0, -0.1, 1.1] {
            let err = match ParallelUctPlanner::new(workers(2), Some(invalid)) {
                Ok(_) => panic!("invalid BU threshold must be rejected"),
                Err(err) => err,
            };
            assert_eq!(err, ParallelUctPlannerInitError::InvalidBuUctMMax);
        }
    }

    #[test]
    fn wu_workers_one_matches_rho_uct_on_deterministic_agent() {
        let mut seq_agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut par_agent = seq_agent.clone();

        let mut sequential = RhoUctPlanner::new();
        let mut parallel =
            ParallelUctPlanner::new(workers(1), None).expect("valid parallel_uct planner");

        let seq_action = sequential.search(&mut seq_agent, &[0], 0, 0, 16);
        let par_action = parallel
            .search(&mut par_agent, &[0], 0, 0, 16)
            .expect("positive-horizon parallel_uct search");

        assert_eq!(seq_action, par_action);
        assert_eq!(parallel.wu_root().visits, 16);
    }

    #[test]
    fn fresh_root_bootstrap_produces_completed_root_edge_for_positive_budget() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut planner =
            ParallelUctPlanner::new(workers(4), None).expect("valid parallel_uct planner");
        let action = planner
            .search(&mut agent, &[0], 0, 0, 4)
            .expect("positive-horizon parallel_uct search");

        let root = planner.wu_root();
        assert!(action < 2);
        assert_eq!(root.visits, 4);
        assert!(
            root_has_completed_edge(root),
            "positive-budget search on a fresh retained root must complete at least one root edge"
        );
        assert!(
            root.action_edges
                .iter()
                .filter_map(Option::as_ref)
                .all(|edge| edge.o == 0),
            "all incomplete counts must be cleared after the search batch completes"
        );
    }

    #[test]
    fn zero_sample_budget_does_not_bootstrap_or_clone_even_at_zero_horizon() {
        let mut agent = CounterAgent::new_with_horizon(0);
        let mut planner =
            ParallelUctPlanner::new(workers(4), None).expect("valid parallel_uct planner");

        let action = planner
            .search(&mut agent, &[0], 0, 0, 0)
            .expect("zero-sample parallel_uct search should not require positive horizon");
        assert_eq!(action, 0);
        assert_eq!(agent.clone_count.load(Ordering::SeqCst), 0);
        assert_eq!(agent.begin_count.load(Ordering::SeqCst), 0);
        assert_eq!(agent.model_updates.load(Ordering::SeqCst), 0);
        let root = planner.wu_root();
        assert_eq!(root.visits, 0);
        assert!(
            !root_has_completed_edge(root),
            "zero-budget search must not synthesize completed root edges"
        );
    }

    #[test]
    fn positive_budget_search_rejects_zero_horizon_at_api_boundary() {
        let mut agent = CounterAgent::new_with_horizon(0);
        let mut planner =
            ParallelUctPlanner::new(workers(4), None).expect("valid parallel_uct planner");

        let err = planner
            .search(&mut agent, &[0], 0, 0, 1)
            .expect_err("positive-budget zero-horizon search must be rejected");
        assert_eq!(
            err,
            ParallelUctSearchError::PositiveSamplesRequirePositiveHorizon
        );
        assert_eq!(agent.clone_count.load(Ordering::SeqCst), 0);
        assert_eq!(agent.begin_count.load(Ordering::SeqCst), 0);
        assert_eq!(agent.model_updates.load(Ordering::SeqCst), 0);
        let root = planner.wu_root();
        assert_eq!(root.visits, 0);
        assert!(
            !root_has_completed_edge(root),
            "rejected zero-horizon search must leave the retained root untouched"
        );
    }

    #[test]
    fn single_sample_bootstrap_avoids_empty_root_fallback() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut planner =
            ParallelUctPlanner::new(workers(4), None).expect("valid parallel_uct planner");

        let action = planner
            .search(&mut agent, &[0], 0, 0, 1)
            .expect("positive-horizon parallel_uct search");
        assert_eq!(action, 0);
        let root = planner.wu_root();
        assert_eq!(root.visits, 1);
        assert!(
            root_has_completed_edge(root),
            "single-sample retained-root bootstrap must leave one completed root edge"
        );
    }

    #[test]
    fn retained_root_same_percept_branch_reuses_bootstrapped_subtree_skeleton() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut planner =
            ParallelUctPlanner::new(workers(2), None).expect("valid parallel_uct planner");
        let action = planner
            .search(&mut agent, &[0], 0, 0, 1)
            .expect("positive-horizon parallel_uct search");
        let root = planner.wu_root();
        let edge = root.action_edges[action as usize]
            .as_ref()
            .expect("root edge");
        let retained = edge
            .chance()
            .percept_children
            .get(&PerceptOutcome::new(vec![0], 0))
            .expect("retained subtree");
        let retained_id = retained.id;

        let _follow_up = planner
            .search(&mut agent, &[0], 0, action, 0)
            .expect("zero-sample parallel_uct search should not require positive horizon");
        let next_root = planner.wu_root();
        assert_eq!(next_root.id, retained_id);
        assert_eq!(next_root.visits, 1);
    }

    #[test]
    fn prune_to_fresh_root_bootstrap_preserves_positive_budget_invariant() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut planner =
            ParallelUctPlanner::new(workers(2), None).expect("valid parallel_uct planner");
        planner.set_wu_root(WuDecisionNode {
            id: 0,
            visits: 8,
            action_edges: vec![Some(WuActionEdge {
                q: 0.0,
                n: 1,
                o: 0,
                child: ChanceNode {
                    percept_children: HashMap::from([(
                        PerceptOutcome::new(vec![0], 0),
                        WuDecisionNode::new(1),
                    )]),
                },
            })],
        });

        let second_action = planner
            .search(&mut agent, &[0], 0, 0, 1)
            .expect("positive-horizon parallel_uct search");
        assert_eq!(second_action, 0);
        let root = planner.wu_root();
        assert_eq!(root.visits, 1);
        assert!(
            root_has_completed_edge(root),
            "prune-to-fresh-root with positive budget must still complete a root edge"
        );
    }

    #[test]
    fn bootstrap_and_batched_rollouts_use_absolute_task_indices_for_seeding() {
        let mut agent = SeedRecordingAgent::new();
        let mut planner =
            ParallelUctPlanner::new(workers(4), None).expect("valid parallel_uct planner");

        let _action = planner
            .search(&mut agent, &[0], 0, 0, 5)
            .expect("positive-horizon parallel_uct search");

        let planner_seed = 0.25f64.to_bits();
        let expected = (0..5)
            .map(|task_index| planner_task_seed(planner_seed, task_index))
            .collect::<Vec<_>>();
        let seen = agent.recorded_seeds.lock().expect("seed list").clone();
        assert_eq!(seen, expected);
    }

    #[test]
    fn dispatch_rollout_records_root_to_leaf_path_and_updates_each_edge_once() {
        let mut node = two_step_wu_root();
        let mut next_node_id = 13u64;
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };

        let dispatch =
            dispatch_rollout::<WuMode>(&mut node, &mut next_node_id, &mut agent, 3, 1, None);
        assert_eq!(dispatch.remaining_horizon, 1);
        assert_eq!(dispatch.path.len(), 2);
        assert_eq!(dispatch.path[0].parent_node_id, 10);
        assert_eq!(dispatch.path[0].action_idx, 0);
        assert_eq!(dispatch.path[0].child_node_id, 11);
        assert_eq!(dispatch.path[0].outcome, PerceptOutcome::new(vec![0], 0));
        assert_eq!(dispatch.path[1].parent_node_id, 11);
        assert_eq!(dispatch.path[1].action_idx, 0);
        assert_eq!(dispatch.path[1].child_node_id, 12);
        assert_eq!(dispatch.path[1].outcome, PerceptOutcome::new(vec![0], 0));

        let root_edge = node.action_edges[0].as_ref().expect("root edge");
        let child = root_edge
            .chance()
            .percept_children
            .get(&PerceptOutcome::new(vec![0], 0))
            .expect("child node");
        let child_edge = child.action_edges[0].as_ref().expect("child edge");
        assert_eq!(root_edge.o, 1);
        assert_eq!(child_edge.o, 1);
    }

    #[test]
    fn ordinary_dispatch_path_completes_single_rollout_without_residual_incomplete_counts() {
        let mut agent = CounterAgent::new_with_horizon(3);
        let mut planner =
            ParallelUctPlanner::new(workers(1), None).expect("valid parallel_uct planner");
        planner.set_wu_root(retained_parent_for(two_step_wu_root()));

        let action = planner
            .search(&mut agent, &[0], 0, 0, 1)
            .expect("positive-horizon parallel_uct search");
        assert_eq!(action, 0);

        let root = planner.wu_root();
        assert_eq!(root.id, 10);
        let root_edge = root.action_edges[0].as_ref().expect("root edge");
        let child = root_edge
            .chance()
            .percept_children
            .get(&PerceptOutcome::new(vec![0], 0))
            .expect("child node");
        let child_edge = child.action_edges[0].as_ref().expect("child edge");
        assert_eq!(root_edge.o, 0);
        assert_eq!(child_edge.o, 0);
        assert_eq!(root_edge.n, 2);
        assert_eq!(child_edge.n, 2);
    }

    #[test]
    fn bu_thresholding_skips_oversubscribed_edges() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut node = BuDecisionNode::new(0);
        node.visits = 8;
        node.action_edges = vec![
            Some(BuActionEdge {
                q: 0.9,
                n: 4,
                o: 0,
                o_bar: 2.0,
                child: BuChanceNode::default(),
            }),
            Some(BuActionEdge {
                q: 0.1,
                n: 4,
                o: 0,
                o_bar: 0.0,
                child: BuChanceNode::default(),
            }),
        ];

        let selected = select_existing_action::<BuMode>(&node, &mut agent, 1, 2, Some(0.5));
        assert_eq!(
            selected,
            Some(1),
            "BU-UCT should skip edges whose average incomplete count exceeds the threshold"
        );
    }

    #[test]
    fn bu_thresholding_returns_none_when_all_edges_are_oversubscribed() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut node = BuDecisionNode::new(0);
        node.visits = 8;
        node.action_edges = vec![
            Some(BuActionEdge {
                q: 0.9,
                n: 4,
                o: 0,
                o_bar: 2.0,
                child: BuChanceNode::default(),
            }),
            Some(BuActionEdge {
                q: 0.1,
                n: 4,
                o: 0,
                o_bar: 2.0,
                child: BuChanceNode::default(),
            }),
        ];

        let selected = select_existing_action::<BuMode>(&node, &mut agent, 1, 2, Some(0.5));
        assert_eq!(selected, None);
    }

    #[test]
    fn bu_thresholding_stops_at_current_node_when_all_expanded_children_forbidden() {
        let mut node = BuDecisionNode::new(0);
        node.visits = 8;
        node.action_edges = vec![
            Some(BuActionEdge {
                q: 0.9,
                n: 4,
                o: 0,
                o_bar: 2.0,
                child: BuChanceNode::default(),
            }),
            Some(BuActionEdge {
                q: 0.1,
                n: 4,
                o: 0,
                o_bar: 2.0,
                child: BuChanceNode::default(),
            }),
        ];
        let mut next_node_id = 1u64;
        let mut agent = ThresholdProbeAgent {
            num_actions: 2,
            model_updates: 0,
            range_result: 0,
        };

        let dispatch =
            dispatch_rollout::<BuMode>(&mut node, &mut next_node_id, &mut agent, 3, 2, Some(0.5));
        assert!(
            dispatch.path.is_empty(),
            "when every expanded child is threshold-forbidden, BU-core must stop at the current node"
        );
        assert_eq!(dispatch.remaining_horizon, 3);
        assert_eq!(
            agent.model_updates, 0,
            "threshold stop must not traverse or expand a forbidden edge"
        );
    }

    #[test]
    fn bu_threshold_stop_expands_unexpanded_legal_action_synchronously() {
        let mut node = BuDecisionNode::new(0);
        node.visits = 8;
        node.action_edges = vec![
            Some(BuActionEdge {
                q: 0.9,
                n: 4,
                o: 0,
                o_bar: 2.0,
                child: BuChanceNode::default(),
            }),
            Some(BuActionEdge {
                q: 0.1,
                n: 4,
                o: 0,
                o_bar: 2.0,
                child: BuChanceNode::default(),
            }),
            None,
        ];
        let mut next_node_id = 7u64;
        let mut agent = ThresholdProbeAgent {
            num_actions: 3,
            model_updates: 0,
            range_result: 0,
        };

        let dispatch =
            dispatch_rollout::<BuMode>(&mut node, &mut next_node_id, &mut agent, 3, 2, Some(0.5));
        assert_eq!(dispatch.path.len(), 1);
        assert_eq!(dispatch.path[0].action_idx, 2);
        assert_eq!(dispatch.path[0].parent_node_id, 0);
        assert_eq!(dispatch.path[0].child_node_id, 7);
        assert_eq!(dispatch.remaining_horizon, 2);
        assert_eq!(agent.model_updates, 1);
        assert!(
            node.action_edges[2].is_some(),
            "BU-core must synchronously expand an unexpanded legal action after threshold stop"
        );
        let edge = node.action_edges[2].as_ref().expect("expanded edge");
        let child = edge
            .chance()
            .percept_children
            .get(&PerceptOutcome::new(vec![0], 0))
            .expect("expanded child");
        assert_eq!(edge.o, 1);
        assert_eq!(child.id, dispatch.path[0].child_node_id);
    }

    #[test]
    fn bu_thresholding_uses_configured_worker_budget_not_batch_size() {
        let mut planner =
            ParallelUctPlanner::new(workers(4), Some(0.8)).expect("valid parallel_uct planner");
        let retained_root = BuDecisionNode {
            id: 1,
            visits: 8,
            action_edges: vec![Some(BuActionEdge {
                q: 0.2,
                n: 1,
                o: 0,
                o_bar: 1.0,
                child: BuChanceNode::default(),
            })],
        };
        planner.set_bu_root(BuDecisionNode {
            id: 0,
            visits: 8,
            action_edges: vec![Some(BuActionEdge {
                n: 1,
                q: 0.0,
                o: 0,
                o_bar: 0.0,
                child: BuChanceNode {
                    percept_children: HashMap::from([(
                        PerceptOutcome::new(vec![0], 0),
                        retained_root,
                    )]),
                },
            })],
        });
        let mut agent = ThresholdProbeAgent {
            num_actions: 1,
            model_updates: 0,
            range_result: 0,
        };

        let action = planner
            .search(&mut agent, &[0], 0, 0, 1)
            .expect("positive-horizon parallel_uct search");
        assert_eq!(action, 0);
        let root = planner.bu_root();
        let edge = root.action_edges[0].as_ref().expect("root edge");
        assert_eq!(
            edge.n, 2,
            "with configured workers=4 and m_max=0.8, O_bar=1.0 must remain admissible even for samples=1"
        );
    }

    #[test]
    fn bu_grouped_backpropagation_keeps_group_count_constant_for_same_child_origin() {
        let mut edge = BuActionEdge::new();
        let mut epoch_state = BuEpochState::default();
        let key = BuEpochKey {
            parent_node_id: 0,
            action_idx: 0,
            child_node_id: 7,
        };

        edge.update_bu_epoch(key, 1.0, &mut epoch_state);
        assert_eq!(edge.n, 1);
        assert_eq!(edge.q, 1.0);

        edge.update_bu_epoch(key, 0.0, &mut epoch_state);
        assert_eq!(edge.n, 1);
        assert!((edge.q - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bu_batch_updates_q_once_from_same_origin_group_mean() {
        let mut planner =
            ParallelUctPlanner::new(workers(4), Some(0.8)).expect("valid parallel_uct planner");
        planner.set_bu_root(BuDecisionNode {
            id: 0,
            visits: 0,
            action_edges: vec![Some(BuActionEdge {
                q: 0.0,
                n: 0,
                o: 2,
                o_bar: 0.0,
                child: BuChanceNode {
                    percept_children: HashMap::from([(
                        PerceptOutcome::new(vec![0], 0),
                        BuDecisionNode::new(1),
                    )]),
                },
            })],
        });

        let completed = vec![
            CompletedRollout {
                task_index: 1,
                path: vec![step(0, 0, 1, 0, 0)],
                tail_reward: 1.0,
            },
            CompletedRollout {
                task_index: 0,
                path: vec![step(0, 0, 1, 0, 0)],
                tail_reward: 0.0,
            },
        ];
        match &mut planner.state {
            ParallelPlannerState::Bu(runtime) => complete_update_batch(runtime, &completed, 1.0),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }

        let root = planner.bu_root();
        let edge = root.action_edges[0].as_ref().expect("root edge");
        assert_eq!(root.visits, 2);
        assert_eq!(edge.n, 1);
        assert_eq!(edge.o, 0);
        assert!((edge.q - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bu_batch_distinguishes_different_child_origins_at_same_ancestor_edge() {
        let mut planner =
            ParallelUctPlanner::new(workers(4), Some(0.8)).expect("valid parallel_uct planner");
        planner.set_bu_root(BuDecisionNode {
            id: 0,
            visits: 0,
            action_edges: vec![Some(BuActionEdge {
                q: 0.0,
                n: 0,
                o: 2,
                o_bar: 0.0,
                child: BuChanceNode {
                    percept_children: HashMap::from([
                        (PerceptOutcome::new(vec![0], 0), BuDecisionNode::new(1)),
                        (PerceptOutcome::new(vec![1], 0), BuDecisionNode::new(2)),
                    ]),
                },
            })],
        });

        let completed = vec![
            CompletedRollout {
                task_index: 0,
                path: vec![step(0, 0, 1, 0, 0)],
                tail_reward: 1.0,
            },
            CompletedRollout {
                task_index: 1,
                path: vec![step(0, 0, 2, 1, 0)],
                tail_reward: 0.0,
            },
        ];
        match &mut planner.state {
            ParallelPlannerState::Bu(runtime) => complete_update_batch(runtime, &completed, 1.0),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }

        let root = planner.bu_root();
        let edge = root.action_edges[0].as_ref().expect("root edge");
        assert_eq!(edge.n, 2);
        assert!((edge.q - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bu_grouping_is_local_to_each_ancestor_edge() {
        let mut planner =
            ParallelUctPlanner::new(workers(4), Some(0.8)).expect("valid parallel_uct planner");
        planner.set_bu_root(BuDecisionNode {
            id: 0,
            visits: 0,
            action_edges: vec![Some(BuActionEdge {
                q: 0.0,
                n: 0,
                o: 2,
                o_bar: 0.0,
                child: BuChanceNode {
                    percept_children: HashMap::from([(
                        PerceptOutcome::new(vec![0], 0),
                        BuDecisionNode {
                            id: 1,
                            visits: 0,
                            action_edges: vec![Some(BuActionEdge {
                                q: 0.0,
                                n: 0,
                                o: 2,
                                o_bar: 0.0,
                                child: BuChanceNode {
                                    percept_children: HashMap::from([
                                        (PerceptOutcome::new(vec![10], 0), BuDecisionNode::new(2)),
                                        (PerceptOutcome::new(vec![11], 0), BuDecisionNode::new(3)),
                                    ]),
                                },
                            })],
                        },
                    )]),
                },
            })],
        });

        let completed = vec![
            CompletedRollout {
                task_index: 0,
                path: vec![step(0, 0, 1, 0, 0), step(1, 0, 2, 10, 0)],
                tail_reward: 1.0,
            },
            CompletedRollout {
                task_index: 1,
                path: vec![step(0, 0, 1, 0, 0), step(1, 0, 3, 11, 0)],
                tail_reward: 0.0,
            },
        ];
        match &mut planner.state {
            ParallelPlannerState::Bu(runtime) => complete_update_batch(runtime, &completed, 1.0),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }

        let root = planner.bu_root();
        let root_edge = root.action_edges[0].as_ref().expect("root edge");
        let child = root_edge
            .chance()
            .percept_children
            .get(&PerceptOutcome::new(vec![0], 0))
            .expect("child node");
        let child_edge = child.action_edges[0].as_ref().expect("child edge");

        assert_eq!(root_edge.n, 1);
        assert_eq!(child_edge.n, 2);
    }

    #[test]
    fn incomplete_update_tracks_o_bar_recurrence() {
        let mut planner =
            ParallelUctPlanner::new(workers(4), Some(0.8)).expect("valid parallel_uct planner");
        planner.set_bu_root(BuDecisionNode {
            id: 0,
            visits: 0,
            action_edges: vec![Some(BuActionEdge {
                q: 0.0,
                n: 0,
                o: 0,
                o_bar: 0.0,
                child: BuChanceNode {
                    percept_children: HashMap::from([(
                        PerceptOutcome::new(vec![0], 0),
                        BuDecisionNode::new(1),
                    )]),
                },
            })],
        });

        let path = vec![step(0, 0, 1, 0, 0)];
        match &mut planner.state {
            ParallelPlannerState::Bu(runtime) => incomplete_update(runtime, &path),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }
        let root = planner.bu_root();
        let edge = root.action_edges[0].as_ref().expect("edge");
        assert_eq!(edge.o, 1);
        assert!((edge.o_bar - 1.0).abs() < 1e-12);

        match &mut planner.state {
            ParallelPlannerState::Bu(runtime) => incomplete_update(runtime, &path),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }
        let root = planner.bu_root();
        let edge = root.action_edges[0].as_ref().expect("edge");
        assert_eq!(edge.o, 2);
        assert!((edge.o_bar - 1.5).abs() < 1e-12);
    }

    #[test]
    fn bu_thresholding_uses_o_bar_not_live_o_after_completion() {
        let mut planner =
            ParallelUctPlanner::new(workers(4), Some(0.5)).expect("valid parallel_uct planner");
        planner.set_bu_root(BuDecisionNode {
            id: 0,
            visits: 8,
            action_edges: vec![
                Some(BuActionEdge {
                    q: 0.9,
                    n: 4,
                    o: 1,
                    o_bar: 2.0,
                    child: BuChanceNode {
                        percept_children: HashMap::from([(
                            PerceptOutcome::new(vec![0], 0),
                            BuDecisionNode::new(1),
                        )]),
                    },
                }),
                Some(BuActionEdge {
                    q: 0.1,
                    n: 4,
                    o: 0,
                    o_bar: 0.0,
                    child: BuChanceNode::default(),
                }),
            ],
        });

        let completed = vec![CompletedRollout {
            task_index: 0,
            path: vec![step(0, 0, 1, 0, 0)],
            tail_reward: 0.0,
        }];
        match &mut planner.state {
            ParallelPlannerState::Bu(runtime) => complete_update_batch(runtime, &completed, 1.0),
            ParallelPlannerState::Wu(_) => panic!("expected BU planner"),
        }

        let root = planner.bu_root();
        let forbidden = root.action_edges[0].as_ref().expect("forbidden edge");
        assert_eq!(
            forbidden.o, 0,
            "completion must clear the live incomplete count"
        );
        assert!(
            (forbidden.o_bar - 2.0).abs() < 1e-12,
            "BU-core Part 1 keeps the paper-style one-sided o_bar statistic after completion"
        );

        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let selected = select_existing_action::<BuMode>(root, &mut agent, 1, 4, Some(0.5));
        assert_eq!(
            selected,
            Some(1),
            "post-completion BU thresholding must continue to follow o_bar rather than current live o"
        );
    }

    #[test]
    fn bu_bootstrap_uses_singleton_epoch_completion() {
        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let mut planner =
            ParallelUctPlanner::new(workers(4), Some(0.5)).expect("valid parallel_uct planner");

        let action = planner
            .search(&mut agent, &[0], 0, 0, 1)
            .expect("positive-horizon parallel_uct search");
        assert_eq!(action, 0);
        let root = planner.bu_root();
        assert_eq!(root.visits, 1);
        let edge = root.action_edges[0].as_ref().expect("bootstrap edge");
        assert_eq!(edge.n, 1);
        assert_eq!(edge.o, 0);
    }

    #[test]
    fn final_root_choice_uses_completed_q_not_bu_thresholding() {
        let mut planner =
            ParallelUctPlanner::new(workers(2), Some(0.5)).expect("valid parallel_uct planner");
        planner.set_bu_root(BuDecisionNode {
            id: 0,
            visits: 8,
            action_edges: vec![
                Some(BuActionEdge {
                    q: 0.9,
                    n: 4,
                    o: 0,
                    o_bar: 2.0,
                    child: BuChanceNode::default(),
                }),
                Some(BuActionEdge {
                    q: 0.1,
                    n: 4,
                    o: 0,
                    o_bar: 0.0,
                    child: BuChanceNode::default(),
                }),
            ],
        });

        let mut agent = DeterministicRewardAgent {
            last_action: 0,
            emit_reward: false,
        };
        let action = best_action::<BuMode>(Some(planner.bu_root()), &mut agent);
        assert_eq!(action, 0);
    }

    #[test]
    fn parallel_search_is_thread_count_independent() {
        fn run_with_threads(threads: usize) -> (Action, u32, Vec<(u32, u32)>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            pool.install(|| {
                let mut agent = DeterministicRewardAgent {
                    last_action: 0,
                    emit_reward: false,
                };
                let mut planner = ParallelUctPlanner::new(workers(4), Some(0.8))
                    .expect("valid parallel_uct planner");
                let action = planner
                    .search(&mut agent, &[0], 0, 0, 32)
                    .expect("positive-horizon parallel_uct search");
                let root = planner.bu_root();
                let child_stats = root
                    .action_edges
                    .iter()
                    .map(|edge| edge.as_ref().map_or((0, 0), |edge| (edge.n, edge.o)))
                    .collect::<Vec<_>>();
                (action, root.visits, child_stats)
            })
        }

        let one = run_with_threads(1);
        let two = run_with_threads(2);
        let four = run_with_threads(4);

        assert_eq!(one, two);
        assert_eq!(two, four);
    }
}
