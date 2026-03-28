//! Standard benchmark environments for AIXI.
//!
//! This module provides a set of environments for testing and evaluating
//! AIXI agents. Each environment implements the `Environment` trait,
//! providing a consistent interface for interaction.

use crate::aixi::common::{Action, PerceptVal, RandomGenerator, Reward};

/// Interface for an agent's environment.
///
/// An environment consumes actions from the agent and produces percepts
/// (observations and rewards) in response.
pub trait Environment {
    /// Executes an action in the environment and updates its internal state.
    fn perform_action(&mut self, action: Action);

    /// Returns the current observation produced by the environment.
    fn get_observation(&self) -> PerceptVal;

    /// Returns a stream of observation symbols produced by the last action.
    ///
    /// Default behavior is a single observation.
    fn drain_observations(&mut self) -> Vec<PerceptVal> {
        vec![self.get_observation()]
    }

    /// Returns the current reward produced by the environment.
    fn get_reward(&self) -> Reward;

    /// Returns true if the environment has reached a terminal state.
    fn is_finished(&self) -> bool;

    /// Returns the number of bits used to encode observations in this environment.
    fn get_observation_bits(&self) -> usize;

    /// Returns the number of bits used to encode rewards in this environment.
    fn get_reward_bits(&self) -> usize;

    /// Returns the number of bits required to represent all possible actions.
    fn get_action_bits(&self) -> usize;

    /// Reseed the environment RNG for deterministic, reproducible runs.
    ///
    /// Deterministic environments can ignore this. Stochastic environments
    /// should reseed and reset any stochastic state so the initial percept
    /// sequence is reproducible from `seed`.
    fn set_random_seed(&mut self, _seed: u64) {}

    /// Returns the total number of valid actions available.
    fn get_num_actions(&self) -> usize {
        1 << self.get_action_bits()
    }

    /// Returns the maximum possible reward value in this environment.
    fn max_reward(&self) -> Reward {
        let bits = self.get_reward_bits();
        if bits == 0 {
            return 0;
        }
        // Prevent overflow for bits >= 64
        if bits >= 64 {
            i64::MAX
        } else {
            (1i64 << (bits - 1)) - 1
        }
    }

    /// Returns the minimum possible reward value in this environment.
    fn min_reward(&self) -> Reward {
        let bits = self.get_reward_bits();
        if bits == 0 {
            return 0;
        }
        // Prevent overflow for bits >= 64
        if bits >= 64 {
            i64::MIN
        } else {
            -(1i64 << (bits - 1))
        }
    }
}

/// A simple biased coin flip environment.
///
/// The agent predicts the outcome of a coin flip. Correct predictions
/// result in a reward of 1, otherwise 0.
pub struct CoinFlip {
    /// Probability of the coin landing heads (1).
    p: f64,
    /// Current observation (coin face).
    obs: PerceptVal,
    /// Last reward received.
    rew: Reward,
    /// Internal RNG.
    rng: RandomGenerator,
}

impl CoinFlip {
    /// Creates a new `CoinFlip` environment with bias `p`.
    pub fn new(p: f64) -> Self {
        Self::new_with_seed(p, None)
    }

    /// Creates a new `CoinFlip` environment with optional deterministic seed.
    pub fn new_with_seed(p: f64, seed: Option<u64>) -> Self {
        let mut env = Self {
            p,
            obs: 0,
            rew: 0,
            rng: seed.map(RandomGenerator::from_seed).unwrap_or_default(),
        };
        // Initial observation
        env.gen_next();
        env
    }

    fn gen_next(&mut self) {
        self.obs = if self.rng.gen_bool(self.p) { 1 } else { 0 };
    }
}

impl Environment for CoinFlip {
    fn perform_action(&mut self, action: Action) {
        self.gen_next();
        self.rew = if action == self.obs { 1 } else { 0 };
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }
    fn get_reward(&self) -> Reward {
        self.rew
    }
    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        1
    }
    fn get_reward_bits(&self) -> usize {
        1
    }

    fn min_reward(&self) -> Reward {
        0
    }

    fn max_reward(&self) -> Reward {
        1
    }
    fn get_action_bits(&self) -> usize {
        1
    }

    fn set_random_seed(&mut self, seed: u64) {
        self.rng = RandomGenerator::from_seed(seed);
        self.rew = 0;
        self.gen_next();
    }
}

/// A synthetic environment for testing CTW performance.
///
/// Generates a deterministic sequence designed to be perfectly
/// predictable by a sufficiently deep Context Tree.
pub struct CtwTest {
    cycle: usize,
    last_action: Action,
    obs: PerceptVal,
    rew: Reward,
}

impl CtwTest {
    /// Creates a new `CtwTest` environment.
    pub fn new() -> Self {
        Self {
            cycle: 0,
            last_action: 0,
            obs: 0,
            rew: 0,
        }
    }
}

impl Default for CtwTest {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CtwTest {
    fn perform_action(&mut self, action: Action) {
        if self.cycle == 0 {
            self.obs = 0;
            self.rew = if self.obs == action { 1 } else { 0 };
        } else {
            self.obs = (self.last_action + 1) % 2;
            self.rew = if self.obs == action { 1 } else { 0 };
        }
        self.last_action = action;
        self.cycle += 1;
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }
    fn get_reward(&self) -> Reward {
        self.rew
    }
    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        1
    }
    fn get_reward_bits(&self) -> usize {
        1
    }

    fn min_reward(&self) -> Reward {
        0
    }

    fn max_reward(&self) -> Reward {
        1
    }
    fn get_action_bits(&self) -> usize {
        1
    }
}

/// A Rock-Paper-Scissors environment with a biased opponent.
///
/// The opponent plays randomly unless it wins a round, in which case
/// it repeats its winning action.
pub struct BiasedRockPaperScissor {
    obs: PerceptVal,
    rew: Reward,
    rng: RandomGenerator,
}

impl BiasedRockPaperScissor {
    /// Creates a new `BiasedRockPaperScissor` environment.
    pub fn new() -> Self {
        Self::new_with_seed(None)
    }

    /// Creates a new `BiasedRockPaperScissor` environment with optional seed.
    pub fn new_with_seed(seed: Option<u64>) -> Self {
        Self {
            // Match reference MC-AIXI/PyAIXI initial percept: non-rock.
            obs: 1,
            rew: 0,
            rng: seed.map(RandomGenerator::from_seed).unwrap_or_default(),
        }
    }
}

impl Default for BiasedRockPaperScissor {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for BiasedRockPaperScissor {
    fn perform_action(&mut self, action: Action) {
        // action 0: Rock, 1: Paper, 2: Scissors
        // Match reference MC-AIXI/PyAIXI bias: repeat rock iff opponent won
        // the previous round by playing rock.
        let opponent_action = if self.obs == 0 && self.rew == -1 {
            0
        } else {
            let r = self.rng.gen_f64();
            if r < 1.0 / 3.0 {
                0
            } else if r < 2.0 / 3.0 {
                1
            } else {
                2
            }
        };

        // Determine Outcome
        if opponent_action == action {
            self.rew = 0; // Draw
        } else if (opponent_action == 0 && action == 1)
            || (opponent_action == 1 && action == 2)
            || (opponent_action == 2 && action == 0)
        {
            self.rew = 1; // Win
        } else {
            self.rew = -1; // Loss
        }
        self.obs = opponent_action as PerceptVal;
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }
    fn get_reward(&self) -> Reward {
        self.rew
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

    fn min_reward(&self) -> Reward {
        -1
    }

    fn max_reward(&self) -> Reward {
        1
    }
    fn get_action_bits(&self) -> usize {
        2
    }
    fn get_num_actions(&self) -> usize {
        3
    }

    fn set_random_seed(&mut self, seed: u64) {
        self.rng = RandomGenerator::from_seed(seed);
        // Match reference initial condition after reseed.
        self.obs = 1;
        self.rew = 0;
    }
}

/// A more complex version of the classic Tiger problem.
///
/// Includes states for sitting and standing, with different rewards
/// and transition probabilities.
pub struct ExtendedTiger {
    state: usize, // 0: sitting, 1: standing
    tiger_door: usize,
    gold_door: usize,
    obs: PerceptVal,
    rew: Reward,
    rng: RandomGenerator,
}

impl ExtendedTiger {
    /// Creates a new `ExtendedTiger` environment.
    pub fn new() -> Self {
        let mut rng = RandomGenerator::new();
        let gold_door = if rng.gen_bool(0.5) { 1 } else { 2 };
        let tiger_door = if gold_door == 1 { 2 } else { 3 };

        Self {
            state: 0,
            gold_door,
            tiger_door,
            obs: 0,
            rew: 0,
            rng,
        }
    }

    fn reset_doors(&mut self) {
        self.gold_door = if self.rng.gen_bool(0.5) { 1 } else { 2 };
        self.tiger_door = if self.gold_door == 1 { 2 } else { 3 };
    }
}

impl Default for ExtendedTiger {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for ExtendedTiger {
    fn perform_action(&mut self, action: Action) {
        // Actions: 0: Stand, 1: Listen, 2: Open 1, 3: Open 2
        match action {
            0 => {
                // Stand
                if self.state == 1 {
                    self.rew = -1;
                } else {
                    self.state = 1;
                    self.rew = -1;
                    if self.obs < 4 {
                        self.obs += 4;
                    }
                }
            }
            1 => {
                // Listen
                if self.state == 1 || self.obs != 0 {
                    self.rew = -1;
                    self.obs = 0;
                } else {
                    self.obs = if self.rng.gen_bool(0.85) {
                        self.tiger_door as PerceptVal
                    } else {
                        self.gold_door as PerceptVal
                    };
                    self.rew = -1;
                }
            }
            2 => {
                // Open 1
                if self.state == 0 {
                    self.rew = -100;
                } else {
                    self.rew = if self.gold_door == 1 { 30 } else { -100 };
                    self.obs = 0;
                    self.state = 0;
                    self.reset_doors();
                }
            }
            3 => {
                // Open 2
                if self.state == 0 {
                    self.rew = -100;
                } else {
                    self.rew = if self.gold_door == 2 { 30 } else { -100 };
                    self.obs = 0;
                    self.state = 0;
                    self.reset_doors();
                }
            }
            _ => {
                self.rew = -100;
            }
        }
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }
    fn get_reward(&self) -> Reward {
        self.rew
    }
    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        3
    }
    fn get_reward_bits(&self) -> usize {
        8
    }

    fn min_reward(&self) -> Reward {
        -100
    }

    fn max_reward(&self) -> Reward {
        30
    }
    fn get_action_bits(&self) -> usize {
        2
    }
    fn get_num_actions(&self) -> usize {
        4
    }

    fn set_random_seed(&mut self, seed: u64) {
        self.rng = RandomGenerator::from_seed(seed);
        self.state = 0;
        self.obs = 0;
        self.rew = 0;
        self.reset_doors();
    }
}

/// A standard Tic-Tac-Toe environment against a random opponent.
pub struct TicTacToe {
    board: [i8; 9], // 0: empty, 1: agent, -1: opponent.
    open_squares: Vec<usize>,
    state: u64,
    obs: PerceptVal,
    rew: Reward,
    rng: RandomGenerator,
}

impl TicTacToe {
    /// Creates a new `TicTacToe` environment.
    pub fn new() -> Self {
        Self {
            board: [0; 9],
            open_squares: (0..9).collect(),
            state: 0,
            obs: 0,
            rew: 0,
            rng: RandomGenerator::new(),
        }
    }

    fn reset_game(&mut self) {
        self.board = [0; 9];
        self.open_squares = (0..9).collect();
        self.state = 0;
    }

    fn check_win(&self, player: i8) -> bool {
        let b = self.board;
        let wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8), // Rows
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8), // Cols
            (0, 4, 8),
            (2, 4, 6), // Diags
        ];
        for &(x, y, z) in &wins {
            if b[x] == player && b[y] == player && b[z] == player {
                return true;
            }
        }
        false
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for TicTacToe {
    fn perform_action(&mut self, action: Action) {
        if action >= 9 {
            self.rew = -3;
            self.obs = self.state as PerceptVal;
            return;
        }

        if self.board[action as usize] != 0 {
            // Illegal move
            self.rew = -3;
        } else {
            // Agent move (1)
            self.state += 1 << (2 * action);
            self.board[action as usize] = 1;

            // Remove from open
            if let Some(pos) = self.open_squares.iter().position(|&x| x == action as usize) {
                self.open_squares.remove(pos);
            }

            self.rew = 0;

            if self.check_win(1) {
                // Agent won
                self.reset_game();
                self.rew = 2;
            } else if self.open_squares.is_empty() {
                // Draw
                self.reset_game();
                self.rew = 1;
            } else {
                // Opponent move (-1, mapped to 2 in base-4)

                // Shuffle open squares
                let n = self.open_squares.len();
                if n > 0 {
                    let idx = self.rng.gen_range(n);
                    let opponent_move = self.open_squares[idx];

                    self.state += 2 << (2 * opponent_move);
                    self.board[opponent_move] = -1;

                    self.open_squares.remove(idx);

                    if self.check_win(-1) {
                        // Opponent won
                        self.reset_game();
                        self.rew = -2;
                    } else if self.open_squares.is_empty() {
                        self.reset_game();
                        self.rew = 1;
                    }
                }
            }
        }
        self.obs = self.state as PerceptVal;
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }
    fn get_reward(&self) -> Reward {
        self.rew
    }
    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        18
    } // 9 squares * 2 bits
    fn get_reward_bits(&self) -> usize {
        3
    }
    fn min_reward(&self) -> Reward {
        -3
    }
    fn max_reward(&self) -> Reward {
        2
    }
    fn get_action_bits(&self) -> usize {
        4
    }
    fn get_num_actions(&self) -> usize {
        9
    }

    fn set_random_seed(&mut self, seed: u64) {
        self.rng = RandomGenerator::from_seed(seed);
        self.reset_game();
        self.obs = 0;
        self.rew = 0;
    }
}

/// A 2-player imperfect information game: Kuhn Poker.
///
/// The agent plays against a Nash-optimized opponent in a simplified
/// 3-card poker game.
pub struct KuhnPoker {
    opponent_card: usize, // 0:J, 1:Q, 2:K
    agent_card: usize,
    opponent_action: usize, // 0: bet, 1: pass
    obs: PerceptVal,
    rew: Reward,
    rng: RandomGenerator,
}

impl KuhnPoker {
    /// Creates a new `KuhnPoker` environment.
    pub fn new() -> Self {
        Self::new_with_seed(None)
    }

    /// Creates a new `KuhnPoker` environment with optional deterministic seed.
    pub fn new_with_seed(seed: Option<u64>) -> Self {
        let mut env = Self {
            opponent_card: 0,
            agent_card: 0,
            opponent_action: 0,
            obs: 0,
            rew: 0,
            rng: seed.map(RandomGenerator::from_seed).unwrap_or_default(),
        };
        env.reset_game();
        env
    }

    #[inline]
    fn random_card(&mut self) -> usize {
        self.rng.gen_range(3)
    }

    fn reset_game(&mut self) {
        // Card encoding matches the reference implementations:
        // 0=Jack, 1=Queen, 2=King.
        self.agent_card = self.random_card();
        self.opponent_card = self.agent_card;
        while self.opponent_card == self.agent_card {
            self.opponent_card = self.random_card();
        }

        const ACTION_BET: usize = 0;
        const ACTION_PASS: usize = 1;
        const BET_PROB_KING: f64 = 0.7;
        const BET_PROB_JACK: f64 = BET_PROB_KING / 3.0;

        // Opponent first action (reference Nash policy).
        self.opponent_action = if self.opponent_card == 0 {
            if self.rng.gen_bool(BET_PROB_JACK) {
                ACTION_BET
            } else {
                ACTION_PASS
            }
        } else if self.opponent_card == 1 {
            ACTION_PASS
        } else if self.rng.gen_bool(BET_PROB_KING) {
            ACTION_BET
        } else {
            ACTION_PASS
        };

        // Observation encoding matches C++/PyAIXI:
        // observation = agent_card + (opponent_pass ? 4 : 0)
        let action_code = if self.opponent_action == ACTION_PASS {
            4
        } else {
            0
        };
        let card_code = self.agent_card;
        self.obs = (action_code + card_code) as PerceptVal;
    }
}

impl Default for KuhnPoker {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for KuhnPoker {
    fn perform_action(&mut self, action: Action) {
        const ACTION_BET: usize = 0;
        const ACTION_PASS: usize = 1;

        // Reference reward levels are encoded as {0,1,3,4}. We emit the
        // offset-removed values {-2,-1,1,2} for direct comparability.
        const R_BET_LOSS: Reward = -2;
        const R_PASS_LOSS: Reward = -1;
        const R_PASS_WIN: Reward = 1;
        const R_BET_WIN: Reward = 2;

        const BET_PROB_KING: f64 = 0.7;
        const BET_PROB_QUEEN: f64 = (1.0 + BET_PROB_KING) / 3.0;

        if action > 1 {
            self.rew = R_BET_LOSS;
            self.reset_game();
            return;
        }

        // If the agent did not call an opponent bet, the agent loses.
        if action as usize == ACTION_PASS && self.opponent_action == ACTION_BET {
            self.rew = R_PASS_LOSS;
            self.reset_game();
            return;
        }

        // If opponent passed and agent bet, opponent may reconsider.
        if action as usize == ACTION_BET && self.opponent_action == ACTION_PASS {
            if self.opponent_card == 1 && self.rng.gen_bool(BET_PROB_QUEEN) {
                self.opponent_action = ACTION_BET;
            } else if self.opponent_card == 2 {
                self.opponent_action = ACTION_BET;
            } else {
                self.rew = R_PASS_WIN;
                self.reset_game();
                return;
            }
        }

        // Showdown.
        let agent_wins =
            self.opponent_card == 0 || (self.opponent_card == 1 && self.agent_card == 2);
        if agent_wins {
            self.rew = if self.opponent_action == ACTION_BET {
                R_BET_WIN
            } else {
                R_PASS_WIN
            };
        } else {
            self.rew = if action as usize == ACTION_BET {
                R_BET_LOSS
            } else {
                R_PASS_LOSS
            };
        }
        self.reset_game();
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }
    fn get_reward(&self) -> Reward {
        self.rew
    }
    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        3
    }
    fn get_reward_bits(&self) -> usize {
        3
    }

    fn min_reward(&self) -> Reward {
        -2
    }

    fn max_reward(&self) -> Reward {
        2
    }
    fn get_action_bits(&self) -> usize {
        1
    } // 0 or 1
    fn get_num_actions(&self) -> usize {
        2
    }

    fn set_random_seed(&mut self, seed: u64) {
        self.rng = RandomGenerator::from_seed(seed);
        self.rew = 0;
        self.reset_game();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tictactoe_illegal_move_preserves_state_and_penalizes() {
        let mut env = TicTacToe::new();
        env.set_random_seed(7);

        env.perform_action(0);
        let occupied_state = env.get_observation();
        assert_ne!(occupied_state, 0, "first move should change the board");

        env.perform_action(0);
        assert_eq!(
            env.get_reward(),
            -3,
            "illegal move should incur the documented penalty"
        );
        assert_eq!(
            env.get_observation(),
            occupied_state,
            "illegal move should not mutate the board state",
        );
    }
}
