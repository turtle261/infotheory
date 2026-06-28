//! GameEngine integration for AIXI environments.
//!
//! This adapter is behind the `aixi-gameengine` feature and keeps core AIXI
//! independent from bundled environments.

use crate::aixi::common::DEFAULT_RANDOM_SEED;
use crate::aixi::common::{Action, ActionAlphabet, PerceptVal, Reward};
use crate::aixi::environment::Environment;
use crate::spec::BuiltinEnvironmentSpec;
use gameengine::GameAuthoring;
#[cfg(feature = "aixi-gameengine-physics")]
use gameengine::builtin::Platformer;
use gameengine::builtin::{
    BiasedCoinFlip, BiasedCoinFlipConfig, BiasedRockPaperScissor, Blackjack, ExtendedTiger,
    KuhnPoker, TicTacToe,
};
use gameengine::{ActionToken, AixiEnvironment as GameEngineAixiEnvironment, DefaultEnvironment};

/// Errors surfaced by GameEngine-backed AIXI environment construction/runtime reset.
#[derive(Debug)]
#[non_exhaustive]
pub enum GameEngineEnvironmentError {
    /// Underlying GameEngine reset failed.
    ResetFailed(String),
    /// Environment produced an observation stream that violates compact spec shape.
    PerceptStreamLengthMismatch {
        /// Actual observation stream length emitted by the environment.
        actual: usize,
        /// Expected observation stream length from compact spec.
        expected: usize,
    },
    /// Coin-flip bias violates the domain invariant.
    InvalidCoinFlipBias {
        /// Configured coin-flip head numerator.
        head_numerator: u64,
        /// Configured coin-flip head denominator.
        head_denominator: u64,
    },
    /// The internal tuner bridge was requested as a standalone runtime environment.
    TunerBridgeOnly,
    /// Requested builtin requires an optional feature that is disabled.
    MissingFeature {
        /// Builtin environment that was requested.
        builtin: BuiltinEnvironmentSpec,
        /// Missing feature gate required for `builtin`.
        feature: &'static str,
    },
}

impl std::fmt::Display for GameEngineEnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResetFailed(err) => {
                write!(f, "failed to reset GameEngine environment: {err}")
            }
            Self::PerceptStreamLengthMismatch { actual, expected } => write!(
                f,
                "GameEngine percept stream length {actual} does not match compact spec length {expected}"
            ),
            Self::InvalidCoinFlipBias {
                head_numerator,
                head_denominator,
            } => write!(
                f,
                "invalid coin-flip bias: expected 0 <= numerator <= denominator (got {head_numerator}/{head_denominator})"
            ),
            Self::TunerBridgeOnly => f.write_str(
                "builtin environment 'tuner_bridge' is an internal tuner planner bridge and cannot be run as a standalone GameEngine environment",
            ),
            Self::MissingFeature { builtin, feature } => write!(
                f,
                "builtin environment '{}' requires feature '{}'",
                builtin.canonical_name(),
                feature
            ),
        }
    }
}

impl std::error::Error for GameEngineEnvironmentError {}

/// Generic adapter from a GameEngine AIXI environment to Infotheory's AIXI trait.
pub struct GameEngineEnvironment<E, const MAX_WORDS: usize>
where
    E: GameEngineAixiEnvironment<MAX_WORDS>,
{
    env: E,
    spec: gameengine::CompactSpec,
    observation_stream: Vec<PerceptVal>,
    observation: PerceptVal,
    reward: Reward,
    finished: bool,
}

impl<E, const MAX_WORDS: usize> GameEngineEnvironment<E, MAX_WORDS>
where
    E: GameEngineAixiEnvironment<MAX_WORDS>,
{
    /// Builds an adapter and resets the underlying environment with `seed`.
    pub fn from_environment(
        mut env: E,
        spec: gameengine::CompactSpec,
        seed: u64,
    ) -> Result<Self, GameEngineEnvironmentError> {
        let initial = env
            .reset_seed(seed)
            .map_err(|err| GameEngineEnvironmentError::ResetFailed(err.to_string()))?;
        let mut adapter = Self {
            env,
            spec,
            observation_stream: Vec::with_capacity(spec.observation_stream_len),
            observation: 0,
            reward: 0,
            finished: false,
        };
        adapter.apply_percept(initial)?;
        Ok(adapter)
    }

    fn apply_percept(
        &mut self,
        percept: gameengine::Percept<MAX_WORDS>,
    ) -> Result<(), GameEngineEnvironmentError> {
        let words = percept.observation_bits.words();
        if words.len() != self.spec.observation_stream_len {
            return Err(GameEngineEnvironmentError::PerceptStreamLengthMismatch {
                actual: words.len(),
                expected: self.spec.observation_stream_len,
            });
        }

        self.observation_stream.clear();
        self.observation_stream.extend(words.iter().copied());
        self.observation = self.observation_stream.first().copied().unwrap_or(0);
        self.reward = percept.reward.raw;
        self.finished = percept.terminated;
        Ok(())
    }

    fn reset_with_seed(&mut self, seed: u64) -> Result<(), GameEngineEnvironmentError> {
        let percept = self
            .env
            .reset_seed(seed)
            .map_err(|err| GameEngineEnvironmentError::ResetFailed(err.to_string()))?;
        self.apply_percept(percept)
    }
}

impl<E, const MAX_WORDS: usize> Environment for GameEngineEnvironment<E, MAX_WORDS>
where
    E: GameEngineAixiEnvironment<MAX_WORDS>,
{
    fn perform_action(&mut self, action: Action) {
        if self.finished {
            return;
        }

        let token = match ActionToken::try_new(action, self.spec.action_count) {
            Ok(token) => token,
            Err(_) => {
                self.reward = self.spec.min_reward;
                return;
            }
        };

        match self.env.step(token) {
            Ok(percept) => {
                if self.apply_percept(percept).is_err() {
                    self.finished = true;
                    self.reward = self.spec.min_reward;
                }
            }
            Err(_) => {
                self.finished = true;
                self.reward = self.spec.min_reward;
            }
        }
    }

    fn get_observation(&self) -> PerceptVal {
        self.observation
    }

    fn drain_observations(&mut self) -> Vec<PerceptVal> {
        self.observation_stream.clone()
    }

    fn get_reward(&self) -> Reward {
        self.reward
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn get_observation_bits(&self) -> usize {
        self.spec.observation_bits as usize
    }

    fn get_reward_bits(&self) -> usize {
        self.spec.reward_bits as usize
    }

    fn get_action_bits(&self) -> usize {
        self.spec.action_bits() as usize
    }

    fn set_random_seed(&mut self, seed: u64) {
        if self.reset_with_seed(seed).is_err() {
            self.finished = true;
            self.reward = self.spec.min_reward;
        }
    }

    fn get_num_actions(&self) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(self.spec.action_count as usize)
            .expect("gameengine environments must expose a non-empty action alphabet")
    }

    fn max_reward(&self) -> Reward {
        self.spec.max_reward
    }

    fn min_reward(&self) -> Reward {
        self.spec.min_reward
    }
}

type TicTacToeEnvironment = GameEngineEnvironment<DefaultEnvironment<TicTacToe, 1>, 1>;
type BlackjackEnvironment = GameEngineEnvironment<DefaultEnvironment<Blackjack, 4>, 4>;
type CoinFlipEnvironment = GameEngineEnvironment<DefaultEnvironment<BiasedCoinFlip, 1>, 1>;
type BiasedRpsEnvironment = GameEngineEnvironment<DefaultEnvironment<BiasedRockPaperScissor, 1>, 1>;
type KuhnPokerEnvironment = GameEngineEnvironment<DefaultEnvironment<KuhnPoker, 1>, 1>;
type ExtendedTigerEnvironment = GameEngineEnvironment<DefaultEnvironment<ExtendedTiger, 1>, 1>;
#[cfg(feature = "aixi-gameengine-physics")]
type PlatformerEnvironment = GameEngineEnvironment<DefaultEnvironment<Platformer, 1>, 1>;

fn build_coin_flip_environment_from_config(
    config: BiasedCoinFlipConfig,
    seed: u64,
) -> Result<Box<dyn Environment>, GameEngineEnvironmentError> {
    if !config.invariant() {
        return Err(GameEngineEnvironmentError::InvalidCoinFlipBias {
            head_numerator: config.head_numerator,
            head_denominator: config.head_denominator,
        });
    }
    let game = BiasedCoinFlip::new(config);
    let spec = game.compact_spec();
    let env = DefaultEnvironment::<BiasedCoinFlip, 1>::new_for_agent(game, seed, 0);
    Ok(Box::new(CoinFlipEnvironment::from_environment(
        env, spec, seed,
    )?))
}

/// Builds a biased coin-flip environment from Bernoulli head probability parts.
pub fn build_coin_flip_environment(
    head_numerator: u64,
    head_denominator: u64,
    seed: u64,
) -> Result<Box<dyn Environment>, GameEngineEnvironmentError> {
    build_coin_flip_environment_from_config(
        BiasedCoinFlipConfig {
            head_numerator,
            head_denominator,
        },
        seed,
    )
}

/// Builds a boxed AIXI environment from the canonical builtin enum.
pub fn build_builtin_environment(
    builtin: BuiltinEnvironmentSpec,
) -> Result<Box<dyn Environment>, GameEngineEnvironmentError> {
    build_builtin_environment_with_seed(builtin, DEFAULT_RANDOM_SEED)
}

/// Builds a boxed AIXI environment from the canonical builtin enum and explicit seed.
pub fn build_builtin_environment_with_seed(
    builtin: BuiltinEnvironmentSpec,
    seed: u64,
) -> Result<Box<dyn Environment>, GameEngineEnvironmentError> {
    match builtin {
        BuiltinEnvironmentSpec::TunerBridge => Err(GameEngineEnvironmentError::TunerBridgeOnly),
        BuiltinEnvironmentSpec::CoinFlip => {
            build_coin_flip_environment_from_config(BiasedCoinFlipConfig::default(), seed)
        }
        BuiltinEnvironmentSpec::BiasedRockPaperScissor => {
            let game = BiasedRockPaperScissor;
            let spec = game.compact_spec();
            let env = DefaultEnvironment::<BiasedRockPaperScissor, 1>::new_for_agent(game, seed, 0);
            Ok(Box::new(BiasedRpsEnvironment::from_environment(
                env, spec, seed,
            )?))
        }
        BuiltinEnvironmentSpec::KuhnPoker => {
            let game = KuhnPoker;
            let spec = game.compact_spec();
            let env = DefaultEnvironment::<KuhnPoker, 1>::new_for_agent(game, seed, 0);
            Ok(Box::new(KuhnPokerEnvironment::from_environment(
                env, spec, seed,
            )?))
        }
        BuiltinEnvironmentSpec::ExtendedTiger => {
            let game = ExtendedTiger;
            let spec = game.compact_spec();
            let env = DefaultEnvironment::<ExtendedTiger, 1>::new_for_agent(game, seed, 0);
            Ok(Box::new(ExtendedTigerEnvironment::from_environment(
                env, spec, seed,
            )?))
        }
        BuiltinEnvironmentSpec::TicTacToe => {
            let game = TicTacToe;
            let spec = game.compact_spec();
            let env = DefaultEnvironment::<TicTacToe, 1>::new_for_agent(game, seed, 0);
            Ok(Box::new(TicTacToeEnvironment::from_environment(
                env, spec, seed,
            )?))
        }
        BuiltinEnvironmentSpec::Blackjack => {
            let game = Blackjack;
            let spec = game.compact_spec();
            let env = DefaultEnvironment::<Blackjack, 4>::new_for_agent(game, seed, 0);
            Ok(Box::new(BlackjackEnvironment::from_environment(
                env, spec, seed,
            )?))
        }
        #[cfg(feature = "aixi-gameengine-physics")]
        BuiltinEnvironmentSpec::Platformer => {
            let game = Platformer::default();
            let spec = game.compact_spec();
            let env = DefaultEnvironment::<Platformer, 1>::new_for_agent(game, seed, 0);
            Ok(Box::new(PlatformerEnvironment::from_environment(
                env, spec, seed,
            )?))
        }
        #[cfg(not(feature = "aixi-gameengine-physics"))]
        BuiltinEnvironmentSpec::Platformer => Err(GameEngineEnvironmentError::MissingFeature {
            builtin: BuiltinEnvironmentSpec::Platformer,
            feature: "aixi-gameengine-physics",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configurable_coin_flip_bias_controls_rewards_and_reseed() {
        let mut heads = build_coin_flip_environment(1, 1, 7).expect("valid heads-only coin flip");
        heads.perform_action(1);
        assert_eq!(heads.get_observation(), 1);
        assert_eq!(heads.get_reward(), 1);
        heads.set_random_seed(99);
        heads.perform_action(0);
        assert_eq!(heads.get_observation(), 1);
        assert_eq!(heads.get_reward(), 0);

        let mut tails = build_coin_flip_environment(0, 1, 7).expect("valid tails-only coin flip");
        tails.perform_action(0);
        assert_eq!(tails.get_observation(), 0);
        assert_eq!(tails.get_reward(), 1);
    }

    #[test]
    fn invalid_coin_flip_bias_is_rejected() {
        let err = match build_coin_flip_environment(2, 1, 0) {
            Ok(_) => panic!("invalid ratio must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            GameEngineEnvironmentError::InvalidCoinFlipBias {
                head_numerator: 2,
                head_denominator: 1,
            }
        ));

        let err = match build_coin_flip_environment(0, 0, 0) {
            Ok(_) => panic!("zero denominator must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            GameEngineEnvironmentError::InvalidCoinFlipBias {
                head_numerator: 0,
                head_denominator: 0,
            }
        ));
    }

    #[test]
    fn default_builtin_environment_matches_explicit_default_seed() {
        let mut implicit =
            build_builtin_environment(BuiltinEnvironmentSpec::CoinFlip).expect("builtin env");
        let mut explicit = build_builtin_environment_with_seed(BuiltinEnvironmentSpec::CoinFlip, 0)
            .expect("seeded builtin env");

        let mut implicit_trace = Vec::new();
        let mut explicit_trace = Vec::new();
        for &action in &[0u64, 1, 1, 0, 1, 0, 0, 1] {
            implicit.perform_action(action);
            explicit.perform_action(action);
            implicit_trace.push((implicit.get_observation(), implicit.get_reward()));
            explicit_trace.push((explicit.get_observation(), explicit.get_reward()));
        }

        assert_eq!(implicit_trace, explicit_trace);
    }
}
