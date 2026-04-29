//! Common types and utilities for the AIXI implementation.

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};

/// Represents a single bit (0 or 1) in the agent's interaction history.
pub type Symbol = bool;

/// A list of symbols, used to represent encoded observations, rewards, or actions.
pub type SymbolList = Vec<Symbol>;

/// Cardinality of an agent/environment action alphabet `|A|`.
///
/// This domain type makes the core AIXI/AIQI invariant explicit: planners,
/// environments, and simulator shims operate over a non-empty action alphabet.
/// The value `0` is therefore unrepresentable once a value has crossed an API
/// boundary into the validated internal domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ActionAlphabet(NonZeroUsize);

impl ActionAlphabet {
    /// Construct an action alphabet cardinality from a known non-zero value.
    pub const fn new(value: NonZeroUsize) -> Self {
        Self(value)
    }

    /// Validate and construct an action alphabet cardinality from a raw `usize`.
    pub fn try_from_usize(value: usize) -> Result<Self, ZeroActionAlphabetError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(ZeroActionAlphabetError)
    }

    /// Return the underlying cardinality.
    pub const fn get(self) -> usize {
        self.0.get()
    }

    /// Return the minimum bit width required to encode the action alphabet.
    pub fn action_bits(self) -> usize {
        bits_for_cardinality(self.get())
    }
}

impl TryFrom<usize> for ActionAlphabet {
    type Error = ZeroActionAlphabetError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::try_from_usize(value)
    }
}

impl fmt::Display for ActionAlphabet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

/// Error returned when attempting to construct an empty action alphabet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZeroActionAlphabetError;

impl fmt::Display for ZeroActionAlphabetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("action alphabet cardinality must be >= 1")
    }
}

impl Error for ZeroActionAlphabetError {}

/// Represents an action that the agent can perform.
pub type Action = u64;

/// Represents a reward received by the agent from the environment.
pub type Reward = i64;

/// Shared default seed for deterministic AIXI/AIQI planner-runtime behavior.
pub const DEFAULT_RANDOM_SEED: u64 = 0;

/// Salt used to derive exploration RNG streams from the planner seed.
pub const EXPLORE_RANDOM_SALT: u64 = 0x4558_504c_4f52_455f;

/// Resolve an optional planner/runtime seed to the canonical deterministic seed.
#[inline]
pub fn resolve_random_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or(DEFAULT_RANDOM_SEED)
}

/// A generic value for a percept component (either an observation or a reward).
pub type PerceptVal = u64;

/// Strategy for mapping an observation stream into a single percept key for tree search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ObservationKeyMode {
    /// Use the full observation stream as the key (paper-accurate expectimax).
    FullStream,
    /// Use the first observation symbol as the key.
    First,
    /// Use the last observation symbol as the key.
    Last,
    /// Hash the entire observation stream into a single key.
    StreamHash,
}

/// Explicit MC-AIXI Monte Carlo Tree Search strategy.
///
/// `rho_uct` is the default sequential planner described in
/// "A Monte-Carlo AIXI Approximation". `parallel_uct` enables an explicit
/// parallel planner family whose exact behavior is controlled by the worker
/// count and optional BU-UCT threshold.
///
/// The `#[non_exhaustive]` attribute reserves room for additional strategy
/// variants (e.g. the supplementary BU-UCT scheduler) without breaking
/// downstream `match` arms.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum MctsStrategy {
    /// Sequential \rhoUCT / UCT planning.
    RhoUct,
    /// Explicit parallel UCT planning.
    ParallelUct {
        /// Number of logical rollout workers (type-enforced non-zero).
        workers: NonZeroUsize,
        /// Optional BU-UCT threshold parameter `m_max`.
        ///
        /// `None` selects WU-UCT behavior. `Some(x)` enables BU-UCT
        /// thresholding with `x in (0, 1)`.
        bu_uct_m_max: Option<f64>,
    },
}

impl MctsStrategy {
    /// Stable canonical document/API kind string for this planner strategy.
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::RhoUct => "rho_uct",
            Self::ParallelUct { .. } => "parallel_uct",
        }
    }
}

impl Default for MctsStrategy {
    fn default() -> Self {
        Self::RhoUct
    }
}

static PARALLEL_UCT_WORKERS_ONE_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static PARALLEL_UCT_WORKERS_ONE_WARNING_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Emit the `parallel_uct(workers = 1)` warning at most once per process.
pub(crate) fn warn_parallel_uct_workers_one_once() -> bool {
    let emitted = PARALLEL_UCT_WORKERS_ONE_WARNING_EMITTED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok();
    if emitted {
        #[cfg(test)]
        PARALLEL_UCT_WORKERS_ONE_WARNING_COUNT.fetch_add(1, Ordering::SeqCst);
        eprintln!(
            "Warning: MC-AIXI parallel_uct configured with workers=1; prefer workers >= 2 or switch to rho_uct."
        );
    }
    emitted
}

#[cfg(test)]
pub(crate) fn reset_parallel_uct_workers_one_warning_for_tests() {
    PARALLEL_UCT_WORKERS_ONE_WARNING_EMITTED.store(false, Ordering::SeqCst);
    PARALLEL_UCT_WORKERS_ONE_WARNING_COUNT.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn parallel_uct_workers_one_warning_count_for_tests() -> usize {
    PARALLEL_UCT_WORKERS_ONE_WARNING_COUNT.load(Ordering::SeqCst)
}

/// Compute the minimum number of bits required to encode a finite cardinality.
pub(crate) fn bits_for_cardinality(cardinality: usize) -> usize {
    let n = cardinality.max(1);
    let bits = (usize::BITS - (n - 1).leading_zeros()) as usize;
    bits.max(1)
}

#[cfg(feature = "aixi")]
pub(crate) fn action_alphabet_from_action_bits(action_bits: usize) -> ActionAlphabet {
    let shift = u32::try_from(action_bits).expect("action bit width must fit within u32");
    let cardinality = 1usize
        .checked_shl(shift)
        .expect("environment action bit width must be < usize::BITS");
    ActionAlphabet::try_from_usize(cardinality)
        .expect("1 << action_bits must always produce a non-zero action alphabet")
}

/// Error returned when the configured reward range cannot be encoded.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RewardEncodingError {
    /// The configured maximum reward is below the configured minimum reward.
    MaxBelowMin {
        /// Configured maximum reward.
        max_reward: i64,
        /// Configured minimum reward.
        min_reward: i64,
    },
    /// The configured reward offset makes the minimum encoded reward negative.
    NegativeShiftedMinimum {
        /// Shifted minimum reward (`min_reward + reward_offset`).
        shifted_minimum: i128,
    },
    /// The configured reward bit width cannot represent the shifted maximum reward.
    RewardBitsTooSmall {
        /// Shifted maximum reward (`max_reward + reward_offset`).
        shifted_maximum: i128,
        /// Configured reward bit width.
        reward_bits: usize,
        /// Maximum representable encoded reward for `reward_bits`.
        maximum_encoded: u128,
    },
}

impl fmt::Display for RewardEncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxBelowMin {
                max_reward,
                min_reward,
            } => write!(
                f,
                "max_reward must be >= min_reward (got {max_reward} < {min_reward})"
            ),
            Self::NegativeShiftedMinimum { shifted_minimum } => write!(
                f,
                "reward_offset too small: min_reward + reward_offset must be >= 0 (got {shifted_minimum})"
            ),
            Self::RewardBitsTooSmall {
                shifted_maximum,
                reward_bits: _,
                maximum_encoded,
            } => write!(
                f,
                "reward_bits too small for configured reward range: max shifted reward {shifted_maximum} exceeds {maximum_encoded}"
            ),
        }
    }
}

impl Error for RewardEncodingError {}

/// Validate that shifted rewards are representable in the configured bit width.
pub(crate) fn validate_reward_encoding_bounds(
    min_reward: i64,
    max_reward: i64,
    reward_offset: i64,
    reward_bits: usize,
) -> Result<(), RewardEncodingError> {
    if max_reward < min_reward {
        return Err(RewardEncodingError::MaxBelowMin {
            max_reward,
            min_reward,
        });
    }

    let min_shifted = (min_reward as i128) + (reward_offset as i128);
    let max_shifted = (max_reward as i128) + (reward_offset as i128);
    if min_shifted < 0 {
        return Err(RewardEncodingError::NegativeShiftedMinimum {
            shifted_minimum: min_shifted,
        });
    }
    if reward_bits < 64 {
        let max_enc = (1u128 << reward_bits) - 1;
        if (max_shifted as u128) > max_enc {
            return Err(RewardEncodingError::RewardBitsTooSmall {
                shifted_maximum: max_shifted,
                reward_bits,
                maximum_encoded: max_enc,
            });
        }
    }

    Ok(())
}

/// Compute a percept key from an observation stream.
pub fn observation_key_from_stream(
    mode: ObservationKeyMode,
    observations: &[PerceptVal],
    observation_bits: usize,
) -> PerceptVal {
    match mode {
        ObservationKeyMode::FullStream => {
            debug_assert!(
                false,
                "observation_key_from_stream called with FullStream; use observation_repr_from_stream"
            );
            // Fallback to hash in release builds to avoid panics.
            observation_key_from_stream(
                ObservationKeyMode::StreamHash,
                observations,
                observation_bits,
            )
        }
        ObservationKeyMode::First => observations.first().copied().unwrap_or(0),
        ObservationKeyMode::Last => observations.last().copied().unwrap_or(0),
        ObservationKeyMode::StreamHash => {
            let mask = if observation_bits >= 64 {
                u64::MAX
            } else if observation_bits == 0 {
                0
            } else {
                (1u64 << observation_bits) - 1
            };
            let mut h = 0u64;
            for &obs in observations {
                let v = obs & mask;
                h = h.rotate_left(7) ^ v;
            }
            h
        }
    }
}

/// Compute the observation representation used for tree branching.
///
/// - `FullStream` returns the full stream (paper-accurate expectimax).
/// - Other modes collapse to a single-key vector.
pub fn observation_repr_from_stream(
    mode: ObservationKeyMode,
    observations: &[PerceptVal],
    observation_bits: usize,
) -> Vec<PerceptVal> {
    match mode {
        ObservationKeyMode::FullStream => observations.to_vec(),
        _ => vec![observation_key_from_stream(
            mode,
            observations,
            observation_bits,
        )],
    }
}

/// A high-performance random number generator using the XorShift64* algorithm.
#[derive(Clone, Copy)]
pub struct RandomGenerator {
    state: u64,
}

impl RandomGenerator {
    #[inline]
    fn initial_seed() -> u64 {
        #[cfg(feature = "backend-zpaq")]
        {
            if let Ok(bytes) = zpaq_rs::random_bytes(8) {
                let mut seed_arr = [0u8; 8];
                seed_arr.copy_from_slice(&bytes);
                return u64::from_le_bytes(seed_arr);
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            // `SystemTime::now()` is unavailable on `wasm32-unknown-unknown` without WASI.
            return 0xCAFEBABEDEADBEEF ^ 0x9E3779B97F4A7C15;
        }

        #[cfg(not(target_arch = "wasm32"))]
        #[allow(clippy::cast_possible_truncation)]
        {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xCAFEBABEDEADBEEF);
            return nanos ^ 0x9E3779B97F4A7C15;
        }

        #[allow(unreachable_code)]
        0xCAFEBABEDEADBEEF
    }

    /// Creates a new `RandomGenerator` with the canonical deterministic seed.
    pub fn new() -> Self {
        Self::from_seed(DEFAULT_RANDOM_SEED)
    }

    /// Creates a new `RandomGenerator` from runtime entropy.
    ///
    /// This is an explicit opt-in escape hatch for callers that need
    /// non-deterministic sampling.
    pub fn from_entropy() -> Self {
        Self::from_seed(Self::initial_seed())
    }

    /// Creates a new `RandomGenerator` from an explicit seed.
    ///
    /// A zero seed is remapped to a fixed non-zero constant to avoid the
    /// xorshift zero-state trap.
    pub fn from_seed(seed: u64) -> Self {
        let state = if seed == 0 { 0xCAFEBABEDEADBEEF } else { seed };
        Self { state }
    }

    /// Generates the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Generates a pseudo-random `usize` in the range `[0, end)`.
    pub fn gen_range(&mut self, end: usize) -> usize {
        if end == 0 {
            return 0;
        }
        (self.next_u64() % (end as u64)) as usize
    }

    /// Generates a boolean value with probability `p` of being `true`.
    pub fn gen_bool(&mut self, p: f64) -> bool {
        self.gen_f64() < p
    }

    /// Generates a pseudo-random `f64` in the range `[0, 1)`.
    pub fn gen_f64(&mut self) -> f64 {
        // 53 bits
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / 9007199254740992.0)
    }

    /// Forks the RNG state with a salt, returning an independent generator.
    pub fn fork_with(&self, salt: u64) -> Self {
        let mixed = Self::splitmix64(self.state ^ salt ^ 0x9E3779B97F4A7C15);
        let state = if mixed == 0 {
            0xCAFEBABEDEADBEEF
        } else {
            mixed
        };
        Self { state }
    }

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

impl Default for RandomGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Encodes a numeric value into its bit representation and appends it to a `SymbolList`.
///
/// Bits are appended in least-significant-bit first order.
pub fn encode(symlist: &mut SymbolList, value: u64, bits: usize) {
    let mut v = value;
    for _ in 0..bits {
        symlist.push((v & 1) == 1);
        v >>= 1;
    }
}

/// Encodes a signed reward value into its bit representation.
pub fn encode_reward(symlist: &mut SymbolList, value: i64, bits: usize) {
    let mut v = value as u64;
    for _ in 0..bits {
        symlist.push((v & 1) == 1);
        v >>= 1;
    }
}

/// Encodes a reward after applying an additive `offset`.
pub fn encode_reward_offset(symlist: &mut SymbolList, value: i64, bits: usize, offset: i64) {
    let shifted = (value + offset) as u64;
    encode(symlist, shifted, bits);
}

/// Decodes a numeric value from its bit representation.
pub fn decode(symlist: &[Symbol], bits: usize) -> u64 {
    if bits == 0 {
        return 0;
    }
    assert!(bits <= symlist.len());
    let mut value = 0u64;
    for i in 0..bits {
        let sym = symlist[symlist.len() - 1 - i];
        value = (value << 1) + (if sym { 1 } else { 0 });
    }
    value
}

/// Decodes a signed reward value from its bit representation.
pub fn decode_reward(symlist: &[Symbol], bits: usize) -> i64 {
    if bits == 0 {
        return 0;
    }
    let v = decode(symlist, bits);
    if bits < 64 && (v & (1 << (bits - 1))) != 0 {
        // Sign bit set, perform two's complement sign extension
        (v | (!0u64 << bits)) as i64
    } else {
        v as i64
    }
}

/// Decodes a reward encoded with [`encode_reward_offset`].
pub fn decode_reward_offset(symlist: &[Symbol], bits: usize, offset: i64) -> i64 {
    if bits == 0 {
        return 0;
    }
    let v = decode(symlist, bits) as i64;
    v - offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_repr_full_stream_is_identity() {
        let obs = vec![1u64, 2u64, 3u64];
        let repr = observation_repr_from_stream(ObservationKeyMode::FullStream, &obs, 8);
        assert_eq!(repr, obs);
    }

    #[test]
    fn observation_key_first_last() {
        let obs = vec![10u64, 20u64, 30u64];
        assert_eq!(
            observation_key_from_stream(ObservationKeyMode::First, &obs, 8),
            10
        );
        assert_eq!(
            observation_key_from_stream(ObservationKeyMode::Last, &obs, 8),
            30
        );

        let empty: Vec<PerceptVal> = vec![];
        assert_eq!(
            observation_key_from_stream(ObservationKeyMode::First, &empty, 8),
            0
        );
        assert_eq!(
            observation_key_from_stream(ObservationKeyMode::Last, &empty, 8),
            0
        );
    }

    #[test]
    fn observation_key_stream_hash_masks_and_mix() {
        // observation_bits=3 => mask=0b111
        // obs[0]=9 -> 1; h=0.rotate_left(7)^1 = 1
        // obs[1]=2 -> 2; h=1.rotate_left(7)^2 = 128^2 = 130
        let obs = vec![9u64, 2u64];
        let h = observation_key_from_stream(ObservationKeyMode::StreamHash, &obs, 3);
        assert_eq!(h, 130);
    }

    #[test]
    fn observation_key_stream_hash_observation_bits_zero_is_zero() {
        let obs = vec![123u64, 456u64, 789u64];
        let h = observation_key_from_stream(ObservationKeyMode::StreamHash, &obs, 0);
        assert_eq!(h, 0);
    }

    #[test]
    fn observation_key_stream_hash_observation_bits_ge_64_uses_full_u64() {
        let obs = vec![u64::MAX, 0x0123_4567_89ab_cdef];
        let h1 = observation_key_from_stream(ObservationKeyMode::StreamHash, &obs, 64);
        let h2 = observation_key_from_stream(ObservationKeyMode::StreamHash, &obs, 128);
        assert_eq!(h1, h2);
    }

    #[test]
    fn bits_for_cardinality_covers_extreme_sizes_without_overflow() {
        assert_eq!(bits_for_cardinality(0), 1);
        assert_eq!(bits_for_cardinality(1), 1);
        assert_eq!(bits_for_cardinality(2), 1);
        assert_eq!(bits_for_cardinality(3), 2);
        assert_eq!(bits_for_cardinality(usize::MAX), usize::BITS as usize);
    }

    #[test]
    fn action_alphabet_rejects_zero_cardinality() {
        assert_eq!(
            ActionAlphabet::try_from_usize(0).expect_err("zero actions must be rejected"),
            ZeroActionAlphabetError
        );
    }

    #[test]
    fn validate_reward_encoding_bounds_rejects_unrepresentable_ranges() {
        let err = validate_reward_encoding_bounds(0, 100, 0, 1).expect_err("must fail");
        assert!(matches!(
            err,
            RewardEncodingError::RewardBitsTooSmall { .. }
        ));
    }

    #[test]
    fn random_generator_default_is_deterministic_seed_zero() {
        let mut via_new = RandomGenerator::new();
        let mut via_seed = RandomGenerator::from_seed(DEFAULT_RANDOM_SEED);
        for _ in 0..32 {
            assert_eq!(via_new.next_u64(), via_seed.next_u64());
        }
    }

    #[test]
    fn parallel_uct_workers_one_warning_emits_once() {
        reset_parallel_uct_workers_one_warning_for_tests();
        assert!(warn_parallel_uct_workers_one_once());
        assert!(!warn_parallel_uct_workers_one_once());
        assert_eq!(parallel_uct_workers_one_warning_count_for_tests(), 1);
    }

    #[test]
    fn random_generator_fork_with_is_stable() {
        let base = RandomGenerator::from_seed(17);
        let mut a = base.fork_with(99);
        let mut b = base.fork_with(99);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn random_generator_entropy_is_explicit_opt_in() {
        let mut deterministic = RandomGenerator::new();
        let mut explicit = RandomGenerator::from_seed(DEFAULT_RANDOM_SEED);
        for _ in 0..16 {
            assert_eq!(deterministic.next_u64(), explicit.next_u64());
        }

        // Entropy path should be callable explicitly and produce a valid stream.
        let mut entropy_rng = RandomGenerator::from_entropy();
        let _ = entropy_rng.next_u64();
    }
}
