//! Common types and utilities for the AIXI implementation.

/// Represents a single bit (0 or 1) in the agent's interaction history.
pub type Symbol = bool;

/// A list of symbols, used to represent encoded observations, rewards, or actions.
pub type SymbolList = Vec<Symbol>;

/// Represents an action that the agent can perform.
pub type Action = u64;

/// Represents a reward received by the agent from the environment.
pub type Reward = i64;

/// A generic value for a percept component (either an observation or a reward).
pub type PerceptVal = u64;

/// Strategy for mapping an observation stream into a single percept key for tree search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

    /// Creates a new `RandomGenerator` with a fresh seed.
    pub fn new() -> Self {
        let seed = Self::initial_seed();
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
}
