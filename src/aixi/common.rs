//! Common types and utilities for the AIXI implementation.

/// Represents a single bit (0 or 1) in the agent's interaction history.
pub type Symbol = bool;

/// A list of symbols, used to represent encoded observations, rewards, or actions.
pub type SymbolList = Vec<Symbol>;

/// Represents an action that the agent can perform.
pub type Action = u64;

/// Represents a reward received by the agent from the environment.
pub type Reward = u64;

/// A generic value for a percept component (either an observation or a reward).
pub type PerceptVal = u64;

/// A high-performance random number generator using the XorShift64* algorithm.
/// 
/// This generator is seeded using `zpaq_rs::random_bytes` to avoid external dependencies
/// like the `rand` crate while maintaining cryptographic-grade entropy for the seed.
pub struct RandomGenerator {
    state: u64,
}

impl RandomGenerator {
    /// Creates a new `RandomGenerator` with a fresh seed.
    pub fn new() -> Self {
        // Seeding from zpaq_rs
        let bytes = zpaq_rs::random_bytes(8).expect("Failed to get random seed");
        let mut seed_arr = [0u8; 8];
        seed_arr.copy_from_slice(&bytes);
        let seed = u64::from_le_bytes(seed_arr);
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
        if end == 0 { return 0; }
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
}

/// Encodes a numeric value into its bit representation and appends it to a `SymbolList`.
/// 
/// Bits are appended in least-significant-bit first order.
pub fn encode(symlist: &mut SymbolList, mut value: u64, bits: usize) {
    for _ in 0..bits {
        symlist.push((value & 1) == 1);
        value >>= 1;
    }
}

/// Decodes a numeric value from its bit representation.
/// 
/// Expects bits to be in least-significant-bit first order.
pub fn decode(symlist: &[Symbol], bits: usize) -> u64 {
    assert!(bits <= symlist.len());
    let mut value = 0;
    for i in 0..bits {
        let sym = symlist[symlist.len() - 1 - i];
        value = (value << 1) + (if sym { 1 } else { 0 });
    }
    value
}
