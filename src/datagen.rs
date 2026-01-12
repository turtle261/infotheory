//! # Datagen: Synthetic Data Generators for Validation
//!
//! This module provides generators that produce synthetic data with known
//! theoretical entropy values. Useful for validating entropy estimators and
//! creating reproducible test cases.
//!
//! ## Example
//! ```rust
//! use infotheory::datagen::{bernoulli, uniform_random};
//!
//! // Bernoulli(0.5) has H(p) = 1.0 bit
//! let data = bernoulli(1000, 0.5, 42);
//!
//! // Uniform random bytes have H = 8.0 bits/byte
//! let uniform = uniform_random(1000, 42);
//! ```

/// Simple PRNG (xorshift64) for reproducible generation.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEF } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
}

// ============================================================================
// Public Generators
// ============================================================================

/// Generate a Bernoulli bit sequence with probability `p` of each bit being 1.
///
/// **Theoretical entropy**: `H(p) = -p*log2(p) - (1-p)*log2(1-p)` bits/symbol.
///
/// Returns bytes where each byte is a sample (0 or 1).
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `p` - Probability of 1 (must be in [0, 1])
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Vector of n bytes, each being 0 or 1.
pub fn bernoulli(n: usize, p: f64, seed: u64) -> Vec<u8> {
    let mut rng = Xorshift64::new(seed);
    (0..n).map(|_| if rng.next_f64() < p { 1 } else { 0 }).collect()
}

/// Theoretical entropy for a Bernoulli(p) source in bits.
pub fn bernoulli_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// Generate uniform random bytes.
///
/// **Theoretical entropy**: 8.0 bits/byte exactly.
///
/// # Arguments
/// * `n` - Number of bytes to generate
/// * `seed` - Random seed for reproducibility
pub fn uniform_random(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Xorshift64::new(seed);
    (0..n).map(|_| (rng.next() & 0xFF) as u8).collect()
}

/// Generate a periodic sequence by repeating a pattern.
///
/// **Theoretical entropy rate**: Approaches 0 as context order → ∞.
/// For order >= pattern length, H_rate = 0.
///
/// # Arguments
/// * `n` - Total length of output
/// * `pattern` - The pattern to repeat
pub fn periodic(n: usize, pattern: &[u8]) -> Vec<u8> {
    if pattern.is_empty() {
        return vec![0; n];
    }
    (0..n).map(|i| pattern[i % pattern.len()]).collect()
}

/// Generate a first-order binary Markov chain.
///
/// **Theoretical entropy rate**: Can be computed from the stationary distribution
/// and transition probabilities.
///
/// # Arguments
/// * `n` - Number of bytes to generate (each 0 or 1)
/// * `p00` - P(next=0 | current=0)
/// * `p11` - P(next=1 | current=1)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Vector of n bytes, each being 0 or 1.
pub fn markov_1_binary(n: usize, p00: f64, p11: f64, seed: u64) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let mut rng = Xorshift64::new(seed);
    let mut out = Vec::with_capacity(n);

    // Start with stationary distribution
    // Stationary: pi_0 = (1-p11) / (2 - p00 - p11), pi_1 = (1-p00) / (2 - p00 - p11)
    let denom = 2.0 - p00 - p11;
    let pi_0 = if denom.abs() < 1e-12 {
        0.5
    } else {
        (1.0 - p11) / denom
    };
    let mut current: u8 = if rng.next_f64() < pi_0 { 0 } else { 1 };
    out.push(current);

    for _ in 1..n {
        let stay_prob = if current == 0 { p00 } else { p11 };
        if rng.next_f64() < stay_prob {
            // Stay in same state
        } else {
            current = 1 - current;
        }
        out.push(current);
    }
    out
}

/// Theoretical entropy rate for a binary Markov chain with transition probs p00, p11.
///
/// H_rate = pi_0 * h(1-p00) + pi_1 * h(1-p11)
/// where h(p) = -p*log2(p) - (1-p)*log2(1-p) is binary entropy.
pub fn markov_1_binary_entropy_rate(p00: f64, p11: f64) -> f64 {
    let denom = 2.0 - p00 - p11;
    if denom.abs() < 1e-12 {
        // Degenerate case: p00 = p11 = 1.0 means deterministic (always stay)
        // This has entropy rate = 0
        return 0.0;
    }
    let pi_0 = (1.0 - p11) / denom;
    let pi_1 = (1.0 - p00) / denom;

    pi_0 * bernoulli_entropy(1.0 - p00) + pi_1 * bernoulli_entropy(1.0 - p11)
}

/// Generate two identical sequences (for testing identity properties).
///
/// MI(X,X) = H(X), NED(X,X) = 0, NTE(X,X) = 0
pub fn identical_pair(n: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let data = uniform_random(n, seed);
    (data.clone(), data)
}

/// Generate two independent random sequences (for testing independence properties).
///
/// MI(X,Y) ≈ 0, NED(X,Y) ≈ 1, NTE(X,Y) ≈ 2 (for similar entropies)
pub fn independent_pair(n: usize, seed1: u64, seed2: u64) -> (Vec<u8>, Vec<u8>) {
    (uniform_random(n, seed1), uniform_random(n, seed2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_generates_correct_length() {
        let data = bernoulli(100, 0.5, 42);
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn bernoulli_respects_probability() {
        // With p=1.0, all should be 1
        let all_ones = bernoulli(100, 1.0, 42);
        assert!(all_ones.iter().all(|&b| b == 1));

        // With p=0.0, all should be 0
        let all_zeros = bernoulli(100, 0.0, 42);
        assert!(all_zeros.iter().all(|&b| b == 0));
    }

    #[test]
    fn bernoulli_entropy_bounds() {
        assert!((bernoulli_entropy(0.5) - 1.0).abs() < 1e-10);
        assert!(bernoulli_entropy(0.0) == 0.0);
        assert!(bernoulli_entropy(1.0) == 0.0);
    }

    #[test]
    fn periodic_repeats_correctly() {
        let pat = b"ABC";
        let data = periodic(10, pat);
        assert_eq!(data, b"ABCABCABCA");
    }

    #[test]
    fn markov_gives_valid_outputs() {
        let data = markov_1_binary(100, 0.9, 0.9, 42);
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&b| b == 0 || b == 1));
    }

    #[test]
    fn markov_entropy_rate_symmetric() {
        // For p00 = p11 = 0.5, should be 1.0 bit (like fair coin)
        let h = markov_1_binary_entropy_rate(0.5, 0.5);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn markov_entropy_rate_deterministic() {
        // For p00 = p11 = 1.0 (always stay same state), entropy rate = 0
        // because no transitions ever happen
        let h = markov_1_binary_entropy_rate(1.0, 1.0);
        assert!(h.abs() < 1e-10, "deterministic chain should have H=0, got {}", h);
    }
}
