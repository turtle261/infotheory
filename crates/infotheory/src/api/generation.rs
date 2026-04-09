//! Generation-focused public API surface.

use super::context::RateBackendSession;
use super::types::{GenerationConfig, GenerationStrategy};
use crate::error::{InfotheoryError, InfotheoryResult};
use crate::spec::CompiledRateBackend;

use crate::with_default_ctx;

pub(crate) struct GenerationRng {
    state: u64,
}

impl GenerationRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xD00D_F00D_CAFE_BABEu64
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

#[inline(always)]
fn argmax_log_prob_byte(logps: &[f64; 256]) -> u8 {
    let mut best_idx = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (idx, &logp) in logps.iter().enumerate() {
        let score = if logp.is_finite() {
            logp
        } else {
            f64::NEG_INFINITY
        };
        if score > best {
            best = score;
            best_idx = idx;
        }
    }
    best_idx as u8
}

pub(crate) fn pick_generated_byte(
    logps: &[f64; 256],
    config: GenerationConfig,
    rng: &mut GenerationRng,
) -> u8 {
    if matches!(config.strategy, GenerationStrategy::Greedy)
        || !config.temperature.is_finite()
        || config.temperature <= 0.0
    {
        return argmax_log_prob_byte(logps);
    }

    let mut entries = [(0u8, f64::NEG_INFINITY); 256];
    for (idx, &logp) in logps.iter().enumerate() {
        let scaled = if logp.is_finite() {
            logp / config.temperature
        } else {
            f64::NEG_INFINITY
        };
        entries[idx] = (idx as u8, scaled);
    }
    entries.sort_by(|a, b| b.1.total_cmp(&a.1));

    let keep_k = if config.top_k == 0 {
        entries.len()
    } else {
        config.top_k.min(entries.len())
    };

    let top_p = if config.top_p.is_finite() {
        config.top_p.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let mut max_logp = f64::NEG_INFINITY;
    for &(_, logp) in entries.iter().take(keep_k) {
        if logp.is_finite() {
            max_logp = max_logp.max(logp);
        }
    }
    if !max_logp.is_finite() {
        return argmax_log_prob_byte(logps);
    }

    let mut weights = [(0u8, 0.0f64); 256];
    let mut total = 0.0;
    for (idx, &(byte, logp)) in entries.iter().take(keep_k).enumerate() {
        let w = if logp.is_finite() {
            (logp - max_logp).exp()
        } else {
            0.0
        };
        weights[idx] = (byte, w);
        total += w;
    }
    if !(total.is_finite()) || total <= 0.0 {
        return argmax_log_prob_byte(logps);
    }

    let cutoff_count = if top_p >= 1.0 {
        keep_k
    } else {
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for &(_, w) in weights.iter().take(keep_k) {
            cumulative += w / total;
            keep += 1;
            if cumulative >= top_p {
                break;
            }
        }
        keep.max(1)
    };

    let mut truncated_total = 0.0;
    for &(_, w) in weights.iter().take(cutoff_count) {
        truncated_total += w;
    }
    if !(truncated_total.is_finite()) || truncated_total <= 0.0 {
        return argmax_log_prob_byte(logps);
    }

    let target = rng.next_f64() * truncated_total;
    let mut cumulative = 0.0;
    let mut picked = weights[0].0;
    for &(byte, weight) in weights.iter().take(cutoff_count) {
        cumulative += weight;
        if cumulative >= target {
            picked = byte;
            break;
        }
    }
    picked
}

pub(crate) fn try_generate_rate_backend_chain(
    prefix_parts: &[&[u8]],
    bytes: usize,
    max_order: i64,
    backend: &CompiledRateBackend,
    config: GenerationConfig,
) -> InfotheoryResult<Vec<u8>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }

    let total = prefix_parts
        .iter()
        .map(|p| p.len() as u64)
        .sum::<u64>()
        .saturating_add(bytes as u64);
    let mut session = RateBackendSession::from_backend(backend.clone(), max_order, Some(total))
        .map_err(|e| {
            InfotheoryError::runtime(format!("rate backend generation init failed: {e}"))
        })?;
    for &part in prefix_parts {
        session.observe(part);
    }
    let out = session.generate_bytes(bytes, config);
    session.finish().map_err(|e| {
        InfotheoryError::runtime(format!("rate backend generation finalize failed: {e}"))
    })?;
    Ok(out)
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn generate_rate_backend_chain(
    prefix_parts: &[&[u8]],
    bytes: usize,
    max_order: i64,
    backend: &CompiledRateBackend,
    config: GenerationConfig,
) -> Vec<u8> {
    try_generate_rate_backend_chain(prefix_parts, bytes, max_order, backend, config)
        .expect("generate_rate_backend_chain")
}

/// Generate a continuation from `prompt`
/// using the current default context and [`GenerationConfig::default()`].
///
/// The default is deterministic frozen sampling with seed `42`.
#[inline(always)]
pub fn try_generate_bytes(
    prompt: &[u8],
    bytes: usize,
    max_order: i64,
) -> InfotheoryResult<Vec<u8>> {
    with_default_ctx(|ctx| ctx.try_generate_bytes(prompt, bytes, max_order))
}

/// Generate a continuation from `prompt` using the current default context.
#[inline(always)]
pub fn try_generate_bytes_with_config(
    prompt: &[u8],
    bytes: usize,
    max_order: i64,
    config: GenerationConfig,
) -> InfotheoryResult<Vec<u8>> {
    with_default_ctx(|ctx| ctx.try_generate_bytes_with_config(prompt, bytes, max_order, config))
}

/// Generate a continuation after conditioning on an explicit chain of prefix parts
/// using the current default context and [`GenerationConfig::default()`].
#[inline(always)]
pub fn try_generate_bytes_conditional_chain(
    prefix_parts: &[&[u8]],
    bytes: usize,
    max_order: i64,
) -> InfotheoryResult<Vec<u8>> {
    with_default_ctx(|ctx| ctx.try_generate_bytes_conditional_chain(prefix_parts, bytes, max_order))
}

/// Generate a continuation after conditioning on an explicit chain of prefix parts
/// using the current default context.
#[inline(always)]
pub fn try_generate_bytes_conditional_chain_with_config(
    prefix_parts: &[&[u8]],
    bytes: usize,
    max_order: i64,
    config: GenerationConfig,
) -> InfotheoryResult<Vec<u8>> {
    with_default_ctx(|ctx| {
        ctx.try_generate_bytes_conditional_chain_with_config(prefix_parts, bytes, max_order, config)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_logps(entries: &[(u8, f64)]) -> [f64; 256] {
        let mut logps = [f64::NEG_INFINITY; 256];
        for &(byte, logp) in entries {
            logps[byte as usize] = logp;
        }
        logps
    }

    #[test]
    fn greedy_generation_picks_argmax() {
        let logps = sparse_logps(&[(7, -0.2), (42, -1.0)]);
        let mut rng = GenerationRng::new(123);
        let picked = pick_generated_byte(&logps, GenerationConfig::greedy_frozen(), &mut rng);
        assert_eq!(picked, 7);
    }

    #[test]
    fn nonpositive_temperature_falls_back_to_argmax() {
        let logps = sparse_logps(&[(3, -0.1), (11, -0.3)]);
        let mut rng = GenerationRng::new(7);
        let mut config = GenerationConfig::sampled_frozen(7);
        config.temperature = 0.0;
        let picked = pick_generated_byte(&logps, config, &mut rng);
        assert_eq!(picked, 3);
    }

    #[test]
    fn top_k_sampling_respects_truncation() {
        let logps = sparse_logps(&[(9, -0.01), (10, -0.02), (11, -0.03)]);
        let mut rng = GenerationRng::new(99);
        let mut config = GenerationConfig::sampled_frozen(99);
        config.top_k = 1;
        let picked = pick_generated_byte(&logps, config, &mut rng);
        assert_eq!(picked, 9);
    }

    #[test]
    fn top_p_sampling_keeps_only_minimal_prefix_mass() {
        let logps = sparse_logps(&[(5, 0.0), (6, -1.5), (7, -3.0)]);
        let mut rng = GenerationRng::new(5);
        let mut config = GenerationConfig::sampled_frozen(5);
        config.top_p = 0.5;
        let picked = pick_generated_byte(&logps, config, &mut rng);
        assert_eq!(picked, 5);
    }

    #[test]
    fn nonfinite_log_probs_do_not_panic() {
        let mut logps = [f64::NEG_INFINITY; 256];
        logps[0] = f64::NAN;
        let mut rng = GenerationRng::new(17);
        let picked = pick_generated_byte(&logps, GenerationConfig::sampled_frozen(17), &mut rng);
        assert_eq!(picked, 0);
    }
}
