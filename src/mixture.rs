//! Online mixtures of probabilistic predictors (log-loss Hedge / Bayes, switching, MDL).
//!
//! This module provides a small, rigorously correct toolkit for sequential model mixing.
//! Predictors expose per-symbol log-probabilities, which allows principled Bayesian
//! mixture updates and clean information-theoretic accounting.
//!
//! ## Rate-Backend Mixtures
//!
//! The mixture primitives here power `RateBackend::Mixture`, enabling Bayes, fading Bayes,
//! switching, and MDL-style selectors to be used anywhere a rate backend is accepted.

use crate::backends::calibration::CalibratorCore;
use crate::backends::match_model::MatchModel;
use crate::backends::ppmd::PpmdModel;
use crate::backends::sparse_match::SparseMatchModel;
use crate::backends::text_context::TextContextAnalyzer;
use crate::ctw::FacContextTree;
#[cfg(feature = "backend-mamba")]
use crate::mambazip;
use crate::neural_mix::{NeuralHistoryState, NeuralMixCore};
use crate::rosaplus::RosaPlus;
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip;
use crate::zpaq_rate::ZpaqRateModel;
use crate::{CalibratedSpec, MixtureKind, MixtureSpec, RateBackend};
use std::sync::Arc;

/// Default minimum probability floor to avoid log(0).
pub const DEFAULT_MIN_PROB: f64 = 5.960_464_477_539_063e-8;

#[inline]
fn clamp_prob(p: f64, min_prob: f64) -> f64 {
    if p.is_finite() {
        p.max(min_prob)
    } else {
        min_prob
    }
}

#[inline]
fn clamp_unit_prob(p: f64, min_prob: f64) -> f64 {
    clamp_prob(p, min_prob).min(1.0 - min_prob)
}

#[inline]
fn build_calibrator(spec: &CalibratedSpec) -> CalibratorCore {
    CalibratorCore::new(spec.context, spec.bins, spec.learning_rate, spec.bias_clip)
}

#[inline]
fn logsumexp(xs: &[f64]) -> f64 {
    let mut max_v = f64::NEG_INFINITY;
    for &v in xs {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0;
    for &v in xs {
        sum += (v - max_v).exp();
    }
    max_v + sum.ln()
}

#[inline]
fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = if a > b { a } else { b };
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[inline]
fn logsumexp_weights(experts: &[ExpertState]) -> f64 {
    let mut max_v = f64::NEG_INFINITY;
    for e in experts {
        if e.log_weight > max_v {
            max_v = e.log_weight;
        }
    }
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0;
    for e in experts {
        sum += (e.log_weight - max_v).exp();
    }
    max_v + sum.ln()
}

/// Trait for online byte-level predictors that expose per-symbol log-probabilities.
pub trait OnlineBytePredictor: Send {
    /// Optional stream-start hook.
    ///
    /// Predictors that require total symbol count (for example percent-based
    /// policy schedules) can initialize runtime state here.
    fn begin_stream(&mut self, _total_symbols: Option<u64>) -> Result<(), String> {
        Ok(())
    }

    /// Log-probability (natural log) of `symbol` given the current history.
    fn log_prob(&mut self, symbol: u8) -> f64;

    /// Bulk 256-way log-probabilities for the next byte.
    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        for (sym, slot) in out.iter_mut().enumerate() {
            *slot = self.log_prob(sym as u8);
        }
    }

    /// Update the predictor with the observed `symbol`.
    fn update(&mut self, symbol: u8);
}

fn fill_fac_tree_log_probs(
    tree: &mut FacContextTree,
    bits_per_symbol: usize,
    msb_first: bool,
    min_logp: f64,
    out: &mut [f64; 256],
) {
    struct RecParams {
        bits: usize,
        msb_first: bool,
        log_before: f64,
        min_logp: f64,
    }

    let bits = bits_per_symbol.clamp(1, 8);
    let patterns = 1usize << bits;
    let mut pattern_logps = [f64::NEG_INFINITY; 256];
    let params = RecParams {
        bits,
        msb_first,
        log_before: tree.get_log_block_probability(),
        min_logp,
    };

    fn rec(
        tree: &mut FacContextTree,
        depth: usize,
        params: &RecParams,
        symbol_acc: u8,
        pattern_logps: &mut [f64; 256],
    ) {
        if depth == params.bits {
            let pat = symbol_acc as usize;
            let logp = (tree.get_log_block_probability() - params.log_before).max(params.min_logp);
            pattern_logps[pat] = logp;
            return;
        }

        for bit in [false, true] {
            tree.update(bit, depth);
            let mut next_symbol = symbol_acc;
            if params.msb_first {
                let shift = 7usize.saturating_sub(depth);
                if bit {
                    next_symbol |= 1u8 << shift;
                }
            } else if bit {
                next_symbol |= 1u8 << depth;
            }
            rec(tree, depth + 1, params, next_symbol, pattern_logps);
            tree.revert(depth);
        }
    }

    rec(tree, 0, &params, 0, &mut pattern_logps);

    if bits == 8 {
        out.copy_from_slice(&pattern_logps);
    } else {
        let aliases = 1usize << (8 - bits);
        let alias_ln = (aliases as f64).ln();
        let mask = patterns - 1;
        for byte in 0..256usize {
            out[byte] = pattern_logps[byte & mask] - alias_ln;
        }
    }
}

/// A concrete online predictor backed by a `RateBackend` configuration.
#[allow(clippy::large_enum_variant)]
pub enum RateBackendPredictor {
    /// ROSA-Plus online suffix automaton.
    Rosa {
        /// ROSA model state.
        model: RosaPlus,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// Local contiguous match predictor.
    Match { model: MatchModel, min_prob: f64 },
    /// Sparse/gapped local match predictor.
    SparseMatch {
        model: SparseMatchModel,
        min_prob: f64,
    },
    /// Bounded-memory PPMD-style predictor.
    Ppmd { model: PpmdModel, min_prob: f64 },
    /// Byte-wise CTW implemented as 8 factorized bit trees (MSB-first).
    Ctw {
        /// FAC-CTW tree stack (8 bits per byte).
        tree: FacContextTree,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// Factorized CTW with configurable bit-encoding (LSB-first).
    FacCtw {
        /// FAC-CTW tree stack for configured bit width.
        tree: FacContextTree,
        /// Active bit-width per symbol.
        bits_per_symbol: usize,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// RWKV-7 neural predictor.
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        /// RWKV compressor/runtime state.
        compressor: rwkvzip::Compressor,
        /// Whether the first-token distribution has been primed.
        primed: bool,
        /// Scratch copy used for update API that borrows immutable PDF.
        pdf_scratch: Vec<f64>,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// Mamba-1 neural predictor.
    #[cfg(feature = "backend-mamba")]
    Mamba {
        /// Mamba compressor/runtime state.
        compressor: mambazip::Compressor,
        /// Whether the first-token distribution has been primed.
        primed: bool,
        /// Scratch copy used for update API that borrows immutable PDF.
        pdf_scratch: Vec<f64>,
        /// Probability floor for numeric stability.
        min_prob: f64,
    },
    /// ZPAQ streaming rate model.
    Zpaq {
        /// ZPAQ rate model state.
        model: ZpaqRateModel,
    },
    /// Online mixture over experts (Bayes, fading Bayes, switching, MDL).
    Mixture {
        /// Active mixture runtime.
        runtime: MixtureRuntime,
    },
    /// Particle-latent filter ensemble.
    Particle {
        /// Particle runtime.
        runtime: crate::particle::ParticleRuntime,
    },
    /// Calibrated wrapper around another predictor.
    Calibrated {
        base: Box<RateBackendPredictor>,
        core: CalibratorCore,
        pdf: [f64; 256],
        valid: bool,
        min_prob: f64,
    },
}

impl RateBackendPredictor {
    /// Create a new online predictor from a rate backend configuration.
    pub fn from_backend(backend: RateBackend, max_order: i64, min_prob: f64) -> Self {
        match backend {
            RateBackend::RosaPlus => {
                let mut model = RosaPlus::new(max_order, false, 0, 42);
                model.build_lm_full_bytes_no_finalize_endpos();
                Self::Rosa { model, min_prob }
            }
            RateBackend::Match {
                hash_bits,
                min_len,
                max_len,
                base_mix,
                confidence_scale,
            } => Self::Match {
                model: MatchModel::new_contiguous(
                    hash_bits,
                    min_len,
                    max_len,
                    base_mix,
                    confidence_scale,
                ),
                min_prob,
            },
            RateBackend::SparseMatch {
                hash_bits,
                min_len,
                max_len,
                gap_min,
                gap_max,
                base_mix,
                confidence_scale,
            } => Self::SparseMatch {
                model: SparseMatchModel::new(
                    hash_bits,
                    min_len,
                    max_len,
                    gap_min,
                    gap_max,
                    base_mix,
                    confidence_scale,
                ),
                min_prob,
            },
            RateBackend::Ppmd { order, memory_mb } => Self::Ppmd {
                model: PpmdModel::new(order, memory_mb),
                min_prob,
            },
            RateBackend::Ctw { depth } => {
                let tree = FacContextTree::new(depth, 8);
                Self::Ctw { tree, min_prob }
            }
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits: _,
                encoding_bits,
            } => {
                let bits_per_symbol = encoding_bits.clamp(1, 8);
                let tree = FacContextTree::new(base_depth, bits_per_symbol);
                Self::FacCtw {
                    tree,
                    bits_per_symbol,
                    min_prob,
                }
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7 { model } => {
                let mut compressor = rwkvzip::Compressor::new_from_model(model);
                let bias = compressor.online_bias_snapshot();
                let logits =
                    compressor
                        .model
                        .forward(&mut compressor.scratch, 0, &mut compressor.state);
                rwkvzip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
                Self::Rwkv7 {
                    pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
                    compressor,
                    primed: true,
                    min_prob,
                }
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7Method { method } => {
                let mut compressor = rwkvzip::Compressor::new_from_method(&method)
                    .unwrap_or_else(|e| panic!("invalid rwkv method '{method}': {e}"));
                let bias = compressor.online_bias_snapshot();
                let logits =
                    compressor
                        .model
                        .forward(&mut compressor.scratch, 0, &mut compressor.state);
                rwkvzip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
                Self::Rwkv7 {
                    pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
                    compressor,
                    primed: true,
                    min_prob,
                }
            }
            #[cfg(feature = "backend-mamba")]
            RateBackend::Mamba { model } => {
                let mut compressor = mambazip::Compressor::new_from_model(model);
                let bias = compressor.online_bias_snapshot();
                let logits =
                    compressor
                        .model
                        .forward(&mut compressor.scratch, 0, &mut compressor.state);
                mambazip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
                Self::Mamba {
                    pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
                    compressor,
                    primed: true,
                    min_prob,
                }
            }
            #[cfg(feature = "backend-mamba")]
            RateBackend::MambaMethod { method } => {
                let mut compressor = mambazip::Compressor::new_from_method(&method)
                    .unwrap_or_else(|e| panic!("invalid mamba method '{method}': {e}"));
                let bias = compressor.online_bias_snapshot();
                let logits =
                    compressor
                        .model
                        .forward(&mut compressor.scratch, 0, &mut compressor.state);
                mambazip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
                Self::Mamba {
                    pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
                    compressor,
                    primed: true,
                    min_prob,
                }
            }
            RateBackend::Zpaq { method } => {
                let model = ZpaqRateModel::new(method, min_prob);
                Self::Zpaq { model }
            }
            RateBackend::Mixture { spec } => {
                let experts = spec.build_experts();
                let runtime = build_mixture_runtime(spec.as_ref(), &experts)
                    .unwrap_or_else(|e| panic!("MixtureSpec invalid: {e}"));
                Self::Mixture { runtime }
            }
            RateBackend::Particle { spec } => {
                let runtime = crate::particle::ParticleRuntime::new(spec.as_ref());
                Self::Particle { runtime }
            }
            RateBackend::Calibrated { spec } => Self::Calibrated {
                base: Box::new(Self::from_backend(spec.base.clone(), max_order, min_prob)),
                core: build_calibrator(spec.as_ref()),
                pdf: [1.0 / 256.0; 256],
                valid: false,
                min_prob,
            },
        }
    }

    /// Human-readable default name for a backend + config.
    pub fn default_name(backend: &RateBackend, max_order: i64) -> String {
        match backend {
            RateBackend::RosaPlus => format!("rosa(mo={})", max_order),
            RateBackend::Match { .. } => "match".to_string(),
            RateBackend::SparseMatch { .. } => "sparse-match".to_string(),
            RateBackend::Ppmd { order, memory_mb } => {
                format!("ppmd(o={},m={}MiB)", order, memory_mb)
            }
            RateBackend::Ctw { depth } => format!("ctw(d={})", depth),
            RateBackend::FacCtw {
                base_depth,
                encoding_bits,
                ..
            } => format!("fac-ctw(d={},b={})", base_depth, encoding_bits),
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7 { .. } => "rwkv7".to_string(),
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7Method { method } => format!("rwkv7({method})"),
            #[cfg(feature = "backend-mamba")]
            RateBackend::Mamba { .. } => "mamba".to_string(),
            #[cfg(feature = "backend-mamba")]
            RateBackend::MambaMethod { method } => format!("mamba({method})"),
            RateBackend::Zpaq { method } => format!("zpaq(m={})", method),
            RateBackend::Mixture { spec } => {
                let kind = match spec.kind {
                    MixtureKind::Bayes => "bayes",
                    MixtureKind::FadingBayes => "fading",
                    MixtureKind::Switching => "switch",
                    MixtureKind::Mdl => "mdl",
                    MixtureKind::Neural => "neural",
                };
                format!("mix({})", kind)
            }
            RateBackend::Particle { spec } => {
                format!("particle(n={},c={})", spec.num_particles, spec.num_cells)
            }
            RateBackend::Calibrated { spec } => {
                format!("calibrated({})", Self::default_name(&spec.base, max_order))
            }
        }
    }
}

impl OnlineBytePredictor for RateBackendPredictor {
    fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        match self {
            RateBackendPredictor::Rosa { model, .. } => {
                if let Some(total) = total_symbols {
                    let reserve = usize::try_from(total).unwrap_or(usize::MAX / 4);
                    model.reserve_for_stream(reserve);
                }
                Ok(())
            }
            RateBackendPredictor::Match { .. }
            | RateBackendPredictor::SparseMatch { .. }
            | RateBackendPredictor::Ppmd { .. } => Ok(()),
            RateBackendPredictor::Ctw { .. }
            | RateBackendPredictor::FacCtw { .. }
            | RateBackendPredictor::Zpaq { .. }
            | RateBackendPredictor::Particle { .. } => Ok(()),
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 { compressor, .. } => compressor
                .begin_online_policy_stream(total_symbols)
                .map_err(|e| e.to_string()),
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba { compressor, .. } => compressor
                .begin_online_policy_stream(total_symbols)
                .map_err(|e| e.to_string()),
            RateBackendPredictor::Mixture { runtime } => runtime.begin_stream(total_symbols),
            RateBackendPredictor::Calibrated { base, .. } => base.begin_stream(total_symbols),
        }
    }

    fn log_prob(&mut self, symbol: u8) -> f64 {
        match self {
            RateBackendPredictor::Rosa { model, min_prob } => {
                let p = clamp_prob(model.prob_for_last(symbol as u32), *min_prob);
                p.ln()
            }
            RateBackendPredictor::Match { model, min_prob } => model.log_prob(symbol, *min_prob),
            RateBackendPredictor::SparseMatch { model, min_prob } => {
                model.log_prob(symbol, *min_prob)
            }
            RateBackendPredictor::Ppmd { model, min_prob } => model.log_prob(symbol, *min_prob),
            RateBackendPredictor::Ctw { tree, min_prob } => {
                let log_before = tree.get_log_block_probability();
                for bit_idx in 0..8 {
                    let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
                    tree.update(bit, bit_idx);
                }
                let log_after = tree.get_log_block_probability();
                for bit_idx in (0..8).rev() {
                    tree.revert(bit_idx);
                }
                let logp = log_after - log_before;
                if logp.is_finite() {
                    logp.max(min_prob.ln())
                } else {
                    min_prob.ln()
                }
            }
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                min_prob,
            } => {
                let log_before = tree.get_log_block_probability();
                for i in 0..*bits_per_symbol {
                    let bit = ((symbol >> i) & 1) == 1;
                    tree.update(bit, i);
                }
                let log_after = tree.get_log_block_probability();
                for i in (0..*bits_per_symbol).rev() {
                    tree.revert(i);
                }
                let logp = log_after - log_before;
                if logp.is_finite() {
                    logp.max(min_prob.ln())
                } else {
                    min_prob.ln()
                }
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    rwkvzip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                let p = clamp_prob(compressor.pdf_buffer[symbol as usize], *min_prob);
                p.ln()
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    mambazip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                let p = clamp_prob(compressor.pdf_buffer[symbol as usize], *min_prob);
                p.ln()
            }
            RateBackendPredictor::Zpaq { model } => model.log_prob(symbol),
            RateBackendPredictor::Mixture { runtime } => runtime.peek_log_prob(symbol),
            RateBackendPredictor::Particle { runtime } => runtime.peek_log_prob(symbol),
            RateBackendPredictor::Calibrated {
                base,
                core,
                pdf,
                valid,
                min_prob,
            } => {
                if !*valid {
                    let mut base_logps = [0.0; 256];
                    base.fill_log_probs(&mut base_logps);
                    let mut base_pdf = [0.0; 256];
                    for (dst, &lp) in base_pdf.iter_mut().zip(base_logps.iter()) {
                        *dst = clamp_prob(lp.exp(), *min_prob);
                    }
                    core.apply_pdf(&base_pdf, pdf);
                    *valid = true;
                }
                pdf[symbol as usize].max(*min_prob).ln()
            }
        }
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        match self {
            RateBackendPredictor::Rosa { model, min_prob } => {
                model.fill_probs_for_last_bytes(out);
                for slot in out.iter_mut() {
                    *slot = clamp_prob(*slot, *min_prob).ln();
                }
            }
            RateBackendPredictor::Match { model, min_prob } => {
                let mut pdf = [0.0; 256];
                model.fill_pdf(&mut pdf);
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            RateBackendPredictor::SparseMatch { model, min_prob } => {
                let mut pdf = [0.0; 256];
                model.fill_pdf(&mut pdf);
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            RateBackendPredictor::Ppmd { model, min_prob } => {
                let mut pdf = [0.0; 256];
                model.fill_pdf(&mut pdf);
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
            RateBackendPredictor::Ctw { tree, min_prob } => {
                fill_fac_tree_log_probs(tree, 8, true, min_prob.ln(), out);
            }
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                min_prob,
            } => {
                fill_fac_tree_log_probs(tree, *bits_per_symbol, false, min_prob.ln(), out);
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    rwkvzip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                for (slot, &p_raw) in out
                    .iter_mut()
                    .take(256)
                    .zip(compressor.pdf_buffer.iter().take(256))
                {
                    let p = clamp_prob(p_raw, *min_prob);
                    *slot = p.ln();
                }
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor,
                primed,
                min_prob,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    mambazip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                for (slot, &p_raw) in out
                    .iter_mut()
                    .take(256)
                    .zip(compressor.pdf_buffer.iter().take(256))
                {
                    let p = clamp_prob(p_raw, *min_prob);
                    *slot = p.ln();
                }
            }
            RateBackendPredictor::Zpaq { model } => {
                model.fill_log_probs(out);
            }
            RateBackendPredictor::Mixture { runtime } => {
                runtime.fill_log_probs(out);
            }
            RateBackendPredictor::Particle { runtime } => {
                runtime.fill_log_probs_cached(out);
            }
            RateBackendPredictor::Calibrated {
                base,
                core,
                pdf,
                valid,
                min_prob,
            } => {
                if !*valid {
                    let mut base_logps = [0.0; 256];
                    base.fill_log_probs(&mut base_logps);
                    let mut base_pdf = [0.0; 256];
                    for (dst, &lp) in base_pdf.iter_mut().zip(base_logps.iter()) {
                        *dst = clamp_prob(lp.exp(), *min_prob);
                    }
                    core.apply_pdf(&base_pdf, pdf);
                    *valid = true;
                }
                for (slot, &p) in out.iter_mut().zip(pdf.iter()) {
                    *slot = clamp_prob(p, *min_prob).ln();
                }
            }
        }
    }

    fn update(&mut self, symbol: u8) {
        match self {
            RateBackendPredictor::Rosa { model, .. } => {
                model.train_byte(symbol);
            }
            RateBackendPredictor::Match { model, .. } => {
                model.update(symbol);
            }
            RateBackendPredictor::SparseMatch { model, .. } => {
                model.update(symbol);
            }
            RateBackendPredictor::Ppmd { model, .. } => {
                model.update(symbol);
            }
            RateBackendPredictor::Ctw { tree, .. } => {
                for bit_idx in 0..8 {
                    let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
                    tree.update(bit, bit_idx);
                }
            }
            RateBackendPredictor::FacCtw {
                tree,
                bits_per_symbol,
                ..
            } => {
                for i in 0..*bits_per_symbol {
                    let bit = ((symbol >> i) & 1) == 1;
                    tree.update(bit, i);
                }
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackendPredictor::Rwkv7 {
                compressor,
                primed,
                pdf_scratch,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    rwkvzip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                if pdf_scratch.len() != compressor.pdf_buffer.len() {
                    pdf_scratch.resize(compressor.pdf_buffer.len(), 0.0);
                }
                pdf_scratch.copy_from_slice(&compressor.pdf_buffer);
                compressor
                    .online_update_from_pdf(symbol, pdf_scratch)
                    .unwrap_or_else(|e| panic!("rwkv online update failed: {e}"));
                let bias = compressor.online_bias_snapshot();
                let logits = compressor.model.forward(
                    &mut compressor.scratch,
                    symbol as u32,
                    &mut compressor.state,
                );
                rwkvzip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
            }
            #[cfg(feature = "backend-mamba")]
            RateBackendPredictor::Mamba {
                compressor,
                primed,
                pdf_scratch,
                ..
            } => {
                if !*primed {
                    let bias = compressor.online_bias_snapshot();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    mambazip::Compressor::logits_to_pdf(
                        logits,
                        bias.as_deref(),
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                if pdf_scratch.len() != compressor.pdf_buffer.len() {
                    pdf_scratch.resize(compressor.pdf_buffer.len(), 0.0);
                }
                pdf_scratch.copy_from_slice(&compressor.pdf_buffer);
                compressor
                    .online_update_from_pdf(symbol, pdf_scratch)
                    .unwrap_or_else(|e| panic!("mamba online update failed: {e}"));
                let bias = compressor.online_bias_snapshot();
                let logits = compressor.model.forward(
                    &mut compressor.scratch,
                    symbol as u32,
                    &mut compressor.state,
                );
                mambazip::Compressor::logits_to_pdf(
                    logits,
                    bias.as_deref(),
                    &mut compressor.pdf_buffer,
                );
            }
            RateBackendPredictor::Zpaq { model } => {
                model.update(symbol);
            }
            RateBackendPredictor::Mixture { runtime } => {
                let _ = runtime.step(symbol);
            }
            RateBackendPredictor::Particle { runtime } => {
                runtime.step(symbol);
            }
            RateBackendPredictor::Calibrated {
                base,
                core,
                pdf,
                valid,
                ..
            } => {
                if !*valid {
                    let mut base_logps = [0.0; 256];
                    base.fill_log_probs(&mut base_logps);
                    let mut base_pdf = [0.0; 256];
                    for (dst, &lp) in base_pdf.iter_mut().zip(base_logps.iter()) {
                        *dst = clamp_prob(lp.exp(), DEFAULT_MIN_PROB);
                    }
                    core.apply_pdf(&base_pdf, pdf);
                }
                core.update(symbol, pdf);
                base.update(symbol);
                *valid = false;
            }
        }
    }
}

/// Configuration for a mixture expert.
#[derive(Clone)]
pub struct ExpertConfig {
    /// Human-readable expert identifier.
    pub name: String,
    /// Log prior weight (natural log). Uniform priors can be `0.0`.
    pub log_prior: f64,
    builder: Arc<dyn Fn() -> Box<dyn OnlineBytePredictor> + Send + Sync>,
}

impl ExpertConfig {
    /// Create a new expert config from a builder closure.
    pub fn new(
        name: impl Into<String>,
        log_prior: f64,
        builder: impl Fn() -> Box<dyn OnlineBytePredictor> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            log_prior,
            builder: Arc::new(builder),
        }
    }

    /// Uniform prior helper.
    pub fn uniform(
        name: impl Into<String>,
        builder: impl Fn() -> Box<dyn OnlineBytePredictor> + Send + Sync + 'static,
    ) -> Self {
        Self::new(name, 0.0, builder)
    }

    /// Expert from a `RateBackend` configuration. `max_order` applies to ROSA.
    pub fn from_rate_backend(
        name: Option<String>,
        log_prior: f64,
        backend: RateBackend,
        max_order: i64,
    ) -> Self {
        let name = name.unwrap_or_else(|| RateBackendPredictor::default_name(&backend, max_order));
        Self::new(name, log_prior, move || {
            Box::new(RateBackendPredictor::from_backend(
                backend.clone(),
                max_order,
                DEFAULT_MIN_PROB,
            ))
        })
    }

    /// ROSA expert (uniform prior).
    pub fn rosa(name: impl Into<String>, max_order: i64) -> Self {
        let name = name.into();
        Self::uniform(name, move || {
            Box::new(RateBackendPredictor::from_backend(
                RateBackend::RosaPlus,
                max_order,
                DEFAULT_MIN_PROB,
            ))
        })
    }

    /// CTW expert (uniform prior).
    pub fn ctw(name: impl Into<String>, depth: usize) -> Self {
        let name = name.into();
        Self::uniform(name, move || {
            Box::new(RateBackendPredictor::from_backend(
                RateBackend::Ctw { depth },
                -1,
                DEFAULT_MIN_PROB,
            ))
        })
    }

    /// FAC-CTW expert (uniform prior).
    pub fn fac_ctw(name: impl Into<String>, base_depth: usize, encoding_bits: usize) -> Self {
        let name = name.into();
        Self::uniform(name, move || {
            Box::new(RateBackendPredictor::from_backend(
                RateBackend::FacCtw {
                    base_depth,
                    num_percept_bits: encoding_bits,
                    encoding_bits,
                },
                -1,
                DEFAULT_MIN_PROB,
            ))
        })
    }

    /// RWKV-7 expert (uniform prior).
    #[cfg(feature = "backend-rwkv")]
    pub fn rwkv(name: impl Into<String>, model: Arc<rwkvzip::Model>) -> Self {
        let name = name.into();
        Self::uniform(name, move || {
            Box::new(RateBackendPredictor::from_backend(
                RateBackend::Rwkv7 {
                    model: model.clone(),
                },
                -1,
                DEFAULT_MIN_PROB,
            ))
        })
    }

    /// Mamba expert (uniform prior).
    #[cfg(feature = "backend-mamba")]
    pub fn mamba(name: impl Into<String>, model: Arc<mambazip::Model>) -> Self {
        let name = name.into();
        Self::uniform(name, move || {
            Box::new(RateBackendPredictor::from_backend(
                RateBackend::Mamba {
                    model: model.clone(),
                },
                -1,
                DEFAULT_MIN_PROB,
            ))
        })
    }

    /// ZPAQ expert (uniform prior).
    pub fn zpaq(name: impl Into<String>, method: impl Into<String>) -> Self {
        let name = name.into();
        let method = method.into();
        Self::uniform(name, move || {
            Box::new(RateBackendPredictor::from_backend(
                RateBackend::Zpaq {
                    method: method.clone(),
                },
                -1,
                DEFAULT_MIN_PROB,
            ))
        })
    }

    /// Expert name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Log prior weight (unnormalized).
    pub fn log_prior(&self) -> f64 {
        self.log_prior
    }

    /// Build a fresh predictor instance for evaluation or analysis.
    pub fn build_predictor(&self) -> Box<dyn OnlineBytePredictor> {
        (self.builder)()
    }

    fn build(&self) -> ExpertState {
        ExpertState {
            name: self.name.clone(),
            log_weight: self.log_prior,
            log_prior: self.log_prior,
            predictor: (self.builder)(),
            cum_log_loss: 0.0,
        }
    }
}

struct ExpertState {
    name: String,
    log_weight: f64,
    log_prior: f64,
    predictor: Box<dyn OnlineBytePredictor>,
    cum_log_loss: f64,
}

impl ExpertState {
    #[inline]
    fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        self.predictor.begin_stream(total_symbols)
    }

    #[inline]
    fn log_prob(&mut self, symbol: u8) -> f64 {
        self.predictor.log_prob(symbol)
    }

    #[inline]
    fn update(&mut self, symbol: u8) {
        self.predictor.update(symbol);
    }
}

/// Exponential-weights Bayes mixture (log-loss Hedge).
pub struct BayesMixture {
    experts: Vec<ExpertState>,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

impl BayesMixture {
    /// Construct a normalized Bayes mixture from expert configs.
    pub fn new(configs: &[ExpertConfig]) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
        let norm = logsumexp(&log_priors);
        for e in &mut experts {
            e.log_weight -= norm;
        }
        Self {
            experts,
            scratch_logps: vec![0.0; configs.len()],
            scratch_mix: vec![0.0; configs.len()],
            cached_symbol: 0,
            cached_log_mix: f64::NEG_INFINITY,
            cache_valid: false,
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        let log_mix = if self.cache_valid && self.cached_symbol == symbol {
            self.cached_log_mix
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob(symbol);
                self.scratch_mix[i] = expert.log_weight + self.scratch_logps[i];
            }
            logsumexp(&self.scratch_mix)
        };
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.log_weight = expert.log_weight + self.scratch_logps[i] - log_mix;
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);
        }
        self.cache_valid = false;
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_logps[i] = expert.log_prob(symbol);
            self.scratch_mix[i] = expert.log_weight + self.scratch_logps[i];
        }
        let log_mix = logsumexp(&self.scratch_mix);
        self.cached_symbol = symbol;
        self.cached_log_mix = log_mix;
        self.cache_valid = true;
        log_mix
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        out.fill(f64::NEG_INFINITY);
        let norm = logsumexp_weights(&self.experts);
        let mut row = [0.0f64; 256];
        for expert in &mut self.experts {
            expert.predictor.fill_log_probs(&mut row);
            let lw = expert.log_weight - norm;
            for b in 0..256 {
                out[b] = logsumexp2(out[b], lw + row[b]);
            }
        }
    }

    /// Posterior weights (normalized) over experts.
    pub fn posterior(&self) -> Vec<f64> {
        let norm = logsumexp_weights(&self.experts);
        self.experts
            .iter()
            .map(|e| (e.log_weight - norm).exp())
            .collect()
    }

    /// Index and log-loss (nats) of the current best expert.
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Index and posterior mass of the most likely expert.
    pub fn max_posterior(&self) -> (usize, f64) {
        let norm = logsumexp_weights(&self.experts);
        let mut best_idx = 0usize;
        let mut best_p = 0.0;
        for (i, e) in self.experts.iter().enumerate() {
            let p = (e.log_weight - norm).exp();
            if p > best_p {
                best_p = p;
                best_idx = i;
            }
        }
        (best_idx, best_p)
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert cumulative log-losses (nats) and names.
    pub fn expert_log_losses(&self) -> Vec<(String, f64)> {
        self.experts
            .iter()
            .map(|e| (e.name.clone(), e.cum_log_loss))
            .collect()
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }
}

/// Exponential-weights Bayes mixture with exponential forgetting on weights.
///
/// This is a non-stationary control: weights are discounted each step by `decay`.
pub struct FadingBayesMixture {
    experts: Vec<ExpertState>,
    decay: f64,
    scratch_logps: Vec<f64>,
    scratch_mix: Vec<f64>,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

impl FadingBayesMixture {
    /// Construct a fading Bayes mixture with decay in `[0, 1]`.
    pub fn new(configs: &[ExpertConfig], decay: f64) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
        let norm = logsumexp(&log_priors);
        for e in &mut experts {
            e.log_weight -= norm;
        }
        let decay = decay.clamp(0.0, 1.0);
        Self {
            experts,
            decay,
            scratch_logps: vec![0.0; configs.len()],
            scratch_mix: vec![0.0; configs.len()],
            cached_symbol: 0,
            cached_log_mix: f64::NEG_INFINITY,
            cache_valid: false,
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the fading mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        let log_mix = if self.cache_valid && self.cached_symbol == symbol {
            self.cached_log_mix
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob(symbol);
                let decayed = self.decay * expert.log_weight;
                self.scratch_mix[i] = decayed + self.scratch_logps[i];
            }
            logsumexp(&self.scratch_mix)
        };
        for (i, expert) in self.experts.iter_mut().enumerate() {
            let decayed = self.decay * expert.log_weight;
            expert.log_weight = decayed + self.scratch_logps[i] - log_mix;
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);
        }
        self.cache_valid = false;
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_logps[i] = expert.log_prob(symbol);
            self.scratch_mix[i] = self.decay * expert.log_weight + self.scratch_logps[i];
        }
        let log_mix = logsumexp(&self.scratch_mix);
        self.cached_symbol = symbol;
        self.cached_log_mix = log_mix;
        self.cache_valid = true;
        log_mix
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        out.fill(f64::NEG_INFINITY);
        let mut decayed = Vec::with_capacity(self.experts.len());
        for expert in &self.experts {
            decayed.push(self.decay * expert.log_weight);
        }
        let norm = logsumexp(&decayed);
        let mut row = [0.0f64; 256];
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.predictor.fill_log_probs(&mut row);
            let lw = decayed[i] - norm;
            for b in 0..256 {
                out[b] = logsumexp2(out[b], lw + row[b]);
            }
        }
    }

    /// Posterior weights (normalized) over experts.
    pub fn posterior(&self) -> Vec<f64> {
        let norm = logsumexp_weights(&self.experts);
        self.experts
            .iter()
            .map(|e| (e.log_weight - norm).exp())
            .collect()
    }

    /// Index and log-loss (nats) of the current best expert (non-discounted loss).
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }
}

/// Switching mixture: allows occasional switches between experts.
pub struct SwitchingMixture {
    experts: Vec<ExpertState>,
    log_prior: Vec<f64>,
    log_alpha: f64,
    log_1m_alpha: f64,
    scratch_logps: Vec<f64>,
    scratch_switch: Vec<f64>,
    cached_symbol: u8,
    cached_log_mix: f64,
    cache_valid: bool,
    total_log_loss: f64,
}

impl SwitchingMixture {
    /// Construct a switching mixture with switch probability `alpha`.
    pub fn new(configs: &[ExpertConfig], alpha: f64) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
        let norm = logsumexp(&log_priors);
        for e in &mut experts {
            e.log_weight -= norm;
        }
        let log_prior: Vec<f64> = experts.iter().map(|e| e.log_prior - norm).collect();
        let alpha = alpha.clamp(1e-12, 1.0 - 1e-12);
        Self {
            experts,
            log_prior,
            log_alpha: alpha.ln(),
            log_1m_alpha: (1.0 - alpha).ln(),
            scratch_logps: vec![0.0; configs.len()],
            scratch_switch: vec![0.0; configs.len()],
            cached_symbol: 0,
            cached_log_mix: f64::NEG_INFINITY,
            cache_valid: false,
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the switching mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        let log_mix = if self.cache_valid && self.cached_symbol == symbol {
            self.cached_log_mix
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob(symbol);
            }
            for i in 0..self.experts.len() {
                let log_switch = logsumexp2(
                    self.log_1m_alpha + self.experts[i].log_weight,
                    self.log_alpha + self.log_prior[i],
                );
                self.scratch_switch[i] = self.scratch_logps[i] + log_switch;
            }
            logsumexp(&self.scratch_switch)
        };
        for i in 0..self.experts.len() {
            let expert = &mut self.experts[i];
            expert.log_weight = self.scratch_switch[i] - log_mix;
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);
        }
        self.cache_valid = false;
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for i in 0..self.experts.len() {
            let lp = self.experts[i].log_prob(symbol);
            self.scratch_logps[i] = lp;
            let log_switch = logsumexp2(
                self.log_1m_alpha + self.experts[i].log_weight,
                self.log_alpha + self.log_prior[i],
            );
            self.scratch_switch[i] = lp + log_switch;
        }
        let log_mix = logsumexp(&self.scratch_switch);
        self.cached_symbol = symbol;
        self.cached_log_mix = log_mix;
        self.cache_valid = true;
        log_mix
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        out.fill(f64::NEG_INFINITY);
        let mut log_switch = vec![0.0f64; self.experts.len()];
        for (i, expert) in self.experts.iter().enumerate() {
            log_switch[i] = logsumexp2(
                self.log_1m_alpha + expert.log_weight,
                self.log_alpha + self.log_prior[i],
            );
        }
        let norm = logsumexp(&log_switch);
        let mut row = [0.0f64; 256];
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.predictor.fill_log_probs(&mut row);
            let lw = log_switch[i] - norm;
            for b in 0..256 {
                out[b] = logsumexp2(out[b], lw + row[b]);
            }
        }
    }

    /// Posterior weights (normalized) over experts.
    pub fn posterior(&self) -> Vec<f64> {
        let norm = logsumexp_weights(&self.experts);
        self.experts
            .iter()
            .map(|e| (e.log_weight - norm).exp())
            .collect()
    }

    /// Index and log-loss (nats) of the current best expert.
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Index and posterior mass of the most likely expert.
    pub fn max_posterior(&self) -> (usize, f64) {
        let norm = logsumexp_weights(&self.experts);
        let mut best_idx = 0usize;
        let mut best_p = 0.0;
        for (i, e) in self.experts.iter().enumerate() {
            let p = (e.log_weight - norm).exp();
            if p > best_p {
                best_p = p;
                best_idx = i;
            }
        }
        (best_idx, best_p)
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert cumulative log-losses (nats) and names.
    pub fn expert_log_losses(&self) -> Vec<(String, f64)> {
        self.experts
            .iter()
            .map(|e| (e.name.clone(), e.cum_log_loss))
            .collect()
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }
}

/// MDL-style selector: predicts with the current best expert (by cumulative loss).
pub struct MdlSelector {
    experts: Vec<ExpertState>,
    scratch_logps: Vec<f64>,
    total_log_loss: f64,
    last_best: usize,
    cached_symbol: u8,
    cached_best_idx: usize,
    cached_best_logp: f64,
    cache_valid: bool,
}

/// Bytewise neural mixer inspired by fx2-cmix online adaptation.
///
/// This model is a context-conditioned two-stage gating network trained online
/// from per-symbol expert likelihoods:
/// 1) context-local first-stage expert gates,
/// 2) context-local second-stage meta-gate over stage-1 outputs,
/// 3) per-symbol SGD updates with optional tiny-error skip.
pub struct NeuralMixture {
    experts: Vec<ExpertState>,
    neural: NeuralMixCore,
    analyzer: TextContextAnalyzer,
    min_prob: f64,
    scratch_expert_logps: Vec<f64>,
    scratch_mix_weights: Vec<f64>,
    eval_cache_valid: bool,
    eval_cache_history: NeuralHistoryState,
    eval_cache_symbol: u8,
    eval_cache_logp: f64,
    total_log_loss: f64,
}

impl NeuralMixture {
    /// Construct a neural mixture. `learning_rate` is taken from `MixtureSpec.alpha`.
    pub fn new(configs: &[ExpertConfig], learning_rate: f64) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let n = experts.len();

        let mut prior_weights = vec![0.0; n];
        if n > 0 {
            let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
            let norm = logsumexp(&log_priors);
            for (i, e) in experts.iter_mut().enumerate() {
                let p = (e.log_prior - norm).exp();
                prior_weights[i] = p;
            }
        }

        let base_lr = if learning_rate.is_finite() {
            learning_rate.abs().clamp(1e-6, 1.0)
        } else {
            0.03
        };
        let effective_lr = (base_lr * 25.0).clamp(1e-6, 1.0);
        let analyzer = TextContextAnalyzer::new();
        let mut neural =
            NeuralMixCore::new(n, &prior_weights, effective_lr * 0.5, effective_lr, 1e-5);
        neural.set_context_state(analyzer.state());
        let eval_cache_history = neural.history_state();

        Self {
            experts,
            neural,
            analyzer,
            min_prob: DEFAULT_MIN_PROB,
            scratch_expert_logps: vec![0.0; n],
            scratch_mix_weights: vec![0.0; n],
            eval_cache_valid: false,
            eval_cache_history,
            eval_cache_symbol: 0,
            eval_cache_logp: f64::NEG_INFINITY,
            total_log_loss: 0.0,
        }
    }

    fn evaluate_symbol(&mut self, symbol: u8) -> f64 {
        self.neural.set_context_state(self.analyzer.state());
        let history = self.neural.history_state();
        if self.eval_cache_valid
            && self.eval_cache_history == history
            && self.eval_cache_symbol == symbol
        {
            let p = self
                .neural
                .evaluate_symbol(&self.scratch_expert_logps, self.min_prob);
            self.eval_cache_logp = clamp_unit_prob(p, self.min_prob).ln();
            return self.eval_cache_logp;
        }

        let expert_count = self.experts.len();
        for i in 0..expert_count {
            self.scratch_expert_logps[i] = self.experts[i].log_prob(symbol);
        }
        let p = self
            .neural
            .evaluate_symbol(&self.scratch_expert_logps, self.min_prob);
        let logp = clamp_unit_prob(p, self.min_prob).ln();
        self.eval_cache_valid = true;
        self.eval_cache_history = self.neural.history_state();
        self.eval_cache_symbol = symbol;
        self.eval_cache_logp = logp;
        logp
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if self.experts.len() == 1 {
            return self.experts[0].log_prob(symbol);
        }
        self.evaluate_symbol(symbol)
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        if self.experts.len() == 1 {
            self.experts[0].predictor.fill_log_probs(out);
            return;
        }
        self.neural.set_context_state(self.analyzer.state());
        self.neural.evaluate_expert_weights();
        self.scratch_mix_weights
            .copy_from_slice(self.neural.expert_weights());
        out.fill(0.0);
        for i in 0..self.experts.len() {
            let mut row = [0.0f64; 256];
            self.experts[i].predictor.fill_log_probs(&mut row);
            let w = self.scratch_mix_weights[i];
            for b in 0..256 {
                out[b] += w * clamp_prob(row[b].exp(), self.min_prob);
            }
        }
        let mut sum = 0.0;
        for &p in out.iter() {
            sum += p;
        }
        if !sum.is_finite() || sum <= 0.0 {
            let u = 1.0f64 / 256.0;
            for slot in out.iter_mut() {
                *slot = u.ln();
            }
            return;
        }
        let inv = 1.0 / sum;
        for slot in out.iter_mut() {
            let p = clamp_unit_prob(*slot * inv, self.min_prob);
            *slot = p.ln();
        }
    }

    /// Log-probability (natural log) of the neural mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }

        if self.experts.len() == 1 {
            let expert = &mut self.experts[0];
            let logp = expert.log_prob(symbol);
            expert.cum_log_loss -= logp;
            expert.update(symbol);
            self.total_log_loss -= logp;
            self.analyzer.update(symbol);
            self.neural.set_context_state(self.analyzer.state());
            self.eval_cache_valid = false;
            return logp;
        }

        let logp = self.evaluate_symbol(symbol);
        let expert_count = self.experts.len();
        self.neural
            .update_weights_symbol(&self.scratch_expert_logps, self.min_prob);

        for i in 0..expert_count {
            let expert = &mut self.experts[i];
            expert.cum_log_loss -= self.scratch_expert_logps[i];
            expert.update(symbol);
        }
        self.total_log_loss -= logp;
        self.analyzer.update(symbol);
        self.neural.set_context_state(self.analyzer.state());
        self.eval_cache_valid = false;
        logp
    }

    /// Total log-loss of the mixture so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }
}

impl MdlSelector {
    /// Construct an MDL-style expert selector.
    pub fn new(configs: &[ExpertConfig]) -> Self {
        let experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let last_best = 0usize;
        Self {
            experts,
            scratch_logps: vec![0.0; configs.len()],
            total_log_loss: 0.0,
            last_best,
            cached_symbol: 0,
            cached_best_idx: 0,
            cached_best_logp: f64::NEG_INFINITY,
            cache_valid: false,
        }
    }

    /// Log-probability (natural log) of the MDL selector for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        let best_idx = if self.cache_valid && self.cached_symbol == symbol {
            self.scratch_logps[self.cached_best_idx] = self.cached_best_logp;
            for (i, expert) in self.experts.iter_mut().enumerate() {
                if i == self.cached_best_idx {
                    continue;
                }
                self.scratch_logps[i] = expert.log_prob(symbol);
            }
            self.cached_best_idx
        } else {
            for (i, expert) in self.experts.iter_mut().enumerate() {
                self.scratch_logps[i] = expert.log_prob(symbol);
            }
            let mut best_idx = 0usize;
            let mut best_loss = f64::INFINITY;
            for (i, expert) in self.experts.iter().enumerate() {
                if expert.cum_log_loss < best_loss {
                    best_loss = expert.cum_log_loss;
                    best_idx = i;
                }
            }
            best_idx
        };
        let logp = self.scratch_logps[best_idx];
        self.cache_valid = false;
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);
        }
        self.total_log_loss -= logp;
        self.last_best = best_idx;
        logp
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, expert) in self.experts.iter().enumerate() {
            if expert.cum_log_loss < best_loss {
                best_loss = expert.cum_log_loss;
                best_idx = i;
            }
        }
        let logp = self.experts[best_idx].log_prob(symbol);
        self.cached_symbol = symbol;
        self.cached_best_idx = best_idx;
        self.cached_best_logp = logp;
        self.cache_valid = true;
        logp
    }

    fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        if self.experts.is_empty() {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, expert) in self.experts.iter().enumerate() {
            if expert.cum_log_loss < best_loss {
                best_loss = expert.cum_log_loss;
                best_idx = i;
            }
        }
        self.experts[best_idx].predictor.fill_log_probs(out);
    }

    /// Index of the current best expert.
    pub fn best_index(&self) -> usize {
        self.last_best
    }

    /// Index and log-loss (nats) of the current best expert.
    pub fn min_expert_log_loss(&self) -> (usize, f64) {
        let mut best_idx = 0usize;
        let mut best_loss = f64::INFINITY;
        for (i, e) in self.experts.iter().enumerate() {
            if e.cum_log_loss < best_loss {
                best_loss = e.cum_log_loss;
                best_idx = i;
            }
        }
        (best_idx, best_loss)
    }

    /// Total log-loss of the selector so far (nats).
    pub fn total_log_loss(&self) -> f64 {
        self.total_log_loss
    }

    /// Expert cumulative log-losses (nats) and names.
    pub fn expert_log_losses(&self) -> Vec<(String, f64)> {
        self.experts
            .iter()
            .map(|e| (e.name.clone(), e.cum_log_loss))
            .collect()
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }
}

// =============================================================================
// Mixture Runtime Helper (for RateBackend::Mixture)
// =============================================================================

/// Runtime wrapper over concrete mixture strategies.
pub enum MixtureRuntime {
    /// Bayes mixture.
    Bayes(BayesMixture),
    /// Fading Bayes mixture.
    Fading(FadingBayesMixture),
    /// Switching mixture.
    Switching(SwitchingMixture),
    /// MDL selector.
    Mdl(MdlSelector),
    /// Bytewise neural logistic mixer.
    Neural(NeuralMixture),
}

impl MixtureRuntime {
    pub(crate) fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
        match self {
            MixtureRuntime::Bayes(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Fading(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Switching(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Mdl(m) => begin_expert_stream(&mut m.experts, total_symbols),
            MixtureRuntime::Neural(m) => begin_expert_stream(&mut m.experts, total_symbols),
        }
    }

    /// Non-mutating log-probability (nats) for `symbol` at current state.
    pub(crate) fn peek_log_prob(&mut self, symbol: u8) -> f64 {
        match self {
            MixtureRuntime::Bayes(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Fading(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Switching(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Mdl(m) => m.predict_log_prob(symbol),
            MixtureRuntime::Neural(m) => m.predict_log_prob(symbol),
        }
    }

    /// Step the mixture and return log-probability (nats).
    pub(crate) fn step(&mut self, symbol: u8) -> f64 {
        match self {
            MixtureRuntime::Bayes(m) => m.step(symbol),
            MixtureRuntime::Fading(m) => m.step(symbol),
            MixtureRuntime::Switching(m) => m.step(symbol),
            MixtureRuntime::Mdl(m) => m.step(symbol),
            MixtureRuntime::Neural(m) => m.step(symbol),
        }
    }

    pub(crate) fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        match self {
            MixtureRuntime::Bayes(m) => m.fill_log_probs(out),
            MixtureRuntime::Fading(m) => m.fill_log_probs(out),
            MixtureRuntime::Switching(m) => m.fill_log_probs(out),
            MixtureRuntime::Mdl(m) => m.fill_log_probs(out),
            MixtureRuntime::Neural(m) => m.fill_log_probs(out),
        }
    }
}

fn begin_expert_stream(
    experts: &mut [ExpertState],
    total_symbols: Option<u64>,
) -> Result<(), String> {
    for expert in experts {
        expert.begin_stream(total_symbols)?;
    }
    Ok(())
}

pub(crate) fn build_mixture_runtime(
    spec: &MixtureSpec,
    experts: &[ExpertConfig],
) -> Result<MixtureRuntime, String> {
    if experts.is_empty() {
        return Err("mixture spec must include at least one expert".to_string());
    }
    match spec.kind {
        MixtureKind::Bayes => Ok(MixtureRuntime::Bayes(BayesMixture::new(experts))),
        MixtureKind::FadingBayes => {
            let decay = spec
                .decay
                .ok_or_else(|| "fading Bayes mixture requires decay".to_string())?;
            Ok(MixtureRuntime::Fading(FadingBayesMixture::new(
                experts, decay,
            )))
        }
        MixtureKind::Switching => Ok(MixtureRuntime::Switching(SwitchingMixture::new(
            experts, spec.alpha,
        ))),
        MixtureKind::Mdl => Ok(MixtureRuntime::Mdl(MdlSelector::new(experts))),
        MixtureKind::Neural => Ok(MixtureRuntime::Neural(NeuralMixture::new(
            experts, spec.alpha,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    };

    struct AlwaysPredict {
        byte: u8,
    }

    impl OnlineBytePredictor for AlwaysPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            if symbol == self.byte {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        }

        fn update(&mut self, _symbol: u8) {}
    }

    #[test]
    fn bayes_mixture_prefers_correct_expert() {
        let configs = vec![
            ExpertConfig::uniform("zero", || Box::new(AlwaysPredict { byte: 0 })),
            ExpertConfig::uniform("one", || Box::new(AlwaysPredict { byte: 1 })),
        ];
        let mut mix = BayesMixture::new(&configs);
        for _ in 0..10 {
            mix.step(0);
        }
        let post = mix.posterior();
        assert!(post[0] > 0.999);
        assert!(post[1] < 1e-6);
    }

    fn counting_cfg(name: &'static str, calls: Arc<AtomicUsize>) -> ExpertConfig {
        ExpertConfig::uniform(name, move || {
            Box::new(CountingPredict {
                calls: calls.clone(),
            })
        })
    }

    #[test]
    fn bayes_predict_then_step_reuses_cached_log_probs() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mix = BayesMixture::new(&[
            counting_cfg("c0", c0.clone()),
            counting_cfg("c1", c1.clone()),
        ]);
        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);
        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn fading_predict_then_step_reuses_cached_log_probs() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mix = FadingBayesMixture::new(
            &[
                counting_cfg("c0", c0.clone()),
                counting_cfg("c1", c1.clone()),
            ],
            0.95,
        );
        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);
        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn switching_predict_then_step_reuses_cached_log_probs() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mix = SwitchingMixture::new(
            &[
                counting_cfg("c0", c0.clone()),
                counting_cfg("c1", c1.clone()),
            ],
            0.05,
        );
        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);
        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn mdl_predict_then_step_reuses_best_expert_log_prob() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let mut mdl = MdlSelector::new(&[
            counting_cfg("c0", c0.clone()),
            counting_cfg("c1", c1.clone()),
        ]);
        let _ = mdl.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 1);
        let _ = mdl.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, 2);
    }

    #[test]
    fn neural_mixture_adapts_to_correct_symbol() {
        let configs = vec![
            ExpertConfig::uniform("zero", || Box::new(AlwaysPredict { byte: 0 })),
            ExpertConfig::uniform("one", || Box::new(AlwaysPredict { byte: 1 })),
        ];
        let mut mix = NeuralMixture::new(&configs, 0.05);

        let mut early = 0.0;
        let mut late = 0.0;
        for t in 0..200 {
            let lp = mix.step(0);
            if t < 20 {
                early -= lp;
            }
            if t >= 180 {
                late -= lp;
            }
        }

        let early_avg = early / 20.0;
        let late_avg = late / 20.0;
        assert!(
            late_avg < early_avg,
            "late_avg={late_avg} early_avg={early_avg}"
        );
        assert!(late_avg < 0.35, "late_avg={late_avg}");
    }

    struct CountingPredict {
        calls: Arc<AtomicUsize>,
    }

    impl OnlineBytePredictor for CountingPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if symbol == 0 { 0.0 } else { -20.0 }
        }

        fn update(&mut self, _symbol: u8) {}
    }

    struct BeginAwarePredict {
        seen_total: Arc<AtomicU64>,
        began: bool,
    }

    impl OnlineBytePredictor for BeginAwarePredict {
        fn begin_stream(&mut self, total_symbols: Option<u64>) -> Result<(), String> {
            let total = total_symbols.ok_or_else(|| "missing total symbols".to_string())?;
            self.seen_total.store(total, Ordering::Relaxed);
            self.began = true;
            Ok(())
        }

        fn log_prob(&mut self, _symbol: u8) -> f64 {
            if self.began { 0.0 } else { f64::NEG_INFINITY }
        }

        fn update(&mut self, _symbol: u8) {}
    }

    #[test]
    fn neural_predict_then_step_reuses_evaluation_cache() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let cfg0 = {
            let c = c0.clone();
            ExpertConfig::uniform("c0", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let cfg1 = {
            let c = c1.clone();
            ExpertConfig::uniform("c1", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let mut mix = NeuralMixture::new(&[cfg0, cfg1], 0.03);

        let _ = mix.predict_log_prob(0);
        let after_predict = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_predict, 2);

        let _ = mix.step(0);
        let after_step = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_step, after_predict);
    }

    #[test]
    fn neural_predict_multiple_symbols_reuses_single_evaluation() {
        let c0 = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::new(AtomicUsize::new(0));
        let cfg0 = {
            let c = c0.clone();
            ExpertConfig::uniform("c0", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let cfg1 = {
            let c = c1.clone();
            ExpertConfig::uniform("c1", move || Box::new(CountingPredict { calls: c.clone() }))
        };
        let mut mix = NeuralMixture::new(&[cfg0, cfg1], 0.03);

        let _ = mix.predict_log_prob(0);
        let after_first = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_first, 2);

        let _ = mix.predict_log_prob(1);
        let after_second = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_second, after_first + 2);
    }

    #[test]
    fn runtime_begin_stream_propagates_to_experts() {
        let seen_total = Arc::new(AtomicU64::new(0));
        let cfg = {
            let seen_total = seen_total.clone();
            ExpertConfig::uniform("begin-aware", move || {
                Box::new(BeginAwarePredict {
                    seen_total: seen_total.clone(),
                    began: false,
                })
            })
        };

        let spec = MixtureSpec::new(MixtureKind::Bayes, vec![]);
        let mut runtime = build_mixture_runtime(&spec, &[cfg]).expect("runtime");
        runtime.begin_stream(Some(123)).expect("begin stream");
        let _ = runtime.step(0);
        assert_eq!(seen_total.load(Ordering::Relaxed), 123);
    }

    #[test]
    fn zpaq_fill_log_probs_does_not_drift_history() {
        let backend = RateBackend::Zpaq {
            method: "1".to_string(),
        };
        let mut baseline =
            RateBackendPredictor::from_backend(backend.clone(), -1, DEFAULT_MIN_PROB);
        let mut probe = RateBackendPredictor::from_backend(backend, -1, DEFAULT_MIN_PROB);

        let history = b"history for zpaq predictor";
        for &b in history {
            baseline.update(b);
            probe.update(b);
        }

        let mut row = [0.0f64; 256];
        probe.fill_log_probs(&mut row);

        let sym = b'k';
        let lp_base = baseline.log_prob(sym);
        let lp_probe = probe.log_prob(sym);
        assert!((lp_base - lp_probe).abs() < 1e-9);
        assert!((row[sym as usize] - lp_base).abs() < 1e-9);

        baseline.update(sym);
        probe.update(sym);
        let next = b'q';
        let next_base = baseline.log_prob(next);
        let next_probe = probe.log_prob(next);
        assert!((next_base - next_probe).abs() < 1e-9);
    }
}
