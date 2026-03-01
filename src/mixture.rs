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

use crate::ctw::FacContextTree;
use crate::rosaplus::RosaPlus;
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip;
use crate::zpaq_rate::ZpaqRateModel;
use crate::{MixtureKind, MixtureSpec, RateBackend};
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
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn logit(p: f64) -> f64 {
    let p = clamp_unit_prob(p, DEFAULT_MIN_PROB);
    (p / (1.0 - p)).ln()
}

#[inline]
fn sanitize_weight(w: f64) -> f64 {
    if w.is_finite() { w } else { 0.0 }
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
    /// Log-probability (natural log) of `symbol` given the current history.
    fn log_prob(&mut self, symbol: u8) -> f64;

    /// Update the predictor with the observed `symbol`.
    fn update(&mut self, symbol: u8);
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
        }
    }

    /// Human-readable default name for a backend + config.
    pub fn default_name(backend: &RateBackend, max_order: i64) -> String {
        match backend {
            RateBackend::RosaPlus => format!("rosa(mo={})", max_order),
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
        }
    }
}

impl OnlineBytePredictor for RateBackendPredictor {
    fn log_prob(&mut self, symbol: u8) -> f64 {
        match self {
            RateBackendPredictor::Rosa { model, min_prob } => {
                let p = clamp_prob(model.prob_for_last(symbol as u32), *min_prob);
                p.ln()
            }
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
            RateBackendPredictor::Zpaq { model } => model.log_prob(symbol),
            RateBackendPredictor::Mixture { runtime } => runtime.peek_log_prob(symbol),
        }
    }

    fn update(&mut self, symbol: u8) {
        match self {
            RateBackendPredictor::Rosa { model, .. } => {
                let mut tx = model.begin_tx();
                model.train_sequence_tx(&mut tx, &[symbol]);
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
                compressor, primed, ..
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
                let pdf = compressor.pdf_buffer.clone();
                let _ = compressor.online_update_from_pdf(symbol, &pdf);
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
            RateBackendPredictor::Zpaq { model } => {
                model.update(symbol);
            }
            RateBackendPredictor::Mixture { runtime } => {
                let _ = runtime.step(symbol);
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
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_logps[i] = expert.log_prob(symbol);
            self.scratch_mix[i] = expert.log_weight + self.scratch_logps[i];
        }
        let log_mix = logsumexp(&self.scratch_mix);
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.log_weight = expert.log_weight + self.scratch_logps[i] - log_mix;
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);
        }
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_mix[i] = expert.log_weight + expert.log_prob(symbol);
        }
        logsumexp(&self.scratch_mix)
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
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the fading mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_logps[i] = expert.log_prob(symbol);
            let decayed = self.decay * expert.log_weight;
            self.scratch_mix[i] = decayed + self.scratch_logps[i];
        }
        let log_mix = logsumexp(&self.scratch_mix);
        for (i, expert) in self.experts.iter_mut().enumerate() {
            let decayed = self.decay * expert.log_weight;
            expert.log_weight = decayed + self.scratch_logps[i] - log_mix;
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);
        }
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for (i, expert) in self.experts.iter_mut().enumerate() {
            self.scratch_mix[i] = self.decay * expert.log_weight + expert.log_prob(symbol);
        }
        logsumexp(&self.scratch_mix)
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
            total_log_loss: 0.0,
        }
    }

    /// Log-probability (natural log) of the switching mixture for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
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
        let log_mix = logsumexp(&self.scratch_switch);
        for i in 0..self.experts.len() {
            let expert = &mut self.experts[i];
            expert.log_weight = self.scratch_switch[i] - log_mix;
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);
        }
        self.total_log_loss -= log_mix;
        log_mix
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        for i in 0..self.experts.len() {
            let lp = self.experts[i].log_prob(symbol);
            let log_switch = logsumexp2(
                self.log_1m_alpha + self.experts[i].log_weight,
                self.log_alpha + self.log_prior[i],
            );
            self.scratch_switch[i] = lp + log_switch;
        }
        logsumexp(&self.scratch_switch)
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
}

struct NeuralWeightEntry {
    weights: Vec<f64>,
    bias: f64,
}

impl NeuralWeightEntry {
    fn new(width: usize) -> Self {
        Self {
            weights: vec![0.0; width],
            bias: 0.0,
        }
    }
}

struct NeuralStage2Entry {
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl NeuralStage2Entry {
    fn new(width: usize) -> Self {
        Self {
            weights: vec![0.0; width],
            bias: vec![0.0; 256],
        }
    }
}

/// Bytewise neural mixer inspired by fx2-cmix logistic online adaptation.
///
/// This model is a context-conditioned two-stage network trained online with
/// multiclass (256-way) log-loss:
/// 1) expert probability stretch/logit features,
/// 2) context-local first-stage logistic units,
/// 3) context-local second-stage softmax classifier,
/// 4) per-symbol SGD updates with optional tiny-error skip.
pub struct NeuralMixture {
    experts: Vec<ExpertState>,
    stage1_tables: Vec<Vec<NeuralWeightEntry>>,
    stage2_table: Vec<NeuralStage2Entry>,
    stage1_lr: f64,
    stage2_lr: f64,
    update_skip_threshold: f64,
    min_prob: f64,
    prev1: u8,
    prev2: u8,
    run_len: u16,
    has_history: bool,
    scratch_expert_logps: Vec<f64>,
    scratch_expert_logits: Vec<f64>,
    scratch_stage1_out: Vec<f64>,
    scratch_energy: Vec<f64>,
    scratch_probs: Vec<f64>,
    scratch_errors: Vec<f64>,
    eval_cache_valid: bool,
    eval_cache_prev1: u8,
    eval_cache_prev2: u8,
    eval_cache_run_len: u16,
    eval_cache_has_history: bool,
    total_log_loss: f64,
}

impl NeuralMixture {
    const STAGE1_CONTEXTS: usize = 3;
    const STAGE1_TABLE_SIZES: [usize; Self::STAGE1_CONTEXTS] = [1, 256, 1024];
    const STAGE2_TABLE_SIZE: usize = 512;

    /// Construct a neural mixture. `learning_rate` is taken from `MixtureSpec.alpha`.
    pub fn new(configs: &[ExpertConfig], learning_rate: f64) -> Self {
        let mut experts: Vec<ExpertState> = configs.iter().map(|c| c.build()).collect();
        let n = experts.len();

        let mut prior_logits = vec![0.0; n];
        if n > 0 {
            let log_priors: Vec<f64> = experts.iter().map(|e| e.log_prior).collect();
            let norm = logsumexp(&log_priors);
            for (i, e) in experts.iter_mut().enumerate() {
                let p = (e.log_prior - norm).exp();
                prior_logits[i] = logit(p);
            }
        }

        let mut stage1_tables = Vec::with_capacity(Self::STAGE1_CONTEXTS);
        for (ctx_idx, table_size) in Self::STAGE1_TABLE_SIZES.iter().enumerate() {
            let mut table = Vec::with_capacity(*table_size);
            for _ in 0..*table_size {
                let mut entry = NeuralWeightEntry::new(n);
                if ctx_idx == 0 {
                    entry.weights.clone_from(&prior_logits);
                }
                table.push(entry);
            }
            stage1_tables.push(table);
        }

        let mut stage2_table = Vec::with_capacity(Self::STAGE2_TABLE_SIZE);
        for _ in 0..Self::STAGE2_TABLE_SIZE {
            let mut entry = NeuralStage2Entry::new(Self::STAGE1_CONTEXTS);
            for w in &mut entry.weights {
                *w = 1.0 / (Self::STAGE1_CONTEXTS as f64);
            }
            stage2_table.push(entry);
        }

        let base_lr = if learning_rate.is_finite() {
            learning_rate.abs().clamp(1e-6, 1.0)
        } else {
            0.03
        };

        Self {
            experts,
            stage1_tables,
            stage2_table,
            stage1_lr: base_lr * 0.5,
            stage2_lr: base_lr,
            update_skip_threshold: 1e-5,
            min_prob: DEFAULT_MIN_PROB,
            prev1: 0,
            prev2: 0,
            run_len: 0,
            has_history: false,
            scratch_expert_logps: vec![0.0; n * 256],
            scratch_expert_logits: vec![0.0; n * 256],
            scratch_stage1_out: vec![0.0; Self::STAGE1_CONTEXTS * 256],
            scratch_energy: vec![0.0; 256],
            scratch_probs: vec![0.0; 256],
            scratch_errors: vec![0.0; 256],
            eval_cache_valid: false,
            eval_cache_prev1: 0,
            eval_cache_prev2: 0,
            eval_cache_run_len: 0,
            eval_cache_has_history: false,
            total_log_loss: 0.0,
        }
    }

    #[inline]
    fn stage1_context_indices(&self) -> [usize; Self::STAGE1_CONTEXTS] {
        if !self.has_history {
            return [0, 0, 0];
        }
        let run_bucket = (self.run_len.min(63) as usize) & 0x3f;
        let h = ((self.prev1 as usize) << 10)
            ^ ((self.prev2 as usize) << 2)
            ^ run_bucket
            ^ (((self.prev1 ^ self.prev2) as usize) << 5);
        [0, self.prev1 as usize, h % Self::STAGE1_TABLE_SIZES[2]]
    }

    #[inline]
    fn stage2_context_index(&self) -> usize {
        if !self.has_history {
            return 0;
        }
        let run_bucket = (self.run_len.min(127) as usize) & 0x7f;
        let h = ((self.prev1 as usize) << 8) ^ (self.prev2 as usize) ^ run_bucket;
        h % Self::STAGE2_TABLE_SIZE
    }

    #[inline]
    fn update_history(&mut self, symbol: u8) {
        if self.has_history && symbol == self.prev1 {
            self.run_len = self.run_len.saturating_add(1).min(255);
        } else {
            self.run_len = 1;
        }
        self.prev2 = self.prev1;
        self.prev1 = symbol;
        self.has_history = true;
        self.eval_cache_valid = false;
    }

    fn evaluate_state(&mut self, stage1_idx: [usize; Self::STAGE1_CONTEXTS], stage2_idx: usize) {
        let expert_count = self.experts.len();
        for i in 0..expert_count {
            let expert = &mut self.experts[i];
            for b in 0..256usize {
                let lp = expert.log_prob(b as u8);
                self.scratch_expert_logps[i * 256 + b] = lp;
                let p = clamp_unit_prob(lp.exp(), self.min_prob);
                self.scratch_expert_logits[i * 256 + b] = logit(p);
            }
        }

        for b in 0..256usize {
            for (k, &ctx_i) in stage1_idx.iter().enumerate() {
                let entry = &self.stage1_tables[k][ctx_i];
                let mut z = entry.bias;
                for i in 0..expert_count {
                    z += entry.weights[i] * self.scratch_expert_logits[i * 256 + b];
                }
                self.scratch_stage1_out[k * 256 + b] = logistic(z);
            }

            let entry2 = &self.stage2_table[stage2_idx];
            let mut e = entry2.bias[b];
            for k in 0..Self::STAGE1_CONTEXTS {
                e += entry2.weights[k] * self.scratch_stage1_out[k * 256 + b];
            }
            self.scratch_energy[b] = e;
        }

        let log_z = logsumexp(&self.scratch_energy);
        for b in 0..256usize {
            self.scratch_probs[b] = (self.scratch_energy[b] - log_z).exp();
        }
    }

    fn ensure_evaluated(&mut self) {
        if self.eval_cache_valid
            && self.eval_cache_prev1 == self.prev1
            && self.eval_cache_prev2 == self.prev2
            && self.eval_cache_run_len == self.run_len
            && self.eval_cache_has_history == self.has_history
        {
            return;
        }

        let stage1_idx = self.stage1_context_indices();
        let stage2_idx = self.stage2_context_index();
        self.evaluate_state(stage1_idx, stage2_idx);

        self.eval_cache_valid = true;
        self.eval_cache_prev1 = self.prev1;
        self.eval_cache_prev2 = self.prev2;
        self.eval_cache_run_len = self.run_len;
        self.eval_cache_has_history = self.has_history;
    }

    fn predict_log_prob(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
        if self.experts.len() == 1 {
            return self.experts[0].log_prob(symbol);
        }
        self.ensure_evaluated();
        clamp_unit_prob(self.scratch_probs[symbol as usize], self.min_prob).ln()
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
            self.update_history(symbol);
            return logp;
        }

        let y = symbol as usize;
        let stage1_idx = self.stage1_context_indices();
        let stage2_idx = self.stage2_context_index();
        self.ensure_evaluated();
        for b in 0..256usize {
            self.scratch_errors[b] = if b == y { 1.0 } else { 0.0 } - self.scratch_probs[b];
        }
        let logp = clamp_unit_prob(self.scratch_probs[y], self.min_prob).ln();
        let expert_count = self.experts.len();

        let error_mag = (1.0 - self.scratch_probs[y]).abs();
        if error_mag > self.update_skip_threshold {
            let old_stage2_weights = {
                let w = &self.stage2_table[stage2_idx].weights;
                [w[0], w[1], w[2]]
            };

            {
                let entry2 = &mut self.stage2_table[stage2_idx];
                for k in 0..Self::STAGE1_CONTEXTS {
                    let mut grad = 0.0;
                    for b in 0..256usize {
                        grad += self.scratch_errors[b] * self.scratch_stage1_out[k * 256 + b];
                    }
                    entry2.weights[k] = sanitize_weight(entry2.weights[k] + self.stage2_lr * grad);
                }
                for b in 0..256usize {
                    entry2.bias[b] =
                        sanitize_weight(entry2.bias[b] + self.stage2_lr * self.scratch_errors[b]);
                }
            }

            for (k, &ctx_i) in stage1_idx.iter().enumerate() {
                let v = old_stage2_weights[k];
                let entry = &mut self.stage1_tables[k][ctx_i];

                let mut grad_bias = 0.0;
                for b in 0..256usize {
                    let q = self.scratch_stage1_out[k * 256 + b];
                    grad_bias += self.scratch_errors[b] * v * q * (1.0 - q);
                }
                entry.bias = sanitize_weight(entry.bias + self.stage1_lr * grad_bias);

                for i in 0..expert_count {
                    let mut grad = 0.0;
                    for b in 0..256usize {
                        let q = self.scratch_stage1_out[k * 256 + b];
                        grad += self.scratch_errors[b]
                            * v
                            * q
                            * (1.0 - q)
                            * self.scratch_expert_logits[i * 256 + b];
                    }
                    entry.weights[i] = sanitize_weight(entry.weights[i] + self.stage1_lr * grad);
                }
            }
        }

        for i in 0..expert_count {
            let expert = &mut self.experts[i];
            expert.cum_log_loss -= self.scratch_expert_logps[i * 256 + y];
            expert.update(symbol);
        }
        self.total_log_loss -= logp;
        self.update_history(symbol);
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
        }
    }

    /// Log-probability (natural log) of the MDL selector for `symbol`, then update.
    pub fn step(&mut self, symbol: u8) -> f64 {
        if self.experts.is_empty() {
            return f64::NEG_INFINITY;
        }
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
        let logp = self.scratch_logps[best_idx];
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
        self.experts[best_idx].log_prob(symbol)
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
        MixtureKind::Neural => Ok(MixtureRuntime::Neural(NeuralMixture::new(experts, spec.alpha))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
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
        assert!(late_avg < early_avg, "late_avg={late_avg} early_avg={early_avg}");
        assert!(late_avg < 0.30, "late_avg={late_avg}");
    }

    struct CountingPredict {
        calls: Arc<AtomicUsize>,
    }

    impl OnlineBytePredictor for CountingPredict {
        fn log_prob(&mut self, symbol: u8) -> f64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if symbol == 0 {
                0.0
            } else {
                -20.0
            }
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
        assert_eq!(after_predict, 512);

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
        assert_eq!(after_first, 512);

        let _ = mix.predict_log_prob(1);
        let after_second = c0.load(Ordering::Relaxed) + c1.load(Ordering::Relaxed);
        assert_eq!(after_second, after_first);
    }
}
