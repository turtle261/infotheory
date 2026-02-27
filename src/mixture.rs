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
    Rosa { model: RosaPlus, min_prob: f64 },
    /// Byte-wise CTW implemented as 8 factorized bit trees (MSB-first).
    Ctw { tree: FacContextTree, min_prob: f64 },
    /// Factorized CTW with configurable bit-encoding (LSB-first).
    FacCtw {
        tree: FacContextTree,
        bits_per_symbol: usize,
        min_prob: f64,
    },
    /// RWKV-7 neural predictor.
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        compressor: rwkvzip::Compressor,
        primed: bool,
        min_prob: f64,
    },
    /// ZPAQ streaming rate model.
    Zpaq { model: ZpaqRateModel },
    /// Online mixture over experts (Bayes, fading Bayes, switching, MDL).
    Mixture {
        runtime: MixtureRuntime,
        pending_symbol: Option<u8>,
        pending_logp: f64,
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
                Self::Mixture {
                    runtime,
                    pending_symbol: None,
                    pending_logp: 0.0,
                }
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
            RateBackendPredictor::Mixture {
                runtime,
                pending_symbol,
                pending_logp,
            } => {
                if let Some(pending) = *pending_symbol {
                    if pending == symbol {
                        return *pending_logp;
                    }
                    *pending_symbol = None;
                }
                let logp = runtime.step(symbol);
                *pending_symbol = Some(symbol);
                *pending_logp = logp;
                logp
            }
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
            RateBackendPredictor::Mixture {
                runtime,
                pending_symbol,
                ..
            } => {
                if let Some(pending) = *pending_symbol
                    && pending == symbol
                {
                    *pending_symbol = None;
                    return;
                }
                *pending_symbol = None;
                let _ = runtime.step(symbol);
            }
        }
    }
}

/// Configuration for a mixture expert.
#[derive(Clone)]
pub struct ExpertConfig {
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

impl MdlSelector {
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

pub enum MixtureRuntime {
    Bayes(BayesMixture),
    Fading(FadingBayesMixture),
    Switching(SwitchingMixture),
    Mdl(MdlSelector),
}

impl MixtureRuntime {
    /// Step the mixture and return log-probability (nats).
    pub(crate) fn step(&mut self, symbol: u8) -> f64 {
        match self {
            MixtureRuntime::Bayes(m) => m.step(symbol),
            MixtureRuntime::Fading(m) => m.step(symbol),
            MixtureRuntime::Switching(m) => m.step(symbol),
            MixtureRuntime::Mdl(m) => m.step(symbol),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
