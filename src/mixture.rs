//! Online mixtures of probabilistic predictors (log-loss Hedge / Bayes, switching, MDL).
//!
//! This module provides a small, rigorously correct toolkit for sequential model mixing.
//! Predictors expose per-symbol log-probabilities, which allows principled Bayesian
//! mixture updates and clean information-theoretic accounting.
//!
//! ## Cost-Aware Mixtures
//!
//! The [`CostAwareSwitchingMixture`] implements a Lagrangian approach to resource-aware
//! model selection. The objective is:
//!
//! ```text
//! J_λ(π) = B(π) + λ * T(π)
//! ```
//!
//! where B(π) is total bits and T(π) is total time. The Lagrange multiplier λ (nats/ns)
//! controls the tradeoff between compression quality and computational cost.
//!
//! - λ = 0: Pure compression (standard behavior)
//! - λ > 0: Prefer faster experts, accepting some compression loss
//!
//! Use [`pareto_sweep`] to explore the Pareto frontier of bits vs time tradeoffs,
//! or [`bisect_for_time_budget`] to find λ that achieves a specific time budget.
//!
//! ### Example
//!
//! ```rust,ignore
//! use infotheory::mixture::{ExpertConfig, CostAwareSwitchingMixture, log_lambda_grid, pareto_sweep};
//!
//! let experts = vec![
//!     ExpertConfig::ctw("ctw-d8", 8),
//!     ExpertConfig::rosa("rosa-mo8", 8),
//! ];
//!
//! // Sweep λ to find Pareto frontier
//! let lambdas = log_lambda_grid(-12, -6, 5);
//! let data = b"some test data sequence to compress";
//! let pareto_points = pareto_sweep(&experts, data, 0.001, &lambdas);
//!
//! for p in &pareto_points {
//!     println!("λ={:.2e}: {:.1} bits, {:.1} ms", p.lambda, p.total_bits, p.total_expected_time_ns as f64 / 1e6);
//! }
//! ```

use crate::ctw::FacContextTree;
use crate::zpaq_rate::ZpaqRateModel;
use crate::RateBackend;
use rosaplus::RosaPlus;
use rwkvzip::coders::softmax_pdf_floor_inplace;
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
pub enum RateBackendPredictor {
    /// ROSA-Plus online suffix automaton.
    Rosa {
        model: RosaPlus,
        min_prob: f64,
    },
    /// Byte-wise CTW implemented as 8 factorized bit trees (MSB-first).
    Ctw {
        tree: FacContextTree,
        min_prob: f64,
    },
    /// Factorized CTW with configurable bit-encoding (LSB-first).
    FacCtw {
        tree: FacContextTree,
        bits_per_symbol: usize,
        min_prob: f64,
    },
    /// RWKV-7 neural predictor.
    Rwkv7 {
        compressor: rwkvzip::Compressor,
        primed: bool,
        min_prob: f64,
    },
    /// ZPAQ streaming rate model.
    Zpaq {
        model: ZpaqRateModel,
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
                let bits_per_symbol = encoding_bits.min(8).max(1);
                let tree = FacContextTree::new(base_depth, bits_per_symbol);
                Self::FacCtw {
                    tree,
                    bits_per_symbol,
                    min_prob,
                }
            }
            RateBackend::Rwkv7 { model } => {
                let mut compressor = rwkvzip::Compressor::new_from_model(model);
                let vocab_size = compressor.vocab_size();
                let logits = compressor
                    .model
                    .forward(&mut compressor.scratch, 0, &mut compressor.state);
                softmax_pdf_floor_inplace(logits, vocab_size, &mut compressor.pdf_buffer);
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
            RateBackend::Rwkv7 { .. } => "rwkv7".to_string(),
            RateBackend::Zpaq { method } => format!("zpaq(m={})", method),
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
            RateBackendPredictor::Rwkv7 {
                compressor,
                primed,
                min_prob,
            } => {
                if !*primed {
                    let vocab_size = compressor.vocab_size();
                    let logits = compressor
                        .model
                        .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    softmax_pdf_floor_inplace(logits, vocab_size, &mut compressor.pdf_buffer);
                    *primed = true;
                }
                let p = clamp_prob(compressor.pdf_buffer[symbol as usize], *min_prob);
                p.ln()
            }
            RateBackendPredictor::Zpaq { model } => model.log_prob(symbol),
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
            RateBackendPredictor::Rwkv7 {
                compressor,
                primed,
                ..
            } => {
                if !*primed {
                    let vocab_size = compressor.vocab_size();
                    let logits = compressor
                        .model
                        .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    softmax_pdf_floor_inplace(logits, vocab_size, &mut compressor.pdf_buffer);
                    *primed = true;
                }
                let vocab_size = compressor.vocab_size();
                let logits = compressor.model.forward(
                    &mut compressor.scratch,
                    symbol as u32,
                    &mut compressor.state,
                );
                softmax_pdf_floor_inplace(logits, vocab_size, &mut compressor.pdf_buffer);
            }
            RateBackendPredictor::Zpaq { model } => {
                model.update(symbol);
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
    pub fn rwkv(name: impl Into<String>, model: Arc<rwkvzip::Model>) -> Self {
        let name = name.into();
        Self::uniform(name, move || {
            Box::new(RateBackendPredictor::from_backend(
                RateBackend::Rwkv7 { model: model.clone() },
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
// Cost-Aware Switching Mixture
// =============================================================================

/// Result of a single step in the cost-aware switching mixture.
#[derive(Clone, Debug)]
pub struct CostAwareStepResult {
    /// Log-probability from the augmented mixture (policy-adjusted).
    pub log_prob_augmented: f64,
    /// Log-probability from the plain mixture (true codelength).
    pub log_prob_plain: f64,
    /// Per-expert log-probabilities.
    pub expert_log_probs: Vec<f64>,
    /// Per-expert evaluation times in nanoseconds.
    pub expert_times_ns: Vec<u64>,
    /// Expected time under the posterior (weighted by posterior probabilities).
    pub expected_time_ns: f64,
    /// Effective number of experts (Neff) after update.
    pub neff: f64,
    /// Posterior entropy in bits after update.
    pub entropy_bits: f64,
    /// Max posterior weight after update.
    pub max_posterior: f64,
}

/// Accumulated metrics from cost-aware mixture processing.
#[derive(Clone, Debug, Default)]
pub struct CostAwareMetrics {
    /// Total bits (plain codelength, not augmented).
    pub total_bits: f64,
    /// Total bits (augmented codelength).
    pub total_bits_augmented: f64,
    /// Total expected time under posterior (nanoseconds).
    pub total_expected_time_ns: f64,
    /// Total wall-clock time (nanoseconds) - sum of all expert evaluation times.
    pub total_wall_time_ns: u64,
    /// Number of symbols processed.
    pub num_symbols: usize,
    /// Per-expert cumulative evaluation times.
    pub expert_cum_times_ns: Vec<u64>,
    /// Per-expert cumulative bits.
    pub expert_cum_bits: Vec<f64>,
    /// Min Neff over time.
    pub neff_min: f64,
    /// Max Neff over time.
    pub neff_max: f64,
    /// Sum of Neff over time.
    pub neff_sum: f64,
    /// Min entropy (bits) over time.
    pub entropy_min: f64,
    /// Max entropy (bits) over time.
    pub entropy_max: f64,
    /// Sum of entropy (bits) over time.
    pub entropy_sum: f64,
    /// Max posterior mass observed.
    pub max_posterior: f64,
}

impl CostAwareMetrics {
    /// Bits per symbol (plain).
    pub fn bps(&self) -> f64 {
        if self.num_symbols == 0 {
            0.0
        } else {
            self.total_bits / self.num_symbols as f64
        }
    }

    /// Expected time per symbol (nanoseconds).
    pub fn expected_time_per_symbol_ns(&self) -> f64 {
        if self.num_symbols == 0 {
            0.0
        } else {
            self.total_expected_time_ns / self.num_symbols as f64
        }
    }

    /// Wall-clock time per symbol (nanoseconds).
    pub fn wall_time_per_symbol_ns(&self) -> f64 {
        if self.num_symbols == 0 {
            0.0
        } else {
            self.total_wall_time_ns as f64 / self.num_symbols as f64
        }
    }

    /// Mean Neff over time.
    pub fn neff_mean(&self) -> f64 {
        if self.num_symbols == 0 {
            0.0
        } else {
            self.neff_sum / self.num_symbols as f64
        }
    }

    /// Mean entropy (bits) over time.
    pub fn entropy_mean(&self) -> f64 {
        if self.num_symbols == 0 {
            0.0
        } else {
            self.entropy_sum / self.num_symbols as f64
        }
    }
}

/// Cost-aware switching mixture with Lagrangian time penalty.
///
/// This mixture uses an augmented objective: `J_λ(π) = B(π) + λ * T(π)`
/// where B(π) is bits and T(π) is time. The mixture weights are computed
/// using augmented log-probabilities, but the actual codelength is tracked
/// separately for reporting.
///
/// Use `set_lambda(0.0)` to recover standard switching behavior.
pub struct CostAwareSwitchingMixture {
    experts: Vec<ExpertState>,
    log_prior: Vec<f64>,
    log_alpha: f64,
    log_1m_alpha: f64,
    /// Lagrange multiplier: nats per nanosecond.
    lambda: f64,
    /// Scratch space for per-step log-probabilities.
    scratch_logps: Vec<f64>,
    /// Scratch space for per-step augmented log-probabilities.
    scratch_aug_logps: Vec<f64>,
    /// Scratch space for switch computation.
    scratch_switch: Vec<f64>,
    /// Scratch space for per-step times (nanoseconds).
    scratch_times: Vec<u64>,
    /// Cumulative metrics.
    metrics: CostAwareMetrics,
}

impl CostAwareSwitchingMixture {
    /// Create a new cost-aware switching mixture.
    ///
    /// # Arguments
    /// * `configs` - Expert configurations.
    /// * `alpha` - Switching probability (per step).
    /// * `lambda` - Lagrange multiplier: nats per nanosecond. Use 0.0 for standard behavior.
    pub fn new(configs: &[ExpertConfig], alpha: f64, lambda: f64) -> Self {
        let n = configs.len();
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
            lambda,
            scratch_logps: vec![0.0; n],
            scratch_aug_logps: vec![0.0; n],
            scratch_switch: vec![0.0; n],
            scratch_times: vec![0; n],
            metrics: CostAwareMetrics {
                expert_cum_times_ns: vec![0; n],
                expert_cum_bits: vec![0.0; n],
                neff_min: f64::INFINITY,
                neff_max: 0.0,
                neff_sum: 0.0,
                entropy_min: f64::INFINITY,
                entropy_max: 0.0,
                entropy_sum: 0.0,
                max_posterior: 0.0,
                ..Default::default()
            },
        }
    }

    /// Set the Lagrange multiplier (nats per nanosecond).
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// Get the current lambda value.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Reset the mixture state (weights and metrics).
    pub fn reset(&mut self, configs: &[ExpertConfig]) {
        let n = configs.len();
        self.experts = configs.iter().map(|c| c.build()).collect();
        let log_priors: Vec<f64> = self.experts.iter().map(|e| e.log_prior).collect();
        let norm = logsumexp(&log_priors);
        for e in &mut self.experts {
            e.log_weight -= norm;
        }
        self.log_prior = self.experts.iter().map(|e| e.log_prior - norm).collect();
        self.scratch_logps.resize(n, 0.0);
        self.scratch_aug_logps.resize(n, 0.0);
        self.scratch_switch.resize(n, 0.0);
        self.scratch_times.resize(n, 0);
        self.metrics = CostAwareMetrics {
            expert_cum_times_ns: vec![0; n],
            expert_cum_bits: vec![0.0; n],
            neff_min: f64::INFINITY,
            neff_max: 0.0,
            neff_sum: 0.0,
            entropy_min: f64::INFINITY,
            entropy_max: 0.0,
            entropy_sum: 0.0,
            max_posterior: 0.0,
            ..Default::default()
        };
    }

    /// Process a symbol with time measurement. Returns detailed step result.
    pub fn step(&mut self, symbol: u8) -> CostAwareStepResult {
        use std::time::Instant;

        let n = self.experts.len();
        if n == 0 {
            return CostAwareStepResult {
                log_prob_augmented: f64::NEG_INFINITY,
                log_prob_plain: f64::NEG_INFINITY,
                expert_log_probs: vec![],
                expert_times_ns: vec![],
                expected_time_ns: 0.0,
                neff: 0.0,
                entropy_bits: 0.0,
                max_posterior: 0.0,
            };
        }

        let mut total_step_time_ns = 0u64;

        // Evaluate each expert with timing
        for (i, expert) in self.experts.iter_mut().enumerate() {
            let t0 = Instant::now();
            let logp = expert.log_prob(symbol);
            let elapsed_ns = t0.elapsed().as_nanos() as u64;

            self.scratch_logps[i] = logp;
            self.scratch_times[i] = elapsed_ns;
            total_step_time_ns += elapsed_ns;

            // Augmented log-prob: logp - lambda * time
            // (subtracting because logp is positive reward, we want to penalize time)
            self.scratch_aug_logps[i] = logp - self.lambda * (elapsed_ns as f64);
        }

        // Compute switch transition using AUGMENTED log-probs
        for i in 0..n {
            let log_switch = logsumexp2(
                self.log_1m_alpha + self.experts[i].log_weight,
                self.log_alpha + self.log_prior[i],
            );
            self.scratch_switch[i] = self.scratch_aug_logps[i] + log_switch;
        }
        let log_mix_aug = logsumexp(&self.scratch_switch);

        // Update weights based on augmented mixture
        for i in 0..n {
            self.experts[i].log_weight = self.scratch_switch[i] - log_mix_aug;
        }

        // Compute PLAIN mixture log-prob under the new posterior for codelength accounting
        // This uses the weights determined by the augmented policy, but plain log-probs
        let mut plain_switch = vec![0.0; n];
        for i in 0..n {
            plain_switch[i] = self.experts[i].log_weight + self.scratch_logps[i];
        }
        let log_mix_plain = logsumexp(&plain_switch);

        // Compute expected time under posterior
        let mut expected_time_ns = 0.0;
        let mut sum_sq = 0.0;
        let mut entropy = 0.0;
        let mut max_posterior = 0.0;
        for i in 0..n {
            let weight = self.experts[i].log_weight.exp();
            expected_time_ns += weight * (self.scratch_times[i] as f64);
            sum_sq += weight * weight;
            if weight > 0.0 {
                entropy -= weight * (weight.ln() / std::f64::consts::LN_2);
            }
            if weight > max_posterior {
                max_posterior = weight;
            }
        }
        let neff = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

        // Update experts and accumulate metrics
        for (i, expert) in self.experts.iter_mut().enumerate() {
            let bits = -self.scratch_logps[i] / std::f64::consts::LN_2;
            expert.cum_log_loss -= self.scratch_logps[i];
            expert.update(symbol);

            self.metrics.expert_cum_times_ns[i] += self.scratch_times[i];
            self.metrics.expert_cum_bits[i] += bits;
        }

        let plain_bits = -log_mix_plain / std::f64::consts::LN_2;
        let aug_bits = -log_mix_aug / std::f64::consts::LN_2;

        self.metrics.total_bits += plain_bits;
        self.metrics.total_bits_augmented += aug_bits;
        self.metrics.total_expected_time_ns += expected_time_ns;
        self.metrics.total_wall_time_ns += total_step_time_ns;
        self.metrics.num_symbols += 1;
        self.metrics.neff_min = self.metrics.neff_min.min(neff);
        self.metrics.neff_max = self.metrics.neff_max.max(neff);
        self.metrics.neff_sum += neff;
        self.metrics.entropy_min = self.metrics.entropy_min.min(entropy);
        self.metrics.entropy_max = self.metrics.entropy_max.max(entropy);
        self.metrics.entropy_sum += entropy;
        if max_posterior > self.metrics.max_posterior {
            self.metrics.max_posterior = max_posterior;
        }

        CostAwareStepResult {
            log_prob_augmented: log_mix_aug,
            log_prob_plain: log_mix_plain,
            expert_log_probs: self.scratch_logps.clone(),
            expert_times_ns: self.scratch_times.clone(),
            expected_time_ns,
            neff,
            entropy_bits: entropy,
            max_posterior,
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

    /// Get accumulated metrics.
    pub fn metrics(&self) -> &CostAwareMetrics {
        &self.metrics
    }

    /// Expert names in order.
    pub fn expert_names(&self) -> Vec<String> {
        self.experts.iter().map(|e| e.name.clone()).collect()
    }

    /// Expert cumulative log-losses (nats) and names.
    pub fn expert_log_losses(&self) -> Vec<(String, f64)> {
        self.experts
            .iter()
            .map(|e| (e.name.clone(), e.cum_log_loss))
            .collect()
    }
}

/// Result of a Pareto sweep over lambda values.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    /// Lambda value used.
    pub lambda: f64,
    /// Total bits (plain codelength).
    pub total_bits: f64,
    /// Bits per symbol.
    pub bps: f64,
    /// Total expected time (nanoseconds).
    pub total_expected_time_ns: f64,
    /// Expected time per symbol (nanoseconds).
    pub expected_time_per_symbol_ns: f64,
    /// Total wall-clock time (nanoseconds).
    pub total_wall_time_ns: u64,
    /// Wall-clock time per symbol (nanoseconds).
    pub wall_time_per_symbol_ns: f64,
    /// Number of symbols.
    pub num_symbols: usize,
    /// Per-expert cumulative bits.
    pub expert_bits: Vec<f64>,
    /// Per-expert cumulative times.
    pub expert_times_ns: Vec<u64>,
    /// Min Neff over time.
    pub neff_min: f64,
    /// Mean Neff over time.
    pub neff_mean: f64,
    /// Min entropy (bits) over time.
    pub entropy_min: f64,
    /// Mean entropy (bits) over time.
    pub entropy_mean: f64,
    /// Max posterior mass observed.
    pub max_posterior: f64,
}

/// Run a Pareto sweep over lambda values.
///
/// Returns a vector of Pareto points, one per lambda value.
pub fn pareto_sweep(
    configs: &[ExpertConfig],
    data: &[u8],
    alpha: f64,
    lambdas: &[f64],
) -> Vec<ParetoPoint> {
    lambdas
        .iter()
        .map(|&lambda| {
            // Build fresh experts for each lambda
            let mut mix = CostAwareSwitchingMixture::new(configs, alpha, lambda);

            for &sym in data {
                mix.step(sym);
            }

            let m = mix.metrics();
            ParetoPoint {
                lambda,
                total_bits: m.total_bits,
                bps: m.bps(),
                total_expected_time_ns: m.total_expected_time_ns,
                expected_time_per_symbol_ns: m.expected_time_per_symbol_ns(),
                total_wall_time_ns: m.total_wall_time_ns,
                wall_time_per_symbol_ns: m.wall_time_per_symbol_ns(),
                num_symbols: m.num_symbols,
                expert_bits: m.expert_cum_bits.clone(),
                expert_times_ns: m.expert_cum_times_ns.clone(),
                neff_min: m.neff_min,
                neff_mean: m.neff_mean(),
                entropy_min: m.entropy_min,
                entropy_mean: m.entropy_mean(),
                max_posterior: m.max_posterior,
            }
        })
        .collect()
}

/// Find lambda that approximately achieves the target time budget using bisection.
///
/// # Arguments
/// * `configs` - Expert configurations.
/// * `data` - Data to process.
/// * `alpha` - Switching probability.
/// * `target_time_ns` - Target total expected time in nanoseconds.
/// * `lambda_min` - Minimum lambda to search.
/// * `lambda_max` - Maximum lambda to search.
/// * `tolerance` - Relative tolerance for time budget (e.g., 0.05 = 5%).
/// * `max_iters` - Maximum bisection iterations.
///
/// Returns the best Pareto point found.
pub fn bisect_for_time_budget(
    configs: &[ExpertConfig],
    data: &[u8],
    alpha: f64,
    target_time_ns: f64,
    lambda_min: f64,
    lambda_max: f64,
    tolerance: f64,
    max_iters: usize,
) -> ParetoPoint {
    let run_with_lambda = |lambda: f64| -> ParetoPoint {
        let mut mix = CostAwareSwitchingMixture::new(configs, alpha, lambda);
        for &sym in data {
            mix.step(sym);
        }
        let m = mix.metrics();
        ParetoPoint {
            lambda,
            total_bits: m.total_bits,
            bps: m.bps(),
            total_expected_time_ns: m.total_expected_time_ns,
            expected_time_per_symbol_ns: m.expected_time_per_symbol_ns(),
            total_wall_time_ns: m.total_wall_time_ns,
            wall_time_per_symbol_ns: m.wall_time_per_symbol_ns(),
            num_symbols: m.num_symbols,
            expert_bits: m.expert_cum_bits.clone(),
            expert_times_ns: m.expert_cum_times_ns.clone(),
            neff_min: m.neff_min,
            neff_mean: m.neff_mean(),
            entropy_min: m.entropy_min,
            entropy_mean: m.entropy_mean(),
            max_posterior: m.max_posterior,
        }
    };

    let mut lo = lambda_min;
    let mut hi = lambda_max;
    let mut best = run_with_lambda(lo);

    for _ in 0..max_iters {
        let mid = (lo + hi) / 2.0;
        let point = run_with_lambda(mid);

        let rel_err = (point.total_expected_time_ns - target_time_ns).abs() / target_time_ns;
        if rel_err < tolerance {
            return point;
        }

        // Higher lambda -> lower time (prefer faster experts)
        if point.total_expected_time_ns > target_time_ns {
            lo = mid;
        } else {
            hi = mid;
            best = point;
        }
    }

    best
}

/// Generate a log-spaced grid of lambda values.
pub fn log_lambda_grid(min_exp: i32, max_exp: i32, points_per_decade: usize) -> Vec<f64> {
    let mut lambdas = vec![0.0]; // Always include λ=0 (bits-only)
    for exp in min_exp..=max_exp {
        let base = 10.0_f64.powi(exp);
        for i in 0..points_per_decade {
            let mult = 10.0_f64.powf(i as f64 / points_per_decade as f64);
            lambdas.push(base * mult);
        }
    }
    lambdas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lambdas.dedup();
    lambdas
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

    #[test]
    fn cost_aware_mixture_lambda_zero_matches_regular() {
        let configs = vec![
            ExpertConfig::uniform("zero", || Box::new(AlwaysPredict { byte: 0 })),
            ExpertConfig::uniform("one", || Box::new(AlwaysPredict { byte: 1 })),
        ];

        // Lambda = 0 should behave like regular switching
        let mut cost_mix = CostAwareSwitchingMixture::new(&configs, 0.01, 0.0);
        for _ in 0..10 {
            cost_mix.step(0);
        }
        let post = cost_mix.posterior();
        assert!(post[0] > 0.99, "expert 0 should dominate: {:?}", post);
    }

    #[test]
    fn cost_aware_mixture_tracks_metrics() {
        let configs = vec![
            ExpertConfig::uniform("zero", || Box::new(AlwaysPredict { byte: 0 })),
        ];

        let mut mix = CostAwareSwitchingMixture::new(&configs, 0.01, 0.0);
        for _ in 0..100 {
            mix.step(0);
        }

        let m = mix.metrics();
        assert_eq!(m.num_symbols, 100);
        assert!(m.total_wall_time_ns > 0);
        assert!(m.total_bits.is_finite());
    }

    #[test]
    fn log_lambda_grid_generates_valid_values() {
        let grid = log_lambda_grid(-9, -6, 3);
        assert!(grid[0] == 0.0, "should start with 0");
        assert!(grid.len() > 10, "should have multiple values");
        for i in 1..grid.len() {
            assert!(grid[i] > grid[i - 1], "should be sorted");
        }
    }
}
