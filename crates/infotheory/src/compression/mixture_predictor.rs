//! Compression-side mixture predictor.
//!
//! This owns mixture-specific state and bitwise policy; the parent compression
//! module retains generic coders, predictors, and framing.

use super::*;
use crate::byte_prefix::BytePrefixCdfScratch;
use crate::neural_mix::{
    LogisticBytePdfSession, LogisticMixContext, LogisticMixCore, LogisticMixUndo, NeuralMixCore,
    fold_logistic_match_states, logistic_stretch_mixer_active,
};
use anyhow::{Result, bail};

const LOGISTIC_DIAGNOSTICS_UNSUPPORTED: &str = "ac-log-loss diagnostics are not supported for logistic mixtures (stretch-domain bit mixer has no truthful simplex expert-weight decomposition yet)";

#[derive(Clone)]
struct MixExpert {
    predictor: Box<RatePdfPredictor>,
    log_weight: f64,
    log_prior: f64,
    cum_log_loss: f64,
}

/// Bit-mixing law selected by a compression mixture.
///
/// Ordinary mixture kinds use externally prepared simplex weights. A genuine
/// multi-expert logistic mixture owns a stretch-domain mixer instead, so its
/// PDF and recursive-bit state are kept together in [`StretchBitMixer`] rather
/// than leaking through boolean checks across `MixturePredictor`.
#[derive(Clone)]
enum MixtureBitMixPolicy {
    Simplex,
    /// Allocated only for true multi-expert logistic mixtures. Keeping this
    /// large scratch owner indirect makes the common simplex predictor compact.
    Stretch(Box<StretchBitMixer>),
}

/// Compression adapter for the shared PAQ-style stretch-domain mixer.
///
/// The mathematical update law and speculative byte-PDF materialization live
/// in `crate::neural_mix`; this state owns only compression-specific expert
/// prefix caches and the active byte-prefix context.
struct StretchBitMixer {
    core: LogisticMixCore,
    prefix_cdfs: BytePrefixCdfScratch,
    prefix_ranges: Vec<MsbPrefixRange>,
    speculative_undo: LogisticMixUndo,
    prefix: u16,
    match_state: LogisticMatchState,
}

impl Clone for StretchBitMixer {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            // Full byte-PDF materialization overwrites these buffers. Keeping
            // their warmed allocations out of predictor clones avoids copying
            // one large CDF per expert and a speculative undo journal.
            prefix_cdfs: Vec::new(),
            prefix_ranges: Vec::new(),
            speculative_undo: LogisticMixUndo::default(),
            prefix: self.prefix,
            match_state: self.match_state,
        }
    }
}

impl StretchBitMixer {
    fn new(expert_count: usize, prior_weights: &[f64], learning_rate: f64) -> Self {
        Self {
            core: LogisticMixCore::new(expert_count, prior_weights, learning_rate),
            prefix_cdfs: Vec::new(),
            prefix_ranges: Vec::new(),
            speculative_undo: LogisticMixCore::new_undo(expert_count),
            prefix: 1,
            match_state: LogisticMatchState::default(),
        }
    }

    fn materialize_pdf(
        &mut self,
        experts: &mut [MixExpert],
        analyzer: &TextContextAnalyzer,
        bit_probs: &mut Vec<f64>,
        out: &mut [f64],
    ) -> Result<()> {
        let n = experts.len();
        debug_assert!(logistic_stretch_mixer_active(n));
        self.prefix_cdfs.resize_with(n, zeroed_prefix_cdf_box);
        self.prefix_ranges.resize(n, MsbPrefixRange::FULL);
        bit_probs.resize(n, 0.5);
        for (expert, cdf) in experts.iter_mut().zip(self.prefix_cdfs.iter_mut()) {
            let pdf = expert.predictor.pdf_next()?;
            fill_prefix_cdf_from_pdf(cdf, pdf, PDF_MIN);
        }
        self.match_state = aggregate_compression_logistic_match_state(experts);
        let mut session = LogisticBytePdfSession {
            logistic: &mut self.core,
            expert_prefix_cdfs: &self.prefix_cdfs,
            history: analyzer.state(),
            match_state: self.match_state,
            min_prob: PDF_MIN,
            bit_probs,
            ranges: &mut self.prefix_ranges,
            undo: &mut self.speculative_undo,
        };
        session.materialize_adaptive(out);
        Ok(())
    }

    fn begin_byte_step(&mut self, experts: &mut [MixExpert]) {
        self.prefix = 1;
        self.match_state = aggregate_compression_logistic_match_state(experts);
    }

    fn bit_context(&self, analyzer: &TextContextAnalyzer, bit_idx: usize) -> LogisticMixContext {
        LogisticMixContext {
            history: analyzer.state(),
            bit_idx: bit_idx as u8,
            prefix: self.prefix,
            match_len_bucket: self.match_state.len_bucket,
            match_predicted_class: self.match_state.predicted_class,
        }
    }

    fn predict_bit(
        &mut self,
        analyzer: &TextContextAnalyzer,
        bit_idx: usize,
        expert_probs: &[f64],
    ) -> f64 {
        self.core.set_context(self.bit_context(analyzer, bit_idx));
        self.core.predict_bit(expert_probs, PDF_MIN)
    }

    fn observe_bit(
        &mut self,
        analyzer: &TextContextAnalyzer,
        bit_idx: usize,
        expert_probs: &[f64],
        bit: bool,
    ) -> f64 {
        self.core.set_context(self.bit_context(analyzer, bit_idx));
        let probability = self.core.observe_bit(expert_probs, bit, PDF_MIN);
        self.prefix = advanced_prefix_code(self.prefix, bit);
        probability
    }

    fn finish_byte(&mut self, analyzer: &TextContextAnalyzer) {
        self.core.set_context(LogisticMixContext {
            history: analyzer.state(),
            bit_idx: 0,
            prefix: 1,
            match_len_bucket: 0,
            match_predicted_class: 0,
        });
    }
}

#[derive(Clone)]
pub(crate) struct MixturePredictor {
    kind: MixtureKind,
    schedule: MixtureScheduleMode,
    alpha: f64,
    decay: f64,
    experts: Vec<MixExpert>,
    prior_weights: Vec<f64>,
    /// Neural tables exist only for a multi-expert neural mixture; the
    /// one-expert neural case is the identity of its expert.
    neural: Option<NeuralMixCore>,
    bit_mix_policy: MixtureBitMixPolicy,
    analyzer: TextContextAnalyzer,
    bitwise_expert_states: Vec<PredictorBitwiseStepState>,
    // Reused per-expert observation scratch: symbol paths store log p(symbol),
    // while bitwise AC temporarily stages p(bit = 1) before collapsing back to
    // the symbol log-probability at byte completion.
    expert_observation_scratch: Vec<f64>,
    scratch: Vec<f64>,
    scratch2: Vec<f64>,
    projection_scratch: Vec<f64>,
    // Reused output buffer for `predictive_weights`; avoids a fresh
    // per-symbol `Vec<f64>` allocation on every PDF materialization.
    weights_scratch: Vec<f64>,
    pdf: Vec<f64>,
    valid: bool,
    switch_updates: u64,
    convex_updates: u64,
}

impl MixturePredictor {
    pub(crate) fn new_from_compiled(backend: &CompiledRateBackend) -> Result<Self> {
        let crate::spec::core::RateBackendPlan::Mixture {
            kind,
            schedule,
            alpha,
            decay,
            experts: plan_experts,
            ..
        } = backend.plan()
        else {
            bail!("compiled backend is not a mixture backend");
        };
        let mut experts = Vec::with_capacity(plan_experts.len());
        for expert_plan in plan_experts.iter() {
            let compiled =
                crate::spec::core::compiled_rate_backend_from_plan(expert_plan.backend.clone())
                    .map_err(anyhow::Error::msg)?;
            experts.push(MixExpert {
                predictor: Box::new(crate::runtime::build_rate_pdf_predictor(&compiled)?),
                log_weight: expert_plan.log_prior,
                log_prior: expert_plan.log_prior,
                cum_log_loss: 0.0,
            });
        }
        let m = logsumexp_expert_weights(&experts);
        for e in &mut experts {
            e.log_weight -= m;
        }

        let mut prior_weights = vec![0.0; experts.len()];
        normalized_mix_expert_prior_weights(&experts, &mut prior_weights);
        let analyzer = TextContextAnalyzer::new();
        let neural = if *kind == MixtureKind::Neural && experts.len() > 1 {
            let mut neural_prior_weights = prior_weights.clone();
            for weight in &mut neural_prior_weights {
                *weight = weight.clamp(PDF_MIN, 1.0 - PDF_MIN);
            }
            let base_lr = alpha.abs().clamp(1e-6, 1.0);
            let effective_lr = (base_lr * 25.0).clamp(1e-6, 1.0);
            let mut neural = NeuralMixCore::new(
                experts.len(),
                &neural_prior_weights,
                effective_lr * 0.5,
                effective_lr,
                1e-5,
            );
            neural.set_context_state(analyzer.state());
            Some(neural)
        } else {
            None
        };
        if *kind == MixtureKind::Logistic {
            crate::neural_mix::validate_logistic_learning_rate(*alpha)
                .map_err(anyhow::Error::msg)?;
        }
        let bit_mix_policy =
            if *kind == MixtureKind::Logistic && logistic_stretch_mixer_active(experts.len()) {
                MixtureBitMixPolicy::Stretch(Box::new(StretchBitMixer::new(
                    experts.len(),
                    &prior_weights,
                    *alpha,
                )))
            } else {
                MixtureBitMixPolicy::Simplex
            };
        Ok(Self {
            kind: *kind,
            schedule: *schedule,
            alpha: *alpha,
            decay: decay.unwrap_or(1.0).clamp(0.0, 1.0),
            experts,
            prior_weights,
            neural,
            bit_mix_policy,
            analyzer,
            bitwise_expert_states: Vec::new(),
            expert_observation_scratch: vec![0.0; plan_experts.len()],
            scratch: Vec::new(),
            scratch2: Vec::new(),
            projection_scratch: Vec::new(),
            weights_scratch: Vec::new(),
            pdf: vec![0.0; 256],
            valid: false,
            switch_updates: 0,
            convex_updates: 0,
        })
    }

    fn best_expert_index(&self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_loss = f64::INFINITY;
        for (index, expert) in self.experts.iter().enumerate() {
            if expert.cum_log_loss < best_loss {
                best_loss = expert.cum_log_loss;
                best_idx = Some(index);
            }
        }
        best_idx
    }

    // A free-standing helper (rather than a `&mut self` method) so callers can
    // pass `&mut self.weights_scratch` as the output buffer alongside other
    // disjoint fields of `self` without hitting the self-borrow conflict that
    // a `&mut self -> &[f64]` method signature would create at call sites
    // that still need to touch other fields of `self` afterward. `out` is
    // resized and fully overwritten; no allocation occurs once its capacity
    // has grown to `experts.len()`.
    #[allow(clippy::too_many_arguments)]
    fn compute_predictive_weights(
        kind: MixtureKind,
        decay: f64,
        experts: &[MixExpert],
        neural: Option<&mut NeuralMixCore>,
        analyzer: &TextContextAnalyzer,
        best_expert_index: Option<usize>,
        out: &mut Vec<f64>,
    ) {
        out.clear();
        if experts.is_empty() {
            return;
        }
        out.resize(experts.len(), 0.0);

        match kind {
            MixtureKind::Neural => {
                if experts.len() == 1 {
                    out[0] = 1.0;
                    return;
                }
                let neural =
                    neural.expect("multi-expert neural MixturePredictor must own a NeuralMixCore");
                neural.set_context_state(analyzer.state());
                neural.evaluate_expert_weights();
                out.copy_from_slice(neural.expert_weights());
                normalize_simplex_weights(out);
            }
            MixtureKind::Mdl => {
                if let Some(best_idx) = best_expert_index {
                    out[best_idx] = 1.0;
                }
            }
            MixtureKind::FadingBayes => {
                let max_log = experts
                    .iter()
                    .map(|expert| decay * expert.log_weight)
                    .fold(f64::NEG_INFINITY, f64::max);
                for (slot, expert) in out.iter_mut().zip(experts.iter()) {
                    *slot = if max_log.is_finite() {
                        (decay * expert.log_weight - max_log).exp()
                    } else {
                        0.0
                    };
                }
                normalize_simplex_weights(out);
            }
            MixtureKind::Convex => {
                for (slot, expert) in out.iter_mut().zip(experts.iter()) {
                    *slot = expert.log_weight.exp();
                }
                normalize_simplex_weights(out);
            }
            MixtureKind::Bayes | MixtureKind::Switching => {
                let max_log = experts
                    .iter()
                    .map(|expert| expert.log_weight)
                    .fold(f64::NEG_INFINITY, f64::max);
                for (slot, expert) in out.iter_mut().zip(experts.iter()) {
                    *slot = if max_log.is_finite() {
                        (expert.log_weight - max_log).exp()
                    } else {
                        0.0
                    };
                }
                normalize_simplex_weights(out);
            }
            MixtureKind::Logistic => {
                // Logistic is a stretch-domain bit mixer, not a simplex mixture.
                // A one-expert logistic spec is the identity of that expert,
                // which does have a truthful simplex row `[1]` for diagnostics.
                if experts.len() == 1 {
                    out[0] = 1.0;
                } else {
                    // PDF materialization uses `ensure_logistic_pdf`; no
                    // simplex weights exist for a true stretch mixer.
                    out.clear();
                }
            }
        }
    }

    fn refresh_predictive_weights(&mut self) {
        let best_idx = self.best_expert_index();
        Self::compute_predictive_weights(
            self.kind,
            self.decay,
            &self.experts,
            self.neural.as_mut(),
            &self.analyzer,
            best_idx,
            &mut self.weights_scratch,
        );
    }

    fn ensure_logistic_pdf(&mut self) -> Result<&[f64]> {
        if self.valid {
            return Ok(&self.pdf);
        }
        if self.experts.is_empty() {
            self.pdf.fill(0.0);
            self.valid = true;
            return Ok(&self.pdf);
        }
        if self.experts.len() == 1 {
            self.pdf
                .copy_from_slice(self.experts[0].predictor.pdf_next()?);
            normalize_pdf(&mut self.pdf, PDF_MIN);
            self.valid = true;
            return Ok(&self.pdf);
        }

        let MixtureBitMixPolicy::Stretch(mixer) = &mut self.bit_mix_policy else {
            bail!("multi-expert logistic mixture is missing its stretch bit-mix policy");
        };
        mixer.materialize_pdf(
            &mut self.experts,
            &self.analyzer,
            &mut self.expert_observation_scratch,
            &mut self.pdf,
        )?;
        self.valid = true;
        Ok(&self.pdf)
    }

    pub(super) fn ensure_pdf(&mut self) -> Result<&[f64]> {
        if self.valid {
            return Ok(&self.pdf);
        }
        if self.kind == MixtureKind::Logistic {
            return self.ensure_logistic_pdf();
        }
        self.refresh_predictive_weights();
        self.pdf.fill(0.0);
        for (index, expert) in self.experts.iter_mut().enumerate() {
            let weight = self.weights_scratch.get(index).copied().unwrap_or(0.0);
            if weight <= 0.0 {
                continue;
            }
            let epdf = expert.predictor.pdf_next()?;
            for (slot, &p) in self.pdf.iter_mut().zip(epdf.iter()) {
                *slot += weight * p;
            }
        }

        normalize_pdf(&mut self.pdf, PDF_MIN);
        self.valid = true;
        Ok(&self.pdf)
    }

    pub(super) fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        for expert in &mut self.experts {
            match &mut *expert.predictor {
                // Direct CTW benefits from pre-reserving, but inside mixtures that extra
                // headroom can dominate peak RSS without a proportional runtime gain.
                #[cfg(feature = "backend-ctw")]
                RatePdfPredictor::Ctw(_) | RatePdfPredictor::FacCtw(_) => {}
                _ => expert.predictor.begin_stream(total_len)?,
            }
        }
        Ok(())
    }

    fn diagnostic_collect_children(
        experts: &mut [MixExpert],
        symbol: u8,
        weights: &[f64],
        effective_prefix: f64,
        pool: Option<&ThreadPool>,
    ) -> Result<Vec<AcLogLossSubtreeSnapshot>> {
        let use_parallel = pool.is_some() && experts.len() >= DIAGNOSTIC_PARALLEL_THRESHOLD;
        if use_parallel {
            let pool = pool.expect("checked is_some");
            pool.install(|| {
                experts
                    .par_iter_mut()
                    .enumerate()
                    .map(|(index, expert)| {
                        let local_weight = weights.get(index).copied().unwrap_or(0.0);
                        let effective_weight = effective_prefix * local_weight;
                        expert.predictor.diagnostic_snapshot_subtree(
                            symbol,
                            local_weight,
                            effective_weight,
                            None,
                        )
                    })
                    .collect()
            })
        } else {
            let mut children = Vec::with_capacity(experts.len());
            for (index, expert) in experts.iter_mut().enumerate() {
                let local_weight = weights.get(index).copied().unwrap_or(0.0);
                let effective_weight = effective_prefix * local_weight;
                children.push(expert.predictor.diagnostic_snapshot_subtree(
                    symbol,
                    local_weight,
                    effective_weight,
                    pool,
                )?);
            }
            Ok(children)
        }
    }

    pub(super) fn diagnostic_subtree_snapshot(
        &mut self,
        symbol: u8,
        local_weight: f64,
        effective_weight: f64,
        pool: Option<&ThreadPool>,
    ) -> Result<AcLogLossSubtreeSnapshot> {
        if matches!(&self.bit_mix_policy, MixtureBitMixPolicy::Stretch(_)) {
            bail!("{LOGISTIC_DIAGNOSTICS_UNSUPPORTED}");
        }
        self.refresh_predictive_weights();
        let children = Self::diagnostic_collect_children(
            &mut self.experts,
            symbol,
            &self.weights_scratch,
            effective_weight,
            pool,
        )?;
        let mix_prob = children
            .iter()
            .enumerate()
            .map(|(index, child)| {
                self.weights_scratch.get(index).copied().unwrap_or(0.0) * child.prob
            })
            .sum::<f64>()
            .max(PDF_MIN);
        let total_rows = 1 + children.iter().map(|child| child.rows.len()).sum::<usize>();
        let mut rows = Vec::with_capacity(total_rows);
        rows.push(AcLogLossNodeValue {
            prob: mix_prob,
            local_weight,
            effective_weight,
        });
        for child in children {
            rows.extend(child.rows);
        }
        Ok(AcLogLossSubtreeSnapshot {
            prob: mix_prob,
            rows,
        })
    }

    pub(super) fn diagnostic_root_snapshot(
        &mut self,
        symbol: u8,
        pool: Option<&ThreadPool>,
        out: &mut Vec<AcLogLossNodeValue>,
    ) -> Result<AcLogLossRootSnapshot> {
        if matches!(&self.bit_mix_policy, MixtureBitMixPolicy::Stretch(_)) {
            bail!("{LOGISTIC_DIAGNOSTICS_UNSUPPORTED}");
        }
        self.refresh_predictive_weights();
        let children = Self::diagnostic_collect_children(
            &mut self.experts,
            symbol,
            &self.weights_scratch,
            1.0,
            pool,
        )?;
        out.clear();
        out.reserve(children.iter().map(|child| child.rows.len()).sum::<usize>());
        for child in &children {
            out.extend_from_slice(&child.rows);
        }

        let mix_prob = children
            .iter()
            .enumerate()
            .map(|(index, child)| {
                self.weights_scratch.get(index).copied().unwrap_or(0.0) * child.prob
            })
            .sum::<f64>()
            .max(PDF_MIN);

        let mut top1 = None;
        let mut top2 = None;
        for (index, &weight) in self.weights_scratch.iter().enumerate() {
            match top1 {
                None => top1 = Some((index, weight)),
                Some((best_idx, best_weight)) if weight > best_weight => {
                    top2 = Some((best_idx, best_weight));
                    top1 = Some((index, weight));
                }
                _ => match top2 {
                    None => top2 = Some((index, weight)),
                    Some((_, second_weight)) if weight > second_weight => {
                        top2 = Some((index, weight));
                    }
                    _ => {}
                },
            }
        }

        let root_weight_entropy_bits = self
            .weights_scratch
            .iter()
            .copied()
            .filter(|weight| *weight > 0.0)
            .map(|weight| -weight * weight.log2())
            .sum::<f64>();
        Ok(AcLogLossRootSnapshot {
            mix_prob,
            root_weight_entropy_bits,
            root_top1_child_index: top1.map(|(index, _)| index),
            root_top1_weight: top1.map(|(_, weight)| weight).unwrap_or(0.0),
            root_top2_child_index: top2.map(|(index, _)| index),
            root_top2_weight: top2.map(|(_, weight)| weight).unwrap_or(0.0),
        })
    }

    pub(super) fn update(&mut self, symbol: u8) -> Result<()> {
        let _ = self.ensure_pdf()?;

        match self.kind {
            MixtureKind::Bayes => {
                let n = self.experts.len();
                self.scratch.resize(n, 0.0);
                self.scratch2.resize(n, 0.0);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    let p = e.predictor.pdf_next()?[symbol as usize].max(PDF_MIN);
                    let lp = p.ln();
                    self.scratch[i] = lp;
                    self.scratch2[i] = e.log_weight + lp;
                }
                let log_mix = logsumexp_slice(&self.scratch2[..n]);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    e.log_weight = e.log_weight + self.scratch[i] - log_mix;
                    e.cum_log_loss -= self.scratch[i];
                    e.predictor.update(symbol)?;
                }
            }
            MixtureKind::FadingBayes => {
                let n = self.experts.len();
                self.scratch.resize(n, 0.0);
                self.scratch2.resize(n, 0.0);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    let p = e.predictor.pdf_next()?[symbol as usize].max(PDF_MIN);
                    let lp = p.ln();
                    self.scratch[i] = lp;
                }
                for (i, e) in self.experts.iter_mut().enumerate() {
                    self.scratch2[i] = self.decay * e.log_weight + self.scratch[i];
                }
                let log_mix = logsumexp_slice(&self.scratch2[..n]);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    e.log_weight = self.decay * e.log_weight + self.scratch[i] - log_mix;
                    e.cum_log_loss -= self.scratch[i];
                    e.predictor.update(symbol)?;
                }
            }
            MixtureKind::Switching => {
                let n = self.experts.len();
                self.scratch.resize(n, 0.0);
                self.scratch2.resize(n, 0.0);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    let p = e.predictor.pdf_next()?[symbol as usize].max(PDF_MIN);
                    let lp = p.ln();
                    self.scratch[i] = lp;
                    self.scratch2[i] = e.log_weight + lp;
                }
                let log_mix = logsumexp_slice(&self.scratch2[..n]);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    self.scratch2[i] = (self.scratch2[i] - log_mix).exp();
                    e.cum_log_loss -= self.scratch[i];
                    e.predictor.update(symbol)?;
                }
                let alpha =
                    switching_alpha_for_update(self.schedule, self.alpha, self.switch_updates);
                self.switch_updates = self.switch_updates.saturating_add(1);
                apply_switching_weights(
                    &mut self.experts,
                    &self.prior_weights[..n],
                    alpha,
                    &mut self.scratch2[..n],
                    &mut self.scratch[..n],
                );
            }
            MixtureKind::Convex => {
                let n = self.experts.len();
                self.scratch.resize(n, 0.0);
                self.scratch2.resize(n, 0.0);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    let p = e.predictor.pdf_next()?[symbol as usize].max(PDF_MIN);
                    let lp = p.ln();
                    self.scratch[i] = lp;
                    self.scratch2[i] = e.log_weight.exp();
                    e.cum_log_loss -= lp;
                    e.predictor.update(symbol)?;
                }
                let mix_prob = self
                    .scratch
                    .iter()
                    .zip(self.scratch2.iter())
                    .map(|(&lp, &w)| w * lp.exp())
                    .sum::<f64>()
                    .max(PDF_MIN);
                let log_mix = mix_prob.ln();
                self.convex_updates = self.convex_updates.saturating_add(1);
                let eta =
                    convex_step_size_for_update(self.schedule, self.alpha, self.convex_updates);
                for i in 0..n {
                    let grad = -(self.scratch[i] - log_mix).exp();
                    self.scratch2[i] -= eta * grad;
                }
                project_simplex_with_scratch(&mut self.scratch2[..n], &mut self.projection_scratch);
                for i in 0..n {
                    self.experts[i].log_weight = self.scratch2[i].max(PDF_MIN).ln();
                }
            }
            MixtureKind::Mdl => {
                let n = self.experts.len();
                self.scratch.resize(n, 0.0);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    let p = e.predictor.pdf_next()?[symbol as usize].max(PDF_MIN);
                    let lp = p.ln();
                    self.scratch[i] = lp;
                }
                for (i, e) in self.experts.iter_mut().enumerate() {
                    e.cum_log_loss -= self.scratch[i];
                    e.predictor.update(symbol)?;
                }
            }
            MixtureKind::Neural => {
                let y = symbol as usize;
                if self.experts.len() == 1 {
                    let lp = self.experts[0].predictor.pdf_next()?[y].max(PDF_MIN).ln();
                    self.experts[0].cum_log_loss -= lp;
                    self.experts[0].predictor.update(symbol)?;
                    self.analyzer.update(symbol);
                    self.valid = false;
                    return Ok(());
                }
                let n = self.experts.len();
                let history = self.analyzer.state();
                let neural = self
                    .neural
                    .as_mut()
                    .expect("multi-expert neural MixturePredictor must own a NeuralMixCore");
                neural.set_context_state(history);
                self.expert_observation_scratch.resize(n, 0.0);
                for i in 0..n {
                    let p = self.experts[i].predictor.pdf_next()?[y].max(PDF_MIN);
                    let lp = p.ln();
                    self.expert_observation_scratch[i] = lp;
                    self.experts[i].cum_log_loss -= lp;
                }
                neural.evaluate_symbol(&self.expert_observation_scratch, PDF_MIN);
                neural.update_weights_symbol(&self.expert_observation_scratch, PDF_MIN);
                for e in &mut self.experts {
                    e.predictor.update(symbol)?;
                }
                self.analyzer.update(symbol);
                neural.set_context_state(self.analyzer.state());
            }
            MixtureKind::Logistic => {
                self.update_logistic_symbol(symbol)?;
            }
        }

        self.valid = false;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn has_logistic_mixer(&self) -> bool {
        matches!(&self.bit_mix_policy, MixtureBitMixPolicy::Stretch(_))
    }

    #[cfg(test)]
    pub(super) fn has_neural_mixer(&self) -> bool {
        self.neural.is_some()
    }

    fn update_logistic_symbol(&mut self, symbol: u8) -> Result<()> {
        if self.experts.is_empty() {
            return Ok(());
        }
        if !matches!(&self.bit_mix_policy, MixtureBitMixPolicy::Stretch(_)) {
            let y = symbol as usize;
            let lp = self.experts[0].predictor.pdf_next()?[y].max(PDF_MIN).ln();
            self.experts[0].cum_log_loss -= lp;
            self.experts[0].predictor.update(symbol)?;
            self.analyzer.update(symbol);
            return Ok(());
        }
        if !self.begin_bitwise_byte_step()? {
            bail!("logistic mixture requires an active bitwise byte step");
        }
        for bit_idx in 0..8usize {
            let bit = (symbol & (1u8 << (7 - bit_idx))) != 0;
            let _ = self.observe_known_bit_msb(bit_idx, bit)?;
        }
        self.finish_bitwise_symbol(symbol)
    }

    pub(super) fn finish_stream(&mut self) -> Result<()> {
        for expert in &mut self.experts {
            expert.predictor.finish_stream()?;
        }
        Ok(())
    }

    #[inline]
    pub(super) fn has_recursive_native_bitwise_expert(&self) -> bool {
        let experts_have_recursive = self
            .experts
            .iter()
            .any(|expert| expert.predictor.has_recursive_native_bitwise_path());
        if self.kind == MixtureKind::Logistic {
            return match &self.bit_mix_policy {
                // A true stretch mixer can derive every bit from expert PDFs.
                MixtureBitMixPolicy::Stretch(_) => !self.experts.is_empty(),
                // One-expert logistic is the exact expert identity.
                MixtureBitMixPolicy::Simplex => experts_have_recursive,
            };
        }
        experts_have_recursive
    }

    pub(super) fn begin_bitwise_byte_step(&mut self) -> Result<bool> {
        // Multi-expert logistic reports recursive-native via the stretch mixer
        // (PDF-prefix fallback for non-native experts). One-expert logistic
        // reports only what the sole expert itself exposes.
        if !self.has_recursive_native_bitwise_expert() {
            return Ok(false);
        }

        let n = self.experts.len();
        self.scratch.resize(n, 0.0);
        match self.kind {
            MixtureKind::Neural if n > 1 => {
                let neural = self
                    .neural
                    .as_mut()
                    .expect("multi-expert neural MixturePredictor must own a NeuralMixCore");
                neural.set_context_state(self.analyzer.state());
                neural.evaluate_expert_weights();
                self.scratch.copy_from_slice(neural.expert_weights());
            }
            MixtureKind::Logistic => {
                self.scratch.fill(0.0);
                match &mut self.bit_mix_policy {
                    MixtureBitMixPolicy::Stretch(mixer) => mixer.begin_byte_step(&mut self.experts),
                    MixtureBitMixPolicy::Simplex => {
                        // Identity: unit-weight bit mix over the sole expert.
                        if n == 1 {
                            self.scratch[0] = 1.0;
                        }
                    }
                }
            }
            _ => {
                self.refresh_predictive_weights();
                self.scratch.copy_from_slice(&self.weights_scratch);
            }
        }
        self.scratch2.resize(n, 1.0);
        self.scratch2.fill(1.0);
        self.expert_observation_scratch.resize(n, 0.0);
        self.bitwise_expert_states
            .resize_with(n, PredictorBitwiseStepState::default);
        for i in 0..n {
            self.bitwise_expert_states[i].prepare(&mut self.experts[i].predictor)?;
        }
        Ok(true)
    }

    pub(super) fn bit_prob_one_msb(&mut self, bit_idx: usize) -> Result<f64> {
        if let MixtureBitMixPolicy::Stretch(mixer) = &mut self.bit_mix_policy {
            let n = self.experts.len();
            for i in 0..n {
                let p1 = self.bitwise_expert_states[i]
                    .bit_prob_one_msb(&mut self.experts[i].predictor, bit_idx)?;
                self.expert_observation_scratch[i] = p1;
            }
            return Ok(mixer.predict_bit(
                &self.analyzer,
                bit_idx,
                &self.expert_observation_scratch[..n],
            ));
        }
        let mut denom = 0.0;
        let mut numer1 = 0.0;
        for i in 0..self.experts.len() {
            let p1 = self.bitwise_expert_states[i]
                .bit_prob_one_msb(&mut self.experts[i].predictor, bit_idx)?;
            self.expert_observation_scratch[i] = p1;
            let wp = self.scratch[i] * self.scratch2[i];
            denom += wp;
            numer1 += wp * p1;
        }
        Ok(if denom.is_finite() && denom > 0.0 {
            (numer1 / denom).clamp(PDF_MIN, 1.0 - PDF_MIN)
        } else {
            panic!(
                "MixturePredictor bit_prob_one_msb: invalid denom (finite>0 violated); \
                 this is an internal invariant failure in the bitwise mixture state machine"
            )
        })
    }

    pub(super) fn observe_bit_msb(&mut self, bit_idx: usize, bit: bool) -> Result<()> {
        if let MixtureBitMixPolicy::Stretch(mixer) = &mut self.bit_mix_policy {
            let n = self.experts.len();
            let _ = mixer.observe_bit(
                &self.analyzer,
                bit_idx,
                &self.expert_observation_scratch[..n],
                bit,
            );
            for i in 0..n {
                let p1 = self.expert_observation_scratch[i];
                let pb = if bit { p1 } else { 1.0 - p1 };
                self.scratch2[i] = (self.scratch2[i] * pb).max(PDF_MIN);
                self.bitwise_expert_states[i].observe_bit_msb(
                    &mut self.experts[i].predictor,
                    bit_idx,
                    bit,
                )?;
            }
            return Ok(());
        }
        for i in 0..self.experts.len() {
            let p1 = self.expert_observation_scratch[i];
            let pb = if bit { p1 } else { 1.0 - p1 };
            self.scratch2[i] = (self.scratch2[i] * pb).max(PDF_MIN);
            self.bitwise_expert_states[i].observe_bit_msb(
                &mut self.experts[i].predictor,
                bit_idx,
                bit,
            )?;
        }
        Ok(())
    }

    pub(super) fn observe_known_bit_msb(&mut self, bit_idx: usize, bit: bool) -> Result<f64> {
        if let MixtureBitMixPolicy::Stretch(mixer) = &mut self.bit_mix_policy {
            let n = self.experts.len();
            for i in 0..n {
                let p1 = self.bitwise_expert_states[i]
                    .bit_prob_one_msb(&mut self.experts[i].predictor, bit_idx)?;
                self.expert_observation_scratch[i] = p1;
            }
            let p_mix = mixer.observe_bit(
                &self.analyzer,
                bit_idx,
                &self.expert_observation_scratch[..n],
                bit,
            );
            for i in 0..n {
                let p1 = self.expert_observation_scratch[i];
                let pb = if bit { p1 } else { 1.0 - p1 };
                self.scratch2[i] = (self.scratch2[i] * pb).max(PDF_MIN);
                self.bitwise_expert_states[i].observe_bit_msb(
                    &mut self.experts[i].predictor,
                    bit_idx,
                    bit,
                )?;
            }
            return Ok(p_mix);
        }
        let mut denom = 0.0;
        let mut numer1 = 0.0;
        for i in 0..self.experts.len() {
            let p1 = self.bitwise_expert_states[i]
                .bit_prob_one_msb(&mut self.experts[i].predictor, bit_idx)?;
            let wp = self.scratch[i] * self.scratch2[i];
            denom += wp;
            numer1 += wp * p1;
            let pb = if bit { p1 } else { 1.0 - p1 };
            self.scratch2[i] = (self.scratch2[i] * pb).max(PDF_MIN);
            self.bitwise_expert_states[i].observe_bit_msb(
                &mut self.experts[i].predictor,
                bit_idx,
                bit,
            )?;
        }

        Ok(if denom.is_finite() && denom > 0.0 {
            (numer1 / denom).clamp(PDF_MIN, 1.0 - PDF_MIN)
        } else {
            panic!(
                "MixturePredictor observe_known_bit_msb: invalid denom (finite>0 violated); \
                 this is an internal invariant failure in the bitwise mixture state machine"
            )
        })
    }

    pub(super) fn finish_bitwise_symbol(&mut self, symbol: u8) -> Result<()> {
        let n = self.experts.len();
        for i in 0..n {
            let lp = self.scratch2[i].max(PDF_MIN).ln();
            self.expert_observation_scratch[i] = lp;
            self.experts[i].cum_log_loss -= lp;
            self.bitwise_expert_states[i].finish_symbol(&mut self.experts[i].predictor, symbol)?;
        }

        match self.kind {
            MixtureKind::Bayes => {
                for i in 0..n {
                    self.scratch[i] =
                        self.experts[i].log_weight + self.expert_observation_scratch[i];
                }
                let log_mix = logsumexp_slice(&self.scratch[..n]);
                for i in 0..n {
                    self.experts[i].log_weight += self.expert_observation_scratch[i] - log_mix;
                }
            }
            MixtureKind::FadingBayes => {
                for i in 0..n {
                    self.scratch[i] = self.decay * self.experts[i].log_weight
                        + self.expert_observation_scratch[i];
                }
                let log_mix = logsumexp_slice(&self.scratch[..n]);
                for i in 0..n {
                    self.experts[i].log_weight = self.scratch[i] - log_mix;
                }
            }
            MixtureKind::Switching => {
                for i in 0..n {
                    self.scratch[i] =
                        self.experts[i].log_weight + self.expert_observation_scratch[i];
                }
                let log_mix = logsumexp_slice(&self.scratch[..n]);
                for weight in &mut self.scratch[..n] {
                    *weight = (*weight - log_mix).exp();
                }
                let alpha =
                    switching_alpha_for_update(self.schedule, self.alpha, self.switch_updates);
                self.switch_updates = self.switch_updates.saturating_add(1);
                apply_switching_weights(
                    &mut self.experts,
                    &self.prior_weights[..n],
                    alpha,
                    &mut self.scratch[..n],
                    &mut self.scratch2[..n],
                );
            }
            MixtureKind::Convex => {
                self.scratch.resize(n, 0.0);
                self.scratch2.resize(n, 0.0);
                for i in 0..n {
                    self.scratch2[i] = self.experts[i].log_weight.exp();
                }
                let mix_prob = self
                    .expert_observation_scratch
                    .iter()
                    .zip(self.scratch2.iter())
                    .map(|(&lp, &w)| w * lp.exp())
                    .sum::<f64>()
                    .max(PDF_MIN);
                let log_mix = mix_prob.ln();
                self.convex_updates = self.convex_updates.saturating_add(1);
                let eta =
                    convex_step_size_for_update(self.schedule, self.alpha, self.convex_updates);
                for i in 0..n {
                    let grad = -(self.expert_observation_scratch[i] - log_mix).exp();
                    self.scratch2[i] -= eta * grad;
                }
                project_simplex_with_scratch(&mut self.scratch2[..n], &mut self.projection_scratch);
                for i in 0..n {
                    self.experts[i].log_weight = self.scratch2[i].max(PDF_MIN).ln();
                }
            }
            MixtureKind::Mdl => {}
            MixtureKind::Neural => {
                if n > 1 {
                    let history = self.analyzer.state();
                    let neural = self
                        .neural
                        .as_mut()
                        .expect("multi-expert neural MixturePredictor must own a NeuralMixCore");
                    neural.set_context_state(history);
                    neural.evaluate_symbol(&self.expert_observation_scratch, PDF_MIN);
                    neural.update_weights_symbol(&self.expert_observation_scratch, PDF_MIN);
                }
                self.analyzer.update(symbol);
                if let Some(neural) = self.neural.as_mut() {
                    neural.set_context_state(self.analyzer.state());
                }
            }
            MixtureKind::Logistic => {
                self.analyzer.update(symbol);
                if let MixtureBitMixPolicy::Stretch(mixer) = &mut self.bit_mix_policy {
                    mixer.finish_byte(&self.analyzer);
                }
            }
        }
        self.valid = false;
        Ok(())
    }

    pub(super) fn logistic_match_state(&mut self) -> LogisticMatchState {
        aggregate_compression_logistic_match_state(&mut self.experts)
    }
}
fn aggregate_compression_logistic_match_state(experts: &mut [MixExpert]) -> LogisticMatchState {
    fold_logistic_match_states(
        experts
            .iter_mut()
            .map(|e| e.predictor.logistic_match_state()),
    )
}

#[inline]
fn logsumexp_slice(vals: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in vals {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &v in vals {
        s += (v - m).exp();
    }
    m + s.ln()
}

#[inline]
fn logsumexp_expert_weights(experts: &[MixExpert]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for e in experts {
        if e.log_weight > m {
            m = e.log_weight;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for e in experts {
        s += (e.log_weight - m).exp();
    }
    m + s.ln()
}

fn normalize_simplex_weights(weights: &mut [f64]) {
    if weights.is_empty() {
        return;
    }
    let mut sum = 0.0;
    for weight in weights.iter_mut() {
        if !weight.is_finite() || *weight < 0.0 {
            *weight = 0.0;
        }
        sum += *weight;
    }
    if !sum.is_finite() || sum <= 0.0 {
        let uniform = 1.0 / (weights.len() as f64);
        weights.fill(uniform);
        return;
    }
    for weight in weights.iter_mut() {
        *weight /= sum;
    }
}

fn normalized_mix_expert_prior_weights(experts: &[MixExpert], out: &mut [f64]) {
    debug_assert_eq!(experts.len(), out.len());
    let max_log = experts
        .iter()
        .map(|expert| expert.log_prior)
        .fold(f64::NEG_INFINITY, f64::max);
    for (slot, expert) in out.iter_mut().zip(experts.iter()) {
        *slot = if max_log.is_finite() {
            (expert.log_prior - max_log).exp()
        } else {
            0.0
        };
    }
    normalize_simplex_weights(out);
}

fn set_mix_expert_log_weights_from_linear(experts: &mut [MixExpert], weights: &[f64]) {
    for (expert, &weight) in experts.iter_mut().zip(weights.iter()) {
        expert.log_weight = if weight > 0.0 {
            weight.ln()
        } else {
            f64::NEG_INFINITY
        };
    }
}

fn apply_switching_weights(
    experts: &mut [MixExpert],
    prior_weights: &[f64],
    alpha: f64,
    posterior: &mut [f64],
    scratch: &mut [f64],
) {
    if experts.is_empty() {
        return;
    }
    debug_assert_eq!(experts.len(), prior_weights.len());

    normalize_simplex_weights(posterior);
    if experts.len() == 1 || alpha <= 0.0 {
        set_mix_expert_log_weights_from_linear(experts, posterior);
        return;
    }

    let num_switch_targets = prior_weights.iter().filter(|&&prior| prior < 1.0).count();
    if num_switch_targets <= 1 {
        set_mix_expert_log_weights_from_linear(experts, posterior);
        return;
    }

    let mut switch_out_sum = 0.0;
    for i in 0..experts.len() {
        let denom = 1.0 - prior_weights[i];
        if denom > 0.0 {
            switch_out_sum += posterior[i] / denom;
        }
    }

    for i in 0..experts.len() {
        let prior = prior_weights[i];
        let stay = (1.0 - alpha) * posterior[i];
        let switch_in = if prior > 0.0 {
            let denom = 1.0 - prior;
            let switchable_mass = if denom > 0.0 {
                switch_out_sum - posterior[i] / denom
            } else {
                0.0
            };
            alpha * prior * switchable_mass
        } else {
            0.0
        };
        scratch[i] = stay + switch_in;
    }

    normalize_simplex_weights(scratch);
    set_mix_expert_log_weights_from_linear(experts, scratch);
}
