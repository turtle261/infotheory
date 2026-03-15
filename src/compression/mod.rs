//! Rate-coded compression helpers (AC/rANS) with optional framing.
//!
//! The functions in this module implement lossless byte compression by combining:
//! - a predictive rate model (`RateBackend`) that emits per-symbol PDFs,
//! - an entropy coder (`AC` or `rANS`),
//! - optional framing metadata for robust decompression.

use anyhow::{Result, bail};

use crate::backends::calibration::CalibratorCore;
use crate::backends::match_model::MatchModel;
use crate::backends::ppmd::PpmdModel;
use crate::backends::sparse_match::SparseMatchModel;
use crate::backends::text_context::TextContextAnalyzer;
use crate::coders::{
    ANS_TOTAL, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder, BlockedRansEncoder,
    CDF_TOTAL, Cdf, CoderType, crc32, quantize_pdf_to_rans_cdf_with_buffer,
};
use crate::ctw::FacContextTree;
#[cfg(feature = "backend-mamba")]
use crate::mambazip;
use crate::mixture::DEFAULT_MIN_PROB;
use crate::neural_mix::NeuralMixCore;
use crate::rosaplus::RosaPlus;
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip;
use crate::zpaq_rate::ZpaqRateModel;
use crate::{CalibratedSpec, MixtureKind, MixtureSpec, RateBackend};

const FRAMED_MAGIC: u32 = 0x4354_4946; // "FITC"
const FRAMED_VERSION: u8 = 1;
const PDF_MIN: f64 = DEFAULT_MIN_PROB;

#[inline]
fn build_calibrator(spec: &CalibratedSpec) -> CalibratorCore {
    CalibratorCore::new(spec.context, spec.bins, spec.learning_rate, spec.bias_clip)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Wire format mode for rate-coded payloads.
pub enum FramingMode {
    /// Emit only coder payload bytes (no integrity/length header).
    Raw,
    /// Emit framed payload with magic/version/length/checksum header.
    #[default]
    Framed,
}

#[derive(Clone, Copy, Debug)]
struct FramedHeader {
    magic: u32,
    version: u8,
    coder: u8,
    original_len: u64,
    crc32: u32,
}

impl FramedHeader {
    const SIZE: usize = 4 + 1 + 1 + 8 + 4;

    fn new(coder: CoderType, original_len: u64, crc32: u32) -> Self {
        Self {
            magic: FRAMED_MAGIC,
            version: FRAMED_VERSION,
            coder: match coder {
                CoderType::AC => 0,
                CoderType::RANS => 1,
            },
            original_len,
            crc32,
        }
    }

    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.magic.to_le_bytes());
        out.push(self.version);
        out.push(self.coder);
        out.extend_from_slice(&self.original_len.to_le_bytes());
        out.extend_from_slice(&self.crc32.to_le_bytes());
    }

    fn read(input: &[u8]) -> Result<Self> {
        if input.len() < Self::SIZE {
            bail!("framed payload too short");
        }
        let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        if magic != FRAMED_MAGIC {
            bail!("invalid framed magic: expected 0x{FRAMED_MAGIC:08X}, got 0x{magic:08X}");
        }
        let version = input[4];
        if version != FRAMED_VERSION {
            bail!("unsupported framed version: {version}");
        }
        let coder = input[5];
        let original_len = u64::from_le_bytes([
            input[6], input[7], input[8], input[9], input[10], input[11], input[12], input[13],
        ]);
        let crc32 = u32::from_le_bytes([input[14], input[15], input[16], input[17]]);
        Ok(Self {
            magic,
            version,
            coder,
            original_len,
            crc32,
        })
    }

    fn coder_type(&self) -> CoderType {
        match self.coder {
            0 => CoderType::AC,
            _ => CoderType::RANS,
        }
    }
}

#[derive(Clone)]
struct CtwPredictor {
    tree: FacContextTree,
    bits_per_symbol: usize,
    msb_first: bool,
    pdf: Vec<f64>,
    pattern_logps: Vec<f64>,
    valid: bool,
}

impl CtwPredictor {
    fn new_ctw(depth: usize) -> Self {
        Self {
            tree: FacContextTree::new(depth, 8),
            bits_per_symbol: 8,
            msb_first: true,
            pdf: vec![0.0; 256],
            pattern_logps: vec![f64::NEG_INFINITY; 256],
            valid: false,
        }
    }

    fn new_fac(base_depth: usize, bits_per_symbol: usize) -> Self {
        Self {
            tree: FacContextTree::new(base_depth, bits_per_symbol),
            bits_per_symbol,
            msb_first: false,
            pdf: vec![0.0; 256],
            pattern_logps: vec![f64::NEG_INFINITY; 256],
            valid: false,
        }
    }

    fn fill_pattern_log_probs(&mut self) -> usize {
        fn rec(
            tree: &mut FacContextTree,
            bits: usize,
            msb_first: bool,
            depth: usize,
            pattern: usize,
            log_before: f64,
            out: &mut [f64],
        ) {
            if depth == bits {
                out[pattern] = tree.get_log_block_probability() - log_before;
                return;
            }
            for bit in [false, true] {
                tree.update(bit, depth);
                let next_pattern = if msb_first {
                    (pattern << 1) | (bit as usize)
                } else {
                    pattern | ((bit as usize) << depth)
                };
                rec(
                    tree,
                    bits,
                    msb_first,
                    depth + 1,
                    next_pattern,
                    log_before,
                    out,
                );
                tree.revert(depth);
            }
        }

        let bits = self.bits_per_symbol.clamp(1, 8);
        let patterns = 1usize << bits;
        let log_before = self.tree.get_log_block_probability();
        self.pattern_logps[..patterns].fill(f64::NEG_INFINITY);
        rec(
            &mut self.tree,
            bits,
            self.msb_first,
            0,
            0,
            log_before,
            &mut self.pattern_logps[..patterns],
        );
        patterns
    }

    #[cfg(test)]
    fn log_prob_symbol_bruteforce(&mut self, symbol: u8) -> f64 {
        let bits = self.bits_per_symbol.clamp(1, 8);
        let before = self.tree.get_log_block_probability();
        if self.msb_first {
            for bit_idx in 0..bits {
                let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
                self.tree.update(bit, bit_idx);
            }
            let after = self.tree.get_log_block_probability();
            for bit_idx in (0..bits).rev() {
                self.tree.revert(bit_idx);
            }
            after - before
        } else {
            for bit_idx in 0..bits {
                let bit = ((symbol >> bit_idx) & 1) == 1;
                self.tree.update(bit, bit_idx);
            }
            let after = self.tree.get_log_block_probability();
            for bit_idx in (0..bits).rev() {
                self.tree.revert(bit_idx);
            }
            after - before
        }
    }

    fn normalize_pdf(pdf: &mut [f64]) {
        let mut sum = 0.0f64;
        for p in pdf.iter_mut() {
            let v = if p.is_finite() { *p } else { 0.0 };
            *p = v.max(PDF_MIN);
            sum += *p;
        }
        if sum <= 0.0 || !sum.is_finite() {
            let u = 1.0 / (pdf.len() as f64);
            for p in pdf.iter_mut() {
                *p = u;
            }
            return;
        }
        let inv = 1.0 / sum;
        for p in pdf.iter_mut() {
            *p *= inv;
        }
    }

    fn pdf_next(&mut self) -> &[f64] {
        if !self.valid {
            let bits = self.bits_per_symbol.clamp(1, 8);
            let patterns = self.fill_pattern_log_probs();
            if bits == 8 {
                for sym in 0..256usize {
                    self.pdf[sym] = self.pattern_logps[sym].exp();
                }
            } else {
                let aliases = 1usize << (8 - bits);
                for byte in 0..256usize {
                    let pat = if self.msb_first {
                        byte >> (8 - bits)
                    } else {
                        byte & (patterns - 1)
                    };
                    self.pdf[byte] = self.pattern_logps[pat].exp() / (aliases as f64);
                }
            }
            Self::normalize_pdf(&mut self.pdf);
            self.valid = true;
        }
        &self.pdf
    }

    fn update(&mut self, symbol: u8) {
        if self.msb_first {
            for bit_idx in 0..self.bits_per_symbol {
                let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
                self.tree.update(bit, bit_idx);
            }
        } else {
            for bit_idx in 0..self.bits_per_symbol {
                let bit = ((symbol >> bit_idx) & 1) == 1;
                self.tree.update(bit, bit_idx);
            }
        }
        self.valid = false;
    }

    #[inline]
    fn can_fast_ac_bitwise(&self) -> bool {
        self.bits_per_symbol == 8 && self.msb_first
    }

    #[inline]
    fn bit_prob_one_msb(&mut self, bit_idx: usize) -> f64 {
        debug_assert!(self.can_fast_ac_bitwise());
        self.tree
            .predict(true, bit_idx)
            .clamp(PDF_MIN, 1.0 - PDF_MIN)
    }

    #[inline]
    fn update_bit_msb(&mut self, bit_idx: usize, bit: bool) {
        debug_assert!(self.can_fast_ac_bitwise());
        self.tree.update_predicted(bit, bit_idx);
        self.valid = false;
    }
}

#[derive(Clone)]
struct RosaPredictor {
    model: RosaPlus,
    pdf: Vec<f64>,
    cdf: [f64; 257],
    valid: bool,
    cdf_valid: bool,
}

impl RosaPredictor {
    fn new(max_order: i64) -> Self {
        let mut model = RosaPlus::new(max_order, false, 0, 42);
        model.build_lm_full_bytes_no_finalize_endpos();
        Self {
            model,
            pdf: vec![0.0; 256],
            cdf: uniform_cdf_row(),
            valid: false,
            cdf_valid: false,
        }
    }

    fn pdf_next(&mut self) -> &[f64] {
        self.ensure_pdf(false);
        &self.pdf
    }

    fn cdf_next(&mut self) -> &[f64; 257] {
        self.ensure_pdf(true);
        &self.cdf
    }

    fn ensure_pdf(&mut self, want_cdf: bool) {
        if self.valid {
            if want_cdf && !self.cdf_valid {
                build_cdf_row_from_pdf_slice(&self.pdf, &mut self.cdf);
                self.cdf_valid = true;
            }
            return;
        }
        self.model.fill_probs_for_last_bytes(&mut self.pdf);
        normalize_pdf_vec_and_maybe_build_cdf(
            &mut self.pdf,
            if want_cdf { Some(&mut self.cdf) } else { None },
        );
        self.valid = true;
        self.cdf_valid = want_cdf;
    }

    fn update(&mut self, symbol: u8) {
        self.model.train_byte(symbol);
        self.valid = false;
        self.cdf_valid = false;
    }

    fn begin_stream(&mut self, total_len: usize) {
        self.model.reserve_for_stream(total_len);
    }
}

#[derive(Clone)]
#[cfg(feature = "backend-mamba")]
struct MambaPredictor {
    compressor: mambazip::Compressor,
    primed: bool,
    pdf: Vec<f64>,
    valid: bool,
}

#[derive(Clone)]
#[cfg(feature = "backend-rwkv")]
struct RwkvPredictor {
    compressor: rwkvzip::Compressor,
    primed: bool,
}

#[derive(Clone)]
struct ZpaqPredictor {
    method: String,
    history: Vec<u8>,
    pdf: Vec<f64>,
    valid: bool,
}

impl ZpaqPredictor {
    fn new(method: String) -> Self {
        Self {
            method,
            history: Vec::new(),
            pdf: vec![0.0; 256],
            valid: false,
        }
    }

    fn pdf_next(&mut self) -> &[f64] {
        if !self.valid {
            for sym in 0..256usize {
                let mut model = ZpaqRateModel::new(self.method.clone(), PDF_MIN);
                if !self.history.is_empty() {
                    let _ = model.update_and_score(&self.history);
                }
                let logp = model.log_prob(sym as u8);
                self.pdf[sym] = logp.exp().max(PDF_MIN);
            }
            normalize_pdf(&mut self.pdf);
            self.valid = true;
        }
        &self.pdf
    }

    fn update(&mut self, symbol: u8) {
        self.history.push(symbol);
        self.valid = false;
    }
}

#[cfg(feature = "backend-mamba")]
impl MambaPredictor {
    fn from_model(model: std::sync::Arc<mambazip::Model>) -> Self {
        let compressor = mambazip::Compressor::new_from_model(model);
        let vocab = compressor.vocab_size();
        Self {
            compressor,
            primed: false,
            pdf: vec![0.0; vocab],
            valid: false,
        }
    }

    fn from_method(method: &str) -> Result<Self> {
        let compressor = mambazip::Compressor::new_from_method(method)?;
        let vocab = compressor.vocab_size();
        Ok(Self {
            compressor,
            primed: false,
            pdf: vec![0.0; vocab],
            valid: false,
        })
    }

    fn ensure_predicted(&mut self) {
        if self.valid {
            return;
        }
        if !self.primed {
            self.compressor.forward_to_pdf(0, &mut self.pdf);
            self.primed = true;
            self.valid = true;
            return;
        }
        self.valid = true;
    }

    fn pdf_next(&mut self) -> &[f64] {
        self.ensure_predicted();
        &self.pdf
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        self.ensure_predicted();
        self.compressor.online_update_from_pdf(symbol, &self.pdf)?;
        self.compressor.forward_to_pdf(symbol as u32, &mut self.pdf);
        self.valid = true;
        Ok(())
    }

    fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        self.compressor
            .begin_online_policy_stream(Some(total_len as u64))
    }
}

#[cfg(feature = "backend-rwkv")]
impl RwkvPredictor {
    fn from_model(model: std::sync::Arc<rwkvzip::Model>) -> Self {
        let compressor = rwkvzip::Compressor::new_from_model(model);
        Self {
            compressor,
            primed: false,
        }
    }

    fn from_method(method: &str) -> Result<Self> {
        let compressor = rwkvzip::Compressor::new_from_method(method)?;
        Ok(Self {
            compressor,
            primed: false,
        })
    }

    fn ensure_predicted(&mut self) {
        if !self.primed {
            self.compressor.reset_and_prime();
            self.primed = true;
        }
    }

    fn pdf_next(&mut self) -> &[f64] {
        self.ensure_predicted();
        &self.compressor.pdf_buffer
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        self.ensure_predicted();
        self.compressor.observe_symbol_from_current_pdf(symbol)?;
        Ok(())
    }

    fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        self.compressor
            .begin_online_policy_stream(Some(total_len as u64))
    }

    fn finish_stream(&mut self) -> Result<()> {
        self.compressor.finish_online_policy_stream()
    }
}

#[derive(Clone)]
struct MixExpert {
    predictor: Box<RatePdfPredictor>,
    log_weight: f64,
    log_prior: f64,
    cum_log_loss: f64,
}

#[derive(Clone)]
struct MixturePredictor {
    kind: MixtureKind,
    alpha: f64,
    decay: f64,
    experts: Vec<MixExpert>,
    neural: NeuralMixCore,
    analyzer: TextContextAnalyzer,
    neural_logps: Vec<f64>,
    neural_bit_modes: Vec<u8>,
    neural_lo: Vec<usize>,
    neural_hi: Vec<usize>,
    neural_pdf_cdf_rows: Vec<Vec<f64>>,
    scratch: Vec<f64>,
    scratch2: Vec<f64>,
    pdf: Vec<f64>,
    valid: bool,
}

impl MixturePredictor {
    fn new(spec: &MixtureSpec) -> Result<Self> {
        if spec.experts.is_empty() {
            bail!("mixture spec must include at least one expert");
        }
        let mut experts = Vec::with_capacity(spec.experts.len());
        for e in &spec.experts {
            experts.push(MixExpert {
                predictor: Box::new(RatePdfPredictor::from_rate_backend(
                    e.backend.clone(),
                    e.max_order,
                )?),
                log_weight: e.log_prior,
                log_prior: e.log_prior,
                cum_log_loss: 0.0,
            });
        }
        let m = logsumexp_expert_weights(&experts);
        for e in &mut experts {
            e.log_weight -= m;
        }

        let mut prior_weights = vec![0.0; experts.len()];
        for (i, e) in experts.iter().enumerate() {
            let p = (e.log_weight).exp().clamp(PDF_MIN, 1.0 - PDF_MIN);
            prior_weights[i] = p;
        }

        let base_lr = spec.alpha.abs().clamp(1e-6, 1.0);
        let effective_lr = (base_lr * 25.0).clamp(1e-6, 1.0);
        let analyzer = TextContextAnalyzer::new();
        let mut neural = NeuralMixCore::new(
            experts.len(),
            &prior_weights,
            effective_lr * 0.5,
            effective_lr,
            1e-5,
        );
        neural.set_context_state(analyzer.state());
        Ok(Self {
            kind: spec.kind,
            alpha: spec.alpha.clamp(1e-12, 1.0 - 1e-12),
            decay: spec.decay.unwrap_or(1.0).clamp(0.0, 1.0),
            experts,
            neural,
            analyzer,
            neural_logps: vec![0.0; spec.experts.len()],
            neural_bit_modes: vec![0; spec.experts.len()],
            neural_lo: vec![0; spec.experts.len()],
            neural_hi: vec![256; spec.experts.len()],
            neural_pdf_cdf_rows: vec![vec![0.0; 257]; spec.experts.len()],
            scratch: Vec::new(),
            scratch2: Vec::new(),
            pdf: vec![0.0; 256],
            valid: false,
        })
    }

    fn ensure_pdf(&mut self) -> Result<&[f64]> {
        if self.valid {
            return Ok(&self.pdf);
        }
        match self.kind {
            MixtureKind::Neural => {
                if self.experts.len() == 1 {
                    self.pdf.fill(0.0);
                    let epdf = self.experts[0].predictor.pdf_next()?;
                    self.pdf.copy_from_slice(epdf);
                    normalize_pdf(&mut self.pdf);
                    self.valid = true;
                    return Ok(&self.pdf);
                }
                self.neural.set_context_state(self.analyzer.state());
                self.neural.evaluate_expert_weights();
                let n = self.experts.len();
                self.scratch.resize(n, 0.0);
                self.scratch.copy_from_slice(self.neural.expert_weights());
                self.pdf.fill(0.0);
                for i in 0..n {
                    let epdf = self.experts[i].predictor.pdf_next()?;
                    let w = self.scratch[i];
                    for (pdf_slot, &p) in self.pdf.iter_mut().zip(epdf.iter()) {
                        *pdf_slot += w * p;
                    }
                }
                normalize_pdf(&mut self.pdf);
                self.valid = true;
                return Ok(&self.pdf);
            }
            _ => {
                self.pdf.fill(0.0);

                let lw_norm = logsumexp_expert_weights(&self.experts);
                for e in &mut self.experts {
                    let w = (e.log_weight - lw_norm).exp();
                    let epdf = e.predictor.pdf_next()?;
                    for (i, p) in epdf.iter().enumerate().take(256) {
                        self.pdf[i] += w * *p;
                    }
                }
            }
        }

        normalize_pdf(&mut self.pdf);
        self.valid = true;
        Ok(&self.pdf)
    }

    fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        for expert in &mut self.experts {
            match &mut *expert.predictor {
                // Direct CTW benefits from pre-reserving, but inside mixtures that extra
                // headroom can dominate peak RSS without a proportional runtime gain.
                RatePdfPredictor::Ctw(_) | RatePdfPredictor::FacCtw(_) => {}
                _ => expert.predictor.begin_stream(total_len)?,
            }
        }
        Ok(())
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
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
                    self.scratch2[i] = e.log_weight + lp;
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
                let log_alpha = self.alpha.ln();
                let log_1m_alpha = (1.0 - self.alpha).ln();
                for (i, e) in self.experts.iter_mut().enumerate() {
                    let switched = logsumexp2(log_1m_alpha + e.log_weight, log_alpha + e.log_prior);
                    self.scratch2[i] = switched + self.scratch[i];
                }
                let log_mix = logsumexp_slice(&self.scratch2[..n]);
                for (i, e) in self.experts.iter_mut().enumerate() {
                    e.log_weight = self.scratch2[i] - log_mix;
                    e.cum_log_loss -= self.scratch[i];
                    e.predictor.update(symbol)?;
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
                    self.neural.set_context_state(self.analyzer.state());
                    self.valid = false;
                    return Ok(());
                }
                let n = self.experts.len();
                self.neural.set_context_state(self.analyzer.state());
                self.neural_logps.resize(n, 0.0);
                for i in 0..n {
                    let p = self.experts[i].predictor.pdf_next()?[y].max(PDF_MIN);
                    let lp = p.ln();
                    self.neural_logps[i] = lp;
                    self.experts[i].cum_log_loss -= lp;
                }
                self.neural.evaluate_symbol(&self.neural_logps, PDF_MIN);
                self.neural
                    .update_weights_symbol(&self.neural_logps, PDF_MIN);
                for e in &mut self.experts {
                    e.predictor.update(symbol)?;
                }
                self.analyzer.update(symbol);
                self.neural.set_context_state(self.analyzer.state());
            }
        }

        self.valid = false;
        Ok(())
    }

    fn finish_stream(&mut self) -> Result<()> {
        for expert in &mut self.experts {
            expert.predictor.finish_stream()?;
        }
        Ok(())
    }

    #[inline]
    fn can_fast_ac_bitwise(&self) -> bool {
        self.experts.iter().any(|e| {
            if let RatePdfPredictor::Ctw(ctw) = &*e.predictor {
                ctw.can_fast_ac_bitwise()
            } else {
                false
            }
        })
    }

    fn ac_step_bitwise<F>(&mut self, mut choose_bit: F) -> Result<u8>
    where
        F: FnMut(usize, f64) -> Result<u8>,
    {
        let n = self.experts.len();
        self.scratch.resize(n, 0.0);
        match self.kind {
            MixtureKind::Neural if n > 1 => {
                self.neural.set_context_state(self.analyzer.state());
                self.neural.evaluate_expert_weights();
                self.scratch.copy_from_slice(self.neural.expert_weights());
            }
            _ => {
                let lw_norm = logsumexp_expert_weights(&self.experts);
                for (i, expert) in self.experts.iter().enumerate() {
                    self.scratch[i] = (expert.log_weight - lw_norm).exp();
                }
            }
        }
        self.scratch2.resize(n, 1.0);
        self.scratch2.fill(1.0);
        self.neural_logps.resize(n, 0.0);
        self.neural_bit_modes.resize(n, 0);
        self.neural_lo.resize(n, 0);
        self.neural_hi.resize(n, 256);
        if self.neural_pdf_cdf_rows.len() < n {
            self.neural_pdf_cdf_rows.resize_with(n, || vec![0.0; 257]);
        }

        for i in 0..n {
            self.neural_bit_modes[i] = 1;
            self.neural_lo[i] = 0;
            self.neural_hi[i] = 256;

            let mut handled_ctw = false;
            if let RatePdfPredictor::Ctw(ctw) = &mut *self.experts[i].predictor
                && ctw.can_fast_ac_bitwise()
            {
                self.neural_bit_modes[i] = 0;
                handled_ctw = true;
            }
            if handled_ctw {
                continue;
            }

            if self.experts[i]
                .predictor
                .prepare_cached_cdf_fast_bitwise()?
            {
                self.neural_bit_modes[i] = 2;
                continue;
            }

            let pdf = self.experts[i].predictor.pdf_next()?;
            let row = &mut self.neural_pdf_cdf_rows[i];
            if row.len() != 257 {
                row.resize(257, 0.0);
            }
            row[0] = 0.0;
            for b in 0..256usize {
                row[b + 1] = row[b] + pdf[b].max(PDF_MIN);
            }
            if !row[256].is_finite() || row[256] <= 0.0 {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = (j as f64) / 256.0;
                }
            }
        }

        let mut symbol = 0u8;
        for bit_idx in 0..8usize {
            let mut denom = 0.0;
            let mut numer1 = 0.0;

            for i in 0..n {
                let p1 = if self.neural_bit_modes[i] == 0 {
                    match &mut *self.experts[i].predictor {
                        RatePdfPredictor::Ctw(ctw) => ctw.bit_prob_one_msb(bit_idx),
                        _ => 0.5,
                    }
                } else if self.neural_bit_modes[i] == 2 {
                    self.experts[i]
                        .predictor
                        .cached_cdf_bit_prob_one_msb(self.neural_lo[i], self.neural_hi[i])
                        .unwrap_or(0.5)
                } else {
                    let lo = self.neural_lo[i];
                    let hi = self.neural_hi[i];
                    let mid = (lo + hi) >> 1;
                    let row = &self.neural_pdf_cdf_rows[i];
                    let total = (row[hi] - row[lo]).max(PDF_MIN);
                    let one = (row[hi] - row[mid]).max(0.0);
                    (one / total).clamp(PDF_MIN, 1.0 - PDF_MIN)
                };
                self.neural_logps[i] = p1;
                let wp = self.scratch[i] * self.scratch2[i];
                denom += wp;
                numer1 += wp * p1;
            }

            let p1_mix = if denom.is_finite() && denom > 0.0 {
                (numer1 / denom).clamp(PDF_MIN, 1.0 - PDF_MIN)
            } else {
                0.5
            };
            let bit = choose_bit(bit_idx, p1_mix)? & 1;
            symbol |= bit << (7 - bit_idx);

            for i in 0..n {
                let p1 = self.neural_logps[i];
                let pb = if bit == 1 { p1 } else { 1.0 - p1 };
                self.scratch2[i] = (self.scratch2[i] * pb).max(PDF_MIN);

                if self.neural_bit_modes[i] == 0 {
                    if let RatePdfPredictor::Ctw(ctw) = &mut *self.experts[i].predictor {
                        ctw.update_bit_msb(bit_idx, bit == 1);
                    }
                } else {
                    let lo = self.neural_lo[i];
                    let hi = self.neural_hi[i];
                    let mid = (lo + hi) >> 1;
                    if bit == 1 {
                        self.neural_lo[i] = mid;
                        self.neural_hi[i] = hi;
                    } else {
                        self.neural_lo[i] = lo;
                        self.neural_hi[i] = mid;
                    }
                }
            }
        }

        for i in 0..n {
            let lp = self.scratch2[i].max(PDF_MIN).ln();
            self.neural_logps[i] = lp;
            self.experts[i].cum_log_loss -= lp;
            if self.neural_bit_modes[i] != 0 {
                self.experts[i].predictor.update(symbol)?;
            }
        }

        match self.kind {
            MixtureKind::Bayes => {
                for i in 0..n {
                    self.scratch[i] = self.experts[i].log_weight + self.neural_logps[i];
                }
                let log_mix = logsumexp_slice(&self.scratch[..n]);
                for i in 0..n {
                    self.experts[i].log_weight += self.neural_logps[i] - log_mix;
                }
            }
            MixtureKind::FadingBayes => {
                for i in 0..n {
                    self.scratch[i] =
                        self.decay * self.experts[i].log_weight + self.neural_logps[i];
                }
                let log_mix = logsumexp_slice(&self.scratch[..n]);
                for i in 0..n {
                    self.experts[i].log_weight = self.scratch[i] - log_mix;
                }
            }
            MixtureKind::Switching => {
                let log_alpha = self.alpha.ln();
                let log_1m_alpha = (1.0 - self.alpha).ln();
                for i in 0..n {
                    let expert = &self.experts[i];
                    let switched = logsumexp2(
                        log_1m_alpha + expert.log_weight,
                        log_alpha + expert.log_prior,
                    );
                    self.scratch[i] = switched + self.neural_logps[i];
                }
                let log_mix = logsumexp_slice(&self.scratch[..n]);
                for i in 0..n {
                    self.experts[i].log_weight = self.scratch[i] - log_mix;
                }
            }
            MixtureKind::Mdl => {}
            MixtureKind::Neural => {
                if n > 1 {
                    self.neural.set_context_state(self.analyzer.state());
                    self.neural.evaluate_symbol(&self.neural_logps, PDF_MIN);
                    self.neural
                        .update_weights_symbol(&self.neural_logps, PDF_MIN);
                }
                self.analyzer.update(symbol);
                self.neural.set_context_state(self.analyzer.state());
            }
        }
        self.valid = false;
        Ok(symbol)
    }
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum RatePdfPredictor {
    Rosa(RosaPredictor),
    Match {
        model: MatchModel,
    },
    SparseMatch {
        model: SparseMatchModel,
    },
    Ppmd {
        model: PpmdModel,
    },
    Ctw(CtwPredictor),
    FacCtw(CtwPredictor),
    #[cfg(feature = "backend-mamba")]
    Mamba(MambaPredictor),
    #[cfg(feature = "backend-rwkv")]
    Rwkv(RwkvPredictor),
    Zpaq(ZpaqPredictor),
    Mixture(MixturePredictor),
    Particle(crate::particle::ParticleRuntime),
    Calibrated {
        base: Box<RatePdfPredictor>,
        core: CalibratorCore,
        pdf: Vec<f64>,
        valid: bool,
    },
}

impl RatePdfPredictor {
    fn from_rate_backend(backend: RateBackend, max_order: i64) -> Result<Self> {
        match backend {
            RateBackend::RosaPlus => Ok(Self::Rosa(RosaPredictor::new(max_order))),
            RateBackend::Match {
                hash_bits,
                min_len,
                max_len,
                base_mix,
                confidence_scale,
            } => Ok(Self::Match {
                model: MatchModel::new_contiguous(
                    hash_bits,
                    min_len,
                    max_len,
                    base_mix,
                    confidence_scale,
                ),
            }),
            RateBackend::SparseMatch {
                hash_bits,
                min_len,
                max_len,
                gap_min,
                gap_max,
                base_mix,
                confidence_scale,
            } => Ok(Self::SparseMatch {
                model: SparseMatchModel::new(
                    hash_bits,
                    min_len,
                    max_len,
                    gap_min,
                    gap_max,
                    base_mix,
                    confidence_scale,
                ),
            }),
            RateBackend::Ppmd { order, memory_mb } => Ok(Self::Ppmd {
                model: PpmdModel::new(order, memory_mb),
            }),
            RateBackend::Ctw { depth } => Ok(Self::Ctw(CtwPredictor::new_ctw(depth))),
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits: _,
                encoding_bits,
            } => {
                let bits = encoding_bits.clamp(1, 8);
                Ok(Self::FacCtw(CtwPredictor::new_fac(base_depth, bits)))
            }
            #[cfg(feature = "backend-mamba")]
            RateBackend::Mamba { model } => Ok(Self::Mamba(MambaPredictor::from_model(model))),
            #[cfg(feature = "backend-mamba")]
            RateBackend::MambaMethod { method } => {
                Ok(Self::Mamba(MambaPredictor::from_method(&method)?))
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7 { model } => Ok(Self::Rwkv(RwkvPredictor::from_model(model))),
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7Method { method } => {
                Ok(Self::Rwkv(RwkvPredictor::from_method(&method)?))
            }
            RateBackend::Zpaq { method } => Ok(Self::Zpaq(ZpaqPredictor::new(method))),
            RateBackend::Mixture { spec } => {
                Ok(Self::Mixture(MixturePredictor::new(spec.as_ref())?))
            }
            RateBackend::Particle { spec } => Ok(Self::Particle(
                crate::particle::ParticleRuntime::new(spec.as_ref()),
            )),
            RateBackend::Calibrated { spec } => Ok(Self::Calibrated {
                base: Box::new(Self::from_rate_backend(spec.base.clone(), max_order)?),
                core: build_calibrator(spec.as_ref()),
                pdf: vec![1.0 / 256.0; 256],
                valid: false,
            }),
        }
    }

    fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        self.finish_stream()?;
        match self {
            Self::Rosa(m) => {
                m.begin_stream(total_len);
                Ok(())
            }
            Self::Match { .. }
            | Self::SparseMatch { .. }
            | Self::Ppmd { .. }
            | Self::Zpaq(_)
            | Self::Particle(_) => Ok(()),
            Self::Ctw(m) | Self::FacCtw(m) => {
                m.tree.reserve_for_symbols(total_len);
                Ok(())
            }
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => m.begin_stream(total_len),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => m.begin_stream(total_len),
            Self::Mixture(m) => m.begin_stream(total_len),
            Self::Calibrated { base, .. } => base.begin_stream(total_len),
        }
    }

    fn finish_stream(&mut self) -> Result<()> {
        match self {
            Self::Rosa(_)
            | Self::Match { .. }
            | Self::SparseMatch { .. }
            | Self::Ppmd { .. }
            | Self::Ctw(_)
            | Self::FacCtw(_)
            | Self::Zpaq(_)
            | Self::Particle(_) => Ok(()),
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(_) => Ok(()),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => m.finish_stream(),
            Self::Mixture(m) => m.finish_stream(),
            Self::Calibrated { base, .. } => base.finish_stream(),
        }
    }

    fn pdf_next(&mut self) -> Result<&[f64]> {
        match self {
            Self::Rosa(m) => Ok(m.pdf_next()),
            Self::Match { model } => Ok(model.pdf()),
            Self::Ctw(m) => Ok(m.pdf_next()),
            Self::FacCtw(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => Ok(m.pdf_next()),
            Self::Zpaq(m) => Ok(m.pdf_next()),
            Self::Mixture(m) => m.ensure_pdf(),
            Self::Particle(m) => Ok(m.pdf_next()),
            Self::SparseMatch { model } => Ok(model.pdf()),
            Self::Ppmd { model } => Ok(model.pdf()),
            Self::Calibrated {
                base,
                core,
                pdf,
                valid,
            } => {
                if !*valid {
                    let base_pdf = base.pdf_next()?;
                    core.apply_pdf(base_pdf, pdf);
                    normalize_pdf(pdf);
                    *valid = true;
                }
                Ok(pdf)
            }
        }
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        match self {
            Self::Rosa(m) => {
                m.update(symbol);
                Ok(())
            }
            Self::Match { model } => {
                model.update(symbol);
                Ok(())
            }
            Self::SparseMatch { model } => {
                model.update(symbol);
                Ok(())
            }
            Self::Ppmd { model } => {
                model.update(symbol);
                Ok(())
            }
            Self::Ctw(m) => {
                m.update(symbol);
                Ok(())
            }
            Self::FacCtw(m) => {
                m.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => m.update(symbol),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => m.update(symbol),
            Self::Zpaq(m) => {
                m.update(symbol);
                Ok(())
            }
            Self::Mixture(m) => m.update(symbol),
            Self::Particle(m) => {
                m.step(symbol);
                Ok(())
            }
            Self::Calibrated {
                base,
                core,
                pdf,
                valid,
            } => {
                if !*valid {
                    let base_pdf = base.pdf_next()?;
                    core.apply_pdf(base_pdf, pdf);
                    normalize_pdf(pdf);
                }
                core.update(symbol, pdf);
                base.update(symbol)?;
                *valid = false;
                Ok(())
            }
        }
    }

    fn prepare_cached_cdf_fast_bitwise(&mut self) -> Result<bool> {
        match self {
            Self::Rosa(m) => {
                let _ = m.cdf_next();
                Ok(true)
            }
            Self::Match { model } => {
                let _ = model.cdf();
                Ok(true)
            }
            Self::SparseMatch { model } => {
                let _ = model.cdf();
                Ok(true)
            }
            Self::Ppmd { model } => {
                let _ = model.cdf();
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn cached_cdf_bit_prob_one_msb(&mut self, lo: usize, hi: usize) -> Option<f64> {
        match self {
            Self::Rosa(m) => Some(cdf_bit_prob_one_msb(&m.cdf, lo, hi)),
            Self::Match { model } => Some(cdf_bit_prob_one_msb(model.cdf(), lo, hi)),
            Self::SparseMatch { model } => Some(cdf_bit_prob_one_msb(model.cdf(), lo, hi)),
            Self::Ppmd { model } => Some(cdf_bit_prob_one_msb(model.cdf(), lo, hi)),
            _ => None,
        }
    }

    #[inline]
    fn can_fast_ac_bitwise(&self) -> bool {
        match self {
            Self::Ctw(m) => m.can_fast_ac_bitwise(),
            Self::Mixture(m) => m.can_fast_ac_bitwise(),
            _ => false,
        }
    }

    fn ac_step_fast_bitwise<F>(&mut self, choose_bit: F) -> Result<u8>
    where
        F: FnMut(usize, f64) -> Result<u8>,
    {
        match self {
            Self::Ctw(m) => ctw_ac_step_bitwise(m, choose_bit),
            Self::Mixture(m) => m.ac_step_bitwise(choose_bit),
            _ => unreachable!("fast bitwise path requested for unsupported predictor"),
        }
    }
}

fn ctw_ac_step_bitwise<F>(ctw: &mut CtwPredictor, mut choose_bit: F) -> Result<u8>
where
    F: FnMut(usize, f64) -> Result<u8>,
{
    debug_assert!(ctw.can_fast_ac_bitwise());
    let mut symbol = 0u8;
    for bit_idx in 0..8usize {
        let p1 = ctw.bit_prob_one_msb(bit_idx);
        let bit = choose_bit(bit_idx, p1)? & 1;
        symbol |= bit << (7 - bit_idx);
        ctw.update_bit_msb(bit_idx, bit == 1);
    }
    Ok(symbol)
}

#[inline]
fn binary_split_from_prob_one(p1: f64) -> u32 {
    let p1 = p1.clamp(PDF_MIN, 1.0 - PDF_MIN);
    let p0 = 1.0 - p1;
    let mut split = (p0 * (CDF_TOTAL as f64)) as u32;
    if split == 0 {
        split = 1;
    } else if split >= CDF_TOTAL {
        split = CDF_TOTAL - 1;
    }
    split
}

fn encode_payload_ac(data: &[u8], predictor: &mut RatePdfPredictor) -> Result<Vec<u8>> {
    predictor.begin_stream(data.len())?;
    if predictor.can_fast_ac_bitwise() {
        let mut out = Vec::new();
        {
            let mut enc = ArithmeticEncoder::new(&mut out);
            for &symbol in data {
                predictor.ac_step_fast_bitwise(|bit_idx, p1_mix| {
                    let bit = (symbol >> (7 - bit_idx)) & 1;
                    let split = binary_split_from_prob_one(p1_mix);
                    if bit == 0 {
                        enc.encode_counts(0, split as u64, CDF_TOTAL as u64)?;
                    } else {
                        enc.encode_counts(split as u64, CDF_TOTAL as u64, CDF_TOTAL as u64)?;
                    }
                    Ok(bit)
                })?;
            }
            let _ = enc.finish()?;
        }
        predictor.finish_stream()?;
        return Ok(out);
    }

    let mut out = Vec::new();
    {
        let mut enc = ArithmeticEncoder::new(&mut out);
        let mut cdf = vec![0u32; 257];
        for &b in data {
            let pdf = predictor.pdf_next()?;
            crate::coders::quantize_pdf_to_integer_cdf_dense_positive_with_buffer(
                pdf, CDF_TOTAL, &mut cdf,
            );
            let sym = b as usize;
            enc.encode_counts(cdf[sym] as u64, cdf[sym + 1] as u64, CDF_TOTAL as u64)?;
            predictor.update(b)?;
        }
        let _ = enc.finish()?;
    }
    predictor.finish_stream()?;
    Ok(out)
}

fn decode_payload_ac(
    payload: &[u8],
    out_len: usize,
    predictor: &mut RatePdfPredictor,
) -> Result<Vec<u8>> {
    predictor.begin_stream(out_len)?;
    if predictor.can_fast_ac_bitwise() {
        let mut dec = ArithmeticDecoder::new(payload)?;
        let mut out = Vec::with_capacity(out_len);
        for _ in 0..out_len {
            let symbol = predictor.ac_step_fast_bitwise(|_, p1_mix| {
                let split = binary_split_from_prob_one(p1_mix);
                let cdf = [0u32, split, CDF_TOTAL];
                Ok(dec.decode_symbol_counts(&cdf, CDF_TOTAL)? as u8)
            })?;
            out.push(symbol);
        }
        predictor.finish_stream()?;
        return Ok(out);
    }

    let mut dec = ArithmeticDecoder::new(payload)?;
    let mut out = Vec::with_capacity(out_len);
    let mut cdf = vec![0u32; 257];
    for _ in 0..out_len {
        let pdf = predictor.pdf_next()?;
        crate::coders::quantize_pdf_to_integer_cdf_dense_positive_with_buffer(
            pdf, CDF_TOTAL, &mut cdf,
        );
        let sym = dec.decode_symbol_counts(&cdf, CDF_TOTAL)? as u8;
        out.push(sym);
        predictor.update(sym)?;
    }
    predictor.finish_stream()?;
    Ok(out)
}

fn encode_payload_rans(data: &[u8], predictor: &mut RatePdfPredictor) -> Result<Vec<u8>> {
    predictor.begin_stream(data.len())?;
    let mut encoder = BlockedRansEncoder::new();
    let mut cdf = vec![0u32; 257];
    let mut freq = vec![0i64; 256];

    for &b in data {
        let pdf = predictor.pdf_next()?;
        quantize_pdf_to_rans_cdf_with_buffer(pdf, &mut cdf, &mut freq);
        let s = b as usize;
        encoder.encode(Cdf::new(cdf[s], cdf[s + 1], ANS_TOTAL));
        predictor.update(b)?;
    }

    let blocks = encoder.finish();
    let mut out = Vec::new();
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for block in blocks {
        out.extend_from_slice(&(block.len() as u32).to_le_bytes());
        out.extend_from_slice(&block);
    }
    predictor.finish_stream()?;
    Ok(out)
}

fn decode_payload_rans(
    payload: &[u8],
    out_len: usize,
    predictor: &mut RatePdfPredictor,
) -> Result<Vec<u8>> {
    predictor.begin_stream(out_len)?;
    if payload.len() < 4 {
        bail!("rANS payload too short");
    }
    let block_count = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut pos = 4usize;
    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        if pos + 4 > payload.len() {
            bail!("truncated rANS block header");
        }
        let len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;
        if pos + len > payload.len() {
            bail!("truncated rANS block data");
        }
        blocks.push(&payload[pos..pos + len]);
        pos += len;
    }

    let mut dec = BlockedRansDecoder::new(blocks, out_len)?;
    let mut out = Vec::with_capacity(out_len);
    let mut cdf = vec![0u32; 257];
    let mut freq = vec![0i64; 256];

    for _ in 0..out_len {
        let pdf = predictor.pdf_next()?;
        quantize_pdf_to_rans_cdf_with_buffer(pdf, &mut cdf, &mut freq);
        let sym = dec.decode(&cdf)? as u8;
        out.push(sym);
        predictor.update(sym)?;
    }
    predictor.finish_stream()?;
    Ok(out)
}

/// Compress bytes using a predictive rate backend and entropy coder.
///
/// When `framing` is [`FramingMode::Framed`], output includes a compact header
/// with payload metadata and CRC for safer transport/storage.
pub fn compress_rate_bytes(
    data: &[u8],
    rate_backend: &RateBackend,
    max_order: i64,
    coder: CoderType,
    framing: FramingMode,
) -> Result<Vec<u8>> {
    let mut predictor = RatePdfPredictor::from_rate_backend(rate_backend.clone(), max_order)?;
    let payload = match coder {
        CoderType::AC => encode_payload_ac(data, &mut predictor)?,
        CoderType::RANS => encode_payload_rans(data, &mut predictor)?,
    };

    if framing == FramingMode::Raw {
        return Ok(payload);
    }

    let mut out = Vec::with_capacity(FramedHeader::SIZE + payload.len());
    let hdr = FramedHeader::new(coder, data.len() as u64, crc32(data));
    hdr.write(&mut out);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Return compressed size (in bytes) for `data` using rate coding.
pub fn compress_rate_size(
    data: &[u8],
    rate_backend: &RateBackend,
    max_order: i64,
    coder: CoderType,
    framing: FramingMode,
) -> Result<u64> {
    let encoded = compress_rate_bytes(data, rate_backend, max_order, coder, framing)?;
    Ok(encoded.len() as u64)
}

/// Return compressed size (in bytes) for concatenated slices under one stream.
pub fn compress_rate_size_chain(
    parts: &[&[u8]],
    rate_backend: &RateBackend,
    max_order: i64,
    coder: CoderType,
    framing: FramingMode,
) -> Result<u64> {
    let total = parts.iter().map(|p| p.len()).sum();
    let mut data = Vec::with_capacity(total);
    for p in parts {
        data.extend_from_slice(p);
    }
    compress_rate_size(&data, rate_backend, max_order, coder, framing)
}

/// Decompress bytes produced by [`compress_rate_bytes`].
pub fn decompress_rate_bytes(
    input: &[u8],
    rate_backend: &RateBackend,
    max_order: i64,
    _coder: CoderType,
    framing: FramingMode,
) -> Result<Vec<u8>> {
    let (payload, coder, out_len, expected_crc) = if framing == FramingMode::Framed {
        let hdr = FramedHeader::read(input)?;
        (
            &input[FramedHeader::SIZE..],
            hdr.coder_type(),
            hdr.original_len as usize,
            Some(hdr.crc32),
        )
    } else {
        bail!("raw payload decompression requires explicit output length and is not supported");
    };

    let _ = coder;
    let mut predictor = RatePdfPredictor::from_rate_backend(rate_backend.clone(), max_order)?;
    let decoded = match coder {
        CoderType::AC => decode_payload_ac(payload, out_len, &mut predictor)?,
        CoderType::RANS => decode_payload_rans(payload, out_len, &mut predictor)?,
    };

    if let Some(crc) = expected_crc {
        let got = crc32(&decoded);
        if got != crc {
            bail!("CRC32 mismatch: expected 0x{crc:08X}, got 0x{got:08X}");
        }
    }

    Ok(decoded)
}

fn normalize_pdf(pdf: &mut [f64]) {
    let mut sum = 0.0;
    for p in pdf.iter_mut() {
        *p = if p.is_finite() {
            (*p).max(PDF_MIN)
        } else {
            PDF_MIN
        };
        sum += *p;
    }
    if !(sum.is_finite()) || sum <= 0.0 {
        let u = 1.0 / (pdf.len() as f64);
        for p in pdf.iter_mut() {
            *p = u;
        }
        return;
    }
    let inv = 1.0 / sum;
    for p in pdf.iter_mut() {
        *p *= inv;
    }
}

#[inline]
fn uniform_cdf_row() -> [f64; 257] {
    let mut cdf = [0.0; 257];
    let inv = 1.0 / 256.0;
    for (i, slot) in cdf.iter_mut().enumerate() {
        *slot = (i as f64) * inv;
    }
    cdf
}

#[inline]
fn build_cdf_row_from_pdf_slice(pdf: &[f64], cdf: &mut [f64; 257]) {
    cdf[0] = 0.0;
    let mut acc = 0.0;
    for i in 0..256 {
        acc += pdf[i];
        cdf[i + 1] = acc;
    }
}

fn normalize_pdf_vec_and_maybe_build_cdf(pdf: &mut [f64], mut cdf: Option<&mut [f64; 257]>) {
    let mut sum = 0.0;
    for p in pdf.iter_mut() {
        *p = if p.is_finite() {
            (*p).max(PDF_MIN)
        } else {
            PDF_MIN
        };
        sum += *p;
    }
    if !(sum.is_finite()) || sum <= 0.0 {
        let u = 1.0 / (pdf.len() as f64);
        pdf.fill(u);
        if let Some(cdf) = cdf.as_deref_mut() {
            *cdf = uniform_cdf_row();
        }
        return;
    }
    let inv = 1.0 / sum;
    if let Some(cdf) = cdf.as_deref_mut() {
        cdf[0] = 0.0;
        let mut acc = 0.0;
        for i in 0..256 {
            pdf[i] *= inv;
            acc += pdf[i];
            cdf[i + 1] = acc;
        }
    } else {
        for p in pdf.iter_mut() {
            *p *= inv;
        }
    }
}

#[inline]
fn cdf_bit_prob_one_msb(cdf: &[f64; 257], lo: usize, hi: usize) -> f64 {
    let mid = (lo + hi) >> 1;
    let total = (cdf[hi] - cdf[lo]).max(PDF_MIN);
    let one = (cdf[hi] - cdf[mid]).max(0.0);
    (one / total).clamp(PDF_MIN, 1.0 - PDF_MIN)
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

fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = if a > b { a } else { b };
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[allow(dead_code)]
fn _zpaq_marker(_: &ZpaqRateModel) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn assert_pdf_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let delta = (a - b).abs();
            assert!(
                delta <= tol,
                "pdf mismatch at symbol {idx}: lhs={a} rhs={b} delta={delta}"
            );
        }
    }

    fn brute_force_pdf(predictor: &mut CtwPredictor) -> Vec<f64> {
        let bits = predictor.bits_per_symbol.clamp(1, 8);
        let mut out = vec![0.0; 256];

        if bits == 8 {
            for (sym, slot) in out.iter_mut().enumerate().take(256usize) {
                *slot = predictor.log_prob_symbol_bruteforce(sym as u8).exp();
            }
        } else {
            let patterns = 1usize << bits;
            let aliases = 1usize << (8 - bits);
            let mut pat_prob = vec![0.0; patterns];
            for (pat, value) in pat_prob.iter_mut().enumerate() {
                let symbol = if predictor.msb_first {
                    (pat as u8) << (8 - bits)
                } else {
                    pat as u8
                };
                *value = predictor.log_prob_symbol_bruteforce(symbol).exp();
            }
            for (byte, slot) in out.iter_mut().enumerate().take(256usize) {
                let pat = if predictor.msb_first {
                    byte >> (8 - bits)
                } else {
                    byte & (patterns - 1)
                };
                *slot = pat_prob[pat] / (aliases as f64);
            }
        }

        CtwPredictor::normalize_pdf(&mut out);
        out
    }

    #[test]
    fn ctw_pdf_fast_matches_bruteforce() {
        let mut predictor = CtwPredictor::new_ctw(6);
        for &b in b"ctw fast-path regression corpus 1234567890" {
            predictor.update(b);
        }

        let fast = predictor.pdf_next().to_vec();
        predictor.valid = false;
        let brute = brute_force_pdf(&mut predictor);

        for i in 0..256usize {
            let delta = (fast[i] - brute[i]).abs();
            assert!(
                delta < 1e-12,
                "symbol={i} fast={} brute={} delta={delta}",
                fast[i],
                brute[i]
            );
        }
    }

    #[test]
    fn fac_pdf_fast_matches_bruteforce_subbyte() {
        let mut predictor = CtwPredictor::new_fac(5, 5);
        for &b in b"fac ctw subbyte regression corpus abcdefghijklmnopqrstuvwxyz" {
            predictor.update(b);
        }

        let fast = predictor.pdf_next().to_vec();
        predictor.valid = false;
        let brute = brute_force_pdf(&mut predictor);

        for i in 0..256usize {
            let delta = (fast[i] - brute[i]).abs();
            assert!(
                delta < 1e-12,
                "symbol={i} fast={} brute={} delta={delta}",
                fast[i],
                brute[i]
            );
        }
    }

    #[test]
    fn roundtrip_rate_ac_ctw() {
        let data = b"ctw backend roundtrip payload";
        let backend = RateBackend::Ctw { depth: 8 };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_match_family_and_ppmd() {
        let data = b"repeat repeat repeat sparse sparse repeat payload";
        for backend in [
            RateBackend::Match {
                hash_bits: 20,
                min_len: 4,
                max_len: 255,
                base_mix: 0.02,
                confidence_scale: 1.0,
            },
            RateBackend::SparseMatch {
                hash_bits: 19,
                min_len: 3,
                max_len: 64,
                gap_min: 1,
                gap_max: 2,
                base_mix: 0.05,
                confidence_scale: 1.0,
            },
            RateBackend::Ppmd {
                order: 8,
                memory_mb: 8,
            },
        ] {
            let enc = compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed)
                .unwrap();
            let dec = decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed)
                .unwrap();
            assert_eq!(dec, data);
        }
    }

    #[test]
    fn roundtrip_rate_ac_ppmd_high_order_text_payload() {
        let seed = include_bytes!("../../README.md");
        let mut data = Vec::with_capacity(4096);
        while data.len() < 4096 {
            data.extend_from_slice(seed);
        }
        data.truncate(4096);

        let backend = RateBackend::Ppmd {
            order: 12,
            memory_mb: 256,
        };
        let enc = compress_rate_bytes(&data, &backend, -1, CoderType::AC, FramingMode::Framed)
            .expect("ppmd high-order compression");
        let dec = decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed)
            .expect("ppmd high-order decompression");
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_calibrated_backend() {
        let data = b"calibration wrapper payload calibration wrapper payload";
        let backend = RateBackend::Calibrated {
            spec: Arc::new(crate::CalibratedSpec {
                base: RateBackend::Ctw { depth: 8 },
                context: crate::CalibrationContextKind::Text,
                bins: 33,
                learning_rate: 0.02,
                bias_clip: 4.0,
            }),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_single_expert_ctw_neural_mixture() {
        let data = b"single expert neural ctw fast path payload";
        let spec = MixtureSpec::new(
            MixtureKind::Neural,
            vec![crate::MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Ctw { depth: 8 },
            }],
        )
        .with_alpha(0.03);
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_single_expert_ctw_bayes_mixture() {
        let data = b"single expert bayes ctw fast path payload";
        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![crate::MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                max_order: -1,
                backend: RateBackend::Ctw { depth: 8 },
            }],
        )
        .with_alpha(0.03);
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_rans_recursive_mixture() {
        let data = b"recursive mixture payload";
        let nested = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Ctw { depth: 6 },
                },
                crate::MixtureExpertSpec {
                    name: Some("fac".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::FacCtw {
                        base_depth: 6,
                        num_percept_bits: 8,
                        encoding_bits: 8,
                    },
                },
            ],
        );
        let root = MixtureSpec::new(
            MixtureKind::Switching,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("nested".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Mixture {
                        spec: Arc::new(nested),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("zpaq".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Zpaq {
                        method: "1".to_string(),
                    },
                },
            ],
        )
        .with_alpha(0.05);

        let backend = RateBackend::Mixture {
            spec: Arc::new(root),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::RANS, FramingMode::Framed).unwrap();
        let dec = decompress_rate_bytes(&enc, &backend, -1, CoderType::RANS, FramingMode::Framed)
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_recursive_neural_mixture() {
        let data = b"neural recursive mixture payload for ac coder";
        let inner = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Ctw { depth: 6 },
                },
                crate::MixtureExpertSpec {
                    name: Some("fac".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::FacCtw {
                        base_depth: 6,
                        num_percept_bits: 8,
                        encoding_bits: 8,
                    },
                },
            ],
        );
        let root = MixtureSpec::new(
            MixtureKind::Neural,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("nested".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Mixture {
                        spec: Arc::new(inner),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("zpaq".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Zpaq {
                        method: "1".to_string(),
                    },
                },
            ],
        )
        .with_alpha(0.03);

        let backend = RateBackend::Mixture {
            spec: Arc::new(root),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn neural_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(
            MixtureKind::Neural,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Ctw { depth: 7 },
                },
                crate::MixtureExpertSpec {
                    name: Some("fac".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::FacCtw {
                        base_depth: 7,
                        num_percept_bits: 8,
                        encoding_bits: 8,
                    },
                },
            ],
        )
        .with_alpha(0.03);

        let backend = RateBackend::Mixture {
            spec: Arc::new(spec.clone()),
        };
        let mut predictor = RatePdfPredictor::from_rate_backend(backend, -1).unwrap();
        let experts = spec.build_experts();
        let mut runtime = crate::mixture::build_mixture_runtime(&spec, &experts).unwrap();

        let data = b"neural alignment check sequence";
        for &b in data {
            let pdf = predictor.pdf_next().unwrap();
            let p_comp = pdf[b as usize];
            let p_runtime = runtime.peek_log_prob(b).exp();
            assert!(
                (p_comp - p_runtime).abs() < 1e-8,
                "p_comp={p_comp} p_runtime={p_runtime} symbol={b}"
            );
            predictor.update(b).unwrap();
            runtime.step(b);
        }
    }

    fn assert_cached_cdf_fast_bitwise_matches_pdf_rows(mut predictor: RatePdfPredictor) {
        let data = b"cached cdf parity check payload";
        for &symbol in data {
            let pdf = predictor.pdf_next().unwrap().to_vec();
            assert!(predictor.prepare_cached_cdf_fast_bitwise().unwrap());

            let mut row = [0.0; 257];
            row[0] = 0.0;
            for i in 0..256 {
                row[i + 1] = row[i] + pdf[i].max(PDF_MIN);
            }

            let mut stack = vec![(0usize, 256usize)];
            while let Some((lo, hi)) = stack.pop() {
                if hi - lo <= 1 {
                    continue;
                }
                let expected = cdf_bit_prob_one_msb(&row, lo, hi);
                let got = predictor
                    .cached_cdf_bit_prob_one_msb(lo, hi)
                    .expect("cached cdf branch probability");
                let diff = (expected - got).abs();
                assert!(
                    diff <= 1e-12,
                    "lo={lo} hi={hi} expected={expected} got={got} diff={diff}"
                );
                let mid = (lo + hi) >> 1;
                stack.push((lo, mid));
                stack.push((mid, hi));
            }

            predictor.update(symbol).unwrap();
        }
    }

    #[test]
    fn cached_cdf_fast_bitwise_matches_pdf_rows_for_specialized_predictors() {
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(RateBackend::RosaPlus, -1).unwrap(),
        );
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(
                RateBackend::Ppmd {
                    order: 6,
                    memory_mb: 8,
                },
                -1,
            )
            .unwrap(),
        );
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(
                RateBackend::Match {
                    hash_bits: 20,
                    min_len: 4,
                    max_len: 255,
                    base_mix: 0.02,
                    confidence_scale: 1.0,
                },
                -1,
            )
            .unwrap(),
        );
    }

    #[test]
    fn raw_size_not_larger_than_framed_size() {
        let data = b"raw/framed size check payload";
        let backend = RateBackend::RosaPlus;
        let raw = compress_rate_size(data, &backend, 8, CoderType::AC, FramingMode::Raw).unwrap();
        let framed =
            compress_rate_size(data, &backend, 8, CoderType::AC, FramingMode::Framed).unwrap();
        assert!(framed >= raw);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn roundtrip_rate_rwkv_method_cfg() {
        let data = b"rwkv cfg method backend";
        let backend = RateBackend::Rwkv7Method {
            method: "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer".to_string(),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_rate_predictor_preserves_backend_pdf_exactly() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
        let mut predictor = RwkvPredictor::from_method(method).expect("rwkv predictor");
        let mut backend = rwkvzip::Compressor::new_from_method(method).expect("rwkv backend");
        let mut direct = vec![0.0; backend.vocab_size()];

        let predicted = predictor.pdf_next().to_vec();
        backend.forward_to_pdf(0, &mut direct);
        assert_pdf_close(&predicted, &direct, 1e-18);

        predictor.update(b'x').expect("predictor update");
        backend
            .online_update_from_pdf(b'x', &direct)
            .expect("backend update");
        backend.forward_to_pdf(u32::from(b'x'), &mut direct);
        assert_pdf_close(predictor.pdf_next(), &direct, 1e-18);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_rate_predictor_matches_backend_after_partial_tbptt_stream() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=29,train=adam,lr=0.0008,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.0008,stride=1,bptt=8,clip=0,momentum=0.9)";
        let data = b"abcdefghij";
        let mut predictor = RwkvPredictor::from_method(method).expect("rwkv predictor");
        let mut backend = rwkvzip::Compressor::new_from_method(method).expect("rwkv backend");
        let mut direct = vec![0.0; backend.vocab_size()];

        predictor
            .begin_stream(data.len())
            .expect("begin predictor stream");
        backend
            .begin_online_policy_stream(Some(data.len() as u64))
            .expect("begin backend stream");
        backend.reset_and_prime();

        for &byte in data {
            let predicted = predictor.pdf_next().to_vec();
            backend.copy_current_pdf_to(&mut direct);
            assert_pdf_close(&predicted, &direct, 1e-18);

            predictor.update(byte).expect("predictor update");
            backend
                .observe_symbol_from_current_pdf(byte)
                .expect("backend update");
        }

        predictor.finish_stream().expect("finish predictor stream");
        backend
            .finish_online_policy_stream()
            .expect("finish backend stream");
        backend.copy_current_pdf_to(&mut direct);
        assert_pdf_close(predictor.pdf_next(), &direct, 1e-18);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn roundtrip_rate_rwkv_two_json_method_2m() {
        let two_json: serde_json::Value =
            serde_json::from_str(include_str!("../../examples/two.json")).unwrap();
        let method = two_json["experts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|expert| expert["name"].as_str() == Some("rwkv"))
            .and_then(|expert| expert["method"].as_str())
            .unwrap()
            .to_string();

        let backend = RateBackend::Rwkv7Method { method };
        let seed = include_bytes!("../../README.md");
        let target_len = 2_097_152usize;
        let mut data = Vec::with_capacity(target_len);
        while data.len() < target_len {
            let remaining = target_len - data.len();
            data.extend_from_slice(&seed[..seed.len().min(remaining)]);
        }

        let enc =
            compress_rate_bytes(&data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_rate_predictor_preserves_backend_pdf_exactly() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=7,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
        let mut predictor = MambaPredictor::from_method(method).expect("mamba predictor");
        let mut backend = mambazip::Compressor::new_from_method(method).expect("mamba backend");
        let mut direct = vec![0.0; backend.vocab_size()];

        let predicted = predictor.pdf_next().to_vec();
        backend.forward_to_pdf(0, &mut direct);
        assert_pdf_close(&predicted, &direct, 1e-18);

        predictor.update(b'x').expect("predictor update");
        backend
            .online_update_from_pdf(b'x', &direct)
            .expect("backend update");
        backend.forward_to_pdf(u32::from(b'x'), &mut direct);
        assert_pdf_close(predictor.pdf_next(), &direct, 1e-18);
    }

    #[test]
    fn roundtrip_rate_ac_particle() {
        let spec = crate::ParticleSpec {
            num_particles: 4,
            num_cells: 4,
            cell_dim: 8,
            num_rules: 2,
            selector_hidden: 16,
            rule_hidden: 16,
            context_window: 8,
            unroll_steps: 1,
            ..crate::ParticleSpec::default()
        };
        let data = b"particle ac roundtrip payload";
        let backend = RateBackend::Particle {
            spec: Arc::new(spec),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_rans_particle() {
        let spec = crate::ParticleSpec {
            num_particles: 4,
            num_cells: 4,
            cell_dim: 8,
            num_rules: 2,
            selector_hidden: 16,
            rule_hidden: 16,
            context_window: 8,
            unroll_steps: 1,
            ..crate::ParticleSpec::default()
        };
        let data = b"particle rans roundtrip payload";
        let backend = RateBackend::Particle {
            spec: Arc::new(spec),
        };
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::RANS, FramingMode::Framed).unwrap();
        let dec = decompress_rate_bytes(&enc, &backend, -1, CoderType::RANS, FramingMode::Framed)
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn mixture_with_particle_expert_roundtrip() {
        let particle_spec = crate::ParticleSpec {
            num_particles: 4,
            num_cells: 4,
            cell_dim: 8,
            num_rules: 2,
            selector_hidden: 16,
            rule_hidden: 16,
            context_window: 8,
            unroll_steps: 1,
            ..crate::ParticleSpec::default()
        };
        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("particle".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Particle {
                        spec: Arc::new(particle_spec),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Ctw { depth: 6 },
                },
            ],
        );
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let data = b"mixture with particle expert roundtrip";
        let enc =
            compress_rate_bytes(data, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, -1, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }
}
