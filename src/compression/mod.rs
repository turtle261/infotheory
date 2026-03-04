//! Rate-coded compression helpers (AC/rANS) with optional framing.
//!
//! The functions in this module implement lossless byte compression by combining:
//! - a predictive rate model (`RateBackend`) that emits per-symbol PDFs,
//! - an entropy coder (`AC` or `rANS`),
//! - optional framing metadata for robust decompression.

use anyhow::{Result, bail};

#[cfg(feature = "backend-rwkv")]
use crate::backends::rwkvzip;
use crate::coders::{
    ANS_TOTAL, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder, BlockedRansEncoder,
    CDF_TOTAL, Cdf, quantize_pdf_to_cdf_inplace, quantize_pdf_to_rans_cdf_with_buffer,
};
use crate::ctw::FacContextTree;
use crate::mixture::DEFAULT_MIN_PROB;
use crate::neural_mix::NeuralMixCore;
use crate::rosaplus::RosaPlus;
use crate::zpaq_rate::ZpaqRateModel;
use crate::{MixtureKind, MixtureSpec, RateBackend};

const FRAMED_MAGIC: u32 = 0x4354_4946; // "FITC"
const FRAMED_VERSION: u8 = 1;
const PDF_MIN: f64 = DEFAULT_MIN_PROB;

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

    fn new(coder: rwkvzip::CoderType, original_len: u64, crc32: u32) -> Self {
        Self {
            magic: FRAMED_MAGIC,
            version: FRAMED_VERSION,
            coder: match coder {
                rwkvzip::CoderType::AC => 0,
                rwkvzip::CoderType::RANS => 1,
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

    fn coder_type(&self) -> rwkvzip::CoderType {
        match self.coder {
            0 => rwkvzip::CoderType::AC,
            _ => rwkvzip::CoderType::RANS,
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
    fn bit_prob_one_msb(&mut self, bit_idx: usize) -> f64 {
        debug_assert!(self.bits_per_symbol == 8);
        debug_assert!(self.msb_first);
        self.tree.predict(true, bit_idx).clamp(PDF_MIN, 1.0 - PDF_MIN)
    }

    #[inline]
    fn update_bit_msb(&mut self, bit_idx: usize, bit: bool) {
        debug_assert!(self.bits_per_symbol == 8);
        debug_assert!(self.msb_first);
        self.tree.update(bit, bit_idx);
        self.valid = false;
    }
}

#[derive(Clone)]
struct RosaPredictor {
    model: RosaPlus,
    pdf: Vec<f64>,
    valid: bool,
}

impl RosaPredictor {
    fn new(max_order: i64) -> Self {
        let mut model = RosaPlus::new(max_order, false, 0, 42);
        model.build_lm_full_bytes_no_finalize_endpos();
        Self {
            model,
            pdf: vec![0.0; 256],
            valid: false,
        }
    }

    fn pdf_next(&mut self) -> &[f64] {
        if !self.valid {
            self.model.fill_probs_for_last_bytes(&mut self.pdf);
            for p in &mut self.pdf {
                *p = (*p).max(PDF_MIN);
            }
            normalize_pdf(&mut self.pdf);
            self.valid = true;
        }
        &self.pdf
    }

    fn update(&mut self, symbol: u8) {
        let mut tx = self.model.begin_tx();
        self.model.train_sequence_tx(&mut tx, &[symbol]);
        self.valid = false;
    }
}

#[derive(Clone)]
struct RwkvPredictor {
    compressor: rwkvzip::Compressor,
    primed: bool,
    pdf: Vec<f64>,
    valid: bool,
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

impl RwkvPredictor {
    #[cfg(feature = "backend-rwkv")]
    fn from_model(model: std::sync::Arc<rwkvzip::Model>) -> Self {
        let compressor = rwkvzip::Compressor::new_from_model(model);
        let vocab = compressor.vocab_size();
        Self {
            compressor,
            primed: false,
            pdf: vec![0.0; vocab],
            valid: false,
        }
    }

    #[cfg(feature = "backend-rwkv")]
    fn from_method(method: &str) -> Result<Self> {
        let compressor = rwkvzip::Compressor::new_from_method(method)?;
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
            let bias = self.compressor.online_bias_snapshot();
            let logits = self.compressor.model.forward(
                &mut self.compressor.scratch,
                0,
                &mut self.compressor.state,
            );
            rwkvzip::Compressor::logits_to_pdf(
                logits,
                bias.as_deref(),
                &mut self.compressor.pdf_buffer,
            );
            self.pdf.copy_from_slice(&self.compressor.pdf_buffer);
            self.primed = true;
            self.valid = true;
            return;
        }
        self.pdf.copy_from_slice(&self.compressor.pdf_buffer);
        self.valid = true;
    }

    fn pdf_next(&mut self) -> &[f64] {
        self.ensure_predicted();
        &self.pdf
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        self.ensure_predicted();
        self.compressor.online_update_from_pdf(symbol, &self.pdf)?;
        let bias = self.compressor.online_bias_snapshot();
        let logits = self.compressor.model.forward(
            &mut self.compressor.scratch,
            symbol as u32,
            &mut self.compressor.state,
        );
        rwkvzip::Compressor::logits_to_pdf(
            logits,
            bias.as_deref(),
            &mut self.compressor.pdf_buffer,
        );
        self.valid = false;
        Ok(())
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
        let m = logsumexp(experts.iter().map(|e| e.log_weight));
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
        let neural = NeuralMixCore::new(
            experts.len(),
            &prior_weights,
            effective_lr * 0.5,
            effective_lr,
            1e-5,
        );
        Ok(Self {
            kind: spec.kind,
            alpha: spec.alpha.clamp(1e-12, 1.0 - 1e-12),
            decay: spec.decay.unwrap_or(1.0).clamp(0.0, 1.0),
            experts,
            neural,
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
                self.neural.evaluate_expert_weights();
                let n = self.experts.len();
                self.scratch.resize(n, 0.0);
                self.scratch.copy_from_slice(self.neural.expert_weights());
                self.pdf.fill(0.0);
                for i in 0..n {
                    let epdf = self.experts[i].predictor.pdf_next()?;
                    let w = self.scratch[i];
                    for b in 0..256 {
                        self.pdf[b] += w * epdf[b];
                    }
                }
                normalize_pdf(&mut self.pdf);
                self.valid = true;
                return Ok(&self.pdf);
            }
            _ => {
                self.pdf.fill(0.0);

                let lw_norm = logsumexp(self.experts.iter().map(|e| e.log_weight));
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
                let log_mix = logsumexp(self.scratch2.iter().copied());
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
                let log_mix = logsumexp(self.scratch2.iter().copied());
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
                let log_mix = logsumexp(self.scratch2.iter().copied());
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
                    self.neural.update_history(symbol);
                    self.valid = false;
                    return Ok(());
                }
                let n = self.experts.len();
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
                self.neural.update_history(symbol);
            }
        }

        self.valid = false;
        Ok(())
    }

    #[inline]
    fn can_fast_neural_ac_bitwise(&self) -> bool {
        if self.kind != MixtureKind::Neural || self.experts.len() <= 1 {
            return false;
        }
        self.experts.iter().any(|e| {
            if let RatePdfPredictor::Ctw(ctw) = &*e.predictor {
                ctw.bits_per_symbol == 8 && ctw.msb_first
            } else {
                false
            }
        })
    }

    fn ac_step_neural_bitwise<F>(&mut self, mut choose_bit: F) -> Result<u8>
    where
        F: FnMut(usize, f64) -> Result<u8>,
    {
        debug_assert_eq!(self.kind, MixtureKind::Neural);
        debug_assert!(self.experts.len() > 1);

        let n = self.experts.len();
        self.neural.evaluate_expert_weights();
        self.scratch.resize(n, 0.0);
        self.scratch.copy_from_slice(self.neural.expert_weights());
        self.scratch2.resize(n, 1.0);
        self.scratch2.fill(1.0);
        self.neural_logps.resize(n, 0.0);
        self.neural_bit_modes.resize(n, 0);
        self.neural_lo.resize(n, 0);
        self.neural_hi.resize(n, 256);
        if self.neural_pdf_cdf_rows.len() < n {
            self.neural_pdf_cdf_rows
                .resize_with(n, || vec![0.0; 257]);
        }

        for i in 0..n {
            self.neural_bit_modes[i] = 1;
            self.neural_lo[i] = 0;
            self.neural_hi[i] = 256;

            let mut handled_ctw = false;
            if let RatePdfPredictor::Ctw(ctw) = &mut *self.experts[i].predictor {
                if ctw.bits_per_symbol == 8 && ctw.msb_first {
                    self.neural_bit_modes[i] = 0;
                    handled_ctw = true;
                }
            }
            if handled_ctw {
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
            let norm = row[256];
            if norm.is_finite() && norm > 0.0 {
                let inv = 1.0 / norm;
                for v in row.iter_mut() {
                    *v *= inv;
                }
            } else {
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

        self.neural.evaluate_symbol(&self.neural_logps, PDF_MIN);
        self.neural.update_weights_symbol(&self.neural_logps, PDF_MIN);
        self.neural.update_history(symbol);
        self.valid = false;
        Ok(symbol)
    }
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum RatePdfPredictor {
    Rosa(RosaPredictor),
    Ctw(CtwPredictor),
    FacCtw(CtwPredictor),
    #[cfg(feature = "backend-rwkv")]
    Rwkv(RwkvPredictor),
    Zpaq(ZpaqPredictor),
    Mixture(MixturePredictor),
    Particle(crate::particle::ParticleRuntime),
}

impl RatePdfPredictor {
    fn from_rate_backend(backend: RateBackend, max_order: i64) -> Result<Self> {
        match backend {
            RateBackend::RosaPlus => Ok(Self::Rosa(RosaPredictor::new(max_order))),
            RateBackend::Ctw { depth } => Ok(Self::Ctw(CtwPredictor::new_ctw(depth))),
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits: _,
                encoding_bits,
            } => {
                let bits = encoding_bits.clamp(1, 8);
                Ok(Self::FacCtw(CtwPredictor::new_fac(base_depth, bits)))
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
        }
    }

    fn pdf_next(&mut self) -> Result<&[f64]> {
        match self {
            Self::Rosa(m) => Ok(m.pdf_next()),
            Self::Ctw(m) => Ok(m.pdf_next()),
            Self::FacCtw(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => Ok(m.pdf_next()),
            Self::Zpaq(m) => Ok(m.pdf_next()),
            Self::Mixture(m) => m.ensure_pdf(),
            Self::Particle(m) => Ok(m.pdf_next()),
        }
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        match self {
            Self::Rosa(m) => {
                m.update(symbol);
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
        }
    }
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
    if let RatePdfPredictor::Mixture(mix) = predictor {
        if mix.can_fast_neural_ac_bitwise() {
            let mut out = Vec::new();
            {
                let mut enc = ArithmeticEncoder::new(&mut out);
                for &symbol in data {
                    mix.ac_step_neural_bitwise(|bit_idx, p1_mix| {
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
            return Ok(out);
        }
    }

    let mut out = Vec::new();
    {
        let mut enc = ArithmeticEncoder::new(&mut out);
        let mut cdf = vec![0u32; 257];
        for &b in data {
            let pdf = predictor.pdf_next()?;
            quantize_pdf_to_cdf_inplace(pdf, &mut cdf);
            let sym = b as usize;
            enc.encode_counts(cdf[sym] as u64, cdf[sym + 1] as u64, CDF_TOTAL as u64)?;
            predictor.update(b)?;
        }
        let _ = enc.finish()?;
    }
    Ok(out)
}

fn decode_payload_ac(
    payload: &[u8],
    out_len: usize,
    predictor: &mut RatePdfPredictor,
) -> Result<Vec<u8>> {
    if let RatePdfPredictor::Mixture(mix) = predictor {
        if mix.can_fast_neural_ac_bitwise() {
            let mut dec = ArithmeticDecoder::new(payload)?;
            let mut out = Vec::with_capacity(out_len);
            for _ in 0..out_len {
                let symbol = mix.ac_step_neural_bitwise(|_, p1_mix| {
                    let split = binary_split_from_prob_one(p1_mix);
                    let cdf = [0u32, split, CDF_TOTAL];
                    Ok(dec.decode_symbol_counts(&cdf, CDF_TOTAL)? as u8)
                })?;
                out.push(symbol);
            }
            return Ok(out);
        }
    }

    let mut dec = ArithmeticDecoder::new(payload)?;
    let mut out = Vec::with_capacity(out_len);
    let mut cdf = vec![0u32; 257];
    for _ in 0..out_len {
        let pdf = predictor.pdf_next()?;
        quantize_pdf_to_cdf_inplace(pdf, &mut cdf);
        let sym = dec.decode_symbol_counts(&cdf, CDF_TOTAL)? as u8;
        out.push(sym);
        predictor.update(sym)?;
    }
    Ok(out)
}

fn encode_payload_rans(data: &[u8], predictor: &mut RatePdfPredictor) -> Result<Vec<u8>> {
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
    Ok(out)
}

fn decode_payload_rans(
    payload: &[u8],
    out_len: usize,
    predictor: &mut RatePdfPredictor,
) -> Result<Vec<u8>> {
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

    let mut dec = BlockedRansDecoder::new(blocks);
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
    coder: rwkvzip::CoderType,
    framing: FramingMode,
) -> Result<Vec<u8>> {
    let mut predictor = RatePdfPredictor::from_rate_backend(rate_backend.clone(), max_order)?;
    let payload = match coder {
        rwkvzip::CoderType::AC => encode_payload_ac(data, &mut predictor)?,
        rwkvzip::CoderType::RANS => encode_payload_rans(data, &mut predictor)?,
    };

    if framing == FramingMode::Raw {
        return Ok(payload);
    }

    let mut out = Vec::with_capacity(FramedHeader::SIZE + payload.len());
    let hdr = FramedHeader::new(coder, data.len() as u64, rwkvzip::crc32(data));
    hdr.write(&mut out);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Return compressed size (in bytes) for `data` using rate coding.
pub fn compress_rate_size(
    data: &[u8],
    rate_backend: &RateBackend,
    max_order: i64,
    coder: rwkvzip::CoderType,
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
    coder: rwkvzip::CoderType,
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
    _coder: rwkvzip::CoderType,
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
        rwkvzip::CoderType::AC => decode_payload_ac(payload, out_len, &mut predictor)?,
        rwkvzip::CoderType::RANS => decode_payload_rans(payload, out_len, &mut predictor)?,
    };

    if let Some(crc) = expected_crc {
        let got = rwkvzip::crc32(&decoded);
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

fn logsumexp<I: Iterator<Item = f64>>(it: I) -> f64 {
    let vals: Vec<f64> = it.collect();
    let mut m = f64::NEG_INFINITY;
    for &v in &vals {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &v in &vals {
        s += (v - m).exp();
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

    fn brute_force_pdf(predictor: &mut CtwPredictor) -> Vec<f64> {
        let bits = predictor.bits_per_symbol.clamp(1, 8);
        let mut out = vec![0.0; 256];

        if bits == 8 {
            for sym in 0..256usize {
                out[sym] = predictor.log_prob_symbol_bruteforce(sym as u8).exp();
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
            for byte in 0..256usize {
                let pat = if predictor.msb_first {
                    byte >> (8 - bits)
                } else {
                    byte & (patterns - 1)
                };
                out[byte] = pat_prob[pat] / (aliases as f64);
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
        let enc = compress_rate_bytes(
            data,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        let dec = decompress_rate_bytes(
            &enc,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
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
        let enc = compress_rate_bytes(
            data,
            &backend,
            -1,
            rwkvzip::CoderType::RANS,
            FramingMode::Framed,
        )
        .unwrap();
        let dec = decompress_rate_bytes(
            &enc,
            &backend,
            -1,
            rwkvzip::CoderType::RANS,
            FramingMode::Framed,
        )
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
        let enc = compress_rate_bytes(
            data,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        let dec = decompress_rate_bytes(
            &enc,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
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

    #[test]
    fn raw_size_not_larger_than_framed_size() {
        let data = b"raw/framed size check payload";
        let backend = RateBackend::RosaPlus;
        let raw = compress_rate_size(data, &backend, 8, rwkvzip::CoderType::AC, FramingMode::Raw)
            .unwrap();
        let framed = compress_rate_size(
            data,
            &backend,
            8,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        assert!(framed >= raw);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn roundtrip_rate_rwkv_method_cfg() {
        let data = b"rwkv cfg method backend";
        let backend = RateBackend::Rwkv7Method {
            method: "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1".to_string(),
        };
        let enc = compress_rate_bytes(
            data,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        let dec = decompress_rate_bytes(
            &enc,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        assert_eq!(dec, data);
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
        let enc = compress_rate_bytes(
            data,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        let dec = decompress_rate_bytes(
            &enc,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
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
        let enc = compress_rate_bytes(
            data,
            &backend,
            -1,
            rwkvzip::CoderType::RANS,
            FramingMode::Framed,
        )
        .unwrap();
        let dec = decompress_rate_bytes(
            &enc,
            &backend,
            -1,
            rwkvzip::CoderType::RANS,
            FramingMode::Framed,
        )
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
        let enc = compress_rate_bytes(
            data,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        let dec = decompress_rate_bytes(
            &enc,
            &backend,
            -1,
            rwkvzip::CoderType::AC,
            FramingMode::Framed,
        )
        .unwrap();
        assert_eq!(dec, data);
    }
}
