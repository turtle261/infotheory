//! Rate-coded compression helpers (AC/rANS) with optional framing.
//!
//! The functions in this module implement lossless byte compression by combining:
//! - a predictive rate model (`RateBackend`) that emits per-symbol PDFs,
//! - an entropy coder (`AC` or `rANS`),
//! - optional framing metadata for robust decompression.
#![cfg_attr(
    not(feature = "all-backends"),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)]

use anyhow::{Result, bail};

use crate::api::{MixtureKind, MixtureScheduleMode};
#[cfg(test)]
use crate::api::{MixtureSpec, RateBackend};
#[cfg(feature = "backend-bit-reservoir")]
use crate::backends::bit_reservoir::{BitReservoirModel, BitReservoirPrediction};
#[cfg(feature = "backend-calibrated")]
use crate::backends::calibration::CalibratorCore;
#[cfg(feature = "backend-context")]
use crate::backends::context_counter::{OrderNGramModel, WordContextModel};
#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{ContextTree, FacContextTree, ctw_symbol_bit_msb};
#[cfg(feature = "backend-match")]
use crate::backends::match_model::MatchModel;
#[cfg(feature = "backend-ppmd")]
use crate::backends::ppmd::PpmdModel;
#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::RosaPlus;
#[cfg(feature = "backend-sequitur")]
use crate::backends::sequitur::SequiturModel;
#[cfg(feature = "backend-match")]
use crate::backends::sparse_match::SparseMatchModel;
use crate::backends::text_context::{TextContextAnalyzer, bucket_repeat_len, classify_byte};
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
#[cfg(all(test, feature = "all-backends"))]
use crate::byte_prefix::zeroed_prefix_cdf;
use crate::byte_prefix::{
    BytePrefixCdf, MsbPrefixRange, advanced_prefix_code, fill_prefix_cdf_from_pdf, normalize_pdf,
    zeroed_prefix_cdf_box,
};
use crate::coders::{
    ANS_TOTAL, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder, BlockedRansEncoder,
    CDF_TOTAL, Cdf, CoderType, crc32, quantize_pdf_to_rans_cdf_with_buffer,
};
#[cfg(feature = "backend-mamba")]
use crate::mambazip;
use crate::mixture::{
    DEFAULT_MIN_PROB, convex_step_size_for_update, project_simplex_with_scratch,
    switching_alpha_for_update,
};
use crate::neural_mix::LogisticMatchState;
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip;
use crate::spec::CompiledRateBackend;
use rayon::{ThreadPool, prelude::*};

mod mixture_predictor;

pub(crate) use mixture_predictor::MixturePredictor;

const FRAMED_MAGIC: u32 = 0x4354_4946; // "FITC"
const FRAMED_VERSION: u8 = 1;
const PDF_MIN: f64 = DEFAULT_MIN_PROB;
const DIAGNOSTIC_PARALLEL_THRESHOLD: usize = 4;

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
        let coder = match coder {
            CoderType::AC => 0,
            CoderType::RANS => 1,
        };
        Self {
            magic: FRAMED_MAGIC,
            version: FRAMED_VERSION,
            coder,
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

#[cfg(feature = "backend-ctw")]
#[derive(Clone)]
enum CtwCompressionTree {
    Ac(Box<ContextTree>),
    Fac(FacContextTree),
}

#[cfg(feature = "backend-ctw")]
#[derive(Clone)]
pub(crate) struct CtwPredictor {
    tree: CtwCompressionTree,
    bits_per_symbol: usize,
    msb_first: bool,
    pdf: Vec<f64>,
    pattern_logps: Vec<f64>,
    valid: bool,
}

#[cfg(feature = "backend-ctw")]
impl CtwPredictor {
    pub(crate) fn new_ctw(depth: usize) -> Self {
        Self {
            tree: CtwCompressionTree::Ac(Box::new(ContextTree::new(depth))),
            bits_per_symbol: 8,
            msb_first: true,
            pdf: vec![0.0; 256],
            pattern_logps: vec![f64::NEG_INFINITY; 256],
            valid: false,
        }
    }

    pub(crate) fn new_fac(
        base_depth: usize,
        bits_per_symbol: usize,
        msb_first: Option<bool>,
    ) -> Self {
        let effective_msb_first: bool = msb_first.unwrap_or(bits_per_symbol == 8);
        Self {
            tree: CtwCompressionTree::Fac(FacContextTree::new(base_depth, bits_per_symbol)),
            bits_per_symbol,
            msb_first: effective_msb_first,
            pdf: vec![0.0; 256],
            pattern_logps: vec![f64::NEG_INFINITY; 256],
            valid: false,
        }
    }

    fn fill_pattern_log_probs(&mut self) -> usize {
        let bits = self.bits_per_symbol.clamp(1, 8);
        let patterns = 1usize << bits;
        self.pattern_logps[..patterns].fill(f64::NEG_INFINITY);
        match &mut self.tree {
            CtwCompressionTree::Ac(tree) => {
                fn rec(
                    tree: &mut ContextTree,
                    depth: usize,
                    bits: usize,
                    pattern: usize,
                    log_before: f64,
                    out: &mut [f64],
                ) {
                    if depth == bits {
                        out[pattern] = tree.get_log_block_probability() - log_before;
                        return;
                    }
                    for bit in [false, true] {
                        tree.update(bit);
                        rec(
                            tree,
                            depth + 1,
                            bits,
                            (pattern << 1) | (bit as usize),
                            log_before,
                            out,
                        );
                        tree.revert();
                    }
                }

                let log_before = tree.get_log_block_probability();
                rec(
                    tree,
                    0,
                    bits,
                    0,
                    log_before,
                    &mut self.pattern_logps[..patterns],
                );
            }
            CtwCompressionTree::Fac(tree) => {
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

                let log_before = tree.get_log_block_probability();
                rec(
                    tree,
                    bits,
                    self.msb_first,
                    0,
                    0,
                    log_before,
                    &mut self.pattern_logps[..patterns],
                );
            }
        }
        patterns
    }

    #[cfg(test)]
    fn log_prob_symbol_bruteforce(&mut self, symbol: u8) -> f64 {
        let bits = self.bits_per_symbol.clamp(1, 8);
        match &mut self.tree {
            CtwCompressionTree::Ac(tree) => {
                debug_assert!(self.msb_first);
                let before = tree.get_log_block_probability();
                for bit_idx in 0..bits {
                    tree.update(ctw_symbol_bit_msb(symbol, bits, bit_idx));
                }
                let after = tree.get_log_block_probability();
                for _ in 0..bits {
                    tree.revert();
                }
                after - before
            }
            CtwCompressionTree::Fac(tree) => {
                let before = tree.get_log_block_probability();
                if self.msb_first {
                    for bit_idx in 0..bits {
                        let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
                        tree.update(bit, bit_idx);
                    }
                    let after = tree.get_log_block_probability();
                    for bit_idx in (0..bits).rev() {
                        tree.revert(bit_idx);
                    }
                    after - before
                } else {
                    for bit_idx in 0..bits {
                        let bit = ((symbol >> bit_idx) & 1) == 1;
                        tree.update(bit, bit_idx);
                    }
                    let after = tree.get_log_block_probability();
                    for bit_idx in (0..bits).rev() {
                        tree.revert(bit_idx);
                    }
                    after - before
                }
            }
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
            normalize_pdf(&mut self.pdf, PDF_MIN);
            self.valid = true;
        }
        &self.pdf
    }

    fn update(&mut self, symbol: u8) {
        match &mut self.tree {
            CtwCompressionTree::Ac(tree) => {
                debug_assert!(self.msb_first);
                for bit_idx in 0..self.bits_per_symbol {
                    tree.update(ctw_symbol_bit_msb(symbol, self.bits_per_symbol, bit_idx));
                }
            }
            CtwCompressionTree::Fac(tree) => {
                if self.msb_first {
                    for bit_idx in 0..self.bits_per_symbol {
                        let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
                        tree.update(bit, bit_idx);
                    }
                } else {
                    for bit_idx in 0..self.bits_per_symbol {
                        let bit = ((symbol >> bit_idx) & 1) == 1;
                        tree.update(bit, bit_idx);
                    }
                }
            }
        }
        self.valid = false;
    }

    fn reserve_for_symbols(&mut self, total_symbols: usize) {
        match &mut self.tree {
            CtwCompressionTree::Ac(tree) => tree.reserve_for_symbols(
                total_symbols.saturating_mul(self.bits_per_symbol.clamp(1, 8)),
            ),
            CtwCompressionTree::Fac(tree) => tree.reserve_for_symbols(total_symbols),
        }
    }

    #[inline]
    fn can_fast_ac_bitwise(&self) -> bool {
        self.bits_per_symbol == 8 && self.msb_first
    }

    #[inline]
    fn bit_prob_one_msb(&mut self, bit_idx: usize) -> f64 {
        debug_assert!(self.can_fast_ac_bitwise());
        match &mut self.tree {
            CtwCompressionTree::Ac(tree) => tree.predict(true).clamp(PDF_MIN, 1.0 - PDF_MIN),
            CtwCompressionTree::Fac(tree) => {
                tree.predict_one(bit_idx).clamp(PDF_MIN, 1.0 - PDF_MIN)
            }
        }
    }

    #[inline]
    fn update_bit_msb(&mut self, bit_idx: usize, bit: bool) {
        debug_assert!(self.can_fast_ac_bitwise());
        match &mut self.tree {
            CtwCompressionTree::Ac(tree) => tree.update(bit),
            CtwCompressionTree::Fac(tree) => tree.update_predicted(bit, bit_idx),
        }
        self.valid = false;
    }
}

#[cfg(feature = "backend-rosa")]
#[derive(Clone)]
pub(crate) struct RosaPredictor {
    model: RosaPlus,
    pdf: Vec<f64>,
    cdf: [f64; 257],
    valid: bool,
    cdf_valid: bool,
}

#[cfg(feature = "backend-rosa")]
impl RosaPredictor {
    pub(crate) fn new(max_order: i64) -> Self {
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
pub(crate) struct MambaPredictor {
    compressor: mambazip::Compressor,
    primed: bool,
    pdf: Vec<f64>,
    cdf: [f64; 257],
    valid: bool,
    cdf_valid: bool,
}

#[derive(Clone)]
#[cfg(feature = "backend-rwkv")]
pub(crate) struct RwkvPredictor {
    compressor: rwkvzip::Compressor,
    primed: bool,
    cdf: [f64; 257],
    cdf_valid: bool,
}

#[cfg(feature = "backend-zpaq")]
#[derive(Clone)]
pub(crate) struct ZpaqPredictor {
    method: String,
    history: Vec<u8>,
    pdf: Vec<f64>,
    valid: bool,
}

#[cfg(feature = "backend-zpaq")]
impl ZpaqPredictor {
    pub(crate) fn new(method: String) -> Self {
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
            normalize_pdf(&mut self.pdf, PDF_MIN);
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
    #[cfg(test)]
    fn from_method(method: &str) -> Result<Self> {
        let spec = mambazip::parse_method_spec(method)?;
        Self::from_method_spec(&spec)
    }

    pub(crate) fn from_method_spec(method: &mambazip::MethodSpec) -> Result<Self> {
        let compressor = mambazip::Compressor::new_from_method_spec(method)?;
        let vocab = compressor.vocab_size();
        Ok(Self {
            compressor,
            primed: false,
            pdf: vec![0.0; vocab],
            cdf: uniform_cdf_row(),
            valid: false,
            cdf_valid: false,
        })
    }

    fn ensure_predicted(&mut self, want_cdf: bool) {
        if self.valid {
            if want_cdf && !self.cdf_valid {
                debug_assert!(self.pdf.len() >= 256);
                build_cdf_row_from_pdf_slice(&self.pdf[..256], &mut self.cdf);
                self.cdf_valid = true;
            }
            return;
        }
        if !self.primed {
            self.compressor.forward_to_pdf(0, &mut self.pdf);
            self.primed = true;
            self.valid = true;
            self.cdf_valid = false;
            if want_cdf {
                debug_assert!(self.pdf.len() >= 256);
                build_cdf_row_from_pdf_slice(&self.pdf[..256], &mut self.cdf);
                self.cdf_valid = true;
            }
            return;
        }
        self.valid = true;
        self.cdf_valid = false;
        if want_cdf {
            debug_assert!(self.pdf.len() >= 256);
            build_cdf_row_from_pdf_slice(&self.pdf[..256], &mut self.cdf);
            self.cdf_valid = true;
        }
    }

    fn pdf_next(&mut self) -> &[f64] {
        self.ensure_predicted(false);
        &self.pdf
    }

    fn cdf_next(&mut self) -> &[f64; 257] {
        self.ensure_predicted(true);
        &self.cdf
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        self.ensure_predicted(false);
        self.compressor.online_update_from_pdf(symbol, &self.pdf)?;
        self.compressor.forward_to_pdf(symbol as u32, &mut self.pdf);
        self.valid = true;
        self.cdf_valid = false;
        Ok(())
    }

    fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        self.compressor
            .begin_online_policy_stream(Some(total_len as u64))
    }
}

#[cfg(feature = "backend-rwkv")]
impl RwkvPredictor {
    #[cfg(test)]
    fn from_method(method: &str) -> Result<Self> {
        let spec = rwkvzip::parse_method_spec(method)?;
        Self::from_method_spec(&spec)
    }

    pub(crate) fn from_method_spec(method: &rwkvzip::MethodSpec) -> Result<Self> {
        let compressor = rwkvzip::Compressor::new_from_method_spec(method)?;
        Ok(Self {
            compressor,
            primed: false,
            cdf: uniform_cdf_row(),
            cdf_valid: false,
        })
    }

    fn ensure_predicted(&mut self, want_cdf: bool) {
        if !self.primed {
            self.compressor.reset_and_prime();
            self.primed = true;
            self.cdf_valid = false;
        }
        if want_cdf && !self.cdf_valid {
            debug_assert!(self.compressor.pdf_buffer.len() >= 256);
            build_cdf_row_from_pdf_slice(&self.compressor.pdf_buffer[..256], &mut self.cdf);
            self.cdf_valid = true;
        }
    }

    fn pdf_next(&mut self) -> &[f64] {
        self.ensure_predicted(false);
        &self.compressor.pdf_buffer
    }

    fn cdf_next(&mut self) -> &[f64; 257] {
        self.ensure_predicted(true);
        &self.cdf
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        self.ensure_predicted(false);
        self.compressor.observe_symbol_from_current_pdf(symbol)?;
        self.cdf_valid = false;
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

#[derive(Clone, Debug)]
pub(crate) enum PredictorBitwiseStepState {
    NativeRecursive,
    CachedCdf {
        range: MsbPrefixRange,
    },
    PdfPrefix {
        cdf: Box<BytePrefixCdf>,
        range: MsbPrefixRange,
    },
}

impl Default for PredictorBitwiseStepState {
    fn default() -> Self {
        Self::PdfPrefix {
            cdf: zeroed_prefix_cdf_box(),
            range: MsbPrefixRange::FULL,
        }
    }
}

impl PredictorBitwiseStepState {
    fn prepare(&mut self, predictor: &mut RatePdfPredictor) -> Result<()> {
        if predictor.begin_native_recursive_bitwise_byte_step()? {
            *self = Self::NativeRecursive;
            return Ok(());
        }
        if predictor.prepare_cached_cdf_fast_bitwise()? {
            *self = Self::CachedCdf {
                range: MsbPrefixRange::FULL,
            };
            return Ok(());
        }

        self.prepare_pdf_prefix(predictor)
    }

    fn prepare_pdf_prefix(&mut self, predictor: &mut RatePdfPredictor) -> Result<()> {
        let mut cdf = match std::mem::take(self) {
            Self::PdfPrefix { cdf, .. } => cdf,
            _ => zeroed_prefix_cdf_box(),
        };
        fill_prefix_cdf_from_pdf(&mut cdf, predictor.pdf_next()?, PDF_MIN);
        *self = Self::PdfPrefix {
            cdf,
            range: MsbPrefixRange::FULL,
        };
        Ok(())
    }

    fn bit_prob_one_msb(
        &mut self,
        predictor: &mut RatePdfPredictor,
        bit_idx: usize,
    ) -> Result<f64> {
        match self {
            Self::NativeRecursive => predictor.native_recursive_bit_prob_one_msb(bit_idx),
            Self::CachedCdf { range } => Ok(predictor
                .cached_cdf_bit_prob_one_msb(*range)
                .expect("CachedCdf state invariant violated: missing cached CDF entry")),
            Self::PdfPrefix { cdf, range } => Ok(range.prob_one(cdf.as_ref(), PDF_MIN)),
        }
    }

    fn observe_bit_msb(
        &mut self,
        predictor: &mut RatePdfPredictor,
        bit_idx: usize,
        bit: bool,
    ) -> Result<()> {
        match self {
            Self::NativeRecursive => predictor.native_recursive_observe_bit_msb(bit_idx, bit),
            Self::CachedCdf { range } | Self::PdfPrefix { range, .. } => {
                range.observe(bit);
                Ok(())
            }
        }
    }

    fn observe_known_bit_msb(
        &mut self,
        predictor: &mut RatePdfPredictor,
        bit_idx: usize,
        bit: bool,
    ) -> Result<f64> {
        match self {
            Self::NativeRecursive => predictor.native_recursive_observe_known_bit_msb(bit_idx, bit),
            Self::CachedCdf { range } => {
                let p1 = predictor
                    .cached_cdf_bit_prob_one_msb(*range)
                    .expect("CachedCdf state invariant violated: missing cached CDF entry");
                range.observe(bit);
                Ok(p1)
            }
            Self::PdfPrefix { cdf, range } => {
                let p1 = range.prob_one(cdf.as_ref(), PDF_MIN);
                range.observe(bit);
                Ok(p1)
            }
        }
    }

    fn finish_symbol(&mut self, predictor: &mut RatePdfPredictor, symbol: u8) -> Result<()> {
        match self {
            Self::NativeRecursive => predictor.finish_native_recursive_bitwise_byte_step(symbol),
            Self::CachedCdf { .. } | Self::PdfPrefix { .. } => predictor.update(symbol),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AcLogLossNodeValue {
    pub(crate) prob: f64,
    pub(crate) local_weight: f64,
    pub(crate) effective_weight: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct AcLogLossSubtreeSnapshot {
    pub(crate) prob: f64,
    pub(crate) rows: Vec<AcLogLossNodeValue>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AcLogLossRootSnapshot {
    pub(crate) mix_prob: f64,
    pub(crate) root_weight_entropy_bits: f64,
    pub(crate) root_top1_child_index: Option<usize>,
    pub(crate) root_top1_weight: f64,
    pub(crate) root_top2_child_index: Option<usize>,
    pub(crate) root_top2_weight: f64,
}

pub(crate) struct DiagnosticRatePredictor {
    inner: RatePdfPredictor,
    // Reused across the whole diagnostic stream (see `ac_step_bitwise_with_state`
    // doc comment) so per-byte AC stepping does not pay a fresh scratch
    // allocation for backends that immediately discard it in `prepare()`.
    bitwise_scratch: PredictorBitwiseStepState,
}

impl DiagnosticRatePredictor {
    #[cfg(test)]
    pub(crate) fn from_rate_backend(backend: RateBackend) -> Result<Self> {
        let compiled = backend.compile().map_err(anyhow::Error::msg)?;
        Self::from_compiled(&compiled)
    }

    pub(crate) fn from_compiled(backend: &CompiledRateBackend) -> Result<Self> {
        Ok(Self {
            inner: crate::runtime::build_rate_pdf_predictor(backend)?,
            bitwise_scratch: PredictorBitwiseStepState::default(),
        })
    }

    pub(crate) fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        self.bitwise_scratch = PredictorBitwiseStepState::default();
        self.inner.begin_stream(total_len)
    }

    pub(crate) fn finish_stream(&mut self) -> Result<()> {
        self.inner.finish_stream()
    }

    #[cfg(test)]
    pub(crate) fn pdf_next(&mut self) -> Result<&[f64]> {
        self.inner.pdf_next()
    }

    #[cfg(test)]
    pub(crate) fn update(&mut self, symbol: u8) -> Result<()> {
        self.inner.update(symbol)
    }

    pub(crate) fn diagnostic_root_snapshot(
        &mut self,
        symbol: u8,
        pool: Option<&ThreadPool>,
        out: &mut Vec<AcLogLossNodeValue>,
    ) -> Result<AcLogLossRootSnapshot> {
        self.inner.diagnostic_root_snapshot(symbol, pool, out)
    }

    pub(crate) fn encode_symbol_ac_step<W: std::io::Write>(
        &mut self,
        symbol: u8,
        encoder: &mut ArithmeticEncoder<W>,
        cdf: &mut [u32; 257],
    ) -> Result<()> {
        self.inner
            .encode_symbol_ac_step(&mut self.bitwise_scratch, symbol, encoder, cdf)
    }
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum RatePdfPredictor {
    #[cfg(feature = "backend-rosa")]
    Rosa(RosaPredictor),
    #[cfg(feature = "backend-match")]
    Match { model: MatchModel },
    #[cfg(feature = "backend-match")]
    SparseMatch { model: SparseMatchModel },
    #[cfg(feature = "backend-context")]
    OrderNGram { model: OrderNGramModel },
    #[cfg(feature = "backend-context")]
    WordContext { model: WordContextModel },
    #[cfg(feature = "backend-ppmd")]
    Ppmd { model: PpmdModel },
    #[cfg(feature = "backend-sequitur")]
    Sequitur { model: SequiturModel },
    #[cfg(feature = "backend-ctw")]
    Ctw(CtwPredictor),
    #[cfg(feature = "backend-ctw")]
    FacCtw(CtwPredictor),
    #[cfg(feature = "backend-mamba")]
    Mamba(MambaPredictor),
    #[cfg(feature = "backend-rwkv")]
    Rwkv(RwkvPredictor),
    #[cfg(feature = "backend-bit-reservoir")]
    BitReservoir {
        model: BitReservoirModel,
        pdf: Vec<f64>,
        valid: bool,
        native_prefix_progress: Option<usize>,
        native_prediction: Option<(usize, BitReservoirPrediction)>,
    },
    #[cfg(feature = "backend-zpaq")]
    Zpaq(ZpaqPredictor),
    #[cfg(feature = "backend-mixture")]
    Mixture(MixturePredictor),
    #[cfg(feature = "backend-particle")]
    Particle(crate::backends::particle::ParticleRuntime),
    #[cfg(feature = "backend-calibrated")]
    Calibrated {
        base: Box<RatePdfPredictor>,
        core: CalibratorCore,
        bitwise: PredictorBitwiseStepState,
        pdf: Vec<f64>,
        valid: bool,
    },
    #[allow(dead_code)]
    Disabled { reason: String },
}

impl RatePdfPredictor {
    #[cfg(test)]
    fn from_compiled(backend: &CompiledRateBackend) -> Result<Self> {
        crate::runtime::build_rate_pdf_predictor(backend)
    }

    #[cfg(test)]
    pub(crate) fn from_rate_backend(backend: RateBackend) -> Result<Self> {
        let compiled = backend.compile().map_err(anyhow::Error::msg)?;
        Self::from_compiled(&compiled)
    }

    fn begin_stream(&mut self, total_len: usize) -> Result<()> {
        self.finish_stream()?;
        match self {
            #[cfg(feature = "backend-rosa")]
            Self::Rosa(m) => {
                m.begin_stream(total_len);
                Ok(())
            }
            #[cfg(feature = "backend-match")]
            Self::Match { .. } => Ok(()),
            #[cfg(feature = "backend-match")]
            Self::SparseMatch { .. } => Ok(()),
            #[cfg(feature = "backend-context")]
            Self::OrderNGram { .. } => Ok(()),
            #[cfg(feature = "backend-context")]
            Self::WordContext { .. } => Ok(()),
            #[cfg(feature = "backend-ppmd")]
            Self::Ppmd { .. } => Ok(()),
            #[cfg(feature = "backend-zpaq")]
            Self::Zpaq(_) => Ok(()),
            #[cfg(feature = "backend-particle")]
            Self::Particle(_) => Ok(()),
            #[cfg(feature = "backend-sequitur")]
            Self::Sequitur { model } => {
                model.begin_stream(Some(total_len as u64));
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) | Self::FacCtw(m) => {
                m.reserve_for_symbols(total_len);
                Ok(())
            }
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => m.begin_stream(total_len),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => m.begin_stream(total_len),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                model,
                valid,
                native_prefix_progress,
                native_prediction,
                ..
            } => {
                model.reset_all();
                *valid = false;
                *native_prefix_progress = None;
                *native_prediction = None;
                Ok(())
            }
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.begin_stream(total_len),
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                bitwise,
                valid,
                ..
            } => {
                *bitwise = PredictorBitwiseStepState::default();
                *valid = false;
                base.begin_stream(total_len)
            }
            Self::Disabled { reason } => bail!("{reason}"),
        }
    }

    fn finish_stream(&mut self) -> Result<()> {
        match self {
            #[cfg(feature = "backend-rosa")]
            Self::Rosa(_) => Ok(()),
            #[cfg(feature = "backend-match")]
            Self::Match { .. } => Ok(()),
            #[cfg(feature = "backend-match")]
            Self::SparseMatch { .. } => Ok(()),
            #[cfg(feature = "backend-context")]
            Self::OrderNGram { .. } => Ok(()),
            #[cfg(feature = "backend-context")]
            Self::WordContext { .. } => Ok(()),
            #[cfg(feature = "backend-ppmd")]
            Self::Ppmd { .. } => Ok(()),
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(_) => Ok(()),
            #[cfg(feature = "backend-ctw")]
            Self::FacCtw(_) => Ok(()),
            #[cfg(feature = "backend-zpaq")]
            Self::Zpaq(_) => Ok(()),
            #[cfg(feature = "backend-particle")]
            Self::Particle(_) => Ok(()),
            #[cfg(feature = "backend-sequitur")]
            Self::Sequitur { .. } => Ok(()),
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => m.compressor.finish_online_policy_stream(),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => m.finish_stream(),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                valid,
                native_prefix_progress,
                native_prediction,
                ..
            } => {
                *valid = false;
                *native_prefix_progress = None;
                *native_prediction = None;
                Ok(())
            }
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.finish_stream(),
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                bitwise,
                valid,
                ..
            } => {
                *bitwise = PredictorBitwiseStepState::default();
                *valid = false;
                base.finish_stream()
            }
            Self::Disabled { .. } => Ok(()),
        }
    }

    fn pdf_next(&mut self) -> Result<&[f64]> {
        match self {
            #[cfg(feature = "backend-rosa")]
            Self::Rosa(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-match")]
            Self::Match { model } => Ok(model.pdf()),
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-ctw")]
            Self::FacCtw(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                model, pdf, valid, ..
            } => {
                if !*valid {
                    let mut row = [0.0f64; 256];
                    model.fill_byte_pdf(&mut row, PDF_MIN);
                    pdf.copy_from_slice(&row);
                    *valid = true;
                }
                Ok(pdf)
            }
            #[cfg(feature = "backend-zpaq")]
            Self::Zpaq(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.ensure_pdf(),
            #[cfg(feature = "backend-particle")]
            Self::Particle(m) => Ok(m.pdf_next()),
            #[cfg(feature = "backend-match")]
            Self::SparseMatch { model } => Ok(model.pdf()),
            #[cfg(feature = "backend-context")]
            Self::OrderNGram { model } => Ok(model.pdf(PDF_MIN)),
            #[cfg(feature = "backend-context")]
            Self::WordContext { model } => Ok(model.pdf(PDF_MIN)),
            #[cfg(feature = "backend-ppmd")]
            Self::Ppmd { model } => Ok(model.pdf()),
            #[cfg(feature = "backend-sequitur")]
            Self::Sequitur { model } => Ok(model.pdf()),
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                core,
                bitwise: _,
                pdf,
                valid,
            } => {
                if !*valid {
                    let base_pdf = base.pdf_next()?;
                    core.apply_pdf(base_pdf, pdf);
                    normalize_pdf(pdf, PDF_MIN);
                    *valid = true;
                }
                Ok(pdf)
            }
            Self::Disabled { reason } => bail!("{reason}"),
        }
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        match self {
            #[cfg(feature = "backend-rosa")]
            Self::Rosa(m) => {
                m.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-match")]
            Self::Match { model } => {
                model.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-match")]
            Self::SparseMatch { model } => {
                model.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-context")]
            Self::OrderNGram { model } => {
                model.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-context")]
            Self::WordContext { model } => {
                model.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-ppmd")]
            Self::Ppmd { model } => {
                model.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-sequitur")]
            Self::Sequitur { model } => {
                model.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) => {
                m.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-ctw")]
            Self::FacCtw(m) => {
                m.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => m.update(symbol),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => m.update(symbol),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                model,
                valid,
                native_prefix_progress,
                native_prediction,
                ..
            } => {
                debug_assert!(
                    native_prefix_progress.is_none(),
                    "bit-reservoir symbol update while native bitwise byte step is active"
                );
                *native_prefix_progress = None;
                *native_prediction = None;
                model.update_byte(symbol, true);
                *valid = false;
                Ok(())
            }
            #[cfg(feature = "backend-zpaq")]
            Self::Zpaq(m) => {
                m.update(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.update(symbol),
            #[cfg(feature = "backend-particle")]
            Self::Particle(m) => {
                m.step(symbol);
                Ok(())
            }
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                core,
                bitwise: _,
                valid,
                ..
            } => {
                let base_pdf = base.pdf_next()?;
                core.observe_symbol_from_base_pdf(symbol, base_pdf)
                    .map_err(anyhow::Error::msg)?;
                base.update(symbol)?;
                *valid = false;
                Ok(())
            }
            Self::Disabled { reason } => bail!("{reason}"),
        }
    }

    fn prepare_cached_cdf_fast_bitwise(&mut self) -> Result<bool> {
        match self {
            #[cfg(feature = "backend-rosa")]
            Self::Rosa(m) => {
                let _ = m.cdf_next();
                Ok(true)
            }
            #[cfg(feature = "backend-match")]
            Self::Match { model } => {
                let _ = model.cdf();
                Ok(true)
            }
            #[cfg(feature = "backend-match")]
            Self::SparseMatch { model } => {
                let _ = model.cdf();
                Ok(true)
            }
            #[cfg(feature = "backend-ppmd")]
            Self::Ppmd { model } => {
                let _ = model.cdf();
                Ok(true)
            }
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => {
                let _ = m.cdf_next();
                Ok(true)
            }
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => {
                let _ = m.cdf_next();
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn cached_cdf_bit_prob_one_msb(&mut self, range: MsbPrefixRange) -> Option<f64> {
        match self {
            #[cfg(feature = "backend-rosa")]
            Self::Rosa(m) => Some(range.prob_one(&m.cdf, PDF_MIN)),
            #[cfg(feature = "backend-match")]
            Self::Match { model } => Some(range.prob_one(model.cdf(), PDF_MIN)),
            #[cfg(feature = "backend-match")]
            Self::SparseMatch { model } => Some(range.prob_one(model.cdf(), PDF_MIN)),
            #[cfg(feature = "backend-ppmd")]
            Self::Ppmd { model } => Some(range.prob_one(model.cdf(), PDF_MIN)),
            #[cfg(feature = "backend-mamba")]
            Self::Mamba(m) => Some(range.prob_one(m.cdf_next(), PDF_MIN)),
            #[cfg(feature = "backend-rwkv")]
            Self::Rwkv(m) => Some(range.prob_one(m.cdf_next(), PDF_MIN)),
            _ => None,
        }
    }

    #[inline]
    fn has_recursive_native_bitwise_path(&self) -> bool {
        match self {
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) | Self::FacCtw(m) => m.can_fast_ac_bitwise(),
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.has_recursive_native_bitwise_expert(),
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated { .. } => true,
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir { .. } => true,
            _ => false,
        }
    }

    fn begin_native_recursive_bitwise_byte_step(&mut self) -> Result<bool> {
        match self {
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) | Self::FacCtw(m) => Ok(m.can_fast_ac_bitwise()),
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.begin_bitwise_byte_step(),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                native_prefix_progress,
                native_prediction,
                valid,
                ..
            } => {
                if native_prefix_progress.is_some() {
                    bail!("native recursive bitwise byte step is already active");
                }
                *native_prefix_progress = Some(0);
                *native_prediction = None;
                *valid = false;
                Ok(true)
            }
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                core.begin_byte().map_err(anyhow::Error::msg)?;
                if let Err(err) = bitwise.prepare(base) {
                    let _ = core.abort_empty_byte();
                    return Err(err);
                }
                *valid = false;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn native_recursive_bit_prob_one_msb(&mut self, bit_idx: usize) -> Result<f64> {
        match self {
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) | Self::FacCtw(m) => Ok(m.bit_prob_one_msb(bit_idx)),
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.bit_prob_one_msb(bit_idx),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                model,
                native_prefix_progress,
                native_prediction,
                ..
            } => {
                validate_bit_reservoir_prefix_index(*native_prefix_progress, bit_idx)?;
                let prediction = model.predict_for_training();
                let p1 = prediction.prob_one().clamp(PDF_MIN, 1.0 - PDF_MIN);
                *native_prediction = Some((bit_idx, prediction));
                Ok(p1)
            }
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                core,
                bitwise,
                ..
            } => {
                let base_p1: f64 = bitwise.bit_prob_one_msb(base, bit_idx)?;
                debug_assert!(core.byte_is_active());
                Ok(core.predict_bit_unchecked(base_p1))
            }
            _ => bail!("native recursive bitwise stepping is unavailable for this predictor"),
        }
    }

    fn native_recursive_observe_bit_msb(&mut self, bit_idx: usize, bit: bool) -> Result<()> {
        match self {
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) | Self::FacCtw(m) => {
                m.update_bit_msb(bit_idx, bit);
                Ok(())
            }
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.observe_bit_msb(bit_idx, bit),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                model,
                native_prefix_progress,
                valid,
                native_prediction,
                ..
            } => {
                validate_bit_reservoir_prefix_index(*native_prefix_progress, bit_idx)?;
                if let Some(next_bit_idx) = native_prefix_progress {
                    *next_bit_idx += 1;
                }
                match native_prediction.take() {
                    Some((cached_bit_idx, prediction)) if cached_bit_idx == bit_idx => {
                        model.observe_bit_with_prediction(bit, &prediction);
                    }
                    _ => model.observe_bit(bit, true),
                }
                *valid = false;
                Ok(())
            }
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                debug_assert!(core.byte_is_active());
                core.observe_bit_unchecked(bit);
                bitwise.observe_bit_msb(base, bit_idx, bit)?;
                *valid = false;
                Ok(())
            }
            _ => bail!("native recursive bitwise stepping is unavailable for this predictor"),
        }
    }

    fn native_recursive_observe_known_bit_msb(&mut self, bit_idx: usize, bit: bool) -> Result<f64> {
        match self {
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(m) | Self::FacCtw(m) => {
                let p1 = m.bit_prob_one_msb(bit_idx);
                m.update_bit_msb(bit_idx, bit);
                Ok(p1)
            }
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.observe_known_bit_msb(bit_idx, bit),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                model,
                native_prefix_progress,
                valid,
                native_prediction,
                ..
            } => {
                validate_bit_reservoir_prefix_index(*native_prefix_progress, bit_idx)?;
                if let Some(next_bit_idx) = native_prefix_progress {
                    *next_bit_idx += 1;
                }
                let prediction = model.predict_for_training();
                let p1 = prediction.prob_one().clamp(PDF_MIN, 1.0 - PDF_MIN);
                model.observe_bit_with_prediction(bit, &prediction);
                *native_prediction = None;
                *valid = false;
                Ok(p1)
            }
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                debug_assert!(core.byte_is_active());
                let base_p1: f64 = bitwise.observe_known_bit_msb(base, bit_idx, bit)?;
                let p1 = core.predict_bit_unchecked(base_p1);
                core.observe_bit_unchecked(bit);
                *valid = false;
                Ok(p1)
            }
            _ => bail!("native recursive bitwise stepping is unavailable for this predictor"),
        }
    }

    fn finish_native_recursive_bitwise_byte_step(&mut self, symbol: u8) -> Result<()> {
        match self {
            #[cfg(feature = "backend-ctw")]
            Self::Ctw(_) | Self::FacCtw(_) => Ok(()),
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.finish_bitwise_symbol(symbol),
            #[cfg(feature = "backend-bit-reservoir")]
            Self::BitReservoir {
                native_prefix_progress,
                native_prediction,
                ..
            } => match native_prefix_progress {
                Some(8) => {
                    *native_prefix_progress = None;
                    *native_prediction = None;
                    let _ = symbol;
                    Ok(())
                }
                Some(bits) => {
                    bail!("native recursive bitwise finish requires 8 observed bits, got {bits}")
                }
                None => bail!("native recursive bitwise byte step is not active"),
            },
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated {
                base,
                core,
                bitwise,
                valid,
                ..
            } => {
                core.validate_complete_byte().map_err(anyhow::Error::msg)?;
                bitwise.finish_symbol(base, symbol)?;
                core.finish_byte().map_err(anyhow::Error::msg)?;
                *valid = false;
                Ok(())
            }
            _ => bail!("native recursive bitwise stepping is unavailable for this predictor"),
        }
    }

    #[inline]
    fn can_fast_ac_bitwise(&self) -> bool {
        self.has_recursive_native_bitwise_path()
    }

    // Keep this separate from the live AC payload path so framed AC preserves
    // the v1 wire contract while still allowing backend-agnostic bitwise
    // stepping as an internal utility.
    //
    // Takes the `PredictorBitwiseStepState` scratch as a parameter so hot
    // streaming loops (encode/decode over an entire payload) can reuse one
    // allocation across all symbols instead of paying a fresh
    // `Box<BytePrefixCdf>` allocation (2056 bytes) per byte, most of which is
    // immediately discarded by `prepare()` whenever the backend takes the
    // `NativeRecursive`/`CachedCdf` fast paths (e.g. CTW, mixtures containing
    // a native-bitwise expert, Calibrated).
    fn ac_step_bitwise_with_state<F>(
        &mut self,
        state: &mut PredictorBitwiseStepState,
        mut choose_bit: F,
    ) -> Result<u8>
    where
        F: FnMut(usize, f64) -> Result<u8>,
    {
        state.prepare(self)?;
        let mut symbol = 0u8;
        for bit_idx in 0..8usize {
            let p1 = state.bit_prob_one_msb(self, bit_idx)?;
            let bit = choose_bit(bit_idx, p1)? & 1;
            if bit == 1 {
                symbol |= 1u8 << (7 - bit_idx);
            }
            state.observe_bit_msb(self, bit_idx, bit == 1)?;
        }
        state.finish_symbol(self, symbol)?;
        Ok(symbol)
    }

    #[cfg(test)]
    fn ac_step_bitwise<F>(&mut self, choose_bit: F) -> Result<u8>
    where
        F: FnMut(usize, f64) -> Result<u8>,
    {
        let mut state = PredictorBitwiseStepState::default();
        self.ac_step_bitwise_with_state(&mut state, choose_bit)
    }

    fn ac_step_fast_bitwise_with_state<F>(
        &mut self,
        state: &mut PredictorBitwiseStepState,
        choose_bit: F,
    ) -> Result<u8>
    where
        F: FnMut(usize, f64) -> Result<u8>,
    {
        debug_assert!(self.can_fast_ac_bitwise());
        self.ac_step_bitwise_with_state(state, choose_bit)
    }

    fn encode_known_symbol_ac_fast_bitwise_with_state(
        &mut self,
        state: &mut PredictorBitwiseStepState,
        symbol: u8,
        encoder: &mut ArithmeticEncoder<&mut Vec<u8>>,
    ) -> Result<()> {
        debug_assert!(self.can_fast_ac_bitwise());
        state.prepare(self)?;
        for bit_idx in 0..8usize {
            let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
            let p1_mix = state.observe_known_bit_msb(self, bit_idx, bit)?;
            let split = binary_split_from_prob_one(p1_mix);
            if bit {
                encoder.encode_counts(split as u64, CDF_TOTAL as u64, CDF_TOTAL as u64)?;
            } else {
                encoder.encode_counts(0, split as u64, CDF_TOTAL as u64)?;
            }
        }
        state.finish_symbol(self, symbol)
    }

    fn diagnostic_snapshot_subtree(
        &mut self,
        symbol: u8,
        local_weight: f64,
        effective_weight: f64,
        pool: Option<&ThreadPool>,
    ) -> Result<AcLogLossSubtreeSnapshot> {
        match self {
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => {
                m.diagnostic_subtree_snapshot(symbol, local_weight, effective_weight, pool)
            }
            _ => {
                let prob = self.pdf_next()?[symbol as usize].max(PDF_MIN);
                Ok(AcLogLossSubtreeSnapshot {
                    prob,
                    rows: vec![AcLogLossNodeValue {
                        prob,
                        local_weight,
                        effective_weight,
                    }],
                })
            }
        }
    }

    // The mixture implementation mutates the Vec allocation; no-mixture builds
    // only see this forwarding signature and would otherwise flag it as `ptr_arg`.
    #[allow(clippy::ptr_arg)]
    fn diagnostic_root_snapshot(
        &mut self,
        symbol: u8,
        pool: Option<&ThreadPool>,
        out: &mut Vec<AcLogLossNodeValue>,
    ) -> Result<AcLogLossRootSnapshot> {
        match self {
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.diagnostic_root_snapshot(symbol, pool, out),
            _ => anyhow::bail!("AC log-loss diagnostics require a top-level mixture backend"),
        }
    }

    fn encode_symbol_ac_step<W: std::io::Write>(
        &mut self,
        bitwise_state: &mut PredictorBitwiseStepState,
        symbol: u8,
        encoder: &mut ArithmeticEncoder<W>,
        cdf: &mut [u32; 257],
    ) -> Result<()> {
        if self.can_fast_ac_bitwise() {
            self.ac_step_fast_bitwise_with_state(bitwise_state, |bit_idx, p1_mix| {
                let bit = (symbol >> (7 - bit_idx)) & 1;
                let split = binary_split_from_prob_one(p1_mix);
                if bit == 0 {
                    encoder.encode_counts(0, split as u64, CDF_TOTAL as u64)?;
                } else {
                    encoder.encode_counts(split as u64, CDF_TOTAL as u64, CDF_TOTAL as u64)?;
                }
                Ok(bit)
            })?;
            return Ok(());
        }

        let pdf = self.pdf_next()?;
        crate::coders::quantize_pdf_to_integer_cdf_dense_positive_with_buffer(
            pdf,
            CDF_TOTAL,
            cdf.as_mut_slice(),
        );
        let sym = symbol as usize;
        encoder.encode_counts(cdf[sym] as u64, cdf[sym + 1] as u64, CDF_TOTAL as u64)?;
        self.update(symbol)
    }

    fn logistic_match_state(&mut self) -> LogisticMatchState {
        match self {
            #[cfg(feature = "backend-match")]
            Self::Match { model } => LogisticMatchState {
                len_bucket: bucket_repeat_len(model.match_len()),
                predicted_class: model.predicted_byte().map(classify_byte).unwrap_or(0),
            },
            #[cfg(feature = "backend-match")]
            Self::SparseMatch { model } => LogisticMatchState {
                len_bucket: bucket_repeat_len(model.match_len()),
                predicted_class: model.predicted_byte().map(classify_byte).unwrap_or(0),
            },
            #[cfg(feature = "backend-mixture")]
            Self::Mixture(m) => m.logistic_match_state(),
            #[cfg(feature = "backend-calibrated")]
            Self::Calibrated { base, .. } => base.logistic_match_state(),
            _ => LogisticMatchState::default(),
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

#[cfg(feature = "backend-bit-reservoir")]
fn validate_bit_reservoir_prefix_index(next_bit_idx: Option<usize>, bit_idx: usize) -> Result<()> {
    let Some(expected) = next_bit_idx else {
        bail!("native recursive bitwise byte step is not active");
    };
    if bit_idx >= 8 {
        bail!("native recursive bitwise bit index {bit_idx} is out of range; expected 0..8");
    }
    if bit_idx != expected {
        bail!(
            "native recursive bitwise bit index {bit_idx} violated sequential stepping; expected {expected}"
        );
    }
    Ok(())
}

/// This function should be considered when fine-tuning Compression/decompression for a particular runtime case. In particular, my benchmarking has shown that inlining is non-obvious in how it affects performance
/// Inlining both encode and decode seems to cause performance issues with Match+AC decompression specifically, hence the odd configuration here for balance.
/// Encode default: inline
/// Technical note: this fast-path preserves the same bit ordering and CDF split mapping as the generic AC path (MSB-first with `binary_split_from_prob_one`).
#[cfg_attr(not(infotheory_ac_encode_deinline), inline(always))]
#[cfg_attr(infotheory_ac_encode_deinline, inline(never))]
fn encode_payload_ac_fast_bitwise(
    data: &[u8],
    predictor: &mut RatePdfPredictor,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = ArithmeticEncoder::new(&mut out);
        let mut state = PredictorBitwiseStepState::default();
        for &symbol in data {
            predictor
                .encode_known_symbol_ac_fast_bitwise_with_state(&mut state, symbol, &mut enc)?;
        }
        let _ = enc.finish()?;
    }
    Ok(out)
}

/// This function should be considered when fine-tuning Compression/decompression for a particular runtime case. In particular, my benchmarking has shown that inlining is non-obvious in how it affects performance
/// Inlining both encode and decode seems to cause performance issues with Match+AC decompression specifically, hence the odd configuration here for balance.
/// Decode default: deinline
/// Technical note: this decodes exactly `out_len` symbols from the same binary CDF domain (`CDF_TOTAL`) used by the paired encode fast-path.
#[cfg_attr(infotheory_ac_decode_inline, inline(always))]
#[cfg_attr(not(infotheory_ac_decode_inline), inline(never))]
fn decode_payload_ac_fast_bitwise(
    payload: &[u8],
    out_len: usize,
    predictor: &mut RatePdfPredictor,
) -> Result<Vec<u8>> {
    let mut dec = ArithmeticDecoder::new(payload)?;
    let mut out = Vec::with_capacity(out_len);
    let mut state = PredictorBitwiseStepState::default();
    for _ in 0..out_len {
        let symbol = predictor.ac_step_fast_bitwise_with_state(&mut state, |_, p1_mix| {
            let split = binary_split_from_prob_one(p1_mix);
            dec.decode_binary_counts(split, CDF_TOTAL)
        })?;
        out.push(symbol);
    }
    Ok(out)
}

fn encode_payload_ac(data: &[u8], predictor: &mut RatePdfPredictor) -> Result<Vec<u8>> {
    predictor.begin_stream(data.len())?;

    if predictor.can_fast_ac_bitwise() {
        let out = encode_payload_ac_fast_bitwise(data, predictor)?;
        predictor.finish_stream()?;
        return Ok(out);
    }

    let mut out = Vec::new();
    {
        let mut enc = ArithmeticEncoder::new(&mut out);
        // Reuse one CDF scratch buffer for the full stream to avoid per-symbol allocation.
        let mut cdf = [0u32; 257];
        for &symbol in data {
            let pdf = predictor.pdf_next()?;
            crate::coders::quantize_pdf_to_integer_cdf_dense_positive_with_buffer(
                pdf,
                CDF_TOTAL,
                cdf.as_mut_slice(),
            );
            let sym = symbol as usize;
            enc.encode_counts(cdf[sym] as u64, cdf[sym + 1] as u64, CDF_TOTAL as u64)?;
            predictor.update(symbol)?;
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
        let out = decode_payload_ac_fast_bitwise(payload, out_len, predictor)?;
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
    rate_backend: &CompiledRateBackend,
    coder: CoderType,
    framing: FramingMode,
) -> Result<Vec<u8>> {
    let mut predictor = crate::runtime::build_rate_pdf_predictor(rate_backend)?;
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
    rate_backend: &CompiledRateBackend,
    coder: CoderType,
    framing: FramingMode,
) -> Result<u64> {
    let encoded = compress_rate_bytes(data, rate_backend, coder, framing)?;
    Ok(encoded.len() as u64)
}

/// Return compressed size (in bytes) for concatenated slices under one stream.
pub fn compress_rate_size_chain(
    parts: &[&[u8]],
    rate_backend: &CompiledRateBackend,
    coder: CoderType,
    framing: FramingMode,
) -> Result<u64> {
    let total = parts.iter().map(|p| p.len()).sum();
    let mut data = Vec::with_capacity(total);
    for p in parts {
        data.extend_from_slice(p);
    }
    compress_rate_size(&data, rate_backend, coder, framing)
}

/// Decompress bytes produced by [`compress_rate_bytes`].
pub fn decompress_rate_bytes(
    input: &[u8],
    rate_backend: &CompiledRateBackend,
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
    let mut predictor = crate::runtime::build_rate_pdf_predictor(rate_backend)?;
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

fn normalize_pdf_vec_and_maybe_build_cdf(pdf: &mut [f64], cdf: Option<&mut [f64; 257]>) {
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
        if let Some(cdf) = cdf {
            *cdf = uniform_cdf_row();
        }
        return;
    }
    let inv = 1.0 / sum;
    if let Some(cdf) = cdf {
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

#[allow(dead_code)]
#[cfg(feature = "backend-zpaq")]
fn _zpaq_marker(_: &ZpaqRateModel) {}

#[cfg(all(test, feature = "all-backends"))]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn compiled_rate_backend(backend: &RateBackend) -> CompiledRateBackend {
        backend
            .compile()
            .unwrap_or_else(|err| panic!("failed to compile rate backend for test: {err}"))
    }

    fn compress_rate_bytes(
        data: &[u8],
        rate_backend: &RateBackend,
        coder: CoderType,
        framing: FramingMode,
    ) -> Result<Vec<u8>> {
        super::compress_rate_bytes(data, &compiled_rate_backend(rate_backend), coder, framing)
    }

    fn compress_rate_size(
        data: &[u8],
        rate_backend: &RateBackend,
        coder: CoderType,
        framing: FramingMode,
    ) -> Result<u64> {
        super::compress_rate_size(data, &compiled_rate_backend(rate_backend), coder, framing)
    }

    fn decompress_rate_bytes(
        input: &[u8],
        rate_backend: &RateBackend,
        coder: CoderType,
        framing: FramingMode,
    ) -> Result<Vec<u8>> {
        super::decompress_rate_bytes(input, &compiled_rate_backend(rate_backend), coder, framing)
    }

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

        normalize_pdf(&mut out, PDF_MIN);
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
        let mut predictor = CtwPredictor::new_fac(5, 5, None);
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
    fn fac_ctw_default_bit_order_is_byte_msb_and_subbyte_lsb() {
        let byte_default = CtwPredictor::new_fac(5, 8, None);
        assert!(
            byte_default.can_fast_ac_bitwise(),
            "8-bit FacCtw without explicit order should use MSB-first native bitwise path"
        );

        let subbyte_default = CtwPredictor::new_fac(5, 5, None);
        assert!(
            !subbyte_default.can_fast_ac_bitwise(),
            "non-byte FacCtw without explicit order keeps legacy LSB-first behavior"
        );

        let explicit_lsb = CtwPredictor::new_fac(5, 8, Some(false));
        assert!(
            !explicit_lsb.can_fast_ac_bitwise(),
            "explicit msb_first=false must preserve legacy LSB-first behavior"
        );

        let explicit_subbyte_msb = CtwPredictor::new_fac(5, 5, Some(true));
        assert!(
            !explicit_subbyte_msb.can_fast_ac_bitwise(),
            "subbyte widths do not use byte-packed native fast path even when MSB-first"
        );
    }

    fn assert_ctw_pdf_next_preserves_state(mut predictor: CtwPredictor) {
        for &b in b"ctw predictor state preservation payload" {
            predictor.update(b);
        }
        let mut baseline = [0.0f64; 256];
        for (sym, slot) in baseline.iter_mut().enumerate() {
            *slot = predictor.log_prob_symbol_bruteforce(sym as u8);
        }
        let _ = predictor.pdf_next();
        for (sym, &expected) in baseline.iter().enumerate() {
            let after = predictor.log_prob_symbol_bruteforce(sym as u8);
            assert!(
                (expected - after).abs() < 1e-12,
                "symbol {sym} drift: {expected} vs {after}"
            );
        }
    }

    #[test]
    fn ctw_pdf_next_preserves_state() {
        assert_ctw_pdf_next_preserves_state(CtwPredictor::new_ctw(7));
    }

    #[test]
    fn fac_pdf_next_preserves_state() {
        assert_ctw_pdf_next_preserves_state(CtwPredictor::new_fac(7, 8, None));
    }

    fn assert_fill_pattern_preserves_symbol_log_probs(mut predictor: CtwPredictor) {
        for &b in b"fill-pattern preservation regression payload" {
            predictor.update(b);
        }
        let mut baseline = [0.0f64; 256];
        for (sym, slot) in baseline.iter_mut().enumerate() {
            *slot = predictor.log_prob_symbol_bruteforce(sym as u8);
        }
        let _ = predictor.fill_pattern_log_probs();
        for (sym, &expected) in baseline.iter().enumerate() {
            let got = predictor.log_prob_symbol_bruteforce(sym as u8);
            let diff = (expected - got).abs();
            assert!(
                diff < 1e-12,
                "symbol={sym} expected={expected} got={got} diff={diff}"
            );
        }
    }

    #[test]
    fn ctw_fill_pattern_preserves_symbol_log_probs() {
        assert_fill_pattern_preserves_symbol_log_probs(CtwPredictor::new_ctw(7));
    }

    #[test]
    fn fac_fill_pattern_preserves_symbol_log_probs() {
        assert_fill_pattern_preserves_symbol_log_probs(CtwPredictor::new_fac(7, 8, None));
    }

    fn assert_pdf_then_update_matches_plain_update(mut base: CtwPredictor) {
        for &b in b"pdf then update parity payload" {
            base.update(b);
        }
        let observed = b'n';
        let mut with_pdf = base.clone();
        let mut plain = base;

        let _ = with_pdf.pdf_next();
        with_pdf.update(observed);
        plain.update(observed);

        for sym in 0u8..=255u8 {
            let lp_with_pdf = with_pdf.log_prob_symbol_bruteforce(sym);
            let lp_plain = plain.log_prob_symbol_bruteforce(sym);
            let diff = (lp_with_pdf - lp_plain).abs();
            assert!(
                diff < 1e-12,
                "symbol={sym} with_pdf={lp_with_pdf} plain={lp_plain} diff={diff}"
            );
        }
    }

    #[test]
    fn ctw_pdf_then_update_matches_plain_update() {
        assert_pdf_then_update_matches_plain_update(CtwPredictor::new_ctw(7));
    }

    #[test]
    fn fac_pdf_then_update_matches_plain_update() {
        assert_pdf_then_update_matches_plain_update(CtwPredictor::new_fac(7, 8, None));
    }

    #[test]
    fn roundtrip_rate_ac_ctw() {
        let data = b"ctw backend roundtrip payload";
        let backend = RateBackend::Ctw { depth: 8 };
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
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
            RateBackend::OrderNGram {
                order: 2,
                hash_bits: 12,
            },
            RateBackend::WordContext { hash_bits: 12 },
            RateBackend::Ppmd {
                order: 8,
                memory_mb: 8,
            },
        ] {
            let enc =
                compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
            let dec =
                decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
            assert_eq!(dec, data);
        }
    }

    #[test]
    fn framed_rate_ac_keeps_v1_coder_byte_for_ctw() {
        let data = b"legacy framed ac header payload";
        let backend = RateBackend::Ctw { depth: 8 };
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let hdr = FramedHeader::read(&enc).expect("framed header");
        assert_eq!(hdr.coder_type(), CoderType::AC);
        assert_eq!(hdr.coder, 0);
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn framed_rate_rans_keeps_v1_coder_byte() {
        let data = b"legacy framed rans header payload";
        let backend = RateBackend::Ctw { depth: 8 };
        let enc =
            compress_rate_bytes(data, &backend, CoderType::RANS, FramingMode::Framed).unwrap();
        let hdr = FramedHeader::read(&enc).expect("framed header");
        assert_eq!(hdr.coder_type(), CoderType::RANS);
        assert_eq!(hdr.coder, 1);
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::RANS, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn framed_rate_ac_keeps_byte_prefix_models_on_legacy_path() {
        let data = b"byte prefix adapter exists but byte ac remains default";
        let backend = RateBackend::Match {
            hash_bits: 20,
            min_len: 4,
            max_len: 255,
            base_mix: 0.02,
            confidence_scale: 1.0,
        };
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let hdr = FramedHeader::read(&enc).expect("framed header");
        assert_eq!(hdr.coder_type(), CoderType::AC);
        assert_eq!(hdr.coder, 0);
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_ppmd_high_order_text_payload() {
        let seed = include_bytes!("../../../../README.md");
        let mut data = Vec::with_capacity(4096);
        while data.len() < 4096 {
            data.extend_from_slice(seed);
        }
        data.truncate(4096);

        let backend = RateBackend::Ppmd {
            order: 12,
            memory_mb: 256,
        };
        let enc = compress_rate_bytes(&data, &backend, CoderType::AC, FramingMode::Framed)
            .expect("ppmd high-order compression");
        let dec = decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed)
            .expect("ppmd high-order decompression");
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_calibrated_backend() {
        let data = b"calibration wrapper payload calibration wrapper payload";
        let backend = RateBackend::Calibrated {
            spec: Arc::new(crate::CalibratedSpec::new(
                RateBackend::Ctw { depth: 8 },
                crate::CalibrationContextKind::Text,
            )),
        };
        let predictor = RatePdfPredictor::from_rate_backend(backend.clone()).unwrap();
        assert!(
            predictor.can_fast_ac_bitwise(),
            "calibrated CTW should expose the SSE bitwise AC path"
        );
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_calibrated_byte_pdf_base_backend() {
        let data = b"calibrated byte pdf base payload calibrated byte pdf base payload";
        let backend = RateBackend::Calibrated {
            spec: Arc::new(crate::CalibratedSpec::new(
                RateBackend::Match {
                    hash_bits: 18,
                    min_len: 3,
                    max_len: 64,
                    base_mix: 0.08,
                    confidence_scale: 1.0,
                },
                crate::CalibrationContextKind::ByteClass,
            )),
        };
        let predictor = RatePdfPredictor::from_rate_backend(backend.clone()).unwrap();
        assert!(
            predictor.can_fast_ac_bitwise(),
            "calibrated byte-PDF bases should use the generic SSE bitwise adapter"
        );
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
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
                backend: RateBackend::Ctw { depth: 8 },
            }],
        )
        .with_alpha(0.03);
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_logistic_mixture() {
        let data = b"logistic mixture ac roundtrip payload with repeated repeated words";
        let spec = MixtureSpec::new(
            MixtureKind::Logistic,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Ctw { depth: 7 },
                },
                crate::MixtureExpertSpec {
                    name: Some("match".to_string()),
                    log_prior: -0.2,
                    backend: RateBackend::Match {
                        hash_bits: 18,
                        min_len: 3,
                        max_len: 64,
                        base_mix: 0.03,
                        confidence_scale: 1.0,
                    },
                },
            ],
        )
        .with_alpha(0.03);
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let predictor = RatePdfPredictor::from_rate_backend(backend.clone()).unwrap();
        assert!(
            predictor.can_fast_ac_bitwise(),
            "logistic mixtures should use their bitwise native path for AC"
        );
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_single_expert_ctw_logistic_mixture() {
        let data = b"single expert logistic ctw fast path payload";
        let base = RateBackend::Ctw { depth: 8 };
        let spec = MixtureSpec::new(
            MixtureKind::Logistic,
            vec![crate::MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                backend: base.clone(),
            }],
        )
        .with_alpha(0.03);
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let mix_predictor = RatePdfPredictor::from_rate_backend(backend.clone()).unwrap();
        let base_predictor = RatePdfPredictor::from_rate_backend(base.clone()).unwrap();
        assert!(
            mix_predictor.can_fast_ac_bitwise(),
            "one-expert logistic over CTW should expose the expert's native AC path"
        );
        assert_eq!(
            mix_predictor.can_fast_ac_bitwise(),
            base_predictor.can_fast_ac_bitwise(),
            "one-expert logistic AC eligibility must match the sole expert"
        );
        let enc_mix =
            compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let enc_base =
            compress_rate_bytes(data, &base, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(
            enc_mix, enc_base,
            "one-expert logistic AC bitstream must match the sole expert (no LogisticMixCore train)"
        );
        let dec =
            decompress_rate_bytes(&enc_mix, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);

        let enc_rans =
            compress_rate_bytes(data, &backend, CoderType::RANS, FramingMode::Framed).unwrap();
        let dec_rans =
            decompress_rate_bytes(&enc_rans, &backend, CoderType::RANS, FramingMode::Framed)
                .unwrap();
        assert_eq!(dec_rans, data);
    }

    #[test]
    fn single_expert_logistic_match_does_not_claim_logistic_ac_fast_path() {
        let backend = RateBackend::Mixture {
            spec: Arc::new(
                MixtureSpec::new(
                    MixtureKind::Logistic,
                    vec![crate::MixtureExpertSpec {
                        name: Some("match".to_string()),
                        log_prior: 0.0,
                        backend: RateBackend::Match {
                            hash_bits: 18,
                            min_len: 3,
                            max_len: 64,
                            base_mix: 0.03,
                            confidence_scale: 1.0,
                        },
                    }],
                )
                .with_alpha(0.03),
            ),
        };
        let predictor = RatePdfPredictor::from_rate_backend(backend).unwrap();
        assert!(
            !predictor.can_fast_ac_bitwise(),
            "one-expert logistic over a non-bitwise expert must not enable the stretch-mixer AC path"
        );
    }

    #[test]
    fn roundtrip_rate_ac_single_expert_ctw_bayes_mixture() {
        let data = b"single expert bayes ctw fast path payload";
        let spec = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![crate::MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                backend: RateBackend::Ctw { depth: 8 },
            }],
        )
        .with_alpha(0.03);
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
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
                    backend: RateBackend::Ctw { depth: 6 },
                },
                crate::MixtureExpertSpec {
                    name: Some("fac".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::FacCtw {
                        base_depth: 6,
                        num_percept_bits: 8,
                        encoding_bits: 8,
                        msb_first: None,
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
                    backend: RateBackend::Mixture {
                        spec: Arc::new(nested),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("zpaq".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Zpaq {
                        method: crate::api::ZpaqMethodSpec::literal("1"),
                    },
                },
            ],
        )
        .with_alpha(0.05);

        let backend = RateBackend::Mixture {
            spec: Arc::new(root),
        };
        let enc =
            compress_rate_bytes(data, &backend, CoderType::RANS, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::RANS, FramingMode::Framed).unwrap();
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
                    backend: RateBackend::Ctw { depth: 6 },
                },
                crate::MixtureExpertSpec {
                    name: Some("fac".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::FacCtw {
                        base_depth: 6,
                        num_percept_bits: 8,
                        encoding_bits: 8,
                        msb_first: None,
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
                    backend: RateBackend::Mixture {
                        spec: Arc::new(inner),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("zpaq".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Zpaq {
                        method: crate::api::ZpaqMethodSpec::literal("1"),
                    },
                },
            ],
        )
        .with_alpha(0.03);

        let backend = RateBackend::Mixture {
            spec: Arc::new(root),
        };
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_rate_ac_recursive_native_bitwise_mixture() {
        let data = b"recursive native bitwise mixture payload";
        let backend = recursive_native_bitwise_backend();
        let predictor = RatePdfPredictor::from_rate_backend(backend.clone()).unwrap();
        assert!(predictor.can_fast_ac_bitwise());

        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    fn assert_runtime_and_compression_predictor_align(spec: MixtureSpec, data: &[u8], tol: f64) {
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec.clone()),
        };
        let mut predictor = RatePdfPredictor::from_rate_backend(backend).unwrap();
        let experts = spec.build_experts();
        let mut runtime = crate::mixture::build_mixture_runtime(&spec, &experts).unwrap();

        for (t, &symbol) in data.iter().enumerate() {
            let pdf = predictor.pdf_next().unwrap();
            let p_comp = pdf[symbol as usize];
            let p_runtime = runtime.peek_log_prob(symbol).exp();
            assert!(
                (p_comp - p_runtime).abs() < tol,
                "t={t} p_comp={p_comp} p_runtime={p_runtime} symbol={symbol}"
            );
            predictor.update(symbol).unwrap();
            runtime.step(symbol);
        }
    }

    fn alignment_experts() -> Vec<crate::MixtureExpertSpec> {
        vec![
            crate::MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                backend: RateBackend::Ctw { depth: 7 },
            },
            crate::MixtureExpertSpec {
                name: Some("fac".to_string()),
                log_prior: -0.7,
                backend: RateBackend::FacCtw {
                    base_depth: 7,
                    num_percept_bits: 8,
                    encoding_bits: 8,
                    msb_first: None,
                },
            },
        ]
    }

    #[test]
    fn bayes_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::Bayes, alignment_experts());
        assert_runtime_and_compression_predictor_align(
            spec,
            b"bayes predictor alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn fading_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::FadingBayes, alignment_experts()).with_decay(0.97);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"fading predictor alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn switching_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::Switching, alignment_experts()).with_alpha(0.17);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"switching predictor alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn switching_theorem_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::Switching, alignment_experts())
            .with_schedule(MixtureScheduleMode::Theorem)
            .with_alpha(0.91);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"switching theorem predictor alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn convex_runtime_and_compression_predictor_align_for_alpha_above_one() {
        let spec = MixtureSpec::new(MixtureKind::Convex, alignment_experts()).with_alpha(1.25);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"convex predictor alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn convex_theorem_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::Convex, alignment_experts())
            .with_schedule(MixtureScheduleMode::Theorem)
            .with_alpha(7.5);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"convex theorem predictor alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn neural_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::Neural, alignment_experts()).with_alpha(0.03);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"neural alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn logistic_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::Logistic, alignment_experts()).with_alpha(0.03);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"logistic alignment check sequence",
            1e-8,
        );
    }

    #[test]
    fn logistic_diagnostics_are_rejected() {
        let spec = MixtureSpec::new(MixtureKind::Logistic, alignment_experts()).with_alpha(0.03);
        let compiled = RateBackend::Mixture {
            spec: Arc::new(spec),
        }
        .compile()
        .expect("compile logistic mixture");
        let mut predictor =
            DiagnosticRatePredictor::from_compiled(&compiled).expect("build diagnostic predictor");
        predictor.begin_stream(1).expect("begin stream");
        let mut rows = Vec::new();
        let err = predictor
            .diagnostic_root_snapshot(b'a', None, &mut rows)
            .expect_err("logistic ac-log-loss diagnostics must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("not supported for logistic") || msg.contains("stretch-domain"),
            "unexpected diagnostic rejection message: {msg}"
        );
    }

    #[test]
    fn non_logistic_mixture_does_not_allocate_logistic_tables() {
        let spec = MixtureSpec::new(MixtureKind::Bayes, alignment_experts());
        let compiled = RateBackend::Mixture {
            spec: Arc::new(spec),
        }
        .compile()
        .expect("compile bayes mixture");
        let predictor =
            RatePdfPredictor::from_compiled(&compiled).expect("build compression predictor");
        match predictor {
            RatePdfPredictor::Mixture(m) => {
                assert!(
                    !m.has_logistic_mixer(),
                    "Bayes compression mixture must not allocate LogisticMixCore tables"
                );
                assert!(
                    !m.has_neural_mixer(),
                    "Bayes compression mixture must not allocate NeuralMixCore tables"
                );
            }
            _ => panic!("expected mixture predictor"),
        }
    }

    #[test]
    fn mdl_runtime_and_compression_predictor_align() {
        let spec = MixtureSpec::new(MixtureKind::Mdl, alignment_experts());
        assert_runtime_and_compression_predictor_align(spec, b"mdl alignment check sequence", 1e-8);
    }

    #[test]
    fn nested_runtime_and_compression_predictor_align() {
        let nested = MixtureSpec::new(MixtureKind::Bayes, alignment_experts());
        let spec = MixtureSpec::new(
            MixtureKind::Switching,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("nested".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Mixture {
                        spec: Arc::new(nested),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("ppmd".to_string()),
                    log_prior: -0.2,
                    backend: RateBackend::Ppmd {
                        order: 5,
                        memory_mb: 8,
                    },
                },
            ],
        )
        .with_alpha(0.13);
        assert_runtime_and_compression_predictor_align(
            spec,
            b"nested mixture predictor alignment check sequence",
            1e-8,
        );
    }

    fn recursive_native_bitwise_backend() -> RateBackend {
        let nested = MixtureSpec::new(MixtureKind::Bayes, alignment_experts());
        let root = MixtureSpec::new(
            MixtureKind::Switching,
            vec![
                crate::MixtureExpertSpec {
                    name: Some("nested".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Mixture {
                        spec: Arc::new(nested),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("ppmd".to_string()),
                    log_prior: -0.2,
                    backend: RateBackend::Ppmd {
                        order: 5,
                        memory_mb: 8,
                    },
                },
            ],
        )
        .with_alpha(0.13);
        RateBackend::Mixture {
            spec: Arc::new(root),
        }
    }

    fn assert_bitwise_byte_step_matches_pdf_and_plain_update(
        mut predictor: RatePdfPredictor,
        data: &[u8],
        tol_prob: f64,
        tol_pdf: f64,
    ) {
        for &symbol in data {
            let expected_pdf = predictor.pdf_next().unwrap().to_vec();
            let expected_prob = expected_pdf[symbol as usize];

            let mut stepped = predictor.clone();
            let mut product = 1.0f64;
            let produced = stepped
                .ac_step_bitwise(|bit_idx, p1| {
                    let bit = (symbol >> (7 - bit_idx)) & 1;
                    let pb = if bit == 1 { p1 } else { 1.0 - p1 };
                    product *= pb;
                    Ok(bit)
                })
                .unwrap();
            assert_eq!(produced, symbol);
            assert!(
                (product - expected_prob).abs() <= tol_prob,
                "symbol={symbol} product={product} expected_prob={expected_prob}"
            );

            let mut plain = predictor.clone();
            plain.update(symbol).unwrap();
            let stepped_pdf = stepped.pdf_next().unwrap().to_vec();
            let plain_pdf = plain.pdf_next().unwrap().to_vec();
            assert_pdf_close(&stepped_pdf, &plain_pdf, tol_pdf);

            predictor.update(symbol).unwrap();
        }
    }

    #[test]
    fn bitwise_byte_step_matches_pdf_and_plain_update_for_native_and_recursive_mixtures() {
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            RatePdfPredictor::from_rate_backend(RateBackend::Ctw { depth: 7 }).unwrap(),
            b"direct ctw bitwise byte step parity",
            1e-12,
            1e-12,
        );

        let direct_fac = RateBackend::FacCtw {
            base_depth: 7,
            num_percept_bits: 8,
            encoding_bits: 8,
            msb_first: None,
        };
        let direct_fac_predictor = RatePdfPredictor::from_rate_backend(direct_fac).unwrap();
        assert!(
            direct_fac_predictor.can_fast_ac_bitwise(),
            "byte-wide MSB fac-ctw must expose its native recursive AC path"
        );
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            direct_fac_predictor,
            b"direct fac ctw bitwise byte step parity",
            1e-12,
            1e-12,
        );

        let single_expert = RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![crate::MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Ctw { depth: 7 },
                }],
            )),
        };
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            RatePdfPredictor::from_rate_backend(single_expert).unwrap(),
            b"single expert ctw mixture bitwise byte step parity",
            1e-12,
            1e-12,
        );

        let single_fac_neural = RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Neural,
                vec![crate::MixtureExpertSpec {
                    name: Some("fac-ctw".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::FacCtw {
                        base_depth: 7,
                        num_percept_bits: 8,
                        encoding_bits: 8,
                        msb_first: None,
                    },
                }],
            )),
        };
        let single_fac_neural_predictor =
            RatePdfPredictor::from_rate_backend(single_fac_neural).unwrap();
        assert!(
            single_fac_neural_predictor.can_fast_ac_bitwise(),
            "neural mixtures containing only byte-wide MSB fac-ctw must not fall back to PDF-prefix AC"
        );
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            single_fac_neural_predictor,
            b"single expert fac neural mixture bitwise byte step parity",
            1e-12,
            1e-12,
        );

        let single_logistic = RateBackend::Mixture {
            spec: Arc::new(
                MixtureSpec::new(
                    MixtureKind::Logistic,
                    vec![crate::MixtureExpertSpec {
                        name: Some("ctw".to_string()),
                        log_prior: 0.0,
                        backend: RateBackend::Ctw { depth: 7 },
                    }],
                )
                .with_alpha(0.03),
            ),
        };
        let single_logistic_predictor =
            RatePdfPredictor::from_rate_backend(single_logistic).unwrap();
        assert!(
            single_logistic_predictor.can_fast_ac_bitwise(),
            "one-expert logistic over CTW should use the expert recursive AC path"
        );
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            single_logistic_predictor,
            b"single expert logistic ctw bitwise byte step parity",
            1e-12,
            1e-12,
        );

        let logistic = RateBackend::Mixture {
            spec: Arc::new(
                MixtureSpec::new(
                    MixtureKind::Logistic,
                    vec![
                        crate::MixtureExpertSpec {
                            name: Some("ctw".to_string()),
                            log_prior: 0.0,
                            backend: RateBackend::Ctw { depth: 7 },
                        },
                        crate::MixtureExpertSpec {
                            name: Some("match".to_string()),
                            log_prior: -0.2,
                            backend: RateBackend::Match {
                                hash_bits: 18,
                                min_len: 3,
                                max_len: 64,
                                base_mix: 0.03,
                                confidence_scale: 1.0,
                            },
                        },
                    ],
                )
                .with_alpha(0.03),
            ),
        };
        let logistic_predictor = RatePdfPredictor::from_rate_backend(logistic).unwrap();
        assert!(
            logistic_predictor.can_fast_ac_bitwise(),
            "multi-expert logistic mixtures expose a bitwise AC path even when experts are byte-PDF predictors"
        );
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            logistic_predictor,
            b"logistic mixture bitwise byte step parity",
            1e-8,
            1e-8,
        );

        let mixed_direct = RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![
                    crate::MixtureExpertSpec {
                        name: Some("ctw".to_string()),
                        log_prior: 0.0,
                        backend: RateBackend::Ctw { depth: 7 },
                    },
                    crate::MixtureExpertSpec {
                        name: Some("match".to_string()),
                        log_prior: -0.3,
                        backend: RateBackend::Match {
                            hash_bits: 20,
                            min_len: 4,
                            max_len: 255,
                            base_mix: 0.02,
                            confidence_scale: 1.0,
                        },
                    },
                ],
            )),
        };
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            RatePdfPredictor::from_rate_backend(mixed_direct).unwrap(),
            b"mixed direct mixture bitwise byte step parity",
            1e-11,
            1e-11,
        );

        let recursive = recursive_native_bitwise_backend();
        let predictor = RatePdfPredictor::from_rate_backend(recursive).unwrap();
        assert!(predictor.can_fast_ac_bitwise());
        assert_bitwise_byte_step_matches_pdf_and_plain_update(
            predictor,
            b"recursive nested mixture bitwise byte step parity",
            1e-10,
            1e-10,
        );
    }

    fn assert_cached_cdf_fast_bitwise_matches_pdf_rows(mut predictor: RatePdfPredictor) {
        let data = b"cached cdf parity check payload";
        for &symbol in data {
            let pdf = predictor.pdf_next().unwrap().to_vec();
            assert!(predictor.prepare_cached_cdf_fast_bitwise().unwrap());

            let mut row = zeroed_prefix_cdf();
            fill_prefix_cdf_from_pdf(&mut row, &pdf, PDF_MIN);

            let mut stack = vec![MsbPrefixRange::FULL];
            while let Some(range) = stack.pop() {
                if range.hi() - range.lo() <= 1 {
                    continue;
                }
                let expected = range.prob_one(&row, PDF_MIN);
                let got = predictor
                    .cached_cdf_bit_prob_one_msb(range)
                    .expect("cached cdf branch probability");
                let diff = (expected - got).abs();
                assert!(
                    diff <= 1e-12,
                    "lo={} hi={} expected={expected} got={got} diff={diff}",
                    range.lo(),
                    range.hi()
                );
                stack.push(range.observed(false));
                stack.push(range.observed(true));
            }

            predictor.update(symbol).unwrap();
        }
    }

    #[test]
    fn cached_cdf_fast_bitwise_matches_pdf_rows_for_specialized_predictors() {
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(RateBackend::RosaPlus { max_order: -1 }).unwrap(),
        );
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(RateBackend::Ppmd {
                order: 6,
                memory_mb: 8,
            })
            .unwrap(),
        );
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(RateBackend::Match {
                hash_bits: 20,
                min_len: 4,
                max_len: 255,
                base_mix: 0.02,
                confidence_scale: 1.0,
            })
            .unwrap(),
        );
        #[cfg(feature = "backend-rwkv")]
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(RateBackend::Rwkv7Method {
                method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("rwkv method spec"),
            })
            .unwrap(),
        );
        #[cfg(feature = "backend-mamba")]
        assert_cached_cdf_fast_bitwise_matches_pdf_rows(
            RatePdfPredictor::from_rate_backend(RateBackend::MambaMethod {
                method: crate::mambazip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=7,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("mamba method spec"),
            })
            .unwrap(),
        );
    }

    #[test]
    fn raw_size_not_larger_than_framed_size() {
        let data = b"raw/framed size check payload";
        let backend = RateBackend::RosaPlus { max_order: 8 };
        let raw = compress_rate_size(data, &backend, CoderType::AC, FramingMode::Raw).unwrap();
        let framed =
            compress_rate_size(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert!(framed >= raw);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn roundtrip_rate_rwkv_method_cfg() {
        let data = b"rwkv cfg method backend";
        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("rwkv method spec"),
        };
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
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
    fn compiled_rwkv_rate_pdf_predictor_preserves_backend_pdf_exactly() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec(method).expect("rwkv method spec"),
        }
        .compile()
        .expect("compiled rwkv backend");
        let spec = rwkvzip::parse_method_spec(method).expect("parsed rwkv spec");
        let mut predictor =
            RatePdfPredictor::from_compiled(&backend).expect("compiled rwkv predictor");
        let mut direct =
            rwkvzip::Compressor::new_from_method_spec(&spec).expect("rwkv backend from spec");
        let mut pdf = vec![0.0; direct.vocab_size()];

        let predicted = predictor.pdf_next().expect("predictor pdf").to_vec();
        direct.forward_to_pdf(0, &mut pdf);
        assert_pdf_close(&predicted, &pdf, 1e-18);

        predictor.update(b'x').expect("predictor update");
        direct
            .online_update_from_pdf(b'x', &pdf)
            .expect("backend update");
        direct.forward_to_pdf(u32::from(b'x'), &mut pdf);
        assert_pdf_close(predictor.pdf_next().expect("predictor pdf"), &pdf, 1e-18);
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
            serde_json::from_str(include_str!("../../../../configs/bench/two.json")).unwrap();
        let experts = two_json["experts"]
            .as_array()
            .expect("two.json must define experts array");
        let method = experts
            .iter()
            .find(|expert| expert["kind"].as_str() == Some("rwkv7"))
            .and_then(|expert| expert["method"].as_str())
            .expect("two.json must include rwkv7 expert with string method")
            .to_string();

        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec(&method).expect("rwkv method spec"),
        };
        let seed = include_bytes!("../../../../README.md");
        let target_len = 2_097_152usize;
        let mut data = Vec::with_capacity(target_len);
        while data.len() < target_len {
            let remaining = target_len - data.len();
            data.extend_from_slice(&seed[..seed.len().min(remaining)]);
        }

        let enc = compress_rate_bytes(&data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn benchmark_two_json_matches_examples_and_historical_alpha() {
        let canonical: serde_json::Value =
            serde_json::from_str(include_str!("../../../../configs/bench/two.json")).unwrap();
        let example: serde_json::Value =
            serde_json::from_str(include_str!("../../../../examples/two.json")).unwrap();

        assert_eq!(canonical, example, "benchmark specs drifted");
        assert_eq!(canonical["kind"].as_str(), Some("neural"));
        let alpha = canonical["alpha"].as_f64().expect("neural alpha");
        assert!(
            (alpha - 0.03).abs() <= 1e-12,
            "expected historical neural alpha 0.03, got {alpha}"
        );
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

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn compiled_mamba_rate_pdf_predictor_preserves_backend_pdf_exactly() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,state=8,conv=3,dt_rank=4,seed=7,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer";
        let backend = RateBackend::MambaMethod {
            method: crate::mambazip::parse_method_spec(method).expect("mamba method spec"),
        }
        .compile()
        .expect("compiled mamba backend");
        let spec = mambazip::parse_method_spec(method).expect("parsed mamba spec");
        let mut predictor =
            RatePdfPredictor::from_compiled(&backend).expect("compiled mamba predictor");
        let mut direct =
            mambazip::Compressor::new_from_method_spec(&spec).expect("mamba backend from spec");
        let mut pdf = vec![0.0; direct.vocab_size()];

        let predicted = predictor.pdf_next().expect("predictor pdf").to_vec();
        direct.forward_to_pdf(0, &mut pdf);
        assert_pdf_close(&predicted, &pdf, 1e-18);

        predictor.update(b'x').expect("predictor update");
        direct
            .online_update_from_pdf(b'x', &pdf)
            .expect("backend update");
        direct.forward_to_pdf(u32::from(b'x'), &mut pdf);
        assert_pdf_close(predictor.pdf_next().expect("predictor pdf"), &pdf, 1e-18);
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
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
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
            compress_rate_bytes(data, &backend, CoderType::RANS, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::RANS, FramingMode::Framed).unwrap();
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
                    backend: RateBackend::Particle {
                        spec: Arc::new(particle_spec),
                    },
                },
                crate::MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    backend: RateBackend::Ctw { depth: 6 },
                },
            ],
        );
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let data = b"mixture with particle expert roundtrip";
        let enc = compress_rate_bytes(data, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        let dec =
            decompress_rate_bytes(&enc, &backend, CoderType::AC, FramingMode::Framed).unwrap();
        assert_eq!(dec, data);
    }
}
