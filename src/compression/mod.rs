use anyhow::{Result, bail};

#[cfg(feature = "backend-rwkv")]
use crate::backends::rwkvzip;
use crate::coders::{
    ANS_TOTAL, ArithmeticDecoder, ArithmeticEncoder, BlockedRansDecoder, BlockedRansEncoder,
    CDF_TOTAL, Cdf, quantize_pdf_to_cdf_inplace, quantize_pdf_to_rans_cdf_with_buffer,
};
use crate::ctw::FacContextTree;
use crate::mixture::DEFAULT_MIN_PROB;
use crate::rosaplus::RosaPlus;
use crate::zpaq_rate::ZpaqRateModel;
use crate::{MixtureKind, MixtureSpec, RateBackend};

const FRAMED_MAGIC: u32 = 0x4354_4946; // "FITC"
const FRAMED_VERSION: u8 = 1;
const PDF_MIN: f64 = DEFAULT_MIN_PROB;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FramingMode {
    Raw,
    Framed,
}

impl Default for FramingMode {
    fn default() -> Self {
        Self::Framed
    }
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
    valid: bool,
}

impl CtwPredictor {
    fn new_ctw(depth: usize) -> Self {
        Self {
            tree: FacContextTree::new(depth, 8),
            bits_per_symbol: 8,
            msb_first: true,
            pdf: vec![0.0; 256],
            valid: false,
        }
    }

    fn new_fac(base_depth: usize, bits_per_symbol: usize) -> Self {
        Self {
            tree: FacContextTree::new(base_depth, bits_per_symbol),
            bits_per_symbol,
            msb_first: false,
            pdf: vec![0.0; 256],
            valid: false,
        }
    }

    fn log_prob_symbol(&mut self, symbol: u8) -> f64 {
        let before = self.tree.get_log_block_probability();
        if self.msb_first {
            for bit_idx in 0..self.bits_per_symbol {
                let bit = ((symbol >> (7 - bit_idx)) & 1) == 1;
                self.tree.update(bit, bit_idx);
            }
            let after = self.tree.get_log_block_probability();
            for bit_idx in (0..self.bits_per_symbol).rev() {
                self.tree.revert(bit_idx);
            }
            after - before
        } else {
            for bit_idx in 0..self.bits_per_symbol {
                let bit = ((symbol >> bit_idx) & 1) == 1;
                self.tree.update(bit, bit_idx);
            }
            let after = self.tree.get_log_block_probability();
            for bit_idx in (0..self.bits_per_symbol).rev() {
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
            if self.bits_per_symbol >= 8 {
                for sym in 0..256u16 {
                    self.pdf[sym as usize] = self.log_prob_symbol(sym as u8).exp();
                }
            } else {
                let patterns = 1usize << self.bits_per_symbol;
                let aliases = 1usize << (8 - self.bits_per_symbol);
                let mut ppat = vec![0.0f64; patterns];
                for pat in 0..patterns {
                    ppat[pat] = self.log_prob_symbol(pat as u8).exp();
                }
                for byte in 0..256usize {
                    let pat = byte & (patterns - 1);
                    self.pdf[byte] = ppat[pat] / (aliases as f64);
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
            for s in 0..256usize {
                self.pdf[s] = self.model.prob_for_last(s as u32).max(PDF_MIN);
            }
            let sum: f64 = self.pdf.iter().sum();
            let inv = if sum.is_finite() && sum > 0.0 {
                1.0 / sum
            } else {
                1.0 / 256.0
            };
            for p in &mut self.pdf {
                *p = (*p * inv).max(PDF_MIN);
            }
            let norm: f64 = self.pdf.iter().sum();
            if norm > 0.0 {
                let invn = 1.0 / norm;
                for p in &mut self.pdf {
                    *p *= invn;
                }
            }
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
        Ok(Self {
            kind: spec.kind,
            alpha: spec.alpha.clamp(1e-12, 1.0 - 1e-12),
            decay: spec.decay.unwrap_or(1.0).clamp(0.0, 1.0),
            experts,
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
        self.pdf.fill(0.0);

        let lw_norm = logsumexp(self.experts.iter().map(|e| e.log_weight));
        for e in &mut self.experts {
            let w = (e.log_weight - lw_norm).exp();
            let epdf = e.predictor.pdf_next()?;
            for (i, p) in epdf.iter().enumerate().take(256) {
                self.pdf[i] += w * *p;
            }
        }

        normalize_pdf(&mut self.pdf);
        self.valid = true;
        Ok(&self.pdf)
    }

    fn update(&mut self, symbol: u8) -> Result<()> {
        let _ = self.ensure_pdf()?;

        let n = self.experts.len();
        self.scratch.resize(n, 0.0);
        self.scratch2.resize(n, 0.0);

        for (i, e) in self.experts.iter_mut().enumerate() {
            let p = e.predictor.pdf_next()?[symbol as usize].max(PDF_MIN);
            let lp = p.ln();
            self.scratch[i] = lp;
            self.scratch2[i] = e.log_weight + lp;
        }

        match self.kind {
            MixtureKind::Bayes => {
                let log_mix = logsumexp(self.scratch2.iter().copied());
                for (i, e) in self.experts.iter_mut().enumerate() {
                    e.log_weight = e.log_weight + self.scratch[i] - log_mix;
                    e.cum_log_loss -= self.scratch[i];
                    e.predictor.update(symbol)?;
                }
            }
            MixtureKind::FadingBayes => {
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
                for (i, e) in self.experts.iter_mut().enumerate() {
                    e.cum_log_loss -= self.scratch[i];
                    e.predictor.update(symbol)?;
                }
            }
        }

        self.valid = false;
        Ok(())
    }
}

#[derive(Clone)]
enum RatePdfPredictor {
    Rosa(RosaPredictor),
    Ctw(CtwPredictor),
    FacCtw(CtwPredictor),
    #[cfg(feature = "backend-rwkv")]
    Rwkv(RwkvPredictor),
    Zpaq(ZpaqPredictor),
    Mixture(MixturePredictor),
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
        }
    }
}

fn encode_payload_ac(data: &[u8], predictor: &mut RatePdfPredictor) -> Result<Vec<u8>> {
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
}
