use std::sync::{Arc, OnceLock};

use crate::api::CalibrationContextKind;
use crate::backends::text_context::{NeuralContextState, TextContextAnalyzer};
use crate::byte_prefix::{
    BytePrefixCdf, MsbPrefixRange, advanced_prefix_code, fill_normalized_prefix_cdf_from_pdf,
    normalize_pdf, zeroed_prefix_cdf,
};

const PROB_SCALE: u32 = 32_767;
const COUNT_BITS: u32 = 10;
const COUNT_MASK: u32 = (1 << COUNT_BITS) - 1;
const COUNT_RECIP_LEN: usize = COUNT_MASK as usize + 3;
const CORRECTION_BITS: u32 = 32 - COUNT_BITS;
const CORRECTION_MASK: u32 = (1 << CORRECTION_BITS) - 1;
const CORRECTION_UNITS_PER_NAT: f64 = 512.0;
const CORRECTION_CLIP: i32 = 16_384;
const MAX_CORRECTION_STEP: f64 = 512.0;
const MIN_TRAIN_VARIANCE: f64 = 1.0 / 4096.0;
const INTERP_SCALE: i32 = 256;
/// Number of internal nodes in the 1-rooted MSB byte-prefix tree (`2^8 - 1`).
const BYTE_PREFIX_STATES: usize = 255;
const MIN_BINS: usize = 2;
const MAX_BINS: usize = 256;
const MIN_LEARNING_RATE: f64 = 1.0 / 4096.0;
const DEFAULT_LEARNING_RATE: f64 = 1.0 / 32.0;
const DEFAULT_STRETCH_CLIP: f64 = 16.0;
const MIN_STRETCH_CLIP: f64 = 1.0;
const MAX_STRETCH_CLIP: f64 = 32.0;

type SseTableRows = Vec<Option<Arc<[u32]>>>;

#[derive(Clone, Copy, Debug)]
struct SseQuantization {
    lower_bin: usize,
    weight_hi: i32,
    nearest_bin: usize,
}

#[derive(Clone, Copy, Debug)]
struct SseMappedBit {
    prob_one: f64,
    nearest_row: usize,
    nearest_bin: usize,
}

#[derive(Clone, Copy, Debug)]
struct SseUndo {
    row: usize,
    bin: usize,
    previous: Option<u32>,
}

const EMPTY_SSE_UNDO: SseUndo = SseUndo {
    row: 0,
    bin: 0,
    previous: None,
};

#[derive(Clone, Debug)]
/// Online secondary symbol estimator for MSB-first bit prediction.
///
/// The table is indexed by a byte-level context family, the current in-byte
/// MSB prefix, and a stretched-logit quantization of the wrapped predictor's
/// bit probability. Each entry stores a signed log-odds correction plus a
/// saturating count. Entries are initialized to zero correction, so wrapping a
/// predictor is conservative before online evidence accumulates.
pub struct CalibratorCore {
    analyzer: TextContextAnalyzer,
    context: CalibrationContextKind,
    bins: usize,
    first_stretch: i32,
    last_stretch: i32,
    stretch_scale: f64,
    max_quantized_pos: i64,
    initial_entry: u32,
    table: Arc<SseTableRows>,
    prefix: u16,
    active_bits: Option<u8>,
    active_context: usize,
    last_nearest_row: usize,
    last_nearest_bin: usize,
    last_prob_one: f64,
    last_valid: bool,
    active_undo: [SseUndo; 8],
    active_undo_len: u8,
}

impl CalibratorCore {
    /// Create an SSE calibrator with bounded table dimensions and adaptation.
    pub fn new(
        context: CalibrationContextKind,
        bins: usize,
        learning_rate: f64,
        bias_clip: f64,
    ) -> Self {
        let bins: usize = bins.clamp(MIN_BINS, MAX_BINS);
        let learning_rate: f64 = sanitize_learning_rate(learning_rate);
        let initial_count: u16 = initial_count_from_learning_rate(learning_rate);
        let (first_stretch, last_stretch) = stretch_range_from_clip(bias_clip);
        let stretch_span: i32 = (last_stretch - first_stretch).max(1);
        let max_quantized_pos: i64 = ((bins - 1) * (INTERP_SCALE as usize)) as i64;
        let stretch_scale: f64 = (max_quantized_pos as f64) / f64::from(stretch_span);
        let rows: usize = context_cardinality(context) * BYTE_PREFIX_STATES;
        let initial_entry: u32 = pack_entry(0, initial_count);
        let mut table: SseTableRows = Vec::with_capacity(rows);
        table.resize_with(rows, || None);

        Self {
            analyzer: TextContextAnalyzer::new(),
            context,
            bins,
            first_stretch,
            last_stretch,
            stretch_scale,
            max_quantized_pos,
            initial_entry,
            table: Arc::new(table),
            prefix: 1,
            active_bits: None,
            active_context: 0,
            last_nearest_row: 0,
            last_nearest_bin: 0,
            last_prob_one: 0.5,
            last_valid: false,
            active_undo: [EMPTY_SSE_UNDO; 8],
            active_undo_len: 0,
        }
    }

    /// Start an MSB-first byte-prefix prediction session.
    pub fn begin_byte(&mut self) -> Result<(), String> {
        if self.active_bits.is_some() {
            return Err("calibrated SSE byte-prefix step is already active".to_string());
        }
        self.active_context = context_index(self.context, self.analyzer.state());
        self.active_bits = Some(0);
        self.prefix = 1;
        self.last_valid = false;
        self.active_undo_len = 0;
        Ok(())
    }

    /// Abort an active byte-prefix session before any bits have been observed.
    pub fn abort_empty_byte(&mut self) -> Result<(), String> {
        self.validate_empty_byte()?;
        self.active_bits = None;
        self.prefix = 1;
        self.active_context = 0;
        self.last_valid = false;
        self.active_undo_len = 0;
        Ok(())
    }

    /// Validate that an active byte-prefix session has not consumed any bits.
    pub fn validate_empty_byte(&self) -> Result<(), String> {
        match self.active_bits {
            Some(0) | None => Ok(()),
            Some(bits) => Err(format!(
                "calibrated SSE byte-prefix abort requires zero observed bits, got {bits}"
            )),
        }
    }

    /// Return whether a byte-prefix session is currently open.
    pub fn byte_is_active(&self) -> bool {
        self.active_bits.is_some()
    }

    /// Validate that the active byte-prefix session consumed all eight bits.
    pub fn validate_complete_byte(&self) -> Result<(), String> {
        match self.active_bits {
            Some(8) => Ok(()),
            Some(bits) => Err(format!(
                "calibrated SSE byte-prefix finish requires 8 observed bits, got {bits}"
            )),
            None => Err("calibrated SSE byte-prefix step is not active".to_string()),
        }
    }

    /// Finish a complete MSB-first byte-prefix prediction session.
    pub fn finish_byte(&mut self) -> Result<(), String> {
        self.validate_complete_byte()?;
        self.active_bits = None;
        self.prefix = 1;
        self.active_context = 0;
        self.last_valid = false;
        self.active_undo_len = 0;
        Ok(())
    }

    /// Roll back training from an incomplete active byte-prefix session.
    pub fn rollback_incomplete_byte(&mut self) -> Result<(), String> {
        match self.active_bits {
            Some(bits) if bits < 8 => {}
            Some(8) => {
                return Err(
                    "calibrated SSE byte-prefix rollback cannot undo a completed byte".to_string(),
                );
            }
            None => return Ok(()),
            Some(bits) => {
                return Err(format!(
                    "calibrated SSE byte-prefix rollback saw invalid bit count {bits}"
                ));
            }
        }

        if self.active_undo_len > 0 {
            let table: &mut SseTableRows = Arc::make_mut(&mut self.table);
            for idx in (0..usize::from(self.active_undo_len)).rev() {
                let undo: SseUndo = self.active_undo[idx];
                match undo.previous {
                    Some(previous) => {
                        if let Some(row) = table[undo.row].as_mut() {
                            Arc::make_mut(row)[undo.bin] = previous;
                        }
                    }
                    None => {
                        table[undo.row] = None;
                    }
                }
            }
        }

        self.active_bits = None;
        self.prefix = 1;
        self.active_context = 0;
        self.last_valid = false;
        self.active_undo_len = 0;
        Ok(())
    }

    /// Predict `P(bit = 1)` for the active MSB-first prefix bit.
    pub fn predict_bit(&mut self, bit_idx: usize, base_prob_one: f64) -> Result<f64, String> {
        self.validate_active_bit(bit_idx)?;
        Ok(self.predict_bit_unchecked(base_prob_one))
    }

    /// Predict the next active bit without rechecking the sequential protocol.
    #[inline]
    pub(crate) fn predict_bit_unchecked(&mut self, base_prob_one: f64) -> f64 {
        debug_assert!(
            self.active_bits.is_some(),
            "unchecked calibrated SSE prediction requires an active byte"
        );
        let mapped: SseMappedBit = self.map_bit(self.active_context, self.prefix, base_prob_one);
        self.last_nearest_row = mapped.nearest_row;
        self.last_nearest_bin = mapped.nearest_bin;
        self.last_prob_one = mapped.prob_one;
        self.last_valid = true;
        mapped.prob_one
    }

    /// Train the last predicted SSE cell from the observed bit.
    pub fn observe_bit(&mut self, bit_idx: usize, bit: bool) -> Result<(), String> {
        self.validate_active_bit(bit_idx)?;
        if !self.last_valid {
            return Err(
                "calibrated SSE bit observation requires a preceding prediction".to_string(),
            );
        }
        self.observe_bit_unchecked(bit);
        Ok(())
    }

    /// Train the active SSE bit, first materializing its prediction if needed.
    ///
    /// Prefix consumers are allowed to use either predict-then-observe or
    /// update-only APIs. The latter still needs an exact calibrated probability
    /// for the online log-odds update, so callers provide the wrapped predictor's
    /// current bit probability and this method primes the pending SSE cell only
    /// when no prior prediction is active.
    pub fn observe_bit_from_base(
        &mut self,
        bit_idx: usize,
        base_prob_one: f64,
        bit: bool,
    ) -> Result<(), String> {
        self.validate_active_bit(bit_idx)?;
        if !self.last_valid {
            let _ = self.predict_bit_unchecked(base_prob_one);
        }
        self.observe_bit_unchecked(bit);
        Ok(())
    }

    /// Advance an active frozen prefix without fitting SSE entries.
    pub fn condition_prefix_bit_for_rollback(
        &mut self,
        bit_idx: usize,
        bit: bool,
    ) -> Result<(), String> {
        self.validate_active_bit(bit_idx)?;
        self.last_valid = false;
        if bit_idx < 7 {
            self.prefix = advanced_prefix_code(self.prefix, bit);
            if let Some(active_bits) = self.active_bits.as_mut() {
                *active_bits += 1;
            }
        }
        Ok(())
    }

    /// Train the last predicted bit without rechecking the sequential protocol.
    #[inline]
    pub(crate) fn observe_bit_unchecked(&mut self, bit: bool) {
        debug_assert!(
            self.last_valid,
            "unchecked calibrated SSE observation requires a preceding prediction"
        );
        debug_assert!(
            self.active_bits.is_some(),
            "unchecked calibrated SSE observation requires an active byte"
        );
        self.train_nearest(bit);
        self.last_valid = false;
        self.advance_prefix(bit);
        if let Some(active_bits) = self.active_bits.as_mut() {
            *active_bits += 1;
        }
    }

    /// Apply the current SSE table to a base byte distribution.
    ///
    /// This is a byte-facing fallback. The primary compression path uses
    /// [`Self::predict_bit`] and [`Self::observe_bit`] directly so calibration
    /// updates occur at the same granularity as arithmetic coding. This method
    /// only projects the current model; context advances through
    /// [`Self::observe_symbol_from_base_pdf`] or [`Self::update_context_only`].
    pub fn apply_pdf(&mut self, base: &[f64], out: &mut [f64]) {
        let ctx: usize = context_index(self.context, self.analyzer.state());
        let mut cdf: BytePrefixCdf = zeroed_prefix_cdf();
        fill_normalized_prefix_cdf_from_pdf(&mut cdf, base);
        let prefix0: u16 = self.prefix;
        let len: usize = out.len().min(256);
        for (symbol, slot) in out.iter_mut().enumerate().take(len) {
            let mut range: MsbPrefixRange = MsbPrefixRange::FULL;
            let mut prefix: u16 = prefix0;
            let mut prob: f64 = 1.0;
            for bit_idx in 0..8usize {
                let base_p1: f64 = range.prob_one(&cdf, f64::MIN_POSITIVE);
                let calibrated_p1: f64 = self.map_bit(ctx, prefix, base_p1).prob_one;
                let bit: bool = ((symbol as u8) & (1u8 << (7 - bit_idx))) != 0;
                prob *= if bit {
                    calibrated_p1
                } else {
                    1.0 - calibrated_p1
                };
                range.observe(bit);
                prefix = advanced_prefix_code(prefix, bit);
            }
            *slot = prob;
        }
        for slot in out.iter_mut().skip(len) {
            *slot = 0.0;
        }
        normalize_pdf(out, f64::MIN_POSITIVE);
    }

    /// Train SSE from a complete symbol using a base byte distribution.
    pub fn observe_symbol_from_base_pdf(&mut self, symbol: u8, base: &[f64]) -> Result<(), String> {
        self.begin_byte()?;
        let mut cdf: BytePrefixCdf = zeroed_prefix_cdf();
        fill_normalized_prefix_cdf_from_pdf(&mut cdf, base);
        let mut range: MsbPrefixRange = MsbPrefixRange::FULL;
        for bit_idx in 0..8usize {
            let base_p1: f64 = range.prob_one(&cdf, f64::MIN_POSITIVE);
            let bit: bool = (symbol & (1u8 << (7 - bit_idx))) != 0;
            self.predict_bit(bit_idx, base_p1)?;
            self.observe_bit(bit_idx, bit)?;
            range.observe(bit);
        }
        self.finish_byte()
    }

    /// Reset only dynamic context state while preserving learned SSE entries.
    pub fn reset_context(&mut self) {
        self.analyzer = TextContextAnalyzer::new();
        self.prefix = 1;
        self.active_bits = None;
        self.active_context = 0;
        self.last_valid = false;
        self.active_undo_len = 0;
    }

    /// Advance context state without updating fitted SSE entries.
    pub fn update_context_only(&mut self, symbol: u8) {
        self.analyzer.update(symbol);
        self.prefix = 1;
        self.active_bits = None;
        self.active_context = 0;
        self.last_valid = false;
        self.active_undo_len = 0;
    }

    fn validate_active_bit(&self, bit_idx: usize) -> Result<(), String> {
        if bit_idx >= 8 {
            return Err(format!(
                "calibrated SSE byte-prefix bit index {bit_idx} is out of range; expected 0..8"
            ));
        }
        match self.active_bits {
            Some(expected) if expected as usize == bit_idx => Ok(()),
            Some(expected) => Err(format!(
                "calibrated SSE byte-prefix bit index {bit_idx} violated sequential stepping; expected {expected}"
            )),
            None => Err("calibrated SSE byte-prefix step is not active".to_string()),
        }
    }

    fn map_bit(&self, context_idx: usize, prefix: u16, base_prob_one: f64) -> SseMappedBit {
        let quantized: SseQuantization = self.quantize(base_prob_one);
        let row: usize = self.row_index(context_idx, prefix);
        let mixed_delta: i32 = match self.table[row].as_deref() {
            Some(entries) => interpolate_delta(
                unpack_delta(entries[quantized.lower_bin]),
                unpack_delta(entries[quantized.lower_bin + 1]),
                quantized.weight_hi,
            ),
            None => 0,
        };
        SseMappedBit {
            prob_one: apply_logit_correction(base_prob_one, mixed_delta),
            nearest_row: row,
            nearest_bin: quantized.nearest_bin,
        }
    }

    fn row_index(&self, context_idx: usize, prefix: u16) -> usize {
        let prefix_idx: usize = usize::from(prefix.saturating_sub(1)).min(BYTE_PREFIX_STATES - 1);
        (context_idx * BYTE_PREFIX_STATES) + prefix_idx
    }

    fn quantize(&self, prob: f64) -> SseQuantization {
        let stretch: i32 = stretch_probability(prob).clamp(self.first_stretch, self.last_stretch);
        let offset: i32 = (stretch - self.first_stretch).max(0);
        let pos: i64 =
            ((f64::from(offset) * self.stretch_scale) as i64).clamp(0, self.max_quantized_pos);
        let lower_bin: usize = ((pos / i64::from(INTERP_SCALE)) as usize).min(self.bins - 2);
        let weight_hi: i32 = if pos >= self.max_quantized_pos {
            INTERP_SCALE
        } else {
            (pos % i64::from(INTERP_SCALE)) as i32
        };
        let nearest_bin: usize = if weight_hi * 2 >= INTERP_SCALE {
            lower_bin + 1
        } else {
            lower_bin
        };
        SseQuantization {
            lower_bin,
            weight_hi,
            nearest_bin,
        }
    }

    fn train_nearest(&mut self, bit: bool) {
        let row_idx: usize = self.last_nearest_row;
        let bin: usize = self.last_nearest_bin;
        let bins: usize = self.bins;
        let initial_entry: u32 = self.initial_entry;
        let table: &mut SseTableRows = Arc::make_mut(&mut self.table);
        let previous: Option<u32> = table[row_idx].as_deref().map(|row| row[bin]);
        if usize::from(self.active_undo_len) < self.active_undo.len() {
            self.active_undo[usize::from(self.active_undo_len)] = SseUndo {
                row: row_idx,
                bin,
                previous,
            };
            self.active_undo_len += 1;
        } else {
            debug_assert!(
                false,
                "calibrated SSE undo log exceeded one byte of bit updates"
            );
        }
        let row = table[row_idx].get_or_insert_with(|| initial_sse_row(bins, initial_entry));
        let entries: &mut [u32] = Arc::make_mut(row);
        let entry: u32 = entries[bin];
        let delta: i32 = unpack_delta(entry);
        let count: u16 = unpack_count(entry);
        let target: f64 = if bit { 1.0 } else { 0.0 };
        let prob: f64 = sanitize_unit_probability(self.last_prob_one);
        let error: f64 = target - prob;
        let variance: f64 = (prob * (1.0 - prob)).max(MIN_TRAIN_VARIANCE);
        let denom: usize = usize::from(count) + 2;
        let inv_denom: f64 = count_reciprocal_table()[denom];
        let step: i32 = (CORRECTION_UNITS_PER_NAT * error * inv_denom / variance)
            .round()
            .clamp(-MAX_CORRECTION_STEP, MAX_CORRECTION_STEP) as i32;
        let updated: i32 = (delta + step).clamp(-CORRECTION_CLIP, CORRECTION_CLIP);
        let next_count: u16 = count.saturating_add(1).min(COUNT_MASK as u16);
        entries[bin] = pack_entry(updated, next_count);
    }

    fn advance_prefix(&mut self, bit: bool) {
        let next: u16 = advanced_prefix_code(self.prefix, bit);
        if next >= 256 {
            self.analyzer.update((next & 0xff) as u8);
            self.prefix = 1;
        } else {
            self.prefix = next;
        }
    }
}

fn sanitize_learning_rate(learning_rate: f64) -> f64 {
    if learning_rate.is_finite() && learning_rate > 0.0 {
        learning_rate.clamp(MIN_LEARNING_RATE, 1.0)
    } else {
        DEFAULT_LEARNING_RATE
    }
}

fn initial_count_from_learning_rate(learning_rate: f64) -> u16 {
    let count: f64 = (1.0 / learning_rate) - 1.5;
    count.round().clamp(0.0, COUNT_MASK as f64) as u16
}

fn stretch_range_from_clip(bias_clip: f64) -> (i32, i32) {
    let clip: f64 = if bias_clip.is_finite() && bias_clip > 0.0 {
        bias_clip
    } else {
        DEFAULT_STRETCH_CLIP
    }
    .clamp(MIN_STRETCH_CLIP, MAX_STRETCH_CLIP);
    let clip_units: i32 = (clip * 64.0).round() as i32;
    let first: i32 = -clip_units + 32;
    let last: i32 = clip_units - 32;
    if first < last {
        (first, last)
    } else {
        (-32, 32)
    }
}

fn pack_entry(delta: i32, count: u16) -> u32 {
    let encoded_delta: u32 =
        (delta.clamp(-CORRECTION_CLIP, CORRECTION_CLIP) as u32) & CORRECTION_MASK;
    (encoded_delta << COUNT_BITS) | (u32::from(count) & COUNT_MASK)
}

fn initial_sse_row(bins: usize, initial_entry: u32) -> Arc<[u32]> {
    Arc::from(vec![initial_entry; bins].into_boxed_slice())
}

fn unpack_count(entry: u32) -> u16 {
    (entry & COUNT_MASK) as u16
}

fn unpack_delta(entry: u32) -> i32 {
    let raw: u32 = (entry >> COUNT_BITS) & CORRECTION_MASK;
    let shift: u32 = 32 - CORRECTION_BITS;
    (((raw << shift) as i32) >> shift).clamp(-CORRECTION_CLIP, CORRECTION_CLIP)
}

fn interpolate_delta(lo_delta: i32, hi_delta: i32, weight_hi: i32) -> i32 {
    debug_assert!((0..=INTERP_SCALE).contains(&weight_hi));
    let numerator: i32 = lo_delta * (INTERP_SCALE - weight_hi) + hi_delta * weight_hi;
    if numerator >= 0 {
        (numerator + INTERP_SCALE / 2) >> 8
    } else {
        -((-numerator + INTERP_SCALE / 2) >> 8)
    }
}

fn sanitize_unit_probability(prob: f64) -> f64 {
    if prob.is_finite() {
        prob.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    } else {
        0.5
    }
}

fn apply_logit_correction(base_prob_one: f64, delta: i32) -> f64 {
    let base: f64 = sanitize_unit_probability(base_prob_one);
    if delta == 0 {
        return base;
    }
    let factor: f64 = correction_factor(delta);
    let numerator: f64 = base * factor;
    let denominator: f64 = (1.0 - base) + numerator;
    if denominator.is_finite() && denominator > 0.0 {
        (numerator / denominator).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    } else if delta > 0 {
        1.0 - f64::EPSILON
    } else {
        f64::MIN_POSITIVE
    }
}

fn stretch_probability(prob: f64) -> i32 {
    let p: f64 = if prob.is_finite() { prob } else { 0.5 };
    let scaled: usize = (p.clamp(0.0, 1.0) * (PROB_SCALE as f64)).round() as usize;
    i32::from(stretch_table()[scaled.min(PROB_SCALE as usize)])
}

fn stretch_table() -> &'static [i16; 32_768] {
    // 64 KiB process-lifetime lookup table: this removes a hot-path ln() from
    // every SSE probability quantization while keeping binary size unaffected.
    static TABLE: OnceLock<[i16; 32_768]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table: [i16; 32_768] = [0; 32_768];
        for (idx, slot) in table.iter_mut().enumerate() {
            let numerator: f64 = idx as f64 + 0.5;
            let denominator: f64 = 32_767.5 - idx as f64;
            let stretch: f64 = 64.0 * (numerator / denominator).ln();
            *slot = stretch.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
        table
    })
}

fn correction_factor(delta: i32) -> f64 {
    let clipped: i32 = delta.clamp(-CORRECTION_CLIP, CORRECTION_CLIP);
    correction_factor_table()[(clipped + CORRECTION_CLIP) as usize]
}

fn correction_factor_table() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let len: usize = (CORRECTION_CLIP as usize) * 2 + 1;
        let mut table: Vec<f64> = Vec::with_capacity(len);
        for delta in -CORRECTION_CLIP..=CORRECTION_CLIP {
            table.push((f64::from(delta) / CORRECTION_UNITS_PER_NAT).exp());
        }
        table
    })
}

fn count_reciprocal_table() -> &'static [f64; COUNT_RECIP_LEN] {
    static TABLE: OnceLock<[f64; COUNT_RECIP_LEN]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table: [f64; COUNT_RECIP_LEN] = [0.0; COUNT_RECIP_LEN];
        for (idx, slot) in table.iter_mut().enumerate().skip(1) {
            *slot = 1.0 / (idx as f64);
        }
        table
    })
}

fn context_cardinality(kind: CalibrationContextKind) -> usize {
    match kind {
        CalibrationContextKind::Global => 1,
        CalibrationContextKind::ByteClass => 8,
        CalibrationContextKind::Text => 256,
        CalibrationContextKind::Repeat => 64,
        CalibrationContextKind::TextRepeat => 512,
    }
}

fn context_index(kind: CalibrationContextKind, state: NeuralContextState) -> usize {
    match kind {
        CalibrationContextKind::Global => 0,
        CalibrationContextKind::ByteClass => state.prev1_class as usize,
        CalibrationContextKind::Text => hash_state(
            &[
                state.prev1_class,
                state.prev2_class,
                state.word_len_bucket,
                state.prev_word_class,
                state.bracket_bucket,
                state.quote_flags,
                state.utf8_left,
                state.sentence_boundary as u8,
                state.paragraph_break as u8,
            ],
            256,
        ),
        CalibrationContextKind::Repeat => hash_state(
            &[
                state.repeat_len_bucket,
                state.copied_last_byte as u8,
                state.run_len.min(31) as u8,
            ],
            64,
        ),
        CalibrationContextKind::TextRepeat => hash_state(
            &[
                state.prev1_class,
                state.word_len_bucket,
                state.prev_word_class,
                state.bracket_bucket,
                state.quote_flags,
                state.repeat_len_bucket,
                state.copied_last_byte as u8,
                state.paragraph_break as u8,
            ],
            512,
        ),
    }
}

fn hash_state(values: &[u8], modulo: usize) -> usize {
    let mut h: u32 = 0x9E37_79B9;
    for &value in values {
        h ^= value as u32;
        h = h.rotate_left(5).wrapping_mul(0x85EB_CA6B);
    }
    (h as usize) % modulo
}

#[cfg(test)]
mod tests {
    use super::CalibratorCore;
    use crate::api::CalibrationContextKind;
    use crate::mixture::DEFAULT_MIN_PROB;
    use std::sync::Arc;

    #[test]
    fn new_calibrator_preserves_base_bit_probabilities_before_training() {
        let mut core = CalibratorCore::new(CalibrationContextKind::Global, 32, 1.0 / 32.0, 16.0);
        let probes = [
            DEFAULT_MIN_PROB,
            1.0e-5,
            0.001,
            0.1,
            0.5,
            0.9,
            0.999,
            1.0 - DEFAULT_MIN_PROB,
        ];

        core.begin_byte().expect("begin prefix");
        for &base_p1 in &probes {
            let calibrated = core
                .predict_bit(0, base_p1)
                .expect("untrained calibrator prediction");
            assert_eq!(
                calibrated, base_p1,
                "zero-correction SSE table must preserve base probability {base_p1:e}"
            );
        }
        core.abort_empty_byte().expect("abort empty prefix");
    }

    #[test]
    fn sse_rows_are_materialized_only_by_training() {
        let mut core = CalibratorCore::new(CalibrationContextKind::Global, 32, 1.0 / 32.0, 16.0);
        assert!(
            core.table.iter().all(Option::is_none),
            "new zero-correction SSE tables should not allocate dense rows"
        );

        core.begin_byte().expect("begin prefix");
        core.predict_bit(0, 0.5).expect("predict bit");
        assert!(
            core.table.iter().all(Option::is_none),
            "prediction alone must keep absent rows implicit"
        );

        core.observe_bit(0, true).expect("observe bit");
        let materialized_rows: usize = core.table.iter().filter(|row| row.is_some()).count();
        assert_eq!(
            materialized_rows, 1,
            "first trained bit should materialize exactly one SSE row"
        );
    }

    #[test]
    fn observe_symbol_from_base_pdf_reports_active_prefix_violation() {
        let mut core = CalibratorCore::new(CalibrationContextKind::Global, 32, 1.0 / 32.0, 16.0);
        let base = [1.0 / 256.0; 256];

        core.begin_byte().expect("begin prefix");
        let err = core
            .observe_symbol_from_base_pdf(0, &base)
            .expect_err("fallback byte observation must reject active prefix state");

        assert!(err.contains("already active"));
        core.abort_empty_byte().expect("abort empty prefix");
    }

    #[test]
    fn byte_prefix_finish_and_abort_validate_without_closing_nonempty_session() {
        let mut core = CalibratorCore::new(CalibrationContextKind::Global, 32, 1.0 / 32.0, 16.0);

        core.begin_byte().expect("begin prefix");
        let finish_err = core
            .finish_byte()
            .expect_err("empty prefix cannot finish as a byte");
        assert!(finish_err.contains("requires 8 observed bits, got 0"));
        assert!(core.byte_is_active());

        core.predict_bit(0, 0.5).expect("predict bit 0");
        core.observe_bit(0, true).expect("observe bit 0");
        let abort_err = core
            .abort_empty_byte()
            .expect_err("non-empty prefix cannot be aborted as empty");
        assert!(abort_err.contains("got 1"));
        assert!(core.byte_is_active());

        for bit_idx in 1..8usize {
            core.predict_bit(bit_idx, 0.5)
                .expect("predict remaining prefix bit");
            core.observe_bit(bit_idx, false)
                .expect("observe remaining prefix bit");
        }
        assert!(core.byte_is_active());
        core.finish_byte().expect("complete byte should finish");
        assert!(!core.byte_is_active());
    }

    #[test]
    fn observe_symbol_from_base_pdf_closes_completed_prefix_session() {
        let mut core = CalibratorCore::new(CalibrationContextKind::Global, 32, 1.0 / 32.0, 16.0);
        let base = [1.0 / 256.0; 256];

        core.observe_symbol_from_base_pdf(0x80, &base)
            .expect("first fallback observation");
        assert!(!core.byte_is_active());
        core.observe_symbol_from_base_pdf(0x00, &base)
            .expect("second fallback observation should not see stale active state");
    }

    #[test]
    fn cloned_calibrator_shares_sse_table_until_training_mutates_it() {
        let mut core =
            CalibratorCore::new(CalibrationContextKind::TextRepeat, 32, 1.0 / 32.0, 16.0);
        let checkpoint = core.clone();

        assert!(
            Arc::ptr_eq(&core.table, &checkpoint.table),
            "checkpoint clone should share the large SSE table"
        );

        core.begin_byte().expect("begin prefix");
        core.predict_bit(0, 0.5).expect("predict bit");
        core.observe_bit(0, true)
            .expect("training should detach table");

        assert!(
            !Arc::ptr_eq(&core.table, &checkpoint.table),
            "training with a live checkpoint must fork the shared SSE table"
        );
    }

    #[test]
    fn incomplete_byte_rollback_restores_sse_rows() {
        let mut core = CalibratorCore::new(CalibrationContextKind::Global, 32, 1.0 / 32.0, 16.0);

        core.begin_byte().expect("begin prefix");
        core.predict_bit(0, 0.5).expect("predict bit");
        core.observe_bit(0, true).expect("observe bit");
        assert_eq!(
            core.table.iter().filter(|row| row.is_some()).count(),
            1,
            "training should materialize one row before rollback"
        );

        core.rollback_incomplete_byte()
            .expect("rollback incomplete byte");
        assert!(
            core.table.iter().all(Option::is_none),
            "rollback should remove rows first materialized by the abandoned byte"
        );
        assert!(!core.byte_is_active());
    }
}
