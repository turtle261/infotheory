//! Bit-native reservoir predictor.
//!
//! `BitReservoirModel` is a primitive sequential Bernoulli model. It consumes the bit
//! stream one bit at a time, updates a deterministic fixed nonlinear reservoir,
//! trains learned sparse history embeddings, and fits a phase-specific logistic
//! readout online on the observed bit. Byte APIs are only MSB-first traversals
//! of the same bit process.

use crate::api::BitReservoirConfig;
use std::mem::MaybeUninit;
use std::sync::Arc;

const PHASES: usize = 8;
const LOGIT_CLIP: f32 = 18.0;
const HISTORY_WINDOWS: [usize; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64];
const BYTE_WINDOWS: [usize; 7] = [1, 2, 3, 4, 5, 6, 8];
const TOTAL_SPARSE_FEATURES: usize = HISTORY_WINDOWS.len() + BYTE_WINDOWS.len();
// `1 / sqrt(TOTAL_SPARSE_FEATURES)`, fixed to keep sparse SGD steps stable as
// features are summed into one Bernoulli logit.
const SPARSE_FEATURE_SCALE: f32 = 0.218_217_9;

/// Pre-update prediction state needed to train the online readout exactly once.
#[derive(Clone)]
pub struct BitReservoirPrediction {
    p1: f32,
    sparse_indices: [usize; TOTAL_SPARSE_FEATURES],
}

impl BitReservoirPrediction {
    /// Raw model probability before any caller-specific probability floor.
    #[inline]
    pub(crate) fn prob_one(&self) -> f64 {
        self.p1 as f64
    }
}

#[derive(Clone)]
struct BitReservoirParams {
    source: Box<[usize]>,
    recurrent: Box<[f32]>,
    input: Box<[f32]>,
    decay: Box<[f32]>,
    phase: Box<[f32]>,
}

#[derive(Clone)]
struct BitReservoirByteBranch {
    state: Vec<f32>,
    scratch: Vec<f32>,
    history: Vec<f32>,
    history_pos: usize,
    history_len: usize,
    phase: usize,
    phase_steps: [u64; PHASES],
    history_bits: u64,
    history_bit_len: usize,
    byte_history: u64,
    byte_history_len: usize,
    partial_byte: u8,
    partial_bits: usize,
    history_w_overrides: Vec<(usize, f32)>,
}

impl BitReservoirByteBranch {
    fn from_model(model: &BitReservoirModel) -> Self {
        Self {
            state: model.state.clone(),
            scratch: model.scratch.clone(),
            history: model.history.clone(),
            history_pos: model.history_pos,
            history_len: model.history_len,
            phase: model.phase,
            phase_steps: model.phase_steps,
            history_bits: model.history_bits,
            history_bit_len: model.history_bit_len,
            byte_history: model.byte_history,
            byte_history_len: model.byte_history_len,
            partial_byte: model.partial_byte,
            partial_bits: model.partial_bits,
            history_w_overrides: Vec::with_capacity(8 * TOTAL_SPARSE_FEATURES),
        }
    }

    fn history_weight(&self, model: &BitReservoirModel, idx: usize) -> f32 {
        self.history_w_overrides
            .iter()
            .rev()
            .find_map(|&(stored_idx, value)| (stored_idx == idx).then_some(value))
            .unwrap_or_else(|| history_weight_at(model.history_w.as_slice(), idx))
    }

    fn set_history_weight(&mut self, idx: usize, value: f32) {
        if let Some((_, stored_value)) = self
            .history_w_overrides
            .iter_mut()
            .find(|(stored_idx, _)| *stored_idx == idx)
        {
            *stored_value = value;
        } else {
            self.history_w_overrides.push((idx, value));
        }
    }
}

/// Online reservoir predictor over individual bits.
#[derive(Clone)]
pub struct BitReservoirModel {
    config: BitReservoirConfig,
    params: Arc<BitReservoirParams>,
    state: Vec<f32>,
    scratch: Vec<f32>,
    history: Vec<f32>,
    history_pos: usize,
    history_len: usize,
    phase: usize,
    phase_steps: [u64; PHASES],
    out_w: Arc<Vec<f32>>,
    delay_w: Arc<Vec<f32>>,
    history_w: Arc<Vec<f32>>,
    history_bits: u64,
    history_bit_len: usize,
    byte_history: u64,
    byte_history_len: usize,
    partial_byte: u8,
    partial_bits: usize,
    history_mask: usize,
    grad_clip: f32,
    weight_decay: f32,
    bias: [f32; PHASES],
}

impl BitReservoirModel {
    /// Construct a new model from a validated configuration.
    pub fn new(config: BitReservoirConfig) -> Result<Self, String> {
        config.validate().map_err(|err| err.to_string())?;
        let params = Arc::new(BitReservoirParams::new(&config));
        let hidden = config.hidden;
        let delay_bits = config.delay_bits;
        let history_slots = history_slots(&config);
        let history_mask = history_slots - 1;
        let grad_clip = config.grad_clip as f32;
        let weight_decay = config.weight_decay as f32;
        Ok(Self {
            config,
            params,
            state: vec![0.0; hidden],
            scratch: vec![0.0; hidden],
            history: vec![0.0; delay_bits],
            history_pos: 0,
            history_len: 0,
            phase: 0,
            phase_steps: [0; PHASES],
            out_w: Arc::new(vec![0.0; PHASES * hidden]),
            delay_w: Arc::new(vec![0.0; PHASES * delay_bits]),
            history_w: Arc::new(vec![0.0; TOTAL_SPARSE_FEATURES * history_slots]),
            history_bits: 0,
            history_bit_len: 0,
            byte_history: 0,
            byte_history_len: 0,
            partial_byte: 0,
            partial_bits: 0,
            history_mask,
            grad_clip,
            weight_decay,
            bias: [0.0; PHASES],
        })
    }

    /// Reset recurrent state and learned readout parameters.
    pub fn reset_all(&mut self) {
        self.state.fill(0.0);
        self.scratch.fill(0.0);
        self.history.fill(0.0);
        self.history_pos = 0;
        self.history_len = 0;
        self.phase = 0;
        self.phase_steps = [0; PHASES];
        Arc::make_mut(&mut self.out_w).fill(0.0);
        Arc::make_mut(&mut self.delay_w).fill(0.0);
        Arc::make_mut(&mut self.history_w).fill(0.0);
        self.history_bits = 0;
        self.history_bit_len = 0;
        self.byte_history = 0;
        self.byte_history_len = 0;
        self.partial_byte = 0;
        self.partial_bits = 0;
        self.bias = [0.0; PHASES];
    }

    /// Reset only conditioning state, preserving learned readout parameters.
    pub fn reset_state_only(&mut self) {
        self.state.fill(0.0);
        self.scratch.fill(0.0);
        self.history.fill(0.0);
        self.history_pos = 0;
        self.history_len = 0;
        self.history_bits = 0;
        self.history_bit_len = 0;
        self.byte_history = 0;
        self.byte_history_len = 0;
        self.partial_byte = 0;
        self.partial_bits = 0;
        self.phase = 0;
    }

    /// Current probability of the next bit being one.
    #[inline]
    pub fn predict_prob_one(&self) -> f64 {
        sigmoid(self.logit()) as f64
    }

    /// Capture the current prediction and sparse feature addresses for a
    /// subsequent learned observation.
    #[inline]
    pub(crate) fn predict_for_training(&self) -> BitReservoirPrediction {
        let mut sparse_indices = MaybeUninit::<[usize; TOTAL_SPARSE_FEATURES]>::uninit();
        // SAFETY: `logit_with_sparse_indices` writes every element by iterating
        // over `sparse_indices.iter_mut().enumerate()` before any element is
        // read or returned.
        let sparse_indices_ref = unsafe { &mut *sparse_indices.as_mut_ptr() };
        let logit = self.logit_with_sparse_indices(sparse_indices_ref);
        // SAFETY: Established above: every `usize` element has been initialized.
        let sparse_indices = unsafe { sparse_indices.assume_init() };
        BitReservoirPrediction {
            p1: sigmoid(logit),
            sparse_indices,
        }
    }

    /// Observe a bit using the exact pre-update prediction previously returned
    /// by [`Self::predict_for_training`].
    #[inline]
    pub(crate) fn observe_bit_with_prediction(
        &mut self,
        bit: bool,
        prediction: &BitReservoirPrediction,
    ) {
        self.train_readout_with_prediction(bit, prediction);
        self.advance_state(bit);
    }

    /// Observe one bit, optionally updating learned readout parameters.
    #[inline]
    pub fn observe_bit(&mut self, bit: bool, learn: bool) {
        if learn {
            let prediction = self.predict_for_training();
            self.observe_bit_with_prediction(bit, &prediction);
        } else {
            self.advance_state(bit);
        }
    }

    /// Log-probability of `byte` under the current state, without mutation.
    pub fn log_prob_byte(&self, byte: u8) -> f64 {
        self.log_prob_byte_with_min_prob(byte, crate::mixture::DEFAULT_MIN_PROB)
    }

    /// Log-probability of `byte` under the current state and probability floor.
    pub fn log_prob_byte_with_min_prob(&self, byte: u8, min_prob: f64) -> f64 {
        let mut branch = BitReservoirByteBranch::from_model(self);
        self.score_branch_byte(&mut branch, byte, min_prob)
    }

    /// Log-probability of `byte`, followed by an MSB-first byte update.
    pub fn log_prob_update_byte(&mut self, byte: u8, learn: bool) -> f64 {
        self.log_prob_update_byte_with_min_prob(byte, learn, crate::mixture::DEFAULT_MIN_PROB)
    }

    /// Log-probability of `byte`, followed by an MSB-first byte update.
    pub fn log_prob_update_byte_with_min_prob(
        &mut self,
        byte: u8,
        learn: bool,
        min_prob: f64,
    ) -> f64 {
        if learn {
            self.score_and_learn_byte(byte, min_prob)
        } else {
            self.score_and_advance_byte(byte, min_prob)
        }
    }

    fn score_and_learn_byte(&mut self, byte: u8, min_prob: f64) -> f64 {
        let mut logp = 0.0f64;
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            let prediction = self.predict_for_training();
            let p1 = prediction.prob_one();
            let p = if bit { p1 } else { 1.0 - p1 };
            logp += p.max(min_prob).ln();
            self.observe_bit_with_prediction(bit, &prediction);
        }
        logp
    }

    fn score_and_advance_byte(&mut self, byte: u8, min_prob: f64) -> f64 {
        let mut logp = 0.0f64;
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            let p1 = self.predict_prob_one();
            let p = if bit { p1 } else { 1.0 - p1 };
            logp += p.max(min_prob).ln();
            self.advance_state(bit);
        }
        logp
    }

    fn score_branch_byte(
        &self,
        branch: &mut BitReservoirByteBranch,
        byte: u8,
        min_prob: f64,
    ) -> f64 {
        let mut logp = 0.0f64;
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            let p1 = self.branch_predict_prob_one(branch);
            let p = if bit { p1 } else { 1.0 - p1 };
            logp += p.max(min_prob).ln();
            self.branch_observe_bit_with_prob(branch, bit, p1 as f32);
        }
        logp
    }

    /// Log-probability of `byte` under a frozen MSB-first byte update.
    pub fn log_prob_byte_frozen_with_min_prob(&self, byte: u8, min_prob: f64) -> f64 {
        let mut branch = BitReservoirByteBranch::from_model(self);
        self.score_branch_byte_frozen(&mut branch, byte, min_prob)
    }

    fn score_branch_byte_frozen(
        &self,
        branch: &mut BitReservoirByteBranch,
        byte: u8,
        min_prob: f64,
    ) -> f64 {
        let mut logp = 0.0f64;
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            let p1 = self.branch_predict_prob_one(branch);
            let p = if bit { p1 } else { 1.0 - p1 };
            logp += p.max(min_prob).ln();
            self.branch_advance_state(branch, bit);
        }
        logp
    }

    /// Update by one byte without returning its score.
    pub fn update_byte(&mut self, byte: u8, learn: bool) {
        if learn {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                self.observe_bit(bit, true);
            }
        } else {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                self.advance_state(bit);
            }
        }
    }

    /// Fill a byte PDF by exact MSB-first traversal of the bit model.
    pub fn fill_byte_pdf(&self, out: &mut [f64; 256], min_prob: f64) {
        let mut log_probs = [0.0f64; 256];
        self.fill_byte_log_probs(&mut log_probs, min_prob);
        for (slot, &logp) in out.iter_mut().zip(log_probs.iter()) {
            *slot = logp.exp().clamp(min_prob, 1.0);
        }
        normalize_pdf(out, min_prob);
    }

    /// Fill exact true bit-online byte log-probabilities by sharing MSB-first
    /// prefixes.
    ///
    /// Branches clone only dynamic recurrent/history state and keep a tiny
    /// overlay for sparse history-weight updates. This preserves exact
    /// intra-byte learning, including cross-phase sparse hash collisions,
    /// without copying the large learned parameter tables.
    pub fn fill_byte_log_probs(&self, out: &mut [f64; 256], min_prob: f64) {
        debug_assert_eq!(PHASES, 8);
        let mut paths = Vec::with_capacity(256);
        let mut next_paths = Vec::with_capacity(256);
        paths.push((BitReservoirByteBranch::from_model(self), 0.0f64));
        for _ in 0..8usize {
            next_paths.clear();
            for (branch, logp) in paths.drain(..) {
                let p1 = self.branch_predict_prob_one(&branch);

                let mut zero = branch.clone();
                self.branch_observe_bit_with_prob(&mut zero, false, p1 as f32);
                next_paths.push((zero, logp + (1.0 - p1).max(min_prob).ln()));

                let mut one = branch;
                self.branch_observe_bit_with_prob(&mut one, true, p1 as f32);
                next_paths.push((one, logp + p1.max(min_prob).ln()));
            }
            std::mem::swap(&mut paths, &mut next_paths);
        }
        for (slot, (_, logp)) in out.iter_mut().zip(paths.into_iter()) {
            *slot = logp;
        }
    }

    /// Fill exact frozen byte log-probabilities by sharing MSB-first prefixes.
    ///
    /// This advances only recurrent/conditioning state inside each candidate
    /// branch. It intentionally does not apply sparse readout updates while
    /// scoring, matching [`Self::update_byte`] with `learn = false`.
    pub fn fill_byte_log_probs_frozen(&self, out: &mut [f64; 256], min_prob: f64) {
        debug_assert_eq!(PHASES, 8);
        let mut paths = Vec::with_capacity(256);
        let mut next_paths = Vec::with_capacity(256);
        paths.push((BitReservoirByteBranch::from_model(self), 0.0f64));
        for _ in 0..8usize {
            next_paths.clear();
            for (branch, logp) in paths.drain(..) {
                let p1 = self.branch_predict_prob_one(&branch);

                let mut zero = branch.clone();
                self.branch_advance_state(&mut zero, false);
                next_paths.push((zero, logp + (1.0 - p1).max(min_prob).ln()));

                let mut one = branch;
                self.branch_advance_state(&mut one, true);
                next_paths.push((one, logp + p1.max(min_prob).ln()));
            }
            std::mem::swap(&mut paths, &mut next_paths);
        }
        for (slot, (_, logp)) in out.iter_mut().zip(paths.into_iter()) {
            *slot = logp;
        }
    }

    /// Score and learn a byte slice, returning total code length in bits.
    pub fn update_and_score_bits(&mut self, data: &[u8]) -> f64 {
        let mut bits = 0.0f64;
        for &byte in data {
            bits -= self.log_prob_update_byte(byte, true) / std::f64::consts::LN_2;
        }
        bits
    }

    #[inline]
    fn logit(&self) -> f32 {
        let phase = self.phase;
        let hidden = self.config.hidden;
        let mut acc = self.bias[phase];
        let out_base = phase * hidden;
        for (&w, &s) in self.out_w[out_base..out_base + hidden]
            .iter()
            .zip(self.state.iter())
        {
            acc += w * s;
        }

        let delay_bits = self.config.delay_bits;
        let delay_base = phase * delay_bits;
        let mut history_index = self.history_pos;
        for age in 0..self.history_len {
            if history_index == 0 {
                history_index = delay_bits;
            }
            history_index -= 1;
            acc += self.delay_w[delay_base + age] * self.history[history_index];
        }
        for group in 0..TOTAL_SPARSE_FEATURES {
            let idx = self.history_feature_index(group);
            acc += history_weight_at(self.history_w.as_slice(), idx) * SPARSE_FEATURE_SCALE;
        }
        acc.clamp(-LOGIT_CLIP, LOGIT_CLIP)
    }

    #[inline]
    fn logit_with_sparse_indices(
        &self,
        sparse_indices: &mut [usize; TOTAL_SPARSE_FEATURES],
    ) -> f32 {
        let phase = self.phase;
        let hidden = self.config.hidden;
        let mut acc = self.bias[phase];
        let out_base = phase * hidden;
        for (&w, &s) in self.out_w[out_base..out_base + hidden]
            .iter()
            .zip(self.state.iter())
        {
            acc += w * s;
        }

        let delay_bits = self.config.delay_bits;
        let delay_base = phase * delay_bits;
        let mut history_index = self.history_pos;
        for age in 0..self.history_len {
            if history_index == 0 {
                history_index = delay_bits;
            }
            history_index -= 1;
            acc += self.delay_w[delay_base + age] * self.history[history_index];
        }
        for (group, slot) in sparse_indices.iter_mut().enumerate() {
            *slot = self.history_feature_index(group);
        }
        for &idx in sparse_indices.iter() {
            acc += history_weight_at(self.history_w.as_slice(), idx) * SPARSE_FEATURE_SCALE;
        }
        acc.clamp(-LOGIT_CLIP, LOGIT_CLIP)
    }

    #[inline]
    fn branch_predict_prob_one(&self, branch: &BitReservoirByteBranch) -> f64 {
        sigmoid(self.branch_logit(branch)) as f64
    }

    #[inline]
    fn branch_logit(&self, branch: &BitReservoirByteBranch) -> f32 {
        let phase = branch.phase;
        let hidden = self.config.hidden;
        let mut acc = self.bias[phase];
        let out_base = phase * hidden;
        for (&w, &s) in self.out_w[out_base..out_base + hidden]
            .iter()
            .zip(branch.state.iter())
        {
            acc += w * s;
        }

        let delay_bits = self.config.delay_bits;
        let delay_base = phase * delay_bits;
        let mut history_index = branch.history_pos;
        for age in 0..branch.history_len {
            if history_index == 0 {
                history_index = delay_bits;
            }
            history_index -= 1;
            acc += self.delay_w[delay_base + age] * branch.history[history_index];
        }
        for group in 0..TOTAL_SPARSE_FEATURES {
            let idx = self.branch_history_feature_index(branch, group);
            acc += branch.history_weight(self, idx) * SPARSE_FEATURE_SCALE;
        }
        acc.clamp(-LOGIT_CLIP, LOGIT_CLIP)
    }

    fn branch_observe_bit_with_prob(
        &self,
        branch: &mut BitReservoirByteBranch,
        bit: bool,
        p1: f32,
    ) {
        self.branch_train_sparse_readout(branch, bit, p1);
        self.branch_advance_state(branch, bit);
    }

    fn branch_train_sparse_readout(&self, branch: &mut BitReservoirByteBranch, bit: bool, p1: f32) {
        let phase = branch.phase;
        let target = if bit { 1.0f32 } else { 0.0f32 };
        let error = (target - p1).clamp(-self.grad_clip, self.grad_clip);
        let eta = self.learning_rate_for_phase_step(branch.phase_steps[phase]);
        branch.phase_steps[phase] = branch.phase_steps[phase].saturating_add(1);
        let shrink = 1.0f32 - (eta * self.weight_decay).min(0.25);

        // A byte visits each of the eight dense readout phases exactly once, so
        // dense phase-local updates cannot influence another bit in the same
        // candidate byte. Sparse history weights are hash-indexed rather than
        // phase-segmented, so cross-phase collisions must be overlaid exactly.
        let mut sparse_indices = [0usize; TOTAL_SPARSE_FEATURES];
        for (group, slot) in sparse_indices.iter_mut().enumerate() {
            *slot = self.branch_history_feature_index(branch, group);
        }
        for idx in sparse_indices {
            let weight = branch.history_weight(self, idx);
            branch.set_history_weight(idx, weight * shrink + eta * error * SPARSE_FEATURE_SCALE);
        }
    }

    #[inline]
    fn train_readout_with_prediction(&mut self, bit: bool, prediction: &BitReservoirPrediction) {
        let phase = self.phase;
        let target = if bit { 1.0f32 } else { 0.0f32 };
        let error = (target - prediction.p1).clamp(-self.grad_clip, self.grad_clip);
        let eta = self.learning_rate_for_phase(phase);
        self.phase_steps[phase] = self.phase_steps[phase].saturating_add(1);

        self.bias[phase] += eta * error;

        let hidden = self.config.hidden;
        let out_base = phase * hidden;
        let shrink = 1.0f32 - (eta * self.weight_decay).min(0.25);
        let out_w = Arc::make_mut(&mut self.out_w);
        for (weight, &feature) in out_w[out_base..out_base + hidden]
            .iter_mut()
            .zip(self.state.iter())
        {
            *weight = *weight * shrink + eta * error * feature;
        }

        let delay_bits = self.config.delay_bits;
        let delay_base = phase * delay_bits;
        let delay_w = Arc::make_mut(&mut self.delay_w);
        let mut history_index = self.history_pos;
        for age in 0..self.history_len {
            if history_index == 0 {
                history_index = delay_bits;
            }
            history_index -= 1;
            let feature = self.history[history_index];
            let weight = &mut delay_w[delay_base + age];
            *weight = *weight * shrink + eta * error * feature;
        }
        let history_w = Arc::make_mut(&mut self.history_w);
        for &idx in &prediction.sparse_indices {
            let weight = history_weight_at_mut(history_w.as_mut_slice(), idx);
            *weight = *weight * shrink + eta * error * SPARSE_FEATURE_SCALE;
        }
    }

    #[inline]
    fn learning_rate_for_phase(&self, phase: usize) -> f32 {
        self.learning_rate_for_phase_step(self.phase_steps[phase])
    }

    #[inline]
    fn learning_rate_for_phase_step(&self, phase_step: u64) -> f32 {
        let t = phase_step as f64;
        let denom = (1.0 + self.config.learning_rate_decay * t).sqrt();
        (self.config.learning_rate / denom) as f32
    }

    #[inline]
    fn advance_state(&mut self, bit: bool) {
        let x = if bit { 1.0f32 } else { -1.0f32 };
        let hidden = self.config.hidden;
        let phase_base = self.phase * hidden;
        for i in 0..hidden {
            let recurrent = self.params.recurrent[i] * self.state[self.params.source[i]];
            let raw = self.params.decay[i] * self.state[i]
                + recurrent
                + self.params.input[i] * x
                + self.params.phase[phase_base + i];
            self.scratch[i] = softsign(raw);
        }
        std::mem::swap(&mut self.state, &mut self.scratch);

        let delay_bits = self.config.delay_bits;
        self.history[self.history_pos] = x;
        self.history_pos += 1;
        if self.history_pos == delay_bits {
            self.history_pos = 0;
        }
        if self.history_len < delay_bits {
            self.history_len += 1;
        }
        self.history_bits = (self.history_bits << 1) | u64::from(bit);
        if self.history_bit_len < 64 {
            self.history_bit_len += 1;
        }
        self.partial_byte = (self.partial_byte << 1) | u8::from(bit);
        self.partial_bits += 1;
        if self.partial_bits == 8 {
            self.byte_history = (self.byte_history << 8) | u64::from(self.partial_byte);
            if self.byte_history_len < 8 {
                self.byte_history_len += 1;
            }
            self.observe_completed_byte(self.partial_byte);
            self.partial_byte = 0;
            self.partial_bits = 0;
        }
        self.phase = (self.phase + 1) & 7;
    }

    #[inline]
    fn branch_advance_state(&self, branch: &mut BitReservoirByteBranch, bit: bool) {
        let x = if bit { 1.0f32 } else { -1.0f32 };
        let hidden = self.config.hidden;
        let phase_base = branch.phase * hidden;
        for i in 0..hidden {
            let recurrent = self.params.recurrent[i] * branch.state[self.params.source[i]];
            let raw = self.params.decay[i] * branch.state[i]
                + recurrent
                + self.params.input[i] * x
                + self.params.phase[phase_base + i];
            branch.scratch[i] = softsign(raw);
        }
        std::mem::swap(&mut branch.state, &mut branch.scratch);

        let delay_bits = self.config.delay_bits;
        branch.history[branch.history_pos] = x;
        branch.history_pos += 1;
        if branch.history_pos == delay_bits {
            branch.history_pos = 0;
        }
        if branch.history_len < delay_bits {
            branch.history_len += 1;
        }
        branch.history_bits = (branch.history_bits << 1) | u64::from(bit);
        if branch.history_bit_len < 64 {
            branch.history_bit_len += 1;
        }
        branch.partial_byte = (branch.partial_byte << 1) | u8::from(bit);
        branch.partial_bits += 1;
        if branch.partial_bits == 8 {
            branch.byte_history = (branch.byte_history << 8) | u64::from(branch.partial_byte);
            if branch.byte_history_len < 8 {
                branch.byte_history_len += 1;
            }
            self.branch_observe_completed_byte(branch, branch.partial_byte);
            branch.partial_byte = 0;
            branch.partial_bits = 0;
        }
        branch.phase = (branch.phase + 1) & 7;
    }

    #[inline]
    fn history_feature_index(&self, group: usize) -> usize {
        history_feature_index_from_state(
            self.phase,
            self.history_bits,
            self.history_bit_len,
            self.byte_history,
            self.byte_history_len,
            self.partial_byte,
            self.partial_bits,
            group,
            self.history_mask,
            self.config.seed,
        )
    }

    #[inline]
    fn branch_history_feature_index(&self, branch: &BitReservoirByteBranch, group: usize) -> usize {
        history_feature_index_from_state(
            branch.phase,
            branch.history_bits,
            branch.history_bit_len,
            branch.byte_history,
            branch.byte_history_len,
            branch.partial_byte,
            branch.partial_bits,
            group,
            self.history_mask,
            self.config.seed,
        )
    }

    #[inline]
    fn observe_completed_byte(&mut self, _byte: u8) {}

    #[inline]
    fn branch_observe_completed_byte(&self, _branch: &mut BitReservoirByteBranch, _byte: u8) {}
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn history_feature_index_from_state(
    phase: usize,
    history_bits: u64,
    history_bit_len: usize,
    byte_history: u64,
    byte_history_len: usize,
    partial_byte: u8,
    partial_bits: usize,
    group: usize,
    history_mask: usize,
    seed: u64,
) -> usize {
    let (context, effective_width) = if group < HISTORY_WINDOWS.len() {
        let width = HISTORY_WINDOWS[group];
        let effective_width = width.min(history_bit_len);
        let context = if effective_width == 64 {
            history_bits
        } else if effective_width == 0 {
            0
        } else {
            history_bits & ((1u64 << effective_width) - 1)
        };
        (context, effective_width)
    } else {
        let byte_group = group - HISTORY_WINDOWS.len();
        let width = BYTE_WINDOWS[byte_group];
        let effective_width = width.min(byte_history_len);
        let byte_bits = effective_width * 8;
        let context = if byte_bits == 64 {
            byte_history
        } else if byte_bits == 0 {
            0
        } else {
            byte_history & ((1u64 << byte_bits) - 1)
        };
        let partial = u64::from(partial_byte) << 3 | partial_bits as u64;
        (context ^ partial.rotate_left(41), byte_bits + partial_bits)
    };
    let key = context
        ^ ((effective_width as u64) << 48)
        ^ ((phase as u64) << 56)
        ^ ((group as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ seed.rotate_left((group as u32) & 31);
    group * (history_mask + 1) + (mix64(key) as usize & history_mask)
}

impl BitReservoirParams {
    fn new(config: &BitReservoirConfig) -> Self {
        let hidden = config.hidden;
        let mut rng = SplitMix64::new(config.seed);
        let mut source = Vec::with_capacity(hidden);
        let mut recurrent = Vec::with_capacity(hidden);
        let mut input = Vec::with_capacity(hidden);
        let mut decay = Vec::with_capacity(hidden);
        let mut phase = Vec::with_capacity(PHASES * hidden);

        for i in 0..hidden {
            let src_offset = 1 + (rng.next_usize(hidden.saturating_sub(1).max(1)) % hidden);
            source.push((i + src_offset) % hidden);
            recurrent.push(rng.uniform_symmetric(config.recurrent_scale as f32));
            input.push(rng.uniform_symmetric(config.input_scale as f32));
            let jitter = 0.85f32 + 0.15f32 * rng.next_f32();
            decay.push((config.state_decay as f32 * jitter).clamp(0.0, 0.999));
        }
        for _ in 0..PHASES * hidden {
            phase.push(rng.uniform_symmetric(config.phase_scale as f32));
        }

        Self {
            source: source.into_boxed_slice(),
            recurrent: recurrent.into_boxed_slice(),
            input: input.into_boxed_slice(),
            decay: decay.into_boxed_slice(),
            phase: phase.into_boxed_slice(),
        }
    }
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        let raw = self.next_u64() >> 40;
        (raw as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    #[inline]
    fn uniform_symmetric(&mut self, scale: f32) -> f32 {
        (2.0 * self.next_f32() - 1.0) * scale
    }
}

#[inline]
fn history_slots(config: &BitReservoirConfig) -> usize {
    1usize << config.embedding_bits
}

#[inline]
fn history_weight_at(weights: &[f32], idx: usize) -> f32 {
    debug_assert!(idx < weights.len());
    // SAFETY: Sparse indices are only produced by `history_feature_index_from_state`
    // for `group < TOTAL_SPARSE_FEATURES`. That helper returns
    // `group * history_slots + masked_hash`, where `masked_hash < history_slots`
    // and `weights.len() == TOTAL_SPARSE_FEATURES * history_slots`.
    unsafe { *weights.get_unchecked(idx) }
}

#[inline]
fn history_weight_at_mut(weights: &mut [f32], idx: usize) -> &mut f32 {
    debug_assert!(idx < weights.len());
    // SAFETY: See `history_weight_at`; prediction sparse indices are captured
    // from the same bounded helper before being used for the corresponding
    // online update.
    unsafe { weights.get_unchecked_mut(idx) }
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-LOGIT_CLIP, LOGIT_CLIP);
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

fn normalize_pdf(pdf: &mut [f64; 256], min_prob: f64) {
    let mut sum = 0.0f64;
    for p in pdf.iter_mut() {
        if !p.is_finite() || *p < min_prob {
            *p = min_prob;
        }
        sum += *p;
    }
    if !sum.is_finite() || sum <= 0.0 {
        pdf.fill(1.0 / 256.0);
        return;
    }
    for p in pdf.iter_mut() {
        *p /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reservoir_probabilities_stay_finite() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 16,
            embedding_bits: 10,
            ..BitReservoirConfig::default()
        };
        let mut model = BitReservoirModel::new(config).expect("model");
        for &byte in b"abracadabra" {
            let logp = model.log_prob_update_byte(byte, true);
            assert!(logp.is_finite());
        }
        let p1 = model.predict_prob_one();
        assert!(p1.is_finite() && p1 > 0.0 && p1 < 1.0);
    }

    #[test]
    fn bit_reservoir_reset_state_preserves_predictions_deterministically() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 8,
            embedding_bits: 10,
            ..BitReservoirConfig::default()
        };
        let mut left = BitReservoirModel::new(config.clone()).expect("left");
        let mut right = BitReservoirModel::new(config).expect("right");
        for &byte in b"training bytes" {
            left.log_prob_update_byte(byte, true);
            right.log_prob_update_byte(byte, true);
        }
        left.reset_state_only();
        right.reset_state_only();
        assert_eq!(
            left.log_prob_update_byte(b'x', false).to_bits(),
            right.log_prob_update_byte(b'x', false).to_bits()
        );
    }

    #[test]
    fn bit_reservoir_fill_byte_pdf_matches_bruteforce_traversal() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 12,
            embedding_bits: 10,
            ..BitReservoirConfig::default()
        };
        let mut model = BitReservoirModel::new(config).expect("model");
        for &byte in b"adaptive bit pdf" {
            model.log_prob_update_byte(byte, true);
        }

        let min_prob = 1.0e-12;
        let mut expected = [0.0f64; 256];
        for (symbol, slot) in expected.iter_mut().enumerate() {
            let mut probe = model.clone();
            let logp = manual_bit_online_score_byte(&mut probe, symbol as u8, min_prob);
            *slot = logp.exp().clamp(min_prob, 1.0);
        }
        normalize_pdf(&mut expected, min_prob);

        let mut actual = [0.0f64; 256];
        model.fill_byte_pdf(&mut actual, min_prob);
        for (&left, &right) in actual.iter().zip(expected.iter()) {
            assert!((left - right).abs() <= 1.0e-15);
        }
    }

    #[test]
    fn bit_reservoir_fill_byte_log_probs_matches_symbol_queries() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 12,
            embedding_bits: 10,
            ..BitReservoirConfig::default()
        };
        let mut model = BitReservoirModel::new(config).expect("model");
        for &byte in b"adaptive bit log row" {
            model.log_prob_update_byte(byte, true);
        }

        let min_prob = 1.0e-12;
        let mut row = [0.0f64; 256];
        model.fill_byte_log_probs(&mut row, min_prob);
        for (symbol, &logp) in row.iter().enumerate() {
            let mut probe = model.clone();
            let queried = manual_bit_online_score_byte(&mut probe, symbol as u8, min_prob);
            assert!((logp - queried).abs() <= 1.0e-15);
        }
    }

    #[test]
    fn bit_reservoir_byte_scoring_matches_native_bitwise_learning() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 12,
            embedding_bits: 10,
            ..BitReservoirConfig::default()
        };
        let mut model = BitReservoirModel::new(config).expect("model");
        for &byte in b"native bitwise equivalence prompt" {
            model.log_prob_update_byte(byte, true);
        }

        let mut byte_model = model.clone();
        let mut bit_model = model.clone();
        let byte_logp = byte_model.log_prob_update_byte_with_min_prob(0b1011_0010, true, 1.0e-12);
        let bit_logp = manual_bit_online_score_byte(&mut bit_model, 0b1011_0010, 1.0e-12);

        assert!((byte_logp - bit_logp).abs() <= 1.0e-15);
        assert_eq!(
            byte_model.predict_prob_one().to_bits(),
            bit_model.predict_prob_one().to_bits()
        );
    }

    #[test]
    fn bit_reservoir_update_and_score_matches_native_bitwise_stream() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 12,
            embedding_bits: 10,
            ..BitReservoirConfig::default()
        };
        let data = b"native bitwise stream equivalence";
        let mut byte_model = BitReservoirModel::new(config.clone()).expect("byte model");
        let mut bit_model = BitReservoirModel::new(config).expect("bit model");

        let scored_bits = byte_model.update_and_score_bits(data);
        let mut manual_bits = 0.0f64;
        for &byte in data {
            manual_bits -= manual_bit_online_score_byte(
                &mut bit_model,
                byte,
                crate::mixture::DEFAULT_MIN_PROB,
            ) / std::f64::consts::LN_2;
        }

        assert!((scored_bits - manual_bits).abs() <= 1.0e-12);
        assert_eq!(
            byte_model.predict_prob_one().to_bits(),
            bit_model.predict_prob_one().to_bits()
        );
    }

    #[test]
    fn bit_reservoir_frozen_byte_scoring_matches_native_no_learn_bits() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 12,
            embedding_bits: 8,
            learning_rate: 0.05,
            learning_rate_decay: 0.0,
            weight_decay: 0.0,
            ..BitReservoirConfig::default()
        };
        let mut model = BitReservoirModel::new(config).expect("model");
        for &byte in b"frozen sparse collision prompt" {
            model.log_prob_update_byte(byte, true);
        }

        let min_prob = 1.0e-12;
        let symbol = 0b1011_0110;
        let mut direct = model.clone();
        let expected = manual_bit_frozen_score_byte(&mut direct, symbol, min_prob);
        let actual = model.log_prob_byte_frozen_with_min_prob(symbol, min_prob);

        assert!((actual - expected).abs() <= 1.0e-15);
        let mut row = [0.0f64; 256];
        model.fill_byte_log_probs_frozen(&mut row, min_prob);
        assert!((row[symbol as usize] - expected).abs() <= 1.0e-15);
    }

    #[test]
    fn bit_reservoir_rejects_f64_values_outside_internal_numeric_domain() {
        let config = BitReservoirConfig {
            learning_rate: 1.0e100,
            ..BitReservoirConfig::default()
        };
        assert!(BitReservoirModel::new(config).is_err());
    }

    #[test]
    fn bit_reservoir_clone_training_detaches_learned_weights() {
        let config = BitReservoirConfig {
            hidden: 8,
            delay_bits: 8,
            embedding_bits: 10,
            ..BitReservoirConfig::default()
        };
        let mut original = BitReservoirModel::new(config).expect("original");
        for &byte in b"shared weights" {
            original.log_prob_update_byte(byte, true);
        }

        let mut branch = original.clone();
        assert!(std::sync::Arc::ptr_eq(&original.out_w, &branch.out_w));
        assert!(std::sync::Arc::ptr_eq(&original.delay_w, &branch.delay_w));
        assert!(std::sync::Arc::ptr_eq(
            &original.history_w,
            &branch.history_w
        ));
        let before = original.log_prob_byte(b'?').to_bits();
        branch.log_prob_update_byte(b'z', true);
        let after = original.log_prob_byte(b'?').to_bits();

        assert_eq!(before, after);
        assert!(!std::sync::Arc::ptr_eq(&original.out_w, &branch.out_w));
        assert!(!std::sync::Arc::ptr_eq(&original.delay_w, &branch.delay_w));
        assert!(!std::sync::Arc::ptr_eq(
            &original.history_w,
            &branch.history_w
        ));
    }

    fn manual_bit_online_score_byte(model: &mut BitReservoirModel, byte: u8, min_prob: f64) -> f64 {
        let mut logp = 0.0f64;
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            let p1 = model.predict_prob_one();
            let p = if bit { p1 } else { 1.0 - p1 };
            logp += p.max(min_prob).ln();
            model.observe_bit(bit, true);
        }
        logp
    }

    fn manual_bit_frozen_score_byte(model: &mut BitReservoirModel, byte: u8, min_prob: f64) -> f64 {
        let mut logp = 0.0f64;
        for bit_idx in 0..8usize {
            let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
            let p1 = model.predict_prob_one();
            let p = if bit { p1 } else { 1.0 - p1 };
            logp += p.max(min_prob).ln();
            model.observe_bit(bit, false);
        }
        logp
    }
}
