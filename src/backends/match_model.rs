use ahash::AHashMap;

#[derive(Clone, Debug)]
/// Local match predictor with configurable contiguous or gapped matching.
pub struct MatchModel {
    hash_bits: usize,
    min_len: usize,
    max_len: usize,
    stride_min: usize,
    stride_max: usize,
    base_mix: f64,
    confidence_scale: f64,
    history: Vec<u8>,
    tables: Vec<AHashMap<u64, (usize, usize)>>,
    pdf: [f64; 256],
    cdf: [f64; 257],
    valid: bool,
    cdf_valid: bool,
    predicted: Option<u8>,
    match_len: usize,
}

impl MatchModel {
    /// Create a match model with an inclusive stride range `[gap_min+1, gap_max+1]`.
    pub fn new(
        hash_bits: usize,
        min_len: usize,
        max_len: usize,
        gap_min: usize,
        gap_max: usize,
        base_mix: f64,
        confidence_scale: f64,
    ) -> Self {
        let stride_min = gap_min.saturating_add(1);
        let stride_max = gap_max.saturating_add(1).max(stride_min);
        let mut tables = Vec::new();
        for _ in stride_min..=stride_max {
            tables.push(AHashMap::new());
        }
        Self {
            hash_bits,
            min_len: min_len.max(1),
            max_len: max_len.max(min_len.max(1)),
            stride_min,
            stride_max,
            base_mix: base_mix.clamp(1e-6, 0.99),
            confidence_scale: confidence_scale.max(0.0),
            history: Vec::new(),
            tables,
            pdf: [1.0 / 256.0; 256],
            cdf: uniform_cdf(),
            valid: false,
            cdf_valid: false,
            predicted: None,
            match_len: 0,
        }
    }

    /// Convenience constructor for contiguous matching (`gap_min = gap_max = 0`).
    pub fn new_contiguous(
        hash_bits: usize,
        min_len: usize,
        max_len: usize,
        base_mix: f64,
        confidence_scale: f64,
    ) -> Self {
        Self::new(
            hash_bits,
            min_len,
            max_len,
            0,
            0,
            base_mix,
            confidence_scale,
        )
    }

    /// Fill `out` with the current normalized byte PDF.
    pub fn fill_pdf(&mut self, out: &mut [f64; 256]) {
        self.ensure_pdf_inner(false);
        out.copy_from_slice(&self.pdf);
    }

    /// Borrow the current normalized byte PDF.
    pub fn pdf(&mut self) -> &[f64; 256] {
        self.ensure_pdf_inner(false);
        &self.pdf
    }

    /// Borrow the cumulative distribution derived from the current PDF.
    pub fn cdf(&mut self) -> &[f64; 257] {
        self.ensure_pdf_inner(true);
        &self.cdf
    }

    /// Return `ln(max(P(symbol), min_prob))`.
    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.ensure_pdf_inner(false);
        self.pdf[symbol as usize].max(min_prob).ln()
    }

    /// Observe one symbol and update match tables/history.
    pub fn update(&mut self, symbol: u8) {
        self.history.push(symbol);
        for stride in self.stride_min..=self.stride_max {
            if let Some(key) = self.suffix_key(stride) {
                let end = self.history.len() - 1;
                self.tables[stride - self.stride_min]
                    .entry(key)
                    .and_modify(|entry| {
                        entry.1 = entry.0;
                        entry.0 = end;
                    })
                    .or_insert((end, usize::MAX));
            }
        }
        self.valid = false;
        self.cdf_valid = false;
    }

    /// Reset only the conditioning history while preserving learned match tables.
    pub fn reset_history(&mut self) {
        self.history.clear();
        self.valid = false;
        self.cdf_valid = false;
        self.predicted = None;
        self.match_len = 0;
        self.pdf.fill(1.0 / 256.0);
        self.cdf = uniform_cdf();
    }

    /// Advance conditioning history without updating learned match tables.
    pub fn update_history_only(&mut self, symbol: u8) {
        self.history.push(symbol);
        self.valid = false;
        self.cdf_valid = false;
    }

    /// Length of the best match used for the last computed distribution.
    pub fn match_len(&mut self) -> usize {
        self.ensure_pdf_inner(false);
        self.match_len
    }

    /// Predicted next byte from the best match, if any.
    pub fn predicted_byte(&mut self) -> Option<u8> {
        self.ensure_pdf_inner(false);
        self.predicted
    }

    fn ensure_pdf_inner(&mut self, want_cdf: bool) {
        if self.valid {
            if want_cdf && !self.cdf_valid {
                build_cdf_from_pdf(&self.pdf, &mut self.cdf);
                self.cdf_valid = true;
            }
            return;
        }
        self.predicted = None;
        self.match_len = 0;
        self.pdf.fill(1.0 / 256.0);
        if self.history.len() < self.min_len {
            self.valid = true;
            if want_cdf {
                self.cdf = uniform_cdf();
                self.cdf_valid = true;
            } else {
                self.cdf_valid = false;
            }
            return;
        }

        let mut best = None;
        for stride in self.stride_min..=self.stride_max {
            let Some(key) = self.suffix_key(stride) else {
                continue;
            };
            let Some(&(latest, previous)) = self.tables[stride - self.stride_min].get(&key) else {
                continue;
            };
            let current_end = self.history.len() - 1;
            let candidate_end = if latest == current_end {
                previous
            } else {
                latest
            };
            if candidate_end == usize::MAX || candidate_end + stride >= self.history.len() {
                continue;
            }
            let matched = self.extend_match(candidate_end, stride);
            if matched < self.min_len {
                continue;
            }
            let predicted = self.history[candidate_end + stride];
            match best {
                Some((best_len, _, _)) if matched <= best_len => {}
                _ => best = Some((matched, predicted, stride)),
            }
        }

        if let Some((match_len, predicted, _stride)) = best {
            self.predicted = Some(predicted);
            self.match_len = match_len;
            let base = 1.0 / 256.0;
            let span = self.max_len.saturating_sub(self.min_len).max(1);
            let covered = match_len.saturating_sub(self.min_len).min(span);
            let confidence = ((covered as f64) / (span as f64)).sqrt() * self.confidence_scale;
            let p_copy = (base + confidence.clamp(0.0, 1.0) * ((1.0 - self.base_mix) - base))
                .clamp(base, 1.0 - self.base_mix);
            let rest = ((1.0 - p_copy) / 255.0).max(0.0);
            self.pdf.fill(rest);
            self.pdf[predicted as usize] = p_copy;
        }
        if want_cdf {
            build_cdf_from_pdf(&self.pdf, &mut self.cdf);
        }
        self.valid = true;
        self.cdf_valid = want_cdf;
    }

    fn suffix_key(&self, stride: usize) -> Option<u64> {
        let need = self
            .min_len
            .checked_sub(1)?
            .saturating_mul(stride)
            .saturating_add(1);
        if self.history.len() < need {
            return None;
        }
        let mut h = 0x517C_C1B7_2722_0A95u64;
        let mut idx = self.history.len() - 1;
        for _ in 0..self.min_len {
            h ^= self.history[idx] as u64;
            h = h.rotate_left(7).wrapping_mul(0x9E37_79B1);
            if idx < stride {
                break;
            }
            idx -= stride;
        }
        let bits = self.hash_bits.clamp(4, 63);
        Some(h & ((1u64 << bits) - 1))
    }

    fn extend_match(&self, candidate_end: usize, stride: usize) -> usize {
        let current_end = self.history.len() - 1;
        let mut matched = self.min_len;
        while matched < self.max_len
            && current_end >= matched * stride
            && candidate_end >= matched * stride
            && self.history[current_end - matched * stride]
                == self.history[candidate_end - matched * stride]
        {
            matched += 1;
        }
        matched
    }
}

#[inline]
fn uniform_cdf() -> [f64; 257] {
    let mut cdf = [0.0; 257];
    let inv = 1.0 / 256.0;
    for (i, slot) in cdf.iter_mut().enumerate() {
        *slot = (i as f64) * inv;
    }
    cdf
}

#[inline]
fn build_cdf_from_pdf(pdf: &[f64; 256], cdf: &mut [f64; 257]) {
    cdf[0] = 0.0;
    for i in 0..256 {
        cdf[i + 1] = cdf[i] + pdf[i];
    }
}
