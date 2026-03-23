use ahash::AHashMap;
use std::collections::VecDeque;

const PDF_MIN: f64 = crate::mixture::DEFAULT_MIN_PROB;

#[derive(Clone, Debug, Default)]
struct ContextStats {
    counts: Vec<(u8, u16)>,
    total: u32,
}

impl ContextStats {
    fn observe(&mut self, symbol: u8) {
        if let Some((_, count)) = self.counts.iter_mut().find(|(s, _)| *s == symbol) {
            *count = count.saturating_add(1);
        } else {
            self.counts.push((symbol, 1));
        }
        self.total = self.total.saturating_add(1);
        if self.total > 4096 {
            self.rescale();
        }
    }

    fn rescale(&mut self) {
        self.total = 0;
        self.counts.retain_mut(|(_, count)| {
            *count = (*count).div_ceil(2).max(1);
            self.total += *count as u32;
            true
        });
    }
}

#[derive(Clone, Debug)]
pub struct PpmdModel {
    order: usize,
    max_contexts: usize,
    contexts: Vec<AHashMap<u64, ContextStats>>,
    queue: VecDeque<(usize, u64)>,
    history: Vec<u8>,
    pdf: [f64; 256],
    cdf: [f64; 257],
    valid: bool,
    cdf_valid: bool,
}

impl PpmdModel {
    pub fn new(order: usize, memory_mb: usize) -> Self {
        let order = order.max(1);
        let max_contexts = (memory_mb.max(1) * 1024 * 1024) / 96;
        Self {
            order,
            max_contexts: max_contexts.max(1024),
            contexts: (0..=order).map(|_| AHashMap::new()).collect(),
            queue: VecDeque::new(),
            history: Vec::new(),
            pdf: [1.0 / 256.0; 256],
            cdf: uniform_cdf(),
            valid: false,
            cdf_valid: false,
        }
    }

    pub fn fill_pdf(&mut self, out: &mut [f64; 256]) {
        self.ensure_pdf_inner(false);
        out.copy_from_slice(&self.pdf);
    }

    pub fn pdf(&mut self) -> &[f64; 256] {
        self.ensure_pdf_inner(false);
        &self.pdf
    }

    pub fn cdf(&mut self) -> &[f64; 257] {
        self.ensure_pdf_inner(true);
        &self.cdf
    }

    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.ensure_pdf_inner(false);
        self.pdf[symbol as usize].max(min_prob).ln()
    }

    pub fn update(&mut self, symbol: u8) {
        let max_order = self.order.min(self.history.len());
        for ord in 0..=max_order {
            let key = self.context_key(ord);
            let map = &mut self.contexts[ord];
            if !map.contains_key(&key) {
                map.insert(key, ContextStats::default());
                self.queue.push_back((ord, key));
            }
            if let Some(ctx) = map.get_mut(&key) {
                ctx.observe(symbol);
            }
        }
        self.prune();
        self.history.push(symbol);
        self.valid = false;
        self.cdf_valid = false;
    }

    /// Reset only the conditioning history while preserving fitted contexts.
    pub fn reset_history(&mut self) {
        self.history.clear();
        self.valid = false;
        self.cdf_valid = false;
        self.pdf.fill(1.0 / 256.0);
        self.cdf = uniform_cdf();
    }

    /// Advance conditioning history without updating fitted context counts.
    pub fn update_history_only(&mut self, symbol: u8) {
        self.history.push(symbol);
        self.valid = false;
        self.cdf_valid = false;
    }

    fn ensure_pdf_inner(&mut self, want_cdf: bool) {
        if self.valid {
            if want_cdf && !self.cdf_valid {
                build_cdf_from_pdf(&self.pdf, &mut self.cdf);
                self.cdf_valid = true;
            }
            return;
        }
        let mut lower = [1.0 / 256.0; 256];
        let max_order = self.order.min(self.history.len());
        for ord in 0..=max_order {
            let key = self.context_key(ord);
            if let Some(ctx) = self.contexts[ord].get(&key) {
                lower = interpolate_context(ctx, &lower);
            }
        }
        self.pdf.copy_from_slice(&lower);
        normalize_pdf_and_maybe_cdf(
            &mut self.pdf,
            if want_cdf { Some(&mut self.cdf) } else { None },
        );
        self.valid = true;
        self.cdf_valid = want_cdf;
    }

    fn prune(&mut self) {
        let mut total_contexts: usize = self.contexts.iter().map(|m| m.len()).sum();
        while total_contexts > self.max_contexts {
            let Some((ord, key)) = self.queue.pop_front() else {
                break;
            };
            if self.contexts[ord].remove(&key).is_some() {
                total_contexts -= 1;
            }
        }
    }

    fn context_key(&self, ord: usize) -> u64 {
        if ord == 0 {
            return 0;
        }
        let start = self.history.len() - ord;
        hash_bytes(&self.history[start..])
    }
}

fn interpolate_context(ctx: &ContextStats, lower: &[f64; 256]) -> [f64; 256] {
    let distinct = ctx.counts.len() as f64;
    let denom = (ctx.total as f64) + distinct + 1.0;
    let escape = (distinct + 1.0) / denom;
    let mut out = [0.0; 256];
    for i in 0..256 {
        out[i] = lower[i] * escape;
    }
    for &(symbol, count) in &ctx.counts {
        out[symbol as usize] += (count as f64) / denom;
    }
    out
}

fn normalize_pdf_and_maybe_cdf(pdf: &mut [f64; 256], mut cdf: Option<&mut [f64; 257]>) {
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
        let u = 1.0 / 256.0;
        pdf.fill(u);
        if let Some(cdf) = cdf.as_deref_mut() {
            *cdf = uniform_cdf();
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
    let mut acc = 0.0;
    for i in 0..256 {
        acc += pdf[i];
        cdf[i + 1] = acc;
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01B3);
    }
    h
}
