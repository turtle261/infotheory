use ahash::AHashMap;
use std::collections::{VecDeque, hash_map::Entry};

const PDF_MIN: f64 = crate::mixture::DEFAULT_MIN_PROB;
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x1000_0000_01B3;
const INLINE_CONTEXT_COUNTS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CountEntry {
    symbol: u8,
    count: u16,
}

#[derive(Clone, Debug, Default)]
struct ContextStats {
    inline_counts: [CountEntry; INLINE_CONTEXT_COUNTS],
    inline_len: u8,
    spill_counts: Option<Box<[CountEntry]>>,
    total: u32,
}

impl ContextStats {
    fn distinct_len(&self) -> usize {
        self.inline_len as usize
            + self
                .spill_counts
                .as_ref()
                .map(|entries| entries.len())
                .unwrap_or(0)
    }

    #[cfg(any(test, feature = "research-tooling"))]
    fn spill_bytes(&self) -> usize {
        self.spill_counts
            .as_ref()
            .map(|entries| {
                entries
                    .len()
                    .saturating_mul(std::mem::size_of::<CountEntry>())
            })
            .unwrap_or(0)
    }

    fn for_each(&self, mut f: impl FnMut(CountEntry)) {
        for idx in 0..(self.inline_len as usize) {
            f(self.inline_counts[idx]);
        }
        if let Some(spill_counts) = &self.spill_counts {
            for &entry in spill_counts.iter() {
                f(entry);
            }
        }
    }

    fn find_mut(&mut self, symbol: u8) -> Option<&mut u16> {
        if let Some(entry) = self.inline_counts[..(self.inline_len as usize)]
            .iter_mut()
            .find(|entry| entry.symbol == symbol)
        {
            return Some(&mut entry.count);
        }
        if let Some(spill_counts) = self.spill_counts.as_mut()
            && let Some(entry) = spill_counts.iter_mut().find(|entry| entry.symbol == symbol)
        {
            return Some(&mut entry.count);
        }
        None
    }

    fn push_new(&mut self, symbol: u8) {
        let entry = CountEntry { symbol, count: 1 };
        if (self.inline_len as usize) < INLINE_CONTEXT_COUNTS {
            self.inline_counts[self.inline_len as usize] = entry;
            self.inline_len += 1;
            return;
        }

        let mut spill = Vec::with_capacity(
            self.spill_counts
                .as_ref()
                .map(|entries| entries.len())
                .unwrap_or(0)
                .saturating_add(1),
        );
        if let Some(existing) = self.spill_counts.take() {
            spill.extend_from_slice(existing.as_ref());
        }
        spill.push(entry);
        self.spill_counts = Some(spill.into_boxed_slice());
    }

    fn observe(&mut self, symbol: u8) {
        if let Some(count) = self.find_mut(symbol) {
            *count = count.saturating_add(1);
        } else {
            self.push_new(symbol);
        }
        self.total = self.total.saturating_add(1);
        if self.total > 4096 {
            self.rescale();
        }
    }

    fn rescale(&mut self) {
        self.total = 0;
        for idx in 0..(self.inline_len as usize) {
            let count = &mut self.inline_counts[idx].count;
            *count = (*count).div_ceil(2).max(1);
            self.total += *count as u32;
        }
        if let Some(spill_counts) = self.spill_counts.as_mut() {
            for entry in spill_counts.iter_mut() {
                entry.count = entry.count.div_ceil(2).max(1);
                self.total += entry.count as u32;
            }
        }
    }
}

#[cfg(any(test, feature = "research-tooling"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PpmdMemoryUsage {
    pub context_table_bytes: usize,
    pub context_spill_bytes: usize,
    pub queue_bytes: usize,
    pub history_bytes: usize,
    pub suffix_key_bytes: usize,
    pub pdf_cache_bytes: usize,
}

#[cfg(any(test, feature = "research-tooling"))]
impl PpmdMemoryUsage {
    pub(crate) fn total_bytes(self) -> usize {
        self.context_table_bytes
            + self.context_spill_bytes
            + self.queue_bytes
            + self.history_bytes
            + self.suffix_key_bytes
            + self.pdf_cache_bytes
    }
}

#[derive(Clone, Debug)]
/// Bounded-memory PPMD-inspired byte model with interpolation across orders.
pub struct PpmdModel {
    order: usize,
    max_contexts: usize,
    contexts: Vec<AHashMap<u64, ContextStats>>,
    queue: VecDeque<(usize, u64)>,
    history: Vec<u8>,
    suffix_keys: Vec<u64>,
    pdf: [f64; 256],
    cdf: [f64; 257],
    valid: bool,
    cdf_valid: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PpmdLifecycleSnapshot {
    history: Vec<u8>,
    suffix_keys: Vec<u64>,
    pdf: [f64; 256],
    cdf: [f64; 257],
    valid: bool,
    cdf_valid: bool,
}

impl PpmdModel {
    /// Create a model with maximum `order` and approximate memory budget in MiB.
    pub fn new(order: usize, memory_mb: usize) -> Self {
        let order = order.max(1);
        let max_contexts = (memory_mb.max(1) * 1024 * 1024) / 96;
        Self {
            order,
            max_contexts: max_contexts.max(1024),
            contexts: (0..=order).map(|_| AHashMap::new()).collect(),
            queue: VecDeque::new(),
            history: Vec::new(),
            suffix_keys: vec![0; order + 1],
            pdf: [1.0 / 256.0; 256],
            cdf: uniform_cdf(),
            valid: false,
            cdf_valid: false,
        }
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

    // This intentionally still routes through the dense `ensure_pdf_inner`
    // path rather than the sparse, distinct-symbol-only `SparseQueryState`
    // walk (below, test-only). The sparse walk is bit-identical (see
    // `exact_symbol_queries_match_dense_pdf`) and is asymptotically cheaper
    // per context order when few distinct symbols have been observed, but
    // measured standalone on 10 MB of enwik7 (`h_rate`, CPU-pinned,
    // `RAYON_NUM_THREADS=1`, order=12/256 MiB) it was ~18% *slower*
    // end-to-end (18.3s -> 21.6s user time) at identical bpb. Real text
    // saturates the order-0 context's distinct-symbol set to a large
    // fraction of the 256-byte alphabet almost immediately (enwik mixes
    // ASCII markup with multi-byte UTF-8), so `SparseQueryState`'s
    // touched-set is not actually small in the corpora this project cares
    // about; its branchy, index-gathered `touched_symbols` walk then loses
    // to the dense path's plain, auto-vectorizable `for p in
    // lower.iter_mut() { *p *= escape }` sweep. Keep the dense path as the
    // real `symbol_prob` implementation; do not promote `SparseQueryState`
    // to production without re-measuring on realistic (not synthetic
    // low-alphabet) input.
    pub(crate) fn symbol_prob(&mut self, symbol: u8) -> f64 {
        self.ensure_pdf_inner(false);
        self.pdf[symbol as usize]
    }

    #[cfg(test)]
    fn interval_mass(&mut self, lo: usize, hi: usize) -> f64 {
        if lo >= hi {
            return 0.0;
        }
        let lo = lo.min(256);
        let hi = hi.min(256);
        if lo >= hi {
            return 0.0;
        }
        if self.cdf_valid {
            return self.cdf[hi] - self.cdf[lo];
        }
        if self.valid {
            let mut acc_lo = 0.0;
            let mut acc_hi = 0.0;
            for i in 0..hi {
                acc_hi += self.pdf[i];
                if i + 1 == lo {
                    acc_lo = acc_hi;
                }
            }
            return acc_hi - acc_lo;
        }
        self.ensure_pdf_inner(true);
        self.cdf[hi] - self.cdf[lo]
    }

    /// Return `ln(max(P(symbol), min_prob))`.
    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.symbol_prob(symbol).max(min_prob).ln()
    }

    /// Observe one symbol and update all active contexts up to model order.
    pub fn update(&mut self, symbol: u8) {
        let max_order = self.order.min(self.history.len());
        for ord in 0..=max_order {
            let key = self.context_key(ord);
            let map = &mut self.contexts[ord];
            match map.entry(key) {
                Entry::Occupied(mut entry) => {
                    entry.get_mut().observe(symbol);
                }
                Entry::Vacant(entry) => {
                    self.queue.push_back((ord, key));
                    entry.insert(ContextStats::default()).observe(symbol);
                }
            }
        }
        self.prune();
        self.append_history_symbol(symbol);
        self.valid = false;
        self.cdf_valid = false;
    }

    /// Reset only the conditioning history while preserving fitted contexts.
    pub fn reset_history(&mut self) {
        self.history.clear();
        self.suffix_keys.fill(0);
        self.valid = false;
        self.cdf_valid = false;
        self.pdf.fill(1.0 / 256.0);
        self.cdf = uniform_cdf();
    }

    pub(crate) fn lifecycle_snapshot(&self) -> PpmdLifecycleSnapshot {
        PpmdLifecycleSnapshot {
            history: self.history.clone(),
            suffix_keys: self.suffix_keys.clone(),
            pdf: self.pdf,
            cdf: self.cdf,
            valid: self.valid,
            cdf_valid: self.cdf_valid,
        }
    }

    pub(crate) fn restore_lifecycle_snapshot(&mut self, snapshot: PpmdLifecycleSnapshot) {
        self.history = snapshot.history;
        self.suffix_keys = snapshot.suffix_keys;
        self.pdf = snapshot.pdf;
        self.cdf = snapshot.cdf;
        self.valid = snapshot.valid;
        self.cdf_valid = snapshot.cdf_valid;
    }

    /// Advance conditioning history without updating fitted context counts.
    pub fn update_history_only(&mut self, symbol: u8) {
        self.append_history_symbol(symbol);
        self.valid = false;
        self.cdf_valid = false;
    }

    #[cfg(any(test, feature = "research-tooling"))]
    pub(crate) fn memory_usage_breakdown(&self) -> PpmdMemoryUsage {
        let context_table_bytes: usize = self
            .contexts
            .iter()
            .map(|map| {
                map.capacity().saturating_mul(
                    std::mem::size_of::<(u64, ContextStats)>() + std::mem::size_of::<u8>(),
                )
            })
            .sum();
        let context_spill_bytes: usize = self
            .contexts
            .iter()
            .flat_map(|map| map.values())
            .map(ContextStats::spill_bytes)
            .sum();
        PpmdMemoryUsage {
            context_table_bytes,
            context_spill_bytes,
            queue_bytes: self
                .queue
                .capacity()
                .saturating_mul(std::mem::size_of::<(usize, u64)>()),
            history_bytes: self
                .history
                .capacity()
                .saturating_mul(std::mem::size_of::<u8>()),
            suffix_key_bytes: self
                .suffix_keys
                .capacity()
                .saturating_mul(std::mem::size_of::<u64>()),
            pdf_cache_bytes: std::mem::size_of::<[f64; 256]>() + std::mem::size_of::<[f64; 257]>(),
        }
    }

    #[cfg(any(test, feature = "research-tooling"))]
    pub(crate) fn estimated_size_bytes(&self) -> usize {
        self.memory_usage_breakdown().total_bytes()
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
                interpolate_context_in_place(ctx, &mut lower);
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
        debug_assert!(ord <= self.order);
        debug_assert!(ord <= self.history.len());
        self.suffix_keys[ord]
    }

    #[cfg(test)]
    fn sparse_query_state(&self) -> SparseQueryState {
        let mut state = SparseQueryState::new();
        let max_order = self.order.min(self.history.len());
        for ord in 0..=max_order {
            let key = self.context_key(ord);
            if let Some(ctx) = self.contexts[ord].get(&key) {
                state.interpolate_context(ctx);
            }
        }
        state
    }

    #[cfg(test)]
    fn flooring_diagnostics(&self) -> FlooringDiagnostics {
        let state = self.sparse_query_state();
        let mut min_unfloored_probability = f64::INFINITY;
        let mut floored_count = 0usize;
        let mut mass_added_by_flooring = 0.0;
        for symbol in 0..256usize {
            let value = state.raw_value(symbol);
            min_unfloored_probability = min_unfloored_probability.min(value);
            if !value.is_finite() || value < PDF_MIN {
                floored_count += 1;
                mass_added_by_flooring += PDF_MIN - if value.is_finite() { value } else { 0.0 };
            }
        }
        let normalization_sum = state.normalization_sum();
        let normalization_factor_after_flooring =
            if normalization_sum.is_finite() && normalization_sum > 0.0 {
                1.0 / normalization_sum
            } else {
                1.0
            };
        FlooringDiagnostics {
            min_unfloored_probability,
            floored_count,
            mass_added_by_flooring,
            normalization_factor_after_flooring,
        }
    }

    fn append_history_symbol(&mut self, symbol: u8) {
        let new_max_order = self.order.min(self.history.len() + 1);
        for ord in (1..=new_max_order).rev() {
            let prev_hash = if ord == 1 {
                FNV_OFFSET
            } else {
                self.suffix_keys[ord - 1]
            };
            self.suffix_keys[ord] = extend_hash(prev_hash, symbol);
        }
        self.suffix_keys[0] = 0;
        self.history.push(symbol);
    }
}

#[cfg(test)]
struct SparseQueryState {
    base: f64,
    values: [f64; 256],
    touched: [bool; 256],
    touched_symbols: [u8; 256],
    touched_len: usize,
}

#[cfg(test)]
struct FlooringDiagnostics {
    min_unfloored_probability: f64,
    floored_count: usize,
    mass_added_by_flooring: f64,
    normalization_factor_after_flooring: f64,
}

#[cfg(test)]
impl SparseQueryState {
    fn new() -> Self {
        Self {
            base: 1.0 / 256.0,
            values: [0.0; 256],
            touched: [false; 256],
            touched_symbols: [0; 256],
            touched_len: 0,
        }
    }

    fn interpolate_context(&mut self, ctx: &ContextStats) {
        let distinct = ctx.distinct_len() as f64;
        let denom = (ctx.total as f64) + distinct + 1.0;
        let escape = (distinct + 1.0) / denom;
        self.base *= escape;
        for idx in 0..self.touched_len {
            let symbol = self.touched_symbols[idx] as usize;
            self.values[symbol] *= escape;
        }
        ctx.for_each(|entry| {
            let idx = entry.symbol as usize;
            if !self.touched[idx] {
                self.touched[idx] = true;
                self.touched_symbols[self.touched_len] = entry.symbol;
                self.touched_len += 1;
                self.values[idx] = self.base;
            }
            self.values[idx] += (entry.count as f64) / denom;
        });
    }

    fn raw_value(&self, symbol: usize) -> f64 {
        if self.touched[symbol] {
            self.values[symbol]
        } else {
            self.base
        }
    }

    fn floored_value(&self, symbol: usize) -> f64 {
        let value = self.raw_value(symbol);
        if value.is_finite() {
            value.max(PDF_MIN)
        } else {
            PDF_MIN
        }
    }

    fn normalization_sum(&self) -> f64 {
        let mut sum = 0.0;
        for symbol in 0..256usize {
            sum += self.floored_value(symbol);
        }
        sum
    }
}

fn interpolate_context_in_place(ctx: &ContextStats, lower: &mut [f64; 256]) {
    let distinct = ctx.distinct_len() as f64;
    let denom = (ctx.total as f64) + distinct + 1.0;
    let escape = (distinct + 1.0) / denom;
    for p in lower.iter_mut() {
        *p *= escape;
    }
    ctx.for_each(|entry| {
        lower[entry.symbol as usize] += (entry.count as f64) / denom;
    });
}

fn normalize_pdf_and_maybe_cdf(pdf: &mut [f64; 256], cdf: Option<&mut [f64; 257]>) {
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
        if let Some(cdf) = cdf {
            *cdf = uniform_cdf();
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

#[inline]
fn extend_hash(hash: u64, byte: u8) -> u64 {
    (hash ^ (byte as u64)).wrapping_mul(FNV_PRIME)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut h = FNV_OFFSET;
        for &b in bytes {
            h = extend_hash(h, b);
        }
        h
    }

    fn reference_context_key(history: &[u8], ord: usize) -> u64 {
        if ord == 0 {
            0
        } else {
            hash_bytes(&history[history.len() - ord..])
        }
    }

    fn assert_suffix_keys_match_reference(model: &PpmdModel) {
        let max_order = model.order.min(model.history.len());
        for ord in 0..=max_order {
            assert_eq!(
                model.context_key(ord),
                reference_context_key(&model.history, ord)
            );
        }
    }

    fn reference_interpolate_context(ctx: &ContextStats, lower: &[f64; 256]) -> [f64; 256] {
        let distinct = ctx.distinct_len() as f64;
        let denom = (ctx.total as f64) + distinct + 1.0;
        let escape = (distinct + 1.0) / denom;
        let mut out = [0.0; 256];
        for i in 0..256 {
            out[i] = lower[i] * escape;
        }
        ctx.for_each(|entry| {
            out[entry.symbol as usize] += (entry.count as f64) / denom;
        });
        out
    }

    fn train_query_regression_model() -> PpmdModel {
        let mut model = PpmdModel::new(12, 1);
        let data = b"abracadabra abracadabra mississippi banana bandana ppmd query exactness";
        for &byte in data {
            model.update(byte);
        }
        model.reset_history();
        for &byte in b"abracadabra mississippi" {
            model.update_history_only(byte);
        }
        model
    }

    #[test]
    fn rolling_suffix_keys_match_recomputed_suffix_hashes() {
        let mut model = PpmdModel::new(12, 1);
        let bytes = [
            0, 1, 2, 3, 255, 128, 64, 32, 16, 8, 4, 2, 1, 0, 251, 17, 99, 100,
        ];
        assert_suffix_keys_match_reference(&model);
        for &byte in &bytes {
            model.update(byte);
            assert_suffix_keys_match_reference(&model);
        }

        let mut cloned = model.clone();
        assert_suffix_keys_match_reference(&cloned);
        for &byte in &[7, 6, 5, 4, 3, 2, 1] {
            cloned.update_history_only(byte);
            assert_suffix_keys_match_reference(&cloned);
        }

        cloned.reset_history();
        assert_suffix_keys_match_reference(&cloned);
        assert!(cloned.suffix_keys.iter().all(|&key| key == 0));

        for &byte in &[42, 43, 44, 45] {
            cloned.update_history_only(byte);
            assert_suffix_keys_match_reference(&cloned);
        }
    }

    #[test]
    fn in_place_interpolation_matches_out_of_place_reference() {
        let mut ctx = ContextStats::default();
        for &symbol in &[0, 1, 1, 2, 3, 3, 3, 128, 255, 255] {
            ctx.observe(symbol);
        }

        let mut lower = [0.0; 256];
        for (i, p) in lower.iter_mut().enumerate() {
            *p = ((i + 1) as f64) / 32896.0;
        }

        let expected = reference_interpolate_context(&ctx, &lower);
        interpolate_context_in_place(&ctx, &mut lower);
        for (expected, actual) in expected.iter().zip(lower.iter()) {
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
    }

    #[test]
    fn exact_symbol_queries_match_dense_pdf() {
        let query_model = train_query_regression_model();
        let mut dense = query_model.clone();
        let pdf = *dense.pdf();

        for symbol in 0..=255u8 {
            let mut queried = query_model.clone();
            let got = queried.symbol_prob(symbol);
            let expected = pdf[symbol as usize];
            assert_eq!(
                expected.to_bits(),
                got.to_bits(),
                "symbol={symbol} expected={expected:?} got={got:?}"
            );
        }
    }

    #[test]
    fn exact_interval_queries_match_dense_cdf_differences() {
        let query_model = train_query_regression_model();
        let mut dense = query_model.clone();
        let cdf = *dense.cdf();
        let ranges = [
            (0usize, 1usize),
            (0, 2),
            (0, 128),
            (0, 256),
            (1, 2),
            (3, 17),
            (17, 42),
            (42, 128),
            (64, 192),
            (127, 128),
            (128, 256),
            (255, 256),
        ];

        for &(lo, hi) in &ranges {
            let mut queried = query_model.clone();
            let expected = cdf[hi] - cdf[lo];
            let got = queried.interval_mass(lo, hi);
            assert_eq!(
                expected.to_bits(),
                got.to_bits(),
                "range={lo}..{hi} expected={expected:?} got={got:?}"
            );
        }
    }

    #[test]
    fn exact_queries_match_dense_pdf_when_probability_flooring_is_active() {
        let mut query_model = PpmdModel::new(12, 1);
        for _ in 0..512usize {
            query_model.update(b'a');
        }

        let diagnostics = query_model.flooring_diagnostics();
        assert!(diagnostics.min_unfloored_probability < PDF_MIN);
        assert!(diagnostics.floored_count > 0);
        assert!(diagnostics.mass_added_by_flooring > 0.0);
        assert!(diagnostics.normalization_factor_after_flooring.is_finite());

        let mut dense = query_model.clone();
        let pdf = *dense.pdf();
        for symbol in [0u8, b'a', b'b', 127, 255] {
            let mut queried = query_model.clone();
            let got = queried.symbol_prob(symbol);
            let expected = pdf[symbol as usize];
            assert_eq!(
                expected.to_bits(),
                got.to_bits(),
                "symbol={symbol} expected={expected:?} got={got:?}"
            );
        }
    }

    #[test]
    fn context_stats_spills_after_inline_capacity_without_changing_counts() {
        let mut ctx = ContextStats::default();
        for &symbol in &[1u8, 2, 3, 4, 5, 5, 4, 3, 2, 1] {
            ctx.observe(symbol);
        }

        assert_eq!(ctx.inline_len as usize, INLINE_CONTEXT_COUNTS);
        assert!(ctx.spill_counts.is_some());
        assert_eq!(ctx.distinct_len(), 5);

        let mut seen = [0u16; 256];
        ctx.for_each(|entry| {
            seen[entry.symbol as usize] = entry.count;
        });
        assert_eq!(seen[1], 2);
        assert_eq!(seen[2], 2);
        assert_eq!(seen[3], 2);
        assert_eq!(seen[4], 2);
        assert_eq!(seen[5], 2);
    }

    #[test]
    fn ppmd_memory_usage_breakdown_sums_to_estimated_total() {
        let mut model = PpmdModel::new(12, 1);
        for &byte in
            b"abracadabra abracadabra mississippi banana bandana ppmd memory validation payload"
        {
            model.update(byte);
        }

        let usage = model.memory_usage_breakdown();
        assert_eq!(model.estimated_size_bytes(), usage.total_bytes());
        assert!(usage.context_table_bytes > 0);
        assert!(usage.pdf_cache_bytes > 0);
    }
}
