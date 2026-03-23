//! Context Tree Weighting (CTW) and Factorized Action-Conditional CTW (FAC-CTW).
//!
//! This implementation stores maximal non-root unary runs as chains while keeping
//! the root and true branching points explicit. Updates and reverts rebuild only
//! the touched context path from exact per-node state, which preserves predictive
//! semantics while avoiding the `O(depth)` explicit-node blow-up for singleton
//! paths.

use std::cell::RefCell;
use std::f64;
use std::mem::size_of;

type Symbol = bool;

#[inline(always)]
fn ensure_log_caches(log_int: &mut Vec<f64>, log_half: &mut Vec<f64>, upto: usize) {
    if upto < log_int.len() {
        return;
    }
    let start = log_int.len();
    log_int.reserve(upto + 1 - start);
    log_half.reserve(upto + 1 - start);
    for n in start..=upto {
        if n == 0 {
            log_int.push(f64::NEG_INFINITY);
        } else {
            log_int.push((n as f64).ln());
        }
        log_half.push((n as f64 + 0.5).ln());
    }
}

#[derive(Default)]
struct SharedLogCache {
    log_int: Vec<f64>,
    log_half: Vec<f64>,
}

impl SharedLogCache {
    fn new() -> Self {
        Self {
            log_int: vec![f64::NEG_INFINITY],
            log_half: vec![(0.5f64).ln()],
        }
    }

    #[inline(always)]
    fn ensure(&mut self, upto: usize) {
        ensure_log_caches(&mut self.log_int, &mut self.log_half, upto);
    }

    #[inline(always)]
    fn memory_usage(&self) -> usize {
        self.log_int.capacity() * size_of::<f64>() + self.log_half.capacity() * size_of::<f64>()
    }
}

thread_local! {
    static CTW_SHARED_LOG_CACHE: RefCell<SharedLogCache> =
        RefCell::new(SharedLogCache::new());
}

#[inline]
fn with_shared_log_cache<R>(upto: usize, f: impl FnOnce(&[f64], &[f64]) -> R) -> R {
    CTW_SHARED_LOG_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        cache.ensure(upto);
        f(&cache.log_int, &cache.log_half)
    })
}

#[inline]
fn shared_log_cache_memory_usage() -> usize {
    CTW_SHARED_LOG_CACHE.with(|cache_cell| cache_cell.borrow().memory_usage())
}

#[cfg(test)]
#[inline]
fn shared_log_cache_lens() -> (usize, usize) {
    CTW_SHARED_LOG_CACHE.with(|cache_cell| {
        let cache = cache_cell.borrow();
        (cache.log_int.len(), cache.log_half.len())
    })
}

#[inline(always)]
fn history_symbol(history: &[Symbol], depth: usize) -> Symbol {
    let idx = history.len().wrapping_sub(depth + 1);
    if depth < history.len() {
        unsafe { *history.get_unchecked(idx) }
    } else {
        false
    }
}

#[inline(always)]
unsafe fn history_at_or_zero(history_ptr: *const Symbol, history_len: isize, idx: isize) -> Symbol {
    if idx >= 0 && idx < history_len {
        *history_ptr.add(idx as usize)
    } else {
        false
    }
}

const INDEX_BITS: u32 = 31;
const INDEX_LIMIT: usize = 1usize << INDEX_BITS;
const CHILD_SEGMENT_TAG: u32 = 1u32 << INDEX_BITS;
const CHILD_INDEX_MASK: u32 = CHILD_SEGMENT_TAG - 1;
const SEG_META_MODE_SHIFT: u32 = 30;
const SEG_META_MODE_MASK: u32 = 0b11 << SEG_META_MODE_SHIFT;
const SEG_LEN_MASK: u32 = !SEG_META_MODE_MASK;
const SEG_MODE_EXACT: u32 = 0 << SEG_META_MODE_SHIFT;
const SEG_MODE_HISTORY: u32 = 1 << SEG_META_MODE_SHIFT;
const SEG_MODE_HISTORY_INVERT: u32 = 2 << SEG_META_MODE_SHIFT;
const SEG_MODE_CONST: u32 = 3 << SEG_META_MODE_SHIFT;
const SEG_EXACT_MAX_LEN: u32 = 64;

/// Index into the explicit-node arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIndex(u32);

impl NodeIndex {
    #[cold]
    #[inline(never)]
    fn overflow() -> ! {
        panic!("ctw node index overflow");
    }

    #[inline(always)]
    fn from_usize(idx: usize) -> Self {
        if idx >= INDEX_LIMIT {
            Self::overflow();
        }
        Self(idx as u32)
    }

    #[inline(always)]
    fn get(self) -> usize {
        self.0 as usize
    }
}

/// Index into the unary-segment arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SegmentIndex(u32);

impl SegmentIndex {
    #[cold]
    #[inline(never)]
    fn overflow() -> ! {
        panic!("ctw segment index overflow");
    }

    #[inline(always)]
    fn from_usize(idx: usize) -> Self {
        if idx >= INDEX_LIMIT {
            Self::overflow();
        }
        Self(idx as u32)
    }

    #[inline(always)]
    fn get(self) -> usize {
        self.0 as usize
    }
}

/// Tagged child reference: `None`, explicit node, or unary segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChildRef(u32);

impl ChildRef {
    const NONE: ChildRef = ChildRef(u32::MAX);

    #[inline(always)]
    fn from_node(idx: NodeIndex) -> Self {
        debug_assert!(idx.0 < CHILD_SEGMENT_TAG);
        Self(idx.0)
    }

    #[inline(always)]
    fn from_segment(idx: SegmentIndex) -> Self {
        debug_assert!(idx.0 < CHILD_SEGMENT_TAG);
        Self(CHILD_SEGMENT_TAG | idx.0)
    }

    #[inline(always)]
    fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline(always)]
    fn is_some(self) -> bool {
        self.0 != u32::MAX
    }

    #[inline(always)]
    fn as_node(self) -> Option<NodeIndex> {
        if self.is_none() || (self.0 & CHILD_SEGMENT_TAG) != 0 {
            None
        } else {
            Some(NodeIndex(self.0))
        }
    }

    #[inline(always)]
    fn as_segment(self) -> Option<SegmentIndex> {
        if self.is_none() || (self.0 & CHILD_SEGMENT_TAG) == 0 {
            None
        } else {
            Some(SegmentIndex(self.0 & CHILD_INDEX_MASK))
        }
    }
}

impl Default for ChildRef {
    fn default() -> Self {
        Self::NONE
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SegmentPayload {
    repr_lo: u32,
    repr_hi: u32,
    meta: u32,
}

impl SegmentPayload {
    #[inline(always)]
    fn exact(bits: u64, len: u32) -> Self {
        debug_assert!(len <= SEG_EXACT_MAX_LEN);
        debug_assert!(len <= SEG_LEN_MASK);
        Self {
            repr_lo: bits as u32,
            repr_hi: (bits >> 32) as u32,
            meta: SEG_MODE_EXACT | len,
        }
    }

    #[inline(always)]
    fn history(anchor: u32, len: u32, invert: bool) -> Self {
        debug_assert!(len <= SEG_LEN_MASK);
        Self {
            repr_lo: anchor,
            repr_hi: 0,
            meta: if invert {
                SEG_MODE_HISTORY_INVERT | len
            } else {
                SEG_MODE_HISTORY | len
            },
        }
    }

    #[inline(always)]
    fn constant(bit: bool, len: u32) -> Self {
        debug_assert!(len <= SEG_LEN_MASK);
        Self {
            repr_lo: bit as u32,
            repr_hi: 0,
            meta: SEG_MODE_CONST | len,
        }
    }

    #[inline(always)]
    fn len(self) -> u32 {
        self.meta & SEG_LEN_MASK
    }

    #[inline(always)]
    fn set_len(&mut self, len: u32) {
        debug_assert!(len <= SEG_LEN_MASK);
        self.meta = (self.meta & SEG_META_MODE_MASK) | len;
    }

    #[inline(always)]
    fn mode(self) -> u32 {
        self.meta & SEG_META_MODE_MASK
    }

    #[inline(always)]
    fn is_exact(self) -> bool {
        self.mode() == SEG_MODE_EXACT
    }

    #[inline(always)]
    fn exact_bits(self) -> u64 {
        (self.repr_lo as u64) | ((self.repr_hi as u64) << 32)
    }

    #[inline(always)]
    fn anchor_or_const(self) -> u32 {
        self.repr_lo
    }

    #[inline(always)]
    fn const_bit(self) -> bool {
        (self.repr_lo & 1) != 0
    }

    #[inline(always)]
    fn prepend_exact(self, edge: usize) -> Option<Self> {
        if !self.is_exact() || self.len() >= SEG_EXACT_MAX_LEN {
            return None;
        }
        let len = self.len() + 1;
        let bits = ((edge as u64) & 1) | (self.exact_bits() << 1);
        Some(Self::exact(bits, len))
    }

    #[inline(always)]
    fn prefix(self, len: u32) -> Self {
        debug_assert!(len <= self.len());
        match self.mode() {
            SEG_MODE_EXACT => Self::exact(self.exact_bits() & low_bits_mask_u64(len), len),
            SEG_MODE_HISTORY | SEG_MODE_HISTORY_INVERT => Self {
                meta: (self.meta & SEG_META_MODE_MASK) | len,
                ..self
            },
            SEG_MODE_CONST => Self::constant(self.const_bit(), len),
            _ => unreachable!("invalid ctw segment payload mode"),
        }
    }

    #[inline(always)]
    fn suffix_after(self, skip: u32) -> Self {
        debug_assert!(skip <= self.len());
        let new_len = self.len() - skip;
        match self.mode() {
            SEG_MODE_EXACT => Self::exact(self.exact_bits() >> skip, new_len),
            SEG_MODE_HISTORY | SEG_MODE_HISTORY_INVERT => Self {
                repr_lo: self
                    .anchor_or_const()
                    .checked_sub(skip)
                    .expect("ctw history segment anchor underflow"),
                meta: (self.meta & SEG_META_MODE_MASK) | new_len,
                ..self
            },
            SEG_MODE_CONST => Self::constant(self.const_bit(), new_len),
            _ => unreachable!("invalid ctw segment payload mode"),
        }
    }

    #[inline(always)]
    fn from_path(history: &[Symbol], depth: usize, len: u32) -> Option<Self> {
        if len > SEG_EXACT_MAX_LEN {
            return None;
        }
        Some(Self::exact(
            path_bits_from_history(history, depth, len as usize),
            len,
        ))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LevelState {
    symbol_count: [u32; 2],
    log_prob_kt: f64,
    sibling: ChildRef,
}

#[cfg(test)]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
struct PredictEntry {
    symbol_count: [u32; 2],
    log_prob_kt: f64,
    log_prob_weighted: f64,
    sibling_weight: f64,
    has_sibling: bool,
}

#[derive(Clone, Copy, Debug)]
enum Detach {
    NodeChild { node: NodeIndex, edge: usize },
    SegmentNext { segment: SegmentIndex, new_len: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExistingSource {
    None,
    Node(NodeIndex),
    Segment(SegmentIndex, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreparedEnd {
    MaxDepth,
    MissingAtRoot,
    MissingAfterCurrent,
    MismatchAtCurrentSegment,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PreparedStep {
    source: ExistingSource,
    counts: [u32; 2],
    kt_log_prob: f64,
    span: u32,
    sibling_weight: f64,
    has_sibling: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct CtNode {
    children: [ChildRef; 2],
    log_prob_kt: f64,
    log_prob_weighted: f64,
    symbol_count: [u32; 2],
}

#[derive(Clone, Copy, Debug)]
struct CtSegment {
    tail: ChildRef,
    log_prob_kt: f64,
    head_log_prob_weighted: f64,
    symbol_count: [u32; 2],
    payload: SegmentPayload,
}

impl Default for CtSegment {
    fn default() -> Self {
        Self {
            tail: ChildRef::NONE,
            log_prob_kt: 0.0,
            head_log_prob_weighted: 0.0,
            symbol_count: [0, 0],
            payload: SegmentPayload::default(),
        }
    }
}

impl CtSegment {
    #[inline(always)]
    fn len(self) -> u32 {
        self.payload.len()
    }

    #[inline(always)]
    fn set_len(&mut self, len: u32) {
        self.payload.set_len(len);
    }
}

#[inline(always)]
fn low_bits_mask_u64(len: u32) -> u64 {
    if len >= 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

#[inline(always)]
fn path_bits_from_history(history: &[Symbol], depth: usize, len: usize) -> u64 {
    let history_len = history.len();
    let available = history_len.saturating_sub(depth).min(len);
    if available == 0 {
        return 0;
    }

    let mut bits = 0u64;
    let mut hist_idx = history_len - depth - 1;
    for offset in 0..available {
        bits |= (unsafe { *history.get_unchecked(hist_idx) } as u64) << offset;
        if hist_idx == 0 {
            break;
        }
        hist_idx -= 1;
    }
    bits
}

#[inline(always)]
fn shift_path_bits(path_bits: u64, consumed: usize) -> u64 {
    if consumed >= 64 {
        0
    } else {
        path_bits >> consumed
    }
}

#[inline(always)]
fn first_exact_segment_mismatch(
    exact_bits: u64,
    path_bits: u64,
    comparable_len: usize,
) -> Option<(usize, bool, bool)> {
    if comparable_len == 0 {
        return None;
    }

    let diff = (exact_bits ^ path_bits) & low_bits_mask_u64(comparable_len as u32);
    if diff == 0 {
        None
    } else {
        let offset = diff.trailing_zeros() as usize;
        Some((
            offset,
            ((path_bits >> offset) & 1) != 0,
            ((exact_bits >> offset) & 1) != 0,
        ))
    }
}

#[inline(always)]
fn predict_ratio_kt(counts: [u32; 2], sym_idx: usize) -> f64 {
    let total = (counts[0] + counts[1]) as f64;
    let sym_count = counts[sym_idx] as f64;
    (sym_count + 0.5) / (total + 1.0)
}

#[inline(always)]
fn predict_ratio_kt_one(counts: [u32; 2]) -> f64 {
    let total = (counts[0] + counts[1]) as f64;
    let sym_count = counts[1] as f64;
    (sym_count + 0.5) / (total + 1.0)
}

#[inline(always)]
fn update_weighted_log_prob_non_leaf(kt_log_prob: f64, log_prob_w0: f64, log_prob_w1: f64) -> f64 {
    let child_log_prob = log_prob_w0 + log_prob_w1;
    let delta = child_log_prob - kt_log_prob;
    let log_prob_weighted = if delta >= 0.0 {
        child_log_prob + (-delta).exp().ln_1p() - std::f64::consts::LN_2
    } else {
        kt_log_prob + delta.exp().ln_1p() - std::f64::consts::LN_2
    };
    clamp_log_prob(log_prob_weighted)
}

#[inline(always)]
fn update_weighted_log_prob(
    kt_log_prob: f64,
    log_prob_w0: f64,
    log_prob_w1: f64,
    is_leaf: bool,
) -> f64 {
    if is_leaf {
        clamp_log_prob(kt_log_prob)
    } else {
        update_weighted_log_prob_non_leaf(kt_log_prob, log_prob_w0, log_prob_w1)
    }
}

#[inline(always)]
fn clamp_log_prob(log_prob: f64) -> f64 {
    if log_prob > 1.0e-10 { 0.0 } else { log_prob }
}

#[inline(always)]
fn logsumexp_pair(lhs: f64, rhs: f64) -> f64 {
    if lhs == f64::NEG_INFINITY {
        return rhs;
    }
    if rhs == f64::NEG_INFINITY {
        return lhs;
    }
    let pivot = lhs.max(rhs);
    pivot + ((lhs - pivot).exp() + (rhs - pivot).exp()).ln()
}

#[inline(always)]
fn unary_chain_log_weight(kt_log_prob: f64, continuation_log_prob: f64, len: u32) -> f64 {
    debug_assert!(len > 0);
    if kt_log_prob.to_bits() == continuation_log_prob.to_bits() {
        return kt_log_prob;
    }
    let log_alpha = -(len as f64) * std::f64::consts::LN_2;
    let alpha = log_alpha.exp();
    let log_kt_mass = kt_log_prob + (-alpha).ln_1p();
    let log_cont_mass = continuation_log_prob + log_alpha;
    clamp_log_prob(logsumexp_pair(log_kt_mass, log_cont_mass))
}

#[inline(always)]
fn combined_weight_ratio_internal(
    kt_log_prob: f64,
    counts: [u32; 2],
    path_child_log_prob: f64,
    sibling_log_prob: f64,
    child_ratio: f64,
    sym_idx: usize,
) -> (f64, f64) {
    let kt_ratio = predict_ratio_kt(counts, sym_idx);
    let child_log_prob = path_child_log_prob + sibling_log_prob;
    let delta = child_log_prob - kt_log_prob;
    if delta >= 0.0 {
        let x = (-delta).exp();
        (
            clamp_log_prob(child_log_prob + x.ln_1p() - std::f64::consts::LN_2),
            (kt_ratio * x + child_ratio) / (1.0 + x),
        )
    } else {
        let x = delta.exp();
        (
            clamp_log_prob(kt_log_prob + x.ln_1p() - std::f64::consts::LN_2),
            (kt_ratio + x * child_ratio) / (1.0 + x),
        )
    }
}

#[inline(always)]
fn combined_weight_ratio_internal_one(
    kt_log_prob: f64,
    counts: [u32; 2],
    path_child_log_prob: f64,
    sibling_log_prob: f64,
    child_ratio: f64,
) -> (f64, f64) {
    let kt_ratio = predict_ratio_kt_one(counts);
    let child_log_prob = path_child_log_prob + sibling_log_prob;
    let delta = child_log_prob - kt_log_prob;
    if delta >= 0.0 {
        let x = (-delta).exp();
        (
            clamp_log_prob(child_log_prob + x.ln_1p() - std::f64::consts::LN_2),
            (kt_ratio * x + child_ratio) / (1.0 + x),
        )
    } else {
        let x = delta.exp();
        (
            clamp_log_prob(kt_log_prob + x.ln_1p() - std::f64::consts::LN_2),
            (kt_ratio + x * child_ratio) / (1.0 + x),
        )
    }
}

#[inline(always)]
fn unary_chain_log_weight_precomputed(
    kt_log_prob: f64,
    continuation_log_prob: f64,
    alpha: f64,
    log_alpha: f64,
    log_one_minus_alpha: f64,
) -> f64 {
    if kt_log_prob.to_bits() == continuation_log_prob.to_bits() {
        return clamp_log_prob(kt_log_prob);
    }

    let delta = continuation_log_prob - kt_log_prob;
    let log_prob_weighted = if delta >= 0.0 {
        let x = ((1.0 - alpha) / alpha) * (-delta).exp();
        continuation_log_prob + log_alpha + x.ln_1p()
    } else {
        let x = (alpha / (1.0 - alpha)) * delta.exp();
        kt_log_prob + log_one_minus_alpha + x.ln_1p()
    };
    clamp_log_prob(log_prob_weighted)
}

#[inline(always)]
fn unary_chain_ratio_transform_precomputed(
    kt_log_prob: f64,
    counts: [u32; 2],
    continuation_log_prob: f64,
    continuation_ratio: f64,
    alpha: f64,
    log_alpha: f64,
    log_one_minus_alpha: f64,
    sym_idx: usize,
) -> (f64, f64) {
    let kt_ratio = predict_ratio_kt(counts, sym_idx);
    if kt_log_prob.to_bits() == continuation_log_prob.to_bits()
        && kt_ratio.to_bits() == continuation_ratio.to_bits()
    {
        return (clamp_log_prob(kt_log_prob), kt_ratio);
    }

    let delta = continuation_log_prob - kt_log_prob;
    if delta >= 0.0 {
        let x = ((1.0 - alpha) / alpha) * (-delta).exp();
        (
            clamp_log_prob(continuation_log_prob + log_alpha + x.ln_1p()),
            (kt_ratio * x + continuation_ratio) / (1.0 + x),
        )
    } else {
        let x = (alpha / (1.0 - alpha)) * delta.exp();
        (
            clamp_log_prob(kt_log_prob + log_one_minus_alpha + x.ln_1p()),
            (kt_ratio + x * continuation_ratio) / (1.0 + x),
        )
    }
}

#[inline(always)]
fn unary_chain_ratio_transform_precomputed_one(
    kt_log_prob: f64,
    counts: [u32; 2],
    continuation_log_prob: f64,
    continuation_ratio: f64,
    alpha: f64,
    log_alpha: f64,
    log_one_minus_alpha: f64,
) -> (f64, f64) {
    let kt_ratio = predict_ratio_kt_one(counts);
    if kt_log_prob.to_bits() == continuation_log_prob.to_bits()
        && kt_ratio.to_bits() == continuation_ratio.to_bits()
    {
        return (clamp_log_prob(kt_log_prob), kt_ratio);
    }

    let delta = continuation_log_prob - kt_log_prob;
    if delta >= 0.0 {
        let x = ((1.0 - alpha) / alpha) * (-delta).exp();
        (
            clamp_log_prob(continuation_log_prob + log_alpha + x.ln_1p()),
            (kt_ratio * x + continuation_ratio) / (1.0 + x),
        )
    } else {
        let x = (alpha / (1.0 - alpha)) * delta.exp();
        (
            clamp_log_prob(kt_log_prob + log_one_minus_alpha + x.ln_1p()),
            (kt_ratio + x * continuation_ratio) / (1.0 + x),
        )
    }
}

#[inline(always)]
fn predict_ratio_internal(
    kt_log_prob: f64,
    counts: [u32; 2],
    path_child_log_prob: f64,
    sibling_log_prob: f64,
    child_ratio: f64,
    sym_idx: usize,
) -> f64 {
    let kt_ratio = predict_ratio_kt(counts, sym_idx);
    let delta = path_child_log_prob + sibling_log_prob - kt_log_prob;
    if delta >= 0.0 {
        let inv_rho = (-delta).exp();
        (kt_ratio * inv_rho + child_ratio) / (1.0 + inv_rho)
    } else {
        let rho = delta.exp();
        (kt_ratio + rho * child_ratio) / (1.0 + rho)
    }
}

#[inline(always)]
fn predict_ratio_internal_one(
    kt_log_prob: f64,
    counts: [u32; 2],
    path_child_log_prob: f64,
    sibling_log_prob: f64,
    child_ratio: f64,
) -> f64 {
    let kt_ratio = predict_ratio_kt_one(counts);
    let delta = path_child_log_prob + sibling_log_prob - kt_log_prob;
    if delta >= 0.0 {
        let inv_rho = (-delta).exp();
        (kt_ratio * inv_rho + child_ratio) / (1.0 + inv_rho)
    } else {
        let rho = delta.exp();
        (kt_ratio + rho * child_ratio) / (1.0 + rho)
    }
}

#[inline(always)]
fn path_edge_at_depth(history: &[Symbol], history_len: usize, depth: usize) -> bool {
    if depth < history_len {
        history[history_len - depth - 1]
    } else {
        false
    }
}

#[inline(always)]
fn segment_edge_from_parts(
    segment: CtSegment,
    offset: usize,
    history: &[Symbol],
    history_len: usize,
) -> bool {
    match segment.payload.mode() {
        SEG_MODE_EXACT => ((segment.payload.exact_bits() >> offset) & 1) != 0,
        SEG_MODE_HISTORY | SEG_MODE_HISTORY_INVERT => {
            if segment.payload.anchor_or_const() as usize >= offset {
                let hist_idx = segment.payload.anchor_or_const() as usize - offset;
                if hist_idx < history_len {
                    let raw = history[hist_idx];
                    if segment.payload.mode() == SEG_MODE_HISTORY_INVERT {
                        !raw
                    } else {
                        raw
                    }
                } else {
                    segment.payload.mode() == SEG_MODE_HISTORY_INVERT
                }
            } else {
                segment.payload.mode() == SEG_MODE_HISTORY_INVERT
            }
        }
        SEG_MODE_CONST => segment.payload.const_bit(),
        _ => unreachable!("invalid ctw segment payload mode"),
    }
}

#[inline(always)]
fn first_segment_mismatch(
    segment: CtSegment,
    depth: usize,
    history: &[Symbol],
    comparable_len: usize,
) -> Option<(usize, bool, bool)> {
    if comparable_len == 0 {
        return None;
    }

    match segment.payload.mode() {
        SEG_MODE_EXACT => first_exact_segment_mismatch(
            segment.payload.exact_bits(),
            path_bits_from_history(history, depth, comparable_len),
            comparable_len,
        ),
        SEG_MODE_HISTORY | SEG_MODE_HISTORY_INVERT => {
            let history_ptr = history.as_ptr();
            let history_len = history.len() as isize;
            let mut path_hist_idx = history_len - depth as isize - 1;
            let mut seg_hist_idx = segment.payload.anchor_or_const() as isize;
            let invert = segment.payload.mode() == SEG_MODE_HISTORY_INVERT;
            for offset in 0..comparable_len {
                let path_edge =
                    unsafe { history_at_or_zero(history_ptr, history_len, path_hist_idx) };
                let existing_raw =
                    unsafe { history_at_or_zero(history_ptr, history_len, seg_hist_idx) };
                let existing_edge = if invert { !existing_raw } else { existing_raw };
                if existing_edge != path_edge {
                    return Some((offset, path_edge, existing_edge));
                }
                path_hist_idx -= 1;
                seg_hist_idx -= 1;
            }
            None
        }
        SEG_MODE_CONST => {
            let history_ptr = history.as_ptr();
            let history_len = history.len() as isize;
            let mut path_hist_idx = history_len - depth as isize - 1;
            let existing_edge = segment.payload.const_bit();
            for offset in 0..comparable_len {
                let path_edge =
                    unsafe { history_at_or_zero(history_ptr, history_len, path_hist_idx) };
                if existing_edge != path_edge {
                    return Some((offset, path_edge, existing_edge));
                }
                path_hist_idx -= 1;
            }
            None
        }
        _ => unreachable!("invalid ctw segment payload mode"),
    }
}

#[inline]
fn apply_update_to_state_raw(
    log_int: &[f64],
    log_half: &[f64],
    symbol_count: &mut [u32; 2],
    log_prob_kt: &mut f64,
    sym_idx: usize,
) {
    let total_before = (symbol_count[0] + symbol_count[1]) as usize;
    let sym_before = symbol_count[sym_idx] as usize;
    debug_assert!(sym_before <= total_before);
    debug_assert!(sym_before < log_half.len());
    debug_assert!(total_before + 1 < log_int.len());
    let log_half_before = unsafe { *log_half.get_unchecked(sym_before) };
    let log_total_after = unsafe { *log_int.get_unchecked(total_before + 1) };
    *log_prob_kt += log_half_before - log_total_after;
    if *log_prob_kt > 1.0e-10 {
        *log_prob_kt = 0.0;
    }
    symbol_count[sym_idx] = symbol_count[sym_idx]
        .checked_add(1)
        .expect("ctw symbol count overflow");
}

#[inline]
fn apply_revert_to_state_raw(
    log_int: &[f64],
    log_half: &[f64],
    symbol_count: &mut [u32; 2],
    log_prob_kt: &mut f64,
    sym_idx: usize,
) {
    let total = (symbol_count[0] + symbol_count[1]) as usize;
    let sym_count = symbol_count[sym_idx] as usize;
    if sym_count > 0 && total > 0 {
        debug_assert!(sym_count - 1 < log_half.len());
        debug_assert!(total < log_int.len());
        let log_half_before = unsafe { *log_half.get_unchecked(sym_count - 1) };
        let log_total = unsafe { *log_int.get_unchecked(total) };
        *log_prob_kt -= log_half_before - log_total;
        symbol_count[sym_idx] -= 1;
    }
    if *log_prob_kt > 1.0e-10 {
        *log_prob_kt = 0.0;
    }
}

#[derive(Clone, Debug)]
pub struct CtArena {
    nodes: Vec<CtNode>,
    segments: Vec<CtSegment>,
    free_nodes: Vec<NodeIndex>,
    free_segments: Vec<SegmentIndex>,
}

impl CtArena {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            segments: Vec::with_capacity(1024),
            free_nodes: Vec::new(),
            free_segments: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            segments: Vec::with_capacity(cap / 4 + 1),
            free_nodes: Vec::new(),
            free_segments: Vec::new(),
        }
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.nodes.reserve_exact(additional);
        self.segments.reserve_exact(additional / 4 + 1);
    }

    #[inline(always)]
    fn reset_node_slot(&mut self, idx: NodeIndex) {
        self.nodes[idx.get()] = CtNode {
            children: [ChildRef::NONE, ChildRef::NONE],
            log_prob_kt: 0.0,
            log_prob_weighted: 0.0,
            symbol_count: [0, 0],
        };
    }

    #[inline(always)]
    fn reset_segment_slot(&mut self, idx: SegmentIndex) {
        self.segments[idx.get()] = CtSegment::default();
    }

    #[inline(always)]
    fn alloc_node(&mut self) -> NodeIndex {
        if let Some(idx) = self.free_nodes.pop() {
            self.reset_node_slot(idx);
            idx
        } else {
            let idx = NodeIndex::from_usize(self.nodes.len());
            self.nodes.push(CtNode {
                children: [ChildRef::NONE, ChildRef::NONE],
                log_prob_kt: 0.0,
                log_prob_weighted: 0.0,
                symbol_count: [0, 0],
            });
            idx
        }
    }

    #[inline(always)]
    fn alloc_node_with_state(&mut self, symbol_count: [u32; 2], log_prob_kt: f64) -> NodeIndex {
        let idx = self.alloc_node();
        self.nodes[idx.get()].symbol_count = symbol_count;
        self.nodes[idx.get()].log_prob_kt = log_prob_kt;
        idx
    }

    #[inline(always)]
    fn free_node(&mut self, idx: NodeIndex) {
        self.free_nodes.push(idx);
    }

    #[inline(always)]
    fn alloc_segment(&mut self) -> SegmentIndex {
        if let Some(idx) = self.free_segments.pop() {
            self.reset_segment_slot(idx);
            idx
        } else {
            let idx = SegmentIndex::from_usize(self.segments.len());
            self.segments.push(CtSegment::default());
            idx
        }
    }

    #[inline(always)]
    fn free_segment(&mut self, idx: SegmentIndex) {
        self.reset_segment_slot(idx);
        self.free_segments.push(idx);
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.segments.clear();
        self.free_nodes.clear();
        self.free_segments.clear();
    }

    #[inline(always)]
    fn child(&self, parent_idx: NodeIndex, child_idx: usize) -> ChildRef {
        debug_assert!(parent_idx.get() < self.nodes.len());
        debug_assert!(child_idx < 2);
        unsafe {
            *self
                .nodes
                .get_unchecked(parent_idx.get())
                .children
                .get_unchecked(child_idx)
        }
    }

    #[inline(always)]
    fn set_child(&mut self, parent_idx: NodeIndex, child_idx: usize, child: ChildRef) {
        debug_assert!(parent_idx.get() < self.nodes.len());
        debug_assert!(child_idx < 2);
        unsafe {
            *self
                .nodes
                .get_unchecked_mut(parent_idx.get())
                .children
                .get_unchecked_mut(child_idx) = child;
        }
    }

    #[inline(always)]
    fn set_segment_tail(&mut self, segment_idx: SegmentIndex, child: ChildRef) {
        self.segments[segment_idx.get()].tail = child;
    }

    #[inline(always)]
    fn counts(&self, idx: NodeIndex) -> [u32; 2] {
        self.nodes[idx.get()].symbol_count
    }

    #[inline(always)]
    fn visits(&self, idx: NodeIndex) -> u32 {
        let counts = self.nodes[idx.get()].symbol_count;
        counts[0] + counts[1]
    }

    #[inline(always)]
    fn segment_symbol_count(&self, segment_idx: SegmentIndex) -> [u32; 2] {
        self.segments[segment_idx.get()].symbol_count
    }

    #[inline(always)]
    fn segment_log_prob_kt(&self, segment_idx: SegmentIndex) -> f64 {
        self.segments[segment_idx.get()].log_prob_kt
    }

    #[inline(always)]
    fn segment_len(&self, segment_idx: SegmentIndex) -> u32 {
        self.segments[segment_idx.get()].len()
    }

    #[inline(always)]
    fn segment_has_child(&self, segment_idx: SegmentIndex, offset: u32) -> bool {
        let segment = self.segments[segment_idx.get()];
        offset + 1 < segment.len() || segment.tail.is_some()
    }

    #[inline(always)]
    fn log_prob_weighted(&self, idx: NodeIndex) -> f64 {
        self.nodes[idx.get()].log_prob_weighted
    }

    #[inline(always)]
    fn log_prob_kt(&self, idx: NodeIndex) -> f64 {
        self.nodes[idx.get()].log_prob_kt
    }

    #[inline(always)]
    unsafe fn child_ref_weighted_unchecked(&self, child: ChildRef) -> f64 {
        if child.is_none() {
            return 0.0;
        }

        let raw = child.0;
        if (raw & CHILD_SEGMENT_TAG) == 0 {
            debug_assert!((raw as usize) < self.nodes.len());
            self.nodes.get_unchecked(raw as usize).log_prob_weighted
        } else {
            let idx = (raw & CHILD_INDEX_MASK) as usize;
            debug_assert!(idx < self.segments.len());
            self.segments.get_unchecked(idx).head_log_prob_weighted
        }
    }

    #[inline(always)]
    fn child_ref_weighted(&self, child: ChildRef) -> f64 {
        unsafe { self.child_ref_weighted_unchecked(child) }
    }

    #[inline(always)]
    fn singleton_segment_payload(&self, edge: usize) -> SegmentPayload {
        SegmentPayload::exact((edge & 1) as u64, 1)
    }

    #[inline(always)]
    fn segment_edge(&self, segment_idx: SegmentIndex, offset: u32, history: &[Symbol]) -> usize {
        let segment = self.segments[segment_idx.get()];
        segment_edge_from_parts(segment, offset as usize, history, history.len()) as usize
    }

    fn segment_suffix_weight(&self, segment_idx: SegmentIndex, offset: u32) -> f64 {
        let segment = self.segments[segment_idx.get()];
        if offset >= segment.len() {
            return self.child_ref_weighted(segment.tail);
        }
        if segment.tail.is_none() {
            return segment.log_prob_kt;
        }
        let remaining = segment.len() - offset;
        unary_chain_log_weight(
            segment.log_prob_kt,
            self.child_ref_weighted(segment.tail),
            remaining,
        )
    }

    #[inline(always)]
    fn segment_continuation_weight(&self, segment_idx: SegmentIndex, offset: u32) -> f64 {
        let segment = self.segments[segment_idx.get()];
        if offset + 1 < segment.len() {
            self.segment_suffix_weight(segment_idx, offset + 1)
        } else {
            self.child_ref_weighted(segment.tail)
        }
    }

    fn recompute_segment_head(&mut self, segment_idx: SegmentIndex) {
        let segment = self.segments[segment_idx.get()];
        let head = if segment.tail.is_some() {
            unary_chain_log_weight(
                segment.log_prob_kt,
                self.child_ref_weighted(segment.tail),
                segment.len(),
            )
        } else {
            segment.log_prob_kt
        };
        self.segments[segment_idx.get()].head_log_prob_weighted = head;
    }

    fn recompute_node_weight(&mut self, idx: NodeIndex) {
        let slot = idx.get();
        debug_assert!(slot < self.nodes.len());
        let node = unsafe { *self.nodes.get_unchecked(slot) };
        let [left, right] = node.children;
        let weighted = if left.is_none() && right.is_none() {
            clamp_log_prob(node.log_prob_kt)
        } else {
            let w0 = unsafe { self.child_ref_weighted_unchecked(left) };
            let w1 = unsafe { self.child_ref_weighted_unchecked(right) };
            update_weighted_log_prob_non_leaf(node.log_prob_kt, w0, w1)
        };
        unsafe {
            self.nodes.get_unchecked_mut(slot).log_prob_weighted = weighted;
        }
    }

    fn alloc_segment_with_parts(
        &mut self,
        symbol_count: [u32; 2],
        log_prob_kt: f64,
        tail: ChildRef,
        payload: SegmentPayload,
    ) -> SegmentIndex {
        let segment_idx = self.alloc_segment();
        self.segments[segment_idx.get()] = CtSegment {
            tail,
            log_prob_kt,
            head_log_prob_weighted: 0.0,
            symbol_count,
            payload,
        };
        if payload.len() == 1 && tail.is_none() {
            self.segments[segment_idx.get()].head_log_prob_weighted = log_prob_kt;
        } else {
            self.recompute_segment_head(segment_idx);
        }
        segment_idx
    }

    fn detach_segment_continuation(
        &mut self,
        segment_idx: SegmentIndex,
        offset: u32,
        detaches: &mut Vec<Detach>,
    ) -> ChildRef {
        let segment = self.segments[segment_idx.get()];
        if offset + 1 < segment.len() {
            let suffix = self.alloc_segment_with_parts(
                segment.symbol_count,
                segment.log_prob_kt,
                segment.tail,
                segment.payload.suffix_after(offset + 1),
            );
            detaches.push(Detach::SegmentNext {
                segment: segment_idx,
                new_len: offset + 1,
            });
            ChildRef::from_segment(suffix)
        } else {
            let tail = segment.tail;
            if tail.is_some() {
                detaches.push(Detach::SegmentNext {
                    segment: segment_idx,
                    new_len: segment.len(),
                });
            }
            tail
        }
    }

    fn prepend_or_alloc_segment(
        &mut self,
        history: &[Symbol],
        depth: usize,
        symbol_count: [u32; 2],
        log_prob_kt: f64,
        child: ChildRef,
        edge: usize,
        allow_history_pattern: bool,
    ) -> ChildRef {
        let singleton_payload = self.singleton_segment_payload(edge);

        if let Some(segment_idx) = child.as_segment() {
            let segment = self.segments[segment_idx.get()];
            let same_state = segment.symbol_count == symbol_count
                && segment.log_prob_kt.to_bits() == log_prob_kt.to_bits();
            if same_state && segment.tail == child {
                let segment = &mut self.segments[segment_idx.get()];
                let extended_payload = if segment.payload.is_exact() {
                    segment.payload.prepend_exact(edge)
                } else if allow_history_pattern {
                    let path_payload =
                        SegmentPayload::from_path(history, depth, segment.len().saturating_add(1));
                    path_payload.filter(|payload| {
                        let mut matches = true;
                        for offset in 0..segment.len() as usize {
                            let seg_edge =
                                segment_edge_from_parts(*segment, offset, history, history.len());
                            let payload_edge = ((payload.exact_bits() >> (offset + 1)) & 1) != 0;
                            if seg_edge != payload_edge {
                                matches = false;
                                break;
                            }
                        }
                        matches
                    })
                } else {
                    None
                };
                if let Some(payload) = extended_payload {
                    let old_head = segment.head_log_prob_weighted;
                    segment.payload = payload;
                    segment.head_log_prob_weighted =
                        update_weighted_log_prob(log_prob_kt, old_head, 0.0, false);
                    return ChildRef::from_segment(segment_idx);
                }
            }
        }

        let segment_idx =
            self.alloc_segment_with_parts(symbol_count, log_prob_kt, child, singleton_payload);
        ChildRef::from_segment(segment_idx)
    }

    fn free_child_ref(&mut self, child: ChildRef) {
        let mut stack = Vec::with_capacity(16);
        if child.is_some() {
            stack.push(child);
        }
        while let Some(next) = stack.pop() {
            if let Some(node_idx) = next.as_node() {
                let children = self.nodes[node_idx.get()].children;
                if children[0].is_some() {
                    stack.push(children[0]);
                }
                if children[1].is_some() {
                    stack.push(children[1]);
                }
                self.free_node(node_idx);
            } else if let Some(segment_idx) = next.as_segment() {
                let tail = self.segments[segment_idx.get()].tail;
                if tail.is_some() {
                    stack.push(tail);
                }
                self.free_segment(segment_idx);
            }
        }
    }

    pub fn memory_usage(&self) -> usize {
        self.nodes.capacity() * size_of::<CtNode>()
            + self.segments.capacity() * size_of::<CtSegment>()
            + self.free_nodes.capacity() * size_of::<NodeIndex>()
            + self.free_segments.capacity() * size_of::<SegmentIndex>()
    }
}

impl Default for CtArena {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct CtEngine {
    arena: CtArena,
    root: NodeIndex,
    max_depth: usize,
    segment_alpha: Vec<f64>,
    segment_log_alpha: Vec<f64>,
    segment_log_one_minus_alpha: Vec<f64>,
    levels: Vec<LevelState>,
    detaches: Vec<Detach>,
    prepared_steps: Vec<PreparedStep>,
    prepared_levels: usize,
    prepared_end: PreparedEnd,
}

impl CtEngine {
    const RESERVE_MIN_NODES: usize = 4 * 1024;
    const RESERVE_MAX_NODES: usize = 1 << 18;
    const HOT_PREFIX_DEPTH: usize = 10;

    fn new(depth: usize) -> Self {
        let mut arena = CtArena::with_capacity(1024.min(1 << depth.min(16)));
        let root = arena.alloc_node();
        let mut segment_alpha = Vec::with_capacity(depth + 1);
        let mut segment_log_alpha = Vec::with_capacity(depth + 1);
        let mut segment_log_one_minus_alpha = Vec::with_capacity(depth + 1);
        segment_alpha.push(1.0);
        segment_log_alpha.push(0.0);
        segment_log_one_minus_alpha.push(f64::NEG_INFINITY);
        let mut alpha = 1.0f64;
        for len in 1..=depth {
            alpha *= 0.5;
            segment_alpha.push(alpha);
            segment_log_alpha.push(-(len as f64) * std::f64::consts::LN_2);
            segment_log_one_minus_alpha.push((-alpha).ln_1p());
        }
        Self {
            arena,
            root,
            max_depth: depth,
            segment_alpha,
            segment_log_alpha,
            segment_log_one_minus_alpha,
            levels: vec![LevelState::default(); depth],
            detaches: Vec::with_capacity(depth),
            prepared_steps: Vec::with_capacity(depth),
            prepared_levels: 0,
            prepared_end: PreparedEnd::MaxDepth,
        }
    }

    #[inline(always)]
    fn root_visits(&self) -> usize {
        self.arena.visits(self.root) as usize
    }

    #[inline(always)]
    fn hot_prefix_depth(&self) -> usize {
        self.max_depth.min(Self::HOT_PREFIX_DEPTH)
    }

    fn clear(&mut self) {
        self.arena.clear();
        self.root = self.arena.alloc_node();
        self.levels.fill(LevelState::default());
        self.detaches.clear();
        self.prepared_steps.clear();
        self.prepared_levels = 0;
        self.prepared_end = PreparedEnd::MaxDepth;
    }

    #[inline]
    fn reserve_for_symbols(&mut self, total_symbols: usize) {
        if total_symbols == 0 {
            return;
        }

        let depth_scale = self.max_depth.saturating_add(1);
        let reserve_nodes = total_symbols
            .saturating_div(depth_scale)
            .clamp(Self::RESERVE_MIN_NODES, Self::RESERVE_MAX_NODES);
        let free_nodes = self
            .arena
            .nodes
            .capacity()
            .saturating_sub(self.arena.nodes.len());
        if reserve_nodes > free_nodes {
            self.arena.reserve_exact(reserve_nodes - free_nodes);
        }
    }

    #[inline]
    fn get_log_block_probability(&self) -> f64 {
        self.arena.log_prob_weighted(self.root)
    }

    #[inline]
    fn with_logs<R>(&mut self, upto: usize, f: impl FnOnce(&mut Self, &[f64], &[f64]) -> R) -> R {
        with_shared_log_cache(upto, |log_int, log_half| f(self, log_int, log_half))
    }

    #[inline]
    fn log_cache_memory_usage(&self) -> usize {
        shared_log_cache_memory_usage()
    }

    #[inline(always)]
    fn segment_constants(&self, len: u32) -> (f64, f64, f64) {
        let idx = len as usize;
        debug_assert!(idx < self.segment_alpha.len());
        debug_assert!(idx < self.segment_log_alpha.len());
        debug_assert!(idx < self.segment_log_one_minus_alpha.len());
        (
            unsafe { *self.segment_alpha.get_unchecked(idx) },
            unsafe { *self.segment_log_alpha.get_unchecked(idx) },
            unsafe { *self.segment_log_one_minus_alpha.get_unchecked(idx) },
        )
    }

    fn build_missing_segment_path(
        &mut self,
        depth: usize,
        history: &[Symbol],
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        if depth > self.max_depth {
            return ChildRef::NONE;
        }

        let mut counts = [0u32; 2];
        counts[sym_idx] = 1;
        let log_prob_kt = singleton_log_prob_kt;
        let total_len = self.max_depth - depth + 1;

        if let Some(payload) = SegmentPayload::from_path(history, depth, total_len as u32) {
            let segment =
                self.arena
                    .alloc_segment_with_parts(counts, log_prob_kt, ChildRef::NONE, payload);
            return ChildRef::from_segment(segment);
        }

        let history_nodes = if depth < history.len() {
            (self.max_depth.min(history.len() - 1) - depth) + 1
        } else {
            0
        };
        let const_nodes = total_len - history_nodes;

        let mut built = ChildRef::NONE;
        if const_nodes > 0 {
            let const_segment = self.arena.alloc_segment_with_parts(
                counts,
                log_prob_kt,
                ChildRef::NONE,
                SegmentPayload::constant(false, const_nodes as u32),
            );
            built = ChildRef::from_segment(const_segment);
        }
        if history_nodes > 0 {
            let history_segment = self.arena.alloc_segment_with_parts(
                counts,
                log_prob_kt,
                built,
                SegmentPayload::history(
                    (history.len() - depth - 1) as u32,
                    history_nodes as u32,
                    false,
                ),
            );
            built = ChildRef::from_segment(history_segment);
        }
        built
    }

    fn build_missing_path(
        &mut self,
        depth: usize,
        history: &[Symbol],
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        if depth > self.max_depth {
            return ChildRef::NONE;
        }

        let hot_prefix_depth = self.hot_prefix_depth();
        if depth > hot_prefix_depth {
            return self.build_missing_segment_path(depth, history, sym_idx, singleton_log_prob_kt);
        }

        let mut counts = [0u32; 2];
        counts[sym_idx] = 1;
        let mut built = if hot_prefix_depth < self.max_depth {
            self.build_missing_segment_path(
                hot_prefix_depth + 1,
                history,
                sym_idx,
                singleton_log_prob_kt,
            )
        } else {
            ChildRef::NONE
        };

        for node_depth in (depth..=hot_prefix_depth).rev() {
            let node = self
                .arena
                .alloc_node_with_state(counts, singleton_log_prob_kt);
            if node_depth < self.max_depth {
                let edge = history_symbol(history, node_depth) as usize;
                self.arena.set_child(node, edge, built);
            }
            self.arena.recompute_node_weight(node);
            built = ChildRef::from_node(node);
        }
        built
    }

    #[inline(always)]
    fn build_missing_segment_path_exact_bits(
        &mut self,
        depth: usize,
        path_bits: u64,
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        debug_assert!(self.max_depth <= SEG_EXACT_MAX_LEN as usize);
        if depth > self.max_depth {
            return ChildRef::NONE;
        }

        let mut counts = [0u32; 2];
        counts[sym_idx] = 1;
        let total_len = self.max_depth - depth + 1;
        let payload = SegmentPayload::exact(
            path_bits & low_bits_mask_u64(total_len as u32),
            total_len as u32,
        );
        let segment = self.arena.alloc_segment_with_parts(
            counts,
            singleton_log_prob_kt,
            ChildRef::NONE,
            payload,
        );
        ChildRef::from_segment(segment)
    }

    #[inline(always)]
    fn build_missing_path_exact_bits(
        &mut self,
        depth: usize,
        path_bits: u64,
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        debug_assert!(self.max_depth <= SEG_EXACT_MAX_LEN as usize);
        if depth > self.max_depth {
            return ChildRef::NONE;
        }

        let hot_prefix_depth = self.hot_prefix_depth();
        if depth > hot_prefix_depth {
            return self.build_missing_segment_path_exact_bits(
                depth,
                path_bits,
                sym_idx,
                singleton_log_prob_kt,
            );
        }

        let mut counts = [0u32; 2];
        counts[sym_idx] = 1;
        let mut built = if hot_prefix_depth < self.max_depth {
            self.build_missing_segment_path_exact_bits(
                hot_prefix_depth + 1,
                shift_path_bits(path_bits, hot_prefix_depth + 1 - depth),
                sym_idx,
                singleton_log_prob_kt,
            )
        } else {
            ChildRef::NONE
        };

        for node_depth in (depth..=hot_prefix_depth).rev() {
            let node = self
                .arena
                .alloc_node_with_state(counts, singleton_log_prob_kt);
            if node_depth < self.max_depth {
                let edge = ((path_bits >> (node_depth - depth)) & 1) as usize;
                self.arena.set_child(node, edge, built);
            }
            self.arena.recompute_node_weight(node);
            built = ChildRef::from_node(node);
        }
        built
    }

    #[inline(always)]
    fn child_to_existing_source(child: ChildRef) -> Option<ExistingSource> {
        if let Some(node) = child.as_node() {
            Some(ExistingSource::Node(node))
        } else if let Some(segment) = child.as_segment() {
            Some(ExistingSource::Segment(segment, 0))
        } else {
            None
        }
    }

    #[inline(always)]
    fn update_source_state(
        &mut self,
        log_int: &[f64],
        log_half: &[f64],
        source: ExistingSource,
        sym_idx: usize,
    ) {
        match source {
            ExistingSource::Node(node_idx) => {
                let slot = node_idx.get();
                let mut counts = self.arena.nodes[slot].symbol_count;
                let mut log_prob_kt = self.arena.nodes[slot].log_prob_kt;
                apply_update_to_state_raw(
                    log_int,
                    log_half,
                    &mut counts,
                    &mut log_prob_kt,
                    sym_idx,
                );
                self.arena.nodes[slot].symbol_count = counts;
                self.arena.nodes[slot].log_prob_kt = log_prob_kt;
            }
            ExistingSource::Segment(segment_idx, _) => {
                let slot = segment_idx.get();
                let mut counts = self.arena.segments[slot].symbol_count;
                let mut log_prob_kt = self.arena.segments[slot].log_prob_kt;
                apply_update_to_state_raw(
                    log_int,
                    log_half,
                    &mut counts,
                    &mut log_prob_kt,
                    sym_idx,
                );
                self.arena.segments[slot].symbol_count = counts;
                self.arena.segments[slot].log_prob_kt = log_prob_kt;
            }
            ExistingSource::None => unreachable!("prepared update should never visit None"),
        }
    }

    #[inline(always)]
    fn recompute_source_weight(&mut self, source: ExistingSource) {
        match source {
            ExistingSource::Node(node_idx) => self.arena.recompute_node_weight(node_idx),
            ExistingSource::Segment(segment_idx, _) => self.recompute_segment_head(segment_idx),
            ExistingSource::None => unreachable!("prepared update should never visit None"),
        }
    }

    #[inline(always)]
    fn recompute_segment_head(&mut self, segment_idx: SegmentIndex) {
        let segment = self.arena.segments[segment_idx.get()];
        let head = if segment.tail.is_some() {
            let (alpha, log_alpha, log_one_minus_alpha) = self.segment_constants(segment.len());
            unary_chain_log_weight_precomputed(
                segment.log_prob_kt,
                self.arena.child_ref_weighted(segment.tail),
                alpha,
                log_alpha,
                log_one_minus_alpha,
            )
        } else {
            segment.log_prob_kt
        };
        self.arena.segments[segment_idx.get()].head_log_prob_weighted = head;
    }

    fn attach_missing_after_prepared_path(
        &mut self,
        history: &[Symbol],
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) {
        let Some(last_step) = self.prepared_steps.last().copied() else {
            return;
        };
        let depth = self.prepared_levels;
        match last_step.source {
            ExistingSource::Node(node_idx) => {
                debug_assert!(depth < self.max_depth);
                let path_edge = history_symbol(history, depth) as usize;
                debug_assert!(self.arena.child(node_idx, path_edge).is_none());
                let new_child =
                    self.build_missing_path(depth + 1, history, sym_idx, singleton_log_prob_kt);
                self.arena.set_child(node_idx, path_edge, new_child);
            }
            ExistingSource::Segment(segment_idx, offset) => {
                debug_assert!(depth < self.max_depth);
                debug_assert_eq!(offset + 1, self.arena.segment_len(segment_idx));
                debug_assert!(self.arena.segments[segment_idx.get()].tail.is_none());
                let new_tail =
                    self.build_missing_path(depth + 1, history, sym_idx, singleton_log_prob_kt);
                self.arena.set_segment_tail(segment_idx, new_tail);
            }
            ExistingSource::None => unreachable!("prepared path should never end in None source"),
        }
    }

    fn replace_prepared_child(
        &mut self,
        history: &[Symbol],
        step_index: usize,
        current_start_depth: usize,
        new_child: ChildRef,
    ) {
        if step_index == 0 {
            let root_edge = history_symbol(history, 0) as usize;
            self.arena.set_child(self.root, root_edge, new_child);
            return;
        }

        match self.prepared_steps[step_index - 1].source {
            ExistingSource::Node(node_idx) => {
                let edge = history_symbol(history, current_start_depth - 1) as usize;
                self.arena.set_child(node_idx, edge, new_child);
            }
            ExistingSource::Segment(segment_idx, offset) => {
                debug_assert_eq!(offset + 1, self.arena.segment_len(segment_idx));
                self.arena.set_segment_tail(segment_idx, new_child);
            }
            ExistingSource::None => unreachable!("prepared path should never parent from None"),
        }
    }

    fn update_prepared_mismatch(
        &mut self,
        log_int: &[f64],
        log_half: &[f64],
        history: &[Symbol],
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        let last_index = self.prepared_steps.len() - 1;
        for idx in 0..last_index {
            self.update_source_state(log_int, log_half, self.prepared_steps[idx].source, sym_idx);
        }

        let last_step = self.prepared_steps[last_index];
        let ExistingSource::Segment(segment_idx, offset_u32) = last_step.source else {
            unreachable!("prepared segment mismatch must end at a segment");
        };

        let original = self.arena.segments[segment_idx.get()];
        let offset = offset_u32 as usize;
        let seg_len = original.len() as usize;
        let history_len = history.len();
        let current_start_depth = self.prepared_levels - last_step.span as usize + 1;
        let node_depth = current_start_depth + offset;
        let path_edge = path_edge_at_depth(history, history_len, node_depth);
        let existing_edge = segment_edge_from_parts(original, offset, history, history_len);
        debug_assert_ne!(path_edge, existing_edge);

        let old_continuation = if offset + 1 < seg_len {
            if offset == 0 {
                let segment = &mut self.arena.segments[segment_idx.get()];
                segment.payload = original.payload.suffix_after(1);
                segment.tail = original.tail;
                segment.symbol_count = original.symbol_count;
                segment.log_prob_kt = original.log_prob_kt;
                self.recompute_segment_head(segment_idx);
                ChildRef::from_segment(segment_idx)
            } else {
                ChildRef::from_segment(self.arena.alloc_segment_with_parts(
                    original.symbol_count,
                    original.log_prob_kt,
                    original.tail,
                    original.payload.suffix_after(offset as u32 + 1),
                ))
            }
        } else {
            original.tail
        };

        let new_tail =
            self.build_missing_path(node_depth + 1, history, sym_idx, singleton_log_prob_kt);
        let mut updated_counts = original.symbol_count;
        let mut updated_log_prob_kt = original.log_prob_kt;
        apply_update_to_state_raw(
            log_int,
            log_half,
            &mut updated_counts,
            &mut updated_log_prob_kt,
            sym_idx,
        );

        let branch = self
            .arena
            .alloc_node_with_state(updated_counts, updated_log_prob_kt);
        self.arena
            .set_child(branch, existing_edge as usize, old_continuation);
        self.arena.set_child(branch, path_edge as usize, new_tail);
        self.arena.recompute_node_weight(branch);

        if offset == 0 {
            if seg_len == 1 {
                self.arena.free_segment(segment_idx);
            }
            self.replace_prepared_child(
                history,
                last_index,
                current_start_depth,
                ChildRef::from_node(branch),
            );
        } else {
            let segment = &mut self.arena.segments[segment_idx.get()];
            segment.payload = original.payload.prefix(offset as u32);
            segment.tail = ChildRef::from_node(branch);
            segment.symbol_count = updated_counts;
            segment.log_prob_kt = updated_log_prob_kt;
            self.recompute_segment_head(segment_idx);
        }

        for idx in (0..last_index).rev() {
            self.recompute_source_weight(self.prepared_steps[idx].source);
        }

        let root_edge = history_symbol(history, 0) as usize;
        self.arena.child(self.root, root_edge)
    }

    fn update_prepared_cached_path(
        &mut self,
        log_int: &[f64],
        log_half: &[f64],
        history: &[Symbol],
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) {
        debug_assert!(!self.prepared_steps.is_empty());
        debug_assert!(matches!(
            self.prepared_end,
            PreparedEnd::MaxDepth | PreparedEnd::MissingAfterCurrent
        ));

        if self.prepared_end == PreparedEnd::MissingAfterCurrent {
            self.attach_missing_after_prepared_path(history, sym_idx, singleton_log_prob_kt);
        }

        let last_index = self.prepared_steps.len() - 1;
        let mut child_weight = if self.prepared_end == PreparedEnd::MissingAfterCurrent {
            let last_step = self.prepared_steps[last_index];
            match last_step.source {
                ExistingSource::Node(node_idx) => {
                    let depth = self.prepared_levels;
                    let edge = history_symbol(history, depth) as usize;
                    self.arena
                        .child_ref_weighted(self.arena.child(node_idx, edge))
                }
                ExistingSource::Segment(segment_idx, offset) => {
                    debug_assert_eq!(offset + 1, self.arena.segment_len(segment_idx));
                    self.arena
                        .child_ref_weighted(self.arena.segments[segment_idx.get()].tail)
                }
                ExistingSource::None => unreachable!("prepared path should never end in None"),
            }
        } else {
            0.0
        };

        for idx in (0..=last_index).rev() {
            let step = self.prepared_steps[idx];
            match step.source {
                ExistingSource::Node(node_idx) => {
                    let mut counts = step.counts;
                    let mut log_prob_kt = step.kt_log_prob;
                    apply_update_to_state_raw(
                        log_int,
                        log_half,
                        &mut counts,
                        &mut log_prob_kt,
                        sym_idx,
                    );
                    let weighted =
                        if idx == last_index && self.prepared_end == PreparedEnd::MaxDepth {
                            debug_assert_eq!(step.has_sibling, 0);
                            clamp_log_prob(log_prob_kt)
                        } else {
                            update_weighted_log_prob(
                                log_prob_kt,
                                child_weight,
                                step.sibling_weight,
                                false,
                            )
                        };
                    let slot = node_idx.get();
                    self.arena.nodes[slot].symbol_count = counts;
                    self.arena.nodes[slot].log_prob_kt = log_prob_kt;
                    self.arena.nodes[slot].log_prob_weighted = weighted;
                    child_weight = weighted;
                }
                ExistingSource::Segment(segment_idx, offset) => {
                    let mut counts = step.counts;
                    let mut log_prob_kt = step.kt_log_prob;
                    apply_update_to_state_raw(
                        log_int,
                        log_half,
                        &mut counts,
                        &mut log_prob_kt,
                        sym_idx,
                    );
                    let slot = segment_idx.get();
                    let weighted =
                        if idx == last_index && self.prepared_end == PreparedEnd::MaxDepth {
                            debug_assert_eq!(offset + 1, self.arena.segment_len(segment_idx));
                            debug_assert!(self.arena.segments[slot].tail.is_none());
                            clamp_log_prob(log_prob_kt)
                        } else {
                            let (alpha, log_alpha, log_one_minus_alpha) =
                                self.segment_constants(self.arena.segments[slot].len());
                            unary_chain_log_weight_precomputed(
                                log_prob_kt,
                                child_weight,
                                alpha,
                                log_alpha,
                                log_one_minus_alpha,
                            )
                        };
                    self.arena.segments[slot].symbol_count = counts;
                    self.arena.segments[slot].log_prob_kt = log_prob_kt;
                    self.arena.segments[slot].head_log_prob_weighted = weighted;
                    child_weight = weighted;
                }
                ExistingSource::None => unreachable!("prepared update should never visit None"),
            }
        }
    }

    fn update_child_fast(
        &mut self,
        log_int: &[f64],
        log_half: &[f64],
        child: ChildRef,
        depth: usize,
        history: &[Symbol],
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        if depth > self.max_depth {
            return child;
        }
        if child.is_none() {
            return self.build_missing_path(depth, history, sym_idx, singleton_log_prob_kt);
        }

        if let Some(node_idx) = child.as_node() {
            if depth < self.max_depth {
                let path_edge = history_symbol(history, depth) as usize;
                let next = self.arena.child(node_idx, path_edge);
                let updated = self.update_child_fast(
                    log_int,
                    log_half,
                    next,
                    depth + 1,
                    history,
                    sym_idx,
                    singleton_log_prob_kt,
                );
                if updated != next {
                    self.arena.set_child(node_idx, path_edge, updated);
                }
            }
            let mut counts = self.arena.nodes[node_idx.get()].symbol_count;
            let mut log_prob_kt = self.arena.nodes[node_idx.get()].log_prob_kt;
            apply_update_to_state_raw(log_int, log_half, &mut counts, &mut log_prob_kt, sym_idx);
            self.arena.nodes[node_idx.get()].symbol_count = counts;
            self.arena.nodes[node_idx.get()].log_prob_kt = log_prob_kt;
            self.arena.recompute_node_weight(node_idx);
            return ChildRef::from_node(node_idx);
        }

        let segment_idx = child.as_segment().unwrap();
        let original = self.arena.segments[segment_idx.get()];
        let seg_len = original.len() as usize;
        let mut updated_counts = original.symbol_count;
        let mut updated_log_prob_kt = original.log_prob_kt;
        apply_update_to_state_raw(
            log_int,
            log_half,
            &mut updated_counts,
            &mut updated_log_prob_kt,
            sym_idx,
        );

        let depth_budget = self.max_depth.saturating_sub(depth);
        let comparable_len = if original.tail.is_none() {
            seg_len.saturating_sub(1)
        } else {
            seg_len
        }
        .min(depth_budget);
        let mismatch = first_segment_mismatch(original, depth, history, comparable_len).map(
            |(offset, path_edge, existing_edge)| (offset, depth + offset, path_edge, existing_edge),
        );

        if let Some((offset, node_depth, path_edge, existing_edge)) = mismatch {
            let old_continuation = if offset + 1 < seg_len {
                if offset == 0 {
                    let segment = &mut self.arena.segments[segment_idx.get()];
                    segment.payload = original.payload.suffix_after(1);
                    segment.tail = original.tail;
                    segment.symbol_count = original.symbol_count;
                    segment.log_prob_kt = original.log_prob_kt;
                    self.recompute_segment_head(segment_idx);
                    ChildRef::from_segment(segment_idx)
                } else {
                    ChildRef::from_segment(self.arena.alloc_segment_with_parts(
                        original.symbol_count,
                        original.log_prob_kt,
                        original.tail,
                        original.payload.suffix_after(offset as u32 + 1),
                    ))
                }
            } else {
                original.tail
            };

            let new_tail =
                self.build_missing_path(node_depth + 1, history, sym_idx, singleton_log_prob_kt);
            let branch = self
                .arena
                .alloc_node_with_state(updated_counts, updated_log_prob_kt);
            self.arena
                .set_child(branch, existing_edge as usize, old_continuation);
            self.arena.set_child(branch, path_edge as usize, new_tail);
            self.arena.recompute_node_weight(branch);

            if offset == 0 {
                if offset + 1 >= seg_len {
                    self.arena.free_segment(segment_idx);
                }
                return ChildRef::from_node(branch);
            }

            let segment = &mut self.arena.segments[segment_idx.get()];
            segment.payload = original.payload.prefix(offset as u32);
            segment.tail = ChildRef::from_node(branch);
            segment.symbol_count = updated_counts;
            segment.log_prob_kt = updated_log_prob_kt;
            self.recompute_segment_head(segment_idx);
            return ChildRef::from_segment(segment_idx);
        }

        if depth_budget < seg_len {
            self.arena.segments[segment_idx.get()].symbol_count = updated_counts;
            self.arena.segments[segment_idx.get()].log_prob_kt = updated_log_prob_kt;
            self.recompute_segment_head(segment_idx);
            return ChildRef::from_segment(segment_idx);
        }

        if original.tail.is_none() {
            let new_tail =
                self.build_missing_path(depth + seg_len, history, sym_idx, singleton_log_prob_kt);
            self.arena.segments[segment_idx.get()].tail = new_tail;
            self.arena.segments[segment_idx.get()].symbol_count = updated_counts;
            self.arena.segments[segment_idx.get()].log_prob_kt = updated_log_prob_kt;
            self.recompute_segment_head(segment_idx);
            return ChildRef::from_segment(segment_idx);
        }

        let tail = original.tail;
        let updated_tail = self.update_child_fast(
            log_int,
            log_half,
            tail,
            depth + seg_len,
            history,
            sym_idx,
            singleton_log_prob_kt,
        );
        if updated_tail != tail {
            self.arena.set_segment_tail(segment_idx, updated_tail);
        }
        self.arena.segments[segment_idx.get()].symbol_count = updated_counts;
        self.arena.segments[segment_idx.get()].log_prob_kt = updated_log_prob_kt;
        self.recompute_segment_head(segment_idx);
        ChildRef::from_segment(segment_idx)
    }

    fn update_child_fast_exact(
        &mut self,
        log_int: &[f64],
        log_half: &[f64],
        child: ChildRef,
        depth: usize,
        history: &[Symbol],
        path_bits: u64,
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        debug_assert!(self.max_depth <= SEG_EXACT_MAX_LEN as usize);
        if depth > self.max_depth {
            return child;
        }
        if child.is_none() {
            return self.build_missing_path_exact_bits(
                depth,
                path_bits,
                sym_idx,
                singleton_log_prob_kt,
            );
        }

        if let Some(node_idx) = child.as_node() {
            if depth < self.max_depth {
                let path_edge = (path_bits & 1) as usize;
                let next = self.arena.child(node_idx, path_edge);
                let updated = self.update_child_fast_exact(
                    log_int,
                    log_half,
                    next,
                    depth + 1,
                    history,
                    shift_path_bits(path_bits, 1),
                    sym_idx,
                    singleton_log_prob_kt,
                );
                if updated != next {
                    self.arena.set_child(node_idx, path_edge, updated);
                }
            }
            let mut counts = self.arena.nodes[node_idx.get()].symbol_count;
            let mut log_prob_kt = self.arena.nodes[node_idx.get()].log_prob_kt;
            apply_update_to_state_raw(log_int, log_half, &mut counts, &mut log_prob_kt, sym_idx);
            self.arena.nodes[node_idx.get()].symbol_count = counts;
            self.arena.nodes[node_idx.get()].log_prob_kt = log_prob_kt;
            self.arena.recompute_node_weight(node_idx);
            return ChildRef::from_node(node_idx);
        }

        let segment_idx = child.as_segment().unwrap();
        let original = self.arena.segments[segment_idx.get()];
        if !original.payload.is_exact() {
            return self.update_child_fast(
                log_int,
                log_half,
                child,
                depth,
                history,
                sym_idx,
                singleton_log_prob_kt,
            );
        }

        let seg_len = original.len() as usize;
        let mut updated_counts = original.symbol_count;
        let mut updated_log_prob_kt = original.log_prob_kt;
        apply_update_to_state_raw(
            log_int,
            log_half,
            &mut updated_counts,
            &mut updated_log_prob_kt,
            sym_idx,
        );

        let depth_budget = self.max_depth.saturating_sub(depth);
        let comparable_len = if original.tail.is_none() {
            seg_len.saturating_sub(1)
        } else {
            seg_len
        }
        .min(depth_budget);
        let mismatch =
            first_exact_segment_mismatch(original.payload.exact_bits(), path_bits, comparable_len)
                .map(|(offset, path_edge, existing_edge)| {
                    (offset, depth + offset, path_edge, existing_edge)
                });

        if let Some((offset, node_depth, path_edge, existing_edge)) = mismatch {
            let old_continuation = if offset + 1 < seg_len {
                if offset == 0 {
                    let segment = &mut self.arena.segments[segment_idx.get()];
                    segment.payload = original.payload.suffix_after(1);
                    segment.tail = original.tail;
                    segment.symbol_count = original.symbol_count;
                    segment.log_prob_kt = original.log_prob_kt;
                    self.recompute_segment_head(segment_idx);
                    ChildRef::from_segment(segment_idx)
                } else {
                    ChildRef::from_segment(self.arena.alloc_segment_with_parts(
                        original.symbol_count,
                        original.log_prob_kt,
                        original.tail,
                        original.payload.suffix_after(offset as u32 + 1),
                    ))
                }
            } else {
                original.tail
            };

            let new_tail = self.build_missing_path_exact_bits(
                node_depth + 1,
                shift_path_bits(path_bits, offset + 1),
                sym_idx,
                singleton_log_prob_kt,
            );
            let branch = self
                .arena
                .alloc_node_with_state(updated_counts, updated_log_prob_kt);
            self.arena
                .set_child(branch, existing_edge as usize, old_continuation);
            self.arena.set_child(branch, path_edge as usize, new_tail);
            self.arena.recompute_node_weight(branch);

            if offset == 0 {
                if offset + 1 >= seg_len {
                    self.arena.free_segment(segment_idx);
                }
                return ChildRef::from_node(branch);
            }

            let segment = &mut self.arena.segments[segment_idx.get()];
            segment.payload = original.payload.prefix(offset as u32);
            segment.tail = ChildRef::from_node(branch);
            segment.symbol_count = updated_counts;
            segment.log_prob_kt = updated_log_prob_kt;
            self.recompute_segment_head(segment_idx);
            return ChildRef::from_segment(segment_idx);
        }

        if depth_budget < seg_len {
            self.arena.segments[segment_idx.get()].symbol_count = updated_counts;
            self.arena.segments[segment_idx.get()].log_prob_kt = updated_log_prob_kt;
            self.recompute_segment_head(segment_idx);
            return ChildRef::from_segment(segment_idx);
        }

        if original.tail.is_none() {
            let new_tail = self.build_missing_path_exact_bits(
                depth + seg_len,
                shift_path_bits(path_bits, seg_len),
                sym_idx,
                singleton_log_prob_kt,
            );
            self.arena.segments[segment_idx.get()].tail = new_tail;
            self.arena.segments[segment_idx.get()].symbol_count = updated_counts;
            self.arena.segments[segment_idx.get()].log_prob_kt = updated_log_prob_kt;
            self.recompute_segment_head(segment_idx);
            return ChildRef::from_segment(segment_idx);
        }

        let tail = original.tail;
        let updated_tail = self.update_child_fast_exact(
            log_int,
            log_half,
            tail,
            depth + seg_len,
            history,
            shift_path_bits(path_bits, seg_len),
            sym_idx,
            singleton_log_prob_kt,
        );
        if updated_tail != tail {
            self.arena.set_segment_tail(segment_idx, updated_tail);
        }
        self.arena.segments[segment_idx.get()].symbol_count = updated_counts;
        self.arena.segments[segment_idx.get()].log_prob_kt = updated_log_prob_kt;
        self.recompute_segment_head(segment_idx);
        ChildRef::from_segment(segment_idx)
    }

    #[inline(always)]
    fn update_root_child(
        &mut self,
        log_int: &[f64],
        log_half: &[f64],
        child: ChildRef,
        history: &[Symbol],
        sym_idx: usize,
        singleton_log_prob_kt: f64,
    ) -> ChildRef {
        if self.max_depth <= SEG_EXACT_MAX_LEN as usize {
            let path_bits = path_bits_from_history(history, 1, self.max_depth);
            self.update_child_fast_exact(
                log_int,
                log_half,
                child,
                1,
                history,
                path_bits,
                sym_idx,
                singleton_log_prob_kt,
            )
        } else {
            self.update_child_fast(
                log_int,
                log_half,
                child,
                1,
                history,
                sym_idx,
                singleton_log_prob_kt,
            )
        }
    }

    fn collect_existing_levels(&mut self, history: &[Symbol]) -> ChildRef {
        if self.max_depth == 0 {
            self.detaches.clear();
            return ChildRef::NONE;
        }

        self.detaches.clear();
        self.levels.fill(LevelState::default());

        let root_edge = history_symbol(history, 0) as usize;
        let old_child = self.arena.child(self.root, root_edge);
        let mut source = if let Some(node) = old_child.as_node() {
            ExistingSource::Node(node)
        } else if let Some(segment) = old_child.as_segment() {
            ExistingSource::Segment(segment, 0)
        } else {
            ExistingSource::None
        };

        for depth in 1..=self.max_depth {
            let slot = depth - 1;
            self.levels[slot] = LevelState::default();

            match source {
                ExistingSource::None => {}
                ExistingSource::Node(node_idx) => {
                    self.levels[slot].symbol_count = self.arena.counts(node_idx);
                    self.levels[slot].log_prob_kt = self.arena.log_prob_kt(node_idx);
                    if depth < self.max_depth {
                        let path_edge = history_symbol(history, depth) as usize;
                        let sibling_edge = path_edge ^ 1;
                        let sibling = self.arena.child(node_idx, sibling_edge);
                        self.levels[slot].sibling = sibling;
                        if sibling.is_some() {
                            self.detaches.push(Detach::NodeChild {
                                node: node_idx,
                                edge: sibling_edge,
                            });
                        }
                        let next = self.arena.child(node_idx, path_edge);
                        source = if let Some(next_node) = next.as_node() {
                            ExistingSource::Node(next_node)
                        } else if let Some(next_segment) = next.as_segment() {
                            ExistingSource::Segment(next_segment, 0)
                        } else {
                            ExistingSource::None
                        };
                    }
                }
                ExistingSource::Segment(segment_idx, offset) => {
                    self.levels[slot].symbol_count = self.arena.segment_symbol_count(segment_idx);
                    self.levels[slot].log_prob_kt = self.arena.segment_log_prob_kt(segment_idx);
                    if depth < self.max_depth {
                        let path_edge = history_symbol(history, depth) as usize;
                        if self.arena.segment_has_child(segment_idx, offset) {
                            let existing_edge =
                                self.arena.segment_edge(segment_idx, offset, history);
                            if path_edge == existing_edge {
                                let seg_len = self.arena.segment_len(segment_idx);
                                if offset + 1 < seg_len {
                                    source = ExistingSource::Segment(segment_idx, offset + 1);
                                } else {
                                    let tail = self.arena.segments[segment_idx.get()].tail;
                                    source = if let Some(next_node) = tail.as_node() {
                                        ExistingSource::Node(next_node)
                                    } else if let Some(next_segment) = tail.as_segment() {
                                        ExistingSource::Segment(next_segment, 0)
                                    } else {
                                        ExistingSource::None
                                    };
                                }
                            } else {
                                let continuation = self.arena.detach_segment_continuation(
                                    segment_idx,
                                    offset,
                                    &mut self.detaches,
                                );
                                self.levels[slot].sibling = continuation;
                                source = ExistingSource::None;
                            }
                        } else {
                            source = ExistingSource::None;
                        }
                    }
                }
            }
        }

        old_child
    }

    fn rebuild_path_subtree(&mut self, history: &[Symbol]) -> ChildRef {
        let mut built = ChildRef::NONE;

        for depth in (1..=self.max_depth).rev() {
            let level = self.levels[depth - 1];
            let visits = level.symbol_count[0] + level.symbol_count[1];
            if visits == 0 {
                built = ChildRef::NONE;
                continue;
            }

            let path_edge = if depth < self.max_depth {
                history_symbol(history, depth) as usize
            } else {
                0
            };
            let has_path_child = built.is_some();
            let has_sibling = level.sibling.is_some();
            let force_node = depth <= self.hot_prefix_depth();

            if force_node || (has_path_child && has_sibling) {
                let node = self
                    .arena
                    .alloc_node_with_state(level.symbol_count, level.log_prob_kt);
                if has_path_child {
                    self.arena.set_child(node, path_edge, built);
                }
                if has_sibling {
                    self.arena.set_child(node, path_edge ^ 1, level.sibling);
                }
                self.arena.recompute_node_weight(node);
                built = ChildRef::from_node(node);
            } else {
                let (edge, child) = if has_path_child {
                    (path_edge, built)
                } else if has_sibling {
                    (path_edge ^ 1, level.sibling)
                } else {
                    (path_edge, ChildRef::NONE)
                };
                built = self.arena.prepend_or_alloc_segment(
                    history,
                    depth,
                    level.symbol_count,
                    level.log_prob_kt,
                    child,
                    edge,
                    false,
                );
            }
        }

        built
    }

    fn apply_detaches(&mut self) {
        for detach in self.detaches.drain(..) {
            match detach {
                Detach::NodeChild { node, edge } => {
                    self.arena.set_child(node, edge, ChildRef::NONE);
                }
                Detach::SegmentNext { segment, new_len } => {
                    self.arena.segments[segment.get()].set_len(new_len);
                    self.arena.set_segment_tail(segment, ChildRef::NONE);
                }
            }
        }
    }

    fn update_with_logs(
        &mut self,
        log_int: &[f64],
        log_half: &[f64],
        sym: Symbol,
        history: &[Symbol],
    ) {
        let sym_idx = sym as usize;
        let singleton_log_prob_kt = log_half[0] - log_int[1];
        {
            let slot = self.root.get();
            let mut counts = self.arena.nodes[slot].symbol_count;
            let mut log_prob_kt = self.arena.nodes[slot].log_prob_kt;
            apply_update_to_state_raw(log_int, log_half, &mut counts, &mut log_prob_kt, sym_idx);
            self.arena.nodes[slot].symbol_count = counts;
            self.arena.nodes[slot].log_prob_kt = log_prob_kt;
        }

        if self.max_depth > 0 {
            let root_edge = history_symbol(history, 0) as usize;
            let old_child = self.arena.child(self.root, root_edge);
            let new_child = self.update_root_child(
                log_int,
                log_half,
                old_child,
                history,
                sym_idx,
                singleton_log_prob_kt,
            );
            self.arena.set_child(self.root, root_edge, new_child);
        }

        self.arena.recompute_node_weight(self.root);
    }

    fn update(&mut self, sym: Symbol, history: &[Symbol]) {
        let upto = self.root_visits() + 1;
        self.with_logs(upto, |this, log_int, log_half| {
            this.update_with_logs(log_int, log_half, sym, history);
        });
    }

    fn update_prepared(&mut self, sym: Symbol, history: &[Symbol], use_prepared: bool) {
        let upto = self.root_visits() + 1;
        let sym_idx = sym as usize;
        self.with_logs(upto, |this, log_int, log_half| {
            let singleton_log_prob_kt = log_half[0] - log_int[1];
            {
                let slot = this.root.get();
                let mut counts = this.arena.nodes[slot].symbol_count;
                let mut log_prob_kt = this.arena.nodes[slot].log_prob_kt;
                apply_update_to_state_raw(
                    log_int,
                    log_half,
                    &mut counts,
                    &mut log_prob_kt,
                    sym_idx,
                );
                this.arena.nodes[slot].symbol_count = counts;
                this.arena.nodes[slot].log_prob_kt = log_prob_kt;
            }

            if this.max_depth > 0 {
                let root_edge = history_symbol(history, 0) as usize;
                let old_child = this.arena.child(this.root, root_edge);
                let new_child = if use_prepared {
                    match this.prepared_end {
                        PreparedEnd::MissingAtRoot => {
                            this.build_missing_path(1, history, sym_idx, singleton_log_prob_kt)
                        }
                        PreparedEnd::MaxDepth | PreparedEnd::MissingAfterCurrent => {
                            if !this.prepared_steps.is_empty() {
                                this.update_prepared_cached_path(
                                    log_int,
                                    log_half,
                                    history,
                                    sym_idx,
                                    singleton_log_prob_kt,
                                );
                            }
                            old_child
                        }
                        PreparedEnd::MismatchAtCurrentSegment => this.update_prepared_mismatch(
                            log_int,
                            log_half,
                            history,
                            sym_idx,
                            singleton_log_prob_kt,
                        ),
                    }
                } else {
                    this.update_root_child(
                        log_int,
                        log_half,
                        old_child,
                        history,
                        sym_idx,
                        singleton_log_prob_kt,
                    )
                };
                this.arena.set_child(this.root, root_edge, new_child);
            }

            this.arena.recompute_node_weight(this.root);
        });
    }

    fn revert(&mut self, sym: Symbol, history: &[Symbol]) {
        let upto = self.root_visits();
        let sym_idx = sym as usize;
        self.with_logs(upto, |this, log_int, log_half| {
            let old_child = this.collect_existing_levels(history);

            {
                let slot = this.root.get();
                let mut counts = this.arena.nodes[slot].symbol_count;
                let mut log_prob_kt = this.arena.nodes[slot].log_prob_kt;
                apply_revert_to_state_raw(
                    log_int,
                    log_half,
                    &mut counts,
                    &mut log_prob_kt,
                    sym_idx,
                );
                this.arena.nodes[slot].symbol_count = counts;
                this.arena.nodes[slot].log_prob_kt = log_prob_kt;
            }

            for level in &mut this.levels {
                let mut counts = level.symbol_count;
                let mut log_prob_kt = level.log_prob_kt;
                apply_revert_to_state_raw(
                    log_int,
                    log_half,
                    &mut counts,
                    &mut log_prob_kt,
                    sym_idx,
                );
                level.symbol_count = counts;
                level.log_prob_kt = log_prob_kt;
            }

            if this.max_depth > 0 {
                let new_child = this.rebuild_path_subtree(history);
                let root_edge = history_symbol(history, 0) as usize;
                this.apply_detaches();
                this.arena.free_child_ref(old_child);
                this.arena.set_child(this.root, root_edge, new_child);
            }

            this.arena.recompute_node_weight(this.root);
        });
    }

    fn predict(&mut self, sym: Symbol, history: &[Symbol]) -> f64 {
        self.prepared_steps.clear();
        self.prepared_levels = 0;
        self.prepared_end = PreparedEnd::MaxDepth;

        let (root_sibling, root_has_sibling, mut source) = if self.max_depth > 0 {
            let root_edge = history_symbol(history, 0) as usize;
            let path_child = self.arena.child(self.root, root_edge);
            let sibling = self.arena.child(self.root, root_edge ^ 1);
            (
                self.arena.child_ref_weighted(sibling),
                sibling.is_some() as u8,
                Self::child_to_existing_source(path_child).unwrap_or(ExistingSource::None),
            )
        } else {
            (0.0, 0, ExistingSource::None)
        };
        if self.max_depth > 0 && matches!(source, ExistingSource::None) {
            self.prepared_end = PreparedEnd::MissingAtRoot;
        }

        let history_len = history.len();
        let mut depth = 1usize;
        'walk: while depth <= self.max_depth {
            match source {
                ExistingSource::None => break,
                ExistingSource::Node(node_idx) => {
                    let slot = node_idx.get();
                    let counts = self.arena.nodes[slot].symbol_count;
                    let kt_log_prob = self.arena.nodes[slot].log_prob_kt;
                    if depth == self.max_depth {
                        self.prepared_steps.push(PreparedStep {
                            source: ExistingSource::Node(node_idx),
                            counts,
                            kt_log_prob,
                            span: 1,
                            sibling_weight: 0.0,
                            has_sibling: 0,
                        });
                        self.prepared_levels += 1;
                        break;
                    }
                    let path_edge = history_symbol(history, depth) as usize;
                    let sibling = self.arena.child(node_idx, path_edge ^ 1);
                    self.prepared_steps.push(PreparedStep {
                        source: ExistingSource::Node(node_idx),
                        counts,
                        kt_log_prob,
                        span: 1,
                        sibling_weight: self.arena.child_ref_weighted(sibling),
                        has_sibling: sibling.is_some() as u8,
                    });
                    self.prepared_levels += 1;
                    let next = self.arena.child(node_idx, path_edge);
                    source = Self::child_to_existing_source(next).unwrap_or(ExistingSource::None);
                    if matches!(source, ExistingSource::None) {
                        self.prepared_end = PreparedEnd::MissingAfterCurrent;
                        break;
                    }
                    depth += 1;
                }
                ExistingSource::Segment(segment_idx, _) => {
                    let segment = self.arena.segments[segment_idx.get()];
                    let seg_len = segment.len() as usize;
                    let counts = segment.symbol_count;
                    let kt_log_prob = segment.log_prob_kt;
                    for offset in 0..seg_len {
                        let node_depth = depth + offset;
                        let span = (offset + 1) as u32;

                        if node_depth == self.max_depth {
                            self.prepared_steps.push(PreparedStep {
                                source: ExistingSource::Segment(segment_idx, offset as u32),
                                counts,
                                kt_log_prob,
                                span,
                                sibling_weight: 0.0,
                                has_sibling: 0,
                            });
                            self.prepared_levels += span as usize;
                            break 'walk;
                        }

                        if offset + 1 >= seg_len && segment.tail.is_none() {
                            self.prepared_steps.push(PreparedStep {
                                source: ExistingSource::Segment(segment_idx, offset as u32),
                                counts,
                                kt_log_prob,
                                span,
                                sibling_weight: 0.0,
                                has_sibling: 0,
                            });
                            self.prepared_levels += span as usize;
                            self.prepared_end = PreparedEnd::MissingAfterCurrent;
                            break 'walk;
                        }

                        let path_edge = path_edge_at_depth(history, history_len, node_depth);
                        let existing_edge =
                            segment_edge_from_parts(segment, offset, history, history_len);
                        if path_edge != existing_edge {
                            self.prepared_steps.push(PreparedStep {
                                source: ExistingSource::Segment(segment_idx, offset as u32),
                                counts,
                                kt_log_prob,
                                span,
                                sibling_weight: self
                                    .arena
                                    .segment_continuation_weight(segment_idx, offset as u32),
                                has_sibling: 1,
                            });
                            self.prepared_levels += span as usize;
                            self.prepared_end = PreparedEnd::MismatchAtCurrentSegment;
                            break 'walk;
                        }

                        if offset + 1 < seg_len {
                            continue;
                        }

                        self.prepared_steps.push(PreparedStep {
                            source: ExistingSource::Segment(segment_idx, offset as u32),
                            counts,
                            kt_log_prob,
                            span,
                            sibling_weight: 0.0,
                            has_sibling: 0,
                        });
                        self.prepared_levels += span as usize;
                        let tail = segment.tail;
                        source =
                            Self::child_to_existing_source(tail).unwrap_or(ExistingSource::None);
                        if matches!(source, ExistingSource::None) {
                            self.prepared_end = PreparedEnd::MissingAfterCurrent;
                            break 'walk;
                        }
                        depth = node_depth + 1;
                        continue 'walk;
                    }
                }
            }
        }

        let sym_idx = sym as usize;
        if self.prepared_levels == 0 {
            let counts = self.arena.counts(self.root);
            let kt_log_prob = self.arena.log_prob_kt(self.root);
            return if self.prepared_end == PreparedEnd::MaxDepth || root_has_sibling == 0 {
                predict_ratio_kt(counts, sym_idx)
            } else {
                predict_ratio_internal(kt_log_prob, counts, 0.0, root_sibling, 0.5, sym_idx)
            };
        }

        let last_step = *self.prepared_steps.last().unwrap();
        let last_counts = last_step.counts;
        let last_kt_log_prob = last_step.kt_log_prob;
        let (mut child_weight, mut ratio) = if self.prepared_end == PreparedEnd::MaxDepth
            && self.prepared_levels == self.max_depth
        {
            (last_kt_log_prob, predict_ratio_kt(last_counts, sym_idx))
        } else if last_step.has_sibling == 0 {
            (last_kt_log_prob, predict_ratio_kt(last_counts, sym_idx))
        } else {
            combined_weight_ratio_internal(
                last_kt_log_prob,
                last_counts,
                0.0,
                last_step.sibling_weight,
                0.5,
                sym_idx,
            )
        };

        if let ExistingSource::Segment(_, _) = last_step.source {
            if last_step.span > 1 {
                let (alpha, log_alpha, log_one_minus_alpha) =
                    self.segment_constants(last_step.span - 1);
                (child_weight, ratio) = unary_chain_ratio_transform_precomputed(
                    last_step.kt_log_prob,
                    last_step.counts,
                    child_weight,
                    ratio,
                    alpha,
                    log_alpha,
                    log_one_minus_alpha,
                    sym_idx,
                );
            }
        }

        for idx in (0..self.prepared_steps.len() - 1).rev() {
            let step = self.prepared_steps[idx];
            match step.source {
                ExistingSource::Node(_) => {
                    (child_weight, ratio) = combined_weight_ratio_internal(
                        step.kt_log_prob,
                        step.counts,
                        child_weight,
                        step.sibling_weight,
                        ratio,
                        sym_idx,
                    );
                }
                ExistingSource::Segment(_, _) => {
                    let (alpha, log_alpha, log_one_minus_alpha) = self.segment_constants(step.span);
                    (child_weight, ratio) = unary_chain_ratio_transform_precomputed(
                        step.kt_log_prob,
                        step.counts,
                        child_weight,
                        ratio,
                        alpha,
                        log_alpha,
                        log_one_minus_alpha,
                        sym_idx,
                    );
                }
                ExistingSource::None => unreachable!("prepared step should never store None"),
            }
        }

        let root_counts = self.arena.counts(self.root);
        let root_kt_log_prob = self.arena.log_prob_kt(self.root);
        predict_ratio_internal(
            root_kt_log_prob,
            root_counts,
            child_weight,
            root_sibling,
            ratio,
            sym_idx,
        )
    }

    fn predict_one(&mut self, history: &[Symbol]) -> f64 {
        self.prepared_steps.clear();
        self.prepared_levels = 0;
        self.prepared_end = PreparedEnd::MaxDepth;

        let (root_sibling, root_has_sibling, mut source) = if self.max_depth > 0 {
            let root_edge = history_symbol(history, 0) as usize;
            let path_child = self.arena.child(self.root, root_edge);
            let sibling = self.arena.child(self.root, root_edge ^ 1);
            (
                self.arena.child_ref_weighted(sibling),
                sibling.is_some() as u8,
                Self::child_to_existing_source(path_child).unwrap_or(ExistingSource::None),
            )
        } else {
            (0.0, 0, ExistingSource::None)
        };
        if self.max_depth > 0 && matches!(source, ExistingSource::None) {
            self.prepared_end = PreparedEnd::MissingAtRoot;
        }

        let history_len = history.len();
        let mut depth = 1usize;
        'walk: while depth <= self.max_depth {
            match source {
                ExistingSource::None => break,
                ExistingSource::Node(node_idx) => {
                    let slot = node_idx.get();
                    let counts = self.arena.nodes[slot].symbol_count;
                    let kt_log_prob = self.arena.nodes[slot].log_prob_kt;
                    if depth == self.max_depth {
                        self.prepared_steps.push(PreparedStep {
                            source: ExistingSource::Node(node_idx),
                            counts,
                            kt_log_prob,
                            span: 1,
                            sibling_weight: 0.0,
                            has_sibling: 0,
                        });
                        self.prepared_levels += 1;
                        break;
                    }
                    let path_edge = history_symbol(history, depth) as usize;
                    let sibling = self.arena.child(node_idx, path_edge ^ 1);
                    self.prepared_steps.push(PreparedStep {
                        source: ExistingSource::Node(node_idx),
                        counts,
                        kt_log_prob,
                        span: 1,
                        sibling_weight: self.arena.child_ref_weighted(sibling),
                        has_sibling: sibling.is_some() as u8,
                    });
                    self.prepared_levels += 1;
                    let next = self.arena.child(node_idx, path_edge);
                    source = Self::child_to_existing_source(next).unwrap_or(ExistingSource::None);
                    if matches!(source, ExistingSource::None) {
                        self.prepared_end = PreparedEnd::MissingAfterCurrent;
                        break;
                    }
                    depth += 1;
                }
                ExistingSource::Segment(segment_idx, _) => {
                    let segment = self.arena.segments[segment_idx.get()];
                    let seg_len = segment.len() as usize;
                    let counts = segment.symbol_count;
                    let kt_log_prob = segment.log_prob_kt;
                    for offset in 0..seg_len {
                        let node_depth = depth + offset;
                        let span = (offset + 1) as u32;

                        if node_depth == self.max_depth {
                            self.prepared_steps.push(PreparedStep {
                                source: ExistingSource::Segment(segment_idx, offset as u32),
                                counts,
                                kt_log_prob,
                                span,
                                sibling_weight: 0.0,
                                has_sibling: 0,
                            });
                            self.prepared_levels += span as usize;
                            break 'walk;
                        }

                        if offset + 1 >= seg_len && segment.tail.is_none() {
                            self.prepared_steps.push(PreparedStep {
                                source: ExistingSource::Segment(segment_idx, offset as u32),
                                counts,
                                kt_log_prob,
                                span,
                                sibling_weight: 0.0,
                                has_sibling: 0,
                            });
                            self.prepared_levels += span as usize;
                            self.prepared_end = PreparedEnd::MissingAfterCurrent;
                            break 'walk;
                        }

                        let path_edge = path_edge_at_depth(history, history_len, node_depth);
                        let existing_edge =
                            segment_edge_from_parts(segment, offset, history, history_len);
                        if path_edge != existing_edge {
                            self.prepared_steps.push(PreparedStep {
                                source: ExistingSource::Segment(segment_idx, offset as u32),
                                counts,
                                kt_log_prob,
                                span,
                                sibling_weight: self
                                    .arena
                                    .segment_continuation_weight(segment_idx, offset as u32),
                                has_sibling: 1,
                            });
                            self.prepared_levels += span as usize;
                            self.prepared_end = PreparedEnd::MismatchAtCurrentSegment;
                            break 'walk;
                        }

                        if offset + 1 < seg_len {
                            continue;
                        }

                        self.prepared_steps.push(PreparedStep {
                            source: ExistingSource::Segment(segment_idx, offset as u32),
                            counts,
                            kt_log_prob,
                            span,
                            sibling_weight: 0.0,
                            has_sibling: 0,
                        });
                        self.prepared_levels += span as usize;
                        let tail = segment.tail;
                        source =
                            Self::child_to_existing_source(tail).unwrap_or(ExistingSource::None);
                        if matches!(source, ExistingSource::None) {
                            self.prepared_end = PreparedEnd::MissingAfterCurrent;
                            break 'walk;
                        }
                        depth = node_depth + 1;
                        continue 'walk;
                    }
                }
            }
        }

        if self.prepared_levels == 0 {
            let counts = self.arena.counts(self.root);
            let kt_log_prob = self.arena.log_prob_kt(self.root);
            return if self.prepared_end == PreparedEnd::MaxDepth || root_has_sibling == 0 {
                predict_ratio_kt_one(counts)
            } else {
                predict_ratio_internal_one(kt_log_prob, counts, 0.0, root_sibling, 0.5)
            };
        }

        let last_step = *self.prepared_steps.last().unwrap();
        let last_counts = last_step.counts;
        let last_kt_log_prob = last_step.kt_log_prob;
        let (mut child_weight, mut ratio) = if self.prepared_end == PreparedEnd::MaxDepth
            && self.prepared_levels == self.max_depth
        {
            (last_kt_log_prob, predict_ratio_kt_one(last_counts))
        } else if last_step.has_sibling == 0 {
            (last_kt_log_prob, predict_ratio_kt_one(last_counts))
        } else {
            combined_weight_ratio_internal_one(
                last_kt_log_prob,
                last_counts,
                0.0,
                last_step.sibling_weight,
                0.5,
            )
        };

        if let ExistingSource::Segment(_, _) = last_step.source
            && last_step.span > 1
        {
            let (alpha, log_alpha, log_one_minus_alpha) =
                self.segment_constants(last_step.span - 1);
            (child_weight, ratio) = unary_chain_ratio_transform_precomputed_one(
                last_step.kt_log_prob,
                last_step.counts,
                child_weight,
                ratio,
                alpha,
                log_alpha,
                log_one_minus_alpha,
            );
        }

        for idx in (0..self.prepared_steps.len() - 1).rev() {
            let step = self.prepared_steps[idx];
            match step.source {
                ExistingSource::Node(_) => {
                    (child_weight, ratio) = combined_weight_ratio_internal_one(
                        step.kt_log_prob,
                        step.counts,
                        child_weight,
                        step.sibling_weight,
                        ratio,
                    );
                }
                ExistingSource::Segment(_, _) => {
                    let (alpha, log_alpha, log_one_minus_alpha) = self.segment_constants(step.span);
                    (child_weight, ratio) = unary_chain_ratio_transform_precomputed_one(
                        step.kt_log_prob,
                        step.counts,
                        child_weight,
                        ratio,
                        alpha,
                        log_alpha,
                        log_one_minus_alpha,
                    );
                }
                ExistingSource::None => unreachable!("prepared step should never store None"),
            }
        }

        let root_counts = self.arena.counts(self.root);
        let root_kt_log_prob = self.arena.log_prob_kt(self.root);
        predict_ratio_internal_one(
            root_kt_log_prob,
            root_counts,
            child_weight,
            root_sibling,
            ratio,
        )
    }

    fn memory_usage(&self) -> usize {
        self.arena.memory_usage()
            + self.segment_alpha.capacity() * size_of::<f64>()
            + self.segment_log_alpha.capacity() * size_of::<f64>()
            + self.segment_log_one_minus_alpha.capacity() * size_of::<f64>()
            + self.levels.capacity() * size_of::<LevelState>()
            + self.detaches.capacity() * size_of::<Detach>()
            + self.prepared_steps.capacity() * size_of::<PreparedStep>()
    }
}

/// A Context Tree for binary sequence prediction.
#[derive(Clone)]
pub struct ContextTree {
    engine: CtEngine,
    history: Vec<Symbol>,
}

impl ContextTree {
    pub fn new(depth: usize) -> Self {
        Self {
            engine: CtEngine::new(depth),
            history: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.engine.clear();
    }

    #[inline]
    pub fn update(&mut self, sym: Symbol) {
        self.engine.update(sym, &self.history);
        self.history.push(sym);
    }

    #[inline]
    pub fn revert(&mut self) {
        let Some(last_sym) = self.history.pop() else {
            return;
        };
        self.engine.revert(last_sym, &self.history);
    }

    #[inline]
    pub fn update_history(&mut self, symbols: &[Symbol]) {
        self.history.extend_from_slice(symbols);
    }

    #[inline]
    pub fn revert_history(&mut self) {
        self.history.pop();
    }

    pub fn truncate_history(&mut self, new_size: usize) {
        if new_size < self.history.len() {
            self.history.truncate(new_size);
        }
    }

    #[inline]
    pub fn predict(&mut self, sym: Symbol) -> f64 {
        self.engine.predict(sym, &self.history)
    }

    #[inline]
    pub fn predict_sym_prob(&mut self) -> f64 {
        self.predict(true)
    }

    #[inline]
    pub fn get_log_block_probability(&self) -> f64 {
        self.engine.get_log_block_probability()
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.engine.max_depth
    }

    #[inline]
    pub fn history_size(&self) -> usize {
        self.history.len()
    }
}

#[derive(Clone)]
struct ContextTreeCore {
    engine: CtEngine,
    prepared_valid: bool,
    prepared_history_len: usize,
    prepared_history_version: u64,
}

impl ContextTreeCore {
    fn new(depth: usize) -> Self {
        Self {
            engine: CtEngine::new(depth),
            prepared_valid: false,
            prepared_history_len: 0,
            prepared_history_version: 0,
        }
    }

    fn clear(&mut self) {
        self.engine.clear();
        self.prepared_valid = false;
        self.prepared_history_len = 0;
        self.prepared_history_version = 0;
    }

    #[inline]
    fn reserve_for_symbols(&mut self, total_symbols: usize) {
        self.engine.reserve_for_symbols(total_symbols);
    }

    #[inline]
    fn update(&mut self, sym: Symbol, shared_history: &[Symbol]) {
        self.prepared_valid = false;
        self.engine.update(sym, shared_history);
    }

    #[inline]
    fn update_predicted(&mut self, sym: Symbol, shared_history: &[Symbol], history_version: u64) {
        let use_prepared = self.prepared_valid
            && self.prepared_history_len == shared_history.len()
            && self.prepared_history_version == history_version;
        self.prepared_valid = false;
        self.engine
            .update_prepared(sym, shared_history, use_prepared);
    }

    #[inline]
    fn revert(&mut self, last_sym: Symbol, shared_history: &[Symbol]) {
        self.prepared_valid = false;
        self.engine.revert(last_sym, shared_history);
    }

    #[inline]
    fn predict(&mut self, sym: Symbol, shared_history: &[Symbol], history_version: u64) -> f64 {
        let prob = self.engine.predict(sym, shared_history);
        self.prepared_valid = true;
        self.prepared_history_len = shared_history.len();
        self.prepared_history_version = history_version;
        prob
    }

    #[inline]
    fn predict_one(&mut self, shared_history: &[Symbol], history_version: u64) -> f64 {
        let prob = self.engine.predict_one(shared_history);
        self.prepared_valid = true;
        self.prepared_history_len = shared_history.len();
        self.prepared_history_version = history_version;
        prob
    }

    #[inline]
    fn get_log_block_probability(&self) -> f64 {
        self.engine.get_log_block_probability()
    }
}

/// Factorized Action-Conditional Context Tree Weighting.
#[derive(Clone)]
pub struct FacContextTree {
    trees: Vec<ContextTreeCore>,
    shared_history: Vec<Symbol>,
    base_depth: usize,
    num_bits: usize,
    shared_history_version: u64,
}

impl FacContextTree {
    pub fn new(base_depth: usize, num_percept_bits: usize) -> Self {
        let trees = (0..num_percept_bits)
            .map(|i| ContextTreeCore::new(base_depth + i))
            .collect();
        Self {
            trees,
            shared_history: Vec::new(),
            base_depth,
            num_bits: num_percept_bits,
            shared_history_version: 0,
        }
    }

    #[inline(always)]
    fn bump_shared_history_version(&mut self) {
        self.shared_history_version = self.shared_history_version.wrapping_add(1);
    }

    #[inline]
    pub fn reserve_for_symbols(&mut self, total_symbols: usize) {
        if total_symbols == 0 {
            return;
        }
        self.shared_history
            .reserve_exact(total_symbols.saturating_mul(self.num_bits));
        for tree in &mut self.trees {
            tree.reserve_for_symbols(total_symbols);
        }
    }

    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    #[inline]
    pub fn base_depth(&self) -> usize {
        self.base_depth
    }

    #[inline]
    pub fn update(&mut self, sym: Symbol, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits);
        self.trees[bit_index].update(sym, &self.shared_history);
        self.shared_history.push(sym);
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn update_byte_msb(&mut self, byte: u8) {
        if self.num_bits != 8 {
            for bit_idx in 0..self.num_bits {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                self.update(bit, bit_idx);
            }
            return;
        }

        let upto = self.trees[0].engine.root_visits() + 1;
        debug_assert!(
            self.trees
                .iter()
                .all(|tree| tree.engine.root_visits() + 1 == upto)
        );
        with_shared_log_cache(upto, |log_int, log_half| {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                let tree = &mut self.trees[bit_idx];
                tree.prepared_valid = false;
                tree.engine
                    .update_with_logs(log_int, log_half, bit, &self.shared_history);
                self.shared_history.push(bit);
            }
        });
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn update_byte_lsb(&mut self, byte: u8) {
        let bits = self.num_bits.clamp(1, 8);
        let upto = self.trees[0].engine.root_visits() + 1;
        debug_assert!(
            self.trees
                .iter()
                .take(bits)
                .all(|tree| tree.engine.root_visits() + 1 == upto)
        );
        with_shared_log_cache(upto, |log_int, log_half| {
            for bit_idx in 0..bits {
                let bit = ((byte >> bit_idx) & 1) == 1;
                let tree = &mut self.trees[bit_idx];
                tree.prepared_valid = false;
                tree.engine
                    .update_with_logs(log_int, log_half, bit, &self.shared_history);
                self.shared_history.push(bit);
            }
        });
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn update_predicted(&mut self, sym: Symbol, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits);
        self.trees[bit_index].update_predicted(
            sym,
            &self.shared_history,
            self.shared_history_version,
        );
        self.shared_history.push(sym);
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn predict(&mut self, sym: Symbol, bit_index: usize) -> f64 {
        debug_assert!(bit_index < self.num_bits);
        self.trees[bit_index].predict(sym, &self.shared_history, self.shared_history_version)
    }

    #[inline]
    pub(crate) fn predict_one(&mut self, bit_index: usize) -> f64 {
        debug_assert!(bit_index < self.num_bits);
        self.trees[bit_index].predict_one(&self.shared_history, self.shared_history_version)
    }

    #[inline]
    pub fn revert(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits);
        let Some(last_sym) = self.shared_history.pop() else {
            return;
        };
        self.trees[bit_index].revert(last_sym, &self.shared_history);
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn update_history(&mut self, symbols: &[Symbol]) {
        if symbols.is_empty() {
            return;
        }
        self.shared_history.extend_from_slice(symbols);
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn revert_history(&mut self, count: usize) {
        let old_len = self.shared_history.len();
        let new_len = self.shared_history.len().saturating_sub(count);
        if new_len == old_len {
            return;
        }
        self.shared_history.truncate(new_len);
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn reset_history_only(&mut self) {
        if self.shared_history.is_empty() {
            return;
        }
        self.shared_history.clear();
        self.bump_shared_history_version();
    }

    #[inline]
    pub fn get_log_block_probability(&self) -> f64 {
        self.trees
            .iter()
            .map(|t| t.get_log_block_probability())
            .sum()
    }

    pub fn clear(&mut self) {
        for tree in &mut self.trees {
            tree.clear();
        }
        self.shared_history.clear();
        self.shared_history_version = 0;
    }

    pub fn memory_usage(&self) -> usize {
        let tree_mem: usize = self.trees.iter().map(|t| t.engine.memory_usage()).sum();
        let log_cache_mem = self
            .trees
            .first()
            .map(|t| t.engine.log_cache_memory_usage())
            .unwrap_or(0);
        let history_mem = self.shared_history.capacity() * size_of::<Symbol>();
        tree_mem + log_cache_mem + history_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct RefNode {
        children: [Option<Box<RefNode>>; 2],
        log_prob_kt: f64,
        log_prob_weighted: f64,
        symbol_count: [u32; 2],
    }

    impl Default for RefNode {
        fn default() -> Self {
            Self {
                children: [None, None],
                log_prob_kt: 0.0,
                log_prob_weighted: 0.0,
                symbol_count: [0, 0],
            }
        }
    }

    #[derive(Clone)]
    struct RefContextTree {
        root: RefNode,
        history: Vec<Symbol>,
        max_depth: usize,
        log_int: Vec<f64>,
        log_half: Vec<f64>,
    }

    impl RefContextTree {
        fn new(depth: usize) -> Self {
            Self {
                root: RefNode::default(),
                history: Vec::new(),
                max_depth: depth,
                log_int: vec![f64::NEG_INFINITY],
                log_half: vec![(0.5f64).ln()],
            }
        }

        fn root_visits(&self) -> usize {
            (self.root.symbol_count[0] + self.root.symbol_count[1]) as usize
        }

        fn recompute(node: &mut RefNode) {
            let w0 = node.children[0]
                .as_ref()
                .map(|c| c.log_prob_weighted)
                .unwrap_or(0.0);
            let w1 = node.children[1]
                .as_ref()
                .map(|c| c.log_prob_weighted)
                .unwrap_or(0.0);
            let is_leaf = node.children[0].is_none() && node.children[1].is_none();
            node.log_prob_weighted = update_weighted_log_prob(node.log_prob_kt, w0, w1, is_leaf);
        }

        fn update(&mut self, sym: Symbol) {
            let upto = self.root_visits() + 1;
            ensure_log_caches(&mut self.log_int, &mut self.log_half, upto);
            let sym_idx = sym as usize;
            Self::update_node(
                &mut self.root,
                0,
                self.max_depth,
                &self.history,
                sym_idx,
                &self.log_int,
                &self.log_half,
            );
            self.history.push(sym);
        }

        fn revert(&mut self) {
            let Some(last_sym) = self.history.pop() else {
                return;
            };
            let upto = self.root_visits();
            ensure_log_caches(&mut self.log_int, &mut self.log_half, upto);
            let sym_idx = last_sym as usize;
            let _ = Self::revert_node(
                &mut self.root,
                0,
                self.max_depth,
                &self.history,
                sym_idx,
                &self.log_int,
                &self.log_half,
            );
        }

        fn predict(&mut self, sym: Symbol) -> f64 {
            let sym_idx = sym as usize;
            let mut entries = Vec::with_capacity(self.max_depth + 1);
            let reached_max_depth = Self::collect_predict_entries(
                &self.root,
                0,
                self.max_depth,
                &self.history,
                &mut entries,
            );

            let deepest = entries.len() - 1;
            let mut ratio = if reached_max_depth && deepest == self.max_depth {
                predict_ratio_kt(entries[deepest].symbol_count, sym_idx)
            } else {
                0.5
            };
            for idx in (0..=deepest).rev() {
                if reached_max_depth && idx == deepest {
                    continue;
                }
                let child_weight = if idx + 1 <= deepest {
                    entries[idx + 1].log_prob_weighted
                } else {
                    0.0
                };
                ratio = predict_ratio_internal(
                    entries[idx].log_prob_kt,
                    entries[idx].symbol_count,
                    child_weight,
                    entries[idx].sibling_weight,
                    ratio,
                    sym_idx,
                );
            }
            ratio
        }

        fn get_log_block_probability(&self) -> f64 {
            self.root.log_prob_weighted
        }

        fn update_node(
            node: &mut RefNode,
            depth: usize,
            max_depth: usize,
            history: &[Symbol],
            sym_idx: usize,
            log_int: &[f64],
            log_half: &[f64],
        ) {
            if depth < max_depth {
                let edge = history_symbol(history, depth) as usize;
                if node.children[edge].is_none() {
                    node.children[edge] = Some(Box::new(RefNode::default()));
                }
                Self::update_node(
                    node.children[edge].as_deref_mut().unwrap(),
                    depth + 1,
                    max_depth,
                    history,
                    sym_idx,
                    log_int,
                    log_half,
                );
            }
            apply_update_to_state_raw(
                log_int,
                log_half,
                &mut node.symbol_count,
                &mut node.log_prob_kt,
                sym_idx,
            );
            Self::recompute(node);
        }

        fn revert_node(
            node: &mut RefNode,
            depth: usize,
            max_depth: usize,
            history: &[Symbol],
            sym_idx: usize,
            log_int: &[f64],
            log_half: &[f64],
        ) -> bool {
            if depth < max_depth {
                let edge = history_symbol(history, depth) as usize;
                let remove_child = if let Some(child) = node.children[edge].as_deref_mut() {
                    Self::revert_node(
                        child,
                        depth + 1,
                        max_depth,
                        history,
                        sym_idx,
                        log_int,
                        log_half,
                    )
                } else {
                    false
                };
                if remove_child {
                    node.children[edge] = None;
                }
            }
            apply_revert_to_state_raw(
                log_int,
                log_half,
                &mut node.symbol_count,
                &mut node.log_prob_kt,
                sym_idx,
            );
            Self::recompute(node);
            node.symbol_count[0] + node.symbol_count[1] == 0
        }

        fn collect_predict_entries(
            node: &RefNode,
            depth: usize,
            max_depth: usize,
            history: &[Symbol],
            entries: &mut Vec<PredictEntry>,
        ) -> bool {
            let sibling_weight = if depth < max_depth {
                let path_edge = history_symbol(history, depth) as usize;
                node.children[path_edge ^ 1]
                    .as_ref()
                    .map(|c| c.log_prob_weighted)
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            entries.push(PredictEntry {
                symbol_count: node.symbol_count,
                log_prob_kt: node.log_prob_kt,
                log_prob_weighted: node.log_prob_weighted,
                sibling_weight,
                has_sibling: depth < max_depth
                    && node.children[(history_symbol(history, depth) as usize) ^ 1].is_some(),
            });
            if depth == max_depth {
                return true;
            }
            let edge = history_symbol(history, depth) as usize;
            let Some(child) = node.children[edge].as_ref() else {
                return false;
            };
            Self::collect_predict_entries(child, depth + 1, max_depth, history, entries)
        }
    }

    #[derive(Clone)]
    struct RefFacContextTree {
        trees: Vec<RefContextTree>,
        history: Vec<Symbol>,
    }

    impl RefFacContextTree {
        fn new(base_depth: usize, num_bits: usize) -> Self {
            Self {
                trees: (0..num_bits)
                    .map(|i| RefContextTree::new(base_depth + i))
                    .collect(),
                history: Vec::new(),
            }
        }

        fn update(&mut self, sym: Symbol, bit_index: usize) {
            let tree = &mut self.trees[bit_index];
            tree.history = self.history.clone();
            tree.update(sym);
            self.history.push(sym);
        }

        fn predict(&mut self, sym: Symbol, bit_index: usize) -> f64 {
            let tree = &mut self.trees[bit_index];
            tree.history = self.history.clone();
            tree.predict(sym)
        }

        fn revert(&mut self, bit_index: usize) {
            let Some(last_sym) = self.history.pop() else {
                return;
            };
            let tree = &mut self.trees[bit_index];
            tree.history = self.history.clone();
            tree.history.push(last_sym);
            tree.revert();
        }

        fn get_log_block_probability(&self) -> f64 {
            self.trees
                .iter()
                .map(RefContextTree::get_log_block_probability)
                .sum()
        }
    }

    fn assert_close(a: f64, b: f64) {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(diff <= 1e-12 * scale, "a={a} b={b} diff={diff}");
    }

    fn child_after_hot_prefix(tree: &ContextTree, history_before_update: &[Symbol]) -> ChildRef {
        let hot_prefix_depth = tree.engine.hot_prefix_depth();
        if hot_prefix_depth == 0 {
            return ChildRef::NONE;
        }

        let root_edge = history_symbol(history_before_update, 0) as usize;
        let mut current = tree
            .engine
            .arena
            .child(tree.engine.root, root_edge)
            .as_node()
            .expect("hot-prefix node");
        for node_depth in 1..hot_prefix_depth {
            let edge = history_symbol(history_before_update, node_depth) as usize;
            current = tree
                .engine
                .arena
                .child(current, edge)
                .as_node()
                .expect("next hot-prefix node");
        }
        let tail_edge = history_symbol(history_before_update, hot_prefix_depth) as usize;
        tree.engine.arena.child(current, tail_edge)
    }

    #[test]
    #[should_panic(expected = "ctw node index overflow")]
    fn node_index_from_usize_rejects_overflow() {
        let _ = NodeIndex::from_usize(INDEX_LIMIT);
    }

    #[test]
    #[should_panic(expected = "ctw node index overflow")]
    fn node_index_from_usize_rejects_large_values() {
        let _ = NodeIndex::from_usize(u32::MAX as usize);
    }

    #[test]
    fn ctw_count_lane_stays_packed() {
        assert_eq!(std::mem::size_of::<CtNode>(), 32);
    }

    #[test]
    fn ctw_segment_payload_stays_packed() {
        assert_eq!(std::mem::size_of::<CtSegment>(), 40);
    }

    #[test]
    fn context_tree_singleton_paths_use_hot_prefix_nodes() {
        let mut tree = ContextTree::new(12);
        tree.update(false);

        let hot_prefix_depth = tree.engine.hot_prefix_depth();
        let child = tree.engine.arena.child(tree.engine.root, 0);
        let mut current = child.as_node().expect("hot-prefix node");
        let mut visited_hot_prefix_nodes = 1usize;
        for depth in 1..hot_prefix_depth {
            let next = tree.engine.arena.child(current, 0);
            current = next.as_node().expect("next hot-prefix node");
            visited_hot_prefix_nodes += 1;
            assert!(depth < hot_prefix_depth);
        }
        assert_eq!(visited_hot_prefix_nodes, hot_prefix_depth);
        let segment = tree
            .engine
            .arena
            .child(current, 0)
            .as_segment()
            .expect("segment tail");
        assert!(tree.engine.arena.child(current, 1).is_none());
        assert!(tree.engine.arena.segments[segment.get()].tail.is_none());
        assert_close(
            tree.engine.arena.segments[segment.get()].head_log_prob_weighted,
            -std::f64::consts::LN_2,
        );
        assert_close(tree.get_log_block_probability(), -std::f64::consts::LN_2);
    }

    #[test]
    fn context_tree_missing_path_tail_uses_exact_segment_payloads() {
        let mut tree = ContextTree::new(12);
        tree.update(true);
        let child = tree.engine.arena.child(tree.engine.root, 0);
        let mut current = child.as_node().expect("hot-prefix node");
        for _ in 1..tree.engine.hot_prefix_depth() {
            current = tree
                .engine
                .arena
                .child(current, 0)
                .as_node()
                .expect("next hot-prefix node");
        }
        let segment = tree
            .engine
            .arena
            .child(current, 0)
            .as_segment()
            .expect("segment tail");
        let payload = tree.engine.arena.segments[segment.get()].payload;
        assert!(payload.is_exact());
        assert_eq!(
            payload.len() as usize,
            tree.engine.max_depth - tree.engine.hot_prefix_depth()
        );
        assert_eq!(payload.exact_bits() & low_bits_mask_u64(payload.len()), 0);
    }

    #[test]
    fn context_tree_missing_path_tail_uses_const_payload_beyond_exact_limit() {
        let mut tree = ContextTree::new(80);
        let history_before = tree.history.clone();
        tree.update(false);

        let segment = child_after_hot_prefix(&tree, &history_before)
            .as_segment()
            .expect("segment tail");
        let segment = tree.engine.arena.segments[segment.get()];
        assert_eq!(segment.payload.mode(), SEG_MODE_CONST);
        assert_eq!(
            segment.payload.len() as usize,
            tree.engine.max_depth - tree.engine.hot_prefix_depth()
        );
        assert!(!segment.payload.const_bit());
        assert!(segment.tail.is_none());
    }

    #[test]
    fn context_tree_missing_path_tail_uses_history_and_const_payloads_beyond_exact_limit() {
        let mut tree = ContextTree::new(80);
        let seeded_history: Vec<Symbol> = (0..80).map(|i| (i & 1) == 1).collect();
        tree.update_history(&seeded_history);
        let history_before = tree.history.clone();
        tree.update(false);

        let first_segment = child_after_hot_prefix(&tree, &history_before)
            .as_segment()
            .expect("history-backed segment tail");
        let first_segment = tree.engine.arena.segments[first_segment.get()];
        assert_eq!(first_segment.payload.mode(), SEG_MODE_HISTORY);
        assert_eq!(first_segment.payload.len(), 69);
        for offset in [0usize, 1, 7, 31, 68] {
            assert_eq!(
                segment_edge_from_parts(
                    first_segment,
                    offset,
                    &history_before,
                    history_before.len()
                ),
                history_symbol(&history_before, tree.engine.hot_prefix_depth() + 1 + offset)
            );
        }

        let tail_segment = first_segment
            .tail
            .as_segment()
            .expect("constant fallback tail");
        let tail_segment = tree.engine.arena.segments[tail_segment.get()];
        assert_eq!(tail_segment.payload.mode(), SEG_MODE_CONST);
        assert_eq!(tail_segment.payload.len(), 1);
        assert!(!tail_segment.payload.const_bit());
        assert!(tail_segment.tail.is_none());
    }

    #[test]
    fn context_tree_matches_reference_on_short_sequences() {
        for depth in 0..=6usize {
            for len in 0..=6usize {
                for mask in 0..(1usize << len) {
                    let mut prod = ContextTree::new(depth);
                    let mut reference = RefContextTree::new(depth);
                    for step in 0..len {
                        let p_prod_0 = prod.predict(false);
                        let p_ref_0 = reference.predict(false);
                        assert!(
                            (p_prod_0 - p_ref_0).abs()
                                <= 1e-12 * p_prod_0.abs().max(p_ref_0.abs()).max(1.0),
                            "predict0 mismatch depth={depth} len={len} mask={mask} step={step} prod={p_prod_0} ref={p_ref_0} history={:?}",
                            prod.history
                        );
                        let p_prod_1 = prod.predict(true);
                        let p_ref_1 = reference.predict(true);
                        assert!(
                            (p_prod_1 - p_ref_1).abs()
                                <= 1e-12 * p_prod_1.abs().max(p_ref_1.abs()).max(1.0),
                            "predict1 mismatch depth={depth} len={len} mask={mask} step={step} prod={p_prod_1} ref={p_ref_1} history={:?}",
                            prod.history
                        );
                        let log_prod = prod.get_log_block_probability();
                        let log_ref = reference.get_log_block_probability();
                        assert!(
                            (log_prod - log_ref).abs()
                                <= 1e-12 * log_prod.abs().max(log_ref.abs()).max(1.0),
                            "log mismatch before update depth={depth} len={len} mask={mask} step={step} prod={log_prod} ref={log_ref} history={:?}",
                            prod.history
                        );
                        let bit = ((mask >> step) & 1) == 1;
                        prod.update(bit);
                        reference.update(bit);
                        let log_prod = prod.get_log_block_probability();
                        let log_ref = reference.get_log_block_probability();
                        assert!(
                            (log_prod - log_ref).abs()
                                <= 1e-12 * log_prod.abs().max(log_ref.abs()).max(1.0),
                            "log mismatch after update depth={depth} len={len} mask={mask} step={step} bit={bit} prod={log_prod} ref={log_ref} history={:?}",
                            prod.history
                        );
                    }
                    while prod.history_size() > 0 {
                        let p_prod_0 = prod.predict(false);
                        let p_ref_0 = reference.predict(false);
                        assert!(
                            (p_prod_0 - p_ref_0).abs()
                                <= 1e-12 * p_prod_0.abs().max(p_ref_0.abs()).max(1.0),
                            "revert predict0 mismatch depth={depth} len={len} mask={mask} prod={p_prod_0} ref={p_ref_0} history={:?}",
                            prod.history
                        );
                        let p_prod_1 = prod.predict(true);
                        let p_ref_1 = reference.predict(true);
                        assert!(
                            (p_prod_1 - p_ref_1).abs()
                                <= 1e-12 * p_prod_1.abs().max(p_ref_1.abs()).max(1.0),
                            "revert predict1 mismatch depth={depth} len={len} mask={mask} prod={p_prod_1} ref={p_ref_1} history={:?}",
                            prod.history
                        );
                        prod.revert();
                        reference.revert();
                        let log_prod = prod.get_log_block_probability();
                        let log_ref = reference.get_log_block_probability();
                        assert!(
                            (log_prod - log_ref).abs()
                                <= 1e-12 * log_prod.abs().max(log_ref.abs()).max(1.0),
                            "revert log mismatch depth={depth} len={len} mask={mask} prod={log_prod} ref={log_ref} history={:?}",
                            prod.history
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn context_tree_long_depth_matches_reference_on_short_sequences() {
        for &depth in &[65usize, 80usize] {
            for len in 0..=6usize {
                for mask in 0..(1usize << len) {
                    let mut prod = ContextTree::new(depth);
                    let mut reference = RefContextTree::new(depth);
                    for step in 0..len {
                        let p_prod_0 = prod.predict(false);
                        let p_ref_0 = reference.predict(false);
                        assert!(
                            (p_prod_0 - p_ref_0).abs()
                                <= 1e-12 * p_prod_0.abs().max(p_ref_0.abs()).max(1.0),
                            "long-depth predict0 mismatch depth={depth} len={len} mask={mask} step={step} prod={p_prod_0} ref={p_ref_0} history={:?}",
                            prod.history
                        );
                        let p_prod_1 = prod.predict(true);
                        let p_ref_1 = reference.predict(true);
                        assert!(
                            (p_prod_1 - p_ref_1).abs()
                                <= 1e-12 * p_prod_1.abs().max(p_ref_1.abs()).max(1.0),
                            "long-depth predict1 mismatch depth={depth} len={len} mask={mask} step={step} prod={p_prod_1} ref={p_ref_1} history={:?}",
                            prod.history
                        );
                        assert_close(
                            prod.get_log_block_probability(),
                            reference.get_log_block_probability(),
                        );
                        let bit = ((mask >> step) & 1) == 1;
                        prod.update(bit);
                        reference.update(bit);
                        assert_close(
                            prod.get_log_block_probability(),
                            reference.get_log_block_probability(),
                        );
                    }

                    while prod.history_size() > 0 {
                        assert_close(prod.predict(false), reference.predict(false));
                        assert_close(prod.predict(true), reference.predict(true));
                        prod.revert();
                        reference.revert();
                        assert_close(
                            prod.get_log_block_probability(),
                            reference.get_log_block_probability(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fac_ctw_matches_reference_on_short_sequences() {
        let mut fac = FacContextTree::new(4, 4);
        let mut reference = RefFacContextTree::new(4, 4);
        let stream = [
            (true, 0usize),
            (false, 1usize),
            (true, 2usize),
            (true, 3usize),
            (false, 0usize),
            (false, 1usize),
            (true, 2usize),
            (false, 3usize),
        ];

        for &(bit, idx) in &stream {
            assert_close(fac.predict(false, idx), reference.predict(false, idx));
            assert_close(fac.predict(true, idx), reference.predict(true, idx));
            fac.update(bit, idx);
            reference.update(bit, idx);
            assert_close(
                fac.get_log_block_probability(),
                reference.get_log_block_probability(),
            );
        }

        for &(_, idx) in stream.iter().rev() {
            fac.revert(idx);
            reference.revert(idx);
            assert_close(
                fac.get_log_block_probability(),
                reference.get_log_block_probability(),
            );
        }
    }

    #[test]
    fn fac_ctw_history_consistency() {
        let mut fac = FacContextTree::new(4, 4);

        fac.update_history(&[true, false, true]);
        assert_eq!(fac.shared_history.len(), 3);

        fac.update(true, 0);
        fac.update(false, 1);
        assert_eq!(fac.shared_history.len(), 5);

        fac.revert(1);
        assert_eq!(fac.shared_history.len(), 4);

        fac.revert(0);
        assert_eq!(fac.shared_history.len(), 3);
    }

    #[test]
    fn fac_ctw_predict_one_matches_predict_true() {
        let mut fac = FacContextTree::new(6, 8);
        for &byte in b"predict-one exactness regression payload" {
            for bit_idx in 0..8usize {
                let p_generic = fac.predict(true, bit_idx);
                let p_one = fac.predict_one(bit_idx);
                assert_close(p_generic, p_one);
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                fac.update_predicted(bit, bit_idx);
            }
        }
    }

    #[test]
    fn fac_ctw_long_depth_predict_one_matches_predict_true() {
        let mut fac = FacContextTree::new(78, 4);
        for step in 0..24usize {
            for bit_idx in 0..fac.num_bits() {
                let p_generic = fac.predict(true, bit_idx);
                let p_one = fac.predict_one(bit_idx);
                assert_close(p_generic, p_one);
                let bit = ((step * 5 + bit_idx * 3) & 1) == 1;
                fac.update_predicted(bit, bit_idx);
            }
        }
    }

    #[test]
    fn fac_ctw_update_byte_msb_matches_bit_updates() {
        let mut by_byte = FacContextTree::new(6, 8);
        let mut by_bits = FacContextTree::new(6, 8);
        for &byte in b"byte update msb regression payload" {
            by_byte.update_byte_msb(byte);
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                by_bits.update(bit, bit_idx);
            }
            assert_close(
                by_byte.get_log_block_probability(),
                by_bits.get_log_block_probability(),
            );
            assert_eq!(by_byte.shared_history, by_bits.shared_history);
        }
    }

    #[test]
    fn fac_ctw_update_byte_lsb_matches_bit_updates() {
        let mut by_byte = FacContextTree::new(6, 5);
        let mut by_bits = FacContextTree::new(6, 5);
        for &byte in b"byte update lsb regression payload" {
            by_byte.update_byte_lsb(byte);
            for bit_idx in 0..5usize {
                let bit = ((byte >> bit_idx) & 1) == 1;
                by_bits.update(bit, bit_idx);
            }
            assert_close(
                by_byte.get_log_block_probability(),
                by_bits.get_log_block_probability(),
            );
            assert_eq!(by_byte.shared_history, by_bits.shared_history);
        }
    }

    #[test]
    fn fac_ctw_log_cache_tracks_tree_visits_not_shared_history() {
        let mut fac = FacContextTree::new(8, 8);
        let updates_per_tree = 512usize;
        let (log_int_before, log_half_before) = shared_log_cache_lens();

        for step in 0..updates_per_tree {
            let bit = (step & 1) == 1;
            for bit_idx in 0..8usize {
                fac.update(bit, bit_idx);
            }
        }

        assert_eq!(fac.shared_history.len(), updates_per_tree * 8);
        for tree in &fac.trees {
            let visits = tree.engine.arena.visits(tree.engine.root) as usize;
            assert_eq!(visits, updates_per_tree);
        }

        let (log_int_after, log_half_after) = shared_log_cache_lens();
        let expected_len = updates_per_tree + 1;
        assert!(
            log_int_after <= log_int_before.max(expected_len),
            "log_int grew to {log_int_after} (before={log_int_before}, expected_len={expected_len})"
        );
        assert!(
            log_half_after <= log_half_before.max(expected_len),
            "log_half grew to {log_half_after} (before={log_half_before}, expected_len={expected_len})"
        );
    }

    fn seed_fac_cache_regression_state(fac: &mut FacContextTree) {
        for step in 0..24usize {
            for bit_idx in 0..fac.num_bits() {
                let bit = ((step * 3 + bit_idx) & 1) == 1;
                fac.update(bit, bit_idx);
            }
        }
    }

    fn assert_update_predicted_matches_fresh_after_history_rewrite<F>(mut rewrite: F)
    where
        F: FnMut(&mut FacContextTree),
    {
        let mut predicted = FacContextTree::new(6, 4);
        seed_fac_cache_regression_state(&mut predicted);
        let mut fresh = predicted.clone();
        let original_history = predicted.shared_history.clone();
        let target_bit = 2usize;

        let _ = predicted.predict(true, target_bit);
        rewrite(&mut predicted);
        rewrite(&mut fresh);

        assert_eq!(predicted.shared_history.len(), original_history.len());
        assert_ne!(predicted.shared_history, original_history);
        assert_eq!(predicted.shared_history, fresh.shared_history);

        predicted.update_predicted(false, target_bit);
        fresh.update(false, target_bit);

        assert_eq!(predicted.shared_history, fresh.shared_history);
        assert_close(
            predicted.get_log_block_probability(),
            fresh.get_log_block_probability(),
        );
        for bit_idx in 0..predicted.num_bits() {
            assert_close(
                predicted.predict(false, bit_idx),
                fresh.predict(false, bit_idx),
            );
            assert_close(
                predicted.predict(true, bit_idx),
                fresh.predict(true, bit_idx),
            );
        }
    }

    #[test]
    fn fac_ctw_update_predicted_ignores_stale_cache_after_reset_and_rewrite() {
        assert_update_predicted_matches_fresh_after_history_rewrite(|fac| {
            let mut rewritten = fac.shared_history.clone();
            for bit in &mut rewritten {
                *bit = !*bit;
            }
            fac.reset_history_only();
            fac.update_history(&rewritten);
        });
    }

    #[test]
    fn fac_ctw_update_predicted_ignores_stale_cache_after_revert_and_rewrite() {
        assert_update_predicted_matches_fresh_after_history_rewrite(|fac| {
            let original = fac.shared_history.clone();
            let keep = original.len() / 3;
            let remove = original.len() - keep;
            let mut rewritten_suffix = original[keep..].to_vec();
            for bit in &mut rewritten_suffix {
                *bit = !*bit;
            }
            fac.revert_history(remove);
            fac.update_history(&rewritten_suffix);
        });
    }

    #[test]
    fn fac_ctw_shared_history_version_tracks_mutations() {
        let mut fac = FacContextTree::new(4, 2);
        let mut version = fac.shared_history_version;

        fac.update_history(&[]);
        assert_eq!(fac.shared_history_version, version);

        fac.update_history(&[true, false]);
        assert_ne!(fac.shared_history_version, version);
        version = fac.shared_history_version;

        fac.revert_history(0);
        assert_eq!(fac.shared_history_version, version);

        fac.revert_history(1);
        assert_ne!(fac.shared_history_version, version);
        version = fac.shared_history_version;

        let _ = fac.predict(true, 0);
        assert_eq!(fac.shared_history_version, version);

        fac.update_predicted(true, 0);
        assert_ne!(fac.shared_history_version, version);
        version = fac.shared_history_version;

        fac.reset_history_only();
        assert_ne!(fac.shared_history_version, version);
    }

    #[test]
    fn context_tree_predict_preserves_state() {
        let mut tree = ContextTree::new(6);
        for &bit in &[true, false, true, true, false, false, true, false] {
            tree.update(bit);
        }
        let p0_before = tree.predict(false);
        let p1_before = tree.predict(true);
        let log_before = tree.get_log_block_probability();
        let history_before = tree.history.clone();
        let _ = tree.predict(true);

        assert_eq!(tree.history, history_before);
        assert_close(tree.get_log_block_probability(), log_before);
        assert_close(tree.predict(false), p0_before);
        assert_close(tree.predict(true), p1_before);
    }

    #[test]
    fn context_tree_predict_matches_update_ratio() {
        let mut tree = ContextTree::new(7);
        for &bit in &[true, false, true, false, true, true, false, true, false] {
            tree.update(bit);
        }
        for &sym in &[false, true] {
            let predicted = tree.predict(sym);
            let mut reference = tree.clone();
            let before = reference.get_log_block_probability();
            reference.update(sym);
            let after = reference.get_log_block_probability();
            assert_close(predicted, (after - before).exp());
        }
    }

    #[test]
    fn fac_ctw_predict_preserves_state() {
        let mut fac = FacContextTree::new(5, 8);
        for &byte in b"fac ctw state preservation" {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                fac.update(bit, bit_idx);
            }
        }
        let p0_before = fac.predict(false, 3);
        let p1_before = fac.predict(true, 3);
        let log_before = fac.get_log_block_probability();
        let history_before = fac.shared_history.clone();
        let _ = fac.predict(true, 3);

        assert_eq!(fac.shared_history, history_before);
        assert_close(fac.get_log_block_probability(), log_before);
        assert_close(fac.predict(false, 3), p0_before);
        assert_close(fac.predict(true, 3), p1_before);
    }

    #[test]
    fn fac_ctw_predict_matches_update_ratio() {
        let mut fac = FacContextTree::new(6, 8);
        for &byte in b"fac ctw exact predictive ratio" {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                fac.update(bit, bit_idx);
            }
        }
        for &sym in &[false, true] {
            let predicted = fac.predict(sym, 4);
            let mut reference = fac.clone();
            let before = reference.get_log_block_probability();
            reference.update(sym, 4);
            let after = reference.get_log_block_probability();
            assert_close(predicted, (after - before).exp());
        }
    }

    #[test]
    fn fac_ctw_update_predicted_matches_fresh_update_on_byte_stream() {
        let mut predicted = FacContextTree::new(6, 8);
        let mut fresh = predicted.clone();
        let stream = b"fac-ctw prepared update exactness regression";

        for (byte_pos, &byte) in stream.iter().enumerate() {
            for bit_idx in 0..8usize {
                let bit = ((byte >> (7 - bit_idx)) & 1) == 1;
                let _ = predicted.predict(true, bit_idx);
                predicted.update_predicted(bit, bit_idx);
                fresh.update(bit, bit_idx);

                let predicted_log = predicted.get_log_block_probability();
                let fresh_log = fresh.get_log_block_probability();
                assert!(
                    (predicted_log - fresh_log).abs()
                        <= 1e-12 * predicted_log.abs().max(fresh_log.abs()).max(1.0),
                    "log mismatch byte_pos={byte_pos} bit_idx={bit_idx} bit={bit} predicted={predicted_log} fresh={fresh_log}\nshared_history={:?}\npredicted_arena={:#?}\nfresh_arena={:#?}\npredicted_steps={:?}\nfresh_steps={:?}",
                    predicted.shared_history,
                    predicted.trees[bit_idx].engine.arena,
                    fresh.trees[bit_idx].engine.arena,
                    predicted.trees[bit_idx].engine.prepared_steps,
                    fresh.trees[bit_idx].engine.prepared_steps,
                );
                for probe_idx in 0..8usize {
                    let p_pred_0 = predicted.predict(false, probe_idx);
                    let p_fresh_0 = fresh.predict(false, probe_idx);
                    assert!(
                        (p_pred_0 - p_fresh_0).abs()
                            <= 1e-12 * p_pred_0.abs().max(p_fresh_0.abs()).max(1.0),
                        "predict0 mismatch byte_pos={byte_pos} bit_idx={bit_idx} probe_idx={probe_idx} predicted={p_pred_0} fresh={p_fresh_0}",
                    );
                    let p_pred_1 = predicted.predict(true, probe_idx);
                    let p_fresh_1 = fresh.predict(true, probe_idx);
                    assert!(
                        (p_pred_1 - p_fresh_1).abs()
                            <= 1e-12 * p_pred_1.abs().max(p_fresh_1.abs()).max(1.0),
                        "predict1 mismatch byte_pos={byte_pos} bit_idx={bit_idx} probe_idx={probe_idx} predicted={p_pred_1} fresh={p_fresh_1}",
                    );
                }
            }
        }
    }

    #[test]
    fn fac_ctw_long_depth_update_predicted_matches_fresh_update_on_bit_stream() {
        let mut predicted = FacContextTree::new(78, 4);
        let mut fresh = predicted.clone();

        for step in 0..20usize {
            for bit_idx in 0..predicted.num_bits() {
                let bit = ((step * 7 + bit_idx * 11) & 1) == 1;
                let _ = predicted.predict(true, bit_idx);
                predicted.update_predicted(bit, bit_idx);
                fresh.update(bit, bit_idx);
                assert_eq!(predicted.shared_history, fresh.shared_history);
                assert_close(
                    predicted.get_log_block_probability(),
                    fresh.get_log_block_probability(),
                );
            }
        }

        for bit_idx in 0..predicted.num_bits() {
            assert_close(
                predicted.predict(false, bit_idx),
                fresh.predict(false, bit_idx),
            );
            assert_close(
                predicted.predict(true, bit_idx),
                fresh.predict(true, bit_idx),
            );
        }
    }

    fn scan_symbol_space(tree: &mut FacContextTree, bits: usize) {
        fn rec(tree: &mut FacContextTree, bits: usize, depth: usize) {
            if depth == bits {
                return;
            }
            for bit in [false, true] {
                let bit_idx = depth;
                tree.update(bit, bit_idx);
                rec(tree, bits, depth + 1);
                tree.revert(bit_idx);
            }
        }
        rec(tree, bits, 0);
    }

    fn byte_log_prob(tree: &mut FacContextTree, symbol: u8, msb_first: bool, bits: usize) -> f64 {
        let before = tree.get_log_block_probability();
        if msb_first {
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

    fn assert_symbol_scan_then_update_matches_plain(msb_first: bool) {
        let bits = 8usize;
        let mut with_scan = FacContextTree::new(7, bits);
        let mut plain = with_scan.clone();
        for &byte in b"pdf then update parity payload" {
            for bit_idx in 0..bits {
                let bit = if msb_first {
                    ((byte >> (7 - bit_idx)) & 1) == 1
                } else {
                    ((byte >> bit_idx) & 1) == 1
                };
                with_scan.update(bit, bit_idx);
                plain.update(bit, bit_idx);
            }
        }

        scan_symbol_space(&mut with_scan, bits);

        let observed = b'n';
        for bit_idx in 0..bits {
            let bit = if msb_first {
                ((observed >> (7 - bit_idx)) & 1) == 1
            } else {
                ((observed >> bit_idx) & 1) == 1
            };
            with_scan.update(bit, bit_idx);
            plain.update(bit, bit_idx);
        }

        for sym in 0u8..=255u8 {
            let lp_scan = byte_log_prob(&mut with_scan, sym, msb_first, bits);
            let lp_plain = byte_log_prob(&mut plain, sym, msb_first, bits);
            let diff = (lp_scan - lp_plain).abs();
            assert!(
                diff < 1e-12,
                "symbol={sym} lp_scan={lp_scan} lp_plain={lp_plain} diff={diff}",
            );
        }
    }

    #[test]
    fn fac_ctw_symbol_scan_then_update_matches_plain_msb() {
        assert_symbol_scan_then_update_matches_plain(true);
    }

    #[test]
    fn fac_ctw_symbol_scan_then_update_matches_plain_lsb() {
        assert_symbol_scan_then_update_matches_plain(false);
    }
}
