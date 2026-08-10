//! Bounded-memory hashed context counter predictors.
//!
//! These are deliberately primitive byte predictors: they own only a
//! context-to-byte-count law and expose a truthful next-byte distribution.
//! Composition with match, CTW, PPMD, calibration, or mixtures belongs to the
//! public mixture/spec/runtime layers.

use crate::backends::text_context::{TextContextAnalyzer, bucket_word_len, is_word_byte};
use crate::byte_prefix::normalize_pdf;

use crate::rate_defaults::{CONTEXT_COUNTER_MAX_HASH_BITS, ORDER_NGRAM_MAX_ORDER};
use ahash::AHashMap;
use std::sync::Arc;

const BYTE_SYMBOLS: usize = 256;
const ALPHA: f64 = 0.5;
const RESCALE_TOTAL: u32 = 1 << 20;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Return the maximum resident allocation payload for one hashed counter table.
///
/// The result conservatively includes sparse-map bucket storage and one
/// 256-entry `u16` count table for every possible slot, but excludes allocator
/// and `Arc` metadata. Replacing a collided slot drops its old count table
/// before installing the replacement, so historical collisions do not
/// accumulate resident arrays. Model clones share occupied count tables
/// copy-on-write; mutating a shared context detaches only that context's
/// 512-byte count table.
///
/// This can be deliberately large. With `hash_bits = 24`, the count tables
/// alone require 8 GiB (`2^24 * 256 * size_of::<u16>()`), in addition to the
/// sparse slot map. Construction itself is lazy and does not reserve this
/// maximum; call this when deciding whether a configuration may eventually
/// consume the full logical table under a long or adversarial stream.
///
/// # Errors
///
/// Returns an error when `hash_bits` lies outside the model's accepted range
/// or the target's `usize` cannot represent the requested payload size.
pub fn maximum_resident_bytes(hash_bits: usize) -> Result<usize, String> {
    let slot_count = context_counter_slot_count(hash_bits)?;
    // AHashMap/hashbrown keeps spare buckets, control bytes, and an alignment
    // tail. Four key/value-plus-control payloads per logical entry deliberately
    // over-approximate those implementation details without presenting an
    // unstable allocator layout as an exact API contract.
    let sparse_bucket_bytes = std::mem::size_of::<(usize, ContextSlot)>()
        .checked_add(1)
        .and_then(|bytes| bytes.checked_mul(4))
        .ok_or_else(|| "context-counter sparse bucket size overflow".to_string())?;
    let slot_bytes = slot_count
        .checked_mul(sparse_bucket_bytes)
        .ok_or_else(|| "context-counter sparse slot payload size overflow".to_string())?;
    let count_bytes = slot_count
        .checked_mul(std::mem::size_of::<[u16; BYTE_SYMBOLS]>())
        .ok_or_else(|| "context-counter count payload size overflow".to_string())?;
    slot_bytes
        .checked_add(count_bytes)
        .ok_or_else(|| "context-counter maximum resident payload size overflow".to_string())
}

/// Online hashed order-N byte counter.
#[derive(Clone)]
pub struct OrderNGramModel {
    core: HashedCounterCore,
    order: usize,
    history: [u8; ORDER_NGRAM_MAX_ORDER],
    history_len: usize,
    pdf_cache: PdfCache,
}

/// Text-oriented word-context byte counter.
#[derive(Clone)]
pub struct WordContextModel {
    core: HashedCounterCore,
    analyzer: TextContextAnalyzer,
    current_word_hash: u64,
    current_word_len: u16,
    previous_word_hash: u64,
    pdf_cache: PdfCache,
}

/// A next-byte distribution cache keyed by its semantic probability floor.
///
/// The floor is part of the normalized distribution, not merely a caller-side
/// clamp, so caching only by model history would return incorrect mass after a
/// caller changes floors. `floor_bits = None` is the sole invalid state.
#[derive(Clone)]
struct PdfCache {
    values: [f64; BYTE_SYMBOLS],
    floor_bits: Option<u64>,
}

#[derive(Clone)]
struct HashedCounterCore {
    slots: AHashMap<usize, ContextSlot>,
    slot_mask: usize,
    hash_bits: u32,
    checkpoint_journal: Vec<SlotUndo>,
    checkpoint_depth: usize,
}

#[derive(Clone)]
struct SlotUndo {
    index: usize,
    previous: Option<ContextSlot>,
}

#[derive(Clone)]
struct ContextSlot {
    tag: u64,
    total: u32,
    counts: Arc<[u16; BYTE_SYMBOLS]>,
}

#[derive(Clone)]
#[doc(hidden)]
/// Internal rollback marker for [`OrderNGramModel`].
pub struct OrderNGramCheckpoint {
    journal_len: usize,
    history: [u8; ORDER_NGRAM_MAX_ORDER],
    history_len: usize,
}

#[derive(Clone)]
#[doc(hidden)]
/// Internal rollback marker for [`WordContextModel`].
pub struct WordContextCheckpoint {
    journal_len: usize,
    analyzer: TextContextAnalyzer,
    current_word_hash: u64,
    current_word_len: u16,
    previous_word_hash: u64,
}

#[derive(Clone)]
pub(crate) struct OrderNGramLifecycleSnapshot {
    history: [u8; ORDER_NGRAM_MAX_ORDER],
    history_len: usize,
}

#[derive(Clone)]
pub(crate) struct WordContextLifecycleSnapshot {
    analyzer: TextContextAnalyzer,
    current_word_hash: u64,
    current_word_len: u16,
    previous_word_hash: u64,
}

impl OrderNGramModel {
    /// Construct an order-N byte counter.
    ///
    /// `order` must be in `1..=ORDER_NGRAM_MAX_ORDER`; intended compression
    /// profiles use orders 1-3, while the wider bound leaves room for
    /// controlled experiments. `hash_bits` bounds the lazily populated map to
    /// `2^hash_bits` logical slots, and collisions evict the old slot rather
    /// than growing memory; replacement drops the evicted count table
    /// immediately. Each occupied slot references a copy-on-write 256-entry
    /// `u16` count table
    /// (512 bytes before allocator metadata), so high values can use
    /// substantial memory. Query
    /// [`maximum_resident_bytes`] before construction when the value is not
    /// application-controlled.
    pub fn new(order: usize, hash_bits: usize) -> Result<Self, String> {
        if !(1..=ORDER_NGRAM_MAX_ORDER).contains(&order) {
            return Err(format!(
                "order-ngram order must be in 1..={ORDER_NGRAM_MAX_ORDER}"
            ));
        }
        Ok(Self {
            core: HashedCounterCore::new(hash_bits)?,
            order,
            history: [0; ORDER_NGRAM_MAX_ORDER],
            history_len: 0,
            pdf_cache: PdfCache::new(),
        })
    }

    /// Return the current next-byte PDF.
    pub fn pdf(&mut self, min_prob: f64) -> &[f64] {
        if !self.pdf_cache.is_valid_for(min_prob) {
            let key = self.context_key();
            self.core
                .fill_pdf(key, &mut self.pdf_cache.values, min_prob);
            self.pdf_cache.record_floor(min_prob);
        }
        &self.pdf_cache.values
    }

    /// Fill `out` with the current next-byte PDF.
    pub fn fill_pdf(&mut self, out: &mut [f64; BYTE_SYMBOLS], min_prob: f64) {
        out.copy_from_slice(self.pdf(min_prob));
    }

    /// Log probability of `symbol` under the current context.
    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.pdf(min_prob)[symbol as usize].ln()
    }

    /// Observe `symbol` and train the current context counter.
    pub fn update(&mut self, symbol: u8) {
        if let Some(key) = self.context_key() {
            self.core.observe(key, symbol);
        }
        self.push_history(symbol);
        self.pdf_cache.invalidate();
    }

    /// Advance only the conditioning history without changing counts.
    pub fn update_history_only(&mut self, symbol: u8) {
        self.push_history(symbol);
        self.pdf_cache.invalidate();
    }

    /// Reset transient byte history while preserving learned counts.
    pub fn reset_history(&mut self) {
        self.history = [0; ORDER_NGRAM_MAX_ORDER];
        self.history_len = 0;
        self.pdf_cache.invalidate();
    }

    pub(crate) fn checkpoint(&mut self) -> OrderNGramCheckpoint {
        let journal_len: usize = self.core.begin_checkpoint();
        OrderNGramCheckpoint {
            journal_len,
            history: self.history,
            history_len: self.history_len,
        }
    }

    pub(crate) fn restore_checkpoint(&mut self, checkpoint: &OrderNGramCheckpoint) {
        self.core.restore_checkpoint(checkpoint.journal_len);
        self.history = checkpoint.history;
        self.history_len = checkpoint.history_len;
        self.pdf_cache.invalidate();
    }

    pub(crate) fn discard_checkpoint(&mut self, _checkpoint: OrderNGramCheckpoint) {
        self.core.discard_checkpoint();
    }

    pub(crate) fn clear_checkpoints(&mut self) {
        self.core.clear_checkpoints();
    }

    pub(crate) fn lifecycle_snapshot(&self) -> OrderNGramLifecycleSnapshot {
        OrderNGramLifecycleSnapshot {
            history: self.history,
            history_len: self.history_len,
        }
    }

    pub(crate) fn restore_lifecycle_snapshot(&mut self, snapshot: OrderNGramLifecycleSnapshot) {
        self.history = snapshot.history;
        self.history_len = snapshot.history_len;
        self.pdf_cache.invalidate();
    }

    fn context_key(&self) -> Option<u64> {
        if self.history_len < self.order {
            return None;
        }
        let mut h = FNV_OFFSET ^ (self.order as u64);
        let start = self.history_len - self.order;
        for &byte in &self.history[start..self.history_len] {
            h = hash_byte(h, byte);
        }
        Some(finalize_hash(h))
    }

    fn push_history(&mut self, symbol: u8) {
        if self.history_len < ORDER_NGRAM_MAX_ORDER {
            self.history[self.history_len] = symbol;
            self.history_len += 1;
        } else {
            self.history.copy_within(1..ORDER_NGRAM_MAX_ORDER, 0);
            self.history[ORDER_NGRAM_MAX_ORDER - 1] = symbol;
        }
    }
}

impl WordContextModel {
    /// Construct a word-context byte counter.
    ///
    /// The model hashes previous/current word state and cheap text-structure
    /// features into a lazily populated bounded slot map. Collisions evict old
    /// slots and drop their count arrays immediately. Every occupied slot
    /// references 256 copy-on-write `u16` counters; see
    /// [`maximum_resident_bytes`] for the
    /// configuration-specific upper bound before using a user-controlled
    /// `hash_bits` value.
    pub fn new(hash_bits: usize) -> Result<Self, String> {
        Ok(Self {
            core: HashedCounterCore::new(hash_bits)?,
            analyzer: TextContextAnalyzer::without_repeat_tracking(),
            current_word_hash: 0,
            current_word_len: 0,
            previous_word_hash: 0,
            pdf_cache: PdfCache::new(),
        })
    }

    /// Return the current next-byte PDF.
    pub fn pdf(&mut self, min_prob: f64) -> &[f64] {
        if !self.pdf_cache.is_valid_for(min_prob) {
            let key = self.context_key();
            self.core
                .fill_pdf(Some(key), &mut self.pdf_cache.values, min_prob);
            self.pdf_cache.record_floor(min_prob);
        }
        &self.pdf_cache.values
    }

    /// Fill `out` with the current next-byte PDF.
    pub fn fill_pdf(&mut self, out: &mut [f64; BYTE_SYMBOLS], min_prob: f64) {
        out.copy_from_slice(self.pdf(min_prob));
    }

    /// Log probability of `symbol` under the current context.
    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.pdf(min_prob)[symbol as usize].ln()
    }

    /// Observe `symbol` and train the current word-context counter.
    pub fn update(&mut self, symbol: u8) {
        let key = self.context_key();
        self.core.observe(key, symbol);
        self.update_history(symbol);
        self.pdf_cache.invalidate();
    }

    /// Advance only text conditioning state without changing counts.
    pub fn update_history_only(&mut self, symbol: u8) {
        self.update_history(symbol);
        self.pdf_cache.invalidate();
    }

    /// Reset transient text conditioning state while preserving learned counts.
    pub fn reset_history(&mut self) {
        self.analyzer = TextContextAnalyzer::without_repeat_tracking();
        self.current_word_hash = 0;
        self.current_word_len = 0;
        self.previous_word_hash = 0;
        self.pdf_cache.invalidate();
    }

    pub(crate) fn checkpoint(&mut self) -> WordContextCheckpoint {
        let journal_len: usize = self.core.begin_checkpoint();
        WordContextCheckpoint {
            journal_len,
            analyzer: self.analyzer.clone(),
            current_word_hash: self.current_word_hash,
            current_word_len: self.current_word_len,
            previous_word_hash: self.previous_word_hash,
        }
    }

    pub(crate) fn restore_checkpoint(&mut self, checkpoint: &WordContextCheckpoint) {
        self.core.restore_checkpoint(checkpoint.journal_len);
        self.analyzer = checkpoint.analyzer.clone();
        self.current_word_hash = checkpoint.current_word_hash;
        self.current_word_len = checkpoint.current_word_len;
        self.previous_word_hash = checkpoint.previous_word_hash;
        self.pdf_cache.invalidate();
    }

    pub(crate) fn discard_checkpoint(&mut self, _checkpoint: WordContextCheckpoint) {
        self.core.discard_checkpoint();
    }

    pub(crate) fn clear_checkpoints(&mut self) {
        self.core.clear_checkpoints();
    }

    pub(crate) fn lifecycle_snapshot(&self) -> WordContextLifecycleSnapshot {
        WordContextLifecycleSnapshot {
            analyzer: self.analyzer.clone(),
            current_word_hash: self.current_word_hash,
            current_word_len: self.current_word_len,
            previous_word_hash: self.previous_word_hash,
        }
    }

    pub(crate) fn restore_lifecycle_snapshot(&mut self, snapshot: WordContextLifecycleSnapshot) {
        self.analyzer = snapshot.analyzer;
        self.current_word_hash = snapshot.current_word_hash;
        self.current_word_len = snapshot.current_word_len;
        self.previous_word_hash = snapshot.previous_word_hash;
        self.pdf_cache.invalidate();
    }

    fn context_key(&self) -> u64 {
        let state = self.analyzer.state();
        let mut h = FNV_OFFSET ^ 0x9e37_79b9_7f4a_7c15;
        h = hash_u64(h, self.previous_word_hash);
        h = hash_u64(h, self.current_word_hash);
        h = hash_byte(h, bucket_word_len(self.current_word_len));
        h = hash_byte(h, state.prev1);
        h = hash_byte(h, state.prev2);
        h = hash_byte(h, state.prev1_class);
        h = hash_byte(h, state.prev2_class);
        h = hash_byte(h, u8::from(state.in_word));
        h = hash_byte(h, state.word_len_bucket);
        h = hash_byte(h, state.prev_word_class);
        h = hash_byte(h, state.bracket_bucket);
        h = hash_byte(h, state.quote_flags);
        h = hash_byte(h, u8::from(state.sentence_boundary));
        h = hash_byte(h, u8::from(state.paragraph_break));
        h = hash_byte(h, state.utf8_left);
        finalize_hash(h)
    }

    fn update_history(&mut self, symbol: u8) {
        if is_word_byte(symbol) {
            if self.current_word_len == 0 {
                self.current_word_hash = FNV_OFFSET;
            }
            self.current_word_hash = hash_byte(self.current_word_hash, normalize_word_byte(symbol));
            self.current_word_len = self.current_word_len.saturating_add(1);
        } else if self.current_word_len > 0 {
            self.previous_word_hash = finalize_hash(self.current_word_hash);
            self.current_word_hash = 0;
            self.current_word_len = 0;
        }
        self.analyzer.update(symbol);
    }
}

impl PdfCache {
    fn new() -> Self {
        Self {
            values: [1.0 / (BYTE_SYMBOLS as f64); BYTE_SYMBOLS],
            floor_bits: None,
        }
    }

    #[inline]
    fn is_valid_for(&self, min_prob: f64) -> bool {
        self.floor_bits == Some(min_prob.to_bits())
    }

    #[inline]
    fn record_floor(&mut self, min_prob: f64) {
        self.floor_bits = Some(min_prob.to_bits());
    }

    #[inline]
    fn invalidate(&mut self) {
        self.floor_bits = None;
    }
}

impl HashedCounterCore {
    fn new(hash_bits: usize) -> Result<Self, String> {
        let slot_count: usize = context_counter_slot_count(hash_bits)?;
        Ok(Self {
            slots: AHashMap::new(),
            slot_mask: slot_count - 1,
            hash_bits: hash_bits as u32,
            checkpoint_journal: Vec::new(),
            checkpoint_depth: 0,
        })
    }

    fn fill_pdf(&self, key: Option<u64>, out: &mut [f64; BYTE_SYMBOLS], min_prob: f64) {
        let Some(key) = key else {
            out.fill(1.0 / (BYTE_SYMBOLS as f64));
            return;
        };
        let Some(slot) = self.lookup(key) else {
            out.fill(1.0 / (BYTE_SYMBOLS as f64));
            return;
        };
        let denom = (slot.total as f64) + ALPHA * (BYTE_SYMBOLS as f64);
        if !denom.is_finite() || denom <= 0.0 {
            out.fill(1.0 / (BYTE_SYMBOLS as f64));
            return;
        }
        for (dst, &count) in out.iter_mut().zip(slot.counts.iter()) {
            *dst = (count as f64 + ALPHA) / denom;
        }
        normalize_pdf(out, min_prob);
    }

    fn observe(&mut self, key: u64, symbol: u8) {
        let (idx, tag) = self.index_tag(key);
        if self.checkpoint_depth > 0 {
            self.checkpoint_journal.push(SlotUndo {
                index: idx,
                previous: self.slots.get(&idx).cloned(),
            });
        }
        let slot: &mut ContextSlot = self.slots.entry(idx).or_insert_with(|| ContextSlot {
            tag,
            total: 0,
            counts: Arc::new([0; BYTE_SYMBOLS]),
        });
        if slot.tag != tag {
            *slot = ContextSlot {
                tag,
                total: 0,
                counts: Arc::new([0; BYTE_SYMBOLS]),
            };
        }
        let counts: &mut [u16; BYTE_SYMBOLS] = Arc::make_mut(&mut slot.counts);
        let sym = symbol as usize;
        if slot.total >= RESCALE_TOTAL || counts[sym] == u16::MAX {
            let mut total = 0u32;
            for count in counts.iter_mut() {
                *count = (*count / 2).max(u16::from(*count > 0));
                total += u32::from(*count);
            }
            slot.total = total;
        }
        counts[sym] = counts[sym].saturating_add(1);
        slot.total = slot.total.saturating_add(1);
    }

    fn lookup(&self, key: u64) -> Option<&ContextSlot> {
        let (idx, tag) = self.index_tag(key);
        let slot = self.slots.get(&idx)?;
        (slot.tag == tag).then_some(slot)
    }

    fn index_tag(&self, key: u64) -> (usize, u64) {
        let idx = (key as usize) & self.slot_mask;
        let tag = key >> self.hash_bits;
        (idx, tag)
    }

    fn begin_checkpoint(&mut self) -> usize {
        self.checkpoint_depth = self
            .checkpoint_depth
            .checked_add(1)
            .expect("context-counter checkpoint nesting overflow");
        self.checkpoint_journal.len()
    }

    fn restore_checkpoint(&mut self, journal_len: usize) {
        assert!(
            self.checkpoint_depth > 0,
            "context-counter restore requires an active checkpoint"
        );
        assert!(
            journal_len <= self.checkpoint_journal.len(),
            "context-counter checkpoint marker exceeds its rollback journal"
        );
        while self.checkpoint_journal.len() > journal_len {
            let undo = self
                .checkpoint_journal
                .pop()
                .expect("context-counter checkpoint journal underflow");
            match undo.previous {
                Some(slot) => {
                    self.slots.insert(undo.index, slot);
                }
                None => {
                    self.slots.remove(&undo.index);
                }
            }
        }
    }

    fn discard_checkpoint(&mut self) {
        assert!(
            self.checkpoint_depth > 0,
            "context-counter discard requires an active checkpoint"
        );
        self.checkpoint_depth -= 1;
        if self.checkpoint_depth == 0 {
            self.checkpoint_journal.clear();
        }
    }

    fn clear_checkpoints(&mut self) {
        self.checkpoint_journal.clear();
        self.checkpoint_depth = 0;
    }
}

fn context_counter_slot_count(hash_bits: usize) -> Result<usize, String> {
    if !(1..=CONTEXT_COUNTER_MAX_HASH_BITS).contains(&hash_bits) {
        return Err(format!(
            "context-counter hash_bits must be in 1..={CONTEXT_COUNTER_MAX_HASH_BITS}"
        ));
    }
    1usize
        .checked_shl(hash_bits as u32)
        .ok_or_else(|| "context-counter slot count overflow".to_string())
}

#[inline]
fn normalize_word_byte(byte: u8) -> u8 {
    if byte.is_ascii_uppercase() {
        byte.to_ascii_lowercase()
    } else {
        byte
    }
}

#[inline]
fn hash_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
}

#[inline]
fn hash_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash = hash_byte(hash, byte);
    }
    hash
}

#[inline]
fn finalize_hash(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_ngram_learns_repeated_context() {
        let mut model = OrderNGramModel::new(2, 8).expect("model");
        for &b in b"abababababababab" {
            model.update(b);
        }
        model.update_history_only(b'a');
        let p_b = model.pdf(1e-9)[b'b' as usize];
        let p_x = model.pdf(1e-9)[b'x' as usize];
        assert!(p_b > p_x, "p_b={p_b} p_x={p_x}");
    }

    #[test]
    fn word_context_learns_word_continuation() {
        let mut model = WordContextModel::new(8).expect("model");
        for &b in b"hello hello hello " {
            model.update(b);
        }
        for &b in b"hell" {
            model.update_history_only(b);
        }
        let p_o = model.pdf(1e-9)[b'o' as usize];
        let p_x = model.pdf(1e-9)[b'x' as usize];
        assert!(p_o > p_x, "p_o={p_o} p_x={p_x}");
    }

    #[test]
    fn order_ngram_pdf_cache_is_keyed_by_probability_floor() {
        let mut model = OrderNGramModel::new(1, 8).expect("model");
        for _ in 0..512 {
            model.reset_history();
            model.update_history_only(b'a');
            model.update(b'b');
        }
        model.reset_history();
        model.update_history_only(b'a');

        let low_floor = model.pdf(1e-12).to_owned();
        let high_floor = model.pdf(1e-2).to_owned();
        assert_ne!(
            low_floor, high_floor,
            "changing the floor at fixed history must recompute normalized mass"
        );
        assert!((high_floor.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(model.log_prob(b'x', 1e-2), high_floor[b'x' as usize].ln());
    }

    #[test]
    fn word_context_pdf_cache_is_keyed_by_probability_floor() {
        let mut model = WordContextModel::new(8).expect("model");
        for _ in 0..512 {
            model.reset_history();
            model.update_history_only(b'a');
            model.update(b'b');
        }
        model.reset_history();
        model.update_history_only(b'a');

        let low_floor = model.pdf(1e-12).to_owned();
        let high_floor = model.pdf(1e-2).to_owned();
        assert_ne!(
            low_floor, high_floor,
            "changing the floor at fixed history must recompute normalized mass"
        );
        assert!((high_floor.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(model.log_prob(b'x', 1e-2), high_floor[b'x' as usize].ln());
    }

    #[test]
    fn order_ngram_clone_shares_counts_until_the_cloned_context_is_mutated() {
        let mut original = OrderNGramModel::new(1, 8).expect("model");
        for _ in 0..32 {
            original.reset_history();
            original.update_history_only(b'a');
            original.update(b'b');
        }
        original.reset_history();
        original.update_history_only(b'a');
        let key = original.context_key().expect("complete context");
        let before = original.pdf(1e-12).to_owned();

        let mut cloned = original.clone();
        let original_counts = original
            .core
            .lookup(key)
            .map(|slot| &slot.counts)
            .expect("trained original counts");
        let cloned_counts = cloned
            .core
            .lookup(key)
            .map(|slot| &slot.counts)
            .expect("cloned counts");
        assert!(
            Arc::ptr_eq(original_counts, cloned_counts),
            "model cloning must share immutable count payloads"
        );

        cloned.update(b'b');
        let original_counts = original
            .core
            .lookup(key)
            .map(|slot| &slot.counts)
            .expect("original counts after clone update");
        let cloned_counts = cloned
            .core
            .lookup(key)
            .map(|slot| &slot.counts)
            .expect("detached cloned counts");
        assert!(
            !Arc::ptr_eq(original_counts, cloned_counts),
            "training the clone must detach only its mutated context"
        );
        assert_eq!(original.pdf(1e-12), before);
    }

    #[test]
    fn maximum_resident_bytes_covers_slot_and_count_payloads() {
        let sparse_bucket_bytes = 4 * (std::mem::size_of::<(usize, ContextSlot)>() + 1);
        let per_slot = sparse_bucket_bytes + std::mem::size_of::<[u16; BYTE_SYMBOLS]>();
        assert_eq!(maximum_resident_bytes(1).expect("two slots"), 2 * per_slot);
        assert!(maximum_resident_bytes(0).is_err());
        assert!(maximum_resident_bytes(CONTEXT_COUNTER_MAX_HASH_BITS + 1).is_err());
    }

    #[test]
    fn construction_is_lazy_even_at_the_largest_hash_width() {
        let order = OrderNGramModel::new(1, CONTEXT_COUNTER_MAX_HASH_BITS).expect("order model");
        let word =
            WordContextModel::new(CONTEXT_COUNTER_MAX_HASH_BITS).expect("word-context model");
        assert!(order.core.slots.is_empty());
        assert!(word.core.slots.is_empty());
        assert_eq!(order.core.slots.capacity(), 0);
        assert_eq!(word.core.slots.capacity(), 0);
    }

    #[test]
    fn zero_and_one_tags_remain_distinct() {
        let mut core = HashedCounterCore::new(8).expect("core");
        let low_bits: u64 = 37;
        let zero_tag_key: u64 = low_bits;
        let one_tag_key: u64 = (1u64 << 8) | low_bits;
        let (zero_idx, zero_tag) = core.index_tag(zero_tag_key);
        let (one_idx, one_tag) = core.index_tag(one_tag_key);
        assert_eq!(zero_idx, one_idx, "fixture must target one bounded slot");
        assert_eq!(zero_tag, 0);
        assert_eq!(one_tag, 1);
        assert_ne!(zero_tag, one_tag);

        core.observe(zero_tag_key, b'a');
        assert!(core.lookup(zero_tag_key).is_some());
        core.observe(one_tag_key, b'b');
        assert!(
            core.lookup(zero_tag_key).is_none(),
            "a distinct tag at the same bounded index must evict the old context"
        );
        assert!(core.lookup(one_tag_key).is_some());
    }

    #[test]
    fn order_ngram_checkpoint_restores_counts_and_history_without_cloning_slots() {
        let mut model = OrderNGramModel::new(1, 16).expect("model");
        for _ in 0..32 {
            model.reset_history();
            model.update_history_only(b'a');
            model.update(b'b');
        }
        model.reset_history();
        model.update_history_only(b'a');
        let before = model.pdf(1e-12).to_owned();
        let slot_count_before: usize = model.core.slots.len();

        let checkpoint = model.checkpoint();
        model.update(b'c');
        model.update(b'd');
        assert_ne!(model.pdf(1e-12), before);

        model.restore_checkpoint(&checkpoint);
        assert_eq!(model.pdf(1e-12), before);
        assert_eq!(model.core.slots.len(), slot_count_before);
        model.discard_checkpoint(checkpoint);
        assert_eq!(model.core.checkpoint_depth, 0);
        assert!(model.core.checkpoint_journal.is_empty());
    }

    #[test]
    fn word_context_checkpoint_restores_counts_and_text_state() {
        let mut model = WordContextModel::new(12).expect("model");
        for &symbol in b"hello hello hello " {
            model.update(symbol);
        }
        let before = model.pdf(1e-12).to_owned();
        let checkpoint = model.checkpoint();
        for &symbol in b"different context" {
            model.update(symbol);
        }
        model.restore_checkpoint(&checkpoint);
        assert_eq!(model.pdf(1e-12), before);
        model.discard_checkpoint(checkpoint);
    }
}
