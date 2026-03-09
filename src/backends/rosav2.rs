//! # ROSA: Rapid Online Suffix Automaton
//! A high-performance predictive language model for entropy rate estimation.
//!
//! ROSA uses a **Suffix Automaton** (SAM) to efficiently find the longest matching context
//! for each symbol in a sequence. It then applies **Witten-Bell smoothing** to estimate
//! the conditional probability `P(x_t | x_{<t})`.
//!
//! This allows for accurate estimation of:
//! *   Entropy Rate `Ĥ(X)`
//! *   Cross-Entropy Rate `Ĥ(P, Q)`
//! *   Joint Entropy Rate `Ĥ(X, Y)` (via aligned pair symbols)
//!
//! The implementation is optimized for speed and memory efficiency, using a compact
//! graph representation for the automaton.

#![allow(clippy::needless_range_loop)]

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use density_rs::algorithms::lion::lion::Lion;
use density_rs::codec::codec::Codec;

// Exact predictor memory optimization: keep only a tiny inline transition set and
// spill the rest to the edge list. This balances speed and state RAM.
const SAM_SMALL_MAX: usize = 2;
const SAM_INIT_ST_CAP_SOFT: usize = 4 << 20;
const SAM_INIT_ED_CAP_SOFT: usize = 6 << 20;
const SAM_INIT_TEXT_CAP_SOFT: usize = 4 << 20;
const SAM_INIT_ST_CAP_FAST: usize = 32 << 20;
const SAM_INIT_ED_CAP_FAST: usize = 48 << 20;
const SAM_DENSE_STATE_LIMIT: usize = 1 << 22;
const SAM_DENSE_PROMOTE_DEG: usize = 6;
// NOTE: bump when on-disk format changes.
// v4 adds serialization of `sam.last` and `sam.text_states` (required for reversible conditional updates).
const MAGIC: &[u8] = b"rosa_pb_v4\0";

// This crate is used byte-wise by infotheory; for fast incremental conditional updates we
// support an optional fixed 256-byte alphabet LM build/update path.
const BYTE_ALPHA_N: usize = 256;

#[inline(always)]
fn capped_initial_capacity(
    expected: usize,
    scale: usize,
    bias: usize,
    default_cap: usize,
    soft_cap: usize,
) -> usize {
    if expected == 0 {
        return default_cap;
    }
    expected
        .saturating_mul(scale)
        .saturating_add(bias)
        .min(soft_cap.max(1))
}

#[inline(always)]
fn soft_cap_from_ram(max_ram_bytes: Option<usize>, elem_size: usize, hard_soft_cap: usize) -> usize {
    let Some(limit) = max_ram_bytes else {
        return hard_soft_cap;
    };
    // Keep startup reserves bounded under tight RAM caps.
    let budget = (limit / 16).max(elem_size.saturating_mul(1024));
    let by_ram = budget / elem_size.max(1);
    hard_soft_cap.min(by_ram.max(1024))
}

#[inline(always)]
fn write_u32_slice_le<W: Write>(w: &mut W, xs: &[u32]) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len().saturating_mul(4))
        };
        w.write_all(bytes)
    } else {
        for &x in xs {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
}

#[inline(always)]
fn write_i32_slice_le<W: Write>(w: &mut W, xs: &[i32]) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len().saturating_mul(4))
        };
        w.write_all(bytes)
    } else {
        for &x in xs {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
}

#[inline(always)]
fn write_u64_slice_le<W: Write>(w: &mut W, xs: &[u64]) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len().saturating_mul(8))
        };
        w.write_all(bytes)
    } else {
        for &x in xs {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
}

#[inline(always)]
fn read_u32_slice_le<R: Read>(r: &mut R, xs: &mut [u32]) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut u8, xs.len().saturating_mul(4))
        };
        r.read_exact(bytes)
    } else {
        let mut b4 = [0u8; 4];
        for x in xs {
            r.read_exact(&mut b4)?;
            *x = u32::from_le_bytes(b4);
        }
        Ok(())
    }
}

#[inline(always)]
fn read_i32_slice_le<R: Read>(r: &mut R, xs: &mut [i32]) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut u8, xs.len().saturating_mul(4))
        };
        r.read_exact(bytes)
    } else {
        let mut b4 = [0u8; 4];
        for x in xs {
            r.read_exact(&mut b4)?;
            *x = i32::from_le_bytes(b4);
        }
        Ok(())
    }
}

#[inline(always)]
fn read_u64_slice_le<R: Read>(r: &mut R, xs: &mut [u64]) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut u8, xs.len().saturating_mul(8))
        };
        r.read_exact(bytes)
    } else {
        let mut b8 = [0u8; 8];
        for x in xs {
            r.read_exact(&mut b8)?;
            *x = u64::from_le_bytes(b8);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct SamState {
    link: i32,
    len: i32,
    endpos: i32,
    head: i32,

    small_ch: [u32; SAM_SMALL_MAX],
    small_to: [i32; SAM_SMALL_MAX],
    small_n: u8,
}

#[derive(Clone, Copy, Default)]
struct SamEdge {
    ch: u32,
    to: i32,
    next: i32,
}

#[derive(Clone, Default)]
struct Sam {
    st: Vec<SamState>,
    ed: Vec<SamEdge>,
    last: i32,

    text: Vec<u32>,
    text_states: Vec<i32>,
    boundary_after: Vec<u8>,
}

impl Sam {
    fn new(expected_chars: usize) -> Self {
        let mut s = Sam {
            st: Vec::new(),
            ed: Vec::new(),
            last: 0,
            text: Vec::new(),
            text_states: Vec::new(),
            boundary_after: Vec::new(),
        };

        let st_cap = capped_initial_capacity(expected_chars, 2, 16, 1024, SAM_INIT_ST_CAP_SOFT);
        let ed_cap = capped_initial_capacity(expected_chars, 3, 16, 2048, SAM_INIT_ED_CAP_SOFT);
        let text_cap =
            capped_initial_capacity(expected_chars, 1, 16, 1024, SAM_INIT_TEXT_CAP_SOFT);
        s.st.reserve(st_cap);
        s.ed.reserve(ed_cap);
        s.text.reserve(text_cap);
        s.text_states.reserve(text_cap);
        s.boundary_after.reserve(text_cap);

        let mut root = SamState::default();
        root.link = -1;
        root.len = 0;
        root.endpos = -1;
        root.small_n = 0;
        root.head = -1;
        s.st.push(root);
        s.text_states.push(0); // Root state for empty context
        s
    }

    #[inline(always)]
    fn get_edge(&self, v: i32, ch: u32) -> i32 {
        let st = unsafe { self.st.get_unchecked(v as usize) };
        let sn = st.small_n;
        if sn > 0 && st.small_ch[0] == ch {
            return st.small_to[0];
        }
        if sn > 1 && st.small_ch[1] == ch {
            return st.small_to[1];
        }
        let mut ei = st.head;
        while ei != -1 {
            let e = unsafe { self.ed.get_unchecked(ei as usize) };
            if e.ch == ch {
                return e.to;
            }
            ei = e.next;
        }
        -1
    }

    #[inline(always)]
    fn add_edge(&mut self, v: i32, ch: u32, to: i32) {
        let idx = self.ed.len() as i32;
        let head = self.st[v as usize].head;
        self.ed.push(SamEdge { ch, to, next: head });
        self.st[v as usize].head = idx;
    }

    #[inline(always)]
    fn add_edge_absent(&mut self, v: i32, ch: u32, to: i32) {
        let st = &mut self.st[v as usize];
        if (st.small_n as usize) < SAM_SMALL_MAX {
            let i = st.small_n as usize;
            st.small_n += 1;
            st.small_ch[i] = ch;
            st.small_to[i] = to;
        } else {
            self.add_edge(v, ch, to);
        }
    }

    #[inline(always)]
    fn replace_edge_to(&mut self, v: i32, ch: u32, old_to: i32, new_to: i32) -> bool {
        {
            let st = &mut self.st[v as usize];
            for i in 0..(st.small_n as usize) {
                if st.small_ch[i] == ch && st.small_to[i] == old_to {
                    st.small_to[i] = new_to;
                    return true;
                }
            }
        }
        let mut ei = self.st[v as usize].head;
        while ei != -1 {
            let e = &mut self.ed[ei as usize];
            if e.ch == ch && e.to == old_to {
                e.to = new_to;
                return true;
            }
            ei = e.next;
        }
        false
    }

    fn clone_overflow_edges(&mut self, src: i32, dst: i32) {
        self.st[dst as usize].head = -1;
        let mut ei = self.st[src as usize].head;
        while ei != -1 {
            let e = self.ed[ei as usize];
            self.add_edge(dst, e.ch, e.to);
            ei = e.next;
        }
    }

    fn feed(&mut self, ch: u32) {
        let i = self.text.len() as i32;
        self.text.push(ch);
        self.boundary_after.push(0);

        let g = self.last;
        let r = self.st.len() as i32;
        let mut st_r = SamState::default();
        st_r.link = 0;
        st_r.len = self.st[g as usize].len + 1;
        st_r.endpos = i;
        st_r.small_n = 0;
        st_r.head = -1;
        self.st.push(st_r);

        let mut p = g;
        let mut q;
        while p != -1 {
            q = self.get_edge(p, ch);
            if q != -1 {
                break;
            }
            self.add_edge_absent(p, ch, r);
            p = self.st[p as usize].link;
        }

        if p == -1 {
            self.st[r as usize].link = 0;
        } else {
            q = self.get_edge(p, ch);
            if self.st[p as usize].len + 1 == self.st[q as usize].len {
                self.st[r as usize].link = q;
            } else {
                let u = self.st.len() as i32;
                let mut st_u = self.st[q as usize];
                st_u.len = self.st[p as usize].len + 1;
                self.st.push(st_u);
                self.clone_overflow_edges(q, u);
                while p != -1 && self.replace_edge_to(p, ch, q, u) {
                    p = self.st[p as usize].link;
                }
                self.st[q as usize].link = u;
                self.st[r as usize].link = u;
            }
        }

        self.last = r;
        self.text_states.push(r);

        // Maintain rightmost endpos online (ROSA deterministic predictor).
        let mut v = r;
        while v != -1 && self.st[v as usize].endpos < i {
            self.st[v as usize].endpos = i;
            v = self.st[v as usize].link;
        }
    }

    fn mark_boundary(&mut self) {
        if !self.text.is_empty() {
            let i = self.text.len() - 1;
            self.boundary_after[i] = 1;
        }
        self.last = 0;
    }

    fn finalize_endpos(&mut self) {
        let mut max_len: usize = 0;
        for v in 0..self.st.len() {
            let l = self.st[v].len as usize;
            if l > max_len {
                max_len = l;
            }
        }

        let mut cnt = vec![0usize; max_len + 1];
        for v in 0..self.st.len() {
            cnt[self.st[v].len as usize] += 1;
        }
        let mut pos = vec![0usize; max_len + 1];
        let mut acc = 0usize;
        for l in 0..=max_len {
            pos[l] = acc;
            acc += cnt[l];
        }
        let mut order = vec![0u32; self.st.len()];
        for v in 0..self.st.len() {
            let l = self.st[v].len as usize;
            let idx = pos[l];
            order[idx] = v as u32;
            pos[l] += 1;
        }

        for oi in (0..order.len()).rev() {
            let v = order[oi] as usize;
            let p = self.st[v].link;
            if p >= 0 {
                let p = p as usize;
                if self.st[v].endpos > self.st[p].endpos {
                    self.st[p].endpos = self.st[v].endpos;
                }
            }
        }
    }

    #[inline(always)]
    fn advance(&self, mut v: i32, ch: u32) -> i32 {
        let mut to;
        loop {
            to = self.get_edge(v, ch);
            if to != -1 {
                return to;
            }
            v = self.st[v as usize].link;
            if v == -1 {
                break;
            }
        }
        to = self.get_edge(0, ch);
        if to == -1 { 0 } else { to }
    }

    #[inline(always)]
    fn predict_det(&self, v: i32) -> Option<u32> {
        let mut u = v;
        while u != -1 {
            let st = unsafe { self.st.get_unchecked(u as usize) };
            let i = st.endpos;
            let j = i + 1;
            if st.len > 0 && j >= 0 && (j as usize) < self.text.len() {
                if i >= 0
                    && (i as usize) < self.boundary_after.len()
                    && self.boundary_after[i as usize] != 0
                {
                    u = st.link;
                    continue;
                }
                return Some(self.text[j as usize]);
            }
            u = st.link;
        }
        None
    }

    // ===== Transactional (undo-log) support =====
    fn begin_tx(&self) -> SamTx {
        SamTx {
            old_last: self.last,
            old_text_len: self.text.len(),
            old_text_states_len: self.text_states.len(),
            old_boundary_len: self.boundary_after.len(),
            old_st_len: self.st.len(),
            old_ed_len: self.ed.len(),
            st_changes: Vec::new(),
            ed_changes: Vec::new(),
        }
    }

    fn rollback_tx(&mut self, tx: SamTx) {
        // Restore mutated entries (reverse order is fine even with duplicates).
        for (idx, old) in tx.ed_changes.into_iter().rev() {
            if idx < self.ed.len() {
                self.ed[idx] = old;
            }
        }
        for (idx, old) in tx.st_changes.into_iter().rev() {
            if idx < self.st.len() {
                self.st[idx] = old;
            }
        }

        self.st.truncate(tx.old_st_len);
        self.ed.truncate(tx.old_ed_len);
        self.text.truncate(tx.old_text_len);
        self.text_states.truncate(tx.old_text_states_len);
        self.boundary_after.truncate(tx.old_boundary_len);
        self.last = tx.old_last;
    }

    #[inline(always)]
    fn record_state_change(&self, tx: &mut SamTx, idx: usize) {
        // Duplicates are OK; rollback applies in reverse.
        tx.st_changes.push((idx, self.st[idx]));
    }

    #[inline(always)]
    fn record_edge_change(&self, tx: &mut SamTx, idx: usize) {
        tx.ed_changes.push((idx, self.ed[idx]));
    }

    #[inline(always)]
    fn add_edge_tx(&mut self, tx: &mut SamTx, v: i32, ch: u32, to: i32) {
        let idx = self.ed.len() as i32;
        let head = self.st[v as usize].head;
        self.ed.push(SamEdge { ch, to, next: head });
        self.record_state_change(tx, v as usize);
        self.st[v as usize].head = idx;
    }

    #[inline(always)]
    fn add_edge_absent_tx(&mut self, tx: &mut SamTx, v: i32, ch: u32, to: i32) {
        let v_usize = v as usize;
        let small_n = self.st[v_usize].small_n as usize;
        if small_n < SAM_SMALL_MAX {
            let i = small_n;
            self.record_state_change(tx, v_usize);
            let st = &mut self.st[v_usize];
            st.small_ch[i] = ch;
            st.small_to[i] = to;
            st.small_n += 1;
        } else {
            self.add_edge_tx(tx, v, ch, to);
        }
    }

    #[inline(always)]
    fn replace_edge_to_tx(
        &mut self,
        tx: &mut SamTx,
        v: i32,
        ch: u32,
        old_to: i32,
        new_to: i32,
    ) -> bool {
        // small edges
        {
            let st = &self.st[v as usize];
            for i in 0..(st.small_n as usize) {
                if st.small_ch[i] == ch && st.small_to[i] == old_to {
                    self.record_state_change(tx, v as usize);
                    self.st[v as usize].small_to[i] = new_to;
                    return true;
                }
            }
        }
        // overflow edges
        let mut ei = self.st[v as usize].head;
        while ei != -1 {
            let eidx = ei as usize;
            let e = self.ed[eidx];
            if e.ch == ch && e.to == old_to {
                self.record_edge_change(tx, eidx);
                self.ed[eidx].to = new_to;
                return true;
            }
            ei = e.next;
        }
        false
    }

    fn clone_overflow_edges_tx(&mut self, tx: &mut SamTx, src: i32, dst: i32) {
        self.record_state_change(tx, dst as usize);
        self.st[dst as usize].head = -1;
        let mut ei = self.st[src as usize].head;
        while ei != -1 {
            let e = self.ed[ei as usize];
            self.add_edge_tx(tx, dst, e.ch, e.to);
            ei = e.next;
        }
    }

    fn feed_tx(&mut self, tx: &mut SamTx, ch: u32) {
        let i = self.text.len() as i32;
        self.text.push(ch);
        self.boundary_after.push(0);

        let g = self.last;
        let r = self.st.len() as i32;
        let mut st_r = SamState::default();
        st_r.link = 0;
        st_r.len = self.st[g as usize].len + 1;
        st_r.endpos = i;
        st_r.small_n = 0;
        st_r.head = -1;
        self.st.push(st_r);

        let mut p = g;
        let mut q;
        while p != -1 {
            q = self.get_edge(p, ch);
            if q != -1 {
                break;
            }
            self.add_edge_absent_tx(tx, p, ch, r);
            p = self.st[p as usize].link;
        }

        if p == -1 {
            // link of r is in newly appended state; safe.
            self.st[r as usize].link = 0;
        } else {
            q = self.get_edge(p, ch);
            if self.st[p as usize].len + 1 == self.st[q as usize].len {
                self.st[r as usize].link = q;
            } else {
                let u = self.st.len() as i32;
                let mut st_u = self.st[q as usize];
                st_u.len = self.st[p as usize].len + 1;
                self.st.push(st_u);
                self.clone_overflow_edges_tx(tx, q, u);
                while p != -1 && self.replace_edge_to_tx(tx, p, ch, q, u) {
                    p = self.st[p as usize].link;
                }
                // q is an existing state; record before mutation.
                self.record_state_change(tx, q as usize);
                self.st[q as usize].link = u;
                self.st[r as usize].link = u;
            }
        }

        self.last = r;
        self.text_states.push(r);

        // Maintain rightmost endpos online (ROSA deterministic predictor).
        let mut v = r;
        while v != -1 && self.st[v as usize].endpos < i {
            self.record_state_change(tx, v as usize);
            self.st[v as usize].endpos = i;
            v = self.st[v as usize].link;
        }
    }

    fn mark_boundary_tx(&mut self, tx: &mut SamTx) {
        if !self.text.is_empty() {
            // boundary_after is truncated on rollback, so no need to log.
            let i = self.text.len() - 1;
            self.boundary_after[i] = 1;
        }
        // last is restored on rollback.
        self.last = 0;
        let _ = tx;
    }
}

/// Streaming ROSA exact-match predictor.
///
/// This exposes the deterministic ROSA prediction path only (no smoothing/backoff),
/// which is useful for hybrid coders that want to route only mismatch residuals
/// to another backend model.
#[derive(Clone, Copy, Debug)]
pub struct RosaExactPredictorConfig {
    pub max_ram_bytes: Option<usize>,
    pub chunk_size: usize,
    pub track_boundaries: bool,
}

impl Default for RosaExactPredictorConfig {
    fn default() -> Self {
        Self {
            max_ram_bytes: None,
            chunk_size: 1 << 20,
            track_boundaries: true,
        }
    }
}

fn density_encode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = vec![0u8; Lion::safe_encode_buffer_size(input.len())];
    let size = Lion::encode(input, &mut out).ok()?;
    out.truncate(size);
    Some(out)
}

fn density_decode(input: &[u8], raw_len: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; raw_len];
    let size = Lion::decode(input, &mut out).ok()?;
    if size != raw_len {
        return None;
    }
    Some(out)
}

#[derive(Clone)]
struct CompressedChunk {
    start: usize,
    raw_len: usize,
    compressed: bool,
    data: Vec<u8>,
}

#[derive(Clone)]
struct CompressedPrefix {
    chunks: Vec<CompressedChunk>,
    len: usize,
    stored_bytes: usize,
    cache_idx: usize,
    cache_data: Vec<u8>,
    cache_valid: bool,
}

impl CompressedPrefix {
    fn new(chunk_size: usize) -> Self {
        let _ = chunk_size;
        Self {
            chunks: Vec::new(),
            len: 0,
            stored_bytes: 0,
            cache_idx: 0,
            cache_data: Vec::new(),
            cache_valid: false,
        }
    }

    fn compressed_bytes(&self) -> usize {
        use std::mem::size_of;
        self.stored_bytes
            .saturating_add(self.cache_data.capacity())
            .saturating_add(self.chunks.capacity().saturating_mul(size_of::<CompressedChunk>()))
    }

    fn clear_cache(&mut self) {
        self.cache_valid = false;
        self.cache_data.clear();
    }

    fn push_chunk(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }

        let enc = density_encode(chunk);
        let stored = if let Some(enc) = enc {
            // Keep raw if compression doesn't help.
            if enc.len() + 8 >= chunk.len() {
                CompressedChunk {
                    start: self.len,
                    raw_len: chunk.len(),
                    compressed: false,
                    data: chunk.to_vec(),
                }
            } else {
                CompressedChunk {
                    start: self.len,
                    raw_len: chunk.len(),
                    compressed: true,
                    data: enc,
                }
            }
        } else {
            CompressedChunk {
                start: self.len,
                raw_len: chunk.len(),
                compressed: false,
                data: chunk.to_vec(),
            }
        };

        self.len += stored.raw_len;
        self.stored_bytes = self.stored_bytes.saturating_add(stored.data.capacity());
        self.chunks.push(stored);
        self.cache_valid = false;
    }

    fn get(&mut self, idx: usize) -> u8 {
        debug_assert!(idx < self.len, "compressed prefix index out of range");
        let mut lo = 0usize;
        let mut hi = self.chunks.len();
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if self.chunks[mid].start <= idx {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let chunk_idx = lo;
        let chunk = &self.chunks[chunk_idx];
        let off = idx - chunk.start;
        debug_assert!(off < chunk.raw_len);
        if !chunk.compressed {
            return chunk.data[off];
        }

        if !self.cache_valid || self.cache_idx != chunk_idx {
            let decoded = density_decode(&chunk.data, chunk.raw_len).expect("decode failed");
            self.cache_data = decoded;
            self.cache_idx = chunk_idx;
            self.cache_valid = true;
        }
        self.cache_data[off]
    }

    fn set(&mut self, idx: usize, value: u8) {
        debug_assert!(idx < self.len, "compressed prefix index out of range");
        let mut lo = 0usize;
        let mut hi = self.chunks.len();
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if self.chunks[mid].start <= idx {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let chunk_idx = lo;
        let off = idx - self.chunks[chunk_idx].start;

        if !self.chunks[chunk_idx].compressed {
            self.chunks[chunk_idx].data[off] = value;
            if self.cache_valid && self.cache_idx == chunk_idx {
                self.cache_data[off] = value;
            }
            return;
        }

        let old_cap = self.chunks[chunk_idx].data.capacity();

        let mut raw = density_decode(
            &self.chunks[chunk_idx].data,
            self.chunks[chunk_idx].raw_len,
        )
        .expect("decode failed");
        raw[off] = value;

        if let Some(enc) = density_encode(&raw) {
            if enc.len() + 8 < raw.len() {
                self.chunks[chunk_idx].compressed = true;
                self.chunks[chunk_idx].data = enc;
            } else {
                self.chunks[chunk_idx].compressed = false;
                self.chunks[chunk_idx].data = raw.clone();
            }
        } else {
            self.chunks[chunk_idx].compressed = false;
            self.chunks[chunk_idx].data = raw.clone();
        }
        let new_cap = self.chunks[chunk_idx].data.capacity();
        self.stored_bytes = self
            .stored_bytes
            .saturating_add(new_cap)
            .saturating_sub(old_cap);

        self.cache_valid = true;
        self.cache_idx = chunk_idx;
        self.cache_data = raw;
    }
}

#[derive(Clone, Copy, Default)]
struct ExactState {
    link: i32,
    len: i32,
    endpos: i32,
    head: i32,
    small_ch: [u8; SAM_SMALL_MAX],
    small_to: [i32; SAM_SMALL_MAX],
    small_n: u8,
}

#[derive(Clone, Copy, Default)]
struct ExactEdge {
    ch: u8,
    to: i32,
    next: i32,
}

#[derive(Clone)]
pub struct RosaExactPredictor {
    st: Vec<ExactState>,
    ed: Vec<ExactEdge>,
    root_to: [i32; 256],
    dense_top: Vec<i32>,
    dense_deg: Vec<u8>,
    dense_tables: Vec<[i32; 256]>,
    dense_refcnt: Vec<u32>,
    last: i32,
    st_reserve_chunk: usize,
    ed_reserve_chunk: usize,

    hot_text: Vec<u8>,
    hot_boundary: Vec<u8>,
    hot_start: usize,

    cold_len: usize,
    cold_text: CompressedPrefix,
    cold_boundary: CompressedPrefix,
    observed: usize,
    cfg: RosaExactPredictorConfig,
}

impl RosaExactPredictor {
    /// Create a new predictor.
    ///
    /// `expected_chars` is used only as a capacity hint.
    pub fn new(expected_chars: usize) -> Self {
        Self::new_with_config(expected_chars, RosaExactPredictorConfig::default())
    }

    pub fn new_with_config(expected_chars: usize, cfg: RosaExactPredictorConfig) -> Self {
        let st_soft_cap = if cfg.max_ram_bytes.is_none() {
            SAM_INIT_ST_CAP_FAST
        } else {
            soft_cap_from_ram(
                cfg.max_ram_bytes,
                std::mem::size_of::<ExactState>(),
                SAM_INIT_ST_CAP_SOFT,
            )
        };
        let ed_soft_cap = if cfg.max_ram_bytes.is_none() {
            SAM_INIT_ED_CAP_FAST
        } else {
            soft_cap_from_ram(
                cfg.max_ram_bytes,
                std::mem::size_of::<ExactEdge>(),
                SAM_INIT_ED_CAP_SOFT,
            )
        };
        let st_cap = capped_initial_capacity(expected_chars, 2, 16, 1024, st_soft_cap);
        let ed_cap = capped_initial_capacity(expected_chars, 3, 16, 2048, ed_soft_cap);
        let hot_cap = if cfg.max_ram_bytes.is_none() {
            expected_chars.max(1024)
        } else {
            expected_chars.min(cfg.chunk_size.saturating_mul(16)).max(1024)
        };
        let st_reserve_chunk = (st_soft_cap / 16).clamp(4096, 1 << 18);
        let ed_reserve_chunk = (ed_soft_cap / 16).clamp(4096, 1 << 19);

        let mut st = Vec::with_capacity(st_cap);
        let mut root = ExactState::default();
        root.link = -1;
        root.len = 0;
        root.endpos = -1;
        root.head = -1;
        root.small_n = 0;
        st.push(root);

        Self {
            st,
            ed: Vec::with_capacity(ed_cap),
            root_to: [-1; 256],
            dense_top: vec![-1; SAM_DENSE_STATE_LIMIT],
            dense_deg: vec![0; SAM_DENSE_STATE_LIMIT],
            dense_tables: Vec::new(),
            dense_refcnt: Vec::new(),
            last: 0,
            st_reserve_chunk,
            ed_reserve_chunk,
            hot_text: Vec::with_capacity(hot_cap),
            hot_boundary: if cfg.track_boundaries {
                Vec::with_capacity(hot_cap)
            } else {
                Vec::new()
            },
            hot_start: 0,
            cold_len: 0,
            cold_text: CompressedPrefix::new(cfg.chunk_size),
            cold_boundary: CompressedPrefix::new(cfg.chunk_size),
            observed: 0,
            cfg,
        }
    }

    /// Reset the predictor to an empty state.
    pub fn reset(&mut self, expected_chars: usize) {
        *self = Self::new_with_config(expected_chars, self.cfg);
    }

    /// Number of observed bytes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.cold_len + (self.hot_text.len().saturating_sub(self.hot_start))
    }

    /// Returns true if no bytes have been observed.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn estimated_size_bytes(&self) -> usize {
        use std::mem::size_of;
        self.st.capacity() * size_of::<ExactState>()
            + self.ed.capacity() * size_of::<ExactEdge>()
            + size_of::<[i32; 256]>()
            + self.dense_top.capacity() * size_of::<i32>()
            + self.dense_deg.capacity() * size_of::<u8>()
            + self.dense_tables.capacity() * size_of::<[i32; 256]>()
            + self.dense_refcnt.capacity() * size_of::<u32>()
            + self.hot_text.capacity()
            + self.cold_text.compressed_bytes()
            + if self.cfg.track_boundaries {
                self.hot_boundary.capacity() + self.cold_boundary.compressed_bytes()
            } else {
                0
            }
    }

    #[inline(always)]
    fn ensure_state_capacity(&mut self, additional: usize) -> std::io::Result<()> {
        if self.st.len().saturating_add(additional) <= self.st.capacity() {
            return Ok(());
        }
        let mut add = self.st_reserve_chunk.max(additional);
        if let Some(limit) = self.cfg.max_ram_bytes {
            let cur = self.estimated_size_bytes();
            let elem = std::mem::size_of::<ExactState>().max(1);
            let room = limit.saturating_sub(cur) / elem;
            if room < additional {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "ROSA memory limit exceeded: model={} bytes, limit={} bytes",
                        cur, limit
                    ),
                ));
            }
            add = add.min(room.max(additional));
        }
        self.st.reserve_exact(add);
        Ok(())
    }

    #[inline(always)]
    fn ensure_edge_capacity(&mut self, additional: usize) -> std::io::Result<()> {
        if self.ed.len().saturating_add(additional) <= self.ed.capacity() {
            return Ok(());
        }
        let mut add = self.ed_reserve_chunk.max(additional);
        if let Some(limit) = self.cfg.max_ram_bytes {
            let cur = self.estimated_size_bytes();
            let elem = std::mem::size_of::<ExactEdge>().max(1);
            let room = limit.saturating_sub(cur) / elem;
            if room < additional {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "ROSA memory limit exceeded: model={} bytes, limit={} bytes",
                        cur, limit
                    ),
                ));
            }
            add = add.min(room.max(additional));
        }
        self.ed.reserve_exact(add);
        Ok(())
    }

    #[inline(always)]
    fn hot_len(&self) -> usize {
        self.hot_text.len().saturating_sub(self.hot_start)
    }

    #[inline(always)]
    fn text_get(&mut self, idx: usize) -> u8 {
        if idx < self.cold_len {
            self.cold_text.get(idx)
        } else {
            self.hot_text[self.hot_start + (idx - self.cold_len)]
        }
    }

    #[inline(always)]
    fn boundary_get(&mut self, idx: usize) -> u8 {
        if !self.cfg.track_boundaries {
            return 0;
        }
        if idx < self.cold_len {
            self.cold_boundary.get(idx)
        } else {
            self.hot_boundary[self.hot_start + (idx - self.cold_len)]
        }
    }

    fn reclaim_hot_prefix(&mut self) {
        if self.hot_start == 0 {
            return;
        }
        if self.hot_start >= (self.hot_text.len() / 2).max(self.cfg.chunk_size) {
            self.hot_text.drain(..self.hot_start);
            if self.cfg.track_boundaries {
                self.hot_boundary.drain(..self.hot_start);
            }
            self.hot_start = 0;
        }
    }

    fn maybe_compact(&mut self) -> std::io::Result<()> {
        let Some(limit) = self.cfg.max_ram_bytes else {
            return Ok(());
        };

        while self.estimated_size_bytes() > limit {
            let hot_len = self.hot_len();
            if hot_len <= 1 {
                break;
            }
            let move_len = self.cfg.chunk_size.min(hot_len - 1);
            let start = self.hot_start;
            let end = start + move_len;

            self.cold_text.push_chunk(&self.hot_text[start..end]);
            if self.cfg.track_boundaries {
                self.cold_boundary.push_chunk(&self.hot_boundary[start..end]);
            }
            self.hot_start = end;
            self.cold_len += move_len;
            self.reclaim_hot_prefix();
        }

        if self.estimated_size_bytes() > limit {
            self.cold_text.clear_cache();
            if self.cfg.track_boundaries {
                self.cold_boundary.clear_cache();
            }
        }

        while self.estimated_size_bytes() > limit {
            let hot_len = self.hot_len();
            if hot_len == 0 {
                break;
            }
            let move_len = self.cfg.chunk_size.min(hot_len);
            let start = self.hot_start;
            let end = start + move_len;

            self.cold_text.push_chunk(&self.hot_text[start..end]);
            if self.cfg.track_boundaries {
                self.cold_boundary.push_chunk(&self.hot_boundary[start..end]);
            }
            self.hot_start = end;
            self.cold_len += move_len;
            self.reclaim_hot_prefix();
            self.cold_text.clear_cache();
            if self.cfg.track_boundaries {
                self.cold_boundary.clear_cache();
            }
        }

        if self.estimated_size_bytes() > limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "ROSA memory limit exceeded: model={} bytes, limit={} bytes",
                    self.estimated_size_bytes(),
                    limit
                ),
            ));
        }
        Ok(())
    }

    #[inline(always)]
    fn should_compact_now(&self) -> bool {
        if self.cfg.max_ram_bytes.is_none() {
            return false;
        }
        // Amortize expensive compaction checks while keeping memory bound tight.
        // Every 4096 bytes and also whenever hot tail grows to a chunk.
        (self.observed & 4095) == 0 || self.hot_len() >= self.cfg.chunk_size.saturating_mul(2)
    }

    #[inline(always)]
    fn dense_idx(&self, v: i32) -> i32 {
        if v >= 0 && (v as usize) < SAM_DENSE_STATE_LIMIT {
            unsafe { *self.dense_top.get_unchecked(v as usize) }
        } else {
            -1
        }
    }

    #[inline(always)]
    fn maybe_promote_dense(&mut self, v: i32) {
        if v <= 0 || (v as usize) >= SAM_DENSE_STATE_LIMIT {
            return;
        }
        if self.dense_idx(v) != -1 {
            return;
        }
        let deg = unsafe { *self.dense_deg.get_unchecked(v as usize) } as usize;
        if deg < SAM_DENSE_PROMOTE_DEG {
            return;
        }

        let st = unsafe { self.st.get_unchecked(v as usize) };
        let mut table = [-1i32; 256];
        for i in 0..(st.small_n as usize) {
            table[st.small_ch[i] as usize] = st.small_to[i];
        }
        let mut ei = st.head;
        while ei != -1 {
            let e = unsafe { self.ed.get_unchecked(ei as usize) };
            table[e.ch as usize] = e.to;
            ei = e.next;
        }
        let idx = self.dense_tables.len() as i32;
        self.dense_tables.push(table);
        self.dense_refcnt.push(1);
        unsafe {
            *self.dense_top.get_unchecked_mut(v as usize) = idx;
        }
    }

    #[inline(always)]
    fn dense_ensure_unique(&mut self, v: i32) -> i32 {
        debug_assert!(v > 0 && (v as usize) < SAM_DENSE_STATE_LIMIT);
        let idx = unsafe { *self.dense_top.get_unchecked(v as usize) };
        debug_assert!(idx >= 0);
        let r = unsafe { self.dense_refcnt.get_unchecked_mut(idx as usize) };
        if *r <= 1 {
            return idx;
        }
        *r -= 1;
        let copied = unsafe { *self.dense_tables.get_unchecked(idx as usize) };
        let new_idx = self.dense_tables.len() as i32;
        self.dense_tables.push(copied);
        self.dense_refcnt.push(1);
        unsafe {
            *self.dense_top.get_unchecked_mut(v as usize) = new_idx;
        }
        new_idx
    }

    #[inline(always)]
    fn dense_set_edge(&mut self, v: i32, ch: u8, to: i32) {
        let idx = self.dense_ensure_unique(v);
        unsafe {
            *self
                .dense_tables
                .get_unchecked_mut(idx as usize)
                .get_unchecked_mut(ch as usize) = to;
        }
    }

    #[inline(always)]
    fn bump_dense_deg(&mut self, v: i32) {
        if v <= 0 || (v as usize) >= SAM_DENSE_STATE_LIMIT {
            return;
        }
        let d = unsafe { self.dense_deg.get_unchecked_mut(v as usize) };
        *d = d.saturating_add(1);
    }

    #[inline(always)]
    fn get_edge(&self, v: i32, ch: u8) -> i32 {
        if v == 0 {
            return unsafe { *self.root_to.get_unchecked(ch as usize) };
        }
        let di = self.dense_idx(v);
        if di != -1 {
            return unsafe { *self.dense_tables.get_unchecked(di as usize).get_unchecked(ch as usize) };
        }
        let st = unsafe { self.st.get_unchecked(v as usize) };
        let sn = st.small_n;
        if sn > 0 && st.small_ch[0] == ch {
            return st.small_to[0];
        }
        if sn > 1 && st.small_ch[1] == ch {
            return st.small_to[1];
        }
        let mut ei = st.head;
        while ei != -1 {
            let e = unsafe { self.ed.get_unchecked(ei as usize) };
            if e.ch == ch {
                return e.to;
            }
            ei = e.next;
        }
        -1
    }

    #[inline(always)]
    fn add_edge(&mut self, v: i32, ch: u8, to: i32) -> std::io::Result<()> {
        self.ensure_edge_capacity(1)?;
        let idx = self.ed.len() as i32;
        let head = unsafe { self.st.get_unchecked(v as usize).head };
        self.ed.push(ExactEdge { ch, to, next: head });
        unsafe {
            self.st.get_unchecked_mut(v as usize).head = idx;
        }
        Ok(())
    }

    #[inline(always)]
    fn add_edge_unlimited(&mut self, v: i32, ch: u8, to: i32) {
        let idx = self.ed.len() as i32;
        let head = unsafe { self.st.get_unchecked(v as usize).head };
        self.ed.push(ExactEdge { ch, to, next: head });
        unsafe {
            self.st.get_unchecked_mut(v as usize).head = idx;
        }
    }

    #[inline(always)]
    fn add_edge_absent(&mut self, v: i32, ch: u8, to: i32) -> std::io::Result<()> {
        if v == 0 {
            self.root_to[ch as usize] = to;
            return Ok(());
        }
        let di = self.dense_idx(v);
        if di != -1 {
            self.dense_set_edge(v, ch, to);
            self.bump_dense_deg(v);
            return Ok(());
        }
        let st = unsafe { self.st.get_unchecked_mut(v as usize) };
        if (st.small_n as usize) < SAM_SMALL_MAX {
            let i = st.small_n as usize;
            st.small_n += 1;
            st.small_ch[i] = ch;
            st.small_to[i] = to;
        } else {
            self.add_edge(v, ch, to)?;
        }
        self.bump_dense_deg(v);
        self.maybe_promote_dense(v);
        Ok(())
    }

    #[inline(always)]
    fn add_edge_absent_unlimited(&mut self, v: i32, ch: u8, to: i32) {
        if v == 0 {
            self.root_to[ch as usize] = to;
            return;
        }
        let di = self.dense_idx(v);
        if di != -1 {
            self.dense_set_edge(v, ch, to);
            self.bump_dense_deg(v);
            return;
        }
        let st = unsafe { self.st.get_unchecked_mut(v as usize) };
        if (st.small_n as usize) < SAM_SMALL_MAX {
            let i = st.small_n as usize;
            st.small_n += 1;
            st.small_ch[i] = ch;
            st.small_to[i] = to;
        } else {
            self.add_edge_unlimited(v, ch, to);
        }
        self.bump_dense_deg(v);
        self.maybe_promote_dense(v);
    }

    #[inline(always)]
    fn replace_edge_to(&mut self, v: i32, ch: u8, old_to: i32, new_to: i32) -> bool {
        if v == 0 {
            let slot = unsafe { self.root_to.get_unchecked_mut(ch as usize) };
            if *slot == old_to {
                *slot = new_to;
                return true;
            }
            return false;
        }
        let di = self.dense_idx(v);
        if di != -1 {
            let cur =
                unsafe { *self.dense_tables.get_unchecked(di as usize).get_unchecked(ch as usize) };
            if cur == old_to {
                self.dense_set_edge(v, ch, new_to);
                return true;
            }
            return false;
        }
        let mut replaced = false;
        {
            let st = unsafe { self.st.get_unchecked_mut(v as usize) };
            let sn = st.small_n;
            if sn > 0 && st.small_ch[0] == ch && st.small_to[0] == old_to {
                st.small_to[0] = new_to;
                replaced = true;
            }
            if !replaced && sn > 1 && st.small_ch[1] == ch && st.small_to[1] == old_to {
                st.small_to[1] = new_to;
                replaced = true;
            }
        }
        if !replaced {
            let mut ei = unsafe { self.st.get_unchecked(v as usize).head };
            while ei != -1 {
                let e = unsafe { self.ed.get_unchecked_mut(ei as usize) };
                if e.ch == ch && e.to == old_to {
                    e.to = new_to;
                    replaced = true;
                    break;
                }
                ei = e.next;
            }
        }
        replaced
    }

    #[inline(always)]
    fn clone_overflow_edges(&mut self, src: i32, dst: i32) -> std::io::Result<()> {
        let mut cnt = 0usize;
        let mut ei = unsafe { self.st.get_unchecked(src as usize).head };
        while ei != -1 {
            cnt += 1;
            ei = unsafe { self.ed.get_unchecked(ei as usize).next };
        }
        if cnt == 0 {
            unsafe {
                self.st.get_unchecked_mut(dst as usize).head = -1;
            }
            return Ok(());
        }
        self.ensure_edge_capacity(cnt)?;

        let first_new = self.ed.len() as i32;
        let mut prev_new = -1i32;
        let mut ei = unsafe { self.st.get_unchecked(src as usize).head };
        while ei != -1 {
            let e = unsafe { *self.ed.get_unchecked(ei as usize) };
            let idx = self.ed.len() as i32;
            self.ed.push(ExactEdge {
                ch: e.ch,
                to: e.to,
                next: -1,
            });
            if prev_new != -1 {
                unsafe {
                    self.ed.get_unchecked_mut(prev_new as usize).next = idx;
                }
            }
            prev_new = idx;
            ei = e.next;
        }
        unsafe {
            self.st.get_unchecked_mut(dst as usize).head = first_new;
        }
        Ok(())
    }

    #[inline(always)]
    fn clone_overflow_edges_unlimited(&mut self, src: i32, dst: i32) {
        let mut ei = unsafe { self.st.get_unchecked(src as usize).head };
        if ei == -1 {
            unsafe {
                self.st.get_unchecked_mut(dst as usize).head = -1;
            }
            return;
        }

        let first_new = self.ed.len() as i32;
        let mut prev_new = -1i32;
        while ei != -1 {
            let e = unsafe { *self.ed.get_unchecked(ei as usize) };
            let idx = self.ed.len() as i32;
            self.ed.push(ExactEdge {
                ch: e.ch,
                to: e.to,
                next: -1,
            });
            if prev_new != -1 {
                unsafe {
                    self.ed.get_unchecked_mut(prev_new as usize).next = idx;
                }
            }
            prev_new = idx;
            ei = e.next;
        }
        unsafe {
            self.st.get_unchecked_mut(dst as usize).head = first_new;
        }
    }

    #[inline(always)]
    fn observe_unlimited(&mut self, byte: u8) {
        let ch = byte;
        let i = self.observed as i32;
        self.hot_text.push(byte);
        if self.cfg.track_boundaries {
            self.hot_boundary.push(0);
        }

        let g = self.last;
        let r = self.st.len() as i32;
        let mut st_r = ExactState::default();
        st_r.link = 0;
        st_r.len = unsafe { self.st.get_unchecked(g as usize).len } + 1;
        st_r.endpos = i;
        st_r.small_n = 0;
        st_r.head = -1;
        self.st.push(st_r);

        let mut p = g;
        let mut q = -1;
        while p != -1 {
            q = self.get_edge(p, ch);
            if q != -1 {
                break;
            }
            self.add_edge_absent_unlimited(p, ch, r);
            p = unsafe { self.st.get_unchecked(p as usize).link };
        }

        if p == -1 {
            unsafe {
                self.st.get_unchecked_mut(r as usize).link = 0;
            }
        } else {
            if unsafe { self.st.get_unchecked(p as usize).len } + 1
                == unsafe { self.st.get_unchecked(q as usize).len }
            {
                unsafe {
                    self.st.get_unchecked_mut(r as usize).link = q;
                }
            } else {
                let u = self.st.len() as i32;
                let mut st_u = unsafe { *self.st.get_unchecked(q as usize) };
                st_u.len = unsafe { self.st.get_unchecked(p as usize).len } + 1;
                self.st.push(st_u);
                let qdi = self.dense_idx(q);
                let use_dense_clone = qdi != -1 && (u as usize) < SAM_DENSE_STATE_LIMIT;
                if (u as usize) < SAM_DENSE_STATE_LIMIT {
                    let dq = if (q as usize) < SAM_DENSE_STATE_LIMIT {
                        unsafe { *self.dense_deg.get_unchecked(q as usize) }
                    } else {
                        0
                    };
                    unsafe {
                        *self.dense_deg.get_unchecked_mut(u as usize) = dq;
                    }
                    if use_dense_clone {
                        let rc = unsafe { self.dense_refcnt.get_unchecked_mut(qdi as usize) };
                        *rc = rc.saturating_add(1);
                        unsafe {
                            *self.dense_top.get_unchecked_mut(u as usize) = qdi;
                            let su = self.st.get_unchecked_mut(u as usize);
                            su.small_n = 0;
                            su.head = -1;
                        }
                    }
                }
                if !use_dense_clone {
                    self.clone_overflow_edges_unlimited(q, u);
                }
                while p != -1 && self.replace_edge_to(p, ch, q, u) {
                    p = unsafe { self.st.get_unchecked(p as usize).link };
                }
                unsafe {
                    self.st.get_unchecked_mut(q as usize).link = u;
                    self.st.get_unchecked_mut(r as usize).link = u;
                }
            }
        }

        self.last = r;
        self.observed = self.observed.saturating_add(1);
    }

    #[inline(always)]
    pub fn is_fast_mode(&self) -> bool {
        self.cfg.max_ram_bytes.is_none()
            && !self.cfg.track_boundaries
            && self.cold_len == 0
            && self.hot_start == 0
    }

    #[inline(always)]
    pub fn predict_next_fast(&self) -> Option<u8> {
        debug_assert!(self.is_fast_mode());
        let mut u = self.last;
        while u != -1 {
            let st = unsafe { self.st.get_unchecked(u as usize) };
            let j = st.endpos + 1;
            if st.len > 0 && j >= 0 {
                let ju = j as usize;
                if ju < self.hot_text.len() {
                    return Some(unsafe { *self.hot_text.get_unchecked(ju) });
                }
            }
            u = st.link;
        }
        None
    }

    #[inline(always)]
    pub fn observe_fast(&mut self, byte: u8) {
        debug_assert!(self.is_fast_mode());
        let ch = byte;
        let i = self.observed as i32;
        self.hot_text.push(byte);

        let g = self.last;
        let r = self.st.len() as i32;
        let mut st_r = ExactState::default();
        st_r.link = 0;
        st_r.len = unsafe { self.st.get_unchecked(g as usize).len } + 1;
        st_r.endpos = i;
        st_r.small_n = 0;
        st_r.head = -1;
        self.st.push(st_r);

        let mut p = g;
        let mut q = -1;
        while p != -1 {
            q = self.get_edge(p, ch);
            if q != -1 {
                break;
            }
            self.add_edge_absent_unlimited(p, ch, r);
            p = unsafe { self.st.get_unchecked(p as usize).link };
        }

        if p == -1 {
            unsafe {
                self.st.get_unchecked_mut(r as usize).link = 0;
            }
        } else if unsafe { self.st.get_unchecked(p as usize).len } + 1
            == unsafe { self.st.get_unchecked(q as usize).len }
        {
            unsafe {
                self.st.get_unchecked_mut(r as usize).link = q;
            }
        } else {
            let u = self.st.len() as i32;
            let mut st_u = unsafe { *self.st.get_unchecked(q as usize) };
            st_u.len = unsafe { self.st.get_unchecked(p as usize).len } + 1;
            self.st.push(st_u);
            let qdi = self.dense_idx(q);
            let use_dense_clone = qdi != -1 && (u as usize) < SAM_DENSE_STATE_LIMIT;
            if (u as usize) < SAM_DENSE_STATE_LIMIT {
                let dq = if (q as usize) < SAM_DENSE_STATE_LIMIT {
                    unsafe { *self.dense_deg.get_unchecked(q as usize) }
                } else {
                    0
                };
                unsafe {
                    *self.dense_deg.get_unchecked_mut(u as usize) = dq;
                }
                if use_dense_clone {
                    let rc = unsafe { self.dense_refcnt.get_unchecked_mut(qdi as usize) };
                    *rc = rc.saturating_add(1);
                    unsafe {
                        *self.dense_top.get_unchecked_mut(u as usize) = qdi;
                        let su = self.st.get_unchecked_mut(u as usize);
                        su.small_n = 0;
                        su.head = -1;
                    }
                }
            }
            if !use_dense_clone {
                self.clone_overflow_edges_unlimited(q, u);
            }
            while p != -1 && self.replace_edge_to(p, ch, q, u) {
                p = unsafe { self.st.get_unchecked(p as usize).link };
            }
            unsafe {
                self.st.get_unchecked_mut(q as usize).link = u;
                self.st.get_unchecked_mut(r as usize).link = u;
            }
        }

        self.last = r;
        self.observed = self.observed.saturating_add(1);
    }

    /// Deterministically predict the next byte from the current ROSA state.
    ///
    /// Returns `None` when no exact continuation is available.
    #[inline(always)]
    pub fn predict_next(&mut self) -> Option<u8> {
        let mut u = self.last;
        while u != -1 {
            let (st_len, i, link) = {
                let st = unsafe { self.st.get_unchecked(u as usize) };
                (st.len, st.endpos, st.link)
            };
            let j = i + 1;
            if st_len > 0 && j >= 0 && (j as usize) < self.len() {
                if i >= 0 && self.boundary_get(i as usize) != 0 {
                    u = link;
                    continue;
                }
                return Some(self.text_get(j as usize));
            }
            u = link;
        }
        None
    }

    /// Fill `out` with unique deterministic candidate continuations from the
    /// current suffix-link chain (best match first).
    ///
    /// Returns the number of written candidates.
    #[inline]
    pub fn fill_candidates(&mut self, out: &mut [u8]) -> usize {
        if out.is_empty() {
            return 0;
        }

        let mut seen = [0u64; 4];
        let mut n = 0usize;
        let mut u = self.last;
        let mut hops = 0usize;
        const MAX_HOPS: usize = 128;

        while u != -1 && n < out.len() && hops < MAX_HOPS {
            let (st_len, i, link) = {
                let st = unsafe { self.st.get_unchecked(u as usize) };
                (st.len, st.endpos, st.link)
            };
            let j = i + 1;
            if st_len > 0 && j >= 0 && (j as usize) < self.len() {
                if i >= 0 && self.boundary_get(i as usize) != 0 {
                    u = link;
                    continue;
                }
                let ch = self.text_get(j as usize);
                let c = ch as usize;
                let w = c >> 6;
                let b = 1u64 << (c & 63);
                if (seen[w] & b) == 0 {
                    seen[w] |= b;
                    out[n] = ch;
                    n += 1;
                }
            }
            u = link;
            hops += 1;
        }

        n
    }

    /// Observe one byte and update the online suffix automaton.
    pub fn observe(&mut self, byte: u8) -> std::io::Result<()> {
        if self.cfg.max_ram_bytes.is_none() {
            self.observe_unlimited(byte);
            return Ok(());
        }

        let ch = byte;
        let i = self.observed as i32;
        self.hot_text.push(byte);
        if self.cfg.track_boundaries {
            self.hot_boundary.push(0);
        }

        let g = self.last;
        let r = self.st.len() as i32;
        let mut st_r = ExactState::default();
        st_r.link = 0;
        st_r.len = unsafe { self.st.get_unchecked(g as usize).len } + 1;
        st_r.endpos = i;
        st_r.small_n = 0;
        st_r.head = -1;
        self.ensure_state_capacity(1)?;
        self.st.push(st_r);

        let mut p = g;
        let mut q = -1;
        while p != -1 {
            q = self.get_edge(p, ch);
            if q != -1 {
                break;
            }
            self.add_edge_absent(p, ch, r)?;
            p = unsafe { self.st.get_unchecked(p as usize).link };
        }

        if p == -1 {
            unsafe {
                self.st.get_unchecked_mut(r as usize).link = 0;
            }
        } else {
            if unsafe { self.st.get_unchecked(p as usize).len } + 1
                == unsafe { self.st.get_unchecked(q as usize).len }
            {
                unsafe {
                    self.st.get_unchecked_mut(r as usize).link = q;
                }
            } else {
                let u = self.st.len() as i32;
                let mut st_u = unsafe { *self.st.get_unchecked(q as usize) };
                st_u.len = unsafe { self.st.get_unchecked(p as usize).len } + 1;
                self.ensure_state_capacity(1)?;
                self.st.push(st_u);
                let qdi = self.dense_idx(q);
                let use_dense_clone = qdi != -1 && (u as usize) < SAM_DENSE_STATE_LIMIT;
                if (u as usize) < SAM_DENSE_STATE_LIMIT {
                    let dq = if (q as usize) < SAM_DENSE_STATE_LIMIT {
                        unsafe { *self.dense_deg.get_unchecked(q as usize) }
                    } else {
                        0
                    };
                    unsafe {
                        *self.dense_deg.get_unchecked_mut(u as usize) = dq;
                    }
                    if use_dense_clone {
                        let rc = unsafe { self.dense_refcnt.get_unchecked_mut(qdi as usize) };
                        *rc = rc.saturating_add(1);
                        unsafe {
                            *self.dense_top.get_unchecked_mut(u as usize) = qdi;
                            let su = self.st.get_unchecked_mut(u as usize);
                            su.small_n = 0;
                            su.head = -1;
                        }
                    }
                }
                if !use_dense_clone {
                    self.clone_overflow_edges(q, u)?;
                }
                while p != -1 && self.replace_edge_to(p, ch, q, u) {
                    p = unsafe { self.st.get_unchecked(p as usize).link };
                }
                unsafe {
                    self.st.get_unchecked_mut(q as usize).link = u;
                    self.st.get_unchecked_mut(r as usize).link = u;
                }
            }
        }

        self.last = r;
        self.observed = self.observed.saturating_add(1);
        if self.should_compact_now() {
            self.maybe_compact()?;
        }
        Ok(())
    }

    #[inline(always)]
    pub fn mark_boundary(&mut self) {
        if !self.cfg.track_boundaries {
            self.last = 0;
            return;
        }
        let hot_len = self.hot_len();
        if hot_len > 0 {
            let idx = self.hot_start + hot_len - 1;
            self.hot_boundary[idx] = 1;
        } else if self.cold_len > 0 {
            self.cold_boundary.set(self.cold_len - 1, 1);
        }
        self.last = 0;
    }

    /// Observe a byte slice.
    pub fn observe_slice(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        for &b in bytes {
            self.observe(b)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct SamTx {
    old_last: i32,
    old_text_len: usize,
    old_text_states_len: usize,
    old_boundary_len: usize,
    old_st_len: usize,
    old_ed_len: usize,
    st_changes: Vec<(usize, SamState)>,
    ed_changes: Vec<(usize, SamEdge)>,
}

#[derive(Clone, Copy, Default)]
struct LmState {
    head: i32,
    total_n: u64,
    types_t: u32,

    last_sym: u32,
    last_node: i32,
}

#[derive(Clone, Copy, Default)]
struct CountNode {
    sym_idx: u32,
    cnt: u64,
    next: i32,
}

#[derive(Clone)]
struct LM {
    alphabet: Vec<u32>,
    unigram: Vec<u64>,
    alpha_n: u32,
    total_uni: u64,

    has_byte_map: bool,
    byte_map: [i16; 256],

    ls: Vec<LmState>,
    nodes: Vec<CountNode>,
}

impl Default for LM {
    fn default() -> Self {
        LM {
            alphabet: Vec::new(),
            unigram: Vec::new(),
            alpha_n: 0,
            total_uni: 0,
            has_byte_map: false,
            byte_map: [-1; 256],
            ls: Vec::new(),
            nodes: Vec::new(),
        }
    }
}

impl LM {
    fn build_alphabet(&mut self, sam: &Sam) {
        self.has_byte_map = false;
        self.byte_map = [-1; 256];

        let mut max_cp = 0u32;
        for &v in &sam.text {
            if v > max_cp {
                max_cp = v;
            }
        }

        if max_cp < 256 {
            let mut counts = [0u64; 256];
            for &v in &sam.text {
                counts[v as usize] += 1;
            }
            let mut uniq = 0usize;
            for c in 0..256 {
                if counts[c] != 0 {
                    uniq += 1;
                }
            }

            if uniq == 0 {
                self.alphabet = vec![b'\n' as u32];
                self.unigram = vec![1];
                self.alpha_n = 1;
                self.total_uni = 1;
                self.has_byte_map = true;
                self.byte_map[b'\n' as usize] = 0;
                return;
            }

            self.alphabet = Vec::with_capacity(uniq);
            self.unigram = Vec::with_capacity(uniq);
            self.total_uni = 0;
            for c in 0..256u32 {
                let cnt = counts[c as usize];
                if cnt == 0 {
                    continue;
                }
                self.alphabet.push(c);
                self.unigram.push(cnt);
                self.total_uni += cnt;
            }
            self.alpha_n = self.alphabet.len() as u32;
            self.has_byte_map = true;
            for (i, &c) in self.alphabet.iter().enumerate() {
                self.byte_map[c as usize] = i as i16;
            }
            return;
        }

        let mut tmp = sam.text.clone();
        tmp.sort_unstable();
        tmp.dedup();
        if tmp.is_empty() {
            tmp.push(b'\n' as u32);
        }
        self.alphabet = tmp;
        self.alpha_n = self.alphabet.len() as u32;
        self.unigram = vec![0u64; self.alphabet.len()];
        self.total_uni = 0;
        for &ch in &sam.text {
            if let Ok(i) = self.alphabet.binary_search(&ch) {
                self.unigram[i] += 1;
                self.total_uni += 1;
            }
        }
        if self.total_uni == 0 {
            self.unigram[0] = 1;
            self.total_uni = 1;
        }
    }

    #[inline(always)]
    fn find_sym(&self, ch: u32) -> i32 {
        if self.has_byte_map && ch < 256 {
            return self.byte_map[ch as usize] as i32;
        }
        match self.alphabet.binary_search(&ch) {
            Ok(i) => i as i32,
            Err(_) => -1,
        }
    }

    #[inline(always)]
    fn inc(&mut self, state: u32, sym_idx: u32, add: u64) {
        let ls = &mut self.ls[state as usize];
        let last = ls.last_node;
        if last != -1 && self.nodes[last as usize].sym_idx == sym_idx {
            self.nodes[last as usize].cnt += add;
            ls.total_n += add;
            return;
        }

        let mut ni = ls.head;
        while ni != -1 {
            let node = &mut self.nodes[ni as usize];
            if node.sym_idx == sym_idx {
                node.cnt += add;
                ls.total_n += add;
                ls.last_node = ni;
                ls.last_sym = sym_idx;
                return;
            }
            ni = node.next;
        }

        let idx = self.nodes.len() as i32;
        self.nodes.push(CountNode {
            sym_idx,
            cnt: add,
            next: ls.head,
        });
        ls.head = idx;
        ls.total_n += add;
        ls.types_t += 1;
        ls.last_node = idx;
        ls.last_sym = sym_idx;
    }

    fn build_counts(&mut self, sam: &Sam, max_order: i64) {
        self.ls = vec![
            LmState {
                head: -1,
                last_node: -1,
                ..LmState::default()
            };
            sam.st.len()
        ];
        self.nodes.clear();

        let mut seg_start = 0usize;
        while seg_start < sam.text.len() {
            let mut seg_end = seg_start;
            while seg_end < sam.text.len() {
                let b = sam.boundary_after[seg_end];
                seg_end += 1;
                if b != 0 {
                    break;
                }
            }
            if seg_end - seg_start >= 2 {
                let mut v = 0i32;
                for i in seg_start..(seg_end - 1) {
                    let ch = sam.text[i];
                    v = sam.advance(v, ch);
                    let mut ctx = v;
                    if max_order >= 0 {
                        while ctx != -1 && (sam.st[ctx as usize].len as i64) > max_order {
                            ctx = sam.st[ctx as usize].link;
                        }
                        if ctx == -1 {
                            ctx = 0;
                        }
                    }
                    let nxt = sam.text[i + 1];
                    let si = self.find_sym(nxt);
                    if si >= 0 {
                        self.inc(ctx as u32, si as u32, 1);
                    }
                }
            }
            seg_start = seg_end;
        }

        // propagate up suffix links (counting sort by len)
        let mut max_len: usize = 0;
        for st in &sam.st {
            let l = st.len as usize;
            if l > max_len {
                max_len = l;
            }
        }
        let mut cnt = vec![0usize; max_len + 1];
        for st in &sam.st {
            cnt[st.len as usize] += 1;
        }
        let mut pos = vec![0usize; max_len + 1];
        let mut acc = 0usize;
        for l in 0..=max_len {
            pos[l] = acc;
            acc += cnt[l];
        }
        let mut order = vec![0u32; sam.st.len()];
        for (v, st) in sam.st.iter().enumerate() {
            let l = st.len as usize;
            let idx = pos[l];
            order[idx] = v as u32;
            pos[l] += 1;
        }

        for oi in (0..order.len()).rev() {
            let v = order[oi] as usize;
            let p = sam.st[v].link;
            if p < 0 {
                continue;
            }
            if self.ls[v].total_n == 0 {
                continue;
            }
            let mut ni = self.ls[v].head;
            while ni != -1 {
                let node = self.nodes[ni as usize];
                self.inc(p as u32, node.sym_idx, node.cnt);
                ni = node.next;
            }
        }
    }

    /// Efficient pointwise probability Estimation of a single symbol.
    /// Avoids allocating and writing to a dense distribution array.
    fn prob_for_sym(&self, sam: &Sam, max_order: i64, v: i32, sym_idx: i32) -> f64 {
        if sym_idx < 0 {
            return 1.0 / (self.alpha_n.max(1) as f64);
        }
        let sym_idx = sym_idx as u32;
        let mut p_accum = 0.0f64;
        let mut residual = 1.0f64;
        let mut u = v;

        while u != -1 {
            if !(max_order >= 0 && (sam.st[u as usize].len as i64) > max_order) {
                let n = self.ls[u as usize].total_n;
                let t = self.ls[u as usize].types_t;
                if n > 0 {
                    let lam = if t > 0 {
                        (n as f64) / ((n + (t as u64)) as f64)
                    } else {
                        1.0
                    };

                    // Total probability mass from this state
                    let scale = residual * lam;

                    // Probability of specifically sym_idx in this state
                    let mut count_for_sym = 0u64;
                    let mut ni = self.ls[u as usize].head;
                    while ni != -1 {
                        let node = self.nodes[ni as usize];
                        if node.sym_idx == sym_idx {
                            count_for_sym = node.cnt;
                            break;
                        }
                        ni = node.next;
                    }

                    if count_for_sym > 0 {
                        p_accum += scale * (count_for_sym as f64 / n as f64);
                    }

                    residual *= 1.0 - lam;
                }
            }
            u = sam.st[u as usize].link;
        }

        if self.total_uni > 0 && residual > 0.0 {
            let p_uni = self.unigram[sym_idx as usize] as f64 / self.total_uni as f64;
            p_accum += residual * p_uni;
        } else if residual > 0.0 {
            p_accum += residual * (1.0 / self.alpha_n.max(1) as f64);
        }

        p_accum.clamp(1e-12, 1.0)
    }

    fn probs_for_state(&self, sam: &Sam, max_order: i64, v: i32, out: &mut [f64]) {
        out.fill(0.0);
        let mut residual = 1.0f64;
        let mut u = v;
        while u != -1 {
            if !(max_order >= 0 && (sam.st[u as usize].len as i64) > max_order) {
                let n = self.ls[u as usize].total_n;
                let t = self.ls[u as usize].types_t;
                if n > 0 {
                    let lam = if t > 0 {
                        (n as f64) / ((n + (t as u64)) as f64)
                    } else {
                        1.0
                    };
                    let scale = residual * lam;
                    let inv_n = 1.0 / (n as f64);
                    let mut ni = self.ls[u as usize].head;
                    while ni != -1 {
                        let node = self.nodes[ni as usize];
                        out[node.sym_idx as usize] += scale * ((node.cnt as f64) * inv_n);
                        ni = node.next;
                    }
                    residual *= 1.0 - lam;
                }
            }
            u = sam.st[u as usize].link;
        }

        if self.total_uni > 0 && residual > 0.0 {
            let inv = 1.0 / (self.total_uni as f64);
            for i in 0..(self.alpha_n as usize) {
                out[i] += residual * ((self.unigram[i] as f64) * inv);
            }
        }

        let mut s = 0.0;
        for i in 0..(self.alpha_n as usize) {
            s += out[i];
        }
        if s > 0.0 {
            let invs = 1.0 / s;
            for i in 0..(self.alpha_n as usize) {
                out[i] *= invs;
            }
        } else {
            let uprob = 1.0 / (self.alpha_n.max(1) as f64);
            for i in 0..(self.alpha_n as usize) {
                out[i] = uprob;
            }
        }
    }

    #[inline(always)]
    fn inc_tx(&mut self, tx: &mut LmTx, state: u32, sym_idx: u32, add: u64) {
        let si = state as usize;
        // record old LmState
        tx.ls_changes.push((si, self.ls[si]));

        let ls = &mut self.ls[si];
        let last = ls.last_node;
        if last != -1 && self.nodes[last as usize].sym_idx == sym_idx {
            let ni = last as usize;
            tx.node_changes.push((ni, self.nodes[ni]));
            self.nodes[ni].cnt += add;
            ls.total_n += add;
            return;
        }

        let mut ni = ls.head;
        while ni != -1 {
            let idx = ni as usize;
            if self.nodes[idx].sym_idx == sym_idx {
                tx.node_changes.push((idx, self.nodes[idx]));
                self.nodes[idx].cnt += add;
                ls.total_n += add;
                ls.last_node = ni;
                ls.last_sym = sym_idx;
                return;
            }
            ni = self.nodes[idx].next;
        }

        // New node
        let idx = self.nodes.len() as i32;
        tx.old_nodes_len = tx.old_nodes_len.min(self.nodes.len());
        self.nodes.push(CountNode {
            sym_idx,
            cnt: add,
            next: ls.head,
        });
        ls.head = idx;
        ls.total_n += add;
        ls.types_t += 1;
        ls.last_node = idx;
        ls.last_sym = sym_idx;
    }
}

#[derive(Clone)]
struct LmTx {
    old_ls_len: usize,
    old_nodes_len: usize,
    ls_changes: Vec<(usize, LmState)>,
    node_changes: Vec<(usize, CountNode)>,
    // unigram delta for bytes
    uni_delta: [u64; BYTE_ALPHA_N],
    total_uni_add: u64,
}

#[derive(Clone, Default)]
struct RngStream {
    buf: Vec<u8>,
    pos: usize,
    xs: u64,
}

impl RngStream {
    fn new(seed: u64) -> Self {
        let mut r = RngStream {
            buf: Vec::new(),
            pos: 0,
            xs: 88172645463325252u64,
        };
        if let Ok(path) = std::env::var("ROSAPLUS_RNG_PATH") {
            if !path.is_empty() {
                if let Ok(mut f) = File::open(path) {
                    let mut b = Vec::new();
                    if f.read_to_end(&mut b).is_ok() && b.len() >= 8 {
                        let n = b.len();
                        r.pos = ((seed.wrapping_mul(8)) as usize) % n;
                        r.buf = b;
                    }
                }
            }
        }
        r
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        if self.buf.len() < 8 {
            self.xs ^= self.xs << 7;
            self.xs ^= self.xs >> 9;
            return self.xs;
        }
        let n = self.buf.len();
        let mut b = [0u8; 8];
        for i in 0..8 {
            b[i] = self.buf[self.pos];
            self.pos += 1;
            if self.pos >= n {
                self.pos = 0;
            }
        }
        u64::from_le_bytes(b)
    }

    #[inline(always)]
    fn next_unit(&mut self) -> f64 {
        let x = self.next_u64();
        ((x >> 11) as f64) * (1.0 / 9007199254740992.0)
    }
}

// Helper for debugging/printing byte sequences if needed, but and
// utf8_decode_lossy/utf8_encode are now removed as we follow byte-wise rules.

#[derive(Clone, Default)]
struct SampleScratch {
    idx: Vec<u32>,
    logits: Vec<f64>,
    exps: Vec<f64>,
}

impl SampleScratch {
    fn ensure(&mut self, alpha_n: usize, n: usize) {
        if self.idx.len() != alpha_n {
            self.idx.resize(alpha_n, 0);
        }
        if self.logits.len() < n {
            self.logits.resize(n, 0.0);
            self.exps.resize(n, 0.0);
        }
    }
}

#[derive(Clone)]
pub struct RosaPlus {
    max_order: i64,
    use_eot: bool,
    eot: u32,
    seed: u64,

    sam: Sam,
    lm: LM,
    lm_built: bool,

    rng: RngStream,
    scratch: SampleScratch,
    dist: Vec<f64>,
}

/// A lightweight snapshot of the append-only internal SAM buffers.
///
/// Restoring to a checkpoint is O(1) (via truncation) and is meant to support
/// repeated evaluation of different continuations from the same base training state.
#[derive(Clone, Copy, Debug)]
pub struct RosaCheckpoint {
    sam_st_len: usize,
    sam_ed_len: usize,
    sam_text_len: usize,
    sam_text_states_len: usize,
    sam_boundary_after_len: usize,
    sam_last: i32,
}

/// Transaction object used to roll back a temporary conditional update.
#[derive(Clone)]
pub struct RosaTx {
    sam: SamTx,
    lm: LmTx,
    seg_start: usize,
    seg_len: usize,
}

impl RosaPlus {
    pub fn new(max_order: i64, use_eot: bool, eot_char: u8, seed: u64) -> Self {
        let sam = Sam::new(0);
        RosaPlus {
            max_order,
            use_eot,
            eot: eot_char as u32,
            seed,
            sam,
            lm: LM::default(),
            lm_built: false,
            rng: RngStream::new(seed),
            scratch: SampleScratch::default(),
            dist: Vec::new(),
        }
    }

    pub fn train_example(&mut self, s: &[u8]) {
        if s.is_empty() {
            return;
        }

        if self.sam.text.is_empty() {
            self.sam = Sam::new(s.len());
        }

        for &b in s {
            self.sam.feed(b as u32);
        }

        if self.use_eot {
            self.sam.feed(self.eot);
        }

        self.sam.mark_boundary();
        self.lm_built = false;
    }

    pub fn build_lm(&mut self) {
        self.sam.finalize_endpos();
        self.lm = LM::default();
        self.lm.build_alphabet(&self.sam);
        let mo = if self.max_order < 0 {
            -1
        } else {
            self.max_order
        };
        self.lm.build_counts(&self.sam, mo);
        self.lm_built = true;
        self.dist.resize(self.lm.alpha_n as usize, 0.0);
    }

    /// Deterministically predict the next byte from the current SAM state.
    ///
    /// Returns `None` when no exact continuation is available.
    pub fn predict_next_deterministic(&self) -> Option<u8> {
        let next = self.sam.predict_det(self.sam.last)?;
        if next <= u8::MAX as u32 {
            Some(next as u8)
        } else {
            None
        }
    }

    /// Build the language model without mutating SAM `endpos`.
    ///
    /// This is useful when you want to reuse a trained SAM as a stable base state
    /// (e.g. universal-prior conditioning) and need cheap checkpoint/restore via truncation.
    ///
    /// Note: entropy/cross-entropy estimation does not require `endpos` finalization.
    pub fn build_lm_no_finalize_endpos(&mut self) {
        self.lm = LM::default();
        self.lm.build_alphabet(&self.sam);
        let mo = if self.max_order < 0 {
            -1
        } else {
            self.max_order
        };
        self.lm.build_counts(&self.sam, mo);
        self.lm_built = true;
        self.dist.resize(self.lm.alpha_n as usize, 0.0);
    }

    /// Build an LM with a fixed byte alphabet of size 256.
    ///
    /// This avoids alphabet growth issues and enables fast incremental updates.
    pub fn build_lm_full_bytes_no_finalize_endpos(&mut self) {
        // Fixed alphabet
        self.lm = LM::default();
        self.lm.has_byte_map = true;
        self.lm.alpha_n = BYTE_ALPHA_N as u32;
        self.lm.alphabet = (0..BYTE_ALPHA_N as u32).collect();
        self.lm.byte_map = [-1; 256];
        for i in 0..256 {
            self.lm.byte_map[i] = i as i16;
        }

        // Unigram counts
        let mut counts = [0u64; 256];
        for &v in &self.sam.text {
            if v < 256 {
                counts[v as usize] += 1;
            }
        }
        self.lm.unigram = counts.to_vec();
        self.lm.total_uni = counts.iter().sum();
        if self.lm.total_uni == 0 {
            for i in 0..256 {
                self.lm.unigram[i] = 1;
            }
            self.lm.total_uni = 256;
        }

        // Counts
        let mo = if self.max_order < 0 {
            -1
        } else {
            self.max_order
        };
        self.lm.build_counts(&self.sam, mo);
        self.lm_built = true;
        self.dist.resize(BYTE_ALPHA_N, 0.0);
    }

    /// Begin a reversible conditional update transaction.
    pub fn begin_tx(&mut self) -> RosaTx {
        let sam_tx = self.sam.begin_tx();
        let lm_tx = LmTx {
            old_ls_len: self.lm.ls.len(),
            old_nodes_len: self.lm.nodes.len(),
            ls_changes: Vec::new(),
            node_changes: Vec::new(),
            uni_delta: [0u64; BYTE_ALPHA_N],
            total_uni_add: 0,
        };
        RosaTx {
            sam: sam_tx,
            lm: lm_tx,
            seg_start: self.sam.text.len(),
            seg_len: 0,
        }
    }

    /// Apply a training example and update LM counts incrementally (byte alphabet must be full 256).
    pub fn train_example_tx(&mut self, tx: &mut RosaTx, s: &[u8]) {
        self.train_example_tx_impl(tx, s, true);
    }

    /// Apply a sequential update without inserting a boundary (continuous stream).
    pub fn train_sequence_tx(&mut self, tx: &mut RosaTx, s: &[u8]) {
        self.train_example_tx_impl(tx, s, false);
    }

    /// Apply a sequential update without inserting a boundary (continuous stream).
    ///
    /// This is the non-transactional fast path for streaming updates.
    pub fn train_sequence(&mut self, s: &[u8]) {
        self.train_example_impl_no_tx(s, false);
    }

    fn train_example_tx_impl(&mut self, tx: &mut RosaTx, s: &[u8], mark_boundary: bool) {
        if s.is_empty() {
            return;
        }

        // Ensure LS has entries for current states.
        if self.lm.ls.len() < self.sam.st.len() {
            self.lm.ls.resize(
                self.sam.st.len(),
                LmState {
                    head: -1,
                    last_node: -1,
                    ..LmState::default()
                },
            );
        }

        // Feed all bytes (SAM structure changes are logged).
        for &b in s {
            self.sam.feed_tx(&mut tx.sam, b as u32);
            tx.lm.uni_delta[b as usize] += 1;
            tx.lm.total_uni_add += 1;
        }
        if mark_boundary {
            self.sam.mark_boundary_tx(&mut tx.sam);
        }

        // LM must be built for scoring; we keep it built and update counts incrementally.
        // Extend ls for any new SAM states created by feeding.
        if self.lm.ls.len() < self.sam.st.len() {
            self.lm.ls.resize(
                self.sam.st.len(),
                LmState {
                    head: -1,
                    last_node: -1,
                    ..LmState::default()
                },
            );
        }

        // Update unigram counts (fixed 256 alphabet assumed).
        for i in 0..256 {
            if tx.lm.uni_delta[i] != 0 {
                self.lm.unigram[i] += tx.lm.uni_delta[i];
            }
        }
        self.lm.total_uni += tx.lm.total_uni_add;

        // Update conditional counts for the new segment only.
        let seg_start = tx.seg_start;
        let seg_end = self.sam.text.len();
        tx.seg_len = seg_end - seg_start;
        if tx.seg_len >= 1 {
            let mo = if self.max_order < 0 {
                -1
            } else {
                self.max_order
            };
            // For continuous streams, include the cross-boundary transition from the
            // previous symbol into the first new symbol. For segmented examples,
            // respect boundary markers and skip that transition.
            let mut start_i = seg_start;
            if !mark_boundary
                && seg_start > 0
                && self.sam.boundary_after.get(seg_start - 1).copied().unwrap_or(0) == 0
            {
                start_i = seg_start - 1;
            }
            for i in start_i..(seg_end - 1) {
                // ctx state after consuming sam.text[i] within its segment
                let mut ctx = self.sam.text_states[i + 1];
                if mo >= 0 {
                    while ctx != -1 && (self.sam.st[ctx as usize].len as i64) > mo {
                        ctx = self.sam.st[ctx as usize].link;
                    }
                    if ctx == -1 {
                        ctx = 0;
                    }
                }
                let nxt = self.sam.text[i + 1];
                let si = self.lm.find_sym(nxt);
                if si >= 0 {
                    let mut u = ctx;
                    while u != -1 {
                        self.lm.inc_tx(&mut tx.lm, u as u32, si as u32, 1);
                        u = self.sam.st[u as usize].link;
                    }
                }
            }
        }

        self.lm_built = true;
    }

    fn train_example_impl_no_tx(&mut self, s: &[u8], mark_boundary: bool) {
        if s.is_empty() {
            return;
        }

        if self.sam.text.is_empty() {
            self.sam = Sam::new(s.len());
        }

        let seg_start = self.sam.text.len();

        // Ensure LS has entries for current states.
        if self.lm.ls.len() < self.sam.st.len() {
            self.lm.ls.resize(
                self.sam.st.len(),
                LmState {
                    head: -1,
                    last_node: -1,
                    ..LmState::default()
                },
            );
        }

        for &b in s {
            self.sam.feed(b as u32);
            if self.lm.unigram.len() >= BYTE_ALPHA_N {
                self.lm.unigram[b as usize] += 1;
                self.lm.total_uni += 1;
            }
        }
        if mark_boundary {
            self.sam.mark_boundary();
        }

        if self.lm.ls.len() < self.sam.st.len() {
            self.lm.ls.resize(
                self.sam.st.len(),
                LmState {
                    head: -1,
                    last_node: -1,
                    ..LmState::default()
                },
            );
        }

        let seg_end = self.sam.text.len();
        let seg_len = seg_end - seg_start;
        if seg_len >= 1 {
            let mo = if self.max_order < 0 { -1 } else { self.max_order };
            let mut start_i = seg_start;
            if !mark_boundary
                && seg_start > 0
                && self.sam.boundary_after.get(seg_start - 1).copied().unwrap_or(0) == 0
            {
                start_i = seg_start - 1;
            }
            for i in start_i..(seg_end - 1) {
                let mut ctx = self.sam.text_states[i + 1];
                if mo >= 0 {
                    while ctx != -1 && (self.sam.st[ctx as usize].len as i64) > mo {
                        ctx = self.sam.st[ctx as usize].link;
                    }
                    if ctx == -1 {
                        ctx = 0;
                    }
                }
                let nxt = self.sam.text[i + 1];
                let si = self.lm.find_sym(nxt);
                if si >= 0 {
                    let mut u = ctx;
                    while u != -1 {
                        self.lm.inc(u as u32, si as u32, 1);
                        u = self.sam.st[u as usize].link;
                    }
                }
            }
        }

        self.lm_built = true;
    }

    /// Roll back a transaction, restoring the model to the exact state at begin_tx.
    pub fn rollback_tx(&mut self, tx: RosaTx) {
        // Restore LM changes
        // Unigram rollback
        if self.lm.unigram.len() >= BYTE_ALPHA_N {
            for i in 0..BYTE_ALPHA_N {
                let d = tx.lm.uni_delta[i];
                if d != 0 {
                    self.lm.unigram[i] = self.lm.unigram[i].saturating_sub(d);
                }
            }
            self.lm.total_uni = self.lm.total_uni.saturating_sub(tx.lm.total_uni_add);
        }

        for (idx, old) in tx.lm.node_changes.into_iter().rev() {
            if idx < self.lm.nodes.len() {
                self.lm.nodes[idx] = old;
            }
        }
        for (idx, old) in tx.lm.ls_changes.into_iter().rev() {
            if idx < self.lm.ls.len() {
                self.lm.ls[idx] = old;
            }
        }
        self.lm.nodes.truncate(tx.lm.old_nodes_len);
        self.lm.ls.truncate(tx.lm.old_ls_len);

        // Restore SAM
        self.sam.rollback_tx(tx.sam);
        // lm_built remains true if it was true before; safe to keep true.
    }

    /// Ensure the LM is built (without mutating SAM endpos).
    #[inline(always)]
    pub fn ensure_lm_built_no_finalize_endpos(&mut self) {
        if !self.lm_built {
            self.build_lm_no_finalize_endpos();
        }
    }

    /// Fill `out` with the probability distribution for the next symbol at the current SAM state.
    pub fn probs_for_last_state(&mut self, out: &mut Vec<f64>) {
        if !self.lm_built {
            self.build_lm_full_bytes_no_finalize_endpos();
        }
        let v = self.sam.last;
        let mo = if self.max_order < 0 { -1 } else { self.max_order };
        out.resize(self.lm.alpha_n as usize, 0.0);
        self.lm.probs_for_state(&self.sam, mo, v, out);
    }

    /// Fill `out` in-place with the probability distribution for the next symbol
    /// at the current SAM state. `out` must be at least `lm_alpha_n()` long.
    pub fn probs_for_last_state_inplace(&mut self, out: &mut [f64]) {
        if !self.lm_built {
            self.build_lm_full_bytes_no_finalize_endpos();
        }
        let v = self.sam.last;
        let mo = if self.max_order < 0 { -1 } else { self.max_order };
        let alpha = self.lm.alpha_n as usize;
        debug_assert!(out.len() >= alpha, "out buffer too small");
        self.lm.probs_for_state(&self.sam, mo, v, &mut out[..alpha]);
    }

    fn predictive_entropy_rate_order(data: &[u8], max_order: i64, seed: u64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let num_chunks = 16;
        let chunk_size = (data.len() + num_chunks - 1) / num_chunks;
        let mut total_log_prob = 0.0f64;
        let mut count = 0usize;

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(data.len());
            if start >= end {
                break;
            }
            if i == 0 {
                continue;
            }

            let mut m = RosaPlus::new(max_order, false, 0, seed);
            m.train_example(&data[..start]);
            m.build_lm();
            let mut v = m.sam.last;

            for &b in &data[start..end] {
                let sym_idx = m.lm.find_sym(b as u32);
                let p = m.lm.prob_for_sym(&m.sam, max_order, v, sym_idx);
                total_log_prob += p.log2();
                count += 1;
                v = m.sam.advance(v, b as u32);
            }
        }

        if count == 0 {
            let mut m = RosaPlus::new(max_order, false, 0, seed);
            m.train_example(data);
            m.build_lm();
            m.cross_entropy(data)
        } else {
            -total_log_prob / (count as f64)
        }
    }

    /// Current LM alphabet size (0 if LM not built).
    pub fn lm_alpha_n(&self) -> usize {
        if !self.lm_built {
            0
        } else {
            self.lm.alpha_n as usize
        }
    }

    pub fn estimated_size_bytes(&self) -> usize {
        use std::mem::size_of;

        let mut n = 0usize;

        n = n.saturating_add(self.sam.st.len().saturating_mul(size_of::<SamState>()));
        n = n.saturating_add(self.sam.ed.len().saturating_mul(size_of::<SamEdge>()));
        n = n.saturating_add(self.sam.text.len().saturating_mul(size_of::<u32>()));
        n = n.saturating_add(self.sam.text_states.len().saturating_mul(size_of::<i32>()));
        n = n.saturating_add(
            self.sam
                .boundary_after
                .len()
                .saturating_mul(size_of::<u8>()),
        );

        n = n.saturating_add(self.lm.alphabet.len().saturating_mul(size_of::<u32>()));
        n = n.saturating_add(self.lm.unigram.len().saturating_mul(size_of::<u64>()));
        n = n.saturating_add(self.lm.ls.len().saturating_mul(size_of::<LmState>()));
        n = n.saturating_add(self.lm.nodes.len().saturating_mul(size_of::<CountNode>()));

        n = n.saturating_add(self.dist.len().saturating_mul(size_of::<f64>()));
        n = n.saturating_add(self.scratch.idx.len().saturating_mul(size_of::<u32>()));
        n = n.saturating_add(self.scratch.logits.len().saturating_mul(size_of::<f64>()));
        n = n.saturating_add(self.scratch.exps.len().saturating_mul(size_of::<f64>()));
        n = n.saturating_add(self.rng.buf.len().saturating_mul(size_of::<u8>()));

        n
    }

    pub fn shrink_aux_buffers(&mut self) {
        self.dist.shrink_to_fit();
        self.scratch.idx.shrink_to_fit();
        self.scratch.logits.shrink_to_fit();
        self.scratch.exps.shrink_to_fit();
        self.rng.buf.shrink_to_fit();
    }

    /// Create a new model that shares the same trained SAM state but resets LM-related buffers.
    ///
    /// This is substantially cheaper than cloning the full `RosaPlus` (which includes LM counts,
    /// node tables, and distribution buffers) and is safe for workflows that want to start from
    /// a fixed base training text (e.g. a universal prior) and then add candidate-specific text.
    pub fn fork_from_sam(&self) -> Self {
        Self {
            max_order: self.max_order,
            use_eot: self.use_eot,
            eot: self.eot,
            seed: self.seed,

            sam: self.sam.clone(),
            lm: LM::default(),
            lm_built: false,

            rng: RngStream::new(self.seed),
            scratch: SampleScratch::default(),
            dist: Vec::new(),
        }
    }

    /// A checkpoint that allows restoring the ROSA model back to a previous trained state
    /// by truncating append-only internal buffers.
    ///
    /// Intended for workflows that repeatedly evaluate different continuations from the same base
    /// training text (e.g. universal-prior conditioned scoring).
    pub fn checkpoint(&self) -> RosaCheckpoint {
        RosaCheckpoint {
            sam_st_len: self.sam.st.len(),
            sam_ed_len: self.sam.ed.len(),
            sam_text_len: self.sam.text.len(),
            sam_text_states_len: self.sam.text_states.len(),
            sam_boundary_after_len: self.sam.boundary_after.len(),
            sam_last: self.sam.last,
        }
    }

    /// Restore the model to a previously captured checkpoint.
    ///
    /// This invalidates the LM; callers should rebuild it before scoring.
    pub fn restore(&mut self, ck: &RosaCheckpoint) {
        self.sam.st.truncate(ck.sam_st_len);
        self.sam.ed.truncate(ck.sam_ed_len);
        self.sam.text.truncate(ck.sam_text_len);
        self.sam.text_states.truncate(ck.sam_text_states_len);
        self.sam.boundary_after.truncate(ck.sam_boundary_after_len);
        self.sam.last = ck.sam_last;
        self.lm_built = false;
    }

    #[inline(always)]
    fn sample(&mut self, temperature: f64, top_p: f64, top_k: i32) -> u32 {
        let dist = &self.dist;
        let alpha_n = self.lm.alpha_n as usize;
        self.scratch.ensure(alpha_n, alpha_n);
        for i in 0..alpha_n {
            self.scratch.idx[i] = i as u32;
        }

        // O(n^2) sort by dist desc then idx asc (matches C).
        for i in 0..alpha_n {
            for j in (i + 1)..alpha_n {
                let ii = self.scratch.idx[i] as usize;
                let jj = self.scratch.idx[j] as usize;
                let pi = dist[ii];
                let pj = dist[jj];
                if pj > pi || (pj == pi && jj < ii) {
                    self.scratch.idx.swap(i, j);
                }
            }
        }

        let mut n = alpha_n;
        if top_k > 0 {
            let k = top_k as usize;
            if k < n {
                n = k;
            }
        }

        if top_p > 0.0 && top_p < 1.0 {
            let mut cum = 0.0;
            let mut cut = 0usize;
            for i in 0..n {
                let si = self.scratch.idx[i] as usize;
                cum += dist[si];
                cut += 1;
                if cum >= top_p {
                    break;
                }
            }
            n = if cut > 0 { cut } else { 1 };
        }

        let temperature = if temperature <= 0.0 {
            1e-6
        } else {
            temperature
        };

        self.scratch.ensure(alpha_n, n);
        let mut maxlog = -1e300f64;
        for i in 0..n {
            let si = self.scratch.idx[i] as usize;
            let mut p = dist[si];
            if p < 1e-12 {
                p = 1e-12;
            }
            let z = p.ln() / temperature;
            self.scratch.logits[i] = z;
            if z > maxlog {
                maxlog = z;
            }
        }

        let mut zsum = 0.0;
        for i in 0..n {
            let e = (self.scratch.logits[i] - maxlog).exp();
            self.scratch.exps[i] = e;
            zsum += e;
        }

        let r = self.rng.next_unit() * zsum;
        let mut cum = 0.0;
        let mut pick = 0usize;
        for i in 0..n {
            cum += self.scratch.exps[i];
            if cum > r {
                pick = i;
                break;
            }
        }

        let sym = self.scratch.idx[pick] as usize;
        self.lm.alphabet[sym]
    }

    pub fn generate(&mut self, prompt: &[u8], steps: i32) -> Option<Vec<u8>> {
        if !self.lm_built {
            return None;
        }
        let steps = steps.max(0) as usize;

        let mut v = 0i32;
        for &b in prompt {
            v = self.sam.advance(v, b as u32);
        }

        let mut out: Vec<u32> = Vec::with_capacity(steps);

        for _ in 0..steps {
            let mut ch = self.sam.predict_det(v);
            if ch.is_none() {
                let mo = if self.max_order < 0 {
                    -1
                } else {
                    self.max_order
                };
                self.lm.probs_for_state(&self.sam, mo, v, &mut self.dist);
                ch = Some(self.sample(0.7, 0.9, 0));
            }
            let ch = ch.unwrap();
            out.push(ch);
            if self.use_eot && ch == self.eot {
                break;
            }
            v = self.sam.advance(v, ch);
        }

        Some(out.iter().map(|&c| c as u8).collect())
    }

    // ========== Entropy Estimation API ==========

    /// Returns the probability distribution for the next symbol given a context.
    /// Output: Vec of (codepoint, probability) pairs, sorted by codepoint.
    /// Builds the LM if not already built.
    pub fn get_distribution(&mut self, context: &[u8]) -> Vec<(u32, f64)> {
        if !self.lm_built {
            self.build_lm();
        }

        // Advance through context to get SAM state
        let mut v = 0i32;
        for &b in context {
            v = self.sam.advance(v, b as u32);
        }

        // Get probability distribution at this state
        let mo = if self.max_order < 0 {
            -1
        } else {
            self.max_order
        };
        self.dist.resize(self.lm.alpha_n as usize, 0.0);
        self.lm.probs_for_state(&self.sam, mo, v, &mut self.dist);

        // Build output as (codepoint, probability) pairs
        let mut result = Vec::with_capacity(self.lm.alpha_n as usize);
        for i in 0..(self.lm.alpha_n as usize) {
            if self.dist[i] > 0.0 {
                result.push((self.lm.alphabet[i], self.dist[i]));
            }
        }
        result.sort_by_key(|&(cp, _)| cp);
        result
    }

    /// Compute the predictive entropy rate (bits per symbol) of the given data.
    ///
    /// Uses chunked prequential scoring (train on past chunks, score next chunk).
    pub fn predictive_entropy_rate(&mut self, data: &[u8]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        if self.max_order < 0 {
            let candidates: [i64; 8] = [0, 1, 2, 4, 8, 16, 32, 64];
            let mut best = f64::INFINITY;
            for &mo in &candidates {
                if mo as usize >= data.len() {
                    continue;
                }
                let h = Self::predictive_entropy_rate_order(data, mo, self.seed);
                if h < best {
                    best = h;
                }
            }
            if best.is_finite() {
                return best;
            }
        }
        Self::predictive_entropy_rate_order(data, self.max_order, self.seed)
    }

    pub fn entropy_rate_cps(&mut self, cps: &[u32]) -> f64 {
        if cps.len() < 2 {
            return 0.0;
        }

        self.sam = Sam::new(cps.len());
        self.lm_built = false;

        let num_chunks = 16;
        let chunk_size = (cps.len() + num_chunks - 1) / num_chunks;
        let mut total_log_prob = 0.0f64;
        let mut count = 0usize;

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(cps.len());
            if start >= end {
                break;
            }
            let chunk = &cps[start..end];
            if i > 0 {
                // Avoid endpos finalization since we continue mutating the SAM across chunks.
                self.build_lm_no_finalize_endpos();
                let mut v = self.sam.text_states[start];
                for &ch in chunk {
                    let sym_idx = self.lm.find_sym(ch);
                    let p = self.lm.prob_for_sym(&self.sam, self.max_order, v, sym_idx);
                    total_log_prob += p.log2();
                    count += 1;
                    v = self.sam.advance(v, ch);
                }
            }
            for &ch in chunk {
                self.sam.feed(ch);
            }
        }

        if count == 0 {
            self.build_lm();
            self.entropy_rate_plugin_cps(cps)
        } else {
            -total_log_prob / (count as f64)
        }
    }

    fn entropy_rate_plugin_bytes(&mut self, data: &[u8]) -> f64 {
        let mut v = 0i32;
        let mut total_log_prob = 0.0f64;
        let mut count = 0usize;
        for t in 0..(data.len() - 1) {
            v = self.sam.advance(v, data[t] as u32);
            let next_ch = data[t + 1] as u32;
            let sym_idx = self.lm.find_sym(next_ch);
            let p = self.lm.prob_for_sym(&self.sam, self.max_order, v, sym_idx);
            total_log_prob += p.log2();
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            -total_log_prob / (count as f64)
        }
    }

    pub fn cross_entropy(&self, data: &[u8]) -> f64 {
        if !self.lm_built || data.is_empty() {
            return 0.0;
        }
        let mut total_log_prob = 0.0f64;
        let mut v = 0i32;
        for &b in data {
            let ch = b as u32;
            let sym_idx = self.lm.find_sym(ch);
            let p = self.lm.prob_for_sym(&self.sam, self.max_order, v, sym_idx);
            total_log_prob += p.log2();
            v = self.sam.advance(v, ch);
        }
        -total_log_prob / (data.len() as f64)
    }

    pub fn cross_entropy_cps(&self, data: &[u32]) -> f64 {
        if !self.lm_built || data.is_empty() {
            return 0.0;
        }
        let mut total_log_prob = 0.0f64;
        let mut v = 0i32;
        for &ch in data {
            let sym_idx = self.lm.find_sym(ch);
            let p = self.lm.prob_for_sym(&self.sam, self.max_order, v, sym_idx);
            total_log_prob += p.log2();
            v = self.sam.advance(v, ch);
        }
        -total_log_prob / (data.len() as f64)
    }

    fn entropy_rate_plugin_cps(&mut self, cps: &[u32]) -> f64 {
        let mut v = 0i32;
        let mut total_log_prob = 0.0f64;
        let mut count = 0usize;
        for t in 0..(cps.len() - 1) {
            v = self.sam.advance(v, cps[t]);
            let next_ch = cps[t + 1];
            let sym_idx = self.lm.find_sym(next_ch);
            let p = self.lm.prob_for_sym(&self.sam, self.max_order, v, sym_idx);
            total_log_prob += p.log2();
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            -total_log_prob / (count as f64)
        }
    }

    /// Returns the marginal (unigram) distribution over the training data.
    /// Output: Vec of (codepoint, probability) pairs, sorted by codepoint.
    pub fn marginal_distribution(&self) -> Vec<(u32, f64)> {
        if self.lm.total_uni == 0 {
            return Vec::new();
        }

        let inv = 1.0 / (self.lm.total_uni as f64);
        let mut result = Vec::with_capacity(self.lm.alpha_n as usize);
        for i in 0..(self.lm.alpha_n as usize) {
            let p = (self.lm.unigram[i] as f64) * inv;
            if p > 0.0 {
                result.push((self.lm.alphabet[i], p));
            }
        }
        result.sort_by_key(|&(cp, _)| cp);
        result
    }

    /// Compute the marginal entropy H(X) from the unigram distribution.
    /// Returns bits per symbol.
    pub fn marginal_entropy(&self) -> f64 {
        if self.lm.total_uni == 0 {
            return 0.0;
        }

        let inv = 1.0 / (self.lm.total_uni as f64);
        let mut h = 0.0f64;
        for i in 0..(self.lm.alpha_n as usize) {
            let p = (self.lm.unigram[i] as f64) * inv;
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        if !self.lm_built {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "LM not built",
            ));
        }

        // Transactional conditional updates require a valid prefix-state trace.
        // If this invariant is violated, the loaded model would be unusable.
        if self.sam.text_states.len() != self.sam.text.len() + 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "SAM text_states mismatch (expected text.len()+1)",
            ));
        }
        let mut f = BufWriter::with_capacity(1024 * 1024, File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&self.max_order.to_le_bytes())?;
        f.write_all(&(self.use_eot as i32).to_le_bytes())?;
        f.write_all(&self.eot.to_le_bytes())?;
        f.write_all(&self.seed.to_le_bytes())?;

        // SAM
        f.write_all(&(self.sam.st.len() as u32).to_le_bytes())?;
        f.write_all(&(self.sam.ed.len() as u32).to_le_bytes())?;
        f.write_all(&(self.sam.text.len() as u32).to_le_bytes())?;
        for st in &self.sam.st {
            f.write_all(&st.link.to_le_bytes())?;
            f.write_all(&st.len.to_le_bytes())?;
            f.write_all(&st.endpos.to_le_bytes())?;
            f.write_all(&(st.small_n as u32).to_le_bytes())?;
            for k in 0..(st.small_n as usize) {
                f.write_all(&st.small_ch[k].to_le_bytes())?;
                f.write_all(&st.small_to[k].to_le_bytes())?;
            }
            f.write_all(&st.head.to_le_bytes())?;
        }
        for e in &self.sam.ed {
            f.write_all(&e.ch.to_le_bytes())?;
            f.write_all(&e.to.to_le_bytes())?;
            f.write_all(&e.next.to_le_bytes())?;
        }
        write_u32_slice_le(&mut f, &self.sam.text)?;
        f.write_all(&self.sam.boundary_after)?;

        // Persist SAM cursor + prefix trace.
        f.write_all(&self.sam.last.to_le_bytes())?;
        f.write_all(&(self.sam.text_states.len() as u32).to_le_bytes())?;
        write_i32_slice_le(&mut f, &self.sam.text_states)?;

        // LM
        f.write_all(&self.lm.alpha_n.to_le_bytes())?;
        f.write_all(&self.lm.total_uni.to_le_bytes())?;
        f.write_all(&(self.lm.nodes.len() as u32).to_le_bytes())?;
        write_u32_slice_le(&mut f, &self.lm.alphabet)?;
        write_u64_slice_le(&mut f, &self.lm.unigram)?;
        for ls in &self.lm.ls {
            f.write_all(&ls.head.to_le_bytes())?;
            f.write_all(&ls.total_n.to_le_bytes())?;
            f.write_all(&ls.types_t.to_le_bytes())?;
        }
        for n in &self.lm.nodes {
            f.write_all(&n.sym_idx.to_le_bytes())?;
            f.write_all(&n.cnt.to_le_bytes())?;
            f.write_all(&n.next.to_le_bytes())?;
        }
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut f = BufReader::with_capacity(1024 * 1024, File::open(path)?);
        let mut magic = vec![0u8; MAGIC.len()];
        f.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad magic",
            ));
        }

        let mut b8 = [0u8; 8];
        let mut b4 = [0u8; 4];

        f.read_exact(&mut b8)?;
        let max_order = i64::from_le_bytes(b8);
        f.read_exact(&mut b4)?;
        let use_eot = i32::from_le_bytes(b4) != 0;
        f.read_exact(&mut b4)?;
        let eot = u32::from_le_bytes(b4);
        f.read_exact(&mut b8)?;
        let seed = u64::from_le_bytes(b8);

        let mut m = RosaPlus::new(max_order, use_eot, eot as u8, seed);

        // SAM
        f.read_exact(&mut b4)?;
        let st_n = u32::from_le_bytes(b4) as usize;
        f.read_exact(&mut b4)?;
        let ed_n = u32::from_le_bytes(b4) as usize;
        f.read_exact(&mut b4)?;
        let text_n = u32::from_le_bytes(b4) as usize;

        m.sam = Sam::new(text_n);
        m.sam.st.resize(st_n, SamState::default());
        m.sam.ed.resize(ed_n, SamEdge::default());
        m.sam.text.resize(text_n, 0u32);
        m.sam.boundary_after.resize(text_n, 0u8);

        for i in 0..st_n {
            f.read_exact(&mut b4)?;
            m.sam.st[i].link = i32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.sam.st[i].len = i32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.sam.st[i].endpos = i32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            let sn = u32::from_le_bytes(b4) as usize;
            if sn > SAM_SMALL_MAX {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad small_n",
                ));
            }
            m.sam.st[i].small_n = sn as u8;
            for k in 0..sn {
                f.read_exact(&mut b4)?;
                m.sam.st[i].small_ch[k] = u32::from_le_bytes(b4);
                f.read_exact(&mut b4)?;
                m.sam.st[i].small_to[k] = i32::from_le_bytes(b4);
            }
            f.read_exact(&mut b4)?;
            m.sam.st[i].head = i32::from_le_bytes(b4);
        }
        for i in 0..ed_n {
            f.read_exact(&mut b4)?;
            m.sam.ed[i].ch = u32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.sam.ed[i].to = i32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.sam.ed[i].next = i32::from_le_bytes(b4);
        }
        read_u32_slice_le(&mut f, &mut m.sam.text)?;
        f.read_exact(&mut m.sam.boundary_after)?;

        // SAM cursor + prefix trace.
        f.read_exact(&mut b4)?;
        m.sam.last = i32::from_le_bytes(b4);
        f.read_exact(&mut b4)?;
        let text_states_n = u32::from_le_bytes(b4) as usize;
        if text_states_n != text_n + 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad text_states len",
            ));
        }
        m.sam.text_states.resize(text_states_n, 0);
        read_i32_slice_le(&mut f, &mut m.sam.text_states)?;
        for &v in &m.sam.text_states {
            if v < 0 || (v as usize) >= st_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad text_states entry",
                ));
            }
        }
        if m.sam.last < 0 || (m.sam.last as usize) >= st_n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad sam.last",
            ));
        }

        // LM
        f.read_exact(&mut b4)?;
        let alpha_n = u32::from_le_bytes(b4) as usize;
        f.read_exact(&mut b8)?;
        let total_uni = u64::from_le_bytes(b8);
        f.read_exact(&mut b4)?;
        let nodes_n = u32::from_le_bytes(b4) as usize;

        m.lm = LM::default();
        m.lm.alpha_n = alpha_n as u32;
        m.lm.total_uni = total_uni;
        m.lm.alphabet.resize(alpha_n, 0);
        m.lm.unigram.resize(alpha_n, 0);
        m.lm.ls = vec![
            LmState {
                head: -1,
                last_node: -1,
                ..LmState::default()
            };
            st_n
        ];
        m.lm.nodes.resize(nodes_n, CountNode::default());

        read_u32_slice_le(&mut f, &mut m.lm.alphabet)?;
        read_u64_slice_le(&mut f, &mut m.lm.unigram)?;
        for i in 0..st_n {
            f.read_exact(&mut b4)?;
            m.lm.ls[i].head = i32::from_le_bytes(b4);
            f.read_exact(&mut b8)?;
            m.lm.ls[i].total_n = u64::from_le_bytes(b8);
            f.read_exact(&mut b4)?;
            m.lm.ls[i].types_t = u32::from_le_bytes(b4);
            m.lm.ls[i].last_node = -1;
            m.lm.ls[i].last_sym = 0;
        }
        for i in 0..nodes_n {
            f.read_exact(&mut b4)?;
            m.lm.nodes[i].sym_idx = u32::from_le_bytes(b4);
            f.read_exact(&mut b8)?;
            m.lm.nodes[i].cnt = u64::from_le_bytes(b8);
            f.read_exact(&mut b4)?;
            m.lm.nodes[i].next = i32::from_le_bytes(b4);
        }

        // rebuild byte_map for lookups
        m.lm.has_byte_map = false;
        m.lm.byte_map = [-1; 256];
        let mut max_cp = 0u32;
        for &v in &m.lm.alphabet {
            if v > max_cp {
                max_cp = v;
            }
        }
        if max_cp < 256 {
            m.lm.has_byte_map = true;
            for (i, &c) in m.lm.alphabet.iter().enumerate() {
                m.lm.byte_map[c as usize] = i as i16;
            }
        }

        m.lm_built = true;
        m.dist.resize(alpha_n, 0.0);
        Ok(m)
    }

    pub fn prob_for_last(&mut self, sym: u32) -> f64 {
        if !self.lm_built {
            self.build_lm();
        }
        let v = self.sam.last;
        let sym_idx = self.lm.find_sym(sym);
        let mo = if self.max_order < 0 {
            -1
        } else {
            self.max_order
        };
        self.lm.prob_for_sym(&self.sam, mo, v, sym_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosa_md_example_basic() {
        // From rosa.md: ROSA predicts next token of best previous match.
        let x = b"ababa";
        let mut m = RosaPlus::new(1048576, false, 4, 0);
        m.train_example(x);
        m.build_lm();
        let out = m.generate(b"a", 10).unwrap();
        assert!(!out.is_empty());
    }

    #[test]
    fn tx_rollback_restores_sam_and_unigram_counts() {
        let mut m = RosaPlus::new(4, false, 0, 123);
        m.train_example(b"hello");
        m.build_lm_full_bytes_no_finalize_endpos();

        let base_text = m.sam.text.clone();
        let base_text_len = m.sam.text.len();
        let base_total_uni = m.lm.total_uni;
        assert!(base_text_len > 0);

        let mut tx = m.begin_tx();
        m.train_example_tx(&mut tx, b"abc");
        assert_eq!(m.lm.total_uni, base_total_uni + 3);
        assert_eq!(m.sam.text.len(), base_text_len + 3);

        m.rollback_tx(tx);
        assert_eq!(m.sam.text, base_text);
        assert_eq!(m.lm.total_uni, base_total_uni);
    }

    #[test]
    fn checkpoint_restore_reverts_append_only_buffers() {
        let mut m = RosaPlus::new(3, true, b'\n', 7);
        m.train_example(b"aaaa");

        let ck = m.checkpoint();
        let base_text = m.sam.text.clone();
        let base_states = m.sam.text_states.clone();
        let base_boundary = m.sam.boundary_after.clone();
        let base_last = m.sam.last;

        m.train_example(b"bbbb");
        assert_ne!(m.sam.text, base_text);

        m.restore(&ck);
        assert_eq!(m.sam.text, base_text);
        assert_eq!(m.sam.text_states, base_states);
        assert_eq!(m.sam.boundary_after, base_boundary);
        assert_eq!(m.sam.last, base_last);
        assert!(!m.lm_built);
    }

    #[test]
    fn exact_predictor_emits_deterministic_matches() {
        let mut p = RosaExactPredictor::new(32);
        p.observe_slice(b"abracadabra").unwrap();
        let predicted = p.predict_next();
        assert!(predicted.is_some());
    }

    #[test]
    fn exact_predictor_large_hint_has_bounded_startup_reserve() {
        let p = RosaExactPredictor::new_with_config(1_000_000_000, RosaExactPredictorConfig::default());
        assert!(p.st.capacity() <= SAM_INIT_ST_CAP_SOFT);
        assert!(p.ed.capacity() <= SAM_INIT_ED_CAP_SOFT);
    }

    #[test]
    fn exact_predictor_ram_cap_reduces_startup_reserve() {
        let cfg = RosaExactPredictorConfig {
            max_ram_bytes: Some(64 << 20),
            chunk_size: 1 << 20,
            track_boundaries: true,
        };
        let p = RosaExactPredictor::new_with_config(1_000_000_000, cfg);
        // 1/16 of 64 MiB is 4 MiB reserve budget per structure.
        let st_cap_limit = (4 << 20) / std::mem::size_of::<ExactState>();
        let ed_cap_limit = (4 << 20) / std::mem::size_of::<ExactEdge>();
        assert!(p.st.capacity() <= st_cap_limit + 1024);
        assert!(p.ed.capacity() <= ed_cap_limit + 1024);
    }
}
