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

const SAM_SMALL_MAX: usize = 4;
// NOTE: bump when on-disk format changes.
// v5 promotes SAM/LM overflow-link indices to unsigned 32-bit space for enwik-scale byte models
// and intentionally drops older model compatibility.
const MAGIC_V5: &[u8] = b"rosa_pb_v5\0";

type SamStateIx = i32;
type SamEdgeIx = u32;
type LmNodeIx = u32;

const SAM_STATE_NONE: SamStateIx = -1;
const SAM_EDGE_NONE: SamEdgeIx = u32::MAX;
const LM_NODE_NONE: LmNodeIx = u32::MAX;

// This crate is used byte-wise by infotheory; for fast incremental conditional updates we
// support an optional fixed 256-byte alphabet LM build/update path.
const BYTE_ALPHA_N: usize = 256;

#[inline(always)]
fn state_ix(idx: usize) -> SamStateIx {
    SamStateIx::try_from(idx).expect("rosa sam state index overflow")
}

#[inline(always)]
fn state_usize(idx: SamStateIx) -> usize {
    debug_assert!(idx >= 0, "negative rosa sam state index");
    idx as usize
}

#[inline(always)]
fn edge_ix(idx: usize) -> SamEdgeIx {
    SamEdgeIx::try_from(idx).expect("rosa sam edge index overflow")
}

#[inline(always)]
fn edge_usize(idx: SamEdgeIx) -> usize {
    idx as usize
}

#[inline(always)]
fn node_ix(idx: usize) -> LmNodeIx {
    LmNodeIx::try_from(idx).expect("rosa lm node index overflow")
}

#[inline(always)]
fn node_usize(idx: LmNodeIx) -> usize {
    idx as usize
}

#[inline(always)]
fn write_len64<W: Write>(w: &mut W, len: usize) -> std::io::Result<()> {
    w.write_all(&(len as u64).to_le_bytes())
}

#[inline(always)]
fn read_len64<R: Read>(r: &mut R) -> std::io::Result<usize> {
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8)?;
    usize::try_from(u64::from_le_bytes(b8))
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "length overflow"))
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
    link: SamStateIx,
    len: i32,
    endpos: i32,
    head: SamEdgeIx,

    small_ch: [u32; SAM_SMALL_MAX],
    small_to: [SamStateIx; SAM_SMALL_MAX],
    small_n: u8,
}

#[derive(Clone, Copy, Default)]
struct SamEdge {
    ch: u32,
    to: SamStateIx,
    next: SamEdgeIx,
}

#[derive(Clone, Default)]
struct Sam {
    st: Vec<SamState>,
    ed: Vec<SamEdge>,
    last: SamStateIx,

    text: Vec<u32>,
    text_states: Vec<SamStateIx>,
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

        let st_cap = if expected_chars > 0 {
            expected_chars * 2 + 16
        } else {
            1024
        };
        let ed_cap = if expected_chars > 0 {
            expected_chars * 3 + 16
        } else {
            2048
        };
        let text_cap = if expected_chars > 0 {
            expected_chars + 16
        } else {
            1024
        };
        s.st.reserve(st_cap);
        s.ed.reserve(ed_cap);
        s.text.reserve(text_cap);
        s.text_states.reserve(text_cap);
        s.boundary_after.reserve(text_cap);

        let root = SamState {
            link: SAM_STATE_NONE,
            len: 0,
            endpos: -1,
            small_n: 0,
            head: SAM_EDGE_NONE,
            ..Default::default()
        };
        s.st.push(root);
        s.text_states.push(0); // Root state for empty context
        s
    }

    #[inline(always)]
    fn get_edge(&self, v: SamStateIx, ch: u32) -> SamStateIx {
        let st = unsafe { self.st.get_unchecked(state_usize(v)) };
        for i in 0..(st.small_n as usize) {
            if st.small_ch[i] == ch {
                return st.small_to[i];
            }
        }
        let mut ei = st.head;
        while ei != SAM_EDGE_NONE {
            let e = unsafe { self.ed.get_unchecked(edge_usize(ei)) };
            if e.ch == ch {
                return e.to;
            }
            ei = e.next;
        }
        SAM_STATE_NONE
    }

    #[inline(always)]
    fn add_edge(&mut self, v: SamStateIx, ch: u32, to: SamStateIx) {
        let idx = edge_ix(self.ed.len());
        let head = self.st[state_usize(v)].head;
        self.ed.push(SamEdge { ch, to, next: head });
        self.st[state_usize(v)].head = idx;
    }

    #[inline(always)]
    fn add_edge_absent(&mut self, v: SamStateIx, ch: u32, to: SamStateIx) {
        let st = &mut self.st[state_usize(v)];
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
    fn replace_edge_to(
        &mut self,
        v: SamStateIx,
        ch: u32,
        old_to: SamStateIx,
        new_to: SamStateIx,
    ) -> bool {
        {
            let st = &mut self.st[state_usize(v)];
            for i in 0..(st.small_n as usize) {
                if st.small_ch[i] == ch && st.small_to[i] == old_to {
                    st.small_to[i] = new_to;
                    return true;
                }
            }
        }
        let mut ei = self.st[state_usize(v)].head;
        while ei != SAM_EDGE_NONE {
            let e = &mut self.ed[edge_usize(ei)];
            if e.ch == ch && e.to == old_to {
                e.to = new_to;
                return true;
            }
            ei = e.next;
        }
        false
    }

    fn clone_overflow_edges(&mut self, src: SamStateIx, dst: SamStateIx) {
        self.st[state_usize(dst)].head = SAM_EDGE_NONE;
        let mut ei = self.st[state_usize(src)].head;
        while ei != SAM_EDGE_NONE {
            let e = self.ed[edge_usize(ei)];
            self.add_edge(dst, e.ch, e.to);
            ei = e.next;
        }
    }

    fn feed(&mut self, ch: u32) {
        let i = self.text.len() as i32;
        self.text.push(ch);
        self.boundary_after.push(0);

        let g = self.last;
        let r = state_ix(self.st.len());
        let st_r = SamState {
            link: 0,
            len: self.st[state_usize(g)].len + 1,
            endpos: i,
            small_n: 0,
            head: SAM_EDGE_NONE,
            ..Default::default()
        };
        self.st.push(st_r);

        let mut p = g;
        let mut q;
        while p != SAM_STATE_NONE {
            q = self.get_edge(p, ch);
            if q != SAM_STATE_NONE {
                break;
            }
            self.add_edge_absent(p, ch, r);
            p = self.st[state_usize(p)].link;
        }

        if p == SAM_STATE_NONE {
            self.st[state_usize(r)].link = 0;
        } else {
            q = self.get_edge(p, ch);
            if self.st[state_usize(p)].len + 1 == self.st[state_usize(q)].len {
                self.st[state_usize(r)].link = q;
            } else {
                let u = state_ix(self.st.len());
                let mut st_u = self.st[state_usize(q)];
                st_u.len = self.st[state_usize(p)].len + 1;
                self.st.push(st_u);
                self.clone_overflow_edges(q, u);
                while p != SAM_STATE_NONE && self.replace_edge_to(p, ch, q, u) {
                    p = self.st[state_usize(p)].link;
                }
                self.st[state_usize(q)].link = u;
                self.st[state_usize(r)].link = u;
            }
        }

        self.last = r;
        self.text_states.push(r);

        // Maintain rightmost endpos online (ROSA deterministic predictor).
        let mut v = r;
        while v != SAM_STATE_NONE && self.st[state_usize(v)].endpos < i {
            self.st[state_usize(v)].endpos = i;
            v = self.st[state_usize(v)].link;
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
    fn advance(&self, mut v: SamStateIx, ch: u32) -> SamStateIx {
        let mut to;
        loop {
            to = self.get_edge(v, ch);
            if to != SAM_STATE_NONE {
                return to;
            }
            v = self.st[state_usize(v)].link;
            if v == SAM_STATE_NONE {
                break;
            }
        }
        to = self.get_edge(0, ch);
        if to == SAM_STATE_NONE { 0 } else { to }
    }

    #[inline(always)]
    fn predict_det(&self, v: SamStateIx) -> Option<u32> {
        let mut u = v;
        while u != SAM_STATE_NONE {
            let st = unsafe { self.st.get_unchecked(state_usize(u)) };
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
    fn add_edge_tx(&mut self, tx: &mut SamTx, v: SamStateIx, ch: u32, to: SamStateIx) {
        let idx = edge_ix(self.ed.len());
        let head = self.st[state_usize(v)].head;
        self.ed.push(SamEdge { ch, to, next: head });
        self.record_state_change(tx, state_usize(v));
        self.st[state_usize(v)].head = idx;
    }

    #[inline(always)]
    fn add_edge_absent_tx(&mut self, tx: &mut SamTx, v: SamStateIx, ch: u32, to: SamStateIx) {
        let v_usize = state_usize(v);
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
        v: SamStateIx,
        ch: u32,
        old_to: SamStateIx,
        new_to: SamStateIx,
    ) -> bool {
        // small edges
        {
            let st = &self.st[state_usize(v)];
            for i in 0..(st.small_n as usize) {
                if st.small_ch[i] == ch && st.small_to[i] == old_to {
                    self.record_state_change(tx, state_usize(v));
                    self.st[state_usize(v)].small_to[i] = new_to;
                    return true;
                }
            }
        }
        // overflow edges
        let mut ei = self.st[state_usize(v)].head;
        while ei != SAM_EDGE_NONE {
            let eidx = edge_usize(ei);
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

    fn clone_overflow_edges_tx(&mut self, tx: &mut SamTx, src: SamStateIx, dst: SamStateIx) {
        self.record_state_change(tx, state_usize(dst));
        self.st[state_usize(dst)].head = SAM_EDGE_NONE;
        let mut ei = self.st[state_usize(src)].head;
        while ei != SAM_EDGE_NONE {
            let e = self.ed[edge_usize(ei)];
            self.add_edge_tx(tx, dst, e.ch, e.to);
            ei = e.next;
        }
    }

    fn feed_tx(&mut self, tx: &mut SamTx, ch: u32) {
        let i = self.text.len() as i32;
        self.text.push(ch);
        self.boundary_after.push(0);

        let g = self.last;
        let r = state_ix(self.st.len());
        let st_r = SamState {
            link: 0,
            len: self.st[state_usize(g)].len + 1,
            endpos: i,
            small_n: 0,
            head: SAM_EDGE_NONE,
            ..Default::default()
        };
        self.st.push(st_r);

        let mut p = g;
        let mut q;
        while p != SAM_STATE_NONE {
            q = self.get_edge(p, ch);
            if q != SAM_STATE_NONE {
                break;
            }
            self.add_edge_absent_tx(tx, p, ch, r);
            p = self.st[state_usize(p)].link;
        }

        if p == SAM_STATE_NONE {
            // link of r is in newly appended state; safe.
            self.st[state_usize(r)].link = 0;
        } else {
            q = self.get_edge(p, ch);
            if self.st[state_usize(p)].len + 1 == self.st[state_usize(q)].len {
                self.st[state_usize(r)].link = q;
            } else {
                let u = state_ix(self.st.len());
                let mut st_u = self.st[state_usize(q)];
                st_u.len = self.st[state_usize(p)].len + 1;
                self.st.push(st_u);
                self.clone_overflow_edges_tx(tx, q, u);
                while p != SAM_STATE_NONE && self.replace_edge_to_tx(tx, p, ch, q, u) {
                    p = self.st[state_usize(p)].link;
                }
                // q is an existing state; record before mutation.
                self.record_state_change(tx, state_usize(q));
                self.st[state_usize(q)].link = u;
                self.st[state_usize(r)].link = u;
            }
        }

        self.last = r;
        self.text_states.push(r);

        // Maintain rightmost endpos online (ROSA deterministic predictor).
        let mut v = r;
        while v != SAM_STATE_NONE && self.st[state_usize(v)].endpos < i {
            self.record_state_change(tx, state_usize(v));
            self.st[state_usize(v)].endpos = i;
            v = self.st[state_usize(v)].link;
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

#[derive(Clone)]
struct SamTx {
    old_last: SamStateIx,
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
    head: LmNodeIx,
    total_n: u64,
    types_t: u32,

    last_sym: u32,
    last_node: LmNodeIx,
}

#[derive(Clone, Copy, Default)]
struct CountNode {
    sym_idx: u32,
    cnt: u64,
    next: LmNodeIx,
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
        if last != LM_NODE_NONE && self.nodes[node_usize(last)].sym_idx == sym_idx {
            self.nodes[node_usize(last)].cnt += add;
            ls.total_n += add;
            return;
        }

        let mut ni = ls.head;
        while ni != LM_NODE_NONE {
            let node = &mut self.nodes[node_usize(ni)];
            if node.sym_idx == sym_idx {
                node.cnt += add;
                ls.total_n += add;
                ls.last_node = ni;
                ls.last_sym = sym_idx;
                return;
            }
            ni = node.next;
        }

        let idx = node_ix(self.nodes.len());
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
                head: LM_NODE_NONE,
                last_node: LM_NODE_NONE,
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
                let mut v = 0;
                for i in seg_start..(seg_end - 1) {
                    let ch = sam.text[i];
                    v = sam.advance(v, ch);
                    let mut ctx = v;
                    if max_order >= 0 {
                        while ctx != SAM_STATE_NONE
                            && (sam.st[state_usize(ctx)].len as i64) > max_order
                        {
                            ctx = sam.st[state_usize(ctx)].link;
                        }
                        if ctx == SAM_STATE_NONE {
                            ctx = 0;
                        }
                    }
                    let nxt = sam.text[i + 1];
                    let si = self.find_sym(nxt);
                    if si >= 0 {
                        self.inc(state_usize(ctx) as u32, si as u32, 1);
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
            while ni != LM_NODE_NONE {
                let node = self.nodes[node_usize(ni)];
                self.inc(state_usize(p) as u32, node.sym_idx, node.cnt);
                ni = node.next;
            }
        }
    }

    /// Efficient pointwise probability Estimation of a single symbol.
    /// Avoids allocating and writing to a dense distribution array.
    fn prob_for_sym(&self, sam: &Sam, max_order: i64, v: SamStateIx, sym_idx: i32) -> f64 {
        if sym_idx < 0 {
            return 1.0 / (self.alpha_n.max(1) as f64);
        }
        let sym_idx = sym_idx as u32;
        let mut p_accum = 0.0f64;
        let mut residual = 1.0f64;
        let mut u = v;

        while u != SAM_STATE_NONE {
            if !(max_order >= 0 && (sam.st[state_usize(u)].len as i64) > max_order) {
                let n = self.ls[state_usize(u)].total_n;
                let t = self.ls[state_usize(u)].types_t;
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
                    let mut ni = self.ls[state_usize(u)].head;
                    while ni != LM_NODE_NONE {
                        let node = self.nodes[node_usize(ni)];
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
            u = sam.st[state_usize(u)].link;
        }

        if self.total_uni > 0 && residual > 0.0 {
            let p_uni = self.unigram[sym_idx as usize] as f64 / self.total_uni as f64;
            p_accum += residual * p_uni;
        } else if residual > 0.0 {
            p_accum += residual * (1.0 / self.alpha_n.max(1) as f64);
        }

        p_accum.clamp(1e-12, 1.0)
    }

    fn probs_for_state(&self, sam: &Sam, max_order: i64, v: SamStateIx, out: &mut [f64]) {
        out.fill(0.0);
        let mut residual = 1.0f64;
        let mut u = v;
        while u != SAM_STATE_NONE {
            if !(max_order >= 0 && (sam.st[state_usize(u)].len as i64) > max_order) {
                let n = self.ls[state_usize(u)].total_n;
                let t = self.ls[state_usize(u)].types_t;
                if n > 0 {
                    let lam = if t > 0 {
                        (n as f64) / ((n + (t as u64)) as f64)
                    } else {
                        1.0
                    };
                    let scale = residual * lam;
                    let inv_n = 1.0 / (n as f64);
                    let mut ni = self.ls[state_usize(u)].head;
                    while ni != LM_NODE_NONE {
                        let node = self.nodes[node_usize(ni)];
                        out[node.sym_idx as usize] += scale * ((node.cnt as f64) * inv_n);
                        ni = node.next;
                    }
                    residual *= 1.0 - lam;
                }
            }
            u = sam.st[state_usize(u)].link;
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
        if last != LM_NODE_NONE && self.nodes[node_usize(last)].sym_idx == sym_idx {
            let ni = node_usize(last);
            tx.node_changes.push((ni, self.nodes[ni]));
            self.nodes[ni].cnt += add;
            ls.total_n += add;
            return;
        }

        let mut ni = ls.head;
        while ni != LM_NODE_NONE {
            let idx = node_usize(ni);
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
        let idx = node_ix(self.nodes.len());
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
        if let Ok(path) = std::env::var("ROSAPLUS_RNG_PATH")
            && !path.is_empty()
            && let Ok(mut f) = File::open(path)
        {
            let mut b = Vec::new();
            if f.read_to_end(&mut b).is_ok() && b.len() >= 8 {
                let n = b.len();
                r.pos = ((seed.wrapping_mul(8)) as usize) % n;
                r.buf = b;
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
/// ROSA+ predictive model with optional transactional updates.
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
    sam_last: SamStateIx,
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
    /// Create a new ROSA+ model.
    ///
    /// `max_order < 0` enables adaptive order selection in predictive scoring.
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

    /// Train on one byte sequence, optionally appending EOT marker.
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

    /// Build the language model from current SAM state.
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

    fn train_example_tx_impl(&mut self, tx: &mut RosaTx, s: &[u8], mark_boundary: bool) {
        if s.is_empty() {
            return;
        }

        // Ensure LS has entries for current states.
        if self.lm.ls.len() < self.sam.st.len() {
            self.lm.ls.resize(
                self.sam.st.len(),
                LmState {
                    head: LM_NODE_NONE,
                    last_node: LM_NODE_NONE,
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
                    head: LM_NODE_NONE,
                    last_node: LM_NODE_NONE,
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
                && self
                    .sam
                    .boundary_after
                    .get(seg_start - 1)
                    .copied()
                    .unwrap_or(0)
                    == 0
            {
                start_i = seg_start - 1;
            }
            for i in start_i..(seg_end - 1) {
                // ctx state after consuming sam.text[i] within its segment
                let mut ctx = self.sam.text_states[i + 1];
                if mo >= 0 {
                    while ctx != SAM_STATE_NONE && (self.sam.st[state_usize(ctx)].len as i64) > mo {
                        ctx = self.sam.st[state_usize(ctx)].link;
                    }
                    if ctx == SAM_STATE_NONE {
                        ctx = 0;
                    }
                }
                let nxt = self.sam.text[i + 1];
                let si = self.lm.find_sym(nxt);
                if si >= 0 {
                    let mut u = ctx;
                    while u != SAM_STATE_NONE {
                        self.lm
                            .inc_tx(&mut tx.lm, state_usize(u) as u32, si as u32, 1);
                        u = self.sam.st[state_usize(u)].link;
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

    fn predictive_entropy_rate_order(data: &[u8], max_order: i64, seed: u64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let num_chunks = 16;
        let chunk_size = data.len().div_ceil(num_chunks);
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

    /// Approximate in-memory footprint of major model buffers.
    pub fn estimated_size_bytes(&self) -> usize {
        use std::mem::size_of;

        let mut n = 0usize;

        n = n.saturating_add(self.sam.st.len().saturating_mul(size_of::<SamState>()));
        n = n.saturating_add(self.sam.ed.len().saturating_mul(size_of::<SamEdge>()));
        n = n.saturating_add(self.sam.text.len().saturating_mul(size_of::<u32>()));
        n = n.saturating_add(
            self.sam
                .text_states
                .len()
                .saturating_mul(size_of::<SamStateIx>()),
        );
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

    /// Shrink auxiliary scratch buffers to fit current usage.
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

    /// Generate continuation bytes from a prompt.
    ///
    /// Returns `None` if LM is not built yet.
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

    /// Predictive entropy rate on codepoint streams.
    pub fn entropy_rate_cps(&mut self, cps: &[u32]) -> f64 {
        if cps.len() < 2 {
            return 0.0;
        }

        self.sam = Sam::new(cps.len());
        self.lm_built = false;

        let num_chunks = 16;
        let chunk_size = cps.len().div_ceil(num_chunks);
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

    #[allow(dead_code)]
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

    /// Cross entropy of byte data under current LM state.
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

    /// Cross entropy of codepoint data under current LM state.
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

    /// Persist trained SAM+LM state to disk.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        if !self.lm_built {
            return Err(std::io::Error::other("LM not built"));
        }

        // Transactional conditional updates require a valid prefix-state trace.
        // If this invariant is violated, the loaded model would be unusable.
        if self.sam.text_states.len() != self.sam.text.len() + 1 {
            return Err(std::io::Error::other(
                "SAM text_states mismatch (expected text.len()+1)",
            ));
        }
        let mut f = BufWriter::with_capacity(1024 * 1024, File::create(path)?);
        f.write_all(MAGIC_V5)?;
        f.write_all(&self.max_order.to_le_bytes())?;
        f.write_all(&(self.use_eot as i32).to_le_bytes())?;
        f.write_all(&self.eot.to_le_bytes())?;
        f.write_all(&self.seed.to_le_bytes())?;

        // SAM
        write_len64(&mut f, self.sam.st.len())?;
        write_len64(&mut f, self.sam.ed.len())?;
        write_len64(&mut f, self.sam.text.len())?;
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
        write_len64(&mut f, self.sam.text_states.len())?;
        write_i32_slice_le(&mut f, &self.sam.text_states)?;

        // LM
        f.write_all(&self.lm.alpha_n.to_le_bytes())?;
        f.write_all(&self.lm.total_uni.to_le_bytes())?;
        write_len64(&mut f, self.lm.nodes.len())?;
        write_u32_slice_le(&mut f, &self.lm.alphabet)?;
        write_u64_slice_le(&mut f, &self.lm.unigram)?;
        for ls in &self.lm.ls {
            f.write_all(&ls.head.to_le_bytes())?;
            f.write_all(&ls.total_n.to_le_bytes())?;
            f.write_all(&ls.types_t.to_le_bytes())?;
            f.write_all(&ls.last_sym.to_le_bytes())?;
            f.write_all(&ls.last_node.to_le_bytes())?;
        }
        for n in &self.lm.nodes {
            f.write_all(&n.sym_idx.to_le_bytes())?;
            f.write_all(&n.cnt.to_le_bytes())?;
            f.write_all(&n.next.to_le_bytes())?;
        }
        f.flush()?;
        Ok(())
    }

    /// Load a previously saved ROSA+ model from disk.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut f = BufReader::with_capacity(1024 * 1024, File::open(path)?);
        let mut magic = vec![0u8; MAGIC_V5.len()];
        f.read_exact(&mut magic)?;
        if magic != MAGIC_V5 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad magic or unsupported ROSA+ model version",
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
        let st_n = read_len64(&mut f)?;
        let ed_n = read_len64(&mut f)?;
        let text_n = read_len64(&mut f)?;

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
            m.sam.st[i].head = u32::from_le_bytes(b4);
        }
        for i in 0..ed_n {
            f.read_exact(&mut b4)?;
            m.sam.ed[i].ch = u32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.sam.ed[i].to = i32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.sam.ed[i].next = u32::from_le_bytes(b4);
        }
        read_u32_slice_le(&mut f, &mut m.sam.text)?;
        f.read_exact(&mut m.sam.boundary_after)?;

        // SAM cursor + prefix trace.
        f.read_exact(&mut b4)?;
        m.sam.last = i32::from_le_bytes(b4);
        let text_states_n = read_len64(&mut f)?;
        if text_states_n != text_n + 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad text_states len",
            ));
        }
        m.sam.text_states.resize(text_states_n, 0);
        read_i32_slice_le(&mut f, &mut m.sam.text_states)?;
        for &v in &m.sam.text_states {
            if v < 0 || state_usize(v) >= st_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad text_states entry",
                ));
            }
        }
        if m.sam.last < 0 || state_usize(m.sam.last) >= st_n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad sam.last",
            ));
        }
        for st in &m.sam.st {
            if st.link != SAM_STATE_NONE && state_usize(st.link) >= st_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad sam link",
                ));
            }
            for k in 0..(st.small_n as usize) {
                let to = st.small_to[k];
                if to < 0 || state_usize(to) >= st_n {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "bad sam small edge",
                    ));
                }
            }
            if st.head != SAM_EDGE_NONE && edge_usize(st.head) >= ed_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad sam edge head",
                ));
            }
        }
        for edge in &m.sam.ed {
            if edge.to < 0 || state_usize(edge.to) >= st_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad sam edge target",
                ));
            }
            if edge.next != SAM_EDGE_NONE && edge_usize(edge.next) >= ed_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad sam edge next",
                ));
            }
        }

        // LM
        f.read_exact(&mut b4)?;
        let alpha_n = u32::from_le_bytes(b4) as usize;
        f.read_exact(&mut b8)?;
        let total_uni = u64::from_le_bytes(b8);
        let nodes_n = read_len64(&mut f)?;

        m.lm = LM::default();
        m.lm.alpha_n = alpha_n as u32;
        m.lm.total_uni = total_uni;
        m.lm.alphabet.resize(alpha_n, 0);
        m.lm.unigram.resize(alpha_n, 0);
        m.lm.ls = vec![
            LmState {
                head: LM_NODE_NONE,
                last_node: LM_NODE_NONE,
                ..LmState::default()
            };
            st_n
        ];
        m.lm.nodes.resize(nodes_n, CountNode::default());

        read_u32_slice_le(&mut f, &mut m.lm.alphabet)?;
        read_u64_slice_le(&mut f, &mut m.lm.unigram)?;
        for i in 0..st_n {
            f.read_exact(&mut b4)?;
            m.lm.ls[i].head = u32::from_le_bytes(b4);
            f.read_exact(&mut b8)?;
            m.lm.ls[i].total_n = u64::from_le_bytes(b8);
            f.read_exact(&mut b4)?;
            m.lm.ls[i].types_t = u32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.lm.ls[i].last_sym = u32::from_le_bytes(b4);
            f.read_exact(&mut b4)?;
            m.lm.ls[i].last_node = u32::from_le_bytes(b4);
        }
        for i in 0..nodes_n {
            f.read_exact(&mut b4)?;
            m.lm.nodes[i].sym_idx = u32::from_le_bytes(b4);
            f.read_exact(&mut b8)?;
            m.lm.nodes[i].cnt = u64::from_le_bytes(b8);
            f.read_exact(&mut b4)?;
            m.lm.nodes[i].next = u32::from_le_bytes(b4);
        }
        for ls in &m.lm.ls {
            if ls.head != LM_NODE_NONE && node_usize(ls.head) >= nodes_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad lm head",
                ));
            }
            if ls.last_node != LM_NODE_NONE && node_usize(ls.last_node) >= nodes_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad lm last_node",
                ));
            }
        }
        for node in &m.lm.nodes {
            if node.next != LM_NODE_NONE && node_usize(node.next) >= nodes_n {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad lm next",
                ));
            }
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

    /// Probability of `sym` from current SAM cursor (`sam.last`).
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

    /// Fill a dense byte-wise probability vector for the current SAM cursor (`sam.last`).
    ///
    /// `out` must have length at least 256. The output is normalized.
    pub fn fill_probs_for_last_bytes(&mut self, out: &mut [f64]) {
        debug_assert!(out.len() >= 256);
        if !self.lm_built {
            self.build_lm();
        }

        let v = self.sam.last;
        let mo = if self.max_order < 0 {
            -1
        } else {
            self.max_order
        };
        self.dist.resize(self.lm.alpha_n as usize, 0.0);
        self.lm.probs_for_state(&self.sam, mo, v, &mut self.dist);

        out[..256].fill(0.0);
        for (i, &cp) in self.lm.alphabet.iter().enumerate() {
            if cp < 256 {
                out[cp as usize] = self.dist[i];
            }
        }

        let sum: f64 = out[..256].iter().sum();
        if sum.is_finite() && sum > 0.0 {
            let inv = 1.0 / sum;
            for p in &mut out[..256] {
                *p *= inv;
            }
        } else {
            let u = 1.0 / 256.0;
            for p in &mut out[..256] {
                *p = u;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_path(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "infotheory_rosaplus_{tag}_{}_{}.bin",
            std::process::id(),
            nanos
        ))
    }

    fn manual_chunked_entropy_rate_bytes(data: &[u8], max_order: i64, seed: u64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let num_chunks = 16;
        let chunk_size = data.len().div_ceil(num_chunks);
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

    fn manual_chunked_entropy_rate_cps(data: &[u32], max_order: i64, seed: u64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mut m = RosaPlus::new(max_order, false, 0, seed);
        m.sam = Sam::new(data.len());
        m.lm_built = false;

        let num_chunks = 16;
        let chunk_size = data.len().div_ceil(num_chunks);
        let mut total_log_prob = 0.0f64;
        let mut count = 0usize;

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(data.len());
            if start >= end {
                break;
            }
            let chunk = &data[start..end];
            if i > 0 {
                m.build_lm_no_finalize_endpos();
                let mut v = m.sam.text_states[start];
                for &ch in chunk {
                    let sym_idx = m.lm.find_sym(ch);
                    let p = m.lm.prob_for_sym(&m.sam, max_order, v, sym_idx);
                    total_log_prob += p.log2();
                    count += 1;
                    v = m.sam.advance(v, ch);
                }
            }
            for &ch in chunk {
                m.sam.feed(ch);
            }
        }

        if count == 0 {
            m.build_lm();
            m.entropy_rate_plugin_cps(data)
        } else {
            -total_log_prob / (count as f64)
        }
    }

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
    fn predictive_entropy_rate_matches_chunked_reference_fixed_order() {
        let data = b"abracadabra abracadabra abracadabra";
        let seed = 11;
        let expected = manual_chunked_entropy_rate_bytes(data, 4, seed);
        let mut m = RosaPlus::new(4, false, 0, seed);
        let got = m.predictive_entropy_rate(data);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn predictive_entropy_rate_uncapped_matches_candidate_search() {
        let data = b"the quick brown fox jumps over the lazy dog the quick brown fox";
        let seed = 29;
        let mut expected = f64::INFINITY;
        for &mo in &[0, 1, 2, 4, 8, 16, 32, 64] {
            if mo as usize >= data.len() {
                continue;
            }
            expected = expected.min(manual_chunked_entropy_rate_bytes(data, mo, seed));
        }
        let mut m = RosaPlus::new(-1, false, 0, seed);
        let got = m.predictive_entropy_rate(data);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn entropy_rate_cps_matches_chunked_reference() {
        let data = [0u32, 7, 0, 42, 7, 42, 0, 7, 42, 42];
        let seed = 31;
        let expected = manual_chunked_entropy_rate_cps(&data, -1, seed);
        let mut m = RosaPlus::new(-1, false, 0, seed);
        let got = m.entropy_rate_cps(&data);
        assert!((got - expected).abs() < 1e-12);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn wide_index_helpers_preserve_large_indices() {
        let large = (i32::MAX as usize) + 17;
        assert_eq!(edge_usize(edge_ix(large)), large);
        assert_eq!(node_usize(node_ix(large)), large);
    }

    #[test]
    fn save_load_roundtrip_preserves_state_and_probabilities() {
        let path = temp_model_path("roundtrip");
        let mut m = RosaPlus::new(8, true, b'\n', 1234);
        m.train_example(b"abracadabra");
        m.build_lm();
        let before_prob = m.prob_for_last(b'a' as u32);
        let before_size = m.estimated_size_bytes();
        let before_text = m.sam.text.clone();
        let before_states = m.sam.text_states.clone();
        let before_last = m.sam.last;
        let before_nodes = m.lm.nodes.len();
        let path_str = path.to_string_lossy().into_owned();

        m.save(&path_str).expect("save failed");
        let mut loaded = RosaPlus::load(&path_str).expect("load failed");
        fs::remove_file(&path).expect("cleanup failed");

        assert_eq!(loaded.max_order, m.max_order);
        assert_eq!(loaded.use_eot, m.use_eot);
        assert_eq!(loaded.eot, m.eot);
        assert_eq!(loaded.seed, m.seed);
        assert_eq!(loaded.sam.text, before_text);
        assert_eq!(loaded.sam.text_states, before_states);
        assert_eq!(loaded.sam.last, before_last);
        assert_eq!(loaded.lm.nodes.len(), before_nodes);
        assert_eq!(loaded.estimated_size_bytes(), before_size);
        assert!((loaded.prob_for_last(b'a' as u32) - before_prob).abs() < 1e-12);
    }
}
