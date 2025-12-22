#![allow(clippy::needless_range_loop)]

use std::fs::File;
use std::io::{Read, Write};

const SAM_SMALL_MAX: usize = 4;
const MAGIC: &[u8] = b"rosa_pb_v3\0";

#[derive(Clone, Copy, Default)]
struct SamState {
    link: i32,
    len: i32,
    endpos: i32,

    small_n: u8,
    small_ch: [u32; SAM_SMALL_MAX],
    small_to: [i32; SAM_SMALL_MAX],

    head: i32,
}

#[derive(Clone, Copy, Default)]
struct SamEdge {
    ch: u32,
    to: i32,
    next: i32,
}

#[derive(Default)]
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
        for i in 0..(st.small_n as usize) {
            if st.small_ch[i] == ch {
                return st.small_to[i];
            }
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
}

#[derive(Default)]
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

#[derive(Default)]
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

    /// Compute the unbiased predictive entropy rate (bits per symbol) of the given data.
    ///
    /// This uses a chunk-based prequential approach (training on past chunks to score the current one)
    /// to eliminate the "in-sample bias" present in simple plugin estimators.
    /// Complexity: O(N * Chunks) where Chunks is small (default 16).
    pub fn predictive_entropy_rate(&mut self, data: &[u8]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let cps: Vec<u32> = data.iter().map(|&b| b as u32).collect();

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
                self.build_lm();
                // Context state at the start of this chunk is the last state of the previous chunk.
                // text_states[start] is the state reached after feeding symbols 0..start-1.
                let mut v = self.sam.text_states[start];

                for &ch in chunk {
                    let sym_idx = self.lm.find_sym(ch);
                    let p = self.lm.prob_for_sym(&self.sam, self.max_order, v, sym_idx);
                    total_log_prob += p.log2();
                    count += 1;

                    // Advance context
                    v = self.sam.advance(v, ch);
                }
            }

            // Incremental training (adds to self.sam.text and updates structure)
            for &ch in chunk {
                self.sam.feed(ch);
            }
        }

        if count == 0 {
            // Fallback if data is too small for chunking
            self.build_lm();
            self.entropy_rate_plugin_cps(&cps)
        } else {
            -total_log_prob / (count as f64)
        }
    }

    /// Optimized entry point for already-decoded codepoints (used for joint entropy).
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
                self.build_lm();
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

    pub fn cross_entropy(&self, data: &[u8]) -> f64 {
        if !self.lm_built {
            return 0.0;
        }
        let cps: Vec<u32> = data.iter().map(|&b| b as u32).collect();
        self.cross_entropy_cps(&cps)
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
        let mut f = File::create(path)?;
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
        for &t in &self.sam.text {
            f.write_all(&t.to_le_bytes())?;
        }
        f.write_all(&self.sam.boundary_after)?;

        // LM
        f.write_all(&self.lm.alpha_n.to_le_bytes())?;
        f.write_all(&self.lm.total_uni.to_le_bytes())?;
        f.write_all(&(self.lm.nodes.len() as u32).to_le_bytes())?;
        for &a in &self.lm.alphabet {
            f.write_all(&a.to_le_bytes())?;
        }
        for &u in &self.lm.unigram {
            f.write_all(&u.to_le_bytes())?;
        }
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
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut f = File::open(path)?;
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
        for i in 0..text_n {
            f.read_exact(&mut b4)?;
            m.sam.text[i] = u32::from_le_bytes(b4);
        }
        f.read_exact(&mut m.sam.boundary_after)?;

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

        for i in 0..alpha_n {
            f.read_exact(&mut b4)?;
            m.lm.alphabet[i] = u32::from_le_bytes(b4);
        }
        for i in 0..alpha_n {
            f.read_exact(&mut b8)?;
            m.lm.unigram[i] = u64::from_le_bytes(b8);
        }
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
}
