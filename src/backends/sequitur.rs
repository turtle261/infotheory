use ahash::AHashMap;

const PDF_MIN: f64 = crate::mixture::DEFAULT_MIN_PROB;
const RAW_FALLBACK_MAX: usize = 4;

type NodeIx = u32;
type RuleId = u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Symbol {
    Terminal(u8),
    NonTerminal(RuleId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NodeData {
    Guard(RuleId),
    Sym(Symbol),
}

#[derive(Clone, Copy, Debug)]
struct Node {
    prev: NodeIx,
    next: NodeIx,
    data: NodeData,
}

#[derive(Clone, Debug)]
struct Rule {
    guard: NodeIx,
    ref_count: u32,
    active: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ContextFollowers {
    counts: Vec<(u8, u64)>,
    total: u64,
}

impl ContextFollowers {
    fn observe(&mut self, symbol: u8) {
        if let Some((_, count)) = self.counts.iter_mut().find(|(s, _)| *s == symbol) {
            *count += 1;
        } else {
            self.counts.push((symbol, 1));
        }
        self.total += 1;
    }

    fn distinct(&self) -> usize {
        self.counts.len()
    }
}

#[derive(Clone, Debug)]
enum UndoOp {
    SetPrev {
        node: NodeIx,
        old: NodeIx,
    },
    SetNext {
        node: NodeIx,
        old: NodeIx,
    },
    SetRuleRefCount {
        rule: RuleId,
        old: u32,
    },
    SetRuleActive {
        rule: RuleId,
        old: bool,
    },
    SetDigram {
        key: u64,
        old: Option<NodeIx>,
    },
    SetContextFollowers {
        key: Box<[u8]>,
        old: Option<ContextFollowers>,
    },
    SetUnigramSymbol {
        symbol: u8,
        old_count: u64,
        old_total: u64,
    },
    SetCommittedRawTail {
        old: Vec<u8>,
    },
    SetFrozenRawTail {
        old: Vec<u8>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequiturCheckpoint {
    undo_len: usize,
    node_len: usize,
    rule_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalRule {
    pub id: usize,
    pub rhs: Vec<CanonicalSymbol>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalSymbol {
    Terminal(u8),
    NonTerminal(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalGrammar {
    pub rules: Vec<CanonicalRule>,
}

#[derive(Clone, Debug)]
pub struct SequiturModel {
    context_bytes: usize,
    nodes: Vec<Node>,
    rules: Vec<Rule>,
    dummy: NodeIx,
    digrams: AHashMap<u64, NodeIx>,
    followers: AHashMap<Box<[u8]>, ContextFollowers>,
    unigram: [u64; 256],
    unigram_total: u64,
    committed_raw_tail: Vec<u8>,
    frozen_raw_tail: Vec<u8>,
    pdf: [f64; 256],
    pdf_valid: bool,
    undo: Vec<UndoOp>,
    undo_enabled: bool,
}

impl SequiturModel {
    pub fn new(context_bytes: usize) -> Self {
        let context_bytes = context_bytes.max(2);
        let mut nodes = Vec::with_capacity(16);
        nodes.push(Node {
            prev: 0,
            next: 0,
            data: NodeData::Guard(0),
        });
        nodes.push(Node {
            prev: 1,
            next: 1,
            data: NodeData::Guard(0),
        });
        let rules = vec![Rule {
            guard: 0,
            ref_count: 0,
            active: true,
        }];
        Self {
            context_bytes,
            nodes,
            rules,
            dummy: 1,
            digrams: AHashMap::new(),
            followers: AHashMap::new(),
            unigram: [0; 256],
            unigram_total: 0,
            committed_raw_tail: Vec::with_capacity(RAW_FALLBACK_MAX),
            frozen_raw_tail: Vec::with_capacity(RAW_FALLBACK_MAX),
            pdf: [1.0 / 256.0; 256],
            pdf_valid: false,
            undo: Vec::new(),
            undo_enabled: false,
        }
    }

    pub fn reserve_for_stream(&mut self, additional_symbols: usize) {
        self.nodes.reserve(additional_symbols.saturating_mul(2));
        self.digrams.reserve(additional_symbols);
    }

    pub fn begin_stream(&mut self, total_symbols: Option<u64>) {
        if let Some(total) = total_symbols {
            let reserve = usize::try_from(total).unwrap_or(usize::MAX / 4);
            self.reserve_for_stream(reserve);
        }
        self.frozen_raw_tail.clear();
        self.pdf_valid = false;
    }

    pub fn finish_stream(&mut self) {}

    pub fn checkpoint(&mut self) -> SequiturCheckpoint {
        self.undo_enabled = true;
        SequiturCheckpoint {
            undo_len: self.undo.len(),
            node_len: self.nodes.len(),
            rule_len: self.rules.len(),
        }
    }

    pub fn restore(&mut self, checkpoint: &SequiturCheckpoint) {
        let saved = self.undo_enabled;
        self.undo_enabled = false;
        while self.undo.len() > checkpoint.undo_len {
            let op = self.undo.pop().expect("undo underflow");
            self.apply_undo(op);
        }
        self.nodes.truncate(checkpoint.node_len);
        self.rules.truncate(checkpoint.rule_len);
        self.undo_enabled = saved;
        self.pdf_valid = false;
    }

    pub fn clear_checkpoints(&mut self) {
        self.undo.clear();
        self.undo_enabled = false;
    }

    pub fn reset_frozen(&mut self) {
        self.frozen_raw_tail.clear();
        self.pdf_valid = false;
    }

    pub fn fill_pdf(&mut self, out: &mut [f64; 256]) {
        self.ensure_pdf();
        out.copy_from_slice(&self.pdf);
    }

    pub fn pdf(&mut self) -> &[f64; 256] {
        self.ensure_pdf();
        &self.pdf
    }

    pub fn log_prob(&mut self, symbol: u8, min_prob: f64) -> f64 {
        self.ensure_pdf();
        self.pdf[symbol as usize].max(min_prob).ln()
    }

    pub fn update(&mut self, symbol: u8) {
        self.observe_symbol_in_stats(symbol);
        self.append_terminal(symbol);
        self.record_committed_raw_tail(symbol);
        if !self.frozen_raw_tail.is_empty() {
            self.record_frozen_raw_tail_inner(Vec::new());
        }
        self.pdf_valid = false;
    }

    pub fn update_frozen(&mut self, symbol: u8) {
        let mut next = self.frozen_raw_tail.clone();
        next.push(symbol);
        if next.len() > RAW_FALLBACK_MAX {
            let drain = next.len() - RAW_FALLBACK_MAX;
            next.drain(0..drain);
        }
        self.record_frozen_raw_tail_inner(next);
        self.pdf_valid = false;
    }

    pub fn decode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.decode_rule(0, &mut out);
        out
    }

    pub fn canonical_grammar(&self) -> CanonicalGrammar {
        let mut order = Vec::<RuleId>::new();
        let mut seen = AHashMap::<RuleId, usize>::new();
        self.collect_rule_preorder(0, &mut order, &mut seen);
        let mut canonical = Vec::with_capacity(order.len());
        for (idx, &rule_id) in order.iter().enumerate() {
            let mut rhs = Vec::new();
            let guard = self.rules[rule_id as usize].guard;
            let mut node = self.nodes[guard as usize].next;
            while node != guard {
                match self.nodes[node as usize].data {
                    NodeData::Guard(_) => unreachable!("guard in rule body"),
                    NodeData::Sym(Symbol::Terminal(byte)) => {
                        rhs.push(CanonicalSymbol::Terminal(byte))
                    }
                    NodeData::Sym(Symbol::NonTerminal(child)) => {
                        let mapped = *seen
                            .get(&child)
                            .expect("canonical grammar missing child mapping");
                        rhs.push(CanonicalSymbol::NonTerminal(mapped));
                    }
                }
                node = self.nodes[node as usize].next;
            }
            canonical.push(CanonicalRule { id: idx, rhs });
        }
        CanonicalGrammar { rules: canonical }
    }

    pub fn predictive_trace(&mut self, data: &[u8], alphabet_prefix: usize) -> Vec<Vec<f64>> {
        let mut trace = Vec::with_capacity(data.len());
        let mut pdf = [0.0; 256];
        for &byte in data {
            self.fill_pdf(&mut pdf);
            trace.push(pdf[..alphabet_prefix.min(256)].to_vec());
            self.update(byte);
        }
        trace
    }

    #[cfg(test)]
    pub fn validate_invariants(&self) -> Result<(), String> {
        self.validate_rule_shapes()?;
        self.validate_rule_refcounts()?;
        self.validate_digram_uniqueness()?;
        Ok(())
    }

    fn collect_rule_preorder(
        &self,
        rule_id: RuleId,
        order: &mut Vec<RuleId>,
        seen: &mut AHashMap<RuleId, usize>,
    ) {
        if seen.contains_key(&rule_id) {
            return;
        }
        let idx = order.len();
        seen.insert(rule_id, idx);
        order.push(rule_id);
        let guard = self.rules[rule_id as usize].guard;
        let mut node = self.nodes[guard as usize].next;
        while node != guard {
            if let NodeData::Sym(Symbol::NonTerminal(child)) = self.nodes[node as usize].data {
                if self.rules[child as usize].active {
                    self.collect_rule_preorder(child, order, seen);
                }
            }
            node = self.nodes[node as usize].next;
        }
    }

    fn apply_undo(&mut self, op: UndoOp) {
        match op {
            UndoOp::SetPrev { node, old } => {
                self.nodes[node as usize].prev = old;
            }
            UndoOp::SetNext { node, old } => {
                self.nodes[node as usize].next = old;
            }
            UndoOp::SetRuleRefCount { rule, old } => {
                self.rules[rule as usize].ref_count = old;
            }
            UndoOp::SetRuleActive { rule, old } => {
                self.rules[rule as usize].active = old;
            }
            UndoOp::SetDigram { key, old } => {
                if let Some(node) = old {
                    self.digrams.insert(key, node);
                } else {
                    self.digrams.remove(&key);
                }
            }
            UndoOp::SetContextFollowers { key, old } => {
                if let Some(state) = old {
                    self.followers.insert(key, state);
                } else {
                    self.followers.remove(key.as_ref());
                }
            }
            UndoOp::SetUnigramSymbol {
                symbol,
                old_count,
                old_total,
            } => {
                self.unigram[symbol as usize] = old_count;
                self.unigram_total = old_total;
            }
            UndoOp::SetCommittedRawTail { old } => {
                self.committed_raw_tail = old;
            }
            UndoOp::SetFrozenRawTail { old } => {
                self.frozen_raw_tail = old;
            }
        }
    }

    fn push_undo(&mut self, op: UndoOp) {
        if self.undo_enabled {
            self.undo.push(op);
        }
    }

    fn set_prev(&mut self, node: NodeIx, prev: NodeIx) {
        let old = self.nodes[node as usize].prev;
        if old != prev {
            self.push_undo(UndoOp::SetPrev { node, old });
            self.nodes[node as usize].prev = prev;
        }
    }

    fn set_next(&mut self, node: NodeIx, next: NodeIx) {
        let old = self.nodes[node as usize].next;
        if old != next {
            self.push_undo(UndoOp::SetNext { node, old });
            self.nodes[node as usize].next = next;
        }
    }

    fn set_rule_ref_count(&mut self, rule: RuleId, ref_count: u32) {
        let old = self.rules[rule as usize].ref_count;
        if old != ref_count {
            self.push_undo(UndoOp::SetRuleRefCount { rule, old });
            self.rules[rule as usize].ref_count = ref_count;
        }
    }

    fn set_rule_active(&mut self, rule: RuleId, active: bool) {
        let old = self.rules[rule as usize].active;
        if old != active {
            self.push_undo(UndoOp::SetRuleActive { rule, old });
            self.rules[rule as usize].active = active;
        }
    }

    fn set_digram(&mut self, key: u64, value: Option<NodeIx>) {
        let old = self.digrams.get(&key).copied();
        if old == value {
            return;
        }
        self.push_undo(UndoOp::SetDigram { key, old });
        if let Some(node) = value {
            self.digrams.insert(key, node);
        } else {
            self.digrams.remove(&key);
        }
    }

    fn record_context_followers(&mut self, key: &[u8], new_state: Option<ContextFollowers>) {
        let boxed: Box<[u8]> = key.to_vec().into_boxed_slice();
        let old = self.followers.get(boxed.as_ref()).cloned();
        if old == new_state {
            return;
        }
        self.push_undo(UndoOp::SetContextFollowers {
            key: boxed.clone(),
            old,
        });
        if let Some(state) = new_state {
            self.followers.insert(boxed, state);
        } else {
            self.followers.remove(boxed.as_ref());
        }
    }

    fn record_unigram(&mut self, symbol: u8, next_count: u64, next_total: u64) {
        let old_count = self.unigram[symbol as usize];
        let old_total = self.unigram_total;
        if old_count == next_count && old_total == next_total {
            return;
        }
        self.push_undo(UndoOp::SetUnigramSymbol {
            symbol,
            old_count,
            old_total,
        });
        self.unigram[symbol as usize] = next_count;
        self.unigram_total = next_total;
    }

    fn record_committed_raw_tail(&mut self, symbol: u8) {
        let mut next = self.committed_raw_tail.clone();
        next.push(symbol);
        if next.len() > RAW_FALLBACK_MAX {
            let drain = next.len() - RAW_FALLBACK_MAX;
            next.drain(0..drain);
        }
        if next != self.committed_raw_tail {
            self.push_undo(UndoOp::SetCommittedRawTail {
                old: self.committed_raw_tail.clone(),
            });
            self.committed_raw_tail = next;
        }
    }

    fn record_frozen_raw_tail_inner(&mut self, next: Vec<u8>) {
        if next != self.frozen_raw_tail {
            self.push_undo(UndoOp::SetFrozenRawTail {
                old: self.frozen_raw_tail.clone(),
            });
            self.frozen_raw_tail = next;
        }
    }

    fn ensure_pdf(&mut self) {
        if self.pdf_valid {
            return;
        }

        let denom = (self.unigram_total as f64) + 128.0;
        for (idx, slot) in self.pdf.iter_mut().enumerate() {
            *slot = ((self.unigram[idx] as f64) + 0.5) / denom;
        }

        let contexts = self.current_contexts();
        let mut next = [0.0; 256];
        for context in contexts {
            let Some(stats) = self.followers.get(context.as_slice()) else {
                continue;
            };
            let distinct = stats.distinct();
            if stats.total == 0 || distinct == 0 {
                continue;
            }
            let total = stats.total as f64;
            let types = distinct as f64;
            let escape = types / (total + types);
            for i in 0..256 {
                next[i] = self.pdf[i] * escape;
            }
            for &(symbol, count) in &stats.counts {
                next[symbol as usize] += (count as f64) / (total + types);
            }
            self.pdf.copy_from_slice(&next);
        }

        normalize_pdf(&mut self.pdf);
        self.pdf_valid = true;
    }

    fn current_contexts(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::<Vec<u8>>::new();
        let raw_tail = self.effective_raw_tail();
        for len in 1..=raw_tail.len().min(RAW_FALLBACK_MAX) {
            let ctx = raw_tail[raw_tail.len() - len..].to_vec();
            if !out.iter().any(|existing| existing == &ctx) {
                out.push(ctx);
            }
        }

        let mut rule_chain = Vec::<RuleId>::new();
        rule_chain.push(0);
        let mut current = 0u32;
        loop {
            let guard = self.rules[current as usize].guard;
            let last = self.nodes[guard as usize].prev;
            if last == guard {
                break;
            }
            match self.nodes[last as usize].data {
                NodeData::Sym(Symbol::NonTerminal(child)) if self.rules[child as usize].active => {
                    rule_chain.push(child);
                    current = child;
                }
                _ => break,
            }
        }

        for &rule_id in &rule_chain {
            let ctx = self.rule_tail_bytes(rule_id, self.context_bytes);
            if !ctx.is_empty() && !out.iter().any(|existing| existing == &ctx) {
                out.push(ctx);
            }
        }

        out
    }

    fn effective_raw_tail(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(RAW_FALLBACK_MAX);
        let total = self.committed_raw_tail.len() + self.frozen_raw_tail.len();
        let keep_from = total.saturating_sub(RAW_FALLBACK_MAX);
        for (idx, &byte) in self
            .committed_raw_tail
            .iter()
            .chain(self.frozen_raw_tail.iter())
            .enumerate()
        {
            if idx >= keep_from {
                out.push(byte);
            }
        }
        out
    }

    fn observe_symbol_in_stats(&mut self, symbol: u8) {
        let contexts = self.current_contexts();
        for context in contexts {
            let mut state = self
                .followers
                .get(context.as_slice())
                .cloned()
                .unwrap_or_default();
            state.observe(symbol);
            self.record_context_followers(&context, Some(state));
        }
        let old_count = self.unigram[symbol as usize];
        let old_total = self.unigram_total;
        self.record_unigram(symbol, old_count + 1, old_total + 1);
    }

    fn append_terminal(&mut self, byte: u8) {
        let last = self.last_node_of_rule(0);
        let _ = self.insert_after(last, Symbol::Terminal(byte));
        let _ = self.check(last);
    }

    fn alloc_node(&mut self, data: NodeData) -> NodeIx {
        let idx = self.nodes.len() as NodeIx;
        let (prev, next) = match data {
            NodeData::Guard(_) => (idx, idx),
            NodeData::Sym(_) => (self.dummy, self.dummy),
        };
        self.nodes.push(Node { prev, next, data });
        idx
    }

    fn new_rule(&mut self) -> RuleId {
        let id = self.rules.len() as RuleId;
        let guard = self.alloc_node(NodeData::Guard(id));
        self.rules.push(Rule {
            guard,
            ref_count: 0,
            active: true,
        });
        id
    }

    fn is_guard(&self, node: NodeIx) -> bool {
        matches!(self.nodes[node as usize].data, NodeData::Guard(_))
    }

    fn symbol_of(&self, node: NodeIx) -> Symbol {
        match self.nodes[node as usize].data {
            NodeData::Sym(symbol) => symbol,
            NodeData::Guard(_) => panic!("node_symbol called on guard node"),
        }
    }

    fn symbol_maybe(&self, node: NodeIx) -> Option<Symbol> {
        match self.nodes[node as usize].data {
            NodeData::Sym(symbol) => Some(symbol),
            NodeData::Guard(_) => None,
        }
    }

    fn guard_rule(&self, node: NodeIx) -> Option<RuleId> {
        match self.nodes[node as usize].data {
            NodeData::Guard(rule) => Some(rule),
            NodeData::Sym(_) => None,
        }
    }

    fn first_node_of_rule(&self, rule: RuleId) -> NodeIx {
        let guard = self.rules[rule as usize].guard;
        self.nodes[guard as usize].next
    }

    fn last_node_of_rule(&self, rule: RuleId) -> NodeIx {
        let guard = self.rules[rule as usize].guard;
        self.nodes[guard as usize].prev
    }

    fn encode_symbol(symbol: Symbol) -> u32 {
        match symbol {
            Symbol::Terminal(byte) => byte as u32,
            Symbol::NonTerminal(rule) => 256u32.wrapping_add(rule),
        }
    }

    fn digram_key_from_symbols(left: Symbol, right: Symbol) -> u64 {
        ((Self::encode_symbol(left) as u64) << 32) | (Self::encode_symbol(right) as u64)
    }

    fn digram_key_at(&self, node: NodeIx) -> Option<u64> {
        if self.is_guard(node) {
            return None;
        }
        let next = self.nodes[node as usize].next;
        if self.is_guard(next) {
            return None;
        }
        Some(Self::digram_key_from_symbols(
            self.symbol_of(node),
            self.symbol_of(next),
        ))
    }

    fn link(&mut self, left: NodeIx, right: NodeIx) {
        let left_prev = self.nodes[left as usize].prev;
        let left_next = self.nodes[left as usize].next;
        let right_prev = self.nodes[right as usize].prev;
        let right_next = self.nodes[right as usize].next;

        if !self.is_guard(left_next) {
            self.delete_digram(left);

            match (
                self.symbol_maybe(right_prev),
                self.symbol_maybe(right),
                self.symbol_maybe(right_next),
            ) {
                (Some(sym1), Some(sym2), Some(sym3)) if sym1 == sym2 && sym2 == sym3 => {
                    self.set_digram(Self::digram_key_from_symbols(sym2, sym3), Some(right));
                }
                _ => {}
            }

            match (
                self.symbol_maybe(left_prev),
                self.symbol_maybe(left),
                self.symbol_maybe(left_next),
            ) {
                (Some(sym1), Some(sym2), Some(sym3)) if sym1 == sym2 && sym2 == sym3 => {
                    self.set_digram(Self::digram_key_from_symbols(sym1, sym2), Some(left_prev));
                }
                _ => {}
            }
        }

        self.set_next(left, right);
        self.set_prev(right, left);
    }

    fn insert_after(&mut self, node: NodeIx, symbol: Symbol) -> NodeIx {
        let new_node = self.alloc_node(NodeData::Sym(symbol));
        let next = self.nodes[node as usize].next;
        self.link(new_node, next);
        self.link(node, new_node);
        if let Symbol::NonTerminal(rule) = symbol {
            let next_count = self.rules[rule as usize].ref_count.saturating_add(1);
            self.set_rule_ref_count(rule, next_count);
        }
        new_node
    }

    fn delete_digram(&mut self, node: NodeIx) {
        let Some(key) = self.digram_key_at(node) else {
            return;
        };
        match self.digrams.get(&key).copied() {
            Some(existing) if existing != node => {}
            _ => self.set_digram(key, None),
        }
    }

    fn check(&mut self, node: NodeIx) -> bool {
        let Some(key) = self.digram_key_at(node) else {
            return false;
        };
        let existing = self.digrams.get(&key).copied();
        match existing {
            None => {
                self.set_digram(key, Some(node));
                false
            }
            Some(other) => {
                let other_next = self.nodes[other as usize].next;
                let node_next = self.nodes[node as usize].next;
                if node == other_next || other == node_next {
                    false
                } else {
                    self.match_nodes(node, other);
                    true
                }
            }
        }
    }

    fn match_nodes(&mut self, ss: NodeIx, m: NodeIx) {
        let m_prev = self.nodes[m as usize].prev;
        let m_next = self.nodes[m as usize].next;
        let m_next_next = self.nodes[m_next as usize].next;

        let rule = if let Some(rule) = self.guard_rule(m_prev) {
            if rule != 0 && self.is_guard(m_next_next) {
                self.substitute(ss, rule);
                rule
            } else {
                let rule = self.new_rule();
                let ss2 = self.nodes[ss as usize].next;
                let last = self.last_node_of_rule(rule);
                let node1 = self.insert_after(last, self.symbol_of(ss));
                let node2 = self.insert_after(node1, self.symbol_of(ss2));
                self.substitute(m, rule);
                self.substitute(ss, rule);
                self.set_digram(
                    Self::digram_key_from_symbols(self.symbol_of(node1), self.symbol_of(node2)),
                    Some(node1),
                );
                rule
            }
        } else {
            let rule = self.new_rule();
            let ss2 = self.nodes[ss as usize].next;
            let last = self.last_node_of_rule(rule);
            let node1 = self.insert_after(last, self.symbol_of(ss));
            let node2 = self.insert_after(node1, self.symbol_of(ss2));
            self.substitute(m, rule);
            self.substitute(ss, rule);
            self.set_digram(
                Self::digram_key_from_symbols(self.symbol_of(node1), self.symbol_of(node2)),
                Some(node1),
            );
            rule
        };

        let first = self.first_node_of_rule(rule);
        if let Symbol::NonTerminal(child) = self.symbol_of(first) {
            if self.rules[child as usize].ref_count == 1 {
                self.expand(first, child);
            }
        }
    }

    fn delete_node(&mut self, node: NodeIx) {
        debug_assert!(!self.is_guard(node), "delete_node called on guard");
        let prev = self.nodes[node as usize].prev;
        let next = self.nodes[node as usize].next;
        self.link(prev, next);
        self.delete_digram(node);
        if let Symbol::NonTerminal(rule) = self.symbol_of(node) {
            let next_count = self.rules[rule as usize].ref_count.saturating_sub(1);
            self.set_rule_ref_count(rule, next_count);
        }
    }

    fn substitute(&mut self, node: NodeIx, rule: RuleId) {
        let prev = self.nodes[node as usize].prev;
        let first = self.nodes[prev as usize].next;
        debug_assert!(!self.is_guard(first), "substitute first guard");
        self.delete_node(first);
        let second = self.nodes[prev as usize].next;
        debug_assert!(!self.is_guard(second), "substitute second guard");
        self.delete_node(second);
        let _ = self.insert_after(prev, Symbol::NonTerminal(rule));
        if !self.check(prev) {
            let next = self.nodes[prev as usize].next;
            let _ = self.check(next);
        }
    }

    fn expand(&mut self, node: NodeIx, rule: RuleId) {
        let left = self.nodes[node as usize].prev;
        let right = self.nodes[node as usize].next;
        self.delete_node(node);

        let first = self.first_node_of_rule(rule);
        let last = self.last_node_of_rule(rule);
        self.link(left, first);
        self.link(last, right);

        let next = self.nodes[last as usize].next;
        self.set_digram(
            Self::digram_key_from_symbols(self.symbol_of(last), self.symbol_of(next)),
            Some(last),
        );

        let guard = self.rules[rule as usize].guard;
        self.link(guard, guard);
        self.set_rule_active(rule, false);
    }

    fn decode_rule(&self, rule: RuleId, out: &mut Vec<u8>) {
        let guard = self.rules[rule as usize].guard;
        let mut node = self.nodes[guard as usize].next;
        while node != guard {
            match self.nodes[node as usize].data {
                NodeData::Guard(_) => unreachable!("guard encountered in active rule body"),
                NodeData::Sym(Symbol::Terminal(byte)) => out.push(byte),
                NodeData::Sym(Symbol::NonTerminal(child)) => self.decode_rule(child, out),
            }
            node = self.nodes[node as usize].next;
        }
    }

    fn rule_tail_bytes(&self, rule: RuleId, limit: usize) -> Vec<u8> {
        let mut rev = Vec::with_capacity(limit);
        self.collect_rule_tail_rev(rule, limit, &mut rev);
        rev.reverse();
        rev
    }

    fn collect_rule_tail_rev(&self, rule: RuleId, limit: usize, out_rev: &mut Vec<u8>) {
        if out_rev.len() >= limit {
            return;
        }
        let guard = self.rules[rule as usize].guard;
        let mut node = self.nodes[guard as usize].prev;
        while node != guard && out_rev.len() < limit {
            match self.nodes[node as usize].data {
                NodeData::Guard(_) => break,
                NodeData::Sym(Symbol::Terminal(byte)) => out_rev.push(byte),
                NodeData::Sym(Symbol::NonTerminal(child)) => {
                    self.collect_rule_tail_rev(child, limit, out_rev);
                }
            }
            node = self.nodes[node as usize].prev;
        }
    }

    #[cfg(test)]
    fn active_rule_ids(&self) -> Vec<RuleId> {
        self.rules
            .iter()
            .enumerate()
            .filter_map(|(idx, rule)| rule.active.then_some(idx as RuleId))
            .collect()
    }

    #[cfg(test)]
    fn rule_body_symbols(&self, rule: RuleId) -> Vec<Symbol> {
        let guard = self.rules[rule as usize].guard;
        let mut out = Vec::new();
        let mut node = self.nodes[guard as usize].next;
        while node != guard {
            out.push(self.symbol_of(node));
            node = self.nodes[node as usize].next;
        }
        out
    }

    #[cfg(test)]
    fn validate_rule_shapes(&self) -> Result<(), String> {
        for rule in self.active_rule_ids() {
            if rule == 0 {
                continue;
            }
            let len = self.rule_body_symbols(rule).len();
            if len < 2 {
                return Err(format!("rule {rule} has rhs length {len}, expected >= 2"));
            }
            if self.rules[rule as usize].ref_count < 2 {
                return Err(format!(
                    "rule {rule} has utility {}, expected >= 2",
                    self.rules[rule as usize].ref_count
                ));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn validate_rule_refcounts(&self) -> Result<(), String> {
        let mut counts = vec![0u32; self.rules.len()];
        for rule in self.active_rule_ids() {
            for sym in self.rule_body_symbols(rule) {
                if let Symbol::NonTerminal(child) = sym {
                    counts[child as usize] += 1;
                }
            }
        }
        for rule in self.active_rule_ids() {
            if rule == 0 {
                continue;
            }
            let actual = self.rules[rule as usize].ref_count;
            let expected = counts[rule as usize];
            if actual != expected {
                return Err(format!(
                    "rule {rule} ref_count mismatch: actual={actual}, expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn validate_digram_uniqueness(&self) -> Result<(), String> {
        let mut seen = AHashMap::<u64, NodeIx>::new();
        for rule in self.active_rule_ids() {
            let guard = self.rules[rule as usize].guard;
            let mut node = self.nodes[guard as usize].next;
            while node != guard {
                let next = self.nodes[node as usize].next;
                if next == guard {
                    break;
                }
                let key = Self::digram_key_from_symbols(self.symbol_of(node), self.symbol_of(next));
                if let Some(&other) = seen.get(&key) {
                    let other_next = self.nodes[other as usize].next;
                    if other_next != node && self.nodes[node as usize].next != other {
                        return Err(format!(
                            "duplicate non-overlapping digram for key {key}: {other} and {node}"
                        ));
                    }
                } else {
                    seen.insert(key, node);
                }
                node = next;
            }
        }
        Ok(())
    }
}

fn normalize_pdf(pdf: &mut [f64; 256]) {
    let mut sum = 0.0f64;
    for value in pdf.iter_mut() {
        *value = if value.is_finite() {
            (*value).max(PDF_MIN)
        } else {
            PDF_MIN
        };
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        pdf.fill(1.0 / 256.0);
        return;
    }
    let inv = 1.0 / sum;
    for value in pdf.iter_mut() {
        *value *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train_model(data: &[u8], context_bytes: usize) -> SequiturModel {
        let mut model = SequiturModel::new(context_bytes);
        for &byte in data {
            model.update(byte);
        }
        model
    }

    #[test]
    fn sequitur_roundtrips_and_preserves_invariants() {
        let data = b"abcabcabcabcabc";
        let model = train_model(data, 64);
        assert_eq!(model.decode(), data);
        model.validate_invariants().unwrap();
    }

    #[test]
    fn sequitur_invariants_hold_after_each_step() {
        let mut model = SequiturModel::new(32);
        for &byte in b"abracadabra abracadabra" {
            model.update(byte);
            model.validate_invariants().unwrap();
        }
    }

    #[test]
    fn sequitur_canonical_grammar_is_deterministic() {
        let model_a = train_model(b"abcabcabcabc", 64);
        let model_b = train_model(b"abcabcabcabc", 64);
        assert_eq!(model_a.canonical_grammar(), model_b.canonical_grammar());
    }

    #[test]
    fn sequitur_pdf_is_normalized() {
        let mut model = train_model(b"abababababa", 32);
        let mut pdf = [0.0; 256];
        model.fill_pdf(&mut pdf);
        let sum: f64 = pdf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
        assert!(pdf.iter().all(|p| p.is_finite() && *p > 0.0));
    }

    #[test]
    fn frozen_updates_do_not_mutate_learned_distribution_after_reset() {
        let mut model = train_model(b"banana banana banana", 32);
        let mut before = [0.0; 256];
        model.fill_pdf(&mut before);
        let grammar_before = model.canonical_grammar();
        let followers_before = model.followers.clone();
        model.reset_frozen();
        for &byte in b"ZZZZ" {
            model.update_frozen(byte);
        }
        assert_eq!(grammar_before, model.canonical_grammar());
        assert_eq!(followers_before, model.followers);
        model.reset_frozen();
        let mut after = [0.0; 256];
        model.fill_pdf(&mut after);
        assert_eq!(before, after);
    }

    #[test]
    fn checkpoint_restore_recovers_exact_state() {
        let mut model = train_model(b"mississippi", 32);
        let checkpoint = model.checkpoint();
        let grammar_before = model.canonical_grammar();
        let pdf_before = {
            let mut pdf = [0.0; 256];
            model.fill_pdf(&mut pdf);
            pdf
        };
        for &byte in b" river" {
            model.update(byte);
        }
        model.restore(&checkpoint);
        model.clear_checkpoints();
        assert_eq!(grammar_before, model.canonical_grammar());
        let mut pdf_after = [0.0; 256];
        model.fill_pdf(&mut pdf_after);
        assert_eq!(pdf_before, pdf_after);
        model.validate_invariants().unwrap();
    }

    #[test]
    fn sequitur_repetitive_binary_inputs_preserve_invariants() {
        for data in [
            b"\x00\x00\x00\x00\x00\x00\x00\x00".as_slice(),
            b"\x00\x01\x00\x01\x00\x01\x00\x01".as_slice(),
        ] {
            let mut model = SequiturModel::new(32);
            for (idx, &byte) in data.iter().enumerate() {
                model.update(byte);
                if let Err(err) = model.validate_invariants() {
                    let active = model
                        .active_rule_ids()
                        .into_iter()
                        .map(|rule| {
                            (
                                rule,
                                model.rules[rule as usize].ref_count,
                                model.rule_body_symbols(rule),
                            )
                        })
                        .collect::<Vec<_>>();
                    panic!(
                        "invariants failed at step {idx} for {:?}: {err}; active={active:?}",
                        data
                    );
                }
            }
        }
    }
}
