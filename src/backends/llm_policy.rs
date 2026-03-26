use anyhow::{Context, Result, bail};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
/// Position expression used in policy schedules.
pub enum PositionExpr {
    /// Absolute byte/token offset.
    Bytes(u64),
    /// Percentage of the known total stream length.
    Percent(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Optimizer family selected by a train action.
pub enum OptimizerKind {
    /// Stochastic gradient descent.
    Sgd,
    /// Adam optimizer.
    Adam,
}

#[derive(Clone, Debug, PartialEq)]
/// Hyper-parameters attached to a training action.
pub struct OptimizerHyperParams {
    /// Learning rate (clamped non-negative by parser).
    pub lr: f32,
    /// Apply one update every `stride` eligible steps.
    pub stride: usize,
    /// Truncated backprop length for full-parameter training.
    pub bptt: usize,
    /// Optional gradient clipping threshold (`0` disables clipping).
    pub clip: f32,
    /// Momentum used by SGD-style updates.
    pub momentum: f32,
}

impl Default for OptimizerHyperParams {
    fn default() -> Self {
        Self {
            lr: 0.001,
            stride: 1,
            bptt: 1,
            clip: 0.0,
            momentum: 0.9,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Canonical set of train scopes (`all`, `none`, or sorted named scopes).
pub struct TrainScopeSet {
    /// If true, every allowed scope is enabled.
    pub all: bool,
    /// Sorted explicit scope names used when `all == false`.
    pub names: Vec<String>,
}

impl TrainScopeSet {
    /// Enable all scopes.
    pub fn all() -> Self {
        Self {
            all: true,
            names: Vec::new(),
        }
    }

    /// Disable all scopes.
    pub fn none() -> Self {
        Self {
            all: false,
            names: Vec::new(),
        }
    }

    /// Returns whether `name` is enabled by this set.
    pub fn contains(&self, name: &str) -> bool {
        self.all
            || self
                .names
                .binary_search_by(|s| s.as_str().cmp(name))
                .is_ok()
    }

    /// Returns `true` when the set explicitly enables no scopes.
    pub fn is_none(&self) -> bool {
        !self.all && self.names.is_empty()
    }

    /// Parse a scope expression against an allow-list.
    ///
    /// Supports `all`, `none`, or a `+`/`|`/`/` separated list.
    pub fn parse(value: &str, allowed_scopes: &[&str]) -> Result<Self> {
        let v = value.trim().to_ascii_lowercase();
        if v.is_empty() {
            bail!("empty train scope");
        }
        if v == "all" {
            return Ok(Self::all());
        }
        if v == "none" {
            return Ok(Self::none());
        }

        let mut out = BTreeSet::<String>::new();
        for tok in v.split(['+', '|', '/']) {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            if !allowed_scopes.contains(&t) {
                bail!(
                    "unknown train scope '{t}', allowed: {}",
                    allowed_scopes.join(",")
                );
            }
            if t == "all" || t == "none" {
                bail!("scope list cannot mix '{t}' with named scopes");
            }
            out.insert(t.to_string());
        }

        Ok(Self {
            all: false,
            names: out.into_iter().collect(),
        })
    }

    /// Canonical string form (`all`, `none`, or `a+b+c`).
    pub fn canonical(&self) -> String {
        if self.all {
            return "all".to_string();
        }
        if self.names.is_empty() {
            return "none".to_string();
        }
        self.names.join("+")
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Concrete training directive.
pub struct TrainAction {
    /// Parameter subsets to update.
    pub scope: TrainScopeSet,
    /// Optimizer family.
    pub optimizer: OptimizerKind,
    /// Optimizer hyper-parameters.
    pub hyper: OptimizerHyperParams,
}

#[derive(Clone, Debug, PartialEq)]
/// Action emitted by the policy at each stream position.
pub enum PolicyAction {
    /// Inference-only step (no adaptation).
    Infer,
    /// Training step with provided parameters.
    Train(TrainAction),
}

#[derive(Clone, Debug, PartialEq)]
/// Interval schedule rule (`start..end:action`).
pub struct PolicyRule {
    /// Inclusive interval start.
    pub start: PositionExpr,
    /// Exclusive interval end.
    pub end: PositionExpr,
    /// Action applied in the interval.
    pub action: PolicyAction,
}

#[derive(Clone, Debug, PartialEq)]
/// One segment inside a repeating pattern.
pub struct RepeatSegment {
    /// Segment duration.
    pub span: PositionExpr,
    /// Action active for this segment.
    pub action: PolicyAction,
}

#[derive(Clone, Debug, PartialEq)]
/// Repeat schedule rule with cycle period and inner pattern.
pub struct RepeatRule {
    /// Inclusive active-start boundary.
    pub start: PositionExpr,
    /// Exclusive active-end boundary.
    pub end: PositionExpr,
    /// Repeat cycle length.
    pub period: PositionExpr,
    /// Ordered segments inside one cycle.
    pub pattern: Vec<RepeatSegment>,
}

#[derive(Clone, Debug, PartialEq)]
/// Top-level schedule rule.
pub enum ScheduleRule {
    /// Single contiguous interval.
    Interval(PolicyRule),
    /// Periodic repeating schedule.
    Repeat(RepeatRule),
}

#[derive(Clone, Debug, PartialEq)]
/// Parsed policy specification for online LLM adaptation.
pub struct LlmPolicy {
    /// Optional initial weights/source to load before running schedule.
    pub load_from: Option<PathBuf>,
    /// Ordered schedule rules.
    pub schedule: Vec<ScheduleRule>,
}

/// Returns `true` if any schedule branch can emit a training action.
pub fn policy_can_train(policy: &LlmPolicy) -> bool {
    for rule in &policy.schedule {
        match rule {
            ScheduleRule::Interval(interval) => {
                if matches!(interval.action, PolicyAction::Train(_)) {
                    return true;
                }
            }
            ScheduleRule::Repeat(repeat) => {
                if repeat
                    .pattern
                    .iter()
                    .any(|seg| matches!(seg.action, PolicyAction::Train(_)))
                {
                    return true;
                }
            }
        }
    }
    false
}

#[derive(Clone, Debug)]
struct CompiledPatternSegment {
    end: u64,
    action: PolicyAction,
}

#[derive(Clone, Debug)]
enum CompiledScheduleRule {
    Interval {
        start: u64,
        end: u64,
        action: PolicyAction,
    },
    Repeat {
        start: u64,
        end: u64,
        period: u64,
        pattern_total: u64,
        pattern: Vec<CompiledPatternSegment>,
    },
}

#[derive(Clone, Debug)]
/// Compiled, position-resolved policy ready for runtime evaluation.
pub struct CompiledPolicy {
    rules: Vec<CompiledScheduleRule>,
}

impl CompiledPolicy {
    /// Resolve the action at absolute position `pos`.
    pub fn action_at(&self, pos: u64) -> PolicyAction {
        for rule in &self.rules {
            match rule {
                CompiledScheduleRule::Interval { start, end, action }
                    if pos >= *start && pos < *end =>
                {
                    return action.clone();
                }
                CompiledScheduleRule::Repeat {
                    start,
                    end,
                    period,
                    pattern_total,
                    pattern,
                } if pos >= *start && pos < *end => {
                    let phase = (pos - *start) % *period;
                    if phase >= *pattern_total {
                        return PolicyAction::Infer;
                    }
                    for seg in pattern {
                        if phase < seg.end {
                            return seg.action.clone();
                        }
                    }
                    return PolicyAction::Infer;
                }
                _ => {}
            }
        }
        PolicyAction::Infer
    }
}

#[derive(Clone, Debug)]
/// Stateful cursor over a [`CompiledPolicy`].
pub struct PolicyRuntime {
    compiled: CompiledPolicy,
    cursor: u64,
}

impl PolicyRuntime {
    /// Create a runtime positioned at cursor `0`.
    pub fn new(compiled: CompiledPolicy) -> Self {
        Self {
            compiled,
            cursor: 0,
        }
    }

    #[inline]
    /// Current cursor position.
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    #[inline]
    /// Set cursor position directly.
    pub fn set_cursor(&mut self, cursor: u64) {
        self.cursor = cursor;
    }

    #[inline]
    /// Peek current action without advancing cursor.
    pub fn peek_action(&self) -> PolicyAction {
        self.compiled.action_at(self.cursor)
    }

    #[inline]
    /// Return current action and advance cursor by one.
    pub fn next_action(&mut self) -> PolicyAction {
        let action = self.compiled.action_at(self.cursor);
        self.cursor = self.cursor.saturating_add(1);
        action
    }
}

/// Split `method` into base method segment and optional `policy:...` segment.
pub fn split_method_policy_segments(method: &str) -> Result<(String, Option<String>)> {
    let trimmed = method.trim();
    if trimmed.is_empty() {
        bail!("empty method string");
    }
    let mut iter = trimmed.split(';');
    let base = iter.next().unwrap_or_default().trim().to_string();
    if base.is_empty() {
        bail!("method is missing cfg/file segment");
    }

    let mut policy = None::<String>;
    for seg in iter {
        let s = seg.trim();
        if s.is_empty() {
            continue;
        }
        if let Some(rest) = s.strip_prefix("policy:") {
            if policy.is_some() {
                bail!("duplicate policy segment in method string");
            }
            policy = Some(rest.trim().to_string());
            continue;
        }
        bail!("unknown method segment '{s}', expected 'policy:...'");
    }

    Ok((base, policy))
}

/// Parse a `policy:...` string into [`LlmPolicy`].
pub fn parse_policy_segment(policy_segment: &str, allowed_scopes: &[&str]) -> Result<LlmPolicy> {
    let body = policy_segment
        .trim()
        .strip_prefix("policy:")
        .unwrap_or(policy_segment.trim())
        .trim();
    if body.is_empty() {
        bail!("empty policy segment");
    }

    let mut load_from = None::<PathBuf>;
    let mut schedule_raw = None::<String>;

    for entry in split_top_level(body, ',')? {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (k, v) = entry
            .split_once('=')
            .with_context(|| format!("invalid policy key/value pair '{entry}'"))?;
        let key = k.trim().to_ascii_lowercase();
        let val = v.trim();
        match key.as_str() {
            "load_from" => {
                if val.is_empty() {
                    bail!("policy load_from must not be empty");
                }
                load_from = Some(PathBuf::from(val));
            }
            "schedule" => {
                if val.is_empty() {
                    bail!("policy schedule must not be empty");
                }
                schedule_raw = Some(val.to_string());
            }
            other => bail!("unknown policy key '{other}'"),
        }
    }

    let schedule_raw =
        schedule_raw.ok_or_else(|| anyhow::anyhow!("policy requires schedule=..."))?;
    let mut schedule = Vec::<ScheduleRule>::new();
    for token in split_top_level(&schedule_raw, '|')? {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with("repeat(") {
            schedule.push(ScheduleRule::Repeat(parse_repeat_rule(t, allowed_scopes)?));
        } else {
            schedule.push(ScheduleRule::Interval(parse_interval_rule(
                t,
                allowed_scopes,
            )?));
        }
    }

    if schedule.is_empty() {
        bail!("policy schedule must contain at least one rule");
    }

    Ok(LlmPolicy {
        load_from,
        schedule,
    })
}

impl LlmPolicy {
    /// Compile policy boundaries/spans using optional known total symbol count.
    pub fn compile(&self, total_symbols: Option<u64>) -> Result<CompiledPolicy> {
        let mut out = Vec::<CompiledScheduleRule>::with_capacity(self.schedule.len());
        for rule in &self.schedule {
            match rule {
                ScheduleRule::Interval(r) => {
                    let start = resolve_boundary(&r.start, total_symbols)?;
                    let end = resolve_boundary(&r.end, total_symbols)?;
                    if end <= start {
                        bail!("invalid interval with end <= start ({start}..{end})");
                    }
                    out.push(CompiledScheduleRule::Interval {
                        start,
                        end,
                        action: r.action.clone(),
                    });
                }
                ScheduleRule::Repeat(r) => {
                    let start = resolve_boundary(&r.start, total_symbols)?;
                    let end = resolve_boundary(&r.end, total_symbols)?;
                    if end <= start {
                        bail!("invalid repeat interval with end <= start ({start}..{end})");
                    }
                    let period = resolve_span(&r.period, total_symbols)?;
                    if period == 0 {
                        bail!("repeat period must be > 0");
                    }

                    let mut pattern = Vec::<CompiledPatternSegment>::with_capacity(r.pattern.len());
                    let mut acc = 0u64;
                    for seg in &r.pattern {
                        let span = resolve_span(&seg.span, total_symbols)?;
                        if span == 0 {
                            bail!("repeat pattern segment span must be > 0");
                        }
                        acc = acc.saturating_add(span);
                        pattern.push(CompiledPatternSegment {
                            end: acc,
                            action: seg.action.clone(),
                        });
                    }
                    if pattern.is_empty() {
                        bail!("repeat pattern must contain at least one segment");
                    }

                    out.push(CompiledScheduleRule::Repeat {
                        start,
                        end,
                        period,
                        pattern_total: acc,
                        pattern,
                    });
                }
            }
        }
        Ok(CompiledPolicy { rules: out })
    }

    /// Serialize back to canonical `load_from=...,schedule=...` form.
    pub fn canonical(&self) -> String {
        let mut out = String::new();
        if let Some(path) = &self.load_from {
            out.push_str("load_from=");
            out.push_str(&path.display().to_string());
            out.push(',');
        }
        out.push_str("schedule=");
        for (idx, r) in self.schedule.iter().enumerate() {
            if idx > 0 {
                out.push('|');
            }
            match r {
                ScheduleRule::Interval(i) => {
                    out.push_str(&position_to_string(&i.start));
                    out.push_str("..");
                    out.push_str(&position_to_string(&i.end));
                    out.push(':');
                    out.push_str(&action_to_string(&i.action));
                }
                ScheduleRule::Repeat(rep) => {
                    out.push_str("repeat(");
                    out.push_str(&position_to_string(&rep.start));
                    out.push_str("..");
                    out.push_str(&position_to_string(&rep.end));
                    out.push_str(",period=");
                    out.push_str(&position_to_string(&rep.period));
                    out.push_str(",pattern=");
                    for (j, seg) in rep.pattern.iter().enumerate() {
                        if j > 0 {
                            out.push('+');
                        }
                        out.push_str(&position_to_string(&seg.span));
                        out.push(':');
                        out.push_str(&action_to_string(&seg.action));
                    }
                    out.push(')');
                }
            }
        }
        out
    }
}

fn parse_interval_rule(token: &str, allowed_scopes: &[&str]) -> Result<PolicyRule> {
    let (range, action_s) = token.split_once(':').with_context(|| {
        format!("invalid schedule rule '{token}', expected <start>..<end>:<action>")
    })?;
    let (start, end) = parse_range(range)?;
    let action = parse_action(action_s, allowed_scopes)?;
    Ok(PolicyRule { start, end, action })
}

fn parse_repeat_rule(token: &str, allowed_scopes: &[&str]) -> Result<RepeatRule> {
    let inner = token
        .strip_prefix("repeat(")
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "invalid repeat rule '{token}', expected repeat(<start>..<end>,period=...,pattern=...)"
            )
        })?;

    let args = split_top_level(inner, ',')?;
    if args.is_empty() {
        bail!("repeat rule is empty");
    }

    let (start, end) = parse_range(args[0].trim())?;
    let mut period = None::<PositionExpr>;
    let mut pattern = None::<Vec<RepeatSegment>>;

    for arg in args.into_iter().skip(1) {
        let arg = arg.trim();
        if arg.is_empty() {
            continue;
        }
        let (k, v) = arg
            .split_once('=')
            .with_context(|| format!("invalid repeat argument '{arg}'"))?;
        let key = k.trim().to_ascii_lowercase();
        let val = v.trim();
        match key.as_str() {
            "period" => period = Some(parse_position_expr(val)?),
            "pattern" => {
                let mut segs = Vec::<RepeatSegment>::new();
                for seg in split_top_level(val, '+')? {
                    let s = seg.trim();
                    if s.is_empty() {
                        continue;
                    }
                    let (span_s, action_s) = s
                        .split_once(':')
                        .with_context(|| format!("invalid repeat pattern segment '{s}'"))?;
                    segs.push(RepeatSegment {
                        span: parse_position_expr(span_s.trim())?,
                        action: parse_action(action_s.trim(), allowed_scopes)?,
                    });
                }
                pattern = Some(segs);
            }
            other => bail!("unknown repeat key '{other}'"),
        }
    }

    let period = period.ok_or_else(|| anyhow::anyhow!("repeat rule requires period=..."))?;
    let pattern = pattern.ok_or_else(|| anyhow::anyhow!("repeat rule requires pattern=..."))?;
    if pattern.is_empty() {
        bail!("repeat pattern must not be empty");
    }

    Ok(RepeatRule {
        start,
        end,
        period,
        pattern,
    })
}

fn parse_action(token: &str, allowed_scopes: &[&str]) -> Result<PolicyAction> {
    let t = token.trim();
    if t.eq_ignore_ascii_case("infer") {
        return Ok(PolicyAction::Infer);
    }

    if t.eq_ignore_ascii_case("train") {
        return Ok(PolicyAction::Train(TrainAction {
            scope: TrainScopeSet::all(),
            optimizer: OptimizerKind::Sgd,
            hyper: OptimizerHyperParams::default(),
        }));
    }

    let inner = t
        .strip_prefix("train(")
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| anyhow::anyhow!("invalid action '{token}', expected infer or train(...)"))?;

    let mut scope = TrainScopeSet::all();
    let mut optimizer = OptimizerKind::Sgd;
    let mut hyper = OptimizerHyperParams::default();

    for arg in split_top_level(inner, ',')? {
        let arg = arg.trim();
        if arg.is_empty() {
            continue;
        }
        let (k, v) = arg
            .split_once('=')
            .with_context(|| format!("invalid train argument '{arg}'"))?;
        let key = k.trim().to_ascii_lowercase();
        let val = v.trim();
        match key.as_str() {
            "scope" => scope = TrainScopeSet::parse(val, allowed_scopes)?,
            "opt" | "optimizer" => {
                optimizer = match val.to_ascii_lowercase().as_str() {
                    "sgd" => OptimizerKind::Sgd,
                    "adam" => OptimizerKind::Adam,
                    other => bail!("unknown optimizer '{other}'"),
                };
            }
            "lr" => {
                hyper.lr = val
                    .parse::<f32>()
                    .with_context(|| format!("invalid lr '{val}'"))?
                    .max(0.0)
            }
            "stride" => {
                hyper.stride = val
                    .parse::<usize>()
                    .with_context(|| format!("invalid stride '{val}'"))?
                    .max(1)
            }
            "bptt" => {
                hyper.bptt = val
                    .parse::<usize>()
                    .with_context(|| format!("invalid bptt '{val}'"))?
                    .max(1)
            }
            "clip" => {
                hyper.clip = val
                    .parse::<f32>()
                    .with_context(|| format!("invalid clip '{val}'"))?
                    .max(0.0)
            }
            "momentum" => {
                hyper.momentum = val
                    .parse::<f32>()
                    .with_context(|| format!("invalid momentum '{val}'"))?
            }
            other => bail!("unknown train argument key '{other}'"),
        }
    }

    Ok(PolicyAction::Train(TrainAction {
        scope,
        optimizer,
        hyper,
    }))
}

fn parse_range(range: &str) -> Result<(PositionExpr, PositionExpr)> {
    let (start, end) = range
        .split_once("..")
        .with_context(|| format!("invalid range '{range}', expected <start>..<end>"))?;
    Ok((parse_position_expr(start)?, parse_position_expr(end)?))
}

fn parse_position_expr(token: &str) -> Result<PositionExpr> {
    let t = token.trim();
    if let Some(pct_s) = t.strip_suffix('%') {
        let pct = pct_s
            .trim()
            .parse::<f64>()
            .with_context(|| format!("invalid percent position '{t}'"))?;
        if !(0.0..=100.0).contains(&pct) {
            bail!("percent position must be in [0,100], got {pct}");
        }
        return Ok(PositionExpr::Percent(pct));
    }
    let abs = t
        .parse::<u64>()
        .with_context(|| format!("invalid absolute position '{t}'"))?;
    Ok(PositionExpr::Bytes(abs))
}

fn resolve_boundary(expr: &PositionExpr, total_symbols: Option<u64>) -> Result<u64> {
    match expr {
        PositionExpr::Bytes(v) => Ok(match total_symbols {
            Some(total) => (*v).min(total),
            None => *v,
        }),
        PositionExpr::Percent(pct) => {
            let total = total_symbols.ok_or_else(|| {
                anyhow::anyhow!(
                    "percent policy boundary requires known total symbol count at runtime"
                )
            })?;
            let resolved = ((total as f64) * (*pct / 100.0)).floor() as u64;
            Ok(resolved.min(total))
        }
    }
}

fn resolve_span(expr: &PositionExpr, total_symbols: Option<u64>) -> Result<u64> {
    match expr {
        PositionExpr::Bytes(v) => Ok(*v),
        PositionExpr::Percent(pct) => {
            let total = total_symbols.ok_or_else(|| {
                anyhow::anyhow!("percent policy span requires known total symbol count at runtime")
            })?;
            Ok(((total as f64) * (*pct / 100.0)).floor() as u64)
        }
    }
}

fn split_top_level(input: &str, delim: char) -> Result<Vec<&str>> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (idx, ch) in input.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    bail!("unbalanced ')' in '{input}'");
                }
            }
            _ if ch == delim && depth == 0 => {
                parts.push(&input[start..idx]);
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    if depth != 0 {
        bail!("unbalanced '(' in '{input}'");
    }
    parts.push(&input[start..]);
    Ok(parts)
}

fn position_to_string(expr: &PositionExpr) -> String {
    match expr {
        PositionExpr::Bytes(v) => v.to_string(),
        PositionExpr::Percent(p) => {
            if p.fract() == 0.0 {
                format!("{}%", *p as i64)
            } else {
                format!("{p}%")
            }
        }
    }
}

fn action_to_string(action: &PolicyAction) -> String {
    match action {
        PolicyAction::Infer => "infer".to_string(),
        PolicyAction::Train(train) => {
            let opt = match train.optimizer {
                OptimizerKind::Sgd => "sgd",
                OptimizerKind::Adam => "adam",
            };
            format!(
                "train(scope={},opt={},lr={},stride={},bptt={},clip={},momentum={})",
                train.scope.canonical(),
                opt,
                train.hyper.lr,
                train.hyper.stride,
                train.hyper.bptt,
                train.hyper.clip,
                train.hyper.momentum,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RWKV_SCOPES: &[&str] = &[
        "embed",
        "pre_norm",
        "attn_norm",
        "ffn_norm",
        "attn",
        "ffn",
        "head",
        "bias",
        "all",
        "none",
    ];

    #[test]
    fn parse_policy_basic_and_compile() {
        let p = parse_policy_segment(
            "policy:schedule=0..10:infer|10..100%:train(scope=head+bias,opt=adam,lr=0.01,stride=2,bptt=4,clip=1.0,momentum=0.95)",
            RWKV_SCOPES,
        )
        .expect("policy");
        let c = p.compile(Some(100)).expect("compile");
        assert!(matches!(c.action_at(0), PolicyAction::Infer));
        match c.action_at(15) {
            PolicyAction::Train(t) => {
                assert!(t.scope.contains("head"));
                assert!(t.scope.contains("bias"));
                assert_eq!(t.hyper.stride, 2);
                assert_eq!(t.hyper.bptt, 4);
            }
            _ => panic!("expected train"),
        }
    }

    #[test]
    fn parse_repeat_policy() {
        let p = parse_policy_segment(
            "schedule=0..100:repeat(0..100,period=10,pattern=3:train(scope=head,opt=sgd,lr=0.1,stride=1,bptt=1,clip=0,momentum=0.9)+7:infer)",
            RWKV_SCOPES,
        );
        assert!(p.is_err());

        let p = parse_policy_segment(
            "schedule=repeat(0..100,period=10,pattern=3:train(scope=head,opt=sgd,lr=0.1,stride=1,bptt=1,clip=0,momentum=0.9)+7:infer)",
            RWKV_SCOPES,
        )
        .expect("repeat policy");
        let c = p.compile(Some(100)).expect("compile");
        assert!(matches!(c.action_at(0), PolicyAction::Train(_)));
        assert!(matches!(c.action_at(3), PolicyAction::Infer));
        assert!(matches!(c.action_at(10), PolicyAction::Train(_)));
    }

    #[test]
    fn split_method_policy() {
        let (base, pol) =
            split_method_policy_segments("cfg:hidden=64;policy:schedule=0..100:infer")
                .expect("split");
        assert_eq!(base, "cfg:hidden=64");
        assert_eq!(pol.as_deref(), Some("schedule=0..100:infer"));
    }
}
