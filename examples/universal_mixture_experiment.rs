use infotheory::datagen;
use infotheory::mixture::{
    BayesMixture, ExpertConfig, FadingBayesMixture, MdlSelector, OnlineBytePredictor,
    SwitchingMixture,
};

use rayon::prelude::*;
use std::env;
use std::f64::consts::LN_2;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

#[derive(Clone, Copy)]
struct Segment {
    name: &'static str,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy)]
struct GapConfig {
    name: &'static str,
    seg2_mut: f64,
    seg3_mut: f64,
    seg3_lag: usize,
}

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEF } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn next_u8(&mut self, max_exclusive: u8) -> u8 {
        if max_exclusive <= 1 {
            return 0;
        }
        (self.next_u64() % (max_exclusive as u64)) as u8
    }
}

#[inline]
fn logsumexp(xs: &[f64]) -> f64 {
    let mut max_v = f64::NEG_INFINITY;
    for &v in xs {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0;
    for &v in xs {
        sum += (v - max_v).exp();
    }
    max_v + sum.ln()
}

#[inline]
fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = if a > b { a } else { b };
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[inline]
fn neff(weights: &[f64]) -> f64 {
    1.0 / weights.iter().map(|w| w * w).sum::<f64>().max(1e-12)
}

#[inline]
fn entropy_bits(weights: &[f64]) -> f64 {
    let mut h = 0.0;
    for &w in weights {
        if w > 0.0 {
            h -= w * (w.ln() / LN_2);
        }
    }
    h
}

#[inline]
fn logodds_gap_bits(weights: &[f64]) -> Option<f64> {
    if weights.len() < 2 {
        return None;
    }
    let mut top = 0.0;
    let mut second = 0.0;
    for &w in weights {
        if w > top {
            second = top;
            top = w;
        } else if w > second {
            second = w;
        }
    }
    if top <= 0.0 || second <= 0.0 {
        return None;
    }
    Some((top / second).ln() / LN_2)
}

#[inline]
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

const ZPAQ_METHODS: [&str; 5] = [
    "1",
    "2",
    //    "3",
    //    "x5.0c0i1.1.1a24.1.1w1.65.26.223.20.0m8.24t8.24s8.32.255",
    //    "x4.0c256.0.255.255.255s8.32.255m8.24",
    //    "x4.0w1.65.26.223.20.0m8.24",
    //    "x4.0a24.1.1m8.24",
    "x4.0ci1.1.2",
    "x4.0c0",
    "x4.0ci8",
];

const ZPAQ_EXTENDED_METHODS: [&str; 5] = [
    "x6.0c0i1.1.1.2a24.1.1w2.65.26.223.20.0m8.24m16.24t8.24s8.32.255",
    "x5.0c64.0.255.255c0i1.1a24.1.1m8.24t8.24s8.32.255",
    "x5.0c0i1.1.2w2.65.26.223.20.0m8.24t8.24",
    "x5.0a24.0.0m8.24s8.32.255",
    "x5.0c0i1.1.1.1.2",
];

fn include_zpaq_from_args() -> bool {
    env::args().any(|a| a == "--include-zpaq")
}

fn include_zpaq_extended_from_args() -> bool {
    env::args().any(|a| a == "--include-zpaq-extended")
}

fn periodic_with_mutation(
    n: usize,
    pattern: &[u8],
    alphabet: u8,
    mutate_prob: f64,
    rng: &mut Xorshift64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut sym = pattern[i % pattern.len()];
        if rng.next_f64() < mutate_prob {
            sym = rng.next_u8(alphabet);
        }
        out.push(sym);
    }
    out
}

fn copy_mutate(
    n: usize,
    alphabet: u8,
    lag: usize,
    mutate_prob: f64,
    rng: &mut Xorshift64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = if i >= lag {
            out[i - lag]
        } else {
            rng.next_u8(alphabet)
        };
        let mut sym = base;
        if rng.next_f64() < mutate_prob {
            sym = rng.next_u8(alphabet);
        }
        out.push(sym);
    }
    out
}

fn build_dataset(seed: u64, seg_len: usize, gap: GapConfig) -> (Vec<u8>, Vec<Segment>) {
    let mut data = Vec::with_capacity(seg_len * 3);
    let mut segments = Vec::new();

    let seg1 = datagen::markov_1_binary(seg_len, 0.97, 0.93, seed);
    let start1 = data.len();
    data.extend_from_slice(&seg1);
    let end1 = data.len();
    segments.push(Segment {
        name: "markov-binary",
        start: start1,
        end: end1,
    });

    let mut rng = Xorshift64::new(seed.wrapping_add(1));
    let pattern = [0u8, 1, 2, 3, 2, 1, 0, 1, 2, 3, 1, 0];
    let seg2 = periodic_with_mutation(seg_len, &pattern, 4, gap.seg2_mut, &mut rng);
    let start2 = data.len();
    data.extend_from_slice(&seg2);
    let end2 = data.len();
    segments.push(Segment {
        name: "periodic+mutation",
        start: start2,
        end: end2,
    });

    let mut rng = Xorshift64::new(seed.wrapping_add(2));
    let seg3 = copy_mutate(seg_len, 8, gap.seg3_lag, gap.seg3_mut, &mut rng);
    let start3 = data.len();
    data.extend_from_slice(&seg3);
    let end3 = data.len();
    segments.push(Segment {
        name: "copy-mutate",
        start: start3,
        end: end3,
    });

    (data, segments)
}

struct MixStats {
    name: &'static str,
    total_bits: f64,
    regret_bits: f64,
    outperformance_steps: usize,
    longest_outperformance_run: usize,
    collapse_time: Option<usize>,
    time_below_neff: usize,
    min_neff: f64,
    min_entropy_bits: f64,
    inertia_bits: Option<f64>,
    adopt_times: Vec<Option<usize>>,
    soft_recovery_times: Vec<Option<usize>>,
}

fn summarize_mix(
    name: &'static str,
    total_bits: f64,
    min_bits: f64,
    outperformance_steps: usize,
    longest_run: usize,
    collapse_time: Option<usize>,
    time_below_neff: usize,
    min_neff: f64,
    min_entropy_bits: f64,
    inertia_bits: Option<f64>,
    adopt_times: Vec<Option<usize>>,
    soft_recovery_times: Vec<Option<usize>>,
) -> MixStats {
    MixStats {
        name,
        total_bits,
        regret_bits: total_bits - min_bits,
        outperformance_steps,
        longest_outperformance_run: longest_run,
        collapse_time,
        time_below_neff,
        min_neff,
        min_entropy_bits,
        inertia_bits,
        adopt_times,
        soft_recovery_times,
    }
}

struct RunSummary {
    seg_len: usize,
    alpha: f64,
    gap_name: &'static str,
    bayes_collapse: Option<usize>,
    bayes_min_neff: f64,
    bayes_min_entropy_bits: f64,
    bayes_recover2: Option<usize>,
    bayes_regret_bits: f64,
    bayes_inertia_bits: Option<f64>,
    bayes_inertia_seg2_bits: Option<f64>,
    bayes_inertia_seg3_bits: Option<f64>,
    bayes_inertia_boundary_max_bits: Option<f64>,
    bayes_max_logodds_bits: Option<f64>,
    bayes_delta_bpb_seg2: Option<f64>,
    bayes_delta_bpb_seg3: Option<f64>,
    bayes_oracle_adopt_seg2: Option<usize>,
    bayes_oracle_adopt_seg3: Option<usize>,
    bayes_predicted_time_seg2: Option<f64>,
    bayes_predicted_time_seg3: Option<f64>,
    fading_collapse: Option<usize>,
    fading_min_neff: f64,
    fading_min_entropy_bits: f64,
    fading_recover2: Option<usize>,
    fading_regret_bits: f64,
    fading_inertia_bits: Option<f64>,
    fading_inertia_seg2_bits: Option<f64>,
    fading_inertia_seg3_bits: Option<f64>,
    fading_inertia_boundary_max_bits: Option<f64>,
    fading_max_logodds_bits: Option<f64>,
    fading_delta_bpb_seg2: Option<f64>,
    fading_delta_bpb_seg3: Option<f64>,
    fading_oracle_adopt_seg2: Option<usize>,
    fading_oracle_adopt_seg3: Option<usize>,
    fading_predicted_time_seg2: Option<f64>,
    fading_predicted_time_seg3: Option<f64>,
    switch_excess_oracle_bits: f64,
    switch_excess_viterbi_bits: f64,
    switch_min_neff: f64,
    switch_min_entropy_bits: f64,
    switch_recover2: Option<usize>,
    switch_inertia_bits: Option<f64>,
    switch_inertia_seg2_bits: Option<f64>,
    switch_inertia_seg3_bits: Option<f64>,
    switch_inertia_boundary_max_bits: Option<f64>,
    switch_max_logodds_bits: Option<f64>,
    switch_delta_bpb_seg2: Option<f64>,
    switch_delta_bpb_seg3: Option<f64>,
    switch_oracle_adopt_seg2: Option<usize>,
    switch_oracle_adopt_seg3: Option<usize>,
    switch_predicted_time_seg2: Option<f64>,
    switch_predicted_time_seg3: Option<f64>,
    forward_vs_switch_max_diff: f64,
}

fn run_once(
    seed: u64,
    seg_len: usize,
    alpha: f64,
    decay: f64,
    gap: GapConfig,
    neff_collapse: f64,
    neff_recover: f64,
    include_zpaq: bool,
    include_zpaq_extended: bool,
    trace_path: Option<&str>,
    verbose: bool,
) -> (RunSummary, Vec<String>) {
    let (data, segments) = build_dataset(seed, seg_len, gap);

    // Keep a consistent byte alphabet across experts to avoid early-posterior artifacts.
    let symbol_bits = 8usize;

    let experts = vec![
        ExpertConfig::ctw("ctw-d8", 8),
        ExpertConfig::ctw("ctw-d16", 16),
        ExpertConfig::rosa("rosa-mo8", 8),
        ExpertConfig::rosa("rosa-mo32", 32),
        ExpertConfig::rosa("rosa-auto", -1),
        ExpertConfig::fac_ctw("fac-ctw-d6b8", 6, symbol_bits),
    ];
    let mut experts = experts;
    if include_zpaq {
        for method in ZPAQ_METHODS {
            let name = format!("zpaq-{}", method);
            experts.push(ExpertConfig::zpaq(name, method));
        }
    }
    if include_zpaq_extended {
        for method in ZPAQ_EXTENDED_METHODS {
            let name = format!("zpaq-{}", method);
            experts.push(ExpertConfig::zpaq(name, method));
        }
    }

    let mut bayes = BayesMixture::new(&experts);
    let mut switch = SwitchingMixture::new(&experts, alpha);
    let mut fading = FadingBayesMixture::new(&experts, decay);
    let mut mdl = MdlSelector::new(&experts);

    let mut bayes_bits = 0.0;
    let mut switch_bits = 0.0;
    let mut forward_bits = 0.0;
    let mut fading_bits = 0.0;
    let mut mdl_bits = 0.0;

    let mut bayes_out_count = 0usize;
    let mut bayes_out_run = 0usize;
    let mut bayes_out_run_max = 0usize;

    let mut switch_out_count = 0usize;
    let mut switch_out_run = 0usize;
    let mut switch_out_run_max = 0usize;

    let mut fading_out_count = 0usize;
    let mut fading_out_run = 0usize;
    let mut fading_out_run_max = 0usize;

    let mut mdl_out_count = 0usize;
    let mut mdl_out_run = 0usize;
    let mut mdl_out_run_max = 0usize;

    let mut bayes_collapse_time = None;
    let mut bayes_time_below = 0usize;
    let mut bayes_min_neff = f64::INFINITY;
    let mut bayes_min_entropy_bits = f64::INFINITY;
    let mut bayes_inertia_bits = None;
    let mut bayes_max_logodds_bits: Option<f64> = None;

    let mut switch_collapse_time = None;
    let mut switch_time_below = 0usize;
    let mut switch_min_neff = f64::INFINITY;
    let mut switch_min_entropy_bits = f64::INFINITY;
    let mut switch_inertia_bits = None;
    let mut switch_max_logodds_bits: Option<f64> = None;

    let mut fading_collapse_time = None;
    let mut fading_time_below = 0usize;
    let mut fading_min_neff = f64::INFINITY;
    let mut fading_min_entropy_bits = f64::INFINITY;
    let mut fading_inertia_bits = None;
    let mut fading_max_logodds_bits: Option<f64> = None;

    let mut bayes_adopt_times = vec![None; segments.len()];
    let mut switch_adopt_times = vec![None; segments.len()];
    let mut fading_adopt_times = vec![None; segments.len()];
    let mut bayes_soft_recovery = vec![None; segments.len()];
    let mut switch_soft_recovery = vec![None; segments.len()];
    let mut fading_soft_recovery = vec![None; segments.len()];
    let mut pre_shift_bayes_top = vec![None; segments.len()];
    let mut pre_shift_switch_top = vec![None; segments.len()];
    let mut pre_shift_fading_top = vec![None; segments.len()];
    let mut bayes_boundary_weights: Vec<Option<Vec<f64>>> = vec![None; segments.len()];
    let mut switch_boundary_weights: Vec<Option<Vec<f64>>> = vec![None; segments.len()];
    let mut fading_boundary_weights: Vec<Option<Vec<f64>>> = vec![None; segments.len()];
    let mut bayes_boundary_top: Vec<Option<usize>> = vec![None; segments.len()];
    let mut switch_boundary_top: Vec<Option<usize>> = vec![None; segments.len()];
    let mut fading_boundary_top: Vec<Option<usize>> = vec![None; segments.len()];
    let mut bayes_first_collapse_in_seg: Vec<Option<usize>> = vec![None; segments.len()];
    let mut switch_first_collapse_in_seg: Vec<Option<usize>> = vec![None; segments.len()];
    let mut fading_first_collapse_in_seg: Vec<Option<usize>> = vec![None; segments.len()];

    let expert_names: Vec<String> = experts.iter().map(|e| e.name().to_string()).collect();
    let expert_names_sanitized: Vec<String> =
        expert_names.iter().map(|n| sanitize_name(n)).collect();

    let mut expert_predictors: Vec<Box<dyn OnlineBytePredictor>> =
        experts.iter().map(|e| e.build_predictor()).collect();
    let n_experts = expert_predictors.len();

    let mut bayes_oracle_adopt_top = vec![vec![None; n_experts]; segments.len()];
    let mut switch_oracle_adopt_top = vec![vec![None; n_experts]; segments.len()];
    let mut fading_oracle_adopt_top = vec![vec![None; n_experts]; segments.len()];
    let mut bayes_oracle_adopt_conf = vec![vec![None; n_experts]; segments.len()];
    let mut switch_oracle_adopt_conf = vec![vec![None; n_experts]; segments.len()];
    let mut fading_oracle_adopt_conf = vec![vec![None; n_experts]; segments.len()];

    let mut expert_bits_by_segment = vec![vec![0.0; n_experts]; segments.len()];
    let mut bayes_bits_by_segment = vec![0.0; segments.len()];
    let mut switch_bits_by_segment = vec![0.0; segments.len()];
    let mut fading_bits_by_segment = vec![0.0; segments.len()];
    let mut mdl_bits_by_segment = vec![0.0; segments.len()];
    let mut expert_cum_bits = vec![0.0; n_experts];

    let log_priors_raw: Vec<f64> = experts.iter().map(|e| e.log_prior()).collect();
    let log_prior_norm = logsumexp(&log_priors_raw);
    let log_priors: Vec<f64> = log_priors_raw
        .iter()
        .map(|lp| lp - log_prior_norm)
        .collect();

    let mut forward_logw = log_priors.clone();
    let mut forward_logw_next = vec![0.0; n_experts];
    let mut forward_max_abs_diff: f64 = 0.0;

    let mut viterbi_cost = log_priors.iter().map(|lp| -lp).collect::<Vec<f64>>();
    let mut viterbi_cost_next = vec![0.0; n_experts];

    let mut seg_idx = 0usize;
    let mut seg_end = segments[seg_idx].end;

    let log_alpha = alpha.ln();
    let log_1m_alpha = (1.0 - alpha).ln();

    let ema_alpha = 0.01;
    let mut ema_bayes = 0.0;
    let mut ema_switch = 0.0;
    let mut ema_fading = 0.0;
    let mut ema_mdl = 0.0;
    let mut ema_experts = vec![0.0; n_experts];
    let mut ema_initialized = vec![false; n_experts];
    let mut ema_bayes_init = false;
    let mut ema_switch_init = false;
    let mut ema_fading_init = false;
    let mut ema_mdl_init = false;

    let mut trace_writer = if let Some(path) = trace_path {
        create_dir_all("examples/outputs").ok();
        let file = File::create(path).expect("failed to create trace CSV");
        let mut w = BufWriter::new(file);
        write!(w, "t,seg_idx,segment,symbol,").unwrap();
        write!(w, "bayes_bps,switch_bps,fading_bps,mdl_bps,").unwrap();
        write!(w, "bayes_ema,switch_ema,fading_ema,mdl_ema,").unwrap();
        write!(w, "bayes_neff,switch_neff,fading_neff,").unwrap();
        write!(
            w,
            "bayes_entropy_bits,switch_entropy_bits,fading_entropy_bits"
        )
        .unwrap();
        for name in &expert_names_sanitized {
            write!(w, ",bayes_w_{}", name).unwrap();
        }
        for name in &expert_names_sanitized {
            write!(w, ",switch_w_{}", name).unwrap();
        }
        for name in &expert_names_sanitized {
            write!(w, ",fading_w_{}", name).unwrap();
        }
        for name in &expert_names_sanitized {
            write!(w, ",exp_bps_{}", name).unwrap();
        }
        for name in &expert_names_sanitized {
            write!(w, ",exp_ema_{}", name).unwrap();
        }
        writeln!(w).unwrap();
        Some(w)
    } else {
        None
    };

    let mut step_logps = vec![0.0; n_experts];

    for (t, &sym) in data.iter().enumerate() {
        while t >= seg_end {
            seg_idx += 1;
            seg_end = segments[seg_idx].end;
        }

        let logp_bayes = bayes.step(sym);
        let logp_switch = switch.step(sym);
        let logp_fading = fading.step(sym);
        let logp_mdl = mdl.step(sym);

        let bps_bayes = -logp_bayes / LN_2;
        let bps_switch = -logp_switch / LN_2;
        let bps_fading = -logp_fading / LN_2;
        let bps_mdl = -logp_mdl / LN_2;

        bayes_bits += bps_bayes;
        switch_bits += bps_switch;
        fading_bits += bps_fading;
        mdl_bits += bps_mdl;

        bayes_bits_by_segment[seg_idx] += bps_bayes;
        switch_bits_by_segment[seg_idx] += bps_switch;
        fading_bits_by_segment[seg_idx] += bps_fading;
        mdl_bits_by_segment[seg_idx] += bps_mdl;

        for (i, pred) in expert_predictors.iter_mut().enumerate() {
            let logp = pred.log_prob(sym);
            pred.update(sym);
            step_logps[i] = logp;
            let bits = -logp / LN_2;
            expert_bits_by_segment[seg_idx][i] += bits;
            expert_cum_bits[i] += bits;
        }

        for i in 0..n_experts {
            let log_switch = logsumexp2(log_1m_alpha + forward_logw[i], log_alpha + log_priors[i]);
            forward_logw_next[i] = step_logps[i] + log_switch;
        }
        let log_mix_forward = logsumexp(&forward_logw_next);
        forward_bits += -log_mix_forward / LN_2;
        for i in 0..n_experts {
            forward_logw[i] = forward_logw_next[i] - log_mix_forward;
        }
        forward_max_abs_diff = forward_max_abs_diff.max((log_mix_forward - logp_switch).abs());

        let prev_best = viterbi_cost.iter().copied().fold(f64::INFINITY, f64::min);
        for i in 0..n_experts {
            let stay = viterbi_cost[i] - log_1m_alpha;
            let switch_cost = prev_best - log_alpha - log_priors[i];
            viterbi_cost_next[i] = stay.min(switch_cost) - step_logps[i];
        }
        viterbi_cost.clone_from_slice(&viterbi_cost_next);

        let best_expert_bits_t = expert_cum_bits
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        if bayes_bits < best_expert_bits_t {
            bayes_out_count += 1;
            bayes_out_run += 1;
            bayes_out_run_max = bayes_out_run_max.max(bayes_out_run);
        } else {
            bayes_out_run = 0;
        }

        if switch_bits < best_expert_bits_t {
            switch_out_count += 1;
            switch_out_run += 1;
            switch_out_run_max = switch_out_run_max.max(switch_out_run);
        } else {
            switch_out_run = 0;
        }

        if fading_bits < best_expert_bits_t {
            fading_out_count += 1;
            fading_out_run += 1;
            fading_out_run_max = fading_out_run_max.max(fading_out_run);
        } else {
            fading_out_run = 0;
        }

        if mdl_bits < best_expert_bits_t {
            mdl_out_count += 1;
            mdl_out_run += 1;
            mdl_out_run_max = mdl_out_run_max.max(mdl_out_run);
        } else {
            mdl_out_run = 0;
        }

        let bayes_weights = bayes.posterior();
        let switch_weights = switch.posterior();
        let fading_weights = fading.posterior();

        let (mut bayes_top_idx, mut bayes_top_w) = (0usize, 0.0);
        for (i, &w) in bayes_weights.iter().enumerate() {
            if w > bayes_top_w {
                bayes_top_w = w;
                bayes_top_idx = i;
            }
        }
        let (mut switch_top_idx, mut switch_top_w) = (0usize, 0.0);
        for (i, &w) in switch_weights.iter().enumerate() {
            if w > switch_top_w {
                switch_top_w = w;
                switch_top_idx = i;
            }
        }
        let (mut fading_top_idx, mut fading_top_w) = (0usize, 0.0);
        for (i, &w) in fading_weights.iter().enumerate() {
            if w > fading_top_w {
                fading_top_w = w;
                fading_top_idx = i;
            }
        }

        let time_in_seg = t + 1 - segments[seg_idx].start;
        if bayes_oracle_adopt_top[seg_idx][bayes_top_idx].is_none() {
            bayes_oracle_adopt_top[seg_idx][bayes_top_idx] = Some(time_in_seg);
        }
        if bayes_top_w >= 0.5 && bayes_oracle_adopt_conf[seg_idx][bayes_top_idx].is_none() {
            bayes_oracle_adopt_conf[seg_idx][bayes_top_idx] = Some(time_in_seg);
        }
        if switch_oracle_adopt_top[seg_idx][switch_top_idx].is_none() {
            switch_oracle_adopt_top[seg_idx][switch_top_idx] = Some(time_in_seg);
        }
        if switch_top_w >= 0.5 && switch_oracle_adopt_conf[seg_idx][switch_top_idx].is_none() {
            switch_oracle_adopt_conf[seg_idx][switch_top_idx] = Some(time_in_seg);
        }
        if fading_oracle_adopt_top[seg_idx][fading_top_idx].is_none() {
            fading_oracle_adopt_top[seg_idx][fading_top_idx] = Some(time_in_seg);
        }
        if fading_top_w >= 0.5 && fading_oracle_adopt_conf[seg_idx][fading_top_idx].is_none() {
            fading_oracle_adopt_conf[seg_idx][fading_top_idx] = Some(time_in_seg);
        }

        let bayes_neff = neff(&bayes_weights);
        let switch_neff = neff(&switch_weights);
        let fading_neff = neff(&fading_weights);

        let bayes_entropy = entropy_bits(&bayes_weights);
        let switch_entropy = entropy_bits(&switch_weights);
        let fading_entropy = entropy_bits(&fading_weights);

        bayes_min_neff = bayes_min_neff.min(bayes_neff);
        switch_min_neff = switch_min_neff.min(switch_neff);
        fading_min_neff = fading_min_neff.min(fading_neff);

        bayes_min_entropy_bits = bayes_min_entropy_bits.min(bayes_entropy);
        switch_min_entropy_bits = switch_min_entropy_bits.min(switch_entropy);
        fading_min_entropy_bits = fading_min_entropy_bits.min(fading_entropy);

        if bayes_neff <= neff_collapse {
            bayes_time_below += 1;
            if bayes_collapse_time.is_none() {
                bayes_collapse_time = Some(t + 1);
                bayes_inertia_bits = logodds_gap_bits(&bayes_weights);
            }
        }
        if switch_neff <= neff_collapse {
            switch_time_below += 1;
            if switch_collapse_time.is_none() {
                switch_collapse_time = Some(t + 1);
                switch_inertia_bits = logodds_gap_bits(&switch_weights);
            }
        }
        if fading_neff <= neff_collapse {
            fading_time_below += 1;
            if fading_collapse_time.is_none() {
                fading_collapse_time = Some(t + 1);
                fading_inertia_bits = logodds_gap_bits(&fading_weights);
            }
        }

        if let Some(gap_bits) = logodds_gap_bits(&bayes_weights) {
            bayes_max_logodds_bits = Some(
                bayes_max_logodds_bits
                    .map(|v| v.max(gap_bits))
                    .unwrap_or(gap_bits),
            );
        }
        if let Some(gap_bits) = logodds_gap_bits(&switch_weights) {
            switch_max_logodds_bits = Some(
                switch_max_logodds_bits
                    .map(|v| v.max(gap_bits))
                    .unwrap_or(gap_bits),
            );
        }
        if let Some(gap_bits) = logodds_gap_bits(&fading_weights) {
            fading_max_logodds_bits = Some(
                fading_max_logodds_bits
                    .map(|v| v.max(gap_bits))
                    .unwrap_or(gap_bits),
            );
        }

        for (seg_i, seg) in segments.iter().enumerate() {
            if seg.start == 0 {
                continue;
            }
            if t + 1 == seg.start {
                pre_shift_bayes_top[seg_i] = Some(bayes_top_idx);
                pre_shift_switch_top[seg_i] = Some(switch_top_idx);
                pre_shift_fading_top[seg_i] = Some(fading_top_idx);
                bayes_boundary_weights[seg_i] = Some(bayes_weights.clone());
                switch_boundary_weights[seg_i] = Some(switch_weights.clone());
                fading_boundary_weights[seg_i] = Some(fading_weights.clone());
                bayes_boundary_top[seg_i] = Some(bayes_top_idx);
                switch_boundary_top[seg_i] = Some(switch_top_idx);
                fading_boundary_top[seg_i] = Some(fading_top_idx);
            }
            if t + 1 >= seg.start && bayes_adopt_times[seg_i].is_none() {
                if let Some(prev) = pre_shift_bayes_top[seg_i] {
                    let (top, _) = bayes.max_posterior();
                    if top != prev && bayes_neff <= neff_collapse {
                        bayes_adopt_times[seg_i] = Some(t + 1 - seg.start);
                    }
                }
            }
            if t + 1 >= seg.start && switch_adopt_times[seg_i].is_none() {
                if let Some(prev) = pre_shift_switch_top[seg_i] {
                    let (top, _) = switch.max_posterior();
                    if top != prev && switch_neff <= neff_collapse {
                        switch_adopt_times[seg_i] = Some(t + 1 - seg.start);
                    }
                }
            }
            if t + 1 >= seg.start && fading_adopt_times[seg_i].is_none() {
                if let Some(prev) = pre_shift_fading_top[seg_i] {
                    let mut top = 0usize;
                    let mut best = 0.0;
                    for (idx, &p) in fading_weights.iter().enumerate() {
                        if p > best {
                            best = p;
                            top = idx;
                        }
                    }
                    if top != prev && fading_neff <= neff_collapse {
                        fading_adopt_times[seg_i] = Some(t + 1 - seg.start);
                    }
                }
            }
            if t + 1 >= seg.start && bayes_soft_recovery[seg_i].is_none() {
                if bayes_first_collapse_in_seg[seg_i].is_none() && bayes_neff <= neff_collapse {
                    bayes_first_collapse_in_seg[seg_i] = Some(t + 1);
                }
                if let Some(collapse_t) = bayes_first_collapse_in_seg[seg_i] {
                    if bayes_neff >= neff_recover {
                        bayes_soft_recovery[seg_i] = Some(t + 1 - collapse_t);
                    }
                }
            }
            if t + 1 >= seg.start && switch_soft_recovery[seg_i].is_none() {
                if switch_first_collapse_in_seg[seg_i].is_none() && switch_neff <= neff_collapse {
                    switch_first_collapse_in_seg[seg_i] = Some(t + 1);
                }
                if let Some(collapse_t) = switch_first_collapse_in_seg[seg_i] {
                    if switch_neff >= neff_recover {
                        switch_soft_recovery[seg_i] = Some(t + 1 - collapse_t);
                    }
                }
            }
            if t + 1 >= seg.start && fading_soft_recovery[seg_i].is_none() {
                if fading_first_collapse_in_seg[seg_i].is_none() && fading_neff <= neff_collapse {
                    fading_first_collapse_in_seg[seg_i] = Some(t + 1);
                }
                if let Some(collapse_t) = fading_first_collapse_in_seg[seg_i] {
                    if fading_neff >= neff_recover {
                        fading_soft_recovery[seg_i] = Some(t + 1 - collapse_t);
                    }
                }
            }
        }

        if ema_bayes_init {
            ema_bayes = (1.0 - ema_alpha) * ema_bayes + ema_alpha * bps_bayes;
        } else {
            ema_bayes = bps_bayes;
            ema_bayes_init = true;
        }
        if ema_switch_init {
            ema_switch = (1.0 - ema_alpha) * ema_switch + ema_alpha * bps_switch;
        } else {
            ema_switch = bps_switch;
            ema_switch_init = true;
        }
        if ema_fading_init {
            ema_fading = (1.0 - ema_alpha) * ema_fading + ema_alpha * bps_fading;
        } else {
            ema_fading = bps_fading;
            ema_fading_init = true;
        }
        if ema_mdl_init {
            ema_mdl = (1.0 - ema_alpha) * ema_mdl + ema_alpha * bps_mdl;
        } else {
            ema_mdl = bps_mdl;
            ema_mdl_init = true;
        }
        for i in 0..n_experts {
            if ema_initialized[i] {
                ema_experts[i] =
                    (1.0 - ema_alpha) * ema_experts[i] + ema_alpha * (-step_logps[i] / LN_2);
            } else {
                ema_experts[i] = -step_logps[i] / LN_2;
                ema_initialized[i] = true;
            }
        }

        if let Some(writer) = trace_writer.as_mut() {
            write!(
                writer,
                "{},{},{},{},",
                t, seg_idx, segments[seg_idx].name, sym
            )
            .unwrap();
            write!(
                writer,
                "{:.6},{:.6},{:.6},{:.6},",
                bps_bayes, bps_switch, bps_fading, bps_mdl
            )
            .unwrap();
            write!(
                writer,
                "{:.6},{:.6},{:.6},{:.6},",
                ema_bayes, ema_switch, ema_fading, ema_mdl
            )
            .unwrap();
            write!(
                writer,
                "{:.6},{:.6},{:.6},",
                bayes_neff, switch_neff, fading_neff
            )
            .unwrap();
            write!(
                writer,
                "{:.6},{:.6},{:.6}",
                bayes_entropy, switch_entropy, fading_entropy
            )
            .unwrap();
            for w in &bayes_weights {
                write!(writer, ",{:.6}", w).unwrap();
            }
            for w in &switch_weights {
                write!(writer, ",{:.6}", w).unwrap();
            }
            for w in &fading_weights {
                write!(writer, ",{:.6}", w).unwrap();
            }
            for i in 0..n_experts {
                write!(writer, ",{:.6}", -step_logps[i] / LN_2).unwrap();
            }
            for i in 0..n_experts {
                write!(writer, ",{:.6}", ema_experts[i]).unwrap();
            }
            writeln!(writer).unwrap();
        }
    }

    if let Some(writer) = trace_writer.as_mut() {
        writer.flush().ok();
    }

    let total_len = data.len() as f64;
    let best_static_bits = expert_cum_bits
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let bayes_stats = summarize_mix(
        "bayes",
        bayes_bits,
        best_static_bits,
        bayes_out_count,
        bayes_out_run_max,
        bayes_collapse_time,
        bayes_time_below,
        bayes_min_neff,
        bayes_min_entropy_bits,
        bayes_inertia_bits,
        bayes_adopt_times.clone(),
        bayes_soft_recovery.clone(),
    );
    let switch_stats = summarize_mix(
        "switch",
        switch_bits,
        best_static_bits,
        switch_out_count,
        switch_out_run_max,
        switch_collapse_time,
        switch_time_below,
        switch_min_neff,
        switch_min_entropy_bits,
        switch_inertia_bits,
        switch_adopt_times.clone(),
        switch_soft_recovery.clone(),
    );
    let fading_stats = summarize_mix(
        "fading",
        fading_bits,
        best_static_bits,
        fading_out_count,
        fading_out_run_max,
        fading_collapse_time,
        fading_time_below,
        fading_min_neff,
        fading_min_entropy_bits,
        fading_inertia_bits,
        fading_adopt_times.clone(),
        fading_soft_recovery.clone(),
    );
    let mdl_stats = summarize_mix(
        "mdl",
        mdl_bits,
        best_static_bits,
        mdl_out_count,
        mdl_out_run_max,
        None,
        0,
        f64::NAN,
        f64::NAN,
        None,
        vec![None; segments.len()],
        vec![None; segments.len()],
    );

    let mut oracle_segment_bits = 0.0;
    let mut oracle_segment_best = Vec::new();
    for seg_i in 0..segments.len() {
        let mut best = f64::INFINITY;
        let mut best_i = 0usize;
        for i in 0..n_experts {
            let bits = expert_bits_by_segment[seg_i][i];
            if bits < best {
                best = bits;
                best_i = i;
            }
        }
        oracle_segment_bits += best;
        oracle_segment_best.push(best_i);
    }
    let viterbi_best_bits = viterbi_cost.iter().copied().fold(f64::INFINITY, f64::min) / LN_2;

    let mut bayes_inertia_boundary = vec![None; segments.len()];
    let mut switch_inertia_boundary = vec![None; segments.len()];
    let mut fading_inertia_boundary = vec![None; segments.len()];
    let mut bayes_inertia_boundary_max: Option<f64> = None;
    let mut switch_inertia_boundary_max: Option<f64> = None;
    let mut fading_inertia_boundary_max: Option<f64> = None;
    let mut bayes_delta_bpb_seg2: Option<f64> = None;
    let mut bayes_delta_bpb_seg3: Option<f64> = None;
    let mut bayes_oracle_adopt_seg2: Option<usize> = None;
    let mut bayes_oracle_adopt_seg3: Option<usize> = None;
    let mut bayes_predicted_time_seg2: Option<f64> = None;
    let mut bayes_predicted_time_seg3: Option<f64> = None;
    let mut switch_delta_bpb_seg2: Option<f64> = None;
    let mut switch_delta_bpb_seg3: Option<f64> = None;
    let mut switch_oracle_adopt_seg2: Option<usize> = None;
    let mut switch_oracle_adopt_seg3: Option<usize> = None;
    let mut switch_predicted_time_seg2: Option<f64> = None;
    let mut switch_predicted_time_seg3: Option<f64> = None;
    let mut fading_delta_bpb_seg2: Option<f64> = None;
    let mut fading_delta_bpb_seg3: Option<f64> = None;
    let mut fading_oracle_adopt_seg2: Option<usize> = None;
    let mut fading_oracle_adopt_seg3: Option<usize> = None;
    let mut fading_predicted_time_seg2: Option<f64> = None;
    let mut fading_predicted_time_seg3: Option<f64> = None;

    for seg_i in 1..segments.len() {
        let oracle_idx = oracle_segment_best[seg_i];
        let seg_len_f = (segments[seg_i].end - segments[seg_i].start) as f64;
        let oracle_bpb = expert_bits_by_segment[seg_i][oracle_idx] / seg_len_f;
        if let Some(weights) = bayes_boundary_weights[seg_i].as_ref() {
            let mut top = 0.0;
            for &w in weights {
                if w > top {
                    top = w;
                }
            }
            let w_oracle = weights[oracle_idx];
            if w_oracle > 0.0 && top > 0.0 {
                let bits = (top / w_oracle).ln() / LN_2;
                bayes_inertia_boundary[seg_i] = Some(bits);
                bayes_inertia_boundary_max = Some(
                    bayes_inertia_boundary_max
                        .map(|v| v.max(bits))
                        .unwrap_or(bits),
                );
            }
        }
        if let Some(weights) = switch_boundary_weights[seg_i].as_ref() {
            let mut top = 0.0;
            for &w in weights {
                if w > top {
                    top = w;
                }
            }
            let w_oracle = weights[oracle_idx];
            if w_oracle > 0.0 && top > 0.0 {
                let bits = (top / w_oracle).ln() / LN_2;
                switch_inertia_boundary[seg_i] = Some(bits);
                switch_inertia_boundary_max = Some(
                    switch_inertia_boundary_max
                        .map(|v| v.max(bits))
                        .unwrap_or(bits),
                );
            }
        }
        if let Some(weights) = fading_boundary_weights[seg_i].as_ref() {
            let mut top = 0.0;
            for &w in weights {
                if w > top {
                    top = w;
                }
            }
            let w_oracle = weights[oracle_idx];
            if w_oracle > 0.0 && top > 0.0 {
                let bits = (top / w_oracle).ln() / LN_2;
                fading_inertia_boundary[seg_i] = Some(bits);
                fading_inertia_boundary_max = Some(
                    fading_inertia_boundary_max
                        .map(|v| v.max(bits))
                        .unwrap_or(bits),
                );
            }
        }

        if let Some(top_idx) = bayes_boundary_top[seg_i] {
            let top_bpb = expert_bits_by_segment[seg_i][top_idx] / seg_len_f;
            let delta = oracle_bpb - top_bpb;
            let inertia = bayes_inertia_boundary[seg_i];
            let predicted = inertia.and_then(|bits| {
                if delta < 0.0 {
                    Some(bits / (-delta))
                } else {
                    None
                }
            });
            let adopt = bayes_oracle_adopt_conf[seg_i][oracle_idx]
                .or(bayes_oracle_adopt_top[seg_i][oracle_idx]);
            if seg_i == 1 {
                bayes_delta_bpb_seg2 = Some(delta);
                bayes_oracle_adopt_seg2 = adopt;
                bayes_predicted_time_seg2 = predicted;
            } else if seg_i == 2 {
                bayes_delta_bpb_seg3 = Some(delta);
                bayes_oracle_adopt_seg3 = adopt;
                bayes_predicted_time_seg3 = predicted;
            }
        }
        if let Some(top_idx) = switch_boundary_top[seg_i] {
            let top_bpb = expert_bits_by_segment[seg_i][top_idx] / seg_len_f;
            let delta = oracle_bpb - top_bpb;
            let inertia = switch_inertia_boundary[seg_i];
            let predicted = inertia.and_then(|bits| {
                if delta < 0.0 {
                    Some(bits / (-delta))
                } else {
                    None
                }
            });
            let adopt = switch_oracle_adopt_conf[seg_i][oracle_idx]
                .or(switch_oracle_adopt_top[seg_i][oracle_idx]);
            if seg_i == 1 {
                switch_delta_bpb_seg2 = Some(delta);
                switch_oracle_adopt_seg2 = adopt;
                switch_predicted_time_seg2 = predicted;
            } else if seg_i == 2 {
                switch_delta_bpb_seg3 = Some(delta);
                switch_oracle_adopt_seg3 = adopt;
                switch_predicted_time_seg3 = predicted;
            }
        }
        if let Some(top_idx) = fading_boundary_top[seg_i] {
            let top_bpb = expert_bits_by_segment[seg_i][top_idx] / seg_len_f;
            let delta = oracle_bpb - top_bpb;
            let inertia = fading_inertia_boundary[seg_i];
            let predicted = inertia.and_then(|bits| {
                if delta < 0.0 {
                    Some(bits / (-delta))
                } else {
                    None
                }
            });
            let adopt = fading_oracle_adopt_conf[seg_i][oracle_idx]
                .or(fading_oracle_adopt_top[seg_i][oracle_idx]);
            if seg_i == 1 {
                fading_delta_bpb_seg2 = Some(delta);
                fading_oracle_adopt_seg2 = adopt;
                fading_predicted_time_seg2 = predicted;
            } else if seg_i == 2 {
                fading_delta_bpb_seg3 = Some(delta);
                fading_oracle_adopt_seg3 = adopt;
                fading_predicted_time_seg3 = predicted;
            }
        }
    }

    if verbose {
        println!(
            "Mixture experiment (seed={}, total_len={}, gap={}, alpha={:.1e})",
            seed,
            data.len(),
            gap.name,
            alpha
        );
        println!("Experts: {:?}", expert_names);
        println!("Segments:");
        for seg in &segments {
            println!("  - {}: [{}..{})", seg.name, seg.start, seg.end);
        }
        println!();

        for stats in [&bayes_stats, &switch_stats, &fading_stats, &mdl_stats] {
            println!("{}:", stats.name);
            println!("  total bits/byte = {:.4}", stats.total_bits / total_len);
            println!("  regret (bits)   = {:.2}", stats.regret_bits);
            println!(
                "  outperformance steps = {} (max run {})",
                stats.outperformance_steps, stats.longest_outperformance_run
            );
            if let Some(t) = stats.collapse_time {
                println!("  collapse @t (Neff<= {:.2}) = {}", neff_collapse, t);
                println!(
                    "  time Neff<= {:.2} = {} steps",
                    neff_collapse, stats.time_below_neff
                );
                if let Some(bits) = stats.inertia_bits {
                    println!("  posterior inertia = {:.2} bits", bits);
                }
            }
            if stats.name != "mdl" {
                println!("  min Neff = {:.3}", stats.min_neff);
                println!("  min entropy (bits) = {:.3}", stats.min_entropy_bits);
                println!(
                    "  adopt times after shifts (steps): {:?}",
                    stats.adopt_times
                );
                println!(
                    "  soft recovery (Neff>= {:.2}) times: {:?}",
                    neff_recover, stats.soft_recovery_times
                );
            }
            println!();
        }

        println!(
            "Boundary inertia (bits) Bayes: {:?}",
            bayes_inertia_boundary
        );
        println!(
            "Boundary inertia (bits) Switch: {:?}",
            switch_inertia_boundary
        );
        println!(
            "Boundary inertia (bits) Fading: {:?}",
            fading_inertia_boundary
        );
        println!(
            "Max log-odds gap (bits) Bayes: {:?}",
            bayes_max_logodds_bits
        );
        println!(
            "Max log-odds gap (bits) Switch: {:?}",
            switch_max_logodds_bits
        );
        println!(
            "Max log-odds gap (bits) Fading: {:?}",
            fading_max_logodds_bits
        );
        println!();

        println!(
            "Oracle segmented total bits/byte = {:.4}",
            oracle_segment_bits / total_len
        );
        println!(
            "Forward switch total bits/byte = {:.4}",
            forward_bits / total_len
        );
        println!(
            "Forward-vs-switch max |Δlogp| = {:.3e}",
            forward_max_abs_diff
        );
        println!(
            "Viterbi best-switch total bits/byte = {:.4}",
            viterbi_best_bits / total_len
        );
        println!(
            "Switch excess vs oracle (bits) = {:.2}",
            switch_bits - oracle_segment_bits
        );
        println!(
            "Switch excess vs Viterbi (bits) = {:.2}",
            switch_bits - viterbi_best_bits
        );
        println!();

        let name_width = expert_names
            .iter()
            .map(|n| n.len())
            .max()
            .unwrap_or(6)
            .max("segment".len());
        let col_width = expert_names
            .iter()
            .map(|n| n.len())
            .max()
            .unwrap_or(10)
            .max(10)
            + 2;

        println!("Per-segment bits/byte:");
        print!("{:width$}", "segment", width = name_width + 2);
        for name in &expert_names {
            print!("{:>width$}", name, width = col_width);
        }
        print!(
            "{:>width$}{:>width$}{:>width$}{:>width$}\n",
            "bayes",
            "switch",
            "fading",
            "mdl",
            width = col_width
        );
        for (seg_i, seg) in segments.iter().enumerate() {
            print!("{:width$}", seg.name, width = name_width + 2);
            let seg_len_f = (seg.end - seg.start) as f64;
            for i in 0..n_experts {
                let bpb = expert_bits_by_segment[seg_i][i] / seg_len_f;
                print!("{:>width$.4}", bpb, width = col_width);
            }
            print!(
                "{:>width$.4}{:>width$.4}{:>width$.4}{:>width$.4}\n",
                bayes_bits_by_segment[seg_i] / seg_len_f,
                switch_bits_by_segment[seg_i] / seg_len_f,
                fading_bits_by_segment[seg_i] / seg_len_f,
                mdl_bits_by_segment[seg_i] / seg_len_f,
                width = col_width
            );
        }

        let oracle_names: Vec<&str> = oracle_segment_best
            .iter()
            .map(|&idx| expert_names[idx].as_str())
            .collect();
        println!("\nOracle best expert per segment: {:?}", oracle_names);
        println!("Final posteriors (Bayes): {:?}", bayes.posterior());
        println!("Final posteriors (Switch): {:?}", switch.posterior());
        println!("Final posteriors (Fading): {:?}", fading.posterior());
    }

    let summary = RunSummary {
        seg_len,
        alpha,
        gap_name: gap.name,
        bayes_collapse: bayes_collapse_time,
        bayes_min_neff,
        bayes_min_entropy_bits,
        bayes_recover2: bayes_soft_recovery.get(2).copied().unwrap_or(None),
        bayes_regret_bits: bayes_stats.regret_bits,
        bayes_inertia_bits,
        bayes_inertia_seg2_bits: bayes_inertia_boundary.get(1).copied().unwrap_or(None),
        bayes_inertia_seg3_bits: bayes_inertia_boundary.get(2).copied().unwrap_or(None),
        bayes_inertia_boundary_max_bits: bayes_inertia_boundary_max,
        bayes_max_logodds_bits: bayes_max_logodds_bits,
        bayes_delta_bpb_seg2,
        bayes_delta_bpb_seg3,
        bayes_oracle_adopt_seg2,
        bayes_oracle_adopt_seg3,
        bayes_predicted_time_seg2,
        bayes_predicted_time_seg3,
        fading_collapse: fading_collapse_time,
        fading_min_neff,
        fading_min_entropy_bits,
        fading_recover2: fading_soft_recovery.get(2).copied().unwrap_or(None),
        fading_regret_bits: fading_stats.regret_bits,
        fading_inertia_bits,
        fading_inertia_seg2_bits: fading_inertia_boundary.get(1).copied().unwrap_or(None),
        fading_inertia_seg3_bits: fading_inertia_boundary.get(2).copied().unwrap_or(None),
        fading_inertia_boundary_max_bits: fading_inertia_boundary_max,
        fading_max_logodds_bits: fading_max_logodds_bits,
        fading_delta_bpb_seg2,
        fading_delta_bpb_seg3,
        fading_oracle_adopt_seg2,
        fading_oracle_adopt_seg3,
        fading_predicted_time_seg2,
        fading_predicted_time_seg3,
        switch_excess_oracle_bits: switch_bits - oracle_segment_bits,
        switch_excess_viterbi_bits: switch_bits - viterbi_best_bits,
        switch_min_neff,
        switch_min_entropy_bits,
        switch_recover2: switch_soft_recovery.get(2).copied().unwrap_or(None),
        switch_inertia_bits,
        switch_inertia_seg2_bits: switch_inertia_boundary.get(1).copied().unwrap_or(None),
        switch_inertia_seg3_bits: switch_inertia_boundary.get(2).copied().unwrap_or(None),
        switch_inertia_boundary_max_bits: switch_inertia_boundary_max,
        switch_max_logodds_bits: switch_max_logodds_bits,
        switch_delta_bpb_seg2,
        switch_delta_bpb_seg3,
        switch_oracle_adopt_seg2,
        switch_oracle_adopt_seg3,
        switch_predicted_time_seg2,
        switch_predicted_time_seg3,
        forward_vs_switch_max_diff: forward_max_abs_diff,
    };

    (summary, expert_names)
}

fn main() {
    let include_zpaq = include_zpaq_from_args();
    let include_zpaq_extended = include_zpaq_extended_from_args();
    let include_zpaq_all = include_zpaq || include_zpaq_extended;
    let seed = 42u64;
    let seg_len = 20_000usize;
    let alpha = 0.001f64;
    let decay = 0.999;
    let neff_collapse = 1.2;
    let neff_recover = 1.5;

    let gap = GapConfig {
        name: "medium",
        seg2_mut: 0.03,
        seg3_mut: 0.08,
        seg3_lag: 11,
    };

    let trace_path = "examples/outputs/universal_mixture_trace.csv";
    let (_summary, _names) = run_once(
        seed,
        seg_len,
        alpha,
        decay,
        gap,
        neff_collapse,
        neff_recover,
        include_zpaq_all,
        include_zpaq_extended,
        Some(trace_path),
        true,
    );

    let seg_lengths = [2000usize, 5000, 10_000, 20_000, 50_000];
    let alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
    let gaps = [
        GapConfig {
            name: "easy",
            seg2_mut: 0.01,
            seg3_mut: 0.03,
            seg3_lag: 8,
        },
        GapConfig {
            name: "medium",
            seg2_mut: 0.03,
            seg3_mut: 0.08,
            seg3_lag: 11,
        },
        GapConfig {
            name: "hard",
            seg2_mut: 0.08,
            seg3_mut: 0.15,
            seg3_lag: 16,
        },
    ];

    let mut grid = Vec::new();
    for &seg_len in &seg_lengths {
        for &alpha in &alphas {
            for &gap in &gaps {
                grid.push((seg_len, alpha, gap));
            }
        }
    }

    let mut summaries: Vec<(usize, RunSummary)> = grid
        .par_iter()
        .enumerate()
        .map(|(idx, (seg_len, alpha, gap))| {
            let (summary, _) = run_once(
                seed,
                *seg_len,
                *alpha,
                decay,
                *gap,
                neff_collapse,
                neff_recover,
                include_zpaq_all,
                include_zpaq_extended,
                None,
                false,
            );
            (idx, summary)
        })
        .collect();
    summaries.sort_by_key(|(idx, _)| *idx);

    let sweep_path = "examples/outputs/phase_sweep.csv";
    create_dir_all("examples/outputs").ok();
    let file = File::create(sweep_path).expect("failed to create sweep CSV");
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "seg_len,alpha,gap,bayes_collapse,bayes_min_neff,bayes_min_entropy_bits,bayes_recover2,bayes_regret_bits,bayes_inertia_bits,bayes_inertia_seg2_bits,bayes_inertia_seg3_bits,bayes_inertia_boundary_max_bits,bayes_max_logodds_bits,bayes_delta_bpb_seg2,bayes_delta_bpb_seg3,bayes_oracle_adopt_seg2,bayes_oracle_adopt_seg3,bayes_predicted_time_seg2,bayes_predicted_time_seg3,fading_collapse,fading_min_neff,fading_min_entropy_bits,fading_recover2,fading_regret_bits,fading_inertia_bits,fading_inertia_seg2_bits,fading_inertia_seg3_bits,fading_inertia_boundary_max_bits,fading_max_logodds_bits,fading_delta_bpb_seg2,fading_delta_bpb_seg3,fading_oracle_adopt_seg2,fading_oracle_adopt_seg3,fading_predicted_time_seg2,fading_predicted_time_seg3,switch_excess_oracle_bits,switch_excess_viterbi_bits,switch_min_neff,switch_min_entropy_bits,switch_recover2,switch_inertia_bits,switch_inertia_seg2_bits,switch_inertia_seg3_bits,switch_inertia_boundary_max_bits,switch_max_logodds_bits,switch_delta_bpb_seg2,switch_delta_bpb_seg3,switch_oracle_adopt_seg2,switch_oracle_adopt_seg3,switch_predicted_time_seg2,switch_predicted_time_seg3,forward_vs_switch_max_diff"
    )
    .unwrap();

    let fmt_opt_usize = |v: Option<usize>| v.map(|x| x.to_string()).unwrap_or_default();
    let fmt_opt_f64 = |v: Option<f64>| v.map(|x| format!("{:.6}", x)).unwrap_or_default();

    for (_, summary) in summaries {
        let mut row = Vec::with_capacity(64);
        row.push(summary.seg_len.to_string());
        row.push(format!("{:.1e}", summary.alpha));
        row.push(summary.gap_name.to_string());
        row.push(fmt_opt_usize(summary.bayes_collapse));
        row.push(format!("{:.6}", summary.bayes_min_neff));
        row.push(format!("{:.6}", summary.bayes_min_entropy_bits));
        row.push(fmt_opt_usize(summary.bayes_recover2));
        row.push(format!("{:.6}", summary.bayes_regret_bits));
        row.push(fmt_opt_f64(summary.bayes_inertia_bits));
        row.push(fmt_opt_f64(summary.bayes_inertia_seg2_bits));
        row.push(fmt_opt_f64(summary.bayes_inertia_seg3_bits));
        row.push(fmt_opt_f64(summary.bayes_inertia_boundary_max_bits));
        row.push(fmt_opt_f64(summary.bayes_max_logodds_bits));
        row.push(fmt_opt_f64(summary.bayes_delta_bpb_seg2));
        row.push(fmt_opt_f64(summary.bayes_delta_bpb_seg3));
        row.push(fmt_opt_usize(summary.bayes_oracle_adopt_seg2));
        row.push(fmt_opt_usize(summary.bayes_oracle_adopt_seg3));
        row.push(fmt_opt_f64(summary.bayes_predicted_time_seg2));
        row.push(fmt_opt_f64(summary.bayes_predicted_time_seg3));
        row.push(fmt_opt_usize(summary.fading_collapse));
        row.push(format!("{:.6}", summary.fading_min_neff));
        row.push(format!("{:.6}", summary.fading_min_entropy_bits));
        row.push(fmt_opt_usize(summary.fading_recover2));
        row.push(format!("{:.6}", summary.fading_regret_bits));
        row.push(fmt_opt_f64(summary.fading_inertia_bits));
        row.push(fmt_opt_f64(summary.fading_inertia_seg2_bits));
        row.push(fmt_opt_f64(summary.fading_inertia_seg3_bits));
        row.push(fmt_opt_f64(summary.fading_inertia_boundary_max_bits));
        row.push(fmt_opt_f64(summary.fading_max_logodds_bits));
        row.push(fmt_opt_f64(summary.fading_delta_bpb_seg2));
        row.push(fmt_opt_f64(summary.fading_delta_bpb_seg3));
        row.push(fmt_opt_usize(summary.fading_oracle_adopt_seg2));
        row.push(fmt_opt_usize(summary.fading_oracle_adopt_seg3));
        row.push(fmt_opt_f64(summary.fading_predicted_time_seg2));
        row.push(fmt_opt_f64(summary.fading_predicted_time_seg3));
        row.push(format!("{:.6}", summary.switch_excess_oracle_bits));
        row.push(format!("{:.6}", summary.switch_excess_viterbi_bits));
        row.push(format!("{:.6}", summary.switch_min_neff));
        row.push(format!("{:.6}", summary.switch_min_entropy_bits));
        row.push(fmt_opt_usize(summary.switch_recover2));
        row.push(fmt_opt_f64(summary.switch_inertia_bits));
        row.push(fmt_opt_f64(summary.switch_inertia_seg2_bits));
        row.push(fmt_opt_f64(summary.switch_inertia_seg3_bits));
        row.push(fmt_opt_f64(summary.switch_inertia_boundary_max_bits));
        row.push(fmt_opt_f64(summary.switch_max_logodds_bits));
        row.push(fmt_opt_f64(summary.switch_delta_bpb_seg2));
        row.push(fmt_opt_f64(summary.switch_delta_bpb_seg3));
        row.push(fmt_opt_usize(summary.switch_oracle_adopt_seg2));
        row.push(fmt_opt_usize(summary.switch_oracle_adopt_seg3));
        row.push(fmt_opt_f64(summary.switch_predicted_time_seg2));
        row.push(fmt_opt_f64(summary.switch_predicted_time_seg3));
        row.push(format!("{:.3e}", summary.forward_vs_switch_max_diff));
        writeln!(w, "{}", row.join(",")).unwrap();
    }

    println!("\nWrote trace CSV: {}", trace_path);
    println!("Wrote sweep CSV: {}", sweep_path);
    println!(
        "ZPAQ experts: {}",
        if include_zpaq_all {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "ZPAQ extended experts: {}",
        if include_zpaq_extended {
            "enabled"
        } else {
            "disabled"
        }
    );
}
