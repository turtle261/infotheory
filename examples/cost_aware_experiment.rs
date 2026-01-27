//! Cost-Aware Mixture Experiment
//!
//! This experiment runs a Pareto sweep over λ values to find the optimal
//! bits-vs-time tradeoff. It outputs clean CSV data for analysis.
//!
//! Usage:
//!   cargo run --release --example cost_aware_experiment -- [options]
//!
//! Options:
//!   --lambda-min-exp <int>    Minimum λ exponent (default: -12)
//!   --lambda-max-exp <int>    Maximum λ exponent (default: -6)
//!   --points-per-decade <n>   λ grid density (default: 5)
//!   --data-size <n>           Bytes of synthetic data (default: 10000)
//!   --data <path>             Input data file (default: data.bin if present)
//!   --data-limit <n>          Truncate input to n bytes
//!   --synthetic               Force synthetic data instead of file
//!   --seed <u64>              Seed for synthetic data (default: 42)
//!   --alpha <f64>             Switching probability (default: 0.001)
//!   --output <path>           Output CSV path (default: examples/outputs/cost_aware_pareto.csv)
//!   --trace <path>            Trace CSV path (optional, outputs per-step data)
//!   --time-budget <ns>        Target time budget (runs bisection instead of sweep)
//!   --trace-lambda <f64>      Lambda value for trace (overrides bisection)
//!   --bisection-only          Only run bisection (skip sweep)
//!   --include-zpaq            Include ZPAQ experts
//!   --include-zpaq-extended   Include extended ZPAQ experts
//!   --verbose                 Print progress

use infotheory::datagen;
use infotheory::mixture::{
    bisect_for_time_budget, log_lambda_grid, CostAwareSwitchingMixture, ExpertConfig,
    ParetoPoint,
};

use std::env;
use std::f64::consts::LN_2;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use rayon::prelude::*;

const ZPAQ_METHODS: [&str; 10] = [
    "1",
    "2",
    "3",
    "x5.0c0i1.1.1a24.1.1w1.65.26.223.20.0m8.24t8.24s8.32.255",
    "x4.0c256.0.255.255.255s8.32.255m8.24",
    "x4.0w1.65.26.223.20.0m8.24",
    "x4.0a24.1.1m8.24",
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

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn build_experts(include_zpaq: bool, include_zpaq_extended: bool) -> Vec<ExpertConfig> {
    let symbol_bits = 8usize;

    let mut experts = vec![
        ExpertConfig::ctw("ctw-d8", 8),
        ExpertConfig::ctw("ctw-d16", 16),
        ExpertConfig::rosa("rosa-mo8", 8),
        ExpertConfig::rosa("rosa-mo32", 32),
        ExpertConfig::rosa("rosa-auto", -1),
        ExpertConfig::fac_ctw("fac-ctw-d6b8", 6, symbol_bits),
    ];

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

    experts
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
        let base = if i >= lag { out[i - lag] } else { rng.next_u8(alphabet) };
        let mut sym = base;
        if rng.next_f64() < mutate_prob {
            sym = rng.next_u8(alphabet);
        }
        out.push(sym);
    }
    out
}

fn build_synthetic_data(size: usize, seed: u64) -> Vec<u8> {
    // Mix of different data types to stress-test expert selection.
    let seg_len = size / 3;
    let mut data = Vec::with_capacity(size);

    // Segment 1: Markov binary
    let seg1 = datagen::markov_1_binary(seg_len, 0.97, 0.93, seed);
    data.extend_from_slice(&seg1);

    // Segment 2: Periodic with mutation
    let pattern = [0u8, 1, 2, 3, 2, 1, 0, 1, 2, 3, 1, 0];
    let mut rng = Xorshift64::new(seed.wrapping_add(1));
    let seg2 = periodic_with_mutation(seg_len, &pattern, 4, 0.03, &mut rng);
    data.extend_from_slice(&seg2);

    // Segment 3: Copy-mutate
    let mut rng = Xorshift64::new(seed.wrapping_add(2));
    let seg3 = copy_mutate(seg_len, 8, 11, 0.08, &mut rng);
    data.extend_from_slice(&seg3);

    data
}

struct ExpertEval {
    name: String,
    total_bits: f64,
    bps: f64,
    total_time_ns: u64,
    time_per_symbol_ns: f64,
}

fn evaluate_experts(data: &[u8], configs: &[ExpertConfig]) -> Vec<ExpertEval> {
    configs
        .par_iter()
        .map(|cfg| {
            let mut pred = cfg.build_predictor();
            let mut bits = 0.0;
            let mut time_ns: u64 = 0;
            for &sym in data {
                let t0 = Instant::now();
                let logp = pred.log_prob(sym);
                pred.update(sym);
                time_ns += t0.elapsed().as_nanos() as u64;
                bits += -logp / LN_2;
            }
            let n = data.len().max(1) as f64;
            ExpertEval {
                name: cfg.name().to_string(),
                total_bits: bits,
                bps: bits / n,
                total_time_ns: time_ns,
                time_per_symbol_ns: time_ns as f64 / n,
            }
        })
        .collect()
}

fn read_data(
    data_path: Option<&str>,
    data_limit: Option<usize>,
    synthetic: bool,
    data_size: usize,
    seed: u64,
) -> (Vec<u8>, String) {
    let default_path = "data.bin";
    let path = data_path.unwrap_or(default_path);
    if !synthetic && Path::new(path).exists() {
        let mut data = std::fs::read(path).expect("failed to read data file");
        if let Some(limit) = data_limit {
            if data.len() > limit {
                data.truncate(limit);
            }
        }
        return (data, path.to_string());
    }
    let mut data = build_synthetic_data(data_size, seed);
    if let Some(limit) = data_limit {
        if data.len() > limit {
            data.truncate(limit);
        }
    }
    (data, "synthetic".to_string())
}

fn write_pareto_csv(path: &str, points: &[ParetoPoint], expert_names: &[String]) {
    create_dir_all("examples/outputs").ok();
    let file = File::create(path).expect("failed to create output CSV");
    let mut w = BufWriter::new(file);

    // Header
    write!(
        w,
        "lambda,total_bits,bps,total_expected_time_ns,expected_time_per_symbol_ns,"
    )
    .unwrap();
    write!(
        w,
        "total_wall_time_ns,wall_time_per_symbol_ns,num_symbols,neff_min,neff_mean,entropy_min,entropy_mean,max_posterior"
    )
    .unwrap();
    for name in expert_names {
        write!(w, ",bits_{}", sanitize_name(name)).unwrap();
    }
    for name in expert_names {
        write!(w, ",time_ns_{}", sanitize_name(name)).unwrap();
    }
    writeln!(w).unwrap();

    // Data rows
    for p in points {
        write!(
            w,
            "{:.12e},{:.6},{:.6},{:.0},{:.2},",
            p.lambda, p.total_bits, p.bps, p.total_expected_time_ns, p.expected_time_per_symbol_ns
        )
        .unwrap();
        write!(
            w,
            "{},{:.2},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            p.total_wall_time_ns,
            p.wall_time_per_symbol_ns,
            p.num_symbols,
            p.neff_min,
            p.neff_mean,
            p.entropy_min,
            p.entropy_mean,
            p.max_posterior
        )
        .unwrap();
        for bits in &p.expert_bits {
            write!(w, ",{:.6}", bits).unwrap();
        }
        for time_ns in &p.expert_times_ns {
            write!(w, ",{}", time_ns).unwrap();
        }
        writeln!(w).unwrap();
    }

    w.flush().unwrap();
}

fn write_expert_csv(path: &str, evals: &[ExpertEval]) {
    create_dir_all("examples/outputs").ok();
    let file = File::create(path).expect("failed to create expert CSV");
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "name,total_bits,bps,total_time_ns,time_per_symbol_ns"
    )
    .unwrap();
    for e in evals {
        writeln!(
            w,
            "{},{:.6},{:.6},{},{}",
            e.name, e.total_bits, e.bps, e.total_time_ns, e.time_per_symbol_ns
        )
        .unwrap();
    }
    w.flush().unwrap();
}

#[derive(Clone)]
struct CandidatePoint {
    name: String,
    kind: String,
    total_bits: f64,
    bps: f64,
    total_time_ns: f64,
    time_per_symbol_ns: f64,
}

fn pareto_frontier(mut points: Vec<CandidatePoint>) -> Vec<CandidatePoint> {
    points.sort_by(|a, b| a.total_time_ns.partial_cmp(&b.total_time_ns).unwrap());
    let mut frontier = Vec::new();
    let mut best_bits = f64::INFINITY;
    for p in points {
        if p.total_bits < best_bits {
            best_bits = p.total_bits;
            frontier.push(p);
        }
    }
    frontier
}

fn write_trace_csv(
    path: &str,
    data: &[u8],
    configs: &[ExpertConfig],
    alpha: f64,
    lambda: f64,
) {
    create_dir_all("examples/outputs").ok();
    let file = File::create(path).expect("failed to create trace CSV");
    let mut w = BufWriter::new(file);

    let expert_names: Vec<String> = configs.iter().map(|e| e.name().to_string()).collect();
    let expert_names_sanitized: Vec<String> = expert_names.iter().map(|n| sanitize_name(n)).collect();

    // Header
    write!(w, "t,symbol,log_prob_plain,log_prob_aug,bits_plain,bits_aug,").unwrap();
    write!(w, "expected_time_ns,neff,entropy_bits,max_posterior,").unwrap();
    write!(w, "cum_bits,cum_expected_time_ns").unwrap();
    for name in &expert_names_sanitized {
        write!(w, ",w_{}", name).unwrap();
    }
    for name in &expert_names_sanitized {
        write!(w, ",logp_{}", name).unwrap();
    }
    for name in &expert_names_sanitized {
        write!(w, ",time_ns_{}", name).unwrap();
    }
    writeln!(w).unwrap();

    let mut mix = CostAwareSwitchingMixture::new(configs, alpha, lambda);

    for (t, &sym) in data.iter().enumerate() {
        let result = mix.step(sym);
        let m = mix.metrics();
        let post = mix.posterior();

        let bits_plain = -result.log_prob_plain / LN_2;
        let bits_aug = -result.log_prob_augmented / LN_2;

        write!(
            w,
            "{},{},{:.6},{:.6},{:.6},{:.6},",
            t, sym, result.log_prob_plain, result.log_prob_augmented, bits_plain, bits_aug
        )
        .unwrap();
        write!(
            w,
            "{:.2},{:.6},{:.6},{:.6},{:.6},{:.0}",
            result.expected_time_ns,
            result.neff,
            result.entropy_bits,
            result.max_posterior,
            m.total_bits,
            m.total_expected_time_ns
        )
        .unwrap();

        for &weight in &post {
            write!(w, ",{:.6}", weight).unwrap();
        }
        for &logp in &result.expert_log_probs {
            write!(w, ",{:.6}", logp).unwrap();
        }
        for &time_ns in &result.expert_times_ns {
            write!(w, ",{}", time_ns).unwrap();
        }
        writeln!(w).unwrap();
    }

    w.flush().unwrap();
}

fn pareto_sweep_parallel(
    configs: &[ExpertConfig],
    data: &[u8],
    alpha: f64,
    lambdas: &[f64],
) -> Vec<ParetoPoint> {
    let mut points: Vec<(usize, ParetoPoint)> = lambdas
        .par_iter()
        .enumerate()
        .map(|(idx, &lambda)| {
            let mut mix = CostAwareSwitchingMixture::new(configs, alpha, lambda);
            for &sym in data {
                mix.step(sym);
            }
            let m = mix.metrics();
            (
                idx,
                ParetoPoint {
                    lambda,
                    total_bits: m.total_bits,
                    bps: m.bps(),
                    total_expected_time_ns: m.total_expected_time_ns,
                    expected_time_per_symbol_ns: m.expected_time_per_symbol_ns(),
                    total_wall_time_ns: m.total_wall_time_ns,
                    wall_time_per_symbol_ns: m.wall_time_per_symbol_ns(),
                    num_symbols: m.num_symbols,
                    expert_bits: m.expert_cum_bits.clone(),
                    expert_times_ns: m.expert_cum_times_ns.clone(),
                    neff_min: m.neff_min,
                    neff_mean: m.neff_mean(),
                    entropy_min: m.entropy_min,
                    entropy_mean: m.entropy_mean(),
                    max_posterior: m.max_posterior,
                },
            )
        })
        .collect();
    points.sort_by_key(|(idx, _)| *idx);
    points.into_iter().map(|(_, p)| p).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut lambda_min_exp: i32 = -12;
    let mut lambda_max_exp: i32 = -6;
    let mut points_per_decade: usize = 5;
    let mut data_size: usize = 10000;
    let mut data_path: Option<String> = None;
    let mut data_limit: Option<usize> = None;
    let mut synthetic = false;
    let mut seed: u64 = 42;
    let mut alpha: f64 = 0.001;
    let mut output_path = "examples/outputs/cost_aware_pareto.csv".to_string();
    let mut trace_path: Option<String> = None;
    let mut trace_lambda: Option<f64> = None;
    let mut time_budget: Option<f64> = None;
    let mut bisection_only = false;
    let mut include_zpaq = false;
    let mut include_zpaq_extended = false;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lambda-min-exp" => {
                i += 1;
                lambda_min_exp = args[i].parse().expect("invalid lambda-min-exp");
            }
            "--lambda-max-exp" => {
                i += 1;
                lambda_max_exp = args[i].parse().expect("invalid lambda-max-exp");
            }
            "--points-per-decade" => {
                i += 1;
                points_per_decade = args[i].parse().expect("invalid points-per-decade");
            }
            "--data-size" => {
                i += 1;
                data_size = args[i].parse().expect("invalid data-size");
            }
            "--data" => {
                i += 1;
                data_path = Some(args[i].clone());
            }
            "--data-limit" => {
                i += 1;
                data_limit = Some(args[i].parse().expect("invalid data-limit"));
            }
            "--synthetic" => {
                synthetic = true;
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("invalid seed");
            }
            "--alpha" => {
                i += 1;
                alpha = args[i].parse().expect("invalid alpha");
            }
            "--output" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--trace" => {
                i += 1;
                trace_path = Some(args[i].clone());
            }
            "--trace-lambda" => {
                i += 1;
                trace_lambda = Some(args[i].parse().expect("invalid trace-lambda"));
            }
            "--time-budget" => {
                i += 1;
                time_budget = Some(args[i].parse().expect("invalid time-budget"));
            }
            "--bisection-only" => {
                bisection_only = true;
            }
            "--include-zpaq" => {
                include_zpaq = true;
            }
            "--include-zpaq-extended" => {
                include_zpaq_extended = true;
            }
            "--verbose" => {
                verbose = true;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let include_zpaq_all = include_zpaq || include_zpaq_extended;

    if verbose {
        eprintln!("Cost-Aware Mixture Experiment");
        eprintln!("  lambda_min_exp: {}", lambda_min_exp);
        eprintln!("  lambda_max_exp: {}", lambda_max_exp);
        eprintln!("  points_per_decade: {}", points_per_decade);
        eprintln!("  data_size: {}", data_size);
        if let Some(ref p) = data_path {
            eprintln!("  data: {}", p);
        }
        if let Some(limit) = data_limit {
            eprintln!("  data_limit: {}", limit);
        }
        eprintln!("  synthetic: {}", synthetic);
        eprintln!("  seed: {}", seed);
        eprintln!("  alpha: {}", alpha);
        eprintln!("  output: {}", output_path);
        if let Some(ref tp) = trace_path {
            eprintln!("  trace: {}", tp);
        }
        if let Some(tl) = trace_lambda {
            eprintln!("  trace_lambda: {}", tl);
        }
        if let Some(tb) = time_budget {
            eprintln!("  time_budget: {} ns", tb);
        }
        eprintln!("  bisection_only: {}", bisection_only);
        eprintln!("  include_zpaq: {}", include_zpaq_all);
        eprintln!("  include_zpaq_extended: {}", include_zpaq_extended);
    }

    let experts = build_experts(include_zpaq_all, include_zpaq_extended);
    let expert_names: Vec<String> = experts.iter().map(|e| e.name().to_string()).collect();
    let (data, dataset_name) = read_data(
        data_path.as_deref(),
        data_limit,
        synthetic,
        data_size,
        seed,
    );

    if verbose {
        eprintln!("  experts: {}", expert_names.join(", "));
        eprintln!("  data_len: {}", data.len());
        eprintln!("  dataset: {}", dataset_name);
    }

    let expert_evals = evaluate_experts(&data, &experts);
    let expert_csv = if output_path.ends_with(".csv") {
        output_path.replace(".csv", "_experts.csv")
    } else {
        format!("{output_path}_experts.csv")
    };
    write_expert_csv(&expert_csv, &expert_evals);

    println!("Dataset: {} ({} bytes)", dataset_name, data.len());
    println!("Experts: {}", expert_names.join(", "));
    println!("Expert baselines (standalone):");
    println!(
        "{:>24} {:>12} {:>12} {:>16} {:>16}",
        "name", "bps", "bits", "time/ns", "ns/sym"
    );
    println!("{:-<24} {:-<12} {:-<12} {:-<16} {:-<16}", "", "", "", "", "");
    for e in &expert_evals {
        println!(
            "{:>24} {:>12.4} {:>12.2} {:>16} {:>16.2}",
            e.name, e.bps, e.total_bits, e.total_time_ns, e.time_per_symbol_ns
        );
    }
    println!("Wrote expert baselines CSV: {}", expert_csv);

    if let Some(best_bits) = expert_evals
        .iter()
        .min_by(|a, b| a.total_bits.partial_cmp(&b.total_bits).unwrap())
    {
        println!(
            "Best expert by bits: {} (bps={:.4}, bits={:.2})",
            best_bits.name, best_bits.bps, best_bits.total_bits
        );
    }
    if let Some(fastest) = expert_evals
        .iter()
        .min_by(|a, b| a.total_time_ns.cmp(&b.total_time_ns))
    {
        println!(
            "Fastest expert: {} (ns/sym={:.2}, time_ns={})",
            fastest.name, fastest.time_per_symbol_ns, fastest.total_time_ns
        );
    }

    let mut pareto_points: Vec<ParetoPoint> = Vec::new();
    if !bisection_only {
        let lambdas = log_lambda_grid(lambda_min_exp, lambda_max_exp, points_per_decade);
        if verbose {
            eprintln!("Running Pareto sweep with {} λ values...", lambdas.len());
        }
        pareto_points = pareto_sweep_parallel(&experts, &data, alpha, &lambdas);
        write_pareto_csv(&output_path, &pareto_points, &expert_names);
        if verbose {
            eprintln!("Wrote {} Pareto points to: {}", pareto_points.len(), output_path);
        }
        println!("\nPareto Frontier Summary (mixture points):");
        println!(
            "{:>12} {:>12} {:>12} {:>18} {:>10} {:>10}",
            "lambda", "bits", "bps", "expected_time_ns", "Neff_min", "H_min"
        );
        println!(
            "{:-<12} {:-<12} {:-<12} {:-<18} {:-<10} {:-<10}",
            "", "", "", "", "", ""
        );
        for p in &pareto_points {
            println!(
                "{:>12.4e} {:>12.2} {:>12.4} {:>18.0} {:>10.3} {:>10.3}",
                p.lambda, p.total_bits, p.bps, p.total_expected_time_ns, p.neff_min, p.entropy_min
            );
        }
    }

    let mut bisection_point: Option<ParetoPoint> = None;
    if let Some(tb) = time_budget {
        if verbose {
            eprintln!("Running bisection for time budget {} ns...", tb);
        }
        let point = bisect_for_time_budget(
            &experts,
            &data,
            alpha,
            tb,
            10.0_f64.powi(lambda_min_exp),
            10.0_f64.powi(lambda_max_exp),
            0.05,
            20,
        );
        bisection_point = Some(point.clone());
        if bisection_only {
            write_pareto_csv(&output_path, &[point.clone()], &expert_names);
        }
        println!("\nBisection result:");
        println!("  lambda: {:.6e}", point.lambda);
        println!("  total_bits: {:.2}", point.total_bits);
        println!("  bps: {:.4}", point.bps);
        println!("  total_expected_time_ns: {:.0}", point.total_expected_time_ns);
        println!("  expected_time_per_symbol_ns: {:.2}", point.expected_time_per_symbol_ns);
        println!("  total_wall_time_ns: {}", point.total_wall_time_ns);
    }

    let mut candidates: Vec<CandidatePoint> = expert_evals
        .iter()
        .map(|e| CandidatePoint {
            name: e.name.clone(),
            kind: "expert".to_string(),
            total_bits: e.total_bits,
            bps: e.bps,
            total_time_ns: e.total_time_ns as f64,
            time_per_symbol_ns: e.time_per_symbol_ns,
        })
        .collect();
    for p in &pareto_points {
        candidates.push(CandidatePoint {
            name: format!("mix λ={:.1e}", p.lambda),
            kind: "mixture".to_string(),
            total_bits: p.total_bits,
            bps: p.bps,
            total_time_ns: p.total_expected_time_ns,
            time_per_symbol_ns: p.expected_time_per_symbol_ns,
        });
    }
    if let Some(ref p) = bisection_point {
        candidates.push(CandidatePoint {
            name: format!("mix λ={:.1e}", p.lambda),
            kind: "mixture".to_string(),
            total_bits: p.total_bits,
            bps: p.bps,
            total_time_ns: p.total_expected_time_ns,
            time_per_symbol_ns: p.expected_time_per_symbol_ns,
        });
    }

    let frontier = pareto_frontier(candidates.clone());
    println!("\nNon-dominated frontier (experts + mixtures):");
    println!(
        "{:>8} {:>22} {:>12} {:>12} {:>16} {:>16}",
        "kind", "name", "bps", "bits", "time/ns", "ns/sym"
    );
    println!("{:-<8} {:-<22} {:-<12} {:-<12} {:-<16} {:-<16}", "", "", "", "", "", "");
    for p in frontier {
        println!(
            "{:>8} {:>22} {:>12.4} {:>12.2} {:>16.0} {:>16.2}",
            p.kind, p.name, p.bps, p.total_bits, p.total_time_ns, p.time_per_symbol_ns
        );
    }

    if let Some(tb) = time_budget {
        let mut best: Option<CandidatePoint> = None;
        for p in candidates {
            if p.total_time_ns <= tb {
                match best {
                    None => best = Some(p),
                    Some(ref b) if p.total_bits < b.total_bits => best = Some(p),
                    _ => {}
                }
            }
        }
        println!("\nBest under time budget ({} ns):", tb);
        if let Some(b) = best {
            println!(
                "  {} {} | bits={:.2} bps={:.4} time_ns={:.0}",
                b.kind, b.name, b.total_bits, b.bps, b.total_time_ns
            );
        } else {
            println!("  No candidate met the time budget.");
        }
    }

    if let Some(ref tp) = trace_path {
        let lambda = trace_lambda
            .or_else(|| bisection_point.as_ref().map(|p| p.lambda))
            .unwrap_or(0.0);
        write_trace_csv(tp, &data, &experts, alpha, lambda);
        if verbose {
            eprintln!("Wrote trace to: {} (lambda={})", tp, lambda);
        }
    }
}
