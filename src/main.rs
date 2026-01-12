//! # InfoTheory CLI
//!
//! Command-line interface for the `infotheory` library.
//! Provides access to compression-based (NCD) and entropy-based (Shannon, ROSA)
//! estimators for files.

use infotheory::aixi::agent::{Agent, AgentConfig};
use infotheory::aixi::environment::{
    BiasedRockPaperScissor, CoinFlip, CtwTest, Environment, ExtendedTiger, KuhnPoker, TicTacToe,
};
use infotheory::*;
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use std::sync::Arc;

mod search;

fn rwkv7_model_path_from_env() -> String {
    env::var("RWKV7_MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("Error: RWKV7_MODEL_PATH env var must be set when using rwkv7 backends");
        std::process::exit(1);
    })
}

fn parse_rate_backend(v: &str) -> Option<&'static str> {
    match v {
        "rosaplus" | "rosa" => Some("rosaplus"),
        "rwkv7" | "rwkv" => Some("rwkv7"),
        "ctw" => Some("ctw"),
        _ => None,
    }
}

fn parse_ncd_backend(v: &str) -> Option<&'static str> {
    match v {
        "zpaq" => Some("zpaq"),
        "rwkv7" | "rwkv" => Some("rwkv7"),
        _ => None,
    }
}

fn parse_rwkv7_coder(v: &str) -> Option<rwkvzip::CoderType> {
    match v {
        "ac" | "AC" => Some(rwkvzip::CoderType::AC),
        "rans" | "RANS" | "rANS" => Some(rwkvzip::CoderType::RANS),
        _ => None,
    }
}

fn build_ctx(rate_backend: &str, ncd_backend: &str, method: Option<&str>) -> InfotheoryCtx {
    let rate_backend = match rate_backend {
        "rwkv7" => {
            let p = rwkv7_model_path_from_env();
            let model = load_rwkv7_model_from_path(&p);
            RateBackend::Rwkv7 { model }
        }
        "ctw" => {
            let depth = if let Some(m) = method {
                m.parse::<usize>().unwrap_or(20)
            } else {
                20
            };
            RateBackend::Ctw { depth }
        }
        _ => RateBackend::RosaPlus,
    };

    let ncd_backend = match ncd_backend {
        "rwkv7" => {
            let p = rwkv7_model_path_from_env();
            let model = load_rwkv7_model_from_path(&p);
            let coder = method
                .and_then(parse_rwkv7_coder)
                .unwrap_or(rwkvzip::CoderType::AC);
            NcdBackend::Rwkv7 { model, coder }
        }
        _ => {
            let m = method.unwrap_or("5").to_string();
            NcdBackend::Zpaq { method: m }
        }
    };

    InfotheoryCtx::new(rate_backend, ncd_backend)
}

fn read_file(path: &str) -> Vec<u8> {
    match std::fs::read(path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            std::process::exit(1);
        }
    }
}

// Extractors for manual JSON parsing (simple versions)
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":"#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        let rest = rest.trim_start();
        if rest.starts_with('"') {
            let rest = &rest[1..];
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
    }
    None
}

fn extract_json_i64(json: &str, key: &str) -> Option<i64> {
    let pattern = format!(r#""{}":"#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        let rest = rest.trim_start();
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(rest.len());
        return rest[..end].parse().ok();
    }
    None
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!(r#""{}":"#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        let rest = rest.trim_start();
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
            .unwrap_or(rest.len());
        return rest[..end].parse().ok();
    }
    None
}

fn extract_json_array(json: &str, key: &str) -> Vec<String> {
    let pattern = format!(r#""{}":["#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        if let Some(end) = rest.find(']') {
            let array_content = &rest[..end];
            return array_content
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    Vec::new()
}

fn process_json_line(line: &str) -> String {
    let op = extract_json_string(line, "op").unwrap_or_default();
    match op.as_str() {
        "metrics" => {
            let text = extract_json_string(line, "text").unwrap_or_default();
            let max_order = extract_json_i64(line, "max_order").unwrap_or(-1);
            let data = text.as_bytes();
            let h0 = marginal_entropy_bytes(data);
            let h_rate = entropy_rate_bytes(data, max_order);
            let id = if h0 < 1e-9 {
                0.0
            } else {
                ((h0 - h_rate) / h0).clamp(0.0, 1.0)
            };
            format!(
                r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                h0,
                h_rate,
                id,
                data.len()
            )
        }
        "ncd" => {
            let t1 = extract_json_string(line, "text1").unwrap_or_default();
            let t2 = extract_json_string(line, "text2").unwrap_or_default();
            let method = extract_json_string(line, "method").unwrap_or_else(|| "5".to_string());
            let ncd = ncd_bytes(t1.as_bytes(), t2.as_bytes(), &method, NcdVariant::Vitanyi);
            format!(r#"{{"ncd":{:.6}}}"#, ncd)
        }
        _ => format!(r#"{{"error":"unknown op: {}"}}"#, op),
    }
}

fn run_batch_mode() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(l) = line {
            println!("{}", process_json_line(&l));
        }
    }
}

fn run_aixi_mode(config_path: &str) -> anyhow::Result<()> {
    let mut file = File::open(config_path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: serde_json::Value = serde_json::from_str(&content)?;

    let config = AgentConfig {
        algorithm: v["algorithm"].as_str().unwrap_or("ctw").to_string(),
        ct_depth: v["ct_depth"].as_u64().unwrap_or(20) as usize,
        agent_horizon: v["agent_horizon"].as_u64().unwrap_or(3) as usize,
        observation_bits: v["observation_bits"].as_u64().unwrap_or(1) as usize,
        reward_bits: v["reward_bits"].as_u64().unwrap_or(1) as usize,
        agent_actions: v["agent_actions"].as_u64().unwrap_or(2) as usize,
        num_simulations: v["num_simulations"].as_u64().unwrap_or(50) as usize,
        exploration_exploitation_ratio: v["exploration_exploitation_ratio"].as_f64().unwrap_or(1.4),
        rwkv_model_path: v["rwkv_model_path"].as_str().map(|s| s.to_string()),
        rosa_max_order: v["rosa_max_order"].as_u64().map(|n| n as i64),
    };

    let env_name = v["environment"].as_str().unwrap_or("coin-flip");
    let mut env: Box<dyn Environment> = match env_name {
        "coin-flip" => Box::new(CoinFlip::new(0.9)),
        "ctw-test" | "ctwtest" => Box::new(CtwTest::new()),
        "extended-tiger" => Box::new(ExtendedTiger::new()),
        "tictactoe" => Box::new(TicTacToe::new()),
        "biased-rock-paper-scissor" => Box::new(BiasedRockPaperScissor::new()),
        "kuhn-poker" => Box::new(KuhnPoker::new()),
        _ => return Err(anyhow::anyhow!("Unknown environment: {}", env_name)),
    };

    let mut agent = Agent::new(config);
    println!(
        "Agent initialized with {} algorithm for {} environment.",
        v["algorithm"].as_str().unwrap_or("ctw"),
        env_name
    );

    let cycles = v["terminate-lifetime"].as_u64().unwrap_or(20) as usize;
    let mut total_reward = 0;
    let mut prev_action = 0;
    let mut obs = env.get_observation();
    let mut rew = env.get_reward();

    for t in 0..cycles {
        println!("Cycle {}: Obs={}, Rew={}", t, obs, rew);
        agent.model_update_percept(obs, rew);
        total_reward += rew;
        let action = agent.get_planned_action(obs, rew, prev_action);
        println!("Cycle {}: Planned Action={}", t, action);
        agent.model_update_action_external(action);
        env.perform_action(action);
        obs = env.get_observation();
        rew = env.get_reward();
        prev_action = action;
    }
    println!("Total Reward: {}", total_reward);
    Ok(())
}

fn search_command(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Error: 'search' requires query and target path.");
        std::process::exit(1);
    }
    let query = &args[2];
    let target = &args[3];

    let mut opts = search::SearchOptions::default();
    let mut rate_backend = "rosaplus".to_string();
    let mut ncd_backend = "zpaq".to_string();
    let mut method: Option<String> = None;

    let mut i = 4usize;
    while i < args.len() {
        match args[i].as_str() {
            "--level" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --level requires snippet|file");
                opts.granularity = if v == "snippet" {
                    search::SearchGranularity::Snippet
                } else {
                    search::SearchGranularity::File
                };
            }
            "--prior" => {
                i += 1;
                opts.universal_prior = args.get(i).cloned();
            }
            "--max-order" => {
                i += 1;
                opts.max_order = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "--top-k" => {
                i += 1;
                opts.top_k = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--rate-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend requires a value");
                rate_backend = parse_rate_backend(v).unwrap_or("rosaplus").to_string();
            }
            "--method" => {
                i += 1;
                method = args.get(i).cloned();
            }
            _ => {
                i += 1;
            }
        }
        i += 1;
    }
    opts.ctx = build_ctx(&rate_backend, &ncd_backend, method.as_deref());
    search::run_search_with_options(query, target, &opts);
}

trait OptionExt<T> {
    fn unwrap_or_exit(self, msg: &str) -> T;
}
impl<T> OptionExt<T> for Option<T> {
    fn unwrap_or_exit(self, msg: &str) -> T {
        self.unwrap_or_else(|| {
            eprintln!("{}", msg);
            std::process::exit(1);
        })
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check for help flag early
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        print_usage();
        return;
    }

    if args.len() < 2 {
        print_usage();
        return;
    }

    let primitive = &args[1];
    if primitive == "batch" {
        run_batch_mode();
        return;
    }

    // Common positional and flag parsing
    let mut file1: Option<String> = None;
    let mut file2: Option<String> = None;
    let mut pos_arg3: Option<String> = None;
    let mut flags_start = 2usize;

    if primitive != "search" && primitive != "aixi" {
        if let Some(f1) = args.get(2) {
            if !f1.starts_with('-') {
                file1 = Some(f1.clone());
                flags_start = 3;
            }
        }
        if let Some(f2) = args.get(3) {
            if !f2.starts_with('-') {
                file2 = Some(f2.clone());
                flags_start = 4;
            }
        }
        if let Some(a3) = args.get(4) {
            if !a3.starts_with('-') {
                pos_arg3 = Some(a3.clone());
                flags_start = 5;
            }
        }
    }

    let mut rate_backend_str = "rosaplus".to_string();
    let mut ncd_backend_str = "zpaq".to_string();
    let mut method_str: Option<String> = None;
    let mut rate_backend_specified = false;

    let mut i = flags_start;
    while i < args.len() {
        match args[i].as_str() {
            "--rate-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend requires a value");
                rate_backend_str = parse_rate_backend(v).unwrap_or("rosaplus").to_string();
                rate_backend_specified = true;
            }
            "--ncd-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --ncd-backend requires a value");
                ncd_backend_str = parse_ncd_backend(v).unwrap_or("zpaq").to_string();
            }
            "--method" => {
                i += 1;
                method_str = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }

    let ctx = build_ctx(&rate_backend_str, &ncd_backend_str, method_str.as_deref());
    set_default_ctx(ctx.clone());

    match primitive.as_str() {
        "aixi" => {
            if let Some(p) = args.get(2) {
                if let Err(e) = run_aixi_mode(p) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            } else {
                eprintln!("Error: 'aixi' requires config.json");
                std::process::exit(1);
            }
        }
        "search" => search_command(&args),
        "ncd" | "ncd_vitanyi" | "ncd_sym" | "ncd_sym_vitanyi" | "ncd_cons" | "ncd_sym_cons" => {
            let f1 = file1.unwrap_or_exit("Error: NCD requires two files");
            let f2 = file2.unwrap_or_exit("Error: NCD requires two files");
            let method = pos_arg3.or(method_str).unwrap_or_else(|| "5".to_string());
            let variant = match primitive.as_str() {
                "ncd_sym" | "ncd_sym_vitanyi" => NcdVariant::SymVitanyi,
                "ncd_cons" => NcdVariant::Cons,
                "ncd_sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };
            println!("{}", ncd_paths_backend(&f1, &f2, &ctx.ncd_backend, variant));
        }
        "entropy" | "h" | "entropy_rate" | "h_rate" => {
            let f1 = file1.unwrap_or_exit("Error: 'h' requires a file");
            let default_order = if primitive.contains("rate") || rate_backend_specified {
                -1
            } else {
                0
            };
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_order);
            let data = read_file(&f1);
            if max_order == 0 && !primitive.contains("rate") && !rate_backend_specified {
                println!("{}", marginal_entropy_bytes(&data));
            } else {
                println!("{}", entropy_rate_bytes(&data, max_order));
            }
        }
        "id" | "intrinsic_dep" => {
            let f1 = file1.unwrap_or_exit("Error: 'id' requires a file");
            let max_order = pos_arg3.and_then(|s| s.parse().ok()).unwrap_or(-1);
            println!(
                "{:.6}",
                intrinsic_dependence_bytes(&read_file(&f1), max_order)
            );
        }
        other => {
            let f1 = file1.unwrap_or_exit("Error: requires two files");
            let f2 = file2.unwrap_or_exit("Error: requires two files");
            let default_order = if rate_backend_specified { -1 } else { 0 };
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_order);
            let b1 = read_file(&f1);
            let b2 = read_file(&f2);
            let res = match other {
                "ned" => ned_bytes(&b1, &b2, max_order),
                "ned_cons" => ned_cons_bytes(&b1, &b2, max_order),
                "nte" => nte_bytes(&b1, &b2, max_order),
                "mi" | "mutual_info" => mutual_information_bytes(&b1, &b2, max_order),
                "ce" | "conditional_entropy" => conditional_entropy_bytes(&b1, &b2, max_order),
                "xe" | "cross_entropy" => cross_entropy_bytes(&b1, &b2, max_order),
                "joint_entropy" | "h_xy" => {
                    if max_order == 0 {
                        joint_marginal_entropy_bytes(&b1, &b2)
                    } else {
                        joint_entropy_rate_bytes(&b1, &b2, max_order)
                    }
                }
                "rt" | "resistance" => resistance_to_transformation_bytes(&b1, &b2, max_order),
                "tvd" => tvd_paths(&f1, &f2, max_order),
                "nhd" => nhd_paths(&f1, &f2, max_order),
                "kl" | "kl_divergence" => kl_divergence_paths(&f1, &f2),
                "js" | "js_divergence" => js_divergence_paths(&f1, &f2),
                _ => {
                    eprintln!("Unknown primitive: {}", other);
                    print_usage();
                    return;
                }
            };
            println!("{}", res);
        }
    }
}

fn print_usage() {
    eprintln!(
        r#"InfoTheory CLI
Usage: infotheory <primitive> [args...] [options]

Primitives:
  Entropy & Information:
    h, entropy <file> [max_order]           Entropy (marginal if order=0, rate if >0)
    h_rate, entropy_rate <file> [max_order] Force entropy rate estimation
    mi, mutual_info <f1> <f2> [max_order]   Mutual Information I(X;Y)
    xe, cross_entropy <f1> <f2> [max_order] Cross Entropy H(X,Y) - H(Y)? (Check def)
    ce, conditional_entropy <f1> <f2>       Conditional Entropy H(X|Y)
    joint_entropy, h_xy <f1> <f2>           Joint Entropy H(X,Y)
    id, intrinsic_dep <file> [max_order]    Intrinsic Dependence

  Distance & Divergence:
    ncd <f1> <f2> [method]                  Normalized Compression Distance (Vitanyi)
    ncd_sym, ncd_cons                       NCD variants (Symmetric, Consistent, etc.)
    ned <f1> <f2> [max_order]               Normalized Entropy Distance
    nte <f1> <f2> [max_order]               Normalized Transform Effort
    kl, kl_divergence <f1> <f2>             Kullback-Leibler Divergence
    js, js_divergence <f1> <f2>             Jensen-Shannon Divergence
    tvd <f1> <f2>                           Total Variation Distance
    nhd <f1> <f2>                           Normalized Hellinger Distance
    rt, resistance <f1> <f2>                Resistance to Transformation

  Tools:
    search <query> <target> [options]       Search target using info-theoretic ranking
    aixi <config.json>                      Run AIXI agent
    batch                                   Run in JSON-L batch mode

Options:
  --rate-backend <name>   Backend for rate estimation: 'rosaplus' (default), 'ctw', 'rwkv7'
  --ncd-backend <name>    Backend for NCD: 'zpaq' (default), 'rwkv7'
  --method <val>          Method/Depth parameter (e.g. '5' for zpaq, '16' for ctw)

Examples:
  infotheory ncd file1.txt file2.txt --ncd-backend zpaq --method 5
  infotheory h file.txt --rate-backend ctw --method 32
  infotheory search "encryption" ./src --prior "codebase context"
"#
    );
}
