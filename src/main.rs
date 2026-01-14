//! # InfoTheory CLI
//!
//! Command-line interface for the `infotheory` library.
//! Provides access to compression-based (NCD) and entropy-based (Shannon, ROSA, CTW)
//! estimators for files, as well as AIXI agents.
//!
//! ## Usage
//!
//! ### Single-file mode:
//! ```bash
//! infotheory <primitive> <file1> <file2> [method/max_order]
//! ```
//!
//! ### Search mode:
//! ```bash
//! infotheory search <query> <target> [options]
//! ```
//!
//! ### AIXI Agent mode:
//! ```bash
//! infotheory aixi <config.json>
//! ```
//!
//! ### Batch JSON mode (for programmatic use):
//! ```bash
//! infotheory batch < input.json > output.json
//! echo '{"op":"metrics","text":"hello world"}' | infotheory batch
//! ```
//!
//! See `print_usage` for details on supported primitives.

use infotheory::aixi::agent::{Agent, AgentConfig};
use infotheory::aixi::environment::{
    BiasedRockPaperScissor, CoinFlip, CtwTest, Environment, ExtendedTiger, KuhnPoker,
    ProcessEnvironment, TicTacToe,
};
use infotheory::*;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read};

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
        "fac-ctw" | "facctw" => Some("fac-ctw"),
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
        "fac-ctw" => {
            let depth = if let Some(m) = method {
                m.parse::<usize>().unwrap_or(20)
            } else {
                20
            };
            RateBackend::FacCtw {
                base_depth: depth,
                num_percept_bits: 8, // Default for byte-oriented CLI
                encoding_bits: 8,    // Default for byte-oriented CLI
            }
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

// ============================================================
// Batch JSON Mode - For programmatic use from Python
// ============================================================

/// ROSA-based symmetric codelength distance (NCD-like but faster)
/// d_ROSA(x,y) = 0.5 * (H_y(x)/H_x(x) + H_x(y)/H_y(y)) - 1
/// Clamped to [0, 1]
fn rosa_distance(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 1.0;
    }

    // Self-entropy rates (biased/plugin estimator for consistency)
    let h_x_x = biased_entropy_rate_bytes(x, max_order);
    let h_y_y = biased_entropy_rate_bytes(y, max_order);

    // Cross-entropy rates
    let h_y_x = cross_entropy_rate_bytes(x, y, max_order); // score x under model trained on y
    let h_x_y = cross_entropy_rate_bytes(y, x, max_order); // score y under model trained on x

    // Avoid division by zero
    if h_x_x < 1e-9 || h_y_y < 1e-9 {
        return 1.0;
    }

    let d = 0.5 * (h_y_x / h_x_x + h_x_y / h_y_y) - 1.0;
    d.clamp(0.0, 1.0)
}

/// Process a single JSON line and return result
fn process_json_line(line: &str) -> String {
    // Parse JSON manually to avoid serde dependency
    let line = line.trim();
    if line.is_empty() {
        return r#"{"error":"empty input"}"#.to_string();
    }

    // Extract operation type
    let op = extract_json_string(line, "op").unwrap_or_default();

    match op.as_str() {
        "metrics" => {
            // Single text metrics: H0, H_rate, ID
            let text = extract_json_string(line, "text").unwrap_or_default();
            let max_order = extract_json_i64(line, "max_order").unwrap_or(-1);
            let data = text.as_bytes();

            if data.is_empty() {
                return r#"{"error":"empty text"}"#.to_string();
            }

            let h0 = marginal_entropy_bytes(data);
            let h_rate = entropy_rate_bytes(data, max_order);
            let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };

            format!(
                r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                h0, h_rate, id, data.len()
            )
        }

        "metrics_file" => {
            // File-based metrics
            let path = extract_json_string(line, "path").unwrap_or_default();
            let max_order = extract_json_i64(line, "max_order").unwrap_or(-1);

            match std::fs::read(&path) {
                Ok(data) => {
                    let h0 = marginal_entropy_bytes(&data);
                    let h_rate = entropy_rate_bytes(&data, max_order);
                    let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };

                    format!(
                        r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                        h0, h_rate, id, data.len()
                    )
                }
                Err(e) => format!(r#"{{"error":"failed to read file: {}"}}"#, e),
            }
        }

        "ncd" => {
            // NCD between two texts
            let text1 = extract_json_string(line, "text1").unwrap_or_default();
            let text2 = extract_json_string(line, "text2").unwrap_or_default();
            let method = extract_json_string(line, "method").unwrap_or_else(|| "5".to_string());
            let variant = extract_json_string(line, "variant").unwrap_or_else(|| "vitanyi".to_string());

            let x = text1.as_bytes();
            let y = text2.as_bytes();

            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            let ncd_variant = match variant.as_str() {
                "sym" | "sym_vitanyi" => NcdVariant::SymVitanyi,
                "cons" => NcdVariant::Cons,
                "sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };

            let ncd = ncd_bytes(x, y, &method, ncd_variant);
            format!(r#"{{"ncd":{:.6}}}"#, ncd)
        }
        "ncd_files" => {
            // NCD between two files
            let path1 = extract_json_string(line, "path1").unwrap_or_default();
            let path2 = extract_json_string(line, "path2").unwrap_or_default();
            let method = extract_json_string(line, "method").unwrap_or_else(|| "5".to_string());
            let variant = extract_json_string(line, "variant").unwrap_or_else(|| "vitanyi".to_string());

            let ncd_variant = match variant.as_str() {
                "sym" | "sym_vitanyi" => NcdVariant::SymVitanyi,
                "cons" => NcdVariant::Cons,
                "sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };

            let ncd = ncd_paths(&path1, &path2, &method, ncd_variant);
            format!(r#"{{"ncd":{:.6}}}"#, ncd)
        }

        "rosa_dist" => {
            // ROSA-based distance (faster than NCD)
            let text1 = extract_json_string(line, "text1").unwrap_or_default();
            let text2 = extract_json_string(line, "text2").unwrap_or_default();
            let max_order = extract_json_i64(line, "max_order").unwrap_or(-1);

            let x = text1.as_bytes();
            let y = text2.as_bytes();

            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            let dist = rosa_distance(x, y, max_order);
            format!(r#"{{"rosa_dist":{:.6}}}"#, dist)
        }

        "cross_entropy" => {
            // Cross-entropy H_y(x) - score x under model trained on y
            let text_x = extract_json_string(line, "text_x").unwrap_or_default();
            let text_y = extract_json_string(line, "text_y").unwrap_or_default();
            let max_order = extract_json_i64(line, "max_order").unwrap_or(-1);

            let x = text_x.as_bytes();
            let y = text_y.as_bytes();

            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            let xe = cross_entropy_rate_bytes(x, y, max_order);
            format!(r#"{{"cross_entropy":{:.6}}}"#, xe)
        }
        "batch_metrics" => {
            // Batch metrics for multiple texts
            let texts = extract_json_array(line, "texts");
            let max_order = extract_json_i64(line, "max_order").unwrap_or(-1);

            let results: Vec<String> = texts.iter().map(|text| {
                let data = text.as_bytes();
                if data.is_empty() {
                    r#"{"h0":0,"h_rate":0,"id":0,"len":0}"#.to_string()
                } else {
                    let h0 = marginal_entropy_bytes(data);
                    let h_rate = entropy_rate_bytes(data, max_order);
                    let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };
                    format!(
                        r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                        h0, h_rate, id, data.len()
                    )
                }
            }).collect();

            format!(r#"{{"results":[{}]}}"#, results.join(","))
        }

        "ncd_matrix" => {
            // NCD matrix for multiple texts (for diversity/clustering)
            let texts = extract_json_array(line, "texts");
            let method = extract_json_string(line, "method").unwrap_or_else(|| "5".to_string());
            let variant = extract_json_string(line, "variant").unwrap_or_else(|| "vitanyi".to_string());

            let ncd_variant = match variant.as_str() {
                "sym" | "sym_vitanyi" => NcdVariant::SymVitanyi,
                "cons" => NcdVariant::Cons,
                "sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };

            let datas: Vec<Vec<u8>> = texts.iter().map(|t| t.as_bytes().to_vec()).collect();
            let matrix = ncd_matrix_bytes(&datas, &method, ncd_variant);
            let n = datas.len();

            // Format as row-major array of arrays
            let rows: Vec<String> = (0..n).map(|i| {
                let row: Vec<String> = (0..n).map(|j| format!("{:.6}", matrix[i * n + j])).collect();
                format!("[{}]", row.join(","))
            }).collect();

            format!(r#"{{"matrix":[{}],"n":{}}}"#, rows.join(","), n)
        }
        "rosa_matrix" => {
            // ROSA distance matrix (faster than NCD matrix)
            let texts = extract_json_array(line, "texts");
            let max_order = extract_json_i64(line, "max_order").unwrap_or(-1);

            let n = texts.len();
            let datas: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();

            // Compute matrix (symmetric)
            let mut matrix = vec![0.0f64; n * n];
            for i in 0..n {
                for j in i..n {
                    let d = if i == j {
                        0.0
                    } else {
                        rosa_distance(datas[i], datas[j], max_order)
                    };
                    matrix[i * n + j] = d;
                    matrix[j * n + i] = d;
                }
            }

            // Format as row-major array of arrays
            let rows: Vec<String> = (0..n).map(|i| {
                let row: Vec<String> = (0..n).map(|j| format!("{:.6}", matrix[i * n + j])).collect();
                format!("[{}]", row.join(","))
            }).collect();

            format!(r#"{{"matrix":[{}],"n":{}}}"#, rows.join(","), n)
        }
        "spam_check" => {
            // Quick spam/quality check for a single text
            let text = extract_json_string(line, "text").unwrap_or_default();
            let h0_threshold = extract_json_f64(line, "h0_min").unwrap_or(1.0);
            let h_rate_threshold = extract_json_f64(line, "h_rate_min").unwrap_or(0.5);
            let id_threshold = extract_json_f64(line, "id_max").unwrap_or(0.95);
            let min_len = extract_json_i64(line, "min_len").unwrap_or(10) as usize;

            let data = text.as_bytes();
            let len = data.len();

            if len < min_len {
                return format!(r#"{{"pass":false,"reason":"too_short","len":{}}}"#, len);
            }

            let h0 = marginal_entropy_bytes(data);
            if h0 < h0_threshold {
                return format!(r#"{{"pass":false,"reason":"low_entropy","h0":{:.4}}}"#, h0);
            }

            let h_rate = entropy_rate_bytes(data, -1);
            if h_rate < h_rate_threshold {
                return format!(r#"{{"pass":false,"reason":"low_entropy_rate","h_rate":{:.4}}}"#, h_rate);
            }

            let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };
            if id > id_threshold {
                return format!(r#"{{"pass":false,"reason":"high_redundancy","id":{:.4}}}"#, id);
            }

            format!(r#"{{"pass":true,"h0":{:.4},"h_rate":{:.4},"id":{:.4},"len":{}}}"#, h0, h_rate, id, len)
        }
        "help" => {
            r#"{"ops":["metrics","metrics_file","ncd","ncd_files","rosa_dist","cross_entropy","batch_metrics","ncd_matrix","rosa_matrix","spam_check"]}"#.to_string()
        }
        _ => {
            format!(r#"{{"error":"unknown op: {}"}}"#, op)
        }
    }
}

/// Extract a string value from JSON (simple parser, no serde needed)
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":"#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        // Optimized scanning for quote
        if let Some(start_quote) = rest.find('"') {
            let rest = &rest[start_quote + 1..];
            let mut end = 0;
            let mut escaped = false;
            for (i, c) in rest.char_indices() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if c == '\\' {
                    escaped = true;
                    continue;
                }
                if c == '"' {
                    end = i;
                    break;
                }
            }
            return Some(unescape_json_string(&rest[..end]));
        }
    }
    None
}

/// Extract an i64 value from JSON
fn extract_json_i64(json: &str, key: &str) -> Option<i64> {
    let pattern = format!(r#""{}":"#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        // Skip potential whitespace/quotes if any (though standard JSON number doesn't have quotes)
        // Adjust for simple numeric find
        let rest = rest.trim_start_matches(|c| c == ':' || c == ' ' || c == '"');
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(rest.len());
        // Simple trim in case we consumed quotes incorrectly?
        // Let's assume valid JSON input
        return rest[..end].parse().ok();
    }
    None
}

/// Extract a f64 value from JSON
fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!(r#""{}":"#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        let rest = rest.trim_start_matches(|c| c == ':' || c == ' ' || c == '"');
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
            .unwrap_or(rest.len());
        return rest[..end].parse().ok();
    }
    None
}

/// Extract a string array from JSON
fn extract_json_array(json: &str, key: &str) -> Vec<String> {
    let pattern = format!(r#""{}":["#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        // Find matching ]
        let mut depth = 1;
        let mut end = 0;
        for (i, c) in rest.char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let array_content = &rest[..end];
        // Parse strings from array
        let mut results = Vec::new();
        let mut in_string = false;
        let mut escaped = false;
        let mut current = String::new();

        for c in array_content.chars() {
            if escaped {
                current.push(c);
                escaped = false;
                continue;
            }
            match c {
                '\\' if in_string => {
                    escaped = true;
                    current.push(c);
                }
                '"' => {
                    if in_string {
                        results.push(unescape_json_string(&current));
                        current.clear();
                    }
                    in_string = !in_string;
                }
                _ if in_string => {
                    current.push(c);
                }
                _ => {}
            }
        }
        return results;
    }
    Vec::new()
}

/// Unescape JSON string
fn unescape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                match next {
                    'n' => {
                        result.push('\n');
                        chars.next();
                    }
                    'r' => {
                        result.push('\r');
                        chars.next();
                    }
                    't' => {
                        result.push('\t');
                        chars.next();
                    }
                    '"' => {
                        result.push('"');
                        chars.next();
                    }
                    '\\' => {
                        result.push('\\');
                        chars.next();
                    }
                    _ => {
                        result.push(c);
                    }
                }
            } else {
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }
    result
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
        "external" => {
            let ext = &v["external_config"];
            let cmd = ext["command"].as_str().unwrap_or("/bin/bash");
            let args: Vec<String> = ext["args"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|a| a.as_str().unwrap_or_default().to_string())
                .collect();
            let actions: Vec<String> = ext["actions"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|a| a.as_str().unwrap_or_default().to_string())
                .collect();
            let pattern = ext["reward_pattern"].as_str().map(|s| s.to_string());
            let step_cost = ext["step_cost"].as_u64().unwrap_or(1);
            let debug_mode = ext["verbose"].as_bool().unwrap_or(false);

            Box::new(ProcessEnvironment::new(
                cmd,
                &args,
                actions,
                config.observation_bits,
                config.reward_bits,
                pattern,
                step_cost,
                debug_mode,
            )?)
        }
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

    // Preserve the legacy behavior (and avoid extra parsing work) when no flags are given.
    if args.len() == 4 {
        search::run_search(query, target);
        return;
    }

    let mut opts = search::SearchOptions::default();
    let mut rate_backend = "rosaplus".to_string();
    let ncd_backend = "zpaq".to_string();
    let mut method: Option<String> = None;
    let mut stage2_prior_mode: Option<search::Stage2PriorMode> = None;

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
            "--stage2-prior-mode" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    stage2_prior_mode = match v.as_str() {
                        "none" | "no-prior" => Some(search::Stage2PriorMode::NoPrior),
                        "summarize" | "summarize-prior" => {
                            Some(search::Stage2PriorMode::SummarizePrior)
                        }
                        "use" | "use-prior" | _ => Some(search::Stage2PriorMode::UsePrior),
                    };
                }
            }
            _ => {
                i += 1;
            }
        }
        i += 1;
    }
    if let Some(mode) = stage2_prior_mode {
        opts.stage2_prior_mode = mode;
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
            let _method = pos_arg3.or(method_str).unwrap_or_else(|| "5".to_string());
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
  --rate-backend <name>   Backend for rate estimation: 'rosaplus' (default), 'ctw', 'fac-ctw', 'rwkv7'
  --ncd-backend <name>    Backend for NCD: 'zpaq' (default), 'rwkv7'
  --method <val>          Method/Depth parameter (e.g. '5' for zpaq, '16' for ctw)

Examples:
  infotheory ncd file1.txt file2.txt --ncd-backend zpaq --method 5
  infotheory h file.txt --rate-backend ctw --method 32
  infotheory search "encryption" ./src --prior "codebase context"
"#
    );
}
