//! # InfoTheory CLI
//!
//! Command-line interface for the `infotheory` library.
//! Provides access to compression-based (NCD) and entropy-based (Shannon, ROSA)
//! estimators for files.
//!
//! ## Usage
//!
//! ### Single-file mode:
//! ```bash
//! infotheory <primitive> <file1> <file2> [method/max_order]
//! ```
//!
//! ### Batch JSON mode (for programmatic use):
//! ```bash
//! infotheory batch < input.json > output.json
//! echo '{"op":"metrics","text":"hello world"}' | infotheory batch
//! ```
//!
//! See `print_usage` for details on supported primitives.

use infotheory::*;
use std::env;
use std::io::{self, BufRead, Write};

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
        _ => RateBackend::RosaPlus,
    };

    let ncd_backend = match ncd_backend {
        "rwkv7" => {
            let p = rwkv7_model_path_from_env();
            let model = load_rwkv7_model_from_path(&p);
            let coder = method.and_then(parse_rwkv7_coder).unwrap_or(rwkvzip::CoderType::AC);
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
        if rest.starts_with('"') {
            // String value
            let rest = &rest[1..];
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
        let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(rest.len());
        return rest[..end].parse().ok();
    }
    None
}

/// Extract a f64 value from JSON
fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!(r#""{}":"#, key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.').unwrap_or(rest.len());
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
                    'n' => { result.push('\n'); chars.next(); }
                    'r' => { result.push('\r'); chars.next(); }
                    't' => { result.push('\t'); chars.next(); }
                    '"' => { result.push('"'); chars.next(); }
                    '\\' => { result.push('\\'); chars.next(); }
                    _ => { result.push(c); }
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

/// Run batch mode - read JSON lines from stdin, write results to stdout
fn run_batch_mode() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    
    for line in stdin.lock().lines() {
        match line {
            Ok(input) => {
                let result = process_json_line(&input);
                writeln!(stdout, "{}", result).ok();
                stdout.flush().ok();
            }
            Err(_) => break,
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let primitive = &args[1];
    
    // Batch JSON mode for programmatic use
    if primitive == "batch" {
        run_batch_mode();
        return;
    }

    if args.len() < 3 {
        print_usage();
        return;
    }

    match primitive.as_str() {
        // Entropy-based primitives
        "ned"
        | "nte"
        | "tvd"
        | "nhd"
        | "mi"
        | "mutual_info"
        | "ce"
        | "conditional_entropy"
        | "xe"
        | "cross_entropy"
        | "joint_entropy"
        | "h_xy"
        | "kl"
        | "kl_divergence"
        | "js"
        | "js_divergence" => {
            if args.len() < 4 {
                eprintln!("Error: '{}' requires two files.", primitive);
                std::process::exit(1);
            }
            let file1 = &args[2];
            let file2 = &args[3];
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

            let mut rate_backend = "rosaplus";
            let mut ncd_backend = "zpaq";
            let mut method: Option<&str> = None;
            let mut i = 5usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--rate-backend" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --rate-backend requires a value (rwkv7|rosaplus)");
                            std::process::exit(1);
                        });
                        rate_backend = parse_rate_backend(v).unwrap_or_else(|| {
                            eprintln!("Error: invalid --rate-backend '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    "--ncd-backend" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --ncd-backend requires a value (rwkv7|zpaq)");
                            std::process::exit(1);
                        });
                        ncd_backend = parse_ncd_backend(v).unwrap_or_else(|| {
                            eprintln!("Error: invalid --ncd-backend '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    "--method" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --method requires a value");
                            std::process::exit(1);
                        });
                        method = Some(v.as_str());
                    }
                    other => {
                        eprintln!("Error: unknown flag '{}'", other);
                        print_usage();
                        std::process::exit(1);
                    }
                }
                i += 1;
            }

            let ctx = build_ctx(rate_backend, ncd_backend, method);

            match primitive.as_str() {
                "ned" => {
                    let (bx, by) = rayon::join(|| read_file(file1), || read_file(file2));
                    println!("{}", ctx.ned_bytes(&bx, &by, max_order))
                }
                "nte" => {
                    let (bx, by) = rayon::join(|| read_file(file1), || read_file(file2));
                    println!("{}", ctx.nte_bytes(&bx, &by, max_order))
                }
                "tvd" => println!("{}", tvd_paths(file1, file2, max_order)),
                "nhd" => println!("{}", nhd_paths(file1, file2, max_order)),
                "mi" | "mutual_info" => {
                    let (bx, by) = rayon::join(|| read_file(file1), || read_file(file2));
                    if max_order == 0 {
                        println!("{}", mutual_information_bytes(&bx, &by, 0));
                    } else {
                        println!("{}", ctx.mutual_information_rate_bytes(&bx, &by, max_order));
                    }
                }
                "ce" | "conditional_entropy" => {
                    println!("{}", conditional_entropy_paths(file1, file2, max_order))
                }
                "xe" | "cross_entropy" => {
                    println!("{}", cross_entropy_paths(file1, file2, max_order))
                }
                "kl" | "kl_divergence" => println!("{}", kl_divergence_paths(file1, file2)),
                "js" | "js_divergence" => println!("{}", js_divergence_paths(file1, file2)),
                "joint_entropy" | "h_xy" => {
                    let (bx, by) = rayon::join(|| read_file(file1), || read_file(file2));
                    if max_order == 0 {
                        println!("{}", joint_marginal_entropy_bytes(&bx, &by));
                    } else {
                        println!("{}", ctx.joint_entropy_rate_bytes(&bx, &by, max_order));
                    }
                }
                _ => unreachable!(),
            }
        }

        "ned_cons" => {
            if args.len() < 4 {
                eprintln!("Error: 'ned_cons' requires two files.");
                std::process::exit(1);
            }
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let bx = read_file(&args[2]);
            let by = read_file(&args[3]);
            println!("{}", ned_cons_bytes(&bx, &by, max_order));
        }

        "entropy" | "h" | "entropy_rate" | "h_rate" => {
            let data = read_file(&args[2]);
            let default_order = if primitive.contains("rate") { 8 } else { 0 };
            let max_order = args
                .get(3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_order);

            if max_order == 0 && !primitive.contains("rate") {
                println!("{}", marginal_entropy_bytes(&data));
            } else {
                println!("{}", entropy_rate_bytes(&data, max_order));
            }
        }

        "id" | "intrinsic_dep" => {
            let max_order = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(-1);
            let data = read_file(&args[2]);
            let h_marginal = marginal_entropy_bytes(&data);
            let h_rate = entropy_rate_bytes(&data, max_order);
            let redundancy = if h_marginal < 1e-9 {
                0.0
            } else {
                ((h_marginal - h_rate) / h_marginal).clamp(0.0, 1.0)
            };
            println!(
                "{:.6} [Internal Redundancy] (Ĥ: {:.4}, H_marg: {:.4})",
                redundancy, h_rate, h_marginal
            );
        }

        "rt" | "resistance" => {
            if args.len() < 4 {
                eprintln!("Error: 'rt' requires two files (original and transformed).");
                std::process::exit(1);
            }
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(-1);
            let bx = read_file(&args[2]);
            let btx = read_file(&args[3]);
            println!(
                "{}",
                resistance_to_transformation_bytes(&bx, &btx, max_order)
            );
        }

        "ncd" | "ncd_vitanyi" | "ncd_sym" | "ncd_sym_vitanyi" | "ncd_cons" | "ncd_sym_cons" => {
            if args.len() < 4 {
                eprintln!("Error: NCD primitives require two files.");
                std::process::exit(1);
            }
            let file1 = &args[2];
            let file2 = &args[3];
            let mut ncd_backend = "zpaq";
            let mut method: Option<&str> = args.get(4).map(|s| s.as_str());
            let mut i = 5usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--ncd-backend" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --ncd-backend requires a value (rwkv7|zpaq)");
                            std::process::exit(1);
                        });
                        ncd_backend = parse_ncd_backend(v).unwrap_or_else(|| {
                            eprintln!("Error: invalid --ncd-backend '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    "--method" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --method requires a value");
                            std::process::exit(1);
                        });
                        method = Some(v.as_str());
                    }
                    other => {
                        eprintln!("Error: unknown flag '{}'", other);
                        print_usage();
                        std::process::exit(1);
                    }
                }
                i += 1;
            }

            let ctx = build_ctx("rosaplus", ncd_backend, method);
            match primitive.as_str() {
                "ncd" | "ncd_vitanyi" => {
                    println!("{}", ncd_paths_backend(file1, file2, &ctx.ncd_backend, NcdVariant::Vitanyi))
                }
                "ncd_sym" | "ncd_sym_vitanyi" => {
                    println!("{}", ncd_paths_backend(file1, file2, &ctx.ncd_backend, NcdVariant::SymVitanyi))
                }
                "ncd_cons" => println!("{}", ncd_paths_backend(file1, file2, &ctx.ncd_backend, NcdVariant::Cons)),
                "ncd_sym_cons" => println!("{}", ncd_paths_backend(file1, file2, &ctx.ncd_backend, NcdVariant::SymCons)),
                _ => unreachable!(),
            }
        }

        "search" => {
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
            let mut rate_backend = "rosaplus";
            let mut ncd_backend = "zpaq";
            let mut method: Option<&str> = None;
            let mut i = 4usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--level" | "--granularity" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --level requires a value (snippet|file)");
                            std::process::exit(1);
                        });
                        opts.granularity = match v.as_str() {
                            "snippet" => search::SearchGranularity::Snippet,
                            "file" => search::SearchGranularity::File,
                            _ => {
                                eprintln!("Error: invalid --level '{}'; expected snippet|file", v);
                                std::process::exit(1);
                            }
                        };
                    }
                    "--prior" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --prior requires a path (file or directory)");
                            std::process::exit(1);
                        });
                        opts.universal_prior = Some(v.clone());
                    }
                    "--stage2-prior" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --stage2-prior requires a value (full|off|summarize)");
                            std::process::exit(1);
                        });
                        opts.stage2_prior_mode = match v.as_str() {
                            "full" => search::Stage2PriorMode::UsePrior,
                            "off" | "none" | "false" => search::Stage2PriorMode::NoPrior,
                            "summarize" | "summary" => search::Stage2PriorMode::SummarizePrior,
                            _ => {
                                eprintln!("Error: invalid --stage2-prior '{}'; expected full|off|summarize", v);
                                std::process::exit(1);
                            }
                        };
                    }
                    "--max-order" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --max-order requires an integer");
                            std::process::exit(1);
                        });
                        opts.max_order = v.parse().unwrap_or_else(|_| {
                            eprintln!("Error: invalid --max-order '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    "--top-k" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --top-k requires an integer");
                            std::process::exit(1);
                        });
                        opts.top_k = v.parse().unwrap_or_else(|_| {
                            eprintln!("Error: invalid --top-k '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    "--method" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --method requires a value");
                            std::process::exit(1);
                        });
                        method = Some(v.as_str());
                    }
                    "--stage0-frac" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --stage0-frac requires a float in [0,1]");
                            std::process::exit(1);
                        });
                        opts.stage0_keep_frac = v.parse().unwrap_or_else(|_| {
                            eprintln!("Error: invalid --stage0-frac '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    "--rate-backend" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --rate-backend requires a value (rwkv7|rosaplus)");
                            std::process::exit(1);
                        });
                        rate_backend = parse_rate_backend(v).unwrap_or_else(|| {
                            eprintln!("Error: invalid --rate-backend '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    "--ncd-backend" => {
                        i += 1;
                        let v = args.get(i).unwrap_or_else(|| {
                            eprintln!("Error: --ncd-backend requires a value (rwkv7|zpaq)");
                            std::process::exit(1);
                        });
                        ncd_backend = parse_ncd_backend(v).unwrap_or_else(|| {
                            eprintln!("Error: invalid --ncd-backend '{}'", v);
                            std::process::exit(1);
                        });
                    }
                    other => {
                        eprintln!("Error: unknown search flag '{}'", other);
                        print_usage();
                        std::process::exit(1);
                    }
                }
                i += 1;
            }

            opts.ctx = build_ctx(rate_backend, ncd_backend, method);

            if opts.stage2_prior_mode == search::Stage2PriorMode::SummarizePrior
                && opts.universal_prior.is_none()
            {
                eprintln!("Error: --stage2-prior summarize requires --prior");
                std::process::exit(1);
            }

            search::run_search_with_options(query, target, &opts);
        }

        _ => {
            eprintln!("Unknown primitive: {}", primitive);
            print_usage();
        }
    }
}

fn print_usage() {
    eprintln!("Usage: infotheory <primitive> <file1> <file2> [method/max_order]");
    eprintln!("       infotheory search <query> <target_path> [--level snippet|file] [--prior <path>] [--stage2-prior full|off|summarize]");
    eprintln!("                              [--max-order <i64>] [--top-k <n>] [--method <method>] [--stage0-frac <f64>]");
    eprintln!();
    eprintln!("=== BATCH JSON MODE (for programmatic use) ===");
    eprintln!("  infotheory batch        Read JSON lines from stdin, write results to stdout");
    eprintln!();
    eprintln!("  Supported operations:");
    eprintln!("    metrics       {{\"op\":\"metrics\",\"text\":\"...\",\"max_order\":-1}}");
    eprintln!("                  Returns: {{\"h0\":...,\"h_rate\":...,\"id\":...,\"len\":...}}");
    eprintln!();
    eprintln!("    batch_metrics {{\"op\":\"batch_metrics\",\"texts\":[\"...\",\"...\"],\"max_order\":-1}}");
    eprintln!("                  Returns: {{\"results\":[{{...}},{{...}}]}}");
    eprintln!();
    eprintln!("    ncd           {{\"op\":\"ncd\",\"text1\":\"...\",\"text2\":\"...\",\"method\":\"5\",\"variant\":\"vitanyi\"}}");
    eprintln!("                  Returns: {{\"ncd\":...}}");
    eprintln!();
    eprintln!("    rosa_dist     {{\"op\":\"rosa_dist\",\"text1\":\"...\",\"text2\":\"...\",\"max_order\":-1}}");
    eprintln!("                  Returns: {{\"rosa_dist\":...}}  (faster than NCD)");
    eprintln!();
    eprintln!("    ncd_matrix    {{\"op\":\"ncd_matrix\",\"texts\":[...],\"method\":\"5\"}}");
    eprintln!("                  Returns: {{\"matrix\":[[...],...],\"n\":...}}");
    eprintln!();
    eprintln!("    rosa_matrix   {{\"op\":\"rosa_matrix\",\"texts\":[...],\"max_order\":-1}}");
    eprintln!("                  Returns: {{\"matrix\":[[...],...],\"n\":...}}  (faster)");
    eprintln!();
    eprintln!("    spam_check    {{\"op\":\"spam_check\",\"text\":\"...\",\"h0_min\":1.0,\"h_rate_min\":0.5,\"id_max\":0.95}}");
    eprintln!("                  Returns: {{\"pass\":true/false,\"reason\":\"...\",\"h0\":...}}");
    eprintln!();
    eprintln!("=== FILE-BASED MODE ===");
    eprintln!();
    eprintln!("Compression-based (NCD via ZPAQ):");
    eprintln!("  ncd, ncd_vitanyi       NCD Vitanyi formula");
    eprintln!("  ncd_sym, ncd_sym_vitanyi  Symmetric NCD Vitanyi");
    eprintln!("  ncd_cons               NCD Conservative");
    eprintln!("  ncd_sym_cons           Symmetric NCD Conservative");
    eprintln!("  [method]: ZPAQ method (default: \"5\"), e.g. \"1\", \"5\", \"x4.3ci1\"");
    eprintln!();
    eprintln!("Entropy-based (dispatch: max_order=0 for Marginal, !=0 for Rate):");
    eprintln!("  ned                    Normalized Entropy Distance");
    eprintln!("  ned_cons               NED Conservative");
    eprintln!("  nte                    Normalized Transform Effort (VI)");
    eprintln!("  tvd                    Total Variation Distance (Marginal only)");
    eprintln!("  nhd                    Normalized Hellinger Distance (Marginal only)");
    eprintln!();
    eprintln!("Information measures:");
    eprintln!("  entropy, h             Shannon entropy H(X) (marginal if no order)");
    eprintln!("  entropy_rate, h_rate   Unbiased predictive entropy rate (ROSA)");
    eprintln!("  joint_entropy, h_xy    Joint entropy H(X,Y) (uses max_order if provided)");
    eprintln!("  mi, mutual_info        Mutual info I(X;Y) (uses max_order if provided)");
    eprintln!("  ce, conditional_entropy Conditional entropy H(X|Y) (uses max_order if provided)");
    eprintln!("  xe, cross_entropy      Cross-entropy H(P,Q)");
    eprintln!("  kl, kl_divergence      KL Divergence D_KL(P||Q) (marginal only)");
    eprintln!("  js, js_divergence      JS Divergence JSD(P||Q) (marginal only)");
    eprintln!();
    eprintln!("Structural measures (Primitive 6 & 7):");
    eprintln!("  id, intrinsic_dep      Intrinsic Dependence (Redundancy Ratio)");
    eprintln!("  rt, resistance         Resistance to transformation");
    eprintln!();
    eprintln!("  [max_order]: Markov context depth (default: -1 [unlimited], 0 for Marginal)");
}
