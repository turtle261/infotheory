use infotheory::{cross_entropy_rate_bytes, entropy_rate_bytes};
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Snippet {
    pub path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub content: Vec<u8>,
    pub score: f64, 
}

pub fn run_search(query: &str, target_path: &str) {
    let debug = std::env::var("DEBUG_SEARCH").is_ok();
    // 1. Resolve Query
    let query_bytes = if Path::new(query).exists() && fs::metadata(query).map(|m| m.is_file()).unwrap_or(false) {
        fs::read(query).unwrap_or_else(|_| query.as_bytes().to_vec())
    } else {
        query.as_bytes().to_vec()
    };

    if query_bytes.is_empty() {
        eprintln!("Error: Query is empty.");
        return;
    }

    // 2. Collect Candidates (Snippets)
    if debug { println!("Scanning target: {}", target_path); }
    let candidates = collect_snippets(target_path);
    if candidates.is_empty() {
        eprintln!("No accessible files found in target '{}'.", target_path);
        return;
    }

    if debug { println!("Found {} snippets. Filtering...", candidates.len()); }

    // 3. Stage 1: Filter (Mutual Information)
    // We want to maximize I(q; x) = H(q) - H(q|x)
    // This removes binary blobs that predict 'q' well purely due to structure, 
    // but don't share semantic information with 'q'.
    let max_order = 8;
    let h_q = entropy_rate_bytes(&query_bytes, max_order);

    let mut scored_candidates: Vec<Snippet> = candidates
        .into_par_iter()
        .map(|mut snippet| {
             let h_q_x = cross_entropy_rate_bytes(&query_bytes, &snippet.content, max_order);
             // Mutual Information I(q; x) = H(q) - H(q|x)
             // We want to MAXIMIZE this.
             snippet.score = h_q - h_q_x;
             snippet
        })
        .collect();

    // Sort descending (higher Mutual Information is better)
    scored_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    if debug {
        println!("Top 5 after ROSA Mutual Information filter (H(q)={:.4}):", h_q);
        for i in 0..5.min(scored_candidates.len()) {
            println!("  {}. {} (I(q;x): {:.4}) - {}", i+1, scored_candidates[i].path.display(), scored_candidates[i].score, 
                String::from_utf8_lossy(&scored_candidates[i].content).lines().next().unwrap_or(""));
        }
    }

    // Keep top K (e.g., 50)
    let top_k_size = 50.min(scored_candidates.len());
    let top_candidates = &mut scored_candidates[0..top_k_size];

    if debug { println!("Reranking top {} candidates with Kolmogorov Mutual Information...", top_k_size); }

    // 4. Stage 2: Rerank (Kolmogorov Mutual Information)
    // I(q; x) = C(q) + C(x) - C(qx)
    // Since C(q) is constant, maximizing I(q; x) is maximizing C(x) - C(qx)
    // We use Symmetric ZPAQ (min(C(xq), C(qx))) to ensure the best compression is found.
    
    // Pre-calculate C(q)
    let cq = zpaq_rs::compress_size(&query_bytes, "5").unwrap_or(0);

    top_candidates.par_iter_mut().for_each(|snippet| {
        let cx = zpaq_rs::compress_size(&snippet.content, "5").unwrap_or(0);
        
        let mut joint1 = Vec::with_capacity(snippet.content.len() + query_bytes.len());
        joint1.extend_from_slice(&snippet.content);
        joint1.extend_from_slice(&query_bytes);
        
        let mut joint2 = Vec::with_capacity(snippet.content.len() + query_bytes.len());
        joint2.extend_from_slice(&query_bytes);
        joint2.extend_from_slice(&snippet.content);
        
        let (cxq, cqx) = rayon::join(
            || zpaq_rs::compress_size(&joint1, "5").unwrap_or(u64::MAX),
            || zpaq_rs::compress_size(&joint2, "5").unwrap_or(u64::MAX),
        );
        
        let c_joint = cxq.min(cqx);
        
        // I(q; x) = C(q) + C(x) - C(qx)
        if c_joint == u64::MAX {
            snippet.score = 0.0;
        } else {
            snippet.score = (cq as f64 + cx as f64 - c_joint as f64).max(0.0);
        }
    });

    // Sort descending (higher Mutual Information is better)
    top_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // 5. Output
    for (i, snippet) in top_candidates.iter().take(5).enumerate() {
        if debug {
            println!("Rank {}: MI={:.4}, Path={}", i+1, snippet.score, snippet.path.display());
        }
        println!("sed -n '{},{}p' {}", snippet.start_line, snippet.end_line, snippet.path.display());
    }
}

fn collect_snippets(target: &str) -> Vec<Snippet> {
    let mut snippets = Vec::new();
    let path = Path::new(target);
    
    if path.exists() {
        if path.is_file() {
            snippets.extend(file_to_snippets(path));
        } else if path.is_dir() {
            visit_dirs(path, &mut snippets);
        }
    }
    
    snippets
}

fn visit_dirs(dir: &Path, snippets: &mut Vec<Snippet>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name() {
                        if let Some(name_str) = name.to_str() {
                            if !name_str.starts_with('.') {
                                visit_dirs(&path, snippets);
                            }
                        }
                    }
                } else {
                    snippets.extend(file_to_snippets(&path));
                }
            }
        }
    }
}

fn file_to_snippets(path: &Path) -> Vec<Snippet> {
    let mut snippets = Vec::new();
    
    // Only process text files
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy();
        if matches!(ext_str.as_ref(), "o" | "a" | "so" | "dll" | "exe" | "bin" | "png" | "jpg" | "zip" | "gz") {
            return snippets;
        }
    }

    if let Ok(file) = fs::File::open(path) {
        let reader = io::BufReader::new(file);
        let lines: Vec<String> = reader
            .lines()
            .filter_map(Result::ok)
            .collect();

        if lines.is_empty() { return snippets; }

        let window = 50;
        let stride = 20;
        
        let mut i = 0;
        while i < lines.len() {
            let end = (i + window).min(lines.len());
            let chunk_lines = &lines[i..end];
            let content = chunk_lines.join("\n").into_bytes();
            
            if content.len() > 50 {
                snippets.push(Snippet {
                    path: path.to_path_buf(),
                    start_line: i + 1,
                    end_line: end,
                    content,
                    score: 0.0,
                });
            }
            
            if end == lines.len() { break; }
            i += stride;
        }
    }
    snippets
}
