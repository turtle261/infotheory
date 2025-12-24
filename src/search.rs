use infotheory::{cross_entropy_rate_bytes, entropy_rate_bytes};
use rayon::prelude::*;
use rosaplus::RosaPlus;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Snippet {
    pub path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub content: Vec<u8>,
    pub score: f64, 
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SearchGranularity {
    Snippet,
    File,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Stage2PriorMode {
    /// Use the (full or summarized) universal prior as a prefix for compression metrics.
    UsePrior,
    /// Do NOT use the universal prior in Stage 2 (pure NCD/KMI rerank on Stage-1-filtered set).
    NoPrior,
    /// Summarize the universal prior via an inner prior-less search over the prior corpus.
    SummarizePrior,
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub granularity: SearchGranularity,
    /// Universal prior corpus path (file or directory). If set:
    /// - Stage 1 always uses it.
    /// - Stage 2 uses it by default (unless Stage2PriorMode::NoPrior).
    pub universal_prior: Option<String>,
    pub stage2_prior_mode: Stage2PriorMode,
    pub max_order: i64,
    pub top_k: usize,
    pub zpaq_method: String,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            granularity: SearchGranularity::Snippet,
            universal_prior: None,
            stage2_prior_mode: Stage2PriorMode::UsePrior,
            max_order: 8,
            top_k: 50,
            zpaq_method: "5".to_string(),
        }
    }
}

pub fn run_search(query: &str, target_path: &str) {
    run_search_with_options(query, target_path, &SearchOptions::default());
}

pub fn run_search_with_options(query: &str, target_path: &str, opts: &SearchOptions) {
    let debug = std::env::var("DEBUG_SEARCH").is_ok();
    let query_bytes = resolve_query_bytes(query);
    if query_bytes.is_empty() {
        eprintln!("Error: Query is empty.");
        return;
    }

    if debug {
        println!(
            "Scanning target: {} (granularity={:?}, prior={}, stage2_prior_mode={:?})",
            target_path,
            opts.granularity,
            opts.universal_prior.as_deref().unwrap_or("<none>"),
            opts.stage2_prior_mode
        );
    }

    let candidates = collect_candidates(target_path, opts.granularity);
    if candidates.is_empty() {
        eprintln!("No accessible files found in target '{}'.", target_path);
        return;
    }
    if debug {
        println!("Found {} candidates. Filtering...", candidates.len());
    }

    // Stage 1: Filter
    let mut scored_candidates = if let Some(prior_path) = opts.universal_prior.as_deref() {
        stage1_filter_with_universal_prior(&query_bytes, prior_path, candidates, opts)
    } else {
        stage1_filter_no_prior(&query_bytes, candidates, opts)
    };

    scored_candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_k_size = opts.top_k.min(scored_candidates.len());
    let top_candidates = &mut scored_candidates[0..top_k_size];
    if debug {
        println!("Reranking top {} candidates with Kolmogorov Mutual Information...", top_k_size);
    }

    // Stage 2: Rerank
    stage2_rerank_kmi(&query_bytes, top_candidates, opts);
    top_candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, snippet) in top_candidates.iter().take(5).enumerate() {
        if debug {
            println!("Rank {}: Score={:.6}, Path={}", i + 1, snippet.score, snippet.path.display());
        }
        println!(
            "sed -n '{},{}p' {}",
            snippet.start_line,
            snippet.end_line,
            snippet.path.display()
        );
    }
}

fn resolve_query_bytes(query: &str) -> Vec<u8> {
    let p = Path::new(query);
    if p.exists() && fs::metadata(p).map(|m| m.is_file()).unwrap_or(false) {
        fs::read(p).unwrap_or_else(|_| query.as_bytes().to_vec())
    } else {
        query.as_bytes().to_vec()
    }
}

fn stage1_filter_no_prior(query_bytes: &[u8], candidates: Vec<Snippet>, opts: &SearchOptions) -> Vec<Snippet> {
    let h_q = entropy_rate_bytes(query_bytes, opts.max_order);

    let scored: Vec<Snippet> = candidates
        .into_par_iter()
        .map(|mut snippet| {
            let h_q_x = cross_entropy_rate_bytes(query_bytes, &snippet.content, opts.max_order);
            snippet.score = h_q - h_q_x;
            snippet
        })
        .collect();

    // Keep equivalence with old behavior by not clamping.
    scored
}

fn stage1_filter_with_universal_prior(
    query_bytes: &[u8],
    prior_path: &str,
    candidates: Vec<Snippet>,
    opts: &SearchOptions,
) -> Vec<Snippet> {
    // PERFORMANCE NOTE:
    // Training the prior using snippet-level windows would duplicate overlapping content
    // and explode runtime. We *always* train/load the prior at file granularity.
    let mut base = load_or_train_prior_model(prior_path, opts);
    base.ensure_lm_built_no_finalize_endpos();

    // Precompute query codepoints once (cross_entropy() would allocate this per call).
    let query_cps: Vec<u32> = query_bytes.iter().map(|&b| b as u32).collect();
    let h_u_q = base.cross_entropy_cps(&query_cps);

    // Stage-1 scoring relative to the prior baseline:
    // score(x) = H_U(q) - H_x(q)
    // (how much better the candidate model predicts q than the universal background)
    candidates
        .into_par_iter()
        .map(|mut snippet| {
            let mut m = RosaPlus::new(opts.max_order, false, 0, 42);
            m.train_example(&snippet.content);
            m.ensure_lm_built_no_finalize_endpos();
            let h_x_q = m.cross_entropy_cps(&query_cps);
            snippet.score = h_u_q - h_x_q;
            snippet
        })
        .collect()
}

fn stage2_rerank_kmi(query_bytes: &[u8], top_candidates: &mut [Snippet], opts: &SearchOptions) {
    let prior_prefix: Option<Vec<u8>> = match (opts.universal_prior.as_deref(), opts.stage2_prior_mode) {
        (None, _) => None,
        (Some(_), Stage2PriorMode::NoPrior) => None,
        (Some(prior_path), Stage2PriorMode::UsePrior) => Some(corpus_bytes(prior_path, SearchGranularity::File)),
        (Some(prior_path), Stage2PriorMode::SummarizePrior) => Some(summarize_prior_for_query(query_bytes, prior_path, opts)),
    };

    let method = opts.zpaq_method.as_str();

    let cq = if let Some(prefix) = prior_prefix.as_deref() {
        let mut pq = Vec::with_capacity(prefix.len() + query_bytes.len());
        pq.extend_from_slice(prefix);
        pq.extend_from_slice(query_bytes);
        zpaq_rs::compress_size(&pq, method).unwrap_or(0)
    } else {
        zpaq_rs::compress_size(query_bytes, method).unwrap_or(0)
    };

    top_candidates.par_iter_mut().for_each(|snippet| {
        let (cx, joint1, joint2) = if let Some(prefix) = prior_prefix.as_deref() {
            let mut px = Vec::with_capacity(prefix.len() + snippet.content.len());
            px.extend_from_slice(prefix);
            px.extend_from_slice(&snippet.content);

            let mut j1 = Vec::with_capacity(prefix.len() + snippet.content.len() + query_bytes.len());
            j1.extend_from_slice(prefix);
            j1.extend_from_slice(&snippet.content);
            j1.extend_from_slice(query_bytes);

            let mut j2 = Vec::with_capacity(prefix.len() + snippet.content.len() + query_bytes.len());
            j2.extend_from_slice(prefix);
            j2.extend_from_slice(query_bytes);
            j2.extend_from_slice(&snippet.content);

            (zpaq_rs::compress_size(&px, method).unwrap_or(0), j1, j2)
        } else {
            let mut j1 = Vec::with_capacity(snippet.content.len() + query_bytes.len());
            j1.extend_from_slice(&snippet.content);
            j1.extend_from_slice(query_bytes);

            let mut j2 = Vec::with_capacity(snippet.content.len() + query_bytes.len());
            j2.extend_from_slice(query_bytes);
            j2.extend_from_slice(&snippet.content);

            (zpaq_rs::compress_size(&snippet.content, method).unwrap_or(0), j1, j2)
        };

        let (c1, c2) = rayon::join(
            || zpaq_rs::compress_size(&joint1, method).unwrap_or(u64::MAX),
            || zpaq_rs::compress_size(&joint2, method).unwrap_or(u64::MAX),
        );
        let c_joint = c1.min(c2);
        snippet.score = if c_joint == u64::MAX {
            0.0
        } else {
            (cq as f64 + cx as f64 - c_joint as f64).max(0.0)
        };
    });
}

fn summarize_prior_for_query(query_bytes: &[u8], prior_path: &str, opts: &SearchOptions) -> Vec<u8> {
    // Prior-less search inside the prior corpus itself.
    // We approximate K(q|x) via conditional compression: min(C(xq),C(qx)) - C(x), and select the MIN.
    let candidates = collect_candidates(prior_path, opts.granularity);
    if candidates.is_empty() {
        return Vec::new();
    }

    let method = opts.zpaq_method.as_str();
    let cq = zpaq_rs::compress_size(query_bytes, method).unwrap_or(0);

    let mut best: Option<(f64, Vec<u8>)> = None;
    for c in candidates {
        let cx = zpaq_rs::compress_size(&c.content, method).unwrap_or(0);

        let mut xq = Vec::with_capacity(c.content.len() + query_bytes.len());
        xq.extend_from_slice(&c.content);
        xq.extend_from_slice(query_bytes);
        let mut qx = Vec::with_capacity(c.content.len() + query_bytes.len());
        qx.extend_from_slice(query_bytes);
        qx.extend_from_slice(&c.content);

        let (cxq, cqx) = rayon::join(
            || zpaq_rs::compress_size(&xq, method).unwrap_or(u64::MAX),
            || zpaq_rs::compress_size(&qx, method).unwrap_or(u64::MAX),
        );
        let c_joint = cxq.min(cqx);
        if c_joint == u64::MAX {
            continue;
        }
        // Conditional complexity proxy.
        let k_q_given_x = (c_joint as f64 - cx as f64).max(0.0);
        // Tie-breaker: if equal, prefer smaller candidate.
        let candidate_key = (k_q_given_x, cx as f64, cq as f64);
        let is_better = match &best {
            None => true,
            Some((best_k, best_bytes)) => {
                let best_cx = zpaq_rs::compress_size(best_bytes, method).unwrap_or(0) as f64;
                (candidate_key.0, candidate_key.1) < (*best_k, best_cx)
            }
        };
        if is_better {
            best = Some((k_q_given_x, c.content));
        }
    }

    best.map(|(_, b)| b).unwrap_or_default()
}

fn train_rosa_on_corpus(m: &mut RosaPlus, corpus_path: &str, granularity: SearchGranularity) {
    // Train incrementally on each candidate to avoid giant concatenations.
    for c in collect_candidates(corpus_path, granularity) {
        if !c.content.is_empty() {
            m.train_example(&c.content);
        }
    }
}

fn prior_cache_path(prior_path: &str, max_order: i64) -> Option<PathBuf> {
    let home = std::env::var("XDG_CACHE_HOME")
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{}/.cache", h)));
    let cache_root = match home {
        Some(h) => PathBuf::from(h).join("infotheory").join("rosa_prior"),
        None => return None,
    };

    let mut hasher = DefaultHasher::new();
    // Cache format/version (bump when training or serialization semantics change).
    (2u32).hash(&mut hasher);
    prior_path.hash(&mut hasher);
    max_order.hash(&mut hasher);
    // file-granularity is baked into the cache key (we always use it for prior training)
    ("file" as &str).hash(&mut hasher);
    let key = hasher.finish();
    Some(cache_root.join(format!("prior_{:016x}.rosa", key)))
}

fn load_or_train_prior_model(prior_path: &str, opts: &SearchOptions) -> RosaPlus {
    // Load cached prior model if present.
    if let Some(cache_path) = prior_cache_path(prior_path, opts.max_order) {
        if let Some(parent) = cache_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if cache_path.exists() {
            if let Ok(m) = RosaPlus::load(cache_path.to_string_lossy().as_ref()) {
                return m;
            }
        }

        // Train + save.
        let mut m = RosaPlus::new(opts.max_order, false, 0, 42);
        train_rosa_on_corpus(&mut m, prior_path, SearchGranularity::File);
        // Build LM once so the saved model is the full state.
        m.build_lm_no_finalize_endpos();
        let _ = m.save(cache_path.to_string_lossy().as_ref());
        return m;
    }

    // Fallback: no cache location available.
    let mut m = RosaPlus::new(opts.max_order, false, 0, 42);
    train_rosa_on_corpus(&mut m, prior_path, SearchGranularity::File);
    m
}

fn corpus_bytes(corpus_path: &str, granularity: SearchGranularity) -> Vec<u8> {
    // Compression prior prefix requires a concrete byte buffer.
    // We join candidates with a simple delimiter to preserve boundaries.
    let mut out = Vec::new();
    for c in collect_candidates(corpus_path, granularity) {
        if c.content.is_empty() {
            continue;
        }
        out.extend_from_slice(&c.content);
        out.extend_from_slice(b"\n\n");
    }
    out
}

fn collect_candidates(target: &str, granularity: SearchGranularity) -> Vec<Snippet> {
    let mut snippets = Vec::new();
    let path = Path::new(target);
    
    if path.exists() {
        if path.is_file() {
            snippets.extend(file_to_candidates(path, granularity));
        } else if path.is_dir() {
            visit_dirs(path, &mut snippets, granularity);
        }
    }
    
    snippets
}

fn visit_dirs(dir: &Path, snippets: &mut Vec<Snippet>, granularity: SearchGranularity) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name() {
                        if let Some(name_str) = name.to_str() {
                            if !name_str.starts_with('.') {
                                visit_dirs(&path, snippets, granularity);
                            }
                        }
                    }
                } else {
                    snippets.extend(file_to_candidates(&path, granularity));
                }
            }
        }
    }
}

fn file_to_candidates(path: &Path, granularity: SearchGranularity) -> Vec<Snippet> {
    let mut snippets = Vec::new();
    
    // Only process text files
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy();
        if matches!(ext_str.as_ref(), "o" | "a" | "so" | "dll" | "exe" | "bin" | "png" | "jpg" | "zip" | "gz") {
            return snippets;
        }
    }

    match granularity {
        SearchGranularity::File => {
            if let Ok(bytes) = fs::read(path) {
                if !bytes.is_empty() {
                    // Best-effort line count for `sed` output.
                    let lines = bytes.iter().filter(|&&b| b == b'\n').count() + 1;
                    snippets.push(Snippet {
                        path: path.to_path_buf(),
                        start_line: 1,
                        end_line: lines.max(1),
                        content: bytes,
                        score: 0.0,
                    });
                }
            }
        }
        SearchGranularity::Snippet => {
            if let Ok(file) = fs::File::open(path) {
                let reader = io::BufReader::new(file);
                let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();

                if lines.is_empty() {
                    return snippets;
                }

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

                    if end == lines.len() {
                        break;
                    }
                    i += stride;
                }
            }
        }
    }
    snippets
}
