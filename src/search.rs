use infotheory::rosaplus::RosaPlus;
use infotheory::{InfotheoryCtx, RateBackend, cross_entropy_bytes, marginal_entropy_bytes};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Snippet {
    pub path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub content: Vec<u8>,
    pub score: f64,
}

fn stage0_prefilter(
    query_bytes: &[u8],
    mut candidates: Vec<Snippet>,
    opts: &SearchOptions,
    debug: bool,
) -> Vec<Snippet> {
    let n = candidates.len();
    if n == 0 {
        return candidates;
    }

    let frac = opts.stage0_keep_frac.clamp(0.0, 1.0);
    if frac >= 1.0 {
        return candidates;
    }

    // Option A: Unigram (i.i.d.) likelihood-gain proxy.
    // score0(x) = H0(Q) - H0(Q|X)
    // where H0(Q|X) is computed as cross-entropy of Q under X's unigram model.
    let h0_q = marginal_entropy_bytes(query_bytes);
    candidates.par_iter_mut().for_each(|s| {
        let h0_q_x = cross_entropy_bytes(query_bytes, &s.content, 0);
        s.score = h0_q - h0_q_x;
    });

    let mut keep = ((n as f64) * frac).ceil() as usize;
    keep = keep.max(opts.top_k).min(n);
    if keep < n {
        let nth = keep.saturating_sub(1);
        candidates.select_nth_unstable_by(nth, |a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(keep);
    }

    if debug {
        println!(
            "Stage-0 prefilter kept {}/{} candidates (frac={:.4})",
            candidates.len(),
            n,
            frac
        );
    }

    candidates
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

#[derive(Clone)]
pub struct SearchOptions {
    pub granularity: SearchGranularity,
    /// Universal prior corpus path (file or directory). If set:
    /// - Stage 1 always uses it.
    /// - Stage 2 uses it by default (unless Stage2PriorMode::NoPrior).
    pub universal_prior: Option<String>,
    pub stage2_prior_mode: Stage2PriorMode,
    pub max_order: i64,
    pub top_k: usize,
    pub stage0_keep_frac: f64,
    pub ctx: InfotheoryCtx,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            granularity: SearchGranularity::Snippet,
            universal_prior: None,
            stage2_prior_mode: Stage2PriorMode::UsePrior,
            max_order: 8,
            top_k: 50,
            stage0_keep_frac: 0.2,
            ctx: InfotheoryCtx::with_zpaq("5"),
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

    let candidates = stage0_prefilter(query_bytes.as_slice(), candidates, opts, debug);
    if candidates.is_empty() {
        eprintln!("No candidates remain after Stage-0 prefilter.");
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

    let top_k_size = opts.top_k.min(scored_candidates.len());
    if top_k_size < scored_candidates.len() {
        let nth = top_k_size.saturating_sub(1);
        scored_candidates.select_nth_unstable_by(nth, |a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_candidates.truncate(top_k_size);
    }

    scored_candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_candidates = &mut scored_candidates[..top_k_size];
    if debug {
        println!(
            "Reranking top {} candidates with Kolmogorov Mutual Information...",
            top_k_size
        );
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
            println!(
                "Rank {}: Score={:.6}, Path={}",
                i + 1,
                snippet.score,
                snippet.path.display()
            );
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

fn stage1_filter_no_prior(
    query_bytes: &[u8],
    candidates: Vec<Snippet>,
    opts: &SearchOptions,
) -> Vec<Snippet> {
    let h_q = opts.ctx.entropy_rate_bytes(query_bytes, opts.max_order);

    let scored: Vec<Snippet> = candidates
        .into_par_iter()
        .map(|mut snippet| {
            let h_q_x =
                opts.ctx
                    .cross_entropy_rate_bytes(query_bytes, &snippet.content, opts.max_order);
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
    if !matches!(opts.ctx.rate_backend, RateBackend::RosaPlus) {
        let prior_prefix = corpus_bytes(prior_path, SearchGranularity::File);
        let h_u_q = opts
            .ctx
            .cross_entropy_conditional_chain(&[prior_prefix.as_slice()], query_bytes);
        return candidates
            .into_par_iter()
            .map(|mut snippet| {
                let h_ux_q = opts.ctx.cross_entropy_conditional_chain(
                    &[prior_prefix.as_slice(), snippet.content.as_slice()],
                    query_bytes,
                );
                snippet.score = h_u_q - h_ux_q;
                snippet
            })
            .collect();
    }

    // PERFORMANCE NOTE:
    // Training the prior using snippet-level windows would duplicate overlapping content
    // and explode runtime. We *always* train/load the prior at file granularity.
    let mut base = load_or_train_prior_model(prior_path, opts);
    // For true conditional updates we require the fixed 256-byte alphabet LM.
    // This ensures symbol indices remain stable across incremental updates.
    base.ensure_lm_built_no_finalize_endpos();
    // Reduce the cost of cloning `base` per worker.
    base.shrink_aux_buffers();

    // Precompute query codepoints once (cross_entropy() would allocate this per call).
    let query_cps: Vec<u32> = query_bytes.iter().map(|&b| b as u32).collect();
    let h_u_q = base.cross_entropy_cps(&query_cps);

    // True conditional update:
    // score(x) = H_U(q) - H_{U+x}(q)
    // by applying a reversible candidate update to the *full* prior model.
    //
    // MEMORY NOTE:
    // `map_init(|| base.clone(), ...)` clones the model once per Rayon worker.
    // For large priors this can blow up RSS. We cap worker count based on an estimate
    // of model bytes and best-effort available memory (Linux).
    let model_bytes = base.estimated_size_bytes().max(1);
    let threads = memory_aware_threads(model_bytes);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");

    pool.install(|| {
        candidates
            .into_par_iter()
            .map_init(
                || base.clone(),
                |m, mut snippet| {
                    let mut tx = m.begin_tx();
                    m.train_example_tx(&mut tx, &snippet.content);
                    let h_ux_q = m.cross_entropy_cps(&query_cps);
                    m.rollback_tx(tx);
                    snippet.score = h_u_q - h_ux_q;
                    snippet
                },
            )
            .collect()
    })
}

fn memory_aware_threads(model_bytes: usize) -> usize {
    let hw = num_cpus::get().max(1);
    let avail = linux_mem_available_bytes().unwrap_or(0);
    if avail == 0 {
        return hw;
    }

    // Heuristic: allow up to 25% of available memory for (worker clones + overhead).
    let budget = (avail / 4).max(model_bytes as u64);
    let max_by_mem = (budget / (model_bytes as u64)).max(1) as usize;
    hw.min(max_by_mem).max(1)
}

fn linux_mem_available_bytes() -> Option<u64> {
    // Linux-only best-effort. If parsing fails, fall back to unconstrained.
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.is_empty() {
                return None;
            }
            let kb: u64 = parts[0].parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn stage2_rerank_kmi(query_bytes: &[u8], top_candidates: &mut [Snippet], opts: &SearchOptions) {
    let prior_prefix: Option<Vec<u8>> =
        match (opts.universal_prior.as_deref(), opts.stage2_prior_mode) {
            (None, _) => None,
            (Some(_), Stage2PriorMode::NoPrior) => None,
            (Some(prior_path), Stage2PriorMode::UsePrior) => {
                Some(corpus_bytes(prior_path, SearchGranularity::File))
            }
            (Some(prior_path), Stage2PriorMode::SummarizePrior) => {
                Some(summarize_prior_for_query(query_bytes, prior_path, opts))
            }
        };

    let cq = if let Some(prefix) = prior_prefix.as_deref() {
        opts.ctx.compress_size_chain(&[prefix, query_bytes])
    } else {
        opts.ctx.compress_size_chain(&[query_bytes])
    };

    top_candidates.par_iter_mut().for_each(|snippet| {
        let cx = if let Some(prefix) = prior_prefix.as_deref() {
            opts.ctx
                .compress_size_chain(&[prefix, snippet.content.as_slice()])
        } else {
            opts.ctx.compress_size_chain(&[snippet.content.as_slice()])
        };

        let c1 = if let Some(prefix) = prior_prefix.as_deref() {
            opts.ctx
                .compress_size_chain(&[prefix, snippet.content.as_slice(), query_bytes])
        } else {
            opts.ctx
                .compress_size_chain(&[snippet.content.as_slice(), query_bytes])
        };

        let c2 = if let Some(prefix) = prior_prefix.as_deref() {
            opts.ctx
                .compress_size_chain(&[prefix, query_bytes, snippet.content.as_slice()])
        } else {
            opts.ctx
                .compress_size_chain(&[query_bytes, snippet.content.as_slice()])
        };

        let c_joint = c1.min(c2);
        snippet.score = if c_joint == u64::MAX {
            0.0
        } else {
            (cq as f64 + cx as f64 - c_joint as f64).max(0.0)
        };
    });
}

fn summarize_prior_for_query(
    query_bytes: &[u8],
    prior_path: &str,
    opts: &SearchOptions,
) -> Vec<u8> {
    // Prior-less search inside the prior corpus itself.
    // We approximate K(q|x) via conditional compression: min(C(xq),C(qx)) - C(x), and select the MIN.
    let candidates = collect_candidates(prior_path, opts.granularity);
    if candidates.is_empty() {
        return Vec::new();
    }

    let cq = opts.ctx.compress_size_chain(&[query_bytes]);

    let mut best: Option<(f64, Vec<u8>)> = None;
    for c in candidates {
        let cx = opts.ctx.compress_size_chain(&[c.content.as_slice()]);

        let cxq = opts
            .ctx
            .compress_size_chain(&[c.content.as_slice(), query_bytes]);
        let cqx = opts
            .ctx
            .compress_size_chain(&[query_bytes, c.content.as_slice()]);
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
                let best_cx = opts.ctx.compress_size(best_bytes) as f64;
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
    (4u32).hash(&mut hasher);
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
            if let Ok(mut m) = RosaPlus::load(cache_path.to_string_lossy().as_ref()) {
                // Ensure fixed 256-byte alphabet LM for incremental conditional updates.
                if m.lm_alpha_n() != 256 {
                    m.build_lm_full_bytes_no_finalize_endpos();
                    let _ = m.save(cache_path.to_string_lossy().as_ref());
                }
                return m;
            }
        }

        // Train + save.
        let mut m = RosaPlus::new(opts.max_order, false, 0, 42);
        train_rosa_on_corpus(&mut m, prior_path, SearchGranularity::File);
        // Build a fixed-byte alphabet LM once so the saved model is the full state.
        m.build_lm_full_bytes_no_finalize_endpos();
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
        if matches!(
            ext_str.as_ref(),
            "o" | "a" | "so" | "dll" | "exe" | "bin" | "png" | "jpg" | "zip" | "gz"
        ) {
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
            if let Ok(bytes) = fs::read(path) {
                if bytes.is_empty() {
                    return snippets;
                }

                let window = 50usize;
                let stride = 20usize;

                let mut line_starts: Vec<usize> = Vec::new();
                line_starts.push(0);
                for (i, &b) in bytes.iter().enumerate() {
                    if b == b'\n' {
                        let next = i + 1;
                        if next < bytes.len() {
                            line_starts.push(next);
                        }
                    }
                }

                if line_starts.is_empty() {
                    return snippets;
                }

                let mut i = 0usize;
                while i < line_starts.len() {
                    let end = (i + window).min(line_starts.len());
                    let start_b = line_starts[i];
                    let end_b = if end >= line_starts.len() {
                        bytes.len()
                    } else {
                        line_starts[end]
                    };

                    if end_b > start_b {
                        let content = bytes[start_b..end_b].to_vec();
                        if content.len() > 50 {
                            snippets.push(Snippet {
                                path: path.to_path_buf(),
                                start_line: i + 1,
                                end_line: end,
                                content,
                                score: 0.0,
                            });
                        }
                    }

                    if end == line_starts.len() {
                        break;
                    }
                    i += stride;
                }
            }
        }
    }
    snippets
}
