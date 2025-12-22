use rayon::prelude::*;

use std::sync::OnceLock;

static NUM_THREADS: OnceLock<usize> = OnceLock::new();

/// ------- Base Compression Functions -------
#[inline(always)]
pub fn get_compressed_size(path: &str, method: &str) -> u64 {
    // Convert Input file to Vec<u8>, and reference that (compress_size only takes &[u8] input), and pass method.
    // Will panic if file does not exist, so it must be prevalidated.
    zpaq_rs::compress_size(&std::fs::read(path).unwrap(), method).unwrap()
}
#[inline(always)]
pub fn get_compressed_size_parallel(path: &str, method: &str, threads: usize) -> u64 {
    // Convert Input file to Vec<u8>, and reference that (compress_size only takes &[u8] input), and pass method.
    // Will panic if file does not exist, so it must be prevalidated.
    zpaq_rs::compress_size_parallel(&std::fs::read(path).unwrap(), method, threads).unwrap()
}

#[inline(always)]
pub fn get_bytes_from_paths(paths: &[&str]) -> Vec<Vec<u8>> {
    paths
        .par_iter()
        .map(|path| std::fs::read(*path).expect("failed to read file"))
        .collect()
}

/// ------- Bulk File Compression Functions -------
#[inline(always)]
pub fn get_sequential_compressed_sizes_from_sequential_paths(
    paths: &[&str],
    method: &str,
) -> Vec<u64> {
    // This will, in parallel load all files into memory, THEN in parallel compress each one, each with one thread.
    // Use when File IO is the bottleneck
    // Only uses ONE ZPAQ THREAD.
    // For VERY large n (relative to threads) with small files (relative to memory) this may be useful.
    get_bytes_from_paths(paths)
        .par_iter()
        .map(|data| zpaq_rs::compress_size(data, method).unwrap())
        .collect()
}

#[inline(always)]
pub fn get_parallel_compressed_sizes_from_sequential_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> Vec<u64> {
    // This will, in parallel load all files into memory, THEN in parallel compress each one, with THREADS. (for each file, the thread count is THREADS)
    // Use when File IO is the bottleneck.
    // Balanced parallelization between RAYON_NUM_THREADS and ZPAQ `THREADS` const. For when total dataset will fit in memory.
    get_bytes_from_paths(paths)
        .par_iter()
        .map(|data| zpaq_rs::compress_size_parallel(data, method, threads).unwrap())
        .collect()
}

#[inline(always)]
pub fn get_sequential_compressed_sizes_from_parallel_paths(
    paths: &[&str],
    method: &str,
) -> Vec<u64> {
    // This will, in parallel, for each file, read it from disk and compress it with one thread. (one file, one thread)
    // Use when File IO is not the bottleneck. Lower memory usage. (does not preload dataset)
    // Only uses ONE ZPAQ THREAD. For VERY large n(relative to threads) with large files(relative to memory) this may be useful.
    paths
        .par_iter()
        .map(|path| get_compressed_size(path, method))
        .collect()
}

#[inline(always)]
pub fn get_parallel_compressed_sizes_from_parallel_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> Vec<u64> {
    // This will, in parallel, for each file, read it from disk and compress it with THREADS. (for each file, the thread count is THREADS)
    // Use when File IO is not the bottleneck. Lower memory usage. (does not preload dataset)
    // For large n(relative to threads) with VERY large files(relative to memory) this may be useful.
    // This will reflect RAYON_NUM_THREADS and THREAD const values.
    paths
        .par_iter()
        .map(|path| get_compressed_size_parallel(path, method, threads))
        .collect()
}

/// Optimizes parallelization
#[inline(always)]
pub fn get_compressed_sizes_from_paths(paths: &[&str], method: &str) -> Vec<u64> {
    let n: usize = paths.len();
    let num_threads: usize = *NUM_THREADS.get_or_init(|| num_cpus::get());
    if n < num_threads {
        get_parallel_compressed_sizes_from_parallel_paths(paths, method, (num_threads + n - 1) / n)
    } else {
        get_sequential_compressed_sizes_from_parallel_paths(paths, method)
    }
}

/// ----- NCD ------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcdVariant {
    /// NCD_vitanyi(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    Vitanyi,
    /// NCD_sym_vitanyi(x,y) = (min(C(xy), C(yx)) - min(C(x), C(y))) / max(C(x), C(y))
    SymVitanyi,
    /// NCD_cons(x,y) = (C(xy) - min(C(x), C(y))) / C(xy)
    Cons,
    /// NCD_sym_cons(x,y) = (min(C(xy), C(yx)) - min(C(x), C(y))) / min(C(xy), C(yx))
    SymCons,
}

#[inline(always)]
fn compress_size_bytes(data: &[u8], method: &str) -> u64 {
    zpaq_rs::compress_size(data, method).unwrap()
}

#[inline(always)]
fn ncd_from_sizes(cx: u64, cy: u64, cxy: u64, cyx: Option<u64>, variant: NcdVariant) -> f64 {
    let min_c = cx.min(cy) as f64;
    let max_c = cx.max(cy) as f64;

    match variant {
        NcdVariant::Vitanyi => {
            if max_c == 0.0 {
                0.0
            } else {
                (cxy as f64 - min_c) / max_c
            }
        }
        NcdVariant::SymVitanyi => {
            let m = cxy.min(cyx.expect("cyx required for SymVitanyi")) as f64;
            if max_c == 0.0 {
                0.0
            } else {
                (m - min_c) / max_c
            }
        }
        NcdVariant::Cons => {
            let denom = cxy as f64;
            if denom == 0.0 {
                0.0
            } else {
                (cxy as f64 - min_c) / denom
            }
        }
        NcdVariant::SymCons => {
            let m = cxy.min(cyx.expect("cyx required for SymCons")) as f64;
            if m == 0.0 { 0.0 } else { (m - min_c) / m }
        }
    }
}

#[inline(always)]
pub fn ncd_bytes(x: &[u8], y: &[u8], method: &str, variant: NcdVariant) -> f64 {
    let (cx, cy) = rayon::join(
        || compress_size_bytes(x, method),
        || compress_size_bytes(y, method),
    );

    let mut buf = Vec::with_capacity(x.len() + y.len());
    buf.extend_from_slice(x);
    buf.extend_from_slice(y);
    let cxy = compress_size_bytes(&buf, method);

    let cyx = match variant {
        NcdVariant::SymVitanyi | NcdVariant::SymCons => {
            buf.clear();
            buf.extend_from_slice(y);
            buf.extend_from_slice(x);
            Some(compress_size_bytes(&buf, method))
        }
        _ => None,
    };

    ncd_from_sizes(cx, cy, cxy, cyx, variant)
}

#[inline(always)]
pub fn ncd_paths(x: &str, y: &str, method: &str, variant: NcdVariant) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    ncd_bytes(&bx, &by, method, variant)
}

/// Back-compat convenience wrappers (operate on file paths).
#[inline(always)]
pub fn ncd_vitanyi(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::Vitanyi)
}
#[inline(always)]
pub fn ncd_sym_vitanyi(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::SymVitanyi)
}
#[inline(always)]
pub fn ncd_cons(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::Cons)
}
#[inline(always)]
pub fn ncd_sym_cons(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::SymCons)
}

/// Computes an NCD matrix (row-major, len = n*n) for in-memory byte blobs.
///
/// Note: For symmetric variants, this computes each unordered pair once and writes both (i,j) and (j,i).
pub fn ncd_matrix_bytes(datas: &[Vec<u8>], method: &str, variant: NcdVariant) -> Vec<f64> {
    let n = datas.len();
    let cx: Vec<u64> = datas
        .par_iter()
        .map(|d| compress_size_bytes(d, method))
        .collect();

    let mut out = vec![0.0f64; n * n];
    let out_ptr = std::sync::atomic::AtomicPtr::new(out.as_mut_ptr());

    match variant {
        NcdVariant::SymVitanyi | NcdVariant::SymCons => {
            (0..n)
                .into_par_iter()
                .flat_map_iter(|i| (i + 1..n).map(move |j| (i, j)))
                .for_each_init(Vec::<u8>::new, |buf, (i, j)| {
                    let x = &datas[i];
                    let y = &datas[j];

                    buf.clear();
                    buf.reserve(x.len() + y.len());
                    buf.extend_from_slice(x);
                    buf.extend_from_slice(y);
                    let cxy = compress_size_bytes(buf, method);

                    buf.clear();
                    buf.reserve(x.len() + y.len());
                    buf.extend_from_slice(y);
                    buf.extend_from_slice(x);
                    let cyx = compress_size_bytes(buf, method);

                    let d = ncd_from_sizes(cx[i], cx[j], cxy, Some(cyx), variant);

                    // Safety: each (i,j) cell is written exactly once across all iterations.
                    let p = out_ptr.load(std::sync::atomic::Ordering::Relaxed);
                    unsafe {
                        *p.add(i * n + j) = d;
                        *p.add(j * n + i) = d;
                    }
                });
        }
        NcdVariant::Vitanyi | NcdVariant::Cons => {
            (0..n)
                .into_par_iter()
                .for_each_init(Vec::<u8>::new, |buf, i| {
                    let x = &datas[i];
                    for j in 0..n {
                        let d = if i == j {
                            0.0
                        } else {
                            let y = &datas[j];
                            buf.clear();
                            buf.reserve(x.len() + y.len());
                            buf.extend_from_slice(x);
                            buf.extend_from_slice(y);
                            let cxy = compress_size_bytes(buf, method);
                            ncd_from_sizes(cx[i], cx[j], cxy, None, variant)
                        };

                        let p = out_ptr.load(std::sync::atomic::Ordering::Relaxed);
                        unsafe {
                            *p.add(i * n + j) = d;
                        }
                    }
                });
        }
    }

    out
}

/// Computes an NCD matrix (row-major, len = n*n) for files (preloads all files into memory once).
pub fn ncd_matrix_paths(paths: &[&str], method: &str, variant: NcdVariant) -> Vec<f64> {
    let datas = get_bytes_from_paths(paths);
    ncd_matrix_bytes(&datas, method, variant)
}

// ============================================================
// Entropy-Based Distance Primitives (via ROSA)
// ============================================================
//
// These use ROSA's Witten-Bell language model to estimate entropy
// and compute information-theoretic distances.

/// Compute entropy rate Ĥ(X) in bits/symbol using ROSA LM.
///
/// `max_order`: Maximum context order for the suffix automaton LM.
/// A value of -1 means unlimited context.
#[inline(always)]
pub fn entropy_rate_bytes(data: &[u8], max_order: i64) -> f64 {
    let mut m = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m.entropy_rate(data)
}

/// Compute joint entropy rate Ĥ(X,Y) using pair-as-symbol encoding.
///
/// Maps each aligned pair (x_t, y_t) to a unique symbol z_t, then computes
/// Ĥ(X,Y) = −(1/N) Σ log₂ p̂(z_t | context)
///
/// Requires `x` and `y` to have the same length. If lengths differ,
/// truncates to the shorter length.
///
/// This is the information-theoretically correct approach:
/// H(X,Y) = entropy rate of the joint process {(X_t, Y_t)}.
#[inline(always)]
pub fn joint_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        // For single symbols, joint entropy = sum of marginals (assuming independence)
        // This is an upper bound; for correlated data it's an approximation
        return entropy_rate_bytes(x, max_order) + entropy_rate_bytes(y, max_order);
    }

    // Map each byte pair (x, y) to a unique Unicode codepoint in the supplementary plane.
    // Codepoint = 0x10000 + x * 256 + y, giving range [0x10000, 0x1FFFF].
    // This is bijective and survives utf8_decode_lossy without distortion.
    // Each codepoint encodes to exactly 4 UTF-8 bytes.
    let mut joint_seq = Vec::with_capacity(n * 4);
    for i in 0..n {
        let pair_code = 0x10000u32 + (x[i] as u32) * 256 + (y[i] as u32);
        // Encode as 4-byte UTF-8: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        joint_seq.push((0xF0 | (pair_code >> 18)) as u8);
        joint_seq.push((0x80 | ((pair_code >> 12) & 0x3F)) as u8);
        joint_seq.push((0x80 | ((pair_code >> 6) & 0x3F)) as u8);
        joint_seq.push((0x80 | (pair_code & 0x3F)) as u8);
    }

    // Train ROSA on the joint sequence and compute entropy rate
    let mut m = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    let h_joint_bits_per_codepoint = m.entropy_rate(&joint_seq);

    // entropy_rate returns bits per codepoint (each pair = one codepoint)
    // so this directly gives H(X,Y) in bits per symbol-pair
    h_joint_bits_per_codepoint
}

/// Compute conditional entropy rate Ĥ(X|Y) = Ĥ(X,Y) − Ĥ(Y)
///
/// Uses the chain rule of entropy. Clamps result to non-negative.
#[inline(always)]
pub fn conditional_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let h_xy = joint_entropy_rate_bytes(x, y, max_order);
    let h_y = entropy_rate_bytes(y, max_order);
    (h_xy - h_y).max(0.0)
}

/// Compute mutual information Î(X;Y) = Ĥ(X) + Ĥ(Y) − Ĥ(X,Y)
#[inline(always)]
pub fn mutual_information_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let h_x = entropy_rate_bytes(x, max_order);
    let h_y = entropy_rate_bytes(y, max_order);
    let h_xy = joint_entropy_rate_bytes(x, y, max_order);
    (h_x + h_y - h_xy).max(0.0) // Clamp to non-negative
}

// ====== NED: Normalized Entropy Distance ======

/// NED(X,Y) = (Ĥ(X,Y) - min(Ĥ(X), Ĥ(Y))) / max(Ĥ(X), Ĥ(Y))
///
/// Measures fraction of larger variable's uncertainty remaining after observing the other.
/// Range: [0, 1]. 0 = perfectly redundant, 1 = independent.
#[inline(always)]
pub fn ned_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let h_x = entropy_rate_bytes(x, max_order);
    let h_y = entropy_rate_bytes(y, max_order);
    let h_xy = joint_entropy_rate_bytes(x, y, max_order);
    
    let min_h = h_x.min(h_y);
    let max_h = h_x.max(h_y);
    
    if max_h == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / max_h).clamp(0.0, 1.0)
    }
}

/// NED_cons(X,Y) = (Ĥ(X,Y) - min(Ĥ(X), Ĥ(Y))) / Ĥ(X,Y)
///
/// Conservative variant using joint entropy as denominator.
#[inline(always)]
pub fn ned_cons_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let h_x = entropy_rate_bytes(x, max_order);
    let h_y = entropy_rate_bytes(y, max_order);
    let h_xy = joint_entropy_rate_bytes(x, y, max_order);
    
    let min_h = h_x.min(h_y);
    
    if h_xy == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / h_xy).clamp(0.0, 1.0)
    }
}

// ====== NTE: Normalized Transform Effort (Variation of Information) ======

/// NTE(X,Y) = VI(X,Y) / max(Ĥ(X), Ĥ(Y))
/// where VI = H(X|Y) + H(Y|X) = 2·H(X,Y) - H(X) - H(Y)
///
/// Measures total effort to transform X into Y and vice versa.
/// Range: [0, 2]. 0 = identical, 2 = completely different.
#[inline(always)]
pub fn nte_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let h_x = entropy_rate_bytes(x, max_order);
    let h_y = entropy_rate_bytes(y, max_order);
    let h_xy = joint_entropy_rate_bytes(x, y, max_order);
    
    let vi = 2.0 * h_xy - h_x - h_y;
    let max_h = h_x.max(h_y);
    
    if max_h == 0.0 {
        0.0
    } else {
        (vi / max_h).max(0.0)
    }
}

// ====== TVD: Total Variation Distance ======

/// TVD_marg(X,Y) = (1/2) Σᵢ |p_X(i) - p_Y(i)|
///
/// Total Variation Distance over marginal distributions.
/// True metric on probability space. Range: [0, 1].
#[inline(always)]
pub fn tvd_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    // Build marginal distributions
    let mut m_x = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m_x.train_example(x);
    m_x.build_lm();
    let dist_x: std::collections::HashMap<u32, f64> = m_x.marginal_distribution().into_iter().collect();
    
    let mut m_y = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m_y.train_example(y);
    m_y.build_lm();
    let dist_y: std::collections::HashMap<u32, f64> = m_y.marginal_distribution().into_iter().collect();
    
    // Collect all symbols
    let mut all_symbols: std::collections::HashSet<u32> = dist_x.keys().copied().collect();
    all_symbols.extend(dist_y.keys().copied());
    
    // Compute TVD
    let mut tvd = 0.0f64;
    for sym in all_symbols {
        let p_x = dist_x.get(&sym).copied().unwrap_or(0.0);
        let p_y = dist_y.get(&sym).copied().unwrap_or(0.0);
        tvd += (p_x - p_y).abs();
    }
    
    (tvd / 2.0).clamp(0.0, 1.0)
}

// ====== NHD: Normalized Hellinger Distance ======

/// NHD(X,Y) = sqrt(1 - BC(X,Y)) where BC = Σᵢ sqrt(p_X(i) · p_Y(i))
///
/// Normalized Hellinger Distance over marginal distributions.
/// True metric. Range: [0, 1]. 0 = identical, 1 = disjoint support.
#[inline(always)]
pub fn nhd_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    // Build marginal distributions
    let mut m_x = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m_x.train_example(x);
    m_x.build_lm();
    let dist_x: std::collections::HashMap<u32, f64> = m_x.marginal_distribution().into_iter().collect();
    
    let mut m_y = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m_y.train_example(y);
    m_y.build_lm();
    let dist_y: std::collections::HashMap<u32, f64> = m_y.marginal_distribution().into_iter().collect();
    
    // Collect all symbols present in both
    let mut bc = 0.0f64;
    for (sym, p_x) in &dist_x {
        if let Some(&p_y) = dist_y.get(sym) {
            bc += (p_x * p_y).sqrt();
        }
    }
    
    // NHD = sqrt(1 - BC)
    (1.0 - bc).max(0.0).sqrt()
}

// ====== Path-based convenience wrappers ======

/// NED for files.
pub fn ned_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    ned_bytes(&bx, &by, max_order)
}

/// NTE for files.
pub fn nte_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    nte_bytes(&bx, &by, max_order)
}

/// TVD for files.
pub fn tvd_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    tvd_bytes(&bx, &by, max_order)
}

/// NHD for files.
pub fn nhd_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    nhd_bytes(&bx, &by, max_order)
}
