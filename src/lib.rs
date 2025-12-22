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

/// Compute marginal (Shannon) entropy H(X) = −Σ p(x) log₂ p(x) in bits/symbol.
///
/// This is the simple first-order entropy from the byte histogram,
/// NOT the context-conditional entropy rate from a language model.
#[inline(always)]
pub fn marginal_entropy_bytes(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }

    let n = data.len() as f64;
    let mut h = 0.0f64;
    for i in 0..256 {
        if counts[i] > 0 {
            let p = counts[i] as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

/// Compute entropy rate Ĥ(X) in bits/symbol using ROSA LM.
///
/// This uses ROSA's context-conditional Witten-Bell model to estimate
/// the entropy rate, which accounts for sequential dependencies.
/// For i.i.d. data, this should approximately equal marginal_entropy_bytes.
///
/// `max_order`: Maximum context order for the suffix automaton LM.
/// A value of -1 means unlimited context.
#[inline(always)]
pub fn entropy_rate_bytes(data: &[u8], max_order: i64) -> f64 {
    let mut m = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m.entropy_rate(data)
}

/// Compute joint marginal entropy H(X,Y) = −Σ p(x,y) log₂ p(x,y) in bits/symbol-pair.
///
/// Uses a direct histogram of (x_i, y_i) pairs. This is the exact first-order
/// joint entropy, matching the spec.md definition.
#[inline(always)]
pub fn joint_marginal_entropy_bytes(x: &[u8], y: &[u8]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    // Count pair occurrences using a HashMap for (x, y) pairs
    // There are up to 65536 possible pairs, so we can use a flat array
    let mut counts = vec![0u64; 256 * 256];
    for i in 0..n {
        let pair_idx = (x[i] as usize) * 256 + (y[i] as usize);
        counts[pair_idx] += 1;
    }

    let n_f64 = n as f64;
    let mut h = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n_f64;
            h -= p * p.log2();
        }
    }
    h
}

/// Compute joint entropy rate Ĥ(X,Y) using ROSA's context-conditional model.
///
/// Maps each aligned pair (x_t, y_t) to a unique symbol z_t = x_t * 256 + y_t,
/// then computes the entropy rate Ĥ(Z) of the resulting sequence.
/// This matches the definition for aligned sequences.
#[inline(always)]
pub fn joint_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    // Map pairs (x_i, y_i) to joint symbols z_i in [0, 65535]
    let joint_symbols: Vec<u32> = (0..n)
        .map(|i| (x[i] as u32) * 256 + (y[i] as u32))
        .collect();

    // Compute entropy rate on joint symbol sequence
    let mut m = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m.entropy_rate_cps(&joint_symbols)
}

/// Compute conditional entropy H(X|Y) = H(X,Y) − H(Y)
///
/// Dispatches based on `max_order`.
#[inline(always)]
pub fn conditional_entropy_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if max_order == 0 {
        let h_xy = joint_marginal_entropy_bytes(x, y);
        let h_y = marginal_entropy_bytes(y);
        (h_xy - h_y).max(0.0)
    } else {
        conditional_entropy_rate_bytes(x, y, max_order)
    }
}

/// Compute conditional entropy rate Ĥ(X|Y) = Ĥ(X,Y) − Ĥ(Y)
#[inline(always)]
pub fn conditional_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let h_xy = joint_entropy_rate_bytes(x, y, max_order);
    let h_y = entropy_rate_bytes(y, max_order);
    (h_xy - h_y).max(0.0)
}

/// Compute mutual information I(X;Y) = H(X) + H(Y) − H(X,Y)
///
/// Dispatches based on `max_order`. If 0, uses marginals; else uses rates.
#[inline(always)]
pub fn mutual_information_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if max_order == 0 {
        mutual_information_marg_bytes(x, y)
    } else {
        mutual_information_rate_bytes(x, y, max_order)
    }
}

/// Marginal Mutual Information (exact/histogram)
pub fn mutual_information_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    (h_x + h_y - h_xy).max(0.0)
}

/// Entropy Rate Mutual Information (ROSA predictive)
pub fn mutual_information_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let h_x = entropy_rate_bytes(x, max_order);
    let h_x_given_y = conditional_entropy_rate_bytes(x, y, max_order);
    (h_x - h_x_given_y).max(0.0)
}

// ====== NED: Normalized Entropy Distance ======

/// NED(X,Y) = (H(X,Y) - min(H(X), H(Y))) / max(H(X), H(Y))
///
/// Dispatches based on `max_order`. If 0, uses marginals; else uses rates.
#[inline(always)]
pub fn ned_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if max_order == 0 {
        ned_marg_bytes(x, y)
    } else {
        ned_rate_bytes(x, y, max_order)
    }
}

/// Marginal NED (exact/histogram)
pub fn ned_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    let min_h = h_x.min(h_y);
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / max_h).clamp(0.0, 1.0)
    }
}

/// Entropy Rate NED (ROSA predictive)
pub fn ned_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
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

/// NED_cons(X,Y) = (H(X,Y) - min(H(X), H(Y))) / H(X,Y)
///
/// Conservative variant. Dispatches based on `max_order`.
#[inline(always)]
pub fn ned_cons_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if max_order == 0 {
        ned_cons_marg_bytes(x, y)
    } else {
        ned_cons_rate_bytes(x, y, max_order)
    }
}

pub fn ned_cons_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    let min_h = h_x.min(h_y);
    if h_xy == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / h_xy).clamp(0.0, 1.0)
    }
}

pub fn ned_cons_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
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

/// NTE(X,Y) = VI(X,Y) / max(H(X), H(Y))
/// where VI = H(X|Y) + H(Y|X) = 2·H(X,Y) - H(X) - H(Y)
///
/// Dispatches based on `max_order`.
#[inline(always)]
pub fn nte_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if max_order == 0 {
        nte_marg_bytes(x, y)
    } else {
        nte_rate_bytes(x, y, max_order)
    }
}

pub fn nte_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    let vi = 2.0 * h_xy - h_x - h_y;
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        (vi / max_h).max(0.0)
    }
}

pub fn nte_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
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

/// Compute marginal byte histogram p(i) = count(i) / N for i ∈ [0, 255]
#[inline(always)]
fn byte_histogram(data: &[u8]) -> [f64; 256] {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let n = data.len() as f64;
    let mut probs = [0.0f64; 256];
    if n > 0.0 {
        for i in 0..256 {
            probs[i] = counts[i] as f64 / n;
        }
    }
    probs
}

/// TVD_marg(X,Y) = (1/2) Σᵢ |p_X(i) - p_Y(i)|
///
/// Total Variation Distance over marginal byte distributions.
/// True metric on probability space. Range: [0, 1].
/// 0 = identical distributions, 1 = completely disjoint support.
#[inline(always)]
pub fn tvd_bytes(x: &[u8], y: &[u8], _max_order: i64) -> f64 {
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);

    let mut sum = 0.0f64;
    for i in 0..256 {
        sum += (p_x[i] - p_y[i]).abs();
    }

    (sum / 2.0).clamp(0.0, 1.0)
}

// ====== NHD: Normalized Hellinger Distance ======

/// NHD(X,Y) = sqrt(1 - BC(X,Y)) where BC = Σᵢ sqrt(p_X(i) · p_Y(i))
///
/// Normalized Hellinger Distance over marginal byte distributions.
/// True metric. Range: [0, 1]. 0 = identical, 1 = disjoint support.
#[inline(always)]
pub fn nhd_bytes(x: &[u8], y: &[u8], _max_order: i64) -> f64 {
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);

    // Bhattacharyya coefficient: BC = Σᵢ sqrt(p_X(i) · p_Y(i))
    let mut bc = 0.0f64;
    for i in 0..256 {
        bc += (p_x[i] * p_y[i]).sqrt();
    }

    // NHD = sqrt(1 - BC)
    (1.0 - bc).max(0.0).sqrt()
}

// ====== Other Information-Theoretic Measures ======

/// Compute cross-entropy H(P,Q) = -Σ p(x) log q(x)
///
/// Dispatches based on `max_order`.
pub fn cross_entropy_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if max_order == 0 {
        let p_x = byte_histogram(x);
        let p_y = byte_histogram(y);
        let mut h = 0.0f64;
        for i in 0..256 {
            if p_x[i] > 0.0 {
                // If y has no support where x does, cross-entropy is effectively infinite
                // but we clamp p_y to a small epsilon for stability.
                let q_y = p_y[i].max(1e-12);
                h -= p_x[i] * q_y.log2();
            }
        }
        h
    } else {
        cross_entropy_rate_bytes(x, y, max_order)
    }
}

/// Compute cross-entropy rate using ROSA.
/// Training model on Y and evaluating probability of X.
pub fn cross_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    let mut m = rosaplus::RosaPlus::new(max_order, false, 0, 42);
    m.train_example(y);
    m.build_lm();
    m.cross_entropy(x)
}

/// Kullback-Leibler Divergence D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
///
/// Marginal only. Measure of how one probability distribution is different from a second.
pub fn d_kl_bytes(x: &[u8], y: &[u8]) -> f64 {
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);
    let mut d_kl = 0.0f64;
    for i in 0..256 {
        if p_x[i] > 0.0 {
            let q_y = p_y[i].max(1e-12);
            d_kl += p_x[i] * (p_x[i] / q_y).log2();
        }
    }
    d_kl.max(0.0)
}

/// Jensen-Shannon Divergence JSD(P || Q) = 1/2 D_KL(P || M) + 1/2 D_KL(Q || M)
/// where M = 1/2 (P + Q)
///
/// Marginal only. Symmetrized and smoothed version of KL divergence. Range [0,1].
pub fn js_div_bytes(x: &[u8], y: &[u8]) -> f64 {
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);
    let mut m = [0.0f64; 256];
    for i in 0..256 {
        m[i] = 0.5 * (p_x[i] + p_y[i]);
    }

    let mut kl_pm = 0.0f64;
    let mut kl_qm = 0.0f64;
    for i in 0..256 {
        if p_x[i] > 0.0 {
            kl_pm += p_x[i] * (p_x[i] / m[i]).log2();
        }
        if p_y[i] > 0.0 {
            kl_qm += p_y[i] * (p_y[i] / m[i]).log2();
        }
    }
    (0.5 * kl_pm + 0.5 * kl_qm).max(0.0)
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

/// Mutual Information for files.
pub fn mutual_information_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    mutual_information_bytes(&bx, &by, max_order)
}

/// Conditional Entropy for files.
pub fn conditional_entropy_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    conditional_entropy_bytes(&bx, &by, max_order)
}

/// Cross-Entropy for files.
pub fn cross_entropy_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    cross_entropy_bytes(&bx, &by, max_order)
}

/// KL Divergence for files.
pub fn kl_divergence_paths(x: &str, y: &str) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    d_kl_bytes(&bx, &by)
}

/// Jensen-Shannon Divergence for files.
pub fn js_divergence_paths(x: &str, y: &str) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    js_div_bytes(&bx, &by)
}

// ====== Primitives 6 & 7 ======

/// Primitive 6: Intrinsic vs Extrinsic Dependence.
///
/// Returns a ratio representing how much of the data's structure is internal (periodicity/symmetry)
/// vs external (Shannon entropy).
/// Ratio closer to 0 means high intrinsic dependence (very predictable).
/// Ratio closer to 1 means low intrinsic dependence (looks random or depends on external priors).
pub fn intrinsic_dependence_bytes(data: &[u8], max_order: i64) -> f64 {
    let h_marginal = marginal_entropy_bytes(data);
    if h_marginal == 0.0 {
        return 0.0;
    }
    let h_rate = entropy_rate_bytes(data, max_order);
    (h_rate / h_marginal).clamp(0.0, 1.0)
}

/// Primitive 7: Resistance under Allowed Transformations.
///
/// Measures how much information is preserved after a transformation T is applied to X.
/// Resistance(X, T) = I(X; T(X)) / H(X).
/// Range [0,1]. 1 means perfectly resistant, 0 means the transformation destroyed all information.
pub fn resistance_to_transformation_bytes(x: &[u8], tx: &[u8], max_order: i64) -> f64 {
    let h_x = entropy_rate_bytes(x, max_order);
    if h_x == 0.0 {
        return 1.0;
    }
    let mi = mutual_information_bytes(x, tx, max_order);
    (mi / h_x).clamp(0.0, 1.0)
}
