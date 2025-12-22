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
