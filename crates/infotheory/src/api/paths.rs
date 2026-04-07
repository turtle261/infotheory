//! Path-oriented convenience API surface.

use rayon::prelude::*;

use super::compression::{NcdVariant, try_ncd_bytes, try_ncd_bytes_backend, try_ncd_matrix_bytes};
use super::metrics::{
    d_kl_bytes, js_div_bytes, nhd_bytes, try_conditional_entropy_bytes, try_cross_entropy_bytes,
    try_mutual_information_bytes, try_ned_bytes, try_nte_bytes, tvd_bytes,
};
use super::types::CompressionBackend;
use crate::error::{InfotheoryError, InfotheoryResult};
use crate::{NUM_THREADS, try_zpaq_compress_size_bytes, try_zpaq_compress_size_parallel_bytes};

#[inline(always)]
fn try_read_path_pair(x: &str, y: &str) -> InfotheoryResult<(Vec<u8>, Vec<u8>)> {
    let (bx, by) = rayon::join(
        || std::fs::read(x).map_err(InfotheoryError::from),
        || std::fs::read(y).map_err(InfotheoryError::from),
    );
    Ok((bx?, by?))
}

#[inline(always)]
/// Read `path` and return its compressed size (bytes) using ZPAQ `method`.
pub fn try_get_compressed_size(path: &str, method: &str) -> InfotheoryResult<u64> {
    let data = std::fs::read(path)?;
    try_zpaq_compress_size_bytes(&data, method)
}

#[inline(always)]
/// Read `path` and return its compressed size (bytes) using parallel ZPAQ compression.
pub fn try_get_compressed_size_parallel(
    path: &str,
    method: &str,
    threads: usize,
) -> InfotheoryResult<u64> {
    let data = std::fs::read(path)?;
    try_zpaq_compress_size_parallel_bytes(&data, method, threads)
}

#[inline(always)]
/// Read all files in `paths` in parallel and return their contents.
pub fn try_get_bytes_from_paths(paths: &[&str]) -> InfotheoryResult<Vec<Vec<u8>>> {
    paths
        .par_iter()
        .map(|path| std::fs::read(*path).map_err(InfotheoryError::from))
        .collect()
}

#[inline(always)]
/// Read all files once, then compress each buffer with single-stream ZPAQ.
pub fn try_get_sequential_compressed_sizes_from_sequential_paths(
    paths: &[&str],
    method: &str,
) -> InfotheoryResult<Vec<u64>> {
    let datas = try_get_bytes_from_paths(paths)?;
    datas
        .par_iter()
        .map(|data| try_zpaq_compress_size_bytes(data, method))
        .collect()
}

#[inline(always)]
/// Read all files once, then compress each buffer with parallel ZPAQ.
pub fn try_get_parallel_compressed_sizes_from_sequential_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> InfotheoryResult<Vec<u64>> {
    let datas = try_get_bytes_from_paths(paths)?;
    datas
        .par_iter()
        .map(|data| try_zpaq_compress_size_parallel_bytes(data, method, threads))
        .collect()
}

#[inline(always)]
/// Compress each file path independently with single-stream ZPAQ.
pub fn try_get_sequential_compressed_sizes_from_parallel_paths(
    paths: &[&str],
    method: &str,
) -> InfotheoryResult<Vec<u64>> {
    Ok(paths
        .par_iter()
        .map(|path| try_get_compressed_size(path, method))
        .collect::<InfotheoryResult<Vec<_>>>()?)
}

#[inline(always)]
/// Compress each file path independently with parallel ZPAQ.
pub fn try_get_parallel_compressed_sizes_from_parallel_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> InfotheoryResult<Vec<u64>> {
    Ok(paths
        .par_iter()
        .map(|path| try_get_compressed_size_parallel(path, method, threads))
        .collect::<InfotheoryResult<Vec<_>>>()?)
}

#[inline(always)]
/// Compute compressed sizes for `paths` with an adaptive thread strategy.
///
/// For small batches (`len(paths) < NUM_THREADS`), this uses stronger per-file
/// parallelism. Otherwise, it parallelizes across paths.
pub fn try_get_compressed_sizes_from_paths(
    paths: &[&str],
    method: &str,
) -> InfotheoryResult<Vec<u64>> {
    let n = paths.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let num_threads = *NUM_THREADS.get_or_init(num_cpus::get);
    if n < num_threads {
        try_get_parallel_compressed_sizes_from_parallel_paths(
            paths,
            method,
            num_threads.div_ceil(n),
        )
    } else {
        try_get_sequential_compressed_sizes_from_parallel_paths(paths, method)
    }
}

#[inline(always)]
/// Compute NCD for two files using ZPAQ `method`.
pub fn try_ncd_paths(x: &str, y: &str, method: &str, variant: NcdVariant) -> InfotheoryResult<f64> {
    let (bx, by) = rayon::join(
        || std::fs::read(x).map_err(InfotheoryError::from),
        || std::fs::read(y).map_err(InfotheoryError::from),
    );
    try_ncd_bytes(&bx?, &by?, method, variant)
}

#[inline(always)]
/// Compute NCD for two files with an explicit compression backend.
pub fn try_ncd_paths_backend(
    x: &str,
    y: &str,
    backend: &CompressionBackend,
    variant: NcdVariant,
) -> InfotheoryResult<f64> {
    let (bx, by) = rayon::join(
        || std::fs::read(x).map_err(InfotheoryError::from),
        || std::fs::read(y).map_err(InfotheoryError::from),
    );
    try_ncd_bytes_backend(&bx?, &by?, backend, variant)
}

/// Compute an `n x n` pairwise NCD matrix (row-major) for file paths.
pub fn try_ncd_matrix_paths(
    paths: &[&str],
    method: &str,
    variant: NcdVariant,
) -> InfotheoryResult<Vec<f64>> {
    let datas = try_get_bytes_from_paths(paths)?;
    try_ncd_matrix_bytes(&datas, method, variant)
}

/// Compute normalized entropy distance (NED) for two files.
pub fn try_ned_paths(x: &str, y: &str, max_order: i64) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_ned_bytes(&bx, &by, max_order)
}

/// Compute normalized transform effort (NTE) for two files.
pub fn try_nte_paths(x: &str, y: &str, max_order: i64) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_nte_bytes(&bx, &by, max_order)
}

/// Compute total variation distance (TVD) between marginal byte distributions of two files.
pub fn try_tvd_paths(x: &str, y: &str, max_order: i64) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(tvd_bytes(&bx, &by, max_order))
}

/// Compute normalized Hellinger distance (NHD) between marginal byte distributions of two files.
pub fn try_nhd_paths(x: &str, y: &str, max_order: i64) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(nhd_bytes(&bx, &by, max_order))
}

/// Compute mutual information estimate for two files.
pub fn try_mutual_information_paths(x: &str, y: &str, max_order: i64) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_mutual_information_bytes(&bx, &by, max_order)
}

/// Compute conditional entropy estimate `H(X|Y)` for two files.
pub fn try_conditional_entropy_paths(x: &str, y: &str, max_order: i64) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_conditional_entropy_bytes(&bx, &by, max_order)
}

/// Compute cross-entropy estimate `H_train(test)` for two files.
pub fn try_cross_entropy_paths(x: &str, y: &str, max_order: i64) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_cross_entropy_bytes(&bx, &by, max_order)
}

/// Compute KL divergence between marginal byte histograms of two files.
pub fn try_kl_divergence_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(d_kl_bytes(&bx, &by))
}

/// Compute Jensen-Shannon divergence between marginal byte histograms of two files.
pub fn try_js_divergence_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(js_div_bytes(&bx, &by))
}
