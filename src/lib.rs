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
