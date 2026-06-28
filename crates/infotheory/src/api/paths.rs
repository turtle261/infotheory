//! Path-oriented convenience API surface.

use rayon::prelude::*;

use super::compression::{
    NcdComputeOptions, NcdVariant, OperationParallelism, try_compress_size_backend,
    try_ncd_bytes_backend_with_options, try_ncd_matrix_bytes_backend_with_options,
};
use super::metrics::{
    d_kl_bytes, js_div_bytes, nhd_bytes, try_conditional_entropy_bytes, try_cross_entropy_bytes,
    try_mutual_information_bytes, try_ned_bytes, try_nte_bytes, tvd_bytes,
};
use super::types::CompressionBackend;
use crate::error::{InfotheoryError, InfotheoryResult};
use crate::spec::CompiledCompressionBackend;
use rayon::ThreadPoolBuilder;

#[inline(always)]
fn try_read_path_pair(x: &str, y: &str) -> InfotheoryResult<(Vec<u8>, Vec<u8>)> {
    let (bx, by) = rayon::join(
        || std::fs::read(x).map_err(InfotheoryError::from),
        || std::fs::read(y).map_err(InfotheoryError::from),
    );
    Ok((bx?, by?))
}

/// Options for backend-first path compression-size operations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompressionPathBatchOptions {
    /// Operation-level parallelism policy used while processing the path batch. (external parellization, doesn't affect compression algorithm itself)
    pub parallelism: OperationParallelism,
}

/// Read `path` and return its compressed size (bytes) using `backend`.
pub fn try_get_compressed_size_path_backend(
    path: &str,
    backend: &CompiledCompressionBackend,
) -> InfotheoryResult<u64> {
    let data = std::fs::read(path)?;
    try_compress_size_backend(&data, backend)
}

#[inline(always)]
/// Read all files in `paths` in parallel and return their contents.
pub fn try_get_bytes_from_paths(paths: &[&str]) -> InfotheoryResult<Vec<Vec<u8>>> {
    paths
        .par_iter()
        .map(|path| std::fs::read(*path).map_err(InfotheoryError::from))
        .collect()
}

/// Compute compressed sizes for `paths` using `backend`.
pub fn try_get_compressed_sizes_from_paths_backend(
    paths: &[&str],
    backend: &CompiledCompressionBackend,
) -> InfotheoryResult<Vec<u64>> {
    try_get_compressed_sizes_from_paths_backend_with_options(
        paths,
        backend,
        CompressionPathBatchOptions::default(),
    )
}

/// Compute compressed sizes for `paths` using `backend` with explicit
/// operation-level parallelism controls.
pub fn try_get_compressed_sizes_from_paths_backend_with_options(
    paths: &[&str],
    backend: &CompiledCompressionBackend,
    options: CompressionPathBatchOptions,
) -> InfotheoryResult<Vec<u64>> {
    match options.parallelism {
        OperationParallelism::Serial => paths
            .iter()
            .map(|path| try_get_compressed_size_path_backend(path, backend))
            .collect(),
        OperationParallelism::Auto => paths
            .par_iter()
            .map(|path| try_get_compressed_size_path_backend(path, backend))
            .collect(),
        OperationParallelism::Threads(threads) => {
            if threads <= 1 {
                return paths
                    .iter()
                    .map(|path| try_get_compressed_size_path_backend(path, backend))
                    .collect();
            }
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|err| {
                    InfotheoryError::runtime(format!("failed to build rayon pool: {err}"))
                })?;
            pool.install(|| {
                paths
                    .par_iter()
                    .map(|path| try_get_compressed_size_path_backend(path, backend))
                    .collect()
            })
        }
    }
}

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
    let compiled = backend
        .compile()
        .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?;
    try_ncd_bytes_backend_with_options(&bx?, &by?, &compiled, variant, NcdComputeOptions::default())
}

#[inline(always)]
/// Compute NCD for two files with an explicit compiled compression backend.
pub fn try_ncd_paths_compiled_backend(
    x: &str,
    y: &str,
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
) -> InfotheoryResult<f64> {
    try_ncd_paths_compiled_backend_with_options(
        x,
        y,
        backend,
        variant,
        NcdComputeOptions::default(),
    )
}

/// Compute NCD for two files with an explicit compiled compression backend and
/// explicit operation-level parallelism controls.
pub fn try_ncd_paths_compiled_backend_with_options(
    x: &str,
    y: &str,
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
    options: NcdComputeOptions,
) -> InfotheoryResult<f64> {
    let (bx, by) = rayon::join(
        || std::fs::read(x).map_err(InfotheoryError::from),
        || std::fs::read(y).map_err(InfotheoryError::from),
    );
    try_ncd_bytes_backend_with_options(&bx?, &by?, backend, variant, options)
}

/// Compute an `n x n` pairwise NCD matrix (row-major) for file paths with an explicit compiled compression backend.
pub fn try_ncd_matrix_paths_backend(
    paths: &[&str],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
) -> InfotheoryResult<Vec<f64>> {
    try_ncd_matrix_paths_backend_with_options(paths, backend, variant, NcdComputeOptions::default())
}

/// Compute an `n x n` pairwise NCD matrix (row-major) for file paths with an
/// explicit compiled compression backend and operation-level parallelism
/// controls.
pub fn try_ncd_matrix_paths_backend_with_options(
    paths: &[&str],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
    options: NcdComputeOptions,
) -> InfotheoryResult<Vec<f64>> {
    let datas = try_get_bytes_from_paths(paths)?;
    try_ncd_matrix_bytes_backend_with_options(&datas, backend, variant, options)
}

/// Compute normalized entropy distance (NED) for two files using the default rate backend.
pub fn try_ned_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_ned_bytes(&bx, &by)
}

/// Compute normalized transform effort (NTE) for two files using the default rate backend.
pub fn try_nte_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_nte_bytes(&bx, &by)
}

/// Compute total variation distance (TVD) between the byte distributions of two files.
pub fn try_tvd_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(tvd_bytes(&bx, &by))
}

/// Compute normalized Hellinger distance (NHD) between the byte distributions of two files.
pub fn try_nhd_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(nhd_bytes(&bx, &by))
}

/// Compute mutual information estimate for two files using the default rate backend.
pub fn try_mutual_information_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_mutual_information_bytes(&bx, &by)
}

/// Compute conditional entropy estimate `H(X|Y)` for two files using the default rate backend.
pub fn try_conditional_entropy_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_conditional_entropy_bytes(&bx, &by)
}

/// Compute cross-entropy estimate `H_train(test)` for two files using the default rate backend.
pub fn try_cross_entropy_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    try_cross_entropy_bytes(&bx, &by)
}

/// Compute KL divergence between the byte histograms of two files.
pub fn try_kl_divergence_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(d_kl_bytes(&bx, &by))
}

/// Compute Jensen-Shannon divergence between the byte histograms of two files.
pub fn try_js_divergence_paths(x: &str, y: &str) -> InfotheoryResult<f64> {
    let (bx, by) = try_read_path_pair(x, y)?;
    Ok(js_div_bytes(&bx, &by))
}
