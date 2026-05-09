//! Compression-focused public API surface.

use rayon::prelude::*;

use crate::error::{InfotheoryError, InfotheoryResult};
use crate::spec::CompiledCompressionBackend;

use crate::runtime::CompressionRuntime;
use crate::with_default_ctx;

/// Per-call control over operation-level parallelism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationParallelism {
    /// Execute operation-level work serially.
    Serial,
    /// Use adaptive/default parallel operation behavior.
    Auto,
    /// Execute operation-level work on a bounded Rayon pool with `threads`.
    Threads(usize),
}

impl Default for OperationParallelism {
    fn default() -> Self {
        Self::Auto
    }
}

/// NCD compute options (operation-level parallelism only).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NcdComputeOptions {
    pub parallelism: OperationParallelism,
}

/// Compute compressed size (bytes) for a logical concatenation of `parts` using `backend`.
pub fn try_compress_size_chain_backend(
    parts: &[&[u8]],
    backend: &CompiledCompressionBackend,
) -> InfotheoryResult<u64> {
    let mut runtime = crate::runtime::build_compression_runtime(backend)
        .map_err(InfotheoryError::invalid_backend_config)?;
    runtime.compress_size_chain(parts)
}

/// Compute compressed size (bytes) for `data` using `backend`.
pub fn try_compress_size_backend(
    data: &[u8],
    backend: &CompiledCompressionBackend,
) -> InfotheoryResult<u64> {
    let mut runtime = crate::runtime::build_compression_runtime(backend)
        .map_err(InfotheoryError::invalid_backend_config)?;
    runtime.compress_size(data)
}

/// Compress `data` with `backend` and return encoded bytes.
pub fn try_compress_bytes_backend(
    data: &[u8],
    backend: &CompiledCompressionBackend,
) -> InfotheoryResult<Vec<u8>> {
    let mut runtime = crate::runtime::build_compression_runtime(backend)
        .map_err(InfotheoryError::invalid_backend_config)?;
    runtime.compress_bytes(data)
}

/// Decompress `input` with `backend` and return decoded bytes.
pub fn try_decompress_bytes_backend(
    input: &[u8],
    backend: &CompiledCompressionBackend,
) -> InfotheoryResult<Vec<u8>> {
    let mut runtime = crate::runtime::build_compression_runtime(backend)
        .map_err(InfotheoryError::invalid_backend_config)?;
    runtime.decompress_bytes(input)
}

/// Normalized compression-distance formula variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcdVariant {
    /// Vitanyi-style NCD: `(C(xy) - min(C(x), C(y))) / max(C(x), C(y))`.
    Vitanyi,
    /// Symmetric Vitanyi-style NCD using `min(C(xy), C(yx))`.
    SymVitanyi,
    /// Constructive NCD: `(C(xy) - min(C(x), C(y))) / C(xy)`.
    Cons,
    /// Symmetric constructive NCD using `min(C(xy), C(yx))` as denominator.
    SymCons,
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
/// Compute NCD for byte slices using the thread-local default context.
pub fn try_ncd_bytes_default(x: &[u8], y: &[u8], variant: NcdVariant) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ncd_bytes(x, y, variant))
}

/// Compute NCD for byte slices with an explicit compression backend.
pub fn try_ncd_bytes_backend(
    x: &[u8],
    y: &[u8],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
) -> InfotheoryResult<f64> {
    try_ncd_bytes_backend_with_options(x, y, backend, variant, NcdComputeOptions::default())
}

/// Compute NCD for byte slices with an explicit compression backend and
/// explicit operation-level parallelism controls.
pub fn try_ncd_bytes_backend_with_options(
    x: &[u8],
    y: &[u8],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
    options: NcdComputeOptions,
) -> InfotheoryResult<f64> {
    let compute = || -> InfotheoryResult<f64> {
        let (cx, cy) = rayon::join(
            || try_compress_size_backend(x, backend),
            || try_compress_size_backend(y, backend),
        );
        let cx = cx?;
        let cy = cy?;

        let cxy = try_compress_size_chain_backend(&[x, y], backend)?;

        let cyx = match variant {
            NcdVariant::SymVitanyi | NcdVariant::SymCons => {
                Some(try_compress_size_chain_backend(&[y, x], backend)?)
            }
            _ => None,
        };

        Ok(ncd_from_sizes(cx, cy, cxy, cyx, variant))
    };

    match options.parallelism {
        OperationParallelism::Serial => {
            let cx = try_compress_size_backend(x, backend)?;
            let cy = try_compress_size_backend(y, backend)?;
            let cxy = try_compress_size_chain_backend(&[x, y], backend)?;
            let cyx = match variant {
                NcdVariant::SymVitanyi | NcdVariant::SymCons => {
                    Some(try_compress_size_chain_backend(&[y, x], backend)?)
                }
                _ => None,
            };
            Ok(ncd_from_sizes(cx, cy, cxy, cyx, variant))
        }
        OperationParallelism::Auto => compute(),
        OperationParallelism::Threads(threads) => {
            if threads <= 1 {
                return try_ncd_bytes_backend_with_options(
                    x,
                    y,
                    backend,
                    variant,
                    NcdComputeOptions {
                        parallelism: OperationParallelism::Serial,
                    },
                );
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|err| {
                    InfotheoryError::runtime(format!("failed to build rayon pool: {err}"))
                })?;
            pool.install(compute)
        }
    }
}

/// Compute an `n x n` pairwise NCD matrix (row-major) using the thread-local default context.
///
/// `out[i * n + j]` corresponds to `NCD(datas[i], datas[j])`.
pub fn try_ncd_matrix_bytes_default(
    datas: &[Vec<u8>],
    variant: NcdVariant,
) -> InfotheoryResult<Vec<f64>> {
    with_default_ctx(|ctx| try_ncd_matrix_bytes_backend(datas, &ctx.compression_backend, variant))
}

/// Compute an `n x n` pairwise NCD matrix (row-major) with an explicit compression backend.
///
/// `out[i * n + j]` corresponds to `NCD(datas[i], datas[j])`.
pub fn try_ncd_matrix_bytes_backend(
    datas: &[Vec<u8>],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
) -> InfotheoryResult<Vec<f64>> {
    try_ncd_matrix_bytes_backend_with_options(datas, backend, variant, NcdComputeOptions::default())
}

/// Compute an `n x n` pairwise NCD matrix with explicit operation-level
/// parallelism controls.
pub fn try_ncd_matrix_bytes_backend_with_options(
    datas: &[Vec<u8>],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
    options: NcdComputeOptions,
) -> InfotheoryResult<Vec<f64>> {
    let compute = || try_ncd_matrix_bytes_backend_impl(datas, backend, variant);
    match options.parallelism {
        OperationParallelism::Serial => {
            try_ncd_matrix_bytes_backend_serial(datas, backend, variant)
        }
        OperationParallelism::Auto => compute(),
        OperationParallelism::Threads(threads) => {
            if threads <= 1 {
                return try_ncd_matrix_bytes_backend_serial(datas, backend, variant);
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|err| {
                    InfotheoryError::runtime(format!("failed to build rayon pool: {err}"))
                })?;
            pool.install(compute)
        }
    }
}

fn try_ncd_matrix_bytes_backend_impl(
    datas: &[Vec<u8>],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
) -> InfotheoryResult<Vec<f64>> {
    let n = datas.len();
    let cx = datas
        .par_iter()
        .map(|d| try_compress_size_backend(d, backend))
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<InfotheoryResult<Vec<_>>>()?;

    let mut out = vec![0.0f64; n * n];

    match variant {
        NcdVariant::SymVitanyi | NcdVariant::SymCons => {
            let pairs = (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .collect::<Vec<_>>();
            let pair_results = pairs
                .into_par_iter()
                .map(|(i, j)| -> InfotheoryResult<(usize, usize, f64)> {
                    let x = &datas[i];
                    let y = &datas[j];
                    let cxy = try_compress_size_chain_backend(&[x, y], backend)?;
                    let cyx = try_compress_size_chain_backend(&[y, x], backend)?;

                    let d = ncd_from_sizes(cx[i], cx[j], cxy, Some(cyx), variant);
                    Ok((i, j, d))
                })
                .collect::<Vec<_>>();
            for entry in pair_results {
                let (i, j, d) = entry?;
                out[i * n + j] = d;
                out[j * n + i] = d;
            }
        }
        NcdVariant::Vitanyi | NcdVariant::Cons => {
            let rows = (0..n)
                .into_par_iter()
                .map(|i| -> InfotheoryResult<Vec<(usize, usize, f64)>> {
                    let x = &datas[i];
                    let mut row = Vec::with_capacity(n);
                    for j in 0..n {
                        let d = if i == j {
                            0.0
                        } else {
                            let y = &datas[j];
                            let cxy = try_compress_size_chain_backend(&[x, y], backend)?;
                            ncd_from_sizes(cx[i], cx[j], cxy, None, variant)
                        };
                        row.push((i, j, d));
                    }
                    Ok(row)
                })
                .collect::<Vec<_>>();
            for row in rows {
                for (i, j, d) in row? {
                    out[i * n + j] = d;
                }
            }
        }
    }

    Ok(out)
}

fn try_ncd_matrix_bytes_backend_serial(
    datas: &[Vec<u8>],
    backend: &CompiledCompressionBackend,
    variant: NcdVariant,
) -> InfotheoryResult<Vec<f64>> {
    let n = datas.len();
    let mut cx = Vec::with_capacity(n);
    for d in datas {
        cx.push(try_compress_size_backend(d, backend)?);
    }
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                out[i * n + j] = 0.0;
                continue;
            }
            let cxy = try_compress_size_chain_backend(
                &[datas[i].as_slice(), datas[j].as_slice()],
                backend,
            )?;
            let cyx = match variant {
                NcdVariant::SymVitanyi | NcdVariant::SymCons => {
                    Some(try_compress_size_chain_backend(
                        &[datas[j].as_slice(), datas[i].as_slice()],
                        backend,
                    )?)
                }
                _ => None,
            };
            out[i * n + j] = ncd_from_sizes(cx[i], cx[j], cxy, cyx, variant);
        }
    }
    Ok(out)
}
