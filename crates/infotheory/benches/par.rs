use infotheory::api::{
    CompressionBackend, CompressionPathBatchOptions, OperationParallelism,
    try_get_compressed_sizes_from_paths_backend_with_options,
};
use infotheory::spec::CompiledCompressionBackend;
use std::hint::black_box;
use std::time::Instant;

const RUNS: usize = 32;
const PATHS: &[&str] = &["compressme", "scompressme", "largebench"];

fn main() {
    let backend = CompressionBackend::try_default()
        .expect("default compression backend")
        .compile()
        .expect("compiled default compression backend");

    benchmark_variant(
        "serial",
        PATHS,
        &backend,
        CompressionPathBatchOptions {
            parallelism: OperationParallelism::Serial,
        },
    );
    benchmark_variant(
        "auto",
        PATHS,
        &backend,
        CompressionPathBatchOptions {
            parallelism: OperationParallelism::Auto,
        },
    );
    benchmark_variant(
        "threads(4)",
        PATHS,
        &backend,
        CompressionPathBatchOptions {
            parallelism: OperationParallelism::Threads(4),
        },
    );
}

fn benchmark_variant(
    label: &str,
    paths: &[&str],
    backend: &CompiledCompressionBackend,
    options: CompressionPathBatchOptions,
) {
    println!(
        "{label} warmup: {:?}",
        get_compressed_sizes(paths, backend, options)
    );
    let now = Instant::now();
    for _ in 0..RUNS {
        black_box(get_compressed_sizes(paths, backend, options));
    }
    println!("{label}: elapsed time for {RUNS} runs: {:?}", now.elapsed());
}

fn get_compressed_sizes(
    paths: &[&str],
    backend: &CompiledCompressionBackend,
    options: CompressionPathBatchOptions,
) -> Vec<u64> {
    try_get_compressed_sizes_from_paths_backend_with_options(paths, backend, options)
        .expect("compressed sizes")
}
