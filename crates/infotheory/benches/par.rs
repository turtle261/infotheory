use infotheory::api::{
    try_get_parallel_compressed_sizes_from_parallel_paths,
    try_get_parallel_compressed_sizes_from_sequential_paths,
    try_get_sequential_compressed_sizes_from_parallel_paths,
    try_get_sequential_compressed_sizes_from_sequential_paths,
};
use std::time::Instant;

fn main() {
    let paths: Vec<&str> = vec!["compressme", "scompressme", "largebench"];
    for _ in 0..32 {
        println!(
            "{:?}",
            get_sequential_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1")
        );
    }
    let now = Instant::now();
    for _ in 0..32 {
        println!(
            "{:?}",
            get_sequential_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1")
        );
    }
    let elapsed = now.elapsed();
    println!("seq_seq: Elapsed time for 32 runs: {:?}", elapsed);
    for _ in 0..32 {
        println!(
            "{:?}",
            get_sequential_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1")
        );
    }
    let now = Instant::now();
    for _ in 0..32 {
        println!(
            "{:?}",
            get_sequential_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1")
        );
    }
    let elapsed = now.elapsed();
    println!("seq_par: Elapsed time for 32 runs: {:?}", elapsed);
    for _ in 0..32 {
        println!(
            "{:?}",
            get_parallel_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1", 4)
        );
    }
    let now = Instant::now();
    for _ in 0..32 {
        println!(
            "{:?}",
            get_parallel_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1", 4)
        );
    }
    let elapsed = now.elapsed();
    println!("par_seq: Elapsed time for 32 runs: {:?}", elapsed);
    for _ in 0..32 {
        println!(
            "{:?}",
            get_parallel_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1", 4)
        );
    }
    let now = Instant::now();
    for _ in 0..32 {
        println!(
            "{:?}",
            get_parallel_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1", 4)
        );
    }
    let elapsed = now.elapsed();
    println!("par_par: Elapsed time for 32 runs: {:?}", elapsed);
    let now = Instant::now();
    for _ in 0..32 {
        println!("{:?}", get_compressed_sizes_from_paths(&paths, "x4.3ci1"));
    }
    let elapsed = now.elapsed();
    println!("par_par: Elapsed time for 32 runs: {:?}", elapsed);
}
fn get_sequential_compressed_sizes_from_sequential_paths(paths: &[&str], method: &str) -> Vec<u64> {
    try_get_sequential_compressed_sizes_from_sequential_paths(paths, method).expect("sizes")
}

fn get_sequential_compressed_sizes_from_parallel_paths(paths: &[&str], method: &str) -> Vec<u64> {
    try_get_sequential_compressed_sizes_from_parallel_paths(paths, method).expect("sizes")
}

fn get_parallel_compressed_sizes_from_sequential_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> Vec<u64> {
    try_get_parallel_compressed_sizes_from_sequential_paths(paths, method, threads).expect("sizes")
}

fn get_parallel_compressed_sizes_from_parallel_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> Vec<u64> {
    try_get_parallel_compressed_sizes_from_parallel_paths(paths, method, threads).expect("sizes")
}
