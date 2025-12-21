use infotheory::*;
use std::time::Instant;

fn main() {

    let paths: Vec<&str> = vec!["compressme", "scompressme", "largebench"];
    for i in 0..32 {
        println!("{:?}", get_sequential_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1"));
    }
    let now = Instant::now();
    for i in 0..32 {
        println!("{:?}", get_sequential_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1"));
    }
    let elapsed = now.elapsed();
    println!("seq_seq: Elapsed time for 32 runs: {:?}", elapsed);
    for i in 0..32 {
        println!("{:?}", get_sequential_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1"));
    }
    let now = Instant::now();
    for i in 0..32 {
        println!("{:?}", get_sequential_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1"));
    }
    let elapsed = now.elapsed();
    println!("seq_par: Elapsed time for 32 runs: {:?}", elapsed);
    for i in 0..32 {
        println!("{:?}", get_parallel_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1"));
    }
    let now = Instant::now();
    for i in 0..32 {
        println!("{:?}", get_parallel_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1"));
    }
    let elapsed = now.elapsed();
    println!("par_seq: Elapsed time for 32 runs: {:?}", elapsed);
    for i in 0..32 {
        println!("{:?}", get_parallel_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1"));
    }
    let now = Instant::now();
    for i in 0..32 {
        println!("{:?}", get_parallel_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1"));
    }
    let elapsed = now.elapsed();
    println!("par_par: Elapsed time for 32 runs: {:?}", elapsed);
}
