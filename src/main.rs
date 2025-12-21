//use infotheory::get_compressed_size;
use infotheory::*;

fn main() {
    //println!("{}", get_compressed_size("compressme", "5"));
    //println!("{}", get_compressed_size("scompressme", "5"));

    let paths: Vec<&str> = vec!["compressme", "scompressme", "largebench"];
    for i in 0..32 {
        println!("{:?}", get_sequential_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1"));
    }
    //println!("{:?}", get_parallel_compressed_sizes_from_sequential_paths(&paths, "x4.3ci1"));
    //println!("{:?}", get_sequential_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1"));
    //println!("{:?}", get_parallel_compressed_sizes_from_parallel_paths(&paths, "x4.3ci1"));
}
