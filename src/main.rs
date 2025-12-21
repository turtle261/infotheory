use infotheory::get_compressed_size;
use infotheory::get_compressed_sizes_from_paths;

fn main() {
    println!("{}", get_compressed_size("compressme", "5"));
    println!("{}", get_compressed_size("scompressme", "5"));

    let paths: Vec<&str> = vec!["compressme", "scompressme"];
    println!("{:?}", get_compressed_sizes_from_paths(&paths, "5"));
}
