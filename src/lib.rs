const THREADS: usize = 12;

#[inline(always)]
pub fn get_compressed_size(path: &str, method: &str) -> u64 {
    // Convert Input file to Vec<u8>, and reference that (compress_size only takes &[u8] input), and pass method.
    // Will panic if file does not exist, so it must be prevalidated.
    zpaq_rs::compress_size_parallel(&std::fs::read(path).unwrap(), method, THREADS).unwrap()

}

pub fn get_compressed_size_archival(path: &str, method: &str) -> u64 {
    // Only ideal for methods 1-5 in some cases where you want to count deduplication as compression.
    zpaq_rs::zpaq_add_archive_size_file(path, method, THREADS).expect("compression failed")
}


#[inline(always)]
pub fn get_bytes_from_paths(paths: &[&str]) -> Vec<Vec<u8>> {
    paths
        .iter()
        .map(|path| std::fs::read(*path).expect("failed to read file"))
        .collect()
}

#[inline(always)]
pub fn get_compressed_sizes_from_paths(paths: &[&str], method: &str) -> Vec<u64> {
    get_bytes_from_paths(paths).iter()
        .map(|data| zpaq_rs::compress_size_parallel(data, method, THREADS).unwrap())
        .collect()
}