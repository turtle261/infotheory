use rosaplus::RosaPlus;
use std::time::Instant;

fn main() {
    // Keep this dependency-free (no criterion), consistent with workspace benches.
    let max_order = 8i64;
    let seed = 42u64;

    // Synthetic corpus: repeat a moderately-sized pattern to simulate larger priors.
    let pat = b"the quick brown fox jumps over the lazy dog\n";
    let repeats = 200_000usize;
    let mut corpus = Vec::with_capacity(pat.len() * repeats);
    for _ in 0..repeats {
        corpus.extend_from_slice(pat);
    }

    let query = b"the quick brown";

    let mut m = RosaPlus::new(max_order, false, 0, seed);

    let t0 = Instant::now();
    m.train_example(&corpus);
    let t_train = t0.elapsed();

    let t0 = Instant::now();
    m.build_lm_full_bytes_no_finalize_endpos();
    let t_build = t0.elapsed();

    let est_bytes = m.estimated_size_bytes();

    let t0 = Instant::now();
    let ce = m.cross_entropy(query);
    let t_score = t0.elapsed();

    let path = "/tmp/rosaplus_bench.rosa";
    let t0 = Instant::now();
    m.save(path).expect("save failed");
    let t_save = t0.elapsed();

    let t0 = Instant::now();
    let m2 = RosaPlus::load(path).expect("load failed");
    let t_load = t0.elapsed();

    let t0 = Instant::now();
    let ce2 = m2.cross_entropy(query);
    let t_score2 = t0.elapsed();

    println!("corpus_bytes={} query_bytes={}", corpus.len(), query.len());
    println!("train_time_ms={:.3}", t_train.as_secs_f64() * 1e3);
    println!("build_time_ms={:.3}", t_build.as_secs_f64() * 1e3);
    println!("estimated_model_bytes={}", est_bytes);
    println!("score_time_ms={:.3} cross_entropy={:.6}", t_score.as_secs_f64() * 1e3, ce);
    println!("save_time_ms={:.3}", t_save.as_secs_f64() * 1e3);
    println!("load_time_ms={:.3}", t_load.as_secs_f64() * 1e3);
    println!("score2_time_ms={:.3} cross_entropy2={:.6}", t_score2.as_secs_f64() * 1e3, ce2);
}
