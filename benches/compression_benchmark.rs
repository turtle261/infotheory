use infotheory::{get_compressed_size, get_compressed_size_v2};
use std::time::Instant;

fn main() {
    let methods = vec!["5", "4", "3", "2", "1", "x4.3ci1"];
    let files = vec![("./compressme", "small"), ("./largebench", "large")];

    println!("\n{}", "=".repeat(60));
    println!("Compression Benchmark: get_compressed_size vs get_compressed_size_v2");
    println!("{}", "=".repeat(60));
    println!();

    for (file_path, file_label) in &files {
        let file_len = std::fs::metadata(file_path)
            .map(|m| m.len())
            .unwrap_or(0);
        println!("\n📁 File: {} ({})", file_label, file_path);
        println!("  Size: {} bytes", file_len);
        println!("{}", "-".repeat(60));

        for method in &methods {
            println!("\n  Method: '{}' ", method);
            println!("  {}", "-".repeat(50));

            // Heuristic: more iterations for fast/small cases to reduce noise.
            let iterations = if file_len <= 64 * 1024 {
                if *method == "1" { 200 }
                else if *method == "2" { 100 }
                else if *method == "3" { 80 }
                else { 30 }
            } else {
                if *method == "5" { 3 } else { 5 }
            };

            // Warmup
            for _ in 0..2 {
                let _ = get_compressed_size(file_path, method);
                let _ = get_compressed_size_v2(file_path, method);
            }

            // Benchmark get_compressed_size (v1)
            let mut times_v1 = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                let result = get_compressed_size(file_path, method);
                let elapsed = start.elapsed();
                times_v1.push(elapsed);
                if iterations <= 10 {
                    println!("  v1: {:?} -> {} bytes", elapsed, result);
                }
            }

            // Benchmark get_compressed_size_v2 (v2)
            let mut times_v2 = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                let result = get_compressed_size_v2(file_path, method);
                let elapsed = start.elapsed();
                times_v2.push(elapsed);
                if iterations <= 10 {
                    println!("  v2: {:?} -> {} bytes", elapsed, result);
                }
            }

            // Calculate averages
            let avg_v1 = times_v1.iter().sum::<std::time::Duration>() / times_v1.len() as u32;
            let avg_v2 = times_v2.iter().sum::<std::time::Duration>() / times_v2.len() as u32;
            let min_v1 = *times_v1.iter().min().unwrap();
            let min_v2 = *times_v2.iter().min().unwrap();

            let diff_pct = if avg_v1 > avg_v2 {
                (avg_v1 - avg_v2).as_nanos() as f64 / avg_v1.as_nanos() as f64 * 100.0
            } else {
                -((avg_v2 - avg_v1).as_nanos() as f64 / avg_v2.as_nanos() as f64 * 100.0)
            };

            let faster = if avg_v2 < avg_v1 {
                "v2 FASTER"
            } else {
                "v1 FASTER"
            };

            println!("\n  Summary for method '{}':", method);
            println!("    iterations: {}", iterations);
            println!("    v1 avg/min: {:?} / {:?}", avg_v1, min_v1);
            println!("    v2 avg/min: {:?} / {:?}", avg_v2, min_v2);
            println!("    {} ({:.1}%)", faster, diff_pct.abs());
        }

        println!("\n");
    }

    println!("\n{}", "=".repeat(60));
    println!("Benchmark complete!");
    println!("{}", "=".repeat(60));
    println!();
}
