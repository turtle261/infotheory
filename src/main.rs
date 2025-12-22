use infotheory::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 4 {
        print_usage();
        return;
    }

    let primitive = &args[1];
    let file1 = &args[2];
    let file2 = &args[3];

    match primitive.as_str() {
        // NCD variants (ZPAQ-based)
        "ncd" | "ncd_vitanyi" => {
            let method = args.get(4).map(|s| s.as_str()).unwrap_or("5");
            println!("{}", ncd_vitanyi(file1, file2, method));
        }
        "ncd_sym" | "ncd_sym_vitanyi" => {
            let method = args.get(4).map(|s| s.as_str()).unwrap_or("5");
            println!("{}", ncd_sym_vitanyi(file1, file2, method));
        }
        "ncd_cons" => {
            let method = args.get(4).map(|s| s.as_str()).unwrap_or("5");
            println!("{}", ncd_cons(file1, file2, method));
        }
        "ncd_sym_cons" => {
            let method = args.get(4).map(|s| s.as_str()).unwrap_or("5");
            println!("{}", ncd_sym_cons(file1, file2, method));
        }
        
        // Entropy-based primitives (ROSA-based)
        "ned" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            println!("{}", ned_paths(file1, file2, max_order));
        }
        "ned_cons" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let (bx, by) = rayon::join(
                || std::fs::read(file1).expect("failed to read file1"),
                || std::fs::read(file2).expect("failed to read file2"),
            );
            println!("{}", ned_cons_bytes(&bx, &by, max_order));
        }
        "nte" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            println!("{}", nte_paths(file1, file2, max_order));
        }
        "tvd" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            println!("{}", tvd_paths(file1, file2, max_order));
        }
        "nhd" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            println!("{}", nhd_paths(file1, file2, max_order));
        }
        "entropy" | "h" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let data = std::fs::read(file1).expect("failed to read file");
            println!("H(X) = {}", entropy_rate_bytes(&data, max_order));
        }
        "joint_entropy" | "h_xy" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let (bx, by) = rayon::join(
                || std::fs::read(file1).expect("failed to read file1"),
                || std::fs::read(file2).expect("failed to read file2"),
            );
            println!("H(X,Y) = {}", joint_entropy_rate_bytes(&bx, &by, max_order));
        }
        "mi" | "mutual_info" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let (bx, by) = rayon::join(
                || std::fs::read(file1).expect("failed to read file1"),
                || std::fs::read(file2).expect("failed to read file2"),
            );
            println!("I(X;Y) = {}", mutual_information_bytes(&bx, &by, max_order));
        }
        
        _ => {
            eprintln!("Unknown primitive: {}", primitive);
            print_usage();
        }
    }
}

fn print_usage() {
    eprintln!("Usage: infotheory <primitive> <file1> <file2> [method/max_order]");
    eprintln!();
    eprintln!("Compression-based (NCD via ZPAQ):");
    eprintln!("  ncd, ncd_vitanyi       NCD Vitanyi formula");
    eprintln!("  ncd_sym, ncd_sym_vitanyi  Symmetric NCD Vitanyi");
    eprintln!("  ncd_cons               NCD Conservative");
    eprintln!("  ncd_sym_cons           Symmetric NCD Conservative");
    eprintln!("  [method]: ZPAQ method (default: \"5\"), e.g. \"1\", \"5\", \"x4.3ci1\"");
    eprintln!();
    eprintln!("Entropy-based (via ROSA):");
    eprintln!("  ned                    Normalized Entropy Distance");
    eprintln!("  ned_cons               NED Conservative");
    eprintln!("  nte                    Normalized Transform Effort (VI)");
    eprintln!("  tvd                    Total Variation Distance");
    eprintln!("  nhd                    Normalized Hellinger Distance");
    eprintln!("  entropy, h             Entropy rate H(X) (uses file1 only)");
    eprintln!("  joint_entropy, h_xy    Joint entropy H(X,Y)");
    eprintln!("  mi, mutual_info        Mutual information I(X;Y)");
    eprintln!("  [max_order]: ROSA max context order (default: 8, -1 = unlimited)");
}
