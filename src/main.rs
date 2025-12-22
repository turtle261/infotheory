use infotheory::*;
use std::env;

fn read_file(path: &str) -> Vec<u8> {
    match std::fs::read(path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        return;
    }

    let primitive = &args[1];

    match primitive.as_str() {
        // Entropy-based primitives
        "ned" | "nte" | "tvd" | "nhd" | "mi" | "mutual_info" | "ce" | "conditional_entropy" | "xe" | "cross_entropy" | "joint_entropy" | "h_xy" | "kl" | "kl_divergence" | "js" | "js_divergence" => {
            if args.len() < 4 {
                eprintln!("Error: '{}' requires two files.", primitive);
                std::process::exit(1);
            }
            let file1 = &args[2];
            let file2 = &args[3];
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

            match primitive.as_str() {
                "ned" => println!("{}", ned_paths(file1, file2, max_order)),
                "nte" => println!("{}", nte_paths(file1, file2, max_order)),
                "tvd" => println!("{}", tvd_paths(file1, file2, max_order)),
                "nhd" => println!("{}", nhd_paths(file1, file2, max_order)),
                "mi" | "mutual_info" => println!("{}", mutual_information_paths(file1, file2, max_order)),
                "ce" | "conditional_entropy" => println!("{}", conditional_entropy_paths(file1, file2, max_order)),
                "xe" | "cross_entropy" => println!("{}", cross_entropy_paths(file1, file2, max_order)),
                "kl" | "kl_divergence" => println!("{}", kl_divergence_paths(file1, file2)),
                "js" | "js_divergence" => println!("{}", js_divergence_paths(file1, file2)),
                "joint_entropy" | "h_xy" => {
                    let bx = read_file(file1);
                    let by = read_file(file2);
                    if max_order == 0 {
                        println!("{}", joint_marginal_entropy_bytes(&bx, &by));
                    } else {
                        println!("{}", joint_entropy_rate_bytes(&bx, &by, max_order));
                    }
                }
                _ => unreachable!(),
            }
        }

        "ned_cons" => {
            if args.len() < 4 {
                eprintln!("Error: 'ned_cons' requires two files.");
                std::process::exit(1);
            }
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let bx = read_file(&args[2]);
            let by = read_file(&args[3]);
            println!("{}", ned_cons_bytes(&bx, &by, max_order));
        }

        "entropy" | "h" | "entropy_rate" | "h_rate" => {
            let data = read_file(&args[2]);
            let default_order = if primitive.contains("rate") { 8 } else { 0 };
            let max_order = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(default_order);
            
            if max_order == 0 && !primitive.contains("rate") {
                println!("{}", marginal_entropy_bytes(&data));
            } else {
                println!("{}", entropy_rate_bytes(&data, max_order));
            }
        }

        "id" | "intrinsic_dep" => {
            let data = read_file(&args[2]);
            let max_order = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
            let h_marginal = marginal_entropy_bytes(&data);
            let h_rate = entropy_rate_bytes(&data, max_order);
            let ratio = if h_marginal == 0.0 { 0.0 } else { h_rate / h_marginal };
            println!("{:.6} (Rate: {:.4}, Marg: {:.4})", ratio.clamp(0.0, 1.0), h_rate, h_marginal);
        }

        "rt" | "resistance" => {
            if args.len() < 4 {
                eprintln!("Error: 'rt' requires two files (original and transformed).");
                std::process::exit(1);
            }
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let bx = read_file(&args[2]);
            let btx = read_file(&args[3]);
            println!("{}", resistance_to_transformation_bytes(&bx, &btx, max_order));
        }

        "ncd" | "ncd_vitanyi" | "ncd_sym" | "ncd_sym_vitanyi" | "ncd_cons" | "ncd_sym_cons" => {
            if args.len() < 4 {
                eprintln!("Error: NCD primitives require two files.");
                std::process::exit(1);
            }
            let file1 = &args[2];
            let file2 = &args[3];
            let method = args.get(4).map(|s| s.as_str()).unwrap_or("5");
            match primitive.as_str() {
                "ncd" | "ncd_vitanyi" => println!("{}", ncd_vitanyi(file1, file2, method)),
                "ncd_sym" | "ncd_sym_vitanyi" => println!("{}", ncd_sym_vitanyi(file1, file2, method)),
                "ncd_cons" => println!("{}", ncd_cons(file1, file2, method)),
                "ncd_sym_cons" => println!("{}", ncd_sym_cons(file1, file2, method)),
                _ => unreachable!(),
            }
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
    eprintln!("Entropy-based (dispatch: max_order=0 for Marginal, !=0 for Rate):");
    eprintln!("  ned                    Normalized Entropy Distance");
    eprintln!("  ned_cons               NED Conservative");
    eprintln!("  nte                    Normalized Transform Effort (VI)");
    eprintln!("  tvd                    Total Variation Distance (Marginal only)");
    eprintln!("  nhd                    Normalized Hellinger Distance (Marginal only)");
    eprintln!();
    eprintln!("Information measures:");
    eprintln!("  entropy, h             Shannon entropy H(X) (marginal if no order)");
    eprintln!("  entropy_rate, h_rate   Unbiased predictive entropy rate (ROSA)");
    eprintln!("  joint_entropy, h_xy    Joint entropy H(X,Y) (uses max_order if provided)");
    eprintln!("  mi, mutual_info        Mutual info I(X;Y) (uses max_order if provided)");
    eprintln!("  ce, conditional_entropy Conditional entropy H(X|Y) (uses max_order if provided)");
    eprintln!("  xe, cross_entropy      Cross-entropy H(P,Q)");
    eprintln!("  kl, kl_divergence      KL Divergence D_KL(P||Q) (marginal only)");
    eprintln!("  js, js_divergence      JS Divergence JSD(P||Q) (marginal only)");
    eprintln!();
    eprintln!("Structural measures:");
    eprintln!("  id, intrinsic_dep      Primitive 6: Intrinsic vs Extrinsic Dependence");
    eprintln!("  rt, resistance         Primitive 7: Resistance to Transformation");
    eprintln!("  [max_order]: ROSA order (default: 8, 0 = Marginal-only, -1 = Unlimited)");
}
