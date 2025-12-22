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
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{}", ned_paths(file1, file2, max_order));
        }
        "ned_cons" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let (bx, by) = rayon::join(
                || std::fs::read(file1).expect("failed to read file1"),
                || std::fs::read(file2).expect("failed to read file2"),
            );
            println!("{}", ned_cons_bytes(&bx, &by, max_order));
        }
        "nte" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{}", nte_paths(file1, file2, max_order));
        }
        "tvd" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{}", tvd_paths(file1, file2, max_order));
        }
        "nhd" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{}", nhd_paths(file1, file2, max_order));
        }
        "entropy" | "h" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let data = std::fs::read(file1).expect("failed to read file");
            if max_order == 0 {
                println!("{}", marginal_entropy_bytes(&data));
            } else {
                println!("{}", entropy_rate_bytes(&data, max_order));
            }
        }
        "entropy_rate" | "h_rate" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let data = std::fs::read(file1).expect("failed to read file");
            println!("{}", entropy_rate_bytes(&data, max_order));
        }
        "joint_entropy" | "h_xy" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let (bx, by) = rayon::join(
                || std::fs::read(file1).expect("failed to read file1"),
                || std::fs::read(file2).expect("failed to read file2"),
            );
            if max_order == 0 {
                println!("{}", joint_marginal_entropy_bytes(&bx, &by));
            } else {
                println!("{}", joint_entropy_rate_bytes(&bx, &by, max_order));
            }
        }
        "mi" | "mutual_info" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{}", mutual_information_paths(file1, file2, max_order));
        }
        "ce" | "conditional_entropy" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{}", conditional_entropy_paths(file1, file2, max_order));
        }
        "xe" | "cross_entropy" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{}", cross_entropy_paths(file1, file2, max_order));
        }
        "kl" | "kl_divergence" => {
            println!("{}", kl_divergence_paths(file1, file2));
        }
        "js" | "js_divergence" => {
            println!("{}", js_divergence_paths(file1, file2));
        }
        "id" | "intrinsic_dep" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let bx = std::fs::read(file1).expect("failed to read file");
            println!("{}", intrinsic_dependence_bytes(&bx, max_order));
        }
        "rt" | "resistance" => {
            let max_order = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let (bx, btx) = rayon::join(
                || std::fs::read(file1).expect("failed to read file1"),
                || std::fs::read(file2).expect("failed to read file2"),
            );
            println!("{}", resistance_to_transformation_bytes(&bx, &btx, max_order));
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
