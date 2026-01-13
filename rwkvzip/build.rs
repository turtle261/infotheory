// build.rs - Build script for compiling CUDA kernels
//
// This compiles the WKV7 CUDA kernel and links it with the Rust binary.
// The user has GCC14 installed, and GCC15 -- we should use GCC14 only for GCC. The current
// version tries to force GCC15, but should use GCC14 instead.

fn main() {
    #[cfg(feature = "training")]
    {
        // Check if CUDA toolkit is available
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            build_cuda_kernel(&cuda_path);
        } else if std::path::Path::new("/opt/cuda").exists() {
            build_cuda_kernel("/opt/cuda");
        } else if std::path::Path::new("/usr/local/cuda").exists() {
            build_cuda_kernel("/usr/local/cuda");
        } else {
            println!("cargo:warning=CUDA toolkit not found, WKV kernel will use fallback");
        }
    }
}

#[cfg(feature = "training")]
fn build_cuda_kernel(cuda_path: &str) {
    let cuda_lib = format!("{}/lib64", cuda_path);

    // Find nvcc
    let nvcc = format!("{}/bin/nvcc", cuda_path);
    if !std::path::Path::new(&nvcc).exists() {
        println!(
            "cargo:warning=nvcc not found at {}, skipping CUDA kernel",
            nvcc
        );
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let kernel_src = "src/rwkv7/training/cuda/wkv7_kernel.cu";

    // Check if kernel source exists
    if !std::path::Path::new(kernel_src).exists() {
        println!(
            "cargo:warning=CUDA kernel source not found at {}",
            kernel_src
        );
        return;
    }

    // Training kernel with checkpointing
    let train_kernel_src = "src/rwkv7/training/cuda/wkv7_train.cu";

    let obj_file = format!("{}/wkv7_kernel.o", out_dir);
    let train_obj_file = format!("{}/wkv7_train.o", out_dir);
    let lib_file = format!("{}/libwkv7_kernel.a", out_dir);

    // Find gcc-14 (nvcc requires compatible host compiler, gcc-15 is too new)
    let gcc14 = if std::path::Path::new("/usr/bin/gcc-14").exists() {
        "/usr/bin/gcc-14"
    } else {
        println!("cargo:warning=gcc-14 not found, CUDA kernel may fail to compile");
        "gcc"
    };

    // Compile inference kernel
    let compile_status = std::process::Command::new(&nvcc)
        .args([
            "-c",
            kernel_src,
            "-o",
            &obj_file,
            "-O3",
            "--compiler-options",
            "-fPIC",
            "-ccbin",
            gcc14,      // Use gcc-14 as host compiler
            "-D_N_=64", // Head dimension
            "-gencode",
            "arch=compute_61,code=sm_61",  // Pascal (P2000)
            "-Wno-deprecated-gpu-targets", // Suppress old arch warning
        ])
        .status();

    let compile_ok = match &compile_status {
        Ok(s) if s.success() => true,
        _ => {
            println!("cargo:warning=Inference kernel compilation failed");
            false
        }
    };

    // Compile training kernel with checkpointing
    let train_compile_ok = if std::path::Path::new(train_kernel_src).exists() {
        let train_status = std::process::Command::new(&nvcc)
            .args([
                "-c",
                train_kernel_src,
                "-o",
                &train_obj_file,
                "-O3",
                "--compiler-options",
                "-fPIC",
                "-ccbin",
                gcc14,
                "-D_N_=64",
                "-D_CHUNK_LEN_=32", // Checkpoint every 32 timesteps
                "-gencode",
                "arch=compute_61,code=sm_61",
                "-Wno-deprecated-gpu-targets",
            ])
            .status();

        match train_status {
            Ok(s) if s.success() => true,
            _ => {
                println!("cargo:warning=Training kernel compilation failed");
                false
            }
        }
    } else {
        println!(
            "cargo:warning=Training kernel source not found at {}",
            train_kernel_src
        );
        false
    };

    if compile_ok || train_compile_ok {
        // Create static library from all compiled objects
        let mut ar_args = vec!["rcs".to_string(), lib_file.clone()];
        if compile_ok {
            ar_args.push(obj_file.clone());
        }
        if train_compile_ok {
            ar_args.push(train_obj_file.clone());
        }

        let ar_status = std::process::Command::new("ar").args(&ar_args).status();

        match ar_status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-link-search=native={}", out_dir);
                println!("cargo:rustc-link-search=native={}", cuda_lib);
                println!("cargo:rustc-link-lib=static=wkv7_kernel");
                println!("cargo:rustc-link-lib=cudart");
                println!("cargo:rustc-cfg=feature=\"cuda_wkv\"");
                println!("cargo:rerun-if-changed={}", kernel_src);
                if train_compile_ok {
                    println!("cargo:rerun-if-changed={}", train_kernel_src);
                }
                println!("cargo:warning=CUDA WKV kernels compiled successfully (inference={}, training={})", compile_ok, train_compile_ok);
            }
            _ => {
                println!("cargo:warning=Failed to create static library from CUDA kernels");
            }
        }
    } else {
        println!("cargo:warning=No CUDA kernels compiled successfully");
    }
}
