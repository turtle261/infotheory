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
        } else {
            // Try platform-specific default paths
            #[cfg(unix)]
            let default_paths = vec!["/opt/cuda", "/usr/local/cuda"];
            
            #[cfg(windows)]
            let default_paths: Vec<&str> = vec![
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0",
            ];
            
            #[cfg(not(any(unix, windows)))]
            let default_paths: Vec<&str> = vec![];
            
            let mut found = false;
            for path in default_paths {
                if std::path::Path::new(path).exists() {
                    build_cuda_kernel(path);
                    found = true;
                    break;
                }
            }
            
            if !found {
                println!("cargo:warning=CUDA toolkit not found, WKV kernel will use fallback");
            }
        }
    }
}

#[cfg(feature = "training")]
fn build_cuda_kernel(cuda_path: &str) {
    // CUDA library path differs between platforms
    #[cfg(unix)]
    let cuda_lib = format!("{}/lib64", cuda_path);
    
    #[cfg(windows)]
    let cuda_lib = format!("{}\\lib\\x64", cuda_path);
    
    #[cfg(not(any(unix, windows)))]
    let cuda_lib = format!("{}/lib", cuda_path);

    // Find nvcc (platform-specific path separator)
    #[cfg(unix)]
    let nvcc = format!("{}/bin/nvcc", cuda_path);
    
    #[cfg(windows)]
    let nvcc = format!("{}\\bin\\nvcc.exe", cuda_path);
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

    // Find appropriate host compiler (platform-specific)
    #[cfg(unix)]
    let (ccbin_flag, ccbin_value) = {
        let gcc14 = if std::path::Path::new("/usr/bin/gcc-14").exists() {
            "/usr/bin/gcc-14"
        } else {
            println!("cargo:warning=gcc-14 not found, using default gcc");
            "gcc"
        };
        (Some("-ccbin"), Some(gcc14))
    };
    
    #[cfg(windows)]
    let (ccbin_flag, ccbin_value): (Option<&str>, Option<&str>) = {
        // On Windows, nvcc uses MSVC by default, so we don't need to specify -ccbin
        (None, None)
    };
    
    #[cfg(not(any(unix, windows)))]
    let (ccbin_flag, ccbin_value): (Option<&str>, Option<&str>) = (None, None);

    // Compile inference kernel
    let mut nvcc_cmd = std::process::Command::new(&nvcc);
    nvcc_cmd.args([
        "-c",
        kernel_src,
        "-o",
        &obj_file,
        "-O3",
    ]);
    
    // Add platform-specific compiler options
    #[cfg(unix)]
    nvcc_cmd.args(["--compiler-options", "-fPIC"]);
    
    // Add ccbin if specified (UNIX only)
    if let (Some(flag), Some(value)) = (ccbin_flag, ccbin_value) {
        nvcc_cmd.args([flag, value]);
    }
    
    nvcc_cmd.args([
        "-D_N_=64", // Head dimension
        "-gencode",
        "arch=compute_61,code=sm_61",  // Pascal (P2000)
        "-Wno-deprecated-gpu-targets", // Suppress old arch warning
    ]);
    
    let compile_status = nvcc_cmd.status();

    let compile_ok = match &compile_status {
        Ok(s) if s.success() => true,
        _ => {
            println!("cargo:warning=Inference kernel compilation failed");
            false
        }
    };

    // Compile training kernel with checkpointing
    let train_compile_ok = if std::path::Path::new(train_kernel_src).exists() {
        let mut train_cmd = std::process::Command::new(&nvcc);
        train_cmd.args([
            "-c",
            train_kernel_src,
            "-o",
            &train_obj_file,
            "-O3",
        ]);
        
        // Add platform-specific compiler options
        #[cfg(unix)]
        train_cmd.args(["--compiler-options", "-fPIC"]);
        
        // Add ccbin if specified (UNIX only)
        if let (Some(flag), Some(value)) = (ccbin_flag, ccbin_value) {
            train_cmd.args([flag, value]);
        }
        
        train_cmd.args([
            "-D_N_=64",
            "-D_CHUNK_LEN_=32", // Checkpoint every 32 timesteps
            "-gencode",
            "arch=compute_61,code=sm_61",
            "-Wno-deprecated-gpu-targets",
        ]);
        
        let train_status = train_cmd.status();

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
        // Platform-specific archiver
        #[cfg(unix)]
        let ar_result = {
            let mut ar_args = vec!["rcs".to_string(), lib_file.clone()];
            if compile_ok {
                ar_args.push(obj_file.clone());
            }
            if train_compile_ok {
                ar_args.push(train_obj_file.clone());
            }
            std::process::Command::new("ar").args(&ar_args).status()
        };
        
        #[cfg(windows)]
        let ar_result = {
            // On Windows, use lib.exe (MSVC's librarian)
            let mut lib_args = vec![
                format!("/OUT:{}", lib_file),
                "/NOLOGO".to_string(),
            ];
            if compile_ok {
                lib_args.push(obj_file.clone());
            }
            if train_compile_ok {
                lib_args.push(train_obj_file.clone());
            }
            std::process::Command::new("lib.exe").args(&lib_args).status()
        };
        
        #[cfg(not(any(unix, windows)))]
        let ar_result: std::io::Result<std::process::ExitStatus> = {
            Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "Unsupported platform"))
        };

        match ar_result {
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
