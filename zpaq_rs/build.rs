use std::env;
use std::process::Command;

fn exe_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-changed=zpaq/libzpaq.cpp");
    println!("cargo:rerun-if-changed=zpaq/libzpaq.h");
    println!("cargo:rerun-if-changed=zpaq_rs_ffi.cpp");

    let mut build = cc::Build::new();

    // Toolchain selection:
    // - If `ZPAQ_RS_CXX` is set, use it (e.g. `clang++` or `g++`).
    // - Else, if the standard `CXX` env var is set, let the `cc` crate honor it.
    // - Otherwise, prefer clang++ when available (easy to compare with Rust's LLVM backend).
    if let Ok(cxx) = env::var("ZPAQ_RS_CXX") {
        if !cxx.trim().is_empty() {
            build.compiler(cxx);
        }
    } else if env::var_os("CXX").is_none() && exe_exists("clang++") {
        build.compiler("clang++");
    }

    build
        .cpp(true)
        .include("zpaq")
        .file("zpaq/libzpaq.cpp")
        .file("zpaq/zpaq.cpp")
        .file("zpaq_rs_ffi.cpp")
        // zpaq.cpp contains a `main()`. Rename it so it can be linked into this library.
        .define("main", "zpaq_cli_main")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-fvisibility=hidden")
        .flag_if_supported("-fPIC")
        .flag_if_supported("-pthread")
        .flag_if_supported("-march=native")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-null-pointer-subtraction")
        .flag_if_supported("-Wno-unused-const-variable")
        .define("unix", None)
        .define("NDEBUG", None);

    if env::var_os("CARGO_FEATURE_NOJIT").is_some() {
        build.define("NOJIT", None);
    }

    // Always optimize zpaq
    build.flag_if_supported("-O3");

    // Try to enable LTO for the C++ objects in release-like profiles.
    // Cross-language LTO (Rust <-> C++) is toolchain-dependent; this at least
    // enables LTO within the C++ compilation unit(s) when supported.
    let profile = env::var("PROFILE").unwrap_or_default();
    if profile == "release" || profile == "bench" {
        build.flag_if_supported("-flto");
    }

    build.compile("zpaq_rs_ffi");
}
