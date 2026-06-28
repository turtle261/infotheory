fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let trimmed: &str = value.trim();
            trimmed == "1"
                || trimmed.eq_ignore_ascii_case("true")
                || trimmed.eq_ignore_ascii_case("yes")
                || trimmed.eq_ignore_ascii_case("on")
        }
        Err(_) => false,
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=INFOTHEORY_AC_ENCODE_DEINLINE");
    println!("cargo:rerun-if-env-changed=INFOTHEORY_AC_DECODE_INLINE");

    println!("cargo:rustc-check-cfg=cfg(infotheory_ac_encode_deinline)");
    println!("cargo:rustc-check-cfg=cfg(infotheory_ac_decode_inline)");

    if env_flag_enabled("INFOTHEORY_AC_ENCODE_DEINLINE") {
        println!("cargo:rustc-cfg=infotheory_ac_encode_deinline");
    }
    if env_flag_enabled("INFOTHEORY_AC_DECODE_INLINE") {
        println!("cargo:rustc-cfg=infotheory_ac_decode_inline");
    }
}
