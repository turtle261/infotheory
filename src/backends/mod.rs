pub mod ctw;
pub mod rosaplus;
#[cfg(feature = "backend-rwkv")]
pub mod rwkvzip;
pub mod zpaq_rate;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendAvailability {
    Enabled(&'static str),
    Disabled {
        canonical: &'static str,
        feature: &'static str,
    },
}

#[cfg(feature = "backend-rwkv")]
pub const AVAILABLE_RATE_BACKENDS: &[&str] =
    &["rosaplus", "ctw", "fac-ctw", "rwkv7", "zpaq", "mixture"];
#[cfg(not(feature = "backend-rwkv"))]
pub const AVAILABLE_RATE_BACKENDS: &[&str] = &["rosaplus", "ctw", "fac-ctw", "zpaq", "mixture"];

#[cfg(feature = "backend-rwkv")]
pub const AVAILABLE_NCD_BACKENDS: &[&str] = &["zpaq", "rwkv7"];
#[cfg(not(feature = "backend-rwkv"))]
pub const AVAILABLE_NCD_BACKENDS: &[&str] = &["zpaq"];

pub fn resolve_rate_backend_name(input: &str) -> Option<BackendAvailability> {
    let key = input.trim().to_ascii_lowercase();
    match key.as_str() {
        "rosaplus" | "rosa" => Some(BackendAvailability::Enabled("rosaplus")),
        "ctw" => Some(BackendAvailability::Enabled("ctw")),
        "fac-ctw" | "facctw" => Some(BackendAvailability::Enabled("fac-ctw")),
        "zpaq" => Some(BackendAvailability::Enabled("zpaq")),
        "mixture" | "mix" => Some(BackendAvailability::Enabled("mixture")),
        "rwkv7" | "rwkv" => {
            if cfg!(feature = "backend-rwkv") {
                Some(BackendAvailability::Enabled("rwkv7"))
            } else {
                Some(BackendAvailability::Disabled {
                    canonical: "rwkv7",
                    feature: "backend-rwkv",
                })
            }
        }
        _ => None,
    }
}

pub fn resolve_ncd_backend_name(input: &str) -> Option<BackendAvailability> {
    let key = input.trim().to_ascii_lowercase();
    match key.as_str() {
        "zpaq" => Some(BackendAvailability::Enabled("zpaq")),
        "rwkv7" | "rwkv" => {
            if cfg!(feature = "backend-rwkv") {
                Some(BackendAvailability::Enabled("rwkv7"))
            } else {
                Some(BackendAvailability::Disabled {
                    canonical: "rwkv7",
                    feature: "backend-rwkv",
                })
            }
        }
        _ => None,
    }
}

#[cfg(feature = "backend-rwkv")]
pub fn parse_rwkv7_coder(v: &str) -> Option<rwkvzip::CoderType> {
    match v {
        "ac" | "AC" => Some(rwkvzip::CoderType::AC),
        "rans" | "RANS" | "rANS" => Some(rwkvzip::CoderType::RANS),
        _ => None,
    }
}
