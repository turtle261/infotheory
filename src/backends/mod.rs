//! Backend discovery helpers and canonical backend naming.
//!
//! This module provides:
//! - canonical backend name resolution for CLI/Python inputs,
//! - feature-aware availability reporting,
//! - exported lists of enabled backend families.

pub mod ctw;
pub mod rosaplus;
/// RWKV7-based rate/compression backend.
#[cfg(feature = "backend-rwkv")]
pub mod rwkvzip;
pub mod zpaq_rate;

/// Outcome of resolving a backend alias to a canonical backend name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendAvailability {
    /// Backend is compiled and available.
    Enabled(&'static str),
    /// Backend alias is recognized, but the required Cargo feature is disabled.
    Disabled {
        /// Canonical backend name.
        canonical: &'static str,
        /// Cargo feature needed to enable this backend.
        feature: &'static str,
    },
}

/// Canonical names for available rate backends in this build.
#[cfg(feature = "backend-rwkv")]
pub const AVAILABLE_RATE_BACKENDS: &[&str] = &[
    "rosaplus",
    "ctw",
    "fac-ctw",
    "rwkv7",
    #[cfg(feature = "backend-zpaq")]
    "zpaq",
    "mixture",
    "particle",
];
/// Canonical names for available rate backends in this build.
#[cfg(not(feature = "backend-rwkv"))]
pub const AVAILABLE_RATE_BACKENDS: &[&str] = &[
    "rosaplus",
    "ctw",
    "fac-ctw",
    #[cfg(feature = "backend-zpaq")]
    "zpaq",
    "mixture",
    "particle",
];

/// Canonical names for available compression backends in this build.
#[cfg(feature = "backend-rwkv")]
pub const AVAILABLE_COMPRESSION_BACKENDS: &[&str] = &[
    #[cfg(feature = "backend-zpaq")]
    "zpaq",
    "rwkv7",
    "rate-ac",
    "rate-rans",
];
/// Canonical names for available compression backends in this build.
#[cfg(not(feature = "backend-rwkv"))]
pub const AVAILABLE_COMPRESSION_BACKENDS: &[&str] = &[
    #[cfg(feature = "backend-zpaq")]
    "zpaq",
];

/// Resolve a user-provided rate backend alias to a canonical backend name.
///
/// Returns `None` when the alias is unknown, and `BackendAvailability::Disabled`
/// when known but not enabled in the current feature set.
pub fn resolve_rate_backend_name(input: &str) -> Option<BackendAvailability> {
    let key = input.trim().to_ascii_lowercase();
    match key.as_str() {
        "rosaplus" | "rosa" => Some(BackendAvailability::Enabled("rosaplus")),
        "ctw" => Some(BackendAvailability::Enabled("ctw")),
        "fac-ctw" | "facctw" => Some(BackendAvailability::Enabled("fac-ctw")),
        "zpaq" => {
            if cfg!(feature = "backend-zpaq") {
                Some(BackendAvailability::Enabled("zpaq"))
            } else {
                Some(BackendAvailability::Disabled {
                    canonical: "zpaq",
                    feature: "backend-zpaq",
                })
            }
        }
        "mixture" | "mix" => Some(BackendAvailability::Enabled("mixture")),
        "particle" | "particles" => Some(BackendAvailability::Enabled("particle")),
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

/// Resolve a user-provided compression backend alias to a canonical backend name.
///
/// Returns `None` when the alias is unknown, and `BackendAvailability::Disabled`
/// when known but not enabled in the current feature set.
pub fn resolve_compression_backend_name(input: &str) -> Option<BackendAvailability> {
    let key = input.trim().to_ascii_lowercase();
    match key.as_str() {
        "zpaq" => {
            if cfg!(feature = "backend-zpaq") {
                Some(BackendAvailability::Enabled("zpaq"))
            } else {
                Some(BackendAvailability::Disabled {
                    canonical: "zpaq",
                    feature: "backend-zpaq",
                })
            }
        }
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
        "rate-ac" | "rate_ac" | "rateac" => {
            if cfg!(feature = "backend-rwkv") {
                Some(BackendAvailability::Enabled("rate-ac"))
            } else {
                Some(BackendAvailability::Disabled {
                    canonical: "rate-ac",
                    feature: "backend-rwkv",
                })
            }
        }
        "rate-rans" | "rate_rans" | "raterans" => {
            if cfg!(feature = "backend-rwkv") {
                Some(BackendAvailability::Enabled("rate-rans"))
            } else {
                Some(BackendAvailability::Disabled {
                    canonical: "rate-rans",
                    feature: "backend-rwkv",
                })
            }
        }
        _ => None,
    }
}

/// Parse an RWKV entropy coder alias (`"ac"`/`"rans"`).
#[cfg(feature = "backend-rwkv")]
pub fn parse_rwkv7_coder(v: &str) -> Option<rwkvzip::CoderType> {
    match v {
        "ac" | "AC" => Some(rwkvzip::CoderType::AC),
        "rans" | "RANS" | "rANS" => Some(rwkvzip::CoderType::RANS),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_rate_backend_name_canonicalizes_aliases() {
        assert_eq!(
            resolve_rate_backend_name("  Rosa  "),
            Some(BackendAvailability::Enabled("rosaplus"))
        );
        assert_eq!(
            resolve_rate_backend_name("facctw"),
            Some(BackendAvailability::Enabled("fac-ctw"))
        );
        assert_eq!(
            resolve_rate_backend_name("mix"),
            Some(BackendAvailability::Enabled("mixture"))
        );
        assert_eq!(resolve_rate_backend_name("unknown"), None);
    }

    #[test]
    fn resolve_rate_backend_name_reports_feature_disabled() {
        if cfg!(feature = "backend-zpaq") {
            assert_eq!(
                resolve_rate_backend_name("zpaq"),
                Some(BackendAvailability::Enabled("zpaq"))
            );
        } else {
            assert_eq!(
                resolve_rate_backend_name("zpaq"),
                Some(BackendAvailability::Disabled {
                    canonical: "zpaq",
                    feature: "backend-zpaq",
                })
            );
        }

        if cfg!(feature = "backend-rwkv") {
            assert_eq!(
                resolve_rate_backend_name("rwkv7"),
                Some(BackendAvailability::Enabled("rwkv7"))
            );
        } else {
            assert_eq!(
                resolve_rate_backend_name("rwkv"),
                Some(BackendAvailability::Disabled {
                    canonical: "rwkv7",
                    feature: "backend-rwkv",
                })
            );
        }
    }

    #[test]
    fn resolve_compression_backend_name_canonicalizes_aliases() {
        assert_eq!(resolve_compression_backend_name("unknown"), None);

        if cfg!(feature = "backend-zpaq") {
            assert_eq!(
                resolve_compression_backend_name("zpaq"),
                Some(BackendAvailability::Enabled("zpaq"))
            );
        } else {
            assert_eq!(
                resolve_compression_backend_name("zpaq"),
                Some(BackendAvailability::Disabled {
                    canonical: "zpaq",
                    feature: "backend-zpaq",
                })
            );
        }

        if cfg!(feature = "backend-rwkv") {
            assert_eq!(
                resolve_compression_backend_name("rate_ac"),
                Some(BackendAvailability::Enabled("rate-ac"))
            );
            assert_eq!(
                resolve_compression_backend_name("raterans"),
                Some(BackendAvailability::Enabled("rate-rans"))
            );
            assert_eq!(
                resolve_compression_backend_name("rwkv"),
                Some(BackendAvailability::Enabled("rwkv7"))
            );
        } else {
            assert_eq!(
                resolve_compression_backend_name("rate-ac"),
                Some(BackendAvailability::Disabled {
                    canonical: "rate-ac",
                    feature: "backend-rwkv",
                })
            );
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_rwkv7_coder_accepts_common_aliases() {
        assert_eq!(parse_rwkv7_coder("ac"), Some(rwkvzip::CoderType::AC));
        assert_eq!(parse_rwkv7_coder("AC"), Some(rwkvzip::CoderType::AC));
        assert_eq!(parse_rwkv7_coder("rans"), Some(rwkvzip::CoderType::RANS));
        assert_eq!(parse_rwkv7_coder("RANS"), Some(rwkvzip::CoderType::RANS));
        assert_eq!(parse_rwkv7_coder("nope"), None);
    }
}
