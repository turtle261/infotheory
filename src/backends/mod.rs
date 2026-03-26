//! Backend discovery helpers and canonical backend naming.
//!
//! This module provides:
//! - canonical backend name resolution for CLI/Python inputs,
//! - feature-aware availability reporting,
//! - exported lists of enabled backend families.

/// Online probability calibration wrapper for rate predictors.
pub mod calibration;
pub mod ctw;
/// Shared policy parser/compiler for online LLM backends.
pub mod llm_policy;
/// Mamba-1 based rate/compression backend.
#[cfg(feature = "backend-mamba")]
pub mod mambazip;
/// Contiguous/sparse local match predictor primitives.
pub mod match_model;
/// Particle-latent rate backend.
pub mod particle;
/// Bounded-memory PPMD-style byte model.
pub mod ppmd;
pub mod rosaplus;
/// RWKV7-based rate/compression backend.
#[cfg(feature = "backend-rwkv")]
pub mod rwkvzip;
/// Sparse/gapped match predictor that wraps [`match_model`].
pub mod sparse_match;
/// Text/repeat context feature extraction for adaptive backends.
pub mod text_context;
pub mod zpaq_rate;
use crate::coders::CoderType;

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

/// Canonical names for rate backends recognized by CLI/API alias resolution.
pub const AVAILABLE_RATE_BACKENDS: &[&str] = &[
    "rosaplus",
    "ctw",
    "fac-ctw",
    "match",
    "sparse-match",
    "ppmd",
    "calibrated",
    #[cfg(feature = "backend-mamba")]
    "mamba",
    #[cfg(feature = "backend-rwkv")]
    "rwkv7",
    #[cfg(feature = "backend-zpaq")]
    "zpaq",
    "mixture",
    "particle",
];

/// Canonical names for available compression backends in this build.
pub const AVAILABLE_COMPRESSION_BACKENDS: &[&str] = &[
    #[cfg(feature = "backend-zpaq")]
    "zpaq",
    "rate-ac",
    "rate-rans",
    #[cfg(feature = "backend-rwkv")]
    "rwkv7",
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
        "match" => Some(BackendAvailability::Enabled("match")),
        "sparse-match" | "sparse_match" | "sparsematch" => {
            Some(BackendAvailability::Enabled("sparse-match"))
        }
        "ppmd" | "ppm" => Some(BackendAvailability::Enabled("ppmd")),
        "calibrated" | "cal" => Some(BackendAvailability::Enabled("calibrated")),
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
        "mamba" | "mamba1" => {
            if cfg!(feature = "backend-mamba") {
                Some(BackendAvailability::Enabled("mamba"))
            } else {
                Some(BackendAvailability::Disabled {
                    canonical: "mamba",
                    feature: "backend-mamba",
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
        "rate-ac" | "rate_ac" | "rateac" => Some(BackendAvailability::Enabled("rate-ac")),
        "rate-rans" | "rate_rans" | "raterans" => Some(BackendAvailability::Enabled("rate-rans")),
        _ => None,
    }
}

/// Parse a generic entropy coder alias (`"ac"`/`"rans"`).
pub fn parse_rate_coder(v: &str) -> Option<CoderType> {
    match v {
        "ac" | "AC" => Some(CoderType::AC),
        "rans" | "RANS" | "rANS" => Some(CoderType::RANS),
        _ => None,
    }
}

/// Parse an RWKV entropy coder alias (`"ac"`/`"rans"`).
#[cfg(feature = "backend-rwkv")]
pub fn parse_rwkv7_coder(v: &str) -> Option<CoderType> {
    parse_rate_coder(v)
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

        if cfg!(feature = "backend-mamba") {
            assert_eq!(
                resolve_rate_backend_name("mamba1"),
                Some(BackendAvailability::Enabled("mamba"))
            );
        } else {
            assert_eq!(
                resolve_rate_backend_name("mamba"),
                Some(BackendAvailability::Disabled {
                    canonical: "mamba",
                    feature: "backend-mamba",
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

        assert_eq!(
            resolve_compression_backend_name("rate_ac"),
            Some(BackendAvailability::Enabled("rate-ac"))
        );
        assert_eq!(
            resolve_compression_backend_name("raterans"),
            Some(BackendAvailability::Enabled("rate-rans"))
        );

        if cfg!(feature = "backend-rwkv") {
            assert_eq!(
                resolve_compression_backend_name("rwkv"),
                Some(BackendAvailability::Enabled("rwkv7"))
            );
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_rwkv7_coder_accepts_common_aliases() {
        assert_eq!(parse_rwkv7_coder("ac"), Some(CoderType::AC));
        assert_eq!(parse_rwkv7_coder("AC"), Some(CoderType::AC));
        assert_eq!(parse_rwkv7_coder("rans"), Some(CoderType::RANS));
        assert_eq!(parse_rwkv7_coder("RANS"), Some(CoderType::RANS));
        assert_eq!(parse_rwkv7_coder("nope"), None);
    }
}
