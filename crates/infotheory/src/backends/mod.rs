//! Backend discovery helpers and canonical backend naming.
//!
//! This module provides:
//! - canonical backend name resolution for CLI/Python inputs,
//! - feature-aware availability reporting,
//! - exported lists of enabled backend families.

/// Online probability calibration wrapper for rate predictors.
#[cfg(feature = "backend-calibrated")]
pub mod calibration;
#[cfg(feature = "backend-ctw")]
pub mod ctw;
/// Shared policy parser/compiler for online LLM backends.
pub mod llm_policy;
/// Mamba-1 based rate/compression backend.
#[cfg(feature = "backend-mamba")]
pub mod mambazip;
/// Contiguous/sparse local match predictor primitives.
#[cfg(feature = "backend-match")]
pub mod match_model;
/// Particle-latent rate backend.
#[cfg(feature = "backend-particle")]
pub mod particle;
/// Bounded-memory PPMD-style byte model.
#[cfg(feature = "backend-ppmd")]
pub mod ppmd;
#[cfg(feature = "backend-rosa")]
pub mod rosaplus;
/// RWKV7-based rate/compression backend.
#[cfg(feature = "backend-rwkv")]
pub mod rwkvzip;
/// Exact online Sequitur grammar backend with byte-level predictive readout.
#[cfg(feature = "backend-sequitur")]
pub mod sequitur;
/// Sparse/gapped match predictor that wraps [`match_model`].
#[cfg(feature = "backend-match")]
pub mod sparse_match;
/// Text/repeat context feature extraction for adaptive backends.
pub mod text_context;
#[cfg(feature = "backend-zpaq")]
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

/// Canonical names for enabled rate backends in this build.
pub fn available_rate_backends() -> Vec<&'static str> {
    available_backend_names(crate::runtime::RATE_BACKEND_REGISTRY)
}

/// Canonical names for enabled compression backends in this build.
pub fn available_compression_backends() -> Vec<&'static str> {
    available_backend_names(crate::runtime::COMPRESSION_BACKEND_REGISTRY)
}

fn available_backend_names(
    registry: &'static [crate::runtime::BackendDescriptor],
) -> Vec<&'static str> {
    registry
        .iter()
        .filter(|descriptor| descriptor.enabled)
        .map(|descriptor| descriptor.canonical)
        .collect()
}

fn resolve_backend_name_from_registry(
    registry: &'static [crate::runtime::BackendDescriptor],
    input: &str,
) -> Option<BackendAvailability> {
    let descriptor = crate::runtime::find_backend_descriptor_in_registry(registry, input)?;
    Some(if descriptor.enabled || descriptor.feature.is_none() {
        BackendAvailability::Enabled(descriptor.canonical)
    } else {
        BackendAvailability::Disabled {
            canonical: descriptor.canonical,
            feature: descriptor
                .feature
                .expect("disabled backends must declare required feature"),
        }
    })
}

/// Resolve a user-provided rate backend alias to a canonical backend name.
///
/// Returns `None` when the alias is unknown, and `BackendAvailability::Disabled`
/// when known but not enabled in the current feature set.
pub fn resolve_rate_backend_name(input: &str) -> Option<BackendAvailability> {
    resolve_backend_name_from_registry(crate::runtime::RATE_BACKEND_REGISTRY, input)
}

/// Resolve a user-provided compression backend alias to a canonical backend name.
///
/// Returns `None` when the alias is unknown, and `BackendAvailability::Disabled`
/// when known but not enabled in the current feature set.
pub fn resolve_compression_backend_name(input: &str) -> Option<BackendAvailability> {
    resolve_backend_name_from_registry(crate::runtime::COMPRESSION_BACKEND_REGISTRY, input)
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
        assert_eq!(
            resolve_rate_backend_name("sequitur"),
            Some(BackendAvailability::Enabled("sequitur"))
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

    #[test]
    fn available_backend_lists_track_runtime_registry() {
        assert_eq!(
            available_rate_backends(),
            crate::runtime::RATE_BACKEND_REGISTRY
                .iter()
                .filter(|descriptor| descriptor.enabled)
                .map(|descriptor| descriptor.canonical)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            available_compression_backends(),
            crate::runtime::COMPRESSION_BACKEND_REGISTRY
                .iter()
                .filter(|descriptor| descriptor.enabled)
                .map(|descriptor| descriptor.canonical)
                .collect::<Vec<_>>()
        );
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
