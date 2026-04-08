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

/// Method-backed neural backend family shared by CLI-facing integration helpers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MethodBackendFamily {
    Mamba,
    Rwkv7,
}

/// Canonical names for enabled rate backends in this build.
pub fn available_rate_backends() -> Vec<&'static str> {
    available_backend_names(crate::runtime::RATE_BACKEND_REGISTRY)
}

/// Canonical names for enabled compression backends in this build.
pub fn available_compression_backends() -> Vec<&'static str> {
    available_backend_names(crate::runtime::COMPRESSION_BACKEND_REGISTRY)
}

fn available_backend_names<K: Copy + Eq>(
    registry: &'static [crate::runtime::BackendDescriptor<K>],
) -> Vec<&'static str> {
    registry
        .iter()
        .filter(|descriptor| descriptor.enabled)
        .map(|descriptor| descriptor.canonical)
        .collect()
}

fn resolve_backend_name_from_registry<K: Copy + Eq>(
    registry: &'static [crate::runtime::BackendDescriptor<K>],
    input: &str,
) -> Option<BackendAvailability> {
    const INTERNAL_MISMATCH_FEATURE: &str = "__internal-registry-mismatch__";

    let descriptor = crate::runtime::find_backend_descriptor_in_registry(registry, input)?;
    Some(if descriptor.enabled || descriptor.feature.is_none() {
        BackendAvailability::Enabled(descriptor.canonical)
    } else {
        BackendAvailability::Disabled {
            canonical: descriptor.canonical,
            feature: descriptor.feature.unwrap_or(INTERNAL_MISMATCH_FEATURE),
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

/// Normalize a compression backend for file roundtrip helpers.
///
/// File-oriented encode/decode APIs always use framed rate-coded payloads so the
/// backend choice roundtrips predictably across CLI and Python entrypoints.
pub fn normalize_file_roundtrip_backend(
    backend: &crate::api::CompressionBackend,
) -> crate::api::CompressionBackend {
    match backend {
        crate::api::CompressionBackend::Rate {
            rate_backend,
            coder,
            ..
        } => crate::api::CompressionBackend::Rate {
            rate_backend: rate_backend.clone(),
            coder: *coder,
            framing: crate::compression::FramingMode::Framed,
        },
        _ => backend.clone(),
    }
}

/// Extract a method string from a rate backend when it belongs to a method-backed family.
pub fn rate_backend_method_string(
    backend: &crate::api::RateBackend,
    family: MethodBackendFamily,
) -> Option<&str> {
    match (family, backend) {
        #[cfg(feature = "backend-rwkv")]
        (MethodBackendFamily::Rwkv7, crate::api::RateBackend::Rwkv7Method { method }) => {
            Some(method.as_str())
        }
        #[cfg(feature = "backend-mamba")]
        (MethodBackendFamily::Mamba, crate::api::RateBackend::MambaMethod { method }) => {
            Some(method.as_str())
        }
        _ => None,
    }
}

/// Extract a method string from a compression backend or its wrapped rate backend.
pub fn compression_backend_method_string(
    backend: &crate::api::CompressionBackend,
    family: MethodBackendFamily,
) -> Option<&str> {
    match (family, backend) {
        #[cfg(feature = "backend-rwkv")]
        (MethodBackendFamily::Rwkv7, crate::api::CompressionBackend::Rwkv7 { method, .. }) => {
            Some(method.as_str())
        }
        (_, crate::api::CompressionBackend::Rate { rate_backend, .. }) => {
            rate_backend_method_string(rate_backend, family)
        }
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
        let rosa = if cfg!(feature = "backend-rosa") {
            BackendAvailability::Enabled("rosaplus")
        } else {
            BackendAvailability::Disabled {
                canonical: "rosaplus",
                feature: "backend-rosa",
            }
        };
        let fac_ctw = if cfg!(feature = "backend-ctw") {
            BackendAvailability::Enabled("fac-ctw")
        } else {
            BackendAvailability::Disabled {
                canonical: "fac-ctw",
                feature: "backend-ctw",
            }
        };
        let mixture = if cfg!(feature = "backend-mixture") {
            BackendAvailability::Enabled("mixture")
        } else {
            BackendAvailability::Disabled {
                canonical: "mixture",
                feature: "backend-mixture",
            }
        };
        let sequitur = if cfg!(feature = "backend-sequitur") {
            BackendAvailability::Enabled("sequitur")
        } else {
            BackendAvailability::Disabled {
                canonical: "sequitur",
                feature: "backend-sequitur",
            }
        };

        assert_eq!(resolve_rate_backend_name("  Rosa  "), Some(rosa));
        assert_eq!(resolve_rate_backend_name("facctw"), Some(fac_ctw));
        assert_eq!(resolve_rate_backend_name("mix"), Some(mixture));
        assert_eq!(resolve_rate_backend_name("sequitur"), Some(sequitur));
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

    #[test]
    fn normalize_file_roundtrip_backend_forces_framed_rate_payloads() {
        let rate = crate::api::CompressionBackend::Rate {
            rate_backend: crate::api::RateBackend::Ctw { depth: 8 },
            coder: CoderType::RANS,
            framing: crate::compression::FramingMode::Raw,
        };
        let normalized = normalize_file_roundtrip_backend(&rate);
        match normalized {
            crate::api::CompressionBackend::Rate {
                rate_backend: crate::api::RateBackend::Ctw { depth },
                coder,
                framing,
            } => {
                assert_eq!(depth, 8);
                assert_eq!(coder, CoderType::RANS);
                assert_eq!(framing, crate::compression::FramingMode::Framed);
            }
            _ => panic!("expected normalized rate backend"),
        }

        let zpaq = crate::api::CompressionBackend::Zpaq {
            method: "5".to_string(),
        };
        match normalize_file_roundtrip_backend(&zpaq) {
            crate::api::CompressionBackend::Zpaq { method } => assert_eq!(method, "5"),
            _ => panic!("expected zpaq backend to remain unchanged"),
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
