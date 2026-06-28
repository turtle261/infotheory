//! Shared default-policy definitions for rate backend entrypoints.
//!
//! This module intentionally keeps distinct projections for:
//! - runtime implicit defaults,
//! - JSON parse defaults,
//! - shorthand parsing defaults.

use crate::api::RateBackend;
use crate::runtime::RateBackendKind;
use std::sync::Arc;

pub(crate) const JSON_DEFAULT_CTW_DEPTH: usize = 16;
pub(crate) const JSON_DEFAULT_FAC_CTW_BASE_DEPTH: usize = 16;
pub(crate) const JSON_DEFAULT_FAC_CTW_ENCODING_BITS: usize = 8;
pub(crate) const FAC_CTW_DEFAULT_NUM_PERCEPT_BITS: usize = 8;
pub(crate) const JSON_DEFAULT_MATCH_HASH_BITS: usize = 20;
pub(crate) const JSON_DEFAULT_MATCH_MIN_LEN: usize = 4;
pub(crate) const JSON_DEFAULT_MATCH_MAX_LEN: usize = 255;
pub(crate) const JSON_DEFAULT_MATCH_BASE_MIX: f64 = 0.02;
pub(crate) const JSON_DEFAULT_MATCH_CONFIDENCE_SCALE: f64 = 1.0;
pub(crate) const JSON_DEFAULT_SPARSE_MATCH_HASH_BITS: usize = 19;
pub(crate) const JSON_DEFAULT_SPARSE_MATCH_MIN_LEN: usize = 3;
pub(crate) const JSON_DEFAULT_SPARSE_MATCH_MAX_LEN: usize = 64;
pub(crate) const JSON_DEFAULT_SPARSE_MATCH_GAP_MIN: usize = 1;
pub(crate) const JSON_DEFAULT_SPARSE_MATCH_GAP_MAX: usize = 2;
pub(crate) const JSON_DEFAULT_SPARSE_MATCH_BASE_MIX: f64 = 0.05;
pub(crate) const JSON_DEFAULT_SPARSE_MATCH_CONFIDENCE_SCALE: f64 = 1.0;
pub(crate) const JSON_DEFAULT_PPMD_ORDER: usize = 10;
pub(crate) const JSON_DEFAULT_PPMD_MEMORY_MB: usize = 64;
pub(crate) const JSON_DEFAULT_SEQUITUR_CONTEXT_BYTES: usize = 64;
pub(crate) const JSON_DEFAULT_ZPAQ_RATE_METHOD: &str = "2";

pub(crate) const SHORTHAND_DEFAULT_CTW_DEPTH: usize = JSON_DEFAULT_CTW_DEPTH;
pub(crate) const SHORTHAND_DEFAULT_FAC_CTW_BASE_DEPTH: usize = JSON_DEFAULT_FAC_CTW_BASE_DEPTH;
pub(crate) const SHORTHAND_DEFAULT_FAC_CTW_NUM_PERCEPT_BITS: usize =
    FAC_CTW_DEFAULT_NUM_PERCEPT_BITS;
pub(crate) const SHORTHAND_DEFAULT_FAC_CTW_ENCODING_BITS: usize =
    JSON_DEFAULT_FAC_CTW_ENCODING_BITS;
pub(crate) const SHORTHAND_DEFAULT_PPMD_ORDER: usize = JSON_DEFAULT_PPMD_ORDER;
pub(crate) const SHORTHAND_DEFAULT_PPMD_MEMORY_MB: usize = JSON_DEFAULT_PPMD_MEMORY_MB;
pub(crate) const SHORTHAND_DEFAULT_SEQUITUR_CONTEXT_BYTES: usize =
    JSON_DEFAULT_SEQUITUR_CONTEXT_BYTES;
pub(crate) const SHORTHAND_DEFAULT_ZPAQ_RATE_METHOD: &str = JSON_DEFAULT_ZPAQ_RATE_METHOD;

/// Construct a [`RateBackend::FacCtw`] from explicit field values.
///
/// `msb_first: None` defers to compile-time default (`encoding_bits == 8` → MSB-first).
pub fn fac_ctw_rate_backend(
    base_depth: usize,
    num_percept_bits: usize,
    encoding_bits: usize,
    msb_first: Option<bool>,
) -> RateBackend {
    RateBackend::FacCtw {
        base_depth,
        num_percept_bits,
        encoding_bits,
        msb_first,
    }
}

/// JSON leaf object for a factorized CTW rate backend.
///
/// Omits `msb_first` when `None` so compile-time defaults apply consistently.
///
/// This is intentionally test-only; production code should construct
/// `RateBackend::FacCtw` via typed APIs and parse paths.
#[cfg(all(test, feature = "backend-ctw"))]
pub fn fac_ctw_spec_json(
    base_depth: usize,
    num_percept_bits: usize,
    encoding_bits: usize,
    msb_first: Option<bool>,
) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert(
        "kind".to_string(),
        serde_json::Value::String("fac-ctw".to_string()),
    );
    object.insert(
        "base_depth".to_string(),
        serde_json::Value::Number(base_depth.into()),
    );
    object.insert(
        "num_percept_bits".to_string(),
        serde_json::Value::Number(num_percept_bits.into()),
    );
    object.insert(
        "encoding_bits".to_string(),
        serde_json::Value::Number(encoding_bits.into()),
    );
    if let Some(msb_first) = msb_first {
        object.insert("msb_first".to_string(), serde_json::Value::Bool(msb_first));
    }
    serde_json::Value::Object(object)
}

pub(crate) fn runtime_default_rate_backend_spec(kind: RateBackendKind) -> Option<RateBackend> {
    match kind {
        RateBackendKind::RosaPlus => Some(RateBackend::RosaPlus { max_order: -1 }),
        RateBackendKind::Match => Some(RateBackend::Match {
            hash_bits: 18,
            min_len: 4,
            max_len: 96,
            base_mix: 0.02,
            confidence_scale: 1.0,
        }),
        RateBackendKind::SparseMatch => Some(RateBackend::SparseMatch {
            hash_bits: 17,
            min_len: 3,
            max_len: 48,
            gap_min: 1,
            gap_max: 2,
            base_mix: 0.05,
            confidence_scale: 1.0,
        }),
        RateBackendKind::Ppmd => Some(RateBackend::Ppmd {
            order: 6,
            memory_mb: 16,
        }),
        RateBackendKind::Sequitur => Some(RateBackend::Sequitur { context_bytes: 32 }),
        RateBackendKind::Ctw => Some(RateBackend::Ctw { depth: 8 }),
        RateBackendKind::FacCtw => Some(fac_ctw_rate_backend(
            8,
            FAC_CTW_DEFAULT_NUM_PERCEPT_BITS,
            JSON_DEFAULT_FAC_CTW_ENCODING_BITS,
            None,
        )),
        RateBackendKind::Zpaq => Some(RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("2"),
        }),
        RateBackendKind::Particle => Some(RateBackend::Particle {
            spec: Arc::new(crate::api::ParticleSpec::default()),
        }),
        RateBackendKind::Mixture | RateBackendKind::Calibrated => None,
        #[cfg(feature = "backend-mamba")]
        RateBackendKind::Mamba => Some(RateBackend::MambaMethod {
            method: crate::mambazip::MethodSpec::Online {
                cfg: crate::mambazip::OnlineConfig {
                    hidden: 64,
                    layers: 1,
                    intermediate: 96,
                    state: 16,
                    conv: 4,
                    dt_rank: 16,
                    seed: 26,
                    train_mode: crate::mambazip::OnlineTrainMode::None,
                    lr: 0.0,
                    stride: 1,
                },
                policy: Some(crate::backends::llm_policy::LlmPolicy {
                    load_from: None,
                    schedule: vec![crate::backends::llm_policy::ScheduleRule::Interval(
                        crate::backends::llm_policy::PolicyRule {
                            start: crate::backends::llm_policy::PositionExpr::Bytes(0),
                            end: crate::backends::llm_policy::PositionExpr::Bytes(100),
                            action: crate::backends::llm_policy::PolicyAction::Infer,
                        },
                    )],
                }),
            },
        }),
        #[cfg(not(feature = "backend-mamba"))]
        RateBackendKind::Mamba => None,
        #[cfg(feature = "backend-rwkv")]
        RateBackendKind::Rwkv7 => Some(RateBackend::Rwkv7Method {
            method: crate::rwkvzip::MethodSpec::Online {
                cfg: crate::rwkvzip::OnlineConfig {
                    hidden: 64,
                    layers: 1,
                    intermediate: 64,
                    decay_rank: 32,
                    a_rank: 32,
                    v_rank: 32,
                    g_rank: 64,
                    seed: 0,
                    train_mode: crate::rwkvzip::OnlineTrainMode::Sgd,
                    lr: 0.01,
                    stride: 1,
                },
                policy: Some(crate::backends::llm_policy::LlmPolicy {
                    load_from: None,
                    schedule: vec![crate::backends::llm_policy::ScheduleRule::Interval(
                        crate::backends::llm_policy::PolicyRule {
                            start: crate::backends::llm_policy::PositionExpr::Bytes(0),
                            end: crate::backends::llm_policy::PositionExpr::Bytes(100),
                            action: crate::backends::llm_policy::PolicyAction::Infer,
                        },
                    )],
                }),
            },
        }),
        #[cfg(not(feature = "backend-rwkv"))]
        RateBackendKind::Rwkv7 => None,
    }
}
