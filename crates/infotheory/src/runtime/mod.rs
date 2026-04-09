//! Internal runtime builders and backend registry metadata.
//!
//! This module is the spec -> runtime boundary for predictor and compression
//! execution paths.

use crate::api::{CompressionBackend, RateBackend};
#[cfg(feature = "backend-calibrated")]
use crate::backends::calibration::CalibratorCore;
#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{ContextTree, FacContextTree};
#[cfg(feature = "backend-match")]
use crate::backends::match_model::MatchModel;
#[cfg(feature = "backend-particle")]
use crate::backends::particle::ParticleRuntime;
#[cfg(feature = "backend-ppmd")]
use crate::backends::ppmd::PpmdModel;
#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::RosaPlus;
#[cfg(feature = "backend-sequitur")]
use crate::backends::sequitur::SequiturModel;
#[cfg(feature = "backend-match")]
use crate::backends::sparse_match::SparseMatchModel;
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
use crate::error::{InfotheoryError, InfotheoryResult};
#[cfg(feature = "backend-mamba")]
use crate::mambazip;
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip;
use crate::spec::core::{
    CompressionBackendCapabilities, CompressionBackendPlan, MethodBackendFamily,
    RateBackendCapabilities, RateBackendPlan, RateBackendTraceStrategy as PublicTraceStrategy,
    SpecEnvironment,
};
use crate::spec::{CompiledCompressionBackend, CompiledRateBackend, SpecResult};
use std::sync::Arc;

/// Stable internal identity for each rate-backend family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RateBackendKind {
    RosaPlus,
    Ctw,
    FacCtw,
    Match,
    SparseMatch,
    Ppmd,
    Sequitur,
    Calibrated,
    Zpaq,
    Mixture,
    Particle,
    Mamba,
    Rwkv7,
}

/// Stable internal identity for each compression-backend family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CompressionBackendKind {
    Zpaq,
    Rwkv7,
    RateAc,
    RateRans,
}

/// Stable identity for method-backed neural families shared by VM glue.
#[cfg(feature = "vm")]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MethodBackendKind {
    Mamba,
    Rwkv7,
}

/// Shared trace-model execution strategy used by VM glue.
#[cfg(feature = "vm")]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TraceModelStrategy {
    Rosa,
    Ctw,
    FacCtw,
    PredictorBacked,
    Zpaq,
    Mamba,
    Rwkv7,
}

/// Canonical backend metadata entry shared by parsers and runtime builders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendDescriptor<K> {
    /// Stable internal backend identity.
    pub kind: K,
    /// Canonical backend name.
    pub canonical: &'static str,
    /// Accepted aliases for the backend, including the canonical spelling.
    pub aliases: &'static [&'static str],
    /// Required Cargo feature for availability, when the backend is feature-gated.
    pub feature: Option<&'static str>,
    /// Whether the backend is enabled in the current build.
    pub enabled: bool,
}

/// Canonical metadata for rate backends.
pub type RateBackendDescriptor = BackendDescriptor<RateBackendKind>;
/// Canonical metadata for compression backends.
pub type CompressionBackendDescriptor = BackendDescriptor<CompressionBackendKind>;

macro_rules! backend_feature_option {
    (none) => {
        None
    };
    ($feature:literal) => {
        Some($feature)
    };
}

macro_rules! backend_feature_enabled {
    (none) => {
        true
    };
    ($feature:literal) => {
        cfg!(feature = $feature)
    };
}

/// Shared declarative catalog for rate-backend metadata and runtime wiring.
///
/// Adding a new rate backend should require:
/// 1. the backend-specific implementation code,
/// 2. one entry in this catalog.
///
/// The catalog emits both the user-facing registry metadata and the runtime
/// kernel dispatch table so the family list cannot silently drift.
macro_rules! define_rate_backend_catalog {
    ($(
        backend {
            kind: $kind:ident,
            canonical: $canonical:literal,
            aliases: [$($alias:literal),* $(,)?],
            feature: $feature:tt,
            compile_plan: $compile_plan:path,
            to_wrapper: $to_wrapper:path,
            encode_payload: $encode_payload:path,
            display_label: $display_label:path,
            default_name: $default_name:path,
            trace_strategy: $trace_strategy:expr,
            supports_biased_entropy: $supports_biased_entropy:expr,
            supports_frozen_conditioning: $supports_frozen_conditioning:expr,
            supports_rate_coded_compression: $supports_rate_coded_compression:expr,
            method_family: $method_family:expr,
            contains_zpaq: $contains_zpaq:path,
            supports_bit_token_adaptation: $supports_bit_token_adaptation:path,
            adapt_for_bit_tokens: $adapt_for_bit_tokens:path,
            build_predictor: $build_predictor:path,
            build_pdf_predictor: $build_pdf_predictor:path,
            entropy_rate: $entropy_rate:path,
            joint_entropy_rate: $joint_entropy_rate:path,
            conditional_chain_rate: $conditional_chain_rate:path,
        }
    ),* $(,)?) => {
        /// Registry of rate-backend metadata.
        pub const RATE_BACKEND_REGISTRY: &[RateBackendDescriptor] = &[
            $(
                BackendDescriptor {
                    kind: RateBackendKind::$kind,
                    canonical: $canonical,
                    aliases: &[$($alias),*],
                    feature: backend_feature_option!($feature),
                    enabled: backend_feature_enabled!($feature),
                },
            )*
        ];

        pub(crate) const RATE_BACKEND_KERNELS: &[RateBackendKernel] = &[
            $(
                RateBackendKernel {
                    kind: RateBackendKind::$kind,
                    compile_plan: $compile_plan,
                    to_wrapper: $to_wrapper,
                    encode_payload: $encode_payload,
                    display_label: $display_label,
                    default_name: $default_name,
                    trace_strategy: $trace_strategy,
                    supports_biased_entropy: $supports_biased_entropy,
                    supports_frozen_conditioning: $supports_frozen_conditioning,
                    supports_rate_coded_compression: $supports_rate_coded_compression,
                    method_family: $method_family,
                    contains_zpaq: $contains_zpaq,
                    supports_bit_token_adaptation: $supports_bit_token_adaptation,
                    adapt_for_bit_tokens: $adapt_for_bit_tokens,
                    build_predictor: $build_predictor,
                    build_pdf_predictor: $build_pdf_predictor,
                    entropy_rate: $entropy_rate,
                    joint_entropy_rate: $joint_entropy_rate,
                    conditional_chain_rate: $conditional_chain_rate,
                },
            )*
        ];
    };
}

/// Shared declarative catalog for compression-backend metadata and runtime wiring.
macro_rules! define_compression_backend_catalog {
    ($(
        backend {
            kind: $kind:ident,
            canonical: $canonical:literal,
            aliases: [$($alias:literal),* $(,)?],
            feature: $feature:tt,
            compile_plan: $compile_plan:path,
            to_wrapper: $to_wrapper:path,
            encode_payload: $encode_payload:path,
            display_label: $display_label:path,
            uses_rate_backend: $uses_rate_backend:expr,
            supports_decompression: $supports_decompression:expr,
            build_runtime: $build_runtime:path,
        }
    ),* $(,)?) => {
        /// Registry of compression-backend metadata.
        pub const COMPRESSION_BACKEND_REGISTRY: &[CompressionBackendDescriptor] = &[
            $(
                BackendDescriptor {
                    kind: CompressionBackendKind::$kind,
                    canonical: $canonical,
                    aliases: &[$($alias),*],
                    feature: backend_feature_option!($feature),
                    enabled: backend_feature_enabled!($feature),
                },
            )*
        ];

        pub(crate) const COMPRESSION_BACKEND_KERNELS: &[CompressionBackendKernel] = &[
            $(
                CompressionBackendKernel {
                    kind: CompressionBackendKind::$kind,
                    compile_plan: $compile_plan,
                    to_wrapper: $to_wrapper,
                    encode_payload: $encode_payload,
                    display_label: $display_label,
                    uses_rate_backend: $uses_rate_backend,
                    supports_decompression: $supports_decompression,
                    build_runtime: $build_runtime,
                },
            )*
        ];
    };
}

type RateBackendPredictorBuilder =
    fn(&CompiledRateBackend, i64, f64) -> Result<crate::mixture::RateBackendPredictor, String>;
type RatePdfPredictorBuilder =
    fn(&CompiledRateBackend, i64) -> anyhow::Result<crate::compression::RatePdfPredictor>;
type RateEntropyFn = fn(&[u8], i64, &CompiledRateBackend) -> InfotheoryResult<f64>;
type RateJointEntropyFn = fn(&[u8], &[u8], i64, &CompiledRateBackend) -> InfotheoryResult<f64>;
type RateConditionalChainFn = fn(&[&[u8]], &[u8], &CompiledRateBackend) -> InfotheoryResult<f64>;
type RatePlanCompiler = fn(&RateBackend, &SpecEnvironment, usize) -> SpecResult<RateBackendPlan>;
type RateWrapperBuilder = fn(&RateBackendPlan) -> RateBackend;
type RatePayloadEncoder = fn(&RateBackendPlan, &mut Vec<u8>);
type RateDisplayLabelFn = fn(&RateBackendPlan, i64) -> String;
type RateDefaultNameFn = fn(&RateBackendPlan, i64) -> String;
type RateContainsZpaqFn = fn(&RateBackendPlan) -> bool;
type RateSupportsBitTokenAdaptationFn = fn(&RateBackendPlan) -> bool;
type RateBitTokenAdapter = fn(&RateBackendPlan) -> RateBackendPlan;
type CompressionRuntimeBuilder =
    fn(&CompiledCompressionBackend) -> Result<CompressionRuntimeHandle, String>;
type CompressionPlanCompiler =
    fn(&CompressionBackend, &SpecEnvironment) -> SpecResult<CompressionBackendPlan>;
type CompressionWrapperBuilder = fn(&CompressionBackendPlan) -> CompressionBackend;
type CompressionPayloadEncoder = fn(&CompressionBackendPlan, &mut Vec<u8>);
type CompressionDisplayLabelFn = fn(&CompressionBackendPlan) -> String;

#[derive(Clone, Copy)]
pub(crate) struct RateBackendKernel {
    pub kind: RateBackendKind,
    pub compile_plan: RatePlanCompiler,
    pub to_wrapper: RateWrapperBuilder,
    pub encode_payload: RatePayloadEncoder,
    pub display_label: RateDisplayLabelFn,
    pub default_name: RateDefaultNameFn,
    pub trace_strategy: PublicTraceStrategy,
    pub supports_biased_entropy: bool,
    pub supports_frozen_conditioning: bool,
    pub supports_rate_coded_compression: bool,
    pub method_family: Option<MethodBackendFamily>,
    pub contains_zpaq: RateContainsZpaqFn,
    pub supports_bit_token_adaptation: RateSupportsBitTokenAdaptationFn,
    pub adapt_for_bit_tokens: RateBitTokenAdapter,
    pub build_predictor: RateBackendPredictorBuilder,
    pub build_pdf_predictor: RatePdfPredictorBuilder,
    pub entropy_rate: RateEntropyFn,
    pub joint_entropy_rate: RateJointEntropyFn,
    pub conditional_chain_rate: RateConditionalChainFn,
}

#[derive(Clone, Copy)]
pub(crate) struct CompressionBackendKernel {
    pub kind: CompressionBackendKind,
    pub compile_plan: CompressionPlanCompiler,
    pub to_wrapper: CompressionWrapperBuilder,
    pub encode_payload: CompressionPayloadEncoder,
    pub display_label: CompressionDisplayLabelFn,
    pub uses_rate_backend: bool,
    pub supports_decompression: bool,
    pub build_runtime: CompressionRuntimeBuilder,
}

define_rate_backend_catalog! {
    backend {
        kind: RosaPlus,
        canonical: "rosaplus",
        aliases: ["rosaplus", "rosa"],
        feature: "backend-rosa",
        compile_plan: crate::spec::core::compile_rate_plan_rosa,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_rosa,
        encode_payload: crate::spec::core::encode_rate_payload_rosa,
        display_label: crate::spec::core::rate_plan_display_label_rosa,
        default_name: crate::spec::core::rate_plan_default_name_rosa,
        trace_strategy: PublicTraceStrategy::Rosa,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_rosa,
        build_pdf_predictor: build_pdf_predictor_rosa,
        entropy_rate: entropy_rosa,
        joint_entropy_rate: joint_entropy_rosa,
        conditional_chain_rate: conditional_chain_rosa,
    },
    backend {
        kind: Match,
        canonical: "match",
        aliases: ["match"],
        feature: "backend-match",
        compile_plan: crate::spec::core::compile_rate_plan_match,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_match,
        encode_payload: crate::spec::core::encode_rate_payload_match,
        display_label: crate::spec::core::rate_plan_display_label_match,
        default_name: crate::spec::core::rate_plan_default_name_match,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_match,
        build_pdf_predictor: build_pdf_predictor_match,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: SparseMatch,
        canonical: "sparse-match",
        aliases: ["sparse-match", "sparse_match", "sparsematch"],
        feature: "backend-match",
        compile_plan: crate::spec::core::compile_rate_plan_sparse_match,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_sparse_match,
        encode_payload: crate::spec::core::encode_rate_payload_sparse_match,
        display_label: crate::spec::core::rate_plan_display_label_sparse_match,
        default_name: crate::spec::core::rate_plan_default_name_sparse_match,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_sparse_match,
        build_pdf_predictor: build_pdf_predictor_sparse_match,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Ppmd,
        canonical: "ppmd",
        aliases: ["ppmd", "ppm"],
        feature: "backend-ppmd",
        compile_plan: crate::spec::core::compile_rate_plan_ppmd,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_ppmd,
        encode_payload: crate::spec::core::encode_rate_payload_ppmd,
        display_label: crate::spec::core::rate_plan_display_label_ppmd,
        default_name: crate::spec::core::rate_plan_default_name_ppmd,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_ppmd,
        build_pdf_predictor: build_pdf_predictor_ppmd,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Sequitur,
        canonical: "sequitur",
        aliases: ["sequitur"],
        feature: "backend-sequitur",
        compile_plan: crate::spec::core::compile_rate_plan_sequitur,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_sequitur,
        encode_payload: crate::spec::core::encode_rate_payload_sequitur,
        display_label: crate::spec::core::rate_plan_display_label_sequitur,
        default_name: crate::spec::core::rate_plan_default_name_sequitur,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_sequitur,
        build_pdf_predictor: build_pdf_predictor_sequitur,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Ctw,
        canonical: "ctw",
        aliases: ["ctw"],
        feature: "backend-ctw",
        compile_plan: crate::spec::core::compile_rate_plan_ctw,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_ctw,
        encode_payload: crate::spec::core::encode_rate_payload_ctw,
        display_label: crate::spec::core::rate_plan_display_label_ctw,
        default_name: crate::spec::core::rate_plan_default_name_ctw,
        trace_strategy: PublicTraceStrategy::Ctw,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_ctw,
        build_predictor: build_predictor_ctw,
        build_pdf_predictor: build_pdf_predictor_ctw,
        entropy_rate: entropy_ctw,
        joint_entropy_rate: joint_entropy_ctw,
        conditional_chain_rate: conditional_chain_ctw,
    },
    backend {
        kind: FacCtw,
        canonical: "fac-ctw",
        aliases: ["fac-ctw", "facctw"],
        feature: "backend-ctw",
        compile_plan: crate::spec::core::compile_rate_plan_fac_ctw,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_fac_ctw,
        encode_payload: crate::spec::core::encode_rate_payload_fac_ctw,
        display_label: crate::spec::core::rate_plan_display_label_fac_ctw,
        default_name: crate::spec::core::rate_plan_default_name_fac_ctw,
        trace_strategy: PublicTraceStrategy::FacCtw,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_fac_ctw,
        build_predictor: build_predictor_fac_ctw,
        build_pdf_predictor: build_pdf_predictor_fac_ctw,
        entropy_rate: entropy_fac_ctw,
        joint_entropy_rate: joint_entropy_fac_ctw,
        conditional_chain_rate: conditional_chain_fac_ctw,
    },
    backend {
        kind: Zpaq,
        canonical: "zpaq",
        aliases: ["zpaq"],
        feature: "backend-zpaq",
        compile_plan: crate::spec::core::compile_rate_plan_zpaq,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_zpaq,
        encode_payload: crate::spec::core::encode_rate_payload_zpaq,
        display_label: crate::spec::core::rate_plan_display_label_zpaq,
        default_name: crate::spec::core::rate_plan_default_name_zpaq,
        trace_strategy: PublicTraceStrategy::Zpaq,
        supports_biased_entropy: false,
        supports_frozen_conditioning: false,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_true,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_false,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_zpaq,
        build_pdf_predictor: build_pdf_predictor_zpaq,
        entropy_rate: entropy_zpaq,
        joint_entropy_rate: joint_entropy_zpaq,
        conditional_chain_rate: conditional_chain_zpaq,
    },
    backend {
        kind: Mixture,
        canonical: "mixture",
        aliases: ["mixture", "mix"],
        feature: "backend-mixture",
        compile_plan: crate::spec::core::compile_rate_plan_mixture,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_mixture,
        encode_payload: crate::spec::core::encode_rate_payload_mixture,
        display_label: crate::spec::core::rate_plan_display_label_mixture,
        default_name: crate::spec::core::rate_plan_default_name_mixture,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_mixture,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_mixture,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_mixture,
        build_predictor: build_predictor_mixture,
        build_pdf_predictor: build_pdf_predictor_mixture,
        entropy_rate: entropy_mixture,
        joint_entropy_rate: joint_entropy_mixture,
        conditional_chain_rate: conditional_chain_mixture,
    },
    backend {
        kind: Particle,
        canonical: "particle",
        aliases: ["particle", "particles"],
        feature: "backend-particle",
        compile_plan: crate::spec::core::compile_rate_plan_particle,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_particle,
        encode_payload: crate::spec::core::encode_rate_payload_particle,
        display_label: crate::spec::core::rate_plan_display_label_particle,
        default_name: crate::spec::core::rate_plan_default_name_particle,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_particle,
        build_pdf_predictor: build_pdf_predictor_particle,
        entropy_rate: entropy_particle,
        joint_entropy_rate: joint_entropy_particle,
        conditional_chain_rate: conditional_chain_particle,
    },
    backend {
        kind: Calibrated,
        canonical: "calibrated",
        aliases: ["calibrated", "cal"],
        feature: "backend-calibrated",
        compile_plan: crate::spec::core::compile_rate_plan_calibrated,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_calibrated,
        encode_payload: crate::spec::core::encode_rate_payload_calibrated,
        display_label: crate::spec::core::rate_plan_display_label_calibrated,
        default_name: crate::spec::core::rate_plan_default_name_calibrated,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_calibrated,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_calibrated,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_calibrated,
        build_predictor: build_predictor_calibrated,
        build_pdf_predictor: build_pdf_predictor_calibrated,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Mamba,
        canonical: "mamba",
        aliases: ["mamba", "mamba1"],
        feature: "backend-mamba",
        compile_plan: crate::spec::core::compile_rate_plan_mamba,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_mamba,
        encode_payload: crate::spec::core::encode_rate_payload_mamba,
        display_label: crate::spec::core::rate_plan_display_label_mamba,
        default_name: crate::spec::core::rate_plan_default_name_mamba,
        trace_strategy: PublicTraceStrategy::Mamba,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: Some(MethodBackendFamily::Mamba),
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_mamba,
        build_pdf_predictor: build_pdf_predictor_mamba,
        entropy_rate: entropy_mamba,
        joint_entropy_rate: joint_entropy_mamba,
        conditional_chain_rate: conditional_chain_mamba,
    },
    backend {
        kind: Rwkv7,
        canonical: "rwkv7",
        aliases: ["rwkv7", "rwkv"],
        feature: "backend-rwkv",
        compile_plan: crate::spec::core::compile_rate_plan_rwkv7,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_rwkv7,
        encode_payload: crate::spec::core::encode_rate_payload_rwkv7,
        display_label: crate::spec::core::rate_plan_display_label_rwkv7,
        default_name: crate::spec::core::rate_plan_default_name_rwkv7,
        trace_strategy: PublicTraceStrategy::Rwkv7,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        method_family: Some(MethodBackendFamily::Rwkv7),
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        supports_bit_token_adaptation: crate::spec::core::rate_plan_supports_bit_token_adaptation_true,
        adapt_for_bit_tokens: crate::spec::core::adapt_rate_plan_identity,
        build_predictor: build_predictor_rwkv,
        build_pdf_predictor: build_pdf_predictor_rwkv,
        entropy_rate: entropy_rwkv,
        joint_entropy_rate: joint_entropy_rwkv,
        conditional_chain_rate: conditional_chain_rwkv,
    },
}

define_compression_backend_catalog! {
    backend {
        kind: Zpaq,
        canonical: "zpaq",
        aliases: ["zpaq"],
        feature: "backend-zpaq",
        compile_plan: crate::spec::core::compile_compression_plan_zpaq,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_zpaq,
        encode_payload: crate::spec::core::encode_compression_payload_zpaq,
        display_label: crate::spec::core::compression_plan_display_label_zpaq,
        uses_rate_backend: false,
        supports_decompression: true,
        build_runtime: build_compression_runtime_zpaq,
    },
    backend {
        kind: Rwkv7,
        canonical: "rwkv7",
        aliases: ["rwkv7", "rwkv"],
        feature: "backend-rwkv",
        compile_plan: crate::spec::core::compile_compression_plan_rwkv7,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_rwkv7,
        encode_payload: crate::spec::core::encode_compression_payload_rwkv7,
        display_label: crate::spec::core::compression_plan_display_label_rwkv7,
        uses_rate_backend: false,
        supports_decompression: true,
        build_runtime: build_compression_runtime_rwkv,
    },
    backend {
        kind: RateAc,
        canonical: "rate-ac",
        aliases: ["rate-ac", "rate_ac", "rateac"],
        feature: none,
        compile_plan: crate::spec::core::compile_compression_plan_rate,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_rate,
        encode_payload: crate::spec::core::encode_compression_payload_rate,
        display_label: crate::spec::core::compression_plan_display_label_rate,
        uses_rate_backend: true,
        supports_decompression: true,
        build_runtime: build_compression_runtime_rate,
    },
    backend {
        kind: RateRans,
        canonical: "rate-rans",
        aliases: ["rate-rans", "rate_rans", "raterans"],
        feature: none,
        compile_plan: crate::spec::core::compile_compression_plan_rate,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_rate,
        encode_payload: crate::spec::core::encode_compression_payload_rate,
        display_label: crate::spec::core::compression_plan_display_label_rate,
        uses_rate_backend: true,
        supports_decompression: true,
        build_runtime: build_compression_runtime_rate,
    },
}

pub(crate) fn find_backend_descriptor_in_registry<K: Copy + Eq>(
    registry: &'static [BackendDescriptor<K>],
    input: &str,
) -> Option<&'static BackendDescriptor<K>> {
    let key = input.trim().to_ascii_lowercase();
    registry
        .iter()
        .find(|descriptor| descriptor.aliases.iter().any(|alias| *alias == key))
}

fn backend_descriptor_by_kind<K: Copy + Eq>(
    registry: &'static [BackendDescriptor<K>],
    kind: K,
) -> Option<&'static BackendDescriptor<K>> {
    registry.iter().find(|descriptor| descriptor.kind == kind)
}

fn backend_descriptor_by_kind_checked<K: Copy + Eq>(
    registry: &'static [BackendDescriptor<K>],
    kind: K,
    registry_name: &'static str,
) -> Result<&'static BackendDescriptor<K>, String>
where
    K: std::fmt::Debug,
{
    backend_descriptor_by_kind(registry, kind).ok_or_else(|| {
        format!(
            "internal backend registry mismatch: backend '{kind:?}' is missing from {registry_name}"
        )
    })
}

pub(crate) fn describe_rate_backend_kind(
    kind: RateBackendKind,
) -> Result<&'static RateBackendDescriptor, String> {
    backend_descriptor_by_kind_checked(RATE_BACKEND_REGISTRY, kind, "RATE_BACKEND_REGISTRY")
}

pub(crate) fn describe_compression_backend_kind(
    kind: CompressionBackendKind,
) -> Result<&'static CompressionBackendDescriptor, String> {
    backend_descriptor_by_kind_checked(
        COMPRESSION_BACKEND_REGISTRY,
        kind,
        "COMPRESSION_BACKEND_REGISTRY",
    )
}

#[allow(dead_code)]
fn rate_backend_feature_error(kind: RateBackendKind) -> String {
    describe_rate_backend_kind(kind)
        .map(|descriptor| match descriptor.feature {
            Some(feature) => format!(
                "backend '{}' requires infotheory feature '{}'",
                descriptor.canonical, feature
            ),
            None => format!("backend '{}' is unavailable", descriptor.canonical),
        })
        .unwrap_or_else(|err| err)
}

#[allow(dead_code)]
fn compression_backend_feature_error(kind: CompressionBackendKind) -> String {
    describe_compression_backend_kind(kind)
        .map(|descriptor| match descriptor.feature {
            Some(feature) => format!(
                "compression backend '{}' requires infotheory feature '{}'",
                descriptor.canonical, feature
            ),
            None => format!(
                "compression backend '{}' is unavailable",
                descriptor.canonical
            ),
        })
        .unwrap_or_else(|err| err)
}

pub(crate) fn rate_backend_kernel(kind: RateBackendKind) -> &'static RateBackendKernel {
    RATE_BACKEND_KERNELS
        .iter()
        .find(|kernel| kernel.kind == kind)
        .unwrap_or_else(|| panic!("missing RATE_BACKEND_KERNELS entry for {kind:?}"))
}

pub(crate) fn compression_backend_kernel(
    kind: CompressionBackendKind,
) -> &'static CompressionBackendKernel {
    COMPRESSION_BACKEND_KERNELS
        .iter()
        .find(|kernel| kernel.kind == kind)
        .unwrap_or_else(|| panic!("missing COMPRESSION_BACKEND_KERNELS entry for {kind:?}"))
}

pub(crate) fn rate_backend_canonical_name(kind: RateBackendKind) -> &'static str {
    describe_rate_backend_kind(kind)
        .unwrap_or_else(|err| panic!("{err}"))
        .canonical
}

pub(crate) fn compression_backend_canonical_name(kind: CompressionBackendKind) -> &'static str {
    describe_compression_backend_kind(kind)
        .unwrap_or_else(|err| panic!("{err}"))
        .canonical
}

pub(crate) fn compile_rate_backend_plan_via_kernel(
    kind: RateBackendKind,
    backend: &RateBackend,
    env: &SpecEnvironment,
    depth: usize,
) -> SpecResult<RateBackendPlan> {
    (rate_backend_kernel(kind).compile_plan)(backend, env, depth)
}

pub(crate) fn compile_compression_backend_plan_via_kernel(
    kind: CompressionBackendKind,
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    (compression_backend_kernel(kind).compile_plan)(backend, env)
}

pub(crate) fn rate_backend_wrapper_via_kernel(plan: &RateBackendPlan) -> RateBackend {
    (rate_backend_kernel(plan.kind()).to_wrapper)(plan)
}

pub(crate) fn compression_backend_wrapper_via_kernel(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    (compression_backend_kernel(plan.kind()).to_wrapper)(plan)
}

pub(crate) fn encode_rate_backend_payload_via_kernel(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    (rate_backend_kernel(plan.kind()).encode_payload)(plan, out);
}

pub(crate) fn encode_compression_backend_payload_via_kernel(
    plan: &CompressionBackendPlan,
    out: &mut Vec<u8>,
) {
    (compression_backend_kernel(plan.kind()).encode_payload)(plan, out);
}

pub(crate) fn rate_backend_display_label_via_kernel(
    plan: &RateBackendPlan,
    max_order: i64,
) -> String {
    (rate_backend_kernel(plan.kind()).display_label)(plan, max_order)
}

pub(crate) fn rate_backend_default_name_via_kernel(
    plan: &RateBackendPlan,
    max_order: i64,
) -> String {
    (rate_backend_kernel(plan.kind()).default_name)(plan, max_order)
}

pub(crate) fn compression_backend_display_label_via_kernel(
    plan: &CompressionBackendPlan,
) -> String {
    (compression_backend_kernel(plan.kind()).display_label)(plan)
}

pub(crate) fn rate_backend_capabilities_via_kernel(
    plan: &RateBackendPlan,
) -> RateBackendCapabilities {
    let kernel = rate_backend_kernel(plan.kind());
    RateBackendCapabilities {
        canonical_name: rate_backend_canonical_name(plan.kind()),
        display_label: Arc::<str>::from((kernel.display_label)(plan, -1)),
        trace_strategy: kernel.trace_strategy,
        supports_biased_entropy: kernel.supports_biased_entropy,
        supports_frozen_conditioning: kernel.supports_frozen_conditioning,
        supports_rate_coded_compression: kernel.supports_rate_coded_compression,
        supports_bit_token_adaptation: (kernel.supports_bit_token_adaptation)(plan),
        contains_zpaq: (kernel.contains_zpaq)(plan),
        method_family: kernel.method_family,
    }
}

pub(crate) fn compression_backend_capabilities_via_kernel(
    plan: &CompressionBackendPlan,
) -> CompressionBackendCapabilities {
    let kernel = compression_backend_kernel(plan.kind());
    CompressionBackendCapabilities {
        canonical_name: compression_backend_canonical_name(plan.kind()),
        display_label: Arc::<str>::from(compression_backend_display_label_via_kernel(plan)),
        uses_rate_backend: kernel.uses_rate_backend,
        supports_decompression: kernel.supports_decompression,
    }
}

pub(crate) fn adapt_rate_backend_for_bit_tokens_via_kernel(
    plan: &RateBackendPlan,
) -> RateBackendPlan {
    (rate_backend_kernel(plan.kind()).adapt_for_bit_tokens)(plan)
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
pub(crate) fn rate_backend_method(
    backend: &CompiledRateBackend,
    family: MethodBackendKind,
) -> Option<&str> {
    match (family, backend.capabilities().method_family) {
        #[cfg(feature = "backend-rwkv")]
        (MethodBackendKind::Rwkv7, Some(MethodBackendFamily::Rwkv7)) => backend.method_string(),
        #[cfg(feature = "backend-mamba")]
        (MethodBackendKind::Mamba, Some(MethodBackendFamily::Mamba)) => backend.method_string(),
        _ => None,
    }
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
pub(crate) fn rate_backend_trace_model_strategy(
    backend: &CompiledRateBackend,
) -> TraceModelStrategy {
    match backend.capabilities().trace_strategy {
        PublicTraceStrategy::Rosa => TraceModelStrategy::Rosa,
        PublicTraceStrategy::Ctw => TraceModelStrategy::Ctw,
        PublicTraceStrategy::FacCtw => TraceModelStrategy::FacCtw,
        PublicTraceStrategy::PredictorBacked => TraceModelStrategy::PredictorBacked,
        PublicTraceStrategy::Zpaq => TraceModelStrategy::Zpaq,
        PublicTraceStrategy::Mamba => TraceModelStrategy::Mamba,
        PublicTraceStrategy::Rwkv7 => TraceModelStrategy::Rwkv7,
    }
}

#[cfg(feature = "backend-rosa")]
fn build_predictor_rosa(
    _backend: &CompiledRateBackend,
    max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let mut model = RosaPlus::new(max_order, false, 0, 42);
    model.build_lm_full_bytes_no_finalize_endpos();
    Ok(crate::mixture::RateBackendPredictor::Rosa {
        model,
        min_prob,
        checkpoint_journal: Vec::new(),
        checkpoint_depth: 0,
    })
}

#[cfg(not(feature = "backend-rosa"))]
fn build_predictor_rosa(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::RosaPlus))
}

#[cfg(feature = "backend-match")]
fn build_predictor_match(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = backend.plan()
    else {
        unreachable!("match kernel used with non-match plan")
    };
    Ok(crate::mixture::RateBackendPredictor::Match {
        model: MatchModel::new_contiguous(
            *hash_bits,
            *min_len,
            *max_len,
            *base_mix,
            *confidence_scale,
        ),
        min_prob,
    })
}

#[cfg(not(feature = "backend-match"))]
fn build_predictor_match(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Match))
}

#[cfg(feature = "backend-match")]
fn build_predictor_sparse_match(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = backend.plan()
    else {
        unreachable!("sparse-match kernel used with non-sparse-match plan")
    };
    Ok(crate::mixture::RateBackendPredictor::SparseMatch {
        model: SparseMatchModel::new(
            *hash_bits,
            *min_len,
            *max_len,
            *gap_min,
            *gap_max,
            *base_mix,
            *confidence_scale,
        ),
        min_prob,
    })
}

#[cfg(not(feature = "backend-match"))]
fn build_predictor_sparse_match(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::SparseMatch))
}

#[cfg(feature = "backend-ppmd")]
fn build_predictor_ppmd(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Ppmd { order, memory_mb } = backend.plan() else {
        unreachable!("ppmd kernel used with non-ppmd plan")
    };
    Ok(crate::mixture::RateBackendPredictor::Ppmd {
        model: PpmdModel::new(*order, *memory_mb),
        min_prob,
    })
}

#[cfg(not(feature = "backend-ppmd"))]
fn build_predictor_ppmd(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Ppmd))
}

#[cfg(feature = "backend-sequitur")]
fn build_predictor_sequitur(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Sequitur { context_bytes } = backend.plan() else {
        unreachable!("sequitur kernel used with non-sequitur plan")
    };
    Ok(crate::mixture::RateBackendPredictor::Sequitur {
        model: SequiturModel::new(*context_bytes),
        min_prob,
    })
}

#[cfg(not(feature = "backend-sequitur"))]
fn build_predictor_sequitur(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Sequitur))
}

#[cfg(feature = "backend-ctw")]
fn build_predictor_ctw(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Ctw { depth } = backend.plan() else {
        unreachable!("ctw kernel used with non-ctw plan")
    };
    Ok(crate::mixture::RateBackendPredictor::Ctw {
        tree: FacContextTree::new(*depth, 8),
        min_prob,
        checkpoint_journal: Vec::new(),
        checkpoint_depth: 0,
    })
}

#[cfg(not(feature = "backend-ctw"))]
fn build_predictor_ctw(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Ctw))
}

#[cfg(feature = "backend-ctw")]
fn build_predictor_fac_ctw(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits: _,
        encoding_bits,
    } = backend.plan()
    else {
        unreachable!("fac-ctw kernel used with non-fac-ctw plan")
    };
    let bits_per_symbol = (*encoding_bits).clamp(1, 8);
    Ok(crate::mixture::RateBackendPredictor::FacCtw {
        tree: FacContextTree::new(*base_depth, bits_per_symbol),
        bits_per_symbol,
        min_prob,
        checkpoint_journal: Vec::new(),
        checkpoint_depth: 0,
    })
}

#[cfg(not(feature = "backend-ctw"))]
fn build_predictor_fac_ctw(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::FacCtw))
}

#[cfg(feature = "backend-rwkv")]
fn build_predictor_rwkv(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Rwkv7 { parsed_method, .. } = backend.plan() else {
        unreachable!("rwkv kernel used with non-rwkv plan")
    };
    let mut compressor = rwkvzip::Compressor::new_from_method_spec(parsed_method)
        .map_err(|e| format!("invalid rwkv method: {e}"))?;
    compressor.reset_and_prime();
    Ok(crate::mixture::RateBackendPredictor::Rwkv7 {
        pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
        compressor,
        primed: true,
        min_prob,
    })
}

#[cfg(not(feature = "backend-rwkv"))]
fn build_predictor_rwkv(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Rwkv7))
}

#[cfg(feature = "backend-mamba")]
fn build_predictor_mamba(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Mamba { parsed_method, .. } = backend.plan() else {
        unreachable!("mamba kernel used with non-mamba plan")
    };
    let mut compressor = mambazip::Compressor::new_from_method_spec(parsed_method)
        .map_err(|e| format!("invalid mamba method: {e}"))?;
    let bias = compressor.online_bias_snapshot();
    let logits = compressor
        .model
        .forward(&mut compressor.scratch, 0, &mut compressor.state);
    mambazip::Compressor::logits_to_pdf(logits, bias.as_deref(), &mut compressor.pdf_buffer);
    Ok(crate::mixture::RateBackendPredictor::Mamba {
        pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
        compressor,
        primed: true,
        min_prob,
    })
}

#[cfg(not(feature = "backend-mamba"))]
fn build_predictor_mamba(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Mamba))
}

#[cfg(feature = "backend-zpaq")]
fn build_predictor_zpaq(
    backend: &CompiledRateBackend,
    _max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Zpaq { method } = backend.plan() else {
        unreachable!("zpaq kernel used with non-zpaq plan")
    };
    Ok(crate::mixture::RateBackendPredictor::Zpaq {
        model: ZpaqRateModel::new(method.clone(), min_prob),
    })
}

#[cfg(not(feature = "backend-zpaq"))]
fn build_predictor_zpaq(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Zpaq))
}

#[cfg(feature = "backend-mixture")]
fn build_predictor_mixture(
    backend: &CompiledRateBackend,
    max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let experts = crate::mixture::expert_configs_from_compiled_mixture(backend, max_order)?;
    let runtime = crate::mixture::build_mixture_runtime_from_compiled(backend, &experts)
        .map_err(|e| format!("MixtureSpec invalid: {e}"))?;
    Ok(crate::mixture::RateBackendPredictor::Mixture { runtime })
}

#[cfg(not(feature = "backend-mixture"))]
fn build_predictor_mixture(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Mixture))
}

#[cfg(feature = "backend-particle")]
fn build_predictor_particle(
    backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Particle { spec } = backend.plan() else {
        unreachable!("particle kernel used with non-particle plan")
    };
    Ok(crate::mixture::RateBackendPredictor::Particle {
        runtime: ParticleRuntime::new(spec),
    })
}

#[cfg(not(feature = "backend-particle"))]
fn build_predictor_particle(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Particle))
}

#[cfg(feature = "backend-calibrated")]
fn build_predictor_calibrated(
    backend: &CompiledRateBackend,
    max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    let crate::spec::core::RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = backend.plan()
    else {
        unreachable!("calibrated kernel used with non-calibrated plan")
    };
    let base_backend = crate::spec::core::compiled_rate_backend_from_plan_unchecked(base.clone());
    Ok(crate::mixture::RateBackendPredictor::Calibrated {
        base: Box::new(build_rate_backend_predictor_via_kernel(
            &base_backend,
            max_order,
            min_prob,
        )?),
        core: CalibratorCore::new(*context, *bins, *learning_rate, *bias_clip),
        pdf: [1.0 / 256.0; 256],
        valid: false,
        min_prob,
    })
}

#[cfg(not(feature = "backend-calibrated"))]
fn build_predictor_calibrated(
    _backend: &CompiledRateBackend,
    _max_order: i64,
    _min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    Err(rate_backend_feature_error(RateBackendKind::Calibrated))
}

#[cfg(feature = "backend-rosa")]
fn build_pdf_predictor_rosa(
    _backend: &CompiledRateBackend,
    max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    Ok(crate::compression::RatePdfPredictor::Rosa(
        crate::compression::RosaPredictor::new(max_order),
    ))
}

#[cfg(not(feature = "backend-rosa"))]
fn build_pdf_predictor_rosa(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::RosaPlus))
}

#[cfg(feature = "backend-match")]
fn build_pdf_predictor_match(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = backend.plan()
    else {
        unreachable!("match kernel used with non-match plan")
    };
    Ok(crate::compression::RatePdfPredictor::Match {
        model: MatchModel::new_contiguous(
            *hash_bits,
            *min_len,
            *max_len,
            *base_mix,
            *confidence_scale,
        ),
    })
}

#[cfg(not(feature = "backend-match"))]
fn build_pdf_predictor_match(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Match))
}

#[cfg(feature = "backend-match")]
fn build_pdf_predictor_sparse_match(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = backend.plan()
    else {
        unreachable!("sparse-match kernel used with non-sparse-match plan")
    };
    Ok(crate::compression::RatePdfPredictor::SparseMatch {
        model: SparseMatchModel::new(
            *hash_bits,
            *min_len,
            *max_len,
            *gap_min,
            *gap_max,
            *base_mix,
            *confidence_scale,
        ),
    })
}

#[cfg(not(feature = "backend-match"))]
fn build_pdf_predictor_sparse_match(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!(
        "{}",
        rate_backend_feature_error(RateBackendKind::SparseMatch)
    )
}

#[cfg(feature = "backend-ppmd")]
fn build_pdf_predictor_ppmd(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Ppmd { order, memory_mb } = backend.plan() else {
        unreachable!("ppmd kernel used with non-ppmd plan")
    };
    Ok(crate::compression::RatePdfPredictor::Ppmd {
        model: PpmdModel::new(*order, *memory_mb),
    })
}

#[cfg(not(feature = "backend-ppmd"))]
fn build_pdf_predictor_ppmd(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Ppmd))
}

#[cfg(feature = "backend-sequitur")]
fn build_pdf_predictor_sequitur(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Sequitur { context_bytes } = backend.plan() else {
        unreachable!("sequitur kernel used with non-sequitur plan")
    };
    Ok(crate::compression::RatePdfPredictor::Sequitur {
        model: SequiturModel::new(*context_bytes),
    })
}

#[cfg(not(feature = "backend-sequitur"))]
fn build_pdf_predictor_sequitur(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Sequitur))
}

#[cfg(feature = "backend-ctw")]
fn build_pdf_predictor_ctw(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Ctw { depth } = backend.plan() else {
        unreachable!("ctw kernel used with non-ctw plan")
    };
    Ok(crate::compression::RatePdfPredictor::Ctw(
        crate::compression::CtwPredictor::new_ctw(*depth),
    ))
}

#[cfg(not(feature = "backend-ctw"))]
fn build_pdf_predictor_ctw(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Ctw))
}

#[cfg(feature = "backend-ctw")]
fn build_pdf_predictor_fac_ctw(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits: _,
        encoding_bits,
    } = backend.plan()
    else {
        unreachable!("fac-ctw kernel used with non-fac-ctw plan")
    };
    Ok(crate::compression::RatePdfPredictor::FacCtw(
        crate::compression::CtwPredictor::new_fac(*base_depth, (*encoding_bits).clamp(1, 8)),
    ))
}

#[cfg(not(feature = "backend-ctw"))]
fn build_pdf_predictor_fac_ctw(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::FacCtw))
}

#[cfg(feature = "backend-mamba")]
fn build_pdf_predictor_mamba(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Mamba { parsed_method, .. } = backend.plan() else {
        unreachable!("mamba kernel used with non-mamba plan")
    };
    Ok(crate::compression::RatePdfPredictor::Mamba(
        crate::compression::MambaPredictor::from_method_spec(parsed_method)?,
    ))
}

#[cfg(not(feature = "backend-mamba"))]
fn build_pdf_predictor_mamba(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Mamba))
}

#[cfg(feature = "backend-rwkv")]
fn build_pdf_predictor_rwkv(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Rwkv7 { parsed_method, .. } = backend.plan() else {
        unreachable!("rwkv kernel used with non-rwkv plan")
    };
    Ok(crate::compression::RatePdfPredictor::Rwkv(
        crate::compression::RwkvPredictor::from_method_spec(parsed_method)?,
    ))
}

#[cfg(not(feature = "backend-rwkv"))]
fn build_pdf_predictor_rwkv(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Rwkv7))
}

#[cfg(feature = "backend-zpaq")]
fn build_pdf_predictor_zpaq(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Zpaq { method } = backend.plan() else {
        unreachable!("zpaq kernel used with non-zpaq plan")
    };
    Ok(crate::compression::RatePdfPredictor::Zpaq(
        crate::compression::ZpaqPredictor::new(method.clone()),
    ))
}

#[cfg(not(feature = "backend-zpaq"))]
fn build_pdf_predictor_zpaq(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Zpaq))
}

#[cfg(feature = "backend-mixture")]
fn build_pdf_predictor_mixture(
    backend: &CompiledRateBackend,
    max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    Ok(crate::compression::RatePdfPredictor::Mixture(
        crate::compression::MixturePredictor::new_from_compiled(backend, max_order)?,
    ))
}

#[cfg(not(feature = "backend-mixture"))]
fn build_pdf_predictor_mixture(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Mixture))
}

#[cfg(feature = "backend-particle")]
fn build_pdf_predictor_particle(
    backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Particle { spec } = backend.plan() else {
        unreachable!("particle kernel used with non-particle plan")
    };
    Ok(crate::compression::RatePdfPredictor::Particle(
        ParticleRuntime::new(spec),
    ))
}

#[cfg(not(feature = "backend-particle"))]
fn build_pdf_predictor_particle(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!("{}", rate_backend_feature_error(RateBackendKind::Particle))
}

#[cfg(feature = "backend-calibrated")]
fn build_pdf_predictor_calibrated(
    backend: &CompiledRateBackend,
    max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    let crate::spec::core::RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = backend.plan()
    else {
        unreachable!("calibrated kernel used with non-calibrated plan")
    };
    let base_backend = crate::spec::core::compiled_rate_backend_from_plan_unchecked(base.clone());
    Ok(crate::compression::RatePdfPredictor::Calibrated {
        base: Box::new(build_rate_pdf_predictor_via_kernel(
            &base_backend,
            max_order,
        )?),
        core: CalibratorCore::new(*context, *bins, *learning_rate, *bias_clip),
        pdf: vec![1.0 / 256.0; 256],
        valid: false,
    })
}

#[cfg(not(feature = "backend-calibrated"))]
fn build_pdf_predictor_calibrated(
    _backend: &CompiledRateBackend,
    _max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    anyhow::bail!(
        "{}",
        rate_backend_feature_error(RateBackendKind::Calibrated)
    )
}

fn entropy_prequential(
    data: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::try_prequential_rate_backend(data, &[], max_order, backend)
}

fn joint_entropy_prequential(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    entropy_prequential(&joint, max_order, backend).map(|bits| bits * 2.0)
}

fn conditional_chain_prequential(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::try_prequential_rate_backend(data, prefix_parts, -1, backend)
}

#[cfg(feature = "backend-rosa")]
fn entropy_rosa(
    data: &[u8],
    max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut model = RosaPlus::new(max_order, false, 0, 42);
    Ok(model.predictive_entropy_rate(data))
}

#[cfg(not(feature = "backend-rosa"))]
fn entropy_rosa(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::RosaPlus,
    )))
}

#[cfg(feature = "backend-rosa")]
fn joint_entropy_rosa(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint_symbols: Vec<u32> = (0..x.len())
        .map(|idx| (x[idx] as u32) * 256 + (y[idx] as u32))
        .collect();
    let mut model = RosaPlus::new(max_order, false, 0, 42);
    Ok(model.entropy_rate_cps(&joint_symbols))
}

#[cfg(not(feature = "backend-rosa"))]
fn joint_entropy_rosa(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::RosaPlus,
    )))
}

fn conditional_chain_rosa(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::try_frozen_plugin_rate_backend(data, prefix_parts, -1, backend)
}

#[cfg(feature = "backend-rwkv")]
fn entropy_rwkv(
    data: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Rwkv7 {
        method,
        parsed_method,
        ..
    } = backend.plan()
    else {
        unreachable!("rwkv kernel used with non-rwkv plan")
    };
    crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
        c.cross_entropy(data).map_err(|err| {
            InfotheoryError::runtime(format!("rwkv method entropy scoring failed: {err:#}"))
        })
    })
}

#[cfg(not(feature = "backend-rwkv"))]
fn entropy_rwkv(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Rwkv7,
    )))
}

#[cfg(feature = "backend-rwkv")]
fn joint_entropy_rwkv(
    x: &[u8],
    y: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Rwkv7 {
        method,
        parsed_method,
        ..
    } = backend.plan()
    else {
        unreachable!("rwkv kernel used with non-rwkv plan")
    };
    crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
        c.joint_cross_entropy_aligned_min(x, y).map_err(|err| {
            InfotheoryError::runtime(format!("rwkv method joint-entropy scoring failed: {err:#}"))
        })
    })
}

#[cfg(not(feature = "backend-rwkv"))]
fn joint_entropy_rwkv(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Rwkv7,
    )))
}

#[cfg(feature = "backend-rwkv")]
fn conditional_chain_rwkv(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Rwkv7 {
        method,
        parsed_method,
        ..
    } = backend.plan()
    else {
        unreachable!("rwkv kernel used with non-rwkv plan")
    };
    crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
        c.cross_entropy_conditional_chain(prefix_parts, data)
            .map_err(|err| {
                InfotheoryError::runtime(format!(
                    "rwkv method conditional-chain scoring failed: {err:#}"
                ))
            })
    })
}

#[cfg(not(feature = "backend-rwkv"))]
fn conditional_chain_rwkv(
    _prefix_parts: &[&[u8]],
    _data: &[u8],
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Rwkv7,
    )))
}

#[cfg(feature = "backend-mamba")]
fn entropy_mamba(
    data: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Mamba {
        method,
        parsed_method,
        ..
    } = backend.plan()
    else {
        unreachable!("mamba kernel used with non-mamba plan")
    };
    crate::with_mamba_method_spec_tls(method, parsed_method, |c| {
        c.cross_entropy(data).map_err(|err| {
            InfotheoryError::runtime(format!("mamba method entropy scoring failed: {err:#}"))
        })
    })
}

#[cfg(not(feature = "backend-mamba"))]
fn entropy_mamba(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Mamba,
    )))
}

#[cfg(feature = "backend-mamba")]
fn joint_entropy_mamba(
    x: &[u8],
    y: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Mamba {
        method,
        parsed_method,
        ..
    } = backend.plan()
    else {
        unreachable!("mamba kernel used with non-mamba plan")
    };
    crate::with_mamba_method_spec_tls(method, parsed_method, |c| {
        c.joint_cross_entropy_aligned_min(x, y).map_err(|err| {
            InfotheoryError::runtime(format!(
                "mamba method joint-entropy scoring failed: {err:#}"
            ))
        })
    })
}

#[cfg(not(feature = "backend-mamba"))]
fn joint_entropy_mamba(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Mamba,
    )))
}

#[cfg(feature = "backend-mamba")]
fn conditional_chain_mamba(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Mamba {
        method,
        parsed_method,
        ..
    } = backend.plan()
    else {
        unreachable!("mamba kernel used with non-mamba plan")
    };
    crate::with_mamba_method_spec_tls(method, parsed_method, |c| {
        c.cross_entropy_conditional_chain(prefix_parts, data)
            .map_err(|err| {
                InfotheoryError::runtime(format!(
                    "mamba method conditional-chain scoring failed: {err:#}"
                ))
            })
    })
}

#[cfg(not(feature = "backend-mamba"))]
fn conditional_chain_mamba(
    _prefix_parts: &[&[u8]],
    _data: &[u8],
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Mamba,
    )))
}

#[cfg(feature = "backend-zpaq")]
fn entropy_zpaq(
    data: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Zpaq { method } = backend.plan() else {
        unreachable!("zpaq kernel used with non-zpaq plan")
    };
    zpaq_conditional_chain_rate_bits(method, &[], data)
}

#[cfg(not(feature = "backend-zpaq"))]
fn entropy_zpaq(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Zpaq,
    )))
}

#[cfg(feature = "backend-zpaq")]
fn joint_entropy_zpaq(
    x: &[u8],
    y: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Zpaq { method } = backend.plan() else {
        unreachable!("zpaq kernel used with non-zpaq plan")
    };
    zpaq_joint_entropy_rate_bits(method, x, y)
}

#[cfg(not(feature = "backend-zpaq"))]
fn joint_entropy_zpaq(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Zpaq,
    )))
}

#[cfg(feature = "backend-zpaq")]
fn conditional_chain_zpaq(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Zpaq { method } = backend.plan() else {
        unreachable!("zpaq kernel used with non-zpaq plan")
    };
    zpaq_conditional_chain_rate_bits(method, prefix_parts, data)
}

#[cfg(not(feature = "backend-zpaq"))]
fn conditional_chain_zpaq(
    _prefix_parts: &[&[u8]],
    _data: &[u8],
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Zpaq,
    )))
}

#[cfg(feature = "backend-mixture")]
fn entropy_mixture(
    data: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    mixture_entropy_rate_bits(data, backend, max_order)
}

#[cfg(not(feature = "backend-mixture"))]
fn entropy_mixture(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Mixture,
    )))
}

#[cfg(feature = "backend-mixture")]
fn joint_entropy_mixture(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    mixture_joint_entropy_rate_bits(x, y, backend, max_order)
}

#[cfg(not(feature = "backend-mixture"))]
fn joint_entropy_mixture(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Mixture,
    )))
}

#[cfg(feature = "backend-mixture")]
fn conditional_chain_mixture(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    mixture_conditional_chain_rate_bits(prefix_parts, data, backend)
}

#[cfg(not(feature = "backend-mixture"))]
fn conditional_chain_mixture(
    _prefix_parts: &[&[u8]],
    _data: &[u8],
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Mixture,
    )))
}

#[cfg(feature = "backend-particle")]
fn entropy_particle(
    data: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Particle { spec } = backend.plan() else {
        unreachable!("particle kernel used with non-particle plan")
    };
    particle_stream_entropy_rate_bits(data, spec)
}

#[cfg(not(feature = "backend-particle"))]
fn entropy_particle(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Particle,
    )))
}

#[cfg(feature = "backend-particle")]
fn joint_entropy_particle(
    x: &[u8],
    y: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Particle { spec } = backend.plan() else {
        unreachable!("particle kernel used with non-particle plan")
    };
    particle_joint_entropy_rate_bits(x, y, spec)
}

#[cfg(not(feature = "backend-particle"))]
fn joint_entropy_particle(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Particle,
    )))
}

#[cfg(feature = "backend-particle")]
fn conditional_chain_particle(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Particle { spec } = backend.plan() else {
        unreachable!("particle kernel used with non-particle plan")
    };
    particle_conditional_chain_rate_bits(prefix_parts, data, spec)
}

#[cfg(not(feature = "backend-particle"))]
fn conditional_chain_particle(
    _prefix_parts: &[&[u8]],
    _data: &[u8],
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Particle,
    )))
}

#[cfg(feature = "backend-ctw")]
fn entropy_ctw(
    data: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Ctw { depth } = backend.plan() else {
        unreachable!("ctw kernel used with non-ctw plan")
    };
    ctw_entropy_rate_bits(*depth, data)
}

#[cfg(not(feature = "backend-ctw"))]
fn entropy_ctw(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Ctw,
    )))
}

#[cfg(feature = "backend-ctw")]
fn joint_entropy_ctw(
    x: &[u8],
    y: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Ctw { depth } = backend.plan() else {
        unreachable!("ctw kernel used with non-ctw plan")
    };
    ctw_joint_entropy_rate_bits(*depth, x, y)
}

#[cfg(not(feature = "backend-ctw"))]
fn joint_entropy_ctw(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Ctw,
    )))
}

#[cfg(feature = "backend-ctw")]
fn conditional_chain_ctw(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::Ctw { depth } = backend.plan() else {
        unreachable!("ctw kernel used with non-ctw plan")
    };
    ctw_conditional_chain_rate_bits(*depth, prefix_parts, data)
}

#[cfg(not(feature = "backend-ctw"))]
fn conditional_chain_ctw(
    _prefix_parts: &[&[u8]],
    _data: &[u8],
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::Ctw,
    )))
}

#[cfg(feature = "backend-ctw")]
fn entropy_fac_ctw(
    data: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits: _,
        encoding_bits,
    } = backend.plan()
    else {
        unreachable!("fac-ctw kernel used with non-fac-ctw plan")
    };
    fac_ctw_entropy_rate_bits(*base_depth, *encoding_bits, data)
}

#[cfg(not(feature = "backend-ctw"))]
fn entropy_fac_ctw(
    _data: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::FacCtw,
    )))
}

#[cfg(feature = "backend-ctw")]
fn joint_entropy_fac_ctw(
    x: &[u8],
    y: &[u8],
    _max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits: _,
        encoding_bits,
    } = backend.plan()
    else {
        unreachable!("fac-ctw kernel used with non-fac-ctw plan")
    };
    fac_ctw_joint_entropy_rate_bits(*base_depth, *encoding_bits, x, y)
}

#[cfg(not(feature = "backend-ctw"))]
fn joint_entropy_fac_ctw(
    _x: &[u8],
    _y: &[u8],
    _max_order: i64,
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::FacCtw,
    )))
}

#[cfg(feature = "backend-ctw")]
fn conditional_chain_fac_ctw(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let crate::spec::core::RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits: _,
        encoding_bits,
    } = backend.plan()
    else {
        unreachable!("fac-ctw kernel used with non-fac-ctw plan")
    };
    fac_ctw_conditional_chain_rate_bits(*base_depth, *encoding_bits, prefix_parts, data)
}

#[cfg(not(feature = "backend-ctw"))]
fn conditional_chain_fac_ctw(
    _prefix_parts: &[&[u8]],
    _data: &[u8],
    _backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    Err(InfotheoryError::unsupported(rate_backend_feature_error(
        RateBackendKind::FacCtw,
    )))
}

fn build_compression_runtime_zpaq(
    backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    let crate::spec::core::CompressionBackendPlan::Zpaq { method } = backend.plan() else {
        unreachable!("zpaq compression kernel used with non-zpaq plan")
    };
    Ok(CompressionRuntimeHandle::Zpaq {
        method: method.clone(),
    })
}

#[cfg(feature = "backend-rwkv")]
fn build_compression_runtime_rwkv(
    backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    let crate::spec::core::CompressionBackendPlan::Rwkv7 {
        method,
        parsed_method,
        coder,
        ..
    } = backend.plan()
    else {
        unreachable!("rwkv compression kernel used with non-rwkv compression plan")
    };
    Ok(CompressionRuntimeHandle::Rwkv7 {
        method: method.clone(),
        parsed_method: parsed_method.clone(),
        coder: *coder,
    })
}

#[cfg(not(feature = "backend-rwkv"))]
fn build_compression_runtime_rwkv(
    _backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    Err(compression_backend_feature_error(
        CompressionBackendKind::Rwkv7,
    ))
}

fn build_compression_runtime_rate(
    backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    let crate::spec::core::CompressionBackendPlan::Rate {
        rate_backend,
        coder,
        framing,
    } = backend.plan()
    else {
        unreachable!("rate compression kernel used with non-rate compression plan")
    };
    Ok(CompressionRuntimeHandle::Rate {
        rate_backend: crate::spec::core::compiled_rate_backend_from_plan_unchecked(
            rate_backend.clone(),
        ),
        coder: *coder,
        framing: *framing,
    })
}

fn build_rate_backend_predictor_via_kernel(
    backend: &CompiledRateBackend,
    max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    (rate_backend_kernel(backend.plan().kind()).build_predictor)(backend, max_order, min_prob)
}

fn build_rate_pdf_predictor_via_kernel(
    backend: &CompiledRateBackend,
    max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    (rate_backend_kernel(backend.plan().kind()).build_pdf_predictor)(backend, max_order)
}

/// Shared byte-level runtime predictor trait.
pub trait BytePredictor: crate::mixture::OnlineBytePredictor {}

impl<T> BytePredictor for T where T: crate::mixture::OnlineBytePredictor + ?Sized {}

/// Runtime predictors that support checkpoint/rollback.
#[allow(dead_code)]
pub trait CheckpointablePredictor {
    /// Concrete checkpoint type.
    type Checkpoint: Clone;

    /// Snapshot the current runtime state.
    fn checkpoint(&mut self) -> Self::Checkpoint;

    /// Restore a previous snapshot.
    fn restore_checkpoint(&mut self, checkpoint: &Self::Checkpoint);
}

/// Shared runtime factory trait for byte-level predictors.
pub trait PredictorFactory {
    /// Predictor type produced by this factory.
    type Predictor: BytePredictor + CheckpointablePredictor;

    /// Build a predictor runtime from a spec object.
    fn build_predictor(&self, max_order: i64, min_prob: f64) -> Result<Self::Predictor, String>;
}

/// Shared runtime trait for compression-capable backends.
pub trait CompressionRuntime {
    /// Compressed size of a single byte slice.
    fn compress_size(&mut self, data: &[u8]) -> InfotheoryResult<u64>;

    /// Compressed size of chained slices encoded as one stream.
    fn compress_size_chain(&mut self, parts: &[&[u8]]) -> InfotheoryResult<u64>;

    /// Encode raw bytes.
    fn compress_bytes(&mut self, data: &[u8]) -> InfotheoryResult<Vec<u8>>;

    /// Decode previously encoded bytes.
    fn decompress_bytes(&mut self, input: &[u8]) -> InfotheoryResult<Vec<u8>>;
}

/// Shared runtime factory trait for compression backends.
pub trait CompressionFactory {
    /// Runtime type produced by this factory.
    type Runtime: CompressionRuntime;

    /// Build a compression runtime from a spec object.
    fn build_compression_runtime(&self) -> Result<Self::Runtime, String>;
}

impl CheckpointablePredictor for crate::mixture::RateBackendPredictor {
    type Checkpoint = crate::mixture::RateBackendPredictorCheckpoint;

    fn checkpoint(&mut self) -> Self::Checkpoint {
        crate::mixture::RateBackendPredictor::checkpoint(self)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self::Checkpoint) {
        crate::mixture::RateBackendPredictor::restore_checkpoint(self, checkpoint);
    }
}

impl PredictorFactory for CompiledRateBackend {
    type Predictor = crate::mixture::RateBackendPredictor;

    fn build_predictor(&self, max_order: i64, min_prob: f64) -> Result<Self::Predictor, String> {
        build_rate_backend_predictor_via_kernel(self, max_order, min_prob)
    }
}

struct SliceChainReader<'a> {
    parts: &'a [&'a [u8]],
    i: usize,
    off: usize,
}

impl<'a> SliceChainReader<'a> {
    fn new(parts: &'a [&'a [u8]]) -> Self {
        Self {
            parts,
            i: 0,
            off: 0,
        }
    }
}

impl<'a> std::io::Read for SliceChainReader<'a> {
    fn read(&mut self, mut buf: &mut [u8]) -> std::io::Result<usize> {
        let mut total = 0;
        if buf.is_empty() {
            return Ok(0);
        }
        while self.i < self.parts.len() {
            let p = self.parts[self.i];
            if self.off >= p.len() {
                self.i += 1;
                self.off = 0;
                continue;
            }
            let n = (p.len() - self.off).min(buf.len());
            buf[..n].copy_from_slice(&p[self.off..self.off + n]);
            self.off += n;
            total += n;
            let tmp = buf;
            buf = &mut tmp[n..];
            if buf.is_empty() {
                break;
            }
        }
        Ok(total)
    }
}

/// Concrete compression runtime handle built from a [`CompressionBackend`] spec.
pub enum CompressionRuntimeHandle {
    Zpaq {
        method: String,
    },
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        method: String,
        parsed_method: crate::rwkvzip::MethodSpec,
        coder: crate::coders::CoderType,
    },
    Rate {
        rate_backend: CompiledRateBackend,
        coder: crate::coders::CoderType,
        framing: crate::compression::FramingMode,
    },
}

impl CompressionRuntime for CompressionRuntimeHandle {
    fn compress_size(&mut self, data: &[u8]) -> InfotheoryResult<u64> {
        match self {
            CompressionRuntimeHandle::Zpaq { method } => {
                crate::try_zpaq_compress_size_bytes(data, method.as_str())
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                coder,
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
                c.compress_size(data, *coder).map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 compression failed: {err:#}"))
                })
            }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_size(data, rate_backend, -1, *coder, *framing)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rate-coded compression failed: {err:#}"))
                }),
        }
    }

    fn compress_size_chain(&mut self, parts: &[&[u8]]) -> InfotheoryResult<u64> {
        match self {
            CompressionRuntimeHandle::Zpaq { method } => {
                let reader = SliceChainReader::new(parts);
                crate::try_zpaq_compress_size_stream(reader, method.as_str())
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                coder,
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
                c.compress_size_chain(parts, *coder).map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 chain compression failed: {err:#}"))
                })
            }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_size_chain(
                parts,
                rate_backend,
                -1,
                *coder,
                *framing,
            )
            .map_err(|err| {
                InfotheoryError::runtime(format!("rate-coded chain compression failed: {err:#}"))
            }),
        }
    }

    fn compress_bytes(&mut self, data: &[u8]) -> InfotheoryResult<Vec<u8>> {
        match self {
            CompressionRuntimeHandle::Zpaq { method } => crate::zpaq_compress_to_vec(data, method)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("zpaq byte compression failed: {err:#}"))
                }),
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                coder,
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
                c.compress(data, *coder)
            })
            .map_err(|err| {
                InfotheoryError::runtime(format!("rwkv7 byte compression failed: {err:#}"))
            }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_bytes(data, rate_backend, -1, *coder, *framing)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rate-coded byte compression failed: {err:#}"))
                }),
        }
    }

    fn decompress_bytes(&mut self, input: &[u8]) -> InfotheoryResult<Vec<u8>> {
        match self {
            CompressionRuntimeHandle::Zpaq { .. } => {
                crate::zpaq_decompress_to_vec(input).map_err(|err| {
                    InfotheoryError::runtime(format!("zpaq decompression failed: {err:#}"))
                })
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                ..
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| c.decompress(input))
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 decompression failed: {err:#}"))
                }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => {
                crate::compression::decompress_rate_bytes(input, rate_backend, -1, *coder, *framing)
                    .map_err(|err| {
                        InfotheoryError::runtime(format!(
                            "rate-coded decompression failed: {err:#}"
                        ))
                    })
            }
        }
    }
}

impl CompressionFactory for CompiledCompressionBackend {
    type Runtime = CompressionRuntimeHandle;

    fn build_compression_runtime(&self) -> Result<Self::Runtime, String> {
        (compression_backend_kernel(self.plan().kind()).build_runtime)(self)
    }
}

pub(crate) fn try_describe_rate_backend(
    backend: &RateBackend,
) -> Result<&'static RateBackendDescriptor, String> {
    describe_rate_backend_kind(backend.kind())
}

pub(crate) fn try_describe_compression_backend(
    backend: &CompressionBackend,
) -> Result<&'static CompressionBackendDescriptor, String> {
    describe_compression_backend_kind(backend.kind())
}

/// Shared spec -> predictor runtime builder using the default probability floor.
pub(crate) fn build_rate_backend_predictor(
    backend: &CompiledRateBackend,
    max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    backend.build_predictor(max_order, min_prob)
}

/// Shared spec -> predictor runtime builder using the library's default probability floor.
pub(crate) fn build_rate_backend_predictor_default(
    backend: &CompiledRateBackend,
    max_order: i64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    build_rate_backend_predictor(backend, max_order, crate::mixture::DEFAULT_MIN_PROB)
}

/// Shared spec -> compression predictor runtime builder.
pub(crate) fn build_rate_pdf_predictor(
    backend: &CompiledRateBackend,
    max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    build_rate_pdf_predictor_via_kernel(backend, max_order)
}

/// Shared spec -> compression runtime builder.
pub(crate) fn build_compression_runtime(
    backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    backend.build_compression_runtime()
}

fn interleave_aligned_bytes(x: &[u8], y: &[u8]) -> Vec<u8> {
    let mut joint = Vec::with_capacity(x.len() * 2);
    for (&xb, &yb) in x.iter().zip(y.iter()) {
        joint.push(xb);
        joint.push(yb);
    }
    joint
}

#[cfg(feature = "backend-zpaq")]
fn zpaq_conditional_chain_rate_bits(
    method: &str,
    prefix_parts: &[&[u8]],
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut model = ZpaqRateModel::new(method.to_owned(), 2f64.powi(-24));
    for &part in prefix_parts {
        model.update_and_score(part);
    }
    let bits = model.update_and_score(data);
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-zpaq")]
fn zpaq_joint_entropy_rate_bits(method: &str, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    let mut model = ZpaqRateModel::new(method.to_owned(), 2f64.powi(-24));
    let bits = model.update_and_score(&joint);
    Ok(bits / (x.len() as f64))
}

#[cfg(feature = "backend-mixture")]
fn build_compiled_mixture_runtime(
    backend: &CompiledRateBackend,
    max_order_fallback: i64,
) -> Result<crate::mixture::MixtureRuntime, InfotheoryError> {
    let experts =
        crate::mixture::expert_configs_from_compiled_mixture(backend, max_order_fallback)?;
    crate::mixture::build_mixture_runtime_from_compiled(backend, &experts).map_err(|err| {
        InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {err}"))
    })
}

#[cfg(feature = "backend-mixture")]
fn mixture_entropy_rate_bits(
    data: &[u8],
    backend: &CompiledRateBackend,
    max_order: i64,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut mix = build_compiled_mixture_runtime(backend, max_order)?;
    mix.begin_stream(Some(data.len() as u64))
        .map_err(|err| InfotheoryError::runtime(format!("Mixture stream init failed: {err}")))?;
    let mut bits = 0.0;
    for &byte in data {
        bits -= mix.step(byte) / std::f64::consts::LN_2;
    }
    mix.finish_stream().map_err(|err| {
        InfotheoryError::runtime(format!("Mixture stream finalize failed: {err}"))
    })?;
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-mixture")]
fn mixture_joint_entropy_rate_bits(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
    max_order: i64,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    let mut mix = build_compiled_mixture_runtime(backend, max_order)?;
    mix.begin_stream(Some(joint.len() as u64))
        .map_err(|err| InfotheoryError::runtime(format!("Mixture stream init failed: {err}")))?;
    let mut bits = 0.0;
    for &byte in &joint {
        bits -= mix.step(byte) / std::f64::consts::LN_2;
    }
    mix.finish_stream().map_err(|err| {
        InfotheoryError::runtime(format!("Mixture stream finalize failed: {err}"))
    })?;
    Ok(bits / (x.len() as f64))
}

#[cfg(feature = "backend-mixture")]
fn mixture_conditional_chain_rate_bits(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut mix = build_compiled_mixture_runtime(backend, -1)?;
    let total = prefix_parts
        .iter()
        .map(|part| part.len() as u64)
        .sum::<u64>()
        .saturating_add(data.len() as u64);
    mix.begin_stream(Some(total))
        .map_err(|err| InfotheoryError::runtime(format!("Mixture stream init failed: {err}")))?;
    for &part in prefix_parts {
        for &byte in part {
            mix.step(byte);
        }
    }
    let mut bits = 0.0;
    for &byte in data {
        bits -= mix.step(byte) / std::f64::consts::LN_2;
    }
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-particle")]
fn particle_stream_entropy_rate_bits(
    data: &[u8],
    spec: &crate::api::ParticleSpec,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut runtime = ParticleRuntime::new(spec);
    let mut bits = 0.0;
    for &byte in data {
        bits -= runtime.step(byte) / std::f64::consts::LN_2;
    }
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-particle")]
fn particle_conditional_chain_rate_bits(
    prefix_parts: &[&[u8]],
    data: &[u8],
    spec: &crate::api::ParticleSpec,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut runtime = ParticleRuntime::new(spec);
    for &part in prefix_parts {
        for &byte in part {
            runtime.step(byte);
        }
    }
    let mut bits = 0.0;
    for &byte in data {
        bits -= runtime.step(byte) / std::f64::consts::LN_2;
    }
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-particle")]
fn particle_joint_entropy_rate_bits(
    x: &[u8],
    y: &[u8],
    spec: &crate::api::ParticleSpec,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    let mut runtime = ParticleRuntime::new(spec);
    let mut bits = 0.0;
    for &byte in &joint {
        bits -= runtime.step(byte) / std::f64::consts::LN_2;
    }
    Ok(bits / (x.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn ctw_entropy_rate_bits(depth: usize, data: &[u8]) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut fac = FacContextTree::new(depth, 8);
    fac.reserve_for_symbols(data.len());
    for &byte in data {
        fac.update_byte_msb(byte);
    }
    let ln_p = fac.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (data.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn fac_ctw_entropy_rate_bits(
    base_depth: usize,
    encoding_bits: usize,
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let bits_per_byte = encoding_bits.clamp(1, 8);
    let mut fac = FacContextTree::new(base_depth, bits_per_byte);
    fac.reserve_for_symbols(data.len());
    for &byte in data {
        fac.update_byte_lsb(byte);
    }
    let ln_p = fac.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (data.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn ctw_joint_entropy_rate_bits(depth: usize, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    let mut fac = FacContextTree::new(depth, 16);
    for (&xb, &yb) in x.iter().zip(y.iter()) {
        for bit_idx in 0..8 {
            fac.update(((xb >> (7 - bit_idx)) & 1) == 1, bit_idx);
            fac.update(((yb >> (7 - bit_idx)) & 1) == 1, bit_idx + 8);
        }
    }
    let ln_p = fac.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (x.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn fac_ctw_joint_entropy_rate_bits(
    base_depth: usize,
    encoding_bits: usize,
    x: &[u8],
    y: &[u8],
) -> InfotheoryResult<f64> {
    let bits_per_byte = encoding_bits.clamp(1, 8);
    let mut fac = FacContextTree::new(base_depth, bits_per_byte * 2);
    for (&xb, &yb) in x.iter().zip(y.iter()) {
        for idx in 0..bits_per_byte {
            let bit_idx_x = idx * 2;
            let bit_idx_y = bit_idx_x + 1;
            fac.update(((xb >> idx) & 1) == 1, bit_idx_x);
            fac.update(((yb >> idx) & 1) == 1, bit_idx_y);
        }
    }
    let ln_p = fac.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (x.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn ctw_conditional_chain_rate_bits(
    depth: usize,
    prefix_parts: &[&[u8]],
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut tree = ContextTree::new(depth);
    for &part in prefix_parts {
        for &byte in part {
            for idx in (0..8).rev() {
                tree.update(((byte >> idx) & 1) == 1);
            }
        }
    }
    let log_p_prefix = tree.get_log_block_probability();
    for &byte in data {
        for idx in (0..8).rev() {
            tree.update(((byte >> idx) & 1) == 1);
        }
    }
    let log_p_joint = tree.get_log_block_probability();
    let bits = -(log_p_joint - log_p_prefix) / std::f64::consts::LN_2;
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn fac_ctw_conditional_chain_rate_bits(
    base_depth: usize,
    encoding_bits: usize,
    prefix_parts: &[&[u8]],
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let bits_per_byte = encoding_bits.clamp(1, 8);
    let mut fac = FacContextTree::new(base_depth, bits_per_byte);
    for &part in prefix_parts {
        for &byte in part {
            for idx in 0..bits_per_byte {
                fac.update(((byte >> idx) & 1) == 1, idx);
            }
        }
    }
    let log_p_prefix = fac.get_log_block_probability();
    for &byte in data {
        for idx in 0..bits_per_byte {
            fac.update(((byte >> idx) & 1) == 1, idx);
        }
    }
    let log_p_joint = fac.get_log_block_probability();
    let bits = -(log_p_joint - log_p_prefix) / std::f64::consts::LN_2;
    Ok(bits / (data.len() as f64))
}

fn execute_entropy_rate_backend(
    data: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    (rate_backend_kernel(backend.plan().kind()).entropy_rate)(data, max_order, backend)
}

pub(crate) fn try_entropy_rate_backend_direct(
    data: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    execute_entropy_rate_backend(data, max_order, backend)
}

pub(crate) fn try_cross_entropy_rate_backend_direct(
    test_data: &[u8],
    train_data: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if backend.plan().kind() == RateBackendKind::Zpaq {
        return (rate_backend_kernel(backend.plan().kind()).conditional_chain_rate)(
            &[train_data],
            test_data,
            backend,
        );
    }
    crate::try_frozen_plugin_rate_backend(test_data, &[train_data], max_order, backend)
}

pub(crate) fn try_joint_entropy_rate_backend_direct(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let n = x.len().min(y.len());
    let x = &x[..n];
    let y = &y[..n];

    (rate_backend_kernel(backend.plan().kind()).joint_entropy_rate)(x, y, max_order, backend)
}

pub(crate) fn try_cross_entropy_conditional_chain_backend(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    (rate_backend_kernel(backend.plan().kind()).conditional_chain_rate)(prefix_parts, data, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn assert_registry_is_injective<K>(registry: &[BackendDescriptor<K>], label: &str)
    where
        K: Copy + Eq + std::fmt::Debug + std::hash::Hash,
    {
        let mut kinds = HashSet::new();
        let mut aliases = HashSet::new();

        for descriptor in registry {
            assert!(
                kinds.insert(descriptor.kind),
                "{label} duplicates backend kind {:?}",
                descriptor.kind
            );
            assert!(
                descriptor.aliases.contains(&descriptor.canonical),
                "{label} descriptor '{}' must list its canonical alias",
                descriptor.canonical
            );
            for alias in descriptor.aliases {
                assert!(
                    aliases.insert(*alias),
                    "{label} alias '{alias}' is assigned to multiple backends"
                );
            }
        }
    }

    #[test]
    fn rate_backend_registry_resolves_aliases() {
        let rosa = find_backend_descriptor_in_registry(RATE_BACKEND_REGISTRY, "rosa")
            .expect("rosa descriptor");
        assert_eq!(rosa.canonical, "rosaplus");

        let mix = find_backend_descriptor_in_registry(RATE_BACKEND_REGISTRY, "mix")
            .expect("mixture descriptor");
        assert_eq!(mix.canonical, "mixture");
    }

    #[test]
    fn compression_registry_resolves_aliases() {
        let ac = find_backend_descriptor_in_registry(COMPRESSION_BACKEND_REGISTRY, "rate_ac")
            .expect("rate-ac descriptor");
        assert_eq!(ac.canonical, "rate-ac");
    }

    #[test]
    fn rate_backend_registry_has_unique_kinds_and_aliases() {
        assert_registry_is_injective(RATE_BACKEND_REGISTRY, "RATE_BACKEND_REGISTRY");
    }

    #[test]
    fn compression_backend_registry_has_unique_kinds_and_aliases() {
        assert_registry_is_injective(COMPRESSION_BACKEND_REGISTRY, "COMPRESSION_BACKEND_REGISTRY");
    }

    #[test]
    fn rate_backend_registry_kinds_have_runtime_kernels() {
        let kernel_kinds: HashSet<_> = RATE_BACKEND_KERNELS
            .iter()
            .map(|kernel| kernel.kind)
            .collect();
        for descriptor in RATE_BACKEND_REGISTRY {
            assert!(
                kernel_kinds.contains(&descriptor.kind),
                "missing runtime kernel for rate backend kind {:?}",
                descriptor.kind
            );
        }
    }

    #[test]
    fn compression_backend_registry_kinds_have_runtime_kernels() {
        let kernel_kinds: HashSet<_> = COMPRESSION_BACKEND_KERNELS
            .iter()
            .map(|kernel| kernel.kind)
            .collect();
        for descriptor in COMPRESSION_BACKEND_REGISTRY {
            assert!(
                kernel_kinds.contains(&descriptor.kind),
                "missing runtime kernel for compression backend kind {:?}",
                descriptor.kind
            );
        }
    }

    #[test]
    fn describe_compression_backend_uses_canonical_lookup_not_positional_indices() {
        let ac = try_describe_compression_backend(&CompressionBackend::Rate {
            rate_backend: RateBackend::RosaPlus,
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Framed,
        })
        .expect("descriptor for rate-ac");
        assert_eq!(ac.canonical, "rate-ac");

        let rans = try_describe_compression_backend(&CompressionBackend::Rate {
            rate_backend: RateBackend::RosaPlus,
            coder: crate::coders::CoderType::RANS,
            framing: crate::compression::FramingMode::Framed,
        })
        .expect("descriptor for rate-rans");
        assert_eq!(rans.canonical, "rate-rans");
    }

    #[test]
    fn missing_descriptor_reports_registry_mismatch_error() {
        let err = backend_descriptor_by_kind_checked(
            &[],
            RateBackendKind::RosaPlus,
            "RATE_BACKEND_REGISTRY",
        )
        .expect_err("missing descriptor should return an error");
        assert!(err.contains("internal backend registry mismatch"));
        assert!(err.contains("RosaPlus"));
    }
}
